"""Polymarket archive: discover + describe resolved markets.

Three top-level entry points map 1:1 onto the router:

    fetch_resolved_markets(start, end, theme, limit, offset) -> list[dict]
    fetch_archive_market_detail(slug)                        -> dict
    archive_themes_distribution()                            -> dict

All upstream IO goes through Polymarket Gamma (markets metadata) and CLOB
(daily price-history with ``fidelity=1440``). Results are cached for one
hour in a ``pfm.cache_utils`` namespace because resolved markets are by
definition immutable — the only churn is that *new* markets become
resolved, which we let the cache TTL absorb naturally.

Theme classification reuses the same ``theme / category / tags`` heuristic
used by :mod:`pfm.terminal_homepage` so the archive is consistent with the
live tape.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx
import numpy as np
import pandas as pd

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)


# --- constants ---------------------------------------------------------------

GAMMA_URL: str = "https://gamma-api.polymarket.com"
CLOB_URL: str = "https://clob.polymarket.com"
DAILY_FIDELITY: int = 1440  # minutes per bucket — see ADR-0006
ARCHIVE_CACHE_TTL: int = 3600  # 1 hour; resolved markets are immutable

# Page size for Gamma — Gamma caps at ~500, we keep it conservative.
_GAMMA_PAGE_SIZE: int = 200
_THEMES_DISCOVER_PAGES: int = 5  # ~1000 markets sampled for the theme stats


# --- helpers -----------------------------------------------------------------


def _safe_float(x: Any) -> float | None:
    """Coerce ``x`` to a finite float or ``None``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _parse_iso(ts: Any) -> datetime | None:
    """Parse an ISO-8601 string (with or without ``Z``) to UTC ``datetime``."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _theme_for_market(m: dict[str, Any]) -> str:
    """Theme heuristic: explicit ``theme``/``category`` field, else first tag.

    Same convention as :func:`pfm.terminal_homepage._theme_for_market` so
    archive and live tape report identical theme labels.
    """
    explicit = m.get("theme") or m.get("category")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    tags = m.get("tags")
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, str) and first.strip():
            return first.strip().lower()
        if isinstance(first, dict):
            label = first.get("label") or first.get("slug") or first.get("name")
            if isinstance(label, str) and label.strip():
                return label.strip().lower()
    return "uncategorized"


def _resolution_label(m: dict[str, Any]) -> str:
    """Map a Gamma market dict to ``YES`` / ``NO`` / ``AMBIGUOUS`` / ``PENDING``.

    We treat ``umaResolutionStatuses == "disputed"`` (or any explicit
    ``ambiguous`` field) as ``AMBIGUOUS``. Otherwise look at ``outcomePrices``
    (a JSON-encoded list ``[YES, NO]``) — if YES > 0.5 the YES side won,
    if NO > 0.5 NO won. Anything we can't classify but that *is* closed gets
    ``PENDING`` (rare — usually disputed/withdrawn).
    """
    if not bool(m.get("closed", False)):
        return "PENDING"
    # Explicit ambiguous flags (Gamma sets these when UMA disputes a result).
    if m.get("ambiguous") is True:
        return "AMBIGUOUS"
    statuses = m.get("umaResolutionStatuses")
    if isinstance(statuses, str) and "dispute" in statuses.lower():
        return "AMBIGUOUS"
    raw = m.get("outcomePrices")
    arr: list[Any] | None = None
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            arr = None
    elif isinstance(raw, list):
        arr = raw
    if arr is not None and len(arr) >= 2:
        yes = _safe_float(arr[0])
        no = _safe_float(arr[1])
        if yes is not None and yes >= 0.99:
            return "YES"
        if no is not None and no >= 0.99:
            return "NO"
        if yes is not None and no is not None:
            return "YES" if yes > no else "NO"
    return "PENDING"


def _final_price(m: dict[str, Any]) -> float | None:
    """Best-effort YES-side final price from ``outcomePrices`` or last trade."""
    raw = m.get("outcomePrices")
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and arr:
                return _safe_float(arr[0])
        except (TypeError, json.JSONDecodeError):
            pass
    elif isinstance(raw, list) and raw:
        return _safe_float(raw[0])
    return _safe_float(m.get("lastTradePrice"))


def _yes_token_id(m: dict[str, Any]) -> str | None:
    """Pull the YES token id out of ``clobTokenIds`` (a JSON string list)."""
    raw = m.get("clobTokenIds")
    if not raw:
        return None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if isinstance(ids, list) and ids:
        return str(ids[0])
    return None


# --- light-weight summary used by /markets list ------------------------------


def _summary_row(m: dict[str, Any]) -> dict[str, Any]:
    """Compact, list-view-friendly view of one resolved market."""
    end = (m.get("endDate") or "")[:10] or None
    return {
        "id": str(m.get("id") or m.get("conditionId") or ""),
        "slug": m.get("slug"),
        "question": m.get("question") or "",
        "theme": _theme_for_market(m),
        "end_date": end,
        "resolution": _resolution_label(m),
        "final_price": _final_price(m),
        "total_volume": _safe_float(m.get("volume") or m.get("volumeNum")),
        "total_traders": _safe_float(m.get("traders") or m.get("uniqueTraders")),
    }


# --- Gamma fetch helpers (sync httpx) ---------------------------------------


def _gamma_fetch_resolved_page(
    client: httpx.Client,
    *,
    start: date,
    end: date,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    """Pull one page of closed markets within ``[start, end]`` end-date range.

    Gamma supports ``date_end_min`` / ``date_end_max`` filters on the
    ``endDate`` field (ISO-8601). We always pin ``closed=true``.
    """
    params: dict[str, str | int] = {
        "closed": "true",
        "limit": int(limit),
        "offset": int(offset),
        "order": "endDate",
        "ascending": "false",
        "date_end_min": start.isoformat(),
        "date_end_max": end.isoformat(),
    }
    r = client.get(f"{GAMMA_URL}/markets", params=params, timeout=15.0)
    r.raise_for_status()
    payload = r.json() or []
    return list(payload) if isinstance(payload, list) else []


def _gamma_fetch_market(client: httpx.Client, slug: str) -> dict[str, Any] | None:
    """Single market by slug, falling back to ``closed=true`` for resolved markets."""
    base = GAMMA_URL.rstrip("/")
    r = client.get(f"{base}/markets", params={"slug": slug}, timeout=10.0)
    if r.status_code == 200:
        arr = r.json() or []
        if isinstance(arr, list) and arr:
            return arr[0]
    r2 = client.get(
        f"{base}/markets",
        params={"slug": slug, "closed": "true"},
        timeout=10.0,
    )
    if r2.status_code == 200:
        arr2 = r2.json() or []
        if isinstance(arr2, list) and arr2:
            return arr2[0]
    return None


def _clob_fetch_history(client: httpx.Client, token_id: str) -> list[dict[str, Any]]:
    """Daily history for a YES token. Returns ``[{t, p}]`` (unix seconds, price)."""
    params: dict[str, str | int] = {
        "market": token_id,
        "fidelity": DAILY_FIDELITY,
        "interval": "max",
    }
    r = client.get(f"{CLOB_URL}/prices-history", params=params, timeout=15.0)
    r.raise_for_status()
    raw = r.json().get("history", []) or []
    return [{"t": int(b["t"]), "p": float(b["p"])} for b in raw if "t" in b and "p" in b]


# --- statistics --------------------------------------------------------------


@dataclass(frozen=True)
class _ArchiveStats:
    peak_price: float | None
    peak_date: str | None
    trough_price: float | None
    trough_date: str | None
    max_volume_day: str | None
    total_volume: float | None
    half_life_to_resolution: int | None
    volatility_realized: float | None
    hurst_exponent: float | None
    dfa_alpha: float | None
    n_unique_traders: int | None
    whale_concentration: float | None


def _hurst(series: pd.Series) -> float | None:
    """Quick R/S Hurst on first differences. Returns ``None`` if N < 30."""
    s = series.dropna()
    if len(s) < 30:
        return None
    diffs = np.diff(s.to_numpy())
    if diffs.size < 20:
        return None
    log_n: list[float] = []
    log_rs: list[float] = []
    for n in [10, 14, 20, 28, 40, 56, 80]:
        if n > diffs.size:
            break
        n_subs = diffs.size // n
        if n_subs < 2:
            continue
        rs_vals: list[float] = []
        for i in range(n_subs):
            sub = diffs[i * n : (i + 1) * n]
            mean = sub.mean()
            cum = np.cumsum(sub - mean)
            R = float(cum.max() - cum.min())
            S = float(np.std(sub, ddof=1))
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            log_n.append(math.log(n))
            log_rs.append(math.log(float(np.mean(rs_vals))))
    if len(log_n) < 3:
        return None
    a = np.asarray(log_n)
    b = np.asarray(log_rs)
    slope = float(np.polyfit(a, b, 1)[0])
    return slope if math.isfinite(slope) else None


def _dfa_alpha(series: pd.Series) -> float | None:
    """Light DFA — only used so the archive panel can show a single number."""
    try:
        from pfm.dfa import dfa as _dfa_run
    except ImportError:
        return None
    s = series.dropna()
    if len(s) < 40:
        return None
    res = _dfa_run(s)
    return res.alpha if math.isfinite(res.alpha) else None


def _half_life_to_resolution(history: pd.DataFrame, resolution: str) -> int | None:
    """Days from the first crossing of 0.5 to the first crossing of 0.9 (YES win).

    ``None`` if resolution wasn't ``YES`` or if the series never crossed 0.9.
    Mirrors how Polymarket's own analytics show "decay-to-resolution" speed.
    """
    if resolution != "YES" or history.empty:
        return None
    after_half = history[history["price"] >= 0.5]
    if after_half.empty:
        return None
    after_high = history[history["price"] >= 0.9]
    if after_high.empty:
        return None
    t0 = pd.Timestamp(after_half["date"].iloc[0])
    t1 = pd.Timestamp(after_high["date"].iloc[0])
    days = int((t1 - t0).total_seconds() // 86400)
    return days if days >= 0 else None


def _compute_stats(
    history: pd.DataFrame,
    market: dict[str, Any],
    resolution: str,
) -> _ArchiveStats:
    """All derived stats in one place. Tolerant of empty / sparse history."""
    if history.empty or "price" not in history.columns:
        return _ArchiveStats(
            peak_price=None,
            peak_date=None,
            trough_price=None,
            trough_date=None,
            max_volume_day=None,
            total_volume=_safe_float(market.get("volume") or market.get("volumeNum")),
            half_life_to_resolution=None,
            volatility_realized=None,
            hurst_exponent=None,
            dfa_alpha=None,
            n_unique_traders=int(market.get("traders") or market.get("uniqueTraders") or 0) or None,
            whale_concentration=None,
        )

    px = history["price"].astype(float)
    peak_idx = int(px.idxmax())
    trough_idx = int(px.idxmin())
    peak_price = float(px.loc[peak_idx])
    trough_price = float(px.loc[trough_idx])
    peak_date = str(history["date"].loc[peak_idx])[:10]
    trough_date = str(history["date"].loc[trough_idx])[:10]

    # Realized vol on log returns (annualised w/ √365 since markets trade 24/7).
    if len(px) >= 3:
        log_ret = np.log(px.clip(lower=1e-4) / px.clip(lower=1e-4).shift(1)).dropna()
        rv = float(log_ret.std(ddof=1) * math.sqrt(365)) if len(log_ret) > 1 else None
    else:
        rv = None

    max_vol_day: str | None = None
    if "volume" in history.columns and not history["volume"].isna().all():
        idx = int(history["volume"].astype(float).idxmax())
        max_vol_day = str(history["date"].loc[idx])[:10]

    total_volume = _safe_float(market.get("volume") or market.get("volumeNum"))
    n_traders = market.get("traders") or market.get("uniqueTraders")
    try:
        n_traders_int = int(n_traders) if n_traders is not None else None
    except (TypeError, ValueError):
        n_traders_int = None

    whale_conc = _safe_float(market.get("whaleConcentration"))
    if whale_conc is None:
        # Fall back to a Gamma-side aggregate if the market reports top-wallet share.
        top_wallets = market.get("topWalletsShare") or market.get("topTradersShare")
        whale_conc = _safe_float(top_wallets)

    return _ArchiveStats(
        peak_price=peak_price,
        peak_date=peak_date,
        trough_price=trough_price,
        trough_date=trough_date,
        max_volume_day=max_vol_day,
        total_volume=total_volume,
        half_life_to_resolution=_half_life_to_resolution(history, resolution),
        volatility_realized=rv,
        hurst_exponent=_hurst(px),
        dfa_alpha=_dfa_alpha(px),
        n_unique_traders=n_traders_int,
        whale_concentration=whale_conc,
    )


# --- public API --------------------------------------------------------------


def fetch_resolved_markets(
    start_date: date,
    end_date: date,
    theme: str | None = None,
    limit: int = 100,
    offset: int = 0,
    *,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Return a paginated list of resolved Polymarket markets.

    Args:
        start_date: lower bound on ``endDate`` (inclusive).
        end_date: upper bound on ``endDate`` (inclusive).
        theme: optional case-insensitive theme filter (``politics``, ``crypto``, …).
        limit: page size (Gamma is capped to ~500 — we don't enforce that here).
        offset: page offset.
        client: pre-built ``httpx.Client``; mostly for tests with respx.

    Theme filtering is applied client-side after the Gamma fetch since Gamma
    doesn't expose a stable theme parameter. To compensate, when ``theme`` is
    set we widen the page server-side (request 4× ``limit``) and trim to
    ``limit`` after filtering. Cache key includes the filter so two
    different themes don't collide.
    """
    cache = get_cache("archive_polymarket", ttl=ARCHIVE_CACHE_TTL)
    cache_key = (
        "markets",
        start_date.isoformat(),
        end_date.isoformat(),
        (theme or "").lower(),
        int(limit),
        int(offset),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    owns = client is None
    http = client or httpx.Client(timeout=15.0)
    try:
        if theme:
            # Need to oversample so the post-filter list still fills a page.
            page = _gamma_fetch_resolved_page(
                http,
                start=start_date,
                end=end_date,
                limit=max(int(limit) * 4, _GAMMA_PAGE_SIZE),
                offset=int(offset),
            )
            theme_lc = theme.lower()
            filtered = [m for m in page if _theme_for_market(m) == theme_lc]
            page = filtered[: int(limit)]
        else:
            page = _gamma_fetch_resolved_page(
                http,
                start=start_date,
                end=end_date,
                limit=int(limit),
                offset=int(offset),
            )
        rows = [_summary_row(m) for m in page if m.get("slug")]
    finally:
        if owns:
            http.close()

    cache.set(cache_key, rows)
    return rows


def fetch_archive_market_detail(
    slug: str,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Return market metadata + full daily history + computed stats.

    Raises:
        LookupError: if no Gamma market matches the slug.
    """
    cache = get_cache("archive_polymarket", ttl=ARCHIVE_CACHE_TTL)
    cache_key = ("detail", slug)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    owns = client is None
    http = client or httpx.Client(timeout=15.0)
    try:
        market = _gamma_fetch_market(http, slug)
        if market is None:
            raise LookupError(f"no archive market for slug={slug!r}")

        token_id = _yes_token_id(market)
        history_rows: list[dict[str, Any]] = []
        if token_id:
            try:
                history_rows = _clob_fetch_history(http, token_id)
            except httpx.HTTPError as exc:
                logger.warning("archive: clob history failed for %s: %s", slug, exc)
                history_rows = []

        if history_rows:
            df = pd.DataFrame(history_rows)
            df["date"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.normalize()
            df = df.rename(columns={"p": "price"})
            df = df.groupby("date", as_index=False).last()
            df = df.sort_values("date").reset_index(drop=True)
            # Volume column is optional — Gamma history doesn't surface it; we
            # leave it blank so the schema still validates.
            if "volume" not in df.columns:
                df["volume"] = float("nan")
        else:
            df = pd.DataFrame(columns=["date", "price", "volume"])

        resolution = _resolution_label(market)
        stats = _compute_stats(df, market, resolution)

        # Serialize history as list of [date_iso, price, volume].
        if df.empty:
            history_out: list[list[Any]] = []
        else:
            history_out = [
                [str(d)[:10], float(p), (float(v) if not pd.isna(v) else None)]
                for d, p, v in zip(df["date"], df["price"], df["volume"], strict=True)
            ]

        # Top news around resolution: pulled from Gamma's optional ``relatedNews``
        # if present, else empty. We avoid the live news router here so the
        # archive is a single-source-of-truth IO module.
        related = market.get("relatedNews") or []
        news_out: list[dict[str, Any]] = []
        if isinstance(related, list):
            for n in related[:5]:
                if isinstance(n, dict):
                    news_out.append(
                        {
                            "title": n.get("title") or "",
                            "url": n.get("url") or "",
                            "ts": n.get("publishedAt") or n.get("ts") or "",
                        }
                    )

        result: dict[str, Any] = {
            "slug": slug,
            "question": market.get("question") or "",
            "theme": _theme_for_market(market),
            "end_date": (market.get("endDate") or "")[:10] or None,
            "resolution": resolution,
            "final_price": _final_price(market),
            "history": history_out,
            "stats": {
                "peak_price": stats.peak_price,
                "peak_date": stats.peak_date,
                "trough_price": stats.trough_price,
                "trough_date": stats.trough_date,
                "max_volume_day": stats.max_volume_day,
                "total_volume": stats.total_volume,
                "half_life_to_resolution": stats.half_life_to_resolution,
                "volatility_realized": stats.volatility_realized,
                "hurst_exponent": stats.hurst_exponent,
                "dfa_alpha": stats.dfa_alpha,
                "n_unique_traders": stats.n_unique_traders,
                "whale_concentration": stats.whale_concentration,
            },
            "top_news_around_resolution": news_out,
        }
    finally:
        if owns:
            http.close()

    cache.set(cache_key, result)
    return result


def archive_themes_distribution(
    *,
    pages: int = _THEMES_DISCOVER_PAGES,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Aggregate stats across the most recent ``pages * _GAMMA_PAGE_SIZE`` resolved markets.

    Returns ``{themes: [{theme, n_markets, pct_yes, pct_no, pct_ambiguous,
    avg_duration_days, avg_volume}], n_markets_total}``.
    """
    cache = get_cache("archive_polymarket", ttl=ARCHIVE_CACHE_TTL)
    cache_key = ("themes", int(pages))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    owns = client is None
    http = client or httpx.Client(timeout=15.0)
    rows: list[dict[str, Any]] = []
    try:
        for i in range(int(pages)):
            params: dict[str, str | int] = {
                "closed": "true",
                "limit": _GAMMA_PAGE_SIZE,
                "offset": i * _GAMMA_PAGE_SIZE,
                "order": "endDate",
                "ascending": "false",
            }
            r = http.get(f"{GAMMA_URL}/markets", params=params, timeout=15.0)
            r.raise_for_status()
            payload = r.json() or []
            if not isinstance(payload, list) or not payload:
                break
            rows.extend(payload)
    finally:
        if owns:
            http.close()

    by_theme: dict[str, dict[str, Any]] = {}
    for m in rows:
        theme = _theme_for_market(m)
        bucket = by_theme.setdefault(
            theme,
            {
                "theme": theme,
                "n_markets": 0,
                "n_yes": 0,
                "n_no": 0,
                "n_ambiguous": 0,
                "duration_sum": 0.0,
                "duration_count": 0,
                "volume_sum": 0.0,
                "volume_count": 0,
            },
        )
        bucket["n_markets"] += 1
        res = _resolution_label(m)
        if res == "YES":
            bucket["n_yes"] += 1
        elif res == "NO":
            bucket["n_no"] += 1
        elif res == "AMBIGUOUS":
            bucket["n_ambiguous"] += 1

        start_dt = _parse_iso(m.get("startDate"))
        end_dt = _parse_iso(m.get("endDate"))
        if start_dt and end_dt and end_dt > start_dt:
            bucket["duration_sum"] += (end_dt - start_dt).total_seconds() / 86400.0
            bucket["duration_count"] += 1
        vol = _safe_float(m.get("volume") or m.get("volumeNum"))
        if vol is not None:
            bucket["volume_sum"] += vol
            bucket["volume_count"] += 1

    themes_out: list[dict[str, Any]] = []
    for theme, b in sorted(by_theme.items(), key=lambda kv: -kv[1]["n_markets"]):
        n = b["n_markets"]
        themes_out.append(
            {
                "theme": theme,
                "n_markets": n,
                "pct_yes": round(b["n_yes"] / n, 4) if n else 0.0,
                "pct_no": round(b["n_no"] / n, 4) if n else 0.0,
                "pct_ambiguous": round(b["n_ambiguous"] / n, 4) if n else 0.0,
                "avg_duration_days": (
                    round(b["duration_sum"] / b["duration_count"], 2)
                    if b["duration_count"]
                    else None
                ),
                "avg_volume": (
                    round(b["volume_sum"] / b["volume_count"], 2) if b["volume_count"] else None
                ),
            }
        )

    result = {"themes": themes_out, "n_markets_total": len(rows)}
    cache.set(cache_key, result)
    return result


def search_archive(
    query: str,
    *,
    limit: int = 25,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Substring-search resolved markets by slug or question.

    Trades exhaustiveness for simplicity: we walk a few pages of the most
    recent closed markets and filter client-side. Good enough for a demo
    archive; not a replacement for a real search index.
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    owns = client is None
    http = client or httpx.Client(timeout=15.0)
    out: list[dict[str, Any]] = []
    try:
        for i in range(_THEMES_DISCOVER_PAGES):
            params: dict[str, str | int] = {
                "closed": "true",
                "limit": _GAMMA_PAGE_SIZE,
                "offset": i * _GAMMA_PAGE_SIZE,
                "order": "endDate",
                "ascending": "false",
            }
            r = http.get(f"{GAMMA_URL}/markets", params=params, timeout=15.0)
            r.raise_for_status()
            page = r.json() or []
            if not isinstance(page, list) or not page:
                break
            for m in page:
                slug = (m.get("slug") or "").lower()
                question = (m.get("question") or "").lower()
                if q in slug or q in question:
                    out.append(_summary_row(m))
                    if len(out) >= int(limit):
                        return out
    finally:
        if owns:
            http.close()
    return out


__all__ = [
    "CLOB_URL",
    "GAMMA_URL",
    "archive_themes_distribution",
    "fetch_archive_market_detail",
    "fetch_resolved_markets",
    "search_archive",
]
