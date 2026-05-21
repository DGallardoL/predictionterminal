"""Kalshi settled-markets archive.

Three public callables, all caching to the ``archive_kalshi`` namespace
(TTL=1h):

- :func:`fetch_settled_markets` — paginated list of Kalshi markets that
  have already settled, with optional date and series filters.
- :func:`fetch_archive_kalshi_detail` — full per-market detail (metadata
  + price history + summary stats) for a single settled ticker.
- :func:`kalshi_archive_series_distribution` — series-level statistics:
  for each series ticker, count of markets, average volume, and YES
  resolution rate.

External I/O is performed with :class:`httpx.AsyncClient` against
Kalshi's public ``/v2`` endpoints; the candlestick fetch reuses the
synchronous :class:`pfm.sources.kalshi.KalshiClient` because it already
encodes the rate-limit / 429-retry policy. Both clients are
constructor-injectable so tests can swap in a respx-mocked client
without touching the network.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

import httpx
import pandas as pd

from pfm.cache_utils import get_cache
from pfm.sources.kalshi import (
    DAILY_FIDELITY_SECONDS,
    KalshiClient,
    KalshiError,
    series_from_market,
)

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
ARCHIVE_CACHE_NS = "archive_kalshi"
ARCHIVE_CACHE_TTL_S = 3600

_PAGE_SIZE_MAX = 1000

# Single-flight asyncio locks keyed on (function_name, cache_key). Two
# concurrent first-callers that miss the cache would otherwise both hit
# Kalshi /markets; whichever finishes second clobbers the cache and may
# raise on a transient 5xx. Holding the lock makes the second caller wait
# for the first to populate the cache and then read it back.
#
# We lazily create the locks on first use to avoid binding the registry
# to whichever event loop happens to import this module — every loop
# uses its own dict of locks (keyed via id()).
_SINGLEFLIGHT_LOCKS: dict[tuple[int, Any], asyncio.Lock] = {}


def _singleflight_lock(key: Any) -> asyncio.Lock:
    """Return the per-key asyncio lock for the current event loop."""
    loop = asyncio.get_event_loop()
    composite = (id(loop), key)
    lock = _SINGLEFLIGHT_LOCKS.get(composite)
    if lock is None:
        lock = asyncio.Lock()
        _SINGLEFLIGHT_LOCKS[composite] = lock
    return lock


# ─────────────────────────── settled markets list ──────────────────────────


async def fetch_settled_markets(
    start_date: date | None = None,
    end_date: date | None = None,
    series_ticker: str | None = None,
    limit: int = 100,
    offset: int = 0,
    *,
    client: httpx.AsyncClient | None = None,
    base_url: str = KALSHI_BASE_URL,
) -> list[dict[str, Any]]:
    """Return Kalshi markets in ``status=settled`` matching the filters.

    Pagination is handled internally: Kalshi's ``/v2/markets`` endpoint
    uses cursor-based paging (``cursor`` query param + ``cursor`` field
    in the response). We walk pages until we've satisfied
    ``offset + limit`` matched rows or the cursor goes empty.

    Args:
        start_date: Lower bound on ``settle_time`` (inclusive). ``None``
            means no lower bound.
        end_date: Upper bound on ``settle_time`` (inclusive). ``None``
            means no upper bound.
        series_ticker: If provided, restricts the fetch to markets in
            this series (forwarded as ``series_ticker`` to Kalshi).
        limit: Max rows to return after offset is applied. Capped at
            1000 to keep memory bounded.
        offset: Number of matched rows to skip from the front of the
            stream. Useful for naive UI pagination on top of cursor
            paging.
        client: Optional shared :class:`httpx.AsyncClient`. Injected by
            tests; in production a fresh per-call client is created.
        base_url: Override the Kalshi base URL (test hook).

    Returns:
        List of dicts with keys: ``ticker``, ``title``, ``series``,
        ``settle_date``, ``settle_value`` (``"YES"`` / ``"NO"`` /
        ``None``), ``open_interest``, ``total_volume``,
        ``last_trade_price``.
    """
    limit = max(0, min(int(limit), _PAGE_SIZE_MAX))
    offset = max(0, int(offset))
    if limit == 0:
        return []

    cache_key = (
        "settled",
        start_date.isoformat() if start_date else None,
        end_date.isoformat() if end_date else None,
        series_ticker,
        limit,
        offset,
    )
    cache = get_cache(ARCHIVE_CACHE_NS, ttl=ARCHIVE_CACHE_TTL_S)
    hit = cache.get(cache_key)
    if hit is not None:
        return hit

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=15.0)
    matched: list[dict[str, Any]] = []
    try:
        cursor: str | None = None
        # Pull a generous superset since Kalshi's settle_time filter is
        # post-hoc (we trim client-side too so date bounds work whether
        # the upstream applied them or not).
        target = offset + limit
        seen_pages = 0
        max_pages = 25  # hard cap → ≤25 * 1000 = 25k markets
        while seen_pages < max_pages:
            params: dict[str, Any] = {
                "status": "settled",
                "limit": min(_PAGE_SIZE_MAX, max(50, target * 2)),
            }
            if cursor:
                params["cursor"] = cursor
            if series_ticker:
                params["series_ticker"] = series_ticker
            if start_date:
                params["min_close_ts"] = _date_to_ts(start_date)
            if end_date:
                params["max_close_ts"] = _date_to_ts(end_date, end_of_day=True)

            r = await http.get(f"{base_url}/markets", params=params)
            r.raise_for_status()
            payload = r.json()
            page = payload.get("markets") or []
            for raw in page:
                row = _normalize_settled_row(raw)
                if not _passes_filters(row, start_date, end_date, series_ticker):
                    continue
                matched.append(row)
                if len(matched) >= target:
                    break

            cursor = payload.get("cursor") or None
            seen_pages += 1
            if not cursor or len(matched) >= target:
                break
    finally:
        if owns_client:
            await http.aclose()

    out = matched[offset : offset + limit]
    cache.set(cache_key, out)
    return out


# ─────────────────────────── per-market detail ─────────────────────────────


def fetch_archive_kalshi_detail(
    ticker: str,
    *,
    kalshi_client: KalshiClient | None = None,
    http_client: httpx.Client | None = None,
    base_url: str = KALSHI_BASE_URL,
) -> dict[str, Any]:
    """Return metadata + full daily history + summary stats for one market.

    Output shape::

        {
            "ticker": str,
            "title": str,
            "series": str,
            "status": "settled" | ...,
            "settle_date": "YYYY-MM-DD" | None,
            "settle_value": "YES" | "NO" | None,
            "open_time": iso8601 | None,
            "close_time": iso8601 | None,
            "history": [{"date": "YYYY-MM-DD", "price": float,
                         "volume": float, "open_interest": float,
                         "yes_bid": float, "yes_ask": float,
                         "spread": float}, ...],
            "stats": {
                "peak_price": float,
                "trough_price": float,
                "total_volume": float,
                "n_days": int,
                "half_life_to_settle": float | None,  # days from peak
                "realized_vol": float | None,         # daily log-return σ
                "n_traders": int | None,              # if Kalshi exposes it
                "top_wallets": list[str],             # ditto, may be empty
            },
        }
    """
    cache = get_cache(ARCHIVE_CACHE_NS, ttl=ARCHIVE_CACHE_TTL_S)
    hit = cache.get(("detail", ticker))
    if hit is not None:
        return hit

    owns_http = http_client is None
    http = http_client or httpx.Client(timeout=15.0)
    owns_kalshi = kalshi_client is None
    kc = kalshi_client or KalshiClient(client=http)

    try:
        # 1. Market metadata + settled outcome
        r = http.get(f"{base_url}/markets/{ticker}")
        r.raise_for_status()
        market = r.json().get("market") or {}
        if not market:
            raise KalshiError(f"no market found for ticker={ticker!r}")

        series = str((market.get("event_ticker") or series_from_market(ticker)).split("-", 1)[0])
        settle_value = _coerce_settle_value(market.get("result"))
        settle_dt = _coerce_settle_date(market.get("settle_time") or market.get("close_time"))
        open_time = market.get("open_time")
        close_time = market.get("close_time")

        # 2. Full candlestick history (daily). Bound the window to
        # [open_time, close_time] when present, else last 2 years.
        start_ts = _iso_to_ts(open_time) or _ts_n_days_ago(730)
        end_ts = _iso_to_ts(close_time) or _ts_n_days_ago(0)
        try:
            df = kc.get_candlesticks(
                ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=DAILY_FIDELITY_SECONDS,
            )
        except KalshiError:
            df = pd.DataFrame(
                columns=["price", "volume", "open_interest", "yes_bid", "yes_ask", "spread"]
            )

        history_rows = _df_to_history_rows(df)
        stats = _compute_history_stats(df, market=market, settle_dt=settle_dt)

        result = {
            "ticker": str(market.get("ticker", ticker)),
            "title": str(market.get("title", "")),
            "series": series,
            "status": market.get("status"),
            "settle_date": settle_dt.isoformat() if settle_dt else None,
            "settle_value": settle_value,
            "open_time": open_time,
            "close_time": close_time,
            "history": history_rows,
            "stats": stats,
        }
        cache.set(("detail", ticker), result)
        return result
    finally:
        if owns_kalshi:
            kc.close()
        if owns_http:
            http.close()


# ─────────────────────────── series distribution ──────────────────────────


async def kalshi_archive_series_distribution(
    *,
    series_tickers: Iterable[str] | None = None,
    pages_per_series: int = 1,
    client: httpx.AsyncClient | None = None,
    base_url: str = KALSHI_BASE_URL,
) -> dict[str, Any]:
    """Return per-series stats over all settled markets.

    For each series, computes:
        - ``n_markets``: count of settled markets in that series
        - ``avg_volume``: mean of ``volume`` across those markets
        - ``pct_yes``: fraction whose ``result == YES``
        - ``total_volume``: aggregate volume

    If ``series_tickers`` is provided we only query those, one request
    per series. Otherwise we pull a single broad page of settled markets
    and group on ``event_ticker``.
    """
    cache = get_cache(ARCHIVE_CACHE_NS, ttl=ARCHIVE_CACHE_TTL_S)
    cache_key = ("series_distribution", tuple(series_tickers or ()), int(pages_per_series))
    hit = cache.get(cache_key)
    if hit is not None:
        return hit

    # Single-flight: serialize concurrent first-callers so we only hit
    # Kalshi /markets once per cache key. Without the lock two requests
    # arriving simultaneously each fire their own HTTP, and either can
    # propagate a transient upstream 5xx as a server 500 to the user —
    # exactly the integration-probe symptom (first call 500, retry 200).
    lock = _singleflight_lock(cache_key)
    async with lock:
        # Re-check the cache: the lock-holder before us may have just
        # populated it.
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

        owns_client = client is None
        http = client or httpx.AsyncClient(timeout=15.0)
        rows: list[dict[str, Any]] = []
        try:
            if series_tickers:
                tasks = [
                    _fetch_series_markets(http, base_url, st, pages_per_series)
                    for st in series_tickers
                ]
                chunks = await asyncio.gather(*tasks)
                for chunk in chunks:
                    rows.extend(chunk)
            else:
                # Single broad page; rely on Kalshi's default ordering.
                r = await http.get(
                    f"{base_url}/markets",
                    params={"status": "settled", "limit": _PAGE_SIZE_MAX},
                )
                r.raise_for_status()
                for raw in r.json().get("markets") or []:
                    rows.append(_normalize_settled_row(raw))
        finally:
            if owns_client:
                await http.aclose()

        by_series: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_series.setdefault(row["series"], []).append(row)

        series_stats: dict[str, dict[str, Any]] = {}
        for series, items in sorted(by_series.items()):
            n = len(items)
            vols = [float(x.get("total_volume") or 0.0) for x in items]
            yes_count = sum(1 for x in items if x.get("settle_value") == "YES")
            series_stats[series] = {
                "n_markets": n,
                "avg_volume": (sum(vols) / n) if n else 0.0,
                "total_volume": sum(vols),
                "pct_yes": (yes_count / n) if n else 0.0,
            }

        payload = {
            "series": series_stats,
            "n_total_markets": sum(s["n_markets"] for s in series_stats.values()),
            "n_series": len(series_stats),
        }
        cache.set(cache_key, payload)
        return payload


# ──────────────────────────────── helpers ──────────────────────────────────


async def _fetch_series_markets(
    http: httpx.AsyncClient,
    base_url: str,
    series_ticker: str,
    pages: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(max(1, int(pages))):
        params: dict[str, Any] = {
            "status": "settled",
            "series_ticker": series_ticker,
            "limit": _PAGE_SIZE_MAX,
        }
        if cursor:
            params["cursor"] = cursor
        r = await http.get(f"{base_url}/markets", params=params)
        r.raise_for_status()
        payload = r.json()
        for raw in payload.get("markets") or []:
            out.append(_normalize_settled_row(raw))
        cursor = payload.get("cursor") or None
        if not cursor:
            break
    return out


def _normalize_settled_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Shape a Kalshi market dict into the archive row format."""
    ticker = str(raw.get("ticker") or "")
    series = str(
        (raw.get("event_ticker") or series_from_market(ticker)).split("-", 1)[0]
        if ticker
        else (raw.get("event_ticker") or "")
    )
    settle_dt = _coerce_settle_date(raw.get("settle_time") or raw.get("close_time"))
    return {
        "ticker": ticker,
        "title": str(raw.get("title") or ""),
        "series": series,
        "settle_date": settle_dt.isoformat() if settle_dt else None,
        "settle_value": _coerce_settle_value(raw.get("result")),
        "open_interest": float(raw.get("open_interest") or 0.0),
        "total_volume": float(raw.get("volume") or 0.0),
        "last_trade_price": _coerce_price(raw.get("last_price") or raw.get("yes_ask")),
    }


def _passes_filters(
    row: Mapping[str, Any],
    start_date: date | None,
    end_date: date | None,
    series_ticker: str | None,
) -> bool:
    if series_ticker and row.get("series") != series_ticker:
        return False
    settle_str = row.get("settle_date")
    if not settle_str:
        # No settle_date → can't compare; keep only if no date filter.
        return start_date is None and end_date is None
    try:
        settle_dt = date.fromisoformat(str(settle_str))
    except ValueError:
        return start_date is None and end_date is None
    if start_date and settle_dt < start_date:
        return False
    return not (end_date and settle_dt > end_date)


def _coerce_settle_value(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in {"YES", "NO"}:
        return s
    # Kalshi sometimes encodes as 1/0 or "1"/"0"
    if s in {"1", "TRUE"}:
        return "YES"
    if s in {"0", "FALSE"}:
        return "NO"
    return None


def _coerce_settle_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    s = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        # Attempt unix-seconds
        try:
            return datetime.utcfromtimestamp(float(s)).date()
        except (TypeError, ValueError):
            return None


def _coerce_price(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # Kalshi expresses prices in cents (0–100) on /markets; downscale to
    # dollars to match the candlestick close_dollars convention.
    if v > 1.5:
        v = v / 100.0
    return round(v, 6)


def _date_to_ts(d: date, *, end_of_day: bool = False) -> int:
    """Convert a date to a unix timestamp (UTC midnight or 23:59:59)."""
    dt = datetime(d.year, d.month, d.day)
    ts = int(pd.Timestamp(dt, tz="UTC").timestamp())
    if end_of_day:
        ts += 86399
    return ts


def _iso_to_ts(value: Any) -> int | None:
    if not value:
        return None
    s = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return int(pd.Timestamp(dt, tz="UTC").timestamp())
    return int(pd.Timestamp(dt).tz_convert("UTC").timestamp())


def _ts_n_days_ago(n: int) -> int:
    return int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(n))).timestamp())


def _df_to_history_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        out.append(
            {
                "date": ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts),
                "price": float(row.get("price", 0.0) or 0.0),
                "volume": float(row.get("volume", 0.0) or 0.0),
                "open_interest": float(row.get("open_interest", 0.0) or 0.0),
                "yes_bid": float(row.get("yes_bid", 0.0) or 0.0),
                "yes_ask": float(row.get("yes_ask", 0.0) or 0.0),
                "spread": float(row.get("spread", 0.0) or 0.0),
            }
        )
    return out


def _compute_history_stats(
    df: pd.DataFrame,
    *,
    market: Mapping[str, Any],
    settle_dt: date | None,
) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "peak_price": 0.0,
            "trough_price": 0.0,
            "total_volume": float(market.get("volume") or 0.0),
            "n_days": 0,
            "half_life_to_settle": None,
            "realized_vol": None,
            "n_traders": _to_int_or_none(market.get("n_traders")),
            "top_wallets": list(market.get("top_wallets") or []),
        }

    prices = df["price"].astype(float)
    peak = float(prices.max())
    trough = float(prices.min())
    total_vol = float(df["volume"].astype(float).sum())
    n_days = int(len(df))

    # Realized vol: stdev of log returns (skip if <2 obs or zero/neg prices).
    realized_vol: float | None = None
    if n_days >= 2:
        clean = prices.clip(lower=1e-6)
        log_rets = pd.Series(clean).pipe(lambda s: (s.shift(-1) / s).apply(_safe_log)).dropna()
        if len(log_rets) >= 2 and float(log_rets.std()) > 0:
            realized_vol = float(log_rets.std())

    # Half-life-to-settle: days between peak day and settle_dt.
    half_life: float | None = None
    if settle_dt is not None:
        try:
            peak_idx = prices.idxmax()
            peak_day: date = (
                peak_idx.date()
                if hasattr(peak_idx, "date")
                else date.fromisoformat(str(peak_idx)[:10])
            )
            half_life = float((settle_dt - peak_day).days)
        except (AttributeError, ValueError, TypeError):
            half_life = None

    return {
        "peak_price": round(peak, 6),
        "trough_price": round(trough, 6),
        "total_volume": round(total_vol, 6),
        "n_days": n_days,
        "half_life_to_settle": half_life,
        "realized_vol": round(realized_vol, 6) if realized_vol is not None else None,
        "n_traders": _to_int_or_none(market.get("n_traders")),
        "top_wallets": list(market.get("top_wallets") or []),
    }


def _safe_log(x: float) -> float:
    try:
        return math.log(float(x)) if x and x > 0 else math.nan
    except (TypeError, ValueError):
        return math.nan


def _to_int_or_none(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "ARCHIVE_CACHE_NS",
    "ARCHIVE_CACHE_TTL_S",
    "KALSHI_BASE_URL",
    "fetch_archive_kalshi_detail",
    "fetch_settled_markets",
    "kalshi_archive_series_distribution",
]
