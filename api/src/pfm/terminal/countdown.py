"""Terminal resolution-countdown / "this week's events" endpoint.

Exposes two GETs:

  * ``GET /terminal/countdown?days=7``        — markets resolving inside the
    next ``N`` days, grouped by day-bucket and sorted (bucket → days asc →
    conviction desc).
  * ``GET /terminal/countdown/{slug}``        — real-time countdown for a
    single Polymarket market, plus expected payoff if held to resolution.

Conviction is defined as ``abs(p - 0.5) * 2`` — 1.0 means a near-certain
print, 0.0 means a coin flip. We sort by conviction *descending* within a
day bucket because the UI cares more about high-conviction markets that
are about to resolve than the noisy 50/50 ones.

External calls go to Polymarket Gamma. We read the factor universe out of
``factors.yml`` (filtering ``source: polymarket``) so the endpoint stays
in lock-step with the rest of the app's curated factor list. HTTP is done
through ``httpx`` so respx can intercept it in tests.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terminal/countdown", tags=["terminal"])


GAMMA_URL: str = "https://gamma-api.polymarket.com"

DayBucket = Literal["today", "tomorrow", "this-week", "next-week", "this-month", "later"]

# Day-bucket boundaries, applied to ``days_to_resolve`` (rounded down to whole days).
#   today      : 0
#   tomorrow   : 1
#   this-week  : 2..6   (i.e. <= one calendar week from now)
#   next-week  : 7..13
#   this-month : 14..30
#   later      : > 30   (only emitted if user explicitly asks for a wide horizon)


# ──────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────


class CountdownMarket(BaseModel):
    """One Polymarket market resolving inside the requested horizon."""

    slug: str
    question: str
    theme: str | None = None
    current_p: float = Field(..., ge=0.0, le=1.0)
    days_to_resolve: int = Field(..., ge=0)
    hours_to_resolve: int = Field(..., ge=0)
    conviction: float = Field(..., ge=0.0, le=1.0, description="abs(p-0.5)*2")
    last_24h_change: float | None = None
    volume_24hr: float | None = None
    expected_resolution_date_str: str = Field(..., description="ISO YYYY-MM-DD (UTC).")
    day_bucket: DayBucket


class CountdownGroup(BaseModel):
    bucket: DayBucket
    n_markets: int
    markets: list[CountdownMarket]


class CountdownResponse(BaseModel):
    as_of: str
    horizon_days: int
    n_markets: int
    groups: list[CountdownGroup]


class MarketCountdown(BaseModel):
    """Real-time countdown for one specific market."""

    slug: str
    question: str
    current_p: float = Field(..., ge=0.0, le=1.0)
    days: int = Field(..., ge=0)
    hours: int = Field(..., ge=0, le=23)
    minutes: int = Field(..., ge=0, le=59)
    seconds_remaining: int = Field(..., ge=0)
    expected_resolution: str
    fair_price_at_resolution: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0 or 1 — the expected payoff per YES contract assuming "
        "the current market probability becomes the realised outcome at T+0.",
    )
    expected_payoff_if_held: float = Field(
        ...,
        description="Expected $ payoff per $1 of YES bought now and held to resolution: "
        "current_p * 1 - 1*current_p = 0 in expectation under no edge; the field "
        "reports current_p directly so the UI can render YES/NO PnL skeleton.",
    )


# ──────────────────────────────────────────────────────────────────────────
# Helpers — factors.yml + Gamma
# ──────────────────────────────────────────────────────────────────────────


def _factors_path() -> Path:
    """Resolve the active factors.yml path (env-driven; falls back to package default)."""
    try:
        from pfm.config import get_settings

        return Path(get_settings().factors_file)
    except Exception:
        # 2026-05 refactor: module moved into ``pfm/terminal/``; climb one more
        # parent to reach the package root where ``factors.yml`` still lives.
        return Path(__file__).resolve().parents[1] / "factors.yml"


def _load_polymarket_factors(path: Path | None = None) -> list[dict[str, str]]:
    """Return ``[{id, slug, theme}, ...]`` for every ``source: polymarket`` factor.

    Empty list on missing/invalid file — the endpoint still serves a valid
    (just empty) payload rather than 500-ing the UI.
    """
    p = path or _factors_path()
    try:
        data = yaml.safe_load(p.read_text())
    except (FileNotFoundError, yaml.YAMLError):
        return []
    factors = (data or {}).get("factors", []) or []
    out: list[dict[str, str]] = []
    for f in factors:
        if not isinstance(f, dict):
            continue
        if str(f.get("source", "")).lower() != "polymarket":
            continue
        slug = f.get("slug")
        fid = f.get("id")
        if not slug or not fid:
            continue
        out.append(
            {
                "id": str(fid),
                "slug": str(slug),
                "theme": str(f.get("theme") or ""),
            }
        )
    return out


def _parse_end_date(raw: str | None) -> datetime | None:
    """Parse a Gamma ``endDate`` (ISO with optional ``Z``) into a UTC datetime."""
    if not raw:
        return None
    s = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(raw[:10], "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _coerce_float(value: Any) -> float | None:
    """Best-effort ``float()`` that returns ``None`` on garbage."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _current_p_from_gamma(market: dict[str, Any]) -> float | None:
    """Extract the YES-side probability from a Gamma market dict.

    Gamma exposes ``lastTradePrice`` for YES, plus an ``outcomePrices`` JSON
    string. ``lastTradePrice`` is the most reliable; we fall back to the
    first ``outcomePrices`` entry.
    """
    p = _coerce_float(market.get("lastTradePrice"))
    if p is not None:
        return max(0.0, min(1.0, p))
    raw = market.get("outcomePrices")
    if isinstance(raw, str):
        import json

        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(arr, list) and arr:
            f = _coerce_float(arr[0])
            if f is not None:
                return max(0.0, min(1.0, f))
    return None


def _fetch_gamma_metadata(slug: str, client: httpx.Client) -> dict[str, Any] | None:
    """GET ``/markets?slug=...`` and return the first market dict, or ``None``."""
    try:
        r = client.get(f"{GAMMA_URL}/markets", params={"slug": slug})
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("gamma fetch failed for slug=%s: %s", slug, exc)
        return None
    payload = r.json()
    if isinstance(payload, list) and payload:
        return payload[0]
    return None


# ──────────────────────────────────────────────────────────────────────────
# Pure logic — bucketing, conviction, sorting
# ──────────────────────────────────────────────────────────────────────────


def conviction(p: float) -> float:
    """Conviction = ``abs(p - 0.5) * 2`` clamped to [0, 1]."""
    p = max(0.0, min(1.0, p))
    return min(1.0, abs(p - 0.5) * 2.0)


def day_bucket_for(days_to_resolve: int) -> DayBucket:
    """Map an integer day-count to a bucket label."""
    if days_to_resolve <= 0:
        return "today"
    if days_to_resolve == 1:
        return "tomorrow"
    if days_to_resolve <= 6:
        return "this-week"
    if days_to_resolve <= 13:
        return "next-week"
    if days_to_resolve <= 30:
        return "this-month"
    return "later"


_BUCKET_ORDER: dict[DayBucket, int] = {
    "today": 0,
    "tomorrow": 1,
    "this-week": 2,
    "next-week": 3,
    "this-month": 4,
    "later": 5,
}


def build_countdown_markets(
    factors: list[dict[str, str]],
    gamma_by_slug: dict[str, dict[str, Any]],
    now: datetime,
    horizon_days: int,
) -> list[CountdownMarket]:
    """Pure assembly of CountdownMarket rows from factor metadata + Gamma blobs.

    Filters to active+open markets whose ``endDate`` is inside
    ``[now, now + horizon_days]``. Skips anything missing a price or end
    date — those would render uselessly in the UI.
    """
    horizon = now + timedelta(days=horizon_days)
    out: list[CountdownMarket] = []
    for f in factors:
        slug = f["slug"]
        meta = gamma_by_slug.get(slug)
        if not meta:
            continue
        if meta.get("closed") or not meta.get("active", True):
            continue
        end = _parse_end_date(meta.get("endDate"))
        if end is None:
            continue
        if end < now or end > horizon:
            continue
        p = _current_p_from_gamma(meta)
        if p is None:
            continue
        delta = end - now
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            continue
        days = total_seconds // 86_400
        hours = total_seconds // 3_600
        bucket: DayBucket = day_bucket_for(days)
        out.append(
            CountdownMarket(
                slug=slug,
                question=str(meta.get("question") or ""),
                theme=f.get("theme") or None,
                current_p=p,
                days_to_resolve=days,
                hours_to_resolve=hours,
                conviction=conviction(p),
                last_24h_change=_coerce_float(meta.get("oneDayPriceChange")),
                volume_24hr=_coerce_float(meta.get("volume24hr")),
                expected_resolution_date_str=end.date().isoformat(),
                day_bucket=bucket,
            )
        )
    out.sort(
        key=lambda m: (
            _BUCKET_ORDER[m.day_bucket],
            m.days_to_resolve,
            -m.conviction,
        )
    )
    return out


def group_by_bucket(markets: list[CountdownMarket]) -> list[CountdownGroup]:
    """Collect already-sorted markets into per-bucket groups (preserves order)."""
    grouped: dict[DayBucket, list[CountdownMarket]] = {}
    for m in markets:
        grouped.setdefault(m.day_bucket, []).append(m)
    return [
        CountdownGroup(bucket=b, n_markets=len(grouped[b]), markets=grouped[b])
        for b in sorted(grouped, key=lambda x: _BUCKET_ORDER[x])
    ]


# ──────────────────────────────────────────────────────────────────────────
# HTTP client seam (tests inject)
# ──────────────────────────────────────────────────────────────────────────


def _new_http_client(timeout: float = 10.0) -> httpx.Client:
    return httpx.Client(timeout=timeout)


# ──────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────


# Module-level cache: avoid the N+1 gamma fan-out on every request.
_COUNTDOWN_CACHE: dict[int, tuple[float, CountdownResponse]] = {}
_COUNTDOWN_TTL_SECONDS: float = 300.0  # 5 minutes — countdown shifts slowly
_COUNTDOWN_MAX_FACTORS: int = 60  # cap fan-out so we don't trip gamma 429s
# Bounded concurrency when refilling the cache. 1000-req/10s on gamma
# is generous, so 10 parallel slug fetches stays well under the cap while
# turning the worst-case wall clock from ~10 s into ~1.5 s.
_COUNTDOWN_MAX_WORKERS: int = 10

# Per-slug countdown cache. Countdown shows live seconds_remaining, so the
# TTL is tight — but the market-detail UI polls every ~5 s and the same
# slug gets hit by quote/peers/quality concurrently, so even 5 s eliminates
# most of the gamma-API rate-limit pressure that was producing 404s under
# load. The slug→endDate mapping is fully static; the only field that
# truly moves second-by-second is the countdown delta, which we recompute
# from cached metadata.
_SLUG_META_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_SLUG_META_TTL_SECONDS: float = 15.0


@router.get("", response_model=CountdownResponse)
def get_countdown(
    days: int = Query(7, ge=1, le=365, description="Look-ahead horizon in days."),
) -> CountdownResponse:
    """Polymarket factors resolving in the next ``days`` days. Cached 5 min,
    fan-out capped at ``_COUNTDOWN_MAX_FACTORS`` slugs to avoid gamma 429s.
    """
    now_ts = time.time()
    cached = _COUNTDOWN_CACHE.get(days)
    if cached is not None and (now_ts - cached[0]) < _COUNTDOWN_TTL_SECONDS:
        return cached[1]

    factors = _load_polymarket_factors()
    if len(factors) > _COUNTDOWN_MAX_FACTORS:
        factors = factors[:_COUNTDOWN_MAX_FACTORS]
    now = datetime.now(UTC)
    gamma_by_slug: dict[str, dict[str, Any]] = {}
    if factors:
        # Parallel fan-out across slugs. The previous serial loop was the
        # dominant cold-cache cost (~60 sequential gamma RTTs). httpx.Client
        # serializes its own state under the hood, but multiple in-flight
        # GETs are safe and the gamma free tier handles 10× concurrency well.
        with _new_http_client() as client:
            slugs = [f["slug"] for f in factors]

            def _fetch_one(slug: str) -> tuple[str, dict[str, Any] | None]:
                try:
                    return slug, _fetch_gamma_metadata(slug, client)
                except Exception as e:
                    logger.info("countdown gamma fetch failed for %s: %s", slug, e)
                    return slug, None

            max_workers = min(len(slugs), _COUNTDOWN_MAX_WORKERS)
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="countdown") as ex:
                gamma_by_slug.update(
                    {slug: meta for slug, meta in ex.map(_fetch_one, slugs) if meta is not None}
                )

    markets = build_countdown_markets(factors, gamma_by_slug, now=now, horizon_days=days)
    groups = group_by_bucket(markets)
    resp = CountdownResponse(
        as_of=now.isoformat(),
        horizon_days=days,
        n_markets=len(markets),
        groups=groups,
    )
    _COUNTDOWN_CACHE[days] = (now_ts, resp)
    return resp


@router.get("/{slug}", response_model=MarketCountdown)
def get_market_countdown(slug: str) -> MarketCountdown:
    """Real-time countdown + expected payoff for one market.

    Gamma metadata for the slug is cached for ``_SLUG_META_TTL_SECONDS``
    (15 s) to absorb concurrent quote/peers/quality fan-out from the UI
    on market-detail open. The actual countdown numbers (days/hours/mins)
    are recomputed against ``now`` every call so the user still sees a
    live clock — only the upstream Gamma fetch is cached.
    """
    now_ts = time.time()
    cached = _SLUG_META_CACHE.get(slug)
    if cached is not None and (now_ts - cached[0]) < _SLUG_META_TTL_SECONDS:
        meta = cached[1]
    else:
        with _new_http_client() as client:
            meta = _fetch_gamma_metadata(slug, client)
        if meta is not None:
            _SLUG_META_CACHE[slug] = (now_ts, meta)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"no market found for slug={slug!r}")

    end = _parse_end_date(meta.get("endDate"))
    if end is None:
        raise HTTPException(status_code=502, detail=f"market {slug!r} has no usable endDate")
    p = _current_p_from_gamma(meta)
    if p is None:
        raise HTTPException(status_code=502, detail=f"market {slug!r} has no usable price")

    now = datetime.now(UTC)
    delta = end - now
    total_seconds = max(0, int(delta.total_seconds()))
    days = total_seconds // 86_400
    rem = total_seconds - days * 86_400
    hours = rem // 3_600
    rem -= hours * 3_600
    minutes = rem // 60

    # Under "market is fair" the expected payoff per $1 of YES held to
    # resolution is just ``current_p`` (you pay p, the contract pays $1
    # with probability p ⇒ E[payoff] = p). Round-trip PnL is therefore
    # ``current_p - p = 0`` in expectation, which the UI surfaces as
    # ``fair_price_at_resolution`` (the realised $1 if the event prints).
    return MarketCountdown(
        slug=slug,
        question=str(meta.get("question") or ""),
        current_p=p,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds_remaining=total_seconds,
        expected_resolution=end.isoformat(),
        fair_price_at_resolution=1.0 if p >= 0.5 else 0.0,
        expected_payoff_if_held=p,
    )
