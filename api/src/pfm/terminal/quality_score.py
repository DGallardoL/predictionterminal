"""Terminal market-quality-score endpoint.

Exposes ``GET /terminal/quality/{slug}`` which folds spread, depth, 24h
volume, market age, days-to-resolution, and recent trade activity into a
single 0–100 quality score plus a letter grade and human-readable flags.

The aim is one number a trader can scan to decide "is this market worth
looking at?". Each sub-score is bounded to ``[0, 100]`` and combined as a
weighted average:

    spread 25%, depth 20%, vol 25%, age 10%, dte 10%, activity 10%

External calls (all GETs, all public, no auth):

  * Gamma     ``/markets?slug=...``          metadata + 24h volume + dates
  * CLOB      ``/book?token_id=...``         live YES-side orderbook
  * data-api  ``/trades?market=...&limit=N`` recent trade count proxy

HTTP is done through ``httpx`` so respx can intercept it in tests.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.terminal_export import respond as _export_respond

logger = logging.getLogger(__name__)

GAMMA_URL: str = "https://gamma-api.polymarket.com"
CLOB_URL: str = "https://clob.polymarket.com"
DATA_API_URL: str = "https://data-api.polymarket.com"

# Public router — main.py is responsible for ``app.include_router(...)``.
router = APIRouter(prefix="/terminal/quality", tags=["terminal"])

# Composite quality is dominated by quasi-static features (24h volume, age,
# dte, n_trades_24h). Spread / depth move on the second scale but the score
# itself only changes a few points across a minute. 30 s cache + a key on
# slug+format keeps successive market-detail opens (and 429-prone retries)
# off Gamma / CLOB / data-api.
_QUALITY_CACHE = get_cache("terminal_quality", ttl=30)


# ---- Weights (sum to 1.0) --------------------------------------------------

WEIGHTS: dict[str, float] = {
    "spread": 0.25,
    "depth": 0.20,
    "vol": 0.25,
    "age": 0.10,
    "dte": 0.10,
    "activity": 0.10,
}

Grade = Literal["A", "B", "C", "D", "F"]


# ---- Schemas ---------------------------------------------------------------


class QualityComponents(BaseModel):
    """Each sub-score in ``[0, 100]`` — what feeds the weighted average."""

    spread_score: float = Field(..., ge=0.0, le=100.0)
    depth_score: float = Field(..., ge=0.0, le=100.0)
    vol_score: float = Field(..., ge=0.0, le=100.0)
    age_score: float = Field(..., ge=0.0, le=100.0)
    dte_score: float = Field(..., ge=0.0, le=100.0)
    activity_score: float = Field(..., ge=0.0, le=100.0)


class QualityResponse(BaseModel):
    """Top-level quality assessment for a single Polymarket market."""

    slug: str
    quality_score: float = Field(..., ge=0.0, le=100.0)
    components: QualityComponents
    grade: Grade
    flags: list[str]
    # Diagnostic fields the UI can render alongside the score.
    spread_cents: float | None
    top_of_book_size: float | None
    volume_24hr: float | None
    age_days: int | None
    days_to_resolution: int | None
    n_trades_24h: int | None


# ---- Pure scoring helpers --------------------------------------------------


def _clip01_100(x: float) -> float:
    """Clip ``x`` into the ``[0, 100]`` interval and round to 2dp."""
    return round(max(0.0, min(100.0, x)), 2)


def score_spread(spread_cents: float | None) -> float:
    """100 if spread <= 1c, decays linearly to 0 at 10c. None → 0."""
    if spread_cents is None or spread_cents < 0:
        return 0.0
    if spread_cents <= 1.0:
        return 100.0
    if spread_cents >= 10.0:
        return 0.0
    # Linear: 1c → 100, 10c → 0.
    return _clip01_100(100.0 * (10.0 - spread_cents) / 9.0)


def score_depth(top_bid_size: float, top_ask_size: float, depth_3c: float) -> float:
    """Combine ToB sizes with $-depth inside a 3c band of mid.

    A market is "deep" if BOTH sides have meaningful size at the top AND
    there's bulk liquidity within 3c of mid. We score each piece separately
    and average them so a great ToB with a thin mid-book is penalised.

    Calibration (rough, eyeballed against typical sub-$1 contracts):
        - ToB 0      shares          → 0
        - ToB 5_000  shares per side → 100
        - depth_3c 0                → 0
        - depth_3c 50_000 shares    → 100
    """
    tob_min = min(top_bid_size, top_ask_size)
    tob_score = _clip01_100(100.0 * (tob_min / 5_000.0))
    depth_score_local = _clip01_100(100.0 * (depth_3c / 50_000.0))
    return _clip01_100(0.5 * tob_score + 0.5 * depth_score_local)


def score_volume(volume_24hr: float | None) -> float:
    """Score 24h dollar volume. 100 at $100k+, scales down log10ly to 0 at $10.

    Calibration (linear in ``log10($)``, with a hard floor at $10):
        $0..$10  → 0
        $100     → 25
        $1_000   → 50
        $10_000  → 75
        $100_000+→ 100

    The $10 floor matters: a market that traded $50 in 24h is genuinely
    illiquid, and log10(50)/log10(100k) ≈ 0.34 felt too generous. Anchoring
    to $10 → 0 / $100k → 100 puts $50 around 14, which is closer to the
    "this is unusable" reality.
    """
    if volume_24hr is None or volume_24hr <= 10.0:
        return 0.0
    # Linear in log10: $10 → 0, $100k → 100. Slope = 100/(log10(100k) - log10(10)) = 25.
    return _clip01_100(25.0 * (math.log10(volume_24hr) - 1.0))


def score_age(age_days: int | None) -> float:
    """100 if age >= 90d, scales linearly down to 50 at 7d, then to 0 at 0d.

    A two-segment ramp because brand-new markets aren't *failed* markets,
    they're just unproven. The 50-floor at 7d keeps a freshly-launched
    high-volume market from being graded F purely on age.
    """
    if age_days is None or age_days < 0:
        return 0.0
    if age_days >= 90:
        return 100.0
    if age_days >= 7:
        # 7 → 50, 90 → 100.
        return _clip01_100(50.0 + 50.0 * (age_days - 7) / (90 - 7))
    # 0 → 0, 7 → 50.
    return _clip01_100(50.0 * age_days / 7.0)


def score_dte(days_to_resolution: int | None) -> float:
    """100 if 30..300 days; 50 outside that range; 0 if <7 days.

    The "sweet spot" for fitting factor models is medium-dated markets:
    short-dated ones are dominated by news and don't have the room to
    move; very-long-dated ones don't update enough to give a clean
    history.
    """
    if days_to_resolution is None or days_to_resolution < 0:
        return 0.0
    if days_to_resolution < 7:
        return 0.0
    if 30 <= days_to_resolution <= 300:
        return 100.0
    return 50.0


def score_activity(n_trades_24h: int | None) -> float:
    """Score recent trade count. log-based; 100 at >=200 trades/24h.

    Calibration:
        0    trades → 0
        10   trades → ~50
        100  trades → ~85
        200+ trades → 100
    """
    if n_trades_24h is None or n_trades_24h <= 0:
        return 0.0
    # log10(trades+1)/log10(201) * 100; saturates at trades=200.
    return _clip01_100(100.0 * math.log10(n_trades_24h + 1) / math.log10(201.0))


def grade_from_score(score: float) -> Grade:
    """Bucket a 0-100 score into a letter grade."""
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def collect_flags(
    components: QualityComponents,
    age_days: int | None,
    dte: int | None,
    spread_cents: float | None,
    volume_24hr: float | None,
) -> list[str]:
    """Derive human-readable warnings from the sub-scores + raw inputs."""
    flags: list[str] = []
    if components.depth_score < 30.0:
        flags.append("thin_book")
    if components.vol_score < 30.0:
        flags.append("low_vol")
    if components.spread_score < 30.0:
        flags.append("wide_spread")
    if components.activity_score < 30.0:
        flags.append("low_activity")
    if dte is not None and 0 <= dte < 14:
        flags.append("near_resolution")
    if age_days is not None and 0 <= age_days < 14:
        flags.append("newly_launched")
    if spread_cents is not None and spread_cents >= 5.0:
        flags.append("crossed_or_wide")
    if volume_24hr is not None and volume_24hr < 1_000.0:
        flags.append("illiquid_24h")
    return flags


def weighted_score(c: QualityComponents) -> float:
    """Apply ``WEIGHTS`` to ``c`` and return a 0-100 composite."""
    total = (
        WEIGHTS["spread"] * c.spread_score
        + WEIGHTS["depth"] * c.depth_score
        + WEIGHTS["vol"] * c.vol_score
        + WEIGHTS["age"] * c.age_score
        + WEIGHTS["dte"] * c.dte_score
        + WEIGHTS["activity"] * c.activity_score
    )
    return _clip01_100(total)


# ---- Date / metadata parsing -----------------------------------------------


def _parse_iso_utc(raw: str | None) -> datetime | None:
    """Parse a Gamma ISO timestamp (with optional ``Z``) into a UTC datetime."""
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
    """Best-effort ``float()`` returning ``None`` on garbage."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---- HTTP fetchers ---------------------------------------------------------


_QUALITY_GAMMA_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_QUALITY_GAMMA_TTL_S: float = 3600.0


def _fetch_gamma_market(slug: str, client: httpx.Client) -> dict[str, Any]:
    """Resolve a slug to its full Gamma market dict. Raises HTTPException on miss.

    Cached for 1h with single 429-retry to absorb Polymarket gamma rate-limit
    cascades (slug→market data is effectively immutable for our use).
    """
    now = time.time()
    cached = _QUALITY_GAMMA_CACHE.get(slug)
    if cached is not None and (now - cached[0]) < _QUALITY_GAMMA_TTL_S:
        return cached[1]
    r = client.get(f"{GAMMA_URL}/markets", params={"slug": slug})
    if r.status_code == 429:
        time.sleep(1.5)
        r = client.get(f"{GAMMA_URL}/markets", params={"slug": slug})
    r.raise_for_status()
    payload = r.json()
    market = payload[0] if isinstance(payload, list) and payload else None
    if not market:
        raise HTTPException(status_code=404, detail=f"no market found for slug={slug!r}")
    _QUALITY_GAMMA_CACHE[slug] = (now, market)
    return market


def _yes_token_id(market: dict[str, Any], slug: str) -> str:
    """Pull the YES (first) entry out of ``clobTokenIds``.

    ``clobTokenIds`` arrives as a JSON-encoded string inside the JSON
    response — see PLAN.md and CLAUDE.md.
    """
    raw = market.get("clobTokenIds")
    if not raw:
        raise HTTPException(status_code=502, detail=f"market {slug!r} has no clobTokenIds")
    try:
        token_ids = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"clobTokenIds for {slug!r} is not valid JSON"
        ) from exc
    if not isinstance(token_ids, list) or not token_ids:
        raise HTTPException(status_code=502, detail=f"empty clobTokenIds for {slug!r}")
    return str(token_ids[0])


def _condition_id(market: dict[str, Any]) -> str | None:
    """Pull the conditionId for the data-api /trades query, if available."""
    cid = market.get("conditionId") or market.get("condition_id")
    return str(cid) if cid else None


def _fetch_book(token_id: str, client: httpx.Client) -> dict[str, Any]:
    """GET ``/book?token_id=...`` — returns ``{bids, asks}``."""
    r = client.get(f"{CLOB_URL}/book", params={"token_id": token_id})
    r.raise_for_status()
    return r.json()


def _fetch_trade_count(condition_id: str, client: httpx.Client, limit: int = 200) -> int:
    """Best-effort 24h trade count from data-api.

    The endpoint returns at most ``limit`` recent trades; we use the
    response length as a *lower bound* on 24h activity (saturates at
    ``limit``). Any HTTP error → ``0`` rather than failing the score.
    """
    try:
        r = client.get(
            f"{DATA_API_URL}/trades",
            params={"market": condition_id, "limit": int(limit)},
        )
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPError as exc:
        logger.warning("data-api /trades fetch failed: %s", exc)
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        trades = payload.get("trades")
        if isinstance(trades, list):
            return len(trades)
    return 0


# ---- Book-side aggregation -------------------------------------------------


def _coerce_levels(
    raw: list[dict[str, Any]], side: Literal["bid", "ask"]
) -> list[tuple[float, float]]:
    """Sort + clean book levels into ``[(price, size), ...]``."""
    parsed: list[tuple[float, float]] = []
    for entry in raw or []:
        try:
            price = float(entry["price"])
            size = float(entry["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if size <= 0:
            continue
        parsed.append((price, size))
    parsed.sort(key=lambda t: t[0], reverse=(side == "bid"))
    return parsed


def _book_summary(book: dict[str, Any]) -> tuple[float | None, float, float, float]:
    """Return ``(spread_cents, top_bid_size, top_ask_size, depth_3c)``.

    ``depth_3c`` is the sum of share sizes on either side within 3 cents
    of the mid. ``spread_cents`` is ``None`` if either side is empty.
    """
    bids = _coerce_levels(book.get("bids", []) or [], side="bid")
    asks = _coerce_levels(book.get("asks", []) or [], side="ask")
    if not bids or not asks:
        return None, 0.0, 0.0, 0.0
    best_bid_p, best_bid_sz = bids[0]
    best_ask_p, best_ask_sz = asks[0]
    spread_cents = round((best_ask_p - best_bid_p) * 100.0, 4)
    mid = (best_bid_p + best_ask_p) / 2.0
    band = 0.03
    depth_3c = sum(sz for p, sz in bids if abs(p - mid) <= band + 1e-12) + sum(
        sz for p, sz in asks if abs(p - mid) <= band + 1e-12
    )
    return spread_cents, best_bid_sz, best_ask_sz, depth_3c


# ---- Endpoint --------------------------------------------------------------


@router.get("/{slug}", response_model=None)
def get_quality(
    slug: str,
    timeout: float = Query(default=10.0, gt=0.0, le=30.0),
    format: Literal["json", "csv", "pdf"] = Query(default="json"),
) -> QualityResponse | FastAPIResponse:
    """Compute and return the composite quality score for ``slug``.

    The default ``format=json`` returns the :class:`QualityResponse`
    Pydantic model (so OpenAPI schemas stay accurate). ``csv`` / ``pdf``
    route through :func:`pfm.terminal_export.respond` for download.
    """
    # JSON path is cache-eligible (CSV/PDF go through binary export which
    # caller may want fresh). The composite shifts only slowly.
    if format == "json":
        cache_key = ("q", slug)
        cached_resp = _QUALITY_CACHE.get(cache_key)
        if cached_resp is not None:
            return cached_resp

    now = datetime.now(UTC)
    with httpx.Client(timeout=timeout) as client:
        market = _fetch_gamma_market(slug, client)
        token_id = _yes_token_id(market, slug)

        # Book — if CLOB is down we still want a score, just with depth/spread = 0.
        try:
            book = _fetch_book(token_id, client)
        except httpx.HTTPError as exc:
            logger.warning("CLOB /book failed for %s: %s", slug, exc)
            book = {"bids": [], "asks": []}

        # Trade count via data-api when conditionId is exposed.
        cid = _condition_id(market)
        n_trades_24h = _fetch_trade_count(cid, client) if cid else 0

    spread_cents, tob_bid, tob_ask, depth_3c = _book_summary(book)
    volume_24hr = _coerce_float(market.get("volume24hr"))

    start = _parse_iso_utc(market.get("startDate"))
    end = _parse_iso_utc(market.get("endDate"))
    age_days = max(0, int((now - start).total_seconds() // 86_400)) if start else None
    dte = max(0, int((end - now).total_seconds() // 86_400)) if end else None

    components = QualityComponents(
        spread_score=score_spread(spread_cents),
        depth_score=score_depth(tob_bid, tob_ask, depth_3c),
        vol_score=score_volume(volume_24hr),
        age_score=score_age(age_days),
        dte_score=score_dte(dte),
        activity_score=score_activity(n_trades_24h),
    )
    quality = weighted_score(components)
    grade = grade_from_score(quality)
    flags = collect_flags(
        components,
        age_days=age_days,
        dte=dte,
        spread_cents=spread_cents,
        volume_24hr=volume_24hr,
    )

    tob_size = min(tob_bid, tob_ask) if (tob_bid > 0 and tob_ask > 0) else None

    resp = QualityResponse(
        slug=slug,
        quality_score=quality,
        components=components,
        grade=grade,
        flags=flags,
        spread_cents=spread_cents,
        top_of_book_size=tob_size,
        volume_24hr=volume_24hr,
        age_days=age_days,
        days_to_resolution=dte,
        n_trades_24h=n_trades_24h if cid else None,
    )
    if format == "json":
        _QUALITY_CACHE.set(("q", slug), resp)
        return resp
    return _export_respond(resp, format, filename=f"quality-{slug}", kind="quality")
