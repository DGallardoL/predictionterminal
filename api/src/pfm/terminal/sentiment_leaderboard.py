"""Sentiment-vs-price *mispricing-signal* leaderboard.

For the top-50 markets by 24h volume we run the existing
``/terminal/jumps/{slug}`` pipeline and rank them by the *density* of
**price-jump / news-sentiment disagreements**:

A "disagrees" row from :mod:`pfm.terminal.jumps` means the aggregate
news sentiment in the [-2h, +1h] window around the jump **disagrees
with the direction of the jump** (e.g. positive headlines, price
went down). These are the rows most worth a human read — either the
market is mispriced relative to public information, or there is an
unobserved driver that the wire hasn't caught yet. Either way, the
top of this leaderboard is where the alpha hunting starts.

Routing
-------
Owns its own :class:`fastapi.APIRouter`; ``main.py`` wires it the
same way as every other ``terminal.*`` sub-router::

    from pfm.terminal.sentiment_leaderboard import (
        router as terminal_sentiment_leaderboard_router,
    )
    app.include_router(terminal_sentiment_leaderboard_router)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.config import Settings, get_settings
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal.homepage import _fetch_top_markets_async, _safe_float
from pfm.terminal.jumps import DEFAULT_MAD_K, DEFAULT_MIN_JUMP_PP, get_jumps

_get_jumps_endpoint = get_jumps

logger = logging.getLogger(__name__)

# --- limits / cache ---------------------------------------------------------

DEFAULT_DAYS: int = 7
MIN_DAYS: int = 1
MAX_DAYS: int = 30
DEFAULT_MIN_JUMPS: int = 3
MIN_MIN_JUMPS: int = 0
MAX_MIN_JUMPS: int = 50
# Each /terminal/jumps call costs ~8-12s cold (gathers GDELT + Reddit + HN +
# 9 RSS feeds in parallel). 50 markets × 8 concurrency = ~7 batches = ~80s,
# which times out the gateway. With 15 we land in ~15-25s for a cold call,
# 5ms warm (10-min cache). The user can call min_jumps=2 to widen coverage.
# Lowered 2026-05-18 from 15 to 8 — even with 12-way concurrency the
# 15-market cold fan-out routinely exceeded the 15 s gateway timeout
# because each /jumps fetch gathers GDELT+Reddit+HN+9 RSS in parallel.
# Eight markets × 8-way concurrency ≈ ~10 s cold worst case.
TOP_MARKETS_CONSIDERED: int = 8
TOP_LEADERBOARD_ROWS: int = 15
CONCURRENCY: int = 8
HTTP_TIMEOUT_SECONDS: float = 10.0
CACHE_TTL_SECONDS: int = 600  # 10 min — matches jumps' own TTL.

_CACHE = get_cache("terminal_sentiment_leaderboard", ttl=CACHE_TTL_SECONDS)


def clear_cache() -> None:
    """Test/utility — drop the cached leaderboard payload."""
    _CACHE.clear()


# --- schemas ----------------------------------------------------------------


class SentimentLeaderboardRow(BaseModel):
    """One row in the leaderboard: a single market summarised."""

    rank: int = Field(
        ...,
        ge=0,
        description="1-based rank in the leaderboard (0 = pre-sort placeholder, never returned).",
    )
    slug: str
    name: str | None = Field(
        None, description="Market question text, if known from the gamma listing."
    )
    theme: str | None = None
    volume_24h: float | None = Field(
        None, description="24h notional volume from the gamma listing."
    )
    n_jumps: int = Field(..., ge=0, description="Total detected price jumps over the window.")
    n_explained: int = Field(
        ...,
        ge=0,
        description="Jumps with ≥1 matched news article (any sentiment).",
    )
    n_disagrees: int = Field(
        ...,
        ge=0,
        description="Jumps where news sentiment disagrees with the price-jump direction.",
    )
    disagrees_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="100 * n_disagrees / max(n_jumps, 1). Mispricing-signal density.",
    )


class SentimentLeaderboardResponse(BaseModel):
    """Top markets by mispricing-signal density."""

    days: int
    min_jumps: int
    n_markets_considered: int = Field(
        ...,
        ge=0,
        description="Number of top-by-volume markets that were probed.",
    )
    n_markets_qualified: int = Field(
        ...,
        ge=0,
        description="Markets that met the min_jumps filter and entered the ranking.",
    )
    rows: list[SentimentLeaderboardRow] = Field(default_factory=list)
    interpretation: str


# --- helpers ----------------------------------------------------------------


def _get_polymarket_client(request: Request) -> PolymarketClient:
    poly: PolymarketClient | None = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


async def _jumps_for_slug(
    request: Request,
    poly: PolymarketClient,
    slug: str,
    days: int,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Wrap :func:`pfm.terminal.jumps.get_jumps` with bounded concurrency.

    Returns the response's ``model_dump()`` dict or ``None`` on any error
    (a single bad slug must not poison the whole leaderboard).
    """
    async with sem:
        try:
            payload = await _get_jumps_endpoint(
                request=request,
                slug=slug,
                days=days,
                mad_k=DEFAULT_MAD_K,
                min_jump_pp=DEFAULT_MIN_JUMP_PP,
                poly=poly,
            )
        except HTTPException as e:
            logger.debug("sentiment-leaderboard: jumps failed for %s: %s", slug, e.detail)
            return None
        except Exception as e:
            logger.debug("sentiment-leaderboard: jumps crashed for %s: %s", slug, e)
            return None
    return payload.model_dump()


# --- router -----------------------------------------------------------------


router = APIRouter(prefix="/terminal", tags=["terminal-sentiment-leaderboard"])


@router.get(
    "/sentiment-leaderboard",
    response_model=SentimentLeaderboardResponse,
    summary="Rank top-volume markets by news-sentiment / price-jump disagreement density.",
)
async def get_sentiment_leaderboard(
    request: Request,
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
    min_jumps: Annotated[int, Query(ge=MIN_MIN_JUMPS, le=MAX_MIN_JUMPS)] = DEFAULT_MIN_JUMPS,
) -> SentimentLeaderboardResponse:
    """Top-25 markets where news-sentiment most often disagrees with price jumps.

    Reads the top-50 markets by 24h volume from the same gamma listing the
    homepage uses, fans out ``/terminal/jumps/{slug}`` calls with bounded
    concurrency, and ranks by ``disagrees_pct = n_disagrees / n_jumps``
    descending. Markets with fewer than ``min_jumps`` total jumps are
    excluded — the ranking is noise-dominated when the denominator is tiny.

    Cached for ``CACHE_TTL_SECONDS = 600`` keyed on ``(days, min_jumps)``;
    a warm hit returns in <2 ms.
    """
    cache_key = (int(days), int(min_jumps))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return SentimentLeaderboardResponse.model_validate(cached)

    settings: Settings = get_settings()
    shared_http: httpx.AsyncClient | None = getattr(request.app.state, "async_http", None)

    # 1. Top-50 markets by 24h volume from the gamma listing.
    try:
        async with AsyncExitStack() as stack:
            http = shared_http
            if http is None:
                http = await stack.enter_async_context(
                    httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
                )
            markets = await _fetch_top_markets_async(http, settings.polymarket_gamma_url, pages=3)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"upstream gamma error: {e}") from e

    # Sort by 24h volume desc, dedup by slug, take top N.
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for m in sorted(
        markets,
        key=lambda mm: -(_safe_float(mm.get("volume24hr")) or 0.0),
    ):
        slug = str(m.get("slug") or "").strip()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        candidates.append(m)
        if len(candidates) >= TOP_MARKETS_CONSIDERED:
            break

    poly = _get_polymarket_client(request)
    sem = asyncio.Semaphore(CONCURRENCY)

    # 2. Fan-out jumps with bounded concurrency.
    tasks = [_jumps_for_slug(request, poly, str(m.get("slug")), int(days), sem) for m in candidates]
    results = await asyncio.gather(*tasks)

    # 3. Build per-market rows.
    rows: list[SentimentLeaderboardRow] = []
    for m, payload in zip(candidates, results, strict=False):
        if payload is None:
            continue
        n_jumps = int(payload.get("n_jumps", 0) or 0)
        if n_jumps < min_jumps:
            continue
        n_explained = int(payload.get("n_explained", 0) or 0)
        jumps = payload.get("jumps") or []
        n_disagrees = sum(1 for j in jumps if (j or {}).get("sentiment_alignment") == "disagrees")
        disagrees_pct = 100.0 * n_disagrees / max(n_jumps, 1)
        rows.append(
            SentimentLeaderboardRow(
                rank=0,  # filled in after sort
                slug=str(m.get("slug") or ""),
                name=str(m.get("question") or "") or None,
                theme=None,  # gamma's market dict doesn't pre-tag a theme
                volume_24h=_safe_float(m.get("volume24hr")),
                n_jumps=n_jumps,
                n_explained=n_explained,
                n_disagrees=n_disagrees,
                disagrees_pct=round(disagrees_pct, 1),
            )
        )

    # 4. Rank by disagrees_pct desc, tie-break on absolute n_disagrees, then volume.
    rows.sort(
        key=lambda r: (
            -r.disagrees_pct,
            -r.n_disagrees,
            -(r.volume_24h or 0.0),
        )
    )
    rows = rows[:TOP_LEADERBOARD_ROWS]
    for i, r in enumerate(rows, start=1):
        r.rank = i

    n_qualified = len(rows)
    if n_qualified == 0:
        interpretation = (
            f"No markets with ≥{min_jumps} detected jumps in the last {days}d "
            "across the top-50 by 24h volume. Lower min_jumps or widen the window."
        )
    else:
        top = rows[0]
        interpretation = (
            f"Top mispricing-signal density: {top.slug} "
            f"({top.n_disagrees}/{top.n_jumps} jumps disagreeing with news sentiment, "
            f"{top.disagrees_pct:.1f}%). Higher = more rows where the price moved "
            "opposite the wire's tone — worth a human read."
        )

    resp = SentimentLeaderboardResponse(
        days=int(days),
        min_jumps=int(min_jumps),
        n_markets_considered=len(candidates),
        n_markets_qualified=n_qualified,
        rows=rows,
        interpretation=interpretation,
    )
    _CACHE.set(cache_key, resp.model_dump(), ttl=CACHE_TTL_SECONDS)
    return resp


__all__ = [
    "CACHE_TTL_SECONDS",
    "SentimentLeaderboardResponse",
    "SentimentLeaderboardRow",
    "clear_cache",
    "get_sentiment_leaderboard",
    "router",
]
