"""Aligned multi-market jump timelines for correlated event days.

``GET /terminal/jumps/compare?slugs=a,b,c`` runs the existing
``/terminal/jumps/{slug}`` detector for each of the supplied slugs and
returns:

1. ``jumps_by_slug``: the full per-slug jump list (verbatim from
   :func:`pfm.terminal.jumps.get_jumps`) so a frontend can render each
   timeline individually.
2. ``common_days``: a chronologically sorted list of UTC dates on which
   ≥2 of the supplied slugs had at least one detected jump, with the
   largest signed ``delta_pp`` per slug on that day plus a count of
   articles that appear across **more than one** of those slugs (so the
   frontend can flag "this looked like a macro day, not a one-market
   story").

How is this different from ``/terminal/jumps/cluster``?
-------------------------------------------------------
``/jumps/cluster`` groups jumps by tight time proximity (±5 min default)
and semantic-term Jaccard — it surfaces *intra-day* macro events. This
endpoint operates on UTC-date granularity and is meant for "show me
all the days these three markets co-moved" plots. Coarser, looser,
visualisation-first.

Shared-news count semantics
---------------------------
For a given common-day across slugs ``S = {a, b, ...}`` we collect
every article that any member jump on that date pinned, then count
articles whose **identifier** appears in jumps of ≥2 distinct slugs
on that date. The identifier prefers ``url`` (cheapest, most precise);
falls back to a SimHash bucket on the headline so near-duplicates
across wires still collapse correctly. This honours T20's news-dedupe
contract without forcing the comparison router to drag in the full
``dedupe_news`` pipeline.

Caching
-------
Keyed on the **sorted** slug tuple so ``?slugs=a,b,c`` and
``?slugs=c,b,a`` share the same cache entry. 120s TTL — short enough
that a new headline lands within the polling cadence the frontend uses,
long enough that scrubbing across markets in the UI is responsive.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal.jumps import (
    DEFAULT_MAD_K,
    DEFAULT_MIN_JUMP_PP,
    Jump,
    TerminalJumpsResponse,
    get_jumps,
)

try:  # SimHash dedupe (from T20). Optional — fall back to URL-only matching.
    from pfm.terminal.news_dedupe import simhash as _simhash  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised in the fallback path
    _simhash = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# --- knobs / cache ----------------------------------------------------------

MAX_SLUGS: int = 8
MIN_DAYS: int = 1
MAX_DAYS: int = 90
DEFAULT_DAYS: int = 14
CONCURRENCY: int = 8  # one task per slug — never exceeds MAX_SLUGS
CACHE_TTL_SECONDS: int = 120
SIMHASH_BUCKET_BITS: int = 8  # keep top-N bits of the SimHash as a bucket id

_CACHE = get_cache("terminal_jumps_compare", ttl=CACHE_TTL_SECONDS)


# --- schemas ----------------------------------------------------------------


class CommonDay(BaseModel):
    """One UTC date on which ≥2 of the requested slugs had a jump."""

    date: str = Field(..., description="UTC date in YYYY-MM-DD form.")
    jumps: dict[str, float | None] = Field(
        default_factory=dict,
        description=(
            "Per-slug largest signed delta_pp on this date, or null if "
            "the slug had no jump on this date."
        ),
    )
    shared_news_count: int = Field(
        0,
        ge=0,
        description=(
            "Number of distinct articles that appear in jumps of ≥2 "
            "slugs on this date. Identifier prefers URL with a SimHash "
            "fallback for near-duplicates across wires."
        ),
    )


class JumpsCompareResponse(BaseModel):
    slugs: list[str] = Field(..., description="Echoed slugs in the order received.")
    days: int = Field(..., ge=MIN_DAYS, le=MAX_DAYS)
    common_days: list[CommonDay] = Field(
        default_factory=list,
        description="UTC dates with ≥2 slugs having a jump, sorted ascending.",
    )
    jumps_by_slug: dict[str, list[Jump]] = Field(
        default_factory=dict,
        description="Full per-slug jump list for every requested slug.",
    )


# --- helpers ----------------------------------------------------------------


def _utc_date(ts_iso: str) -> str | None:
    """Return the ``YYYY-MM-DD`` UTC date of an ISO timestamp, or ``None``."""
    if not ts_iso:
        return None
    try:
        ts = pd.Timestamp(ts_iso)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d")


def _article_identifier(url: str | None, headline: str | None) -> str | None:
    """Stable identifier per article.

    Prefer ``url`` because it's the cheapest exact match. Fall back to a
    SimHash bucket (top ``SIMHASH_BUCKET_BITS`` bits) on the headline so
    near-duplicates across wires collapse, mirroring T20's contract. If
    neither URL nor headline is usable return None and the caller skips.
    """
    u = (url or "").strip()
    if u:
        return f"u:{u}"
    h = (headline or "").strip()
    if not h:
        return None
    if _simhash is not None:
        try:
            sig = _simhash(h)
        except Exception:  # pragma: no cover - defensive
            return f"h:{h.lower()}"
        # Right-shift to widen the equivalence class — exact-hash would
        # almost never collide on hand-typed headlines.
        bucket = sig >> max(0, 64 - SIMHASH_BUCKET_BITS)
        return f"s:{bucket:x}"
    return f"h:{h.lower()}"


def _largest_signed_delta_pp(jumps: list[Jump]) -> float:
    """Pick the signed ``delta_pp`` with the **largest magnitude** in a list.

    A 7pp drop is more newsworthy than a 5pp rise; the sign is kept so the
    frontend can colour the cell. Empty list → 0.0 (caller checks
    membership separately).
    """
    if not jumps:
        return 0.0
    return max((float(j.delta_pp) for j in jumps), key=abs)


def _build_common_days(jumps_by_slug: dict[str, list[Jump]]) -> list[CommonDay]:
    """Compute the ``common_days`` array from per-slug jumps.

    Two passes:

    1. Bucket every jump under ``(date, slug)``.
    2. For each ``date`` where ≥2 slugs are represented, build a
       :class:`CommonDay` whose ``jumps`` map contains a per-slug entry
       (signed delta_pp or null). Then count articles whose identifier
       appears across ≥2 slugs on that date — that's ``shared_news_count``.
    """
    # date -> slug -> list[Jump]
    by_date: dict[str, dict[str, list[Jump]]] = defaultdict(lambda: defaultdict(list))
    all_slugs = list(jumps_by_slug.keys())
    for slug, jumps in jumps_by_slug.items():
        for jump in jumps:
            d = _utc_date(jump.ts_iso)
            if d is None:
                continue
            by_date[d][slug].append(jump)

    out: list[CommonDay] = []
    for d, slug_map in by_date.items():
        if len(slug_map) < 2:
            # Cluster-of-one day — uninteresting for a "co-move" view.
            continue
        # Build per-slug deltas. Slugs with no jump on this date are explicit
        # nulls so the frontend can render an empty cell.
        per_slug: dict[str, float | None] = {}
        for s in all_slugs:
            js = slug_map.get(s) or []
            per_slug[s] = _largest_signed_delta_pp(js) if js else None

        # Shared-news: count identifiers that appear under ≥2 distinct slugs.
        ident_to_slugs: dict[str, set[str]] = defaultdict(set)
        for s, js in slug_map.items():
            for jump in js:
                for art in jump.top_articles or []:
                    ident = _article_identifier(getattr(art, "url", None), art.headline)
                    if ident is None:
                        continue
                    ident_to_slugs[ident].add(s)
        shared = sum(1 for slugs in ident_to_slugs.values() if len(slugs) >= 2)

        out.append(
            CommonDay(
                date=d,
                jumps=per_slug,
                shared_news_count=int(shared),
            )
        )

    out.sort(key=lambda c: c.date)
    return out


# --- routing ---------------------------------------------------------------


router = APIRouter(prefix="/terminal", tags=["terminal-jumps-compare"])


def _get_polymarket_client(request: Request) -> PolymarketClient:
    poly: PolymarketClient | None = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


async def _safe_get_jumps_for_slug(
    request: Request,
    slug: str,
    days: int,
    mad_k: float,
    min_jump_pp: float,
    poly: PolymarketClient,
    sem: asyncio.Semaphore,
) -> tuple[str, list[Jump]]:
    """Call ``get_jumps`` for one slug; degrade to empty on per-slug error.

    Mirrors :func:`pfm.terminal.jumps_cluster._safe_get_jumps_for_slug` — a
    flaky single slug must not nuke the whole comparison response.
    """
    async with sem:
        try:
            resp: TerminalJumpsResponse = await get_jumps(  # type: ignore[misc]
                request=request,
                slug=slug,
                days=days,
                mad_k=mad_k,
                min_jump_pp=min_jump_pp,
                poly=poly,
            )
            return slug, list(resp.jumps)
        except HTTPException as e:
            logger.info("jumps_compare: slug %s skipped (%s)", slug, e.detail)
            return slug, []
        except Exception as e:
            logger.warning("jumps_compare: slug %s failed: %s", slug, e)
            return slug, []


@router.get(
    "/jumps/compare",
    response_model=JumpsCompareResponse,
    summary="Aligned per-slug jump timelines + shared-news days for ≤8 slugs.",
)
async def get_jumps_compare(
    request: Request,
    slugs: Annotated[
        str,
        Query(
            description=("Comma-separated Polymarket slugs (1..8). At least one slug is required."),
            min_length=1,
            max_length=2000,
        ),
    ],
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
    mad_k: Annotated[float, Query(ge=1.0, le=10.0)] = DEFAULT_MAD_K,
    min_jump_pp: Annotated[float, Query(ge=0.5, le=50.0)] = DEFAULT_MIN_JUMP_PP,
) -> JumpsCompareResponse:
    """Run jump detection on every slug and surface co-move days."""
    # 1. Parse + cap. Preserve original order in the response, but cache on
    #    the sorted tuple so any permutation of the same set is a hit.
    requested = [s.strip() for s in (slugs or "").split(",") if s.strip()]
    # Dedupe while preserving order — repeated slugs would only waste a fan-out.
    seen: set[str] = set()
    requested_unique: list[str] = []
    for s in requested:
        if s not in seen:
            seen.add(s)
            requested_unique.append(s)
    if not requested_unique:
        raise HTTPException(status_code=400, detail="at least one slug required")
    if len(requested_unique) > MAX_SLUGS:
        raise HTTPException(
            status_code=400,
            detail=f"too many slugs (max {MAX_SLUGS}, got {len(requested_unique)})",
        )

    cache_key = (
        tuple(sorted(requested_unique)),
        int(days),
        round(float(mad_k), 2),
        round(float(min_jump_pp), 2),
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        # Re-project ``slugs`` order so the response respects what the
        # *caller* sent (not the sorted cache-key form).
        payload = JumpsCompareResponse(**cached)
        payload.slugs = list(requested_unique)
        # Re-order jumps_by_slug + per-day jump dicts to match input order.
        ordered_jumps = {s: payload.jumps_by_slug.get(s, []) for s in requested_unique}
        payload.jumps_by_slug = ordered_jumps
        for day in payload.common_days:
            day.jumps = {s: day.jumps.get(s) for s in requested_unique}
        return payload

    poly = _get_polymarket_client(request)

    # 2. Fan out — bounded by both ``CONCURRENCY`` and ``MAX_SLUGS``.
    sem = asyncio.Semaphore(min(CONCURRENCY, MAX_SLUGS))
    tasks = [
        _safe_get_jumps_for_slug(
            request=request,
            slug=s,
            days=int(days),
            mad_k=float(mad_k),
            min_jump_pp=float(min_jump_pp),
            poly=poly,
            sem=sem,
        )
        for s in requested_unique
    ]
    pairs = await asyncio.gather(*tasks)
    # Preserve caller order even if asyncio.gather happens to return out-of-order
    # (gather *does* preserve order today; this is defensive against future churn).
    pair_map: dict[str, list[Jump]] = dict(pairs)
    jumps_by_slug: dict[str, list[Jump]] = {s: pair_map.get(s, []) for s in requested_unique}

    # 3. Align by UTC date + compute shared-news counts.
    common_days = _build_common_days(jumps_by_slug)

    payload = JumpsCompareResponse(
        slugs=list(requested_unique),
        days=int(days),
        common_days=common_days,
        jumps_by_slug=jumps_by_slug,
    )
    _CACHE.set(cache_key, payload.model_dump(), ttl=CACHE_TTL_SECONDS)
    return payload


# Re-export the article-identifier helper for tests that want to exercise
# the SimHash-fallback path directly without standing up a TestClient.
_PUBLIC_HELPERS = (_article_identifier, _build_common_days, _utc_date)


__all__ = [
    "MAX_SLUGS",
    "CommonDay",
    "JumpsCompareResponse",
    "_article_identifier",
    "_build_common_days",
    "_utc_date",
    "get_jumps_compare",
    "router",
]
