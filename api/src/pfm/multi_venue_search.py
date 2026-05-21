"""Multi-venue search across Polymarket, Kalshi, Manifold, and PredictIt.

Fans a single user query out to all four venues in parallel via
``asyncio.gather`` and returns a normalised payload keyed by venue. Failure
on any one venue is captured (no exception surfaced) so a flaky upstream
doesn't blank the whole response.

Endpoints
---------
- ``GET /multi-venue/search?q=trump``
        Returns ``{polymarket, kalshi, manifold, predictit}`` with up to
        ``limit`` markets per venue (default 10).

- ``GET /multi-venue/concept/{concept_id}``
        Resolves a curated 4-venue concept (see
        :data:`pfm.arb_scanner.CONCEPT_MAPS`) to a unified per-venue
        snapshot — handy as the "hub" call powering Terminal's
        cross-venue concept page.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.arb_scanner import CONCEPT_MAPS, get_concept_map
from pfm.cache_utils import get_cache
from pfm.sources.manifold import ManifoldClient
from pfm.sources.predictit import PredictItClient

logger = logging.getLogger(__name__)

GAMMA_URL: str = "https://gamma-api.polymarket.com"
KALSHI_URL: str = "https://api.elections.kalshi.com/trade-api/v2"

_SEARCH_CACHE = get_cache("multi_venue_search", ttl=60)
_CONCEPT_CACHE = get_cache("multi_venue_concept", ttl=60)


# ---------------------------------------------------------------------------
# Per-venue async search helpers
# ---------------------------------------------------------------------------


async def _search_polymarket(
    query: str, limit: int, http: httpx.AsyncClient
) -> list[dict[str, Any]]:
    """Polymarket Gamma free-text via ``q=`` query param."""
    try:
        r = await http.get(
            f"{GAMMA_URL}/markets",
            params={"q": query, "limit": int(limit), "active": "true"},
        )
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("polymarket search failed: %s", exc)
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for m in data[: int(limit)]:
        out.append(
            {
                "venue": "polymarket",
                "id": str(m.get("id", "")),
                "slug": m.get("slug", ""),
                "title": m.get("question") or m.get("title") or "",
                "end_date": m.get("endDate"),
            }
        )
    return out


async def _search_kalshi(query: str, limit: int, http: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Kalshi public market search via ``GET /markets?status=open&q=``."""
    try:
        r = await http.get(
            f"{KALSHI_URL}/markets",
            params={"status": "open", "limit": int(limit)},
        )
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("kalshi search failed: %s", exc)
        return []
    markets = data.get("markets") if isinstance(data, dict) else None
    if not isinstance(markets, list):
        return []
    qlow = query.lower()
    out: list[dict[str, Any]] = []
    for m in markets:
        title = m.get("title") or m.get("subtitle") or ""
        if qlow and qlow not in title.lower() and qlow not in (m.get("ticker", "") or "").lower():
            continue
        out.append(
            {
                "venue": "kalshi",
                "id": str(m.get("ticker", "")),
                "slug": str(m.get("ticker", "")),
                "title": title,
                "end_date": m.get("close_time"),
            }
        )
        if len(out) >= int(limit):
            break
    return out


async def _search_manifold(query: str, limit: int, http: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Manifold via :class:`ManifoldClient.search_markets`."""
    try:
        async with ManifoldClient(client=http) as mc:
            data = await mc.search_markets(query, limit=int(limit))
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("manifold search failed: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for m in data:
        out.append(
            {
                "venue": "manifold",
                "id": str(m.get("id", "")),
                "slug": m.get("slug", ""),
                "title": m.get("question") or m.get("title") or "",
                "end_date": m.get("closeTime"),
            }
        )
    return out


async def _search_predictit(
    query: str, limit: int, http: httpx.AsyncClient
) -> list[dict[str, Any]]:
    """PredictIt: filter the all-markets snapshot client-side."""
    try:
        async with PredictItClient(client=http) as pic:
            markets = await pic.fetch_all_markets()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("predictit search failed: %s", exc)
        return []
    qlow = query.lower()
    out: list[dict[str, Any]] = []
    for m in markets:
        name = (m.get("name") or m.get("shortName") or "").lower()
        if qlow and qlow not in name:
            continue
        out.append(
            {
                "venue": "predictit",
                "id": str(m.get("id", "")),
                "slug": m.get("url") or str(m.get("id", "")),
                "title": m.get("name") or m.get("shortName") or "",
                "end_date": m.get("dateEnd"),
            }
        )
        if len(out) >= int(limit):
            break
    return out


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


async def search_all_venues(
    query: str,
    *,
    limit: int = 10,
    http: httpx.AsyncClient | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run all four venue searches in parallel and assemble the result.

    Failure on any venue → empty list for that venue. We never raise.
    """
    if not query:
        return {"polymarket": [], "kalshi": [], "manifold": [], "predictit": []}

    own_http = http is None
    http = http or httpx.AsyncClient(timeout=15.0)
    try:
        results = await asyncio.gather(
            _search_polymarket(query, limit, http),
            _search_kalshi(query, limit, http),
            _search_manifold(query, limit, http),
            _search_predictit(query, limit, http),
            return_exceptions=True,
        )
    finally:
        if own_http:
            await http.aclose()

    venues = ("polymarket", "kalshi", "manifold", "predictit")
    out: dict[str, list[dict[str, Any]]] = {}
    for venue, res in zip(venues, results, strict=True):
        if isinstance(res, BaseException):
            logger.info("multi-venue search %s raised: %s", venue, res)
            out[venue] = []
        else:
            out[venue] = res
    return out


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class VenueMarket(BaseModel):
    """One market hit from any of the four venues."""

    venue: str
    id: str
    slug: str = ""
    title: str = ""
    end_date: Any = None


class MultiVenueSearchResponse(BaseModel):
    query: str
    n_total: int = Field(..., ge=0)
    polymarket: list[VenueMarket]
    kalshi: list[VenueMarket]
    manifold: list[VenueMarket]
    predictit: list[VenueMarket]


class ConceptVenueIds(BaseModel):
    polymarket: str | None = None
    kalshi: str | None = None
    manifold: str | None = None
    predictit: int | None = None


class MultiVenueConceptResponse(BaseModel):
    concept_id: str
    label: str = ""
    theme: str = ""
    venues: ConceptVenueIds
    n_legs_present: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/multi-venue", tags=["multi-venue"])


@router.get("/search", response_model=MultiVenueSearchResponse)
async def get_search(
    q: str = Query(..., min_length=1, description="free-text search term"),
    limit: int = Query(default=10, ge=1, le=50),
) -> MultiVenueSearchResponse:
    """Free-text search fanned out across all four venues."""
    cache_key = ("search", q.lower(), int(limit))
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None:
        return MultiVenueSearchResponse(**cached)

    by_venue = await search_all_venues(q, limit=limit)
    payload: dict[str, Any] = {
        "query": q,
        "n_total": sum(len(v) for v in by_venue.values()),
        **by_venue,
    }
    _SEARCH_CACHE.set(cache_key, payload, ttl=60)
    return MultiVenueSearchResponse(**payload)


@router.get("/concept/{concept_id}", response_model=MultiVenueConceptResponse)
def get_concept(concept_id: str) -> MultiVenueConceptResponse:
    """Unified per-venue view for a curated 4-venue concept map."""
    cache_key = ("concept", concept_id.lower())
    cached = _CONCEPT_CACHE.get(cache_key)
    if cached is not None:
        return MultiVenueConceptResponse(**cached)

    concept = get_concept_map(concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail=f"unknown concept_id: {concept_id!r}")

    venues = {
        "polymarket": concept.get("polymarket"),
        "kalshi": concept.get("kalshi"),
        "manifold": concept.get("manifold"),
        "predictit": concept.get("predictit"),
    }
    n_legs = sum(1 for v in venues.values() if v is not None and v != "")
    payload: dict[str, Any] = {
        "concept_id": concept["concept_id"],
        "label": concept.get("label", ""),
        "theme": concept.get("theme", ""),
        "venues": venues,
        "n_legs_present": n_legs,
    }
    _CONCEPT_CACHE.set(cache_key, payload, ttl=60)
    return MultiVenueConceptResponse(**payload)


@router.get("/concepts")
def list_concepts() -> dict[str, Any]:
    """List every curated 4-venue concept (id + label + theme)."""
    return {
        "n": len(CONCEPT_MAPS),
        "concepts": [
            {
                "concept_id": c["concept_id"],
                "label": c.get("label", ""),
                "theme": c.get("theme", ""),
            }
            for c in CONCEPT_MAPS
        ],
    }


__all__ = [
    "ConceptVenueIds",
    "MultiVenueConceptResponse",
    "MultiVenueSearchResponse",
    "VenueMarket",
    "router",
    "search_all_venues",
]
