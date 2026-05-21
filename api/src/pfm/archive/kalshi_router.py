"""FastAPI router for the Kalshi archive + cross-venue comparator.

Endpoints (all GET, all cached at the function layer for 1h):

- ``/archive/kalshi/markets``
- ``/archive/kalshi/market/{ticker}``  (optional ``?format=csv`` export)
- ``/archive/kalshi/series``
- ``/archive/cross-venue/concepts``
- ``/archive/cross-venue/{concept}``

The router carries tag ``archive-kalshi``. Mounting is left to whoever
assembles the app — :mod:`pfm.main` is intentionally not edited in this
slice. Tests build a throw-away :class:`fastapi.FastAPI` and mount the
router directly.
"""

from __future__ import annotations

import csv
import io
from datetime import date
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from pfm.archive.cross_venue_archive import (
    CROSS_VENUE_CONCEPTS,
    cross_venue_resolved_pairs,
    list_concepts,
)
from pfm.archive.kalshi_archive import (
    fetch_archive_kalshi_detail,
    fetch_settled_markets,
    kalshi_archive_series_distribution,
)
from pfm.sources.kalshi import KalshiError

router = APIRouter(prefix="/archive", tags=["archive-kalshi"])


# ─── Pydantic schemas ─────────────────────────────────────────────────────


class SettledMarketRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticker: str
    title: str
    series: str
    settle_date: str | None = None
    settle_value: str | None = None
    open_interest: float = 0.0
    total_volume: float = 0.0
    last_trade_price: float | None = None


class SettledMarketsResponse(BaseModel):
    items: list[SettledMarketRow]
    n: int = Field(ge=0)
    limit: int
    offset: int
    series_ticker: str | None = None
    start: str | None = None
    end: str | None = None


class HistoryRow(BaseModel):
    date: str
    price: float
    volume: float
    open_interest: float
    yes_bid: float
    yes_ask: float
    spread: float


class ArchiveDetailStats(BaseModel):
    peak_price: float
    trough_price: float
    total_volume: float
    n_days: int
    half_life_to_settle: float | None = None
    realized_vol: float | None = None
    n_traders: int | None = None
    top_wallets: list[str] = []


class ArchiveDetailResponse(BaseModel):
    ticker: str
    title: str
    series: str
    status: str | None = None
    settle_date: str | None = None
    settle_value: str | None = None
    open_time: str | None = None
    close_time: str | None = None
    history: list[HistoryRow]
    stats: ArchiveDetailStats


class SeriesStatsRow(BaseModel):
    n_markets: int
    avg_volume: float
    total_volume: float
    pct_yes: float


class SeriesDistributionResponse(BaseModel):
    series: dict[str, SeriesStatsRow]
    n_total_markets: int
    n_series: int


class ConceptCatalogEntry(BaseModel):
    concept: str
    description: str | None = None
    polymarket_slug: str | None = None
    kalshi_ticker: str | None = None
    resolved_outcome: str | None = None


class ConceptListResponse(BaseModel):
    concepts: list[ConceptCatalogEntry]
    n: int


class CrossVenueResponse(BaseModel):
    concept: str
    description: str | None = None
    polymarket_slug: str | None = None
    kalshi_ticker: str | None = None
    resolved_outcome: str | None = None
    n_overlap_days: int
    first_overlap_day: str | None = None
    last_overlap_day: str | None = None
    spread_at_resolution: float | None = None
    max_spread_observed: float | None = None
    days_diverged: int
    divergence_threshold: float
    pct_time_pm_higher: float | None = None
    error: str | None = None


# ─── endpoints ────────────────────────────────────────────────────────────


@router.get(
    "/kalshi/markets",
    response_model=SettledMarketsResponse,
    summary="Paginated list of settled Kalshi markets.",
)
async def list_settled_markets(
    start: Annotated[date | None, Query(description="Lower bound on settle date.")] = None,
    end: Annotated[date | None, Query(description="Upper bound on settle date.")] = None,
    series: Annotated[
        str | None, Query(description="Restrict to one series ticker (e.g. KXFEDDECISION).")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SettledMarketsResponse:
    items = await fetch_settled_markets(
        start_date=start,
        end_date=end,
        series_ticker=series,
        limit=limit,
        offset=offset,
    )
    return SettledMarketsResponse(
        items=[SettledMarketRow(**i) for i in items],
        n=len(items),
        limit=limit,
        offset=offset,
        series_ticker=series,
        start=start.isoformat() if start else None,
        end=end.isoformat() if end else None,
    )


@router.get(
    "/kalshi/market/{ticker}",
    summary="Per-market detail (metadata + history + stats), optionally as CSV.",
)
def get_archive_detail(
    ticker: Annotated[str, Path(min_length=1)],
    format: Annotated[str | None, Query(pattern="^(json|csv)$")] = "json",
) -> Any:
    try:
        payload = fetch_archive_kalshi_detail(ticker)
    except httpx.HTTPStatusError as e:
        # Upstream Kalshi 4xx (unknown ticker) → return our own 404 instead
        # of bubbling a 502 the frontend has to special-case.
        if 400 <= e.response.status_code < 500:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "kalshi_market_not_found",
                    "message": f"No Kalshi archive entry for ticker {ticker!r}.",
                    "ticker": ticker,
                },
            ) from e
        raise HTTPException(status_code=502, detail=f"kalshi archive error: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"kalshi archive error: {e}") from e

    if format == "csv":
        return PlainTextResponse(_history_to_csv(payload), media_type="text/csv")
    return ArchiveDetailResponse(**payload)


@router.get(
    "/kalshi/series",
    response_model=SeriesDistributionResponse,
    summary="Per-series stats over all settled Kalshi markets.",
)
async def get_series_distribution() -> SeriesDistributionResponse:
    # Wrap upstream errors as a 502 (matching ``get_archive_detail`` above)
    # so a transient Kalshi blip surfaces as a meaningful proxy error
    # rather than a bare 500. The single-flight inside
    # ``kalshi_archive_series_distribution`` dedupes concurrent first-callers.
    try:
        payload = await kalshi_archive_series_distribution()
    except (httpx.HTTPError, KalshiError) as e:
        raise HTTPException(status_code=502, detail=f"kalshi archive error: {e}") from e
    return SeriesDistributionResponse(
        series={k: SeriesStatsRow(**v) for k, v in payload.get("series", {}).items()},
        n_total_markets=int(payload.get("n_total_markets", 0)),
        n_series=int(payload.get("n_series", 0)),
    )


@router.get(
    "/cross-venue/concepts",
    response_model=ConceptListResponse,
    summary="Catalog of pre-mapped cross-venue concepts (PM vs Kalshi).",
)
def get_concept_catalog() -> ConceptListResponse:
    items = list_concepts()
    return ConceptListResponse(
        concepts=[ConceptCatalogEntry(**i) for i in items],
        n=len(items),
    )


@router.get(
    "/cross-venue/{concept}",
    response_model=CrossVenueResponse,
    summary="Polymarket vs Kalshi divergence metrics for a resolved concept.",
)
def get_cross_venue(concept: Annotated[str, Path(min_length=1)]) -> CrossVenueResponse:
    if concept not in CROSS_VENUE_CONCEPTS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown concept {concept!r}; see /archive/cross-venue/concepts",
        )
    payload = cross_venue_resolved_pairs(concept)
    return CrossVenueResponse(**payload)


# ─── helpers ──────────────────────────────────────────────────────────────


def _history_to_csv(payload: dict[str, Any]) -> str:
    """Render the per-day history rows as RFC 4180 CSV."""
    buf = io.StringIO()
    fieldnames = [
        "date",
        "price",
        "volume",
        "open_interest",
        "yes_bid",
        "yes_ask",
        "spread",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in payload.get("history", []):
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return buf.getvalue()


__all__ = ["router"]
