"""FastAPI router exposing data-source health and the delisted registry.

Endpoints:

- ``GET /sources/health`` — run every probe in :mod:`pfm.sources.health`
  and return a per-source ``{ok, latency_ms, configured, detail}`` map.
- ``GET /sources/delisted`` — list tickers in the on-disk delisted cache.
- ``POST /sources/delisted/{ticker}`` — manually mark a ticker delisted.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from pfm.sources.equity import (
    list_delisted,
    mark_delisted,
)
from pfm.sources.health import acheck_all_sources

router = APIRouter(prefix="/sources", tags=["sources"])


class SourceProbe(BaseModel):
    """Per-source health probe result."""

    ok: bool
    latency_ms: float | None = None
    configured: bool = True
    detail: str | None = None


class SourcesHealthResponse(BaseModel):
    """Response for ``GET /sources/health``."""

    sources: dict[str, SourceProbe]
    summary: dict[str, int] = Field(..., description="Counts: total / up / down / not_configured.")


class DelistedListResponse(BaseModel):
    """Response for ``GET /sources/delisted``."""

    tickers: list[str]
    count: int


class DelistedMarkResponse(BaseModel):
    """Response for ``POST /sources/delisted/{ticker}``."""

    ticker: str
    marked: bool
    tickers: list[str]


@router.get("/health", response_model=SourcesHealthResponse)
async def sources_health() -> SourcesHealthResponse:
    raw: dict[str, dict[str, Any]] = await acheck_all_sources()
    probes = {name: SourceProbe(**payload) for name, payload in raw.items()}
    up = sum(1 for p in probes.values() if p.ok)
    not_configured = sum(1 for p in probes.values() if not p.configured)
    down = sum(1 for p in probes.values() if not p.ok and p.configured)
    return SourcesHealthResponse(
        sources=probes,
        summary={
            "total": len(probes),
            "up": up,
            "down": down,
            "not_configured": not_configured,
        },
    )


@router.get("/delisted", response_model=DelistedListResponse)
def get_delisted() -> DelistedListResponse:
    tickers = list_delisted()
    return DelistedListResponse(tickers=tickers, count=len(tickers))


@router.post("/delisted/{ticker}", response_model=DelistedMarkResponse)
def post_delisted(ticker: str) -> DelistedMarkResponse:
    mark_delisted(ticker)
    tickers = list_delisted()
    return DelistedMarkResponse(
        ticker=ticker.upper(),
        marked=True,
        tickers=tickers,
    )


__all__ = ["router"]
