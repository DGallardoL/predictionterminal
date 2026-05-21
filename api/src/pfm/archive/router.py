"""FastAPI router for the Polymarket archive.

Exposes five GETs and one POST under ``/archive/polymarket/*``:

    GET  /archive/polymarket/markets?start=&end=&theme=&limit=&offset=
    GET  /archive/polymarket/market/{slug}        (with ?format=csv option)
    GET  /archive/polymarket/themes
    GET  /archive/polymarket/resolutions/{slug}
    GET  /archive/polymarket/search?q=...&limit=...
    POST /archive/polymarket/export-bulk          {slugs, format}

Per CLAUDE.md, ``main.py`` is left untouched here — Damian wires the
router in when he's ready by adding ``app.include_router(router)``.
Tests build a throw-away FastAPI app and mount the router directly via
:class:`fastapi.testclient.TestClient`.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import date, datetime, timedelta
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from pfm.archive.polymarket_archive import (
    archive_themes_distribution,
    fetch_archive_market_detail,
    fetch_resolved_markets,
    search_archive,
)
from pfm.archive.resolutions import get_resolution

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/archive/polymarket",
    tags=["archive-polymarket"],
)


# --- Pydantic schemas (inline, v2) -------------------------------------------


class ResolvedMarketSummary(BaseModel):
    """One row of the ``/markets`` list response."""

    id: str
    slug: str | None
    question: str
    theme: str
    end_date: str | None
    resolution: Literal["YES", "NO", "AMBIGUOUS", "PENDING"]
    final_price: float | None
    total_volume: float | None
    total_traders: float | None


class ResolvedMarketsListResponse(BaseModel):
    n_markets: int
    limit: int
    offset: int
    markets: list[ResolvedMarketSummary]


class ArchiveStats(BaseModel):
    peak_price: float | None = None
    peak_date: str | None = None
    trough_price: float | None = None
    trough_date: str | None = None
    max_volume_day: str | None = None
    total_volume: float | None = None
    half_life_to_resolution: int | None = None
    volatility_realized: float | None = None
    hurst_exponent: float | None = None
    dfa_alpha: float | None = None
    n_unique_traders: int | None = None
    whale_concentration: float | None = None


class HistoryPoint(BaseModel):
    """Tuple-style point: ``[date, price, volume]``."""

    date: str
    price: float
    volume: float | None = None


class NewsItem(BaseModel):
    title: str
    url: str
    ts: str


class ArchiveMarketDetail(BaseModel):
    slug: str
    question: str
    theme: str
    end_date: str | None
    resolution: Literal["YES", "NO", "AMBIGUOUS", "PENDING"]
    final_price: float | None
    history: list[list[Any]] = Field(
        ..., description="Each entry is ``[date_iso, price, volume_or_null]``."
    )
    stats: ArchiveStats
    top_news_around_resolution: list[NewsItem]


class ThemeRow(BaseModel):
    theme: str
    n_markets: int
    pct_yes: float
    pct_no: float
    pct_ambiguous: float
    avg_duration_days: float | None = None
    avg_volume: float | None = None


class ThemesDistributionResponse(BaseModel):
    n_markets_total: int
    themes: list[ThemeRow]


class ResolutionRecord(BaseModel):
    slug: str
    resolution: Literal["YES", "NO", "AMBIGUOUS", "PENDING"]
    resolution_date: str | None
    resolution_source: str | None
    payout_per_share: float | None
    dispute_history: list[dict[str, Any]]


class SearchResponse(BaseModel):
    q: str
    n_results: int
    results: list[ResolvedMarketSummary]


class BulkExportRequest(BaseModel):
    slugs: list[str] = Field(..., min_length=1, max_length=100)
    format: Literal["csv", "json", "parquet"] = "csv"


# --- helpers -----------------------------------------------------------------


def _parse_date(s: str, *, label: str) -> date:
    """Parse an ISO date or 422 with a helpful detail."""
    try:
        return datetime.fromisoformat(s).date()
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{label} must be ISO-8601 (YYYY-MM-DD); got {s!r}",
        ) from exc


def _detail_to_csv(detail: dict[str, Any]) -> str:
    """Serialize the daily-history portion of a detail dict as CSV.

    Columns: ``date,price,volume,sentiment``. ``sentiment`` is left empty
    here — the archive doesn't compute it inline; callers that want it can
    join with ``/terminal/sentiment_trend`` separately.
    """
    rows = []
    for point in detail.get("history") or []:
        if isinstance(point, list) and len(point) >= 2:
            rows.append(
                {
                    "date": point[0],
                    "price": point[1],
                    "volume": point[2] if len(point) >= 3 else None,
                    "sentiment": "",
                }
            )
    if not rows:
        return "date,price,volume,sentiment\n"
    df = pd.DataFrame(rows, columns=["date", "price", "volume", "sentiment"])
    return df.to_csv(index=False)


# --- endpoints ---------------------------------------------------------------


@router.get(
    "/markets",
    response_model=ResolvedMarketsListResponse,
    summary="Paginated list of resolved Polymarket markets in a date range.",
)
def list_resolved_markets(
    start: str | None = Query(
        None,
        description="Lower bound on resolution end-date (ISO YYYY-MM-DD). Defaults to 1 year ago.",
    ),
    end: str | None = Query(
        None,
        description="Upper bound on resolution end-date (ISO YYYY-MM-DD). Defaults to today.",
    ),
    theme: str | None = Query(
        None, description="Optional theme filter, e.g. ``politics``, ``crypto``, ``sports``."
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ResolvedMarketsListResponse:
    """List resolved markets with paging + optional theme filter."""
    end_date = _parse_date(end, label="end") if end else date.today()
    start_date = _parse_date(start, label="start") if start else (end_date - timedelta(days=365))
    if start_date > end_date:
        raise HTTPException(status_code=422, detail="start must be <= end")

    rows = fetch_resolved_markets(
        start_date=start_date,
        end_date=end_date,
        theme=theme,
        limit=limit,
        offset=offset,
    )
    return ResolvedMarketsListResponse(
        n_markets=len(rows),
        limit=limit,
        offset=offset,
        markets=[ResolvedMarketSummary(**r) for r in rows],
    )


@router.get(
    "/market/{slug}",
    summary="Full archive detail (history + stats) for one resolved market.",
)
def get_archive_market(
    slug: str = Path(..., min_length=1, max_length=200),
    format: Literal["json", "csv"] = Query(
        "json", description="``json`` (default) or ``csv`` for the price-history table."
    ),
) -> Response:
    """Return market detail. ``?format=csv`` streams the daily-history table."""
    try:
        detail = fetch_archive_market_detail(slug)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if format == "csv":
        csv_text = _detail_to_csv(detail)
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{slug}-history.csv"',
            },
        )

    # FastAPI will validate via the response_model only when one is set.
    # We construct the model explicitly so that the JSON serialization
    # matches the schema even when ``format=json`` is the default.
    payload = ArchiveMarketDetail(**detail)
    return Response(
        content=payload.model_dump_json(),
        media_type="application/json",
    )


@router.get(
    "/themes",
    response_model=ThemesDistributionResponse,
    summary="Aggregate stats per theme across the most recent resolved markets.",
)
def get_themes_distribution() -> ThemesDistributionResponse:
    raw = archive_themes_distribution()
    return ThemesDistributionResponse(
        n_markets_total=raw["n_markets_total"],
        themes=[ThemeRow(**t) for t in raw["themes"]],
    )


@router.get(
    "/resolutions/{slug}",
    response_model=ResolutionRecord,
    summary="Resolution outcome only (no price history).",
)
def get_resolution_record(
    slug: str = Path(..., min_length=1, max_length=200),
) -> ResolutionRecord:
    try:
        record = get_resolution(slug)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResolutionRecord(**record)


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Substring search over resolved-market slug + question.",
)
def search_resolved_markets(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(25, ge=1, le=100),
) -> SearchResponse:
    rows = search_archive(q, limit=limit)
    return SearchResponse(
        q=q,
        n_results=len(rows),
        results=[ResolvedMarketSummary(**r) for r in rows],
    )


@router.post(
    "/export-bulk",
    summary="Bulk-export N archive markets as a ZIP of per-slug files.",
)
def export_bulk(req: BulkExportRequest) -> Response:
    """Build a ZIP with one file per slug in the requested ``format``.

    ``parquet`` falls back to ``csv`` if pyarrow / fastparquet aren't
    available — pandas' ``to_parquet`` raises a clear error which we
    translate into a 501 so the frontend can warn the user. We dedupe slugs
    and tolerate per-slug 404s by writing an ``ERROR-<slug>.txt`` entry
    instead of failing the whole bundle.
    """
    seen: dict[str, None] = {}
    for s in req.slugs:
        seen.setdefault(s, None)
    slugs = list(seen.keys())

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for slug in slugs:
            try:
                detail = fetch_archive_market_detail(slug)
            except LookupError as exc:
                zf.writestr(f"ERROR-{slug}.txt", f"not found: {exc}")
                continue
            except Exception as exc:
                logger.warning("archive bulk export failed for %s: %s", slug, exc)
                zf.writestr(f"ERROR-{slug}.txt", f"upstream error: {exc}")
                continue

            if req.format == "json":
                zf.writestr(f"{slug}.json", json.dumps(detail, default=str, indent=2))
                continue

            # Materialize the history into a DataFrame for csv / parquet.
            rows = []
            for point in detail.get("history") or []:
                if isinstance(point, list) and len(point) >= 2:
                    rows.append(
                        {
                            "date": point[0],
                            "price": point[1],
                            "volume": point[2] if len(point) >= 3 else None,
                        }
                    )
            df = pd.DataFrame(rows, columns=["date", "price", "volume"])

            if req.format == "csv":
                zf.writestr(f"{slug}.csv", df.to_csv(index=False))
            elif req.format == "parquet":
                pq_buf = io.BytesIO()
                try:
                    df.to_parquet(pq_buf, index=False)
                except (ImportError, ValueError) as exc:
                    logger.info(
                        "parquet unavailable, falling back to csv for %s: %s",
                        slug,
                        exc,
                    )
                    zf.writestr(f"{slug}.csv", df.to_csv(index=False))
                else:
                    zf.writestr(f"{slug}.parquet", pq_buf.getvalue())

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="polymarket-archive-bulk.zip"',
        },
    )


# ──────────────────────────────────────────────────────────────────────────
# Alias router under the bare /archive prefix
# ──────────────────────────────────────────────────────────────────────────
#
# UX audit (2026-05-14): the footer "Archive" pill calls ``/archive/list``
# but the canonical endpoint is ``/archive/polymarket/markets``. Surface a
# thin alias so the pill works without touching the front-end. Mounted by
# ``main.py`` alongside the primary router.

alias_router = APIRouter(prefix="/archive", tags=["archive-polymarket"])


@alias_router.get(
    "/list",
    response_model=ResolvedMarketsListResponse,
    summary="Alias of /archive/polymarket/markets (footer pill).",
)
def list_archive_alias(
    start: str | None = Query(None),
    end: str | None = Query(None),
    theme: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ResolvedMarketsListResponse:
    """Footer-pill friendly alias — delegates to ``list_resolved_markets``."""
    return list_resolved_markets(
        start=start,
        end=end,
        theme=theme,
        limit=limit,
        offset=offset,
    )


__all__ = ["alias_router", "router"]
