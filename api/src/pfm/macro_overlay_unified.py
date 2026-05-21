"""Unified macro-overlay endpoint.

Lightweight, multi-series fetcher meant to back front-end charts that
need to overlay several macro series on the same time axis. This is
distinct from :mod:`pfm.terminal_macro_overlay` (which is *slug-driven*
and computes correlation / lag stats vs a Polymarket factor).

    GET /macro/overlay?series=DFF,DGS10,VIXCLS&start=2020-01-01&end=2026-05-08

Response::

    {
        "start": "2020-01-01",
        "end":   "2026-05-08",
        "count": 3,
        "series": [
            {"id": "DFF", "name": "...", "units": "Percent",
             "dates": [...], "values": [...]},
            ...
        ]
    }

NaNs are emitted as ``null``. All series are returned on the same daily
UTC calendar so the frontend can plot them on a shared x-axis without
re-aligning.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from pfm.sources.fred import (
    _SERIES_REGISTRY as FRED_REGISTRY,
)
from pfm.sources.fred import (
    FredDataError,
    fetch_fred_series_cached,
)

logger = logging.getLogger(__name__)


class MacroOverlaySeries(BaseModel):
    """A single series in the overlay response."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    units: str
    frequency: str
    dates: list[str] = Field(default_factory=list)
    values: list[float | None] = Field(default_factory=list)


class MacroOverlayResponse(BaseModel):
    """Top-level overlay response."""

    model_config = ConfigDict(extra="forbid")

    start: str
    end: str
    count: int
    series: list[MacroOverlaySeries]


router = APIRouter(prefix="/macro", tags=["macro-overlay"])


def _parse_series_param(raw: str) -> list[str]:
    """Split comma-separated series ids, trim whitespace, drop blanks."""
    parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
    return parts


@router.get("/overlay", response_model=MacroOverlayResponse)
def macro_overlay(
    series: str = Query(
        ..., description="Comma-separated FRED series ids, e.g. 'DFF,DGS10,VIXCLS'"
    ),
    start: str = Query(..., description="ISO date YYYY-MM-DD"),
    end: str = Query(..., description="ISO date YYYY-MM-DD"),
) -> MacroOverlayResponse:
    """Fetch multiple FRED series aligned on a daily UTC calendar.

    All requested series must be in :data:`pfm.sources.fred._SERIES_REGISTRY`.
    A 404 is raised if any are unknown — we'd rather fail loud than
    silently drop a series the user asked for.
    """
    sids = _parse_series_param(series)
    if not sids:
        raise HTTPException(status_code=400, detail="series query param is empty")
    unknown = [s for s in sids if s not in FRED_REGISTRY]
    if unknown:
        raise HTTPException(
            status_code=404,
            detail=f"unknown FRED series: {', '.join(unknown)}",
        )

    try:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"bad date: {e}") from e
    if start_ts >= end_ts:
        raise HTTPException(status_code=400, detail="start must be < end")

    out_series: list[MacroOverlaySeries] = []
    for sid in sids:
        meta = FRED_REGISTRY[sid]
        try:
            s = fetch_fred_series_cached(sid, start_ts, end_ts)
        except FredDataError as e:
            raise HTTPException(
                status_code=502,
                detail=f"fred fetch failed for {sid}: {e}",
            ) from e
        dates: list[str] = []
        values: list[float | None] = []
        for ts, v in s.items():
            dates.append(ts.date().isoformat())
            values.append(None if pd.isna(v) else float(v))
        out_series.append(
            MacroOverlaySeries(
                id=sid,
                name=meta["name"],
                units=meta["units"],
                frequency=meta["frequency"],
                dates=dates,
                values=values,
            )
        )

    return MacroOverlayResponse(
        start=start_ts.date().isoformat(),
        end=end_ts.date().isoformat(),
        count=len(out_series),
        series=out_series,
    )


__all__ = [
    "MacroOverlayResponse",
    "MacroOverlaySeries",
    "macro_overlay",
    "router",
]


def _coerce_value(v: Any) -> float | None:
    """Module-level utility — coerce JSON-safe float (None for NaN)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f
