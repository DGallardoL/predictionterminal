"""FastAPI router for the asymmetric volatility-model endpoints.

Three POST endpoints under ``/vol``:

* ``POST /vol/gjr-garch``     — :func:`fit_gjr_garch_11`
* ``POST /vol/egarch``        — :func:`fit_egarch_11`
* ``POST /vol/garch-compare`` — :func:`compare_garch_models`

All three auto-fetch daily log-returns via
:func:`pfm.sources.equity.get_log_returns` (yfinance with cascaded
fallbacks). The responses are cached for 600s under the
``volatility_models`` namespace via :class:`pfm.cache_utils.TerminalCache`,
keyed on (endpoint, ticker, start, end, distribution / model list).

Routing
-------
This module owns its :class:`fastapi.APIRouter` but **is not** auto-
mounted. Wire it into ``main.py`` explicitly::

    from pfm.garch_router import router as garch_router
    app.include_router(garch_router)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.garch import compare_garch_models, fit_egarch_11, fit_gjr_garch_11

logger = logging.getLogger(__name__)


CACHE_TTL_SECONDS: int = 600
_NAMESPACE = "volatility_models"


router = APIRouter(prefix="/vol", tags=["volatility-models"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class _BaseVolRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    start: str = Field(..., description="ISO date (YYYY-MM-DD) or datetime")
    end: str = Field(..., description="ISO date (YYYY-MM-DD) or datetime")


class _GJRRequest(_BaseVolRequest):
    distribution: Literal["normal", "t", "skewed-t"] = "normal"


class _EGARCHRequest(_BaseVolRequest):
    pass


class _CompareRequest(_BaseVolRequest):
    models: list[Literal["garch11", "gjr11", "egarch11"]] = Field(
        default_factory=lambda: ["garch11", "gjr11", "egarch11"],
        max_length=3,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dates(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        s = pd.Timestamp(start, tz="UTC").normalize()
        e = pd.Timestamp(end, tz="UTC").normalize()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"bad date: {exc}") from exc
    if s >= e:
        raise HTTPException(status_code=422, detail="start must be < end")
    return s, e


def _fetch_returns(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    from pfm.sources.equity import EquityDataError, get_log_returns

    try:
        s = get_log_returns(ticker, start, end, return_type="log")
    except EquityDataError as exc:
        raise HTTPException(status_code=502, detail=f"equity source failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover — defensive
        raise HTTPException(
            status_code=502,
            detail=f"unexpected equity-fetch error: {exc}",
        ) from exc
    if s is None or len(s) == 0:
        raise HTTPException(status_code=404, detail=f"no returns available for {ticker!r}")
    return s.astype(float)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/gjr-garch")
def post_gjr_garch(body: _GJRRequest) -> dict[str, Any]:
    """Fit GJR-GARCH(1,1) to the ticker's daily log-returns."""
    start, end = _parse_dates(body.start, body.end)
    cache = get_cache(_NAMESPACE, ttl=CACHE_TTL_SECONDS)
    key = ("gjr", body.ticker.upper(), str(start), str(end), body.distribution)
    cached = cache.get(key)
    if cached is not None:
        return cached

    returns = _fetch_returns(body.ticker, start, end)
    try:
        out = fit_gjr_garch_11(returns, distribution=body.distribution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out["ticker"] = body.ticker.upper()
    out["start"] = str(start.date())
    out["end"] = str(end.date())
    cache.set(key, out, ttl=CACHE_TTL_SECONDS)
    return out


@router.post("/egarch")
def post_egarch(body: _EGARCHRequest) -> dict[str, Any]:
    """Fit EGARCH(1,1) to the ticker's daily log-returns."""
    start, end = _parse_dates(body.start, body.end)
    cache = get_cache(_NAMESPACE, ttl=CACHE_TTL_SECONDS)
    key = ("egarch", body.ticker.upper(), str(start), str(end))
    cached = cache.get(key)
    if cached is not None:
        return cached

    returns = _fetch_returns(body.ticker, start, end)
    try:
        out = fit_egarch_11(returns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out["ticker"] = body.ticker.upper()
    out["start"] = str(start.date())
    out["end"] = str(end.date())
    cache.set(key, out, ttl=CACHE_TTL_SECONDS)
    return out


@router.post("/garch-compare")
def post_garch_compare(body: _CompareRequest) -> dict[str, Any]:
    """Fit multiple GARCH-family models and pick the AIC / BIC winner."""
    if not body.models:
        raise HTTPException(status_code=422, detail="models list cannot be empty")
    start, end = _parse_dates(body.start, body.end)
    cache = get_cache(_NAMESPACE, ttl=CACHE_TTL_SECONDS)
    models_key = tuple(sorted(body.models))
    key = ("compare", body.ticker.upper(), str(start), str(end), models_key)
    cached = cache.get(key)
    if cached is not None:
        return cached

    returns = _fetch_returns(body.ticker, start, end)
    try:
        out = compare_garch_models(returns, models=list(body.models))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out["ticker"] = body.ticker.upper()
    out["start"] = str(start.date())
    out["end"] = str(end.date())
    cache.set(key, out, ttl=CACHE_TTL_SECONDS)
    return out


__all__ = ["router"]
