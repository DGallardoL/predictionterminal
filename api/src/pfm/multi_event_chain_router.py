"""FastAPI router for the cross-asset multi-event-chain endpoints.

Five POST endpoints under ``/multi-event``:

- ``/lasso``                — :func:`fit_multi_event_lasso`
- ``/sector-attribution``   — :func:`sector_attribution`
- ``/chains``               — :func:`find_chains`
- ``/macro-correlation``    — :func:`event_macro_correlation`
- ``/systemic-factor``      — :func:`extract_systemic_pm_factor`

All compute paths are wrapped in a 600s :class:`TerminalCache` to avoid
re-fitting on dashboard refresh. The router takes no shared state — it
hydrates fetchers per-request from ``request.app.state``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.multi_event_chain import (
    DEFAULT_SECTOR_ETFS,
    event_macro_correlation,
    extract_systemic_pm_factor,
    find_chains,
    fit_multi_event_lasso,
    sector_attribution,
)

logger = logging.getLogger(__name__)


CACHE_TTL_SECONDS: int = 600
NAMESPACE = "multi_event_chain"


# ---------------------------------------------------------------------------
# Pydantic v2 schemas
# ---------------------------------------------------------------------------


class _LassoRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    factor_ids: list[str] = Field(..., min_length=1, max_length=200)
    start: str = Field(..., description="ISO date or datetime")
    end: str = Field(..., description="ISO date or datetime")
    alpha: float = Field(0.01, gt=0.0, le=10.0)


class _SectorAttributionRequest(BaseModel):
    sectors_etfs: list[str] | None = Field(None, max_length=50)
    factor_ids: list[str] = Field(..., min_length=1, max_length=200)
    start: str
    end: str


class _ChainsRequest(BaseModel):
    start_factor: str = Field(..., min_length=1)
    end_ticker: str = Field(..., min_length=1, max_length=20)
    candidate_intermediate_factors: list[str] = Field(..., max_length=100)
    max_depth: int = Field(3, ge=1, le=5)
    start: str
    end: str


class _MacroCorrelationRequest(BaseModel):
    factor_id: str = Field(..., min_length=1)
    macro_series: list[str] = Field(..., min_length=1, max_length=20)
    start: str
    end: str


class _SystemicFactorRequest(BaseModel):
    factor_ids: list[str] = Field(..., min_length=2, max_length=200)
    n_factors: int = Field(1, ge=1, le=10)
    start: str
    end: str


# ---------------------------------------------------------------------------
# Fetcher factories — wire production sources to the pure-function API
# ---------------------------------------------------------------------------


def _make_factor_fetcher(request: Request):
    """Return a FactorFetcher backed by Polymarket / Kalshi.

    Defers imports so unit tests of ``multi_event_chain`` don't need them.
    Accepts factor id, slug, *or* human-readable name via the unified
    resolver so users can paste any of the three.
    """
    from pfm.factor_resolver import resolve_factor
    from pfm.factors import CHAIN_SOURCE
    from pfm.sources.chain import fetch_chained_history
    from pfm.sources.kalshi import fetch_factor_history as fetch_kalshi_history
    from pfm.sources.polymarket import fetch_factor_history as fetch_poly_history

    poly = getattr(request.app.state, "poly", None)
    kalshi = getattr(request.app.state, "kalshi", None)
    factors_map: dict = getattr(request.app.state, "factors", {}) or {}

    def _fetch(factor_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        # Slug / name fallback so /multi-event/* don't reject a perfectly
        # valid slug just because the body field is named factor_ids.
        cfg = resolve_factor(factor_id, factors_map)
        if cfg is not None and getattr(cfg, "source", None) == CHAIN_SOURCE:
            df = fetch_chained_history(
                segments=cfg.segments,
                start=start,
                end=end,
                poly_client=poly,
                kalshi_client=kalshi,
                fetch_polymarket=lambda c, slug, start, end: fetch_poly_history(
                    c, slug, start=start, end=end
                ),
                fetch_kalshi=lambda c, market_ticker, start, end: fetch_kalshi_history(
                    c, market_ticker, start=start, end=end
                ),
            )
        elif cfg is not None and getattr(cfg, "source", "polymarket") == "kalshi":
            df = fetch_kalshi_history(kalshi, cfg.slug, start=start, end=end)
        else:
            slug = cfg.slug if cfg is not None else factor_id
            df = fetch_poly_history(poly, slug, start=start, end=end)
        if df is None or len(df) == 0:
            return pd.Series(dtype=float)
        # Normalise to a price Series with UTC daily index.
        if "price" in df.columns:
            s = df["price"].copy()
        else:
            # Fallback: take the first numeric column.
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return pd.Series(dtype=float)
            s = df[num_cols[0]].copy()
        s = s.astype(float)
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, utc=True)
        if s.index.tzinfo is None:
            s.index = s.index.tz_localize("UTC")
        s.index = s.index.normalize()
        return s

    return _fetch


def _make_returns_fetcher(_request: Request):
    """Return a ReturnFetcher backed by the equity adapter."""
    from pfm.sources.equity import get_log_returns

    def _fetch(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        s = get_log_returns(ticker, start, end, return_type="log")
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, utc=True)
        if s.index.tzinfo is None:
            s.index = s.index.tz_localize("UTC")
        s.index = s.index.normalize()
        return s.astype(float)

    return _fetch


def _make_macro_fetcher(_request: Request):
    """Return a MacroFetcher backed by FRED."""
    from pfm.sources.fred import fetch_fred_series

    def _fetch(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        s = fetch_fred_series(series_id, start, end, transform="raw")
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, utc=True)
        if s.index.tzinfo is None:
            s.index = s.index.tz_localize("UTC")
        s.index = s.index.normalize()
        return s.astype(float)

    return _fetch


# ---------------------------------------------------------------------------
# Cache key + dispatcher
# ---------------------------------------------------------------------------


def _cache_key(prefix: str, body: BaseModel) -> tuple:
    """Build a stable cache key from the request body."""
    try:
        return (prefix, repr(sorted(body.model_dump().items())))
    except Exception:
        return (prefix, repr(body.model_dump()))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/multi-event", tags=["multi-event"])


@router.post(
    "/lasso",
    summary="Fit LassoCV across N PM-factor Δlogits to predict ticker log returns.",
)
def post_lasso(
    body: Annotated[_LassoRequest, Body()],
    request: Request,
) -> dict[str, Any]:
    cache = get_cache(NAMESPACE, ttl=CACHE_TTL_SECONDS)
    key = _cache_key("lasso", body)
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        result = fit_multi_event_lasso(
            ticker=body.ticker,
            factor_ids=body.factor_ids,
            start=body.start,
            end=body.end,
            alpha=body.alpha,
            fetch_factor=_make_factor_fetcher(request),
            fetch_returns=_make_returns_fetcher(request),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("lasso fit failed")
        raise HTTPException(status_code=500, detail=f"lasso fit failed: {e}") from e
    cache.set(key, result)
    return result


@router.post(
    "/sector-attribution",
    summary="Per-sector OLS-HAC and variance attribution across PM factors.",
)
def post_sector_attribution(
    body: Annotated[_SectorAttributionRequest, Body()],
    request: Request,
) -> dict[str, Any]:
    cache = get_cache(NAMESPACE, ttl=CACHE_TTL_SECONDS)
    key = _cache_key("sector_attribution", body)
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        result = sector_attribution(
            sectors_etfs=body.sectors_etfs or DEFAULT_SECTOR_ETFS,
            factor_ids=body.factor_ids,
            start=body.start,
            end=body.end,
            fetch_factor=_make_factor_fetcher(request),
            fetch_returns=_make_returns_fetcher(request),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("sector attribution failed")
        raise HTTPException(status_code=500, detail=f"sector attribution failed: {e}") from e
    cache.set(key, result)
    return result


@router.post(
    "/chains",
    summary="Find Granger-significant chains start_factor -> ... -> ticker.",
)
def post_chains(
    body: Annotated[_ChainsRequest, Body()],
    request: Request,
) -> dict[str, Any]:
    cache = get_cache(NAMESPACE, ttl=CACHE_TTL_SECONDS)
    key = _cache_key("chains", body)
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        paths = find_chains(
            start_factor=body.start_factor,
            end_ticker=body.end_ticker,
            candidate_intermediate_factors=body.candidate_intermediate_factors,
            max_depth=body.max_depth,
            start=body.start,
            end=body.end,
            fetch_factor=_make_factor_fetcher(request),
            fetch_returns=_make_returns_fetcher(request),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("chain search failed")
        raise HTTPException(status_code=500, detail=f"chain search failed: {e}") from e
    payload = {
        "start_factor": body.start_factor,
        "end_ticker": body.end_ticker,
        "max_depth": body.max_depth,
        "n_paths": len(paths),
        "paths": paths,
    }
    cache.set(key, payload)
    return payload


@router.post(
    "/macro-correlation",
    summary="Δlogit(factor) vs Δ(macro) correlation, t-stat, and lead-lag.",
)
def post_macro_correlation(
    body: Annotated[_MacroCorrelationRequest, Body()],
    request: Request,
) -> dict[str, Any]:
    cache = get_cache(NAMESPACE, ttl=CACHE_TTL_SECONDS)
    key = _cache_key("macro_corr", body)
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        result = event_macro_correlation(
            factor_id=body.factor_id,
            macro_series=body.macro_series,
            start=body.start,
            end=body.end,
            fetch_factor=_make_factor_fetcher(request),
            fetch_macro=_make_macro_fetcher(request),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("macro correlation failed")
        raise HTTPException(status_code=500, detail=f"macro correlation failed: {e}") from e
    cache.set(key, result)
    return result


@router.post(
    "/systemic-factor",
    summary="Extract a PM-PCA systemic risk-on/off factor from N PM factors.",
)
def post_systemic_factor(
    body: Annotated[_SystemicFactorRequest, Body()],
    request: Request,
) -> dict[str, Any]:
    cache = get_cache(NAMESPACE, ttl=CACHE_TTL_SECONDS)
    key = _cache_key("systemic", body)
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        result = extract_systemic_pm_factor(
            factor_ids=body.factor_ids,
            n_factors=body.n_factors,
            start=body.start,
            end=body.end,
            fetch_factor=_make_factor_fetcher(request),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("systemic factor failed")
        raise HTTPException(status_code=500, detail=f"systemic factor failed: {e}") from e
    cache.set(key, result)
    return result


__all__ = ["CACHE_TTL_SECONDS", "NAMESPACE", "router"]
