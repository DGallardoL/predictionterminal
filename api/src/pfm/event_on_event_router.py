"""FastAPI router for the event-on-event factor model.

Wires :mod:`pfm.event_on_event` to the live factor catalog. Each endpoint
fetches PM probability series via the same cached path as ``/fit``
(``main._cached_factor_history``) so warm caches and rate-limit safeguards
are shared, then calls into the pure module.

Routing
-------
This module owns its :class:`fastapi.APIRouter`. ``main.py`` is left
untouched per project convention — Damian wires it explicitly via::

    from pfm.event_on_event_router import router as event_on_event_router
    app.include_router(event_on_event_router)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.event_on_event import (
    event_correlation_matrix,
    event_lead_lag,
    event_pca_decomposition,
    event_vector_autoregression,
    fit_event_on_event,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/event-model", tags=["event-model"])

CACHE_TTL_SECONDS: int = 600
_cache = get_cache("event_on_event", ttl=CACHE_TTL_SECONDS)


# --- Pydantic schemas ------------------------------------------------------


class FitBody(BaseModel):
    """Body for ``POST /event-model/fit``."""

    target_factor_id: str = Field(min_length=1)
    predictor_factor_ids: list[str] = Field(min_length=1, max_length=20)
    start: date
    end: date
    return_type: Literal["delta_logit", "level"] = "delta_logit"
    epsilon: float = Field(0.01, gt=0.0, lt=0.5)


class CorrelationBody(BaseModel):
    """Body for ``POST /event-model/correlation-matrix``."""

    factor_ids: list[str] = Field(min_length=2, max_length=40)
    start: date
    end: date
    method: Literal["pearson", "spearman", "kendall"] = "pearson"
    on: Literal["delta_logit", "level"] = "delta_logit"
    epsilon: float = Field(0.01, gt=0.0, lt=0.5)


class LeadLagBody(BaseModel):
    """Body for ``POST /event-model/lead-lag``."""

    target_id: str = Field(min_length=1)
    predictor_id: str = Field(min_length=1)
    start: date
    end: date
    max_lag: int = Field(5, ge=1, le=30)
    epsilon: float = Field(0.01, gt=0.0, lt=0.5)


class VarBody(BaseModel):
    """Body for ``POST /event-model/var``."""

    factor_ids: list[str] = Field(min_length=2, max_length=10)
    start: date
    end: date
    lags: int = Field(5, ge=1, le=20)
    epsilon: float = Field(0.01, gt=0.0, lt=0.5)


class PcaBody(BaseModel):
    """Body for ``POST /event-model/pca``."""

    factor_ids: list[str] = Field(min_length=2, max_length=40)
    start: date
    end: date
    n_components: int = Field(5, ge=1, le=20)
    epsilon: float = Field(0.01, gt=0.0, lt=0.5)


# --- factor history fetcher (router-side) ----------------------------------


def _make_history_fetcher() -> Callable[..., pd.Series]:
    """Return a ``(factor_id, start, end) -> pd.Series`` backed by main's cache.

    Raises ``HTTPException`` with the unknown-factor / fetch-error reason
    if any single id can't be resolved. Routes id, slug, *and* name through
    :func:`pfm.factor_resolver.resolve_factor` so the ``factor_ids``
    parameter is forgiving across all three identifier shapes.
    """
    from pfm import main as main_mod  # local import to avoid cycles
    from pfm.factor_resolver import resolve_factor, suggest_factors_with_meta
    from pfm.factors import FactorConfig

    factors: dict[str, FactorConfig] = getattr(main_mod.app.state, "factors", {}) or {}
    poly = main_mod.app.state.poly
    cache = main_mod.app.state.cache
    settings = main_mod.get_settings()

    def _fetch(fid: str, start: date, end: date) -> pd.Series:
        fc = resolve_factor(fid, factors)
        if fc is None:
            # Last-ditch fallback for ad-hoc slugs that aren't in the catalog
            # yet (mirrors replay_mode / alpha_lab). The synthetic FactorConfig
            # only flies if the slug actually has Polymarket history; if not
            # the fetcher below 400s with did_you_mean.
            fc = FactorConfig(
                id=fid,
                name=fid,
                slug=fid,
                source="polymarket",
                description="event-model ad-hoc",
                theme="other",
            )
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        try:
            df = main_mod._cached_factor_history(fc, start_ts, end_ts, poly, cache, settings)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"failed to fetch history for {fid!r}: {e}",
                    "query": fid,
                    "did_you_mean": suggest_factors_with_meta(fid, factors, top_k=3),
                },
            ) from e
        if df is None or df.empty or "price" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"no history for factor {fid!r}",
                    "query": fid,
                    "did_you_mean": suggest_factors_with_meta(fid, factors, top_k=3),
                },
            )
        s = df["price"].dropna().astype(float)
        s.name = fid
        return s

    return _fetch


# --- caching helper --------------------------------------------------------


def _cache_get_or_compute(key: tuple, fn):
    """Look up ``key`` in the module-level cache; otherwise call ``fn``."""
    hit = _cache.get(key)
    if hit is not None:
        return hit
    result = fn()
    _cache.set(key, result, ttl=CACHE_TTL_SECONDS)
    return result


# --- endpoints -------------------------------------------------------------


@router.post("/fit")
def event_model_fit(body: FitBody) -> dict:
    """Fit Δlogit(target) ~ Σ β · Δlogit(predictors) with HAC SEs."""
    if body.start >= body.end:
        raise HTTPException(status_code=422, detail="start must be < end")

    key = (
        "fit",
        body.target_factor_id,
        tuple(body.predictor_factor_ids),
        body.start.isoformat(),
        body.end.isoformat(),
        body.return_type,
        round(body.epsilon, 6),
    )

    def _run() -> dict:
        try:
            return fit_event_on_event(
                target_factor_id=body.target_factor_id,
                predictor_factor_ids=body.predictor_factor_ids,
                start=body.start,
                end=body.end,
                return_type=body.return_type,
                epsilon=body.epsilon,
                fetch_history=_make_history_fetcher(),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return _cache_get_or_compute(key, _run)


@router.post("/correlation-matrix")
def event_model_correlation(body: CorrelationBody) -> dict:
    """Pairwise correlation matrix on Δlogit (or level) probability series."""
    if body.start >= body.end:
        raise HTTPException(status_code=422, detail="start must be < end")

    key = (
        "corr",
        tuple(body.factor_ids),
        body.start.isoformat(),
        body.end.isoformat(),
        body.method,
        body.on,
        round(body.epsilon, 6),
    )

    def _run() -> dict:
        try:
            return event_correlation_matrix(
                factor_ids=body.factor_ids,
                start=body.start,
                end=body.end,
                method=body.method,
                on=body.on,
                epsilon=body.epsilon,
                fetch_history=_make_history_fetcher(),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return _cache_get_or_compute(key, _run)


@router.post("/lead-lag")
def event_model_lead_lag(body: LeadLagBody) -> dict:
    """Cross-correlation function and Granger causality between two events."""
    if body.start >= body.end:
        raise HTTPException(status_code=422, detail="start must be < end")

    key = (
        "leadlag",
        body.target_id,
        body.predictor_id,
        body.start.isoformat(),
        body.end.isoformat(),
        body.max_lag,
        round(body.epsilon, 6),
    )

    def _run() -> dict:
        try:
            return event_lead_lag(
                target_id=body.target_id,
                predictor_id=body.predictor_id,
                start=body.start,
                end=body.end,
                max_lag=body.max_lag,
                epsilon=body.epsilon,
                fetch_history=_make_history_fetcher(),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return _cache_get_or_compute(key, _run)


@router.post("/var")
def event_model_var(body: VarBody) -> dict:
    """VAR(p) on a Δlogit panel of N events."""
    if body.start >= body.end:
        raise HTTPException(status_code=422, detail="start must be < end")

    key = (
        "var",
        tuple(body.factor_ids),
        body.start.isoformat(),
        body.end.isoformat(),
        body.lags,
        round(body.epsilon, 6),
    )

    def _run() -> dict:
        try:
            return event_vector_autoregression(
                factor_ids=body.factor_ids,
                start=body.start,
                end=body.end,
                lags=body.lags,
                epsilon=body.epsilon,
                fetch_history=_make_history_fetcher(),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return _cache_get_or_compute(key, _run)


@router.post("/pca")
def event_model_pca(body: PcaBody) -> dict:
    """PCA decomposition of Δlogit innovations across N events."""
    if body.start >= body.end:
        raise HTTPException(status_code=422, detail="start must be < end")

    key = (
        "pca",
        tuple(body.factor_ids),
        body.start.isoformat(),
        body.end.isoformat(),
        body.n_components,
        round(body.epsilon, 6),
    )

    def _run() -> dict:
        try:
            return event_pca_decomposition(
                factor_ids=body.factor_ids,
                start=body.start,
                end=body.end,
                n_components=body.n_components,
                epsilon=body.epsilon,
                fetch_history=_make_history_fetcher(),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return _cache_get_or_compute(key, _run)


__all__ = ["router"]
