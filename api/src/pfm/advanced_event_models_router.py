"""FastAPI router exposing advanced event-model fits.

Six endpoints, all POST + JSON in/out, mounted under
``/advanced-model``. Each endpoint:

  1.  Resolves ``factor_id`` against the in-process ``factors.yml``.
  2.  Fetches the prediction-market YES probability series.
  3.  Fetches the equity log-return (or price level for VECM) series.
  4.  Calls into the pure ``pfm.advanced_event_models`` core.
  5.  Returns the dict payload as JSON.

To activate, ``pfm.main`` only needs::

    from pfm.advanced_event_models_router import router as advanced_event_models_router
    app.include_router(advanced_event_models_router)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated, Any

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from pfm.advanced_event_models import (
    compute_tail_dependence_core,
    fit_conditional_model_core,
    fit_garch_x_core,
    fit_polynomial_factor_model_core,
    fit_regime_switching_model_core,
    fit_vecm_core,
)
from pfm.cache_utils import get_cache as _get_cache_ns
from pfm.equity_factors import EquityFactorError, fetch_equity_history
from pfm.factors import FactorConfig
from pfm.model import DEFAULT_EPSILON, delta_logit
from pfm.sources.polymarket import PolymarketClient, PolymarketError, fetch_factor_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/advanced-model", tags=["advanced-event-models"])

# Shared 600 s TTL cache. Hits are keyed by the JSON body so re-runs of the
# same ticker × factor × window are essentially free.
_CACHE_NS = "advanced_event_models"
_CACHE_TTL = 600


# ---------------------------------------------------------------------------
# Dependency wiring (matches the pattern in pfm.terminal_*).
# ---------------------------------------------------------------------------


def _get_polymarket_client() -> PolymarketClient:
    from pfm.main import app  # local import to avoid circulars

    return app.state.poly


def _get_factors_dep() -> dict[str, FactorConfig]:
    from pfm.main import app  # local import to avoid circulars

    return app.state.factors


# ---------------------------------------------------------------------------
# Request models — shared base.
# ---------------------------------------------------------------------------


class _BaseAdvancedRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    factor_id: str = Field(..., min_length=1, max_length=120)
    start: date
    end: date

    def assert_window(self) -> None:
        if self.start >= self.end:
            raise HTTPException(status_code=400, detail="start must be < end")


class ConditionalRequest(_BaseAdvancedRequest):
    conditioning_thresholds: list[float] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Cut points in (0, 1), strictly ascending.",
    )
    epsilon: float = Field(DEFAULT_EPSILON, gt=0.0, lt=0.5)


class PolynomialRequest(_BaseAdvancedRequest):
    degree: int = Field(2, ge=1, le=6)
    epsilon: float = Field(DEFAULT_EPSILON, gt=0.0, lt=0.5)


class RegimeSwitchingRequest(_BaseAdvancedRequest):
    n_regimes: int = Field(2, ge=2, le=4)
    epsilon: float = Field(DEFAULT_EPSILON, gt=0.0, lt=0.5)


class VecmRequest(_BaseAdvancedRequest):
    det_order: int = Field(0, ge=-1, le=1)
    k_ar_diff: int = Field(1, ge=1, le=5)
    epsilon: float = Field(DEFAULT_EPSILON, gt=0.0, lt=0.5)


class GarchXRequest(_BaseAdvancedRequest):
    epsilon: float = Field(DEFAULT_EPSILON, gt=0.0, lt=0.5)


class TailDependenceRequest(_BaseAdvancedRequest):
    quantile: float = Field(0.05, gt=0.0, lt=0.5)
    epsilon: float = Field(DEFAULT_EPSILON, gt=0.0, lt=0.5)


# ---------------------------------------------------------------------------
# Data assembly helpers
# ---------------------------------------------------------------------------


def _resolve_factor(factor_id: str, factors: dict[str, FactorConfig]) -> FactorConfig:
    """Resolve via id / slug / name, raising 400 with ``did_you_mean`` on miss."""
    from pfm.factor_resolver import resolve_or_404

    return resolve_or_404(factor_id, factors, status_code=400)


def _fetch_factor_probs(
    fc: FactorConfig, start: pd.Timestamp, end: pd.Timestamp, poly: PolymarketClient
) -> pd.Series:
    """Fetch the YES-probability series for a factor as a UTC-normalised Series."""
    if fc.source != "polymarket":
        # The advanced models work on any single-source factor in [0, 1]; we keep
        # the router conservative and only handle polymarket here.
        raise HTTPException(
            status_code=400,
            detail=f"factor {fc.id!r}: only source=polymarket is supported in advanced-model endpoints",
        )
    try:
        df = fetch_factor_history(poly, fc.slug, start=start, end=end)
    except PolymarketError as e:
        raise HTTPException(status_code=502, detail=f"polymarket error: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e
    if df is None or df.empty or "price" not in df.columns:
        raise HTTPException(
            status_code=502,
            detail=f"no probability history for slug={fc.slug!r}",
        )
    s = df["price"].astype(float)
    s.index = pd.to_datetime(s.index, utc=True).normalize()
    return s.rename(fc.id)


def _fetch_equity_returns(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Daily log returns of the adjusted close, UTC-normalised."""
    try:
        closes = fetch_equity_history(ticker, start=start, end=end)
    except EquityFactorError as e:
        raise HTTPException(status_code=502, detail=f"yfinance error: {e}") from e
    import numpy as np

    log_p = pd.Series(np.log(closes.values), index=closes.index, name=ticker)
    log_p.index = pd.to_datetime(log_p.index, utc=True).normalize()
    return log_p.diff().dropna().rename("r")


def _fetch_equity_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Daily adjusted close levels (used for VECM, which logs internally)."""
    try:
        closes = fetch_equity_history(ticker, start=start, end=end)
    except EquityFactorError as e:
        raise HTTPException(status_code=502, detail=f"yfinance error: {e}") from e
    closes.index = pd.to_datetime(closes.index, utc=True).normalize()
    return closes.rename(ticker)


def _cache_key(prefix: str, body: BaseModel) -> str:
    return f"{prefix}:{body.model_dump_json()}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/conditional")
def post_conditional(
    body: ConditionalRequest,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client)],
) -> dict[str, Any]:
    """Bucketed conditional model: separate (alpha, beta) per probability regime."""
    body.assert_window()
    cache = _get_cache_ns(_CACHE_NS, ttl=_CACHE_TTL)
    key = _cache_key("conditional", body)

    def _compute() -> dict[str, Any]:
        fc = _resolve_factor(body.factor_id, factors)
        start_ts = pd.Timestamp(body.start, tz="UTC")
        end_ts = pd.Timestamp(body.end, tz="UTC")
        probs = _fetch_factor_probs(fc, start_ts, end_ts, poly)
        rets = _fetch_equity_returns(body.ticker, start_ts, end_ts)
        try:
            return fit_conditional_model_core(
                rets,
                probs,
                conditioning_thresholds=body.conditioning_thresholds,
                epsilon=body.epsilon,
                ticker=body.ticker,
                factor_id=body.factor_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return cache.get_or_compute(key, _compute, ttl=_CACHE_TTL)


@router.post("/polynomial")
def post_polynomial(
    body: PolynomialRequest,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client)],
) -> dict[str, Any]:
    """Polynomial-in-Δlogit factor model with HAC SEs and an LR test vs linear."""
    body.assert_window()
    cache = _get_cache_ns(_CACHE_NS, ttl=_CACHE_TTL)
    key = _cache_key("polynomial", body)

    def _compute() -> dict[str, Any]:
        fc = _resolve_factor(body.factor_id, factors)
        start_ts = pd.Timestamp(body.start, tz="UTC")
        end_ts = pd.Timestamp(body.end, tz="UTC")
        probs = _fetch_factor_probs(fc, start_ts, end_ts, poly)
        rets = _fetch_equity_returns(body.ticker, start_ts, end_ts)
        try:
            return fit_polynomial_factor_model_core(
                rets,
                probs,
                degree=body.degree,
                epsilon=body.epsilon,
                ticker=body.ticker,
                factor_id=body.factor_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return cache.get_or_compute(key, _compute, ttl=_CACHE_TTL)


@router.post("/regime-switching")
def post_regime_switching(
    body: RegimeSwitchingRequest,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client)],
) -> dict[str, Any]:
    """Hamilton (1989) Markov-switching regression on r_t = alpha_s + beta_s · Δlogit_t."""
    body.assert_window()
    cache = _get_cache_ns(_CACHE_NS, ttl=_CACHE_TTL)
    key = _cache_key("regime", body)

    def _compute() -> dict[str, Any]:
        fc = _resolve_factor(body.factor_id, factors)
        start_ts = pd.Timestamp(body.start, tz="UTC")
        end_ts = pd.Timestamp(body.end, tz="UTC")
        probs = _fetch_factor_probs(fc, start_ts, end_ts, poly)
        rets = _fetch_equity_returns(body.ticker, start_ts, end_ts)
        try:
            return fit_regime_switching_model_core(
                rets,
                probs,
                n_regimes=body.n_regimes,
                epsilon=body.epsilon,
                ticker=body.ticker,
                factor_id=body.factor_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return cache.get_or_compute(key, _compute, ttl=_CACHE_TTL)


@router.post("/vecm")
def post_vecm(
    body: VecmRequest,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client)],
) -> dict[str, Any]:
    """Johansen + VECM(1) on (log P_equity, logit p)."""
    body.assert_window()
    cache = _get_cache_ns(_CACHE_NS, ttl=_CACHE_TTL)
    key = _cache_key("vecm", body)

    def _compute() -> dict[str, Any]:
        fc = _resolve_factor(body.factor_id, factors)
        start_ts = pd.Timestamp(body.start, tz="UTC")
        end_ts = pd.Timestamp(body.end, tz="UTC")
        probs = _fetch_factor_probs(fc, start_ts, end_ts, poly)
        prices = _fetch_equity_prices(body.ticker, start_ts, end_ts)
        try:
            return fit_vecm_core(
                prices,
                probs,
                det_order=body.det_order,
                k_ar_diff=body.k_ar_diff,
                epsilon=body.epsilon,
                ticker=body.ticker,
                factor_id=body.factor_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return cache.get_or_compute(key, _compute, ttl=_CACHE_TTL)


@router.post("/garch-x")
def post_garch_x(
    body: GarchXRequest,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client)],
) -> dict[str, Any]:
    """GARCH(1,1) augmented with |Δlogit| as exogenous variance regressor."""
    body.assert_window()
    cache = _get_cache_ns(_CACHE_NS, ttl=_CACHE_TTL)
    key = _cache_key("garchx", body)

    def _compute() -> dict[str, Any]:
        fc = _resolve_factor(body.factor_id, factors)
        start_ts = pd.Timestamp(body.start, tz="UTC")
        end_ts = pd.Timestamp(body.end, tz="UTC")
        probs = _fetch_factor_probs(fc, start_ts, end_ts, poly)
        rets = _fetch_equity_returns(body.ticker, start_ts, end_ts)
        try:
            return fit_garch_x_core(
                rets,
                probs,
                epsilon=body.epsilon,
                ticker=body.ticker,
                factor_id=body.factor_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return cache.get_or_compute(key, _compute, ttl=_CACHE_TTL)


@router.post("/tail-dependence")
def post_tail_dependence(
    body: TailDependenceRequest,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client)],
) -> dict[str, Any]:
    """Empirical lower- and upper-tail dependence between equity returns and Δlogit."""
    body.assert_window()
    cache = _get_cache_ns(_CACHE_NS, ttl=_CACHE_TTL)
    key = _cache_key("tail", body)

    def _compute() -> dict[str, Any]:
        fc = _resolve_factor(body.factor_id, factors)
        start_ts = pd.Timestamp(body.start, tz="UTC")
        end_ts = pd.Timestamp(body.end, tz="UTC")
        probs = _fetch_factor_probs(fc, start_ts, end_ts, poly)
        rets = _fetch_equity_returns(body.ticker, start_ts, end_ts)
        dlogit = delta_logit(probs, epsilon=body.epsilon).rename("dlogit")
        try:
            return compute_tail_dependence_core(
                rets,
                dlogit,
                quantile=body.quantile,
                ticker=body.ticker,
                factor_id=body.factor_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return cache.get_or_compute(key, _compute, ttl=_CACHE_TTL)


__all__ = ["router"]
