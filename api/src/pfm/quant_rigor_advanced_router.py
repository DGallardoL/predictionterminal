"""FastAPI router for the advanced quant-rigor endpoints.

Three POST endpoints, all JSON-in / JSON-out:

* ``POST /quant/oos-r-squared`` — Campbell & Thompson (2008) out-of-sample
  R-squared with Clark-West (2007) nested-model correction.
* ``POST /quant/diebold-mariano`` — Diebold-Mariano (1995) test for equal
  forecast accuracy with the Harvey-Leybourne-Newbold (1997) finite-sample
  fix.
* ``POST /quant/whites-reality-check`` — White (2000) Reality Check plus
  Hansen's SPA, Romano-Wolf stepwise multiple testing.

The router is **not** auto-mounted; ``main.py`` imports and registers it::

    from pfm.quant_rigor_advanced_router import router as qr_router
    app.include_router(qr_router)
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pfm.cache_utils import cached
from pfm.forecast_comparison import diebold_mariano
from pfm.oos_metrics import oos_r_squared_campbell_thompson
from pfm.whites_reality_check import stepwise_spa, whites_reality_check

router = APIRouter(prefix="/quant", tags=["quant-rigor-advanced"])


# ---------------------------------------------------------------------------
# 1) Out-of-sample R^2 (Campbell-Thompson)
# ---------------------------------------------------------------------------


class OOSRSquaredRequest(BaseModel):
    """Body of ``POST /quant/oos-r-squared``."""

    y_actual: list[float] = Field(..., description="Realised target values.")
    y_pred_model: list[float] = Field(..., description="Candidate model forecasts.")
    y_pred_baseline: list[float] = Field(
        ...,
        description="Baseline forecasts (typically the recursive mean).",
    )
    nested: bool = Field(
        True,
        description="If True (default) use Clark-West nested-model correction.",
    )
    hac_lag: int | None = Field(
        None,
        ge=0,
        description="Newey-West truncation lag. Default: floor(T^(1/3)).",
    )


class OOSRSquaredResponse(BaseModel):
    r_squared_oos: float
    mse_model: float
    mse_baseline: float
    n_obs: int
    hac_t_stat_clark_west: float
    hac_p_value: float
    hac_lag: int
    model_beats_baseline: bool


@cached(namespace="quant_oos_r2", ttl=900)
def _oos_cached(
    y_actual: tuple[float, ...],
    y_pred_model: tuple[float, ...],
    y_pred_baseline: tuple[float, ...],
    nested: bool,
    hac_lag: int | None,
) -> dict[str, Any]:
    return oos_r_squared_campbell_thompson(
        list(y_actual),
        list(y_pred_model),
        list(y_pred_baseline),
        nested=nested,
        hac_lag=hac_lag,
    )


@router.post("/oos-r-squared", response_model=OOSRSquaredResponse)
def post_oos_r_squared(body: OOSRSquaredRequest) -> dict[str, Any]:
    """Compute the Campbell-Thompson R^2_OOS and Clark-West stat."""
    try:
        return _oos_cached(
            tuple(body.y_actual),
            tuple(body.y_pred_model),
            tuple(body.y_pred_baseline),
            body.nested,
            body.hac_lag,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# 2) Diebold-Mariano test
# ---------------------------------------------------------------------------


class DieboldMarianoRequest(BaseModel):
    """Body of ``POST /quant/diebold-mariano``."""

    forecast_errors_1: list[float] = Field(..., description="Errors of model 1.")
    forecast_errors_2: list[float] = Field(..., description="Errors of model 2.")
    h: int = Field(1, ge=1, description="Forecast horizon (default 1).")
    loss: Literal["MSE", "MAE", "Quad", "Abs"] = Field(
        "MSE",
        description="Loss function applied to errors before differencing.",
    )
    hac_lag: int | None = Field(
        None,
        ge=0,
        description="Newey-West truncation lag. Default: h - 1.",
    )


class DieboldMarianoResponse(BaseModel):
    dm_stat: float
    p_value: float
    dm_stat_hln: float
    p_value_hln: float
    mean_loss_diff: float
    prefer_model: int | str
    n_obs: int
    hac_lag: int
    loss: str
    h: int


@router.post("/diebold-mariano", response_model=DieboldMarianoResponse)
def post_diebold_mariano(body: DieboldMarianoRequest) -> dict[str, Any]:
    """Compare two forecast-error series via the Diebold-Mariano test."""
    try:
        return diebold_mariano(
            body.forecast_errors_1,
            body.forecast_errors_2,
            h=body.h,
            loss=body.loss,
            hac_lag=body.hac_lag,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# 3) White's Reality Check + Stepwise SPA
# ---------------------------------------------------------------------------


class WhitesRequest(BaseModel):
    """Body of ``POST /quant/whites-reality-check``."""

    strategy_returns_matrix: list[list[float]] = Field(
        ...,
        description="T-by-K matrix; row t holds the per-strategy returns at t.",
    )
    benchmark_returns: list[float] = Field(
        ...,
        description="Length-T benchmark return series; pass zeros for raw excess-vs-zero.",
    )
    n_bootstrap: int = Field(1000, ge=100, le=20000)
    block_size: float | None = Field(
        None,
        gt=0,
        description="Stationary-bootstrap block length. Default ~ T^(1/3).",
    )
    seed: int = Field(42)
    run_stepwise_spa: bool = Field(
        True,
        description="If True, also run Romano-Wolf stepwise multiple testing.",
    )
    alpha: float = Field(0.05, gt=0.0, lt=1.0)


class WhitesResponse(BaseModel):
    n_strategies: int
    n_obs: int
    best_strategy_idx: int
    best_excess_return: float
    test_statistic_v_t: float
    white_pvalue: float
    hansen_spa_pvalue: float
    n_strategies_significant_at_05: int
    block_size: float
    n_bootstrap: int
    stepwise_rejected_indices: list[int] | None = None
    stepwise_p_values: list[float] | None = None
    stepwise_n_rejected: int | None = None


@router.post("/whites-reality-check", response_model=WhitesResponse)
def post_whites_reality_check(body: WhitesRequest) -> dict[str, Any]:
    """Run White's RC + Hansen SPA + (optional) Romano-Wolf stepwise SPA."""
    try:
        rc = whites_reality_check(
            body.strategy_returns_matrix,
            body.benchmark_returns,
            n_bootstrap=body.n_bootstrap,
            block_size=body.block_size,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    out: dict[str, Any] = dict(rc)
    out["stepwise_rejected_indices"] = None
    out["stepwise_p_values"] = None
    out["stepwise_n_rejected"] = None

    if body.run_stepwise_spa:
        try:
            sw = stepwise_spa(
                body.strategy_returns_matrix,
                body.benchmark_returns,
                alpha=body.alpha,
                n_bootstrap=body.n_bootstrap,
                block_size=body.block_size,
                seed=body.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        out["stepwise_rejected_indices"] = sw["rejected_strategy_indices"]
        out["stepwise_p_values"] = sw["p_values_per_strategy"]
        out["stepwise_n_rejected"] = sw["n_rejected"]

    return out
