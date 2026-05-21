"""``POST /regression/elastic-net`` — Elastic Net regression endpoint (W12-13).

Exposes W11-57's :func:`pfm.quant.regression_methods.fit_elastic_net` over
HTTP. The handler builds the design matrix from one or more prediction-market
factors (Δlogit / Δlevel) and pulls equity log-returns for the target ticker,
then delegates the actual fit to the elastic-net solver.

This router is intentionally standalone — it can be mounted on the main
``pfm.main`` app via ``app.include_router(router)`` without touching the
``main.py:routes`` section, which is a hot-claim file in the multi-session
protocol. Tests in ``tests/test_elnet_endpoint.py`` mount the router on a
throw-away FastAPI app and patch the two upstream data calls
(``_cached_factor_history`` and ``get_log_returns``) so no network IO is
required.

Request schema::

    {
      "ticker": "NVDA",
      "factors": ["bitcoin", "trump-win"],
      "start":   "2024-01-01",
      "end":     null,
      "alpha":   "auto" | <float > 0>,
      "l1_ratio": <0 < float <= 1>,
      "cv_splits": <int >= 2>
    }

Response shape (mirrors :class:`ElasticNetResult` but renames a few fields
to the sklearn-style public API used elsewhere in the project)::

    {
      "ticker": "NVDA",
      "coefficients": {"bitcoin": 0.45, "trump-win": 0.02},
      "selected":     ["bitcoin"],
      "alpha":         0.31,
      "l1_ratio":      0.5,
      "n_iter":        1234,
      "mse_cv":        0.018,
      "r_squared_train": 0.42
    }

Naming note: the user-facing ``alpha`` is the overall regularisation strength
(sklearn calls it ``alpha``; W11-57's :func:`fit_elastic_net` calls it
``lambda_``). The user-facing ``l1_ratio`` is the L1 mixing fraction
(sklearn convention; W11-57 calls it ``alpha``). The translation happens at
the boundary of this router; downstream all the dataclass field names from
W11-57 are preserved.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from pfm.cache import CacheBackend
from pfm.config import Settings, get_settings
from pfm.dependencies import get_cache, get_factors_dep, get_polymarket_client
from pfm.factor_resolver import (
    resolve_factor as _resolve_factor_unified,
)
from pfm.factor_resolver import (
    suggest_factors_with_meta as _factor_suggest_meta,
)
from pfm.factors import FactorConfig
from pfm.model import DEFAULT_EPSILON, delta_level, delta_logit
from pfm.quant.regression_methods import ElasticNetResult, fit_elastic_net
from pfm.sources.polymarket import PolymarketClient

router = APIRouter(tags=["regression"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ElasticNetFitRequest(BaseModel):
    """Body schema for ``POST /regression/elastic-net``."""

    ticker: str = Field(min_length=1, description="Equity ticker, e.g. ``NVDA``.")
    factors: list[str] = Field(
        min_length=1,
        description="Factor ids / slugs from the catalog (at least one).",
    )
    start: date = Field(description="UTC start date (inclusive).")
    end: date | None = Field(
        default=None,
        description="UTC end date (inclusive). ``null`` → today.",
    )
    alpha: float | Literal["auto"] = Field(
        default="auto",
        description=(
            "Overall regularisation strength. Either the literal string "
            '``"auto"`` for CV selection, or a positive float.'
        ),
    )
    l1_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="L1 mixing fraction in ``(0, 1]``. 1.0 → pure LASSO.",
    )
    cv_splits: int = Field(
        default=5,
        ge=2,
        description="Number of ``TimeSeriesSplit`` folds (>=2).",
    )


class ElasticNetFitResponse(BaseModel):
    """Response shape for ``POST /regression/elastic-net``."""

    ticker: str
    coefficients: dict[str, float]
    selected: list[str]
    alpha: float
    l1_ratio: float
    n_iter: int
    mse_cv: float
    r_squared_train: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_factors(factor_ids: list[str], catalog: dict[str, FactorConfig]) -> list[FactorConfig]:
    """Resolve user-supplied ids/slugs/names against the loaded catalog.

    On miss, raises 404 with a structured ``did_you_mean`` hint so the
    caller can immediately fix the typo without paging through ``/factors``.
    """
    out: list[FactorConfig] = []
    seen: set[str] = set()
    unknown: list[dict[str, object]] = []
    for fid in factor_ids:
        fc = _resolve_factor_unified(fid, catalog)
        if fc is None:
            unknown.append(
                {
                    "query": fid,
                    "did_you_mean": _factor_suggest_meta(fid, catalog, top_k=3),
                }
            )
        elif fc.id not in seen:
            out.append(fc)
            seen.add(fc.id)
    if unknown:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"{len(unknown)} factor id(s) not found",
                "unknown": unknown,
            },
        )
    return out


def _build_design_matrix(
    factor_specs: list[FactorConfig],
    start: pd.Timestamp,
    end: pd.Timestamp,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.DataFrame:
    """Pull each factor's daily history, transform to a regressor column.

    Probability factors (sources: polymarket / kalshi / manifold / predictit /
    chain) are mapped through ``delta_logit``; level factors (BLS / FRED /
    sentiment / etc.) get ``delta_level``. The columns are concatenated and
    inner-joined on the date index.

    We import :func:`pfm.regression_core._cached_factor_history` lazily so
    test code can monkeypatch the symbol on ``pfm.regression_core`` (the
    real production cache wrapper goes through ``pfm.main`` indirection
    which test envs sometimes skip).
    """
    from pfm.regression_core import _cached_factor_history

    cols: dict[str, pd.Series] = {}
    for fc in factor_specs:
        df = _cached_factor_history(fc, start, end, poly, cache, settings)
        if df.empty:
            raise HTTPException(
                status_code=502,
                detail=(f"{fc.source} returned no history for factor {fc.id!r} (slug={fc.slug!r})"),
            )
        series = df["price"]
        series = series[(series.index >= start) & (series.index <= end)]
        if fc.is_probability:
            cols[fc.id] = delta_logit(series, epsilon=epsilon).rename(fc.id)
        else:
            cols[fc.id] = delta_level(series).rename(fc.id)
    return pd.concat(cols.values(), axis=1).dropna()


def _fetch_log_returns(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Wrap :func:`pfm.main.get_log_returns` so tests can monkeypatch it.

    The shim resolves ``get_log_returns`` through ``pfm.main`` at call time
    (not import time) so the conftest pattern of
    ``monkeypatch.setattr(pfm.main, "get_log_returns", fake)`` works
    transparently here too.
    """
    from pfm import main as _main

    return _main.get_log_returns(ticker, start, end, return_type="log")


def _safe_float(x: float) -> float:
    """Coerce NaN/Inf → 0.0 so the JSON response always validates."""
    if x is None:
        return 0.0
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return 0.0
    return float(x)


def _r_squared_train(result: ElasticNetResult, y: pd.Series, X: pd.DataFrame) -> float:
    """Compute in-sample R² of the fitted coefficients on (y, X).

    The :class:`ElasticNetResult` from W11-57 ships ``r_squared_cv`` (the
    CV-folded R²) but the HTTP contract for this endpoint also asks for an
    in-sample ``r_squared_train``. Computing it here keeps the dataclass
    unchanged.
    """
    joined = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if joined.empty:
        return 0.0
    y_arr = joined["__y__"].to_numpy(dtype=float)
    X_arr = joined.drop(columns="__y__").to_numpy(dtype=float)
    feat_names = list(result.coefficients)
    coef_vec = np.array([result.coefficients.get(name, 0.0) for name in feat_names], dtype=float)
    y_hat = X_arr @ coef_vec + result.intercept
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/regression/elastic-net",
    response_model=ElasticNetFitResponse,
    summary="Fit an Elastic Net regression of stock returns on factor changes",
)
def fit_elastic_net_endpoint(
    body: ElasticNetFitRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> ElasticNetFitResponse:
    """HTTP front for :func:`pfm.quant.regression_methods.fit_elastic_net`.

    Validation rules (enforced before any network IO):

    * ``alpha`` must be the literal string ``"auto"`` or a positive float.
      Pydantic accepts ``float`` for the union arm but allows zero/negative
      values — we re-check here so the user gets a clear 422 instead of
      sklearn's downstream ``ValueError``.
    * ``l1_ratio`` is enforced ``in (0, 1]`` by the Pydantic field.
    * ``cv_splits >= 2`` is enforced by the Pydantic field.
    * Unknown factor ids → 404 with ``did_you_mean`` hints.

    Returns the W11-57 dataclass fields renamed for the public API
    (see module docstring for the mapping).
    """

    # --- extra validation that Pydantic can't express on a Union arm ----
    if isinstance(body.alpha, (int, float)):
        if not float(body.alpha) > 0:
            raise HTTPException(
                status_code=422,
                detail="alpha must be > 0 (or the literal string 'auto')",
            )

    # --- date plumbing ---------------------------------------------------
    end_date = body.end or date.today()
    if body.start >= end_date:
        raise HTTPException(status_code=422, detail="start must be strictly before end")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")

    # --- resolve factors -------------------------------------------------
    factor_specs = _resolve_factors(body.factors, factors)

    # --- build (y, X) ----------------------------------------------------
    X = _build_design_matrix(
        factor_specs,
        start_ts,
        end_ts,
        poly,
        cache,
        settings,
    )
    y = _fetch_log_returns(body.ticker, start_ts, end_ts)

    common = X.index.intersection(y.index)
    y = y.loc[common]
    X = X.loc[common]

    # cv_splits requires at least 2*cv_splits non-NaN rows; the underlying
    # fit_elastic_net raises ValueError if not met. Re-surface as 422.
    if len(y) < 2 * body.cv_splits:
        raise HTTPException(
            status_code=422,
            detail=(
                f"only {len(y)} overlapping obs for ticker {body.ticker!r} "
                f"+ {len(factor_specs)} factor(s); need >= {2 * body.cv_splits} "
                f"for cv_splits={body.cv_splits}"
            ),
        )

    # --- call into W11-57 ------------------------------------------------
    # W11-57 maps:
    #   our public ``l1_ratio`` -> stub's ``alpha`` (L1 mixing fraction)
    #   our public ``alpha``    -> stub's ``lambda_`` (overall strength)
    try:
        result = fit_elastic_net(
            y,
            X,
            alpha=float(body.l1_ratio),
            lambda_=body.alpha if body.alpha == "auto" else float(body.alpha),
            cv_splits=body.cv_splits,
            random_state=0,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # --- translate to public response ------------------------------------
    # ``n_iter`` is not surfaced by the W11-57 dataclass; we expose the
    # length of the regularisation path as a proxy (each entry = one CV
    # tuple). This is non-zero for both auto-alpha and fixed-alpha runs.
    n_iter = len(result.regularisation_path) or 1

    # ``mse_cv`` is derivable from ``r_squared_cv`` and the y variance: the
    # CV R^2 is computed inside W11-57 as ``1 - cv_mse / var(y_centered)``,
    # so ``cv_mse = (1 - r2_cv) * var(y_centered)``. We recompute from the
    # cleaned (y, X) so it's an honest reflection of the fit.
    y_centered_var = float(np.var(y.dropna().to_numpy(dtype=float))) if len(y) else 0.0
    mse_cv = max(0.0, (1.0 - result.r_squared_cv) * y_centered_var)

    return ElasticNetFitResponse(
        ticker=body.ticker,
        coefficients={k: _safe_float(v) for k, v in result.coefficients.items()},
        selected=list(result.selected_factors),
        alpha=_safe_float(result.optimal_lambda),
        l1_ratio=_safe_float(result.optimal_alpha),
        n_iter=int(n_iter),
        mse_cv=_safe_float(mse_cv),
        r_squared_train=_safe_float(_r_squared_train(result, y, X)),
    )
