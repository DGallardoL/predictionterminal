"""Counterfactual backtest — what would the price path have looked like
if a (now-resolved) prediction-market factor had resolved the *other* way?

Conceptually we run a single-factor attribution decomposition over a
window ``[start, end]`` and then either:

  - subtract the factor-attributable component from the realised
    return path (giving the "without this factor" path), or
  - add a *flipped* counterfactual factor path (the negative of the
    realised Δlogit series) on top of the realised return-path
    decomposition.

Both views share a common engine in :func:`counterfactual_path`. For
multi-factor waterfalls, :func:`attribution_decomposition` returns each
factor's cumulative contribution to the total period return so the UI
can render a Bloomberg-style attribution bar.

The math is intentionally simple:

    r_{t,counter} = r_{t,actual} - β · (Δlogit_actual_t - Δlogit_counter_t)

For ``scenario`` opposite to the realised one, we set
``Δlogit_counter_t = -Δlogit_actual_t`` (i.e. price the contract walked
the *opposite* path with the same magnitude). For ``scenario`` matching
realised, the counterfactual coincides with the actual path and the
attribution is informational only.

This module mocks two ingredients to keep the POC self-contained:

  * **Beta** — supplied via ``betas_override`` in tests; in production
    you'd plug in the result of ``fit_ols_hac`` over the same window.
  * **Realised series** — we accept dataframes/series directly; the API
    layer reads them from cache before calling.

Endpoints (mounted via the module's ``router``):

  - ``POST /counterfactual``           single-factor what-if
  - ``POST /counterfactual/multi``     multi-factor waterfall
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)

_CF_CACHE = get_cache("counterfactual", ttl=900)

Scenario = Literal["YES", "NO"]


# ---------------------------------------------------------------------------
# Synthetic data store (POC — replace with real fetchers in v0.2)
# ---------------------------------------------------------------------------
# We carry a compact synthetic dataset so the demo never 500s when run
# offline. Tests inject their own series via the function-level kwargs.


def _synthetic_returns(ticker: str, start: date, end: date) -> pd.Series:
    rng = np.random.default_rng(seed=hash((ticker.upper(), start.toordinal())) & 0xFFFF)
    idx = pd.date_range(start=start, end=end, freq="B")
    if len(idx) == 0:
        idx = pd.date_range(start=start, periods=20, freq="B")
    rets = rng.normal(0.0005, 0.012, size=len(idx))
    return pd.Series(rets, index=idx, name=f"{ticker.upper()}_logret")


def _synthetic_dlogit(factor_id: str, start: date, end: date) -> pd.Series:
    rng = np.random.default_rng(seed=hash((factor_id, end.toordinal())) & 0xFFFF)
    idx = pd.date_range(start=start, end=end, freq="B")
    if len(idx) == 0:
        idx = pd.date_range(start=start, periods=20, freq="B")
    dlog = rng.normal(0.0, 0.18, size=len(idx))
    # Inject a structural drift toward the realised resolution.
    drift = np.linspace(0.0, 0.4, len(idx))
    return pd.Series(dlog + drift, index=idx, name=f"{factor_id}_dlogit")


def _ticker_price_path(returns: pd.Series, base_price: float = 100.0) -> pd.Series:
    """Cumulate log-returns into a price path starting at ``base_price``."""
    cum = returns.cumsum()
    return pd.Series(base_price * np.exp(cum.values), index=returns.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def counterfactual_path(
    ticker: str,
    factor_id: str,
    scenario: Scenario,
    start: date,
    end: date,
    *,
    actual_resolution: Scenario = "YES",
    beta: float | None = None,
    returns: pd.Series | None = None,
    dlogit: pd.Series | None = None,
    base_price: float = 100.0,
) -> dict[str, Any]:
    """Construct the counterfactual price path for one ticker / one factor.

    Args:
        ticker: Equity ticker (purely a label).
        factor_id: Factor slug (purely a label / cache key).
        scenario: What-if resolution. ``YES`` or ``NO``.
        start, end: Inclusive date window.
        actual_resolution: How the factor *did* resolve. If
            ``scenario == actual_resolution`` the counterfactual = actual.
        beta: Pre-fit beta of ticker on Δlogit_factor. Required unless
            ``returns`` and ``dlogit`` are passed and tests want to
            short-circuit.
        returns: Optional pre-loaded daily log-return series for the
            ticker. If ``None`` a synthetic series is generated (POC).
        dlogit: Optional pre-loaded daily Δlogit series. Same fallback.
        base_price: Starting price for both paths.

    Returns:
        dict with keys ``ticker, factor_id, scenario, actual_resolution,
        beta, actual_path, counterfactual_path, attribution_pct,
        attributable_return_total, n_obs``.
    """
    rets = returns if returns is not None else _synthetic_returns(ticker, start, end)
    dlog = dlogit if dlogit is not None else _synthetic_dlogit(factor_id, start, end)

    # Align on the intersection of indices.
    df = pd.concat({"r": rets, "d": dlog}, axis=1).dropna()
    if df.empty:
        raise ValueError("no overlapping observations between returns and Δlogit")

    if beta is None:
        # Closed-form OLS slope (no intercept correction — fine for a POC).
        x = np.asarray(df["d"].to_numpy(), dtype=float)
        y = np.asarray(df["r"].to_numpy(), dtype=float)
        denom = float(np.dot(x - x.mean(), x - x.mean()))
        beta = float(np.dot(x - x.mean(), y - y.mean()) / denom) if denom > 0 else 0.0

    actual_dlogit = df["d"]
    if scenario == actual_resolution:
        counter_dlogit = actual_dlogit  # same world
    else:
        # Flip the path — the contract walks the mirror trajectory.
        counter_dlogit = -actual_dlogit

    # Δreturn = β · (Δlogit_counter − Δlogit_actual)
    delta_r = beta * (counter_dlogit - actual_dlogit)
    counter_r = df["r"] + delta_r

    actual_prices = _ticker_price_path(df["r"], base_price=base_price)
    counter_prices = _ticker_price_path(counter_r, base_price=base_price)

    total_return_actual = float(np.exp(df["r"].sum()) - 1.0)
    total_return_counter = float(np.exp(counter_r.sum()) - 1.0)
    attributable = float(np.exp(beta * actual_dlogit.sum()) - 1.0)

    return {
        "ticker": ticker.upper(),
        "factor_id": factor_id,
        "scenario": scenario,
        "actual_resolution": actual_resolution,
        "beta": round(beta, 6),
        "n_obs": int(len(df)),
        "actual_path": [
            {"date": pd.Timestamp(ts).date().isoformat(), "price": float(round(p, 4))}
            for ts, p in actual_prices.items()
        ],
        "counterfactual_path": [
            {"date": pd.Timestamp(ts).date().isoformat(), "price": float(round(p, 4))}
            for ts, p in counter_prices.items()
        ],
        "total_return_actual_pct": round(total_return_actual * 100.0, 4),
        "total_return_counterfactual_pct": round(total_return_counter * 100.0, 4),
        "attributable_return_total_pct": round(attributable * 100.0, 4),
        "attribution_pct": round(
            (attributable / total_return_actual * 100.0)
            if abs(total_return_actual) > 1e-9
            else 0.0,
            4,
        ),
    }


def attribution_decomposition(
    ticker: str,
    factors_list: list[str],
    start: date,
    end: date,
    *,
    betas: dict[str, float] | None = None,
    returns: pd.Series | None = None,
    dlogits: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    """Multi-factor attribution waterfall over a fixed window.

    Decomposes the total period return into
        ``Σ_i β_i · Σ_t Δlogit_{i,t} + residual``
    and reports each summand as a row of the waterfall.

    Args:
        ticker: Label only.
        factors_list: Factor IDs to include.
        start, end: Inclusive window.
        betas: Optional pre-fit ``factor_id -> β`` map. Synthesised if
            missing.
        returns: Optional return series. Synthesised if missing.
        dlogits: Optional ``factor_id -> Δlogit_series`` map. Synthesised
            per factor if missing.

    Returns:
        dict with ``ticker, start, end, total_return_pct, residual_pct,
        rows: [{factor_id, beta, sum_dlogit, contribution_pct,
        contribution_share}]``.
    """
    if not factors_list:
        raise ValueError("factors_list must contain at least one factor")

    rets = returns if returns is not None else _synthetic_returns(ticker, start, end)

    rows: list[dict[str, Any]] = []
    total_attributable_log = 0.0
    for fid in factors_list:
        d = (dlogits or {}).get(fid)
        if d is None:
            d = _synthetic_dlogit(fid, start, end)
        df = pd.concat({"r": rets, "d": d}, axis=1).dropna()
        if df.empty:
            continue
        if betas and fid in betas:
            beta = float(betas[fid])
        else:
            x = np.asarray(df["d"].to_numpy(), dtype=float)
            y = np.asarray(df["r"].to_numpy(), dtype=float)
            denom = float(np.dot(x - x.mean(), x - x.mean()))
            beta = float(np.dot(x - x.mean(), y - y.mean()) / denom) if denom > 0 else 0.0
        sum_d = float(df["d"].sum())
        contrib_log = beta * sum_d
        total_attributable_log += contrib_log
        rows.append(
            {
                "factor_id": fid,
                "beta": round(beta, 6),
                "sum_dlogit": round(sum_d, 6),
                "contribution_pct": round((np.exp(contrib_log) - 1.0) * 100.0, 4),
                "contribution_log_return": round(contrib_log, 6),
            }
        )

    total_log = float(rets.sum())
    residual_log = total_log - total_attributable_log
    total_pct = float((np.exp(total_log) - 1.0) * 100.0)

    # Compute share over absolute contributions so a +5%/-3% split sums to 100%.
    abs_total = sum(abs(r["contribution_log_return"]) for r in rows) or 1.0
    for r in rows:
        r["contribution_share"] = round(abs(r["contribution_log_return"]) / abs_total, 6)

    return {
        "ticker": ticker.upper(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_factors": len(rows),
        "total_return_pct": round(total_pct, 4),
        "residual_pct": round((np.exp(residual_log) - 1.0) * 100.0, 4),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CounterfactualRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)
    factor_id: str = Field(..., min_length=1)
    scenario: Scenario
    start: date
    end: date
    actual_resolution: Scenario = "YES"
    beta: float | None = None


class PathPoint(BaseModel):
    date: str
    price: float


class CounterfactualResponse(BaseModel):
    ticker: str
    factor_id: str
    scenario: Scenario
    actual_resolution: Scenario
    beta: float
    n_obs: int = Field(..., ge=0)
    actual_path: list[PathPoint]
    counterfactual_path: list[PathPoint]
    total_return_actual_pct: float
    total_return_counterfactual_pct: float
    attributable_return_total_pct: float
    attribution_pct: float


class MultiAttributionRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)
    factors_list: list[str] = Field(..., min_length=1)
    start: date
    end: date
    betas: dict[str, float] | None = None


class AttributionRow(BaseModel):
    factor_id: str
    beta: float
    sum_dlogit: float
    contribution_pct: float
    contribution_log_return: float
    contribution_share: float = Field(..., ge=0.0, le=1.0)


class MultiAttributionResponse(BaseModel):
    ticker: str
    start: str
    end: str
    n_factors: int = Field(..., ge=0)
    total_return_pct: float
    residual_pct: float
    rows: list[AttributionRow]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/counterfactual", tags=["counterfactual"])


@router.post("", response_model=CounterfactualResponse)
def post_counterfactual(req: CounterfactualRequest) -> CounterfactualResponse:
    if req.end < req.start:
        raise HTTPException(status_code=400, detail="end must be >= start")

    cache_key = (
        "single",
        req.ticker.upper(),
        req.factor_id,
        req.scenario,
        req.actual_resolution,
        req.start.isoformat(),
        req.end.isoformat(),
        req.beta,
    )
    cached = _CF_CACHE.get(cache_key)
    if cached is not None:
        return CounterfactualResponse(**cached)

    try:
        payload = counterfactual_path(
            ticker=req.ticker,
            factor_id=req.factor_id,
            scenario=req.scenario,
            start=req.start,
            end=req.end,
            actual_resolution=req.actual_resolution,
            beta=req.beta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _CF_CACHE.set(cache_key, payload, ttl=900)
    return CounterfactualResponse(**payload)


@router.post("/multi", response_model=MultiAttributionResponse)
def post_multi(req: MultiAttributionRequest) -> MultiAttributionResponse:
    if req.end < req.start:
        raise HTTPException(status_code=400, detail="end must be >= start")

    cache_key = (
        "multi",
        req.ticker.upper(),
        tuple(req.factors_list),
        req.start.isoformat(),
        req.end.isoformat(),
        tuple(sorted((req.betas or {}).items())),
    )
    cached = _CF_CACHE.get(cache_key)
    if cached is not None:
        return MultiAttributionResponse(**cached)

    try:
        payload = attribution_decomposition(
            ticker=req.ticker,
            factors_list=req.factors_list,
            start=req.start,
            end=req.end,
            betas=req.betas,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _CF_CACHE.set(cache_key, payload, ttl=900)
    return MultiAttributionResponse(**payload)


__all__ = [
    "AttributionRow",
    "CounterfactualRequest",
    "CounterfactualResponse",
    "MultiAttributionRequest",
    "MultiAttributionResponse",
    "PathPoint",
    "attribution_decomposition",
    "counterfactual_path",
    "router",
]
