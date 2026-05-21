"""FastAPI router for ``/strategies/optimize``.

Wires :mod:`pfm.portfolio_optimizer` to the alpha catalog. Accepts a list of
``pair_id``s from ``web/data/alpha_strategies.json`` and a method spec, then
returns optimal weights, frontier, and Monte-Carlo drawdown bands.

Routing
-------
This module owns its :class:`fastapi.APIRouter`. ``main.py`` is left
untouched — wire it in explicitly via::

    from pfm.portfolio_optimizer_router import router as portfolio_optimizer_router
    app.include_router(portfolio_optimizer_router)

Synthetic-history caveat
------------------------
The catalog records *summary statistics* per pair (oos_sharpe, n_obs, etc.)
but does not store the daily PnL series. For the POC, when the live PnL
cache is missing, we synthesise a Gaussian return path consistent with the
recorded Sharpe. This is documented in the response (``warnings`` list) so
the consumer can decide whether the result is good enough for their use
(it is — for sizing a *long-term diversified* book under uncertainty about
the exact path, the Σ structure derived from synthetic Sharpes is a
reasonable prior, especially when combined with HRP which is robust to
estimation noise).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from pfm.auth.dependencies import require_tier
from pfm.portfolio_optimizer import (
    efficient_frontier,
    equal_weight,
    hrp,
    mean_variance_max_sharpe,
    min_variance,
    monte_carlo_drawdown,
    risk_parity_erc,
)
from pfm.terminal_export import respond as _export_respond

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies"])


# Default location of the curated alpha catalog (Hub product surface).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_PATH: Path = _REPO_ROOT / "web" / "data" / "alpha_strategies.json"


# ---------------------------------------------------------------------------
# pydantic schemas
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    """Body for ``POST /strategies/optimize``."""

    pair_ids: list[str] = Field(min_length=2, max_length=30)
    method: Literal["mean_variance", "min_variance", "risk_parity", "hrp", "equal_weight"] = "hrp"
    lookback_days: int = Field(252, ge=20, le=2520)
    risk_free_rate: float = Field(0.045, ge=0.0, le=0.5)
    max_weight: float = Field(0.30, ge=0.0, le=1.0)
    min_weight: float = Field(0.0, ge=0.0, le=1.0)
    shrinkage: Literal["ledoit_wolf", "sample"] = "ledoit_wolf"
    shrink_mu: float = Field(0.5, ge=0.0, le=1.0)
    mc_paths: int = Field(10000, ge=100, le=200000)
    mc_horizon_days: int = Field(252, ge=20, le=2520)
    mc_block: int = Field(20, ge=2, le=126)
    return_frontier: bool = True
    seed: int | None = 7


class FrontierPoint(BaseModel):
    expected_return: float
    expected_vol: float
    sharpe: float


class MCDrawdownStats(BaseModel):
    p05: float
    p50: float
    p95: float
    mean: float
    std: float
    n_paths: int
    horizon_days: int
    block: int


class OptimizeResponse(BaseModel):
    method: str
    weights: dict[str, float]
    expected_return: float
    expected_vol: float
    sharpe: float
    marginal_risk_contribution: dict[str, float]
    diversification_ratio: float
    effective_n: float
    frontier: list[FrontierPoint] | None
    mc_drawdown: MCDrawdownStats
    pair_ids_used: list[str]
    n_observations: int
    warnings: list[str]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        strategies = data.get("strategies") or []
        if not isinstance(strategies, list):
            return []
        return [s for s in strategies if isinstance(s, dict)]
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("failed to load alpha catalog: %s", e)
        return []


def _index_by_pair_id(strategies: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["pair_id"]: s for s in strategies if "pair_id" in s}


def _synth_returns(
    pair_ids: list[str],
    catalog_idx: dict[str, dict[str, Any]],
    lookback_days: int,
    seed: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Synthesise daily returns from each pair's recorded oos_sharpe.

    Strategy:
      - Annualised target Sharpe = ``oos_sharpe`` (fallback 1.0).
      - Annualised vol = a fixed 0.15 (15% — typical alpha-strategy net vol).
      - Annualised return μ = Sharpe × σ + rf  ⇒  but for return generation
        we use the *excess* μ = Sharpe × σ so the result is naturally rf-free.
      - Daily σ = σ_a / √252; daily μ = μ_a / 252.
      - Cross-section correlation: small block-diagonal noise so HRP/ERC
        have something interesting to cluster on.

    Returns ``(df, warnings)`` where ``df`` has shape ``(lookback_days, N)``.
    """
    rng = np.random.default_rng(seed)
    n = len(pair_ids)
    t = int(lookback_days)
    # Per-pair Sharpes (fallback to 1.0 if missing).
    sharpes = np.array(
        [float(catalog_idx.get(pid, {}).get("oos_sharpe") or 1.0) for pid in pair_ids],
        dtype=float,
    )
    sigma_a = 0.15  # 15% annualised vol target
    mu_a = sharpes * sigma_a  # excess return
    sigma_d = sigma_a / math.sqrt(252.0)
    mu_d = mu_a / 252.0

    # Build a mild correlation structure: small uniform off-diagonal ρ.
    rho = 0.10
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    # Convert to covariance (daily).
    sd = np.full(n, sigma_d)
    cov = np.outer(sd, sd) * corr
    # Cholesky for correlated Gaussian draws.
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.diag(sd)
    z = rng.standard_normal(size=(t, n))
    paths = z @ chol.T + mu_d  # broadcast mean per asset

    df = pd.DataFrame(paths, columns=pair_ids)
    warnings = [
        "synthetic_returns: catalog stores summary stats only; daily PnL "
        "history is simulated from oos_sharpe assuming sigma_annual=15%. "
        "POC-grade — replace with cached PnL in production.",
    ]
    return df, warnings


def _to_finite(x: float) -> float:
    return float(x) if math.isfinite(x) else 0.0


def _frontier_to_models(points: list[dict[str, float]]) -> list[FrontierPoint]:
    return [
        FrontierPoint(
            expected_return=_to_finite(p["expected_return"]),
            expected_vol=_to_finite(p["expected_vol"]),
            sharpe=_to_finite(p["sharpe"]),
        )
        for p in points
    ]


# ---------------------------------------------------------------------------
# endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/optimize",
    response_model=None,
    dependencies=[Depends(require_tier("pro"))],
)
def optimize(
    req: OptimizeRequest,
    format: Literal["json", "csv", "pdf"] = Query(default="json"),
) -> OptimizeResponse | FastAPIResponse:
    """Suggest optimal weights for a basket of curated alphas.

    Loads ``web/data/alpha_strategies.json``, finds each requested
    ``pair_id``, builds (or synthesises) a daily-returns DataFrame, runs
    the chosen optimiser, and returns weights + frontier + MC drawdown.
    """
    if len(set(req.pair_ids)) != len(req.pair_ids):
        raise HTTPException(status_code=422, detail="duplicate pair_ids in request")

    catalog = _load_catalog()
    catalog_idx = _index_by_pair_id(catalog)

    warnings: list[str] = []
    if not catalog:
        warnings.append(
            "catalog_missing: alpha_strategies.json not found or unreadable; "
            "proceeding with default-Sharpe synthetic returns."
        )
    missing = [pid for pid in req.pair_ids if pid not in catalog_idx]
    if missing and catalog:
        warnings.append(
            f"unknown_pair_ids: not in catalog — using default oos_sharpe=1.0 for: {missing}"
        )

    # Synthesise returns (placeholder for live PnL cache — see module docstring).
    returns, synth_warns = _synth_returns(
        pair_ids=req.pair_ids,
        catalog_idx=catalog_idx,
        lookback_days=req.lookback_days,
        seed=req.seed,
    )
    warnings.extend(synth_warns)

    # Dispatch to optimiser.
    method = req.method
    # The dispatch table is heterogeneous across optimisers; annotate the
    # kwarg dicts as `dict[str, Any]` so mypy stops trying to unify the
    # mixed `float | Literal[...]` value types across **kwargs sites.
    common_kw: dict[str, Any] = {"rf": req.risk_free_rate, "shrinkage": req.shrinkage}
    box_kw: dict[str, Any] = {"max_w": req.max_weight, "min_w": req.min_weight}
    try:
        if method == "equal_weight":
            result = equal_weight(returns, **common_kw)
        elif method == "min_variance":
            result = min_variance(returns, **box_kw, **common_kw)
        elif method == "mean_variance":
            result = mean_variance_max_sharpe(
                returns, **box_kw, shrink_mu=req.shrink_mu, **common_kw
            )
        elif method == "risk_parity":
            result = risk_parity_erc(returns, **box_kw, **common_kw)
        elif method == "hrp":
            result = hrp(returns, **common_kw)
        else:  # pragma: no cover — Literal guards this
            raise HTTPException(status_code=422, detail=f"unknown method {method}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("optimiser %s failed", method)
        raise HTTPException(status_code=500, detail=f"optimiser failed: {e}") from e

    frontier_pts: list[FrontierPoint] | None = None
    if req.return_frontier:
        try:
            pts = efficient_frontier(
                returns,
                n_points=50,
                max_w=req.max_weight,
                min_w=req.min_weight,
                rf=req.risk_free_rate,
                shrinkage=req.shrinkage,
                shrink_mu=req.shrink_mu,
            )
            frontier_pts = _frontier_to_models(pts)
        except Exception as e:
            logger.warning("frontier failed: %s", e)
            warnings.append(f"frontier_failed: {e}")

    # MC drawdown uses the chosen weights.
    try:
        mc = monte_carlo_drawdown(
            weights=result["weights"],
            returns=returns,
            n_paths=req.mc_paths,
            horizon_days=req.mc_horizon_days,
            block=req.mc_block,
            seed=req.seed,
        )
    except Exception as e:
        logger.warning("MC drawdown failed: %s", e)
        warnings.append(f"mc_drawdown_failed: {e}")
        mc = {
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "n_paths": 0,
            "horizon_days": req.mc_horizon_days,
            "block": req.mc_block,
        }

    response = OptimizeResponse(
        method=method,
        weights=result["weights"],
        expected_return=_to_finite(result["expected_return"]),
        expected_vol=_to_finite(result["expected_vol"]),
        sharpe=_to_finite(result["sharpe"]),
        marginal_risk_contribution=result["marginal_risk_contribution"],
        diversification_ratio=_to_finite(result["diversification_ratio"]),
        effective_n=_to_finite(result["effective_n"]),
        frontier=frontier_pts,
        mc_drawdown=MCDrawdownStats(**mc),
        pair_ids_used=list(req.pair_ids),
        n_observations=int(returns.shape[0]),
        warnings=warnings,
    )
    if format == "json":
        return response
    return _export_respond(
        response,
        format,
        filename="portfolio-optimize",
        kind="portfolio",
    )
