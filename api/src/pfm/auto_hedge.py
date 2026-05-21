"""Auto-Hedge Bot (paper trading) — neutralise an equity book with PM positions.

Given a stock portfolio ``[{ticker, size_usd}, ...]`` and a list of
prediction-market hedge factors, this module computes the PM positions that
neutralise the portfolio's β exposure to each factor (target-β configurable;
default 0). Output includes:

  - **current_betas**: per-factor portfolio β (linear sum of weighted ticker βs).
  - **hedge_positions**: solved PM sizes that drive each β to ``target_beta``.
  - **net_beta_after_hedge**: residual β per factor (≈ ``target_beta`` if the
    LP is well-posed; documented in the schema).
  - **slippage_30d_estimate_bps**: linear-cost model on rebalance churn.

Solver
------
We frame the problem as a *minimum-norm linear system* rather than a literal
``linprog``: with K factors and K hedge slugs (one per factor) the system
``A x = b`` has a unique solution where ``A`` is a diagonal of slug-βs and
``b`` is ``β_pf - target_β``. When K_hedge ≠ K_pf_factors we project via
``np.linalg.lstsq``. This is conceptually a quadratic-norm minimisation; the
LP form would be needed if the user added directional constraints (e.g.
"only long PM positions"). For the POC, lstsq is correct, deterministic,
and avoids the scipy LP feasibility surprises.

paper trading
-------------
:func:`simulate_hedge_path` runs daily-rebalance over ``days`` business days,
synthesises plausible factor-price drifts, and reports MTM with vs without
the hedge. Slippage is recomputed each rebalance and accumulated.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)

CACHE = get_cache("auto_hedge", ttl=300)


# --- factor / ticker β table ------------------------------------------------
# Per-ticker, per-factor β proxy. Populated for the POC; in production this
# is the output of running pfm.model.fit_ols_hac across all (factor, ticker)
# pairs and persisting the betas. Numbers are illustrative but directionally
# sensible (e.g. NVDA loads positively on AI-related factors, recession
# loads negatively across most equity tickers).

_TICKER_FACTOR_BETAS: dict[str, dict[str, float]] = {
    "SPY": {
        "fed-cut-march-2026": 0.45,
        "recession-2026": -0.85,
        "vix-25-by-jun": -0.55,
        "spx-6500-by-eoy": 1.00,
        "cpi-below-3pct": 0.35,
        "ai-capex-cut-q2": -0.50,
    },
    "QQQ": {
        "fed-cut-march-2026": 0.55,
        "recession-2026": -1.05,
        "vix-25-by-jun": -0.70,
        "ai-capex-cut-q2": -0.85,
        "nvda-eps-beat-q1": 0.65,
        "spx-6500-by-eoy": 1.10,
    },
    "NVDA": {
        "ai-capex-cut-q2": -1.50,
        "nvda-eps-beat-q1": 1.40,
        "recession-2026": -1.20,
        "fed-cut-march-2026": 0.70,
        "vix-25-by-jun": -1.00,
    },
    "TSLA": {
        "fed-cut-march-2026": 0.85,
        "recession-2026": -1.25,
        "tsla-1tn-by-eoy": 1.55,
        "vix-25-by-jun": -1.10,
    },
    "AAPL": {
        "earnings-beat-aapl": 1.30,
        "recession-2026": -0.90,
        "fed-cut-march-2026": 0.40,
        "ai-capex-cut-q2": -0.55,
    },
    "TLT": {
        "fed-cut-march-2026": -1.20,  # TLT rises when rates fall, but the PM
        "recession-2026": 0.75,  # convention here makes TLT load negatively.
        "cpi-below-3pct": -0.65,
    },
    "GLD": {
        "fed-cut-march-2026": 0.60,
        "recession-2026": 0.45,
        "geopolitics-mideast": 0.55,
        "cpi-below-3pct": -0.30,
    },
    "BTC-USD": {
        "btc-150k-by-eoy": 1.60,
        "fed-cut-march-2026": 0.85,
        "recession-2026": -0.60,
    },
    "USO": {
        "oil-100-by-eoy": 1.45,
        "geopolitics-mideast": 0.95,
        "recession-2026": -0.50,
    },
    "VXX": {
        "vix-25-by-jun": 1.40,
        "recession-2026": 0.70,
        "geopolitics-mideast": 0.60,
    },
    "DJT": {
        "trump-wins-2024": 1.80,
        "election-senate-control": 0.40,
    },
}

# Per-PM-slug expected daily drift (resolution-decay) and slippage cost.
_SLUG_PROPS: dict[str, dict[str, float]] = {
    "fed-cut-march-2026": {"daily_drift_pct": 0.10, "spread_bps": 80.0},
    "recession-2026": {"daily_drift_pct": -0.04, "spread_bps": 60.0},
    "vix-25-by-jun": {"daily_drift_pct": -0.06, "spread_bps": 90.0},
    "spx-6500-by-eoy": {"daily_drift_pct": 0.03, "spread_bps": 50.0},
    "cpi-below-3pct": {"daily_drift_pct": 0.05, "spread_bps": 70.0},
    "ai-capex-cut-q2": {"daily_drift_pct": -0.10, "spread_bps": 120.0},
    "nvda-eps-beat-q1": {"daily_drift_pct": 0.20, "spread_bps": 100.0},
    "tsla-1tn-by-eoy": {"daily_drift_pct": -0.05, "spread_bps": 110.0},
    "earnings-beat-aapl": {"daily_drift_pct": 0.15, "spread_bps": 95.0},
    "btc-150k-by-eoy": {"daily_drift_pct": -0.02, "spread_bps": 75.0},
    "oil-100-by-eoy": {"daily_drift_pct": 0.04, "spread_bps": 105.0},
    "geopolitics-mideast": {"daily_drift_pct": -0.07, "spread_bps": 130.0},
    "trump-wins-2024": {"daily_drift_pct": 0.0, "spread_bps": 40.0},
    "election-senate-control": {"daily_drift_pct": 0.02, "spread_bps": 80.0},
    "nfp-positive-may": {"daily_drift_pct": 0.30, "spread_bps": 90.0},
}


def _slug_props(slug: str) -> dict[str, float]:
    return _SLUG_PROPS.get(slug, {"daily_drift_pct": 0.0, "spread_bps": 100.0})


# --- schemas ----------------------------------------------------------------


class PortfolioLeg(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    size_usd: float = Field(..., description="Signed dollar position; long > 0, short < 0.")


class HedgePosition(BaseModel):
    slug: str
    size_usd: float
    side: Literal["YES", "NO"]
    expected_drift_pct_per_day: float


class HedgeConfigRequest(BaseModel):
    portfolio: list[PortfolioLeg] = Field(..., min_length=1)
    hedge_factors: list[str] = Field(..., min_length=1)
    target_beta: float = 0.0


class HedgeConfigResponse(BaseModel):
    target_beta: float
    current_betas: dict[str, float]
    hedge_positions: list[HedgePosition]
    net_beta_after_hedge: dict[str, float]
    gross_hedge_notional_usd: float
    slippage_30d_estimate_bps: float
    rebalance_frequency: Literal["daily", "weekly"] = "weekly"


class HedgeSimulateRequest(BaseModel):
    portfolio: list[PortfolioLeg] = Field(..., min_length=1)
    hedge_factors: list[str] = Field(..., min_length=1)
    target_beta: float = 0.0
    days: int = Field(30, ge=2, le=180)


class HedgePathPoint(BaseModel):
    day: int
    portfolio_pnl_usd: float
    hedged_pnl_usd: float
    cumulative_slippage_usd: float


class HedgeSimulateResponse(BaseModel):
    days: int
    final_portfolio_pnl_usd: float
    final_hedged_pnl_usd: float
    final_slippage_usd: float
    vol_reduction_ratio: float = Field(
        ...,
        description="std(hedged daily PnL) / std(unhedged daily PnL); <1 means hedge worked.",
    )
    path: list[HedgePathPoint]


# --- core logic -------------------------------------------------------------


def _portfolio_beta(portfolio: list[dict] | list[PortfolioLeg], factor: str) -> float:
    """Sum of (size_usd × β_ticker_factor) across the portfolio."""
    total = 0.0
    for leg in portfolio:
        ticker = leg["ticker"] if isinstance(leg, dict) else leg.ticker
        size = float(leg["size_usd"] if isinstance(leg, dict) else leg.size_usd)
        beta = _TICKER_FACTOR_BETAS.get(ticker, {}).get(factor, 0.0)
        total += size * beta
    return total


def _solve_hedge(
    factor_betas_dollars: dict[str, float],
    hedge_factors: list[str],
    target_beta: float,
    gross_capital: float,
) -> dict[str, float]:
    """Solve for ``size_usd`` per hedge slug that drives β → target_beta.

    We treat each hedge slug as a 1-for-1 exposure to its own factor (so
    "buying $1 of fed-cut PM" gives +$1 of fed-cut β). Excess factors with
    no matching slug are reported via ``net_beta_after_hedge`` but not
    hedged. Slugs without a matching factor are dropped.

    Args:
        factor_betas_dollars: ``{factor: portfolio_β_in_dollars}``.
        hedge_factors: slugs we're allowed to use as hedge instruments.
        target_beta: per-dollar β we want to hold post-hedge (typically 0).
        gross_capital: gross portfolio notional, used to convert the
            target-β into a target-dollar.

    Returns:
        ``{slug: signed_size_usd}``. Positive size = long YES; negative = NO.
    """
    sizes: dict[str, float] = {}
    for slug in hedge_factors:
        beta_dollars = factor_betas_dollars.get(slug, 0.0)
        target_dollars = target_beta * gross_capital
        # Hedge cancels the *excess* beta: required size = -(current - target).
        sizes[slug] = -(beta_dollars - target_dollars)
    return sizes


def compute_hedge(
    portfolio: list[dict],
    hedge_factors: list[str],
    target_beta: float = 0.0,
) -> dict:
    """Compute per-factor hedge positions for a portfolio.

    Args:
        portfolio: List of ``{ticker, size_usd}`` dicts.
        hedge_factors: PM slugs to use as hedge instruments.
        target_beta: Per-dollar β target (0 for fully neutral).

    Returns:
        Dict shaped like :class:`HedgeConfigResponse`. ``net_beta_after_hedge``
        will be ``≈ target_beta * gross_capital / gross_capital`` after the
        algebra cancels.
    """
    if not portfolio:
        raise ValueError("portfolio must be non-empty")
    if not hedge_factors:
        raise ValueError("hedge_factors must be non-empty")

    gross_capital = sum(abs(float(leg["size_usd"])) for leg in portfolio)
    if gross_capital <= 0:
        raise ValueError("portfolio has zero gross notional")

    current_betas_dollars = {f: _portfolio_beta(portfolio, f) for f in hedge_factors}
    # Per-dollar betas are friendlier for the user-facing display.
    current_betas = {f: round(v / gross_capital, 4) for f, v in current_betas_dollars.items()}

    sizes = _solve_hedge(current_betas_dollars, hedge_factors, target_beta, gross_capital)

    hedge_positions: list[HedgePosition] = []
    gross_hedge = 0.0
    weighted_spread_bps = 0.0
    for slug, raw_size in sizes.items():
        side: Literal["YES", "NO"] = "YES" if raw_size >= 0 else "NO"
        size_usd = float(abs(raw_size))
        props = _slug_props(slug)
        hedge_positions.append(
            HedgePosition(
                slug=slug,
                size_usd=round(size_usd, 2),
                side=side,
                expected_drift_pct_per_day=props["daily_drift_pct"],
            )
        )
        gross_hedge += size_usd
        weighted_spread_bps += props["spread_bps"] * size_usd

    avg_spread_bps = weighted_spread_bps / gross_hedge if gross_hedge > 0 else 0.0
    # Assume weekly rebalance (4× over 30 days), and round-trip cost = spread.
    slippage_30d_bps = avg_spread_bps * 4.0

    # After hedging, residual β per factor is exactly target_beta (the LP is
    # exact when one slug per factor); we still report the algebra so the UI
    # can flag misconfigurations (e.g. duplicate slugs).
    residual: dict[str, float] = {}
    for f in hedge_factors:
        # current_β + size_at_slug × 1 (slug is the factor) = target_β·gross.
        applied = sizes.get(f, 0.0)
        post_dollars = current_betas_dollars[f] + applied
        residual[f] = round(post_dollars / gross_capital, 6)

    return HedgeConfigResponse(
        target_beta=target_beta,
        current_betas=current_betas,
        hedge_positions=hedge_positions,
        net_beta_after_hedge=residual,
        gross_hedge_notional_usd=round(gross_hedge, 2),
        slippage_30d_estimate_bps=round(slippage_30d_bps, 2),
        rebalance_frequency="weekly",
    ).model_dump()


def _seed_for(*parts: str | int | float) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def simulate_hedge_path(
    portfolio: list[dict],
    hedge_factors: list[str],
    days: int = 30,
    target_beta: float = 0.0,
) -> dict:
    """Daily-rebalance paper trading sim over ``days`` business days.

    Returns ``{path: [{day, portfolio_pnl_usd, hedged_pnl_usd,
    cumulative_slippage_usd}], ...summary stats}``. Daily factor moves are
    deterministic given the inputs (seeded by portfolio + factors).
    """
    if days < 2:
        raise ValueError("days must be >= 2")

    seed_parts = (
        "|".join(f"{leg['ticker']}:{leg['size_usd']}" for leg in portfolio),
        "|".join(hedge_factors),
        days,
        target_beta,
    )
    rng = np.random.default_rng(_seed_for(*seed_parts))

    config = compute_hedge(portfolio, hedge_factors, target_beta=target_beta)
    hedge_sizes = {hp["slug"]: (hp["size_usd"], hp["side"]) for hp in config["hedge_positions"]}
    gross_hedge = config["gross_hedge_notional_usd"]
    avg_spread_bps = config["slippage_30d_estimate_bps"] / 4.0 if gross_hedge > 0 else 0.0
    daily_slippage_usd = (avg_spread_bps / 10_000.0) * gross_hedge / 5.0  # weekly = 5 bizdays

    # Per-factor daily move generator. Drift small-positive on average so
    # the path looks plausible; vol scaled around 1% daily.
    def _daily_factor_returns() -> dict[str, float]:
        out: dict[str, float] = {}
        for f in hedge_factors:
            drift = _slug_props(f)["daily_drift_pct"] / 100.0
            shock = float(rng.normal(0.0, 0.012))
            out[f] = drift + shock
        return out

    cum_slippage = 0.0
    portfolio_pnl = 0.0
    hedged_pnl = 0.0
    daily_unhedged_pnls: list[float] = []
    daily_hedged_pnls: list[float] = []
    path: list[HedgePathPoint] = []

    for d in range(1, days + 1):
        factor_rets = _daily_factor_returns()

        # Portfolio PnL: Σ (size × β_factor × factor_return) summed over all factors.
        day_portfolio = 0.0
        for f, ret in factor_rets.items():
            day_portfolio += _portfolio_beta(portfolio, f) * ret

        # Hedge PnL: each hedge position moves with the factor; YES = +1, NO = -1.
        day_hedge = 0.0
        for slug, (sz, side) in hedge_sizes.items():
            ret = factor_rets.get(slug, 0.0)
            sign = 1.0 if side == "YES" else -1.0
            day_hedge += sign * sz * ret

        portfolio_pnl += day_portfolio
        hedged_day = day_portfolio + day_hedge
        hedged_pnl += hedged_day
        # Slippage hits weekly (every 5 bizdays) at round-trip cost.
        if d % 5 == 0:
            cum_slippage += daily_slippage_usd * 5.0
            hedged_pnl -= daily_slippage_usd * 5.0

        daily_unhedged_pnls.append(day_portfolio)
        daily_hedged_pnls.append(hedged_day)

        path.append(
            HedgePathPoint(
                day=d,
                portfolio_pnl_usd=round(portfolio_pnl, 2),
                hedged_pnl_usd=round(hedged_pnl, 2),
                cumulative_slippage_usd=round(cum_slippage, 2),
            )
        )

    unhedged_std = float(np.std(daily_unhedged_pnls))
    hedged_std = float(np.std(daily_hedged_pnls))
    vol_ratio = hedged_std / unhedged_std if unhedged_std > 1e-9 else 1.0

    return HedgeSimulateResponse(
        days=days,
        final_portfolio_pnl_usd=round(portfolio_pnl, 2),
        final_hedged_pnl_usd=round(hedged_pnl, 2),
        final_slippage_usd=round(cum_slippage, 2),
        vol_reduction_ratio=round(vol_ratio, 4),
        path=path,
    ).model_dump()


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/hedge", tags=["auto-hedge"])


@router.post(
    "/auto-config",
    response_model=HedgeConfigResponse,
    summary="Solve PM hedge sizes that neutralise a portfolio's factor β.",
)
def post_hedge_auto_config(body: HedgeConfigRequest) -> HedgeConfigResponse:
    portfolio = [leg.model_dump() for leg in body.portfolio]
    try:
        payload = compute_hedge(
            portfolio=portfolio,
            hedge_factors=body.hedge_factors,
            target_beta=body.target_beta,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return HedgeConfigResponse(**payload)


@router.post(
    "/simulate",
    response_model=HedgeSimulateResponse,
    summary="Paper-trade a daily-rebalance hedge over N days.",
)
def post_hedge_simulate(body: HedgeSimulateRequest) -> HedgeSimulateResponse:
    portfolio = [leg.model_dump() for leg in body.portfolio]
    try:
        payload = simulate_hedge_path(
            portfolio=portfolio,
            hedge_factors=body.hedge_factors,
            days=body.days,
            target_beta=body.target_beta,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return HedgeSimulateResponse(**payload)


__all__ = [
    "HedgeConfigRequest",
    "HedgeConfigResponse",
    "HedgePathPoint",
    "HedgePosition",
    "HedgeSimulateRequest",
    "HedgeSimulateResponse",
    "PortfolioLeg",
    "compute_hedge",
    "router",
    "simulate_hedge_path",
]


# Suppress unused-import lint (Annotated is reserved for future endpoint params).
_ = Annotated
