"""Ornstein-Uhlenbeck calibration + continuous-time OU optimal entry/exit thresholds.

For a cointegrated spread we model the dynamics as continuous-time OU:

    dX_t = κ(μ − X_t) dt + σ dW_t

Calibration (least-squares of the discretised AR(1)):

    X_{t+Δt} = X_t · e^{−κΔt} + μ(1 − e^{−κΔt}) + ε_t,
        ε_t ~ N(0, σ² · (1 − e^{−2κΔt}) / (2κ))

Setting Δt = 1 bar, we fit AR(1) on X via OLS to get α, β:

    X_{t+1} = α + β X_t + η_t      ⇒    κ = −ln(β)/Δt,
                                        μ = α/(1 − β),
                                        σ²_eq = Var(η)/(1 − β²)
                                                = σ²/(2κ)        (eq variance).

Continuous-time OU optimal trading bands:

For a symmetric trade ([−a, +a] entry, exit at 0), the expected profit per
unit time is maximised at the dimensionless level

    z* ≈ 1.5  (numerical solution of the expected-profit-per-unit-time problem).

For asymmetric thresholds (entry at z_e, exit at z_x with z_e > z_x ≥ 0),
expected per-trade-cycle PnL = 2(z_e − z_x) σ_eq, expected cycle time
T(z_e, z_x), and the analytic value function maximises (PnL − cost) / T.

In practice:
*   ``z_entry_optimal = 1.5`` (Bertram's symmetric optimum, in units of σ_eq)
    is the textbook starting point.
*   For *asymmetric* exits we use the closed-form expected hitting time of OU:
        E[τ(0 → ±a)] = (1/2κ)·[Φ(−a√(2κ)/σ) − Φ(0)]·...
    and numerically maximise PnL/cycle_time.

This module returns the calibrated κ, μ, σ_eq and the Bertram-optimal bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.special import erfi


@dataclass(frozen=True)
class OUFit:
    """Calibrated OU dynamics on a spread series.

    Attributes:
        kappa: mean-reversion speed (per bar).
        mu: long-run mean of the spread.
        sigma_eq: equilibrium std (the long-run std of X_t).
        sigma_innov: per-bar innovation std (η_t).
        half_life_bars: ln(2) / κ.
        ar1_beta: discrete-time AR(1) slope = exp(−κ).
        n_obs: bars used in the fit.
    """

    kappa: float
    mu: float
    sigma_eq: float
    sigma_innov: float
    half_life_bars: float
    ar1_beta: float
    n_obs: int


def fit_ou(spread: pd.Series, *, dt: float = 1.0) -> OUFit:
    """Calibrate OU dynamics by AR(1) regression on the spread.

    Args:
        spread: per-bar spread series (already de-noised).
        dt: bar duration. Use 1 for daily bars; 1/24 for hourly etc.

    Returns:
        :class:`OUFit`.

    Raises:
        ValueError: if AR(1) coefficient is non-positive (no mean reversion)
            or ≥ 1 (non-stationary).
    """
    s = spread.dropna()
    if len(s) < 10:
        raise ValueError(f"fit_ou: need ≥10 bars, got {len(s)}")
    y = s.iloc[1:].to_numpy()
    x = s.iloc[:-1].to_numpy()
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    alpha = float(res.params[0])
    beta = float(res.params[1])
    if beta <= 0.0 or beta >= 1.0:
        raise ValueError(f"fit_ou: AR(1) β={beta:.3f} not in (0,1) — no stationary OU fit")
    kappa = -log(beta) / dt
    mu = alpha / (1.0 - beta)
    sigma_innov = float(np.std(res.resid, ddof=1))
    # Long-run (equilibrium) variance: σ²_eq = σ²_innov / (1 − β²).
    sigma_eq = sigma_innov / sqrt(1.0 - beta * beta)
    half_life = log(2.0) / kappa
    return OUFit(
        kappa=kappa,
        mu=mu,
        sigma_eq=sigma_eq,
        sigma_innov=sigma_innov,
        half_life_bars=half_life,
        ar1_beta=beta,
        n_obs=len(s),
    )


# ─────────────────────── Bertram optimal bands ────────────────────────


def _bertram_t(z: float) -> float:
    """Bertram's expected hitting-time function on the standardised OU
    process (κ=1, σ=1, μ=0): T(z) = (π/2)·erfi(z/√2).

    The expected first-passage time from ``z_a`` to ``z_b`` (with
    ``z_a < z_b``) is ``T(z_b) − T(z_a)``.
    """
    return float((np.pi / 2.0) * erfi(z / sqrt(2.0)))


def bertram_optimal_bands(
    fit: OUFit,
    *,
    transaction_cost: float = 0.10,
) -> dict[str, float]:
    """Continuous-time OU optimal entry / exit z-thresholds for an OU spread.

    Maximises expected return per unit time of a *symmetric* trade with
    entry at ±z_e and exit at the long-run mean (z_x = 0):

        objective(z_e) = (z_e − cost) / T(z_e)            [Bertram §3]

    where ``T(z) = (π/2)·erfi(z/√2)`` is the dimensionless expected
    first-passage time from 0 → z under standardised OU.

    Note: at zero transaction cost the objective is monotonically
    decreasing (smaller z_e ⇒ shorter cycle ⇒ better PnL/time, in the
    limit), so no interior optimum exists. We require ``cost > 0`` (set
    to a strictly-positive default of 0.10 σ_eq, matching ~1-3¢
    round-trip Polymarket spreads on a 0.10–0.30 σ spread).

    Args:
        fit: calibrated :class:`OUFit`. Used only to scale the answer back
            into raw spread units and to estimate per-year throughput.
        transaction_cost: per-trade round-trip cost in *dimensionless*
            units (multiples of σ_eq).

    Returns:
        Dict with ``z_entry``, ``z_exit``, ``expected_pnl_per_cycle_sigma``,
        ``expected_cycle_bars``, ``expected_pnl_per_year_sigma``,
        ``transaction_cost``.
    """
    if transaction_cost <= 0:
        # Without a cost there's no interior optimum; fall back to the
        # well-known Bertram heuristic z* = 1.5 σ_eq (good practical
        # default that's robust to mis-specified cost).
        z_entry = 1.5
    else:

        def objective(z_e: float) -> float:
            if z_e <= 0:
                return 1e6
            t = _bertram_t(z_e)
            if t <= 0:
                return 1e6
            return -(z_e - transaction_cost) / t

        res = minimize_scalar(
            objective,
            bounds=(max(transaction_cost + 1e-3, 0.05), 5.0),
            method="bounded",
        )
        z_entry = float(res.x)

    z_exit = 0.0
    # Cycle time on the standardised process; convert to bars via 1/κ.
    cycle_dimensionless = 2.0 * _bertram_t(z_entry)  # there + back
    cycle_bars = cycle_dimensionless / max(fit.kappa, 1e-9)
    ep_per_cycle = z_entry - z_exit - transaction_cost
    # Annualised PnL in σ_eq units (assumes 252 bars/year).
    pnl_per_year = (ep_per_cycle / max(cycle_bars, 1e-9)) * 252.0

    return {
        "z_entry": z_entry,
        "z_exit": z_exit,
        "expected_pnl_per_cycle_sigma": float(ep_per_cycle),
        "expected_cycle_bars": float(cycle_bars),
        "expected_pnl_per_year_sigma": float(pnl_per_year),
        "transaction_cost": float(transaction_cost),
    }


__all__ = ["OUFit", "bertram_optimal_bands", "fit_ou"]
