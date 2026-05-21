"""Three literature-grade strategy primitives.

1.  **Hasbrouck (1995) Information Share** — for two cointegrated price
    series (e.g., Kalshi vs Polymarket on the same Fed event), this
    decomposes the *proportion of long-run price-discovery contribution*
    each venue provides. The venue with higher IS leads; the other
    follows. Used directly: long the follower, hedge with the leader.

    Formulation: fit a VAR on Δprices with one cointegrating relation;
    extract orthogonalised covariance matrix Ω; IS_a = (γ·Ω)_a² /
    (γ·Ω·γ') where γ is the orthogonalised vector aligned with the
    cointegrating equation.

2.  **Markov Regime-Switching** (Hamilton 1989) — fit a 2-state hidden
    Markov chain on the spread's first differences. State 0 = "tight
    mean-reversion" (low vol, fast revert), state 1 = "broken" (high
    vol, persistent). Returns smoothed regime probabilities.

3.  **Almgren-Chriss (2001) Optimal Execution** — for a given target
    position, time horizon, permanent and temporary market-impact
    coefficients, and risk-aversion, returns the closed-form optimal
    trading trajectory: x*(t) = X₀ · sinh(κ·(T−t)) / sinh(κ·T) where
    κ = √(λσ²/η) is the urgency parameter.

References:
    Hasbrouck, J. (1995). "One Security, Many Markets: Determining the
        Contributions to Price Discovery." Journal of Finance 50(4).
    Hamilton, J. (1989). "A New Approach to the Economic Analysis of
        Nonstationary Time Series and the Business Cycle." Econometrica.
    Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio
        Transactions." Journal of Risk 3(2), 5-39.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sinh, sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ────────────────────── Hasbrouck Information Share ───────────────────


@dataclass(frozen=True)
class InformationShareResult:
    """Output of :func:`hasbrouck_information_share`.

    Attributes:
        venue_a_id / venue_b_id: identifiers (passthrough for output).
        n_obs: paired sample size after dropna.
        is_a_lower: lower-bound information share for venue A.
        is_a_upper: upper-bound information share for venue A.
        is_b_lower: 1 − is_a_upper (by construction, sums to 1).
        is_b_upper: 1 − is_a_lower.
        leader: ``"A"`` / ``"B"`` / ``"tied"`` based on the midpoints.
        midpoint_a: average of upper and lower for A.
        beta_cointeg: cointegrating slope used (β: P_A = α + β · P_B + ε).
    """

    venue_a_id: str
    venue_b_id: str
    n_obs: int
    is_a_lower: float
    is_a_upper: float
    is_b_lower: float
    is_b_upper: float
    leader: str
    midpoint_a: float
    beta_cointeg: float


def hasbrouck_information_share(
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    venue_a_id: str = "A",
    venue_b_id: str = "B",
    var_lags: int = 5,
) -> InformationShareResult:
    """Hasbrouck (1995) information share decomposition for two
    cointegrated price series.

    Steps (Hasbrouck 1995 §I.B):
        1. Fit cointegrating regression P_a = α + β·P_b + ε; spread = ε.
        2. Build VECM in differences: ΔP = γ·ε_{t-1} + Σ A_k ΔP_{t-k} + u_t.
        3. Extract residual covariance Ω. Cholesky-decompose to get F.
        4. IS_a (upper bound) = (γ_a · F_row_1)² / (γ_a · Ω · γ_a').
        5. IS_a (lower bound) = same with venues reversed in the Cholesky.

    The two bounds bracket the true information share. When they're
    close, the result is unambiguous.

    Args:
        p_a, p_b: aligned probability series (or any prices).
        venue_a_id / venue_b_id: labels.
        var_lags: lags in the VECM. Default 5.

    Returns:
        :class:`InformationShareResult`.
    """
    df = pd.concat({"a": p_a, "b": p_b}, axis=1).dropna()
    n = len(df)
    if n < max(50, var_lags * 4):
        raise ValueError(f"hasbrouck_information_share: need ≥ max(50, 4·var_lags) bars, got {n}")

    a_arr = df["a"].to_numpy(dtype=float)
    b_arr = df["b"].to_numpy(dtype=float)

    # Step 1: cointegrating regression (Engle-Granger 1st step).
    X = sm.add_constant(b_arr)
    ols = sm.OLS(a_arr, X).fit()
    beta = float(ols.params[1])
    spread = a_arr - beta * b_arr - float(ols.params[0])

    # Step 2: VECM in first differences with cointegrating residual as
    # error-correction term.
    da = np.diff(a_arr)
    db = np.diff(b_arr)
    ec = spread[:-1]
    T = len(da)
    if var_lags + 5 >= T:
        raise ValueError(f"hasbrouck_information_share: not enough rows for var_lags={var_lags}")

    # Build regressors: intercept + ec + lagged Δ's.
    rows: list[list[float]] = []
    for t in range(var_lags, T):
        row = [1.0, ec[t]]
        for k in range(1, var_lags + 1):
            row.append(da[t - k])
            row.append(db[t - k])
        rows.append(row)
    X_mat = np.array(rows)
    y_a = da[var_lags:T]
    y_b = db[var_lags:T]
    res_a = sm.OLS(y_a, X_mat).fit()
    res_b = sm.OLS(y_b, X_mat).fit()
    u_a = res_a.resid
    u_b = res_b.resid
    gamma_a = float(res_a.params[1])
    gamma_b = float(res_b.params[1])

    # Step 3: residual covariance.
    cov = np.cov(np.vstack([u_a, u_b]), ddof=1)
    if not np.all(np.isfinite(cov)):
        raise ValueError("residual covariance not finite")

    # Step 4 & 5: Cholesky-based bounds.
    # Order (a, b): IS_a UPPER bound (a's innovation gets all common shocks).
    F1 = np.linalg.cholesky(cov)
    # Order (b, a): IS_a LOWER bound.
    cov_rev = cov[[1, 0], :][:, [1, 0]]
    F2 = np.linalg.cholesky(cov_rev)

    # γ vector in (a, b) and (b, a) orderings.
    gamma_ab = np.array([gamma_a, gamma_b])
    gamma_ba = np.array([gamma_b, gamma_a])

    # Variance contribution from venue 1 = (γ · F · e_1)² where e_1 is unit vec.
    def _contrib_var1(gamma: np.ndarray, F: np.ndarray, F_full: np.ndarray) -> float:
        # γ·F first column squared (variance from first-ordered venue's innovation).
        v = float((gamma @ F[:, 0]) ** 2)
        # Total variance of γ·u = γ Ω γ'.
        total = float(gamma @ F_full @ F_full.T @ gamma)
        return v / total if total > 0 else 0.0

    is_a_upper = _contrib_var1(gamma_ab, F1, F1)
    is_a_lower_via_b_first = 1.0 - _contrib_var1(gamma_ba, F2, F2)
    # Standard convention: lower bound is min, upper is max.
    is_a_lo = min(is_a_upper, is_a_lower_via_b_first)
    is_a_hi = max(is_a_upper, is_a_lower_via_b_first)
    midpoint = 0.5 * (is_a_lo + is_a_hi)
    if midpoint > 0.55:
        leader = venue_a_id
    elif midpoint < 0.45:
        leader = venue_b_id
    else:
        leader = "tied"

    return InformationShareResult(
        venue_a_id=venue_a_id,
        venue_b_id=venue_b_id,
        n_obs=n,
        is_a_lower=float(is_a_lo),
        is_a_upper=float(is_a_hi),
        is_b_lower=float(1.0 - is_a_hi),
        is_b_upper=float(1.0 - is_a_lo),
        leader=leader,
        midpoint_a=float(midpoint),
        beta_cointeg=beta,
    )


# ─────────────────────── Markov Regime-Switching ──────────────────────


@dataclass(frozen=True)
class RegimeSwitchingResult:
    """Output of :func:`markov_regime_switching`.

    Attributes:
        n_obs: sample size.
        regime_probs: per-bar smoothed P(state=1). State 1 is the
            *higher-vol* regime by convention.
        n_state0: bars classified to state 0 (low-vol / mean-reverting).
        n_state1: bars in state 1 (high-vol / broken).
        sigma_state0: estimated σ in state 0.
        sigma_state1: estimated σ in state 1.
        mean_state0: estimated μ in state 0.
        mean_state1: estimated μ in state 1.
        transition_p00: P(stay in state 0).
        transition_p11: P(stay in state 1).
        current_regime: latest state classification (0 or 1).
        current_regime_prob: P(state=1) at the last bar.
        verdict: ``"tradeable"`` (state 0, mean-reversion alive) /
            ``"broken"`` (state 1, regime change risk).
    """

    n_obs: int
    regime_probs: pd.Series
    n_state0: int
    n_state1: int
    sigma_state0: float
    sigma_state1: float
    mean_state0: float
    mean_state1: float
    transition_p00: float
    transition_p11: float
    current_regime: int
    current_regime_prob: float
    verdict: str


def markov_regime_switching(
    spread: pd.Series,
    *,
    k_regimes: int = 2,
    max_iter: int = 50,
) -> RegimeSwitchingResult:
    """Fit a Hamilton (1989) Markov-switching mean+variance model on the
    spread's first differences. Returns smoothed regime probabilities.

    State 1 is *defined as* the higher-σ regime (we re-label after fit
    if statsmodels picked the other ordering).

    Args:
        spread: per-bar spread (level, not differences).
        k_regimes: 2 (default) — the standard tradeable/broken split.
        max_iter: EM iterations.

    Returns:
        :class:`RegimeSwitchingResult`.
    """
    s = spread.dropna()
    diffs = s.diff().dropna()
    n = len(diffs)
    if n < 50:
        raise ValueError(f"markov_regime_switching: need ≥50 bars, got {n}")

    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    try:
        model = MarkovRegression(
            diffs.values,
            k_regimes=k_regimes,
            switching_variance=True,
        )
        # Provide a warm start to speed up convergence; defaults can fail
        # on tiny samples.
        res = model.fit(disp=False, maxiter=max_iter)
    except Exception as e:
        raise ValueError(f"Markov fit failed: {e}") from e

    # Extract per-state mean & variance from the flat params array.
    # MarkovRegression with k_regimes=2 + switching_variance=True parameterises:
    #   [p[i->0] for i=0..k-1, const[0..k-1], sigma2[0..k-1]].
    # We use the regime_transition matrix attribute and the params slices.
    p = np.asarray(res.params, dtype=float)
    n_trans = k_regimes * (k_regimes - 1)
    const_offset = n_trans
    sigma2_offset = n_trans + k_regimes
    means = [float(p[const_offset + k]) for k in range(k_regimes)]
    sigma2s = [float(p[sigma2_offset + k]) for k in range(k_regimes)]
    sigmas = [sqrt(max(s2, 1e-12)) for s2 in sigma2s]

    # Identify which is state 1 (higher σ).
    state1_idx = int(np.argmax(sigmas))
    state0_idx = 1 - state1_idx if k_regimes == 2 else (state1_idx + 1) % k_regimes

    # Smoothed marginal probabilities. statsmodels returns shape
    # (n_obs, k_regimes) — column index = state.
    smp = np.asarray(res.smoothed_marginal_probabilities)
    if smp.ndim == 2 and smp.shape[1] == k_regimes:
        smooth_probs = smp[:, state1_idx]
    elif smp.ndim == 2 and smp.shape[0] == k_regimes:
        smooth_probs = smp[state1_idx, :]
    else:
        smooth_probs = smp.flatten()[: len(diffs)]
    regime_probs = pd.Series(smooth_probs, index=diffs.index, name="p_state1")

    # Transition probs: P(j|i) is res.regime_transition[i, j].
    rt = res.regime_transition
    # statsmodels can return rank-3 tensor (i, j, t); take first if so
    if hasattr(rt, "shape") and len(rt.shape) == 3:
        rt = rt[:, :, 0]
    p00 = float(rt[state0_idx, state0_idx])
    p11 = float(rt[state1_idx, state1_idx])

    n_state1 = int((regime_probs > 0.5).sum())
    n_state0 = n - n_state1
    current_p = float(regime_probs.iloc[-1])
    current_regime = 1 if current_p > 0.5 else 0
    verdict = "broken" if current_regime == 1 else "tradeable"

    return RegimeSwitchingResult(
        n_obs=n,
        regime_probs=regime_probs,
        n_state0=n_state0,
        n_state1=n_state1,
        sigma_state0=sigmas[state0_idx],
        sigma_state1=sigmas[state1_idx],
        mean_state0=means[state0_idx],
        mean_state1=means[state1_idx],
        transition_p00=p00,
        transition_p11=p11,
        current_regime=current_regime,
        current_regime_prob=current_p,
        verdict=verdict,
    )


# ─────────────────── Almgren-Chriss optimal execution ────────────────


@dataclass(frozen=True)
class AlmgrenChrissSchedule:
    """Output of :func:`almgren_chriss_schedule`.

    Attributes:
        n_intervals: number of trading intervals.
        x_remaining: per-interval remaining position (X₀ → 0).
        n_per_interval: per-interval shares to trade (negative = sell).
        kappa: urgency parameter √(λσ²/η).
        time_horizon: total trade duration.
        expected_cost: E[total cost] = permanent impact × X₀ +
            temporary impact integral.
        variance_cost: Var(cost) — risk component.
        utility: E[cost] + λ · Var(cost). Lower = better.
    """

    n_intervals: int
    x_remaining: list[float]
    n_per_interval: list[float]
    kappa: float
    time_horizon: float
    expected_cost: float
    variance_cost: float
    utility: float


def almgren_chriss_schedule(
    target_position: float,
    *,
    n_intervals: int = 10,
    time_horizon: float = 1.0,
    sigma: float = 0.10,
    eta: float = 0.01,
    epsilon: float = 0.005,
    gamma_perm: float = 0.0,
    risk_aversion: float = 1.0,
) -> AlmgrenChrissSchedule:
    """Closed-form Almgren-Chriss (2001) optimal execution trajectory.

    For a target position ``X₀`` to be liquidated over ``[0, T]`` in
    ``n_intervals`` steps, with per-interval volatility σ, temporary
    market-impact coefficient η, fixed cost ε, permanent impact γ, and
    risk-aversion λ, the optimal trajectory is

        x*(t_k) = X₀ · sinh(κ·(T − t_k)) / sinh(κ·T)

    where κ = √(λσ²/η) is the urgency parameter. λ=0 ⇒ TWAP; λ→∞ ⇒
    instant execution.

    Args:
        target_position: signed position to acquire (positive = buy, negative = sell).
        n_intervals: number of equal-time slices.
        time_horizon: total duration in units consistent with σ.
        sigma: per-unit-time price volatility.
        eta: temporary impact coefficient (price drift per share-rate).
        epsilon: per-interval fixed cost.
        gamma_perm: permanent market-impact coefficient.
        risk_aversion: λ in mean-variance utility. 0 = TWAP, ∞ = ASAP.

    Returns:
        :class:`AlmgrenChrissSchedule`.
    """
    if n_intervals < 2:
        raise ValueError(f"n_intervals must be ≥ 2, got {n_intervals}")
    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")
    tau = time_horizon / n_intervals
    eta_tilde = eta - 0.5 * gamma_perm * tau
    if eta_tilde <= 0:
        raise ValueError(
            f"effective temporary impact eta_tilde = {eta_tilde:.4f} ≤ 0; "
            f"increase eta or decrease gamma_perm·τ"
        )
    kappa_sq = risk_aversion * sigma * sigma / eta_tilde
    kappa = float(np.sqrt(max(kappa_sq, 1e-12)))

    if kappa * time_horizon < 1e-6:
        # Limit case: TWAP.
        x = [target_position * (1.0 - k / n_intervals) for k in range(n_intervals + 1)]
    else:
        denom = sinh(kappa * time_horizon)
        x = [
            target_position * sinh(kappa * (time_horizon - k * tau)) / denom
            for k in range(n_intervals + 1)
        ]
    n_per = [x[k] - x[k + 1] for k in range(n_intervals)]

    # E[cost] (Almgren-Chriss eq. 19, simplified):
    # E[X] = 0.5·γ·X₀² + (X₀² · η_tilde / T) · (κT·coth(κT))   for the κ>0 case.
    if kappa * time_horizon > 1e-6:
        ct = float(np.cosh(kappa * time_horizon) / sinh(kappa * time_horizon))
        e_cost = (
            0.5 * gamma_perm * target_position**2
            + (target_position**2) * eta_tilde * (kappa * ct) / time_horizon
        )
        # Var(cost) = X₀² · σ² · T/3 in the κ→0 limit; with κ>0 use the AC
        # closed form. We use the standard AC variance formula:
        v_cost = (target_position**2 * sigma * sigma) * (
            0.5
            * time_horizon
            * (
                (
                    np.tanh(0.5 * kappa * time_horizon)
                    * (
                        np.tanh(0.5 * kappa * time_horizon)
                        + (kappa * time_horizon * (1 / np.cosh(kappa * time_horizon))) ** 2 / 2
                    )
                )
                / (kappa * time_horizon) ** 2
            )
        )
    else:
        e_cost = (
            0.5 * gamma_perm * target_position**2 + eta_tilde * target_position**2 / time_horizon
        )
        v_cost = target_position**2 * sigma * sigma * time_horizon / 3.0

    utility = e_cost + risk_aversion * v_cost

    return AlmgrenChrissSchedule(
        n_intervals=n_intervals,
        x_remaining=[float(v) for v in x],
        n_per_interval=[float(v) for v in n_per],
        kappa=kappa,
        time_horizon=time_horizon,
        expected_cost=float(e_cost),
        variance_cost=float(v_cost),
        utility=float(utility),
    )


__all__ = [
    "AlmgrenChrissSchedule",
    "InformationShareResult",
    "RegimeSwitchingResult",
    "almgren_chriss_schedule",
    "hasbrouck_information_share",
    "markov_regime_switching",
]
