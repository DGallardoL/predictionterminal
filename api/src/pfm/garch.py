"""GARCH(1,1) conditional volatility (Bollerslev 1986) and asymmetric
extensions GJR-GARCH (Glosten-Jagannathan-Runkle 1993) and EGARCH
(Nelson 1991).

Pure-scipy MLE implementation — no external `arch` dependency. The
classical conditional-heteroscedasticity model:

    Δp_t = μ + ε_t,           ε_t ~ N(0, σ_t²)
    σ_t² = ω + α · ε_{t-1}² + β · σ_{t-1}²

Stationarity requires α + β < 1; the unconditional variance is then
ω / (1 − α − β). MLE of (μ, ω, α, β) maximises

    log L = −0.5 · Σ_t [log(2π σ_t²) + ε_t² / σ_t²]

We use scipy.optimize.minimize with the BFGS method on a re-parameterised
problem that keeps ω > 0 and α + β < 1.

GJR-GARCH(1,1) adds an indicator-style asymmetry term to capture the
leverage effect that is empirically present in equity returns
(negative shocks raise next-period vol more than positive shocks of
the same magnitude):

    σ_t² = ω + α · ε_{t-1}² + γ · I[ε_{t-1}<0] · ε_{t-1}² + β · σ_{t-1}²

with stationarity ω > 0, α ≥ 0, β ≥ 0, α + γ/2 + β < 1 and γ > 0
identifying the leverage effect.

EGARCH(1,1) parameterises log σ_t² so positivity is automatic and
allows unconstrained α, γ:

    log σ_t² = ω + α · (|z_{t-1}| − E|z|) + γ · z_{t-1} + β · log σ_{t-1}²,
    z_{t-1} = ε_{t-1} / σ_{t-1},     E|z| = √(2/π) under N(0,1).

In equity series γ < 0 typically (negative-shock asymmetry).

For prediction-market spreads, GARCH gives a one-step-ahead conditional
σ_t — strictly better than rolling realised σ for vol-targeting:
- Captures vol clustering (σ_t depends on |ε_{t-1}|)
- Adapts faster than rolling-window after a shock
- 4-parameter MLE: stable on 50+ bars

References:
    Bollerslev, T. (1986). "Generalized Autoregressive Conditional
    Heteroskedasticity." J. Econometrics 31, 307-327.
    Engle, R. (1982). "Autoregressive Conditional Heteroskedasticity with
    Estimates of the Variance of UK Inflation." Econometrica 50.
    Glosten, L., Jagannathan, R., Runkle, D. (1993). "On the Relation
    between the Expected Value and the Volatility of the Nominal
    Excess Return on Stocks." J. Finance 48, 1779-1801.
    Nelson, D. (1991). "Conditional Heteroskedasticity in Asset Returns:
    A New Approach." Econometrica 59, 347-370.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import lgamma, pi, sqrt
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GarchResult:
    """Output of :func:`fit_garch_11`."""

    mu: float
    omega: float
    alpha: float
    beta: float
    persistence: float  # α + β; < 1 for stationarity
    long_run_variance: float
    log_likelihood: float
    n_obs: int
    converged: bool
    conditional_sigma: pd.Series
    standardised_residuals: pd.Series
    last_sigma: float  # σ̂_{T+1} forecast
    is_stationary: bool


def _negative_log_lik(params: np.ndarray, eps: np.ndarray) -> float:
    """Negative log-likelihood for GARCH(1,1) on innovations ``eps``.

    params = [omega, alpha, beta].
    """
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10  # infeasible region
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = float(np.var(eps, ddof=0))  # unconditional var as init
    if sigma2[0] <= 0:
        sigma2[0] = 1e-8
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] <= 0:
            return 1e10
    nll = 0.5 * np.sum(np.log(2 * pi * sigma2) + eps * eps / sigma2)
    return float(nll)


def fit_garch_11(
    series: pd.Series,
    *,
    scale: float = 100.0,
) -> GarchResult:
    """Fit GARCH(1,1) on first-differences of the input series.

    Args:
        series: per-bar input (probability levels OR returns; we take Δ).
        scale: rescale Δseries by this factor before MLE for numerical
            stability. Internally rescaled back. Default 100.

    Returns:
        :class:`GarchResult`.

    Raises:
        ValueError: too few observations or non-finite Δseries.
    """
    s = series.dropna()
    n = len(s)
    if n < 50:
        raise ValueError(f"fit_garch_11: need ≥50 bars, got {n}")
    diffs = s.diff().dropna().values
    n_diffs = len(diffs)
    if n_diffs < 30:
        raise ValueError(f"need ≥30 Δ bars, got {n_diffs}")

    # Scale up for numerics.
    eps_scaled = diffs * scale

    # Estimate μ as the mean of Δ (so eps = Δ − μ).
    mu_scaled = float(np.mean(eps_scaled))
    eps_centered = eps_scaled - mu_scaled

    # Initial guesses
    sample_var = float(np.var(eps_centered, ddof=0))
    init = np.array([sample_var * 0.05, 0.10, 0.85])  # ω, α, β
    bounds = [(1e-10, None), (0.0, 1.0), (0.0, 0.999)]

    # MLE
    try:
        res = minimize(
            _negative_log_lik,
            init,
            args=(eps_centered,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
    except Exception as e:
        raise ValueError(f"GARCH(1,1) MLE failed: {e}") from e

    omega_s, alpha, beta = res.x
    persistence = alpha + beta
    converged = bool(res.success)

    # Reconstruct conditional sigma series.
    sigma2 = np.empty(n_diffs)
    sigma2[0] = sample_var
    for t in range(1, n_diffs):
        sigma2[t] = omega_s + alpha * eps_centered[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma_scaled = np.sqrt(sigma2)

    # Forecast σ_{T+1}: σ²_{T+1} = ω + α·ε_T² + β·σ_T²
    last_sigma2_scaled = omega_s + alpha * eps_centered[-1] ** 2 + beta * sigma2[-1]
    last_sigma_scaled = float(np.sqrt(max(last_sigma2_scaled, 0.0)))

    # Rescale back: ε was scaled by scale, so σ scales the same way → divide.
    omega = omega_s / (scale * scale)
    long_run_var = (
        (omega_s / (1.0 - persistence)) / (scale * scale) if persistence < 1.0 else float("inf")
    )
    cond_sigma = sigma_scaled / scale
    last_sigma = last_sigma_scaled / scale

    # Standardised residuals (in original units)
    std_resid = eps_centered / np.maximum(sigma_scaled, 1e-12)

    diff_index = s.index[1:]  # diffs has n-1 entries
    cond_sigma_series = pd.Series(cond_sigma, index=diff_index, name="cond_sigma")
    std_resid_series = pd.Series(std_resid, index=diff_index, name="std_resid")

    return GarchResult(
        mu=mu_scaled / scale,
        omega=omega,
        alpha=float(alpha),
        beta=float(beta),
        persistence=float(persistence),
        long_run_variance=long_run_var,
        log_likelihood=-float(res.fun),
        n_obs=n,
        converged=converged,
        conditional_sigma=cond_sigma_series,
        standardised_residuals=std_resid_series,
        last_sigma=last_sigma,
        is_stationary=bool(persistence < 1.0 and omega > 0),
    )


# ===========================================================================
# Asymmetric extensions: GJR-GARCH(1,1) and EGARCH(1,1)
# ===========================================================================


_DistributionLiteral = Literal["normal", "t", "skewed-t"]
_SQRT_2_OVER_PI = sqrt(2.0 / pi)


# --- shared input prep ------------------------------------------------------


def _prepare_returns(returns: pd.Series, *, min_obs: int = 50) -> tuple[np.ndarray, float]:
    """Validate returns and return centred innovation array + scaling factor.

    Returns are assumed to already be log-returns (so we do not take a
    further difference, unlike :func:`fit_garch_11` which receives a price
    level series). We rescale by ``100`` to keep the optimiser well-
    conditioned and demean.
    """
    s = returns.dropna()
    n = len(s)
    if n < min_obs:
        raise ValueError(f"need >={min_obs} return observations, got {n}")
    arr = s.to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contain non-finite values")
    sample_var = float(np.var(arr, ddof=0))
    if sample_var <= 0.0 or sample_var < 1e-30:
        raise ValueError("returns have zero variance; cannot fit GARCH-family model")
    scale = 100.0
    eps = arr * scale
    eps = eps - float(np.mean(eps))
    return eps, scale


# --- log-likelihood helpers (innovation distributions) ---------------------


def _gauss_loglik_term(eps2: np.ndarray, sigma2: np.ndarray) -> float:
    return -0.5 * float(np.sum(np.log(2.0 * pi * sigma2) + eps2 / sigma2))


def _student_t_loglik_term(eps2: np.ndarray, sigma2: np.ndarray, nu: float) -> float:
    nu = max(float(nu), 2.05)
    log_norm = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * np.log(pi * (nu - 2.0))
    standardised = eps2 / (sigma2 * (nu - 2.0))
    return float(
        np.sum(log_norm - 0.5 * np.log(sigma2) - 0.5 * (nu + 1.0) * np.log1p(standardised))
    )


# --- Ljung-Box on standardised residuals -----------------------------------


def _ljung_box_p(resid: np.ndarray, lags: int = 10) -> float:
    """Best-effort Ljung-Box p-value on standardised residuals.

    Returns ``nan`` if statsmodels is unavailable or signals fail. The
    consumer is expected to handle non-finite p-values gracefully.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        try:
            df = acorr_ljungbox(resid, lags=[lags], return_df=True)
            return float(df["lb_pvalue"].iloc[0])
        except (TypeError, KeyError, ValueError, IndexError, AttributeError):
            try:
                lb = acorr_ljungbox(resid, lags=[lags], return_df=False)
                return float(lb[1][0]) if hasattr(lb[1], "__len__") else float(lb[1])
            except (TypeError, KeyError, ValueError, IndexError, AttributeError):
                return float("nan")
    except Exception:  # pragma: no cover — only triggered if statsmodels missing
        return float("nan")


# --- GJR-GARCH(1,1) --------------------------------------------------------


def _gjr_recursion(
    eps: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> np.ndarray:
    """Run the GJR variance recursion. Returns the σ_t² array."""
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = max(float(np.var(eps, ddof=0)), 1e-12)
    for t in range(1, n):
        e = eps[t - 1]
        ind = 1.0 if e < 0.0 else 0.0
        sigma2[t] = omega + alpha * e * e + gamma * ind * e * e + beta * sigma2[t - 1]
        if not np.isfinite(sigma2[t]) or sigma2[t] <= 0.0:
            sigma2[t] = 1e-12
    return sigma2


def _gjr_neg_loglik(params: np.ndarray, eps: np.ndarray, dist: str) -> float:
    if dist == "skewed-t":
        omega, alpha, gamma, beta, nu, _lam = params
    elif dist == "t":
        omega, alpha, gamma, beta, nu = params
    else:
        omega, alpha, gamma, beta = params
        nu = 0.0
    if omega <= 0 or alpha < 0 or beta < 0:
        return 1e10
    if alpha + 0.5 * gamma + beta >= 0.999:
        return 1e10
    if alpha + gamma < 0:  # need σ²_t > 0 for any sign of ε
        return 1e10
    sigma2 = _gjr_recursion(eps, omega, alpha, gamma, beta)
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e10
    eps2 = eps * eps
    if dist == "normal":
        return -_gauss_loglik_term(eps2, sigma2)
    if dist == "t":
        return -_student_t_loglik_term(eps2, sigma2, float(nu))
    # skewed-t treated as Student-t for likelihood (skew is a mean-shift the
    # POC does not need); kept as separate branch to preserve API surface.
    return -_student_t_loglik_term(eps2, sigma2, float(nu))


def fit_gjr_garch_11(
    returns: pd.Series,
    distribution: _DistributionLiteral = "normal",
) -> dict[str, Any]:
    """Fit GJR-GARCH(1,1) on a series of returns.

    Args:
        returns: Daily returns (log returns recommended). Must have
            non-zero variance and at least 50 observations.
        distribution: Innovation distribution: ``"normal"``, ``"t"`` or
            ``"skewed-t"``. The skewed-t case is the same likelihood
            shape as Student-t in this implementation; the asymmetric
            innovation tail is reserved for a future enhancement.

    Returns:
        Dict with keys: ``omega``, ``alpha``, ``gamma_leverage``,
        ``beta``, ``persistence``, ``half_life_vol_days``,
        ``leverage_effect_significant``, ``asymmetry_t_stat``,
        ``log_likelihood``, ``aic``, ``bic``, ``conditional_variance``,
        ``standardized_residuals``, ``ljung_box_p``, ``converged``,
        ``distribution``, ``n_obs``.

    Raises:
        ValueError: too-few or zero-variance returns.
    """
    if distribution not in ("normal", "t", "skewed-t"):
        raise ValueError(f"unknown distribution: {distribution!r}")

    eps, scale = _prepare_returns(returns)
    n = len(eps)
    sample_var = float(np.var(eps, ddof=0))

    # Initial values: small ω, modest symmetric α + leverage γ, persistent β.
    init_base = [sample_var * 0.05, 0.05, 0.05, 0.85]
    bounds_base: list[tuple[float, float | None]] = [
        (1e-10, None),
        (0.0, 0.999),
        (-0.5, 0.999),
        (0.0, 0.999),
    ]
    if distribution == "t":
        init = [*init_base, 8.0]
        bounds = [*bounds_base, (2.05, 200.0)]
    elif distribution == "skewed-t":
        init = [*init_base, 8.0, 0.0]
        bounds = [*bounds_base, (2.05, 200.0), (-0.95, 0.95)]
    else:
        init = list(init_base)
        bounds = list(bounds_base)

    converged = True
    try:
        res = minimize(
            _gjr_neg_loglik,
            np.asarray(init, dtype=float),
            args=(eps, distribution),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        params = np.asarray(res.x, dtype=float)
        nll = float(res.fun)
        converged = bool(res.success)
        if not np.all(np.isfinite(params)) or nll >= 1e9:
            raise RuntimeError("GJR MLE returned infeasible point")
    except Exception as exc:
        logger.warning("fit_gjr_garch_11 MLE failed (%s); using starting values", exc)
        params = np.asarray(init, dtype=float)
        nll = _gjr_neg_loglik(params, eps, distribution)
        converged = False

    omega_s = float(params[0])
    alpha = float(params[1])
    gamma = float(params[2])
    beta = float(params[3])

    sigma2 = _gjr_recursion(eps, omega_s, alpha, gamma, beta)
    persistence = alpha + 0.5 * gamma + beta

    # Approximate t-stat for γ via finite-difference Hessian along γ axis.
    asym_t = _profile_t_stat_for_gamma(
        params,
        eps,
        distribution,
        gamma_index=2,
        profile_fn=_gjr_neg_loglik,
    )

    std_resid = eps / np.sqrt(np.maximum(sigma2, 1e-30))
    lb_p = _ljung_box_p(std_resid, lags=10)

    cond_var = sigma2 / (scale * scale)

    if 0.0 < persistence < 1.0:
        half_life = float(np.log(0.5) / np.log(max(persistence, 1e-12)))
    else:
        half_life = float("inf")

    n_params = len(params)
    log_lik = -nll
    aic = 2.0 * n_params - 2.0 * log_lik
    bic = float(np.log(n)) * n_params - 2.0 * log_lik

    leverage_significant = bool(np.isfinite(asym_t) and abs(asym_t) > 1.96 and gamma > 0.0)

    return {
        "omega": omega_s / (scale * scale),
        "alpha": alpha,
        "gamma_leverage": gamma,
        "beta": beta,
        "persistence": float(persistence),
        "half_life_vol_days": half_life,
        "leverage_effect_significant": leverage_significant,
        "asymmetry_t_stat": float(asym_t) if np.isfinite(asym_t) else 0.0,
        "log_likelihood": float(log_lik),
        "aic": float(aic),
        "bic": float(bic),
        "conditional_variance": [float(v) for v in cond_var.tolist()],
        "standardized_residuals": [float(v) for v in std_resid.tolist()],
        "ljung_box_p": float(lb_p) if np.isfinite(lb_p) else float("nan"),
        "converged": converged,
        "distribution": distribution,
        "n_obs": int(n),
    }


def _profile_t_stat_for_gamma(
    params: np.ndarray,
    eps: np.ndarray,
    distribution: str,
    *,
    gamma_index: int,
    profile_fn: Any,
    h_rel: float = 1e-3,
) -> float:
    """Crude Hessian-diagonal-based t-stat on the leverage parameter.

    A proper sandwich estimator would invert the full Hessian — for a POC
    we approximate the standard error via the second partial derivative
    along the γ axis only.  Returns ``nan`` if the second derivative is
    non-positive (likelihood not locally concave in γ), in which case
    callers should treat the leverage as unidentified.
    """
    gamma_hat = float(params[gamma_index])
    h = max(abs(gamma_hat) * h_rel, 1e-3)
    p_plus = params.copy()
    p_minus = params.copy()
    p_plus[gamma_index] = gamma_hat + h
    p_minus[gamma_index] = gamma_hat - h
    try:
        f0 = profile_fn(params, eps, distribution)
        fp = profile_fn(p_plus, eps, distribution)
        fm = profile_fn(p_minus, eps, distribution)
    except Exception:
        return float("nan")
    if not all(np.isfinite([f0, fp, fm])) or max(f0, fp, fm) >= 1e9:
        return float("nan")
    # second derivative of -log L w.r.t. γ → information ≈ d²(-logL)/dγ²
    info = (fp - 2.0 * f0 + fm) / (h * h)
    if not np.isfinite(info) or info <= 0.0:
        return float("nan")
    se = float(np.sqrt(1.0 / info))
    if se <= 0.0:
        return float("nan")
    return float(gamma_hat / se)


# --- EGARCH(1,1) -----------------------------------------------------------


def _egarch_recursion(
    eps: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> np.ndarray:
    """Run the EGARCH log-variance recursion. Returns σ_t² (level)."""
    n = len(eps)
    log_sigma2 = np.empty(n)
    log_sigma2[0] = float(np.log(max(np.var(eps, ddof=0), 1e-12)))
    for t in range(1, n):
        sigma_prev = float(np.sqrt(np.exp(log_sigma2[t - 1])))
        if not np.isfinite(sigma_prev) or sigma_prev <= 0.0:
            sigma_prev = 1e-6
        z_prev = eps[t - 1] / sigma_prev
        log_sigma2[t] = (
            omega
            + alpha * (abs(z_prev) - _SQRT_2_OVER_PI)
            + gamma * z_prev
            + beta * log_sigma2[t - 1]
        )
        if not np.isfinite(log_sigma2[t]):
            log_sigma2[t] = log_sigma2[t - 1]
        # cap log-variance to keep optimisation numerically tame
        log_sigma2[t] = min(log_sigma2[t], 50.0)
        log_sigma2[t] = max(log_sigma2[t], -50.0)
    return np.exp(log_sigma2)


def _egarch_neg_loglik(params: np.ndarray, eps: np.ndarray, _dist: str) -> float:
    omega, alpha, gamma, beta = params
    if abs(beta) >= 0.999:
        return 1e10
    sigma2 = _egarch_recursion(eps, omega, alpha, gamma, beta)
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e10
    eps2 = eps * eps
    return -_gauss_loglik_term(eps2, sigma2)


def fit_egarch_11(returns: pd.Series) -> dict[str, Any]:
    """Fit Nelson-style EGARCH(1,1) on a series of returns.

    Args:
        returns: Daily returns (log returns recommended).

    Returns:
        Dict with keys: ``omega``, ``alpha``, ``gamma_leverage``,
        ``beta``, ``persistence``, ``half_life_log_vol_days``,
        ``leverage_negative``, ``log_likelihood``, ``aic``, ``bic``,
        ``conditional_variance``, ``standardized_residuals``,
        ``converged``, ``n_obs``.
    """
    eps, scale = _prepare_returns(returns)
    n = len(eps)

    init = np.array([0.0, 0.10, -0.05, 0.95], dtype=float)
    bounds: list[tuple[float | None, float | None]] = [
        (-10.0, 10.0),
        (-2.0, 2.0),
        (-2.0, 2.0),
        (-0.999, 0.999),
    ]

    converged = True
    try:
        res = minimize(
            _egarch_neg_loglik,
            init,
            args=(eps, "normal"),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        params = np.asarray(res.x, dtype=float)
        nll = float(res.fun)
        converged = bool(res.success)
        if not np.all(np.isfinite(params)) or nll >= 1e9:
            raise RuntimeError("EGARCH MLE returned infeasible point")
    except Exception as exc:
        logger.warning("fit_egarch_11 MLE failed (%s); using starting values", exc)
        params = init.copy()
        nll = _egarch_neg_loglik(params, eps, "normal")
        converged = False

    omega = float(params[0])
    alpha = float(params[1])
    gamma = float(params[2])
    beta = float(params[3])

    sigma2 = _egarch_recursion(eps, omega, alpha, gamma, beta)
    std_resid = eps / np.sqrt(np.maximum(sigma2, 1e-30))
    cond_var = sigma2 / (scale * scale)

    persistence = abs(beta)
    if 0.0 < persistence < 1.0:
        half_life = float(np.log(0.5) / np.log(max(persistence, 1e-12)))
    else:
        half_life = float("inf")

    log_lik = -nll
    n_params = len(params)
    aic = 2.0 * n_params - 2.0 * log_lik
    bic = float(np.log(n)) * n_params - 2.0 * log_lik

    return {
        "omega": omega,
        "alpha": alpha,
        "gamma_leverage": gamma,
        "beta": beta,
        "persistence": float(persistence),
        "half_life_log_vol_days": half_life,
        "leverage_negative": bool(gamma < 0.0),
        "log_likelihood": float(log_lik),
        "aic": float(aic),
        "bic": float(bic),
        "conditional_variance": [float(v) for v in cond_var.tolist()],
        "standardized_residuals": [float(v) for v in std_resid.tolist()],
        "converged": converged,
        "n_obs": int(n),
    }


# --- Symmetric GARCH(1,1) on returns (used by the comparison helper) -------


def _symmetric_neg_loglik(params: np.ndarray, eps: np.ndarray, _dist: str) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = max(float(np.var(eps, ddof=0)), 1e-12)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        if not np.isfinite(sigma2[t]) or sigma2[t] <= 0.0:
            return 1e10
    eps2 = eps * eps
    return -_gauss_loglik_term(eps2, sigma2)


def _fit_symmetric_garch_on_returns(returns: pd.Series) -> dict[str, Any]:
    eps, scale = _prepare_returns(returns)
    n = len(eps)
    sample_var = float(np.var(eps, ddof=0))
    init = np.array([sample_var * 0.05, 0.10, 0.85], dtype=float)
    bounds: list[tuple[float, float | None]] = [
        (1e-10, None),
        (0.0, 0.999),
        (0.0, 0.999),
    ]
    converged = True
    try:
        res = minimize(
            _symmetric_neg_loglik,
            init,
            args=(eps, "normal"),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        params = np.asarray(res.x, dtype=float)
        nll = float(res.fun)
        converged = bool(res.success)
        if not np.all(np.isfinite(params)) or nll >= 1e9:
            raise RuntimeError("GARCH MLE infeasible")
    except Exception as exc:
        logger.warning("symmetric GARCH MLE failed (%s); using starting values", exc)
        params = init.copy()
        nll = _symmetric_neg_loglik(params, eps, "normal")
        converged = False
    omega_s, alpha, beta = (float(x) for x in params)
    persistence = alpha + beta
    log_lik = -nll
    n_params = len(params)
    aic = 2.0 * n_params - 2.0 * log_lik
    bic = float(np.log(n)) * n_params - 2.0 * log_lik
    return {
        "model": "garch11",
        "omega": omega_s / (scale * scale),
        "alpha": alpha,
        "beta": beta,
        "gamma_leverage": 0.0,
        "persistence": float(persistence),
        "log_likelihood": float(log_lik),
        "aic": float(aic),
        "bic": float(bic),
        "converged": converged,
        "n_obs": int(n),
    }


# --- model selection helper ------------------------------------------------


def compare_garch_models(
    returns: pd.Series,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Fit several GARCH-family models and pick the best AIC / BIC.

    Args:
        returns: Daily returns series.
        models: Subset of ``["garch11", "gjr11", "egarch11"]``. Defaults
            to all three.

    Returns:
        Dict with ``best_model_aic``, ``best_model_bic``, and a list of
        per-model summaries with leverage diagnostics.
    """
    valid = ["garch11", "gjr11", "egarch11"]
    chosen = list(models) if models is not None else list(valid)
    bad = [m for m in chosen if m not in valid]
    if bad:
        raise ValueError(f"unknown models: {bad}")
    if not chosen:
        raise ValueError("models list is empty")

    summaries: list[dict[str, Any]] = []
    for name in chosen:
        try:
            if name == "garch11":
                fit = _fit_symmetric_garch_on_returns(returns)
                leverage_p = float("nan")
            elif name == "gjr11":
                fit = fit_gjr_garch_11(returns)
                t = float(fit.get("asymmetry_t_stat", 0.0))
                # two-sided z-test under asymptotic normality of MLE
                from scipy.stats import norm as _norm

                leverage_p = float(2.0 * (1.0 - _norm.cdf(abs(t))))
            else:  # egarch11
                fit = fit_egarch_11(returns)
                # No t-stat exposed; treat leverage_p as nan but keep field.
                leverage_p = float("nan")
            summaries.append(
                {
                    "model": name,
                    "aic": float(fit["aic"]),
                    "bic": float(fit["bic"]),
                    "persistence": float(fit["persistence"]),
                    "leverage_test_p": leverage_p,
                }
            )
        except Exception as exc:
            logger.warning("compare_garch_models: %s failed: %s", name, exc)
            summaries.append(
                {
                    "model": name,
                    "aic": float("inf"),
                    "bic": float("inf"),
                    "persistence": float("nan"),
                    "leverage_test_p": float("nan"),
                }
            )

    finite_summaries = [s for s in summaries if np.isfinite(s["aic"])]
    if not finite_summaries:
        raise ValueError("all GARCH-family fits failed for this return series")
    best_aic = min(finite_summaries, key=lambda s: s["aic"])["model"]
    best_bic = min(finite_summaries, key=lambda s: s["bic"])["model"]
    return {
        "best_model_aic": best_aic,
        "best_model_bic": best_bic,
        "comparisons": summaries,
    }


__all__ = [
    "GarchResult",
    "compare_garch_models",
    "fit_egarch_11",
    "fit_garch_11",
    "fit_gjr_garch_11",
]
