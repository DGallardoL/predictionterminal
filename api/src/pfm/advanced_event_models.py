"""Advanced event-conditioned factor models.

This module extends the linear OLS-HAC factor model in :mod:`pfm.model`
with several richer specifications that exploit the special structure of
prediction-market factors:

A) Conditional model — separate :math:`(\\alpha, \\beta)` per probability
   regime of the factor (e.g. ``p < 30%`` / ``30% \\le p < 70%`` / ``p
   \\ge 70%``). Captures the empirical fact that equity sensitivity to
   prediction-market news is regime-dependent.

B) Polynomial / non-linear model — fits

   .. math::
       r_t = \\alpha + \\sum_{k=1}^{d} \\beta_k (\\Delta logit_t)^k + \\epsilon_t

   with HAC SEs. The likelihood-ratio test against the linear (``d=1``)
   nested model and an AIC-selected optimal degree are reported.

C) Markov regime-switching — Hamilton (1989) two-state mean+variance with
   the prediction-market :math:`\\Delta logit` as switching regressor.
   Reuses :func:`pfm.advanced_strategies.markov_regime_switching` where
   possible.

D) VECM — Johansen-tested cointegration between :math:`\\log P_{equity}`
   and the (clipped) :math:`logit(p)` series. If cointegrated, fit a
   one-cointegration-vector VECM and surface the loading vector and the
   error-correction half-life.

E) GARCH-X — augments :func:`pfm.garch.fit_garch_11` with the factor as
   exogenous regressor on the conditional variance recursion.

F) Tail-dependence — empirical lower- and upper-tail conditional
   probabilities (a non-parametric copula-style measure).

All functions are pure: they take pre-fetched probability and return
series and return :class:`dict` payloads that the router layer JSONifies.
The data-IO path lives in
:mod:`pfm.advanced_event_models_router`.
"""

from __future__ import annotations

import warnings
from math import log as mlog

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from pfm.model import DEFAULT_EPSILON, delta_logit, hac_lag_andrews, logit_transform

# ---------------------------------------------------------------------------
# A) Conditional factor model
# ---------------------------------------------------------------------------


def fit_conditional_model_core(
    returns: pd.Series,
    probabilities: pd.Series,
    *,
    conditioning_thresholds: list[float],
    epsilon: float = DEFAULT_EPSILON,
    ticker: str = "TICKER",
    factor_id: str = "factor",
) -> dict:
    """Fit a separate OLS-HAC model in each probability bucket.

    Args:
        returns: log returns of the equity, indexed by UTC date.
        probabilities: prediction-market YES probabilities in [0, 1],
            same calendar as ``returns`` (intersected internally).
        conditioning_thresholds: list of cut points in (0, 1), strictly
            ascending. Buckets are ``[0, t_1), [t_1, t_2), …, [t_K, 1]``.
        epsilon: clipping bound for the logit transform of the factor.
        ticker, factor_id: used only for output labelling.

    Returns:
        dict with shape::

            {
                "ticker": ..., "factor_id": ...,
                "buckets": [
                    {"range": [lo, hi], "n_obs": n,
                     "alpha": a, "beta": b, "t_stat": t, "p_value": p,
                     "r_squared": r2, "ci_low": l, "ci_high": h},
                    ...
                ],
                "homoscedasticity_test_p": float | None,
                "n_obs": int,
            }
    """
    if not conditioning_thresholds:
        raise ValueError("conditioning_thresholds must be non-empty")
    th = sorted(set(conditioning_thresholds))
    if any(not 0.0 < t < 1.0 for t in th):
        raise ValueError("conditioning_thresholds must lie in (0, 1)")

    df = pd.concat({"r": returns, "p": probabilities}, axis=1).dropna()
    if len(df) < 30:
        raise ValueError(f"need >=30 joint obs, got {len(df)}")

    df["dlogit"] = delta_logit(df["p"], epsilon=epsilon)
    df = df.dropna()

    # Build buckets [0, t1), [t1, t2), ..., [tK, 1].
    edges = [0.0, *th, 1.0]
    buckets: list[dict] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        # Inclusive on left, exclusive on right; final bucket inclusive on right.
        if i == len(edges) - 2:
            mask = (df["p"] >= lo) & (df["p"] <= hi)
        else:
            mask = (df["p"] >= lo) & (df["p"] < hi)
        sub = df.loc[mask]
        if len(sub) < 10:
            buckets.append(
                {
                    "range": [float(lo), float(hi)],
                    "n_obs": int(len(sub)),
                    "alpha": None,
                    "beta": None,
                    "std_err": None,
                    "t_stat": None,
                    "p_value": None,
                    "ci_low": None,
                    "ci_high": None,
                    "r_squared": None,
                }
            )
            continue
        X = sm.add_constant(sub[["dlogit"]].values, has_constant="add")
        y = sub["r"].values
        lag = hac_lag_andrews(len(sub))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        ci = res.conf_int()
        buckets.append(
            {
                "range": [float(lo), float(hi)],
                "n_obs": int(len(sub)),
                "alpha": float(res.params[0]),
                "beta": float(res.params[1]),
                "std_err": float(res.bse[1]),
                "t_stat": float(res.tvalues[1]),
                "p_value": float(res.pvalues[1]),
                "ci_low": float(ci[1, 0]),
                "ci_high": float(ci[1, 1]),
                "r_squared": float(res.rsquared),
            }
        )

    # Pooled Breusch-Pagan as a (rough) homoscedasticity gauge — heteroscedasticity
    # across buckets is exactly what the conditional model is set up to capture.
    homoscedasticity_p: float | None
    try:
        x_pooled = sm.add_constant(df[["dlogit"]].values, has_constant="add")
        pooled = sm.OLS(df["r"].values, x_pooled).fit()
        bp = het_breuschpagan(pooled.resid, x_pooled)
        homoscedasticity_p = float(bp[1])
    except (ValueError, np.linalg.LinAlgError):
        homoscedasticity_p = None

    return {
        "ticker": ticker,
        "factor_id": factor_id,
        "buckets": buckets,
        "homoscedasticity_test_p": homoscedasticity_p,
        "n_obs": int(len(df)),
        "epsilon": epsilon,
    }


# ---------------------------------------------------------------------------
# B) Polynomial / non-linear factor model
# ---------------------------------------------------------------------------


def fit_polynomial_factor_model_core(
    returns: pd.Series,
    probabilities: pd.Series,
    *,
    degree: int = 2,
    epsilon: float = DEFAULT_EPSILON,
    ticker: str = "TICKER",
    factor_id: str = "factor",
    aic_max_degree: int = 5,
) -> dict:
    """Fit ``r = alpha + sum_k beta_k * (dlogit)^k`` with HAC SEs.

    Returns betas, t-stats, R², an LR test against the nested linear
    model, the AIC-optimal degree, and a marginal-effect grid
    ``dy/dx`` for plotting.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    aic_max_degree = max(aic_max_degree, degree)

    df = pd.concat({"r": returns, "p": probabilities}, axis=1).dropna()
    df["dlogit"] = delta_logit(df["p"], epsilon=epsilon)
    df = df.dropna()
    n = len(df)
    if n < max(30, degree + 5):
        raise ValueError(f"need >=max(30, degree+5), got {n}")

    x = df["dlogit"].values
    y = df["r"].values
    lag = hac_lag_andrews(n)

    def _fit(d: int) -> sm.regression.linear_model.RegressionResultsWrapper:
        cols = np.column_stack([x**k for k in range(1, d + 1)])
        Xc = sm.add_constant(cols, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lag})

    res = _fit(degree)

    # LR-style F test against the linear nested model (HAC robust): use the
    # statsmodels f_test on the joint zero restriction of beta_2..beta_d.
    lr_p: float | None
    if degree > 1:
        # Restriction matrix R such that R · beta = 0 for the higher-order coefs.
        # params order: [const, b1, b2, ..., bd] -> indices 2..d
        R = np.zeros((degree - 1, degree + 1))
        for i in range(degree - 1):
            R[i, 2 + i] = 1.0
        try:
            ftest = res.f_test(R)
            lr_p = float(np.asarray(ftest.pvalue).item())
        except (ValueError, np.linalg.LinAlgError):
            lr_p = None
    else:
        lr_p = None

    # AIC sweep: degrees 1..aic_max_degree.
    aic_pairs: list[tuple[int, float]] = []
    for d in range(1, aic_max_degree + 1):
        try:
            r = _fit(d)
            aic_pairs.append((d, float(r.aic)))
        except (ValueError, np.linalg.LinAlgError):
            continue
    optimal_degree = min(aic_pairs, key=lambda pair: pair[1])[0] if aic_pairs else degree

    # Marginal effect dy/dx evaluated on a grid spanning the empirical Δlogit.
    grid_lo = float(np.quantile(x, 0.02))
    grid_hi = float(np.quantile(x, 0.98))
    if grid_hi <= grid_lo:
        grid_lo, grid_hi = float(np.min(x)), float(np.max(x))
    grid = np.linspace(grid_lo, grid_hi, 21)
    betas = np.asarray(res.params, dtype=float)
    # dy/dx = sum_k k * beta_k * x^(k-1)  (params[1:] are b1..bd)
    derivative = np.zeros_like(grid)
    for k in range(1, degree + 1):
        derivative = derivative + k * betas[k] * (grid ** (k - 1))

    coef_payload = []
    ci = res.conf_int()
    for k in range(1, degree + 1):
        coef_payload.append(
            {
                "order": int(k),
                "beta": float(betas[k]),
                "std_err": float(res.bse[k]),
                "t_stat": float(res.tvalues[k]),
                "p_value": float(res.pvalues[k]),
                "ci_low": float(ci[k, 0]),
                "ci_high": float(ci[k, 1]),
            }
        )

    return {
        "ticker": ticker,
        "factor_id": factor_id,
        "degree": int(degree),
        "alpha": float(betas[0]),
        "betas": coef_payload,
        "r_squared": float(res.rsquared),
        "r_squared_adj": float(res.rsquared_adj),
        "vs_linear_lr_test_p": lr_p,
        "optimal_degree_aic": int(optimal_degree),
        "aic_by_degree": [{"degree": d, "aic": a} for d, a in aic_pairs],
        "marginal_effects": [
            {"x": float(xi), "dy_dx": float(yi)} for xi, yi in zip(grid, derivative, strict=True)
        ],
        "n_obs": int(n),
        "hac_lag": int(lag),
    }


# ---------------------------------------------------------------------------
# C) Markov regime-switching factor model
# ---------------------------------------------------------------------------


def fit_regime_switching_model_core(
    returns: pd.Series,
    probabilities: pd.Series,
    *,
    n_regimes: int = 2,
    epsilon: float = DEFAULT_EPSILON,
    ticker: str = "TICKER",
    factor_id: str = "factor",
    max_iter: int = 100,
) -> dict:
    """Fit a Markov-switching regression :math:`r_t = \\alpha_{s_t} +
    \\beta_{s_t} \\Delta logit_t + \\sigma_{s_t} \\epsilon_t`.

    Reports per-regime coefficients, transition probabilities, the
    ergodic distribution, and the smoothed P(state=k) for the last 30
    observations.
    """
    if n_regimes < 2 or n_regimes > 4:
        raise ValueError("n_regimes must be in [2, 4]")

    df = pd.concat({"r": returns, "p": probabilities}, axis=1).dropna()
    df["dlogit"] = delta_logit(df["p"], epsilon=epsilon)
    df = df.dropna()
    if len(df) < 60:
        raise ValueError(f"need >=60 joint obs for regime-switching, got {len(df)}")

    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    y = df["r"].values
    exog = df[["dlogit"]].values
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovRegression(
                y,
                k_regimes=n_regimes,
                exog=exog,
                switching_variance=True,
                switching_exog=True,
            )
            res = model.fit(disp=False, maxiter=max_iter)
    except Exception as e:  # statsmodels can raise many flavours
        raise ValueError(f"Markov fit failed: {e}") from e

    params = np.asarray(res.params, dtype=float)
    # Parameter layout for MarkovRegression with switching_variance=True and
    # switching_exog=True (k regimes, m exog columns):
    #   [k*(k-1) transition probs,
    #    k constants,
    #    k * m exog coefficients,
    #    k variances]
    n_trans = n_regimes * (n_regimes - 1)
    const_off = n_trans
    exog_off = const_off + n_regimes
    var_off = exog_off + n_regimes  # m=1 here (single dlogit)

    means = [float(params[const_off + k]) for k in range(n_regimes)]
    betas = [float(params[exog_off + k]) for k in range(n_regimes)]
    variances = [float(max(params[var_off + k], 1e-12)) for k in range(n_regimes)]
    stds = [float(np.sqrt(v)) for v in variances]

    # Transition matrix as P[i, j] = P(s_t = j | s_{t-1} = i).
    rt = np.asarray(res.regime_transition)
    if rt.ndim == 3:
        rt = rt[:, :, 0]
    transition_matrix = [[float(rt[i, j]) for j in range(n_regimes)] for i in range(n_regimes)]

    # Ergodic distribution = stationary left-eigenvector of P (eigenvalue 1).
    try:
        evals, evecs = np.linalg.eig(rt.T)
        # closest-to-1 eigenvalue
        idx = int(np.argmin(np.abs(evals - 1.0)))
        v = np.real(evecs[:, idx])
        if v.sum() == 0:
            ergodic = [1.0 / n_regimes] * n_regimes
        else:
            v = v / v.sum()
            ergodic = [float(max(x, 0.0)) for x in v]
            # renormalise after clipping any small negatives
            s = sum(ergodic)
            ergodic = [x / s for x in ergodic] if s > 0 else [1.0 / n_regimes] * n_regimes
    except np.linalg.LinAlgError:
        ergodic = [1.0 / n_regimes] * n_regimes

    smp = np.asarray(res.smoothed_marginal_probabilities)
    # statsmodels returns shape (n_obs, k_regimes) typically.
    if smp.ndim == 2 and smp.shape[1] == n_regimes:
        smp_last = smp[-30:, :]
    elif smp.ndim == 2 and smp.shape[0] == n_regimes:
        smp_last = smp.T[-30:, :]
    else:
        smp_last = np.zeros((min(30, len(df)), n_regimes))

    # Order regimes by mean return so the API output is stable across runs.
    order = sorted(range(n_regimes), key=lambda k: means[k])
    regimes = []
    for new_idx, old_idx in enumerate(order):
        regimes.append(
            {
                "idx": int(new_idx),
                "mean_return": float(means[old_idx]),
                "beta_factor": float(betas[old_idx]),
                "std": float(stds[old_idx]),
                "variance": float(variances[old_idx]),
                "ergodic_prob": float(ergodic[old_idx]),
            }
        )

    # Re-permute the transition matrix and smoothed probs into the new ordering.
    perm = np.array(order, dtype=int)
    tm_perm = rt[np.ix_(perm, perm)]
    transition_matrix = [[float(tm_perm[i, j]) for j in range(n_regimes)] for i in range(n_regimes)]
    smp_last_perm = smp_last[:, perm] if smp_last.shape[1] == n_regimes else smp_last
    smoothed_last_30 = [
        [float(smp_last_perm[t, k]) for k in range(n_regimes)]
        for t in range(smp_last_perm.shape[0])
    ]

    return {
        "ticker": ticker,
        "factor_id": factor_id,
        "n_regimes": int(n_regimes),
        "regimes": regimes,
        "transition_matrix": transition_matrix,
        "smoothed_state_probs_last_30": smoothed_last_30,
        "log_likelihood": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "n_obs": int(len(df)),
    }


# ---------------------------------------------------------------------------
# D) VECM — cointegration of log(equity) and logit(factor)
# ---------------------------------------------------------------------------


def fit_vecm_core(
    equity_prices: pd.Series,
    probabilities: pd.Series,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
    epsilon: float = DEFAULT_EPSILON,
    ticker: str = "TICKER",
    factor_id: str = "factor",
) -> dict:
    """Test cointegration between :math:`\\log P_{equity}` and
    :math:`logit(p)`. If significant, fit a one-vector VECM and return
    the loading vector and the half-life of error correction.

    Args:
        equity_prices: positive equity close prices.
        probabilities: factor YES probabilities in (0, 1).
        det_order: deterministic-trend assumption. ``-1`` no const,
            ``0`` const (default), ``1`` linear.
        k_ar_diff: lag order in the VAR-in-differences. Default 1.

    Returns:
        Payload with Johansen p-values (approx, via critical-value
        comparison), loading vectors, and the implied half-life.
    """
    if k_ar_diff < 1:
        raise ValueError("k_ar_diff must be >= 1")

    log_eq = np.log(equity_prices.dropna()).rename("log_eq")
    logit_p = logit_transform(probabilities.dropna(), epsilon=epsilon).rename("logit_p")
    df = pd.concat([log_eq, logit_p], axis=1).dropna()
    n = len(df)
    if n < max(40, k_ar_diff + 10):
        raise ValueError(f"need >=40 joint obs for VECM, got {n}")

    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joh = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stats = [float(x) for x in joh.lr1]
    eigen_stats = [float(x) for x in joh.lr2]
    trace_crit_95 = [float(x) for x in joh.cvt[:, 1]]
    eigen_crit_95 = [float(x) for x in joh.cvm[:, 1]]

    # Continuous p-values via MacKinnon-Haug-Michelis (1999) Gamma response
    # surface. Replaces the previous bucketed lookup (>=0.10 / <0.05 / <0.01).
    from pfm.mhm_critical import johansen_pvalue

    n_series = int(df.shape[1])
    trace_p = johansen_pvalue(
        trace_stats[0],
        n_vars=n_series,
        det_order=det_order,
        test="trace",
        r0=0,
    )
    eigen_p = johansen_pvalue(
        eigen_stats[0],
        n_vars=n_series,
        det_order=det_order,
        test="eigen",
        r0=0,
    )

    # Cointegrated if either statistic for r=0 exceeds its 95% critical value.
    is_coint = bool(trace_stats[0] > trace_crit_95[0] or eigen_stats[0] > eigen_crit_95[0])

    beta_long_run: float | None = None
    alpha_loading_target: float | None = None
    alpha_loading_factor: float | None = None
    half_life: float | None = None

    if is_coint:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vecm = VECM(
                    df.values,
                    k_ar_diff=k_ar_diff,
                    coint_rank=1,
                    deterministic="ci" if det_order == 0 else ("nc" if det_order == -1 else "co"),
                )
                vres = vecm.fit()
            # beta is (k, r); for k=2, r=1 it's a 2x1 cointegration vector.
            beta = np.asarray(vres.beta).flatten()
            # Normalise so first coordinate = 1 (target = log_eq).
            if abs(beta[0]) > 1e-12:
                beta_norm = beta / beta[0]
            else:
                beta_norm = beta
            beta_long_run = float(beta_norm[1]) if len(beta_norm) > 1 else None

            alpha = np.asarray(vres.alpha).flatten()
            alpha_loading_target = float(alpha[0])
            alpha_loading_factor = float(alpha[1]) if len(alpha) > 1 else None
            # Half-life of EC adjustment on the *target* equation.
            # |alpha_target| close to 0 -> slow adjustment; close to 1 -> fast.
            if alpha_loading_target is not None and -2.0 < alpha_loading_target < 0.0:
                rho = 1.0 + alpha_loading_target
                if 0.0 < rho < 1.0:
                    half_life = float(-mlog(2.0) / mlog(rho))
        except (ValueError, np.linalg.LinAlgError):
            pass

    return {
        "ticker": ticker,
        "factor_id": factor_id,
        "n_obs": int(n),
        "det_order": int(det_order),
        "k_ar_diff": int(k_ar_diff),
        "johansen_p_eigenvalue": float(eigen_p),
        "johansen_p_trace": float(trace_p),
        "johansen_eigen_stat": float(eigen_stats[0]),
        "johansen_trace_stat": float(trace_stats[0]),
        "johansen_eigen_crit_95": float(eigen_crit_95[0]),
        "johansen_trace_crit_95": float(trace_crit_95[0]),
        "is_cointegrated": bool(is_coint),
        "beta_long_run": beta_long_run,
        "alpha_loading_target": alpha_loading_target,
        "alpha_loading_factor": alpha_loading_factor,
        "half_life_correction_days": half_life,
    }


# ---------------------------------------------------------------------------
# E) GARCH-X — conditional variance with prediction-market signal as exog
# ---------------------------------------------------------------------------


def fit_garch_x_core(
    returns: pd.Series,
    probabilities: pd.Series,
    *,
    epsilon: float = DEFAULT_EPSILON,
    ticker: str = "TICKER",
    factor_id: str = "factor",
    scale: float = 100.0,
) -> dict:
    """Fit GARCH(1,1)-X with the absolute prediction-market Δlogit as
    exogenous variance regressor:

        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
                  + gamma * |dlogit_t|

    Pure scipy MLE. Compares to the no-X GARCH(1,1) likelihood and
    reports the share of conditional variance attributable to the factor.
    """
    df = pd.concat({"r": returns, "p": probabilities}, axis=1).dropna()
    df["dlogit"] = delta_logit(df["p"], epsilon=epsilon).abs()
    df = df.dropna()
    n = len(df)
    if n < 60:
        raise ValueError(f"need >=60 obs for GARCH-X, got {n}")

    from scipy.optimize import minimize

    eps = (df["r"].values - df["r"].mean()) * scale
    x = df["dlogit"].values
    sample_var = float(np.var(eps, ddof=0))
    if sample_var <= 0:
        raise ValueError("zero-variance returns")

    def _nll(params: np.ndarray) -> float:
        omega, a, b, gamma = params
        if omega <= 0 or a < 0 or b < 0 or a + b >= 1.0:
            return 1e10
        sigma2 = np.empty(n)
        sigma2[0] = sample_var
        for t in range(1, n):
            sigma2[t] = omega + a * eps[t - 1] ** 2 + b * sigma2[t - 1] + gamma * x[t]
            if sigma2[t] <= 0:
                return 1e10
        return float(0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps * eps / sigma2))

    init = np.array([sample_var * 0.05, 0.10, 0.85, 0.0])
    bounds = [(1e-10, None), (0.0, 1.0), (0.0, 0.999), (0.0, None)]
    res = minimize(_nll, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    omega_s, a, b, gamma = res.x
    persistence = a + b

    # Reconstruct conditional variance to compute the variance contribution.
    sigma2 = np.empty(n)
    sigma2[0] = sample_var
    for t in range(1, n):
        sigma2[t] = omega_s + a * eps[t - 1] ** 2 + b * sigma2[t - 1] + gamma * x[t]

    # Decompose mean(sigma2) = omega + a·E[eps^2] + b·E[sigma2] + gamma·E[X].
    var_share_factor: float
    mean_sigma2 = float(np.mean(sigma2))
    if mean_sigma2 > 0:
        contribution_factor = float(gamma * np.mean(x))
        var_share_factor = max(0.0, min(1.0, contribution_factor / mean_sigma2))
    else:
        var_share_factor = 0.0

    half_life_vol_days: float | None
    if 0.0 < persistence < 1.0:
        half_life_vol_days = float(-mlog(2.0) / mlog(persistence))
    else:
        half_life_vol_days = None

    # Rescale ω and γ back to original return units (variance scales by scale^2).
    omega_orig = float(omega_s / (scale * scale))
    gamma_orig = float(gamma / (scale * scale))

    return {
        "ticker": ticker,
        "factor_id": factor_id,
        "n_obs": int(n),
        "omega": omega_orig,
        "alpha": float(a),
        "beta": float(b),
        "factor_exogenous_coef": gamma_orig,
        "persistence": float(persistence),
        "is_stationary": bool(persistence < 1.0),
        "half_life_vol_days": half_life_vol_days,
        "log_likelihood": float(-res.fun),
        "converged": bool(res.success),
        "conditional_variance_explained_by_factor_pct": float(100.0 * var_share_factor),
    }


# ---------------------------------------------------------------------------
# F) Tail dependence (empirical copula)
# ---------------------------------------------------------------------------


def compute_tail_dependence_core(
    equity_returns: pd.Series,
    factor_dlogit: pd.Series,
    *,
    quantile: float = 0.05,
    ticker: str = "TICKER",
    factor_id: str = "factor",
) -> dict:
    """Empirical lower- and upper-tail dependence coefficients.

    .. math::
        \\hat\\lambda_L(q) = \\hat P(F_R^{-1}(q) | F_X^{-1}(q))
                          = \\frac{|R_t < r_q,\\ X_t < x_q|}{|X_t < x_q|}

    Compared to the under-independence baseline ``q``. A ratio
    ``> 2`` indicates substantial joint left-tail clustering.
    """
    if not 0.0 < quantile < 0.5:
        raise ValueError(f"quantile must be in (0, 0.5), got {quantile}")

    df = pd.concat({"r": equity_returns, "x": factor_dlogit}, axis=1).dropna()
    n = len(df)
    if n < 50:
        raise ValueError(f"need >=50 joint obs for tail dependence, got {n}")

    r_lo = float(df["r"].quantile(quantile))
    r_hi = float(df["r"].quantile(1.0 - quantile))
    x_lo = float(df["x"].quantile(quantile))
    x_hi = float(df["x"].quantile(1.0 - quantile))

    n_x_lo = int((df["x"] <= x_lo).sum())
    n_x_hi = int((df["x"] >= x_hi).sum())
    n_joint_lo = int(((df["r"] <= r_lo) & (df["x"] <= x_lo)).sum())
    n_joint_hi = int(((df["r"] >= r_hi) & (df["x"] >= x_hi)).sum())

    lower = float(n_joint_lo / n_x_lo) if n_x_lo > 0 else 0.0
    upper = float(n_joint_hi / n_x_hi) if n_x_hi > 0 else 0.0

    # Asymmetry: > 0 means lower-tail clustering dominates.
    asymmetry = float(lower - upper)

    return {
        "ticker": ticker,
        "factor_id": factor_id,
        "n_obs": int(n),
        "quantile": float(quantile),
        "lower_tail_dependence": lower,
        "upper_tail_dependence": upper,
        "asymmetry": asymmetry,
        "lower_ratio_vs_independence": float(lower / quantile) if quantile > 0 else 0.0,
        "upper_ratio_vs_independence": float(upper / quantile) if quantile > 0 else 0.0,
        "n_extreme_obs_lower": int(n_x_lo),
        "n_extreme_obs_upper": int(n_x_hi),
    }


__all__ = [
    "compute_tail_dependence_core",
    "fit_conditional_model_core",
    "fit_garch_x_core",
    "fit_polynomial_factor_model_core",
    "fit_regime_switching_model_core",
    "fit_vecm_core",
]
