"""Core quant: logit transform, ΔLogit, OLS with HAC standard errors, diagnostics.

The model fitted by the API is

    r_{j,t} = α_j + Σ_i β_{j,i} · Δlogit(p_{i,t}) + ε_{j,t}

with a HAC covariance estimator using automatic bandwidth selection:

    lag = floor(4 · (T/100)^(2/9))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss

DEFAULT_EPSILON: float = 0.01


def logit_transform(p: pd.Series | np.ndarray, epsilon: float = DEFAULT_EPSILON) -> pd.Series:
    """Apply logit transform to probabilities, clipping to [epsilon, 1 - epsilon].

    Args:
        p: Probabilities in [0, 1].
        epsilon: Clipping bound. Must satisfy 0 < epsilon < 0.5.

    Returns:
        Series of logit(p) values, same index as ``p`` if it was a Series.
    """
    if not 0.0 < epsilon < 0.5:
        raise ValueError(f"epsilon must be in (0, 0.5), got {epsilon}")
    series = pd.Series(p) if not isinstance(p, pd.Series) else p
    clipped = series.clip(lower=epsilon, upper=1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def delta_logit(
    prices: pd.Series | np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.Series:
    """First difference of the logit-transformed probability series.

    The first observation has no predecessor, so the result starts at index 1
    of the input. Callers are expected to align the resulting series with the
    return series before fitting (e.g. via ``df.dropna()``).

    Guardrail
    ---------
    If the input series carries values clearly outside ``[0, 1]`` (e.g. a
    BLS / FRED level series fed in by mistake) we emit a warning and fall
    back to a plain first difference instead of forcing a logit. The clip
    inside :func:`logit_transform` would otherwise mask the problem and
    return all-zero ``Δlogit`` values silently. Use
    :func:`delta_level` directly when the non-probability path is intended.
    """
    series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
    finite = series.dropna()
    if len(finite) > 0:
        lo = float(finite.min())
        hi = float(finite.max())
        # Allow a tiny floating-point margin so genuine ``[0, 1]`` series
        # don't trigger this branch.
        if lo < -1e-6 or hi > 1.0 + 1e-6:
            warnings.warn(
                "delta_logit received series outside [0, 1] "
                f"(min={lo:.4g}, max={hi:.4g}); falling back to plain first "
                "differences. Set is_probability=False on the factor or call "
                "delta_level() directly to silence this warning.",
                stacklevel=2,
            )
            return series.astype(float).diff()
    logits = logit_transform(prices, epsilon=epsilon)
    return logits.diff()


def count_clipping_events(
    prices: pd.Series | np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> int:
    """Count observations where the logit clip would be binding.

    A clip is "binding" when the input price lies outside
    ``(epsilon, 1 - epsilon)`` — i.e. the Δlogit step at that point would
    be zero (or near-zero) because the clip flattens both the numerator and
    denominator of the logit transform. Reporting this lets the caller
    detect when ``epsilon`` is silently masking signal at the tails.

    Returns 0 for non-probability series (values outside ``[0, 1]``) since
    the logit branch isn't used for those — :func:`delta_logit` falls back
    to plain differencing.
    """
    if not 0.0 < epsilon < 0.5:
        return 0
    series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
    finite = series.dropna()
    if len(finite) == 0:
        return 0
    lo = float(finite.min())
    hi = float(finite.max())
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        return 0  # not a probability series; plain diff is used.
    binding = (finite < epsilon) | (finite > 1.0 - epsilon)
    return int(binding.sum())


def delta_level(
    levels: pd.Series | np.ndarray,
) -> pd.Series:
    """Plain first-difference for non-probability factors (BLS/FRED).

    Mirrors :func:`delta_logit` so callers can dispatch on
    ``FactorConfig.is_probability`` without branching their code.
    Use this for level series — yields, indices, claim counts — where a
    logit transform is not meaningful. The first row is ``NaN`` and is
    typically dropped by the caller's ``df.dropna()``.
    """
    series = pd.Series(levels) if not isinstance(levels, pd.Series) else levels
    return series.astype(float).diff()


def hac_lag_andrews(n_obs: int) -> int:
    """Automatic bandwidth selection: ``floor(4 · (T/100)^(2/9))``.

    Floored at 1 so that even short windows get at least one lag of correction.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    raw = 4.0 * (n_obs / 100.0) ** (2.0 / 9.0)
    return max(1, int(np.floor(raw)))


@dataclass(frozen=True)
class FactorEstimate:
    """One estimated coefficient with HAC inference."""

    factor_id: str
    beta: float
    std_err: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class ModelStats:
    """Summary of the fitted regression."""

    alpha: float
    r_squared: float
    r_squared_adj: float
    f_stat: float
    f_pvalue: float
    residual_std: float


@dataclass(frozen=True)
class Diagnostics:
    """Auxiliary diagnostics — not used in inference but reported to the user."""

    vif: dict[str, float]
    durbin_watson: float
    hac_lag: int
    # Stationarity of the residual ("spread") — flags spurious regressions.
    # ADF null = unit root (non-stationary); reject (small p) ⇒ stationary.
    # KPSS null = stationary; reject (small p) ⇒ unit root.
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float


@dataclass(frozen=True)
class FitResult:
    """Bundle of estimates + stats + diagnostics + the fitted statsmodels object.

    The statsmodels object is exposed so attribution.py can recover predictions
    and residuals without re-running the fit.
    """

    factors: list[FactorEstimate]
    stats: ModelStats
    diagnostics: Diagnostics
    fitted: sm.regression.linear_model.RegressionResultsWrapper


RegressionType = Literal["ols", "hac"]


def fit_ols_hac(
    y: pd.Series,
    X: pd.DataFrame,
    hac_lag: int | None = None,
    regression: RegressionType = "hac",
) -> FitResult:
    """Fit OLS with the chosen covariance estimator.

    Args:
        y: Dependent variable (e.g. log returns), length T.
        X: Regressor matrix with one column per factor (Δlogit values).
            Must NOT contain a constant — one is added internally.
        hac_lag: HAC lag. Used only when ``regression='hac'``. If
            ``None``, uses automatic bandwidth selection.
        regression: ``'hac'`` (HAC standard errors) or ``'ols'`` (classic non-robust
            covariance — fast and naive, useful for sensitivity checks).

    Returns:
        ``FitResult`` carrying coefficients, fit stats, diagnostics, and the
        underlying statsmodels results object.

    Raises:
        ValueError: if ``y`` and ``X`` have mismatched length, or if there are
            fewer observations than regressors + 1.
    """
    if len(y) != len(X):
        raise ValueError(f"length mismatch: len(y)={len(y)} vs len(X)={len(X)}")
    if len(y) <= X.shape[1] + 1:
        raise ValueError(f"too few observations ({len(y)}) for {X.shape[1]} factors + intercept")

    n_obs = len(y)
    lag = hac_lag if hac_lag is not None else hac_lag_andrews(n_obs)
    # Guard against pathological HAC bandwidth: when ``maxlags`` approaches
    # ``n_obs`` the HAC kernel becomes degenerate and statsmodels
    # produces SEs that are numerically meaningless (often inflating to Inf
    # or returning NaN p-values). The automatic bandwidth is bounded above by
    # ``n_obs`` by construction, but a user-supplied ``hac_lag`` is not.
    if regression == "hac" and lag >= n_obs - 1:
        raise ValueError(
            f"hac_lag={lag} too large for n_obs={n_obs}; must satisfy hac_lag < n_obs - 1"
        )

    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y.values, X_const.values, hasconst=True)
    if regression == "hac":
        fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    else:
        fitted = model.fit()  # classic OLS, non-robust SEs

    # Map statsmodels' positional output back to factor names.
    column_names = list(X_const.columns)
    params = dict(zip(column_names, fitted.params, strict=True))
    bse = dict(zip(column_names, fitted.bse, strict=True))
    tvalues = dict(zip(column_names, fitted.tvalues, strict=True))
    pvalues = dict(zip(column_names, fitted.pvalues, strict=True))
    ci = fitted.conf_int()  # shape (k, 2)
    ci_map = {col: (ci[i, 0], ci[i, 1]) for i, col in enumerate(column_names)}

    factor_estimates = [
        FactorEstimate(
            factor_id=col,
            beta=float(params[col]),
            std_err=float(bse[col]),
            t_stat=float(tvalues[col]),
            p_value=float(pvalues[col]),
            ci_low=float(ci_map[col][0]),
            ci_high=float(ci_map[col][1]),
        )
        for col in column_names
        if col != "const"
    ]

    stats = ModelStats(
        alpha=float(params["const"]),
        r_squared=float(fitted.rsquared),
        r_squared_adj=float(fitted.rsquared_adj),
        f_stat=float(fitted.fvalue),
        f_pvalue=float(fitted.f_pvalue),
        residual_std=float(np.sqrt(fitted.mse_resid)),
    )

    diagnostics = compute_diagnostics(X, fitted, hac_lag=lag)

    return FitResult(
        factors=factor_estimates,
        stats=stats,
        diagnostics=diagnostics,
        fitted=fitted,
    )


def stationarity_tests(residuals: np.ndarray | pd.Series) -> dict[str, float]:
    """Run ADF and KPSS on the regression residuals.

    Why: if the residual ("spread") is non-stationary, the regression is
    spurious — high R² can come from two random walks moving together. ADF
    and KPSS have opposite null hypotheses, so concordance between them is
    informative:

      - ADF rejects + KPSS doesn't reject  → stationary (confident)
      - ADF doesn't reject + KPSS rejects  → unit root  (confident)
      - both reject / neither rejects      → ambiguous

    Returns NaNs when the sample is too short to run a meaningful test.
    """
    arr = pd.Series(residuals).dropna().astype(float).values
    out = {
        "adf_stat": float("nan"),
        "adf_pvalue": float("nan"),
        "kpss_stat": float("nan"),
        "kpss_pvalue": float("nan"),
    }
    if len(arr) < 12:
        return out
    try:
        adf_stat, adf_p, *_ = adfuller(arr, autolag="AIC")
        out["adf_stat"] = float(adf_stat)
        out["adf_pvalue"] = float(adf_p)
    except (ValueError, np.linalg.LinAlgError, OverflowError):
        # OverflowError fires when statsmodels' autolag bandwidth calc gets
        # gamma_hat=inf (residual covariance is singular). The diagnostic
        # is not estimable — return NaN, let the regression itself succeed.
        pass
    # KPSS warns when its small-sample p-value lookup hits the boundary [0.01,
    # 0.10] — the test is still valid, the warning just says "p is at least /
    # at most this value". Silence it so server logs stay tidy.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(arr, regression="c", nlags="auto")
        out["kpss_stat"] = float(kpss_stat)
        out["kpss_pvalue"] = float(kpss_p)
    except (ValueError, np.linalg.LinAlgError, OverflowError):
        # Same defensive catch: degenerate residuals make the Andrews
        # bandwidth formula overflow before the test runs.
        pass
    return out


#: Sentinel for "VIF is mathematically infinite (perfect or near-perfect
#: collinearity)". We cap at 1e9 instead of returning Inf so JSON encoders
#: don't silently turn it into ``null`` and hide the diagnostic from the
#: caller. Anything ≥ this value should be read as "unbounded" by clients.
VIF_INF_SENTINEL: float = 1e9


def compute_diagnostics(
    X: pd.DataFrame,
    fitted: sm.regression.linear_model.RegressionResultsWrapper,
    hac_lag: int,
) -> Diagnostics:
    """Compute VIF, Durbin-Watson, and stationarity tests on residuals."""
    vif: dict[str, float] = {}
    if X.shape[1] >= 2:
        X_const = sm.add_constant(X, has_constant="add")
        with warnings.catch_warnings():
            # Suppress the "divide by zero" RuntimeWarning emitted by
            # statsmodels when a regressor is collinear — we surface that
            # condition via the sentinel value below, which is the actual
            # signal callers should look at.
            warnings.simplefilter("ignore", RuntimeWarning)
            for i, col in enumerate(X_const.columns):
                if col == "const":
                    continue
                raw = float(variance_inflation_factor(X_const.values, i))
                if not np.isfinite(raw) or raw > VIF_INF_SENTINEL:
                    vif[col] = VIF_INF_SENTINEL
                else:
                    vif[col] = raw
    else:
        # VIF is undefined with a single regressor — report 1.0 by convention.
        for col in X.columns:
            vif[col] = 1.0

    stat = stationarity_tests(fitted.resid)

    return Diagnostics(
        vif=vif,
        durbin_watson=float(durbin_watson(fitted.resid)),
        hac_lag=hac_lag,
        adf_stat=stat["adf_stat"],
        adf_pvalue=stat["adf_pvalue"],
        kpss_stat=stat["kpss_stat"],
        kpss_pvalue=stat["kpss_pvalue"],
    )
