"""Advanced analyses on top of the basic OLS+HAC fit.

This module groups the optional, opt-in analyses that the API can compute
alongside (or instead of) the default OLS+HAC fit:

  * apply_lag        — shift each factor by k days for predictive regression
  * apply_pca        — reduce factors to top-k principal components
  * fit_ridge        — RidgeCV (regularised L2)
  * fit_lasso        — LassoCV (regularised L1, automatic feature selection)
  * fit_quantile     — quantile regression (statsmodels QuantReg)
  * oos_split        — chronological train/test split + R² on test
  * bootstrap_betas  — non-parametric residual bootstrap CIs
  * rolling_betas    — β(t) over a moving window
  * granger          — Granger causality F-tests per lag
  * factor_stationarity — ADF & KPSS on each Δlogit column
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, RidgeCV
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

from pfm.model import DEFAULT_EPSILON, FactorEstimate, ModelStats

# ─────────────  Lag  ─────────────


def apply_lag(X: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Shift every factor column by ``lag`` business days, dropping the head NaNs.

    With ``lag>0`` the regressor at row ``t`` is the factor's value ``lag`` days
    ago — turning the regression from descriptive to predictive.
    """
    if lag <= 0:
        return X
    return X.shift(lag).dropna()


# ─────────────  Resolution-collapse filter  ─────────────


def is_resolving_factor(prices: pd.Series, tail_days: int = 14, max_z: float = 3.0) -> bool:
    """Heuristic: True if the last ``tail_days`` of the series contain extreme
    moves not seen before, suggesting the market is in its resolution-collapse
    phase (price crashing to 0 or 1 as the event resolves).

    These markets are toxic for OOS regression because the test set lands
    exactly on the collapse, blowing up Δlogit and crushing test R².
    """
    s = prices.dropna()
    if len(s) < tail_days * 2:
        return False
    head = s.iloc[:-tail_days]
    tail = s.iloc[-tail_days:]
    if len(head) < 5 or len(tail) < 3:
        return False
    # Rolling Δlogit-equivalent: log(p/(1-p)) differences in head vs tail.
    head_clipped = head.clip(0.01, 0.99)
    tail_clipped = tail.clip(0.01, 0.99)
    head_logit = np.log(head_clipped / (1 - head_clipped))
    tail_logit = np.log(tail_clipped / (1 - tail_clipped))
    head_dl = pd.Series(head_logit, index=head_clipped.index).diff().dropna()
    tail_dl = pd.Series(tail_logit, index=tail_clipped.index).diff().dropna()
    if len(head_dl) < 3 or len(tail_dl) < 2:
        return False
    head_std = head_dl.std()
    if head_std < 1e-6:
        return False
    # Any tail Δlogit beyond max_z × head std is suspicious.
    z = tail_dl.abs() / head_std
    return bool(z.max() > max_z)


# ─────────────  Z-score normalisation  ─────────────


def build_theme_composites(
    factor_series: dict[str, pd.Series],
    theme_lookup: dict[str, str],
) -> dict[str, pd.Series]:
    """Average all (z-scored) factors within each theme into one composite.

    Why: with ~80 obs and 30+ candidates, stepwise overfits. Averaging within
    theme reduces dimensionality dramatically (30+ → ~7) while preserving the
    economic signal. Verified to move 5/10 tickers from "indistinguishable
    from noise" (permutation p > 0.20) to "marginal" (p < 0.10).

    Args:
        factor_series: {factor_id: pd.Series of Δlogit values}
        theme_lookup:  {factor_id: theme_string}

    Returns:
        {f"COMP_{theme}": averaged-z-scored composite Series}
    """
    by_theme: dict[str, list[pd.Series]] = {}
    for fid, s in factor_series.items():
        theme = theme_lookup.get(fid, "other")
        by_theme.setdefault(theme, []).append(s)

    out: dict[str, pd.Series] = {}
    for theme, series_list in by_theme.items():
        zs = []
        for s in series_list:
            sd = float(s.std())
            if sd > 1e-9:
                zs.append((s - s.mean()) / sd)
        if not zs:
            continue
        out[f"COMP_{theme}"] = pd.concat(zs, axis=1).mean(axis=1)
    return out


def zscore_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Per-column z-score: each factor gets mean=0, std=1.

    Why: stepwise / regularised regressions are sensitive to the *scale* of
    each Δlogit. A factor with a 0.4→0.0 collapse has 10× the variance of one
    moving in 0.45→0.55 — so its β looks bigger / smaller without that being
    economically meaningful. Z-scoring puts them on the same footing.
    """
    out = X.copy()
    for col in list(out.columns):
        s = out[col]
        # Guard against duplicate columns returning a DataFrame.
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        sd = float(s.std())
        if sd > 1e-9:
            out[col] = (s - s.mean()) / sd
    return out


# ─────────────  PCA  ─────────────


@dataclass(frozen=True)
class PcaInfo:
    components: list[str]  # ["PC1", "PC2", ...]
    explained_variance: list[float]  # variance ratio per component
    loadings: dict[str, dict[str, float]]  # {factor_id: {PC1: weight, ...}}


def apply_pca(X: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, PcaInfo]:
    """Reduce ``X`` to its top-``n_components`` principal components."""
    k = min(n_components, X.shape[1])
    pca = PCA(n_components=k)
    arr = pca.fit_transform(X.values)
    cols = [f"PC{i + 1}" for i in range(k)]
    df = pd.DataFrame(arr, index=X.index, columns=cols)
    # Loadings: outer key = PC, inner key = original factor → weight.
    loadings: dict[str, dict[str, float]] = {}
    for i, pc in enumerate(cols):
        loadings[pc] = {col: float(pca.components_[i, j]) for j, col in enumerate(X.columns)}
    info = PcaInfo(
        components=cols,
        explained_variance=[float(v) for v in pca.explained_variance_ratio_],
        loadings=loadings,
    )
    return df, info


# ─────────────  Regularised regression  ─────────────


@dataclass(frozen=True)
class RegularizedFit:
    """Lightweight result for ridge/lasso: β only; SE/t/p are not standard."""

    alpha: float
    alpha_grid: list[float]
    intercept: float
    beta: dict[str, float]
    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float
    selected_factors: list[str]  # for lasso, the non-zero set


def _r2(y: npt.ArrayLike, yhat: npt.ArrayLike) -> float:
    y_a = np.asarray(y, dtype=float)
    yhat_a = np.asarray(yhat, dtype=float)
    sse = float(np.sum((y_a - yhat_a) ** 2))
    sst = float(np.sum((y_a - y_a.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else 0.0


def fit_ridge(y: pd.Series, X: pd.DataFrame, cv: int = 5) -> RegularizedFit:
    """Ridge with α chosen by cross-validation."""
    alphas = np.logspace(-3, 2, 25)
    n_splits = max(2, min(cv, max(2, len(y) // 10)))
    model = RidgeCV(alphas=alphas, cv=n_splits)
    model.fit(X.values, y.values)
    yhat = model.predict(X.values)
    return RegularizedFit(
        alpha=float(model.alpha_),
        alpha_grid=[float(a) for a in alphas],
        intercept=float(model.intercept_),
        beta={col: float(model.coef_[i]) for i, col in enumerate(X.columns)},
        fitted_values=yhat,
        residuals=y.values - yhat,
        r_squared=_r2(y.values, yhat),
        selected_factors=list(X.columns),
    )


def fit_lasso(y: pd.Series, X: pd.DataFrame, cv: int = 5) -> RegularizedFit:
    """Lasso with α chosen by cross-validation. Performs feature selection."""
    n_splits = max(2, min(cv, max(2, len(y) // 10)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LassoCV(cv=n_splits, max_iter=20_000)
        model.fit(X.values, y.values)
    yhat = model.predict(X.values)
    selected = [col for col, b in zip(X.columns, model.coef_, strict=True) if abs(b) > 1e-10]
    return RegularizedFit(
        alpha=float(model.alpha_),
        alpha_grid=[float(a) for a in model.alphas_],
        intercept=float(model.intercept_),
        beta={col: float(model.coef_[i]) for i, col in enumerate(X.columns)},
        fitted_values=yhat,
        residuals=y.values - yhat,
        r_squared=_r2(y.values, yhat),
        selected_factors=selected,
    )


# ─────────────  Quantile regression  ─────────────


@dataclass(frozen=True)
class QuantileFit:
    quantile: float
    intercept: float
    beta: dict[str, float]
    std_err: dict[str, float]
    t_stat: dict[str, float]
    p_value: dict[str, float]
    r_squared: float  # pseudo-R² for QuantReg
    fitted_values: np.ndarray
    residuals: np.ndarray


def fit_quantile(y: pd.Series, X: pd.DataFrame, q: float = 0.5) -> QuantileFit:
    """Quantile regression (statsmodels QuantReg). q=0.5 is the median."""
    X_const = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = QuantReg(y.values, X_const.values)
        res = model.fit(q=q)
    cols = list(X_const.columns)
    yhat = res.fittedvalues
    return QuantileFit(
        quantile=q,
        intercept=float(res.params[0]),
        beta={col: float(res.params[i]) for i, col in enumerate(cols) if col != "const"},
        std_err={col: float(res.bse[i]) for i, col in enumerate(cols) if col != "const"},
        t_stat={col: float(res.tvalues[i]) for i, col in enumerate(cols) if col != "const"},
        p_value={col: float(res.pvalues[i]) for i, col in enumerate(cols) if col != "const"},
        r_squared=float(getattr(res, "prsquared", _r2(y.values, yhat))),
        fitted_values=yhat,
        residuals=y.values - yhat,
    )


# ─────────────  Out-of-sample R²  ─────────────


@dataclass(frozen=True)
class OosResult:
    train_n: int
    test_n: int
    train_r2: float
    test_r2: float
    test_dates: list[pd.Timestamp]
    test_observed: list[float]
    test_predicted: list[float]


def oos_split(
    y: pd.Series,
    X: pd.DataFrame,
    test_fraction: float = 0.2,
) -> OosResult | None:
    """Chronological split: first (1−f) is train, last f is test.

    Fits OLS on train, predicts on test, reports both R² values. Returns
    ``None`` if the test set would be too small to be meaningful.
    """
    if not 0.0 < test_fraction < 1.0:
        return None
    n = len(y)
    n_test = max(5, int(n * test_fraction))
    n_train = n - n_test
    if n_train <= X.shape[1] + 1:
        return None

    X_const = sm.add_constant(X, has_constant="add")
    X_train, X_test = X_const.iloc[:n_train].values, X_const.iloc[n_train:].values
    y_train, y_test = y.iloc[:n_train].values, y.iloc[n_train:].values
    fit = sm.OLS(y_train, X_train).fit()
    train_yhat = fit.fittedvalues
    test_yhat = X_test @ fit.params
    return OosResult(
        train_n=n_train,
        test_n=n_test,
        train_r2=_r2(y_train, train_yhat),
        test_r2=_r2(y_test, test_yhat),
        test_dates=list(y.index[n_train:]),
        test_observed=[float(v) for v in y_test],
        test_predicted=[float(v) for v in test_yhat],
    )


# ─────────────  Bootstrap CIs  ─────────────


def bootstrap_betas(
    y: pd.Series,
    X: pd.DataFrame,
    n_iter: int = 500,
    seed: int = 42,
    pct_low: float = 2.5,
    pct_high: float = 97.5,
) -> dict[str, dict[str, float]]:
    """Residual bootstrap: fit OLS, resample residuals, refit, build CIs.

    Returns a dict ``{factor_id: {ci_low, ci_high, mean}}`` of bootstrap
    percentile confidence intervals — robust to non-normal residuals.
    """
    if n_iter < 50:
        return {}
    X_const = sm.add_constant(X, has_constant="add")
    base = sm.OLS(y.values, X_const.values).fit()
    resid = base.resid
    yhat = base.fittedvalues
    rng = np.random.default_rng(seed=seed)
    n = len(y)
    samples = np.empty((n_iter, X_const.shape[1]))
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_boot = yhat + resid[idx]
        try:
            samples[i] = sm.OLS(y_boot, X_const.values).fit().params
        except (np.linalg.LinAlgError, ValueError):
            samples[i] = np.nan
    out: dict[str, dict[str, float]] = {}
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        s = samples[:, i]
        s = s[~np.isnan(s)]
        if not len(s):
            continue
        out[col] = {
            "ci_low": float(np.percentile(s, pct_low)),
            "ci_high": float(np.percentile(s, pct_high)),
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
        }
    return out


# ─────────────  Rolling betas  ─────────────


def rolling_betas(
    y: pd.Series,
    X: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Roll an OLS over ``window`` business days and record β per factor.

    Returns a DataFrame indexed by the END date of each window with one column
    per factor. Useful for visualising regime changes in β(t).
    """
    n = len(y)
    if window <= X.shape[1] + 1 or window > n:
        return pd.DataFrame(columns=X.columns)
    X_const = sm.add_constant(X, has_constant="add")
    rows = []
    idx_dates = []
    for end in range(window, n + 1):
        start = end - window
        try:
            fit = sm.OLS(y.values[start:end], X_const.values[start:end]).fit()
            rows.append(
                {col: fit.params[i] for i, col in enumerate(X_const.columns) if col != "const"}
            )
            idx_dates.append(y.index[end - 1])
        except (np.linalg.LinAlgError, ValueError):
            continue
    if not rows:
        return pd.DataFrame(columns=X.columns)
    return pd.DataFrame(rows, index=idx_dates)


# ─────────────  Granger causality  ─────────────


def granger_test(
    y: pd.Series,
    x: pd.Series,
    max_lag: int = 5,
) -> dict[int, dict[str, float]]:
    """Granger F-test per lag for ``x → y``: does past ``x`` help predict ``y``?

    Returns ``{lag: {f_stat, p_value}}``. Quietly skips lags that fail
    (insufficient observations, perfect collinearity, etc.).
    """
    if max_lag <= 0:
        return {}
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < max_lag * 3 + 5:
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(df[["y", "x"]], maxlag=max_lag, verbose=False)
    except (ValueError, np.linalg.LinAlgError, IndexError):
        return {}
    # statsmodels' grangercausalitytests returns
    #   {lag: ({test_name: (F, p, df_denom, df_num), ...}, [...models])}
    out: dict[int, dict[str, float]] = {}
    for lag, payload in res.items():
        try:
            test_dict = payload[0]
            f_stat, p_value, _, _ = test_dict["ssr_ftest"]
            out[int(lag)] = {"f_stat": float(f_stat), "p_value": float(p_value)}
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


# ─────────────  Per-factor stationarity  ─────────────


def permutation_test(
    y: pd.Series,
    X: pd.DataFrame,
    n_iters: int = 50,
    seed: int = 42,
    test_fraction: float = 0.20,
) -> dict[str, float | list[float]]:
    """Permutation test for ``is the OOS R² distinguishable from random?``.

    Strategy:
        1. Compute the real test R² with the actual factor data.
        2. For each of ``n_iters`` iterations, shuffle each factor's values
           independently (preserves marginal distribution, breaks alignment
           with returns) and re-fit.
        3. Report the empirical p-value: fraction of null R²s ≥ real R².

    Determinism: with the same ``seed``, the null draws are reproducible.

    Returns a dict with keys:
        real_test_r2, null_test_r2s, null_median, null_pct95, null_max,
        p_value, n_iters_completed.
    """
    if len(y) != len(X) or len(y) < 20:
        return {
            "real_test_r2": float("nan"),
            "null_test_r2s": [],
            "null_median": float("nan"),
            "null_pct95": float("nan"),
            "null_max": float("nan"),
            "p_value": float("nan"),
            "n_iters_completed": 0,
        }

    n = len(y)
    n_test = max(5, int(n * test_fraction))
    n_train = n - n_test
    if n_train <= X.shape[1] + 1:
        return {
            "real_test_r2": float("nan"),
            "null_test_r2s": [],
            "null_median": float("nan"),
            "null_pct95": float("nan"),
            "null_max": float("nan"),
            "p_value": float("nan"),
            "n_iters_completed": 0,
        }

    def _split_r2(y_arr: npt.ArrayLike, X_arr: npt.ArrayLike) -> float:
        y_a = np.asarray(y_arr, dtype=float)
        x_a = np.asarray(X_arr, dtype=float)
        X_const = np.column_stack([np.ones(len(x_a)), x_a])
        try:
            fit = sm.OLS(y_a[:n_train], X_const[:n_train]).fit()
            yhat_test = X_const[n_train:] @ fit.params
        except (np.linalg.LinAlgError, ValueError):
            return float("nan")
        sse = float(np.sum((y_a[n_train:] - yhat_test) ** 2))
        sst = float(np.sum((y_a[n_train:] - y_a[n_train:].mean()) ** 2))
        return 1.0 - sse / sst if sst > 0 else 0.0

    y_arr = y.values
    X_arr = X.values
    real_r2 = _split_r2(y_arr, X_arr)

    rng = np.random.default_rng(seed=seed)
    null_r2s: list[float] = []
    for _ in range(n_iters):
        # Independent shuffle per column → breaks alignment with y, preserves
        # marginal distribution per factor.
        X_perm = np.column_stack([rng.permutation(X_arr[:, j]) for j in range(X_arr.shape[1])])
        r2 = _split_r2(y_arr, X_perm)
        if not np.isnan(r2):
            null_r2s.append(float(r2))

    if not null_r2s:
        return {
            "real_test_r2": float(real_r2),
            "null_test_r2s": [],
            "null_median": float("nan"),
            "null_pct95": float("nan"),
            "null_max": float("nan"),
            "p_value": float("nan"),
            "n_iters_completed": 0,
        }

    null_arr = np.array(null_r2s)
    p_val = float((null_arr >= real_r2).mean())
    return {
        "real_test_r2": float(real_r2),
        "null_test_r2s": [float(x) for x in null_arr],
        "null_median": float(np.median(null_arr)),
        "null_pct95": float(np.percentile(null_arr, 95)),
        "null_max": float(np.max(null_arr)),
        "p_value": p_val,
        "n_iters_completed": len(null_r2s),
    }


def factor_stationarity(X: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """ADF + KPSS on each column of ``X`` (each factor's Δlogit series)."""
    out: dict[str, dict[str, float | None]] = {}
    for col in X.columns:
        arr = X[col].dropna().astype(float).values
        rec: dict[str, float | None] = {
            "adf_pvalue": None,
            "kpss_pvalue": None,
        }
        if len(arr) >= 12:
            try:
                _, p, *_ = adfuller(arr, autolag="AIC")
                rec["adf_pvalue"] = float(p)
            except (ValueError, np.linalg.LinAlgError, OverflowError):
                # OverflowError surfaces when the autolag bandwidth formula
                # divides by a singular residual covariance — diagnostic
                # is unestimable but the fit itself is valid.
                pass
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p, *_ = kpss(arr, regression="c", nlags="auto")
                rec["kpss_pvalue"] = float(p)
            except (ValueError, np.linalg.LinAlgError, OverflowError):
                # Same defensive catch — KPSS autolag uses the Andrews
                # bandwidth which overflows on degenerate residuals.
                pass
        out[col] = rec
    return out


# ─────────────  Helper: convert sklearn fit to FactorEstimate-shape  ─────────────


def regularized_to_estimates(
    fit: RegularizedFit,
    bootstrap_ci: dict[str, dict[str, float]] | None = None,
) -> tuple[list[FactorEstimate], ModelStats]:
    """Adapt a ridge/lasso fit to the same ``FactorEstimate`` / ``ModelStats``
    shape used by OLS+HAC, so the API response stays uniform. SE/t/p come from
    the bootstrap if provided, else they are NaN with the field marked.
    """
    n = len(fit.fitted_values)
    k = len([b for b in fit.beta.values() if abs(b) > 1e-12])
    sse = float(np.sum(fit.residuals**2))
    sst = float(
        np.sum(
            (fit.fitted_values + fit.residuals - np.mean(fit.fitted_values + fit.residuals)) ** 2
        )
    )
    r2 = 1 - sse / sst if sst > 0 else 0.0
    r2_adj = 1 - (1 - r2) * (n - 1) / max(1, n - k - 1)

    estimates: list[FactorEstimate] = []
    for col, b in fit.beta.items():
        boot = (bootstrap_ci or {}).get(col)
        if boot:
            estimates.append(
                FactorEstimate(
                    factor_id=col,
                    beta=b,
                    std_err=boot.get("std", float("nan")),
                    t_stat=b / boot["std"] if boot.get("std", 0) > 0 else float("nan"),
                    p_value=float("nan"),  # bootstrap doesn't give a t-test p
                    ci_low=boot["ci_low"],
                    ci_high=boot["ci_high"],
                )
            )
        else:
            estimates.append(
                FactorEstimate(
                    factor_id=col,
                    beta=b,
                    std_err=float("nan"),
                    t_stat=float("nan"),
                    p_value=float("nan"),
                    ci_low=float("nan"),
                    ci_high=float("nan"),
                )
            )

    stats = ModelStats(
        alpha=fit.intercept,
        r_squared=r2,
        r_squared_adj=r2_adj,
        f_stat=float("nan"),
        f_pvalue=float("nan"),
        residual_std=float(np.sqrt(np.mean(fit.residuals**2))),
    )
    return estimates, stats


# Public-ish constant
__all__ = [
    "DEFAULT_EPSILON",
    "OosResult",
    "PcaInfo",
    "QuantileFit",
    "RegularizedFit",
    "apply_lag",
    "apply_pca",
    "bootstrap_betas",
    "factor_stationarity",
    "fit_lasso",
    "fit_quantile",
    "fit_ridge",
    "granger_test",
    "oos_split",
    "regularized_to_estimates",
    "rolling_betas",
]
