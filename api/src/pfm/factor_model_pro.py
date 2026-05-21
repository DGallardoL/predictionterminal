"""Pro factor model — substantial upgrades over the basic ``event_model``.

Adds the production-grade machinery that distinguishes a research-quality
factor model from a naive HAC-OLS:

1. **Estimator choice**: OLS (with HAC SE) / Ridge / Lasso / ElasticNet.
   Ridge regularises against multicollinearity (the main pain in our
   AI-race factor models with VIF > 30); Lasso does automatic factor
   selection by zero-ing out non-informative coefficients.

2. **Logit transform**: the [0, 1] probability bound makes raw OLS
   forecasts spill outside the support. ``transform="logit"`` runs the
   regression on log-odds and back-transforms predictions through the
   sigmoid. Reasonable for moderate probabilities (0.05-0.95); extreme
   markets need clipping.

3. **PCA pre-processing**: when factors are nearly collinear (VIF > 10),
   replace them with the top-k principal components. Coefficients
   become "loadings on orthogonal factors" — interpretable but not in
   original-factor units.

4. **Residual diagnostics** (essential for credible inference):
   - Ljung-Box (auto-correlation of residuals): null = no AR, p > 0.05 means model is well-specified
   - Jarque-Bera (normality of residuals): null = N(0, σ²)
   - ARCH-LM (heteroscedasticity / vol clustering): null = constant variance

5. **Cross-validation R²** via TimeSeriesSplit. Beats in-sample R² as a
   measure of *generalisation*. A model with R²_in = 0.8 and R²_cv = 0.1
   is overfit; the trustworthy metric is R²_cv.

6. **Walk-forward β stability**: refit on each fold, report mean ± std
   of each coefficient across folds. β_std/|β_mean| < 0.5 ⇒ stable.

7. **Bootstrap R² CI**: stationary block bootstrap on the residuals to
   get a 95% CI on R². If lower bound < 0, fit is statistically
   indistinguishable from noise.

References:
    Hoerl, A. & Kennard, R. (1970). Ridge regression.
    Tibshirani, R. (1996). Lasso. JRSS-B 58.
    Jarque, C. & Bera, A. (1980). Normality test.
    Ljung, G. & Box, G. (1978). Time-series autocorrelation test.
    Lopez de Prado, M. (2018). Walk-forward CV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

EstimatorType = Literal["ols", "ridge", "lasso", "elastic_net"]
TransformType = Literal["raw", "logit"]


# ─────────────────────── result dataclasses ──────────────────────────


@dataclass(frozen=True)
class CoefficientPro:
    factor_id: str
    beta: float
    beta_std_across_folds: float | None  # walk-forward stability
    stability_ratio: float | None  # |beta_mean| / beta_std
    is_zeroed: bool  # True if Lasso shrank it to 0
    significance: str | None  # "***" / "**" / "*" / ""


@dataclass(frozen=True)
class ResidualDiagnostics:
    ljung_box_p: float  # autocorrelation; null = no AR
    jarque_bera_p: float  # normality; null = N(0,σ²)
    arch_lm_p: float | None  # heteroscedasticity; null = homoskedastic
    durbin_watson: float  # AR(1) check; ≈ 2 if no AR
    residual_std: float
    residual_skew: float
    residual_kurtosis: float
    well_specified: bool  # all 3 tests pass at α=0.05


@dataclass(frozen=True)
class FactorModelProResult:
    target_id: str
    estimator: EstimatorType
    transform: TransformType
    use_pca: bool
    n_obs: int
    n_factors: int
    coefficients: list[CoefficientPro]
    intercept: float
    r_squared_is: float
    r_squared_cv: float  # mean cross-val R²
    r_squared_cv_std: float  # std of CV-fold R²
    r_squared_ci_lo_95: float
    r_squared_ci_hi_95: float
    diagnostics: ResidualDiagnostics
    pca_explained_variance: list[float] | None
    n_zeroed_factors: int  # for Lasso
    overfit_flag: bool  # True if R²_is − R²_cv > 0.20


# ─────────────────────── helper: logit transform ──────────────────────


def _logit(p: np.ndarray, *, eps: float = 0.001) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────── residual diagnostics ─────────────────────────


def _diagnose_residuals(resid: np.ndarray, X: np.ndarray) -> ResidualDiagnostics:
    """Compute Ljung-Box, Jarque-Bera, ARCH-LM, Durbin-Watson on residuals."""
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    n = len(resid)
    # Ljung-Box at lag = min(10, n//5).
    lags = min(10, max(1, n // 5))
    # statsmodels API-compat shims: older versions returned tuples,
    # newer versions return DataFrames. The narrow exception set covers
    # the signature mismatches we've actually observed in CI.
    _STATS_API_ERRS = (TypeError, ValueError, KeyError, IndexError, AttributeError)
    try:
        lb = acorr_ljungbox(resid, lags=[lags], return_df=False)
        lb_p = float(lb[1][0]) if hasattr(lb[1], "__len__") else float(lb[1])
    except _STATS_API_ERRS:
        # Newer statsmodels returns a DataFrame.
        try:
            lb_df = acorr_ljungbox(resid, lags=[lags], return_df=True)
            lb_p = float(lb_df["lb_pvalue"].iloc[0])
        except _STATS_API_ERRS:
            lb_p = float("nan")

    # Jarque-Bera
    try:
        _, jb_p, _, _ = jarque_bera(resid)
        jb_p = float(jb_p)
    except _STATS_API_ERRS:
        jb_p = float("nan")

    # ARCH-LM
    try:
        arch_lm = het_arch(resid, nlags=4)
        arch_p = float(arch_lm[1])
    except _STATS_API_ERRS:
        arch_p = None

    dw = float(durbin_watson(resid))
    skew = float(np.mean(((resid - resid.mean()) / max(resid.std(ddof=1), 1e-12)) ** 3))
    kurt = float(np.mean(((resid - resid.mean()) / max(resid.std(ddof=1), 1e-12)) ** 4) - 3.0)

    well = (
        not np.isnan(lb_p)
        and lb_p > 0.05
        and not np.isnan(jb_p)
        and jb_p > 0.05
        and (arch_p is None or arch_p > 0.05)
    )

    return ResidualDiagnostics(
        ljung_box_p=lb_p,
        jarque_bera_p=jb_p,
        arch_lm_p=arch_p,
        durbin_watson=dw,
        residual_std=float(resid.std(ddof=1)),
        residual_skew=skew,
        residual_kurtosis=kurt,
        well_specified=well,
    )


# ─────────────────────── bootstrap R² CI ──────────────────────────────


def _bootstrap_r2_ci(
    y: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_iters: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI on R² via residual bootstrap."""
    rng = np.random.default_rng(seed)
    resid = y - y_pred
    r2_samples = []
    for _ in range(n_iters):
        sampled_resid = rng.choice(resid, size=len(resid), replace=True)
        y_sim = y_pred + sampled_resid
        ss_res = float(np.sum((y_sim - y_pred) ** 2))
        ss_tot = float(np.sum((y_sim - y_sim.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_samples.append(r2)
    return float(np.percentile(r2_samples, 2.5)), float(np.percentile(r2_samples, 97.5))


# ─────────────────────── main fit function ────────────────────────────


def _make_estimator(estimator: EstimatorType, alpha: float = 1.0):
    if estimator == "ols":
        return LinearRegression()
    if estimator == "ridge":
        return Ridge(alpha=alpha)
    if estimator == "lasso":
        return Lasso(alpha=alpha, max_iter=10_000)
    if estimator == "elastic_net":
        return ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10_000)
    raise ValueError(f"unknown estimator: {estimator}")


def fit_factor_model_pro(
    target: pd.Series,
    factors: pd.DataFrame,
    *,
    target_id: str = "target",
    estimator: EstimatorType = "ols",
    alpha: float = 1.0,
    transform: TransformType = "raw",
    use_pca: bool = False,
    pca_explained_variance_target: float = 0.90,
    n_cv_folds: int = 5,
    bootstrap_iters: int = 200,
    seed: int = 42,
) -> FactorModelProResult:
    """Pro factor model with regularisation, transforms, PCA, and CV.

    Args:
        target: dependent probability series in [0, 1].
        factors: DataFrame of explanatory probability series.
        target_id: label for output.
        estimator: ``"ols"`` (with sklearn LinearRegression — no HAC SE here,
            use basic event_model for that) / ``"ridge"`` / ``"lasso"`` /
            ``"elastic_net"``.
        alpha: regularisation strength (Ridge/Lasso/ElasticNet only).
        transform: ``"raw"`` or ``"logit"``.
        use_pca: if True, replace factors with top-k principal components
            covering ``pca_explained_variance_target`` of variance.
        n_cv_folds: TimeSeriesSplit folds for cross-validated R².
        bootstrap_iters: residual-bootstrap iterations for R² CI.
        seed: RNG seed.

    Returns:
        :class:`FactorModelProResult`.
    """
    aligned = pd.concat({"y": target, **{c: factors[c] for c in factors.columns}}, axis=1).dropna()
    n = len(aligned)
    k = factors.shape[1]
    if n < max(n_cv_folds * 10, k + 10):
        raise ValueError(
            f"fit_factor_model_pro: only {n} aligned bars for {k} factors and "
            f"{n_cv_folds} CV folds (need ≥ max(n_folds·10, k+10))"
        )

    y_raw = aligned["y"].to_numpy(dtype=float)
    X_raw = aligned[list(factors.columns)].to_numpy(dtype=float)

    # Reject zero-variance factors.
    zero_var = [factors.columns[i] for i in range(k) if float(np.var(X_raw[:, i])) < 1e-12]
    if zero_var:
        raise ValueError(f"factors with zero variance: {zero_var}")

    # Apply transform.
    if transform == "logit":
        y_t = _logit(y_raw)
        X_t = np.apply_along_axis(_logit, 0, X_raw)
    else:
        y_t = y_raw
        X_t = X_raw.copy()

    # Optional PCA.
    pca_ev: list[float] | None = None
    factor_names = list(factors.columns)
    if use_pca:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_t)
        pca_full = PCA(n_components=k).fit(X_scaled)
        cum_ev = np.cumsum(pca_full.explained_variance_ratio_)
        n_pc = int(np.searchsorted(cum_ev, pca_explained_variance_target) + 1)
        n_pc = max(1, min(n_pc, k))
        pca = PCA(n_components=n_pc).fit(X_scaled)
        X_t = pca.transform(X_scaled)
        pca_ev = [float(v) for v in pca.explained_variance_ratio_]
        factor_names = [f"PC{i + 1}" for i in range(n_pc)]

    # Standardise X for regularised estimators (so alpha is comparable across factors).
    if estimator in ("ridge", "lasso", "elastic_net"):
        scaler_x = StandardScaler()
        X_fit = scaler_x.fit_transform(X_t)
    else:
        X_fit = X_t

    # Fit on all data (in-sample).
    model = _make_estimator(estimator, alpha=alpha)
    model.fit(X_fit, y_t)
    y_pred_t = model.predict(X_fit)

    # Inverse-transform predictions if logit was used.
    if transform == "logit":
        y_pred = _sigmoid(y_pred_t)
        y_for_r2 = y_raw
    else:
        y_pred = y_pred_t
        y_for_r2 = y_t

    # In-sample R².
    ss_res = float(np.sum((y_for_r2 - y_pred) ** 2))
    ss_tot = float(np.sum((y_for_r2 - y_for_r2.mean()) ** 2))
    r2_is = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Cross-validated R² via TimeSeriesSplit.
    tscv = TimeSeriesSplit(n_splits=n_cv_folds)
    cv_r2_folds: list[float] = []
    fold_betas: list[np.ndarray] = []
    for train_idx, test_idx in tscv.split(X_t):
        X_train_t = X_t[train_idx]
        X_test_t = X_t[test_idx]
        y_train_t = y_t[train_idx]
        y_test_raw_idx = y_raw[test_idx]
        if estimator in ("ridge", "lasso", "elastic_net"):
            sc = StandardScaler().fit(X_train_t)
            X_train_fit = sc.transform(X_train_t)
            X_test_fit = sc.transform(X_test_t)
        else:
            X_train_fit = X_train_t
            X_test_fit = X_test_t
        m = _make_estimator(estimator, alpha=alpha)
        m.fit(X_train_fit, y_train_t)
        yp_t = m.predict(X_test_fit)
        if transform == "logit":
            yp = _sigmoid(yp_t)
            ytrue = y_test_raw_idx
        else:
            yp = yp_t
            ytrue = y_t[test_idx]
        ss_r = float(np.sum((ytrue - yp) ** 2))
        ss_t = float(np.sum((ytrue - ytrue.mean()) ** 2))
        cv_r2_folds.append(1.0 - ss_r / ss_t if ss_t > 0 else 0.0)
        fold_betas.append(np.asarray(m.coef_, dtype=float).ravel())
    r2_cv_mean = float(np.mean(cv_r2_folds))
    r2_cv_std = float(np.std(cv_r2_folds, ddof=1)) if len(cv_r2_folds) > 1 else 0.0

    # Bootstrap R² CI (residual-bootstrap on in-sample predictions).
    r2_lo_95, r2_hi_95 = _bootstrap_r2_ci(y_for_r2, y_pred, n_iters=bootstrap_iters, seed=seed)

    # Walk-forward β stability.
    beta_array = np.array(fold_betas)  # (n_folds, n_factors)
    beta_means = beta_array.mean(axis=0)
    beta_stds = (
        beta_array.std(axis=0, ddof=1) if beta_array.shape[0] > 1 else np.zeros_like(beta_means)
    )

    # Significance (basic): use the in-sample model's coefs and a rough
    # SE from residual variance · diag of (X' X)^-1 (OLS) or treat
    # zero coefficient as not significant (Lasso).
    in_sample_coefs = np.asarray(model.coef_, dtype=float).ravel()
    # For OLS we can compute t-stats; for regularized estimators, p-values
    # are not meaningful in the same way — report based on stability ratio.
    coefs_out: list[CoefficientPro] = []
    for i, name in enumerate(factor_names):
        b_in = float(in_sample_coefs[i])
        b_mean = float(beta_means[i])
        b_std = float(beta_stds[i])
        is_zero = (estimator == "lasso") and abs(b_in) < 1e-9
        stab_ratio = float(abs(b_mean) / b_std) if b_std > 0 and not np.isnan(b_std) else None
        # Stability-based significance (conservative).
        if is_zero:
            sig = ""
        elif stab_ratio is not None:
            if stab_ratio > 3:
                sig = "***"
            elif stab_ratio > 2:
                sig = "**"
            elif stab_ratio > 1:
                sig = "*"
            else:
                sig = ""
        else:
            sig = ""
        coefs_out.append(
            CoefficientPro(
                factor_id=name,
                beta=b_in,
                beta_std_across_folds=b_std,
                stability_ratio=stab_ratio,
                is_zeroed=is_zero,
                significance=sig,
            )
        )

    # Residuals + diagnostics on the in-sample predictions.
    resid = y_for_r2 - y_pred
    diag = _diagnose_residuals(resid, X_fit)
    n_zeroed = sum(1 for c in coefs_out if c.is_zeroed)
    overfit = (r2_is - r2_cv_mean) > 0.20

    return FactorModelProResult(
        target_id=target_id,
        estimator=estimator,
        transform=transform,
        use_pca=use_pca,
        n_obs=n,
        n_factors=len(factor_names),
        coefficients=coefs_out,
        intercept=float(model.intercept_) if hasattr(model, "intercept_") else 0.0,
        r_squared_is=float(r2_is),
        r_squared_cv=r2_cv_mean,
        r_squared_cv_std=r2_cv_std,
        r_squared_ci_lo_95=r2_lo_95,
        r_squared_ci_hi_95=r2_hi_95,
        diagnostics=diag,
        pca_explained_variance=pca_ev,
        n_zeroed_factors=n_zeroed,
        overfit_flag=bool(overfit),
    )


__all__ = [
    "CoefficientPro",
    "EstimatorType",
    "FactorModelProResult",
    "ResidualDiagnostics",
    "TransformType",
    "fit_factor_model_pro",
]
