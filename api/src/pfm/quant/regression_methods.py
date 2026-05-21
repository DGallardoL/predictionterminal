"""Signature-only stubs for alternative regression methods.

Companion to ``docs/regression-methodology-improvements.md`` (task T79).

These three functions are the picks worth shipping behind a ``method=`` query
parameter on ``POST /fit``:

1. :func:`fit_bayes_conjugate` — Bayesian linear regression with a conjugate
   normal-inverse-gamma prior. Cheap (analytic posterior), honest CIs in the
   small-n regime where frequentist t-stats are unreliable.
2. :func:`fit_elastic_net` — Elastic Net (LASSO + Ridge). The auto-selector
   for users with many correlated factor candidates.
3. :func:`fit_quantile` — Quantile regression at one or more τ values. Reveals
   tail asymmetry that OLS averages away.

NONE of these are implemented. Each raises ``NotImplementedError``. Bodies
will be filled in once the design is approved; the signatures here are the
contract callers can build against. Tests, schema entries, and endpoint
plumbing should NOT be added until the implementations land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BayesResult:
    """Posterior summary from a conjugate Bayesian linear regression.

    Attributes:
        posterior_mean: Posterior mean of beta. Shape (p,).
        posterior_cov: Posterior covariance of beta. Shape (p, p).
        credible_intervals: 95% equal-tail credible intervals, one row per
            factor: array of shape (p, 2) with columns (lo, hi).
        prob_positive: For each factor, ``P(beta_j > 0 | y, X)``. Shape (p,).
        sigma2_posterior_mean: Posterior mean of the residual variance.
        log_marginal_likelihood: Analytic log marginal likelihood under the
            conjugate prior, useful for model comparison.
        prior: Echo of the prior used, for reproducibility.
        n_obs: Number of observations after dropna.
    """

    posterior_mean: np.ndarray
    posterior_cov: np.ndarray
    credible_intervals: np.ndarray
    prob_positive: np.ndarray
    sigma2_posterior_mean: float
    log_marginal_likelihood: float
    prior: dict[str, object]
    n_obs: int


@dataclass(frozen=True, slots=True)
class ElasticNetResult:
    """Fit summary from an Elastic Net with optional CV regularisation path.

    Attributes:
        coefficients: Estimated beta on the *original* (un-standardised) scale,
            indexed by factor name.
        intercept: Estimated intercept on the original scale.
        selected_factors: Names of factors with non-zero coefficients.
        optimal_lambda: Chosen regularisation strength (only meaningful if
            ``lambda_`` was ``"auto"``; otherwise echoes the user's value).
        optimal_alpha: Chosen L1 mixing fraction (same caveat).
        regularisation_path: List of dicts with keys ``lambda``, ``r2_cv``,
            ``n_selected``. Empty when CV is disabled.
        r_squared_cv: 5-fold ``TimeSeriesSplit`` CV R-squared at the chosen
            ``(lambda, alpha)``.
        n_obs: Number of observations after dropna.
    """

    coefficients: dict[str, float]
    intercept: float
    selected_factors: list[str]
    optimal_lambda: float
    optimal_alpha: float
    regularisation_path: list[dict[str, float]]
    r_squared_cv: float
    n_obs: int


@dataclass(frozen=True, slots=True)
class QuantileResult:
    """Fit summary from quantile regression across one or more tau values.

    Attributes:
        taus: Quantile levels actually fitted (e.g. ``[0.1, 0.5, 0.9]``).
        coefficients_by_quantile: DataFrame with one row per factor and one
            column per tau in ``taus``.
        bootstrap_cis: Dict keyed by tau, each value an ``(p, 2)`` array of
            (lo, hi) 95% paired-bootstrap CIs. Empty if bootstrap was off.
        pseudo_r2_by_quantile: Koenker-Machado pseudo R-squared per tau.
        tail_asymmetry: ``beta(tau_high) - beta(tau_low)`` per factor, where
            (tau_low, tau_high) default to (0.1, 0.9). Series indexed by
            factor name.
        n_obs: Number of observations after dropna.
        n_bootstrap: Number of bootstrap replicates actually run.
    """

    taus: list[float]
    coefficients_by_quantile: pd.DataFrame
    bootstrap_cis: dict[float, np.ndarray]
    pseudo_r2_by_quantile: dict[float, float]
    tail_asymmetry: pd.Series
    n_obs: int
    n_bootstrap: int


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BayesianResult:
    """Posterior summary in a dict-friendly shape (task W12-24 contract).

    This is the more ergonomic surface intended for tests, notebooks, and any
    caller that prefers per-coefficient dicts over numpy arrays. See
    :class:`BayesResult` for the array-shaped result used by the API router.

    Attributes:
        posterior_mean: ``E[beta_j | y, X]`` keyed by feature name (including
            ``intercept``).
        credible_intervals: 95% equal-tail credible intervals as
            ``{name: (lo, hi)}`` keyed by feature name.
        posterior_samples: Draws from the joint posterior. Shape
            ``(n_samples, p)``; columns aligned with ``feature_names``. ``None``
            if ``n_samples == 0`` (sampling disabled).
        feature_names: Ordered list of column names matching
            ``posterior_samples`` columns. First entry is ``"intercept"``.
        sigma2_mean: ``E[sigma^2 | y, X]``.
        log_marginal_likelihood: Analytic log marginal likelihood under the
            conjugate prior (Bishop §3.5, eq. 3.86 generalised), useful for
            model comparison. ``None`` if it could not be computed.
    """

    posterior_mean: dict[str, float]
    credible_intervals: dict[str, tuple[float, float]]
    posterior_samples: np.ndarray | None
    feature_names: list[str]
    sigma2_mean: float
    log_marginal_likelihood: float | None


# ---------------------------------------------------------------------------
# Bayesian linear regression — conjugate normal-inverse-gamma posterior
# ---------------------------------------------------------------------------


def _coerce_xy(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Coerce ``(y, X)`` to numpy arrays + feature names, dropping NaN rows."""

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X))
        X.columns = [f"x{i}" for i in range(X.shape[1])]
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y))

    if len(y) != len(X):
        raise ValueError(f"len(y)={len(y)} != len(X)={len(X)}")

    joined = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(joined) == 0:
        raise ValueError("no non-NaN rows in (y, X) after dropna")
    y_arr = joined["__y__"].to_numpy(dtype=float)
    x_df = joined.drop(columns="__y__")
    feature_names = [str(c) for c in x_df.columns]
    return y_arr, x_df.to_numpy(dtype=float), feature_names


def fit_bayes_conjugate(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    prior_mean: np.ndarray | None = None,
    prior_precision: np.ndarray | None = None,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    practical_threshold: float = 0.005,
) -> BayesResult:
    """Bayesian linear regression with a normal-inverse-gamma conjugate prior.

    Posterior is analytic (no MCMC). Returns posterior mean, covariance,
    95% equal-tail credible intervals, and per-factor ``P(beta_j > 0)``.

    The model (Bishop 2006 §3.5):

        beta | sigma^2 ~ N(mu_0, sigma^2 * Lambda_0^{-1})
        sigma^2 ~ InvGamma(a_0, b_0)
        y | beta, sigma^2 ~ N(X beta, sigma^2 I)

    Closed-form posterior:

        Lambda_n = Lambda_0 + X^T X
        mu_n     = Lambda_n^{-1} (Lambda_0 mu_0 + X^T y)
        a_n      = a_0 + n/2
        b_n      = b_0 + 0.5 (y^T y + mu_0^T Lambda_0 mu_0 - mu_n^T Lambda_n mu_n)

    Marginal posterior for beta is a multivariate Student-t with location
    ``mu_n``, scale ``(b_n / a_n) * Lambda_n^{-1}``, and ``2 * a_n`` df.

    Args:
        y: Target series (typically log returns).
        X: Design matrix; column names are factor identifiers. Must NOT
            include an intercept column — it is added internally and reported
            separately at index 0.
        prior_mean: Prior mean ``mu_0`` for beta (including the intercept slot
            at index 0). Defaults to a zero vector.
        prior_precision: Prior precision ``Lambda_0`` for beta. Defaults to
            ``0.01 * I`` (weakly informative).
        prior_a: ``InvGamma`` shape parameter for sigma^2. Default 1.0.
        prior_b: ``InvGamma`` scale parameter for sigma^2. Default 1.0.
        practical_threshold: Reserved for downstream ``prob_practical`` use.

    Returns:
        :class:`BayesResult` with posterior summaries. The first row/column
        of array fields corresponds to the intercept.

    Raises:
        ValueError: On degenerate inputs (mismatched lengths, non-positive
            ``prior_a``/``prior_b``, empty data after dropna, or wrong-shape
            prior arrays).
    """

    if prior_a <= 0:
        raise ValueError(f"prior_a must be > 0, got {prior_a}")
    if prior_b <= 0:
        raise ValueError(f"prior_b must be > 0, got {prior_b}")
    _ = practical_threshold  # reserved for downstream use

    y_arr, x_arr, feature_names = _coerce_xy(y, X)
    n = x_arr.shape[0]

    # Prepend intercept column.
    x_design = np.column_stack([np.ones(n), x_arr])
    p = x_design.shape[1]
    names_with_intercept = ["intercept", *feature_names]

    # Resolve prior mean.
    if prior_mean is None:
        mu_0 = np.zeros(p)
    else:
        mu_0 = np.asarray(prior_mean, dtype=float).reshape(-1)
        if mu_0.shape[0] != p:
            raise ValueError(
                f"prior_mean must have length {p} (1 intercept + {p - 1} features), "
                f"got {mu_0.shape[0]}"
            )

    # Resolve prior precision.
    if prior_precision is None:
        lambda_0 = 0.01 * np.eye(p)
    else:
        lambda_0 = np.asarray(prior_precision, dtype=float)
        if lambda_0.shape != (p, p):
            raise ValueError(f"prior_precision must have shape ({p}, {p}), got {lambda_0.shape}")

    # ------------------------------------------------------------------
    # Conjugate posterior (closed form).
    # ------------------------------------------------------------------
    xtx = x_design.T @ x_design
    xty = x_design.T @ y_arr

    lambda_n = lambda_0 + xtx
    rhs = lambda_0 @ mu_0 + xty
    mu_n = np.linalg.solve(lambda_n, rhs)

    a_n = prior_a + n / 2.0
    yty = float(y_arr @ y_arr)
    quad_prior = float(mu_0 @ lambda_0 @ mu_0)
    quad_post = float(mu_n @ lambda_n @ mu_n)
    b_n = prior_b + 0.5 * (yty + quad_prior - quad_post)
    b_n = max(b_n, 1e-12)  # numerical safety

    sigma2_mean = float(b_n / (a_n - 1.0)) if a_n > 1.0 else float("inf")

    df = 2.0 * a_n
    lambda_n_inv = np.linalg.inv(lambda_n)
    if df > 2.0:
        cov = (b_n / a_n) * (df / (df - 2.0)) * lambda_n_inv
    else:
        cov = (b_n / a_n) * lambda_n_inv

    from scipy import stats

    scale_diag = np.sqrt(np.maximum(np.diag((b_n / a_n) * lambda_n_inv), 0.0))
    t_crit = stats.t.ppf(0.975, df=df)
    ci_lo = mu_n - t_crit * scale_diag
    ci_hi = mu_n + t_crit * scale_diag
    credible = np.column_stack([ci_lo, ci_hi])

    safe_scale = np.where(scale_diag > 0, scale_diag, 1.0)
    z = -mu_n / safe_scale
    prob_positive = 1.0 - stats.t.cdf(z, df=df)
    degenerate = scale_diag <= 0
    prob_positive = np.where(degenerate, (mu_n > 0).astype(float), prob_positive)

    sign0, logdet_l0 = np.linalg.slogdet(lambda_0)
    signn, logdet_ln = np.linalg.slogdet(lambda_n)
    if sign0 <= 0 or signn <= 0:
        log_marg = float("nan")
    else:
        from math import lgamma, log, pi

        log_marg = (
            0.5 * logdet_l0
            - 0.5 * logdet_ln
            + prior_a * log(prior_b)
            - a_n * log(b_n)
            + lgamma(a_n)
            - lgamma(prior_a)
            - (n / 2.0) * log(2.0 * pi)
        )

    return BayesResult(
        posterior_mean=mu_n,
        posterior_cov=cov,
        credible_intervals=credible,
        prob_positive=prob_positive,
        sigma2_posterior_mean=float(sigma2_mean),
        log_marginal_likelihood=float(log_marg),
        prior={
            "prior_mean": mu_0.tolist(),
            "prior_precision_diag": np.diag(lambda_0).tolist(),
            "prior_a": float(prior_a),
            "prior_b": float(prior_b),
            "feature_names": names_with_intercept,
            "a_n": float(a_n),
            "b_n": float(b_n),
            "mu_n": mu_n.tolist(),
            "lambda_n": lambda_n.tolist(),
        },
        n_obs=int(n),
    )


def fit_bayes(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    prior: Literal["weakly_informative", "informative"] = "weakly_informative",
    mu_0: np.ndarray | None = None,
    lambda_0: np.ndarray | None = None,
    a_0: float = 1e-3,
    b_0: float = 1e-3,
    n_samples: int = 2000,
    random_state: int | None = None,
    include_intercept: bool = True,
) -> BayesianResult:
    """Bayesian linear regression — ergonomic wrapper around the conjugate solver.

    Pure numpy / scipy (no PyMC). Returns a :class:`BayesianResult` with
    per-feature dicts, posterior draws, and the log marginal likelihood.

    Posterior draws are taken from the joint conjugate posterior: first
    ``sigma^2 ~ InvGamma(a_n, b_n)``, then
    ``beta | sigma^2 ~ N(mu_n, sigma^2 * Lambda_n^{-1})``.

    Args:
        X: Design matrix. Pandas or numpy; intercept added internally when
            ``include_intercept=True``.
        y: Target vector.
        prior: ``"weakly_informative"`` uses ``Lambda_0 = 0.01 * I`` (prior
            variance ~ 100 in residual units). ``"informative"`` uses
            ``Lambda_0 = 10 * I`` (prior variance ~ 0.1), biasing toward
            ``mu_0``.
        mu_0: Optional explicit prior mean (overrides the ``prior`` preset).
        lambda_0: Optional explicit prior precision (overrides ``prior``).
        a_0: InvGamma shape. Default 1e-3 (very diffuse).
        b_0: InvGamma scale. Default 1e-3.
        n_samples: Number of posterior draws to return. ``0`` disables sampling.
        random_state: Seed for posterior sampling.
        include_intercept: Whether to prepend an intercept column (default).

    Returns:
        :class:`BayesianResult`.
    """

    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}")

    if not isinstance(X, pd.DataFrame):
        x_df = pd.DataFrame(np.asarray(X))
        x_df.columns = [f"x{i}" for i in range(x_df.shape[1])]
    else:
        x_df = X
    if not isinstance(y, pd.Series):
        y_s = pd.Series(np.asarray(y))
    else:
        y_s = y

    p_feat = x_df.shape[1]
    p_design = p_feat + 1 if include_intercept else p_feat

    if lambda_0 is None:
        if prior == "weakly_informative":
            lambda_0_eff = 0.01 * np.eye(p_design)
        elif prior == "informative":
            lambda_0_eff = 10.0 * np.eye(p_design)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown prior preset: {prior!r}")
    else:
        lambda_0_eff = np.asarray(lambda_0, dtype=float)
        if lambda_0_eff.shape != (p_design, p_design):
            raise ValueError(
                f"lambda_0 must have shape ({p_design}, {p_design}), got {lambda_0_eff.shape}"
            )

    if mu_0 is None:
        mu_0_eff = np.zeros(p_design)
    else:
        mu_0_eff = np.asarray(mu_0, dtype=float).reshape(-1)
        if mu_0_eff.shape[0] != p_design:
            raise ValueError(f"mu_0 must have length {p_design}, got {mu_0_eff.shape[0]}")

    # Common posterior derivation. We compute everything inline so we can
    # support the no-intercept variant without re-running the conjugate helper.
    y_arr, x_arr, feat = _coerce_xy(y_s, x_df)
    n_obs = x_arr.shape[0]
    if include_intercept:
        x_design = np.column_stack([np.ones(n_obs), x_arr])
        feature_names = ["intercept", *feat]
    else:
        x_design = x_arr
        feature_names = list(feat)

    xtx = x_design.T @ x_design
    xty = x_design.T @ y_arr
    lambda_n = lambda_0_eff + xtx
    mu_n = np.linalg.solve(lambda_n, lambda_0_eff @ mu_0_eff + xty)
    a_n = a_0 + n_obs / 2.0
    b_n = max(
        b_0
        + 0.5
        * (
            float(y_arr @ y_arr)
            + float(mu_0_eff @ lambda_0_eff @ mu_0_eff)
            - float(mu_n @ lambda_n @ mu_n)
        ),
        1e-12,
    )

    from scipy import stats

    df = 2.0 * a_n
    lambda_n_inv = np.linalg.inv(lambda_n)
    scale_diag = np.sqrt(np.maximum(np.diag((b_n / a_n) * lambda_n_inv), 0.0))
    t_crit = stats.t.ppf(0.975, df=df)
    ci_lo = mu_n - t_crit * scale_diag
    ci_hi = mu_n + t_crit * scale_diag

    sigma2_mean = float(b_n / (a_n - 1.0)) if a_n > 1.0 else float("inf")

    # Log marginal likelihood.
    sign0, logdet_l0 = np.linalg.slogdet(lambda_0_eff)
    signn, logdet_ln = np.linalg.slogdet(lambda_n)
    if sign0 > 0 and signn > 0 and a_0 > 0 and b_0 > 0:
        from math import lgamma, log, pi

        log_marg: float | None = (
            0.5 * logdet_l0
            - 0.5 * logdet_ln
            + a_0 * log(b_0)
            - a_n * log(b_n)
            + lgamma(a_n)
            - lgamma(a_0)
            - (n_obs / 2.0) * log(2.0 * pi)
        )
    else:
        log_marg = None

    # Posterior draws.
    samples: np.ndarray | None
    if n_samples > 0:
        rng = np.random.default_rng(random_state)
        # InvGamma(a, b) via 1 / Gamma(shape=a, scale=1/b).
        sigma2_draws = 1.0 / rng.gamma(shape=a_n, scale=1.0 / b_n, size=n_samples)
        cov_unit = 0.5 * (lambda_n_inv + lambda_n_inv.T)
        # Cholesky with small jitter for numerical PD.
        jitter = 1e-12 * np.eye(cov_unit.shape[0])
        try:
            chol = np.linalg.cholesky(cov_unit + jitter)
        except np.linalg.LinAlgError:
            # Fall back to eigendecomposition.
            w, v = np.linalg.eigh(cov_unit)
            w = np.maximum(w, 0.0)
            chol = v @ np.diag(np.sqrt(w))
        z = rng.standard_normal((n_samples, mu_n.shape[0]))
        beta_unit = z @ chol.T
        samples = mu_n[None, :] + np.sqrt(sigma2_draws)[:, None] * beta_unit
    else:
        samples = None

    posterior_mean_dict = {name: float(mu_n[i]) for i, name in enumerate(feature_names)}
    credible_dict: dict[str, tuple[float, float]] = {
        name: (float(ci_lo[i]), float(ci_hi[i])) for i, name in enumerate(feature_names)
    }

    return BayesianResult(
        posterior_mean=posterior_mean_dict,
        credible_intervals=credible_dict,
        posterior_samples=samples,
        feature_names=list(feature_names),
        sigma2_mean=float(sigma2_mean),
        log_marginal_likelihood=(None if log_marg is None else float(log_marg)),
    )


def fit_elastic_net(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    alpha: float | Literal["auto"] = 0.5,
    lambda_: float | Literal["auto"] = "auto",
    cv_splits: int = 5,
    standardise: bool = True,
    inference: Literal["none", "desparsified"] = "none",
    random_state: int | None = None,
) -> ElasticNetResult:
    """Elastic Net regression with optional time-series cross-validated lambda.

    Implementation uses :class:`sklearn.model_selection.TimeSeriesSplit`
    when ``lambda_="auto"`` to avoid look-ahead bias. Factors are
    z-score-standardised before fitting if ``standardise`` is true; reported
    coefficients are un-standardised back to the original units.

    Args:
        y: Target series.
        X: Design matrix without intercept column.
        alpha: L1 mixing fraction in ``[0, 1]``; 1.0 is pure LASSO, 0.0 is
            pure Ridge. ``"auto"`` runs a grid over
            ``{0.1, 0.3, 0.5, 0.7, 0.9}``.
        lambda_: Overall regularisation strength. ``"auto"`` does
            cross-validated selection.
        cv_splits: Number of ``TimeSeriesSplit`` folds for CV. Ignored if
            both ``alpha`` and ``lambda_`` are concrete.
        standardise: Whether to z-score factors before fitting.
        inference: ``"none"`` returns point estimates only. ``"desparsified"``
            is not yet implemented; passing it raises ``NotImplementedError``.
        random_state: Seed for any internal randomisation.

    Returns:
        :class:`ElasticNetResult` with un-standardised coefficients.

    Raises:
        ValueError: If ``alpha`` is a number outside ``[0, 1]``, if
            ``cv_splits < 2``, or if there are fewer than ``cv_splits * 2``
            non-NaN rows.
        NotImplementedError: If ``inference="desparsified"``.
    """

    # Lazy imports — keep top-level cost low for non-callers.
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    if inference == "desparsified":
        raise NotImplementedError(
            "desparsified inference is not yet implemented (Javanmard-Montanari)."
        )

    if isinstance(alpha, (int, float)) and not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if cv_splits < 2:
        raise ValueError(f"cv_splits must be >= 2, got {cv_splits}")

    # Coerce inputs to typed DataFrame / Series so we have column names.
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X))
        X.columns = [f"x{i}" for i in range(X.shape[1])]
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y))

    if len(y) != len(X):
        raise ValueError(f"len(y)={len(y)} != len(X)={len(X)}")

    # Drop rows with NaN in either y or X.
    joined = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(joined) < cv_splits * 2:
        raise ValueError(
            f"need at least {cv_splits * 2} non-NaN rows for cv_splits={cv_splits}, "
            f"got {len(joined)}"
        )
    y_clean = joined["__y__"].to_numpy(dtype=float)
    X_clean = joined.drop(columns="__y__")
    feature_names = list(X_clean.columns)
    X_arr = X_clean.to_numpy(dtype=float)

    n, p = X_arr.shape

    # Standardise factors (and centre y) if requested. We always centre to make
    # the un-standardisation arithmetic clean.
    x_mean = X_arr.mean(axis=0)
    x_std = X_arr.std(axis=0, ddof=0)
    # Guard against zero-variance columns: leave them as-is (std=1 effectively).
    safe_std = np.where(x_std > 1e-12, x_std, 1.0)
    if standardise:
        X_fit = (X_arr - x_mean) / safe_std
    else:
        # Still centre to compute intercept cleanly, but don't divide.
        X_fit = X_arr - x_mean
        safe_std = np.ones(p)

    y_mean = float(y_clean.mean())
    y_centered = y_clean - y_mean

    # ------------------------------------------------------------------
    # Solver selection
    # ------------------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    reg_path: list[dict[str, float]] = []

    auto_alpha = alpha == "auto"
    auto_lambda = lambda_ == "auto"

    if auto_alpha or auto_lambda:
        # CV path. ElasticNetCV picks lambda automatically; if alpha is also
        # auto we iterate over the standard grid and keep the best CV R^2.
        alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9] if auto_alpha else [float(alpha)]
        best = None  # (cv_r2, l1_ratio, model)
        for l1 in alpha_grid:
            if auto_lambda:
                model = ElasticNetCV(
                    l1_ratio=l1,
                    cv=tscv,
                    max_iter=20000,
                    fit_intercept=False,  # already centred
                    random_state=random_state,
                    alphas=50,  # sklearn >=1.7 accepts int; older accepts via n_alphas
                    tol=1e-5,
                )
                model.fit(X_fit, y_centered)
                # mse_path_ shape: (n_alphas, n_folds). Lower is better.
                fold_mses = model.mse_path_.mean(axis=1)
                best_idx = int(np.argmin(fold_mses))
                cv_mse = float(fold_mses[best_idx])
                # Approximate CV R^2 from CV MSE vs total variance.
                total_var = float(np.var(y_centered)) + 1e-12
                cv_r2 = 1.0 - cv_mse / total_var
                reg_path.append(
                    {
                        "lambda": float(model.alphas_[best_idx]),
                        "alpha": float(l1),
                        "r2_cv": float(cv_r2),
                        "n_selected": int(np.count_nonzero(model.coef_)),
                    }
                )
                if best is None or cv_r2 > best[0]:
                    best = (cv_r2, float(l1), model)
            else:
                # alpha auto, lambda fixed: just fit a single ElasticNet at this l1.
                model = ElasticNet(
                    alpha=float(lambda_),  # type: ignore[arg-type]
                    l1_ratio=l1,
                    max_iter=20000,
                    fit_intercept=False,
                    random_state=random_state,
                    tol=1e-5,
                )
                model.fit(X_fit, y_centered)
                # Compute CV R^2 manually for ranking.
                cv_scores = cross_val_score(
                    ElasticNet(
                        alpha=float(lambda_),  # type: ignore[arg-type]
                        l1_ratio=l1,
                        max_iter=20000,
                        fit_intercept=False,
                        random_state=random_state,
                        tol=1e-5,
                    ),
                    X_fit,
                    y_centered,
                    cv=tscv,
                    scoring="neg_mean_squared_error",
                )
                cv_mse = float(-cv_scores.mean())
                total_var = float(np.var(y_centered)) + 1e-12
                cv_r2 = 1.0 - cv_mse / total_var
                reg_path.append(
                    {
                        "lambda": float(lambda_),  # type: ignore[arg-type]
                        "alpha": float(l1),
                        "r2_cv": float(cv_r2),
                        "n_selected": int(np.count_nonzero(model.coef_)),
                    }
                )
                if best is None or cv_r2 > best[0]:
                    best = (cv_r2, float(l1), model)

        assert best is not None
        cv_r2_final, l1_final, fitted = best
        if isinstance(fitted, ElasticNetCV):
            opt_lambda = float(fitted.alpha_)
        else:
            opt_lambda = float(lambda_)  # type: ignore[arg-type]
        opt_alpha = l1_final
        coef_std_scale = fitted.coef_
    else:
        # Both concrete — single ElasticNet fit. l1_ratio=0 is technically
        # pure Ridge; sklearn warns at exactly 0, so we clip to a tiny value.
        l1_ratio_use = max(float(alpha), 1e-8)
        model = ElasticNet(
            alpha=float(lambda_),
            l1_ratio=l1_ratio_use,
            max_iter=20000,
            fit_intercept=False,
            random_state=random_state,
            tol=1e-5,
        )
        model.fit(X_fit, y_centered)
        # Single-shot CV R^2 for reporting.
        cv_scores = cross_val_score(
            ElasticNet(
                alpha=float(lambda_),
                l1_ratio=l1_ratio_use,
                max_iter=20000,
                fit_intercept=False,
                random_state=random_state,
                tol=1e-5,
            ),
            X_fit,
            y_centered,
            cv=tscv,
            scoring="neg_mean_squared_error",
        )
        cv_mse = float(-cv_scores.mean())
        total_var = float(np.var(y_centered)) + 1e-12
        cv_r2_final = 1.0 - cv_mse / total_var
        opt_lambda = float(lambda_)
        opt_alpha = float(alpha)
        coef_std_scale = model.coef_
        reg_path.append(
            {
                "lambda": opt_lambda,
                "alpha": opt_alpha,
                "r2_cv": float(cv_r2_final),
                "n_selected": int(np.count_nonzero(model.coef_)),
            }
        )

    # ------------------------------------------------------------------
    # Un-standardise coefficients back to original units.
    # ------------------------------------------------------------------
    coef_orig = coef_std_scale / safe_std
    intercept_orig = y_mean - float(np.dot(coef_orig, x_mean))

    coefficients = {name: float(coef_orig[i]) for i, name in enumerate(feature_names)}
    selected = [name for i, name in enumerate(feature_names) if abs(coef_orig[i]) > 1e-10]

    return ElasticNetResult(
        coefficients=coefficients,
        intercept=float(intercept_orig),
        selected_factors=selected,
        optimal_lambda=float(opt_lambda),
        optimal_alpha=float(opt_alpha),
        regularisation_path=reg_path,
        r_squared_cv=float(cv_r2_final),
        n_obs=int(n),
    )


def fit_quantile(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    taus: list[float] | None = None,
    n_bootstrap: int = 0,
    bootstrap_method: Literal["paired", "xy"] = "paired",
    max_iter: int = 2000,
    random_state: int | None = None,
) -> QuantileResult:
    """Quantile regression at one or more tau values.

    Solves :math:`\\hat\\beta(\\tau) = \\arg\\min_\\beta \\sum_i \\rho_\\tau(y_i - x_i^T \\beta)`
    via the interior-point method (delegating to
    ``statsmodels.regression.quantile_regression.QuantReg``).

    If ``n_bootstrap > 0`` runs paired-bootstrap to produce 95% CIs on
    :math:`\\hat\\beta(\\tau)` for each tau.

    Args:
        y: Target series.
        X: Design matrix without intercept column.
        taus: Quantile levels to fit. Defaults to ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
            Each value must be in ``(0, 1)`` exclusive.
        n_bootstrap: Number of bootstrap replicates per tau. 0 disables.
            Recommended: 500 for production.
        bootstrap_method: ``"paired"`` resamples ``(x_i, y_i)`` pairs;
            ``"xy"`` resamples residuals (assumes iid residuals — rarely
            appropriate for prediction-market data).
        random_state: Seed for bootstrap resampling.

    Returns:
        :class:`QuantileResult`.

    Raises:
        ValueError: On non-positive sample size, mismatched ``len(y)`` /
            ``len(X)``, any ``tau`` outside the open interval ``(0, 1)``,
            ``n_bootstrap < 0``, or after dropping NaN rows fewer
            observations than columns (rank-deficient design).
    """

    # Lazy import — keeps top-level import cost negligible for non-callers.
    from statsmodels.regression.quantile_regression import QuantReg

    # ------------------------------------------------------------------
    # Defaults & validation
    # ------------------------------------------------------------------
    if taus is None:
        taus_list: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    else:
        taus_list = [float(t) for t in taus]

    if not taus_list:
        raise ValueError("taus must contain at least one value")
    for t in taus_list:
        if not (0.0 < t < 1.0):
            raise ValueError(f"tau must be in the open interval (0, 1), got {t}")

    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be >= 0, got {n_bootstrap}")

    if bootstrap_method not in ("paired", "xy"):
        raise ValueError(f"bootstrap_method must be 'paired' or 'xy', got {bootstrap_method!r}")

    # Coerce inputs so we always have column names.
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X))
        X.columns = [f"x{i}" for i in range(X.shape[1])]
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y))

    if len(y) != len(X):
        raise ValueError(f"len(y)={len(y)} != len(X)={len(X)}")

    if len(y) == 0:
        raise ValueError("y is empty")

    # Drop rows with NaN in either y or X.
    joined = pd.concat([y.rename("__y__"), X.reset_index(drop=True)], axis=1).dropna()
    if len(joined) == 0:
        raise ValueError("no rows remain after dropping NaN")

    y_clean = joined["__y__"].to_numpy(dtype=float)
    X_clean = joined.drop(columns="__y__")
    feature_names = list(X_clean.columns)
    X_arr = X_clean.to_numpy(dtype=float)
    n, p = X_arr.shape

    if n <= p + 1:
        raise ValueError(f"need more rows than columns+intercept ({p + 1}), got {n} after dropna")

    # Design matrix with intercept prepended.
    X_design = np.column_stack([np.ones(n), X_arr])

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Fit each tau
    # ------------------------------------------------------------------
    coef_table: dict[str, list[float]] = {name: [] for name in feature_names}
    pseudo_r2: dict[float, float] = {}
    cis: dict[float, np.ndarray] = {}

    # Median-only (tau=0.5) absolute deviation baseline used for Koenker-Machado
    # pseudo R^2. Computed once and reused per-tau via the tau-specific rho.
    def _rho(u: np.ndarray, tau: float) -> float:
        return float(np.sum(u * (tau - (u < 0).astype(float))))

    # Median model (intercept-only) for each tau — needed for pseudo R^2 baseline.
    np.sort(y_clean)

    def _tau_quantile(values: np.ndarray, tau: float) -> float:
        # numpy quantile uses default linear interpolation which matches the
        # standard sample quantile definition used by Koenker-Machado.
        return float(np.quantile(values, tau))

    for tau in taus_list:
        model = QuantReg(y_clean, X_design)
        # ``max_iter`` is exposed via the fit kwarg; statsmodels accepts it.
        try:
            res = model.fit(q=tau, max_iter=max_iter)
        except TypeError:
            # Older statsmodels may not accept max_iter — fall back to defaults.
            res = model.fit(q=tau)

        params = np.asarray(res.params, dtype=float)
        for i, name in enumerate(feature_names):
            # +1 because index 0 is the intercept.
            coef_table[name].append(float(params[i + 1]))

        # Koenker-Machado pseudo R^2: 1 - rho_tau(full) / rho_tau(null)
        # where the null model fits only an intercept at the tau-th sample quantile.
        residuals_full = y_clean - X_design @ params
        rho_full = _rho(residuals_full, tau)
        null_intercept = _tau_quantile(y_clean, tau)
        residuals_null = y_clean - null_intercept
        rho_null = _rho(residuals_null, tau)
        if rho_null <= 1e-12:
            r2 = 0.0
        else:
            r2 = 1.0 - rho_full / rho_null
        # Clamp to [0, 1] — for an unconstrained fit on training data this is
        # natural, but pathological edge cases may produce tiny negatives.
        pseudo_r2[float(tau)] = float(max(0.0, min(1.0, r2)))

        # --------------------------------------------------------------
        # Bootstrap CIs
        # --------------------------------------------------------------
        if n_bootstrap > 0:
            boot_betas = np.zeros((n_bootstrap, p), dtype=float)
            n_success = 0
            base_residuals = residuals_full.copy()
            for _b in range(n_bootstrap):
                if bootstrap_method == "paired":
                    idx = rng.integers(0, n, size=n)
                    Xb = X_design[idx]
                    yb = y_clean[idx]
                else:  # "xy" — residual bootstrap
                    idx = rng.integers(0, n, size=n)
                    yb = X_design @ params + base_residuals[idx]
                    Xb = X_design
                try:
                    b_res = QuantReg(yb, Xb).fit(q=tau)
                    b_params = np.asarray(b_res.params, dtype=float)
                    boot_betas[n_success] = b_params[1:]
                    n_success += 1
                except Exception:
                    # Some bootstrap samples can yield singular designs; skip.
                    continue
            if n_success >= 2:
                lo = np.quantile(boot_betas[:n_success], 0.025, axis=0)
                hi = np.quantile(boot_betas[:n_success], 0.975, axis=0)
            else:
                lo = np.full(p, np.nan)
                hi = np.full(p, np.nan)
            cis[float(tau)] = np.column_stack([lo, hi])

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    coef_df = pd.DataFrame(
        {tau: [coef_table[name][i] for name in feature_names] for i, tau in enumerate(taus_list)},
        index=feature_names,
    )
    coef_df.columns = [float(t) for t in taus_list]

    # Tail asymmetry: highest tau minus lowest tau.
    tau_lo = min(taus_list)
    tau_hi = max(taus_list)
    if tau_lo == tau_hi:
        tail_asym = pd.Series(0.0, index=feature_names, name="tail_asymmetry")
    else:
        tail_asym = (coef_df[float(tau_hi)] - coef_df[float(tau_lo)]).rename("tail_asymmetry")

    return QuantileResult(
        taus=[float(t) for t in taus_list],
        coefficients_by_quantile=coef_df,
        bootstrap_cis=cis,
        pseudo_r2_by_quantile=pseudo_r2,
        tail_asymmetry=tail_asym,
        n_obs=int(n),
        n_bootstrap=int(n_bootstrap),
    )
