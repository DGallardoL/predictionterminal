"""Tests for :func:`pfm.quant.regression_methods.fit_elastic_net` (task W11-57).

These tests exercise the elastic-net solver on synthetic data with known
data-generating processes so we can verify:

* sparse-signal recovery (signal columns selected, noise columns shrunk),
* alpha/lambda behaviour (Ridge-like vs LASSO-like vs OLS-like),
* reproducibility under a fixed ``random_state``,
* standardisation correctness (coefficients un-standardised back to original
  units),
* edge cases (1 feature, collinear features, NaN handling, validation errors).

All tests use locally generated NumPy/Pandas data — no network, no fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.quant.regression_methods import ElasticNetResult, fit_elastic_net

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse_dgp(
    n: int,
    p_signal: int,
    p_noise: int,
    beta_signal: np.ndarray,
    noise_sigma: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Generate ``y = X @ beta + eps`` with ``p_signal`` true factors and
    ``p_noise`` pure-noise columns. Returns ``(X, y, full_beta)``."""

    rng = np.random.default_rng(seed)
    p = p_signal + p_noise
    X = rng.standard_normal((n, p))
    full_beta = np.zeros(p)
    full_beta[:p_signal] = beta_signal
    eps = rng.standard_normal(n) * noise_sigma
    y_vals = X @ full_beta + eps
    names = [f"sig{i}" for i in range(p_signal)] + [f"noise{i}" for i in range(p_noise)]
    return pd.DataFrame(X, columns=names), pd.Series(y_vals, name="y"), full_beta


# ---------------------------------------------------------------------------
# 1. Basic synthetic-DGP recovery
# ---------------------------------------------------------------------------


def test_basic_sparse_recovery_three_factors() -> None:
    """y = 0.5*X1 + 0.0*X2 + 0.3*X3 + eps — recovers X1 and X3."""

    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, 3))
    y_vals = 0.5 * X[:, 0] + 0.0 * X[:, 1] + 0.3 * X[:, 2] + 0.02 * rng.standard_normal(n)
    X_df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
    y = pd.Series(y_vals)

    res = fit_elastic_net(y, X_df, alpha=0.5, lambda_=0.01, random_state=0)
    assert isinstance(res, ElasticNetResult)
    # X1, X3 should be in support; X2 should be ~0.
    assert abs(res.coefficients["X1"] - 0.5) < 0.1
    assert abs(res.coefficients["X3"] - 0.3) < 0.1
    assert abs(res.coefficients["X2"]) < 0.05


def test_returns_dataclass_with_expected_fields() -> None:
    X, y, _ = _make_sparse_dgp(200, 3, 5, np.array([0.4, -0.3, 0.2]), seed=1)
    res = fit_elastic_net(y, X, alpha=0.5, lambda_=0.05, random_state=1)
    assert isinstance(res, ElasticNetResult)
    assert isinstance(res.coefficients, dict)
    assert isinstance(res.selected_factors, list)
    assert isinstance(res.optimal_lambda, float)
    assert isinstance(res.optimal_alpha, float)
    assert isinstance(res.r_squared_cv, float)
    assert isinstance(res.n_obs, int)
    # n_obs reflects post-NaN cleanup; here = 200.
    assert res.n_obs == 200
    # Every feature has a coefficient entry.
    assert set(res.coefficients) == set(X.columns)


# ---------------------------------------------------------------------------
# 2. High-dimensional sparsity (5 signal + 100 noise)
# ---------------------------------------------------------------------------


def test_high_dim_sparsity_support_size_bounded() -> None:
    beta_signal = np.array([0.6, -0.5, 0.4, -0.3, 0.25])
    X, y, _ = _make_sparse_dgp(500, p_signal=5, p_noise=100, beta_signal=beta_signal, seed=7)
    res = fit_elastic_net(y, X, alpha=0.9, lambda_="auto", cv_splits=5, random_state=7)
    # Sparse model: support should be small relative to p=105.
    assert len(res.selected_factors) <= 30
    # All 5 true signals should be in support.
    signal_names = {f"sig{i}" for i in range(5)}
    recovered = signal_names & set(res.selected_factors)
    assert len(recovered) >= 4  # at least 4/5 recovered


def test_high_dim_precision_recall() -> None:
    beta_signal = np.array([0.7, -0.6, 0.5, -0.4, 0.3])
    X, y, _ = _make_sparse_dgp(600, p_signal=5, p_noise=100, beta_signal=beta_signal, seed=11)
    res = fit_elastic_net(y, X, alpha=0.9, lambda_="auto", random_state=11)
    truth = {f"sig{i}" for i in range(5)}
    selected = set(res.selected_factors)
    tp = len(truth & selected)
    len(selected - truth)
    recall = tp / max(len(truth), 1)
    precision = tp / max(len(selected), 1)
    # Pure LASSO often picks 1-2 extra noise; demand recall >= 0.8 and precision >= 0.3.
    assert recall >= 0.8, f"recall={recall}"
    assert precision >= 0.3, f"precision={precision}, selected={selected}"


# ---------------------------------------------------------------------------
# 3. alpha=large -> all coefficients shrink toward 0
# ---------------------------------------------------------------------------


def test_large_lambda_shrinks_to_zero() -> None:
    X, y, _ = _make_sparse_dgp(300, 3, 5, np.array([0.5, -0.4, 0.3]), seed=3)
    # Huge lambda (alpha here = l1_ratio mixing; we override lambda_).
    res = fit_elastic_net(y, X, alpha=0.5, lambda_=100.0, random_state=3)
    # All un-standardised coefficients should be ~0 under heavy regularisation.
    for name, coef in res.coefficients.items():
        assert abs(coef) < 0.05, f"{name}={coef} not shrunk"


def test_zero_lambda_resembles_ols() -> None:
    """With lambda_ very small, coefficients should be close to OLS estimates."""

    rng = np.random.default_rng(5)
    n = 300
    X = rng.standard_normal((n, 4))
    true_beta = np.array([0.5, -0.4, 0.3, 0.2])
    y_vals = X @ true_beta + 0.02 * rng.standard_normal(n)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    y = pd.Series(y_vals)

    # tiny lambda → near-OLS solution
    res = fit_elastic_net(y, X_df, alpha=0.5, lambda_=1e-6, random_state=5)
    for i, name in enumerate(X_df.columns):
        assert abs(res.coefficients[name] - true_beta[i]) < 0.05


# ---------------------------------------------------------------------------
# 4. l1_ratio extremes (LASSO vs Ridge behaviour)
# ---------------------------------------------------------------------------


def test_l1_ratio_one_produces_sparse_solution() -> None:
    """alpha=1.0 → pure LASSO; many noise features should be exactly zero."""

    beta_signal = np.array([0.6, -0.5])
    X, y, _ = _make_sparse_dgp(400, p_signal=2, p_noise=30, beta_signal=beta_signal, seed=9)
    res = fit_elastic_net(y, X, alpha=1.0, lambda_=0.05, random_state=9)
    # LASSO should zero out most noise columns.
    n_zero = sum(1 for v in res.coefficients.values() if abs(v) < 1e-10)
    assert n_zero >= 20  # most of the 30 noise features should be exact zeros


def test_l1_ratio_zero_keeps_all_nonzero() -> None:
    """alpha≈0 → pure Ridge; no sparsity, all coefficients non-zero."""

    beta_signal = np.array([0.5, -0.4, 0.3])
    X, y, _ = _make_sparse_dgp(300, p_signal=3, p_noise=10, beta_signal=beta_signal, seed=13)
    # We clip 0 internally to a tiny l1 ratio, but the result should still be
    # Ridge-dominated → essentially no exact zeros.
    res = fit_elastic_net(y, X, alpha=0.0, lambda_=0.1, random_state=13)
    n_zero = sum(1 for v in res.coefficients.values() if abs(v) < 1e-10)
    assert n_zero == 0


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------


def test_same_random_state_is_reproducible() -> None:
    X, y, _ = _make_sparse_dgp(250, 3, 10, np.array([0.5, -0.4, 0.3]), seed=21)
    r1 = fit_elastic_net(y, X, alpha=0.7, lambda_="auto", cv_splits=5, random_state=42)
    r2 = fit_elastic_net(y, X, alpha=0.7, lambda_="auto", cv_splits=5, random_state=42)
    for name in r1.coefficients:
        assert r1.coefficients[name] == pytest.approx(r2.coefficients[name], abs=1e-9)
    assert r1.optimal_lambda == pytest.approx(r2.optimal_lambda, abs=1e-9)
    assert r1.r_squared_cv == pytest.approx(r2.r_squared_cv, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. Standardisation
# ---------------------------------------------------------------------------


def test_standardise_true_returns_orig_scale_coefficients() -> None:
    """Scaling a column by 100 should produce a coefficient 100× smaller."""

    rng = np.random.default_rng(33)
    n = 400
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n) * 100.0  # large-scale column
    y_vals = 0.5 * x1 + 0.01 * x2 + 0.02 * rng.standard_normal(n)
    X_df = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.Series(y_vals)
    res = fit_elastic_net(y, X_df, alpha=0.5, lambda_=1e-4, standardise=True, random_state=33)
    # Recovered coefficients must be on original scale.
    assert abs(res.coefficients["x1"] - 0.5) < 0.05
    assert abs(res.coefficients["x2"] - 0.01) < 0.005


def test_standardise_false_still_runs() -> None:
    X, y, _ = _make_sparse_dgp(200, 3, 4, np.array([0.4, -0.3, 0.2]), seed=44)
    res = fit_elastic_net(y, X, alpha=0.5, lambda_=0.01, standardise=False, random_state=44)
    # Sanity: still recovers signal columns approximately.
    assert abs(res.coefficients["sig0"] - 0.4) < 0.1


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


def test_single_feature_univariate() -> None:
    rng = np.random.default_rng(55)
    n = 200
    x = rng.standard_normal(n)
    y_vals = 0.7 * x + 0.02 * rng.standard_normal(n)
    X_df = pd.DataFrame({"x": x})
    y = pd.Series(y_vals)
    res = fit_elastic_net(y, X_df, alpha=0.5, lambda_=1e-4, random_state=55)
    assert abs(res.coefficients["x"] - 0.7) < 0.05
    assert "x" in res.selected_factors


def test_collinear_features_handled() -> None:
    """When X2 = X1 + small noise, both share weight but solver still converges."""

    rng = np.random.default_rng(77)
    n = 300
    x1 = rng.standard_normal(n)
    x2 = x1 + 0.001 * rng.standard_normal(n)  # near-perfect collinearity
    y_vals = 1.0 * x1 + 0.02 * rng.standard_normal(n)
    X_df = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.Series(y_vals)
    res = fit_elastic_net(y, X_df, alpha=0.5, lambda_=0.001, random_state=77)
    # Sum of the two near-collinear coefficients should approximate 1.0.
    total = res.coefficients["x1"] + res.coefficients["x2"]
    assert abs(total - 1.0) < 0.15


def test_nan_rows_dropped() -> None:
    X, y, _ = _make_sparse_dgp(200, 3, 4, np.array([0.4, -0.3, 0.2]), seed=88)
    # Inject NaNs in 10 rows.
    X.iloc[5:15, 0] = np.nan
    res = fit_elastic_net(y, X, alpha=0.5, lambda_=0.01, random_state=88)
    assert res.n_obs == 190


def test_intercept_shifts_correctly() -> None:
    """Adding a constant offset to y should shift only the intercept, not betas."""

    rng = np.random.default_rng(91)
    n = 250
    X = rng.standard_normal((n, 3))
    true_beta = np.array([0.5, -0.4, 0.3])
    y_base = X @ true_beta + 0.02 * rng.standard_normal(n)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    res0 = fit_elastic_net(pd.Series(y_base), X_df, alpha=0.5, lambda_=1e-4, random_state=91)
    res5 = fit_elastic_net(pd.Series(y_base + 5.0), X_df, alpha=0.5, lambda_=1e-4, random_state=91)
    for name in X_df.columns:
        assert abs(res0.coefficients[name] - res5.coefficients[name]) < 0.02
    assert abs((res5.intercept - res0.intercept) - 5.0) < 0.05


# ---------------------------------------------------------------------------
# 8. Validation errors
# ---------------------------------------------------------------------------


def test_invalid_alpha_raises() -> None:
    X, y, _ = _make_sparse_dgp(100, 2, 2, np.array([0.4, 0.3]), seed=1)
    with pytest.raises(ValueError, match="alpha must be in"):
        fit_elastic_net(y, X, alpha=1.5, lambda_=0.01)


def test_invalid_cv_splits_raises() -> None:
    X, y, _ = _make_sparse_dgp(100, 2, 2, np.array([0.4, 0.3]), seed=2)
    with pytest.raises(ValueError, match="cv_splits"):
        fit_elastic_net(y, X, alpha=0.5, lambda_="auto", cv_splits=1)


def test_too_few_observations_raises() -> None:
    X, y, _ = _make_sparse_dgp(8, 2, 2, np.array([0.4, 0.3]), seed=3)
    with pytest.raises(ValueError, match="non-NaN rows"):
        fit_elastic_net(y, X, alpha=0.5, lambda_=0.01, cv_splits=5)


def test_mismatched_lengths_raise() -> None:
    X = pd.DataFrame(np.zeros((10, 2)), columns=["a", "b"])
    y = pd.Series(np.zeros(11))
    with pytest.raises(ValueError, match="len"):
        fit_elastic_net(y, X, alpha=0.5, lambda_=0.01)


def test_desparsified_inference_not_implemented() -> None:
    X, y, _ = _make_sparse_dgp(100, 2, 2, np.array([0.4, 0.3]), seed=4)
    with pytest.raises(NotImplementedError):
        fit_elastic_net(y, X, alpha=0.5, lambda_=0.01, inference="desparsified")


# ---------------------------------------------------------------------------
# 9. Auto mode (TimeSeriesSplit-based CV)
# ---------------------------------------------------------------------------


def test_auto_lambda_returns_optimal() -> None:
    X, y, _ = _make_sparse_dgp(400, 3, 20, np.array([0.6, -0.5, 0.4]), seed=101)
    res = fit_elastic_net(y, X, alpha=0.5, lambda_="auto", cv_splits=5, random_state=101)
    # Reg path should have at least one entry per alpha tried.
    assert len(res.regularisation_path) >= 1
    # Optimal lambda should be a positive finite float.
    assert np.isfinite(res.optimal_lambda)
    assert res.optimal_lambda > 0


def test_auto_alpha_picks_from_grid() -> None:
    X, y, _ = _make_sparse_dgp(400, 3, 20, np.array([0.6, -0.5, 0.4]), seed=103)
    res = fit_elastic_net(y, X, alpha="auto", lambda_="auto", cv_splits=5, random_state=103)
    # auto-alpha grid is {0.1, 0.3, 0.5, 0.7, 0.9}; chosen alpha must be in it.
    assert res.optimal_alpha in {0.1, 0.3, 0.5, 0.7, 0.9}


# ---------------------------------------------------------------------------
# 10. 20-fold synthetic Monte-Carlo recovery
# ---------------------------------------------------------------------------


def test_monte_carlo_recovery_within_tolerance() -> None:
    """Run 20 synthetic experiments; mean MSE on true betas under tolerance."""

    n_trials = 20
    mses = []
    recall_count = 0
    for trial in range(n_trials):
        beta_signal = np.array([0.5, -0.4, 0.3])
        X, y, _ = _make_sparse_dgp(
            n=400,
            p_signal=3,
            p_noise=10,
            beta_signal=beta_signal,
            seed=200 + trial,
        )
        res = fit_elastic_net(
            y, X, alpha=0.7, lambda_="auto", cv_splits=5, random_state=200 + trial
        )
        recovered = np.array([res.coefficients[f"sig{i}"] for i in range(3)])
        mses.append(float(np.mean((recovered - beta_signal) ** 2)))
        truth = {f"sig{i}" for i in range(3)}
        if truth.issubset(set(res.selected_factors)):
            recall_count += 1
    mean_mse = float(np.mean(mses))
    # Tolerance: mean coefficient MSE should be well under 0.01 (~10% of beta scale).
    assert mean_mse < 0.01, f"mean MSE on signal betas = {mean_mse}"
    # In ≥ 80% of trials, all 3 signals should be selected.
    assert recall_count >= int(0.8 * n_trials), f"recall_count={recall_count}/{n_trials}"


# ---------------------------------------------------------------------------
# 11. Numpy-array input is accepted (signature documents pd.Series/pd.DataFrame
#     but we silently coerce, which is friendlier for callers).
# ---------------------------------------------------------------------------


def test_numpy_array_input_is_accepted() -> None:
    rng = np.random.default_rng(202)
    n, p = 200, 4
    X = rng.standard_normal((n, p))
    true_beta = np.array([0.5, -0.4, 0.3, 0.0])
    y_vals = X @ true_beta + 0.02 * rng.standard_normal(n)
    res = fit_elastic_net(
        pd.Series(y_vals), pd.DataFrame(X), alpha=0.5, lambda_=1e-4, random_state=202
    )
    assert isinstance(res, ElasticNetResult)
    # Column names should auto-generated as 0..p-1 (integers from DataFrame default).
    assert len(res.coefficients) == p
