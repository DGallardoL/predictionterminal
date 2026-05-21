"""Tests for :func:`pfm.quant.regression_methods.fit_quantile` (task W12-23).

Synthetic data only — no network. We verify:

* tail behaviour on heteroskedastic data (variance grows with X => tau=0.9
  coefficients exceed tau=0.5 coefficients),
* coefficient consistency across taus on a symmetric (homoskedastic) DGP,
* point recovery within tolerance for both 1-feature and 5-feature DGPs,
* bootstrap CIs cover the true beta most of the time,
* Koenker-Machado pseudo R-squared lies in ``[0, 1]``,
* edge cases (empty inputs, len mismatch, NaN drops, too-few-obs,
  invalid tau / n_bootstrap / bootstrap_method),
* reproducibility under a fixed ``random_state``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.quant.regression_methods import QuantileResult, fit_quantile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hetero_dgp(
    n: int = 600,
    beta: float = 1.0,
    seed: int = 11,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """``y = beta * x + (1 + x^2) * eps`` — variance grows with |x|.

    For this DGP the conditional quantile of y given x is
    ``beta*x + (1 + x^2) * Phi^{-1}(tau)``, so a *linear* quantile regression
    sees an effective beta that increases with |tau - 0.5|. tau=0.9 beta
    should be materially larger (in absolute value) than tau=0.5 beta.
    """

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    y = beta * x + (1.0 + x**2) * eps
    return pd.DataFrame({"x": x}), pd.Series(y, name="y"), beta


def _symmetric_dgp(
    n: int = 600,
    beta: float = 0.8,
    sigma: float = 0.3,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """``y = beta * x + sigma * eps`` with iid Gaussian eps."""

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n) * sigma
    y = beta * x + eps
    return pd.DataFrame({"x": x}), pd.Series(y, name="y"), beta


def _multi_dgp(
    n: int = 800,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """5-feature DGP with mild noise; conditional median == OLS solution."""

    rng = np.random.default_rng(seed)
    p = 5
    X = rng.standard_normal((n, p))
    true = np.array([1.0, -0.5, 0.25, 0.0, 0.75])
    eps = rng.standard_normal(n) * 0.2
    y = X @ true + eps
    names = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=names), pd.Series(y, name="y"), true


# ---------------------------------------------------------------------------
# Smoke / shape
# ---------------------------------------------------------------------------


def test_returns_quantile_result_with_expected_taus():
    X, y, _ = _symmetric_dgp()
    out = fit_quantile(y, X)
    assert isinstance(out, QuantileResult)
    assert out.taus == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert list(out.coefficients_by_quantile.columns) == out.taus
    assert list(out.coefficients_by_quantile.index) == ["x"]


def test_custom_taus_respected():
    X, y, _ = _symmetric_dgp(n=400)
    out = fit_quantile(y, X, taus=[0.2, 0.5, 0.8])
    assert out.taus == [0.2, 0.5, 0.8]
    assert list(out.coefficients_by_quantile.columns) == [0.2, 0.5, 0.8]


def test_n_obs_reflects_after_dropna():
    X, y, _ = _symmetric_dgp(n=300)
    X.loc[0, "x"] = np.nan
    y.iloc[5] = np.nan
    out = fit_quantile(y, X, taus=[0.5])
    assert out.n_obs == 298


def test_n_bootstrap_zero_by_default_no_cis():
    X, y, _ = _symmetric_dgp()
    out = fit_quantile(y, X, taus=[0.5])
    assert out.bootstrap_cis == {}
    assert out.n_bootstrap == 0


# ---------------------------------------------------------------------------
# Coefficient recovery
# ---------------------------------------------------------------------------


def test_one_feature_recovery_within_five_percent():
    X, y, beta = _symmetric_dgp(n=1500, beta=1.25, sigma=0.2, seed=3)
    out = fit_quantile(y, X, taus=[0.5])
    est = out.coefficients_by_quantile.loc["x", 0.5]
    assert abs(est - beta) / abs(beta) < 0.05


def test_five_feature_recovery_at_median():
    X, y, true = _multi_dgp(n=1500, seed=29)
    out = fit_quantile(y, X, taus=[0.5])
    est = out.coefficients_by_quantile[0.5].to_numpy()
    # Median quantile reg on Gaussian noise behaves like LAD, slightly less
    # efficient than OLS — allow generous tolerance.
    assert np.max(np.abs(est - true)) < 0.10


def test_symmetric_dgp_consistent_across_taus():
    X, y, beta = _symmetric_dgp(n=2000, beta=0.8, sigma=0.3, seed=99)
    out = fit_quantile(y, X, taus=[0.25, 0.5, 0.75])
    coefs = out.coefficients_by_quantile.loc["x"]
    # All three taus should give approximately the same slope for a
    # homoskedastic linear DGP.
    spread = float(coefs.max() - coefs.min())
    assert spread < 0.10
    # And all three should be near the true beta.
    for tau in [0.25, 0.5, 0.75]:
        assert abs(coefs[tau] - beta) < 0.10


# ---------------------------------------------------------------------------
# Tail asymmetry
# ---------------------------------------------------------------------------


def test_heteroskedastic_tail_beta_exceeds_median():
    X, y, _ = _hetero_dgp(n=2000, beta=1.0, seed=21)
    out = fit_quantile(y, X, taus=[0.1, 0.5, 0.9])
    b50 = out.coefficients_by_quantile.loc["x", 0.5]
    b90 = out.coefficients_by_quantile.loc["x", 0.9]
    b10 = out.coefficients_by_quantile.loc["x", 0.1]
    # In the upper tail the (1+x^2)*eps slope makes the conditional 90th
    # percentile rise faster in x than the median.
    assert b90 > b50
    # Symmetric in the lower tail: the 10th percentile decreases faster.
    assert b10 < b50


def test_tail_asymmetry_field_matches_high_minus_low():
    X, y, _ = _hetero_dgp(n=1500, seed=44)
    out = fit_quantile(y, X, taus=[0.1, 0.5, 0.9])
    expected = out.coefficients_by_quantile[0.9] - out.coefficients_by_quantile[0.1]
    pd.testing.assert_series_equal(
        out.tail_asymmetry.rename(None),
        expected.rename(None),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def test_bootstrap_returns_cis_shape_per_tau():
    X, y, _ = _symmetric_dgp(n=400, seed=2)
    out = fit_quantile(y, X, taus=[0.25, 0.75], n_bootstrap=50, random_state=0)
    assert set(out.bootstrap_cis.keys()) == {0.25, 0.75}
    for ci in out.bootstrap_cis.values():
        assert ci.shape == (1, 2)  # 1 feature, (lo, hi)
        assert ci[0, 0] <= ci[0, 1]


def test_bootstrap_ci_contains_true_beta_on_symmetric_dgp():
    X, y, beta = _symmetric_dgp(n=800, beta=0.9, sigma=0.25, seed=8)
    out = fit_quantile(y, X, taus=[0.5], n_bootstrap=200, random_state=42)
    lo, hi = out.bootstrap_cis[0.5][0]
    assert lo <= beta <= hi


def test_bootstrap_xy_method_runs():
    X, y, _ = _symmetric_dgp(n=400, seed=4)
    out = fit_quantile(y, X, taus=[0.5], n_bootstrap=40, bootstrap_method="xy", random_state=1)
    assert 0.5 in out.bootstrap_cis
    assert out.bootstrap_cis[0.5].shape == (1, 2)


def test_bootstrap_reproducible_under_random_state():
    X, y, _ = _symmetric_dgp(n=400, seed=5)
    a = fit_quantile(y, X, taus=[0.5], n_bootstrap=60, random_state=123)
    b = fit_quantile(y, X, taus=[0.5], n_bootstrap=60, random_state=123)
    np.testing.assert_allclose(a.bootstrap_cis[0.5], b.bootstrap_cis[0.5])


def test_bootstrap_different_seeds_differ():
    X, y, _ = _symmetric_dgp(n=400, seed=6)
    a = fit_quantile(y, X, taus=[0.5], n_bootstrap=60, random_state=1)
    b = fit_quantile(y, X, taus=[0.5], n_bootstrap=60, random_state=2)
    assert not np.allclose(a.bootstrap_cis[0.5], b.bootstrap_cis[0.5])


# ---------------------------------------------------------------------------
# Pseudo R^2
# ---------------------------------------------------------------------------


def test_pseudo_r2_in_unit_interval():
    X, y, _ = _symmetric_dgp(n=500, seed=13)
    out = fit_quantile(y, X, taus=[0.1, 0.5, 0.9])
    for tau in out.taus:
        r2 = out.pseudo_r2_by_quantile[tau]
        assert 0.0 <= r2 <= 1.0


def test_pseudo_r2_higher_when_signal_strong():
    # Strong signal vs noise-only DGP — pseudo R^2 should rank correctly.
    X_strong, y_strong, _ = _symmetric_dgp(n=600, beta=2.0, sigma=0.1, seed=15)
    out_strong = fit_quantile(y_strong, X_strong, taus=[0.5])

    rng = np.random.default_rng(15)
    X_noise = pd.DataFrame({"x": rng.standard_normal(600)})
    y_noise = pd.Series(rng.standard_normal(600))
    out_noise = fit_quantile(y_noise, X_noise, taus=[0.5])

    assert out_strong.pseudo_r2_by_quantile[0.5] > out_noise.pseudo_r2_by_quantile[0.5]


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_rows_dropped():
    X, y, _ = _symmetric_dgp(n=300, seed=20)
    X.iloc[10:15, 0] = np.nan
    y.iloc[20] = np.nan
    out = fit_quantile(y, X, taus=[0.5])
    assert out.n_obs == 300 - 5 - 1


# ---------------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------------


def test_empty_y_raises():
    with pytest.raises(ValueError):
        fit_quantile(pd.Series([], dtype=float), pd.DataFrame({"x": []}), taus=[0.5])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        fit_quantile(
            pd.Series([1.0, 2.0, 3.0]),
            pd.DataFrame({"x": [1.0, 2.0]}),
            taus=[0.5],
        )


def test_too_few_observations_raises():
    # 2 rows, 1 feature => intercept + slope leaves 0 dof; should raise.
    X = pd.DataFrame({"x": [0.1, 0.4]})
    y = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[0.5])


def test_single_observation_raises():
    X = pd.DataFrame({"x": [0.1]})
    y = pd.Series([1.0])
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[0.5])


def test_invalid_tau_raises():
    X, y, _ = _symmetric_dgp(n=200, seed=30)
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[0.0])
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[1.0])
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[-0.1])
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[1.5])


def test_negative_n_bootstrap_raises():
    X, y, _ = _symmetric_dgp(n=200, seed=31)
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[0.5], n_bootstrap=-1)


def test_invalid_bootstrap_method_raises():
    X, y, _ = _symmetric_dgp(n=200, seed=32)
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[0.5], bootstrap_method="block")  # type: ignore[arg-type]


def test_empty_taus_raises():
    X, y, _ = _symmetric_dgp(n=200, seed=33)
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[])


def test_all_nan_rows_raise():
    X = pd.DataFrame({"x": [np.nan, np.nan, np.nan, np.nan]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        fit_quantile(y, X, taus=[0.5])


# ---------------------------------------------------------------------------
# Numpy-array inputs accepted
# ---------------------------------------------------------------------------


def test_accepts_ndarray_inputs():
    rng = np.random.default_rng(40)
    X_arr = rng.standard_normal((300, 2))
    y_arr = X_arr @ np.array([0.5, -0.3]) + 0.1 * rng.standard_normal(300)
    out = fit_quantile(y_arr, X_arr, taus=[0.5])  # type: ignore[arg-type]
    assert out.n_obs == 300
    assert list(out.coefficients_by_quantile.index) == ["x0", "x1"]
    # Coefficient signs should match the DGP.
    assert out.coefficients_by_quantile.loc["x0", 0.5] > 0
    assert out.coefficients_by_quantile.loc["x1", 0.5] < 0
