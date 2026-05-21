"""Tests for :func:`pfm.quant.regression_methods.fit_bayes` (task W12-24).

These tests exercise the conjugate normal-inverse-gamma Bayesian linear
regression on synthetic data with known data-generating processes so we can
verify:

* posterior mean recovers the true beta within 5% for moderate ``n``,
* 95% credible intervals contain the true beta in approximately 95% of
  replicates (frequentist coverage check),
* weakly-informative prior behaves like OLS,
* strong (informative) prior biases the posterior toward the prior mean,
* residual variance ``sigma^2`` is recovered accurately,
* more posterior samples produce tighter sample-based summaries,
* 1-feature perfect-data edge case behaves sensibly,
* the analytic log marginal likelihood matches a hand-derived value.

All tests use locally generated NumPy/Pandas data — no network, no fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.quant.regression_methods import BayesianResult, fit_bayes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dgp(
    n: int,
    beta: np.ndarray,
    sigma: float = 0.1,
    seed: int = 0,
    intercept: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate ``y = intercept + X @ beta + eps``, ``eps ~ N(0, sigma^2)``."""

    rng = np.random.default_rng(seed)
    p = len(beta)
    X = rng.standard_normal((n, p))
    eps = rng.standard_normal(n) * sigma
    y = intercept + X @ beta + eps
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# 1. Synthetic-DGP recovery
# ---------------------------------------------------------------------------


def test_recovers_known_beta_within_5pct_two_factors() -> None:
    """Single-factor recovery: known beta should be matched within 5%."""

    true_beta = np.array([0.7, -0.4])
    X, y = _make_dgp(n=400, beta=true_beta, sigma=0.05, seed=11)
    res = fit_bayes(X, y, n_samples=0, random_state=0)

    assert isinstance(res, BayesianResult)
    est = np.array([res.posterior_mean[c] for c in X.columns])
    rel_err = np.abs(est - true_beta) / np.abs(true_beta)
    assert np.all(rel_err < 0.05), f"relative errors {rel_err} exceed 5%"


def test_recovers_intercept() -> None:
    """Non-zero intercept in DGP should be recovered."""

    X, y = _make_dgp(n=500, beta=np.array([0.3]), sigma=0.05, intercept=1.5, seed=2)
    res = fit_bayes(X, y, n_samples=0)
    assert abs(res.posterior_mean["intercept"] - 1.5) < 0.02
    assert abs(res.posterior_mean["f0"] - 0.3) < 0.02


def test_recovers_three_factors() -> None:
    """Three-factor recovery with mixed signs."""

    true_beta = np.array([0.5, -0.3, 0.2])
    X, y = _make_dgp(n=600, beta=true_beta, sigma=0.08, seed=3)
    res = fit_bayes(X, y, n_samples=0)
    for i, c in enumerate(X.columns):
        assert abs(res.posterior_mean[c] - true_beta[i]) < 0.03


# ---------------------------------------------------------------------------
# 2. Credible-interval coverage
# ---------------------------------------------------------------------------


def test_credible_intervals_have_approx_95pct_coverage() -> None:
    """Across 60 replicates, 95% CI should contain truth for >= 85% of cases.

    Tolerance is loose because of small replicate count + multiple coefficients,
    but the rate must be in a reasonable neighbourhood of 0.95.
    """

    true_beta = np.array([0.4, -0.25])
    hits = 0
    total = 0
    for seed in range(60):
        X, y = _make_dgp(n=200, beta=true_beta, sigma=0.1, seed=100 + seed)
        res = fit_bayes(X, y, n_samples=0)
        for i, c in enumerate(X.columns):
            lo, hi = res.credible_intervals[c]
            if lo <= true_beta[i] <= hi:
                hits += 1
            total += 1
    rate = hits / total
    assert 0.85 <= rate <= 1.0, f"coverage rate {rate:.3f} far from 0.95"


# ---------------------------------------------------------------------------
# 3. Weakly informative vs OLS
# ---------------------------------------------------------------------------


def test_weakly_informative_prior_close_to_ols() -> None:
    """With a weakly informative prior, posterior mean ~ OLS estimate."""

    true_beta = np.array([0.6, -0.2, 0.35])
    X, y = _make_dgp(n=300, beta=true_beta, sigma=0.1, seed=5)
    res = fit_bayes(X, y, prior="weakly_informative", n_samples=0)

    # Compute OLS for comparison.
    X_d = np.column_stack([np.ones(len(y)), X.to_numpy()])
    ols_beta = np.linalg.lstsq(X_d, y.to_numpy(), rcond=None)[0]

    for i, c in enumerate(["intercept", *X.columns]):
        assert abs(res.posterior_mean[c] - ols_beta[i]) < 0.02


# ---------------------------------------------------------------------------
# 4. Strong prior bias
# ---------------------------------------------------------------------------


def test_strong_prior_biases_posterior_toward_prior_mean() -> None:
    """A heavily-precise prior centered at 0 should shrink the posterior mean.

    The likelihood pulls toward the OLS estimate; an informative prior
    centered at zero pulls toward zero. The posterior mean must lie strictly
    between them in absolute value.
    """

    true_beta = np.array([0.8])
    X, y = _make_dgp(n=50, beta=true_beta, sigma=0.1, seed=7)

    # Strong prior at zero (Lambda_0 = 1000 * I -> prior variance ~ 1e-3).
    p_design = 2  # intercept + 1 feat
    strong_lambda = 1000.0 * np.eye(p_design)
    res_strong = fit_bayes(X, y, mu_0=np.zeros(p_design), lambda_0=strong_lambda, n_samples=0)
    res_weak = fit_bayes(X, y, prior="weakly_informative", n_samples=0)

    # Strong prior must shrink the feature toward zero.
    assert abs(res_strong.posterior_mean["f0"]) < abs(res_weak.posterior_mean["f0"])
    # And specifically be far below the true value.
    assert res_strong.posterior_mean["f0"] < 0.5


def test_informative_preset_pulls_toward_zero_more_than_weak() -> None:
    """The ``informative`` preset should shrink more than ``weakly_informative``."""

    true_beta = np.array([0.6])
    X, y = _make_dgp(n=30, beta=true_beta, sigma=0.15, seed=9)
    res_weak = fit_bayes(X, y, prior="weakly_informative", n_samples=0)
    res_inf = fit_bayes(X, y, prior="informative", n_samples=0)
    assert abs(res_inf.posterior_mean["f0"]) < abs(res_weak.posterior_mean["f0"])


def test_strong_prior_centered_off_zero_biases_toward_that_value() -> None:
    """Prior mean centered at a non-zero value pulls the posterior there."""

    true_beta = np.array([0.5])
    X, y = _make_dgp(n=40, beta=true_beta, sigma=0.2, seed=12)
    # Strong prior centered at -1.0 (intercept + feature).
    prior_mu = np.array([0.0, -1.0])
    strong_lambda = 500.0 * np.eye(2)
    res = fit_bayes(X, y, mu_0=prior_mu, lambda_0=strong_lambda, n_samples=0)
    # Posterior should be far closer to -1 than to the true 0.5.
    assert res.posterior_mean["f0"] < 0.0


# ---------------------------------------------------------------------------
# 5. Residual variance recovery
# ---------------------------------------------------------------------------


def test_sigma2_recovered_within_20pct() -> None:
    """Posterior mean of sigma^2 should be close to the true noise variance."""

    sigma_true = 0.15
    X, y = _make_dgp(n=500, beta=np.array([0.4, -0.3]), sigma=sigma_true, seed=21)
    res = fit_bayes(X, y, n_samples=0)
    sigma2_true = sigma_true**2
    rel = abs(res.sigma2_mean - sigma2_true) / sigma2_true
    assert rel < 0.20, f"sigma^2 estimate {res.sigma2_mean:.5f} vs truth {sigma2_true:.5f}"


# ---------------------------------------------------------------------------
# 6. Posterior-sample scaling
# ---------------------------------------------------------------------------


def test_posterior_samples_shape_and_disabled() -> None:
    """``n_samples`` controls the shape of ``posterior_samples``; 0 disables."""

    X, y = _make_dgp(n=100, beta=np.array([0.4, -0.2]), sigma=0.1, seed=31)

    res_no = fit_bayes(X, y, n_samples=0)
    assert res_no.posterior_samples is None

    res_yes = fit_bayes(X, y, n_samples=500, random_state=0)
    assert res_yes.posterior_samples is not None
    assert res_yes.posterior_samples.shape == (500, 3)  # intercept + 2 features


def test_more_samples_means_tighter_sample_mean_estimate() -> None:
    """More draws -> sample mean closer to analytic posterior mean."""

    X, y = _make_dgp(n=100, beta=np.array([0.4, -0.2]), sigma=0.1, seed=41)
    res_small = fit_bayes(X, y, n_samples=100, random_state=1)
    res_big = fit_bayes(X, y, n_samples=5000, random_state=1)

    analytic = np.array([res_big.posterior_mean[c] for c in res_big.feature_names])
    assert res_small.posterior_samples is not None
    assert res_big.posterior_samples is not None

    err_small = np.abs(res_small.posterior_samples.mean(axis=0) - analytic).max()
    err_big = np.abs(res_big.posterior_samples.mean(axis=0) - analytic).max()
    assert err_big < err_small + 1e-9  # noise can flip sign at boundary; +epsilon


def test_posterior_samples_match_credible_intervals_approximately() -> None:
    """Sample 2.5/97.5 percentiles should be close to analytic CI bounds."""

    X, y = _make_dgp(n=300, beta=np.array([0.5]), sigma=0.1, seed=51)
    res = fit_bayes(X, y, n_samples=10000, random_state=7)
    assert res.posterior_samples is not None
    sample_lo = np.percentile(res.posterior_samples, 2.5, axis=0)
    sample_hi = np.percentile(res.posterior_samples, 97.5, axis=0)
    for i, c in enumerate(res.feature_names):
        lo_ana, hi_ana = res.credible_intervals[c]
        assert abs(sample_lo[i] - lo_ana) < 0.05
        assert abs(sample_hi[i] - hi_ana) < 0.05


def test_random_state_reproducibility() -> None:
    """Same seed -> identical posterior draws."""

    X, y = _make_dgp(n=80, beta=np.array([0.3]), sigma=0.1, seed=61)
    r1 = fit_bayes(X, y, n_samples=200, random_state=42)
    r2 = fit_bayes(X, y, n_samples=200, random_state=42)
    assert r1.posterior_samples is not None and r2.posterior_samples is not None
    np.testing.assert_allclose(r1.posterior_samples, r2.posterior_samples)


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


def test_one_feature_near_perfect_data() -> None:
    """With near-zero noise and one feature, posterior mean is essentially OLS."""

    rng = np.random.default_rng(99)
    n = 100
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n) * 1e-6
    y = 0.7 * x + eps
    X = pd.DataFrame({"x": x})
    y_s = pd.Series(y)

    res = fit_bayes(X, y_s, n_samples=0)
    assert abs(res.posterior_mean["x"] - 0.7) < 1e-3
    assert abs(res.posterior_mean["intercept"]) < 1e-3
    # CI should be very tight (dominated by the diffuse a_0/b_0 prior, but
    # still orders of magnitude smaller than the noisy-data case).
    lo, hi = res.credible_intervals["x"]
    assert (hi - lo) < 1e-2


def test_numpy_input_accepted() -> None:
    """Raw numpy arrays should also work (no DataFrame required)."""

    rng = np.random.default_rng(77)
    X_np = rng.standard_normal((50, 2))
    y_np = X_np @ np.array([0.5, -0.3]) + 0.05 * rng.standard_normal(50)
    res = fit_bayes(X_np, y_np, n_samples=0)
    # Default column names from coercion are x0, x1.
    assert "x0" in res.posterior_mean
    assert "x1" in res.posterior_mean
    assert abs(res.posterior_mean["x0"] - 0.5) < 0.1
    assert abs(res.posterior_mean["x1"] - -0.3) < 0.1


def test_mismatched_lengths_raises() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="len"):
        fit_bayes(X, y)


def test_negative_n_samples_raises() -> None:
    X, y = _make_dgp(n=20, beta=np.array([0.3]), sigma=0.1)
    with pytest.raises(ValueError, match="n_samples"):
        fit_bayes(X, y, n_samples=-1)


def test_wrong_mu0_shape_raises() -> None:
    X, y = _make_dgp(n=20, beta=np.array([0.3]), sigma=0.1)
    # X has 1 feat; design has 2 cols (intercept + f0). mu_0 of length 5 wrong.
    with pytest.raises(ValueError, match="mu_0"):
        fit_bayes(X, y, mu_0=np.zeros(5), n_samples=0)


# ---------------------------------------------------------------------------
# 8. Log marginal likelihood
# ---------------------------------------------------------------------------


def test_log_marginal_likelihood_is_finite_number() -> None:
    """LML should be a finite float for any reasonable fit."""

    X, y = _make_dgp(n=80, beta=np.array([0.4, -0.2]), sigma=0.1, seed=71)
    res = fit_bayes(X, y, n_samples=0)
    assert res.log_marginal_likelihood is not None
    assert np.isfinite(res.log_marginal_likelihood)


def test_log_marginal_likelihood_higher_for_better_fit() -> None:
    """Adding pure-noise covariates with a weakly informative prior should
    decrease (or not significantly increase) the marginal likelihood relative
    to the correctly-specified model."""

    rng = np.random.default_rng(81)
    n = 200
    x_signal = rng.standard_normal(n)
    eps = rng.standard_normal(n) * 0.1
    y_arr = 0.5 * x_signal + eps
    y_s = pd.Series(y_arr)

    X_good = pd.DataFrame({"signal": x_signal})
    X_bad = pd.DataFrame(
        {
            "signal": x_signal,
            "noise1": rng.standard_normal(n),
            "noise2": rng.standard_normal(n),
            "noise3": rng.standard_normal(n),
            "noise4": rng.standard_normal(n),
            "noise5": rng.standard_normal(n),
        }
    )

    lml_good = fit_bayes(X_good, y_s, n_samples=0).log_marginal_likelihood
    lml_bad = fit_bayes(X_bad, y_s, n_samples=0).log_marginal_likelihood
    assert lml_good is not None and lml_bad is not None
    # The simpler correctly-specified model should be preferred (Occam).
    assert lml_good > lml_bad


def test_log_marginal_likelihood_matches_manual_calculation() -> None:
    """For a tiny fixed problem, the LML must equal the hand-derived value."""

    from math import lgamma, log, pi

    rng = np.random.default_rng(0)
    n = 5
    X = rng.standard_normal((n, 1))
    beta_true = 0.5
    y = X[:, 0] * beta_true + 0.1 * rng.standard_normal(n)

    res = fit_bayes(
        pd.DataFrame(X, columns=["x"]),
        pd.Series(y),
        prior="weakly_informative",
        a_0=1.0,
        b_0=1.0,
        n_samples=0,
        include_intercept=True,
    )

    # Re-derive: design matrix with intercept, Lambda_0 = 0.01 I.
    x_design = np.column_stack([np.ones(n), X])
    lambda_0 = 0.01 * np.eye(2)
    mu_0 = np.zeros(2)
    a_0, b_0 = 1.0, 1.0
    lam_n = lambda_0 + x_design.T @ x_design
    mu_n = np.linalg.solve(lam_n, lambda_0 @ mu_0 + x_design.T @ y)
    a_n = a_0 + n / 2.0
    b_n = b_0 + 0.5 * (float(y @ y) + float(mu_0 @ lambda_0 @ mu_0) - float(mu_n @ lam_n @ mu_n))
    _, ld0 = np.linalg.slogdet(lambda_0)
    _, ldn = np.linalg.slogdet(lam_n)
    expected = (
        0.5 * ld0
        - 0.5 * ldn
        + a_0 * log(b_0)
        - a_n * log(b_n)
        + lgamma(a_n)
        - lgamma(a_0)
        - (n / 2.0) * log(2.0 * pi)
    )
    assert res.log_marginal_likelihood is not None
    assert abs(res.log_marginal_likelihood - expected) < 1e-8


# ---------------------------------------------------------------------------
# 9. Structural sanity
# ---------------------------------------------------------------------------


def test_feature_names_and_keys_align() -> None:
    """``feature_names`` should match the keys of the dict outputs."""

    X, y = _make_dgp(n=50, beta=np.array([0.3, -0.1]), sigma=0.1, seed=91)
    res = fit_bayes(X, y, n_samples=10, random_state=0)
    assert set(res.posterior_mean.keys()) == set(res.feature_names)
    assert set(res.credible_intervals.keys()) == set(res.feature_names)
    assert res.posterior_samples is not None
    assert res.posterior_samples.shape[1] == len(res.feature_names)


def test_credible_intervals_are_ordered_lo_then_hi() -> None:
    X, y = _make_dgp(n=80, beta=np.array([0.4, -0.2]), sigma=0.1, seed=93)
    res = fit_bayes(X, y, n_samples=0)
    for name, (lo, hi) in res.credible_intervals.items():
        assert lo < hi, f"CI for {name} not ordered: ({lo}, {hi})"


def test_no_intercept_path_dimensions() -> None:
    """``include_intercept=False`` should drop the intercept slot."""

    X, y = _make_dgp(n=100, beta=np.array([0.5]), sigma=0.05, intercept=0.0, seed=95)
    res = fit_bayes(X, y, include_intercept=False, n_samples=0)
    assert "intercept" not in res.posterior_mean
    assert list(res.posterior_mean.keys()) == ["f0"]
    assert abs(res.posterior_mean["f0"] - 0.5) < 0.05
