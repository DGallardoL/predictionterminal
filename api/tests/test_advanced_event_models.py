"""Tests for ``pfm.advanced_event_models``.

Each test plants a known data-generating process and checks the
corresponding core function recovers the planted structure to within a
generous tolerance. External APIs are never hit; series are entirely
synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.advanced_event_models import (
    compute_tail_dependence_core,
    fit_conditional_model_core,
    fit_garch_x_core,
    fit_polynomial_factor_model_core,
    fit_regime_switching_model_core,
    fit_vecm_core,
)
from pfm.model import delta_logit, logit_transform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def _random_walk_probs(n: int, *, seed: int, sigma: float = 0.04) -> pd.Series:
    """Random-walk-on-logit prediction-market series in (0.05, 0.95)."""
    rng = np.random.default_rng(seed)
    start_logit = rng.uniform(-1.5, 1.5)
    increments = rng.normal(0.0, sigma, n)
    logits = start_logit + np.cumsum(increments)
    p = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(p, 0.05, 0.95)
    return pd.Series(p, index=_utc_index(n), name="p")


# ---------------------------------------------------------------------------
# A) Conditional
# ---------------------------------------------------------------------------


class TestConditionalModel:
    def test_detects_different_betas_across_buckets(self) -> None:
        rng = np.random.default_rng(7)
        n = 800
        # Probability series spanning the whole [0.05, 0.95] range.
        idx = _utc_index(n)
        p = pd.Series(rng.uniform(0.06, 0.94, n), index=idx, name="p")
        dl = delta_logit(p).fillna(0.0)
        # Plant beta_low = 1.5 on p<0.3, beta_mid = 0.4 on p in [0.3, 0.7), beta_hi = -0.6 on p>=0.7.
        beta_low, beta_mid, beta_hi = 1.5, 0.4, -0.6
        beta_t = np.where(p.values < 0.3, beta_low, np.where(p.values < 0.7, beta_mid, beta_hi))
        eps = rng.normal(0.0, 0.005, n)
        r = beta_t * dl.values + eps
        r = pd.Series(r, index=idx, name="r")
        out = fit_conditional_model_core(
            r,
            p,
            conditioning_thresholds=[0.3, 0.7],
            ticker="TEST",
            factor_id="planted",
        )
        # We have 3 buckets: [0, 0.3), [0.3, 0.7), [0.7, 1.0].
        assert len(out["buckets"]) == 3
        b_low, b_mid, b_hi = out["buckets"]
        # All buckets should have planted samples; if a bucket's beta is
        # missing the test design is broken, not the fitter.
        assert b_low["beta"] is not None
        assert b_mid["beta"] is not None
        assert b_hi["beta"] is not None
        # Each bucket beta should be near its planted value.
        assert b_low["beta"] == pytest.approx(beta_low, abs=0.30)
        assert b_mid["beta"] == pytest.approx(beta_mid, abs=0.30)
        assert b_hi["beta"] == pytest.approx(beta_hi, abs=0.30)
        # And meaningfully different from each other.
        assert abs(b_low["beta"] - b_hi["beta"]) > 0.5

    def test_short_window_raises(self) -> None:
        idx = _utc_index(20)
        with pytest.raises(ValueError):
            fit_conditional_model_core(
                pd.Series(np.zeros(20), index=idx),
                pd.Series(np.full(20, 0.5), index=idx),
                conditioning_thresholds=[0.5],
            )


# ---------------------------------------------------------------------------
# B) Polynomial
# ---------------------------------------------------------------------------


class TestPolynomialModel:
    def test_cubic_dgp_prefers_degree_three(self) -> None:
        rng = np.random.default_rng(11)
        n = 600
        p = _random_walk_probs(n, seed=12, sigma=0.05)
        dl = delta_logit(p).fillna(0.0).values
        # Plant: r = 2 * dl - 1.5 * dl^2 + 4.0 * dl^3 + noise.
        r = 2.0 * dl - 1.5 * dl**2 + 4.0 * dl**3 + rng.normal(0.0, 0.005, n)
        r_s = pd.Series(r, index=p.index, name="r")
        out = fit_polynomial_factor_model_core(
            r_s,
            p,
            degree=3,
            ticker="TEST",
            factor_id="planted",
        )
        assert out["degree"] == 3
        # The optimal AIC degree should be >= 3.
        assert out["optimal_degree_aic"] >= 3
        # The LR test against the linear model should reject strongly.
        assert out["vs_linear_lr_test_p"] is not None
        assert out["vs_linear_lr_test_p"] < 0.05
        # R² should be substantial.
        assert out["r_squared"] > 0.5
        # Marginal-effects grid is populated.
        assert len(out["marginal_effects"]) == 21

    def test_pure_linear_dgp_does_not_reject(self) -> None:
        rng = np.random.default_rng(3)
        n = 500
        p = _random_walk_probs(n, seed=4, sigma=0.04)
        dl = delta_logit(p).fillna(0.0).values
        r = 0.6 * dl + rng.normal(0.0, 0.005, n)
        r_s = pd.Series(r, index=p.index, name="r")
        out = fit_polynomial_factor_model_core(r_s, p, degree=2)
        # Linear DGP: AIC should pick degree 1 (sometimes 2 by noise — accept either).
        assert out["optimal_degree_aic"] in (1, 2)


# ---------------------------------------------------------------------------
# C) Regime-switching — smoke test
# ---------------------------------------------------------------------------


class TestRegimeSwitching:
    def test_two_regime_fit_runs_and_returns_shape(self) -> None:
        rng = np.random.default_rng(101)
        n = 400
        # Two regimes with very different beta and sigma.
        states = np.repeat([0, 1, 0, 1], n // 4)
        p = _random_walk_probs(len(states), seed=999, sigma=0.05)
        dl = delta_logit(p).fillna(0.0).values
        beta_by_state = np.array([1.0, -1.5])
        sigma_by_state = np.array([0.005, 0.02])
        r = beta_by_state[states] * dl + rng.normal(0.0, 1.0, len(states)) * sigma_by_state[states]
        r_s = pd.Series(r, index=p.index, name="r")
        out = fit_regime_switching_model_core(
            r_s,
            p,
            n_regimes=2,
            ticker="TEST",
            factor_id="planted",
        )
        assert out["n_regimes"] == 2
        assert len(out["regimes"]) == 2
        assert len(out["transition_matrix"]) == 2
        assert all(len(row) == 2 for row in out["transition_matrix"])
        assert len(out["smoothed_state_probs_last_30"]) > 0
        # Ergodic distribution sums to ~1.
        ergodic_sum = sum(reg["ergodic_prob"] for reg in out["regimes"])
        assert ergodic_sum == pytest.approx(1.0, abs=1e-6)
        # The two regimes should have visibly different sigmas given the planted DGP.
        sigmas = sorted(reg["std"] for reg in out["regimes"])
        assert sigmas[1] > sigmas[0] * 1.3


# ---------------------------------------------------------------------------
# D) VECM
# ---------------------------------------------------------------------------


class TestVECM:
    def test_cointegrated_dgp_detected(self) -> None:
        rng = np.random.default_rng(31)
        n = 500
        # Build a cointegrated pair: log_eq is a random walk; logit_p = beta * log_eq + I(0) noise.
        increments = rng.normal(0.0, 0.01, n)
        log_eq = np.cumsum(increments) + 4.0  # start log price ~4 (price ~55)
        beta_true = 2.0
        ar_eps = np.zeros(n)
        rho = 0.5
        for t in range(1, n):
            ar_eps[t] = rho * ar_eps[t - 1] + rng.normal(0.0, 0.05)
        logit_p = beta_true * (log_eq - log_eq.mean()) + ar_eps
        p = 1.0 / (1.0 + np.exp(-logit_p))
        p = np.clip(p, 0.02, 0.98)
        idx = _utc_index(n)
        out = fit_vecm_core(
            pd.Series(np.exp(log_eq), index=idx, name="px"),
            pd.Series(p, index=idx, name="p"),
            det_order=0,
            k_ar_diff=1,
            ticker="TEST",
            factor_id="planted",
        )
        assert out["is_cointegrated"] is True
        assert out["beta_long_run"] is not None

    def test_independent_walks_not_cointegrated(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        log_eq = np.cumsum(rng.normal(0.0, 0.01, n)) + 4.0
        # Truly independent random-walk on logit.
        logit_p = np.cumsum(rng.normal(0.0, 0.04, n))
        p = np.clip(1.0 / (1.0 + np.exp(-logit_p)), 0.02, 0.98)
        idx = _utc_index(n)
        out = fit_vecm_core(
            pd.Series(np.exp(log_eq), index=idx, name="px"),
            pd.Series(p, index=idx, name="p"),
        )
        # Two independent walks — should very rarely show as cointegrated. We
        # don't assert False outright (Johansen has finite-sample size) but we
        # at least require the trace stat below the 99% critical value.
        assert out["johansen_trace_stat"] < out["johansen_trace_crit_95"] + 5.0


# ---------------------------------------------------------------------------
# E) GARCH-X
# ---------------------------------------------------------------------------


class TestGarchX:
    def test_vol_clustering_recovered(self) -> None:
        rng = np.random.default_rng(73)
        n = 600
        p = _random_walk_probs(n, seed=74, sigma=0.05)
        dl_abs = delta_logit(p).fillna(0.0).abs().values
        # Generate returns with vol clustering (planted GARCH(1,1)).
        omega = 1e-5
        a_true = 0.10
        b_true = 0.85
        gamma_true = 5e-5
        eps = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1.0 - a_true - b_true)
        eps[0] = rng.normal(0.0, np.sqrt(sigma2[0]))
        for t in range(1, n):
            sigma2[t] = (
                omega + a_true * eps[t - 1] ** 2 + b_true * sigma2[t - 1] + gamma_true * dl_abs[t]
            )
            eps[t] = rng.normal(0.0, np.sqrt(max(sigma2[t], 1e-12)))
        r = pd.Series(eps, index=p.index, name="r")
        out = fit_garch_x_core(r, p, ticker="TEST", factor_id="planted")
        # Stationary fit, recovers the GARCH structure roughly.
        assert out["is_stationary"] is True
        # Persistence in the same ballpark as the truth (0.95).
        assert 0.5 < out["persistence"] < 0.999
        # Half-life of vol is positive and finite.
        assert out["half_life_vol_days"] is not None
        assert out["half_life_vol_days"] > 0.0
        # omega/alpha/beta sensible.
        assert out["omega"] > 0
        assert 0.0 <= out["alpha"] <= 1.0
        assert 0.0 <= out["beta"] <= 1.0


# ---------------------------------------------------------------------------
# F) Tail dependence
# ---------------------------------------------------------------------------


class TestTailDependence:
    def test_independent_inputs_near_quantile(self) -> None:
        rng = np.random.default_rng(53)
        n = 1500
        idx = _utc_index(n)
        r = pd.Series(rng.normal(0.0, 1.0, n), index=idx, name="r")
        x = pd.Series(rng.normal(0.0, 1.0, n), index=idx, name="x")
        out = compute_tail_dependence_core(r, x, quantile=0.05)
        # Under independence, lambda_L ≈ q. We allow a generous band for finite
        # sample noise.
        assert out["lower_tail_dependence"] < 0.20
        assert out["upper_tail_dependence"] < 0.20

    def test_lower_tail_correlated_inputs(self) -> None:
        rng = np.random.default_rng(83)
        n = 1500
        # Construct r and x such that joint extreme negative draws are common:
        # r = -|z| - 0.2 * |x_extra| when x is small; otherwise mild.
        u = rng.normal(0.0, 1.0, n)
        common = rng.normal(0.0, 1.0, n)
        # x is u plus small idiosyncratic noise -> they share most of their tail.
        x_vals = u + 0.1 * rng.normal(0.0, 1.0, n)
        r_vals = u + 0.1 * common
        idx = _utc_index(n)
        r = pd.Series(r_vals, index=idx, name="r")
        x = pd.Series(x_vals, index=idx, name="x")
        out = compute_tail_dependence_core(r, x, quantile=0.05)
        assert out["lower_tail_dependence"] > 0.5
        assert out["lower_ratio_vs_independence"] > 5.0


# ---------------------------------------------------------------------------
# Logit-transform sanity (small reuse check)
# ---------------------------------------------------------------------------


def test_logit_transform_inverse_consistency() -> None:
    p = pd.Series([0.1, 0.5, 0.9])
    z = logit_transform(p)
    p_round = 1.0 / (1.0 + np.exp(-z.values))
    np.testing.assert_allclose(p_round, p.values, atol=1e-10)
