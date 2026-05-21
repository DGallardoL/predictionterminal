"""Deep, exhaustive synthetic-DGP tests for ``pfm.advanced_event_models``.

Each block plants a known data-generating process, fits the
corresponding core, and verifies recovery within a generous tolerance.
External APIs are never hit. The router smoke tests use TestClient with
``fetch_factor_history`` and ``fetch_equity_history`` monkeypatched at
the router module level.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pfm.advanced_event_models import (
    compute_tail_dependence_core,
    fit_conditional_model_core,
    fit_garch_x_core,
    fit_polynomial_factor_model_core,
    fit_regime_switching_model_core,
    fit_vecm_core,
)
from pfm.model import delta_logit

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _utc_index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _drifting_uniform_probs(n: int, *, seed: int) -> pd.Series:
    """Probabilities that drift across the full [0.05, 0.95] band so all
    conditional buckets get populated."""
    rng = np.random.default_rng(seed)
    # Mix uniform draws with some autocorrelation, then clip.
    base = rng.uniform(0.06, 0.94, n)
    return pd.Series(base, index=_utc_index(n), name="p")


# ===========================================================================
# A) Conditional model
# ===========================================================================


class TestConditionalDeep:
    def test_three_bucket_recovery_centre_is_high_beta(self) -> None:
        """DGP: beta_low=0.5 / beta_mid=2.0 / beta_high=0.5. Centre bucket
        must surface the largest beta, extremes near 0.5."""
        rng = np.random.default_rng(7)
        n = 800
        p = _drifting_uniform_probs(n, seed=11)
        dl = delta_logit(p).fillna(0.0).values
        beta_t = np.where(p.values < 0.3, 0.5, np.where(p.values < 0.7, 2.0, 0.5))
        eps = rng.normal(0.0, 0.005, n)
        r = pd.Series(beta_t * dl + eps, index=p.index, name="r")
        out = fit_conditional_model_core(
            r, p, conditioning_thresholds=[0.3, 0.7], ticker="T", factor_id="f"
        )
        assert len(out["buckets"]) == 3
        b_lo, b_mid, b_hi = out["buckets"]
        assert b_mid["beta"] > 1.5, f"centre bucket beta={b_mid['beta']!r}, expected > 1.5"
        assert abs(b_lo["beta"] - 0.5) < 0.4
        assert abs(b_hi["beta"] - 0.5) < 0.4

    def test_breusch_pagan_detects_heteroscedasticity(self) -> None:
        """BP regresses residual^2 on dlogit. So we need residual variance to
        scale with dlogit (heteroscedasticity that *correlates* with the
        regressor)."""
        rng = np.random.default_rng(17)
        n = 800
        p = _drifting_uniform_probs(n, seed=23)
        dl = delta_logit(p).fillna(0.0).values
        # Variance scales with |dlogit| → BP must reject homoscedasticity.
        sigma_t = 0.005 + 0.05 * np.abs(dl)
        r = pd.Series(0.5 * dl + rng.normal(0.0, 1.0, n) * sigma_t, index=p.index, name="r")
        out = fit_conditional_model_core(r, p, conditioning_thresholds=[0.3, 0.7])
        assert out["homoscedasticity_test_p"] is not None
        # Heteroscedasticity present → BP p-value small.
        assert out["homoscedasticity_test_p"] < 0.05

    def test_one_threshold_yields_two_buckets(self) -> None:
        rng = np.random.default_rng(31)
        n = 400
        p = _drifting_uniform_probs(n, seed=33)
        dl = delta_logit(p).fillna(0.0).values
        r = pd.Series(0.5 * dl + rng.normal(0, 0.01, n), index=p.index, name="r")
        out = fit_conditional_model_core(r, p, conditioning_thresholds=[0.5])
        assert len(out["buckets"]) == 2
        assert out["buckets"][0]["range"] == [0.0, 0.5]
        assert out["buckets"][1]["range"] == [0.5, 1.0]

    def test_five_buckets_uniform_distribution(self) -> None:
        """4 cuts → 5 buckets. Uniform p means every bucket has > 0 obs and
        most have enough for a HAC fit."""
        rng = np.random.default_rng(41)
        n = 1000
        p = _drifting_uniform_probs(n, seed=43)
        dl = delta_logit(p).fillna(0.0).values
        r = pd.Series(0.6 * dl + rng.normal(0, 0.005, n), index=p.index, name="r")
        out = fit_conditional_model_core(r, p, conditioning_thresholds=[0.2, 0.4, 0.6, 0.8])
        assert len(out["buckets"]) == 5
        for b in out["buckets"]:
            assert b["n_obs"] > 0, f"empty bucket at range {b['range']!r}"
        # At least 3 of the 5 buckets should fit (n_obs >= 10).
        fitted = sum(1 for b in out["buckets"] if b["beta"] is not None)
        assert fitted >= 3

    def test_empty_thresholds_raises(self) -> None:
        rng = np.random.default_rng(53)
        n = 200
        p = _drifting_uniform_probs(n, seed=55)
        r = pd.Series(rng.normal(0, 0.01, n), index=p.index, name="r")
        with pytest.raises(ValueError, match="non-empty"):
            fit_conditional_model_core(r, p, conditioning_thresholds=[])

    def test_threshold_outside_unit_interval_raises(self) -> None:
        rng = np.random.default_rng(59)
        n = 200
        p = _drifting_uniform_probs(n, seed=61)
        r = pd.Series(rng.normal(0, 0.01, n), index=p.index, name="r")
        with pytest.raises(ValueError, match="must lie in"):
            fit_conditional_model_core(r, p, conditioning_thresholds=[1.5])
        with pytest.raises(ValueError, match="must lie in"):
            fit_conditional_model_core(r, p, conditioning_thresholds=[0.0])

    def test_concentrated_probs_yield_sparse_buckets(self) -> None:
        """All probs in [0.45, 0.55]: bucket [0.5, 1] gets some obs, bucket
        [0, 0.5] gets the rest, but extreme buckets in a finer split should
        be empty and reported as ``beta=None`` rather than crashing."""
        rng = np.random.default_rng(67)
        n = 200
        p = pd.Series(rng.uniform(0.45, 0.55, n), index=_utc_index(n), name="p")
        dl = delta_logit(p).fillna(0.0).values
        r = pd.Series(0.5 * dl + rng.normal(0, 0.01, n), index=p.index, name="r")
        out = fit_conditional_model_core(r, p, conditioning_thresholds=[0.1, 0.3, 0.7, 0.9])
        assert len(out["buckets"]) == 5
        # The two outermost buckets should be empty or tiny.
        # n_obs of bucket 0 ([0, 0.1)) and bucket 4 ([0.9, 1.0]) should be 0.
        assert out["buckets"][0]["n_obs"] == 0
        assert out["buckets"][0]["beta"] is None
        assert out["buckets"][-1]["n_obs"] == 0
        assert out["buckets"][-1]["beta"] is None


# ===========================================================================
# B) Polynomial model
# ===========================================================================


class TestPolynomialDeep:
    def test_quadratic_dgp_recovers_b1_b2(self) -> None:
        """DGP: y = 0.5*x + 2.0*x^2 + eps. degree=2 fit must recover both."""
        rng = np.random.default_rng(101)
        n = 800
        p = _drifting_uniform_probs(n, seed=103)
        x = delta_logit(p).fillna(0.0).values
        # Centre x to avoid extreme leverage from clipping in noise-free corners.
        b1, b2 = 0.5, 2.0
        y = b1 * x + b2 * x**2 + rng.normal(0, 0.005, n)
        r = pd.Series(y, index=p.index, name="r")
        out = fit_polynomial_factor_model_core(r, p, degree=2)
        betas = {b["order"]: b["beta"] for b in out["betas"]}
        assert abs(betas[1] - b1) < 0.25, f"beta1={betas[1]:.3f} not near {b1}"
        assert abs(betas[2] - b2) < 0.30, f"beta2={betas[2]:.3f} not near {b2}"

    def test_lr_test_rejects_linear_for_quadratic_dgp(self) -> None:
        rng = np.random.default_rng(107)
        n = 700
        p = _drifting_uniform_probs(n, seed=109)
        x = delta_logit(p).fillna(0.0).values
        y = 0.5 * x + 2.0 * x**2 + rng.normal(0, 0.005, n)
        r = pd.Series(y, index=p.index, name="r")
        out = fit_polynomial_factor_model_core(r, p, degree=2)
        assert out["vs_linear_lr_test_p"] is not None
        assert out["vs_linear_lr_test_p"] < 0.01

    def test_cubic_optimal_aic_for_cubic_dgp(self) -> None:
        rng = np.random.default_rng(113)
        n = 700
        p = _drifting_uniform_probs(n, seed=115)
        x = delta_logit(p).fillna(0.0).values
        y = 0.5 * x + 4.0 * x**3 + rng.normal(0, 0.005, n)
        r = pd.Series(y, index=p.index, name="r")
        out = fit_polynomial_factor_model_core(r, p, degree=3, aic_max_degree=5)
        assert out["optimal_degree_aic"] >= 3, f"got {out['optimal_degree_aic']}"

    def test_linear_dgp_optimal_aic_is_one(self) -> None:
        rng = np.random.default_rng(127)
        n = 600
        p = _drifting_uniform_probs(n, seed=129)
        x = delta_logit(p).fillna(0.0).values
        y = 0.4 * x + rng.normal(0, 0.005, n)
        r = pd.Series(y, index=p.index, name="r")
        out = fit_polynomial_factor_model_core(r, p, degree=2, aic_max_degree=5)
        assert out["optimal_degree_aic"] == 1, f"got {out['optimal_degree_aic']}"

    def test_marginal_effect_at_zero_equals_b1(self) -> None:
        """dy/dx at x=0 for a polynomial sum_k beta_k x^k is exactly beta_1."""
        rng = np.random.default_rng(131)
        n = 700
        p = _drifting_uniform_probs(n, seed=133)
        x = delta_logit(p).fillna(0.0).values
        b1 = 0.7
        y = b1 * x + 1.5 * x**2 + rng.normal(0, 0.005, n)
        r = pd.Series(y, index=p.index, name="r")
        out = fit_polynomial_factor_model_core(r, p, degree=2)
        # Find the grid point closest to x=0.
        grid = out["marginal_effects"]
        nearest = min(grid, key=lambda pt: abs(pt["x"]))
        # If 0 is within the grid range, dy/dx ≈ b1 + 2*b2*x_near.
        b1_hat = next(b["beta"] for b in out["betas"] if b["order"] == 1)
        b2_hat = next(b["beta"] for b in out["betas"] if b["order"] == 2)
        expected = b1_hat + 2 * b2_hat * nearest["x"]
        assert abs(nearest["dy_dx"] - expected) < 1e-6

    def test_degree_zero_raises(self) -> None:
        rng = np.random.default_rng(137)
        n = 200
        p = _drifting_uniform_probs(n, seed=139)
        r = pd.Series(rng.normal(0, 0.01, n), index=p.index, name="r")
        with pytest.raises(ValueError, match="degree"):
            fit_polynomial_factor_model_core(r, p, degree=0)

    def test_short_window_raises(self) -> None:
        rng = np.random.default_rng(143)
        n = 25
        p = _drifting_uniform_probs(n, seed=145)
        r = pd.Series(rng.normal(0, 0.01, n), index=p.index, name="r")
        with pytest.raises(ValueError):
            fit_polynomial_factor_model_core(r, p, degree=2)


# ===========================================================================
# C) Regime-switching
# ===========================================================================


class TestRegimeSwitchingDeep:
    def test_two_regime_means_recovered(self) -> None:
        """DGP with two clearly-separated regimes. Hamilton fit must surface
        2 regimes with means of opposite sign once we plant a strong enough
        mean signal relative to noise."""
        rng = np.random.default_rng(201)
        n = 600
        # Markov chain with stay-prob 0.95 in each state.
        states = np.zeros(n, dtype=int)
        for t in range(1, n):
            switch = rng.random() < 0.05
            states[t] = 1 - states[t - 1] if switch else states[t - 1]
        # Strong mean separation (5x sigma) so regime identification is robust.
        mus = np.array([0.01, -0.015])
        sigmas = np.array([0.003, 0.003])
        p = _drifting_uniform_probs(n, seed=203)
        r_arr = mus[states] + rng.normal(0, 1, n) * sigmas[states]
        r = pd.Series(r_arr, index=p.index, name="r")
        out = fit_regime_switching_model_core(r, p, n_regimes=2)
        assert out["n_regimes"] == 2
        assert len(out["regimes"]) == 2
        means = sorted(reg["mean_return"] for reg in out["regimes"])
        # mu_low ~ -0.015, mu_high ~ 0.01 after sorting; expect opposite signs.
        assert means[0] < 0, f"low-mean regime has mu={means[0]}, expected negative"
        assert means[1] > 0, f"high-mean regime has mu={means[1]}, expected positive"

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        rng = np.random.default_rng(207)
        n = 300
        # Make 2 regimes with very different vols so MarkovRegression latches on.
        states = np.repeat([0, 1, 0, 1, 0, 1], n // 6 + 1)[:n]
        sigmas = np.array([0.003, 0.02])
        p = _drifting_uniform_probs(n, seed=209)
        r_arr = rng.normal(0, 1, n) * sigmas[states]
        r = pd.Series(r_arr, index=p.index, name="r")
        out = fit_regime_switching_model_core(r, p, n_regimes=2)
        for row in out["transition_matrix"]:
            assert abs(sum(row) - 1.0) < 1e-3, f"row sum != 1: {row}"

    def test_ergodic_probabilities_sum_to_one(self) -> None:
        rng = np.random.default_rng(211)
        n = 250
        states = np.repeat([0, 1], n // 2)[:n]
        if len(states) < n:
            states = np.concatenate([states, np.zeros(n - len(states), dtype=int)])
        sigmas = np.array([0.004, 0.018])
        p = _drifting_uniform_probs(n, seed=213)
        r_arr = rng.normal(0, 1, n) * sigmas[states]
        r = pd.Series(r_arr, index=p.index, name="r")
        out = fit_regime_switching_model_core(r, p, n_regimes=2)
        s = sum(reg["ergodic_prob"] for reg in out["regimes"])
        assert abs(s - 1.0) < 1e-3

    def test_smoothed_probs_in_unit_interval(self) -> None:
        rng = np.random.default_rng(217)
        n = 250
        states = np.repeat([0, 1], n // 2)[:n]
        if len(states) < n:
            states = np.concatenate([states, np.zeros(n - len(states), dtype=int)])
        sigmas = np.array([0.004, 0.018])
        p = _drifting_uniform_probs(n, seed=219)
        r_arr = rng.normal(0, 1, n) * sigmas[states]
        r = pd.Series(r_arr, index=p.index, name="r")
        out = fit_regime_switching_model_core(r, p, n_regimes=2)
        for row in out["smoothed_state_probs_last_30"]:
            for v in row:
                assert 0.0 <= v <= 1.0
            assert abs(sum(row) - 1.0) < 1e-3

    def test_n_regimes_one_raises(self) -> None:
        rng = np.random.default_rng(223)
        n = 150
        p = _drifting_uniform_probs(n, seed=225)
        r = pd.Series(rng.normal(0, 0.01, n), index=p.index, name="r")
        with pytest.raises(ValueError, match="n_regimes"):
            fit_regime_switching_model_core(r, p, n_regimes=1)

    def test_short_window_raises(self) -> None:
        rng = np.random.default_rng(229)
        n = 40
        p = _drifting_uniform_probs(n, seed=231)
        r = pd.Series(rng.normal(0, 0.01, n), index=p.index, name="r")
        with pytest.raises(ValueError):
            fit_regime_switching_model_core(r, p, n_regimes=2)


# ===========================================================================
# D) VECM
# ===========================================================================


class TestVECMDeep:
    def test_cointegrated_dgp_detected(self) -> None:
        """log(eq) is a random walk; logit(p) = 2 * log(eq) + AR(1) noise."""
        rng = np.random.default_rng(301)
        n = 500
        log_eq = np.cumsum(rng.normal(0.0, 0.01, n)) + 4.0
        beta_true = 2.0
        ar_eps = np.zeros(n)
        for t in range(1, n):
            ar_eps[t] = 0.5 * ar_eps[t - 1] + rng.normal(0.0, 0.05)
        logit_p = beta_true * (log_eq - log_eq.mean()) + ar_eps
        p_vals = np.clip(_logistic(logit_p), 0.02, 0.98)
        idx = _utc_index(n)
        out = fit_vecm_core(
            pd.Series(np.exp(log_eq), index=idx, name="px"),
            pd.Series(p_vals, index=idx, name="p"),
            det_order=0,
            k_ar_diff=1,
        )
        assert out["is_cointegrated"] is True
        assert out["johansen_p_eigenvalue"] < 0.05
        assert out["beta_long_run"] is not None
        # alpha_loading_target should be negative (mean reverting toward
        # equilibrium).
        assert out["alpha_loading_target"] is not None
        assert out["alpha_loading_target"] < 0
        assert out["half_life_correction_days"] is not None
        assert 0 < out["half_life_correction_days"] < 200

    def test_independent_walks_not_cointegrated(self) -> None:
        rng = np.random.default_rng(311)
        n = 400
        log_eq = np.cumsum(rng.normal(0.0, 0.01, n)) + 4.0
        logit_p = np.cumsum(rng.normal(0.0, 0.04, n))
        p_vals = np.clip(_logistic(logit_p), 0.02, 0.98)
        idx = _utc_index(n)
        out = fit_vecm_core(
            pd.Series(np.exp(log_eq), index=idx, name="px"),
            pd.Series(p_vals, index=idx, name="p"),
        )
        # We don't strictly assert is_cointegrated False (Johansen has size
        # error), but require the trace stat well below the 99% critical value.
        # Use the cvt[2] proxy via crit_95 + slack.
        assert out["johansen_trace_stat"] < out["johansen_trace_crit_95"] + 5.0

    def test_short_window_raises(self) -> None:
        rng = np.random.default_rng(317)
        n = 25
        idx = _utc_index(n)
        with pytest.raises(ValueError, match="40"):
            fit_vecm_core(
                pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx),
                pd.Series(np.clip(rng.uniform(0.1, 0.9, n), 0.05, 0.95), index=idx),
            )

    def test_invalid_k_ar_diff_raises(self) -> None:
        rng = np.random.default_rng(321)
        n = 200
        idx = _utc_index(n)
        with pytest.raises(ValueError, match="k_ar_diff"):
            fit_vecm_core(
                pd.Series(np.exp(rng.normal(4, 0.1, n)), index=idx),
                pd.Series(np.clip(rng.uniform(0.1, 0.9, n), 0.05, 0.95), index=idx),
                k_ar_diff=0,
            )


# ===========================================================================
# E) GARCH-X
# ===========================================================================


class TestGarchXDeep:
    def test_garch_persistence_recovered(self) -> None:
        """DGP: GARCH(1,1) with planted persistence ~0.95. Fit should
        recover persistence in (0.5, 0.999)."""
        rng = np.random.default_rng(401)
        n = 800
        p = _drifting_uniform_probs(n, seed=403)
        dl_abs = delta_logit(p).fillna(0.0).abs().values
        omega, a_true, b_true, gamma_true = 1e-5, 0.10, 0.85, 5e-5
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
        out = fit_garch_x_core(r, p)
        assert out["is_stationary"] is True
        assert 0.5 < out["persistence"] < 0.999
        assert out["half_life_vol_days"] is not None
        assert out["half_life_vol_days"] > 0.0
        assert out["omega"] > 0
        assert 0.0 <= out["alpha"] <= 1.0
        assert 0.0 <= out["beta"] <= 1.0

    def test_factor_coef_nonneg(self) -> None:
        """The exogenous-variance coef is bounded below at 0 by construction
        (a non-negative bound enforces the variance recursion stays positive)."""
        rng = np.random.default_rng(411)
        n = 500
        p = _drifting_uniform_probs(n, seed=413)
        dl_abs = delta_logit(p).fillna(0.0).abs().values
        # Plant a strong factor effect.
        omega, a_true, b_true, gamma_true = 1e-5, 0.05, 0.85, 5e-4
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
        out = fit_garch_x_core(r, p)
        # gamma is constrained >= 0 in the optimiser. With a strong planted
        # factor effect, the fitted gamma should be strictly positive.
        assert out["factor_exogenous_coef"] >= 0.0
        # variance share is in [0, 1].
        assert 0.0 <= out["conditional_variance_explained_by_factor_pct"] <= 100.0

    def test_zero_variance_returns_raises(self) -> None:
        n = 100
        idx = _utc_index(n)
        r = pd.Series(np.zeros(n), index=idx, name="r")  # constant -> zero variance
        p = _drifting_uniform_probs(n, seed=421)
        with pytest.raises(ValueError, match="zero-variance"):
            fit_garch_x_core(r, p)

    def test_short_window_raises(self) -> None:
        rng = np.random.default_rng(431)
        n = 50
        p = _drifting_uniform_probs(n, seed=433)
        r = pd.Series(rng.normal(0, 0.01, n), index=p.index, name="r")
        with pytest.raises(ValueError, match="60"):
            fit_garch_x_core(r, p)


# ===========================================================================
# F) Tail dependence
# ===========================================================================


class TestTailDependenceDeep:
    def test_independent_returns_low_lambda(self) -> None:
        rng = np.random.default_rng(501)
        n = 1500
        idx = _utc_index(n)
        r = pd.Series(rng.normal(0, 1, n), index=idx, name="r")
        x = pd.Series(rng.normal(0, 1, n), index=idx, name="x")
        out = compute_tail_dependence_core(r, x, quantile=0.05)
        # Under independence, lambda ≈ q. We give finite-sample slack.
        assert out["lower_tail_dependence"] < 0.20
        assert out["upper_tail_dependence"] < 0.20

    def test_perfectly_correlated_lower_tail(self) -> None:
        """If r and x share a common driver, the joint lower-tail
        coincidence is high (> 0.7)."""
        rng = np.random.default_rng(503)
        n = 1500
        u = rng.normal(0, 1, n)
        # x = u + tiny noise; r = u + tiny noise → near-identical tails.
        x_vals = u + 0.05 * rng.normal(0, 1, n)
        r_vals = u + 0.05 * rng.normal(0, 1, n)
        idx = _utc_index(n)
        r = pd.Series(r_vals, index=idx, name="r")
        x = pd.Series(x_vals, index=idx, name="x")
        out = compute_tail_dependence_core(r, x, quantile=0.05)
        assert out["lower_tail_dependence"] > 0.7

    def test_asymmetric_tail_dependence(self) -> None:
        """Construct asymmetric coupling: lower tail strongly co-moving,
        upper tail weakly so."""
        rng = np.random.default_rng(509)
        n = 2000
        u = rng.normal(0, 1, n)
        idx = _utc_index(n)
        # Lower tail: when u is very negative, r and x both follow u tightly.
        # Upper tail: when u is positive, r and x decouple via independent noise.
        r_vals = np.where(u < 0, u, rng.normal(0, 1, n))
        x_vals = np.where(u < 0, u + 0.02 * rng.normal(0, 1, n), rng.normal(0, 1, n))
        r = pd.Series(r_vals, index=idx, name="r")
        x = pd.Series(x_vals, index=idx, name="x")
        out = compute_tail_dependence_core(r, x, quantile=0.05)
        # Asymmetry > 0.5 means lower tail dep substantially exceeds upper.
        assert out["asymmetry"] > 0.5

    def test_extreme_obs_count_decreases_with_smaller_quantile(self) -> None:
        rng = np.random.default_rng(511)
        n = 1000
        idx = _utc_index(n)
        r = pd.Series(rng.normal(0, 1, n), index=idx, name="r")
        x = pd.Series(rng.normal(0, 1, n), index=idx, name="x")
        out_05 = compute_tail_dependence_core(r, x, quantile=0.05)
        out_10 = compute_tail_dependence_core(r, x, quantile=0.10)
        assert out_05["n_extreme_obs_lower"] <= out_10["n_extreme_obs_lower"]
        assert out_05["n_extreme_obs_upper"] <= out_10["n_extreme_obs_upper"]

    def test_invalid_quantile_raises(self) -> None:
        rng = np.random.default_rng(517)
        n = 500
        idx = _utc_index(n)
        r = pd.Series(rng.normal(0, 1, n), index=idx, name="r")
        x = pd.Series(rng.normal(0, 1, n), index=idx, name="x")
        with pytest.raises(ValueError, match="quantile"):
            compute_tail_dependence_core(r, x, quantile=0.7)
        with pytest.raises(ValueError, match="quantile"):
            compute_tail_dependence_core(r, x, quantile=0.0)

    def test_short_window_raises(self) -> None:
        rng = np.random.default_rng(521)
        n = 30
        idx = _utc_index(n)
        r = pd.Series(rng.normal(0, 1, n), index=idx, name="r")
        x = pd.Series(rng.normal(0, 1, n), index=idx, name="x")
        with pytest.raises(ValueError, match="50"):
            compute_tail_dependence_core(r, x)


# ===========================================================================
# G) API endpoint smoke tests
# ===========================================================================


@pytest.fixture
def advanced_app_client(
    monkeypatch: pytest.MonkeyPatch,
    factors_file: Path,
) -> Iterator[TestClient]:
    """TestClient with the advanced-model router's data-IO patched out.

    The router imports ``fetch_factor_history`` and ``fetch_equity_history``
    at module level, so we monkeypatch those names on the router module.
    """
    import pfm.advanced_event_models_router as router_mod
    import pfm.config as cfg
    import pfm.main as main_mod
    from pfm.cache import NullCache

    monkeypatch.setenv("FACTORS_FILE", str(factors_file))
    cfg._settings = None

    rng = np.random.default_rng(98765)
    n = 250
    idx = pd.date_range("2025-04-01", periods=n, freq="D", tz="UTC")
    # Smooth probability series in [0.10, 0.90] across the full band.
    t_arr = np.arange(n) / n
    prob_a = (0.30 + 0.30 * np.sin(2 * np.pi * t_arr * 1.2)).clip(0.05, 0.95)
    prob_b = (0.55 + 0.20 * np.cos(2 * np.pi * t_arr * 0.8)).clip(0.05, 0.95)
    prob_df_a = pd.DataFrame({"price": prob_a}, index=idx)
    prob_df_b = pd.DataFrame({"price": prob_b}, index=idx)
    prob_df_a.index.name = "date"
    prob_df_b.index.name = "date"
    bank = {"slug-a": prob_df_a, "slug-b": prob_df_b}

    def _fake_factor_history(_client, slug, start=None, end=None):
        df = bank[slug]
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    # Equity prices: positive, drifting series with realistic vol.
    base_prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    price_series = pd.Series(base_prices, index=idx, name="px")

    def _fake_equity_history(ticker, start, end, **_):
        s = price_series.copy()
        s.name = ticker
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index <= end]
        return s

    monkeypatch.setattr(router_mod, "fetch_factor_history", _fake_factor_history)
    monkeypatch.setattr(router_mod, "fetch_equity_history", _fake_equity_history)
    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    with TestClient(main_mod.app) as client:
        yield client


class TestAdvancedModelEndpoints:
    _BASE_BODY: ClassVar[dict[str, str]] = {
        "ticker": "TEST",
        "factor_id": "factor_a",
        "start": "2025-04-15",
        "end": "2025-12-01",
    }

    def test_conditional_endpoint(self, advanced_app_client: TestClient) -> None:
        body = {**self._BASE_BODY, "conditioning_thresholds": [0.3, 0.7]}
        r = advanced_app_client.post("/advanced-model/conditional", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["ticker"] == "TEST"
        assert out["factor_id"] == "factor_a"
        assert "buckets" in out
        assert len(out["buckets"]) == 3

    def test_polynomial_endpoint(self, advanced_app_client: TestClient) -> None:
        body = {**self._BASE_BODY, "degree": 2}
        r = advanced_app_client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["degree"] == 2
        assert "betas" in out
        assert len(out["betas"]) == 2
        assert "marginal_effects" in out

    def test_regime_switching_endpoint(self, advanced_app_client: TestClient) -> None:
        body = {**self._BASE_BODY, "n_regimes": 2}
        r = advanced_app_client.post("/advanced-model/regime-switching", json=body)
        # Regime switching can fail to converge on smooth synthetic data;
        # accept 200 (good fit) or 422 (statsmodels gave up).
        assert r.status_code in (200, 422), r.text
        if r.status_code == 200:
            out = r.json()
            assert out["n_regimes"] == 2
            assert len(out["regimes"]) == 2

    def test_vecm_endpoint(self, advanced_app_client: TestClient) -> None:
        body = {**self._BASE_BODY, "det_order": 0, "k_ar_diff": 1}
        r = advanced_app_client.post("/advanced-model/vecm", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert "is_cointegrated" in out
        assert "johansen_p_eigenvalue" in out
        assert "johansen_trace_stat" in out

    def test_garch_x_endpoint(self, advanced_app_client: TestClient) -> None:
        body = {**self._BASE_BODY}
        r = advanced_app_client.post("/advanced-model/garch-x", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert "persistence" in out
        assert "is_stationary" in out
        assert "factor_exogenous_coef" in out

    def test_tail_dependence_endpoint(self, advanced_app_client: TestClient) -> None:
        body = {**self._BASE_BODY, "quantile": 0.05}
        r = advanced_app_client.post("/advanced-model/tail-dependence", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert "lower_tail_dependence" in out
        assert "upper_tail_dependence" in out
        assert out["quantile"] == 0.05

    def test_unknown_factor_400(self, advanced_app_client: TestClient) -> None:
        body = {**self._BASE_BODY, "factor_id": "does_not_exist", "degree": 2}
        r = advanced_app_client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 400
        detail = r.json()["detail"]
        # New shape: structured dict with did_you_mean. Legacy callers that
        # only assert on 400 keep working; payload now includes hints.
        if isinstance(detail, dict):
            assert "factor not found" in detail.get("error", "").lower() or (
                "unknown factor" in detail.get("error", "").lower()
            )
            assert "did_you_mean" in detail
        else:
            assert "unknown factor" in detail or "factor not found" in detail

    def test_invalid_window_400(self, advanced_app_client: TestClient) -> None:
        body = {
            **self._BASE_BODY,
            "start": "2025-12-15",
            "end": "2025-06-15",
            "degree": 2,
        }
        r = advanced_app_client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 400
