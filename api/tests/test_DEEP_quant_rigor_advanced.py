"""Deep tests for the advanced quant-rigor additions.

Modules covered:

* :mod:`pfm.mhm_critical` - MacKinnon-Haug-Michelis Johansen p-values.
* :mod:`pfm.oos_metrics`  - Campbell-Thompson R^2_OOS + Clark-West.
* :mod:`pfm.forecast_comparison` - Diebold-Mariano + HLN correction.
* :mod:`pfm.whites_reality_check` - White RC + Hansen SPA + Romano-Wolf.
* :mod:`pfm.multitest`    - Bailey-Lopez de Prado deflated_sharpe_full.

Tests are synthetic-DGP recovery checks where possible. No network IO,
no live API calls.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pfm.forecast_comparison import diebold_mariano
from pfm.mhm_critical import johansen_pvalue
from pfm.multitest import deflated_sharpe_full
from pfm.oos_metrics import oos_r_squared_campbell_thompson
from pfm.quant_rigor_advanced_router import router as qr_router
from pfm.whites_reality_check import stepwise_spa, whites_reality_check

# ---------------------------------------------------------------------------
# 1) MacKinnon-Haug-Michelis p-values
# ---------------------------------------------------------------------------


class TestMHMPvalues:
    """The MHM Gamma response surface should be calibrated to the
    Osterwald-Lenum/Johansen 1995 critical value tables."""

    @pytest.mark.parametrize(
        ("test", "n_vars", "det_order", "stat", "expected_p"),
        [
            # trace, n=2, det=0: cv90/95/99 = 17.85/19.96/24.60
            ("trace", 2, 0, 17.85, 0.10),
            ("trace", 2, 0, 24.60, 0.01),
            ("trace", 3, 0, 32.00, 0.10),
            ("trace", 3, 0, 41.07, 0.01),
            # eigen
            ("eigen", 2, 0, 13.75, 0.10),
            ("eigen", 2, 0, 20.20, 0.01),
            ("eigen", 3, 0, 19.77, 0.10),
            ("eigen", 3, 0, 26.81, 0.01),
            # det=-1
            ("trace", 2, -1, 10.47, 0.10),
            ("trace", 2, -1, 16.36, 0.01),
            # det=1
            ("trace", 2, 1, 22.95, 0.10),
            ("trace", 2, 1, 30.45, 0.01),
        ],
    )
    def test_pvalue_at_table_boundaries(
        self,
        test,
        n_vars,
        det_order,
        stat,
        expected_p,
    ):
        """At each tabulated 90%/99% boundary the MHM p-value should match
        the bucket exactly (0.10 / 0.01)."""
        p = johansen_pvalue(stat, n_vars=n_vars, det_order=det_order, test=test)
        assert abs(p - expected_p) < 0.005, (
            f"{test} n={n_vars} det={det_order} stat={stat}: expected ~{expected_p}, got {p}"
        )

    def test_p_value_at_95_boundary_is_close(self):
        # Mid-bucket (95%) should land within 0.01 of 0.05.
        for test, det, n, cv95 in [
            ("trace", 0, 2, 19.96),
            ("trace", 0, 3, 34.91),
            ("eigen", 0, 2, 15.67),
        ]:
            p = johansen_pvalue(cv95, n_vars=n, det_order=det, test=test)
            assert abs(p - 0.05) < 0.01, f"95% {test} det={det} n={n}: p={p}"

    def test_monotone_decreasing_in_stat(self):
        # As statistic grows, p-value must monotonically decrease.
        prev = 1.0
        for stat in np.linspace(1.0, 80.0, 50):
            p = johansen_pvalue(stat, n_vars=3, det_order=0, test="trace")
            assert p <= prev + 1e-12, f"non-monotone at stat={stat}: {p} > {prev}"
            prev = p

    def test_finer_grained_than_bucketed(self):
        # The bucketed lookup returns one of {0.005, 0.025, 0.075, 0.20}.
        # The continuous p-value must populate intermediate values.
        stats = np.linspace(15.0, 25.0, 20)
        pvals = [johansen_pvalue(s, n_vars=2, det_order=0, test="trace") for s in stats]
        unique = {round(p, 4) for p in pvals}
        assert len(unique) >= 15, f"too coarse: {len(unique)} unique p-values"

    def test_negative_stat_returns_one(self):
        assert johansen_pvalue(-1.0, n_vars=2, det_order=0, test="trace") == 1.0
        assert johansen_pvalue(0.0, n_vars=2, det_order=0, test="trace") == 1.0

    def test_unsupported_n_vars_raises(self):
        with pytest.raises(KeyError):
            johansen_pvalue(50.0, n_vars=10, det_order=0, test="trace")

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            johansen_pvalue(10.0, n_vars=2, det_order=2, test="trace")
        with pytest.raises(ValueError):
            johansen_pvalue(10.0, n_vars=2, det_order=0, test="bogus")


# ---------------------------------------------------------------------------
# 2) Campbell-Thompson R^2_OOS + Clark-West
# ---------------------------------------------------------------------------


class TestCampbellThompson:
    def test_r2_oos_positive_when_model_outperforms(self):
        rng = np.random.default_rng(42)
        n = 200
        y = rng.standard_normal(n) * 0.05
        # Model = y + small noise; baseline = pure noise far from y.
        y_model = y + rng.standard_normal(n) * 0.02
        y_base = rng.standard_normal(n) * 0.05  # uncorrelated baseline
        out = oos_r_squared_campbell_thompson(y, y_model, y_base)
        assert out["r_squared_oos"] > 0.5, out
        assert out["model_beats_baseline"]
        assert out["mse_model"] < out["mse_baseline"]
        assert out["hac_t_stat_clark_west"] > 0

    def test_r2_oos_negative_when_model_worse(self):
        rng = np.random.default_rng(7)
        n = 200
        y = rng.standard_normal(n) * 0.05
        y_base = y * 0.5
        # model adds a lot of pure noise.
        y_model = y_base + rng.standard_normal(n) * 0.5
        out = oos_r_squared_campbell_thompson(y, y_model, y_base, nested=False)
        assert out["r_squared_oos"] < 0
        assert not out["model_beats_baseline"]

    def test_clark_west_significant_when_model_truly_better(self):
        rng = np.random.default_rng(123)
        n = 500
        signal = 0.1 * rng.standard_normal(n)
        noise = 0.05 * rng.standard_normal(n)
        y = signal + noise
        y_model = signal  # captures the truth
        y_base = np.full(n, y.mean())  # historical mean baseline
        out = oos_r_squared_campbell_thompson(y, y_model, y_base, nested=True)
        assert out["hac_p_value"] < 0.01

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            oos_r_squared_campbell_thompson([1.0, 2.0], [1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# 3) Diebold-Mariano
# ---------------------------------------------------------------------------


class TestDieboldMariano:
    def test_dm_prefers_model2_when_smaller_errors(self):
        rng = np.random.default_rng(11)
        n = 400
        e1 = rng.standard_normal(n) * 1.0
        e2 = rng.standard_normal(n) * 0.5  # consistently smaller
        out = diebold_mariano(e1, e2, h=1, loss="MSE")
        assert out["dm_stat"] > 0
        assert out["p_value"] < 0.05
        assert out["prefer_model"] == 2

    def test_dm_tie_when_same_distribution(self):
        # Average across multiple seeds: under H0 the per-seed rejection
        # rate should be close to 5%; with 20 seeds we expect at most a
        # few rejections.
        n_rejects = 0
        for seed in range(20):
            rng = np.random.default_rng(seed)
            n = 300
            e1 = rng.standard_normal(n)
            e2 = rng.standard_normal(n)
            out = diebold_mariano(e1, e2, h=1, loss="MSE")
            if out["p_value"] < 0.05:
                n_rejects += 1
        # Under the null nominal level is 5%; out of 20 seeds, anything
        # >= 6 rejections would indicate real size distortion.
        assert n_rejects <= 5, f"too many rejections under null: {n_rejects}/20"

    def test_dm_prefers_model1_when_smaller_errors(self):
        rng = np.random.default_rng(99)
        n = 400
        e1 = rng.standard_normal(n) * 0.3
        e2 = rng.standard_normal(n) * 1.2
        out = diebold_mariano(e1, e2, h=1, loss="MSE")
        assert out["dm_stat"] < 0
        assert out["p_value"] < 0.05
        assert out["prefer_model"] == 1

    def test_dm_mae_loss(self):
        rng = np.random.default_rng(13)
        n = 200
        e1 = rng.standard_normal(n) * 1.0
        e2 = rng.standard_normal(n) * 0.5
        out = diebold_mariano(e1, e2, h=1, loss="MAE")
        assert out["loss"] == "MAE"
        assert out["dm_stat"] > 0

    def test_hln_correction_shrinks_stat_for_long_horizons(self):
        rng = np.random.default_rng(21)
        n = 100
        e1 = rng.standard_normal(n) * 1.0
        e2 = rng.standard_normal(n) * 0.7
        out = diebold_mariano(e1, e2, h=4, loss="MSE")
        assert abs(out["dm_stat_hln"]) <= abs(out["dm_stat"]) + 1e-12

    def test_dm_invalid_input(self):
        with pytest.raises(ValueError):
            diebold_mariano([1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            diebold_mariano(np.zeros(10), np.zeros(10), h=0)


# ---------------------------------------------------------------------------
# 4) White Reality Check + Romano-Wolf stepwise
# ---------------------------------------------------------------------------


class TestWhitesRealityCheck:
    def test_white_pvalue_significant_when_real_edge(self):
        rng = np.random.default_rng(1)
        n, k = 500, 100
        # 99 noise strategies + 1 strategy with positive drift.
        noise = rng.standard_normal((n, k)) * 0.01
        noise[:, 0] += 0.005  # 50 bps daily drift on column 0
        bench = np.zeros(n)
        out = whites_reality_check(
            noise,
            bench,
            n_bootstrap=400,
            seed=42,
        )
        assert out["best_strategy_idx"] == 0
        assert out["white_pvalue"] < 0.05
        assert out["hansen_spa_pvalue"] < 0.05

    def test_white_pvalue_insignificant_when_pure_noise(self):
        rng = np.random.default_rng(2)
        n, k = 300, 50
        noise = rng.standard_normal((n, k)) * 0.01
        bench = np.zeros(n)
        out = whites_reality_check(
            noise,
            bench,
            n_bootstrap=400,
            seed=42,
        )
        # Under pure null the p-value is uniform on (0,1) so single trials
        # are noisy. Just check that we don't reject at 0.01.
        assert out["white_pvalue"] > 0.05 or out["hansen_spa_pvalue"] > 0.05

    def test_stepwise_spa_recovers_real_edges(self):
        rng = np.random.default_rng(3)
        n, k = 600, 100
        rets = rng.standard_normal((n, k)) * 0.01
        # Add 5 real edges with strong drift.
        for j in range(5):
            rets[:, j] += 0.008
        bench = np.zeros(n)
        out = stepwise_spa(
            rets,
            bench,
            alpha=0.05,
            n_bootstrap=400,
            seed=11,
        )
        rejected = set(out["rejected_strategy_indices"])
        true_edges = {0, 1, 2, 3, 4}
        # FWER control: rejections among the 95 noisers should be rare.
        false_rejects = rejected - true_edges
        assert len(false_rejects) <= 5, (rejected, false_rejects)
        # We should recover most of the 5 real edges.
        assert len(rejected & true_edges) >= 4

    def test_stepwise_spa_no_false_rejections_under_null(self):
        rng = np.random.default_rng(4)
        n, k = 400, 30
        rets = rng.standard_normal((n, k)) * 0.01
        bench = np.zeros(n)
        out = stepwise_spa(
            rets,
            bench,
            alpha=0.05,
            n_bootstrap=400,
            seed=22,
        )
        # FWER control should keep false-positive count near zero.
        assert out["n_rejected"] <= 2

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            whites_reality_check(np.zeros(50), np.zeros(50))


# ---------------------------------------------------------------------------
# 5) Deflated Sharpe (Bailey-Lopez de Prado, full)
# ---------------------------------------------------------------------------


class TestDeflatedSharpeFull:
    def test_normal_returns_no_data_mining_pvalue_below_05(self):
        # Strong Sharpe = 2.0 over T = 252, only 1 trial -> easily significant.
        out = deflated_sharpe_full(
            sharpe_observed=2.0,
            n_obs=252,
            n_trials=1,
            skew=0.0,
            kurtosis=3.0,
        )
        # With n_trials=1 the null max-Sharpe is zero (no data-mining bias).
        assert out["expected_max_sharpe_under_null"] == 0.0
        assert out["deflated_p_value"] < 0.05
        # And the per-period DSR equals per-period Sharpe.
        assert math.isclose(
            out["deflated_sharpe"],
            2.0 / math.sqrt(252.0),
            abs_tol=1e-9,
        )

    def test_high_n_trials_inflates_required_sharpe(self):
        # Same Sharpe, more trials -> higher expected max -> higher p-value.
        # Use sigma_SR ~ 1 / sqrt(annualisation) for realistic per-period scale.
        sigma_sr = 1.0 / math.sqrt(252.0)
        out_low = deflated_sharpe_full(
            sharpe_observed=2.0,
            n_obs=2520,
            n_trials=10,
            sd_of_trial_sharpes=sigma_sr,
        )
        out_high = deflated_sharpe_full(
            sharpe_observed=2.0,
            n_obs=2520,
            n_trials=10000,
            sd_of_trial_sharpes=sigma_sr,
        )
        assert (
            out_high["expected_max_sharpe_under_null"] > out_low["expected_max_sharpe_under_null"]
        )
        assert out_high["deflated_p_value"] > out_low["deflated_p_value"]

    def test_negative_skew_inflates_se_versus_normal(self):
        # Negative skew should *increase* the SE per BLDP eq. 9: the
        # numerator becomes 1 - gamma3 * SR_per = 1 + |skew|*SR_per > 1.
        # We verify the SE inflation directly (the p-value direction is
        # only monotone in SE when z_star clearly has one sign).
        normal = deflated_sharpe_full(
            sharpe_observed=2.5,
            n_obs=252,
            n_trials=1,
            skew=0.0,
            kurtosis=3.0,
        )
        neg_skew = deflated_sharpe_full(
            sharpe_observed=2.5,
            n_obs=252,
            n_trials=1,
            skew=-0.6,
            kurtosis=3.0,
        )
        assert neg_skew["sigma_se"] > normal["sigma_se"]
        # With n_trials=1, expected_max = 0, so z* is positive and a larger
        # SE strictly increases the p-value.
        assert neg_skew["deflated_p_value"] > normal["deflated_p_value"]

    def test_excess_kurtosis_inflates_se(self):
        # Same logic as above for the kurtosis term.
        normal = deflated_sharpe_full(
            sharpe_observed=2.5,
            n_obs=252,
            n_trials=1,
            skew=0.0,
            kurtosis=3.0,
        )
        fat = deflated_sharpe_full(
            sharpe_observed=2.5,
            n_obs=252,
            n_trials=1,
            skew=0.0,
            kurtosis=8.0,
        )
        assert fat["sigma_se"] > normal["sigma_se"]
        assert fat["deflated_p_value"] > normal["deflated_p_value"]

    def test_backward_compat_with_existing_dsr(self):
        # Compare against the existing implementation in robust_validation.
        # Both should be in the same ballpark (within ~0.2 expected-max
        # Sharpe and ~0.10 deflated p-value); the new version uses the
        # canonical BLDP eq. (5) blend rather than the slightly different
        # composition in robust_validation.deflated_sharpe_ratio.
        from pfm.robust_validation import deflated_sharpe_ratio

        old = deflated_sharpe_ratio(
            sharpe=1.8,
            n_obs=252,
            n_trials=200,
            skew=0.0,
            kurtosis=3.0,
        )
        new = deflated_sharpe_full(
            sharpe_observed=1.8,
            n_obs=252,
            n_trials=200,
            skew=0.0,
            kurtosis=3.0,
        )
        assert (
            abs(old["expected_max_sharpe_under_null"] - new["expected_max_sharpe_under_null"])
            < 0.30
        )
        assert abs(old["deflated_p_value"] - new["deflated_p_value"]) < 0.20

    def test_paper_example_reasonable_values(self):
        # Bailey & Lopez de Prado (2014): with sigma_SR ~ 1/sqrt(252)
        # (per-period scale) and N = 100 trials the expected-max per-period
        # Sharpe is roughly Phi^{-1}(0.99) * sigma_SR ~ 2.3 * 0.063 ~ 0.14.
        sigma_sr_per_period = 1.0 / math.sqrt(252.0)
        out = deflated_sharpe_full(
            sharpe_observed=2.0,
            n_obs=1250,
            n_trials=100,
            sd_of_trial_sharpes=sigma_sr_per_period,
        )
        # Should be a small per-period quantity, well below 0.5.
        assert 0.05 < out["expected_max_sharpe_under_null"] < 0.20

    def test_t_too_small_returns_safe_defaults(self):
        out = deflated_sharpe_full(sharpe_observed=2.0, n_obs=3)
        assert out["deflated_p_value"] == 1.0


# ---------------------------------------------------------------------------
# 6) End-to-end router smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(qr_router)
    return TestClient(app)


class TestRouterEndpoints:
    def test_oos_r_squared_endpoint(self, client):
        rng = np.random.default_rng(7)
        n = 100
        y = list(rng.standard_normal(n) * 0.05)
        ym = list(np.array(y) + rng.standard_normal(n) * 0.02)
        yb = [float(np.mean(y))] * n
        resp = client.post(
            "/quant/oos-r-squared",
            json={
                "y_actual": y,
                "y_pred_model": ym,
                "y_pred_baseline": yb,
                "nested": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_obs"] == n
        assert "r_squared_oos" in body
        assert "hac_t_stat_clark_west" in body

    def test_diebold_mariano_endpoint(self, client):
        rng = np.random.default_rng(8)
        n = 300
        e1 = list(rng.standard_normal(n) * 1.0)
        e2 = list(rng.standard_normal(n) * 0.5)
        resp = client.post(
            "/quant/diebold-mariano",
            json={"forecast_errors_1": e1, "forecast_errors_2": e2, "h": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["prefer_model"] == 2
        assert body["p_value"] < 0.05

    def test_whites_endpoint(self, client):
        rng = np.random.default_rng(9)
        n, k = 200, 20
        rets = rng.standard_normal((n, k)) * 0.01
        rets[:, 0] += 0.008
        bench = list(np.zeros(n))
        resp = client.post(
            "/quant/whites-reality-check",
            json={
                "strategy_returns_matrix": rets.tolist(),
                "benchmark_returns": bench,
                "n_bootstrap": 200,
                "run_stepwise_spa": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["best_strategy_idx"] == 0
        assert body["white_pvalue"] < 0.10
        assert body["stepwise_n_rejected"] is not None
        assert 0 in (body["stepwise_rejected_indices"] or [])

    def test_oos_r_squared_400_on_mismatch(self, client):
        resp = client.post(
            "/quant/oos-r-squared",
            json={
                "y_actual": [1.0, 2.0, 3.0],
                "y_pred_model": [1.0, 2.0],
                "y_pred_baseline": [1.0, 2.0, 3.0],
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 7) Integration: VECM uses MHM continuous p-value
# ---------------------------------------------------------------------------


class TestVECMMHMIntegration:
    def test_fit_vecm_returns_mhm_pvalues(self):
        # Build a clearly-cointegrated synthetic pair: stock log-price and
        # logit(prob) share a common stochastic trend.
        rng = np.random.default_rng(33)
        n = 300
        common = np.cumsum(rng.standard_normal(n) * 0.01)
        stock_price = np.exp(50 + common + rng.standard_normal(n) * 0.005)
        # Probability bound to common via inverse-logit.
        logit_p = -2.0 + 0.5 * common + rng.standard_normal(n) * 0.05
        prob = 1.0 / (1.0 + np.exp(-logit_p))
        prob = np.clip(prob, 0.01, 0.99)
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        from pfm.advanced_event_models import fit_vecm_core

        out = fit_vecm_core(
            pd.Series(stock_price, index=idx, name="px"),
            pd.Series(prob, index=idx, name="prob"),
            det_order=0,
            k_ar_diff=1,
            ticker="SYN",
            factor_id="syn",
        )
        # Continuous p-values: NOT bucketed to {0.005, 0.025, 0.075, 0.20}.
        bucketed = {0.005, 0.025, 0.075, 0.20}
        assert out["johansen_p_trace"] not in bucketed
        assert out["johansen_p_eigenvalue"] not in bucketed
        assert 0.0 < out["johansen_p_trace"] < 1.0
        assert 0.0 < out["johansen_p_eigenvalue"] < 1.0
        # Stationary common factor should produce significant rejection.
        assert out["johansen_p_trace"] < 0.10 or out["johansen_p_eigenvalue"] < 0.10


# ---------------------------------------------------------------------------
# 8) Sanity: HAC variance helper
# ---------------------------------------------------------------------------


def test_newey_west_helper_matches_simple_variance_at_lag_zero():
    from pfm.oos_metrics import _newey_west_var

    rng = np.random.default_rng(2)
    x = rng.standard_normal(200)
    # At lag = 0 the long-run variance is just sample variance.
    expected = float(np.var(x, ddof=0)) / x.size
    got = _newey_west_var(x, lag=0)
    assert math.isclose(got, expected, rel_tol=1e-9)
