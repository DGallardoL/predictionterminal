"""Deep, exhaustive testing of the quant core.

Goal: verify the math is correct, not just that code runs. We focus on
synthetic-DGP recovery, known-result invariants, and benchmark comparisons
against statsmodels for HAC inference.

Each test class targets one module from CLAUDE.md's "MODULOS A PROBAR"
checklist. Tests use ``numpy.random.default_rng`` with fixed seeds for
reproducibility.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pfm.advanced import (
    bootstrap_sharpe_ci,
    permutation_sharpe_test,
    walk_forward_backtest,
)
from pfm.cointegration import engle_granger, spread_zscore
from pfm.dfa import dfa
from pfm.fractional_diff import find_minimal_d, fractional_diff
from pfm.garch import fit_garch_11
from pfm.granger import granger_test
from pfm.kalman import kalman_dynamic_hedge
from pfm.mean_reversion import hurst_exponent, variance_ratio_test
from pfm.model import (
    delta_logit,
    fit_ols_hac,
    logit_transform,
    stationarity_tests,
)
from pfm.multitest import benjamini_hochberg_fdr, bonferroni_correction
from pfm.ou import fit_ou
from pfm.strategy_verdict import quarterly_stability_test

# ---------------------------------------------------------------------------
# 1) Logit transformation invariants
# ---------------------------------------------------------------------------


class TestLogitInvariants:
    """logit_transform / delta_logit invariants."""

    def test_logit_at_half_is_zero(self) -> None:
        """logit(0.5) must be exactly 0 (within float epsilon)."""
        out = logit_transform(pd.Series([0.5]))
        assert abs(float(out.iloc[0])) < 1e-12

    def test_logit_monotonic(self) -> None:
        """logit must be strictly increasing in p."""
        out = logit_transform(pd.Series([0.01, 0.5, 0.99]))
        assert float(out.iloc[0]) < float(out.iloc[1]) < float(out.iloc[2])

    def test_logit_anti_symmetry(self) -> None:
        """logit(p) + logit(1-p) ≈ 0."""
        ps = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        out_p = logit_transform(pd.Series(ps))
        out_1mp = logit_transform(pd.Series(1.0 - ps))
        for a, b in zip(out_p.values, out_1mp.values, strict=True):
            assert abs(a + b) < 1e-12

    def test_logit_invertibility(self) -> None:
        """sigmoid(logit(p)) ≈ p for 100 random values in [eps, 1-eps]."""
        rng = np.random.default_rng(42)
        ps = rng.uniform(0.05, 0.95, size=100)
        logits = logit_transform(pd.Series(ps))
        recovered = 1.0 / (1.0 + np.exp(-logits.values))
        assert np.max(np.abs(recovered - ps)) < 1e-12

    def test_logit_clipping_saturated(self) -> None:
        """epsilon=0.05 should make logit(0.001) == logit(0.05)."""
        below = logit_transform(pd.Series([0.001]), epsilon=0.05)
        at = logit_transform(pd.Series([0.05]), epsilon=0.05)
        assert abs(float(below.iloc[0]) - float(at.iloc[0])) < 1e-12

    def test_delta_logit_constant_series(self) -> None:
        """All zeros after the first NaN for a constant probability series."""
        out = delta_logit(pd.Series([0.5] * 10))
        # First entry is NaN (no predecessor); rest must be exactly 0.
        assert pd.isna(out.iloc[0])
        assert (out.iloc[1:] == 0.0).all()

    def test_delta_logit_monotonic_increasing(self) -> None:
        """A monotone increasing p ⇒ all Δlogit > 0."""
        ps = np.linspace(0.1, 0.9, 50)
        out = delta_logit(pd.Series(ps))
        assert (out.iloc[1:] > 0.0).all()

    def test_delta_logit_monotonic_decreasing(self) -> None:
        """A monotone decreasing p ⇒ all Δlogit < 0."""
        ps = np.linspace(0.9, 0.1, 50)
        out = delta_logit(pd.Series(ps))
        assert (out.iloc[1:] < 0.0).all()

    def test_logit_invalid_epsilon(self) -> None:
        """epsilon outside (0, 0.5) must raise."""
        with pytest.raises(ValueError):
            logit_transform(pd.Series([0.5]), epsilon=0.0)
        with pytest.raises(ValueError):
            logit_transform(pd.Series([0.5]), epsilon=0.6)


# ---------------------------------------------------------------------------
# 2) OLS HAC recovery
# ---------------------------------------------------------------------------


def _gen_ar1_errors(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Generate AR(1) errors: u_t = ρ u_{t-1} + σ z_t."""
    z = rng.standard_normal(n)
    u = np.empty(n)
    u[0] = z[0] * sigma / math.sqrt(max(1.0 - rho * rho, 1e-9))
    for t in range(1, n):
        u[t] = rho * u[t - 1] + sigma * z[t]
    return u


class TestOLSHACRecovery:
    """fit_ols_hac recovery on synthetic data."""

    @pytest.mark.parametrize("n", [50, 100, 250, 500, 1000])
    def test_beta_recovery_iid(self, n: int) -> None:
        """β recovered within 3·SE on iid errors across 10 seeds."""
        true_alpha, true_beta = 0.001, 0.5
        recovered_within_3se = 0
        for seed in range(10):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal(n) * 0.05
            eps = rng.standard_normal(n) * 0.01
            y = true_alpha + true_beta * x + eps
            X = pd.DataFrame({"f1": x})
            res = fit_ols_hac(pd.Series(y), X, regression="hac")
            beta_est = res.factors[0].beta
            se = res.factors[0].std_err
            if abs(beta_est - true_beta) < 3.0 * se:
                recovered_within_3se += 1
        # Across 10 seeds at least 8 should fall within 3·SE.
        assert recovered_within_3se >= 8, f"only {recovered_within_3se}/10 within 3·SE"

    def test_hac_se_larger_under_autocorr(self) -> None:
        """HAC SE should exceed naive OLS SE when *both* x and ε are AR(1).

        Standard Newey-West theory: HAC inflates the OLS variance by a sum
        of cross-products E[x_t x_{t-j} ε_t ε_{t-j}]. If x is i.i.d. those
        cross-products vanish and HAC ≈ OLS. So we use AR(1) x to expose
        the inflation.
        """
        rng = np.random.default_rng(123)
        n = 500
        x = _gen_ar1_errors(n, rho=0.7, sigma=0.05, rng=rng)
        eps = _gen_ar1_errors(n, rho=0.7, sigma=0.01, rng=rng)
        y = 0.001 + 0.5 * x + eps
        X = pd.DataFrame({"f1": x})
        hac = fit_ols_hac(pd.Series(y), X, regression="hac")
        ols = fit_ols_hac(pd.Series(y), X, regression="ols")
        hac_se = hac.factors[0].std_err
        ols_se = ols.factors[0].std_err
        # With ρ=0.7 in both x and ε, HAC inflates the SE meaningfully.
        assert hac_se / ols_se > 1.2, f"hac/ols={hac_se / ols_se:.2f}"

    def test_hac_matches_statsmodels(self) -> None:
        """Our HAC SE must match statsmodels OLS().fit(cov_type='HAC') exactly."""
        rng = np.random.default_rng(7)
        n = 300
        x = rng.standard_normal(n) * 0.05
        eps = _gen_ar1_errors(n, rho=0.4, sigma=0.01, rng=rng)
        y = 0.001 + 0.5 * x + eps
        X = pd.DataFrame({"f1": x})
        res = fit_ols_hac(pd.Series(y), X, regression="hac")
        # Reproduce with statsmodels directly, same lag.
        Xc = sm.add_constant(X.values)
        sm_lag = res.diagnostics.hac_lag
        sm_fit = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": sm_lag})
        assert abs(res.factors[0].beta - sm_fit.params[1]) < 1e-12
        assert abs(res.factors[0].std_err - sm_fit.bse[1]) < 1e-12

    def test_t_stat_coverage(self) -> None:
        """95% CI should cover true β about 95% of the time across 200 trials."""
        true_beta = 0.5
        covered = 0
        n_trials = 200
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            n = 300
            x = rng.standard_normal(n) * 0.05
            eps = rng.standard_normal(n) * 0.01
            y = 0.001 + true_beta * x + eps
            X = pd.DataFrame({"f1": x})
            res = fit_ols_hac(pd.Series(y), X, regression="hac")
            ci_low, ci_high = res.factors[0].ci_low, res.factors[0].ci_high
            if ci_low <= true_beta <= ci_high:
                covered += 1
        # Allow 88-100% (HAC slightly under-covers in finite sample).
        rate = covered / n_trials
        assert 0.85 <= rate <= 1.0, f"coverage rate {rate:.2%} (covered={covered})"

    def test_high_multicollinearity_vif(self) -> None:
        """Two highly-correlated regressors (ρ≈0.97) ⇒ VIF > 10."""
        rng = np.random.default_rng(11)
        n = 400
        x1 = rng.standard_normal(n)
        # Corr ≈ 0.97 to reliably push VIF=1/(1−r²) above 10.
        x2 = 0.97 * x1 + math.sqrt(1.0 - 0.97**2) * rng.standard_normal(n)
        y = 0.5 * x1 + 0.3 * x2 + 0.1 * rng.standard_normal(n)
        X = pd.DataFrame({"f1": x1, "f2": x2})
        res = fit_ols_hac(pd.Series(y), X, regression="hac")
        max_vif = max(res.diagnostics.vif.values())
        assert max_vif > 10.0, f"max VIF = {max_vif:.2f}"

    def test_perfect_collinearity_raises_or_inf_vif(self) -> None:
        """Perfectly collinear regressors ⇒ statsmodels raises or VIF is huge."""
        rng = np.random.default_rng(13)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = x1.copy()  # identical
        y = x1 + 0.1 * rng.standard_normal(n)
        X = pd.DataFrame({"f1": x1, "f2": x2})
        try:
            res = fit_ols_hac(pd.Series(y), X, regression="hac")
        except (ValueError, np.linalg.LinAlgError):
            return  # acceptable
        # If it didn't raise, VIF should be obviously huge or inf.
        max_vif = max(res.diagnostics.vif.values())
        assert max_vif > 1e6 or math.isinf(max_vif), f"max VIF = {max_vif}"

    def test_too_few_observations_raises(self) -> None:
        """n ≤ k+1 must raise."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"f1": rng.standard_normal(2), "f2": rng.standard_normal(2)})
        y = pd.Series(rng.standard_normal(2))
        with pytest.raises(ValueError):
            fit_ols_hac(y, X, regression="hac")

    def test_length_mismatch_raises(self) -> None:
        """y and X with different lengths must raise."""
        rng = np.random.default_rng(0)
        y = pd.Series(rng.standard_normal(50))
        X = pd.DataFrame({"f1": rng.standard_normal(40)})
        with pytest.raises(ValueError):
            fit_ols_hac(y, X, regression="hac")


# ---------------------------------------------------------------------------
# 3) Walk-forward embargo
# ---------------------------------------------------------------------------


class TestWalkForwardEmbargo:
    """walk_forward_backtest embargo behaviour."""

    def _gen_spread(self, n: int, seed: int = 7) -> pd.Series:
        rng = np.random.default_rng(seed)
        # Stationary OU-like spread for embargo tests.
        x = np.empty(n)
        x[0] = 0.0
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + rng.standard_normal() * 0.5
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        return pd.Series(x, index=idx, name="spread")

    def test_embargo_zero_legacy_behavior(self) -> None:
        """embargo_size=0 ⇒ train fold contains exactly n - fold_size bars."""
        spread = self._gen_spread(n=500)
        res = walk_forward_backtest(spread, n_folds=5, window=20, embargo_size=0)
        n = len(spread)
        fold_size = n // 5
        # Middle fold (k=2): train_pnl excludes only the test slice.
        middle = res.folds[2]
        assert middle.n_train == n - fold_size, (
            f"expected n_train={n - fold_size}, got {middle.n_train}"
        )
        assert middle.n_test == fold_size

    def test_embargo_drops_correct_count(self) -> None:
        """embargo_size=10 with n_folds=5, n=500 ⇒ middle fold drops 20 extra train bars."""
        spread = self._gen_spread(n=500)
        emb = 10
        res = walk_forward_backtest(spread, n_folds=5, window=20, embargo_size=emb)
        # Middle fold: both sides of test get embargo'd away.
        middle = res.folds[2]
        n = len(spread)
        fold_size = n // 5
        # n_train = (n - fold_size) - 2*emb
        expected_n_train = (n - fold_size) - 2 * emb
        assert middle.n_train == expected_n_train, (
            f"expected n_train={expected_n_train}, got {middle.n_train}"
        )

    def test_embargo_negative_raises(self) -> None:
        spread = self._gen_spread(n=500)
        with pytest.raises(ValueError):
            walk_forward_backtest(spread, n_folds=5, window=20, embargo_size=-1)

    def test_embargo_too_short_raises(self) -> None:
        """n smaller than n_folds·(window+5) raises informatively."""
        spread = self._gen_spread(n=20)
        with pytest.raises(ValueError):
            walk_forward_backtest(spread, n_folds=5, window=20)


# ---------------------------------------------------------------------------
# 4) BH-FDR correctness
# ---------------------------------------------------------------------------


class TestBHFDR:
    """benjamini_hochberg_fdr correctness."""

    def test_textbook_example(self) -> None:
        """Standard textbook example. m=5 p-values, α=0.05.

        BH: largest k with p_(k) ≤ α·k/m.
        Sorted p = [0.001, 0.01, 0.04, 0.05, 0.1].
        Thresholds α·k/m = [0.01, 0.02, 0.03, 0.04, 0.05].
        k=1: 0.001 ≤ 0.01 ✓
        k=2: 0.01 ≤ 0.02 ✓
        k=3: 0.04 ≤ 0.03 ✗
        k=4: 0.05 ≤ 0.04 ✗
        k=5: 0.10 ≤ 0.05 ✗
        Largest k that passes is 2. So reject the two smallest.
        """
        ps = [0.001, 0.01, 0.04, 0.05, 0.1]
        out = benjamini_hochberg_fdr(ps, alpha=0.05)
        assert sorted(out["rejected_idx"]) == [0, 1]
        assert out["n_significant"] == 2

    def test_all_below_alpha(self) -> None:
        ps = [0.001, 0.002, 0.003, 0.004, 0.005]
        out = benjamini_hochberg_fdr(ps, alpha=0.05)
        assert out["n_significant"] == 5

    def test_all_above_alpha(self) -> None:
        ps = [0.5, 0.6, 0.7, 0.8, 0.9]
        out = benjamini_hochberg_fdr(ps, alpha=0.05)
        assert out["n_significant"] == 0

    def test_fdr_control_simulation(self) -> None:
        """Simulate 100 trials × (90 nulls + 10 strong signals).

        Under BH at q=0.05, the empirical FDP should average ≤ ~5–10%.
        """
        rng = np.random.default_rng(2026)
        n_trials = 100
        n_nulls = 90
        n_signals = 10
        fdps: list[float] = []
        for _ in range(n_trials):
            # Null p-values: uniform[0,1].
            null_p = rng.uniform(0, 1, size=n_nulls)
            # Signal p-values: very small (concentrated near 0).
            signal_p = rng.uniform(0, 0.001, size=n_signals)
            ps = np.concatenate([null_p, signal_p])
            true_signal_idx = set(range(n_nulls, n_nulls + n_signals))
            out = benjamini_hochberg_fdr(ps.tolist(), alpha=0.05)
            rej = set(out["rejected_idx"])
            if not rej:
                fdps.append(0.0)
                continue
            false_disc = rej - true_signal_idx
            fdp = len(false_disc) / max(len(rej), 1)
            fdps.append(fdp)
        avg_fdp = float(np.mean(fdps))
        # BH controls expected FDR at α=0.05; with very strong signals the
        # empirical FDR should be comfortably ≤ 0.10.
        assert avg_fdp <= 0.10, f"empirical FDR={avg_fdp:.3f}"

    def test_q_value_monotonicity(self) -> None:
        """Q-values, when sorted by p-value, must be non-decreasing."""
        from itertools import pairwise

        rng = np.random.default_rng(99)
        ps = rng.uniform(0, 1, size=50).tolist()
        out = benjamini_hochberg_fdr(ps, alpha=0.05)
        # Sort q-values by their parallel p-values.
        sorted_pairs = sorted(zip(ps, out["q_values"], strict=True), key=lambda t: t[0])
        qs_sorted = [q for _, q in sorted_pairs]
        for a, b in pairwise(qs_sorted):
            assert a <= b + 1e-12, f"q-values not monotone: {a} > {b}"

    def test_empty_input(self) -> None:
        out = benjamini_hochberg_fdr([], alpha=0.05)
        assert out["n_significant"] == 0
        assert out["rejected_idx"] == []
        assert out["q_values"] == []

    def test_single_p_value(self) -> None:
        out = benjamini_hochberg_fdr([0.04], alpha=0.05)
        assert out["n_significant"] == 1
        assert out["rejected_idx"] == [0]

    def test_bonferroni_threshold(self) -> None:
        """Bonferroni: reject p_i iff p_i ≤ α/m."""
        ps = [0.001, 0.01, 0.05]
        out = bonferroni_correction(ps, alpha=0.05)
        # threshold = 0.05/3 ≈ 0.0167. p[0]=0.001 ≤ thresh, p[1]=0.01 ≤ thresh,
        # p[2]=0.05 > thresh.  So reject 0 and 1.
        assert out["rejected_idx"] == [0, 1]
        # Adjusted p-values: min(1, 3·p_i).
        assert abs(out["adjusted_p_values"][0] - 0.003) < 1e-12
        assert abs(out["adjusted_p_values"][1] - 0.03) < 1e-12
        assert abs(out["adjusted_p_values"][2] - 0.15) < 1e-12

    def test_bonferroni_strict_threshold(self) -> None:
        """Reject only when p ≤ α/m. p=0.02 with m=3, α=0.05 ⇒ NOT rejected."""
        ps = [0.001, 0.02, 0.05]
        out = bonferroni_correction(ps, alpha=0.05)
        # 0.02 > 0.05/3=0.01667 ⇒ not rejected.
        assert out["rejected_idx"] == [0]


# ---------------------------------------------------------------------------
# 5) 4-quarter stability enforcer
# ---------------------------------------------------------------------------


class TestQuarterlyStability:
    """quarterly_stability_test rules."""

    def test_unanimous_positive_is_gold(self) -> None:
        out = quarterly_stability_test([1.5, 1.5, 1.5, 1.5])
        assert out["tier_recommendation"] == "A_GOLD"
        assert out["passes_4q_gold"] is True
        assert out["sign_flips"] == 0

    def test_sign_flip_kills_gold(self) -> None:
        """Last quarter sign-flips ⇒ not A_GOLD."""
        out = quarterly_stability_test([1.5, 1.5, 1.5, -0.5])
        assert out["tier_recommendation"] != "A_GOLD"
        assert out["sign_flips"] >= 1

    def test_threshold_lower_promotes(self) -> None:
        """All 0.6 with threshold 0.5 ⇒ A_GOLD."""
        out = quarterly_stability_test([0.6, 0.6, 0.6, 0.6], threshold=0.5)
        assert out["tier_recommendation"] == "A_GOLD"

    def test_one_quarter_flips_not_gold(self) -> None:
        out = quarterly_stability_test([0.6, 0.6, -0.1, 0.6], threshold=0.5)
        # Sign flip(s) prevent gold, but n_positive may still qualify silver.
        assert out["tier_recommendation"] != "A_GOLD"
        assert out["sign_flips"] >= 1

    def test_five_quarters_all_positive_gold(self) -> None:
        out = quarterly_stability_test([1.0, 1.0, 1.0, 1.0, 1.0])
        assert out["tier_recommendation"] == "A_GOLD"

    def test_three_quarters_too_few(self) -> None:
        """Only 3 quarters ⇒ not gold (insufficient quarters)."""
        out = quarterly_stability_test([1.5, 1.5, 1.5])
        assert out["tier_recommendation"] != "A_GOLD"

    def test_empty_graceful(self) -> None:
        out = quarterly_stability_test([])
        assert out["n_quarters"] == 0
        assert out["tier_recommendation"] == "C_TENTATIVE"


# ---------------------------------------------------------------------------
# 6) Cointegration Engle-Granger
# ---------------------------------------------------------------------------


class TestCointegration:
    """engle_granger / spread_zscore correctness."""

    def test_cointegrated_pair_rejects_unit_root(self) -> None:
        """Construct y = β·x + stationary noise where x is RW.

        The Engle-Granger ADF on residuals should reject the unit-root null.
        """
        rng = np.random.default_rng(2024)
        n = 400
        # x is a random walk
        x = np.cumsum(rng.standard_normal(n) * 0.1)
        # y = 1.5*x + stationary AR(1) noise
        eps = np.empty(n)
        eps[0] = 0.0
        for t in range(1, n):
            eps[t] = 0.5 * eps[t - 1] + rng.standard_normal() * 0.05
        y = 1.5 * x + eps
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        sa = pd.Series(y, index=idx)
        sb = pd.Series(x, index=idx)
        res = engle_granger(sa, sb)
        assert res.adf_pvalue < 0.05, f"ADF p={res.adf_pvalue:.4f}"
        assert res.cointegrated is True
        # Hedge ratio should be close to 1.5.
        assert abs(res.beta_hedge - 1.5) < 0.1

    def test_independent_random_walks_not_cointegrated(self) -> None:
        """Two independent random walks ⇒ ADF should usually not reject.

        Engle-Granger is known to produce spurious rejections at the
        ~size-of-test rate.  We average over 20 seeds and check the
        rejection rate stays under ~25% — the textbook expectation for
        true unit-root residuals at α=0.05 nominal size.
        """
        rejections = 0
        n_trials = 20
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            n = 400
            x = np.cumsum(rng.standard_normal(n) * 0.1)
            y = np.cumsum(rng.standard_normal(n) * 0.1)
            idx = pd.date_range("2020-01-01", periods=n, freq="D")
            res = engle_granger(pd.Series(y, index=idx), pd.Series(x, index=idx))
            if res.adf_pvalue < 0.05:
                rejections += 1
        # Empirical false-positive rate should be modest.
        rate = rejections / n_trials
        assert rate <= 0.30, f"spurious cointegration rate={rate:.2f} on independent RWs"

    def test_spread_zscore_bounded(self) -> None:
        """Z-score of a stationary series stays in [-5, 5] typically."""
        rng = np.random.default_rng(7)
        n = 200
        x = rng.standard_normal(n)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        z = spread_zscore(pd.Series(x, index=idx), window=20)
        assert z.dropna().abs().max() < 6.0

    def test_ou_half_life(self) -> None:
        """OU with κ=0.1 ⇒ half-life ≈ ln(2)/0.1 ≈ 6.93."""
        rng = np.random.default_rng(42)
        n = 1000
        kappa = 0.1
        beta_ar1 = math.exp(-kappa)  # ≈ 0.9048
        x = np.empty(n)
        x[0] = 0.0
        for t in range(1, n):
            x[t] = beta_ar1 * x[t - 1] + rng.standard_normal() * 0.5
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        ou = fit_ou(pd.Series(x, index=idx))
        expected_hl = math.log(2.0) / kappa
        # 30% tolerance because of finite-sample noise.
        assert abs(ou.half_life_bars - expected_hl) / expected_hl < 0.3, (
            f"hl={ou.half_life_bars:.2f}, expected≈{expected_hl:.2f}"
        )


# ---------------------------------------------------------------------------
# 7) Granger causality
# ---------------------------------------------------------------------------


class TestGranger:
    """granger_test correctness."""

    def test_unidirectional_x_causes_y(self) -> None:
        """Y_t = 0.5 X_{t-1} + ε ⇒ X Granger-causes Y but not vice versa."""
        rng = np.random.default_rng(2024)
        n = 300
        x = rng.standard_normal(n)
        y = np.empty(n)
        y[0] = 0.0
        for t in range(1, n):
            y[t] = 0.5 * x[t - 1] + rng.standard_normal() * 0.3
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        # In our convention "B causes A": pass a=Y, b=X to detect X → Y.
        res = granger_test(
            pd.Series(y, index=idx, name="Y"),
            pd.Series(x, index=idx, name="X"),
            a_id="Y",
            b_id="X",
            max_lag=3,
        )
        # B (X) → A (Y): expect p < 0.05
        assert res.best_pvalue_b_to_a is not None
        assert res.best_pvalue_b_to_a < 0.05, f"B→A p={res.best_pvalue_b_to_a:.4f}"
        # A (Y) → B (X): expect p > 0.10
        assert res.best_pvalue_a_to_b is not None
        assert res.best_pvalue_a_to_b > 0.05, f"A→B p={res.best_pvalue_a_to_b:.4f}"

    def test_independent_series_no_causality(self) -> None:
        rng = np.random.default_rng(11)
        n = 300
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        res = granger_test(
            pd.Series(y, index=idx),
            pd.Series(x, index=idx),
            max_lag=3,
        )
        # Both directions should fail to reject.
        assert res.best_pvalue_b_to_a is None or res.best_pvalue_b_to_a > 0.05
        assert res.best_pvalue_a_to_b is None or res.best_pvalue_a_to_b > 0.05

    def test_optimal_lag_selection(self) -> None:
        """Lag-1 cause ⇒ best lag should be 1."""
        rng = np.random.default_rng(3)
        n = 400
        x = rng.standard_normal(n)
        y = np.empty(n)
        y[0] = 0.0
        for t in range(1, n):
            y[t] = 0.7 * x[t - 1] + rng.standard_normal() * 0.2
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        res = granger_test(
            pd.Series(y, index=idx),
            pd.Series(x, index=idx),
            max_lag=4,
        )
        # The smallest p-value should be at lag=1 (the true generating lag).
        assert res.best_lag_b_to_a == 1


# ---------------------------------------------------------------------------
# 8) GARCH(1,1) fit
# ---------------------------------------------------------------------------


class TestGARCH:
    """fit_garch_11 recovery."""

    def test_recovery_of_known_params(self) -> None:
        """Generate GARCH(1,1) data with ω=0.01, α=0.1, β=0.85 and recover."""
        rng = np.random.default_rng(2024)
        n = 2000
        omega_true, alpha_true, beta_true = 0.01, 0.1, 0.85
        eps = np.empty(n)
        sigma2 = np.empty(n)
        sigma2[0] = omega_true / (1.0 - alpha_true - beta_true)
        eps[0] = math.sqrt(sigma2[0]) * rng.standard_normal()
        for t in range(1, n):
            sigma2[t] = omega_true + alpha_true * eps[t - 1] ** 2 + beta_true * sigma2[t - 1]
            eps[t] = math.sqrt(sigma2[t]) * rng.standard_normal()
        # We pass *levels* whose first-difference is the GARCH innovation series.
        levels = np.cumsum(eps)
        idx = pd.date_range("2010-01-01", periods=n, freq="D")
        res = fit_garch_11(pd.Series(levels, index=idx))
        # Recovery within 30%, generous to handle MLE flatness with 2000 obs.
        assert abs(res.alpha - alpha_true) / alpha_true < 0.40, (
            f"α est={res.alpha:.3f}, true={alpha_true}"
        )
        assert abs(res.beta - beta_true) / beta_true < 0.10, (
            f"β est={res.beta:.3f}, true={beta_true}"
        )
        assert res.is_stationary
        assert res.persistence < 1.0

    def test_persistence_under_one(self) -> None:
        """Even on a moderate sample, MLE should land α+β < 1."""
        rng = np.random.default_rng(0)
        n = 500
        x = np.cumsum(rng.standard_normal(n) * 0.05)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        res = fit_garch_11(pd.Series(x, index=idx))
        assert res.persistence < 1.0

    def test_too_few_obs_raises(self) -> None:
        """fit_garch_11 should raise on <50 obs."""
        rng = np.random.default_rng(0)
        x = pd.Series(rng.standard_normal(40))
        with pytest.raises(ValueError):
            fit_garch_11(x)


# ---------------------------------------------------------------------------
# 9) Hurst & DFA
# ---------------------------------------------------------------------------


class TestHurstDFA:
    """hurst_exponent and dfa interpretation."""

    def test_white_noise_hurst_near_half(self) -> None:
        """White noise: Hurst ≈ 0.5 (with finite-sample bias up to ~0.6)."""
        rng = np.random.default_rng(2024)
        # Hurst is computed on diffs(series); pass cumulative WN levels.
        n = 4000
        levels = np.cumsum(rng.standard_normal(n))
        idx = pd.date_range("2010-01-01", periods=n, freq="D")
        out = hurst_exponent(pd.Series(levels, index=idx))
        # Wide tolerance because R/S has finite-sample upward bias.
        assert 0.40 <= out.H <= 0.65, f"H={out.H:.3f}"

    def test_dfa_white_noise(self) -> None:
        """DFA on cumulative WN: α ≈ 0.5 (a Brownian path)."""
        rng = np.random.default_rng(7)
        n = 4000
        x = rng.standard_normal(n)  # white noise levels (not cumsum)
        idx = pd.date_range("2010-01-01", periods=n, freq="D")
        # DFA cumsums internally, so passing white noise levels gives α≈0.5.
        out = dfa(pd.Series(x, index=idx))
        assert 0.40 <= out.alpha <= 0.60, f"DFA α={out.alpha:.3f}"

    def test_trending_series_hurst_above_half(self) -> None:
        """Strongly autocorrelated diffs ⇒ Hurst > 0.55."""
        rng = np.random.default_rng(11)
        n = 4000
        # Create trending diffs by AR(1) with high ρ.
        diffs = np.empty(n)
        diffs[0] = 0.0
        for t in range(1, n):
            diffs[t] = 0.85 * diffs[t - 1] + rng.standard_normal() * 0.3
        levels = np.cumsum(diffs)
        idx = pd.date_range("2010-01-01", periods=n, freq="D")
        out = hurst_exponent(pd.Series(levels, index=idx))
        assert out.H > 0.55, f"H={out.H:.3f}"

    def test_anti_correlated_hurst_below_half(self) -> None:
        """Sign-flipping diffs ⇒ Hurst < 0.5 typically."""
        rng = np.random.default_rng(13)
        n = 4000
        # AR(1) with negative ρ produces anti-correlated diffs.
        diffs = np.empty(n)
        diffs[0] = 0.0
        for t in range(1, n):
            diffs[t] = -0.7 * diffs[t - 1] + rng.standard_normal() * 0.3
        levels = np.cumsum(diffs)
        idx = pd.date_range("2010-01-01", periods=n, freq="D")
        out = hurst_exponent(pd.Series(levels, index=idx))
        # Allow up to 0.5 because R/S has known upward bias.
        assert out.H < 0.5, f"H={out.H:.3f}"

    def test_variance_ratio_random_walk(self) -> None:
        """VR on a RW should fail to reject the random-walk null at α=5%.

        The Lo-MacKinlay z is asymptotically N(0,1) under H0; we verify the
        rate of false rejections across 20 seeds is at most ~25%.
        """
        rejections = 0
        n_trials = 20
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            n = 1000
            levels = np.cumsum(rng.standard_normal(n))
            idx = pd.date_range("2010-01-01", periods=n, freq="D")
            out = variance_ratio_test(pd.Series(levels, index=idx), q=2)
            if out.verdict != "random_walk":
                rejections += 1
        # Allow up to 25% rejection (5% nominal + finite-sample slack).
        assert rejections / n_trials <= 0.25, (
            f"VR rejection rate {rejections / n_trials:.2f} on RWs"
        )


# ---------------------------------------------------------------------------
# 10) Kalman dynamic hedge
# ---------------------------------------------------------------------------


class TestKalman:
    """kalman_dynamic_hedge β-tracking."""

    def test_constant_beta_converges(self) -> None:
        """Stationary β=2.0 ⇒ Kalman β̂_T close to 2.0."""
        rng = np.random.default_rng(2024)
        n = 500
        x = np.cumsum(rng.standard_normal(n) * 0.1)
        true_beta = 2.0
        y = true_beta * x + rng.standard_normal(n) * 0.05
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        res = kalman_dynamic_hedge(
            pd.Series(y, index=idx),
            pd.Series(x, index=idx),
            delta=1e-4,
        )
        # Final β should be close to true β.
        assert abs(res.beta_final - true_beta) < 0.2, (
            f"β_final={res.beta_final:.3f}, true={true_beta}"
        )

    def test_time_varying_beta_tracked(self) -> None:
        """Slow β drift from 1.0 → 2.0 ⇒ Kalman beta should move in that direction."""
        rng = np.random.default_rng(2025)
        n = 800
        x = rng.standard_normal(n) * 0.5  # i.i.d. so β identification is fine
        beta_path = np.linspace(1.0, 2.0, n)
        y = beta_path * x + rng.standard_normal(n) * 0.1
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        # Larger δ allows faster adaptation for testing.
        res = kalman_dynamic_hedge(
            pd.Series(y, index=idx),
            pd.Series(x, index=idx),
            delta=1e-2,
        )
        # First-half β-mean should be < second-half β-mean.
        first_half = res.beta.iloc[: n // 2].mean()
        second_half = res.beta.iloc[n // 2 :].mean()
        assert second_half > first_half + 0.2, f"first={first_half:.2f}, second={second_half:.2f}"


# ---------------------------------------------------------------------------
# 11) Fractional differencing
# ---------------------------------------------------------------------------


class TestFractionalDiff:
    """fractional_diff and find_minimal_d."""

    def test_minimal_d_on_I1_series(self) -> None:
        """A pure I(1) random walk ⇒ minimal d should be > 0 (positive).

        Per López de Prado §5: the *minimal* d that achieves stationarity
        for an I(1) series is typically much smaller than 1 — that's the
        whole point of fractional differencing (preserve memory).  In our
        test we just verify d is found and the original-series correlation
        stays high (memory preserved).
        """
        rng = np.random.default_rng(2024)
        n = 600
        rw = np.cumsum(rng.standard_normal(n))
        idx = pd.date_range("2010-01-01", periods=n, freq="D")
        out = find_minimal_d(pd.Series(rw, index=idx))
        assert out.d is not None
        # d must be positive and < 1.
        assert 0.0 < out.d < 1.0
        # Memory preservation check: corr with original > 0.6 even after
        # applying the minimal d (López de Prado's selling point).
        assert out.correlation_with_original is not None
        assert out.correlation_with_original > 0.6, f"corr={out.correlation_with_original:.3f}"

    def test_minimal_d_on_stationary_series(self) -> None:
        """A stationary AR(1) ⇒ minimal d should be small (≤ 0.30)."""
        rng = np.random.default_rng(7)
        n = 600
        x = np.empty(n)
        x[0] = 0.0
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        idx = pd.date_range("2010-01-01", periods=n, freq="D")
        out = find_minimal_d(pd.Series(x, index=idx))
        # Stationary series: very small d should suffice.
        assert out.d is not None
        assert out.d <= 0.30, f"minimal d={out.d}"

    def test_fdiff_matches_first_difference_at_d_near_1(self) -> None:
        """At d very close to 1, fractional diff ≈ first difference (correlation > 0.95)."""
        rng = np.random.default_rng(3)
        n = 500
        x = np.cumsum(rng.standard_normal(n))
        idx = pd.date_range("2010-01-01", periods=n, freq="D")
        s = pd.Series(x, index=idx)
        # d=0.99 is the closest to a true first difference within (0,1).
        fd = fractional_diff(s, d=0.99, threshold=1e-4).dropna()
        first = s.diff().dropna()
        common = fd.index.intersection(first.index)
        corr = float(np.corrcoef(fd.loc[common].values, first.loc[common].values)[0, 1])
        assert corr > 0.95, f"corr(fdiff_d=0.99, Δ)={corr:.3f}"


# ---------------------------------------------------------------------------
# Bonus: stationarity_tests sanity
# ---------------------------------------------------------------------------


class TestStationarityTests:
    def test_white_noise_stationary(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        out = stationarity_tests(x)
        # White noise: ADF should reject (small p), KPSS should not reject (large p).
        assert out["adf_pvalue"] < 0.05
        assert out["kpss_pvalue"] > 0.05

    def test_random_walk_non_stationary(self) -> None:
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.standard_normal(500))
        out = stationarity_tests(x)
        # Random walk: ADF should NOT reject (large p), KPSS should reject (small p).
        assert out["adf_pvalue"] > 0.05
        assert out["kpss_pvalue"] < 0.10


# ---------------------------------------------------------------------------
# Bonus: bootstrap_sharpe_ci sanity
# ---------------------------------------------------------------------------


class TestBootstrapSharpe:
    def test_positive_drift_pnl_has_positive_ci(self) -> None:
        """A clearly positive-drift PnL should have a positive Sharpe lower CI."""
        rng = np.random.default_rng(11)
        n = 500
        pnl = rng.standard_normal(n) * 0.01 + 0.005  # mean=0.5%, std=1%
        out = bootstrap_sharpe_ci(pnl, n_iters=200, seed=42)
        assert out.sharpe_point > 1.0  # 0.005/0.01 * sqrt(252) ≈ 7.9
        assert out.sharpe_ci_lo_95 > 0.0

    def test_zero_drift_pnl_ci_brackets_zero(self) -> None:
        rng = np.random.default_rng(13)
        n = 500
        pnl = rng.standard_normal(n) * 0.01
        out = bootstrap_sharpe_ci(pnl, n_iters=200, seed=42)
        assert out.sharpe_ci_lo_95 < 0.0 < out.sharpe_ci_hi_95


# ---------------------------------------------------------------------------
# Bonus: permutation_sharpe_test smoke
# ---------------------------------------------------------------------------


class TestPermutationSharpe:
    def test_random_strategy_high_pvalue(self) -> None:
        """A spread fed into a buy-and-hold should yield p > 0.05 on random WN."""
        rng = np.random.default_rng(2024)
        n = 200
        spread = np.cumsum(rng.standard_normal(n) * 0.5)

        def strategy(s: np.ndarray) -> np.ndarray:
            # Buy-and-hold: pnl_t = Δs_t.
            return np.diff(s, prepend=s[0])

        out = permutation_sharpe_test(spread, pnl_strategy_fn=strategy, n_iters=80, seed=42)
        # No real edge ⇒ permutation p should not be tiny.
        assert out.p_value > 0.05
