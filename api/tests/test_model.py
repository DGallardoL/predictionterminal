"""Tests for ``fit_ols_hac`` using a known data-generating process.

Strategy: simulate ``y = α + Xβ + ε`` with known α and β, fit, and check that
the recovered coefficients are close to the truth and that t-stats are large
for true non-zero effects. This validates the math before any real network
calls are introduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.attribution import attribute
from pfm.model import fit_ols_hac, hac_lag_andrews


@pytest.fixture
def synthetic_dataset() -> tuple[pd.Series, pd.DataFrame, dict[str, float], float]:
    """Generate a 250-day daily series with two factors and known coefficients."""
    rng = np.random.default_rng(seed=42)
    n = 250
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    f1 = rng.normal(0, 0.3, size=n)
    f2 = rng.normal(0, 0.5, size=n)
    X = pd.DataFrame({"f1": f1, "f2": f2}, index=idx)

    alpha_true = 0.0005
    beta_true = {"f1": 0.012, "f2": -0.020}
    eps = rng.normal(0, 0.012, size=n)
    y = pd.Series(
        alpha_true + beta_true["f1"] * f1 + beta_true["f2"] * f2 + eps,
        index=idx,
        name="r",
    )
    return y, X, beta_true, alpha_true


class TestFitOLSHAC:
    def test_recovers_known_betas(
        self, synthetic_dataset: tuple[pd.Series, pd.DataFrame, dict[str, float], float]
    ) -> None:
        y, X, beta_true, alpha_true = synthetic_dataset
        result = fit_ols_hac(y, X)

        recovered = {est.factor_id: est.beta for est in result.factors}
        # 250 obs / σ_ε=0.012 should pin βs to within ~10% of truth.
        assert abs(recovered["f1"] - beta_true["f1"]) < 0.003
        assert abs(recovered["f2"] - beta_true["f2"]) < 0.003
        assert abs(result.stats.alpha - alpha_true) < 0.002

    def test_significant_tstats_on_true_effects(
        self, synthetic_dataset: tuple[pd.Series, pd.DataFrame, dict[str, float], float]
    ) -> None:
        y, X, _, _ = synthetic_dataset
        result = fit_ols_hac(y, X)
        for est in result.factors:
            # Both factors are real and large enough to clear |t| > 2.
            assert abs(est.t_stat) > 2.0

    def test_andrews_lag_used_when_none(
        self, synthetic_dataset: tuple[pd.Series, pd.DataFrame, dict[str, float], float]
    ) -> None:
        y, X, _, _ = synthetic_dataset
        result = fit_ols_hac(y, X, hac_lag=None)
        assert result.diagnostics.hac_lag == hac_lag_andrews(len(y))

    def test_explicit_lag_overrides(
        self, synthetic_dataset: tuple[pd.Series, pd.DataFrame, dict[str, float], float]
    ) -> None:
        y, X, _, _ = synthetic_dataset
        result = fit_ols_hac(y, X, hac_lag=7)
        assert result.diagnostics.hac_lag == 7

    def test_vif_computed_for_each_factor(
        self, synthetic_dataset: tuple[pd.Series, pd.DataFrame, dict[str, float], float]
    ) -> None:
        y, X, _, _ = synthetic_dataset
        result = fit_ols_hac(y, X)
        assert set(result.diagnostics.vif.keys()) == {"f1", "f2"}
        # Independent regressors ⇒ VIF should be near 1.
        for v in result.diagnostics.vif.values():
            assert 0.9 < v < 1.5

    def test_length_mismatch_raises(self) -> None:
        y = pd.Series([0.1, 0.2, 0.3])
        X = pd.DataFrame({"f1": [0.1, 0.2]})
        with pytest.raises(ValueError, match="length mismatch"):
            fit_ols_hac(y, X)

    def test_too_few_obs_raises(self) -> None:
        y = pd.Series([0.1, 0.2])
        X = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
        with pytest.raises(ValueError, match="too few observations"):
            fit_ols_hac(y, X)

    def test_adj_r_squared_is_computed(self) -> None:
        """Adjusted R² must equal 1 - (1-R²)·(n-1)/(n-k-1) — never literally 0
        when R² is non-trivial. Regression test for an audit script that read
        the wrong field name (``adj_r_squared`` vs ``r_squared_adj``) and
        defaulted to 0; we anchor here so any future regression on
        ``ModelStats.r_squared_adj`` fails loudly.
        """
        rng = np.random.default_rng(seed=7)
        n = 50
        x = rng.normal(0.0, 1.0, n)
        # y = 0.5 x + ε with σ_ε tuned so R² ≈ 0.5.
        y = 0.5 * x + rng.normal(0.0, 0.5, n)
        result = fit_ols_hac(pd.Series(y, name="r"), pd.DataFrame({"f1": x}), regression="ols")
        r2 = result.stats.r_squared
        r2_adj = result.stats.r_squared_adj
        # Closed-form check: k=1 regressor, n=50.
        expected = 1.0 - (1.0 - r2) * (n - 1) / (n - 1 - 1)
        assert abs(r2_adj - expected) < 1e-9, (
            f"adj_R² formula mismatch: got {r2_adj:.6f}, expected {expected:.6f}"
        )
        # Sanity: with this DGP R² should land near 0.5, and adj_R² ≈ 0.49.
        assert 0.40 < r2 < 0.60, f"R² out of expected band: {r2}"
        assert 0.39 < r2_adj < 0.59, f"adj_R² out of expected band: {r2_adj}"
        # Adj_R² strictly less than R² (and definitely not zero).
        assert r2_adj < r2
        assert r2_adj > 0.1


class TestAndrewsBandwidth:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (100, 4),  # 4 * 1^(2/9) = 4
            (200, 4),  # 4 * 2^(2/9) ≈ 4.66 ⇒ 4
            (1000, 6),  # 4 * 10^(2/9) ≈ 6.69 ⇒ 6
        ],
    )
    def test_known_lengths(self, n: int, expected: int) -> None:
        assert hac_lag_andrews(n) == expected

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError):
            hac_lag_andrews(1)


class TestAttribution:
    def test_residual_plus_predicted_equals_observed(
        self, synthetic_dataset: tuple[pd.Series, pd.DataFrame, dict[str, float], float]
    ) -> None:
        y, X, _, _ = synthetic_dataset
        result = fit_ols_hac(y, X)
        target = y.index[100]
        attr = attribute(result, y, X, target)
        assert abs(attr.observed_return - (attr.predicted_return + attr.residual)) < 1e-12

    def test_contributions_sum_to_predicted(
        self, synthetic_dataset: tuple[pd.Series, pd.DataFrame, dict[str, float], float]
    ) -> None:
        y, X, _, _ = synthetic_dataset
        result = fit_ols_hac(y, X)
        target = y.index[50]
        attr = attribute(result, y, X, target)
        total = sum(c.contribution for c in attr.contributions)
        assert abs(total - attr.predicted_return) < 1e-12

    def test_unknown_date_raises(
        self, synthetic_dataset: tuple[pd.Series, pd.DataFrame, dict[str, float], float]
    ) -> None:
        y, X, _, _ = synthetic_dataset
        result = fit_ols_hac(y, X)
        with pytest.raises(KeyError):
            attribute(result, y, X, pd.Timestamp("2099-01-01"))

    def test_includes_alpha_and_each_factor(
        self, synthetic_dataset: tuple[pd.Series, pd.DataFrame, dict[str, float], float]
    ) -> None:
        y, X, _, _ = synthetic_dataset
        result = fit_ols_hac(y, X)
        target = y.index[10]
        attr = attribute(result, y, X, target)
        ids = [c.factor_id for c in attr.contributions]
        assert ids == ["alpha", "f1", "f2"]
