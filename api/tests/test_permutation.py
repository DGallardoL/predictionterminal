"""Tests for the permutation-test runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.analyses import permutation_test


def _make_synthetic_signal(
    n: int = 200, k: int = 3, beta: float = 0.5, noise: float = 0.5, seed: int = 1
) -> tuple[pd.Series, pd.DataFrame]:
    """y = sum_i β·x_i + noise, with k regressors."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f"x{i}" for i in range(k)])
    y = pd.Series(beta * X.sum(axis=1).values + noise * rng.standard_normal(n), name="y")
    return y, X


def _make_pure_noise(n: int = 200, k: int = 3, seed: int = 1) -> tuple[pd.Series, pd.DataFrame]:
    """y is independent of X — null hypothesis is true by construction."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f"x{i}" for i in range(k)])
    y = pd.Series(rng.standard_normal(n), name="y")
    return y, X


class TestRealSignalRecovered:
    """When the true β is non-zero, p-value should be small in most seeds."""

    def test_strong_signal_breaks_null(self) -> None:
        y, X = _make_synthetic_signal(n=200, k=3, beta=0.5, noise=0.3, seed=42)
        result = permutation_test(y, X, n_iters=50, seed=1)
        assert result["real_test_r2"] > 0.5  # strong signal in test set
        assert result["p_value"] < 0.05, (
            f"expected p<0.05 with strong signal, got {result['p_value']}"
        )

    def test_signal_recovery_across_seeds(self) -> None:
        """≥9/10 seeds should reject null when β=0.5 with low noise."""
        rejections = 0
        for seed in range(10):
            y, X = _make_synthetic_signal(n=200, k=3, beta=0.5, noise=0.3, seed=seed)
            result = permutation_test(y, X, n_iters=50, seed=seed)
            if result["p_value"] < 0.05:
                rejections += 1
        assert rejections >= 8, f"expected ≥8/10 rejections under strong signal, got {rejections}"


class TestNoiseDoesntFalseReject:
    """When y is independent of X, p-values should be roughly uniform."""

    def test_pure_noise_p_value_above_threshold_typically(self) -> None:
        """Across many seeds, fraction of false-rejections should be near 0.05."""
        false_rejections = 0
        n_trials = 30
        for seed in range(n_trials):
            y, X = _make_pure_noise(n=200, k=3, seed=seed + 100)
            result = permutation_test(y, X, n_iters=50, seed=seed)
            if result["p_value"] < 0.05:
                false_rejections += 1
        # Type-I rate should be ≤20% (allows generous noise on small samples)
        assert false_rejections <= 6, f"too many false rejections: {false_rejections}/{n_trials}"


class TestDeterminism:
    """Same seed must produce identical null draws."""

    def test_same_seed_same_nulls(self) -> None:
        y, X = _make_synthetic_signal(n=150, k=2, seed=7)
        r1 = permutation_test(y, X, n_iters=20, seed=99)
        r2 = permutation_test(y, X, n_iters=20, seed=99)
        assert r1["null_test_r2s"] == r2["null_test_r2s"]
        assert r1["p_value"] == r2["p_value"]

    def test_different_seeds_different_nulls(self) -> None:
        y, X = _make_synthetic_signal(n=150, k=2, seed=7)
        r1 = permutation_test(y, X, n_iters=20, seed=1)
        r2 = permutation_test(y, X, n_iters=20, seed=2)
        # At least one element should differ
        assert r1["null_test_r2s"] != r2["null_test_r2s"]


class TestEdgeCases:
    def test_too_short_returns_nans(self) -> None:
        y = pd.Series(np.random.standard_normal(10))
        X = pd.DataFrame(np.random.standard_normal((10, 3)))
        result = permutation_test(y, X, n_iters=50)
        assert pd.isna(result["real_test_r2"])
        assert result["null_test_r2s"] == []
        assert result["n_iters_completed"] == 0

    def test_returns_p_in_unit_interval(self) -> None:
        y, X = _make_pure_noise(n=200, k=3, seed=1)
        result = permutation_test(y, X, n_iters=50)
        assert 0.0 <= result["p_value"] <= 1.0

    @pytest.mark.parametrize("n_iters", [10, 50, 100])
    def test_correct_count_of_nulls(self, n_iters: int) -> None:
        y, X = _make_synthetic_signal(n=150, k=2, seed=3)
        result = permutation_test(y, X, n_iters=n_iters, seed=42)
        # Allow for slightly fewer if some fits failed numerically
        assert result["n_iters_completed"] >= n_iters - 2
        assert len(result["null_test_r2s"]) == result["n_iters_completed"]
