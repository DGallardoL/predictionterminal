"""Tests for ``pfm.mean_reversion`` — Hurst + variance ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.mean_reversion import hurst_exponent, variance_ratio_test


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


def _ar1(n: int, rho: float, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0]
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


class TestHurst:
    def test_random_walk_near_half(self) -> None:
        rng = np.random.default_rng(0)
        n = 2000
        rw = pd.Series(np.cumsum(rng.normal(0, 0.02, n)), index=_idx(n))
        out = hurst_exponent(rw)
        # Allow R/S small-sample bias toward 0.55-0.65; just check we're not
        # clearly trending or strongly mean-reverting.
        assert 0.40 <= out.H <= 0.70
        assert out.r_squared > 0.5

    def test_mean_reverting_diffs_low_h(self) -> None:
        """Build a series whose first differences are strongly anti-persistent."""
        n = 2000
        diffs = _ar1(n, rho=-0.6, sigma=0.02, seed=1)
        s = pd.Series(np.cumsum(diffs), index=_idx(n))
        out = hurst_exponent(s)
        # Expect H < 0.5; allow some bias.
        assert out.H < 0.55

    def test_trending_diffs_high_h(self) -> None:
        n = 2000
        diffs = _ar1(n, rho=0.85, sigma=0.02, seed=2)
        s = pd.Series(np.cumsum(diffs), index=_idx(n))
        out = hurst_exponent(s)
        # Highly persistent first-differences → H > 0.6 typically.
        assert out.H > 0.55

    def test_insufficient_data(self) -> None:
        s = pd.Series([0.5] * 20, index=_idx(20))
        out = hurst_exponent(s)
        assert out.interpretation == "insufficient-data"


class TestVarianceRatio:
    def test_random_walk_fail_to_reject(self) -> None:
        rng = np.random.default_rng(0)
        n = 1000
        rw = pd.Series(np.cumsum(rng.normal(0, 0.02, n)), index=_idx(n))
        out = variance_ratio_test(rw, q=2)
        # |z| < 1.96 ⇒ random_walk
        assert abs(out.z_stat) < 2.0
        assert out.verdict == "random_walk"

    def test_mean_reverting_diffs_vr_lt_one(self) -> None:
        n = 1500
        diffs = _ar1(n, rho=-0.5, sigma=0.02, seed=11)
        s = pd.Series(np.cumsum(diffs), index=_idx(n))
        out = variance_ratio_test(s, q=2)
        assert out.vr < 1.0
        assert out.verdict in ("mean_reverting", "random_walk")

    def test_momentum_diffs_vr_gt_one(self) -> None:
        n = 1500
        diffs = _ar1(n, rho=0.7, sigma=0.02, seed=12)
        s = pd.Series(np.cumsum(diffs), index=_idx(n))
        out = variance_ratio_test(s, q=2)
        assert out.vr > 1.0
        assert out.verdict in ("momentum", "random_walk")

    def test_invalid_q_raises(self) -> None:
        s = pd.Series([0.5] * 50, index=_idx(50))
        with pytest.raises(ValueError, match="q must be ≥ 2"):
            variance_ratio_test(s, q=1)

    def test_insufficient_data(self) -> None:
        s = pd.Series([0.5] * 5, index=_idx(5))
        out = variance_ratio_test(s, q=2)
        assert out.verdict == "insufficient-data"
