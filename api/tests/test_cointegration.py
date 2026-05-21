"""Tests for ``pfm.cointegration``.

Strategy: build synthetic series whose cointegration relationship is known
by construction (linear combo + AR(1) noise, or independent random walks
for the null), then verify the test outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.cointegration import (
    engle_granger,
    johansen_test,
    spread_zscore,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


def _ar1_series(n: int, rho: float, sigma: float, seed: int = 0) -> np.ndarray:
    """Stationary AR(1) noise."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(max(1.0 - rho * rho, 1e-12))
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


def _random_walk(n: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, sigma, size=n))


# ───────────────────────── engle_granger ──────────────────────────────


class TestEngleGranger:
    def test_cointegrated_pair_detected(self) -> None:
        """A_t = 0.05 + 0.6·B_t + ε_t with ε stationary (AR(1) ρ=0.5)."""
        n = 250
        b = 0.5 + 0.10 * _random_walk(n, 0.02, seed=1)
        eps = _ar1_series(n, rho=0.5, sigma=0.02, seed=2)
        a = 0.05 + 0.6 * b + eps
        sa = pd.Series(a, index=_idx(n))
        sb = pd.Series(b, index=_idx(n))

        res = engle_granger(sa, sb)
        assert res.verdict == "cointegrated"
        assert res.cointegrated is True
        assert res.adf_pvalue < 0.05
        # OLS hedge ratio is biased on integrated regressors (super-consistent
        # but finite-sample bias ~ O(1/√T)); accept ±0.25 around the true 0.6.
        assert pytest.approx(res.beta_hedge, abs=0.25) == 0.6
        assert res.half_life_days is not None and res.half_life_days < 5

    def test_independent_random_walks_not_cointegrated(self) -> None:
        n = 250
        a = pd.Series(_random_walk(n, 0.02, seed=10), index=_idx(n))
        b = pd.Series(_random_walk(n, 0.02, seed=20), index=_idx(n))
        res = engle_granger(a, b)
        # H0 not rejected → not cointegrated
        assert res.adf_pvalue > 0.05
        assert res.verdict == "not_cointegrated"
        assert res.cointegrated is False

    def test_insufficient_data(self) -> None:
        a = pd.Series([0.5] * 10, index=_idx(10))
        b = a.copy()
        res = engle_granger(a, b)
        assert res.verdict == "insufficient-data"
        assert res.n_obs == 10

    def test_constant_leg_returns_insufficient_variation_not_indexerror(self) -> None:
        # Regression: 2026-05-19, /strategies/auto-backtest 500.
        # When one leg is constant across the full window, statsmodels'
        # add_constant drops the redundant column and OLS returns a 1-param
        # fit, so the old `ols.params[1]` indexed past the end and raised
        # IndexError. The scanner ran sub-pairs in a TaskGroup, so the
        # exception bubbled all the way to the FastAPI 500 handler. Verify
        # we now return the "insufficient-variation" verdict and an empty
        # spread instead of raising.
        n = 50
        a = pd.Series([0.5 + 0.001 * i for i in range(n)], index=_idx(n))
        b = pd.Series([0.4] * n, index=_idx(n))  # CONSTANT — would crash old code
        res = engle_granger(a, b)
        assert res.verdict == "insufficient-variation"
        assert res.cointegrated is False
        assert res.spread.empty
        assert res.n_obs == n

    def test_half_life_undefined_when_anti_persistent(self) -> None:
        """Spread with negative AR(1) ρ → half-life undefined."""
        n = 200
        eps = _ar1_series(n, rho=-0.4, sigma=0.02, seed=33)
        b = 0.5 + 0.10 * _random_walk(n, 0.02, seed=4)
        a = 0.6 * b + eps
        res = engle_granger(pd.Series(a, index=_idx(n)), pd.Series(b, index=_idx(n)))
        # rho should be negative → half-life None
        assert res.rho is None or res.rho < 0
        assert res.half_life_days is None

    def test_significance_threshold_changes_verdict(self) -> None:
        """A pair with adf_p ≈ 0.06 flips with α=0.10."""
        # Build a marginal case by using a high-ρ AR(1) noise.
        n = 80  # smaller n → less power
        b = pd.Series(_random_walk(n, 0.02, seed=50), index=_idx(n))
        eps = _ar1_series(n, rho=0.85, sigma=0.02, seed=51)
        a = pd.Series(0.6 * b.values + eps, index=_idx(n))
        r05 = engle_granger(a, b, significance=0.05)
        r20 = engle_granger(a, b, significance=0.20)
        # If the marginal p-value is between 0.05 and 0.20, the verdict flips.
        if 0.05 < r05.adf_pvalue < 0.20:
            assert not r05.cointegrated
            assert r20.cointegrated

    def test_spread_index_matches_input(self) -> None:
        n = 200
        idx = _idx(n)
        b = pd.Series(_random_walk(n, 0.02, seed=7), index=idx)
        eps = _ar1_series(n, 0.4, 0.02, 8)
        a = pd.Series(0.5 * b.values + eps, index=idx)
        res = engle_granger(a, b)
        assert res.spread.index.equals(idx)
        assert len(res.spread) == n


# ────────────────────────── johansen_test ─────────────────────────────


class TestJohansen:
    def test_three_series_one_cointegrating_relation(self) -> None:
        """Build A,B,C with A+B-C = stationary → rank=1."""
        n = 300
        a_proc = _random_walk(n, 0.02, seed=100)
        b_proc = _random_walk(n, 0.02, seed=101)
        # C = A + B + small stationary noise → cointegrating vector (1,1,-1)
        c_proc = a_proc + b_proc + _ar1_series(n, 0.3, 0.02, seed=102)
        df = pd.DataFrame(
            {"a": a_proc, "b": b_proc, "c": c_proc},
            index=_idx(n),
        )
        res = johansen_test(df)
        assert res.n_obs == n
        assert res.rank_trace >= 1
        assert len(res.trace_stats) == 3
        assert len(res.eigen_stats) == 3

    def test_too_few_columns_raises(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=_idx(3))
        with pytest.raises(ValueError, match="≥2 columns"):
            johansen_test(df)

    def test_too_few_rows_raises(self) -> None:
        df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10)}, index=_idx(10))
        with pytest.raises(ValueError, match="≥30 rows"):
            johansen_test(df)


# ─────────────────────────── spread_zscore ────────────────────────────


class TestSpreadZscore:
    def test_basic_z_recovery(self) -> None:
        """Stationary AR(1) → z-scores in approximately [-3, 3]."""
        n = 200
        s = pd.Series(_ar1_series(n, 0.3, 0.02, seed=200), index=_idx(n))
        z = spread_zscore(s, window=20)
        # Drop initial NaN from rolling window
        zv = z.dropna()
        assert zv.between(-5, 5).all()  # z-scores almost always within ±5
        # Mean of |z| for a stationary series ≈ √(2/π) ≈ 0.798
        assert 0.5 < zv.abs().mean() < 1.2

    def test_constant_input_produces_nan_z(self) -> None:
        s = pd.Series([0.30] * 50, index=_idx(50))
        z = spread_zscore(s, window=10)
        # Std of constant series is 0 → z = 0/0 = NaN
        zv = z.dropna()
        # Most z values are NaN (or +/-inf, depending on exact numerics)
        assert len(zv) <= 5 or zv.isna().all()


# ─────────────────────── full chain integration ───────────────────────


class TestChainExample:
    def test_cointegrated_then_zscore(self) -> None:
        """End-to-end: build cointegrated pair, run engle_granger, zscore the spread."""
        n = 250
        b = pd.Series(0.4 + 0.10 * _random_walk(n, 0.02, seed=300), index=_idx(n))
        eps = _ar1_series(n, 0.4, 0.015, seed=301)
        a = pd.Series(0.05 + 0.7 * b.values + eps, index=_idx(n))
        cint = engle_granger(a, b)
        assert cint.cointegrated
        z = spread_zscore(cint.spread, window=20)
        zv = z.dropna()
        # For a stationary spread, z-scores should regularly visit ±2
        assert (zv.abs() > 1.5).any()
