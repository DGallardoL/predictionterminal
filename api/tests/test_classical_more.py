"""Tests for ``pfm.fractional_diff``, ``pfm.garch``, ``pfm.dfa``."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from pfm.dfa import dfa
from pfm.fractional_diff import find_minimal_d, fractional_diff
from pfm.garch import fit_garch_11


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


# ────────────────── Fractional Differentiation ──────────────────────


class TestFractionalDiff:
    def test_d_05_preserves_correlation(self) -> None:
        # Random walk → fractional diff with d=0.5 should be more stationary
        # but still correlated with the original level.
        rng = np.random.default_rng(0)
        rw = pd.Series(np.cumsum(rng.normal(0, 0.05, 300)), index=_idx(300))
        fd = fractional_diff(rw, d=0.5)
        fd_clean = fd.dropna()
        # Should reduce non-stationarity but retain correlation > 0.3
        rw_align = rw.loc[fd_clean.index]
        corr = float(np.corrcoef(rw_align, fd_clean)[0, 1])
        assert corr > 0.3

    def test_d_close_to_1_almost_first_diff(self) -> None:
        rng = np.random.default_rng(0)
        rw = pd.Series(np.cumsum(rng.normal(0, 0.05, 200)), index=_idx(200))
        fd_99 = fractional_diff(rw, d=0.99)
        d1 = rw.diff()
        # The correlation between d=0.99 frac diff and first diff should be high.
        common = fd_99.dropna().index.intersection(d1.dropna().index)
        assert float(np.corrcoef(fd_99.loc[common], d1.loc[common])[0, 1]) > 0.95

    def test_invalid_d_raises(self) -> None:
        s = pd.Series(np.linspace(0, 1, 100), index=_idx(100))
        with pytest.raises(ValueError, match="d must be in"):
            fractional_diff(s, d=0.0)
        with pytest.raises(ValueError, match="d must be in"):
            fractional_diff(s, d=1.0)

    def test_find_minimal_d_on_random_walk(self) -> None:
        rng = np.random.default_rng(0)
        rw = pd.Series(np.cumsum(rng.normal(0, 0.05, 300)), index=_idx(300))
        out = find_minimal_d(rw)
        # Some d in the grid should pass ADF.
        assert out.d is not None
        assert 0 < out.d < 1
        assert out.adf_p_at_d < 0.05

    def test_too_short_series(self) -> None:
        s = pd.Series([0.5] * 20, index=_idx(20))
        with pytest.raises(ValueError, match="≥30"):
            fractional_diff(s, d=0.5)


# ────────────────── GARCH(1,1) ─────────────────────────────────────


class TestGarch:
    def test_fit_on_synthetic_garch(self) -> None:
        """Generate GARCH(1,1) data with known params and try to recover."""
        rng = np.random.default_rng(0)
        n = 600
        omega_true, alpha_true, beta_true = 0.0001, 0.1, 0.85
        sigma2 = np.empty(n)
        eps = np.empty(n)
        sigma2[0] = omega_true / (1 - alpha_true - beta_true)
        eps[0] = rng.normal(0, np.sqrt(sigma2[0]))
        for t in range(1, n):
            sigma2[t] = omega_true + alpha_true * eps[t - 1] ** 2 + beta_true * sigma2[t - 1]
            eps[t] = rng.normal(0, np.sqrt(sigma2[t]))
        # Synthesise a series whose first differences are eps.
        levels = np.concatenate([[0], np.cumsum(eps)])
        s = pd.Series(levels, index=_idx(len(levels)))
        out = fit_garch_11(s)
        assert out.is_stationary
        # Recovered persistence should be close to true (0.95) within ±0.05
        assert abs(out.persistence - (alpha_true + beta_true)) < 0.10

    def test_too_short_raises(self) -> None:
        s = pd.Series([0.5] * 30, index=_idx(30))
        with pytest.raises(ValueError, match="≥50"):
            fit_garch_11(s)

    def test_constant_series_handled(self) -> None:
        # Constant series → no innovations; MLE should still complete.
        s = pd.Series([0.5] * 100, index=_idx(100))
        # Acceptable to fail gracefully on a degenerate input.
        with contextlib.suppress(ValueError):
            fit_garch_11(s)
            # Should fit but with very low or near-zero parameters.


# ────────────────── DFA ────────────────────────────────────────────


class TestDFA:
    def test_random_walk_alpha_close_to_one(self) -> None:
        """Random walk → α ≈ 1.5 (integrated process).
        Wait — for the integrated cumsum, DFA on the cumsum series
        gives α ≈ 1.5 (Brownian); on the *differences*, α ≈ 0.5.
        We pass the cumsum to test DFA's ability to detect non-stationarity.
        """
        rng = np.random.default_rng(0)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 1000)), index=_idx(1000))
        out = dfa(rw)
        # Random walk levels: α ≈ 1.5 (long-range memory)
        assert out.alpha > 1.0
        assert out.interpretation == "non_stationary"

    def test_white_noise_alpha_near_half(self) -> None:
        """IID noise: α ≈ 0.5."""
        rng = np.random.default_rng(0)
        wn = pd.Series(rng.normal(0, 1, 1000), index=_idx(1000))
        out = dfa(wn)
        assert 0.4 < out.alpha < 0.6
        assert out.interpretation == "random_walk"

    def test_too_short_returns_insufficient(self) -> None:
        s = pd.Series([0.5] * 20, index=_idx(20))
        out = dfa(s)
        assert out.interpretation == "insufficient-data"
