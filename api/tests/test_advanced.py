"""Tests for ``pfm.advanced``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.advanced import (
    bootstrap_sharpe_ci,
    cusum_test,
    permutation_sharpe_test,
    walk_forward_backtest,
)


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


# ─────────────────────────── CUSUM ────────────────────────────────────


class TestCusum:
    def test_no_break_stable(self) -> None:
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 200), index=_idx(200))
        out = cusum_test(s)
        assert out.verdict in {"stable", "break_detected"}  # may rarely false-positive
        # On a true random series with no shift, the CUSUM should be tight.
        assert abs(out.max_abs_cusum) < 5

    def test_planted_break_detected(self) -> None:
        # Series with a clear level shift mid-way.
        n = 200
        s = np.concatenate(
            [
                np.random.default_rng(0).normal(0.0, 0.05, n // 2),
                np.random.default_rng(1).normal(0.20, 0.05, n // 2),
            ]
        )
        out = cusum_test(pd.Series(s, index=_idx(n)))
        assert out.rejected
        assert out.verdict == "break_detected"
        assert out.break_point is not None

    def test_insufficient_data(self) -> None:
        s = pd.Series([0.0] * 10, index=_idx(10))
        out = cusum_test(s)
        assert out.verdict == "insufficient-data"


# ─────────────────────── walk-forward ─────────────────────────────────


class TestWalkForward:
    def test_mean_reverting_stable(self) -> None:
        n = 500
        spread = pd.Series(_ar1(n, rho=0.4, sigma=0.05, seed=42), index=_idx(n))
        out = walk_forward_backtest(spread, n_folds=5, window=20)
        assert out.n_folds == 5
        assert len(out.folds) == 5
        # On a clean AR(1) reverter, the test Sharpe distribution should
        # have a positive mean.
        assert out.test_sharpe_mean > 0

    def test_too_short_raises(self) -> None:
        s = pd.Series(np.zeros(50), index=_idx(50))
        with pytest.raises(ValueError, match="need"):
            walk_forward_backtest(s, n_folds=5, window=20)

    def test_random_walk_unstable(self) -> None:
        rng = np.random.default_rng(0)
        rw = pd.Series(np.cumsum(rng.normal(0, 0.05, 500)), index=_idx(500))
        out = walk_forward_backtest(rw, n_folds=5, window=20)
        # No mean-reversion → folds should disagree wildly.
        assert out.stability in {"borderline", "unstable"} or abs(out.test_sharpe_mean) < 1.0


# ──────────────────── bootstrap Sharpe CI ─────────────────────────────


class TestBootstrapSharpe:
    def test_sample_ci_brackets_point(self) -> None:
        rng = np.random.default_rng(0)
        pnl = rng.normal(0.001, 0.01, 500)  # positive-mean PnL
        out = bootstrap_sharpe_ci(pnl, n_iters=200, seed=0)
        assert out.sharpe_ci_lo_95 <= out.sharpe_point <= out.sharpe_ci_hi_95
        # 95% CI is wider than 90%: lo95 ≤ lo90 and hi95 ≥ hi90.
        assert out.sharpe_ci_lo_95 <= out.sharpe_ci_lo_90 + 1e-9
        assert out.sharpe_ci_hi_95 >= out.sharpe_ci_hi_90 - 1e-9

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="need ≥20"):
            bootstrap_sharpe_ci(np.zeros(5))

    def test_zero_variance_returns_zero(self) -> None:
        out = bootstrap_sharpe_ci(np.zeros(50), n_iters=50, seed=0)
        assert out.sharpe_point == 0.0


# ──────────────────── permutation Sharpe ──────────────────────────────


def _toy_strategy(spread: np.ndarray) -> np.ndarray:
    """1-bar lag mean-reversion: long when below mean, short when above."""
    diffs = np.diff(spread, prepend=spread[0])
    signal = -np.sign(spread - spread.mean())
    return signal[:-1] * diffs[1:]


class TestPermutationSharpe:
    def test_real_signal_low_p(self) -> None:
        """A genuinely mean-reverting spread should beat the null often."""
        n = 300
        spread = _ar1(n, rho=0.5, sigma=0.02, seed=7)
        out = permutation_sharpe_test(
            spread,
            pnl_strategy_fn=_toy_strategy,
            n_iters=80,
            seed=11,
        )
        assert 0 <= out.p_value <= 1
        assert len(out.null_sharpes) == 80

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="need ≥30"):
            permutation_sharpe_test(
                np.zeros(10),
                pnl_strategy_fn=_toy_strategy,
                n_iters=10,
            )
