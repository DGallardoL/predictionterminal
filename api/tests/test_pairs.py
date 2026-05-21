"""Tests for ``pfm.pairs`` — z-score signals + walk-forward backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.pairs import pairs_backtest


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


def _ar1(n: int, rho: float, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(max(1.0 - rho * rho, 1e-12))
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


class TestSignalsAndPnl:
    def test_known_mean_reverter_yields_positive_sharpe(self) -> None:
        """A strongly mean-reverting AR(1) (ρ=0.4) should make the
        z-score trader profitable in expectation."""
        n = 600
        spread = pd.Series(_ar1(n, rho=0.4, sigma=0.05, seed=42), index=_idx(n))
        out = pairs_backtest(spread, window=20, entry_z=1.5, exit_z=0.3, stop_z=5.0)
        assert out.n_trades > 5
        assert out.sharpe > 0  # positive expectation
        # Hit rate should comfortably beat 50% for a real reverter.
        assert out.hit_rate >= 0.55

    def test_random_walk_hovers_near_zero(self) -> None:
        """A pure random walk has no edge; sharpe should be small."""
        n = 600
        rng = np.random.default_rng(123)
        rw = pd.Series(np.cumsum(rng.normal(0, 0.05, size=n)), index=_idx(n))
        out = pairs_backtest(rw, window=20, entry_z=2.0, exit_z=0.5, stop_z=4.0)
        # Not strictly zero but with no reversion the Sharpe should be small.
        assert abs(out.sharpe) < 1.5

    def test_too_few_bars_raises(self) -> None:
        s = pd.Series([0.0] * 15, index=_idx(15))
        with pytest.raises(ValueError, match="need at least"):
            pairs_backtest(s, window=20)

    def test_threshold_validation(self) -> None:
        s = pd.Series(_ar1(80, 0.3, 0.02), index=_idx(80))
        with pytest.raises(ValueError, match="entry_z must be greater than exit_z"):
            pairs_backtest(s, entry_z=0.5, exit_z=0.5)
        with pytest.raises(ValueError, match="stop_z must be greater than entry_z"):
            pairs_backtest(s, entry_z=2.0, exit_z=0.5, stop_z=2.0)

    def test_equity_curve_monotonic_for_constant_zero(self) -> None:
        """If spread never moves, no trades open and equity stays flat."""
        s = pd.Series([0.0] * 200, index=_idx(200))
        out = pairs_backtest(s, window=20)
        assert out.n_trades == 0
        assert out.equity_curve.abs().max() == 0.0
        assert out.max_drawdown == 0.0

    def test_trade_records_have_complete_metadata(self) -> None:
        n = 400
        s = pd.Series(_ar1(n, 0.4, 0.05, seed=7), index=_idx(n))
        out = pairs_backtest(s, window=20, entry_z=1.5, exit_z=0.3, stop_z=5.0)
        assert out.n_trades > 0
        for t in out.trades:
            assert t.direction in (-1, 1)
            assert t.holding_days >= 1
            assert t.exit_reason in {"mean_reversion", "stopped_out", "end_of_data"}
            assert isinstance(t.entry_date, pd.Timestamp)
            assert t.exit_date >= t.entry_date

    def test_positions_zero_until_first_entry(self) -> None:
        n = 200
        s = pd.Series(_ar1(n, 0.5, 0.04, seed=11), index=_idx(n))
        out = pairs_backtest(s, window=20, entry_z=2.0, exit_z=0.5, stop_z=4.0)
        # First few bars (within rolling-window warm-up) must be zero.
        assert (out.positions.iloc[:10] == 0).all()
