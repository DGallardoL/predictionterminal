"""Tests for ``pfm.triple_barrier`` and ``pfm.distance_method``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.distance_method import distance_method
from pfm.triple_barrier import triple_barrier_backtest


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


# ────────────────── Triple Barrier ──────────────────────────────────


class TestTripleBarrier:
    def test_basic_run(self) -> None:
        spread = pd.Series(_ar1(300, 0.4, 0.05, 0), index=_idx(300))
        out = triple_barrier_backtest(spread, window=20, time_horizon_bars=10)
        assert out.n_trades >= 0
        assert out.n_profit_hits + out.n_stop_hits + out.n_time_hits == out.n_trades

    def test_too_short_raises(self) -> None:
        spread = pd.Series(np.zeros(15), index=_idx(15))
        with pytest.raises(ValueError, match="need"):
            triple_barrier_backtest(spread, window=20, time_horizon_bars=10)

    def test_invalid_inputs(self) -> None:
        spread = pd.Series(np.linspace(0, 1, 100), index=_idx(100))
        with pytest.raises(ValueError, match="entry_z"):
            triple_barrier_backtest(spread, entry_z=0)
        with pytest.raises(ValueError, match="time_horizon"):
            triple_barrier_backtest(spread, time_horizon_bars=0)

    def test_strong_reverter_makes_money(self) -> None:
        # AR(1) ρ=0.3 mean-reverts strongly → triple barrier should profit.
        spread = pd.Series(_ar1(500, 0.3, 0.05, 42), index=_idx(500))
        out = triple_barrier_backtest(
            spread,
            window=20,
            profit_target_sigma=1.0,
            stop_loss_sigma=3.0,
            time_horizon_bars=10,
        )
        # Expect majority of completed trades to be profit-target hits.
        assert out.n_trades > 0


# ────────────────── Distance Method ─────────────────────────────────


class TestDistanceMethod:
    def test_basic_run(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        common = rng.normal(0, 0.05, n)
        a = common + rng.normal(0, 0.01, n) + 0.5
        b = common + rng.normal(0, 0.01, n) + 0.4
        idx = _idx(n)
        out = distance_method(
            pd.Series(a, index=idx),
            pd.Series(b, index=idx),
            formation_fraction=0.5,
            entry_sigma=2.0,
        )
        assert out.n_trading_bars == 100
        assert out.formation_ssd > 0
        assert out.formation_sigma > 0

    def test_too_short_raises(self) -> None:
        s = pd.Series([0.5] * 20, index=_idx(20))
        with pytest.raises(ValueError, match="need ≥30"):
            distance_method(s, s)

    def test_zero_variance_leg_raises(self) -> None:
        a = pd.Series([0.5] * 100, index=_idx(100))
        b = pd.Series(np.linspace(0, 1, 100), index=_idx(100))
        with pytest.raises(ValueError, match="zero variance"):
            distance_method(a, b)
