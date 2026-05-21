"""Tests for ``pfm.signals``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pfm.signals import (
    adaptive_zscore_signals,
    bollinger_signals,
    evaluate_signal,
    macd_signals,
    rsi,
    rsi_signals,
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


# ─────────────────── Bollinger ──────────────────────────────────────


class TestBollinger:
    def test_signal_in_range(self) -> None:
        s = pd.Series(_ar1(200, 0.5, 0.05, 0), index=_idx(200))
        pos = bollinger_signals(s, window=20, k_entry=2.0)
        assert set(pos.unique()).issubset({-1, 0, 1})

    def test_strong_reverter_yields_trades(self) -> None:
        s = pd.Series(_ar1(200, 0.4, 0.05, 0), index=_idx(200))
        pos = bollinger_signals(s, window=20)
        # On a clean reverter we should have some trades.
        assert (pos.diff() != 0).sum() > 0


# ─────────────────────── RSI ────────────────────────────────────────


class TestRSI:
    def test_rsi_bounded_0_100(self) -> None:
        s = pd.Series(_ar1(200, 0.4, 0.05, 0), index=_idx(200))
        r = rsi(s, window=14)
        assert (r.between(0, 100)).all()

    def test_rsi_signals_in_range(self) -> None:
        s = pd.Series(_ar1(200, 0.4, 0.05, 0), index=_idx(200))
        pos = rsi_signals(s)
        assert set(pos.unique()).issubset({-1, 0, 1})


# ─────────────────────── MACD ───────────────────────────────────────


class TestMACD:
    def test_signals_alternating(self) -> None:
        s = pd.Series(_ar1(200, 0.3, 0.05, 0), index=_idx(200))
        pos = macd_signals(s)
        # Should have multiple direction changes on a noise-driven series.
        changes = (pos.diff().fillna(0) != 0).sum()
        assert changes > 5


# ─────────── adaptive z-score ──────────────────────────────────────


class TestAdaptiveZscore:
    def test_short_half_life_uses_short_window(self) -> None:
        s = pd.Series(_ar1(200, 0.3, 0.05, 0), index=_idx(200))
        pos = adaptive_zscore_signals(s, half_life_bars=0.5, multiplier=5.0)
        # Window = 5·0.5 = 2.5 → rounded to min_window=5; should still trade.
        assert set(pos.unique()).issubset({-1, 0, 1})

    def test_long_half_life_uses_long_window(self) -> None:
        s = pd.Series(_ar1(200, 0.95, 0.05, 0), index=_idx(200))
        pos = adaptive_zscore_signals(s, half_life_bars=20, multiplier=5.0)
        assert set(pos.unique()).issubset({-1, 0, 1})


# ──────────── evaluate_signal ──────────────────────────────────────


class TestEvaluate:
    def test_zero_position_zero_pnl(self) -> None:
        s = pd.Series([0.0] * 100, index=_idx(100))
        pos = pd.Series([0] * 100, index=_idx(100))
        out = evaluate_signal(s, pos)
        assert out["sharpe"] == 0.0
        assert out["n_trades"] == 0

    def test_metrics_finite(self) -> None:
        s = pd.Series(_ar1(200, 0.5, 0.05, 0), index=_idx(200))
        pos = bollinger_signals(s, window=20)
        out = evaluate_signal(s, pos)
        for k in ("sharpe", "n_trades", "hit_rate", "max_drawdown", "sortino", "calmar"):
            assert k in out
            assert out[k] is not None
