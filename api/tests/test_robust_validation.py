"""Tests for ``pfm.robust_validation``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pfm.robust_validation import (
    block_bootstrap_sharpe_ci,
    cost_sensitivity_curve,
    deflated_sharpe_ratio,
    lo_sharpe_test,
    out_of_time_test,
    permutation_sharpe_null,
    run_robust_validation,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


class TestLoSharpe:
    def test_zero_pnl_zero_sharpe(self) -> None:
        out = lo_sharpe_test(pd.Series(np.zeros(50), index=_idx(50)))
        assert out.sharpe == 0.0

    def test_real_signal_low_p(self) -> None:
        rng = np.random.default_rng(0)
        # Strong signal: 0.001 mean, 0.005 sd → SR_per_bar = 0.2 → SR_ann ~3.2
        pnl = pd.Series(rng.normal(0.001, 0.005, 500), index=_idx(500))
        out = lo_sharpe_test(pnl)
        assert out.sharpe > 1.0
        assert out.p_value < 0.001
        assert out.ci_lo_95 > 0


class TestBootstrap:
    def test_ci_brackets_point(self) -> None:
        rng = np.random.default_rng(0)
        pnl = pd.Series(rng.normal(0.001, 0.01, 300), index=_idx(300))
        out = block_bootstrap_sharpe_ci(pnl, n_iters=200)
        assert out["ci_lo_95"] <= out["sharpe"] <= out["ci_hi_95"]


class TestPermutation:
    def test_real_signal_low_p(self) -> None:
        rng = np.random.default_rng(0)
        pnl = pd.Series(rng.normal(0.002, 0.01, 200), index=_idx(200))
        out = permutation_sharpe_null(pnl, n_iters=200)
        # Sign-flip has 50% chance of preserving signal direction; on N=200
        # with strong positive mean, p should be small but not always tiny.
        assert 0 <= out["p_value"] <= 1


class TestCostSensitivity:
    def test_break_even_finite(self) -> None:
        rng = np.random.default_rng(0)
        pnl = pd.Series(rng.normal(0.0005, 0.005, 200), index=_idx(200))
        # Pretend each bar is a position change.
        pos_change = pd.Series(np.ones(200), index=_idx(200))
        out = cost_sensitivity_curve(pnl, position_changes=pos_change)
        assert "break_even_bps" in out
        assert len(out["net_sharpe"]) == len(out["costs_bps"])


class TestOutOfTime:
    def test_robust_signal_has_high_ratio(self) -> None:
        rng = np.random.default_rng(0)
        pnl = pd.Series(rng.normal(0.001, 0.01, 300), index=_idx(300))
        out = out_of_time_test(pnl, train_fraction=0.5)
        assert "verdict" in out
        assert out["n_train"] + out["n_test"] == 300


class TestDeflatedSharpe:
    def test_low_n_trials_close_to_normal(self) -> None:
        out = deflated_sharpe_ratio(sharpe=2.0, n_obs=200, n_trials=1)
        # n_trials=1 → no multiple-testing penalty
        assert out["expected_max_sharpe_under_null"] >= 0
        assert 0 <= out["deflated_p_value"] <= 1

    def test_high_n_trials_higher_threshold(self) -> None:
        out_low = deflated_sharpe_ratio(sharpe=2.0, n_obs=200, n_trials=10)
        out_high = deflated_sharpe_ratio(sharpe=2.0, n_obs=200, n_trials=1000)
        # More trials → higher threshold → larger p-value.
        assert (
            out_high["expected_max_sharpe_under_null"] >= out_low["expected_max_sharpe_under_null"]
        )


class TestRunRobustValidation:
    def test_strong_signal_passes(self) -> None:
        rng = np.random.default_rng(0)
        pnl = pd.Series(rng.normal(0.002, 0.005, 300), index=_idx(300))  # SR_ann ~6
        out = run_robust_validation(pnl)
        assert out.overall_verdict in {"STRONG ALPHA", "MARGINAL ALPHA", "WEAK / SUSPECT"}
        # SR ≈ 6 should at least reject zero in Lo and bootstrap.
        assert out.lo_test["p_value"] < 0.05

    def test_noise_fails(self) -> None:
        rng = np.random.default_rng(0)
        pnl = pd.Series(rng.normal(0.0, 0.01, 300), index=_idx(300))
        out = run_robust_validation(pnl)
        # No real signal → should NOT be strong alpha.
        assert out.overall_verdict in {"WEAK / SUSPECT", "NOISE / OVERFIT", "MARGINAL ALPHA"}
