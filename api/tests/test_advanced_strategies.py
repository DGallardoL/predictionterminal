"""Tests for ``pfm.advanced_strategies``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.advanced_strategies import (
    almgren_chriss_schedule,
    hasbrouck_information_share,
    markov_regime_switching,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


# ─────────────────── Hasbrouck Information Share ─────────────────────


class TestHasbrouck:
    def test_a_leads_recovered(self) -> None:
        """A leads B: B is just A shifted by one bar + small noise."""
        rng = np.random.default_rng(0)
        n = 300
        a_walk = np.cumsum(rng.normal(0, 0.02, n))
        # B follows A with one-bar lag.
        b = np.concatenate([[a_walk[0]], a_walk[:-1]]) + rng.normal(0, 0.005, n)
        idx = _idx(n)
        out = hasbrouck_information_share(
            pd.Series(a_walk, index=idx),
            pd.Series(b, index=idx),
            venue_a_id="A",
            venue_b_id="B",
        )
        # A should be the leader (most price discovery).
        assert out.midpoint_a > 0.5
        assert out.leader in {"A", "tied"}

    def test_bounds_in_unit_interval(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        common = np.cumsum(rng.normal(0, 0.02, n))
        idx = _idx(n)
        a = pd.Series(common + rng.normal(0, 0.01, n), index=idx)
        b = pd.Series(common + rng.normal(0, 0.01, n), index=idx)
        out = hasbrouck_information_share(a, b)
        assert 0.0 <= out.is_a_lower <= out.is_a_upper <= 1.0
        # Bounds for A and B sum to 1 (by construction).
        assert pytest.approx(out.is_a_lower + out.is_b_upper, abs=1e-6) == 1.0

    def test_too_short_raises(self) -> None:
        a = pd.Series([0.5] * 30, index=_idx(30))
        b = a.copy()
        with pytest.raises(ValueError, match="need ≥"):
            hasbrouck_information_share(a, b)


# ─────────────────────── Markov regime-switching ──────────────────────


class TestMarkovRegime:
    def test_two_regimes_recovered(self) -> None:
        """Build a series with a clear regime split: low-vol then high-vol."""
        rng = np.random.default_rng(0)
        n_per = 150
        low = rng.normal(0, 0.01, n_per)
        high = rng.normal(0, 0.10, n_per)
        diffs = np.concatenate([low, high])
        levels = np.concatenate([[0.0], np.cumsum(diffs)])
        spread = pd.Series(levels, index=_idx(len(levels)))
        out = markov_regime_switching(spread)
        assert out.sigma_state1 > out.sigma_state0
        # Late bars should be in state 1 (high-vol regime).
        assert out.regime_probs.iloc[-10:].mean() > 0.5

    def test_too_short_raises(self) -> None:
        s = pd.Series(np.zeros(20), index=_idx(20))
        with pytest.raises(ValueError, match="need ≥50"):
            markov_regime_switching(s)


# ─────────────────── Almgren-Chriss optimal execution ────────────────


class TestAlmgrenChriss:
    def test_zero_risk_aversion_twap(self) -> None:
        """λ = 0 ⇒ uniform liquidation (TWAP)."""
        out = almgren_chriss_schedule(
            target_position=1000.0,
            n_intervals=10,
            time_horizon=1.0,
            sigma=0.10,
            eta=0.01,
            risk_aversion=0.0,
        )
        # Each interval should sell ~100 (10% of position).
        for n in out.n_per_interval:
            assert pytest.approx(n, rel=0.05) == 100.0
        # Final remaining = 0
        assert pytest.approx(out.x_remaining[-1], abs=1e-9) == 0.0

    def test_positive_risk_aversion_front_loaded(self) -> None:
        """λ > 0 ⇒ trade more aggressively early (variance-averse)."""
        out = almgren_chriss_schedule(
            target_position=1000.0,
            n_intervals=10,
            time_horizon=1.0,
            sigma=0.20,
            eta=0.01,
            risk_aversion=5.0,
        )
        # First-half trade volume > second-half.
        first_half = sum(out.n_per_interval[:5])
        second_half = sum(out.n_per_interval[5:])
        assert first_half > second_half * 1.1

    def test_kappa_grows_with_risk_aversion(self) -> None:
        out_low = almgren_chriss_schedule(
            target_position=1000.0,
            sigma=0.1,
            eta=0.01,
            risk_aversion=0.5,
        )
        out_high = almgren_chriss_schedule(
            target_position=1000.0,
            sigma=0.1,
            eta=0.01,
            risk_aversion=10.0,
        )
        assert out_high.kappa > out_low.kappa

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="n_intervals"):
            almgren_chriss_schedule(target_position=100.0, n_intervals=1)
        with pytest.raises(ValueError, match="time_horizon"):
            almgren_chriss_schedule(target_position=100.0, time_horizon=0.0)

    def test_utility_equals_cost_plus_risk(self) -> None:
        out = almgren_chriss_schedule(
            target_position=1000.0,
            risk_aversion=2.0,
        )
        assert (
            pytest.approx(
                out.utility,
                rel=1e-6,
            )
            == out.expected_cost + 2.0 * out.variance_cost
        )
