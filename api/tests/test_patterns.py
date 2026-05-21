"""Tests for ``pfm.patterns``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pfm.patterns import (
    cluster_pairs_by_signature,
    correlate_pair_pnls,
    day_of_week_effect,
    pre_resolution_regime,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


# ─────────────── correlate_pair_pnls ─────────────────────────────────


class TestCorrelatePairPnls:
    def test_independent_pairs_low_correlation(self) -> None:
        rng = np.random.default_rng(0)
        idx = _idx(200)
        pnls = {
            "p1": pd.Series(rng.normal(0, 0.01, 200), index=idx),
            "p2": pd.Series(rng.normal(0, 0.01, 200), index=idx),
            "p3": pd.Series(rng.normal(0, 0.01, 200), index=idx),
        }
        out = correlate_pair_pnls(pnls)
        assert out.mean_off_diagonal < 0.20  # independent → low |ρ|
        assert out.diversification_ratio > 0.85  # close to 1

    def test_correlated_pairs_detected(self) -> None:
        rng = np.random.default_rng(0)
        idx = _idx(200)
        common = rng.normal(0, 0.01, 200)
        pnls = {
            "p1": pd.Series(common + rng.normal(0, 0.001, 200), index=idx),
            "p2": pd.Series(common + rng.normal(0, 0.001, 200), index=idx),
            "p3": pd.Series(common + rng.normal(0, 0.001, 200), index=idx),
        }
        out = correlate_pair_pnls(pnls)
        assert out.mean_off_diagonal > 0.80
        assert out.diversification_ratio < 0.65

    def test_single_pair(self) -> None:
        out = correlate_pair_pnls({"p1": pd.Series([1, 2, 3])})
        assert out.diversification_ratio == 1.0


# ──────────────── day_of_week_effect ─────────────────────────────────


class TestDayOfWeekEffect:
    def test_planted_monday_effect(self) -> None:
        # 200 days with positive PnL on Mondays (dayofweek=0).
        idx = _idx(200)
        rng = np.random.default_rng(0)
        base = rng.normal(0.0, 0.01, 200)
        # +0.05 boost on Mondays
        boost = np.where(idx.dayofweek == 0, 0.05, 0.0)
        s = pd.Series(base + boost, index=idx)
        out = day_of_week_effect(s)
        assert "Mon" in out.means
        assert out.means["Mon"] > out.means.get("Wed", 0.0)
        # Mon should be flagged significant.
        assert "Mon" in out.significant_days

    def test_no_effect_no_significance(self) -> None:
        idx = _idx(150)
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 0.01, 150), index=idx)
        out = day_of_week_effect(s)
        # On pure noise we expect ≤1 false positive at α=0.05 across 7 weekdays.
        assert len(out.significant_days) <= 2

    def test_too_short(self) -> None:
        s = pd.Series([0.01] * 5, index=_idx(5))
        out = day_of_week_effect(s)
        assert out.means == {}


# ──────────── pre_resolution_regime ──────────────────────────────────


class TestPreResolution:
    def test_planted_vol_explosion(self) -> None:
        # First 100 bars: σ=0.01; last 30 bars: σ=0.05 — vol exploded.
        rng = np.random.default_rng(0)
        far = rng.normal(0, 0.01, 100)
        near = rng.normal(0, 0.05, 30)
        s = pd.Series(np.concatenate([far, near]), index=_idx(130))
        out = pre_resolution_regime(s, days_to_resolution=30)
        assert out.vol_ratio > 2.0
        assert out.vol_shift_significant

    def test_too_short(self) -> None:
        s = pd.Series([0.0] * 30, index=_idx(30))
        out = pre_resolution_regime(s, days_to_resolution=30)
        assert out.far_n == 0


# ──────── cluster_pairs_by_signature ──────────────────────────────────


class TestClustering:
    def test_two_planted_clusters_recovered(self) -> None:
        # Two clusters: high-Sharpe/short-half-life vs low-Sharpe/long-half-life.
        signatures = {
            "p1": {"sharpe": 4.0, "half_life": 0.5, "hit_rate": 0.95},
            "p2": {"sharpe": 4.5, "half_life": 0.8, "hit_rate": 0.90},
            "p3": {"sharpe": 3.8, "half_life": 0.6, "hit_rate": 0.88},
            "p4": {"sharpe": 0.5, "half_life": 30.0, "hit_rate": 0.50},
            "p5": {"sharpe": 0.3, "half_life": 35.0, "hit_rate": 0.55},
            "p6": {"sharpe": 0.4, "half_life": 28.0, "hit_rate": 0.52},
        }
        out = cluster_pairs_by_signature(signatures, n_clusters=2)
        assert out.n_clusters == 2
        # The 6 pairs must be split 3-3 (each cluster).
        sizes = sorted(c.n_members for c in out.clusters)
        assert sizes == [3, 3]

    def test_too_few_pairs(self) -> None:
        out = cluster_pairs_by_signature(
            {"p1": {"sharpe": 1.0}},
            n_clusters=3,
        )
        assert out.n_clusters == 0
