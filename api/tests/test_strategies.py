"""Synthetic-data tests for ``pfm.strategies``.

Strategy: build deterministic probability series with known properties, then
verify each detector recovers the planted truth (or absence of it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.strategies import (
    conditional_regression,
    frechet_bounds,
    implication_test,
)


def _series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    """Date-indexed UTC series."""
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, name="p")


# ───────────────────────── implication_test ───────────────────────────


class TestImplication:
    def test_consistent_when_p_a_below_p_b(self) -> None:
        p_a = _series([0.10, 0.12, 0.15, 0.18, 0.20] * 10)
        p_b = _series([0.30, 0.32, 0.35, 0.38, 0.40] * 10)
        out = implication_test(p_a, p_b, tolerance=0.02)
        assert out.verdict == "consistent"
        assert out.violation_dates == []
        assert out.max_gap < 0.0  # P(A) − P(B) is always negative here

    def test_violated_when_p_a_above_p_b_persistently(self) -> None:
        # 30 days where P(A)=0.7 but P(B)=0.4 — clear logical violation.
        p_a = _series([0.70] * 30)
        p_b = _series([0.40] * 30)
        out = implication_test(p_a, p_b, tolerance=0.02)
        assert out.verdict == "violated"
        assert len(out.violation_dates) == 30
        assert pytest.approx(out.max_gap) == pytest.approx(0.30)

    def test_borderline_with_handful_of_dates(self) -> None:
        # 2 violation dates → triggers borderline (default thresholds: 1 / 5).
        p_a = _series([0.10, 0.10, 0.55, 0.55, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
        p_b = _series([0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30])
        out = implication_test(p_a, p_b, tolerance=0.02)
        assert out.verdict == "borderline"
        assert len(out.violation_dates) == 2

    def test_tolerance_swallows_small_violations(self) -> None:
        # P(A) − P(B) = 0.01 every day; tolerance=0.02 → no violation.
        p_a = _series([0.31] * 20)
        p_b = _series([0.30] * 20)
        out = implication_test(p_a, p_b, tolerance=0.02)
        assert out.verdict == "consistent"
        assert out.violation_dates == []

    def test_logit_gap_computed(self) -> None:
        p_a = _series([0.50] * 5)
        p_b = _series([0.20] * 5)
        out = implication_test(p_a, p_b)
        # logit(0.5) − logit(0.2) = 0 − ln(0.25) ≈ 1.386
        assert pytest.approx(out.logit_gap_series.iloc[0], abs=1e-3) == pytest.approx(
            1.386, abs=1e-3
        )

    def test_empty_alignment_returns_insufficient(self) -> None:
        # Two series with disjoint indexes → no overlap.
        a = pd.Series([0.5], index=pd.DatetimeIndex(["2026-01-01"], tz="UTC"))
        b = pd.Series([0.5], index=pd.DatetimeIndex(["2026-02-01"], tz="UTC"))
        out = implication_test(a, b)
        assert out.verdict == "insufficient-data"
        assert out.n_obs == 0


# ─────────────────────── conditional_regression ───────────────────────


class TestConditionalRegression:
    def test_recovers_known_beta(self) -> None:
        """Plant β=0.6 in P_A = 0.1 + 0.6·P_B + ε. HAC-OLS should recover."""
        rng = np.random.default_rng(42)
        n = 250
        p_b = rng.uniform(0.1, 0.9, size=n)
        eps = rng.normal(0, 0.02, size=n)
        p_a = (0.1 + 0.6 * p_b + eps).clip(0.01, 0.99)

        idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
        sa = pd.Series(p_a, index=idx)
        sb = pd.Series(p_b, index=idx)

        out = conditional_regression(sa, sb)
        assert out.n_obs == n
        assert pytest.approx(out.beta, abs=0.05) == pytest.approx(0.6, abs=0.05)
        assert out.beta_ci_lo < 0.6 < out.beta_ci_hi
        assert out.r_squared > 0.7

    def test_independent_series_beta_near_zero(self) -> None:
        rng = np.random.default_rng(7)
        n = 300
        p_a = rng.uniform(0.2, 0.8, size=n)
        p_b = rng.uniform(0.2, 0.8, size=n)
        idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
        out = conditional_regression(
            pd.Series(p_a, index=idx),
            pd.Series(p_b, index=idx),
        )
        # 95% CI should comfortably include 0
        assert out.beta_ci_lo < 0 < out.beta_ci_hi
        assert abs(out.beta) < 0.15

    def test_conditional_means_match_construction(self) -> None:
        """If P_A is mechanically high when P_B>0.5, the empirical
        cond means must reflect that."""
        # 100 days of (P_B=0.7, P_A=0.8) then 100 days of (P_B=0.3, P_A=0.2).
        idx = pd.date_range("2025-01-01", periods=200, freq="D", tz="UTC")
        sa = pd.Series([0.8] * 100 + [0.2] * 100, index=idx)
        sb = pd.Series([0.7] * 100 + [0.3] * 100, index=idx)
        out = conditional_regression(sa, sb)
        assert pytest.approx(out.cond_mean_when_b_high, abs=1e-6) == 0.8
        assert pytest.approx(out.cond_mean_when_b_low, abs=1e-6) == 0.2
        assert out.n_b_high == 100

    def test_too_few_obs_raises(self) -> None:
        a = pd.Series(
            [0.5] * 5,
            index=pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"),
        )
        b = a.copy()
        with pytest.raises(ValueError, match="need ≥10"):
            conditional_regression(a, b)


# ───────────────────────── frechet_bounds ─────────────────────────────


class TestFrechetBounds:
    def test_bounds_satisfy_inequality(self) -> None:
        """For all dates, lower ≤ independence_joint ≤ upper."""
        rng = np.random.default_rng(1)
        n = 100
        idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
        sa = pd.Series(rng.uniform(0.01, 0.99, size=n), index=idx)
        sb = pd.Series(rng.uniform(0.01, 0.99, size=n), index=idx)
        out = frechet_bounds(sa, sb)
        # Independence joint = p_a * p_b must lie inside [lower, upper] always
        assert (out.independence_joint >= out.lower - 1e-12).all()
        assert (out.independence_joint <= out.upper + 1e-12).all()
        assert (out.width >= -1e-12).all()

    def test_disjoint_marginals_yield_zero_lower(self) -> None:
        sa = _series([0.30] * 20)
        sb = _series([0.40] * 20)
        out = frechet_bounds(sa, sb)
        # 0.3 + 0.4 - 1 = -0.3 → clipped to 0
        assert (out.lower == 0).all()
        # min(0.3, 0.4) = 0.3
        assert (out.upper == 0.30).all()
        assert pytest.approx(out.mean_width) == 0.30

    def test_high_marginals_yield_positive_lower(self) -> None:
        sa = _series([0.80] * 10)
        sb = _series([0.70] * 10)
        out = frechet_bounds(sa, sb)
        # 0.8 + 0.7 - 1 = 0.5 (must occur jointly at least 50% of the time)
        assert (out.lower == 0.50).all()
        assert (out.upper == 0.70).all()  # min(0.8, 0.7)

    def test_empty_alignment(self) -> None:
        a = pd.Series([0.5], index=pd.DatetimeIndex(["2026-01-01"], tz="UTC"))
        b = pd.Series([0.5], index=pd.DatetimeIndex(["2026-02-01"], tz="UTC"))
        out = frechet_bounds(a, b)
        assert out.n_obs == 0
        assert out.lower.empty
