"""Tests for ``pfm.quant.half_life``.

The synthetic DGP is the differenced AR(1) form

    Δy_t = α + β·y_{t-1} + ε_t,    ε_t ~ N(0, σ²)

which has equilibrium ``μ* = -α/β`` (for β < 0) and unit-shock half-life
``h = -log(2) / log(1 + β)``. All recovery tests use a fixed numpy
Generator seed so they are deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pfm.quant.half_life import (
    _half_life_from_beta,
    estimate_half_life,
    half_life_universe,
)

# ---------------------------------------------------------------------------
# Synthetic DGP helper
# ---------------------------------------------------------------------------


def _simulate_ar1_diff(
    beta: float,
    *,
    n: int = 2_000,
    alpha: float = 0.0,
    sigma: float = 1.0,
    y0: float = 0.0,
    seed: int = 7,
) -> pd.Series:
    """Simulate y from the differenced AR(1) DGP. Returns exactly ``n`` rows."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n - 1)
    y = np.empty(n, dtype=float)
    y[0] = y0
    for t in range(n - 1):
        y[t + 1] = y[t] + alpha + beta * y[t] + eps[t]
    return pd.Series(y)


# ---------------------------------------------------------------------------
# Closed-form helper
# ---------------------------------------------------------------------------


def test_half_life_from_beta_known_values():
    # β = -0.5 -> h = -log(2)/log(0.5) = 1.0
    assert _half_life_from_beta(-0.5) == pytest.approx(1.0)
    # β = -0.1 -> h = -log(2)/log(0.9) ~ 6.5788
    assert _half_life_from_beta(-0.1) == pytest.approx(6.578813, rel=1e-4)
    # β = -0.3 -> h = -log(2)/log(0.7) ~ 1.9434
    assert _half_life_from_beta(-0.3) == pytest.approx(1.943361, rel=1e-4)


def test_half_life_from_beta_boundaries():
    assert math.isinf(_half_life_from_beta(0.0))
    assert math.isinf(_half_life_from_beta(0.25))
    assert math.isnan(_half_life_from_beta(-2.0))
    assert math.isnan(_half_life_from_beta(-2.5))
    # β in (-2, -1]: 1+β in (-1, 0], log undefined -> NaN
    assert math.isnan(_half_life_from_beta(-1.0))
    assert math.isnan(_half_life_from_beta(-1.5))
    assert math.isnan(_half_life_from_beta(float("nan")))


# ---------------------------------------------------------------------------
# estimate_half_life: DGP recovery
# ---------------------------------------------------------------------------


def test_estimate_half_life_recovers_beta_minus_half():
    s = _simulate_ar1_diff(beta=-0.5, n=4_000, seed=11)
    res = estimate_half_life(s)
    assert res["ar1_coef"] == pytest.approx(-0.5, abs=0.03)
    assert res["half_life_days"] == pytest.approx(1.0, abs=0.1)
    assert res["p_value"] < 1e-6
    assert res["n_obs"] == 3_999


def test_estimate_half_life_beta_minus_one_tenth():
    # β=-0.1 -> h ~ 6.58
    s = _simulate_ar1_diff(beta=-0.1, n=20_000, seed=23)
    res = estimate_half_life(s)
    assert res["ar1_coef"] == pytest.approx(-0.1, abs=0.02)
    assert res["half_life_days"] == pytest.approx(6.58, abs=0.8)
    assert res["p_value"] < 1e-6


def test_estimate_half_life_random_walk_beta_zero():
    # β = 0 exactly: random walk, no mean reversion, half-life = inf.
    s = _simulate_ar1_diff(beta=0.0, n=4_000, seed=31)
    res = estimate_half_life(s)
    # β should be statistically indistinguishable from 0 -> typically p > 0.05
    assert abs(res["ar1_coef"]) < 0.02
    # half-life is +inf when estimated β >= 0; otherwise large but finite.
    # We accept either: not-mean-reverting in the meaningful sense.
    hl = res["half_life_days"]
    assert math.isinf(hl) or hl > 50.0


def test_estimate_half_life_explosive_beta_positive():
    # β > 0 -> explosive AR(1); half-life is +inf by convention.
    rng = np.random.default_rng(41)
    # Build explosive by hand to avoid blow-ups in the simulator
    n = 300
    y = np.zeros(n + 1)
    for t in range(n):
        y[t + 1] = y[t] + 0.05 * y[t] + rng.normal(0, 0.5)
    res = estimate_half_life(pd.Series(y))
    assert res["ar1_coef"] > 0.0
    assert math.isinf(res["half_life_days"])


def test_estimate_half_life_white_noise():
    # White noise (no autocorrelation at all): differenced regression β
    # estimate should be close to -1 (Δy_t ~ -y_{t-1} + new noise) because
    # y_{t-1} essentially IS noise from the prior period. Half-life will
    # be tiny but VALID (β in (-2, 0)). The point of this test is that
    # the code does NOT crash on noise and returns finite output.
    rng = np.random.default_rng(53)
    s = pd.Series(rng.normal(0, 1, size=2_000))
    res = estimate_half_life(s)
    assert res["n_obs"] == 1_999  # 2000 raw obs -> 1999 differenced rows
    assert math.isfinite(res["ar1_coef"])
    # p-value is a real number in [0, 1] (or NaN); not crashy.
    assert (0.0 <= res["p_value"] <= 1.0) or math.isnan(res["p_value"])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_estimate_half_life_reproducible():
    s = _simulate_ar1_diff(beta=-0.3, n=1_500, seed=101)
    a = estimate_half_life(s)
    b = estimate_half_life(s.copy())
    assert a == b


def test_estimate_half_life_index_invariant():
    # Result must not depend on the pandas index labels.
    s = _simulate_ar1_diff(beta=-0.4, n=1_000, seed=103)
    s2 = s.copy()
    s2.index = pd.date_range("2020-01-01", periods=len(s2), freq="D")
    res_a = estimate_half_life(s)
    res_b = estimate_half_life(s2)
    assert res_a["ar1_coef"] == pytest.approx(res_b["ar1_coef"], abs=1e-10)
    assert res_a["half_life_days"] == pytest.approx(res_b["half_life_days"], abs=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_estimate_half_life_one_observation():
    res = estimate_half_life(pd.Series([1.0]))
    assert math.isnan(res["half_life_days"])
    assert math.isnan(res["ar1_coef"])
    assert math.isnan(res["p_value"])
    assert res["n_obs"] == 0


def test_estimate_half_life_two_observations():
    # Need >= 3 raw obs to get >= 2 differenced rows.
    res = estimate_half_life(pd.Series([1.0, 2.0]))
    assert math.isnan(res["half_life_days"])
    assert res["n_obs"] == 0


def test_estimate_half_life_empty_series():
    res = estimate_half_life(pd.Series([], dtype=float))
    assert math.isnan(res["half_life_days"])
    assert math.isnan(res["ar1_coef"])
    assert res["n_obs"] == 0


def test_estimate_half_life_all_nan():
    res = estimate_half_life(pd.Series([np.nan, np.nan, np.nan, np.nan]))
    assert math.isnan(res["half_life_days"])
    assert res["n_obs"] == 0


def test_estimate_half_life_all_constant():
    res = estimate_half_life(pd.Series([3.14] * 100))
    assert math.isnan(res["half_life_days"])
    assert math.isnan(res["ar1_coef"])
    # n_obs is reported (98 after diff alignment with our policy), but the
    # output is still NaN due to zero variance.
    assert res["n_obs"] >= 0


def test_estimate_half_life_nan_in_middle_is_dropped():
    # NaNs are dropped before regression; result should match the no-NaN case.
    s_clean = _simulate_ar1_diff(beta=-0.4, n=1_000, seed=131)
    s_holes = s_clean.copy()
    # Insert a handful of NaNs
    idxs = [10, 250, 500, 750, 999]
    for i in idxs:
        s_holes.iloc[i] = np.nan
    res_clean = estimate_half_life(s_clean)
    res_holes = estimate_half_life(s_holes)
    # Both should recover β close to -0.4 (within MC noise).
    assert res_clean["ar1_coef"] == pytest.approx(-0.4, abs=0.05)
    assert res_holes["ar1_coef"] == pytest.approx(-0.4, abs=0.05)


def test_estimate_half_life_handles_non_numeric_gracefully():
    # Strings should coerce to NaN and then be dropped; if everything drops,
    # we get the all-NaN result rather than a crash.
    s = pd.Series(["a", "b", "c", "d"])
    res = estimate_half_life(s)
    assert math.isnan(res["half_life_days"])
    assert res["n_obs"] == 0


# ---------------------------------------------------------------------------
# Universe-level
# ---------------------------------------------------------------------------


def test_half_life_universe_basic_shape():
    panel = pd.DataFrame(
        {
            "slow": _simulate_ar1_diff(beta=-0.1, n=2_000, seed=201),
            "medium": _simulate_ar1_diff(beta=-0.3, n=2_000, seed=202),
            "fast": _simulate_ar1_diff(beta=-0.7, n=2_000, seed=203),
        }
    )
    out = half_life_universe(panel)
    assert list(out.columns) == ["slug", "half_life_days", "ar1_coef", "p_value", "n_obs"]
    assert list(out["slug"]) == ["slow", "medium", "fast"]
    # 2_000 raw obs -> 1_999 differenced rows
    assert (out["n_obs"] == 1_999).all()


def test_half_life_universe_ranking_correlates_with_planted_beta():
    # Build 5 series with monotone planted β; recovered half-life should rank
    # consistently (smaller |β| -> longer half-life).
    betas = [-0.05, -0.1, -0.2, -0.4, -0.8]
    panel = pd.DataFrame(
        {
            f"b={b:+.2f}": _simulate_ar1_diff(beta=b, n=5_000, seed=300 + i)
            for i, b in enumerate(betas)
        }
    )
    out = half_life_universe(panel)
    # half_life ordering: largest (slowest, β=-0.05) should top, smallest (β=-0.8) bottom.
    hl = out["half_life_days"].to_numpy()
    # Strictly decreasing as β becomes more negative.
    for i in range(len(hl) - 1):
        assert hl[i] > hl[i + 1], f"half-life ordering broken at i={i}: {hl}"
    # β estimates should themselves rank consistently with the planted values.
    beta_hat = out["ar1_coef"].to_numpy()
    spearman = pd.Series(beta_hat).rank().corr(pd.Series(betas).rank())
    assert spearman == pytest.approx(1.0)


def test_half_life_universe_empty():
    out = half_life_universe(pd.DataFrame())
    assert list(out.columns) == ["slug", "half_life_days", "ar1_coef", "p_value", "n_obs"]
    assert len(out) == 0


def test_half_life_universe_handles_degenerate_columns():
    panel = pd.DataFrame(
        {
            "constant": [5.0] * 200,
            "good": _simulate_ar1_diff(beta=-0.3, n=200, seed=401),
            "all_nan": [np.nan] * 200,
        }
    )
    out = half_life_universe(panel).set_index("slug")
    assert math.isnan(out.loc["constant", "ar1_coef"])
    assert math.isnan(out.loc["all_nan", "ar1_coef"])
    assert out.loc["all_nan", "n_obs"] == 0
    # The good column still produces a sensible estimate.
    assert out.loc["good", "ar1_coef"] == pytest.approx(-0.3, abs=0.15)
    assert math.isfinite(out.loc["good", "half_life_days"])


def test_half_life_universe_random_walk_column_marked_infinite():
    panel = pd.DataFrame(
        {
            "rw": _simulate_ar1_diff(beta=0.0, n=3_000, seed=501),
            "mr": _simulate_ar1_diff(beta=-0.5, n=3_000, seed=502),
        }
    )
    out = half_life_universe(panel).set_index("slug")
    rw_hl = out.loc["rw", "half_life_days"]
    mr_hl = out.loc["mr", "half_life_days"]
    assert math.isinf(rw_hl) or rw_hl > 50.0
    assert mr_hl == pytest.approx(1.0, abs=0.15)
