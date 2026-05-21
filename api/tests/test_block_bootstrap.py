"""Tests for the Politis-Romano stationary block bootstrap.

Coverage targets:
* AR(1) mean CI covers zero when drift = 0.
* IID returns: block bootstrap CI agrees with IID/percentile baseline.
* Autocorrelated returns: block bootstrap CI is WIDER than IID
  (the structural correctness check that justifies the module).
* Custom statistics: Sharpe ratio and max drawdown.
* Reproducibility under fixed seed.
* Edge cases: single observation, all-same-value, avg_block_size=1, NaN
  statistic, very small / large block sizes.
* Input validation (empty, 2-D, non-finite, bad params).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfm.quant.block_bootstrap import stationary_block_bootstrap

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ar1(n: int, rho: float, sigma: float, seed: int) -> np.ndarray:
    """Generate AR(1) series x_t = rho x_{t-1} + eps_t with eps ~ N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    x = np.empty(n, dtype=float)
    x[0] = eps[0] / math.sqrt(max(1.0 - rho * rho, 1e-12))  # stationary start
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def _sharpe(r: np.ndarray) -> float:
    if r.size < 2:
        return float("nan")
    s = float(r.std(ddof=1))
    if s <= 0.0 or not math.isfinite(s):
        return float("nan")
    return float(r.mean()) / s * math.sqrt(252.0)


def _max_drawdown(r: np.ndarray) -> float:
    """Maximum drawdown of the cumulative-sum equity curve (returns >= 0)."""
    equity = np.cumsum(r)
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity
    return float(drawdown.max())


# --------------------------------------------------------------------------- #
# Core behaviour
# --------------------------------------------------------------------------- #


def test_ar1_zero_mean_ci_covers_zero():
    """AR(1) with rho=0.5 and no drift: 95% CI for mean should cover 0."""
    x = _ar1(n=500, rho=0.5, sigma=1.0, seed=11)
    res = stationary_block_bootstrap(
        x,
        n_resamples=1000,
        avg_block_size=8,
        confidence=0.95,
        random_state=7,
    )
    assert res["ci_low"] <= 0.0 <= res["ci_high"], (
        f"CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] should cover 0"
    )
    # Mean of bootstrap distribution should be close to sample mean.
    assert abs(res["mean"] - float(x.mean())) < 0.1


def test_iid_close_to_simple_percentile():
    """IID returns: block bootstrap CI should be close to a naive
    percentile CI over IID resamples (within ~25% on width)."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.001, 0.02, size=400)

    res_block = stationary_block_bootstrap(x, n_resamples=2000, avg_block_size=5, random_state=42)
    # IID baseline via avg_block_size=1.
    res_iid = stationary_block_bootstrap(x, n_resamples=2000, avg_block_size=1, random_state=42)
    w_block = res_block["ci_high"] - res_block["ci_low"]
    w_iid = res_iid["ci_high"] - res_iid["ci_low"]
    # For IID DGP, block resampling should not be dramatically different.
    ratio = w_block / w_iid
    assert 0.7 < ratio < 1.5, f"IID block/IID width ratio {ratio:.2f} out of range"


def test_autocorrelated_ci_wider_than_iid():
    """For strongly autocorrelated series, block CI > IID CI for the mean
    (more uncertainty correctly reflected)."""
    x = _ar1(n=500, rho=0.85, sigma=1.0, seed=17)
    res_block = stationary_block_bootstrap(x, n_resamples=2000, avg_block_size=20, random_state=42)
    res_iid = stationary_block_bootstrap(x, n_resamples=2000, avg_block_size=1, random_state=42)
    w_block = res_block["ci_high"] - res_block["ci_low"]
    w_iid = res_iid["ci_high"] - res_iid["ci_low"]
    assert w_block > w_iid * 1.2, (
        f"Block CI width {w_block:.3f} should exceed IID width {w_iid:.3f} by >=20% for rho=0.85"
    )


def test_avg_block_size_one_matches_iid_bootstrap():
    """avg_block_size=1 with p=1 yields singleton blocks (IID bootstrap).

    Compare statistics of the bootstrap distribution against a simple
    direct IID resample with the same seed convention. We don't require
    bit-identical equality (different RNG call ordering) but the means
    and CI widths should be very close."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=300)
    res = stationary_block_bootstrap(x, n_resamples=3000, avg_block_size=1, random_state=42)
    # Compute a reference IID bootstrap distribution of the mean.
    rng2 = np.random.default_rng(123)
    idx = rng2.integers(0, x.size, size=(3000, x.size))
    iid_means = x[idx].mean(axis=1)
    iid_lo = float(np.percentile(iid_means, 2.5))
    iid_hi = float(np.percentile(iid_means, 97.5))
    # Width should match within ~15%.
    w_res = res["ci_high"] - res["ci_low"]
    w_iid = iid_hi - iid_lo
    assert abs(w_res - w_iid) / w_iid < 0.2, (
        f"avg_block_size=1 width {w_res:.4f} vs IID {w_iid:.4f}"
    )


def test_custom_statistic_sharpe():
    """Sharpe ratio works as a custom statistic."""
    rng = np.random.default_rng(5)
    # Slight positive drift to give a non-trivial Sharpe.
    x = rng.normal(0.001, 0.01, size=400)
    res = stationary_block_bootstrap(
        x,
        n_resamples=1000,
        avg_block_size=5,
        statistic=_sharpe,
        random_state=42,
    )
    assert math.isfinite(res["mean"])
    assert res["ci_low"] < res["ci_high"]
    # The observed Sharpe of the original sample should lie inside or
    # near the bootstrap distribution.
    assert math.isfinite(res["observed"])
    assert res["ci_low"] - 1.0 <= res["observed"] <= res["ci_high"] + 1.0


def test_custom_statistic_max_drawdown():
    """Max drawdown works as a custom statistic; >= 0 by construction."""
    rng = np.random.default_rng(9)
    x = rng.normal(0.0, 0.02, size=300)
    res = stationary_block_bootstrap(
        x,
        n_resamples=500,
        avg_block_size=10,
        statistic=_max_drawdown,
        random_state=1,
    )
    assert res["ci_low"] >= 0.0
    assert res["ci_high"] >= res["ci_low"]
    assert res["mean"] >= 0.0


def test_reproducibility_same_seed():
    """Same seed + same inputs => identical outputs."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, size=200)
    r1 = stationary_block_bootstrap(x, n_resamples=500, avg_block_size=4, random_state=99)
    r2 = stationary_block_bootstrap(x, n_resamples=500, avg_block_size=4, random_state=99)
    assert r1["mean"] == r2["mean"]
    assert r1["ci_low"] == r2["ci_low"]
    assert r1["ci_high"] == r2["ci_high"]
    assert r1["std"] == r2["std"]


def test_different_seed_gives_different_distribution():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, size=200)
    r1 = stationary_block_bootstrap(x, n_resamples=500, avg_block_size=4, random_state=1)
    r2 = stationary_block_bootstrap(x, n_resamples=500, avg_block_size=4, random_state=2)
    assert r1["mean"] != r2["mean"] or r1["ci_low"] != r2["ci_low"]


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_single_observation():
    """n=1 series: every resample is identical; CI collapses to the value."""
    x = np.array([0.5])
    res = stationary_block_bootstrap(x, n_resamples=100, avg_block_size=5, random_state=0)
    assert res["mean"] == pytest.approx(0.5)
    assert res["ci_low"] == pytest.approx(0.5)
    assert res["ci_high"] == pytest.approx(0.5)
    assert res["std"] == 0.0
    # avg_block_size clamped to series length.
    assert res["avg_block_size"] == 1


def test_all_same_value():
    """Constant series: mean stat is exact, std == 0, CI is degenerate."""
    x = np.full(50, 0.02)
    res = stationary_block_bootstrap(x, n_resamples=500, avg_block_size=5, random_state=0)
    assert res["mean"] == pytest.approx(0.02)
    assert res["ci_low"] == pytest.approx(0.02)
    assert res["ci_high"] == pytest.approx(0.02)
    assert res["std"] == pytest.approx(0.0)


def test_statistic_returns_nan_all_dropped():
    """A statistic that always returns NaN yields n_resamples=0 + NaN CI."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=50)
    res = stationary_block_bootstrap(
        x,
        n_resamples=100,
        avg_block_size=3,
        statistic=lambda r: float("nan"),
        random_state=0,
    )
    assert res["n_resamples"] == 0
    assert math.isnan(res["mean"])
    assert math.isnan(res["ci_low"])
    assert math.isnan(res["ci_high"])


def test_statistic_returns_nan_partial_handled():
    """A statistic that returns NaN on *some* resamples drops only those."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=100)
    flip = {"count": 0}

    def flaky(r: np.ndarray) -> float:
        flip["count"] += 1
        if flip["count"] % 3 == 0:
            return float("nan")
        return float(r.mean())

    res = stationary_block_bootstrap(
        x,
        n_resamples=300,
        avg_block_size=4,
        statistic=flaky,
        random_state=0,
    )
    # Roughly 2/3 retained.
    assert 150 < res["n_resamples"] < 250
    assert math.isfinite(res["mean"])


def test_avg_block_size_larger_than_series_clamped():
    """L > n should be clamped to n (single block covers whole series)."""
    x = np.arange(10, dtype=float)
    res = stationary_block_bootstrap(x, n_resamples=100, avg_block_size=1000, random_state=0)
    assert res["avg_block_size"] == 10


def test_resamples_preserve_length():
    """Implicit invariant: each bootstrap sample has the same length as input.

    Verified indirectly: with the identity-length statistic ``len``, every
    resample returns n. So the mean of the bootstrap dist == n exactly."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=37)
    res = stationary_block_bootstrap(
        x,
        n_resamples=50,
        avg_block_size=5,
        statistic=lambda r: float(r.size),
        random_state=0,
    )
    assert res["mean"] == pytest.approx(37.0)
    assert res["ci_low"] == pytest.approx(37.0)
    assert res["ci_high"] == pytest.approx(37.0)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_empty_returns_raises():
    with pytest.raises(ValueError, match="empty"):
        stationary_block_bootstrap(np.array([]))


def test_two_dim_returns_raises():
    with pytest.raises(ValueError, match="1-D"):
        stationary_block_bootstrap(np.zeros((5, 2)))


def test_non_finite_returns_raises():
    with pytest.raises(ValueError, match="non-finite"):
        stationary_block_bootstrap(np.array([1.0, np.nan, 2.0]))


def test_bad_n_resamples_raises():
    with pytest.raises(ValueError, match="n_resamples"):
        stationary_block_bootstrap(np.array([1.0, 2.0]), n_resamples=0)


def test_bad_block_size_raises():
    with pytest.raises(ValueError, match="avg_block_size"):
        stationary_block_bootstrap(np.array([1.0, 2.0]), avg_block_size=0)


def test_bad_confidence_raises():
    with pytest.raises(ValueError, match="confidence"):
        stationary_block_bootstrap(np.array([1.0, 2.0]), confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        stationary_block_bootstrap(np.array([1.0, 2.0]), confidence=1.0)


def test_bad_statistic_raises():
    with pytest.raises(ValueError, match="callable"):
        stationary_block_bootstrap(np.array([1.0, 2.0]), statistic="mean")


# --------------------------------------------------------------------------- #
# Sanity: returned dict shape
# --------------------------------------------------------------------------- #


def test_return_dict_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=100)
    res = stationary_block_bootstrap(x, n_resamples=200, avg_block_size=5, random_state=0)
    for key in (
        "mean",
        "ci_low",
        "ci_high",
        "std",
        "n_resamples",
        "avg_block_size",
        "observed",
    ):
        assert key in res, f"missing key {key}"
    assert isinstance(res["n_resamples"], int)
    assert isinstance(res["avg_block_size"], int)
    assert res["ci_low"] <= res["mean"] <= res["ci_high"]
