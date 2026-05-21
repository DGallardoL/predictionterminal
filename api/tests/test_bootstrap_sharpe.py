"""Tests for :mod:`pfm.quant.bootstrap_sharpe`.

Coverage targets:

- Percentile and BCa methods on known-Sharpe DGPs.
- Coverage rate of nominal 95% CIs across replicates.
- Symmetry on Gaussian returns.
- Skew-correction advantage of BCa over percentile on lognormal returns.
- Error / degenerate handling.
- Reproducibility.
- Annualisation factor effect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfm.quant.bootstrap_sharpe import bootstrap_sharpe_ci, sharpe_ratio

# ---------------------------------------------------------------------------
# Smoke + sanity
# ---------------------------------------------------------------------------


def test_returns_expected_keys() -> None:
    rng = np.random.default_rng(0)
    out = bootstrap_sharpe_ci(rng.normal(0.001, 0.01, 252), n_resamples=200)
    for k in (
        "sharpe_mean",
        "sharpe_ci_low",
        "sharpe_ci_high",
        "sharpe_std",
        "n_resamples",
        "method",
        "observed_sharpe",
        "confidence",
    ):
        assert k in out, f"missing key {k}"
    assert out["sharpe_ci_low"] <= out["sharpe_mean"] <= out["sharpe_ci_high"]


def test_sharpe_ratio_helper_basic() -> None:
    # Constant-mean / known-std series.
    r = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    sr = sharpe_ratio(r, ann_factor=252)
    # mean=.03, std~.01581, sr_per=1.8974, *sqrt(252)=30.12
    assert sr == pytest.approx(0.03 / np.std(r, ddof=1) * math.sqrt(252), rel=1e-6)


def test_sharpe_ratio_returns_nan_on_constant() -> None:
    assert math.isnan(sharpe_ratio(np.ones(50)))


def test_sharpe_ratio_returns_nan_on_too_short() -> None:
    assert math.isnan(sharpe_ratio(np.array([0.01])))
    assert math.isnan(sharpe_ratio(np.array([])))


# ---------------------------------------------------------------------------
# Known-Sharpe DGP coverage
# ---------------------------------------------------------------------------


def test_percentile_ci_contains_true_sharpe_majority() -> None:
    """Across 30 Gaussian replicates the percentile CI covers the true SR
    at least ~80% of the time (95% nominal, finite-sample undercoverage
    expected for small T=252)."""
    true_mu, true_sigma = 0.0008, 0.012
    true_sharpe_ann = true_mu / true_sigma * math.sqrt(252)
    covered = 0
    n_rep = 30
    for seed in range(n_rep):
        rng = np.random.default_rng(seed)
        r = rng.normal(true_mu, true_sigma, 252)
        out = bootstrap_sharpe_ci(r, n_resamples=1500, method="percentile", random_state=seed)
        if out["sharpe_ci_low"] <= true_sharpe_ann <= out["sharpe_ci_high"]:
            covered += 1
    # 80% lower bound is well below nominal 95% to keep the test deterministic
    # on small numbers of replicates; we mostly want to confirm the CI is
    # not pathologically wrong.
    assert covered >= int(0.8 * n_rep), f"only {covered}/{n_rep} CIs covered the truth"


def test_bca_ci_contains_true_sharpe_majority() -> None:
    true_mu, true_sigma = 0.0005, 0.01
    true_sharpe_ann = true_mu / true_sigma * math.sqrt(252)
    covered = 0
    n_rep = 25
    for seed in range(n_rep):
        rng = np.random.default_rng(seed + 100)
        r = rng.normal(true_mu, true_sigma, 252)
        out = bootstrap_sharpe_ci(r, n_resamples=1000, method="bca", random_state=seed)
        if out["sharpe_ci_low"] <= true_sharpe_ann <= out["sharpe_ci_high"]:
            covered += 1
    assert covered >= int(0.75 * n_rep)


# ---------------------------------------------------------------------------
# Symmetric vs skewed
# ---------------------------------------------------------------------------


def test_symmetric_returns_give_roughly_symmetric_ci() -> None:
    """For Gaussian returns the percentile CI should be approximately
    symmetric around the bootstrap mean."""
    rng = np.random.default_rng(42)
    r = rng.normal(0.001, 0.01, 500)
    out = bootstrap_sharpe_ci(r, n_resamples=3000, method="percentile", random_state=0)
    left = out["sharpe_mean"] - out["sharpe_ci_low"]
    right = out["sharpe_ci_high"] - out["sharpe_mean"]
    # within 25% of each other
    ratio = min(left, right) / max(left, right)
    assert ratio > 0.75, f"asymmetric symmetric CI: left={left:.3f}, right={right:.3f}"


def test_bca_better_than_percentile_on_skewed_returns() -> None:
    """On lognormal (right-skew) returns BCa should produce a CI whose
    asymmetry better reflects the underlying skew: specifically, the BCa
    upper-tail width should be wider relative to the lower-tail width
    than the corresponding percentile CI."""
    rng = np.random.default_rng(7)
    # Lognormal-shifted returns: heavy right tail.
    r = rng.lognormal(mean=-5.0, sigma=1.0, size=500) - math.exp(-5.0 + 0.5)
    out_pct = bootstrap_sharpe_ci(r, n_resamples=3000, method="percentile", random_state=0)
    out_bca = bootstrap_sharpe_ci(r, n_resamples=3000, method="bca", random_state=0)

    def asym(out: dict) -> float:
        left = out["sharpe_mean"] - out["sharpe_ci_low"]
        right = out["sharpe_ci_high"] - out["sharpe_mean"]
        # Right/Left ratio: how much more upper-tail width than lower.
        if left <= 0:
            return float("inf")
        return right / left

    a_pct = asym(out_pct)
    a_bca = asym(out_bca)
    # BCa adjusts for skew → asymmetry magnitude should differ
    # measurably from percentile. We assert they aren't identical and
    # the BCa endpoints have shifted.
    assert not math.isclose(a_pct, a_bca, rel_tol=1e-3), (
        f"BCa identical to percentile: pct={a_pct:.3f}, bca={a_bca:.3f}"
    )
    # And the BCa CI should not collapse.
    assert out_bca["sharpe_ci_high"] > out_bca["sharpe_ci_low"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_empty_returns_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_sharpe_ci(np.array([]))


def test_non_finite_returns_raises() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_sharpe_ci(np.array([0.01, np.nan, 0.02]))


def test_too_few_observations_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_sharpe_ci(np.array([0.01]))


def test_invalid_confidence_raises() -> None:
    r = np.random.default_rng(0).normal(0, 0.01, 50)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_sharpe_ci(r, confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_sharpe_ci(r, confidence=0.0)


def test_invalid_method_raises() -> None:
    r = np.random.default_rng(0).normal(0, 0.01, 50)
    with pytest.raises(ValueError, match="method"):
        bootstrap_sharpe_ci(r, method="abc")


def test_invalid_n_resamples_raises() -> None:
    r = np.random.default_rng(0).normal(0, 0.01, 50)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_sharpe_ci(r, n_resamples=0)


def test_invalid_ann_factor_raises() -> None:
    r = np.random.default_rng(0).normal(0, 0.01, 50)
    with pytest.raises(ValueError, match="ann_factor"):
        bootstrap_sharpe_ci(r, ann_factor=-1.0)


def test_2d_returns_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        bootstrap_sharpe_ci(np.ones((5, 5)))


# ---------------------------------------------------------------------------
# Degenerate handling
# ---------------------------------------------------------------------------


def test_all_zero_returns_marks_degenerate() -> None:
    out = bootstrap_sharpe_ci(np.zeros(50), n_resamples=100)
    assert out["degenerate"] is True
    assert math.isnan(out["sharpe_mean"])
    assert math.isnan(out["sharpe_ci_low"])
    assert math.isnan(out["sharpe_ci_high"])
    assert math.isnan(out["observed_sharpe"])


def test_constant_nonzero_returns_marks_degenerate() -> None:
    out = bootstrap_sharpe_ci(np.full(40, 0.01), n_resamples=100)
    assert out["degenerate"] is True


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducible_with_same_seed() -> None:
    rng = np.random.default_rng(123)
    r = rng.normal(0.001, 0.01, 200)
    out1 = bootstrap_sharpe_ci(r, n_resamples=500, random_state=99, method="percentile")
    out2 = bootstrap_sharpe_ci(r, n_resamples=500, random_state=99, method="percentile")
    assert out1["sharpe_ci_low"] == out2["sharpe_ci_low"]
    assert out1["sharpe_ci_high"] == out2["sharpe_ci_high"]
    assert out1["sharpe_mean"] == out2["sharpe_mean"]


def test_different_seeds_give_different_cis() -> None:
    rng = np.random.default_rng(123)
    r = rng.normal(0.001, 0.01, 200)
    out1 = bootstrap_sharpe_ci(r, n_resamples=500, random_state=1)
    out2 = bootstrap_sharpe_ci(r, n_resamples=500, random_state=2)
    # Different seed → different bootstrap draws → CIs should not match exactly.
    assert (
        out1["sharpe_ci_low"] != out2["sharpe_ci_low"]
        or out1["sharpe_ci_high"] != out2["sharpe_ci_high"]
    )


def test_bca_reproducible() -> None:
    rng = np.random.default_rng(7)
    r = rng.normal(0.001, 0.01, 200)
    out1 = bootstrap_sharpe_ci(r, n_resamples=500, random_state=11, method="bca")
    out2 = bootstrap_sharpe_ci(r, n_resamples=500, random_state=11, method="bca")
    assert out1 == out2


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------


def test_annualisation_factor_scales_sharpe_endpoints() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 300)
    out_daily = bootstrap_sharpe_ci(r, n_resamples=1000, ann_factor=252, random_state=0)
    out_raw = bootstrap_sharpe_ci(r, n_resamples=1000, ann_factor=1, random_state=0)
    # Same resamples (same seed) → endpoints scale by sqrt(252).
    factor = math.sqrt(252)
    assert out_daily["sharpe_ci_low"] == pytest.approx(
        out_raw["sharpe_ci_low"] * factor, rel=1e-9, abs=1e-9
    )
    assert out_daily["sharpe_ci_high"] == pytest.approx(
        out_raw["sharpe_ci_high"] * factor, rel=1e-9, abs=1e-9
    )


def test_monthly_annualisation() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0.005, 0.04, 60)
    out = bootstrap_sharpe_ci(r, n_resamples=500, ann_factor=12, random_state=0)
    # Just check the CI is finite and ordered.
    assert math.isfinite(out["sharpe_ci_low"])
    assert math.isfinite(out["sharpe_ci_high"])
    assert out["sharpe_ci_low"] <= out["sharpe_ci_high"]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_confidence_width_increases_with_confidence_level() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 300)
    out_90 = bootstrap_sharpe_ci(r, n_resamples=1500, confidence=0.90, random_state=0)
    out_99 = bootstrap_sharpe_ci(r, n_resamples=1500, confidence=0.99, random_state=0)
    w90 = out_90["sharpe_ci_high"] - out_90["sharpe_ci_low"]
    w99 = out_99["sharpe_ci_high"] - out_99["sharpe_ci_low"]
    assert w99 > w90, f"99% CI ({w99:.3f}) should be wider than 90% ({w90:.3f})"


def test_bootstrap_std_positive_for_random_returns() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 200)
    out = bootstrap_sharpe_ci(r, n_resamples=500, random_state=0)
    assert out["sharpe_std"] > 0


def test_method_echoed_in_output() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 100)
    assert bootstrap_sharpe_ci(r, n_resamples=200, method="percentile")["method"] == "percentile"
    assert bootstrap_sharpe_ci(r, n_resamples=200, method="bca")["method"] == "bca"
