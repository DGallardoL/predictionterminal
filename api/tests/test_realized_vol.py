"""Tests for pfm.quant.realized_vol."""

from __future__ import annotations

import numpy as np
import pytest

from pfm.quant.realized_vol import (
    VALID_METHODS,
    realized_vol,
    realized_vol_harmonic_mean,
)

# -- helpers ---------------------------------------------------------------


def _synth_gbm_ohlc(
    n: int,
    sigma_daily: float,
    mu_daily: float = 0.0,
    seed: int = 0,
    intraday_steps: int = 78,
) -> np.ndarray:
    """Generate synthetic OHLC bars from a discretised GBM with known sigma.

    Each "day" is built from ``intraday_steps`` sub-steps so high/low capture
    intraday range. Returns a 2-D (n, 4) array of [open, high, low, close].
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / intraday_steps
    px = 100.0
    bars = np.empty((n, 4), dtype=float)
    for i in range(n):
        open_p = px
        path = [open_p]
        for _ in range(intraday_steps):
            shock = rng.normal(
                loc=(mu_daily - 0.5 * sigma_daily**2) * dt,
                scale=sigma_daily * np.sqrt(dt),
            )
            px = px * np.exp(shock)
            path.append(px)
        close_p = px
        high_p = max(path)
        low_p = min(path)
        bars[i, 0] = open_p
        bars[i, 1] = high_p
        bars[i, 2] = low_p
        bars[i, 3] = close_p
    return bars


# -- basic dispatch / validation --------------------------------------------


def test_valid_methods_exposed():
    assert set(VALID_METHODS) == {
        "close-to-close",
        "parkinson",
        "garman-klass",
        "rogers-satchell",
        "yang-zhang",
    }


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        realized_vol(np.array([0.01, -0.02]), method="bogus")


def test_invalid_ann_factor_raises():
    with pytest.raises(ValueError, match="ann_factor must be positive"):
        realized_vol(np.array([0.01, 0.02]), ann_factor=0)


# -- close-to-close --------------------------------------------------------


def test_close_to_close_matches_numpy_std_unannualised():
    rng = np.random.default_rng(42)
    r = rng.normal(0.0, 0.01, size=500)
    got = realized_vol(r, method="close-to-close", annualize=False)
    expected = float(np.std(r, ddof=1))
    assert got == pytest.approx(expected, rel=1e-12)


def test_close_to_close_all_zero_returns_zero():
    r = np.zeros(50)
    got = realized_vol(r, method="close-to-close", annualize=True)
    assert got == 0.0


def test_close_to_close_known_dgp_recovers_sigma():
    rng = np.random.default_rng(7)
    sigma_daily = 0.30 / np.sqrt(252)  # 30% annual vol
    r = rng.normal(0.0, sigma_daily, size=10_000)
    got = realized_vol(r, method="close-to-close", annualize=True, ann_factor=252)
    assert got == pytest.approx(0.30, rel=0.05)


def test_annualization_factor_correct():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.01, size=1000)
    daily = realized_vol(r, method="close-to-close", annualize=False)
    annual = realized_vol(r, method="close-to-close", annualize=True, ann_factor=252)
    assert annual == pytest.approx(daily * np.sqrt(252), rel=1e-12)


def test_close_to_close_rolling_window_shape():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, size=100)
    out = realized_vol(r, method="close-to-close", window=20)
    assert isinstance(out, np.ndarray)
    assert out.shape == (100 - 20 + 1,)
    assert np.all(out >= 0)


# -- OHLC estimators -------------------------------------------------------


def test_parkinson_known_value_literature_example():
    # Two bars where ln(H/L) = ln(2) for both -> sigma^2 = ln(2)^2 / (4 ln 2)
    # => sigma = sqrt(ln(2) / 4) = 0.5 * sqrt(ln(2))
    ohlc = np.array(
        [
            [1.0, 2.0, 1.0, 1.5],
            [1.5, 3.0, 1.5, 2.0],
        ]
    )
    sigma = realized_vol(ohlc, method="parkinson", annualize=False)
    expected = 0.5 * np.sqrt(np.log(2.0))
    assert sigma == pytest.approx(expected, rel=1e-12)


def test_parkinson_recovers_sigma_on_synthetic_gbm():
    sigma_daily = 0.02
    ohlc = _synth_gbm_ohlc(n=2000, sigma_daily=sigma_daily, seed=11)
    est = realized_vol(ohlc, method="parkinson", annualize=False)
    # Parkinson is biased low if you forget the 1/(4 ln 2) factor; with it,
    # should be close to the true sigma_daily.
    assert est == pytest.approx(sigma_daily, rel=0.15)


def test_garman_klass_more_efficient_than_close_to_close():
    # Across many seeds, GK estimator variance should be lower than C2C.
    sigma_daily = 0.02
    gk_vals, cc_vals = [], []
    for seed in range(40):
        ohlc = _synth_gbm_ohlc(n=60, sigma_daily=sigma_daily, seed=seed)
        gk_vals.append(realized_vol(ohlc, method="garman-klass", annualize=False))
        # Close-to-close from close column.
        close = ohlc[:, 3]
        r = np.diff(np.log(close))
        cc_vals.append(realized_vol(r, method="close-to-close", annualize=False))
    assert np.std(gk_vals) < np.std(cc_vals)


def test_rogers_satchell_drift_independence():
    # Adding a drift term shouldn't shift the RS estimate much.
    sigma_daily = 0.02
    no_drift = _synth_gbm_ohlc(n=1500, sigma_daily=sigma_daily, mu_daily=0.0, seed=21)
    with_drift = _synth_gbm_ohlc(n=1500, sigma_daily=sigma_daily, mu_daily=0.005, seed=21)
    rs1 = realized_vol(no_drift, method="rogers-satchell", annualize=False)
    rs2 = realized_vol(with_drift, method="rogers-satchell", annualize=False)
    # Allow modest tolerance because drift slightly biases sample mean of paths.
    assert abs(rs1 - rs2) / rs1 < 0.20


def test_yang_zhang_handles_overnight_gaps():
    # Build OHLC where each open != prior close (overnight gap).
    n = 500
    sigma_daily = 0.015
    rng = np.random.default_rng(99)
    ohlc = _synth_gbm_ohlc(n=n, sigma_daily=sigma_daily, seed=4)
    # Inject overnight jumps: open_t = close_{t-1} * exp(N(0, 0.005)).
    for i in range(1, n):
        gap = rng.normal(0.0, 0.005)
        scale = np.exp(gap)
        ohlc[i, 0] *= scale
        ohlc[i, 1] *= scale
        ohlc[i, 2] *= scale
        ohlc[i, 3] *= scale
    yz = realized_vol(ohlc, method="yang-zhang", annualize=False)
    assert yz > 0
    assert np.isfinite(yz)


def test_ohlc_methods_run_on_structured_array():
    ohlc_2d = _synth_gbm_ohlc(n=100, sigma_daily=0.02, seed=2)
    structured = np.zeros(
        100,
        dtype=[("open", float), ("high", float), ("low", float), ("close", float)],
    )
    structured["open"] = ohlc_2d[:, 0]
    structured["high"] = ohlc_2d[:, 1]
    structured["low"] = ohlc_2d[:, 2]
    structured["close"] = ohlc_2d[:, 3]
    for m in ("parkinson", "garman-klass", "rogers-satchell", "yang-zhang"):
        v = realized_vol(structured, method=m, annualize=False)
        assert v > 0 and np.isfinite(v), f"{m} produced bad value {v}"


def test_ohlc_invalid_shape_raises():
    bad = np.array([[1.0, 2.0, 3.0]])  # only 3 columns
    with pytest.raises(ValueError, match="OHLC array must be 2-D"):
        realized_vol(bad, method="garman-klass")


def test_rolling_window_ohlc_shape():
    ohlc = _synth_gbm_ohlc(n=80, sigma_daily=0.02, seed=5)
    out = realized_vol(ohlc, method="garman-klass", window=20, annualize=False)
    assert isinstance(out, np.ndarray)
    assert out.shape == (80 - 20 + 1,)


# -- edge cases ------------------------------------------------------------


def test_empty_returns_scalar_zero():
    assert realized_vol(np.array([]), method="close-to-close") == 0.0


def test_length_one_returns_zero():
    assert realized_vol(np.array([0.01]), method="close-to-close") == 0.0


def test_nan_returns_ignored():
    r = np.array([0.01, np.nan, -0.02, np.nan, 0.005])
    got = realized_vol(r, method="close-to-close", annualize=False)
    clean = np.array([0.01, -0.02, 0.005])
    expected = float(np.std(clean, ddof=1))
    assert got == pytest.approx(expected, rel=1e-12)


def test_rolling_window_larger_than_series_returns_empty():
    r = np.array([0.01, -0.02, 0.005])
    out = realized_vol(r, method="close-to-close", window=10)
    assert isinstance(out, np.ndarray) and out.size == 0


def test_rolling_window_invalid_raises():
    r = np.array([0.01, -0.02, 0.005, 0.001])
    with pytest.raises(ValueError, match="window must be a positive integer"):
        realized_vol(r, method="close-to-close", window=0)


# -- harmonic mean ---------------------------------------------------------


def test_harmonic_mean_less_than_arithmetic_for_varying_inputs():
    rng = np.random.default_rng(33)
    series_list = []
    sigmas = [0.10, 0.20, 0.50, 0.80]  # widely varying annualised vols
    for s in sigmas:
        daily = s / np.sqrt(252)
        series_list.append(rng.normal(0.0, daily, size=5000))
    hm = realized_vol_harmonic_mean(series_list)
    am = float(np.mean(sigmas))
    assert hm < am
    # Should be in the neighbourhood of true harmonic mean of the sigmas.
    true_hm = len(sigmas) / sum(1.0 / s for s in sigmas)
    assert hm == pytest.approx(true_hm, rel=0.10)


def test_harmonic_mean_empty_returns_zero():
    assert realized_vol_harmonic_mean([]) == 0.0


def test_harmonic_mean_all_zero_returns_zero():
    series = [np.zeros(100), np.zeros(50)]
    assert realized_vol_harmonic_mean(series) == 0.0


def test_harmonic_mean_skips_invalid_series():
    rng = np.random.default_rng(7)
    good = rng.normal(0.0, 0.01, size=500)
    bad_short = np.array([0.01])  # length < 2 -> skipped
    bad_zero = np.zeros(200)  # zero vol -> skipped
    hm = realized_vol_harmonic_mean([good, bad_short, bad_zero])
    expected = float(np.std(good, ddof=1) * np.sqrt(252))
    assert hm == pytest.approx(expected, rel=1e-9)


def test_harmonic_mean_equal_inputs_equals_value():
    rng = np.random.default_rng(8)
    s = rng.normal(0.0, 0.01, size=500)
    hm = realized_vol_harmonic_mean([s, s.copy(), s.copy()])
    expected = float(np.std(s, ddof=1) * np.sqrt(252))
    assert hm == pytest.approx(expected, rel=1e-9)
