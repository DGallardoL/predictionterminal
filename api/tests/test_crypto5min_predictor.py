"""Synthetic-DGP tests for ``pfm.crypto5min.predictor``.

We deliberately *don't* use scipy/numpy here — the predictor is plain math
and the assertions are easy to express with python's stdlib.
"""

from __future__ import annotations

import math

import pytest

from pfm.crypto5min.predictor import (
    MU_OFI_SCALE,
    MU_WHALE_SCALE,
    SECONDS_PER_YEAR,
    SIGMA_CEILING,
    SIGMA_FLOOR,
    Z_REV_THRESHOLD,
    ModelPrediction,
    PredictorInputs,
    blend_sigma,
    ofi_drift,
    predict_for_window,
    reversion_overlay,
    whale_drift,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _five_min(seconds: float = 300.0) -> dict[str, float]:
    return {"seconds_remaining": seconds}


# ---------------------------------------------------------------------------
# blend_sigma
# ---------------------------------------------------------------------------


def test_blend_sigma_both_none_returns_default_clipped() -> None:
    assert blend_sigma(None, None) == pytest.approx(0.60)


def test_blend_sigma_long_only() -> None:
    assert blend_sigma(0.50, None) == pytest.approx(0.50)


def test_blend_sigma_short_only() -> None:
    assert blend_sigma(None, 0.80) == pytest.approx(0.80)


def test_blend_sigma_variance_weighted() -> None:
    # σ_long²=0.36, σ_short²=0.64; λ=0.4 → 0.6·0.36 + 0.4·0.64 = 0.472
    blended = blend_sigma(0.60, 0.80, lambda_short=0.4)
    assert blended == pytest.approx(math.sqrt(0.472), rel=1e-6)


def test_blend_sigma_clamps_to_floor() -> None:
    assert blend_sigma(0.01, 0.01) == pytest.approx(SIGMA_FLOOR)


def test_blend_sigma_clamps_to_ceiling() -> None:
    assert blend_sigma(10.0, 10.0) == pytest.approx(SIGMA_CEILING)


# ---------------------------------------------------------------------------
# ofi_drift / whale_drift
# ---------------------------------------------------------------------------


def test_ofi_drift_zero_at_zero_ofi() -> None:
    assert ofi_drift(0.0) == 0.0


def test_ofi_drift_capped_at_extreme() -> None:
    assert ofi_drift(1.0) == pytest.approx(MU_OFI_SCALE)
    assert ofi_drift(-1.0) == pytest.approx(-MU_OFI_SCALE)
    assert ofi_drift(99.0) == pytest.approx(MU_OFI_SCALE)
    assert ofi_drift(-99.0) == pytest.approx(-MU_OFI_SCALE)


def test_ofi_drift_linear_in_between() -> None:
    assert ofi_drift(0.5) == pytest.approx(MU_OFI_SCALE * 0.5)


def test_whale_drift_none_returns_zero() -> None:
    assert whale_drift(None, 100.0) == 0.0
    assert whale_drift(50.0, None) == 0.0
    assert whale_drift(50.0, 0.0) == 0.0


def test_whale_drift_capped_at_extreme_ratio() -> None:
    # whale_signed > notional shouldn't happen but defend against it.
    assert whale_drift(1_000_000.0, 1_000.0) == pytest.approx(MU_WHALE_SCALE)
    assert whale_drift(-1_000_000.0, 1_000.0) == pytest.approx(-MU_WHALE_SCALE)


def test_whale_drift_linear() -> None:
    assert whale_drift(500.0, 1000.0) == pytest.approx(MU_WHALE_SCALE * 0.5)


# ---------------------------------------------------------------------------
# reversion_overlay
# ---------------------------------------------------------------------------


def test_reversion_overlay_inactive_below_threshold() -> None:
    drift, opp = reversion_overlay(1.5, base_drift=0.20)
    assert drift == pytest.approx(0.20)
    assert opp == 0.0


def test_reversion_overlay_at_threshold_no_pull() -> None:
    drift, opp = reversion_overlay(Z_REV_THRESHOLD, base_drift=0.20)
    # shrink=1.0, opposite_pull = 0
    assert drift == pytest.approx(0.20)
    assert opp == pytest.approx(0.0)


def test_reversion_overlay_partial_shrink_above_threshold() -> None:
    drift, opp = reversion_overlay(3.0, base_drift=0.20)
    # shrink = (4-3)/(4-2) = 0.5
    assert drift == pytest.approx(0.10)
    # opposite_pull = (1-0.5)*(MU_OFI_SCALE/2) = 0.5 * 0.15 = 0.075, signed opposite of z
    assert opp == pytest.approx(-MU_OFI_SCALE * 0.25)


def test_reversion_overlay_full_takeover_at_high_z() -> None:
    drift, opp = reversion_overlay(5.0, base_drift=0.20)
    assert drift == 0.0
    # |z|>4 → shrink stays at 0 → opposite_pull = MU_OFI_SCALE/2, sign opposite of z (positive z → negative pull)
    assert opp == pytest.approx(-MU_OFI_SCALE * 0.5)


def test_reversion_overlay_handles_none_z() -> None:
    drift, opp = reversion_overlay(None, base_drift=0.10)
    assert drift == 0.10
    assert opp == 0.0


def test_reversion_overlay_negative_z_pulls_positive() -> None:
    drift, opp = reversion_overlay(-3.0, base_drift=-0.20)
    assert drift == pytest.approx(-0.10)
    assert opp > 0.0


# ---------------------------------------------------------------------------
# predict_up_prob — core invariants
# ---------------------------------------------------------------------------


def test_predict_returns_modelprediction() -> None:
    out = predict_for_window(spot_t=60_000.0, spot_0=60_000.0, **_five_min())
    assert isinstance(out, ModelPrediction)
    assert 0.005 <= out.prob_up <= 0.995


def test_predict_atm_is_near_half() -> None:
    out = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=0.65,
    )
    # σ²/2 drift makes it slightly below 0.5 at zero mu.
    assert 0.49 < out.prob_up < 0.50


def test_predict_deep_itm_is_near_one() -> None:
    out = predict_for_window(
        spot_t=60_600.0,
        spot_0=60_000.0,
        seconds_remaining=30.0,
        sigma_long_annual=0.65,
    )
    assert out.prob_up >= 0.99


def test_predict_deep_otm_is_near_zero() -> None:
    out = predict_for_window(
        spot_t=59_400.0,
        spot_0=60_000.0,
        seconds_remaining=30.0,
        sigma_long_annual=0.65,
    )
    assert out.prob_up <= 0.01


def test_predict_at_expiry_is_deterministic() -> None:
    up = predict_for_window(spot_t=60_001.0, spot_0=60_000.0, seconds_remaining=0.0)
    down = predict_for_window(spot_t=59_999.0, spot_0=60_000.0, seconds_remaining=0.0)
    tie = predict_for_window(spot_t=60_000.0, spot_0=60_000.0, seconds_remaining=0.0)
    assert up.prob_up == 1.0
    assert down.prob_up == 0.0
    assert tie.prob_up == 1.0


def test_predict_clipped_to_interior() -> None:
    """Output never lands exactly on 0/1 except for expiry edge case."""
    out = predict_for_window(
        spot_t=70_000.0,
        spot_0=60_000.0,
        seconds_remaining=10.0,
        sigma_long_annual=0.30,
    )
    assert out.prob_up < 1.0
    assert out.prob_up >= 0.99


def test_predict_positive_ofi_raises_prob() -> None:
    base = predict_for_window(spot_t=60_000.0, spot_0=60_000.0, seconds_remaining=300.0)
    pos = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        ofi_1m=0.8,
    )
    assert pos.prob_up > base.prob_up


def test_predict_negative_ofi_lowers_prob() -> None:
    base = predict_for_window(spot_t=60_000.0, spot_0=60_000.0, seconds_remaining=300.0)
    neg = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        ofi_1m=-0.8,
    )
    assert neg.prob_up < base.prob_up


def test_predict_extreme_z_flips_drift_sign() -> None:
    """Above |z|>=4 the overlay should fully take over: very positive z → drift negative."""
    out_extreme_pos_z = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        ofi_1m=0.8,
        z_vwap=5.0,
    )
    # OFI says up, but z says way too high → net drift is negative.
    assert out_extreme_pos_z.mu_used_annual < 0


def test_predict_whale_inflow_raises_prob() -> None:
    no_whale = predict_for_window(spot_t=60_000.0, spot_0=60_000.0, seconds_remaining=300.0)
    with_whale = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        whale_signed_notional_5m=5_000_000.0,
        notional_5m=5_000_000.0,
    )
    assert with_whale.prob_up > no_whale.prob_up


def test_predict_high_sigma_pulls_atm_to_half() -> None:
    """At very high σ a *slightly* ITM call gets pulled back toward 0.5."""
    low_vol = predict_for_window(
        spot_t=60_300.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=0.30,
    )
    high_vol = predict_for_window(
        spot_t=60_300.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=2.50,
    )
    assert low_vol.prob_up > high_vol.prob_up


def test_predict_short_window_more_sensitive_to_logratio() -> None:
    """Smaller seconds_remaining ⇒ stronger conviction at the same log ratio."""
    short = predict_for_window(
        spot_t=60_100.0,
        spot_0=60_000.0,
        seconds_remaining=30.0,
        sigma_long_annual=0.65,
    )
    long = predict_for_window(
        spot_t=60_100.0,
        spot_0=60_000.0,
        seconds_remaining=600.0,
        sigma_long_annual=0.65,
    )
    assert short.prob_up > long.prob_up


def test_predict_returns_dict_serializable() -> None:
    out = predict_for_window(spot_t=60_000.0, spot_0=60_000.0, seconds_remaining=300.0)
    d = out.as_dict()
    assert "prob_up" in d and "components" in d
    assert d["components"]["lambda_short"] == pytest.approx(0.4)


def test_predict_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError):
        PredictorInputs(spot_t=0.0, spot_0=60_000.0, seconds_remaining=300.0)
    with pytest.raises(ValueError):
        PredictorInputs(spot_t=60_000.0, spot_0=-1.0, seconds_remaining=300.0)


def test_predict_rejects_negative_seconds() -> None:
    with pytest.raises(ValueError):
        PredictorInputs(spot_t=60_000.0, spot_0=60_000.0, seconds_remaining=-1.0)


def test_predict_rejects_ofi_out_of_range() -> None:
    with pytest.raises(ValueError):
        PredictorInputs(spot_t=60_000.0, spot_0=60_000.0, seconds_remaining=300.0, ofi_1m=2.0)


def test_predict_seconds_per_year_constant() -> None:
    assert SECONDS_PER_YEAR == 365 * 24 * 3600


def test_predict_mu_capped_at_overall_cap() -> None:
    """Stacking OFI + whale + extreme inputs still respects the global MU_CAP."""
    out = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        ofi_1m=1.0,
        whale_signed_notional_5m=10_000_000.0,
        notional_5m=10_000_000.0,
    )
    assert abs(out.mu_used_annual) <= 0.4501  # MU_CAP plus epsilon


def test_predict_monotone_in_logratio() -> None:
    """As spot_t increases past spot_0 at fixed σ/T, prob is strictly increasing."""
    probs = []
    for delta in (-500, -200, 0, 200, 500):
        out = predict_for_window(
            spot_t=60_000.0 + delta,
            spot_0=60_000.0,
            seconds_remaining=300.0,
            sigma_long_annual=0.65,
        )
        probs.append(out.prob_up)
    assert probs == sorted(probs)
