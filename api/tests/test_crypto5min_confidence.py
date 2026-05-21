"""Tests for ``pfm.crypto5min.confidence``.

Three areas:
* pure-math z-scores (``compute_z_model``, ``compute_z_edge``)
* confidence rubric (``compute_confidence_score``, breakdown invariants)
* signal strength bucketing (``signal_strength_from_confidence``)
"""

from __future__ import annotations

import math

import pytest

from pfm.crypto5min.confidence import (
    EDGE_FULL_CREDIT,
    N_FULL_CREDIT,
    STRENGTH_MEDIUM_THRESHOLD,
    STRENGTH_STRONG_THRESHOLD,
    WEIGHT_DATA_QUALITY,
    WEIGHT_EDGE_MAGNITUDE,
    WEIGHT_ENGINE_QUALITY,
    WEIGHT_TIME_DECAY,
    build_confidence_result,
    compute_confidence_score,
    compute_z_edge,
    compute_z_model,
    signal_strength_from_confidence,
)
from pfm.crypto5min.predictor import PredictorInputs, predict_up_prob

# ---------------------------------------------------------------------------
# compute_z_model
# ---------------------------------------------------------------------------


def test_z_model_zero_at_atm_zero_drift() -> None:
    """spot_t == spot_0, μ=0, σ>0 → z is slightly negative (only -σ²/2 drift)."""
    z = compute_z_model(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_used_annual=0.65,
        mu_used_annual=0.0,
    )
    # log_ratio=0; numerator = -0.5 σ² τ
    assert z < 0
    assert abs(z) < 0.01  # tiny because τ is small


def test_z_model_positive_when_spot_t_above_spot_0() -> None:
    z = compute_z_model(
        spot_t=60_300.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_used_annual=0.65,
        mu_used_annual=0.0,
    )
    assert z > 0


def test_z_model_negative_when_below() -> None:
    z = compute_z_model(
        spot_t=59_700.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_used_annual=0.65,
        mu_used_annual=0.0,
    )
    assert z < 0


def test_z_model_returns_signed_infinity_proxy_at_expiry() -> None:
    up = compute_z_model(60_100.0, 60_000.0, 0.0, 0.65, 0.0)
    dn = compute_z_model(59_900.0, 60_000.0, 0.0, 0.65, 0.0)
    eq = compute_z_model(60_000.0, 60_000.0, 0.0, 0.65, 0.0)
    assert up == pytest.approx(1e9)
    assert dn == pytest.approx(-1e9)
    assert eq == pytest.approx(1e9)


def test_z_model_zero_sigma_collapses_to_step() -> None:
    z = compute_z_model(60_100.0, 60_000.0, 300.0, 0.0, 0.0)
    assert z == pytest.approx(1e9)


def test_z_model_rejects_bad_prices() -> None:
    with pytest.raises(ValueError):
        compute_z_model(0.0, 60_000.0, 300.0, 0.65, 0.0)
    with pytest.raises(ValueError):
        compute_z_model(60_000.0, -1.0, 300.0, 0.65, 0.0)


def test_z_model_monotone_in_logratio() -> None:
    zs = [
        compute_z_model(60_000.0 + d, 60_000.0, 300.0, 0.65, 0.0) for d in (-500, -100, 0, 100, 500)
    ]
    assert zs == sorted(zs)


# ---------------------------------------------------------------------------
# compute_z_edge
# ---------------------------------------------------------------------------


def _basic_inputs() -> PredictorInputs:
    """Small offset + 5min window → prediction lands ~0.65 (not clipped).

    Important: the σ-jackknife test needs the GBM to be *off* the 0.005 /
    0.995 endpoint clip so perturbing σ produces a non-zero SE. Picking
    spot_t = spot_0 * (1 + 0.0008) puts us about 0.4σ above ATM.
    """
    return PredictorInputs(
        spot_t=60_050.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=0.65,
        sigma_short_annual=0.55,
        ofi_1m=0.1,
    )


def test_z_edge_none_when_no_market() -> None:
    inputs = _basic_inputs()
    pred = predict_up_prob(inputs)
    assert compute_z_edge(inputs, pred.prob_up, None) is None


def test_z_edge_positive_when_model_above_market() -> None:
    inputs = _basic_inputs()
    pred = predict_up_prob(inputs)
    z = compute_z_edge(inputs, pred.prob_up, market_prob=pred.prob_up - 0.10)
    assert z is not None and z > 0


def test_z_edge_negative_when_model_below_market() -> None:
    inputs = _basic_inputs()
    pred = predict_up_prob(inputs)
    z = compute_z_edge(inputs, pred.prob_up, market_prob=pred.prob_up + 0.10)
    assert z is not None and z < 0


def test_z_edge_is_clipped_for_sanity() -> None:
    inputs = _basic_inputs()
    pred = predict_up_prob(inputs)
    z = compute_z_edge(inputs, pred.prob_up, market_prob=pred.prob_up - 0.40)
    assert z is not None
    assert -50.0 <= z <= 50.0


def test_z_edge_zero_at_zero_edge() -> None:
    inputs = _basic_inputs()
    pred = predict_up_prob(inputs)
    z = compute_z_edge(inputs, pred.prob_up, market_prob=pred.prob_up)
    assert z == 0.0 or z is None  # SE>0 case ⇒ 0.0; SE=0 case ⇒ None


def test_z_edge_none_at_expiry_when_se_zero() -> None:
    """At T=0 the predictor collapses to a step — perturbing σ doesn't move the prob."""
    inputs = PredictorInputs(
        spot_t=60_500.0,
        spot_0=60_000.0,
        seconds_remaining=0.0,
        sigma_long_annual=0.65,
    )
    pred = predict_up_prob(inputs)
    z = compute_z_edge(inputs, pred.prob_up, market_prob=0.40)
    assert z is None


def test_z_edge_handles_missing_sigmas() -> None:
    """When σ_short is None, predictor still works → z_edge derives from σ_long only.

    Use a small offset so perturbing σ produces a measurable prob shift.
    """
    inputs = PredictorInputs(
        spot_t=60_050.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=0.65,
        sigma_short_annual=None,
    )
    pred = predict_up_prob(inputs)
    z = compute_z_edge(inputs, pred.prob_up, market_prob=pred.prob_up - 0.10)
    assert z is not None and z > 0


# ---------------------------------------------------------------------------
# compute_confidence_score
# ---------------------------------------------------------------------------


def test_confidence_zero_for_empty_state() -> None:
    b = compute_confidence_score(
        n_samples=0,
        live_engine_used=False,
        edge=None,
        seconds_remaining=300.0,
        window_seconds=300,
    )
    assert b.total == 0.0
    assert b.data_quality == 0.0
    assert b.engine_quality == 0.0
    assert b.edge_magnitude == 0.0
    assert b.time_decay == 0.0


def test_confidence_engine_only() -> None:
    b = compute_confidence_score(
        n_samples=0,
        live_engine_used=True,
        edge=None,
        seconds_remaining=300.0,
        window_seconds=300,
    )
    assert b.engine_quality == WEIGHT_ENGINE_QUALITY
    assert b.total == WEIGHT_ENGINE_QUALITY


def test_confidence_data_full_credit_saturates() -> None:
    b = compute_confidence_score(
        n_samples=N_FULL_CREDIT,
        live_engine_used=False,
        edge=None,
        seconds_remaining=300.0,
        window_seconds=300,
    )
    assert b.data_quality == WEIGHT_DATA_QUALITY


def test_confidence_data_saturates_above_full_credit() -> None:
    b = compute_confidence_score(
        n_samples=N_FULL_CREDIT * 10,
        live_engine_used=False,
        edge=None,
        seconds_remaining=300.0,
        window_seconds=300,
    )
    assert b.data_quality == WEIGHT_DATA_QUALITY


def test_confidence_edge_magnitude_saturates() -> None:
    b = compute_confidence_score(
        n_samples=0,
        live_engine_used=False,
        edge=EDGE_FULL_CREDIT,
        seconds_remaining=300.0,
        window_seconds=300,
    )
    assert b.edge_magnitude == WEIGHT_EDGE_MAGNITUDE
    b2 = compute_confidence_score(
        n_samples=0,
        live_engine_used=False,
        edge=EDGE_FULL_CREDIT * 5,
        seconds_remaining=300.0,
        window_seconds=300,
    )
    assert b2.edge_magnitude == WEIGHT_EDGE_MAGNITUDE


def test_confidence_time_decay_at_expiry() -> None:
    b = compute_confidence_score(
        n_samples=0,
        live_engine_used=False,
        edge=None,
        seconds_remaining=0.0,
        window_seconds=300,
    )
    assert b.time_decay == WEIGHT_TIME_DECAY


def test_confidence_time_decay_at_window_open() -> None:
    b = compute_confidence_score(
        n_samples=0,
        live_engine_used=False,
        edge=None,
        seconds_remaining=300.0,
        window_seconds=300,
    )
    assert b.time_decay == 0.0


def test_confidence_saturates_at_100() -> None:
    b = compute_confidence_score(
        n_samples=N_FULL_CREDIT,
        live_engine_used=True,
        edge=EDGE_FULL_CREDIT,
        seconds_remaining=0.0,
        window_seconds=300,
    )
    assert b.total == pytest.approx(100.0)


def test_confidence_negative_edge_same_credit() -> None:
    b = compute_confidence_score(
        n_samples=0,
        live_engine_used=False,
        edge=-0.10,
        seconds_remaining=300.0,
        window_seconds=300,
    )
    assert b.edge_magnitude > 0


def test_confidence_handles_zero_window() -> None:
    b = compute_confidence_score(
        n_samples=0,
        live_engine_used=False,
        edge=None,
        seconds_remaining=0.0,
        window_seconds=0,
    )
    assert b.time_decay == 0.0


# ---------------------------------------------------------------------------
# signal_strength_from_confidence
# ---------------------------------------------------------------------------


def test_signal_strength_strong_at_threshold() -> None:
    assert signal_strength_from_confidence(STRENGTH_STRONG_THRESHOLD) == "STRONG"
    assert signal_strength_from_confidence(100.0) == "STRONG"


def test_signal_strength_medium_in_middle() -> None:
    assert signal_strength_from_confidence(STRENGTH_MEDIUM_THRESHOLD) == "MEDIUM"
    assert signal_strength_from_confidence(STRENGTH_STRONG_THRESHOLD - 0.01) == "MEDIUM"


def test_signal_strength_weak_below_medium() -> None:
    assert signal_strength_from_confidence(0.0) == "WEAK"
    assert signal_strength_from_confidence(STRENGTH_MEDIUM_THRESHOLD - 0.01) == "WEAK"


# ---------------------------------------------------------------------------
# build_confidence_result — end-to-end glue
# ---------------------------------------------------------------------------


def test_build_confidence_result_no_market() -> None:
    inputs = _basic_inputs()
    pred = predict_up_prob(inputs)
    out = build_confidence_result(
        base_inputs=inputs,
        base_model_prob=pred.prob_up,
        market_prob=None,
        sigma_used_annual=pred.sigma_used_annual,
        mu_used_annual=pred.mu_used_annual,
        n_samples=30,
        live_engine_used=True,
        window_seconds=300,
    )
    assert out.z_edge is None
    assert math.isfinite(out.z_model)
    assert 0.0 <= out.confidence_score <= 100.0
    assert out.signal_strength in {"STRONG", "MEDIUM", "WEAK"}


def test_build_confidence_result_with_market() -> None:
    inputs = _basic_inputs()
    pred = predict_up_prob(inputs)
    out = build_confidence_result(
        base_inputs=inputs,
        base_model_prob=pred.prob_up,
        market_prob=pred.prob_up - 0.10,
        sigma_used_annual=pred.sigma_used_annual,
        mu_used_annual=pred.mu_used_annual,
        n_samples=N_FULL_CREDIT,
        live_engine_used=True,
        window_seconds=300,
    )
    assert out.z_edge is not None
    # 10% edge with engine on and full-credit data → confidence should be solid
    assert out.confidence_score > 40.0


def test_build_confidence_result_full_credit_strong() -> None:
    """High n_samples + live engine + big edge + near-expiry → STRONG."""
    inputs = PredictorInputs(
        spot_t=60_050.0,
        spot_0=60_000.0,
        seconds_remaining=30.0,  # near expiry → high time-decay credit
        sigma_long_annual=0.65,
        sigma_short_annual=0.55,
        ofi_1m=0.5,
    )
    pred = predict_up_prob(inputs)
    out = build_confidence_result(
        base_inputs=inputs,
        base_model_prob=pred.prob_up,
        market_prob=0.30,  # big edge ~0.4 → full edge credit
        sigma_used_annual=pred.sigma_used_annual,
        mu_used_annual=pred.mu_used_annual,
        n_samples=N_FULL_CREDIT * 2,
        live_engine_used=True,
        window_seconds=300,
    )
    assert out.confidence_score >= STRENGTH_STRONG_THRESHOLD
    assert out.signal_strength == "STRONG"


def test_build_confidence_result_returns_dict_serializable() -> None:
    inputs = _basic_inputs()
    pred = predict_up_prob(inputs)
    out = build_confidence_result(
        base_inputs=inputs,
        base_model_prob=pred.prob_up,
        market_prob=0.50,
        sigma_used_annual=pred.sigma_used_annual,
        mu_used_annual=pred.mu_used_annual,
        n_samples=10,
        live_engine_used=False,
        window_seconds=300,
    )
    d = out.as_dict()
    expected_keys = {
        "confidence_score",
        "signal_strength",
        "z_model",
        "z_edge",
        "confidence_breakdown",
        "confidence_components",
    }
    assert expected_keys.issubset(d.keys())
    breakdown = d["confidence_breakdown"]
    assert "data_quality" in breakdown
    assert "engine_quality" in breakdown
    assert "edge_magnitude" in breakdown
    assert "time_decay" in breakdown
    assert "total" in breakdown
