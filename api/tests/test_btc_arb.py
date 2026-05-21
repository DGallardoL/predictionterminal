"""Unit tests for `pfm.btc_arb`."""

from __future__ import annotations

import math

import pytest

from pfm.btc_arb import (
    SECONDS_PER_YEAR,
    arb_signal,
    compute_fair_up_prob,
    realized_volatility,
)

# ---------------------------------------------------------------------------
# compute_fair_up_prob
# ---------------------------------------------------------------------------


def test_at_the_money_midpoint_is_near_half():
    """When BTC_t == BTC_0 with mu=0, the GBM Up-prob is slightly under 0.5
    because of the -sigma^2/2 drift term. For a 5-min window at 65% vol it
    should still be very close to 0.5 (within ~0.005)."""
    p = compute_fair_up_prob(btc_t=60_000.0, btc_0=60_000.0, seconds_remaining=300.0, vol_ann=0.65)
    assert 0.0 < p < 1.0
    # Slightly below 0.5 because of -sigma^2/2 drift; very close at short tau.
    assert abs(p - 0.5) < 0.01
    assert p < 0.5


def test_deep_in_the_money_and_out_of_the_money():
    """A 1% rally with little time left should make Up nearly certain;
    a 1% drop should make Up nearly impossible."""
    up = compute_fair_up_prob(btc_t=60_600.0, btc_0=60_000.0, seconds_remaining=30.0, vol_ann=0.65)
    down = compute_fair_up_prob(
        btc_t=59_400.0, btc_0=60_000.0, seconds_remaining=30.0, vol_ann=0.65
    )
    assert up > 0.99
    assert down < 0.01


def test_zero_remaining_time_is_deterministic():
    """At T=t the answer collapses to the indicator function."""
    assert compute_fair_up_prob(60_001.0, 60_000.0, 0.0) == 1.0
    assert compute_fair_up_prob(59_999.0, 60_000.0, 0.0) == 0.0
    # Exact tie counts as Up (Polymarket resolves on >=).
    assert compute_fair_up_prob(60_000.0, 60_000.0, 0.0) == 1.0


def test_higher_vol_pulls_probability_toward_half():
    """With BTC slightly above start, raising sigma should reduce the
    Up-probability toward 0.5 (more uncertainty about the terminal)."""
    args = {"btc_t": 60_300.0, "btc_0": 60_000.0, "seconds_remaining": 240.0}
    p_low = compute_fair_up_prob(**args, vol_ann=0.30)
    p_mid = compute_fair_up_prob(**args, vol_ann=0.65)
    p_hi = compute_fair_up_prob(**args, vol_ann=1.50)
    assert p_low > p_mid > p_hi > 0.5


def test_compute_fair_up_prob_input_validation():
    with pytest.raises(ValueError):
        compute_fair_up_prob(0.0, 60_000.0, 60.0)
    with pytest.raises(ValueError):
        compute_fair_up_prob(60_000.0, -1.0, 60.0)
    with pytest.raises(ValueError):
        compute_fair_up_prob(60_000.0, 60_000.0, 60.0, vol_ann=-0.1)
    with pytest.raises(ValueError):
        compute_fair_up_prob(60_000.0, 60_000.0, -1.0)


# ---------------------------------------------------------------------------
# arb_signal
# ---------------------------------------------------------------------------


def test_arb_signal_edge_threshold_logic():
    # Fair=0.60, market=0.50 -> 10pp underpriced -> BUY_UP at 3pp threshold.
    assert arb_signal(poly_up_mid=0.50, fair_up=0.60) == "BUY_UP"
    # Fair=0.40, market=0.50 -> 10pp overpriced -> SELL_UP.
    assert arb_signal(poly_up_mid=0.50, fair_up=0.40) == "SELL_UP"
    # 2pp gap < 3pp threshold -> HOLD.
    assert arb_signal(poly_up_mid=0.50, fair_up=0.52) == "HOLD"
    assert arb_signal(poly_up_mid=0.50, fair_up=0.48) == "HOLD"
    # Exactly at threshold -> trade fires.
    assert arb_signal(poly_up_mid=0.50, fair_up=0.53, edge_threshold=0.03) == "BUY_UP"
    assert arb_signal(poly_up_mid=0.50, fair_up=0.47, edge_threshold=0.03) == "SELL_UP"
    # Custom (looser) threshold suppresses small-edge trades.
    assert arb_signal(poly_up_mid=0.50, fair_up=0.55, edge_threshold=0.10) == "HOLD"
    with pytest.raises(ValueError):
        arb_signal(0.5, 0.5, edge_threshold=-0.01)


# ---------------------------------------------------------------------------
# realized_volatility
# ---------------------------------------------------------------------------


def test_realized_volatility_recovers_known_sigma():
    """Generate a synthetic GBM-ish series with known per-step vol and
    confirm we recover roughly the right annualized number."""
    import random

    random.seed(42)
    dt = 1.0  # one second between samples
    sigma_ann_true = 0.65
    sigma_step = sigma_ann_true * math.sqrt(dt / SECONDS_PER_YEAR)
    px = [60_000.0]
    for _ in range(2000):
        px.append(px[-1] * math.exp(random.gauss(0.0, sigma_step)))
    vol_hat = realized_volatility(px, dt_seconds=dt)
    # 2000 samples is plenty to be within ~5% of true.
    assert abs(vol_hat - sigma_ann_true) / sigma_ann_true < 0.10


def test_realized_volatility_edge_cases():
    # Too few prices -> return 0.0 (don't blow up).
    assert realized_volatility([60_000.0], dt_seconds=1.0) == 0.0
    assert realized_volatility([60_000.0, 60_010.0], dt_seconds=1.0) == 0.0
    # Bad inputs raise.
    with pytest.raises(ValueError):
        realized_volatility([60_000.0, 60_010.0, 60_020.0], dt_seconds=0.0)
    with pytest.raises(ValueError):
        realized_volatility([60_000.0, 0.0, 60_020.0], dt_seconds=1.0)
