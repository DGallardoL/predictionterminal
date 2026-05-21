"""Tests for the comparator + Kelly stake fraction."""

from __future__ import annotations

import pytest

from pfm.crypto5min.comparator import (
    DEFAULT_EDGE_THRESHOLD,
    ComparisonResult,
    compare_market_vs_model,
    decide_signal,
    kelly_fraction,
)
from pfm.crypto5min.predictor import predict_for_window

# ---------------------------------------------------------------------------
# decide_signal
# ---------------------------------------------------------------------------


def test_decide_signal_buy_yes() -> None:
    assert decide_signal(0.10) == "BUY_YES"


def test_decide_signal_buy_no() -> None:
    assert decide_signal(-0.10) == "BUY_NO"


def test_decide_signal_wait_below_threshold() -> None:
    assert decide_signal(0.02) == "WAIT"
    assert decide_signal(-0.02) == "WAIT"


def test_decide_signal_at_threshold_triggers() -> None:
    assert decide_signal(DEFAULT_EDGE_THRESHOLD) == "BUY_YES"
    assert decide_signal(-DEFAULT_EDGE_THRESHOLD) == "BUY_NO"


def test_decide_signal_custom_threshold() -> None:
    assert decide_signal(0.05, threshold=0.10) == "WAIT"
    assert decide_signal(0.15, threshold=0.10) == "BUY_YES"


def test_decide_signal_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError):
        decide_signal(0.10, threshold=-0.01)


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------


def test_kelly_zero_when_no_edge() -> None:
    assert kelly_fraction(0.50, 0.50) == 0.0


def test_kelly_positive_when_model_higher_than_market() -> None:
    f = kelly_fraction(0.60, 0.50)
    assert 0.0 < f <= 0.20


def test_kelly_capped() -> None:
    f = kelly_fraction(0.99, 0.05, cap=0.20)
    assert f == pytest.approx(0.20)


def test_kelly_no_side_when_model_lower() -> None:
    """Even when model says NO, we return a *non-negative* stake (just a NO bet)."""
    f = kelly_fraction(0.30, 0.50)
    assert f > 0.0


def test_kelly_handles_extreme_probs_without_blowing_up() -> None:
    # Pure 0 or 1 inputs would divide by zero in textbook Kelly — guard clips.
    f = kelly_fraction(1.0, 0.5)
    assert 0.0 <= f <= 0.20
    f2 = kelly_fraction(0.0, 0.5)
    assert 0.0 <= f2 <= 0.20


def test_kelly_rejects_negative_cap() -> None:
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 0.5, cap=-0.01)


# ---------------------------------------------------------------------------
# compare_market_vs_model
# ---------------------------------------------------------------------------


def _basic_pred(prob_up: float = 0.55):
    """Build a real ModelPrediction by running the predictor for a tame case.

    We pick spot/spot_0/σ to land near 0.50 so the +0.20/-0.20 nudges used
    in :func:`test_compare_edge_sign_matches_signal` stay inside [0, 1].
    """
    pred = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=0.65,
    )
    return pred


def test_compare_returns_comparison_result() -> None:
    pred = _basic_pred()
    out = compare_market_vs_model(
        slug="btc-updown-5m-1700000000",
        asset="BTC",
        window_minutes=5,
        market_prob_up=0.50,
        prediction=pred,
    )
    assert isinstance(out, ComparisonResult)
    assert out.slug == "btc-updown-5m-1700000000"
    assert out.asset == "BTC"
    assert out.window_minutes == 5


def test_compare_edge_sign_matches_signal() -> None:
    """After market-anchor (weight=0.90) the edge shrinks 10× — use wide
    gaps so the resulting edge clears the 3% threshold in both directions."""
    pred = _basic_pred()
    # gap=0.45 → edge = 0.10·0.45 = 0.045 > 0.03 threshold ⇒ fires
    out_buy = compare_market_vs_model(
        slug="x",
        asset="BTC",
        window_minutes=5,
        market_prob_up=max(0.001, pred.prob_up - 0.45),
        prediction=pred,
    )
    out_sell = compare_market_vs_model(
        slug="x",
        asset="BTC",
        window_minutes=5,
        market_prob_up=min(0.999, pred.prob_up + 0.45),
        prediction=pred,
    )
    out_wait = compare_market_vs_model(
        slug="x",
        asset="BTC",
        window_minutes=5,
        market_prob_up=pred.prob_up + 0.01,
        prediction=pred,
    )
    assert out_buy.signal == "BUY_YES"
    assert out_sell.signal == "BUY_NO"
    assert out_wait.signal == "WAIT"


def test_compare_kelly_is_zero_on_wait() -> None:
    pred = _basic_pred()
    out = compare_market_vs_model(
        slug="x",
        asset="BTC",
        window_minutes=5,
        market_prob_up=pred.prob_up,
        prediction=pred,
    )
    assert out.signal == "WAIT"
    assert out.kelly_fraction == 0.0


def test_compare_rejects_market_prob_out_of_range() -> None:
    pred = _basic_pred()
    with pytest.raises(ValueError):
        compare_market_vs_model(
            slug="x",
            asset="BTC",
            window_minutes=5,
            market_prob_up=1.5,
            prediction=pred,
        )


def test_compare_as_dict_round_trip() -> None:
    pred = _basic_pred()
    out = compare_market_vs_model(
        slug="abc",
        asset="BTC",
        window_minutes=15,
        market_prob_up=0.40,
        prediction=pred,
    )
    d = out.as_dict()
    assert d["slug"] == "abc"
    assert d["window_minutes"] == 15
    assert "components" in d
    # After market-anchor, edge = anchored_model - market. The anchored
    # model = w·market + (1-w)·gbm, so edge = (1-w)·(gbm - market).
    from pfm.crypto5min.comparator import MARKET_ANCHOR_WEIGHT

    expected_edge = (1.0 - MARKET_ANCHOR_WEIGHT) * (pred.prob_up - 0.40)
    assert d["edge"] == pytest.approx(expected_edge)
    # Raw GBM should also be exposed for diagnostics.
    assert d["model_prob_gbm_raw"] == pytest.approx(pred.prob_up)
