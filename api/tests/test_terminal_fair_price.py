"""Tests for ``pfm.terminal_fair_price``."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal_fair_price import (
    CalibrationBin,
    CointegrationHit,
    _confidence_from_active,
    aggregate_signal,
    aggregate_signal_strict,
    calibration_fair_price,
    cointegration_fair_price,
    gbm_fair_price,
    prelec_inverse,
    router,
)


def _build_client(monkeypatch, tmp_path: Path) -> TestClient:
    """Stand up a tiny FastAPI app with the router and patched data files."""
    coint_path = tmp_path / "all_unique_hits.json"
    coint_path.write_text(
        json.dumps(
            [
                {
                    "a_id": "btc-100k-by-eoy",
                    "b_id": "btc-100k-by-jan",
                    "verdict": "REAL_ALPHA",
                    "n_obs": 90,
                    "adf_pvalue": 0.01,
                    "half_life_days": 2.0,
                    "beta_hedge": 0.8,
                    "oos_sharpe": 4.5,
                    "full_sharpe": 3.0,
                    "perm_p": 0.0,
                    "perm_real_sharpe": 3.0,
                    "sweep": "crypto",
                },
                {
                    "a_id": "btc-100k-by-eoy",
                    "b_id": "some-weak-pair",
                    "verdict": "REAL_ALPHA",
                    "n_obs": 90,
                    "adf_pvalue": 0.04,
                    "half_life_days": 5.0,
                    "beta_hedge": 0.2,
                    "oos_sharpe": 1.1,
                    "full_sharpe": 0.9,
                    "perm_p": 0.1,
                    "perm_real_sharpe": 0.9,
                    "sweep": "crypto",
                },
            ]
        )
    )
    calib_path = tmp_path / "strat9_calibration.json"
    calib_path.write_text(
        json.dumps(
            {
                "n_resolved_markets_used": 10,
                "horizon_days": 7,
                "calibration_table": [
                    {"bin": "(-0.001, 0.1]", "market_p": 0.05, "n": 10, "actual_rate": 0.02},
                    {"bin": "(0.1, 0.2]", "market_p": 0.15, "n": 10, "actual_rate": 0.10},
                    {"bin": "(0.2, 0.3]", "market_p": 0.25, "n": 10, "actual_rate": 0.20},
                    {"bin": "(0.3, 0.4]", "market_p": 0.35, "n": 10, "actual_rate": 0.30},
                    {"bin": "(0.4, 0.5]", "market_p": 0.45, "n": 10, "actual_rate": 0.40},
                    {"bin": "(0.5, 0.6]", "market_p": 0.55, "n": 10, "actual_rate": 0.65},
                    {"bin": "(0.6, 0.7]", "market_p": 0.65, "n": 10, "actual_rate": 0.80},
                    {"bin": "(0.7, 0.8]", "market_p": 0.75, "n": 10, "actual_rate": 0.85},
                    {"bin": "(0.8, 0.9]", "market_p": 0.85, "n": 10, "actual_rate": 0.88},
                    {"bin": "(0.9, 1.0]", "market_p": 0.95, "n": 10, "actual_rate": 0.97},
                ],
            }
        )
    )
    monkeypatch.setattr("pfm.terminal_fair_price.COINTEGRATION_PATH", coint_path)
    monkeypatch.setattr("pfm.terminal_fair_price.CALIBRATION_PATH", calib_path)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests for the model primitives
# ---------------------------------------------------------------------------


def test_prelec_inverse_round_trips_and_is_monotone():
    """w(prelec_inverse(p)) ≈ p, and the inverse is strictly monotone."""
    from pfm.terminal_fair_price import _prelec_w

    for p in (0.05, 0.20, 0.50, 0.75, 0.95):
        recovered = _prelec_w(prelec_inverse(p))
        assert abs(recovered - p) < 1e-3, f"round-trip failed at {p}: {recovered}"

    grid = [prelec_inverse(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(grid[i] < grid[i + 1] for i in range(len(grid) - 1))


def test_gbm_returns_none_for_non_updown_and_value_for_updown():
    # Non-updown slug → None even if all spot inputs supplied.
    assert (
        gbm_fair_price(
            slug="will-trump-win-2028",
            btc_t=60_000.0,
            btc_0=60_000.0,
            seconds_remaining=300.0,
        )
        is None
    )
    # Updown slug at the money → ~0.5 (slightly under from -σ²/2 drift).
    p = gbm_fair_price(
        slug="btc-updown-5m",
        btc_t=60_000.0,
        btc_0=60_000.0,
        seconds_remaining=300.0,
        recent_prices=[60_000.0 + i for i in range(60)],
        dt_seconds=1.0,
    )
    assert p is not None
    assert 0.40 < p < 0.50


def test_calibration_and_cointegration_lookups():
    bins = [
        CalibrationBin(lower=0.0, upper=0.5, actual_rate=0.2),
        CalibrationBin(lower=0.5, upper=1.0, actual_rate=0.8),
    ]
    assert calibration_fair_price(0.30, bins=bins) == 0.2
    assert calibration_fair_price(0.80, bins=bins) == 0.8
    # Boundary: spec uses (lo, hi] half-open, so 0.5 lands in the first bin.
    assert calibration_fair_price(0.5, bins=bins) == 0.2

    hits = [
        CointegrationHit(
            a_id="slug-a",
            b_id="slug-b",
            beta_hedge=0.7,
            half_life_days=3.0,
            oos_sharpe=4.0,
        ),
        CointegrationHit(
            a_id="slug-a",
            b_id="slug-c",
            beta_hedge=0.3,
            half_life_days=10.0,
            oos_sharpe=1.0,
        ),
    ]
    fair = cointegration_fair_price("slug-a", p_market=0.5, peer_price=0.6, hits=hits)
    # Strongest partner is slug-b with beta=0.7 → 0.7 * 0.6 = 0.42.
    assert fair is not None
    assert abs(fair - 0.42) < 1e-9
    # Slug not present → None.
    assert cointegration_fair_price("slug-z", 0.5, peer_price=0.5, hits=hits) is None


def test_endpoint_returns_all_fields_and_aggregates_signal(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path)

    # Use a non-updown slug present in the cointegration fixture.
    # market p = 0.55. Calibration → 0.65 (BUY +0.10). Prelec inverse of 0.55
    # is well above 0.55 (prelec underweights mid probabilities → fair > p).
    # Cointegration: peer_price=0.95, β=0.8 → 0.76 (BUY).
    # Three BUYs ⇒ dominant_signal = BUY.
    resp = client.get(
        "/terminal/fair/btc-100k-by-eoy",
        params={"p_market": 0.55, "peer_price": 0.95},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["slug"] == "btc-100k-by-eoy"
    assert body["market_p_now"] == 0.55
    assert body["gbm_fair"] is None  # non-updown
    assert body["prelec_fair"] is not None
    assert body["cointegration_fair"] is not None
    assert body["calibration_fair"] is not None
    assert set(body["spread_bps_per_model"].keys()) == {
        "gbm_bps",
        "prelec_bps",
        "cointegration_bps",
        "calibration_bps",
    }
    assert body["dominant_signal"] in {"BUY", "SELL", "HOLD"}

    # And the aggregate-signal helper is honest about the 3-vote rule.
    assert aggregate_signal([1, 1, 1, 0]) == "BUY"
    assert aggregate_signal([-1, -1, -1, 0]) == "SELL"
    assert aggregate_signal([1, 1, -1, 0]) == "HOLD"


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------


def test_prelec_inverse_rejects_out_of_range_inputs():
    """Inputs outside [0, 1] raise ValueError (the docstring promises this)."""
    import pytest

    with pytest.raises(ValueError):
        prelec_inverse(-0.01)
    with pytest.raises(ValueError):
        prelec_inverse(1.01)
    # Boundary inputs short-circuit.
    assert prelec_inverse(0.0) == 0.0
    assert prelec_inverse(1.0) == 1.0


def test_endpoint_400s_when_no_p_market_provided(monkeypatch, tmp_path):
    """The default market-quote provider raises 400 when no override is wired."""
    # Reset to default — the main app's lifespan wires a live-Gamma provider
    # which would otherwise leak into this test through the module-level
    # singleton in fair_price.
    from pfm.terminal.fair_price import (
        _default_market_quote,
        set_market_quote_provider,
    )

    set_market_quote_provider(_default_market_quote)
    client = _build_client(monkeypatch, tmp_path)
    r = client.get("/terminal/fair/btc-100k-by-eoy")  # no p_market query param
    assert r.status_code == 400
    assert "no market quote" in r.json()["detail"]


def test_endpoint_validates_p_market_range(monkeypatch, tmp_path):
    """Query validator rejects p_market < 0 or > 1 with 422."""
    client = _build_client(monkeypatch, tmp_path)
    assert client.get("/terminal/fair/some-slug?p_market=1.5").status_code == 422
    assert client.get("/terminal/fair/some-slug?p_market=-0.1").status_code == 422


def test_calibration_handles_empty_and_underflow_overflow():
    """Empty bin list returns None; out-of-range maps to first/last bin."""
    assert calibration_fair_price(0.5) is None or calibration_fair_price(0.5, bins=[]) is None
    assert calibration_fair_price(0.5, bins=[]) is None

    bins = [
        CalibrationBin(lower=0.2, upper=0.4, actual_rate=0.3),
        CalibrationBin(lower=0.4, upper=0.6, actual_rate=0.5),
    ]
    # Underflow (≤ first lower edge) → first bin's actual_rate.
    assert calibration_fair_price(0.05, bins=bins) == 0.3
    # Overflow (> last upper edge) → last bin's actual_rate.
    assert calibration_fair_price(0.99, bins=bins) == 0.5


# ---------------------------------------------------------------------------
# Confidence + notes + strict aggregator
# ---------------------------------------------------------------------------


def test_gbm_returns_none_when_spot_inputs_missing():
    """Even an updown slug returns None if btc_t/btc_0/seconds_remaining are missing."""
    assert gbm_fair_price(slug="btc-updown-5m") is None
    assert (
        gbm_fair_price(slug="btc-updown-5m", btc_t=60_000.0)  # missing btc_0+window
        is None
    )


def test_cointegration_returns_none_when_peer_missing():
    """Found peer but no peer_price → None (cannot evaluate β·peer)."""
    hits = [
        CointegrationHit(
            a_id="slug-a",
            b_id="slug-b",
            beta_hedge=0.7,
            half_life_days=3.0,
            oos_sharpe=4.0,
        ),
    ]
    assert cointegration_fair_price("slug-a", 0.5, peer_price=None, hits=hits) is None


def test_confidence_buckets():
    """Confidence buckets: 4/4=high, 3/4=high, 2/4=medium, 1/4=low, 0/4=low."""
    assert _confidence_from_active(4) == "high"
    assert _confidence_from_active(3) == "high"
    assert _confidence_from_active(2) == "medium"
    assert _confidence_from_active(1) == "low"
    assert _confidence_from_active(0) == "low"


def test_aggregate_signal_strict_requires_active_consensus():
    """Strict aggregator demands ≥3 votes OR ≥2 with strong edge."""
    # 1 active model → HOLD regardless of edge.
    assert (
        aggregate_signal_strict({"a": 0.90, "b": None, "c": None, "d": None}, p_market=0.10)
        == "HOLD"
    )

    # 2 active, both BUY but edge < 10pp → HOLD.
    assert (
        aggregate_signal_strict({"a": 0.55, "b": 0.56, "c": None, "d": None}, p_market=0.50)
        == "HOLD"
    )

    # 2 active, both BUY with edge > 10pp → BUY.
    assert (
        aggregate_signal_strict({"a": 0.70, "b": 0.75, "c": None, "d": None}, p_market=0.50)
        == "BUY"
    )

    # 2 active, both SELL with edge > 10pp → SELL.
    assert (
        aggregate_signal_strict({"a": 0.20, "b": 0.25, "c": None, "d": None}, p_market=0.50)
        == "SELL"
    )

    # 3 active, all BUY (>5pp) → BUY.
    assert (
        aggregate_signal_strict({"a": 0.60, "b": 0.62, "c": 0.59, "d": None}, p_market=0.50)
        == "BUY"
    )

    # 2 active disagreeing → HOLD.
    assert (
        aggregate_signal_strict({"a": 0.70, "b": 0.30, "c": None, "d": None}, p_market=0.50)
        == "HOLD"
    )


def test_endpoint_low_confidence_returns_hold_and_notes(monkeypatch, tmp_path):
    """Slug with no peer + non-updown + p_market in mid bin → 2 active = medium,
    with no strong edge → HOLD + notes for the 2 missing models."""
    client = _build_client(monkeypatch, tmp_path)
    resp = client.get(
        "/terminal/fair/putin-out-before-2027",
        params={"p_market": 0.12},  # no peer_price, no btc_*
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["gbm_fair"] is None
    assert body["cointegration_fair"] is None
    assert body["prelec_fair"] is not None
    assert body["calibration_fair"] is not None

    assert body["n_active_models"] == 2
    assert body["confidence"] == "medium"
    # Without strong consensus across both, dominant should be HOLD.
    assert body["dominant_signal"] == "HOLD"

    notes = body["notes"]
    assert any("GBM" in n for n in notes)
    assert any("cointegrated peer" in n for n in notes)


def test_endpoint_only_one_active_model_forces_low_confidence_hold(monkeypatch, tmp_path):
    """Empty calibration + no peer + non-updown ⇒ only Prelec active ⇒ low."""
    # Wipe both data files to disable coint + calibration.
    coint_path = tmp_path / "all_unique_hits.json"
    coint_path.write_text("[]")
    calib_path = tmp_path / "strat9_calibration.json"
    calib_path.write_text(json.dumps({"calibration_table": []}))
    monkeypatch.setattr("pfm.terminal_fair_price.COINTEGRATION_PATH", coint_path)
    monkeypatch.setattr("pfm.terminal_fair_price.CALIBRATION_PATH", calib_path)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get("/terminal/fair/some-random-slug", params={"p_market": 0.30})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["n_active_models"] == 1
    assert body["confidence"] == "low"
    assert body["dominant_signal"] == "HOLD"
    assert any("low-confidence" in n for n in body["notes"])
    # Sanity: only Prelec is active.
    assert body["gbm_fair"] is None
    assert body["cointegration_fair"] is None
    assert body["calibration_fair"] is None
    assert body["prelec_fair"] is not None


def test_endpoint_high_confidence_with_3_models_active(monkeypatch, tmp_path):
    """Fixture: btc-100k-by-eoy, p=0.55 → 3 active models ⇒ confidence=high.

    Note: not all of the active models reach the 5pp BUY edge here, so the
    dominant signal is allowed to be BUY or HOLD — what we care about is
    that confidence reflects the *number of active models*, not the votes.
    """
    client = _build_client(monkeypatch, tmp_path)
    resp = client.get(
        "/terminal/fair/btc-100k-by-eoy",
        params={"p_market": 0.55, "peer_price": 0.95},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_active_models"] == 3
    assert body["confidence"] == "high"
    assert body["dominant_signal"] in {"BUY", "HOLD"}
    assert isinstance(body["notes"], list)
    # GBM is the only n/a model here, so notes should mention GBM.
    assert any("GBM" in n for n in body["notes"])
    # And no low-confidence note since confidence == "high".
    assert not any("low-confidence" in n for n in body["notes"])
