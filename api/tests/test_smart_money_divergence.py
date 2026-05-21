"""Tests for ``pfm.smart_money_divergence``.

The strength function and detection logic are deterministic given inputs,
so we test the pure helpers with synthetic flow numbers and assert the
expected divergence semantics.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.smart_money_divergence import (
    _strength,
    detect_divergence,
    router,
    scan_all_divergences,
)

# --- pure-function tests ----------------------------------------------------


def test_strength_zero_when_flows_align_same_sign() -> None:
    """Same-sign flows should produce a low (consonance) strength <= 0.3."""
    s = _strength(whale_flow=120_000.0, equity_flow=80_000.0)
    assert s <= 0.30


def test_strength_high_when_flows_disagree_with_magnitude() -> None:
    """Opposite-sign flows of comparable magnitude → strong divergence."""
    s = _strength(whale_flow=200_000.0, equity_flow=-180_000.0)
    assert s > 0.6


def test_strength_low_when_one_side_tiny() -> None:
    """If equity flow is 100x smaller than whale flow, balance damps."""
    s = _strength(whale_flow=200_000.0, equity_flow=-2_000.0)
    assert s < 0.20


def test_strength_zero_when_both_flows_zero() -> None:
    assert _strength(0.0, 0.0) == 0.0


def test_detect_divergence_returns_required_fields() -> None:
    """Output dict must include all schema fields."""
    out = detect_divergence("nvda-eps-beat-q1", "NVDA", lookback_hours=24)
    expected = {
        "slug",
        "ticker_proxy",
        "lookback_hours",
        "whale_flow_pm",
        "equity_flow",
        "divergence_strength",
        "is_diverging",
        "historical_lead_winrate",
        "suggested_trade",
    }
    assert expected <= set(out.keys())
    assert 0.0 <= out["historical_lead_winrate"] <= 1.0
    assert 0.0 <= out["divergence_strength"] <= 1.0


def test_detect_divergence_deterministic() -> None:
    a = detect_divergence("nvda-eps-beat-q1", "NVDA", lookback_hours=24)
    b = detect_divergence("nvda-eps-beat-q1", "NVDA", lookback_hours=24)
    assert a == b


def test_detect_divergence_aligned_flows_synthetic() -> None:
    """Force aligned flows by using the strength helper directly.

    We can't easily inject flows into ``detect_divergence`` (it computes
    them from seeds), so this test asserts the *strength function*'s
    behaviour for the aligned case, which is the contract the detector
    relies on.
    """
    # Large positive flows on both sides → not diverging.
    s = _strength(150_000.0, 150_000.0)
    assert s < 0.30


def test_detect_divergence_opposite_signs_synthetic() -> None:
    """Whale buys $100k, equity sells $80k → strong divergence."""
    s = _strength(100_000.0, -80_000.0)
    assert s > 0.40


def test_scan_all_divergences_returns_at_most_10() -> None:
    rows = scan_all_divergences(min_strength=0.0)
    assert len(rows) <= 10


def test_scan_all_divergences_sorted_by_strength_desc() -> None:
    rows = scan_all_divergences(min_strength=0.0)
    if len(rows) >= 2:
        strengths = [r["divergence_strength"] for r in rows]
        assert strengths == sorted(strengths, reverse=True)


def test_scan_all_divergences_min_strength_filter() -> None:
    threshold = 0.40
    rows = scan_all_divergences(min_strength=threshold)
    for r in rows:
        assert r["divergence_strength"] >= threshold


# --- HTTP integration -------------------------------------------------------


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_get_smart_money_divergences_endpoint() -> None:
    c = _client()
    r = c.get("/divergence/smart-money?min_strength=0.0")
    assert r.status_code == 200
    body = r.json()
    assert body["min_strength"] == 0.0
    assert isinstance(body["results"], list)


def test_get_divergence_for_known_slug() -> None:
    c = _client()
    r = c.get("/divergence/nvda-eps-beat-q1")
    assert r.status_code == 200
    body = r.json()
    assert body["slug"] == "nvda-eps-beat-q1"
    assert body["ticker_proxy"] == "NVDA"


def test_get_divergence_unknown_slug_without_proxy_returns_404() -> None:
    c = _client()
    r = c.get("/divergence/some-totally-unknown-slug")
    assert r.status_code == 404


def test_get_divergence_unknown_slug_with_proxy_works() -> None:
    c = _client()
    r = c.get("/divergence/some-custom-slug?ticker_proxy=AAPL")
    assert r.status_code == 200
    body = r.json()
    assert body["slug"] == "some-custom-slug"
    assert body["ticker_proxy"] == "AAPL"
