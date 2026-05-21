"""Tests for the PM-VIX composite index module.

Mounts ``router`` on a throw-away FastAPI app so we don't touch
``main.py``. The Polymarket Gamma fetch is short-circuited with the
``overrides`` parameter on :func:`compute_pm_vix` for the unit tests,
and patched module-wide for the endpoint integration test.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import pm_vix
from pfm.cache_utils import get_cache
from pfm.pm_vix import (
    BUCKET_SLUGS,
    BUCKET_WEIGHTS,
    compute_pm_vix,
    pm_vix_history,
    router,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    get_cache("pm_vix").clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _market(prob: float, vol: float = 10_000.0) -> dict[str, Any]:
    """Build a Gamma-shaped market dict with mid ≈ ``prob``."""
    return {
        "bestBid": max(0.0, prob - 0.005),
        "bestAsk": min(1.0, prob + 0.005),
        "lastTradePrice": prob,
        "volume24hr": vol,
    }


def _all_overrides(prob: float) -> dict[str, dict[str, Any]]:
    """Return a slug → market_dict map covering every bucket."""
    out: dict[str, dict[str, Any]] = {}
    for slugs in BUCKET_SLUGS.values():
        for s in slugs:
            out[s] = _market(prob)
    return out


# ---------------------------------------------------------------------------
# compute_pm_vix
# ---------------------------------------------------------------------------


def test_compute_pm_vix_returns_score_in_0_100() -> None:
    overrides = _all_overrides(0.30)
    snap = compute_pm_vix(overrides=overrides, http=MagicMock())
    assert 0.0 <= snap["score"] <= 100.0
    assert snap["regime"] in {"RISK_ON", "NEUTRAL", "RISK_OFF"}
    assert isinstance(snap["history_30d"], list)
    assert len(snap["history_30d"]) == 30


def test_components_sum_matches_score() -> None:
    """Headline score must equal the sum of bucket contributions (rounded)."""
    overrides = _all_overrides(0.45)
    snap = compute_pm_vix(overrides=overrides, http=MagicMock())
    summed = sum(c["contribution"] for c in snap["components"])
    assert abs(snap["score"] - round(summed, 3)) < 0.01

    # Every bucket from BUCKET_WEIGHTS must appear once in components.
    bucket_names = {c["bucket"] for c in snap["components"]}
    assert bucket_names == set(BUCKET_WEIGHTS.keys())

    # Each component's weight must match BUCKET_WEIGHTS.
    for c in snap["components"]:
        assert abs(c["weight"] - BUCKET_WEIGHTS[c["bucket"]]) < 1e-6


def test_regime_classification_risk_on_when_probs_low() -> None:
    """All probs at 0.02 → headline score < 25 → RISK_ON."""
    snap = compute_pm_vix(overrides=_all_overrides(0.02), http=MagicMock())
    assert snap["regime"] == "RISK_ON"
    assert snap["score"] < 25


def test_regime_classification_risk_off_when_probs_high() -> None:
    """All probs at 0.85 → headline score >= 60 → RISK_OFF."""
    snap = compute_pm_vix(overrides=_all_overrides(0.85), http=MagicMock())
    assert snap["regime"] == "RISK_OFF"
    assert snap["score"] >= 60


def test_compute_pm_vix_handles_missing_markets() -> None:
    """Empty overrides → every bucket reports n_used=0 and score=0 (no NaN)."""
    snap = compute_pm_vix(overrides={}, http=MagicMock())
    assert snap["score"] >= 0.0
    for c in snap["components"]:
        assert c["n_used"] == 0


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_returns_n_points() -> None:
    pts = pm_vix_history(days=14)
    assert len(pts) == 14
    for p in pts:
        assert 0.0 <= p["score"] <= 100.0
        assert "date" in p


def test_history_is_deterministic() -> None:
    a = pm_vix_history(days=10)
    b = pm_vix_history(days=10)
    assert [p["score"] for p in a] == [p["score"] for p in b]


def test_history_anchor_pins_last_point() -> None:
    pts = pm_vix_history(days=5, anchor_score=42.0)
    assert pts[-1]["score"] == 42.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_get_pm_vix_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch fetch_gamma_market so /indices/pm-vix returns a sane snapshot."""

    def _fake_fetch(http, gamma_url, slug, **_kwargs):
        return _market(0.40)

    monkeypatch.setattr(pm_vix, "fetch_gamma_market", _fake_fetch)

    r = client.get("/indices/pm-vix")
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["score"] <= 100.0
    assert body["regime"] in {"RISK_ON", "NEUTRAL", "RISK_OFF"}
    assert len(body["history_30d"]) == 30
    assert body["components"], "expected non-empty components"


def test_get_pm_vix_components_endpoint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pm_vix, "fetch_gamma_market", lambda *a, **k: _market(0.50))
    r = client.get("/indices/pm-vix/components")
    assert r.status_code == 200, r.text
    body = r.json()
    assert {c["bucket"] for c in body["components"]} == set(BUCKET_WEIGHTS.keys())


def test_get_pm_vix_history_endpoint(client: TestClient) -> None:
    r = client.get("/indices/pm-vix/history", params={"days": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 7
    assert len(body["points"]) == 7


def test_get_pm_vix_invalid_as_of(client: TestClient) -> None:
    r = client.get("/indices/pm-vix", params={"as_of": "not-a-date"})
    assert r.status_code == 400
