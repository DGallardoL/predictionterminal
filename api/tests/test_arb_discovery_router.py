"""Router tests for the discovery pipeline endpoints (seams mocked, no network)."""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from pfm import strategies_arb_router as r


class _FakeStore:
    def __init__(self):
        self._items = [
            types.SimpleNamespace(
                to_dict=lambda: {
                    "arb_key": "KXFOO|foo-slug",
                    "kalshi_ticker": "KXFOO",
                    "poly_slug": "foo-slug",
                    "count": 5,
                    "first_seen": "2026-05-20T00:00:00Z",
                    "last_seen": "2026-05-21T00:00:00Z",
                    "max_profit_pct": 4.2,
                }
            )
        ]

    def confirmed(self, min_count=3):
        return [c for c in self._items if c.to_dict()["count"] >= min_count]

    def stats(self):
        return {"total_seen": 12, "n_confirmed": 1, "n_markets": 12, "oldest": None, "newest": None}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(r, "_discovery_store", lambda: _FakeStore())
    monkeypatch.setattr(
        r,
        "_run_discovery_step",
        lambda **kw: types.SimpleNamespace(
            as_dict=lambda: {
                "mode": kw.get("mode"),
                "n_kalshi": 1000,
                "n_poly": 100,
                "n_candidates": 7,
                "n_high": 2,
                "n_recorded": 0,
                "checkpoint": {"kalshi_cursor": "abc", "poly_offset": 300},
                "summary": {"by_tier": {"high": 2, "borderline": 5, "reject": 40}},
                "candidates": [
                    {
                        "kalshi_ticker": "KXFOO",
                        "poly_slug": "foo-slug",
                        "score": 0.83,
                        "tier": "high",
                    }
                ],
            }
        ),
    )
    # Override the admin gate with the SAME callable the router bound in Depends().
    from pfm.auth.dependencies import require_admin
    from pfm.main import app

    app.dependency_overrides[require_admin] = lambda: None
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def test_confirmed_endpoint(client):
    resp = client.get("/strategies/arb/discovery/confirmed?min_count=3")
    assert resp.status_code == 200
    d = resp.json()
    assert d["count"] == 1
    assert d["items"][0]["arb_key"] == "KXFOO|foo-slug"
    assert d["stats"]["total_seen"] == 12


def test_confirmed_min_count_filters(client):
    resp = client.get("/strategies/arb/discovery/confirmed?min_count=100")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0  # the one item has count 5 < 100


def test_status_endpoint(client, monkeypatch):
    monkeypatch.setattr(
        "pfm.arb.market_crawler.load_checkpoint",
        lambda path: types.SimpleNamespace(
            kalshi_cursor="abc", poly_offset=300, last_seen_poly_start_iso="2026-05-21T00:00:00Z"
        ),
    )
    resp = client.get("/strategies/arb/discovery/status")
    assert resp.status_code == 200
    d = resp.json()
    assert d["checkpoint"]["poly_offset"] == 300
    assert d["store"]["total_seen"] == 12


def test_discovery_step_new(client):
    resp = client.post("/strategies/arb/discovery/step?mode=new&max_pages=2")
    assert resp.status_code == 200
    d = resp.json()
    assert d["mode"] == "new"
    assert d["n_candidates"] == 7
    assert d["summary"]["by_tier"]["high"] == 2


def test_discovery_step_sweep(client):
    resp = client.post("/strategies/arb/discovery/step?mode=sweep&max_pages=5")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "sweep"


def test_discovery_step_bad_mode_422(client):
    assert client.post("/strategies/arb/discovery/step?mode=bogus").status_code == 422


def test_discovery_step_upstream_error_502(client, monkeypatch):
    def _boom(**kw):
        raise RuntimeError("kalshi down")

    monkeypatch.setattr(r, "_run_discovery_step", _boom)
    resp = client.post("/strategies/arb/discovery/step?mode=new&max_pages=4")
    assert resp.status_code == 502
