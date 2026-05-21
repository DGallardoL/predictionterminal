"""Unit tests for the watchlist + alerts backend.

Storage is redirected at a tmp directory and the two HTTP-fetch seams are
monkeypatched, so these tests do zero real network I/O and zero pollution
of ``/tmp/watchlists``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_watchlist as tw
from pfm.terminal_watchlist import compute_z_score, is_alert_triggered, router

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A fresh FastAPI app + watchlist dir redirected to ``tmp_path``."""
    monkeypatch.setattr(tw, "WATCHLIST_DIR", tmp_path / "watchlists")
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def patched_fetchers(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, object]]:
    """Stub the two external-IO seams with an in-memory price/history table.

    Tests mutate the returned dict to control what the endpoints "see".
    """
    state: dict[str, dict[str, object]] = {
        "prices": {},  # slug -> current_p
        "history": {},  # slug -> [p_t, ...]
    }

    def _fake_price(slug: str, _client: object) -> float | None:
        return state["prices"].get(slug)  # type: ignore[return-value]

    def _fake_history(slug: str, _client: object) -> list[float]:
        return list(state["history"].get(slug, []))  # type: ignore[arg-type]

    monkeypatch.setattr(tw, "_fetch_current_price", _fake_price)
    monkeypatch.setattr(tw, "_fetch_price_history", _fake_history)
    return state


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


def test_compute_z_score_and_alert_logic() -> None:
    """Pure-function checks: z-score, alert eval, edge cases."""
    # Constant history → undefined z-score (zero variance).
    assert compute_z_score(0.6, [0.5, 0.5, 0.5, 0.5]) is None
    # Empty / too-short → undefined.
    assert compute_z_score(0.6, []) is None
    assert compute_z_score(0.6, [0.5]) is None

    # A spike well above the recent mean should produce a large positive z.
    history = [0.50] * 10 + [0.51, 0.49, 0.50, 0.50, 0.50]
    z = compute_z_score(0.95, history)
    assert z is not None and z > 5.0

    # Symmetry: a spike below produces a large negative z.
    z_low = compute_z_score(0.05, history)
    assert z_low is not None and z_low < -5.0

    # Alert evaluation
    assert is_alert_triggered(2.5, 2.0) is True
    assert is_alert_triggered(-2.5, 2.0) is True  # absolute threshold
    assert is_alert_triggered(1.5, 2.0) is False
    assert is_alert_triggered(None, 2.0) is False  # missing z
    assert is_alert_triggered(3.0, None) is False  # no threshold set


def test_add_is_idempotent_and_persists_to_disk(
    client: TestClient, patched_fetchers: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    """POSTing the same slug twice does not duplicate; alert_z updates in place."""
    # First add — new entry.
    r = client.post(
        "/terminal/watchlist",
        json={"user_id": "default", "slug": "btc-100k", "alert_z": 2.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"user_id": "default", "slug": "btc-100k", "alert_z": 2.0, "added": True}

    # On-disk file exists with one row.
    storage = tmp_path / "watchlists" / "default.json"
    assert storage.exists()
    rows = json.loads(storage.read_text())
    assert rows == [{"slug": "btc-100k", "alert_z": 2.0}]

    # Second add — same slug, different threshold → idempotent update.
    r2 = client.post(
        "/terminal/watchlist",
        json={"user_id": "default", "slug": "btc-100k", "alert_z": 1.5},
    )
    assert r2.status_code == 200
    assert r2.json()["added"] is False

    rows2 = json.loads(storage.read_text())
    assert rows2 == [{"slug": "btc-100k", "alert_z": 1.5}]
    # Critically, no duplicate.
    assert len([r for r in rows2 if r["slug"] == "btc-100k"]) == 1

    # Default user_id behaviour: the request body's default applies if omitted.
    r3 = client.post("/terminal/watchlist", json={"slug": "fed-cut"})
    assert r3.status_code == 200, r3.text
    assert r3.json()["user_id"] == "default"


def test_list_returns_current_price_and_z_score(
    client: TestClient, patched_fetchers: dict[str, dict[str, object]]
) -> None:
    """GET /terminal/watchlist/{user} enriches entries with live price + z-score."""
    # Seed the stubbed fetchers.
    patched_fetchers["prices"]["btc-100k"] = 0.95
    patched_fetchers["history"]["btc-100k"] = [0.50] * 14 + [0.51]  # mean≈0.50, std≈tiny
    patched_fetchers["prices"]["fed-cut"] = 0.60
    patched_fetchers["history"]["fed-cut"] = [0.55, 0.58, 0.60, 0.62, 0.59, 0.61, 0.60]

    # Add two entries, only one with a threshold.
    client.post(
        "/terminal/watchlist",
        json={"user_id": "alice", "slug": "btc-100k", "alert_z": 2.0},
    )
    client.post("/terminal/watchlist", json={"user_id": "alice", "slug": "fed-cut"})

    r = client.get("/terminal/watchlist/alice")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_id"] == "alice"
    assert body["n_items"] == 2

    by_slug = {it["slug"]: it for it in body["items"]}
    btc = by_slug["btc-100k"]
    assert btc["current_p"] == pytest.approx(0.95)
    assert btc["z_score"] is not None and btc["z_score"] > 5.0
    assert btc["alert_triggered"] is True
    assert btc["alert_z"] == 2.0

    fed = by_slug["fed-cut"]
    assert fed["current_p"] == pytest.approx(0.60)
    assert fed["z_score"] is not None
    # No threshold set → never triggered.
    assert fed["alert_triggered"] is False
    assert fed["alert_z"] is None

    # Empty user → empty list, no error.
    r_empty = client.get("/terminal/watchlist/nobody")
    assert r_empty.status_code == 200
    assert r_empty.json() == {"user_id": "nobody", "n_items": 0, "items": []}


def test_alerts_endpoint_filters_to_triggered_and_delete_works(
    client: TestClient, patched_fetchers: dict[str, dict[str, object]]
) -> None:
    """``/alerts`` returns only breached rows; DELETE removes a slug from the list."""
    # One slug spikes above 2σ, one stays calm.
    patched_fetchers["prices"]["spike"] = 0.90
    patched_fetchers["history"]["spike"] = [0.50] * 14 + [0.51]
    # "calm" history has realistic noise; current price sits ~0.5σ off the mean.
    patched_fetchers["prices"]["calm"] = 0.55
    patched_fetchers["history"]["calm"] = [
        0.50,
        0.55,
        0.45,
        0.52,
        0.48,
        0.53,
        0.47,
        0.51,
        0.49,
        0.54,
        0.46,
        0.50,
        0.52,
        0.48,
        0.51,
    ]

    client.post("/terminal/watchlist", json={"user_id": "bob", "slug": "spike", "alert_z": 2.0})
    client.post("/terminal/watchlist", json={"user_id": "bob", "slug": "calm", "alert_z": 2.0})

    r = client.get("/terminal/watchlist/bob/alerts")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_id"] == "bob"
    assert body["n_alerts"] == 1
    assert body["alerts"][0]["slug"] == "spike"
    assert body["alerts"][0]["alert_triggered"] is True

    # Deleting the spike removes it from both list and alerts.
    rdel = client.delete("/terminal/watchlist/bob/spike")
    assert rdel.status_code == 200
    assert rdel.json() == {"user_id": "bob", "slug": "spike", "removed": True}

    r_alerts = client.get("/terminal/watchlist/bob/alerts")
    assert r_alerts.json() == {"user_id": "bob", "n_alerts": 0, "alerts": []}

    r_list = client.get("/terminal/watchlist/bob")
    slugs = [it["slug"] for it in r_list.json()["items"]]
    assert slugs == ["calm"]

    # Deleting something that isn't there → removed=False, no error.
    rdel2 = client.delete("/terminal/watchlist/bob/spike")
    assert rdel2.status_code == 200
    assert rdel2.json()["removed"] is False
