"""Tests for the per-IP daily cap on ``POST /auth/demo-key``."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.auth.router import DEMO_KEY_DAILY_CAP_PER_IP
from pfm.auth.router import router as auth_router
from pfm.auth.storage import APIKeyStore, get_api_key_store


@pytest.fixture
def store() -> APIKeyStore:
    return APIKeyStore(":memory:")


@pytest.fixture
def client(store: APIKeyStore) -> TestClient:
    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_api_key_store] = lambda: store
    return TestClient(app)


def test_first_5_demo_keys_succeed(client: TestClient) -> None:
    """5 demo-key requests from same IP → all 200."""
    for i in range(DEMO_KEY_DAILY_CAP_PER_IP):
        r = client.post("/auth/demo-key", headers={"X-Forwarded-For": "1.2.3.4"})
        assert r.status_code == 200, (i, r.text)
        body = r.json()
        assert body["key"].startswith("sk_pfm_")
        assert body["demo_keys_issued_today"] == i + 1
        assert body["demo_keys_remaining_today"] == DEMO_KEY_DAILY_CAP_PER_IP - (i + 1)


def test_sixth_demo_key_returns_429(client: TestClient) -> None:
    """6th request from same IP same day → 429 with Retry-After."""
    for _ in range(DEMO_KEY_DAILY_CAP_PER_IP):
        r = client.post("/auth/demo-key", headers={"X-Forwarded-For": "9.9.9.9"})
        assert r.status_code == 200
    over = client.post("/auth/demo-key", headers={"X-Forwarded-For": "9.9.9.9"})
    assert over.status_code == 429
    assert "Retry-After" in over.headers
    assert int(over.headers["Retry-After"]) >= 1
    assert "cap reached" in over.json()["detail"].lower()


def test_two_distinct_ips_independent(client: TestClient) -> None:
    """5 from IP A + 5 from IP B → all succeed (independent counters)."""
    for _ in range(DEMO_KEY_DAILY_CAP_PER_IP):
        r = client.post("/auth/demo-key", headers={"X-Forwarded-For": "10.0.0.1"})
        assert r.status_code == 200
    for _ in range(DEMO_KEY_DAILY_CAP_PER_IP):
        r = client.post("/auth/demo-key", headers={"X-Forwarded-For": "10.0.0.2"})
        assert r.status_code == 200
    # Each IP is now capped.
    over_a = client.post("/auth/demo-key", headers={"X-Forwarded-For": "10.0.0.1"})
    over_b = client.post("/auth/demo-key", headers={"X-Forwarded-For": "10.0.0.2"})
    assert over_a.status_code == 429
    assert over_b.status_code == 429


def test_day_boundary_resets_counter(store: APIKeyStore) -> None:
    """Manual day-boundary roll: counter resets when ``day`` key changes."""
    # Saturate IP X for day=20260101.
    for _ in range(DEMO_KEY_DAILY_CAP_PER_IP):
        store.increment_demo_quota("ipX", day="20260101")
    assert store.get_demo_quota_count("ipX", day="20260101") == (DEMO_KEY_DAILY_CAP_PER_IP)
    # Next day → fresh counter.
    assert store.get_demo_quota_count("ipX", day="20260102") == 0
    new_count = store.increment_demo_quota("ipX", day="20260102")
    assert new_count == 1


def test_x_forwarded_for_first_hop_used(client: TestClient) -> None:
    """Multi-hop XFF: first IP wins."""
    for _ in range(DEMO_KEY_DAILY_CAP_PER_IP):
        r = client.post(
            "/auth/demo-key",
            headers={"X-Forwarded-For": "1.1.1.1, 5.5.5.5, 6.6.6.6"},
        )
        assert r.status_code == 200
    # Same first-hop IP → capped.
    over = client.post(
        "/auth/demo-key",
        headers={"X-Forwarded-For": "1.1.1.1, 9.9.9.9"},
    )
    assert over.status_code == 429
    # Different first-hop IP → fresh quota.
    diff = client.post(
        "/auth/demo-key",
        headers={"X-Forwarded-For": "2.2.2.2, 9.9.9.9"},
    )
    assert diff.status_code == 200
