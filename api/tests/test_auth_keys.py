"""Tests for pfm.auth: storage CRUD + /auth/keys router."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.auth.dependencies import auth_enabled
from pfm.auth.models import APIKey
from pfm.auth.router import router as auth_router
from pfm.auth.storage import APIKeyStore, get_api_key_store

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def store() -> APIKeyStore:
    return APIKeyStore(":memory:")


@pytest.fixture
def admin_token(monkeypatch: pytest.MonkeyPatch) -> str:
    tok = "test-admin-token-xyz"
    monkeypatch.setenv("PFM_ADMIN_TOKEN", tok)
    monkeypatch.setenv("PFM_AUTH_ENABLED", "1")
    return tok


@pytest.fixture
def client(store: APIKeyStore) -> TestClient:
    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_api_key_store] = lambda: store
    return TestClient(app)


# ---------------------------------------------------------------- storage


def test_storage_save_and_get_key(store: APIKeyStore) -> None:
    k = APIKey.new(user_id="alice", tier="pro")
    store.save_key(k)
    loaded = store.get_key(k.key)
    assert loaded is not None
    assert loaded.user_id == "alice"
    assert loaded.tier == "pro"
    assert loaded.rate_limit_per_minute == 300
    assert loaded.daily_quota == 10_000
    assert loaded.enabled is True


def test_storage_list_keys_filters_by_user(store: APIKeyStore) -> None:
    k1 = APIKey.new(user_id="alice", tier="free")
    k2 = APIKey.new(user_id="bob", tier="quant")
    store.save_key(k1)
    store.save_key(k2)
    alices = store.list_keys(user_id="alice")
    assert len(alices) == 1
    assert alices[0].user_id == "alice"


def test_storage_revoke_disables_but_keeps_row(store: APIKeyStore) -> None:
    k = APIKey.new(user_id="alice", tier="free")
    store.save_key(k)
    assert store.revoke_key(k.key) is True
    loaded = store.get_key(k.key)
    assert loaded is not None
    assert loaded.enabled is False


def test_storage_revoke_unknown_returns_false(store: APIKeyStore) -> None:
    assert store.revoke_key("sk_pfm_nope") is False


def test_storage_expired_key_is_gone(store: APIKeyStore) -> None:
    k = APIKey.new(user_id="demo", tier="free")
    store.save_key(k, expires_at=time.time() - 5)
    assert store.get_key(k.key) is None


def test_storage_update_tier_changes_limits(store: APIKeyStore) -> None:
    k = APIKey.new(user_id="alice", tier="free")
    store.save_key(k)
    upgraded = store.update_tier(k.key, "quant")
    assert upgraded is not None
    assert upgraded.tier == "quant"
    assert upgraded.rate_limit_per_minute == 3_000
    assert upgraded.daily_quota == 100_000


def test_storage_increment_returns_running_count(store: APIKeyStore) -> None:
    k = APIKey.new(user_id="alice", tier="free")
    store.save_key(k)
    a, b = store.increment(k.key)
    assert a == 1 and b == 1
    a2, b2 = store.increment(k.key)
    assert a2 == 2 and b2 == 2


# ---------------------------------------------------------------- router


def test_create_key_requires_admin_token_when_set(client: TestClient, admin_token: str) -> None:
    # No header → 403
    r1 = client.post("/auth/keys", json={"user_id": "alice", "tier": "pro"})
    assert r1.status_code == 403

    # Wrong header → 403
    r2 = client.post(
        "/auth/keys",
        json={"user_id": "alice", "tier": "pro"},
        headers={"X-Admin-Token": "wrong"},
    )
    assert r2.status_code == 403

    # Correct → 200
    r3 = client.post(
        "/auth/keys",
        json={"user_id": "alice", "tier": "pro"},
        headers={"X-Admin-Token": admin_token},
    )
    assert r3.status_code == 200, r3.text
    body = r3.json()
    assert body["key"].startswith("sk_pfm_")
    assert body["tier"] == "pro"
    assert body["rate_limit_per_minute"] == 300


def test_create_key_disabled_when_admin_token_unset(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)
    r = client.post(
        "/auth/keys",
        json={"user_id": "alice", "tier": "pro"},
        headers={"X-Admin-Token": "anything"},
    )
    assert r.status_code == 403


def test_get_my_key_returns_masked_key(
    client: TestClient, store: APIKeyStore, admin_token: str
) -> None:
    k = APIKey.new(user_id="alice", tier="quant")
    store.save_key(k)
    r = client.get("/auth/keys/me", headers={"Authorization": f"Bearer {k.key}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_id"] == "alice"
    assert body["tier"] == "quant"
    assert "…" in body["key_masked"]
    # The plaintext secret must NOT round-trip.
    assert k.key not in body["key_masked"]


def test_get_my_key_without_auth_when_enabled_returns_401(
    client: TestClient, admin_token: str
) -> None:
    r = client.get("/auth/keys/me")
    assert r.status_code == 401


def test_get_my_usage_counts_track_increments(
    client: TestClient, store: APIKeyStore, admin_token: str
) -> None:
    k = APIKey.new(user_id="alice", tier="pro")
    store.save_key(k)
    store.increment(k.key)
    store.increment(k.key)
    r = client.get("/auth/keys/me/usage", headers={"Authorization": f"Bearer {k.key}"})
    assert r.status_code == 200
    body = r.json()
    assert body["requests_today"] == 2
    assert body["requests_this_minute"] == 2
    assert body["minute_remaining"] == 300 - 2
    assert body["daily_remaining"] == 10_000 - 2


def test_revoke_key_disables_it(client: TestClient, store: APIKeyStore, admin_token: str) -> None:
    k = APIKey.new(user_id="alice", tier="free")
    store.save_key(k)
    r = client.delete(f"/auth/keys/{k.key}", headers={"X-Admin-Token": admin_token})
    assert r.status_code == 200
    assert r.json()["revoked"] is True
    # Now it should fail to authenticate.
    r2 = client.get("/auth/keys/me", headers={"Authorization": f"Bearer {k.key}"})
    assert r2.status_code == 401


def test_revoke_unknown_key_returns_404(client: TestClient, admin_token: str) -> None:
    r = client.delete(
        "/auth/keys/sk_pfm_doesnotexist",
        headers={"X-Admin-Token": admin_token},
    )
    assert r.status_code == 404


def test_revoke_rejects_non_pfm_key(client: TestClient, admin_token: str) -> None:
    r = client.delete("/auth/keys/not-a-key", headers={"X-Admin-Token": admin_token})
    assert r.status_code == 400


def test_demo_key_is_open_and_short_lived(
    client: TestClient, admin_token: str, store: APIKeyStore
) -> None:
    r = client.post("/auth/demo-key")
    assert r.status_code == 200
    body = r.json()
    assert body["key"].startswith("sk_pfm_")
    assert body["tier"] == "free"
    # Demo keys are immediately usable (auth_enabled=1 path).
    r2 = client.get("/auth/keys/me", headers={"Authorization": f"Bearer {body['key']}"})
    assert r2.status_code == 200


def test_usage_dashboard_requires_admin(
    client: TestClient, store: APIKeyStore, admin_token: str
) -> None:
    # Seed some traffic.
    k = APIKey.new(user_id="alice", tier="pro")
    store.save_key(k)
    store.increment(k.key, endpoint="/strategies/optimize")
    store.increment(k.key, endpoint="/strategies/optimize")

    r1 = client.get("/auth/usage/dashboard")
    assert r1.status_code == 403

    r2 = client.get("/auth/usage/dashboard", headers={"X-Admin-Token": admin_token})
    assert r2.status_code == 200
    body = r2.json()
    assert "total_requests_today" in body
    assert "by_tier" in body
    assert "top_endpoints" in body
    assert body["total_requests_today"] >= 2


def test_auth_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PFM_AUTH_ENABLED", raising=False)
    assert auth_enabled() is False


def test_auth_disabled_lets_unauthenticated_calls_through(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When auth is OFF, /auth/keys/me yields the synthetic system key."""
    monkeypatch.delenv("PFM_AUTH_ENABLED", raising=False)
    r = client.get("/auth/keys/me")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == "system"
    assert body["tier"] == "enterprise"
