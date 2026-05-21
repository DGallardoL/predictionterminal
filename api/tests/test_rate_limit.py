"""Tests for pfm.auth.rate_limiter — middleware + token-bucket logic."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.auth.models import APIKey
from pfm.auth.rate_limiter import RateLimitMiddleware, check_and_increment
from pfm.auth.storage import APIKeyStore, get_api_key_store

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def store() -> APIKeyStore:
    return APIKeyStore(":memory:")


@pytest.fixture
def app(store: APIKeyStore, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("PFM_AUTH_ENABLED", "1")
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, store_factory=lambda: store)

    @app.get("/health")
    def _health() -> dict:
        return {"ok": True}

    @app.get("/embed/foo")
    def _embed() -> dict:
        return {"ok": True}

    @app.get("/auth/keys/me")
    def _auth() -> dict:
        return {"ok": True}

    @app.get("/api/data")
    def _data() -> dict:
        return {"ok": True}

    app.dependency_overrides[get_api_key_store] = lambda: store
    return app


# ---------------------------------------------------------------- bucket math


def test_check_and_increment_allows_under_quota(store: APIKeyStore) -> None:
    ok, info = check_and_increment(
        "k1", "free", rate_limit_per_minute=5, daily_quota=100, store=store
    )
    assert ok is True
    assert info["minute_count"] == 1
    assert info["minute_remaining"] == 4
    assert info["day_remaining"] == 99
    assert info["retry_after"] == 0


def test_check_and_increment_blocks_at_minute_limit(store: APIKeyStore) -> None:
    for _ in range(5):
        ok, _ = check_and_increment(
            "k1", "free", rate_limit_per_minute=5, daily_quota=100, store=store
        )
        assert ok is True
    ok, info = check_and_increment(
        "k1", "free", rate_limit_per_minute=5, daily_quota=100, store=store
    )
    assert ok is False
    assert info["retry_after"] >= 1
    assert info["reset_at"] > 0


def test_check_and_increment_blocks_at_daily_quota(store: APIKeyStore) -> None:
    for _ in range(3):
        check_and_increment("k2", "free", rate_limit_per_minute=10_000, daily_quota=3, store=store)
    ok, info = check_and_increment(
        "k2", "free", rate_limit_per_minute=10_000, daily_quota=3, store=store
    )
    assert ok is False
    assert info["day_remaining"] == 0


def test_unlimited_daily_quota_never_trips(store: APIKeyStore) -> None:
    for _ in range(50):
        ok, info = check_and_increment(
            "ent",
            "enterprise",
            rate_limit_per_minute=10_000,
            daily_quota=0,
            store=store,
        )
        assert ok is True
        assert info["day_remaining"] == -1


# ---------------------------------------------------------------- middleware


def test_health_bypasses_limiter(app: FastAPI, store: APIKeyStore) -> None:
    client = TestClient(app)
    for _ in range(50):
        r = client.get("/health")
        assert r.status_code == 200
    # No counters were touched for /health.
    minute, day = store.get_counts("anon:testclient", endpoint="/health")
    assert minute == 0
    assert day == 0


def test_embed_bypasses_limiter(app: FastAPI) -> None:
    client = TestClient(app)
    for _ in range(50):
        r = client.get("/embed/foo")
        assert r.status_code == 200


def test_auth_paths_respect_rate_limit(app: FastAPI) -> None:
    """``/auth/*`` is NO LONGER bypassed: anon clients hit 10/min on it too.

    Prevents demo-key abuse via spam against ``/auth/demo-key``.
    """
    client = TestClient(app)
    # First 10 anon requests: 200 (anon = 10/min).
    for _ in range(10):
        r = client.get("/auth/keys/me")
        assert r.status_code == 200
    # 11th request → 429 (rate-limited).
    r = client.get("/auth/keys/me")
    assert r.status_code == 429


def test_anonymous_free_tier_blocks_at_eleventh(app: FastAPI) -> None:
    """Anon = 10/min. 11th request → 429."""
    client = TestClient(app)
    for i in range(10):
        r = client.get("/api/data")
        assert r.status_code == 200, f"req {i + 1} failed: {r.text}"
        assert r.headers["X-RateLimit-Tier"] == "free"
    r = client.get("/api/data")
    assert r.status_code == 429
    body = r.json()
    assert body["detail"] == "Rate limit exceeded"
    assert body["retry_after"] >= 1
    assert "Retry-After" in r.headers


def test_pro_key_does_not_block_at_eleventh(app: FastAPI, store: APIKeyStore) -> None:
    pro = APIKey.new(user_id="pro_user", tier="pro")
    store.save_key(pro)
    client = TestClient(app)
    for _ in range(15):
        r = client.get("/api/data", headers={"Authorization": f"Bearer {pro.key}"})
        assert r.status_code == 200
        assert r.headers["X-RateLimit-Tier"] == "pro"


def test_quota_exhaustion_response_headers(app: FastAPI, store: APIKeyStore) -> None:
    """Headers must announce remaining + reset time."""
    client = TestClient(app)
    r = client.get("/api/data")
    assert r.headers["X-RateLimit-Tier"] == "free"
    assert int(r.headers["X-RateLimit-Remaining"]) == 9
    assert int(r.headers["X-RateLimit-Reset"]) > 0


def test_disabled_key_falls_back_to_anonymous(app: FastAPI, store: APIKeyStore) -> None:
    k = APIKey.new(user_id="banned", tier="enterprise")
    store.save_key(k)
    store.revoke_key(k.key)
    client = TestClient(app)
    r = client.get("/api/data", headers={"Authorization": f"Bearer {k.key}"})
    assert r.status_code == 200
    assert r.headers["X-RateLimit-Tier"] == "free"  # fell back to anon


def test_x_api_key_header_also_works(app: FastAPI, store: APIKeyStore) -> None:
    pro = APIKey.new(user_id="alice", tier="pro")
    store.save_key(pro)
    client = TestClient(app)
    r = client.get("/api/data", headers={"X-API-Key": pro.key})
    assert r.status_code == 200
    assert r.headers["X-RateLimit-Tier"] == "pro"


def test_middleware_disabled_when_env_unset(
    store: APIKeyStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With PFM_AUTH_ENABLED unset, the middleware never throttles."""
    monkeypatch.delenv("PFM_AUTH_ENABLED", raising=False)
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, store_factory=lambda: store)

    @app.get("/api/data")
    def _data() -> dict:
        return {"ok": True}

    client = TestClient(app)
    for _ in range(50):
        r = client.get("/api/data")
        assert r.status_code == 200
        # No rate-limit headers when off.
        assert "X-RateLimit-Tier" not in r.headers
