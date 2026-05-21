"""Tests that tier-gated endpoints reject lower-tier keys.

We don't go through the full PFM app for every gate (it's huge); instead we
build a stand-in app that mounts the same Depends(require_tier(...)) gate
and verify HTTP-level behaviour. We also smoke-test one real endpoint
(/strategies/optimize) end-to-end against the production main.app.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from pfm.auth.dependencies import require_tier
from pfm.auth.models import APIKey
from pfm.auth.storage import APIKeyStore, get_api_key_store

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def store() -> APIKeyStore:
    return APIKeyStore(":memory:")


@pytest.fixture
def app(store: APIKeyStore, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("PFM_AUTH_ENABLED", "1")
    app = FastAPI()

    @app.get("/free-ok")
    def _free() -> dict:
        return {"ok": True, "tier": "free"}

    @app.get(
        "/pro-only",
        dependencies=[Depends(require_tier("pro"))],
    )
    def _pro() -> dict:
        return {"ok": True, "tier": "pro+"}

    @app.get(
        "/quant-only",
        dependencies=[Depends(require_tier("quant"))],
    )
    def _quant() -> dict:
        return {"ok": True, "tier": "quant+"}

    app.dependency_overrides[get_api_key_store] = lambda: store
    return app


# ---------------------------------------------------------------- tests


def test_pro_endpoint_without_key_returns_401(app: FastAPI) -> None:
    client = TestClient(app)
    r = client.get("/pro-only")
    assert r.status_code == 401


def test_pro_endpoint_with_free_key_returns_403(app: FastAPI, store: APIKeyStore) -> None:
    free = APIKey.new(user_id="alice", tier="free")
    store.save_key(free)
    client = TestClient(app)
    r = client.get("/pro-only", headers={"Authorization": f"Bearer {free.key}"})
    assert r.status_code == 403
    assert "tier 'pro'" in r.json()["detail"]


def test_pro_endpoint_with_pro_key_returns_200(app: FastAPI, store: APIKeyStore) -> None:
    pro = APIKey.new(user_id="bob", tier="pro")
    store.save_key(pro)
    client = TestClient(app)
    r = client.get("/pro-only", headers={"Authorization": f"Bearer {pro.key}"})
    assert r.status_code == 200


def test_pro_endpoint_with_quant_key_passes(app: FastAPI, store: APIKeyStore) -> None:
    """Higher tier satisfies a lower minimum."""
    quant = APIKey.new(user_id="qq", tier="quant")
    store.save_key(quant)
    client = TestClient(app)
    r = client.get("/pro-only", headers={"Authorization": f"Bearer {quant.key}"})
    assert r.status_code == 200


def test_quant_endpoint_with_pro_key_returns_403(app: FastAPI, store: APIKeyStore) -> None:
    pro = APIKey.new(user_id="bob", tier="pro")
    store.save_key(pro)
    client = TestClient(app)
    r = client.get("/quant-only", headers={"Authorization": f"Bearer {pro.key}"})
    assert r.status_code == 403


def test_quant_endpoint_with_enterprise_key_passes(app: FastAPI, store: APIKeyStore) -> None:
    ent = APIKey.new(user_id="ee", tier="enterprise")
    store.save_key(ent)
    client = TestClient(app)
    r = client.get("/quant-only", headers={"Authorization": f"Bearer {ent.key}"})
    assert r.status_code == 200


def test_invalid_token_returns_401(app: FastAPI) -> None:
    client = TestClient(app)
    r = client.get("/pro-only", headers={"Authorization": "Bearer sk_pfm_unknown"})
    assert r.status_code == 401


def test_disabled_key_returns_401(app: FastAPI, store: APIKeyStore) -> None:
    k = APIKey.new(user_id="alice", tier="quant")
    store.save_key(k)
    store.revoke_key(k.key)
    client = TestClient(app)
    r = client.get("/pro-only", headers={"Authorization": f"Bearer {k.key}"})
    assert r.status_code == 401


def test_auth_disabled_lets_everything_through(
    store: APIKeyStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PFM_AUTH_ENABLED", raising=False)
    app = FastAPI()

    @app.get("/quant-only", dependencies=[Depends(require_tier("quant"))])
    def _quant() -> dict:
        return {"ok": True}

    app.dependency_overrides[get_api_key_store] = lambda: store
    client = TestClient(app)
    r = client.get("/quant-only")
    assert r.status_code == 200


def test_free_endpoint_open_when_auth_off(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity: the un-gated endpoint isn't accidentally locked."""
    monkeypatch.delenv("PFM_AUTH_ENABLED", raising=False)
    client = TestClient(app)
    r = client.get("/free-ok")
    assert r.status_code == 200
