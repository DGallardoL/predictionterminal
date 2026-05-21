"""Tests for production-aware auth defaults + admin-token autogen.

Covers :mod:`pfm.auth.production` (env detection, token autogen, first-boot
flag) and the integration points that consume it: the rate-limit middleware,
``/health/detail``, and the new ``/auth/first-boot-info`` endpoint.

Each test gets fresh tmp paths via ``monkeypatch.setattr`` on the module-level
``ADMIN_TOKEN_PATH`` / ``FIRST_BOOT_FLAG_PATH`` constants so we never touch the
real ``/tmp`` files (which would leak state across the whole suite).
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.auth import production as prod_mod
from pfm.auth.dependencies import auth_enabled
from pfm.auth.models import APIKey
from pfm.auth.rate_limiter import RateLimitMiddleware
from pfm.auth.router import router as auth_router
from pfm.auth.storage import APIKeyStore, get_api_key_store

# Env vars that influence detection — clear on every test to isolate.
_ENV_VARS = (
    "PFM_AUTH_ENABLED",
    "PFM_ADMIN_TOKEN",
    "ENV",
    "FLY_APP_NAME",
    "RENDER",
    "NODE_ENV",
)


@pytest.fixture(autouse=True)
def _clean_env_and_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strip prod-detection env vars and redirect autogen files into tmp_path."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    token_path = tmp_path / "pfm_admin_token.json"
    flag_path = tmp_path / "pfm_first_boot_done.flag"
    monkeypatch.setattr(prod_mod, "ADMIN_TOKEN_PATH", token_path)
    monkeypatch.setattr(prod_mod, "FIRST_BOOT_FLAG_PATH", flag_path)


# ---------------------------------------------------------------- is_auth_enabled


def test_is_auth_enabled_off_when_no_signals() -> None:
    assert prod_mod.is_auth_enabled() is False
    assert prod_mod.detect_env_reason() == "off"


def test_is_auth_enabled_explicit_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFM_AUTH_ENABLED", "1")
    assert prod_mod.is_auth_enabled() is True
    assert prod_mod.detect_env_reason() == "explicit_on"


def test_is_auth_enabled_explicit_off_overrides_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("PFM_AUTH_ENABLED", "0")
    assert prod_mod.is_auth_enabled() is False
    assert prod_mod.detect_env_reason() == "explicit_off"


def test_is_auth_enabled_env_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "production")
    assert prod_mod.is_auth_enabled() is True
    assert prod_mod.detect_env_reason() == "production"


def test_is_auth_enabled_env_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "dev")
    assert prod_mod.is_auth_enabled() is False
    assert prod_mod.detect_env_reason() == "off"


def test_is_auth_enabled_fly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLY_APP_NAME", "pfm-prod")
    assert prod_mod.is_auth_enabled() is True
    assert prod_mod.detect_env_reason() == "fly"


def test_is_auth_enabled_render(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENDER", "true")
    assert prod_mod.is_auth_enabled() is True
    assert prod_mod.detect_env_reason() == "render"


def test_is_auth_enabled_node_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NODE_ENV", "production")
    assert prod_mod.is_auth_enabled() is True
    assert prod_mod.detect_env_reason() == "node_production"


def test_legacy_auth_enabled_alias_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pfm.auth.dependencies.auth_enabled`` must keep its old contract."""
    monkeypatch.setenv("PFM_AUTH_ENABLED", "1")
    assert auth_enabled() is True
    monkeypatch.delenv("PFM_AUTH_ENABLED")
    assert auth_enabled() is False


# --------------------------------------------------------- get_or_generate_admin_token


def test_admin_token_returns_env_var_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "supersecret-from-env")
    monkeypatch.setenv("ENV", "production")
    tok = prod_mod.get_or_generate_admin_token()
    assert tok == "supersecret-from-env"
    # Env-supplied token does NOT count as autogen.
    assert prod_mod.is_admin_token_autogen() is False
    # And no file was created.
    assert not prod_mod.ADMIN_TOKEN_PATH.exists()


def test_admin_token_empty_when_auth_disabled() -> None:
    assert prod_mod.is_auth_enabled() is False
    assert prod_mod.get_or_generate_admin_token() == ""
    assert prod_mod.admin_token_configured() is False


def test_admin_token_autogen_persists_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "production")
    tok = prod_mod.get_or_generate_admin_token()
    assert tok.startswith("sk_admin_")
    assert len(tok) > len("sk_admin_") + 30  # urlsafe(32) ⇒ 43 chars
    assert prod_mod.ADMIN_TOKEN_PATH.exists()
    payload = json.loads(prod_mod.ADMIN_TOKEN_PATH.read_text())
    assert payload["token"] == tok
    assert "generated_at_iso" in payload
    assert "warning" in payload
    assert prod_mod.is_admin_token_autogen() is True


def test_admin_token_autogen_reuses_persisted_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "production")
    tok1 = prod_mod.get_or_generate_admin_token()
    tok2 = prod_mod.get_or_generate_admin_token()
    assert tok1 == tok2  # second call reads the persisted file


def test_admin_token_file_permissions_0600(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "production")
    prod_mod.get_or_generate_admin_token()
    mode = stat.S_IMODE(prod_mod.ADMIN_TOKEN_PATH.stat().st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_admin_token_corrupt_file_regenerated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "production")
    prod_mod.ADMIN_TOKEN_PATH.write_text("not json {{{")
    tok = prod_mod.get_or_generate_admin_token()
    assert tok.startswith("sk_admin_")
    # File now contains valid JSON.
    payload = json.loads(prod_mod.ADMIN_TOKEN_PATH.read_text())
    assert payload["token"] == tok


def test_admin_token_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ENV", "production")
    with caplog.at_level("WARNING", logger="pfm.auth.production"):
        tok = prod_mod.get_or_generate_admin_token()
    assert any("Generated admin token" in rec.message for rec in caplog.records)
    assert any(tok in rec.message for rec in caplog.records)


def test_admin_token_configured_reflects_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # Off, no env, no file → False.
    assert prod_mod.admin_token_configured() is False
    # On, no env, no file → False (until autogen runs).
    monkeypatch.setenv("ENV", "production")
    assert prod_mod.admin_token_configured() is False
    # After autogen → True.
    prod_mod.get_or_generate_admin_token()
    assert prod_mod.admin_token_configured() is True
    # Env var set → always True.
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "explicit")
    assert prod_mod.admin_token_configured() is True


# ---------------------------------------------------------------- first-boot endpoint


@pytest.fixture
def app_with_auth_router() -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router)
    return app


def test_first_boot_endpoint_returns_token_then_410(
    app_with_auth_router: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENV", "production")
    client = TestClient(app_with_auth_router)
    r1 = client.get("/auth/first-boot-info")
    assert r1.status_code == 200, r1.text
    body = r1.json()
    assert body["admin_token"].startswith("sk_admin_")
    assert "Save this token" in body["message"]
    assert "warning" in body
    # Second call: 410 Gone.
    r2 = client.get("/auth/first-boot-info")
    assert r2.status_code == 410


def test_first_boot_endpoint_returns_404_when_auth_off(
    app_with_auth_router: FastAPI,
) -> None:
    """Auth OFF: endpoint must 404, never leak the dev posture."""
    client = TestClient(app_with_auth_router)
    r = client.get("/auth/first-boot-info")
    assert r.status_code == 404
    # No flag written — still pristine.
    assert not prod_mod.FIRST_BOOT_FLAG_PATH.exists()


def test_first_boot_endpoint_uses_env_token_when_present(
    app_with_auth_router: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "from-the-env")
    client = TestClient(app_with_auth_router)
    r = client.get("/auth/first-boot-info")
    assert r.status_code == 200
    assert r.json()["admin_token"] == "from-the-env"


# ---------------------------------------------------------------- /health/detail


def test_health_detail_includes_auth_status() -> None:
    """/health/detail should expose the auth posture without leaking the token."""
    from pfm.health_router import router as health_router

    app = FastAPI()
    app.include_router(health_router)
    client = TestClient(app)
    r = client.get("/health/detail")
    assert r.status_code == 200
    body = r.json()
    assert "auth_status" in body
    assert body["auth_status"]["enabled"] is False
    assert body["auth_status"]["env_detection"] == "off"
    assert body["auth_status"]["admin_token_configured"] is False
    assert body["auth_status"]["autogen_token_in_use"] is False
    # Token must never appear in the response.
    assert "sk_admin_" not in r.text


def test_health_detail_env_detection_reports_fly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pfm.health_router import router as health_router

    monkeypatch.setenv("FLY_APP_NAME", "pfm-prod")
    app = FastAPI()
    app.include_router(health_router)
    client = TestClient(app)
    r = client.get("/health/detail")
    body = r.json()
    assert body["auth_status"]["enabled"] is True
    assert body["auth_status"]["env_detection"] == "fly"


# ---------------------------------------------------------------- integration


@pytest.fixture
def store() -> APIKeyStore:
    return APIKeyStore(":memory:")


@pytest.fixture
def integration_app(store: APIKeyStore) -> FastAPI:
    """Auth-protected mini-app mirroring the real prod posture."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, store_factory=lambda: store)
    app.include_router(auth_router)

    # Stand-in for /strategies/optimize: gate behind require_admin so the
    # autogen-token path can be exercised end-to-end.
    from fastapi import Depends

    from pfm.auth.dependencies import require_admin

    @app.get("/strategies/optimize", dependencies=[Depends(require_admin)])
    def _optimize() -> dict:
        return {"ok": True}

    app.dependency_overrides[get_api_key_store] = lambda: store
    return app


def test_protected_endpoint_blocks_anon_in_production(
    integration_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENV", "production")
    client = TestClient(integration_app)
    r = client.get("/strategies/optimize")
    # No admin token header → 403 from require_admin.
    assert r.status_code == 403


def test_protected_endpoint_accepts_autogen_admin_token(
    integration_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENV", "production")
    # Simulate startup autogen.
    token = prod_mod.get_or_generate_admin_token()
    assert token.startswith("sk_admin_")
    client = TestClient(integration_app)
    r = client.get("/strategies/optimize", headers={"X-Admin-Token": token})
    assert r.status_code == 200, r.text


def test_protected_endpoint_rejects_wrong_admin_token(
    integration_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENV", "production")
    prod_mod.get_or_generate_admin_token()  # mints
    client = TestClient(integration_app)
    r = client.get("/strategies/optimize", headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 403


def test_rate_limit_middleware_active_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without explicit PFM_AUTH_ENABLED, ENV=production should engage the limiter."""
    monkeypatch.setenv("ENV", "production")
    store = APIKeyStore(":memory:")
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, store_factory=lambda: store)

    @app.get("/api/data")
    def _data() -> dict:
        return {"ok": True}

    client = TestClient(app)
    r = client.get("/api/data")
    assert r.status_code == 200
    # Rate-limit header proves the middleware ran (instead of fast-path).
    assert "X-RateLimit-Tier" in r.headers


def test_rate_limit_middleware_idle_when_no_signals() -> None:
    """Default (dev) state: middleware fast-paths, no headers stamped."""
    store = APIKeyStore(":memory:")
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, store_factory=lambda: store)

    @app.get("/api/data")
    def _data() -> dict:
        return {"ok": True}

    client = TestClient(app)
    r = client.get("/api/data")
    assert r.status_code == 200
    assert "X-RateLimit-Tier" not in r.headers


def test_protected_endpoint_with_pfm_admin_token_env(
    integration_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "env-supplied-token")
    client = TestClient(integration_app)
    r = client.get("/strategies/optimize", headers={"X-Admin-Token": "env-supplied-token"})
    assert r.status_code == 200
    # Wrong token still 403.
    r2 = client.get("/strategies/optimize", headers={"X-Admin-Token": "nope"})
    assert r2.status_code == 403


def test_api_key_bypass_disabled_in_production(
    integration_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    store: APIKeyStore,
) -> None:
    """A real API key (Bearer) hits the rate limiter but not require_admin.

    Sanity check that the auth-on path doesn't accidentally let a Bearer key
    through the require_admin gate.
    """
    monkeypatch.setenv("ENV", "production")
    pro = APIKey.new(user_id="alice", tier="pro")
    store.save_key(pro)
    client = TestClient(integration_app)
    r = client.get("/strategies/optimize", headers={"Authorization": f"Bearer {pro.key}"})
    assert r.status_code == 403  # require_admin still wants X-Admin-Token
