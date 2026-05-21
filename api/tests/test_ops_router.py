"""Tests for ``GET /ops/sessions`` and ``GET /ops/config``
(:mod:`pfm.ops_router`).

All filesystem reads go through ``PFM_OPS_ACTIVE_EDITS_PATH`` so we never
touch the real coordination file. The router is mounted on a fresh
``FastAPI`` app — no lifespan / Redis / Polymarket dependencies — so the
suite stays under 1 second.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.ops_router import (
    _mask_env_value,
    _mask_url_password,
    _parse_iso_utc,
    router,
)

# ───────────────────────── fixtures ─────────────────────────


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


@pytest.fixture()
def active_edits_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Empty active-edits.json under tmp_path; tests overwrite as needed."""
    path = tmp_path / "active-edits.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("PFM_OPS_ACTIVE_EDITS_PATH", str(path))
    return path


@pytest.fixture()
def client(
    active_edits_file: Path,
) -> TestClient:
    """Bare FastAPI app mounting only the ops router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ───────────────────────── /ops/sessions ─────────────────────────


class TestOpsSessions:
    def test_returns_200_and_empty_array_for_empty_file(self, client: TestClient) -> None:
        resp = client.get("/ops/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_sessions"] == []
        assert body["count"] == 0
        assert "checked_at" in body

    def test_returns_200_when_file_missing(
        self, client: TestClient, active_edits_file: Path
    ) -> None:
        active_edits_file.unlink()
        resp = client.get("/ops/sessions")
        assert resp.status_code == 200
        assert resp.json()["active_sessions"] == []

    def test_filters_expired_and_keeps_active(
        self, client: TestClient, active_edits_file: Path
    ) -> None:
        """Mock 2 active + 1 expired → only 2 returned."""
        now = datetime.now(UTC)
        entries = [
            {
                "session_id": "active-1",
                "scope": "scope-1",
                "files": ["api/src/pfm/a.py"],
                "started_at": _iso(now - timedelta(minutes=5)),
                "expires_at": _iso(now + timedelta(minutes=25)),
            },
            {
                "session_id": "active-2",
                "scope": "scope-2",
                "files": ["api/src/pfm/b.py"],
                "started_at": _iso(now - timedelta(minutes=10)),
                "expires_at": _iso(now + timedelta(minutes=20)),
            },
            {
                "session_id": "expired-1",
                "scope": "scope-3",
                "files": ["api/src/pfm/c.py"],
                "started_at": _iso(now - timedelta(hours=2)),
                "expires_at": _iso(now - timedelta(minutes=30)),
            },
        ]
        active_edits_file.write_text(json.dumps(entries), encoding="utf-8")

        resp = client.get("/ops/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        ids = {s["session_id"] for s in body["active_sessions"]}
        assert ids == {"active-1", "active-2"}

    def test_filters_completed_status(self, client: TestClient, active_edits_file: Path) -> None:
        """Entries with ``status == COMPLETED`` are hidden even if unexpired."""
        now = datetime.now(UTC)
        entries = [
            {
                "session_id": "done",
                "scope": "scope-done",
                "files": [],
                "status": "COMPLETED",
                "started_at": _iso(now - timedelta(minutes=10)),
                "expires_at": _iso(now + timedelta(minutes=20)),
            },
            {
                "session_id": "active",
                "scope": "scope-active",
                "files": [],
                "started_at": _iso(now - timedelta(minutes=10)),
                "expires_at": _iso(now + timedelta(minutes=20)),
            },
        ]
        active_edits_file.write_text(json.dumps(entries), encoding="utf-8")

        body = client.get("/ops/sessions").json()
        assert body["count"] == 1
        assert body["active_sessions"][0]["session_id"] == "active"

    def test_malformed_json_returns_empty(
        self, client: TestClient, active_edits_file: Path
    ) -> None:
        active_edits_file.write_text("{not json", encoding="utf-8")
        resp = client.get("/ops/sessions")
        assert resp.status_code == 200
        assert resp.json()["active_sessions"] == []

    def test_non_array_root_returns_empty(
        self, client: TestClient, active_edits_file: Path
    ) -> None:
        active_edits_file.write_text('{"not": "an array"}', encoding="utf-8")
        body = client.get("/ops/sessions").json()
        assert body["active_sessions"] == []
        assert body["count"] == 0

    def test_preserves_full_entry_schema(self, client: TestClient, active_edits_file: Path) -> None:
        """Every field on the source entry round-trips to the response."""
        now = datetime.now(UTC)
        entry = {
            "session_id": "session-x",
            "scope": "scope-x",
            "files": ["a", "b", "c"],
            "started_at": _iso(now),
            "expires_at": _iso(now + timedelta(minutes=10)),
            "task_id": "T27",
            "wave": "wave-10",
            "custom_field": "hello",
        }
        active_edits_file.write_text(json.dumps([entry]), encoding="utf-8")

        out = client.get("/ops/sessions").json()["active_sessions"][0]
        for key, val in entry.items():
            assert out[key] == val


# ───────────────────────── /ops/config ─────────────────────────


class TestOpsConfig:
    def test_returns_200_with_expected_sections(self, client: TestClient) -> None:
        resp = client.get("/ops/config")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) >= {"env", "runtime", "openapi", "cache_stats"}
        assert "workers" in body["runtime"]
        assert "uptime_s" in body["runtime"]
        assert "factor_count" in body["runtime"]
        assert "path_count" in body["openapi"]

    def test_sensitive_env_var_masked(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PFM_ADMIN_TOKEN", "super-secret-value")
        monkeypatch.setenv("PFM_API_SECRET", "another-secret")
        monkeypatch.setenv("PFM_DB_PASSWORD", "hunter2")
        body = client.get("/ops/config").json()
        assert body["env"]["PFM_ADMIN_TOKEN"] == "***"
        assert body["env"]["PFM_API_SECRET"] == "***"
        assert body["env"]["PFM_DB_PASSWORD"] == "***"

    def test_non_secret_env_passes_through(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PFM_ENV", "dev")
        monkeypatch.setenv("PFM_LOG_LEVEL", "INFO")
        body = client.get("/ops/config").json()
        assert body["env"]["PFM_ENV"] == "dev"
        assert body["env"]["PFM_LOG_LEVEL"] == "INFO"

    def test_redis_url_password_masked(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "REDIS_URL",
            "redis://default:hunter2@my-redis.internal:6379/0",
        )
        body = client.get("/ops/config").json()
        assert body["env"]["REDIS_URL"] == "redis://default:***@my-redis.internal:6379/0"
        # Password substring must NOT leak anywhere in the env block.
        assert "hunter2" not in json.dumps(body["env"])

    def test_factor_count_matches_app_state(self, active_edits_file: Path) -> None:
        """``runtime.factor_count`` reflects ``app.state.factors`` length."""
        app = FastAPI()
        app.state.factors = {f"slug-{i}": object() for i in range(17)}
        app.include_router(router)
        client = TestClient(app)
        body = client.get("/ops/config").json()
        assert body["runtime"]["factor_count"] == 17

    def test_factor_count_zero_when_state_missing(self, client: TestClient) -> None:
        body = client.get("/ops/config").json()
        assert body["runtime"]["factor_count"] == 0

    def test_openapi_path_count_matches_spec(self, client: TestClient) -> None:
        """``openapi.path_count`` matches the live ``/openapi.json`` keys."""
        spec = client.get("/openapi.json").json()
        expected = len(spec["paths"])
        body = client.get("/ops/config").json()
        assert body["openapi"]["path_count"] == expected
        # The bare app has /ops/sessions, /ops/config, and /openapi.json
        # is built lazily; verify the count is at least 2 to catch a
        # silent zero.
        assert body["openapi"]["path_count"] >= 2

    def test_workers_from_web_concurrency(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_CONCURRENCY", "4")
        body = client.get("/ops/config").json()
        assert body["runtime"]["workers"] == 4

    def test_workers_defaults_to_one(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
        monkeypatch.delenv("GUNICORN_WORKERS", raising=False)
        body = client.get("/ops/config").json()
        assert body["runtime"]["workers"] == 1

    def test_uptime_is_positive(self, client: TestClient) -> None:
        body = client.get("/ops/config").json()
        assert isinstance(body["runtime"]["uptime_s"], (int, float))
        assert body["runtime"]["uptime_s"] >= 0

    def test_only_pfm_and_whitelisted_env_keys_returned(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Random env vars without the ``PFM_`` prefix don't leak."""
        monkeypatch.setenv("UNRELATED_VAR", "should-not-appear")
        monkeypatch.setenv("PFM_FEATURE_X", "yes")
        body = client.get("/ops/config").json()
        assert "UNRELATED_VAR" not in body["env"]
        assert body["env"]["PFM_FEATURE_X"] == "yes"


# ───────────────────────── helper-level units ─────────────────────────


class TestMaskHelpers:
    def test_mask_password_in_url(self) -> None:
        assert _mask_url_password("redis://user:pass@host:6379/0") == "redis://user:***@host:6379/0"

    def test_mask_password_idempotent(self) -> None:
        already = "redis://user:***@host:6379/0"
        assert _mask_url_password(already) == already

    def test_mask_password_noop_when_no_password(self) -> None:
        assert _mask_url_password("redis://host:6379/0") == "redis://host:6379/0"

    def test_mask_env_value_secret_name(self) -> None:
        assert _mask_env_value("PFM_API_TOKEN", "abc123") == "***"
        assert _mask_env_value("STRIPE_SECRET", "sk_live_x") == "***"
        # Case-insensitive
        assert _mask_env_value("pfm_password", "pw") == "***"

    def test_mask_env_value_url_password_only(self) -> None:
        out = _mask_env_value("REDIS_URL", "redis://u:p@h/0")
        assert out == "redis://u:***@h/0"

    def test_mask_env_value_plain_passthrough(self) -> None:
        assert _mask_env_value("PFM_ENV", "dev") == "dev"

    def test_parse_iso_utc_z_suffix(self) -> None:
        dt = _parse_iso_utc("2026-05-16T12:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_iso_utc_offset_suffix(self) -> None:
        dt = _parse_iso_utc("2026-05-16T12:00:00+00:00")
        assert dt is not None

    def test_parse_iso_utc_invalid_returns_none(self) -> None:
        assert _parse_iso_utc("not a date") is None
        assert _parse_iso_utc("") is None
