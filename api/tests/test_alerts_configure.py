"""Tests for ``GET /alerts/configure`` and ``POST /alerts/configure``
(:mod:`pfm.alerts.configure_router`).

All tests redirect the on-disk persistence path to a per-test temporary
file via the ``PFM_ALERTS_CONFIG_PATH`` env var, so the production
``/tmp/pfm-alerts-config.json`` is never touched. The router itself
re-resolves the path on every call, so this is enough to fully isolate
each test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.alerts.configure_router import (
    DEFAULT_ARB_MIN_SPREAD_PCT,
    DEFAULT_JUMP_THRESHOLD_PP,
    DEFAULT_SENTIMENT_DISAGREE_PCT,
    AlertConfig,
    _load,
    _save,
    router,
)

# ─────────────────────────────────────────────────────────── fixtures


@pytest.fixture()
def tmp_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the router at a fresh per-test config file."""
    target = tmp_path / "pfm-alerts-config.json"
    monkeypatch.setenv("PFM_ALERTS_CONFIG_PATH", str(target))
    return target


@pytest.fixture()
def client(tmp_config_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ──────────────────────────────────────────────── GET returns defaults


class TestGetDefaults:
    def test_get_with_no_file_returns_defaults(
        self, client: TestClient, tmp_config_path: Path
    ) -> None:
        assert not tmp_config_path.exists()
        resp = client.get("/alerts/configure")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {
            "jump_threshold_pp": DEFAULT_JUMP_THRESHOLD_PP,
            "sentiment_disagree_pct": DEFAULT_SENTIMENT_DISAGREE_PCT,
            "arb_min_spread_pct": DEFAULT_ARB_MIN_SPREAD_PCT,
        }

    def test_get_with_corrupt_file_falls_back_to_defaults(
        self, client: TestClient, tmp_config_path: Path
    ) -> None:
        tmp_config_path.write_text("{not json", encoding="utf-8")
        resp = client.get("/alerts/configure")
        assert resp.status_code == 200
        assert resp.json()["jump_threshold_pp"] == DEFAULT_JUMP_THRESHOLD_PP

    def test_get_with_non_object_json_falls_back_to_defaults(
        self, client: TestClient, tmp_config_path: Path
    ) -> None:
        tmp_config_path.write_text("[1, 2, 3]", encoding="utf-8")
        resp = client.get("/alerts/configure")
        assert resp.status_code == 200
        assert resp.json()["sentiment_disagree_pct"] == DEFAULT_SENTIMENT_DISAGREE_PCT


# ────────────────────────────────────────────────── POST round-trip


class TestPostRoundTrip:
    def test_post_full_payload_persists_and_returns_full_config(
        self, client: TestClient, tmp_config_path: Path
    ) -> None:
        payload = {
            "jump_threshold_pp": 7.5,
            "sentiment_disagree_pct": 55.0,
            "arb_min_spread_pct": 3.0,
        }
        resp = client.post("/alerts/configure", json=payload)
        assert resp.status_code == 200
        assert resp.json() == payload

        # File has been written and contains exactly the new values.
        assert tmp_config_path.exists()
        on_disk = json.loads(tmp_config_path.read_text(encoding="utf-8"))
        assert on_disk == payload

        # And a subsequent GET reads them back.
        get_resp = client.get("/alerts/configure")
        assert get_resp.json() == payload

    def test_post_partial_payload_merges_with_existing(
        self, client: TestClient, tmp_config_path: Path
    ) -> None:
        # Seed full state.
        client.post(
            "/alerts/configure",
            json={
                "jump_threshold_pp": 9.0,
                "sentiment_disagree_pct": 60.0,
                "arb_min_spread_pct": 4.0,
            },
        )
        # Patch only one knob.
        resp = client.post("/alerts/configure", json={"jump_threshold_pp": 2.5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["jump_threshold_pp"] == 2.5
        # The others were preserved.
        assert body["sentiment_disagree_pct"] == 60.0
        assert body["arb_min_spread_pct"] == 4.0

    def test_post_empty_payload_returns_current_unchanged(
        self, client: TestClient, tmp_config_path: Path
    ) -> None:
        # Seed.
        client.post("/alerts/configure", json={"arb_min_spread_pct": 1.25})
        # Empty body = no changes.
        resp = client.post("/alerts/configure", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["arb_min_spread_pct"] == 1.25
        assert body["jump_threshold_pp"] == DEFAULT_JUMP_THRESHOLD_PP

    def test_post_null_field_is_treated_as_no_change(
        self, client: TestClient, tmp_config_path: Path
    ) -> None:
        # Seed.
        client.post("/alerts/configure", json={"jump_threshold_pp": 8.0})
        # Explicit null for the same knob should NOT reset it to default.
        resp = client.post(
            "/alerts/configure",
            json={"jump_threshold_pp": None, "sentiment_disagree_pct": 42.0},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["jump_threshold_pp"] == 8.0
        assert body["sentiment_disagree_pct"] == 42.0


# ────────────────────────────────────────────────────── Validation


class TestValidation:
    def test_post_negative_jump_threshold_is_422(self, client: TestClient) -> None:
        resp = client.post("/alerts/configure", json={"jump_threshold_pp": -1.0})
        assert resp.status_code == 422

    def test_post_above_max_sentiment_is_422(self, client: TestClient) -> None:
        resp = client.post("/alerts/configure", json={"sentiment_disagree_pct": 250.0})
        assert resp.status_code == 422

    def test_post_non_numeric_is_422(self, client: TestClient) -> None:
        resp = client.post("/alerts/configure", json={"arb_min_spread_pct": "lots"})
        assert resp.status_code == 422

    def test_post_unknown_field_is_ignored(self, client: TestClient, tmp_config_path: Path) -> None:
        # Pydantic by default ignores extras → request still succeeds.
        resp = client.post(
            "/alerts/configure",
            json={"jump_threshold_pp": 3.0, "made_up_knob": 999},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["jump_threshold_pp"] == 3.0
        # The extra key did not leak onto disk.
        on_disk = json.loads(tmp_config_path.read_text(encoding="utf-8"))
        assert "made_up_knob" not in on_disk


# ──────────────────────────────────────────────── Persistence layer


class TestPersistenceLayer:
    def test_save_writes_atomically_via_replace(self, tmp_config_path: Path) -> None:
        _save(
            {
                "jump_threshold_pp": 6.0,
                "sentiment_disagree_pct": 50.0,
                "arb_min_spread_pct": 1.5,
            }
        )
        assert tmp_config_path.exists()
        # No leftover temp files in the directory.
        leftovers = [
            p
            for p in tmp_config_path.parent.iterdir()
            if p.name.startswith(".pfm-alerts-config-") and p.suffix == ".tmp"
        ]
        assert leftovers == []

    def test_load_clamps_out_of_range_values_back_to_defaults(self, tmp_config_path: Path) -> None:
        # Hand-write a file with one out-of-range value.
        tmp_config_path.write_text(
            json.dumps(
                {
                    "jump_threshold_pp": 9999.0,  # absurd → ignored
                    "sentiment_disagree_pct": 33.0,  # OK
                    "arb_min_spread_pct": -5.0,  # negative → ignored
                }
            ),
            encoding="utf-8",
        )
        loaded = _load()
        # Bad fields fall back to defaults, good field is kept.
        assert loaded["jump_threshold_pp"] == DEFAULT_JUMP_THRESHOLD_PP
        assert loaded["sentiment_disagree_pct"] == 33.0
        assert loaded["arb_min_spread_pct"] == DEFAULT_ARB_MIN_SPREAD_PCT

    def test_save_then_load_roundtrip(self, tmp_config_path: Path) -> None:
        payload = {
            "jump_threshold_pp": 2.0,
            "sentiment_disagree_pct": 25.0,
            "arb_min_spread_pct": 0.5,
        }
        _save(payload)
        assert _load() == payload


# ──────────────────────────────────────────────── Pydantic model


class TestAlertConfigModel:
    def test_all_fields_optional(self) -> None:
        m = AlertConfig()
        assert m.jump_threshold_pp is None
        assert m.sentiment_disagree_pct is None
        assert m.arb_min_spread_pct is None

    def test_model_dump_exclude_unset_skips_missing(self) -> None:
        m = AlertConfig(jump_threshold_pp=4.0)
        dumped = m.model_dump(exclude_unset=True, exclude_none=True)
        assert dumped == {"jump_threshold_pp": 4.0}

    def test_zero_is_a_valid_value(self) -> None:
        # 0 means "alert on everything"; not an error.
        m = AlertConfig(
            jump_threshold_pp=0.0,
            sentiment_disagree_pct=0.0,
            arb_min_spread_pct=0.0,
        )
        assert m.jump_threshold_pp == 0.0
