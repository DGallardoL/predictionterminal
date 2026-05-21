"""Tests for ``pfm.strategies_catalog_router`` — /strategies/list and discovery."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.strategies_catalog_router import STRATEGY_CATALOG
from pfm.strategies_catalog_router import router as strategies_catalog_router


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(strategies_catalog_router)
    return TestClient(app)


REQUIRED_FIELDS = {"id", "endpoint", "method", "description", "tag"}


def test_list_returns_all_strategies() -> None:
    client = _make_app()
    r = client.get("/strategies/list")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 10
    assert len(body["items"]) == body["total"]


def test_list_each_entry_has_required_fields() -> None:
    client = _make_app()
    r = client.get("/strategies/list")
    body = r.json()
    for item in body["items"]:
        missing = REQUIRED_FIELDS - set(item.keys())
        assert not missing, f"missing {missing} in {item.get('id')}"
        assert item["endpoint"].startswith("/strategies/")
        assert item["method"] in {"GET", "POST"}
        assert len(item["description"]) > 0


def test_list_ids_are_unique() -> None:
    ids = [s.id for s in STRATEGY_CATALOG]
    assert len(set(ids)) == len(ids)


def test_list_known_classical_present() -> None:
    client = _make_app()
    r = client.get("/strategies/list")
    body = r.json()
    ids = {item["id"] for item in body["items"]}
    for must in ("cointegration", "pairs-backtest", "ou-bands", "kalman-hedge", "granger"):
        assert must in ids


def test_discovery_filter_by_tag_classical() -> None:
    client = _make_app()
    r = client.get("/strategies/discovery?tag=classical")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    for item in body["items"]:
        assert item["tag"] == "classical"


def test_discovery_filter_by_tag_stat_arb() -> None:
    client = _make_app()
    r = client.get("/strategies/discovery?tag=stat-arb")
    assert r.status_code == 200
    body = r.json()
    for item in body["items"]:
        assert item["tag"] == "stat-arb"


def test_discovery_all_returns_full_catalog() -> None:
    client = _make_app()
    r = client.get("/strategies/discovery?tag=all")
    body = r.json()
    assert body["total"] == len(STRATEGY_CATALOG)


def test_discovery_invalid_tag_rejected() -> None:
    client = _make_app()
    r = client.get("/strategies/discovery?tag=__nope__")
    assert r.status_code == 422
