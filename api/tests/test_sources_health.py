"""Tests for source health probes + ``/sources/*`` router."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources import equity as equity_mod
from pfm.sources import health as health_mod
from pfm.sources.health import check_all_sources
from pfm.sources.health_router import router as sources_router


@pytest.fixture
def app() -> FastAPI:
    a = FastAPI()
    a.include_router(sources_router)
    return a


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Fresh delisted registry + caches per test."""
    monkeypatch.setattr(
        equity_mod,
        "DELISTED_REGISTRY_PATH",
        tmp_path / "delisted.json",
    )
    equity_mod._EQUITY_CACHE.clear()
    yield
    equity_mod._EQUITY_CACHE.clear()


@respx.mock
def test_check_all_sources_all_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "fake-token")

    respx.get(url__regex=r"https://query1\.finance\.yahoo\.com/.*").mock(
        return_value=httpx.Response(200, json={"chart": {"result": []}})
    )
    respx.get(url__regex=r"https://api\.tiingo\.com/.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__regex=r"https://stooq\.com/.*").mock(
        return_value=httpx.Response(200, text="Date,Close\n2025-01-02,100.0\n")
    )
    respx.get(url__regex=r"https://gamma-api\.polymarket\.com/.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__regex=r"https://api\.elections\.kalshi\.com/.*").mock(
        return_value=httpx.Response(200, json={"events": []})
    )
    respx.get(url__regex=r"https://fred\.stlouisfed\.org/.*").mock(
        return_value=httpx.Response(200, text="DATE,DFF\n2024-01-02,5.32\n")
    )

    out = check_all_sources()

    assert set(out.keys()) == {"yfinance", "tiingo", "stooq", "polymarket", "kalshi", "fred"}
    for name, payload in out.items():
        assert payload["ok"] is True, f"{name} not ok: {payload}"
        assert payload["configured"] is True
        assert payload["latency_ms"] is not None


def test_tiingo_not_configured_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    out = health_mod.check_tiingo()
    assert out["configured"] is False
    assert out["ok"] is False
    assert out["latency_ms"] is None


@respx.mock
def test_yfinance_down_marks_unhealthy() -> None:
    respx.get(url__regex=r"https://query1\.finance\.yahoo\.com/.*").mock(
        return_value=httpx.Response(503, text="Down")
    )
    out = health_mod.check_yfinance()
    assert out["ok"] is False
    assert "503" in (out["detail"] or "")


@respx.mock
def test_health_endpoint(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "fake-token")
    # Patch all probes to return UP via respx routes.
    respx.get(url__regex=r"https://query1\.finance\.yahoo\.com/.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(url__regex=r"https://api\.tiingo\.com/.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__regex=r"https://stooq\.com/.*").mock(
        return_value=httpx.Response(200, text="Date,Close\n2025-01-02,100.0\n")
    )
    respx.get(url__regex=r"https://gamma-api\.polymarket\.com/.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__regex=r"https://api\.elections\.kalshi\.com/.*").mock(
        return_value=httpx.Response(200, json={"events": []})
    )
    respx.get(url__regex=r"https://fred\.stlouisfed\.org/.*").mock(
        return_value=httpx.Response(200, text="DATE,DFF\n2024-01-02,5.32\n")
    )

    with TestClient(app) as client:
        r = client.get("/sources/health")
    assert r.status_code == 200
    body = r.json()
    assert "sources" in body
    assert "summary" in body
    assert set(body["sources"].keys()) == {
        "yfinance",
        "tiingo",
        "stooq",
        "polymarket",
        "kalshi",
        "fred",
    }
    assert body["summary"]["total"] == 6
    assert body["summary"]["up"] == 6


def test_delisted_endpoints_roundtrip(app: FastAPI) -> None:
    with TestClient(app) as client:
        r = client.get("/sources/delisted")
        assert r.status_code == 200
        assert r.json()["count"] == 0

        r = client.post("/sources/delisted/DEADCO")
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] == "DEADCO"
        assert body["marked"] is True
        assert "DEADCO" in body["tickers"]

        r = client.get("/sources/delisted")
        assert r.status_code == 200
        assert "DEADCO" in r.json()["tickers"]
