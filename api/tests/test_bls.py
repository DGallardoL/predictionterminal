"""Tests for ``pfm.sources.bls`` — BLS public-API client + router."""

from __future__ import annotations

import time

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.sources.bls import (
    _BLS_SERIES_REGISTRY,
    BLS_API_BASE,
    BLSClient,
    BlsDataError,
    _parse_bls_payload,
    fetch_bls_series,
)
from pfm.sources.bls import (
    router as bls_router,
)

# A trimmed but realistic v2 success payload — three months of UNRATE.
SAMPLE_PAYLOAD = {
    "status": "REQUEST_SUCCEEDED",
    "responseTime": 12,
    "message": [],
    "Results": {
        "series": [
            {
                "seriesID": "LNS14000000",
                "data": [
                    {
                        "year": "2026",
                        "period": "M03",
                        "periodName": "March",
                        "value": "4.1",
                        "footnotes": [{}],
                    },
                    {
                        "year": "2026",
                        "period": "M02",
                        "periodName": "February",
                        "value": "4.0",
                        "footnotes": [{}],
                    },
                    {
                        "year": "2026",
                        "period": "M01",
                        "periodName": "January",
                        "value": "3.9",
                        "footnotes": [{}],
                    },
                    # Annual aggregate row that should be ignored:
                    {
                        "year": "2025",
                        "period": "M13",
                        "periodName": "Annual",
                        "value": "3.8",
                        "footnotes": [{}],
                    },
                ],
            }
        ]
    },
}


@pytest.fixture(autouse=True)
def _clear_caches():
    """Wipe cache state between tests so repeated calls hit the mock."""
    reset_caches()
    yield
    reset_caches()


def test_parse_bls_payload_ignores_annual_aggregates() -> None:
    s = _parse_bls_payload(SAMPLE_PAYLOAD, "LNS14000000")
    assert len(s) == 3
    assert s.iloc[0] == 3.9  # earliest after sort
    assert s.iloc[-1] == 4.1
    assert s.name == "LNS14000000"


def test_parse_bls_payload_handles_failure_status() -> None:
    bad = {"status": "REQUEST_NOT_PROCESSED", "message": ["bad key"], "Results": {}}
    with pytest.raises(BlsDataError, match="REQUEST_NOT_PROCESSED"):
        _parse_bls_payload(bad, "LNS14000000")


@respx.mock
def test_fetch_bls_series_basic() -> None:
    respx.post(BLS_API_BASE).mock(return_value=httpx.Response(200, json=SAMPLE_PAYLOAD))
    df = fetch_bls_series("LNS14000000", 2026, 2026)
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 3
    assert df["value"].iloc[0] == 3.9
    assert df["value"].iloc[-1] == 4.1


@respx.mock
def test_fetch_bls_series_uses_api_key() -> None:
    captured: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=SAMPLE_PAYLOAD)

    respx.post(BLS_API_BASE).mock(side_effect=_handler)
    fetch_bls_series("LNS14000000", 2026, 2026, api_key="secret-key")
    assert captured["body"]["registrationkey"] == "secret-key"
    assert captured["body"]["seriesid"] == ["LNS14000000"]
    assert captured["body"]["startyear"] == "2026"


@respx.mock
def test_fetch_bls_series_no_key_does_not_send_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without BLS_API_KEY the body must omit ``registrationkey`` entirely."""
    monkeypatch.delenv("BLS_API_KEY", raising=False)
    captured: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=SAMPLE_PAYLOAD)

    respx.post(BLS_API_BASE).mock(side_effect=_handler)
    fetch_bls_series("LNS14000000", 2026, 2026)
    assert "registrationkey" not in captured["body"]


@respx.mock
def test_fetch_bls_retries_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    route = respx.post(BLS_API_BASE).mock(
        side_effect=[
            httpx.Response(503, text="busy"),
            httpx.Response(200, json=SAMPLE_PAYLOAD),
        ]
    )
    df = fetch_bls_series("LNS14000000", 2026, 2026, max_retries=3)
    assert route.call_count == 2
    assert len(df) == 3


@respx.mock
def test_fetch_bls_persistent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    respx.post(BLS_API_BASE).mock(return_value=httpx.Response(500, text="err"))
    with pytest.raises(BlsDataError):
        fetch_bls_series("LNS14000000", 2026, 2026, max_retries=2)


@respx.mock
def test_bls_client_caches_between_calls() -> None:
    route = respx.post(BLS_API_BASE).mock(return_value=httpx.Response(200, json=SAMPLE_PAYLOAD))
    cli = BLSClient(api_key=None)
    df1 = cli.fetch("LNS14000000", 2026, 2026)
    df2 = cli.fetch("LNS14000000", 2026, 2026)
    # Second call should be served from cache — exactly 1 HTTP hit.
    assert route.call_count == 1
    assert df1.equals(df2)


def test_registry_has_required_series() -> None:
    expected = {"LNS14000000", "CES0500000003", "CUUR0000SA0L1E", "WPSFD49207", "LNS12300000"}
    assert expected.issubset(_BLS_SERIES_REGISTRY.keys())


# --- router-level tests -----------------------------------------------------


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(bls_router)
    return TestClient(app)


@respx.mock
def test_router_series_endpoint() -> None:
    respx.post(BLS_API_BASE).mock(return_value=httpx.Response(200, json=SAMPLE_PAYLOAD))
    client = _make_app()
    r = client.get("/macro/bls/LNS14000000?start=2026&end=2026")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["series_id"] == "LNS14000000"
    assert body["units"] == "Percent"
    assert len(body["data"]) == 3


def test_router_unknown_series_404() -> None:
    client = _make_app()
    r = client.get("/macro/bls/NOPE?start=2026&end=2026")
    assert r.status_code == 404


def test_router_catalog() -> None:
    client = _make_app()
    r = client.get("/macro/bls/catalog")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(_BLS_SERIES_REGISTRY)
    ids = {s["series_id"] for s in body["series"]}
    assert "LNS14000000" in ids
