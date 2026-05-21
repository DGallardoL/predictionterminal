"""Tests for ``pfm.macro_overlay_unified`` — the multi-series overlay endpoint."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.macro_overlay_unified import router as overlay_router
from pfm.sources.fred import FREDGRAPH_BASE


@pytest.fixture(autouse=True)
def _clear_caches():
    reset_caches()
    yield
    reset_caches()


def _build_csv(series_id: str, start: str, end: str, base: float = 100.0) -> str:
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    lines = [f"DATE,{series_id}"]
    for i, ts in enumerate(idx):
        lines.append(f"{ts.strftime('%Y-%m-%d')},{base + 0.05 * i:.4f}")
    return "\n".join(lines) + "\n"


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(overlay_router)
    return TestClient(app)


@respx.mock
def test_overlay_single_series() -> None:
    csv = _build_csv("DFF", "2025-01-01", "2025-01-05")
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=csv))
    client = _make_app()
    r = client.get("/macro/overlay?series=DFF&start=2025-01-01&end=2025-01-05")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1
    assert len(body["series"]) == 1
    s = body["series"][0]
    assert s["id"] == "DFF"
    assert s["units"] == "Percent"
    assert len(s["dates"]) == 5
    assert len(s["values"]) == 5


@respx.mock
def test_overlay_multiple_series_aligned() -> None:
    """All requested series should appear in the response, in order."""
    requested = ["DFF", "DGS10", "VIXCLS"]

    def _handler(request: httpx.Request) -> httpx.Response:
        sid = request.url.params["id"]
        return httpx.Response(200, text=_build_csv(sid, "2025-01-01", "2025-01-05"))

    respx.get(FREDGRAPH_BASE).mock(side_effect=_handler)
    client = _make_app()
    r = client.get("/macro/overlay?series=DFF,DGS10,VIXCLS&start=2025-01-01&end=2025-01-05")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 3
    ids = [s["id"] for s in body["series"]]
    assert ids == requested
    # All series share the same daily calendar.
    for s in body["series"]:
        assert len(s["dates"]) == 5


@respx.mock
def test_overlay_unknown_series_404() -> None:
    client = _make_app()
    r = client.get("/macro/overlay?series=DFF,FAKESERIES&start=2025-01-01&end=2025-01-05")
    assert r.status_code == 404
    assert "FAKESERIES" in r.json()["detail"]


def test_overlay_empty_series_400() -> None:
    client = _make_app()
    r = client.get("/macro/overlay?series=&start=2025-01-01&end=2025-01-05")
    # Either 400 (semantic empty) or 422 (FastAPI validation) — both are
    # legitimate "rejected empty input" outcomes.
    assert r.status_code in (400, 422)


def test_overlay_bad_window_400() -> None:
    client = _make_app()
    r = client.get("/macro/overlay?series=DFF&start=2025-12-31&end=2025-01-01")
    assert r.status_code == 400


@respx.mock
def test_overlay_handles_lowercase_series() -> None:
    """Series ids are case-insensitive on input — uppercase on output."""
    csv = _build_csv("DFF", "2025-01-01", "2025-01-03")
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=csv))
    client = _make_app()
    r = client.get("/macro/overlay?series=dff&start=2025-01-01&end=2025-01-03")
    assert r.status_code == 200
    assert r.json()["series"][0]["id"] == "DFF"


@respx.mock
def test_overlay_includes_metadata() -> None:
    csv = _build_csv("ICSA", "2025-01-01", "2025-01-03")
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=csv))
    client = _make_app()
    r = client.get("/macro/overlay?series=ICSA&start=2025-01-01&end=2025-01-03")
    assert r.status_code == 200
    s = r.json()["series"][0]
    assert s["name"] == "Initial Jobless Claims"
    assert s["frequency"] == "weekly"
    assert s["units"] == "Number"
