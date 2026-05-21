"""Tests for the /strategies/fred-cointegration endpoint.

Covers the happy path (Polymarket factor vs DFF) with the FRED HTTP call
mocked via respx, plus a 400 case for an unknown factor id (no network
call expected).
"""

from __future__ import annotations

import httpx
import pandas as pd
import respx
from fastapi.testclient import TestClient

from pfm.sources.fred import FREDGRAPH_BASE


def _build_fred_csv(start: str, end: str, *, series_id: str = "DFF") -> str:
    """Build a daily CSV (header + every UTC date) compatible with fredgraph."""
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    lines = [f"DATE,{series_id}"]
    # A slowly-drifting rate series in (4.5, 5.5) — varies enough for the
    # ADF test to run without numerical issues.
    for i, ts in enumerate(idx):
        val = 5.00 + 0.005 * (i % 50) - 0.003 * (i % 13)
        lines.append(f"{ts.strftime('%Y-%m-%d')},{val:.4f}")
    return "\n".join(lines) + "\n"


@respx.mock
def test_strategies_fred_cointegration_basic(app_client: TestClient) -> None:
    """Happy path: Polymarket factor vs DFF returns a well-formed verdict."""
    csv = _build_fred_csv("2025-06-01", "2025-12-31")
    route = respx.get(FREDGRAPH_BASE).mock(
        return_value=httpx.Response(200, text=csv),
    )
    r = app_client.post(
        "/strategies/fred-cointegration",
        json={
            "factor_id": "factor_a",
            "fred_series": "DFF",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "transform": "raw",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert route.called
    assert body["factor_id"] == "factor_a"
    assert body["fred_series"] == "DFF"
    assert body["n_obs"] > 30
    assert body["verdict"] in {"cointegrated", "not_cointegrated", "insufficient-data"}
    assert isinstance(body["cointegrated"], bool)
    # The FRED summary fields should reflect our synthetic CSV (~5.0 level).
    assert body["fred_first"] is not None
    assert 4.0 <= body["fred_first"] <= 6.0
    assert body["fred_min"] <= body["fred_max"]


def test_strategies_fred_cointegration_unknown_factor(
    app_client: TestClient,
) -> None:
    """Unknown factor id should 400 *before* any FRED HTTP call is made."""
    r = app_client.post(
        "/strategies/fred-cointegration",
        json={
            "factor_id": "does_not_exist",
            "fred_series": "DGS10",
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    # Detail is now a structured dict with did_you_mean. Either the legacy
    # string lives in detail.error, or is the value itself for older paths.
    if isinstance(detail, dict):
        assert "unknown factor" in detail.get("error", "").lower()
    else:
        assert "unknown factor" in detail.lower()


@respx.mock
def test_strategies_fred_cointegration_bad_window(
    app_client: TestClient,
) -> None:
    """``start >= end`` returns 400 without hitting FRED."""
    r = app_client.post(
        "/strategies/fred-cointegration",
        json={
            "factor_id": "factor_a",
            "fred_series": "DFF",
            "start": "2025-12-15",
            "end": "2025-06-15",
        },
    )
    assert r.status_code == 400


def test_fred_cointegration_route_in_openapi(app_client: TestClient) -> None:
    r = app_client.get("/openapi.json")
    paths = set(r.json()["paths"].keys())
    assert "/strategies/fred-cointegration" in paths
