"""Tests for the extended FRED catalog and ``/macro/fred`` router.

These tests verify:
    * 20-series catalog completeness (6 original + 14 wave-10 additions)
    * Pydantic schema for series metadata
    * The ``/macro/fred/catalog`` and ``/macro/fred/series/{id}`` endpoints
    * Each new series can be parsed and round-tripped through the fetcher
"""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.sources.fred import (
    _SERIES_REGISTRY,
    FREDGRAPH_BASE,
    SUPPORTED_SERIES,
    FredSeriesMetadata,
    fetch_fred_series,
    list_catalog,
)
from pfm.sources.fred import (
    router as fred_router,
)

WAVE_10_NEW_SERIES = [
    "ICSA",
    "CCSA",
    "PAYEMS",
    "MANEMP",
    "PERMIT",
    "HOUST",
    "RSXFS",
    "INDPRO",
    "T10Y2Y",
    "BAMLH0A0HYM2",
    "DCOILWTICO",
    "GOLDAMGBD228NLBM",
    "DEXUSEU",
    "DEXJPUS",
]

ORIGINAL_6 = ["DFF", "DGS2", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS"]


@pytest.fixture(autouse=True)
def _clear_caches():
    reset_caches()
    yield
    reset_caches()


def _build_csv(series_id: str, start: str, end: str) -> str:
    """Build a daily CSV (header + every UTC date) with synthetic values."""
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    lines = [f"DATE,{series_id}"]
    for i, ts in enumerate(idx):
        lines.append(f"{ts.strftime('%Y-%m-%d')},{100.0 + 0.1 * i:.4f}")
    return "\n".join(lines) + "\n"


# --- registry --------------------------------------------------------------


def test_registry_has_20_series() -> None:
    # 20 base series (wave-10) + 2 vol-benchmark series (OVXCLS, GVZCLS).
    assert len(_SERIES_REGISTRY) == 22, f"expected 22 series, got {len(_SERIES_REGISTRY)}"


def test_registry_contains_originals() -> None:
    for sid in ORIGINAL_6:
        assert sid in _SERIES_REGISTRY


def test_registry_contains_wave10_additions() -> None:
    for sid in WAVE_10_NEW_SERIES:
        assert sid in _SERIES_REGISTRY, f"missing wave-10 series {sid}"


def test_registry_entries_have_required_fields() -> None:
    required = {"name", "frequency", "units", "desc", "citation"}
    for sid, meta in _SERIES_REGISTRY.items():
        missing = required - meta.keys()
        assert not missing, f"{sid} missing fields: {missing}"
        assert meta["citation"].startswith("https://fred.stlouisfed.org/")


def test_supported_series_alias_consistent() -> None:
    """``SUPPORTED_SERIES`` is the legacy name — must keep the same keys."""
    assert set(SUPPORTED_SERIES.keys()) == set(_SERIES_REGISTRY.keys())


def test_pydantic_metadata_schema() -> None:
    catalog = list_catalog()
    assert len(catalog) == 22
    for entry in catalog:
        assert isinstance(entry, FredSeriesMetadata)
        # Round-trip through dict so we know the schema serializes cleanly.
        d = entry.model_dump()
        assert d["series_id"] in _SERIES_REGISTRY
        assert d["citation"].startswith("https://")


# --- fetcher with new series ------------------------------------------------


@pytest.mark.parametrize("series_id", WAVE_10_NEW_SERIES)
@respx.mock
def test_each_new_series_fetches_through(series_id: str) -> None:
    """Each new series should fetch and parse without complaint."""
    csv = _build_csv(series_id, "2025-01-01", "2025-01-10")
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=csv))
    s = fetch_fred_series(
        series_id,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-10", tz="UTC"),
    )
    assert s.name == series_id
    assert len(s) == 10
    assert s.iloc[0] == pytest.approx(100.0)


# --- router endpoints -------------------------------------------------------


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(fred_router)
    return TestClient(app)


def test_catalog_endpoint_lists_20() -> None:
    client = _make_app()
    r = client.get("/macro/fred/catalog")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 22
    ids = {s["series_id"] for s in body["series"]}
    assert "DFF" in ids
    assert "ICSA" in ids
    assert "BAMLH0A0HYM2" in ids
    assert "OVXCLS" in ids
    assert "GVZCLS" in ids


@respx.mock
def test_series_endpoint_happy_path() -> None:
    csv = _build_csv("ICSA", "2025-01-01", "2025-01-05")
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=csv))
    client = _make_app()
    r = client.get("/macro/fred/series/ICSA?start=2025-01-01&end=2025-01-05")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["series_id"] == "ICSA"
    assert body["frequency"] == "weekly"
    assert len(body["data"]) == 5


def test_series_endpoint_unknown_404() -> None:
    client = _make_app()
    r = client.get("/macro/fred/series/NOTAREALSERIES?start=2025-01-01&end=2025-01-05")
    assert r.status_code == 404


def test_series_endpoint_bad_window_400() -> None:
    client = _make_app()
    r = client.get("/macro/fred/series/DFF?start=2025-12-31&end=2025-01-01")
    assert r.status_code == 400
