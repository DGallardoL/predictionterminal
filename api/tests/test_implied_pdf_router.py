"""Standalone tests for the implied-PDF Terminal router.

Everything external is mocked: the router is mounted on a *fresh* ``FastAPI()``
app, ``get_kalshi_client`` is overridden with a dummy, and the two indirection
seams ``_discover`` / ``_compute`` are monkeypatched to return canned objects.
We build *real* :class:`LadderFamily` / :class:`ImpliedPDFResult` instances so
the endpoint's ``response_model`` validation is genuinely exercised.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.dependencies import get_kalshi_client
from pfm.vol import implied_pdf_router as router_mod
from pfm.vol.implied_pdf_schemas import (
    ImpliedPDFResult,
    LadderEntry,
    LadderFamily,
    MarketPoint,
    Moments,
    Quantiles,
)

# ---------------------------------------------------------------------------
# Canned domain objects
# ---------------------------------------------------------------------------


def _make_family() -> LadderFamily:
    """Build a minimal but valid terminal-bucket ladder family."""
    return LadderFamily(
        asset="SPX",
        asset_class="equity_index",
        data_shape="terminal_buckets",
        maturity_utc=datetime(2026, 5, 15, 16, 0, tzinfo=UTC),
        spot=5300.0,
        entries=[
            LadderEntry(direction="between", prob=0.3, floor=5200.0, cap=5300.0),
            LadderEntry(direction="between", prob=0.4, floor=5300.0, cap=5400.0),
            LadderEntry(direction="between", prob=0.3, floor=5400.0, cap=5500.0),
        ],
        source="kalshi:KXINX-26MAY15H1600",
    )


def _make_result() -> ImpliedPDFResult:
    """Build a minimal but valid ImpliedPDFResult."""
    grid = [5200.0, 5300.0, 5400.0, 5500.0]
    return ImpliedPDFResult(
        asset="SPX",
        data_shape="terminal_buckets",
        distribution_of="terminal_price",
        maturity_utc=datetime(2026, 5, 15, 16, 0, tzinfo=UTC),
        time_to_maturity_years=0.05,
        spot=5300.0,
        grid=grid,
        pdf=[0.001, 0.004, 0.004, 0.001],
        cdf=[0.1, 0.4, 0.8, 1.0],
        market_points=[
            MarketPoint(k=5250.0, prob=0.3, kind="mass", floor=5200.0, cap=5300.0),
        ],
        moments=Moments(mean=5350.0, median=5345.0, mode=5340.0, std=80.0, skew=0.1, kurtosis=0.2),
        quantiles=Quantiles(p5=5220.0, p25=5290.0, p50=5345.0, p75=5400.0, p95=5470.0),
        method="pchip_monotone",
        eps=0.01,
        n_strikes=3,
    )


class _DummyKalshi:
    """Stand-in Kalshi client; the seams are monkeypatched so it's never used."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Reset the process-wide cache before each test to avoid cross-talk."""
    reset_caches()


@pytest.fixture
def client() -> TestClient:
    """Fresh FastAPI app mounting only the implied-PDF router."""
    app = FastAPI()
    app.include_router(router_mod.router)
    app.dependency_overrides[get_kalshi_client] = lambda: _DummyKalshi()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_assets_returns_registry(client: TestClient) -> None:
    resp = client.get("/terminal/implied-pdf/assets")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == len(router_mod.SUPPORTED)
    keys = {a["asset"] for a in body["assets"]}
    assert keys == set(router_mod.SUPPORTED)
    spx = next(a for a in body["assets"] if a["asset"] == "SPX")
    assert spx["venue"] == "kalshi"
    assert spx["default_shape"] == "terminal_buckets"
    assert "name" in spx


def test_implied_pdf_happy_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_mod, "_discover", lambda *a, **k: _make_family())
    monkeypatch.setattr(router_mod, "_compute", lambda family, **k: _make_result())

    resp = client.get("/terminal/implied-pdf/SPX")
    assert resp.status_code == 200
    body = resp.json()
    for field in ("grid", "pdf", "cdf", "moments", "quantiles", "data_shape"):
        assert field in body
    assert body["data_shape"] == "terminal_buckets"
    assert body["asset"] == "SPX"
    assert len(body["grid"]) == len(body["pdf"]) == len(body["cdf"])


def test_lowercase_asset_normalised(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_mod, "_discover", lambda *a, **k: _make_family())
    monkeypatch.setattr(router_mod, "_compute", lambda family, **k: _make_result())
    resp = client.get("/terminal/implied-pdf/spx")
    assert resp.status_code == 200


def test_unknown_asset_404(client: TestClient) -> None:
    resp = client.get("/terminal/implied-pdf/DOGE")
    assert resp.status_code == 404


def test_empty_asset_422(client: TestClient) -> None:
    # A whitespace-only path segment reaches the handler and trips _validate_asset.
    resp = client.get("/terminal/implied-pdf/%20")
    assert resp.status_code == 422


def test_compute_value_error_422(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_mod, "_discover", lambda *a, **k: _make_family())

    def _boom(family, **k):
        raise ValueError("degenerate ladder")

    monkeypatch.setattr(router_mod, "_compute", _boom)
    resp = client.get("/terminal/implied-pdf/SPX")
    assert resp.status_code == 422
    assert "degenerate ladder" in resp.json()["detail"]


def test_discover_failure_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("kalshi down")

    monkeypatch.setattr(router_mod, "_discover", _boom)
    resp = client.get("/terminal/implied-pdf/SPX")
    assert resp.status_code == 502
    assert "kalshi" in resp.json()["detail"].lower()


def test_discover_none_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_mod, "_discover", lambda *a, **k: None)
    resp = client.get("/terminal/implied-pdf/SPX")
    assert resp.status_code == 404


def test_eps_out_of_range_422(client: TestClient) -> None:
    resp = client.get("/terminal/implied-pdf/SPX", params={"eps": 0.5})
    assert resp.status_code == 422


def test_grid_size_out_of_range_422(client: TestClient) -> None:
    resp = client.get("/terminal/implied-pdf/SPX", params={"grid_size": 4})
    assert resp.status_code == 422


def test_query_params_forwarded(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _capture_discover(asset_key, cli, *, maturity_filter=None, prefer_shape=None):
        captured["maturity_filter"] = maturity_filter
        captured["prefer_shape"] = prefer_shape
        return _make_family()

    def _capture_compute(family, **kwargs):
        captured.update(kwargs)
        return _make_result()

    monkeypatch.setattr(router_mod, "_discover", _capture_discover)
    monkeypatch.setattr(router_mod, "_compute", _capture_compute)

    resp = client.get(
        "/terminal/implied-pdf/SPX",
        params={
            "maturity": "2026-05-15",
            "shape": "terminal_ladder",
            "method": "lognormal",
            "eps": 0.02,
            "grid_size": 128,
            "barrier_to_terminal": True,
            "tail_model": "linear",
        },
    )
    assert resp.status_code == 200
    assert captured["maturity_filter"] == "2026-05-15"
    assert captured["prefer_shape"] == "terminal_ladder"
    assert captured["method"] == "lognormal"
    assert captured["eps"] == 0.02
    assert captured["grid_size"] == 128
    assert captured["barrier_to_terminal"] is True
    assert captured["tail_model"] == "linear"


def test_default_shape_used_when_omitted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _capture_discover(asset_key, cli, *, maturity_filter=None, prefer_shape=None):
        captured["prefer_shape"] = prefer_shape
        return _make_family()

    monkeypatch.setattr(router_mod, "_discover", _capture_discover)
    monkeypatch.setattr(router_mod, "_compute", lambda family, **k: _make_result())

    resp = client.get("/terminal/implied-pdf/SPX")
    assert resp.status_code == 200
    assert captured["prefer_shape"] == "terminal_buckets"


def test_cache_hit_skips_second_discover(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"discover": 0, "compute": 0}

    def _count_discover(*a, **k):
        calls["discover"] += 1
        return _make_family()

    def _count_compute(family, **k):
        calls["compute"] += 1
        return _make_result()

    monkeypatch.setattr(router_mod, "_discover", _count_discover)
    monkeypatch.setattr(router_mod, "_compute", _count_compute)

    first = client.get("/terminal/implied-pdf/SPX")
    second = client.get("/terminal/implied-pdf/SPX")
    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["discover"] == 1
    assert calls["compute"] == 1
    assert first.json() == second.json()
