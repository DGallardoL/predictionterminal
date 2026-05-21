"""Router tests for /terminal/pricing-kernel/* (compute seam mocked)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from pfm.vol import pricing_kernel_router as pkr
from pfm.vol.pricing_kernel import CrossVenue, DensitySeries, PricingKernelResult


def _fake_result(asset: str = "SPX") -> PricingKernelResult:
    grid = [7000.0, 7350.0, 7700.0]
    dens = [
        DensitySeries(
            label="Kalshi",
            measure="risk_neutral",
            venue="kalshi",
            pdf=[0.0, 1.0, 0.0],
            cdf=[0.0, 0.5, 1.0],
            mean=7350.0,
            std=110.0,
        ),
        DensitySeries(
            label="Options",
            measure="risk_neutral",
            venue="options",
            pdf=[0.0, 1.0, 0.0],
            cdf=[0.0, 0.5, 1.0],
            mean=7350.0,
            std=60.0,
        ),
        DensitySeries(
            label="Physical",
            measure="physical",
            venue="garch",
            pdf=[0.0, 1.0, 0.0],
            cdf=[0.0, 0.5, 1.0],
            mean=7350.0,
            std=50.0,
        ),
    ]
    return PricingKernelResult(
        asset=asset,
        spot=7350.0,
        forward=7350.0,
        maturity_utc=datetime(2026, 5, 20, 20, 0, tzinfo=UTC),
        time_to_maturity_years=0.0021,
        risk_free=0.045,
        annual_drift=0.06,
        grid=grid,
        densities=dens,
        cross_venue=CrossVenue(
            kl_kalshi_given_options=0.6,
            kl_options_given_kalshi=0.36,
            jensen_shannon=0.1,
            mean_gap=3.0,
            std_ratio=1.75,
            ratio=[0.0, 1.8, 0.0],
        ),
        pricing_kernel=[None, 1.2, None],
        implied_risk_aversion=[None, 0.01, None],
        relative_risk_aversion=[None, 4.2, None],
        variance_risk_premium=0.012,
        warnings=[],
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(pkr, "_compute", lambda asset, c, **kw: _fake_result(asset))
    from pfm.main import app

    return TestClient(app)


def test_assets_endpoint(client):
    r = client.get("/terminal/pricing-kernel/assets")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert {a["asset"] for a in body["assets"]} == {"SPX", "NDX"}


def test_pricing_kernel_spx(client):
    r = client.get("/terminal/pricing-kernel/SPX")
    assert r.status_code == 200
    d = r.json()
    assert d["asset"] == "SPX"
    assert len(d["densities"]) == 3
    assert d["cross_venue"]["std_ratio"] == pytest.approx(1.75)
    assert d["variance_risk_premium"] == pytest.approx(0.012)


def test_unknown_asset_404(client):
    assert client.get("/terminal/pricing-kernel/DOGE").status_code == 404


def test_compute_value_error_is_422(monkeypatch):
    def _boom(asset, c, **kw):
        raise ValueError("only 2 usable OTM IVs")

    monkeypatch.setattr(pkr, "_compute", _boom)
    from pfm.main import app

    # distinct grid_size → distinct cache key, so we don't hit a prior success
    r = TestClient(app).get("/terminal/pricing-kernel/SPX?grid_size=128")
    assert r.status_code == 422


def test_compute_upstream_error_is_502(monkeypatch):
    def _boom(asset, c, **kw):
        raise RuntimeError("kalshi down")

    monkeypatch.setattr(pkr, "_compute", _boom)
    from pfm.main import app

    r = TestClient(app).get("/terminal/pricing-kernel/SPX?grid_size=200")
    assert r.status_code == 502
