"""Tests for the A3 σ-gap router ``pfm.vol.pm_iv_router``."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.stats import norm

from pfm.cache_utils import get_cache
from pfm.dependencies import get_polymarket_client
from pfm.vol.pm_iv_extractor import LADDER_REGISTRY
from pfm.vol.pm_iv_router import router as pm_iv_router
from pfm.vol.vol_benchmarks import BINANCE_KLINES_URL, DERIBIT_INDEX_URL


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    get_cache("pm_iv_extractor").clear()
    get_cache("vol_benchmarks").clear()
    get_cache("pm_iv_gap").clear()
    yield
    get_cache("pm_iv_extractor").clear()
    get_cache("vol_benchmarks").clear()
    get_cache("pm_iv_gap").clear()


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------


def _lognormal_above_prob(strike: float, mu: float, sigma_t: float) -> float:
    return float(1.0 - norm.cdf((math.log(strike) - mu) / sigma_t))


def _t_years_to(maturity: datetime) -> float:
    return max((maturity - datetime.now(tz=UTC)).total_seconds() / (365.25 * 86_400.0), 1e-6)


def _build_above_probs(
    strikes: list[float],
    spot: float,
    sigma_annual: float,
    t_years: float,
) -> dict[float, float]:
    sigma_t = sigma_annual * math.sqrt(t_years)
    mu = math.log(spot) - 0.5 * sigma_t * sigma_t
    return {k: _lognormal_above_prob(k, mu, sigma_t) for k in strikes}


def _fred_csv(series_id: str, values: list[float]) -> str:
    base = datetime.now(tz=UTC).date() - timedelta(days=10)
    dates = [(base + timedelta(days=i)).isoformat() for i in range(len(values))]
    rows = "\n".join(f"{d},{v}" for d, v in zip(dates, values, strict=True))
    return f"DATE,{series_id}\n{rows}\n"


def _binance_klines_payload(closes: list[float]) -> list[list]:
    base_ts_ms = int(datetime.now(tz=UTC).timestamp() * 1000) - 86_400_000 * len(closes)
    day_ms = 86_400_000
    rows: list[list] = []
    for i, c in enumerate(closes):
        open_t = base_ts_ms + i * day_ms
        close_t = open_t + day_ms - 1
        rows.append(
            [
                open_t,
                str(c),
                str(c * 1.01),
                str(c * 0.99),
                str(c),
                "100.0",
                close_t,
                "1000.0",
                1000,
                "50.0",
                "500.0",
                "0",
            ]
        )
    return rows


class _SlugMidpoints:
    def __init__(self, midpoints: dict[str, float]) -> None:
        self.midpoints = midpoints

    def get_market_metadata(self, slug: str) -> dict[str, Any]:
        p = self.midpoints.get(slug, 0.5)
        return {"bestBid": p - 0.005, "bestAsk": p + 0.005, "lastTradePrice": p}


def _make_app(client_stub: _SlugMidpoints) -> FastAPI:
    """Mount the router on a fresh FastAPI() with a polymarket DI override."""
    app = FastAPI()
    app.include_router(pm_iv_router)
    app.dependency_overrides[get_polymarket_client] = lambda: client_stub
    return app


_BTC_LADDER_SLUGS_FULL: dict[float, str] = {
    90_000.0: "will-bitcoin-reach-90000-by-december-31-2026-113-862-581",
    100_000.0: "will-bitcoin-reach-100000-by-december-31-2026-571-361-361",
    140_000.0: "will-bitcoin-reach-140000-by-december-31-2026-131-829-299",
    150_000.0: "will-bitcoin-reach-150000-by-december-31-2026-557-246-971",
    160_000.0: "will-bitcoin-reach-160000-by-december-31-2026-934-934-164",
    190_000.0: "will-bitcoin-reach-190000-by-december-31-2026-936-485-627",
    200_000.0: "will-bitcoin-reach-200000-by-december-31-2026-752-232-389",
    250_000.0: "will-bitcoin-reach-250000-by-december-31-2026-579-442",
    500_000.0: "will-bitcoin-reach-500000-by-december-31-2026-864",
    1_000_000.0: "will-bitcoin-reach-1000000-by-december-31-2026-946",
}


def _btc_client(sigma_annual: float = 0.55) -> _SlugMidpoints:
    """10-strike BTC above-ladder stub at EOY-2026 (resolves 2027-01-01).

    Mirrors the live LADDER_REGISTRY shape after the 2026-05-15 discovery
    sweep replaced the dead `btc-above-Xk-eoy-2026` slugs with the long-form
    `will-bitcoin-reach-…-by-december-31-2026` markets.
    """
    maturity = datetime(2027, 1, 1, 0, 0, tzinfo=UTC)
    t_years = _t_years_to(maturity)
    strikes = list(_BTC_LADDER_SLUGS_FULL.keys())
    probs = _build_above_probs(strikes, spot=110_000.0, sigma_annual=sigma_annual, t_years=t_years)
    midpoints = {_BTC_LADDER_SLUGS_FULL[k]: probs[k] for k in strikes}
    return _SlugMidpoints(midpoints)


# ---------------------------------------------------------------------------
# Endpoint smoke tests
# ---------------------------------------------------------------------------


def test_get_assets_returns_known_list() -> None:
    """GET /vol/pm-iv/assets returns the LADDER_REGISTRY keys."""
    app = _make_app(_SlugMidpoints({}))
    with TestClient(app) as client:
        resp = client.get("/vol/pm-iv/assets")
        assert resp.status_code == 200
        body = resp.json()
        assert "assets" in body
        assert set(body["assets"]) == set(LADDER_REGISTRY.keys())
        assert body["count"] == len(LADDER_REGISTRY)


def test_get_pm_iv_for_btc_returns_pmivresult() -> None:
    """GET /vol/pm-iv/BTC returns 200 with a valid PMIVResult shape.

    Previously SPX (5 strikes) — SPX was dropped from LADDER_REGISTRY on
    2026-05-15 after discovery found no live SPX ladder. BTC EOY-2026 is the
    safest replacement (6 live above-strike rungs).
    """
    app = _make_app(_btc_client(sigma_annual=0.55))
    with TestClient(app) as client:
        resp = client.get("/vol/pm-iv/BTC")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["asset"] == "BTC"
        assert body["sigma_annual"] > 0.0
        assert body["n_strikes"] == 10
        # The sigma_annual field name is on PMIVResult, not PMIVGapSnapshot
        assert "sigma_method" in body
        assert "fitted_mean" in body
        # Some warnings might fire (bootstrap noise, monotonicity violations).
        assert isinstance(body["warnings"], list)


@respx.mock
def test_get_pm_iv_gap_for_btc_returns_snapshot() -> None:
    """GET /vol/pm-iv/gap/BTC returns a PMIVGapSnapshot with benchmark data."""
    # DVOL = 60%
    respx.get(DERIBIT_INDEX_URL).mock(
        return_value=httpx.Response(200, json={"result": {"index_price": 60.0}})
    )
    # Realized leg
    rng = np.random.default_rng(11)
    closes = [50_000.0]
    for r in rng.normal(0.0, 0.03, size=30):
        closes.append(closes[-1] * math.exp(r))
    respx.get(BINANCE_KLINES_URL).mock(
        return_value=httpx.Response(200, json=_binance_klines_payload(closes))
    )

    app = _make_app(_btc_client(sigma_annual=0.55))
    with TestClient(app) as client:
        resp = client.get("/vol/pm-iv/gap/BTC")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["asset"] == "BTC"
        assert body["sigma_pm"] > 0.0
        # BTC above-ladder now has 10 strikes after the 2026-05-15 slug refresh.
        assert body["sigma_pm_n_strikes"] == 10
        assert "benchmarks" in body and "dvol" in body["benchmarks"]
        assert body["primary_benchmark"] == "dvol"
        assert body["signal"] in ("pm_richer", "benchmark_richer", "flat")
        assert "gaps" in body and "dvol" in body["gaps"]


def test_get_pm_iv_for_unknown_returns_404() -> None:
    """GET /vol/pm-iv/UNKNOWN returns 404 for an asset not in the registry."""
    app = _make_app(_SlugMidpoints({}))
    with TestClient(app) as client:
        resp = client.get("/vol/pm-iv/UNKNOWN")
        assert resp.status_code == 404
        assert "UNKNOWN" in resp.json()["detail"]


def test_get_pm_iv_gap_for_unknown_returns_404() -> None:
    """GET /vol/pm-iv/gap/UNKNOWN returns 404 for an unknown asset."""
    app = _make_app(_SlugMidpoints({}))
    with TestClient(app) as client:
        resp = client.get("/vol/pm-iv/gap/UNKNOWN")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Feature-flag verification
# ---------------------------------------------------------------------------


def test_feature_flag_off_means_router_not_mounted() -> None:
    """Router exists as an importable APIRouter with the expected paths.

    main.py only mounts ``pm_iv_router`` when ``PFM_VOL_PM_IV_ENABLED=1`` —
    this test asserts the router itself is well-formed (3 paths under the
    expected prefix) so a future gate misfire would be caught.
    """
    paths = {route.path for route in pm_iv_router.routes}
    # APIRouter routes carry the prefix already
    assert "/vol/pm-iv/assets" in paths
    assert "/vol/pm-iv/{asset}" in paths
    assert "/vol/pm-iv/gap/{asset}" in paths
    assert len(paths) == 3
