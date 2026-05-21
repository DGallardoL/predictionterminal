"""Tests for ``GET /pricing/binary/{slug}`` (W11-25).

The router under test ships at ``pfm.pricing.router`` and exposes the four
T81 binary pricers (``logit``, ``bsd``, ``brownian``, ``beta``) over HTTP.
It is not wired into the running app yet (main.py:routes has other active
claims), so each test mounts the router into a fresh ``FastAPI`` instance
and stages factor history via the monkeypatch hook the regression suite
uses (``pfm.regression_core._cached_factor_history``).
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import pricing
from pfm.cache import NullCache
from pfm.config import Settings, get_settings
from pfm.dependencies import (
    get_cache,
    get_factors_dep,
    get_polymarket_client,
)
from pfm.factors import FactorConfig
from pfm.pricing import router as pricing_router_mod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_factor(fid: str, slug: str, *, is_probability: bool = True) -> FactorConfig:
    return FactorConfig(
        id=fid,
        name=fid.replace("_", " ").title(),
        slug=slug,
        source="polymarket",
        description=f"Test factor {fid}",
        theme="test",
        is_probability=is_probability,
    )


@pytest.fixture
def factor_catalog() -> dict[str, FactorConfig]:
    """Small synthetic catalog with two anchor factors."""
    return {
        "fed": _make_factor("fed", "fed-rate-cut-march-2026"),
        "btc": _make_factor("btc", "btc-100k-by-2026"),
    }


@pytest.fixture
def history_bank() -> dict[str, pd.DataFrame]:
    """Synthetic 60-day price history per slug — daily UTC midnight index."""
    idx = pd.date_range("2026-03-01", periods=60, freq="D", tz="UTC")
    # Fed series rises 0.40 → 0.62 (current market price = 0.62).
    fed = pd.Series(np.linspace(0.40, 0.62, len(idx)), index=idx)
    # BTC series wobbles around 0.55 (current market price = 0.55).
    btc = pd.Series(np.linspace(0.50, 0.55, len(idx)), index=idx)

    def _wrap(s: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"price": s.values}, index=s.index)
        df.index.name = "date"
        return df

    return {
        "fed-rate-cut-march-2026": _wrap(fed),
        "btc-100k-by-2026": _wrap(btc),
    }


@pytest.fixture
def mock_factor_history(
    monkeypatch: pytest.MonkeyPatch, history_bank: dict[str, pd.DataFrame]
) -> dict[str, int]:
    """Patch ``_cached_factor_history`` to return our synthetic bank.

    Returns a dict that counts calls per slug — handy for the cache-hit
    assertion.
    """
    call_counter: dict[str, int] = {}

    def _fake_cached(fc, start, end, poly, cache, settings):
        call_counter[fc.slug] = call_counter.get(fc.slug, 0) + 1
        df = history_bank.get(fc.slug)
        if df is None:
            return pd.DataFrame()
        return df

    import pfm.regression_core as rc

    monkeypatch.setattr(rc, "_cached_factor_history", _fake_cached)
    return call_counter


@pytest.fixture
def client(factor_catalog: dict[str, FactorConfig]) -> Iterator[TestClient]:
    """Mount the pricing router on a fresh FastAPI app with DI overridden."""
    # Every test starts from a cold cache so order doesn't matter.
    pricing_router_mod._cache_clear()

    app = FastAPI()
    app.include_router(pricing_router_mod.router)

    fake_settings = Settings(
        polymarket_gamma_url="http://gamma.test",
        polymarket_clob_url="http://clob.test",
    )
    fake_poly = MagicMock()

    app.dependency_overrides[get_factors_dep] = lambda: factor_catalog
    app.dependency_overrides[get_cache] = lambda: NullCache()
    app.dependency_overrides[get_polymarket_client] = lambda: fake_poly
    app.dependency_overrides[get_settings] = lambda: fake_settings

    with TestClient(app) as c:
        yield c

    pricing_router_mod._cache_clear()


# ---------------------------------------------------------------------------
# Tests — model coverage (all four pricers)
# ---------------------------------------------------------------------------


def test_logit_model_returns_valid_envelope(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Default ``model=logit`` returns a well-formed PricingResponse."""
    r = client.get("/pricing/binary/fed-rate-cut-march-2026")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["slug"] == "fed-rate-cut-march-2026"
    assert payload["model"] == "logit"
    assert 0.0 <= payload["fair_price"] <= 1.0
    assert 0.0 <= payload["market_price"] <= 1.0
    assert isinstance(payload["confidence_interval"], list)
    assert len(payload["confidence_interval"]) == 2
    assert isinstance(payload["diagnostics"], dict)


def test_bsd_model_works(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Black-Scholes digital model returns a valid envelope."""
    r = client.get("/pricing/binary/fed-rate-cut-march-2026?model=bsd")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["model"] == "bsd"
    assert 0.0 <= payload["fair_price"] <= 1.0
    # BSD with no underlying degenerates to market_price — diagnostics
    # should expose the ``degenerate`` flag.
    assert "degenerate" in payload["diagnostics"] or "sigma" in payload["diagnostics"]


def test_brownian_model_works(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Brownian-bridge model returns a valid envelope."""
    r = client.get("/pricing/binary/fed-rate-cut-march-2026?model=brownian")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["model"] == "brownian"
    assert 0.0 <= payload["fair_price"] <= 1.0
    # Brownian-bridge always exposes ``sigma`` and ``tau_years``.
    assert "sigma" in payload["diagnostics"]
    assert "tau_years" in payload["diagnostics"]


def test_beta_model_works(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Beta-Binomial Bayes model returns a valid envelope."""
    r = client.get("/pricing/binary/fed-rate-cut-march-2026?model=beta")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["model"] == "beta"
    assert 0.0 <= payload["fair_price"] <= 1.0
    # Beta posterior always exposes alpha_post / beta_post.
    assert "alpha_post" in payload["diagnostics"]
    assert "beta_post" in payload["diagnostics"]


# ---------------------------------------------------------------------------
# Tests — error paths
# ---------------------------------------------------------------------------


def test_unknown_slug_returns_404(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Slug not in factor catalog → 404."""
    r = client.get("/pricing/binary/does-not-exist")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_invalid_model_returns_422(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """An unknown ``model=`` value must be rejected with 422 by Literal validation."""
    r = client.get("/pricing/binary/fed-rate-cut-march-2026?model=neuralnet")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Tests — invariants
# ---------------------------------------------------------------------------


def test_mispricing_equals_fair_minus_market(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """``mispricing == fair_price − market_price`` for every model."""
    for model_name in ("logit", "bsd", "brownian", "beta"):
        # Clear cache so each model gets a fresh compute.
        pricing_router_mod._cache_clear()
        r = client.get(f"/pricing/binary/fed-rate-cut-march-2026?model={model_name}")
        assert r.status_code == 200, (model_name, r.text)
        payload = r.json()
        expected = payload["fair_price"] - payload["market_price"]
        assert payload["mispricing"] == pytest.approx(expected, abs=1e-9), model_name


def test_confidence_interval_brackets_fair_or_collapses(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """CI low ≤ high and both in [0, 1]. CI may collapse to a point for degenerate states."""
    r = client.get("/pricing/binary/fed-rate-cut-march-2026?model=brownian")
    assert r.status_code == 200
    lo, hi = r.json()["confidence_interval"]
    assert 0.0 <= lo <= hi <= 1.0


def test_response_uses_default_model_when_omitted(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Omitting ``?model=`` defaults to logit."""
    r = client.get("/pricing/binary/fed-rate-cut-march-2026")
    assert r.status_code == 200
    assert r.json()["model"] == "logit"


# ---------------------------------------------------------------------------
# Tests — caching
# ---------------------------------------------------------------------------


def test_cache_hit_does_not_refetch(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Second call within TTL must NOT re-invoke ``_cached_factor_history``."""
    r1 = client.get("/pricing/binary/fed-rate-cut-march-2026?model=logit")
    assert r1.status_code == 200
    calls_after_first = sum(mock_factor_history.values())
    assert calls_after_first > 0

    r2 = client.get("/pricing/binary/fed-rate-cut-march-2026?model=logit")
    assert r2.status_code == 200
    calls_after_second = sum(mock_factor_history.values())
    assert calls_after_second == calls_after_first
    assert r1.json() == r2.json()


def test_cache_expiry_via_patched_clock(
    client: TestClient,
    mock_factor_history: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the patched clock advances past 60 s TTL, a refetch happens."""
    clock = [1000.0]
    monkeypatch.setattr(pricing_router_mod, "_PERF_COUNTER", lambda: clock[0])

    r1 = client.get("/pricing/binary/fed-rate-cut-march-2026?model=logit")
    assert r1.status_code == 200
    calls_after_first = sum(mock_factor_history.values())

    # Within TTL — no refetch.
    clock[0] = 1000.0 + pricing_router_mod._CACHE_TTL_S - 1.0
    client.get("/pricing/binary/fed-rate-cut-march-2026?model=logit")
    assert sum(mock_factor_history.values()) == calls_after_first

    # Past TTL — refetch fires.
    clock[0] = 1000.0 + pricing_router_mod._CACHE_TTL_S + 1.0
    client.get("/pricing/binary/fed-rate-cut-march-2026?model=logit")
    assert sum(mock_factor_history.values()) > calls_after_first


def test_cache_distinguishes_model(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Cache key includes the model; switching models triggers a recompute."""
    r_logit = client.get("/pricing/binary/fed-rate-cut-march-2026?model=logit")
    r_beta = client.get("/pricing/binary/fed-rate-cut-march-2026?model=beta")
    assert r_logit.status_code == 200
    assert r_beta.status_code == 200
    assert r_logit.json()["model"] == "logit"
    assert r_beta.json()["model"] == "beta"
    # Two distinct cache entries.
    assert len(pricing_router_mod._CACHE) == 2


# ---------------------------------------------------------------------------
# Tests — market_price extraction
# ---------------------------------------------------------------------------


def test_market_price_reflects_last_history_value(
    client: TestClient,
    mock_factor_history: dict[str, int],
    history_bank: dict[str, pd.DataFrame],
) -> None:
    """``market_price`` is the most-recent observation in the series."""
    r = client.get("/pricing/binary/fed-rate-cut-march-2026?model=logit")
    assert r.status_code == 200
    payload = r.json()
    expected_last = float(history_bank["fed-rate-cut-march-2026"]["price"].iloc[-1])
    assert payload["market_price"] == pytest.approx(expected_last, abs=1e-9)


def test_empty_history_degrades_gracefully(
    factor_catalog: dict[str, FactorConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When upstream returns an empty DataFrame, the route still returns 200 with market_price=0.5."""
    pricing_router_mod._cache_clear()

    def _fake_cached(fc, start, end, poly, cache, settings):
        return pd.DataFrame()

    import pfm.regression_core as rc

    monkeypatch.setattr(rc, "_cached_factor_history", _fake_cached)

    app = FastAPI()
    app.include_router(pricing_router_mod.router)
    fake_settings = Settings(
        polymarket_gamma_url="http://gamma.test",
        polymarket_clob_url="http://clob.test",
    )
    app.dependency_overrides[get_factors_dep] = lambda: factor_catalog
    app.dependency_overrides[get_cache] = lambda: NullCache()
    app.dependency_overrides[get_polymarket_client] = lambda: MagicMock()
    app.dependency_overrides[get_settings] = lambda: fake_settings

    with TestClient(app) as c:
        r = c.get("/pricing/binary/fed-rate-cut-march-2026?model=logit")
    assert r.status_code == 200
    assert r.json()["market_price"] == pytest.approx(0.5, abs=1e-9)
    pricing_router_mod._cache_clear()


# ---------------------------------------------------------------------------
# Tests — module-surface sanity
# ---------------------------------------------------------------------------


def test_pricing_package_exports_router_symbols() -> None:
    """The ``pfm.pricing`` package still exposes the model classes used by T81 callers."""
    # We don't import the router via the package __all__, but the package
    # must keep working — the router lives in a sibling module.
    assert hasattr(pricing, "RiskNeutralLogit")
    assert hasattr(pricing, "BlackScholesDigital")
    assert hasattr(pricing, "BrownianBridge")
    assert hasattr(pricing, "BetaBinomialBayes")
    assert hasattr(pricing, "MarketState")
    # Router lives at pfm.pricing.router (sibling, not re-exported on
    # purpose so the math package stays HTTP-free).
    assert pricing_router_mod.router.routes, "router should have at least one route"


def test_router_path_is_pricing_binary_slug() -> None:
    """The single mounted route is ``/pricing/binary/{slug}``."""
    paths = {r.path for r in pricing_router_mod.router.routes}
    assert "/pricing/binary/{slug}" in paths
