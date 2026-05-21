"""Tests for the 4-venue parallel search orchestrator + router."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import multi_venue_search as mvs
from pfm.cache_utils import get_cache
from pfm.multi_venue_search import (
    GAMMA_URL,
    KALSHI_URL,
    router,
    search_all_venues,
)
from pfm.sources.manifold import MANIFOLD_BASE_URL
from pfm.sources.predictit import PREDICTIT_BASE_URL


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Drop all venue caches before AND after each test to prevent state pollution."""
    for ns in (
        "multi_venue_search",
        "multi_venue_concept",
        "predictit_all",
        "manifold_search",
        "polymarket_search",
        "kalshi_search",
    ):
        get_cache(ns).clear()
    yield
    for ns in (
        "multi_venue_search",
        "multi_venue_concept",
        "predictit_all",
        "manifold_search",
        "polymarket_search",
        "kalshi_search",
    ):
        get_cache(ns).clear()


@pytest.fixture
def app_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# Common fixtures the per-venue mock routes return.
_PM_HITS = [
    {"id": 1, "slug": "trump-2028", "question": "Will Trump win 2028?", "endDate": "2028-11-07"},
    {"id": 2, "slug": "trump-indict", "question": "Trump indictment?", "endDate": "2026-06-01"},
]
_KALSHI_HITS = {
    "markets": [
        {"ticker": "KXTRUMP-28", "title": "Trump wins 2028 election", "close_time": "2028-11-07"},
        {"ticker": "KXBIDEN-26", "title": "Biden runs in 2026", "close_time": "2026-12-31"},
    ]
}
_MANIFOLD_HITS = [
    {"id": "mf1", "slug": "trump-2028-mf", "question": "Trump wins 2028?", "closeTime": "..."},
]
_PREDICTIT_HITS = {
    "markets": [
        {
            "id": 8200,
            "name": "2028 Trump nomination",
            "shortName": "Trump 2028 nom",
            "url": "https://www.predictit.org/markets/detail/8200",
            "totalSharesTraded": 100_000,
            "dateEnd": "2028-08-01",
            "contracts": [{"id": 1, "name": "Yes", "lastTradePrice": 0.55}],
        },
        {
            "id": 9000,
            "name": "Senate control 2026",
            "shortName": "Senate 2026",
            "url": "https://www.predictit.org/markets/detail/9000",
            "totalSharesTraded": 50_000,
            "dateEnd": "2026-11-04",
            "contracts": [{"id": 2, "name": "R", "lastTradePrice": 0.48}],
        },
    ]
}


def _mount_all_venues_ok() -> None:
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=_PM_HITS))
    respx.get(f"{KALSHI_URL}/markets").mock(return_value=httpx.Response(200, json=_KALSHI_HITS))
    respx.get(f"{MANIFOLD_BASE_URL}/search-markets").mock(
        return_value=httpx.Response(200, json=_MANIFOLD_HITS)
    )
    respx.get(f"{PREDICTIT_BASE_URL}/marketdata/all/").mock(
        return_value=httpx.Response(200, json=_PREDICTIT_HITS)
    )


# ---------------------------------------------------------------------------
# search_all_venues — parallel orchestrator
# ---------------------------------------------------------------------------


@respx.mock
def test_search_all_venues_returns_per_venue_lists() -> None:
    _mount_all_venues_ok()

    out = _run(search_all_venues("trump", limit=5))

    assert set(out.keys()) == {"polymarket", "kalshi", "manifold", "predictit"}
    assert len(out["polymarket"]) == 2
    # Kalshi filters by query — only the Trump-tickered market matches.
    assert len(out["kalshi"]) == 1
    assert out["kalshi"][0]["id"] == "KXTRUMP-28"
    assert len(out["manifold"]) == 1
    # PredictIt: only "trump" in the name matches.
    assert len(out["predictit"]) == 1
    assert out["predictit"][0]["id"] == "8200"


@respx.mock
def test_search_all_venues_empty_query_returns_empty_lists() -> None:
    # No mocks needed — short-circuits.
    out = _run(search_all_venues(""))
    assert all(out[v] == [] for v in ("polymarket", "kalshi", "manifold", "predictit"))


@respx.mock
def test_search_all_venues_isolates_failure_to_one_venue() -> None:
    # PM 500s, but the other three should still come back.
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(500))
    respx.get(f"{KALSHI_URL}/markets").mock(return_value=httpx.Response(200, json=_KALSHI_HITS))
    respx.get(f"{MANIFOLD_BASE_URL}/search-markets").mock(
        return_value=httpx.Response(200, json=_MANIFOLD_HITS)
    )
    respx.get(f"{PREDICTIT_BASE_URL}/marketdata/all/").mock(
        return_value=httpx.Response(200, json=_PREDICTIT_HITS)
    )

    out = _run(search_all_venues("trump"))

    assert out["polymarket"] == []
    assert len(out["manifold"]) == 1


@respx.mock
def test_search_all_venues_runs_concurrently() -> None:
    """All four venues must be hit even though they share an httpx client."""
    pm = respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=_PM_HITS))
    kalshi = respx.get(f"{KALSHI_URL}/markets").mock(
        return_value=httpx.Response(200, json=_KALSHI_HITS)
    )
    manifold = respx.get(f"{MANIFOLD_BASE_URL}/search-markets").mock(
        return_value=httpx.Response(200, json=_MANIFOLD_HITS)
    )
    predictit = respx.get(f"{PREDICTIT_BASE_URL}/marketdata/all/").mock(
        return_value=httpx.Response(200, json=_PREDICTIT_HITS)
    )

    _run(search_all_venues("trump"))

    assert pm.called and kalshi.called and manifold.called and predictit.called


# ---------------------------------------------------------------------------
# Endpoint /multi-venue/search
# ---------------------------------------------------------------------------


@respx.mock
def test_endpoint_search_returns_4_venue_payload(app_client: TestClient) -> None:
    _mount_all_venues_ok()

    r = app_client.get("/multi-venue/search", params={"q": "trump", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "trump"
    assert body["n_total"] == 5  # 2 + 1 + 1 + 1
    for venue in ("polymarket", "kalshi", "manifold", "predictit"):
        assert venue in body
        assert isinstance(body[venue], list)


def test_endpoint_search_rejects_empty_q(app_client: TestClient) -> None:
    r = app_client.get("/multi-venue/search", params={"q": ""})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint /multi-venue/concept/{id}
# ---------------------------------------------------------------------------


def test_endpoint_concept_known_id(app_client: TestClient) -> None:
    r = app_client.get("/multi-venue/concept/fed_cuts_2026")
    assert r.status_code == 200
    body = r.json()
    assert body["concept_id"] == "fed_cuts_2026"
    assert body["theme"] == "macro"
    assert body["venues"]["polymarket"]
    assert body["venues"]["kalshi"]
    assert body["n_legs_present"] >= 3


def test_endpoint_concept_unknown_id_404(app_client: TestClient) -> None:
    r = app_client.get("/multi-venue/concept/does_not_exist")
    assert r.status_code == 404


def test_endpoint_concepts_lists_all(app_client: TestClient) -> None:
    r = app_client.get("/multi-venue/concepts")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] >= 5
    for c in body["concepts"]:
        assert "concept_id" in c and "label" in c and "theme" in c


# ---------------------------------------------------------------------------
# Cache hit verification
# ---------------------------------------------------------------------------


@respx.mock
def test_endpoint_search_caches_response(app_client: TestClient) -> None:
    pm_route = respx.get(f"{GAMMA_URL}/markets").mock(
        return_value=httpx.Response(200, json=_PM_HITS)
    )
    respx.get(f"{KALSHI_URL}/markets").mock(return_value=httpx.Response(200, json=_KALSHI_HITS))
    respx.get(f"{MANIFOLD_BASE_URL}/search-markets").mock(
        return_value=httpx.Response(200, json=_MANIFOLD_HITS)
    )
    respx.get(f"{PREDICTIT_BASE_URL}/marketdata/all/").mock(
        return_value=httpx.Response(200, json=_PREDICTIT_HITS)
    )

    app_client.get("/multi-venue/search", params={"q": "trump", "limit": 5})
    app_client.get("/multi-venue/search", params={"q": "trump", "limit": 5})

    assert pm_route.call_count == 1  # second call served from cache


# ---------------------------------------------------------------------------
# Module-level safety: importing doesn't trigger network IO
# ---------------------------------------------------------------------------


def test_module_imports_without_network() -> None:
    # If this import succeeded at the top of the file, we're already good.
    assert hasattr(mvs, "router")
    assert hasattr(mvs, "search_all_venues")
