"""Tests for ``pfm.terminal_gdelt_news``. All HTTP is mocked via respx."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_gdelt_news import _CACHE, GDELT_DOC_URL
from pfm.terminal_gdelt_news import router as gdelt_router

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    _CACHE.clear()
    yield
    _CACHE.clear()


@pytest.fixture
def app_client() -> TestClient:
    app = FastAPI()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    app.include_router(gdelt_router)
    return TestClient(app)


def _gdelt_payload() -> dict:
    """Realistic GDELT 2.0 ArtList JSON shape."""
    return {
        "articles": [
            {
                "url": "https://www.bbc.com/news/world-1",
                "title": "Trump faces impeachment vote in Senate",
                "domain": "bbc.com",
                "sourcecountry": "United Kingdom",
                "language": "English",
                "seendate": "20260304T171500Z",
                "socialimage": "https://example.com/img1.jpg",
                "tone": -3.42,
            },
            {
                "url": "https://www.reuters.com/politics/2",
                "title": "Senate to consider Trump resign call after hearing",
                "domain": "reuters.com",
                "sourcecountry": "United States",
                "language": "English",
                "seendate": "20260304T180000Z",
                "socialimage": "https://example.com/img2.jpg",
                "tone": 1.10,
            },
            {
                "url": "https://www.bbc.com/news/world-3",
                "title": "Analysts split on Trump presidency outlook",
                "domain": "bbc.com",
                "sourcecountry": "United Kingdom",
                "language": "English",
                "seendate": "20260304T183000Z",
                "socialimage": "",
                "tone": 0.0,
            },
        ]
    }


@respx.mock
def test_per_slug_endpoint_aggregates_tone_and_top_sources(
    app_client: TestClient,
) -> None:
    respx.get(f"{GAMMA}/markets", params={"slug": "trump-out-by-2027"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "trump-out-by-2027",
                    "question": "Will Trump resign or be impeached by 2027?",
                }
            ],
        )
    )
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload()))

    resp = app_client.get("/terminal/gdelt/trump-out-by-2027?limit=20")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["slug"] == "trump-out-by-2027"
    # Keywords extracted from "Will Trump resign or be impeached by 2027?"
    assert "trump" in body["query_used"].lower()
    assert body["n_articles"] == 3

    # Mean tone = (-3.42 + 1.10 + 0.0) / 3 ≈ -0.7733
    assert body["mean_tone"] == pytest.approx(-0.7733, abs=1e-3)

    # bbc.com appears twice → top source.
    assert body["top_sources"][0]["source"] == "bbc.com"
    assert body["top_sources"][0]["n_articles"] == 2
    sources = {s["source"] for s in body["top_sources"]}
    assert sources == {"bbc.com", "reuters.com"}

    # dominant_topic should be a non-stopword token from titles.
    assert body["dominant_topic"]
    assert body["dominant_topic"] not in {"the", "and", "with"}

    # Spot-check first article schema. Backward-compatible: legacy keys
    # must still be present; new ``relevance_score`` / ``matched_terms``
    # additive fields are allowed.
    a0 = body["articles"][0]
    expected_keys = {"url", "title", "source", "country", "ts", "tone", "language", "image_url"}
    assert expected_keys <= set(a0.keys())
    # Articles are now relevance-sorted with recency as tie-breaker, so
    # the *exact* order is no longer "as returned by upstream". Verify
    # the seendate→ISO conversion works on whichever article surfaces.
    all_ts = {a["ts"] for a in body["articles"]}
    assert "2026-03-04T17:15:00Z" in all_ts
    all_countries = {a["country"] for a in body["articles"]}
    assert "United Kingdom" in all_countries


@respx.mock
def test_per_slug_endpoint_404s_when_market_missing(app_client: TestClient) -> None:
    respx.get(f"{GAMMA}/markets", params={"slug": "ghost-mkt"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    gdelt_route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"articles": []})
    )
    resp = app_client.get("/terminal/gdelt/ghost-mkt")
    assert resp.status_code == 404
    assert "no market" in resp.json()["detail"]
    # We never called GDELT.
    assert gdelt_route.call_count == 0


@respx.mock
def test_per_slug_endpoint_handles_gdelt_throttle_gracefully(
    app_client: TestClient,
) -> None:
    """GDELT replies with a plaintext throttle message → endpoint returns 0 articles."""
    respx.get(f"{GAMMA}/markets", params={"slug": "btc-200k"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "btc-200k",
                    "question": "Will Bitcoin reach 200k by end of year?",
                }
            ],
        )
    )
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            text="Please limit requests to one every 5 seconds...",
        )
    )

    resp = app_client.get("/terminal/gdelt/btc-200k")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_articles"] == 0
    assert body["articles"] == []
    assert body["mean_tone"] == 0.0
    # dominant_topic falls back to first keyword.
    assert (
        body["dominant_topic"] == "bitcoin" or body["dominant_topic"] in body["query_used"].lower()
    )


@respx.mock
def test_breaking_endpoint_returns_top_global_headlines(
    app_client: TestClient,
) -> None:
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "articles": [
                    {
                        "url": "https://www.cnn.com/breaking-1",
                        "title": "Major earthquake strikes Pacific region",
                        "domain": "cnn.com",
                        "sourcecountry": "United States",
                        "language": "English",
                        "seendate": "20260502T120000Z",
                        "socialimage": "https://example.com/eq.jpg",
                        "tone": -5.5,
                    },
                    {
                        "url": "https://www.aljazeera.com/breaking-2",
                        "title": "Diplomatic breakthrough announced today",
                        "domain": "aljazeera.com",
                        "sourcecountry": "Qatar",
                        "language": "English",
                        "seendate": "20260502T130000Z",
                        "socialimage": "",
                        "tone": 4.0,
                    },
                ]
            },
        )
    )

    resp = app_client.get("/terminal/gdelt/breaking?limit=10")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["timespan"] == "6h"
    assert body["n_articles"] == 2
    assert {a["source"] for a in body["articles"]} == {"cnn.com", "aljazeera.com"}
    # Verify GDELT was called with timespan=6h and sort=hybridrel.
    call = respx.calls.last
    qs = dict(call.request.url.params)
    assert qs.get("timespan") == "6h"
    assert qs.get("sort") == "hybridrel"
    assert qs.get("mode") == "artlist"


# ---------------------------------------------------------------------------
# Cache layering: L1 + Redis L2 (cross-worker)
# ---------------------------------------------------------------------------


class _FakeRedisL2:
    """In-memory stand-in for the Redis cache wrapper."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.enabled = True

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        self.store[key] = value

    def setnx(self, key: str, value: bytes, ttl_seconds: int) -> bool:
        if key in self.store:
            return False
        self.store[key] = value
        return True


@respx.mock
def test_gdelt_per_slug_l2_redis_promotes_into_l1(app_client: TestClient) -> None:
    """A seeded L2 cache entry must serve without hitting GDELT/Gamma.

    Models the cross-worker case: worker A paid the ~8.3 s GDELT cost
    and wrote both L1 and L2; worker B (this test) starts with an empty
    L1 but sees the L2 entry and short-circuits.
    """
    import json as _json

    fake_l2 = _FakeRedisL2()
    seeded = {
        "slug": "trump-out-by-2027",
        "query_used": "trump impeach",
        "n_articles": 0,
        "articles": [],
        "mean_tone": 0.0,
        "dominant_topic": "seeded",
        "top_sources": [],
        "anchors": [],
        "topics": [],
        "relevance_min": 0.15,
    }
    fake_l2.store["terminal_gdelt:payload:slug:trump-out-by-2027:20"] = _json.dumps(seeded).encode(
        "utf-8"
    )
    app_client.app.state.cache = fake_l2

    gamma_route = respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(500, json={"error": "should-not-be-called"})
    )
    gdelt_route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(500, json={"error": "should-not-be-called"})
    )

    resp = app_client.get("/terminal/gdelt/trump-out-by-2027?limit=20")
    assert resp.status_code == 200, resp.text
    assert resp.json()["dominant_topic"] == "seeded"
    assert gamma_route.call_count == 0
    assert gdelt_route.call_count == 0


@respx.mock
def test_gdelt_per_slug_writes_to_l2_after_fresh_fetch(
    app_client: TestClient,
) -> None:
    """After a cold fetch the L2 cache must hold the encoded payload."""
    import json as _json

    fake_l2 = _FakeRedisL2()
    app_client.app.state.cache = fake_l2

    respx.get(f"{GAMMA}/markets", params={"slug": "trump-out-by-2027"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "trump-out-by-2027",
                    "question": "Will Trump resign or be impeached by 2027?",
                }
            ],
        )
    )
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload()))

    resp = app_client.get("/terminal/gdelt/trump-out-by-2027?limit=20")
    assert resp.status_code == 200
    key = "terminal_gdelt:payload:slug:trump-out-by-2027:20"
    assert key in fake_l2.store
    stored = _json.loads(fake_l2.store[key])
    assert stored["n_articles"] == resp.json()["n_articles"]
