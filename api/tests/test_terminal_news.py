"""Tests for ``pfm.terminal_news``. All HTTP is mocked via respx."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_news import (
    _CACHE,
    HN_SEARCH_URL,
    REDDIT_SEARCH_URL,
    classify_sentiment,
    extract_keywords,
)
from pfm.terminal_news import (
    router as news_router,
)

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """The news endpoint has a process-wide cache; clear it between tests."""
    _CACHE.clear()
    yield
    _CACHE.clear()


@pytest.fixture
def app_client() -> TestClient:
    """Minimal FastAPI app wiring just the news router + a poly client."""
    app = FastAPI()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    app.include_router(news_router)
    return TestClient(app)


def test_extract_keywords_drops_stop_words_and_short_tokens() -> None:
    kws = extract_keywords("Will the Fed cut rates in December?")
    # "the", "in", "will" → stop-words; "Fed" stays.
    assert "fed" in kws
    assert "cut" in kws
    assert "the" not in kws
    assert len(kws) <= 3


def test_classify_sentiment_word_lists() -> None:
    assert classify_sentiment("Stocks rally on great earnings") == "positive"
    assert classify_sentiment("Market crash and panic in bonds") == "negative"
    assert classify_sentiment("Fed meeting scheduled tomorrow") == "neutral"


@respx.mock
def test_news_endpoint_merges_reddit_and_hn(app_client: TestClient) -> None:
    # Gamma returns the market question.
    respx.get(f"{GAMMA}/markets", params={"slug": "fed-cut-dec"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "fed-cut-dec",
                    "question": "Will the Fed cut rates in December?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    # Reddit returns one post.
    respx.get(REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "Fed signals rate cut",
                                "permalink": "/r/economics/comments/abc/fed/",
                                "created_utc": 1735776000,  # 2025-01-02 UTC
                                "score": 42,
                            }
                        }
                    ]
                }
            },
        )
    )
    # HN returns one story.
    respx.get(HN_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": "Bond market rallies after Fed announcement",
                        "url": "https://example.com/bonds",
                        "created_at": "2025-02-01T12:00:00Z",
                        "points": 100,
                        "objectID": "999",
                    }
                ]
            },
        )
    )

    resp = app_client.get("/terminal/news/fed-cut-dec?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == "fed-cut-dec"
    assert "fed" in body["keywords"]
    assert body["n_items"] == 2
    sources = {it["source"] for it in body["items"]}
    assert sources == {"reddit", "hn"}
    # Both items match the "fed" topic and pass the relevance floor.
    # Reddit's title ("Fed signals rate cut") matches the extra topic
    # "cut" so it now ranks above HN ("Bond market rallies after Fed
    # announcement") which only hits "fed". Old behaviour was strict
    # recency-sort; new behaviour ranks by relevance first.
    assert body["items"][0]["source"] == "reddit"
    assert body["items"][0]["relevance_score"] >= body["items"][1]["relevance_score"]
    # User-Agent was set on the Reddit call.
    reddit_call = next(c for c in respx.calls if str(c.request.url).startswith(REDDIT_SEARCH_URL))
    assert reddit_call.request.headers["user-agent"] == "polymarket-terminal/1.0"


@respx.mock
def test_news_endpoint_handles_reddit_429_gracefully(app_client: TestClient) -> None:
    respx.get(f"{GAMMA}/markets", params={"slug": "btc-100k"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "btc-100k",
                    "question": "Will Bitcoin reach 100k by end of year?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    respx.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(429))
    respx.get(HN_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": "Bitcoin crosses 100k milestone",
                        "url": "https://example.com/btc",
                        "created_at": "2025-12-01T00:00:00Z",
                        "points": 250,
                        "objectID": "111",
                    }
                ]
            },
        )
    )

    resp = app_client.get("/terminal/news/btc-100k?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    # Reddit died but the endpoint still returns HN results.
    assert body["n_items"] == 1
    assert body["items"][0]["source"] == "hn"
    assert body["items"][0]["sentiment"] == "neutral"


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------


@respx.mock
def test_news_endpoint_404s_when_market_missing(app_client: TestClient) -> None:
    """Empty Gamma response → 404 with helpful detail (no Reddit/HN call)."""
    respx.get(f"{GAMMA}/markets", params={"slug": "ghost-mkt"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    reddit_route = respx.get(REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}})
    )
    resp = app_client.get("/terminal/news/ghost-mkt")
    assert resp.status_code == 404
    assert "no market" in resp.json()["detail"]
    assert reddit_route.call_count == 0


@respx.mock
def test_news_endpoint_502_on_gamma_500(app_client: TestClient) -> None:
    """Upstream 5xx from Gamma surfaces as a degraded-mode payload (not 502).

    The news endpoint intentionally degrades to an empty-but-valid envelope
    when Gamma returns 429/5xx so the UI renders a friendly empty state
    instead of a red error card. The test name is preserved for git history.
    """
    respx.get(f"{GAMMA}/markets", params={"slug": "any-slug"}).mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    resp = app_client.get("/terminal/news/any-slug")
    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded_mode"] is True
    assert body["n_items"] == 0
    assert body["items"] == []


@respx.mock
def test_news_dedupes_on_url_and_respects_limit(app_client: TestClient) -> None:
    """If Reddit and HN return the same URL, dedupe; honour `limit`."""
    respx.get(f"{GAMMA}/markets", params={"slug": "dup-mkt"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "dup-mkt",
                    "question": "Will the rally continue?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    # Reddit returns 3 unique posts. Titles include "rally" so they
    # pass the relevance floor (matches the question topic).
    respx.get(REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": f"Markets rally post {i}",
                                "permalink": f"/r/x/{i}/",
                                "created_utc": 1735776000 + i,
                                "score": i,
                            }
                        }
                        for i in range(3)
                    ]
                }
            },
        )
    )
    # HN returns 5 stories — we cap at limit=2 in total. Titles all
    # contain "rally" so the relevance filter keeps them.
    respx.get(HN_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": f"Bond rally update {i}",
                        "url": f"https://example.com/p{i}",
                        "created_at": f"2025-12-0{i + 1}T00:00:00Z",
                        "points": 10 * i,
                        "objectID": str(i),
                    }
                    for i in range(5)
                ]
            },
        )
    )
    resp = app_client.get("/terminal/news/dup-mkt?limit=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_items"] == 2
    assert len(body["items"]) == 2


@respx.mock
def test_news_response_schema_fields(app_client: TestClient) -> None:
    """Each item payload exposes exactly the documented Pydantic fields."""
    respx.get(f"{GAMMA}/markets", params={"slug": "schema-mkt"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "schema-mkt",
                    "question": "Will Bitcoin crash?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    respx.get(REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}})
    )
    respx.get(HN_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": "Bitcoin plunge sparks panic",
                        "url": "https://example.com/btc-plunge",
                        "created_at": "2025-12-15T00:00:00Z",
                        "points": 50,
                        "objectID": "abc",
                    }
                ]
            },
        )
    )
    body = app_client.get("/terminal/news/schema-mkt").json()
    expected_top = {"slug", "question", "keywords", "n_items", "items"}
    # Top-level shape is backward-compatible: old fields must still be
    # present; additive fields (anchors/topics/relevance_min) are allowed.
    assert expected_top <= set(body.keys())
    assert body["items"]
    expected_item = {"source", "title", "url", "ts", "score", "sentiment"}
    assert expected_item <= set(body["items"][0].keys())
    # The negative-keyword title should classify as 'negative'.
    assert body["items"][0]["sentiment"] == "negative"
    # New relevance fields are surfaced.
    assert "relevance_score" in body["items"][0]
    assert "matched_terms" in body["items"][0]


# ---------------------------------------------------------------------------
# Resilience hardening: 429-retry + degraded-mode envelope
# ---------------------------------------------------------------------------


@respx.mock
def test_news_reddit_429_then_200_recovers_on_single_retry(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single 429 from Reddit must trigger a 1.5s retry that succeeds."""
    # Patch the backoff sleep to zero so tests stay fast.
    import pfm.terminal.news as news_module

    monkeypatch.setattr(news_module.time, "sleep", lambda _s: None)

    respx.get(f"{GAMMA}/markets", params={"slug": "fed-retry"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "fed-retry",
                    "question": "Will the Fed cut rates?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    reddit_route = respx.get(REDDIT_SEARCH_URL).mock(
        side_effect=[
            httpx.Response(429),  # first try gets rate limited
            httpx.Response(
                200,
                json={
                    "data": {
                        "children": [
                            {
                                "data": {
                                    "title": "Fed signals rate cut soon",
                                    "permalink": "/r/x/y/",
                                    "created_utc": 1735776000,
                                    "score": 9,
                                }
                            },
                        ]
                    }
                },
            ),
        ]
    )
    respx.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))

    resp = app_client.get("/terminal/news/fed-retry?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    # Retry succeeded, item surfaces, response is NOT marked degraded.
    assert reddit_route.call_count == 2
    assert body["n_items"] == 1
    assert body["degraded_mode"] is False


@respx.mock
def test_news_degraded_mode_when_all_upstreams_fail(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both Reddit (429 twice) AND HN errors out → degraded_mode=True, items=[]."""
    import pfm.terminal.news as news_module

    monkeypatch.setattr(news_module.time, "sleep", lambda _s: None)

    respx.get(f"{GAMMA}/markets", params={"slug": "dead-news"}).mock(
        return_value=httpx.Response(
            200,
            json=[{"slug": "dead-news", "question": "Will the bond market crash?"}],
        )
    )
    # Reddit stays at 429 even after the retry.
    respx.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(429))
    # HN returns 500 — also marks the source as "not ok".
    respx.get(HN_SEARCH_URL).mock(return_value=httpx.Response(500))

    resp = app_client.get("/terminal/news/dead-news")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["degraded_mode"] is True
    assert body["n_items"] == 0
    assert body["items"] == []
    # The endpoint must NOT cache a degraded payload — next caller gets
    # a fresh shot.
    assert ("dead-news", 20) not in _CACHE


@respx.mock
def test_news_cache_hit_skips_upstream_calls(
    app_client: TestClient,
) -> None:
    """A second request within the TTL must NOT hit Reddit/HN/Gamma again."""
    gamma_route = respx.get(f"{GAMMA}/markets", params={"slug": "cache-mkt"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "cache-mkt",
                    "question": "Will Bitcoin rally?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    reddit_route = respx.get(REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "Bitcoin rally update",
                                "permalink": "/r/x/y/",
                                "created_utc": 1735776000,
                                "score": 5,
                            }
                        }
                    ]
                }
            },
        )
    )
    hn_route = respx.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))

    r1 = app_client.get("/terminal/news/cache-mkt?limit=5")
    assert r1.status_code == 200
    n_calls_after_first = gamma_route.call_count + reddit_route.call_count + hn_route.call_count

    r2 = app_client.get("/terminal/news/cache-mkt?limit=5")
    assert r2.status_code == 200
    assert r2.json() == r1.json()
    # No additional upstream traffic on the second call.
    assert (
        gamma_route.call_count + reddit_route.call_count + hn_route.call_count
        == n_calls_after_first
    )
