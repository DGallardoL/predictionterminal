"""Tests for ``pfm.terminal.news_relevance`` + the relevance filter wiring
across the four news modules (Reddit/HN, GDELT, RSS, news-impact).

These tests guard the user complaint that "las noticias de los eventos
no parecen ser taaan de los eventos": off-topic results must be
filtered out by the relevance scorer.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal.news_relevance import (
    RELEVANCE_MIN,
    build_anchor_phrase,
    build_phrase_query,
    build_reddit_query,
    build_terms,
    score_relevance,
)
from pfm.terminal_gdelt_news import _CACHE as GDELT_CACHE
from pfm.terminal_gdelt_news import GDELT_DOC_URL
from pfm.terminal_gdelt_news import router as gdelt_router
from pfm.terminal_news import _CACHE as NEWS_CACHE
from pfm.terminal_news import HN_SEARCH_URL, REDDIT_SEARCH_URL
from pfm.terminal_news import router as news_router
from pfm.terminal_rss_news import _CACHE as RSS_CACHE
from pfm.terminal_rss_news import SOURCES
from pfm.terminal_rss_news import router as rss_router

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"


# ---------------------------------------------------------------------------
# Pure-function tests (no HTTP)
# ---------------------------------------------------------------------------


class TestBuildTerms:
    def test_extracts_proper_noun_anchor(self) -> None:
        terms = build_terms("Will Trump be impeached by 2027?")
        # "Will" is a stopword even though capitalised — must be skipped.
        assert "Trump" in terms.anchors
        assert "Will" not in terms.anchors
        assert "Will Trump" not in terms.anchors
        assert "impeached" in terms.topics

    def test_groups_multiword_entity(self) -> None:
        terms = build_terms("Will Joe Biden win the 2024 election?")
        assert "Joe Biden" in terms.anchors
        assert "election" in terms.topics

    def test_keeps_all_caps_ticker(self) -> None:
        terms = build_terms("Will NVDA mention AI 50 times by Q3?")
        assert "NVDA" in terms.anchors
        assert "AI" in terms.anchors
        # "mention" survives as a topic.
        assert "mention" in terms.topics

    def test_short_question_only_stopwords(self) -> None:
        terms = build_terms("Will it happen?")
        # Nothing useful — both anchors and topics empty.
        assert terms.anchors == ()
        assert terms.topics == ()

    def test_dedupes_case_insensitive(self) -> None:
        terms = build_terms("Trump vs Trump in 2026 — will Trump win?")
        assert terms.anchors.count("Trump") == 1

    def test_drops_generic_stopwords(self) -> None:
        terms = build_terms("Will the year-end rally continue?")
        # "year" and "end" are filtered as generic noise; "rally" survives.
        assert "rally" in terms.topics
        assert "year" not in terms.topics
        assert "end" not in terms.topics


class TestScoreRelevance:
    def test_anchor_in_title_scores_high(self) -> None:
        terms = build_terms("Will NVDA hit $1500 by year end?")
        score, matched = score_relevance("NVDA reports blowout earnings", terms)
        assert score >= 0.40
        assert "NVDA" in matched

    def test_topic_only_match_scores_modestly(self) -> None:
        terms = build_terms("Will Joe Biden win 2024?")
        # Title mentions an unrelated entity but uses the "win" topic word.
        # Joe Biden anchor missing → score below floor.
        score, _ = score_relevance("Lakers win NBA championship", terms)
        assert score < RELEVANCE_MIN

    def test_stem_match_handles_morphology(self) -> None:
        terms = build_terms("Will Trump be impeached by 2027?")
        # The headline uses "impeachment" (longer suffix); 5-char prefix
        # match must still recognise it.
        score, matched = score_relevance("Senate hearing: Trump faces impeachment vote", terms)
        assert score >= 0.40
        assert any(m.lower() == "trump" for m in matched)
        assert any(m.startswith("impeach") for m in matched)

    def test_negative_context_penalises(self) -> None:
        terms = build_terms("Will NVDA hit $1500?")
        # "Why this rally is NOT about NVDA" — negative context.
        score_pos, _ = score_relevance("NVDA shares rally on AI demand", terms)
        score_neg, _ = score_relevance("Why this rally is not about NVDA but TSMC", terms)
        assert score_neg < score_pos

    def test_empty_text_returns_zero(self) -> None:
        terms = build_terms("Will Trump win?")
        score, matched = score_relevance("", terms)
        assert score == 0.0
        assert matched == []

    def test_question_with_no_anchors_promotes_first_topic(self) -> None:
        # Anchorless question — topic match must still reach the floor.
        terms = build_terms("Will the rally continue?")
        assert terms.anchors == ()
        assert "rally" in terms.topics
        score, _ = score_relevance("Bond rally continues into May", terms)
        assert score >= RELEVANCE_MIN


class TestQueryBuilders:
    def test_phrase_query_quotes_multiword_anchors(self) -> None:
        terms = build_terms("Will Joe Biden win 2024?")
        q = build_phrase_query(terms)
        assert '"Joe Biden"' in q

    def test_reddit_query_combines_anchor_and_topic(self) -> None:
        terms = build_terms("Will Trump be impeached by 2027?")
        q = build_reddit_query(terms)
        # Single-word anchor doesn't need quoting; topics follow.
        assert "Trump" in q
        assert "impeached" in q

    def test_anchor_phrase_falls_back_to_first_topic(self) -> None:
        terms = build_terms("Will the rally continue?")
        # No anchors — should return first topic.
        assert build_anchor_phrase(terms) == "rally"


# ---------------------------------------------------------------------------
# Integration tests — the relevance filter wires through the endpoints.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    NEWS_CACHE.clear()
    GDELT_CACHE.clear()
    RSS_CACHE.clear()
    yield
    NEWS_CACHE.clear()
    GDELT_CACHE.clear()
    RSS_CACHE.clear()


@pytest.fixture
def news_app_client() -> TestClient:
    app = FastAPI()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    app.include_router(news_router)
    return TestClient(app)


@pytest.fixture
def gdelt_app_client() -> TestClient:
    app = FastAPI()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    app.include_router(gdelt_router)
    return TestClient(app)


@pytest.fixture
def rss_app_client() -> TestClient:
    app = FastAPI()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    app.include_router(rss_router)
    return TestClient(app)


@respx.mock
def test_news_endpoint_drops_offtopic_results(news_app_client: TestClient) -> None:
    """A Reddit hit unrelated to the question is filtered out by relevance."""
    respx.get(f"{GAMMA}/markets", params={"slug": "nvda-1500"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "nvda-1500",
                    "question": "Will NVDA hit $1500 by year end?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    respx.get(REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        # On-topic: mentions NVDA.
                        {
                            "data": {
                                "title": "NVDA crushes earnings, stock pops",
                                "permalink": "/r/stocks/nvda1/",
                                "created_utc": 1735776000,
                                "score": 100,
                            }
                        },
                        # Off-topic: pure AI hype, no NVDA mention.
                        {
                            "data": {
                                "title": "OpenAI announces new model rollout",
                                "permalink": "/r/ai/openai1/",
                                "created_utc": 1735776100,
                                "score": 200,
                            }
                        },
                    ]
                }
            },
        )
    )
    respx.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))

    resp = news_app_client.get("/terminal/news/nvda-1500?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    # The off-topic OpenAI post must be dropped.
    titles = [it["title"] for it in body["items"]]
    assert "NVDA crushes earnings, stock pops" in titles
    assert "OpenAI announces new model rollout" not in titles
    # New backward-compatible fields are surfaced.
    assert "anchors" in body and "NVDA" in body["anchors"]
    assert "relevance_min" in body
    # Each item has a relevance_score in [0, 1].
    for it in body["items"]:
        assert 0.0 <= it["relevance_score"] <= 1.0
        assert isinstance(it["matched_terms"], list)


@respx.mock
def test_news_endpoint_short_question_still_responds(
    news_app_client: TestClient,
) -> None:
    """Question made of nothing but stopwords degrades gracefully."""
    respx.get(f"{GAMMA}/markets", params={"slug": "all-stop"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "all-stop",
                    "question": "Will it happen?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    respx.get(REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "Random news",
                                "permalink": "/r/x/1/",
                                "created_utc": 1735776000,
                                "score": 1,
                            }
                        }
                    ]
                }
            },
        )
    )
    respx.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))

    resp = news_app_client.get("/terminal/news/all-stop?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    # No anchors and no topics → endpoint must NOT silently 0-out.
    assert body["anchors"] == []
    assert body["topics"] == []
    # The (questionable) "Random news" still appears because we fall
    # back to legacy recency sort when there are no usable terms.
    assert body["n_items"] >= 1


@respx.mock
def test_gdelt_endpoint_filters_offtopic_articles(
    gdelt_app_client: TestClient,
) -> None:
    """GDELT slug endpoint drops articles that don't mention the entity."""
    respx.get(f"{GAMMA}/markets", params={"slug": "nvda-1500"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "nvda-1500",
                    "question": "Will NVDA hit $1500 by year end?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "articles": [
                    {
                        "url": "https://reuters.com/nvda-blowout",
                        "title": "NVDA reports record earnings, stock soars",
                        "domain": "reuters.com",
                        "sourcecountry": "United States",
                        "language": "English",
                        "seendate": "20260304T120000Z",
                        "tone": 4.0,
                    },
                    {
                        "url": "https://bbc.com/openai-news",
                        "title": "OpenAI unveils new product line",
                        "domain": "bbc.com",
                        "sourcecountry": "United Kingdom",
                        "language": "English",
                        "seendate": "20260304T130000Z",
                        "tone": 1.0,
                    },
                    {
                        "url": "https://ft.com/macro",
                        "title": "Macro headwinds intensify globally",
                        "domain": "ft.com",
                        "sourcecountry": "United Kingdom",
                        "language": "English",
                        "seendate": "20260304T140000Z",
                        "tone": -2.0,
                    },
                ]
            },
        )
    )

    resp = gdelt_app_client.get("/terminal/gdelt/nvda-1500?limit=20")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    titles = [a["title"] for a in body["articles"]]
    assert any("NVDA" in t for t in titles)
    assert not any("OpenAI" in t for t in titles)
    assert not any("Macro" in t for t in titles)
    assert "NVDA" in body["anchors"]
    # The kept article has a non-zero relevance_score.
    assert body["articles"][0]["relevance_score"] > 0.0


@respx.mock
def test_rss_slug_endpoint_drops_unrelated_headlines(
    rss_app_client: TestClient,
) -> None:
    """``/terminal/rss/{slug}`` only returns headlines that pass the floor."""
    rss_bbc = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<rss version="2.0"><channel>'
        b"  <item>"
        b"    <title>NVDA crushes earnings expectations</title>"
        b"    <link>https://bbc.test/nvda-earnings</link>"
        b"    <pubDate>Fri, 02 May 2026 12:00:00 GMT</pubDate>"
        b"    <description>Strong demand for AI chips.</description>"
        b"  </item>"
        b"  <item>"
        b"    <title>Cricket world cup final result</title>"
        b"    <link>https://bbc.test/cricket</link>"
        b"    <pubDate>Fri, 02 May 2026 13:00:00 GMT</pubDate>"
        b"    <description>India wins by 6 wickets.</description>"
        b"  </item>"
        b"</channel></rss>"
    )
    empty = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
    for src in SOURCES:
        if src.slug == "bbc_world":
            respx.get(src.url).mock(return_value=httpx.Response(200, content=rss_bbc))
        else:
            respx.get(src.url).mock(return_value=httpx.Response(200, content=empty))

    respx.get(f"{GAMMA}/markets", params={"slug": "nvda-1500"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "nvda-1500",
                    "question": "Will NVDA hit $1500 by year end?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )

    resp = rss_app_client.get("/terminal/rss/nvda-1500?limit=10")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    titles = [it["title"] for it in body["items"]]
    assert any("NVDA" in t for t in titles)
    assert not any("Cricket" in t for t in titles)
    # New backward-compatible fields.
    assert "anchors" in body and "NVDA" in body["anchors"]
    assert body["items"][0]["relevance_score"] > 0.0


# ---------------------------------------------------------------------------
# Regression: query strings sent upstream are tightened.
# ---------------------------------------------------------------------------


@respx.mock
def test_reddit_query_uses_relevance_sort_and_recency_window(
    news_app_client: TestClient,
) -> None:
    """Reddit calls now use ``sort=relevance`` and a 1-month time bound."""
    respx.get(f"{GAMMA}/markets", params={"slug": "nvda-1500"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "nvda-1500",
                    "question": "Will NVDA hit $1500?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    reddit_route = respx.get(REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}})
    )
    respx.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))

    news_app_client.get("/terminal/news/nvda-1500")
    assert reddit_route.call_count == 1
    qs = dict(reddit_route.calls[0].request.url.params)
    assert qs["sort"] == "relevance"
    assert qs.get("t") == "month"
    # Query uses the anchor phrase (NVDA) — not a generic OR'd token list.
    assert "NVDA" in qs["q"]


@respx.mock
def test_hn_query_bounds_by_creation_recency(
    news_app_client: TestClient,
) -> None:
    """HN calls now include a 90-day ``numericFilters=created_at_i>...``."""
    respx.get(f"{GAMMA}/markets", params={"slug": "nvda-1500"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "nvda-1500",
                    "question": "Will NVDA hit $1500?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    respx.get(REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}})
    )
    hn_route = respx.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))

    news_app_client.get("/terminal/news/nvda-1500")
    assert hn_route.call_count == 1
    qs = dict(hn_route.calls[0].request.url.params)
    assert "numericFilters" in qs
    assert qs["numericFilters"].startswith("created_at_i>")


@respx.mock
def test_gdelt_query_quotes_multiword_anchor(
    gdelt_app_client: TestClient,
) -> None:
    """GDELT request quotes the multi-word anchor as an atomic phrase."""
    respx.get(f"{GAMMA}/markets", params={"slug": "biden-2024"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "biden-2024",
                    "question": "Will Joe Biden win 2024?",
                    "clobTokenIds": '["111", "222"]',
                }
            ],
        )
    )
    gdelt_route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"articles": []})
    )

    gdelt_app_client.get("/terminal/gdelt/biden-2024")
    assert gdelt_route.call_count == 1
    qs = dict(gdelt_route.calls[0].request.url.params)
    assert '"Joe Biden"' in qs["query"]
