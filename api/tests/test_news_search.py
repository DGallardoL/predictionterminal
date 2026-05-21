"""Tests for ``pfm.news_search_router`` — ``GET /news/search``.

All HTTP is mocked via respx. Tests use a stub ``FactorConfig``-like
factor catalog wired onto ``app.state.factors_by_slug`` so the
factor-match path is exercised without needing the full 1228-factor YAML.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.factors import FactorConfig
from pfm.news_search_router import (
    GDELT_DOC_URL,
    _cache_clear,
    _jaccard,
    _tokenize,
)
from pfm.news_search_router import router as news_search_router

# --- helpers ----------------------------------------------------------------


def _make_factor(slug: str, name: str) -> FactorConfig:
    """Build a polymarket-shaped FactorConfig for tests."""
    return FactorConfig(
        id=slug,
        name=name,
        slug=slug,
        source="polymarket",
        description=f"test factor {slug}",
    )


def _gdelt_payload(articles: list[dict]) -> dict:
    """Wrap a list of GDELT-shaped article dicts in the expected envelope."""
    return {"articles": articles}


def _article(
    *,
    url: str,
    title: str,
    domain: str = "example.com",
    seendate: str = "20260515T120000Z",
) -> dict:
    return {
        "url": url,
        "title": title,
        "domain": domain,
        "sourcecountry": "United States",
        "language": "English",
        "seendate": seendate,
        "socialimage": "",
        "tone": 0.0,
    }


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Drop the module-level cache before AND after each test."""
    _cache_clear()
    yield
    _cache_clear()


@pytest.fixture
def app_client() -> TestClient:
    """FastAPI test app with the news_search router mounted.

    ``factors_by_slug`` is pre-populated with three factors whose names
    contain Fed / inflation / NVDA tokens so the factor-match assertions
    have something to bind to.
    """
    app = FastAPI()
    app.state.factors_by_slug = {
        "fed-rate-cut-march-2026": _make_factor(
            "fed-rate-cut-march-2026", "Fed cuts rates at March 2026 meeting"
        ),
        "fed-rate-decision-q2-2026": _make_factor(
            "fed-rate-decision-q2-2026", "Fed rate decision Q2 2026"
        ),
        "nvda-1trn-mcap": _make_factor(
            "nvda-1trn-mcap", "NVDA market cap exceeds 1 trillion by 2026"
        ),
        "us-inflation-above-3pct": _make_factor(
            "us-inflation-above-3pct", "US inflation prints above 3% in 2026"
        ),
    }
    app.include_router(news_search_router)
    return TestClient(app)


# --- tokenisation / jaccard unit tests --------------------------------------


def test_tokenize_drops_stopwords_and_lowercases() -> None:
    """Stop-words removed, output is a lowercase set of alnum tokens ≥2 chars."""
    toks = _tokenize("The Fed cut rates by 25 bps")
    assert "the" not in toks
    assert "by" not in toks
    assert toks == {"fed", "cut", "rates", "25", "bps"}


def test_jaccard_perfect_match() -> None:
    a = _tokenize("Fed rate cut March")
    b = _tokenize("Fed rate cut March")
    assert _jaccard(a, b) == pytest.approx(1.0)


def test_jaccard_no_overlap() -> None:
    a = _tokenize("crypto bitcoin")
    b = _tokenize("election polls")
    assert _jaccard(a, b) == 0.0


def test_jaccard_partial_overlap() -> None:
    """{fed, cut} vs {fed, rates}: intersection 1, union 3 → 1/3."""
    a = _tokenize("fed cut")
    b = _tokenize("fed rates")
    assert _jaccard(a, b) == pytest.approx(1 / 3)


# --- endpoint tests ---------------------------------------------------------


@respx.mock
def test_basic_query_returns_200(app_client: TestClient) -> None:
    """A query with matching upstream articles returns 200 + populated results."""
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_payload(
                [
                    _article(
                        url="https://reuters.com/fed-1",
                        title="Fed signals rate cut in March 2026",
                        domain="reuters.com",
                    ),
                ]
            ),
        )
    )
    resp = app_client.get("/news/search?q=fed+rate+cut")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["q"] == "fed rate cut"
    assert body["since"] == "7d"
    assert body["count"] == 1
    assert body["results"][0]["title"] == "Fed signals rate cut in March 2026"
    assert body["results"][0]["url"] == "https://reuters.com/fed-1"
    assert body["results"][0]["source"] == "reuters.com"
    # Token-Jaccard between {fed,rate,cut} and {fed,signals,rate,cut,march,2026}
    # = 3/6 = 0.5
    assert body["results"][0]["score"] == pytest.approx(0.5)


def test_missing_q_returns_422(app_client: TestClient) -> None:
    """``q`` is required → 422 from FastAPI's Query validation."""
    resp = app_client.get("/news/search")
    assert resp.status_code == 422


def test_empty_q_returns_422(app_client: TestClient) -> None:
    """An empty ``q=`` fails min_length=1 → 422."""
    resp = app_client.get("/news/search?q=")
    assert resp.status_code == 422


@respx.mock
def test_whitespace_only_q_returns_422(app_client: TestClient) -> None:
    """A whitespace-only ``q`` is rejected by the post-strip guard."""
    resp = app_client.get("/news/search?q=%20%20%20")
    assert resp.status_code == 422
    assert "non-whitespace" in resp.json()["detail"]


@respx.mock
def test_results_sorted_by_score_desc(app_client: TestClient) -> None:
    """Results are returned in descending score order."""
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_payload(
                [
                    # Low overlap: only 'fed' matches → 1/4 = 0.25
                    _article(
                        url="https://x.com/1",
                        title="Fed mentioned in passing",
                    ),
                    # High overlap: fed,rate,cut all present → 3/4 = 0.75
                    _article(
                        url="https://x.com/2",
                        title="Fed announces rate cut",
                    ),
                    # Medium: fed,rate match → 2/4 = 0.5
                    _article(
                        url="https://x.com/3",
                        title="Fed holds rate steady",
                    ),
                ]
            ),
        )
    )
    resp = app_client.get("/news/search?q=fed+rate+cut")
    body = resp.json()
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True), scores
    assert body["results"][0]["url"] == "https://x.com/2"


@respx.mock
def test_factor_attachment_when_factors_true(app_client: TestClient) -> None:
    """Articles get ``matched_factors`` populated when factor names match tokens."""
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_payload(
                [
                    _article(
                        url="https://r.com/1",
                        title="Fed rate cut expected at March meeting",
                    ),
                ]
            ),
        )
    )
    resp = app_client.get("/news/search?q=fed+rate&factors=true")
    body = resp.json()
    matched = body["results"][0]["matched_factors"]
    # Both Fed-named factors share fed/rate tokens with the title.
    assert "fed-rate-cut-march-2026" in matched
    assert "fed-rate-decision-q2-2026" in matched
    # NVDA and inflation factors should NOT match.
    assert "nvda-1trn-mcap" not in matched
    assert "us-inflation-above-3pct" not in matched


@respx.mock
def test_factor_attachment_skipped_when_factors_false(
    app_client: TestClient,
) -> None:
    """``factors=false`` produces empty ``matched_factors`` even with hits."""
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_payload(
                [
                    _article(
                        url="https://r.com/1",
                        title="Fed rate cut expected at March meeting",
                    ),
                ]
            ),
        )
    )
    resp = app_client.get("/news/search?q=fed+rate&factors=false")
    body = resp.json()
    assert body["results"][0]["matched_factors"] == []


@respx.mock
def test_since_parsing_24h_maps_to_gdelt_timespan(
    app_client: TestClient,
) -> None:
    """``since=24h`` passes ``timespan=24h`` to GDELT."""
    route = respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
    resp = app_client.get("/news/search?q=fed&since=24h")
    assert resp.status_code == 200
    assert resp.json()["since"] == "24h"
    assert route.call_count == 1
    sent_url = str(route.calls[0].request.url)
    assert "timespan=24h" in sent_url


@respx.mock
def test_since_parsing_7d_maps_to_gdelt_7days(app_client: TestClient) -> None:
    """``since=7d`` (default) maps to GDELT's ``timespan=7days`` value."""
    route = respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
    resp = app_client.get("/news/search?q=fed&since=7d")
    assert resp.status_code == 200
    sent_url = str(route.calls[0].request.url)
    # urlencoded form: "timespan=7days"
    assert "timespan=7days" in sent_url


@respx.mock
def test_since_parsing_30d_maps_to_gdelt_30days(app_client: TestClient) -> None:
    """``since=30d`` maps to GDELT's ``timespan=30days``."""
    route = respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
    resp = app_client.get("/news/search?q=election&since=30d")
    assert resp.status_code == 200
    sent_url = str(route.calls[0].request.url)
    assert "timespan=30days" in sent_url


@respx.mock
def test_invalid_since_returns_422(app_client: TestClient) -> None:
    """Unknown ``since`` values are rejected by the Literal type."""
    resp = app_client.get("/news/search?q=fed&since=2y")
    assert resp.status_code == 422


@respx.mock
def test_empty_results_returns_count_zero(app_client: TestClient) -> None:
    """When GDELT returns no articles the response is 200 + empty list."""
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
    resp = app_client.get("/news/search?q=does+not+match+anything")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["results"] == []


@respx.mock
def test_gdelt_throttle_returns_empty_results(app_client: TestClient) -> None:
    """GDELT's plaintext throttle message degrades gracefully to ``count: 0``."""
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, text="Please limit requests to one every 5 seconds")
    )
    resp = app_client.get("/news/search?q=fed")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


@respx.mock
def test_gdelt_5xx_returns_empty_results(app_client: TestClient) -> None:
    """An upstream 503 does NOT cause a 5xx on our endpoint."""
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(503, json={"error": "down"}))
    resp = app_client.get("/news/search?q=fed")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


@respx.mock
def test_results_capped_at_50(app_client: TestClient) -> None:
    """More than 50 articles are truncated to the MAX_RESULTS cap."""
    articles = [
        _article(
            url=f"https://x.com/{i}",
            title=f"Fed rate cut analysis number {i}",
        )
        for i in range(75)
    ]
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload(articles)))
    resp = app_client.get("/news/search?q=fed+rate+cut")
    body = resp.json()
    assert body["count"] == 50
    assert len(body["results"]) == 50


@respx.mock
def test_cache_hit_avoids_second_upstream_call(app_client: TestClient) -> None:
    """A second identical request within the TTL is served from cache."""
    route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_payload([_article(url="https://x.com/1", title="Fed signals cut")]),
        )
    )
    r1 = app_client.get("/news/search?q=fed+cut")
    r2 = app_client.get("/news/search?q=fed+cut")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()
    assert route.call_count == 1


@respx.mock
def test_cache_keyed_on_factors_flag(app_client: TestClient) -> None:
    """Different ``factors`` flag produces a distinct cache key → 2 upstream hits."""
    route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_payload([_article(url="https://x.com/1", title="Fed signals cut")]),
        )
    )
    app_client.get("/news/search?q=fed+cut&factors=true")
    app_client.get("/news/search?q=fed+cut&factors=false")
    assert route.call_count == 2


@respx.mock
def test_cache_keyed_on_since_window(app_client: TestClient) -> None:
    """Different ``since`` values bypass each other's cache entries."""
    route = respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
    app_client.get("/news/search?q=fed&since=24h")
    app_client.get("/news/search?q=fed&since=7d")
    assert route.call_count == 2


@respx.mock
def test_articles_missing_required_fields_are_skipped(
    app_client: TestClient,
) -> None:
    """Articles with empty url or title are dropped, not counted."""
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_payload(
                [
                    {"url": "", "title": "Has title no url", "seendate": "20260515T000000Z"},
                    {"url": "https://x.com/2", "title": "", "seendate": "20260515T000000Z"},
                    _article(url="https://x.com/3", title="Fed cuts rate"),
                ]
            ),
        )
    )
    resp = app_client.get("/news/search?q=fed+cut")
    body = resp.json()
    assert body["count"] == 1
    assert body["results"][0]["url"] == "https://x.com/3"


@respx.mock
def test_seendate_converted_to_iso(app_client: TestClient) -> None:
    """GDELT ``YYYYMMDDTHHMMSSZ`` is converted to ISO-8601 in ``published_at``."""
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_payload(
                [
                    _article(
                        url="https://x.com/1",
                        title="Fed cuts rate today",
                        seendate="20260515T143000Z",
                    ),
                ]
            ),
        )
    )
    resp = app_client.get("/news/search?q=fed+cut")
    body = resp.json()
    assert body["results"][0]["published_at"] == "2026-05-15T14:30:00Z"


@respx.mock
def test_factors_attachment_empty_when_no_factors_on_state(
    app_client: TestClient,
) -> None:
    """If ``factors_by_slug`` is missing/None we don't crash — empty list."""
    # Wipe the state to simulate cold lifespan.
    app_client.app.state.factors_by_slug = None
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_payload(
                [
                    _article(url="https://x.com/1", title="Fed cuts rate"),
                ]
            ),
        )
    )
    resp = app_client.get("/news/search?q=fed&factors=true")
    assert resp.status_code == 200
    assert resp.json()["results"][0]["matched_factors"] == []
