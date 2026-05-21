"""Tests for ``pfm.news_tagger`` — entity NER, factor scoring, sentiment, router."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.news_tagger import (
    DEFAULT_THRESHOLD,
    all_entities,
    clear_recent_tagged,
    enhanced_sentiment,
    extract_entities,
    load_entity_factor_map,
    record_tagged_items,
    score_factor_match,
    tag_news_to_factors,
)
from pfm.news_tagger import (
    router as news_tagger_router,
)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    reset_caches()
    clear_recent_tagged()
    yield
    reset_caches()
    clear_recent_tagged()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(news_tagger_router)
    return TestClient(app)


@pytest.fixture
def small_catalog() -> list[dict]:
    return [
        {
            "id": "trump-tariffs-china",
            "name": "Trump tariffs on China",
            "slug": "trump-tariffs-china",
            "theme": "geopolitics",
            "description": "Probability Trump signs new tariffs on China.",
        },
        {
            "id": "fed-cut-march",
            "name": "Fed rate cut in March",
            "slug": "fed-cut-march",
            "theme": "macro",
            "description": "Probability the FOMC cuts rates in March.",
        },
        {
            "id": "btc-150k",
            "name": "BTC reaches 150k",
            "slug": "btc-150k",
            "theme": "crypto",
            "description": "Probability Bitcoin trades at or above 150k.",
        },
        {
            "id": "nvda-earnings-beat",
            "name": "NVDA earnings beat",
            "slug": "nvda-earnings-beat",
            "theme": "equities",
            "description": "Probability NVDA beats EPS this quarter.",
        },
        {
            "id": "iran-strait-attack",
            "name": "Iran attack in Strait of Hormuz",
            "slug": "iran-strait",
            "theme": "geopolitics",
            "description": "Iran-related strait of Hormuz incident.",
        },
    ]


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


def test_extract_entities_trump_china_tariffs() -> None:
    text = "Trump signs executive order on China tariffs"
    out = extract_entities(text)
    assert "Trump" in out["politicians"]
    assert "China" in out["countries"]
    assert "ExecutiveOrder" in out["events"]
    assert "Tariff" in out["events"]


def test_extract_entities_finds_ticker_and_commodity() -> None:
    text = "NVDA jumps as Bitcoin and oil rally on Fed signal"
    out = extract_entities(text)
    assert "NVDA" in out["tickers"]
    assert "BTC" in out["commodities"]
    assert "Oil" in out["commodities"]
    # No politician here, but "Fed" alone is not a politician — Powell isn't
    # named, so politicians may be empty.
    assert out["politicians"] == [] or "Powell" in out["politicians"]


def test_extract_entities_filters_common_words() -> None:
    # "US" and "CEO" should not turn into tickers.
    out = extract_entities("The US CEO of Apple gave a speech")
    assert "US" not in out["tickers"]
    assert "CEO" not in out["tickers"]
    assert "AAPL" not in out["tickers"]  # "Apple" string, not ticker form
    assert "USA" in out["countries"]


def test_extract_entities_dollar_ticker_form() -> None:
    out = extract_entities("$F surges 8% after earnings")
    assert "F" in out["tickers"]


def test_extract_entities_empty_text() -> None:
    out = extract_entities("")
    assert out["tickers"] == []
    assert out["politicians"] == []
    assert out["countries"] == []
    assert out["events"] == []
    assert out["commodities"] == []


def test_extract_entities_multiword_politician() -> None:
    out = extract_entities("Donald Trump met Vladimir Putin in Geneva")
    assert "Trump" in out["politicians"]
    assert "Putin" in out["politicians"]
    # Each appears only once even though "Donald Trump" + "Trump" both match.
    assert out["politicians"].count("Trump") == 1


def test_all_entities_flattens() -> None:
    text = "Trump tariffs on China hit NVDA earnings"
    flat = all_entities(extract_entities(text))
    assert "Trump" in flat
    assert "China" in flat
    assert "NVDA" in flat
    assert "Tariff" in flat


# ---------------------------------------------------------------------------
# score_factor_match
# ---------------------------------------------------------------------------


def test_score_factor_match_strong_match(small_catalog: list[dict]) -> None:
    text = "Trump signs executive order on China tariffs ahead of summit"
    factor = small_catalog[0]  # trump-tariffs-china
    score = score_factor_match(text, factor)
    assert score > 0.5, f"expected strong match, got {score:.3f}"


def test_score_factor_match_no_match(small_catalog: list[dict]) -> None:
    text = "Local bakery wins prize in dessert competition"
    factor = small_catalog[0]
    score = score_factor_match(text, factor)
    assert score < 0.2


def test_score_factor_match_macro_theme_bonus(small_catalog: list[dict]) -> None:
    text = "FOMC signals March rate cut"
    factor = small_catalog[1]  # fed-cut-march, theme=macro
    score = score_factor_match(text, factor)
    assert score > 0.5


def test_score_factor_match_btc(small_catalog: list[dict]) -> None:
    text = "Bitcoin rallies past 130k as ETF inflows surge"
    factor = small_catalog[2]  # btc-150k
    score = score_factor_match(text, factor)
    assert score > 0.3


# ---------------------------------------------------------------------------
# tag_news_to_factors
# ---------------------------------------------------------------------------


def test_tag_news_to_factors_three_news_five_factors(small_catalog: list[dict]) -> None:
    items = [
        {"title": "Trump signs new China tariffs"},
        {"title": "Bitcoin surges past 130k"},
        {"title": "Quiet trading day on Wall Street"},
    ]
    out = tag_news_to_factors(items, small_catalog, threshold=0.3)
    assert len(out) == 3

    # 1st: should match trump-tariffs-china
    matches_0 = {m["factor_id"] for m in out[0]["matched_factors"]}
    assert "trump-tariffs-china" in matches_0

    # 2nd: should match btc-150k
    matches_1 = {m["factor_id"] for m in out[1]["matched_factors"]}
    assert "btc-150k" in matches_1

    # 3rd: too generic, should be empty (or near-empty).
    assert len(out[2]["matched_factors"]) <= 1


def test_tag_news_to_factors_empty_text(small_catalog: list[dict]) -> None:
    out = tag_news_to_factors([{"title": ""}], small_catalog)
    assert out[0]["matched_factors"] == []


def test_tag_news_to_factors_threshold_strict(small_catalog: list[dict]) -> None:
    """A high threshold should filter out unrelated factors but keep best ones."""
    items = [{"title": "Local bakery wins prize"}]
    strict = tag_news_to_factors(items, small_catalog, threshold=0.5)
    # An unrelated headline yields zero matches at a strict threshold.
    assert all(len(o["matched_factors"]) == 0 for o in strict)


def test_tag_news_to_factors_default_catalog_no_crash() -> None:
    # Must not raise even when factors.yml is huge.
    out = tag_news_to_factors([{"title": "Trump tariffs"}], factor_catalog=None)
    assert isinstance(out, list)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# enhanced_sentiment
# ---------------------------------------------------------------------------


def test_enhanced_sentiment_positive() -> None:
    out = enhanced_sentiment("NVDA earnings beat expectations, boosting investor optimism")
    assert out["overall_sentiment"] > 0
    assert out["dominant"] == "positive"
    # NVDA appears in the same sentence -> per-entity score should be positive.
    assert out["sentiment_per_entity"].get("NVDA", 0) > 0


def test_enhanced_sentiment_negative() -> None:
    out = enhanced_sentiment("Iran attack on tankers triggers panic selloff and crash")
    assert out["overall_sentiment"] < 0
    assert out["dominant"] == "negative"
    assert out["sentiment_per_entity"].get("Iran", 0) < 0


def test_enhanced_sentiment_aspect_split() -> None:
    text = (
        "NVDA earnings beat expectations and shares surged. "
        "Meanwhile Iran attack and panic dragged oil markets into a crash."
    )
    out = enhanced_sentiment(text)
    spe = out["sentiment_per_entity"]
    # NVDA in positive sentence, Iran in negative sentence.
    assert spe.get("NVDA", 0) > 0
    assert spe.get("Iran", 0) < 0


def test_enhanced_sentiment_neutral() -> None:
    out = enhanced_sentiment("The meeting will take place next week")
    assert out["dominant"] == "neutral"
    assert abs(out["overall_sentiment"]) < 0.05


# ---------------------------------------------------------------------------
# Router endpoints
# ---------------------------------------------------------------------------


def test_post_tag_endpoint(client: TestClient) -> None:
    resp = client.post(
        "/news/tag",
        json={
            "news_text": "Trump signs executive order on China tariffs",
            "threshold": DEFAULT_THRESHOLD,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["news_text"]
    assert "Trump" in body["entities"]["politicians"]
    assert "China" in body["entities"]["countries"]
    assert "Tariff" in body["entities"]["events"]
    assert "sentiment" in body
    # matched_factors may be empty if the default factors.yml has no matching
    # slug, but the field must be present and a list.
    assert isinstance(body["matched_factors"], list)


def test_post_tag_endpoint_with_factor_filter(client: TestClient) -> None:
    """Filtering to a non-existent factor id should yield zero matches."""
    resp = client.post(
        "/news/tag",
        json={
            "news_text": "Trump tariffs",
            "factor_ids": ["__nope__"],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["matched_factors"] == []


def test_post_tag_batch_endpoint(client: TestClient) -> None:
    resp = client.post(
        "/news/tag-batch",
        json={
            "news_items": [
                {"title": "Trump signs new China tariffs"},
                {"title": "Bitcoin rallies past 130k"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_items"] == 2
    assert len(body["results"]) == 2


def test_get_factor_recent_empty_returns_zero(client: TestClient) -> None:
    resp = client.get("/news/factor/some-factor/recent?hours=24&n=20")
    assert resp.status_code == 200
    body = resp.json()
    assert body["factor_id"] == "some-factor"
    assert body["items"] == []
    assert body["n_returned"] == 0


def test_get_factor_recent_after_record(client: TestClient) -> None:
    # Seed the in-memory recent store via the public helper.
    record_tagged_items(
        [
            {
                "news_item": {
                    "title": "Trump tariffs",
                    "url": "https://example.com/a",
                    "ts": "2026-05-08T10:00:00Z",
                },
                "matched_factors": [
                    {
                        "factor_id": "trump-tariffs-china",
                        "factor_name": "Trump tariffs",
                        "match_score": 0.7,
                    }
                ],
            },
            {
                "news_item": {
                    "title": "Trump again",
                    "url": "https://example.com/b",
                    "ts": "2026-05-08T11:00:00Z",
                },
                "matched_factors": [
                    {
                        "factor_id": "trump-tariffs-china",
                        "factor_name": "Trump tariffs",
                        "match_score": 0.6,
                    }
                ],
            },
        ]
    )
    resp = client.get("/news/factor/trump-tariffs-china/recent?n=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_returned"] == 2
    # Newest first
    assert body["items"][0]["url"] == "https://example.com/b"


def test_get_entity_factors_endpoint(client: TestClient) -> None:
    resp = client.get("/news/entity/Trump/factors?n=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity"] == "Trump"
    assert isinstance(body["factors"], list)


# ---------------------------------------------------------------------------
# entity_factor_map JSON
# ---------------------------------------------------------------------------


def test_entity_factor_map_loads() -> None:
    emap = load_entity_factor_map()
    assert isinstance(emap, dict)
    assert "Trump" in emap
    assert "Fed" in emap
    assert "BTC" in emap
    # Internal _meta key must be filtered out.
    assert "_meta" not in emap


# ---------------------------------------------------------------------------
# news_causal_chain integration — uses the new tagger via _keyword_overlap.
# ---------------------------------------------------------------------------


def test_causal_chain_uses_tagger_for_entity_match() -> None:
    """Without literal token overlap, an entity match should still tag.

    The factor id 'trump-out-by-2027' has 'trump' in its tokens. A headline
    using "Donald Trump" but no other shared token still produces overlap
    >= 1 because the politician "Trump" maps via the entity-factor JSON.
    """
    from pfm.news_causal_chain import (
        BETA_REGISTRY,
        build_causal_chain,
        register_betas,
    )

    BETA_REGISTRY.clear()
    register_betas("trump-out-by-2027", {"DJT": 0.5})

    items = [
        {
            "title": "Donald Trump faces fresh political headwinds",
            "ts": "2026-05-08T10:00:00Z",
            "price_before": 0.30,
            "price_after": 0.50,
        },
    ]
    resp = build_causal_chain("trump-out-by-2027", items, lookback_hours=48)
    assert resp["n_tagged"] == 1
    assert resp["chain"][0]["tagged_factor"] == "trump-out-by-2027"
    BETA_REGISTRY.clear()
