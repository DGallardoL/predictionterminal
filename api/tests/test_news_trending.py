"""Tests for pfm.terminal.news_trending_router (W12-21).

Verifies the trending-news aggregator end-to-end with the upstream
GDELT / Reddit / HN / RSS fetchers mocked out. The unit tests cover:

- The composite score formula on synthetic inputs.
- Recency, corroboration, and sentiment factors in isolation.
- SimHash dedupe + cross-source merging into a single cluster.
- Lookback-window filtering of stale items.
- Best-effort behaviour when individual source pipes raise.
- Cache TTL & invalidation.
- FastAPI integration via ``TestClient`` against the router only.

Run standalone with::

    pytest tests/test_news_trending.py -q --noconftest
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Make `pfm` importable when running with --noconftest (skips the
# tests/conftest fixtures that pull in optional heavy deps).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal import news_trending_router as ntr
from pfm.terminal.news_dedupe import NewsItem

UTC = UTC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(
    title: str,
    *,
    source: str = "gdelt",
    minutes_ago: int = 30,
    url: str | None = None,
    tone: float | None = None,
) -> NewsItem:
    """Build a ``NewsItem`` ``minutes_ago`` before "now"."""
    when = datetime.now(tz=UTC) - timedelta(minutes=minutes_ago)
    return NewsItem(
        title=title,
        url=url or f"https://example.com/{source}/{abs(hash(title))}",
        source=source,
        published_at=when,
        tone=tone,
    )


@pytest.fixture(autouse=True)
def _clear_response_cache():
    """Each test starts with a fresh response cache."""
    ntr.RESPONSE_CACHE.clear()
    yield
    ntr.RESPONSE_CACHE.clear()


@pytest.fixture
def patch_fetchers(monkeypatch):
    """Helper to replace SOURCE_FETCHERS with deterministic stubs."""

    def _apply(stubs: dict[str, Callable[[int], list[NewsItem]]]) -> None:
        # Replace by reassignment so the router picks up the patched table.
        monkeypatch.setattr(ntr, "SOURCE_FETCHERS", dict(stubs))

    return _apply


# ---------------------------------------------------------------------------
# Pure scoring
# ---------------------------------------------------------------------------


def test_compute_score_basic_formula():
    # 1 source, 1h ago, neutral sentiment → (1/1)*log(2)*1 = log(2)
    expected = math.log(2.0)
    got = ntr.compute_score(hours_since=1.0, n_sources=1, compound=0.0)
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_compute_score_recency_dominates_with_close_corroboration():
    # 1h ago, 3 sources, neutral vs 4h ago, 3 sources, neutral.
    near = ntr.compute_score(hours_since=1.0, n_sources=3, compound=0.0)
    far = ntr.compute_score(hours_since=4.0, n_sources=3, compound=0.0)
    assert near > far


def test_compute_score_corroboration_boost():
    # Same recency + sentiment; more sources wins.
    one = ntr.compute_score(hours_since=2.0, n_sources=1, compound=0.0)
    five = ntr.compute_score(hours_since=2.0, n_sources=5, compound=0.0)
    assert five > one
    # log(6)/log(2) ratio sanity check on the corroboration factor.
    assert math.isclose(five / one, math.log(6.0) / math.log(2.0), rel_tol=1e-9)


def test_compute_score_sentiment_intensity_boost():
    base = ntr.compute_score(hours_since=2.0, n_sources=2, compound=0.0)
    hot = ntr.compute_score(hours_since=2.0, n_sources=2, compound=-0.9)
    # |compound| matters, not sign — negative news is just as "trending".
    assert hot > base
    assert math.isclose(hot / base, 1.0 + 0.9, rel_tol=1e-9)


def test_compute_score_clamps_zero_hours_safely():
    # Article published in the future / right now → hours_since clamps
    # to MIN_HOURS_SINCE rather than dividing by zero.
    got = ntr.compute_score(hours_since=0.0, n_sources=1, compound=0.0)
    assert math.isfinite(got)
    assert got > 0


# ---------------------------------------------------------------------------
# rank_trending — dedupe + ranking
# ---------------------------------------------------------------------------


def test_rank_trending_merges_cross_source_duplicates(monkeypatch):
    # All three sources report the same headline → ONE cluster with
    # n_sources == 3 and sources covers all three tags.
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: 0.5)
    now = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)
    items = [
        NewsItem(
            title="Fed hikes rates after surprise CPI print",
            url="https://gdelt.example/a",
            source="gdelt",
            published_at=now - timedelta(hours=2),
        ),
        NewsItem(
            title="Fed hikes rates after surprise CPI print!",  # punctuation diff
            url="https://reddit.example/a",
            source="reddit",
            published_at=now - timedelta(hours=1, minutes=30),
        ),
        NewsItem(
            title="Fed hikes rates, after surprise CPI print.",  # punctuation diff
            url="https://hn.example/a",
            source="hn",
            published_at=now - timedelta(hours=1),
        ),
    ]
    ranked, total = ntr.rank_trending(items, now=now, limit=10)
    assert total == 1
    assert len(ranked) == 1
    cluster = ranked[0]
    assert cluster.n_sources == 3
    assert set(cluster.sources) == {"gdelt", "reddit", "hn"}
    # Earliest URL/title wins as the cluster representative.
    assert cluster.url == "https://gdelt.example/a"


def test_rank_trending_distinct_stories_are_separate_clusters(monkeypatch):
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: 0.0)
    now = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)
    items = [
        NewsItem(
            title="Fed hikes 25bps surprise CPI shock today",
            url="https://gdelt.example/fed",
            source="gdelt",
            published_at=now - timedelta(hours=2),
        ),
        NewsItem(
            title="Bitcoin breaks all-time high amid ETF inflows surge",
            url="https://gdelt.example/btc",
            source="gdelt",
            published_at=now - timedelta(hours=2),
        ),
    ]
    ranked, total = ntr.rank_trending(items, now=now, limit=10)
    assert total == 2
    titles = {r.title for r in ranked}
    assert any("Fed hikes" in t for t in titles)
    assert any("Bitcoin" in t for t in titles)


def test_rank_trending_orders_by_descending_score(monkeypatch):
    # Headline A: fresh + multi-source. Headline B: stale + single-source.
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: 0.0)
    now = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)
    items = [
        # Fresh, 3 sources — should win.
        NewsItem(
            title="ECB cuts deposit rate by surprise move today",
            url="https://gdelt.example/ecb",
            source="gdelt",
            published_at=now - timedelta(minutes=30),
        ),
        NewsItem(
            title="ECB cuts deposit rate by surprise move today!",
            url="https://reddit.example/ecb",
            source="reddit",
            published_at=now - timedelta(minutes=25),
        ),
        NewsItem(
            title="ECB cuts deposit rate by surprise move today.",
            url="https://hn.example/ecb",
            source="hn",
            published_at=now - timedelta(minutes=20),
        ),
        # Old, single source — should rank lower.
        NewsItem(
            title="Argentine peso stabilises after central bank intervention",
            url="https://rss.example/ars",
            source="rss",
            published_at=now - timedelta(hours=20),
        ),
    ]
    ranked, _ = ntr.rank_trending(items, now=now, limit=5)
    assert len(ranked) == 2
    assert ranked[0].score > ranked[1].score
    assert ranked[0].title.startswith("ECB")
    assert ranked[0].n_sources == 3


def test_rank_trending_limit_truncates_but_reports_total(monkeypatch):
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: 0.0)
    now = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)
    items = [
        NewsItem(
            title=f"Story number {i} happens in the world today now",
            url=f"https://gdelt.example/{i}",
            source="gdelt",
            published_at=now - timedelta(hours=i + 1),
        )
        for i in range(10)
    ]
    ranked, total = ntr.rank_trending(items, now=now, limit=3)
    assert total == 10
    assert len(ranked) == 3
    # Strictly decreasing scores.
    assert ranked[0].score >= ranked[1].score >= ranked[2].score


def test_rank_trending_empty_input():
    ranked, total = ntr.rank_trending([], limit=20)
    assert ranked == []
    assert total == 0


# ---------------------------------------------------------------------------
# Aggregator: build_trending with mocked sources
# ---------------------------------------------------------------------------


def test_build_trending_aggregates_across_all_sources(patch_fetchers, monkeypatch):
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: 0.2)

    def gdelt(_):
        return [
            _item("Quantum computing breakthrough announced today", source="gdelt", minutes_ago=20)
        ]

    def reddit(_):
        return [
            _item("Quantum computing breakthrough announced today", source="reddit", minutes_ago=15)
        ]

    def hn(_):
        return [
            _item("Quantum computing breakthrough announced today", source="hn", minutes_ago=10)
        ]

    def rss(_):
        # Distinct story — should NOT merge into the quantum cluster.
        return [_item("US dollar plunges on weak jobs report data", source="rss", minutes_ago=120)]

    patch_fetchers({"gdelt": gdelt, "reddit": reddit, "hn": hn, "rss": rss})
    resp = ntr.build_trending(lookback_hours=24, limit=10)
    assert isinstance(resp, ntr.TrendingResponse)
    assert resp.lookback_hours == 24
    assert resp.n_clusters == 2
    # Quantum cluster should be top because it's fresher AND has 3 sources.
    top = resp.trending[0]
    assert "Quantum" in top.title
    assert top.n_sources == 3
    assert set(top.sources) == {"gdelt", "reddit", "hn"}


def test_build_trending_single_source_dominated_by_recency(patch_fetchers, monkeypatch):
    """Even with no corroboration, freshness alone should rank items."""
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: 0.0)

    def gdelt(_):
        return [
            _item("Story A fresh news happened just moments ago today", minutes_ago=5),
            _item("Story B much older news from this morning happened", minutes_ago=300),
        ]

    patch_fetchers(
        {"gdelt": gdelt, "reddit": lambda _: [], "hn": lambda _: [], "rss": lambda _: []}
    )
    resp = ntr.build_trending(lookback_hours=24, limit=10)
    assert resp.n_clusters == 2
    assert resp.trending[0].title.startswith("Story A")


def test_build_trending_source_failure_is_isolated(patch_fetchers, monkeypatch):
    """If one source raises, the others still produce a response."""
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: 0.0)

    def boom(_):
        raise RuntimeError("upstream borked")

    def ok(_):
        return [
            _item(
                "World cup final shocks fans tonight in stunning upset",
                source="gdelt",
                minutes_ago=10,
            )
        ]

    patch_fetchers(
        {
            "gdelt": ok,
            "reddit": boom,
            "hn": boom,
            "rss": boom,
        }
    )
    resp = ntr.build_trending(lookback_hours=24, limit=10)
    # Despite three of four sources blowing up, we still get the gdelt story.
    assert resp.n_clusters == 1
    assert resp.trending[0].sources == ["gdelt"]


def test_build_trending_all_sources_empty_returns_empty(patch_fetchers):
    patch_fetchers({k: lambda _: [] for k in ("gdelt", "reddit", "hn", "rss")})
    resp = ntr.build_trending(lookback_hours=24, limit=20)
    assert resp.n_clusters == 0
    assert resp.trending == []
    assert resp.lookback_hours == 24


def test_build_trending_clips_limit_into_bounds(patch_fetchers, monkeypatch):
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: 0.0)

    def gdelt(_):
        return [
            _item(
                # Use 4 mutually-distinct fragments per row so the simhash
                # signatures don't accidentally collide on rare bit flips.
                f"Topic-{i * 7 + 3} alpha-{i * 13 + 5} beta-{i * 19 + 11} gamma-{i * 23} reports issued",
                source="gdelt",
                minutes_ago=10 + i,
            )
            for i in range(30)
        ]

    patch_fetchers(
        {"gdelt": gdelt, "reddit": lambda _: [], "hn": lambda _: [], "rss": lambda _: []}
    )
    # Bogus limit gets clamped to the max-limit.
    resp = ntr.build_trending(lookback_hours=24, limit=99999)
    assert len(resp.trending) <= ntr.MAX_LIMIT
    assert resp.n_clusters == 30


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_response_cache_round_trip():
    ntr.RESPONSE_CACHE.set("k1", {"hello": "world"})
    assert ntr.RESPONSE_CACHE.get("k1") == {"hello": "world"}
    ntr.RESPONSE_CACHE.clear()
    assert ntr.RESPONSE_CACHE.get("k1") is None


def test_response_cache_expiry(monkeypatch):
    # Force TTL=0 so any cached read is immediately stale.
    cache = ntr._ResponseCache(ttl_seconds=0)
    cache.set("k", {"x": 1})
    # ttl_seconds=0 means expires_at == now → already-expired on read.
    assert cache.get("k") is None


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ntr.router)
    return app


def test_endpoint_returns_trending_payload(patch_fetchers, monkeypatch):
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: -0.4)

    def gdelt(_):
        return [
            _item(
                "Hurricane landfall devastates coastal city overnight",
                source="gdelt",
                minutes_ago=10,
            )
        ]

    def reddit(_):
        return [
            _item(
                "Hurricane landfall devastates coastal city overnight",
                source="reddit",
                minutes_ago=8,
            )
        ]

    patch_fetchers({"gdelt": gdelt, "reddit": reddit, "hn": lambda _: [], "rss": lambda _: []})

    app = _build_app()
    client = TestClient(app)
    r = client.get("/terminal/news/trending?limit=5&hours=24")
    assert r.status_code == 200
    body = r.json()
    assert body["lookback_hours"] == 24
    assert body["n_clusters"] == 1
    assert len(body["trending"]) == 1
    item = body["trending"][0]
    assert item["n_sources"] == 2
    assert set(item["sources"]) == {"gdelt", "reddit"}
    assert item["sentiment"] == pytest.approx(-0.4)
    # Score must be positive given the formula.
    assert item["score"] > 0
    assert "first_seen" in item
    assert "checked_at" in body


def test_endpoint_validates_query_params():
    app = _build_app()
    client = TestClient(app)
    # limit must be >= 1
    assert client.get("/terminal/news/trending?limit=0").status_code == 422
    # hours capped at MAX_LOOKBACK_HOURS (168)
    assert client.get("/terminal/news/trending?hours=999").status_code == 422
    # negative hours rejected
    assert client.get("/terminal/news/trending?hours=-5").status_code == 422


def test_endpoint_is_cached_for_repeated_calls(patch_fetchers, monkeypatch):
    """Second identical call should NOT re-invoke the fetchers."""
    monkeypatch.setattr(ntr, "_sentiment_compound", lambda *a, **k: 0.0)
    n_calls = {"gdelt": 0}

    def gdelt(_):
        n_calls["gdelt"] += 1
        return [
            _item(
                "Cached headline test story for endpoint repeats again",
                source="gdelt",
                minutes_ago=20,
            )
        ]

    patch_fetchers(
        {"gdelt": gdelt, "reddit": lambda _: [], "hn": lambda _: [], "rss": lambda _: []}
    )

    app = _build_app()
    client = TestClient(app)
    r1 = client.get("/terminal/news/trending?limit=5&hours=24")
    r2 = client.get("/terminal/news/trending?limit=5&hours=24")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()
    # Fetcher only ran once thanks to the TTL cache.
    assert n_calls["gdelt"] == 1
