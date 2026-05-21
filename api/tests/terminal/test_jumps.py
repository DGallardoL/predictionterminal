"""Tests for the /terminal/jumps endpoint and its jump-detection logic.

The pipeline has two interesting layers: (1) ``detect_jumps`` is a pure
function over a price series — we exercise it on synthetic series where
we KNOW where the jumps are; (2) the endpoint joins GDELT articles to
those jumps — we mock the GDELT + price fetches and verify the join
behavior (explained vs unexplained, ordering, top-K).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal.jumps import (
    _CACHE,
    DEFAULT_MAD_K,
    DEFAULT_MIN_JUMP_PP,
    _articles_for_jump,
    _articles_for_jump_with_floor,
    detect_jumps,
    router,
)
from pfm.terminal_gdelt_news import GDELTArticle


@pytest.fixture(autouse=True)
def _clear_jumps_cache() -> None:
    """Endpoint tests share a module-level cache keyed by (slug, days, k, pp).
    Clear between runs so one test's response doesn't leak into the next."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Pure-function tests on detect_jumps
# ---------------------------------------------------------------------------


def _hourly_series(values: list[float], start: str = "2026-05-01T00:00:00Z") -> pd.Series:
    """Helper: build a UTC-indexed hourly Series from a list of probabilities."""
    idx = pd.date_range(start=start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, name="price", dtype=float)


def test_detect_jumps_flags_known_spike() -> None:
    """A series that's flat at 0.40 with one +20pp spike at hour 10 should
    return exactly one jump."""
    vals = [0.40] * 10 + [0.60] + [0.60] * 10
    s = _hourly_series(vals)
    jumps = detect_jumps(s, mad_k=2.5, min_jump_pp=3.0)
    assert len(jumps) == 1
    j = jumps[0]
    assert abs(j["price_before"] - 0.40) < 1e-9
    assert abs(j["price_after"] - 0.60) < 1e-9
    assert j["delta_pp"] == pytest.approx(20.0, abs=0.1)
    # logit(0.6) - logit(0.4) ≈ 0.81
    assert j["delta_logit"] == pytest.approx(0.811, abs=0.05)


def test_detect_jumps_ignores_micro_noise_below_floor() -> None:
    """A series with uniform noise of ±1pp should yield zero jumps even
    when the rolling MAD is tiny — the absolute floor protects users
    from a stream of meaningless detections."""
    rng = np.random.default_rng(seed=42)
    base = 0.50 + rng.normal(0, 0.01, size=60)  # σ = 1pp
    s = _hourly_series(list(base))
    jumps = detect_jumps(s, mad_k=2.5, min_jump_pp=3.0)
    # The MAD floor (3pp) means we should detect 0; the rolling MAD floor
    # (2.5×) might catch some outliers in 60 draws, but the absolute
    # floor at 3pp filters them out.
    assert all(abs(j["delta_pp"]) >= 3.0 for j in jumps)


def test_detect_jumps_returns_sorted_by_magnitude_then_caps() -> None:
    """Three spikes of different magnitude — the cap returns the biggest
    by ``|delta_logit|`` first."""
    vals = [0.50] * 5 + [0.55] + [0.55] * 5 + [0.70] + [0.70] * 5 + [0.52] + [0.52] * 5
    s = _hourly_series(vals)
    jumps = detect_jumps(s, mad_k=2.0, min_jump_pp=2.0)
    # All three present
    assert len(jumps) >= 3
    # Internal sort by |delta_logit| descending (before chronological
    # reorder which only happens in the endpoint).
    magnitudes = [abs(j["delta_logit"]) for j in jumps]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_detect_jumps_returns_empty_on_short_series() -> None:
    s = _hourly_series([0.4, 0.5])  # len < 3 minimum
    assert detect_jumps(s) == []


def test_detect_jumps_handles_extreme_probabilities() -> None:
    """A market going from 0.02 → 0.20 is a 10× move in odds — the
    Δlogit treatment should flag it even though Δp is only 18pp."""
    vals = [0.02] * 10 + [0.20] + [0.20] * 10
    s = _hourly_series(vals)
    jumps = detect_jumps(s, mad_k=2.5, min_jump_pp=3.0)
    assert len(jumps) == 1
    # logit(0.2) - logit(0.02) ≈ -1.39 - (-3.89) = 2.50
    assert jumps[0]["delta_logit"] == pytest.approx(2.50, abs=0.1)


def test_detect_jumps_signs_direction_correctly() -> None:
    up_vals = [0.30] * 8 + [0.55] + [0.55] * 8
    down_vals = [0.55] * 8 + [0.30] + [0.30] * 8
    up = detect_jumps(_hourly_series(up_vals), mad_k=2.0, min_jump_pp=3.0)
    down = detect_jumps(_hourly_series(down_vals), mad_k=2.0, min_jump_pp=3.0)
    assert up[0]["delta_logit"] > 0
    assert down[0]["delta_logit"] < 0


# ---------------------------------------------------------------------------
# Article-matching helper
# ---------------------------------------------------------------------------


def _make_article(
    ts: str, title: str = "Generic headline", source: str = "test.com"
) -> GDELTArticle:
    return GDELTArticle(
        ts=ts,
        title=title,
        source=source,
        country="us",
        language="english",
        tone=0.0,
        url=f"https://{source}/x",
    )


def test_articles_for_jump_picks_in_window_only() -> None:
    jump_ts = pd.Timestamp("2026-05-01T12:00:00Z")
    scored = [
        # Within [-2h, +1h]
        (_make_article("2026-05-01T11:30:00Z", "in-window 1"), 0.8, ["foo"]),
        (_make_article("2026-05-01T12:30:00Z", "in-window 2"), 0.6, ["bar"]),
        # Outside the window
        (_make_article("2026-05-01T09:00:00Z", "too early"), 0.9, ["foo"]),
        (_make_article("2026-05-01T15:00:00Z", "too late"), 0.9, ["foo"]),
    ]
    picked, n_window = _articles_for_jump(jump_ts, scored, top_k=5)
    assert n_window == 2
    headlines = {p.headline for p in picked}
    assert headlines == {"in-window 1", "in-window 2"}


def test_articles_for_jump_ranks_by_relevance_times_proximity() -> None:
    """A perfectly-on-time article with 0.5 relevance should beat a 2h-distant
    article with 0.9 relevance (proximity decay halves the score every 2h)."""
    jump_ts = pd.Timestamp("2026-05-01T12:00:00Z")
    scored = [
        (_make_article("2026-05-01T12:00:00Z", "on-time"), 0.5, []),
        (_make_article("2026-05-01T10:00:00Z", "two-hours-early"), 0.9, []),
    ]
    picked, _ = _articles_for_jump(jump_ts, scored, top_k=2)
    assert picked[0].headline == "on-time"


def test_articles_for_jump_caps_to_top_k() -> None:
    jump_ts = pd.Timestamp("2026-05-01T12:00:00Z")
    scored = []
    for i in range(10):
        scored.append(
            (
                _make_article(f"2026-05-01T11:{i:02d}:00Z", f"hl-{i}", source=f"s{i}.com"),
                0.5 + i * 0.04,
                ["term"],
            )
        )
    picked, n_window = _articles_for_jump(jump_ts, scored, top_k=3)
    assert n_window == 10
    assert len(picked) == 3


# ---------------------------------------------------------------------------
# Endpoint integration test (mocked fetches)
# ---------------------------------------------------------------------------


class _StubMeta:
    def __init__(self) -> None:
        self.question = "Will Trump win the 2024 election?"
        self.yes_token_id = "1234567890"


class _StubPoly:
    def __init__(self) -> None:
        self._client = None  # not used by mocks
        self.clob_url = "https://clob.example"

    def get_market_metadata(self, slug: str) -> _StubMeta:
        return _StubMeta()


def _stub_prices(*_args, **_kwargs) -> pd.Series:
    # 24h flat at 0.40 + spike to 0.60 at hour 12 + flat
    return _hourly_series([0.40] * 12 + [0.60] + [0.60] * 11)


def _stub_articles(*_args, **_kwargs) -> list[GDELTArticle]:
    # Two articles: one in the window of the spike, one far away.
    return [
        GDELTArticle(
            ts="2026-05-01T11:30:00Z",
            title="Trump leads in new poll — election odds shift",
            source="reuters.com",
            country="us",
            language="english",
            tone=2.3,
            url="https://reuters.example/a",
        ),
        GDELTArticle(
            ts="2026-04-30T08:00:00Z",
            title="Generic political coverage from yesterday",
            source="cnn.com",
            country="us",
            language="english",
            tone=0.0,
            url="https://cnn.example/b",
        ),
    ]


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)

    class _State:
        polymarket_client = _StubPoly()

    app.state.poly = _StubPoly()  # type: ignore[assignment]
    return TestClient(app)


def test_endpoint_returns_jump_with_attached_article() -> None:
    with (
        patch("pfm.terminal.jumps._fetch_hourly_prices", _stub_prices),
        patch("pfm.terminal.jumps._fetch_gdelt", _stub_articles),
    ):
        c = _client()
        r = c.get("/terminal/jumps/trump-2024?days=2&mad_k=2.0&min_jump_pp=3")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["slug"] == "trump-2024"
    assert j["n_jumps"] == 1
    jump = j["jumps"][0]
    assert jump["explained"] is True
    assert jump["delta_pp"] == pytest.approx(20.0, abs=0.1)
    assert len(jump["top_articles"]) == 1
    assert jump["top_articles"][0]["headline"].startswith("Trump leads")
    assert jump["top_articles"][0]["seconds_from_jump"] < 0  # article preceded jump


def test_endpoint_marks_unexplained_when_no_article_in_window() -> None:
    """Same spike at hour 12, but the only article is days away — the
    jump should be returned with explained=False."""

    def far_articles(*_args, **_kwargs) -> list[GDELTArticle]:
        return [
            GDELTArticle(
                ts="2026-04-20T08:00:00Z",
                title="Trump rally last week",
                source="reuters.com",
                country="us",
                language="english",
                tone=0.0,
                url="https://reuters.example/c",
            ),
        ]

    with (
        patch("pfm.terminal.jumps._fetch_hourly_prices", _stub_prices),
        patch("pfm.terminal.jumps._fetch_gdelt", far_articles),
    ):
        c = _client()
        r = c.get("/terminal/jumps/trump-2024?days=2&mad_k=2.0&min_jump_pp=3")
    assert r.status_code == 200
    j = r.json()
    assert j["n_jumps"] == 1
    assert j["jumps"][0]["explained"] is False
    assert j["jumps"][0]["n_articles"] == 0
    assert j["n_explained"] == 0


def test_endpoint_interpretation_summarizes_counts() -> None:
    with (
        patch("pfm.terminal.jumps._fetch_hourly_prices", _stub_prices),
        patch("pfm.terminal.jumps._fetch_gdelt", _stub_articles),
    ):
        c = _client()
        r = c.get("/terminal/jumps/trump-2024?days=2&mad_k=2.0&min_jump_pp=3")
    j = r.json()
    assert "jumps detected" in j["interpretation"]
    assert "explained by news" in j["interpretation"]


def test_endpoint_validates_query_params() -> None:
    c = _client()
    # mad_k must be in [1, 10]
    r = c.get("/terminal/jumps/x?mad_k=0.1")
    assert r.status_code == 422
    # days must be in [1, 90]
    r = c.get("/terminal/jumps/x?days=0")
    assert r.status_code == 422


def test_endpoint_response_includes_sentiment_fields() -> None:
    """Every explained jump must carry news_sentiment_score / _label /
    sentiment_alignment; every JumpArticle must carry sentiment_score / _label.
    These are first-class API contract fields the frontend depends on."""
    with (
        patch("pfm.terminal.jumps._fetch_hourly_prices", _stub_prices),
        patch("pfm.terminal.jumps._fetch_gdelt", _stub_articles),
    ):
        c = _client()
        r = c.get("/terminal/jumps/trump-2024?days=2&mad_k=2.0&min_jump_pp=3")
    assert r.status_code == 200
    j = r.json()
    assert j["n_jumps"] == 1
    jump = j["jumps"][0]
    # Per-jump aggregate fields exist
    for k in ("news_sentiment_score", "news_sentiment_label", "sentiment_alignment"):
        assert k in jump, f"missing {k} in jump"
    assert jump["news_sentiment_label"] in {"positive", "negative", "neutral"}
    assert jump["sentiment_alignment"] in {"agrees", "disagrees", "neutral"}
    assert -1.0 <= jump["news_sentiment_score"] <= 1.0
    # Per-article fields exist on each top_article
    for a in jump["top_articles"]:
        assert "sentiment_score" in a
        assert "sentiment_label" in a
        assert -1.0 <= a["sentiment_score"] <= 1.0
        assert a["sentiment_label"] in {"positive", "negative", "neutral"}


def test_endpoint_sentiment_alignment_matches_direction() -> None:
    """A positive-tone bullish headline matched to an upward jump should
    yield sentiment_alignment='agrees'. Pins the cross-component contract."""

    def bullish_articles(*_args, **_kwargs):
        return [
            GDELTArticle(
                ts="2026-05-01T11:30:00Z",
                title="Trump surges in new poll as rally boosts election odds",
                source="reuters.com",
                country="us",
                language="english",
                tone=2.5,
                url="https://example.com/x",
            ),
        ]

    with (
        patch("pfm.terminal.jumps._fetch_hourly_prices", _stub_prices),
        patch("pfm.terminal.jumps._fetch_gdelt", bullish_articles),
    ):
        c = _client()
        r = c.get("/terminal/jumps/trump-2024?days=2&mad_k=2.0&min_jump_pp=3")
    assert r.status_code == 200
    jump = r.json()["jumps"][0]
    assert jump["direction"] == "up"
    assert jump["news_sentiment_label"] == "positive"
    assert jump["sentiment_alignment"] == "agrees"


def test_default_thresholds_match_strict_settings() -> None:
    """User asked 'solo saltos bruscos, no spamear' (2026-05-16) — defaults must
    stay at ≥5pp absolute AND ≥3σ rolling. Pin to catch accidental loosening."""
    assert DEFAULT_MIN_JUMP_PP == 5.0
    assert DEFAULT_MAD_K == 3.0


def test_articles_for_jump_with_floor_drops_pre_market_news() -> None:
    """No article older than ``market_start_ts`` may be returned — even if
    it falls inside the [-2h, +1h] proximity window."""
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T08:00:00Z")  # market started 4h before jump
    scored = [
        # Pre-market: 30 min before market started, ignored even though it's
        # within the [-2h, +1h] window from jump_ts.
        (
            _make_article("2026-05-15T07:30:00Z", "stale wire from before market"),
            0.9,
            ["term"],
        ),
        # Post-market and near jump: should appear.
        (
            _make_article("2026-05-15T11:30:00Z", "news right before the jump"),
            0.8,
            ["term"],
        ),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert len(picked) == 1
    assert picked[0].headline == "news right before the jump"


def test_articles_for_jump_with_floor_without_floor_matches_normal_behavior() -> None:
    """When ``market_start_ts`` is None, behavior must match the plain helper."""
    jump_ts = pd.Timestamp("2026-05-01T12:00:00Z")
    scored = [
        (_make_article("2026-05-01T11:00:00Z", "old enough"), 0.5, []),
        (_make_article("2026-05-01T12:30:00Z", "post-jump"), 0.5, []),
    ]
    a, n_a = _articles_for_jump(jump_ts, scored)
    b, n_b = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=None)
    assert n_a == n_b == 2
    assert {x.headline for x in a} == {x.headline for x in b}


def test_endpoint_returns_404_when_market_not_found() -> None:
    class _BadPoly:
        _client = None
        clob_url = "https://clob.example"

        def get_market_metadata(self, slug: str):
            raise ValueError("no such slug")

    app = FastAPI()
    app.include_router(router)
    app.state.poly = _BadPoly()
    c = TestClient(app)
    r = c.get("/terminal/jumps/does-not-exist")
    assert r.status_code == 404
    assert "market not found" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Additional edge cases (filling coverage gaps in jumps.py)
# ---------------------------------------------------------------------------


def test_articles_for_jump_with_floor_market_start_equals_article_ts() -> None:
    """Boundary: an article whose ts is **exactly** equal to ``market_start_ts``
    must be admitted (the floor is a half-open lower bound)."""
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T10:00:00Z")
    scored = [
        # Article ts == market_start_ts — must be admitted.
        (
            _make_article("2026-05-15T10:00:00Z", "exact-match"),
            0.8,
            ["term"],
        ),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert len(picked) == 1
    assert picked[0].headline == "exact-match"


def test_articles_for_jump_with_floor_market_start_in_future_drops_all() -> None:
    """If ``market_start_ts`` is in the future (after the jump and after all
    articles), every candidate falls below the floor → empty result."""
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    # Market start ts is AFTER the jump itself — pathological case but the
    # floor must not crash; it clamps lo > hi, leaving no admitted articles.
    market_start = pd.Timestamp("2026-05-16T00:00:00Z")
    scored = [
        (
            _make_article("2026-05-15T11:30:00Z", "pre-future-market"),
            0.8,
            ["term"],
        ),
        (
            _make_article("2026-05-15T12:30:00Z", "post-jump"),
            0.7,
            ["term"],
        ),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 0
    assert picked == []


def test_articles_for_jump_with_floor_clamps_window_to_market_start() -> None:
    """When ``market_start_ts > lo`` but ``< jump_ts``, the window is clamped:
    articles between [market_start, jump_ts + 1h] should still appear."""
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    # market_start is at jump_ts - 30 min — clamps the lo edge from
    # (jump - 2h) up to (jump - 30 min).
    market_start = pd.Timestamp("2026-05-15T11:30:00Z")
    scored = [
        # Falls inside the [-2h, +1h] window but BEFORE market_start.
        (
            _make_article("2026-05-15T11:00:00Z", "before-clamp"),
            0.9,
            ["term"],
        ),
        # Falls after market_start, inside the clamped window.
        (
            _make_article("2026-05-15T11:45:00Z", "after-clamp"),
            0.8,
            ["term"],
        ),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert picked[0].headline == "after-clamp"


def test_endpoint_503_when_polymarket_client_missing() -> None:
    """The Depends(_get_polymarket_client) chain must raise 503 when
    ``request.app.state.poly`` was not set on startup."""
    app = FastAPI()
    app.include_router(router)
    # NOTE: deliberately no app.state.poly
    c = TestClient(app)
    r = c.get("/terminal/jumps/anything")
    assert r.status_code == 503


def test_endpoint_serves_response_from_cache_on_repeat_call() -> None:
    """First call populates the module cache; second call must short-circuit
    on the cache lookup (covers the early-return branch of `get_jumps`)."""
    with (
        patch("pfm.terminal.jumps._fetch_hourly_prices", _stub_prices),
        patch("pfm.terminal.jumps._fetch_gdelt", _stub_articles),
    ):
        c = _client()
        r1 = c.get("/terminal/jumps/trump-2024?days=2&mad_k=2.0&min_jump_pp=3")
        # Drop the price/article stubs so a non-cached second call would raise.
        # We do not nest the second call inside the patch context to prove
        # the cache served it.
    assert r1.status_code == 200
    c2 = _client()
    r2 = c2.get("/terminal/jumps/trump-2024?days=2&mad_k=2.0&min_jump_pp=3")
    assert r2.status_code == 200
    assert r2.json()["n_jumps"] == r1.json()["n_jumps"]


def test_endpoint_uses_market_start_date_metadata_to_drop_pre_market_news() -> None:
    """If meta.start_date is set, no article from before that ts is attached.
    This wires `_articles_for_jump_with_floor` into the endpoint contract."""

    class _MetaWithStart:
        question = "Will Trump win the 2024 election?"
        yes_token_id = "1234567890"
        start_date = "2026-05-01T11:00:00Z"

    class _PolyWithStart:
        _client = None
        clob_url = "https://clob.example"

        def get_market_metadata(self, slug: str):
            return _MetaWithStart()

    def _articles_pre_and_post(*_args, **_kwargs):
        # The pre-market article sits within the [-2h, +1h] window of the
        # synthetic jump at hour 12 (10:30 is 90 min before) but should be
        # filtered by the market_start_ts=11:00 floor. The 11:30 article is
        # admitted.
        return [
            GDELTArticle(
                ts="2026-05-01T10:30:00Z",
                title="Pre-market wire (must be dropped)",
                source="reuters.com",
                country="us",
                language="english",
                tone=0.0,
                url="https://example.com/pre",
            ),
            GDELTArticle(
                ts="2026-05-01T11:30:00Z",
                title="Trump leads — election news (allowed)",
                source="reuters.com",
                country="us",
                language="english",
                tone=2.0,
                url="https://example.com/post",
            ),
        ]

    app = FastAPI()
    app.include_router(router)
    app.state.poly = _PolyWithStart()
    with (
        patch("pfm.terminal.jumps._fetch_hourly_prices", _stub_prices),
        patch("pfm.terminal.jumps._fetch_gdelt", _articles_pre_and_post),
    ):
        c = TestClient(app)
        r = c.get("/terminal/jumps/trump-2024?days=2&mad_k=2.0&min_jump_pp=3")
    assert r.status_code == 200, r.text
    jump = r.json()["jumps"][0]
    # Only the post-market article must appear among top_articles.
    headlines = {a["headline"] for a in jump["top_articles"]}
    assert "Pre-market wire (must be dropped)" not in headlines
    assert any("election news" in h for h in headlines)


def test_endpoint_502_on_polymarket_http_error() -> None:
    """An ``httpx.HTTPError`` from get_market_metadata maps to 502 (gamma
    upstream error), distinct from the 404 path covered above."""
    import httpx as _httpx

    class _FlakyPoly:
        _client = None
        clob_url = "https://clob.example"

        def get_market_metadata(self, slug: str):
            raise _httpx.ConnectError("gamma down")

    app = FastAPI()
    app.include_router(router)
    app.state.poly = _FlakyPoly()
    c = TestClient(app)
    r = c.get("/terminal/jumps/any-slug")
    assert r.status_code == 502
    assert "polymarket gamma error" in r.json()["detail"]


def test_to_gdelt_shape_returns_none_for_empty_inputs() -> None:
    """`_to_gdelt_shape` must reject empty ts/title and unparseable ts —
    pure function gate covered directly."""
    from pfm.terminal.jumps import _to_gdelt_shape

    # Empty ts → None
    assert _to_gdelt_shape(ts="", title="x", source="s", url=None) is None
    # Empty title → None
    assert _to_gdelt_shape(ts="2026-05-01T00:00:00Z", title="", source="s", url=None) is None
    # Unparseable ts → None
    assert _to_gdelt_shape(ts="not-a-date", title="x", source="s", url=None) is None


def test_to_gdelt_shape_builds_envelope_for_well_formed_inputs() -> None:
    """Happy path: well-formed inputs produce a GDELTArticle with the
    expected fields preserved."""
    from pfm.terminal.jumps import _to_gdelt_shape

    art = _to_gdelt_shape(
        ts="2026-05-01T12:00:00Z",
        title="A headline",
        source="reddit:r/wallstreetbets",
        url="https://reddit.com/r/x",
        tone=1.5,
    )
    assert art is not None
    assert art.title == "A headline"
    assert art.source == "reddit:r/wallstreetbets"
    assert art.url == "https://reddit.com/r/x"
    assert art.tone == pytest.approx(1.5)


def test_to_gdelt_shape_defaults_url_to_source_when_missing() -> None:
    """When ``url`` is None we should still build the envelope using a
    synthesised ``https://{source}`` URL (downstream UI is happier with
    *some* URL than none)."""
    from pfm.terminal.jumps import _to_gdelt_shape

    art = _to_gdelt_shape(
        ts="2026-05-01T12:00:00Z",
        title="No URL headline",
        source="hn",
        url=None,
    )
    assert art is not None
    assert art.url == "https://hn"


def test_detect_jumps_handles_empty_series_safely() -> None:
    """Explicit guard: empty series returns ``[]`` (covered alongside the
    short-series guard at line 237)."""
    empty = pd.Series(dtype=float)
    assert detect_jumps(empty) == []
