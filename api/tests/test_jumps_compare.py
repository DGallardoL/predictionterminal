"""Tests for ``GET /terminal/jumps/compare`` — aligned multi-slug jumps.

We never hit Polymarket / GDELT here: ``get_jumps`` is monkey-patched to
return pre-built :class:`Jump` payloads, and the FastAPI app is composed
with the standalone router so this test does not depend on
``pfm.main`` (whose route file is owned by other concurrent sessions).
"""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal.jumps import Jump, JumpArticle, TerminalJumpsResponse
from pfm.terminal.jumps_compare_router import (
    _CACHE,
    _article_identifier,
    _build_common_days,
    router,
)


@pytest.fixture(autouse=True)
def _clear_compare_cache() -> None:
    """Compare endpoint caches by sorted slug tuple; reset between tests."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------


def _make_article(
    headline: str,
    *,
    url: str | None = "https://news.example/x",
    ts_iso: str = "2026-04-12T12:00:00Z",
    relevance: float = 0.7,
    terms: Iterable[str] | None = None,
) -> JumpArticle:
    return JumpArticle(
        ts_iso=ts_iso,
        seconds_from_jump=0,
        headline=headline,
        source="test.com",
        url=url,
        tone=0.0,
        relevance_score=relevance,
        matched_terms=list(terms or []),
        sentiment_score=0.0,
        sentiment_label="neutral",
    )


def _make_jump(
    ts_iso: str,
    *,
    delta_pp: float = 8.2,
    articles: list[JumpArticle] | None = None,
    direction: str = "up",
) -> Jump:
    arts = list(articles or [])
    return Jump(
        ts_iso=ts_iso,
        price_before=0.40,
        price_after=0.48,
        delta_pp=delta_pp,
        delta_logit=0.2 if direction == "up" else -0.2,
        z_score=4.0,
        direction=direction,
        explained=bool(arts),
        n_articles=len(arts),
        top_articles=arts,
        news_sentiment_score=0.0,
        news_sentiment_label="neutral",
        sentiment_alignment="neutral",
    )


class _StubPoly:
    def __init__(self) -> None:
        self._client = None
        self.clob_url = "https://clob.example"


def _client_with_router() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.poly = _StubPoly()  # type: ignore[assignment]
    return TestClient(app)


def _stub_jumps_factory(jumps_by_slug: dict[str, list[Jump]]):
    async def _fake_get_jumps(
        request,
        slug: str,
        days: int = 14,
        mad_k: float = 3.0,
        min_jump_pp: float = 5.0,
        poly=None,
    ) -> TerminalJumpsResponse:
        del request, poly  # not used; signature must match get_jumps
        js = jumps_by_slug.get(slug, [])
        return TerminalJumpsResponse(
            slug=slug,
            days=days,
            threshold_mad_k=mad_k,
            threshold_min_jump_pp=min_jump_pp,
            n_jumps=len(js),
            n_explained=sum(1 for j in js if j.explained),
            explained_pct=100.0 if js else 0.0,
            jumps=js,
            interpretation="stubbed",
        )

    return _fake_get_jumps


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


def test_article_identifier_prefers_url() -> None:
    """URL wins over headline; URL-only callers get an exact-match identifier."""
    ident = _article_identifier("https://x.com/a", "Some headline")
    assert ident == "u:https://x.com/a"


def test_article_identifier_falls_back_to_simhash_when_url_missing() -> None:
    """No URL → identifier comes from headline (SimHash bucket if available)."""
    ident_a = _article_identifier(None, "Fed holds rates steady")
    ident_b = _article_identifier(None, "Fed holds rates steady")
    assert ident_a is not None
    assert ident_a == ident_b
    # Different enough headline should land in a different bucket.
    ident_c = _article_identifier(None, "Completely unrelated topic about sports")
    assert ident_c is not None


def test_article_identifier_returns_none_for_blank_input() -> None:
    assert _article_identifier(None, None) is None
    assert _article_identifier("", "") is None


def test_build_common_days_filters_singleton_days() -> None:
    """A date with a jump in only one slug is excluded from common_days."""
    jumps_by_slug = {
        "a": [_make_jump("2026-04-12T12:00:00Z", delta_pp=8.2)],
        "b": [_make_jump("2026-04-15T12:00:00Z", delta_pp=4.0)],
    }
    days = _build_common_days(jumps_by_slug)
    assert days == []  # no shared date


def test_build_common_days_picks_largest_magnitude_delta() -> None:
    """When a slug has multiple jumps on the same day, the largest |delta_pp|
    wins (sign preserved)."""
    jumps_by_slug = {
        "a": [
            _make_jump("2026-04-12T10:00:00Z", delta_pp=3.0),
            _make_jump("2026-04-12T18:00:00Z", delta_pp=-7.5),  # largest magnitude
        ],
        "b": [_make_jump("2026-04-12T12:00:00Z", delta_pp=5.6)],
    }
    days = _build_common_days(jumps_by_slug)
    assert len(days) == 1
    assert days[0].date == "2026-04-12"
    assert days[0].jumps["a"] == pytest.approx(-7.5)
    assert days[0].jumps["b"] == pytest.approx(5.6)


def test_build_common_days_counts_shared_articles_across_slugs() -> None:
    """An article URL that appears in jumps of ≥2 slugs counts once."""
    shared_art = _make_article("Trump tariff news", url="https://wire.example/1")
    other_art = _make_article("Tangential", url="https://wire.example/2")
    jumps_by_slug = {
        "a": [_make_jump("2026-04-12T10:00:00Z", articles=[shared_art, other_art])],
        "b": [_make_jump("2026-04-12T12:00:00Z", articles=[shared_art])],
        "c": [_make_jump("2026-04-12T14:00:00Z", articles=[])],
    }
    days = _build_common_days(jumps_by_slug)
    assert len(days) == 1
    assert days[0].shared_news_count == 1  # only the shared URL


# ---------------------------------------------------------------------------
# Endpoint: shape + happy path
# ---------------------------------------------------------------------------


def test_three_slugs_return_all_three_keyed_in_jumps_by_slug() -> None:
    """Three slugs in → all three keys in jumps_by_slug, in input order."""
    shared_art = _make_article("Trump tariff", url="https://w/1")
    jumps_by_slug = {
        "a": [_make_jump("2026-04-12T10:00:00Z", delta_pp=8.2, articles=[shared_art])],
        "b": [],
        "c": [_make_jump("2026-04-12T11:00:00Z", delta_pp=5.6, articles=[shared_art])],
    }
    with patch("pfm.terminal.jumps_compare_router.get_jumps", _stub_jumps_factory(jumps_by_slug)):
        c = _client_with_router()
        r = c.get("/terminal/jumps/compare?slugs=a,b,c")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slugs"] == ["a", "b", "c"]
    # all three keys present
    assert set(body["jumps_by_slug"].keys()) == {"a", "b", "c"}
    # b had no jumps but is still keyed
    assert body["jumps_by_slug"]["b"] == []


def test_common_days_correctly_identified() -> None:
    """Days with ≥2 slugs having a jump appear in common_days; others don't."""
    shared = _make_article("Macro headline", url="https://w/macro")
    jumps_by_slug = {
        "a": [
            # day 1 — shared with c
            _make_jump("2026-04-12T08:00:00Z", delta_pp=8.2, articles=[shared]),
            # day 2 — solo, must NOT show up
            _make_jump("2026-04-15T08:00:00Z", delta_pp=4.0),
        ],
        "b": [],
        "c": [_make_jump("2026-04-12T20:00:00Z", delta_pp=5.6, articles=[shared])],
    }
    with patch("pfm.terminal.jumps_compare_router.get_jumps", _stub_jumps_factory(jumps_by_slug)):
        cli = _client_with_router()
        r = cli.get("/terminal/jumps/compare?slugs=a,b,c")
    body = r.json()
    days = body["common_days"]
    assert len(days) == 1
    day = days[0]
    assert day["date"] == "2026-04-12"
    assert day["jumps"]["a"] == pytest.approx(8.2)
    assert day["jumps"]["b"] is None
    assert day["jumps"]["c"] == pytest.approx(5.6)
    assert day["shared_news_count"] == 1


def test_slugs_over_cap_returns_400() -> None:
    """More than 8 slugs is rejected with a 400."""
    slugs = ",".join(f"slug-{i}" for i in range(9))
    cli = _client_with_router()
    r = cli.get(f"/terminal/jumps/compare?slugs={slugs}")
    assert r.status_code == 400
    assert "too many slugs" in r.json()["detail"]


def test_single_slug_returns_single_slug_response() -> None:
    """One slug in → returns a valid response; common_days is empty by definition."""
    jumps_by_slug = {
        "a": [_make_jump("2026-04-12T10:00:00Z", delta_pp=8.2)],
    }
    with patch("pfm.terminal.jumps_compare_router.get_jumps", _stub_jumps_factory(jumps_by_slug)):
        cli = _client_with_router()
        r = cli.get("/terminal/jumps/compare?slugs=a")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slugs"] == ["a"]
    assert list(body["jumps_by_slug"].keys()) == ["a"]
    assert body["common_days"] == []  # need ≥2 slugs for a "common" day


def test_unknown_slug_returns_empty_jumps_for_that_slug() -> None:
    """An unknown slug (no entry in the stub) appears with empty jumps."""
    # Stub only knows about "a"; "ghost" must still appear keyed-with-empty.
    jumps_by_slug = {
        "a": [_make_jump("2026-04-12T10:00:00Z", delta_pp=8.2)],
    }
    with patch("pfm.terminal.jumps_compare_router.get_jumps", _stub_jumps_factory(jumps_by_slug)):
        cli = _client_with_router()
        r = cli.get("/terminal/jumps/compare?slugs=a,ghost")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["jumps_by_slug"].keys()) == {"a", "ghost"}
    assert body["jumps_by_slug"]["ghost"] == []
    # And no common days because only "a" has a jump.
    assert body["common_days"] == []


def test_cache_hit_short_circuits_get_jumps() -> None:
    """Second identical call must not re-invoke ``get_jumps``."""
    jumps_by_slug = {
        "a": [_make_jump("2026-04-12T10:00:00Z", delta_pp=8.2)],
        "b": [_make_jump("2026-04-12T11:00:00Z", delta_pp=5.6)],
    }
    call_counter = {"n": 0}
    stub = _stub_jumps_factory(jumps_by_slug)

    async def _counting_stub(*args, **kwargs):
        call_counter["n"] += 1
        return await stub(*args, **kwargs)

    with patch("pfm.terminal.jumps_compare_router.get_jumps", _counting_stub):
        cli = _client_with_router()
        r1 = cli.get("/terminal/jumps/compare?slugs=a,b")
        r2 = cli.get("/terminal/jumps/compare?slugs=a,b")
    assert r1.status_code == 200 and r2.status_code == 200
    # First call → 2 invocations (one per slug); second call → 0.
    assert call_counter["n"] == 2


def test_cache_hit_is_order_invariant() -> None:
    """``?slugs=a,b`` and ``?slugs=b,a`` share a cache entry (sorted key)."""
    jumps_by_slug = {
        "a": [_make_jump("2026-04-12T10:00:00Z", delta_pp=8.2)],
        "b": [_make_jump("2026-04-12T11:00:00Z", delta_pp=5.6)],
    }
    call_counter = {"n": 0}
    stub = _stub_jumps_factory(jumps_by_slug)

    async def _counting_stub(*args, **kwargs):
        call_counter["n"] += 1
        return await stub(*args, **kwargs)

    with patch("pfm.terminal.jumps_compare_router.get_jumps", _counting_stub):
        cli = _client_with_router()
        cli.get("/terminal/jumps/compare?slugs=a,b")
        r2 = cli.get("/terminal/jumps/compare?slugs=b,a")
    body = r2.json()
    # No new fan-out calls — purely from cache.
    assert call_counter["n"] == 2
    # But the response respects the *new* caller order.
    assert body["slugs"] == ["b", "a"]
    assert list(body["jumps_by_slug"].keys()) == ["b", "a"]


def test_empty_slug_param_returns_400() -> None:
    """An empty slug list (commas only) is rejected."""
    cli = _client_with_router()
    r = cli.get("/terminal/jumps/compare?slugs=%20")  # whitespace only
    assert r.status_code in (400, 422)


def test_missing_polymarket_state_returns_503() -> None:
    """If the app has no ``state.poly`` the dependency raises 503."""
    app = FastAPI()
    app.include_router(router)
    # NB: deliberately no app.state.poly
    jumps_by_slug = {"a": [_make_jump("2026-04-12T10:00:00Z", delta_pp=8.2)]}
    with patch("pfm.terminal.jumps_compare_router.get_jumps", _stub_jumps_factory(jumps_by_slug)):
        cli = TestClient(app)
        r = cli.get("/terminal/jumps/compare?slugs=a")
    assert r.status_code == 503


def test_response_envelope_has_required_fields() -> None:
    """Schema sanity: every documented top-level field is present."""
    jumps_by_slug = {
        "a": [_make_jump("2026-04-12T10:00:00Z", delta_pp=8.2)],
        "b": [_make_jump("2026-04-12T11:00:00Z", delta_pp=5.6)],
    }
    with patch("pfm.terminal.jumps_compare_router.get_jumps", _stub_jumps_factory(jumps_by_slug)):
        cli = _client_with_router()
        r = cli.get("/terminal/jumps/compare?slugs=a,b")
    body = r.json()
    for k in ("slugs", "days", "common_days", "jumps_by_slug"):
        assert k in body, f"missing field: {k}"
    for day in body["common_days"]:
        for k in ("date", "jumps", "shared_news_count"):
            assert k in day


def test_duplicate_slugs_deduped_in_input() -> None:
    """Repeating a slug in the input must not duplicate it in the output."""
    jumps_by_slug = {
        "a": [_make_jump("2026-04-12T10:00:00Z", delta_pp=8.2)],
        "b": [_make_jump("2026-04-12T11:00:00Z", delta_pp=5.6)],
    }
    with patch("pfm.terminal.jumps_compare_router.get_jumps", _stub_jumps_factory(jumps_by_slug)):
        cli = _client_with_router()
        r = cli.get("/terminal/jumps/compare?slugs=a,b,a,b")
    body = r.json()
    assert body["slugs"] == ["a", "b"]
    assert set(body["jumps_by_slug"].keys()) == {"a", "b"}
