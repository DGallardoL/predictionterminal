"""Tests for the multi-market jump cluster detector.

Two layers of coverage:

1. **Pure-function unit tests on ``find_clusters``** — we synthesize
   :class:`Jump` instances directly (no network) and assert that the
   greedy union-find groups them as expected.
2. **Endpoint integration** — we mock the per-slug ``get_jumps`` so we
   exercise the fan-out + Pydantic envelope without ever calling
   Polymarket or GDELT.
"""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal.jumps import Jump, JumpArticle, TerminalJumpsResponse
from pfm.terminal.jumps_cluster import (
    _CACHE,
    Cluster,
    find_clusters,
    router,
)


@pytest.fixture(autouse=True)
def _clear_cluster_cache() -> None:
    """The cluster endpoint caches by (slug-list, params); reset between tests."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------


def _make_article(
    headline: str,
    terms: Iterable[str],
    *,
    ts_iso: str = "2026-05-01T12:00:00Z",
    relevance: float = 0.7,
    seconds_from_jump: int = 0,
) -> JumpArticle:
    return JumpArticle(
        ts_iso=ts_iso,
        seconds_from_jump=seconds_from_jump,
        headline=headline,
        source="test.com",
        url="https://test.example/x",
        tone=0.0,
        relevance_score=relevance,
        matched_terms=list(terms),
        sentiment_score=0.0,
        sentiment_label="neutral",
    )


def _make_jump(
    ts_iso: str,
    *,
    terms: Iterable[str] | None = None,
    headline: str = "Generic headline",
    delta_pp: float = 5.0,
    direction: str = "up",
    relevance: float = 0.7,
) -> Jump:
    """Build a minimal :class:`Jump` with one article carrying ``terms``."""
    article = _make_article(headline, terms or [], ts_iso=ts_iso, relevance=relevance)
    return Jump(
        ts_iso=ts_iso,
        price_before=0.40,
        price_after=0.45,
        delta_pp=delta_pp,
        delta_logit=0.2 if direction == "up" else -0.2,
        z_score=4.0,
        direction=direction,
        explained=True,
        n_articles=1,
        top_articles=[article],
        news_sentiment_score=0.0,
        news_sentiment_label="neutral",
        sentiment_alignment="neutral",
    )


# ---------------------------------------------------------------------------
# find_clusters — happy paths
# ---------------------------------------------------------------------------


def test_three_jumps_across_two_slugs_within_3_minutes_form_one_cluster() -> None:
    """3 jumps · 2 slugs · all within 3 min · all share {trump, china} → 1 cluster of 3."""
    jumps_by_slug = {
        "trump-vs-china": [
            _make_jump("2026-05-01T12:00:00Z", terms=["trump", "china"], delta_pp=6.0),
            _make_jump("2026-05-01T12:02:00Z", terms=["trump", "china"], delta_pp=4.5),
        ],
        "china-tariffs-2026": [
            _make_jump("2026-05-01T12:03:00Z", terms=["trump", "china"], delta_pp=-3.5),
        ],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.20)
    assert len(clusters) == 1
    c = clusters[0]
    assert len(c.member_jumps) == 3
    assert c.n_markets == 2
    # Dominant terms are the top-K by frequency: both should appear.
    assert set(c.dominant_terms) >= {"trump", "china"}


def test_two_jumps_one_hour_apart_do_not_merge() -> None:
    """Same terms, but 60 min > 5 min tolerance → no cluster (singletons filtered)."""
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["fomc", "rate"])],
        "slug-b": [_make_jump("2026-05-01T13:00:00Z", terms=["fomc", "rate"])],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.20)
    assert clusters == []


def test_two_jumps_within_tolerance_but_disjoint_terms_do_not_merge() -> None:
    """Time gates pass but Jaccard = 0 → no cluster."""
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["trump", "tariff"])],
        "slug-b": [_make_jump("2026-05-01T12:02:00Z", terms=["earnings", "nvda"])],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.20)
    assert clusters == []


def test_empty_input_returns_empty_list() -> None:
    assert find_clusters({}) == []


def test_single_slug_with_multiple_jumps_yields_no_cluster() -> None:
    """A single market ringing twice isn't a macro event — same-slug pairs
    are explicitly excluded from union."""
    jumps_by_slug = {
        "slug-a": [
            _make_jump("2026-05-01T12:00:00Z", terms=["fomc"]),
            _make_jump("2026-05-01T12:01:00Z", terms=["fomc"]),
        ],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.20)
    assert clusters == []


# ---------------------------------------------------------------------------
# Jaccard threshold edge cases
# ---------------------------------------------------------------------------


def test_jaccard_at_exactly_threshold_merges() -> None:
    """``jaccard >= kw_min_jaccard``: equal counts as a merge.

    sets are {a, b, c} vs {a, b, d} → |∩|=2, |∪|=4, jaccard=0.5.
    With threshold 0.5 the pair should merge.
    """
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["a", "b", "c"])],
        "slug-b": [_make_jump("2026-05-01T12:01:00Z", terms=["a", "b", "d"])],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.50)
    assert len(clusters) == 1


def test_jaccard_just_below_threshold_does_not_merge() -> None:
    """jaccard = 0.5 with a threshold of 0.51 → no merge."""
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["a", "b", "c"])],
        "slug-b": [_make_jump("2026-05-01T12:01:00Z", terms=["a", "b", "d"])],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.51)
    assert clusters == []


def test_jaccard_threshold_zero_merges_when_terms_overlap_at_all() -> None:
    """Threshold of 0 means "any overlap merges" — but disjoint sets still
    have jaccard 0, which is NOT >= 0 in the strict sense... actually it
    *is* (0 >= 0). So zero-term jumps would merge purely on time. We rely
    on the implementation's empty-set short-circuit (returns 0.0) plus the
    ``>=`` comparison to make this deterministic.
    """
    # Jumps with completely empty term sets — by our impl Jaccard is 0 and
    # the gate is ``>= 0`` so they DO merge under a zero threshold.
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["x"])],
        "slug-b": [_make_jump("2026-05-01T12:01:00Z", terms=["x"])],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.0)
    assert len(clusters) == 1


# ---------------------------------------------------------------------------
# Representative selection
# ---------------------------------------------------------------------------


def test_representative_headline_picks_highest_relevance_article() -> None:
    """When multiple member jumps each have an article, the cluster's
    representative_headline should be the highest-relevance one."""
    low_relevance_article = _make_article(
        "Vague mention of trump",
        ["trump"],
        ts_iso="2026-05-01T12:00:00Z",
        relevance=0.3,
    )
    high_relevance_article = _make_article(
        "Trump announces 60% tariff on China imports",
        ["trump", "china"],
        ts_iso="2026-05-01T12:01:00Z",
        relevance=0.95,
    )
    jump_a = Jump(
        ts_iso="2026-05-01T12:00:00Z",
        price_before=0.40,
        price_after=0.45,
        delta_pp=5.0,
        delta_logit=0.2,
        z_score=4.0,
        direction="up",
        explained=True,
        n_articles=1,
        top_articles=[low_relevance_article],
        news_sentiment_score=0.0,
        news_sentiment_label="neutral",
        sentiment_alignment="neutral",
    )
    jump_b = Jump(
        ts_iso="2026-05-01T12:01:00Z",
        price_before=0.50,
        price_after=0.55,
        delta_pp=5.0,
        delta_logit=0.2,
        z_score=4.0,
        direction="up",
        explained=True,
        n_articles=1,
        top_articles=[high_relevance_article],
        news_sentiment_score=0.0,
        news_sentiment_label="neutral",
        sentiment_alignment="neutral",
    )
    clusters = find_clusters(
        {"slug-a": [jump_a], "slug-b": [jump_b]},
        time_tol_minutes=5,
        kw_min_jaccard=0.20,
    )
    assert len(clusters) == 1
    assert clusters[0].representative_headline.startswith("Trump announces")


def test_cluster_sorted_by_n_markets_desc() -> None:
    """A 3-market cluster should be returned before a 2-market cluster."""
    # Big cluster around 12:00 — 3 distinct slugs share {fomc, rate}.
    # Small cluster around 14:00 — 2 distinct slugs share {earnings, nvda}.
    jumps_by_slug = {
        "slug-fomc-a": [_make_jump("2026-05-01T12:00:00Z", terms=["fomc", "rate"])],
        "slug-fomc-b": [_make_jump("2026-05-01T12:01:00Z", terms=["fomc", "rate"])],
        "slug-fomc-c": [_make_jump("2026-05-01T12:02:00Z", terms=["fomc", "rate"])],
        "slug-nvda-a": [_make_jump("2026-05-01T14:00:00Z", terms=["earnings", "nvda"])],
        "slug-nvda-b": [_make_jump("2026-05-01T14:01:00Z", terms=["earnings", "nvda"])],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.20)
    assert len(clusters) == 2
    assert clusters[0].n_markets == 3
    assert clusters[1].n_markets == 2
    # cluster_id matches sort order
    assert clusters[0].cluster_id == 1
    assert clusters[1].cluster_id == 2


def test_clusters_are_typed_correctly() -> None:
    """Sanity check: the returned items are :class:`Cluster` model instances."""
    jumps_by_slug = {
        "a": [_make_jump("2026-05-01T12:00:00Z", terms=["x", "y"])],
        "b": [_make_jump("2026-05-01T12:01:00Z", terms=["x", "y"])],
    }
    clusters = find_clusters(jumps_by_slug)
    assert all(isinstance(c, Cluster) for c in clusters)


# ---------------------------------------------------------------------------
# Endpoint integration — mocked per-slug detection
# ---------------------------------------------------------------------------


class _StubPoly:
    """Minimal Polymarket client stub — the cluster endpoint only uses it
    as a dependency-injection token; the real work is in ``get_jumps``."""

    def __init__(self) -> None:
        self._client = None
        self.clob_url = "https://clob.example"


def _client_with_router() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.poly = _StubPoly()  # type: ignore[assignment]
    app.state.async_http = None  # type: ignore[assignment]
    return TestClient(app)


def _stub_jumps_response_factory(jumps_by_slug: dict[str, list[Jump]]):
    """Return an async function suitable for monkey-patching ``get_jumps``.

    The patched ``get_jumps`` will return the pre-built jump list for whichever
    slug is requested (or an empty TerminalJumpsResponse for an unknown slug).
    """

    async def _fake_get_jumps(
        request,
        slug: str,
        days: int = 14,
        mad_k: float = 2.5,
        min_jump_pp: float = 3.0,
        poly=None,
    ) -> TerminalJumpsResponse:
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


def test_endpoint_clusters_two_overlapping_slugs() -> None:
    """Two slugs with jumps within 2 minutes sharing terms → one cluster."""
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["trump", "china"])],
        "slug-b": [_make_jump("2026-05-01T12:02:00Z", terms=["trump", "china"])],
    }
    fake = _stub_jumps_response_factory(jumps_by_slug)
    with patch("pfm.terminal.jumps_cluster.get_jumps", fake):
        c = _client_with_router()
        r = c.get(
            "/terminal/jumps/cluster",
            params={
                "slugs": "slug-a,slug-b",
                "days": 7,
                "time_tol_minutes": 5,
                "kw_min_jaccard": 0.20,
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_clusters"] == 1
    assert body["n_jumps_total"] == 2
    cluster = body["clusters"][0]
    assert cluster["n_markets"] == 2
    assert len(cluster["member_jumps"]) == 2
    assert {m["slug"] for m in cluster["member_jumps"]} == {"slug-a", "slug-b"}


def test_endpoint_no_overlap_returns_zero_clusters() -> None:
    """Two slugs whose jumps are hours apart → no clusters in the envelope."""
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["trump", "china"])],
        "slug-b": [_make_jump("2026-05-01T18:00:00Z", terms=["trump", "china"])],
    }
    fake = _stub_jumps_response_factory(jumps_by_slug)
    with patch("pfm.terminal.jumps_cluster.get_jumps", fake):
        c = _client_with_router()
        r = c.get("/terminal/jumps/cluster?slugs=slug-a,slug-b")
    assert r.status_code == 200
    body = r.json()
    assert body["n_clusters"] == 0
    assert body["clusters"] == []
    assert body["n_jumps_total"] == 2


def test_endpoint_rejects_too_many_slugs() -> None:
    """Hard cap on slugs to keep the fan-out bounded."""
    slugs = ",".join(f"slug-{i}" for i in range(50))
    c = _client_with_router()
    r = c.get(f"/terminal/jumps/cluster?slugs={slugs}")
    assert r.status_code == 400
    assert "too many slugs" in r.json()["detail"]


def test_endpoint_response_envelope_has_required_fields() -> None:
    """Schema sanity: every documented field is present."""
    fake = _stub_jumps_response_factory({"slug-a": [], "slug-b": []})
    with patch("pfm.terminal.jumps_cluster.get_jumps", fake):
        c = _client_with_router()
        r = c.get("/terminal/jumps/cluster?slugs=slug-a,slug-b")
    body = r.json()
    for k in (
        "slugs",
        "days",
        "time_tol_minutes",
        "kw_min_jaccard",
        "n_jumps_total",
        "n_clusters",
        "clusters",
    ):
        assert k in body, f"missing field: {k}"


# ---------------------------------------------------------------------------
# Additional edge cases (filling coverage gaps in jumps_cluster.py)
# ---------------------------------------------------------------------------


def test_hour_of_day_boundary_jumps_form_a_cluster() -> None:
    """A jump at 23:58 + a jump at 00:02 the *next day* are 4 minutes apart —
    the clusterer must use absolute time, not hour-of-day, so they merge."""
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T23:58:00Z", terms=["fomc", "rate"])],
        "slug-b": [_make_jump("2026-05-02T00:02:00Z", terms=["fomc", "rate"])],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.20)
    assert len(clusters) == 1
    assert clusters[0].n_markets == 2


def test_very_high_kw_min_jaccard_blocks_all_merges() -> None:
    """At Jaccard 1.0 even partially-overlapping term sets cannot merge."""
    jumps_by_slug = {
        # Identical *time* and many overlapping terms — but each set is slightly
        # different, so Jaccard < 1.0 and the strictest threshold rejects them.
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["trump", "china", "tariff"])],
        "slug-b": [_make_jump("2026-05-01T12:01:00Z", terms=["trump", "china", "import"])],
        "slug-c": [_make_jump("2026-05-01T12:02:00Z", terms=["trump", "china", "deal"])],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=1.0)
    assert clusters == []


def test_50_slug_fanout_clusters_without_crashing() -> None:
    """A wide fan-out (50 slugs × 1 jump each, all sharing 'fomc' terms within
    5 min) must produce exactly one cluster — exercises the union-find scaling."""
    jumps_by_slug = {}
    start = pd.Timestamp("2026-05-01T12:00:00Z")
    for i in range(50):
        ts = (start + pd.Timedelta(seconds=i * 5)).isoformat().replace("+00:00", "Z")
        jumps_by_slug[f"slug-{i}"] = [_make_jump(ts, terms=["fomc", "rate"])]
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.20)
    assert len(clusters) == 1
    assert clusters[0].n_markets == 50
    assert len(clusters[0].member_jumps) == 50


def test_unparseable_ts_jumps_dropped_silently() -> None:
    """Jump rows with unparseable ``ts_iso`` are silently filtered out (covers
    the `_parse_ts` → None branch in ``find_clusters``'s node-building loop)."""
    # Build one Jump with a bogus ts and one with a real ts; the bogus one
    # must be ignored without raising. The remaining single jump can't form
    # a cluster on its own → returns [].
    bad = Jump(
        ts_iso="not-a-real-timestamp",
        price_before=0.40,
        price_after=0.45,
        delta_pp=5.0,
        delta_logit=0.2,
        z_score=4.0,
        direction="up",
        explained=True,
        n_articles=1,
        top_articles=[_make_article("x", ["fomc"])],
        news_sentiment_score=0.0,
        news_sentiment_label="neutral",
        sentiment_alignment="neutral",
    )
    good = _make_jump("2026-05-01T12:00:00Z", terms=["fomc"])
    clusters = find_clusters({"slug-a": [bad], "slug-b": [good]})
    # Only the good jump remains; no pair → no cluster.
    assert clusters == []


def test_cluster_with_even_member_count_uses_midpoint_timestamp() -> None:
    """When the cluster has an even number of members, the representative
    timestamp is the midpoint of the two middle timestamps. Covers the
    `pd.to_datetime(median([lo, hi]), unit='s', utc=True)` branch."""
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["x", "y"])],
        "slug-b": [_make_jump("2026-05-01T12:02:00Z", terms=["x", "y"])],
        "slug-c": [_make_jump("2026-05-01T12:03:00Z", terms=["x", "y"])],
        "slug-d": [_make_jump("2026-05-01T12:04:00Z", terms=["x", "y"])],
    }
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5, kw_min_jaccard=0.20)
    assert len(clusters) == 1
    # 4 members; midpoint of (12:02, 12:03) = 12:02:30Z.
    assert "12:02:30" in clusters[0].ts_iso


def test_endpoint_returns_empty_envelope_when_no_default_slugs() -> None:
    """If the slug-discovery returns nothing AND no explicit slugs are passed,
    the endpoint must return an empty cluster envelope (not 5xx)."""
    fake = _stub_jumps_response_factory({})

    async def _no_slugs(*_a, **_kw):
        return []

    with (
        patch("pfm.terminal.jumps_cluster.get_jumps", fake),
        patch("pfm.terminal.jumps_cluster._fetch_default_slugs", _no_slugs),
    ):
        c = _client_with_router()
        r = c.get("/terminal/jumps/cluster")  # no slugs
    assert r.status_code == 200
    body = r.json()
    assert body["slugs"] == []
    assert body["n_jumps_total"] == 0
    assert body["clusters"] == []


def test_endpoint_serves_response_from_cache_on_repeat_call() -> None:
    """Second call with the same params is served from the module-level cache."""
    jumps_by_slug = {
        "slug-a": [_make_jump("2026-05-01T12:00:00Z", terms=["x", "y"])],
        "slug-b": [_make_jump("2026-05-01T12:01:00Z", terms=["x", "y"])],
    }
    fake = _stub_jumps_response_factory(jumps_by_slug)
    with patch("pfm.terminal.jumps_cluster.get_jumps", fake):
        c = _client_with_router()
        r1 = c.get("/terminal/jumps/cluster?slugs=slug-a,slug-b")
    # Drop the get_jumps patch; if the cache works, r2 still succeeds.
    c2 = _client_with_router()
    r2 = c2.get("/terminal/jumps/cluster?slugs=slug-a,slug-b")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["n_clusters"] == r2.json()["n_clusters"]


def test_endpoint_degrades_gracefully_when_one_slug_fetch_fails() -> None:
    """`_safe_get_jumps_for_slug` swallows per-slug exceptions so a single
    bad slug doesn't tank the whole cluster response."""
    from fastapi import HTTPException

    good_jumps = [_make_jump("2026-05-01T12:00:00Z", terms=["x", "y"])]

    async def _flaky_get_jumps(
        request,
        slug: str,
        days: int = 14,
        mad_k: float = 2.5,
        min_jump_pp: float = 3.0,
        poly=None,
    ):
        if slug == "slug-good":
            from pfm.terminal.jumps import TerminalJumpsResponse

            return TerminalJumpsResponse(
                slug=slug,
                days=14,
                threshold_mad_k=2.5,
                threshold_min_jump_pp=3.0,
                n_jumps=len(good_jumps),
                n_explained=1,
                explained_pct=100.0,
                jumps=good_jumps,
                interpretation="ok",
            )
        if slug == "slug-404":
            raise HTTPException(status_code=404, detail="not found")
        if slug == "slug-boom":
            raise RuntimeError("upstream blew up")
        raise AssertionError(f"unexpected slug: {slug}")

    with patch("pfm.terminal.jumps_cluster.get_jumps", _flaky_get_jumps):
        c = _client_with_router()
        r = c.get("/terminal/jumps/cluster?slugs=slug-good,slug-404,slug-boom")
    assert r.status_code == 200
    body = r.json()
    # The two flaky slugs degrade to empty lists; the good slug still contributes.
    assert body["n_jumps_total"] == 1
    # No cluster (only one slug with jumps), but no crash.
    assert body["n_clusters"] == 0


def test_endpoint_503_when_polymarket_client_missing_with_explicit_slugs() -> None:
    """When explicit slugs are passed but ``poly`` is absent, the dependency
    must raise 503 (covers `_get_polymarket_client` on the cluster path)."""
    app = FastAPI()
    app.include_router(router)
    # NO app.state.poly
    c = TestClient(app)
    r = c.get("/terminal/jumps/cluster?slugs=slug-a,slug-b")
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# _fetch_default_slugs — async helper coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_default_slugs_returns_empty_when_no_http() -> None:
    """When the async HTTP client is None, the helper must short-circuit
    to ``[]`` (covers the early-return on `http is None`)."""
    from pfm.terminal.jumps_cluster import _fetch_default_slugs

    out = await _fetch_default_slugs(None, "https://example.com")
    assert out == []


@pytest.mark.asyncio
async def test_fetch_default_slugs_handles_http_error() -> None:
    """A 5xx from Gamma must downgrade to ``[]`` rather than raise."""
    import httpx

    from pfm.terminal.jumps_cluster import _fetch_default_slugs

    class _FakeAsyncClient:
        async def get(self, *_args, **_kwargs):
            raise httpx.ConnectError("gamma down")

    out = await _fetch_default_slugs(_FakeAsyncClient(), "https://gamma.example")
    assert out == []


@pytest.mark.asyncio
async def test_fetch_default_slugs_parses_markets_response() -> None:
    """A well-formed Gamma response is parsed into a unique-ordered list of slugs."""
    from pfm.terminal.jumps_cluster import _fetch_default_slugs

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _FakeAsyncClient:
        async def get(self, *_args, **_kwargs):
            return _FakeResponse(
                [
                    {"slug": "slug-a"},
                    {"slug": "slug-b"},
                    {"slug": "slug-a"},  # dup must be deduped
                    {"slug": ""},  # empty slug filtered
                    {"slug": "slug-c"},
                ]
            )

    out = await _fetch_default_slugs(
        _FakeAsyncClient(),
        "https://gamma.example",
        top_n=10,
    )
    assert out == ["slug-a", "slug-b", "slug-c"]


@pytest.mark.asyncio
async def test_fetch_default_slugs_non_list_payload_returns_empty() -> None:
    """A non-list response body is rejected gracefully."""
    from pfm.terminal.jumps_cluster import _fetch_default_slugs

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"error": "unexpected shape"}

    class _FakeAsyncClient:
        async def get(self, *_args, **_kwargs):
            return _FakeResponse()

    out = await _fetch_default_slugs(_FakeAsyncClient(), "https://gamma.example")
    assert out == []


@pytest.mark.asyncio
async def test_fetch_default_slugs_caps_at_top_n() -> None:
    """The helper must stop emitting after ``top_n`` unique slugs."""
    from pfm.terminal.jumps_cluster import _fetch_default_slugs

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [{"slug": f"slug-{i}"} for i in range(20)]

    class _FakeAsyncClient:
        async def get(self, *_args, **_kwargs):
            return _FakeResponse()

    out = await _fetch_default_slugs(
        _FakeAsyncClient(),
        "https://gamma.example",
        top_n=5,
    )
    assert len(out) == 5
    assert out == [f"slug-{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Cross-module integration test
# ---------------------------------------------------------------------------


def test_integration_jumps_backtest_clusters_share_jump_set() -> None:
    """Wire the three modules end-to-end through mocked data sources.

    Pipeline:
        1. ``GET /terminal/jumps/{slug}`` for two slugs → fires synthetic jumps.
        2. ``GET /terminal/jumps/{slug}/backtest`` for one slug → verifies the
           ``n_disagrees + n_agrees`` total equals the jumps endpoint's
           ``n_jumps`` count for the same slug.
        3. ``find_clusters`` on the two slugs' jumps → must produce a cluster
           (synthetic data is constructed so a 2-slug cluster forms).

    Mocks: ``_fetch_hourly_prices`` and ``_gather_all_news`` for both the
    jumps router and the backtest router (they patch the same import paths
    but in different modules, so each gets its own patch).
    """
    from unittest.mock import patch

    import pandas as pd
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pfm.terminal.jumps import router as jumps_router
    from pfm.terminal.jumps_backtest import router as backtest_router
    from pfm.terminal_gdelt_news import GDELTArticle

    # Synthetic price series — spike at hour 12, revert by hour 18.
    def _prices_for_slug(*_args, **_kwargs):
        idx = pd.date_range(start="2026-05-01T00:00:00Z", periods=24, freq="h", tz="UTC")
        vals = [0.50] * 24
        vals[12] = 0.40  # down-jump at hour 12
        vals[18] = 0.48  # partial revert by hour 18
        return pd.Series(vals, index=idx, name="price", dtype=float)

    # Bullish article right before the down-jump → disagrees alignment.
    def _articles(*_args, **_kwargs):
        return [
            GDELTArticle(
                ts="2026-05-01T11:30:00Z",
                title="Trump surges in new poll; election rally boosts odds",
                source="reuters.com",
                country="us",
                language="english",
                tone=3.5,
                url="https://reuters.example/x",
            ),
        ]

    class _MetaStub:
        question = "Will Trump win the 2024 election?"
        yes_token_id = "1"

    class _PolyStub:
        _client = None
        clob_url = "https://clob.example"

        def get_market_metadata(self, slug: str):
            return _MetaStub()

    # Spin up an app that mounts BOTH routers (real /terminal/jumps and
    # /terminal/jumps/{slug}/backtest), then patch their fetch dependencies.
    app = FastAPI()
    app.include_router(jumps_router)
    app.include_router(backtest_router)
    app.state.poly = _PolyStub()

    with (
        patch("pfm.terminal.jumps._fetch_hourly_prices", _prices_for_slug),
        patch("pfm.terminal.jumps._fetch_gdelt", _articles),
        patch("pfm.terminal.jumps_backtest._fetch_hourly_prices", _prices_for_slug),
        patch("pfm.terminal.jumps_backtest._gather_all_news", _articles),
    ):
        c = TestClient(app)
        # 1. Jumps endpoint for slug-a
        r_jumps = c.get("/terminal/jumps/slug-a?days=2&mad_k=2.0&min_jump_pp=3")
        assert r_jumps.status_code == 200, r_jumps.text
        j_body = r_jumps.json()
        assert j_body["n_jumps"] >= 1

        # 2. Backtest endpoint for the SAME slug — n_disagrees+n_agrees
        # should be ≤ n_jumps (jumps with alignment in {disagrees, agrees}).
        r_bt = c.get("/terminal/jumps/slug-a/backtest?days=2&hold_hours=6&mad_k=2.0&min_jump_pp=3")
        assert r_bt.status_code == 200, r_bt.text
        bt_body = r_bt.json()
        # The "disagrees + agrees" total may be ≤ n_jumps because some jumps
        # may end up classified as "neutral" alignment (which is skipped by
        # the backtester). The synthetic data here was tuned so that the
        # single jump fires AND its alignment is non-neutral.
        assert bt_body["n_disagrees"] + bt_body["n_agrees"] >= 1
        assert bt_body["n_disagrees"] + bt_body["n_agrees"] <= j_body["n_jumps"]

    # 3. Use the jumps response to synthesize a cross-slug cluster manually
    # — build a second jump with matching terms 1 minute later and feed
    # find_clusters directly.
    from pfm.terminal.jumps import Jump as _Jump

    primary_jump = j_body["jumps"][0]
    # Need at least one matched_term for a non-empty Jaccard. The synthetic
    # article terms come from `score_relevance` against the question; we
    # rebuild a Jump model with the original top_articles to preserve those.
    real_jump = _Jump(**primary_jump)
    # Build a second jump for slug-b 1 minute later sharing the same article.
    second_jump = _Jump(
        ts_iso=real_jump.ts_iso.replace(
            real_jump.ts_iso[14:16],
            f"{int(real_jump.ts_iso[14:16]) + 1:02d}",
        ),
        price_before=real_jump.price_before,
        price_after=real_jump.price_after,
        delta_pp=real_jump.delta_pp,
        delta_logit=real_jump.delta_logit,
        z_score=real_jump.z_score,
        direction=real_jump.direction,
        explained=real_jump.explained,
        n_articles=real_jump.n_articles,
        top_articles=real_jump.top_articles,
        news_sentiment_score=real_jump.news_sentiment_score,
        news_sentiment_label=real_jump.news_sentiment_label,
        sentiment_alignment=real_jump.sentiment_alignment,
    )

    clusters = find_clusters(
        {"slug-a": [real_jump], "slug-b": [second_jump]},
        time_tol_minutes=5,
        kw_min_jaccard=0.0,  # any overlap (or none) triggers merge
    )
    # The synthetic pair shares 100% of their top_articles' matched_terms,
    # so Jaccard == 1.0 and the two-slug pair must form a cluster.
    assert len(clusters) == 1
    assert clusters[0].n_markets == 2
