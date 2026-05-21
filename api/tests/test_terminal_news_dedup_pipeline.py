"""End-to-end pipeline tests for cross-source news dedupe (T38 / W11-31).

These tests verify that T20's ``pfm.terminal.news_dedupe.dedupe_news`` SimHash
merger integrates correctly with the existing GDELT + Reddit + HN + RSS
ingestion pipeline.

Status note
-----------
W11-13 (migration of ``pfm.terminal.news`` to use SimHash dedupe instead of
URL-set dedupe) has NOT landed as of 2026-05-16. The endpoint
``GET /terminal/news/{slug}`` still merges Reddit + HN items and dedupes on
``url``.  Until the migration ships, these tests focus on the standalone
``dedupe_news`` integration: we simulate a multi-source fan-out by
constructing :class:`NewsItem` lists drawn from three or four "sources"
(``gdelt``, ``reddit``, ``hn``, ``rss``) and exercise dedupe with the same
mix the real pipeline produces.

Tests cover (see W11-31 spec):

1.  3 sources returning the same headline -> 1 result, 3 source entries
2.  Same content via different sources (gdelt+reddit+hn) -> merged
3.  Distinct titles across sources -> no merge
4.  Empty source -> other sources still work
5.  All sources return same item -> 1 result
6.  Concurrent 3-source fetch is bounded under 5 s (simulated)
7.  Stopword-only difference ("Fed rate cut" vs "The Fed rate cut") -> merged
8.  Source order doesn't affect output identity
9.  Earliest published_at wins after merge
10. Pipeline returns items with ``sources: list[str]`` populated when merging

Run standalone::

    pytest tests/test_terminal_news_dedup_pipeline.py -q --noconftest
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Make ``pfm`` importable when running with ``--noconftest`` (which skips
# the tests/conftest fixtures that pull in optional heavy deps).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

from pfm.terminal.news_dedupe import (
    NewsItem,
    dedupe_news,
)

UTC = UTC
BASE_TS = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers — simulate per-source fetchers as small callables that each
# return their slice of the merged feed. This is the shape the production
# pipeline will adopt once W11-13 lands: each source emits raw
# :class:`NewsItem` instances and the merge layer hands them to
# ``dedupe_news``.
# ---------------------------------------------------------------------------


def _item(
    title: str,
    *,
    source: str,
    url: str | None = None,
    published_at: datetime | None = None,
    tone: float | None = None,
) -> NewsItem:
    """Build a single NewsItem with sensible defaults."""

    return NewsItem(
        title=title,
        url=url or f"https://example.com/{source}/{abs(hash(title))}",
        source=source,
        published_at=published_at or BASE_TS,
        tone=tone,
    )


def _fetch_gdelt(title: str, *, delay_s: float = 0.0) -> list[NewsItem]:
    if delay_s:
        time.sleep(delay_s)
    return [
        _item(title, source="gdelt", url=f"https://gdelt.example/{abs(hash(title))}"),
    ]


def _fetch_reddit(title: str, *, delay_s: float = 0.0) -> list[NewsItem]:
    if delay_s:
        time.sleep(delay_s)
    return [
        _item(title, source="reddit", url=f"https://reddit.example/{abs(hash(title))}"),
    ]


def _fetch_hn(title: str, *, delay_s: float = 0.0) -> list[NewsItem]:
    if delay_s:
        time.sleep(delay_s)
    return [
        _item(title, source="hn", url=f"https://hn.example/{abs(hash(title))}"),
    ]


def _fetch_rss(title: str, *, delay_s: float = 0.0) -> list[NewsItem]:
    if delay_s:
        time.sleep(delay_s)
    return [
        _item(title, source="rss", url=f"https://rss.example/{abs(hash(title))}"),
    ]


def _run_pipeline(
    sources: list[list[NewsItem]],
    *,
    threshold_bits: int = 4,
) -> list[NewsItem]:
    """Mimic the post-W11-13 merge pipeline: concat + dedupe_news."""

    merged: list[NewsItem] = []
    for src in sources:
        merged.extend(src)
    return dedupe_news(merged, threshold_bits=threshold_bits)


# ---------------------------------------------------------------------------
# 1. Three sources returning the SAME headline collapse to a single
# deduped item whose ``sources`` list contains all three feed names.
# ---------------------------------------------------------------------------


def test_pipeline_three_sources_same_headline_collapses_to_one() -> None:
    title = "Fed cuts rates 25bps after Powell speech"
    out = _run_pipeline(
        [
            _fetch_gdelt(title),
            _fetch_reddit(title),
            _fetch_hn(title),
        ]
    )
    assert len(out) == 1
    assert sorted(out[0].sources) == ["gdelt", "hn", "reddit"]
    assert len(out[0].urls) == 3  # one URL per upstream feed


# ---------------------------------------------------------------------------
# 2. Same logical content surfaced via different sources gets merged even
# when the URLs are different (real-world feeds: GDELT links to a wire
# story, Reddit links to a discussion thread, HN links to the original).
# ---------------------------------------------------------------------------


def test_pipeline_same_story_different_urls_across_sources_merges() -> None:
    base = "BTC ETF gets long-awaited SEC approval"
    items = [
        _item(base, source="gdelt", url="https://reuters.com/btc-etf-sec"),
        _item(base, source="reddit", url="https://reddit.com/r/cryptocurrency/btc-etf"),
        _item(base, source="hn", url="https://news.ycombinator.com/item?id=99999"),
    ]
    out = dedupe_news(items)
    assert len(out) == 1
    assert sorted(out[0].sources) == ["gdelt", "hn", "reddit"]
    # All three distinct URLs end up in the merged URL list.
    assert sorted(out[0].urls) == [
        "https://news.ycombinator.com/item?id=99999",
        "https://reddit.com/r/cryptocurrency/btc-etf",
        "https://reuters.com/btc-etf-sec",
    ]


# ---------------------------------------------------------------------------
# 3. Different sources, truly distinct titles -> no merge.
# ---------------------------------------------------------------------------


def test_pipeline_distinct_titles_across_sources_no_merge() -> None:
    out = _run_pipeline(
        [
            _fetch_gdelt("Fed cuts rates by 50 basis points after Powell remarks"),
            _fetch_reddit("Lakers beat Celtics in overtime to clinch playoff series"),
            _fetch_hn("Apple unveils foldable iPhone with new flexible display"),
            _fetch_rss("OpenAI releases reasoning-focused next-gen model"),
        ]
    )
    assert len(out) == 4
    titles = {o.title for o in out}
    assert "Fed cuts rates by 50 basis points after Powell remarks" in titles
    assert "Lakers beat Celtics in overtime to clinch playoff series" in titles
    assert "Apple unveils foldable iPhone with new flexible display" in titles
    assert "OpenAI releases reasoning-focused next-gen model" in titles
    # Each survivor has exactly one source.
    for o in out:
        assert len(o.sources) == 1


# ---------------------------------------------------------------------------
# 4. An empty source (e.g. RSS returned zero hits) does not break the
# rest of the pipeline; the surviving sources still merge their dupes.
# ---------------------------------------------------------------------------


def test_pipeline_empty_source_does_not_break_others() -> None:
    title = "ECB holds rates steady, signals possible cut in September"
    out = _run_pipeline(
        [
            _fetch_gdelt(title),
            [],  # RSS came back empty
            _fetch_reddit(title),
            _fetch_hn(title),
        ]
    )
    assert len(out) == 1
    assert sorted(out[0].sources) == ["gdelt", "hn", "reddit"]


def test_pipeline_all_sources_empty_returns_empty() -> None:
    # Belt-and-braces: every source dry should yield no items, not error.
    out = _run_pipeline([[], [], [], []])
    assert out == []


# ---------------------------------------------------------------------------
# 5. All sources return the same item -> 1 result (a stricter form of #1
# that also covers four-source fan-out, RSS included).
# ---------------------------------------------------------------------------


def test_pipeline_four_sources_all_same_item_returns_one() -> None:
    title = "Trump signs executive order on tariffs"
    out = _run_pipeline(
        [
            _fetch_gdelt(title),
            _fetch_reddit(title),
            _fetch_hn(title),
            _fetch_rss(title),
        ]
    )
    assert len(out) == 1
    assert sorted(out[0].sources) == ["gdelt", "hn", "reddit", "rss"]
    assert len(out[0].urls) == 4


# ---------------------------------------------------------------------------
# 6. Cross-source latency: concurrent fan-out completes under 5 s.
# This guards the SLA contract — pipeline must never serialize fetches.
# ---------------------------------------------------------------------------


def test_pipeline_concurrent_fetch_bounded_under_5s() -> None:
    title = "Senate passes infrastructure bill"
    # Each "source" sleeps 200 ms. Serial would be ~600 ms; parallel should
    # be ~200 ms. The 5 s ceiling is a generous SLA-style guard rail.
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [
            ex.submit(_fetch_gdelt, title, delay_s=0.2),
            ex.submit(_fetch_reddit, title, delay_s=0.2),
            ex.submit(_fetch_hn, title, delay_s=0.2),
        ]
        sources = [f.result() for f in futs]
    fetch_elapsed = time.perf_counter() - start

    out = _run_pipeline(sources)
    total_elapsed = time.perf_counter() - start

    assert fetch_elapsed < 5.0, f"fetch fan-out took {fetch_elapsed:.2f}s; expected <5s"
    assert total_elapsed < 5.0, f"pipeline took {total_elapsed:.2f}s; expected <5s"
    # Sanity: 3-source dupes still collapse.
    assert len(out) == 1
    assert sorted(out[0].sources) == ["gdelt", "hn", "reddit"]
    # And the parallel fan-out actually overlapped (serial would be ~0.6 s).
    assert fetch_elapsed < 0.45, (
        f"fan-out elapsed {fetch_elapsed:.3f}s suggests serialised execution"
    )


# ---------------------------------------------------------------------------
# 7. Stopword-only difference between sources still merges.
# ---------------------------------------------------------------------------


def test_pipeline_stopword_only_diff_across_sources_merges() -> None:
    items = [
        _item("Fed rate cut", source="gdelt"),
        _item("The Fed rate cut", source="reddit"),
        _item("A Fed rate cut", source="hn"),
    ]
    out = dedupe_news(items, threshold_bits=2)
    assert len(out) == 1
    assert sorted(out[0].sources) == ["gdelt", "hn", "reddit"]


# ---------------------------------------------------------------------------
# 8. Source order doesn't affect output identity. The set of clusters
# (and their populated source lists) must be invariant under permutation
# of the per-source fan-out lists.
# ---------------------------------------------------------------------------


def test_pipeline_source_order_does_not_affect_output_identity() -> None:
    title_a = "Powell hints at September rate cut"
    title_b = "Apple unveils new MacBook with M5 chip"
    title_c = "Solana hits all-time high above $300"

    perm_one = [
        _fetch_gdelt(title_a) + _fetch_gdelt(title_b),
        _fetch_reddit(title_a) + _fetch_reddit(title_c),
        _fetch_hn(title_a) + _fetch_hn(title_b),
    ]
    perm_two = [
        _fetch_hn(title_a) + _fetch_hn(title_b),
        _fetch_gdelt(title_a) + _fetch_gdelt(title_b),
        _fetch_reddit(title_a) + _fetch_reddit(title_c),
    ]
    perm_three = [
        _fetch_reddit(title_a) + _fetch_reddit(title_c),
        _fetch_hn(title_a) + _fetch_hn(title_b),
        _fetch_gdelt(title_a) + _fetch_gdelt(title_b),
    ]

    def _summarise(out: list[NewsItem]) -> dict[str, tuple[str, ...]]:
        # Map each cluster's title to its sorted sources tuple.
        return {o.title: tuple(sorted(o.sources)) for o in out}

    s_one = _summarise(_run_pipeline(perm_one))
    s_two = _summarise(_run_pipeline(perm_two))
    s_three = _summarise(_run_pipeline(perm_three))

    # Same set of clusters with same source membership regardless of
    # fan-out ordering.
    assert s_one == s_two == s_three
    # And the semantics we care about hold:
    assert s_one[title_a] == ("gdelt", "hn", "reddit")
    assert s_one[title_b] == ("gdelt", "hn")
    assert s_one[title_c] == ("reddit",)


# ---------------------------------------------------------------------------
# 9. Earliest published_at wins after merge across sources.
# ---------------------------------------------------------------------------


def test_pipeline_earliest_published_at_wins_after_merge() -> None:
    title = "OPEC announces surprise production cut"
    early = BASE_TS
    mid = BASE_TS + timedelta(minutes=12)
    late = BASE_TS + timedelta(hours=1)

    out = dedupe_news(
        [
            # Late wire pickup first.
            _item(title, source="reddit", published_at=late),
            # Earliest from GDELT.
            _item(title, source="gdelt", published_at=early),
            # Mid-pack HN.
            _item(title, source="hn", published_at=mid),
        ]
    )
    assert len(out) == 1
    assert out[0].published_at == early
    # The winner's primary ``source`` field is the earliest source.
    assert out[0].source == "gdelt"
    # All three are still represented in the merged sources list.
    assert sorted(out[0].sources) == ["gdelt", "hn", "reddit"]


# ---------------------------------------------------------------------------
# 10. Returned items have ``sources: list[str]`` populated whenever a
# merge happened, and singletons keep a one-element sources list with
# just their own source name.
# ---------------------------------------------------------------------------


def test_pipeline_sources_list_str_populated_on_merge() -> None:
    items = [
        # Two-source cluster.
        _item("Fed cuts rates 25bps", source="gdelt"),
        _item("Fed cuts rates 25bps", source="reddit"),
        # Singleton.
        _item("Lakers win in overtime", source="hn"),
        # Three-source cluster.
        _item("Apple Q2 earnings beat estimates", source="gdelt"),
        _item("Apple Q2 earnings beat estimates", source="reddit"),
        _item("Apple Q2 earnings beat estimates", source="rss"),
    ]
    out = dedupe_news(items)
    assert len(out) == 3

    by_title = {o.title: o for o in out}
    fed = by_title["Fed cuts rates 25bps"]
    apple = by_title["Apple Q2 earnings beat estimates"]
    lakers = by_title["Lakers win in overtime"]

    # All survivors carry a list[str] of sources.
    for o in out:
        assert isinstance(o.sources, list)
        assert all(isinstance(s, str) for s in o.sources)

    # Multi-source clusters list every contributing source.
    assert sorted(fed.sources) == ["gdelt", "reddit"]
    assert sorted(apple.sources) == ["gdelt", "reddit", "rss"]
    # Singleton retains its own source.
    assert lakers.sources == ["hn"]


# ---------------------------------------------------------------------------
# Bonus: end-to-end "realistic" pipeline shape — 4 sources, mixed cluster
# sizes, stopword variations, mixed timestamps. Sanity-checks the dedupe
# rate and confirms the source-merging behaviour holds at scale.
# ---------------------------------------------------------------------------


def test_pipeline_realistic_four_source_fanout_dedupe_rate() -> None:
    base = BASE_TS
    # 12 raw items across 4 logical stories + 2 singletons.
    items = [
        # Story A (3 sources, stopword variation)
        _item("Fed cuts rates 25bps", source="gdelt", published_at=base),
        _item(
            "The Fed cuts rates 25bps", source="reddit", published_at=base + timedelta(minutes=5)
        ),
        _item("Fed cuts rates 25bps", source="hn", published_at=base + timedelta(minutes=8)),
        # Story B (4 sources, identical)
        _item("BTC ETF approved by SEC", source="gdelt", published_at=base + timedelta(minutes=10)),
        _item(
            "BTC ETF approved by SEC", source="reddit", published_at=base + timedelta(minutes=11)
        ),
        _item("BTC ETF approved by SEC", source="hn", published_at=base + timedelta(minutes=12)),
        _item("BTC ETF approved by SEC", source="rss", published_at=base + timedelta(minutes=13)),
        # Story C (2 sources, identical)
        _item(
            "Apple announces share buyback",
            source="gdelt",
            published_at=base + timedelta(minutes=20),
        ),
        _item(
            "Apple announces share buyback", source="rss", published_at=base + timedelta(minutes=21)
        ),
        # Singletons
        _item(
            "Lakers beat Celtics in overtime",
            source="reddit",
            published_at=base + timedelta(minutes=30),
        ),
        _item(
            "OpenAI releases new reasoning model",
            source="hn",
            published_at=base + timedelta(minutes=40),
        ),
        _item(
            "Trump campaign tops fundraising record",
            source="gdelt",
            published_at=base + timedelta(minutes=50),
        ),
    ]
    out = dedupe_news(items, threshold_bits=4)
    # 3 multi-source clusters + 3 singletons = 6 deduped items.
    assert len(out) == 6
    # Dedupe rate: 12 -> 6 = 50 %.
    dedupe_rate = 1.0 - (len(out) / len(items))
    assert dedupe_rate == pytest.approx(0.5, abs=1e-9)

    by_title = {o.title: o for o in out}
    # Story A: earliest wins, 3 sources merged.
    fed = next(o for o in out if o.title.startswith("Fed cuts"))
    assert sorted(fed.sources) == ["gdelt", "hn", "reddit"]
    assert fed.published_at == base
    # Story B: 4 sources merged.
    btc = by_title["BTC ETF approved by SEC"]
    assert sorted(btc.sources) == ["gdelt", "hn", "reddit", "rss"]
    # Story C: 2 sources merged.
    apple = by_title["Apple announces share buyback"]
    assert sorted(apple.sources) == ["gdelt", "rss"]
    # Singletons.
    for solo in (
        "Lakers beat Celtics in overtime",
        "OpenAI releases new reasoning model",
        "Trump campaign tops fundraising record",
    ):
        assert len(by_title[solo].sources) == 1
