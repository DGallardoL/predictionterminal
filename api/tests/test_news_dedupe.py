"""Tests for pfm.terminal.news_dedupe.

Run standalone with::

    pytest tests/test_news_dedupe.py -q --noconftest
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Make `pfm` importable when running with --noconftest (skips the
# tests/conftest fixtures that pull in optional heavy deps).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

from pfm.terminal.news_dedupe import (
    NewsItem,
    dedupe_news,
    hamming,
    simhash,
    tokenize,
)

UTC = UTC


def _item(
    title: str,
    *,
    source: str = "gdelt",
    url: str | None = None,
    published_at: datetime | None = None,
    tone: float | None = None,
) -> NewsItem:
    return NewsItem(
        title=title,
        url=url or f"https://example.com/{source}/{abs(hash(title))}",
        source=source,
        published_at=published_at or datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
        tone=tone,
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Fed hikes 25bps!") == ["fed", "hikes", "25bps"]


def test_tokenize_removes_stopwords():
    # "the" and "to" should drop.
    assert tokenize("The Fed cuts rates to zero") == ["fed", "cuts", "rates", "zero"]


def test_tokenize_empty_and_pure_punct():
    assert tokenize("") == []
    assert tokenize("...!?") == []


# ---------------------------------------------------------------------------
# simhash + hamming primitives
# ---------------------------------------------------------------------------


def test_simhash_deterministic():
    assert simhash("Fed hikes rates") == simhash("Fed hikes rates")


def test_simhash_empty_is_zero():
    assert simhash("") == 0
    assert simhash("the a an") == 0  # all stopwords


def test_simhash_rejects_non_byte_aligned_bits():
    with pytest.raises(ValueError):
        simhash("Fed hikes", bits=7)
    with pytest.raises(ValueError):
        simhash("Fed hikes", bits=0)


def test_hamming_zero_for_identical():
    assert hamming(0xDEADBEEF, 0xDEADBEEF) == 0


def test_hamming_counts_bit_differences():
    # 0b1010 vs 0b0101 -> 4 differing bits
    assert hamming(0b1010, 0b0101) == 4
    # Single-bit flip
    assert hamming(0b1000_0000, 0b0000_0000) == 1


def test_hamming_rejects_negative():
    with pytest.raises(ValueError):
        hamming(-1, 0)


def test_simhash_close_for_similar_titles():
    # Stopword removal + shared core tokens should yield small Hamming.
    a = simhash("Fed cuts rates")
    b = simhash("The Fed cuts rates")
    assert hamming(a, b) == 0


def test_simhash_distant_for_unrelated_titles():
    a = simhash("Fed cuts rates by 25 basis points")
    b = simhash("Lakers beat Celtics in overtime thriller")
    # 64-bit SimHash on totally disjoint vocab: expect ~32 bit distance
    # in expectation; require well above the default threshold.
    assert hamming(a, b) > 20


# ---------------------------------------------------------------------------
# dedupe_news
# ---------------------------------------------------------------------------


def test_dedupe_empty_returns_empty():
    assert dedupe_news([]) == []


def test_dedupe_identical_titles_collapses_to_one():
    items = [
        _item("Fed hikes 25bps", source="gdelt"),
        _item("Fed hikes 25bps", source="reddit"),
        _item("Fed hikes 25bps", source="hn"),
    ]
    out = dedupe_news(items)
    assert len(out) == 1
    assert sorted(out[0].sources) == ["gdelt", "hn", "reddit"]
    assert len(out[0].urls) == 3


def test_dedupe_stopword_only_variants_collapse():
    items = [
        _item("Fed cuts rates", source="gdelt"),
        _item("The Fed cuts rates", source="rss"),
    ]
    out = dedupe_news(items, threshold_bits=2)
    assert len(out) == 1
    assert set(out[0].sources) == {"gdelt", "rss"}


def test_dedupe_distinct_titles_kept_separate():
    items = [
        _item("Fed cuts rates by 50 basis points", source="gdelt"),
        _item("Lakers beat Celtics in overtime", source="reddit"),
        _item("Apple unveils new iPhone with foldable display", source="hn"),
    ]
    out = dedupe_news(items, threshold_bits=4)
    assert len(out) == 3


def test_dedupe_threshold_zero_only_exact_signature_matches():
    # These two share most tokens but differ by punctuation/wording
    # enough that their signatures rarely match exactly.
    items = [
        _item("Fed hikes 25 bps", source="a"),
        _item("Federal Reserve raises 25 basis points", source="b"),
    ]
    out = dedupe_news(items, threshold_bits=0)
    assert len(out) == 2


def test_dedupe_earliest_published_at_wins():
    early = datetime(2026, 5, 16, 9, 0, tzinfo=UTC)
    mid = datetime(2026, 5, 16, 10, 0, tzinfo=UTC)
    late = datetime(2026, 5, 16, 11, 0, tzinfo=UTC)
    items = [
        _item("Powell signals dovish pivot", source="late", published_at=late),
        _item("Powell signals dovish pivot", source="early", published_at=early),
        _item("Powell signals dovish pivot", source="mid", published_at=mid),
    ]
    out = dedupe_news(items)
    assert len(out) == 1
    assert out[0].published_at == early
    assert out[0].source == "early"  # winner's primary source field
    # All three feeds are accumulated.
    assert set(out[0].sources) == {"early", "mid", "late"}


def test_dedupe_three_sources_merged():
    items = [
        _item("BTC ETF approved by SEC", source="gdelt", url="https://gdelt/1"),
        _item("BTC ETF approved by SEC", source="reddit", url="https://reddit/1"),
        _item("BTC ETF approved by SEC", source="hn", url="https://hn/1"),
    ]
    out = dedupe_news(items)
    assert len(out) == 1
    assert sorted(out[0].sources) == ["gdelt", "hn", "reddit"]
    assert sorted(out[0].urls) == ["https://gdelt/1", "https://hn/1", "https://reddit/1"]


def test_dedupe_preserves_first_seen_order_across_clusters():
    items = [
        _item("Apple unveils foldable iPhone", source="hn"),
        _item("Fed hikes 25bps", source="gdelt"),
        _item("Fed hikes 25bps", source="reddit"),
        _item("Apple unveils foldable iPhone", source="rss"),
    ]
    out = dedupe_news(items)
    assert len(out) == 2
    assert out[0].title == "Apple unveils foldable iPhone"
    assert out[1].title == "Fed hikes 25bps"


def test_dedupe_does_not_mutate_input():
    items = [
        _item("Fed hikes 25bps", source="gdelt"),
        _item("Fed hikes 25bps", source="reddit"),
    ]
    original_sources = [it.source for it in items]
    original_urls = [it.url for it in items]
    _ = dedupe_news(items)
    assert [it.source for it in items] == original_sources
    assert [it.url for it in items] == original_urls
    # The originals do not get the merged accumulators populated.
    assert items[0].sources == []
    assert items[0].urls == []


def test_dedupe_rejects_negative_threshold():
    with pytest.raises(ValueError):
        dedupe_news([_item("Fed hikes")], threshold_bits=-1)


def test_dedupe_tone_propagated_from_winner():
    early = datetime(2026, 5, 16, 9, 0, tzinfo=UTC)
    late = early + timedelta(hours=2)
    items = [
        _item("Powell hawkish remarks rattle markets", source="late", published_at=late, tone=-0.4),
        _item(
            "Powell hawkish remarks rattle markets", source="early", published_at=early, tone=-0.7
        ),
    ]
    out = dedupe_news(items)
    assert len(out) == 1
    assert out[0].tone == -0.7  # winner = earliest


def test_dedupe_single_item_passes_through():
    items = [_item("Solo headline", source="rss")]
    out = dedupe_news(items)
    assert len(out) == 1
    assert out[0].title == "Solo headline"
    assert out[0].sources == ["rss"]
    assert out[0].urls == [items[0].url]


def test_dedupe_realistic_messy_fixture():
    """Hand-built fixture of 10 messy items spanning 4 logical stories."""

    base = datetime(2026, 5, 16, 9, 0, tzinfo=UTC)
    items = [
        # Story A: Fed rate cut (3 near-duplicates)
        _item("Fed cuts rates by 25 basis points", source="gdelt", published_at=base),
        _item(
            "The Fed cuts rates by 25 basis points",
            source="reddit",
            published_at=base + timedelta(minutes=10),
        ),
        _item(
            "Fed cuts rates 25 basis points", source="hn", published_at=base + timedelta(minutes=15)
        ),
        # Story B: BTC ETF (2 near-duplicates)
        _item(
            "SEC approves spot Bitcoin ETF",
            source="gdelt",
            published_at=base + timedelta(minutes=20),
        ),
        _item(
            "SEC approves spot Bitcoin ETF", source="rss", published_at=base + timedelta(minutes=22)
        ),
        # Story C: Apple foldable (2 near-duplicates)
        _item(
            "Apple unveils foldable iPhone with new display",
            source="hn",
            published_at=base + timedelta(minutes=30),
        ),
        _item(
            "Apple unveils foldable iPhone with new display",
            source="rss",
            published_at=base + timedelta(minutes=35),
        ),
        # Story D: Standalone unrelated headlines (3 singletons)
        _item(
            "Lakers beat Celtics in overtime thriller",
            source="reddit",
            published_at=base + timedelta(minutes=40),
        ),
        _item(
            "Trump campaign raises record sum in April",
            source="gdelt",
            published_at=base + timedelta(minutes=50),
        ),
        _item(
            "OpenAI releases new reasoning model",
            source="hn",
            published_at=base + timedelta(minutes=60),
        ),
    ]
    out = dedupe_news(items, threshold_bits=4)
    # 3 multi-source clusters (Fed / BTC / Apple) + 3 singletons
    # (Lakers / Trump / OpenAI) = 6 deduped items.  Dedupe rate: 10 -> 6
    # = 40 % reduction.
    assert len(out) == 6
    titles = {o.title for o in out}
    # Earliest survivor of Story A keeps the original input title.
    assert "Fed cuts rates by 25 basis points" in titles
    # Find Story A cluster and verify 3 sources merged.
    a = next(o for o in out if o.title.startswith("Fed cuts"))
    assert sorted(a.sources) == ["gdelt", "hn", "reddit"]
    assert a.published_at == base  # earliest wins
    # Find Story B (BTC ETF) and verify 2 sources merged.
    b = next(o for o in out if "Bitcoin" in o.title)
    assert sorted(b.sources) == ["gdelt", "rss"]
    # Singletons each retain just their own source.
    for story in ("Lakers", "Trump", "OpenAI"):
        s = next(o for o in out if story in o.title)
        assert len(s.sources) == 1
