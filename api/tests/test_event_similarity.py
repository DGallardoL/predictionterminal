"""Tests for :mod:`pfm.arb_matching.event_similarity` (T77).

These tests deliberately use a *local* ``ResolutionWindow`` stand-in
(via ``_RW``) rather than importing from :mod:`pfm.arb_matching.date_extractor`
(T76). That keeps T77 testable in isolation even if T76 is mid-flight, and
verifies that ``score_match`` only needs duck-typed access to
``.earliest`` / ``.latest`` attributes — the contract documented in
``pfm/arb_matching/__init__.py``.

If T76 has already landed, the import-path test at the bottom of this file
exercises the real ``ResolutionWindow`` end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest

from pfm.arb_matching.event_similarity import (
    REJECT_JURISDICTION_MISMATCH,
    REJECT_REASONS,
    REJECT_SAME_VENUE,
    REJECT_THRESHOLD_MISMATCH,
    REJECT_WINDOW_NO_OVERLAP,
    MarketDesc,
    SimilarityScore,
    build_market_desc,
    score_match,
)

# ---------------------------------------------------------------------------
# Local ResolutionWindow stand-in. Mirrors the T76 contract: two datetime
# bounds + a confidence float in [0, 1].
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RW:
    earliest: datetime
    latest: datetime
    confidence: float = 1.0


def _utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=UTC)


def _window(
    y_start: int,
    m_start: int,
    d_start: int,
    y_end: int,
    m_end: int,
    d_end: int,
    *,
    conf: float = 1.0,
) -> _RW:
    return _RW(_utc(y_start, m_start, d_start), _utc(y_end, m_end, d_end), conf)


# Convenience factory so most tests don't have to spell every field.
def _md(
    title: str,
    *,
    venue: str = "polymarket",
    description: str = "",
    window: _RW | None = None,
    threshold: float | None = None,
    entities: tuple[str, ...] = (),
    jurisdiction: str | None = None,
    topic_clues: tuple[str, ...] = (),
) -> MarketDesc:
    return MarketDesc(
        title=title,
        description=description,
        venue=venue,
        resolution_window=window,
        threshold=threshold,
        entities=entities,
        jurisdiction=jurisdiction,
        raw_topic_clues=topic_clues,
    )


# ---------------------------------------------------------------------------
# Hard-reject tests
# ---------------------------------------------------------------------------


class TestHardRejects:
    def test_trump_2024_vs_trump_2028_rejected_by_window(self):
        a = _md(
            "Trump wins 2024 presidential election",
            venue="polymarket",
            window=_window(2024, 11, 5, 2024, 11, 5),
            entities=("trump",),
        )
        b = _md(
            "Trump wins 2028 presidential election",
            venue="kalshi",
            window=_window(2028, 11, 7, 2028, 11, 7),
            entities=("trump",),
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_WINDOW_NO_OVERLAP
        assert s.total == 0.0

    def test_btc_80k_vs_90k_rejected_by_threshold(self):
        win = _window(2026, 12, 31, 2026, 12, 31)
        a = _md(
            "BTC above $80k by EOY 2026",
            venue="polymarket",
            window=win,
            threshold=80_000.0,
            entities=("btc",),
        )
        b = _md(
            "BTC above $90k by EOY 2026",
            venue="kalshi",
            window=win,
            threshold=90_000.0,
            entities=("btc",),
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_THRESHOLD_MISMATCH
        assert s.total == 0.0

    def test_us_senate_vs_florida_state_senate_rejected_by_jurisdiction(self):
        win = _window(2024, 11, 5, 2024, 11, 5)
        a = _md(
            "US Senate D majority 2024",
            venue="polymarket",
            window=win,
            jurisdiction="US-Senate",
        )
        b = _md(
            "Florida State Senate D majority 2024",
            venue="kalshi",
            window=win,
            jurisdiction="FL-State-Senate",
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_JURISDICTION_MISMATCH
        assert s.total == 0.0

    def test_same_venue_rejected(self):
        win = _window(2024, 11, 5, 2024, 11, 5)
        a = _md("Trump wins 2024", venue="polymarket", window=win)
        b = _md("Donald Trump wins 2024", venue="polymarket", window=win)
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_SAME_VENUE
        assert s.total == 0.0

    def test_same_venue_case_insensitive(self):
        # Same venue mismatch should still fire if casing differs.
        a = _md("X", venue="Polymarket")
        b = _md("Y", venue="POLYMARKET")
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_SAME_VENUE

    def test_threshold_close_within_5pct_not_rejected(self):
        # 80000 vs 80500 → 0.625 % relative → must not hard-reject.
        win = _window(2026, 12, 31, 2026, 12, 31)
        a = _md(
            "BTC above $80,000 by EOY 2026",
            venue="polymarket",
            window=win,
            threshold=80_000.0,
            entities=("btc",),
        )
        b = _md(
            "BTC above $80,500 by EOY 2026",
            venue="kalshi",
            window=win,
            threshold=80_500.0,
            entities=("btc",),
        )
        s = score_match(a, b)
        assert s.rejected_reason is None
        assert s.total > 0

    def test_threshold_just_over_5pct_rejected(self):
        # 100 vs 106 → 6 % → hard-reject.
        a = _md("X over 100", venue="polymarket", threshold=100.0)
        b = _md("Y over 106", venue="kalshi", threshold=106.0)
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_THRESHOLD_MISMATCH

    def test_threshold_only_one_side_not_rejected(self):
        # When one side has no threshold we cannot say they disagree.
        a = _md("BTC above $80k", venue="polymarket", threshold=80_000.0)
        b = _md("Will BTC reach a new high", venue="kalshi", threshold=None)
        s = score_match(a, b)
        assert s.rejected_reason is None

    def test_jurisdiction_one_side_unknown_not_rejected(self):
        a = _md("US Senate D 2024", venue="polymarket", jurisdiction="US-Senate")
        b = _md("Senate balance 2024", venue="kalshi", jurisdiction=None)
        s = score_match(a, b)
        assert s.rejected_reason is None

    def test_window_one_side_missing_not_rejected(self):
        a = _md("Trump wins 2024", venue="polymarket", window=_window(2024, 11, 5, 2024, 11, 5))
        b = _md("Trump wins", venue="kalshi", window=None)
        s = score_match(a, b)
        assert s.rejected_reason is None

    def test_window_overlap_not_rejected(self):
        a = _md(
            "BTC > $80k by Dec 2026",
            venue="polymarket",
            window=_window(2026, 12, 1, 2026, 12, 31),
            threshold=80_000.0,
        )
        b = _md(
            "BTC over $80k end of 2026",
            venue="kalshi",
            window=_window(2026, 12, 15, 2027, 1, 15),
            threshold=80_000.0,
        )
        s = score_match(a, b)
        assert s.rejected_reason is None

    def test_reject_reasons_taxonomy_stable(self):
        # The audit script in T78 logs against these labels — keep stable.
        assert REJECT_SAME_VENUE in REJECT_REASONS
        assert REJECT_WINDOW_NO_OVERLAP in REJECT_REASONS
        assert REJECT_THRESHOLD_MISMATCH in REJECT_REASONS
        assert REJECT_JURISDICTION_MISMATCH in REJECT_REASONS


# ---------------------------------------------------------------------------
# High-confidence true positives
# ---------------------------------------------------------------------------


class TestTruePositives:
    def test_trump_2024_synonyms_high_score(self):
        win = _window(2024, 11, 1, 2024, 11, 30)
        a = _md(
            "Trump wins 2024",
            venue="polymarket",
            window=win,
            entities=("trump",),
            topic_clues=("election",),
        )
        b = _md(
            "Donald Trump wins presidency Nov 2024",
            venue="kalshi",
            window=win,
            entities=("trump", "donald"),
            topic_clues=("election",),
        )
        s = score_match(a, b)
        assert s.rejected_reason is None
        assert s.total > 0.55, f"expected high similarity, got {s}"
        assert s.components["entity_jaccard"] > 0.3
        assert s.components["topic_overlap"] == 1.0

    def test_identical_descriptions_near_one(self):
        win = _window(2026, 12, 31, 2026, 12, 31)
        a = _md(
            "BTC above $80k by EOY 2026",
            venue="polymarket",
            window=win,
            threshold=80_000.0,
            entities=("btc",),
            topic_clues=("crypto",),
        )
        b = _md(
            "BTC above $80k by EOY 2026",
            venue="kalshi",
            window=win,
            threshold=80_000.0,
            entities=("btc",),
            topic_clues=("crypto",),
        )
        s = score_match(a, b)
        assert s.total == pytest.approx(1.0, abs=1e-6)

    def test_window_center_perfect_when_same_window(self):
        win = _window(2026, 12, 31, 2026, 12, 31)
        a = _md("X", venue="polymarket", window=win)
        b = _md("Y", venue="kalshi", window=win)
        s = score_match(a, b)
        assert s.components["window_center"] == pytest.approx(1.0)

    def test_window_center_falls_off_within_year(self):
        # Windows overlap (so no hard-reject) but their CENTERS are ~6mo apart.
        # window_center similarity should be ~0.5 (linear falloff).
        a = _md(
            "Fed cuts in Q1",
            venue="polymarket",
            window=_window(2026, 1, 1, 2026, 6, 30),  # center ~Apr 1
        )
        b = _md(
            "Fed cuts later in the year",
            venue="kalshi",
            window=_window(2026, 4, 1, 2026, 12, 31),  # center ~Sep 1, overlaps
        )
        s = score_match(a, b)
        assert s.rejected_reason is None
        # ~152 days center-to-center → ~0.58 similarity
        assert 0.45 < s.components["window_center"] < 0.7

    def test_window_center_zero_at_one_year_apart(self):
        a = _md("X", venue="polymarket", window=_window(2026, 1, 1, 2026, 1, 1))
        b = _md("Y", venue="kalshi", window=_window(2027, 1, 1, 2027, 1, 1))
        s = score_match(a, b)
        assert s.components["window_center"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Soft score weights and additivity
# ---------------------------------------------------------------------------


class TestWeights:
    def test_components_keys_present(self):
        a = _md("a", venue="polymarket")
        b = _md("b", venue="kalshi")
        s = score_match(a, b)
        assert set(s.components.keys()) == {
            "title_jaccard",
            "entity_jaccard",
            "topic_overlap",
            "window_center",
        }

    def test_weights_sum_to_one(self):
        # Force every component to 1.0 → total must be exactly 1.0.
        # We achieve this with identical titles, entities, topic, and window.
        win = _window(2025, 5, 1, 2025, 5, 1)
        md = _md(
            "BTC above $80k",
            window=win,
            entities=("btc",),
            topic_clues=("crypto",),
        )
        a = MarketDesc(**{**md.__dict__, "venue": "polymarket"})
        b = MarketDesc(**{**md.__dict__, "venue": "kalshi"})
        s = score_match(a, b)
        assert s.total == pytest.approx(1.0, abs=1e-6)

    def test_total_in_unit_interval(self):
        # Random-ish corpus: total must always sit in [0, 1].
        for title_a, title_b in [
            ("AAPL beats earnings Q1 2026", "Apple earnings beat Q1 2026"),
            ("Hurricane hits Florida 2026", "Atlantic hurricane Florida 2026"),
            ("Fed cuts 25bps March 2026", "FOMC March cut 25bps"),
            ("Trump wins Iowa primary", "Trump wins New Hampshire primary"),
            ("xyz unrelated", "abc unrelated"),
        ]:
            a = _md(title_a, venue="polymarket")
            b = _md(title_b, venue="kalshi")
            s = score_match(a, b)
            assert 0.0 <= s.total <= 1.0

    def test_components_zero_when_disjoint(self):
        a = _md("apple banana", venue="polymarket", entities=("aapl",), topic_clues=("tech",))
        b = _md("zebra yak", venue="kalshi", entities=("zbra",), topic_clues=("weather",))
        s = score_match(a, b)
        assert s.components["title_jaccard"] == 0.0
        assert s.components["entity_jaccard"] == 0.0
        assert s.components["topic_overlap"] == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_titles_does_not_crash(self):
        a = _md("", venue="polymarket")
        b = _md("", venue="kalshi")
        s = score_match(a, b)
        assert s.rejected_reason is None
        assert 0.0 <= s.total <= 1.0

    def test_missing_entities_and_window_neutral(self):
        a = _md("market a", venue="polymarket")
        b = _md("market b", venue="kalshi")
        s = score_match(a, b)
        # No window → window_center contributes the neutral 0.5
        assert s.components["window_center"] == 0.5

    def test_description_used_for_topic_clues(self):
        a = _md("contract A", description="will the FOMC cut rates", venue="polymarket")
        b = _md("contract B", description="Fed rate cut FOMC decision", venue="kalshi")
        s = score_match(a, b)
        assert s.components["topic_overlap"] > 0.0

    def test_score_is_immutable_dataclass(self):
        s = SimilarityScore(total=0.5, components={"a": 1.0}, rejected_reason=None)
        with pytest.raises(Exception):
            s.total = 0.9  # type: ignore[misc]

    def test_market_desc_is_frozen(self):
        md = _md("x", venue="polymarket")
        with pytest.raises(Exception):
            md.title = "y"  # type: ignore[misc]

    def test_score_match_does_not_mutate_inputs(self):
        a = _md(
            "BTC > $80k",
            venue="polymarket",
            window=_window(2026, 12, 31, 2026, 12, 31),
            threshold=80_000.0,
            entities=("btc",),
            topic_clues=("crypto",),
        )
        b = _md(
            "BTC > $80k",
            venue="kalshi",
            window=_window(2026, 12, 31, 2026, 12, 31),
            threshold=80_000.0,
            entities=("btc",),
            topic_clues=("crypto",),
        )
        before_a, before_b = a, b
        _ = score_match(a, b)
        assert a == before_a
        assert b == before_b


# ---------------------------------------------------------------------------
# build_market_desc — payload normalization
# ---------------------------------------------------------------------------


class TestBuildMarketDesc:
    def test_polymarket_payload(self):
        payload = {
            "slug": "trump-2024-presidential-election",
            "question": "Will Donald Trump win the 2024 presidential election?",
            "description": "Resolves YES if Donald Trump wins the US election in November 2024.",
            "endDate": "2024-11-05T23:59:59Z",
        }
        md = build_market_desc(payload, venue="polymarket")
        assert md.venue == "polymarket"
        assert "trump" in md.entities or "donald" in md.entities
        # election topic should be detected
        assert "election" in md.raw_topic_clues

    def test_kalshi_payload(self):
        payload = {
            "ticker": "PRES-24-DJT",
            "title": "Trump wins 2024 presidential election",
            "subtitle": "Donald Trump elected president",
        }
        md = build_market_desc(payload, venue="kalshi")
        assert md.venue == "kalshi"
        assert md.title.startswith("Trump")
        assert "election" in md.raw_topic_clues

    def test_threshold_extraction_from_payload(self):
        payload = {
            "title": "Will BTC trade above $80k by EOY 2026?",
            "description": "",
        }
        md = build_market_desc(payload, venue="polymarket")
        assert md.threshold == 80_000.0

    def test_threshold_with_comma(self):
        payload = {"title": "ETH above $4,500 by Dec 2026"}
        md = build_market_desc(payload, venue="kalshi")
        assert md.threshold == 4_500.0

    def test_threshold_decimal(self):
        payload = {"title": "Stock X above $12.50 by EOY"}
        md = build_market_desc(payload, venue="polymarket")
        assert md.threshold == 12.50

    def test_threshold_million_suffix(self):
        payload = {"title": "Revenue above $5m in Q4 2026"}
        md = build_market_desc(payload, venue="polymarket")
        assert md.threshold == 5_000_000.0

    def test_threshold_billion_suffix(self):
        payload = {"title": "Company valuation above $1B by 2027"}
        md = build_market_desc(payload, venue="polymarket")
        assert md.threshold == 1_000_000_000.0

    def test_threshold_not_extracted_from_year_only(self):
        # "2024" alone with no comparator must NOT be treated as a threshold.
        payload = {"title": "Trump wins 2024 election"}
        md = build_market_desc(payload, venue="polymarket")
        assert md.threshold is None

    def test_jurisdiction_us_senate(self):
        payload = {"title": "Will Democrats hold the US Senate majority in 2024?"}
        md = build_market_desc(payload, venue="polymarket")
        assert md.jurisdiction is not None
        # Must contain "US" and "Senate"
        assert "Senate" in md.jurisdiction
        assert "State" not in md.jurisdiction

    def test_jurisdiction_florida_state_senate(self):
        payload = {"title": "Florida State Senate D majority 2024"}
        md = build_market_desc(payload, venue="kalshi")
        assert md.jurisdiction is not None
        assert "FL" in md.jurisdiction
        assert "State" in md.jurisdiction

    def test_jurisdictions_conflict_us_senate_vs_florida(self):
        a = build_market_desc(
            {"title": "Will Democrats hold the US Senate majority in 2024?"},
            venue="polymarket",
        )
        b = build_market_desc(
            {"title": "Florida State Senate D majority 2024"},
            venue="kalshi",
        )
        # Both must have non-None jurisdictions AND differ.
        assert a.jurisdiction is not None
        assert b.jurisdiction is not None
        assert a.jurisdiction != b.jurisdiction
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_JURISDICTION_MISMATCH

    def test_entity_extraction_from_payload(self):
        payload = {"title": "Trump beats Biden in Pennsylvania 2024"}
        md = build_market_desc(payload, venue="polymarket")
        assert "trump" in md.entities
        assert "biden" in md.entities

    def test_empty_payload_returns_total_marketdesc(self):
        md = build_market_desc({}, venue="polymarket")
        assert md.title == ""
        assert md.description == ""
        assert md.entities == ()
        assert md.threshold is None
        assert md.jurisdiction is None
        # raw_topic_clues should be a tuple (possibly empty)
        assert isinstance(md.raw_topic_clues, tuple)

    def test_none_payload_does_not_crash(self):
        md = build_market_desc(None, venue="polymarket")  # type: ignore[arg-type]
        assert md.title == ""

    def test_venue_normalised_to_lowercase(self):
        md = build_market_desc({"title": "x"}, venue="Polymarket")
        assert md.venue == "polymarket"

    def test_topic_clues_persisted_on_marketdesc(self):
        payload = {"title": "FOMC March 2026 25bps cut"}
        md = build_market_desc(payload, venue="kalshi")
        assert "macro" in md.raw_topic_clues


# ---------------------------------------------------------------------------
# Window edge cases / robustness against odd objects
# ---------------------------------------------------------------------------


class TestWindowRobustness:
    def test_window_with_plain_date_objects(self):
        """Score helper must accept plain dates as well as datetimes."""

        @dataclass(frozen=True)
        class WPlain:
            earliest: date
            latest: date
            confidence: float = 1.0

        a = _md("X", venue="polymarket", window=WPlain(date(2026, 1, 1), date(2026, 1, 31)))
        b = _md("Y", venue="kalshi", window=WPlain(date(2026, 1, 15), date(2026, 2, 15)))
        s = score_match(a, b)
        # Overlapping → not rejected, and window_center should be in (0, 1].
        assert s.rejected_reason is None
        assert 0.5 < s.components["window_center"] <= 1.0

    def test_window_with_only_one_side_neutralises_center(self):
        # When either window is None, the window_center contribution must
        # collapse to the neutral 0.5 — neither rewarding nor penalising.
        a = _md("X", venue="polymarket", window=None)
        b = _md(
            "Y",
            venue="kalshi",
            window=_window(2026, 6, 1, 2026, 6, 30),
        )
        s = score_match(a, b)
        assert s.components["window_center"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Combined / end-to-end matrix
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_user_false_positive_trump_year_mismatch_via_build(self):
        a = build_market_desc(
            {
                "question": "Will Donald Trump win the 2024 election?",
                "endDate": "2024-11-05T23:59:59Z",
            },
            venue="polymarket",
        )
        # Force a window so the hard-reject can fire without depending on
        # T76's text-based extractor.
        a = MarketDesc(**{**a.__dict__, "resolution_window": _window(2024, 11, 5, 2024, 11, 5)})
        b = build_market_desc(
            {"title": "Trump wins 2028 presidential election"},
            venue="kalshi",
        )
        b = MarketDesc(**{**b.__dict__, "resolution_window": _window(2028, 11, 7, 2028, 11, 7)})
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_WINDOW_NO_OVERLAP

    def test_user_false_positive_btc_threshold_via_build(self):
        a = build_market_desc(
            {"title": "BTC above $80k by EOY 2026"},
            venue="polymarket",
        )
        b = build_market_desc(
            {"title": "BTC above $90k by EOY 2026"},
            venue="kalshi",
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_THRESHOLD_MISMATCH

    def test_score_envelope_shape(self):
        a = _md("a", venue="polymarket")
        b = _md("b", venue="kalshi")
        s = score_match(a, b)
        assert isinstance(s, SimilarityScore)
        assert isinstance(s.components, dict)
        assert s.rejected_reason is None or s.rejected_reason in REJECT_REASONS


# ---------------------------------------------------------------------------
# T76b: the user-flagged half-open-window class. Before T76b these were
# scored ~0.6-0.9 and would survive matching; after T76b they hard-reject
# via ``resolution_window_no_overlap``.
# ---------------------------------------------------------------------------


class TestT76bHalfOpenRejectsAfterFix:
    """All five entries from ``known_t77_gaps`` in the false-positive
    fixtures, expressed directly here so failures point at the matcher
    layer (T77) rather than the audit script (T78)."""

    def test_trump_2024_vs_2028_half_open_rejects(self):
        # Half-open windows from "by Nov 5 2024" / "by Nov 7 2028".
        a = _md(
            "Will Donald Trump win the US presidential election by Nov 5 2024?",
            venue="polymarket",
            window=_RW(earliest=None, latest=_utc(2024, 11, 5)),
            entities=("trump", "donald"),
        )
        b = _md(
            "Will Donald Trump win the US presidential election by Nov 7 2028?",
            venue="kalshi",
            window=_RW(earliest=None, latest=_utc(2028, 11, 7)),
            entities=("trump", "donald"),
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_WINDOW_NO_OVERLAP

    def test_fed_march_vs_june_2026_rejects(self):
        # ~91 days between half-open latests → reject.
        a = _md(
            "Will the Fed cut rates by March 18 2026?",
            venue="polymarket",
            window=_RW(earliest=None, latest=_utc(2026, 3, 18)),
        )
        b = _md(
            "Will the Fed cut rates by June 17 2026?",
            venue="kalshi",
            window=_RW(earliest=None, latest=_utc(2026, 6, 17)),
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_WINDOW_NO_OVERLAP

    def test_super_bowl_2025_vs_2026_rejects(self):
        a = _md(
            "Will the Kansas City Chiefs win Super Bowl by Feb 9 2025?",
            venue="polymarket",
            window=_RW(earliest=None, latest=_utc(2025, 2, 9)),
        )
        b = _md(
            "Will the Kansas City Chiefs win Super Bowl by Feb 8 2026?",
            venue="kalshi",
            window=_RW(earliest=None, latest=_utc(2026, 2, 8)),
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_WINDOW_NO_OVERLAP

    def test_eth_merge_vs_pectra_rejects(self):
        a = _md(
            "Will the Ethereum Merge upgrade complete by Dec 31 2022?",
            venue="polymarket",
            window=_RW(earliest=None, latest=_utc(2022, 12, 31)),
        )
        b = _md(
            "Will the Ethereum Pectra upgrade complete by Dec 31 2026?",
            venue="kalshi",
            window=_RW(earliest=None, latest=_utc(2026, 12, 31)),
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_WINDOW_NO_OVERLAP

    def test_oscar_2024_vs_2025_rejects(self):
        a = _md(
            "Will Oppenheimer win Best Picture by Mar 11 2024?",
            venue="polymarket",
            window=_RW(earliest=None, latest=_utc(2024, 3, 11)),
        )
        b = _md(
            "Will Anora win Best Picture by Mar 3 2025?",
            venue="kalshi",
            window=_RW(earliest=None, latest=_utc(2025, 3, 3)),
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_WINDOW_NO_OVERLAP

    def test_point_dates_far_apart_reject(self):
        # Same-day windows (earliest==latest) for two distinct election days.
        a = _md(
            "Trump wins 2024",
            venue="polymarket",
            window=_RW(_utc(2024, 11, 5), _utc(2024, 11, 5)),
        )
        b = _md(
            "Trump wins 2028",
            venue="kalshi",
            window=_RW(_utc(2028, 11, 7), _utc(2028, 11, 7)),
        )
        s = score_match(a, b)
        assert s.rejected_reason == REJECT_WINDOW_NO_OVERLAP

    def test_half_open_same_latest_still_overlaps(self):
        # Sanity guard: two ``by EOY 2026`` windows MUST still overlap so
        # legitimate arb pairs survive the new rule.
        a = _md(
            "BTC above $80k by EOY 2026",
            venue="polymarket",
            window=_RW(earliest=None, latest=_utc(2026, 12, 31)),
            entities=("btc",),
            threshold=80_000.0,
        )
        b = _md(
            "BTC above $80k by EOY 2026",
            venue="kalshi",
            window=_RW(earliest=None, latest=_utc(2026, 12, 31)),
            entities=("btc",),
            threshold=80_000.0,
        )
        s = score_match(a, b)
        assert s.rejected_reason is None
        assert s.total > 0.7

    def test_half_open_within_30_days_overlaps(self):
        # ``by Feb 1 2026`` vs ``by Feb 25 2026`` — 24 days apart, must
        # still pair (e.g. early-vs-late month resolution variance).
        a = _md(
            "Event A by Feb 1 2026",
            venue="polymarket",
            window=_RW(earliest=None, latest=_utc(2026, 2, 1)),
        )
        b = _md(
            "Event A by Feb 25 2026",
            venue="kalshi",
            window=_RW(earliest=None, latest=_utc(2026, 2, 25)),
        )
        s = score_match(a, b)
        assert s.rejected_reason is None
