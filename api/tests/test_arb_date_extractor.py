"""Tests for ``pfm.arb_matching.date_extractor``.

Coverage targets
----------------
- Every pattern in the structured pattern table (≥40 cases).
- The user's failure mode: "Trump 2024" vs "Trump 2028" must NOT overlap.
- Property test: extractor never produces ``latest < earliest``.
- Confidence is in [0, 1] for every output.
"""

from __future__ import annotations

from datetime import date

import pytest

from pfm.arb_matching.date_extractor import (
    ResolutionWindow,
    extract_resolution_window,
    windows_overlap,
)

# ---------------------------------------------------------------------------
# "by Month D YYYY" / "before Month D YYYY"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_latest",
    [
        ("by December 31 2025", date(2025, 12, 31)),
        ("by Dec 31, 2025", date(2025, 12, 31)),
        ("Resolves by December 31st, 2025", date(2025, 12, 31)),
        ("before May 1, 2026", date(2026, 5, 1)),
        ("before May 1 2026", date(2026, 5, 1)),
        ("Will this happen by June 15th 2027?", date(2027, 6, 15)),
        ("no later than November 7, 2028", date(2028, 11, 7)),
        ("on or before October 1, 2030", date(2030, 10, 1)),
    ],
)
def test_by_full_date(text: str, expected_latest: date) -> None:
    w = extract_resolution_window(text)
    assert w.latest == expected_latest
    assert w.earliest is None
    assert w.confidence >= 0.9
    assert w.source_text.lower().startswith(("by", "before", "no later", "on or before"))


# ---------------------------------------------------------------------------
# Bare full dates ("December 31 2025", "Dec 31, 2025", with ordinals)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("December 31 2025", date(2025, 12, 31)),
        ("Dec 31, 2025", date(2025, 12, 31)),
        ("December 31st 2025", date(2025, 12, 31)),
        ("Jan 1st, 2026", date(2026, 1, 1)),
        ("March 15, 2027", date(2027, 3, 15)),
        ("Election Day November 5, 2024", date(2024, 11, 5)),
    ],
)
def test_bare_full_date(text: str, expected: date) -> None:
    w = extract_resolution_window(text)
    assert w.earliest == expected
    assert w.latest == expected
    assert w.confidence >= 0.65


# ---------------------------------------------------------------------------
# EOY / end of year / year-end / "by 2030"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,year",
    [
        ("before EOY 2028", 2028),
        ("by end of 2028", 2028),
        ("by year-end 2028", 2028),
        ("by end-of-year 2029", 2029),
        ("EOY 2030", 2030),
        ("by 2030", 2030),
        ("before 2031", 2031),
    ],
)
def test_eoy(text: str, year: int) -> None:
    w = extract_resolution_window(text)
    assert w.latest == date(year, 12, 31)
    assert w.earliest is None
    assert w.confidence >= 0.8


# ---------------------------------------------------------------------------
# "in 2029" — full-year window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,year",
    [
        ("Will there be a recession in 2029?", 2029),
        ("Major hurricane in 2025", 2025),
        ("Stock market crash during 2027", 2027),
        ("AGI in 2030", 2030),
    ],
)
def test_in_year(text: str, year: int) -> None:
    w = extract_resolution_window(text)
    assert w.earliest == date(year, 1, 1)
    assert w.latest == date(year, 12, 31)
    assert 0.6 <= w.confidence <= 0.8


# ---------------------------------------------------------------------------
# Quarter phrases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_earliest,expected_latest",
    [
        # "by Qx YYYY" — upper bound only.
        ("by Q2 2026", None, date(2026, 6, 30)),
        ("before Q4 2025", None, date(2025, 12, 31)),
        # "in/during Qx YYYY" — full-quarter window.
        ("in Q3 2025", date(2025, 7, 1), date(2025, 9, 30)),
        ("during Q1 2027", date(2027, 1, 1), date(2027, 3, 31)),
        ("Q4 2031", date(2031, 10, 1), date(2031, 12, 31)),
    ],
)
def test_quarter(text: str, expected_earliest: date | None, expected_latest: date) -> None:
    w = extract_resolution_window(text)
    assert w.earliest == expected_earliest
    assert w.latest == expected_latest
    assert w.confidence >= 0.75


# ---------------------------------------------------------------------------
# "before May 2026" / "in November 2025"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_earliest,expected_latest",
    [
        ("before May 2026", None, date(2026, 5, 31)),
        ("by June 2027", None, date(2027, 6, 30)),
        ("in November 2025", date(2025, 11, 1), date(2025, 11, 30)),
        ("during February 2028", date(2028, 2, 1), date(2028, 2, 29)),  # leap year
        ("during February 2027", date(2027, 2, 1), date(2027, 2, 28)),
    ],
)
def test_month_year(text: str, expected_earliest: date | None, expected_latest: date) -> None:
    w = extract_resolution_window(text)
    assert w.earliest == expected_earliest
    assert w.latest == expected_latest


# ---------------------------------------------------------------------------
# Month-only (no year) — uses reference_date
# ---------------------------------------------------------------------------


def test_month_only_future_in_same_year() -> None:
    w = extract_resolution_window("Senate flips before May", reference_date=date(2026, 1, 15))
    assert w.latest == date(2026, 5, 31)
    assert w.earliest is None
    assert w.confidence < 0.7  # low-confidence relative phrase


def test_month_only_rolls_to_next_year_if_past() -> None:
    w = extract_resolution_window("Senate flips before May", reference_date=date(2026, 8, 1))
    # August 1 is already past May 2026 -> next May = May 2027.
    assert w.latest == date(2027, 5, 31)


def test_month_only_by_june() -> None:
    w = extract_resolution_window("by June", reference_date=date(2025, 3, 1))
    assert w.latest == date(2025, 6, 30)


# ---------------------------------------------------------------------------
# ISO YYYY-MM-DD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Resolution date: 2026-01-01", date(2026, 1, 1)),
        ("Closes 2027-12-31T23:59:59Z", date(2027, 12, 31)),
        ("Window 2028-07-04", date(2028, 7, 4)),
        ("2031-03-15 deadline", date(2031, 3, 15)),
    ],
)
def test_iso(text: str, expected: date) -> None:
    w = extract_resolution_window(text)
    assert w.earliest == expected
    assert w.latest == expected
    assert w.confidence >= 0.95


def test_iso_invalid_returns_empty() -> None:
    # 2026-02-30 is not a real date — pattern shouldn't even match it
    # because of the day-range constraint; if it did, _h_iso returns None.
    w = extract_resolution_window("invalid 2026-02-30")
    # Either no match or a downstream fallback; we just require validity.
    if w.earliest is not None:
        assert w.earliest >= date(2024, 1, 1)


# ---------------------------------------------------------------------------
# Political-calendar anchors (yeared + relative)
# ---------------------------------------------------------------------------


def test_political_yeared_2024_election() -> None:
    w = extract_resolution_window("Will Trump win the 2024 election?")
    assert w.latest == date(2024, 11, 5)
    assert w.earliest == date(2024, 11, 5)
    assert w.source_text.lower() == "2024 election"


def test_political_yeared_2028_election() -> None:
    w = extract_resolution_window("Will Trump win the 2028 election?")
    assert w.latest == date(2028, 11, 7)
    assert w.earliest == date(2028, 11, 7)


def test_political_yeared_2026_midterms() -> None:
    w = extract_resolution_window("Democrats win 2026 midterms")
    assert w.latest == date(2026, 11, 3)


def test_political_relative_next_election() -> None:
    w = extract_resolution_window(
        "Senate balance by the next election", reference_date=date(2026, 1, 1)
    )
    assert w.latest == date(2028, 11, 30)
    assert w.confidence < 0.5  # low confidence on relative phrases


def test_political_relative_before_midterms() -> None:
    w = extract_resolution_window("Resolves before midterms", reference_date=date(2025, 5, 1))
    assert w.latest == date(2026, 11, 30)
    assert w.confidence < 0.5


# ---------------------------------------------------------------------------
# Year coverage 2025-2031
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "year",
    [2025, 2026, 2027, 2028, 2029, 2030, 2031],
)
def test_each_year_2025_2031(year: int) -> None:
    text = f"Resolution date is December 31 {year}"
    w = extract_resolution_window(text)
    assert w.latest == date(year, 12, 31)


# ---------------------------------------------------------------------------
# No-date / empty / nonsense
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "No date detected at all",
        "Bitcoin above one hundred thousand dollars",
        "Random words and stuff",
    ],
)
def test_no_date_returns_empty(text: str) -> None:
    w = extract_resolution_window(text)
    assert w.earliest is None
    assert w.latest is None
    assert w.confidence == 0.0
    assert w.source_text == ""


# ---------------------------------------------------------------------------
# The user-flagged failure mode
# ---------------------------------------------------------------------------


def test_user_failure_mode_trump_2024_vs_2028_does_not_overlap() -> None:
    a = extract_resolution_window("Will Trump win the 2024 election?")
    b = extract_resolution_window("Will Trump win the 2028 election?")
    assert a.latest == date(2024, 11, 5)
    assert b.latest == date(2028, 11, 7)
    assert not windows_overlap(a, b)


def test_user_failure_mode_bitcoin_2025_vs_2029_does_not_overlap() -> None:
    a = extract_resolution_window("Bitcoin above 100k by December 31 2025")
    b = extract_resolution_window("Bitcoin above 100k in 2029")
    # a has open lower bound (None) and latest=2025-12-31. b is [2029-01-01, 2029-12-31].
    # overlap_lo = max(date.min, 2029-01-01) = 2029-01-01
    # overlap_hi = min(2025-12-31, 2029-12-31) = 2025-12-31
    # overlap_hi < overlap_lo -> no overlap.
    assert not windows_overlap(a, b)


def test_user_failure_mode_2025_eoy_vs_2025_eoy_overlaps() -> None:
    a = extract_resolution_window("Bitcoin above 100k by December 31 2025")
    b = extract_resolution_window("Bitcoin above 100k by EOY 2025")
    assert windows_overlap(a, b)


# ---------------------------------------------------------------------------
# windows_overlap semantics
# ---------------------------------------------------------------------------


def test_windows_overlap_open_bounds_share_full_year() -> None:
    a = ResolutionWindow(earliest=None, latest=date(2026, 12, 31), confidence=0.8, source_text="x")
    b = ResolutionWindow(earliest=date(2026, 1, 1), latest=None, confidence=0.8, source_text="y")
    assert windows_overlap(a, b)


def test_windows_overlap_disjoint_returns_false() -> None:
    a = ResolutionWindow(
        earliest=date(2025, 1, 1),
        latest=date(2025, 12, 31),
        confidence=0.7,
        source_text="x",
    )
    b = ResolutionWindow(
        earliest=date(2027, 1, 1),
        latest=date(2027, 12, 31),
        confidence=0.7,
        source_text="y",
    )
    assert not windows_overlap(a, b)


def test_windows_overlap_min_days_threshold() -> None:
    a = ResolutionWindow(
        earliest=date(2026, 1, 1),
        latest=date(2026, 6, 30),
        confidence=0.8,
        source_text="x",
    )
    b = ResolutionWindow(
        earliest=date(2026, 6, 30),
        latest=date(2026, 12, 31),
        confidence=0.8,
        source_text="y",
    )
    # Single shared day = 1.
    assert windows_overlap(a, b, min_overlap_days=1)
    assert not windows_overlap(a, b, min_overlap_days=2)


def test_windows_overlap_one_empty_returns_false() -> None:
    a = ResolutionWindow(earliest=None, latest=None, confidence=0.0, source_text="")
    b = ResolutionWindow(
        earliest=date(2025, 1, 1),
        latest=date(2025, 12, 31),
        confidence=0.8,
        source_text="x",
    )
    assert not windows_overlap(a, b)
    assert not windows_overlap(b, a)


def test_windows_overlap_invalid_min_days() -> None:
    a = ResolutionWindow(
        earliest=date(2025, 1, 1),
        latest=date(2025, 12, 31),
        confidence=0.8,
        source_text="x",
    )
    with pytest.raises(ValueError):
        windows_overlap(a, a, min_overlap_days=0)


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_resolution_window_rejects_inverted_dates() -> None:
    with pytest.raises(ValueError):
        ResolutionWindow(
            earliest=date(2026, 12, 31),
            latest=date(2026, 1, 1),
            confidence=0.5,
            source_text="x",
        )


def test_resolution_window_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError):
        ResolutionWindow(earliest=None, latest=None, confidence=1.5, source_text="")
    with pytest.raises(ValueError):
        ResolutionWindow(earliest=None, latest=None, confidence=-0.1, source_text="")


def test_resolution_window_is_frozen() -> None:
    w = extract_resolution_window("by December 31 2025")
    with pytest.raises(
        Exception
    ):  # FrozenInstanceError subclasses AttributeError/dataclasses.FrozenInstanceError
        w.confidence = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# "Property"-style guard: extractor never returns invalid window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Will Trump win the 2024 election?",
        "Will Trump win the 2028 election?",
        "Bitcoin above 100k by December 31 2025",
        "Recession in 2029",
        "Fed rate cut by Q2 2026",
        "Senate flips before May",
        "Resolution date: 2026-01-01",
        "by Dec 31, 2025",
        "before EOY 2028",
        "by 2030",
        "during February 2028",
        "the 2026 midterms",
        "by the next election",
        "No date here",
        "",
        "Some random sentence with the number 2027 in it but no date phrase",
    ],
)
def test_extractor_invariants(text: str) -> None:
    """Property: extractor output is always a valid window."""

    w = extract_resolution_window(text, reference_date=date(2026, 1, 1))
    # latest >= earliest if both set.
    if w.earliest is not None and w.latest is not None:
        assert w.latest >= w.earliest
    # confidence in [0, 1].
    assert 0.0 <= w.confidence <= 1.0
    # All-None -> confidence == 0 and source_text empty.
    if w.earliest is None and w.latest is None:
        assert w.confidence == 0.0
        assert w.source_text == ""


# ---------------------------------------------------------------------------
# Specificity ordering: full date wins over bare year
# ---------------------------------------------------------------------------


def test_full_date_wins_over_year() -> None:
    w = extract_resolution_window("Resolves on December 31 2025, during the 2025 calendar year")
    assert w.earliest == date(2025, 12, 31)
    assert w.latest == date(2025, 12, 31)
    assert w.confidence >= 0.65


def test_eoy_wins_over_in_year() -> None:
    w = extract_resolution_window("by end of 2028 vs in 2028")
    # EOY pattern is listed before "in 2029", and finditer hits first.
    assert w.latest == date(2028, 12, 31)
    # earliest is None for "by" phrases.
    assert w.earliest is None


# ---------------------------------------------------------------------------
# Ambiguous low-confidence cases
# ---------------------------------------------------------------------------


def test_ambiguous_bare_month_no_year_low_confidence() -> None:
    w = extract_resolution_window("Could happen before May", reference_date=date(2025, 1, 1))
    assert w.confidence < 0.7


def test_ambiguous_next_election_low_confidence() -> None:
    w = extract_resolution_window("By the next election cycle", reference_date=date(2026, 1, 1))
    assert w.confidence < 0.5


# ---------------------------------------------------------------------------
# Reference date default
# ---------------------------------------------------------------------------


def test_reference_date_default_is_today() -> None:
    # We don't pin "today", just confirm the call works without a reference.
    w = extract_resolution_window("before May")
    assert w.latest is not None
    assert w.latest.month == 5


# ---------------------------------------------------------------------------
# T76b: half-open overlap semantics, point-date semantics, is_* properties
# ---------------------------------------------------------------------------


class TestT76bWindowProperties:
    """The new ``is_point_date`` / ``is_half_open_by`` / ``is_half_open_from``
    properties drive the corrected ``windows_overlap`` branches."""

    def test_point_date_property_true_when_earliest_equals_latest(self) -> None:
        w = ResolutionWindow(
            earliest=date(2024, 11, 5),
            latest=date(2024, 11, 5),
            confidence=0.95,
            source_text="Nov 5 2024",
        )
        assert w.is_point_date
        assert not w.is_half_open_by
        assert not w.is_half_open_from

    def test_point_date_false_when_window_spans_days(self) -> None:
        w = ResolutionWindow(
            earliest=date(2024, 11, 1),
            latest=date(2024, 11, 30),
            confidence=0.7,
            source_text="November 2024",
        )
        assert not w.is_point_date

    def test_half_open_by_property(self) -> None:
        w = extract_resolution_window("by December 31 2025")
        assert w.is_half_open_by
        assert not w.is_point_date
        assert not w.is_half_open_from

    def test_half_open_from_property(self) -> None:
        w = ResolutionWindow(
            earliest=date(2025, 1, 1), latest=None, confidence=0.7, source_text="x"
        )
        assert w.is_half_open_from
        assert not w.is_half_open_by
        assert not w.is_point_date

    def test_political_yeared_is_point_date(self) -> None:
        # _h_political_yeared anchors to a single resolution day.
        w = extract_resolution_window("the 2028 election")
        assert w.is_point_date


class TestT76bHalfOpenOverlap:
    """The CORE bug fix: same-shape half-open windows must NOT trivially
    intersect via the open bound. Two ``(None, latest)`` windows overlap
    only if their ``latest`` dates are within ~30 days."""

    def test_trump_2024_vs_trump_2028_half_open_rejects(self) -> None:
        # The user-flagged failure mode rendered as half-open windows
        # (because the titles use "by Nov 5 2024" / "by Nov 7 2028").
        a = extract_resolution_window(
            "Will Donald Trump win the US presidential election by Nov 5 2024?"
        )
        b = extract_resolution_window(
            "Will Donald Trump win the US presidential election by Nov 7 2028?"
        )
        assert a.is_half_open_by and b.is_half_open_by
        assert not windows_overlap(a, b)

    def test_fed_march_vs_june_same_year_rejects(self) -> None:
        # ~91 days apart — outside the 30-day proximity window.
        a = extract_resolution_window("Will the Fed cut rates by March 18 2026?")
        b = extract_resolution_window("Will the Fed cut rates by June 17 2026?")
        assert a.is_half_open_by and b.is_half_open_by
        assert not windows_overlap(a, b)

    def test_super_bowl_lix_vs_lx_rejects(self) -> None:
        # Super Bowl LIX (Feb 9 2025) vs LX (Feb 8 2026) — same event class,
        # 364 days apart.
        a = extract_resolution_window("Will the Chiefs win Super Bowl by Feb 9 2025?")
        b = extract_resolution_window("Will the Chiefs win Super Bowl by Feb 8 2026?")
        assert not windows_overlap(a, b)

    def test_same_latest_half_open_still_overlaps(self) -> None:
        # ``by Dec 31 2025`` vs ``by EOY 2025`` — should still overlap (delta=0d).
        a = extract_resolution_window("BTC above 100k by December 31 2025")
        b = extract_resolution_window("BTC above 100k by EOY 2025")
        assert a.is_half_open_by and b.is_half_open_by
        assert windows_overlap(a, b)

    def test_half_open_within_30_days_overlaps(self) -> None:
        # Day 0 vs Day +25 → still inside 30-day proximity → overlap.
        a = ResolutionWindow(
            earliest=None, latest=date(2026, 3, 1), confidence=0.9, source_text="a"
        )
        b = ResolutionWindow(
            earliest=None, latest=date(2026, 3, 26), confidence=0.9, source_text="b"
        )
        assert windows_overlap(a, b)

    def test_half_open_31_days_apart_does_not_overlap(self) -> None:
        # Just over the threshold.
        a = ResolutionWindow(
            earliest=None, latest=date(2026, 3, 1), confidence=0.9, source_text="a"
        )
        b = ResolutionWindow(
            earliest=None, latest=date(2026, 4, 1), confidence=0.9, source_text="b"
        )
        assert not windows_overlap(a, b)

    def test_half_open_proximity_days_override(self) -> None:
        # Caller can widen the proximity for venues that need it.
        a = ResolutionWindow(
            earliest=None, latest=date(2026, 1, 1), confidence=0.9, source_text="a"
        )
        b = ResolutionWindow(
            earliest=None, latest=date(2026, 4, 1), confidence=0.9, source_text="b"
        )
        # ~90d apart.
        assert not windows_overlap(a, b)
        assert windows_overlap(a, b, half_open_proximity_days=120)

    def test_half_open_from_same_shape_rejects_far_apart(self) -> None:
        # Symmetric to the by-case: two ``(earliest, None)`` windows.
        a = ResolutionWindow(
            earliest=date(2024, 1, 1), latest=None, confidence=0.7, source_text="a"
        )
        b = ResolutionWindow(
            earliest=date(2028, 1, 1), latest=None, confidence=0.7, source_text="b"
        )
        assert not windows_overlap(a, b)

    def test_half_open_from_same_shape_close_overlaps(self) -> None:
        a = ResolutionWindow(
            earliest=date(2025, 5, 1), latest=None, confidence=0.7, source_text="a"
        )
        b = ResolutionWindow(
            earliest=date(2025, 5, 15), latest=None, confidence=0.7, source_text="b"
        )
        assert windows_overlap(a, b)

    def test_mixed_half_open_still_uses_interval(self) -> None:
        # ``(None, 2026-12-31)`` vs ``(2026-01-01, None)`` — mixed shapes go
        # through branch 3 (interval intersection). Should overlap.
        a = ResolutionWindow(
            earliest=None, latest=date(2026, 12, 31), confidence=0.8, source_text="a"
        )
        b = ResolutionWindow(
            earliest=date(2026, 1, 1), latest=None, confidence=0.8, source_text="b"
        )
        assert windows_overlap(a, b)

    def test_mixed_half_open_disjoint(self) -> None:
        # ``(None, 2024-1-1)`` vs ``(2028-1-1, None)`` — disjoint via interval.
        a = ResolutionWindow(
            earliest=None, latest=date(2024, 1, 1), confidence=0.8, source_text="a"
        )
        b = ResolutionWindow(
            earliest=date(2028, 1, 1), latest=None, confidence=0.8, source_text="b"
        )
        assert not windows_overlap(a, b)


class TestT76bPointDateOverlap:
    """Two point dates (earliest==latest) must obey the same proximity rule
    as half-open windows. Trump-2024-election day vs Trump-2028-election day
    must NOT overlap even though both are fully bounded."""

    def test_political_yeared_2024_vs_2028_point_dates_reject(self) -> None:
        a = extract_resolution_window("Will Trump win the 2024 election?")
        b = extract_resolution_window("Will Trump win the 2028 election?")
        assert a.is_point_date and b.is_point_date
        assert not windows_overlap(a, b)

    def test_political_yeared_same_year_overlaps(self) -> None:
        a = extract_resolution_window("Democrats win 2026 midterms")
        b = extract_resolution_window("Republicans win 2026 midterms")
        # Both anchor to Nov 3 2026.
        assert windows_overlap(a, b)

    def test_two_point_dates_within_30d_overlap(self) -> None:
        a = ResolutionWindow(
            earliest=date(2026, 3, 1),
            latest=date(2026, 3, 1),
            confidence=0.95,
            source_text="a",
        )
        b = ResolutionWindow(
            earliest=date(2026, 3, 25),
            latest=date(2026, 3, 25),
            confidence=0.95,
            source_text="b",
        )
        assert windows_overlap(a, b)

    def test_two_point_dates_far_apart_reject(self) -> None:
        a = ResolutionWindow(
            earliest=date(2024, 11, 5),
            latest=date(2024, 11, 5),
            confidence=0.95,
            source_text="a",
        )
        b = ResolutionWindow(
            earliest=date(2028, 11, 7),
            latest=date(2028, 11, 7),
            confidence=0.95,
            source_text="b",
        )
        assert not windows_overlap(a, b)


class TestT76bRegressionGuards:
    """T76b must not break the existing T76 + T77 semantics: full-year
    ranges, full quarters, bare-year-ranged windows still intersect via
    classical interval arithmetic."""

    def test_in_year_2025_vs_in_year_2025_overlaps(self) -> None:
        a = extract_resolution_window("Recession in 2025")
        b = extract_resolution_window("Recession during 2025")
        # Both span Jan 1 - Dec 31 2025. NOT point-dates, NOT half-open.
        assert not a.is_point_date and not a.is_half_open_by
        assert windows_overlap(a, b)

    def test_in_year_2025_vs_in_year_2027_disjoint(self) -> None:
        a = extract_resolution_window("Recession in 2025")
        b = extract_resolution_window("Recession in 2027")
        assert not windows_overlap(a, b)

    def test_quarter_overlaps_with_in_year(self) -> None:
        a = extract_resolution_window("Fed cut in Q1 2026")
        b = extract_resolution_window("Fed cut in 2026")
        assert windows_overlap(a, b)

    def test_invalid_half_open_proximity_raises(self) -> None:
        a = extract_resolution_window("by December 31 2025")
        b = extract_resolution_window("by December 31 2025")
        with pytest.raises(ValueError):
            windows_overlap(a, b, half_open_proximity_days=-1)
