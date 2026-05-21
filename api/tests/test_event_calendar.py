"""Tests for ``pfm.vol.event_calendar`` — integrity + helpers.

The critical test is :func:`test_every_slug_exists_in_factors_yml`: it
guarantees the curated calendar never silently references a slug that
has been pruned from ``factors.yml``.
"""

from __future__ import annotations

import functools
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pfm.vol.event_calendar import (
    CALENDAR,
    EventEntry,
    OutcomeSlug,
    get_event,
    list_by_kind,
    list_upcoming,
)

# ---------------------------------------------------------------------------
# Cached factors.yml loader
# ---------------------------------------------------------------------------


@functools.cache
def _load_factors_text() -> str:
    """Read ``api/src/pfm/factors.yml`` once and cache the contents."""
    here = Path(__file__).resolve().parent
    factors_path = here.parent / "src" / "pfm" / "factors.yml"
    assert factors_path.exists(), f"factors.yml not found at {factors_path}"
    return factors_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------


def test_calendar_loaded_and_non_empty() -> None:
    assert isinstance(CALENDAR, list)
    assert len(CALENDAR) >= 6, f"Expected ≥6 curated events, got {len(CALENDAR)}"


def test_each_event_has_at_least_3_outcomes() -> None:
    for entry in CALENDAR:
        assert len(entry.outcome_slugs) >= 3, (
            f"Event {entry.event_id!r} has only {len(entry.outcome_slugs)} "
            "outcomes; partition coherence requires ≥3."
        )


def test_each_event_has_unique_id() -> None:
    ids = [entry.event_id for entry in CALENDAR]
    assert len(set(ids)) == len(ids), f"Duplicate event_id in CALENDAR: {ids}"


def test_every_slug_exists_in_factors_yml() -> None:
    """The integrity guard. If any slug below is missing from
    factors.yml, this test fails loudly so the calendar can never
    drift away from the actual contract universe.
    """
    text = _load_factors_text()
    missing: list[tuple[str, str]] = []
    for entry in CALENDAR:
        for outcome in entry.outcome_slugs:
            if outcome.slug not in text:
                missing.append((entry.event_id, outcome.slug))
    assert not missing, (
        "Slugs referenced by event_calendar.CALENDAR but absent from "
        "factors.yml:\n" + "\n".join(f"  - {eid}: {slug}" for eid, slug in missing)
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_get_event_returns_none_for_unknown() -> None:
    assert get_event("fake-id") is None
    assert get_event("") is None


def test_get_event_returns_entry_for_known() -> None:
    entry = get_event("fomc-2026-06")
    assert entry is not None
    assert isinstance(entry, EventEntry)
    assert entry.event_id == "fomc-2026-06"
    assert entry.event_kind == "fomc"
    assert entry.underlying_ticker == "SPY"


def test_list_upcoming_filters_by_date_window() -> None:
    now = datetime(2026, 5, 15, tzinfo=UTC)
    upcoming = list_upcoming(now, lookahead_days=60)
    ids = {e.event_id for e in upcoming}
    # FOMC Jun-17 and CPI May-release (Jun-11) both fall in [May 15, Jul 14].
    assert "fomc-2026-06" in ids
    assert "cpi-2026-05" in ids
    # Midterms (Nov-3) is FAR outside the 60-day window.
    assert "midterms-2026" not in ids
    # December FOMC and Brazil-Oct are also outside.
    assert "fomc-2026-12" not in ids
    assert "brazil-pres-2026" not in ids


def test_list_upcoming_handles_naive_datetime() -> None:
    """A naive datetime should be treated as UTC, not error out."""
    naive = datetime(2026, 5, 15)
    upcoming = list_upcoming(naive, lookahead_days=60)
    assert any(e.event_id == "fomc-2026-06" for e in upcoming)


def test_list_by_kind_fomc() -> None:
    fomcs = list_by_kind("fomc")
    assert len(fomcs) >= 3
    ids = {e.event_id for e in fomcs}
    assert {"fomc-2026-06", "fomc-2026-07", "fomc-2026-12"} <= ids


def test_list_by_kind_unknown_returns_empty() -> None:
    assert list_by_kind("nonexistent-kind") == []


# ---------------------------------------------------------------------------
# Anchor-value semantics
# ---------------------------------------------------------------------------


def test_anchor_values_for_fomc_are_signed_correctly() -> None:
    """Negative for cuts, zero for no-change, positive for hikes."""
    for entry in list_by_kind("fomc"):
        for outcome in entry.outcome_slugs:
            label = outcome.label.lower()
            if "cut" in label:
                assert outcome.anchor_value < 0, (
                    f"{entry.event_id}/{outcome.label}: cut anchor must be "
                    f"negative, got {outcome.anchor_value}"
                )
            elif "hike" in label:
                assert outcome.anchor_value > 0, (
                    f"{entry.event_id}/{outcome.label}: hike anchor must be "
                    f"positive, got {outcome.anchor_value}"
                )
            elif "no_change" in label:
                assert outcome.anchor_value == 0.0, (
                    f"{entry.event_id}/{outcome.label}: no_change anchor "
                    f"must be 0, got {outcome.anchor_value}"
                )


def test_calendar_anchor_values_are_distinct_per_event() -> None:
    """Within a single event, no two outcomes share an anchor_value."""
    for entry in CALENDAR:
        anchors = [o.anchor_value for o in entry.outcome_slugs]
        assert len(set(anchors)) == len(anchors), (
            f"Event {entry.event_id!r} has duplicate anchor_values: {anchors}"
        )


# ---------------------------------------------------------------------------
# Pydantic schema sanity
# ---------------------------------------------------------------------------


def test_outcome_slug_venue_validated() -> None:
    with pytest.raises(Exception):
        OutcomeSlug(label="x", anchor_value=0.0, venue="bogus-venue", slug="s")  # type: ignore[arg-type]


def test_event_entry_kind_validated() -> None:
    with pytest.raises(Exception):
        EventEntry(
            event_id="x",
            event_kind="weather",  # type: ignore[arg-type]
            description="d",
            scheduled_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
            underlying_ticker="SPY",
            outcome_slugs=[
                OutcomeSlug(label="a", anchor_value=0.0, venue="polymarket", slug="s"),
            ],
        )
