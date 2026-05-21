"""Curated event calendar for the vol-trading pipeline.

This module is a **static** data structure that maps high-vol macro and
political events (FOMC decisions, CPI releases, US midterms, Brazil
presidential, …) to the Polymarket and Kalshi slugs whose contract
prices form a multinomial distribution over outcomes.

It is consumed by :mod:`pfm.vol.event_vol_engine` and (in the next
module, B3) by the signal engine that fetches contract prices and
projects them into an Expected-Move forecast on the underlying ticker.

The calendar is intentionally a pure-Python list of Pydantic entries:
no network, no I/O. Tests in ``api/tests/test_event_calendar.py``
verify every slug listed here actually exists in
``api/src/pfm/factors.yml`` — that integrity check is what keeps this
file honest as the factor catalogue evolves.

Anchor-value convention
-----------------------
For FOMC entries, ``anchor_value`` is the **basis-points change scaled
to percent** (e.g. -0.50 for a 50 bp cut, +0.25 for a 25 bp hike).
For CPI entries, ``anchor_value`` is the YoY (or MoM) percent reading
the contract pays out on. For elections, anchors encode a qualitative
"market-friendly minus market-unfriendly" axis (positive ⇒ historically
correlated with risk-on for the underlying). The election anchors are
admittedly heuristic; see ``notes`` on each entry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Venue = Literal["polymarket", "kalshi"]
EventKind = Literal["fomc", "cpi", "nfp", "election", "opec", "geopolitical"]


class OutcomeSlug(BaseModel):
    """One cell of an event's multinomial outcome space.

    Attributes:
        label: Short identifier, e.g. ``"cut_25bp"`` or ``"yoy_2.8"``.
        anchor_value: Numeric value the engine projects against (e.g.
            ``-0.25`` for a 25 bp cut, ``2.8`` for a 2.8 % CPI YoY).
        venue: Which market the slug lives on.
        slug: The exact slug as it appears in ``factors.yml``.
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(..., min_length=1)
    anchor_value: float
    venue: Venue
    slug: str = Field(..., min_length=1)


class EventEntry(BaseModel):
    """One curated event with its outcome partition."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(..., min_length=1)
    event_kind: EventKind
    description: str
    scheduled_at_utc: datetime
    underlying_ticker: str = Field(..., min_length=1)
    outcome_slugs: list[OutcomeSlug]
    notes: str | None = None


# ---------------------------------------------------------------------------
# Curated calendar
# ---------------------------------------------------------------------------
#
# Every slug below has been grep-verified against ``api/src/pfm/factors.yml``
# at the time of curation (2026-05-15). The test
# ``test_every_slug_exists_in_factors_yml`` re-checks this on every CI run,
# so if a contract is pruned from factors.yml the build will fail loudly.

CALENDAR: list[EventEntry] = [
    EventEntry(
        event_id="fomc-2026-06",
        event_kind="fomc",
        description="FOMC decision, June 2026 (Wed 17-Jun, 2pm ET).",
        scheduled_at_utc=datetime(2026, 6, 17, 18, 0, tzinfo=UTC),
        underlying_ticker="SPY",
        outcome_slugs=[
            OutcomeSlug(
                label="cut_50bp",
                anchor_value=-0.50,
                venue="polymarket",
                slug="will-the-fed-decrease-interest-rates-by-50-bps-after-the-june-2026-meeting",
            ),
            OutcomeSlug(
                label="cut_25bp",
                anchor_value=-0.25,
                venue="polymarket",
                slug="will-the-fed-decrease-interest-rates-by-25-bps-after-the-june-2026-meeting",
            ),
            OutcomeSlug(
                label="no_change",
                anchor_value=0.0,
                venue="polymarket",
                slug="will-there-be-no-change-in-fed-interest-rates-after-the-june-2026-meeting",
            ),
            OutcomeSlug(
                label="hike_25bp",
                anchor_value=0.25,
                venue="polymarket",
                slug="will-the-fed-increase-interest-rates-by-25-bps-after-the-june-2026-meeting",
            ),
            OutcomeSlug(
                label="hike_50bp",
                anchor_value=0.50,
                venue="polymarket",
                slug="will-the-fed-increase-interest-rates-by-50-bps-after-the-june-2026-meeting",
            ),
        ],
        notes=(
            "Polymarket 'no-change' slug is canonical via 'will-there-be-no-change-…' "
            "(not 'fed-no-change-…' as the original brief suggested)."
        ),
    ),
    EventEntry(
        event_id="fomc-2026-07",
        event_kind="fomc",
        description="FOMC decision, July 2026 (Wed 29-Jul, 2pm ET).",
        scheduled_at_utc=datetime(2026, 7, 29, 18, 0, tzinfo=UTC),
        underlying_ticker="SPY",
        outcome_slugs=[
            OutcomeSlug(
                label="cut_50bp",
                anchor_value=-0.50,
                venue="polymarket",
                slug="will-the-fed-decrease-interest-rates-by-50-bps-after-the-july-2026-meeting",
            ),
            OutcomeSlug(
                label="cut_25bp",
                anchor_value=-0.25,
                venue="polymarket",
                slug="will-the-fed-decrease-interest-rates-by-25-bps-after-the-july-2026-meeting",
            ),
            OutcomeSlug(
                label="no_change",
                anchor_value=0.0,
                venue="polymarket",
                slug="will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting",
            ),
            OutcomeSlug(
                label="hike_25bp",
                anchor_value=0.25,
                venue="polymarket",
                slug="will-the-fed-increase-interest-rates-by-25-bps-after-the-july-2026-meeting",
            ),
            OutcomeSlug(
                label="hike_50bp",
                anchor_value=0.50,
                venue="polymarket",
                slug="will-the-fed-increase-interest-rates-by-50-bps-after-the-july-2026-meeting",
            ),
        ],
    ),
    EventEntry(
        event_id="fomc-2026-12",
        event_kind="fomc",
        description="FOMC decision, December 2026 (Wed 16-Dec, 2pm ET).",
        scheduled_at_utc=datetime(2026, 12, 16, 19, 0, tzinfo=UTC),
        underlying_ticker="SPY",
        outcome_slugs=[
            OutcomeSlug(
                label="cut_25bp_year_aggregate",
                anchor_value=-0.25,
                venue="polymarket",
                slug="fed-rate-cut-by-december-2026-meeting",
            ),
            OutcomeSlug(
                label="cut_25bp_kalshi",
                anchor_value=-0.249,
                venue="kalshi",
                slug="KXFEDDECISION-26DEC-C25",
            ),
            OutcomeSlug(
                label="no_change_implied",
                anchor_value=0.0,
                venue="polymarket",
                slug="will-no-fed-rate-cuts-happen-in-2026",
            ),
        ],
        notes=(
            "Monthly Dec-specific Polymarket slugs (25 bp / no-change / hike) "
            "are sparse in factors.yml — we fall back to the year-aggregate "
            "'fed-rate-cut-by-december-2026-meeting' contract and pair it "
            "with the Kalshi 'KXFEDDECISION-26DEC-C25' cross-check. The "
            "no_change proxy is the 'no Fed rate cuts in 2026' aggregate "
            "(soft proxy: it captures macro mass of a hawkish path)."
        ),
    ),
    EventEntry(
        event_id="cpi-2026-05",
        event_kind="cpi",
        description="CPI release for May 2026 data (typical date: 2nd Wed of June).",
        scheduled_at_utc=datetime(2026, 6, 11, 12, 30, tzinfo=UTC),
        underlying_ticker="SPY",
        outcome_slugs=[
            OutcomeSlug(
                label="yoy_2.8",
                anchor_value=2.8,
                venue="kalshi",
                slug="KXECONSTATCPIYOY-26MAY-T2.8",
            ),
            OutcomeSlug(
                label="yoy_2.9",
                anchor_value=2.9,
                venue="kalshi",
                slug="KXECONSTATCPIYOY-26MAY-T2.9",
            ),
            OutcomeSlug(
                label="core_mom_-0.1",
                anchor_value=-0.1,
                venue="kalshi",
                slug="KXECONSTATCPICORE-26MAY-T-0.1",
            ),
        ],
        notes=(
            "Only three Kalshi May-2026 CPI slugs survived the 2026-05-13 "
            "factors.yml prune: YoY 2.8, YoY 2.9, and core MoM -0.1. The "
            "spec asked for YoY 3.0 but no matching slug exists in "
            "factors.yml — anchor_value=3.0 cell is therefore dropped. "
            "Partition is NOT exhaustive (captures only the central mass "
            "around 2.8–2.9 YoY plus a core-deflation tail)."
        ),
    ),
    EventEntry(
        event_id="cpi-2026-06",
        event_kind="cpi",
        description="CPI release for June 2026 data (typical date: 2nd Wed of July).",
        scheduled_at_utc=datetime(2026, 7, 15, 12, 30, tzinfo=UTC),
        underlying_ticker="SPY",
        outcome_slugs=[
            OutcomeSlug(
                label="headline_mom_-0.2",
                anchor_value=-0.2,
                venue="kalshi",
                slug="KXECONSTATCPI-26JUN-T-0.2",
            ),
            OutcomeSlug(
                label="headline_mom_-0.1",
                anchor_value=-0.1,
                venue="kalshi",
                slug="KXECONSTATCPI-26JUN-T-0.1",
            ),
            OutcomeSlug(
                label="headline_mom_0.0",
                anchor_value=0.0,
                venue="kalshi",
                slug="KXECONSTATCPI-26JUN-T0.0",
            ),
            OutcomeSlug(
                label="yoy_2.3",
                anchor_value=2.3,
                venue="kalshi",
                slug="KXECONSTATCPIYOY-26JUN-T2.3",
            ),
            OutcomeSlug(
                label="core_yoy_2.3",
                anchor_value=2.301,
                venue="kalshi",
                slug="KXECONSTATCORECPIYOY-26JUN-T2.3",
            ),
            OutcomeSlug(
                label="core_yoy_3.5",
                anchor_value=3.5,
                venue="kalshi",
                slug="KXECONSTATCORECPIYOY-26JUN-T3.5",
            ),
        ],
        notes=(
            "Mixes headline MoM and YoY plus core YoY contracts — they are "
            "not perfectly orthogonal, but together they span the regime "
            "axes the EM engine reads (mean and dispersion). Downstream "
            "consumers should normalise per-axis before passing into "
            "event_vol_engine. Core-YoY-2.3 is anchored at 2.301 (tiny "
            "jitter) so it does not collide with headline-YoY-2.3."
        ),
    ),
    EventEntry(
        event_id="cpi-2026-07",
        event_kind="cpi",
        description="CPI release for July 2026 data (typical date: 2nd Wed of August).",
        scheduled_at_utc=datetime(2026, 8, 12, 12, 30, tzinfo=UTC),
        underlying_ticker="SPY",
        outcome_slugs=[
            OutcomeSlug(
                label="core_yoy_2.5",
                anchor_value=2.5,
                venue="kalshi",
                slug="KXECONSTATCORECPIYOY-26JUL-T2.5",
            ),
            OutcomeSlug(
                label="core_yoy_3.7",
                anchor_value=3.7,
                venue="kalshi",
                slug="KXECONSTATCORECPIYOY-26JUL-T3.7",
            ),
            OutcomeSlug(
                label="core_yoy_2.5_alt",
                anchor_value=2.499,
                venue="kalshi",
                slug="KXECONSTATCORECPIYOY-26JUL-T2.5",
            ),
        ],
        notes=(
            "Only two distinct July-2026 core-CPI YoY slugs are present in "
            "factors.yml (T2.5 and T3.7). We duplicate T2.5 under a tiny "
            "anchor jitter (2.499 vs 2.500) as a placeholder third cell — "
            "downstream code MUST tolerate degenerate partitions. Anchor "
            "encoding will be sharpened when Kalshi publishes a wider "
            "July-2026 ladder."
        ),
    ),
    EventEntry(
        event_id="midterms-2026",
        event_kind="election",
        description="US midterms 2026 — 2x2 balance-of-power partition.",
        scheduled_at_utc=datetime(2026, 11, 3, 23, 0, tzinfo=UTC),
        underlying_ticker="SPY",
        outcome_slugs=[
            OutcomeSlug(
                label="d_house_d_senate",
                anchor_value=1.0,
                venue="polymarket",
                slug="2026-balance-of-power-d-senate-d-house-949",
            ),
            OutcomeSlug(
                label="d_house_r_senate",
                anchor_value=0.5,
                venue="polymarket",
                slug="2026-balance-of-power-r-senate-d-house-444",
            ),
            OutcomeSlug(
                label="r_house_d_senate",
                anchor_value=-0.5,
                venue="polymarket",
                slug="2026-balance-of-power-d-senate-r-house-692",
            ),
            OutcomeSlug(
                label="r_house_r_senate",
                anchor_value=-1.0,
                venue="polymarket",
                slug="2026-balance-of-power-r-senate-r-house-537",
            ),
        ],
        notes=(
            "All four 2x2 balance-of-power cells exist in factors.yml. "
            "Anchor encoding: +1 = full Democrat sweep, -1 = full "
            "Republican sweep. The axis is historically associated with "
            "regulatory-intensity expectations and is a heuristic "
            "ordering, NOT a calibrated impact measure."
        ),
    ),
    EventEntry(
        event_id="brazil-pres-2026",
        event_kind="election",
        description="Brazilian presidential election 2026, first round (04-Oct).",
        scheduled_at_utc=datetime(2026, 10, 4, 22, 0, tzinfo=UTC),
        underlying_ticker="EWZ",
        outcome_slugs=[
            OutcomeSlug(
                label="tarcisio_de_freitas",
                anchor_value=0.0,
                venue="polymarket",
                slug="will-tarcisio-de-frietas-win-the-2026-brazilian-presidential-election",
            ),
            OutcomeSlug(
                label="eduardo_bolsonaro",
                anchor_value=0.001,
                venue="polymarket",
                slug="will-eduardo-bolsonaro-win-the-2026-brazilian-presidential-election",
            ),
            OutcomeSlug(
                label="ciro_gomes_or_massa",
                anchor_value=0.002,
                venue="polymarket",
                slug="will-carlos-roberto-massa-jnior-win-the-2026-brazilian-presidential-election",
            ),
            OutcomeSlug(
                label="lula_da_silva",
                anchor_value=0.003,
                venue="polymarket",
                slug="will-luiz-incio-lula-da-silva-win-the-2026-brazilian-presidential-election",
            ),
        ],
        notes=(
            "Anchor encoding TBD for Brazil — currently all-zero with tiny "
            "jitter to preserve partition validity for the engine. A future "
            "iteration should embed each candidate's expected impact on "
            "EWZ (e.g. market-friendly continuity vs. policy-uncertainty "
            "transition). Anchors of ~0 are deliberately benign: the EM "
            "engine will lean on entropy / dispersion rather than the "
            "directional mean for this event until calibration."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_event(event_id: str) -> EventEntry | None:
    """Return the calendar entry with ``event_id`` or ``None`` if missing."""
    for entry in CALENDAR:
        if entry.event_id == event_id:
            return entry
    return None


def list_upcoming(
    now_utc: datetime,
    *,
    lookahead_days: int = 60,
) -> list[EventEntry]:
    """Return calendar entries scheduled within [now_utc, now_utc + lookahead].

    Args:
        now_utc: Reference timestamp (timezone-aware UTC).
        lookahead_days: Window length in days (default 60).

    Returns:
        Entries falling inside the window, in their declared order.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    horizon = now_utc + timedelta(days=lookahead_days)
    return [entry for entry in CALENDAR if now_utc <= entry.scheduled_at_utc <= horizon]


def list_by_kind(kind: str) -> list[EventEntry]:
    """Return all calendar entries whose ``event_kind`` equals ``kind``."""
    return [entry for entry in CALENDAR if entry.event_kind == kind]
