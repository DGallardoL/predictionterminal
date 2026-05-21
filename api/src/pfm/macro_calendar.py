"""Macro events calendar (dense, multi-region).

Hardcoded calendar of macro releases for 2026 H1 + H2.

Original (US-only) coverage — kept as the canonical "core six" for
backward compatibility:

* **FOMC** — 8 decision days (Federal Reserve)
* **CPI** — 12 monthly releases (BLS)
* **Nonfarm Payrolls** — 12 monthly releases (BLS, first Friday)
* **PPI** — 12 monthly releases (BLS)
* **Retail Sales** — 12 monthly releases (Census Bureau)
* **GDP** — 4 quarterly advance estimates (BEA)

Wave-10 dense expansion (2026 multi-region calendar):

* **Jobless Claims** — 52 weekly releases (BLS, every Thursday)
* **ECB** — 8 governing-council rate decisions
* **BoJ** — 8 monetary-policy meetings
* **OPEC** — 12 monthly MOMR + JMMC datapoints
* **CPI Eurozone** — 12 monthly flash releases (Eurostat)
* **CPI Japan** — 12 monthly nationwide releases (StatJapan)
* **China NBS PMI** — 12 monthly composite PMI prints (NBS)

Goal: ~150 events across 2026 with per-event ``importance`` (1=low,
2=medium, 3=major market mover) and ``region`` for client-side filtering.

Backward compatibility
----------------------
The originals (``_FOMC_2026``, ``_CPI_2026``, ``_NFP_2026``, ``_PPI_2026``,
``_RETAIL_SALES_2026``, ``_GDP_2026``) and ``_ALL_EVENTS`` keep their
historical shapes and counts — pre-existing tests assert exact lengths.
New events live in ``_ALL_EVENTS_DENSE`` (superset). The endpoint and
:func:`next_releases` use ``_ALL_EVENTS_DENSE`` so the public surface is
denser without breaking the old constants.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from pfm.cache_utils import get_cache

# Cache TTL — 12h. Calendar changes rarely; daily refresh is overkill.
_CAL_TTL_SECONDS = 12 * 3600


# --- raw 2026 schedule (US core six — UNCHANGED) ---------------------------

# FOMC two-day meetings — second-day is the decision/SEP press conference.
_FOMC_2026: list[date] = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]

_CPI_2026: list[date] = [
    date(2026, 1, 14),
    date(2026, 2, 11),
    date(2026, 3, 11),
    date(2026, 4, 14),
    date(2026, 5, 13),
    date(2026, 6, 10),
    date(2026, 7, 15),
    date(2026, 8, 12),
    date(2026, 9, 11),
    date(2026, 10, 14),
    date(2026, 11, 13),
    date(2026, 12, 10),
]

# Nonfarm Payrolls — first Friday of each month.
_NFP_2026: list[date] = [
    date(2026, 1, 2),
    date(2026, 2, 6),
    date(2026, 3, 6),
    date(2026, 4, 3),
    date(2026, 5, 1),
    date(2026, 6, 5),
    date(2026, 7, 3),
    date(2026, 8, 7),
    date(2026, 9, 4),
    date(2026, 10, 2),
    date(2026, 11, 6),
    date(2026, 12, 4),
]

_PPI_2026: list[date] = [
    date(2026, 1, 15),
    date(2026, 2, 12),
    date(2026, 3, 12),
    date(2026, 4, 15),
    date(2026, 5, 14),
    date(2026, 6, 11),
    date(2026, 7, 16),
    date(2026, 8, 13),
    date(2026, 9, 10),
    date(2026, 10, 15),
    date(2026, 11, 12),
    date(2026, 12, 11),
]

_RETAIL_SALES_2026: list[date] = [
    date(2026, 1, 16),
    date(2026, 2, 17),
    date(2026, 3, 17),
    date(2026, 4, 16),
    date(2026, 5, 15),
    date(2026, 6, 16),
    date(2026, 7, 17),
    date(2026, 8, 14),
    date(2026, 9, 16),
    date(2026, 10, 16),
    date(2026, 11, 17),
    date(2026, 12, 16),
]

_GDP_2026: list[date] = [
    date(2026, 1, 29),
    date(2026, 4, 29),
    date(2026, 7, 30),
    date(2026, 10, 29),
]


# --- Wave-10 dense schedule -------------------------------------------------


# Jobless Claims — every Thursday in 2026 (BLS schedule). Jan 1 is New
# Year's Day so the first release of 2026 lands on Thursday Jan 8.
def _all_thursdays_2026() -> list[date]:
    """Return every Thursday in calendar year 2026, skipping Jan 1 (holiday)."""
    d = date(2026, 1, 1)
    while d.weekday() != 3:
        d += timedelta(days=1)
    out: list[date] = []
    while d.year == 2026:
        out.append(d)
        d += timedelta(days=7)
    # Drop Jan 1 (federal holiday) — release shifts to next Thursday.
    return [x for x in out if x != date(2026, 1, 1)]


_JOBLESS_CLAIMS_2026: list[date] = _all_thursdays_2026()

# ECB Governing Council rate decisions — 8/year, official ECB calendar.
_ECB_2026: list[date] = [
    date(2026, 1, 22),
    date(2026, 3, 12),
    date(2026, 4, 16),
    date(2026, 6, 4),
    date(2026, 7, 23),
    date(2026, 9, 10),
    date(2026, 10, 29),
    date(2026, 12, 17),
]

# Bank of Japan Monetary Policy Meetings — 8/year.
_BOJ_2026: list[date] = [
    date(2026, 1, 23),
    date(2026, 3, 19),
    date(2026, 4, 28),
    date(2026, 6, 19),
    date(2026, 7, 31),
    date(2026, 9, 18),
    date(2026, 10, 30),
    date(2026, 12, 18),
]

# OPEC — Monthly Oil Market Report (MOMR), mid-month, 12/year. JMMC + OPEC+
# meetings are on a different cadence; we treat MOMR as the headline
# monthly print (one event per month).
_OPEC_2026: list[date] = [
    date(2026, 1, 14),
    date(2026, 2, 12),
    date(2026, 3, 12),
    date(2026, 4, 14),
    date(2026, 5, 13),
    date(2026, 6, 11),
    date(2026, 7, 14),
    date(2026, 8, 12),
    date(2026, 9, 10),
    date(2026, 10, 13),
    date(2026, 11, 12),
    date(2026, 12, 10),
]

# Eurostat flash CPI — late month or early next month. 12/year.
_CPI_EUROZONE_2026: list[date] = [
    date(2026, 1, 30),
    date(2026, 2, 27),
    date(2026, 3, 31),
    date(2026, 4, 30),
    date(2026, 5, 29),
    date(2026, 6, 30),
    date(2026, 7, 31),
    date(2026, 8, 31),
    date(2026, 9, 30),
    date(2026, 10, 30),
    date(2026, 11, 30),
    date(2026, 12, 30),
]

# Japan nationwide CPI — Statistics Bureau, late-month for prior month.
_CPI_JAPAN_2026: list[date] = [
    date(2026, 1, 23),
    date(2026, 2, 20),
    date(2026, 3, 20),
    date(2026, 4, 24),
    date(2026, 5, 22),
    date(2026, 6, 19),
    date(2026, 7, 24),
    date(2026, 8, 21),
    date(2026, 9, 18),
    date(2026, 10, 23),
    date(2026, 11, 20),
    date(2026, 12, 25),
]

# China NBS Manufacturing + Non-Manufacturing PMI — last day of month.
_CHINA_PMI_2026: list[date] = [
    date(2026, 1, 31),
    date(2026, 2, 28),
    date(2026, 3, 31),
    date(2026, 4, 30),
    date(2026, 5, 31),
    date(2026, 6, 30),
    date(2026, 7, 31),
    date(2026, 8, 31),
    date(2026, 9, 30),
    date(2026, 10, 31),
    date(2026, 11, 30),
    date(2026, 12, 31),
]


# --- event materialisation --------------------------------------------------

# (kind, title, importance 1-3, region, source, expected_impact, dates)
# ``impact`` (legacy) is derived from ``importance``: 3 -> high, 2 -> medium, 1 -> low.
_IMPORTANCE_TO_IMPACT = {3: "high", 2: "medium", 1: "low"}


def _emit(
    dates: list[date],
    *,
    kind: str,
    title: str,
    importance: int,
    region: str,
    source: str,
    expected_impact: str,
) -> list[dict[str, Any]]:
    """Materialise one row per date with both legacy + dense keys."""
    impact = _IMPORTANCE_TO_IMPACT.get(importance, "low")
    return [
        {
            # Legacy keys (kept verbatim to preserve old test contracts).
            "date": d.isoformat(),
            "event": title,
            "type": kind,
            "source": source,
            "impact": impact,
            # Dense / Wave-10 keys.
            "kind": kind,
            "title": title,
            "importance": importance,
            "region": region,
            "expected_impact": expected_impact,
        }
        for d in dates
    ]


def _build_event_list_core() -> list[dict[str, Any]]:
    """The original six-source aggregation — UNCHANGED count = 60."""
    events: list[dict[str, Any]] = []
    events += _emit(
        _FOMC_2026,
        kind="fomc",
        title="FOMC Decision",
        importance=3,
        region="US",
        source="Federal Reserve",
        expected_impact="USD volatility, rate-sensitive equity sectors, front-end Treasuries",
    )
    events += _emit(
        _CPI_2026,
        kind="cpi",
        title="CPI Release",
        importance=3,
        region="US",
        source="BLS",
        expected_impact="USD, breakevens, rate-sensitive sectors",
    )
    events += _emit(
        _NFP_2026,
        kind="nfp",
        title="Nonfarm Payrolls",
        importance=3,
        region="US",
        source="BLS",
        expected_impact="USD, front-end yields, cyclical equities",
    )
    events += _emit(
        _PPI_2026,
        kind="ppi",
        title="PPI Release",
        importance=2,
        region="US",
        source="BLS",
        expected_impact="Inflation read-through ahead of CPI",
    )
    events += _emit(
        _RETAIL_SALES_2026,
        kind="retail_sales",
        title="Retail Sales",
        importance=2,
        region="US",
        source="Census Bureau",
        expected_impact="Consumer-discretionary equities, USD",
    )
    events += _emit(
        _GDP_2026,
        kind="gdp",
        title="GDP Advance Estimate",
        importance=3,
        region="US",
        source="BEA",
        expected_impact="USD, equity beta, growth/value rotation",
    )
    events.sort(key=lambda e: (e["date"], e["type"]))
    return events


def _build_event_list_dense() -> list[dict[str, Any]]:
    """Core six + Wave-10 dense additions. Total goal ~150 events."""
    events: list[dict[str, Any]] = list(_build_event_list_core())
    events += _emit(
        _JOBLESS_CLAIMS_2026,
        kind="jobless_claims",
        title="Initial Jobless Claims",
        importance=1,
        region="US",
        source="BLS",
        expected_impact="Front-end yields on surprises >25k vs. consensus",
    )
    events += _emit(
        _ECB_2026,
        kind="ecb",
        title="ECB Rate Decision",
        importance=3,
        region="EU",
        source="European Central Bank",
        expected_impact="EUR, Bund yields, European banks",
    )
    events += _emit(
        _BOJ_2026,
        kind="boj",
        title="BoJ Monetary Policy",
        importance=3,
        region="JP",
        source="Bank of Japan",
        expected_impact="JPY crosses, JGB curve, Nikkei",
    )
    events += _emit(
        _OPEC_2026,
        kind="opec",
        title="OPEC Monthly Oil Market Report",
        importance=2,
        region="GLOBAL",
        source="OPEC Secretariat",
        expected_impact="Brent / WTI, energy equities",
    )
    events += _emit(
        _CPI_EUROZONE_2026,
        kind="cpi_eurozone",
        title="Eurozone CPI Flash",
        importance=2,
        region="EU",
        source="Eurostat",
        expected_impact="EUR, Bund breakevens",
    )
    events += _emit(
        _CPI_JAPAN_2026,
        kind="cpi_japan",
        title="Japan CPI",
        importance=2,
        region="JP",
        source="Statistics Bureau of Japan",
        expected_impact="JPY, BoJ pivot expectations",
    )
    events += _emit(
        _CHINA_PMI_2026,
        kind="china_pmi",
        title="China NBS Manufacturing PMI",
        importance=2,
        region="CN",
        source="National Bureau of Statistics of China",
        expected_impact="CNH, AUD, copper, China-sensitive equities",
    )
    events.sort(key=lambda e: (e["date"], e["type"]))
    return events


# Materialise once at import — lists never mutate.
# ``_ALL_EVENTS`` is the legacy 60-row core (kept for backward-compat tests).
_ALL_EVENTS: list[dict[str, Any]] = _build_event_list_core()
# ``_ALL_EVENTS_DENSE`` is the public, dense superset used at runtime.
_ALL_EVENTS_DENSE: list[dict[str, Any]] = _build_event_list_dense()


def next_releases(
    days_ahead: int = 30,
    *,
    today: date | None = None,
    kind: str | None = None,
    importance_min: int | None = None,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Return upcoming macro events within ``days_ahead`` days, optionally filtered.

    Args:
        days_ahead: lookahead window in days (default 30).
        today: anchor date — defaults to UTC today. Useful in tests.
        kind: keep only events with this ``kind`` (e.g. ``"fomc"``).
        importance_min: drop events with ``importance < importance_min``.
        region: keep only events in this region (case-insensitive,
            e.g. ``"US"``, ``"EU"``, ``"JP"``, ``"CN"``, ``"GLOBAL"``).

    Returns:
        List of dicts with both legacy keys (``event``, ``type``, ``impact``)
        and dense Wave-10 keys (``kind``, ``title``, ``importance``,
        ``region``, ``expected_impact``), plus ``days_until``.
    """
    if today is None:
        today = datetime.now(tz=UTC).date()
    days_ahead = max(days_ahead, 0)
    horizon = today + timedelta(days=days_ahead)
    region_norm = region.upper() if region else None
    out: list[dict[str, Any]] = []
    for ev in _ALL_EVENTS_DENSE:
        d = date.fromisoformat(ev["date"])
        if d < today or d > horizon:
            continue
        if kind is not None and ev.get("kind") != kind:
            continue
        if importance_min is not None and int(ev.get("importance", 0)) < importance_min:
            continue
        if region_norm is not None and str(ev.get("region", "")).upper() != region_norm:
            continue
        out.append({**ev, "days_until": (d - today).days})
    return out


# --- iCalendar export -------------------------------------------------------


def _ics_escape(text: str) -> str:
    """Escape commas / semicolons / newlines per RFC 5545 §3.3.11."""
    return text.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")


def render_ics(events: list[dict[str, Any]]) -> str:
    """Render an iCalendar VCALENDAR with one all-day VEVENT per event."""
    lines: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//pfm//macro-calendar//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    for ev in events:
        d = date.fromisoformat(ev["date"])
        dt_start = d.strftime("%Y%m%d")
        dt_end = (d + timedelta(days=1)).strftime("%Y%m%d")
        title = _ics_escape(str(ev.get("title") or ev.get("event") or ev.get("kind", "Event")))
        kind = str(ev.get("kind") or ev.get("type", "event"))
        region = str(ev.get("region", ""))
        importance = int(ev.get("importance", 1))
        descr = _ics_escape(
            f"kind={kind} importance={importance} region={region} source={ev.get('source', '')}"
        )
        uid = f"{ev['date']}-{kind}@pfm-macro-calendar"
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTART;VALUE=DATE:{dt_start}",
            f"DTEND;VALUE=DATE:{dt_end}",
            f"SUMMARY:{title}",
            f"DESCRIPTION:{descr}",
            f"CATEGORIES:{kind}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/macro", tags=["macro-calendar"])


@router.get("/upcoming")
def macro_upcoming(
    days: int = Query(30, ge=1, le=365, description="Lookahead window in days"),
    kind: str | None = Query(None, description="Keep only this kind (e.g. fomc, cpi)."),
    importance_min: int | None = Query(
        None, ge=1, le=3, description="Drop events with importance < this (1=low,3=high)."
    ),
    region: str | None = Query(None, description="Region filter: US, EU, JP, CN, GLOBAL."),
) -> dict[str, Any]:
    """Return the upcoming macro events within ``days`` days.

    Cached for 12h since the schedule rarely changes.
    """
    cache = get_cache("macro-calendar", ttl=_CAL_TTL_SECONDS)
    today = datetime.now(tz=UTC).date()
    cache_key = (today.isoformat(), int(days), kind, importance_min, region)
    hit = cache.get(cache_key)
    if hit is not None:
        return hit
    events = next_releases(
        days, today=today, kind=kind, importance_min=importance_min, region=region
    )
    payload = {
        "as_of": today.isoformat(),
        "days_ahead": days,
        "filters": {"kind": kind, "importance_min": importance_min, "region": region},
        "count": len(events),
        "events": events,
    }
    cache.set(cache_key, payload, ttl=_CAL_TTL_SECONDS)
    return payload


@router.get(
    "/calendar/export.ics",
    response_class=PlainTextResponse,
    summary="iCalendar export of the macro calendar (Google Calendar friendly).",
)
def macro_calendar_ics(
    days: int = Query(365, ge=1, le=365),
    kind: str | None = Query(None),
    importance_min: int | None = Query(None, ge=1, le=3),
    region: str | None = Query(None),
) -> PlainTextResponse:
    """Return all matching events as a ``text/calendar`` ICS file."""
    today = datetime.now(tz=UTC).date()
    events = next_releases(
        days, today=today, kind=kind, importance_min=importance_min, region=region
    )
    body = render_ics(events)
    return PlainTextResponse(
        body,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=pfm-macro-calendar.ics"},
    )


__all__ = [
    "next_releases",
    "render_ics",
    "router",
]
