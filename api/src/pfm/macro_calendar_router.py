"""W12-15 — Upcoming macro events calendar (lightweight, hardcoded 2026).

This module is intentionally separate from :mod:`pfm.macro_calendar` (which
exposes a denser multi-region calendar at ``/macro/upcoming``). Here we
ship a *simpler*, US-centric, hardcoded 2026 cadence behind
``GET /macro/calendar`` with a flat, single-shape response geared for
quick dashboard use:

```
{
  "checked_at": "2026-05-16T07:00:00Z",
  "window_days": 30,
  "events": [
    {"date": "2026-05-20", "event": "FOMC Minutes",
     "category": "fed", "importance": "high"}
  ]
}
```

Recurring rules (per task spec):

* **FOMC** — 8 fixed decision dates
* **CPI** — 12-15th each month (we use the 13th as representative)
* **NFP** — first Friday of each month
* **GDP** — late month of each quarter (advance/2nd/3rd estimates)
* **PPI** — 13-16th each month (14th representative)
* **Retail Sales** — 14-17th each month (15th representative)
* **FOMC Minutes** — ~3 weeks after each FOMC

The list is computed *once at import* and filtered per request. No
network I/O; safe to call inside hot endpoints.
"""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query

CategoryT = Literal["fed", "inflation", "growth", "consumer"]
ImportanceT = Literal["low", "medium", "high"]


# --- 2026 FOMC decision dates (per task spec) -------------------------------
_FOMC_2026: list[date] = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 5, 6),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 11, 4),
    date(2026, 12, 16),
]


def _first_friday(year: int, month: int) -> date:
    """Return the first Friday of ``year``/``month``."""
    for day in range(1, 8):
        d = date(year, month, day)
        if d.weekday() == calendar.FRIDAY:
            return d
    raise RuntimeError("no Friday in first week — impossible")  # pragma: no cover


def _gdp_release_date(year: int, quarter: int) -> date:
    """Approximate GDP advance/2nd/3rd release date for a given quarter.

    Quarters report ~late in the month one quarter after the period:

    * Q4 prior year advance → late Jan (28th)
    * Q1 advance → late Apr (28th)
    * Q2 advance → late Jul (29th)
    * Q3 advance → late Oct (28th)
    """
    schedule = {1: (1, 28), 2: (4, 28), 3: (7, 29), 4: (10, 28)}
    month, day = schedule[quarter]
    return date(year, month, day)


def _build_events_2026() -> list[dict[str, Any]]:
    """Construct the 2026 event list once at import."""
    events: list[dict[str, Any]] = []

    # FOMC decisions
    for d in _FOMC_2026:
        events.append(
            {
                "date": d.isoformat(),
                "event": "FOMC Decision",
                "category": "fed",
                "importance": "high",
            }
        )
        # Minutes ~3 weeks later
        minutes_date = d + timedelta(days=21)
        events.append(
            {
                "date": minutes_date.isoformat(),
                "event": "FOMC Minutes",
                "category": "fed",
                "importance": "medium",
            }
        )

    # Monthly releases
    for month in range(1, 13):
        # CPI — 13th
        events.append(
            {
                "date": date(2026, month, 13).isoformat(),
                "event": "CPI",
                "category": "inflation",
                "importance": "high",
            }
        )
        # PPI — 14th
        events.append(
            {
                "date": date(2026, month, 14).isoformat(),
                "event": "PPI",
                "category": "inflation",
                "importance": "medium",
            }
        )
        # Retail sales — 15th
        events.append(
            {
                "date": date(2026, month, 15).isoformat(),
                "event": "Retail Sales",
                "category": "consumer",
                "importance": "medium",
            }
        )
        # NFP — first Friday
        events.append(
            {
                "date": _first_friday(2026, month).isoformat(),
                "event": "Nonfarm Payrolls",
                "category": "growth",
                "importance": "high",
            }
        )

    # GDP — quarterly
    for q in range(1, 5):
        events.append(
            {
                "date": _gdp_release_date(2026, q).isoformat(),
                "event": f"GDP Q{q} Advance Estimate",
                "category": "growth",
                "importance": "high",
            }
        )

    events.sort(key=lambda e: (e["date"], e["event"]))
    return events


_EVENTS_2026: list[dict[str, Any]] = _build_events_2026()

_VALID_CATEGORIES = {"fed", "inflation", "growth", "consumer"}
_VALID_IMPORTANCE = {"low", "medium", "high"}


def upcoming_events(
    *,
    today: date,
    window_days: int,
    category: str | None = None,
    importance: str | None = None,
) -> list[dict[str, Any]]:
    """Return events between ``today`` (inclusive) and ``today + window_days``
    (inclusive), optionally filtered by category and/or importance.

    Results are sorted ascending by date, then by event name (stable).
    """
    end = today + timedelta(days=window_days)
    out: list[dict[str, Any]] = []
    for e in _EVENTS_2026:
        d = date.fromisoformat(e["date"])
        if d < today or d > end:
            continue
        if category is not None and e["category"] != category:
            continue
        if importance is not None and e["importance"] != importance:
            continue
        out.append(dict(e))
    return out


# --- router ------------------------------------------------------------------

router = APIRouter(prefix="/macro", tags=["macro-calendar-upcoming"])


@router.get("/calendar")
def macro_calendar(
    days: int = Query(30, ge=1, le=365, description="Lookahead window in days."),
    category: str | None = Query(
        None,
        description="Optional category filter: fed, inflation, growth, consumer.",
    ),
    importance: str | None = Query(
        None, description="Optional importance filter: low, medium, high."
    ),
) -> dict[str, Any]:
    """Return upcoming hardcoded 2026 macro events within ``days`` days.

    Args:
        days: Lookahead window. ``1 <= days <= 365``.
        category: Optional filter; one of ``fed`` / ``inflation`` / ``growth``
            / ``consumer``. Invalid values return 400.
        importance: Optional filter; one of ``low`` / ``medium`` / ``high``.

    Returns:
        Dict with ``checked_at`` (UTC iso), ``window_days``, and ``events``.
    """
    if category is not None and category not in _VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid category '{category}'; expected one of {sorted(_VALID_CATEGORIES)}",
        )
    if importance is not None and importance not in _VALID_IMPORTANCE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid importance '{importance}'; expected one of {sorted(_VALID_IMPORTANCE)}"
            ),
        )

    now = datetime.now(tz=UTC)
    events = upcoming_events(
        today=now.date(),
        window_days=days,
        category=category,
        importance=importance,
    )
    return {
        "checked_at": now.isoformat().replace("+00:00", "Z"),
        "window_days": days,
        "events": events,
    }


__all__ = [
    "_EVENTS_2026",
    "_FOMC_2026",
    "router",
    "upcoming_events",
]
