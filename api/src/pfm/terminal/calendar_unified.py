"""Unified calendar — the single front-door for date-driven items.

Composes three sources into one chronologically-sorted feed:

  1. **Resolutions** of Polymarket / Kalshi prediction-market contracts
     (read from :func:`pfm.factors.load_factors`; resolution dates are
     parsed from factor IDs / segment metadata when available, with a
     conservative fallback to the curated calendar dates).
  2. **Earnings** — hardcoded calendar of high-volume US tickers for
     2026 H1/H2. ``# TODO: fetch from Yahoo / Finnhub Calendar in v0.2``.
  3. **Macro events** — FOMC, CPI, NFP, OPEC, ECB. Hardcoded for 2026.
     ``# TODO: fetch from FRED / BLS RSS in v0.2``.

Routing note: this module owns its own :class:`fastapi.APIRouter` with
prefix ``/terminal/calendar`` (alongside the existing macro-events
router). ``main.py`` only needs::

    from pfm.terminal_calendar_unified import router as terminal_calendar_unified_router
    app.include_router(terminal_calendar_unified_router)

The four pre-existing calendar routers (``terminal_event_calendar``,
``terminal_calendar_curated``, ``terminal_calendar_scanner``,
``terminal_calendar_pair``) remain wired and unchanged. This module is
**additive** — it does not replace them.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- constants --------------------------------------------------------------

CalendarKind = Literal["resolution", "earnings", "macro"]

ALL_KINDS: tuple[CalendarKind, ...] = ("resolution", "earnings", "macro")


# --- earnings (POC hardcoded) -----------------------------------------------
# TODO: fetch from Yahoo / Finnhub Calendar in v0.2.
# Each entry: (iso_date, ticker, title, importance 1-3).

_EARNINGS: tuple[tuple[str, str, str, int], ...] = (
    ("2026-04-23", "TSLA", "Tesla Q1 2026 earnings", 3),
    ("2026-04-24", "GOOGL", "Alphabet Q1 2026 earnings", 3),
    ("2026-04-29", "META", "Meta Q1 2026 earnings", 3),
    ("2026-04-30", "MSFT", "Microsoft Q1 2026 earnings", 3),
    ("2026-04-30", "AAPL", "Apple Q1 2026 earnings", 3),
    ("2026-05-01", "AMZN", "Amazon Q1 2026 earnings", 3),
    ("2026-05-07", "DIS", "Disney Q1 2026 earnings", 2),
    ("2026-05-21", "NVDA", "NVIDIA Q1 2026 earnings", 3),
    ("2026-05-22", "PANW", "Palo Alto Networks Q3 2026 earnings", 2),
    ("2026-05-29", "CRM", "Salesforce Q1 2026 earnings", 2),
    ("2026-06-04", "LULU", "Lululemon Q1 2026 earnings", 1),
    ("2026-06-25", "MU", "Micron Q3 2026 earnings", 2),
    ("2026-07-15", "JPM", "JPMorgan Q2 2026 earnings", 2),
    ("2026-07-16", "BAC", "Bank of America Q2 2026 earnings", 2),
    ("2026-07-22", "TSLA", "Tesla Q2 2026 earnings", 3),
    ("2026-07-23", "GOOGL", "Alphabet Q2 2026 earnings", 3),
    ("2026-07-28", "META", "Meta Q2 2026 earnings", 3),
    ("2026-07-29", "MSFT", "Microsoft Q2 2026 earnings", 3),
    ("2026-07-30", "AAPL", "Apple Q2 2026 earnings", 3),
    ("2026-07-30", "AMZN", "Amazon Q2 2026 earnings", 3),
    ("2026-08-19", "NVDA", "NVIDIA Q2 2026 earnings", 3),
    ("2026-09-23", "MU", "Micron Q4 2026 earnings", 2),
    ("2026-10-21", "TSLA", "Tesla Q3 2026 earnings", 3),
    ("2026-10-22", "GOOGL", "Alphabet Q3 2026 earnings", 3),
    ("2026-10-28", "META", "Meta Q3 2026 earnings", 3),
    ("2026-10-29", "MSFT", "Microsoft Q3 2026 earnings", 3),
    ("2026-10-29", "AAPL", "Apple Q3 2026 earnings", 3),
    ("2026-10-29", "AMZN", "Amazon Q3 2026 earnings", 3),
    ("2026-11-18", "NVDA", "NVIDIA Q3 2026 earnings", 3),
    ("2026-12-17", "MU", "Micron Q1 2027 earnings", 2),
)

# Theme tags for filter matching. Earnings get a "earnings" theme plus
# the ticker lower-cased so callers can ``theme=nvda`` and find it.
_EARNINGS_THEMES: tuple[str, ...] = ("earnings", "equity")


# --- macro (POC hardcoded) --------------------------------------------------
# TODO: fetch from FRED / BLS RSS / ECB calendar in v0.2.
# Each entry: (iso_date, kind_label, title, importance 1-3, themes).

_MACRO: tuple[tuple[str, str, str, int, tuple[str, ...]], ...] = (
    # ── FOMC 2026 ─────────────────────────────────────────────────────────
    ("2026-01-28", "FOMC", "FOMC rate decision (Jan)", 3, ("rates", "fed", "macro")),
    ("2026-03-18", "FOMC", "FOMC rate decision + SEP (Mar)", 3, ("rates", "fed", "macro")),
    ("2026-04-29", "FOMC", "FOMC rate decision (Apr)", 3, ("rates", "fed", "macro")),
    ("2026-06-17", "FOMC", "FOMC rate decision + SEP (Jun)", 3, ("rates", "fed", "macro")),
    ("2026-07-29", "FOMC", "FOMC rate decision (Jul)", 3, ("rates", "fed", "macro")),
    ("2026-09-16", "FOMC", "FOMC rate decision + SEP (Sep)", 3, ("rates", "fed", "macro")),
    ("2026-11-04", "FOMC", "FOMC rate decision (Nov)", 3, ("rates", "fed", "macro")),
    ("2026-12-16", "FOMC", "FOMC rate decision + SEP (Dec)", 3, ("rates", "fed", "macro")),
    # ── CPI 2026 ──────────────────────────────────────────────────────────
    ("2026-01-13", "CPI", "CPI release (Dec 2025 print)", 3, ("inflation", "macro")),
    ("2026-02-11", "CPI", "CPI release (Jan 2026 print)", 3, ("inflation", "macro")),
    ("2026-03-12", "CPI", "CPI release (Feb 2026 print)", 3, ("inflation", "macro")),
    ("2026-04-14", "CPI", "CPI release (Mar 2026 print)", 3, ("inflation", "macro")),
    ("2026-05-13", "CPI", "CPI release (Apr 2026 print)", 3, ("inflation", "macro")),
    ("2026-06-11", "CPI", "CPI release (May 2026 print)", 3, ("inflation", "macro")),
    ("2026-07-15", "CPI", "CPI release (Jun 2026 print)", 3, ("inflation", "macro")),
    ("2026-08-12", "CPI", "CPI release (Jul 2026 print)", 3, ("inflation", "macro")),
    ("2026-09-10", "CPI", "CPI release (Aug 2026 print)", 3, ("inflation", "macro")),
    ("2026-10-15", "CPI", "CPI release (Sep 2026 print)", 3, ("inflation", "macro")),
    ("2026-11-12", "CPI", "CPI release (Oct 2026 print)", 3, ("inflation", "macro")),
    ("2026-12-10", "CPI", "CPI release (Nov 2026 print)", 3, ("inflation", "macro")),
    # ── NFP 2026 ──────────────────────────────────────────────────────────
    ("2026-01-09", "NFP", "Non-farm payrolls (Dec 2025)", 3, ("labor", "macro")),
    ("2026-02-06", "NFP", "Non-farm payrolls (Jan 2026)", 3, ("labor", "macro")),
    ("2026-03-06", "NFP", "Non-farm payrolls (Feb 2026)", 3, ("labor", "macro")),
    ("2026-04-03", "NFP", "Non-farm payrolls (Mar 2026)", 3, ("labor", "macro")),
    ("2026-05-01", "NFP", "Non-farm payrolls (Apr 2026)", 3, ("labor", "macro")),
    ("2026-06-05", "NFP", "Non-farm payrolls (May 2026)", 3, ("labor", "macro")),
    ("2026-07-02", "NFP", "Non-farm payrolls (Jun 2026)", 3, ("labor", "macro")),
    ("2026-08-07", "NFP", "Non-farm payrolls (Jul 2026)", 3, ("labor", "macro")),
    ("2026-09-04", "NFP", "Non-farm payrolls (Aug 2026)", 3, ("labor", "macro")),
    ("2026-10-02", "NFP", "Non-farm payrolls (Sep 2026)", 3, ("labor", "macro")),
    ("2026-11-06", "NFP", "Non-farm payrolls (Oct 2026)", 3, ("labor", "macro")),
    ("2026-12-04", "NFP", "Non-farm payrolls (Nov 2026)", 3, ("labor", "macro")),
    # ── OPEC 2026 ─────────────────────────────────────────────────────────
    ("2026-03-05", "OPEC", "OPEC+ JMMC meeting", 2, ("oil", "commodities", "macro")),
    ("2026-06-01", "OPEC", "OPEC+ ministerial meeting", 2, ("oil", "commodities", "macro")),
    ("2026-09-07", "OPEC", "OPEC+ JMMC meeting", 2, ("oil", "commodities", "macro")),
    ("2026-12-01", "OPEC", "OPEC+ ministerial meeting", 2, ("oil", "commodities", "macro")),
    # ── ECB 2026 ──────────────────────────────────────────────────────────
    ("2026-01-22", "ECB", "ECB rate decision (Jan)", 3, ("rates", "ecb", "eur", "macro")),
    (
        "2026-03-12",
        "ECB",
        "ECB rate decision + projections (Mar)",
        3,
        ("rates", "ecb", "eur", "macro"),
    ),
    ("2026-04-16", "ECB", "ECB rate decision (Apr)", 3, ("rates", "ecb", "eur", "macro")),
    (
        "2026-06-04",
        "ECB",
        "ECB rate decision + projections (Jun)",
        3,
        ("rates", "ecb", "eur", "macro"),
    ),
    ("2026-07-23", "ECB", "ECB rate decision (Jul)", 3, ("rates", "ecb", "eur", "macro")),
    (
        "2026-09-10",
        "ECB",
        "ECB rate decision + projections (Sep)",
        3,
        ("rates", "ecb", "eur", "macro"),
    ),
    ("2026-10-29", "ECB", "ECB rate decision (Oct)", 3, ("rates", "ecb", "eur", "macro")),
    (
        "2026-12-17",
        "ECB",
        "ECB rate decision + projections (Dec)",
        3,
        ("rates", "ecb", "eur", "macro"),
    ),
)


# --- resolution sources -----------------------------------------------------
# Pulled from the curated calendar clusters in
# ``pfm.terminal_calendar_curated`` so resolution dates stay
# consistent with the rest of the terminal. We import lazily inside
# the function so this module stays cheap to import.


def _load_resolution_items() -> list[tuple[str, str, str, str, int, tuple[str, ...]]]:
    """Return resolution rows ``(iso_date, slug, title, factor_id, importance, themes)``.

    Pulls from the curated cluster catalog. If that import fails for
    any reason (e.g. test environment without ``factors.yml``), we
    return an empty list — the unified endpoint then degrades to
    earnings + macro only.
    """
    try:
        from pfm.terminal_calendar_curated import (  # local: avoid cycles
            _CURATED_CLUSTERS,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.info("curated-clusters import failed: %s", e)
        return []

    out: list[tuple[str, str, str, str, int, tuple[str, ...]]] = []
    for cluster in _CURATED_CLUSTERS:
        for leg in cluster.legs:
            iso = leg.deadline.isoformat()
            title = f"Resolution · {cluster.title} · {leg.factor_id}"
            # Theme tagging derived from cluster_id keywords so a
            # ``theme=fed`` filter picks up Fed-related markets, etc.
            themes = tuple(
                t
                for t in {
                    "resolution",
                    cluster.cluster_id.split("_")[0],
                    *cluster.cluster_id.split("_"),
                }
                if t
            )
            out.append((iso, leg.factor_id, title, leg.factor_id, 2, themes))
    return out


# --- schemas ----------------------------------------------------------------


class CalendarItem(BaseModel):
    """One row in the unified calendar feed."""

    date: str = Field(..., description="ISO-8601 date (YYYY-MM-DD).")
    kind: CalendarKind
    title: str
    slug: str | None = Field(
        None, description="Market slug for ``kind=resolution``; null otherwise."
    )
    ticker: str | None = Field(
        None, description="Equity ticker for ``kind=earnings``; null otherwise."
    )
    importance: int = Field(2, ge=1, le=3)
    detail: str | None = None


class UnifiedCalendarResponse(BaseModel):
    start: str
    end: str
    items: list[CalendarItem]
    total: int


# --- helpers ----------------------------------------------------------------


def _parse_iso_date(s: str, *, name: str) -> date:
    """Parse an ISO-8601 date or raise an HTTP-422-friendly exception."""
    try:
        return datetime.fromisoformat(s).date()
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=422, detail=f"{name} must be ISO-8601 (YYYY-MM-DD), got {s!r}"
        ) from e


def _parse_kinds(kinds: str | None) -> tuple[CalendarKind, ...]:
    """Parse a comma-separated ``kinds`` query into a tuple of valid kinds."""
    if not kinds:
        return ALL_KINDS
    parts = [k.strip().lower() for k in kinds.split(",") if k.strip()]
    bad = [k for k in parts if k not in ALL_KINDS]
    if bad:
        raise HTTPException(
            status_code=422,
            detail=f"unknown kinds: {bad!r}; allowed: {list(ALL_KINDS)!r}",
        )
    # de-dup while preserving order
    seen: set[str] = set()
    ordered: list[CalendarKind] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            ordered.append(p)  # type: ignore[arg-type]
    return tuple(ordered)


def _theme_match(theme: str, item_themes: tuple[str, ...]) -> bool:
    """Case-insensitive substring match on any of ``item_themes``."""
    needle = theme.strip().lower()
    if not needle:
        return True
    return any(needle in t.lower() for t in item_themes)


def _build_items(
    start: date,
    end: date,
    kinds: tuple[CalendarKind, ...],
    theme: str | None,
) -> list[CalendarItem]:
    """Compose the three sources, then filter by date/kind/theme."""
    items: list[CalendarItem] = []

    if "resolution" in kinds:
        for iso, slug, title, factor_id, importance, themes in _load_resolution_items():
            d = _parse_iso_date(iso, name="resolution.date")
            if d < start or d > end:
                continue
            if theme and not _theme_match(theme, themes):
                continue
            items.append(
                CalendarItem(
                    date=iso,
                    kind="resolution",
                    title=title,
                    slug=slug,
                    ticker=None,
                    importance=importance,
                    detail=f"factor_id={factor_id}",
                )
            )

    if "earnings" in kinds:
        for iso, ticker, title, importance in _EARNINGS:
            d = _parse_iso_date(iso, name="earnings.date")
            if d < start or d > end:
                continue
            themes = (*_EARNINGS_THEMES, ticker.lower())
            if theme and not _theme_match(theme, themes):
                continue
            items.append(
                CalendarItem(
                    date=iso,
                    kind="earnings",
                    title=title,
                    slug=None,
                    ticker=ticker,
                    importance=importance,
                    detail=None,
                )
            )

    if "macro" in kinds:
        for iso, kind_label, title, importance, themes in _MACRO:
            d = _parse_iso_date(iso, name="macro.date")
            if d < start or d > end:
                continue
            if theme and not _theme_match(theme, themes):
                continue
            items.append(
                CalendarItem(
                    date=iso,
                    kind="macro",
                    title=title,
                    slug=None,
                    ticker=None,
                    importance=importance,
                    detail=kind_label,
                )
            )

    items.sort(key=lambda x: (x.date, x.kind, x.title))
    return items


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-calendar-unified"])


@router.get("/calendar", response_model=UnifiedCalendarResponse)
def unified_calendar(
    start: str | None = Query(
        None,
        description=(
            "Inclusive start date, ISO-8601. Defaults to *today* when omitted "
            "— combined with the default 7-day window this makes the bare "
            "``/terminal/calendar`` URL a useful one-shot."
        ),
    ),
    end: str | None = Query(
        None,
        description=("Inclusive end date, ISO-8601. Defaults to ``start + 7 days`` when omitted."),
    ),
    kinds: str | None = Query(
        None,
        description=(
            "Comma-separated subset of ``resolution,earnings,macro``. "
            "Defaults to all three when omitted."
        ),
    ),
    theme: str | None = Query(
        None,
        description=(
            "Optional case-insensitive substring filter applied to each "
            "item's theme tags (rates, fed, inflation, oil, NVDA, …)."
        ),
    ),
) -> UnifiedCalendarResponse:
    """Return a chronologically-sorted, multi-source calendar slice.

    Default window when neither ``start`` nor ``end`` is provided: today
    through 7 days out. Lets the UI fetch `/terminal/calendar` without
    having to thread date math through every caller.
    """
    from datetime import date as _date
    from datetime import timedelta as _td

    today = _date.today()
    if not start:
        start = today.isoformat()
    if not end:
        start_default = _parse_iso_date(start, name="start")
        end = (start_default + _td(days=7)).isoformat()
    start_d = _parse_iso_date(start, name="start")
    end_d = _parse_iso_date(end, name="end")
    if start_d > end_d:
        raise HTTPException(status_code=422, detail=f"start ({start}) must be <= end ({end})")
    selected_kinds = _parse_kinds(kinds)
    items = _build_items(start_d, end_d, selected_kinds, theme)
    return UnifiedCalendarResponse(
        start=start_d.isoformat(),
        end=end_d.isoformat(),
        items=items,
        total=len(items),
    )


__all__ = [
    "ALL_KINDS",
    "CalendarItem",
    "CalendarKind",
    "UnifiedCalendarResponse",
    "router",
]
