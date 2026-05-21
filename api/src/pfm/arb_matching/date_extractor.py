"""Resolution-window extraction for prediction-market titles.

Motivation
----------
The cross-venue arbitrage scanner used naive title overlap to pair Kalshi
and Polymarket markets. That produced false positives such as pairing
"Will Trump win the 2024 election?" with "Will Trump win the 2028
election?" — same actor, *very different* resolution window.

This module exposes :class:`ResolutionWindow` plus two functions:

- :func:`extract_resolution_window` — parse a free-form title/description
  into an ``earliest..latest`` date window with a confidence score in
  ``[0, 1]`` and the substring that drove the inference.
- :func:`windows_overlap` — return ``True`` iff two windows overlap by
  at least ``min_overlap_days``.

Design choices
--------------
- **Structured pattern table** instead of a regex shotgun. Each pattern
  carries the regex, a parser callback, and a confidence score. New
  patterns are added by appending to ``_PATTERNS``.
- **stdlib + dateutil.parser** only. ``dateutil`` is already a dep.
- **Conservative defaults**: if nothing matches, return all-None and
  ``confidence=0.0`` — never invent dates.
- **Political calendar** is intentionally small (2024/2026/2028 US
  elections + midterms) because that's what shows up in the
  Kalshi/Polymarket pairs we see today. Confidence is capped at 0.4 for
  these relative phrases to reflect the assumption.

The dates returned are always Python :class:`datetime.date` objects
(timezone-free); callers that need timezone semantics should treat them
as UTC dates (consistent with the rest of the codebase, ADR-0006).
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from dateutil import parser as dateutil_parser

__all__ = [
    "ResolutionWindow",
    "extract_resolution_window",
    "windows_overlap",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionWindow:
    """Inferred resolution window for a prediction market.

    Attributes
    ----------
    earliest:
        Earliest plausible resolution date, or ``None`` if not inferable.
    latest:
        Latest plausible resolution date, or ``None`` if not inferable.
    confidence:
        Heuristic score in ``[0, 1]`` reflecting how strongly the source
        text supports the window. Explicit ISO/full-date wins are ~0.95;
        year-only is ~0.7; quarter is ~0.8; "by next election" is ~0.3.
    source_text:
        The substring of the input that drove the inference. Empty
        string when no match.
    """

    earliest: date | None
    latest: date | None
    confidence: float
    source_text: str

    def __post_init__(self) -> None:
        # Defensive invariant: latest >= earliest if both set.
        if self.earliest is not None and self.latest is not None and self.latest < self.earliest:
            raise ValueError(
                "ResolutionWindow.latest must be >= earliest "
                f"(got earliest={self.earliest}, latest={self.latest})"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ResolutionWindow.confidence must be in [0, 1] (got {self.confidence})"
            )

    # ------------------------------------------------------------------
    # Window-shape predicates used by ``windows_overlap`` (T76b).
    # ------------------------------------------------------------------

    @property
    def is_point_date(self) -> bool:
        """``True`` iff this window resolves on a *single* day.

        Used by :func:`windows_overlap` to enforce strict latest-date
        proximity for point-events (elections, Super Bowls, FOMC meetings),
        which would otherwise spuriously overlap with similarly-shaped
        windows years away.
        """

        return (
            self.earliest is not None and self.latest is not None and self.earliest == self.latest
        )

    @property
    def is_half_open_by(self) -> bool:
        """``True`` iff this window is of shape ``(None, latest)``.

        Markets parsed from ``"by DATE"`` / ``"before DATE"`` / ``"EOY YEAR"``
        text produce this shape — they assert an upper bound but no lower
        bound. Two such windows are treated as bounded *events* near their
        ``latest`` for overlap purposes (see :func:`windows_overlap`).
        """

        return self.earliest is None and self.latest is not None

    @property
    def is_half_open_from(self) -> bool:
        """``True`` iff this window is of shape ``(earliest, None)``.

        The mirror of :attr:`is_half_open_by`: a lower bound with no upper
        bound. Rare in prediction-market titles but kept symmetric so the
        overlap logic is uniform.
        """

        return self.earliest is not None and self.latest is None


_EMPTY = ResolutionWindow(earliest=None, latest=None, confidence=0.0, source_text="")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_MONTH_ALT = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


def _eom(year: int, month: int) -> int:
    """Return the last day-of-month for ``(year, month)``."""

    return calendar.monthrange(year, month)[1]


def _end_of_quarter(year: int, q: int) -> date:
    if q == 1:
        return date(year, 3, 31)
    if q == 2:
        return date(year, 6, 30)
    if q == 3:
        return date(year, 9, 30)
    if q == 4:
        return date(year, 12, 31)
    raise ValueError(f"invalid quarter {q}")


def _start_of_quarter(year: int, q: int) -> date:
    return date(year, {1: 1, 2: 4, 3: 7, 4: 10}[q], 1)


# Relative political-calendar anchors. Confidence is intentionally low —
# these are best-effort fallbacks for phrases like "by the next election".
_POLITICAL_ANCHORS: dict[str, tuple[date, date]] = {
    # 2024 presidential election: Nov 5, 2024.
    "2024 election": (date(2024, 11, 5), date(2024, 11, 5)),
    # 2026 US midterms: first Tuesday after first Monday of November.
    "2026 midterms": (date(2026, 11, 3), date(2026, 11, 3)),
    "midterms 2026": (date(2026, 11, 3), date(2026, 11, 3)),
    # 2028 presidential election: Nov 7, 2028.
    "2028 election": (date(2028, 11, 7), date(2028, 11, 7)),
    # 2030 midterms: Nov 5, 2030.
    "2030 midterms": (date(2030, 11, 5), date(2030, 11, 5)),
}


def _next_political_event(phrase: str, reference: date) -> tuple[date, date] | None:
    """Return the (earliest, latest) for "next election" / "next midterm"."""

    cleaned = phrase.lower().strip()
    if "midterm" in cleaned:
        # Midterms: 2022, 2026, 2030, ...
        for y in (2022, 2026, 2030, 2034):
            if date(y, 11, 5) >= reference:
                return (date(y, 11, 1), date(y, 11, 30))
        return None
    if "election" in cleaned:
        # Presidential elections: 2024, 2028, 2032, ...
        for y in (2024, 2028, 2032, 2036):
            if date(y, 11, 5) >= reference:
                return (date(y, 11, 1), date(y, 11, 30))
        return None
    return None


# ---------------------------------------------------------------------------
# Pattern handlers
# ---------------------------------------------------------------------------

PatternResult = ResolutionWindow | None
PatternFn = Callable[[re.Match[str], date], PatternResult]


def _h_iso(m: re.Match[str], _ref: date) -> PatternResult:
    """Match ISO ``YYYY-MM-DD``."""

    y, mo, d = int(m["y"]), int(m["mo"]), int(m["d"])
    try:
        target = date(y, mo, d)
    except ValueError:
        return None
    return ResolutionWindow(earliest=target, latest=target, confidence=0.97, source_text=m.group(0))


def _h_by_full(m: re.Match[str], _ref: date) -> PatternResult:
    """Match ``by Month D[, ]YYYY`` or ``before December 31 2025``."""

    raw = m.group(0)
    month = _MONTHS[m["mon"].lower()]
    day = int(m["d"])
    year = int(m["y"])
    try:
        target = date(year, month, day)
    except ValueError:
        return None
    # "by" / "before" = upper bound; lower bound is open.
    return ResolutionWindow(earliest=None, latest=target, confidence=0.95, source_text=raw)


def _h_on_full(m: re.Match[str], _ref: date) -> PatternResult:
    """``on Month D YYYY`` / ``December 31st 2025`` (no "by" preposition)."""

    raw = m.group(0)
    month = _MONTHS[m["mon"].lower()]
    day = int(m["d"])
    year = int(m["y"])
    try:
        target = date(year, month, day)
    except ValueError:
        return None
    return ResolutionWindow(earliest=target, latest=target, confidence=0.93, source_text=raw)


def _h_eoy(m: re.Match[str], _ref: date) -> PatternResult:
    """``EOY 2028`` / ``end of 2028`` / ``year-end 2028`` / ``by 2030``."""

    year = int(m["y"])
    return ResolutionWindow(
        earliest=None,
        latest=date(year, 12, 31),
        confidence=0.85,
        source_text=m.group(0),
    )


def _h_in_year(m: re.Match[str], _ref: date) -> PatternResult:
    """``in 2029`` — full-year window."""

    year = int(m["y"])
    return ResolutionWindow(
        earliest=date(year, 1, 1),
        latest=date(year, 12, 31),
        confidence=0.70,
        source_text=m.group(0),
    )


def _h_quarter(m: re.Match[str], _ref: date) -> PatternResult:
    """``Q3 2025`` / ``by Q2 2026`` / ``in Q1 2027``."""

    q = int(m["q"])
    year = int(m["y"])
    preposition = (m["prep"] or "").lower().strip()
    if preposition in {"by", "before"}:
        return ResolutionWindow(
            earliest=None,
            latest=_end_of_quarter(year, q),
            confidence=0.82,
            source_text=m.group(0),
        )
    return ResolutionWindow(
        earliest=_start_of_quarter(year, q),
        latest=_end_of_quarter(year, q),
        confidence=0.82,
        source_text=m.group(0),
    )


def _h_month_year(m: re.Match[str], _ref: date) -> PatternResult:
    """``before May 2026`` / ``by June 2027`` / ``in November 2025``."""

    month = _MONTHS[m["mon"].lower()]
    year = int(m["y"])
    preposition = (m["prep"] or "").lower().strip()
    last = _eom(year, month)
    if preposition in {"by", "before"}:
        return ResolutionWindow(
            earliest=None,
            latest=date(year, month, last),
            confidence=0.80,
            source_text=m.group(0),
        )
    return ResolutionWindow(
        earliest=date(year, month, 1),
        latest=date(year, month, last),
        confidence=0.78,
        source_text=m.group(0),
    )


def _h_month_only(m: re.Match[str], ref: date) -> PatternResult:
    """``before May`` / ``by June`` — infer the *next* occurrence of that month."""

    month = _MONTHS[m["mon"].lower()]
    year = ref.year
    # If we're already past this month in ``ref.year``, roll to next year.
    candidate_last = date(year, month, _eom(year, month))
    if candidate_last < ref:
        year += 1
        candidate_last = date(year, month, _eom(year, month))
    return ResolutionWindow(
        earliest=None,
        latest=candidate_last,
        confidence=0.55,
        source_text=m.group(0),
    )


def _h_political(m: re.Match[str], ref: date) -> PatternResult:
    """``by next election`` / ``before midterms``."""

    phrase = m.group(0)
    found = _next_political_event(phrase, ref)
    if found is None:
        return None
    _earliest, latest = found
    return ResolutionWindow(
        earliest=None,
        latest=latest,
        confidence=0.35,
        source_text=phrase,
    )


def _h_political_yeared(m: re.Match[str], _ref: date) -> PatternResult:
    """``the 2028 election`` / ``2024 election`` / ``2026 midterms``."""

    raw = m.group(0)
    key = re.sub(r"\s+", " ", raw.lower().strip())
    # Find a key in _POLITICAL_ANCHORS that appears in the matched phrase.
    for anchor_key, (earliest, latest) in _POLITICAL_ANCHORS.items():
        if anchor_key in key:
            return ResolutionWindow(
                earliest=earliest,
                latest=latest,
                confidence=0.65,
                source_text=raw,
            )
    return None


# ---------------------------------------------------------------------------
# Pattern table
# ---------------------------------------------------------------------------
# Patterns are tried in order. The first non-None result wins. Where two
# patterns could match the same text we order from *most specific* to
# *least specific* so e.g. "December 31 2025" wins over a generic "in 2025".

_PATTERNS: list[tuple[re.Pattern[str], PatternFn]] = [
    # ISO 2026-01-01 (allow optional T-time suffix)
    (
        re.compile(
            r"\b(?P<y>20\d{2})-(?P<mo>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12]\d|3[01])"
            r"(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"
        ),
        _h_iso,
    ),
    # "by December 31 2025" / "by Dec 31, 2025" / "before May 1, 2026"
    (
        re.compile(
            r"\b(?:by|before|on or before|no later than)\s+(?P<mon>"
            + _MONTH_ALT[1:-1]
            + r")\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>20\d{2})\b",
            re.IGNORECASE,
        ),
        _h_by_full,
    ),
    # Bare "December 31 2025" / "Dec 31, 2025" / "December 31st, 2025"
    (
        re.compile(
            r"\b(?P<mon>"
            + _MONTH_ALT[1:-1]
            + r")\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>20\d{2})\b",
            re.IGNORECASE,
        ),
        _h_on_full,
    ),
    # "EOY 2028" / "end of 2028" / "year-end 2028" / "end-of-year 2028"
    (
        re.compile(
            r"\b(?:by\s+)?(?:eoy|end[\s-]of[\s-]year|year[\s-]end|end\s+of)\s+(?P<y>20\d{2})\b",
            re.IGNORECASE,
        ),
        _h_eoy,
    ),
    # "by 2030" — bare year as an upper bound
    (
        re.compile(r"\b(?:by|before|no later than)\s+(?P<y>20\d{2})\b", re.IGNORECASE),
        _h_eoy,
    ),
    # "by Q2 2026" / "in Q3 2025"
    (
        re.compile(
            r"\b(?P<prep>by|before|in|during)?\s*Q(?P<q>[1-4])\s+(?P<y>20\d{2})\b",
            re.IGNORECASE,
        ),
        _h_quarter,
    ),
    # "before May 2026" / "by June 2027" / "in November 2025"
    (
        re.compile(
            r"\b(?P<prep>by|before|in|during)\s+(?P<mon>"
            + _MONTH_ALT[1:-1]
            + r")\s+(?P<y>20\d{2})\b",
            re.IGNORECASE,
        ),
        _h_month_year,
    ),
    # "the 2028 election" / "2026 midterms" — political-yeared anchors.
    (
        re.compile(
            r"\b(?P<y>20\d{2})\s+(?:presidential\s+)?(?:election|midterms?)\b",
            re.IGNORECASE,
        ),
        _h_political_yeared,
    ),
    # "in 2029" — full-year window. Must come after political-yeared so
    # "in 2024 election" doesn't get swallowed as a year-only match.
    (
        re.compile(r"\b(?:in|during)\s+(?P<y>20\d{2})\b", re.IGNORECASE),
        _h_in_year,
    ),
    # "before May" / "by June" — month only (no year).
    (
        re.compile(
            r"\b(?P<prep>by|before)\s+(?P<mon>" + _MONTH_ALT[1:-1] + r")\b(?!\s+\d)",
            re.IGNORECASE,
        ),
        _h_month_only,
    ),
    # Relative political ("by the next election", "before midterms")
    (
        re.compile(
            r"\b(?:by|before)\s+(?:the\s+)?(?:next\s+)?(?:presidential\s+)?(?:election|midterms?)\b",
            re.IGNORECASE,
        ),
        _h_political,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_resolution_window(text: str, *, reference_date: date | None = None) -> ResolutionWindow:
    """Infer a :class:`ResolutionWindow` from free-form market text.

    Parameters
    ----------
    text:
        Title / description / question body. May contain multiple dates;
        the *most specific* match wins (see pattern ordering).
    reference_date:
        "Now" for relative phrases (default: ``date.today()``). Lets tests
        be deterministic.

    Returns
    -------
    ResolutionWindow
        ``ResolutionWindow(None, None, 0.0, "")`` if nothing matched.

    Notes
    -----
    The function is *intentionally* conservative: it would rather return
    a low-confidence window than guess a year. Callers should treat
    ``confidence < 0.5`` as "do not block arb pairing on this alone."
    """

    if not text or not text.strip():
        return _EMPTY

    ref = reference_date or date.today()

    for pattern, handler in _PATTERNS:
        for m in pattern.finditer(text):
            result = handler(m, ref)
            if result is not None:
                return result

    # Last-ditch: try dateutil on the whole string. Only trust it if it
    # *clearly* extracts a year. We deliberately avoid `fuzzy=True` here
    # because it would hallucinate "today" out of generic words.
    try:
        parsed = dateutil_parser.parse(text, default=date(1900, 1, 1))
    except (ValueError, OverflowError, TypeError):
        return _EMPTY
    if parsed.year >= 2024 and parsed.year != 1900:
        as_date = parsed.date()
        return ResolutionWindow(
            earliest=as_date,
            latest=as_date,
            confidence=0.40,
            source_text=text.strip()[:64],
        )
    return _EMPTY


#: Default proximity (in days) used to decide whether two same-shape
#: half-open windows likely describe the *same* event vs distinct events
#: a long time apart. 30 days is the empirically-derived sweet spot for
#: prediction-market arb candidates: it covers FOMC-to-FOMC slack and same-
#: month resolution drift while rejecting same-month-different-year false
#: positives ("Trump 2024" vs "Trump 2028"). Override via
#: ``half_open_proximity_days`` to tune for new venues.
_HALF_OPEN_PROXIMITY_DAYS = 30


def windows_overlap(
    a: ResolutionWindow,
    b: ResolutionWindow,
    *,
    min_overlap_days: int = 1,
    half_open_proximity_days: int = _HALF_OPEN_PROXIMITY_DAYS,
) -> bool:
    """Return ``True`` iff ``a`` and ``b`` overlap by at least ``min_overlap_days``.

    Semantics
    ---------
    This function has THREE branches, ordered most-specific to most-general:

    1. **Both windows are point dates** (``earliest == latest``). Two point
       dates "overlap" iff they are within ``half_open_proximity_days`` of
       each other. This is the only safe interpretation: a single-day
       election resolving Nov 5 2024 should NOT match a single-day election
       resolving Nov 7 2028.

    2. **Both windows are half-open of the same shape** (both
       ``is_half_open_by``, i.e. both extracted from ``"by DATE"``-style
       text; or both ``is_half_open_from``). Substituting ``date.min`` for
       the missing lower bound would make every two ``"by DATE"`` windows
       overlap on (date.min, min(latest_a, latest_b)) — the bug fixed by
       T76b. Instead, we treat each window as an *event near its known
       bound* and require the two known bounds to be within
       ``half_open_proximity_days``.

    3. **Otherwise (general case)** we fall back to interval intersection,
       substituting :data:`date.min` / :data:`date.max` for missing bounds.
       Mixed half-open shapes ("by X" vs "after Y") and fully-bounded
       windows go through here. ``min_overlap_days`` is the inclusive
       day-count threshold (``=1`` requires sharing at least one full day).

    All-``None`` windows always return ``False`` (we cannot make a safe
    claim that an un-extracted window overlaps anything).
    """

    if min_overlap_days < 1:
        raise ValueError("min_overlap_days must be >= 1")
    if half_open_proximity_days < 0:
        raise ValueError("half_open_proximity_days must be >= 0")

    # Duck-typed access: T77 unit tests build a local `_RW` stand-in that
    # only exposes ``.earliest`` / ``.latest`` / ``.confidence``. We must
    # not depend on the rich ``ResolutionWindow.is_*`` properties because
    # those stand-ins won't have them. Recover the equivalent predicates
    # from the bare fields.
    a_earliest = getattr(a, "earliest", None)
    a_latest = getattr(a, "latest", None)
    b_earliest = getattr(b, "earliest", None)
    b_latest = getattr(b, "latest", None)

    a_empty = a_earliest is None and a_latest is None
    b_empty = b_earliest is None and b_latest is None
    if a_empty or b_empty:
        return False

    a_is_point = a_earliest is not None and a_latest is not None and a_earliest == a_latest
    b_is_point = b_earliest is not None and b_latest is not None and b_earliest == b_latest
    a_half_open_by = a_earliest is None and a_latest is not None
    b_half_open_by = b_earliest is None and b_latest is not None
    a_half_open_from = a_earliest is not None and a_latest is None
    b_half_open_from = b_earliest is not None and b_latest is None

    def _days_between(d1: Any, d2: Any) -> int:
        # ``date - date`` returns timedelta; ``datetime - datetime`` ditto.
        # Both expose ``.days``. Mixed (date, datetime) does not work, so
        # we coerce datetimes to dates first.
        if hasattr(d1, "date") and not isinstance(d1, date):
            d1 = d1.date()
        if hasattr(d2, "date") and not isinstance(d2, date):
            d2 = d2.date()
        return abs((d1 - d2).days)

    # Branch 1: both windows are point dates. Strict proximity required.
    if a_is_point and b_is_point:
        return _days_between(a_latest, b_latest) <= half_open_proximity_days

    # Branch 2: both windows are half-open of the SAME shape. Without this
    # check, the open side trivially intersects via date.min/date.max and
    # every "by 2024" vs "by 2028" pair is reported as overlapping. The
    # fix is to compare the two *known* bounds directly: if they are far
    # apart, the markets are about different events.
    if a_half_open_by and b_half_open_by:
        return _days_between(a_latest, b_latest) <= half_open_proximity_days
    if a_half_open_from and b_half_open_from:
        return _days_between(a_earliest, b_earliest) <= half_open_proximity_days

    # Branch 3: general interval intersection. Mixed half-open shapes and
    # fully-bounded windows land here. Substituting date.min/date.max for
    # missing bounds is safe in this branch because at least one side
    # of each window is bounded — open-vs-open is handled above.
    def _to_date_like(value: Any, fallback: Any) -> Any:
        if value is None:
            return fallback
        return value

    # Pick fallbacks that are comparable to whatever the caller passed. If
    # the bounds are ``datetime`` (T77 ``_RW`` stand-in), use ``datetime.min``
    # /``datetime.max`` so ``min`` / ``max`` don't raise TypeError.
    sample = a_latest or b_latest or a_earliest or b_earliest
    if hasattr(sample, "tzinfo"):
        from datetime import datetime as _dt

        lo_sentinel: Any = _dt.min.replace(tzinfo=getattr(sample, "tzinfo", None))
        hi_sentinel: Any = _dt.max.replace(tzinfo=getattr(sample, "tzinfo", None))
    else:
        lo_sentinel = date.min
        hi_sentinel = date.max

    a_lo = _to_date_like(a_earliest, lo_sentinel)
    a_hi = _to_date_like(a_latest, hi_sentinel)
    b_lo = _to_date_like(b_earliest, lo_sentinel)
    b_hi = _to_date_like(b_latest, hi_sentinel)

    overlap_lo = max(a_lo, b_lo)
    overlap_hi = min(a_hi, b_hi)
    if overlap_hi < overlap_lo:
        return False
    overlap_delta = overlap_hi - overlap_lo
    overlap_days = overlap_delta.days + 1  # inclusive
    return overlap_days >= min_overlap_days
