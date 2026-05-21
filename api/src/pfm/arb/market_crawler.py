"""Unlimited, resumable, newest-first market-universe crawler.

This module powers the cross-venue arb discovery upgrade. It walks the full
open-market universe of both venues *step by step* (a cycle does one bounded
step; the next cycle resumes from the saved checkpoint), so that over many
cycles the crawler covers an effectively unlimited universe without ever
blocking on a single long sweep.

Two venue shapes are handled:

* **Kalshi** — cursor-based pagination (``cursor`` token in the response;
  empty cursor or empty ``markets`` array means the sweep is exhausted).
  Page-size cap is 1000. There is no server-side ordering, so for
  "newest-first" semantics we sort client-side by ``open_time`` descending.
* **Polymarket (gamma)** — offset-based pagination, already sorted
  newest-first by ``startDate``. Page-size clamps to 100 and the offset has a
  **hard cap**: ``offset >= 10100`` returns HTTP 422. We treat that 422 as a
  graceful end-of-sweep, never an error.

All network access is funnelled through :func:`_get`, which tests monkeypatch.
The freshness helpers (:func:`new_kalshi_markets`, :func:`new_poly_events`)
read **market-level** timestamps — fixing the prior bug where Kalshi freshness
read event-level fields that do not exist on the markets endpoint.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants (verified against live API probes — 2026-05).
# ---------------------------------------------------------------------------

KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
KALSHI_EVENTS_PAGE_LIMIT = 200
KALSHI_PAGE_LIMIT = 1000
KALSHI_PACE_S = 0.3  # ~3 req/s

POLY_EVENTS_URL = "https://gamma-api.polymarket.com/events"
POLY_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
POLY_PAGE_LIMIT = 100
POLY_OFFSET_CAP = 10100  # offset >= this -> HTTP 422 "offset exceeds maximum"
POLY_PACE_S = 0.2  # ~5 req/s

DEFAULT_TIMEOUT_S = 15.0
MAX_BACKOFF_RETRIES = 4
DEFAULT_BACKOFF_S = 1.0

DEFAULT_CHECKPOINT_PATH = "arbstuff/crawl_state.json"


# ---------------------------------------------------------------------------
# HTTP layer (the single network seam tests monkeypatch).
# ---------------------------------------------------------------------------


class CrawlHTTPError(RuntimeError):
    """Raised when an HTTP request fails after exhausting retries."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


def _get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    session: Any = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Perform a GET with 429 backoff and a hard timeout.

    This is the only function that touches the network. Tests monkeypatch it
    (or the ``session.get`` it calls) to stay fully offline.

    Args:
        url: Absolute request URL.
        params: Query parameters.
        session: Optional ``requests``-like session with a ``.get`` method. A
            fresh ``requests`` session is created lazily when ``None``.
        timeout: Per-request timeout in seconds.

    Returns:
        The parsed JSON body as a dict-like object.

    Raises:
        CrawlHTTPError: For any non-retryable HTTP error, or after the retry
            budget for 429s is exhausted. Callers that want to treat specific
            status codes (e.g. Polymarket's 422 offset cap) as terminal must
            inspect :attr:`CrawlHTTPError.status_code`.
    """
    if session is None:
        import requests  # lazy: keep import cost off the module import path

        session = requests.Session()

    backoff = DEFAULT_BACKOFF_S
    for attempt in range(MAX_BACKOFF_RETRIES + 1):
        resp = session.get(url, params=params, timeout=timeout)
        status = getattr(resp, "status_code", 200)

        if status == 429 and attempt < MAX_BACKOFF_RETRIES:
            retry_after = _retry_after_seconds(resp, default=backoff)
            time.sleep(retry_after)
            backoff *= 2
            continue

        if status >= 400:
            raise CrawlHTTPError(status, _safe_text(resp))

        return resp.json()

    # Retry budget exhausted on repeated 429s.
    raise CrawlHTTPError(429, "rate limited (retry budget exhausted)")


def _retry_after_seconds(resp: Any, *, default: float) -> float:
    """Extract a ``Retry-After`` delay (seconds) from a response, if present."""
    headers = getattr(resp, "headers", {}) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return default


def _safe_text(resp: Any) -> str:
    """Best-effort response body text for error messages."""
    text = getattr(resp, "text", None)
    if isinstance(text, str):
        return text[:300]
    return ""


# ---------------------------------------------------------------------------
# Time helpers.
# ---------------------------------------------------------------------------


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp (``...Z`` or offset) into aware UTC.

    Returns ``None`` for missing or unparseable values so callers can fall
    back to alternative fields without raising.
    """
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _market_open_dt(market: dict[str, Any]) -> datetime | None:
    """Kalshi market-level freshness timestamp: ``open_time`` then ``created_time``.

    Reading the *market-level* field is the fix for the prior freshness bug,
    which looked at event-level fields absent from the markets endpoint.
    """
    return _parse_iso(market.get("open_time")) or _parse_iso(market.get("created_time"))


def _event_start_dt(event: dict[str, Any]) -> datetime | None:
    """Polymarket event freshness timestamp: ``startDate`` then ``createdAt``."""
    return _parse_iso(event.get("startDate")) or _parse_iso(event.get("createdAt"))


# ---------------------------------------------------------------------------
# Ephemeral-series filter.
# ---------------------------------------------------------------------------

#: Case-insensitive regex patterns that mark a market/event as an EPHEMERAL,
#: templated, short-lived series — the kind that resolves in minutes and is
#: never a sensible cross-venue arb target (5m/15m crypto up-down, hourly
#: weather, intraday price-window markets, etc.). Kept as a module constant so
#: it is trivial to extend with new noisy series as they appear. Each entry is
#: compiled with ``re.IGNORECASE`` in :data:`_EPHEMERAL_RES`.
EPHEMERAL_PATTERNS: list[str] = [
    # Up/Down templated short-horizon markets ("Solana Up or Down - ...").
    r"\bup or down\b",
    r"\bup/down\b",
    r"\bup vs\.? down\b",
    # Explicit short-window tags.
    r"\b\d{1,2}\s?m(?:in)?\b(?=.*\b(up|down|window|candle)\b)",
    r"\b(5|10|15|30)\s?m\b",
    r"\b1\s?h(?:our)?\b(?=.*\b(up|down|window|candle|temp)\b)",
    # Intraday time-window patterns, e.g. "3:15PM-3:30PM ET" or "at 4:00 pm".
    r"\d{1,2}:\d{2}\s*[ap]m",
    # Colon-less hourly strike series, e.g. "Ethereum above ___ on May 21, 4PM ET?".
    r"\b\d{1,2}\s*[ap]m\s+et\b",
    # Hourly / daily weather templated series.
    r"\bhighest temperature\b",
    r"\bhigh(?:est)? temp\b",
    r"\bhigh temp in\b",
    r"\blowest temperature\b",
    r"\brain in\b",
    r"\bwill it rain\b",
    r"\btemperature in\b.*\b(today|tonight|tomorrow)\b",
]

_EPHEMERAL_RES: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in EPHEMERAL_PATTERNS]


def is_ephemeral_market(title_or_event: str) -> bool:
    """Return ``True`` for templated, short-lived (ephemeral) market series.

    Ephemeral markets — 5m/15m crypto up-down, intraday price-window contracts,
    hourly/daily weather — resolve in minutes and are not cross-venue arb
    targets, yet they dominate the newest-listing feeds and drown out the real
    candidates (elections, Fed, crypto-price-by-EOY, sports outrights). This
    drops them from discovery while keeping every non-ephemeral new market.

    Args:
        title_or_event: A human title (event-level title preferred) or any text
            describing the market.

    Returns:
        ``True`` when the text matches any :data:`EPHEMERAL_PATTERNS` entry.
    """
    if not title_or_event:
        return False
    text = str(title_or_event)
    return any(rx.search(text) for rx in _EPHEMERAL_RES)


def _event_text(event: dict[str, Any]) -> str:
    """Concatenate the human-facing text fields of an event for filtering."""
    parts = [
        event.get("title"),
        event.get("sub_title"),
        event.get("subtitle"),
        event.get("question"),
    ]
    return " ".join(str(p) for p in parts if p)


def _poly_event_text(event: dict[str, Any]) -> str:
    """Concatenate the human-facing text fields of a Polymarket event."""
    parts = [
        event.get("title"),
        event.get("question"),
        event.get("description"),
    ]
    return " ".join(str(p) for p in parts if p)


def parse_clob_token_ids(market: dict[str, Any]) -> list[str]:
    """Decode Polymarket ``clobTokenIds`` (a JSON string nested in JSON).

    The gamma API serialises this field as a JSON-encoded string *inside* the
    JSON response, so it must be ``json.loads``-ed a second time. Returns an
    empty list when the field is missing or malformed.
    """
    raw = market.get("clobTokenIds")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
        if isinstance(decoded, list):
            return [str(x) for x in decoded]
    return []


# ---------------------------------------------------------------------------
# Crawl result dataclasses.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KalshiCrawlPage:
    """Result of a bounded Kalshi cursor walk."""

    markets: list[dict[str, Any]] = field(default_factory=list)
    next_cursor: str | None = None
    done: bool = False
    n_pages: int = 0


@dataclass(slots=True)
class KalshiEventsPage:
    """Result of a bounded Kalshi *events* cursor walk.

    Events carry the real human ``title``/``sub_title`` (the markets endpoint
    only exposes templated/garbage titles) plus the nested ``markets[]`` we
    price against, so discovery matches on events rather than raw markets.
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    next_cursor: str | None = None
    done: bool = False
    n_pages: int = 0


@dataclass(slots=True)
class PolyCrawlPage:
    """Result of a bounded Polymarket offset walk."""

    events: list[dict[str, Any]] = field(default_factory=list)
    next_offset: int = 0
    done: bool = False
    n_pages: int = 0


# ---------------------------------------------------------------------------
# Kalshi crawl.
# ---------------------------------------------------------------------------


def crawl_kalshi_markets(
    *,
    cursor: str | None = None,
    max_pages: int = 5,
    session: Any = None,
    page_limit: int = KALSHI_PAGE_LIMIT,
    pace_s: float = KALSHI_PACE_S,
) -> KalshiCrawlPage:
    """Walk the Kalshi open-markets cursor for up to ``max_pages`` pages.

    Newest-first: collected markets are sorted client-side by ``open_time``
    descending (the API has no server-side ordering).

    Args:
        cursor: Cursor token to resume from; ``None`` starts a fresh sweep.
        max_pages: Maximum pages to fetch in this step (step-by-step crawl).
        session: Optional HTTP session forwarded to :func:`_get`.
        page_limit: Per-page market count (capped at 1000 server-side).
        pace_s: Sleep between pages to respect the rate limit.

    Returns:
        A :class:`KalshiCrawlPage`. ``done`` is ``True`` when the cursor
        exhausts (empty cursor or empty ``markets`` page) within this step.
    """
    markets: list[dict[str, Any]] = []
    current = cursor
    done = False
    pages = 0

    for i in range(max_pages):
        params: dict[str, Any] = {"status": "open", "limit": page_limit}
        if current:
            params["cursor"] = current
        body = _get(KALSHI_MARKETS_URL, params=params, session=session)
        pages += 1

        page_markets = body.get("markets") or []
        markets.extend(page_markets)

        current = body.get("cursor") or None
        if not current or not page_markets:
            done = True
            break

        if pace_s and i < max_pages - 1:
            time.sleep(pace_s)

    markets.sort(key=_kalshi_sort_key, reverse=True)
    return KalshiCrawlPage(markets=markets, next_cursor=current, done=done, n_pages=pages)


def _kalshi_sort_key(market: dict[str, Any]) -> float:
    """Sort key for newest-first ordering; missing timestamps sort last."""
    dt = _market_open_dt(market)
    return dt.timestamp() if dt is not None else float("-inf")


def new_kalshi_markets(
    *,
    within_hours: float = 24.0,
    session: Any = None,
    max_pages: int = 3,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return Kalshi markets opened within ``within_hours``, newest-first.

    Fixes the prior freshness bug by filtering on the **market-level**
    ``open_time`` (falling back to ``created_time``) rather than non-existent
    event-level fields.

    Args:
        within_hours: Freshness window in hours.
        session: Optional HTTP session.
        max_pages: Page budget for the underlying crawl (kept small).
        now: Reference "now"; defaults to current UTC. Useful for tests.

    Returns:
        Markets whose open time is within the window, sorted newest-first.
    """
    ref = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = ref.timestamp() - within_hours * 3600.0

    page = crawl_kalshi_markets(max_pages=max_pages, session=session)
    fresh = [
        m
        for m in page.markets
        if (dt := _market_open_dt(m)) is not None and dt.timestamp() >= cutoff
    ]
    fresh.sort(key=_kalshi_sort_key, reverse=True)
    return fresh


# ---------------------------------------------------------------------------
# Kalshi *events* crawl (real titles + nested markets).
# ---------------------------------------------------------------------------


def crawl_kalshi_events(
    *,
    cursor: str | None = None,
    max_pages: int = 5,
    session: Any = None,
    page_limit: int = KALSHI_EVENTS_PAGE_LIMIT,
    pace_s: float = KALSHI_PACE_S,
) -> KalshiEventsPage:
    """Walk the Kalshi open *events* cursor for up to ``max_pages`` pages.

    Hits ``GET /events?status=open&with_nested_markets=true`` so each event
    carries its real human ``title``/``sub_title`` and the nested ``markets[]``
    (the markets endpoint alone exposes only templated/garbage titles). Newest
    market titles live at the event level, so this is what discovery matches on.

    Args:
        cursor: Cursor token to resume from; ``None`` starts a fresh sweep.
        max_pages: Maximum pages to fetch in this step (step-by-step crawl).
        session: Optional HTTP session forwarded to :func:`_get`.
        page_limit: Per-page event count (200 server-side default).
        pace_s: Sleep between pages to respect the rate limit.

    Returns:
        A :class:`KalshiEventsPage`. ``done`` is ``True`` when the cursor
        exhausts (empty cursor or empty ``events`` page) within this step.
    """
    events: list[dict[str, Any]] = []
    current = cursor
    done = False
    pages = 0

    for i in range(max_pages):
        params: dict[str, Any] = {
            "status": "open",
            "with_nested_markets": "true",
            "limit": page_limit,
        }
        if current:
            params["cursor"] = current
        body = _get(KALSHI_EVENTS_URL, params=params, session=session)
        pages += 1

        page_events = body.get("events") or []
        events.extend(e for e in page_events if isinstance(e, dict))

        current = body.get("cursor") or None
        if not current or not page_events:
            done = True
            break

        if pace_s and i < max_pages - 1:
            time.sleep(pace_s)

    return KalshiEventsPage(events=events, next_cursor=current, done=done, n_pages=pages)


def _event_freshest_open_dt(event: dict[str, Any]) -> datetime | None:
    """Freshness for a Kalshi event: MAX ``open_time`` across nested markets.

    Events carry no event-level timestamp, so we derive freshness from the most
    recently opened nested market (``open_time`` then ``created_time``). Returns
    ``None`` when the event has no datable nested markets.
    """
    best: datetime | None = None
    for m in event.get("markets") or []:
        if not isinstance(m, dict):
            continue
        dt = _market_open_dt(m)
        if dt is not None and (best is None or dt > best):
            best = dt
    return best


def new_kalshi_events(
    *,
    within_hours: float = 24.0,
    session: Any = None,
    max_pages: int = 3,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return fresh, non-ephemeral Kalshi *events*, newest-first.

    Events have no event-level timestamp, so freshness is derived from the MAX
    ``open_time``/``created_time`` across the event's nested ``markets[]`` — an
    event is kept when its freshest market opened within ``within_hours``. This
    fixes the prior Kalshi-freshness no-op AND yields the real human titles.
    Ephemeral templated series (see :func:`is_ephemeral_market`) are dropped.

    Args:
        within_hours: Freshness window in hours.
        session: Optional HTTP session.
        max_pages: Page budget for the underlying event crawl (kept small).
        now: Reference "now"; defaults to current UTC. Useful for tests.

    Returns:
        Events whose freshest nested market opened within the window, newest-
        first, with ephemeral series filtered out.
    """
    ref = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = ref.timestamp() - within_hours * 3600.0

    page = crawl_kalshi_events(max_pages=max_pages, session=session)
    fresh: list[tuple[float, dict[str, Any]]] = []
    for ev in page.events:
        dt = _event_freshest_open_dt(ev)
        if dt is None or dt.timestamp() < cutoff:
            continue
        if is_ephemeral_market(_event_text(ev)):
            continue
        fresh.append((dt.timestamp(), ev))

    fresh.sort(key=lambda t: t[0], reverse=True)
    return [ev for _, ev in fresh]


# ---------------------------------------------------------------------------
# Polymarket crawl.
# ---------------------------------------------------------------------------


def crawl_poly_events(
    *,
    offset: int = 0,
    max_pages: int = 5,
    session: Any = None,
    page_limit: int = POLY_PAGE_LIMIT,
    pace_s: float = POLY_PACE_S,
) -> PolyCrawlPage:
    """Walk Polymarket events by offset for up to ``max_pages`` pages.

    Events are requested newest-first (``order=startDate&ascending=false``).
    The hard offset cap (``offset >= 10100`` -> HTTP 422) is caught and
    reported as a graceful end-of-sweep; it never propagates as an error.

    Args:
        offset: Starting offset to resume from.
        max_pages: Maximum pages to fetch in this step.
        session: Optional HTTP session.
        page_limit: Per-page event count (clamped to 100 server-side).
        pace_s: Sleep between pages to respect the rate limit.

    Returns:
        A :class:`PolyCrawlPage`. ``done`` is ``True`` on a short/empty page
        or when the offset cap is hit.
    """
    events: list[dict[str, Any]] = []
    current = offset
    done = False
    pages = 0

    for i in range(max_pages):
        if current >= POLY_OFFSET_CAP:
            done = True
            break

        params = {
            "active": "true",
            "closed": "false",
            "order": "startDate",
            "ascending": "false",
            "limit": page_limit,
            "offset": current,
        }
        try:
            body = _get(POLY_EVENTS_URL, params=params, session=session)
        except CrawlHTTPError as exc:
            if exc.status_code == 422:
                done = True
                break
            raise

        pages += 1
        page_events = _coerce_event_list(body)
        events.extend(page_events)
        current += page_limit

        if len(page_events) < page_limit:
            # Short or empty page -> reached the end of the available universe.
            done = True
            break

        if pace_s and i < max_pages - 1:
            time.sleep(pace_s)

    return PolyCrawlPage(events=events, next_offset=current, done=done, n_pages=pages)


def _coerce_event_list(body: Any) -> list[dict[str, Any]]:
    """Normalise the gamma events payload to a list of event dicts."""
    if isinstance(body, list):
        return [e for e in body if isinstance(e, dict)]
    if isinstance(body, dict):
        data = body.get("events") or body.get("data") or []
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict)]
    return []


def _coerce_market_list(body: Any) -> list[dict[str, Any]]:
    """Normalise the gamma *markets* payload to a list of market dicts.

    The ``/markets`` endpoint returns a bare JSON array, but tolerate a dict
    envelope (``markets``/``data``) for forward-compat the same way
    :func:`_coerce_event_list` does.
    """
    if isinstance(body, list):
        return [m for m in body if isinstance(m, dict)]
    if isinstance(body, dict):
        data = body.get("markets") or body.get("data") or []
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
    return []


def crawl_poly_by_volume(
    *,
    offset: int = 0,
    max_pages: int = 5,
    session: Any = None,
    page_limit: int = POLY_PAGE_LIMIT,
    pace_s: float = POLY_PACE_S,
) -> PolyCrawlPage:
    """Walk Polymarket *markets* highest-volume-first for up to ``max_pages`` pages.

    This is the **substantive/liquid coverage** crawl. The newest-first
    ``crawl_poly_events`` path is flooded with ephemeral sports + hourly crypto,
    so after the ephemeral filter only a handful of substantive markets survive
    per page and the discovery starves of the politics/macro/long-dated universe
    where real cross-venue arbs live. Sorting by ``volumeNum`` descending surfaces
    exactly that liquid universe.

    Verified live (2026-05-21): ``GET /markets?closed=false&active=true&
    order=volumeNum&ascending=false&limit=100&offset=`` returns a bare JSON array
    of *market* dicts ordered by descending traded volume (top hits are
    elections / geopolitics / macro, NOT "X vs Y - Halftime"). The market dict
    carries ``question``/``slug``/``description``/``clobTokenIds`` — the exact
    fields the matcher's ``_poly_desc`` reads — so these are usable as poly items
    directly. The offset cap is identical to the events feed (``offset >= 10100``
    -> HTTP 422), handled gracefully as end-of-sweep.

    Args:
        offset: Starting offset to resume from.
        max_pages: Maximum pages to fetch in this step.
        session: Optional HTTP session.
        page_limit: Per-page market count (clamped to 100 server-side).
        pace_s: Sleep between pages to respect the rate limit.

    Returns:
        A :class:`PolyCrawlPage` whose ``events`` field holds the volume-sorted
        *market* dicts (the field name is shared with the events crawl for a
        uniform downstream contract). ``done`` is ``True`` on a short/empty page
        or when the offset cap is hit.
    """
    markets: list[dict[str, Any]] = []
    current = offset
    done = False
    pages = 0

    for i in range(max_pages):
        if current >= POLY_OFFSET_CAP:
            done = True
            break

        params = {
            "active": "true",
            "closed": "false",
            "order": "volumeNum",
            "ascending": "false",
            "limit": page_limit,
            "offset": current,
        }
        try:
            body = _get(POLY_MARKETS_URL, params=params, session=session)
        except CrawlHTTPError as exc:
            if exc.status_code == 422:
                done = True
                break
            raise

        pages += 1
        page_markets = _coerce_market_list(body)
        markets.extend(page_markets)
        current += page_limit

        if len(page_markets) < page_limit:
            # Short or empty page -> reached the end of the available universe.
            done = True
            break

        if pace_s and i < max_pages - 1:
            time.sleep(pace_s)

    return PolyCrawlPage(events=markets, next_offset=current, done=done, n_pages=pages)


def new_poly_events(
    *,
    within_hours: float = 24.0,
    session: Any = None,
    max_pages: int = 20,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return Polymarket events started within ``within_hours``, newest-first.

    Because the feed is already newest-first, this **early-stops** as soon as
    it sees an event older than the window — so it finds all fresh events in a
    few pages without ever approaching the offset cap.

    Args:
        within_hours: Freshness window in hours.
        session: Optional HTTP session.
        max_pages: Hard page budget guarding against pathological feeds.
        now: Reference "now"; defaults to current UTC. Useful for tests.

    Returns:
        Events whose start time is within the window, newest-first.
    """
    ref = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = ref.timestamp() - within_hours * 3600.0

    fresh: list[dict[str, Any]] = []
    offset = 0

    for _ in range(max_pages):
        if offset >= POLY_OFFSET_CAP:
            break
        page = crawl_poly_events(offset=offset, max_pages=1, session=session)
        if not page.events:
            break

        stop = False
        for ev in page.events:
            dt = _event_start_dt(ev)
            if dt is None:
                continue
            if dt.timestamp() >= cutoff:
                # Drop ephemeral templated series (5m crypto, hourly weather…).
                if is_ephemeral_market(_poly_event_text(ev)):
                    continue
                fresh.append(ev)
            else:
                # Feed is newest-first, so the first old event ends the search.
                stop = True
                break

        if stop or page.done:
            break
        offset = page.next_offset

    fresh.sort(
        key=lambda e: d.timestamp() if (d := _event_start_dt(e)) is not None else float("-inf"),
        reverse=True,
    )
    return fresh


# ---------------------------------------------------------------------------
# Resumable checkpoint.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CrawlCheckpoint:
    """Persisted crawl position enabling step-by-step, unlimited sweeps.

    Attributes:
        kalshi_cursor: Cursor to resume the Kalshi sweep; ``None`` restarts it.
        poly_offset: Offset to resume the Polymarket sweep.
        last_seen_poly_start_iso: Newest ``startDate`` observed, for newest-
            first dedupe across cycles.
    """

    kalshi_cursor: str | None = None
    poly_offset: int = 0
    last_seen_poly_start_iso: str | None = None

    def needs_kalshi_reset(self, page: KalshiCrawlPage) -> bool:
        """Whether the Kalshi cursor exhausted and should restart next cycle."""
        return page.done or not page.next_cursor

    def needs_poly_reset(self) -> bool:
        """Whether the Polymarket offset reached the hard cap."""
        return self.poly_offset >= POLY_OFFSET_CAP


def advance_checkpoint(
    ckpt: CrawlCheckpoint,
    *,
    kalshi_page: KalshiCrawlPage | None = None,
    poly_page: PolyCrawlPage | None = None,
) -> CrawlCheckpoint:
    """Advance a checkpoint after a crawl step, resetting on exhaustion.

    When the Kalshi cursor exhausts or the Polymarket offset hits the cap, the
    corresponding position is reset to begin a fresh newest-first sweep on the
    next cycle.

    Args:
        ckpt: Current checkpoint (not mutated).
        kalshi_page: The Kalshi step result, if Kalshi was crawled.
        poly_page: The Polymarket step result, if Polymarket was crawled.

    Returns:
        A new :class:`CrawlCheckpoint` with advanced/reset positions.
    """
    kalshi_cursor = ckpt.kalshi_cursor
    poly_offset = ckpt.poly_offset
    last_seen = ckpt.last_seen_poly_start_iso

    if kalshi_page is not None:
        if kalshi_page.done or not kalshi_page.next_cursor:
            kalshi_cursor = None  # exhausted -> fresh sweep next cycle
        else:
            kalshi_cursor = kalshi_page.next_cursor

    if poly_page is not None:
        if poly_page.next_offset >= POLY_OFFSET_CAP or poly_page.done:
            poly_offset = 0  # cap hit / end -> fresh sweep next cycle
        else:
            poly_offset = poly_page.next_offset

        newest = _newest_poly_start(poly_page.events)
        if newest is not None:
            last_seen = newest

    return CrawlCheckpoint(
        kalshi_cursor=kalshi_cursor,
        poly_offset=poly_offset,
        last_seen_poly_start_iso=last_seen,
    )


def _newest_poly_start(events: list[dict[str, Any]]) -> str | None:
    """Return the newest ``startDate`` ISO string among ``events``."""
    best_dt: datetime | None = None
    best_raw: str | None = None
    for ev in events:
        dt = _event_start_dt(ev)
        if dt is None:
            continue
        if best_dt is None or dt > best_dt:
            best_dt = dt
            best_raw = ev.get("startDate") or ev.get("createdAt")
    return best_raw


def load_checkpoint(path: str | Path = DEFAULT_CHECKPOINT_PATH) -> CrawlCheckpoint:
    """Load a checkpoint from a JSON file, or a fresh one if absent/invalid.

    Args:
        path: JSON file path.

    Returns:
        The persisted :class:`CrawlCheckpoint`, or a default (fresh) one if the
        file does not exist or cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        return CrawlCheckpoint()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        return CrawlCheckpoint()
    if not isinstance(data, dict):
        return CrawlCheckpoint()
    return CrawlCheckpoint(
        kalshi_cursor=data.get("kalshi_cursor"),
        poly_offset=int(data.get("poly_offset") or 0),
        last_seen_poly_start_iso=data.get("last_seen_poly_start_iso"),
    )


def save_checkpoint(path: str | Path, ckpt: CrawlCheckpoint) -> None:
    """Persist a checkpoint to a JSON file, creating parent dirs as needed.

    Args:
        path: JSON file path.
        ckpt: Checkpoint to persist.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(ckpt), indent=2), encoding="utf-8")


__all__ = [
    "EPHEMERAL_PATTERNS",
    "CrawlCheckpoint",
    "CrawlHTTPError",
    "KalshiCrawlPage",
    "KalshiEventsPage",
    "PolyCrawlPage",
    "advance_checkpoint",
    "crawl_kalshi_events",
    "crawl_kalshi_markets",
    "crawl_poly_by_volume",
    "crawl_poly_events",
    "is_ephemeral_market",
    "load_checkpoint",
    "new_kalshi_events",
    "new_kalshi_markets",
    "new_poly_events",
    "parse_clob_token_ids",
    "save_checkpoint",
]
