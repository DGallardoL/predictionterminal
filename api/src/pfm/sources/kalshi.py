"""Kalshi client — daily candlesticks for macro/economic events.

Kalshi is a CFTC-regulated DCM. The public read endpoints don't need auth.

Endpoints used:
    GET /trade-api/v2/series/{series}/markets/{market}/candlesticks
        ?start_ts=<unix>&end_ts=<unix>&period_interval=1440

    GET /trade-api/v2/markets/{market}              (current snapshot)

Data quality vs Polymarket:
    - per-bar volume (volume_fp)
    - per-bar bid-ask (yes_bid.close_dollars / yes_ask.close_dollars)
    - per-bar open interest (open_interest_fp)
    - 200-290 bars of history typical
    - clean end-of-bucket UTC timestamps

Market ticker format: ``{SERIES}-{YYMMM}-{CONDITION}`` (e.g.
``KXFEDDECISION-26JUL-C25``). Series is whatever precedes the first ``-``.

Rate-limit handling:
    Kalshi's unauthenticated public endpoints throttle aggressively
    (HTTP 429). The client carries a built-in token-paced rate limiter
    (``min_interval_s``) plus retry-with-exponential-backoff that honors
    a ``Retry-After`` header when Kalshi sends one. The defaults
    (≈3 req/s + up to 5 retries) keep us well under the per-IP cap
    while remaining responsive for batched factor fetches.
"""

from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from pfm.cache_pool import CachePool
from pfm.vol.implied_pdf_schemas import (
    AssetClass,
    DataShape,
    LadderEntry,
    LadderFamily,
)

DAILY_FIDELITY_SECONDS = 1440  # 1440 minutes = 1 day, used as period_interval

# Conservative defaults for the unauthenticated public API. Empirically Kalshi
# starts returning 429 around ~5 req/s sustained from a single IP, so we pace
# at 3 req/s with up to 5 retries on 429.
DEFAULT_MIN_INTERVAL_S = 0.30
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_S = 1.0
DEFAULT_BACKOFF_CAP_S = 30.0

# Process-local market-metadata cache (W11-14: migrated to CachePool — gains
# optional Redis L2 + heap eviction without changing call-site semantics).
# Ticker→KalshiMarket is stable for the entire lifetime of a market — the
# only field that drifts is ``status`` (active → finalized → settled) and
# that's a once-in-history transition we don't mind being slightly stale
# on the UI for an hour. A top-of-app cache here collapses repeated
# metadata round-trips during the same fit/factor sweep into a single
# upstream call.
_MARKET_CACHE_TTL_S: int = 3600  # 1 h
_MARKET_CACHE_MAX_ENTRIES: int = 2048

# Exposed at module scope so conftest.py's autouse ``_reset_volatile_caches``
# fixture (which calls ``.clear()``) keeps working — CachePool.clear() is
# API-compatible with dict.clear().
_MARKET_CACHE: CachePool = CachePool(
    namespace="kalshi_market", l1_maxsize=_MARKET_CACHE_MAX_ENTRIES
)


class KalshiError(RuntimeError):
    """Raised when Kalshi returns a usable HTTP response but bad data."""


class KalshiRateLimitError(KalshiError):
    """Raised when retries are exhausted on HTTP 429 (rate limit)."""


@dataclass(frozen=True)
class KalshiMarket:
    """Subset of fields we use from Kalshi's market detail endpoint."""

    ticker: str
    series_ticker: str
    title: str
    status: str | None  # "active" | "finalized" | "settled" | None
    open_time: str | None
    close_time: str | None


def series_from_market(market_ticker: str) -> str:
    """Extract series ticker from a market ticker (everything before first ``-``)."""
    return market_ticker.split("-", 1)[0]


class _MinIntervalLimiter:
    """Thread-safe min-interval rate limiter.

    Blocks ``acquire()`` until at least ``min_interval_s`` has elapsed since
    the previous acquire. Cheaper than a full token bucket and sufficient
    for the single-process, low-concurrency use case here.
    """

    def __init__(self, min_interval_s: float) -> None:
        self._min = max(0.0, float(min_interval_s))
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self._min <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._min - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header. Accepts integer seconds OR HTTP-date.

    Returns the wait duration in seconds, or ``None`` if unparseable / past.
    """
    if not value:
        return None
    s = value.strip()
    # Integer seconds form
    try:
        return max(0.0, float(s))
    except ValueError:
        pass
    # HTTP-date form (RFC 7231)
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(s)
        if dt is None:
            return None
        delta = (dt - pd.Timestamp.now(tz="UTC").to_pydatetime()).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


class KalshiClient:
    """Sync httpx client for Kalshi public read endpoints.

    Built-in throttling and 429-retry. Configurable for tests via
    ``min_interval_s`` (set to 0 to disable pacing) and ``max_retries``.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
        *,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
        backoff_cap_s: float = DEFAULT_BACKOFF_CAP_S,
        sleep: Any = None,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._limiter = _MinIntervalLimiter(min_interval_s)
        self._max_retries = max(0, int(max_retries))
        self._backoff_base = float(backoff_base_s)
        self._backoff_cap = float(backoff_cap_s)
        # Injectable for tests. Real callers leave it as ``time.sleep``.
        self._sleep = sleep or time.sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> KalshiClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    # ---- HTTP layer with throttling + 429 retries -------------------------

    def _request(
        self, method: str, url: str, *, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Execute a request through the limiter, with bounded retries on 429.

        On 429, prefer a server-supplied ``Retry-After`` if present and short
        enough. Otherwise back off exponentially with ±25% jitter. Raises
        ``KalshiRateLimitError`` after exhausting retries.
        """
        last_retry_after: str | None = None
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            r = self._client.request(method, url, params=params)
            if r.status_code != 429:
                r.raise_for_status()
                return r
            last_retry_after = r.headers.get("Retry-After")
            if attempt >= self._max_retries:
                break
            wait = _parse_retry_after(last_retry_after)
            if wait is None or wait > self._backoff_cap:
                # Exponential backoff with jitter: base * 2^attempt ± 25%.
                base = self._backoff_base * (2**attempt)
                wait = base * (1.0 + random.uniform(-0.25, 0.25))
            wait = min(self._backoff_cap, max(0.0, wait))
            self._sleep(wait)
        raise KalshiRateLimitError(
            f"kalshi rate-limit exceeded after {self._max_retries} retries "
            f"on {method} {url} (Retry-After={last_retry_after!r})"
        )

    # ---- Market detail -----------------------------------------------------

    def get_market(self, ticker: str) -> KalshiMarket:
        """Look up current market state. Used to get the series_ticker for
        candlestick fetches.

        Cached for 1 h in a process-local dict. The series_ticker /
        title fields are immutable; status drifts on a multi-day scale
        once a market resolves so an hour of staleness is acceptable.
        On miss we still pay the full retry-on-429 path via ``_request``.
        """
        cached = _MARKET_CACHE.get(ticker)
        if cached is not None:
            return cached

        r = self._request("GET", f"{self.BASE_URL}/markets/{ticker}")
        data = r.json().get("market")
        if not data:
            raise KalshiError(f"no market found for ticker={ticker!r}")
        meta = KalshiMarket(
            ticker=str(data.get("ticker", ticker)),
            series_ticker=str(
                data.get("event_ticker", series_from_market(ticker)).split("-", 1)[0]
            ),
            title=str(data.get("title", "")),
            status=data.get("status"),
            open_time=data.get("open_time"),
            close_time=data.get("close_time"),
        )
        _MARKET_CACHE.set(ticker, meta, ttl=_MARKET_CACHE_TTL_S)
        return meta

    # ---- Candlesticks (history) -------------------------------------------

    def get_candlesticks(
        self,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = DAILY_FIDELITY_SECONDS,
        series_ticker: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily candlesticks. Returns a DataFrame indexed by UTC date
        with columns: ``price`` (close), ``volume``, ``open_interest``,
        ``yes_bid``, ``yes_ask``, ``spread``.

        Empty DataFrame if Kalshi has no data in the range.
        """
        series = series_ticker or series_from_market(market_ticker)
        url = f"{self.BASE_URL}/series/{series}/markets/{market_ticker}/candlesticks"
        params = {
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": int(period_interval),
        }
        r = self._request("GET", url, params=params)
        candles: list[dict[str, Any]] = r.json().get("candlesticks", [])
        if not candles:
            return pd.DataFrame(
                columns=["price", "volume", "open_interest", "yes_bid", "yes_ask", "spread"]
            )

        rows = []
        for c in candles:
            ts = pd.Timestamp(c["end_period_ts"], unit="s", tz="UTC").normalize()
            try:
                close = float(c["price"]["close_dollars"])
                bid = float(c["yes_bid"]["close_dollars"])
                ask = float(c["yes_ask"]["close_dollars"])
                rows.append(
                    {
                        "date": ts,
                        "price": close,
                        "volume": float(c.get("volume_fp", 0) or 0),
                        "open_interest": float(c.get("open_interest_fp", 0) or 0),
                        "yes_bid": bid,
                        "yes_ask": ask,
                        "spread": ask - bid,
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        if not rows:
            return pd.DataFrame(
                columns=["price", "volume", "open_interest", "yes_bid", "yes_ask", "spread"]
            )

        df = pd.DataFrame(rows).set_index("date").sort_index()
        # Collapse duplicate days (Kalshi can emit a final "current" bar).
        df = df.groupby(df.index).last()
        return df

    # ---- Events / nested-market ladders (index-ladder discovery) ----------

    def get_events(
        self,
        series_ticker: str,
        *,
        status: str = "open",
        with_nested_markets: bool = True,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch events (one per expiry) for a series, with nested markets.

        Calls ``GET /events?series_ticker=…&status=…&with_nested_markets=…``.
        Each returned event carries a nested ``markets`` list = the strike /
        bucket ladder for that expiry.

        Args:
            series_ticker: Kalshi series ticker, e.g. ``"KXINX"``.
            status: Event status filter (``"open"`` by default).
            with_nested_markets: Embed the per-event ``markets`` ladder.
            limit: Page size (max 200 returned in v1; pagination is minimal).

        Returns:
            The raw ``events`` list (list of dicts) from the response.
        """
        url = f"{self.BASE_URL}/events"
        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "limit": int(limit),
            "with_nested_markets": str(bool(with_nested_markets)).lower(),
        }
        if status:
            params["status"] = status
        r = self._request("GET", url, params=params)
        body = r.json()
        events = body.get("events", []) or []
        # TODO: paginate — when body["cursor"] is non-empty, follow it. One
        # page of up to 200 events is sufficient for v1 (index ladders have
        # only a handful of open expiries at a time).
        return list(events)

    def get_event(self, event_ticker: str, *, with_nested_markets: bool = True) -> dict[str, Any]:
        """Fetch a single event by ticker, with its nested-market ladder.

        Calls ``GET /events/{event_ticker}?with_nested_markets=…``.

        Args:
            event_ticker: Full event ticker, e.g. ``"KXINX-26MAY15H1600"``.
            with_nested_markets: Embed the per-event ``markets`` ladder.

        Returns:
            The ``event`` dict from the response.

        Raises:
            KalshiError: If the response carries no ``event`` object.
        """
        url = f"{self.BASE_URL}/events/{event_ticker}"
        params = {"with_nested_markets": str(bool(with_nested_markets)).lower()}
        r = self._request("GET", url, params=params)
        event = r.json().get("event")
        if not event:
            raise KalshiError(f"no event found for event_ticker={event_ticker!r}")
        return dict(event)


def fetch_factor_history(
    client: KalshiClient,
    market_ticker: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    series_ticker: str | None = None,
) -> pd.DataFrame:
    """Convenience wrapper matching the Polymarket source signature.

    Returns a DataFrame indexed by UTC date with at least a ``price`` column
    (and Kalshi's extras: volume, open_interest, yes_bid, yes_ask, spread).
    """
    # Kalshi REQUIRES start_ts and end_ts. Default to a wide window.
    if start is None:
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=730)
    if end is None:
        end = pd.Timestamp.now(tz="UTC")
    start_ts = int(
        start.tz_convert("UTC").timestamp()
        if start.tzinfo
        else start.tz_localize("UTC").timestamp()
    )
    end_ts = int(
        end.tz_convert("UTC").timestamp() if end.tzinfo else end.tz_localize("UTC").timestamp()
    )
    return client.get_candlesticks(
        market_ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        series_ticker=series_ticker,
    )


# ───────────────────────────────────────────────────────────────────────────
# Index-ladder discovery (S&P 500 / Nasdaq-100) — feeds the implied-PDF engine
# ───────────────────────────────────────────────────────────────────────────
#
# A Kalshi index event = one expiry; its nested ``markets`` form a strike/
# bucket ladder. We map that ladder into the shared ``LadderFamily`` contract
# (``pfm.vol.implied_pdf_schemas``) so the implied-PDF math engine can consume
# it uniformly across venues. Discovery is read-only and unauthenticated.

#: Friendly asset key → series options. ``buckets`` = ``between`` range-bucket
#: series (terminal mass); ``ladder`` = above/below threshold series (terminal
#: survival); ``yearly`` = EOY range buckets. ``KXINXMAXY``/``KXINXMINY``
#: (running-max/min barrier series) are intentionally NOT listed — out of scope.
INDEX_SERIES: dict[str, dict[str, Any]] = {
    "SPX": {
        "buckets": "KXINX",
        "ladder": "KXINXU",
        "yearly": "KXINXY",
        "asset_class": "equity_index",
    },
    "NDX": {
        "buckets": "KXNASDAQ100",
        "yearly": "KXNASDAQ100Y",
        "asset_class": "equity_index",
    },
}

#: Reverse map: any known series ticker → its friendly asset key. Lets callers
#: pass a raw series ticker ("KXINX") and still get a friendly ``asset`` label.
_SERIES_TO_ASSET: dict[str, str] = {
    series: asset
    for asset, opts in INDEX_SERIES.items()
    for k, series in opts.items()
    if k in ("buckets", "ladder", "yearly")
}

#: Month-code → month number for parsing event-ticker date codes like
#: ``KXINX-26MAY15H1600`` (2026-05-15 16:00 ET).
_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

# e.g. "26MAY15H1600" or "26MAY15" (the trailing H<time> is optional).
_EVENT_DATE_RE = re.compile(r"(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?:[A-Z](?P<hhmm>\d{4}))?")


def _to_float(value: Any) -> float | None:
    """Coerce a value to float, returning ``None`` on missing/garbage input."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _yes_bid_ask(market: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return the YES (bid, ask) in [0, 1], handling ``*_dollars`` and cents."""
    bid = _to_float(market.get("yes_bid_dollars"))
    ask = _to_float(market.get("yes_ask_dollars"))
    if bid is None and (cents := _to_float(market.get("yes_bid"))) is not None:
        bid = cents / 100.0
    if ask is None and (cents := _to_float(market.get("yes_ask"))) is not None:
        ask = cents / 100.0
    return bid, ask


def _yes_prob(market: dict[str, Any]) -> float | None:
    """Compute the YES probability for a market in [0, 1].

    Prefers the mid of bid/ask when both are present, else the last price.
    Handles both the current ``*_dollars`` (already in [0, 1]) and the legacy
    integer-cent (``yes_bid``/``yes_ask``/``last_price``, divide by 100) schemes.

    Returns:
        The YES probability, or ``None`` if no usable price is present.
    """
    # Current API: *_dollars fields already in [0, 1].
    bid = _to_float(market.get("yes_bid_dollars"))
    ask = _to_float(market.get("yes_ask_dollars"))
    last = _to_float(market.get("last_price_dollars"))
    # Legacy API: integer cents → divide by 100.
    if bid is None and (cents := _to_float(market.get("yes_bid"))) is not None:
        bid = cents / 100.0
    if ask is None and (cents := _to_float(market.get("yes_ask"))) is not None:
        ask = cents / 100.0
    if last is None and (cents := _to_float(market.get("last_price"))) is not None:
        last = cents / 100.0

    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if last is not None:
        return last
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return None


def _parse_maturity(market: dict[str, Any], event_ticker: str) -> datetime | None:
    """Best-effort UTC maturity for an event.

    Tries ``expected_expiration_time`` / ``close_time`` on a market first
    (ISO 8601), then falls back to the date code embedded in the event ticker
    (e.g. ``KXINX-26MAY15H1600`` → 2026-05-15 16:00 UTC). Always returns a
    timezone-aware UTC datetime, or ``None`` if nothing parses.
    """
    for key in ("expected_expiration_time", "close_time"):
        raw = market.get(key)
        if not raw:
            continue
        try:
            ts = pd.Timestamp(raw)
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()

    # Fall back to the event-ticker date code. NOTE: the H<hhmm> is ET wall
    # clock; we record it as UTC for provenance only (the engine recomputes
    # time-to-maturity). # TODO: convert ET→UTC if exact tenor matters.
    m = _EVENT_DATE_RE.search(event_ticker or "")
    if m:
        yy = int(m.group("yy"))
        mon = _MONTHS.get(m.group("mon"))
        dd = int(m.group("dd"))
        hhmm = m.group("hhmm")
        hh, mm = (int(hhmm[:2]), int(hhmm[2:])) if hhmm else (0, 0)
        if mon is not None:
            return datetime(2000 + yy, mon, dd, hh, mm, tzinfo=UTC)
    return None


def _event_sort_key(event: dict[str, Any]) -> datetime:
    """Sort key for choosing the *nearest* open event (earliest maturity)."""
    markets = event.get("markets") or []
    probe = markets[0] if markets else {}
    dt = _parse_maturity(probe, event.get("event_ticker", ""))
    return dt or datetime.max.replace(tzinfo=UTC)


def _event_matches_filter(event: dict[str, Any], maturity_filter: str) -> bool:
    """Whether an event's maturity ISO-date prefix matches ``maturity_filter``."""
    markets = event.get("markets") or []
    probe = markets[0] if markets else {}
    dt = _parse_maturity(probe, event.get("event_ticker", ""))
    if dt is None:
        return False
    return dt.date().isoformat().startswith(maturity_filter)


def discover_index_ladder(
    asset_or_series: str,
    client: KalshiClient,
    *,
    maturity_filter: str | None = None,
    prefer_shape: DataShape | None = None,
) -> LadderFamily:
    """Discover a Kalshi index market ladder as a :class:`LadderFamily`.

    Accepts either a friendly asset key (``"SPX"``, ``"NDX"``) or a raw Kalshi
    series ticker (``"KXINX"``). Fetches open events for the chosen series,
    selects one expiry (by ``maturity_filter`` or the nearest open event), and
    parses its nested markets into ``LadderEntry`` objects ready for the
    implied-PDF engine.

    Args:
        asset_or_series: Friendly key (``"SPX"``/``"NDX"``) or series ticker.
        client: A live :class:`KalshiClient`.
        maturity_filter: Optional ISO-date prefix (``"2026-05-15"``) selecting
            the expiry. If omitted, the nearest open event is used.
        prefer_shape: ``"terminal_buckets"`` forces the range-bucket series;
            ``"terminal_ladder"`` forces the above/below (``…U``) series.
            Defaults to the bucket series for the asset.

    Returns:
        A :class:`LadderFamily` (``spot`` left as ``None``).

    Raises:
        KalshiError: If the asset/series is unknown, no open event is found,
            or the chosen event has no usable markets.
    """
    key = asset_or_series.strip()
    opts = INDEX_SERIES.get(key.upper())

    if opts is not None:
        asset = key.upper()
        asset_class: AssetClass = opts.get("asset_class", "equity_index")
        if prefer_shape == "terminal_ladder":
            series = opts.get("ladder") or opts.get("buckets")
        elif prefer_shape == "terminal_buckets":
            series = opts.get("buckets")
        else:
            series = opts.get("buckets") or opts.get("ladder")
        if not series:
            raise KalshiError(f"asset {asset!r} has no series for prefer_shape={prefer_shape!r}")
    else:
        # Treat the input as a raw series ticker.
        series = key
        asset = _SERIES_TO_ASSET.get(series, series)
        asset_class = "equity_index"

    events = client.get_events(series, status="open", with_nested_markets=True)
    open_events = [e for e in events if (e.get("markets") or [])]
    if not open_events:
        raise KalshiError(f"no open events with markets found for series={series!r}")

    chosen: dict[str, Any] | None = None
    if maturity_filter:
        matches = [e for e in open_events if _event_matches_filter(e, maturity_filter)]
        if not matches:
            raise KalshiError(
                f"no open event matching maturity_filter={maturity_filter!r} for series={series!r}"
            )
        chosen = min(matches, key=_event_sort_key)
    else:
        chosen = min(open_events, key=_event_sort_key)

    event_ticker = str(chosen.get("event_ticker", series))
    markets = chosen.get("markets") or []

    entries: list[LadderEntry] = []
    n_between = 0
    n_threshold = 0
    skipped: list[str] = []
    maturity: datetime | None = None

    for mkt in markets:
        if maturity is None:
            maturity = _parse_maturity(mkt, event_ticker)

        prob = _yes_prob(mkt)
        if prob is None:
            skipped.append(str(mkt.get("ticker", "?")))
            continue

        strike_type = (mkt.get("strike_type") or "").lower()
        slug = mkt.get("ticker")
        floor = _to_float(mkt.get("floor_strike"))
        cap = _to_float(mkt.get("cap_strike"))
        # Raw two-sided quote + activity so a fair-value scanner can trade the
        # executable side and skip dead markets (Kalshi index dailies are often
        # untraded: bid 0, wide ask, no volume).
        yb, ya = _yes_bid_ask(mkt)
        quote = {
            "yes_bid": yb,
            "yes_ask": ya,
            "volume": _to_float(mkt.get("volume")),
            "open_interest": _to_float(mkt.get("open_interest")),
        }

        if strike_type == "between":
            if floor is None and cap is None:
                skipped.append(str(slug))
                continue
            entries.append(
                LadderEntry(
                    direction="between",
                    prob=prob,
                    floor=floor,
                    cap=cap,
                    slug=slug,
                    venue="kalshi",
                    **quote,
                )
            )
            n_between += 1
        elif strike_type in ("greater", "greater_or_equal"):
            entries.append(
                LadderEntry(
                    direction="above",
                    prob=prob,
                    strike=floor,
                    slug=slug,
                    venue="kalshi",
                    **quote,
                )
            )
            n_threshold += 1
        elif strike_type in ("less", "less_or_equal"):
            entries.append(
                LadderEntry(
                    direction="below",
                    prob=prob,
                    strike=cap,
                    slug=slug,
                    venue="kalshi",
                    **quote,
                )
            )
            n_threshold += 1
        else:
            # functional / custom / structured — not a clean strike/bucket.
            skipped.append(str(slug))

    if not entries:
        raise KalshiError(f"no usable markets in event={event_ticker!r} (series={series!r})")

    if maturity is None:
        maturity = _parse_maturity({}, event_ticker)
    if maturity is None:
        raise KalshiError(f"could not determine maturity for event={event_ticker!r}")

    # Bucket-dominant ladders are terminal mass; otherwise terminal survival.
    data_shape: DataShape = "terminal_buckets" if n_between >= n_threshold else "terminal_ladder"

    extra: dict[str, Any] = {
        "series_ticker": series,
        "event_ticker": event_ticker,
        "n_between": n_between,
        "n_threshold": n_threshold,
    }
    if skipped:
        extra["skipped_markets"] = skipped
        extra["note"] = (
            f"skipped {len(skipped)} market(s): no price or "
            "functional/custom/structured strike_type"
        )

    # spot: not provided by these endpoints.
    # TODO: derive spot from a reference market or external quote.
    return LadderFamily(
        asset=asset,
        asset_class=asset_class,
        data_shape=data_shape,
        maturity_utc=maturity,
        spot=None,
        entries=entries,
        source=f"kalshi:{event_ticker}",
        extra=extra,
    )
