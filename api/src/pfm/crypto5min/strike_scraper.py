"""Polymarket Chainlink-strike scraper.

Polymarket's gamma + CLOB APIs don't expose the "Price to Beat" (Chainlink
BTC/USD reference at the window start) as a structured field. The number
*is* present in the HTML of ``polymarket.com/event/<slug>`` — embedded in
the Next.js inline data block as ``eventMetadata.priceToBeat`` /
``eventMetadata.finalPrice`` for each event in the series.

For any active up-down market with start boundary ``S``, the strike is the
Chainlink BTC/USD price at exactly ``S``. We can recover that from the
HTML by:

* Each resolved 5m event has ``priceToBeat = Chainlink at its start_unix``
  and ``finalPrice = Chainlink at its end_unix``.
* finalPrice(event ending at t) == priceToBeat(event starting at t).
* For a 15m active market starting at S, S lies on a 5m boundary, so the
  same lookup works without scraping the 15m series separately.

We scrape once per ``SCRAPE_TTL_SECONDS`` and build an in-memory map
``chainlink_at_unix: {unix_timestamp -> price}``. Callers ask for the
strike of a given start_unix; we hit the map. When the scrape fails we
return ``None`` and the caller falls back to our Binance proxy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

POLYMARKET_BASE: str = "https://polymarket.com"
SCRAPE_TTL_SECONDS: float = 30.0
"""How long a single scrape stays valid. The Chainlink history doesn't
change — only new entries are appended every 5 min — so 30s is generous."""

#: The polymarket.com slug **prefix** for each (asset, window_minutes).
#: A concrete URL is built by appending the next 5/15-min boundary, e.g.
#: ``btc-updown-5m-1778772300``. Polymarket's series page itself (no end
#: unix suffix) returns 404, but a single recent event page embeds the
#: entire visible window of past events with their eventMetadata.
SLUG_PREFIX_BY_ASSET: dict[str, dict[int, str]] = {
    "BTC": {5: "btc-updown-5m", 15: "btc-updown-15m"},
    "ETH": {5: "eth-updown-5m", 15: "eth-updown-15m"},
}


@dataclass(frozen=True, slots=True)
class ChainlinkStrikes:
    """Snapshot of the Chainlink-at-time map for one (asset, window) series."""

    asset: str
    window_minutes: int
    fetched_at_unix: float
    by_unix: dict[int, float]

    def get(self, unix_ts: int) -> float | None:
        return self.by_unix.get(int(unix_ts))


_strike_cache: dict[tuple[str, int], ChainlinkStrikes] = {}

#: Per-(asset, window) single-flight locks. When four concurrent rows
#: (BTC/ETH × 5m/15m) all need the same polymarket.com HTML page on a cold
#: cache, only one fetch should fly; the others wait on the lock and read
#: the cached result. Cuts cold-start scrape cost from N×latency to ~1×.
_strike_locks: dict[tuple[str, int], asyncio.Lock] = {}


def _lock_for(key: tuple[str, int]) -> asyncio.Lock:
    lock = _strike_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _strike_locks[key] = lock
    return lock


def _series_url(asset: str, window_minutes: int, now_unix: float | None = None) -> str | None:
    """Pick a concrete event URL for one (asset, window).

    Polymarket's bare series page (``/event/btc-up-or-down-5m``) returns 404.
    Instead we fetch a *recent* event page; that page's Next.js inline state
    contains the full series (50+ events) with each one's eventMetadata.
    """
    prefix = SLUG_PREFIX_BY_ASSET.get(asset.upper(), {}).get(window_minutes)
    if prefix is None:
        return None
    now = time.time() if now_unix is None else float(now_unix)
    period = window_minutes * 60
    # Pick the most recently-resolved boundary (the one just before now) so
    # we don't depend on whether the next market exists yet.
    last_end = (int(now) // period) * period
    return f"{POLYMARKET_BASE}/event/{prefix}-{last_end}"


# Pattern: matches an event block's ticker + capture its eventMetadata block.
# We use a non-greedy match between ``ticker`` and the ``}}`` that closes
# the eventMetadata object. Limit search length to keep the regex fast.
_EVENT_BLOCK_RE = re.compile(
    r'"ticker":"((?:btc|eth)-updown-\d+m-(\d+))"'
    r"(?:[^{]|\{(?:[^{}]|\{[^{}]*\})*\})*?"
    r'"eventMetadata":(\{[^{}]*\})',
    re.DOTALL,
)


_START_TIME_RE = re.compile(r'"startTime":"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)"')
_END_DATE_RE = re.compile(r'"endDate":"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)"')
_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _iso_to_unix(iso: str) -> int | None:
    """Parse an ISO-Zulu timestamp into a unix integer. Tolerates fractional seconds."""
    # Strip fractional seconds — strptime can't handle them without a format change.
    if "." in iso:
        iso = re.sub(r"\.\d+Z$", "Z", iso)
    try:
        from datetime import UTC, datetime

        return int(datetime.strptime(iso, _ISO_FMT).replace(tzinfo=UTC).timestamp())
    except (ValueError, TypeError):
        return None


def parse_html_for_strikes(html: str) -> dict[int, float]:
    """Return ``{unix_ts -> chainlink_price}`` from a polymarket.com HTML page.

    The HTML embeds Next.js state for each event in the visible series. Each
    event JSON has fields like::

        "startTime": "2026-05-14T15:20:00Z",
        "endDate":   "2026-05-14T15:25:00Z",
        ...
        "eventMetadata": { "priceToBeat": 80994.33, "finalPrice": 81005.50 }

    We scan forward through every ``eventMetadata`` block and look BACKWARD
    a few hundred chars to grab the event's ``startTime`` (always present
    right before metadata). That's our unambiguous ``start_unix`` for the
    priceToBeat. ``endDate`` (also within the backward window) gives the
    ``end_unix`` for finalPrice.

    Both data points populate ``by_unix``. Since 5m and 15m boundaries
    share the same 5m grid, a single 5m page scrape covers 15m strikes too.
    """
    by_unix: dict[int, float] = {}
    metadata_re = re.compile(r'"eventMetadata":(\{[^{}]*\})')
    # Each event JSON is roughly <5 KB; metadata sits at the end, so the
    # event's startTime/endDate appear within ~1 KB before it.
    for md_match in metadata_re.finditer(html):
        try:
            md = json.loads(md_match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(md, dict):
            continue
        pt = md.get("priceToBeat")
        fp = md.get("finalPrice")
        if not isinstance(pt, int | float) and not isinstance(fp, int | float):
            continue
        backward_window = html[max(0, md_match.start() - 1500) : md_match.start()]
        # The LAST startTime/endDate before eventMetadata belongs to *this* event.
        starts = list(_START_TIME_RE.finditer(backward_window))
        ends = list(_END_DATE_RE.finditer(backward_window))
        start_unix = _iso_to_unix(starts[-1].group(1)) if starts else None
        end_unix = _iso_to_unix(ends[-1].group(1)) if ends else None
        if isinstance(pt, int | float) and start_unix is not None:
            by_unix.setdefault(start_unix, float(pt))
        if isinstance(fp, int | float) and end_unix is not None:
            by_unix.setdefault(end_unix, float(fp))
    return by_unix


async def fetch_strikes(
    client: httpx.AsyncClient,
    asset: str,
    window_minutes: int,
    *,
    timeout: float = 6.0,
) -> ChainlinkStrikes | None:
    """Fetch + parse the polymarket.com page for one (asset, window).

    Returns ``None`` on any HTTP error. Caches the parsed map for
    ``SCRAPE_TTL_SECONDS``.
    """
    key = (asset.upper(), int(window_minutes))
    now = time.time()
    cached = _strike_cache.get(key)
    if cached is not None and (now - cached.fetched_at_unix) < SCRAPE_TTL_SECONDS:
        return cached
    url = _series_url(asset, window_minutes)
    if url is None:
        return None
    # Single-flight: only one coroutine per (asset, window) issues the HTTP
    # request; others wait and re-check the cache on entry. This prevents
    # the four concurrent /compare rows from issuing duplicate page fetches
    # (BTC/15m falls back to BTC/5m which the BTC/5m row is already fetching).
    lock = _lock_for(key)
    async with lock:
        cached = _strike_cache.get(key)
        if cached is not None and (time.time() - cached.fetched_at_unix) < SCRAPE_TTL_SECONDS:
            return cached
        try:
            r = await client.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; pfm-bot/1.0)"},
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            logger.debug("polymarket strike scrape failed for %s: %s", url, exc)
            return None
        if r.status_code != 200 or not r.text:
            logger.debug("polymarket strike scrape returned HTTP %s for %s", r.status_code, url)
            return None
        by_unix = parse_html_for_strikes(r.text)
        if not by_unix:
            # Don't cache empty parse — likely a rendering change; retry on next call.
            return None
        snapshot = ChainlinkStrikes(
            asset=key[0],
            window_minutes=key[1],
            fetched_at_unix=time.time(),
            by_unix=by_unix,
        )
        _strike_cache[key] = snapshot
        return snapshot


async def get_strike_for_market(
    client: httpx.AsyncClient,
    *,
    asset: str,
    window_minutes: int,
    start_unix: int,
) -> tuple[float | None, str]:
    """Resolve the strike for one active market.

    Returns ``(price, source)`` where ``source`` is one of:
    * ``"polymarket-scrape"``  – exact priceToBeat from polymarket.com
    * ``"polymarket-scrape-prev"`` – finalPrice of the event ending at start_unix
    * ``"unavailable"``       – fell through; caller should use Binance proxy
    """
    snap = await fetch_strikes(client, asset, window_minutes)
    if snap is None:
        # Try cross-window (5m scrape covers 15m boundaries too — they're on
        # 5m grid). Force a 5m fetch for 15m markets when the 15m scrape misses.
        if window_minutes != 5:
            snap_5m = await fetch_strikes(client, asset, 5)
            if snap_5m is not None:
                price = snap_5m.get(start_unix)
                if price is not None:
                    return (price, "polymarket-scrape")
        return (None, "unavailable")
    price = snap.get(start_unix)
    if price is not None:
        return (price, "polymarket-scrape")
    # 15m windows align to 5m boundaries; the 5m series usually has it.
    if window_minutes != 5:
        snap_5m = await fetch_strikes(client, asset, 5)
        if snap_5m is not None:
            price = snap_5m.get(start_unix)
            if price is not None:
                return (price, "polymarket-scrape")
    return (None, "unavailable")


def _reset_cache() -> None:
    """Test hook."""
    _strike_cache.clear()
    _strike_locks.clear()
