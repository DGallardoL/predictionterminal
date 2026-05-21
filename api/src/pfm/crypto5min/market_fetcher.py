"""Discover & fetch active Polymarket short-dated crypto markets.

Slug pattern (verified live 2026-04-30): ``{asset}-updown-{window}-{end_unix}``
e.g. ``btc-updown-5m-1714589700``. The end_unix is the *next* multiple of the
window period after window-open. There's also the historical
``btc-up-or-down-...`` family which we treat as an alias.

This module is async + httpx-based — the request path that uses it
(``/strategies/crypto/5min/markets``) is already an async endpoint, and
making one network call sync would force a thread switch.

Public surface:

* ``discover_active_markets(client, assets, window_minutes)`` — returns a
  list of :class:`ActiveMarket` for every (asset, window) where Polymarket
  has an open market right now.
* ``fetch_clob_midpoint(client, token_id)`` — sync-style read of the live
  YES-token midpoint in [0, 1]. As of 2026-05-15 this is a *layered* fetch:
  it first checks an in-process WebSocket cache (populated by the leader
  worker's :class:`ClobMidpointSubscriber`), then Redis (followers read what
  the leader publishes), and only falls back to the REST ``/midpoint``
  endpoint when neither path has a fresh value. The signature is unchanged.
* ``parse_active_market(payload, slug, asset, window_minutes)`` — extract
  the typed dataclass from the raw Gamma response.
* ``ClobMidpointSubscriber`` — leader-only WebSocket subscriber. Maintains
  per-token midpoints from the CLOB ``book`` / ``price_change`` event
  stream and publishes them to Redis for follower workers.

The discovery sweep deliberately tries a *small* offset range (``[0..3]``
windows ahead) because Polymarket pre-creates the next few — past that
point we're guessing slugs that don't exist yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

DEFAULT_GAMMA_URL: str = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_URL: str = "https://clob.polymarket.com"
DEFAULT_CLOB_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

#: Redis key prefix for per-token midpoints published by the leader worker.
#: Follower workers read these to satisfy ``fetch_clob_midpoint`` without
#: hitting REST. TTL is short (see :data:`CLOB_WS_REDIS_TTL_S`) so a stale
#: entry from a dead leader auto-expires before it can mislead a follower.
CLOB_WS_REDIS_PREFIX: str = "pfm:clob_ws:midpoint:"

#: TTL on each published midpoint, in seconds. Picked at 10 s so that:
#:   * a normal stream (book or price_change every few seconds) keeps the
#:     entry warm,
#:   * a stalled leader (e.g. WS disconnect) lets entries expire and
#:     ``fetch_clob_midpoint`` falls back to REST after ~10 s rather than
#:     serving silently-stale data.
CLOB_WS_REDIS_TTL_S: int = 10

#: How fresh a cached midpoint (in-process OR Redis) has to be for
#: ``fetch_clob_midpoint`` to prefer it over a REST round-trip. Anything
#: older falls through to ``/midpoint``. 5 s is well under the Redis TTL so
#: we have headroom for clock skew between workers.
CLOB_WS_FRESH_WINDOW_S: float = 5.0

#: Supported asset → Binance symbol + slug-asset mapping. ETH coverage is
#: optional; Polymarket has had ``eth-updown-*`` markets in the past but
#: they cycle in/out. We always *try* to discover and silently skip ones
#: that don't exist.
SUPPORTED_ASSETS: dict[str, dict[str, str]] = {
    "BTC": {"binance_symbol": "BTCUSDT", "slug_asset": "btc"},
    "ETH": {"binance_symbol": "ETHUSDT", "slug_asset": "eth"},
}

#: How many future windows to probe per (asset, period). Polymarket
#: pre-creates 2-3 ahead; beyond that we'd just rack up 404s.
DISCOVERY_LOOKAHEAD: int = 3

#: Slug prefixes to try in order. The new form (``btc-updown-5m-...``)
#: superseded the long ``btc-up-or-down-...`` form in 2025; we keep both
#: so historical markets still resolve.
_SLUG_PREFIXES: list[str] = ["{asset}-updown-{window}m", "{asset}-up-or-down-{window}m"]


@dataclass(frozen=True, slots=True)
class ActiveMarket:
    """One open Polymarket short-dated up/down market.

    ``event_start_unix`` / ``event_end_unix`` come straight from the gamma
    response (``eventStartTime`` / ``endDate``). They're the authoritative
    boundary for the Chainlink reference price — slightly more accurate
    than the natural ``(now // period) * period`` heuristic when Polymarket
    delays market creation by a second or two.
    """

    asset: str
    binance_symbol: str
    window_minutes: int
    slug: str
    market_id: str | None
    up_token_id: str
    down_token_id: str
    start_unix: int
    end_unix: int
    seconds_remaining: float
    event_start_unix: int | None = None
    event_end_unix: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "binance_symbol": self.binance_symbol,
            "window_minutes": self.window_minutes,
            "slug": self.slug,
            "market_id": self.market_id,
            "up_token_id": self.up_token_id,
            "down_token_id": self.down_token_id,
            "start_unix": self.start_unix,
            "end_unix": self.end_unix,
            "seconds_remaining": self.seconds_remaining,
            "event_start_unix": self.event_start_unix,
            "event_end_unix": self.event_end_unix,
        }


def parse_active_market(
    payload: dict[str, Any],
    *,
    asset: str,
    window_minutes: int,
    binance_symbol: str,
    end_unix: int,
    now_unix: float | None = None,
) -> ActiveMarket | None:
    """Parse a Gamma ``/markets`` row into an :class:`ActiveMarket`.

    Returns ``None`` if the payload is malformed or the market is closed.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("closed") is True:
        return None
    raw_tokens = payload.get("clobTokenIds")
    if not raw_tokens:
        return None
    try:
        token_ids = json.loads(raw_tokens) if isinstance(raw_tokens, str) else list(raw_tokens)
    except (TypeError, ValueError):
        return None
    if not isinstance(token_ids, list) or len(token_ids) < 2:
        return None
    period = window_minutes * 60
    start_unix = end_unix - period
    now = time.time() if now_unix is None else float(now_unix)
    seconds_remaining = max(0.0, end_unix - now)
    slug = payload.get("slug")
    if not isinstance(slug, str) or not slug:
        return None
    return ActiveMarket(
        asset=asset,
        binance_symbol=binance_symbol,
        window_minutes=window_minutes,
        slug=slug,
        market_id=str(payload.get("id")) if payload.get("id") is not None else None,
        up_token_id=str(token_ids[0]),
        down_token_id=str(token_ids[1]),
        start_unix=start_unix,
        end_unix=end_unix,
        seconds_remaining=seconds_remaining,
    )


async def _gamma_get(
    client: httpx.AsyncClient,
    gamma_url: str,
    slug: str,
    timeout: float,
) -> dict[str, Any] | None:
    try:
        r = await client.get(
            f"{gamma_url.rstrip('/')}/markets",
            params={"slug": slug},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        logger.debug("gamma fetch failed for %s: %s", slug, exc)
        return None
    if r.status_code != 200:
        return None
    try:
        body = r.json()
    except ValueError:
        return None
    if isinstance(body, list) and body:
        return body[0] if isinstance(body[0], dict) else None
    if isinstance(body, dict) and body.get("data"):
        data = body["data"]
        return data[0] if isinstance(data, list) and data else None
    return None


async def discover_active_markets(
    client: httpx.AsyncClient,
    *,
    assets: list[str] | None = None,
    window_minutes_list: list[int] | None = None,
    gamma_url: str = DEFAULT_GAMMA_URL,
    now_unix: float | None = None,
    lookahead: int = DISCOVERY_LOOKAHEAD,
    timeout: float = 3.0,
) -> list[ActiveMarket]:
    """Sweep upcoming window boundaries and return every open market.

    All ``(asset × window × prefix × offset)`` candidate slugs are fired at
    Polymarket Gamma **in parallel** via :func:`asyncio.gather`. The first
    successful hit per (asset, window) — preferring lower offset, then the
    earlier prefix variant — wins. This brings the worst case (no markets
    active anywhere) from ~3.5s sequential to ~200ms parallel.

    With 2 assets × 2 windows × 2 prefixes × 3 offsets = 24 fanout calls
    per request, well under Polymarket's published gamma rate limit and
    safely within the httpx pool we configure in the lifespan.
    """
    assets = assets or ["BTC", "ETH"]
    window_minutes_list = window_minutes_list or [5, 15]
    now = time.time() if now_unix is None else float(now_unix)

    # Build every candidate slug up front, tagged with the metadata we need
    # to parse the response and pick the winner.
    candidates: list[dict[str, Any]] = []
    for asset in assets:
        meta = SUPPORTED_ASSETS.get(asset.upper())
        if meta is None:
            continue
        slug_asset = meta["slug_asset"]
        binance_symbol = meta["binance_symbol"]
        for window in window_minutes_list:
            period = window * 60
            next_end = ((int(now) // period) + 1) * period
            for offset in range(lookahead):
                end_unix = next_end + offset * period
                for prefix_rank, tmpl in enumerate(_SLUG_PREFIXES):
                    slug = f"{tmpl.format(asset=slug_asset, window=window)}-{end_unix}"
                    candidates.append(
                        {
                            "asset": asset.upper(),
                            "binance_symbol": binance_symbol,
                            "window": window,
                            "end_unix": end_unix,
                            "offset": offset,
                            "prefix_rank": prefix_rank,
                            "slug": slug,
                        }
                    )

    if not candidates:
        return []

    async def _probe(slug: str) -> dict[str, Any] | None:
        return await _gamma_get(client, gamma_url, slug, timeout)

    # Fire all gamma probes in parallel.
    payloads = await asyncio.gather(
        *(_probe(c["slug"]) for c in candidates),
        return_exceptions=False,
    )

    # First-hit-wins per (asset, window): lowest offset first, then lowest
    # prefix_rank. Sort the (candidate, payload) pairs by that priority and
    # pick the first valid payload per key.
    indexed = list(zip(candidates, payloads, strict=True))
    indexed.sort(
        key=lambda pair: (
            pair[0]["asset"],
            pair[0]["window"],
            pair[0]["offset"],
            pair[0]["prefix_rank"],
        )
    )

    out: list[ActiveMarket] = []
    seen_keys: set[tuple[str, int]] = set()
    for c, payload in indexed:
        key = (c["asset"], c["window"])
        if key in seen_keys:
            continue
        if payload is None:
            continue
        # Polymarket sometimes returns adjacent slugs even when filtering
        # by slug; only accept the exact match.
        if payload.get("slug") != c["slug"]:
            continue
        parsed = parse_active_market(
            payload,
            asset=c["asset"],
            window_minutes=c["window"],
            binance_symbol=c["binance_symbol"],
            end_unix=c["end_unix"],
            now_unix=now,
        )
        if parsed is not None and parsed.seconds_remaining > 0:
            out.append(parsed)
            seen_keys.add(key)

    return out


#: Process-wide singleton subscriber instance. Set by the lifespan hook in
#: ``pfm.main`` when ``PFM_CRYPTO_CLOB_WS_ENABLED=1`` and the worker wins
#: the SETNX leader election. Followers leave this ``None`` and read Redis
#: in :func:`fetch_clob_midpoint`.
_subscriber_singleton: ClobMidpointSubscriber | None = None


def set_subscriber(subscriber: ClobMidpointSubscriber | None) -> None:
    """Register (or clear) the process-wide WebSocket subscriber.

    Called from the FastAPI lifespan in ``pfm.main``. Tests can also use
    this to inject a fake subscriber. Pass ``None`` to detach.
    """
    global _subscriber_singleton
    _subscriber_singleton = subscriber


def get_subscriber() -> ClobMidpointSubscriber | None:
    """Return the process-wide subscriber, or ``None`` if not running."""
    return _subscriber_singleton


async def fetch_clob_midpoint(
    client: httpx.AsyncClient,
    token_id: str,
    *,
    clob_url: str = DEFAULT_CLOB_URL,
    timeout: float = 2.5,
    redis_client: Any | None = None,
    now_unix: float | None = None,
) -> float | None:
    """Fetch the CLOB midpoint for one token. Returns None on any error.

    Lookup order (each step is non-blocking and falls through on miss):

    1. **In-process WebSocket cache** — set by the leader worker's
       :class:`ClobMidpointSubscriber`. Hit ratio is ~100 % on the leader
       once the token has been subscribed for >1 s.
    2. **Redis cache** — populated by the leader for cross-worker reads.
       Followers depend on this path. Key is
       ``pfm:clob_ws:midpoint:{token_id}`` with TTL
       :data:`CLOB_WS_REDIS_TTL_S`.
    3. **REST fallback** — the original ``/midpoint?token_id=…`` call.
       Used when WS is disabled, the token isn't yet subscribed, or the
       last update is stale (>:data:`CLOB_WS_FRESH_WINDOW_S` s old).

    The signature is intentionally backward-compatible: existing callers
    pass ``(client, token_id)`` and get the same ``float | None`` they
    always did. The new ``redis_client`` parameter is opt-in for callers
    that want to enable the follower path explicitly; when ``None``, the
    function only consults the leader's in-process cache (which is correct
    for the leader and a graceful no-op for followers without a Redis
    handle, which then fall through to REST).
    """
    if not token_id:
        return None
    now = time.time() if now_unix is None else now_unix

    # 1. In-process WS cache (leader worker).
    sub = _subscriber_singleton
    if sub is not None:
        cached = sub.get_midpoint(token_id, now_unix=now)
        if cached is not None:
            return cached

    # 2. Redis cache (follower worker; populated by the leader).
    if redis_client is not None:
        try:
            raw = redis_client.get(CLOB_WS_REDIS_PREFIX + token_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("clob ws redis read failed for %s: %s", token_id, exc)
            raw = None
        if raw is not None:
            parsed = _parse_redis_midpoint(raw, now_unix=now)
            if parsed is not None:
                return parsed

    # 3. REST fallback — the original code path, unchanged.
    try:
        r = await client.get(
            f"{clob_url.rstrip('/')}/midpoint",
            params={"token_id": token_id},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        logger.debug("clob midpoint fetch failed for %s: %s", token_id, exc)
        return None
    if r.status_code != 200:
        return None
    try:
        body = r.json()
    except ValueError:
        return None
    raw_mid = body.get("mid") if isinstance(body, dict) else None
    if raw_mid is None:
        return None
    try:
        mid = float(raw_mid)
    except (TypeError, ValueError):
        return None
    if mid < 0 or mid > 1:
        return None
    return mid


def _parse_redis_midpoint(raw: Any, *, now_unix: float) -> float | None:
    """Parse the JSON blob the subscriber writes to Redis.

    Format: ``{"mid": 0.5123, "ts": 1714589700.45}``. Returns ``None`` when
    the blob is malformed, the midpoint is out of [0, 1], or the timestamp
    is older than :data:`CLOB_WS_FRESH_WINDOW_S`.
    """
    try:
        if isinstance(raw, bytes | bytearray):
            raw = raw.decode("utf-8")
        body = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    try:
        mid = float(body.get("mid"))
        ts = float(body.get("ts"))
    except (TypeError, ValueError):
        return None
    if mid < 0 or mid > 1:
        return None
    if now_unix - ts > CLOB_WS_FRESH_WINDOW_S:
        return None
    return mid


async def fetch_binance_price_at(
    client: httpx.AsyncClient,
    symbol: str,
    unix_ts: int,
    *,
    binance_base_url: str = "https://api.binance.com",
    timeout: float = 3.0,
    window_seconds: int = 2,
) -> float | None:
    """Binance trade-weighted price within a 1-2s window starting at ``unix_ts``.

    Uses ``/api/v3/aggTrades`` — Binance's compact trade tape — so we get
    the *actual* trades executed at the boundary instant. Returns the
    volume-weighted average price of all trades in the window, or ``None``
    when no trades fall inside or the API errors out.

    This is the cleanest fallback for the priceToBeat when Polymarket's HTML
    hasn't recorded it yet (typical lag: 5-15 min after the boundary).
    Chainlink BTC/USD's median across exchanges almost always matches Binance
    to within ~$5 for BTC, less for ETH.
    """
    start_ms = unix_ts * 1000
    end_ms = start_ms + window_seconds * 1000
    try:
        r = await client.get(
            f"{binance_base_url.rstrip('/')}/api/v3/aggTrades",
            params={
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 500,
            },
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        logger.debug("binance aggTrades fetch failed for %s @ %d: %s", symbol, unix_ts, exc)
        return None
    if r.status_code != 200:
        return None
    try:
        rows = r.json()
    except ValueError:
        return None
    if not isinstance(rows, list) or not rows:
        return None
    total_qty = 0.0
    total_notional = 0.0
    for trade in rows:
        try:
            p = float(trade["p"])
            q = float(trade["q"])
        except (KeyError, TypeError, ValueError):
            continue
        if p <= 0 or q <= 0:
            continue
        total_qty += q
        total_notional += p * q
    if total_qty <= 0:
        return None
    return total_notional / total_qty


async def fetch_binance_mid(
    client: httpx.AsyncClient,
    symbol: str,
    *,
    binance_base_url: str = "https://api.binance.com",
    timeout: float = 2.5,
) -> float | None:
    """Fetch the latest Binance bookTicker midpoint for one Binance symbol."""
    try:
        r = await client.get(
            f"{binance_base_url.rstrip('/')}/api/v3/ticker/bookTicker",
            params={"symbol": symbol},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        logger.debug("binance bookTicker fetch failed for %s: %s", symbol, exc)
        return None
    if r.status_code != 200:
        return None
    try:
        body = r.json()
    except ValueError:
        return None
    try:
        bid = float(body["bidPrice"])
        ask = float(body["askPrice"])
    except (KeyError, TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


# ---------------------------------------------------------------------------
# Polymarket CLOB WebSocket subscriber
# ---------------------------------------------------------------------------
#
# Protocol (verified live 2026-05-15 against
# wss://ws-subscriptions-clob.polymarket.com/ws/market):
#
#   * Subscribe:  ``{"type":"Market","assets_ids":["<token_id>", ...]}``
#   * First message after subscribe is a LIST of ``book`` events — one per
#     subscribed asset — each containing the full L2 snapshot:
#
#         {
#           "event_type": "book",
#           "asset_id":   "...",
#           "market":     "0x...",
#           "bids":       [{"price":"0.51","size":"..."}, ...],
#           "asks":       [{"price":"0.52","size":"..."}, ...],
#           "timestamp":  "1778872752029",
#           ...
#         }
#
#     IMPORTANT: ``bids`` is in ASCENDING price order on the wire (best bid
#     is the LAST entry, not the first). We take ``max(prices)`` for bids
#     and ``min(prices)`` for asks rather than indexing [0].
#
#   * Subsequent updates are ``price_change`` events (single dict, not a
#     list) containing ``price_changes`` — one entry per touched
#     (asset_id × side). Each carries the post-trade ``best_bid`` and
#     ``best_ask`` for that asset; we use those directly instead of
#     replaying the book delta.
#
# Other event types (``last_trade_price``, ``tick_size_change``) are
# ignored — they don't move the midpoint.


class _SubscriberCacheProtocol(Protocol):
    """Subset of ``pfm.cache.RedisCache`` we use for publishing midpoints.

    Defined as a Protocol so tests can swap in a fake (``NullCache``,
    ``fakeredis``-backed, or a tiny stub) without inheriting our concrete
    type.
    """

    @property
    def enabled(self) -> bool: ...


class ClobMidpointSubscriber:
    """Subscribes to Polymarket CLOB and maintains per-token midpoints.

    Designed to be created at boot by *exactly one* gunicorn worker (the
    leader) so we hold a single WebSocket connection upstream rather than
    one per worker. Followers read the published midpoints from Redis (see
    :func:`fetch_clob_midpoint`).

    The instance can be reused across many tokens; call :meth:`add_tokens`
    /:meth:`remove_tokens` as the active-market set rotates. A periodic
    rotation task (started by :meth:`start` if ``rotate_callable`` is
    provided) handles this automatically.
    """

    def __init__(
        self,
        *,
        url: str = DEFAULT_CLOB_WS_URL,
        cache: _SubscriberCacheProtocol | None = None,
        redis_client: Any | None = None,
        rotate_callable: Any | None = None,
        rotate_interval_s: float = 60.0,
        max_subscriptions: int = 40,
        min_backoff_s: float = 0.5,
        max_backoff_s: float = 30.0,
        redis_ttl_s: int = CLOB_WS_REDIS_TTL_S,
        connect_factory: Any | None = None,
    ) -> None:
        self.url = url
        # ``cache`` is the high-level wrapper (``RedisCache``) we got from
        # ``app.state.cache``. We need its underlying ``_client`` for SET
        # with PX so the publish path is non-blocking. Tests can pass
        # ``redis_client=`` directly to bypass.
        self._cache = cache
        self._redis = redis_client or getattr(cache, "_client", None)
        self._rotate = rotate_callable
        self._rotate_interval_s = rotate_interval_s
        self._max_subs = max_subscriptions
        self._min_backoff = min_backoff_s
        self._max_backoff = max_backoff_s
        self._redis_ttl_s = redis_ttl_s
        self._connect_factory = connect_factory

        # Per-token state: ``token_id -> (last_update_unix, midpoint)``.
        self._midpoints: dict[str, tuple[float, float]] = {}
        # Tokens we've told the server to send us. We re-send the full set
        # after every reconnect.
        self._subscribed: set[str] = set()
        # Pending mutations to apply between recv() iterations. WS protocol
        # requires re-subscribing for new asset_ids; on disconnect we
        # rebuild from ``_subscribed``.
        self._pending_add: set[str] = set()
        self._pending_remove: set[str] = set()
        self._mutation_event = asyncio.Event()

        self._task: asyncio.Task[None] | None = None
        self._rotate_task: asyncio.Task[None] | None = None
        self._closed = False
        self._connected_at: float | None = None

    # ----- public API ------------------------------------------------------

    def add_tokens(self, token_ids: list[str] | set[str]) -> None:
        """Subscribe to additional token IDs on the next reconnect/cycle."""
        new = {t for t in token_ids if t and t not in self._subscribed}
        if not new:
            return
        if len(self._subscribed) + len(new) > self._max_subs:
            # Trim oldest from the current set so we don't exceed the cap.
            keep = max(0, self._max_subs - len(new))
            self._subscribed = set(list(self._subscribed)[-keep:]) if keep else set()
            logger.info(
                "clob ws subscriber: trimmed subscription set to %d to admit %d new",
                keep,
                len(new),
            )
        self._pending_add |= new
        self._mutation_event.set()

    def remove_tokens(self, token_ids: list[str] | set[str]) -> None:
        """Stop tracking the given token IDs (drops cached midpoints too)."""
        drop = {t for t in token_ids if t}
        if not drop:
            return
        self._pending_remove |= drop
        for t in drop:
            self._midpoints.pop(t, None)
        self._mutation_event.set()

    def replace_tokens(self, token_ids: list[str] | set[str]) -> None:
        """Atomically set the subscription set to exactly ``token_ids``."""
        wanted = {t for t in token_ids if t}
        # Cap at max_subs to avoid unbounded growth from a runaway caller.
        if len(wanted) > self._max_subs:
            wanted = set(list(wanted)[: self._max_subs])
        current = self._subscribed | self._pending_add - self._pending_remove
        to_add = wanted - current
        to_remove = current - wanted
        if to_add:
            self.add_tokens(to_add)
        if to_remove:
            self.remove_tokens(to_remove)

    def get_midpoint(self, token_id: str, *, now_unix: float | None = None) -> float | None:
        """Return the freshest known midpoint for ``token_id``, or ``None``.

        Returns ``None`` when the token isn't tracked OR when the last
        update is older than :data:`CLOB_WS_FRESH_WINDOW_S` (stale data
        would silently mislead the fallback chain).
        """
        entry = self._midpoints.get(token_id)
        if entry is None:
            return None
        ts, mid = entry
        now = time.time() if now_unix is None else now_unix
        if now - ts > CLOB_WS_FRESH_WINDOW_S:
            return None
        return mid

    def snapshot(self) -> dict[str, tuple[float, float]]:
        """Return a copy of the current ``{token_id: (ts, mid)}`` map.

        Intended for diagnostics / tests. The caller gets a shallow copy
        so they can mutate without affecting the subscriber.
        """
        return dict(self._midpoints)

    async def start(self) -> None:
        """Spawn the consumer + (optional) rotator tasks. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._closed = False
        self._task = asyncio.create_task(self._run(), name="clob-ws-subscriber")
        if self._rotate is not None and self._rotate_task is None:
            self._rotate_task = asyncio.create_task(
                self._rotation_loop(),
                name="clob-ws-rotation",
            )

    async def stop(self) -> None:
        """Tear down the consumer + rotator tasks. Idempotent."""
        self._closed = True
        self._mutation_event.set()
        if self._rotate_task is not None:
            self._rotate_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._rotate_task
            self._rotate_task = None
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ----- event handling --------------------------------------------------

    def ingest_message(self, msg: Any, *, now_unix: float | None = None) -> list[str]:
        """Parse one JSON message; update cache; return touched token IDs.

        Public so tests can feed synthetic payloads without running a
        WebSocket. Returns the list of asset_ids that had their midpoint
        updated (useful for asserting Redis publishes in tests).
        """
        now = time.time() if now_unix is None else now_unix
        touched: list[str] = []
        items = msg if isinstance(msg, list) else [msg]
        for item in items:
            if not isinstance(item, dict):
                continue
            et = item.get("event_type")
            if et == "book":
                tid = item.get("asset_id")
                if not isinstance(tid, str):
                    continue
                mid = _midpoint_from_book(item)
                if mid is None:
                    continue
                self._midpoints[tid] = (now, mid)
                touched.append(tid)
            elif et == "price_change":
                # ``price_changes`` is a list keyed by (asset_id, side);
                # each entry already includes ``best_bid``/``best_ask`` for
                # that token, so we don't need to replay the book delta.
                changes = item.get("price_changes")
                if not isinstance(changes, list):
                    continue
                # Multiple side-flips for the same asset can land in one
                # message; only the last one's best_bid/ask matters.
                latest_per_asset: dict[str, tuple[float, float]] = {}
                for ch in changes:
                    if not isinstance(ch, dict):
                        continue
                    tid = ch.get("asset_id")
                    if not isinstance(tid, str):
                        continue
                    try:
                        bb = float(ch.get("best_bid"))
                        ba = float(ch.get("best_ask"))
                    except (TypeError, ValueError):
                        continue
                    if bb <= 0 or ba <= 0 or bb >= ba:
                        # Either crossed book (bb >= ba) or invalid prices
                        # — ignore. Polymarket has occasionally pushed
                        # mid-update zero placeholders.
                        continue
                    latest_per_asset[tid] = (bb, ba)
                for tid, (bb, ba) in latest_per_asset.items():
                    mid = (bb + ba) / 2.0
                    if 0.0 <= mid <= 1.0:
                        self._midpoints[tid] = (now, mid)
                        touched.append(tid)
            # other event_types (last_trade_price, tick_size_change) — ignored.
        if touched and self._redis is not None:
            self._publish(touched, now_unix=now)
        return touched

    def _publish(self, token_ids: list[str], *, now_unix: float) -> None:
        """Push the freshest midpoint for each token to Redis with TTL."""
        if self._redis is None:
            return
        for tid in token_ids:
            entry = self._midpoints.get(tid)
            if entry is None:
                continue
            ts, mid = entry
            blob = json.dumps({"mid": mid, "ts": ts})
            try:
                self._redis.set(
                    CLOB_WS_REDIS_PREFIX + tid,
                    blob,
                    ex=self._redis_ttl_s,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("clob ws redis publish failed for %s: %s", tid, exc)
                return  # bail on first failure; next message will retry

    # ----- background loops ------------------------------------------------

    async def _run(self) -> None:
        attempt = 0
        # Lazy import so the module is importable without ``websockets``.
        try:
            import websockets  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - install guard
            logger.warning("clob ws subscriber: websockets unavailable: %s", exc)
            return
        connect = self._connect_factory or websockets.connect
        while not self._closed:
            try:
                async with connect(
                    self.url,
                    ping_interval=20.0,
                    ping_timeout=20.0,
                    close_timeout=5.0,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self._connected_at = time.time()
                    attempt = 0  # reset backoff once we have a live socket
                    # Drain any pending mutations into the live set BEFORE
                    # subscribing so we don't immediately need to re-subscribe.
                    self._subscribed |= self._pending_add
                    self._subscribed -= self._pending_remove
                    self._pending_add.clear()
                    self._pending_remove.clear()
                    if self._subscribed:
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "Market",
                                    "assets_ids": sorted(self._subscribed),
                                }
                            )
                        )
                    # Concurrent recv + mutation pump. Mutation pump only
                    # wakes when add/remove_tokens fired ``_mutation_event``.
                    recv_task = asyncio.create_task(self._recv_loop(ws))
                    mut_task = asyncio.create_task(self._mutation_loop(ws))
                    try:
                        done, _pending = await asyncio.wait(
                            {recv_task, mut_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        for t in (recv_task, mut_task):
                            if not t.done():
                                t.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await t
                    # Surface the recv task's exception (if any) to drive
                    # the reconnect branch below.
                    for t in done:
                        exc = t.exception()
                        if exc is not None:
                            raise exc
                    # Clean close (e.g. ``stop()`` cancelled us mid-recv).
                    if self._closed:
                        return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._closed:
                    return
                attempt += 1
                delay = min(
                    self._min_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.5),
                    self._max_backoff,
                )
                logger.warning(
                    "clob ws subscriber reconnecting (attempt %d, delay=%.1fs): %s",
                    attempt,
                    delay,
                    exc,
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return

    async def _recv_loop(self, ws: Any) -> None:
        async for raw in ws:
            if self._closed:
                return
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            self.ingest_message(msg)

    async def _mutation_loop(self, ws: Any) -> None:
        """Wake on add/remove_tokens; re-send a fresh subscribe payload.

        Polymarket's market channel doesn't expose a per-token "unsubscribe"
        op, so the cleanest way to drop tokens is to send a new ``Market``
        message with the wanted set — the server replaces the prior list.
        """
        while not self._closed:
            await self._mutation_event.wait()
            self._mutation_event.clear()
            if self._closed:
                return
            # Apply pending mutations into the canonical set.
            self._subscribed |= self._pending_add
            self._subscribed -= self._pending_remove
            # Drop midpoints for removed tokens so a follower's Redis read
            # doesn't keep serving them past the TTL.
            for tid in self._pending_remove:
                self._midpoints.pop(tid, None)
            self._pending_add.clear()
            self._pending_remove.clear()
            try:
                await ws.send(
                    json.dumps(
                        {
                            "type": "Market",
                            "assets_ids": sorted(self._subscribed),
                        }
                    )
                )
            except Exception as exc:
                # Will trigger reconnect via the recv loop bailing too.
                logger.debug("clob ws subscriber: mutation send failed: %s", exc)
                return

    async def _rotation_loop(self) -> None:
        """Periodically refresh the subscription set via ``rotate_callable``.

        The callable is expected to return the current list of active
        token IDs (BTC/ETH × 5m/15m × 2-3 horizons), typically by hitting
        ``/strategies/crypto/5min/markets`` or a precomputed cache.
        """
        if self._rotate is None:
            return
        while not self._closed:
            try:
                await asyncio.sleep(self._rotate_interval_s)
                got = self._rotate()
                if asyncio.iscoroutine(got) or asyncio.isfuture(got):
                    got = await got  # type: ignore[assignment]
                if isinstance(got, list | set):
                    self.replace_tokens(got)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug("clob ws subscriber: rotation tick failed: %s", exc)


def _midpoint_from_book(book: dict[str, Any]) -> float | None:
    """Compute ``(best_bid + best_ask) / 2`` from a raw ``book`` event.

    Polymarket sends bids and asks as lists of ``{"price": "...", "size":
    "..."}``. The wire-order is ASCENDING price for both, so the best bid
    is ``max(bids)`` and the best ask is ``min(asks)``. We tolerate either
    order by computing min/max explicitly rather than indexing.
    """
    bids = book.get("bids") or book.get("buys")
    asks = book.get("asks") or book.get("sells")
    if not isinstance(bids, list) or not isinstance(asks, list):
        return None
    bid_prices: list[float] = []
    ask_prices: list[float] = []
    for row in bids:
        if not isinstance(row, dict):
            continue
        try:
            p = float(row.get("price"))
            s = float(row.get("size", 0) or 0)
        except (TypeError, ValueError):
            continue
        if p <= 0 or s <= 0:
            continue
        bid_prices.append(p)
    for row in asks:
        if not isinstance(row, dict):
            continue
        try:
            p = float(row.get("price"))
            s = float(row.get("size", 0) or 0)
        except (TypeError, ValueError):
            continue
        if p <= 0 or s <= 0:
            continue
        ask_prices.append(p)
    if not bid_prices or not ask_prices:
        return None
    best_bid = max(bid_prices)
    best_ask = min(ask_prices)
    if best_bid >= best_ask:
        # Crossed book — shouldn't happen on Polymarket but be defensive.
        return None
    mid = (best_bid + best_ask) / 2.0
    if mid < 0 or mid > 1:
        return None
    return mid
