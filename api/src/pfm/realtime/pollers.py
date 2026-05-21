"""Per-kind poll functions used by the realtime hub.

Each ``poll_*`` coroutine resolves the slug to a Polymarket YES ``token_id``
(cached process-wide in :data:`_TOKEN_CACHE`) and issues one CLOB request
to produce a single dict payload. The hub calls these on a fixed cadence
and fans the result out to every subscribed client.

These are intentionally tiny: error handling, sleeping, and fanout all
live in :class:`pfm.realtime.hub.RealtimeHub`.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any, Final

import httpx

# Endpoints — overridable at runtime via ``set_endpoints`` (mostly for tests).
_GAMMA_URL: str = "https://gamma-api.polymarket.com"
_CLOB_URL: str = "https://clob.polymarket.com"

# Process-wide token-id cache. Slug→token resolution is invariant for the
# market's lifetime, so caching avoids re-hitting Gamma every poll cycle.
_TOKEN_CACHE: dict[str, str] = {}
_TOKEN_CACHE_LOCK: asyncio.Lock | None = None  # lazily created in event loop

# Sentinel used to remember "this slug doesn't resolve" so we don't hammer
# Gamma on every poll for a bad slug.
_UNRESOLVABLE: Final[str] = "__unresolvable__"


def set_endpoints(gamma_url: str | None = None, clob_url: str | None = None) -> None:
    """Override the Polymarket base URLs (test hook)."""
    global _GAMMA_URL, _CLOB_URL
    if gamma_url is not None:
        _GAMMA_URL = gamma_url.rstrip("/")
    if clob_url is not None:
        _CLOB_URL = clob_url.rstrip("/")


def _get_lock() -> asyncio.Lock:
    global _TOKEN_CACHE_LOCK
    if _TOKEN_CACHE_LOCK is None:
        _TOKEN_CACHE_LOCK = asyncio.Lock()
    return _TOKEN_CACHE_LOCK


def clear_token_cache() -> None:
    """Drop the slug→token-id cache (test hook)."""
    _TOKEN_CACHE.clear()


async def resolve_yes_token_id(slug: str, http: httpx.AsyncClient) -> str | None:
    """Resolve ``slug`` → YES ``clobTokenIds[0]``. Cached. ``None`` on failure.

    The Gamma payload encodes ``clobTokenIds`` as a JSON string inside JSON
    (a documented Polymarket quirk), so it's parsed twice effectively.
    """
    cached = _TOKEN_CACHE.get(slug)
    if cached == _UNRESOLVABLE:
        return None
    if cached is not None:
        return cached

    async with _get_lock():
        # Double-check after acquiring lock — another coroutine may have
        # populated the cache while we were waiting.
        cached = _TOKEN_CACHE.get(slug)
        if cached == _UNRESOLVABLE:
            return None
        if cached is not None:
            return cached

        token = await _fetch_yes_token_id(slug, http)
        _TOKEN_CACHE[slug] = token if token is not None else _UNRESOLVABLE
        return token


async def _fetch_yes_token_id(slug: str, http: httpx.AsyncClient) -> str | None:
    try:
        r = await http.get(f"{_GAMMA_URL}/markets", params={"slug": slug})
        r.raise_for_status()
        payload = r.json()
    except (httpx.HTTPError, ValueError):
        return None

    market = payload[0] if isinstance(payload, list) and payload else None
    if not market:
        return None
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    try:
        ids = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(ids, list) or not ids:
        return None
    return str(ids[0])


# ---- kind=tick -------------------------------------------------------------


async def poll_tick(slug: str, http: httpx.AsyncClient) -> dict | None:
    """Mid/bid/ask snapshot. ``None`` if the slug can't be resolved."""
    token_id = await resolve_yes_token_id(slug, http)
    if token_id is None:
        return None

    mid_task = _fetch_midpoint(token_id, http)
    bid_task = _fetch_side_price(token_id, "SELL", http)
    ask_task = _fetch_side_price(token_id, "BUY", http)
    mid, bid, ask = await asyncio.gather(mid_task, bid_task, ask_task)
    return {
        "type": "tick",
        "slug": slug,
        "data": {"mid": mid, "bid": bid, "ask": ask},
        "ts": int(time.time()),
    }


async def _fetch_midpoint(token_id: str, http: httpx.AsyncClient) -> float | None:
    try:
        r = await http.get(f"{_CLOB_URL}/midpoint", params={"token_id": token_id})
        r.raise_for_status()
        payload = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    raw = payload.get("mid") if isinstance(payload, dict) else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


async def _fetch_side_price(token_id: str, side: str, http: httpx.AsyncClient) -> float | None:
    try:
        r = await http.get(
            f"{_CLOB_URL}/price",
            params={"token_id": token_id, "side": side},
        )
        r.raise_for_status()
        payload = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    raw = payload.get("price") if isinstance(payload, dict) else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


# ---- kind=book -------------------------------------------------------------


async def poll_book(slug: str, http: httpx.AsyncClient) -> dict | None:
    """L1 orderbook snapshot (top-of-book bid/ask + sizes)."""
    token_id = await resolve_yes_token_id(slug, http)
    if token_id is None:
        return None
    try:
        r = await http.get(f"{_CLOB_URL}/book", params={"token_id": token_id})
        r.raise_for_status()
        payload = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    bids = payload.get("bids") if isinstance(payload, dict) else None
    asks = payload.get("asks") if isinstance(payload, dict) else None
    return {
        "type": "book",
        "slug": slug,
        "data": {
            "bids": _top_levels(bids, n=5),
            "asks": _top_levels(asks, n=5),
        },
        "ts": int(time.time()),
    }


def _top_levels(levels: object, *, n: int) -> list[dict]:
    if not isinstance(levels, list):
        return []
    out: list[dict] = []
    for lvl in levels[:n]:
        if not isinstance(lvl, dict):
            continue
        raw_price = lvl.get("price")
        raw_size = lvl.get("size")
        if raw_price is None or raw_size is None:
            continue
        try:
            price = float(raw_price)
            size = float(raw_size)
        except (TypeError, ValueError):
            continue
        out.append({"price": price, "size": size})
    return out


# ---- kind=tape -------------------------------------------------------------


async def poll_tape(slug: str, http: httpx.AsyncClient) -> dict | None:
    """Recent-trades tape snapshot (last N fills)."""
    token_id = await resolve_yes_token_id(slug, http)
    if token_id is None:
        return None
    try:
        r = await http.get(
            f"{_CLOB_URL}/trades",
            params={"market": token_id, "limit": 20},
        )
        r.raise_for_status()
        payload = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    trades = payload if isinstance(payload, list) else []
    fills: list[dict] = []
    for t in trades[:20]:
        if not isinstance(t, dict):
            continue
        raw_price = t.get("price")
        raw_size = t.get("size")
        if raw_price is None or raw_size is None:
            continue
        try:
            price = float(raw_price)
            size = float(raw_size)
        except (TypeError, ValueError):
            continue
        fills.append(
            {
                "price": price,
                "size": size,
                "side": str(t.get("side") or ""),
                "ts": int(t.get("timestamp") or 0),
            }
        )
    return {
        "type": "tape",
        "slug": slug,
        "data": {"fills": fills},
        "ts": int(time.time()),
    }


# ---- registry --------------------------------------------------------------


POLLERS: dict[str, Callable[..., Any]] = {
    "tick": poll_tick,
    "book": poll_book,
    "tape": poll_tape,
}
"""Registry: ``kind`` → async ``(slug, http) -> dict | None``."""


SUPPORTED_KINDS: frozenset[str] = frozenset(POLLERS.keys())
