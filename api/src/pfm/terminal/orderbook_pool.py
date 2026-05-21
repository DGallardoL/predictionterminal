"""Shared CLOB orderbook pool with per-token-id TTL cache.

Many Terminal panels (quote, orderbook ladder, quality-score, arb scanner,
peer scanner) all hit ``GET https://clob.polymarket.com/book?token_id=...``
within a few hundred milliseconds of each other when a market is opened or
a multi-leg page renders. Today each call site builds (or borrows) its
own ``httpx`` client and re-fetches the same book — wasted RTT and wasted
upstream quota.

This module provides a tiny **process-wide singleton** that:

1. Reuses :pyclass:`pfm.sources.polymarket_pool.PolymarketHTTPPool`'s shared
   CLOB client (HTTP/2 + keep-alive).
2. Caches book snapshots by ``token_id`` with a configurable TTL (default
   30 s). The default is intentionally short — books change fast — but
   *long enough* to coalesce a single page render's 3-5 calls into one.
3. Exposes a batched :meth:`get_snapshots` that fans out fresh fetches in
   parallel with a bounded concurrency (Semaphore=10) so a 50-leg
   portfolio doesn't open 50 sockets at once.

Snapshot shape::

    {
        "bids": [[price, size], ...],
        "asks": [[price, size], ...],
        "updated_at": "<ISO8601 UTC>",
    }

Errors (network failures, HTTP 5xx, 404 for unknown tokens) are logged at
WARNING and surface as ``None`` from :meth:`get_snapshot`. Callers decide
whether to treat that as "skip this leg" or "fall back to last-known".
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import ClassVar

import httpx

from pfm.sources.polymarket_pool import PolymarketHTTPPool

__all__ = ["BATCH_CONCURRENCY", "DEFAULT_MAX_AGE_S", "OrderbookPool"]

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE_S: int = 30
BATCH_CONCURRENCY: int = 10


class OrderbookPool:
    """Singleton CLOB orderbook fetcher with a per-token TTL cache.

    Use via :meth:`instance`. The class itself stores no token state — all
    cache and lock state hangs off the singleton instance, so tests can
    drop it via :meth:`reset_for_testing`.
    """

    _instance: ClassVar[OrderbookPool | None] = None

    def __init__(self) -> None:
        # token_id -> (fetched_at_monotonic, snapshot_dict)
        self._cache: dict[str, tuple[float, dict]] = {}
        # per-token-id locks so a thundering herd on the same token collapses
        # to a single in-flight request.
        self._locks: dict[str, asyncio.Lock] = {}
        self._sem: asyncio.Semaphore = asyncio.Semaphore(BATCH_CONCURRENCY)
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> OrderbookPool:
        """Return the process-wide singleton, constructing on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Drop the singleton (does NOT call ``aclose``).

        Tests that need a fresh pool should call this after awaiting
        :meth:`aclose` on the previous instance.
        """
        cls._instance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_snapshot(
        self,
        token_id: str,
        *,
        max_age_s: int = DEFAULT_MAX_AGE_S,
    ) -> dict | None:
        """Return a cached or freshly-fetched book snapshot for ``token_id``.

        Returns ``None`` on network errors, HTTP 5xx, or empty CLOB response
        — never raises. A WARNING is logged for diagnostic purposes.

        ``max_age_s`` of ``0`` forces a refetch.
        """
        if self._closed:
            raise RuntimeError("OrderbookPool is closed")

        cached = self._cache.get(token_id)
        if cached is not None:
            fetched_at, snap = cached
            if (time.monotonic() - fetched_at) <= max_age_s:
                return snap

        # Per-token lock — collapse a herd of concurrent get_snapshot calls
        # on the same token into one upstream fetch.
        lock = self._locks.setdefault(token_id, asyncio.Lock())
        async with lock:
            # Re-check the cache under the lock (another coroutine may have
            # filled it while we were waiting).
            cached = self._cache.get(token_id)
            if cached is not None:
                fetched_at, snap = cached
                if (time.monotonic() - fetched_at) <= max_age_s:
                    return snap
            return await self._fetch_and_cache(token_id)

    async def get_snapshots(
        self,
        token_ids: list[str],
        *,
        max_age_s: int = DEFAULT_MAX_AGE_S,
    ) -> dict[str, dict]:
        """Parallel batch fetch. Returns ``{token_id: snapshot}`` for any
        token that returned a non-None snapshot.

        Concurrency is bounded by ``BATCH_CONCURRENCY`` (=10). Tokens that
        fail (network, 5xx, etc.) are silently omitted from the result —
        callers can compare ``len(result)`` to ``len(token_ids)`` to
        detect partial failure.
        """
        if self._closed:
            raise RuntimeError("OrderbookPool is closed")
        if not token_ids:
            return {}

        # Dedupe while preserving order — callers sometimes pass duplicates
        # (e.g. the same YES token across two strategies).
        seen: set[str] = set()
        unique: list[str] = []
        for tid in token_ids:
            if tid not in seen:
                seen.add(tid)
                unique.append(tid)

        async def _one(tid: str) -> tuple[str, dict | None]:
            async with self._sem:
                snap = await self.get_snapshot(tid, max_age_s=max_age_s)
                return tid, snap

        results = await asyncio.gather(*(_one(t) for t in unique))
        return {tid: snap for tid, snap in results if snap is not None}

    async def warm(self, token_ids: list[str]) -> None:
        """Prefetch ``token_ids`` into the cache in parallel.

        Convenience wrapper around :meth:`get_snapshots`. Useful in
        FastAPI lifespan startup or before a multi-leg endpoint runs its
        own computations.
        """
        await self.get_snapshots(token_ids, max_age_s=DEFAULT_MAX_AGE_S)

    async def aclose(self) -> None:
        """Mark the pool closed and drop cache + lock state.

        Does NOT close the underlying ``PolymarketHTTPPool`` client — that
        is shared across many modules and has its own lifecycle.
        """
        self._closed = True
        self._cache.clear()
        self._locks.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        """Return the shared CLOB client.

        Factored out so tests can monkey-patch a stand-in client.
        """
        return PolymarketHTTPPool.instance().clob_client

    async def _fetch_and_cache(self, token_id: str) -> dict | None:
        """Hit ``/book?token_id=...`` once. Cache + return the snapshot,
        or return ``None`` on failure (and log WARNING)."""
        client = self._client()
        try:
            resp = await client.get("/book", params={"token_id": token_id})
        except httpx.HTTPError as exc:
            logger.warning(
                "OrderbookPool: network error fetching token_id=%s: %s",
                token_id,
                exc,
            )
            return None

        if resp.status_code >= 500:
            logger.warning(
                "OrderbookPool: CLOB %s for token_id=%s",
                resp.status_code,
                token_id,
            )
            return None
        if resp.status_code == 404:
            # Unknown token — no point retrying.
            logger.warning("OrderbookPool: 404 for token_id=%s", token_id)
            return None
        if resp.status_code >= 400:
            logger.warning(
                "OrderbookPool: CLOB %s for token_id=%s",
                resp.status_code,
                token_id,
            )
            return None

        try:
            raw = resp.json()
        except Exception as exc:  # pragma: no cover - extremely unlikely
            logger.warning("OrderbookPool: invalid JSON for token_id=%s: %s", token_id, exc)
            return None

        snap = self._normalize(raw)
        self._cache[token_id] = (time.monotonic(), snap)
        return snap

    @staticmethod
    def _normalize(raw: dict) -> dict:
        """Coerce a raw CLOB ``/book`` response into our snapshot shape.

        CLOB returns ``{"bids": [{price, size}, ...], "asks": [...]}``; we
        flatten to ``[[price, size], ...]`` for lighter JSON across the
        wire and stamp an ``updated_at`` from server-now.
        """

        def _levels(side: list) -> list[list[float]]:
            out: list[list[float]] = []
            for lvl in side or []:
                if isinstance(lvl, dict):
                    p = lvl.get("price")
                    s = lvl.get("size")
                elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    p, s = lvl[0], lvl[1]
                else:
                    continue
                try:
                    out.append([float(p), float(s)])
                except (TypeError, ValueError):
                    continue
            return out

        return {
            "bids": _levels(raw.get("bids", [])),
            "asks": _levels(raw.get("asks", [])),
            "updated_at": datetime.now(UTC).isoformat(),
        }
