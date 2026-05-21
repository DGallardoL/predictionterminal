"""Manifold Markets client (open-source prediction-market venue).

Manifold is a play-money + real-money hybrid venue with a clean public REST
API (no auth required for read endpoints). We expose a thin async client
covering the four endpoints we need cross-venue:

  - ``GET /v0/search-markets?term=...``     market search
  - ``GET /v0/market/{id}`` / ``/slug/{slug}``  single-market metadata
  - ``GET /v0/market/{id}/positions``       top traders by position size
  - ``GET /v0/bets?contractId=...``         recent bets / trades

Probabilities are returned as floats in ``[0, 1]`` (similar to Polymarket),
so no rescaling is required when feeding them into :mod:`pfm.arb_scanner`.

Rate limit
----------
Manifold's public docs cap unauthenticated callers at ~100 requests / minute.
We pace ourselves with a 5-slot ``asyncio.Semaphore`` plus a 0.6 s sleep per
call (≈100 rpm steady-state) so a burst of factor-history fetches does not
trip the limiter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pandas as pd

from pfm.cache_pool import CachePool

logger = logging.getLogger(__name__)

MANIFOLD_BASE_URL: str = "https://api.manifold.markets/v0"

# Public-API rate-limit knobs.
DEFAULT_CONCURRENCY: int = 5
DEFAULT_MIN_INTERVAL_S: float = 0.6

# 429 backoff knob. One retry is enough — Manifold's limiter recovers in <2 s
# under our paced load. More retries amplify a real burst into pathological
# latency; this matches the polymarket.py pattern.
_RETRY_BACKOFF_S: float = 1.5

# Process-local caches (W11-14: migrated to CachePool — gains optional Redis
# L2 + heap-based eviction without changing call-site semantics).
# Manifold search / metadata are stable on a 5-min scale (markets don't
# appear or disappear that fast) and slug→market id is effectively
# immutable. Long item-cache TTL collapses repeated discovery into single
# fetches; short list-cache TTL keeps search fresh enough for the UI.
_SEARCH_CACHE_TTL_S: int = 300
_MARKET_CACHE_TTL_S: int = 3600
_CACHE_MAX_ENTRIES: int = 1024

# CachePool instances exposed at module scope so conftest.py's autouse
# ``_reset_volatile_caches`` fixture (which calls ``.clear()``) keeps
# working — CachePool.clear() is API-compatible with dict.clear().
_SEARCH_CACHE: CachePool = CachePool(namespace="manifold_search", l1_maxsize=_CACHE_MAX_ENTRIES)
_MARKET_CACHE: CachePool = CachePool(namespace="manifold_market", l1_maxsize=_CACHE_MAX_ENTRIES)


class ManifoldError(RuntimeError):
    """Raised when Manifold returns a usable HTTP response but bad data."""


class ManifoldClient:
    """Async httpx-based client for the Manifold public REST API.

    Parameters
    ----------
    base_url:
        Override the default ``https://api.manifold.markets/v0`` for tests.
    client:
        Optional pre-built :class:`httpx.AsyncClient`. When provided the
        caller owns its lifecycle and ``close()`` becomes a no-op for that
        client (we never close a borrowed transport).
    timeout:
        Per-request timeout when we own the client.
    concurrency:
        Maximum concurrent in-flight requests (semaphore size).
    min_interval_s:
        Minimum sleep enforced between successive ``_get`` calls. Combined
        with ``concurrency`` this caps steady-state throughput at roughly
        ``concurrency / min_interval_s`` requests per second.
    """

    def __init__(
        self,
        base_url: str = MANIFOLD_BASE_URL,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        concurrency: int = DEFAULT_CONCURRENCY,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._sem = asyncio.Semaphore(max(1, int(concurrency)))
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._last_call_ts: float = 0.0
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> ManifoldClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    # ---- internals ---------------------------------------------------------

    async def _pace(self) -> None:
        """Enforce the minimum-interval gap between calls (thread-safe)."""
        if self._min_interval_s <= 0:
            return
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            wait = self._min_interval_s - (now - self._last_call_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_ts = loop.time()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Perform a paced GET against ``base_url + path`` and return JSON.

        Retries once on HTTP 429 after a fixed 1.5 s backoff. The semaphore
        + min-interval pacing usually keeps us under Manifold's 100 rpm cap,
        but burst-fanouts (parallel factor discovery) occasionally trip it
        and the simple retry is enough to absorb a single bucket refill.
        """
        async with self._sem:
            await self._pace()
            url = f"{self.base_url}{path}"
            r = await self._client.get(url, params=params)
            if r.status_code == 429:
                logger.warning("manifold 429 on %s — retrying in %.1fs", path, _RETRY_BACKOFF_S)
                await asyncio.sleep(_RETRY_BACKOFF_S)
                r = await self._client.get(url, params=params)
            r.raise_for_status()
            return r.json()

    # ---- search ------------------------------------------------------------

    async def search_markets(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Free-text market search.

        ``term`` is what Manifold calls the query parameter; we mirror our
        own callers' ``query=`` for consistency with the rest of ``pfm``.

        Cached for 5 min keyed on (base_url, query, limit) — list-style
        discovery is stable on that horizon and re-fetching on every
        cross-venue arb refresh wastes the public-API rate budget.
        """
        if not query:
            return []
        # CachePool stores under string keys; the tuple compresses to a
        # ``|``-joined token (base_url|query|limit). Distinct base_url /
        # limit produce distinct cache entries, matching the old tuple-key
        # behaviour.
        cache_key = f"{self.base_url}|{query}|{int(limit)}"
        cached = _SEARCH_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

        data = await self._get(
            "/search-markets",
            params={"term": query, "limit": int(limit)},
        )
        if not isinstance(data, list):
            raise ManifoldError(f"search-markets returned non-list: {type(data).__name__}")
        result = data[: int(limit)]
        _SEARCH_CACHE.set(cache_key, list(result), ttl=_SEARCH_CACHE_TTL_S)
        return result

    # ---- single market ----------------------------------------------------

    async def get_market(self, slug_or_id: str) -> dict[str, Any]:
        """Fetch a single market by slug or by id.

        Manifold exposes both ``/market/{id}`` and ``/slug/{slug}``. We try
        the slug endpoint first (most callers pass slugs); on 404 we fall
        back to ``/market/{id}``.

        Cached for 1 h. Manifold market objects mutate constantly (price,
        volume) but consumers of this method want the *identity* — id,
        question, contract type — which is immutable. Callers that need a
        live snapshot should hit ``/bets`` or ``/positions`` instead.
        """
        if not slug_or_id:
            raise ManifoldError("slug_or_id must be non-empty")
        # CachePool namespace already includes ``manifold_market``; the key
        # only needs base_url + slug to differentiate test/live transports.
        # ``|`` is the chosen separator (':' is reserved by the CachePool
        # namespace machinery).
        cache_key = f"{self.base_url}|{slug_or_id}"
        cached = _MARKET_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached) if isinstance(cached, dict) else cached
        try:
            data = await self._get(f"/slug/{slug_or_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            data = await self._get(f"/market/{slug_or_id}")
        stored = dict(data) if isinstance(data, dict) else data
        _MARKET_CACHE.set(cache_key, stored, ttl=_MARKET_CACHE_TTL_S)
        return data

    # ---- positions / bets -------------------------------------------------

    async def get_market_positions(self, market_id: str, top: int = 20) -> list[dict[str, Any]]:
        """Top traders by position size for a given market."""
        if not market_id:
            return []
        data = await self._get(
            f"/market/{market_id}/positions",
            params={"top": int(top)},
        )
        if not isinstance(data, list):
            raise ManifoldError(f"positions returned non-list: {type(data).__name__}")
        return data[: int(top)]

    async def get_market_bets(self, market_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Most recent bets / trades on a market.

        Manifold's ``/bets`` endpoint takes ``contractId`` (the market id)
        and an optional ``limit`` (max 1000).
        """
        if not market_id:
            return []
        data = await self._get(
            "/bets",
            params={"contractId": market_id, "limit": int(limit)},
        )
        if not isinstance(data, list):
            raise ManifoldError(f"bets returned non-list: {type(data).__name__}")
        return data[: int(limit)]

    # ---- history ----------------------------------------------------------

    async def fetch_history(self, market_id: str, days: int = 90) -> pd.DataFrame:
        """Reconstruct a ``[date, prob, volume]`` history from recent bets.

        Manifold doesn't expose a candlestick endpoint on the public API,
        so we walk ``/bets`` and aggregate to UTC daily buckets:

          - ``prob``   = last ``probAfter`` of the day (close-of-day mid)
          - ``volume`` = sum of ``|amount|`` (M$/USD depending on market)

        ``days`` clips the returned window to the most recent N days. The
        DataFrame is sorted ascending and indexed by row number; ``date``
        is a ``pandas.Timestamp`` (UTC, normalised to date).
        """
        bets = await self.get_market_bets(market_id, limit=1000)
        if not bets:
            return pd.DataFrame(columns=["date", "prob", "volume"])

        rows: list[dict[str, Any]] = []
        for b in bets:
            ts = b.get("createdTime")
            prob = b.get("probAfter")
            amount = b.get("amount", 0.0)
            if ts is None or prob is None:
                continue
            try:
                # Manifold's createdTime is unix ms.
                dt = pd.Timestamp(int(ts), unit="ms", tz="UTC").normalize()
            except (TypeError, ValueError, OverflowError):
                continue
            try:
                rows.append({"date": dt, "prob": float(prob), "amount": abs(float(amount))})
            except (TypeError, ValueError):
                continue

        if not rows:
            return pd.DataFrame(columns=["date", "prob", "volume"])

        df = pd.DataFrame(rows).sort_values("date")
        # Aggregate to daily: last prob of day, sum of abs amounts.
        agg = (
            df.groupby("date", as_index=False)
            .agg(prob=("prob", "last"), volume=("amount", "sum"))
            .sort_values("date")
            .reset_index(drop=True)
        )
        if int(days) > 0 and not agg.empty:
            cutoff = agg["date"].max() - pd.Timedelta(days=int(days))
            agg = agg[agg["date"] >= cutoff].reset_index(drop=True)
        return agg


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MIN_INTERVAL_S",
    "MANIFOLD_BASE_URL",
    "ManifoldClient",
    "ManifoldError",
]
