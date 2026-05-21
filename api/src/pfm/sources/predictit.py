"""PredictIt client (US-government-tracking prediction market).

PredictIt does not publish an official API, but the public market-data feed
at ``https://www.predictit.org/api/marketdata/all/`` is widely used and has
been stable for years. We treat it as a read-only public endpoint with no
auth.

Each market has multiple ``contracts`` (e.g. "Trump", "Harris", "Other" for
a 2024-winner market). We expose helpers to:

  - fetch the entire venue snapshot (cached aggressively, 5 min TTL)
  - fetch one market by id
  - reconstruct a daily ``[date, prob, volume]`` history per *contract*

Caching note
------------
The "all markets" endpoint is the expensive one (~150 markets, ~400 KB).
Callers should always go through :func:`fetch_all_markets` when they want
multiple markets — we cache the parsed payload for 300 s.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
import pandas as pd

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)

PREDICTIT_BASE_URL: str = "https://www.predictit.org/api"
PREDICTIT_ALL_MARKETS_PATH: str = "/marketdata/all/"
PREDICTIT_MARKET_PATH: str = "/marketdata/markets/{market_id}"

# 5-minute TTL on the venue-wide snapshot — it's the cara llamada per spec.
_ALL_TTL_S: int = 300
_ALL_CACHE = get_cache("predictit_all", ttl=_ALL_TTL_S)

# Per-market snapshot cache for the rare cold-path /marketdata/markets/{id}
# branch. Same horizon as the venue snapshot to keep semantics consistent.
_MARKET_CACHE = get_cache("predictit_market", ttl=_ALL_TTL_S)

# Single 429-retry with 1.5 s backoff — matches the polymarket.py pattern.
_RETRY_BACKOFF_S: float = 1.5


async def _get_with_retry(
    client: httpx.AsyncClient, url: str, *, params: dict[str, Any] | None = None
) -> httpx.Response:
    """GET ``url`` with a single 429-retry after :data:`_RETRY_BACKOFF_S`.

    Centralised here so both ``fetch_all_markets`` and ``fetch_market``
    share the same backoff semantics without duplicating logic.
    """
    r = await client.get(url, params=params)
    if r.status_code == 429:
        logger.warning("predictit 429 on %s — retrying in %.1fs", url, _RETRY_BACKOFF_S)
        await asyncio.sleep(_RETRY_BACKOFF_S)
        r = await client.get(url, params=params)
    return r


class PredictItError(RuntimeError):
    """Raised when PredictIt returns a usable response with bad data."""


class PredictItClient:
    """Async httpx-based client for the PredictIt public market-data feed."""

    def __init__(
        self,
        base_url: str = PREDICTIT_BASE_URL,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> PredictItClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    # ---- core endpoints ----------------------------------------------------

    async def fetch_all_markets(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Return every active PredictIt market with its contracts.

        Cached for 5 minutes via :mod:`pfm.cache_utils`. Pass
        ``force_refresh=True`` to bust the cache.
        """
        cache_key = ("all", self.base_url)
        if not force_refresh:
            cached = _ALL_CACHE.get(cache_key)
            if cached is not None:
                return cached

        url = f"{self.base_url}{PREDICTIT_ALL_MARKETS_PATH}"
        r = await _get_with_retry(self._client, url)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict) or "markets" not in payload:
            raise PredictItError("PredictIt /marketdata/all/ payload missing 'markets' key")
        markets = payload.get("markets") or []
        if not isinstance(markets, list):
            raise PredictItError(f"PredictIt 'markets' is not a list: {type(markets).__name__}")
        _ALL_CACHE.set(cache_key, markets, ttl=_ALL_TTL_S)
        return markets

    async def fetch_market(self, market_id: int) -> dict[str, Any]:
        """Fetch a single market by id (uses the all-markets cache when warm).

        We prefer the cache because PredictIt's per-market endpoint is
        sometimes flakier than the bulk one; falling back to a direct GET
        only when the bulk cache is cold.
        """
        if market_id is None:
            raise PredictItError("market_id must not be None")
        target = int(market_id)

        cached = _ALL_CACHE.get(("all", self.base_url))
        if isinstance(cached, list):
            for m in cached:
                if int(m.get("id", -1)) == target:
                    return m

        # Per-market direct lookup — also cached so consecutive cold-path
        # hits on the same market don't re-spin the upstream.
        per_market_key = ("market", self.base_url, target)
        cached_market = _MARKET_CACHE.get(per_market_key)
        if isinstance(cached_market, dict):
            return cached_market

        url = f"{self.base_url}{PREDICTIT_MARKET_PATH.format(market_id=target)}"
        r = await _get_with_retry(self._client, url)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise PredictItError(f"PredictIt market payload not a dict: {type(data).__name__}")
        _MARKET_CACHE.set(per_market_key, data, ttl=_ALL_TTL_S)
        return data

    # ---- history ----------------------------------------------------------

    async def fetch_history(self, market_id: int, days: int = 90) -> pd.DataFrame:
        """Synthesise a ``[date, prob, volume]`` daily history for a market.

        PredictIt's public feed is a *snapshot* — no historical bars. To
        keep the API surface symmetric with Manifold/Polymarket we return
        a single-row DataFrame derived from the current snapshot, taking
        the leading contract's ``lastTradePrice`` as the market-level prob.

        ``days`` is accepted for API symmetry but does not extend the
        snapshot horizon; callers that need real history must wire a
        persistence layer that polls :func:`fetch_market` over time.
        """
        market = await self.fetch_market(market_id)
        contracts = market.get("contracts") or []
        if not isinstance(contracts, list) or not contracts:
            return pd.DataFrame(columns=["date", "prob", "volume", "contract_id"])

        # Pick the contract with the highest lastTradePrice as the market lead.
        def _ltp(c: dict[str, Any]) -> float:
            try:
                v = c.get("lastTradePrice")
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        lead = max(contracts, key=_ltp)
        try:
            prob = float(lead.get("lastTradePrice") or 0.0)
        except (TypeError, ValueError):
            prob = 0.0

        # Volume isn't on the public feed at the contract level; use the
        # market's ``totalSharesTraded`` if present, else 0.
        try:
            vol = float(market.get("totalSharesTraded") or 0.0)
        except (TypeError, ValueError):
            vol = 0.0

        date = pd.Timestamp(time.time(), unit="s", tz="UTC").normalize()
        # ``days`` is honoured by trimming if downstream callers pre-filled
        # additional rows; we have only one snapshot row so this is a no-op
        # in practice but keeps the column schema consistent.
        df = pd.DataFrame(
            [
                {
                    "date": date,
                    "prob": prob,
                    "volume": vol,
                    "contract_id": int(lead.get("id", 0) or 0),
                }
            ]
        )
        if int(days) <= 0:
            return df.iloc[:0].reset_index(drop=True)
        return df


# ---- convenience module-level helpers --------------------------------------


async def fetch_all_markets(
    *,
    base_url: str = PREDICTIT_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Module-level ``fetch_all_markets`` for callers that don't want a client."""
    async with PredictItClient(base_url=base_url, client=client) as pic:
        return await pic.fetch_all_markets()


async def fetch_market(
    market_id: int,
    *,
    base_url: str = PREDICTIT_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Module-level ``fetch_market`` for callers that don't want a client."""
    async with PredictItClient(base_url=base_url, client=client) as pic:
        return await pic.fetch_market(market_id)


async def fetch_history(
    market_id: int,
    days: int = 90,
    *,
    base_url: str = PREDICTIT_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Module-level ``fetch_history`` for callers that don't want a client."""
    async with PredictItClient(base_url=base_url, client=client) as pic:
        return await pic.fetch_history(market_id, days=days)


__all__ = [
    "PREDICTIT_ALL_MARKETS_PATH",
    "PREDICTIT_BASE_URL",
    "PREDICTIT_MARKET_PATH",
    "PredictItClient",
    "PredictItError",
    "fetch_all_markets",
    "fetch_history",
    "fetch_market",
]
