"""Order book reconstruction in Redis using the official Binance algorithm.

Redis structures:
    bids:<SYMBOL>      ZSET (score=price_float, member=price_str)
    asks:<SYMBOL>      ZSET (score=price_float, member=price_str)
    bids_qty:<SYMBOL>  HASH {price_str -> qty_str}
    asks_qty:<SYMBOL>  HASH {price_str -> qty_str}
    ob:meta:<SYMBOL>   HASH {last_update_id, ...}

Algorithm:
    1. Buffer depth updates until snapshot is loaded.
    2. GET /api/v3/depth?symbol=X&limit=1000 -> lastUpdateId.
    3. ZADD snapshot bids/asks.
    4. Discard buffered updates with u < lastUpdateId.
    5. First valid update: U <= lastUpdateId+1 <= u.
    6. Apply: qty==0 -> ZREM, else ZADD.
    7. If gap detected -> resync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx
import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class OrderBookMetrics:
    snapshots_loaded: int = 0
    updates_applied: int = 0
    updates_skipped_old: int = 0
    resyncs_due_to_gap: int = 0
    by_symbol: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "snapshots_loaded": self.snapshots_loaded,
            "updates_applied": self.updates_applied,
            "updates_skipped_old": self.updates_skipped_old,
            "resyncs_due_to_gap": self.resyncs_due_to_gap,
        }


class OrderBookState:
    """Maintains a reconstructed order book in Redis for subscribed symbols."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        rest_base_url: str = "https://api.binance.com",
        depth_levels: int = 20,
    ) -> None:
        self.redis_url = redis_url
        self.rest_base_url = rest_base_url
        self.depth_levels = depth_levels
        self._redis: aioredis.Redis | None = None
        self._http: httpx.AsyncClient | None = None
        self._last_u: dict[str, int] = {}
        self._snapshot_id: dict[str, int] = {}
        self._buffer: dict[str, list[dict[str, Any]]] = {}
        self.metrics = OrderBookMetrics()

    async def connect(self) -> None:
        self._redis = aioredis.from_url(
            self.redis_url,
            decode_responses=True,
            health_check_interval=30,
        )
        self._http = httpx.AsyncClient(base_url=self.rest_base_url, timeout=10.0)

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
        if self._redis is not None:
            await self._redis.aclose()

    async def load_snapshot(self, symbol: str) -> None:
        if self._http is None or self._redis is None:
            raise RuntimeError("not connected")

        sym = symbol.upper()
        resp = await self._http.get("/api/v3/depth", params={"symbol": sym, "limit": 1000})
        resp.raise_for_status()
        snap = resp.json()

        last_update_id = int(snap["lastUpdateId"])

        async with self._redis.pipeline(transaction=True) as pipe:
            await pipe.delete(f"bids:{sym}", f"asks:{sym}", f"bids_qty:{sym}", f"asks_qty:{sym}")
            for price, qty in snap.get("bids", []):
                if Decimal(qty) > 0:
                    await pipe.zadd(f"bids:{sym}", {price: float(price)})
                    await pipe.hset(f"bids_qty:{sym}", price, qty)
            for price, qty in snap.get("asks", []):
                if Decimal(qty) > 0:
                    await pipe.zadd(f"asks:{sym}", {price: float(price)})
                    await pipe.hset(f"asks_qty:{sym}", price, qty)
            await pipe.hset(f"ob:meta:{sym}", mapping={"last_update_id": last_update_id})
            await pipe.execute()

        self._snapshot_id[sym] = last_update_id
        self._last_u[sym] = last_update_id
        self.metrics.snapshots_loaded += 1

        buffered = self._buffer.pop(sym, [])
        for upd in buffered:
            await self.apply_update(upd, _from_buffer=True)

    async def apply_update(self, data: dict[str, Any], _from_buffer: bool = False) -> None:
        if self._redis is None:
            raise RuntimeError("not connected")

        sym = str(data["s"]).upper()
        first_u = int(data["U"])
        last_u = int(data["u"])

        if sym not in self._snapshot_id:
            if not _from_buffer:
                self._buffer.setdefault(sym, []).append(data)
            return

        snapshot_id = self._snapshot_id[sym]
        prev_last = self._last_u[sym]

        if last_u < snapshot_id:
            self.metrics.updates_skipped_old += 1
            return

        if prev_last == snapshot_id and not (first_u <= snapshot_id + 1 <= last_u):
            self.metrics.resyncs_due_to_gap += 1
            await self.load_snapshot(sym)
            return

        if prev_last > snapshot_id and first_u != prev_last + 1:
            self.metrics.resyncs_due_to_gap += 1
            await self.load_snapshot(sym)
            return

        async with self._redis.pipeline(transaction=False) as pipe:
            for price, qty in data.get("b", []):
                p_str, p_f = str(price), float(price)
                if Decimal(str(qty)) == 0:
                    await pipe.zrem(f"bids:{sym}", p_str)
                    await pipe.hdel(f"bids_qty:{sym}", p_str)
                else:
                    await pipe.zadd(f"bids:{sym}", {p_str: p_f})
                    await pipe.hset(f"bids_qty:{sym}", p_str, str(qty))

            for price, qty in data.get("a", []):
                p_str, p_f = str(price), float(price)
                if Decimal(str(qty)) == 0:
                    await pipe.zrem(f"asks:{sym}", p_str)
                    await pipe.hdel(f"asks_qty:{sym}", p_str)
                else:
                    await pipe.zadd(f"asks:{sym}", {p_str: p_f})
                    await pipe.hset(f"asks_qty:{sym}", p_str, str(qty))

            await pipe.hset(f"ob:meta:{sym}", mapping={"last_update_id": last_u})
            await pipe.execute()

        self._last_u[sym] = last_u
        self.metrics.updates_applied += 1
        self.metrics.by_symbol[sym] = self.metrics.by_symbol.get(sym, 0) + 1

    async def get_top_n_bids(self, symbol: str, n: int = 10) -> list[tuple[float, float]]:
        if self._redis is None:
            raise RuntimeError("not connected")
        sym = symbol.upper()
        results = await self._redis.zrevrange(f"bids:{sym}", 0, n - 1, withscores=True)
        if not results:
            return []
        async with self._redis.pipeline(transaction=False) as pipe:
            for price_str, _ in results:
                await pipe.hget(f"bids_qty:{sym}", price_str)
            qtys = await pipe.execute()
        return [(score, float(qty or 0)) for (_, score), qty in zip(results, qtys, strict=True)]

    async def get_top_n_asks(self, symbol: str, n: int = 10) -> list[tuple[float, float]]:
        if self._redis is None:
            raise RuntimeError("not connected")
        sym = symbol.upper()
        results = await self._redis.zrange(f"asks:{sym}", 0, n - 1, withscores=True)
        if not results:
            return []
        async with self._redis.pipeline(transaction=False) as pipe:
            for price_str, _ in results:
                await pipe.hget(f"asks_qty:{sym}", price_str)
            qtys = await pipe.execute()
        return [(score, float(qty or 0)) for (_, score), qty in zip(results, qtys, strict=True)]

    async def get_obi(self, symbol: str, levels: int = 5) -> float | None:
        """Order Book Imbalance for top N levels: (bid_qty - ask_qty) / (bid_qty + ask_qty)."""
        bids = await self.get_top_n_bids(symbol, levels)
        asks = await self.get_top_n_asks(symbol, levels)
        bid_qty = sum(q for _, q in bids)
        ask_qty = sum(q for _, q in asks)
        total = bid_qty + ask_qty
        if total == 0:
            return None
        return (bid_qty - ask_qty) / total
