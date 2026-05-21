"""Real-time crypto event capture using the ``crypto-microstructure`` library.

Runs as a background asyncio task inside the FastAPI lifespan (opt-in via
``PFM_CRYPTO_WS_ENABLED=1``). Streams Binance trade + bookTicker for the
default 10 pairs, feeds them into ``SignalEngine``, and stashes a rolling
buffer of *event-class* signals (whale alerts, VWAP z-score extremes, spread
spikes) for each symbol. The Strategies → Crypto Micro panel reads from this
buffer via ``/strategies/crypto/events``.

Why not just import the WS engine directly into the request path?
Because:
* WS connections are long-lived and shared across users (1 socket × N panels).
* Event detection is stateful (rolling windows, P99 whale threshold) — it has
  to keep running between requests.
* We don't want per-request Binance connections; this module is the single
  shared writer; the HTTP endpoints are pure readers.

The buffer is bounded (``BUFFER_PER_SYMBOL`` events per pair) so memory is
constant regardless of stream duration.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


BUFFER_PER_SYMBOL: int = 200  # last N events per pair
DEFAULT_PAIRS: list[str] = [
    "btcusdt",
    "ethusdt",
    "solusdt",
    "bnbusdt",
    "xrpusdt",
    "adausdt",
    "avaxusdt",
    "maticusdt",
    "dogeusdt",
    "linkusdt",
]
DEFAULT_STREAMS: list[str] = ["trade", "bookTicker"]
#: Only these signal names land in the event buffer. The other VWAP /
#: midprice / OBI signals are continuous (every event) and would drown the
#: buffer; we surface them via /strategies/crypto/snapshot instead.
EVENT_CLASS_SIGNALS: set[str] = {
    "whale_detected",
    "vwap_zscore_30m",  # only when |z| > 2 — gated by metadata.mean_reversion
}


@dataclass
class CryptoEvent:
    """One captured event ready to ship to the UI."""

    symbol: str
    ts_unix: float
    kind: str  # "whale" / "mean_reversion" / "spread_spike"
    name: str  # signal name from SignalEngine
    value: float
    side: str | None = None  # "buy" / "sell" / None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # ISO timestamp helps the UI render times without re-deriving.
        d["ts_iso"] = datetime.utcfromtimestamp(self.ts_unix).isoformat() + "Z"
        return d


class CryptoEventsEngine:
    """Single shared engine that owns the WS subscription + event buffer."""

    def __init__(self, pairs: list[str] | None = None) -> None:
        self.pairs = [p.lower() for p in (pairs or DEFAULT_PAIRS)]
        self._buffers: dict[str, deque[CryptoEvent]] = {
            p.upper(): deque(maxlen=BUFFER_PER_SYMBOL) for p in self.pairs
        }
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._connected_since: float | None = None
        self._stats = {"trades": 0, "book_updates": 0, "events": 0, "events_deduped": 0}
        self._engine_obj: Any | None = None  # SignalEngine
        # Optional Redis publisher. When the engine runs in gunicorn it's
        # owned by exactly one worker (leader election). Other workers'
        # ``/strategies/crypto/events`` endpoint reads from Redis so they
        # all see the same data, instead of returning 0 events.
        self._redis_client: Any | None = None  # set via attach_redis()
        self._redis_key: str = "arb:crypto_events"
        self._redis_max: int = 500  # keep latest N across all symbols
        self._redis_ttl_s: int = 900
        # Per-symbol model_state publishing. Follower gunicorn workers don't
        # run the WS engine, so their ``_live_engine_state(sym)`` returns
        # None and the crypto5min UI flickers between "live engine used:
        # true/false" depending on which worker handles each request. The
        # leader publishes every symbol's full ``model_state(sym)`` dict to
        # Redis at ``{state_key_prefix}{SYM}`` with a short TTL (5 s) so a
        # dead leader's keys self-evict; followers read from there.
        self._state_key_prefix: str = "pfm:crypto_engine:state:"
        self._state_ttl_s: int = 5
        self._state_publisher_task: asyncio.Task | None = None
        # Per-(symbol, kind, sign) throttle: last emit ts. Mean-reversion fires
        # on every trade while |z|>2; without this we'd flood the buffer with
        # ~1/sec near-duplicates. Whales bypass this — each large print is
        # its own event.
        self._last_emit: dict[tuple[str, str, int], float] = {}

    # --- public API ---------------------------------------------------------

    def attach_redis(
        self,
        client: Any,
        key: str = "arb:crypto_events",
        max_keep: int = 500,
        ttl_seconds: int = 900,
        *,
        state_key_prefix: str = "pfm:crypto_engine:state:",
        state_ttl_seconds: int = 5,
    ) -> None:
        """Wire a Redis client so captured events get published cross-worker.

        Uses ZADD with the unix-ts as score so reads can range-by-time and
        ZREMRANGEBYRANK trims the oldest entries to keep the set bounded.
        Refreshes EXPIRE on every push so a dead leader's ZSET self-evicts
        within ``ttl_seconds`` instead of lingering forever.

        Also wires the per-symbol model_state publisher (SETEX one key per
        symbol every 1 s); ``state_ttl_seconds`` MUST be > 1 s so followers
        don't see the key expire mid-poll between writes. Default 5 s gives
        ~5× the publisher cadence — a dead leader's keys evict within that
        budget and follower reads cleanly fall back to ``None``.
        """
        self._redis_client = client
        self._redis_key = key
        self._redis_max = max_keep
        self._redis_ttl_s = int(ttl_seconds)
        self._state_key_prefix = state_key_prefix
        self._state_ttl_s = int(state_ttl_seconds)

    def state_key_for(self, symbol: str) -> str:
        """Redis key where this engine publishes ``model_state(symbol)``."""
        return f"{self._state_key_prefix}{symbol.upper()}"

    def publish_state_now(self, symbol: str) -> bool:
        """Publish one symbol's current ``model_state`` to Redis (best-effort).

        Returns ``True`` on a successful SETEX, ``False`` otherwise (Redis
        unavailable, no state for symbol, serialization error). Idempotent
        and cheap — safe to call from the background loop every second.
        """
        if self._redis_client is None:
            return False
        state = None
        try:
            state = self.model_state(symbol)
        except Exception:
            state = None
        if state is None:
            return False
        try:
            import json as _json

            blob = _json.dumps(state).encode()
            # SETEX = atomic SET + EXPIRE — exactly the semantics we want
            # (write now, evict in TTL seconds if no refresh).
            self._redis_client.setex(
                self.state_key_for(symbol),
                self._state_ttl_s,
                blob,
            )
            return True
        except Exception:
            return False

    async def _state_publisher_loop(self, interval_seconds: float = 1.0) -> None:
        """Background coroutine: publish every symbol's state to Redis at 1 Hz.

        Runs only on the leader (the worker that started the engine). The
        loop is resilient to per-iteration errors so one bad symbol doesn't
        kill the publisher for the others.
        """
        while not self._stop.is_set():
            try:
                if self._redis_client is not None and self._engine_obj is not None:
                    # Snapshot keys to avoid "dict changed during iteration"
                    # if the engine adds a new symbol mid-loop.
                    try:
                        symbols = list(self._engine_obj._states.keys())
                    except Exception:
                        symbols = []
                    for sym in symbols:
                        try:
                            self.publish_state_now(sym)
                        except Exception as exc:
                            logger.debug("state publish %s failed: %s", sym, exc)
            except Exception as exc:
                logger.debug("state publisher iter failed: %s", exc)
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                raise

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        return {
            "running": self.is_running(),
            "connected_since": self._connected_since,
            "pairs": [p.upper() for p in self.pairs],
            "stats": dict(self._stats),
            "buffer_sizes": {sym: len(buf) for sym, buf in self._buffers.items()},
            "buffer_cap_per_symbol": BUFFER_PER_SYMBOL,
        }

    def events(
        self,
        symbol: str | None = None,
        window_min: float = 5.0,
        kinds: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return events newer than ``window_min`` minutes (descending by time)."""
        cutoff = time.time() - window_min * 60
        out: list[CryptoEvent] = []
        buckets = [self._buffers[symbol.upper()]] if symbol else self._buffers.values()
        for buf in buckets:
            for ev in buf:
                if ev.ts_unix < cutoff:
                    continue
                if kinds and ev.kind not in kinds:
                    continue
                out.append(ev)
        out.sort(key=lambda e: e.ts_unix, reverse=True)
        return [ev.as_dict() for ev in out]

    def model_state(self, symbol: str) -> dict[str, Any] | None:
        """Return live cryptostuff signals + a model probability for ``symbol``.

        This is the "our prob vs market prob" data feed. We expose:
        * ``rv_per_trade``  — last per-trade std of log returns (15m window)
        * ``trades_per_sec`` — recent trade frequency, used to annualize σ
        * ``sigma_annual``  — annualized vol estimate from cryptostuff
        * ``ofi_1m``        — order-flow imbalance (∈ [-1, +1]) as drift bias
        * ``last_price``    — current last-trade price
        * ``vwap_30m``      — 30-min VWAP (anchor for mean-reversion)
        * ``z_vwap_30m``    — current z-score vs VWAP

        Returns ``None`` if engine is off or symbol not yet seen.
        """
        if self._engine_obj is None:
            return None
        sym = symbol.upper()
        try:
            state = self._engine_obj.get_state_summary(sym)
        except Exception:
            state = None
        if not state:
            return None
        s = self._engine_obj._states.get(sym)
        if s is None:
            return None
        # Pull rolling buffers from the SignalEngine internal state.
        trades_5m = list(getattr(s, "trades_5m", []))
        trades_1m = list(getattr(s, "trades_1m", []))
        trades_30m_list = list(getattr(s, "trades_30m", []))
        rets_15m = list(getattr(s, "returns_15m", []))
        # Trades-per-second from the 1m window (sample-rate proxy). If the
        # 1m deque is empty (sparse pair, or just-after-pruning gap) fall
        # back to the 5m / 30m windows so ``tps`` is non-zero whenever we
        # have *any* trades on file. This is the difference between
        # ``sigma_annual=None`` (followers think the engine is dead) and a
        # real estimate.
        tps = 0.0
        if trades_1m:
            tps = len(trades_1m) / 60.0
        elif trades_5m:
            tps = len(trades_5m) / 300.0
        elif trades_30m_list:
            tps = len(trades_30m_list) / 1800.0
        # Per-trade std of log-returns. Prefer the engine's pre-computed
        # ``returns_15m`` buffer when populated; otherwise derive returns
        # inline from the trade price series. The inline path is the
        # workaround for the production bug where ``returns_15m`` was
        # observed empty even when trades were flowing — we don't trust
        # the engine to push every return, we re-derive from raw prices.
        rv_per_trade: float | None = None
        if len(rets_15m) >= 10:
            mean_r = sum(rets_15m) / len(rets_15m)
            var_r = sum((r - mean_r) ** 2 for r in rets_15m) / len(rets_15m)
            rv_per_trade = var_r**0.5
        else:
            # Pick the deepest trade buffer that has enough entries.
            source = (
                trades_30m_list
                if len(trades_30m_list) >= 11
                else (trades_5m if len(trades_5m) >= 11 else trades_1m)
            )
            if len(source) >= 11:
                import math as _math

                prices = [t[1] for t in source if t[1] > 0]
                rets_inline: list[float] = []
                for i in range(1, len(prices)):
                    a, b = prices[i - 1], prices[i]
                    if a > 0 and b > 0:
                        rets_inline.append(_math.log(b / a))
                if len(rets_inline) >= 10:
                    mean_r = sum(rets_inline) / len(rets_inline)
                    var_r = sum((r - mean_r) ** 2 for r in rets_inline) / len(rets_inline)
                    rv_per_trade = var_r**0.5
        # σ_annual = σ_per_trade × √(tps × seconds_per_year). Note we use
        # ``rv_per_trade is not None and rv_per_trade > 0`` rather than the
        # falsy ``if rv_per_trade`` so a literal 0.0 doesn't silently fall
        # through (it still won't annualize meaningfully but the intent is
        # clearer).
        sigma_annual: float | None = None
        if rv_per_trade is not None and rv_per_trade > 0 and tps > 0:
            seconds_per_year = 365 * 24 * 3600
            sigma_annual = rv_per_trade * (tps * seconds_per_year) ** 0.5
            # Clip to a sane range (10% – 300%) — pathological spikes during
            # thin minutes can blow up annualization.
            sigma_annual = max(0.10, min(3.0, sigma_annual))
        # OFI from 1m window
        sv = sum(t[4] for t in trades_1m) if trades_1m else 0.0
        total_v = sum(abs(t[4]) for t in trades_1m) if trades_1m else 0.0
        ofi = (sv / total_v) if total_v > 0 else 0.0
        # VWAP 30m + z-score
        trades_30m = trades_30m_list
        vwap_30m = None
        z_vwap = None
        if trades_30m:
            total_notional = sum(t[3] for t in trades_30m)
            total_qty = sum(t[2] for t in trades_30m)
            if total_qty > 0:
                vwap_30m = total_notional / total_qty
                prices = [t[1] for t in trades_30m]
                if len(prices) > 10 and vwap_30m > 0:
                    mean_p = sum(prices) / len(prices)
                    var_p = sum((p - mean_p) ** 2 for p in prices) / len(prices)
                    std_p = (var_p**0.5) if var_p > 0 else 1e-10
                    last = state.get("last_price") or prices[-1]
                    z_vwap = (last - vwap_30m) / std_p
        return {
            "symbol": sym,
            "last_price": state.get("last_price"),
            "trades_per_sec": tps,
            "rv_per_trade": rv_per_trade,
            "sigma_annual": sigma_annual,
            "ofi_1m": ofi,
            "signed_volume_1m": sv,
            "n_trades_1m": len(trades_1m),
            "n_trades_5m": len(trades_5m),
            "vwap_30m": vwap_30m,
            "z_vwap_30m": z_vwap,
            "whale_threshold": state.get("whale_threshold"),
            "total_trades": state.get("total_trades"),
        }

    # --- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_forever(), name="pfm-crypto-events")
        # State publisher runs alongside the WS loop. It's a no-op until
        # ``attach_redis`` is called *and* the engine has populated state
        # for at least one symbol, so it's safe to spawn unconditionally.
        if self._state_publisher_task is None or self._state_publisher_task.done():
            self._state_publisher_task = asyncio.create_task(
                self._state_publisher_loop(),
                name="pfm-crypto-state-publisher",
            )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self._connected_since = None
        if self._state_publisher_task is not None:
            self._state_publisher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._state_publisher_task
            self._state_publisher_task = None

    # --- inner loop ---------------------------------------------------------

    async def _run_forever(self) -> None:
        """Main streaming loop. Auto-reconnects on disconnect."""
        # Lazy imports so the FastAPI app doesn't depend on cryptostuff at
        # module-load time. Missing → log + bail; the rest of the API works.
        try:
            from extraction import (
                BinanceStreamClient,
                parse_book_ticker,
                parse_trade,
            )
            from models import SignalEngine
        except ImportError as exc:
            logger.warning("crypto-microstructure not installed (%s); WS engine disabled", exc)
            return

        engine = SignalEngine()
        self._engine_obj = engine
        # The library yields raw {stream, data} dicts. We dispatch by stream kind.
        while not self._stop.is_set():
            try:
                async with BinanceStreamClient(
                    symbols=self.pairs, streams=DEFAULT_STREAMS
                ) as client:
                    self._connected_since = time.time()
                    logger.info("crypto-events: WS connected — %d pairs", len(self.pairs))
                    async for msg in client.iter_messages():
                        if self._stop.is_set():
                            break
                        stream = msg.get("stream", "")
                        data = msg.get("data") or {}
                        if not stream or not data:
                            continue
                        kind = stream.split("@", 1)[1] if "@" in stream else ""
                        try:
                            if kind == "trade":
                                evt = parse_trade(data)
                                self._stats["trades"] += 1
                                sigs = engine.on_trade(
                                    evt.symbol,
                                    float(evt.price),
                                    float(evt.quantity),
                                    evt.is_buyer_maker,
                                    evt.trade_time,
                                )
                                self._capture(
                                    evt.symbol, sigs, side=("sell" if evt.is_buyer_maker else "buy")
                                )
                            elif kind == "bookTicker":
                                evt = parse_book_ticker(data)
                                self._stats["book_updates"] += 1
                                sigs = engine.on_book_ticker(
                                    evt.symbol,
                                    float(evt.best_bid_price),
                                    float(evt.best_bid_qty),
                                    float(evt.best_ask_price),
                                    float(evt.best_ask_qty),
                                    evt.event_time,
                                )
                                self._capture(evt.symbol, sigs, side=None)
                        except Exception as exc:
                            logger.debug("crypto-events: parse/signal error: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("crypto-events: WS connection lost (%s); reconnecting in 5s", exc)
                self._connected_since = None
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(5)

    def _capture(self, symbol: str, signals: list[Any], side: str | None) -> None:
        """Filter the raw signal stream into event-class buffer entries."""
        buf = self._buffers.get(symbol)
        if buf is None:
            # New symbol — Binance can emit symbols we didn't subscribe to in
            # weird edge cases. Make a buffer on the fly.
            buf = deque(maxlen=BUFFER_PER_SYMBOL)
            self._buffers[symbol] = buf
        for sig in signals:
            name = getattr(sig, "name", "")
            if name not in EVENT_CLASS_SIGNALS:
                continue
            md = getattr(sig, "metadata", {}) or {}
            # Mean-reversion z-score only fires when meta says so (|z|>2).
            if name == "vwap_zscore_30m" and not md.get("mean_reversion"):
                continue
            value = float(getattr(sig, "value", 0.0) or 0.0)
            kind = "whale" if name == "whale_detected" else "mean_reversion"
            try:
                ts = sig.timestamp.timestamp()
            except Exception:
                ts = time.time()
            # Throttle mean-reversion: one event per (symbol, sign) every 60s.
            # Whales bypass — each large print is independently informative.
            if kind == "mean_reversion":
                key = (symbol, kind, 1 if value > 0 else -1)
                last = self._last_emit.get(key, 0.0)
                if ts - last < 60.0:
                    self._stats["events_deduped"] += 1
                    continue
                self._last_emit[key] = ts
            evt = CryptoEvent(
                symbol=symbol,
                ts_unix=ts,
                kind=kind,
                name=name,
                value=value,
                side=md.get("side") or side,
                metadata={k: v for k, v in md.items() if isinstance(v, (int, float, str, bool))},
            )
            buf.append(evt)
            self._stats["events"] += 1
            # Mirror to Redis if a client is wired (set by lifespan on the
            # leader worker). ZADD with ts as score lets all workers do
            # range queries by time window. ZREMRANGEBYRANK keeps the
            # global set bounded so memory stays constant. EXPIRE refresh
            # ensures the ZSET self-evicts within TTL of the last publish
            # — a dead leader doesn't leave a stale set growing forever.
            if self._redis_client is not None:
                try:
                    import json as _json

                    blob = _json.dumps(evt.as_dict()).encode()
                    self._redis_client.zadd(self._redis_key, {blob: ts})
                    self._redis_client.zremrangebyrank(self._redis_key, 0, -(self._redis_max + 1))
                    self._redis_client.expire(self._redis_key, self._redis_ttl_s)
                except Exception:
                    pass


#: Module-level singleton. The lifespan in ``pfm.main`` wires it up; the
#: router in ``pfm.strategies_crypto_router`` reads it.
_engine: CryptoEventsEngine | None = None


def get_engine() -> CryptoEventsEngine:
    global _engine
    if _engine is None:
        _engine = CryptoEventsEngine()
    return _engine


def read_events_from_redis(
    redis_client: Any,
    *,
    key: str = "arb:crypto_events",
    symbol: str | None = None,
    window_min: float = 5.0,
    kinds: set[str] | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Read crypto events from Redis (the leader worker publishes them).

    Used by gunicorn workers that aren't the WS engine leader so every
    ``/strategies/crypto/events`` request sees the same data regardless of
    which worker handled it.
    """
    if redis_client is None:
        return []
    try:
        cutoff = time.time() - window_min * 60
        # ZRANGEBYSCORE: events whose score (unix ts) is in [cutoff, +inf].
        raw = redis_client.zrangebyscore(
            key,
            cutoff,
            "+inf",
            start=0,
            num=limit,
            withscores=False,
        )
        if not raw:
            return []
        import json as _json

        out: list[dict[str, Any]] = []
        for item in raw:
            try:
                ev = _json.loads(item.decode() if isinstance(item, bytes) else item)
            except Exception:
                continue
            if symbol and (ev.get("symbol") or "").upper() != symbol.upper():
                continue
            if kinds and ev.get("kind") not in kinds:
                continue
            out.append(ev)
        out.sort(key=lambda e: e.get("ts_unix", 0), reverse=True)
        return out
    except Exception:
        return []


def read_model_state_from_redis(
    redis_client: Any,
    symbol: str,
    *,
    key_prefix: str = "pfm:crypto_engine:state:",
) -> dict[str, Any] | None:
    """Read one symbol's ``model_state`` dict from Redis.

    Used by follower gunicorn workers (any worker that didn't win the WS
    leader election). The leader publishes via ``publish_state_now`` at
    1 Hz; followers read here so the crypto5min UI sees consistent live
    engine data regardless of which worker the load-balancer picked.

    Returns ``None`` if Redis is unreachable, the key is missing or
    expired, or the payload doesn't deserialize. The caller (router) then
    falls back to the σ_long-only predictor — same behavior as when the
    engine is entirely off.
    """
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(f"{key_prefix}{symbol.upper()}")
    except Exception:
        return None
    if not raw:
        return None
    try:
        import json as _json

        decoded = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        data = _json.loads(decoded)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None
