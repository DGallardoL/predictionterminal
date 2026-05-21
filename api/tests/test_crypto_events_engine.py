"""Tests for ``pfm.crypto_events_engine``.

Two concerns covered here:

1. **σ_annual recovery** — the production bug was ``sigma_annual = None``
   even on the leader worker because ``returns_15m`` wasn't reliably
   populated. The fix derives returns inline from ``trades_5m`` /
   ``trades_30m`` prices. These tests feed a stub SignalEngine that mimics
   the upstream ``cryptostuff`` ``_SymbolState`` shape (no need to install
   the full cryptostuff stack) and assert ``model_state`` returns a real
   number.

2. **Cross-worker state visibility** — the leader publishes
   ``model_state`` to Redis at 1 Hz so follower gunicorn workers' router
   sees the same dict. We mock the Redis client and round-trip the
   publish + follower-read path.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from pfm.crypto_events_engine import (
    CryptoEventsEngine,
    read_model_state_from_redis,
)

# --- helpers ---------------------------------------------------------------


class _FakeSymbolState:
    """Subset of cryptostuff's ``_SymbolState`` that ``model_state`` reads."""

    def __init__(self) -> None:
        self.trades_1m: list[tuple[float, float, float, float, float]] = []
        self.trades_5m: list[tuple[float, float, float, float, float]] = []
        self.trades_30m: list[tuple[float, float, float, float, float]] = []
        self.returns_15m: list[float] = []
        self.last_price: float = 0.0


class _FakeSignalEngine:
    """Mimics ``models.SignalEngine`` just enough for ``model_state``."""

    def __init__(self) -> None:
        self._states: dict[str, _FakeSymbolState] = {}

    def get_state_summary(self, symbol: str) -> dict[str, Any] | None:
        s = self._states.get(symbol)
        if s is None:
            return None
        return {
            "last_price": s.last_price,
            "whale_threshold": 0.0,
            "total_trades": len(s.trades_30m),
        }


def _seed_state(
    sym: str = "BTCUSDT",
    *,
    n_trades: int = 60,
    base_price: float = 60_000.0,
    fill_returns_15m: bool = True,
) -> tuple[_FakeSignalEngine, _FakeSymbolState]:
    """Build a SignalEngine + state populated with deterministic random walk."""
    eng = _FakeSignalEngine()
    s = _FakeSymbolState()
    eng._states[sym] = s
    now = time.time()
    price = base_price
    prev = None
    # Deterministic-looking but non-degenerate price walk so std(returns) > 0.
    for i in range(n_trades):
        ts = now - (n_trades - i)  # one trade per sec
        # alternating +/- delta produces a clean non-zero RV
        delta = (-1) ** i * (price * 0.0001 * (1 + (i % 5)))
        price = max(1.0, price + delta)
        qty = 0.1
        notional = price * qty
        signed_qty = qty if i % 2 == 0 else -qty
        rec = (ts, price, qty, notional, signed_qty)
        s.trades_1m.append(rec)
        s.trades_5m.append(rec)
        s.trades_30m.append(rec)
        if fill_returns_15m and prev is not None and prev > 0 and price > 0:
            s.returns_15m.append(math.log(price / prev))
        prev = price
    s.last_price = price
    return eng, s


# --- σ_annual fix ----------------------------------------------------------


def test_sigma_annual_recovered_from_returns_15m() -> None:
    """When returns_15m is populated (happy path), σ_annual is in range."""
    eng_lib, _state = _seed_state(n_trades=80, fill_returns_15m=True)
    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    out = engine.model_state("BTCUSDT")

    assert out is not None
    assert out["rv_per_trade"] is not None
    assert out["rv_per_trade"] > 0
    assert out["sigma_annual"] is not None
    # Clipped to [0.10, 3.0].
    assert 0.10 <= out["sigma_annual"] <= 3.0
    # Sanity: tps from 1m window ≈ 1 trade/sec (we seeded one per second).
    assert out["trades_per_sec"] > 0


def test_sigma_annual_recovered_inline_when_returns_15m_empty() -> None:
    """Bug-fix path: returns_15m empty, model_state must still annualize σ."""
    eng_lib, state = _seed_state(n_trades=80, fill_returns_15m=False)
    # Explicitly drop the precomputed buffer to simulate the production bug
    # where the engine wasn't feeding returns_15m for some reason.
    state.returns_15m = []
    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    out = engine.model_state("BTCUSDT")

    assert out is not None
    # rv_per_trade must come from the inline log(p_t/p_{t-1}) computation.
    assert out["rv_per_trade"] is not None
    assert out["rv_per_trade"] > 0
    assert out["sigma_annual"] is not None
    assert 0.10 <= out["sigma_annual"] <= 3.0


def test_sigma_annual_none_when_trades_too_few() -> None:
    """Cold-start path: only a handful of trades → σ unknown, no exception."""
    eng_lib, _state = _seed_state(n_trades=4, fill_returns_15m=False)
    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    out = engine.model_state("BTCUSDT")

    assert out is not None
    # < 11 trades → not enough points to compute σ either way.
    assert out["rv_per_trade"] is None
    assert out["sigma_annual"] is None


def test_tps_falls_back_to_5m_when_1m_empty() -> None:
    """Sparse-pair path: trades_1m drained but trades_5m has entries."""
    eng_lib, state = _seed_state(n_trades=40, fill_returns_15m=True)
    # Simulate pruning gap: 1m bucket empty, 5m bucket retains the history.
    state.trades_1m = []
    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    out = engine.model_state("BTCUSDT")

    assert out is not None
    assert out["trades_per_sec"] > 0
    assert out["sigma_annual"] is not None


def test_model_state_none_for_unknown_symbol() -> None:
    eng_lib, _ = _seed_state(sym="BTCUSDT")
    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    assert engine.model_state("ETHUSDT") is None


def test_model_state_none_when_engine_obj_missing() -> None:
    engine = CryptoEventsEngine()
    assert engine.model_state("BTCUSDT") is None


# --- Redis publish / read round-trip ---------------------------------------


def test_publish_state_now_setex_with_ttl() -> None:
    """Leader's publish path: SETEX with TTL, key derived from prefix+sym."""
    eng_lib, _ = _seed_state(sym="BTCUSDT", n_trades=80)
    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    fake_redis = MagicMock()
    engine.attach_redis(
        fake_redis,
        state_key_prefix="pfm:crypto_engine:state:",
        state_ttl_seconds=5,
    )

    assert engine.publish_state_now("BTCUSDT") is True

    fake_redis.setex.assert_called_once()
    args, _kwargs = fake_redis.setex.call_args
    assert args[0] == "pfm:crypto_engine:state:BTCUSDT"
    assert args[1] == 5  # TTL
    payload = json.loads(args[2].decode())
    assert payload["symbol"] == "BTCUSDT"
    assert payload["sigma_annual"] is not None


def test_publish_state_now_returns_false_when_no_redis() -> None:
    eng_lib, _ = _seed_state()
    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    assert engine.publish_state_now("BTCUSDT") is False  # no redis attached


def test_publish_state_now_returns_false_when_no_state() -> None:
    """``publish_state_now`` must not crash on unknown symbols."""
    engine = CryptoEventsEngine()
    engine.attach_redis(MagicMock())
    assert engine.publish_state_now("DOESNOTEXIST") is False


def test_publish_state_swallows_redis_errors() -> None:
    """A flaky Redis MUST NOT propagate into the engine's loop."""
    eng_lib, _ = _seed_state(n_trades=80)
    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    fake_redis = MagicMock()
    fake_redis.setex.side_effect = RuntimeError("redis blew up")
    engine.attach_redis(fake_redis)

    assert engine.publish_state_now("BTCUSDT") is False  # logged, no raise


def test_state_key_for_uppercases_symbol() -> None:
    engine = CryptoEventsEngine()
    assert engine.state_key_for("btcusdt") == "pfm:crypto_engine:state:BTCUSDT"
    engine.attach_redis(MagicMock(), state_key_prefix="custom:")
    assert engine.state_key_for("ethusdt") == "custom:ETHUSDT"


def test_read_model_state_from_redis_roundtrip() -> None:
    """Full round-trip: leader publishes, follower reads back the same dict."""
    eng_lib, _ = _seed_state(sym="BTCUSDT", n_trades=80)
    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    # Use a dict as a tiny KV stand-in so we don't depend on fakeredis.
    store: dict[str, bytes] = {}

    fake_redis = MagicMock()
    fake_redis.setex.side_effect = lambda k, ttl, v: store.__setitem__(
        k, v if isinstance(v, bytes) else v.encode()
    )
    fake_redis.get.side_effect = store.get
    engine.attach_redis(fake_redis)

    engine.publish_state_now("BTCUSDT")

    follower_view = read_model_state_from_redis(fake_redis, "BTCUSDT")
    assert follower_view is not None
    assert follower_view["symbol"] == "BTCUSDT"
    assert follower_view["sigma_annual"] is not None
    assert follower_view["rv_per_trade"] is not None


def test_read_model_state_from_redis_missing_key_returns_none() -> None:
    fake_redis = MagicMock()
    fake_redis.get.return_value = None
    assert read_model_state_from_redis(fake_redis, "BTCUSDT") is None


def test_read_model_state_from_redis_handles_redis_error() -> None:
    fake_redis = MagicMock()
    fake_redis.get.side_effect = RuntimeError("boom")
    assert read_model_state_from_redis(fake_redis, "BTCUSDT") is None


def test_read_model_state_from_redis_none_client() -> None:
    assert read_model_state_from_redis(None, "BTCUSDT") is None


def test_read_model_state_from_redis_decodes_str_payload() -> None:
    """Redis clients with decode_responses=True return str, not bytes."""
    fake_redis = MagicMock()
    fake_redis.get.return_value = json.dumps({"symbol": "BTCUSDT", "sigma_annual": 0.55})
    out = read_model_state_from_redis(fake_redis, "BTCUSDT")
    assert out is not None
    assert out["sigma_annual"] == 0.55


def test_read_model_state_from_redis_bad_json_returns_none() -> None:
    fake_redis = MagicMock()
    fake_redis.get.return_value = b"\xff not json"
    assert read_model_state_from_redis(fake_redis, "BTCUSDT") is None


# --- background state-publisher loop ---------------------------------------


@pytest.mark.asyncio
async def test_state_publisher_loop_publishes_then_stops() -> None:
    """The 1 Hz loop SHOULD publish each symbol's state then exit on stop()."""
    eng_lib, _ = _seed_state(sym="BTCUSDT", n_trades=80)
    # Add a second symbol so we exercise multi-symbol iteration.
    eng_lib._states["ETHUSDT"] = _seed_state(sym="ETHUSDT", n_trades=80)[1]

    engine = CryptoEventsEngine()
    engine._engine_obj = eng_lib

    fake_redis = MagicMock()
    engine.attach_redis(fake_redis, state_ttl_seconds=5)

    import asyncio
    import contextlib

    # Run one iteration of the loop with a short interval, then cancel.
    task = asyncio.create_task(engine._state_publisher_loop(interval_seconds=0.05))
    await asyncio.sleep(0.12)  # ~2 iterations
    engine._stop.set()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    keys_called = {c.args[0] for c in fake_redis.setex.call_args_list}
    assert "pfm:crypto_engine:state:BTCUSDT" in keys_called
    assert "pfm:crypto_engine:state:ETHUSDT" in keys_called
