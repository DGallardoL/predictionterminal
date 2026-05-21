"""Tests for the follower-worker fallback in ``_live_engine_state``.

On the leader gunicorn worker the WS engine is running and
``model_state(symbol)`` returns a dict. On every *other* worker
(followers) the engine handle is dead and used to return ``None``,
causing the crypto5min UI to flicker ``live_engine_used=true/false``
depending on which worker the round-robin load-balancer happened to hit.

These tests pin down the new resolution order:
  1. Leader-local call (if engine is running and has state).
  2. Redis read at ``pfm:crypto_engine:state:{SYMBOL}`` (follower path).
  3. Return ``None`` (engine off + Redis empty).
"""

from __future__ import annotations

import importlib
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ``pfm.crypto5min.__init__`` re-exports ``router`` (the APIRouter), which
# shadows the submodule attribute on the package. Grab the real module via
# importlib to keep patch.object working against the module's namespace.
router_mod = importlib.import_module("pfm.crypto5min.router")
_live_engine_state = router_mod._live_engine_state
_reset_redis_state_client = router_mod._reset_redis_state_client


@pytest.fixture(autouse=True)
def _reset_module_redis_cache() -> None:
    _reset_redis_state_client()
    yield
    _reset_redis_state_client()


def _payload(sym: str = "BTCUSDT", sigma: float = 0.55) -> bytes:
    return json.dumps(
        {
            "symbol": sym,
            "last_price": 60_000.0,
            "trades_per_sec": 5.0,
            "rv_per_trade": 1e-5,
            "sigma_annual": sigma,
            "ofi_1m": 0.12,
            "signed_volume_1m": 1.5,
            "n_trades_1m": 60,
            "n_trades_5m": 250,
            "vwap_30m": 59_950.0,
            "z_vwap_30m": 0.4,
            "whale_threshold": 1_000_000.0,
            "total_trades": 4_200,
        }
    ).encode()


def _stub_engine(*, running: bool, local_state: dict[str, Any] | None) -> MagicMock:
    eng = MagicMock()
    eng.is_running.return_value = running
    eng.model_state.return_value = local_state
    return eng


def test_leader_local_state_takes_precedence_over_redis() -> None:
    """Even if Redis has a stale dict, the leader returns its in-memory one."""
    local = {"symbol": "BTCUSDT", "sigma_annual": 0.71, "ofi_1m": 0.0}
    redis = MagicMock()
    redis.get.return_value = _payload(sigma=0.10)  # different value
    redis.ping.return_value = True

    with (
        patch.object(
            router_mod,
            "_get_redis_for_state_reads",
            return_value=redis,
        ),
        patch(
            "pfm.crypto_events_engine.get_engine",
            return_value=_stub_engine(running=True, local_state=local),
        ),
    ):
        out = _live_engine_state("BTCUSDT")

    assert out is not None
    assert out["sigma_annual"] == 0.71  # leader wins; redis ignored
    # The leader path short-circuits — Redis must not be consulted.
    redis.get.assert_not_called()


def test_follower_reads_from_redis_when_engine_off() -> None:
    """Leader returns None (follower process); Redis has the state."""
    redis = MagicMock()
    redis.get.return_value = _payload(sigma=0.62)

    with (
        patch.object(
            router_mod,
            "_get_redis_for_state_reads",
            return_value=redis,
        ),
        patch(
            "pfm.crypto_events_engine.get_engine",
            return_value=_stub_engine(running=False, local_state=None),
        ),
    ):
        out = _live_engine_state("BTCUSDT")

    assert out is not None
    assert out["sigma_annual"] == 0.62
    redis.get.assert_called_once_with("pfm:crypto_engine:state:BTCUSDT")


def test_follower_returns_none_when_redis_key_missing() -> None:
    redis = MagicMock()
    redis.get.return_value = None

    with (
        patch.object(
            router_mod,
            "_get_redis_for_state_reads",
            return_value=redis,
        ),
        patch(
            "pfm.crypto_events_engine.get_engine",
            return_value=_stub_engine(running=False, local_state=None),
        ),
    ):
        out = _live_engine_state("BTCUSDT")

    assert out is None  # follower correctly degrades to σ_long-only


def test_follower_returns_none_when_redis_unreachable() -> None:
    """Redis client cache returns None → fallback path is short-circuited."""
    with (
        patch.object(
            router_mod,
            "_get_redis_for_state_reads",
            return_value=None,
        ),
        patch(
            "pfm.crypto_events_engine.get_engine",
            return_value=_stub_engine(running=False, local_state=None),
        ),
    ):
        out = _live_engine_state("BTCUSDT")

    assert out is None


def test_redis_client_is_cached_after_first_success() -> None:
    """``_get_redis_for_state_reads`` MUST NOT reconnect on every call."""
    fake_client = MagicMock()
    fake_client.ping.return_value = True

    with patch("redis.Redis.from_url", return_value=fake_client) as from_url:
        c1 = router_mod._get_redis_for_state_reads()
        c2 = router_mod._get_redis_for_state_reads()
    assert c1 is c2
    from_url.assert_called_once()


def test_redis_client_disabled_sentinel_after_failure() -> None:
    """One ping failure → sentinel, no further reconnect attempts."""
    fake_client = MagicMock()
    fake_client.ping.side_effect = RuntimeError("nope")

    with patch("redis.Redis.from_url", return_value=fake_client) as from_url:
        assert router_mod._get_redis_for_state_reads() is None
        assert router_mod._get_redis_for_state_reads() is None
        assert router_mod._get_redis_for_state_reads() is None
    # Subsequent calls return None immediately without re-probing.
    assert from_url.call_count == 1
