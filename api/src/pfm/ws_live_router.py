"""WebSocket live-tick router at ``WS /ws/live``.

A real-time alternative to the SSE stream at ``/strategies/arb/stream``.
Clients connect, subscribe to one or more channels, and receive JSON
envelopes every ~2 s.

Protocol (JSON over WebSocket text frames):

1. **Client → Server** (after connect, must be the FIRST message):
   ``{"action": "subscribe", "channels": ["arb", "jumps", "sentiment"]}``

2. **Server → Client** (one frame per tick, filtered to subscribed channels):
   ``{"channel": "arb", "ts": "<ISO-8601>", "payload": {...}}``

3. **Heartbeat** — server sends ``{"channel": "ping", ...}`` every
   ``HEARTBEAT_INTERVAL_S`` (30 s). Clients may reply with ``{"action": "pong"}``
   (no-op; presence alone is enough to reset the idle timer).

4. **Idle disconnect** — if no client frame arrives within ``IDLE_TIMEOUT_S``
   (60 s), the server closes with code ``4002`` (``idle timeout``).

5. **Errors** — any malformed/unknown action closes with code ``4001``
   (``bad request``). Exceeding ``MAX_CONNECTIONS`` per worker closes new
   connections with ``4003`` (``too many connections``).

Channel payloads:

* ``arb`` — same trimmed envelope as ``GET /strategies/arb/stream`` (best-effort;
  imports lazily and falls back to an empty list on any error).
* ``jumps`` — new jump detections in the last 60 s (deduplicated by slug).
* ``sentiment`` — sentiment-leaderboard delta vs the previous tick.

All three generators are isolated: a failure in one channel never poisons
the others. Each connection has its own generator state so subscribers
see independent deltas.

Why a separate router (and not fold into ``strategies_arb_router``):
WebSocket lifecycle, per-connection state, and shared concurrency caps
are a different shape than SSE's stateless GET stream. Keeping it in its
own module lets us evolve the wire format without disturbing the
existing SSE consumers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

_LOG = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level tunables. Tests monkey-patch these to shrink wall-clock budgets.
# ---------------------------------------------------------------------------

#: Push cadence (seconds). Default 2 s per spec; tests shrink to ~0.05.
TICK_INTERVAL_S: float = float(os.environ.get("PFM_WS_LIVE_TICK_S", "2.0"))

#: Server-initiated heartbeat ping cadence (seconds). 30 s per spec.
HEARTBEAT_INTERVAL_S: float = float(os.environ.get("PFM_WS_LIVE_HEARTBEAT_S", "30.0"))

#: Idle disconnect: no client frame in this many seconds → close 4002.
IDLE_TIMEOUT_S: float = float(os.environ.get("PFM_WS_LIVE_IDLE_S", "60.0"))

#: Per-worker concurrent connection ceiling. New connects past this are
#: closed with 4003 immediately after ``accept()``.
MAX_CONNECTIONS: int = int(os.environ.get("PFM_WS_LIVE_MAX_CONNS", "100"))

#: Valid channel names. Subscribing to anything outside this set closes 4001.
VALID_CHANNELS: frozenset[str] = frozenset({"arb", "jumps", "sentiment"})

# WebSocket application close codes (>=4000 are user-defined per RFC 6455).
CLOSE_BAD_REQUEST = 4001
CLOSE_IDLE_TIMEOUT = 4002
CLOSE_TOO_MANY_CONNS = 4003

# ---------------------------------------------------------------------------
# Process-wide connection counter. Per-worker (process) cap, not per-host —
# behind gunicorn each worker independently enforces ``MAX_CONNECTIONS``.
# ---------------------------------------------------------------------------

_active_connections: int = 0
_connections_lock = asyncio.Lock()


async def _try_acquire_slot() -> bool:
    """Increment the active-connection counter if below the cap. Returns success."""
    global _active_connections
    async with _connections_lock:
        if _active_connections >= MAX_CONNECTIONS:
            return False
        _active_connections += 1
        return True


async def _release_slot() -> None:
    """Decrement the active-connection counter. Never goes below zero."""
    global _active_connections
    async with _connections_lock:
        if _active_connections > 0:
            _active_connections -= 1


def _current_connections() -> int:
    """Snapshot of the active-connection counter (for tests + diagnostics)."""
    return _active_connections


def _now_iso() -> str:
    """ISO-8601 UTC timestamp, second-precision, ``Z`` suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Channel generators. Each is an async generator that yields one dict per
# tick (or ``None`` to skip this tick). Generators are independent across
# connections so per-connection state (e.g. last_seen) does not leak.
# ---------------------------------------------------------------------------


def _arb_payload() -> dict[str, Any]:
    """Best-effort snapshot of the current arb opportunity list.

    Imports lazily so tests don't pay the cost of pulling in the scanner
    module just to verify the websocket plumbing.
    """
    try:
        from pfm import strategies_arb_router as arb_mod  # type: ignore[attr-defined]

        snap_fn = getattr(arb_mod, "_build_state_envelope", None)
        if callable(snap_fn):
            env = snap_fn()
            if isinstance(env, dict):
                # Trim to the same surface the SSE stream sends.
                return {
                    "opportunities": env.get("opportunities", [])[:20],
                    "stats": env.get("stats", {}),
                    "engine_source": env.get("engine_source", "live"),
                }
    except Exception as exc:  # pragma: no cover — defensive
        _LOG.debug("arb_payload upstream error: %s", exc)
    return {"opportunities": [], "stats": {}, "engine_source": "unavailable"}


def _jumps_payload(seen_keys: set[str]) -> dict[str, Any]:
    """New jump detections within the last 60 s, deduplicated by ``(slug, ts)``.

    ``seen_keys`` is mutated in place — caller owns the set lifecycle so
    each connection has independent dedup state.
    """
    try:
        from pfm import jumps as jumps_mod  # type: ignore[attr-defined]

        recent_fn = getattr(jumps_mod, "recent_jumps", None)
        if callable(recent_fn):
            jumps = recent_fn(window_s=60.0) or []
            new_jumps: list[dict[str, Any]] = []
            for jump in jumps:
                if not isinstance(jump, dict):
                    continue
                key = f"{jump.get('slug', '?')}::{jump.get('ts', '?')}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                new_jumps.append(jump)
            return {"new_jumps": new_jumps, "window_s": 60.0}
    except Exception as exc:  # pragma: no cover — defensive
        _LOG.debug("jumps_payload upstream error: %s", exc)
    return {"new_jumps": [], "window_s": 60.0}


def _sentiment_payload(last_snapshot: dict[str, float]) -> dict[str, Any]:
    """Sentiment leaderboard delta vs the previous tick.

    Returns the {topic → score} pairs whose score moved by ≥ 0.01 since the
    previous call. ``last_snapshot`` is mutated in place to track the new
    baseline (per-connection).
    """
    try:
        from pfm import sentiment_factor as sent_mod  # type: ignore[attr-defined]

        snapshot_fn = getattr(sent_mod, "current_leaderboard", None)
        if callable(snapshot_fn):
            current = snapshot_fn() or {}
            delta: dict[str, float] = {}
            for topic, score in current.items():
                prev = last_snapshot.get(topic)
                if prev is None or abs(float(score) - float(prev)) >= 0.01:
                    delta[topic] = float(score)
            last_snapshot.clear()
            last_snapshot.update({k: float(v) for k, v in current.items()})
            return {"delta": delta, "size": len(current)}
    except Exception as exc:  # pragma: no cover — defensive
        _LOG.debug("sentiment_payload upstream error: %s", exc)
    return {"delta": {}, "size": 0}


# Each generator builds a payload dict given any per-connection state.
ChannelBuilder = Callable[[dict[str, Any]], dict[str, Any]]


def _build_channel_builders() -> dict[str, ChannelBuilder]:
    """Wire channel name → callable(state) → payload.

    The state dict is connection-scoped — keys: ``jumps_seen`` (set) and
    ``sentiment_last`` (dict). Returning a fresh mapping on every call so
    monkey-patches in tests are picked up.
    """
    return {
        "arb": lambda _state: _arb_payload(),
        "jumps": lambda state: _jumps_payload(state.setdefault("jumps_seen", set())),
        "sentiment": lambda state: _sentiment_payload(state.setdefault("sentiment_last", {})),
    }


# ---------------------------------------------------------------------------
# Subscribe handshake. Validates the first client frame and returns the
# (channels, error) pair. Empty channels with no error is treated as 4001.
# ---------------------------------------------------------------------------


def _parse_subscribe(raw: str) -> tuple[list[str], str | None]:
    """Parse the subscribe frame. Returns (channels, error_message_or_None)."""
    try:
        msg = json.loads(raw)
    except (TypeError, ValueError):
        return [], "invalid json"
    if not isinstance(msg, dict):
        return [], "frame must be a json object"
    action = msg.get("action")
    if action != "subscribe":
        return [], f"unknown action: {action!r}"
    channels_raw = msg.get("channels")
    if not isinstance(channels_raw, list) or not channels_raw:
        return [], "channels must be a non-empty list"
    bad = [c for c in channels_raw if c not in VALID_CHANNELS]
    if bad:
        return [], f"unknown channel(s): {bad}"
    # Dedupe while preserving insertion order.
    seen: set[str] = set()
    cleaned: list[str] = []
    for c in channels_raw:
        if c not in seen:
            seen.add(c)
            cleaned.append(c)
    return cleaned, None


# ---------------------------------------------------------------------------
# Async helpers — wait for the next client frame with an idle timeout, send
# a frame respecting the open-state of the socket.
# ---------------------------------------------------------------------------


async def _safe_send_json(ws: WebSocket, payload: dict[str, Any]) -> bool:
    """Send a JSON frame, swallowing closed-state errors. Returns success."""
    if ws.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await ws.send_text(json.dumps(payload))
        return True
    except Exception as exc:
        _LOG.debug("ws send failed (likely closed): %s", exc)
        return False


async def _recv_with_timeout(ws: WebSocket, timeout_s: float) -> str | None:
    """Receive a text frame or return ``None`` on timeout."""
    try:
        return await asyncio.wait_for(ws.receive_text(), timeout=timeout_s)
    except TimeoutError:
        return None


# ---------------------------------------------------------------------------
# The endpoint.
# ---------------------------------------------------------------------------


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Real-time multi-channel tick stream over a WebSocket.

    See module docstring for the wire protocol.
    """
    await websocket.accept()

    # Connection cap. We accept first so the client gets a clean close frame
    # (some browsers swallow close codes if the accept never happens).
    if not await _try_acquire_slot():
        await websocket.close(code=CLOSE_TOO_MANY_CONNS, reason="too many connections")
        return

    state: dict[str, Any] = {}
    last_seen = time.monotonic()
    last_ping = time.monotonic()
    channels: list[str] = []

    try:
        # 1. Read subscribe handshake. Allow up to IDLE_TIMEOUT_S for it.
        first = await _recv_with_timeout(websocket, IDLE_TIMEOUT_S)
        if first is None:
            await websocket.close(code=CLOSE_IDLE_TIMEOUT, reason="idle timeout")
            return
        last_seen = time.monotonic()

        channels, err = _parse_subscribe(first)
        if err is not None:
            await _safe_send_json(websocket, {"channel": "error", "ts": _now_iso(), "error": err})
            await websocket.close(code=CLOSE_BAD_REQUEST, reason=err)
            return

        ok = await _safe_send_json(
            websocket,
            {"channel": "subscribed", "ts": _now_iso(), "channels": channels},
        )
        if not ok:
            return

        builders = _build_channel_builders()

        # 2. Tick loop. Push one frame per channel per tick, intersperse with
        #    heartbeats and idle/disconnect checks.
        while True:
            now = time.monotonic()

            # Heartbeat check.
            if now - last_ping >= HEARTBEAT_INTERVAL_S:
                sent = await _safe_send_json(websocket, {"channel": "ping", "ts": _now_iso()})
                if not sent:
                    return
                last_ping = now

            # Idle check.
            if now - last_seen >= IDLE_TIMEOUT_S:
                await websocket.close(code=CLOSE_IDLE_TIMEOUT, reason="idle timeout")
                return

            # One payload per subscribed channel.
            for channel in channels:
                builder = builders.get(channel)
                if builder is None:  # pragma: no cover — defensive
                    continue
                try:
                    payload = builder(state)
                except Exception as exc:  # pragma: no cover — defensive
                    _LOG.debug("channel %s builder error: %s", channel, exc)
                    payload = {"error": "channel temporarily unavailable"}
                sent = await _safe_send_json(
                    websocket,
                    {"channel": channel, "ts": _now_iso(), "payload": payload},
                )
                if not sent:
                    return

            # Sleep until the next tick OR until an inbound frame arrives.
            # Drain any client messages so we can reset the idle clock and
            # honour ``{"action": "unsubscribe"}`` / ``pong`` in the future.
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=TICK_INTERVAL_S)
                last_seen = time.monotonic()
                # Best-effort parse — unknown frames are ignored (not fatal).
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    msg = None
                if isinstance(msg, dict):
                    action = msg.get("action")
                    if action == "pong":
                        # Heartbeat acknowledgement — no reply needed.
                        pass
                    elif action == "unsubscribe":
                        await websocket.close(code=1000, reason="client unsubscribed")
                        return
            except TimeoutError:
                # No client frame in the tick window — that's fine, loop on.
                pass

    except WebSocketDisconnect:
        # Normal client disconnect.
        return
    except Exception as exc:  # pragma: no cover — defensive
        _LOG.warning("ws_live unexpected error: %s", exc)
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011, reason="internal error")
        except Exception:
            pass
    finally:
        await _release_slot()


# ---------------------------------------------------------------------------
# Public surface for tests + main.py wiring.
# ---------------------------------------------------------------------------


__all__ = [
    "CLOSE_BAD_REQUEST",
    "CLOSE_IDLE_TIMEOUT",
    "CLOSE_TOO_MANY_CONNS",
    "HEARTBEAT_INTERVAL_S",
    "IDLE_TIMEOUT_S",
    "MAX_CONNECTIONS",
    "TICK_INTERVAL_S",
    "VALID_CHANNELS",
    "_arb_payload",
    "_current_connections",
    "_jumps_payload",
    "_parse_subscribe",
    "_sentiment_payload",
    "router",
]
