"""Tests for the WebSocket live-tick router at ``WS /ws/live``.

The router exposes a multi-channel real-time stream over a WebSocket. We
exercise it via FastAPI's ``TestClient.websocket_connect`` which speaks
the same handshake/frame protocol as a real client.

Strategy:
- Shrink the tick interval / heartbeat / idle timeout to milliseconds so
  several ticks land inside the test budget.
- Stub the three channel builders so each test has deterministic
  payloads with no network or filesystem fan-out.
- Each test uses its own ``FastAPI()`` app + ``TestClient`` so connection
  caps and global state are isolated (the module's connection counter is
  a real process-wide global, but we reset it between tests too).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import websockets
import websockets.exceptions
from fastapi import FastAPI

from pfm import ws_live_router as wsl

# starlette's ``TestClient.websocket_connect`` uses anyio BlockingPortal and
# deadlocks on ``receive()`` under Python 3.14. We replace it with a small
# adapter that drives the WebSocket over real TCP through ``websockets`` —
# the server is uvicorn-in-a-thread via the ``live_server_factory`` fixture
# (see conftest.py).


class WebSocketDisconnect(Exception):
    """Compat shim for the few tests that raised the starlette class."""

    def __init__(self, code: int, reason: str = "") -> None:
        super().__init__(f"WS {code} {reason}".rstrip())
        self.code = code
        self.reason = reason


class _WSAdapter:
    """Synchronous facade over a ``websockets`` async client.

    Shares one event loop across the lifetime of the connection — the
    ``websockets`` ``ws`` object is loop-bound, so send/recv must run on
    the same loop that performed ``connect``.
    """

    def __init__(self, loop, ws):
        self._loop = loop
        self._ws = ws

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def send_text(self, txt: str) -> None:
        self._run(self._ws.send(txt))

    def send_json(self, obj) -> None:
        self.send_text(json.dumps(obj))

    def _recv(self, timeout: float = 5.0):
        try:
            return self._run(asyncio.wait_for(self._ws.recv(), timeout=timeout))
        except websockets.exceptions.ConnectionClosed as e:
            raise WebSocketDisconnect(code=e.code or 1006, reason=e.reason or "") from e

    def receive(self):
        return self._recv()

    def receive_text(self) -> str:
        return self._recv()

    def receive_json(self):
        return json.loads(self._recv())

    def close(self) -> None:
        try:
            self._run(self._ws.close())
        except Exception:
            pass


class _LiveWSClient:
    """Stand-in for ``TestClient`` exposing ``websocket_connect``."""

    def __init__(self, base_url: str):
        self.base_url = base_url.replace("http://", "ws://", 1).replace(
            "https://",
            "wss://",
            1,
        )

    @contextmanager
    def websocket_connect(self, path: str):
        url = self.base_url + path
        loop = asyncio.new_event_loop()
        try:
            ws = loop.run_until_complete(
                asyncio.wait_for(websockets.connect(url), timeout=5),
            )
        except Exception as e:
            loop.close()
            # Map handshake-level rejects (e.g. 4xx) to the same
            # ``WebSocketDisconnect`` the legacy starlette TestClient raised
            # so callers' ``except WebSocketDisconnect`` arms still trigger.
            code = getattr(e, "status_code", None) or getattr(e, "code", 1006)
            raise WebSocketDisconnect(code=code, reason=str(e)) from e
        adapter = _WSAdapter(loop, ws)
        try:
            yield adapter
        finally:
            adapter.close()
            loop.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_intervals(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Shrink time budgets so the suite stays under a few hundred ms total."""
    monkeypatch.setattr(wsl, "TICK_INTERVAL_S", 0.03)
    monkeypatch.setattr(wsl, "HEARTBEAT_INTERVAL_S", 0.08)
    monkeypatch.setattr(wsl, "IDLE_TIMEOUT_S", 0.5)
    yield


@pytest.fixture(autouse=True)
def _reset_connection_counter() -> Iterator[None]:
    """Force the module-level counter back to zero between tests.

    The counter is process-wide; without a reset, a test that asserts
    "second connection rejected" would pollute the next test's slot
    budget.
    """
    wsl._active_connections = 0
    yield
    wsl._active_connections = 0


@pytest.fixture
def stub_builders(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace the three channel builders with deterministic stubs.

    Returns a counter dict the tests can inspect to verify the right
    channels were invoked.
    """
    counters: dict[str, int] = {"arb": 0, "jumps": 0, "sentiment": 0}

    def _arb(_state: dict) -> dict:
        counters["arb"] += 1
        return {"opportunities": [{"id": f"arb-{counters['arb']}"}], "stats": {}}

    def _jumps(_state: dict) -> dict:
        counters["jumps"] += 1
        return {"new_jumps": [{"slug": f"j-{counters['jumps']}"}], "window_s": 60.0}

    def _sentiment(_state: dict) -> dict:
        counters["sentiment"] += 1
        return {"delta": {"fed-hawkish": 0.5}, "size": 1}

    monkeypatch.setattr(
        wsl,
        "_build_channel_builders",
        lambda: {"arb": _arb, "jumps": _jumps, "sentiment": _sentiment},
    )
    return counters


def _make_client(live_server_factory=None) -> _LiveWSClient:
    """Boot a fresh app + uvicorn and return a websocket-only client."""
    if live_server_factory is None:
        raise RuntimeError("live_server_factory fixture must be threaded through")
    app = FastAPI()
    app.include_router(wsl.router)
    base = live_server_factory(app)
    return _LiveWSClient(base)


@pytest.fixture
def client(live_server_factory) -> _LiveWSClient:
    return _make_client(live_server_factory)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_subscribe_happy_path() -> None:
    """A well-formed subscribe frame returns channels with no error."""
    channels, err = wsl._parse_subscribe(
        json.dumps({"action": "subscribe", "channels": ["arb", "jumps"]})
    )
    assert err is None
    assert channels == ["arb", "jumps"]


def test_parse_subscribe_dedupes() -> None:
    """Duplicate channel names are dropped while preserving first-seen order."""
    channels, err = wsl._parse_subscribe(
        json.dumps({"action": "subscribe", "channels": ["arb", "arb", "jumps", "arb"]})
    )
    assert err is None
    assert channels == ["arb", "jumps"]


def test_parse_subscribe_rejects_invalid_json() -> None:
    """Non-JSON or non-object frames produce an error string."""
    _, err1 = wsl._parse_subscribe("not json at all {")
    assert err1 is not None and "json" in err1.lower()

    _, err2 = wsl._parse_subscribe(json.dumps([1, 2, 3]))
    assert err2 is not None


def test_parse_subscribe_rejects_unknown_channel() -> None:
    """Channels outside ``VALID_CHANNELS`` are rejected with a clear error."""
    _, err = wsl._parse_subscribe(json.dumps({"action": "subscribe", "channels": ["arb", "bogus"]}))
    assert err is not None
    assert "bogus" in err


def test_parse_subscribe_rejects_wrong_action() -> None:
    """Any action other than ``subscribe`` errors out."""
    _, err = wsl._parse_subscribe(json.dumps({"action": "ping", "channels": ["arb"]}))
    assert err is not None
    assert "action" in err.lower()


def test_connect_subscribe_receive(stub_builders: dict[str, int], live_server_factory) -> None:
    """Happy path: connect → subscribe → receive ack + tick frames."""
    client = _make_client(live_server_factory)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "subscribe", "channels": ["arb", "jumps"]}))

        ack = ws.receive_json()
        assert ack["channel"] == "subscribed"
        assert ack["channels"] == ["arb", "jumps"]

        # Collect several frames — we expect interleaved arb + jumps payloads.
        seen_channels: set[str] = set()
        for _ in range(6):
            frame = ws.receive_json()
            seen_channels.add(frame["channel"])
            if {"arb", "jumps"}.issubset(seen_channels):
                break
        assert {"arb", "jumps"}.issubset(seen_channels)


def test_bad_action_closes_with_4001(live_server_factory) -> None:
    """A malformed first frame triggers close code 4001 ``bad request``."""
    client = _make_client(live_server_factory)
    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "barf", "channels": ["arb"]}))
        # First frame is the error payload — then the close.
        for _ in range(5):
            ws.receive()
    assert exc.value.code == wsl.CLOSE_BAD_REQUEST


def test_unknown_channel_closes_with_4001(live_server_factory) -> None:
    """An unknown channel in subscribe also closes 4001."""
    client = _make_client(live_server_factory)
    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "subscribe", "channels": ["arb", "made-up"]}))
        for _ in range(5):
            ws.receive()
    assert exc.value.code == wsl.CLOSE_BAD_REQUEST


def test_heartbeat_emitted(
    monkeypatch: pytest.MonkeyPatch, stub_builders: dict[str, int], live_server_factory
) -> None:
    """With heartbeat interval < tick interval, ping frames appear quickly."""
    # Force heartbeat to fire on the very first iteration of the tick loop.
    monkeypatch.setattr(wsl, "HEARTBEAT_INTERVAL_S", 0.0)
    monkeypatch.setattr(wsl, "TICK_INTERVAL_S", 0.05)

    client = _make_client(live_server_factory)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "subscribe", "channels": ["arb"]}))
        ws.receive_json()  # subscribed ack

        # Within ~15 frames we expect at least one ping.
        saw_ping = False
        for _ in range(15):
            frame = ws.receive_json()
            if frame["channel"] == "ping":
                saw_ping = True
                break
        assert saw_ping, "no heartbeat ping observed"


def test_channel_filter_only_subscribed(stub_builders: dict[str, int], live_server_factory) -> None:
    """Subscribing to only ``sentiment`` produces no arb/jumps frames."""
    client = _make_client(live_server_factory)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "subscribe", "channels": ["sentiment"]}))
        ws.receive_json()  # subscribed ack

        # Collect ~8 frames, then assert only sentiment + ping appear.
        channels_seen: list[str] = []
        for _ in range(8):
            frame = ws.receive_json()
            channels_seen.append(frame["channel"])

        non_meta = [c for c in channels_seen if c not in ("ping", "subscribed")]
        assert non_meta, "got no payload frames at all"
        assert set(non_meta) == {"sentiment"}, (
            f"got channels we didn't subscribe to: {set(non_meta)}"
        )
        # arb + jumps builders were never invoked.
        assert stub_builders["arb"] == 0
        assert stub_builders["jumps"] == 0
        assert stub_builders["sentiment"] >= 1


def test_max_connections_enforced(
    monkeypatch: pytest.MonkeyPatch, stub_builders: dict[str, int], live_server_factory
) -> None:
    """The (N+1)-th concurrent connect is closed with 4003 ``too many``."""
    monkeypatch.setattr(wsl, "MAX_CONNECTIONS", 2)

    client = _make_client(live_server_factory)
    with client.websocket_connect("/ws/live") as ws1:
        ws1.send_text(json.dumps({"action": "subscribe", "channels": ["arb"]}))
        ws1.receive_json()  # ack

        with client.websocket_connect("/ws/live") as ws2:
            ws2.send_text(json.dumps({"action": "subscribe", "channels": ["arb"]}))
            ws2.receive_json()  # ack

            # Third connection should be rejected.
            with (
                pytest.raises(WebSocketDisconnect) as exc,
                client.websocket_connect("/ws/live") as ws3,
            ):
                for _ in range(3):
                    ws3.receive()
            assert exc.value.code == wsl.CLOSE_TOO_MANY_CONNS


def test_idle_timeout_closes_with_4002(
    monkeypatch: pytest.MonkeyPatch, live_server_factory
) -> None:
    """If the client never sends the subscribe frame, the server closes 4002."""
    monkeypatch.setattr(wsl, "IDLE_TIMEOUT_S", 0.05)

    client = _make_client(live_server_factory)
    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect("/ws/live") as ws:
        # Never send anything; wait for the server to time us out.
        for _ in range(5):
            ws.receive()
    assert exc.value.code == wsl.CLOSE_IDLE_TIMEOUT


def test_subscribe_ack_lists_channels(stub_builders: dict[str, int], live_server_factory) -> None:
    """The first frame after subscribe echoes the accepted channel list."""
    client = _make_client(live_server_factory)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "subscribe", "channels": ["sentiment", "arb"]}))
        ack = ws.receive_json()
        assert ack["channel"] == "subscribed"
        assert set(ack["channels"]) == {"sentiment", "arb"}
        assert "ts" in ack


def test_unsubscribe_action_closes_cleanly(
    stub_builders: dict[str, int], live_server_factory
) -> None:
    """A mid-stream ``{"action": "unsubscribe"}`` triggers a clean 1000 close."""
    client = _make_client(live_server_factory)
    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "subscribe", "channels": ["arb"]}))
        ws.receive_json()  # ack
        ws.receive_json()  # at least one payload frame proves we're live
        ws.send_text(json.dumps({"action": "unsubscribe"}))
        for _ in range(5):
            ws.receive()
    assert exc.value.code == 1000


def test_connection_counter_releases_on_disconnect(
    stub_builders: dict[str, int], live_server_factory
) -> None:
    """The active-connection counter goes back to zero after a clean close."""
    import time as _time

    client = _make_client(live_server_factory)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "subscribe", "channels": ["arb"]}))
        ws.receive_json()
        assert wsl._current_connections() == 1
    # Over real TCP, the server's ``finally`` runs asynchronously after the
    # client closes — poll briefly instead of asserting the instant after
    # the context-manager exit.
    deadline = _time.time() + 1.0
    while _time.time() < deadline and wsl._current_connections() != 0:
        _time.sleep(0.02)
    assert wsl._current_connections() == 0


def test_arb_payload_falls_back_when_upstream_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The arb payload returns an empty envelope if the scanner errors out."""
    import sys

    # Pretend the upstream module is broken: replace ``_build_state_envelope``
    # with a raising callable. We import via the real path so a missing
    # module just exercises the import-error branch.
    fake = type(sys)("pfm.strategies_arb_router")
    fake._build_state_envelope = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "pfm.strategies_arb_router", fake)

    out = wsl._arb_payload()
    assert out["opportunities"] == []
    assert out["engine_source"] == "unavailable"
