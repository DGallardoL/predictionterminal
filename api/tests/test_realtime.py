"""Tests for the realtime SSE multiplexing hub and ``/terminal/stream`` endpoint.

The hub is exercised at the unit level (subscribe/unsubscribe + fanout) and
the streaming generator is driven directly with a fake ``Request`` so we
can deterministically test heartbeat, disconnect, and bye-on-slow-client
without standing up a TestClient (which can't cleanly cancel SSE streams).

Sync test functions wrap async work via :func:`asyncio.run` — matches the
existing pattern in ``test_terminal_live_stream.py`` and avoids depending
on the un-installed ``pytest-asyncio`` plugin.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest
import respx

from pfm.realtime import pollers as poll_mod
from pfm.realtime.hub import (
    DROPPED_THRESHOLD,
    MAX_SUBS_PER_CLIENT,
    QUEUE_MAXSIZE,
    RealtimeHub,
)
from pfm.realtime.stream import HEARTBEAT_INTERVAL_S, _generate, format_event, parse_subs

# ---- helpers ---------------------------------------------------------------


class _FakeRequest:
    """Stand-in for ``starlette.requests.Request`` for the streaming generator."""

    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _make_poller(
    name: str,
    counter: dict[str, int],
    *,
    delay_s: float = 0.0,
    payload: dict | None = None,
) -> Callable[[str, httpx.AsyncClient], Awaitable[dict | None]]:
    """Build a poller that records how many times it's called per slug."""

    async def _fn(slug: str, _http: httpx.AsyncClient) -> dict | None:
        counter[slug] = counter.get(slug, 0) + 1
        if delay_s:
            await asyncio.sleep(delay_s)
        return payload or {"type": name, "slug": slug, "data": {"v": counter[slug]}, "ts": 0}

    return _fn


# ---- parse_subs ------------------------------------------------------------


def test_parse_subs_happy_path() -> None:
    assert parse_subs("book:slug-a,tape:slug-a,tick:slug-b") == [
        ("book", "slug-a"),
        ("tape", "slug-a"),
        ("tick", "slug-b"),
    ]


def test_parse_subs_dedupes_and_normalizes() -> None:
    out = parse_subs("TICK:slug-a, tick:slug-a , ,book:slug-b")
    assert out == [("tick", "slug-a"), ("book", "slug-b")]


def test_parse_subs_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported kind"):
        parse_subs("greeks:slug-a")


def test_parse_subs_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="bad subscription"):
        parse_subs("not-a-pair")
    with pytest.raises(ValueError, match="empty kind or slug"):
        parse_subs("tick:")


# ---- format_event ----------------------------------------------------------


def test_format_event_shape() -> None:
    frame = format_event("tick", {"slug": "x", "data": {"mid": 0.5}})
    text = frame.decode()
    assert text.startswith("event: tick\n")
    assert "data: " in text
    assert text.endswith("\n\n")
    body_line = next(ln for ln in text.splitlines() if ln.startswith("data:"))
    payload = json.loads(body_line[len("data: ") :])
    assert payload["slug"] == "x"


# ---- hub: pub/sub ----------------------------------------------------------


def test_two_clients_share_one_poller() -> None:
    """Two subscribers to the same (kind, slug) must produce a SINGLE poller task."""

    async def _run() -> None:
        counter: dict[str, int] = {}
        hub = RealtimeHub(
            http_client=httpx.AsyncClient(),
            poll_interval_s=0.05,
            pollers={"tick": _make_poller("tick", counter)},
        )
        try:
            await hub.create_client("c1")
            await hub.create_client("c2")
            await hub.subscribe("c1", "tick", "slug-a")
            await hub.subscribe("c2", "tick", "slug-a")

            assert len(hub.pollers) == 1
            assert hub.slug_subs[("tick", "slug-a")] == {"c1", "c2"}

            # Wait for at least 2 poll cycles to populate queues.
            await asyncio.sleep(0.15)
            assert hub.clients["c1"].queue.qsize() >= 1
            assert hub.clients["c2"].queue.qsize() >= 1
        finally:
            await hub.shutdown()

    asyncio.run(_run())


def test_unsubscribe_last_client_stops_poller() -> None:
    async def _run() -> None:
        counter: dict[str, int] = {}
        hub = RealtimeHub(
            http_client=httpx.AsyncClient(),
            poll_interval_s=0.05,
            pollers={"tick": _make_poller("tick", counter)},
        )
        try:
            await hub.create_client("c1")
            await hub.subscribe("c1", "tick", "slug-a")
            assert ("tick", "slug-a") in hub.pollers

            await hub.unsubscribe("c1", "tick", "slug-a")
            assert ("tick", "slug-a") not in hub.pollers
            assert ("tick", "slug-a") not in hub.slug_subs
        finally:
            await hub.shutdown()

    asyncio.run(_run())


def test_remove_client_cleans_state() -> None:
    async def _run() -> None:
        counter: dict[str, int] = {}
        hub = RealtimeHub(
            http_client=httpx.AsyncClient(),
            poll_interval_s=0.05,
            pollers={"tick": _make_poller("tick", counter)},
        )
        try:
            await hub.create_client("c1")
            await hub.subscribe("c1", "tick", "slug-a")
            await hub.subscribe("c1", "tick", "slug-b")

            await hub.remove_client("c1")
            assert "c1" not in hub.clients
            assert hub.pollers == {}
            assert hub.slug_subs == {}
        finally:
            await hub.shutdown()

    asyncio.run(_run())


def test_subscribe_unknown_kind_raises() -> None:
    async def _run() -> None:
        hub = RealtimeHub(http_client=httpx.AsyncClient(), poll_interval_s=0.05)
        try:
            await hub.create_client("c1")
            with pytest.raises(ValueError):
                await hub.subscribe("c1", "greeks", "slug-a")
        finally:
            await hub.shutdown()

    asyncio.run(_run())


def test_max_subs_per_client_rejects_overflow() -> None:
    async def _run() -> None:
        counter: dict[str, int] = {}
        hub = RealtimeHub(
            http_client=httpx.AsyncClient(),
            poll_interval_s=10.0,  # don't actually poll while filling
            pollers={"tick": _make_poller("tick", counter)},
        )
        try:
            await hub.create_client("c1")
            for i in range(MAX_SUBS_PER_CLIENT):
                await hub.subscribe("c1", "tick", f"slug-{i}")
            with pytest.raises(RuntimeError, match="too_many_subs"):
                await hub.subscribe("c1", "tick", "slug-overflow")
        finally:
            await hub.shutdown()

    asyncio.run(_run())


# ---- hub: backpressure -----------------------------------------------------


def test_slow_client_does_not_block_fast_one() -> None:
    """If c1 never drains its queue, c2 must still receive events."""

    async def _run() -> None:
        counter: dict[str, int] = {}
        hub = RealtimeHub(
            http_client=httpx.AsyncClient(),
            poll_interval_s=0.02,
            pollers={"tick": _make_poller("tick", counter)},
        )
        try:
            await hub.create_client("slow")
            await hub.create_client("fast")
            await hub.subscribe("slow", "tick", "slug-a")
            await hub.subscribe("fast", "tick", "slug-a")

            drained: list[dict] = []
            for _ in range(5):
                evt = await asyncio.wait_for(hub.clients["fast"].queue.get(), timeout=2.0)
                drained.append(evt)
            assert len(drained) == 5
            assert all(e["type"] == "tick" for e in drained)
        finally:
            await hub.shutdown()

    asyncio.run(_run())


def test_coalescing_replaces_pending_event_for_same_key() -> None:
    """Direct test of ``_enqueue``: same (type, slug) coalesces."""

    async def _run() -> None:
        hub = RealtimeHub(http_client=httpx.AsyncClient(), poll_interval_s=10.0)
        try:
            session = await hub.create_client("c1")
            e1 = {"type": "tick", "slug": "s", "data": {"v": 1}, "ts": 0}
            e2 = {"type": "tick", "slug": "s", "data": {"v": 2}, "ts": 1}
            e3 = {"type": "tick", "slug": "s", "data": {"v": 3}, "ts": 2}
            hub._enqueue(session, e1)
            hub._enqueue(session, e2)
            hub._enqueue(session, e3)
            # Only one slot occupied — replaced in place each time.
            assert session.queue.qsize() == 1
            evt = session.queue.get_nowait()
            assert evt["data"]["v"] == 3
        finally:
            await hub.shutdown()

    asyncio.run(_run())


def test_drop_threshold_emits_bye_and_marks_closed() -> None:
    """Once a slow client crosses ``DROPPED_THRESHOLD`` inside the window,
    the hub closes the session and queues a ``bye`` event."""

    async def _run() -> None:
        hub = RealtimeHub(http_client=httpx.AsyncClient(), poll_interval_s=10.0)
        try:
            session = await hub.create_client("c1")
            # Fill queue with DIFFERENT (kind, slug) tuples so coalescing doesn't kick in.
            for i in range(QUEUE_MAXSIZE):
                session.queue.put_nowait({"type": "tick", "slug": f"s{i}", "data": {}, "ts": i})
            assert session.queue.full()

            # Now hammer with new keys → puts will overflow → drops accumulate.
            for i in range(DROPPED_THRESHOLD + 1):
                hub._enqueue(
                    session,
                    {"type": "book", "slug": f"new{i}", "data": {}, "ts": i},
                )
            assert session.closed

            # A bye should be queued (we made room by popping one slot).
            seen_bye = False
            while not session.queue.empty():
                evt = session.queue.get_nowait()
                if evt.get("type") == "bye":
                    seen_bye = True
                    assert evt["data"]["reason"] == "slow_client"
            assert seen_bye, "expected a bye event after threshold"
        finally:
            await hub.shutdown()

    asyncio.run(_run())


# ---- hub: shutdown ---------------------------------------------------------


def test_shutdown_cancels_all_pollers() -> None:
    async def _run() -> None:
        counter: dict[str, int] = {}
        hub = RealtimeHub(
            http_client=httpx.AsyncClient(),
            poll_interval_s=0.05,
            pollers={"tick": _make_poller("tick", counter)},
        )
        await hub.create_client("c1")
        await hub.subscribe("c1", "tick", "slug-a")
        await hub.subscribe("c1", "tick", "slug-b")
        assert len(hub.pollers) == 2
        await hub.shutdown()
        assert hub.pollers == {}
        assert hub.clients == {}

    asyncio.run(_run())


# ---- streaming generator: heartbeat & disconnect ---------------------------


def test_generator_emits_ready_and_event_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator emits ``ready``, real events, and a periodic ``hb`` frame."""
    # Compress the heartbeat interval so we don't have to wait 10s.
    import pfm.realtime.stream as stream_mod

    monkeypatch.setattr(stream_mod, "HEARTBEAT_INTERVAL_S", 0.01)
    monkeypatch.setattr(stream_mod, "QUEUE_GET_TIMEOUT_S", 0.02)

    async def _run() -> list[bytes]:
        counter: dict[str, int] = {}
        hub = RealtimeHub(
            http_client=httpx.AsyncClient(),
            poll_interval_s=0.02,
            pollers={"tick": _make_poller("tick", counter)},
        )
        try:
            await hub.create_client("c1")
            await hub.subscribe("c1", "tick", "slug-a")
            request = _FakeRequest()
            gen = stream_mod._generate(request, hub, "c1", [("tick", "slug-a")])
            frames: list[bytes] = []
            async for f in gen:
                frames.append(f)
                if len(frames) >= 4:
                    request.disconnected = True
            return frames
        finally:
            await hub.shutdown()

    frames = asyncio.run(_run())
    assert frames, "expected at least one frame"
    assert frames[0].startswith(b"event: ready\n")
    kinds = {f.split(b"\n")[0] for f in frames}
    assert b"event: tick" in kinds
    assert b"event: hb" in kinds


def test_generator_exits_on_disconnect() -> None:
    async def _run() -> list[bytes]:
        counter: dict[str, int] = {}
        hub = RealtimeHub(
            http_client=httpx.AsyncClient(),
            poll_interval_s=10.0,
            pollers={"tick": _make_poller("tick", counter)},
        )
        try:
            await hub.create_client("c1")
            await hub.subscribe("c1", "tick", "slug-a")
            request = _FakeRequest()
            request.disconnected = True  # disconnect immediately
            gen = _generate(request, hub, "c1", [("tick", "slug-a")])
            return [f async for f in gen]
        finally:
            await hub.shutdown()

    frames = asyncio.run(_run())
    # Only the initial ready frame.
    assert len(frames) == 1
    assert frames[0].startswith(b"event: ready\n")


# ---- pollers (mocked HTTP) -------------------------------------------------


@respx.mock
def test_poll_tick_against_mocked_clob() -> None:
    poll_mod.clear_token_cache()
    gamma = "https://gamma-test.example"
    clob = "https://clob-test.example"
    poll_mod.set_endpoints(gamma_url=gamma, clob_url=clob)
    try:
        respx.get(f"{gamma}/markets", params={"slug": "slug-a"}).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "slug": "slug-a",
                        "clobTokenIds": json.dumps(["tok-yes", "tok-no"]),
                    }
                ],
            )
        )
        respx.get(f"{clob}/midpoint", params={"token_id": "tok-yes"}).mock(
            return_value=httpx.Response(200, json={"mid": "0.42"})
        )
        respx.get(f"{clob}/price", params={"token_id": "tok-yes", "side": "BUY"}).mock(
            return_value=httpx.Response(200, json={"price": "0.43"})
        )
        respx.get(f"{clob}/price", params={"token_id": "tok-yes", "side": "SELL"}).mock(
            return_value=httpx.Response(200, json={"price": "0.41"})
        )

        async def _go() -> dict | None:
            async with httpx.AsyncClient() as http:
                return await poll_mod.poll_tick("slug-a", http)

        evt = asyncio.run(_go())
        assert evt is not None
        assert evt["type"] == "tick"
        assert evt["slug"] == "slug-a"
        assert evt["data"]["mid"] == pytest.approx(0.42)
        assert evt["data"]["bid"] == pytest.approx(0.41)
        assert evt["data"]["ask"] == pytest.approx(0.43)
    finally:
        poll_mod.clear_token_cache()
        poll_mod.set_endpoints(
            gamma_url="https://gamma-api.polymarket.com",
            clob_url="https://clob.polymarket.com",
        )


@respx.mock
def test_poll_tick_returns_none_for_bad_slug() -> None:
    poll_mod.clear_token_cache()
    gamma = "https://gamma-test.example"
    clob = "https://clob-test.example"
    poll_mod.set_endpoints(gamma_url=gamma, clob_url=clob)
    try:
        respx.get(f"{gamma}/markets", params={"slug": "ghost"}).mock(
            return_value=httpx.Response(200, json=[])
        )

        async def _go() -> dict | None:
            async with httpx.AsyncClient() as http:
                return await poll_mod.poll_tick("ghost", http)

        evt = asyncio.run(_go())
        assert evt is None
    finally:
        poll_mod.clear_token_cache()
        poll_mod.set_endpoints(
            gamma_url="https://gamma-api.polymarket.com",
            clob_url="https://clob.polymarket.com",
        )


# ---- endpoint shape via TestClient -----------------------------------------


def test_endpoint_validates_subs(app_client) -> None:
    """The /terminal/stream endpoint must reject malformed subs with 400."""
    # Empty subs.
    r = app_client.get("/terminal/stream?subs=")
    assert r.status_code in (400, 422)
    # Unknown kind.
    r = app_client.get("/terminal/stream?subs=greeks:slug-a")
    assert r.status_code == 400


def test_constants_match_design_spec() -> None:
    """Sanity-check the constants the spec calls out."""
    assert MAX_SUBS_PER_CLIENT == 60
    assert HEARTBEAT_INTERVAL_S == 10.0
    assert DROPPED_THRESHOLD == 50
    assert QUEUE_MAXSIZE == 256
