"""Tests for the Polymarket CLOB WebSocket subscriber + REST fallback.

The unit tests feed synthetic ``book`` / ``price_change`` events to
``ingest_message`` and assert that ``get_midpoint`` returns the right value
without ever opening a WebSocket. The integration tests stand up a minimal
asyncio TCP WebSocket server (using the ``websockets`` library that the
implementation depends on) and verify the full
connect → subscribe → recv loop. The Redis tests use a tiny in-process
fake to assert publish + follower-read parity.

All tests are marked sync OR ``@pytest.mark.asyncio`` to match the project's
existing convention; no network access is required.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time

import httpx
import pytest
import respx

from pfm.crypto5min.market_fetcher import (
    CLOB_WS_FRESH_WINDOW_S,
    CLOB_WS_REDIS_PREFIX,
    DEFAULT_CLOB_URL,
    ClobMidpointSubscriber,
    _midpoint_from_book,
    fetch_clob_midpoint,
    get_subscriber,
    set_subscriber,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _book_event(asset_id: str, *, bb: float, ba: float) -> dict:
    """Return a synthetic ``book`` event with one bid + one ask level.

    Polymarket's wire format is ASCENDING price for both books; we use a
    single entry per side so we don't have to worry about ordering here.
    """
    return {
        "event_type": "book",
        "asset_id": asset_id,
        "market": "0xdeadbeef",
        "timestamp": str(int(time.time() * 1000)),
        "bids": [{"price": str(bb), "size": "100"}],
        "asks": [{"price": str(ba), "size": "100"}],
    }


def _price_change_event(*changes: tuple[str, float, float]) -> dict:
    """``changes`` = sequence of (asset_id, best_bid, best_ask)."""
    return {
        "event_type": "price_change",
        "market": "0xdeadbeef",
        "timestamp": str(int(time.time() * 1000)),
        "price_changes": [
            {
                "asset_id": tid,
                "price": str(bb),
                "size": "1.0",
                "side": "BUY",
                "best_bid": str(bb),
                "best_ask": str(ba),
            }
            for tid, bb, ba in changes
        ],
    }


class _FakeRedis:
    """Tiny in-memory Redis stand-in.

    Implements just the operations the subscriber + ``fetch_clob_midpoint``
    use: ``set(name, value, ex=...)`` and ``get(name)``. Times are recorded
    so TTL-expiry assertions can be made deterministically by mutating the
    stored timestamp.
    """

    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, float | None]] = {}

    def set(self, name: str, value: object, ex: int | None = None) -> bool:
        if isinstance(value, str):
            value = value.encode()
        elif not isinstance(value, bytes | bytearray):
            value = str(value).encode()
        expire_at = (time.time() + ex) if ex else None
        self.store[name] = (bytes(value), expire_at)
        return True

    def get(self, name: str) -> bytes | None:
        entry = self.store.get(name)
        if entry is None:
            return None
        value, expire_at = entry
        if expire_at is not None and time.time() > expire_at:
            del self.store[name]
            return None
        return value


@pytest.fixture(autouse=True)
def _clear_singleton() -> None:
    """Make sure tests never leak the process-wide subscriber."""
    set_subscriber(None)
    yield
    set_subscriber(None)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_midpoint_from_book_normal() -> None:
    book = _book_event("tok_a", bb=0.51, ba=0.52)
    assert _midpoint_from_book(book) == pytest.approx(0.515)


def test_midpoint_from_book_picks_max_bid_min_ask() -> None:
    """Polymarket sends bids in ascending price order — best is the LAST."""
    book = {
        "event_type": "book",
        "asset_id": "tok_a",
        "bids": [
            {"price": "0.01", "size": "10"},
            {"price": "0.50", "size": "20"},  # best bid
        ],
        "asks": [
            {"price": "0.52", "size": "20"},  # best ask
            {"price": "0.99", "size": "10"},
        ],
    }
    assert _midpoint_from_book(book) == pytest.approx(0.51)


def test_midpoint_from_book_rejects_crossed() -> None:
    """Best bid >= best ask → crossed book → return None."""
    book = _book_event("tok_a", bb=0.6, ba=0.5)
    assert _midpoint_from_book(book) is None


def test_midpoint_from_book_rejects_empty_side() -> None:
    book = {"event_type": "book", "bids": [], "asks": [{"price": "0.5", "size": "1"}]}
    assert _midpoint_from_book(book) is None


def test_midpoint_from_book_skips_zero_size_rows() -> None:
    book = {
        "event_type": "book",
        "bids": [
            {"price": "0.50", "size": "0"},  # would be best but size=0
            {"price": "0.40", "size": "10"},
        ],
        "asks": [{"price": "0.60", "size": "10"}],
    }
    assert _midpoint_from_book(book) == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# subscriber: ingest_message — no network needed
# ---------------------------------------------------------------------------


def test_ingest_book_event_updates_cache() -> None:
    sub = ClobMidpointSubscriber()
    touched = sub.ingest_message(_book_event("tok_a", bb=0.4, ba=0.5))
    assert touched == ["tok_a"]
    assert sub.get_midpoint("tok_a") == pytest.approx(0.45)


def test_ingest_initial_list_payload() -> None:
    """First message after subscribe arrives as a LIST of book events."""
    sub = ClobMidpointSubscriber()
    msg = [
        _book_event("tok_a", bb=0.4, ba=0.5),
        _book_event("tok_b", bb=0.51, ba=0.55),
    ]
    touched = sub.ingest_message(msg)
    assert sorted(touched) == ["tok_a", "tok_b"]
    assert sub.get_midpoint("tok_a") == pytest.approx(0.45)
    assert sub.get_midpoint("tok_b") == pytest.approx(0.53)


def test_ingest_price_change_uses_best_bid_ask() -> None:
    sub = ClobMidpointSubscriber()
    sub.ingest_message(_price_change_event(("tok_a", 0.48, 0.49), ("tok_b", 0.51, 0.52)))
    assert sub.get_midpoint("tok_a") == pytest.approx(0.485)
    assert sub.get_midpoint("tok_b") == pytest.approx(0.515)


def test_ingest_price_change_skips_crossed() -> None:
    sub = ClobMidpointSubscriber()
    sub.ingest_message(_price_change_event(("tok_a", 0.60, 0.50)))
    assert sub.get_midpoint("tok_a") is None


def test_ingest_price_change_only_keeps_latest_per_asset() -> None:
    """When one message has multiple side-flips for the same asset, the
    LAST entry's best_bid/ask wins."""
    sub = ClobMidpointSubscriber()
    msg = {
        "event_type": "price_change",
        "price_changes": [
            {
                "asset_id": "tok_a",
                "price": "0.1",
                "size": "1",
                "side": "BUY",
                "best_bid": "0.30",
                "best_ask": "0.31",
            },
            {
                "asset_id": "tok_a",
                "price": "0.1",
                "size": "1",
                "side": "SELL",
                "best_bid": "0.45",
                "best_ask": "0.46",
            },
        ],
    }
    sub.ingest_message(msg)
    assert sub.get_midpoint("tok_a") == pytest.approx(0.455)


def test_get_midpoint_returns_none_on_stale_entry() -> None:
    sub = ClobMidpointSubscriber()
    sub.ingest_message(_book_event("tok_a", bb=0.4, ba=0.5))
    # Backdate the entry beyond the freshness window.
    ts, mid = sub._midpoints["tok_a"]
    sub._midpoints["tok_a"] = (ts - CLOB_WS_FRESH_WINDOW_S - 1.0, mid)
    assert sub.get_midpoint("tok_a") is None


def test_ingest_ignores_unknown_event_types() -> None:
    sub = ClobMidpointSubscriber()
    sub.ingest_message({"event_type": "tick_size_change", "asset_id": "tok_a"})
    sub.ingest_message({"event_type": "last_trade_price", "asset_id": "tok_a"})
    assert sub.get_midpoint("tok_a") is None


# ---------------------------------------------------------------------------
# subscriber: token management
# ---------------------------------------------------------------------------


def test_add_remove_tokens_basic() -> None:
    sub = ClobMidpointSubscriber()
    sub.add_tokens(["tok_a", "tok_b"])
    # _pending_add holds them until the consumer loop drains them.
    assert "tok_a" in sub._pending_add
    sub.remove_tokens(["tok_a"])
    assert "tok_a" in sub._pending_remove


def test_replace_tokens_diffs_current_set() -> None:
    sub = ClobMidpointSubscriber()
    sub._subscribed = {"tok_a", "tok_b"}
    sub.replace_tokens(["tok_b", "tok_c"])
    assert "tok_c" in sub._pending_add
    assert "tok_a" in sub._pending_remove
    assert "tok_b" not in sub._pending_add
    assert "tok_b" not in sub._pending_remove


def test_add_tokens_respects_max_subscriptions() -> None:
    sub = ClobMidpointSubscriber(max_subscriptions=2)
    sub._subscribed = {"old_a", "old_b"}
    sub.add_tokens(["new_a"])
    # After admit, the canonical subscribed-to-be set should be <= cap.
    # add_tokens trims _subscribed in-place when admission would overflow.
    assert len(sub._subscribed) + len(sub._pending_add) <= 2


# ---------------------------------------------------------------------------
# Redis publish + follower read
# ---------------------------------------------------------------------------


def test_publish_writes_redis_key_with_ttl() -> None:
    rc = _FakeRedis()
    sub = ClobMidpointSubscriber(redis_client=rc, redis_ttl_s=10)
    sub.ingest_message(_book_event("tok_a", bb=0.4, ba=0.5))
    key = CLOB_WS_REDIS_PREFIX + "tok_a"
    assert key in rc.store
    value, expire_at = rc.store[key]
    body = json.loads(value)
    assert body["mid"] == pytest.approx(0.45)
    assert isinstance(body["ts"], float)
    assert expire_at is not None and expire_at > time.time()


@pytest.mark.asyncio
async def test_fetch_clob_midpoint_prefers_in_process_subscriber() -> None:
    """When a leader subscriber has a fresh midpoint, REST is never called."""
    sub = ClobMidpointSubscriber()
    sub.ingest_message(_book_event("tok_a", bb=0.4, ba=0.5))
    set_subscriber(sub)

    async with httpx.AsyncClient() as client:
        with respx.mock(base_url=DEFAULT_CLOB_URL, assert_all_called=False) as mock:
            route = mock.get("/midpoint").respond(200, json={"mid": 0.9})
            mid = await fetch_clob_midpoint(client, "tok_a")
            assert mid == pytest.approx(0.45)
            assert route.call_count == 0


@pytest.mark.asyncio
async def test_fetch_clob_midpoint_reads_redis_for_follower() -> None:
    """No leader in this process → follower reads what the leader published."""
    # Simulate the leader publishing.
    rc = _FakeRedis()
    leader = ClobMidpointSubscriber(redis_client=rc)
    leader.ingest_message(_book_event("tok_a", bb=0.6, ba=0.7))
    # IMPORTANT: don't register the leader as the singleton — we want the
    # follower path. Clear it explicitly.
    set_subscriber(None)

    async with httpx.AsyncClient() as client:
        with respx.mock(base_url=DEFAULT_CLOB_URL, assert_all_called=False) as mock:
            route = mock.get("/midpoint").respond(200, json={"mid": 0.9})
            mid = await fetch_clob_midpoint(client, "tok_a", redis_client=rc)
            assert mid == pytest.approx(0.65)
            assert route.call_count == 0


@pytest.mark.asyncio
async def test_fetch_clob_midpoint_falls_back_to_rest_when_stale() -> None:
    """Stale Redis entry → fall through to REST."""
    rc = _FakeRedis()
    # Write a stale entry directly (ts well past the freshness window).
    rc.set(
        CLOB_WS_REDIS_PREFIX + "tok_a",
        json.dumps({"mid": 0.42, "ts": time.time() - 120}),
        ex=10,
    )

    async with httpx.AsyncClient() as client:
        with respx.mock(base_url=DEFAULT_CLOB_URL) as mock:
            route = mock.get("/midpoint").respond(200, json={"mid": 0.77})
            mid = await fetch_clob_midpoint(client, "tok_a", redis_client=rc)
            assert mid == pytest.approx(0.77)
            assert route.call_count == 1


@pytest.mark.asyncio
async def test_fetch_clob_midpoint_falls_back_to_rest_when_no_cache() -> None:
    """No subscriber, no redis_client → original REST behaviour."""
    async with httpx.AsyncClient() as client:
        with respx.mock(base_url=DEFAULT_CLOB_URL) as mock:
            mock.get("/midpoint").respond(200, json={"mid": 0.33})
            mid = await fetch_clob_midpoint(client, "tok_a")
            assert mid == pytest.approx(0.33)


@pytest.mark.asyncio
async def test_fetch_clob_midpoint_rest_failure_returns_none() -> None:
    """Backward-compatible: REST 500 → None, same as before."""
    async with httpx.AsyncClient() as client:
        with respx.mock(base_url=DEFAULT_CLOB_URL) as mock:
            mock.get("/midpoint").respond(500)
            mid = await fetch_clob_midpoint(client, "tok_a")
            assert mid is None


# ---------------------------------------------------------------------------
# singleton wiring
# ---------------------------------------------------------------------------


def test_set_and_get_subscriber_singleton() -> None:
    assert get_subscriber() is None
    sub = ClobMidpointSubscriber()
    set_subscriber(sub)
    assert get_subscriber() is sub
    set_subscriber(None)
    assert get_subscriber() is None


# ---------------------------------------------------------------------------
# integration: mock asyncio WebSocket server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscriber_consumes_from_mock_ws_server() -> None:
    """End-to-end: real ``websockets`` client against a tiny local server.

    We boot a ``websockets.serve`` instance, push a synthetic book + a few
    price_change events as soon as a client connects + sends the Market
    subscribe payload, then assert that the subscriber's in-process cache
    reflects them.

    This is the smoke test that proves the connect → send → recv → ingest
    pipeline is wired correctly — the unit tests above only exercise
    ``ingest_message`` in isolation.
    """
    websockets = pytest.importorskip("websockets")

    received_subscribes: list[dict] = []

    async def handler(ws) -> None:  # type: ignore[no-untyped-def]
        # First message from client is the subscribe.
        raw = await ws.recv()
        received_subscribes.append(json.loads(raw))
        # Push 1 initial book snapshot (list) + 2 incremental updates.
        await ws.send(json.dumps([_book_event("tok_a", bb=0.30, ba=0.32)]))
        await ws.send(json.dumps(_price_change_event(("tok_a", 0.40, 0.42))))
        await ws.send(json.dumps(_price_change_event(("tok_a", 0.50, 0.52))))
        # Hold the connection open briefly so the client has time to drain.
        await asyncio.sleep(0.3)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:  # type: ignore[attr-defined]
        port = server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        sub = ClobMidpointSubscriber(
            url=url,
            min_backoff_s=0.05,
            max_backoff_s=0.1,
        )
        sub.add_tokens(["tok_a"])
        await sub.start()
        # Poll until the latest update lands or we time out.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            mid = sub.get_midpoint("tok_a")
            if mid is not None and mid == pytest.approx(0.51):
                break
            await asyncio.sleep(0.05)
        await sub.stop()

    assert received_subscribes, "server never received a subscribe payload"
    assert received_subscribes[0]["type"] == "Market"
    assert "tok_a" in received_subscribes[0]["assets_ids"]
    assert sub.get_midpoint("tok_a", now_unix=time.time()) == pytest.approx(0.51)


@pytest.mark.asyncio
async def test_subscriber_reconnects_with_backoff() -> None:
    """Server closes the connection mid-stream → subscriber reconnects.

    Counts how many times the handler fires and asserts >= 2 (i.e. the
    subscriber didn't just give up after the first disconnect).
    """
    websockets = pytest.importorskip("websockets")
    connect_count = 0

    async def handler(ws) -> None:  # type: ignore[no-untyped-def]
        nonlocal connect_count
        connect_count += 1
        await ws.recv()  # the subscribe
        # Immediately close to force the client to reconnect.
        await ws.close()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:  # type: ignore[attr-defined]
        port = server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        sub = ClobMidpointSubscriber(
            url=url,
            min_backoff_s=0.02,
            max_backoff_s=0.05,
        )
        sub.add_tokens(["tok_a"])
        await sub.start()
        deadline = time.time() + 2.0
        while time.time() < deadline and connect_count < 2:  # noqa: ASYNC110
            await asyncio.sleep(0.05)
        await sub.stop()

    assert connect_count >= 2, "subscriber did not reconnect after disconnect"


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_cancellation_safe() -> None:
    """Calling stop() twice (or before start) must not raise."""
    sub = ClobMidpointSubscriber(url="ws://127.0.0.1:1")  # never connectable
    await sub.stop()  # before start — must be a no-op
    await sub.start()
    # Give the consumer one cycle so it picks up the bad URL + bails into
    # the backoff branch — doesn't matter, we're testing stop().
    await asyncio.sleep(0.05)
    await sub.stop()
    await sub.stop()  # second stop — also a no-op


# ---------------------------------------------------------------------------
# rotation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rotation_callable_drives_replace_tokens() -> None:
    """The rotator's returned list must trigger replace_tokens()."""
    sub = ClobMidpointSubscriber(
        url="ws://127.0.0.1:1",  # never connectable
        rotate_callable=lambda: ["tok_x", "tok_y"],
        rotate_interval_s=0.05,
    )
    # Don't start the consumer; just spin the rotator directly so the
    # test stays hermetic.
    sub._rotate_task = asyncio.create_task(sub._rotation_loop())
    await asyncio.sleep(0.15)
    sub._rotate_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await sub._rotate_task
    pending_add = sub._pending_add | sub._subscribed
    assert "tok_x" in pending_add
    assert "tok_y" in pending_add
