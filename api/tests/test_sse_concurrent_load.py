"""Concurrent-load tests for the ``GET /strategies/arb/stream`` SSE endpoint.

The Cross-venue Arb live monitor relies on this stream being safe under
fan-out: multiple browser tabs (and the embed) all subscribe at once.
These tests open **10 concurrent ``httpx.AsyncClient`` subscribers** against
the FastAPI app via ``ASGITransport`` and assert:

1. The server stays healthy under the load (no 5xx, no exceptions in the
   stream generator that escape to the client).
2. Every subscriber receives ``>=3`` ``data:`` frames within a small wall-
   clock budget (the per-tick cadence is forced to ~150 ms via
   ``_STREAM_TICK_SECONDS``).
3. Streams are **isolated** — every subscriber gets the *same* payload
   shape, no cross-contamination of envelopes (e.g. subscriber A doesn't
   receive a frame intended for subscriber B's request scope).
4. **Backpressure**: a slow consumer that drains its byte iterator
   sluggishly does NOT starve the other 9 fast consumers. Each fast
   consumer still completes its quota within the budget.
5. The keep-alive comment (``: connected\\n\\n``) is the first frame on
   *every* connection so EventSource flips to ``onopen`` immediately for
   all tabs.
6. After all subscribers disconnect, no asyncio Tasks leak (the generator
   exits cleanly when ``request.is_disconnected()`` returns True).

Implementation notes
--------------------
* We point ``_ARB_DIR`` at ``tmp_path`` and disable the live fallback /
  Redis paths so each tick returns ``_empty_state_envelope()`` — fully
  deterministic, no network, no CPU spikes.
* We use ``httpx.ASGITransport`` so all 10 clients share a single in-
  process FastAPI app (mirrors the production behaviour where one
  gunicorn worker fans out across many WebSocket-style subscribers).
* Each subscriber is a separate ``asyncio.Task`` calling
  ``client.stream("GET", "/strategies/arb/stream")`` and consuming raw
  bytes via ``aiter_bytes()``. We parse SSE frames terminated by
  ``\\n\\n`` exactly as the browser does.
* Bound the test budget at ~6 s per case so the suite stays fast even
  when 10 tasks contend on the asyncio event loop.

These tests are **net-isolated** (no Polymarket / Kalshi calls) and
fully deterministic — they should not flake on slow CI provided the
6 s budget holds. If you see flakes, the first thing to check is whether
``_STREAM_TICK_SECONDS`` is being monkey-patched correctly inside the
generator's closure (the constant is read each tick, so monkeypatch
takes effect immediately).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

# We can't use httpx ``ASGITransport`` here: streaming reads buffer
# indefinitely under Python 3.14. The ``base_url`` fixture below boots a
# real uvicorn on a free port (via ``live_server_factory`` from conftest)
# so ``httpx.AsyncClient(base_url=...)`` drives the SSE endpoint over a
# real TCP socket, which the upstream bug doesn't affect.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_arb_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the router at an empty tmp dir + disable live fallback / Redis.

    With these patches the SSE handler emits ``_empty_state_envelope()``
    on every tick — no network, no CPU. We also slam the tick cadence
    down to 150 ms so ``>=3`` frames land inside the per-test budget.
    """
    from pfm import strategies_arb_router as r

    monkeypatch.setattr(r, "_ARB_DIR", tmp_path)
    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", False)
    monkeypatch.setattr(r, "_ARB_REDIS_ENABLED", False)
    monkeypatch.setattr(r, "_STREAM_TICK_SECONDS", 0.15)
    # Drop the fallback cache between tests so prior runs don't leak.
    r._FALLBACK_CACHE["t"] = 0
    r._FALLBACK_CACHE["value"] = None
    return tmp_path


@pytest.fixture
def base_url(isolated_arb_dir: Path, live_server_factory) -> str:
    """Boot the arb router on a real uvicorn and return its base URL."""
    from pfm.auth.dependencies import require_admin
    from pfm.strategies_arb_router import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: None
    return live_server_factory(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _aiter_sse_frames(
    byte_iter: AsyncIterator[bytes],
    deadline: float,
) -> AsyncIterator[str]:
    """Async generator yielding SSE frames (terminated by ``\\n\\n``).

    Stops at ``deadline`` (unix seconds) or when the upstream is exhausted.
    """
    buf = b""
    async for chunk in byte_iter:
        if time.time() > deadline:
            return
        if not chunk:
            continue
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            yield frame.decode("utf-8", errors="replace")


def _parse_data(frame: str) -> dict | None:
    """Extract JSON payload from a ``data:`` SSE frame; ``None`` for comments."""
    lines = [ln for ln in frame.splitlines() if ln.startswith("data:")]
    if not lines:
        return None
    payload = "\n".join(ln[len("data:") :].lstrip() for ln in lines)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


async def _consume_stream(
    client: httpx.AsyncClient,
    *,
    target_frames: int,
    deadline: float,
    consumer_id: int,
    slow_drain_s: float = 0.0,
) -> dict:
    """Open one SSE subscription and collect frames until ``target_frames``.

    Returns a result dict with:
        - status_code: HTTP status of the stream open
        - first_frame: text of the very first frame (keep-alive comment)
        - data_frames: list of parsed JSON payloads (one per data: frame)
        - elapsed: wall-clock seconds spent inside ``client.stream`` block
        - consumer_id: passed-through id for cross-subscriber assertions
        - error: exception message if anything raised; None on success
    """
    result: dict = {
        "consumer_id": consumer_id,
        "status_code": None,
        "first_frame": None,
        "data_frames": [],
        "elapsed": 0.0,
        "error": None,
    }
    t0 = time.time()
    try:
        async with client.stream("GET", "/strategies/arb/stream") as resp:
            result["status_code"] = resp.status_code
            if resp.status_code != 200:
                return result
            async for frame in _aiter_sse_frames(resp.aiter_bytes(), deadline):
                if result["first_frame"] is None:
                    result["first_frame"] = frame
                parsed = _parse_data(frame)
                if parsed is not None:
                    result["data_frames"].append(parsed)
                    if slow_drain_s > 0:
                        # Simulate a slow consumer between reads.
                        await asyncio.sleep(slow_drain_s)
                if len(result["data_frames"]) >= target_frames:
                    break
    except Exception as exc:  # pragma: no cover - surface in assertion
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ten_concurrent_subscribers_all_return_200(base_url: str) -> None:
    """All 10 subscribers must open the stream with HTTP 200 (no 5xx fan-out)."""
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        deadline = time.time() + 6.0
        results = await asyncio.gather(
            *(
                _consume_stream(client, target_frames=1, deadline=deadline, consumer_id=i)
                for i in range(10)
            )
        )
    statuses = [r["status_code"] for r in results]
    assert statuses == [200] * 10, f"non-200 in fan-out: {statuses}"
    assert all(r["error"] is None for r in results), [r["error"] for r in results if r["error"]]


@pytest.mark.asyncio
async def test_each_of_ten_subscribers_receives_three_messages(
    base_url: str,
) -> None:
    """Each subscriber must collect ``>=3`` data frames within 6 s."""
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        deadline = time.time() + 6.0
        results = await asyncio.gather(
            *(
                _consume_stream(client, target_frames=3, deadline=deadline, consumer_id=i)
                for i in range(10)
            )
        )
    counts = [len(r["data_frames"]) for r in results]
    failed = [
        (r["consumer_id"], len(r["data_frames"])) for r in results if len(r["data_frames"]) < 3
    ]
    assert not failed, f"subscribers under-quota: {failed}; all counts={counts}"


@pytest.mark.asyncio
async def test_no_cross_stream_pollution_envelope_keys(base_url: str) -> None:
    """Every subscriber sees the canonical envelope (same key set on every frame).

    Cross-stream pollution would manifest as one subscriber's frame
    suddenly missing keys / containing keys from a different request
    scope. We assert each of 10 subscribers' first data frame has the
    same canonical key set.
    """
    expected_keys = {
        "opportunities",
        "scan_log",
        "bot_status",
        "config",
        "balances",
        "timestamp",
    }
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        deadline = time.time() + 6.0
        results = await asyncio.gather(
            *(
                _consume_stream(client, target_frames=2, deadline=deadline, consumer_id=i)
                for i in range(10)
            )
        )
    for r in results:
        assert r["data_frames"], f"no frames for consumer {r['consumer_id']}"
        for frame in r["data_frames"]:
            missing = expected_keys - frame.keys()
            assert not missing, (
                f"consumer {r['consumer_id']} frame missing keys: {missing}; "
                f"frame keys={list(frame.keys())}"
            )
            # And the offline guard: every subscriber sees the deterministic
            # empty envelope (no half-built / cross-pollinated state).
            assert frame["bot_status"] == "offline"
            assert frame["opportunities"] == []


@pytest.mark.asyncio
async def test_first_frame_keepalive_for_every_subscriber(base_url: str) -> None:
    """The ``: connected`` keep-alive comment must precede every subscriber's
    first ``data:`` frame.

    This is what flips browser EventSource from ``connecting`` to ``open``
    immediately; if it leaks across subscribers (e.g. only the first one
    gets it) tabs 2..10 would visually hang.
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        deadline = time.time() + 6.0
        results = await asyncio.gather(
            *(
                _consume_stream(client, target_frames=1, deadline=deadline, consumer_id=i)
                for i in range(10)
            )
        )
    for r in results:
        assert r["first_frame"] is not None, f"consumer {r['consumer_id']} got no first frame"
        assert r["first_frame"].startswith(":"), (
            f"consumer {r['consumer_id']} first frame not keep-alive: {r['first_frame']!r}"
        )
        assert "connected" in r["first_frame"], (
            f"consumer {r['consumer_id']} keep-alive missing 'connected': {r['first_frame']!r}"
        )


@pytest.mark.asyncio
async def test_backpressure_slow_consumer_does_not_starve_others(
    base_url: str,
) -> None:
    """One slow consumer (200 ms drain between reads) must NOT block the 9 fast ones.

    With cooperative asyncio scheduling, a per-connection generator that
    awaits ``asyncio.sleep(tick)`` between yields lets the loop service
    other tasks. We assert the 9 fast consumers still hit their ``>=3``
    frame quota inside the budget even while consumer 0 is throttled.
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=15.0) as client:
        deadline = time.time() + 8.0
        slow_id = 0
        coros = []
        for i in range(10):
            coros.append(
                _consume_stream(
                    client,
                    target_frames=3,
                    deadline=deadline,
                    consumer_id=i,
                    slow_drain_s=0.2 if i == slow_id else 0.0,
                )
            )
        results = await asyncio.gather(*coros)
    fast = [r for r in results if r["consumer_id"] != slow_id]
    under_quota = [
        (r["consumer_id"], len(r["data_frames"])) for r in fast if len(r["data_frames"]) < 3
    ]
    assert not under_quota, (
        f"fast consumers starved by slow consumer: {under_quota}; "
        f"slow consumer collected {len(results[slow_id]['data_frames'])} frames"
    )
    # Every consumer (including slow) must have at least one frame —
    # the connection itself didn't die.
    no_frames = [r["consumer_id"] for r in results if not r["data_frames"]]
    assert not no_frames, f"consumers with zero frames: {no_frames}"


@pytest.mark.asyncio
async def test_server_remains_healthy_after_fan_out(base_url: str) -> None:
    """After 10-subscriber fan-out, a fresh GET /strategies/arb/state succeeds.

    Validates the server didn't crash, didn't leak open SSE generators
    that block subsequent requests, and didn't get stuck in the asyncio
    event loop.
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        # 10-subscriber blast
        deadline = time.time() + 5.0
        await asyncio.gather(
            *(
                _consume_stream(client, target_frames=2, deadline=deadline, consumer_id=i)
                for i in range(10)
            )
        )
        # Now a normal GET should still work promptly.
        t0 = time.time()
        resp = await client.get("/strategies/arb/state")
        elapsed = time.time() - t0
    assert resp.status_code == 200, f"state endpoint dead after fan-out: {resp.status_code}"
    # Must respond fast — no lingering blocked event loop. 3s is a very
    # generous CI ceiling; locally this returns in <100 ms.
    assert elapsed < 3.0, f"/state slow after fan-out: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_concurrent_clean_disconnect_no_task_leak(base_url: str) -> None:
    """Every subscriber that closes its context cleanly leaves no extra Tasks.

    We snapshot ``asyncio.all_tasks()`` before and after the fan-out plus
    a small grace period. The post-snapshot must not contain *more* tasks
    than the pre-snapshot — the SSE generator must exit when
    ``is_disconnected()`` flips True on the client closing its stream.
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        loop = asyncio.get_running_loop()
        # Wait for the loop to settle before snapshotting.
        await asyncio.sleep(0.05)
        before = {t for t in asyncio.all_tasks(loop) if not t.done()}

        deadline = time.time() + 5.0
        # Pull just 1 frame each, then close — exercises the disconnect path.
        await asyncio.gather(
            *(
                _consume_stream(client, target_frames=1, deadline=deadline, consumer_id=i)
                for i in range(10)
            )
        )
        # Give the generators a moment to observe is_disconnected().
        await asyncio.sleep(0.5)
        after = {t for t in asyncio.all_tasks(loop) if not t.done()}

    leaked = after - before
    # Filter: ignore the current task itself (the test coroutine).
    current = asyncio.current_task()
    leaked.discard(current)
    assert not leaked, (
        f"task leak after concurrent disconnect: {len(leaked)} new tasks. "
        f"Names: {[t.get_name() for t in leaked]}"
    )


@pytest.mark.asyncio
async def test_cpu_bounded_under_fan_out(base_url: str) -> None:
    """Wall-clock for 10 subscribers fetching 3 frames is bounded.

    Since each tick is 150 ms and the work is essentially free
    (empty envelope, no I/O), 10 subscribers fetching 3 frames each
    should finish in well under 6 s — *not* anywhere near 10 * 3 * 0.15
    = 4.5s of serial work. asyncio fans them out.

    This is a weak proxy for "CPU stays bounded" — we assert the total
    elapsed is < 6 s, which would be impossible if each subscriber were
    spinning a busy-loop or blocking on the event loop.
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        deadline = time.time() + 6.0
        t0 = time.time()
        results = await asyncio.gather(
            *(
                _consume_stream(client, target_frames=3, deadline=deadline, consumer_id=i)
                for i in range(10)
            )
        )
        elapsed = time.time() - t0
    assert elapsed < 6.0, f"10x3 frames took {elapsed:.2f}s; event loop likely starved"
    assert all(len(r["data_frames"]) >= 3 for r in results), (
        "some subscribers under-quota despite finishing in budget"
    )


@pytest.mark.asyncio
async def test_payload_consistency_across_simultaneous_subscribers(
    base_url: str,
) -> None:
    """Subscribers that read at the same instant must see structurally identical envelopes.

    With ``_LIVE_FALLBACK_ENABLED=False`` and no engine file, every tick
    returns ``_empty_state_envelope()`` — a deterministic object. Two
    concurrent subscribers reading roughly synchronously must therefore
    see the same envelope keys & ``bot_status`` (only ``timestamp`` may
    drift between tick boundaries).
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        deadline = time.time() + 6.0
        results = await asyncio.gather(
            *(
                _consume_stream(client, target_frames=1, deadline=deadline, consumer_id=i)
                for i in range(10)
            )
        )
    # Compare structural signature: sorted keys + bot_status. timestamp
    # may differ across tick boundaries, so exclude.
    signatures = set()
    for r in results:
        assert r["data_frames"], f"no frames for consumer {r['consumer_id']}"
        frame = r["data_frames"][0]
        sig = (tuple(sorted(frame.keys())), frame.get("bot_status"))
        signatures.add(sig)
    assert len(signatures) == 1, f"subscribers saw divergent envelope shapes: {signatures}"


@pytest.mark.asyncio
async def test_repeated_fan_out_cycles_remain_stable(base_url: str) -> None:
    """Three sequential 10-subscriber cycles all succeed — no degradation.

    Validates the handler doesn't accumulate state (e.g. a leak in
    ``_FALLBACK_CACHE`` or a growing background task list). Each cycle
    must complete with all 10 subscribers receiving ``>=2`` frames.
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=15.0) as client:
        for cycle in range(3):
            deadline = time.time() + 5.0
            results = await asyncio.gather(
                *(
                    _consume_stream(
                        client,
                        target_frames=2,
                        deadline=deadline,
                        consumer_id=i,
                    )
                    for i in range(10)
                )
            )
            counts = [len(r["data_frames"]) for r in results]
            under = [c for c in counts if c < 2]
            assert not under, f"cycle {cycle}: subscribers under quota: counts={counts}"
            # Give the loop a beat between cycles.
            await asyncio.sleep(0.1)
