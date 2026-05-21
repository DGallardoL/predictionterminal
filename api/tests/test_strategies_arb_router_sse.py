"""Tests for the SSE stream at ``GET /strategies/arb/stream``.

The router exposes a Server-Sent-Events endpoint that pushes a (trimmed)
copy of the ``/state`` envelope every ``PFM_ARB_STREAM_TICK_S`` seconds
(default 5.0 in prod). Each tick is wire-encoded as::

    data: {...json...}\n\n

The very first frame is a keep-alive comment (``: connected\n\n``) so the
browser's ``EventSource`` flips to ``onopen`` immediately even when the
first state build is slow (cold-cache scanner sweep).

Strategy:
- Point ``_ARB_DIR`` at ``tmp_path`` so no real ``arbstuff/`` files are read.
- Disable ``_LIVE_FALLBACK_ENABLED`` so the empty envelope is returned
  deterministically (no network → no flakes).
- Force ``_STREAM_TICK_SECONDS`` to a tiny value so 3+ ticks land inside
  the test budget (~8 s wall-clock).
- Use FastAPI's ``TestClient.stream("GET", url)`` context manager and
  read raw bytes off the SSE stream, parsing the ``data:`` lines.

httpx-sse is *not* a hard dep — these tests use the lower-level
``response.iter_bytes()`` API which is always available with the
FastAPI/httpx test client.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
#
# We can't use ``TestClient.stream`` here: httpx ``ASGITransport`` buffers
# streaming reads indefinitely under Python 3.14 (see
# tests/test_sse_concurrent_load.py write-up). Instead we boot uvicorn on
# a free port in a daemon thread via the ``live_server_factory`` fixture
# (defined in conftest.py) and drive the SSE endpoint with a real
# ``httpx.Client`` over TCP — which the bug doesn't affect.


@pytest.fixture
def arb_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the router from any real ``arbstuff/`` directory on disk.

    Also disables the live-fallback scanner so the SSE stream returns a
    deterministic empty envelope (no network calls, no flakes).
    """
    from pfm import strategies_arb_router as r

    monkeypatch.setattr(r, "_ARB_DIR", tmp_path)
    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", False)
    monkeypatch.setattr(r, "_ARB_REDIS_ENABLED", False)
    # Force fast cadence: 3 ticks in ~0.6 s instead of 15 s (prod default).
    monkeypatch.setattr(r, "_STREAM_TICK_SECONDS", 0.2)
    # Drop the fallback cache between tests so prior runs don't leak.
    r._FALLBACK_CACHE["t"] = 0
    r._FALLBACK_CACHE["value"] = None
    return tmp_path


@pytest.fixture
def client(arb_dir: Path, live_server_factory) -> httpx.Client:
    """A real httpx.Client pointed at a uvicorn subprocess hosting the router."""
    from pfm.auth.dependencies import require_admin
    from pfm.strategies_arb_router import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: None
    base_url = live_server_factory(app)
    with httpx.Client(base_url=base_url, timeout=15.0) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_sse_frames(byte_iter: Iterator[bytes], deadline: float) -> Iterator[str]:
    """Yield SSE frames (terminated by ``\\n\\n``) from a raw byte stream.

    Stops when the wall-clock ``deadline`` (unix seconds) is reached or
    when the upstream iterator is exhausted.
    """
    buf = b""
    for chunk in byte_iter:
        if time.time() > deadline:
            return
        if not chunk:
            continue
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            yield frame.decode("utf-8", errors="replace")


def _parse_data_frame(frame: str) -> dict | None:
    """Extract the JSON payload from a ``data: ...`` SSE frame, or None."""
    lines = [ln for ln in frame.splitlines() if ln.startswith("data:")]
    if not lines:
        return None
    payload = "\n".join(ln[len("data:") :].lstrip() for ln in lines)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stream_returns_200_with_event_stream_content_type(client: httpx.Client) -> None:
    """Headers must signal SSE + disable buffering."""
    with client.stream("GET", "/strategies/arb/stream") as resp:
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/event-stream" in ct, f"unexpected content-type: {ct!r}"
        # Streaming hygiene: caches and proxy buffering must be disabled
        # so the browser sees frames in real time.
        assert resp.headers.get("cache-control", "").lower() == "no-cache"
        # Read one chunk so the generator runs at least once, then close.
        for _ in resp.iter_bytes():
            break


def test_stream_first_frame_is_connected_keepalive(client: httpx.Client) -> None:
    """The handler yields ``: connected\\n\\n`` as the very first frame."""
    deadline = time.time() + 8.0
    with client.stream("GET", "/strategies/arb/stream") as resp:
        assert resp.status_code == 200
        frames = _iter_sse_frames(resp.iter_bytes(), deadline)
        first = next(frames, None)
        assert first is not None, "no frame received in 8 s"
        # The keep-alive comment line. SSE comments start with ':'.
        assert first.startswith(":"), f"expected ':' comment frame, got {first!r}"
        assert "connected" in first


def test_stream_emits_at_least_three_data_events(client: httpx.Client) -> None:
    """Cadence: with tick=0.2s, we should see >=3 ``data:`` frames in 8 s."""
    deadline = time.time() + 8.0
    data_frames: list[dict] = []
    with client.stream("GET", "/strategies/arb/stream") as resp:
        assert resp.status_code == 200
        for frame in _iter_sse_frames(resp.iter_bytes(), deadline):
            parsed = _parse_data_frame(frame)
            if parsed is not None:
                data_frames.append(parsed)
            if len(data_frames) >= 3:
                break
    assert len(data_frames) >= 3, (
        f"expected >=3 data frames within 8 s, got {len(data_frames)}: {data_frames!r}"
    )


def test_stream_data_frame_has_valid_sse_format(client: httpx.Client) -> None:
    """Each non-comment frame is ``data: <json>`` terminated by blank line."""
    deadline = time.time() + 8.0
    saw_data_line = False
    with client.stream("GET", "/strategies/arb/stream") as resp:
        for frame in _iter_sse_frames(resp.iter_bytes(), deadline):
            if frame.startswith(":"):
                # keep-alive comment — skip
                continue
            # Strict SSE: each non-comment line must be ``field: value`` or
            # ``field:value``. The handler only emits ``data:``.
            for line in frame.splitlines():
                assert line.startswith("data:"), (
                    f"unexpected non-data SSE line: {line!r} in frame {frame!r}"
                )
            saw_data_line = True
            break
    assert saw_data_line, "no data frame observed before deadline"


def test_stream_payload_parses_as_json(client: httpx.Client) -> None:
    """The body of a ``data:`` frame must be valid JSON."""
    deadline = time.time() + 8.0
    with client.stream("GET", "/strategies/arb/stream") as resp:
        for frame in _iter_sse_frames(resp.iter_bytes(), deadline):
            if frame.startswith(":"):
                continue
            parsed = _parse_data_frame(frame)
            assert parsed is not None, f"could not parse JSON from {frame!r}"
            assert isinstance(parsed, dict)
            return
    pytest.fail("no data frame observed before deadline")


def test_stream_payload_has_expected_envelope_keys(client: httpx.Client) -> None:
    """Empty / offline envelope still includes the canonical keys.

    When ``_LIVE_FALLBACK_ENABLED`` is False and no engine file exists,
    the handler emits ``_empty_state_envelope()`` which carries the same
    schema the UI expects (``opportunities``, ``scan_log``, ``bot_status``,
    ``config``, ``balances``, ``timestamp``).
    """
    deadline = time.time() + 8.0
    payload: dict | None = None
    with client.stream("GET", "/strategies/arb/stream") as resp:
        for frame in _iter_sse_frames(resp.iter_bytes(), deadline):
            if frame.startswith(":"):
                continue
            payload = _parse_data_frame(frame)
            if payload is not None:
                break
    assert payload is not None, "no data frame observed"
    # Required keys for the live monitor UI.
    for key in (
        "opportunities",
        "scan_log",
        "bot_status",
        "config",
        "balances",
        "timestamp",
    ):
        assert key in payload, f"missing key {key!r} in payload: {payload!r}"
    # When fallback is disabled and engine file is missing, status is offline.
    assert payload["bot_status"] == "offline"
    assert payload["opportunities"] == []


def test_stream_surfaces_engine_state_when_dashboard_file_present(
    client: httpx.Client, arb_dir: Path
) -> None:
    """When ``dashboard_state.json`` exists & is fresh, its contents stream."""
    state = {
        "timestamp": "2026-05-16T10:00:00",
        "scan_count": 42,
        "cycle_time_s": 1.2,
        "balances": {"kalshi": 500.0, "polymarket": 750.0},
        "config": {
            "poll_interval": 8,
            "threshold": 0.94,
            "min_alert_profit": 1.0,
            "event_count": 100,
        },
        "bot_status": "running",
        "test_mode": True,
        "scan_mode": "WS",
        "candidates_count": 1,
        "opportunities": [
            {
                "name": "TestEvent",
                "type": "Buy K_YES+P_NO",
                "profit_pct": 1.85,
                "volume": 250.0,
            }
        ],
        "scan_log": [],
    }
    (arb_dir / "dashboard_state.json").write_text(json.dumps(state), encoding="utf-8")

    deadline = time.time() + 8.0
    payload: dict | None = None
    with client.stream("GET", "/strategies/arb/stream") as resp:
        for frame in _iter_sse_frames(resp.iter_bytes(), deadline):
            if frame.startswith(":"):
                continue
            payload = _parse_data_frame(frame)
            if payload is not None:
                break
    assert payload is not None, "no data frame observed"
    assert payload["bot_status"] == "running"
    assert payload["scan_count"] == 42
    assert payload["opportunities"][0]["name"] == "TestEvent"
    assert payload.get("_source") == "engine"


def test_stream_cancels_cleanly_mid_stream(client: httpx.Client) -> None:
    """Closing the stream context mid-flight must not raise / leak resources.

    The handler checks ``await request.is_disconnected()`` between ticks
    and returns from the generator on disconnect.
    """
    with client.stream("GET", "/strategies/arb/stream") as resp:
        assert resp.status_code == 200
        # Pull just one chunk then drop out of the ctx manager. If the
        # generator weren't cancellable cleanly this would hang or raise.
        for chunk in resp.iter_bytes():
            if chunk:
                break
    # Reaching this line means the context manager exited without error,
    # which is the contract — no assertion needed beyond not raising.


def test_stream_payload_truncates_long_scan_log(
    client: httpx.Client, arb_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``scan_log`` is capped at ``_STREAM_SCAN_LOG_MAX`` per tick on the wire.

    Wire frames stay small (< ~50 KB) regardless of how many lines the
    engine has logged. ``/state`` (non-stream) still returns the full log.
    """
    from pfm import strategies_arb_router as r

    monkeypatch.setattr(r, "_STREAM_SCAN_LOG_MAX", 5)
    long_log = [{"ts": i, "msg": f"scan #{i}"} for i in range(50)]
    state = {
        "timestamp": "2026-05-16T10:00:00",
        "bot_status": "running",
        "balances": {"kalshi": 0.0, "polymarket": 0.0},
        "config": {
            "poll_interval": 8,
            "threshold": 0.94,
            "min_alert_profit": 1.0,
            "event_count": 0,
        },
        "opportunities": [],
        "scan_log": long_log,
    }
    (arb_dir / "dashboard_state.json").write_text(json.dumps(state), encoding="utf-8")

    deadline = time.time() + 8.0
    payload: dict | None = None
    with client.stream("GET", "/strategies/arb/stream") as resp:
        for frame in _iter_sse_frames(resp.iter_bytes(), deadline):
            if frame.startswith(":"):
                continue
            payload = _parse_data_frame(frame)
            if payload is not None:
                break
    assert payload is not None
    assert len(payload["scan_log"]) == 5
    # Last-N semantics: should be the most-recent entries.
    assert payload["scan_log"][-1]["msg"] == "scan #49"
    assert payload.get("_scan_log_truncated") == 50
