"""Tests for the SSE variant of /reverse-finder.

The streaming endpoint emits ``event: <name>\\ndata: <json>\\n\\n`` frames,
one per forward-selection step, so the frontend can render bars as
picks land. These tests parse the raw response body into a list of
``(event, data)`` pairs and assert ordering / monotonicity invariants.

We reuse the project-wide ``router_app_client`` fixture (defined in
``tests/test_reverse_finder.py``) which stubs out yfinance + Polymarket
via fake fetchers — no network, no respx mocks needed.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import pytest
from fastapi.testclient import TestClient


# Local fixture mirroring ``tests.test_reverse_finder.router_app_client``:
# mounts the reverse-finder router on ``main.app`` with all external IO
# stubbed (yfinance + Polymarket fake fetchers, NullCache for Redis).
# We duplicate rather than import because pytest does not propagate
# fixtures across sibling test modules without a conftest.
@pytest.fixture
def stream_router_client(monkeypatch, factors_file, fake_factor_history, fake_log_returns):
    monkeypatch.setenv("FACTORS_FILE", str(factors_file))
    import pfm.config as cfg

    cfg._settings = None

    import pfm.main as main_mod

    monkeypatch.setattr(main_mod, "fetch_factor_history", fake_factor_history)
    monkeypatch.setattr(main_mod, "get_log_returns", fake_log_returns)

    from pfm.cache import NullCache

    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    from pfm.reverse_finder_router import router as rff_router

    if not getattr(main_mod.app.state, "_rff_mounted", False):
        main_mod.app.include_router(rff_router)
        main_mod.app.state._rff_mounted = True  # type: ignore[attr-defined]

    with TestClient(main_mod.app) as client:
        yield client


# --- helpers ---------------------------------------------------------------


def _parse_sse(raw: bytes) -> list[tuple[str, dict[str, Any]]]:
    """Split an SSE body into ``[(event, data_dict), ...]`` pairs.

    Each frame is ``event: <name>\\ndata: <json>\\n\\n``. Frames are
    separated by a blank line. We're forgiving about trailing
    whitespace / multiple newlines.
    """
    text = raw.decode("utf-8")
    frames: list[tuple[str, dict[str, Any]]] = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        event: str | None = None
        data: str | None = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        if event is None or data is None:
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            payload = {"_raw": data}
        frames.append((event, payload))
    return frames


# --- tests ----------------------------------------------------------------


def test_stream_emits_meta_factor_done_in_order(stream_router_client: TestClient) -> None:
    """The endpoint must emit ``meta`` first, ≥1 ``factor``, then ``done`` last."""
    r = stream_router_client.post(
        "/reverse-finder/stream",
        json={
            "ticker": "TEST",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "candidate_factor_ids": ["factor_a", "factor_b"],
            "k": 2,
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers.get("cache-control") == "no-cache"
    assert r.headers.get("x-accel-buffering") == "no"

    frames = _parse_sse(r.content)
    assert len(frames) >= 3, frames
    # First frame must be meta.
    assert frames[0][0] == "meta"
    assert frames[0][1]["ticker"] == "TEST"
    assert "pool_used" in frames[0][1]
    # Last frame must be done.
    assert frames[-1][0] == "done"

    # At least one factor event.
    factor_frames = [f for f in frames if f[0] == "factor"]
    assert len(factor_frames) >= 1


def test_stream_cumulative_r2_monotonic(stream_router_client: TestClient) -> None:
    """`cumulative_r2` must be monotonically non-decreasing across ``factor`` events."""
    r = stream_router_client.post(
        "/reverse-finder/stream",
        json={
            "ticker": "TEST",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "candidate_factor_ids": ["factor_a", "factor_b"],
            "k": 2,
        },
    )
    assert r.status_code == 200, r.text
    frames = _parse_sse(r.content)

    cumulative = [f[1]["cumulative_r2"] for f in frames if f[0] == "factor"]
    assert cumulative, "expected at least one factor event"
    for prev, nxt in pairwise(cumulative):
        assert nxt + 1e-9 >= prev, f"cumulative_r2 decreased: {prev} -> {nxt}"

    # rank should also be 1, 2, ... in order.
    ranks = [f[1]["rank"] for f in frames if f[0] == "factor"]
    assert ranks == list(range(1, len(ranks) + 1))


def test_stream_curated_pool_size_bounded(stream_router_client: TestClient) -> None:
    """`pool="curated"` must keep the candidate pool at ≤ 200 factors."""
    r = stream_router_client.post(
        "/reverse-finder/stream",
        json={
            "ticker": "TEST",
            "start": "2025-06-15",
            "end": "2025-12-15",
            # No candidate_factor_ids → curated default
            "k": 2,
        },
    )
    assert r.status_code == 200, r.text
    frames = _parse_sse(r.content)
    meta = next(f for f in frames if f[0] == "meta")
    assert meta[1].get("pool_used", "").startswith("curated_")
    assert meta[1]["n_candidates"] <= 200


def test_stream_pool_all_considers_full_catalogue(stream_router_client: TestClient) -> None:
    """`pool=all` must consider every factor in the (synthetic) catalogue."""
    r = stream_router_client.post(
        "/reverse-finder/stream?pool=all",
        json={
            "ticker": "TEST",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "k": 2,
        },
    )
    assert r.status_code == 200, r.text
    frames = _parse_sse(r.content)
    meta = next(f for f in frames if f[0] == "meta")
    assert meta[1].get("pool_used", "").startswith("all_")
    # Conftest catalogue has exactly 2 factors.
    assert meta[1]["n_candidates"] == 2


def test_stream_done_has_total_r_squared(stream_router_client: TestClient) -> None:
    """The terminal ``done`` event must report ``total_r_squared`` and ``rejected``."""
    r = stream_router_client.post(
        "/reverse-finder/stream",
        json={
            "ticker": "TEST",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "candidate_factor_ids": ["factor_a", "factor_b"],
            "k": 2,
        },
    )
    assert r.status_code == 200, r.text
    frames = _parse_sse(r.content)
    done = frames[-1]
    assert done[0] == "done"
    assert "total_r_squared" in done[1]
    assert "rejected" in done[1]
    # total_r_squared should equal the largest cumulative_r2 we saw.
    factor_cum = [f[1]["cumulative_r2"] for f in frames if f[0] == "factor"]
    if factor_cum:
        assert done[1]["total_r_squared"] == pytest.approx(max(factor_cum), abs=1e-9)


def test_stream_invalid_dates_400(stream_router_client: TestClient) -> None:
    """Invalid date ordering must still 400 before the stream is opened."""
    r = stream_router_client.post(
        "/reverse-finder/stream",
        json={
            "ticker": "TEST",
            "start": "2025-12-15",
            "end": "2025-06-15",
            "candidate_factor_ids": ["factor_a"],
            "k": 1,
        },
    )
    assert r.status_code == 400


def test_stream_unknown_candidates_emit_error(stream_router_client: TestClient) -> None:
    """When all candidate ids are unknown the stream emits an ``error`` frame.

    The non-stream endpoint raises 422; the streaming variant catches
    that and converts it to an in-band SSE error so the client can
    render gracefully without parsing a non-2xx status.
    """
    r = stream_router_client.post(
        "/reverse-finder/stream",
        json={
            "ticker": "TEST",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "candidate_factor_ids": ["this_does_not_exist"],
            "k": 1,
        },
    )
    # The HTTP envelope is still 200 (the stream opened successfully);
    # the error surfaces as an in-band event.
    assert r.status_code == 200
    frames = _parse_sse(r.content)
    kinds = [f[0] for f in frames]
    assert "error" in kinds, frames
