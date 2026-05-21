"""Tests for the terminal live-stream SSE endpoint.

We test the async generator (`_stream_ticks`) directly with a fake
``Request`` instead of going through a streaming TestClient — TestClient
can't cleanly cancel an SSE generator, so a direct async test is faster,
more deterministic, and exercises the same code paths.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_live_stream import (
    GAMMA_URL,
    MAX_SLUGS,
    _parse_slugs,
    _stream_ticks,
    router,
)


def _gamma_response_for(slug: str, yes_token: str) -> httpx.Response:
    """Minimal Gamma payload — just enough for ``get_market_metadata``."""
    return httpx.Response(
        200,
        json=[
            {
                "slug": slug,
                "question": f"Will {slug}?",
                "clobTokenIds": json.dumps([yes_token, f"{yes_token}-no"]),
                "active": True,
                "closed": False,
            }
        ],
    )


class _FakeRequest:
    """Minimal stand-in for ``starlette.requests.Request`` for the generator.

    ``_stream_ticks`` only ever calls ``await request.is_disconnected()``;
    we expose a flag so tests can flip the connection state mid-stream.
    """

    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _parse_sse_bytes(frames: list[bytes]) -> list[tuple[str, dict[str, Any]]]:
    """Decode a list of raw SSE frames into ``(event_name, data_dict)`` tuples."""
    out: list[tuple[str, dict[str, Any]]] = []
    for frame in frames:
        text = frame.decode("utf-8").strip()
        if not text:
            continue
        ev, data = None, None
        for line in text.splitlines():
            if line.startswith("event: "):
                ev = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if ev is not None and data is not None:
            out.append((ev, data))
    return out


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


# ---- Test 1: cap to 30 slugs ----------------------------------------------


def test_parse_slugs_caps_at_30() -> None:
    """``_parse_slugs`` enforces ``MAX_SLUGS`` and dedupes while preserving order."""
    raw = ",".join(f"slug-{i:03d}" for i in range(50))
    parsed = _parse_slugs(raw)
    assert len(parsed) == MAX_SLUGS == 30
    assert parsed[0] == "slug-000"
    assert parsed[-1] == "slug-029"

    # Dedupe + cap together.
    raw_dup = ",".join(["a", "b", "a", "c", "  ", "b", "d"])
    assert _parse_slugs(raw_dup) == ["a", "b", "c", "d"]

    # Empty / whitespace only.
    assert _parse_slugs("") == []
    assert _parse_slugs(", , ,") == []


@respx.mock
def test_stream_only_resolves_first_30_slugs() -> None:
    """50 slugs in → only ``MAX_SLUGS`` Gamma lookups happen and ``ready``
    advertises exactly 30 slugs."""
    gamma_route = respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=lambda req: _gamma_response_for(
            req.url.params["slug"], f"tok-{req.url.params['slug']}"
        )
    )
    respx.get("https://clob.polymarket.com/midpoint").mock(
        return_value=httpx.Response(200, json={"mid": "0.5"})
    )
    respx.get("https://clob.polymarket.com/price").mock(
        return_value=httpx.Response(200, json={"price": "0.5"})
    )

    async def _drive() -> list[bytes]:
        slugs = _parse_slugs(",".join(f"slug-{i:03d}" for i in range(50)))
        assert len(slugs) == MAX_SLUGS
        request = _FakeRequest()
        # Stop the loop right after the ready frame and the first cycle.
        # AsyncClient for the streaming hot path; sync Client only for the
        # one-shot slug→token resolution in PolymarketClient.
        async with httpx.AsyncClient() as http:
            with (
                httpx.Client() as sync_http,
                PolymarketClient(
                    GAMMA_URL,
                    "https://clob.polymarket.com",
                    client=sync_http,
                ) as poly,
            ):
                gen = _stream_ticks(
                    request,  # type: ignore[arg-type]
                    slugs,
                    hz=5.0,
                    deadline_seconds=0.05,  # tiny — generator self-terminates fast
                    http_client=http,
                    poly_client=poly,
                )
                collected: list[bytes] = []
                async for frame in gen:
                    collected.append(frame)
                return collected

    frames = asyncio.run(_drive())
    events = _parse_sse_bytes(frames)
    ready = [d for ev, d in events if ev == "ready"]
    assert ready, "expected a ready frame before any ticks"
    assert len(ready[0]["slugs"]) == MAX_SLUGS
    assert gamma_route.call_count == MAX_SLUGS


# ---- Test 2: heartbeat every interval -------------------------------------


@respx.mock
def test_emits_one_tick_per_slug_per_cycle() -> None:
    """At ``hz=10``, two cycles produce two ticks per slug with proper fields."""
    for slug, tok in [("alpha", "1"), ("bravo", "2")]:
        respx.get(f"{GAMMA_URL}/markets", params={"slug": slug}).mock(
            return_value=_gamma_response_for(slug, tok)
        )
    respx.get("https://clob.polymarket.com/midpoint").mock(
        return_value=httpx.Response(200, json={"mid": "0.42"})
    )
    # CLOB convention (production-verified): side=BUY → best bid (what a
    # seller receives); side=SELL → best ask (what a buyer pays). Keep the
    # mock consistent so bid < ask the way a real book is quoted.
    respx.get(
        "https://clob.polymarket.com/price",
        params={"side": "BUY"},
    ).mock(return_value=httpx.Response(200, json={"price": "0.41"}))
    respx.get(
        "https://clob.polymarket.com/price",
        params={"side": "SELL"},
    ).mock(return_value=httpx.Response(200, json={"price": "0.43"}))

    async def _drive() -> list[bytes]:
        request = _FakeRequest()
        async with httpx.AsyncClient() as http:
            with (
                httpx.Client() as sync_http,
                PolymarketClient(
                    GAMMA_URL,
                    "https://clob.polymarket.com",
                    client=sync_http,
                ) as poly,
            ):
                gen = _stream_ticks(
                    request,  # type: ignore[arg-type]
                    ["alpha", "bravo"],
                    hz=10.0,  # interval = 0.1s
                    deadline_seconds=0.25,  # ≥ 2 cycles, < 3
                    http_client=http,
                    poly_client=poly,
                )
                return [frame async for frame in gen]

    frames = asyncio.run(_drive())
    events = _parse_sse_bytes(frames)
    ticks = [d for ev, d in events if ev == "tick"]
    # Expect at least 4 ticks (2 slugs × 2 cycles); allow more if scheduler
    # squeezed a third cycle in within the 0.25s deadline.
    assert len(ticks) >= 4
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for t in ticks:
        by_slug.setdefault(t["slug"], []).append(t)
    assert set(by_slug) == {"alpha", "bravo"}
    for slug_ticks in by_slug.values():
        assert len(slug_ticks) >= 2

    # Field shape and values.
    sample = ticks[0]
    assert sample["mid"] == pytest.approx(0.42)
    assert sample["bid"] == pytest.approx(0.41)
    assert sample["ask"] == pytest.approx(0.43)
    assert isinstance(sample["ts"], int)
    assert set(sample) == {"slug", "mid", "bid", "ask", "ts"}

    # The deadline should produce a final ``bye`` frame.
    byes = [d for ev, d in events if ev == "bye"]
    assert byes and byes[0]["reason"] == "deadline"


# ---- Test 3: graceful disconnect ------------------------------------------


@respx.mock
def test_graceful_client_disconnect_stops_stream() -> None:
    """Flipping ``request.is_disconnected = True`` cleanly terminates the
    generator without raising and without emitting further ticks."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "alpha"}).mock(
        return_value=_gamma_response_for("alpha", "1")
    )
    midpoint_route = respx.get("https://clob.polymarket.com/midpoint").mock(
        return_value=httpx.Response(200, json={"mid": "0.5"})
    )
    respx.get("https://clob.polymarket.com/price").mock(
        return_value=httpx.Response(200, json={"price": "0.5"})
    )

    async def _drive() -> tuple[list[bytes], int]:
        request = _FakeRequest()
        async with httpx.AsyncClient() as http:
            with (
                httpx.Client() as sync_http,
                PolymarketClient(
                    GAMMA_URL,
                    "https://clob.polymarket.com",
                    client=sync_http,
                ) as poly,
            ):
                gen = _stream_ticks(
                    request,  # type: ignore[arg-type]
                    ["alpha"],
                    hz=10.0,
                    deadline_seconds=10.0,  # would run forever — but we'll cut it
                    http_client=http,
                    poly_client=poly,
                )
                collected: list[bytes] = []
                cycles_seen = 0
                async for frame in gen:
                    collected.append(frame)
                    if b"event: tick" in frame:
                        cycles_seen += 1
                        if cycles_seen == 1:
                            # Pretend the client closed the TCP socket. The next
                            # loop iteration must observe this and return cleanly.
                            request.disconnected = True
                return collected, cycles_seen

    frames, cycles = asyncio.run(_drive())
    events = _parse_sse_bytes(frames)
    # We saw the ready + at least one tick, then exited.
    assert any(ev == "ready" for ev, _ in events)
    assert any(ev == "tick" for ev, _ in events)
    # No ``bye`` (deadline) frame, since we disconnected — graceful exit.
    assert not any(ev == "bye" for ev, _ in events)
    # We saw exactly one tick cycle then bailed.
    assert cycles == 1
    # And the midpoint endpoint was hit at least once.
    assert midpoint_route.call_count >= 1


# ---- Smoke tests for endpoint plumbing ------------------------------------


def test_endpoint_validates_hz_bounds(client: TestClient) -> None:
    """``hz`` must lie in [MIN_HZ, MAX_HZ]; out-of-range returns 422."""
    r1 = client.get("/terminal/live-stream", params={"slugs": "x", "hz": 0.0})
    assert r1.status_code == 422
    r2 = client.get("/terminal/live-stream", params={"slugs": "x", "hz": 99.0})
    assert r2.status_code == 422


def test_endpoint_requires_slugs_param(client: TestClient) -> None:
    """Missing ``slugs`` query parameter → 422."""
    r = client.get("/terminal/live-stream")
    assert r.status_code == 422
