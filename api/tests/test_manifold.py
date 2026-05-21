"""Tests for the Manifold Markets async client (HTTP mocked via respx)."""

from __future__ import annotations

import asyncio

import httpx
import pandas as pd
import pytest
import respx

from pfm.sources.manifold import (
    MANIFOLD_BASE_URL,
    ManifoldClient,
    ManifoldError,
)

BASE = MANIFOLD_BASE_URL


def _run(coro):
    """Run an async coroutine in a fresh event loop (matches test_realtime style)."""
    return asyncio.run(coro)


@pytest.fixture
def fast_client() -> ManifoldClient:
    """Manifold client with rate-limit pacing disabled (faster tests)."""
    return ManifoldClient(
        client=httpx.AsyncClient(),
        concurrency=5,
        min_interval_s=0.0,
    )


# ---------------------------------------------------------------------------
# search_markets
# ---------------------------------------------------------------------------


@respx.mock
def test_search_markets_returns_normalised_list(fast_client: ManifoldClient) -> None:
    respx.get(f"{BASE}/search-markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "m1", "slug": "a", "question": "Will A?", "probability": 0.42},
                {"id": "m2", "slug": "b", "question": "Will B?", "probability": 0.18},
                {"id": "m3", "slug": "c", "question": "Will C?", "probability": 0.71},
            ],
        )
    )

    async def go() -> list[dict]:
        try:
            return await fast_client.search_markets("recession", limit=2)
        finally:
            await fast_client.close()

    out = _run(go())
    assert len(out) == 2
    assert out[0]["id"] == "m1"
    assert out[1]["slug"] == "b"


@respx.mock
def test_search_markets_empty_query_short_circuits(fast_client: ManifoldClient) -> None:
    # No respx route registered — would 500 if a request actually fired.
    async def go() -> list[dict]:
        try:
            return await fast_client.search_markets("", limit=10)
        finally:
            await fast_client.close()

    assert _run(go()) == []


@respx.mock
def test_search_markets_raises_on_non_list_payload(fast_client: ManifoldClient) -> None:
    respx.get(f"{BASE}/search-markets").mock(
        return_value=httpx.Response(200, json={"error": "boom"})
    )

    async def go() -> None:
        try:
            await fast_client.search_markets("foo")
        finally:
            await fast_client.close()

    with pytest.raises(ManifoldError):
        _run(go())


# ---------------------------------------------------------------------------
# get_market
# ---------------------------------------------------------------------------


@respx.mock
def test_get_market_uses_slug_endpoint(fast_client: ManifoldClient) -> None:
    respx.get(f"{BASE}/slug/will-fed-cut").mock(
        return_value=httpx.Response(
            200,
            json={"id": "abc", "slug": "will-fed-cut", "probability": 0.55},
        )
    )

    async def go() -> dict:
        try:
            return await fast_client.get_market("will-fed-cut")
        finally:
            await fast_client.close()

    m = _run(go())
    assert m["slug"] == "will-fed-cut"
    assert m["probability"] == 0.55


@respx.mock
def test_get_market_falls_back_to_id_endpoint_on_404(fast_client: ManifoldClient) -> None:
    respx.get(f"{BASE}/slug/abc").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    respx.get(f"{BASE}/market/abc").mock(
        return_value=httpx.Response(200, json={"id": "abc", "probability": 0.31})
    )

    async def go() -> dict:
        try:
            return await fast_client.get_market("abc")
        finally:
            await fast_client.close()

    m = _run(go())
    assert m["id"] == "abc"


@respx.mock
def test_get_market_re_raises_non_404(fast_client: ManifoldClient) -> None:
    respx.get(f"{BASE}/slug/x").mock(return_value=httpx.Response(500))

    async def go() -> None:
        try:
            await fast_client.get_market("x")
        finally:
            await fast_client.close()

    with pytest.raises(httpx.HTTPStatusError):
        _run(go())


# ---------------------------------------------------------------------------
# get_market_positions / get_market_bets
# ---------------------------------------------------------------------------


@respx.mock
def test_get_market_positions_truncates_to_top(fast_client: ManifoldClient) -> None:
    respx.get(f"{BASE}/market/m1/positions").mock(
        return_value=httpx.Response(
            200,
            json=[{"userId": f"u{i}", "shares": 100 - i} for i in range(10)],
        )
    )

    async def go() -> list[dict]:
        try:
            return await fast_client.get_market_positions("m1", top=3)
        finally:
            await fast_client.close()

    out = _run(go())
    assert len(out) == 3
    assert out[0]["userId"] == "u0"


@respx.mock
def test_get_market_bets_returns_recent_trades(fast_client: ManifoldClient) -> None:
    respx.get(f"{BASE}/bets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "b1", "amount": 50, "probAfter": 0.40, "createdTime": 1735689600000},
                {"id": "b2", "amount": -10, "probAfter": 0.38, "createdTime": 1735689700000},
            ],
        )
    )

    async def go() -> list[dict]:
        try:
            return await fast_client.get_market_bets("m1", limit=10)
        finally:
            await fast_client.close()

    out = _run(go())
    assert len(out) == 2
    assert out[0]["probAfter"] == 0.40


# ---------------------------------------------------------------------------
# fetch_history
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_history_aggregates_bets_to_daily(fast_client: ManifoldClient) -> None:
    # Three bets across two UTC days. Day 1: probs 0.30 → 0.32; Day 2: 0.40.
    day1_a = 1735689600000  # 2025-01-01 00:00:00 UTC
    day1_b = 1735718400000  # 2025-01-01 08:00:00 UTC
    day2 = 1735776000000  # 2025-01-02 00:00:00 UTC

    respx.get(f"{BASE}/bets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "b1", "amount": 100, "probAfter": 0.30, "createdTime": day1_a},
                {"id": "b2", "amount": 50, "probAfter": 0.32, "createdTime": day1_b},
                {"id": "b3", "amount": -20, "probAfter": 0.40, "createdTime": day2},
            ],
        )
    )

    async def go() -> pd.DataFrame:
        try:
            return await fast_client.fetch_history("m1", days=365)
        finally:
            await fast_client.close()

    df = _run(go())
    assert list(df.columns) == ["date", "prob", "volume"]
    assert len(df) == 2
    # Day-1 close-of-day prob = last bet of the day = 0.32
    assert df.iloc[0]["prob"] == pytest.approx(0.32)
    # Day-1 volume sum of |amount| = 150
    assert df.iloc[0]["volume"] == pytest.approx(150.0)
    # Day-2 prob = 0.40
    assert df.iloc[1]["prob"] == pytest.approx(0.40)


@respx.mock
def test_fetch_history_empty_bets_returns_empty_frame(fast_client: ManifoldClient) -> None:
    respx.get(f"{BASE}/bets").mock(return_value=httpx.Response(200, json=[]))

    async def go() -> pd.DataFrame:
        try:
            return await fast_client.fetch_history("m1", days=30)
        finally:
            await fast_client.close()

    df = _run(go())
    assert df.empty
    assert list(df.columns) == ["date", "prob", "volume"]


# ---------------------------------------------------------------------------
# 2026-05-15 upstream-hardening: 429 retry + process-local cache
# ---------------------------------------------------------------------------


@respx.mock
def test_search_retries_once_on_429_then_succeeds(fast_client: ManifoldClient, monkeypatch) -> None:
    """A 429 followed by a 200 should produce a 200 — single transparent retry.

    Without the retry the caller surfaces a 502 to the UI and the cross-venue
    arb scanner blanks out on the first rate-limit blip.
    """
    # Shrink the backoff so the test stays fast (default is 1.5 s).
    import pfm.sources.manifold as mm

    monkeypatch.setattr(mm, "_RETRY_BACKOFF_S", 0.01)

    route = respx.get(f"{BASE}/search-markets").mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate-limit"}),
            httpx.Response(200, json=[{"id": "m1", "slug": "a", "question": "Q"}]),
        ]
    )

    async def go() -> list[dict]:
        try:
            return await fast_client.search_markets("hardening", limit=5)
        finally:
            await fast_client.close()

    out = _run(go())
    assert route.call_count == 2, f"expected 2 calls (1 retry), got {route.call_count}"
    assert len(out) == 1 and out[0]["id"] == "m1"


@respx.mock
def test_search_second_call_hits_cache(fast_client: ManifoldClient) -> None:
    """search-markets is cached for 5 min on (base_url, query, limit).

    Two sequential identical queries should result in a single upstream call.
    """
    route = respx.get(f"{BASE}/search-markets").mock(
        return_value=httpx.Response(200, json=[{"id": "m1", "slug": "a", "question": "Q"}])
    )

    async def go() -> tuple[list[dict], list[dict]]:
        try:
            a = await fast_client.search_markets("cached-q", limit=3)
            b = await fast_client.search_markets("cached-q", limit=3)
            return a, b
        finally:
            await fast_client.close()

    a, b = _run(go())
    assert a == b
    assert route.call_count == 1, (
        f"second identical search should hit cache; got {route.call_count} calls"
    )


@respx.mock
def test_get_market_second_call_hits_cache(fast_client: ManifoldClient) -> None:
    """slug→market metadata is cached for 1 h."""
    payload = {"id": "m1", "slug": "cached-slug", "question": "Q"}
    route = respx.get(f"{BASE}/slug/cached-slug").mock(
        return_value=httpx.Response(200, json=payload)
    )

    async def go() -> tuple[dict, dict]:
        try:
            a = await fast_client.get_market("cached-slug")
            b = await fast_client.get_market("cached-slug")
            return a, b
        finally:
            await fast_client.close()

    a, b = _run(go())
    assert a["id"] == "m1" and b["id"] == "m1"
    assert route.call_count == 1
