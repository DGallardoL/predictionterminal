"""Tests for :mod:`pfm.terminal.orderbook_pool`.

All upstream HTTP is mocked via :mod:`respx`. The shared
``PolymarketHTTPPool`` singleton is reset between tests so each one starts
with a fresh ``httpx.AsyncClient`` and the per-token cache is empty.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

from pfm.sources.polymarket_pool import CLOB_BASE_URL, PolymarketHTTPPool
from pfm.terminal.orderbook_pool import (
    BATCH_CONCURRENCY,
    OrderbookPool,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(autouse=True)
async def _reset_singletons() -> Any:
    """Drop both singletons before AND after each test.

    The HTTPPool's underlying ``httpx.AsyncClient`` is bound to the running
    event loop; pytest-asyncio runs each test in its own loop, so reusing
    a stale client across tests raises ``RuntimeError``.
    """
    await _safe_close()
    yield
    await _safe_close()


async def _safe_close() -> None:
    # OrderbookPool first (it just drops state); HTTPPool actually closes
    # the underlying clients.
    op = OrderbookPool._instance
    if op is not None:
        with contextlib.suppress(Exception):
            await op.aclose()
    OrderbookPool.reset_for_testing()
    hp = PolymarketHTTPPool._instance
    if hp is not None:
        with contextlib.suppress(Exception):
            await hp.aclose()
    PolymarketHTTPPool.reset_for_testing()


def _book_payload(*, bid: float = 0.55, ask: float = 0.57) -> dict:
    return {
        "bids": [
            {"price": str(bid), "size": "1000"},
            {"price": str(round(bid - 0.01, 4)), "size": "500"},
        ],
        "asks": [
            {"price": str(ask), "size": "800"},
            {"price": str(round(ask + 0.01, 4)), "size": "300"},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_singleton_returns_same_instance() -> None:
    a = OrderbookPool.instance()
    b = OrderbookPool.instance()
    assert a is b


@pytest.mark.asyncio
async def test_reset_for_testing_drops_singleton() -> None:
    a = OrderbookPool.instance()
    OrderbookPool.reset_for_testing()
    b = OrderbookPool.instance()
    assert a is not b


@pytest.mark.asyncio
@respx.mock
async def test_get_snapshot_basic_shape() -> None:
    route = respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(200, json=_book_payload())
    )
    pool = OrderbookPool.instance()

    snap = await pool.get_snapshot("tok-1")

    assert route.called
    assert snap is not None
    assert "bids" in snap and "asks" in snap and "updated_at" in snap
    # Normalised to [[price, size], ...]
    assert snap["bids"][0] == [0.55, 1000.0]
    assert snap["asks"][0] == [0.57, 800.0]


@pytest.mark.asyncio
@respx.mock
async def test_cache_hit_within_max_age_s() -> None:
    route = respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(200, json=_book_payload())
    )
    pool = OrderbookPool.instance()

    snap1 = await pool.get_snapshot("tok-cache", max_age_s=60)
    snap2 = await pool.get_snapshot("tok-cache", max_age_s=60)

    assert snap1 is snap2  # exact same dict object — true cache hit
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_cache_miss_for_different_token() -> None:
    route = respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(200, json=_book_payload())
    )
    pool = OrderbookPool.instance()

    await pool.get_snapshot("tok-A")
    await pool.get_snapshot("tok-B")

    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_stale_entry_is_refetched(monkeypatch: pytest.MonkeyPatch) -> None:
    route = respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(200, json=_book_payload())
    )
    pool = OrderbookPool.instance()

    # First call populates cache at t=100
    fake_t = {"now": 100.0}
    monkeypatch.setattr("pfm.terminal.orderbook_pool.time.monotonic", lambda: fake_t["now"])
    await pool.get_snapshot("tok-stale", max_age_s=30)
    assert route.call_count == 1

    # Within window: cache hit
    fake_t["now"] = 125.0
    await pool.get_snapshot("tok-stale", max_age_s=30)
    assert route.call_count == 1

    # Outside window: refetch
    fake_t["now"] = 200.0
    await pool.get_snapshot("tok-stale", max_age_s=30)
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_max_age_zero_forces_refetch() -> None:
    route = respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(200, json=_book_payload())
    )
    pool = OrderbookPool.instance()

    await pool.get_snapshot("tok-zero", max_age_s=60)
    await pool.get_snapshot("tok-zero", max_age_s=0)
    await pool.get_snapshot("tok-zero", max_age_s=0)

    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_network_500_returns_none() -> None:
    respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )
    pool = OrderbookPool.instance()

    snap = await pool.get_snapshot("tok-fail")

    assert snap is None
    # Failed fetches do NOT poison the cache
    assert "tok-fail" not in pool._cache


@pytest.mark.asyncio
@respx.mock
async def test_503_returns_none() -> None:
    respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(503, text="Service Unavailable")
    )
    pool = OrderbookPool.instance()

    snap = await pool.get_snapshot("tok-503")
    assert snap is None


@pytest.mark.asyncio
@respx.mock
async def test_404_returns_none() -> None:
    respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(404, json={"error": "unknown token"})
    )
    pool = OrderbookPool.instance()

    assert await pool.get_snapshot("tok-404") is None


@pytest.mark.asyncio
@respx.mock
async def test_connect_error_returns_none() -> None:
    respx.get(f"{CLOB_BASE_URL}/book").mock(side_effect=httpx.ConnectError("connection refused"))
    pool = OrderbookPool.instance()

    assert await pool.get_snapshot("tok-net") is None


@pytest.mark.asyncio
@respx.mock
async def test_get_snapshots_parallel_bounded_concurrency() -> None:
    """20 tokens fan out, but never more than BATCH_CONCURRENCY (=10) in
    flight at once."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        # Give the scheduler time to ramp up so peak reflects parallelism
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return httpx.Response(200, json=_book_payload())

    respx.get(f"{CLOB_BASE_URL}/book").mock(side_effect=_handler)
    pool = OrderbookPool.instance()

    tokens = [f"tok-{i}" for i in range(20)]
    result = await pool.get_snapshots(tokens)

    assert len(result) == 20
    assert peak <= BATCH_CONCURRENCY
    assert peak >= 2  # sanity: we did fan out


@pytest.mark.asyncio
@respx.mock
async def test_get_snapshots_partial_failure_omits_bad_tokens() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        tid = request.url.params.get("token_id", "")
        if tid.endswith("bad"):
            return httpx.Response(500)
        return httpx.Response(200, json=_book_payload())

    respx.get(f"{CLOB_BASE_URL}/book").mock(side_effect=_handler)
    pool = OrderbookPool.instance()

    out = await pool.get_snapshots(["tok-ok-1", "tok-bad", "tok-ok-2"])
    assert set(out.keys()) == {"tok-ok-1", "tok-ok-2"}


@pytest.mark.asyncio
@respx.mock
async def test_get_snapshots_dedupes_input() -> None:
    route = respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(200, json=_book_payload())
    )
    pool = OrderbookPool.instance()

    out = await pool.get_snapshots(["tok-dup", "tok-dup", "tok-dup"])

    assert len(out) == 1
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_get_snapshots_empty_list() -> None:
    pool = OrderbookPool.instance()
    assert await pool.get_snapshots([]) == {}


@pytest.mark.asyncio
@respx.mock
async def test_warm_prefetches_into_cache() -> None:
    route = respx.get(f"{CLOB_BASE_URL}/book").mock(
        return_value=httpx.Response(200, json=_book_payload())
    )
    pool = OrderbookPool.instance()

    await pool.warm(["tok-w1", "tok-w2", "tok-w3"])
    assert route.call_count == 3

    # Subsequent get_snapshot is served from cache
    snap = await pool.get_snapshot("tok-w1")
    assert snap is not None
    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_concurrent_same_token_collapses_to_one_fetch() -> None:
    """A herd of 5 concurrent calls for the same token issues ONE upstream
    request, not 5 (per-token lock)."""

    async def _slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.03)
        return httpx.Response(200, json=_book_payload())

    route = respx.get(f"{CLOB_BASE_URL}/book").mock(side_effect=_slow)
    pool = OrderbookPool.instance()

    results = await asyncio.gather(*(pool.get_snapshot("tok-herd") for _ in range(5)))

    assert all(r is not None for r in results)
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_aclose_clears_state_and_blocks_further_use() -> None:
    pool = OrderbookPool.instance()
    pool._cache["tok-X"] = (0.0, {"bids": [], "asks": [], "updated_at": "x"})

    await pool.aclose()

    assert pool._cache == {}
    assert pool._locks == {}
    assert pool._closed is True
    with pytest.raises(RuntimeError):
        await pool.get_snapshot("tok-anything")
    with pytest.raises(RuntimeError):
        await pool.get_snapshots(["x"])


@pytest.mark.asyncio
@respx.mock
async def test_normalize_handles_list_levels() -> None:
    """Some CLOB responses come back as [[price, size], ...] instead of
    dicts — both should be accepted."""
    payload = {
        "bids": [[0.6, 100], [0.59, 50]],
        "asks": [[0.61, 80]],
    }
    respx.get(f"{CLOB_BASE_URL}/book").mock(return_value=httpx.Response(200, json=payload))
    pool = OrderbookPool.instance()

    snap = await pool.get_snapshot("tok-listlvl")
    assert snap is not None
    assert snap["bids"] == [[0.6, 100.0], [0.59, 50.0]]
    assert snap["asks"] == [[0.61, 80.0]]


@pytest.mark.asyncio
@respx.mock
async def test_normalize_skips_malformed_levels() -> None:
    payload = {
        "bids": [
            {"price": "0.5", "size": "10"},
            {"price": "garbage", "size": "10"},
            "not-a-level",
            {"price": 0.49, "size": 5},
        ],
        "asks": [],
    }
    respx.get(f"{CLOB_BASE_URL}/book").mock(return_value=httpx.Response(200, json=payload))
    pool = OrderbookPool.instance()

    snap = await pool.get_snapshot("tok-malformed")
    assert snap is not None
    assert snap["bids"] == [[0.5, 10.0], [0.49, 5.0]]


@pytest.mark.asyncio
async def test_reuses_clob_client_from_http_pool() -> None:
    """Sanity check: the pool's ``_client()`` returns the shared CLOB
    client from PolymarketHTTPPool — that's the whole point of this
    module."""
    pool = OrderbookPool.instance()
    expected = PolymarketHTTPPool.instance().clob_client
    assert pool._client() is expected
