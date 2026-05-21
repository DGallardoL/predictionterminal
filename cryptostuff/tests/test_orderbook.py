"""Tests for order book reconstruction."""

from __future__ import annotations

import json

import fakeredis.aioredis
import httpx
import pytest

from extraction.orderbook import OrderBookState


@pytest.fixture
async def ob() -> OrderBookState:
    state = OrderBookState(redis_url="redis://fake")
    state._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    state._http = httpx.AsyncClient(
        transport=httpx.MockTransport(_mock_depth_handler),
        base_url="http://mock",
    )
    yield state
    await state.close()


def _mock_depth_handler(request: httpx.Request) -> httpx.Response:
    snapshot = {
        "lastUpdateId": 100,
        "bids": [["100.0", "1.0"], ["99.5", "2.0"]],
        "asks": [["101.0", "1.0"], ["101.5", "2.5"]],
    }
    return httpx.Response(200, content=json.dumps(snapshot))


async def test_load_snapshot(ob: OrderBookState) -> None:
    await ob.load_snapshot("BTCUSDT")
    bids = await ob.get_top_n_bids("BTCUSDT", n=10)
    asks = await ob.get_top_n_asks("BTCUSDT", n=10)
    assert bids[0] == (100.0, 1.0)
    assert bids[1] == (99.5, 2.0)
    assert asks[0] == (101.0, 1.0)
    assert asks[1] == (101.5, 2.5)


async def test_apply_update(ob: OrderBookState) -> None:
    await ob.load_snapshot("BTCUSDT")
    update = {
        "e": "depthUpdate", "E": 1700000000000, "s": "BTCUSDT",
        "U": 101, "u": 105,
        "b": [["100.5", "0.5"]], "a": [["101.0", "0.0"]],
    }
    await ob.apply_update(update)
    assert ob.metrics.updates_applied == 1


async def test_old_update_skipped(ob: OrderBookState) -> None:
    await ob.load_snapshot("BTCUSDT")
    update = {
        "e": "depthUpdate", "E": 1700000000000, "s": "BTCUSDT",
        "U": 50, "u": 90, "b": [], "a": [],
    }
    await ob.apply_update(update)
    assert ob.metrics.updates_skipped_old == 1


async def test_removes_level(ob: OrderBookState) -> None:
    await ob.load_snapshot("BTCUSDT")
    update = {
        "e": "depthUpdate", "E": 1700000000000, "s": "BTCUSDT",
        "U": 101, "u": 105,
        "b": [["100.0", "0.0"]], "a": [],
    }
    await ob.apply_update(update)
    bids = await ob.get_top_n_bids("BTCUSDT", n=10)
    prices = [b[0] for b in bids]
    assert 100.0 not in prices
    assert 99.5 in prices


async def test_buffers_before_snapshot(ob: OrderBookState) -> None:
    update = {
        "e": "depthUpdate", "E": 1700000000000, "s": "ETHUSDT",
        "U": 1, "u": 5, "b": [], "a": [],
    }
    await ob.apply_update(update)
    assert ob.metrics.updates_applied == 0
    assert "ETHUSDT" in ob._buffer


async def test_obi_calculation(ob: OrderBookState) -> None:
    await ob.load_snapshot("BTCUSDT")
    obi = await ob.get_obi("BTCUSDT", levels=2)
    assert obi is not None
    # bids: 1.0 + 2.0 = 3.0, asks: 1.0 + 2.5 = 3.5
    expected = (3.0 - 3.5) / (3.0 + 3.5)
    assert abs(obi - expected) < 0.001
