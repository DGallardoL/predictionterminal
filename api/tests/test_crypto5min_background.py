"""Tests for the background spot sampler."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from pfm.crypto5min.background import run_sampler
from pfm.crypto5min.state import get_state, reset_state


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_state()
    yield
    reset_state()


def _binance_ticker_response(symbol: str, price: float) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "symbol": symbol,
            "bidPrice": str(price - 1),
            "askPrice": str(price + 1),
            "bidQty": "1",
            "askQty": "1",
        },
    )


@pytest.mark.asyncio
async def test_sampler_records_samples_then_stops() -> None:
    """The sampler must push samples into the singleton buffer."""
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            side_effect=lambda req: _binance_ticker_response(
                req.url.params["symbol"],
                60_000.0,
            )
        )
        stop = asyncio.Event()
        async with httpx.AsyncClient() as client:
            task = asyncio.create_task(
                run_sampler(
                    client=client,
                    symbols=["BTCUSDT"],
                    poll_seconds=0.05,
                    stop_event=stop,
                )
            )
            await asyncio.sleep(0.20)
            stop.set()
            await asyncio.wait_for(task, timeout=2.0)
    state = get_state()
    assert state.n_samples("BTCUSDT") >= 1


@pytest.mark.asyncio
async def test_sampler_handles_server_error() -> None:
    """A 5xx from Binance must not crash the loop — it just skips the sample."""
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=httpx.Response(503)
        )
        stop = asyncio.Event()
        async with httpx.AsyncClient() as client:
            task = asyncio.create_task(
                run_sampler(
                    client=client,
                    symbols=["BTCUSDT"],
                    poll_seconds=0.05,
                    stop_event=stop,
                )
            )
            await asyncio.sleep(0.10)
            stop.set()
            await asyncio.wait_for(task, timeout=2.0)
    state = get_state()
    assert state.n_samples("BTCUSDT") == 0


@pytest.mark.asyncio
async def test_sampler_can_be_cancelled() -> None:
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=httpx.Response(503)
        )
        async with httpx.AsyncClient() as client:
            task = asyncio.create_task(
                run_sampler(
                    client=client,
                    symbols=["BTCUSDT"],
                    poll_seconds=10.0,  # long enough that cancellation is required
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_sampler_serves_multiple_symbols_in_parallel() -> None:
    """One iteration of run_sampler should fan out across symbols."""
    seen: set[str] = set()

    def handler(req: httpx.Request) -> httpx.Response:
        seen.add(req.url.params["symbol"])
        return _binance_ticker_response(req.url.params["symbol"], 1_000.0)

    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(side_effect=handler)
        stop = asyncio.Event()
        async with httpx.AsyncClient() as client:
            task = asyncio.create_task(
                run_sampler(
                    client=client,
                    symbols=["BTCUSDT", "ETHUSDT"],
                    poll_seconds=0.05,
                    stop_event=stop,
                )
            )
            await asyncio.sleep(0.20)
            stop.set()
            await asyncio.wait_for(task, timeout=2.0)
    assert {"BTCUSDT", "ETHUSDT"} <= seen
    state = get_state()
    assert state.n_samples("BTCUSDT") >= 1
    assert state.n_samples("ETHUSDT") >= 1
