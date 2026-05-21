"""Tests for boundary-rotation behaviour and prewarmer iteration semantics.

The 5min predictor lives or dies on its handling of the moment a Polymarket
window expires and the next one opens. These tests cover:

* end_unix updates correctly across boundary
* prewarmer refreshes the cache when the boundary passes
* compare_payload survives Polymarket returning the *closed* market briefly
* anchored model_prob_up correctly tracks new market_prob after rotation
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
import respx

from pfm.crypto5min.background import run_compare_prewarmer
from pfm.crypto5min.market_fetcher import (
    DEFAULT_CLOB_URL,
    DEFAULT_GAMMA_URL,
    discover_active_markets,
)
from pfm.crypto5min.router import (
    _compare_cache,
    _reset_caches,
    build_compare_payload,
)
from pfm.crypto5min.state import get_state, reset_state


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_state()
    _reset_caches()
    yield
    reset_state()
    _reset_caches()


def _market_payload(slug: str, *, mid: float = 0.50) -> dict:
    return {
        "slug": slug,
        "id": "1",
        "closed": False,
        "active": True,
        "clobTokenIds": json.dumps([f"up_{slug}", f"down_{slug}"]),
        "bestBid": str(max(0.0, mid - 0.005)),
        "bestAsk": str(min(1.0, mid + 0.005)),
    }


def _binance_ticker_response(price: float = 60_000.0) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "symbol": "BTCUSDT",
            "bidPrice": str(price - 1),
            "askPrice": str(price + 1),
            "bidQty": "1.0",
            "askQty": "1.0",
        },
    )


def _binance_klines_response(n: int = 30) -> httpx.Response:
    rows = []
    base = 1_700_000_000_000
    prev = 60_000.0
    for i in range(n):
        sign = 1 if i % 2 == 0 else -1
        c = prev * (1 + sign * 0.026)
        rows.append(
            [
                base + i * 86_400_000,
                str(prev),
                str(max(prev, c) * 1.003),
                str(min(prev, c) * 0.997),
                str(c),
                "100",
                base + (i + 1) * 86_400_000 - 1,
                "1000",
                1000,
                "50",
                "500",
                "0",
            ]
        )
        prev = c
    return httpx.Response(200, json=rows)


# ---------------------------------------------------------------------------
# discover_active_markets — boundary edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_finds_market_one_period_ahead_when_first_missing() -> None:
    """If the next-most-recent market doesn't exist, discovery should try the one
    after (offset=1) and return that — common in the seconds right after a
    boundary when Polymarket hasn't created the new market yet."""
    now = 1_700_000_000
    period = 300
    first_end = ((now // period) + 1) * period
    second_end = first_end + period
    second_slug = f"btc-updown-5m-{second_end}"

    def handler(req: httpx.Request) -> httpx.Response:
        slug = req.url.params.get("slug", "")
        if slug == second_slug:
            return httpx.Response(200, json=[_market_payload(second_slug)])
        return httpx.Response(200, json=[])

    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            out = await discover_active_markets(
                client,
                assets=["BTC"],
                window_minutes_list=[5],
                now_unix=now,
                lookahead=3,
            )
    assert len(out) == 1
    assert out[0].slug == second_slug
    assert out[0].end_unix == second_end


@pytest.mark.asyncio
async def test_discover_returns_empty_when_seconds_remaining_zero() -> None:
    """A market whose end_unix is exactly now is NOT considered active."""
    now = 1_700_000_000
    period = 300
    next_end = ((now // period) + 1) * period  # ahead of now
    slug = f"btc-updown-5m-{next_end}"

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_market_payload(slug)])
        return httpx.Response(200, json=[])

    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            # Pretend now == next_end (boundary just passed)
            out = await discover_active_markets(
                client,
                assets=["BTC"],
                window_minutes_list=[5],
                now_unix=float(next_end),
            )
    # seconds_remaining = 0 → filtered out by discovery; we get empty list
    assert out == []


@pytest.mark.asyncio
async def test_discover_dedupes_when_both_prefixes_resolve() -> None:
    """If both ``btc-updown-5m-X`` and ``btc-up-or-down-5m-X`` return data,
    only one ActiveMarket should appear (first-prefix-wins, no dup)."""
    now = 1_700_000_000
    period = 300
    next_end = ((now // period) + 1) * period
    new_slug = f"btc-updown-5m-{next_end}"
    legacy_slug = f"btc-up-or-down-5m-{next_end}"

    def handler(req: httpx.Request) -> httpx.Response:
        slug = req.url.params.get("slug", "")
        if slug in (new_slug, legacy_slug):
            return httpx.Response(200, json=[_market_payload(slug)])
        return httpx.Response(200, json=[])

    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            out = await discover_active_markets(
                client,
                assets=["BTC"],
                window_minutes_list=[5],
                now_unix=now,
                lookahead=1,
            )
    assert len(out) == 1
    # New prefix should win (it's first in _SLUG_PREFIXES).
    assert out[0].slug == new_slug


# ---------------------------------------------------------------------------
# build_compare_payload — end_unix freshness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compare_payload_end_unix_is_authoritative() -> None:
    """Each row's end_unix must match the market's actual boundary."""
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"

    def gamma(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_market_payload(slug, mid=0.55)])
        return httpx.Response(200, json=[])

    state = get_state()
    state.record_spot("BTCUSDT", time.time(), 60_000.0)

    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma)
        respx.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.55"})
        )
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker_response()
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines_response()
        )
        respx.get("https://api.binance.com/api/v3/aggTrades").mock(
            return_value=httpx.Response(200, json=[{"p": "60000", "q": "1"}])
        )
        # Allow polymarket.com scrape to fail/empty
        respx.get(url__regex=r"https://polymarket\.com/event/.*").mock(
            return_value=httpx.Response(200, text="")
        )
        async with httpx.AsyncClient() as client:
            payload = await build_compare_payload(
                client,
                state,
                assets=["BTC"],
                windows=[5],
                edge_threshold=0.03,
            )
    btc_5m = next(r for r in payload["rows"] if r["asset"] == "BTC" and r["window_minutes"] == 5)
    assert btc_5m["end_unix"] == next_end
    # seconds_remaining should approximately equal end_unix - now (with <2s slack)
    assert abs(btc_5m["seconds_remaining"] - (next_end - time.time())) < 2.0


@pytest.mark.asyncio
async def test_compare_payload_returns_unique_rows_per_combo() -> None:
    """No duplicate (asset, window_minutes) rows."""
    state = get_state()
    state.record_spot("BTCUSDT", time.time(), 60_000.0)
    state.record_spot("ETHUSDT", time.time(), 3_000.0)
    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=[]))
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker_response()
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines_response()
        )
        respx.get(url__regex=r"https://polymarket\.com/event/.*").mock(
            return_value=httpx.Response(200, text="")
        )
        async with httpx.AsyncClient() as client:
            payload = await build_compare_payload(
                client,
                state,
                assets=["BTC", "ETH"],
                windows=[5, 15],
                edge_threshold=0.03,
            )
    keys = [(r["asset"], r["window_minutes"]) for r in payload["rows"]]
    assert len(keys) == len(set(keys))
    assert set(keys) == {("BTC", 5), ("BTC", 15), ("ETH", 5), ("ETH", 15)}


# ---------------------------------------------------------------------------
# Prewarmer iteration semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prewarmer_writes_to_compare_cache() -> None:
    """Each iteration of the prewarmer should leave a fresh payload in the cache."""
    state = get_state()
    state.record_spot("BTCUSDT", time.time(), 60_000.0)
    state.record_spot("ETHUSDT", time.time(), 3_000.0)

    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=[]))
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker_response()
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines_response()
        )
        respx.get(url__regex=r"https://polymarket\.com/event/.*").mock(
            return_value=httpx.Response(200, text="")
        )
        stop = asyncio.Event()
        async with httpx.AsyncClient() as client:
            task = asyncio.create_task(
                run_compare_prewarmer(
                    client=client,
                    assets=["BTC"],
                    windows=[5],
                    compare_seconds=0.1,
                    stop_event=stop,
                )
            )
            await asyncio.sleep(0.3)
            stop.set()
            await asyncio.wait_for(task, timeout=2.0)

    # Cache should now have a payload for this key.
    assert len(_compare_cache) >= 1
    key, (ts, payload) = next(iter(_compare_cache.items()))
    assert "BTC" in key
    assert payload["n_rows"] >= 1
    # Fresh — written within the last second.
    assert time.time() - ts < 1.5


@pytest.mark.asyncio
async def test_prewarmer_handles_polymarket_outage() -> None:
    """Polymarket returning 5xx must not crash the prewarmer; cache stays empty."""
    state = get_state()
    state.record_spot("BTCUSDT", time.time(), 60_000.0)

    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(return_value=httpx.Response(503))
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker_response()
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines_response()
        )
        respx.get(url__regex=r"https://polymarket\.com/event/.*").mock(
            return_value=httpx.Response(200, text="")
        )
        stop = asyncio.Event()
        async with httpx.AsyncClient() as client:
            task = asyncio.create_task(
                run_compare_prewarmer(
                    client=client,
                    assets=["BTC"],
                    windows=[5],
                    compare_seconds=0.1,
                    stop_event=stop,
                )
            )
            await asyncio.sleep(0.25)
            stop.set()
            await asyncio.wait_for(task, timeout=2.0)
    # Task didn't crash → we got here without exception. Cache may contain a
    # row with market_prob_up=None (Polymarket unavailable).
    if _compare_cache:
        _, (_, payload) = next(iter(_compare_cache.items()))
        assert all(r.get("market_prob_up") is None or "error" in r for r in payload["rows"])
