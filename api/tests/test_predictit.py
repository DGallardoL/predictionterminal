"""Tests for the PredictIt async client (HTTP mocked via respx)."""

from __future__ import annotations

import asyncio

import httpx
import pandas as pd
import pytest
import respx

from pfm.cache_utils import get_cache
from pfm.sources.predictit import (
    PREDICTIT_BASE_URL,
    PredictItClient,
    PredictItError,
)

BASE = PREDICTIT_BASE_URL


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Drop the all-markets cache before AND after each test."""
    get_cache("predictit_all").clear()
    yield
    get_cache("predictit_all").clear()


# Five-market fixture covering presidential, Senate, recession, Fed, BTC.
SAMPLE_MARKETS: list[dict] = [
    {
        "id": 7456,
        "name": "Who will win the 2028 US presidential election?",
        "shortName": "2028 President",
        "url": "https://www.predictit.org/markets/detail/7456",
        "totalSharesTraded": 250_000,
        "dateEnd": "2028-11-07",
        "contracts": [
            {"id": 1, "name": "Trump", "lastTradePrice": 0.42},
            {"id": 2, "name": "Harris", "lastTradePrice": 0.30},
            {"id": 3, "name": "Other", "lastTradePrice": 0.18},
        ],
    },
    {
        "id": 7300,
        "name": "Will the US enter recession by end of 2026?",
        "shortName": "Recession 2026",
        "url": "https://www.predictit.org/markets/detail/7300",
        "totalSharesTraded": 80_000,
        "dateEnd": "2026-12-31",
        "contracts": [{"id": 11, "name": "Yes", "lastTradePrice": 0.27}],
    },
    {
        "id": 7400,
        "name": "Will the Fed cut rates in 2026?",
        "shortName": "Fed cuts 2026",
        "url": "https://www.predictit.org/markets/detail/7400",
        "totalSharesTraded": 120_000,
        "dateEnd": "2026-12-31",
        "contracts": [{"id": 21, "name": "Yes", "lastTradePrice": 0.78}],
    },
    {
        "id": 7500,
        "name": "Will US CPI be above 3.5% YoY in 2026?",
        "shortName": "CPI 3.5% 2026",
        "url": "https://www.predictit.org/markets/detail/7500",
        "totalSharesTraded": 45_000,
        "dateEnd": "2026-12-31",
        "contracts": [{"id": 31, "name": "Yes", "lastTradePrice": 0.22}],
    },
    {
        "id": 7600,
        "name": "Senate control after 2026 midterms",
        "shortName": "Senate 2026",
        "url": "https://www.predictit.org/markets/detail/7600",
        "totalSharesTraded": 60_000,
        "dateEnd": "2026-11-04",
        "contracts": [
            {"id": 41, "name": "Republicans", "lastTradePrice": 0.55},
            {"id": 42, "name": "Democrats", "lastTradePrice": 0.40},
        ],
    },
]


def _client() -> PredictItClient:
    return PredictItClient(client=httpx.AsyncClient())


# ---------------------------------------------------------------------------
# fetch_all_markets
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_all_markets_returns_list_and_caches() -> None:
    route = respx.get(f"{BASE}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"markets": SAMPLE_MARKETS})
    )

    async def go() -> tuple[list[dict], list[dict]]:
        c = _client()
        try:
            first = await c.fetch_all_markets()
            second = await c.fetch_all_markets()  # served from cache
            return first, second
        finally:
            await c.close()

    a, b = _run(go())
    assert len(a) == 5
    assert a == b
    # Cache hit means only one upstream call.
    assert route.call_count == 1


@respx.mock
def test_fetch_all_markets_force_refresh_bypasses_cache() -> None:
    route = respx.get(f"{BASE}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"markets": SAMPLE_MARKETS})
    )

    async def go() -> None:
        c = _client()
        try:
            await c.fetch_all_markets()
            await c.fetch_all_markets(force_refresh=True)
        finally:
            await c.close()

    _run(go())
    assert route.call_count == 2


@respx.mock
def test_fetch_all_markets_raises_on_missing_markets_key() -> None:
    respx.get(f"{BASE}/marketdata/all/").mock(return_value=httpx.Response(200, json={"oops": []}))

    async def go() -> None:
        c = _client()
        try:
            await c.fetch_all_markets()
        finally:
            await c.close()

    with pytest.raises(PredictItError):
        _run(go())


# ---------------------------------------------------------------------------
# fetch_market
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_market_uses_warm_cache() -> None:
    respx.get(f"{BASE}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"markets": SAMPLE_MARKETS})
    )
    # No per-market route registered — the call must come from the cache.

    async def go() -> dict:
        c = _client()
        try:
            await c.fetch_all_markets()
            return await c.fetch_market(7400)
        finally:
            await c.close()

    m = _run(go())
    assert m["id"] == 7400
    assert m["shortName"] == "Fed cuts 2026"


@respx.mock
def test_fetch_market_falls_back_to_per_market_endpoint_when_cache_cold() -> None:
    respx.get(f"{BASE}/marketdata/markets/7300").mock(
        return_value=httpx.Response(200, json=SAMPLE_MARKETS[1])
    )

    async def go() -> dict:
        c = _client()
        try:
            return await c.fetch_market(7300)
        finally:
            await c.close()

    m = _run(go())
    assert m["id"] == 7300


# ---------------------------------------------------------------------------
# fetch_history
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_history_uses_leading_contract_price_and_market_volume() -> None:
    respx.get(f"{BASE}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"markets": SAMPLE_MARKETS})
    )

    async def go() -> pd.DataFrame:
        c = _client()
        try:
            await c.fetch_all_markets()
            return await c.fetch_history(7456, days=30)
        finally:
            await c.close()

    df = _run(go())
    assert list(df.columns) == ["date", "prob", "volume", "contract_id"]
    # Leading contract is "Trump" at 0.42.
    assert df.iloc[0]["prob"] == pytest.approx(0.42)
    assert df.iloc[0]["contract_id"] == 1
    assert df.iloc[0]["volume"] == pytest.approx(250_000.0)


@respx.mock
def test_fetch_history_empty_when_days_zero() -> None:
    respx.get(f"{BASE}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"markets": SAMPLE_MARKETS})
    )

    async def go() -> pd.DataFrame:
        c = _client()
        try:
            await c.fetch_all_markets()
            return await c.fetch_history(7300, days=0)
        finally:
            await c.close()

    df = _run(go())
    assert df.empty
    assert list(df.columns) == ["date", "prob", "volume", "contract_id"]


# ---------------------------------------------------------------------------
# 2026-05-15 upstream-hardening: 429-retry on the venue snapshot
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_all_markets_retries_once_on_429(monkeypatch) -> None:
    """A 429 followed by a 200 should produce a 200 — one transparent retry.

    The shared 5-min venue cache means a 502 here would lock the cross-venue
    arb scanner out for up to 5 minutes; the retry stops that.
    """
    import pfm.sources.predictit as pi

    monkeypatch.setattr(pi, "_RETRY_BACKOFF_S", 0.01)

    route = respx.get(f"{BASE}/marketdata/all/").mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate-limit"}),
            httpx.Response(200, json={"markets": SAMPLE_MARKETS}),
        ]
    )

    async def go() -> list[dict]:
        c = _client()
        try:
            return await c.fetch_all_markets()
        finally:
            await c.close()

    out = _run(go())
    assert route.call_count == 2, f"expected 2 calls (1 retry), got {route.call_count}"
    assert isinstance(out, list) and len(out) > 0
