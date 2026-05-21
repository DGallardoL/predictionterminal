"""DEEP exhaustive tests for pfm data sources.

Covers:
    - Polymarket (Gamma + CLOB) client
    - Kalshi rate-limit + candlesticks
    - Manifold (search, market, bets, history)
    - PredictIt (snapshot cache, leading-contract, force_refresh)
    - FRED extended (20 series catalog + endpoint + caching)
    - BLS (POST API + catalog + endpoint)
    - Equity multi-source cascade (yfinance -> Tiingo -> Stooq) + delisted
    - Sources health endpoint and probes
    - Macro calendar (FOMC/CPI/NFP/PPI/Retail/GDP for 2026)
    - Macro overlay unified
    - Multi-venue search and concept maps
    - Cross-venue arb scanner

All HTTP is mocked via respx -- zero live network calls.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import arb_scanner as arb_mod
from pfm import macro_calendar as macal_mod
from pfm import multi_venue_search as mvs_mod
from pfm.cache_utils import get_cache, reset_caches
from pfm.macro_calendar import (
    _FOMC_2026,
    next_releases,
)
from pfm.macro_calendar import (
    router as macro_calendar_router,
)
from pfm.macro_overlay_unified import router as macro_overlay_router
from pfm.multi_venue_search import (
    router as multi_venue_router,
)
from pfm.multi_venue_search import (
    search_all_venues,
)
from pfm.sources import equity as equity_mod
from pfm.sources import predictit as predictit_mod
from pfm.sources import stooq as stooq_src
from pfm.sources import tiingo as tiingo_src
from pfm.sources.bls import (
    _BLS_SERIES_REGISTRY,
    BLS_API_BASE,
    BLSClient,
    BlsDataError,
    fetch_bls_series,
)
from pfm.sources.bls import (
    router as bls_router,
)
from pfm.sources.equity import (
    EquityDataError,
    EquityDelistedError,
    get_log_returns,
    is_delisted,
    list_delisted,
)
from pfm.sources.fred import (
    _SERIES_REGISTRY,
    FREDGRAPH_BASE,
    FredDataError,
    fetch_fred_series,
    fetch_fred_series_cached,
)
from pfm.sources.fred import (
    router as fred_router,
)
from pfm.sources.health import (
    check_all_sources,
    check_polymarket,
    check_tiingo,
)
from pfm.sources.health_router import router as sources_router
from pfm.sources.kalshi import (
    KalshiClient,
    KalshiRateLimitError,
)
from pfm.sources.manifold import (
    MANIFOLD_BASE_URL,
    ManifoldClient,
    ManifoldError,
)
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    discover_markets,
    fetch_factor_history,
)
from pfm.sources.predictit import (
    PREDICTIT_BASE_URL,
    PredictItClient,
    PredictItError,
)

GAMMA = "https://gamma-test.local"
CLOB = "https://clob-test.local"
BLS_BASE = BLS_API_BASE
KALSHI_BASE = KalshiClient.BASE_URL
PREDICTIT = PREDICTIT_BASE_URL
MANIFOLD = MANIFOLD_BASE_URL


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _wipe_all_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Force a clean cache state and isolated delisted registry per test."""
    reset_caches()
    monkeypatch.setattr(equity_mod, "DELISTED_REGISTRY_PATH", tmp_path / "delisted.json")
    equity_mod._EQUITY_CACHE.clear()
    predictit_mod._ALL_CACHE.clear()
    mvs_mod._SEARCH_CACHE.clear()
    mvs_mod._CONCEPT_CACHE.clear()
    yield
    reset_caches()
    equity_mod._EQUITY_CACHE.clear()
    predictit_mod._ALL_CACHE.clear()
    mvs_mod._SEARCH_CACHE.clear()
    mvs_mod._CONCEPT_CACHE.clear()


# ---------------------------------------------------------------------------
# 1. Polymarket source
# ---------------------------------------------------------------------------


@pytest.fixture
def pm_client() -> PolymarketClient:
    return PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())


@respx.mock
def test_pm_clob_token_ids_double_json_parse(pm_client: PolymarketClient) -> None:
    """clobTokenIds is a JSON STRING inside the JSON response -- decode twice."""
    inner = json.dumps(["yes_tok_123", "no_tok_456"])
    respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "x",
                    "question": "Q?",
                    "clobTokenIds": inner,
                    "active": True,
                    "closed": False,
                }
            ],
        )
    )
    meta = pm_client.get_market_metadata("x")
    assert meta.yes_token_id == "yes_tok_123"
    assert meta.no_token_id == "no_tok_456"


@respx.mock
def test_pm_fidelity_always_1440(pm_client: PolymarketClient) -> None:
    route = respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": []})
    )
    pm_client.get_price_history("tok")
    assert route.calls.last.request.url.params["fidelity"] == "1440"


@respx.mock
def test_pm_unix_seconds_to_utc_normalized_dates(pm_client: PolymarketClient) -> None:
    # 1735689600 = 2025-01-01 00:00:00 UTC; 1735776000 = 2025-01-02 00:00:00 UTC
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(
            200,
            json={
                "history": [
                    {"t": 1735689600, "p": 0.42},
                    {"t": 1735776000, "p": 0.45},
                ]
            },
        )
    )
    df = pm_client.get_price_history("tok")
    assert df.iloc[0]["date"] == pd.Timestamp("2025-01-01", tz="UTC")
    assert df.iloc[1]["date"] == pd.Timestamp("2025-01-02", tz="UTC")
    # Normalised to midnight UTC: hour=0, no offset shenanigans.
    assert all(d.hour == 0 for d in df["date"])


@respx.mock
def test_pm_empty_history_returns_empty_frame(pm_client: PolymarketClient) -> None:
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(200, json={"history": []}))
    df = pm_client.get_price_history("tok")
    assert df.empty
    assert list(df.columns) == ["date", "price"]


@respx.mock
def test_pm_metadata_falls_back_to_closed_true(pm_client: PolymarketClient) -> None:
    """When default filter returns empty, retry with closed=true."""
    inner = json.dumps(["111", "222"])

    # First call (no closed param) -> empty list.
    # Second call (closed=true) -> the resolved market.
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("closed") == "true":
            return httpx.Response(
                200,
                json=[
                    {
                        "slug": "resolved-mkt",
                        "question": "Q?",
                        "clobTokenIds": inner,
                        "active": False,
                        "closed": True,
                    }
                ],
            )
        return httpx.Response(200, json=[])

    respx.get(f"{GAMMA}/markets").mock(side_effect=_handler)
    meta = pm_client.get_market_metadata("resolved-mkt")
    assert meta.closed is True
    assert meta.yes_token_id == "111"


@respx.mock
def test_pm_metadata_missing_clob_token_ids_raises(pm_client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "x",
                    "question": "Q?",
                    "clobTokenIds": None,
                    "active": True,
                    "closed": False,
                }
            ],
        )
    )
    with pytest.raises(PolymarketError, match="clobTokenIds"):
        pm_client.get_market_metadata("x")


@respx.mock
def test_pm_metadata_too_few_token_ids_raises(pm_client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "x",
                    "question": "Q?",
                    "clobTokenIds": json.dumps(["only_one"]),
                    "active": True,
                    "closed": False,
                }
            ],
        )
    )
    with pytest.raises(PolymarketError, match=r">=2|≥2"):
        pm_client.get_market_metadata("x")


@respx.mock
def test_pm_fetch_factor_history_retries_once_on_timeout(pm_client: PolymarketClient) -> None:
    """ReadTimeout the first time, succeed on the second attempt."""
    inner = json.dumps(["111", "222"])

    calls = {"n": 0}

    def _gamma_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("simulated")
        return httpx.Response(
            200,
            json=[
                {
                    "slug": "x",
                    "question": "?",
                    "clobTokenIds": inner,
                    "active": True,
                    "closed": False,
                }
            ],
        )

    respx.get(f"{GAMMA}/markets").mock(side_effect=_gamma_handler)
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": [{"t": 1735689600, "p": 0.5}]})
    )

    df = fetch_factor_history(pm_client, "x")
    assert calls["n"] == 2
    assert df.iloc[0]["price"] == 0.5


@respx.mock
def test_pm_discover_markets_walks_pages_volume_filter() -> None:
    """discover_markets should paginate using offset and apply min_volume."""
    pm = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    page0 = [
        {
            "slug": f"m{i}",
            "question": f"Q{i}",
            "volume": 2_000_000.0,
            "active": True,
            "closed": False,
            "endDate": "2026-12-31T00:00:00Z",
        }
        for i in range(3)
    ]
    page1 = [
        {
            "slug": f"m{i + 3}",
            "question": f"Q{i + 3}",
            "volume": 500.0,  # below threshold
            "active": True,
            "closed": False,
            "endDate": None,
        }
        for i in range(3)
    ]

    def _handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", 0))
        if offset == 0:
            return httpx.Response(200, json=page0)
        if offset == 100:
            return httpx.Response(200, json=page1)
        return httpx.Response(200, json=[])

    respx.get(f"{GAMMA}/markets").mock(side_effect=_handler)
    out = discover_markets(pm, min_volume=1_000_000.0, limit=10, pages=3)
    assert len(out) == 3
    assert all(c.volume >= 1_000_000.0 for c in out)
    pm.close()


@respx.mock
def test_pm_5xx_propagates(pm_client: PolymarketClient) -> None:
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        pm_client.get_price_history("tok")


# ---------------------------------------------------------------------------
# 2. Kalshi source
# ---------------------------------------------------------------------------


@respx.mock
def test_kalshi_429_with_retry_after_header() -> None:
    slept: list[float] = []
    ticker = "KXFOO-26MAY"
    url = f"{KALSHI_BASE}/markets/{ticker}"
    respx.get(url).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "3"}),
            httpx.Response(
                200,
                json={
                    "market": {
                        "ticker": ticker,
                        "event_ticker": ticker,
                        "title": "T",
                        "status": "active",
                    }
                },
            ),
        ]
    )
    c = KalshiClient(min_interval_s=0.0, max_retries=3, sleep=slept.append)
    m = c.get_market(ticker)
    assert m.ticker == ticker
    assert slept == [3.0]


@respx.mock
def test_kalshi_retries_exhausted_raises_rate_limit_error() -> None:
    """Max retries (5 by default) -> KalshiRateLimitError."""
    slept: list[float] = []
    ticker = "KXFOO-26MAY"
    url = f"{KALSHI_BASE}/markets/{ticker}"
    respx.get(url).mock(return_value=httpx.Response(429, headers={"Retry-After": "0"}))
    c = KalshiClient(min_interval_s=0.0, max_retries=5, sleep=slept.append)
    with pytest.raises(KalshiRateLimitError):
        c.get_market(ticker)
    # 1 initial + 5 retries = 6 attempts, 5 sleeps in between.
    assert len(slept) == 5


@respx.mock
def test_kalshi_no_auth_required() -> None:
    """Kalshi public API: client should NOT require any auth header."""
    ticker = "KXFOO-26MAY"
    url = f"{KALSHI_BASE}/markets/{ticker}"
    route = respx.get(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "market": {
                    "ticker": ticker,
                    "event_ticker": ticker,
                    "title": "T",
                    "status": "active",
                }
            },
        )
    )
    c = KalshiClient(min_interval_s=0.0, max_retries=0)
    c.get_market(ticker)
    req = route.calls.last.request
    # No Authorization or X-API-Key headers added by our client.
    assert "authorization" not in {k.lower() for k in req.headers}
    assert "x-api-key" not in {k.lower() for k in req.headers}


@respx.mock
def test_kalshi_candlesticks_settled_market() -> None:
    """Settled market candlesticks: per-bar bid/ask/spread populated."""
    ticker = "KXFOO-26MAY"
    series = "KXFOO"
    url = f"{KALSHI_BASE}/series/{series}/markets/{ticker}/candlesticks"
    payload = {
        "candlesticks": [
            {
                "end_period_ts": 1735689600,
                "price": {"close_dollars": 0.42},
                "yes_bid": {"close_dollars": 0.41},
                "yes_ask": {"close_dollars": 0.43},
                "volume_fp": 12345.0,
                "open_interest_fp": 9000.0,
            },
            {
                "end_period_ts": 1735776000,
                "price": {"close_dollars": 0.50},
                "yes_bid": {"close_dollars": 0.49},
                "yes_ask": {"close_dollars": 0.51},
                "volume_fp": 5000.0,
                "open_interest_fp": 9500.0,
            },
        ]
    }
    respx.get(url).mock(return_value=httpx.Response(200, json=payload))
    c = KalshiClient(min_interval_s=0.0, max_retries=0)
    df = c.get_candlesticks(ticker, start_ts=1, end_ts=int(time.time()))
    assert len(df) == 2
    # Spread reported for each bar.
    assert df.iloc[0]["spread"] == pytest.approx(0.02)
    assert df.iloc[0]["yes_bid"] == 0.41
    assert df.iloc[0]["yes_ask"] == 0.43
    # UTC date alignment.
    assert df.index[0] == pd.Timestamp("2025-01-01", tz="UTC")


@respx.mock
def test_kalshi_400_does_not_retry() -> None:
    """Non-429 errors are NOT retried."""
    ticker = "KXFOO-26MAY"
    url = f"{KALSHI_BASE}/markets/{ticker}"
    route = respx.get(url).mock(return_value=httpx.Response(400))
    c = KalshiClient(min_interval_s=0.0, max_retries=5)
    with pytest.raises(httpx.HTTPStatusError):
        c.get_market(ticker)
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# 3. Manifold source
# ---------------------------------------------------------------------------


def _manifold_client() -> ManifoldClient:
    return ManifoldClient(client=httpx.AsyncClient(), concurrency=5, min_interval_s=0.0)


@respx.mock
def test_manifold_search_normalises_results() -> None:
    respx.get(f"{MANIFOLD}/search-markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "a", "slug": "s1", "question": "Q1?"},
                {"id": "b", "slug": "s2", "question": "Q2?"},
            ],
        )
    )

    async def go() -> list[dict[str, Any]]:
        c = _manifold_client()
        try:
            return await c.search_markets("foo", limit=5)
        finally:
            await c.close()

    out = _run(go())
    assert len(out) == 2 and out[0]["id"] == "a"


@respx.mock
def test_manifold_404_falls_back_to_id_endpoint() -> None:
    """Slug endpoint 404 -> id endpoint."""
    respx.get(f"{MANIFOLD}/slug/foo").mock(return_value=httpx.Response(404))
    respx.get(f"{MANIFOLD}/market/foo").mock(
        return_value=httpx.Response(200, json={"id": "foo", "probability": 0.5})
    )

    async def go() -> dict[str, Any]:
        c = _manifold_client()
        try:
            return await c.get_market("foo")
        finally:
            await c.close()

    m = _run(go())
    assert m["id"] == "foo"


@respx.mock
def test_manifold_500_propagates() -> None:
    respx.get(f"{MANIFOLD}/slug/foo").mock(return_value=httpx.Response(500))

    async def go() -> dict[str, Any]:
        c = _manifold_client()
        try:
            return await c.get_market("foo")
        finally:
            await c.close()

    with pytest.raises(httpx.HTTPStatusError):
        _run(go())


@respx.mock
def test_manifold_history_aggregates_to_daily_buckets() -> None:
    """Bets in same UTC day -> last probAfter for that day."""
    day1_a = 1735689600000  # 2025-01-01 00:00 UTC
    day1_b = 1735718400000  # 2025-01-01 08:00 UTC
    day2 = 1735776000000  # 2025-01-02 00:00 UTC
    respx.get(f"{MANIFOLD}/bets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "1", "amount": 100.0, "probAfter": 0.20, "createdTime": day1_a},
                {"id": "2", "amount": -50.0, "probAfter": 0.25, "createdTime": day1_b},
                {"id": "3", "amount": 30.0, "probAfter": 0.40, "createdTime": day2},
            ],
        )
    )

    async def go() -> pd.DataFrame:
        c = _manifold_client()
        try:
            return await c.fetch_history("m1", days=365)
        finally:
            await c.close()

    df = _run(go())
    assert len(df) == 2
    assert df.iloc[0]["prob"] == pytest.approx(0.25)  # last of day-1
    assert df.iloc[0]["volume"] == pytest.approx(150.0)  # |100| + |-50|


@respx.mock
def test_manifold_empty_bets_returns_empty_dataframe() -> None:
    respx.get(f"{MANIFOLD}/bets").mock(return_value=httpx.Response(200, json=[]))

    async def go() -> pd.DataFrame:
        c = _manifold_client()
        try:
            return await c.fetch_history("m1", days=30)
        finally:
            await c.close()

    df = _run(go())
    assert df.empty


@respx.mock
def test_manifold_search_non_list_raises_error() -> None:
    respx.get(f"{MANIFOLD}/search-markets").mock(
        return_value=httpx.Response(200, json={"err": "boom"})
    )

    async def go() -> Any:
        c = _manifold_client()
        try:
            return await c.search_markets("x")
        finally:
            await c.close()

    with pytest.raises(ManifoldError):
        _run(go())


def test_manifold_pacing_observable() -> None:
    """When min_interval_s > 0, two sequential calls observe the gap."""

    async def go() -> float:
        async with respx.mock(assert_all_called=False) as respx_mock:
            respx_mock.get(f"{MANIFOLD}/bets").mock(return_value=httpx.Response(200, json=[]))
            mc = ManifoldClient(client=httpx.AsyncClient(), concurrency=5, min_interval_s=0.05)
            try:
                t0 = asyncio.get_event_loop().time()
                await mc.get_market_bets("m1", limit=1)
                await mc.get_market_bets("m1", limit=1)
                return asyncio.get_event_loop().time() - t0
            finally:
                await mc.close()

    elapsed = _run(go())
    # 2 calls, 0.05s spacing -> at least ~0.05s elapsed.
    assert elapsed >= 0.04


# ---------------------------------------------------------------------------
# 4. PredictIt source
# ---------------------------------------------------------------------------


SAMPLE_PREDICTIT = [
    {
        "id": 7456,
        "name": "2028 President",
        "shortName": "2028 President",
        "url": "https://www.predictit.org/markets/detail/7456",
        "totalSharesTraded": 250_000,
        "contracts": [
            {"id": 1, "name": "Trump", "lastTradePrice": 0.42},
            {"id": 2, "name": "Harris", "lastTradePrice": 0.30},
        ],
    },
    {
        "id": 7300,
        "name": "Recession 2026",
        "shortName": "Recession 2026",
        "url": "u",
        "totalSharesTraded": 5_000,
        "contracts": [{"id": 11, "name": "Yes", "lastTradePrice": 0.27}],
    },
    {
        "id": 7400,
        "name": "Fed cuts 2026",
        "shortName": "Fed",
        "url": "u",
        "totalSharesTraded": 5_000,
        "contracts": [{"id": 21, "name": "Yes", "lastTradePrice": 0.78}],
    },
    {
        "id": 7500,
        "name": "CPI > 3.5%",
        "shortName": "CPI",
        "url": "u",
        "totalSharesTraded": 5_000,
        "contracts": [{"id": 31, "name": "Yes", "lastTradePrice": 0.22}],
    },
    {
        "id": 7600,
        "name": "Senate 2026",
        "shortName": "Senate",
        "url": "u",
        "totalSharesTraded": 5_000,
        "contracts": [{"id": 41, "name": "Rep", "lastTradePrice": 0.55}],
    },
]


@respx.mock
def test_predictit_all_markets_caches_300s() -> None:
    """Second call within TTL -> served from cache, no second HTTP hit."""
    route = respx.get(f"{PREDICTIT}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"markets": SAMPLE_PREDICTIT})
    )

    async def go() -> int:
        c = PredictItClient(client=httpx.AsyncClient())
        try:
            await c.fetch_all_markets()
            await c.fetch_all_markets()
            return route.call_count
        finally:
            await c.close()

    assert _run(go()) == 1


@respx.mock
def test_predictit_force_refresh_bypasses_cache() -> None:
    route = respx.get(f"{PREDICTIT}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"markets": SAMPLE_PREDICTIT})
    )

    async def go() -> int:
        c = PredictItClient(client=httpx.AsyncClient())
        try:
            await c.fetch_all_markets()
            await c.fetch_all_markets(force_refresh=True)
            return route.call_count
        finally:
            await c.close()

    assert _run(go()) == 2


@respx.mock
def test_predictit_fetch_market_uses_warm_cache() -> None:
    """When all-markets cache is warm, fetch_market should NOT hit per-market endpoint."""
    respx.get(f"{PREDICTIT}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"markets": SAMPLE_PREDICTIT})
    )
    # No route for /marketdata/markets/ -- if it gets called, respx will assert.

    async def go() -> dict[str, Any]:
        c = PredictItClient(client=httpx.AsyncClient())
        try:
            await c.fetch_all_markets()
            return await c.fetch_market(7400)
        finally:
            await c.close()

    m = _run(go())
    assert m["id"] == 7400


@respx.mock
def test_predictit_fetch_market_falls_back_when_cache_cold() -> None:
    respx.get(f"{PREDICTIT}/marketdata/markets/7300").mock(
        return_value=httpx.Response(200, json=SAMPLE_PREDICTIT[1])
    )

    async def go() -> dict[str, Any]:
        c = PredictItClient(client=httpx.AsyncClient())
        try:
            return await c.fetch_market(7300)
        finally:
            await c.close()

    m = _run(go())
    assert m["id"] == 7300


@respx.mock
def test_predictit_history_uses_leading_contract() -> None:
    respx.get(f"{PREDICTIT}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"markets": SAMPLE_PREDICTIT})
    )

    async def go() -> pd.DataFrame:
        c = PredictItClient(client=httpx.AsyncClient())
        try:
            await c.fetch_all_markets()
            return await c.fetch_history(7456, days=30)
        finally:
            await c.close()

    df = _run(go())
    # Leading contract is Trump @ 0.42 (max lastTradePrice).
    assert df.iloc[0]["prob"] == pytest.approx(0.42)
    assert df.iloc[0]["contract_id"] == 1


@respx.mock
def test_predictit_missing_markets_key_raises() -> None:
    respx.get(f"{PREDICTIT}/marketdata/all/").mock(
        return_value=httpx.Response(200, json={"oops": []})
    )

    async def go() -> Any:
        c = PredictItClient(client=httpx.AsyncClient())
        try:
            return await c.fetch_all_markets()
        finally:
            await c.close()

    with pytest.raises(PredictItError, match="markets"):
        _run(go())


# ---------------------------------------------------------------------------
# 5. FRED extended
# ---------------------------------------------------------------------------


def _fred_csv(series_id: str, start: str, end: str, base: float = 100.0) -> str:
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    lines = [f"DATE,{series_id}"]
    for i, ts in enumerate(idx):
        lines.append(f"{ts.strftime('%Y-%m-%d')},{base + 0.05 * i:.4f}")
    return "\n".join(lines) + "\n"


def test_fred_registry_has_all_20_series_supported() -> None:
    expected = [
        "DFF",
        "DGS2",
        "DGS10",
        "CPIAUCSL",
        "UNRATE",
        "VIXCLS",
        "ICSA",
        "CCSA",
        "PAYEMS",
        "MANEMP",
        "PERMIT",
        "HOUST",
        "RSXFS",
        "INDPRO",
        "T10Y2Y",
        "BAMLH0A0HYM2",
        "DCOILWTICO",
        "GOLDAMGBD228NLBM",
        "DEXUSEU",
        "DEXJPUS",
        # vol-benchmark additions (A2):
        "OVXCLS",
        "GVZCLS",
    ]
    assert len(_SERIES_REGISTRY) == 22
    for sid in expected:
        assert sid in _SERIES_REGISTRY, f"missing FRED series {sid}"


@respx.mock
def test_fred_each_series_parsed_to_dataframe() -> None:
    """Mock fredgraph for every series, verify each parses."""

    def _handler(request: httpx.Request) -> httpx.Response:
        sid = request.url.params["id"]
        return httpx.Response(200, text=_fred_csv(sid, "2024-01-01", "2024-01-10"))

    respx.get(FREDGRAPH_BASE).mock(side_effect=_handler)
    for sid in _SERIES_REGISTRY:
        s = fetch_fred_series(
            sid,
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-10", tz="UTC"),
        )
        assert not s.empty
        assert s.name == sid


@respx.mock
def test_fred_catalog_endpoint_returns_20_with_metadata() -> None:
    app = FastAPI()
    app.include_router(fred_router)
    with TestClient(app) as client:
        r = client.get("/macro/fred/catalog")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 22
    assert len(body["series"]) == 22
    # Every entry has the expected metadata keys.
    for entry in body["series"]:
        assert {"series_id", "name", "frequency", "units", "citation"} <= entry.keys()


@respx.mock
def test_fred_series_endpoint_returns_jsonseries() -> None:
    respx.get(FREDGRAPH_BASE).mock(
        return_value=httpx.Response(200, text=_fred_csv("DFF", "2024-01-01", "2024-01-05"))
    )
    app = FastAPI()
    app.include_router(fred_router)
    with TestClient(app) as client:
        r = client.get("/macro/fred/series/DFF?start=2024-01-01&end=2024-01-05")
    assert r.status_code == 200
    body = r.json()
    assert body["series_id"] == "DFF"
    assert body["frequency"] == "daily"
    assert len(body["data"]) >= 1


@respx.mock
def test_fred_series_unknown_404() -> None:
    app = FastAPI()
    app.include_router(fred_router)
    with TestClient(app) as client:
        r = client.get("/macro/fred/series/NOPE?start=2024-01-01&end=2024-01-05")
    assert r.status_code == 404


@respx.mock
def test_fred_cached_helper_round_trip() -> None:
    route = respx.get(FREDGRAPH_BASE).mock(
        return_value=httpx.Response(200, text=_fred_csv("DFF", "2024-01-01", "2024-01-03"))
    )
    s = fetch_fred_series_cached(
        "DFF",
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03", tz="UTC"),
    )
    s2 = fetch_fred_series_cached(
        "DFF",
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03", tz="UTC"),
    )
    assert not s.empty and not s2.empty
    # Cache hit on second call.
    assert route.call_count == 1


@respx.mock
def test_fred_5xx_retries_and_eventually_raises() -> None:
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(503, text="down"))
    with pytest.raises(FredDataError):
        fetch_fred_series(
            "DFF",
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-05", tz="UTC"),
            max_retries=2,
        )


@respx.mock
def test_fred_forward_fill_aligns_to_daily_calendar() -> None:
    """Sparse weekly series should be resampled to a daily UTC index via ffill."""
    # Two weekly observations for ICSA (Thursdays).
    text = "DATE,ICSA\n2024-01-04,200000\n2024-01-11,210000\n"
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=text))
    s = fetch_fred_series(
        "ICSA",
        pd.Timestamp("2024-01-04", tz="UTC"),
        pd.Timestamp("2024-01-15", tz="UTC"),
    )
    # Daily index, no gaps.
    assert s.index.freq is not None or len(s) == 12  # 2024-01-04 -> 01-15 inclusive
    # Sunday Jan 7 should ffill to Thursday Jan 4's value (200000).
    sun = pd.Timestamp("2024-01-07", tz="UTC")
    assert sun in s.index
    assert s.loc[sun] == 200_000


# ---------------------------------------------------------------------------
# 6. BLS source
# ---------------------------------------------------------------------------


def _bls_payload(series_id: str, rows: list[tuple[int, str, float]]) -> dict[str, Any]:
    """Build a successful BLS API response."""
    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 100,
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": series_id,
                    "data": [
                        {
                            "year": str(y),
                            "period": p,
                            "periodName": "January",
                            "value": str(v),
                            "footnotes": [],
                        }
                        for (y, p, v) in rows
                    ],
                }
            ],
        },
    }


@respx.mock
def test_bls_each_curated_series_parses() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sid = body["seriesid"][0]
        rows = [(2024, "M01", 4.0), (2024, "M02", 4.1), (2024, "M03", 4.2)]
        return httpx.Response(200, json=_bls_payload(sid, rows))

    respx.post(BLS_BASE).mock(side_effect=_handler)
    for sid in _BLS_SERIES_REGISTRY:
        df = fetch_bls_series(sid, 2024, 2024, api_key=None)
        assert not df.empty
        assert list(df.columns) == ["date", "value"]


@respx.mock
def test_bls_with_api_key_passes_registration_key() -> None:
    captured = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_bls_payload("LNS14000000", [(2024, "M01", 3.7)]))

    respx.post(BLS_BASE).mock(side_effect=_handler)
    fetch_bls_series("LNS14000000", 2024, 2024, api_key="secret-key")
    assert captured["body"]["registrationkey"] == "secret-key"


@respx.mock
def test_bls_without_api_key_no_registration_key_in_body() -> None:
    captured = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_bls_payload("LNS14000000", [(2024, "M01", 3.7)]))

    respx.post(BLS_BASE).mock(side_effect=_handler)
    fetch_bls_series("LNS14000000", 2024, 2024, api_key=None)
    assert "registrationkey" not in captured["body"]


@respx.mock
def test_bls_year_range_passed_correctly() -> None:
    captured = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_bls_payload("LNS14000000", [(2020, "M01", 3.5)]))

    respx.post(BLS_BASE).mock(side_effect=_handler)
    fetch_bls_series("LNS14000000", 2020, 2024)
    assert captured["body"]["startyear"] == "2020"
    assert captured["body"]["endyear"] == "2024"


@respx.mock
def test_bls_failure_status_raises() -> None:
    respx.post(BLS_BASE).mock(
        return_value=httpx.Response(
            200,
            json={"status": "REQUEST_NOT_PROCESSED", "message": ["bad year"], "Results": {}},
        )
    )
    with pytest.raises(BlsDataError, match="REQUEST_NOT_PROCESSED"):
        fetch_bls_series("LNS14000000", 2024, 2024)


@respx.mock
def test_bls_5xx_retries_and_raises() -> None:
    respx.post(BLS_BASE).mock(return_value=httpx.Response(503, text="down"))
    with pytest.raises(BlsDataError):
        fetch_bls_series("LNS14000000", 2024, 2024, max_retries=2)


@respx.mock
def test_bls_catalog_endpoint() -> None:
    app = FastAPI()
    app.include_router(bls_router)
    with TestClient(app) as client:
        r = client.get("/macro/bls/catalog")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 5
    sids = [s["series_id"] for s in body["series"]]
    assert set(sids) == set(_BLS_SERIES_REGISTRY)


@respx.mock
def test_bls_caches_within_namespace() -> None:
    route = respx.post(BLS_BASE).mock(
        return_value=httpx.Response(200, json=_bls_payload("LNS14000000", [(2024, "M01", 3.7)]))
    )
    cli = BLSClient(api_key=None)
    cli.fetch("LNS14000000", 2024, 2024)
    cli.fetch("LNS14000000", 2024, 2024)
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# 7. Equity sources cascade
# ---------------------------------------------------------------------------


def _make_yf_df(index: pd.DatetimeIndex, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": prices}, index=index)


def test_equity_yfinance_success_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When yfinance succeeds, downstream sources are not consulted."""
    idx = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
    df = _make_yf_df(idx, [100.0 + i for i in range(len(idx))])

    yf_calls = {"n": 0}

    def fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        yf_calls["n"] += 1
        return df

    monkeypatch.setattr("yfinance.download", fake_download)

    # Tiingo / Stooq must not be called.
    def explode(*a: Any, **k: Any) -> Any:
        raise AssertionError("fallback should not be hit")

    monkeypatch.setattr(tiingo_src, "fetch_daily_prices", explode)
    monkeypatch.setattr(stooq_src, "fetch_daily_prices", explode)

    out = get_log_returns(
        "AAPL",
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2024-01-10", tz="UTC"),
    )
    assert not out.empty
    assert out.name == "r"
    assert yf_calls["n"] == 1


def test_equity_yfinance_fail_falls_through_to_tiingo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.DataFrame())
    # Suppress the "delisted" probe so we DO fall through.
    monkeypatch.setattr(equity_mod, "_check_delisted_via_yf_info", lambda t: False)
    monkeypatch.setenv("TIINGO_API_KEY", "fake-token")

    idx = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
    tiingo_df = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjClose": [100.0 + i for i in range(len(idx))],
            "volume": 1.0,
        },
        index=idx,
    )

    def fake_tiingo(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return tiingo_df

    monkeypatch.setattr(tiingo_src, "fetch_daily_prices", fake_tiingo)
    out = get_log_returns(
        "AAPL",
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2024-01-10", tz="UTC"),
    )
    assert not out.empty


def test_equity_no_tiingo_key_skips_to_stooq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(equity_mod, "_check_delisted_via_yf_info", lambda t: False)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)

    idx = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
    stooq_df = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": [100.0 + i for i in range(len(idx))],
            "volume": 1.0,
        },
        index=idx,
    )
    monkeypatch.setattr(stooq_src, "fetch_daily_prices", lambda *a, **k: stooq_df)

    out = get_log_returns(
        "AAPL",
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2024-01-10", tz="UTC"),
    )
    assert not out.empty


def test_equity_all_sources_fail_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(equity_mod, "_check_delisted_via_yf_info", lambda t: False)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)

    def fail(*a: Any, **k: Any) -> Any:
        raise stooq_src.StooqError("simulated stooq failure")

    monkeypatch.setattr(stooq_src, "fetch_daily_prices", fail)

    with pytest.raises(EquityDataError) as ei:
        get_log_returns(
            "ZZZZ",
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-10", tz="UTC"),
        )
    msg = str(ei.value)
    assert "yfinance" in msg or "stooq" in msg


def test_equity_delisted_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A delisted ticker should fail fast without any source attempts."""
    # Mark first via the registry path patched in our autouse fixture.
    equity_mod.mark_delisted("DEADCO")

    yf_called = {"v": False}

    def yf_explode(*a: Any, **k: Any) -> Any:
        yf_called["v"] = True
        return pd.DataFrame()

    monkeypatch.setattr("yfinance.download", yf_explode)
    with pytest.raises(EquityDelistedError):
        get_log_returns(
            "DEADCO",
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-10", tz="UTC"),
        )
    assert yf_called["v"] is False  # short-circuited before yfinance


def test_equity_delisted_detected_via_yf_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty yfinance + regularMarketPrice=None -> EquityDelistedError, no fallback."""
    monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(equity_mod, "_check_delisted_via_yf_info", lambda ticker: True)

    # Tiingo / Stooq must NOT be called once delisted is detected.
    def explode(*a: Any, **k: Any) -> Any:
        raise AssertionError("fallback should not be hit on delisted")

    monkeypatch.setattr(tiingo_src, "fetch_daily_prices", explode)
    monkeypatch.setattr(stooq_src, "fetch_daily_prices", explode)

    with pytest.raises(EquityDelistedError):
        get_log_returns(
            "DELISTEDX",
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-10", tz="UTC"),
        )
    # And the registry now contains DELISTEDX.
    assert "DELISTEDX" in list_delisted()
    assert is_delisted("DELISTEDX")


@respx.mock
def test_stooq_csv_parser_format() -> None:
    """Stooq CSV format: Date,Open,High,Low,Close,Volume."""
    csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100.0,101.0,99.0,100.5,1000\n"
        "2024-01-03,100.5,102.0,100.0,101.0,1500\n"
    )
    respx.get(stooq_src.STOOQ_BASE).mock(return_value=httpx.Response(200, text=csv))
    df = stooq_src.fetch_daily_prices(
        "AAPL",
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2024-01-03", tz="UTC"),
    )
    assert "close" in df.columns
    assert len(df) == 2


@respx.mock
def test_stooq_no_data_raises() -> None:
    respx.get(stooq_src.STOOQ_BASE).mock(return_value=httpx.Response(200, text="No data"))
    with pytest.raises(stooq_src.StooqError):
        stooq_src.fetch_daily_prices(
            "ZZZZ",
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-03", tz="UTC"),
        )


@respx.mock
def test_tiingo_requires_api_key() -> None:
    with pytest.raises(tiingo_src.TiingoError, match="api_key"):
        tiingo_src.fetch_daily_prices(
            "AAPL",
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-03", tz="UTC"),
            api_key="",
        )


@respx.mock
def test_tiingo_authorization_header_set() -> None:
    captured = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json=[
                {
                    "date": "2024-01-02T00:00:00.000Z",
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "adjClose": 100,
                    "volume": 1000,
                },
                {
                    "date": "2024-01-03T00:00:00.000Z",
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 101,
                    "adjClose": 101,
                    "volume": 1000,
                },
            ],
        )

    respx.get(url__regex=r"https://api\.tiingo\.com/.*").mock(side_effect=_handler)
    df = tiingo_src.fetch_daily_prices(
        "AAPL",
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2024-01-03", tz="UTC"),
        api_key="my-token",
    )
    assert captured["auth"] == "Token my-token"
    assert "adjClose" in df.columns


# ---------------------------------------------------------------------------
# 8. Sources health
# ---------------------------------------------------------------------------


@respx.mock
def test_sources_health_all_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "fake")
    respx.get(url__regex=r"https://query1\.finance\.yahoo\.com/.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(url__regex=r"https://api\.tiingo\.com/.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__regex=r"https://stooq\.com/.*").mock(
        return_value=httpx.Response(200, text="Date,Close\n2025-01-02,100.0\n")
    )
    respx.get(url__regex=r"https://gamma-api\.polymarket\.com/.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__regex=r"https://api\.elections\.kalshi\.com/.*").mock(
        return_value=httpx.Response(200, json={"events": []})
    )
    respx.get(url__regex=r"https://fred\.stlouisfed\.org/.*").mock(
        return_value=httpx.Response(200, text="DATE,DFF\n2024-01-02,5.32\n")
    )

    out = check_all_sources()
    assert set(out) == {"yfinance", "tiingo", "stooq", "polymarket", "kalshi", "fred"}
    for name, payload in out.items():
        assert payload["ok"] is True, f"{name} not ok: {payload}"


def test_sources_health_tiingo_no_api_key_configured_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    out = check_tiingo()
    assert out["configured"] is False
    assert out["ok"] is False


@respx.mock
def test_sources_health_polymarket_timeout_marks_unhealthy() -> None:
    respx.get(url__regex=r"https://gamma-api\.polymarket\.com/.*").mock(
        side_effect=httpx.ReadTimeout("simulated")
    )
    out = check_polymarket()
    assert out["ok"] is False
    assert out["latency_ms"] is not None  # latency captured even on failure


@respx.mock
def test_sources_health_endpoint_total_response_under_30s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when configured, total wall-clock for /sources/health must stay <30s."""
    monkeypatch.setenv("TIINGO_API_KEY", "fake")
    for url in [
        r"https://query1\.finance\.yahoo\.com/.*",
        r"https://api\.tiingo\.com/.*",
        r"https://stooq\.com/.*",
        r"https://gamma-api\.polymarket\.com/.*",
        r"https://api\.elections\.kalshi\.com/.*",
        r"https://fred\.stlouisfed\.org/.*",
    ]:
        respx.get(url__regex=url).mock(return_value=httpx.Response(200, json={}))

    app = FastAPI()
    app.include_router(sources_router)
    t0 = time.monotonic()
    with TestClient(app) as client:
        r = client.get("/sources/health")
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert elapsed < 30.0
    body = r.json()
    assert body["summary"]["total"] == 6


@respx.mock
def test_sources_health_endpoint_summary_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """summary['up'] + summary['down'] + summary['not_configured'] == total."""
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)  # Tiingo not configured
    respx.get(url__regex=r"https://query1\.finance\.yahoo\.com/.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(url__regex=r"https://stooq\.com/.*").mock(
        return_value=httpx.Response(200, text="Date,Close\n2024-01-02,100\n")
    )
    respx.get(url__regex=r"https://gamma-api\.polymarket\.com/.*").mock(
        return_value=httpx.Response(503)
    )
    respx.get(url__regex=r"https://api\.elections\.kalshi\.com/.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(url__regex=r"https://fred\.stlouisfed\.org/.*").mock(
        return_value=httpx.Response(200, text="DATE,DFF\n2024-01-02,5\n")
    )

    app = FastAPI()
    app.include_router(sources_router)
    with TestClient(app) as client:
        body = client.get("/sources/health").json()
    s = body["summary"]
    assert s["up"] + s["down"] + s["not_configured"] == s["total"]
    assert s["not_configured"] >= 1  # Tiingo


# ---------------------------------------------------------------------------
# 9. Macro calendar
# ---------------------------------------------------------------------------


def test_macro_calendar_2026_fomc_dates_known() -> None:
    """Verify the hardcoded 2026 FOMC schedule matches the published Fed calendar."""
    from datetime import date

    expected = {
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    }
    assert set(_FOMC_2026) == expected


def test_macro_calendar_next_releases_window_filter() -> None:
    """days=7 -> only events within next 7 days."""
    from datetime import date

    today = date(2026, 6, 1)
    out = next_releases(7, today=today)
    for ev in out:
        d = date.fromisoformat(ev["date"])
        assert today <= d <= date(2026, 6, 8)


def test_macro_calendar_events_sorted_by_date() -> None:
    from datetime import date

    today = date(2026, 1, 1)
    out = next_releases(60, today=today)
    dates = [ev["date"] for ev in out]
    assert dates == sorted(dates)


def test_macro_calendar_endpoint_includes_fomc_cpi_nfp() -> None:
    app = FastAPI()
    app.include_router(macro_calendar_router)
    with TestClient(app) as client:
        r = client.get("/macro/upcoming?days=365")
    assert r.status_code == 200
    body = r.json()
    types = {ev["type"] for ev in body["events"]}
    # Even if "today" filters some out, across a 365-day window ALL types
    # should be represented at least once.
    assert {"fomc", "cpi", "nfp", "ppi", "retail_sales", "gdp"}.issubset(types)


def test_macro_calendar_caches_within_window(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    app.include_router(macro_calendar_router)
    cache = get_cache("macro-calendar")

    calls = {"n": 0}
    real_next = macal_mod.next_releases

    def counting(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        return real_next(*a, **k)

    monkeypatch.setattr(macal_mod, "next_releases", counting)
    cache.clear()
    with TestClient(app) as client:
        client.get("/macro/upcoming?days=30")
        client.get("/macro/upcoming?days=30")
    # Second hit served from cache.
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 10. Macro overlay unified
# ---------------------------------------------------------------------------


@respx.mock
def test_macro_overlay_multi_series() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        sid = request.url.params["id"]
        return httpx.Response(200, text=_fred_csv(sid, "2025-01-01", "2025-01-05"))

    respx.get(FREDGRAPH_BASE).mock(side_effect=_handler)
    app = FastAPI()
    app.include_router(macro_overlay_router)
    with TestClient(app) as client:
        r = client.get("/macro/overlay?series=DFF,DGS10,VIXCLS&start=2025-01-01&end=2025-01-05")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    ids = [s["id"] for s in body["series"]]
    assert ids == ["DFF", "DGS10", "VIXCLS"]


@respx.mock
def test_macro_overlay_unknown_series_404() -> None:
    app = FastAPI()
    app.include_router(macro_overlay_router)
    with TestClient(app) as client:
        r = client.get("/macro/overlay?series=NOPE&start=2025-01-01&end=2025-01-05")
    assert r.status_code == 404


def test_macro_overlay_bad_window_400() -> None:
    app = FastAPI()
    app.include_router(macro_overlay_router)
    with TestClient(app) as client:
        r = client.get("/macro/overlay?series=DFF&start=2025-12-31&end=2025-01-01")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 11. Multi-venue search
# ---------------------------------------------------------------------------


@respx.mock
def test_multi_venue_search_failure_isolation() -> None:
    """One venue 500 -> others still return."""
    # Polymarket 500
    respx.get(url__regex=r"https://gamma-api\.polymarket\.com/.*").mock(
        return_value=httpx.Response(500)
    )
    # Kalshi OK
    respx.get(url__regex=r"https://api\.elections\.kalshi\.com/.*").mock(
        return_value=httpx.Response(
            200,
            json={"markets": [{"ticker": "KX-FOO", "title": "Foo trump bar", "close_time": None}]},
        )
    )
    # Manifold OK
    respx.get(url__regex=r"https://api\.manifold\.markets/.*").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "m1", "slug": "mslug", "question": "trump?"},
            ],
        )
    )
    # PredictIt OK
    respx.get(url__regex=r"https://www\.predictit\.org/.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "id": 1,
                        "name": "Trump 2028",
                        "shortName": "Trump",
                        "url": "u",
                        "dateEnd": None,
                        "contracts": [],
                    },
                ]
            },
        )
    )

    out = _run(search_all_venues("trump", limit=5))
    # PM failed silently -> [].
    assert out["polymarket"] == []
    assert len(out["kalshi"]) >= 1
    assert len(out["manifold"]) >= 1
    assert len(out["predictit"]) >= 1


@respx.mock
def test_multi_venue_search_endpoint_caches_60s() -> None:
    respx.get(url__regex=r"https://gamma-api\.polymarket\.com/.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__regex=r"https://api\.elections\.kalshi\.com/.*").mock(
        return_value=httpx.Response(200, json={"markets": []})
    )
    respx.get(url__regex=r"https://api\.manifold\.markets/.*").mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(url__regex=r"https://www\.predictit\.org/.*").mock(
        return_value=httpx.Response(200, json={"markets": []})
    )

    app = FastAPI()
    app.include_router(multi_venue_router)
    with TestClient(app) as client:
        r1 = client.get("/multi-venue/search?q=test")
        r2 = client.get("/multi-venue/search?q=test")
    assert r1.status_code == 200 and r2.status_code == 200
    # Both succeed; cache hit on second call (we cannot easily count network
    # calls because respx is shared, but the responses must be identical).
    assert r1.json() == r2.json()


def test_multi_venue_concepts_lists_5() -> None:
    app = FastAPI()
    app.include_router(multi_venue_router)
    with TestClient(app) as client:
        r = client.get("/multi-venue/concepts")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 5
    ids = [c["concept_id"] for c in body["concepts"]]
    for expected in [
        "presidential_election_2028",
        "fed_cuts_2026",
        "recession_2026",
        "btc_ath_2026",
        "cpi_above_3_5_2026",
    ]:
        assert expected in ids


def test_multi_venue_concept_unknown_404() -> None:
    app = FastAPI()
    app.include_router(multi_venue_router)
    with TestClient(app) as client:
        r = client.get("/multi-venue/concept/does_not_exist")
    assert r.status_code == 404


def test_multi_venue_concept_known_returns_4_legs() -> None:
    app = FastAPI()
    app.include_router(multi_venue_router)
    with TestClient(app) as client:
        r = client.get("/multi-venue/concept/fed_cuts_2026")
    assert r.status_code == 200
    body = r.json()
    assert body["concept_id"] == "fed_cuts_2026"
    assert body["n_legs_present"] == 4  # all four venues listed for fed_cuts
    assert body["venues"]["polymarket"]
    assert body["venues"]["kalshi"]
    assert body["venues"]["manifold"]
    assert body["venues"]["predictit"]


# ---------------------------------------------------------------------------
# 12. Cross-venue arb scanner
# ---------------------------------------------------------------------------


def test_arb_scanner_endpoint_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """top_arbs is patched; the endpoint should serialise the result."""
    fake_arbs = [
        {
            "pm_slug": "us-recession-by-end-of-2026",
            "kalshi_slug": "KXRECSSNBER-26",
            "label": "Recession",
            "pm_price": 0.30,
            "kalshi_price": 0.35,
            "spread_pct": 5.0,
            "direction": "buy_pm_sell_kalshi",
            "tradeable_size_usd": 7500.0,
            "half_life_minutes": 30.0,
            "last_seen_iso": "2026-05-08T00:00:00+00:00",
            "confirmed": True,
            "confirmation_window_min": 10,
        }
    ]
    monkeypatch.setattr(arb_mod, "top_arbs", lambda **kw: fake_arbs)
    get_cache("arb_scanner").clear()
    app = FastAPI()
    app.include_router(arb_mod.router)
    with TestClient(app) as client:
        r = client.get("/arb/scanner?min_spread_pct=2.0&n=5")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 1
    assert body["arbs"][0]["spread_pct"] == 5.0


def test_arb_scanner_min_spread_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint passes min_spread_pct through to top_arbs."""
    captured: dict[str, Any] = {}

    def fake_top_arbs(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(arb_mod, "top_arbs", fake_top_arbs)
    get_cache("arb_scanner").clear()
    app = FastAPI()
    app.include_router(arb_mod.router)
    with TestClient(app) as client:
        client.get("/arb/scanner?min_spread_pct=3.5&n=10")
    assert captured["min_spread_pct"] == pytest.approx(3.5)
    assert captured["n"] == 10


def test_arb_concepts_endpoint_lists_5() -> None:
    app = FastAPI()
    app.include_router(arb_mod.router)
    with TestClient(app) as client:
        r = client.get("/arb/concepts")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 5


def test_arb_pre_matched_pairs_constants() -> None:
    """5 hardcoded pairs covering recession, fed, BTC, CPI."""
    assert len(arb_mod.PRE_MATCHED_PAIRS) == 5
    pm_slugs = {p["pm_slug"] for p in arb_mod.PRE_MATCHED_PAIRS}
    assert "us-recession-by-end-of-2026" in pm_slugs


def test_arb_spread_record_filters_below_threshold() -> None:
    """compute_arb_spreads honours the min_spread_pct gate via _spread_record."""
    rec = arb_mod._spread_record(
        {"pm_slug": "a", "kalshi_slug": "b", "label": "L"},
        pm_price=0.50,
        kalshi_price=0.51,
        pm_vol=10000.0,
        kalshi_vol=10000.0,
        min_spread_pct=2.0,
        min_volume_usd=1000.0,
    )
    assert rec is None  # 1pp spread < 2% threshold


def test_arb_spread_record_uses_min_volume_for_size() -> None:
    """tradeable_size_usd = min(pm_vol, kalshi_vol)."""
    rec = arb_mod._spread_record(
        {"pm_slug": "a", "kalshi_slug": "b", "label": "L"},
        pm_price=0.30,
        kalshi_price=0.40,
        pm_vol=8000.0,
        kalshi_vol=15000.0,
        min_spread_pct=2.0,
        min_volume_usd=1000.0,
    )
    assert rec is not None
    assert rec["tradeable_size_usd"] == 8000.0
    assert rec["spread_pct"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------


@respx.mock
def test_pm_malformed_json_propagates_as_value_error(
    pm_client: PolymarketClient,
) -> None:
    """A non-JSON response should bubble out as a parsing error (not silent)."""
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(200, content=b"not-json"))
    # Either ValueError or json.JSONDecodeError -- both are acceptable as long
    # as the malformed payload doesn't silently succeed.
    with pytest.raises((ValueError, json.JSONDecodeError)):
        pm_client.get_price_history("tok")


@respx.mock
def test_fred_429_retries() -> None:
    """fetch_fred_series should retry on 429 then succeed."""
    respx.get(FREDGRAPH_BASE).mock(
        side_effect=[
            httpx.Response(429, text="slow down"),
            httpx.Response(200, text=_fred_csv("DFF", "2024-01-01", "2024-01-03")),
        ]
    )
    s = fetch_fred_series(
        "DFF",
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03", tz="UTC"),
        max_retries=3,
    )
    assert not s.empty


def test_predictit_market_id_none_raises() -> None:
    async def go() -> Any:
        c = PredictItClient(client=httpx.AsyncClient())
        try:
            return await c.fetch_market(None)  # type: ignore[arg-type]
        finally:
            await c.close()

    with pytest.raises(PredictItError):
        _run(go())
