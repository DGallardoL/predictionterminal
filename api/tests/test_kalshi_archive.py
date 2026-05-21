"""Tests for the Kalshi settled-markets archive.

All HTTP calls are mocked with ``respx``. The module reuses the
sync :class:`pfm.sources.kalshi.KalshiClient` for candlestick fetches,
so we mock both ``/markets`` and ``/series/.../candlesticks``.
"""

from __future__ import annotations

import asyncio
from datetime import date

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.archive import kalshi_archive as ka
from pfm.archive.kalshi_archive import (
    KALSHI_BASE_URL,
    fetch_archive_kalshi_detail,
    fetch_settled_markets,
    kalshi_archive_series_distribution,
)
from pfm.archive.kalshi_router import router as kalshi_router
from pfm.cache_utils import get_cache
from pfm.sources.kalshi import KalshiClient


@pytest.fixture(autouse=True)
def _clear_archive_cache():
    """Wipe the archive cache around every test."""
    get_cache(ka.ARCHIVE_CACHE_NS).clear()
    yield
    get_cache(ka.ARCHIVE_CACHE_NS).clear()


# ───────────────────────────── settled markets ─────────────────────────────


def _market_row(ticker: str, *, series: str | None = None, **extra) -> dict:
    return {
        "ticker": ticker,
        "event_ticker": series or ticker.split("-", 1)[0],
        "title": extra.get("title", f"Will {ticker}?"),
        "settle_time": extra.get("settle_time", "2024-11-06T00:00:00Z"),
        "close_time": extra.get("close_time", "2024-11-06T00:00:00Z"),
        "result": extra.get("result", "yes"),
        "open_interest": extra.get("open_interest", 1000),
        "volume": extra.get("volume", 5000),
        "last_price": extra.get("last_price", 78),
    }


@respx.mock
def test_fetch_settled_markets_basic_paginates_and_normalizes() -> None:
    page1 = {
        "markets": [
            _market_row("KXFEDDECISION-24SEP-C50", series="KXFEDDECISION"),
            _market_row("KXFEDDECISION-24NOV-C25", series="KXFEDDECISION", result="no"),
        ],
        "cursor": "page2",
    }
    page2 = {
        "markets": [_market_row("PRES-2024-DJT", series="PRES")],
        "cursor": "",
    }

    route = respx.get(f"{KALSHI_BASE_URL}/markets").mock(
        side_effect=[
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]
    )

    rows = asyncio.run(fetch_settled_markets(limit=10))

    assert route.call_count == 2
    assert [r["ticker"] for r in rows] == [
        "KXFEDDECISION-24SEP-C50",
        "KXFEDDECISION-24NOV-C25",
        "PRES-2024-DJT",
    ]
    fed_yes = rows[0]
    assert fed_yes["series"] == "KXFEDDECISION"
    assert fed_yes["settle_value"] == "YES"
    assert fed_yes["total_volume"] == 5000.0
    assert fed_yes["last_trade_price"] == 0.78  # 78c → $0.78
    assert rows[1]["settle_value"] == "NO"


@respx.mock
def test_fetch_settled_markets_filters_by_series() -> None:
    """series_ticker is forwarded as a query param AND post-filtered."""
    captured: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "markets": [
                    _market_row("KXCPI-24DEC-Y3.0", series="KXCPI"),
                    # An off-series row that should be filtered out client-side.
                    _market_row("PRES-2024-DJT", series="PRES"),
                ],
                "cursor": "",
            },
        )

    respx.get(f"{KALSHI_BASE_URL}/markets").mock(side_effect=_handler)

    rows = asyncio.run(fetch_settled_markets(series_ticker="KXCPI", limit=10))

    assert len(rows) == 1
    assert rows[0]["ticker"] == "KXCPI-24DEC-Y3.0"
    assert captured[0]["series_ticker"] == "KXCPI"


@respx.mock
def test_fetch_settled_markets_respects_offset() -> None:
    rows_payload = {
        "markets": [_market_row(f"KX-2024-{i}", series="KX") for i in range(5)],
        "cursor": "",
    }
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(
        return_value=httpx.Response(200, json=rows_payload)
    )

    out = asyncio.run(fetch_settled_markets(limit=2, offset=2))
    assert [r["ticker"] for r in out] == ["KX-2024-2", "KX-2024-3"]


@respx.mock
def test_fetch_settled_markets_filters_by_date() -> None:
    payload = {
        "markets": [
            _market_row("A", series="S", settle_time="2024-01-15T00:00:00Z"),
            _market_row("B", series="S", settle_time="2024-06-15T00:00:00Z"),
            _market_row("C", series="S", settle_time="2024-12-15T00:00:00Z"),
        ],
        "cursor": "",
    }
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(return_value=httpx.Response(200, json=payload))

    out = asyncio.run(
        fetch_settled_markets(start_date=date(2024, 5, 1), end_date=date(2024, 11, 1), limit=10)
    )
    assert [r["ticker"] for r in out] == ["B"]


@respx.mock
def test_fetch_settled_markets_caches_repeated_calls() -> None:
    payload = {
        "markets": [_market_row("X", series="X")],
        "cursor": "",
    }
    route = respx.get(f"{KALSHI_BASE_URL}/markets").mock(
        return_value=httpx.Response(200, json=payload)
    )

    out1 = asyncio.run(fetch_settled_markets(limit=5))
    out2 = asyncio.run(fetch_settled_markets(limit=5))
    assert out1 == out2
    assert route.call_count == 1  # second call hit cache


# ───────────────────────── per-market detail ───────────────────────────────


def _candles_payload(prices: list[tuple[int, float, float]]) -> dict:
    """Build a Kalshi candlestick response.

    Each tuple is (end_period_unix_ts, close_dollars, volume).
    """
    candles = []
    for ts, close, vol in prices:
        candles.append(
            {
                "end_period_ts": ts,
                "price": {"close_dollars": close},
                "yes_bid": {"close_dollars": max(0.0, close - 0.01)},
                "yes_ask": {"close_dollars": min(1.0, close + 0.01)},
                "volume_fp": vol,
                "open_interest_fp": 100.0,
            }
        )
    return {"candlesticks": candles}


@respx.mock
def test_fetch_archive_kalshi_detail_computes_stats() -> None:
    ticker = "KXFEDDECISION-24SEP-C50"
    series = "KXFEDDECISION"

    respx.get(f"{KALSHI_BASE_URL}/markets/{ticker}").mock(
        return_value=httpx.Response(
            200,
            json={
                "market": {
                    "ticker": ticker,
                    "event_ticker": series + "-24SEP",
                    "title": "Fed cuts >=50bps in Sep 2024?",
                    "status": "settled",
                    "open_time": "2024-06-01T00:00:00Z",
                    "close_time": "2024-09-18T18:00:00Z",
                    "settle_time": "2024-09-18T18:00:00Z",
                    "result": "yes",
                    "n_traders": 432,
                    "top_wallets": ["0xaaa", "0xbbb"],
                    "volume": 999,
                }
            },
        )
    )

    # 5 days, peak on day 3 then decay → settle at $0.95.
    candle_prices = [
        (1717200000, 0.40, 200.0),  # Jun 1
        (1717286400, 0.55, 300.0),
        (1726617600, 0.90, 500.0),  # peak
        (1726704000, 0.85, 400.0),
        (1726790400, 0.95, 600.0),  # last day
    ]
    respx.get(f"{KALSHI_BASE_URL}/series/{series}/markets/{ticker}/candlesticks").mock(
        return_value=httpx.Response(200, json=_candles_payload(candle_prices))
    )

    # Use a deterministic Kalshi client (no rate limiting → tests are fast).
    http = httpx.Client()
    kc = KalshiClient(client=http, min_interval_s=0.0, max_retries=0)

    detail = fetch_archive_kalshi_detail(ticker, kalshi_client=kc, http_client=http)

    assert detail["ticker"] == ticker
    assert detail["series"] == series
    assert detail["settle_value"] == "YES"
    assert detail["settle_date"] == "2024-09-18"
    assert len(detail["history"]) == 5

    stats = detail["stats"]
    assert stats["peak_price"] == 0.95  # max over the 5 closes
    assert stats["trough_price"] == 0.40
    assert stats["n_days"] == 5
    assert stats["realized_vol"] is not None and stats["realized_vol"] > 0
    assert stats["n_traders"] == 432
    assert stats["top_wallets"] == ["0xaaa", "0xbbb"]
    # half_life_to_settle = settle_dt - peak_day, both are dates.
    assert isinstance(stats["half_life_to_settle"], float)


@respx.mock
def test_fetch_archive_kalshi_detail_handles_empty_history() -> None:
    ticker = "KXEMPTY-24DEC"
    respx.get(f"{KALSHI_BASE_URL}/markets/{ticker}").mock(
        return_value=httpx.Response(
            200,
            json={
                "market": {
                    "ticker": ticker,
                    "event_ticker": "KXEMPTY",
                    "title": "Empty",
                    "status": "settled",
                    "result": "no",
                    "settle_time": "2024-12-31T00:00:00Z",
                    "volume": 0,
                }
            },
        )
    )
    respx.get(f"{KALSHI_BASE_URL}/series/KXEMPTY/markets/{ticker}/candlesticks").mock(
        return_value=httpx.Response(200, json={"candlesticks": []})
    )

    http = httpx.Client()
    kc = KalshiClient(client=http, min_interval_s=0.0, max_retries=0)
    detail = fetch_archive_kalshi_detail(ticker, kalshi_client=kc, http_client=http)

    assert detail["history"] == []
    assert detail["stats"]["n_days"] == 0
    assert detail["stats"]["realized_vol"] is None
    assert detail["settle_value"] == "NO"


# ─────────────────────────── series distribution ───────────────────────────


@respx.mock
def test_kalshi_archive_series_distribution_groups_by_series() -> None:
    payload = {
        "markets": [
            _market_row("KXFED-A", series="KXFED", result="yes", volume=100),
            _market_row("KXFED-B", series="KXFED", result="no", volume=200),
            _market_row("KXCPI-A", series="KXCPI", result="yes", volume=400),
        ],
        "cursor": "",
    }
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(return_value=httpx.Response(200, json=payload))

    out = asyncio.run(kalshi_archive_series_distribution())
    series = out["series"]
    assert set(series.keys()) == {"KXFED", "KXCPI"}
    assert series["KXFED"]["n_markets"] == 2
    assert series["KXFED"]["pct_yes"] == 0.5
    assert series["KXFED"]["avg_volume"] == 150.0
    assert series["KXCPI"]["pct_yes"] == 1.0
    assert out["n_total_markets"] == 3
    assert out["n_series"] == 2


# ─────────────────────────────── router smoke ──────────────────────────────


@respx.mock
def test_router_markets_endpoint_returns_settled_payload() -> None:
    payload = {
        "markets": [_market_row("KX-1", series="KX")],
        "cursor": "",
    }
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(return_value=httpx.Response(200, json=payload))

    app = FastAPI()
    app.include_router(kalshi_router)
    client = TestClient(app)
    r = client.get("/archive/kalshi/markets", params={"limit": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 1
    assert body["items"][0]["ticker"] == "KX-1"
    assert body["items"][0]["series"] == "KX"


@respx.mock
def test_router_series_endpoint() -> None:
    payload = {
        "markets": [_market_row("KX-1", series="KX", result="yes", volume=10)],
        "cursor": "",
    }
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(return_value=httpx.Response(200, json=payload))

    app = FastAPI()
    app.include_router(kalshi_router)
    client = TestClient(app)
    r = client.get("/archive/kalshi/series")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_series"] == 1
    assert body["series"]["KX"]["pct_yes"] == 1.0


@respx.mock
def test_router_market_detail_csv_export() -> None:
    ticker = "KX-CSV-1"
    series = "KX"
    respx.get(f"{KALSHI_BASE_URL}/markets/{ticker}").mock(
        return_value=httpx.Response(
            200,
            json={
                "market": {
                    "ticker": ticker,
                    "event_ticker": series,
                    "title": "x",
                    "status": "settled",
                    "open_time": "2024-01-01T00:00:00Z",
                    "close_time": "2024-01-03T00:00:00Z",
                    "settle_time": "2024-01-03T00:00:00Z",
                    "result": "yes",
                    "volume": 10,
                }
            },
        )
    )
    respx.get(f"{KALSHI_BASE_URL}/series/{series}/markets/{ticker}/candlesticks").mock(
        return_value=httpx.Response(
            200,
            json=_candles_payload(
                [
                    (1704067200, 0.50, 10),
                    (1704153600, 0.60, 12),
                ]
            ),
        )
    )

    # Pre-empt the route's per-call client by using an injected one via cache:
    # the simplest path is to call the function once with our test client, then
    # the router will hit the cache.
    http = httpx.Client()
    kc = KalshiClient(client=http, min_interval_s=0.0, max_retries=0)
    fetch_archive_kalshi_detail(ticker, kalshi_client=kc, http_client=http)

    app = FastAPI()
    app.include_router(kalshi_router)
    client = TestClient(app)
    r = client.get(f"/archive/kalshi/market/{ticker}", params={"format": "csv"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    # Header row + 2 data rows.
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert lines[0].startswith("date,price,volume")
    assert len(lines) == 3


# ─────────────────────── series distribution single-flight ─────────────────


@respx.mock
def test_router_series_endpoint_wraps_upstream_5xx_as_502() -> None:
    """Transient Kalshi 5xx surfaces as 502, not a bare 500.

    Regression: the probe found ``/archive/kalshi/series`` returning a
    transient 500 on first call followed by a 200 retry. The handler
    now wraps upstream HTTP errors into a meaningful 502, mirroring
    ``get_archive_detail`` and matching the documented contract.
    """
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(
        return_value=httpx.Response(503, text="kalshi unavailable")
    )

    app = FastAPI()
    app.include_router(kalshi_router)
    client = TestClient(app)
    r = client.get("/archive/kalshi/series")
    assert r.status_code == 502, r.text
    assert "kalshi archive error" in r.json()["detail"]


def test_series_distribution_singleflight_dedupes_concurrent_callers() -> None:
    """Two concurrent first-callers share one upstream fetch (no double-init race).

    Regression: without the asyncio.Lock guarding ``cache_key``, two
    tasks racing on a cold cache each fire their own ``/markets``
    request and either may propagate a transient upstream blip as a
    server 500. The lock makes the second caller wait for the first to
    finish and read back from cache.
    """
    import asyncio as _asyncio

    call_count = {"n": 0}

    class _FakeAsyncClient:
        async def get(self, url, params=None):
            call_count["n"] += 1
            # Simulate a non-trivial upstream so the second caller can
            # race in while we're awaiting.
            await _asyncio.sleep(0.05)
            return httpx.Response(
                200,
                json={
                    "markets": [_market_row("KX-1", series="KX", result="yes", volume=10)],
                    "cursor": "",
                },
                request=httpx.Request("GET", "http://test"),
            )

        async def aclose(self) -> None:
            return None

    fake = _FakeAsyncClient()

    async def _both() -> tuple[dict, dict]:
        # Use a shared client so we can count calls without owning multiple.
        return await _asyncio.gather(
            ka.kalshi_archive_series_distribution(client=fake),
            ka.kalshi_archive_series_distribution(client=fake),
        )

    a, b = _asyncio.run(_both())
    # Both responses match and only ONE upstream call was made — proves
    # the second caller waited on the lock and read from cache.
    assert a == b
    assert call_count["n"] == 1
