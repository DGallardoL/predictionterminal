"""End-to-end tests for the ``/strategies/crypto/5min/*`` router.

These run against the real FastAPI app via TestClient, with all upstream
HTTP calls (Binance REST, Polymarket Gamma + CLOB) mocked through respx.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from pfm.crypto5min.market_fetcher import DEFAULT_CLOB_URL, DEFAULT_GAMMA_URL
from pfm.crypto5min.router import _reset_caches
from pfm.crypto5min.state import get_state, reset_state


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    reset_state()
    _reset_caches()
    yield
    reset_state()
    _reset_caches()


def _binance_ticker(symbol: str, price: float, spread: float = 5.0) -> httpx.Response:
    bid = price - spread / 2
    ask = price + spread / 2
    return httpx.Response(
        200,
        json={
            "symbol": symbol,
            "bidPrice": str(bid),
            "askPrice": str(ask),
            "bidQty": "1.0",
            "askQty": "1.0",
        },
    )


def _binance_klines(price_close: float, n: int = 30) -> httpx.Response:
    rows = []
    base_ts = 1_700_000_000_000
    for i in range(n):
        rows.append(
            [
                base_ts + i * 86_400_000,  # open time ms
                str(price_close + i * 10),
                str(price_close + 50),
                str(price_close - 30),
                str(price_close + i * 12),  # close
                "100.0",
                base_ts + (i + 1) * 86_400_000 - 1,
                "1000.0",
                1000,
                "50.0",
                "500.0",
                "0",
            ]
        )
    return httpx.Response(200, json=rows)


def _gamma_payload(slug: str) -> dict:
    return {
        "slug": slug,
        "id": "987",
        "closed": False,
        "active": True,
        "clobTokenIds": json.dumps(["tok_up", "tok_down"]),
    }


# ---------------------------------------------------------------------------
# /predict/{symbol}
# ---------------------------------------------------------------------------


def test_predict_endpoint_rejects_unsupported_symbol(app_client: TestClient) -> None:
    r = app_client.get("/strategies/crypto/5min/predict/DOGEUSDT")
    assert r.status_code == 400
    assert "unsupported" in r.json()["detail"].lower()


def test_predict_endpoint_returns_503_with_no_samples(app_client: TestClient) -> None:
    """When there are no spot samples AND Binance is mocked away → 503."""
    with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=httpx.Response(503)
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(return_value=httpx.Response(503))
        r = app_client.get("/strategies/crypto/5min/predict/BTCUSDT?window_minutes=5")
    assert r.status_code == 503


def test_predict_endpoint_works_with_binance_only(app_client: TestClient) -> None:
    with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker("BTCUSDT", 60_000.0)
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines(60_000.0)
        )
        r = app_client.get("/strategies/crypto/5min/predict/BTCUSDT?window_minutes=5")
    assert r.status_code == 200
    payload = r.json()
    assert payload["binance_symbol"] == "BTCUSDT"
    assert payload["window_minutes"] == 5
    assert "prediction" in payload
    assert 0.0 <= payload["prediction"]["prob_up"] <= 1.0


def test_predict_endpoint_accepts_btc_without_usdt_suffix(app_client: TestClient) -> None:
    with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker("BTCUSDT", 60_000.0)
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines(60_000.0)
        )
        r = app_client.get("/strategies/crypto/5min/predict/btc?window_minutes=15")
    assert r.status_code == 200
    assert r.json()["binance_symbol"] == "BTCUSDT"


def test_predict_endpoint_rejects_zero_window(app_client: TestClient) -> None:
    r = app_client.get("/strategies/crypto/5min/predict/BTCUSDT?window_minutes=0")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /markets
# ---------------------------------------------------------------------------


def test_markets_endpoint_no_open_markets(app_client: TestClient) -> None:
    """When Polymarket returns nothing for every probe → n_markets=0."""
    with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=[]))
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker("BTCUSDT", 60_000.0)
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines(60_000.0)
        )
        r = app_client.get("/strategies/crypto/5min/markets?use_cache=false")
    assert r.status_code == 200
    payload = r.json()
    assert payload["n_markets"] == 0
    assert payload["markets"] == []


def test_markets_endpoint_returns_comparison_when_polymarket_active(
    app_client: TestClient,
) -> None:
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"

    def gamma_handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_gamma_payload(slug)])
        return httpx.Response(200, json=[])

    with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma_handler)
        respx.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.51"})
        )
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker("BTCUSDT", 60_000.0)
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines(60_000.0)
        )
        r = app_client.get(
            "/strategies/crypto/5min/markets"
            "?assets=BTC&window_minutes_csv=5&use_cache=false&edge_threshold=0.05"
        )
    assert r.status_code == 200
    body = r.json()
    assert body["n_markets"] == 1
    market = body["markets"][0]
    assert market["asset"] == "BTC"
    assert market["window_minutes"] == 5
    assert market["slug"] == slug
    assert "model_prob_up" in market
    assert "market_prob_up" in market
    assert market["market_prob_up"] == pytest.approx(0.51)
    assert "signal" in market
    assert market["signal"] in {"BUY_YES", "BUY_NO", "WAIT"}


def test_markets_endpoint_cached_response(app_client: TestClient) -> None:
    """A second call within TTL must read from cache (mark ``from_cache=True``)."""
    with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=[]))
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker("BTCUSDT", 60_000.0)
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines(60_000.0)
        )
        first = app_client.get("/strategies/crypto/5min/markets?use_cache=true")
        second = app_client.get("/strategies/crypto/5min/markets?use_cache=true")
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["from_cache"] is False
    assert second.json()["from_cache"] is True


def test_markets_endpoint_use_cache_false_bypasses_cache(app_client: TestClient) -> None:
    with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=[]))
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker("BTCUSDT", 60_000.0)
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines(60_000.0)
        )
        app_client.get("/strategies/crypto/5min/markets?use_cache=false")
        second = app_client.get("/strategies/crypto/5min/markets?use_cache=false")
    assert second.json()["from_cache"] is False


def test_markets_endpoint_rejects_bad_window_csv(app_client: TestClient) -> None:
    r = app_client.get("/strategies/crypto/5min/markets?window_minutes_csv=oops")
    assert r.status_code == 400


def test_markets_endpoint_rejects_edge_threshold_out_of_range(app_client: TestClient) -> None:
    r = app_client.get("/strategies/crypto/5min/markets?edge_threshold=2.0")
    assert r.status_code == 422


def test_markets_endpoint_handles_missing_clob_mid(app_client: TestClient) -> None:
    """Polymarket open but CLOB midpoint not reachable → error tag, not 500."""
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"

    def gamma_handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_gamma_payload(slug)])
        return httpx.Response(200, json=[])

    with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma_handler)
        respx.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(return_value=httpx.Response(503))
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=_binance_ticker("BTCUSDT", 60_000.0)
        )
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=_binance_klines(60_000.0)
        )
        r = app_client.get(
            "/strategies/crypto/5min/markets?assets=BTC&window_minutes_csv=5&use_cache=false"
        )
    assert r.status_code == 200
    body = r.json()
    assert body["n_markets"] == 1
    assert body["markets"][0].get("error") == "no_market_midpoint"
    assert body["markets"][0]["market_prob_up"] is None
    assert body["markets"][0]["model_prob_up"] is not None


# ---------------------------------------------------------------------------
# /diag
# ---------------------------------------------------------------------------


def test_diag_endpoint_returns_empty_initially(app_client: TestClient) -> None:
    r = app_client.get("/strategies/crypto/5min/diag")
    assert r.status_code == 200
    payload = r.json()
    assert payload["symbols"] == []
    assert payload["now_unix"] > 0


def test_diag_endpoint_lists_recorded_symbols(app_client: TestClient) -> None:
    state = get_state()
    state.record_spot("BTCUSDT", time.time(), 60_000.0)
    state.record_spot("ETHUSDT", time.time(), 3000.0)
    r = app_client.get("/strategies/crypto/5min/diag")
    assert r.status_code == 200
    syms = {row["symbol"] for row in r.json()["symbols"]}
    assert {"BTCUSDT", "ETHUSDT"}.issubset(syms)


# ---------------------------------------------------------------------------
# Module wiring sanity
# ---------------------------------------------------------------------------


def test_router_paths_present_in_openapi(app_client: TestClient) -> None:
    r = app_client.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json()["paths"].keys())
    expected = {
        "/strategies/crypto/5min/predict/{symbol}",
        "/strategies/crypto/5min/markets",
        "/strategies/crypto/5min/diag",
    }
    assert expected.issubset(paths)
