"""Tests for ``/strategies/crypto/5min/compare``.

This endpoint is the load-bearing one for the UI: it always returns a row
per (asset, window) combo, with the *model* probability filled in even when
Polymarket has no open market for that combo, and the *market* probability
filled in when it does.
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
from pfm.crypto5min.state import reset_state


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
    """Synthetic daily klines with a realistic ~50%/yr σ.

    Daily σ ≈ 50% / √365 ≈ 2.6% per day. We use a deterministic sine wave
    of that amplitude so tests are reproducible. Without this the σ blend
    clips to the 10%/yr floor and the σ-jackknife produces SE=0.
    """
    rows = []
    base_ts = 1_700_000_000_000
    daily_amp = 0.026  # 2.6% per day → ~50%/yr when annualised
    prev = price_close
    for i in range(n):
        # Alternating up/down ±2.6% gives the right log-return std.
        sign = 1 if i % 2 == 0 else -1
        close = prev * (1.0 + sign * daily_amp)
        rows.append(
            [
                base_ts + i * 86_400_000,
                str(prev),
                str(max(prev, close) * 1.003),
                str(min(prev, close) * 0.997),
                str(close),
                "100.0",
                base_ts + (i + 1) * 86_400_000 - 1,
                "1000.0",
                1000,
                "50.0",
                "500.0",
                "0",
            ]
        )
        prev = close
    return httpx.Response(200, json=rows)


def _gamma_payload(slug: str) -> dict:
    return {
        "slug": slug,
        "id": "555",
        "closed": False,
        "active": True,
        "clobTokenIds": json.dumps(["tok_up", "tok_down"]),
    }


def _mock_no_polymarket(respx_mock) -> None:
    respx_mock.get(f"{DEFAULT_GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=[]))


def _mock_binance(respx_mock, *, btc_price: float = 60_000.0, eth_price: float = 3_000.0) -> None:
    def ticker(req: httpx.Request) -> httpx.Response:
        sym = req.url.params.get("symbol", "BTCUSDT")
        if sym == "ETHUSDT":
            return _binance_ticker(sym, eth_price)
        return _binance_ticker(sym, btc_price)

    def klines(req: httpx.Request) -> httpx.Response:
        sym = req.url.params.get("symbol", "BTCUSDT")
        return _binance_klines(eth_price if sym == "ETHUSDT" else btc_price)

    respx_mock.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(side_effect=ticker)
    respx_mock.get("https://api.binance.com/api/v3/klines").mock(side_effect=klines)


# ---------------------------------------------------------------------------
# /compare — always-on rows
# ---------------------------------------------------------------------------


def test_compare_always_returns_all_four_rows(app_client: TestClient) -> None:
    """BTC×5m, BTC×15m, ETH×5m, ETH×15m must all appear even with no markets."""
    with respx.mock as r:
        _mock_no_polymarket(r)
        _mock_binance(r)
        resp = app_client.get("/strategies/crypto/5min/compare?use_cache=false")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == 4
    assert body["n_polymarket_active"] == 0
    keys = {(row["asset"], row["window_minutes"]) for row in body["rows"]}
    assert keys == {("BTC", 5), ("BTC", 15), ("ETH", 5), ("ETH", 15)}
    for row in body["rows"]:
        assert row["model_prob_up"] is not None
        assert 0.0 <= row["model_prob_up"] <= 1.0
        assert row["market_prob_up"] is None
        assert row["signal"] == "WAIT"
        assert row["has_polymarket_market"] is False
        assert row["polymarket_available"] is False


def test_compare_subset_assets_and_windows(app_client: TestClient) -> None:
    with respx.mock as r:
        _mock_no_polymarket(r)
        _mock_binance(r)
        resp = app_client.get(
            "/strategies/crypto/5min/compare?assets=BTC&window_minutes_csv=5&use_cache=false"
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == 1
    assert body["rows"][0]["asset"] == "BTC"
    assert body["rows"][0]["window_minutes"] == 5


def test_compare_attaches_market_prob_when_polymarket_open(app_client: TestClient) -> None:
    """When Polymarket has the BTC 5m market open we expect mkt prob filled in."""
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"

    def gamma_handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_gamma_payload(slug)])
        return httpx.Response(200, json=[])

    with respx.mock as r:
        r.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma_handler)
        r.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.55"})
        )
        _mock_binance(r)
        resp = app_client.get(
            "/strategies/crypto/5min/compare?assets=BTC,ETH&window_minutes_csv=5,15&use_cache=false"
        )
    assert resp.status_code == 200
    body = resp.json()
    btc_5m = next(
        row for row in body["rows"] if row["asset"] == "BTC" and row["window_minutes"] == 5
    )
    assert btc_5m["market_prob_up"] == pytest.approx(0.55)
    assert btc_5m["model_prob_up"] is not None
    assert btc_5m["has_polymarket_market"] is True
    assert btc_5m["polymarket_available"] is True
    assert btc_5m["slug"] == slug
    assert btc_5m["edge"] == pytest.approx(btc_5m["model_prob_up"] - 0.55)
    assert btc_5m["signal"] in {"BUY_YES", "BUY_NO", "WAIT"}
    # Other 3 combos still null on market side
    others = [r for r in body["rows"] if not (r["asset"] == "BTC" and r["window_minutes"] == 5)]
    for row in others:
        assert row["market_prob_up"] is None


def test_compare_when_clob_down_keeps_model_and_marks_market_null(app_client: TestClient) -> None:
    """Polymarket open, CLOB midpoint endpoint returns 5xx — model still works."""
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"

    def gamma_handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_gamma_payload(slug)])
        return httpx.Response(200, json=[])

    with respx.mock as r:
        r.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma_handler)
        r.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(return_value=httpx.Response(503))
        _mock_binance(r)
        resp = app_client.get(
            "/strategies/crypto/5min/compare?assets=BTC&window_minutes_csv=5&use_cache=false"
        )
    assert resp.status_code == 200
    row = resp.json()["rows"][0]
    assert row["model_prob_up"] is not None
    assert row["market_prob_up"] is None
    assert row["has_polymarket_market"] is True
    assert row["polymarket_available"] is False
    assert row["signal"] == "WAIT"


def test_compare_cache_round_trip(app_client: TestClient) -> None:
    with respx.mock as r:
        _mock_no_polymarket(r)
        _mock_binance(r)
        first = app_client.get("/strategies/crypto/5min/compare?use_cache=true")
        second = app_client.get("/strategies/crypto/5min/compare?use_cache=true")
    assert first.json()["from_cache"] is False
    assert second.json()["from_cache"] is True


def test_compare_bypasses_cache_with_use_cache_false(app_client: TestClient) -> None:
    with respx.mock as r:
        _mock_no_polymarket(r)
        _mock_binance(r)
        app_client.get("/strategies/crypto/5min/compare?use_cache=true")
        bypass = app_client.get("/strategies/crypto/5min/compare?use_cache=false")
    assert bypass.json()["from_cache"] is False


def test_compare_rejects_bad_window_csv(app_client: TestClient) -> None:
    r = app_client.get("/strategies/crypto/5min/compare?window_minutes_csv=oops")
    assert r.status_code == 400


def test_compare_rejects_unsupported_assets(app_client: TestClient) -> None:
    r = app_client.get("/strategies/crypto/5min/compare?assets=DOGE")
    assert r.status_code == 400


def test_compare_in_openapi(app_client: TestClient) -> None:
    r = app_client.get("/openapi.json")
    assert "/strategies/crypto/5min/compare" in r.json()["paths"]


def test_compare_rows_have_consistent_edge_sign(app_client: TestClient) -> None:
    """Edge must equal model_prob - market_prob whenever both are set."""
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"

    def gamma_handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_gamma_payload(slug)])
        return httpx.Response(200, json=[])

    with respx.mock as r:
        r.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma_handler)
        r.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.20"})
        )
        _mock_binance(r)
        resp = app_client.get(
            "/strategies/crypto/5min/compare?assets=BTC&window_minutes_csv=5&use_cache=false"
        )
    row = resp.json()["rows"][0]
    assert row["edge"] == pytest.approx(row["model_prob_up"] - 0.20)
    # Big positive edge ⇒ BUY YES
    if row["edge"] >= 0.05:
        assert row["signal"] == "BUY_YES"


def test_compare_rows_include_confidence_and_z_fields(app_client: TestClient) -> None:
    """Every row must carry confidence_score, signal_strength, z_model, breakdown."""
    with respx.mock as r:
        _mock_no_polymarket(r)
        _mock_binance(r)
        resp = app_client.get("/strategies/crypto/5min/compare?use_cache=false")
    assert resp.status_code == 200
    for row in resp.json()["rows"]:
        assert "confidence_score" in row
        assert 0.0 <= row["confidence_score"] <= 100.0
        assert row["signal_strength"] in {"STRONG", "MEDIUM", "WEAK"}
        assert "z_model" in row
        # No market → z_edge is null
        assert row["z_edge"] is None
        breakdown = row["confidence_breakdown"]
        assert {
            "data_quality",
            "engine_quality",
            "edge_magnitude",
            "time_decay",
            "total",
        } <= breakdown.keys()
        # n_samples should appear so the UI can debug warmup
        assert "n_spot_samples" in row


def test_compare_z_edge_populated_when_market_open(app_client: TestClient) -> None:
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"

    def gamma_handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_gamma_payload(slug)])
        return httpx.Response(200, json=[])

    with respx.mock as r:
        r.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma_handler)
        r.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.30"})
        )
        _mock_binance(r)
        resp = app_client.get(
            "/strategies/crypto/5min/compare?assets=BTC&window_minutes_csv=5&use_cache=false"
        )
    row = resp.json()["rows"][0]
    # With a 20% edge in our favor (model ~0.50 vs market 0.30) z_edge should
    # be non-null and signed in the same direction as edge.
    assert row["z_edge"] is not None
    assert (row["z_edge"] > 0) == (row["edge"] > 0)
    assert row["confidence_score"] > 0


def test_compare_returns_end_unix_for_each_row(app_client: TestClient) -> None:
    """Every row must carry an absolute end_unix so the UI can tick accurately."""
    with respx.mock as r:
        _mock_no_polymarket(r)
        _mock_binance(r)
        resp = app_client.get("/strategies/crypto/5min/compare?use_cache=false")
    body = resp.json()
    now = time.time()
    for row in body["rows"]:
        # No Polymarket market → end_unix derives from state.anchor() = next
        # natural boundary. Must be in the future (or 0 if buffer empty).
        if row["end_unix"] is not None:
            assert row["end_unix"] > now - 5  # not way in the past
            # Must match seconds_remaining within ~5s (server clock drift)
            assert abs((row["end_unix"] - now) - row["seconds_remaining"]) < 5


def test_compare_end_unix_matches_polymarket_slug_when_market_open(app_client: TestClient) -> None:
    """The end_unix on the row must equal the unix suffix of the slug."""
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"

    def gamma_handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_gamma_payload(slug)])
        return httpx.Response(200, json=[])

    with respx.mock as r:
        r.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma_handler)
        r.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.50"})
        )
        _mock_binance(r)
        resp = app_client.get(
            "/strategies/crypto/5min/compare?assets=BTC&window_minutes_csv=5&use_cache=false"
        )
    row = resp.json()["rows"][0]
    assert row["end_unix"] == next_end
    # Sanity: seconds_remaining = end_unix - server_now, must be < period
    assert 0 < row["seconds_remaining"] <= period


def test_compare_seconds_remaining_uses_fresh_clock_not_cached(app_client: TestClient) -> None:
    """The fix for the 0:00 bug: build_compare_payload recomputes
    seconds_remaining from time.time() at every iteration, not from when
    discover_active_markets ran. This means if the prewarmer ran 2s ago and
    we serve a fresh build now, seconds_remaining reflects *now*."""
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"

    def gamma_handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == slug:
            return httpx.Response(200, json=[_gamma_payload(slug)])
        return httpx.Response(200, json=[])

    with respx.mock as r:
        r.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma_handler)
        r.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.50"})
        )
        _mock_binance(r)
        resp = app_client.get(
            "/strategies/crypto/5min/compare?assets=BTC&window_minutes_csv=5&use_cache=false"
        )
    row = resp.json()["rows"][0]
    server_now = time.time()
    expected_secs = row["end_unix"] - server_now
    # Allow 1s tolerance for the time between the test's time.time() and
    # the server's time.time() inside the request handler.
    assert abs(row["seconds_remaining"] - expected_secs) < 2.0


def test_compare_handles_partial_polymarket_coverage(app_client: TestClient) -> None:
    """Polymarket only has the BTC 5m market — ETH and BTC 15m must still return."""
    now = int(time.time())
    period = 300
    next_end = ((now // period) + 1) * period
    btc_5m_slug = f"btc-updown-5m-{next_end}"

    def gamma_handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("slug") == btc_5m_slug:
            return httpx.Response(200, json=[_gamma_payload(btc_5m_slug)])
        return httpx.Response(200, json=[])

    with respx.mock as r:
        r.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=gamma_handler)
        r.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.51"})
        )
        _mock_binance(r)
        resp = app_client.get(
            "/strategies/crypto/5min/compare?assets=BTC,ETH&window_minutes_csv=5,15&use_cache=false"
        )
    body = resp.json()
    assert body["n_rows"] == 4
    assert body["n_polymarket_active"] == 1
    by_key = {(r["asset"], r["window_minutes"]): r for r in body["rows"]}
    assert by_key[("BTC", 5)]["market_prob_up"] is not None
    assert by_key[("BTC", 15)]["market_prob_up"] is None
    assert by_key[("ETH", 5)]["market_prob_up"] is None
    assert by_key[("ETH", 15)]["market_prob_up"] is None
    # All four rows still have a model probability
    for row in body["rows"]:
        assert row["model_prob_up"] is not None
