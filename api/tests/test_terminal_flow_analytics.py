"""Tests for the trade-flow analytics endpoint."""

from __future__ import annotations

import json

import httpx
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_flow_analytics import (
    compute_flow_metrics,
    detect_bursts,
    router,
)
from pfm.terminal_flow_analytics import router as flow_router  # noqa: F401
from pfm.terminal_trades import classify_trades
from pfm.terminal_trades import router as trades_router

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"
DATA_API = "https://data-api.polymarket.com"
CONDITION_ID = "0xcondition999"
SLUG = "fed-decision-flow"


def _build_app() -> tuple[TestClient, httpx.Client]:
    """Mount both the trades router and the flow router on a fresh app.

    The flow endpoint internally calls the trades endpoint, so we wire
    both into the test app and let respx intercept the underlying httpx
    calls.
    """
    app = FastAPI()
    app.include_router(trades_router)
    app.include_router(router)
    http = httpx.Client()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=http)
    return TestClient(app), http


def _gamma_market_payload() -> list[dict]:
    return [
        {
            "slug": SLUG,
            "question": "Will the Fed cut rates?",
            "conditionId": CONDITION_ID,
            "clobTokenIds": json.dumps(["111", "222"]),
            "active": True,
            "closed": False,
        }
    ]


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_compute_flow_metrics_aggregates_and_signals_buy() -> None:
    """Eight monotonic up-ticks → all aggressive BUYs → BUY signal."""
    raw = [
        {"timestamp": 1_700_000_000 + 30 * i, "price": 0.40 + 0.01 * i, "size": 100}
        for i in range(8)
    ]
    trades = classify_trades(raw)
    resp = compute_flow_metrics(SLUG, trades, window_minutes=60)

    assert resp.n_trades_total == 8
    assert resp.n_trades_buy == 7  # first trade is AT_MID
    assert resp.n_trades_sell == 0
    assert resp.buy_ratio == 1.0
    # Every BUY printed strictly above the prior trade → aggressive.
    assert resp.aggressive_ratio == 7 / 8
    assert resp.notional_buy_usd > 0
    assert resp.notional_sell_usd == 0.0
    assert resp.net_flow_usd > 0
    assert resp.largest_trade_usd == max(t.price * t.size for t in trades)
    assert len(resp.top_5_trades) == 5
    assert resp.informed_flow_signal == "BUY"


def test_compute_flow_metrics_neutral_when_aggressive_ratio_below_gate() -> None:
    """Flat-tape printing → aggressive_ratio == 0 → NEUTRAL signal."""
    # All trades print at exactly the same price. Quote rule classifies
    # them by mid (BUY if price > mid, SELL if price < mid). We use a
    # mix-of-mids construction by tagging different bid/ask windows so
    # we get classified BUYs and SELLs without any tick movement —
    # meaning aggressive_ratio stays at 0 and the signal is NEUTRAL.
    raw = [
        {"timestamp": 1_700_000_000, "price": 0.50, "size": 10, "bid": 0.48, "ask": 0.51},
        # mid = 0.495 → 0.50 > mid → BUY
        {"timestamp": 1_700_000_010, "price": 0.50, "size": 10, "bid": 0.49, "ask": 0.52},
        # mid = 0.505 → 0.50 < mid → SELL
        {"timestamp": 1_700_000_020, "price": 0.50, "size": 10, "bid": 0.48, "ask": 0.51},
        # mid = 0.495 → BUY
        {"timestamp": 1_700_000_030, "price": 0.50, "size": 10, "bid": 0.49, "ask": 0.52},
        # mid = 0.505 → SELL
    ]
    trades = classify_trades(raw)
    resp = compute_flow_metrics(SLUG, trades, window_minutes=60)

    assert resp.n_trades_buy == 2
    assert resp.n_trades_sell == 2
    # No price movement → aggressive_ratio = 0 → gate fails → NEUTRAL.
    assert resp.aggressive_ratio == 0.0
    assert resp.informed_flow_signal == "NEUTRAL"
    # Notionals are equal → net_flow == 0.
    assert resp.net_flow_usd == 0.0


def test_detect_bursts_flags_dense_minute() -> None:
    """Cram 6 trades into one minute alongside sparse minutes → that bin trips."""
    # Five sparse 1-min buckets with 1 trade each, then one dense bucket with 6.
    base = 1_700_000_000
    sparse = [{"timestamp": base + 60 * i, "price": 0.50, "size": 1.0} for i in range(5)]
    dense = [{"timestamp": base + 60 * 5 + i, "price": 0.50, "size": 1.0} for i in range(6)]
    trades = classify_trades(sparse + dense)
    bursts = detect_bursts(trades)
    assert bursts, "expected at least one burst"
    # The dense bucket sits well above 2× the rolling mean of ~1.83.
    assert any(b.n_trades >= 6 for b in bursts)
    assert all(b.magnitude > 2.0 for b in bursts)


# ---------------------------------------------------------------------------
# Endpoint test
# ---------------------------------------------------------------------------


@respx.mock
def test_endpoint_returns_flow_payload_with_buy_signal() -> None:
    """End-to-end: Gamma + data-api both mocked, BUY signal expected."""
    client, _http = _build_app()

    respx.get(f"{GAMMA}/markets", params={"slug": SLUG}).mock(
        return_value=httpx.Response(200, json=_gamma_market_payload())
    )

    raw_trades = [
        {"timestamp": 1_700_000_030, "price": 0.55, "size": 8},
        {"timestamp": 1_700_000_020, "price": 0.54, "size": 7},
        {"timestamp": 1_700_000_010, "price": 0.53, "size": 6},
        {"timestamp": 1_700_000_000, "price": 0.50, "size": 5},
    ]
    respx.get(f"{DATA_API}/trades").mock(return_value=httpx.Response(200, json=raw_trades))

    resp = client.get(f"/terminal/flow/{SLUG}", params={"window_minutes": 60})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["slug"] == SLUG
    assert body["window_minutes"] == 60
    assert body["n_trades_total"] == 4
    assert body["n_trades_buy"] == 3  # first trade is AT_MID
    assert body["n_trades_sell"] == 0
    assert body["buy_ratio"] == 1.0
    assert body["aggressive_ratio"] == 3 / 4
    assert body["notional_buy_usd"] > 0
    assert body["notional_sell_usd"] == 0.0
    assert body["net_flow_usd"] > 0
    assert body["informed_flow_signal"] == "BUY"
    # top_5 capped to len(trades) when fewer than 5.
    assert len(body["top_5_trades"]) == 4


@respx.mock
def test_flow_and_trades_share_cache_on_repeat_hit() -> None:
    """Concurrent flow + trades fanout for the same slug must share one
    upstream gamma + data-api call thanks to the per-slug trades cache.

    Without the cache the market-detail open made 6+ rate-limited calls
    (orderbook, trades, flow, quality each hitting gamma); the fix is a
    5 s tape cache that collapses repeat hits.
    """
    from pfm.cache_utils import get_cache

    get_cache("terminal_flow").clear()
    get_cache("terminal_trades_cid").clear()
    get_cache("terminal_trades_tape").clear()

    client, _http = _build_app()

    gamma_route = respx.get(f"{GAMMA}/markets", params={"slug": SLUG}).mock(
        return_value=httpx.Response(200, json=_gamma_market_payload())
    )
    data_route = respx.get(f"{DATA_API}/trades").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"timestamp": 1_700_000_030, "price": 0.55, "size": 8},
                {"timestamp": 1_700_000_020, "price": 0.54, "size": 7},
                {"timestamp": 1_700_000_010, "price": 0.53, "size": 6},
                {"timestamp": 1_700_000_000, "price": 0.50, "size": 5},
            ],
        )
    )

    # Two flow hits (same window) + one trades hit at limit=500 (the limit
    # flow internally uses). The tape cache key is (slug, limit), so all
    # three queries collapse to a single upstream call.
    r1 = client.get(f"/terminal/flow/{SLUG}", params={"window_minutes": 60})
    r2 = client.get(f"/terminal/flow/{SLUG}", params={"window_minutes": 60})
    r3 = client.get(f"/terminal/trades/{SLUG}", params={"limit": 500})
    assert r1.status_code == r2.status_code == r3.status_code == 200
    # Gamma must be resolved at most once (slug→conditionId is permanent).
    assert gamma_route.call_count == 1, (
        f"slug→conditionId should be cached, got {gamma_route.call_count} gamma calls"
    )
    # data-api should be hit at most once across the 3 calls.
    assert data_route.call_count == 1, (
        f"trade tape should be cached, got {data_route.call_count} data-api calls"
    )
