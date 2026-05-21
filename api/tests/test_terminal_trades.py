"""Tests for the Polymarket terminal trade tape with Lee-Ready inference."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_trades import (
    classify_trades,
    rolling_buy_ratio,
    router,
)

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"
DATA_API = "https://data-api.polymarket.com"
CONDITION_ID = "0xcondition123"
SLUG = "fed-decision"


def _build_app() -> tuple[TestClient, httpx.Client]:
    """Construct a minimal FastAPI app that mounts only our router.

    Reusing the same httpx.Client respx is patching for both Gamma and
    data-api keeps the test self-contained without booting all of main.py.
    """
    app = FastAPI()
    app.include_router(router)
    http = httpx.Client()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=http)
    return TestClient(app), http


def _gamma_market_payload(condition_id: str = CONDITION_ID) -> list[dict]:
    return [
        {
            "slug": SLUG,
            "question": "Will the Fed cut rates?",
            "conditionId": condition_id,
            "clobTokenIds": json.dumps(["111", "222"]),
            "active": True,
            "closed": False,
        }
    ]


# ---------------------------------------------------------------------------
# Algorithm-level tests (pure functions, no HTTP)
# ---------------------------------------------------------------------------


def test_classify_trades_uses_quote_rule_when_bid_ask_present() -> None:
    """When bid/ask straddle the trade, sign by side of the mid."""
    raw = [
        {"timestamp": 1_700_000_000, "price": 0.55, "size": 10, "bid": 0.50, "ask": 0.52},
        # at-or-above ask → BUY
        {"timestamp": 1_700_000_010, "price": 0.49, "size": 10, "bid": 0.50, "ask": 0.52},
        # below mid → SELL
        {"timestamp": 1_700_000_020, "price": 0.51, "size": 10, "bid": 0.50, "ask": 0.52},
        # exactly at mid (0.51) → tick test against prev (0.49) → BUY
    ]
    out = classify_trades(raw)
    assert [t.side for t in out] == ["BUY", "SELL", "BUY"]


def test_classify_trades_falls_back_to_tick_test_without_quotes() -> None:
    """Without bid/ask, sign by direction relative to the previous trade."""
    raw = [
        {"timestamp": 1_700_000_000, "price": 0.40, "size": 5},  # first → AT_MID
        {"timestamp": 1_700_000_010, "price": 0.42, "size": 5},  # up   → BUY
        {"timestamp": 1_700_000_020, "price": 0.41, "size": 5},  # down → SELL
        {"timestamp": 1_700_000_030, "price": 0.41, "size": 5},  # flat → inherit SELL
    ]
    sides = [t.side for t in classify_trades(raw)]
    assert sides == ["AT_MID", "BUY", "SELL", "SELL"]


def test_rolling_buy_ratio_flags_informed_flow() -> None:
    """Heavily one-sided flow should breach the 0.15 deviation threshold."""
    # Eight consecutive up-ticks → all BUY → ratio ≈ 1.0 well above 0.65.
    raw = [
        {"timestamp": 1_700_000_000 + 30 * i, "price": 0.40 + 0.001 * i, "size": 100}
        for i in range(8)
    ]
    trades = classify_trades(raw)
    flow = rolling_buy_ratio(trades)
    assert flow, "expected non-empty rolling output"
    # Every classified-side trade after the first should sit at ratio == 1.0.
    classified = [w for w, t in zip(flow, trades, strict=True) if t.side == "BUY"]
    assert classified
    assert classified[-1].buy_ratio == pytest.approx(1.0)
    assert classified[-1].informed is True


# ---------------------------------------------------------------------------
# Endpoint-level test (Gamma + data-api both mocked via respx)
# ---------------------------------------------------------------------------


@respx.mock
def test_endpoint_returns_classified_tape_and_informed_flag() -> None:
    client, _http = _build_app()

    respx.get(f"{GAMMA}/markets", params={"slug": SLUG}).mock(
        return_value=httpx.Response(200, json=_gamma_market_payload())
    )

    # data-api returns newest → oldest by convention; the endpoint must
    # sort chronologically before classifying. We mimic that here.
    raw_trades = [
        {"timestamp": 1_700_000_030, "price": 0.55, "size": 8},
        {"timestamp": 1_700_000_020, "price": 0.54, "size": 7},
        {"timestamp": 1_700_000_010, "price": 0.53, "size": 6},
        {"timestamp": 1_700_000_000, "price": 0.50, "size": 5},
    ]
    route = respx.get(f"{DATA_API}/trades").mock(return_value=httpx.Response(200, json=raw_trades))

    resp = client.get(f"/terminal/trades/{SLUG}", params={"limit": 4})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["slug"] == SLUG
    assert body["condition_id"] == CONDITION_ID
    assert body["n_trades"] == 4

    # Verify the data-api request used the resolved conditionId + limit.
    sent = route.calls.last.request
    assert sent.url.params["market"] == CONDITION_ID
    assert sent.url.params["limit"] == "4"

    # Sides: chronological order is 0.50 → 0.53 → 0.54 → 0.55, all up-ticks.
    # First is unclassifiable (AT_MID); the rest are BUYs.
    sides = [t["side"] for t in body["trades"]]
    assert sides == ["AT_MID", "BUY", "BUY", "BUY"]

    # All classifiable flow is BUY → ratio == 1.0 in the 5-min window
    # → informed-flow alert must trip.
    assert body["informed_flow_alert"] is True
    assert any(w["informed"] for w in body["rolling_buy_ratio"])


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------


@respx.mock
def test_endpoint_404_when_slug_missing() -> None:
    """Empty Gamma response → 404 (and data-api never called)."""
    client, _http = _build_app()
    respx.get(f"{GAMMA}/markets", params={"slug": "ghost"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    data_route = respx.get(f"{DATA_API}/trades").mock(return_value=httpx.Response(200, json=[]))
    r = client.get("/terminal/trades/ghost")
    assert r.status_code == 404
    assert "no market found" in r.json()["detail"]
    assert data_route.call_count == 0


@respx.mock
def test_endpoint_502_when_data_api_500s() -> None:
    """A 5xx from the data-api is wrapped into 502."""
    client, _http = _build_app()
    respx.get(f"{GAMMA}/markets", params={"slug": SLUG}).mock(
        return_value=httpx.Response(200, json=_gamma_market_payload())
    )
    respx.get(f"{DATA_API}/trades").mock(return_value=httpx.Response(500))
    r = client.get(f"/terminal/trades/{SLUG}")
    assert r.status_code == 502
    assert "data-api" in r.json()["detail"]


@respx.mock
def test_endpoint_handles_empty_trade_list() -> None:
    """Zero trades → n_trades=0, empty arrays, informed_flow_alert=False."""
    client, _http = _build_app()
    respx.get(f"{GAMMA}/markets", params={"slug": SLUG}).mock(
        return_value=httpx.Response(200, json=_gamma_market_payload())
    )
    respx.get(f"{DATA_API}/trades").mock(return_value=httpx.Response(200, json=[]))
    body = client.get(f"/terminal/trades/{SLUG}?limit=10").json()
    assert body["n_trades"] == 0
    assert body["trades"] == []
    assert body["rolling_buy_ratio"] == []
    assert body["informed_flow_alert"] is False


@respx.mock
def test_endpoint_502_when_condition_id_missing() -> None:
    """A market with no conditionId in the gamma payload → 502 with detail."""
    client, _http = _build_app()
    respx.get(f"{GAMMA}/markets", params={"slug": SLUG}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": SLUG,
                    "question": "Will the Fed cut?",
                    "clobTokenIds": json.dumps(["111", "222"]),
                    "active": True,
                    "closed": False,
                    # conditionId deliberately omitted.
                }
            ],
        )
    )
    r = client.get(f"/terminal/trades/{SLUG}")
    assert r.status_code == 502
    assert "conditionId" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 2026-05-15 upstream-hardening: 429 retry on gamma + data-api
# ---------------------------------------------------------------------------


@respx.mock
def test_data_api_429_retried_once(monkeypatch) -> None:
    """A 429 on data-api /trades is absorbed by the retry; UI sees 200.

    Uses a unique slug+conditionId so prior tests' cached tape can't
    shadow the upstream call we mock here.
    """
    import pfm.terminal.trades as trd

    monkeypatch.setattr(trd, "_RETRY_BACKOFF_S", 0.01)

    unique_slug = "fed-decision-data-retry"
    unique_cid = "0xcondition-data-retry"
    payload = [
        {
            "slug": unique_slug,
            "question": "Q",
            "conditionId": unique_cid,
            "clobTokenIds": json.dumps(["111", "222"]),
            "active": True,
            "closed": False,
        }
    ]

    client, _http = _build_app()
    respx.get(f"{GAMMA}/markets", params={"slug": unique_slug}).mock(
        return_value=httpx.Response(200, json=payload)
    )
    route = respx.get(f"{DATA_API}/trades", params={"market": unique_cid, "limit": 50}).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=[]),
        ]
    )
    r = client.get(f"/terminal/trades/{unique_slug}")
    assert r.status_code == 200, r.text
    assert route.call_count == 2


@respx.mock
def test_gamma_429_retried_once_for_condition_id(monkeypatch) -> None:
    """A 429 on the gamma slug→conditionId resolve is also absorbed.

    Uses a distinct slug to avoid sharing the conditionId cache with the
    other tests in this module — the autouse fixture clears it between
    tests, but a unique slug is belt-and-braces against future ordering.
    """
    import pfm.terminal.trades as trd

    monkeypatch.setattr(trd, "_RETRY_BACKOFF_S", 0.01)

    unique_slug = "fed-decision-gamma-retry"
    payload = [
        {
            "slug": unique_slug,
            "question": "Q",
            "conditionId": CONDITION_ID,
            "clobTokenIds": json.dumps(["111", "222"]),
            "active": True,
            "closed": False,
        }
    ]

    client, _http = _build_app()
    gamma_route = respx.get(f"{GAMMA}/markets", params={"slug": unique_slug}).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=payload),
        ]
    )
    respx.get(f"{DATA_API}/trades").mock(return_value=httpx.Response(200, json=[]))

    r = client.get(f"/terminal/trades/{unique_slug}")
    assert r.status_code == 200, r.text
    assert gamma_route.call_count == 2
