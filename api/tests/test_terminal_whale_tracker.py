"""Tests for the Polymarket whale tracker terminal module."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_whale_tracker import (
    aggregate_whales,
    directional_skew,
    router,
)

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"
DATA_API = "https://data-api.polymarket.com"
CONDITION_ID = "0xcondition_whale"
SLUG = "whale-market"
YES_TOKEN = "111"
NO_TOKEN = "222"


def _build_app() -> tuple[TestClient, httpx.Client]:
    """Spin up a FastAPI test app with only the whale-tracker router mounted."""
    app = FastAPI()
    app.include_router(router)
    http = httpx.Client()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=http)
    return TestClient(app), http


def _gamma_market_payload(condition_id: str = CONDITION_ID) -> list[dict]:
    return [
        {
            "slug": SLUG,
            "question": "Will the whale event happen?",
            "conditionId": condition_id,
            "clobTokenIds": json.dumps([YES_TOKEN, NO_TOKEN]),
            "active": True,
            "closed": False,
        }
    ]


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_aggregate_whales_filters_and_sums_per_address() -> None:
    """Aggregation should sum YES + NO per address and apply the threshold."""
    raw = [
        # Whale A: $15k YES + $3k NO, gross $18k → keep.
        {
            "proxyWallet": "0xA",
            "asset": YES_TOKEN,
            "outcome": "Yes",
            "currentValue": 15_000.0,
        },
        {
            "proxyWallet": "0xA",
            "asset": NO_TOKEN,
            "outcome": "No",
            "currentValue": 3_000.0,
        },
        # Whale B: $12k YES → keep.
        {
            "proxyWallet": "0xB",
            "asset": YES_TOKEN,
            "outcome": "Yes",
            "currentValue": 12_000.0,
        },
        # Minnow: $500 YES → drop.
        {
            "proxyWallet": "0xC",
            "asset": YES_TOKEN,
            "outcome": "Yes",
            "currentValue": 500.0,
        },
        # Row missing address → drop.
        {"asset": YES_TOKEN, "outcome": "Yes", "currentValue": 99_999.0},
    ]
    whales = aggregate_whales(
        raw,
        yes_token_id=YES_TOKEN,
        no_token_id=NO_TOKEN,
        min_position_usd=10_000.0,
        limit=20,
    )
    addrs = [w.address for w in whales]
    assert addrs == ["0xA", "0xB"]  # sorted by gross notional, A=18k > B=12k
    a = whales[0]
    assert a.position_yes_usd == 15_000.0
    assert a.position_no_usd == 3_000.0
    assert a.net_usd == 12_000.0


def test_directional_skew_handles_yes_lean_and_empty() -> None:
    """Skew is YES share; defaults to 0.5 on empty."""
    assert directional_skew([]) == 0.5
    raw = [
        {"proxyWallet": "0xA", "asset": YES_TOKEN, "outcome": "Yes", "currentValue": 80_000.0},
        {"proxyWallet": "0xB", "asset": NO_TOKEN, "outcome": "No", "currentValue": 20_000.0},
    ]
    whales = aggregate_whales(
        raw,
        yes_token_id=YES_TOKEN,
        no_token_id=NO_TOKEN,
        min_position_usd=10_000.0,
        limit=20,
    )
    skew = directional_skew(whales)
    assert abs(skew - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# Endpoint tests (Gamma + data-api both mocked via respx)
# ---------------------------------------------------------------------------


@respx.mock
def test_whales_endpoint_returns_aggregated_response() -> None:
    client, _http = _build_app()

    respx.get(f"{GAMMA}/markets", params={"slug": SLUG}).mock(
        return_value=httpx.Response(200, json=_gamma_market_payload())
    )

    positions = [
        {
            "proxyWallet": "0xA",
            "asset": YES_TOKEN,
            "outcome": "Yes",
            "currentValue": 50_000.0,
        },
        {
            "proxyWallet": "0xB",
            "asset": NO_TOKEN,
            "outcome": "No",
            "currentValue": 25_000.0,
        },
        # Sub-threshold whale that must be filtered out.
        {
            "proxyWallet": "0xC",
            "asset": YES_TOKEN,
            "outcome": "Yes",
            "currentValue": 100.0,
        },
    ]
    pos_route = respx.get(f"{DATA_API}/positions").mock(
        return_value=httpx.Response(200, json=positions)
    )

    # Light-touch trade enrichment: one fresh trade per whale.
    now = datetime.now(UTC)
    trades = [
        {
            "proxyWallet": "0xA",
            "timestamp": int((now - timedelta(hours=1)).timestamp()),
            "price": 0.6,
            "size": 1000,
        },
        {
            "proxyWallet": "0xB",
            "timestamp": int((now - timedelta(hours=2)).timestamp()),
            "price": 0.4,
            "size": 500,
        },
    ]
    respx.get(f"{DATA_API}/trades").mock(return_value=httpx.Response(200, json=trades))

    resp = client.get(
        f"/terminal/whales/{SLUG}",
        params={"min_position_usd": 10_000, "limit": 20},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["slug"] == SLUG
    assert body["condition_id"] == CONDITION_ID
    assert body["n_whales"] == 2
    assert body["total_whale_notional_usd"] == 75_000.0
    # YES-leaning: $50k / ($50k + $25k) ≈ 0.6667
    assert abs(body["net_directional_skew"] - 0.6667) < 1e-3
    addrs = [w["address"] for w in body["whales"]]
    assert addrs == ["0xA", "0xB"]
    # 24h enrichment: each whale has exactly one fresh trade.
    a = body["whales"][0]
    assert a["n_trades_24h"] == 1
    assert a["last_active_iso"] is not None
    # Verify positions endpoint received conditionId.
    sent = pos_route.calls.last.request
    assert sent.url.params["market"] == CONDITION_ID
    # Interpretation string mentions YES skew.
    assert "YES" in body["interpretation"]


@respx.mock
def test_whales_endpoint_404_when_slug_missing() -> None:
    """Missing market → structured 404 with discriminator + slug echo.

    The frontend keys off ``detail.error`` to choose a polite empty
    state instead of leaking the raw "Not Found" string into the UI.
    """
    client, _http = _build_app()
    respx.get(f"{GAMMA}/markets", params={"slug": "ghost"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    pos_route = respx.get(f"{DATA_API}/positions").mock(return_value=httpx.Response(200, json=[]))
    r = client.get("/terminal/whales/ghost")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert isinstance(detail, dict), f"detail must be structured, got: {detail!r}"
    assert detail["error"] == "whale_tracking_unavailable"
    assert detail["slug"] == "ghost"
    assert detail["message"]
    assert pos_route.call_count == 0


@respx.mock
def test_recent_large_trades_filters_by_size_and_window() -> None:
    client, _http = _build_app()

    respx.get(f"{GAMMA}/markets", params={"slug": SLUG}).mock(
        return_value=httpx.Response(200, json=_gamma_market_payload())
    )

    now = datetime.now(UTC)
    trades = [
        # Big, fresh → keep ($6k).
        {
            "proxyWallet": "0xA",
            "timestamp": int((now - timedelta(hours=2)).timestamp()),
            "price": 0.6,
            "size": 10_000,
        },
        # Big, but stale (older than 24h) → drop.
        {
            "proxyWallet": "0xB",
            "timestamp": int((now - timedelta(hours=48)).timestamp()),
            "price": 0.5,
            "size": 50_000,
        },
        # Fresh, but small ($100) → drop.
        {
            "proxyWallet": "0xC",
            "timestamp": int((now - timedelta(hours=3)).timestamp()),
            "price": 0.5,
            "size": 200,
        },
        # Big, fresh, with explicit usdcSize → keep ($8k).
        {
            "proxyWallet": "0xD",
            "timestamp": int((now - timedelta(hours=1)).timestamp()),
            "price": 0.4,
            "size": 20_000,
            "usdcSize": 8_000.0,
        },
    ]
    respx.get(f"{DATA_API}/trades").mock(return_value=httpx.Response(200, json=trades))

    resp = client.get(
        "/terminal/whales/recent-large-trades",
        params={"slug": SLUG, "min_size_usd": 5_000, "hours": 24},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_trades"] == 2
    addrs = [t["address"] for t in body["trades"]]
    # Sorted by size desc: 0xD ($8k) > 0xA ($6k).
    assert addrs == ["0xD", "0xA"]
    assert body["trades"][0]["size_usd"] == 8_000.0
    assert body["trades"][1]["size_usd"] == 6_000.0
    assert body["total_notional_usd"] == 14_000.0
