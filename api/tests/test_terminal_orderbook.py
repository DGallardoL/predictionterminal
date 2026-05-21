"""Tests for the terminal orderbook ladder endpoint.

Mounts the router on a throw-away FastAPI app so we don't touch ``main.py``.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal_orderbook import GAMMA_URL, router

SLUG = "fed-decision"
YES_TOKEN = "111"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _gamma_response() -> httpx.Response:
    return httpx.Response(
        200,
        json=[
            {
                "slug": SLUG,
                "question": "Will the Fed cut rates?",
                "clobTokenIds": json.dumps([YES_TOKEN, "222"]),
                "active": True,
                "closed": False,
            }
        ],
    )


@respx.mock
def test_returns_ladder_with_depth_and_fill_costs(client: TestClient) -> None:
    """Happy path: 10 levels, mid/spread, depth bands, fill costs, imbalance."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(return_value=_gamma_response())
    # 12 bid + 12 ask levels — endpoint should clip to top 10.  Sizes are
    # fat enough that a $1000 marketable order fills inside 10 levels.
    bids = [{"price": round(0.50 - 0.01 * i, 2), "size": 5000.0} for i in range(12)]
    asks = [{"price": round(0.52 + 0.01 * i, 2), "size": 5000.0} for i in range(12)]
    respx.get("https://clob.polymarket.com/book").mock(
        return_value=httpx.Response(200, json={"bids": bids, "asks": asks})
    )

    r = client.get(f"/terminal/book/{SLUG}")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["token_id"] == YES_TOKEN
    assert len(body["bid_levels"]) == 10
    assert len(body["ask_levels"]) == 10

    # Bids descending, asks ascending.
    bid_prices = [lvl["price"] for lvl in body["bid_levels"]]
    ask_prices = [lvl["price"] for lvl in body["ask_levels"]]
    assert bid_prices == sorted(bid_prices, reverse=True)
    assert ask_prices == sorted(ask_prices)

    # Cumulative size monotonic on each side (10 levels × 5000 each).
    assert body["bid_levels"][9]["cumulative"] == pytest.approx(50_000.0)
    assert body["ask_levels"][9]["cumulative"] == pytest.approx(50_000.0)

    # mid = (0.50 + 0.52) / 2 = 0.51, spread = 2 cents.
    assert body["mid"] == pytest.approx(0.51)
    assert body["spread_cents"] == pytest.approx(2.0)

    # Depth bands are non-decreasing as the band widens.
    assert body["depth_at_1c_mid"] <= body["depth_at_3c_mid"] <= body["depth_at_10c_mid"]

    # Fill costs exist for all three sizes and look sane (cents in (0, 100)).
    for size in ("50", "200", "1000"):
        cost = body["fill_cost"][size]
        assert 0 < cost["buy"] < 100
        assert 0 < cost["sell"] < 100
        # Buying lifts the offer (>= mid*100) and selling hits the bid (<= mid*100).
        assert cost["buy"] >= 51.0
        assert cost["sell"] <= 51.0

    # Symmetric book → imbalance should be neutral-ish (not >0.6 / <0.4).
    assert 0.4 <= body["imbalance_top5"] <= 0.6
    assert body["imbalance_signal"] == "neutral"


@respx.mock
def test_imbalance_flags_bullish_pressure(client: TestClient) -> None:
    """Top-5 bid mass dominates → imbalance > 0.6 → 'bullish'."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(return_value=_gamma_response())
    bids = [{"price": round(0.50 - 0.01 * i, 2), "size": 500.0} for i in range(5)]
    asks = [{"price": round(0.52 + 0.01 * i, 2), "size": 50.0} for i in range(5)]
    respx.get("https://clob.polymarket.com/book").mock(
        return_value=httpx.Response(200, json={"bids": bids, "asks": asks})
    )

    body = client.get(f"/terminal/book/{SLUG}").json()
    # 2500 / (2500 + 250) ≈ 0.909.
    assert body["imbalance_top5"] == pytest.approx(2500 / 2750)
    assert body["imbalance_signal"] == "bullish"


@respx.mock
def test_404_when_slug_missing(client: TestClient) -> None:
    """Empty Gamma response → 404, and CLOB is never called."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    book_route = respx.get("https://clob.polymarket.com/book").mock(
        return_value=httpx.Response(200, json={"bids": [], "asks": []})
    )

    r = client.get("/terminal/book/ghost")
    assert r.status_code == 404
    assert "no market found" in r.json()["detail"]
    assert book_route.call_count == 0


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------


@respx.mock
def test_imbalance_flags_bearish_pressure(client: TestClient) -> None:
    """Top-5 ask mass dominates → imbalance < 0.4 → 'bearish'."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(return_value=_gamma_response())
    bids = [{"price": round(0.50 - 0.01 * i, 2), "size": 50.0} for i in range(5)]
    asks = [{"price": round(0.52 + 0.01 * i, 2), "size": 500.0} for i in range(5)]
    respx.get("https://clob.polymarket.com/book").mock(
        return_value=httpx.Response(200, json={"bids": bids, "asks": asks})
    )
    body = client.get(f"/terminal/book/{SLUG}").json()
    assert body["imbalance_top5"] == pytest.approx(250 / 2750)
    assert body["imbalance_signal"] == "bearish"


@respx.mock
def test_empty_book_yields_neutral_signal_and_null_mid(client: TestClient) -> None:
    """A completely empty book returns null mid/spread and neutral imbalance."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(return_value=_gamma_response())
    respx.get("https://clob.polymarket.com/book").mock(
        return_value=httpx.Response(200, json={"bids": [], "asks": []})
    )
    body = client.get(f"/terminal/book/{SLUG}").json()
    assert body["mid"] is None
    assert body["spread_cents"] is None
    assert body["depth_at_1c_mid"] == 0.0
    assert body["depth_at_3c_mid"] == 0.0
    assert body["depth_at_10c_mid"] == 0.0
    assert body["imbalance_top5"] is None
    assert body["imbalance_signal"] == "neutral"
    # Fill costs: with no asks/bids, both sides report None for every size.
    for size in ("50", "200", "1000"):
        assert body["fill_cost"][size]["buy"] is None
        assert body["fill_cost"][size]["sell"] is None


@respx.mock
def test_502_when_clob_book_returns_500(client: TestClient) -> None:
    """A 5xx from /book is wrapped into 502 with a CLOB-specific message."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(return_value=_gamma_response())
    respx.get("https://clob.polymarket.com/book").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    r = client.get(f"/terminal/book/{SLUG}")
    assert r.status_code == 502
    assert "CLOB /book failed" in r.json()["detail"]


@respx.mock
def test_502_when_clobtokenids_malformed(client: TestClient) -> None:
    """Non-JSON or missing clobTokenIds in Gamma payload → 502 (never 500)."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": SLUG,
                    "question": "Will the Fed cut rates?",
                    "clobTokenIds": "not-valid-json[",
                    "active": True,
                    "closed": False,
                }
            ],
        )
    )
    r = client.get(f"/terminal/book/{SLUG}")
    assert r.status_code == 502
    assert "clobTokenIds" in r.json()["detail"]


@respx.mock
def test_response_schema_keys(client: TestClient) -> None:
    """Response includes every documented top-level key."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(return_value=_gamma_response())
    bids = [{"price": 0.50, "size": 100.0}]
    asks = [{"price": 0.52, "size": 100.0}]
    respx.get("https://clob.polymarket.com/book").mock(
        return_value=httpx.Response(200, json={"bids": bids, "asks": asks})
    )
    body = client.get(f"/terminal/book/{SLUG}").json()
    expected = {
        "slug",
        "token_id",
        "bid_levels",
        "ask_levels",
        "mid",
        "spread_cents",
        "depth_at_1c_mid",
        "depth_at_3c_mid",
        "depth_at_10c_mid",
        "fill_cost",
        "imbalance_top5",
        "imbalance_signal",
    }
    assert set(body.keys()) == expected
    assert set(body["fill_cost"].keys()) == {"50", "200", "1000"}
    for cost in body["fill_cost"].values():
        assert set(cost.keys()) == {"buy", "sell"}


@respx.mock
def test_token_id_resolution_is_cached(client: TestClient) -> None:
    """Slug→token_id is immutable; cache eliminates the gamma round-trip
    on subsequent /book calls for the same slug.

    Without the cache, two back-to-back orderbook hits trip the Gamma 429
    quota in production. The fix caches the resolved token_id for 1 h.
    """
    from pfm.cache_utils import get_cache

    get_cache("terminal_orderbook_tokens").clear()
    get_cache("terminal_orderbook_book").clear()

    gamma_route = respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(
        return_value=_gamma_response()
    )
    bids = [{"price": 0.50, "size": 100.0}]
    asks = [{"price": 0.52, "size": 100.0}]
    respx.get("https://clob.polymarket.com/book").mock(
        return_value=httpx.Response(200, json={"bids": bids, "asks": asks})
    )

    # Two consecutive hits — gamma must be queried at most once.
    r1 = client.get(f"/terminal/book/{SLUG}")
    r2 = client.get(f"/terminal/book/{SLUG}")
    assert r1.status_code == r2.status_code == 200
    assert gamma_route.call_count == 1, (
        f"slug→token_id should be cached; gamma was called {gamma_route.call_count} times"
    )


# ---------------------------------------------------------------------------
# 2026-05-15 upstream-hardening: 429 retry on gamma + CLOB
# ---------------------------------------------------------------------------


def _unique_gamma_response(slug: str, token: str) -> httpx.Response:
    return httpx.Response(
        200,
        json=[
            {
                "slug": slug,
                "question": "Q",
                "clobTokenIds": json.dumps([token, "222"]),
                "active": True,
                "closed": False,
            }
        ],
    )


@respx.mock
def test_gamma_429_retried_once_then_succeeds(client: TestClient, monkeypatch) -> None:
    """A single 429 on Gamma is absorbed by the retry; UI sees 200, not 502.

    Uses a unique slug/token so a prior test's cached book (keyed on
    token_id) can't shadow the upstream call we mock here.
    """
    import pfm.terminal.orderbook as ob

    monkeypatch.setattr(ob, "_RETRY_BACKOFF_S", 0.01)
    slug = "fed-decision-gamma-retry"
    token = "tok-gamma-retry"

    gamma_route = respx.get(f"{GAMMA_URL}/markets", params={"slug": slug}).mock(
        side_effect=[
            httpx.Response(429),
            _unique_gamma_response(slug, token),
        ]
    )
    bids = [{"price": 0.50, "size": 100.0}]
    asks = [{"price": 0.52, "size": 100.0}]
    respx.get("https://clob.polymarket.com/book", params={"token_id": token}).mock(
        return_value=httpx.Response(200, json={"bids": bids, "asks": asks})
    )

    r = client.get(f"/terminal/book/{slug}")
    assert r.status_code == 200, r.text
    assert gamma_route.call_count == 2


@respx.mock
def test_clob_book_429_retried_once_then_succeeds(client: TestClient, monkeypatch) -> None:
    """A single 429 on CLOB /book is absorbed; UI sees 200.

    Uses a unique slug/token so a prior test's cached book (keyed on
    token_id) can't shadow the upstream call we mock here.
    """
    import pfm.terminal.orderbook as ob

    monkeypatch.setattr(ob, "_RETRY_BACKOFF_S", 0.01)
    slug = "fed-decision-clob-retry"
    token = "tok-clob-retry"

    respx.get(f"{GAMMA_URL}/markets", params={"slug": slug}).mock(
        return_value=_unique_gamma_response(slug, token)
    )
    bids = [{"price": 0.50, "size": 100.0}]
    asks = [{"price": 0.52, "size": 100.0}]
    book_route = respx.get("https://clob.polymarket.com/book", params={"token_id": token}).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json={"bids": bids, "asks": asks}),
        ]
    )

    r = client.get(f"/terminal/book/{slug}")
    assert r.status_code == 200, r.text
    assert book_route.call_count == 2
