"""Tests for ``pfm.terminal_quote`` — /terminal/quote/{slug}.

External HTTP (Gamma + CLOB + Reddit + HN) is mocked via :mod:`respx`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_quote
from pfm.terminal_quote import clear_cache, router

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


def _build_app() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _gamma_payload(
    slug: str,
    *,
    token_id: str = "tok-x",
    base: float = 0.55,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "question": f"Will the prediction-market quoted slug {slug} happen by year-end?",
        "description": "Test market.",
        "clobTokenIds": json.dumps([token_id, f"{token_id}_no"]),
        "bestBid": base - 0.01,
        "bestAsk": base + 0.01,
        "lastTradePrice": base,
        "volume24hr": 50_000.0,
        "volumeNum": 1_500_000.0,
        "liquidityNum": 18_000.0,
        "oneDayPriceChange": 0.03,
        "oneWeekPriceChange": -0.01,
        "endDate": "2026-12-31T00:00:00Z",
        "startDate": "2025-01-01T00:00:00Z",
        "createdAt": "2025-06-01T00:00:00Z",
        "active": True,
        "closed": False,
        "openInterest": 1234.0,
        "enrichedOrderBook": {
            "holderCount": 87,
        },
    }


def _clob_history(
    *,
    days: int,
    base: float = 0.55,
    seed: int = 1,
    fidelity: int = 1440,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if fidelity >= 1440:
        end_ts = int(pd.Timestamp.utcnow().normalize().timestamp())
        step = 86400
        n = days
    else:
        end_ts = int(pd.Timestamp.utcnow().timestamp())
        step = 60 * fidelity
        n = days * (1440 // fidelity)
    history = []
    p = base
    for i in range(n):
        p = max(0.05, min(0.95, p + 0.01 * rng.standard_normal()))
        ts = end_ts - (n - 1 - i) * step
        history.append({"t": ts, "p": float(p)})
    return {"history": history}


def _mock_basic(slug: str, token_id: str = "tok-x") -> None:
    """Wire Gamma + both CLOB calls (daily + intraday).

    A single ``/markets`` route handles both the per-slug lookup AND the
    ``active=true`` listing used by the similar-markets fetcher; respx
    matches on path so we discriminate on params inside the handler.
    """
    listing = [
        _gamma_payload("alt-market-1", token_id="tok-alt-1", base=0.41),
        _gamma_payload("alt-market-2", token_id="tok-alt-2", base=0.6),
    ]

    def _markets_handler(req: httpx.Request) -> httpx.Response:
        slug_q = req.url.params.get("slug")
        if slug_q == slug:
            return httpx.Response(200, json=[_gamma_payload(slug, token_id=token_id)])
        if slug_q:
            # Any other per-slug call → empty (won't be hit in these tests).
            return httpx.Response(200, json=[])
        # Listing call — return the cohort.
        return httpx.Response(200, json=listing)

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_markets_handler)
    respx.get(f"{CLOB_URL}/prices-history").mock(side_effect=_clob_response_for)


def _clob_response_for(req: httpx.Request) -> httpx.Response:
    """Return daily or hourly history depending on requested fidelity."""
    fidelity = int(req.url.params.get("fidelity") or 1440)
    if fidelity >= 1440:
        return httpx.Response(200, json=_clob_history(days=400, fidelity=1440, seed=1))
    return httpx.Response(200, json=_clob_history(days=2, fidelity=60, seed=2))


@pytest.fixture(autouse=True)
def _drop_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    class _S:
        polymarket_gamma_url = GAMMA_URL
        polymarket_clob_url = CLOB_URL

    monkeypatch.setattr(terminal_quote, "get_settings", _S)


@pytest.fixture(autouse=True)
def _stub_peers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The peer module reads from disk by default — short-circuit to fixtures."""

    def _fake_peers(slug: str, *, top_n: int = 5) -> list[dict[str, Any]]:
        return [
            {"peer_id": "peer-a", "oos_sharpe": 1.2, "half_life_days": 9.5},
            {"peer_id": "peer-b", "oos_sharpe": 0.9, "half_life_days": 11.0},
        ]

    monkeypatch.setattr(terminal_quote.terminal_mod, "find_peers", _fake_peers)


# --- tests ------------------------------------------------------------------


class TestQuoteFullStruct:
    @respx.mock
    def test_returns_full_envelope_with_default_includes(self) -> None:
        _mock_basic("fed-cut-2026")
        # Reddit + HN responses for news.
        respx.get("https://www.reddit.com/search.json").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "children": [
                            {
                                "data": {
                                    "title": "Fed signals positive growth",
                                    "permalink": "/r/x/comments/abc/",
                                    "created_utc": 1735776000,
                                    "score": 42,
                                }
                            }
                        ]
                    }
                },
            )
        )
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "title": "Bond market crash spreads",
                            "url": "https://example.com/bonds",
                            "created_at": "2026-05-08T00:00:00Z",
                            "objectID": "1",
                        }
                    ]
                },
            )
        )

        client = _build_app()
        r = client.get("/terminal/quote/fed-cut-2026?days=30")
        assert r.status_code == 200, r.text
        body = r.json()

        # Shape checks for the major sections.
        assert body["slug"] == "fed-cut-2026"
        assert body["days"] == 30
        for key in (
            "live",
            "meta",
            "stats",
            "day_range",
            "week52_range",
            "implied_vol",
            "holders_estimate",
            "sparkline_30d",
            "sparkline_intraday",
            "peers",
            "news",
            "similar_markets",
        ):
            assert key in body, f"missing top-level key {key!r}"

        # Live block populated.
        assert body["live"]["price"] is not None
        assert body["live"]["best_bid"] is not None
        assert body["live"]["spread_cents"] is not None

        # Meta block carries title/theme/dates.
        assert "Fed" in body["meta"]["title"] or "prediction-market" in body["meta"]["title"]
        assert body["meta"]["end_date"]
        assert body["meta"]["total_open_interest"] == 1234.0

        # Stats include rv_30d and dfa_alpha when daily series is long enough.
        assert body["stats"]["n_obs"] > 30

        # Holders parsed from enrichedOrderBook.
        assert body["holders_estimate"] == 87

        # Sparklines have data.
        assert len(body["sparkline_30d"]) == 30
        assert len(body["sparkline_intraday"]) > 0

        # Peers (default include) populated from the stub.
        assert len(body["peers"]) == 2
        assert body["peers"][0]["slug"] == "peer-a"

        # implied_vol is rv_30d * sqrt(365) — finite and positive.
        if body["implied_vol"] is not None:
            assert body["implied_vol"] > 0


class TestIncludeFiltering:
    @respx.mock
    def test_include_peers_only_excludes_news_and_similar(self) -> None:
        _mock_basic("solo-peers")
        # No reddit/hn mocks — if news ran the request would fail.

        client = _build_app()
        r = client.get("/terminal/quote/solo-peers?days=30&include=peers")
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["includes"] == ["peers"]
        # peers populated.
        assert len(body["peers"]) == 2
        # news + similar empty.
        assert body["news"] == []
        assert body["similar_markets"] == []


class TestCacheHit:
    @respx.mock
    def test_second_request_served_from_cache(self) -> None:
        gamma_route = respx.get(f"{GAMMA_URL}/markets").mock(
            side_effect=lambda req: httpx.Response(200, json=[_gamma_payload("cache-me")])
        )
        clob_route = respx.get(f"{CLOB_URL}/prices-history").mock(side_effect=_clob_response_for)

        client = _build_app()
        r1 = client.get("/terminal/quote/cache-me?days=30&include=peers")
        assert r1.status_code == 200, r1.text
        first_gamma_calls = gamma_route.call_count
        first_clob_calls = clob_route.call_count

        r2 = client.get("/terminal/quote/cache-me?days=30&include=peers")
        assert r2.status_code == 200, r2.text
        # Second request must NOT have hit upstream.
        assert gamma_route.call_count == first_gamma_calls
        assert clob_route.call_count == first_clob_calls
        # Bodies match.
        assert r1.json() == r2.json()


class TestErrorHandling:
    @respx.mock
    def test_unknown_slug_returns_404(self) -> None:
        # Both per-slug forms (default + closed=true fallback) are empty.
        def _markets_handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[])

        respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_markets_handler)

        client = _build_app()
        r = client.get("/terminal/quote/ghost?days=30&include=peers")
        assert r.status_code == 404
        assert "ghost" in r.json()["detail"]
