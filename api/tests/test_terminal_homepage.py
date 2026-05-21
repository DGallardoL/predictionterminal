"""Tests for ``pfm.terminal_homepage`` — /terminal/homepage."""

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

from pfm import terminal_homepage
from pfm.terminal_homepage import clear_cache, router

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


def _build_app() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _mk_market(
    slug: str,
    *,
    question: str | None = None,
    base: float = 0.5,
    change: float = 0.0,
    vol: float = 1000.0,
    token_id: str | None = None,
    created_days_ago: int = 100,
    end_days: int = 200,
    theme: str | None = None,
) -> dict[str, Any]:
    now = pd.Timestamp.utcnow().tz_convert("UTC")
    return {
        "slug": slug,
        "question": question or f"Will {slug} happen?",
        "clobTokenIds": json.dumps([token_id or f"tok-{slug}", "no"]),
        "bestBid": base - 0.01,
        "bestAsk": base + 0.01,
        "lastTradePrice": base,
        "volume24hr": vol,
        "volumeNum": vol * 30.0,
        "oneDayPriceChange": change,
        "oneWeekPriceChange": 0.0,
        "createdAt": (now - pd.Timedelta(days=created_days_ago)).isoformat(),
        "endDate": (now + pd.Timedelta(days=end_days)).isoformat(),
        "active": True,
        "closed": False,
        "theme": theme,
    }


def _spark_payload(seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    end_ts = int(pd.Timestamp.utcnow().normalize().timestamp())
    history = []
    p = 0.5
    for i in range(8):
        p = max(0.05, min(0.95, p + 0.01 * rng.standard_normal()))
        history.append({"t": end_ts - (7 - i) * 86400, "p": float(p)})
    return {"history": history}


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

    monkeypatch.setattr(terminal_homepage, "get_settings", _S)


def _wire_default_routes(markets: list[dict[str, Any]]) -> None:
    """Mock all upstream calls used by the homepage builder."""

    # Gamma listing — return everything on first page; empty on later pages.
    def _markets_handler(req: httpx.Request) -> httpx.Response:
        offset = int(req.url.params.get("offset") or 0)
        if offset == 0:
            return httpx.Response(200, json=markets)
        return httpx.Response(200, json=[])

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_markets_handler)
    # CLOB sparkline — single payload reused.
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_spark_payload())
    )
    # Breaking news — minimal response.
    respx.get("https://hn.algolia.com/api/v1/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": "Markets rally on Fed news",
                        "url": "https://example.com/a",
                        "objectID": "1",
                        "points": 250,
                    },
                    {
                        "title": "Tech sector gains",
                        "url": "https://example.com/b",
                        "objectID": "2",
                        "points": 100,
                    },
                ]
            },
        )
    )


# --- tests ------------------------------------------------------------------


class TestSorting:
    @respx.mock
    def test_gainers_sorted_desc_by_change(self) -> None:
        markets = [
            _mk_market("up-big", change=0.20, vol=10_000.0),
            _mk_market("up-small", change=0.05, vol=8_000.0),
            _mk_market("flat", change=0.0, vol=12_000.0),
            _mk_market("down-small", change=-0.04, vol=4_000.0),
            _mk_market("down-big", change=-0.18, vol=9_000.0),
        ]
        _wire_default_routes(markets)
        client = _build_app()
        r = client.get("/terminal/homepage?hours=24")
        assert r.status_code == 200, r.text
        body = r.json()
        gainers = body["gainers"]
        assert gainers, "expected gainers to be non-empty"
        # Strictly descending change_pct.
        changes = [g["change_pct"] for g in gainers]
        assert changes == sorted(changes, reverse=True)
        assert gainers[0]["slug"] == "up-big"
        # Sparkline attached.
        assert isinstance(gainers[0]["sparkline_7d"], list)
        assert len(gainers[0]["sparkline_7d"]) > 0

    @respx.mock
    def test_losers_sorted_asc_by_change(self) -> None:
        markets = [
            _mk_market("up-big", change=0.20, vol=10_000.0),
            _mk_market("down-big", change=-0.18, vol=9_000.0),
            _mk_market("down-small", change=-0.04, vol=4_000.0),
            _mk_market("flat", change=0.0, vol=12_000.0),
        ]
        _wire_default_routes(markets)
        client = _build_app()
        r = client.get("/terminal/homepage?hours=24")
        assert r.status_code == 200, r.text
        body = r.json()
        losers = body["losers"]
        assert losers
        changes = [lr["change_pct"] for lr in losers]
        assert changes == sorted(changes)
        assert losers[0]["slug"] == "down-big"


class TestPmVix:
    @respx.mock
    def test_pm_vix_returns_value_in_zero_to_hundred(self) -> None:
        markets = [
            _mk_market(
                "us-recession-2026",
                question="Will the US enter a recession by 2026?",
                base=0.42,
                vol=50_000.0,
            ),
            _mk_market(
                "geopolitical-conflict",
                question="Will geopolitical tensions escalate?",
                base=0.38,
                vol=30_000.0,
            ),
            # A non-tail-risk market — should be ignored by pm_vix.
            _mk_market("eth-up", question="Will ETH go up?", base=0.55, vol=80_000.0),
        ]
        _wire_default_routes(markets)
        client = _build_app()
        r = client.get("/terminal/homepage?hours=24")
        assert r.status_code == 200, r.text
        body = r.json()
        pm_vix = body["pm_vix"]
        assert isinstance(pm_vix, (int, float))
        assert 0.0 <= pm_vix <= 100.0
        # With recession at 0.42 and conflict at 0.38 the vol-weighted
        # composite should be nontrivial — strictly above zero.
        assert pm_vix > 0.0

    @respx.mock
    def test_pm_vix_zero_when_no_tail_risk_markets(self) -> None:
        markets = [
            _mk_market("eth-up", question="ETH up by Christmas?", base=0.55, vol=10_000.0),
            _mk_market("nfl-week-3", question="NFL Patriots win?", base=0.5, vol=2_000.0),
        ]
        _wire_default_routes(markets)
        client = _build_app()
        r = client.get("/terminal/homepage?hours=24")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["pm_vix"] == 0.0


class TestPayloadShape:
    @respx.mock
    def test_envelope_contains_all_sections(self) -> None:
        markets = [
            _mk_market(
                "newbie",
                created_days_ago=2,
                change=0.10,
                vol=5_000.0,
            ),
            _mk_market(
                "soon-resolve",
                end_days=3,
                change=-0.02,
                base=0.85,
                vol=20_000.0,
            ),
            _mk_market("filler", change=0.04, vol=2_000.0),
        ]
        _wire_default_routes(markets)
        client = _build_app()
        r = client.get("/terminal/homepage")
        assert r.status_code == 200, r.text
        body = r.json()
        for key in (
            "gainers",
            "losers",
            "most_active",
            "recently_launched",
            "resolving_soon",
            "breaking_news",
            "theme_heatmap",
            "pm_vix",
        ):
            assert key in body, f"missing section {key!r}"

        # New market populated (created 2 days ago).
        assert any(n["slug"] == "newbie" for n in body["recently_launched"])
        # Resolving-soon contains the 3-days-out market.
        assert any(r["slug"] == "soon-resolve" for r in body["resolving_soon"])
        # Breaking news from HN front page.
        assert len(body["breaking_news"]) >= 1
        assert body["breaking_news"][0]["title"]
