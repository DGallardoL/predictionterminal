"""Tests for ``pfm.terminal_news_impact``. All HTTP is mocked via respx."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_gdelt_news import GDELT_DOC_URL
from pfm.terminal_news_impact import _CACHE
from pfm.terminal_news_impact import router as news_impact_router

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"
SLUG = "trump-out-by-2027"
TOKEN_YES = "111111"
TOKEN_NO = "222222"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    _CACHE.clear()
    yield
    _CACHE.clear()


@pytest.fixture
def app_client() -> TestClient:
    app = FastAPI()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    app.include_router(news_impact_router)
    return TestClient(app)


def _gamma_market_payload() -> list[dict]:
    """Realistic Gamma /markets response. clobTokenIds is a JSON STRING."""
    return [
        {
            "slug": SLUG,
            "question": "Will Trump resign or be impeached by 2027?",
            "clobTokenIds": f'["{TOKEN_YES}", "{TOKEN_NO}"]',
            "closed": False,
            "active": True,
            "startDate": "2025-01-01",
            "endDate": "2027-01-01",
        }
    ]


def _gdelt_payload() -> dict:
    """Two articles: one well-inside the price window, one near the edge."""
    return {
        "articles": [
            {
                "url": "https://www.bbc.com/news/world-1",
                "title": "Senate hearing: Trump faces impeachment vote",
                "domain": "bbc.com",
                "sourcecountry": "United Kingdom",
                "language": "English",
                "seendate": "20260315T120000Z",
                "tone": -2.5,
            },
            {
                "url": "https://www.reuters.com/politics/2",
                "title": "Trump remains defiant after hearing",
                "domain": "reuters.com",
                "sourcecountry": "United States",
                "language": "English",
                "seendate": "20260320T180000Z",
                "tone": 0.5,
            },
        ]
    }


def _hourly_history_big_move() -> dict:
    """Hourly bars where the price jumps from 0.34 → 0.42 across event 1's
    6h window (an 8 pp move) and barely budges across event 2 (flat).

    The huge move on event 1 should make σ relatively large but the
    log-return at +6h should still exceed 1.5σ given the otherwise calm
    surrounding bars.
    """
    history: list[dict] = []
    # Calm baseline: 2026-03-14 00:00..2026-03-15 11:00, price ≈ 0.34
    base_ts = 1_773_446_400  # 2026-03-14 00:00:00 UTC
    for i in range(36):  # 36 hourly bars before event 1
        history.append({"t": base_ts + i * 3600, "p": 0.34 + 0.001 * (i % 2)})

    # Event 1 at 2026-03-15T12:00:00Z. Bars +1h, +6h, +24h:
    e1_ts = 1_773_576_000
    history.append({"t": e1_ts + 3600, "p": 0.36})
    history.append({"t": e1_ts + 6 * 3600, "p": 0.42})  # 8pp jump → attributable
    # Filler hours between +6h and +24h, mild drift.
    for h in range(7, 24):
        history.append({"t": e1_ts + h * 3600, "p": 0.41})
    history.append({"t": e1_ts + 24 * 3600, "p": 0.41})

    # Calm gap until event 2 at 2026-03-20T18:00:00Z.
    e2_ts = 1_774_029_600
    fill_t = e1_ts + 25 * 3600
    while fill_t < e2_ts:
        history.append({"t": fill_t, "p": 0.41})
        fill_t += 3600
    # Event 2 reactions: barely move (flat → not attributable)
    history.append({"t": e2_ts + 3600, "p": 0.41})
    history.append({"t": e2_ts + 6 * 3600, "p": 0.412})
    history.append({"t": e2_ts + 24 * 3600, "p": 0.412})

    return {"history": history}


def _mock_gamma(slug: str = SLUG, payload: list[dict] | None = None) -> respx.Route:
    return respx.get(f"{GAMMA}/markets", params={"slug": slug}).mock(
        return_value=httpx.Response(200, json=payload or _gamma_market_payload())
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@respx.mock
def test_news_impact_happy_path_attributable_event(app_client: TestClient) -> None:
    """Two GDELT events: the 8pp move on event 1 is flagged attributable."""
    _mock_gamma()
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload()))
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json=_hourly_history_big_move())
    )

    resp = app_client.get(f"/terminal/news-impact/{SLUG}?days=30")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["slug"] == SLUG
    assert body["days"] == 30
    assert body["n_events"] == 2

    # Event 1: price_before ≈ 0.34, price_6h_after = 0.42, abs_move ~= 8pp.
    e1 = body["events"][0]
    assert e1["headline"].startswith("Senate hearing")
    assert e1["price_before"] == pytest.approx(0.34, abs=0.01)
    assert e1["price_1h_after"] == pytest.approx(0.36, abs=1e-6)
    assert e1["price_6h_after"] == pytest.approx(0.42, abs=1e-6)
    assert e1["price_24h_after"] == pytest.approx(0.41, abs=1e-6)
    assert e1["abs_move_pp"] == pytest.approx(8.0, abs=0.5)
    assert e1["direction"] == "up"
    assert e1["attributable"] is True

    # Event 2: tiny move → not attributable.
    e2 = body["events"][1]
    assert e2["attributable"] is False
    assert e2["direction"] in {"up", "flat"}

    assert body["n_attributable"] == 1
    assert body["attributable_pct"] == pytest.approx(50.0, abs=1e-6)
    assert "1 of 2 GDELT events" in body["interpretation"]
    assert "1.5-sigma" in body["interpretation"]


@respx.mock
def test_news_impact_returns_404_when_market_missing(app_client: TestClient) -> None:
    """Gamma returns an empty list → endpoint returns 404, no GDELT/CLOB calls."""
    respx.get(f"{GAMMA}/markets", params={"slug": "ghost-mkt"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    # Required by the fallback inside `get_market_metadata`.
    respx.get(f"{GAMMA}/markets", params={"slug": "ghost-mkt", "closed": "true"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    gdelt_route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"articles": []})
    )
    clob_route = respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": []})
    )

    resp = app_client.get("/terminal/news-impact/ghost-mkt")
    assert resp.status_code == 404
    assert "market" in resp.json()["detail"].lower()
    # We never called GDELT or CLOB.
    assert gdelt_route.call_count == 0
    assert clob_route.call_count == 0


@respx.mock
def test_news_impact_handles_no_gdelt_articles(app_client: TestClient) -> None:
    """Empty GDELT → 200 with zero events and a clean interpretation string."""
    _mock_gamma()
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json={"articles": []}))
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json=_hourly_history_big_move())
    )

    resp = app_client.get(f"/terminal/news-impact/{SLUG}?days=14")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_events"] == 0
    assert body["events"] == []
    assert body["n_attributable"] == 0
    assert body["attributable_pct"] == 0.0
    assert (
        body["interpretation"] == "0 of 0 GDELT events caused >1.5-sigma price moves in the next 6h"
    )


@respx.mock
def test_news_impact_degrades_when_clob_returns_empty(app_client: TestClient) -> None:
    """No price data → events still returned with null prices and not flagged."""
    _mock_gamma()
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload()))
    # CLOB returns empty history (e.g. brand-new market or data outage).
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(200, json={"history": []}))

    resp = app_client.get(f"/terminal/news-impact/{SLUG}?days=30")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["n_events"] == 2
    for e in body["events"]:
        assert e["price_before"] is None
        assert e["price_1h_after"] is None
        assert e["price_6h_after"] is None
        assert e["price_24h_after"] is None
        assert e["abs_move_pp"] is None
        assert e["direction"] == "flat"
        assert e["attributable"] is False

    assert body["n_attributable"] == 0
    assert body["attributable_pct"] == 0.0

    # Verify CLOB was called with hourly fidelity.
    clob_calls = [c for c in respx.calls if "prices-history" in str(c.request.url)]
    assert clob_calls, "expected at least one /prices-history call"
    qs = dict(clob_calls[-1].request.url.params)
    assert qs.get("fidelity") == "60"
    assert qs.get("market") == TOKEN_YES


# ---------------------------------------------------------------------------
# Cache layering: L1 + Redis L2
# ---------------------------------------------------------------------------


class _FakeRedisL2:
    """In-memory stand-in for the Redis L2 cache used by news_impact.

    Exposes the subset of methods the module actually calls
    (``get`` / ``set`` / ``enabled``) so we can assert cross-worker
    cache promotion without booting Redis in CI.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.enabled = True

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        self.store[key] = value


@respx.mock
def test_news_impact_l1_cache_hit_skips_upstreams(app_client: TestClient) -> None:
    """Second call within the TTL must serve from L1 with zero upstream traffic."""
    gamma_route = _mock_gamma()
    gdelt_route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json=_gdelt_payload())
    )
    clob_route = respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json=_hourly_history_big_move())
    )

    r1 = app_client.get(f"/terminal/news-impact/{SLUG}?days=30")
    assert r1.status_code == 200
    n1 = gamma_route.call_count + gdelt_route.call_count + clob_route.call_count

    r2 = app_client.get(f"/terminal/news-impact/{SLUG}?days=30")
    assert r2.status_code == 200
    assert r2.json() == r1.json()
    # No additional upstream call on the cached hit.
    assert gamma_route.call_count + gdelt_route.call_count + clob_route.call_count == n1


@respx.mock
def test_news_impact_l2_redis_promotes_into_l1(app_client: TestClient) -> None:
    """Pre-seed the Redis L2 cache; assert the endpoint serves from it
    without calling Gamma/GDELT/CLOB."""
    import json as _json

    fake_l2 = _FakeRedisL2()
    # Seed an answer for (slug, days=30). Shape matches what the endpoint writes.
    seeded = {
        "slug": SLUG,
        "days": 30,
        "events": [],
        "n_events": 0,
        "n_attributable": 0,
        "attributable_pct": 0.0,
        "interpretation": "seeded-from-L2",
    }
    fake_l2.store[f"terminal_news_impact:impact:{SLUG}:30"] = _json.dumps(seeded).encode("utf-8")
    app_client.app.state.cache = fake_l2

    # Mock every upstream so an accidental call raises a clear assertion.
    gamma_route = respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(500, json={"error": "should-not-be-called"})
    )
    gdelt_route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(500, json={"error": "should-not-be-called"})
    )
    clob_route = respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(500, json={"error": "should-not-be-called"})
    )

    resp = app_client.get(f"/terminal/news-impact/{SLUG}?days=30")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["interpretation"] == "seeded-from-L2"
    # Zero upstream traffic — the L2 hit short-circuited the whole pipeline.
    assert gamma_route.call_count == 0
    assert gdelt_route.call_count == 0
    assert clob_route.call_count == 0


@respx.mock
def test_news_impact_writes_to_l2_after_fresh_fetch(app_client: TestClient) -> None:
    """After a cold call, the L2 cache must hold the encoded payload."""
    fake_l2 = _FakeRedisL2()
    app_client.app.state.cache = fake_l2

    _mock_gamma()
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload()))
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json=_hourly_history_big_move())
    )

    resp = app_client.get(f"/terminal/news-impact/{SLUG}?days=30")
    assert resp.status_code == 200
    # L2 was populated.
    key = f"terminal_news_impact:impact:{SLUG}:30"
    assert key in fake_l2.store
    # Round-trip the stored bytes — it must be valid JSON matching the body.
    import json as _json

    stored = _json.loads(fake_l2.store[key])
    assert stored["slug"] == SLUG
    assert stored["n_events"] == resp.json()["n_events"]
