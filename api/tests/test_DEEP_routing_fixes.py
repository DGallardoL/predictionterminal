"""Routing / error-handling regression tests for the live-server bug report.

Live testers found six rough edges in 2026-05-08 smoke tests:

  1. ``GET /terminal/orderbook/{slug}`` 404 — only ``/terminal/book/{slug}``
     was wired even though the more discoverable name is what users try.
  2. ``GET /terminal/rss-news?q=...`` 404 — only ``/terminal/rss/headlines``
     was wired; the friendlier alias path didn't exist.
  3. ``GET /terminal/vol-cone/{bad_slug}`` returned 502 instead of 404 when
     the upstream Polymarket Gamma API said "no market found".
  4. Same problem on ``GET /terminal/macro-overlay/{bad_slug}``.
  5. ``GET /terminal/peers/{slug}`` 404'd with
     ``"alpha-hunter sweep cache is empty or missing"`` whenever the cache
     file wasn't on disk — should degrade gracefully instead.
  6. ``GET /`` should 307 to ``/ui/`` and ``/ui/`` should serve HTML.

Every external HTTP call here is patched (respx for orderbook/rss-news and
monkeypatch for the polymarket fetcher) so the tests are hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_macro_overlay as macro_mod
from pfm import terminal_peer_scanner as peer_mod
from pfm import terminal_vol_cone as vc_mod
from pfm.sources.polymarket import PolymarketClient, PolymarketError
from pfm.terminal_macro_overlay import get_polymarket_client as macro_dep
from pfm.terminal_macro_overlay import router as macro_router
from pfm.terminal_orderbook import GAMMA_URL as ORDERBOOK_GAMMA
from pfm.terminal_orderbook import router as orderbook_router
from pfm.terminal_peer_scanner import clear_cache
from pfm.terminal_peer_scanner import router as peer_router
from pfm.terminal_rss_news import (
    _CACHE as RSS_CACHE,
)
from pfm.terminal_rss_news import (
    SOURCES,
)
from pfm.terminal_rss_news import router as rss_router
from pfm.terminal_vol_cone import _get_polymarket_client_dep as vc_dep
from pfm.terminal_vol_cone import router as vc_router

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stub_poly_dep() -> object:
    """Standalone object() factory for FastAPI dependency_overrides.

    Lifted out of each test so ruff PLW0108 (unnecessary lambda) stays happy.
    """
    return object()


@pytest.fixture(autouse=True)
def _clear_rss_cache() -> None:
    RSS_CACHE.clear()
    yield
    RSS_CACHE.clear()


@pytest.fixture
def orderbook_client() -> TestClient:
    app = FastAPI()
    app.include_router(orderbook_router)
    return TestClient(app)


@pytest.fixture
def rss_client() -> TestClient:
    app = FastAPI()
    # rss_news reads app.state.poly via Depends; supply a minimal stub.
    app.state.poly = PolymarketClient(
        gamma_url="https://gamma.test", clob_url="https://clob.test", client=httpx.Client()
    )
    app.include_router(rss_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1) /terminal/orderbook/{slug} alias
# ---------------------------------------------------------------------------


_GAMMA_OK = httpx.Response(
    200,
    json=[
        {
            "slug": "btc-ath-2026",
            "question": "BTC ATH by June 30 2026?",
            "clobTokenIds": json.dumps(["111", "222"]),
            "active": True,
            "closed": False,
        }
    ],
)

_BOOK_OK = httpx.Response(
    200,
    json={
        "bids": [{"price": "0.55", "size": "100"}, {"price": "0.54", "size": "50"}],
        "asks": [{"price": "0.56", "size": "80"}, {"price": "0.57", "size": "40"}],
    },
)


@respx.mock
def test_orderbook_alias_route_returns_200(orderbook_client: TestClient) -> None:
    """The new ``/terminal/orderbook/{slug}`` alias hits the same handler."""
    respx.get(f"{ORDERBOOK_GAMMA}/markets", params={"slug": "btc-ath-2026"}).mock(
        return_value=_GAMMA_OK
    )
    respx.get("https://clob.polymarket.com/book").mock(return_value=_BOOK_OK)

    r = orderbook_client.get("/terminal/orderbook/btc-ath-2026")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slug"] == "btc-ath-2026"
    assert body["token_id"] == "111"
    # The legacy /book path must keep working too.
    r2 = orderbook_client.get("/terminal/book/btc-ath-2026")
    assert r2.status_code == 200, r2.text


# ---------------------------------------------------------------------------
# 2) /terminal/rss-news alias with optional q filter
# ---------------------------------------------------------------------------


_RSS_FIXTURE_BTC = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
  <item>
    <title>Bitcoin breaks 100k milestone</title>
    <link>https://test.example/btc</link>
    <pubDate>Fri, 02 May 2026 12:00:00 GMT</pubDate>
    <description>Crypto rallies hard.</description>
  </item>
  <item>
    <title>Tariff anxiety hits markets</title>
    <link>https://test.example/tariff</link>
    <pubDate>Fri, 02 May 2026 11:00:00 GMT</pubDate>
    <description>Stocks sell off.</description>
  </item>
</channel></rss>
"""


def _mock_every_rss_source(content: bytes = _RSS_FIXTURE_BTC) -> None:
    for src in SOURCES:
        respx.get(src.url).mock(return_value=httpx.Response(200, content=content))


@respx.mock
def test_rss_news_alias_basic_returns_200(rss_client: TestClient) -> None:
    """``/terminal/rss-news`` mirrors ``/headlines`` shape."""
    _mock_every_rss_source()
    r = rss_client.get("/terminal/rss-news?limit=5")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "items" in body
    assert "n_items" in body
    assert body["n_items"] >= 1


@respx.mock
def test_rss_news_alias_q_filter_narrows_results(rss_client: TestClient) -> None:
    """``q=bitcoin`` keeps only headlines whose title/desc mentions bitcoin."""
    _mock_every_rss_source()
    r = rss_client.get("/terminal/rss-news?q=bitcoin&limit=20")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_items"] >= 1
    for item in body["items"]:
        haystack = (item["title"] + " " + item["description"]).lower()
        assert "bitcoin" in haystack


# ---------------------------------------------------------------------------
# 3) /terminal/vol-cone/{slug} → 404 on PolymarketError "no market found"
# ---------------------------------------------------------------------------


def test_vol_cone_no_market_found_yields_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad slug must surface as 404 (client error), not 502 (server error)."""

    def _boom(*_a: Any, **_kw: Any) -> pd.DataFrame:
        raise PolymarketError("no market found for slug='ghost-slug'")

    monkeypatch.setattr(vc_mod, "fetch_factor_history", _boom)
    app = FastAPI()
    app.include_router(vc_router)
    app.dependency_overrides[vc_dep] = _stub_poly_dep
    with TestClient(app) as client:
        r = client.get("/terminal/vol-cone/ghost-slug")
    assert r.status_code == 404, r.text
    assert "ghost-slug" in r.json()["detail"]


def test_vol_cone_other_polymarket_error_still_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-not-found PolymarketError keeps mapping to 502 (service down)."""

    def _boom(*_a: Any, **_kw: Any) -> pd.DataFrame:
        raise PolymarketError("rate limit exceeded")

    monkeypatch.setattr(vc_mod, "fetch_factor_history", _boom)
    app = FastAPI()
    app.include_router(vc_router)
    app.dependency_overrides[vc_dep] = _stub_poly_dep
    with TestClient(app) as client:
        r = client.get("/terminal/vol-cone/some-slug")
    assert r.status_code == 502


# ---------------------------------------------------------------------------
# 4) /terminal/macro-overlay/{slug} → 404 on "no market found"
# ---------------------------------------------------------------------------


def test_macro_overlay_no_market_found_yields_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_kw: Any) -> pd.DataFrame:
        raise PolymarketError("no market found for slug='ghost-slug'")

    monkeypatch.setattr(macro_mod, "fetch_factor_history", _boom)
    app = FastAPI()
    app.include_router(macro_router)
    app.dependency_overrides[macro_dep] = _stub_poly_dep
    with TestClient(app) as client:
        r = client.get("/terminal/macro-overlay/ghost-slug?days=30")
    assert r.status_code == 404, r.text
    assert "ghost-slug" in r.json()["detail"]


def test_macro_overlay_other_polymarket_error_keeps_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-not-found PolymarketError must still hit 502 (preserves contract)."""

    def _boom(*_a: Any, **_kw: Any) -> pd.DataFrame:
        raise PolymarketError("market not found")  # different phrasing on purpose

    monkeypatch.setattr(macro_mod, "fetch_factor_history", _boom)
    app = FastAPI()
    app.include_router(macro_router)
    app.dependency_overrides[macro_dep] = _stub_poly_dep
    with TestClient(app) as client:
        r = client.get("/terminal/macro-overlay/btc_ath_jun?days=30")
    assert r.status_code == 502, r.text


# ---------------------------------------------------------------------------
# 5) /terminal/peers/{slug} graceful degrade
# ---------------------------------------------------------------------------


def _factors_fixture() -> dict[str, dict[str, str]]:
    return {
        "alpha_slug": {"name": "Alpha Macro Market", "theme": "macro", "slug": "alpha-mkt"},
        "beta_slug": {"name": "Beta Macro Market", "theme": "macro", "slug": "beta-mkt"},
        "gamma_slug": {"name": "Gamma Crypto Market", "theme": "crypto", "slug": "gamma-mkt"},
    }


def test_peers_degraded_mode_when_cache_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty hits cache → 200 with degraded_mode=true and theme-matched peers."""
    clear_cache()
    monkeypatch.setattr(peer_mod, "_load_hits", lambda: [])
    monkeypatch.setattr(peer_mod, "_load_factors", _factors_fixture)
    monkeypatch.setattr(peer_mod, "_load_tiers", lambda: {})

    app = FastAPI()
    app.include_router(peer_router)
    with TestClient(app) as client:
        r = client.get("/terminal/peers/alpha_slug")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["degraded_mode"] is True
    assert body["reason"] == "alpha_hunter_cache_unavailable"
    # beta_slug is the only same-theme peer in the fixture.
    peer_ids = {p["peer_slug"] for p in body["peers"]}
    assert "beta_slug" in peer_ids
    assert "gamma_slug" not in peer_ids
    # Each degraded record uses the agreed sentinel verdict + tier.
    for p in body["peers"]:
        assert p["verdict"] == "DEGRADED_THEME_MATCH"
        assert p["tier"] == "UNRANKED"
        assert p["oos_sharpe"] is None


def test_peers_normal_mode_marks_degraded_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populated hits cache + matching slug → degraded_mode=false.

    With the recent UX fix, a slug with *no* peers in a populated sweep is
    flagged ``degraded_mode=true`` with a human-readable reason, so this
    test now asserts the non-empty path: a slug that's actually one of the
    two legs of the hit returns peers AND ``degraded_mode=false``.
    """
    clear_cache()
    # Also clear the TERMINAL_CACHE response layer — earlier tests in this
    # file populate it under various slug keys; without a clear we'd serve
    # a stale degraded payload instead of computing fresh.
    from pfm import terminal as _term_mod

    _term_mod.TERMINAL_CACHE.clear()
    one_hit = [
        {
            "a_id": "alpha_slug",
            "b_id": "beta_slug",
            "verdict": "REAL_ALPHA",
            "n_obs": 100,
            "adf_pvalue": 0.01,
            "half_life_days": 3.0,
            "beta_hedge": 0.4,
            "oos_sharpe": 2.5,
            "perm_p": 0.0,
            "sweep": "macro",
        }
    ]
    monkeypatch.setattr(peer_mod, "_load_hits", lambda: one_hit)
    monkeypatch.setattr(peer_mod, "_load_factors", _factors_fixture)
    monkeypatch.setattr(peer_mod, "_load_tiers", lambda: {})

    app = FastAPI()
    app.include_router(peer_router)
    with TestClient(app) as client:
        # ``alpha_slug`` IS the a-leg of the hit, so peers > 0.
        r = client.get("/terminal/peers/alpha_slug")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_peers"] > 0
    assert body["degraded_mode"] is False
    assert body["reason"] is None


# ---------------------------------------------------------------------------
# 6) Root redirect + /ui/ HTML mount
# ---------------------------------------------------------------------------


def test_root_redirects_to_ui(app_client: TestClient) -> None:
    """``GET /`` returns 307 to ``/ui/`` (no auto-follow)."""
    r = app_client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui/"


def test_ui_root_serves_html_when_frontend_present(app_client: TestClient) -> None:
    """If ``web/index.html`` exists in the repo, ``/ui/`` serves it as HTML."""
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if not web_dir.exists() or not (web_dir / "index.html").exists():
        pytest.skip("web/ frontend not present in this checkout")
    r = app_client.get("/ui/", follow_redirects=False)
    assert r.status_code == 200, r.text
    assert "text/html" in r.headers["content-type"].lower()
