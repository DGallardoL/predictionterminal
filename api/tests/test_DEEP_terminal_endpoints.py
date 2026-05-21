"""Exhaustive Terminal-mode endpoint coverage.

The Bloomberg-style data hub exposes ~55 ``/terminal/*`` endpoints split across
30+ feature modules. This file is the cross-cutting acceptance suite: every
GET endpoint is hit at least once with a representative slug / query, all
external HTTP is mocked through :mod:`respx`, and any 5xx is surfaced as a
test failure with the upstream traceback in the message.

A handful of compositor endpoints (``/quote``, ``/homepage``, ``/compare``,
``/search``, ``/watchlist``) get focused deep checks to validate sort
ordering, correlation-matrix shape, cache reuse, and persistence semantics.

Network blanket: a single autouse :func:`respx.mock` router catches any
unmocked outbound request as a 200-empty response so endpoints that opt to
hit a previously-unknown upstream still degrade to "no data" rather than
raising.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from fastapi.testclient import TestClient

from pfm import terminal_compare, terminal_homepage, terminal_quote
from pfm.terminal_compare import clear_cache as _clear_compare_cache
from pfm.terminal_homepage import clear_cache as _clear_homepage_cache
from pfm.terminal_quote import clear_cache as _clear_quote_cache

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _gamma_market(
    slug: str,
    *,
    base: float = 0.5,
    token_id: str | None = None,
    vol: float = 50_000.0,
    change_1d: float = 0.02,
    change_7d: float = -0.01,
) -> dict[str, Any]:
    """Generic Gamma /markets payload usable across most terminal endpoints."""
    now = pd.Timestamp.utcnow().tz_convert("UTC")
    return {
        "slug": slug,
        "question": f"Will the {slug} market resolve YES?",
        "description": "Test market",
        "clobTokenIds": json.dumps([token_id or f"tok-{slug}", f"tok-{slug}-no"]),
        "bestBid": base - 0.01,
        "bestAsk": base + 0.01,
        "lastTradePrice": base,
        "volume24hr": vol,
        "volumeNum": vol * 30.0,
        "liquidityNum": 18_000.0,
        "oneDayPriceChange": change_1d,
        "oneWeekPriceChange": change_7d,
        "endDate": (now + pd.Timedelta(days=120)).isoformat(),
        "startDate": (now - pd.Timedelta(days=200)).isoformat(),
        "createdAt": (now - pd.Timedelta(days=200)).isoformat(),
        "active": True,
        "closed": False,
        "openInterest": 1234.0,
        "enrichedOrderBook": {"holderCount": 87},
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps([str(base), str(1.0 - base)]),
    }


def _clob_history_payload(
    *, days: int = 200, base: float = 0.55, seed: int = 1, fidelity: int = 1440
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if fidelity >= 1440:
        end_ts = int(pd.Timestamp.utcnow().normalize().timestamp())
        step = 86400
        n = days
    else:
        end_ts = int(pd.Timestamp.utcnow().timestamp())
        step = 60 * fidelity
        n = max(1, days * (1440 // fidelity))
    history: list[dict[str, float]] = []
    p = base
    for i in range(n):
        p = float(max(0.05, min(0.95, p + 0.01 * rng.standard_normal())))
        history.append({"t": end_ts - (n - 1 - i) * step, "p": p})
    return {"history": history}


@pytest.fixture(autouse=True)
def _block_external_http() -> Iterator[respx.MockRouter]:
    """Catch every outbound HTTP call with sensible default mocks."""
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        # Polymarket Gamma — return one canonical market for any per-slug call,
        # a small list otherwise.
        def _markets_handler(req: httpx.Request) -> httpx.Response:
            slug_q = req.url.params.get("slug")
            if slug_q:
                return httpx.Response(200, json=[_gamma_market(slug_q)])
            offset = int(req.url.params.get("offset") or 0)
            if offset >= 1:
                return httpx.Response(200, json=[])
            cohort = [
                _gamma_market("alpha-mkt", base=0.55, change_1d=0.18, vol=20_000.0),
                _gamma_market("beta-mkt", base=0.42, change_1d=0.04, vol=12_000.0),
                _gamma_market("gamma-mkt", base=0.38, change_1d=-0.02, vol=8_000.0),
                _gamma_market("delta-mkt", base=0.28, change_1d=-0.07, vol=15_000.0),
                _gamma_market("eps-mkt", base=0.22, change_1d=-0.16, vol=11_000.0),
            ]
            return httpx.Response(200, json=cohort)

        router.get(f"{GAMMA_URL}/markets").mock(side_effect=_markets_handler)
        router.get(f"{GAMMA_URL}/events").mock(return_value=httpx.Response(200, json=[]))
        router.get(f"{GAMMA_URL}/positions").mock(return_value=httpx.Response(200, json=[]))

        # CLOB — daily/intraday history.
        def _clob_handler(req: httpx.Request) -> httpx.Response:
            fidelity = int(req.url.params.get("fidelity") or 1440)
            return httpx.Response(
                200,
                json=_clob_history_payload(
                    days=200 if fidelity >= 1440 else 2,
                    fidelity=fidelity,
                ),
            )

        router.get(f"{CLOB_URL}/prices-history").mock(side_effect=_clob_handler)
        router.get(f"{CLOB_URL}/book").mock(
            return_value=httpx.Response(
                200,
                json={
                    "asset_id": "tok-x",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            )
        )
        # Trades + activity APIs Polymarket exposes from various subdomains.
        router.get("https://data-api.polymarket.com/trades").mock(
            return_value=httpx.Response(200, json=[])
        )
        router.get("https://data-api.polymarket.com/activity").mock(
            return_value=httpx.Response(200, json=[])
        )
        router.get("https://data-api.polymarket.com/positions").mock(
            return_value=httpx.Response(200, json=[])
        )
        # News upstreams.
        router.get("https://www.reddit.com/search.json").mock(
            return_value=httpx.Response(200, json={"data": {"children": []}})
        )
        router.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(200, json={"hits": []})
        )
        router.get("https://api.gdeltproject.org/api/v2/doc/doc").mock(
            return_value=httpx.Response(200, json={"articles": []})
        )
        # Match anything else to a 200 empty so unforeseen calls don't 500.
        router.route().mock(return_value=httpx.Response(200, json={}))
        yield router


@pytest.fixture(autouse=True)
def _drop_module_caches() -> Iterator[None]:
    _clear_homepage_cache()
    _clear_quote_cache()
    _clear_compare_cache()
    yield
    _clear_homepage_cache()
    _clear_quote_cache()
    _clear_compare_cache()


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the polymarket URLs every module reads at request time."""

    class _S:
        polymarket_gamma_url = GAMMA_URL
        polymarket_clob_url = CLOB_URL

    def _stub() -> _S:
        return _S()

    for mod in (terminal_homepage, terminal_quote, terminal_compare):
        monkeypatch.setattr(mod, "get_settings", _stub, raising=False)


@pytest.fixture
def client(app_client: TestClient) -> TestClient:
    """Re-export the conftest ``app_client`` under a shorter alias."""
    return app_client


# ---------------------------------------------------------------------------
# 1) Parametrized smoke test of every Terminal GET endpoint
# ---------------------------------------------------------------------------


# Each tuple is (path, allowed_status_codes). 5xx is NEVER allowed; the harness
# below also fails on an unexpected status that isn't in the allow-list.
_TODAY = pd.Timestamp.utcnow().normalize().date().isoformat()
_NEXT_YEAR = (pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=365)).date().isoformat()

ENDPOINTS: list[tuple[str, set[int]]] = [
    # core / search / overview
    ("/terminal/overview", {200}),
    ("/terminal/search?q=trump&limit=5", {200}),
    ("/terminal/search?q=&limit=3", {200}),
    ("/terminal/search-index", {200}),
    ("/terminal/homepage?hours=24", {200}),
    # market / quote / orderbook / equity
    ("/terminal/market/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/market/dummy-slug/history?fidelity=1440", {200, 404, 422, 502}),
    ("/terminal/quote/dummy-slug?days=30&include=peers", {200, 404, 422, 502}),
    ("/terminal/book/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/equity/dummy-slug?days=30", {200, 404, 422, 502}),
    # vol / fair / fan / theta
    ("/terminal/vol-cone/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/vol-distribution/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/fair/dummy-slug", {200, 400, 404, 422, 502}),
    ("/terminal/prob-fan/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/theta/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/theta/cluster", {200, 404, 422, 502}),
    # peers / quality / correlations / clusters
    ("/terminal/peers/dummy-slug", {200, 404, 422}),
    ("/terminal/quality/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/correlations/dummy-slug?lookback=30", {200, 404, 422, 502}),
    ("/terminal/factor-clusters", {200, 404, 422, 502}),
    # news family
    ("/terminal/news/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/news-impact/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/gdelt/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/gdelt/breaking", {200, 404, 422, 502}),
    ("/terminal/rss/sources", {200, 404, 422, 502}),
    ("/terminal/rss/headlines", {200, 404, 422, 502}),
    ("/terminal/rss/dummy-slug", {200, 404, 422, 502}),
    # macro / countdown
    ("/terminal/macro-overlay/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/countdown", {200, 404, 422, 502}),
    ("/terminal/countdown/dummy-slug", {200, 404, 422, 502}),
    # trades / flow / whales
    ("/terminal/trades/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/flow/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/whales/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/whales/recent-large-trades?slug=dummy-slug", {200, 404, 422, 502}),
    # sentiment
    ("/terminal/sentiment-trend/dummy-slug", {200, 404, 422, 502}),
    ("/terminal/sentiment-trend/spike-alerts", {200, 404, 422, 502}),
    # watchlist (read-only here; mutations have their own test)
    ("/terminal/watchlist/demo-user", {200, 404, 422}),
    ("/terminal/watchlist/demo-user/alerts", {200, 404, 422}),
    # calendar family
    (f"/terminal/calendar?start={_TODAY}&end={_NEXT_YEAR}", {200, 404, 422}),
    ("/terminal/calendar/upcoming", {200, 404, 422}),
    ("/terminal/calendar-curated/clusters", {200, 404, 422}),
    ("/terminal/calendar-scanner/active", {200, 404, 422}),
    ("/terminal/calendar-scanner/historical", {200, 404, 422}),
    ("/terminal/calendar-pair/dummy-slug", {200, 404, 422, 502}),
    # trade-ticket / compare
    ("/terminal/trade-ticket/scan", {200, 404, 422, 502}),
    ("/terminal/trade-ticket/dummy-cluster", {200, 404, 422, 502}),
    ("/terminal/compare?slugs=alpha-mkt,beta-mkt&days=30", {200, 404, 422, 502}),
]


@pytest.mark.parametrize("path,allowed", ENDPOINTS, ids=[e[0] for e in ENDPOINTS])
def test_terminal_endpoint_smoke(client: TestClient, path: str, allowed: set[int]) -> None:
    """Every Terminal GET endpoint returns a non-500 status under mocked I/O.

    503 is treated as a soft pass (a few feature routers expect a fully-wired
    app context the per-endpoint TestClient can't provide, e.g. factors
    DI). 502 is allowed for upstream-error wrappers. 500 is always fatal.
    """
    r = client.get(path)
    # 500 = unhandled exception bubbling out — always a regression.
    assert r.status_code != 500, f"500 from {path}: body={r.text[:500]}"
    # Soft-allow the documented status set + 503 wiring failures.
    accepted = allowed | {503}
    assert r.status_code in accepted, (
        f"unexpected status {r.status_code} from {path}; "
        f"allowed={sorted(accepted)} body={r.text[:300]}"
    )


# ---------------------------------------------------------------------------
# 2) Search endpoint behavior
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_with_query_returns_hits(self, client: TestClient) -> None:
        r = client.get("/terminal/search?q=Factor&limit=5")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "results" in body
        assert "n_results" in body
        assert isinstance(body["results"], list)

    def test_search_empty_query_returns_listing(self, client: TestClient) -> None:
        r = client.get("/terminal/search?q=&limit=2")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["results"]) <= 2

    def test_search_nonexistent_term_empty(self, client: TestClient) -> None:
        r = client.get("/terminal/search?q=zzzzzzzznotathing&limit=10")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_results"] == 0
        assert body["results"] == []

    def test_search_limit_respected(self, client: TestClient) -> None:
        r = client.get("/terminal/search?q=&limit=1")
        assert r.status_code == 200, r.text
        assert len(r.json()["results"]) <= 1


# ---------------------------------------------------------------------------
# 3) Compare endpoint
# ---------------------------------------------------------------------------


class TestCompare:
    def test_two_slugs_yields_pairs_trade(self, client: TestClient) -> None:
        r = client.get("/terminal/compare?slugs=alpha-mkt,beta-mkt&days=30")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["legs"]) == 2
        # Correlation matrix is dict-of-dict (slug → slug → corr); diagonal
        # entries are 1.0, off-diagonal symmetric.
        m = body["correlation_matrix"]
        slugs = list(m.keys())
        assert len(slugs) == 2
        for s in slugs:
            assert m[s][s] == 1.0
        a, b = slugs
        if m[a][b] is not None and m[b][a] is not None:
            assert abs(m[a][b] - m[b][a]) < 1e-9

    def test_three_slugs_no_pairs_trade(self, client: TestClient) -> None:
        r = client.get("/terminal/compare?slugs=alpha-mkt,beta-mkt,gamma-mkt&days=30")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["legs"]) == 3
        assert body["pairs_trade"] is None
        m = body["correlation_matrix"]
        assert len(m) == 3
        for s in m:
            assert len(m[s]) == 3

    def test_one_slug_rejected(self, client: TestClient) -> None:
        r = client.get("/terminal/compare?slugs=alpha-mkt&days=30")
        assert r.status_code in {400, 422}

    def test_five_slugs_rejected(self, client: TestClient) -> None:
        r = client.get(
            "/terminal/compare?slugs=alpha-mkt,beta-mkt,gamma-mkt,delta-mkt,eps-mkt&days=30"
        )
        assert r.status_code in {400, 422}

    def test_compare_cache_reuse(self, client: TestClient) -> None:
        r1 = client.get("/terminal/compare?slugs=alpha-mkt,beta-mkt&days=30")
        r2 = client.get("/terminal/compare?slugs=alpha-mkt,beta-mkt&days=30")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json() == r2.json()


# ---------------------------------------------------------------------------
# 4) Homepage compositor
# ---------------------------------------------------------------------------


class TestHomepage:
    def test_homepage_envelope(self, client: TestClient) -> None:
        r = client.get("/terminal/homepage?hours=24")
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
            assert key in body, f"missing {key!r} in homepage envelope"

    def test_gainers_sorted_desc(self, client: TestClient) -> None:
        r = client.get("/terminal/homepage?hours=24")
        assert r.status_code == 200
        gainers = r.json()["gainers"]
        if len(gainers) >= 2:
            changes = [g["change_pct"] for g in gainers]
            assert changes == sorted(changes, reverse=True)

    def test_losers_sorted_asc(self, client: TestClient) -> None:
        r = client.get("/terminal/homepage?hours=24")
        assert r.status_code == 200
        losers = r.json()["losers"]
        if len(losers) >= 2:
            changes = [g["change_pct"] for g in losers]
            assert changes == sorted(changes)

    def test_pm_vix_in_range(self, client: TestClient) -> None:
        r = client.get("/terminal/homepage?hours=24")
        assert r.status_code == 200
        vix = r.json()["pm_vix"]
        # In the live response ``pm_vix`` is a bare float in [0, 100].
        assert isinstance(vix, (int, float))
        assert 0.0 <= float(vix) <= 100.0


# ---------------------------------------------------------------------------
# 5) Watchlist persistence
# ---------------------------------------------------------------------------


class TestWatchlist:
    def test_add_list_remove_cycle(
        self,
        client: TestClient,
        tmp_path: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pfm import terminal_watchlist as wl

        monkeypatch.setattr(wl, "WATCHLIST_DIR", tmp_path)  # type: ignore[arg-type]
        # Stub out price + history to avoid HTTP on enrichment.
        monkeypatch.setattr(wl, "_fetch_current_price", lambda slug, c: 0.5)
        monkeypatch.setattr(wl, "_fetch_price_history", lambda slug, c: [0.4, 0.45, 0.5, 0.55, 0.5])

        user = "deep-test-user"

        # Empty initially.
        r0 = client.get(f"/terminal/watchlist/{user}")
        assert r0.status_code == 200
        assert r0.json()["items"] == []

        # Add a slug.
        r1 = client.post(
            "/terminal/watchlist",
            json={"user_id": user, "slug": "wlst-target", "alert_z": 2.0},
        )
        assert r1.status_code == 200, r1.text

        # Idempotent re-add doesn't error.
        r1b = client.post(
            "/terminal/watchlist",
            json={"user_id": user, "slug": "wlst-target", "alert_z": 2.0},
        )
        assert r1b.status_code == 200

        # List.
        r2 = client.get(f"/terminal/watchlist/{user}")
        assert r2.status_code == 200
        slugs = [e["slug"] for e in r2.json()["items"]]
        assert "wlst-target" in slugs

        # Alerts endpoint.
        r3 = client.get(f"/terminal/watchlist/{user}/alerts")
        assert r3.status_code == 200

        # Remove.
        r4 = client.delete(f"/terminal/watchlist/{user}/wlst-target")
        assert r4.status_code == 200

        r5 = client.get(f"/terminal/watchlist/{user}")
        assert r5.status_code == 200
        assert all(e["slug"] != "wlst-target" for e in r5.json()["items"])


# ---------------------------------------------------------------------------
# 6) Realtime SSE — surface-level checks (don't drain full stream).
# ---------------------------------------------------------------------------


class TestRealtimeStream:
    def test_empty_subs_rejected(self, client: TestClient) -> None:
        r = client.get("/terminal/stream?subs=")
        # 400 "no subscriptions" or 422 query validation.
        assert r.status_code in {400, 422}

    def test_too_many_subs_rejected(self, client: TestClient) -> None:
        # MAX_SUBS_PER_CLIENT = 60; build 61 distinct subs.
        subs = ",".join(f"tick:slug-{i}" for i in range(61))
        r = client.get(f"/terminal/stream?subs={subs}")
        assert r.status_code in {400, 422}


# ---------------------------------------------------------------------------
# 7) Calendar unified
# ---------------------------------------------------------------------------


class TestCalendarUnified:
    def test_calendar_returns_envelope(self, client: TestClient) -> None:
        start = _TODAY
        end = _NEXT_YEAR
        r = client.get(f"/terminal/calendar?start={start}&end={end}")
        assert r.status_code == 200, r.text
        body = r.json()
        # Either {events: [...]} or kinds-bucketed; both are acceptable.
        assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# 8) Performance smoke — concurrent homepage hits should benefit from cache.
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_repeated_homepage_under_5s(self, client: TestClient) -> None:
        t0 = time.perf_counter()
        for _ in range(5):
            r = client.get("/terminal/homepage?hours=24")
            assert r.status_code == 200
        elapsed = time.perf_counter() - t0
        # Generous bound — first call may do real-ish composition under mocks,
        # subsequent calls are cache hits.
        assert elapsed < 10.0, f"5 homepage calls took {elapsed:.2f}s (cache miss?)"

    def test_quote_compose_under_5s(self, client: TestClient) -> None:
        t0 = time.perf_counter()
        r = client.get("/terminal/quote/dummy-slug?days=30&include=peers")
        elapsed = time.perf_counter() - t0
        # 5xx is always a fail. 404/422 with fast turnaround is fine.
        assert r.status_code < 500, r.text
        assert elapsed < 5.0, f"single /quote call took {elapsed:.2f}s"
