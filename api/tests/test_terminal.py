"""Tests for the /terminal/* data-hub endpoints.

All HTTP IO is mocked via respx; no real Polymarket / network calls. The
shared ``app_client`` fixture (conftest.py) already patches yfinance and
forces NullCache; we layer respx on top for Gamma + CLOB.
"""

from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest
import respx
from fastapi.testclient import TestClient

import pfm.terminal as terminal_mod

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


@pytest.fixture(autouse=True)
def _clear_terminal_cache() -> None:
    """Force every test to hit the wired-up logic, not stale cache entries."""
    terminal_mod.TERMINAL_CACHE.clear()


@pytest.fixture
def patched_terminal_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the on-disk pickle + AH JSON with in-memory fixtures."""
    import numpy as np

    rng = pd.date_range("2025-12-01", periods=120, freq="D")
    rng_state = np.random.default_rng(0)
    walk_a = np.cumsum(rng_state.normal(0, 0.01, 120))
    walk_b = np.cumsum(rng_state.normal(0, 0.01, 120))
    series_a = pd.Series((0.40 + walk_a).clip(0.05, 0.95), index=rng)
    series_b = pd.Series((0.50 + walk_b).clip(0.05, 0.95), index=rng)

    fake_history = {
        "slug-a": series_a,
        "slug-b": series_b,
    }
    fake_hits = [
        {
            "a_id": "factor_a",
            "b_id": "factor_b",
            "verdict": "REAL_ALPHA",
            "n_obs": 100,
            "adf_pvalue": 0.01,
            "half_life_days": 5.0,
            "beta_hedge": 1.1,
            "oos_sharpe": 2.5,
            "full_sharpe": 1.8,
            "perm_p": 0.01,
            "sweep": "test",
        },
        {
            "a_id": "factor_a",
            "b_id": "factor_c",
            "verdict": "WEAK_ALPHA",
            "n_obs": 80,
            "adf_pvalue": 0.04,
            "half_life_days": 12.0,
            "beta_hedge": 0.5,
            "oos_sharpe": 1.2,
            "full_sharpe": 0.9,
            "perm_p": 0.05,
            "sweep": "test",
        },
    ]

    monkeypatch.setattr(
        terminal_mod,
        "_load_factor_history_cache",
        lambda _path: fake_history,
    )
    monkeypatch.setattr(
        terminal_mod,
        "_load_ah_hits",
        lambda _path: fake_hits,
    )


# --- /terminal/market/{slug} ------------------------------------------------


@respx.mock
def test_terminal_market_returns_live_meta_stats_peers(
    app_client: TestClient, patched_terminal_data: None
) -> None:
    respx.get(f"{GAMMA}/markets", params={"slug": "slug-a"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "slug-a",
                    "question": "Will A happen?",
                    "description": "A test market.",
                    "bestBid": 0.42,
                    "bestAsk": 0.46,
                    "lastTradePrice": 0.44,
                    "volume24hr": 12345.6,
                    "volumeNum": 99999.9,
                    "liquidity": 5000.0,
                    "liquidityNum": 5000.0,
                    "oneDayPriceChange": 0.03,
                    "oneWeekPriceChange": -0.05,
                    "endDate": "2026-12-31T00:00:00Z",
                    "startDate": "2025-01-01T00:00:00Z",
                    "createdAt": "2025-09-01T00:00:00Z",
                    "active": True,
                    "closed": False,
                    "outcomePrices": json.dumps(["0.44", "0.56"]),
                    "clobTokenIds": json.dumps(["111", "222"]),
                }
            ],
        )
    )
    r = app_client.get("/terminal/market/slug-a")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slug"] == "slug-a"
    assert body["meta"]["question"] == "Will A happen?"
    # factor_a → theme=politics in conftest? Actually the conftest factors
    # are theme-less so theme falls back to "other"/None (no theme key).
    # bidirectional bid+ask gives midpoint 0.44 with spread 4 cents.
    assert abs(body["live"]["midpoint"] - 0.44) < 1e-9
    assert abs(body["live"]["spread_cents"] - 4.0) < 1e-9
    assert body["live"]["volume_24hr"] == 12345.6
    # Stats: real series of length 120 should yield n_obs > 100.
    assert body["stats"]["n_obs"] >= 100
    # Peers come from the AH cache → factor_a has two neighbors.
    peer_ids = [p["peer_id"] for p in body["peers"]]
    assert "factor_b" in peer_ids


@respx.mock
def test_terminal_market_404_when_gamma_returns_empty(
    app_client: TestClient, patched_terminal_data: None
) -> None:
    respx.get(f"{GAMMA}/markets", params={"slug": "missing"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    r = app_client.get("/terminal/market/missing")
    assert r.status_code == 404


# --- /terminal/market/{slug}/history ----------------------------------------


@respx.mock
def test_terminal_history_passes_through_clob(
    app_client: TestClient, patched_terminal_data: None
) -> None:
    respx.get(f"{GAMMA}/markets", params={"slug": "slug-a"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "slug-a",
                    "question": "?",
                    "clobTokenIds": json.dumps(["111", "222"]),
                    "active": True,
                    "closed": False,
                }
            ],
        )
    )
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(
            200,
            json={
                "history": [
                    {"t": 1735689600, "p": 0.40},
                    {"t": 1735776000, "p": 0.42},
                ]
            },
        )
    )
    r = app_client.get("/terminal/market/slug-a/history?fidelity=1440")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["yes_token_id"] == "111"
    assert body["fidelity"] == 1440
    assert body["n_bars"] == 2
    assert body["history"][0]["p"] == 0.40


# --- /terminal/overview -----------------------------------------------------


@respx.mock
def test_terminal_overview_aggregates_buckets(
    app_client: TestClient, patched_terminal_data: None
) -> None:
    # Compute end / created dates relative to *now* so the assertions
    # below remain valid as wall-clock time advances. The "upcoming
    # resolutions" bucket has a 7-day window — both legs must fall
    # inside that range.
    from datetime import UTC, datetime, timedelta

    now = datetime.now(tz=UTC)
    end_a = (now + timedelta(days=4)).strftime("%Y-%m-%dT00:00:00Z")
    end_b = (now + timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    created_a = (now - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    created_b = (now - timedelta(days=300)).strftime("%Y-%m-%dT00:00:00Z")

    # Build one Gamma page; every subsequent paginated request returns empty.
    page_one = [
        {
            "slug": "slug-a",
            "question": "Big mover?",
            "bestBid": 0.40,
            "bestAsk": 0.42,
            "lastTradePrice": 0.41,
            "volume24hr": 50_000.0,
            "volumeNum": 1_000_000.0,
            "oneDayPriceChange": 0.10,
            "endDate": end_a,
            "createdAt": created_a,
            "outcomePrices": json.dumps(["0.41", "0.59"]),
            "active": True,
            "closed": False,
        },
        {
            "slug": "slug-b",
            "question": "Quiet but soon-to-resolve",
            "bestBid": 0.85,
            "bestAsk": 0.87,
            "lastTradePrice": 0.86,
            "volume24hr": 1_000.0,
            "volumeNum": 100_000.0,
            "oneDayPriceChange": -0.01,
            "endDate": end_b,
            "createdAt": created_b,
            "outcomePrices": json.dumps(["0.86", "0.14"]),
            "active": True,
            "closed": False,
        },
    ]
    route = respx.get(f"{GAMMA}/markets").mock(
        side_effect=[
            httpx.Response(200, json=page_one),
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[]),
        ]
    )
    r = app_client.get("/terminal/overview?pages=5")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_markets_considered"] == 2
    # Most-traded should rank slug-a above slug-b on volume.
    assert body["most_traded"][0]["slug"] == "slug-a"
    # Top movers requires |chg| ≥ default min volume of 5k → only slug-a qualifies.
    assert [m["slug"] for m in body["top_movers"]] == ["slug-a"]
    # Upcoming resolutions: high-conviction (0.86) ranks before low (0.41).
    assert body["upcoming_resolutions"][0]["slug"] == "slug-b"
    # Recently launched ordering: slug-a (more recent createdAt) first.
    assert body["recently_launched"][0]["slug"] == "slug-a"
    assert route.called


# --- /terminal/search -------------------------------------------------------


def test_terminal_search_token_overlap_scoring(
    app_client: TestClient, patched_terminal_data: None
) -> None:
    r = app_client.get("/terminal/search?q=Factor%20A&limit=5")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == "Factor A"
    ids = [h["factor_id"] for h in body["results"]]
    # factor_a should rank first because its name "Factor A" is an exact match.
    assert ids and ids[0] == "factor_a"


def test_terminal_search_returns_empty_when_no_overlap(
    app_client: TestClient, patched_terminal_data: None
) -> None:
    r = app_client.get("/terminal/search?q=zzz_unrelated_xyz")
    assert r.status_code == 200
    body = r.json()
    assert body["n_results"] == 0
    assert body["results"] == []


# --- helpers (terminal module unit) -----------------------------------------


def test_compute_stats_handles_short_series() -> None:
    s = pd.Series([0.5, 0.51, 0.52], index=pd.date_range("2026-01-01", periods=3))
    out = terminal_mod.compute_stats_from_series(s)
    assert out["n_obs"] == 3
    assert out["dfa_alpha"] is None  # not enough data
    assert out["current_price"] == 0.52


def test_find_peers_dedupes_and_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = [
        {"a_id": "x", "b_id": "y", "oos_sharpe": 1.0, "adf_pvalue": 0.05},
        {"a_id": "y", "b_id": "x", "oos_sharpe": 2.0, "adf_pvalue": 0.01},  # mirror
        {"a_id": "x", "b_id": "z", "oos_sharpe": 3.0, "adf_pvalue": 0.02},
    ]
    monkeypatch.setattr(terminal_mod, "_load_ah_hits", lambda _p: fake)
    peers = terminal_mod.find_peers("x", top_n=10)
    ids = [p["peer_id"] for p in peers]
    # z (sharpe 3) ranks above y; the two y-x mirrors are deduped to one row.
    assert ids == ["z", "y"]
