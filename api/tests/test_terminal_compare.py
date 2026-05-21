"""Tests for ``pfm.terminal_compare`` — /terminal/compare?slugs=...

External HTTP is mocked via :mod:`respx` so the suite is fully offline.
The router is mounted on a fresh :class:`FastAPI` app to avoid pulling
the full ``pfm.main`` lifespan.
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

from pfm import terminal_compare
from pfm.terminal_compare import clear_cache, router

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


# --- helpers ----------------------------------------------------------------


def _build_app() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _gamma_market_payload(slug: str, *, token_id: str, base_price: float = 0.55) -> dict[str, Any]:
    """Mock Gamma /markets response (returns a list with one market dict)."""
    return {
        "slug": slug,
        "question": f"Question for {slug}?",
        "description": f"Test market {slug}",
        "clobTokenIds": json.dumps([token_id, f"{token_id}_no"]),
        "bestBid": base_price - 0.01,
        "bestAsk": base_price + 0.01,
        "lastTradePrice": base_price,
        "volume24hr": 50_000.0,
        "volumeNum": 1_000_000.0,
        "liquidityNum": 25_000.0,
        "oneDayPriceChange": 0.02,
        "oneWeekPriceChange": -0.05,
        "endDate": "2026-12-01T00:00:00Z",
        "startDate": "2025-01-01T00:00:00Z",
        "createdAt": "2025-01-01T00:00:00Z",
        "active": True,
        "closed": False,
    }


def _clob_history_payload(days: int = 90, *, seed: int = 1, base: float = 0.55) -> dict[str, Any]:
    """Mock CLOB /prices-history response."""
    rng = np.random.default_rng(seed)
    end_ts = int(pd.Timestamp.utcnow().normalize().timestamp())
    history = []
    p = base
    for i in range(days):
        # Simple AR(1)-like walk so different seeds give differently
        # correlated paths.
        p = max(0.05, min(0.95, p + 0.01 * rng.standard_normal()))
        ts = end_ts - (days - 1 - i) * 86400
        history.append({"t": ts, "p": float(p)})
    return {"history": history}


def _mock_slug(
    slug: str, *, token_id: str, base_price: float = 0.55, seed: int = 1, days: int = 90
) -> None:
    """Register respx routes for both Gamma and CLOB for a single slug."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": slug}).mock(
        return_value=httpx.Response(
            200, json=[_gamma_market_payload(slug, token_id=token_id, base_price=base_price)]
        )
    )
    respx.get(f"{CLOB_URL}/prices-history", params={"market": token_id}).mock(
        return_value=httpx.Response(
            200, json=_clob_history_payload(days=days, seed=seed, base=base_price)
        )
    )


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _drop_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the gamma/clob URLs to the constants the tests mock."""

    class _S:
        polymarket_gamma_url = GAMMA_URL
        polymarket_clob_url = CLOB_URL

    monkeypatch.setattr(terminal_compare, "get_settings", _S)


# --- tests ------------------------------------------------------------------


class TestValidation:
    def test_n_equals_1_returns_400(self) -> None:
        client = _build_app()
        r = client.get("/terminal/compare?slugs=only-one&days=30")
        assert r.status_code == 400
        assert "between" in r.json()["detail"]

    def test_n_equals_5_returns_400(self) -> None:
        client = _build_app()
        slugs = ",".join([f"s-{i}" for i in range(5)])
        r = client.get(f"/terminal/compare?slugs={slugs}&days=30")
        assert r.status_code == 400
        assert "between" in r.json()["detail"]

    def test_duplicate_slugs_rejected(self) -> None:
        client = _build_app()
        r = client.get("/terminal/compare?slugs=a,a&days=30")
        assert r.status_code == 400
        assert "duplicate" in r.json()["detail"].lower()


class TestN2:
    @respx.mock
    def test_pairs_trade_present(self) -> None:
        _mock_slug("apple-up", token_id="tok-a", base_price=0.55, seed=1)
        _mock_slug("orange-up", token_id="tok-b", base_price=0.45, seed=2)

        client = _build_app()
        r = client.get("/terminal/compare?slugs=apple-up,orange-up&days=60")
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["slugs"] == ["apple-up", "orange-up"]
        assert body["days"] == 60
        assert len(body["legs"]) == 2
        assert {leg["slug"] for leg in body["legs"]} == {"apple-up", "orange-up"}

        # Each leg has live, meta, stats, history.
        for leg in body["legs"]:
            assert leg["live"]["midpoint"] is not None
            assert leg["meta"]["question"]
            assert leg["stats"]["n_obs"] > 0
            assert len(leg["history"]) > 0
            # Indexed series must rebase to 100 at t0.
            assert abs(leg["history"][0]["indexed"] - 100.0) < 1e-6

        # Pairs-trade card present and well-formed.
        pt = body["pairs_trade"]
        assert pt is not None
        assert pt["a"] == "apple-up"
        assert pt["b"] == "orange-up"
        assert pt["beta_hedge"] is not None
        assert pt["spread_now"] is not None
        # z_score should be finite (not None) given non-degenerate spread.
        assert pt["z_score"] is not None


class TestN3:
    @respx.mock
    def test_no_pairs_trade(self) -> None:
        _mock_slug("a", token_id="tok-a", base_price=0.40, seed=1)
        _mock_slug("b", token_id="tok-b", base_price=0.50, seed=2)
        _mock_slug("c", token_id="tok-c", base_price=0.60, seed=3)

        client = _build_app()
        r = client.get("/terminal/compare?slugs=a,b,c&days=60")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["legs"]) == 3
        assert body["pairs_trade"] is None


class TestCorrelationMatrix:
    @respx.mock
    def test_shape_diagonal_symmetric(self) -> None:
        _mock_slug("x", token_id="tok-x", base_price=0.50, seed=11)
        _mock_slug("y", token_id="tok-y", base_price=0.50, seed=22)
        _mock_slug("z", token_id="tok-z", base_price=0.50, seed=33)

        client = _build_app()
        r = client.get("/terminal/compare?slugs=x,y,z&days=80")
        assert r.status_code == 200, r.text
        body = r.json()

        m = body["correlation_matrix"]
        # Shape NxN with the right keys.
        assert set(m.keys()) == {"x", "y", "z"}
        for k in ("x", "y", "z"):
            assert set(m[k].keys()) == {"x", "y", "z"}

        # Diagonal == 1.0.
        for k in ("x", "y", "z"):
            assert m[k][k] == 1.0

        # Off-diagonal symmetric (where both sides defined).
        for a, b in [("x", "y"), ("x", "z"), ("y", "z")]:
            v_ab = m[a][b]
            v_ba = m[b][a]
            assert v_ab is not None
            assert v_ba is not None
            assert abs(v_ab - v_ba) < 1e-9
            # Pearson corr always in [-1, 1].
            assert -1.0 <= v_ab <= 1.0
