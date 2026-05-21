"""Tests for the terminal market-quality-score endpoint.

Mounts the router on a throw-away FastAPI app so we don't touch ``main.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal_quality_score import (
    CLOB_URL,
    DATA_API_URL,
    GAMMA_URL,
    WEIGHTS,
    router,
)

SLUG = "fed-decision"
YES_TOKEN = "111"
CONDITION_ID = "0xabcdef"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _gamma_response(
    slug: str = SLUG,
    *,
    start_days_ago: int = 120,
    end_days_ahead: int = 60,
    volume_24hr: float = 250_000.0,
    yes_token: str = YES_TOKEN,
    condition_id: str | None = CONDITION_ID,
) -> httpx.Response:
    """Build a Gamma /markets payload with tunable age + dte + 24h volume."""
    now = datetime.now(UTC)
    start = (now - timedelta(days=start_days_ago)).isoformat().replace("+00:00", "Z")
    end = (now + timedelta(days=end_days_ahead)).isoformat().replace("+00:00", "Z")
    payload: dict = {
        "slug": slug,
        "question": "Will the Fed cut rates?",
        "clobTokenIds": json.dumps([yes_token, "222"]),
        "active": True,
        "closed": False,
        "startDate": start,
        "endDate": end,
        "volume24hr": volume_24hr,
    }
    if condition_id:
        payload["conditionId"] = condition_id
    return httpx.Response(200, json=[payload])


def _book_response(*, tight: bool = True) -> httpx.Response:
    """Tight book = 1c spread + fat sizes; otherwise wide + thin."""
    if tight:
        bids = [{"price": round(0.50 - 0.01 * i, 2), "size": 5_000.0} for i in range(10)]
        asks = [{"price": round(0.51 + 0.01 * i, 2), "size": 5_000.0} for i in range(10)]
    else:
        # 12c spread, 5 shares per side → minimal depth.
        bids = [{"price": 0.40, "size": 5.0}]
        asks = [{"price": 0.52, "size": 5.0}]
    return httpx.Response(200, json={"bids": bids, "asks": asks})


def _trades_response(n: int) -> httpx.Response:
    """Stub data-api /trades with ``n`` rows — endpoint just counts the list."""
    return httpx.Response(200, json=[{"price": 0.5, "size": 1.0} for _ in range(n)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@respx.mock
def test_high_quality_market_grades_a(client: TestClient) -> None:
    """Tight book + fat 24h volume + medium age & dte + busy tape → A."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(
        return_value=_gamma_response(volume_24hr=500_000.0)
    )
    respx.get(f"{CLOB_URL}/book").mock(return_value=_book_response(tight=True))
    respx.get(f"{DATA_API_URL}/trades").mock(return_value=_trades_response(200))

    r = client.get(f"/terminal/quality/{SLUG}")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["slug"] == SLUG
    assert body["grade"] == "A"
    assert body["quality_score"] >= 85.0

    comps = body["components"]
    # Each major component should be near-perfect on a tight book + busy market.
    assert comps["spread_score"] == 100.0
    assert comps["depth_score"] >= 80.0
    assert comps["vol_score"] >= 85.0
    assert comps["activity_score"] >= 95.0
    # 60d to resolution falls in 30..300 sweet spot.
    assert comps["dte_score"] == 100.0
    # 120d age >= 90d → maxed.
    assert comps["age_score"] == 100.0

    # No alarming flags on a healthy market.
    assert "thin_book" not in body["flags"]
    assert "low_vol" not in body["flags"]
    assert "near_resolution" not in body["flags"]
    assert "newly_launched" not in body["flags"]


@respx.mock
def test_low_quality_market_grades_f(client: TestClient) -> None:
    """Wide spread + thin book + tiny volume + zero trades → F (or near it)."""
    # 100d ahead but only $50 in 24h; book is essentially empty.
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(
        return_value=_gamma_response(start_days_ago=200, end_days_ahead=120, volume_24hr=50.0)
    )
    respx.get(f"{CLOB_URL}/book").mock(return_value=_book_response(tight=False))
    respx.get(f"{DATA_API_URL}/trades").mock(return_value=_trades_response(0))

    r = client.get(f"/terminal/quality/{SLUG}")
    assert r.status_code == 200, r.text
    body = r.json()

    # Composite must be deep in F territory.
    assert body["grade"] in {"F", "D"}
    assert body["quality_score"] < 55.0

    comps = body["components"]
    # 12c spread → 0; thin book → ~0; $50 24h vol → < 40; 0 trades → 0.
    assert comps["spread_score"] == 0.0
    assert comps["depth_score"] < 5.0
    assert comps["vol_score"] < 40.0
    assert comps["activity_score"] == 0.0

    # Flags should fire on the obvious problems.
    flags = set(body["flags"])
    assert {"thin_book", "low_vol", "wide_spread", "low_activity"}.issubset(flags)
    assert "illiquid_24h" in flags


@respx.mock
def test_unknown_slug_returns_404(client: TestClient) -> None:
    """Empty Gamma response → 404, downstream calls never happen."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    book_route = respx.get(f"{CLOB_URL}/book").mock(
        return_value=httpx.Response(200, json={"bids": [], "asks": []})
    )
    trades_route = respx.get(f"{DATA_API_URL}/trades").mock(
        return_value=httpx.Response(200, json=[])
    )

    r = client.get("/terminal/quality/ghost")
    assert r.status_code == 404
    assert "no market found" in r.json()["detail"]
    assert book_route.call_count == 0
    assert trades_route.call_count == 0


@respx.mock
def test_all_warning_flags_trigger(client: TestClient) -> None:
    """A near-resolution + newly-launched + illiquid market trips every flag."""
    # Launched 3 days ago, resolves in 5 days, $100 in 24h volume.
    respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(
        return_value=_gamma_response(start_days_ago=3, end_days_ahead=5, volume_24hr=100.0)
    )
    respx.get(f"{CLOB_URL}/book").mock(return_value=_book_response(tight=False))
    respx.get(f"{DATA_API_URL}/trades").mock(return_value=_trades_response(1))

    body = client.get(f"/terminal/quality/{SLUG}").json()

    # Sanity: weights still sum to 1 (defensive).
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    flags = set(body["flags"])
    expected = {
        "thin_book",
        "low_vol",
        "wide_spread",
        "low_activity",
        "near_resolution",
        "newly_launched",
        "crossed_or_wide",
        "illiquid_24h",
    }
    assert expected.issubset(flags), f"missing flags: {expected - flags}"

    # dte<7 → dte_score must be 0; age=3 → age_score < 50.
    assert body["components"]["dte_score"] == 0.0
    assert body["components"]["age_score"] < 50.0
    # n_trades_24h echoed back from the data-api stub (=1).
    assert body["n_trades_24h"] == 1


@respx.mock
def test_quality_response_is_cached(client: TestClient) -> None:
    """Composite is cached for 30 s on (slug,) — second hit must not refetch.

    Three sequential HTTP calls (gamma + clob + data-api) per cold call is
    expensive and rate-limit-prone; the cache collapses the warm path.
    """
    from pfm.cache_utils import get_cache

    get_cache("terminal_quality").clear()

    gamma_route = respx.get(f"{GAMMA_URL}/markets", params={"slug": SLUG}).mock(
        return_value=_gamma_response(volume_24hr=500_000.0)
    )
    book_route = respx.get(f"{CLOB_URL}/book").mock(return_value=_book_response(tight=True))
    trades_route = respx.get(f"{DATA_API_URL}/trades").mock(return_value=_trades_response(50))

    r1 = client.get(f"/terminal/quality/{SLUG}")
    r2 = client.get(f"/terminal/quality/{SLUG}")
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    # Every upstream call must be cold-once-only.
    assert gamma_route.call_count == 1
    assert book_route.call_count == 1
    assert trades_route.call_count == 1
