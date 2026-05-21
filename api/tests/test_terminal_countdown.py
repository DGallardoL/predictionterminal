"""Unit tests for the resolution-countdown endpoint.

The router is mounted on a throw-away FastAPI app so we don't touch
``main.py``; Gamma HTTP is intercepted with respx.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_countdown as tc
from pfm.terminal_countdown import (
    GAMMA_URL,
    build_countdown_markets,
    conviction,
    day_bucket_for,
    group_by_bucket,
    router,
)

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _write_factors_yml(tmp: Path) -> Path:
    p = tmp / "factors.yml"
    p.write_text(
        """
factors:
  - id: market_today
    name: Resolves today
    slug: market-today
    source: polymarket
    theme: macro
  - id: market_tomorrow
    name: Resolves tomorrow
    slug: market-tomorrow
    source: polymarket
    theme: politics
  - id: market_this_week
    name: Resolves this week
    slug: market-this-week
    source: polymarket
    theme: crypto
  - id: market_far_future
    name: Way out of horizon
    slug: market-far-future
    source: polymarket
    theme: macro
  - id: kalshi_skip
    name: Should be skipped (kalshi)
    slug: kalshi-skip
    source: kalshi
    theme: macro
"""
    )
    return p


def _gamma_market(
    slug: str,
    end_date: datetime,
    last_trade_price: float,
    one_day_change: float = 0.0,
    volume_24hr: float = 1000.0,
    active: bool = True,
    closed: bool = False,
) -> dict:
    return {
        "slug": slug,
        "question": f"Will {slug} resolve YES?",
        "endDate": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lastTradePrice": last_trade_price,
        "outcomePrices": json.dumps([str(last_trade_price), str(1 - last_trade_price)]),
        "oneDayPriceChange": one_day_change,
        "volume24hr": volume_24hr,
        "active": active,
        "closed": closed,
        "clobTokenIds": json.dumps(["aaa", "bbb"]),
    }


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


def test_conviction_and_day_bucket_logic() -> None:
    """Pure helpers: conviction is symmetric around 0.5 and buckets map cleanly."""
    # Conviction
    assert conviction(0.5) == pytest.approx(0.0)
    assert conviction(1.0) == pytest.approx(1.0)
    assert conviction(0.0) == pytest.approx(1.0)
    assert conviction(0.9) == pytest.approx(0.8)
    assert conviction(0.1) == pytest.approx(0.8)
    # Out-of-range inputs are clipped, not raised.
    assert conviction(1.5) == pytest.approx(1.0)
    assert conviction(-0.2) == pytest.approx(1.0)

    # Day buckets
    assert day_bucket_for(0) == "today"
    assert day_bucket_for(1) == "tomorrow"
    assert day_bucket_for(3) == "this-week"
    assert day_bucket_for(6) == "this-week"
    assert day_bucket_for(7) == "next-week"
    assert day_bucket_for(13) == "next-week"
    assert day_bucket_for(14) == "this-month"
    assert day_bucket_for(30) == "this-month"
    assert day_bucket_for(45) == "later"


def test_build_countdown_markets_filters_and_sorts() -> None:
    """Filters out closed/inactive/out-of-window markets and sorts by bucket → days → conviction."""
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    factors = [
        {"id": "a_today_high_conv", "slug": "a", "theme": "macro"},
        {"id": "b_today_low_conv", "slug": "b", "theme": "macro"},
        {"id": "c_tomorrow", "slug": "c", "theme": "politics"},
        {"id": "d_far", "slug": "d", "theme": "crypto"},
        {"id": "e_closed", "slug": "e", "theme": "macro"},
        {"id": "f_past", "slug": "f", "theme": "macro"},
    ]
    gamma = {
        # today, very high conviction (p=0.95)
        "a": _gamma_market("a", now + timedelta(hours=4), 0.95),
        # today, low conviction (p=0.55)
        "b": _gamma_market("b", now + timedelta(hours=8), 0.55),
        # tomorrow
        "c": _gamma_market("c", now + timedelta(days=1, hours=2), 0.8),
        # outside the 7-day horizon
        "d": _gamma_market("d", now + timedelta(days=20), 0.5),
        # closed → filtered
        "e": _gamma_market("e", now + timedelta(hours=6), 0.7, closed=True),
        # already resolved → filtered
        "f": _gamma_market("f", now - timedelta(days=1), 0.7),
    }

    rows = build_countdown_markets(factors, gamma, now=now, horizon_days=7)
    slugs = [m.slug for m in rows]

    # Out-of-horizon, closed, and past-end markets are dropped.
    assert "d" not in slugs
    assert "e" not in slugs
    assert "f" not in slugs

    # Ordering: today first; within "today" higher conviction first.
    assert slugs[:2] == ["a", "b"]
    assert rows[0].day_bucket == "today"
    assert rows[1].day_bucket == "today"
    assert rows[0].conviction > rows[1].conviction
    # Tomorrow comes after today.
    assert rows[2].slug == "c"
    assert rows[2].day_bucket == "tomorrow"

    # Conviction values are correct.
    assert rows[0].conviction == pytest.approx(0.9)
    assert rows[1].conviction == pytest.approx(0.1)


def test_group_by_bucket_preserves_order_and_counts() -> None:
    """``group_by_bucket`` returns groups in canonical bucket order with right counts."""
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    factors = [
        {"id": "a", "slug": "a", "theme": "x"},
        {"id": "b", "slug": "b", "theme": "x"},
        {"id": "c", "slug": "c", "theme": "x"},
    ]
    gamma = {
        "a": _gamma_market("a", now + timedelta(hours=2), 0.92),
        "b": _gamma_market("b", now + timedelta(days=1, hours=5), 0.6),
        "c": _gamma_market("c", now + timedelta(days=4), 0.7),
    }
    rows = build_countdown_markets(factors, gamma, now=now, horizon_days=7)
    groups = group_by_bucket(rows)

    bucket_names = [g.bucket for g in groups]
    assert bucket_names == ["today", "tomorrow", "this-week"]
    assert [g.n_markets for g in groups] == [1, 1, 1]
    assert sum(g.n_markets for g in groups) == len(rows)


@respx.mock
def test_endpoints_end_to_end_with_mocked_gamma(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: factors.yml + mocked Gamma → /terminal/countdown and /terminal/countdown/{slug}."""
    factors_path = _write_factors_yml(tmp_path)
    monkeypatch.setattr(tc, "_factors_path", lambda: factors_path)

    now = datetime.now(UTC)
    end_today = now + timedelta(hours=5)
    end_tomorrow = now + timedelta(days=1, hours=3)
    end_this_week = now + timedelta(days=4, hours=1)
    end_far = now + timedelta(days=60)

    respx.get(f"{GAMMA_URL}/markets", params={"slug": "market-today"}).mock(
        return_value=httpx.Response(200, json=[_gamma_market("market-today", end_today, 0.93)])
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "market-tomorrow"}).mock(
        return_value=httpx.Response(
            200, json=[_gamma_market("market-tomorrow", end_tomorrow, 0.55, one_day_change=0.04)]
        )
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "market-this-week"}).mock(
        return_value=httpx.Response(
            200, json=[_gamma_market("market-this-week", end_this_week, 0.30)]
        )
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "market-far-future"}).mock(
        return_value=httpx.Response(200, json=[_gamma_market("market-far-future", end_far, 0.5)])
    )

    # /terminal/countdown?days=7
    r = client.get("/terminal/countdown?days=7")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["horizon_days"] == 7
    slugs = [m["slug"] for g in body["groups"] for m in g["markets"]]
    assert "market-today" in slugs
    assert "market-tomorrow" in slugs
    assert "market-this-week" in slugs
    # Far-future is filtered out by the 7-day horizon.
    assert "market-far-future" not in slugs
    # Kalshi factor was excluded — no Gamma call should ever have been mocked for it.
    assert "kalshi-skip" not in slugs

    # Order: "today" bucket first, with high-conviction (0.93) market first.
    assert body["groups"][0]["bucket"] == "today"
    assert body["groups"][0]["markets"][0]["slug"] == "market-today"
    assert body["groups"][0]["markets"][0]["current_p"] == pytest.approx(0.93)
    assert body["groups"][0]["markets"][0]["conviction"] == pytest.approx(0.86)

    # last_24h_change is propagated.
    tomorrow_row = next(
        m for g in body["groups"] for m in g["markets"] if m["slug"] == "market-tomorrow"
    )
    assert tomorrow_row["last_24h_change"] == pytest.approx(0.04)
    assert tomorrow_row["day_bucket"] == "tomorrow"

    # /terminal/countdown/{slug}
    r2 = client.get("/terminal/countdown/market-tomorrow")
    assert r2.status_code == 200, r2.text
    detail = r2.json()
    assert detail["slug"] == "market-tomorrow"
    assert detail["days"] == 1
    assert 0 <= detail["hours"] <= 23
    assert 0 <= detail["minutes"] <= 59
    assert detail["seconds_remaining"] > 0
    # current_p=0.55 → fair_price_at_resolution rounds up to 1.0.
    assert detail["fair_price_at_resolution"] == 1.0
    assert detail["expected_payoff_if_held"] == pytest.approx(0.55)
