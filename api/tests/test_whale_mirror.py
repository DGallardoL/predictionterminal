"""Tests for ``pfm.whale_mirror``.

Strategy: the module is deterministic given the seeded synthetic data
generator, so we can assert exact ordering and proportional scaling
without any external IO.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.whale_mirror import (
    mirror_whale,
    router,
    top_whales,
    whale_history,
)


def test_top_whales_returns_sorted_desc_by_abs_pnl() -> None:
    """The list must be ordered by ``abs(pnl_7d_usd)`` descending."""
    rows = top_whales(window_days=7, min_pnl_usd=0.0)
    assert len(rows) > 0
    abs_pnls = [abs(r["pnl_7d_usd"]) for r in rows]
    assert abs_pnls == sorted(abs_pnls, reverse=True)


def test_top_whales_filters_by_min_pnl() -> None:
    """Wallets below ``min_pnl_usd`` must be excluded."""
    threshold = 50_000.0
    rows = top_whales(window_days=7, min_pnl_usd=threshold)
    for r in rows:
        assert abs(r["pnl_7d_usd"]) >= threshold


def test_top_whales_is_deterministic() -> None:
    """Same args must yield the same wallet list (synthetic seeded)."""
    a = top_whales(window_days=7, min_pnl_usd=0.0)
    b = top_whales(window_days=7, min_pnl_usd=0.0)
    assert [r["address"] for r in a] == [r["address"] for r in b]


def test_top_whales_summary_fields_present() -> None:
    rows = top_whales(window_days=7, min_pnl_usd=0.0)
    for r in rows:
        assert {
            "address",
            "pnl_7d_usd",
            "positions_value_usd",
            "win_rate",
            "num_active_positions",
        } <= set(r.keys())
        assert 0.0 <= r["win_rate"] <= 1.0
        assert r["num_active_positions"] >= 0


def test_mirror_whale_scales_to_capital() -> None:
    """Total exposure must respect the capital budget (allow 1¢ rounding)."""
    capital = 25_000.0
    out = mirror_whale("0xWHALE000000000000000000000000000000A001", capital_usd=capital)
    assert out["total_exposure"] <= capital + 0.05
    assert out["total_exposure"] >= capital * 0.99  # all weight allocated


def test_mirror_whale_respects_max_positions() -> None:
    out = mirror_whale(
        "0xWHALE000000000000000000000000000000A001",
        capital_usd=10_000.0,
        max_positions=3,
    )
    assert len(out["suggested_positions"]) <= 3


def test_mirror_whale_scales_linearly_with_capital() -> None:
    """Doubling capital must double the total exposure (proportional sizing)."""
    addr = "0xWHALE000000000000000000000000000000A002"
    a = mirror_whale(addr, capital_usd=10_000.0, max_positions=10)
    b = mirror_whale(addr, capital_usd=20_000.0, max_positions=10)
    assert abs(b["total_exposure"] - 2 * a["total_exposure"]) < 1.0


def test_mirror_whale_rejects_zero_capital() -> None:
    import pytest

    with pytest.raises(ValueError):
        mirror_whale("0xWHALE000000000000000000000000000000A001", capital_usd=0.0)


def test_mirror_whale_equity_beta_finite() -> None:
    out = mirror_whale("0xWHALE000000000000000000000000000000A003", capital_usd=10_000.0)
    beta = out["equivalent_equity_beta_estimate"]
    assert isinstance(beta, float)
    # The pool spans ±1.55 per slug; a portfolio is within ±2.0 in any
    # plausible weighting.
    assert -2.5 < beta < 2.5


def test_whale_history_length_matches_days() -> None:
    out = whale_history("0xWHALE000000000000000000000000000000A001", days=30)
    assert len(out["trace"]) == 30
    for pt in out["trace"]:
        assert {"date_iso", "cumulative_pnl_usd", "positions_value_usd"} <= set(pt.keys())


# --- HTTP integration -------------------------------------------------------


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_get_top_whales_endpoint() -> None:
    c = _client()
    r = c.get("/whales/top?window_days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["window_days"] == 7
    assert isinstance(body["whales"], list)


def test_post_mirror_endpoint() -> None:
    c = _client()
    r = c.post(
        "/whales/mirror",
        json={
            "whale_address": "0xWHALE000000000000000000000000000000A001",
            "capital_usd": 50_000.0,
            "max_positions": 5,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["whale_address"] == "0xWHALE000000000000000000000000000000A001"
    assert body["total_exposure"] <= 50_000.05
    assert len(body["suggested_positions"]) <= 5


def test_get_whale_history_endpoint() -> None:
    c = _client()
    r = c.get("/whales/0xWHALE000000000000000000000000000000A001/history?days=14")
    assert r.status_code == 200
    body = r.json()
    assert body["days"] == 14
    assert len(body["trace"]) == 14
