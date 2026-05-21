"""Tests for ``pfm.auto_hedge``.

We construct synthetic portfolios with known per-factor β and verify:
  - the solver recovers the neutralising PM size,
  - target_beta=0 drives net β to ~0,
  - the simulator produces internally-coherent paths.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.auto_hedge import (
    _portfolio_beta,
    compute_hedge,
    router,
    simulate_hedge_path,
)

# --- portfolio beta sanity --------------------------------------------------


def test_portfolio_beta_uses_factor_table() -> None:
    """SPY has +1.00 β to spx-6500, so $100k SPY → +$100k β."""
    pf = [{"ticker": "SPY", "size_usd": 100_000.0}]
    beta = _portfolio_beta(pf, "spx-6500-by-eoy")
    assert beta == pytest.approx(100_000.0, rel=1e-9)


def test_portfolio_beta_unknown_ticker_zero() -> None:
    pf = [{"ticker": "ZZZZ", "size_usd": 1_000.0}]
    assert _portfolio_beta(pf, "any-factor") == 0.0


# --- compute_hedge ----------------------------------------------------------


def test_compute_hedge_target_beta_zero_neutralises() -> None:
    """target_beta=0 → residual β per factor is exactly 0."""
    pf = [
        {"ticker": "SPY", "size_usd": 100_000.0},
        {"ticker": "NVDA", "size_usd": 50_000.0},
    ]
    out = compute_hedge(
        portfolio=pf,
        hedge_factors=["recession-2026", "vix-25-by-jun"],
        target_beta=0.0,
    )
    for residual_beta in out["net_beta_after_hedge"].values():
        assert abs(residual_beta) < 1e-6


def test_compute_hedge_solves_for_known_beta() -> None:
    """SPY $100k portfolio with -0.85 β to recession → hedge size $85k."""
    pf = [{"ticker": "SPY", "size_usd": 100_000.0}]
    out = compute_hedge(pf, hedge_factors=["recession-2026"], target_beta=0.0)
    hedge = out["hedge_positions"][0]
    # Portfolio β = -85,000; hedge size = +85,000 (long YES of recession-2026).
    assert hedge["slug"] == "recession-2026"
    assert hedge["size_usd"] == pytest.approx(85_000.0, rel=1e-6)
    assert hedge["side"] == "YES"


def test_compute_hedge_negative_beta_ticker_uses_no_side() -> None:
    """Long QQQ has +β to spx-6500 → hedge needs to *short* it (NO side)."""
    pf = [{"ticker": "QQQ", "size_usd": 100_000.0}]
    out = compute_hedge(pf, hedge_factors=["spx-6500-by-eoy"], target_beta=0.0)
    hedge = out["hedge_positions"][0]
    # Portfolio β = +110,000; hedge = -110,000 → NO side.
    assert hedge["side"] == "NO"
    assert hedge["size_usd"] == pytest.approx(110_000.0, rel=1e-6)


def test_compute_hedge_target_beta_nonzero_leaves_residual() -> None:
    """target_beta=0.5 must leave residual ≈ 0.5 per dollar."""
    pf = [{"ticker": "SPY", "size_usd": 100_000.0}]
    out = compute_hedge(pf, hedge_factors=["spx-6500-by-eoy"], target_beta=0.5)
    for residual in out["net_beta_after_hedge"].values():
        assert residual == pytest.approx(0.5, abs=1e-6)


def test_compute_hedge_rejects_empty_portfolio() -> None:
    with pytest.raises(ValueError):
        compute_hedge(portfolio=[], hedge_factors=["x"], target_beta=0.0)


def test_compute_hedge_rejects_empty_hedge_factors() -> None:
    pf = [{"ticker": "SPY", "size_usd": 100_000.0}]
    with pytest.raises(ValueError):
        compute_hedge(portfolio=pf, hedge_factors=[], target_beta=0.0)


def test_compute_hedge_slippage_estimate_positive() -> None:
    pf = [{"ticker": "SPY", "size_usd": 100_000.0}]
    out = compute_hedge(pf, hedge_factors=["recession-2026"], target_beta=0.0)
    assert out["slippage_30d_estimate_bps"] > 0


# --- simulate_hedge_path ----------------------------------------------------


def test_simulate_hedge_path_returns_correct_length() -> None:
    pf = [{"ticker": "SPY", "size_usd": 100_000.0}]
    out = simulate_hedge_path(
        portfolio=pf,
        hedge_factors=["recession-2026", "vix-25-by-jun"],
        days=30,
    )
    assert out["days"] == 30
    assert len(out["path"]) == 30


def test_simulate_hedge_path_is_deterministic() -> None:
    pf = [{"ticker": "SPY", "size_usd": 100_000.0}]
    a = simulate_hedge_path(pf, ["recession-2026"], days=15)
    b = simulate_hedge_path(pf, ["recession-2026"], days=15)
    assert a["final_portfolio_pnl_usd"] == b["final_portfolio_pnl_usd"]
    assert a["final_hedged_pnl_usd"] == b["final_hedged_pnl_usd"]


def test_simulate_hedge_path_reduces_volatility() -> None:
    """The hedged book should have lower daily-PnL std than unhedged."""
    pf = [
        {"ticker": "SPY", "size_usd": 100_000.0},
        {"ticker": "NVDA", "size_usd": 80_000.0},
    ]
    out = simulate_hedge_path(
        pf,
        hedge_factors=["recession-2026", "vix-25-by-jun", "spx-6500-by-eoy"],
        days=60,
    )
    # vol_reduction_ratio < 1 means the hedge attenuated daily PnL std.
    assert out["vol_reduction_ratio"] < 1.0


def test_simulate_hedge_path_slippage_monotonic() -> None:
    """Cumulative slippage must be non-decreasing along the path."""
    pf = [{"ticker": "SPY", "size_usd": 100_000.0}]
    out = simulate_hedge_path(pf, ["recession-2026"], days=30)
    slips = [pt["cumulative_slippage_usd"] for pt in out["path"]]
    for i in range(1, len(slips)):
        assert slips[i] >= slips[i - 1] - 1e-6


def test_simulate_hedge_path_rejects_short_window() -> None:
    pf = [{"ticker": "SPY", "size_usd": 100_000.0}]
    with pytest.raises(ValueError):
        simulate_hedge_path(pf, ["recession-2026"], days=1)


# --- HTTP integration -------------------------------------------------------


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_post_hedge_auto_config_endpoint() -> None:
    c = _client()
    r = c.post(
        "/hedge/auto-config",
        json={
            "portfolio": [{"ticker": "SPY", "size_usd": 100_000.0}],
            "hedge_factors": ["recession-2026", "vix-25-by-jun"],
            "target_beta": 0.0,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["target_beta"] == 0.0
    assert len(body["hedge_positions"]) == 2
    for residual in body["net_beta_after_hedge"].values():
        assert abs(residual) < 1e-6


def test_post_hedge_simulate_endpoint() -> None:
    c = _client()
    r = c.post(
        "/hedge/simulate",
        json={
            "portfolio": [{"ticker": "NVDA", "size_usd": 50_000.0}],
            "hedge_factors": ["ai-capex-cut-q2"],
            "target_beta": 0.0,
            "days": 20,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["days"] == 20
    assert len(body["path"]) == 20
    assert body["final_slippage_usd"] >= 0


def test_post_hedge_auto_config_rejects_empty_portfolio() -> None:
    c = _client()
    r = c.post(
        "/hedge/auto-config",
        json={
            "portfolio": [],
            "hedge_factors": ["recession-2026"],
            "target_beta": 0.0,
        },
    )
    # Pydantic v2 min_length=1 → 422.
    assert r.status_code == 422
