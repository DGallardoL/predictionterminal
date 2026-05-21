"""Tests for ``pfm.terminal_portfolio_sim`` — POST /terminal/portfolio-sim.

External Polymarket calls are patched out so the suite is fully offline.
The router is mounted on a fresh :class:`FastAPI` app so we don't pull in
the full ``pfm.main`` lifespan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_portfolio_sim
from pfm.terminal_portfolio_sim import (
    get_polymarket_client,
    router,
)

# --- fakes ------------------------------------------------------------------


class _FakePoly:
    """Sentinel — fetch_factor_history is monkeypatched, so the client is unused."""


def _make_history(
    days: int = 200,
    base: float = 0.50,
    *,
    drift: float = 0.0,
    seed: int = 1,
) -> pd.DataFrame:
    """Build a synthetic price history matching ``fetch_factor_history``'s contract."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    idx.name = "date"
    walk = np.cumsum(rng.normal(drift, 0.01, days))
    prices = (base + walk).clip(0.05, 0.95)
    return pd.DataFrame({"price": prices}, index=idx)


def _build_client(
    bank: dict[str, pd.DataFrame] | None,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Mount the router on a bare app and patch fetch_factor_history."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_polymarket_client] = _FakePoly

    def _fake_factor_history(_client, slug, start=None, end=None):
        if bank is None or slug not in bank:
            raise KeyError(f"unknown slug {slug!r}")
        df = bank[slug]
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    monkeypatch.setattr(terminal_portfolio_sim, "fetch_factor_history", _fake_factor_history)
    return TestClient(app)


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def two_position_bank() -> dict[str, pd.DataFrame]:
    """Two synthetic markets with deterministic histories."""
    return {
        "alpha": _make_history(days=200, base=0.40, drift=0.0005, seed=11),
        "beta": _make_history(days=200, base=0.60, drift=-0.0003, seed=22),
    }


# --- tests ------------------------------------------------------------------


class TestPortfolioSim:
    def test_two_position_portfolio_basic(
        self,
        two_position_bank: dict[str, pd.DataFrame],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two YES/NO positions: response must surface every documented field
        with sane shapes and finite numbers."""
        client = _build_client(two_position_bank, monkeypatch=monkeypatch)
        body = {
            "positions": [
                {"slug": "alpha", "side": "YES", "size_usd": 1000.0},
                {"slug": "beta", "side": "NO", "size_usd": 2000.0},
            ],
            "days": 180,
        }
        r = client.post("/terminal/portfolio-sim", json=body)
        assert r.status_code == 200, r.text
        out = r.json()

        assert out["n_positions"] == 2
        assert out["gross_notional_usd"] == pytest.approx(3000.0)
        # All required fields present.
        for key in (
            "expected_payoff_usd_at_resolution",
            "current_book_pnl_usd",
            "sharpe_estimate_via_history",
            "max_drawdown",
            "position_correlation_matrix",
            "recommended_hedge",
        ):
            assert key in out

        # Correlation matrix is 2x2.
        cm = out["position_correlation_matrix"]
        assert len(cm) == 2
        for row in cm.values():
            assert len(row) == 2
            for v in row.values():
                if v is not None:
                    assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9

        # Diagonals are 1.0.
        for label, row in cm.items():
            assert row[label] == pytest.approx(1.0, abs=1e-9)

        # Max drawdown is non-positive (it's a loss measure).
        assert out["max_drawdown"] <= 1e-9
        # Expected payoff at resolution is bounded by gross notional.
        assert 0.0 <= out["expected_payoff_usd_at_resolution"] <= out["gross_notional_usd"]

    def test_single_position_edge_case(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single-position book: correlation matrix degenerates to {label: 1.0}
        and recommended_hedge is empty (no peer to hedge against)."""
        bank = {"solo": _make_history(days=200, base=0.50, drift=0.0008, seed=99)}
        client = _build_client(bank, monkeypatch=monkeypatch)
        body = {
            "positions": [{"slug": "solo", "side": "YES", "size_usd": 5000.0}],
            "days": 180,
        }
        r = client.post("/terminal/portfolio-sim", json=body)
        assert r.status_code == 200, r.text
        out = r.json()

        assert out["n_positions"] == 1
        assert out["gross_notional_usd"] == pytest.approx(5000.0)
        # 1x1 matrix with the diagonal == 1.
        cm = out["position_correlation_matrix"]
        assert len(cm) == 1
        only_label = next(iter(cm))
        assert cm[only_label][only_label] == pytest.approx(1.0)
        # No peer exists, so the hedge list must be empty.
        assert out["recommended_hedge"] == []

    def test_empty_positions_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pydantic enforces ``min_length=1`` on positions — FastAPI returns 422
        for schema-level violations. We accept either 400 or 422 here so the
        intent ('reject empty book') is preserved regardless of FastAPI version."""
        client = _build_client({}, monkeypatch=monkeypatch)
        r = client.post("/terminal/portfolio-sim", json={"positions": [], "days": 180})
        assert r.status_code in (400, 422), r.text

    def test_correlation_calc_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two positions with *identical* underlying price series should produce
        a +1 correlation if same-side, and -1 if opposite-side."""
        # Same underlying probabilistic walk.
        same = _make_history(days=180, base=0.45, drift=0.0004, seed=55)
        # Use distinct slugs but identical price data so the position-PnL
        # series differ only by their sign convention.
        bank = {"twin_a": same, "twin_b": same.copy()}
        client = _build_client(bank, monkeypatch=monkeypatch)

        # Same side YES/YES: expect corr ≈ +1.
        body_same = {
            "positions": [
                {"slug": "twin_a", "side": "YES", "size_usd": 1000.0},
                {"slug": "twin_b", "side": "YES", "size_usd": 1000.0},
            ],
            "days": 120,
        }
        r1 = client.post("/terminal/portfolio-sim", json=body_same)
        assert r1.status_code == 200, r1.text
        cm1 = r1.json()["position_correlation_matrix"]
        labels1 = list(cm1)
        off_diag_same = cm1[labels1[0]][labels1[1]]
        assert off_diag_same == pytest.approx(1.0, abs=1e-6)

        # Opposite side YES/NO: same price but flipped sign → corr ≈ -1.
        body_opp = {
            "positions": [
                {"slug": "twin_a", "side": "YES", "size_usd": 1000.0},
                {"slug": "twin_b", "side": "NO", "size_usd": 1000.0},
            ],
            "days": 120,
        }
        r2 = client.post("/terminal/portfolio-sim", json=body_opp)
        assert r2.status_code == 200, r2.text
        cm2 = r2.json()["position_correlation_matrix"]
        labels2 = list(cm2)
        off_diag_opp = cm2[labels2[0]][labels2[1]]
        assert off_diag_opp == pytest.approx(-1.0, abs=1e-6)

        # The opposite-side book is a delta-neutral pair: portfolio Sharpe
        # should be near zero / NaN since portfolio PnL is identically 0.
        sharpe_opp = r2.json()["sharpe_estimate_via_history"]
        assert sharpe_opp is None or abs(sharpe_opp) < 1e-6
