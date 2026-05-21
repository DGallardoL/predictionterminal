"""Tests for ``pfm.terminal_backtest_compare`` — POST /terminal/backtest-compare.

Both the Polymarket client and the alpha-hunter hits file are stubbed
so the suite never touches the network or the host filesystem outside
``tmp_path``. The router is mounted on a fresh :class:`fastapi.FastAPI`
app to avoid the full ``pfm.main`` lifespan (Redis, factors.yml, …).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_backtest_compare as tbc
from pfm import terminal_inline_backtest as tib
from pfm.terminal_backtest_compare import router
from pfm.terminal_inline_backtest import (
    get_hits_path,
    get_polymarket_client,
)

# --- synthetic data ---------------------------------------------------------


def _cointegrated_pair(
    n: int = 400,
    beta: float = 0.6,
    seed: int = 7,
    intercept: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probability series whose spread is a strong AR(1) reverter.

    Identical structure to the inline-backtest test's helper, but the
    seed / intercept are parameterised so each strategy in the compare
    test gets a *different* underlying series — otherwise their PnLs
    would be perfectly correlated and the correlation matrix wouldn't
    exercise the off-diagonal computation.
    """
    rng = np.random.default_rng(seed)
    b = np.zeros(n)
    b[0] = 0.50
    for t in range(1, n):
        b[t] = np.clip(b[t - 1] + rng.normal(0.0, 0.01), 0.10, 0.90)
    rho = 0.4
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = rho * eps[t - 1] + rng.normal(0.0, 0.02)
    a = np.clip(beta * b + intercept + eps, 0.05, 0.95)

    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    idx.name = "date"
    df_a = pd.DataFrame({"price": a}, index=idx)
    df_b = pd.DataFrame({"price": b}, index=idx)
    return df_a, df_b


# --- fakes ------------------------------------------------------------------


class _FakePoly:
    """Sentinel — fetch_factor_history is patched at module level."""


def _make_hits_file(tmp_path: Path) -> Path:
    """Two independent cointegrated pairs in the alpha-hunter sweep."""
    hits = [
        {
            "a_id": "trump_2028_win",
            "b_id": "vance_2028_win",
            "verdict": "REAL_ALPHA",
            "n_obs": 100,
            "adf_pvalue": 0.01,
            "half_life_days": 1.5,
            "beta_hedge": 0.6,
            "oos_sharpe": 5.0,
            "full_sharpe": 3.0,
            "perm_p": 0.0,
            "perm_real_sharpe": 3.0,
            "sweep": "test",
        },
        {
            "a_id": "fed_cut_q3",
            "b_id": "fed_cut_q4",
            "verdict": "REAL_ALPHA",
            "n_obs": 100,
            "adf_pvalue": 0.02,
            "half_life_days": 2.0,
            "beta_hedge": 0.5,
            "oos_sharpe": 4.0,
            "full_sharpe": 2.5,
            "perm_p": 0.0,
            "perm_real_sharpe": 2.5,
            "sweep": "test",
        },
    ]
    p = tmp_path / "hits.json"
    p.write_text(json.dumps(hits))
    return p


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Mount the compare router on a bare FastAPI app with all IO patched."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_polymarket_client] = _FakePoly

    hits_path = _make_hits_file(tmp_path)
    app.dependency_overrides[get_hits_path] = lambda: hits_path

    # Two distinct pairs with different seeds → different PnL paths.
    df_a1, df_b1 = _cointegrated_pair(seed=7, intercept=0.30)
    df_a2, df_b2 = _cointegrated_pair(seed=42, intercept=0.25)

    bank = {
        "trump-2028-win": df_a1,
        "vance-2028-win": df_b1,
        "fed-cut-q3": df_a2,
        "fed-cut-q4": df_b2,
    }

    def _fake_fetch(_client, slug, start=None, end=None):
        return bank.get(slug, pd.DataFrame(columns=["price"]))

    # Patch on BOTH modules: the compare module re-imported the symbol
    # from polymarket directly, and the inline module also exposes it.
    monkeypatch.setattr(tbc, "fetch_factor_history", _fake_fetch)
    monkeypatch.setattr(tib, "fetch_factor_history", _fake_fetch)

    with TestClient(app) as client:
        yield client


# --- tests ------------------------------------------------------------------


class TestBacktestCompare:
    def test_two_strategies_returns_full_payload(self, app_client: TestClient) -> None:
        """Two valid strategies must yield a 200 with all documented keys
        and a 2×2 correlation matrix that has 1.0 on the diagonal."""
        r = app_client.post(
            "/terminal/backtest-compare",
            json={
                "strategies": [
                    {
                        "slug": "trump-2028-win",
                        "side": "both",
                        "entry_z": 1.5,
                        "exit_z": 0.3,
                        "stop_z": 5.0,
                        "window": 20,
                        "hold_days": None,
                    },
                    {
                        "slug": "fed-cut-q3",
                        "side": "both",
                        "entry_z": 1.5,
                        "exit_z": 0.3,
                        "stop_z": 5.0,
                        "window": 20,
                        "hold_days": None,
                    },
                ],
                "days": 365,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()

        # Top-level shape.
        assert set(body.keys()) == {
            "strategies",
            "correlation",
            "combined_portfolio",
        }
        assert len(body["strategies"]) == 2

        # Each strategy carries the full per-strategy payload.
        per_strat_keys = {
            "spec",
            "peer_slug",
            "beta_hedge",
            "n_obs",
            "n_trades",
            "sharpe",
            "hit_rate",
            "max_dd",
            "equity_curve",
        }
        for s in body["strategies"]:
            assert per_strat_keys.issubset(s.keys())
            assert s["max_dd"] <= 0.0
            assert isinstance(s["equity_curve"], list)
            assert len(s["equity_curve"]) > 0
            assert set(s["equity_curve"][0]) == {"t", "equity"}
            assert 0.0 <= s["hit_rate"] <= 1.0

        # Correlation matrix: 2×2, symmetric, 1.0 on the diagonal.
        corr = body["correlation"]
        assert len(corr) == 2 and len(corr[0]) == 2
        assert corr[0][0] == pytest.approx(1.0)
        assert corr[1][1] == pytest.approx(1.0)
        assert corr[0][1] == pytest.approx(corr[1][0])
        assert -1.0 - 1e-9 <= corr[0][1] <= 1.0 + 1e-9

        # Combined portfolio diagnostics.
        combo = body["combined_portfolio"]
        assert set(combo) == {"sharpe", "dd"}
        assert combo["dd"] <= 0.0
        assert isinstance(combo["sharpe"], float)

    def test_invalid_thresholds_return_400(self, app_client: TestClient) -> None:
        """If any single strategy fails the entry > exit / stop > entry
        invariants the whole call must 400 (not silently drop the bad
        strategy and return partial results)."""
        r = app_client.post(
            "/terminal/backtest-compare",
            json={
                "strategies": [
                    {
                        "slug": "trump-2028-win",
                        "side": "both",
                        "entry_z": 1.5,
                        "exit_z": 0.3,
                        "stop_z": 5.0,
                        "window": 20,
                    },
                    {
                        # entry_z <= exit_z → 400
                        "slug": "fed-cut-q3",
                        "side": "both",
                        "entry_z": 0.3,
                        "exit_z": 0.3,
                        "stop_z": 5.0,
                        "window": 20,
                    },
                ],
                "days": 365,
            },
        )
        assert r.status_code == 400
        assert "entry_z must exceed exit_z" in r.json()["detail"]

    def test_unknown_slug_returns_404(self, app_client: TestClient) -> None:
        """A slug missing from the alpha-hunter sweep must 404 the call."""
        r = app_client.post(
            "/terminal/backtest-compare",
            json={
                "strategies": [
                    {
                        "slug": "trump-2028-win",
                        "side": "both",
                        "entry_z": 1.5,
                        "exit_z": 0.3,
                        "stop_z": 5.0,
                        "window": 20,
                    },
                    {
                        "slug": "totally-made-up-market",
                        "side": "both",
                        "entry_z": 1.5,
                        "exit_z": 0.3,
                        "stop_z": 5.0,
                        "window": 20,
                    },
                ],
                "days": 365,
            },
        )
        assert r.status_code == 404
        assert "no cointegrated peer" in r.json()["detail"]
