"""Tests for the three "WOW" features added to ``/fit``.

Covers the additive nullable fields shipped 2026-05-15:

  * ``live_signal``        - predicted next-period return using the latest
                             Δlogit row + the just-fit coefficients.
  * ``pseudo_backtest``    - daily-rebalanced in-sample replay of
                             ``sign(predicted)`` with a flat transaction
                             cost.
  * ``factor_contributions`` - leave-one-out R² impact per factor.

Each helper is exercised both via the FastAPI ``TestClient`` (round-trip
through the schema) and directly with synthetic DGPs where the answer is
known a priori.

We verify:

  * Field shape and types on a happy-path /fit.
  * Recovery of a planted predictive signal (sign + rough magnitude).
  * Recovery of relative factor importance via LOO R².
  * Edge cases: short windows skip pseudo_backtest, single-factor fits
    skip factor_contributions, NaN-tail rows skip live_signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pfm.regression_router import (
    _compute_factor_contributions,
    _compute_live_signal,
    _compute_pseudo_backtest,
)
from pfm.schemas import FactorEstimateOut

# ---------------------------------------------------------------------------
# Helper: build a synthetic (y, X) where y = α + Σ β_i · x_i + ε so the
# helpers can be tested without a network round trip.
# ---------------------------------------------------------------------------


def _synthetic_design(
    *,
    n: int = 200,
    betas: tuple[float, ...] = (0.40, -0.25, 0.10),
    alpha: float = 0.0005,
    noise: float = 0.005,
    seed: int = 42,
    start: str = "2024-01-01",
) -> tuple[pd.Series, pd.DataFrame, list[FactorEstimateOut], float]:
    """Return (y, X, factor_estimates, alpha) with planted coefficients."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    cols = [f"f{i}" for i in range(len(betas))]
    X_arr = rng.normal(0, 0.30, size=(n, len(betas)))
    eps = rng.normal(0, noise, n)
    y_arr = alpha + X_arr @ np.array(betas) + eps
    X = pd.DataFrame(X_arr, index=idx, columns=cols)
    y = pd.Series(y_arr, index=idx, name="r")
    # Build FactorEstimateOut from the known DGP so callers don't have to
    # refit just to exercise the helper.
    estimates = [
        FactorEstimateOut(
            id=cols[i],
            beta=float(betas[i]),
            std_err=0.05,
            t_stat=float(betas[i] / 0.05),
            p_value=0.01,
            ci_low=float(betas[i] - 0.10),
            ci_high=float(betas[i] + 0.10),
        )
        for i in range(len(betas))
    ]
    return y, X, estimates, alpha


# ===========================================================================
# Feature 1 — live_signal
# ===========================================================================


class TestLiveSignal:
    def test_recovers_known_prediction(self) -> None:
        """y = 0.40·f0 - 0.25·f1 + 0.10·f2 + tiny noise → live signal on the
        last row must equal α + β·x_last to within numerical precision."""
        y, X, estimates, alpha = _synthetic_design(noise=1e-6)
        out = _compute_live_signal(y, X, estimates, alpha, low_confidence=False)
        assert out is not None
        # Reproduce the expected prediction by hand.
        last_row = X.iloc[-1]
        expected = alpha + sum(est.beta * float(last_row[est.id]) for est in estimates)
        assert abs(out.predicted_return - expected) < 1e-9

    def test_ci_brackets_prediction(self) -> None:
        y, X, estimates, alpha = _synthetic_design()
        out = _compute_live_signal(y, X, estimates, alpha, low_confidence=False)
        assert out is not None
        assert out.ci_95_lo <= out.predicted_return <= out.ci_95_hi
        assert out.std_err >= 0.0

    def test_edge_bp_is_abs_prediction_times_1e4(self) -> None:
        y, X, estimates, alpha = _synthetic_design(betas=(0.50,), noise=1e-5)
        out = _compute_live_signal(y, X, estimates, alpha, low_confidence=False)
        assert out is not None
        assert abs(out.edge_bp - abs(out.predicted_return) * 1e4) < 1e-6

    def test_returns_none_when_last_row_has_nan(self) -> None:
        y, X, estimates, alpha = _synthetic_design()
        # Inject NaN into the last row to mimic a missing-factor day.
        X.iloc[-1, 0] = np.nan
        out = _compute_live_signal(y, X, estimates, alpha, low_confidence=False)
        assert out is None

    def test_returns_none_on_empty_design(self) -> None:
        y = pd.Series(dtype=float)
        X = pd.DataFrame()
        out = _compute_live_signal(y, X, [], 0.0, low_confidence=False)
        assert out is None

    def test_low_confidence_flag_propagates(self) -> None:
        y, X, estimates, alpha = _synthetic_design()
        out = _compute_live_signal(y, X, estimates, alpha, low_confidence=True)
        assert out is not None
        assert out.low_confidence is True

    def test_latest_factor_logits_match_last_row(self) -> None:
        y, X, estimates, alpha = _synthetic_design()
        out = _compute_live_signal(y, X, estimates, alpha, low_confidence=False)
        assert out is not None
        last = X.iloc[-1]
        for col in X.columns:
            assert out.latest_factor_logits[col] == pytest.approx(float(last[col]))


# ===========================================================================
# Feature 2 — pseudo_backtest
# ===========================================================================


class TestPseudoBacktest:
    def test_skipped_when_n_obs_below_min(self) -> None:
        y, _X, _e, _a = _synthetic_design(n=20)
        pred = pd.Series(np.linspace(-0.01, 0.01, 20), index=y.index)
        out = _compute_pseudo_backtest(y, pred, min_obs=30)
        assert out is None

    def test_perfect_signal_makes_money(self) -> None:
        """When predicted == actual, sign(predicted) wins every day → equity
        curve must end > 1.0 net of the transaction cost."""
        y, _X, _e, _a = _synthetic_design(n=120, noise=0.0)
        # Predicted = actual → hit rate 100%.
        out = _compute_pseudo_backtest(y, y, transaction_cost_bp=1.0)
        assert out is not None
        assert out.hit_rate >= 0.95
        assert out.total_return > 0.0

    def test_inverted_signal_loses_money(self) -> None:
        """When predicted == -actual, every trade is wrong → hit_rate near 0
        and total_return negative."""
        y, _X, _e, _a = _synthetic_design(n=120, noise=0.0)
        pred = -y
        out = _compute_pseudo_backtest(y, pred, transaction_cost_bp=1.0)
        assert out is not None
        assert out.hit_rate <= 0.05
        assert out.total_return < 0.0

    def test_equity_curve_length_matches_y(self) -> None:
        y, _X, _e, _a = _synthetic_design(n=90)
        pred = pd.Series(np.zeros(90), index=y.index)
        # All-zero predicted → no position → flat equity
        out = _compute_pseudo_backtest(y, pred, transaction_cost_bp=0.0)
        assert out is not None
        assert len(out.equity_curve) == 90
        # Flat position means equity stays at 1.0
        assert out.equity_curve[-1].equity == pytest.approx(1.0, abs=1e-9)

    def test_transaction_cost_reduces_return(self) -> None:
        y, _X, _e, _a = _synthetic_design(n=120, noise=0.0)
        cheap = _compute_pseudo_backtest(y, y, transaction_cost_bp=0.0)
        pricey = _compute_pseudo_backtest(y, y, transaction_cost_bp=50.0)
        assert cheap is not None and pricey is not None
        assert cheap.total_return > pricey.total_return

    def test_max_drawdown_is_negative_or_zero(self) -> None:
        y, _X, _e, _a = _synthetic_design(n=120)
        pred = y  # perfect signal still has tiny DD on volatile days
        out = _compute_pseudo_backtest(y, pred, transaction_cost_bp=1.0)
        assert out is not None
        assert out.max_drawdown <= 0.0

    def test_n_trades_counts_position_flips(self) -> None:
        """Alternating +1/-1 predicted should produce ~n_obs trades."""
        n = 60
        y, _X, _e, _a = _synthetic_design(n=n)
        pred = pd.Series(
            [0.01 if i % 2 == 0 else -0.01 for i in range(n)],
            index=y.index,
        )
        out = _compute_pseudo_backtest(y, pred, transaction_cost_bp=1.0)
        assert out is not None
        # Every day is a flip → exactly n trades.
        assert out.n_trades == n

    def test_returns_none_on_length_mismatch(self) -> None:
        y, _X, _e, _a = _synthetic_design(n=40)
        pred = pd.Series([0.0] * 30, index=y.index[:30])
        out = _compute_pseudo_backtest(y, pred)
        assert out is None


# ===========================================================================
# Feature 3 — factor_contributions (leave-one-out R²)
# ===========================================================================


class TestFactorContributions:
    def test_skipped_with_single_factor(self) -> None:
        y, X, _e, _a = _synthetic_design(betas=(0.5,))
        out = _compute_factor_contributions(y, X, 0.5)
        assert out is None

    def test_ranks_high_beta_factor_first(self) -> None:
        """β = (0.50, 0.10, 0.0). f0 should have the largest ΔR², f2 ~ 0."""
        y, X, _e, _a = _synthetic_design(
            betas=(0.50, 0.10, 0.0),
            noise=0.001,
            n=400,
        )
        # Fit on the full design to get a baseline R².
        import statsmodels.api as sm

        X_const = sm.add_constant(X, has_constant="add")
        full_r2 = float(sm.OLS(y.values, X_const.values).fit().rsquared)
        out = _compute_factor_contributions(y, X, full_r2)
        assert out is not None
        assert len(out) == 3
        # f0 should be the strongest contributor.
        assert out[0].factor_id == "f0"
        # f0 contributes far more than f2.
        f0 = next(c for c in out if c.factor_id == "f0")
        f2 = next(c for c in out if c.factor_id == "f2")
        assert f0.delta_r_squared > f2.delta_r_squared

    def test_redundant_factor_shows_near_zero_delta(self) -> None:
        """When f1 = f0 + noise, dropping f1 barely moves R² since f0
        already carries the signal."""
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        rng = np.random.default_rng(99)
        x0 = rng.normal(0, 0.30, n)
        x1 = x0 + rng.normal(0, 0.01, n)  # near-duplicate of x0
        y_arr = 0.50 * x0 + rng.normal(0, 0.005, n)
        X = pd.DataFrame({"f0": x0, "f1": x1}, index=idx)
        y = pd.Series(y_arr, index=idx, name="r")

        import statsmodels.api as sm

        X_const = sm.add_constant(X, has_constant="add")
        full_r2 = float(sm.OLS(y.values, X_const.values).fit().rsquared)
        out = _compute_factor_contributions(y, X, full_r2)
        assert out is not None
        # The redundant factor's ΔR² should be small in absolute terms.
        f1 = next(c for c in out if c.factor_id == "f1")
        assert abs(f1.delta_r_squared) < 0.05

    def test_share_of_explained_sums_to_at_most_one(self) -> None:
        y, X, _e, _a = _synthetic_design(betas=(0.4, 0.3, 0.2, 0.1), n=200)
        out = _compute_factor_contributions(y, X, 0.30)
        assert out is not None
        total = sum(c.share_of_explained_r_squared for c in out)
        assert total <= 1.0 + 1e-6

    def test_sorted_descending_by_delta(self) -> None:
        y, X, _e, _a = _synthetic_design(
            betas=(0.10, 0.50, 0.30, 0.05),
            noise=0.001,
            n=300,
        )
        import statsmodels.api as sm

        full_r2 = float(sm.OLS(y.values, sm.add_constant(X.values)).fit().rsquared)
        out = _compute_factor_contributions(y, X, full_r2)
        assert out is not None
        from itertools import pairwise

        for a, b in pairwise(out):
            assert a.delta_r_squared >= b.delta_r_squared


# ===========================================================================
# End-to-end: /fit returns the three new fields with the right types.
# ===========================================================================


class TestFitResponseWowFields:
    def _post(self, app_client: TestClient, **overrides: Any) -> dict[str, Any]:
        payload = {
            "ticker": "TEST",
            "factors": ["factor_a", "factor_b"],
            "start": "2025-06-01",
            "end": "2025-12-31",
        }
        payload.update(overrides)
        r = app_client.post("/fit", json=payload)
        assert r.status_code == 200, r.text
        return r.json()

    def test_all_three_keys_present(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        for key in ("live_signal", "pseudo_backtest", "factor_contributions"):
            assert key in body, key

    def test_live_signal_shape(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        ls = body["live_signal"]
        if ls is None:
            pytest.skip("design matrix had no usable last row")
        assert {
            "predicted_return",
            "std_err",
            "ci_95_lo",
            "ci_95_hi",
            "edge_bp",
            "latest_date",
            "latest_factor_logits",
            "low_confidence",
        } <= set(ls.keys())
        assert ls["ci_95_lo"] <= ls["predicted_return"] <= ls["ci_95_hi"]
        assert ls["edge_bp"] >= 0.0

    def test_pseudo_backtest_shape(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        pb = body["pseudo_backtest"]
        if pb is None:
            pytest.skip("n_obs below the pseudo-backtest floor")
        assert {
            "equity_curve",
            "total_return",
            "annualized_sharpe",
            "max_drawdown",
            "hit_rate",
            "n_trades",
            "transaction_cost_bp",
            "note",
        } <= set(pb.keys())
        assert 0.0 <= pb["hit_rate"] <= 1.0
        assert pb["max_drawdown"] <= 0.0
        assert len(pb["equity_curve"]) == body["n_obs"]
        # Default cost is 5 bp.
        assert pb["transaction_cost_bp"] == pytest.approx(5.0)

    def test_factor_contributions_shape_when_multiple_factors(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        fc = body["factor_contributions"]
        if fc is None:
            pytest.skip("only one factor or empty design")
        assert isinstance(fc, list)
        for item in fc:
            assert {
                "factor_id",
                "delta_r_squared",
                "share_of_explained_r_squared",
            } <= set(item.keys())
            assert 0.0 <= item["share_of_explained_r_squared"] <= 1.0
        # Sorted descending by delta_r_squared.
        from itertools import pairwise

        for a, b in pairwise(fc):
            assert a["delta_r_squared"] >= b["delta_r_squared"]

    def test_factor_contributions_null_with_single_factor(self, app_client: TestClient) -> None:
        body = self._post(app_client, factors=["factor_a"])
        assert body["factor_contributions"] is None

    def test_low_confidence_flag_set_on_weak_fit(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        ls = body["live_signal"]
        if ls is None:
            pytest.skip("no live_signal in this fit")
        # The flag must agree with the verdict bucket.
        if body["verdict"] in {"weak_fit", "underpowered"}:
            assert ls["low_confidence"] is True
        else:
            assert ls["low_confidence"] is False
