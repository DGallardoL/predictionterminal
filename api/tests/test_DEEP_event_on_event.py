"""Exhaustive synthetic-DGP recovery + edge case tests for event-on-event models.

Covers ``pfm.event_on_event`` (fit_event_on_event, event_correlation_matrix,
event_lead_lag, event_vector_autoregression, event_pca_decomposition) and the
``/event-model/*`` endpoints. All series synthetic; no network.

Sign / convention crib-sheet:
- ``fit_event_on_event``: regresses Δlogit(target) on Δlogit(predictors), HAC SEs.
- ``event_lead_lag``: ``best_lag > 0`` means *predictor leads target* by that many
  bars (because the function reports ``corr(target_t, predictor_{t-lag})`` for
  positive lag — a high correlation at +k says target moves k bars after the
  predictor's contemporaneous move).
- ``event_vector_autoregression``: ``granger_causality_matrix[i][j]`` is the
  p-value that factor j Granger-causes factor i (j → i).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pfm.event_on_event import (
    event_correlation_matrix,
    event_lead_lag,
    event_pca_decomposition,
    event_vector_autoregression,
    fit_event_on_event,
)

# --- shared synthetic helpers ---------------------------------------------


def _idx(n: int, start: str = "2025-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bank_fetcher(bank: dict[str, pd.Series]) -> Callable:
    def _fetch(fid: str, start: date, end: date) -> pd.Series:
        if fid not in bank:
            raise KeyError(f"unknown synthetic factor {fid!r}")
        return bank[fid]

    return _fetch


def _make_dgp_3preds(n: int, seed: int) -> dict[str, pd.Series]:
    """Δlogit(target) = 0.7 dl1 + 0.3 dl2 + 0.1 dl3 + ε."""
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    p1_logit = np.cumsum(rng.normal(0, 0.05, n))
    p2_logit = np.cumsum(rng.normal(0, 0.05, n))
    p3_logit = np.cumsum(rng.normal(0, 0.05, n))
    dl1 = np.diff(p1_logit, prepend=p1_logit[0])
    dl2 = np.diff(p2_logit, prepend=p2_logit[0])
    dl3 = np.diff(p3_logit, prepend=p3_logit[0])
    eps = rng.normal(0, 0.005, n)
    target_dlogit = 0.7 * dl1 + 0.3 * dl2 + 0.1 * dl3 + eps
    target_logit = np.cumsum(target_dlogit)
    return {
        "target": pd.Series(_logistic(target_logit), index=idx),
        "p1": pd.Series(_logistic(p1_logit), index=idx),
        "p2": pd.Series(_logistic(p2_logit), index=idx),
        "p3": pd.Series(_logistic(p3_logit), index=idx),
    }


# ===========================================================================
# 1. fit_event_on_event
# ===========================================================================


class TestFitEventOnEventDeep:
    def test_3pred_recovery_n300(self) -> None:
        bank = _make_dgp_3preds(300, seed=42)
        out = fit_event_on_event(
            target_factor_id="target",
            predictor_factor_ids=["p1", "p2", "p3"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            fetch_history=_bank_fetcher(bank),
        )
        assert 0.6 <= out["betas"]["p1"] <= 0.8, f"p1 beta {out['betas']['p1']}"
        assert 0.2 <= out["betas"]["p2"] <= 0.4, f"p2 beta {out['betas']['p2']}"
        # p3 is small but should still be ~0.1 ish; allow looseness.
        assert -0.05 <= out["betas"]["p3"] <= 0.3
        assert out["r_squared"] > 0.85
        assert out["n_obs"] >= 290

    def test_10_seeds_all_recover(self) -> None:
        passes = 0
        for seed in range(10):
            bank = _make_dgp_3preds(300, seed=seed)
            out = fit_event_on_event(
                target_factor_id="target",
                predictor_factor_ids=["p1", "p2", "p3"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )
            ok = 0.55 <= out["betas"]["p1"] <= 0.85 and 0.15 <= out["betas"]["p2"] <= 0.45
            if ok:
                passes += 1
        assert passes >= 9, f"only {passes}/10 seeds recovered cleanly"

    def test_small_n_50_recovers_with_wide_ci(self) -> None:
        bank = _make_dgp_3preds(50, seed=11)
        out = fit_event_on_event(
            target_factor_id="target",
            predictor_factor_ids=["p1", "p2"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            fetch_history=_bank_fetcher(bank),
        )
        # Point estimate near truth; CIs (proxied by t-stat) less tight.
        assert 0.4 <= out["betas"]["p1"] <= 1.0
        assert out["n_obs"] <= 50

    def test_n_below_minimum_raises(self) -> None:
        bank = _make_dgp_3preds(20, seed=0)
        with pytest.raises(ValueError, match="jointly-observed"):
            fit_event_on_event(
                target_factor_id="target",
                predictor_factor_ids=["p1", "p2"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )

    def test_predictors_identical_to_target_high_r2(self) -> None:
        rng = np.random.default_rng(3)
        n = 200
        idx = _idx(n)
        s = pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx)
        # Three predictors all equal to target → multicollinearity, but R²≈1.
        bank = {"target": s, "p1": s.copy(), "p2": s.copy()}
        out = fit_event_on_event(
            target_factor_id="target",
            predictor_factor_ids=["p1", "p2"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            fetch_history=_bank_fetcher(bank),
        )
        assert out["r_squared"] > 0.999
        # VIF should be huge for identical regressors (or inf).
        vif_vals = list(out["vif"].values())
        assert any(v > 100 or np.isinf(v) for v in vif_vals), f"VIF: {vif_vals}"

    def test_target_in_predictors_rejected(self) -> None:
        bank = _make_dgp_3preds(100, seed=0)
        with pytest.raises(ValueError, match="cannot also appear"):
            fit_event_on_event(
                target_factor_id="target",
                predictor_factor_ids=["target", "p1"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )

    def test_orthogonal_predictors_independent_tstats(self) -> None:
        rng = np.random.default_rng(99)
        n = 400
        idx = _idx(n)
        # Orthogonal random walks.
        p1_logit = np.cumsum(rng.normal(0, 0.1, n))
        p2_logit = np.cumsum(rng.normal(0, 0.1, n))
        dl1 = np.diff(p1_logit, prepend=p1_logit[0])
        dl2 = np.diff(p2_logit, prepend=p2_logit[0])
        eps = rng.normal(0, 0.005, n)
        target_dlogit = 0.5 * dl1 + 0.5 * dl2 + eps
        bank = {
            "target": pd.Series(_logistic(np.cumsum(target_dlogit)), index=idx),
            "p1": pd.Series(_logistic(p1_logit), index=idx),
            "p2": pd.Series(_logistic(p2_logit), index=idx),
        }
        out = fit_event_on_event(
            target_factor_id="target",
            predictor_factor_ids=["p1", "p2"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            fetch_history=_bank_fetcher(bank),
        )
        # Both predictors strongly significant.
        assert abs(out["t_stats"]["p1"]) > 5
        assert abs(out["t_stats"]["p2"]) > 5
        # VIF near 1 for orthogonal regressors.
        for v in out["vif"].values():
            assert v < 2.5, f"VIF too high for orthogonal: {v}"

    def test_nan_gaps_handled(self) -> None:
        bank = _make_dgp_3preds(200, seed=7)
        # Inject NaNs in p1 at scattered positions; fit should align by intersection.
        s = bank["p1"].copy()
        gap_idx = [10, 25, 60, 100, 150]
        s.iloc[gap_idx] = np.nan
        bank["p1"] = s
        out = fit_event_on_event(
            target_factor_id="target",
            predictor_factor_ids=["p1", "p2"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            fetch_history=_bank_fetcher(bank),
        )
        # n_obs reduced but fit still recovers approximately.
        assert out["n_obs"] < 200
        assert out["n_obs"] > 180
        assert 0.55 <= out["betas"]["p1"] <= 0.85

    def test_return_type_level_vs_dlogit_differ(self) -> None:
        bank = _make_dgp_3preds(200, seed=1)
        out_dlogit = fit_event_on_event(
            target_factor_id="target",
            predictor_factor_ids=["p1", "p2"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            fetch_history=_bank_fetcher(bank),
            return_type="delta_logit",
        )
        out_level = fit_event_on_event(
            target_factor_id="target",
            predictor_factor_ids=["p1", "p2"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            fetch_history=_bank_fetcher(bank),
            return_type="level",
        )
        assert out_dlogit["return_type"] == "delta_logit"
        assert out_level["return_type"] == "level"
        # Coefficients differ (level regression is biased & non-stationary).
        assert (
            abs(out_dlogit["betas"]["p1"] - out_level["betas"]["p1"]) > 0.01
            or abs(out_dlogit["r_squared"] - out_level["r_squared"]) > 0.001
        )

    def test_empty_predictors_raises(self) -> None:
        bank = _make_dgp_3preds(100, seed=0)
        with pytest.raises(ValueError, match="at least one predictor"):
            fit_event_on_event(
                target_factor_id="target",
                predictor_factor_ids=[],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )


# ===========================================================================
# 2. event_correlation_matrix
# ===========================================================================


def _make_corr_bank(n: int, seed: int) -> dict[str, pd.Series]:
    """5 factors with planted correlations:
    - a, b nearly identical (ρ=0.9)
    - c, d orthogonal
    - e correlated with all (ρ≈0.5)
    """
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    z_ab = rng.normal(0, 0.1, n)
    z_e = rng.normal(0, 0.1, n)
    a_dl = z_ab + rng.normal(0, 0.025, n)
    b_dl = z_ab + rng.normal(0, 0.025, n)
    c_dl = rng.normal(0, 0.1, n)
    d_dl = rng.normal(0, 0.1, n)
    # e: half from common signal of {a,b,c,d}, half its own.
    e_dl = 0.5 * (a_dl + b_dl + c_dl + d_dl) / 4.0 + z_e
    return {
        "a": pd.Series(_logistic(np.cumsum(a_dl)), index=idx),
        "b": pd.Series(_logistic(np.cumsum(b_dl)), index=idx),
        "c": pd.Series(_logistic(np.cumsum(c_dl)), index=idx),
        "d": pd.Series(_logistic(np.cumsum(d_dl)), index=idx),
        "e": pd.Series(_logistic(np.cumsum(e_dl)), index=idx),
    }


class TestCorrelationMatrixDeep:
    def test_planted_correlations_recovered(self) -> None:
        bank = _make_corr_bank(400, seed=1)
        out = event_correlation_matrix(
            factor_ids=["a", "b", "c", "d", "e"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            method="pearson",
            on="delta_logit",
            fetch_history=_bank_fetcher(bank),
        )
        m = np.array(out["matrix"])
        assert m.shape == (5, 5)
        # Diagonal = 1.
        for i in range(5):
            assert abs(m[i, i] - 1.0) < 1e-10
        # Symmetric.
        for i in range(5):
            for j in range(5):
                assert abs(m[i, j] - m[j, i]) < 1e-10
        # a, b high.
        assert m[0, 1] > 0.85
        # c, d low (≈ 0).
        assert abs(m[2, 3]) < 0.2
        # e correlated with all but to a milder degree.
        for k in range(4):
            assert m[4, k] > 0.0
        # avg_off_diagonal
        n = 5
        off = m[~np.eye(n, dtype=bool)]
        expected_avg = float(np.mean(off))
        assert abs(out["avg_off_diagonal"] - expected_avg) < 1e-9

    def test_spearman_close_to_pearson_for_linear(self) -> None:
        bank = _make_corr_bank(300, seed=2)
        out_p = event_correlation_matrix(
            factor_ids=["a", "b", "c"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            method="pearson",
            fetch_history=_bank_fetcher(bank),
        )
        out_s = event_correlation_matrix(
            factor_ids=["a", "b", "c"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            method="spearman",
            fetch_history=_bank_fetcher(bank),
        )
        # For roughly Gaussian innovations, Spearman ≈ Pearson within ~0.1.
        assert abs(out_p["matrix"][0][1] - out_s["matrix"][0][1]) < 0.15

    def test_kendall_runs(self) -> None:
        bank = _make_corr_bank(200, seed=3)
        out = event_correlation_matrix(
            factor_ids=["a", "b", "c"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            method="kendall",
            fetch_history=_bank_fetcher(bank),
        )
        m = out["matrix"]
        assert -1.0 <= m[0][1] <= 1.0
        # Kendall on a-b should still be strongly positive.
        assert m[0][1] > 0.4

    def test_on_level_vs_dlogit_differ(self) -> None:
        bank = _make_corr_bank(300, seed=4)
        out_dl = event_correlation_matrix(
            factor_ids=["a", "b", "c"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            on="delta_logit",
            fetch_history=_bank_fetcher(bank),
        )
        out_lv = event_correlation_matrix(
            factor_ids=["a", "b", "c"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            on="level",
            fetch_history=_bank_fetcher(bank),
        )
        # Levels are random-walk-like and tend to show spurious correlations.
        # Innovations (delta_logit) reveal the true relation. Just check they
        # differ materially on at least one off-diagonal.
        diffs = [
            abs(out_dl["matrix"][i][j] - out_lv["matrix"][i][j])
            for i in range(3)
            for j in range(3)
            if i != j
        ]
        assert max(diffs) > 0.05

    def test_hierarchical_order_is_permutation(self) -> None:
        bank = _make_corr_bank(200, seed=5)
        ids = ["a", "b", "c", "d", "e"]
        out = event_correlation_matrix(
            factor_ids=ids,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            fetch_history=_bank_fetcher(bank),
        )
        order = out["hierarchical_cluster_order"]
        assert sorted(order) == sorted(ids)

    def test_single_factor_input_raises(self) -> None:
        bank = _make_corr_bank(200, seed=6)
        with pytest.raises(ValueError, match="at least 2"):
            event_correlation_matrix(
                factor_ids=["a"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )

    def test_duplicate_ids_rejected(self) -> None:
        bank = _make_corr_bank(200, seed=7)
        with pytest.raises(ValueError, match="unique"):
            event_correlation_matrix(
                factor_ids=["a", "a"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )


# ===========================================================================
# 3. event_lead_lag
# ===========================================================================


class TestLeadLagDeep:
    def test_predictor_leads_target_by_2(self) -> None:
        rng = np.random.default_rng(0)
        n = 300
        idx = _idx(n)
        pred_logit = np.cumsum(rng.normal(0, 0.1, n))
        pred_dlogit = np.diff(pred_logit, prepend=pred_logit[0])
        target_dlogit = np.zeros(n)
        target_dlogit[2:] = pred_dlogit[:-2] + rng.normal(0, 0.002, n - 2)
        target_logit = np.cumsum(target_dlogit)
        bank = {
            "tgt": pd.Series(_logistic(target_logit), index=idx),
            "pred": pd.Series(_logistic(pred_logit), index=idx),
        }
        out = event_lead_lag(
            target_id="tgt",
            predictor_id="pred",
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            max_lag=5,
            fetch_history=_bank_fetcher(bank),
        )
        assert out["best_lag"] == 2
        assert out["best_correlation"] > 0.7
        assert not np.isnan(out["granger_p_predictor_leads"])
        assert out["granger_p_predictor_leads"] < 0.05
        # No spurious reverse causality.
        assert np.isnan(out["granger_p_target_leads"]) or out["granger_p_target_leads"] > 0.20

    def test_target_leads_predictor_negative_lag(self) -> None:
        """Build target_t = predictor_{t+1} (target leads by 1) → best_lag=-1."""
        rng = np.random.default_rng(11)
        n = 300
        idx = _idx(n)
        # Drive the target with an innovation, predictor follows one bar later.
        tgt_dl = rng.normal(0, 0.1, n)
        pred_dl = np.zeros(n)
        pred_dl[1:] = tgt_dl[:-1] + rng.normal(0, 0.002, n - 1)
        bank = {
            "tgt": pd.Series(_logistic(np.cumsum(tgt_dl)), index=idx),
            "pred": pd.Series(_logistic(np.cumsum(pred_dl)), index=idx),
        }
        out = event_lead_lag(
            target_id="tgt",
            predictor_id="pred",
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            max_lag=5,
            fetch_history=_bank_fetcher(bank),
        )
        assert out["best_lag"] == -1
        assert out["best_correlation"] > 0.7

    def test_no_causality_low_corr_at_all_lags(self) -> None:
        rng = np.random.default_rng(22)
        n = 400
        idx = _idx(n)
        bank = {
            "tgt": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "pred": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        out = event_lead_lag(
            target_id="tgt",
            predictor_id="pred",
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            max_lag=5,
            fetch_history=_bank_fetcher(bank),
        )
        # All correlations small (≤ 0.2 typical for n=400 independent).
        for row in out["ccf"]:
            assert abs(row["correlation"]) < 0.3, f"unexpected corr at lag {row['lag']}"

    def test_contemporaneous_best_lag_zero(self) -> None:
        rng = np.random.default_rng(33)
        n = 300
        idx = _idx(n)
        common = rng.normal(0, 0.1, n)
        bank = {
            "tgt": pd.Series(_logistic(np.cumsum(common + rng.normal(0, 0.005, n))), index=idx),
            "pred": pd.Series(_logistic(np.cumsum(common + rng.normal(0, 0.005, n))), index=idx),
        }
        out = event_lead_lag(
            target_id="tgt",
            predictor_id="pred",
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            max_lag=5,
            fetch_history=_bank_fetcher(bank),
        )
        assert out["best_lag"] == 0
        assert out["best_correlation"] > 0.85

    def test_max_lag_too_large_for_obs_raises(self) -> None:
        # Need T < max(20, 4*max_lag+2). With max_lag=10 → 42; pick T=30.
        rng = np.random.default_rng(44)
        n = 30
        idx = _idx(n)
        bank = {
            "tgt": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "pred": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        with pytest.raises(ValueError, match="jointly-observed"):
            event_lead_lag(
                target_id="tgt",
                predictor_id="pred",
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                max_lag=10,
                fetch_history=_bank_fetcher(bank),
            )

    def test_target_equals_predictor_id_rejected(self) -> None:
        bank = {"a": pd.Series([0.5] * 100, index=_idx(100))}
        with pytest.raises(ValueError, match="must differ"):
            event_lead_lag(
                target_id="a",
                predictor_id="a",
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )


# ===========================================================================
# 4. event_vector_autoregression
# ===========================================================================


class TestVARDeep:
    def test_bivariate_var1_recovers_coefs(self) -> None:
        """DGP: A = [[0.5, 0.2], [0.0, 0.7]]. Recover within ~0.15 tol."""
        rng = np.random.default_rng(7)
        n = 800
        idx = _idx(n)
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + 0.2 * y[t - 1] + rng.normal(0, 0.05)
            y[t] = 0.0 * x[t - 1] + 0.7 * y[t - 1] + rng.normal(0, 0.05)
        bank = {
            "x": pd.Series(_logistic(np.cumsum(x)), index=idx),
            "y": pd.Series(_logistic(np.cumsum(y)), index=idx),
        }
        out = event_vector_autoregression(
            factor_ids=["x", "y"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            lags=1,
            fetch_history=_bank_fetcher(bank),
        )
        # coefficients_matrix[lag][response_i][shock_j]
        # statsmodels orders coefs as A_{i,j} = effect of var j on var i.
        coefs = np.array(out["coefficients_matrix"])
        assert coefs.shape == (1, 2, 2)
        A = coefs[0]
        # We compare with a generous tolerance — single lag fit on Δlogit.
        # Because we cumsum'd then took Δlogit again, shape is preserved.
        assert abs(A[0, 0] - 0.5) < 0.20
        assert abs(A[0, 1] - 0.2) < 0.20
        assert abs(A[1, 0] - 0.0) < 0.20
        assert abs(A[1, 1] - 0.7) < 0.20

        # Granger: x is caused by y (cross-effect 0.2 → significant);
        # y is NOT caused by x.
        gm = out["granger_causality_matrix"]
        # gm[0][1] = p that y → x; should be small (causal).
        assert gm[0][1] < 0.10, f"expected significant y→x, got p={gm[0][1]}"
        # gm[1][0] = p that x → y; should be large.
        assert gm[1][0] > 0.10, f"expected non-significant x→y, got p={gm[1][0]}"

    def test_irf_first_3_periods_shape(self) -> None:
        rng = np.random.default_rng(9)
        n = 500
        idx = _idx(n)
        bank = {
            "x": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "y": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        out = event_vector_autoregression(
            factor_ids=["x", "y"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            lags=2,
            fetch_history=_bank_fetcher(bank),
        )
        irf = out["impulse_response_first_3_periods"]
        # 4 horizons (h=0..3) × 2 responses × 2 shocks.
        assert len(irf) >= 3
        for horizon in irf:
            assert len(horizon) == 2
            for row in horizon:
                assert len(row) == 2

    def test_fevd_rows_sum_to_one(self) -> None:
        rng = np.random.default_rng(13)
        n = 400
        idx = _idx(n)
        bank = {
            "a": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "b": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        out = event_vector_autoregression(
            factor_ids=["a", "b"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            lags=2,
            fetch_history=_bank_fetcher(bank),
        )
        for row in out["forecast_error_variance_decomposition"]:
            assert abs(sum(row) - 1.0) < 1e-6, f"FEVD row sum {sum(row)} != 1"

    def test_three_variable_var_matrix_3x3(self) -> None:
        rng = np.random.default_rng(17)
        n = 500
        idx = _idx(n)
        bank = {
            f"f{i}": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx)
            for i in range(3)
        }
        out = event_vector_autoregression(
            factor_ids=["f0", "f1", "f2"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            lags=1,
            fetch_history=_bank_fetcher(bank),
        )
        gm = out["granger_causality_matrix"]
        assert len(gm) == 3
        assert all(len(row) == 3 for row in gm)
        # Diagonal NaN.
        for i in range(3):
            assert np.isnan(gm[i][i])

    def test_var1_dgp_with_lags5_lag1_dominates(self) -> None:
        """Fit VAR(5) on VAR(1) DGP — lag-1 coefs should dominate later lags."""
        rng = np.random.default_rng(21)
        n = 800
        idx = _idx(n)
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + 0.2 * y[t - 1] + rng.normal(0, 0.05)
            y[t] = 0.7 * y[t - 1] + rng.normal(0, 0.05)
        bank = {
            "x": pd.Series(_logistic(np.cumsum(x)), index=idx),
            "y": pd.Series(_logistic(np.cumsum(y)), index=idx),
        }
        out = event_vector_autoregression(
            factor_ids=["x", "y"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            lags=5,
            fetch_history=_bank_fetcher(bank),
        )
        coefs = np.array(out["coefficients_matrix"])
        assert coefs.shape == (5, 2, 2)
        # Lag-1 magnitudes should typically exceed lag-5 magnitudes on average.
        m1 = float(np.mean(np.abs(coefs[0])))
        m5 = float(np.mean(np.abs(coefs[4])))
        assert m1 > m5, f"lag-1 mag {m1} not > lag-5 mag {m5}"

    def test_n_insufficient_for_var_raises(self) -> None:
        rng = np.random.default_rng(33)
        n = 25
        idx = _idx(n)
        bank = {
            "a": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "b": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        with pytest.raises(ValueError, match="jointly-observed"):
            event_vector_autoregression(
                factor_ids=["a", "b"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                lags=5,
                fetch_history=_bank_fetcher(bank),
            )

    def test_single_factor_var_rejected(self) -> None:
        bank = {"only": pd.Series([0.5] * 200, index=_idx(200))}
        with pytest.raises(ValueError, match=">= 2"):
            event_vector_autoregression(
                factor_ids=["only"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                lags=2,
                fetch_history=_bank_fetcher(bank),
            )


# ===========================================================================
# 5. event_pca_decomposition
# ===========================================================================


class TestPCADeep:
    def test_one_latent_drives_pc1_dominant(self) -> None:
        rng = np.random.default_rng(5)
        n = 400
        idx = _idx(n)
        latent = rng.normal(0, 0.2, n)
        weights = [1.0, 1.2, 0.8, 1.1, 0.9]
        bank = {}
        for i, w in enumerate(weights):
            idio = rng.normal(0, 0.02, n)
            dl = w * latent + idio
            bank[f"f{i}"] = pd.Series(_logistic(np.cumsum(dl)), index=idx)
        out = event_pca_decomposition(
            factor_ids=[f"f{i}" for i in range(5)],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            n_components=5,
            fetch_history=_bank_fetcher(bank),
        )
        evr = out["explained_variance_ratio"]
        assert evr[0] > 0.60, f"PC1 explained {evr[0]}, expected > 0.60"
        # All loadings of PC1 same sign.
        pc1 = out["loadings_matrix"][0]
        assert all(v > 0 for v in pc1) or all(v < 0 for v in pc1), f"PC1 loadings: {pc1}"

    def test_n_components_3_returns_3(self) -> None:
        rng = np.random.default_rng(8)
        n = 300
        idx = _idx(n)
        bank = {
            f"f{i}": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx)
            for i in range(5)
        }
        out = event_pca_decomposition(
            factor_ids=[f"f{i}" for i in range(5)],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            n_components=3,
            fetch_history=_bank_fetcher(bank),
        )
        assert out["n_components"] == 3
        assert len(out["loadings_matrix"]) == 3

    def test_orthogonal_5factors_each_pc_about_20pct(self) -> None:
        rng = np.random.default_rng(15)
        n = 800
        idx = _idx(n)
        bank = {
            f"f{i}": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx)
            for i in range(5)
        }
        out = event_pca_decomposition(
            factor_ids=[f"f{i}" for i in range(5)],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            n_components=5,
            fetch_history=_bank_fetcher(bank),
        )
        evr = out["explained_variance_ratio"]
        # Each component captures roughly 1/5 = 0.20 with finite-sample noise.
        for v in evr:
            assert 0.08 <= v <= 0.40, f"unexpected EVR {v}"

    def test_identical_factors_pc1_explains_all(self) -> None:
        rng = np.random.default_rng(2)
        n = 300
        idx = _idx(n)
        s = pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx)
        bank = {"a": s, "b": s.copy(), "c": s.copy()}
        out = event_pca_decomposition(
            factor_ids=["a", "b", "c"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            n_components=3,
            fetch_history=_bank_fetcher(bank),
        )
        evr = out["explained_variance_ratio"]
        assert evr[0] > 0.999

    def test_interpretation_strings_present(self) -> None:
        rng = np.random.default_rng(99)
        n = 300
        idx = _idx(n)
        latent = rng.normal(0, 0.2, n)
        bank = {
            f"f{i}": pd.Series(_logistic(np.cumsum(latent + rng.normal(0, 0.02, n))), index=idx)
            for i in range(4)
        }
        out = event_pca_decomposition(
            factor_ids=[f"f{i}" for i in range(4)],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            n_components=4,
            fetch_history=_bank_fetcher(bank),
        )
        interp = out["top_3_components_interpretation"]
        assert len(interp) >= 1
        kinds = {entry["kind"] for entry in interp}
        assert kinds.issubset({"broad_market", "spread"})
        # PC1 should be broad_market for one-latent design.
        assert interp[0]["component"] == 1
        assert interp[0]["kind"] == "broad_market"

    def test_pca_single_factor_rejected(self) -> None:
        bank = {"a": pd.Series([0.5] * 100, index=_idx(100))}
        with pytest.raises(ValueError, match=">= 2"):
            event_pca_decomposition(
                factor_ids=["a"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )


# ===========================================================================
# 6. API endpoint smoke tests
# ===========================================================================


@pytest.fixture
def api_bank() -> dict[str, pd.Series]:
    """Big bank covering all factors used by API tests below."""
    rng = np.random.default_rng(1234)
    n = 400
    idx = _idx(n, "2025-06-01")
    common = rng.normal(0, 0.15, n)
    bank: dict[str, pd.Series] = {}
    for fid in ["factor_t", "factor_p1", "factor_p2", "factor_p3", "factor_p4"]:
        idio = rng.normal(0, 0.05, n)
        bank[fid] = pd.Series(_logistic(np.cumsum(common + idio)), index=idx)
    # Lead-lag: predictor leads target by 2.
    pred_dl = rng.normal(0, 0.1, n)
    tgt_dl = np.zeros(n)
    tgt_dl[2:] = pred_dl[:-2] + rng.normal(0, 0.002, n - 2)
    bank["lead_pred"] = pd.Series(_logistic(np.cumsum(pred_dl)), index=idx)
    bank["lead_tgt"] = pd.Series(_logistic(np.cumsum(tgt_dl)), index=idx)
    return bank


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch, api_bank: dict[str, pd.Series]):
    """TestClient with the event-on-event router's fetcher mocked."""
    import pfm.event_on_event_router as router_mod
    import pfm.main as main_mod
    from pfm.cache import NullCache
    from pfm.cache_utils import get_cache

    # Force NullCache so module-level event-on-event cache doesn't pollute.
    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())
    # Clear the event-on-event router cache between fixture instantiations.
    get_cache("event_on_event").clear()

    def _fake_make_fetcher() -> Callable:
        def _fetch(fid: str, start: date, end: date) -> pd.Series:
            if fid not in api_bank:
                from fastapi import HTTPException

                raise HTTPException(status_code=502, detail=f"no history for {fid!r}")
            return api_bank[fid]

        return _fetch

    monkeypatch.setattr(router_mod, "_make_history_fetcher", _fake_make_fetcher)

    with TestClient(main_mod.app) as client:
        yield client


class TestApiEndpoints:
    def test_post_fit(self, api_client: TestClient) -> None:
        r = api_client.post(
            "/event-model/fit",
            json={
                "target_factor_id": "factor_t",
                "predictor_factor_ids": ["factor_p1", "factor_p2"],
                "start": "2025-06-01",
                "end": "2025-12-31",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["target"] == "factor_t"
        assert "betas" in body
        assert set(body["betas"].keys()) == {"factor_p1", "factor_p2"}
        assert "r_squared" in body
        assert body["n_obs"] > 100

    def test_post_correlation_matrix(self, api_client: TestClient) -> None:
        r = api_client.post(
            "/event-model/correlation-matrix",
            json={
                "factor_ids": ["factor_p1", "factor_p2", "factor_p3"],
                "start": "2025-06-01",
                "end": "2025-12-31",
                "method": "pearson",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["matrix"]) == 3
        for i in range(3):
            assert abs(body["matrix"][i][i] - 1.0) < 1e-9

    def test_post_lead_lag_recovers_lag_2(self, api_client: TestClient) -> None:
        r = api_client.post(
            "/event-model/lead-lag",
            json={
                "target_id": "lead_tgt",
                "predictor_id": "lead_pred",
                "start": "2025-06-01",
                "end": "2025-12-31",
                "max_lag": 5,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["best_lag"] == 2
        assert body["best_correlation"] > 0.7

    def test_post_var(self, api_client: TestClient) -> None:
        r = api_client.post(
            "/event-model/var",
            json={
                "factor_ids": ["factor_p1", "factor_p2"],
                "start": "2025-06-01",
                "end": "2025-12-31",
                "lags": 2,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["lags"] == 2
        assert len(body["coefficients_matrix"]) == 2

    def test_post_pca(self, api_client: TestClient) -> None:
        r = api_client.post(
            "/event-model/pca",
            json={
                "factor_ids": [
                    "factor_t",
                    "factor_p1",
                    "factor_p2",
                    "factor_p3",
                    "factor_p4",
                ],
                "start": "2025-06-01",
                "end": "2025-12-31",
                "n_components": 3,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_components"] == 3
        assert len(body["loadings_matrix"]) == 3
        assert len(body["explained_variance_ratio"]) == 3

    def test_start_after_end_rejected(self, api_client: TestClient) -> None:
        r = api_client.post(
            "/event-model/fit",
            json={
                "target_factor_id": "factor_t",
                "predictor_factor_ids": ["factor_p1"],
                "start": "2025-12-31",
                "end": "2025-06-01",
            },
        )
        assert r.status_code == 422

    def test_empty_predictor_list_rejected_by_pydantic(self, api_client: TestClient) -> None:
        r = api_client.post(
            "/event-model/fit",
            json={
                "target_factor_id": "factor_t",
                "predictor_factor_ids": [],
                "start": "2025-06-01",
                "end": "2025-12-31",
            },
        )
        assert r.status_code == 422

    def test_unknown_factor_returns_502(self, api_client: TestClient) -> None:
        r = api_client.post(
            "/event-model/fit",
            json={
                "target_factor_id": "unknown_factor_xyz",
                "predictor_factor_ids": ["factor_p1"],
                "start": "2025-06-01",
                "end": "2025-12-31",
            },
        )
        # The router now wraps unknown ids in a 400 with did_you_mean
        # suggestions; legacy callers that only checked for "not 200" still
        # work, but explicit upstream 502s are exposed as 400 now.
        assert r.status_code in (400, 422, 502)


# ===========================================================================
# 7. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_fit_empty_factor_history_raises(self) -> None:
        idx = _idx(0)
        bank = {
            "target": pd.Series([], dtype=float, index=idx),
            "p1": pd.Series([], dtype=float, index=idx),
        }
        with pytest.raises(ValueError):
            fit_event_on_event(
                target_factor_id="target",
                predictor_factor_ids=["p1"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )

    def test_correlation_matrix_too_few_obs_raises(self) -> None:
        idx = _idx(3)
        bank = {
            "a": pd.Series([0.4, 0.5, 0.6], index=idx),
            "b": pd.Series([0.4, 0.5, 0.6], index=idx),
        }
        with pytest.raises(ValueError, match="jointly-observed"):
            event_correlation_matrix(
                factor_ids=["a", "b"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )

    def test_var_lags_zero_rejected(self) -> None:
        rng = np.random.default_rng(0)
        n = 100
        idx = _idx(n)
        bank = {
            "a": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "b": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        with pytest.raises(ValueError, match=">= 1"):
            event_vector_autoregression(
                factor_ids=["a", "b"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                lags=0,
                fetch_history=_bank_fetcher(bank),
            )

    def test_lead_lag_max_lag_zero_rejected(self) -> None:
        rng = np.random.default_rng(0)
        n = 100
        idx = _idx(n)
        bank = {
            "a": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "b": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        with pytest.raises(ValueError, match=">= 1"):
            event_lead_lag(
                target_id="a",
                predictor_id="b",
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                max_lag=0,
                fetch_history=_bank_fetcher(bank),
            )

    def test_pca_n_components_zero_rejected(self) -> None:
        rng = np.random.default_rng(0)
        n = 100
        idx = _idx(n)
        bank = {
            "a": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "b": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        with pytest.raises(ValueError, match=">= 1"):
            event_pca_decomposition(
                factor_ids=["a", "b"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                n_components=0,
                fetch_history=_bank_fetcher(bank),
            )
