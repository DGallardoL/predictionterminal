"""Tests for ``pfm.event_on_event``.

All tests run on synthetic probability series — no network IO. The
``fetch_history`` callable is the unit under test in some tests and is a
deterministic in-memory bank in others.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pfm.event_on_event import (
    event_correlation_matrix,
    event_lead_lag,
    event_pca_decomposition,
    event_vector_autoregression,
    fit_event_on_event,
)

# --- fixtures --------------------------------------------------------------


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bank_fetcher(bank: dict[str, pd.Series]) -> Callable:
    """Closure that looks up ``factor_id`` in ``bank``."""

    def _fetch(fid: str, start: date, end: date) -> pd.Series:
        if fid not in bank:
            raise KeyError(f"unknown synthetic factor {fid!r}")
        s = bank[fid]
        # Window bounds aren't enforced strictly — synthetic series cover full range.
        return s

    return _fetch


# --- 1. fit_event_on_event --------------------------------------------------


class TestFitEventOnEvent:
    def test_recovers_planted_betas(self) -> None:
        """Δlogit(target) = 0.7·Δlogit(p1) + 0.3·Δlogit(p2) + noise.

        The HAC-OLS fit must recover the betas within tolerance.
        """
        rng = np.random.default_rng(42)
        n = 400
        idx = _make_index(n)

        # Build predictor logits as random walks, then convert back to probs.
        p1_logit = np.cumsum(rng.normal(0, 0.05, n))
        p2_logit = np.cumsum(rng.normal(0, 0.05, n))
        p1 = pd.Series(_logistic(p1_logit), index=idx)
        p2 = pd.Series(_logistic(p2_logit), index=idx)

        # Construct the target so its Δlogit is 0.7·Δlogit(p1) + 0.3·Δlogit(p2)
        # (+ small noise) and re-invert to a probability path.
        dl1 = np.diff(p1_logit, prepend=p1_logit[0])
        dl2 = np.diff(p2_logit, prepend=p2_logit[0])
        eps = rng.normal(0, 0.005, n)
        target_dlogit = 0.7 * dl1 + 0.3 * dl2 + eps
        target_logit = np.cumsum(target_dlogit)
        target = pd.Series(_logistic(target_logit), index=idx)

        bank = {"target": target, "p1": p1, "p2": p2}
        fetcher = _bank_fetcher(bank)

        out = fit_event_on_event(
            target_factor_id="target",
            predictor_factor_ids=["p1", "p2"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            fetch_history=fetcher,
        )

        assert out["target"] == "target"
        assert out["predictors"] == ["p1", "p2"]
        assert out["n_obs"] == n - 1  # one row dropped for diff
        assert pytest.approx(out["betas"]["p1"], abs=0.05) == 0.7
        assert pytest.approx(out["betas"]["p2"], abs=0.05) == 0.3
        # Both predictors should be highly significant.
        assert abs(out["t_stats"]["p1"]) > 5.0
        assert abs(out["t_stats"]["p2"]) > 3.0
        # R² high since the noise is small.
        assert out["r_squared"] > 0.9
        # Diagnostics present.
        assert "vif" in out and "p1" in out["vif"]
        assert "durbin_watson" in out
        assert "condition_number" in out
        assert "residuals_summary" in out
        assert out["return_type"] == "delta_logit"

    def test_target_in_predictors_raises(self) -> None:
        idx = _make_index(50)
        s = pd.Series(np.linspace(0.2, 0.8, 50), index=idx)
        bank = {"a": s, "b": s.copy()}
        with pytest.raises(ValueError, match="cannot also appear"):
            fit_event_on_event(
                target_factor_id="a",
                predictor_factor_ids=["a", "b"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )

    def test_too_few_observations_raises(self) -> None:
        idx = _make_index(15)
        rng = np.random.default_rng(0)
        bank = {
            "y": pd.Series(_logistic(rng.normal(0, 1, 15)), index=idx),
            "x": pd.Series(_logistic(rng.normal(0, 1, 15)), index=idx),
        }
        with pytest.raises(ValueError, match="jointly-observed"):
            fit_event_on_event(
                target_factor_id="y",
                predictor_factor_ids=["x"],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                fetch_history=_bank_fetcher(bank),
            )


# --- 2. event_correlation_matrix -------------------------------------------


class TestCorrelationMatrix:
    def test_correlated_pair_shows_high_offdiag(self) -> None:
        """Three factors, two of which are nearly identical → high pair corr."""
        rng = np.random.default_rng(7)
        n = 200
        idx = _make_index(n)
        # Driver Δlogit innovation.
        z = rng.normal(0, 0.1, n)
        # a and b share the same driver (with tiny independent noise) → corr ≈ 1.
        a_logit = np.cumsum(z + rng.normal(0, 0.005, n))
        b_logit = np.cumsum(z + rng.normal(0, 0.005, n))
        # c is independent.
        c_logit = np.cumsum(rng.normal(0, 0.1, n))

        bank = {
            "a": pd.Series(_logistic(a_logit), index=idx),
            "b": pd.Series(_logistic(b_logit), index=idx),
            "c": pd.Series(_logistic(c_logit), index=idx),
        }

        out = event_correlation_matrix(
            factor_ids=["a", "b", "c"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            method="pearson",
            on="delta_logit",
            fetch_history=_bank_fetcher(bank),
        )
        assert out["factor_ids"] == ["a", "b", "c"]
        m = out["matrix"]
        # diagonal must be 1.0
        for i in range(3):
            assert pytest.approx(m[i][i], abs=1e-9) == 1.0
        # corr(a, b) >> corr(a, c)
        assert m[0][1] > 0.9
        assert abs(m[0][2]) < 0.3
        assert "hierarchical_cluster_order" in out
        assert set(out["hierarchical_cluster_order"]) == {"a", "b", "c"}

    def test_method_spearman_runs(self) -> None:
        rng = np.random.default_rng(2)
        n = 150
        idx = _make_index(n)
        bank = {
            "x": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "y": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        out = event_correlation_matrix(
            factor_ids=["x", "y"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            method="spearman",
            fetch_history=_bank_fetcher(bank),
        )
        assert out["method"] == "spearman"
        assert -1.0 <= out["matrix"][0][1] <= 1.0


# --- 3. event_lead_lag ------------------------------------------------------


class TestLeadLag:
    def test_recovers_lag_2(self) -> None:
        """Build target = predictor.shift(2) so predictor leads target by 2 days.

        Sign convention: ``lag > 0`` ⇒ predictor leads target. So we expect
        ``best_lag == 2``.
        """
        rng = np.random.default_rng(0)
        n = 300
        idx = _make_index(n)
        # Predictor: random-walk logit.
        pred_logit = np.cumsum(rng.normal(0, 0.1, n))
        pred_dlogit = np.diff(pred_logit, prepend=pred_logit[0])
        # Target Δlogit at time t equals predictor Δlogit at t-2 (plus tiny noise).
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
        assert out["target"] == "tgt"
        assert out["predictor"] == "pred"
        assert out["best_lag"] == 2
        assert out["best_correlation"] > 0.9
        # Granger: predictor leads target ⇒ p_predictor_leads should be small.
        assert not np.isnan(out["granger_p_predictor_leads"])
        assert out["granger_p_predictor_leads"] < 0.05


# --- 4. event_vector_autoregression ----------------------------------------


class TestVAR:
    def test_independent_factors_granger_near_identity(self) -> None:
        """Two truly independent random walks → off-diagonal Granger p > 0.05."""
        rng = np.random.default_rng(123)
        n = 400
        idx = _make_index(n)
        bank = {
            "x": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
            "y": pd.Series(_logistic(np.cumsum(rng.normal(0, 0.1, n))), index=idx),
        }
        out = event_vector_autoregression(
            factor_ids=["x", "y"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            lags=3,
            fetch_history=_bank_fetcher(bank),
        )
        assert out["factor_ids"] == ["x", "y"]
        assert out["lags"] == 3
        gm = out["granger_causality_matrix"]
        # Diagonal is NaN; off-diagonals should be large p (no causality).
        assert np.isnan(gm[0][0])
        assert np.isnan(gm[1][1])
        # With independent inputs the causality test should *typically* not
        # reject at 0.05 — allow margin since this is stochastic.
        assert gm[0][1] > 0.05 or gm[1][0] > 0.05
        # Coefficient cube has shape (lags, k, k).
        coefs = out["coefficients_matrix"]
        assert len(coefs) == 3
        assert len(coefs[0]) == 2
        assert len(coefs[0][0]) == 2
        # IRF and FEVD are populated.
        assert len(out["impulse_response_first_3_periods"]) >= 3
        fevd = out["forecast_error_variance_decomposition"]
        # Each FEVD row should sum to ~1 (variance shares).
        for row in fevd:
            assert pytest.approx(sum(row), abs=1e-6) == 1.0


# --- 5. event_pca_decomposition --------------------------------------------


class TestPCA:
    def test_first_component_dominates_when_one_driver(self) -> None:
        """3 factors driven by one common shock + tiny idiosyncratic noise.

        First PC should explain >60% of total variance.
        """
        rng = np.random.default_rng(5)
        n = 400
        idx = _make_index(n)
        # Common driver (Δlogit innovations).
        common = rng.normal(0, 0.2, n)
        bank = {}
        for fid in ("a", "b", "c"):
            idio = rng.normal(0, 0.02, n)
            dl = common + idio
            bank[fid] = pd.Series(_logistic(np.cumsum(dl)), index=idx)

        out = event_pca_decomposition(
            factor_ids=["a", "b", "c"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            n_components=3,
            fetch_history=_bank_fetcher(bank),
        )
        assert out["factor_ids"] == ["a", "b", "c"]
        assert out["n_components"] == 3
        evr = out["explained_variance_ratio"]
        assert evr[0] > 0.60
        # Loadings: 3 components × 3 factors.
        assert len(out["loadings_matrix"]) == 3
        assert len(out["loadings_matrix"][0]) == 3
        # Top-3 interpretation present.
        interp = out["top_3_components_interpretation"]
        assert len(interp) >= 1
        assert interp[0]["component"] == 1
        # PC1 should be a broad-market factor (all loadings same sign).
        assert interp[0]["kind"] == "broad_market"
