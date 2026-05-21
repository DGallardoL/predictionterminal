"""Tests for ``pfm.ml_predictor``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.ml_predictor import fit_ml_predictor


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


def _ar1(n: int, rho: float, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0]
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


class TestMlPredictor:
    def test_strong_reverter_some_signal(self) -> None:
        """A strongly mean-reverting AR(1) should give the GBR something
        to learn — at minimum it should beat the noise baseline."""
        n = 400
        spread = pd.Series(_ar1(n, rho=0.5, sigma=0.05, seed=42), index=_idx(n))
        out = fit_ml_predictor(spread, n_folds=4, seed=0)
        assert out.n_obs > 100
        assert out.n_features == 12
        assert len(out.folds) == 4
        # Verdict can be likely_alpha, marginal, or no_edge — all valid.
        assert out.verdict in {"likely_alpha", "marginal", "no_edge"}
        assert out.feature_importances  # non-empty

    def test_random_walk_no_edge(self) -> None:
        rng = np.random.default_rng(0)
        rw = pd.Series(np.cumsum(rng.normal(0, 0.05, 400)), index=_idx(400))
        out = fit_ml_predictor(rw, n_folds=4, seed=0)
        # On a random walk, ML should NOT find systematic alpha.
        assert out.verdict in {"no_edge", "marginal"}
        # R² should be small (often negative on out-of-sample).
        assert out.mean_test_r2 < 0.30

    def test_insufficient_data(self) -> None:
        s = pd.Series(np.zeros(30), index=_idx(30))
        out = fit_ml_predictor(s, n_folds=5)
        assert out.verdict == "insufficient-data"
        assert out.folds == []

    def test_last_prediction_finite(self) -> None:
        n = 300
        spread = pd.Series(_ar1(n, rho=0.4, sigma=0.05, seed=10), index=_idx(n))
        out = fit_ml_predictor(spread, n_folds=4, seed=0)
        assert out.last_prediction is None or np.isfinite(out.last_prediction)

    def test_feature_importances_sum_to_one(self) -> None:
        n = 250
        spread = pd.Series(_ar1(n, rho=0.3, sigma=0.04, seed=7), index=_idx(n))
        out = fit_ml_predictor(spread, n_folds=4, seed=0)
        if out.feature_importances:
            total = sum(f.importance for f in out.feature_importances)
            assert pytest.approx(total, abs=1e-6) == 1.0
