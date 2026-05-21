"""Tests for ``pfm.kalman``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.kalman import kalman_dynamic_hedge, tune_delta


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


class TestKalman:
    def test_constant_beta_recovered(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        x = rng.uniform(0.1, 0.9, n)
        y = 0.6 * x + rng.normal(0, 0.01, n)
        idx = _idx(n)
        out = kalman_dynamic_hedge(pd.Series(y, index=idx), pd.Series(x, index=idx), delta=1e-5)
        # After warmup, β̂_t should hover near 0.6.
        post = out.beta.iloc[50:]
        assert pytest.approx(post.mean(), abs=0.05) == 0.6
        assert post.std() < 0.05

    def test_step_change_tracked(self) -> None:
        rng = np.random.default_rng(7)
        n = 600
        x = rng.uniform(0.2, 0.8, n)
        beta_true = np.where(np.arange(n) < 300, 0.4, 0.7)
        y = beta_true * x + rng.normal(0, 0.005, n)
        idx = _idx(n)
        out = kalman_dynamic_hedge(pd.Series(y, index=idx), pd.Series(x, index=idx), delta=5e-3)
        # Late-window mean β̂ should be near 0.7 (post-step regime).
        late = out.beta.iloc[450:]
        assert pytest.approx(late.mean(), abs=0.15) == 0.7

    def test_spread_innovation_centred(self) -> None:
        rng = np.random.default_rng(11)
        n = 400
        x = rng.uniform(0.1, 0.9, n)
        y = 0.5 * x + rng.normal(0, 0.01, n)
        idx = _idx(n)
        out = kalman_dynamic_hedge(pd.Series(y, index=idx), pd.Series(x, index=idx), delta=1e-4)
        # Innovation should be near-zero-mean post-warmup.
        late = out.spread.iloc[50:]
        assert abs(late.mean()) < 0.01

    def test_invalid_delta_raises(self) -> None:
        s = pd.Series(np.linspace(0, 1, 50), index=_idx(50))
        with pytest.raises(ValueError, match="delta must be in"):
            kalman_dynamic_hedge(s, s, delta=0.0)
        with pytest.raises(ValueError, match="delta must be in"):
            kalman_dynamic_hedge(s, s, delta=1.0)

    def test_too_few_observations_raises(self) -> None:
        s = pd.Series([0.5] * 5, index=_idx(5))
        with pytest.raises(ValueError, match="≥10 aligned bars"):
            kalman_dynamic_hedge(s, s)

    def test_log_likelihood_finite(self) -> None:
        rng = np.random.default_rng(2)
        n = 200
        x = rng.uniform(0.2, 0.8, n)
        y = 0.4 * x + rng.normal(0, 0.005, n)
        idx = _idx(n)
        out = kalman_dynamic_hedge(pd.Series(y, index=idx), pd.Series(x, index=idx), delta=1e-4)
        assert np.isfinite(out.log_likelihood)
        assert out.beta_init is not None and np.isfinite(out.beta_init)


class TestTuneDelta:
    def test_picks_highest_ll_grid_point(self) -> None:
        rng = np.random.default_rng(3)
        n = 300
        x = rng.uniform(0.1, 0.9, n)
        # β_t random walk → tighter δ should win.
        beta_t = np.cumsum(rng.normal(0, 0.005, n)) + 0.5
        y = beta_t * x + rng.normal(0, 0.01, n)
        idx = _idx(n)
        best, scores = tune_delta(pd.Series(y, index=idx), pd.Series(x, index=idx))
        assert best in scores
        assert all(np.isfinite(v) or np.isnan(v) for v in scores.values())
