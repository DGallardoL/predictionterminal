"""Tests for ``pfm.event_model``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.event_model import event_model


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


class TestEventModel:
    def test_recovers_planted_betas(self) -> None:
        """y = 0.05 + 0.4·x1 + 0.3·x2 + ε.  HAC-OLS should recover."""
        rng = np.random.default_rng(0)
        n = 300
        x1 = rng.uniform(0.1, 0.9, n)
        x2 = rng.uniform(0.1, 0.9, n)
        eps = rng.normal(0, 0.02, n)
        y = (0.05 + 0.4 * x1 + 0.3 * x2 + eps).clip(0.01, 0.99)
        idx = _idx(n)
        out = event_model(
            pd.Series(y, index=idx),
            pd.DataFrame({"x1": x1, "x2": x2}, index=idx),
            target_id="y",
        )
        assert out.target_id == "y"
        assert out.n_obs == n
        coeff_x1 = next(c for c in out.coefficients if c.factor_id == "x1")
        coeff_x2 = next(c for c in out.coefficients if c.factor_id == "x2")
        assert pytest.approx(coeff_x1.beta, abs=0.05) == 0.4
        assert pytest.approx(coeff_x2.beta, abs=0.05) == 0.3
        assert coeff_x1.ci_lo < 0.4 < coeff_x1.ci_hi
        # R² should be quite high given the planted structure.
        assert out.r_squared > 0.7
        # F-stat should reject the null comfortably.
        assert out.f_pvalue < 0.001

    def test_independent_factors_low_r2(self) -> None:
        rng = np.random.default_rng(1)
        n = 250
        idx = _idx(n)
        y = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
        x = pd.DataFrame(
            {f"x{i}": rng.uniform(0.2, 0.8, n) for i in range(3)},
            index=idx,
        )
        out = event_model(y, x)
        assert out.r_squared < 0.10
        # F-test should not reject (no joint relationship).
        assert out.f_pvalue > 0.10
        # Most factors should have CI including 0 (allow ~1 Type I error).
        n_ci_excludes_zero = sum(1 for c in out.coefficients if c.ci_lo > 0 or c.ci_hi < 0)
        assert n_ci_excludes_zero <= 1

    def test_zero_variance_factor_rejected(self) -> None:
        n = 100
        idx = _idx(n)
        y = pd.Series(np.linspace(0.1, 0.9, n), index=idx)
        x = pd.DataFrame({"good": np.linspace(0.2, 0.8, n), "flat": [0.5] * n}, index=idx)
        with pytest.raises(ValueError, match="zero variance"):
            event_model(y, x)

    def test_too_few_observations_raises(self) -> None:
        idx = _idx(15)
        y = pd.Series(np.linspace(0, 1, 15), index=idx)
        x = pd.DataFrame({"a": np.linspace(0, 1, 15), "b": np.linspace(1, 0, 15)}, index=idx)
        with pytest.raises(ValueError, match="only 15 jointly-observed"):
            event_model(y, x)

    def test_predicted_plus_residual_equals_actual(self) -> None:
        rng = np.random.default_rng(7)
        n = 200
        idx = _idx(n)
        x = pd.DataFrame({"a": rng.uniform(0.1, 0.9, n)}, index=idx)
        y = pd.Series(0.5 * x["a"].values + rng.normal(0, 0.02, n), index=idx).clip(0, 1)
        out = event_model(y, x)
        # actual ≈ predicted + residual on the aligned index
        actual = out.actual.loc[out.predicted.index]
        recon = out.predicted + out.residuals
        assert (actual - recon).abs().max() < 1e-9

    def test_no_factors_raises(self) -> None:
        n = 30
        y = pd.Series(np.linspace(0, 1, n), index=_idx(n))
        x = pd.DataFrame(index=_idx(n))
        with pytest.raises(ValueError, match="at least one factor column"):
            event_model(y, x)

    def test_vif_reported_for_collinear_factors(self) -> None:
        rng = np.random.default_rng(9)
        n = 200
        idx = _idx(n)
        a = rng.uniform(0.1, 0.9, n)
        # b ≈ 0.95 * a + tiny noise → collinear
        b = 0.95 * a + rng.normal(0, 0.01, n)
        eps = rng.normal(0, 0.02, n)
        y = 0.4 * a + 0.3 * b + eps
        out = event_model(
            pd.Series(y, index=idx),
            pd.DataFrame({"a": a, "b": b}, index=idx),
        )
        # Both VIFs should be high (typically >10) given the high correlation.
        for c in out.coefficients:
            assert c.vif > 5.0
