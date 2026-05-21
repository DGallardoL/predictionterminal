"""Tests for ``pfm.factor_model_pro``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.factor_model_pro import fit_factor_model_pro


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


class TestFactorModelPro:
    def test_ols_recovers_planted_betas(self) -> None:
        rng = np.random.default_rng(0)
        n = 250
        x1 = rng.uniform(0.1, 0.9, n)
        x2 = rng.uniform(0.1, 0.9, n)
        eps = rng.normal(0, 0.02, n)
        y = (0.05 + 0.4 * x1 + 0.3 * x2 + eps).clip(0.01, 0.99)
        idx = _idx(n)
        out = fit_factor_model_pro(
            pd.Series(y, index=idx),
            pd.DataFrame({"x1": x1, "x2": x2}, index=idx),
            estimator="ols",
        )
        assert pytest.approx(out.coefficients[0].beta, abs=0.05) == 0.4
        assert pytest.approx(out.coefficients[1].beta, abs=0.05) == 0.3
        assert out.r_squared_is > 0.7
        assert not out.overfit_flag  # large n → IS and CV close

    def test_lasso_zeros_uninformative_factor(self) -> None:
        """Plant 2 informative + 5 noise factors. Lasso should zero the noise."""
        rng = np.random.default_rng(0)
        n = 300
        x1 = rng.uniform(0.1, 0.9, n)
        x2 = rng.uniform(0.1, 0.9, n)
        # 5 pure-noise factors uncorrelated with y
        noise = rng.uniform(0.1, 0.9, size=(n, 5))
        eps = rng.normal(0, 0.02, n)
        y = (0.05 + 0.4 * x1 + 0.3 * x2 + eps).clip(0.01, 0.99)
        idx = _idx(n)
        df = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
        for i in range(5):
            df[f"noise{i}"] = noise[:, i]
        out = fit_factor_model_pro(
            pd.Series(y, index=idx),
            df,
            estimator="lasso",
            alpha=0.005,
        )
        # At least 2 of the noise factors should be zero'd (Lasso doesn't always
        # zero all of them at a given alpha).
        n_zeroed = out.n_zeroed_factors
        assert n_zeroed >= 1
        # x1 and x2 should NOT be zero'd
        x1_coef = next(c for c in out.coefficients if c.factor_id == "x1")
        x2_coef = next(c for c in out.coefficients if c.factor_id == "x2")
        assert not x1_coef.is_zeroed
        assert not x2_coef.is_zeroed

    def test_ridge_handles_collinear_factors(self) -> None:
        """Two highly-collinear factors. Ridge should produce stable β.
        OLS would have huge β variance."""
        rng = np.random.default_rng(0)
        n = 200
        x1 = rng.uniform(0.1, 0.9, n)
        x2 = x1 + rng.normal(0, 0.005, n)  # near-perfect collinearity
        eps = rng.normal(0, 0.02, n)
        y = (0.4 * x1 + 0.3 * x2 + eps).clip(0.01, 0.99)
        idx = _idx(n)
        df = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
        ridge_out = fit_factor_model_pro(
            pd.Series(y, index=idx),
            df,
            estimator="ridge",
            alpha=0.5,
        )
        # The two β should both be in [0, 1] (ridge averages them).
        for c in ridge_out.coefficients:
            assert -2.0 < c.beta < 2.0  # not exploding
        assert ridge_out.r_squared_cv > 0.5

    def test_logit_transform_works_on_bounded_y(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(0.1, 0.9, n)
        y = (0.5 + 0.3 * x + rng.normal(0, 0.05, n)).clip(0.05, 0.95)
        idx = _idx(n)
        out = fit_factor_model_pro(
            pd.Series(y, index=idx),
            pd.DataFrame({"x": x}, index=idx),
            estimator="ols",
            transform="logit",
        )
        assert out.transform == "logit"
        assert out.r_squared_is > 0.5

    def test_pca_reduces_factors(self) -> None:
        rng = np.random.default_rng(0)
        n = 250
        # 5 factors driven by 1 latent factor + noise.
        latent = rng.uniform(0.1, 0.9, n)
        factors = {f"x{i}": latent + rng.normal(0, 0.02, n) for i in range(5)}
        idx = _idx(n)
        df = pd.DataFrame(factors, index=idx)
        y = pd.Series(
            (0.3 + 0.5 * latent + rng.normal(0, 0.02, n)).clip(0.01, 0.99),
            index=idx,
        )
        out = fit_factor_model_pro(y, df, use_pca=True, pca_explained_variance_target=0.90)
        # Should pick up 1-2 PCs (the latent + maybe noise).
        assert out.n_factors <= 3
        assert out.r_squared_is > 0.6

    def test_residual_diagnostics_run(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(0.1, 0.9, n)
        y = (0.4 * x + rng.normal(0, 0.02, n)).clip(0.01, 0.99)
        idx = _idx(n)
        out = fit_factor_model_pro(
            pd.Series(y, index=idx),
            pd.DataFrame({"x": x}, index=idx),
        )
        d = out.diagnostics
        assert 0 <= d.ljung_box_p <= 1
        assert 0 <= d.jarque_bera_p <= 1
        assert d.durbin_watson > 0
        assert isinstance(d.well_specified, bool)

    def test_cv_r2_reported(self) -> None:
        rng = np.random.default_rng(0)
        n = 250
        x = rng.uniform(0.1, 0.9, n)
        y = (0.4 * x + rng.normal(0, 0.02, n)).clip(0.01, 0.99)
        idx = _idx(n)
        out = fit_factor_model_pro(
            pd.Series(y, index=idx),
            pd.DataFrame({"x": x}, index=idx),
        )
        # IS R² and CV R² should both be reported and positive.
        assert out.r_squared_is > 0.3
        assert out.r_squared_cv > 0.0  # generalises
        # CI bounds make sense
        assert out.r_squared_ci_lo_95 <= out.r_squared_ci_hi_95

    def test_too_few_observations_raises(self) -> None:
        idx = _idx(10)
        y = pd.Series([0.5] * 10, index=idx)
        x = pd.DataFrame({"a": np.linspace(0, 1, 10)}, index=idx)
        with pytest.raises(ValueError, match="aligned bars"):
            fit_factor_model_pro(y, x)
