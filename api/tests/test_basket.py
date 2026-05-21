"""Tests for ``pfm.basket`` — PCA-residual stat-arb + Kelly sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.basket import basket_pca_residuals, kelly_fraction


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


class TestBasketPcaResiduals:
    def test_planted_common_factor_recovered(self) -> None:
        """Build 5 series sharing a common factor + idiosyncratic noise.
        PCA should explain most variance with 1 component, residuals tight."""
        rng = np.random.default_rng(0)
        n = 250
        idx = _idx(n)
        common = rng.normal(0, 0.10, n)
        loadings = [1.0, 1.1, 0.9, 1.0, 1.05]
        idio = rng.normal(0, 0.02, size=(n, len(loadings)))
        prices = np.outer(common, np.ones(len(loadings))) * np.array(loadings) + idio + 0.5
        df = pd.DataFrame(
            prices,
            index=idx,
            columns=[f"x{i}" for i in range(len(loadings))],
        )

        res = basket_pca_residuals(df, n_components=1, z_window=20)
        assert res.n_components_used == 1
        # First component should explain >80% of variance.
        assert res.explained_variance_ratio[0] > 0.80
        # Residuals should be much smaller than the original series std.
        for c in df.columns:
            assert res.residuals[c].std() < df[c].std() * 0.5

    def test_residual_z_scored(self) -> None:
        rng = np.random.default_rng(11)
        n = 200
        idx = _idx(n)
        common = rng.normal(0, 0.10, n)
        idio = rng.normal(0, 0.02, size=(n, 3))
        prices = np.outer(common, np.ones(3)) + idio + 0.4
        df = pd.DataFrame(prices, index=idx, columns=list("abc"))
        res = basket_pca_residuals(df, n_components=1, z_window=20)
        # After warm-up, residual z-scores should mostly live in [-5, 5].
        zv = res.z_residuals.dropna()
        assert (zv.abs() < 5).all().all()

    def test_n_components_auto_picks_target_variance(self) -> None:
        rng = np.random.default_rng(2)
        n = 200
        idx = _idx(n)
        # Two independent factors driving 4 series.
        f1 = rng.normal(0, 0.1, n)
        f2 = rng.normal(0, 0.1, n)
        prices = (
            np.column_stack(
                [
                    f1 + rng.normal(0, 0.01, n),
                    f1 + rng.normal(0, 0.01, n),
                    f2 + rng.normal(0, 0.01, n),
                    f2 + rng.normal(0, 0.01, n),
                ]
            )
            + 0.5
        )
        df = pd.DataFrame(prices, index=idx, columns=list("abcd"))
        # Auto: target 70% of variance.
        res = basket_pca_residuals(df, n_components=None, explained_variance_target=0.70)
        # Two factors generated → either 1 or 2 components hit 70%.
        assert res.n_components_used in (1, 2)

    def test_too_few_columns_raises(self) -> None:
        df = pd.DataFrame({"a": [0.5] * 50}, index=_idx(50))
        with pytest.raises(ValueError, match="≥2 columns"):
            basket_pca_residuals(df)

    def test_too_few_rows_raises(self) -> None:
        df = pd.DataFrame({"a": [0.5] * 10, "b": [0.5] * 10}, index=_idx(10))
        with pytest.raises(ValueError, match="aligned bars"):
            basket_pca_residuals(df, z_window=20)

    def test_kelly_fractions_returned(self) -> None:
        rng = np.random.default_rng(7)
        n = 100
        idx = _idx(n)
        df = pd.DataFrame(
            {
                "a": 0.5 + rng.normal(0, 0.05, n),
                "b": 0.5 + rng.normal(0, 0.05, n),
                "c": 0.5 + rng.normal(0, 0.05, n),
            },
            index=idx,
        )
        res = basket_pca_residuals(df, n_components=1)
        assert set(res.kelly_fraction_per_market.keys()) == {"a", "b", "c"}
        for v in res.kelly_fraction_per_market.values():
            assert -5.0 <= v <= 5.0


class TestKellyFraction:
    def test_zero_sharpe_zero_size(self) -> None:
        assert kelly_fraction(0.0) == 0.0

    def test_full_kelly_at_sharpe_1(self) -> None:
        assert kelly_fraction(1.0, fractional=1.0) == 0.5
        assert kelly_fraction(1.0, fractional=0.5) == 0.25

    def test_negative_sharpe_negative_size(self) -> None:
        assert kelly_fraction(-2.0, fractional=0.5) < 0
