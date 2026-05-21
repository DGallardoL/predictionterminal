"""Synthetic-DGP tests for the cross-sectional momentum strategy (W12-25).

Test discipline (CLAUDE.md):
    * Test the math with synthetic data where we control the DGP.
    * Never hit real Polymarket/yfinance.
    * For every "wow" claim there must be a quarter-stable robustness check
      — here we approximate that with multiple disjoint random seeds and a
      Sharpe-floor assertion.

The default ``CrossSectionalMomentum`` is registered with
``SHOULD_DEPLOY=False`` per the anti-alpha rule — these tests therefore
exercise the strategy class directly **and** confirm the registry-hook
gating behaviour. They never auto-flip ``SHOULD_DEPLOY``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pfm.strategies.cross_sectional_momentum import (
    LOOKBACK_DAYS,
    SHOULD_DEPLOY,
    STRATEGY_NAME,
    TIER,
    CrossSectionalMomentum,
    _decile_threshold,
    alpha_catalog_entry,
    register_if_ready,
    trailing_log_return,
)

# ---------------------------------------------------------------------------
# Synthetic-DGP fixtures
# ---------------------------------------------------------------------------


def _make_factor_prices(
    n_factors: int = 50,
    n_days: int = 120,
    *,
    momentum_top_k: int = 5,
    reversal_bottom_k: int = 5,
    momentum_drift: float = 0.01,
    reversal_drift: float = -0.01,
    base_vol: float = 0.02,
    seed: int = 0,
) -> pd.DataFrame:
    """Plant a momentum signal in the top-K factors and reversal in the bottom-K.

    Factors 0..(momentum_top_k-1) get a persistent positive drift; factors
    (n-reversal_bottom_k)..(n-1) get a persistent negative drift; the rest
    are iid Gaussian. Prices are log-Brownian: ``P_t = P_0 * exp(cum_sum(r))``.
    """
    rng = np.random.default_rng(seed)
    daily_rets = rng.normal(loc=0.0, scale=base_vol, size=(n_days, n_factors))
    # Plant the momentum / reversal drift in the appropriate columns.
    daily_rets[:, :momentum_top_k] += momentum_drift
    if reversal_bottom_k > 0:
        daily_rets[:, n_factors - reversal_bottom_k :] += reversal_drift
    cum = np.cumsum(daily_rets, axis=0)
    # Bound starting price at 0.5 (binary-market-like mid).
    prices = 0.5 * np.exp(cum)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D")
    cols = [f"factor_{i:02d}" for i in range(n_factors)]
    return pd.DataFrame(prices, index=idx, columns=cols)


def _make_noise_prices(
    n_factors: int = 50,
    n_days: int = 120,
    *,
    base_vol: float = 0.02,
    seed: int = 0,
) -> pd.DataFrame:
    """Pure-noise universe — no planted drift in any column."""
    return _make_factor_prices(
        n_factors=n_factors,
        n_days=n_days,
        momentum_top_k=0,
        reversal_bottom_k=0,
        base_vol=base_vol,
        seed=seed,
    )


def _build_panels(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (factor_returns_panel, realized_returns_panel).

    ``factor_returns_panel`` = trailing 14-day log returns (the signal input).
    ``realized_returns_panel`` = same-day 1-day log returns shifted by -1 so
        each row holds the *next-day* return (avoids look-ahead in compute_pnl).
    """
    panel = trailing_log_return(prices, window=LOOKBACK_DAYS)
    one_day = np.log(prices.astype(float)).diff(1)
    realized = one_day.shift(-1)  # next-day return aligned to today's signal
    return panel, realized


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_tier_is_b_validated(self) -> None:
        assert TIER == "B_VALIDATED"

    def test_should_deploy_default_false(self) -> None:
        assert SHOULD_DEPLOY is False

    def test_strategy_name(self) -> None:
        assert STRATEGY_NAME == "cross-sectional-momentum"

    def test_lookback_is_14(self) -> None:
        assert LOOKBACK_DAYS == 14


class TestDecileThreshold:
    @pytest.mark.parametrize(
        ("n", "frac", "expected"),
        [
            (50, 0.10, 5),
            (100, 0.10, 10),
            (10, 0.10, 1),
            (9, 0.10, 1),  # min 1 per leg
            (2, 0.10, 1),
            (1, 0.10, 0),  # cannot define a pair
            (0, 0.10, 0),
            (20, 0.20, 4),
        ],
    )
    def test_threshold_math(self, n: int, frac: float, expected: int) -> None:
        assert _decile_threshold(n, fraction=frac) == expected


class TestStrategyInitValidation:
    def test_invalid_decile_fraction(self) -> None:
        with pytest.raises(ValueError, match="decile_fraction"):
            CrossSectionalMomentum(decile_fraction=0.0)
        with pytest.raises(ValueError, match="decile_fraction"):
            CrossSectionalMomentum(decile_fraction=0.6)

    def test_invalid_kelly_cap(self) -> None:
        with pytest.raises(ValueError, match="kelly_cap"):
            CrossSectionalMomentum(kelly_cap=0.0)
        with pytest.raises(ValueError, match="kelly_cap"):
            CrossSectionalMomentum(kelly_cap=1.5)


class TestSignalShape:
    def test_signal_returns_multiindex_series(self) -> None:
        prices = _make_factor_prices(seed=1)
        panel, _ = _build_panels(prices)
        alpha = CrossSectionalMomentum()
        sig = alpha.signal(panel)
        assert isinstance(sig, pd.Series)
        assert isinstance(sig.index, pd.MultiIndex)
        assert sig.index.names == ["date", "factor"]
        # 50 factors × n_days dates.
        assert len(sig) == panel.shape[0] * panel.shape[1]

    def test_empty_input(self) -> None:
        alpha = CrossSectionalMomentum()
        sig = alpha.signal(pd.DataFrame())
        assert sig.empty

    def test_decile_boundaries_respected(self) -> None:
        """For a clean DGP, exactly 5 longs + 5 shorts per row (50 factors, 10%)."""
        prices = _make_factor_prices(seed=2)
        panel, _ = _build_panels(prices)
        alpha = CrossSectionalMomentum(decile_fraction=0.10)
        sig = alpha.signal(panel)
        # Inspect the last valid row (no NaN), which has 50 ranked factors.
        wide = sig.unstack("factor")
        last_row = wide.iloc[-1]
        long_mask = last_row > 0
        short_mask = last_row < 0
        assert int(long_mask.sum()) == 5
        assert int(short_mask.sum()) == 5
        # Net dollar-neutral (sums cancel to 0 up to float epsilon).
        assert math.isclose(last_row.sum(), 0.0, abs_tol=1e-12)
        # Gross leverage ≈ 2.0.
        assert math.isclose(last_row.abs().sum(), 2.0, abs_tol=1e-12)
        # Each long leg weight = 1/5, each short leg weight = -1/5.
        np.testing.assert_allclose(sorted(last_row[long_mask].values), [0.2, 0.2, 0.2, 0.2, 0.2])
        np.testing.assert_allclose(
            sorted(last_row[short_mask].values), [-0.2, -0.2, -0.2, -0.2, -0.2]
        )


class TestSignalRecovery:
    def test_planted_momentum_lands_in_top_decile(self) -> None:
        """Top-5 planted factors should appear in the long-leg majority of dates."""
        prices = _make_factor_prices(
            n_factors=50,
            momentum_top_k=5,
            reversal_bottom_k=5,
            momentum_drift=0.02,
            reversal_drift=-0.02,
            seed=42,
        )
        panel, _ = _build_panels(prices)
        alpha = CrossSectionalMomentum()
        sig = alpha.signal(panel)
        wide = sig.unstack("factor").dropna(how="all")
        planted_long = [f"factor_{i:02d}" for i in range(5)]
        planted_short = [f"factor_{i:02d}" for i in range(45, 50)]
        # Drop rows before the lookback fills (trailing returns are NaN).
        active = wide.dropna(how="all").iloc[LOOKBACK_DAYS:]
        long_hits = (active[planted_long] > 0).mean().mean()
        short_hits = (active[planted_short] < 0).mean().mean()
        # Both planted groups should be picked at least 80% of dates.
        assert long_hits >= 0.80
        assert short_hits >= 0.80

    def test_positive_pnl_on_planted_dgp(self) -> None:
        prices = _make_factor_prices(
            n_factors=50,
            momentum_top_k=5,
            reversal_bottom_k=5,
            momentum_drift=0.02,
            reversal_drift=-0.02,
            seed=7,
        )
        panel, realized = _build_panels(prices)
        alpha = CrossSectionalMomentum()
        pnl = alpha.compute_pnl(panel, realized)
        # On a planted DGP, mean PnL should be clearly positive.
        assert pnl.mean() > 0.0
        # And cumulative PnL should end above zero.
        assert pnl.sum() > 0.0


class TestNoSpuriousAlphaOnNoise:
    def test_noise_universe_has_near_zero_mean_pnl(self) -> None:
        # Average across many disjoint seeds to nail the null distribution.
        means: list[float] = []
        for seed in range(8):
            prices = _make_noise_prices(n_factors=50, n_days=120, seed=seed)
            panel, realized = _build_panels(prices)
            alpha = CrossSectionalMomentum()
            pnl = alpha.compute_pnl(panel, realized)
            means.append(float(pnl.mean()))
        grand_mean = float(np.mean(means))
        # No structural alpha → mean across many independent draws is ~0.
        # Use a generous tolerance because each draw still has variance.
        assert abs(grand_mean) < 1e-3


class TestSharpeOnPlantedDGP:
    def test_sharpe_at_least_zero_point_three(self) -> None:
        prices = _make_factor_prices(
            n_factors=50,
            n_days=180,
            momentum_top_k=5,
            reversal_bottom_k=5,
            momentum_drift=0.02,
            reversal_drift=-0.02,
            seed=11,
        )
        panel, realized = _build_panels(prices)
        alpha = CrossSectionalMomentum()
        pnl = alpha.compute_pnl(panel, realized).dropna()
        # Filter out the lookback warmup rows (signal=0 → PnL=0).
        active = pnl[pnl != 0.0]
        assert len(active) >= 50
        # Annualised Sharpe (daily PnL, sqrt(252) scaling).
        sharpe = (active.mean() / active.std(ddof=1)) * math.sqrt(252)
        assert sharpe >= 0.3, f"Sharpe={sharpe:.3f} below 0.3 floor"


class TestReproducibility:
    def test_same_seed_same_pnl(self) -> None:
        prices_a = _make_factor_prices(seed=99)
        prices_b = _make_factor_prices(seed=99)
        pd.testing.assert_frame_equal(prices_a, prices_b)
        panel_a, realized_a = _build_panels(prices_a)
        panel_b, realized_b = _build_panels(prices_b)
        alpha = CrossSectionalMomentum()
        pnl_a = alpha.compute_pnl(panel_a, realized_a)
        pnl_b = alpha.compute_pnl(panel_b, realized_b)
        pd.testing.assert_series_equal(pnl_a, pnl_b)

    def test_different_seeds_differ(self) -> None:
        prices_a = _make_factor_prices(seed=1)
        prices_b = _make_factor_prices(seed=2)
        panel_a, realized_a = _build_panels(prices_a)
        panel_b, realized_b = _build_panels(prices_b)
        alpha = CrossSectionalMomentum()
        pnl_a = alpha.compute_pnl(panel_a, realized_a)
        pnl_b = alpha.compute_pnl(panel_b, realized_b)
        # Almost surely different draws → different PnL series.
        assert not pnl_a.equals(pnl_b)


class TestPositionSizing:
    def test_kelly_cap_clips_individual_names(self) -> None:
        prices = _make_factor_prices(seed=3)
        panel, _ = _build_panels(prices)
        alpha = CrossSectionalMomentum(kelly_cap=0.10)
        sig = alpha.signal(panel)
        pos = alpha.position(sig)
        assert pos.abs().max() <= 0.10 + 1e-12

    def test_kelly_cap_argument_overrides(self) -> None:
        prices = _make_factor_prices(seed=4)
        panel, _ = _build_panels(prices)
        alpha = CrossSectionalMomentum(kelly_cap=0.10)
        sig = alpha.signal(panel)
        pos = alpha.position(sig, kelly_cap=0.05)
        assert pos.abs().max() <= 0.05 + 1e-12

    def test_invalid_kelly_cap_in_position(self) -> None:
        prices = _make_factor_prices(seed=5)
        panel, _ = _build_panels(prices)
        alpha = CrossSectionalMomentum()
        sig = alpha.signal(panel)
        with pytest.raises(ValueError, match="kelly_cap"):
            alpha.position(sig, kelly_cap=0.0)

    def test_position_requires_series(self) -> None:
        alpha = CrossSectionalMomentum()
        with pytest.raises(TypeError, match="signal"):
            alpha.position([0.1, -0.1])  # type: ignore[arg-type]


class TestPnLContract:
    def test_pnl_accepts_long_series(self) -> None:
        prices = _make_factor_prices(seed=8)
        panel, realized = _build_panels(prices)
        alpha = CrossSectionalMomentum()
        sig = alpha.signal(panel)
        pos = alpha.position(sig)
        long_realized = realized.stack(future_stack=True)
        long_realized.index.names = ["date", "factor"]
        pnl_wide = alpha.pnl(pos, realized)
        pnl_long = alpha.pnl(pos, long_realized)
        pd.testing.assert_series_equal(pnl_wide, pnl_long)

    def test_pnl_rejects_wrong_realized_type(self) -> None:
        prices = _make_factor_prices(seed=12)
        panel, _ = _build_panels(prices)
        alpha = CrossSectionalMomentum()
        sig = alpha.signal(panel)
        pos = alpha.position(sig)
        with pytest.raises(TypeError, match="realized"):
            alpha.pnl(pos, [0.1, -0.1])  # type: ignore[arg-type]

    def test_pnl_requires_multiindex(self) -> None:
        alpha = CrossSectionalMomentum()
        bad = pd.Series([0.1, -0.1], index=[0, 1])
        with pytest.raises(ValueError, match="MultiIndex"):
            alpha.pnl(bad, pd.DataFrame())


class TestRegistryGate:
    def test_register_if_ready_blocked_by_default(self) -> None:
        # SHOULD_DEPLOY=False at import time → returns None.
        result = register_if_ready()
        assert result is None

    def test_register_if_ready_force_registers(self) -> None:
        from pfm.strategies_registry import get, names, unregister

        try:
            strat = register_if_ready(force=True)
            assert strat is not None
            assert strat.name == STRATEGY_NAME
            assert STRATEGY_NAME in names()
            # Round-trip via the registry get().
            assert get(STRATEGY_NAME).name == STRATEGY_NAME
        finally:
            unregister(STRATEGY_NAME)

    def test_signal_adapter_handles_single_close_series(self) -> None:
        from pfm.strategies.cross_sectional_momentum import _signal_adapter

        idx = pd.date_range("2024-01-01", periods=40, freq="D")
        prices = pd.DataFrame({"close": np.linspace(1.0, 2.0, 40)}, index=idx)
        sig = _signal_adapter(prices)
        assert isinstance(sig, pd.Series)
        # Trend up → +1 once the lookback fills.
        assert sig.iloc[-1] == pytest.approx(1.0)


class TestCatalogHelper:
    def test_catalog_entry_shape(self) -> None:
        entry = alpha_catalog_entry()
        for key in (
            "pair_id",
            "tier",
            "label",
            "deploy_params",
            "theory_ref",
            "robustness",
            "should_deploy_at_publish_time",
            "anti_alpha_rule",
        ):
            assert key in entry, f"missing key {key}"
        assert entry["pair_id"] == STRATEGY_NAME
        assert entry["tier"] == "B_VALIDATED"
        assert entry["should_deploy_at_publish_time"] is False
        assert entry["deploy_params"]["lookback_days"] == 14

    def test_catalog_entry_carries_robustness(self) -> None:
        rob = {"q1_sharpe": 0.4, "q2_sharpe": 0.5}
        entry = alpha_catalog_entry(robustness=rob)
        assert entry["robustness"] == rob


class TestTrailingLogReturn:
    def test_trailing_log_return_matches_manual(self) -> None:
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = pd.DataFrame(
            {"a": np.linspace(1.0, 2.0, 30), "b": np.linspace(2.0, 1.0, 30)},
            index=idx,
        )
        out = trailing_log_return(prices, window=LOOKBACK_DAYS)
        # Manual check: row[14] vs row[0].
        np.testing.assert_allclose(
            out["a"].iloc[14],
            np.log(prices["a"].iloc[14] / prices["a"].iloc[0]),
        )
        np.testing.assert_allclose(
            out["b"].iloc[14],
            np.log(prices["b"].iloc[14] / prices["b"].iloc[0]),
        )

    def test_trailing_log_return_first_window_rows_nan(self) -> None:
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = pd.DataFrame({"a": np.linspace(1.0, 2.0, 30)}, index=idx)
        out = trailing_log_return(prices, window=LOOKBACK_DAYS)
        assert out["a"].iloc[:LOOKBACK_DAYS].isna().all()
        assert not out["a"].iloc[LOOKBACK_DAYS:].isna().any()
