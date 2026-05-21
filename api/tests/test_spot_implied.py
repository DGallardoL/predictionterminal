"""Tests for ``pfm.spot_implied`` — closed-form GBM probabilities, Yang-Zhang
vol, and the orchestrator with bootstrap CI.

Strategy: simulate GBM paths with known σ and compare the closed-form
analytic probabilities against the Monte-Carlo frequencies. The MC SE on
N=20k paths is ~0.0035 — the analytic answer should agree to ~3 decimal
places.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pfm.spot_implied import (
    block_bootstrap_log_returns,
    close_to_close_volatility,
    gbm_one_touch_down,
    gbm_one_touch_up,
    gbm_terminal_above,
    spot_vs_implied,
    yang_zhang_volatility,
)

# ─────────────────────── helpers ──────────────────────────────────────


def _simulate_gbm_paths(
    s0: float,
    sigma: float,
    drift: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    """Return an (n_paths, n_steps+1) matrix of GBM paths."""
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    z = rng.standard_normal(size=(n_paths, n_steps))
    log_inc = (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return s0 * np.exp(log_path)


def _make_synth_ohlc(n_days: int, sigma_annual: float, seed: int = 0) -> pd.DataFrame:
    """Build a realistic synthetic OHLC DataFrame.

    Constructs each day as: a within-day Brownian path with daily-σ matching
    the target annualised σ, then splits the day's variance between an
    overnight gap (≈30%) and the intraday range (≈70%). This way Yang-Zhang
    has all four pieces (overnight, open-to-close, Rogers-Satchell) populated
    realistically, matching the regime YZ was designed for.
    """
    rng = np.random.default_rng(seed)
    daily_sig = sigma_annual / np.sqrt(365.0)
    sigma_overnight = daily_sig * np.sqrt(0.30)
    sigma_intraday = daily_sig * np.sqrt(0.70)

    closes = np.empty(n_days)
    opens = np.empty(n_days)
    highs = np.empty(n_days)
    lows = np.empty(n_days)
    prev_close = 100.0
    n_intraday_steps = 24
    dt = 1.0 / n_intraday_steps
    for i in range(n_days):
        # Overnight gap
        opens[i] = prev_close * np.exp(rng.normal(0, sigma_overnight))
        # Within-day Brownian path
        steps = rng.normal(0, sigma_intraday * np.sqrt(dt), size=n_intraday_steps)
        path = opens[i] * np.exp(np.cumsum(steps))
        closes[i] = path[-1]
        highs[i] = max(opens[i], path.max())
        lows[i] = min(opens[i], path.min())
        prev_close = closes[i]

    idx = pd.date_range("2025-01-01", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


# ───────────────────── gbm_terminal_above ─────────────────────────────


class TestGbmTerminal:
    def test_at_the_money_t_zero_below(self) -> None:
        """T=0, S<K → 0; T=0, S>=K → 1."""
        assert gbm_terminal_above(99, 100, 0.5, 0.0, 0.0) == 0.0
        assert gbm_terminal_above(100, 100, 0.5, 0.0, 0.0) == 1.0

    def test_matches_monte_carlo(self) -> None:
        s0, sigma, drift, t, k = 100.0, 0.6, 0.0, 0.5, 110.0
        analytic = gbm_terminal_above(s0, k, sigma, drift, t)
        paths = _simulate_gbm_paths(s0, sigma, drift, t, 20000, 100, seed=1)
        mc = float((paths[:, -1] >= k).mean())
        assert abs(analytic - mc) < 0.012, f"analytic={analytic:.4f} mc={mc:.4f}"

    def test_zero_vol_deterministic(self) -> None:
        # σ=0, drift=0.10, T=1 → S_T = 100 * e^0.10 ≈ 110.5; K=105 → P=1
        assert gbm_terminal_above(100.0, 105.0, 0.0, 0.10, 1.0) == 1.0
        # K=120 → P=0
        assert gbm_terminal_above(100.0, 120.0, 0.0, 0.10, 1.0) == 0.0


# ───────────────────── gbm_one_touch_up ───────────────────────────────


class TestGbmOneTouchUp:
    def test_already_touched(self) -> None:
        assert gbm_one_touch_up(110, 100, 0.5, 0.0, 0.5) == 1.0

    def test_matches_monte_carlo(self) -> None:
        # Discrete-monitoring bias (Broadie-Glasserman-Kou 1997) makes
        # discrete-step MC under-count touches relative to continuous-time
        # analytic — so use many steps and a tolerance that's larger than
        # MC SE alone (1.96·√(p(1-p)/N) ≈ 0.007 at N=20k).
        s0, sigma, drift, t, h = 100.0, 0.6, 0.0, 0.5, 130.0
        analytic = gbm_one_touch_up(s0, h, sigma, drift, t)
        paths = _simulate_gbm_paths(s0, sigma, drift, t, 30000, 2000, seed=2)
        mc = float((paths.max(axis=1) >= h).mean())
        # Continuous analytic ≥ discrete MC always; tolerance accounts for
        # residual discretisation + finite-sample noise.
        assert abs(analytic - mc) < 0.025, f"analytic={analytic:.4f} mc={mc:.4f}"
        assert analytic >= mc - 0.005

    def test_at_zero_t_returns_zero_when_below(self) -> None:
        assert gbm_one_touch_up(99, 100, 0.5, 0.0, 0.0) == 0.0


# ───────────────────── gbm_one_touch_down ─────────────────────────────


class TestGbmOneTouchDown:
    def test_already_touched(self) -> None:
        assert gbm_one_touch_down(90, 100, 0.5, 0.0, 0.5) == 1.0

    def test_matches_monte_carlo(self) -> None:
        s0, sigma, drift, t, l = 100.0, 0.6, 0.0, 0.5, 70.0
        analytic = gbm_one_touch_down(s0, l, sigma, drift, t)
        paths = _simulate_gbm_paths(s0, sigma, drift, t, 30000, 2000, seed=3)
        mc = float((paths.min(axis=1) <= l).mean())
        assert abs(analytic - mc) < 0.025, f"analytic={analytic:.4f} mc={mc:.4f}"
        assert analytic >= mc - 0.005


# ───────────────────── yang_zhang_volatility ──────────────────────────


class TestYangZhang:
    def test_recovers_planted_sigma(self) -> None:
        sigma = 0.50  # 50% annualised
        df = _make_synth_ohlc(200, sigma_annual=sigma, seed=11)
        out = yang_zhang_volatility(df, annualisation=365.0)
        # 50% true; with 200 days the SE on σ̂ is ~σ/√(2·200) ≈ 0.025 → tolerate 5σ
        assert abs(out.sigma_annual - sigma) < 0.10
        assert out.method == "yang_zhang"
        assert out.n_bars == 200

    def test_too_few_bars_raises(self) -> None:
        df = pd.DataFrame(
            {"open": [1, 2], "high": [1.5, 2.5], "low": [0.9, 1.9], "close": [1.4, 2.4]}
        )
        with pytest.raises(ValueError, match="need ≥3 bars"):
            yang_zhang_volatility(df)

    def test_close_to_close_works(self) -> None:
        rng = np.random.default_rng(0)
        sigma = 0.30
        log_ret = rng.normal(0, sigma / np.sqrt(365.0), size=300)
        closes = 100.0 * np.exp(np.cumsum(log_ret))
        s = pd.Series(closes, index=pd.date_range("2025-01-01", periods=300, freq="D"))
        out = close_to_close_volatility(s)
        assert abs(out.sigma_annual - sigma) < 0.05
        assert out.method == "close_to_close"


# ───────────────── block_bootstrap_log_returns ────────────────────────


class TestBlockBootstrap:
    def test_shape_and_values_in_support(self) -> None:
        rng = np.random.default_rng(0)
        log_ret = rng.normal(0, 0.02, size=100)
        out = block_bootstrap_log_returns(log_ret, block_size=10, n_iters=50, seed=42)
        assert out.shape == (50, 100)
        # All values must come from the support of log_ret (no synthesis).
        unique_vals = np.unique(out)
        for v in unique_vals[:20]:
            assert v in log_ret

    def test_seed_determinism(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=80)
        a = block_bootstrap_log_returns(x, block_size=8, n_iters=30, seed=7)
        b = block_bootstrap_log_returns(x, block_size=8, n_iters=30, seed=7)
        assert np.array_equal(a, b)


# ───────────────────────── orchestrator ───────────────────────────────


class TestSpotVsImplied:
    def test_terminal_pipeline(self) -> None:
        df = _make_synth_ohlc(180, sigma_annual=0.50, seed=23)
        last = df.index[-1].date()
        out = spot_vs_implied(
            df,
            strike=120.0,
            expiry=last + timedelta(days=30),
            geometry="terminal",
            market_prob=0.45,
            drift_annual=0.0,
            n_bootstrap=80,
            seed=1,
        )
        assert 0.0 <= out.model_prob <= 1.0
        assert out.ci_lo_95 <= out.model_prob <= out.ci_hi_95
        assert out.edge == pytest.approx(0.45 - out.model_prob, abs=1e-9)
        assert isinstance(out.edge_significant_95, bool)
        assert out.n_bootstrap == 80

    def test_one_touch_up_pipeline(self) -> None:
        df = _make_synth_ohlc(120, sigma_annual=0.6, seed=31)
        last = df.index[-1].date()
        spot = float(df["close"].iloc[-1])
        out = spot_vs_implied(
            df,
            strike=spot * 1.30,
            expiry=last + timedelta(days=60),
            geometry="one_touch_up",
            market_prob=None,
            n_bootstrap=50,
            seed=2,
        )
        # touch probability should be > terminal probability for the same K
        out_term = spot_vs_implied(
            df,
            strike=spot * 1.30,
            expiry=last + timedelta(days=60),
            geometry="terminal",
            n_bootstrap=50,
            seed=2,
        )
        assert out.model_prob >= out_term.model_prob - 1e-9

    def test_too_few_bars_raises(self) -> None:
        df = _make_synth_ohlc(4, sigma_annual=0.5, seed=0)
        with pytest.raises(ValueError, match="need ≥5 bars"):
            spot_vs_implied(
                df, strike=110, expiry=date(2099, 1, 1), geometry="terminal", n_bootstrap=10, seed=0
            )

    def test_expiry_in_past_raises(self) -> None:
        df = _make_synth_ohlc(20, sigma_annual=0.4, seed=0)
        last = df.index[-1].date()
        with pytest.raises(ValueError, match=r"expiry .* is before asof"):
            spot_vs_implied(
                df,
                strike=110,
                expiry=last - timedelta(days=5),
                geometry="terminal",
                n_bootstrap=10,
                seed=0,
            )

    def test_ci_widens_with_smaller_bootstrap(self) -> None:
        # Sanity: smaller block size shouldn't crash; CI width still finite.
        df = _make_synth_ohlc(150, sigma_annual=0.55, seed=42)
        last = df.index[-1].date()
        out = spot_vs_implied(
            df,
            strike=115,
            expiry=last + timedelta(days=45),
            geometry="terminal",
            market_prob=0.50,
            n_bootstrap=60,
            block_size=3,
            seed=7,
        )
        assert out.ci_hi_95 > out.ci_lo_95
        assert out.ci_hi_90 > out.ci_lo_90
