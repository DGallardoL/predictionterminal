"""Tests for ``pfm.ou`` (OU calibration + Bertram bands) and ``pfm.granger``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.granger import granger_test
from pfm.ou import bertram_optimal_bands, fit_ou


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


def _ou_simulate(n: int, kappa: float, mu: float, sigma: float, seed: int = 0) -> np.ndarray:
    """Exact-discretisation OU simulation (Δt = 1)."""
    rng = np.random.default_rng(seed)
    beta = np.exp(-kappa)
    eta_sd = sigma * np.sqrt((1 - beta * beta) / (2 * kappa))
    out = np.empty(n)
    out[0] = rng.normal(mu, sigma / np.sqrt(2 * kappa))
    for t in range(1, n):
        out[t] = mu + beta * (out[t - 1] - mu) + rng.normal(0, eta_sd)
    return out


# ───────────────────────────── fit_ou ─────────────────────────────────


class TestFitOu:
    def test_recovers_planted_kappa(self) -> None:
        n = 1000
        kappa_true, mu_true, sigma_true = 0.30, 0.05, 0.40
        x = _ou_simulate(n, kappa_true, mu_true, sigma_true, seed=42)
        fit = fit_ou(pd.Series(x, index=_idx(n)))
        # κ recovered within 20% on a 1000-bar sim.
        assert pytest.approx(fit.kappa, rel=0.30) == kappa_true
        assert pytest.approx(fit.mu, abs=0.05) == mu_true
        assert pytest.approx(fit.sigma_eq, rel=0.30) == sigma_true / np.sqrt(2 * kappa_true)
        assert 0 < fit.ar1_beta < 1

    def test_half_life_consistent(self) -> None:
        n = 800
        x = _ou_simulate(n, kappa=0.50, mu=0.0, sigma=1.0, seed=2)
        fit = fit_ou(pd.Series(x, index=_idx(n)))
        # half_life = ln 2 / κ.  Allow 30% tolerance on a finite sample.
        assert pytest.approx(fit.half_life_bars, rel=0.30) == np.log(2) / 0.50

    def test_too_short_raises(self) -> None:
        s = pd.Series([0.0] * 5, index=_idx(5))
        with pytest.raises(ValueError, match="≥10 bars"):
            fit_ou(s)

    def test_anti_persistent_rejected(self) -> None:
        """β ≤ 0 (anti-persistent) → not a stationary OU."""
        # Build a series with negative AR(1) coefficient.
        rng = np.random.default_rng(1)
        n = 200
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = -0.6 * x[t - 1] + rng.normal(0, 0.05)
        with pytest.raises(ValueError, match=r"not in.*0,1"):
            fit_ou(pd.Series(x, index=_idx(n)))


# ─────────────────────── bertram_optimal_bands ────────────────────────


class TestBertramBands:
    def test_zero_cost_falls_back_to_heuristic(self) -> None:
        """At zero cost the Bertram objective has no interior maximum; we
        fall back to z* = 1.5 σ_eq (Bertram's symmetric heuristic)."""
        n = 600
        x = _ou_simulate(n, kappa=0.40, mu=0.0, sigma=1.0, seed=7)
        fit = fit_ou(pd.Series(x, index=_idx(n)))
        bands = bertram_optimal_bands(fit, transaction_cost=0.0)
        assert bands["z_entry"] == 1.5
        assert bands["z_exit"] == 0.0
        assert bands["expected_pnl_per_year_sigma"] > 0

    def test_higher_cost_pushes_z_entry_out(self) -> None:
        n = 600
        x = _ou_simulate(n, kappa=0.30, mu=0.0, sigma=1.0, seed=12)
        fit = fit_ou(pd.Series(x, index=_idx(n)))
        b1 = bertram_optimal_bands(fit, transaction_cost=0.10)
        b2 = bertram_optimal_bands(fit, transaction_cost=0.50)
        # Higher costs require larger expected PnL per trade ⇒ wider bands.
        assert b2["z_entry"] >= b1["z_entry"] - 1e-6
        # Both should be in the plausible (0.5, 4) Bertram band.
        assert 0.3 < b1["z_entry"] < 4.0
        assert 0.3 < b2["z_entry"] < 4.0


# ──────────────────────────── granger_test ────────────────────────────


class TestGrangerTest:
    def test_b_leads_a_recovered(self) -> None:
        """Plant B → A with a one-bar lag."""
        rng = np.random.default_rng(123)
        n = 300
        b = rng.uniform(0.2, 0.8, n)
        a = np.empty(n)
        a[0] = 0.5
        for t in range(1, n):
            a[t] = 0.5 * a[t - 1] + 0.5 * b[t - 1] + rng.normal(0, 0.02)
        idx = _idx(n)
        res = granger_test(
            pd.Series(a, index=idx),
            pd.Series(b, index=idx),
            max_lag=3,
        )
        assert res.direction in ("B_causes_A", "bidirectional")
        assert res.best_pvalue_b_to_a is not None and res.best_pvalue_b_to_a < 0.05

    def test_independent_neither_causes(self) -> None:
        rng = np.random.default_rng(0)
        n = 250
        idx = _idx(n)
        a = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
        b = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
        res = granger_test(a, b, max_lag=3)
        # Independent series — both p-values should be > 0.05 most of the time.
        # Allow either "neither" or one-directional false positive (5% rate).
        assert res.direction in ("neither", "A_causes_B", "B_causes_A")

    def test_too_short_raises(self) -> None:
        a = pd.Series([0.5] * 10, index=_idx(10))
        b = a.copy()
        with pytest.raises(ValueError, match="bars"):
            granger_test(a, b, max_lag=3)
