"""Tests for ``pfm.scanner``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pfm.factors import FactorConfig
from pfm.scanner import (
    discover_implication_pairs,
    run_scan,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


def _ar1(n: int, rho: float, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(max(1.0 - rho * rho, 1e-12))
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


# ───────────────────── discover_implication_pairs ─────────────────────


class TestDiscoverImplicationPairs:
    def test_oil_above_family(self) -> None:
        ids = ["oil_above_70_jun", "oil_above_115_jun", "oil_above_150_jun", "oil_above_175_jun"]
        pairs = discover_implication_pairs(ids)
        # Higher strike implies lower: e.g. (above_175, above_150).
        ant_con = set(pairs)
        assert ("oil_above_175_jun", "oil_above_150_jun") in ant_con
        assert ("oil_above_150_jun", "oil_above_115_jun") in ant_con
        assert ("oil_above_115_jun", "oil_above_70_jun") in ant_con

    def test_fed_cuts_family(self) -> None:
        ids = ["fed_cuts_2_2026", "fed_cuts_3_2026", "fed_cuts_5_2026"]
        pairs = discover_implication_pairs(ids)
        # More cuts is stricter (subset of "≥3 cuts" inside "≥2 cuts").
        assert ("fed_cuts_5_2026", "fed_cuts_3_2026") in pairs
        assert ("fed_cuts_5_2026", "fed_cuts_2_2026") in pairs
        assert ("fed_cuts_3_2026", "fed_cuts_2_2026") in pairs

    def test_singleton_family_no_pairs(self) -> None:
        ids = ["oil_above_115_jun"]  # only one — no pairs possible
        pairs = discover_implication_pairs(ids)
        assert pairs == []

    def test_unknown_pattern_ignored(self) -> None:
        ids = ["china_invade_taiwan_2026", "powell_out_may", "trump_wins_2024"]
        # None match any strike-family regex.
        assert discover_implication_pairs(ids) == []


# ─────────────────────────── run_scan ─────────────────────────────────


def _fc(fid: str, theme: str = "macro") -> FactorConfig:
    return FactorConfig(
        id=fid,
        name=fid,
        slug=fid,
        source="polymarket",
        description="t",
        theme=theme,
    )


class TestRunScan:
    def test_implication_scanner_finds_planted_violation(self) -> None:
        n = 200
        idx = _idx(n)
        # Plant a violation: oil_above_175 > oil_above_150 systematically.
        rng = np.random.default_rng(0)
        below = rng.uniform(0.20, 0.30, n)  # broader (lower strike)
        above = rng.uniform(0.50, 0.60, n)  # stricter — but priced HIGHER (mispricing)
        prices = {
            "oil_above_150_jun": pd.Series(below, index=idx),
            "oil_above_175_jun": pd.Series(above, index=idx),
        }
        factors = {fid: _fc(fid) for fid in prices}

        report = run_scan(
            factors,
            fetch_prices=lambda fc: prices[fc.id],
            mode="implication",
        )
        assert len(report.implication) == 1
        hit = report.implication[0]
        assert hit.kind == "implication"
        assert hit.a_id == "oil_above_175_jun"
        assert hit.b_id == "oil_above_150_jun"
        assert hit.n_violations >= 5
        assert hit.score > 0

    def test_conditional_scanner_finds_planted_dependency(self) -> None:
        n = 250
        idx = _idx(n)
        rng = np.random.default_rng(7)
        b = rng.uniform(0.10, 0.90, n)
        a = (0.05 + 0.7 * b + rng.normal(0, 0.02, n)).clip(0.01, 0.99)
        prices = {
            "alpha_event_2026": pd.Series(a, index=idx),
            "beta_event_2026": pd.Series(b, index=idx),
        }
        factors = {fid: _fc(fid) for fid in prices}
        report = run_scan(
            factors,
            fetch_prices=lambda fc: prices[fc.id],
            mode="conditional",
            cond_beta_min=0.20,  # lower threshold for the test
        )
        # Either direction (alpha~beta or beta~alpha) should fire because
        # the regression is symmetric in *strength* of dependence.
        assert len(report.conditional) >= 1
        hit = report.conditional[0]
        assert hit.kind == "conditional"
        assert abs(hit.beta) > 0.20
        assert hit.r_squared > 0.10

    def test_cointegration_scanner_finds_planted_pair(self) -> None:
        n = 300
        idx = _idx(n)
        rng = np.random.default_rng(99)
        # B is a random walk; A = 0.5 + 0.7 B + AR(1) noise → cointegrated.
        b = 0.4 + 0.10 * np.cumsum(rng.normal(0, 0.02, n))
        eps = _ar1(n, 0.4, 0.02, seed=100)
        a = 0.05 + 0.7 * b + eps
        prices = {
            "x_2026": pd.Series(a, index=idx),
            "y_2026": pd.Series(b, index=idx),
        }
        factors = {fid: _fc(fid) for fid in prices}
        report = run_scan(
            factors,
            fetch_prices=lambda fc: prices[fc.id],
            mode="cointegration",
        )
        assert len(report.cointegration) >= 1
        hit = report.cointegration[0]
        assert hit.kind == "cointegration"
        assert hit.adf_pvalue < 0.05
        assert hit.half_life_days is not None and hit.half_life_days < 30

    def test_theme_filter_restricts_factors(self) -> None:
        n = 100
        idx = _idx(n)
        rng = np.random.default_rng(0)
        prices = {
            "oil_above_115_jun": pd.Series(rng.uniform(0.3, 0.4, n), index=idx),
            "oil_above_150_jun": pd.Series(
                rng.uniform(0.6, 0.7, n), index=idx
            ),  # planted violation
            "trump_wins_2024": pd.Series(rng.uniform(0.4, 0.5, n), index=idx),
        }
        factors = {
            "oil_above_115_jun": _fc("oil_above_115_jun", theme="commodities"),
            "oil_above_150_jun": _fc("oil_above_150_jun", theme="commodities"),
            "trump_wins_2024": _fc("trump_wins_2024", theme="politics"),
        }
        # Theme filter to commodities — politics factor shouldn't appear.
        report = run_scan(
            factors,
            fetch_prices=lambda fc: prices[fc.id],
            mode="implication",
            theme="commodities",
        )
        assert report.n_factors_scanned == 2

    def test_runtime_recorded(self) -> None:
        prices = {
            "x": pd.Series(np.linspace(0.3, 0.4, 50), index=_idx(50)),
            "y": pd.Series(np.linspace(0.4, 0.5, 50), index=_idx(50)),
        }
        factors = {fid: _fc(fid) for fid in prices}
        report = run_scan(
            factors,
            fetch_prices=lambda fc: prices[fc.id],
            mode="conditional",
        )
        assert report.runtime_seconds >= 0.0

    def test_empty_factor_set(self) -> None:
        report = run_scan({}, fetch_prices=lambda fc: pd.Series(), mode="all")
        assert report.n_factors_scanned == 0
        assert report.implication == []
        assert report.conditional == []
        assert report.cointegration == []
