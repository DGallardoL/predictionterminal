"""Tests for the relative-value strategy layer (edges + Monte-Carlo backtest)."""

from __future__ import annotations

import types

import numpy as np

from pfm.vol.pricing_kernel_strategies import (
    _event_prob,
    _sample_from_cdf,
    backtest_strategies,
    compute_edges,
    scan_opportunities,
)


def _entry(
    direction,
    prob,
    floor=None,
    cap=None,
    strike=None,
    slug="x",
    yes_bid=None,
    yes_ask=None,
    volume=None,
    open_interest=None,
):
    return types.SimpleNamespace(
        direction=direction,
        prob=prob,
        floor=floor,
        cap=cap,
        strike=strike,
        slug=slug,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        volume=volume,
        open_interest=open_interest,
    )


def _gauss_cdf(grid, mean, std):
    from scipy.stats import norm

    return norm.cdf(grid, loc=mean, scale=std)


def test_event_prob_between_below_above():
    grid = np.linspace(6000, 8000, 2000)
    cdf = _gauss_cdf(grid, 7000, 100)
    assert abs(_event_prob(grid, cdf, "between", 6900, 7100) - 0.682) < 0.02  # ±1σ
    assert abs(_event_prob(grid, cdf, "below", None, 7000) - 0.5) < 0.02
    assert abs(_event_prob(grid, cdf, "above", 7000, None) - 0.5) < 0.02


def test_compute_edges_flags_sell_and_buy():
    grid = np.linspace(6000, 8000, 2000)
    cdf = _gauss_cdf(grid, 7000, 100)  # both options and physical = same here
    entries = [
        _entry("between", 0.30, floor=7300, cap=7400, slug="rich_tail"),  # Kalshi rich vs ~0
        _entry("between", 0.05, floor=6950, cap=7050, slug="cheap_core"),  # Kalshi cheap vs ~0.38
    ]
    edges = compute_edges(entries, grid, cdf, grid, cdf, edge_threshold=0.03)
    by = {e.slug: e for e in edges}
    assert by["rich_tail"].action == "SELL"
    assert by["rich_tail"].edge_vs_options > 0
    assert by["cheap_core"].action == "BUY"
    assert by["cheap_core"].edge_vs_options < 0


def test_sample_from_cdf_matches_moments():
    grid = np.linspace(6000, 8000, 4000)
    cdf = _gauss_cdf(grid, 7000, 120)
    s = _sample_from_cdf(grid, cdf, 50000, np.random.default_rng(0))
    assert abs(s.mean() - 7000) < 10
    assert abs(s.std() - 120) < 8


def test_backtest_fade_rich_profitable_when_kalshi_overprices():
    # Physical truth: tight around 7000. Kalshi overprices the far buckets.
    grid = np.linspace(6000, 8000, 2000)
    phys_cdf = _gauss_cdf(grid, 7000, 100)
    opt_cdf = phys_cdf
    entries = [
        _entry("between", 0.20, floor=7300, cap=7400, slug="t1"),
        _entry("between", 0.20, floor=7400, cap=7500, slug="t2"),
        _entry("between", 0.18, floor=6500, cap=6600, slug="t3"),
    ]
    edges = compute_edges(entries, grid, opt_cdf, grid, phys_cdf, edge_threshold=0.03)
    res = {r.name: r for r in backtest_strategies(edges, grid, phys_cdf, cost=0.01, n_sims=20000)}
    fr = res["fade_rich_vs_options"]
    assert fr.n_legs == 3
    assert fr.mean_pnl > 0  # selling overpriced tails wins under physical truth
    assert fr.expected_edge_vs_physical > 0


def test_naive_buy_all_loses_the_spread():
    grid = np.linspace(6000, 8000, 2000)
    phys_cdf = _gauss_cdf(grid, 7000, 100)
    # a flat ladder summing to >1 (overround) → buying everything must lose
    entries = [
        _entry("between", 0.12, floor=f, cap=f + 100, slug=f"b{f}") for f in range(6600, 7400, 100)
    ]
    edges = compute_edges(entries, grid, phys_cdf, grid, phys_cdf, edge_threshold=0.03)
    res = {r.name: r for r in backtest_strategies(edges, grid, phys_cdf, cost=0.01, n_sims=10000)}
    assert res["naive_buy_all"].mean_pnl < 0


def test_empty_edges_returns_empty():
    grid = np.linspace(6000, 8000, 100)
    assert backtest_strategies([], grid, _gauss_cdf(grid, 7000, 100)) == []


# --- executable fair-value scanner ---


def test_scanner_finds_real_edge_with_tight_two_sided_quote():
    grid = np.linspace(6000, 8000, 2000)
    cdf = _gauss_cdf(grid, 7000, 100)  # fair P(between 6950,7050) ≈ 0.38
    # tight quote with the ask well below fair → BUY @ask is a real edge
    entries = [
        _entry(
            "between",
            0.20,
            floor=6950,
            cap=7050,
            slug="core",
            yes_bid=0.18,
            yes_ask=0.22,
            volume=500,
            open_interest=300,
        )
    ]
    opps, summ = scan_opportunities(
        entries, grid, cdf, grid, cdf, opt_strike_lo=6000, opt_strike_hi=8000, min_edge=0.02
    )
    assert summ["n_executable"] == 1
    assert len(opps) == 1
    assert opps[0].action == "BUY @ask"
    assert opps[0].edge > 0.10  # fair 0.38 − ask 0.22
    assert opps[0].confidence == "high"


def test_scanner_rejects_one_sided_untraded_quote():
    # The real Kalshi index-daily case: bid 0, wide ask, no volume → not executable.
    grid = np.linspace(6000, 8000, 2000)
    cdf = _gauss_cdf(grid, 7000, 100)
    entries = [
        _entry("between", 0.12, floor=7250, cap=7300, slug="dead1", yes_bid=0.0, yes_ask=0.24),
        _entry("between", 0.12, floor=7300, cap=7350, slug="dead2", yes_bid=0.0, yes_ask=0.25),
    ]
    opps, summ = scan_opportunities(
        entries, grid, cdf, grid, cdf, opt_strike_lo=6000, opt_strike_hi=8000
    )
    assert summ["n_executable"] == 0
    assert summ["tradeable"] is False
    assert opps == []


def test_scanner_flags_low_confidence_outside_option_strikes():
    grid = np.linspace(6000, 8000, 2000)
    cdf = _gauss_cdf(grid, 7000, 100)
    # bucket far above the fitted option strike range → extrapolated wing
    entries = [
        _entry("above", 0.10, strike=7600, slug="wing", yes_bid=0.05, yes_ask=0.09, volume=50)
    ]
    opps, _ = scan_opportunities(
        entries, grid, cdf, grid, cdf, opt_strike_lo=6800, opt_strike_hi=7200, min_edge=0.01
    )
    if opps:
        assert opps[0].confidence == "low"
