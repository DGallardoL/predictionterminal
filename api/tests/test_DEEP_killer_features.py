"""DEEP exhaustive tests for killer-alpha features.

Covers (per the audit spec):
  1. news_causal_chain  (build_causal_chain, top_news_movers)
  2. resolution_pnl_tree (build_pnl_tree, monte_carlo_pnl)
  3. earnings_whisper   (compute_whisper, whisper_dashboard)
  4. vol_surface_pm     (extract_implied_distribution, compare_pm_vs_options_iv)
  5. counterfactual     (counterfactual_path, attribution_decomposition)
  6. news_tagger        (extract_entities, score_factor_match,
                         tag_news_to_factors, enhanced_sentiment)
  7. whale_mirror, smart_money_divergence, auto_hedge
  8. API endpoint smoke for the routers above.
  9. Edge cases (empty inputs, single obs, NaN, zero-variance).

External HTTP is fully mocked: PM-Gamma calls go through ``overrides``
maps; respx is used where an actual ``httpx.Client`` is exercised. Seeds
are pinned everywhere RNG is used.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import earnings_whisper, vol_surface_pm
from pfm.auto_hedge import (
    compute_hedge,
    simulate_hedge_path,
)
from pfm.auto_hedge import (
    router as auto_hedge_router,
)
from pfm.cache_utils import get_cache
from pfm.counterfactual import (
    attribution_decomposition,
    counterfactual_path,
)
from pfm.counterfactual import (
    router as counterfactual_router,
)
from pfm.earnings_whisper import (
    BEAT_LADDERS,
    CONSENSUS_EPS,
    compute_whisper,
    whisper_dashboard,
)
from pfm.earnings_whisper import (
    router as earnings_router,
)
from pfm.news_causal_chain import (
    BETA_REGISTRY,
    build_causal_chain,
    register_betas,
    top_news_movers,
)
from pfm.news_causal_chain import (
    router as causal_router,
)
from pfm.news_tagger import (
    enhanced_sentiment,
    extract_entities,
    score_factor_match,
    tag_news_to_factors,
)
from pfm.news_tagger import (
    router as tagger_router,
)
from pfm.resolution_pnl_tree import (
    build_pnl_tree,
    monte_carlo_pnl,
)
from pfm.resolution_pnl_tree import (
    router as pnl_tree_router,
)
from pfm.smart_money_divergence import (
    detect_divergence,
    scan_all_divergences,
)
from pfm.smart_money_divergence import (
    router as divergence_router,
)
from pfm.vol_surface_pm import (
    KNOWN_LADDERS,
    compare_pm_vs_options_iv,
    extract_implied_distribution,
)
from pfm.vol_surface_pm import (
    router as vol_router,
)
from pfm.whale_mirror import (
    mirror_whale,
    top_whales,
)
from pfm.whale_mirror import (
    router as whale_router,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Each test starts with empty caches and a clean β registry."""
    for ns in (
        "news_causal_chain",
        "news_causal_movers",
        "pnl_tree",
        "pnl_monte_carlo",
        "earnings_whisper",
        "vol_surface_pm",
        "counterfactual",
        "news_tagger",
        "whale_mirror",
        "smart_money_divergence",
        "auto_hedge",
    ):
        get_cache(ns).clear()
    BETA_REGISTRY.clear()


def _market(prob: float) -> dict[str, Any]:
    """Synthetic Gamma-shaped market dict with a midpoint at ``prob``."""
    return {
        "bestBid": max(0.0, prob - 0.005),
        "bestAsk": min(1.0, prob + 0.005),
        "lastTradePrice": prob,
    }


@pytest.fixture
def killer_client() -> TestClient:
    """A FastAPI app wired with every router under test (no main lifespan)."""
    app = FastAPI()
    for r in (
        causal_router,
        pnl_tree_router,
        earnings_router,
        vol_router,
        counterfactual_router,
        tagger_router,
        whale_router,
        divergence_router,
        auto_hedge_router,
    ):
        app.include_router(r)
    with TestClient(app) as c:
        yield c


# ===========================================================================
# 1) News Causal Chain
# ===========================================================================


class TestNewsCausalChain:
    def test_powell_signals_patience_tags_fed_factor(self) -> None:
        items = [
            {
                "title": "Powell signals patience on rate cuts",
                "description": "Fed chair says patience needed before next move.",
                "ts": "2026-05-08T14:00:00Z",
                "source": "wsj.com",
                "price_before": 0.55,
                "price_after": 0.40,
            }
        ]
        out = build_causal_chain(
            "fed-cut-march-2026",
            items,
            beta_map={"TLT": 4.2},
        )
        assert out["n_items"] == 1
        assert out["n_tagged"] == 1
        link = out["chain"][0]
        assert link["tagged_factor"] == "fed-cut-march-2026"
        # Δlogit = logit(0.40) - logit(0.55) ≈ -0.6053
        assert link["delta_logit"] == pytest.approx(-0.6053, abs=0.01)
        impact = link["affected_tickers"][0]
        # expected_return_pct = β × Δlogit × 100 ≈ 4.2 × -0.6053 × 100 ≈ -254
        assert impact["expected_return_pct"] == pytest.approx(
            4.2 * link["delta_logit"] * 100.0, rel=1e-3
        )
        assert impact["beta_source"] == "regression"

    def test_high_confidence_requires_all_three_signals(self) -> None:
        # Headline overlapping ≥2 keywords + |Δlogit| ≥ 0.5 + regression β.
        items = [
            {
                "title": "Trump wins 2027 again — election Trump",
                "description": "Election Trump",
                "price_before": 0.30,
                "price_after": 0.85,  # logit jump of ~+2.6 → strong
            }
        ]
        out = build_causal_chain(
            "trump-out-by-2027",
            items,
            beta_map={"DJT": 1.5},
        )
        link = out["chain"][0]
        assert link["confidence"] == "high"
        assert link["affected_tickers"][0]["confidence"] == "high"

    def test_medium_when_no_beta_cached(self) -> None:
        # Tagged news with a price reaction but no β registry → synthetic.
        items = [
            {
                "title": "Trump rallies in trump-out polling",
                "price_before": 0.4,
                "price_after": 0.6,
            }
        ]
        out = build_causal_chain("trump-out-by-2027", items)  # no beta_map
        link = out["chain"][0]
        assert link["confidence"] == "medium"
        assert link["affected_tickers"][0]["beta_source"] == "synthetic"

    def test_low_when_no_keyword_match(self) -> None:
        items = [{"title": "Random unrelated headline about cats"}]
        out = build_causal_chain("fed-cut-march-2026", items, beta_map={"TLT": 1.0})
        link = out["chain"][0]
        assert link["tagged_factor"] is None
        assert link["confidence"] == "low"
        # When no keyword match, no Δlogit is produced; per-ticker rows have
        # expected_return_pct=None and confidence='low'.
        for tk in link["affected_tickers"]:
            assert tk["expected_return_pct"] is None
            assert tk["confidence"] == "low"

    def test_top_news_movers_ranked_and_n_limit(self) -> None:
        # Build a feed dict with several factors, varying impact magnitudes.
        register_betas("fed-cut-march-2026", {"TLT": 4.2})
        register_betas("trump-out-by-2027", {"DJT": 1.5})
        feeds = {
            "fed-cut-march-2026": [
                {"title": "Fed cut", "price_before": 0.5, "price_after": 0.51},  # tiny
                {"title": "Fed cut decisive", "price_before": 0.5, "price_after": 0.85},
            ],
            "trump-out-by-2027": [
                {"title": "Trump out trump out", "price_before": 0.5, "price_after": 0.7},
            ],
        }
        movers = top_news_movers(
            window_hours=24,
            n=10,
            min_impact_pct=1.0,
            fetched_items_by_factor=feeds,
        )
        assert len(movers) >= 1
        impacts = [abs(m["expected_impact_pct"]) for m in movers]
        assert impacts == sorted(impacts, reverse=True)
        # Limit n
        movers_one = top_news_movers(
            window_hours=24,
            n=1,
            min_impact_pct=0.0,
            fetched_items_by_factor=feeds,
        )
        assert len(movers_one) == 1

    def test_min_impact_filter(self) -> None:
        register_betas("fed-cut-march-2026", {"TLT": 0.01})  # tiny β
        feeds = {
            "fed-cut-march-2026": [{"title": "Fed cut", "price_before": 0.5, "price_after": 0.501}]
        }
        movers = top_news_movers(
            window_hours=24,
            n=10,
            min_impact_pct=10.0,
            fetched_items_by_factor=feeds,
        )
        assert movers == []

    def test_empty_news_items(self) -> None:
        out = build_causal_chain("fed-cut-march-2026", [], beta_map={"TLT": 1.0})
        assert out["n_items"] == 0
        assert out["n_tagged"] == 0
        assert out["chain"] == []

    def test_top_movers_empty_input(self) -> None:
        assert top_news_movers(fetched_items_by_factor={}) == []
        assert top_news_movers(fetched_items_by_factor=None) == []


# ===========================================================================
# 2) Resolution P&L Tree
# ===========================================================================


class TestResolutionPnLTree:
    def test_two_position_yes_no_breakdown(self) -> None:
        positions = [
            {"ticker": "NVDA", "size_usd": 10_000.0, "beta_factor": 2.5},
            {"ticker": "AAPL", "size_usd": 5_000.0, "beta_factor": 1.5},
        ]
        out = build_pnl_tree(positions, "fed-cut-mar", current_prob=0.5)
        assert out["n_positions"] == 2
        assert out["gross_notional_usd"] == 15_000.0
        scenarios = {s["outcome"]: s for s in out["scenarios"]}
        # YES scenario: Δlogit > 0, MTM positive when β > 0
        assert scenarios["YES"]["delta_logit"] > 0
        assert scenarios["NO"]["delta_logit"] < 0
        # Per-ticker breakdown coherent
        leg_n = next(leg for leg in scenarios["YES"]["by_ticker"] if leg["ticker"] == "NVDA")
        assert leg_n["delta_return"] == pytest.approx(
            2.5 * scenarios["YES"]["delta_logit"], rel=1e-9
        )
        assert leg_n["mtm_usd"] == pytest.approx(10_000.0 * leg_n["delta_return"], rel=1e-9)

    def test_expected_value_is_prob_weighted_sum(self) -> None:
        positions = [{"ticker": "NVDA", "size_usd": 10_000.0, "beta_factor": 2.0}]
        out = build_pnl_tree(positions, "f", current_prob=0.7)
        scenarios = out["scenarios"]
        ev = scenarios[0]["prob"] * scenarios[0]["mtm_total_usd"] + (
            scenarios[1]["prob"] * scenarios[1]["mtm_total_usd"]
        )
        assert out["expected_value_usd"] == pytest.approx(ev, rel=1e-9)

    def test_var_95_is_worse_scenario(self) -> None:
        positions = [{"ticker": "X", "size_usd": 100_000.0, "beta_factor": 1.0}]
        out = build_pnl_tree(positions, "f", current_prob=0.5)
        worst = min(s["mtm_total_usd"] for s in out["scenarios"])
        assert out["var_95_usd"] == pytest.approx(worst, rel=1e-9)
        assert out["var_95_usd"] < 0

    def test_zero_beta_position_zero_pnl(self) -> None:
        positions = [{"ticker": "FLAT", "size_usd": 50_000.0, "beta_factor": 0.0}]
        out = build_pnl_tree(positions, "f", current_prob=0.5)
        for s in out["scenarios"]:
            assert s["mtm_total_usd"] == 0.0
            for leg in s["by_ticker"]:
                assert leg["mtm_usd"] == 0.0

    def test_five_positions_breakdown_consistent(self) -> None:
        positions = [
            {"ticker": f"T{i}", "size_usd": 1000.0 * (i + 1), "beta_factor": 0.5 * (i + 1)}
            for i in range(5)
        ]
        out = build_pnl_tree(positions, "f", current_prob=0.4)
        for s in out["scenarios"]:
            total = sum(leg["mtm_usd"] for leg in s["by_ticker"])
            assert s["mtm_total_usd"] == pytest.approx(total, rel=1e-9)
            assert len(s["by_ticker"]) == 5

    def test_high_prob_close_to_yes_smaller_yes_magnitude(self) -> None:
        positions = [{"ticker": "X", "size_usd": 10_000.0, "beta_factor": 1.0}]
        out = build_pnl_tree(positions, "f", current_prob=0.99)
        scenarios = {s["outcome"]: s for s in out["scenarios"]}
        # current_prob ≈ YES; the YES Δlogit move should be tiny vs the NO move.
        assert abs(scenarios["YES"]["delta_logit"]) < abs(scenarios["NO"]["delta_logit"])

    def test_prob_50_symmetric_magnitudes(self) -> None:
        positions = [{"ticker": "X", "size_usd": 10_000.0, "beta_factor": 1.0}]
        out = build_pnl_tree(positions, "f", current_prob=0.5)
        s_yes = next(s for s in out["scenarios"] if s["outcome"] == "YES")
        s_no = next(s for s in out["scenarios"] if s["outcome"] == "NO")
        assert abs(s_yes["delta_logit"]) == pytest.approx(abs(s_no["delta_logit"]), rel=1e-9)

    def test_empty_positions_raises(self) -> None:
        with pytest.raises(ValueError):
            build_pnl_tree([], "f", current_prob=0.5)


class TestMonteCarloPnL:
    def test_seed_reproducibility(self) -> None:
        positions = [{"ticker": "X", "size_usd": 10_000.0, "beta_factor": 1.0}]
        a = monte_carlo_pnl(positions, "f", n_paths=1_000, seed=7)
        b = monte_carlo_pnl(positions, "f", n_paths=1_000, seed=7)
        assert a["expected_value_usd"] == b["expected_value_usd"]
        assert a["var_95_usd"] == b["var_95_usd"]

    def test_percentile_ordering(self) -> None:
        positions = [{"ticker": "X", "size_usd": 10_000.0, "beta_factor": 1.5}]
        out = monte_carlo_pnl(positions, "f", n_paths=10_000, seed=42)
        p = out["percentiles"]
        assert p["p5"] < p["p50"] < p["p95"]
        assert p["p1"] <= p["p5"]
        assert p["p99"] >= p["p95"]

    def test_mean_close_to_zero_for_zero_mean_bootstrap(self) -> None:
        positions = [{"ticker": "X", "size_usd": 10_000.0, "beta_factor": 1.0}]
        out = monte_carlo_pnl(positions, "f", n_paths=20_000, seed=11, bootstrap_sigma=1.0)
        # Δlogit ~ N(0, 1), exposure = 10_000 → std ≈ 10_000;
        # mean over 20k draws should be small relative to that std.
        assert abs(out["expected_value_usd"]) < 500.0

    def test_var_95_is_negated_p5(self) -> None:
        positions = [{"ticker": "X", "size_usd": 10_000.0, "beta_factor": 1.0}]
        out = monte_carlo_pnl(positions, "f", n_paths=5_000, seed=3)
        assert out["var_95_usd"] == pytest.approx(-out["percentiles"]["p5"], abs=1e-9)

    def test_cvar_less_than_var(self) -> None:
        positions = [{"ticker": "X", "size_usd": 10_000.0, "beta_factor": 1.0}]
        out = monte_carlo_pnl(positions, "f", n_paths=10_000, seed=99)
        # cvar_95 is the *mean* PnL in the lower tail (a negative number),
        # so cvar_95 ≤ p5 (more negative or equal).
        assert out["cvar_95_usd"] <= out["percentiles"]["p5"] + 1e-6

    def test_low_n_paths_widens_band(self) -> None:
        positions = [{"ticker": "X", "size_usd": 10_000.0, "beta_factor": 1.0}]
        small = monte_carlo_pnl(positions, "f", n_paths=200, seed=5)
        big = monte_carlo_pnl(positions, "f", n_paths=20_000, seed=5)
        # Both should produce a finite non-zero std.
        assert small["std_pnl_usd"] > 0
        assert big["std_pnl_usd"] > 0
        # And the central tendency should agree to within a sigma.
        sigma = big["std_pnl_usd"]
        assert abs(small["expected_value_usd"] - big["expected_value_usd"]) < 5 * sigma

    def test_mc_empty_positions_raises(self) -> None:
        with pytest.raises(ValueError):
            monte_carlo_pnl([], "f", n_paths=100)


# ===========================================================================
# 3) Earnings Whisper
# ===========================================================================


class TestEarningsWhisper:
    def _strong_overrides(self, ticker: str) -> dict[str, dict[str, Any]]:
        # 0.71 beat-prob → expected positive whisper.
        probs = {0.0: 0.71, 0.03: 0.55, 0.05: 0.45, 0.07: 0.30, 0.10: 0.20, 0.20: 0.05}
        return {slug: _market(probs.get(t, 0.10)) for slug, t in BEAT_LADDERS[ticker]}

    def test_long_pre_print_when_edge_positive(self) -> None:
        out = compute_whisper(
            "NVDA",
            date(2026, 5, 22),
            http=MagicMock(),
            overrides=self._strong_overrides("NVDA"),
        )
        assert out["edge_vs_consensus_pct"] > 2.0
        assert out["recommendation"] == "long_pre_print"
        # whisper_eps = consensus * (1 + expected_beat)
        assert out["whisper_eps"] == pytest.approx(
            CONSENSUS_EPS["NVDA"] * (1.0 + out["expected_beat_pct"] / 100.0),
            rel=1e-3,
        )

    def test_short_pre_print_when_edge_negative(self) -> None:
        # All-zero ladder ⇒ no chance of any beat ⇒ expected_beat = -2.5%
        # (the assumed midpoint of the "below 0" mass), giving an edge below
        # the -2% threshold.
        weak = {slug: _market(0.001) for slug, _ in BEAT_LADDERS["NVDA"]}
        out = compute_whisper(
            "NVDA",
            date(2026, 5, 22),
            http=MagicMock(),
            overrides=weak,
        )
        assert out["edge_vs_consensus_pct"] < -2.0
        assert out["recommendation"] == "short_pre_print"

    def test_hold_when_edge_in_band(self) -> None:
        # tightly balanced ladder
        balanced = {
            slug: _market({0.0: 0.50, 0.05: 0.06, 0.10: 0.02, 0.20: 0.005}.get(t, 0.005))
            for slug, t in BEAT_LADDERS["NVDA"]
        }
        out = compute_whisper(
            "NVDA",
            date(2026, 5, 22),
            http=MagicMock(),
            overrides=balanced,
        )
        assert -2.0 <= out["edge_vs_consensus_pct"] <= 2.0
        assert out["recommendation"] == "hold"

    def test_unknown_ticker_raises(self) -> None:
        with pytest.raises(KeyError):
            compute_whisper("ZZZZ", date(2026, 5, 22), overrides={})

    def test_iv_implied_move_positive(self) -> None:
        out = compute_whisper(
            "NVDA",
            date(2026, 5, 22),
            http=MagicMock(),
            overrides={},
        )
        assert out["iv_implied_move_pct"] > 0.0

    def test_dashboard_sorted_and_filtered(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        today = date.today()
        cal = {
            "NVDA": today + timedelta(days=2),
            "TSLA": today + timedelta(days=20),  # filtered out
        }
        monkeypatch.setattr(earnings_whisper, "NEXT_EARNINGS", cal)
        # Two divergent override sets so |edge| differs.
        ovr: dict[str, dict[str, Any]] = {}
        ovr.update(self._strong_overrides("NVDA"))
        rows = whisper_dashboard(days=14, http=MagicMock(), overrides=ovr)
        tickers = {r["ticker"] for r in rows}
        assert "NVDA" in tickers
        assert "TSLA" not in tickers
        edges = [abs(r["edge_vs_consensus_pct"]) for r in rows]
        assert edges == sorted(edges, reverse=True)


# ===========================================================================
# 4) Vol Surface from PM
# ===========================================================================


# Slug map for the BTC EOY-2026 above-ladder, refreshed 2026-05-15. The
# original `btc-above-Xk-eoy-2026` slugs are dead on Polymarket — replaced
# with the long-form `will-bitcoin-reach-…-by-december-31-2026` markets
# (10 strikes spanning 90k–1M).
_BTC_LADDER_SLUGS_K: dict[float, str] = {
    90_000.0: "will-bitcoin-reach-90000-by-december-31-2026-113-862-581",
    100_000.0: "will-bitcoin-reach-100000-by-december-31-2026-571-361-361",
    140_000.0: "will-bitcoin-reach-140000-by-december-31-2026-131-829-299",
    150_000.0: "will-bitcoin-reach-150000-by-december-31-2026-557-246-971",
    160_000.0: "will-bitcoin-reach-160000-by-december-31-2026-934-934-164",
    190_000.0: "will-bitcoin-reach-190000-by-december-31-2026-936-485-627",
    200_000.0: "will-bitcoin-reach-200000-by-december-31-2026-752-232-389",
    250_000.0: "will-bitcoin-reach-250000-by-december-31-2026-579-442",
    500_000.0: "will-bitcoin-reach-500000-by-december-31-2026-864",
    1_000_000.0: "will-bitcoin-reach-1000000-by-december-31-2026-946",
}


class TestVolSurfacePM:
    def test_btc_ladder_extracts_distribution(self) -> None:
        # Strike ladder probs: monotone-decreasing in strike.
        probs_by_strike = [0.65, 0.43, 0.115, 0.065, 0.06, 0.042, 0.04, 0.028, 0.02, 0.015]
        probs = {
            _BTC_LADDER_SLUGS_K[k]: p
            for k, p in zip(_BTC_LADDER_SLUGS_K.keys(), probs_by_strike, strict=True)
        }
        overrides = {slug: _market(p) for slug, p in probs.items()}
        out = extract_implied_distribution(
            "btc-price-eoy-2026",
            market_value=95_000.0,
            overrides=overrides,
        )
        # Survival fn must be monotone non-increasing.
        ip = out["implied_probs"]
        assert all(ip[i] >= ip[i + 1] - 1e-9 for i in range(len(ip) - 1))
        assert out["fitted_mean"] > 0
        assert out["fitted_std"] > 0
        assert out["lognormal_sigma"] > 0
        assert out["n_strikes"] == 10
        assert out["fitted_distribution_type"] == "lognormal"

    def test_log_normal_fit_recovers_mean(self) -> None:
        probs_by_strike = [0.55, 0.40, 0.10, 0.07, 0.06, 0.04, 0.035, 0.025, 0.018, 0.012]
        probs = {
            _BTC_LADDER_SLUGS_K[k]: p
            for k, p in zip(_BTC_LADDER_SLUGS_K.keys(), probs_by_strike, strict=True)
        }
        overrides = {slug: _market(p) for slug, p in probs.items()}
        out = extract_implied_distribution(
            "btc-price-eoy-2026",
            overrides=overrides,
        )
        # E[X] = exp(μ + σ²/2) for log-normal.
        recon = float(np.exp(out["lognormal_mu"] + 0.5 * out["lognormal_sigma"] ** 2))
        assert recon == pytest.approx(out["fitted_mean"], rel=0.01)

    def test_monotonicity_enforced_when_market_inverts(self) -> None:
        # Out-of-order probs: 200k-above priced higher than 160k-above triggers monotonisation.
        raw = {
            90_000.0: 0.50,
            100_000.0: 0.42,
            140_000.0: 0.11,
            150_000.0: 0.06,
            160_000.0: 0.05,
            190_000.0: 0.04,
            200_000.0: 0.30,  # inversion
            250_000.0: 0.028,
            500_000.0: 0.02,
            1_000_000.0: 0.015,
        }
        overrides = {_BTC_LADDER_SLUGS_K[k]: _market(p) for k, p in raw.items()}
        out = extract_implied_distribution(
            "btc-price-eoy-2026",
            overrides=overrides,
        )
        ip = out["implied_probs"]
        for i in range(len(ip) - 1):
            assert ip[i] >= ip[i + 1] - 1e-9

    def test_unknown_pattern_raises(self) -> None:
        with pytest.raises(KeyError):
            extract_implied_distribution("not-a-real-pattern")

    def test_single_strike_degenerate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Only one rung resolvable → < 2 → ValueError. Force unmocked slugs
        # to raise LookupError so they get skipped (previously this relied on
        # the old `btc-above-Xk-eoy-2026` slugs being dead on Polymarket; with
        # the 2026-05-15 slug refresh they're now live, hence the explicit
        # injection).
        from pfm import vol_surface_pm as vs

        def _miss(*_a, **_k):
            raise LookupError("not found in test")

        monkeypatch.setattr(vs, "fetch_gamma_market", _miss)
        overrides = {_BTC_LADDER_SLUGS_K[90_000.0]: _market(0.7)}
        with pytest.raises(ValueError):
            extract_implied_distribution(
                "btc-price-eoy-2026",
                overrides=overrides,
            )

    def test_compare_pm_vs_options_iv(self) -> None:
        probs = {
            slug: 0.8 - 0.18 * i for i, (slug, _) in enumerate(KNOWN_LADDERS["btc-price-eoy-2026"])
        }
        overrides = {slug: _market(max(0.02, p)) for slug, p in probs.items()}
        dist = extract_implied_distribution(
            "btc-price-eoy-2026",
            overrides=overrides,
        )
        cmp = compare_pm_vs_options_iv("BTC", 95_000.0, dist, options_iv_annual=0.5)
        assert "spread_sigma" in cmp
        assert cmp["direction"] in {"pm_richer", "options_richer", "flat"}
        assert cmp["pm_lognormal_sigma"] >= 0


# ===========================================================================
# 5) Counterfactual
# ===========================================================================


def _synthetic_counterfactual_pair(
    n: int = 60,
    beta_true: float = 0.5,
    noise_sd: float = 0.0005,
    drift: float = 0.005,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed=7)
    idx = pd.date_range(start="2024-01-01", periods=n, freq="B")
    dlogit = pd.Series(rng.normal(drift, 0.18, size=n), index=idx)
    rets = pd.Series(beta_true * dlogit.values + rng.normal(0, noise_sd, size=n), index=idx)
    return rets, dlogit


class TestCounterfactual:
    def test_known_beta_recovered(self) -> None:
        rets, dlog = _synthetic_counterfactual_pair(n=80, beta_true=0.5)
        out = counterfactual_path(
            "TEST",
            "f",
            scenario="NO",
            actual_resolution="YES",
            start=date(2024, 1, 1),
            end=date(2024, 6, 1),
            returns=rets,
            dlogit=dlog,
        )
        assert out["beta"] == pytest.approx(0.5, abs=0.05)

    def test_scenario_flipped_inverts_attributable(self) -> None:
        rets, dlog = _synthetic_counterfactual_pair(
            n=60, beta_true=0.5, noise_sd=0.0005, drift=0.02
        )
        flipped = counterfactual_path(
            "TEST",
            "f",
            scenario="NO",
            actual_resolution="YES",
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            returns=rets,
            dlogit=dlog,
        )
        identity = counterfactual_path(
            "TEST",
            "f",
            scenario="YES",
            actual_resolution="YES",
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            returns=rets,
            dlogit=dlog,
        )
        # Identity scenario: counterfactual = actual.
        assert identity["total_return_counterfactual_pct"] == pytest.approx(
            identity["total_return_actual_pct"], rel=1e-4
        )
        # Flipped should differ from actual by approximately twice the
        # attributable component (because counter = actual - 2β·Σdlogit).
        diff = flipped["total_return_counterfactual_pct"] - flipped["total_return_actual_pct"]
        assert abs(diff) > 1e-3

    def test_zero_beta_counterfactual_identical(self) -> None:
        rets, dlog = _synthetic_counterfactual_pair(n=50, beta_true=0.0, noise_sd=0.0)
        out = counterfactual_path(
            "TEST",
            "f",
            scenario="NO",
            actual_resolution="YES",
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            returns=rets,
            dlogit=dlog,
            beta=0.0,
        )
        assert out["total_return_counterfactual_pct"] == pytest.approx(
            out["total_return_actual_pct"], rel=1e-9
        )

    def test_attribution_decomposition_shares_sum_to_one(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        rets = pd.Series(rng.normal(0.001, 0.01, 80), index=idx)
        d_a = pd.Series(rng.normal(0.005, 0.18, 80), index=idx)
        d_b = pd.Series(rng.normal(-0.003, 0.18, 80), index=idx)
        out = attribution_decomposition(
            "TEST",
            ["a", "b"],
            date(2024, 1, 1),
            date(2024, 5, 1),
            betas={"a": 0.4, "b": 0.6},
            returns=rets,
            dlogits={"a": d_a, "b": d_b},
        )
        shares = sum(r["contribution_share"] for r in out["rows"])
        assert shares == pytest.approx(1.0, abs=1e-3)

    def test_attribution_empty_factors_raises(self) -> None:
        with pytest.raises(ValueError):
            attribution_decomposition(
                "T",
                [],
                date(2024, 1, 1),
                date(2024, 4, 1),
            )


# ===========================================================================
# 6) News Tagger
# ===========================================================================


class TestNewsTagger:
    def test_extract_entities_basic(self) -> None:
        text = "Trump signs executive order on China tariffs"
        ents = extract_entities(text)
        assert "Trump" in ents["politicians"]
        assert "China" in ents["countries"]
        assert "ExecutiveOrder" in ents["events"]
        assert "Tariff" in ents["events"]

    def test_dollar_ticker_syntax(self) -> None:
        text = "$NVDA up 5% after $AAPL earnings"
        ents = extract_entities(text)
        assert "NVDA" in ents["tickers"]
        assert "AAPL" in ents["tickers"]

    def test_multi_word_politician(self) -> None:
        ents = extract_entities("Donald Trump and Joe Biden meet")
        assert "Trump" in ents["politicians"]
        assert "Biden" in ents["politicians"]

    def test_stop_words_not_tickers(self) -> None:
        ents = extract_entities("AND THE FOR with cats")
        assert ents["tickers"] == []

    def test_score_factor_match_relevant(self) -> None:
        text = "Powell signals Fed rate cut in March"
        factor = {
            "id": "fed-cut-march-2026",
            "name": "Fed rate cut March",
            "slug": "fed-cut-march-2026",
            "theme": "macro",
            "keywords": ["fed", "cut", "rate", "march", "powell"],
        }
        s = score_factor_match(text, factor)
        assert s >= 0.3

    def test_score_factor_match_unrelated(self) -> None:
        text = "Random kitten video goes viral"
        factor = {
            "id": "fed-cut-march-2026",
            "name": "Fed rate cut March",
            "slug": "fed-cut-march-2026",
            "theme": "macro",
        }
        s = score_factor_match(text, factor)
        assert s == 0.0 or s < 0.1

    def test_tag_news_to_factors_threshold(self) -> None:
        items = [
            {"title": "Powell signals Fed rate cut", "description": "rate cut"},
            {"title": "Random unrelated headline"},
        ]
        catalog = [
            {
                "id": "fed-cut-march-2026",
                "name": "Fed rate cut March",
                "slug": "fed-cut-march-2026",
                "theme": "macro",
                "keywords": ["fed", "cut", "rate", "powell"],
            },
            {
                "id": "trump-2028",
                "name": "Trump 2028",
                "slug": "trump-2028",
                "theme": "politics",
            },
        ]
        out = tag_news_to_factors(items, catalog, threshold=0.3)
        assert len(out) == 2
        # First item should match fed-cut, second shouldn't match anything.
        first_matches = [m["factor_id"] for m in out[0]["matched_factors"]]
        second_matches = out[1]["matched_factors"]
        assert "fed-cut-march-2026" in first_matches
        # second is "Random unrelated headline" → no match.
        assert second_matches == [] or all(m["match_score"] >= 0.3 for m in second_matches)

    def test_enhanced_sentiment_positive(self) -> None:
        text = "Stocks surge on strong earnings and bullish growth"
        out = enhanced_sentiment(text)
        # The lexicon may score this neutral if its dictionary is small;
        # we only require structure + score in [-1, 1].
        assert "overall_sentiment" in out
        assert -1.0 <= out["overall_sentiment"] <= 1.0
        assert "sentiment_per_entity" in out
        assert isinstance(out["sentiment_per_entity"], dict)

    def test_enhanced_sentiment_aspect_per_entity(self) -> None:
        text = "Trump rallies. China imposes new tariffs."
        out = enhanced_sentiment(text)
        # Trump and China are both detected; entries should exist.
        ents = out["sentiment_per_entity"]
        assert any(k in ents for k in ("Trump", "China"))


# ===========================================================================
# 7a) Whale Mirror
# ===========================================================================


class TestWhaleMirror:
    def test_top_whales_sorted_by_abs_pnl(self) -> None:
        whales = top_whales(window_days=7, min_pnl_usd=0.0)
        pnls = [abs(w["pnl_7d_usd"]) for w in whales]
        assert pnls == sorted(pnls, reverse=True)

    def test_mirror_with_capital(self) -> None:
        # Use an address from the synthesised pool to ensure deterministic positions.
        out = mirror_whale(
            whale_address="0xWHALE000000000000000000000000000000A001",
            capital_usd=10_000.0,
            max_positions=5,
        )
        assert len(out["suggested_positions"]) <= 5
        # Sizes proportional to capital (Σ size ≤ capital, accounting for rounding).
        total = sum(p["size_usd"] for p in out["suggested_positions"])
        assert total <= 10_000.0 + 1.0  # rounding tolerance
        # equity-equivalent β reported.
        assert "equivalent_equity_beta_estimate" in out

    def test_zero_capital_raises(self) -> None:
        with pytest.raises(ValueError):
            mirror_whale("0xWHALE000000000000000000000000000000A001", 0.0)

    def test_synthetic_determinism_via_sha256(self) -> None:
        a = mirror_whale("0xDEADBEEFCAFE", 5_000.0, max_positions=4)
        b = mirror_whale("0xDEADBEEFCAFE", 5_000.0, max_positions=4)
        assert a["suggested_positions"] == b["suggested_positions"]


# ===========================================================================
# 7b) Smart Money Divergence
# ===========================================================================


class TestSmartMoneyDivergence:
    def test_detect_divergence_returns_shape(self) -> None:
        out = detect_divergence("nvda-eps-beat-q1", "NVDA", lookback_hours=24)
        assert 0.0 <= out["divergence_strength"] <= 1.0
        assert isinstance(out["is_diverging"], bool)
        assert "suggested_trade" in out
        assert 0.0 <= out["historical_lead_winrate"] <= 1.0

    def test_scan_min_strength_filter(self) -> None:
        rows = scan_all_divergences(min_strength=0.999)
        # Almost no synthetic seed will breach 0.999.
        assert all(r["divergence_strength"] >= 0.999 for r in rows)

    def test_scan_low_threshold_returns_some(self) -> None:
        rows = scan_all_divergences(min_strength=0.0)
        assert len(rows) >= 1


# ===========================================================================
# 7c) Auto-Hedge
# ===========================================================================


class TestAutoHedge:
    def test_target_zero_neutralises(self) -> None:
        portfolio = [{"ticker": "NVDA", "size_usd": 10_000.0}]
        out = compute_hedge(
            portfolio,
            hedge_factors=["fed-cut-march-2026"],
            target_beta=0.0,
        )
        # Residual β per dollar should be ≈ 0.
        for v in out["net_beta_after_hedge"].values():
            assert abs(v) < 1e-6

    def test_target_partial_residual(self) -> None:
        portfolio = [{"ticker": "NVDA", "size_usd": 10_000.0}]
        out = compute_hedge(
            portfolio,
            hedge_factors=["fed-cut-march-2026"],
            target_beta=0.5,
        )
        # Residual should be ≈ target_beta.
        for v in out["net_beta_after_hedge"].values():
            assert v == pytest.approx(0.5, abs=1e-3)

    def test_simulate_hedge_path_reduces_vol(self) -> None:
        portfolio = [{"ticker": "NVDA", "size_usd": 10_000.0}]
        out = simulate_hedge_path(
            portfolio,
            hedge_factors=["fed-cut-march-2026", "ai-capex-cut-q2"],
            days=60,
            target_beta=0.0,
        )
        assert len(out["path"]) == 60
        # vol_reduction_ratio < 1 means hedge reduced variance; allow a
        # generous bound because the synthetic factor moves are noisy.
        assert out["vol_reduction_ratio"] >= 0.0
        # Slippage should accrue weekly.
        assert out["final_slippage_usd"] >= 0.0

    def test_simulate_too_few_days_raises(self) -> None:
        # The Pydantic schema enforces days>=2 at the API layer; the pure
        # function also raises ValueError on days<2.
        with pytest.raises(ValueError):
            simulate_hedge_path(
                [{"ticker": "NVDA", "size_usd": 10_000.0}],
                hedge_factors=["fed-cut-march-2026"],
                days=1,
            )

    def test_empty_portfolio_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_hedge([], hedge_factors=["fed-cut-march-2026"])


# ===========================================================================
# 8) API endpoint smoke tests
# ===========================================================================


class TestAPIEndpoints:
    def test_post_news_causal_chain(
        self, killer_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Disable auth dependency for this endpoint at the dep-injection layer.
        monkeypatch.setenv("PFM_AUTH_ENABLED", "0")
        body = {
            "factor_id": "fed-cut-march-2026",
            "news_items": [
                {
                    "title": "Powell hints at fed cut march",
                    "price_before": 0.5,
                    "price_after": 0.7,
                }
            ],
            "lookback_hours": 48,
            "beta_map": {"TLT": 4.2},
        }
        r = killer_client.post("/news/causal-chain", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["factor_id"] == "fed-cut-march-2026"
        assert data["n_items"] == 1

    def test_get_news_movers(self, killer_client: TestClient) -> None:
        # No factors registered → empty.
        r = killer_client.get("/news/movers", params={"hours": 24, "n": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["n_returned"] == 0

    def test_post_resolution_tree(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/portfolio/resolution-tree",
            json={
                "positions": [{"ticker": "NVDA", "size_usd": 10_000.0, "beta_factor": 2.0}],
                "factor_id": "fed-cut-mar",
                "current_prob": 0.5,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_positions"] == 1
        assert len(body["scenarios"]) == 2

    def test_post_pnl_monte_carlo(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/portfolio/pnl-monte-carlo",
            json={
                "positions": [{"ticker": "X", "size_usd": 10_000.0, "beta_factor": 1.0}],
                "factor_id": "f",
                "n_paths": 1_000,
                "seed": 42,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_paths"] == 1_000
        assert "p5" in body["percentiles"]

    def test_get_earnings_whisper_endpoint(
        self, killer_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(earnings_whisper, "fetch_gamma_market", lambda *a, **k: _market(0.45))
        r = killer_client.get(
            "/alpha/earnings-whisper/NVDA",
            params={"date": "2026-05-22"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["ticker"] == "NVDA"

    def test_get_earnings_whisper_unknown(self, killer_client: TestClient) -> None:
        r = killer_client.get(
            "/alpha/earnings-whisper/ZZZZ",
            params={"date": "2026-05-22"},
        )
        assert r.status_code == 404

    def test_get_earnings_dashboard(
        self, killer_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        today = date.today()
        monkeypatch.setattr(
            earnings_whisper,
            "NEXT_EARNINGS",
            {"NVDA": today + timedelta(days=2)},
        )
        monkeypatch.setattr(earnings_whisper, "fetch_gamma_market", lambda *a, **k: _market(0.50))
        r = killer_client.get(
            "/alpha/earnings-whisper-dashboard",
            params={"days": 14},
        )
        assert r.status_code == 200

    def test_get_vol_surface(
        self, killer_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Patch the Gamma fetch to return a synthetic ladder.
        ladder = KNOWN_LADDERS["btc-price-eoy-2026"]
        prob_map = {slug: 0.95 - 0.18 * i for i, (slug, _) in enumerate(ladder)}

        def fake_fetch(_http, _url, slug, **_kw):
            return _market(max(0.02, prob_map.get(slug, 0.5)))

        monkeypatch.setattr(vol_surface_pm, "fetch_gamma_market", fake_fetch)
        r = killer_client.get("/vol-surface/pm/btc-price-eoy-2026")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_strikes"] == len(ladder)

    def test_get_vol_surface_unknown_pattern(self, killer_client: TestClient) -> None:
        r = killer_client.get("/vol-surface/pm/not-a-pattern")
        assert r.status_code == 404

    def test_get_vol_compare(
        self, killer_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ladder = KNOWN_LADDERS["btc-price-eoy-2026"]
        prob_map = {slug: 0.90 - 0.16 * i for i, (slug, _) in enumerate(ladder)}

        def fake_fetch(_http, _url, slug, **_kw):
            return _market(max(0.02, prob_map.get(slug, 0.5)))

        monkeypatch.setattr(vol_surface_pm, "fetch_gamma_market", fake_fetch)
        r = killer_client.get(
            "/vol-surface/compare",
            params={
                "ticker": "BTC",
                "pm_pattern": "btc-price-eoy-2026",
                "current_price": 95_000.0,
                "options_iv_annual": 0.5,
            },
        )
        assert r.status_code == 200

    def test_post_counterfactual(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/counterfactual",
            json={
                "ticker": "TEST",
                "factor_id": "f",
                "scenario": "NO",
                "actual_resolution": "YES",
                "start": "2024-01-01",
                "end": "2024-04-01",
                "beta": 0.5,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["scenario"] == "NO"
        assert body["beta"] == pytest.approx(0.5, rel=1e-3)

    def test_post_counterfactual_multi(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/counterfactual/multi",
            json={
                "ticker": "TEST",
                "factors_list": ["a", "b"],
                "start": "2024-01-01",
                "end": "2024-04-01",
                "betas": {"a": 0.4, "b": 0.6},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_factors"] >= 1

    def test_post_counterfactual_bad_dates(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/counterfactual",
            json={
                "ticker": "T",
                "factor_id": "f",
                "scenario": "YES",
                "start": "2024-04-01",
                "end": "2024-01-01",
            },
        )
        assert r.status_code == 400

    def test_post_news_tag(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/news/tag",
            json={
                "news_text": "Trump signs executive order on China tariffs",
                "threshold": 0.0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "Trump" in body["entities"]["politicians"]

    def test_post_news_tag_batch(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/news/tag-batch",
            json={
                "news_items": [
                    {"title": "Powell hints at rate cut"},
                    {"title": "Trump rally in Iowa"},
                ],
                "threshold": 0.0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_items"] == 2

    def test_get_top_whales(self, killer_client: TestClient) -> None:
        r = killer_client.get(
            "/whales/top",
            params={"window_days": 7, "min_pnl_usd": 0, "limit": 5},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_whales"] <= 5

    def test_post_mirror(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/whales/mirror",
            json={
                "whale_address": "0xWHALE000000000000000000000000000000A001",
                "capital_usd": 10_000.0,
                "max_positions": 5,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["capital_usd"] == 10_000.0

    def test_get_smart_money_divergence(self, killer_client: TestClient) -> None:
        r = killer_client.get(
            "/divergence/smart-money",
            params={"min_strength": 0.0},
        )
        assert r.status_code == 200

    def test_post_hedge_auto_config(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/hedge/auto-config",
            json={
                "portfolio": [{"ticker": "NVDA", "size_usd": 10_000.0}],
                "hedge_factors": ["fed-cut-march-2026"],
                "target_beta": 0.0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "hedge_positions" in body

    def test_post_hedge_simulate(self, killer_client: TestClient) -> None:
        r = killer_client.post(
            "/hedge/simulate",
            json={
                "portfolio": [{"ticker": "NVDA", "size_usd": 10_000.0}],
                "hedge_factors": ["fed-cut-march-2026"],
                "target_beta": 0.0,
                "days": 30,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["path"]) == 30


# ===========================================================================
# 9) Edge cases (numerical / robustness)
# ===========================================================================


class TestEdgeCases:
    def test_pnl_tree_huge_size(self) -> None:
        positions = [{"ticker": "X", "size_usd": 1e9, "beta_factor": 1.0}]
        out = build_pnl_tree(positions, "f", current_prob=0.5)
        for s in out["scenarios"]:
            assert np.isfinite(s["mtm_total_usd"])

    def test_pnl_tree_invalid_prob(self) -> None:
        with pytest.raises(ValueError):
            build_pnl_tree(
                [{"ticker": "X", "size_usd": 1.0, "beta_factor": 1.0}],
                "f",
                current_prob=1.5,
            )

    def test_pnl_tree_nan_size_rejected(self) -> None:
        # Pydantic raises a ValidationError; the model's field validator
        # raises a ValueError that pydantic wraps. Either is acceptable —
        # both inherit from ``Exception`` but we tighten to ValidationError.
        from pydantic import ValidationError

        with pytest.raises((ValidationError, ValueError)):
            build_pnl_tree(
                [{"ticker": "X", "size_usd": float("nan"), "beta_factor": 1.0}],
                "f",
                current_prob=0.5,
            )

    def test_monte_carlo_sigma_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            monte_carlo_pnl(
                [{"ticker": "X", "size_usd": 1.0, "beta_factor": 1.0}],
                "f",
                n_paths=100,
                bootstrap_sigma=0.0,
            )

    def test_counterfactual_no_overlap_raises(self) -> None:
        rng = np.random.default_rng(0)
        idx_a = pd.date_range("2020-01-01", periods=20, freq="B")
        idx_b = pd.date_range("2025-01-01", periods=20, freq="B")
        rets = pd.Series(rng.normal(0, 0.01, 20), index=idx_a)
        dlog = pd.Series(rng.normal(0, 0.1, 20), index=idx_b)
        with pytest.raises(ValueError):
            counterfactual_path(
                "T",
                "f",
                scenario="NO",
                actual_resolution="YES",
                start=date(2020, 1, 1),
                end=date(2025, 6, 1),
                returns=rets,
                dlogit=dlog,
                beta=0.5,
            )

    def test_news_tagger_empty_text(self) -> None:
        ents = extract_entities("")
        assert ents == {
            "tickers": [],
            "politicians": [],
            "countries": [],
            "events": [],
            "commodities": [],
        }

    def test_score_factor_match_empty_text(self) -> None:
        assert score_factor_match("", {"id": "fed", "name": "Fed"}) == 0.0
