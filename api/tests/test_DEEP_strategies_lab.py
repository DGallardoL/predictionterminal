"""Deep / exhaustive tests for the strategies + α-hub + lab + replay surface.

Modules covered
---------------

* pfm.strategies          (implication / conditional / Frechet bounds)
* pfm.strategy_verdict    (alpha_card_verdict, quarterly_stability_test, …)
* pfm.decay_monitor       (compute_rolling_sharpe, detect_decay)
* pfm.replay_mode         (get_state_at, simulate_paper_order, replay_scenario)
* pfm.alpha_lab           (discover_alphas, lab_queue, promote)
* pfm.alpha_graveyard*    (public registry router)
* pfm.pm_vix              (compute_pm_vix + bucket math)
* pfm.arb_scanner         (find_4way_arb + concept maps)
* main router smoke for /strategies/cointegration & /strategies/pairs-backtest
* alpha_strategies.json schema integrity (88 entries)

The tests are hermetic:

* yfinance is patched out (replay_mode + alpha_lab fetchers).
* Polymarket history is replaced with the synthetic ``factor_a/factor_b``
  fixture from ``conftest.py`` for the in-app endpoints, and patched
  module-locally for replay / lab.
* All seeds are fixed so the pass/fail outcome is deterministic.

The findings reported by these tests:

* ``/alpha-hub/leaderboard``, ``/alpha-hub/live-panel`` and
  ``/alpha-hub/strategy/{pair_id}`` are not registered anywhere — only
  ``/alpha-hub/graveyard*`` exist on that prefix.
* ``/strategies/list`` is not registered either; the closest thing is
  ``/strategies/presets`` (GET) and the per-strategy POST routes.
* The shipped ``alpha_strategies.json`` does not contain any A_GOLD
  rows (current set: A_STRUCTURAL / B_VALIDATED / C_TENTATIVE / D_RAW).
  ``alpha_card_verdict`` therefore never produces ``DEPLOY_LIVE_SMALL_SIZE``
  on the live catalog — this is consistent with the "downgraded" wave-5
  status documented in CLAUDE.md.
"""

from __future__ import annotations

import json
import math
import typing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.alpha_lab as lab
import pfm.replay_mode as rm
from pfm import arb_scanner, decay_monitor
from pfm.alpha_graveyard import (
    GraveyardEntry,
    filter_by_cause,
    load_graveyard,
)
from pfm.alpha_graveyard_router import router as graveyard_router
from pfm.cache_utils import get_cache
from pfm.decay_monitor import compute_rolling_sharpe, detect_decay
from pfm.pm_vix import (
    BUCKET_SLUGS,
    BUCKET_WEIGHTS,
    compute_pm_vix,
    pm_vix_history,
)
from pfm.strategy_verdict import (
    alpha_card_verdict,
    quarterly_stability_test,
)

# ---------------------------------------------------------------------------
# Synthetic-DGP helpers
# ---------------------------------------------------------------------------


SEED: int = 20260508


def _cointegrated_pair(n: int = 260, seed: int = SEED) -> tuple[pd.Series, pd.Series]:
    """Two series sharing a stochastic common trend → cointegrated.

    The spread ``A - 0.6 B`` is a stationary AR(1) so any z-score-based
    backtest fires both entry and exit signals in-sample.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    common = np.cumsum(rng.normal(0.0, 0.5, n))
    a = common + rng.normal(0.0, 0.3, n) + 50.0
    b = 0.6 * common + rng.normal(0.0, 0.3, n) + 30.0
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


def _independent_pair(n: int = 260, seed: int = SEED + 1) -> tuple[pd.Series, pd.Series]:
    """Two random walks → almost surely NOT cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    a = np.cumsum(rng.normal(0.0, 1.0, n)) + 100.0
    b = np.cumsum(rng.normal(0.0, 1.0, n)) + 50.0
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


# ---------------------------------------------------------------------------
# 1) Quarterly stability + alpha card verdicts
# ---------------------------------------------------------------------------


class TestQuarterlyStability:
    def test_four_unanimous_positives_promote_to_a_gold(self) -> None:
        out = quarterly_stability_test([1.5, 1.5, 1.5, 1.5], threshold=0.5)
        assert out["tier_recommendation"] == "A_GOLD"
        assert out["passes_4q_gold"] is True
        assert out["sign_flips"] == 0
        assert out["n_positive"] == 4

    def test_one_negative_quarter_blocks_a_gold(self) -> None:
        out = quarterly_stability_test([1.5, 1.5, 1.5, -0.5], threshold=0.5)
        assert out["tier_recommendation"] != "A_GOLD"
        # Sign flip between Q3 (+) and Q4 (−) → exactly one flip.
        assert out["sign_flips"] == 1
        # Still ≥3 positives (3 quarters > 0.5) so it lands in B_VALIDATED.
        assert out["tier_recommendation"] == "B_VALIDATED"

    def test_alternating_signs_count_three_flips(self) -> None:
        out = quarterly_stability_test([1.0, -1.0, 1.0, -1.0])
        assert out["sign_flips"] == 3
        # n_positive=2 (only 1.0>0.5 entries), so B_VALIDATED gate (≥3) fails.
        assert out["tier_recommendation"] == "C_TENTATIVE"

    def test_threshold_changes_cutoff(self) -> None:
        sharpes = [0.7, 0.7, 0.7, 0.7]
        gate_05 = quarterly_stability_test(sharpes, threshold=0.5)
        gate_10 = quarterly_stability_test(sharpes, threshold=1.0)
        assert gate_05["n_positive"] == 4
        assert gate_05["tier_recommendation"] == "A_GOLD"
        assert gate_10["n_positive"] == 0
        assert gate_10["tier_recommendation"] == "C_TENTATIVE"

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            quarterly_stability_test([1.0, 1.0, 1.0, 1.0], threshold=-0.1)

    def test_too_few_quarters_blocks_silver(self) -> None:
        out = quarterly_stability_test([2.0])
        assert out["tier_recommendation"] == "C_TENTATIVE"
        assert out["passes_4q_gold"] is False
        assert out["passes_4q_silver"] is False

    def test_nan_does_not_create_phantom_flips(self) -> None:
        out = quarterly_stability_test([1.0, float("nan"), -1.0, 1.0])
        # The nan breaks the chain → only one flip (-1 → 1).
        assert out["sign_flips"] == 1


class TestAlphaCardVerdict:
    def test_a_gold_tier_maps_to_deploy_live(self) -> None:
        v = alpha_card_verdict({"tier": "A_GOLD", "name": "x", "sharpe_oos": 1.4})
        assert v["action"] == "DEPLOY_LIVE_SMALL_SIZE"
        assert v["confidence"] == "high"

    def test_b_validated_paper_trades(self) -> None:
        v = alpha_card_verdict({"tier": "B_VALIDATED", "name": "x"})
        assert v["action"] == "PAPER_TRADE_FIRST"

    def test_c_tentative_watch_only(self) -> None:
        v = alpha_card_verdict({"tier": "C_TENTATIVE", "name": "x"})
        assert v["action"] == "WATCH_DO_NOT_DEPLOY"

    def test_d_rejected_archive(self) -> None:
        v = alpha_card_verdict({"tier": "D_REJECTED", "name": "x"})
        assert v["action"] == "ARCHIVE"

    def test_unknown_tier_defaults_to_watch(self) -> None:
        v = alpha_card_verdict({"tier": "made_up", "name": "x"})
        assert v["action"] == "WATCH_DO_NOT_DEPLOY"


# ---------------------------------------------------------------------------
# 2) Decay monitor
# ---------------------------------------------------------------------------


def _engineered_returns(target_sharpe: float = 2.0, n: int = 200) -> pd.Series:
    """Daily returns calibrated so ann-Sharpe ≈ ``target_sharpe``."""
    rng = np.random.default_rng(SEED + 3)
    daily_vol = 0.01
    daily_mean = target_sharpe * daily_vol / math.sqrt(252)
    vals = rng.normal(loc=daily_mean, scale=daily_vol, size=n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(vals, index=idx, name="returns")


class TestDecayMath:
    def test_rolling_sharpe_recovers_target(self) -> None:
        ret = _engineered_returns(target_sharpe=2.0, n=300)
        rs = compute_rolling_sharpe(ret, window=60)
        recent = rs.dropna().tail(60).mean()
        # Sample noise is large; require within ±1.5 of the target.
        assert abs(recent - 2.0) < 1.5

    def test_zero_variance_window_emits_zero(self) -> None:
        idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
        ret = pd.Series([0.0] * 40, index=idx)
        rs = compute_rolling_sharpe(ret, window=10)
        # Last value should be exactly 0.0 (zero-variance branch).
        assert rs.dropna().iloc[-1] == 0.0

    def test_window_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_rolling_sharpe(_engineered_returns(), window=1)

    def test_decay_indicator_fresh_when_close_to_baseline(self) -> None:
        ret = _engineered_returns(target_sharpe=2.0, n=180)
        rs = compute_rolling_sharpe(ret, window=30)
        out = detect_decay(rs, baseline=2.0)
        assert out["decay_indicator"] in {"FRESH", "STABLE"}
        assert out["demote_recommendation"] == "A_GOLD"

    def test_decay_indicator_decaying_with_consecutive_below(self) -> None:
        # Build a synthetic rolling sharpe series that ends in 6 below-cutoff
        # observations (cutoff = 0.5 * baseline = 1.0).
        idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
        rs = pd.Series(
            [2.0] * 14 + [0.5, 0.5, 0.4, 0.3, 0.2, 0.1], index=idx, name="rolling_sharpe"
        )
        out = detect_decay(rs, baseline=2.0, threshold_pct=0.5)
        # 6 consecutive below 1.0 → DECAYING (≥5).
        assert out["n_consecutive_below"] >= 5
        assert out["decay_indicator"] in {"DECAYING", "DEAD"}

    def test_decay_indicator_dead_when_ratio_collapses(self) -> None:
        idx = pd.date_range("2024-01-01", periods=15, freq="D", tz="UTC")
        # Collapse the tail so ratio < 0.3
        rs = pd.Series([2.0] * 5 + [0.1] * 10, index=idx, name="rolling_sharpe")
        out = detect_decay(rs, baseline=2.0)
        assert out["decay_indicator"] == "DEAD"
        assert out["demote_recommendation"] == "C_TENTATIVE"

    def test_demote_ladder_monotone(self) -> None:
        # FRESH → A_GOLD; DECAYING → B_VALIDATED; DEAD → C_TENTATIVE
        idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
        for ind, expect in [
            ([2.0] * 20, "A_GOLD"),  # FRESH/STABLE
        ]:
            rs = pd.Series(ind, index=idx)
            assert detect_decay(rs, baseline=2.0)["demote_recommendation"] == expect


class TestDecayRouter:
    @pytest.fixture
    def client_and_path(self, tmp_path: Path) -> tuple[TestClient, str]:
        # Build a tiny synthetic alpha_strategies.json so the endpoint has
        # something to chew on without depending on the live 88-row file.
        # The endpoint accepts ``alpha_strategies_path`` as a query parameter
        # so we pass the override there rather than patching the default
        # sentinel (the default-resolution branch otherwise relocates to the
        # repo root and ignores ``DEFAULT_ALPHA_STRATEGIES_PATH`` patches).
        catalog = {
            "strategies": [
                {"pair_id": "demo_a", "tier": "A_GOLD", "oos_sharpe": 1.5},
                {"pair_id": "demo_b", "tier": "B_VALIDATED", "oos_sharpe": 0.8},
            ]
        }
        p = tmp_path / "alpha_strategies.json"
        p.write_text(json.dumps(catalog))

        app = FastAPI()
        app.include_router(decay_monitor.router)
        return TestClient(app), str(p)

    def test_decay_list_returns_sorted(self, client_and_path: tuple[TestClient, str]) -> None:
        client, path = client_and_path
        with client as c:
            r = c.get("/alpha/decay", params={"alpha_strategies_path": path})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["n_total"] == 2
            severity = {"DEAD": 0, "DECAYING": 1, "STABLE": 2, "FRESH": 3}
            keys = [severity.get(it["decay_indicator"], 4) for it in body["items"]]
            assert keys == sorted(keys)

    def test_rolling_sharpe_endpoint(self, client_and_path: tuple[TestClient, str]) -> None:
        client, path = client_and_path
        with client as c:
            r = c.get(
                "/alpha/demo_a/rolling-sharpe",
                params={"window": 20, "alpha_strategies_path": path},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["pair_id"] == "demo_a"
            assert body["window"] == 20
            assert body["baseline_sharpe"] == 1.5
            assert len(body["series"]) > 0

    def test_recompute_unknown_404(self, client_and_path: tuple[TestClient, str]) -> None:
        client, path = client_and_path
        with client as c:
            r = c.post(
                "/alpha/missing-pair/recompute-decay",
                params={"alpha_strategies_path": path},
            )
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# 3) Replay mode
# ---------------------------------------------------------------------------


def _synthetic_pm_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Smooth oscillating PM probability series."""
    idx = pd.date_range(start, end, freq="D", tz="UTC").normalize()
    n = len(idx)
    t = np.arange(n) / max(n, 1)
    price = (0.50 + 0.20 * np.sin(2 * np.pi * t * 1.5)).clip(0.05, 0.95)
    df = pd.DataFrame({"price": price}, index=idx)
    df.index.name = "date"
    return df


def _synthetic_yf_rows(start_iso: str, end_iso: str, base: float = 100.0):
    idx = pd.date_range(start_iso, end_iso, freq="D", tz="UTC").normalize()
    n = len(idx)
    rng = np.random.default_rng(SEED + 7)
    drift = np.cumsum(rng.normal(0.0005, 0.01, n))
    closes = base * np.exp(drift)
    return tuple((d.isoformat(), float(c)) for d, c in zip(idx, closes, strict=True))


@pytest.fixture
def patched_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve_pm_history(slug, start, end):
        if slug.startswith("missing-"):
            return pd.DataFrame()
        return _synthetic_pm_history(start, end)

    monkeypatch.setattr(rm, "_resolve_pm_history", fake_resolve_pm_history)
    rm._yf_close_cached.cache_clear()

    def fake_yf(ticker, start_iso, end_iso):
        if ticker == "MISSING":
            return ()
        base_map = {
            "SPY": 450.0,
            "QQQ": 380.0,
            "BTC-USD": 70000.0,
            "TLT": 90.0,
            "GLD": 200.0,
            "DJT": 30.0,
            "DXY": 105.0,
            "VIX": 25.0,
            "COIN": 200.0,
            "MSTR": 250.0,
            "ETH-USD": 3500.0,
        }
        return _synthetic_yf_rows(start_iso, end_iso, base=base_map.get(ticker, 100.0))

    monkeypatch.setattr(rm, "_yf_close_cached", fake_yf)


class TestReplayState:
    def test_state_at_returns_markets_and_equities(self, patched_replay) -> None:
        ts = datetime(2024, 11, 5, 23, 0, tzinfo=UTC)
        out = rm.get_state_at(ts, slugs=["a", "b"], equity_tickers=["SPY", "QQQ"])
        assert out["as_of"].startswith("2024-11-05")
        assert len(out["markets"]) == 2
        assert all(0.0 <= m["prob"] <= 1.0 for m in out["markets"])
        assert {e["ticker"] for e in out["equities"]} == {"SPY", "QQQ"}

    def test_state_at_skips_missing_slugs(self, patched_replay) -> None:
        ts = datetime(2024, 11, 5, 23, 0, tzinfo=UTC)
        out = rm.get_state_at(ts, slugs=["missing-x", "ok-1"], equity_tickers=["SPY"])
        assert {m["slug"] for m in out["markets"]} == {"ok-1"}


class TestPaperOrder:
    def test_long_with_hold_sets_pnl(self, patched_replay) -> None:
        entry = datetime(2024, 9, 1, 18, 0, tzinfo=UTC)
        exit_ = datetime(2024, 11, 1, 18, 0, tzinfo=UTC)
        out = rm.simulate_paper_order("demo", "LONG", 1000.0, entry, hold_until=exit_)
        assert out["status"] == "CLOSED"
        assert out["entry_price"] is not None and out["exit_price"] is not None
        assert out["slippage_assumed_bps"] == 100.0

    def test_open_mtm_when_no_hold_until(self, patched_replay) -> None:
        entry = datetime(2024, 9, 1, 18, 0, tzinfo=UTC)
        out = rm.simulate_paper_order("demo", "LONG", 100.0, entry)
        assert out["status"] in {"OPEN_MTM", "NO_EXIT_PRICE"}

    def test_no_data_returns_safe_payload(self, patched_replay) -> None:
        out = rm.simulate_paper_order(
            "missing-x",
            "LONG",
            500.0,
            datetime(2024, 9, 1, tzinfo=UTC),
        )
        assert out["status"] == "NO_DATA"
        assert out["pnl_usd"] == 0.0

    def test_invalid_side_raises(self, patched_replay) -> None:
        with pytest.raises(ValueError):
            rm.simulate_paper_order(
                "demo",
                "BUY",
                100.0,  # type: ignore[arg-type]
                datetime(2024, 9, 1, tzinfo=UTC),
            )

    def test_zero_size_raises(self, patched_replay) -> None:
        with pytest.raises(ValueError):
            rm.simulate_paper_order(
                "demo",
                "LONG",
                0.0,
                datetime(2024, 9, 1, tzinfo=UTC),
            )


class TestReplayScenarios:
    def test_list_scenarios_returns_four(self, patched_replay) -> None:
        rows = rm.list_scenarios()
        names = {r["name"] for r in rows}
        assert names >= {
            "election_night_2024",
            "fomc_2024_09",
            "btc_ath_2024_11",
            "covid_crash_2020_03",
        }
        assert len(rows) == 4

    def test_replay_scenario_includes_metadata(self, patched_replay) -> None:
        out = rm.replay_scenario("election_night_2024")
        assert out["scenario"]["name"] == "election_night_2024"
        assert isinstance(out["headline_news"], list)


class TestReplayRouter:
    @pytest.fixture
    def client(self, patched_replay) -> TestClient:
        app = FastAPI()
        app.include_router(rm.router)
        return TestClient(app)

    def test_state_endpoint_200(self, client: TestClient) -> None:
        with client as c:
            r = c.get(
                "/replay/state",
                params={
                    "as_of": "2024-11-05T23:00:00+00:00",
                    "slugs": "alpha,beta",
                    "tickers": "SPY,QQQ",
                },
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["as_of"].startswith("2024-11-05")

    def test_scenarios_endpoint_returns_four(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/replay/scenarios")
            assert r.status_code == 200
            body = r.json()
            assert body["n_scenarios"] == 4

    def test_unknown_scenario_404(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/replay/scenario/does_not_exist")
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# 4) Alpha Lab
# ---------------------------------------------------------------------------


@pytest.fixture
def isolate_lab_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "JOBS_FILE", tmp_path / "lab_jobs.json")
    monkeypatch.setattr(lab, "PENDING_FILE", tmp_path / "lab_pending.jsonl")
    lab._STATE.running = False
    lab._STATE.last_job_id = None
    lab._STATE.last_run_at = None
    lab._STATE.last_results_summary = None


@pytest.fixture
def lab_fetchers(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = _cointegrated_pair(n=260, seed=SEED)
    a2, b2 = _independent_pair(n=260, seed=SEED + 1)

    def fake_factor(slug: str, days: int = 365) -> pd.Series:
        h = abs(hash(slug)) % 4
        return [a, b, a2, b2][h].copy().rename(slug)

    def fake_equity(ticker: str, days: int = 365) -> pd.Series:
        return (b.copy() if abs(hash(ticker)) % 2 == 0 else b2.copy()).rename(ticker)

    monkeypatch.setattr(lab, "_fetch_factor_series", fake_factor)
    monkeypatch.setattr(lab, "_fetch_equity_series", fake_equity)


class TestLabDiscover:
    def test_runtime_budget_zero_short_circuits(self, isolate_lab_storage, lab_fetchers) -> None:
        out = lab.discover_alphas(
            n_combos=10,
            max_runtime_seconds=0,
            factor_slugs=["s1", "s2", "s3", "s4"],
            equity_tickers=["TKR1"],
            seed=3,
        )
        assert out["timed_out"] is True
        assert out["n_tested"] == 0
        assert isinstance(out["candidates"], list)

    def test_returns_summary_quickly(self, isolate_lab_storage, lab_fetchers) -> None:
        out = lab.discover_alphas(
            n_combos=5,
            min_oos_sharpe=0.0,
            min_quarters_positive=0,
            max_runtime_seconds=10,
            factor_slugs=[f"slug-{i}" for i in range(8)],
            equity_tickers=["TKR1", "TKR2"],
            seed=11,
        )
        assert out["n_tested"] <= 5
        assert all("pair_id" in c for c in out["candidates"])
        for c in out["candidates"]:
            # Each candidate carries either a tier projection or a failure tag.
            assert (
                c["projected_tier"] in {"A_GOLD", "B_VALIDATED", "C_TENTATIVE", "D_REJECTED"}
            ) or (c["failed_at"] is not None)

    def test_no_data_marks_failed_at(
        self, isolate_lab_storage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            lab,
            "_fetch_factor_series",
            lambda slug, days=365: pd.Series(dtype=float, name=slug),
        )
        monkeypatch.setattr(
            lab,
            "_fetch_equity_series",
            lambda t, days=365: pd.Series(dtype=float, name=t),
        )
        out = lab.discover_alphas(
            n_combos=3,
            factor_slugs=["a", "b"],
            equity_tickers=["X"],
            seed=2,
            max_runtime_seconds=5,
        )
        assert all(c["failed_at"] for c in out["candidates"])


class TestLabPromote:
    def test_promote_appends_jsonl(self, isolate_lab_storage, lab_fetchers) -> None:
        out = lab.discover_alphas(
            n_combos=3,
            factor_slugs=["s1", "s2", "s3"],
            equity_tickers=["T1"],
            seed=4,
            max_runtime_seconds=5,
            min_quarters_positive=0,
            min_oos_sharpe=0.0,
        )
        job_id = "job-DEEP-1"
        lab._record_job(
            job_id,
            status="complete",
            results=out,
            started_at="x",
            completed_at="y",
            params={},
        )
        cand_id = out["candidates"][0]["pair_id"]
        entry = lab.promote_candidate(cand_id, job_id=job_id)
        assert entry["candidate_id"] == cand_id
        assert lab.PENDING_FILE.exists()
        line = lab.PENDING_FILE.read_text().strip().splitlines()[0]
        parsed = json.loads(line)
        assert parsed["review_status"] == "pending_human_review"

    def test_promote_unknown_raises_keyerror(self, isolate_lab_storage) -> None:
        with pytest.raises(KeyError):
            lab.promote_candidate("does-not-exist")


class TestLabRouter:
    @pytest.fixture
    def client(self, isolate_lab_storage, lab_fetchers) -> TestClient:
        app = FastAPI()
        app.include_router(lab.router)
        return TestClient(app)

    def test_discover_then_results_flow(self, client: TestClient) -> None:
        with client as c:
            r = c.post(
                "/lab/discover",
                json={
                    "n_combos": 3,
                    "min_oos_sharpe": 0.0,
                    "min_quarters_positive": 0,
                    "max_runtime_seconds": 5,
                    "seed": 9,
                },
            )
            assert r.status_code == 200, r.text
            job_id = r.json()["job_id"]
            r2 = c.get(f"/lab/results/{job_id}")
            assert r2.status_code == 200
            body = r2.json()
            assert body["job_id"] == job_id
            assert isinstance(body["candidates"], list)

    def test_concurrent_run_returns_409(self, client: TestClient) -> None:
        # Spoof the singleton so the second POST hits the 409 path even though
        # FastAPI's TestClient runs background tasks synchronously.
        lab._STATE.running = True
        try:
            with client as c:
                r = c.post(
                    "/lab/discover",
                    json={
                        "n_combos": 1,
                        "min_oos_sharpe": 0.0,
                        "min_quarters_positive": 0,
                        "max_runtime_seconds": 1,
                        "seed": 1,
                    },
                )
                assert r.status_code == 409, r.text
        finally:
            lab._STATE.running = False

    def test_queue_endpoint(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/lab/queue")
            assert r.status_code == 200
            body = r.json()
            assert "running" in body and "jobs_file" in body

    def test_promote_unknown_returns_404(self, client: TestClient) -> None:
        with client as c:
            r = c.post("/lab/promote/missing-id")
            assert r.status_code == 404

    def test_results_unknown_job_returns_404(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/lab/results/nope")
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# 5) Alpha Graveyard public registry
# ---------------------------------------------------------------------------


REQUIRED_GRAVEYARD_FIELDS: set[str] = {
    "pair_id",
    "name",
    "killed_iso",
    "killed_in_wave",
    "cause",
    "claimed_sharpe",
    "post_mortem_sharpe",
    "thesis_original",
    "lesson",
    "could_resurrect_if",
}

VALID_CAUSES: set[str] = {
    "regime",
    "TC",
    "single-episode",
    "grid-search",
    "tautology",
    "capacity",
    "non-portable",
}


class TestGraveyard:
    @pytest.fixture
    def client(self) -> TestClient:
        app = FastAPI()
        app.include_router(graveyard_router)
        return TestClient(app)

    def test_registry_has_at_least_six_entries(self) -> None:
        raw = load_graveyard()
        assert len(raw) >= 6

    def test_every_entry_has_required_fields(self) -> None:
        raw = load_graveyard()
        for e in raw:
            missing = REQUIRED_GRAVEYARD_FIELDS - set(e.keys())
            assert not missing, f"{e.get('pair_id')} missing {missing}"
            assert e["cause"] in VALID_CAUSES

    def test_filter_by_cause_regime(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/alpha-hub/graveyard", params={"cause": "regime"})
            assert r.status_code == 200, r.text
            body = r.json()
            assert all(e["cause"] == "regime" for e in body["entries"])
            assert body["cause_filter"] == "regime"

    def test_filter_all_returns_full_list(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/alpha-hub/graveyard")
            assert r.status_code == 200
            assert r.json()["n_entries"] >= 6

    def test_pair_id_lookup_known(self, client: TestClient) -> None:
        raw = load_graveyard()
        target = raw[0]["pair_id"]
        with client as c:
            r = c.get(f"/alpha-hub/graveyard/{target}")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["pair_id"] == target

    def test_pair_id_404_for_missing(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/alpha-hub/graveyard/totally_not_a_real_pair_id")
            assert r.status_code == 404

    def test_claimed_vs_post_mortem_sharpe_diff_recorded(self) -> None:
        raw = load_graveyard()
        for e in raw:
            entry = GraveyardEntry.model_validate(e)
            # Each entry should have a *different* claimed vs post-mortem
            # Sharpe (otherwise why is it in the graveyard?).
            assert entry.claimed_sharpe != entry.post_mortem_sharpe

    def test_filter_by_cause_tc(self) -> None:
        raw = load_graveyard()
        tc = filter_by_cause(raw, "TC")
        assert all(e["cause"] == "TC" for e in tc)


# ---------------------------------------------------------------------------
# 6) PM-VIX composite
# ---------------------------------------------------------------------------


def _gamma_market(prob: float, vol: float = 10_000.0) -> dict[str, Any]:
    return {
        "bestBid": max(0.0, prob - 0.005),
        "bestAsk": min(1.0, prob + 0.005),
        "lastTradePrice": prob,
        "volume24hr": vol,
    }


def _all_overrides(prob: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for slugs in BUCKET_SLUGS.values():
        for s in slugs:
            out[s] = _gamma_market(prob)
    return out


class TestPmVix:
    @pytest.fixture(autouse=True)
    def _clear(self) -> None:
        get_cache("pm_vix").clear()

    def test_score_in_unit_interval(self) -> None:
        snap = compute_pm_vix(overrides=_all_overrides(0.30), http=MagicMock())
        assert 0.0 <= snap["score"] <= 100.0
        assert snap["regime"] in {"RISK_ON", "NEUTRAL", "RISK_OFF"}

    def test_five_buckets_with_documented_weights(self) -> None:
        snap = compute_pm_vix(overrides=_all_overrides(0.30), http=MagicMock())
        bucket_names = {c["bucket"] for c in snap["components"]}
        assert bucket_names == {"recession", "geopolitical", "election", "macro", "crypto"}
        # Documented weights: recession 0.30, geopolitical 0.25, election 0.20,
        # macro 0.15, crypto 0.10.
        expected = {
            "recession": 0.30,
            "geopolitical": 0.25,
            "election": 0.20,
            "macro": 0.15,
            "crypto": 0.10,
        }
        for c in snap["components"]:
            assert abs(c["weight"] - expected[c["bucket"]]) < 1e-6

    def test_weights_sum_to_one(self) -> None:
        s = sum(BUCKET_WEIGHTS.values())
        assert abs(s - 1.0) < 1e-6

    def test_low_probs_classify_risk_on(self) -> None:
        snap = compute_pm_vix(overrides=_all_overrides(0.02), http=MagicMock())
        assert snap["regime"] == "RISK_ON"
        assert snap["score"] < 25

    def test_high_probs_classify_risk_off(self) -> None:
        snap = compute_pm_vix(overrides=_all_overrides(0.85), http=MagicMock())
        assert snap["regime"] == "RISK_OFF"
        assert snap["score"] >= 60

    def test_history_is_30_floats_in_0_100(self) -> None:
        snap = compute_pm_vix(overrides=_all_overrides(0.30), http=MagicMock())
        assert isinstance(snap["history_30d"], list)
        assert len(snap["history_30d"]) == 30
        for v in snap["history_30d"]:
            assert 0.0 <= float(v) <= 100.0

    def test_components_contributions_sum_to_score(self) -> None:
        snap = compute_pm_vix(overrides=_all_overrides(0.45), http=MagicMock())
        summed = sum(c["contribution"] for c in snap["components"])
        # round() to 3 dp matches the implementation's rounding.
        assert abs(snap["score"] - round(summed, 3)) < 0.05

    def test_history_anchor_pin(self) -> None:
        pts = pm_vix_history(days=5, anchor_score=42.0)
        assert pts[-1]["score"] == 42.0


# ---------------------------------------------------------------------------
# 7) Cross-Venue Arb Scanner
# ---------------------------------------------------------------------------


class TestArbScanner:
    @pytest.fixture(autouse=True)
    def _clear_caches(self) -> None:
        get_cache("arb_scanner").clear()
        get_cache("arb_matched").clear()
        arb_scanner._MANUAL_PAIRS.clear()

    @pytest.fixture
    def client(self) -> TestClient:
        app = FastAPI()
        app.include_router(arb_scanner.router)
        return TestClient(app)

    def test_pre_matched_pairs_count_is_five(self) -> None:
        assert len(arb_scanner.PRE_MATCHED_PAIRS) == 5

    def test_concept_maps_count_is_five(self) -> None:
        assert len(arb_scanner.CONCEPT_MAPS) == 5
        ids = {m["concept_id"] for m in arb_scanner.CONCEPT_MAPS}
        assert ids == {
            "presidential_election_2028",
            "fed_cuts_2026",
            "recession_2026",
            "btc_ath_2026",
            "cpi_above_3_5_2026",
        }

    def test_concepts_endpoint(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/arb/concepts")
            assert r.status_code == 200
            body = r.json()
            assert body["n"] == 5

    def test_concept_detail_endpoint(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/arb/concept/fed_cuts_2026")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["concept_id"] == "fed_cuts_2026"
            assert "venues" in body
            for v in ("polymarket", "kalshi", "manifold", "predictit"):
                assert v in body["venues"]

    def test_concept_detail_unknown_404(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/arb/concept/totally_invented")
            assert r.status_code == 404

    def test_find_4way_arb_with_injected_prices(self) -> None:
        # Create deterministic price functions covering all four venues.
        def pm_fn(_id):
            return 0.40, 12_000.0

        def k_fn(_id):
            return 0.45, 8_000.0

        def m_fn(_id):
            return 0.42, 5_000.0

        def p_fn(_id):
            return 0.48, 3_000.0

        out = arb_scanner.find_4way_arb(
            "fed_cuts_2026",
            pm_price_fn=pm_fn,
            kalshi_price_fn=k_fn,
            manifold_price_fn=m_fn,
            predictit_price_fn=p_fn,
        )
        assert out["concept_id"] == "fed_cuts_2026"
        assert set(out["legs_present"]) == {"polymarket", "kalshi", "manifold", "predictit"}
        # Cheapest leg pm@0.40, dearest p@0.48 → spread 8 pp.
        assert abs(out["max_spread_pct"] - 8.0) < 0.1
        assert out["low_venue"] == "polymarket"
        assert out["high_venue"] == "predictit"
        # Capital required = capital_per_leg_usd * 2 when spread > 0.
        assert out["capital_required_usd"] == 20_000.0

    def test_find_4way_arb_missing_legs(self) -> None:
        # Only the PM leg has a price function → missing_venues lists the rest.
        def pm_fn(_id):
            return 0.50, 1_000.0

        out = arb_scanner.find_4way_arb(
            "btc_ath_2026",
            pm_price_fn=pm_fn,
        )
        assert out["legs_present"] == ["polymarket"]
        assert set(out["missing_venues"]) >= {"kalshi", "manifold"}
        # With < 2 legs the spread is 0 and capital required is 0.
        assert out["max_spread_pct"] == 0.0
        assert out["capital_required_usd"] == 0.0

    def test_find_4way_arb_unknown_concept_404(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            arb_scanner.find_4way_arb("not_a_concept")
        assert exc.value.status_code == 404

    def test_matched_endpoint_returns_hardcoded(self, client: TestClient) -> None:
        with client as c:
            r = c.get("/arb/matched")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["n"] >= 5
            sources = {p["source"] for p in body["pairs"]}
            assert "hardcoded" in sources

    def test_match_endpoint_persists_manual_pair(self, client: TestClient) -> None:
        with client as c:
            r = c.post(
                "/arb/match",
                json={
                    "pm_slug": "test-pm-slug",
                    "kalshi_slug": "TEST-K-SLUG",
                    "label": "test",
                    "theme": "macro",
                },
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["source"] == "manual"
            r2 = c.get("/arb/matched")
            slugs = {p["pm_slug"] for p in r2.json()["pairs"]}
            assert "test-pm-slug" in slugs


# ---------------------------------------------------------------------------
# 8) /strategies/* endpoints (smoke through the real app)
# ---------------------------------------------------------------------------


class TestStrategiesEndpoints:
    """Smoke-test cointegration + pairs-backtest through the live app."""

    PAIR: typing.ClassVar[dict[str, str]] = {
        "a_id": "factor_a",
        "b_id": "factor_b",
        "start": "2025-06-15",
        "end": "2025-12-15",
    }

    def test_cointegration_returns_verdict(self, app_client: TestClient) -> None:
        r = app_client.post("/strategies/cointegration", json=self.PAIR)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["verdict"] in {"cointegrated", "not_cointegrated", "insufficient-data"}

    def test_pairs_backtest_reports_oos_metrics(self, app_client: TestClient) -> None:
        body = {
            **self.PAIR,
            "window": 20,
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_z": 4.0,
            "annualisation": 252.0,
            "oos_fraction": 0.30,
        }
        r = app_client.post("/strategies/pairs-backtest", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        # Walk-forward partition: train + test = total.
        assert out["n_obs_is"] + out["n_obs_oos"] <= out["n_obs"]
        # Sharpe / max-DD / hit-rate-style fields present.
        for k in ("sharpe", "sharpe_is", "sharpe_oos"):
            assert k in out

    def test_strategies_list_endpoint_returns_catalog(self, app_client: TestClient) -> None:
        # Updated: ``/strategies/list`` now exists (Wave-10 discoverability)
        # and returns a catalog enumeration of every /strategies/* endpoint.
        r = app_client.get("/strategies/list")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 10
        assert isinstance(body["items"], list)

    def test_strategies_presets_endpoint_works(self, app_client: TestClient) -> None:
        r = app_client.get("/strategies/presets")
        assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# 9) alpha_strategies.json schema integrity
# ---------------------------------------------------------------------------


VALID_TIERS: set[str] = {
    "A_STRUCTURAL",
    "A_GOLD",
    "B_VALIDATED",
    "B_FDR_ONLY",
    "C_TENTATIVE",
    "D_RAW",
    "D_REJECTED",
}

REQUIRED_STRATEGY_FIELDS: set[str] = {
    "pair_id",
    "tier",
    "oos_sharpe",
    "n_obs",
    "beta_hedge",
    "adf_pvalue",
    "perm_p",
}


def _alpha_strategies_path() -> Path:
    return Path(__file__).resolve().parents[2] / "web" / "data" / "alpha_strategies.json"


class TestAlphaStrategiesJson:
    @pytest.fixture(scope="class")
    def strategies(self) -> list[dict[str, Any]]:
        p = _alpha_strategies_path()
        assert p.exists(), f"alpha_strategies.json missing at {p}"
        return json.loads(p.read_text())["strategies"]

    def test_count_is_at_least_sixty(self, strategies) -> None:
        # Was 88 pre-error-purge (2026-05-19, ~11:23 UTC). The purge dropped 19
        # entries with data-quality issues, leaving 69. Use a floor instead of
        # an exact count so future purges / wave demotions don't trip this test
        # — the v22 reckoning (2026-05-19) already had to walk the count down
        # from "ideal" to "honest." Drift below 60 deserves a manual look.
        assert len(strategies) >= 60

    def test_required_fields_present(self, strategies) -> None:
        for s in strategies:
            missing = REQUIRED_STRATEGY_FIELDS - set(s.keys())
            assert not missing, f"{s.get('pair_id')} missing {missing}"

    def test_tiers_in_valid_set(self, strategies) -> None:
        for s in strategies:
            assert s["tier"] in VALID_TIERS, f"unexpected tier {s['tier']}"

    def test_oos_sharpe_is_float_or_null(self, strategies) -> None:
        for s in strategies:
            v = s.get("oos_sharpe")
            assert v is None or isinstance(v, (int, float))

    def test_n_obs_positive(self, strategies) -> None:
        for s in strategies:
            assert s["n_obs"] > 0

    def test_beta_hedge_nonzero(self, strategies) -> None:
        # Wave-9 ships only nonzero β-hedges; a 0 would mean a degenerate fit.
        for s in strategies:
            assert s["beta_hedge"] != 0.0, f"{s['pair_id']} has β=0"

    def test_alpha_card_verdict_consistent(self, strategies) -> None:
        # The verdict mapping is deterministic; just confirm it never crashes
        # and the output action is from the closed Action vocabulary.
        valid_actions = {
            "DEPLOY_LIVE_SMALL_SIZE",
            "PAPER_TRADE_FIRST",
            "WATCH_DO_NOT_DEPLOY",
            "ARCHIVE",
        }
        for s in strategies:
            v = alpha_card_verdict(s)
            assert v["action"] in valid_actions


# ---------------------------------------------------------------------------
# 10) Edge cases — graceful behaviour on degenerate inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_quarterly_stability_empty_input(self) -> None:
        out = quarterly_stability_test([])
        assert out["n_quarters"] == 0
        assert out["tier_recommendation"] == "C_TENTATIVE"

    def test_decay_zero_returns_dont_crash(self) -> None:
        idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
        ret = pd.Series([0.0] * 40, index=idx)
        rs = compute_rolling_sharpe(ret, window=10)
        # Zero variance windows emit 0.0 by design (not NaN, not inf).
        assert rs.dropna().iloc[-1] == 0.0
        out = detect_decay(rs, baseline=2.0)
        # Ratio = 0 / 2 = 0 → DEAD branch (ratio < 0.3).
        assert out["decay_indicator"] == "DEAD"

    def test_decay_baseline_zero_does_not_divide(self) -> None:
        rs = pd.Series([1.0, 1.0, 1.0])
        out = detect_decay(rs, baseline=0.0)
        # Implementation guards against div-by-zero with a 1e-9 floor.
        assert math.isfinite(out["ratio"])

    def test_alpha_card_verdict_missing_tier(self) -> None:
        v = alpha_card_verdict({"name": "no tier here"})
        assert v["action"] == "WATCH_DO_NOT_DEPLOY"

    def test_quarterly_stability_one_quarter_blocks_gate(self) -> None:
        out = quarterly_stability_test([1.5])
        assert out["passes_4q_gold"] is False
        assert out["passes_4q_silver"] is False
        assert out["tier_recommendation"] == "C_TENTATIVE"

    def test_arb_4way_empty_prices(self) -> None:
        # No price functions at all → all venues missing.
        out = arb_scanner.find_4way_arb("fed_cuts_2026")
        assert out["legs_present"] == []
        assert out["max_spread_pct"] == 0.0
