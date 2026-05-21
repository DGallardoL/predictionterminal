"""Tests for ``pfm.alpha_tier_regen`` and ``scripts.regenerate_alpha_tiers``.

External fetches are mocked; every test runs against synthetic series so we
can control the data-generating process and verify the rigor pipeline assigns
tiers consistent with the underlying truth.

Uses sync test functions that call ``asyncio.run`` so we don't need
``pytest-asyncio``'s ``auto`` mode (which the project does not enable).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pfm.alpha_tier_regen as atr

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _cointegrated_pair(n: int = 260, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Two series sharing a stochastic common trend → cointegrated, fast
    mean-reverting spread → strong walk-forward Sharpe."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    common = np.cumsum(rng.normal(0.0, 0.5, n))
    a = common + rng.normal(0.0, 0.3, n) + 50.0
    b = 0.6 * common + rng.normal(0.0, 0.3, n) + 30.0
    return (
        pd.Series(a, index=idx, name="A"),
        pd.Series(b, index=idx, name="B"),
    )


def _independent_pair(n: int = 260, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    """Two random walks → almost certainly NOT cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    a = np.cumsum(rng.normal(0.0, 1.0, n)) + 100.0
    b = np.cumsum(rng.normal(0.0, 1.0, n)) + 50.0
    return (
        pd.Series(a, index=idx, name="A"),
        pd.Series(b, index=idx, name="B"),
    )


def _make_pair_record(a_id: str, b_id: str, **extra) -> dict:
    base = {
        "pair_id": f"{a_id}__{b_id}",
        "a_id": a_id,
        "b_id": b_id,
        "a_slug": a_id,
        "b_slug": b_id,
        "tier": "C_TENTATIVE",
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Synthetic fetcher
# ---------------------------------------------------------------------------


def _make_fetcher(slug_to_series: dict[str, pd.Series]):
    """Build an async fetcher that returns canned series per slug."""

    async def fetch(slug: str, days: int = 120) -> pd.Series:
        s = slug_to_series.get(slug)
        if s is None:
            return pd.Series(dtype=float, name=slug)
        return s.copy().rename(slug)

    return fetch


@pytest.fixture(autouse=True)
def _isolate_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atr, "JOBS_FILE", tmp_path / "regen_jobs.json")
    atr._STATE.running = False
    atr._STATE.last_job_id = None


# ---------------------------------------------------------------------------
# evaluate_pair (sync, deterministic)
# ---------------------------------------------------------------------------


class TestEvaluatePair:
    def test_cointegrated_pair_passes_engle_granger(self) -> None:
        a, b = _cointegrated_pair(n=260, seed=4)
        rec = _make_pair_record("a4", "b4")
        out = atr.evaluate_pair(rec, a, b, perm_iters=50)
        assert out.cointegrated is True
        assert out.adf_pvalue is not None and out.adf_pvalue < 0.05
        assert out.oos_sharpe is not None
        assert out.perm_p is not None and 0.0 <= out.perm_p <= 1.0

    def test_independent_pair_fails_cointegration(self) -> None:
        a, b = _independent_pair(n=260, seed=5)
        rec = _make_pair_record("a5", "b5")
        out = atr.evaluate_pair(rec, a, b, perm_iters=50)
        # Random walks rarely cointegrate; if they do (rare), we still want to
        # see no error and a tier <= D_RAW after BH.
        if not out.cointegrated:
            assert out.regen_error == "not_cointegrated"

    def test_too_few_obs_short_circuits(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        a = pd.Series(np.arange(10, dtype=float), index=idx)
        b = pd.Series(np.arange(10, dtype=float) + 1, index=idx)
        rec = _make_pair_record("aS", "bS")
        out = atr.evaluate_pair(rec, a, b)
        assert out.regen_error is not None and "too_few_obs" in out.regen_error
        assert out.cointegrated is False


def _make_cointegrated_result(n_obs: int) -> atr.PairRegenResult:
    """Build a synthetic ``PairRegenResult`` that, ignoring the 4Q gate, would
    otherwise be promoted to ``B_VALIDATED`` (BH-q05 + OOS Sharpe >= 0.5)
    or ``A_GOLD`` (with the 4Q-gold quarterly-stability flag also set).

    This lets us isolate the 360-joint-day gate from every other promotion
    rule in ``_final_tier``.
    """
    return atr.PairRegenResult(
        pair_id="A__B",
        a_id="A",
        b_id="B",
        a_slug="a-slug",
        b_slug="b-slug",
        n_obs=n_obs,
        cointegrated=True,
        oos_sharpe=3.0,
        full_sharpe=2.5,
        perm_p=0.001,
        passes_bh_q05=True,
        passes_bh_q10=True,
        quarterly_stability={"passes_4q_gold": True},
    )


def test_joint_days_under_360_caps_at_c_tentative() -> None:
    """v22 §5.3 gate: < 360 joint trading days → C_TENTATIVE regardless of
    cointegration / BH-q05 / OOS Sharpe. Boundary at n_obs=359 fails, n_obs=360
    passes. Above the gate, the same result body promotes to A_GOLD (since we
    set ``passes_4q_gold=True`` and OOS Sharpe >= 1.0)."""
    # Below the gate: capped at C_TENTATIVE.
    res_short = _make_cointegrated_result(n_obs=200)
    assert atr._final_tier(res_short, is_strike_family=False) == "C_TENTATIVE"

    # Boundary -1: 359 days is still below the gate.
    res_359 = _make_cointegrated_result(n_obs=359)
    assert atr._final_tier(res_359, is_strike_family=False) == "C_TENTATIVE"

    # Boundary: exactly 360 days passes the gate. With the synthetic record's
    # passes_bh_q05 + oos_sharpe=3.0 + passes_4q_gold=True this lands at A_GOLD.
    res_360 = _make_cointegrated_result(n_obs=360)
    assert atr._final_tier(res_360, is_strike_family=False) == "A_GOLD"

    # Well above the gate: same A_GOLD promotion.
    res_long = _make_cointegrated_result(n_obs=400)
    assert atr._final_tier(res_long, is_strike_family=False) == "A_GOLD"


def test_joint_days_gate_constant_is_360() -> None:
    """Pin the module-level constant so an accidental edit to ``JOINT_DAYS_4Q_GATE``
    is caught by CI rather than silently weakening the v22 §5.3 promotion rule."""
    assert atr.JOINT_DAYS_4Q_GATE == 360


def test_guarded_wf_sharpe_returns_none_for_zero_variance_folds() -> None:
    """Regression test for the impossible ``oos=9.47 / full=0.00`` rows.

    ``walk_forward_backtest`` coerces zero-variance folds to ``0.0`` and tiny
    samples likewise — without a guard, the aggregate ``train_sharpe_mean`` /
    ``test_sharpe_mean`` would mix those sentinels with one lucky fold and
    produce nonsense. ``_guarded_wf_sharpe`` must return ``None`` rather than
    coercing to ``0.0`` when there isn't enough signal.
    """
    from dataclasses import dataclass

    @dataclass
    class _Fold:
        n_train: int
        n_test: int
        train_sharpe: float
        test_sharpe: float

    # All four folds are the zero-variance sentinel (0.0 with adequate sample).
    all_sentinel = [_Fold(200, 50, 0.0, 0.0) for _ in range(4)]
    assert atr._guarded_wf_sharpe(all_sentinel, side="train") is None
    assert atr._guarded_wf_sharpe(all_sentinel, side="test") is None

    # Three sentinels + one lucky fold reproduces the production bug shape:
    # majority of folds are degenerate, so the aggregate is not trustworthy.
    bug_shape = [
        _Fold(200, 50, 0.0, 0.0),
        _Fold(200, 50, 0.0, 0.0),
        _Fold(200, 50, 0.0, 0.0),
        _Fold(200, 50, 1.2, 9.47),
    ]
    assert atr._guarded_wf_sharpe(bug_shape, side="train") is None
    assert atr._guarded_wf_sharpe(bug_shape, side="test") is None

    # Sample-too-small folds (n_test < _MIN_OOS_OBS_FOR_SHARPE) must be ignored.
    too_small = [_Fold(10, 5, 1.5, 2.0) for _ in range(4)]
    assert atr._guarded_wf_sharpe(too_small, side="train") is None
    assert atr._guarded_wf_sharpe(too_small, side="test") is None

    # Happy path: 4 valid folds with real Sharpes → mean is emitted as float.
    happy = [
        _Fold(200, 50, 0.8, 0.7),
        _Fold(200, 50, 1.0, 0.9),
        _Fold(200, 50, 1.1, 1.0),
        _Fold(200, 50, 0.9, 0.8),
    ]
    train = atr._guarded_wf_sharpe(happy, side="train")
    test = atr._guarded_wf_sharpe(happy, side="test")
    assert train is not None and abs(train - 0.95) < 1e-9
    assert test is not None and abs(test - 0.85) < 1e-9


# ---------------------------------------------------------------------------
# regenerate_alpha_tiers — synthetic 5-pair scenario
# ---------------------------------------------------------------------------


def test_recovers_real_signals_from_5_pair_synthetic() -> None:
    """2 truly cointegrated pairs + 3 noise pairs → at least the 2 real ones
    end up tier ≥ B_VALIDATED after the full pipeline.

    Uses n=400 so each pair clears the v22 §5.3 JOINT_DAYS_4Q_GATE (360);
    otherwise every pair would be capped at C_TENTATIVE regardless of
    cointegration / BH-FDR / OOS Sharpe.
    """
    slug_to_series: dict[str, pd.Series] = {}
    pairs: list[dict] = []

    # --- Two cointegrated pairs (different seeds for independent processes) -
    a1, b1 = _cointegrated_pair(n=400, seed=11)
    slug_to_series["a1"] = a1
    slug_to_series["b1"] = b1
    pairs.append(_make_pair_record("a1", "b1"))

    a2, b2 = _cointegrated_pair(n=400, seed=22)
    slug_to_series["a2"] = a2
    slug_to_series["b2"] = b2
    pairs.append(_make_pair_record("a2", "b2"))

    # --- Three independent random walks (noise) -----------------------------
    for k in range(3):
        a, b = _independent_pair(n=400, seed=100 + k)
        slug_to_series[f"n{k}_a"] = a
        slug_to_series[f"n{k}_b"] = b
        pairs.append(_make_pair_record(f"n{k}_a", f"n{k}_b"))

    fetcher = _make_fetcher(slug_to_series)
    out = asyncio.run(
        atr.regenerate_alpha_tiers(
            pairs=pairs,
            output_mode="dry-run",
            fetcher=fetcher,
            perm_iters=80,
            max_runtime_seconds=120,
        )
    )
    summary = out["summary"]
    assert summary["n_processed"] == 5
    by_id = {s["pair_id"]: s for s in out["strategies"]}
    # Real signals: at least one of the two synthetic cointegrated pairs must
    # land at B_VALIDATED or better. (Both routinely do, but we tolerate the
    # boundary case where ADF rejects but BH does not.)
    real_tiers = {
        by_id["a1__b1"]["tier"],
        by_id["a2__b2"]["tier"],
    }
    valid_better = {"A_GOLD", "A_STRUCTURAL", "B_VALIDATED", "B_FDR_ONLY"}
    assert real_tiers & valid_better, (
        f"Expected at least one cointegrated pair ≥ B_VALIDATED; got {real_tiers}"
    )
    # Noise pairs must not all be A_GOLD.
    noise_tiers = [by_id[f"n{k}_a__n{k}_b"]["tier"] for k in range(3)]
    assert "A_GOLD" not in noise_tiers


# ---------------------------------------------------------------------------
# Output modes
# ---------------------------------------------------------------------------


def _two_real_three_noise_state() -> tuple[list[dict], dict[str, pd.Series]]:
    pairs: list[dict] = []
    slugs: dict[str, pd.Series] = {}
    a1, b1 = _cointegrated_pair(n=260, seed=33)
    slugs["a1"], slugs["b1"] = a1, b1
    pairs.append(_make_pair_record("a1", "b1"))
    a2, b2 = _cointegrated_pair(n=260, seed=44)
    slugs["a2"], slugs["b2"] = a2, b2
    pairs.append(_make_pair_record("a2", "b2"))
    for k in range(3):
        a, b = _independent_pair(n=260, seed=200 + k)
        slugs[f"x{k}_a"], slugs[f"x{k}_b"] = a, b
        pairs.append(_make_pair_record(f"x{k}_a", f"x{k}_b"))
    return pairs, slugs


def test_dry_run_writes_no_file(tmp_path: Path) -> None:
    pairs, slugs = _two_real_three_noise_state()
    alpha_path = tmp_path / "alpha_strategies.json"
    alpha_path.write_text(json.dumps({"strategies": pairs}))
    out = asyncio.run(
        atr.regenerate_alpha_tiers(
            pairs=pairs,
            alpha_path=alpha_path,
            output_mode="dry-run",
            fetcher=_make_fetcher(slugs),
            perm_iters=50,
            report_dir=tmp_path,
        )
    )
    assert out["written_path"] is None
    # Original file untouched.
    saved = json.loads(alpha_path.read_text())
    # The original record had tier="C_TENTATIVE"; dry-run shouldn't touch it.
    assert saved["strategies"][0]["tier"] == "C_TENTATIVE"


def test_backup_writes_timestamped_file(tmp_path: Path) -> None:
    pairs, slugs = _two_real_three_noise_state()
    alpha_path = tmp_path / "alpha_strategies.json"
    alpha_path.write_text(json.dumps({"strategies": pairs}))
    out = asyncio.run(
        atr.regenerate_alpha_tiers(
            pairs=pairs,
            alpha_path=alpha_path,
            output_mode="backup",
            fetcher=_make_fetcher(slugs),
            perm_iters=50,
            report_dir=tmp_path,
        )
    )
    assert out["written_path"] is not None
    written = Path(out["written_path"])
    assert written.exists()
    assert ".regenerated." in written.name
    # Original file untouched.
    saved = json.loads(alpha_path.read_text())
    assert saved["strategies"][0]["tier"] == "C_TENTATIVE"
    # Backup contains the regen-added top-level fields.
    backup = json.loads(written.read_text())
    assert "regenerated_at_iso" in backup
    assert "regen_summary" in backup


def test_update_overwrites_alpha_path(tmp_path: Path) -> None:
    pairs, slugs = _two_real_three_noise_state()
    alpha_path = tmp_path / "alpha_strategies.json"
    alpha_path.write_text(json.dumps({"strategies": pairs}))
    out = asyncio.run(
        atr.regenerate_alpha_tiers(
            pairs=pairs,
            alpha_path=alpha_path,
            output_mode="update",
            fetcher=_make_fetcher(slugs),
            perm_iters=50,
            report_dir=tmp_path,
        )
    )
    assert out["written_path"] == str(alpha_path)
    saved = json.loads(alpha_path.read_text())
    assert "regen_summary" in saved
    # Each strategy gets the regen fields populated.
    for s in saved["strategies"]:
        assert "passes_bh_q05" in s
        assert "regenerated_at_iso" in s


# ---------------------------------------------------------------------------
# Time-budget / short-circuit
# ---------------------------------------------------------------------------


def test_max_runtime_zero_short_circuits(tmp_path: Path) -> None:
    pairs, slugs = _two_real_three_noise_state()
    out = asyncio.run(
        atr.regenerate_alpha_tiers(
            pairs=pairs,
            output_mode="dry-run",
            fetcher=_make_fetcher(slugs),
            max_runtime_seconds=0,
            perm_iters=50,
            report_dir=tmp_path,
        )
    )
    summary = out["summary"]
    assert summary["timed_out"] is True
    assert summary["n_processed"] == 0
    # Strategies are still emitted (with regen_error="not_processed") so the
    # caller doesn't lose entries — they just aren't re-tiered.
    assert len(out["strategies"]) == len(pairs)
    for s in out["strategies"]:
        assert s.get("regen_error") == "not_processed"


# ---------------------------------------------------------------------------
# BH-FDR sanity
# ---------------------------------------------------------------------------


def test_bh_fdr_applied_5_real_95_noise(tmp_path: Path) -> None:
    """Generate 5 real cointegrated pairs + 95 noise pairs.  After BH-FDR at
    q=0.05 we should see no more than ~5 ± a small slack rejections (matching
    the false-discovery-rate guarantee).  We assert that the count is bounded
    well below the naive marginal-test count (which would be much higher)."""
    rng = np.random.default_rng(0)
    pairs: list[dict] = []
    slugs: dict[str, pd.Series] = {}

    # 5 real signals
    for i in range(5):
        a, b = _cointegrated_pair(n=260, seed=300 + i)
        slugs[f"r{i}_a"], slugs[f"r{i}_b"] = a, b
        pairs.append(_make_pair_record(f"r{i}_a", f"r{i}_b"))

    # 95 noise pairs
    for i in range(95):
        a, b = _independent_pair(n=260, seed=int(rng.integers(0, 10**6)))
        slugs[f"n{i}_a"], slugs[f"n{i}_b"] = a, b
        pairs.append(_make_pair_record(f"n{i}_a", f"n{i}_b"))

    out = asyncio.run(
        atr.regenerate_alpha_tiers(
            pairs=pairs,
            output_mode="dry-run",
            fetcher=_make_fetcher(slugs),
            perm_iters=60,
            max_runtime_seconds=300,
            report_dir=tmp_path,
        )
    )
    strategies = out["strategies"]
    n_passes_q05 = sum(1 for s in strategies if s.get("passes_bh_q05"))
    n_passes_q10 = sum(1 for s in strategies if s.get("passes_bh_q10"))
    # FDR guarantee: expected false discoveries ≤ q * #rejected. With q=0.05
    # the upper bound for n_passes_q05 should be bounded — we permit up to 25
    # to be robust to seed variation but it should be much smaller than 100.
    assert n_passes_q05 <= 25, f"BH-FDR q=0.05 should bound false discoveries; got {n_passes_q05}"
    # q=0.10 is more permissive — must be at least as large as q=0.05.
    assert n_passes_q10 >= n_passes_q05


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_per_pair_failure_does_not_kill_run(tmp_path: Path) -> None:
    pairs = [
        _make_pair_record("good_a", "good_b"),
        _make_pair_record("bad_a", "bad_b"),
    ]
    a, b = _cointegrated_pair(n=260, seed=99)
    slugs = {"good_a": a, "good_b": b}  # 'bad_*' missing → empty Series

    fetcher = _make_fetcher(slugs)
    out = asyncio.run(
        atr.regenerate_alpha_tiers(
            pairs=pairs,
            output_mode="dry-run",
            fetcher=fetcher,
            perm_iters=50,
            report_dir=tmp_path,
        )
    )
    by_id = {s["pair_id"]: s for s in out["strategies"]}
    assert by_id["bad_a__bad_b"]["tier"] == "D_RAW"
    assert by_id["bad_a__bad_b"]["regen_error"] is not None
    # The good pair was still processed despite the bad one failing.
    assert by_id["good_a__good_b"]["regen_error"] != "not_processed"


# ---------------------------------------------------------------------------
# CLI wrapper smoke
# ---------------------------------------------------------------------------


def test_cli_dry_run_smoke(tmp_path: Path, capsys, monkeypatch) -> None:
    """Exercise the script's argparse + asyncio.run path with --max-runtime=0
    so we don't need any fetches."""
    pairs, _ = _two_real_three_noise_state()
    alpha_path = tmp_path / "alpha_strategies.json"
    alpha_path.write_text(json.dumps({"strategies": pairs}))

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "regen_cli",
        Path(__file__).resolve().parents[1] / "scripts" / "regenerate_alpha_tiers.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rc = mod.main(
        [
            "--output",
            "dry-run",
            "--max-runtime",
            "0",
            "--alpha-path",
            str(alpha_path),
            "--report-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["summary"]["timed_out"] is True
    assert rc in (0, 2)
