"""Tests for ``scripts/generate_backtest_comparison.py``.

Covers:
  1. Pure metric helpers (Sharpe, deflated Sharpe, max-drawdown, win-rate).
  2. Deterministic seeding from pair_id (same id → same path).
  3. Deployable-tier filtering (only A_STRUCTURAL / A_GOLD / B_VALIDATED).
  4. Per-strategy backtest output shape.
  5. Full ``build_comparison`` payload shape + invariants.
  6. Fixture override path (caller-supplied PnL).
  7. Edge cases: empty inputs and unfilled tiers.
  8. CLI ``main`` end-to-end with a tmp_path input and output.
  9. Aggregate-comparison best/worst routing.
 10. JSON round-trip — output is serializable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``api/scripts`` importable without modifying sys.path globally.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import generate_backtest_comparison as gbc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_strategies() -> list[dict]:
    return [
        {"pair_id": "alpha_a", "tier": "A_GOLD", "oos_sharpe": 2.0, "n_obs": 120},
        {"pair_id": "alpha_b", "tier": "A_STRUCTURAL", "oos_sharpe": 1.0, "n_obs": 90},
        {"pair_id": "alpha_c", "tier": "B_VALIDATED", "oos_sharpe": 0.5, "n_obs": 60},
        {"pair_id": "watchlist_d", "tier": "C_TENTATIVE", "oos_sharpe": 0.3, "n_obs": 60},
        {"pair_id": "noise_e", "tier": "D_RAW", "oos_sharpe": 0.1, "n_obs": 40},
    ]


# ---------------------------------------------------------------------------
# 1. Pure helpers
# ---------------------------------------------------------------------------


def test_sharpe_ratio_recovers_target() -> None:
    """A long enough simulated path should recover the target Sharpe to within a generous tol.

    Sharpe estimators have standard error ≈ √((1+0.5·SR²)/N). With N=25k and
    SR=1.5 that's ~0.01, so ±0.2 is a comfortable upper bound.
    """
    returns = gbc.simulate_daily_returns(sharpe=1.5, n_days=25200, daily_vol=0.01, seed=42)
    sr = gbc.sharpe_ratio(returns)
    assert abs(sr - 1.5) < 0.2


def test_sharpe_ratio_handles_constant_series() -> None:
    constant = np.zeros(50)
    assert gbc.sharpe_ratio(constant) == 0.0
    too_short = np.array([0.01])
    assert gbc.sharpe_ratio(too_short) == 0.0


def test_max_drawdown_is_non_positive_and_correct() -> None:
    # Equity: 1 → 1.2 → 0.9 → 1.05 — peak 1.2, trough 0.9 → DD = (0.9-1.2)/1.2 = -0.25
    dd = gbc.max_drawdown([1.0, 1.2, 0.9, 1.05])
    assert dd == pytest.approx(-0.25, abs=1e-9)
    # Monotonically increasing: no drawdown
    assert gbc.max_drawdown([1.0, 1.1, 1.2]) == 0.0
    # Empty: safe zero
    assert gbc.max_drawdown([]) == 0.0


def test_win_rate_and_n_trades() -> None:
    r = np.array([0.01, -0.02, 0.0, 0.005, -0.001])
    assert gbc.win_rate(r) == pytest.approx(2 / 5)
    # n_trades = days where |r| > 0
    assert gbc.count_trades(r, threshold=0.0) == 4
    assert gbc.count_trades(np.array([]), threshold=0.0) == 0


def test_deflated_sharpe_shrinks_with_multiple_trials() -> None:
    raw = 1.5
    n_obs = 250
    base = gbc.deflated_sharpe(raw, n_obs, n_trials=1)
    many = gbc.deflated_sharpe(raw, n_obs, n_trials=1000)
    # No trials penalty → returns approximately raw Sharpe
    assert base == pytest.approx(raw, abs=1e-9)
    # With trials penalty → strictly smaller (haircut)
    assert many < base
    # Edge cases: n_obs <= 1 → 0
    assert gbc.deflated_sharpe(1.0, n_obs=0) == 0.0


# ---------------------------------------------------------------------------
# 2. Deterministic seeding
# ---------------------------------------------------------------------------


def test_seed_is_deterministic_per_pair_id() -> None:
    s1 = gbc._seed_for_pair("alpha_a")
    s2 = gbc._seed_for_pair("alpha_a")
    s3 = gbc._seed_for_pair("alpha_b")
    assert s1 == s2
    assert s1 != s3
    # Re-running simulate with same seed → bit-identical path
    a = gbc.simulate_daily_returns(1.0, n_days=30, seed=s1)
    b = gbc.simulate_daily_returns(1.0, n_days=30, seed=s2)
    assert np.array_equal(a, b)


def test_simulate_validates_inputs() -> None:
    with pytest.raises(ValueError):
        gbc.simulate_daily_returns(1.0, n_days=0)
    with pytest.raises(ValueError):
        gbc.simulate_daily_returns(1.0, n_days=10, daily_vol=0.0)


# ---------------------------------------------------------------------------
# 3. Deployable filtering
# ---------------------------------------------------------------------------


def test_filter_deployable_excludes_c_and_d(sample_strategies: list[dict]) -> None:
    out = gbc.filter_deployable(sample_strategies)
    tiers = {s["tier"] for s in out}
    assert tiers == {"A_GOLD", "A_STRUCTURAL", "B_VALIDATED"}
    assert all(s["pair_id"] in {"alpha_a", "alpha_b", "alpha_c"} for s in out)


# ---------------------------------------------------------------------------
# 4. Per-strategy backtest output
# ---------------------------------------------------------------------------


def test_backtest_strategy_shape() -> None:
    strat = {"pair_id": "alpha_a", "tier": "A_GOLD", "oos_sharpe": 1.5, "n_obs": 120}
    row = gbc.backtest_strategy(strat, n_days=60)
    assert row["pair_id"] == "alpha_a"
    assert row["tier"] == "A_GOLD"
    assert len(row["equity_curve"]) == 60
    m = row["metrics"]
    for key in (
        "sharpe",
        "deflated_sharpe",
        "max_drawdown",
        "win_rate",
        "n_trades",
        "total_return",
        "annual_vol",
        "target_sharpe",
    ):
        assert key in m
    # Equity curve starts at simulated NAV[0] (≈1 + r_0); test it's near 1.0
    assert 0.5 < row["equity_curve"][0] < 1.5


# ---------------------------------------------------------------------------
# 5. Full build_comparison shape + invariants
# ---------------------------------------------------------------------------


def test_build_comparison_payload_shape(sample_strategies: list[dict]) -> None:
    payload = gbc.build_comparison(
        sample_strategies, n_days=30, generated_at="2026-05-16T00:00:00Z"
    )
    # Top-level keys
    assert set(payload.keys()) >= {
        "generated_at",
        "n_strategies",
        "n_days",
        "deployable_tiers",
        "strategies",
        "comparison",
    }
    assert payload["n_strategies"] == 3
    assert payload["n_days"] == 30
    # Sorted descending by Sharpe
    sharpes = [r["metrics"]["sharpe"] for r in payload["strategies"]]
    assert sharpes == sorted(sharpes, reverse=True)
    # Comparison has the required best/worst sub-keys
    cmp_keys = set(payload["comparison"].keys())
    assert {
        "best_sharpe",
        "best_deflated_sharpe",
        "best_max_drawdown",
        "worst_max_drawdown",
        "best_win_rate",
        "best_total_return",
    } <= cmp_keys
    # Each best/worst entry points to a real strategy
    ids_in_output = {r["pair_id"] for r in payload["strategies"]}
    for k, v in payload["comparison"].items():
        assert v["pair_id"] in ids_in_output, f"{k} points to unknown pair_id"


# ---------------------------------------------------------------------------
# 6. Fixture-override path
# ---------------------------------------------------------------------------


def test_fixtures_override_simulation() -> None:
    strat = {"pair_id": "alpha_fixture", "tier": "A_GOLD", "oos_sharpe": 99.0, "n_obs": 5}
    fixture = [0.01, -0.005, 0.002, 0.0, 0.003]
    row = gbc.backtest_strategy(strat, fixture_returns=fixture)
    # Equity curve length matches fixture length (NOT n_days simulated)
    assert len(row["equity_curve"]) == 5
    # Manually compute expected final NAV from compounding the fixture
    expected_final = 1.0
    for r in fixture:
        expected_final *= 1.0 + r
    assert row["equity_curve"][-1] == pytest.approx(expected_final, rel=1e-5)
    # Sharpe of fixture, not the bogus target
    assert row["metrics"]["sharpe"] != 99.0


# ---------------------------------------------------------------------------
# 7. Empty / unfilled tier edge cases
# ---------------------------------------------------------------------------


def test_build_comparison_with_no_deployable_strategies() -> None:
    only_raw = [{"pair_id": "x", "tier": "D_RAW", "oos_sharpe": 0.2, "n_obs": 30}]
    payload = gbc.build_comparison(only_raw, n_days=10)
    assert payload["n_strategies"] == 0
    assert payload["strategies"] == []
    assert payload["comparison"] == {}


def test_aggregate_comparison_routes_best_and_worst_correctly() -> None:
    rows = [
        {
            "pair_id": "A",
            "tier": "A_GOLD",
            "metrics": {
                "sharpe": 0.5,
                "deflated_sharpe": 0.4,
                "max_drawdown": -0.20,
                "win_rate": 0.55,
                "total_return": 0.10,
            },
            "equity_curve": [1.0],
        },
        {
            "pair_id": "B",
            "tier": "A_STRUCTURAL",
            "metrics": {
                "sharpe": 1.2,
                "deflated_sharpe": 0.9,
                "max_drawdown": -0.05,  # smallest drawdown
                "win_rate": 0.50,
                "total_return": 0.05,
            },
            "equity_curve": [1.0],
        },
    ]
    cmp = gbc.aggregate_comparison(rows)
    assert cmp["best_sharpe"]["pair_id"] == "B"
    assert cmp["best_deflated_sharpe"]["pair_id"] == "B"
    # best (closest-to-zero) drawdown: -0.05 > -0.20
    assert cmp["best_max_drawdown"]["pair_id"] == "B"
    # worst drawdown is the most negative
    assert cmp["worst_max_drawdown"]["pair_id"] == "A"
    assert cmp["best_win_rate"]["pair_id"] == "A"
    assert cmp["best_total_return"]["pair_id"] == "A"


# ---------------------------------------------------------------------------
# 8. CLI end-to-end
# ---------------------------------------------------------------------------


def test_cli_main_end_to_end(tmp_path: Path, sample_strategies: list[dict]) -> None:
    in_path = tmp_path / "alpha_strategies.json"
    in_path.write_text(json.dumps({"strategies": sample_strategies}))
    out_path = tmp_path / "backtest_comparison.json"

    rc = gbc.main(["--input", str(in_path), "--out", str(out_path), "--n-days", "20"])
    assert rc == 0
    assert out_path.exists()

    parsed = json.loads(out_path.read_text())
    assert parsed["n_strategies"] == 3
    assert parsed["n_days"] == 20
    assert all(len(r["equity_curve"]) == 20 for r in parsed["strategies"])
    # generated_at is an ISO Z-suffixed timestamp
    assert parsed["generated_at"].endswith("Z")


def test_cli_main_with_fixtures_file(tmp_path: Path, sample_strategies: list[dict]) -> None:
    in_path = tmp_path / "alpha_strategies.json"
    in_path.write_text(json.dumps({"strategies": sample_strategies}))
    fx_path = tmp_path / "fixtures.json"
    fx_path.write_text(
        json.dumps(
            {
                "alpha_a": [0.01, 0.02, -0.005, 0.0, 0.001, 0.003, -0.002],
                # alpha_b/alpha_c left unfixtured → simulation
            }
        )
    )
    out_path = tmp_path / "out.json"
    rc = gbc.main(
        [
            "--input",
            str(in_path),
            "--out",
            str(out_path),
            "--n-days",
            "40",
            "--fixtures",
            str(fx_path),
        ]
    )
    assert rc == 0
    parsed = json.loads(out_path.read_text())
    by_id = {r["pair_id"]: r for r in parsed["strategies"]}
    # alpha_a used the 7-entry fixture
    assert len(by_id["alpha_a"]["equity_curve"]) == 7
    # alpha_b/alpha_c fell back to simulation @ n_days=40
    assert len(by_id["alpha_b"]["equity_curve"]) == 40


# ---------------------------------------------------------------------------
# 9. Output is JSON-serialisable (no numpy types leaking)
# ---------------------------------------------------------------------------


def test_payload_is_json_round_trippable(sample_strategies: list[dict]) -> None:
    payload = gbc.build_comparison(sample_strategies, n_days=25)
    s = json.dumps(payload)  # would raise on numpy.float64 etc.
    re = json.loads(s)
    assert re["n_days"] == 25


# ---------------------------------------------------------------------------
# 10. Determinism across runs with same seed
# ---------------------------------------------------------------------------


def test_build_comparison_is_deterministic(sample_strategies: list[dict]) -> None:
    a = gbc.build_comparison(sample_strategies, n_days=30, generated_at="2026-05-16T00:00:00Z")
    b = gbc.build_comparison(sample_strategies, n_days=30, generated_at="2026-05-16T00:00:00Z")
    # Drop generated_at (would already match by construction)
    assert a == b
