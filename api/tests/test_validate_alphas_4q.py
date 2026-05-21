"""Synthetic-DGP tests for ``scripts/validate_alphas_4q.py``.

We construct a mocked ``alpha_strategies.json`` with three strategies whose
synthetic PnL paths are deterministic and known:

* ``always_pos`` — every quarter has +25 bps daily drift, no sign flip.
* ``always_pos_b`` — same as above, a second clean PASS.
* ``q3_flip`` — Q1/Q2/Q4 are strongly positive, Q3 is strongly negative.
  The full sample is net positive, so Q3 is a sign-flip and the strategy
  must end up in the proposals file.

The script lives outside the ``pfm`` package, so we import it via
:func:`importlib.util.spec_from_file_location`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Ensure ``src/`` is importable.
_API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_API_ROOT / "src"))

from pfm.strategies_registry import (
    Strategy,
    register,
    unregister,
)


@pytest.fixture()
def validate_module() -> ModuleType:
    """Import ``scripts/validate_alphas_4q.py`` as a module."""
    path = _API_ROOT / "scripts" / "validate_alphas_4q.py"
    spec = importlib.util.spec_from_file_location("pfm_validate_alphas_4q", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Mock strategies — registered globally so resolve_strategy() prefers them
# ---------------------------------------------------------------------------


def _make_positive_pnl(prices: pd.DataFrame, position: pd.Series) -> pd.Series:
    """Every day: +25 bps daily mean, ~40 bps daily vol. Reproducibly seeded."""
    n = len(prices)
    rng = np.random.default_rng(seed=42)
    pnl = rng.normal(loc=0.0025, scale=0.004, size=n)
    return pd.Series(pnl, index=prices.index, name="pnl")


def _make_q3_flip_pnl(prices: pd.DataFrame, position: pd.Series) -> pd.Series:
    """Q1/Q2/Q4 strongly positive, Q3 strongly negative — guaranteed sign flip."""
    rng = np.random.default_rng(seed=7)
    idx = prices.index
    means = np.full(len(idx), 0.003)  # +30 bps default
    # Quarter boundaries: idx[0] is 2024-01-01. Q3 = months 6..9 from start.
    start = idx[0]
    q3_start = start + pd.DateOffset(months=6)
    q3_end = start + pd.DateOffset(months=9)
    mask = (idx >= q3_start) & (idx < q3_end)
    means[mask] = -0.003  # -30 bps in Q3
    noise = rng.normal(loc=0.0, scale=0.0035, size=len(idx))
    return pd.Series(means + noise, index=idx, name="pnl")


def _identity_signal(prices: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=prices.index, name="signal")


@pytest.fixture()
def registered_strategies():
    """Register three strategies; unregister after the test."""
    names = ["always_pos", "always_pos_b", "q3_flip"]
    register(Strategy(name="always_pos", signal=_identity_signal, pnl=_make_positive_pnl))
    register(Strategy(name="always_pos_b", signal=_identity_signal, pnl=_make_positive_pnl))
    register(Strategy(name="q3_flip", signal=_identity_signal, pnl=_make_q3_flip_pnl))
    yield names
    for n in names:
        unregister(n)


@pytest.fixture()
def mock_strategies_file(tmp_path: Path) -> Path:
    """Write a synthetic ``alpha_strategies.json`` with three deployable rows."""
    payload = {
        "generated": "2026-05-16T00:00:00Z",
        "tier_legend": {
            "A_STRUCTURAL": "structural validated",
            "B_VALIDATED": "validated",
            "C_TENTATIVE": "tentative",
            "D_RAW": "raw",
        },
        "strategies": [
            {
                "pair_id": "always_pos",
                "tier": "A_STRUCTURAL",
                "category": "test",
            },
            {
                "pair_id": "always_pos_b",
                "tier": "B_VALIDATED",
                "category": "test",
            },
            {
                "pair_id": "q3_flip",
                "tier": "A_STRUCTURAL",
                "category": "test",
            },
            # An off-tier strategy that must be filtered out by run_validation.
            {
                "pair_id": "not_deployable",
                "tier": "C_TENTATIVE",
                "category": "test",
            },
        ],
    }
    path = tmp_path / "alpha_strategies.json"
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_filter_deployable_keeps_only_a_and_b(validate_module: ModuleType) -> None:
    raw = [
        {"pair_id": "a", "tier": "A_GOLD"},
        {"pair_id": "b", "tier": "A_STRUCTURAL"},
        {"pair_id": "c", "tier": "B_VALIDATED"},
        {"pair_id": "d", "tier": "C_TENTATIVE"},
        {"pair_id": "e", "tier": "D_RAW"},
        {"pair_id": "f", "tier": None},
    ]
    out = validate_module.filter_deployable(raw)
    assert [s["pair_id"] for s in out] == ["a", "b", "c"]


def test_load_strategies_list_layout(tmp_path: Path, validate_module: ModuleType) -> None:
    path = tmp_path / "alpha.json"
    path.write_text(json.dumps([{"pair_id": "x", "tier": "A_GOLD"}]))
    out = validate_module.load_strategies_file(path)
    assert out == [{"pair_id": "x", "tier": "A_GOLD"}]


def test_run_validation_flags_sign_flip(
    mock_strategies_file: Path,
    registered_strategies: list[str],
    validate_module: ModuleType,
) -> None:
    """Q3-flip strategy must end up in the failed/proposals list."""
    rows, failed = validate_module.run_validation(
        mock_strategies_file, start="2024-01", quarters=4, sharpe_floor=0.5
    )
    # Exactly three deployable strategies; C_TENTATIVE was filtered out.
    pair_ids = sorted(r["pair_id"] for r in rows)
    assert pair_ids == ["always_pos", "always_pos_b", "q3_flip"], pair_ids

    # q3_flip must be in the failed list.
    failed_ids = [r["pair_id"] for r in failed]
    assert "q3_flip" in failed_ids, failed_ids

    # And it must have the sign_flip flag set.
    q3_row = next(r for r in rows if r["pair_id"] == "q3_flip")
    assert q3_row["sign_flip"] is True
    assert q3_row["verdict"] == "FAIL"

    # The two clean strategies should pass and NOT be in failed.
    for clean in ("always_pos", "always_pos_b"):
        assert clean not in failed_ids
        row = next(r for r in rows if r["pair_id"] == clean)
        assert row["verdict"] == "PASS", row


def test_main_strict_exit_code(
    mock_strategies_file: Path,
    registered_strategies: list[str],
    validate_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--strict must exit non-zero when any strategy fails."""
    out_path = tmp_path / "proposals.json"
    argv = [
        "--strategies",
        str(mock_strategies_file),
        "--output",
        str(out_path),
        "--strict",
    ]
    code = validate_module.main(argv)
    assert code == 1, "expected non-zero exit when --strict and a strategy fails"

    # Proposals file must exist and include q3_flip.
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["count"] >= 1
    proposed_ids = [p["pair_id"] for p in payload["proposals"]]
    assert "q3_flip" in proposed_ids, proposed_ids

    # Summary table should have been printed.
    captured = capsys.readouterr()
    assert "q3_flip" in captured.out
    assert "FAIL" in captured.out


def test_main_non_strict_exit_zero(
    mock_strategies_file: Path,
    registered_strategies: list[str],
    validate_module: ModuleType,
    tmp_path: Path,
) -> None:
    """Without --strict the script still exits 0 even with failures."""
    out_path = tmp_path / "proposals.json"
    argv = [
        "--strategies",
        str(mock_strategies_file),
        "--output",
        str(out_path),
    ]
    code = validate_module.main(argv)
    assert code == 0


def test_main_does_not_modify_alpha_strategies_file(
    mock_strategies_file: Path,
    registered_strategies: list[str],
    validate_module: ModuleType,
    tmp_path: Path,
) -> None:
    """The input alpha_strategies.json must NEVER be auto-mutated."""
    before = mock_strategies_file.read_bytes()
    out_path = tmp_path / "proposals.json"
    validate_module.main(
        [
            "--strategies",
            str(mock_strategies_file),
            "--output",
            str(out_path),
            "--strict",
        ]
    )
    after = mock_strategies_file.read_bytes()
    assert before == after, "validate_alphas_4q must not edit alpha_strategies.json"


def test_aggregate_row_extracts_quarter_sharpes(
    validate_module: ModuleType,
) -> None:
    fake_report: dict[str, Any] = {
        "strategy": "x",
        "full_sample": {"sharpe": 1.23, "sign": 1, "deflated_sharpe": 0.4},
        "quarter_rows": [
            {"quarter": 1, "sharpe": 0.7, "sign": 1, "fail": False, "fail_reason": ""},
            {"quarter": 2, "sharpe": 0.8, "sign": 1, "fail": False, "fail_reason": ""},
            {"quarter": 3, "sharpe": -0.2, "sign": -1, "fail": True, "fail_reason": "flip"},
            {"quarter": 4, "sharpe": 0.6, "sign": 1, "fail": False, "fail_reason": ""},
        ],
        "verdict": "FAIL",
    }
    row = validate_module.aggregate_row({"pair_id": "x", "tier": "A_GOLD"}, fake_report)
    assert row["q1_sharpe"] == 0.7
    assert row["q3_sharpe"] == -0.2
    assert row["sign_flip"] is True
    assert row["verdict"] == "FAIL"
    assert row["tier"] == "A_GOLD"
