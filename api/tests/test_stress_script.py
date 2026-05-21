"""Synthetic-DGP tests for ``scripts/stress_test.py``.

The harness must correctly flag any quarter whose Sharpe falls below the
configured floor OR whose sign of mean PnL flips relative to the full
sample. We construct a deterministic mock strategy with a known seasonal
alpha schedule (3 positive quarters + 1 negative quarter) and assert the
script's verdict is FAIL with the sign-flip quarter identified.

The script lives under ``scripts/`` (outside the ``pfm`` package), so we
import it via :mod:`importlib` from its file path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

# Make sure ``src/`` is importable (mirrors the script's own bootstrap).
_API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_API_ROOT / "src"))

from pfm.strategies_registry import (
    Strategy,
    get,
    register,
    unregister,
)


@pytest.fixture()
def stress_module() -> ModuleType:
    """Import ``scripts/stress_test.py`` as a module."""
    path = _API_ROOT / "scripts" / "stress_test.py"
    spec = importlib.util.spec_from_file_location("pfm_stress_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Mock strategy: seasonal alpha with a flipped quarter
# ---------------------------------------------------------------------------


def _seasonal_pnl(prices: pd.DataFrame, position: pd.Series) -> pd.Series:
    """PnL fixture: strongly positive in Q1/Q2/Q4, strongly negative in Q3.

    The first three quarters carry a +25 bps daily mean; the third quarter
    carries -25 bps. With ~0.4% daily vol the per-quarter Sharpes are
    well above 0.5 in absolute value and the *signs* differ — so the
    full-sample sign should be positive (3 positive quarters dominate the
    one negative one) and Q3 should fail with a sign-flip.
    """
    idx = prices.index
    rng = np.random.default_rng(20260516)
    noise = rng.normal(0.0, 0.0040, size=len(idx))
    means = np.where(
        (idx.month >= 7) & (idx.month <= 9),  # Q3 = months 7,8,9
        -0.0025,
        0.0025,
    )
    return pd.Series(noise + means, index=idx, name="pnl")


def _always_long_signal(prices: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=prices.index, name="signal")


@pytest.fixture()
def seasonal_strategy() -> Strategy:
    strat = Strategy(
        name="__test_seasonal_flip__",
        signal=_always_long_signal,
        pnl=_seasonal_pnl,
    )
    register(strat)
    yield strat
    unregister(strat.name)


@pytest.fixture()
def all_positive_strategy() -> Strategy:
    """Strategy where every quarter has a clean +1 sign and decent Sharpe."""

    def pnl_fn(prices: pd.DataFrame, position: pd.Series) -> pd.Series:
        rng = np.random.default_rng(7)
        noise = rng.normal(0.0, 0.003, size=len(prices.index))
        return pd.Series(noise + 0.0025, index=prices.index, name="pnl")

    strat = Strategy(
        name="__test_all_positive__",
        signal=_always_long_signal,
        pnl=pnl_fn,
    )
    register(strat)
    yield strat
    unregister(strat.name)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_annualised_sharpe_zero_when_std_zero(self, stress_module: ModuleType) -> None:
        # Empty / single-obs / NaN-std all return 0.0 by guard.
        assert stress_module.annualised_sharpe(pd.Series([], dtype=float)) == 0.0
        assert stress_module.annualised_sharpe(pd.Series([0.01])) == 0.0
        # All-NaN std → guard kicks in
        assert stress_module.annualised_sharpe(pd.Series([float("nan")] * 5)) == 0.0

    def test_annualised_sharpe_positive_when_mean_positive(self, stress_module: ModuleType) -> None:
        rng = np.random.default_rng(42)
        pnl = pd.Series(rng.normal(0.001, 0.01, size=1000))
        s = stress_module.annualised_sharpe(pnl)
        assert s > 0.5

    def test_sortino_handles_no_downside(self, stress_module: ModuleType) -> None:
        pnl = pd.Series([0.01, 0.02, 0.03, 0.005])
        # All positive → no downside; Sortino guard returns 0.0
        assert stress_module.annualised_sortino(pnl) == 0.0

    def test_t_stat_returns_zero_one_for_empty(self, stress_module: ModuleType) -> None:
        assert stress_module.t_stat(pd.Series([], dtype=float)) == (0.0, 1.0)

    def test_mean_sign_zero_for_zero_mean(self, stress_module: ModuleType) -> None:
        assert stress_module.mean_sign(pd.Series([1.0, -1.0])) == 0


# ---------------------------------------------------------------------------
# Quarter-window slicing
# ---------------------------------------------------------------------------


class TestWindows:
    def test_parse_start_yyyy_mm(self, stress_module: ModuleType) -> None:
        ts = stress_module.parse_start("2024-04")
        assert ts.year == 2024 and ts.month == 4 and ts.day == 1
        assert str(ts.tz) == "UTC"

    def test_quarter_windows_disjoint(self, stress_module: ModuleType) -> None:
        from itertools import pairwise

        start = stress_module.parse_start("2024-01")
        wins = stress_module.quarter_windows(start, 4)
        assert len(wins) == 4
        # Disjoint + contiguous: end of window i equals start of window i+1
        for (_a_start, a_end), (b_start, _b_end) in pairwise(wins):
            assert a_end == b_start
        # Each window is exactly 3 months
        for qs, qe in wins:
            assert (qe - qs).days in (89, 90, 91, 92)


# ---------------------------------------------------------------------------
# End-to-end harness
# ---------------------------------------------------------------------------


class TestStressHarness:
    def test_seasonal_strategy_flags_sign_flip(
        self,
        stress_module: ModuleType,
        seasonal_strategy: Strategy,
    ) -> None:
        start = stress_module.parse_start("2024-01")
        report = stress_module.run_stress(seasonal_strategy, start=start, quarters=4)

        assert report["verdict"] == "FAIL"
        assert len(report["quarter_rows"]) == 4

        # Q3 corresponds to months 7-9 → must be the flipped, failing quarter.
        q3 = report["quarter_rows"][2]
        assert q3["fail"] is True, q3
        assert q3["sign"] == -1
        assert "sign flip" in q3["fail_reason"]

        # The other three quarters should pass (positive Sharpe, +1 sign).
        for i in (0, 1, 3):
            row = report["quarter_rows"][i]
            assert row["sign"] == 1, row
            assert row["fail"] is False, row

        # Full-sample sign is +1 (3 positive quarters outweigh 1 negative).
        assert report["full_sample"]["sign"] == 1

    def test_clean_strategy_passes_all_quarters(
        self,
        stress_module: ModuleType,
        all_positive_strategy: Strategy,
    ) -> None:
        start = stress_module.parse_start("2024-01")
        report = stress_module.run_stress(all_positive_strategy, start=start, quarters=4)
        assert report["verdict"] == "PASS"
        for row in report["quarter_rows"]:
            assert row["fail"] is False, row
            assert row["sign"] == 1
            assert row["sharpe"] >= 0.5

    def test_sharpe_floor_can_force_failure(
        self,
        stress_module: ModuleType,
        all_positive_strategy: Strategy,
    ) -> None:
        # Crank the floor sky-high — every quarter should fail.
        start = stress_module.parse_start("2024-01")
        report = stress_module.run_stress(
            all_positive_strategy, start=start, quarters=4, sharpe_floor=50.0
        )
        assert report["verdict"] == "FAIL"
        assert all(r["fail"] for r in report["quarter_rows"])

    def test_report_contains_deflated_sharpe(
        self,
        stress_module: ModuleType,
        seasonal_strategy: Strategy,
    ) -> None:
        start = stress_module.parse_start("2024-01")
        report = stress_module.run_stress(seasonal_strategy, start=start, quarters=4)
        assert "deflated_sharpe" in report["full_sample"]
        assert "deflated_p_value" in report["full_sample"]
        for row in report["quarter_rows"]:
            assert "deflated_sharpe" in row
            assert "expected_max_sharpe_under_null" in row

    def test_format_table_includes_verdict(
        self,
        stress_module: ModuleType,
        seasonal_strategy: Strategy,
    ) -> None:
        start = stress_module.parse_start("2024-01")
        report = stress_module.run_stress(seasonal_strategy, start=start, quarters=4)
        text = stress_module.format_table(report)
        assert "VERDICT: FAIL" in text
        assert "Sharpe" in text and "Sortino" in text


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------


class TestCLI:
    def test_unknown_strategy_returns_exit_2(
        self, stress_module: ModuleType, tmp_path: Path
    ) -> None:
        out = tmp_path / "x.json"
        rc = stress_module.main(
            [
                "--strategy",
                "__definitely_not_registered__",
                "--start",
                "2024-01",
                "--quarters",
                "4",
                "--output",
                str(out),
            ]
        )
        assert rc == 2
        assert not out.exists()

    def test_main_writes_json_and_returns_fail_code(
        self,
        stress_module: ModuleType,
        seasonal_strategy: Strategy,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "report.json"
        rc = stress_module.main(
            [
                "--strategy",
                seasonal_strategy.name,
                "--start",
                "2024-01",
                "--quarters",
                "4",
                "--output",
                str(out),
            ]
        )
        assert rc == 1, "FAIL verdict should return exit code 1"
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["verdict"] == "FAIL"
        assert payload["strategy"] == seasonal_strategy.name
        captured = capsys.readouterr()
        assert "VERDICT: FAIL" in captured.out

    def test_main_passes_clean_strategy_with_exit_zero(
        self,
        stress_module: ModuleType,
        all_positive_strategy: Strategy,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "ok.json"
        rc = stress_module.main(
            [
                "--strategy",
                all_positive_strategy.name,
                "--start",
                "2024-01",
                "--quarters",
                "4",
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        payload = json.loads(out.read_text())
        assert payload["verdict"] == "PASS"

    def test_default_report_path_format(self, stress_module: ModuleType) -> None:
        import datetime as dt

        p = stress_module.default_report_path("buy-and-hold", today=dt.date(2026, 5, 16))
        assert str(p) == "/tmp/stress_buy-and-hold_20260516.json"

    def test_default_report_path_sanitises_unsafe_chars(self, stress_module: ModuleType) -> None:
        import datetime as dt

        p = stress_module.default_report_path("foo/bar", today=dt.date(2026, 5, 16))
        assert "/" not in p.name
        assert "foo_bar" in p.name


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default_buy_and_hold_registered(self) -> None:
        strat = get("buy-and-hold")
        assert strat.name == "buy-and-hold"

    def test_get_raises_keyerror_with_helpful_message(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            get("__never_registered__")
