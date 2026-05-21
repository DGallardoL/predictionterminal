"""Tests for ``scripts/factor_coverage_stats.py``.

The script lives outside the ``pfm`` package, so we load it via
``importlib`` from its file path. Every test injects a synthetic
fetcher — we never hit Polymarket / Kalshi / FRED from ``pytest``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from pfm.factors import FactorConfig

_API_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Module-loader fixture (the script is not a regular package import).
# ---------------------------------------------------------------------------


@pytest.fixture()
def fcs_module() -> ModuleType:
    """Import ``scripts/factor_coverage_stats.py`` as a module."""
    path = _API_ROOT / "scripts" / "factor_coverage_stats.py"
    spec = importlib.util.spec_from_file_location("pfm_factor_coverage_stats", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pfm_factor_coverage_stats"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared synthetic catalog + fetcher helpers.
# ---------------------------------------------------------------------------


def _fc(fid: str, *, theme: str, source: str = "polymarket") -> FactorConfig:
    """Tiny FactorConfig builder."""
    return FactorConfig(
        id=fid,
        name=fid.replace("-", " ").title(),
        slug=fid,
        source=source,
        description="(test)",
        theme=theme,
        is_probability=True,
    )


def _catalog() -> dict[str, FactorConfig]:
    """Mixed-theme synthetic catalog used by multiple tests."""
    return {
        "btc-100k": _fc("btc-100k", theme="crypto"),
        "eth-flip": _fc("eth-flip", theme="crypto"),
        "ada-2": _fc("ada-2", theme="crypto"),
        "trump-2024": _fc("trump-2024", theme="politics"),
        "biden-step-down": _fc("biden-step-down", theme="politics"),
        "fed-march-cut": _fc("fed-march-cut", theme="macro"),
        "cpi-above-3": _fc("cpi-above-3", theme="macro"),
        "lakers-win": _fc("lakers-win", theme="sports"),
    }


def _frame(rows: int) -> pd.DataFrame:
    """Synthetic ``[date, price]`` frame with ``rows`` rows."""
    if rows <= 0:
        return pd.DataFrame(columns=["price"])
    idx = pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame({"price": [0.5] * rows}, index=idx)


# ---------------------------------------------------------------------------
# count_observations — single-factor inspection.
# ---------------------------------------------------------------------------


class TestCountObservations:
    def test_counts_rows_from_returned_frame(self, fcs_module: ModuleType) -> None:
        fc = _fc("btc-100k", theme="crypto")

        def fetcher(_fc: FactorConfig, _s: pd.Timestamp, _e: pd.Timestamp) -> pd.DataFrame:
            return _frame(42)

        obs = fcs_module.count_observations(
            fc,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
            fetcher=fetcher,
        )
        assert obs.factor_id == "btc-100k"
        assert obs.theme == "crypto"
        assert obs.n_obs == 42
        assert obs.error == ""
        assert obs.has_data is True

    def test_zero_rows_marked_not_has_data(self, fcs_module: ModuleType) -> None:
        fc = _fc("lakers-win", theme="sports")
        obs = fcs_module.count_observations(
            fc,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
            fetcher=lambda *_a, **_k: _frame(0),
        )
        assert obs.n_obs == 0
        assert obs.has_data is False

    def test_fetcher_exception_is_swallowed_as_error(self, fcs_module: ModuleType) -> None:
        fc = _fc("fed-march-cut", theme="macro")

        def boom(*_a, **_k) -> pd.DataFrame:
            raise RuntimeError("kaboom")

        obs = fcs_module.count_observations(
            fc,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
            fetcher=boom,
        )
        assert obs.error.startswith("RuntimeError")
        assert obs.n_obs == 0
        assert obs.has_data is False


# ---------------------------------------------------------------------------
# aggregate_by_theme — pure grouping.
# ---------------------------------------------------------------------------


class TestAggregateByTheme:
    def test_groups_by_theme_and_orders_alphabetically(self, fcs_module: ModuleType) -> None:
        Obs = fcs_module.FactorObservation
        observations = [
            Obs(factor_id="a", theme="crypto", n_obs=100),
            Obs(factor_id="b", theme="crypto", n_obs=200),
            Obs(factor_id="c", theme="politics", n_obs=10),
            Obs(factor_id="d", theme="sports", n_obs=5),
        ]
        stats = fcs_module.aggregate_by_theme(observations, min_obs=1)
        themes = [s.theme for s in stats]
        assert themes == sorted(themes)
        assert themes == ["crypto", "politics", "sports"]

    def test_with_data_uses_min_obs_threshold(self, fcs_module: ModuleType) -> None:
        Obs = fcs_module.FactorObservation
        observations = [
            Obs(factor_id="a", theme="crypto", n_obs=300),
            Obs(factor_id="b", theme="crypto", n_obs=5),  # below threshold
            Obs(factor_id="c", theme="crypto", n_obs=0),  # zero rows
        ]
        stats = fcs_module.aggregate_by_theme(observations, min_obs=30)
        assert len(stats) == 1
        only = stats[0]
        assert only.factor_count == 3
        # Only the 300-row factor crosses the 30-row threshold.
        assert only.with_data == 1

    def test_median_and_min_observations_per_theme(self, fcs_module: ModuleType) -> None:
        Obs = fcs_module.FactorObservation
        observations = [
            Obs(factor_id="a", theme="politics", n_obs=10),
            Obs(factor_id="b", theme="politics", n_obs=50),
            Obs(factor_id="c", theme="politics", n_obs=200),
            Obs(factor_id="d", theme="macro", n_obs=42),
        ]
        stats = {s.theme: s for s in fcs_module.aggregate_by_theme(observations)}
        # politics: median of [10, 50, 200] = 50; min = 10
        assert stats["politics"].median_obs == 50
        assert stats["politics"].min_obs == 10
        # macro: single sample → median == min == 42
        assert stats["macro"].median_obs == 42
        assert stats["macro"].min_obs == 42

    def test_missing_theme_defaults_to_other(self, fcs_module: ModuleType) -> None:
        Obs = fcs_module.FactorObservation
        observations = [Obs(factor_id="a", theme="", n_obs=10)]
        stats = fcs_module.aggregate_by_theme(observations)
        assert stats[0].theme == "other"

    def test_error_observation_does_not_count_as_with_data(self, fcs_module: ModuleType) -> None:
        Obs = fcs_module.FactorObservation
        observations = [
            Obs(factor_id="a", theme="crypto", n_obs=0, error="boom"),
            Obs(factor_id="b", theme="crypto", n_obs=10),
        ]
        stats = fcs_module.aggregate_by_theme(observations, min_obs=1)
        only = stats[0]
        assert only.factor_count == 2
        assert only.with_data == 1
        # Errored factor contributes a 0-row sample for min/median.
        assert only.min_obs == 0


# ---------------------------------------------------------------------------
# build_report — end-to-end shape with mocked fetcher.
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_report_top_level_shape(self, fcs_module: ModuleType) -> None:
        cat = _catalog()
        report = fcs_module.build_report(
            cat,
            fetcher=lambda fc, s, e: _frame(50),
            workers=1,
        )
        assert set(report.keys()) == {"checked_at", "themes", "totals"}
        # ISO-8601, Z-suffixed.
        assert report["checked_at"].endswith("Z")
        # Round-trip parseable.
        parsed = datetime.fromisoformat(report["checked_at"].replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

    def test_totals_match_per_theme_sum(self, fcs_module: ModuleType) -> None:
        cat = _catalog()

        # Vary the fetched row count by source/theme so with_data differs.
        def fetcher(fc, _s, _e):
            return _frame({"crypto": 100, "politics": 0, "macro": 30, "sports": 200}[fc.theme])

        report = fcs_module.build_report(cat, fetcher=fetcher, workers=1)
        totals = report["totals"]
        assert totals["factors"] == len(cat)
        assert totals["themes"] == len({fc.theme for fc in cat.values()})
        themes = report["themes"]
        assert totals["with_data"] == sum(t["with_data"] for t in themes)
        assert totals["stale"] == totals["factors"] - totals["with_data"]
        # politics theme: 2 factors, both zero rows → with_data == 0
        politics = next(t for t in themes if t["theme"] == "politics")
        assert politics["factor_count"] == 2
        assert politics["with_data"] == 0
        # crypto theme: 3 factors all with 100 rows
        crypto = next(t for t in themes if t["theme"] == "crypto")
        assert crypto["factor_count"] == 3
        assert crypto["with_data"] == 3
        assert crypto["median_obs"] == 100
        assert crypto["min_obs"] == 100

    def test_empty_catalog_yields_empty_report(self, fcs_module: ModuleType) -> None:
        report = fcs_module.build_report(
            {},
            fetcher=lambda *_a, **_k: _frame(0),
            workers=1,
        )
        assert report["totals"] == {
            "factors": 0,
            "themes": 0,
            "with_data": 0,
            "stale": 0,
        }
        assert report["themes"] == []

    def test_workers_concurrent_path_returns_same_result(self, fcs_module: ModuleType) -> None:
        cat = _catalog()

        def fetcher(fc, _s, _e):
            return _frame(20)

        seq = fcs_module.build_report(cat, fetcher=fetcher, workers=1)
        par = fcs_module.build_report(cat, fetcher=fetcher, workers=4)
        # Themes list is sorted deterministically, so the dicts must match.
        assert seq["themes"] == par["themes"]
        assert seq["totals"] == par["totals"]

    def test_now_override_propagates_to_checked_at(self, fcs_module: ModuleType) -> None:
        cat = {"a": _fc("a", theme="crypto")}
        frozen = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        report = fcs_module.build_report(
            cat,
            fetcher=lambda *_a, **_k: _frame(1),
            workers=1,
            now=frozen,
        )
        assert report["checked_at"] == "2026-01-02T03:04:05Z"

    def test_fetcher_exception_marks_factor_stale(self, fcs_module: ModuleType) -> None:
        cat = {
            "good": _fc("good", theme="crypto"),
            "bad": _fc("bad", theme="crypto"),
        }

        def fetcher(fc, _s, _e):
            if fc.id == "bad":
                raise ValueError("nope")
            return _frame(10)

        report = fcs_module.build_report(cat, fetcher=fetcher, workers=1)
        crypto = report["themes"][0]
        assert crypto["factor_count"] == 2
        assert crypto["with_data"] == 1
        assert report["totals"]["stale"] == 1


# ---------------------------------------------------------------------------
# CLI / run() — output handling.
# ---------------------------------------------------------------------------


class TestCLI:
    def _write_factors_yml(self, tmp: Path) -> Path:
        """Write a minimal factors.yml fixture for the CLI tests."""
        body = """
factors:
  - id: btc-100k
    name: BTC 100k
    slug: btc-100k
    source: polymarket
    description: test
    theme: crypto
  - id: eth-flip
    name: ETH flippening
    slug: eth-flip
    source: polymarket
    description: test
    theme: crypto
  - id: trump-2024
    name: Trump 2024
    slug: trump-2024
    source: polymarket
    description: test
    theme: politics
"""
        path = tmp / "factors.yml"
        path.write_text(body)
        return path

    def test_run_writes_json_to_output_path(
        self,
        fcs_module: ModuleType,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        factors_yml = self._write_factors_yml(tmp_path)
        out = tmp_path / "out" / "report.json"
        args = fcs_module._parse_args(
            [
                "--output",
                str(out),
                "--factors-yml",
                str(factors_yml),
                "--workers",
                "1",
            ]
        )
        rc = fcs_module.run(args, fetcher=lambda *_a, **_k: _frame(7))
        assert rc == 0
        # File created with parents.
        assert out.exists()
        payload = json.loads(out.read_text())
        assert {"checked_at", "themes", "totals"} <= payload.keys()
        assert payload["totals"]["factors"] == 3
        # Stdout summarises what was written.
        captured = capsys.readouterr()
        assert "wrote" in captured.out
        assert "report.json" in captured.out

    def test_run_prints_json_to_stdout_when_no_output(
        self,
        fcs_module: ModuleType,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        factors_yml = self._write_factors_yml(tmp_path)
        args = fcs_module._parse_args(["--factors-yml", str(factors_yml), "--workers", "1"])
        rc = fcs_module.run(args, fetcher=lambda *_a, **_k: _frame(3))
        assert rc == 0
        captured = capsys.readouterr()
        # JSON-parseable stdout.
        payload = json.loads(captured.out)
        assert payload["totals"]["factors"] == 3
        assert payload["totals"]["themes"] == 2  # crypto + politics

    def test_run_limit_truncates_catalog(
        self,
        fcs_module: ModuleType,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        factors_yml = self._write_factors_yml(tmp_path)
        args = fcs_module._parse_args(
            [
                "--factors-yml",
                str(factors_yml),
                "--workers",
                "1",
                "--limit",
                "1",
            ]
        )
        rc = fcs_module.run(args, fetcher=lambda *_a, **_k: _frame(3))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["totals"]["factors"] == 1

    def test_parse_args_defaults(self, fcs_module: ModuleType) -> None:
        args = fcs_module._parse_args([])
        assert args.output is None
        assert args.workers == fcs_module.DEFAULT_WORKERS
        assert args.lookback_days == fcs_module.DEFAULT_LOOKBACK_DAYS
        assert args.min_obs == fcs_module.DEFAULT_MIN_OBS
        assert args.limit is None
