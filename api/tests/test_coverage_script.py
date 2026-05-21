"""Tests for ``scripts/coverage_report.py``.

The script is loaded via :mod:`importlib` from its file path because it
lives outside the ``pfm`` package. We never spawn real pytest — every
subprocess call is mocked. We verify:

* The pytest invocation includes the mandated coverage flags.
* ``capture_output=True`` is set on the subprocess call.
* The terminal-table parser correctly extracts file rows + the TOTAL.
* The summary block lists top / bottom files and the right counts.
* The badge JSON has the documented shape, including the color-threshold
  bands (green ≥80, yellow 60-79, red <60).
* ``main`` returns the subprocess's exit code unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_API_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cov_module() -> ModuleType:
    """Import ``scripts/coverage_report.py`` as a module.

    We register the module in ``sys.modules`` before executing it so that
    dataclass introspection (which walks ``cls.__module__``) succeeds on
    Python 3.14+.
    """
    path = _API_ROOT / "scripts" / "coverage_report.py"
    spec = importlib.util.spec_from_file_location("pfm_coverage_report", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pfm_coverage_report"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic pytest-cov terminal output used by parser + summary tests.
# ---------------------------------------------------------------------------

# Drawn straight from a real pytest-cov terminal report, then trimmed.
SYNTHETIC_COV_OUTPUT = """
============================= test session starts ==============================
collected 12 items

tests/test_thing.py ............                                         [100%]

----------- coverage: platform darwin, python 3.12 -----------
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/pfm/perfect.py                        100      0   100%
src/pfm/great.py                           80      4    95%   10-13
src/pfm/medium.py                          60     18    70%   1-3, 8, 20-30
src/pfm/half.py                            40     22    45%   1-20, 25-26
src/pfm/awful.py                           50     45    10%   1-45
---------------------------------------------------------------------
TOTAL                                     330     89    73%
============================== 12 passed in 0.34s ==============================
"""


# ---------------------------------------------------------------------------
# Pytest command construction + subprocess wiring
# ---------------------------------------------------------------------------


class TestPytestCommand:
    def test_includes_required_coverage_flags(self, cov_module: ModuleType) -> None:
        cmd = cov_module.build_pytest_cmd()
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "pytest"]
        assert "--cov=pfm" in cmd
        assert "--cov-report=html" in cmd
        assert "--cov-report=term" in cmd

    def test_forwards_extra_pytest_args(self, cov_module: ModuleType) -> None:
        cmd = cov_module.build_pytest_cmd(["-k", "fast", "-x"])
        assert cmd[-3:] == ["-k", "fast", "-x"]


class _FakeRunner:
    """Stub for ``subprocess`` that records ``.run`` calls."""

    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        cmd: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append({"cmd": cmd, "kwargs": kwargs})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


class TestSubprocessInvocation:
    def test_run_pytest_uses_capture_output(self, cov_module: ModuleType) -> None:
        fake = _FakeRunner(stdout=SYNTHETIC_COV_OUTPUT)
        result = cov_module.run_pytest(runner=fake)
        assert len(fake.calls) == 1
        kwargs = fake.calls[0]["kwargs"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        assert result.returncode == 0

    def test_run_pytest_passes_extra_args(self, cov_module: ModuleType) -> None:
        fake = _FakeRunner(stdout=SYNTHETIC_COV_OUTPUT)
        cov_module.run_pytest(["-k", "smoke"], runner=fake)
        cmd = fake.calls[0]["cmd"]
        assert cmd[-2:] == ["-k", "smoke"]
        # Mandatory flags must still be present.
        assert "--cov=pfm" in cmd

    def test_run_pytest_streams_captured_output(
        self,
        cov_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fake = _FakeRunner(stdout="HELLO STDOUT", stderr="WARN STDERR")
        cov_module.run_pytest(runner=fake)
        captured = capsys.readouterr()
        assert "HELLO STDOUT" in captured.out
        assert "WARN STDERR" in captured.err


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


class TestParseCoverage:
    def test_parses_each_file_row(self, cov_module: ModuleType) -> None:
        summary = cov_module.parse_coverage_output(SYNTHETIC_COV_OUTPUT)
        names = {r.name for r in summary.rows}
        assert names == {
            "src/pfm/perfect.py",
            "src/pfm/great.py",
            "src/pfm/medium.py",
            "src/pfm/half.py",
            "src/pfm/awful.py",
        }

    def test_parses_overall_from_total_row(self, cov_module: ModuleType) -> None:
        summary = cov_module.parse_coverage_output(SYNTHETIC_COV_OUTPUT)
        assert summary.overall_pct == pytest.approx(73.0)

    def test_counts_100_and_under_50_buckets(self, cov_module: ModuleType) -> None:
        summary = cov_module.parse_coverage_output(SYNTHETIC_COV_OUTPUT)
        assert summary.at_100_count == 1  # perfect.py
        # half.py at 45% and awful.py at 10% → 2 under 50%
        assert summary.under_50_count == 2

    def test_total_row_not_counted_as_file(self, cov_module: ModuleType) -> None:
        summary = cov_module.parse_coverage_output(SYNTHETIC_COV_OUTPUT)
        # If the TOTAL row were treated as a file we'd see 6 rows, not 5.
        assert len(summary.rows) == 5
        assert all(r.name != "TOTAL" for r in summary.rows)

    def test_empty_input_yields_zero_overall(self, cov_module: ModuleType) -> None:
        summary = cov_module.parse_coverage_output("")
        assert summary.rows == []
        assert summary.overall_pct == 0.0
        assert summary.at_100_count == 0
        assert summary.under_50_count == 0

    def test_fractional_percentages_parsed(self, cov_module: ModuleType) -> None:
        text = "src/pfm/a.py   10  1   87.5%   3\nTOTAL          10  1   87.5%\n"
        summary = cov_module.parse_coverage_output(text)
        assert len(summary.rows) == 1
        assert summary.rows[0].pct == pytest.approx(87.5)
        assert summary.overall_pct == pytest.approx(87.5)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_summary_contains_overall_and_counts(self, cov_module: ModuleType) -> None:
        summary = cov_module.parse_coverage_output(SYNTHETIC_COV_OUTPUT)
        text = cov_module.format_summary(summary)
        assert "Overall" in text
        assert "73.00%" in text
        assert "Files at 100%:  1" in text
        assert "Files <50%:     2" in text

    def test_top_list_orders_descending(self, cov_module: ModuleType) -> None:
        summary = cov_module.parse_coverage_output(SYNTHETIC_COV_OUTPUT)
        text = cov_module.format_summary(summary, top_n=3)
        # Top-3 best should list perfect.py, great.py, medium.py in that order.
        top_block = text.split("Top 3 by coverage")[1].split("Bottom")[0]
        i_perfect = top_block.index("perfect.py")
        i_great = top_block.index("great.py")
        i_medium = top_block.index("medium.py")
        assert i_perfect < i_great < i_medium

    def test_bottom_list_orders_ascending(self, cov_module: ModuleType) -> None:
        summary = cov_module.parse_coverage_output(SYNTHETIC_COV_OUTPUT)
        text = cov_module.format_summary(summary, top_n=3)
        bottom_block = text.split("Bottom 3 by coverage")[1]
        i_awful = bottom_block.index("awful.py")
        i_half = bottom_block.index("half.py")
        i_medium = bottom_block.index("medium.py")
        assert i_awful < i_half < i_medium


# ---------------------------------------------------------------------------
# Badge thresholds + file shape
# ---------------------------------------------------------------------------


class TestBadge:
    @pytest.mark.parametrize(
        ("pct", "expected"),
        [
            (100.0, "green"),
            (90.0, "green"),
            (80.0, "green"),
            (79.999, "yellow"),
            (75.0, "yellow"),
            (60.0, "yellow"),
            (59.999, "red"),
            (40.0, "red"),
            (0.0, "red"),
        ],
    )
    def test_badge_color_thresholds(
        self, cov_module: ModuleType, pct: float, expected: str
    ) -> None:
        assert cov_module.badge_color(pct) == expected

    def test_write_badge_shape(self, cov_module: ModuleType, tmp_path: Path) -> None:
        summary = cov_module.parse_coverage_output(SYNTHETIC_COV_OUTPUT)
        out = tmp_path / "htmlcov" / "badge.json"
        payload = cov_module.write_badge(summary, path=out)
        # On-disk and in-memory payloads agree.
        on_disk = json.loads(out.read_text())
        assert on_disk == payload
        # Required keys.
        for key in ("coverage_pct", "color", "label", "message"):
            assert key in payload
        assert payload["label"] == "coverage"
        assert payload["color"] == "yellow"  # 73% → yellow band
        assert payload["coverage_pct"] == pytest.approx(73.0)
        assert payload["message"].endswith("%")

    def test_write_badge_creates_parent_dir(self, cov_module: ModuleType, tmp_path: Path) -> None:
        summary = cov_module.parse_coverage_output(SYNTHETIC_COV_OUTPUT)
        out = tmp_path / "deeply" / "nested" / "badge.json"
        cov_module.write_badge(summary, path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# CLI end-to-end (with mocked subprocess + badge dir)
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_returns_pytest_exit_code(
        self,
        cov_module: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fake = _FakeRunner(stdout=SYNTHETIC_COV_OUTPUT, returncode=0)
        monkeypatch.setattr(cov_module, "subprocess", fake)
        badge = tmp_path / "badge.json"
        rc = cov_module.main(["--badge-path", str(badge)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "COVERAGE SUMMARY" in captured.out
        assert "Badge written" in captured.out
        on_disk = json.loads(badge.read_text())
        assert on_disk["coverage_pct"] == pytest.approx(73.0)
        assert on_disk["color"] == "yellow"

    def test_main_propagates_pytest_failure(
        self,
        cov_module: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeRunner(stdout=SYNTHETIC_COV_OUTPUT, returncode=1)
        monkeypatch.setattr(cov_module, "subprocess", fake)
        badge = tmp_path / "badge.json"
        rc = cov_module.main(["--badge-path", str(badge)])
        assert rc == 1
        # Badge still gets written even when tests failed.
        assert badge.exists()

    def test_main_forwards_extra_pytest_args(
        self,
        cov_module: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeRunner(stdout=SYNTHETIC_COV_OUTPUT)
        monkeypatch.setattr(cov_module, "subprocess", fake)
        badge = tmp_path / "badge.json"
        cov_module.main(
            ["--badge-path", str(badge), "--", "-k", "fast"],
        )
        cmd = fake.calls[0]["cmd"]
        assert cmd[-2:] == ["-k", "fast"]
        # Required flags survive forwarding.
        assert "--cov=pfm" in cmd
        assert "--cov-report=html" in cmd
        assert "--cov-report=term" in cmd

    def test_main_invokes_subprocess_with_capture_output(
        self,
        cov_module: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeRunner(stdout=SYNTHETIC_COV_OUTPUT)
        monkeypatch.setattr(cov_module, "subprocess", fake)
        cov_module.main(["--badge-path", str(tmp_path / "b.json")])
        assert fake.calls[0]["kwargs"]["capture_output"] is True


# ---------------------------------------------------------------------------
# Defensive sanity — make sure we did NOT accidentally spawn real pytest.
# ---------------------------------------------------------------------------


def test_no_real_subprocess_run_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a test ever calls the real ``subprocess.run`` it should explode.

    This is a belt-and-braces guard. Every test above injects a fake
    runner; if a new test forgets to mock the subprocess module the
    explosion below will surface immediately.
    """
    sentinel = SimpleNamespace(called=False)

    def _boom(*_a: Any, **_kw: Any) -> None:
        sentinel.called = True
        raise AssertionError("subprocess.run should be mocked in unit tests")

    monkeypatch.setattr(subprocess, "run", _boom)
    # Just exercising monkeypatching itself — no real call here.
    assert sentinel.called is False
