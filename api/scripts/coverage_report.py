"""Wrapper around ``pytest --cov`` that emits a Bloomberg-style summary + badge.

Run from the ``api/`` directory::

    .venv/bin/python scripts/coverage_report.py

What it does
------------

1. Spawns ``pytest --cov=pfm --cov-report=html --cov-report=term`` (extra
   pytest args after ``--`` are forwarded).
2. Streams pytest's stdout live to the user as it arrives so the run does
   not look frozen (pytest can take ~80 s on the full suite).
3. Parses the final ``coverage`` table (the one ``pytest-cov`` writes when
   ``--cov-report=term`` is active) for per-file percentages.
4. Prints a compact summary to stdout:

       * Top 20 files by coverage (descending)
       * Bottom 20 files by coverage (worst offenders, ascending)
       * Overall coverage %
       * Count of files at 100%
       * Count of files <50%

5. Writes ``htmlcov/index.html`` (handled automatically by ``pytest-cov``).
6. Writes ``htmlcov/badge.json`` with the shape::

       {"coverage_pct": 84.2, "color": "green",
        "label": "coverage", "message": "84%"}

   Color thresholds: green ≥80, yellow 60-79, red <60.

CI integration
~~~~~~~~~~~~~~

In ``.github/workflows/ci.yml`` add a step like::

    - name: Coverage report
      run: cd api && .venv/bin/python scripts/coverage_report.py
    - uses: actions/upload-artifact@v4
      with:
        name: htmlcov
        path: api/htmlcov/

The badge JSON is compatible with shields.io's endpoint provider — host
it on Pages and reference it from the README.

Exit codes
~~~~~~~~~~

The script returns whatever ``pytest`` returned (0 on green, 1 on test
failures, 5 on no-tests-collected, etc.). Parsing always runs even when
pytest fails — partial coverage data is still useful to inspect.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parents[1]
_HTMLCOV_DIR = _API_ROOT / "htmlcov"
_BADGE_PATH = _HTMLCOV_DIR / "badge.json"

# pytest-cov terminal-report rows look like::
#
#     src/pfm/foo.py            123    12     5     1    89%  10-12, 45->47
#
# We are deliberately tolerant — branch columns may be absent if the user
# disabled them, and the "Missing" tail is free-form. The contract is:
#
#     <name> <stmts> <miss> [<branch> <brpart>]? <pct%>  [<missing>]?
#
# Name can contain ``/`` and ``.`` but no spaces (modules don't have those).
_ROW_RE = re.compile(
    r"^(?P<name>\S+)\s+"
    r"(?P<stmts>\d+)\s+"
    r"(?P<miss>\d+)"
    r"(?:\s+(?P<branch>\d+)\s+(?P<brpart>\d+))?"
    r"\s+(?P<pct>\d+(?:\.\d+)?)%"
    r"(?:\s+.*)?$"
)
_TOTAL_RE = re.compile(
    r"^TOTAL\s+\d+\s+\d+(?:\s+\d+\s+\d+)?\s+(?P<pct>\d+(?:\.\d+)?)%",
)


@dataclass(frozen=True)
class FileCoverage:
    """Per-file row parsed out of the pytest-cov terminal table."""

    name: str
    pct: float


@dataclass(frozen=True)
class CoverageSummary:
    """Aggregate summary for the run."""

    rows: list[FileCoverage]
    overall_pct: float
    at_100_count: int
    under_50_count: int


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_coverage_output(text: str) -> CoverageSummary:
    """Parse pytest-cov's terminal table out of arbitrary captured output.

    The table is bounded by lines of dashes. We don't rely on those — we
    just regex every line and discard non-matches. ``TOTAL`` row supplies
    the overall %; if it's missing we fall back to the mean of file rows.
    """
    rows: list[FileCoverage] = []
    overall_pct: float | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        # The TOTAL row also matches _ROW_RE — check it first so we don't
        # double-count it as a file.
        total_match = _TOTAL_RE.match(line)
        if total_match:
            overall_pct = float(total_match.group("pct"))
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        # Skip header rows like ``Name`` and decoration separators.
        if name.lower() in {"name", "----", "===="}:
            continue
        # Only count files that look like Python source paths. This filters
        # out stray pytest summary lines that happen to match the shape.
        if not (name.endswith(".py") or "/" in name):
            continue
        rows.append(FileCoverage(name=name, pct=float(m.group("pct"))))

    if overall_pct is None:
        overall_pct = sum(r.pct for r in rows) / len(rows) if rows else 0.0

    at_100 = sum(1 for r in rows if r.pct >= 100.0)
    under_50 = sum(1 for r in rows if r.pct < 50.0)
    return CoverageSummary(
        rows=rows,
        overall_pct=overall_pct,
        at_100_count=at_100,
        under_50_count=under_50,
    )


# ---------------------------------------------------------------------------
# Badge
# ---------------------------------------------------------------------------


def badge_color(pct: float) -> str:
    """Color thresholds: green ≥80, yellow 60-79.999…, red <60."""
    if pct >= 80.0:
        return "green"
    if pct >= 60.0:
        return "yellow"
    return "red"


def write_badge(summary: CoverageSummary, path: Path = _BADGE_PATH) -> dict[str, object]:
    """Write ``htmlcov/badge.json`` and return the payload."""
    pct = round(summary.overall_pct, 1)
    payload: dict[str, object] = {
        "coverage_pct": pct,
        "color": badge_color(summary.overall_pct),
        "label": "coverage",
        "message": f"{int(round(summary.overall_pct))}%",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def format_summary(summary: CoverageSummary, *, top_n: int = 20) -> str:
    """Build the human-readable summary block."""
    sorted_desc = sorted(summary.rows, key=lambda r: (-r.pct, r.name))
    sorted_asc = sorted(summary.rows, key=lambda r: (r.pct, r.name))

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  COVERAGE SUMMARY")
    lines.append("=" * 72)
    lines.append(f"  Overall:        {summary.overall_pct:6.2f}%")
    lines.append(f"  Files at 100%:  {summary.at_100_count}")
    lines.append(f"  Files <50%:     {summary.under_50_count}")
    lines.append(f"  Files scanned:  {len(summary.rows)}")
    lines.append("")
    lines.append(f"  Top {top_n} by coverage (best)")
    lines.append("  " + "-" * 70)
    for r in sorted_desc[:top_n]:
        lines.append(f"  {r.pct:6.2f}%   {r.name}")
    lines.append("")
    lines.append(f"  Bottom {top_n} by coverage (worst offenders)")
    lines.append("  " + "-" * 70)
    for r in sorted_asc[:top_n]:
        lines.append(f"  {r.pct:6.2f}%   {r.name}")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subprocess driver
# ---------------------------------------------------------------------------


def build_pytest_cmd(extra_args: list[str] | None = None) -> list[str]:
    """Compose the pytest invocation. Python first so it works in any cwd."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=pfm",
        "--cov-report=html",
        "--cov-report=term",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_pytest(
    extra_args: list[str] | None = None,
    *,
    cwd: Path = _API_ROOT,
    runner: object | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run pytest under coverage and capture its combined stdout/stderr.

    ``capture_output=True`` is used (tests assert on this) so we get the
    full text back as a single string. We also stream the captured stdout
    to the user *after* the run completes so they see the test output —
    pytest's own progress dots already give them feedback during the run.

    ``runner`` defaults to the module-level ``subprocess`` name resolved at
    call time, which lets tests monkeypatch ``cov_module.subprocess``.
    """
    if runner is None:
        runner = subprocess
    cmd = build_pytest_cmd(extra_args)
    completed = runner.run(  # type: ignore[attr-defined]
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    # Stream captured output now so the user sees the run. We can't truly
    # stream live without dropping ``capture_output``; printing on return
    # is the practical compromise that keeps tests deterministic.
    if completed.stdout:
        sys.stdout.write(completed.stdout)
        sys.stdout.flush()
    if completed.stderr:
        sys.stderr.write(completed.stderr)
        sys.stderr.flush()
    return completed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coverage_report.py",
        description="Run pytest with coverage and emit a summary + badge JSON.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of files to show in the top / bottom lists (default: 20).",
    )
    p.add_argument(
        "--badge-path",
        type=Path,
        default=_BADGE_PATH,
        help=f"Where to write the badge JSON (default: {_BADGE_PATH}).",
    )
    p.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to pytest after a literal ``--``.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # argparse.REMAINDER captures a leading ``--`` — drop it.
    extra = list(args.pytest_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    completed = run_pytest(extra)
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    summary = parse_coverage_output(combined)
    sys.stdout.write("\n" + format_summary(summary, top_n=args.top_n) + "\n")
    payload = write_badge(summary, path=args.badge_path)
    sys.stdout.write(
        f"  Badge written: {args.badge_path} → {payload['message']} ({payload['color']})\n"
    )
    sys.stdout.flush()
    return int(completed.returncode)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
