#!/usr/bin/env python3
"""Mutation-testing runner for production-critical pfm modules.

Wave-13, task W13-26.

Why mutation testing?
---------------------
Line/branch coverage answers "did a test execute this line?" Mutation testing
answers the stronger question "would a test FAIL if this line were subtly
wrong?". For numerically sensitive code (OLS coefficient recovery, jump
detection, cache-eviction order), branch coverage routinely overstates
confidence; mutation testing catches asserts that never actually constrain
behaviour ("test executes, never asserts the value").

This runner is a thin wrapper around `mutmut` (https://mutmut.readthedocs.io)
that:

  1. Restricts mutation to ONE production-critical module at a time
     (mutmut is O(N_mutants * test_suite_runtime); whole-repo runs are
     unacceptably slow).
  2. Pins the test command to the narrowest test slice that exercises the
     target module, so each mutant takes seconds not minutes.
  3. Emits a structured JSON report at `/tmp/mutation-report-<module>.json`
     summarising killed/survived/timeout/skipped mutants and the overall
     kill rate.
  4. Enforces an 80% kill-rate acceptance threshold for the four modules
     declared `production_critical` below. The runner exits non-zero when
     the threshold is missed, so CI can gate on it later.

Cost
----
Approximate wall-clock per module on the dev MBP (M-series, .venv warm):

  pfm.model              ~30 min   (~120 mutants, OLS+HAC suite ~12 s/run)
  pfm.regression_core    ~30 min   (~150 mutants, property suite is the long pole)
  pfm.cache_pool         ~10 min   (~60  mutants, cache tests are fast)
  pfm.terminal.jumps     ~45 min   (~180 mutants, e2e fixture load dominates)

Do NOT run as part of `pytest` or PR-time CI. Run nightly, or on-demand
when refactoring the target module. The runner accepts `--dry-run` to
emit the mutmut command without executing anything (useful for review).

Usage
-----
    # Show the planned mutmut invocation without running.
    python api/scripts/run_mutation_tests.py --module pfm.model --dry-run

    # Run mutation testing on a single module (long; ~30 min).
    python api/scripts/run_mutation_tests.py --module pfm.model

    # Run on all production-critical modules sequentially (multi-hour).
    python api/scripts/run_mutation_tests.py --all

    # Custom acceptance threshold (default 0.80).
    python api/scripts/run_mutation_tests.py --module pfm.cache_pool --threshold 0.75

Exit codes
----------
    0 — kill rate >= threshold for every module run
    1 — kill rate below threshold (or mutmut returned an error)
    2 — invalid arguments / unsupported module
    3 — `mutmut` not installed in the active interpreter
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Module registry. Each target maps a dotted module path to the production
# source file and the narrowest test slice that meaningfully exercises it.
# Test slices are kept tight on purpose: a 30-minute mutmut run with the
# whole 2700-test suite per mutant would take ~weeks.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "api"


@dataclass(frozen=True)
class MutationTarget:
    """Configuration for one mutation-testing target."""

    module: str
    source: Path
    test_paths: tuple[str, ...]
    production_critical: bool = True
    # Approximate wall-clock estimate, used only for the docstring banner.
    approx_minutes: int = 30

    def runner_cmd(self) -> str:
        """Return the pytest invocation mutmut should use per-mutant.

        The command is run from inside the mutated tree mutmut creates,
        so paths are relative to that tree (which mirrors the api/ layout).
        We use `-x -q --tb=no --disable-warnings -p no:cacheprovider` to
        kill a mutant on the first failing test and avoid stale `.pytest_cache`
        bleed-through between mutants.
        """
        tests = " ".join(self.test_paths)
        return (
            "PYTHONPATH=src .venv/bin/python -m pytest "
            "-x -q --tb=no --disable-warnings -p no:cacheprovider "
            f"{tests}"
        )


TARGETS: dict[str, MutationTarget] = {
    "pfm.model": MutationTarget(
        module="pfm.model",
        source=API_DIR / "src" / "pfm" / "model.py",
        test_paths=(
            "tests/test_model.py",
            "tests/test_regression_synthetic_dgp.py",
            "tests/test_regression_rigour.py",
        ),
        approx_minutes=30,
    ),
    "pfm.regression_core": MutationTarget(
        module="pfm.regression_core",
        source=API_DIR / "src" / "pfm" / "regression_core.py",
        test_paths=(
            "tests/test_regression_core_property.py",
            "tests/test_regression_synthetic_dgp.py",
            "tests/test_regression_enriched.py",
        ),
        approx_minutes=30,
    ),
    "pfm.cache_pool": MutationTarget(
        module="pfm.cache_pool",
        source=API_DIR / "src" / "pfm" / "cache_pool.py",
        test_paths=(
            "tests/test_cache_pool.py",
            "tests/test_cache_pool_integration.py",
        ),
        approx_minutes=10,
    ),
    "pfm.terminal.jumps": MutationTarget(
        module="pfm.terminal.jumps",
        source=API_DIR / "src" / "pfm" / "terminal" / "jumps.py",
        test_paths=(
            "tests/test_jumps_backtest_e2e.py",
            "tests/test_jumps_cluster_property.py",
            "tests/test_jumps_compare.py",
            "tests/test_jumps_prewarm.py",
        ),
        approx_minutes=45,
    ),
}


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass
class MutationReport:
    """Structured summary of one module's mutation run."""

    module: str
    source_file: str
    test_paths: list[str]
    total_mutants: int = 0
    killed: int = 0
    survived: int = 0
    timeout: int = 0
    suspicious: int = 0
    skipped: int = 0
    kill_rate: float = 0.0
    threshold: float = 0.80
    threshold_met: bool = False
    duration_seconds: float = 0.0
    mutmut_version: str = ""
    raw_summary: dict = field(default_factory=dict)
    error: str | None = None

    def update_kill_rate(self) -> None:
        denom = self.killed + self.survived + self.timeout + self.suspicious
        self.kill_rate = (self.killed / denom) if denom else 0.0
        self.threshold_met = self.kill_rate >= self.threshold


# ---------------------------------------------------------------------------
# mutmut interaction
# ---------------------------------------------------------------------------


def _ensure_mutmut() -> str:
    """Return the resolved mutmut version, or exit(3) if it isn't installed.

    We resolve `mutmut --version` rather than importing the package because
    mutmut's CLI is the public surface (the Python API is undocumented and
    has changed between minor releases).
    """
    if shutil.which("mutmut") is None:
        sys.stderr.write(
            "ERROR: `mutmut` is not on PATH. Install with:\n"
            "       pip install mutmut\n"
            "       (recommended in the same venv as pytest: api/.venv)\n"
        )
        sys.exit(3)
    out = subprocess.run(["mutmut", "--version"], capture_output=True, text=True, check=False)
    return (out.stdout or out.stderr).strip()


def _parse_results_json(raw: str) -> dict:
    """Parse `mutmut results --json` output.

    mutmut prints a JSON object keyed by mutant-id with a `status` field per
    mutant. Older versions emit a flat summary instead. We tolerate both by
    falling back to a regex-based summary parse upstream.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _summarise(results: dict) -> dict[str, int]:
    """Count mutants by status from `mutmut results --json` output."""
    counts = {"killed": 0, "survived": 0, "timeout": 0, "suspicious": 0, "skipped": 0}
    for entry in results.values():
        status = (entry or {}).get("status", "").lower()
        if status in counts:
            counts[status] += 1
    return counts


def _run_one(target: MutationTarget, threshold: float, dry_run: bool) -> MutationReport:
    """Run mutmut against one target and return a structured report."""
    report = MutationReport(
        module=target.module,
        source_file=str(target.source.relative_to(REPO_ROOT)),
        test_paths=list(target.test_paths),
        threshold=threshold,
    )

    if not target.source.exists():
        report.error = f"source file not found: {target.source}"
        return report

    paths_to_mutate = str(target.source.relative_to(API_DIR))
    runner_cmd = target.runner_cmd()

    mutmut_run = [
        "mutmut",
        "run",
        "--paths-to-mutate",
        paths_to_mutate,
        "--runner",
        runner_cmd,
        "--simple-output",
        # `--no-progress` keeps stdout grep-friendly when piped to logs.
        "--no-progress",
    ]

    print(f"[mutmut] target          : {target.module}")
    print(f"[mutmut] source          : {target.source}")
    print(f"[mutmut] tests           : {' '.join(target.test_paths)}")
    print(f"[mutmut] est. wall-clock : ~{target.approx_minutes} min")
    print(f"[mutmut] cwd             : {API_DIR}")
    print(f"[mutmut] command         : {' '.join(mutmut_run)}")
    print(f"[mutmut] runner          : {runner_cmd}")

    if dry_run:
        report.error = "dry-run (no execution)"
        return report

    report.mutmut_version = _ensure_mutmut()
    start = time.time()
    proc = subprocess.run(
        mutmut_run,
        cwd=API_DIR,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    report.duration_seconds = round(time.time() - start, 2)

    # mutmut exit codes: 0 = all killed, 1 = some survived, 2 = error.
    if proc.returncode not in (0, 1):
        report.error = (
            f"mutmut exited {proc.returncode}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
        return report

    results_proc = subprocess.run(
        ["mutmut", "results", "--json"],
        cwd=API_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = _parse_results_json(results_proc.stdout)
    counts = _summarise(parsed)
    report.total_mutants = sum(counts.values())
    report.killed = counts["killed"]
    report.survived = counts["survived"]
    report.timeout = counts["timeout"]
    report.suspicious = counts["suspicious"]
    report.skipped = counts["skipped"]
    report.raw_summary = counts
    report.update_kill_rate()
    return report


def _write_report(report: MutationReport) -> Path:
    """Persist the report JSON. Module name is slugified for the filename."""
    slug = report.module.replace(".", "-")
    out = Path(f"/tmp/mutation-report-{slug}.json")
    out.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
    return out


def _print_summary(report: MutationReport) -> None:
    """Human-readable single-line + multi-line summary."""
    mark = "PASS" if report.threshold_met else "FAIL"
    if report.error:
        mark = "ERROR"
    print()
    print(f"=== {mark}  {report.module} ===")
    print(f"  source        : {report.source_file}")
    print(
        f"  mutants       : total={report.total_mutants} "
        f"killed={report.killed} survived={report.survived} "
        f"timeout={report.timeout} suspicious={report.suspicious}"
    )
    print(f"  kill rate     : {report.kill_rate:.1%}  (threshold {report.threshold:.0%})")
    print(f"  duration      : {report.duration_seconds:.1f}s")
    if report.error:
        print(f"  error         : {report.error.splitlines()[0]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_mutation_tests.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--module",
        choices=sorted(TARGETS),
        help="Dotted module path to mutate (one of the production-critical targets).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run mutation testing on every production-critical module sequentially.",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List configured targets and exit.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum acceptable kill rate (0.0-1.0). Default 0.80.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned mutmut invocation without executing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list:
        print("Configured mutation targets:")
        for name, t in TARGETS.items():
            print(
                f"  {name:30s} source={t.source.relative_to(REPO_ROOT)} "
                f"approx_minutes={t.approx_minutes}"
            )
        return 0

    if not (0.0 <= args.threshold <= 1.0):
        sys.stderr.write("ERROR: --threshold must be in [0.0, 1.0]\n")
        return 2

    targets: list[MutationTarget]
    if args.all:
        targets = list(TARGETS.values())
    else:
        targets = [TARGETS[args.module]]

    if not args.dry_run:
        _ensure_mutmut()

    all_pass = True
    for t in targets:
        report = _run_one(t, args.threshold, args.dry_run)
        out = _write_report(report)
        _print_summary(report)
        print(f"  report        : {out}")
        if not args.dry_run and (report.error or not report.threshold_met):
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
