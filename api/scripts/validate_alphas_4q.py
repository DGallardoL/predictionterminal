"""4-quarter validation harness for all deployable alphas.

Wraps :mod:`scripts.stress_test` to drive every strategy currently tagged
``A_GOLD``, ``A_STRUCTURAL`` or ``B_VALIDATED`` in
``web/data/alpha_strategies.json`` through the same 4-quarter
sign-flip / Sharpe-floor stress test that the CLAUDE.md anti-alpha rule
mandates.

For each strategy this script:

1. Looks the strategy up by ``pair_id``. If a registered Strategy exists
   in :mod:`pfm.strategies_registry` we use it; otherwise we register a
   deterministic buy-and-hold strategy whose seed is the ``pair_id`` so
   synthetic prices are reproducible.
2. Calls ``stress_test.run_stress(strategy, prices, start='2024-01',
   quarters=4)``.
3. Records the per-quarter Sharpe values, sign-flip flag, deflated
   Sharpe and verdict.
4. Writes any FAILED strategies to a proposals JSON (``--output``,
   default ``/tmp/anti-alpha-proposals.json``). The
   ``web/data/alpha_strategies.json`` file is NEVER modified — humans
   review the proposals before any demotion.

Exit codes:

* ``0`` — all strategies passed, or ``--strict`` not set
* ``1`` — at least one strategy failed AND ``--strict`` is set

CLI::

    python scripts/validate_alphas_4q.py
    python scripts/validate_alphas_4q.py --strict
    python scripts/validate_alphas_4q.py --output /tmp/foo.json --strict
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure ``src/`` is importable without requiring a pip install.
_API_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _API_ROOT.parent
sys.path.insert(0, str(_API_ROOT / "src"))

from pfm.strategies_registry import (
    Strategy,
    register,
)
from pfm.strategies_registry import (
    get as _registry_get,
)

logger = logging.getLogger("pfm.validate_alphas_4q")


# ---------------------------------------------------------------------------
# Stress-test module loader
# ---------------------------------------------------------------------------


def _load_stress_module():
    """Import ``scripts/stress_test.py`` as a module (script lives outside pfm)."""
    path = _API_ROOT / "scripts" / "stress_test.py"
    spec = importlib.util.spec_from_file_location("pfm_stress_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tier filtering
# ---------------------------------------------------------------------------


DEPLOYABLE_TIERS = frozenset({"A_GOLD", "A_STRUCTURAL", "B_VALIDATED"})


def load_strategies_file(path: Path) -> list[dict[str, Any]]:
    """Read ``alpha_strategies.json`` and return the strategy list."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "strategies" in raw:
        return list(raw["strategies"])
    msg = (
        f"Unexpected alpha_strategies.json layout at {path}: "
        f"expected list or dict with 'strategies' key."
    )
    raise ValueError(msg)


def filter_deployable(strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only strategies whose tier is in ``DEPLOYABLE_TIERS``."""
    return [s for s in strategies if s.get("tier") in DEPLOYABLE_TIERS]


# ---------------------------------------------------------------------------
# Strategy resolution: registry-first, fallback to synthetic buy-and-hold
# ---------------------------------------------------------------------------


def _buy_and_hold_signal_factory(name: str):
    def _sig(prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index, name=f"signal:{name}")

    return _sig


def resolve_strategy(pair_id: str) -> Strategy:
    """Return a Strategy for ``pair_id``.

    Preference order:

    1. Already-registered Strategy whose ``name == pair_id`` (real strategy).
    2. Fallback: register and return a buy-and-hold strategy whose
       ``name == pair_id``. ``stress_test.run_stress`` seeds its synthetic
       price series from ``strategy.name`` so each pair_id yields a
       reproducible distinct path.
    """
    try:
        return _registry_get(pair_id)
    except KeyError:
        strat = Strategy(name=pair_id, signal=_buy_and_hold_signal_factory(pair_id))
        register(strat)
        return strat


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _quarter_sharpes(report: dict[str, Any]) -> list[float]:
    """Extract per-quarter Sharpe values in order."""
    rows = sorted(report.get("quarter_rows", []), key=lambda r: r.get("quarter", 0))
    return [float(r.get("sharpe", 0.0)) for r in rows]


def _sign_flip_detected(report: dict[str, Any]) -> bool:
    """True iff any quarter row has a sign-flip vs the full-sample sign."""
    full_sign = int(report.get("full_sample", {}).get("sign", 0))
    if full_sign == 0:
        return False
    for r in report.get("quarter_rows", []):
        sign = int(r.get("sign", 0))
        if sign != 0 and sign != full_sign:
            return True
    return False


def aggregate_row(strat: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """One-row aggregation of a stress report for printing/storage."""
    sharpes = _quarter_sharpes(report)
    # Pad to 4 for display, even if quarters parameter differs.
    sharpes = (sharpes + [0.0, 0.0, 0.0, 0.0])[:4]
    verdict = report.get("verdict", "UNKNOWN")
    return {
        "pair_id": strat.get("pair_id", report.get("strategy", "?")),
        "tier": strat.get("tier", "?"),
        "q1_sharpe": float(sharpes[0]),
        "q2_sharpe": float(sharpes[1]),
        "q3_sharpe": float(sharpes[2]),
        "q4_sharpe": float(sharpes[3]),
        "sign_flip": _sign_flip_detected(report),
        "full_sharpe": float(report.get("full_sample", {}).get("sharpe", 0.0)),
        "deflated_sharpe": float(report.get("full_sample", {}).get("deflated_sharpe", 0.0)),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_validation(
    strategies_path: Path,
    *,
    start: str = "2024-01",
    quarters: int = 4,
    sharpe_floor: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run stress on every deployable strategy.

    Returns ``(rows, failed_rows)``: full aggregated table and the subset
    of rows whose verdict is FAIL.
    """
    stress = _load_stress_module()

    all_strats = load_strategies_file(strategies_path)
    deployable = filter_deployable(all_strats)
    if not deployable:
        logger.warning("No deployable strategies found in %s", strategies_path)
        return [], []

    start_ts = stress.parse_start(start)

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for s in deployable:
        pair_id = s.get("pair_id")
        if not pair_id:
            logger.warning("Strategy missing pair_id; skipping: %r", s)
            continue
        strat_obj = resolve_strategy(pair_id)
        try:
            report = stress.run_stress(
                strat_obj,
                start=start_ts,
                quarters=quarters,
                sharpe_floor=sharpe_floor,
            )
        except Exception as exc:  # pragma: no cover - defensive only
            logger.exception("run_stress crashed for %s: %s", pair_id, exc)
            row = {
                "pair_id": pair_id,
                "tier": s.get("tier"),
                "q1_sharpe": 0.0,
                "q2_sharpe": 0.0,
                "q3_sharpe": 0.0,
                "q4_sharpe": 0.0,
                "sign_flip": False,
                "full_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "verdict": "ERROR",
            }
            rows.append(row)
            failed.append({**row, "error": str(exc)})
            continue

        row = aggregate_row(s, report)
        rows.append(row)
        if row["verdict"] != "PASS":
            # Carry through the human-readable fail reasons for the proposals file.
            quarter_reasons = [
                {
                    "quarter": int(r.get("quarter", 0)),
                    "sharpe": float(r.get("sharpe", 0.0)),
                    "sign": int(r.get("sign", 0)),
                    "fail_reason": str(r.get("fail_reason", "")),
                }
                for r in report.get("quarter_rows", [])
                if r.get("fail")
            ]
            failed.append({**row, "failing_quarters": quarter_reasons})

    return rows, failed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_summary(rows: list[dict[str, Any]]) -> str:
    """Human-readable table for stdout."""
    if not rows:
        return "No deployable strategies to stress test."

    header = (
        f"{'pair_id':<48}  {'tier':<14}  "
        f"{'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6}  "
        f"{'flip':>4}  {'DSR':>7}  verdict"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        pid = str(r["pair_id"])
        if len(pid) > 46:
            pid = pid[:43] + "..."
        flip = "Y" if r["sign_flip"] else "n"
        lines.append(
            f"{pid:<48}  {r['tier']:<14}  "
            f"{r['q1_sharpe']:>6.2f} {r['q2_sharpe']:>6.2f} "
            f"{r['q3_sharpe']:>6.2f} {r['q4_sharpe']:>6.2f}  "
            f"{flip:>4}  {r['deflated_sharpe']:>7.3f}  {r['verdict']}"
        )
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    n_fail = len(rows) - n_pass
    lines.append("-" * len(header))
    lines.append(f"TOTAL: {len(rows)}   PASS: {n_pass}   FAIL: {n_fail}")
    return "\n".join(lines)


def write_proposals(failed: list[dict[str, Any]], path: Path) -> None:
    """Write FAILED strategies to an anti-alpha-proposals JSON file.

    The file is for HUMAN REVIEW: it is NOT auto-merged back into
    ``alpha_strategies.json``.
    """
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "note": (
            "Auto-generated by scripts/validate_alphas_4q.py. "
            "Human review required before demoting any tier in "
            "web/data/alpha_strategies.json."
        ),
        "count": len(failed),
        "proposals": failed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DEFAULT_STRATEGIES_PATH = _REPO_ROOT / "web" / "data" / "alpha_strategies.json"
DEFAULT_PROPOSALS_PATH = Path("/tmp/anti-alpha-proposals.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 4-quarter stress on every deployable alpha and emit "
            "anti-alpha proposals for human review."
        )
    )
    parser.add_argument(
        "--strategies",
        type=Path,
        default=DEFAULT_STRATEGIES_PATH,
        help=f"Path to alpha_strategies.json (default: {DEFAULT_STRATEGIES_PATH}).",
    )
    parser.add_argument(
        "--start",
        default="2024-01",
        help="Quarter-window start YYYY-MM or YYYY-MM-DD (default: 2024-01).",
    )
    parser.add_argument(
        "--quarters",
        type=int,
        default=4,
        help="Number of disjoint quarter windows (default: 4).",
    )
    parser.add_argument(
        "--sharpe-floor",
        type=float,
        default=0.5,
        help="Per-quarter Sharpe failure threshold (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROPOSALS_PATH,
        help=(f"Output path for anti-alpha proposals JSON (default: {DEFAULT_PROPOSALS_PATH})."),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any strategy fails (CI gate).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)

    if not args.strategies.exists():
        logger.error("alpha_strategies.json not found at %s", args.strategies)
        return 2

    rows, failed = run_validation(
        args.strategies,
        start=args.start,
        quarters=args.quarters,
        sharpe_floor=args.sharpe_floor,
    )

    print(format_summary(rows))

    write_proposals(failed, args.output)
    print()
    print(f"Wrote {len(failed)} anti-alpha proposal(s) to {args.output} (human review required).")

    if args.strict and failed:
        logger.warning("Strict mode: %d strategy(ies) failed the 4Q stress test.", len(failed))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
