"""Regenerate web/data/alpha_strategies.json by running each pair through the
full quant rigor pipeline: cointegration -> walk-forward backtest -> BH-FDR
over all p-values -> 4Q stability gate -> alpha_card_verdict.

Usage:
    python -m scripts.regenerate_alpha_tiers --max-runtime 600 --output backup

Modes:
    --output update  : overwrite alpha_strategies.json
    --output backup  : write to alpha_strategies.json.regenerated.{timestamp}.json
    --output dry-run : print summary, write nothing

The heavy lifting lives in ``pfm.alpha_tier_regen.regenerate_alpha_tiers``;
this script is a thin CLI wrapper that lets ops run the harness from a shell
without touching the API server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure ``src/`` is importable without requiring a pip install of the package.
_API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_API_ROOT / "src"))

from pfm.alpha_tier_regen import (
    DEFAULT_ALPHA_PATH,
    DEFAULT_FETCH_CONCURRENCY,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_N_FOLDS,
    DEFAULT_PERM_ITERS,
    DEFAULT_REPORT_DIR,
    regenerate_alpha_tiers,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate alpha_strategies.json with real walk-forward tiers."
    )
    parser.add_argument(
        "--output",
        choices=["update", "backup", "dry-run"],
        default="backup",
        help="Where to write the regenerated JSON. Defaults to a timestamped backup.",
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=600,
        help="Best-effort wall-clock budget in seconds (default: 600).",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=f"Lookback window in days (default: {DEFAULT_HISTORY_DAYS}).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f"Walk-forward folds (default: {DEFAULT_N_FOLDS}).",
    )
    parser.add_argument(
        "--perm-iters",
        type=int,
        default=DEFAULT_PERM_ITERS,
        help=f"Permutation iterations (default: {DEFAULT_PERM_ITERS}).",
    )
    parser.add_argument(
        "--fetch-concurrency",
        type=int,
        default=DEFAULT_FETCH_CONCURRENCY,
        help=f"Polymarket fetch concurrency (default: {DEFAULT_FETCH_CONCURRENCY}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42).",
    )
    parser.add_argument(
        "--alpha-path",
        type=Path,
        default=DEFAULT_ALPHA_PATH,
        help="Override path to alpha_strategies.json.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Where to write the markdown report.",
    )
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    out = await regenerate_alpha_tiers(
        alpha_path=args.alpha_path,
        output_mode=args.output,
        max_runtime_seconds=args.max_runtime,
        history_days=args.history_days,
        n_folds=args.n_folds,
        perm_iters=args.perm_iters,
        fetch_concurrency=args.fetch_concurrency,
        seed=args.seed,
        report_dir=args.report_dir,
    )
    summary = out["summary"]
    print(
        json.dumps(
            {
                "summary": summary,
                "written_path": out.get("written_path"),
                "report_path": out.get("report_path"),
            },
            indent=2,
            default=str,
        )
    )
    if summary.get("n_errors", 0) and summary["n_processed"] == 0:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
