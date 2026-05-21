"""CLI: alpha-hunter sweep over a list of factor IDs.

Usage:

    python api/scripts/run_alpha_hunter.py \
        --start 2025-09-01 --end 2026-04-30 \
        --themes politics,macro,crypto \
        --max-factors 60 --max-pairs 800 \
        --out /tmp/alpha_run.json

Or pin a specific list:

    python api/scripts/run_alpha_hunter.py \
        --factor-ids "amzn_largest_jun,saudi_aramco_largest,bp_acquired,..."

The script fetches price histories from Polymarket (Kalshi where the
factor's ``source: kalshi``), runs the full alpha-hunter gauntlet
(cointegration → backtest → permutation), and writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pfm.alpha_hunter import run_alpha_hunter
from pfm.factors import load_factors
from pfm.sources.polymarket import PolymarketClient, fetch_factor_history

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")


def _fetch_one(
    client: PolymarketClient, fid: str, slug: str, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[str, pd.Series | None]:
    try:
        df = fetch_factor_history(client, slug, start, end)
        if df.empty:
            return fid, None
        s = df["price"].rename(fid)
        return fid, s
    except Exception as e:
        logging.warning("fetch failed for %s (slug=%s): %s", fid, slug, e)
        return fid, None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--factors-yml", default=str(ROOT / "src" / "pfm" / "factors.yml"))
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument(
        "--themes",
        default="",
        help="Comma-list of themes to include (default: all polymarket factors)",
    )
    p.add_argument(
        "--factor-ids", default="", help="Comma-list of explicit factor IDs (overrides --themes)"
    )
    p.add_argument("--max-factors", type=int, default=80)
    p.add_argument("--max-pairs", type=int, default=2000)
    p.add_argument("--adf-threshold", type=float, default=0.05)
    p.add_argument("--oos-sharpe-floor", type=float, default=0.5)
    p.add_argument("--perm-threshold", type=float, default=1.0)
    p.add_argument("--perm-iters", type=int, default=200)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    factors = load_factors(Path(args.factors_yml))

    if args.factor_ids:
        wanted = [x.strip() for x in args.factor_ids.split(",") if x.strip()]
        selected = [factors[fid] for fid in wanted if fid in factors]
        missing = set(wanted) - {f.id for f in selected}
        if missing:
            print(
                f"WARN: {len(missing)} requested ids not in factors.yml: {sorted(missing)[:5]}...",
                file=sys.stderr,
            )
    else:
        themes = {t.strip() for t in args.themes.split(",") if t.strip()}
        selected = [
            f
            for f in factors.values()
            if f.source == "polymarket" and (not themes or f.theme in themes)
        ]

    selected = selected[: args.max_factors]
    print(f"selected {len(selected)} factors", file=sys.stderr)

    client = PolymarketClient(
        gamma_url="https://gamma-api.polymarket.com",
        clob_url="https://clob.polymarket.com",
    )

    t0 = time.perf_counter()
    prices: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_fetch_one, client, f.id, f.slug, start, end): f.id for f in selected}
        for fut in as_completed(futs):
            fid, s = fut.result()
            if s is not None and len(s.dropna()) >= 60:
                prices[fid] = s
    fetch_secs = time.perf_counter() - t0
    print(f"fetched {len(prices)}/{len(selected)} histories in {fetch_secs:.1f}s", file=sys.stderr)

    if len(prices) < 2:
        print("ERROR: <2 usable histories — nothing to pair", file=sys.stderr)
        return 1

    report = run_alpha_hunter(
        prices,
        adf_threshold=args.adf_threshold,
        oos_sharpe_floor=args.oos_sharpe_floor,
        perm_oos_sharpe_threshold=args.perm_threshold,
        perm_n_iters=args.perm_iters,
        max_pairs=args.max_pairs,
    )

    print("\nALPHA HUNTER COMPLETE", file=sys.stderr)
    print(f"  factors:        {report.n_factors}", file=sys.stderr)
    print(f"  pairs total:    {report.n_pairs_total}", file=sys.stderr)
    print(f"  passed ADF:     {report.n_pairs_passed_adf}", file=sys.stderr)
    print(f"  perm tested:    {report.n_pairs_perm_tested}", file=sys.stderr)
    print(f"  REAL_ALPHA:     {report.n_real_alpha}", file=sys.stderr)
    print(f"  hunter runtime: {report.runtime_seconds:.1f}s", file=sys.stderr)

    print(
        f"\nTop {min(15, len(report.hits))} hits (REAL_ALPHA first, then by OOS Sharpe):",
        file=sys.stderr,
    )
    for h in report.hits[:15]:
        pp = f"{h.perm_p:.3f}" if h.perm_p is not None else "n/a"
        hl = f"{h.half_life_days:.2f}d" if h.half_life_days is not None else "n/a"
        print(
            f"  [{h.verdict:11}] {h.a_id:36} ↔ {h.b_id:36} "
            f"adf={h.adf_pvalue:.3f} hl={hl} oos_sh={h.oos_sharpe:+.2f} "
            f"perm_p={pp}",
            file=sys.stderr,
        )

    out = {
        "n_factors": report.n_factors,
        "n_pairs_total": report.n_pairs_total,
        "n_pairs_passed_adf": report.n_pairs_passed_adf,
        "n_pairs_perm_tested": report.n_pairs_perm_tested,
        "n_real_alpha": report.n_real_alpha,
        "fetch_seconds": fetch_secs,
        "hunter_seconds": report.runtime_seconds,
        "factor_ids": sorted(prices.keys()),
        "hits": [
            {
                "a_id": h.a_id,
                "b_id": h.b_id,
                "verdict": h.verdict,
                "n_obs": h.n_obs,
                "adf_pvalue": h.adf_pvalue,
                "half_life_days": h.half_life_days,
                "beta_hedge": h.beta_hedge,
                "oos_sharpe": h.oos_sharpe,
                "full_sharpe": h.full_sharpe,
                "perm_p": h.perm_p,
                "perm_real_sharpe": h.perm_real_sharpe,
            }
            for h in report.hits
        ],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
