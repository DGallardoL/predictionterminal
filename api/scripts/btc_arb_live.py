"""Live "fair price" overlay for the Polymarket BTC up/down latency-arb.

For each second of `--duration`:
  1. Pull Binance BTCUSDT bookTicker midpoint.
  2. Pull Polymarket Up-token CLOB midpoint for the active 5m market.
  3. Compute the GBM fair Up-probability using BTC_0 (anchored to the
     first Binance tick after the window opened, as a proxy for the
     Chainlink reference at start) and the seconds remaining in the
     current 5-min window.
  4. Log the edge (fair_up - poly_up_mid) in basis points and a discrete
     BUY_UP / SELL_UP / HOLD signal.

At the end:
  - prints median |edge| (bps), max |edge| (bps), signal counts, and the
    3 largest |edge| moments,
  - dumps everything to /tmp/btc_arb_signals.json.

The BTC_0 reference is updated automatically when the 5-min window
rolls over and a new Polymarket market becomes active.

Usage:
    PYTHONPATH=api/src api/.venv/bin/python api/scripts/btc_arb_live.py \\
        --duration 240
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

# Make `pfm` importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Reuse the existing market discovery + IO helpers.
from btc_updown_monitor import (
    discover_active_market,
    get_binance_book,
    get_clob_midpoint,
)

from pfm.btc_arb import (
    arb_signal,
    compute_fair_up_prob,
    realized_volatility,
)

WINDOW_SECONDS = 300  # 5-minute markets


def _window_bounds_for(now_unix: float) -> tuple[int, int]:
    """Return (start_unix, end_unix) of the current 5-min window."""
    end = (int(now_unix) // WINDOW_SECONDS + 1) * WINDOW_SECONDS
    return end - WINDOW_SECONDS, end


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=240)
    p.add_argument(
        "--vol-ann", type=float, default=0.65, help="Annualized BTC vol (default 0.65 = 65 pct)"
    )
    p.add_argument(
        "--edge-threshold",
        type=float,
        default=0.03,
        help="Probability gap to flag BUY_UP / SELL_UP",
    )
    p.add_argument("--out", default="/tmp/btc_arb_signals.json")
    args = p.parse_args()

    client = httpx.Client(timeout=3.0)

    # Discover the active 5m market and its Up-token.
    print("discovering active 5m market...", file=sys.stderr)
    m5 = discover_active_market("5m", client)
    if not m5:
        print("ERROR: no active btc-updown 5m market", file=sys.stderr)
        return 1
    token_ids = json.loads(m5["clobTokenIds"])
    up_token = token_ids[0]
    current_slug = m5["slug"]
    print(f"5m market: {current_slug} ends {m5['endDate']}", file=sys.stderr)

    # Window state.
    _win_start, win_end = _window_bounds_for(time.time())
    btc_0: float | None = None  # anchored on the first Binance tick of the window
    btc_history: list[float] = []  # for live realized-vol estimate

    rows: list[dict] = []
    t_start = time.time()
    deadline = t_start + args.duration
    print(f"polling at 1 Hz for {args.duration}s ...", file=sys.stderr)

    while time.time() < deadline:
        loop_t0 = time.time()

        # Window roll-over: re-discover market, reset anchor.
        if loop_t0 >= win_end:
            _win_start, win_end = _window_bounds_for(loop_t0)
            btc_0 = None
            btc_history.clear()
            m5_new = discover_active_market("5m", client)
            if m5_new and m5_new["slug"] != current_slug:
                token_ids = json.loads(m5_new["clobTokenIds"])
                up_token = token_ids[0]
                current_slug = m5_new["slug"]
                print(f"  -> rolled to {current_slug}", file=sys.stderr)

        book = get_binance_book(client)
        poly_up = get_clob_midpoint(up_token, client)

        if book is None:
            time.sleep(max(0.0, 1.0 - (time.time() - loop_t0)))
            continue
        btc_mid, _bid, _ask = book

        # Anchor BTC_0 to the first Binance tick observed after window open.
        if btc_0 is None:
            btc_0 = btc_mid

        btc_history.append(btc_mid)
        seconds_remaining = max(0.0, win_end - loop_t0)

        fair_up = compute_fair_up_prob(
            btc_t=btc_mid,
            btc_0=btc_0,
            seconds_remaining=seconds_remaining,
            vol_ann=args.vol_ann,
        )

        signal = "HOLD"
        edge_bps: float | None = None
        if poly_up is not None:
            edge = fair_up - poly_up
            edge_bps = edge * 10_000.0
            signal = arb_signal(poly_up, fair_up, edge_threshold=args.edge_threshold)

        rows.append(
            {
                "t": loop_t0,
                "t_rel": loop_t0 - t_start,
                "slug": current_slug,
                "btc": btc_mid,
                "btc_0": btc_0,
                "seconds_remaining": seconds_remaining,
                "fair_up": fair_up,
                "poly_up": poly_up,
                "edge_bps": edge_bps,
                "signal": signal,
            }
        )

        if len(rows) % 10 == 0:
            poly_str = f"{poly_up:.3f}" if poly_up is not None else "—"
            edge_str = f"{edge_bps:+7.1f}" if edge_bps is not None else "    —"
            print(
                f"  t+{loop_t0 - t_start:6.1f}s  BTC={btc_mid:>10,.2f}  "
                f"fair={fair_up:.3f}  poly={poly_str}  edge={edge_str}bps  "
                f"trem={seconds_remaining:5.1f}s  sig={signal}",
                file=sys.stderr,
            )

        elapsed = time.time() - loop_t0
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

    # ---- Aggregate ----
    edged = [r for r in rows if r["edge_bps"] is not None]
    abs_edges = [abs(r["edge_bps"]) for r in edged]
    counts = {"BUY_UP": 0, "SELL_UP": 0, "HOLD": 0}
    for r in rows:
        counts[r["signal"]] = counts.get(r["signal"], 0) + 1

    median_abs = statistics.median(abs_edges) if abs_edges else 0.0
    max_abs = max(abs_edges) if abs_edges else 0.0
    top3 = sorted(edged, key=lambda r: -abs(r["edge_bps"]))[:3]

    # Realized vol over the run, in case we want to sanity-check vol_ann.
    rv = realized_volatility([r["btc"] for r in rows], dt_seconds=1.0) if len(rows) > 5 else 0.0

    summary = {
        "duration_s": args.duration,
        "n_samples": len(rows),
        "n_with_poly": len(edged),
        "vol_ann_used": args.vol_ann,
        "realized_vol_observed": rv,
        "edge_threshold": args.edge_threshold,
        "median_abs_edge_bps": median_abs,
        "max_abs_edge_bps": max_abs,
        "signal_counts": counts,
        "top3_edges": [
            {
                "t_rel": r["t_rel"],
                "slug": r["slug"],
                "btc": r["btc"],
                "btc_0": r["btc_0"],
                "seconds_remaining": r["seconds_remaining"],
                "fair_up": r["fair_up"],
                "poly_up": r["poly_up"],
                "edge_bps": r["edge_bps"],
                "signal": r["signal"],
            }
            for r in top3
        ],
    }

    out = {"summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    print("\n=== summary ===", file=sys.stderr)
    print(f"  samples: {len(rows)} (with poly mid: {len(edged)})", file=sys.stderr)
    print(f"  median |edge|: {median_abs:.1f} bps", file=sys.stderr)
    print(f"  max |edge|:    {max_abs:.1f} bps", file=sys.stderr)
    print(f"  observed realized vol (annualized): {rv:.3f}", file=sys.stderr)
    print(f"  signals: {counts}", file=sys.stderr)
    for i, r in enumerate(top3, 1):
        print(
            f"  top{i}: t+{r['t_rel']:.1f}s  BTC {r['btc']:.2f} (BTC_0 {r['btc_0']:.2f}) "
            f"trem={r['seconds_remaining']:.0f}s  fair={r['fair_up']:.3f} "
            f"poly={r['poly_up']:.3f}  edge={r['edge_bps']:+.1f} bps  ({r['signal']})",
            file=sys.stderr,
        )
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
