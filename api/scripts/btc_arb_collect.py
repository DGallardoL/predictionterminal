"""Long-horizon latency-arb collector for Polymarket BTC up/down 5m markets.

Loops across multiple consecutive 5-min windows (auto-discovering each new
market when the prior one resolves), polling Binance bookTicker + Polymarket
CLOB midpoint at ~2 Hz, then runs aggregate lag/arb statistics across the
full pooled sample set.

A 25-min run yields ~3000 samples (vs ~360 in a single window), which
gives much tighter confidence intervals on the true Binance->Polymarket lag
and on the empirical hit-rate of "Binance moved, therefore Polymarket
will follow within 5s" trade signals.

Usage:
    python btc_arb_collect.py --duration 1500 --hz 2 --out /tmp/btc_arb_long.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from btc_updown_monitor import (
    Sample,
    discover_active_market,
    get_binance_book,
    get_clob_midpoint,
)


def _parse_iso(s: str) -> float:
    """Parse Polymarket ISO timestamp (e.g. '2026-05-02T18:25:00Z') -> unix seconds."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def _next_unique_path(base: str) -> str:
    """If `base` exists, append _2, _3, ... before extension to keep history."""
    p = Path(base)
    if not p.exists():
        return str(p)
    stem, suffix = p.stem, p.suffix
    parent = p.parent
    i = 2
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return str(cand)
        i += 1


def corr(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy)


def cross_corr_lag(binance: list[float], poly: list[float], max_lag: int = 14) -> dict:
    """Cross-correlation lag analysis on first-differenced series.

    Positive lag = polymarket lags binance (k samples = k/hz seconds).
    """
    if len(binance) < 30 or len(poly) < 30:
        return {"error": "insufficient samples", "n": len(binance)}
    db = [binance[i] - binance[i - 1] for i in range(1, len(binance))]
    dp = [poly[i] - poly[i - 1] for i in range(1, len(poly))]
    by_lag: list[tuple[int, float]] = []
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x, y = db[: len(db) - k] if k else db[:], dp[k:] if k else dp[:]
        else:
            x, y = db[-k:], dp[: len(dp) + k]
        if len(x) < 5:
            continue
        by_lag.append((k, corr(x, y)))
    if not by_lag:
        return {"error": "no lag windows", "n": len(binance)}
    by_lag_sorted = sorted(by_lag, key=lambda kv: -kv[1])
    return {
        "n": len(binance),
        "best_lag_samples": by_lag_sorted[0][0],
        "best_corr": round(by_lag_sorted[0][1], 4),
        "by_lag": [(k, round(c, 4)) for k, c in sorted(by_lag)],
    }


def find_trade_opportunities(
    samples: list[Sample],
    hz: float,
    binance_thresh: float = 20.0,
    poly_thresh: float = 0.03,
    look_ahead_s: float = 5.0,
    look_back_s: float = 2.0,
) -> list[dict]:
    """Empirically detectable arb moments.

    A "trade opportunity" is a sample i where:
      - |binance(i) - binance(i - look_back_s)| > binance_thresh ($20)
    The "outcome" is whether polymarket moved in the predicted direction
    by more than poly_thresh (3pp) within look_ahead_s seconds.

    A signal is "BUY UP" when Binance went up; "BUY DOWN" when Binance went down.
    """
    rows = [s for s in samples if s.poly_up_5m is not None]
    if len(rows) < 10:
        return []
    back = max(1, int(round(look_back_s * hz)))
    fwd = max(1, int(round(look_ahead_s * hz)))
    out: list[dict] = []
    for i in range(back, len(rows) - fwd):
        db = rows[i].binance_mid - rows[i - back].binance_mid
        if abs(db) <= binance_thresh:
            continue
        pm_now = rows[i].poly_up_5m
        # find max forward poly move (signed) in the look-ahead window
        forward_window = [rows[j].poly_up_5m for j in range(i, i + fwd + 1)]
        signed_fwd = [pm - pm_now for pm in forward_window]
        # signal direction: up if binance up
        signal = "BUY_UP" if db > 0 else "BUY_DOWN"
        # the maximum favorable poly move in the predicted direction
        if db > 0:
            best_fwd = max(signed_fwd)
        else:
            best_fwd = min(signed_fwd)
        # also record the end-of-window move (5s out) for HR check
        end_fwd = signed_fwd[-1]
        # hit if absolute movement in predicted direction > poly_thresh
        hit = (db > 0 and best_fwd > poly_thresh) or (db < 0 and best_fwd < -poly_thresh)
        out.append(
            {
                "t": rows[i].t,
                "iso": datetime.fromtimestamp(rows[i].t, tz=UTC).isoformat(),
                "binance_mid": rows[i].binance_mid,
                "binance_move_2s": round(db, 2),
                "signal": signal,
                "poly_up_at_t": round(pm_now, 4),
                "poly_best_move_fwd": round(best_fwd, 4),
                "poly_end_move_5s": round(end_fwd, 4),
                "hit": hit,
            }
        )
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def collect_window(
    market: dict,
    deadline: float,
    hz: float,
    client: httpx.Client,
) -> list[Sample]:
    """Poll a single 5m market until min(market endDate, deadline)."""
    token_ids = json.loads(market["clobTokenIds"])
    up_token = token_ids[0]
    end_unix = _parse_iso(market["endDate"])
    window_deadline = min(end_unix, deadline)
    period = 1.0 / hz
    samples: list[Sample] = []
    print(
        f"  polling {market['slug']} until "
        f"{datetime.fromtimestamp(window_deadline, tz=UTC).strftime('%H:%M:%S')}Z",
        file=sys.stderr,
    )
    while time.time() < window_deadline:
        loop_start = time.time()
        results: dict[str, object | None] = {"binance": None, "poly5": None}

        def _b(results: dict[str, object | None] = results) -> None:
            results["binance"] = get_binance_book(client)

        def _p5(results: dict[str, object | None] = results) -> None:
            results["poly5"] = get_clob_midpoint(up_token, client)

        ths = [threading.Thread(target=fn) for fn in (_b, _p5)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout=2.5)
        if results["binance"]:
            mid, bid, ask = results["binance"]  # type: ignore[misc]
            samples.append(
                Sample(
                    t=loop_start,
                    binance_mid=mid,
                    binance_bid=bid,
                    binance_ask=ask,
                    poly_up_5m=results["poly5"],  # type: ignore[arg-type]
                    poly_up_15m=None,
                )
            )
            if len(samples) % 30 == 0:
                p5 = f"{results['poly5']:.3f}" if isinstance(results["poly5"], float) else "—"
                print(
                    f"    n={len(samples):4d}  BTC={mid:>10,.2f}  poly_up={p5}",
                    file=sys.stderr,
                )
        elapsed = time.time() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)
    return samples


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=1500)
    p.add_argument("--hz", type=float, default=2.0)
    p.add_argument("--out", default="/tmp/btc_arb_long.json")
    p.add_argument("--binance-thresh", type=float, default=20.0)
    p.add_argument("--poly-thresh", type=float, default=0.03)
    args = p.parse_args()

    out_path = _next_unique_path(args.out)
    client = httpx.Client(timeout=3.0)
    t_start = time.time()
    deadline = t_start + args.duration
    print(
        f"long-horizon collector: duration={args.duration}s hz={args.hz} -> {out_path}",
        file=sys.stderr,
    )

    windows: list[dict] = []
    all_samples: list[Sample] = []

    while time.time() < deadline:
        market = discover_active_market("5m", client)
        if not market:
            print("  no active 5m market — sleeping 5s and retrying", file=sys.stderr)
            time.sleep(5)
            continue
        # Skip if endDate is already past (rare race condition)
        try:
            end_unix = _parse_iso(market["endDate"])
        except Exception:
            print(
                f"  WARN: cannot parse endDate {market.get('endDate')!r}; skipping",
                file=sys.stderr,
            )
            time.sleep(5)
            continue
        if end_unix <= time.time():
            print(
                f"  market {market['slug']} already ended; waiting 3s for next",
                file=sys.stderr,
            )
            time.sleep(3)
            continue
        print(f"\n[window {len(windows) + 1}] {market['slug']}", file=sys.stderr)
        win_samples = collect_window(market, deadline, args.hz, client)
        windows.append(
            {
                "slug": market["slug"],
                "endDate": market["endDate"],
                "n_samples": len(win_samples),
                "started": (
                    datetime.fromtimestamp(win_samples[0].t, tz=UTC).isoformat()
                    if win_samples
                    else None
                ),
            }
        )
        all_samples.extend(win_samples)
        print(
            f"  window done: {len(win_samples)} samples (total {len(all_samples)})",
            file=sys.stderr,
        )

    print(
        f"\ncollected {len(all_samples)} samples across {len(windows)} window(s)",
        file=sys.stderr,
    )

    # Per-window lag
    per_window_lag: list[dict] = []
    cursor = 0
    for w in windows:
        n = w["n_samples"]
        sub = all_samples[cursor : cursor + n]
        cursor += n
        bn = [s.binance_mid for s in sub if s.poly_up_5m is not None]
        pm = [s.poly_up_5m for s in sub if s.poly_up_5m is not None]
        per_window_lag.append({"slug": w["slug"], "lag": cross_corr_lag(bn, pm)})

    # Aggregate lag (pool all windows)
    bn_all = [s.binance_mid for s in all_samples if s.poly_up_5m is not None]
    pm_all = [s.poly_up_5m for s in all_samples if s.poly_up_5m is not None]
    aggregate_lag = cross_corr_lag(bn_all, pm_all, max_lag=20)

    # Trade opportunities
    opps = find_trade_opportunities(
        all_samples,
        hz=args.hz,
        binance_thresh=args.binance_thresh,
        poly_thresh=args.poly_thresh,
    )
    n_opps = len(opps)
    n_hits = sum(1 for o in opps if o["hit"])
    hit_rate = (n_hits / n_opps) if n_opps else None
    ci_low, ci_high = wilson_ci(n_hits, n_opps) if n_opps else (None, None)
    best5 = sorted(opps, key=lambda o: -abs(o["binance_move_2s"]))[:5]

    out = {
        "started": datetime.fromtimestamp(t_start, tz=UTC).isoformat(),
        "duration_s": args.duration,
        "hz": args.hz,
        "n_windows": len(windows),
        "n_samples": len(all_samples),
        "windows": windows,
        "per_window_lag": per_window_lag,
        "aggregate_lag": aggregate_lag,
        "trade_opportunities": opps,
        "n_opportunities": n_opps,
        "n_hits": n_hits,
        "hit_rate": hit_rate,
        "hit_rate_ci95": (ci_low, ci_high),
        "binance_thresh": args.binance_thresh,
        "poly_thresh": args.poly_thresh,
        "best_5_moments": best5,
        "samples": [asdict(s) for s in all_samples],
    }
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}", file=sys.stderr)

    # Pretty-print final report
    print("\n=== Aggregate lag ===", file=sys.stderr)
    bl = aggregate_lag.get("best_lag_samples")
    bc = aggregate_lag.get("best_corr")
    if bl is not None:
        print(
            f"  best_lag = {bl} samples ({bl / args.hz:.2f}s)  corr={bc}  n={aggregate_lag.get('n')}",
            file=sys.stderr,
        )
    print("\n=== Trade opportunities ===", file=sys.stderr)
    print(
        f"  n_opps={n_opps}  n_hits={n_hits}  hit_rate={hit_rate}  95% CI=({ci_low}, {ci_high})",
        file=sys.stderr,
    )
    print("\n=== Top 5 best moments ===", file=sys.stderr)
    for m in best5:
        print(
            f"  {m['iso']}  {m['signal']:8s}  Δbtc={m['binance_move_2s']:+8.2f}  "
            f"Δpoly_5s={m['poly_end_move_5s']:+.4f}  hit={m['hit']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
