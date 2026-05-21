"""Live latency-arbitrage monitor for Polymarket BTC up/down 5m & 15m markets.

Polymarket "BTC Up or Down" markets resolve via Chainlink BTC/USD stream
at the end of a 5-min or 15-min window. The Chainlink stream lags spot
markets (Binance, Coinbase) by ~1-3 seconds. If Polymarket traders react
to Chainlink rather than Binance, there's a lead-lag edge: Binance moves
first, and Polymarket up-probability adjusts seconds later.

This monitor:
1. Auto-discovers the latest active 5m + 15m BTC up/down markets via the
   Polymarket Gamma API.
2. Polls Polymarket CLOB midpoint + Binance bookTicker at ~2 Hz.
3. Logs (timestamp, binance_btc_mid, poly_up_mid_5m, poly_up_mid_15m).
4. After --duration seconds, computes:
   - Δ-correlation between binance and polymarket
   - Empirical lag (k that maximizes Δbinance ↔ Δpolymarket cross-corr)
   - Largest |Δbinance| moves and how Polymarket reacted (or didn't)

Usage:
    python btc_updown_monitor.py --duration 180 --hz 2 --out /tmp/btc_arb.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BINANCE = "https://api.binance.com"


@dataclass
class Sample:
    t: float  # unix seconds (wall-clock)
    binance_mid: float  # midpoint of Binance bookTicker
    binance_bid: float
    binance_ask: float
    poly_up_5m: float | None  # CLOB midpoint of UP token, 5m market
    poly_up_15m: float | None  # CLOB midpoint of UP token, 15m market


def discover_active_market(window_label: str, client: httpx.Client) -> dict | None:
    """Find the currently-active btc-updown market for window_label ('5m' or '15m').

    Slugs are deterministic: btc-updown-{window}-{resolution_unix}, where
    resolution_unix is the END of the window (next round 5-min or 15-min
    boundary). Try the next 1-2 boundaries via direct slug fetch — the
    "active" one is the upcoming end-time.
    """
    now_unix = int(time.time())
    period = 300 if window_label == "5m" else 900
    # The market that is currently OPEN ends at the next boundary.
    next_end = ((now_unix // period) + 1) * period
    # Try next 3 slugs (the active one + a couple of upcoming).
    for offset in range(0, 3):
        end_unix = next_end + offset * period
        slug = f"btc-updown-{window_label}-{end_unix}"
        try:
            r = client.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=3.0)
            if r.status_code == 200:
                arr = r.json()
                if arr:
                    return arr[0]
        except Exception:
            pass
    return None


def get_clob_midpoint(token_id: str, client: httpx.Client) -> float | None:
    try:
        r = client.get(f"{CLOB}/midpoint", params={"token_id": token_id}, timeout=2.0)
        if r.status_code != 200:
            return None
        return float(r.json().get("mid", "nan"))
    except Exception:
        return None


def get_binance_book(client: httpx.Client) -> tuple[float, float, float] | None:
    try:
        r = client.get(
            f"{BINANCE}/api/v3/ticker/bookTicker", params={"symbol": "BTCUSDT"}, timeout=2.0
        )
        if r.status_code != 200:
            return None
        d = r.json()
        bid = float(d["bidPrice"])
        ask = float(d["askPrice"])
        return ((bid + ask) / 2.0, bid, ask)
    except Exception:
        return None


def empirical_lag(samples: list[Sample], poly_attr: str, max_lag: int = 10) -> dict:
    """Compute lead-lag of polymarket vs binance at integer-sample lags.

    Returns dict with the best lag (positive = polymarket lags binance).
    """
    rows = [s for s in samples if getattr(s, poly_attr) is not None]
    if len(rows) < 30:
        return {"error": "insufficient samples", "n": len(rows)}
    bn = [r.binance_mid for r in rows]
    pm = [getattr(r, poly_attr) for r in rows]
    db = [bn[i] - bn[i - 1] for i in range(1, len(bn))]
    dp = [pm[i] - pm[i - 1] for i in range(1, len(pm))]

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

    by_lag: list[tuple[int, float]] = []
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x, y = db[: len(db) - k], dp[k:]
        else:
            x, y = db[-k:], dp[: len(dp) + k]
        if len(x) < 5:
            continue
        by_lag.append((k, corr(x, y)))
    by_lag.sort(key=lambda kv: -kv[1])
    return {
        "n": len(rows),
        "best_lag_samples": by_lag[0][0] if by_lag else None,
        "best_corr": round(by_lag[0][1], 4) if by_lag else None,
        "by_lag": [(k, round(c, 4)) for k, c in sorted(by_lag)],
    }


def big_moves(samples: list[Sample], poly_attr: str, top_k: int = 5) -> list[dict]:
    """Find the largest binance moves and how polymarket responded after them."""
    rows = [s for s in samples if getattr(s, poly_attr) is not None]
    if len(rows) < 5:
        return []
    out = []
    for i in range(2, len(rows) - 2):
        db_2s = rows[i].binance_mid - rows[i - 2].binance_mid  # ~1-2s window
        # polymarket response over the next 2 samples (1-2s after)
        pm_now = getattr(rows[i], poly_attr)
        pm_after = getattr(rows[min(i + 2, len(rows) - 1)], poly_attr)
        out.append(
            {
                "t": rows[i].t,
                "binance_move_2s": db_2s,
                "binance_mid": rows[i].binance_mid,
                "poly_up_at_t": pm_now,
                "poly_up_after_2s": pm_after,
                "poly_response_2s": (pm_after - pm_now),
            }
        )
    out.sort(key=lambda d: -abs(d["binance_move_2s"]))
    return out[:top_k]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=120, help="seconds to monitor")
    p.add_argument("--hz", type=float, default=2.0, help="poll rate per second")
    p.add_argument("--out", default="/tmp/btc_arb.json")
    args = p.parse_args()

    client = httpx.Client(timeout=3.0)
    print("discovering active markets...", file=sys.stderr)
    m5 = discover_active_market("5m", client)
    m15 = discover_active_market("15m", client)
    if not m5 and not m15:
        print("ERROR: no active btc-updown markets found", file=sys.stderr)
        return 1
    info = {}
    if m5:
        t5 = json.loads(m5["clobTokenIds"])
        info["5m"] = {"slug": m5["slug"], "up_token": t5[0], "endDate": m5["endDate"]}
        print(f"5m  → {m5['slug']} ends {m5['endDate']}", file=sys.stderr)
    if m15:
        t15 = json.loads(m15["clobTokenIds"])
        info["15m"] = {"slug": m15["slug"], "up_token": t15[0], "endDate": m15["endDate"]}
        print(f"15m → {m15['slug']} ends {m15['endDate']}", file=sys.stderr)
    up_token_5m = t5[0] if m5 else None
    up_token_15m = t15[0] if m15 else None

    samples: list[Sample] = []
    period = 1.0 / args.hz
    t0 = time.time()
    deadline = t0 + args.duration
    print(f"polling at {args.hz} Hz for {args.duration}s …", file=sys.stderr)
    while time.time() < deadline:
        loop_start = time.time()
        # Poll all 3 endpoints in parallel via threads (httpx.Client is thread-safe)
        results = {"binance": None, "poly5": None, "poly15": None}

        def _b(results=results):
            results["binance"] = get_binance_book(client)

        def _p5(results=results):
            if up_token_5m:
                results["poly5"] = get_clob_midpoint(up_token_5m, client)

        def _p15(results=results):
            if up_token_15m:
                results["poly15"] = get_clob_midpoint(up_token_15m, client)

        ths = [threading.Thread(target=fn) for fn in (_b, _p5, _p15)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout=2.5)
        if results["binance"]:
            mid, bid, ask = results["binance"]
            samples.append(
                Sample(
                    t=loop_start,
                    binance_mid=mid,
                    binance_bid=bid,
                    binance_ask=ask,
                    poly_up_5m=results["poly5"],
                    poly_up_15m=results["poly15"],
                )
            )
            if len(samples) % 10 == 0:
                p5 = f"{results['poly5']:.3f}" if results["poly5"] else "—"
                p15 = f"{results['poly15']:.3f}" if results["poly15"] else "—"
                print(
                    f"  t+{loop_start - t0:6.1f}s  BTC={mid:>10,.2f}  poly_5m_up={p5}  poly_15m_up={p15}",
                    file=sys.stderr,
                )
        elapsed = time.time() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)

    print(f"\ncollected {len(samples)} samples", file=sys.stderr)
    out = {
        "started": datetime.fromtimestamp(t0, tz=UTC).isoformat(),
        "duration_s": args.duration,
        "hz": args.hz,
        "n_samples": len(samples),
        "markets": info,
        "samples": [s.__dict__ for s in samples],
    }
    if up_token_5m:
        out["lead_lag_5m"] = empirical_lag(samples, "poly_up_5m")
        out["big_moves_5m"] = big_moves(samples, "poly_up_5m")
    if up_token_15m:
        out["lead_lag_15m"] = empirical_lag(samples, "poly_up_15m")
        out["big_moves_15m"] = big_moves(samples, "poly_up_15m")

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {args.out}", file=sys.stderr)
    print("\n=== Lead-lag analysis ===", file=sys.stderr)
    if "lead_lag_5m" in out:
        ll = out["lead_lag_5m"]
        print(
            f"5m  : best_lag={ll.get('best_lag_samples')} samples ({ll.get('best_lag_samples', 0) / args.hz:.2f}s) "
            f"corr={ll.get('best_corr')} n={ll.get('n')}",
            file=sys.stderr,
        )
    if "lead_lag_15m" in out:
        ll = out["lead_lag_15m"]
        print(
            f"15m : best_lag={ll.get('best_lag_samples')} samples ({ll.get('best_lag_samples', 0) / args.hz:.2f}s) "
            f"corr={ll.get('best_corr')} n={ll.get('n')}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
