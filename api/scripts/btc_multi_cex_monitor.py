"""Multi-CEX BTC spot monitor for Polymarket-Chainlink latency arbitrage analysis.

Polymarket "BTC up/down" markets resolve via Chainlink BTC/USD, which
aggregates from multiple CEXes (likely Binance, Coinbase, Bitstamp, Kraken,
etc.). If a single CEX leads Chainlink, that CEX is our cleanest leading
indicator for Polymarket. If they all coincide, an average might be best.

This script polls (at ~2 Hz, concurrently):
  - Binance     : /api/v3/ticker/bookTicker?symbol=BTCUSDT
  - Coinbase    : /products/BTC-USD/ticker
  - Kraken      : /0/public/Ticker?pair=XBTUSDT
  - Bitstamp    : /api/v2/ticker/btcusd/  (optional)
  - Polymarket  : /midpoint for the active 5-minute UP token

After --duration seconds it computes:
  - Per-CEX update rate (mean updates/sec — distinct mids)
  - Per-CEX lead-lag profile vs Polymarket (best lag, best corr)
  - Pairwise lead-lag among the CEXes (which leads which)
  - "Combined" feed: equal-weighted mean of available CEX mids per timestamp,
    and its lead-lag vs Polymarket.

Usage:
    python btc_multi_cex_monitor.py --duration 240 --hz 2 --out /tmp/btc_multi_cex.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Reuse Polymarket discovery from the existing monitor
from btc_updown_monitor import discover_active_market, get_clob_midpoint

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

CEX_NAMES = ["binance", "coinbase", "kraken", "bitstamp"]


@dataclass
class Sample:
    t: float
    binance: float | None
    coinbase: float | None
    kraken: float | None
    bitstamp: float | None
    poly_up_5m: float | None


# ---------- CEX fetchers (return (mid, bid, ask) or None) ----------


class Backoff:
    """Per-CEX exponential backoff state (used after 429s / errors)."""

    def __init__(self) -> None:
        self.skip_until: float = 0.0
        self.delay: float = 0.0
        self.consecutive_429: int = 0

    def trip(self, base: float = 1.0, cap: float = 30.0) -> None:
        self.consecutive_429 += 1
        # Exponential backoff with jitter: 1, 2, 4, 8, 16, 30s cap
        self.delay = min(cap, base * (2 ** (self.consecutive_429 - 1)))
        self.delay *= 0.5 + random.random()
        self.skip_until = time.time() + self.delay

    def ok(self) -> None:
        self.consecutive_429 = 0
        self.delay = 0.0
        self.skip_until = 0.0

    def should_skip(self) -> bool:
        return time.time() < self.skip_until


def fetch_binance(client: httpx.Client, bo: Backoff) -> tuple[float, float, float] | None:
    if bo.should_skip():
        return None
    try:
        r = client.get(
            "https://api.binance.com/api/v3/ticker/bookTicker",
            params={"symbol": "BTCUSDT"},
            timeout=2.0,
        )
        if r.status_code == 429 or r.status_code == 418:
            bo.trip()
            return None
        if r.status_code != 200:
            return None
        d = r.json()
        bid = float(d["bidPrice"])
        ask = float(d["askPrice"])
        bo.ok()
        return ((bid + ask) / 2.0, bid, ask)
    except Exception:
        return None


def fetch_coinbase(client: httpx.Client, bo: Backoff) -> tuple[float, float, float] | None:
    if bo.should_skip():
        return None
    try:
        r = client.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
            timeout=2.0,
            headers={"User-Agent": "btc-multi-cex/1.0"},
        )
        if r.status_code == 429:
            bo.trip()
            return None
        if r.status_code != 200:
            return None
        d = r.json()
        bid = float(d["bid"])
        ask = float(d["ask"])
        bo.ok()
        return ((bid + ask) / 2.0, bid, ask)
    except Exception:
        return None


def fetch_kraken(client: httpx.Client, bo: Backoff) -> tuple[float, float, float] | None:
    if bo.should_skip():
        return None
    try:
        r = client.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": "XBTUSDT"},
            timeout=2.0,
        )
        if r.status_code == 429:
            bo.trip()
            return None
        if r.status_code != 200:
            return None
        d = r.json()
        if d.get("error"):
            return None
        result = d.get("result") or {}
        # Kraken normalizes XBTUSDT to its own pair key (often "XBTUSDT")
        if not result:
            return None
        first_key = next(iter(result))
        info = result[first_key]
        # b = best bid [price, lot, vol], a = best ask
        bid = float(info["b"][0])
        ask = float(info["a"][0])
        bo.ok()
        return ((bid + ask) / 2.0, bid, ask)
    except Exception:
        return None


def fetch_bitstamp(client: httpx.Client, bo: Backoff) -> tuple[float, float, float] | None:
    if bo.should_skip():
        return None
    try:
        r = client.get(
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            timeout=2.0,
            headers={"User-Agent": "btc-multi-cex/1.0"},
        )
        if r.status_code == 429:
            bo.trip()
            return None
        if r.status_code != 200:
            return None
        d = r.json()
        bid = float(d["bid"])
        ask = float(d["ask"])
        bo.ok()
        return ((bid + ask) / 2.0, bid, ask)
    except Exception:
        return None


# ---------- Analytics ----------


def _corr(x: list[float], y: list[float]) -> float:
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


def _diffs(seq: list[float]) -> list[float]:
    return [seq[i] - seq[i - 1] for i in range(1, len(seq))]


def lead_lag(x_series: list[float | None], y_series: list[float | None], max_lag: int = 10) -> dict:
    """Return lead-lag dict: positive lag = y lags x (x leads).

    x and y aligned by index (sample). NaN/None entries are masked
    pairwise: only timestamps where both have a value contribute.
    """
    pairs = [
        (a, b) for a, b in zip(x_series, y_series, strict=False) if a is not None and b is not None
    ]
    if len(pairs) < 30:
        return {"error": "insufficient overlapping samples", "n": len(pairs)}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    dx = _diffs(xs)
    dy = _diffs(ys)
    by_lag: list[tuple[int, float]] = []
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            a, b = dx[: len(dx) - k], dy[k:]
        else:
            a, b = dx[-k:], dy[: len(dy) + k]
        if len(a) < 5:
            continue
        by_lag.append((k, _corr(a, b)))
    if not by_lag:
        return {"error": "no usable lags", "n": len(pairs)}
    best = max(by_lag, key=lambda kv: kv[1])
    return {
        "n": len(pairs),
        "best_lag_samples": best[0],
        "best_corr": round(best[1], 4),
        "by_lag": [(k, round(c, 4)) for k, c in sorted(by_lag)],
    }


def per_cex_update_rate(samples: list[Sample], cex: str) -> dict:
    """Compute fraction of samples with a value, and *distinct* update rate.

    update_rate = number of (mid != prev_mid) transitions / total elapsed seconds.
    """
    seq = [getattr(s, cex) for s in samples]
    n_total = len(seq)
    n_present = sum(1 for v in seq if v is not None)
    # Count distinct transitions across non-None samples.
    transitions = 0
    last = None
    for v in seq:
        if v is None:
            continue
        if last is None or v != last:
            transitions += 1
            last = v
    elapsed = (samples[-1].t - samples[0].t) if len(samples) >= 2 else 0.0
    return {
        "samples_with_value": n_present,
        "samples_total": n_total,
        "coverage": round(n_present / n_total, 3) if n_total else 0.0,
        "distinct_transitions": transitions,
        "updates_per_sec": round(transitions / elapsed, 3) if elapsed > 0 else None,
    }


def combined_mid(samples: list[Sample], cex_list: list[str]) -> list[float | None]:
    """Equal-weighted mean across the listed CEXes; None when fewer than 2 present."""
    out: list[float | None] = []
    for s in samples:
        vals = [getattr(s, c) for c in cex_list if getattr(s, c) is not None]
        if len(vals) >= 2:
            out.append(sum(vals) / len(vals))
        elif len(vals) == 1:
            out.append(vals[0])
        else:
            out.append(None)
    return out


# ---------- Main loop ----------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=240, help="seconds to monitor")
    p.add_argument("--hz", type=float, default=2.0, help="poll rate per second")
    p.add_argument("--out", default="/tmp/btc_multi_cex.json")
    p.add_argument("--max-lag", type=int, default=10, help="max lag (samples) for cross-corr")
    p.add_argument("--no-bitstamp", action="store_true", help="skip Bitstamp")
    args = p.parse_args()

    client = httpx.Client(timeout=3.0)
    print("discovering active 5m polymarket market...", file=sys.stderr)
    m5 = discover_active_market("5m", client)
    if not m5:
        print("ERROR: no active 5m btc-updown market", file=sys.stderr)
        return 1
    t5 = json.loads(m5["clobTokenIds"])
    up_token_5m = t5[0]
    print(f"  5m → {m5['slug']} ends {m5['endDate']}", file=sys.stderr)

    cex_names = list(CEX_NAMES)
    if args.no_bitstamp:
        cex_names.remove("bitstamp")

    backoffs: dict[str, Backoff] = {c: Backoff() for c in cex_names}
    fetchers = {
        "binance": fetch_binance,
        "coinbase": fetch_coinbase,
        "kraken": fetch_kraken,
        "bitstamp": fetch_bitstamp,
    }
    error_counts: dict[str, int] = dict.fromkeys(cex_names, 0)

    samples: list[Sample] = []
    period = 1.0 / args.hz
    t0 = time.time()
    deadline = t0 + args.duration
    print(
        f"polling {len(cex_names)} CEXes + Polymarket at {args.hz} Hz for {args.duration}s …",
        file=sys.stderr,
    )

    while time.time() < deadline:
        loop_start = time.time()
        results: dict[str, object] = dict.fromkeys(cex_names)
        results["poly5"] = None

        def _do_cex(name: str, results: dict[str, object] = results) -> None:
            r = fetchers[name](client, backoffs[name])
            if r is None:
                error_counts[name] += 1
            else:
                results[name] = r[0]  # mid

        def _do_poly(results: dict[str, object] = results) -> None:
            if up_token_5m:
                results["poly5"] = get_clob_midpoint(up_token_5m, client)

        ths: list[threading.Thread] = []
        for name in cex_names:
            ths.append(threading.Thread(target=_do_cex, args=(name,)))
        ths.append(threading.Thread(target=_do_poly))
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout=2.5)

        samples.append(
            Sample(
                t=loop_start,
                binance=results.get("binance"),
                coinbase=results.get("coinbase"),
                kraken=results.get("kraken"),
                bitstamp=results.get("bitstamp"),
                poly_up_5m=results.get("poly5"),
            )
        )

        if len(samples) % 20 == 0:
            mids = " ".join(
                f"{c[:3]}={getattr(samples[-1], c):.0f}"
                if getattr(samples[-1], c)
                else f"{c[:3]}=—"
                for c in cex_names
            )
            poly = f"{samples[-1].poly_up_5m:.3f}" if samples[-1].poly_up_5m is not None else "—"
            print(f"  t+{loop_start - t0:6.1f}s  {mids}  poly={poly}", file=sys.stderr)

        elapsed = time.time() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)

    print(f"\ncollected {len(samples)} samples", file=sys.stderr)

    # ---------- Analyses ----------
    update_rates = {c: per_cex_update_rate(samples, c) for c in cex_names}

    # Per-CEX vs Polymarket lead-lag (positive lag = polymarket lags CEX)
    poly_series = [s.poly_up_5m for s in samples]
    per_cex_vs_poly: dict[str, dict] = {}
    for c in cex_names:
        cex_series = [getattr(s, c) for s in samples]
        per_cex_vs_poly[c] = lead_lag(cex_series, poly_series, max_lag=args.max_lag)

    # Pairwise CEX matrix: [i][j] = lead_lag(i, j); positive = j lags i (i leads)
    pairwise: dict[str, dict[str, dict]] = {}
    for i in cex_names:
        pairwise[i] = {}
        si = [getattr(s, i) for s in samples]
        for j in cex_names:
            if i == j:
                continue
            sj = [getattr(s, j) for s in samples]
            pairwise[i][j] = lead_lag(si, sj, max_lag=args.max_lag)

    # Combined feed (equal-weighted mean of available CEXes per sample)
    combined_series = combined_mid(samples, cex_names)
    combined_vs_poly = lead_lag(combined_series, poly_series, max_lag=args.max_lag)

    # Best leading indicator: max best_corr at lag >= 0 (CEX leads polymarket)
    leaders = []
    for c, ll in per_cex_vs_poly.items():
        if "error" in ll:
            continue
        # Find best corr restricted to non-negative lag (CEX leads or coincides with poly)
        non_neg = [(k, v) for k, v in ll.get("by_lag", []) if k >= 0]
        if non_neg:
            k, v = max(non_neg, key=lambda kv: kv[1])
            leaders.append(
                {
                    "cex": c,
                    "best_lag_at_or_after_0": k,
                    "best_corr_at_or_after_0": v,
                    "best_lag_overall": ll.get("best_lag_samples"),
                    "best_corr_overall": ll.get("best_corr"),
                }
            )
    leaders.sort(key=lambda d: -d["best_corr_at_or_after_0"])

    # Conclusion text
    conclusion: list[str] = []
    if leaders:
        top = leaders[0]
        conclusion.append(
            f"Best single leading CEX vs Polymarket (lag >= 0): "
            f"{top['cex']} (corr={top['best_corr_at_or_after_0']} "
            f"at lag={top['best_lag_at_or_after_0']} samples = "
            f"{top['best_lag_at_or_after_0'] / args.hz:.2f}s)"
        )
    if "error" not in combined_vs_poly:
        conclusion.append(
            f"Combined (equal-weighted mean) vs Polymarket: "
            f"corr={combined_vs_poly['best_corr']} at lag="
            f"{combined_vs_poly['best_lag_samples']} samples "
            f"({combined_vs_poly['best_lag_samples'] / args.hz:.2f}s)"
        )
    # Pairwise summary: who leads whom on average
    pair_summary: list[dict] = []
    for i in cex_names:
        for j in cex_names:
            if i == j:
                continue
            ll = pairwise[i][j]
            if "error" in ll:
                continue
            pair_summary.append(
                {
                    "leader": i,
                    "follower": j,
                    "best_lag_samples": ll.get("best_lag_samples"),
                    "best_lag_seconds": (ll.get("best_lag_samples") or 0) / args.hz,
                    "best_corr": ll.get("best_corr"),
                }
            )

    out = {
        "started": datetime.fromtimestamp(t0, tz=UTC).isoformat(),
        "duration_s": args.duration,
        "hz": args.hz,
        "max_lag_samples": args.max_lag,
        "n_samples": len(samples),
        "polymarket": {"slug": m5["slug"], "up_token": up_token_5m, "endDate": m5["endDate"]},
        "cex_list": cex_names,
        "error_counts": error_counts,
        "update_rates": update_rates,
        "per_cex_vs_polymarket": per_cex_vs_poly,
        "pairwise_lead_lag": pairwise,
        "pairwise_summary": pair_summary,
        "combined_vs_polymarket": combined_vs_poly,
        "ranked_leaders": leaders,
        "conclusion": conclusion,
        # Compact samples (mids only) for downstream re-analysis.
        "samples": [s.__dict__ for s in samples],
    }

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {args.out}", file=sys.stderr)

    print("\n=== Update rates (distinct mid changes / sec) ===", file=sys.stderr)
    for c in cex_names:
        u = update_rates[c]
        print(
            f"  {c:9s}  ups/s={u['updates_per_sec']}  "
            f"coverage={u['coverage']}  errors={error_counts[c]}",
            file=sys.stderr,
        )
    print("\n=== Per-CEX vs Polymarket (positive lag = polymarket lags CEX) ===", file=sys.stderr)
    for c, ll in per_cex_vs_poly.items():
        if "error" in ll:
            print(f"  {c:9s}  {ll}", file=sys.stderr)
        else:
            print(
                f"  {c:9s}  best_lag={ll['best_lag_samples']} samples "
                f"({ll['best_lag_samples'] / args.hz:+.2f}s)  "
                f"corr={ll['best_corr']}  n={ll['n']}",
                file=sys.stderr,
            )
    print("\n=== Combined (mean) vs Polymarket ===", file=sys.stderr)
    print(f"  {combined_vs_poly}", file=sys.stderr)
    print("\n=== Pairwise CEX lead-lag (positive lag = follower lags leader) ===", file=sys.stderr)
    for row in pair_summary:
        print(
            f"  {row['leader']:9s} → {row['follower']:9s}  "
            f"lag={row['best_lag_samples']} samples "
            f"({row['best_lag_seconds']:+.2f}s)  corr={row['best_corr']}",
            file=sys.stderr,
        )
    print("\n=== Conclusion ===", file=sys.stderr)
    for line in conclusion:
        print(f"  {line}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
