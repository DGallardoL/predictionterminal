"""Historical backtest of BTC up/down 5m latency arb.

Discovers the last N resolved 5m markets, fetches Polymarket minute-level
price history (CLOB ``fidelity=1``) and Binance 1-min klines, then per
minute computes a "fair" UP probability from BTC drift over the window
and the realized edge vs the Polymarket UP price.

Strategy (synthetic):
    For each minute t in the 5-min window, if |poly_up - fair_up| > 5pp,
    take the corrective side (long UP if poly underpriced, short UP if
    overpriced) and hold to resolution.

PnL accounting:
    - Resolution price is 1.0 (UP wins) or 0.0 (DOWN wins).
    - Long UP at price p: pnl = resolved_up - p
    - Short UP at price p: pnl = p - resolved_up
    - All in dollars per $1 notional. Reported in basis points.

Caveats:
    - Polymarket charges a ~2% taker fee — we report gross PnL but also
      a "net of fees" view subtracting 2% per side.
    - Min trade size is $5; capital efficiency at $5 is brutal.
    - fair_up assumes 60% annualized vol on log-returns. Real BTC 5m
      realized vol is closer to 40-80% on most days.

Output: /tmp/btc_arb_backtest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BINANCE = "https://api.binance.com"

WINDOW_S = 300  # 5-min market
ANNUALIZED_VOL = 0.60
SECONDS_PER_YEAR = 365.25 * 24 * 3600
EDGE_THRESHOLD = 0.05  # 5 percentage points
TAKER_FEE = 0.02  # 2% per fill, two fills per round trip


@dataclass
class MarketWindow:
    slug: str
    end_unix: int
    up_token: str
    resolved_up: float  # 1.0 if UP won, 0.0 if DOWN won, 0.5 push
    poly_history: list[dict]  # list of {t, p} from CLOB
    binance_klines: list[list]  # raw klines from Binance


def discover_resolved_markets(client: httpx.Client, n: int, max_lookback: int = 600) -> list[dict]:
    """Walk back from the last full 5-min boundary, fetching closed markets.

    Tries up to ``max_lookback`` past boundaries; returns first ``n`` that
    are closed and have a usable outcomePrices (i.e. resolved cleanly).
    """
    now_unix = int(time.time())
    last_boundary = (now_unix // WINDOW_S) * WINDOW_S
    found: list[dict] = []
    # Skip the most-recent boundary (likely still resolving).
    for i in range(2, max_lookback + 2):
        end_unix = last_boundary - i * WINDOW_S
        slug = f"btc-updown-5m-{end_unix}"
        try:
            r = client.get(f"{GAMMA}/markets", params={"slug": slug, "closed": "true"}, timeout=5.0)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        arr = r.json()
        if not arr:
            continue
        m = arr[0]
        if not m.get("closed"):
            continue
        outcomes = m.get("outcomePrices")
        if not outcomes:
            continue
        try:
            op = json.loads(outcomes)
            up_resolved = float(op[0])
        except Exception:
            continue
        # Only take cleanly resolved (0 or 1)
        if up_resolved not in (0.0, 1.0):
            continue
        found.append(
            {
                "slug": slug,
                "end_unix": end_unix,
                "endDate": m.get("endDate"),
                "outcomePrices": [float(x) for x in op],
                "clobTokenIds": json.loads(m.get("clobTokenIds") or "[]"),
                "volume": float(m.get("volume") or 0),
            }
        )
        print(
            f"  found {slug} resolved_up={up_resolved} vol=${found[-1]['volume']:.0f}",
            file=sys.stderr,
        )
        if len(found) >= n:
            break
    return found


def fetch_poly_history(client: httpx.Client, up_token: str, end_unix: int) -> list[dict]:
    """Fetch minute-level (fidelity=1) UP price history for the 5-min window."""
    start_unix = end_unix - WINDOW_S
    # Ask for a bit before/after to anchor pre-window price
    r = client.get(
        f"{CLOB}/prices-history",
        params={
            "market": up_token,
            "fidelity": 1,
            "startTs": start_unix - 60,
            "endTs": end_unix + 60,
        },
        timeout=10.0,
    )
    if r.status_code != 200:
        return []
    return r.json().get("history", [])


def fetch_binance_klines(client: httpx.Client, end_unix: int) -> list[list]:
    """Fetch 1-min Binance klines covering the window plus a 1-min anchor."""
    start_ms = (end_unix - WINDOW_S - 60) * 1000
    end_ms = (end_unix + 60) * 1000
    r = client.get(
        f"{BINANCE}/api/v3/klines",
        params={
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 20,
        },
        timeout=10.0,
    )
    if r.status_code != 200:
        return []
    return r.json()


def fair_up_probability(p_now: float, p_start: float, seconds_remaining: int) -> float:
    """Black-Scholes-flavoured "probability the path ends >= start" given drift.

    Treat log(P) as Brownian with sigma per second from annualized vol; assume
    zero drift. Probability of ending >= P_start, conditional on P_now and
    ``seconds_remaining`` until resolution, is N(d) where:
        d = log(P_now / P_start) / (sigma * sqrt(T))
    """
    if seconds_remaining <= 0:
        return 1.0 if p_now >= p_start else 0.0
    sigma_s = ANNUALIZED_VOL / math.sqrt(SECONDS_PER_YEAR)
    sd = sigma_s * math.sqrt(seconds_remaining)
    if sd <= 0:
        return 1.0 if p_now >= p_start else 0.0
    d = math.log(p_now / p_start) / sd
    # N(d) via erf
    return 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))


def build_minute_grid(window: MarketWindow) -> list[dict]:
    """For each minute boundary inside the window, line up Binance close, poly UP."""
    end_unix = window.end_unix
    start_unix = end_unix - WINDOW_S

    # Binance: map open_time_s -> close
    bn_by_min: dict[int, float] = {}
    for k in window.binance_klines:
        open_ms = int(k[0])
        close_px = float(k[4])
        bn_by_min[open_ms // 1000] = close_px

    # The "window-start price" is the CLOSE of the minute that ends at start_unix
    # (i.e. open_time = start_unix - 60). Polymarket's resolution uses the
    # Chainlink stream price AT start_unix vs end_unix. Approximation: use the
    # binance close at start_unix-60 (close = price at start_unix).
    p_start = bn_by_min.get(start_unix - 60)
    if p_start is None:
        # Fallback: use the open of the first in-window minute.
        first = window.binance_klines[0] if window.binance_klines else None
        p_start = float(first[1]) if first else None
    if p_start is None:
        return []

    # Poly: for each minute boundary t in [start, end-60], take the LATEST
    # poly print at or before t. The history can have multiple prints per
    # minute or none — we forward-fill.
    poly_sorted = sorted(window.poly_history, key=lambda d: d["t"])

    def poly_at(t_unix: int) -> float | None:
        last = None
        for h in poly_sorted:
            if h["t"] <= t_unix:
                last = h["p"]
            else:
                break
        return float(last) if last is not None else None

    rows = []
    # Decision minutes: t = start, start+60, ..., end-60 (5 decisions per window).
    for offset in range(0, WINDOW_S, 60):
        t = start_unix + offset
        # Use the close of the minute ENDING at t (open=t-60). That's the price AT t.
        bn_at_t = bn_by_min.get(t - 60)
        if bn_at_t is None:
            continue
        seconds_remaining = end_unix - t
        fair = fair_up_probability(bn_at_t, p_start, seconds_remaining)
        poly_up = poly_at(t)
        if poly_up is None:
            continue
        edge = poly_up - fair  # poly_up - fair_up; > 0 means poly OVERPRICED
        rows.append(
            {
                "t": t,
                "minute_offset": offset,
                "seconds_remaining": seconds_remaining,
                "binance_at_t": bn_at_t,
                "binance_at_start": p_start,
                "log_return": math.log(bn_at_t / p_start),
                "fair_up": fair,
                "poly_up": poly_up,
                "edge": edge,
            }
        )
    return rows


def simulate_trades(window: MarketWindow, grid: list[dict]) -> list[dict]:
    """Take corrective trade when |edge| > 5pp; one trade per minute decision.

    Direction:
        edge > 0  (poly OVERPRICED UP)  → SHORT UP, pnl = poly_up - resolved
        edge < 0  (poly UNDERPRICED UP) →  LONG UP, pnl = resolved - poly_up
    """
    trades = []
    for row in grid:
        if abs(row["edge"]) <= EDGE_THRESHOLD:
            continue
        side = "SHORT_UP" if row["edge"] > 0 else "LONG_UP"
        entry = row["poly_up"]
        if side == "LONG_UP":
            pnl_gross = window.resolved_up - entry
        else:
            pnl_gross = entry - window.resolved_up
        # Fees: 2% on entry (buy/sell at $1 notional) + 2% on resolution payout.
        # Simplification: subtract 2*TAKER_FEE * notional ($1) ⇒ flat 4¢ per round trip.
        pnl_net = pnl_gross - 2 * TAKER_FEE
        trades.append(
            {
                "slug": window.slug,
                "t": row["t"],
                "minute_offset": row["minute_offset"],
                "side": side,
                "edge": row["edge"],
                "fair_up": row["fair_up"],
                "poly_up": row["poly_up"],
                "binance_at_t": row["binance_at_t"],
                "binance_at_start": row["binance_at_start"],
                "log_return": row["log_return"],
                "resolved_up": window.resolved_up,
                "pnl_gross": pnl_gross,
                "pnl_net": pnl_net,
            }
        )
    return trades


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0}
    n = len(trades)
    gross = [t["pnl_gross"] for t in trades]
    net = [t["pnl_net"] for t in trades]
    wins = sum(1 for g in gross if g > 0)
    avg_gross = sum(gross) / n
    avg_net = sum(net) / n
    var_gross = sum((g - avg_gross) ** 2 for g in gross) / max(n - 1, 1)
    sd_gross = math.sqrt(var_gross)
    # 1 trade per 5-min window ⇒ ~105,120 windows/year. Sharpe is per-trade
    # mean / sd, then * sqrt(N_per_year).
    trades_per_year = SECONDS_PER_YEAR / WINDOW_S
    sharpe = (avg_gross / sd_gross) * math.sqrt(trades_per_year) if sd_gross > 0 else None
    sharpe_net = None
    sd_net = math.sqrt(sum((g - avg_net) ** 2 for g in net) / max(n - 1, 1))
    if sd_net > 0:
        sharpe_net = (avg_net / sd_net) * math.sqrt(trades_per_year)

    sorted_by_pnl = sorted(trades, key=lambda d: d["pnl_gross"])
    worst = sorted_by_pnl[:5]
    best = sorted_by_pnl[-5:][::-1]

    return {
        "n_trades": n,
        "hit_rate": wins / n,
        "avg_pnl_gross_bp": avg_gross * 1e4,
        "avg_pnl_net_bp": avg_net * 1e4,
        "sd_pnl_bp": sd_gross * 1e4,
        "median_pnl_gross_bp": sorted(gross)[n // 2] * 1e4,
        "annualized_sharpe_gross": sharpe,
        "annualized_sharpe_net": sharpe_net,
        "worst_5_trades": worst,
        "best_5_trades": best,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--n-markets", type=int, default=20)
    p.add_argument("--max-lookback", type=int, default=800)
    p.add_argument("--out", default="/tmp/btc_arb_backtest.json")
    args = p.parse_args()

    client = httpx.Client(timeout=15.0)
    print(f"discovering up to {args.n_markets} resolved 5m markets …", file=sys.stderr)
    markets = discover_resolved_markets(client, n=args.n_markets, max_lookback=args.max_lookback)
    print(f"found {len(markets)} resolved markets", file=sys.stderr)
    if not markets:
        print("ERROR: no resolved markets discovered", file=sys.stderr)
        return 1

    windows: list[MarketWindow] = []
    skipped = []
    for m in markets:
        slug = m["slug"]
        tokens = m["clobTokenIds"]
        if len(tokens) < 2:
            skipped.append({"slug": slug, "reason": "no clobTokenIds"})
            continue
        up_token = tokens[0]
        end_unix = m["end_unix"]

        poly_hist = fetch_poly_history(client, up_token, end_unix)
        if not poly_hist:
            skipped.append({"slug": slug, "reason": "no poly history"})
            continue
        if len(poly_hist) < 2:
            skipped.append({"slug": slug, "reason": f"only {len(poly_hist)} poly prints"})
            continue

        bn_klines = fetch_binance_klines(client, end_unix)
        if len(bn_klines) < 5:
            skipped.append({"slug": slug, "reason": f"only {len(bn_klines)} klines"})
            continue

        windows.append(
            MarketWindow(
                slug=slug,
                end_unix=end_unix,
                up_token=up_token,
                resolved_up=m["outcomePrices"][0],
                poly_history=poly_hist,
                binance_klines=bn_klines,
            )
        )
        print(
            f"  loaded {slug}: {len(poly_hist)} poly prints, {len(bn_klines)} klines",
            file=sys.stderr,
        )

    print(f"\nbacktesting {len(windows)} windows …", file=sys.stderr)
    all_trades: list[dict] = []
    grid_summary = []
    for w in windows:
        grid = build_minute_grid(w)
        if not grid:
            skipped.append({"slug": w.slug, "reason": "empty grid (no anchor px)"})
            continue
        trades = simulate_trades(w, grid)
        grid_summary.append(
            {
                "slug": w.slug,
                "n_decisions": len(grid),
                "n_trades": len(trades),
                "resolved_up": w.resolved_up,
            }
        )
        all_trades.extend(trades)

    summary = summarize(all_trades)
    out = {
        "params": {
            "n_markets_requested": args.n_markets,
            "edge_threshold_pp": EDGE_THRESHOLD * 100,
            "annualized_vol": ANNUALIZED_VOL,
            "taker_fee_per_side": TAKER_FEE,
        },
        "n_markets_discovered": len(markets),
        "n_markets_backtested": len(windows),
        "n_decisions_total": sum(g["n_decisions"] for g in grid_summary),
        "skipped": skipped,
        "per_window": grid_summary,
        "summary": summary,
        "all_trades": all_trades,
    }

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {args.out}", file=sys.stderr)

    print("\n=== SUMMARY ===", file=sys.stderr)
    print(f"markets backtested      : {len(windows)}", file=sys.stderr)
    print(f"total decisions (minutes): {out['n_decisions_total']}", file=sys.stderr)
    print(f"trades (|edge|>5pp)     : {summary.get('n_trades', 0)}", file=sys.stderr)
    if summary.get("n_trades", 0) > 0:
        print(f"hit rate                : {summary['hit_rate']:.2%}", file=sys.stderr)
        print(f"avg pnl gross           : {summary['avg_pnl_gross_bp']:.1f} bp", file=sys.stderr)
        print(f"avg pnl net of 4% RT    : {summary['avg_pnl_net_bp']:.1f} bp", file=sys.stderr)
        print(f"sd pnl                  : {summary['sd_pnl_bp']:.1f} bp", file=sys.stderr)
        print(f"sharpe gross (annual.)  : {summary['annualized_sharpe_gross']}", file=sys.stderr)
        print(f"sharpe net (annual.)    : {summary['annualized_sharpe_net']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
