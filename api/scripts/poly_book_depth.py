"""Polymarket BTC up/down 5m & 15m orderbook depth and slippage profiler.

The Polymarket "BTC Up or Down" 5m markets advertise ~$15k liquidity in the
gamma `liquidityNum` field, but that's quoted depth (sum of all resting
orders on both sides). For a latency-arb strategy what actually matters is:

  - How wide is the bid-ask spread?
  - How much $-volume can fill within 1 cent / 3 cents of mid?
  - How does effective fill price degrade for $50, $200, $1000 orders?

This script:
  1. Discovers the currently active 5m and 15m BTC up/down markets.
  2. Fetches the full CLOB orderbook every 2s for 60s.
  3. For each (market, side, snapshot) computes:
       - bid-ask spread (cents)
       - book imbalance (bid_size / (bid_size + ask_size))
       - depth at $1/$5/$10 displacement (notional $-volume to push price by N cents)
       - effective fill price + slippage for $50, $200, $1000 buy/sell walks
  4. Aggregates median/p25/p75 across the snapshot window.
  5. Writes /tmp/poly_book_depth.json.

Usage:
    python poly_book_depth.py [--duration 60] [--interval 2.0]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from btc_updown_monitor import discover_active_market

CLOB = "https://clob.polymarket.com"


# ---------- CLOB fetches ----------


def fetch_book(token_id: str, client: httpx.Client) -> dict | None:
    """Fetch full CLOB book. Returns {'bids': [...], 'asks': [...]} or None.

    Polymarket book format: each level is {'price': '0.51', 'size': '120.0'}
    where size is in *shares* (each share resolves to $1 if YES wins).
    Notional $-cost of taking N shares at price p = N * p dollars.
    """
    try:
        r = client.get(f"{CLOB}/book", params={"token_id": token_id}, timeout=3.0)
        if r.status_code != 200:
            return None
        d = r.json()
        bids = [(float(lv["price"]), float(lv["size"])) for lv in d.get("bids", [])]
        asks = [(float(lv["price"]), float(lv["size"])) for lv in d.get("asks", [])]
        # CLOB returns bids ascending, asks ascending too. We want bids descending
        # (best = highest), asks ascending (best = lowest).
        bids.sort(key=lambda x: -x[0])
        asks.sort(key=lambda x: x[0])
        return {"bids": bids, "asks": asks}
    except Exception:
        return None


# ---------- Per-snapshot metrics ----------


def top_of_book(book: dict) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (best_bid, best_ask, bid_size_top, ask_size_top)."""
    bb = book["bids"][0] if book["bids"] else (None, None)
    ba = book["asks"][0] if book["asks"] else (None, None)
    return bb[0], ba[0], bb[1], ba[1]


def book_imbalance(book: dict, depth_levels: int = 5) -> float | None:
    """sum(bid_size top N) / (sum(bid) + sum(ask))."""
    bid_sum = sum(s for _, s in book["bids"][:depth_levels])
    ask_sum = sum(s for _, s in book["asks"][:depth_levels])
    if bid_sum + ask_sum == 0:
        return None
    return bid_sum / (bid_sum + ask_sum)


def depth_within_displacement(
    book: dict, mid: float, displacement_cents: float
) -> dict[str, float]:
    """How much $-notional can fill on each side within `displacement_cents` of mid.

    Buy side: walks asks from best up to mid + d. Sum of size*price for levels
              with price <= mid + d.
    Sell side: walks bids from best down to mid - d. Sum of size*price for levels
               with price >= mid - d.
    """
    d = displacement_cents / 100.0  # cents -> dollars (probability units)
    buy_notional = 0.0
    buy_shares = 0.0
    for price, size in book["asks"]:
        if price > mid + d:
            break
        buy_notional += size * price
        buy_shares += size
    sell_notional = 0.0
    sell_shares = 0.0
    for price, size in book["bids"]:
        if price < mid - d:
            break
        sell_notional += size * price
        sell_shares += size
    return {
        "buy_notional_usd": buy_notional,
        "buy_shares": buy_shares,
        "sell_notional_usd": sell_notional,
        "sell_shares": sell_shares,
    }


def walk_book(book: dict, mid: float, side: str, target_usd: float) -> dict[str, float | None]:
    """Walk the book to fill `target_usd` of notional on `side` ('buy' or 'sell').

    Buy: walks asks (you pay ask). Sell: walks bids (you receive bid).
    Returns effective avg fill price, slippage-vs-mid in cents, fill_pct.
    """
    levels = book["asks"] if side == "buy" else book["bids"]
    remaining_usd = target_usd
    total_shares = 0.0
    total_cost = 0.0
    last_price = None
    for price, size in levels:
        level_notional = price * size
        if remaining_usd <= 0:
            break
        take_notional = min(level_notional, remaining_usd)
        take_shares = take_notional / price
        total_shares += take_shares
        total_cost += take_notional
        last_price = price
        remaining_usd -= take_notional
    filled_usd = target_usd - remaining_usd
    if total_shares == 0 or filled_usd == 0:
        return {
            "avg_price": None,
            "slippage_cents": None,
            "fill_pct": 0.0,
            "worst_price": None,
            "filled_usd": 0.0,
        }
    avg_price = total_cost / total_shares
    # Slippage: buy slips UP from mid (positive cents), sell slips DOWN from mid
    if side == "buy":
        slippage_cents = (avg_price - mid) * 100.0
    else:
        slippage_cents = (mid - avg_price) * 100.0
    return {
        "avg_price": avg_price,
        "slippage_cents": slippage_cents,
        "fill_pct": filled_usd / target_usd,
        "worst_price": last_price,
        "filled_usd": filled_usd,
    }


def snapshot_metrics(book: dict) -> dict:
    """Compute all per-snapshot metrics for one orderbook."""
    bb, ba, bb_sz, ba_sz = top_of_book(book)
    if bb is None or ba is None:
        return {"empty": True}
    mid = (bb + ba) / 2.0
    spread = (ba - bb) * 100.0  # cents

    out = {
        "mid": mid,
        "best_bid": bb,
        "best_ask": ba,
        "best_bid_size": bb_sz,
        "best_ask_size": ba_sz,
        "top_bid_notional_usd": bb_sz * bb,
        "top_ask_notional_usd": ba_sz * ba,
        "spread_cents": spread,
        "imbalance_top5": book_imbalance(book, 5),
        "n_bid_levels": len(book["bids"]),
        "n_ask_levels": len(book["asks"]),
    }
    for d_cents in (1, 3, 5, 10):
        out[f"depth_{d_cents}c"] = depth_within_displacement(book, mid, d_cents)
    for size_usd in (50, 200, 1000):
        out[f"buy_{size_usd}"] = walk_book(book, mid, "buy", size_usd)
        out[f"sell_{size_usd}"] = walk_book(book, mid, "sell", size_usd)
    return out


# ---------- Aggregation ----------


def _percentile(xs: list[float], q: float) -> float | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def aggregate_snapshots(snaps: list[dict]) -> dict:
    """Compute median/p25/p75 across snapshots for each scalar metric."""
    snaps = [s for s in snaps if not s.get("empty")]
    if not snaps:
        return {"n": 0}

    def stats_of(extract) -> dict:
        vals = [extract(s) for s in snaps]
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "median": statistics.median(vals),
            "mean": statistics.mean(vals),
            "p25": _percentile(vals, 0.25),
            "p75": _percentile(vals, 0.75),
            "min": min(vals),
            "max": max(vals),
        }

    agg: dict = {
        "n_snapshots": len(snaps),
        "spread_cents": stats_of(lambda s: s["spread_cents"]),
        "imbalance_top5": stats_of(lambda s: s["imbalance_top5"]),
        "top_bid_notional_usd": stats_of(lambda s: s["top_bid_notional_usd"]),
        "top_ask_notional_usd": stats_of(lambda s: s["top_ask_notional_usd"]),
        "n_bid_levels": stats_of(lambda s: s["n_bid_levels"]),
        "n_ask_levels": stats_of(lambda s: s["n_ask_levels"]),
    }
    for d_cents in (1, 3, 5, 10):
        for side in ("buy", "sell"):
            key = f"depth_{d_cents}c_{side}_notional"
            agg[key] = stats_of(
                lambda s, dc=d_cents, sd=side: s[f"depth_{dc}c"][f"{sd}_notional_usd"]
            )
    for size_usd in (50, 200, 1000):
        for side in ("buy", "sell"):
            agg[f"slippage_{side}_{size_usd}_cents"] = stats_of(
                lambda s, sz=size_usd, sd=side: s[f"{sd}_{sz}"]["slippage_cents"]
            )
            agg[f"fill_pct_{side}_{size_usd}"] = stats_of(
                lambda s, sz=size_usd, sd=side: s[f"{sd}_{sz}"]["fill_pct"]
            )
    return agg


# ---------- Orchestration ----------


def profile_token(
    label: str,
    token_id: str,
    side_name: str,
    client: httpx.Client,
    duration: float,
    interval: float,
) -> dict:
    """Poll one token's book for `duration` seconds at `interval` cadence."""
    snaps: list[dict] = []
    raw_books: list[dict] = []
    deadline = time.time() + duration
    print(f"  [{label}/{side_name}] polling {token_id[:12]}…", file=sys.stderr)
    while time.time() < deadline:
        loop_start = time.time()
        book = fetch_book(token_id, client)
        if book is None:
            snaps.append({"empty": True, "t": loop_start})
        else:
            m = snapshot_metrics(book)
            m["t"] = loop_start
            snaps.append(m)
            raw_books.append(
                {
                    "t": loop_start,
                    "n_bids": len(book["bids"]),
                    "n_asks": len(book["asks"]),
                    "top10_bids": book["bids"][:10],
                    "top10_asks": book["asks"][:10],
                }
            )
        elapsed = time.time() - loop_start
        if elapsed < interval:
            time.sleep(interval - elapsed)
    print(f"  [{label}/{side_name}] {len(snaps)} snapshots", file=sys.stderr)
    return {
        "token_id": token_id,
        "snapshots": snaps,
        "sample_books": raw_books[:3] + raw_books[-3:],  # first/last 3 for inspection
        "aggregate": aggregate_snapshots(snaps),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--out", default="/tmp/poly_book_depth.json")
    args = p.parse_args()

    client = httpx.Client(timeout=5.0)
    print("Discovering active markets...", file=sys.stderr)
    m5 = discover_active_market("5m", client)
    m15 = discover_active_market("15m", client)
    if not m5 and not m15:
        print("ERROR: no active btc-updown markets found", file=sys.stderr)
        return 1

    markets: dict[str, dict] = {}
    for label, mkt in (("5m", m5), ("15m", m15)):
        if not mkt:
            continue
        tokens = json.loads(mkt["clobTokenIds"])
        # Polymarket convention: tokens[0] = UP/YES, tokens[1] = DOWN/NO
        markets[label] = {
            "slug": mkt["slug"],
            "endDate": mkt.get("endDate"),
            "liquidityNum": mkt.get("liquidityNum"),
            "volumeNum": mkt.get("volumeNum"),
            "up_token": tokens[0],
            "down_token": tokens[1] if len(tokens) > 1 else None,
        }
        print(
            f"  {label}: {mkt['slug']}  ends {mkt.get('endDate')}  "
            f"liquidity=${mkt.get('liquidityNum')}",
            file=sys.stderr,
        )

    print(f"\nProfiling books for {args.duration}s @ {args.interval}s interval...", file=sys.stderr)
    results: dict = {"markets": markets, "params": vars(args)}
    for label, info in markets.items():
        results[label] = {}
        for side_name, tok in (("UP", info["up_token"]), ("DOWN", info["down_token"])):
            if not tok:
                continue
            results[label][side_name] = profile_token(
                label, tok, side_name, client, args.duration, args.interval
            )

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.out}", file=sys.stderr)

    # Print summary
    print("\n=== Summary (median across snapshots) ===", file=sys.stderr)
    for label in markets:
        for side in ("UP", "DOWN"):
            if side not in results.get(label, {}):
                continue
            agg = results[label][side]["aggregate"]
            if not agg or agg.get("n_snapshots", 0) == 0:
                print(f"  {label} {side}: no data", file=sys.stderr)
                continue
            sp = agg["spread_cents"]["median"] if agg.get("spread_cents") else None
            d1b = (
                agg["depth_1c_buy_notional"]["median"] if agg.get("depth_1c_buy_notional") else None
            )
            d3b = (
                agg["depth_3c_buy_notional"]["median"] if agg.get("depth_3c_buy_notional") else None
            )
            slip200 = (
                agg["slippage_buy_200_cents"]["median"]
                if agg.get("slippage_buy_200_cents")
                else None
            )
            slip1000 = (
                agg["slippage_buy_1000_cents"]["median"]
                if agg.get("slippage_buy_1000_cents")
                else None
            )
            print(
                f"  {label} {side}: spread={sp}c  "
                f"depth_1c_buy=${d1b}  depth_3c_buy=${d3b}  "
                f"slip_$200={slip200}c  slip_$1000={slip1000}c",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
