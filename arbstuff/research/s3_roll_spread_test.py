"""Compute Roll's (1984) effective bid-ask spread for cross-venue arb pairs."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

STATE_PATH = "/Users/damiangallardoloya/Desktop/proyectofuentes/arbstuff/dashboard_state.json"
CLOB_URL = "https://clob.polymarket.com/prices-history"
USER_AGENT = "roll-spread-audit/0.1"


def fetch_history(token_id: str, interval: str = "max") -> list[float] | None:
    """Fetch daily price series for a Polymarket CLOB token. Returns list of floats or None."""
    url = f"{CLOB_URL}?market={token_id}&fidelity=1440&interval={interval}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError):
        return None
    except Exception:
        return None
    hist = data.get("history") or []
    prices = [float(pt["p"]) for pt in hist if "p" in pt]
    return prices


def covariance_lag1(dp: list[float]) -> float:
    """Sample covariance between dp[i] and dp[i+1]."""
    if len(dp) < 3:
        return float("nan")
    a = dp[:-1]
    b = dp[1:]
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    n = len(a)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)
    return cov


def roll_spread(prices: list[float]) -> tuple[float | None, float, int]:
    """Return (roll_spread or None if trending, cov1, n_diffs)."""
    if len(prices) < 31:
        return (None, float("nan"), len(prices) - 1)
    dp = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    cov1 = covariance_lag1(dp)
    if not math.isfinite(cov1):
        return (None, cov1, len(dp))
    if cov1 >= 0:
        return (None, cov1, len(dp))
    return (2.0 * math.sqrt(-cov1), cov1, len(dp))


def short_name(name: str, n: int = 60) -> str:
    if len(name) <= n:
        return name
    return name[: n - 1] + "…"


def main() -> int:
    with open(STATE_PATH) as f:
        state = json.load(f)
    opps = state["opportunities"]
    print(f"Loaded {len(opps)} opportunities from dashboard state", file=sys.stderr)

    # Deduplicate by token_id to avoid hammering CLOB for the same series
    unique_tokens: dict[str, dict] = {}
    for o in opps:
        tid = o.get("poly_token_id")
        if not tid:
            continue
        if tid not in unique_tokens:
            unique_tokens[tid] = o
    print(f"{len(unique_tokens)} unique poly token_ids", file=sys.stderr)

    results: list[dict] = []
    skipped_no_hist = 0
    skipped_short = 0

    def work(tid_op: tuple[str, dict]) -> dict | None:
        tid, op = tid_op
        prices = fetch_history(tid)
        if not prices:
            return {"_status": "no_hist", "op": op}
        if len(prices) < 31:
            return {"_status": "short", "op": op, "n": len(prices)}
        s_roll, cov1, n = roll_spread(prices)
        return {
            "_status": "ok",
            "op": op,
            "prices_n": len(prices),
            "cov1": cov1,
            "roll": s_roll,  # None if trending
            "trending": s_roll is None,
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, (tid, op)) for tid, op in unique_tokens.items()]
        for fut in as_completed(futs):
            r = fut.result()
            if r is None:
                continue
            if r["_status"] == "no_hist":
                skipped_no_hist += 1
                continue
            if r["_status"] == "short":
                skipped_short += 1
                continue
            results.append(r)
    print(f"fetched in {time.time() - t0:.1f}s", file=sys.stderr)
    print(
        f"rated={len(results)} skipped_no_hist={skipped_no_hist} skipped_short={skipped_short}",
        file=sys.stderr,
    )

    # Split trending vs rated
    rated = [r for r in results if not r["trending"]]
    trending = [r for r in results if r["trending"]]

    # Compute ratio Roll / displayed for non-trending with displayed > 0
    rows = []
    for r in rated:
        op = r["op"]
        displayed = float(op.get("spread") or 0.0)
        roll = r["roll"]
        if roll is None:
            continue
        ratio = (roll / displayed) if displayed > 1e-9 else float("inf")
        rows.append(
            {
                "name": op["name"],
                "kalshi": op.get("kalshi_ticker", ""),
                "slug": op.get("poly_slug", ""),
                "displayed": displayed,
                "roll": roll,
                "ratio": ratio,
                "cov1": r["cov1"],
                "n": r["prices_n"],
                "kalshi_price": op.get("kalshi_price"),
                "poly_price": op.get("poly_price"),
                "profit_pct": op.get("profit_pct"),
            }
        )

    # Filter out displayed=0 (ratio infinite — not informative for the multiplier)
    finite_rows = [r for r in rows if math.isfinite(r["ratio"])]
    zero_disp = [r for r in rows if not math.isfinite(r["ratio"])]

    finite_rows.sort(key=lambda r: r["ratio"], reverse=True)

    print()
    print("=" * 100)
    print(f"AGGREGATE — rated={len(rows)} (finite-ratio={len(finite_rows)}, "
          f"displayed=0={len(zero_disp)}) trending={len(trending)} "
          f"no_hist={skipped_no_hist} short={skipped_short}")
    if finite_rows:
        ratios = [r["ratio"] for r in finite_rows]
        rolls = [r["roll"] for r in finite_rows]
        disps = [r["displayed"] for r in finite_rows]
        print(f"Roll/displayed: median={statistics.median(ratios):.2f}x  "
              f"mean={statistics.mean(ratios):.2f}x  "
              f"p25={statistics.quantiles(ratios, n=4)[0]:.2f}x  "
              f"p75={statistics.quantiles(ratios, n=4)[2]:.2f}x")
        print(f"Roll spread (cents): median={100*statistics.median(rolls):.2f}  "
              f"mean={100*statistics.mean(rolls):.2f}")
        print(f"Displayed spread (cents): median={100*statistics.median(disps):.2f}  "
              f"mean={100*statistics.mean(disps):.2f}")
    print()

    print("=" * 100)
    print("TOP-10 WORST OFFENDERS (Roll >> displayed)")
    print("-" * 100)
    print(f"{'#':>2}  {'displayed':>9}  {'Roll':>7}  {'ratio':>7}  pair")
    for i, r in enumerate(finite_rows[:10], 1):
        print(f"{i:>2}  {r['displayed']:>9.4f}  {r['roll']:>7.4f}  "
              f"{r['ratio']:>6.2f}x  {short_name(r['name'])}")

    # Honest = ratio closest to 1.0 from either side, prefer 0.8-1.2 band
    honest_candidates = [r for r in finite_rows if 0.5 <= r["ratio"] <= 2.0]
    honest_candidates.sort(key=lambda r: abs(math.log(r["ratio"])))

    print()
    print("=" * 100)
    print(f"TOP-5 HONEST PAIRS (Roll ~ displayed; |log ratio| smallest, "
          f"candidates={len(honest_candidates)})")
    print("-" * 100)
    print(f"{'#':>2}  {'displayed':>9}  {'Roll':>7}  {'ratio':>7}  pair")
    for i, r in enumerate(honest_candidates[:5], 1):
        print(f"{i:>2}  {r['displayed']:>9.4f}  {r['roll']:>7.4f}  "
              f"{r['ratio']:>6.2f}x  {short_name(r['name'])}")

    # Action item
    print()
    print("=" * 100)
    print("ACTION ITEM")
    print("-" * 100)
    if finite_rows:
        med = statistics.median([r["ratio"] for r in finite_rows])
        # Current poly_fee values in the opps
        poly_fees = [float(o.get("poly_fee") or 0.0) for o in opps if o.get("poly_fee")]
        med_fee = statistics.median(poly_fees) if poly_fees else float("nan")
        print(f"Median Roll / displayed = {med:.2f}x")
        print(f"Median current poly_fee in opps = {med_fee:.4f}  ({100*med_fee:.2f}%)")
        print(f"Suggested Polymarket fee multiplier in arb_engine.py: x{med:.2f}")
        if med_fee == med_fee:  # not nan
            print(f"  -> e.g. bump poly_fee from {med_fee:.4f} to "
                  f"{med_fee*med:.4f} (median*current)")
        print()
        print("Interpretation: under bid-ask bounce (Roll), the realized round-trip cost "
              "is on the order of the Roll spread. If median Roll / displayed >> 1, the "
              "engine is treating the mid-price gap as the only cost; the real frictional "
              "cost is materially larger.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
