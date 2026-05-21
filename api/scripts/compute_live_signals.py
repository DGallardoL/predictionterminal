"""Compute live spread z-scores for curated alpha strategies.

Reads /web/data/alpha_strategies.json, pulls fresh prices from Polymarket
for each pair's two legs, fits the cointegration spread on the most recent
60 days, computes the latest z-score, and emits a BUY_A / SELL_A / FLAT
signal per pair.

Output: /web/data/live_signals.json. The frontend loads this alongside
alpha_strategies.json so cards show live actionable signals.

Run this hourly (or less frequently) — Polymarket markets are mostly
slow-moving probability series.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pfm.cointegration import engle_granger
from pfm.factors import load_factors
from pfm.sources.polymarket import PolymarketClient, fetch_factor_history

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")


def _signal_from_z(z: float, entry: float, exit: float, stop: float) -> tuple[str, str]:
    """Map current z-score to a recommended action."""
    if not np.isfinite(z):
        return ("FLAT", "no signal — z-score not finite")
    if abs(z) >= stop:
        return ("FLAT", f"|z|={abs(z):.2f} ≥ stop={stop} — risk-off")
    if z >= entry:
        return (
            "SHORT_SPREAD",
            f"z={z:+.2f} ≥ entry={entry} → short A, long β·B (spread expected to mean-revert down)",
        )
    if z <= -entry:
        return (
            "LONG_SPREAD",
            f"z={z:+.2f} ≤ −entry → long A, short β·B (spread expected to mean-revert up)",
        )
    if abs(z) <= exit:
        return ("FLAT_EXIT", f"|z|={abs(z):.2f} ≤ exit={exit} — flatten if open")
    return ("HOLD", f"z={z:+.2f} ∈ (exit, entry) — hold position")


def _fetch_with_retry(
    client: PolymarketClient,
    slug: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_retries: int = 4,
) -> pd.DataFrame:
    """Fetch factor history with exponential backoff on 429."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fetch_factor_history(client, slug, start, end)
        except Exception as e:
            msg = str(e)
            last_err = e
            if "429" in msg or "Too Many Requests" in msg:
                time.sleep(0.5 * (2**attempt))
                continue
            raise
    raise last_err if last_err else RuntimeError("unknown fetch failure")


def _process_pair(client: PolymarketClient, s: dict, factors: dict, today: pd.Timestamp) -> dict:
    """Fetch live prices for both legs, compute spread + z-score + signal."""
    a_id, b_id = s["a_id"], s["b_id"]
    out = {"pair_id": s["pair_id"], "a_id": a_id, "b_id": b_id, "as_of": today.isoformat()}
    try:
        a_fc, b_fc = factors.get(a_id), factors.get(b_id)
        if a_fc is None or b_fc is None:
            return {**out, "error": f"factor not found ({a_id} or {b_id})"}
        # 60-day lookback ending today
        start = today - pd.Timedelta(days=60)
        df_a = _fetch_with_retry(client, a_fc.slug, start, today)
        df_b = _fetch_with_retry(client, b_fc.slug, start, today)
        if df_a.empty or df_b.empty:
            return {**out, "error": "empty history"}
        a = df_a["price"].rename(a_id)
        b = df_b["price"].rename(b_id)
        cr = engle_granger(a, b)
        if cr.n_obs < 20:
            return {**out, "error": f"too few overlapping bars ({cr.n_obs})"}
        spread = cr.spread
        # Rolling z-score with same window the strategy uses
        win = int(s.get("rule_window", 20))
        mu = spread.rolling(win, min_periods=max(5, win // 2)).mean()
        sd = spread.rolling(win, min_periods=max(5, win // 2)).std(ddof=1)
        z = ((spread - mu) / sd).iloc[-1]
        cur_spread = spread.iloc[-1]
        action, reason = _signal_from_z(
            float(z),
            float(s.get("rule_entry_z", 2.0)),
            float(s.get("rule_exit_z", 0.5)),
            float(s.get("rule_stop_z", 4.0)),
        )
        return {
            **out,
            "n_obs": cr.n_obs,
            "beta_hedge": float(cr.beta_hedge),
            "current_spread": float(cur_spread),
            "current_z": float(z) if np.isfinite(z) else None,
            "current_a_price": float(a.dropna().iloc[-1]),
            "current_b_price": float(b.dropna().iloc[-1]),
            "action": action,
            "reason": reason,
            "mu_window": float(mu.iloc[-1]) if np.isfinite(mu.iloc[-1]) else None,
            "sigma_window": float(sd.iloc[-1]) if np.isfinite(sd.iloc[-1]) else None,
        }
    except Exception as e:
        return {**out, "error": str(e)[:200]}


def main() -> int:
    strategies_path = ROOT.parent / "web" / "data" / "alpha_strategies.json"
    out_path = ROOT.parent / "web" / "data" / "live_signals.json"
    strategies = json.loads(strategies_path.read_text())
    factors = load_factors(ROOT / "src" / "pfm" / "factors.yml")
    today = pd.Timestamp(datetime.now(tz=UTC).date(), tz="UTC")
    print(
        f"computing live signals for {len(strategies['strategies'])} strategies as of {today.date()}",
        file=sys.stderr,
    )
    client = PolymarketClient(
        gamma_url="https://gamma-api.polymarket.com",
        clob_url="https://clob.polymarket.com",
    )
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [
            ex.submit(_process_pair, client, s, factors, today) for s in strategies["strategies"]
        ]
        for fut in as_completed(futs):
            results.append(fut.result())
    print(f"done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    n_actionable = sum(1 for r in results if r.get("action") in ("LONG_SPREAD", "SHORT_SPREAD"))
    n_errors = sum(1 for r in results if "error" in r)
    print(
        f"actionable signals: {n_actionable} / {len(results)}; errors: {n_errors}", file=sys.stderr
    )
    output = {
        "as_of": today.isoformat(),
        "n_strategies": len(results),
        "n_actionable": n_actionable,
        "n_errors": n_errors,
        "signals": {r["pair_id"]: r for r in results if "pair_id" in r},
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)

    # Print top actionable
    actionable = [r for r in results if r.get("action") in ("LONG_SPREAD", "SHORT_SPREAD")]
    actionable.sort(key=lambda r: -abs(r.get("current_z") or 0))
    print("\nTop actionable signals (sorted by |z|):", file=sys.stderr)
    for r in actionable[:15]:
        print(
            f"  z={r['current_z']:+.2f}  {r['action']:13}  {r['a_id'][:30]} ↔ {r['b_id'][:30]}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
