"""High-frequency WebSocket lead-lag monitor for Polymarket BTC up/down 5m markets.

The REST-based monitor at btc_updown_monitor.py polls at 2 Hz (500ms granularity),
which is too coarse to measure the true micro-lag between Binance spot and the
Polymarket CLOB midpoint (likely 100-300ms).

This monitor:
  1. Subscribes to Binance `btcusdt@bookTicker` over WebSocket — pushes every
     book-top update (sub-100ms typical). Each frame has b/a/T (event time ms).
  2. Polls the Polymarket CLOB /midpoint at the highest reasonable rate (5-10 Hz)
     in a worker thread.
  3. Records (event_time_ms, source, value) into one unified timeline.
  4. After --duration s, computes the cross-correlation function on a uniform
     50-ms grid over lags in [-2000, +2000] ms and reports the lag at peak corr.

Usage:
    PYTHONPATH=api/src api/.venv/bin/python \
        api/scripts/btc_updown_ws_monitor.py --duration 240 --out /tmp/btc_arb_ws.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from btc_updown_monitor import discover_active_market

CLOB = "https://clob.polymarket.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"


@dataclass
class Tick:
    t_ms: int  # event-time in ms (Binance T) or local recv-time for poly
    source: str  # "binance" | "polymarket"
    value: float  # binance midpoint OR polymarket UP-token midpoint


# ------------------------ Binance WS producer ----------------------------- #


async def binance_ws_loop(ticks: list[Tick], stop: asyncio.Event) -> None:
    """Subscribe to btcusdt@bookTicker; append (T, mid) to ticks until stop."""
    async with websockets.connect(BINANCE_WS, ping_interval=15, ping_timeout=10) as ws:
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except TimeoutError:
                continue
            try:
                msg = json.loads(raw)
                bid = float(msg["b"])
                ask = float(msg["a"])
                # Some payloads use "T" (transaction time); some use "E" (event).
                # bookTicker payloads include T (per Binance docs).
                t_ms = int(msg.get("T") or msg.get("E") or time.time() * 1000)
                ticks.append(Tick(t_ms=t_ms, source="binance", value=(bid + ask) / 2.0))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue


# ------------------------ Polymarket polling thread ----------------------- #


def poly_poll_loop(
    token_id: str,
    ticks: list[Tick],
    ticks_lock: threading.Lock,
    stop: threading.Event,
    target_hz: float,
) -> None:
    """Poll CLOB /midpoint at target_hz, append to ticks. Runs in its own thread."""
    period = 1.0 / target_hz
    last_value: float | None = None
    with httpx.Client(timeout=2.0, http2=False) as client:
        while not stop.is_set():
            t_start = time.time()
            try:
                r = client.get(f"{CLOB}/midpoint", params={"token_id": token_id})
                if r.status_code == 200:
                    mid = float(r.json().get("mid", "nan"))
                    if not math.isnan(mid):
                        # Record only on change OR with throttled cadence so we
                        # capture true update events rather than re-asserting
                        # the same value every poll. We keep both: record every
                        # poll so consumers can see polling cadence, but mark a
                        # change flag implicitly via value diffs downstream.
                        recv_ms = int(time.time() * 1000)
                        with ticks_lock:
                            ticks.append(Tick(t_ms=recv_ms, source="polymarket", value=mid))
                        last_value = mid
            except Exception:
                pass
            elapsed = time.time() - t_start
            if elapsed < period:
                stop.wait(period - elapsed)


# ------------------------ Cross-correlation ------------------------------- #


def resample_uniform(
    ticks: list[Tick], source: str, t0_ms: int, t1_ms: int, step_ms: int
) -> list[float | None]:
    """Forward-fill source values onto a uniform grid t0..t1 (step_ms)."""
    series = sorted([t for t in ticks if t.source == source], key=lambda x: x.t_ms)
    grid_n = (t1_ms - t0_ms) // step_ms + 1
    out: list[float | None] = [None] * grid_n
    j = 0
    last: float | None = None
    for i in range(grid_n):
        ts = t0_ms + i * step_ms
        while j < len(series) and series[j].t_ms <= ts:
            last = series[j].value
            j += 1
        out[i] = last
    return out


def diff_series(s: list[float | None]) -> list[float | None]:
    out: list[float | None] = [None] * len(s)
    for i in range(1, len(s)):
        if s[i] is None or s[i - 1] is None:
            out[i] = None
        else:
            out[i] = s[i] - s[i - 1]
    return out


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 5:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy)


def cross_corr_ms(
    db: list[float | None],
    dp: list[float | None],
    step_ms: int,
    lag_min_ms: int,
    lag_max_ms: int,
    lag_step_ms: int,
) -> list[tuple[int, float, int]]:
    """Cross-correlation of Binance Δ vs Polymarket Δ at integer lag in ms.

    Positive lag => polymarket lags binance (binance leads).
    Returns list of (lag_ms, corr, n_pairs).
    """
    out = []
    for lag_ms in range(lag_min_ms, lag_max_ms + 1, lag_step_ms):
        k = lag_ms // step_ms  # in grid units
        if k >= 0:
            xs = db[: len(db) - k]
            ys = dp[k:]
        else:
            xs = db[-k:]
            ys = dp[: len(dp) + k]
        pairs = [(a, b) for a, b in zip(xs, ys, strict=False) if a is not None and b is not None]
        if len(pairs) < 5:
            continue
        x = [a for a, _ in pairs]
        y = [b for _, b in pairs]
        out.append((lag_ms, pearson(x, y), len(pairs)))
    return out


# ------------------------ Big-move analysis ------------------------------- #


def big_move_responses(
    binance_ticks: list[Tick],
    poly_ticks: list[Tick],
    top_k: int = 5,
    win_ms: int = 1000,
    response_ms: int = 3000,
) -> list[dict]:
    """Find top_k largest binance moves over win_ms window; report the next
    polymarket midpoint move in [t, t+response_ms]."""
    bn = sorted(binance_ticks, key=lambda x: x.t_ms)
    pm = sorted(poly_ticks, key=lambda x: x.t_ms)
    if len(bn) < 5 or len(pm) < 5:
        return []
    moves = []
    j = 0  # left pointer for win_ms-back lookup
    for i in range(len(bn)):
        ti = bn[i].t_ms
        while j < i and bn[j].t_ms < ti - win_ms:
            j += 1
        delta = bn[i].value - bn[j].value
        moves.append((i, ti, delta, bn[i].value, bn[j].value))
    moves.sort(key=lambda m: -abs(m[2]))
    seen_windows: list[int] = []
    out: list[dict] = []
    for _i, ti, delta, vi, vj in moves:
        # de-duplicate clustered moves: skip if within 5s of an already-picked one
        if any(abs(ti - s) < 5000 for s in seen_windows):
            continue
        # find polymarket value at-or-before ti, and within (ti, ti+response_ms]
        pm_at = None
        pm_after = None
        first_after_ms = None
        for tk in pm:
            if tk.t_ms <= ti:
                pm_at = tk.value
            elif ti < tk.t_ms <= ti + response_ms:
                if pm_after is None or tk.value != pm_at:
                    if first_after_ms is None and pm_at is not None and tk.value != pm_at:
                        first_after_ms = tk.t_ms - ti
                    pm_after = tk.value
            else:
                break
        out.append(
            {
                "binance_event_ms": ti,
                "binance_window_ms": win_ms,
                "binance_move_usd": round(delta, 2),
                "binance_mid_after": round(vi, 2),
                "binance_mid_before": round(vj, 2),
                "poly_up_at_event": pm_at,
                "poly_up_after_response_window": pm_after,
                "first_poly_change_after_ms": first_after_ms,
                "response_window_ms": response_ms,
            }
        )
        seen_windows.append(ti)
        if len(out) >= top_k:
            break
    return out


# ------------------------ Cadence stats ----------------------------------- #


def mean_interval_ms(ticks: list[Tick], source: str) -> float | None:
    arr = sorted([t.t_ms for t in ticks if t.source == source])
    if len(arr) < 2:
        return None
    diffs = [arr[i] - arr[i - 1] for i in range(1, len(arr))]
    return sum(diffs) / len(diffs)


# ------------------------ Main ------------------------------------------- #


async def amain() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=240)
    p.add_argument(
        "--poly-hz",
        type=float,
        default=8.0,
        help="polymarket poll rate (Hz). 5-10 is the safe range.",
    )
    p.add_argument("--out", default="/tmp/btc_arb_ws.json")
    p.add_argument("--grid-step-ms", type=int, default=50)
    p.add_argument("--lag-min-ms", type=int, default=-2000)
    p.add_argument("--lag-max-ms", type=int, default=2000)
    args = p.parse_args()

    print("discovering active 5m btc-updown market...", file=sys.stderr)
    with httpx.Client(timeout=4.0) as c:
        m5 = discover_active_market("5m", c)
    if not m5:
        print("ERROR: no active 5m btc-updown market found", file=sys.stderr)
        return 1
    token_ids = json.loads(m5["clobTokenIds"])
    up_token = token_ids[0]
    print(f"5m  -> {m5['slug']} ends {m5['endDate']}  up_token={up_token[:14]}...", file=sys.stderr)

    ticks: list[Tick] = []
    ticks_lock = threading.Lock()
    asyncio_stop = asyncio.Event()
    thread_stop = threading.Event()

    poly_thread = threading.Thread(
        target=poly_poll_loop,
        args=(up_token, ticks, ticks_lock, thread_stop, args.poly_hz),
        daemon=True,
    )
    poly_thread.start()

    t0 = time.time()
    print(
        f"streaming for {args.duration}s (binance WS + polymarket @ {args.poly_hz} Hz)...",
        file=sys.stderr,
    )

    async def progress() -> None:
        while not asyncio_stop.is_set():
            await asyncio.sleep(20.0)
            with ticks_lock:
                nb = sum(1 for t in ticks if t.source == "binance")
                np_ = sum(1 for t in ticks if t.source == "polymarket")
            elapsed = time.time() - t0
            print(f"  t+{elapsed:5.0f}s  binance_ticks={nb}  poly_ticks={np_}", file=sys.stderr)

    async def runner() -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(binance_ws_loop(ticks, asyncio_stop), timeout=args.duration)

    progress_task = asyncio.create_task(progress())
    try:
        await runner()
    finally:
        asyncio_stop.set()
        thread_stop.set()
        progress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await progress_task
        poly_thread.join(timeout=2.0)

    with ticks_lock:
        local_ticks = list(ticks)

    bn_ticks = [t for t in local_ticks if t.source == "binance"]
    pm_ticks = [t for t in local_ticks if t.source == "polymarket"]
    print(f"\ncollected: binance={len(bn_ticks)}  polymarket={len(pm_ticks)}", file=sys.stderr)

    bn_mean_ms = mean_interval_ms(local_ticks, "binance")
    pm_mean_ms = mean_interval_ms(local_ticks, "polymarket")

    if not bn_ticks or not pm_ticks:
        print("ERROR: insufficient ticks for analysis", file=sys.stderr)
        return 1

    t_start_ms = max(min(t.t_ms for t in bn_ticks), min(t.t_ms for t in pm_ticks))
    t_end_ms = min(max(t.t_ms for t in bn_ticks), max(t.t_ms for t in pm_ticks))
    step_ms = args.grid_step_ms

    bn_grid = resample_uniform(local_ticks, "binance", t_start_ms, t_end_ms, step_ms)
    pm_grid = resample_uniform(local_ticks, "polymarket", t_start_ms, t_end_ms, step_ms)
    db = diff_series(bn_grid)
    dp = diff_series(pm_grid)

    cc = cross_corr_ms(db, dp, step_ms, args.lag_min_ms, args.lag_max_ms, step_ms)
    cc_sorted = sorted(cc, key=lambda r: -r[1])
    best_lag_ms, best_corr, best_n = cc_sorted[0] if cc_sorted else (None, None, None)

    big_moves = big_move_responses(bn_ticks, pm_ticks, top_k=5, win_ms=1000, response_ms=3000)

    out = {
        "started": datetime.fromtimestamp(t0, tz=UTC).isoformat(),
        "duration_s": args.duration,
        "market": {"slug": m5["slug"], "up_token": up_token, "endDate": m5["endDate"]},
        "n_binance": len(bn_ticks),
        "n_polymarket": len(pm_ticks),
        "mean_binance_interval_ms": bn_mean_ms,
        "mean_polymarket_interval_ms": pm_mean_ms,
        "grid_step_ms": step_ms,
        "best_lag_ms": best_lag_ms,
        "best_corr": round(best_corr, 4) if best_corr is not None else None,
        "best_n_pairs": best_n,
        "cross_corr_ms": [{"lag_ms": lag, "corr": round(c, 4), "n": n} for lag, c, n in cc],
        "big_moves": big_moves,
    }

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))  # noqa: ASYNC240
    print(f"wrote {args.out}", file=sys.stderr)
    print("\n=== Sub-second lead-lag analysis ===", file=sys.stderr)
    print(f"binance updates  : mean Δt = {bn_mean_ms:.1f} ms  (n={len(bn_ticks)})", file=sys.stderr)
    print(f"polymarket polls : mean Δt = {pm_mean_ms:.1f} ms  (n={len(pm_ticks)})", file=sys.stderr)
    if best_lag_ms is not None:
        print(
            f"best lag         : {best_lag_ms:+d} ms  (corr={best_corr:.4f}  n={best_n})",
            file=sys.stderr,
        )
    print("\n=== Top binance moves and polymarket response ===", file=sys.stderr)
    for m in big_moves:
        fpc = m["first_poly_change_after_ms"]
        fpc_s = f"{fpc} ms" if fpc is not None else "no change"
        print(
            f"  Δ_binance(1s)={m['binance_move_usd']:+8.2f} USD   "
            f"poly: {m['poly_up_at_event']} -> {m['poly_up_after_response_window']}   "
            f"first_change_after={fpc_s}",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
