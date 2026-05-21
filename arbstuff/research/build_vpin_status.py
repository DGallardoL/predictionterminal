"""Compute live VPIN_bulk on BTCUSDT and write a status file the arb engine
can read as a "toxic-flow gate" — pause alerts when VPIN_bulk > p90 ≈ 0.45.

Theory: Easley, López de Prado, O'Hara (2012) RFS, "Flow Toxicity and
Liquidity in a High-Frequency World". VPIN = E[|V_buy − V_sell|] / E[V_buy +
V_sell] over equal-volume buckets. Bulk-volume classifier uses
``V_buy = Σ vol · Φ(Δp / σ_Δp)`` rather than aggressor-side, which on BTC
Binance spot is contaminated by HFT rebalancing.

Empirical (24h test, 2026-05-19, agent S4):
  - VPIN_bulk:   mean 0.331, p50 0.317, p90 0.453, p99 0.500, max 0.504
  - Regression RV_{t+30m} = α + β·VPIN_t: β=+0.0028, t=+2.71, R²=5.8%
  - VPIN_direct (taker-buy) does NOT predict — use bulk only.

Run periodically (e.g. every 5 min via cron or a sidecar loop). Writes:
  arbstuff/vpin_status.json   { vpin_bulk, level: low|warn|halt, as_of, ... }

The engine treats:
  - vpin_bulk > 0.45  ⇒  level="warn"   (degrade alerts to log-only)
  - vpin_bulk > 0.50  ⇒  level="halt"   (skip alert + execute)
"""

from __future__ import annotations

import json
import math
import statistics as stat
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ARBSTUFF = HERE.parent
STATUS_PATH = ARBSTUFF / "vpin_status.json"

UA = "Mozilla/5.0 pfm-vpin-research"

# Wave-6 (2026-05-19) multi-window audit: β > 0 in 4/4 windows but |t|>2
# only in 1/4. p90 ranges 0.428→0.547 (28%), warn-trigger frequency swings
# 0.4%→17.2% across regimes. The signal is real but regime-driven.
#
# Wave-6 follow-up (2026-05-19, this commit): SWITCHED FROM FIXED to
# ROLLING-WINDOW thresholds. We compute p85 and p98 over the recent rolling
# VPIN history (a longer kline pull => more buckets => more rolling-VPIN
# samples => a percentile that *reflects the current regime*, not an
# absolute level that fires 17.2% in vol windows and 0.4% in calm ones).
# Fallback constants are kept only for the cold-start case (insufficient
# history) and for the `_compute_thresholds()` guard rails.
WARN_PCTL = 85  # p85 over rolling-VPIN series
HALT_PCTL = 98  # p98 over rolling-VPIN series
WARN_VPIN_FALLBACK = 0.45  # used only when we don't have enough samples to compute a percentile
HALT_VPIN_FALLBACK = 0.55
# Clip rolling thresholds into a sane band so a degenerate calm regime
# doesn't push WARN below 0.30 (would fire constantly) and a wild vol
# regime doesn't push HALT above 0.65 (would never fire).
WARN_FLOOR, WARN_CEIL = 0.30, 0.55
HALT_FLOOR, HALT_CEIL = 0.40, 0.65

# Bucket size as multiple of mean per-bar quote volume — 5 gives ~5-min
# buckets on BTCUSDT during normal regimes.
BUCKET_MULT = 5
ROLL_WINDOW = 50
# History pull: 7 days of 1-min klines = 10,080 bars. With BUCKET_MULT=5
# and ROLL_WINDOW=50, that yields ~1,950 rolling-VPIN samples — enough for
# stable p85/p98 estimates while keeping the percentile responsive to the
# *recent* regime (older bars naturally roll out as we re-run every 5 min).
KLINES_LIMIT = 10000  # Binance max is 1000 per request, so we paginate


def _fetch_klines(symbol: str = "BTCUSDT", limit: int = 1000) -> list:
    """Fetch ``limit`` 1-min klines, paginating in batches of 1000 (Binance max).

    The result is in chronological order. We page *backwards* from "now" using
    each batch's earliest open_time as the next batch's ``endTime``. This keeps
    the rolling-percentile threshold reflective of the most recent ``limit``
    minutes rather than an arbitrary fixed window.
    """
    MAX_PER_REQ = 1000
    out: list = []
    end_time: int | None = None  # ms epoch; None = "now"
    remaining = max(1, int(limit))
    while remaining > 0:
        batch = min(remaining, MAX_PER_REQ)
        url = (
            f"https://api.binance.com/api/v3/klines?"
            f"symbol={symbol}&interval=1m&limit={batch}"
        )
        if end_time is not None:
            url += f"&endTime={end_time}"
        req = Request(url, headers={"User-Agent": UA})
        try:
            with urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
        except (HTTPError, OSError):
            break
        if not data:
            break
        # Prepend so the final list stays chronological.
        out = data + out
        remaining -= len(data)
        # Next batch ends 1 ms before the current oldest open_time.
        end_time = int(data[0][0]) - 1
        if len(data) < batch:
            # Exchange returned fewer than asked — no more history.
            break
    return out


def _compute_thresholds(vpin_series: list[float]) -> tuple[float, float]:
    """Return ``(warn, halt)`` clipped into the configured safety bands.

    Uses the empirical p85/p98 of the rolling-VPIN series. Falls back to the
    fixed constants when we have fewer than 100 samples (not enough to fit a
    p98 robustly — the top 2% would be < 2 points).
    """
    if len(vpin_series) < 100:
        return WARN_VPIN_FALLBACK, HALT_VPIN_FALLBACK
    # statistics.quantiles with n=100 gives the 1st-99th percentile cut points;
    # index i corresponds to the (i+1)-th percentile.
    cuts = stat.quantiles(vpin_series, n=100, method="exclusive")
    warn = cuts[WARN_PCTL - 1]
    halt = cuts[HALT_PCTL - 1]
    warn = min(max(warn, WARN_FLOOR), WARN_CEIL)
    halt = min(max(halt, HALT_FLOOR), HALT_CEIL)
    # Guarantee halt > warn even after clipping.
    if halt <= warn:
        halt = min(WARN_CEIL + 0.05, HALT_CEIL)
    return warn, halt


def _phi(z: float) -> float:
    """Standard normal CDF (Abramowitz-Stegun-style erf approximation)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_vpin_bulk(klines: list) -> dict:
    """Return rolling VPIN_bulk on the supplied 1-min kline window."""
    if len(klines) < ROLL_WINDOW * 2:
        return {"ok": False, "reason": f"need ≥ {ROLL_WINDOW*2} klines, got {len(klines)}"}

    closes = [float(k[4]) for k in klines]
    qvols = [float(k[7]) for k in klines]
    # 1-min log returns
    logr = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    sigma_dp = stat.pstdev(logr) or 1e-9

    # Pair each return with the kline AFTER the one that produced it.
    # Bucket by cumulative quote volume.
    tau = (sum(qvols) / len(qvols)) * BUCKET_MULT
    buckets: list[tuple[float, float]] = []  # (V_buy_bulk, V_total)
    cur_buy = 0.0
    cur_total = 0.0
    for i, (vol, lr) in enumerate(zip(qvols[1:], logr), start=1):
        phi = _phi(lr / sigma_dp)
        v_buy = vol * phi
        cur_buy += v_buy
        cur_total += vol
        if cur_total >= tau:
            buckets.append((cur_buy, cur_total))
            cur_buy = 0.0
            cur_total = 0.0

    if len(buckets) < ROLL_WINDOW + 1:
        return {"ok": False, "reason": f"only {len(buckets)} buckets — need {ROLL_WINDOW+1}"}

    # Rolling VPIN_bulk: mean of |V_buy − V_sell| / V_total over ROLL_WINDOW.
    vpin_series = []
    for end in range(ROLL_WINDOW, len(buckets) + 1):
        window = buckets[end - ROLL_WINDOW:end]
        num = sum(abs(b - (t - b)) for b, t in window)
        den = sum(t for _, t in window) or 1.0
        vpin_series.append(num / den)

    vpin_now = vpin_series[-1]
    warn_thr, halt_thr = _compute_thresholds(vpin_series)
    level = "halt" if vpin_now >= halt_thr else ("warn" if vpin_now >= warn_thr else "low")

    return {
        "ok": True,
        "vpin_bulk_now": vpin_now,
        "vpin_bulk_p50": stat.median(vpin_series),
        "vpin_bulk_p90": stat.quantiles(vpin_series, n=10)[8] if len(vpin_series) >= 10 else None,
        "vpin_bulk_max": max(vpin_series),
        "level": level,
        "warn_threshold": warn_thr,
        "halt_threshold": halt_thr,
        "threshold_mode": (
            "rolling_p85_p98" if len(vpin_series) >= 100 else "fallback_fixed"
        ),
        "threshold_history_n": len(vpin_series),
        "n_buckets": len(buckets),
        "rolling_window": ROLL_WINDOW,
    }


def write_status(payload: dict) -> None:
    payload = dict(payload)
    payload["as_of"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["symbol"] = "BTCUSDT"
    payload["interval"] = "1m"
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(STATUS_PATH)


def main() -> None:
    klines = _fetch_klines(limit=KLINES_LIMIT)
    if not klines:
        write_status({"ok": False, "reason": "binance fetch failed", "level": "unknown"})
        sys.exit(0)
    result = compute_vpin_bulk(klines)
    write_status(result)
    if result.get("ok"):
        print(f"VPIN_bulk now: {result['vpin_bulk_now']:.4f}  level: {result['level']}")
        print(
            f"  p50={result['vpin_bulk_p50']:.4f} "
            f"p90={result['vpin_bulk_p90']} max={result['vpin_bulk_max']:.4f}"
        )
        print(
            f"  thresholds: WARN={result['warn_threshold']:.4f} "
            f"HALT={result['halt_threshold']:.4f} "
            f"mode={result['threshold_mode']} "
            f"hist_n={result['threshold_history_n']}"
        )
    else:
        print(f"VPIN compute failed: {result.get('reason')}")


if __name__ == "__main__":
    main()
