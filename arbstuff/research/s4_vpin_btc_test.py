"""
VPIN (Volume-Synchronized Probability of Informed Trading) on BTCUSDT.

Reference: Easley, López de Prado, O'Hara (2012) "Flow Toxicity and Liquidity
in a High-Frequency World", Review of Financial Studies 25(5).

Methodology:
  - τ = bucket volume = mean(quote_volume) * 5  (~5min of volume per bucket)
  - Two classifiers:
       direct:  V_buy = Σ taker_buy_quote_volume_k   (Binance gives aggressor side)
       bulk:    V_buy = Σ vol_k * Φ((Δp_k)/σ_Δp)    (Easley et al. BVC)
  - VPIN[i] = mean_{j∈last N buckets}  |V_buy_j - V_sell_j| / τ      (N=50)
  - Forward test: regress RV_{t+30m} on VPIN_t with HAC SE.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import urllib.request
from typing import Any

import numpy as np

BINANCE = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
N_BUCKETS_WINDOW = 50
TAU_MULT = 5  # τ = 5× mean quote_volume per minute


def fetch_klines(symbol: str, interval: str, limit: int = 1000,
                 end_time: int | None = None) -> list[list[Any]]:
    url = f"{BINANCE}?symbol={symbol}&interval={interval}&limit={limit}"
    if end_time is not None:
        url += f"&endTime={end_time}"
    req = urllib.request.Request(url, headers={"User-Agent": "vpin-test/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_last_24h() -> list[list[Any]]:
    now_ms = int(time.time() * 1000)
    one_day_ms = 24 * 60 * 60 * 1000
    # 1440 minutes in 24h → two calls of 1000 + 440 with endTime windows
    # Easier: pull the latest 1000, then pull another 1000 ending at the first batch's first open_time.
    batch_recent = fetch_klines(SYMBOL, INTERVAL, limit=1000)
    first_open = batch_recent[0][0]
    batch_older = fetch_klines(SYMBOL, INTERVAL, limit=1000, end_time=first_open - 1)
    combined = batch_older + batch_recent
    # Trim to last 24h
    cutoff = now_ms - one_day_ms
    combined = [k for k in combined if k[0] >= cutoff]
    # Dedup + sort
    seen: dict[int, list[Any]] = {}
    for k in combined:
        seen[k[0]] = k
    return [seen[t] for t in sorted(seen)]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def build_buckets(klines: list[list[Any]], tau: float) -> list[dict[str, float]]:
    """Sweep klines, emitting buckets whose cumulative quote_volume reaches τ.

    Each kline may be split across buckets (proportional allocation) so that
    every closed bucket has total quote volume exactly = τ.
    """
    buckets: list[dict[str, float]] = []
    cur_total = 0.0
    cur_buy_direct = 0.0
    cur_buy_bulk = 0.0
    cur_start_idx = 0
    # Precompute log returns of close
    closes = [float(k[4]) for k in klines]
    log_rets = [0.0]
    for i in range(1, len(closes)):
        log_rets.append(math.log(closes[i] / closes[i - 1]))
    sigma_dp = statistics.pstdev(log_rets[1:]) or 1e-9

    for i, k in enumerate(klines):
        q_vol = float(k[7])
        tb_q_vol = float(k[10])  # taker-buy aggressor quote vol
        if q_vol <= 0:
            continue
        buy_frac_direct = tb_q_vol / q_vol
        buy_frac_bulk = norm_cdf(log_rets[i] / sigma_dp)

        remaining_q = q_vol
        while remaining_q > 0:
            need = tau - cur_total
            take = min(need, remaining_q)
            cur_total += take
            cur_buy_direct += take * buy_frac_direct
            cur_buy_bulk += take * buy_frac_bulk
            remaining_q -= take
            if cur_total >= tau - 1e-9:
                end_ms = int(k[6])
                buckets.append({
                    "end_ms": end_ms,
                    "kline_idx_end": i,
                    "V_buy_direct": cur_buy_direct,
                    "V_buy_bulk": cur_buy_bulk,
                    "tau": cur_total,
                })
                cur_total = 0.0
                cur_buy_direct = 0.0
                cur_buy_bulk = 0.0
                cur_start_idx = i + 1
    return buckets


def rolling_vpin(buckets: list[dict[str, float]], key_buy: str, window: int) -> list[float | None]:
    vpin: list[float | None] = []
    for i in range(len(buckets)):
        if i + 1 < window:
            vpin.append(None)
            continue
        s = 0.0
        for j in range(i + 1 - window, i + 1):
            v_buy = buckets[j][key_buy]
            tau = buckets[j]["tau"]
            v_sell = tau - v_buy
            s += abs(v_buy - v_sell) / tau
        vpin.append(s / window)
    return vpin


def pct(arr: list[float], p: float) -> float:
    if not arr:
        return float("nan")
    a = sorted(arr)
    k = (len(a) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return a[int(k)]
    return a[f] * (c - k) + a[c] * (k - f)


def stats_block(name: str, vals: list[float]) -> str:
    return (f"{name}: n={len(vals)} mean={statistics.fmean(vals):.4f} "
            f"p50={pct(vals,50):.4f} p90={pct(vals,90):.4f} "
            f"p99={pct(vals,99):.4f} max={max(vals):.4f}")


def hac_ols(x: np.ndarray, y: np.ndarray, maxlags: int) -> dict[str, float]:
    """OLS with Newey-West HAC SE.  Returns alpha, beta, t_beta, R²."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    yhat = X @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Newey-West
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = (u.T @ u) / n
    for L in range(1, maxlags + 1):
        w = 1.0 - L / (maxlags + 1.0)
        G = (u[L:].T @ u[:-L]) / n
        S = S + w * (G + G.T)
    cov = n * XtX_inv @ S @ XtX_inv
    se_beta = math.sqrt(cov[1, 1])
    t_beta = beta[1] / se_beta if se_beta > 0 else float("nan")
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "t_beta": t_beta, "R2": r2, "se_beta": se_beta}


def main() -> None:
    print("Fetching last 24h of BTCUSDT 1m klines ...")
    klines = fetch_last_24h()
    print(f"  got {len(klines)} klines (first open_time={klines[0][0]}, last close_time={klines[-1][6]})")

    quote_vols = [float(k[7]) for k in klines]
    mean_qv = statistics.fmean(quote_vols)
    tau = mean_qv * TAU_MULT
    print(f"  mean per-minute quote_volume = ${mean_qv:,.0f}")
    print(f"  τ (bucket size) = ${tau:,.0f}")

    buckets = build_buckets(klines, tau)
    print(f"  built {len(buckets)} volume buckets")
    if len(buckets) < N_BUCKETS_WINDOW + 5:
        print("WARNING: too few buckets for a stable rolling VPIN window.")
    # Adjust window if necessary
    window = min(N_BUCKETS_WINDOW, max(10, len(buckets) // 4))
    print(f"  rolling window = {window} buckets")

    vpin_direct = rolling_vpin(buckets, "V_buy_direct", window)
    vpin_bulk = rolling_vpin(buckets, "V_buy_bulk", window)

    vd = [v for v in vpin_direct if v is not None]
    vb = [v for v in vpin_bulk if v is not None]
    print()
    print(stats_block("VPIN_direct", vd))
    print(stats_block("VPIN_bulk  ", vb))

    # Correlation between the two methods at aligned indices
    aligned_d, aligned_b = [], []
    for d, b in zip(vpin_direct, vpin_bulk):
        if d is not None and b is not None:
            aligned_d.append(d)
            aligned_b.append(b)
    if aligned_d:
        corr = float(np.corrcoef(aligned_d, aligned_b)[0, 1])
        print(f"  corr(VPIN_direct, VPIN_bulk) = {corr:.4f}")

    # Bucket-level spike inspection: top-5 VPIN_direct buckets vs subsequent log-return magnitude
    closes = [float(k[4]) for k in klines]
    # map kline_idx_end -> close-price; compute realized vol over next 30 minutes after each bucket
    # RV next 30m = sqrt(sum log_ret^2 over next 30 1-min bars)
    log_rets_full = [0.0]
    for i in range(1, len(closes)):
        log_rets_full.append(math.log(closes[i] / closes[i - 1]))

    def fwd_rv(idx_end: int, horizon: int = 30) -> float | None:
        start = idx_end + 1
        end = start + horizon
        if end > len(log_rets_full):
            return None
        s = sum(r * r for r in log_rets_full[start:end])
        return math.sqrt(s)

    # Top spikes
    enumerated = [(i, v) for i, v in enumerate(vpin_direct) if v is not None]
    enumerated.sort(key=lambda t: t[1], reverse=True)
    print("\nTop-5 VPIN_direct spikes (and forward 30m realized vol):")
    for i, v in enumerated[:5]:
        b = buckets[i]
        rv = fwd_rv(b["kline_idx_end"], 30)
        end_iso = time.strftime("%H:%M", time.gmtime(b["end_ms"] / 1000))
        print(f"  bucket #{i:3d} end={end_iso}Z  VPIN_d={v:.4f}  RV_fwd30m={rv}")

    # Predictive regression: RV_{t+30m} on VPIN_t
    Xs_d, Xs_b, Ys = [], [], []
    for i, b in enumerate(buckets):
        d = vpin_direct[i]
        bk = vpin_bulk[i]
        rv = fwd_rv(b["kline_idx_end"], 30)
        if d is None or bk is None or rv is None:
            continue
        Xs_d.append(d)
        Xs_b.append(bk)
        Ys.append(rv)
    if len(Ys) > 10:
        x_d = np.asarray(Xs_d, dtype=float)
        x_b = np.asarray(Xs_b, dtype=float)
        y = np.asarray(Ys, dtype=float)
        print(f"\nForward-RV regression (N={len(y)} bucket-aligned obs, horizon=30 1-min bars):")
        rd = hac_ols(x_d, y, maxlags=5)
        rb = hac_ols(x_b, y, maxlags=5)
        print(f"  RV ~ a + β·VPIN_direct + ε   β={rd['beta']:.4f}  t={rd['t_beta']:.2f}  R²={rd['R2']:.4f}")
        print(f"  RV ~ a + β·VPIN_bulk   + ε   β={rb['beta']:.4f}  t={rb['t_beta']:.2f}  R²={rb['R2']:.4f}")
    else:
        print("\nNot enough aligned observations to run regression.")


if __name__ == "__main__":
    main()
