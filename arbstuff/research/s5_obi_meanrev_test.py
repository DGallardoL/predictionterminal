"""Test Cartea-Jaimungal-Penalva OBI mean-reversion hypothesis on live BTCUSDT.

Hypothesis: OBI = (bidQty - askQty)/(bidQty + askQty) predicts NEGATIVE short-horizon
return at the tick level on liquid crypto (mean reversion via absorption of imbalance).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

URL = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT"
N_SAMPLES = 300
SLEEP_S = 1.0
HORIZONS = [1, 5, 15, 30]
OUT = Path("/tmp/obi_test")


def collect() -> pd.DataFrame:
    rows = []
    sess = requests.Session()
    t0 = time.time()
    for i in range(N_SAMPLES):
        target = t0 + i * SLEEP_S
        try:
            r = sess.get(URL, timeout=4)
            r.raise_for_status()
            d = r.json()
            ts = time.time()
            bp, bq = float(d["bidPrice"]), float(d["bidQty"])
            ap, aq = float(d["askPrice"]), float(d["askQty"])
            mid = 0.5 * (bp + ap)
            obi = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0.0
            rows.append({"i": i, "ts": ts, "bidPrice": bp, "bidQty": bq,
                         "askPrice": ap, "askQty": aq, "mid": mid, "obi": obi})
        except Exception as e:
            rows.append({"i": i, "ts": time.time(), "bidPrice": np.nan, "bidQty": np.nan,
                         "askPrice": np.nan, "askQty": np.nan, "mid": np.nan, "obi": np.nan,
                         "err": str(e)})
        # pace
        remaining = target + SLEEP_S - time.time()
        if remaining > 0:
            time.sleep(remaining)
        if i % 30 == 0:
            print(f"  sample {i}/{N_SAMPLES} mid={rows[-1].get('mid')} obi={rows[-1].get('obi'):.4f}"
                  if not math.isnan(rows[-1].get("mid", float("nan"))) else f"  sample {i} ERR")
    df = pd.DataFrame(rows).dropna(subset=["mid", "obi"]).reset_index(drop=True)
    return df


def analyse(df: pd.DataFrame) -> dict:
    out: dict = {"n_samples": int(len(df)),
                 "obi_mean": float(df["obi"].mean()),
                 "obi_std": float(df["obi"].std()),
                 "mid_first": float(df["mid"].iloc[0]),
                 "mid_last": float(df["mid"].iloc[-1]),
                 "spread_bps_mean": float(((df["askPrice"] - df["bidPrice"]) /
                                           df["mid"] * 1e4).mean()),
                 "horizons": {}}
    log_mid = np.log(df["mid"].values)
    obi = df["obi"].values
    for h in HORIZONS:
        if len(df) <= h + 5:
            continue
        r = log_mid[h:] - log_mid[:-h]    # r_{t+h} = log(mid_{t+h}/mid_t)
        x = obi[:-h]
        # OLS with HAC (Newey-West, 5 lags)
        X = sm.add_constant(x)
        model = sm.OLS(r, X)
        try:
            res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        except Exception:
            res = model.fit()
        beta = float(res.params[1])
        tstat = float(res.tvalues[1])
        r2 = float(res.rsquared)
        # decile means
        try:
            buckets = pd.qcut(pd.Series(x), 10, labels=False, duplicates="drop")
            dec = pd.DataFrame({"b": buckets, "r": r}).groupby("b")["r"].mean()
            # convert to bps and to annualized bps for intuition
            dec_bps = (dec * 1e4).round(3).to_dict()
        except Exception:
            dec_bps = {}
        # verdict
        signal_mr = (abs(tstat) >= 3.0) and (r2 > 0.005) and (beta < 0)
        signal_mom = (abs(tstat) >= 3.0) and (r2 > 0.005) and (beta > 0)
        out["horizons"][h] = {
            "beta": beta,
            "tstat_HAC5": tstat,
            "r2": r2,
            "n": int(len(r)),
            "decile_mean_r_bps": dec_bps,
            "verdict": ("MEAN_REVERSION_SIGNAL" if signal_mr else
                        "MOMENTUM_SIGNAL" if signal_mom else "NO_SIGNAL"),
        }
    return out


def main():
    print(f"Collecting {N_SAMPLES} bookTicker samples at {SLEEP_S}s cadence "
          f"(~{N_SAMPLES*SLEEP_S/60:.1f} min)...")
    df = collect()
    df.to_csv(OUT / "tape.csv", index=False)
    print(f"Captured {len(df)} clean rows.")
    res = analyse(df)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
