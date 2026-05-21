"""Validation A4 — does σ_PM minus its benchmark predict forward realized σ?

Empirical check of the prediction-market implied volatility (σ_PM) gap
signal across the 5 ladder assets in ``pfm.vol.pm_iv_extractor``.

Pipeline
--------
1. For every slug in ``LADDER_REGISTRY``, load a daily prob-history series
   from one of three sources, in priority order:
     a. the local cache pickle ``/tmp/pm_iv_validation_cache.pkl`` (written
        by previous runs of this script);
     b. the project-wide cache ``/tmp/strat7_factor_history.pkl``;
     c. a live ``fetch_factor_history`` call with 1s spacing.
   All slugs end up in the validation cache so re-runs are instant.
2. For every UTC calendar day, build a synthetic ``LadderFamily`` from the
   day's ladder snapshots (allowing 1 day of forward-fill per slug) and
   call ``fit_implied_sigma``. The resulting daily σ_PM series is stored
   per asset.
3. Pull daily σ_benchmark history:
     * SPX  → FRED VIXCLS
     * WTI  → FRED OVXCLS
     * GOLD → FRED GVZCLS
     * BTC  → Deribit historical DVOL if reachable, else Binance 30d
              realized σ on BTCUSDT.
     * ETH  → same, ETH variants.
   Each is normalized to decimal annualised σ on UTC midnight calendar.
4. Forward realized σ over the next 30 calendar days on the underlying:
     * SPX  → yfinance ^GSPC log returns, √252 annualization
     * BTC  → Binance BTCUSDT daily, √365
     * ETH  → Binance ETHUSDT daily, √365
     * WTI  → yfinance CL=F, √252
     * GOLD → yfinance GC=F, √252
5. Aggregate per asset:
     - corr(σ_PM, σ_fwd_realized)
     - corr(σ_PM - σ_bench, σ_fwd_realized - σ_bench)  -- KEY NUMBER
     - signaled-trade simulator (±2pp threshold) → hit rate, mean PnL,
       Sharpe (annualised), drop-top-3 Sharpe, N signals
6. Write ``docs/vol-pm-iv-validation.md`` and print one-page summary.

Run
---
    cd /Users/damiangallardoloya/Desktop/proyectofuentes
    PYTHONPATH=api/src api/.venv/bin/python api/scripts/validate_pm_iv_gap.py

Be intellectually honest: report sample size, missingness, and an explicit
"INSUFFICIENT DATA" verdict when warranted. Do not manufacture an alpha.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap import path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_API_SRC = _ROOT / "api" / "src"
if str(_API_SRC) not in sys.path:
    sys.path.insert(0, str(_API_SRC))

from pfm.sources.fred import fetch_fred_series
from pfm.sources.polymarket import (
    PolymarketClient,
    fetch_factor_history,
)
from pfm.vol.pm_iv_extractor import (
    LADDER_REGISTRY,
    LadderEntry,
    LadderFamily,
    fit_implied_sigma,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validate_pm_iv_gap")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PM_CACHE = Path("/tmp/pm_iv_validation_cache.pkl")
STRAT7_CACHE = Path("/tmp/strat7_factor_history.pkl")
REPORT = _ROOT / "docs" / "vol-pm-iv-validation.md"
TODAY = pd.Timestamp(datetime.now(tz=UTC)).normalize()
WINDOW_START = pd.Timestamp("2025-10-01", tz="UTC")  # earliest reasonable PM coverage

# ASSET_DEFAULT_SPOT mirrors pm_iv_extractor; we use it for the LadderFamily
# stub when constructing daily snapshots — sigma_annual is insensitive to
# spot for the "above" family but the LadderFamily schema requires it.
_ASSET_DEFAULT_SPOT: dict[str, float] = {
    "SPX": 6_000.0,
    "BTC": 110_000.0,
    "ETH": 4_000.0,
    "WTI": 75.0,
    "GOLD": 4_000.0,
}


# ---------------------------------------------------------------------------
# Step 1 — slug history loader (cache-first, polymarket fallback)
# ---------------------------------------------------------------------------
def load_slug_history(
    slugs: list[str],
    *,
    use_live: bool = True,
    live_budget_s: float = 180.0,
) -> dict[str, pd.Series]:
    """Return ``slug → daily prob Series (UTC-indexed)``.

    Priority:
        1. validation cache (`/tmp/pm_iv_validation_cache.pkl`)
        2. strat7 cache (`/tmp/strat7_factor_history.pkl`)
        3. live `fetch_factor_history` (only if ``use_live`` and budget left)
    Live fetches sleep 1s between calls and stop when ``live_budget_s``
    elapses, returning whatever was collected.
    """
    out: dict[str, pd.Series] = {}

    if PM_CACHE.exists():
        try:
            with PM_CACHE.open("rb") as fh:
                cached = pickle.load(fh)
            if isinstance(cached, dict):
                for s in slugs:
                    if s in cached and isinstance(cached[s], pd.Series) and not cached[s].empty:
                        out[s] = cached[s]
            print(f"[cache] validation pickle: hit {len(out)}/{len(slugs)}")
        except Exception as e:
            print(f"[cache] validation pickle unreadable ({e}); ignoring")

    if STRAT7_CACHE.exists():
        try:
            with STRAT7_CACHE.open("rb") as fh:
                strat7 = pickle.load(fh)
            promoted = 0
            for s in slugs:
                if s in out:
                    continue
                v = strat7.get(s)
                if isinstance(v, pd.Series) and not v.empty:
                    out[s] = v
                    promoted += 1
            print(
                f"[cache] strat7 pickle: promoted {promoted} more (total {len(out)}/{len(slugs)})"
            )
        except Exception as e:
            print(f"[cache] strat7 pickle unreadable ({e}); ignoring")

    missing = [s for s in slugs if s not in out]
    if missing and use_live:
        print(
            f"[live] attempting {len(missing)} live fetches with 1s spacing, budget {live_budget_s}s"
        )
        deadline = time.monotonic() + live_budget_s
        client = PolymarketClient(
            gamma_url="https://gamma-api.polymarket.com",
            clob_url="https://clob.polymarket.com",
            timeout=20.0,
        )
        try:
            for s in missing:
                if time.monotonic() > deadline:
                    print(
                        f"[live] budget exhausted with {len(missing) - len([k for k in missing if k in out])} slugs left"
                    )
                    break
                try:
                    df = fetch_factor_history(client, s)
                except Exception as e:
                    print(f"[live] {s[:50]}... FAILED: {type(e).__name__}: {str(e)[:80]}")
                    time.sleep(1.0)
                    continue
                if df is None or df.empty:
                    print(f"[live] {s[:50]}... empty")
                    time.sleep(1.0)
                    continue
                # df is indexed by 'date' UTC-naive midnight from fetch_factor_history
                if isinstance(df, pd.DataFrame):
                    ser = df["price"].copy()
                else:
                    ser = df.copy()
                # Normalise the index to UTC midnight
                idx = ser.index
                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                else:
                    idx = idx.tz_convert("UTC")
                ser.index = idx
                ser = ser[~ser.index.duplicated(keep="last")].sort_index()
                out[s] = ser
                print(f"[live] {s[:50]}... ok n={len(ser)}")
                time.sleep(1.0)
        finally:
            client.close()

    # Persist to validation cache.
    try:
        with PM_CACHE.open("wb") as fh:
            pickle.dump(out, fh)
        print(f"[cache] wrote {len(out)} series to {PM_CACHE}")
    except Exception as e:
        print(f"[cache] could not persist validation cache: {e}")
    return out


# ---------------------------------------------------------------------------
# Step 2 — daily σ_PM time series per asset
# ---------------------------------------------------------------------------
def build_sigma_pm_series(
    asset: str,
    history: dict[str, pd.Series],
) -> pd.Series:
    """For one asset, build the daily σ_PM time series.

    For each calendar day where ≥3 slugs in any family of the asset have a
    (≤1-day-stale) quote, we construct a LadderFamily snapshot and call
    fit_implied_sigma. If multiple families are configured for the asset
    (e.g. BTC has both ``above`` and ``dip_to``), we prefer the family with
    the most quotes that day; ties → first family in the registry.
    """
    cfg = LADDER_REGISTRY[asset]
    families = cfg["families"]
    spot = _ASSET_DEFAULT_SPOT.get(asset, 0.0)

    # Build per-slug forward-filled (≤1 day) daily series within window.
    full_idx = pd.date_range(WINDOW_START, TODAY, freq="D", tz="UTC")
    slug_daily: dict[str, pd.Series] = {}
    for fam in families:
        for slug, _strike, _direction, _venue in fam["entries"]:
            raw = history.get(slug)
            if raw is None or raw.empty:
                continue
            # Reindex & ffill at most 1 day
            r = raw.copy()
            # Normalise index to UTC daily
            idx = r.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            r.index = idx.normalize()
            r = r[~r.index.duplicated(keep="last")].sort_index()
            r = r.reindex(full_idx)
            r = r.ffill(limit=1)
            slug_daily[slug] = r

    if not slug_daily:
        return pd.Series(dtype=float, name=f"sigma_pm_{asset}")

    sigma_by_day: dict[pd.Timestamp, float] = {}
    for day in full_idx:
        # For each family, count how many slugs have a quote.
        best_quote_count = 0
        best_fam = None
        for fam in families:
            entries_today: list[LadderEntry] = []
            for slug, strike, direction, venue in fam["entries"]:
                ser = slug_daily.get(slug)
                if ser is None:
                    continue
                v = ser.loc[day] if day in ser.index else None
                if v is None or pd.isna(v):
                    continue
                entries_today.append(
                    LadderEntry(
                        slug=slug,
                        strike=float(strike),
                        direction=direction,
                        venue=venue,
                        market_value=float(v),
                    )
                )
            if len(entries_today) > best_quote_count:
                best_quote_count = len(entries_today)
                best_fam = (fam, entries_today)
        if best_fam is None or best_quote_count < 3:
            continue
        fam_dict, entries = best_fam
        mat = fam_dict["maturity_utc"]
        family = LadderFamily(
            asset=asset,
            asset_class=cfg["asset_class"],
            maturity_utc=mat,
            spot_at_lookup=spot,
            entries=entries,
        )
        try:
            res = fit_implied_sigma(family)
        except Exception as e:
            log.debug("fit failed %s %s: %s", asset, day.date(), e)
            continue
        # Sanity-clip: σ above 5 is almost certainly a numerical artifact.
        if 0 < res.sigma_annual < 5.0:
            sigma_by_day[day] = float(res.sigma_annual)
    s = pd.Series(sigma_by_day, name=f"sigma_pm_{asset}").sort_index()
    return s


# ---------------------------------------------------------------------------
# Step 3 — daily σ_benchmark series
# ---------------------------------------------------------------------------
def fetch_fred_bench(series_id: str) -> pd.Series:
    """Daily FRED series in decimal annualised σ form."""
    try:
        raw = fetch_fred_series(
            series_id,
            start=WINDOW_START,
            end=TODAY,
            forward_fill=True,
        )
    except Exception as e:
        print(f"[fred] {series_id} FAILED: {e}")
        return pd.Series(dtype=float, name=series_id)
    # Convert to UTC midnight index
    idx = raw.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    raw.index = idx.normalize()
    return (raw / 100.0).rename(f"sigma_bench_{series_id}")


def fetch_binance_daily_closes(symbol: str) -> pd.Series:
    """Daily Binance close series for ``symbol`` (e.g. BTCUSDT)."""
    # We pull ~250d in two calls if needed (max 1000 per request, more than enough).
    start_ms = int(WINDOW_START.timestamp() * 1000)
    url = "https://api.binance.com/api/v3/klines"
    closes: dict[pd.Timestamp, float] = {}
    cursor = start_ms
    try:
        with httpx.Client(timeout=20.0) as cli:
            for _ in range(8):  # 8 pages × 1000 = 8000 bars max
                resp = cli.get(
                    url,
                    params={
                        "symbol": symbol,
                        "interval": "1d",
                        "startTime": cursor,
                        "limit": 1000,
                    },
                )
                if resp.status_code != 200:
                    print(f"[binance] {symbol} HTTP {resp.status_code}: {resp.text[:120]}")
                    break
                rows = resp.json()
                if not rows:
                    break
                for r in rows:
                    open_ms = int(r[0])
                    close = float(r[4])
                    dt = pd.Timestamp(open_ms, unit="ms", tz="UTC").normalize()
                    closes[dt] = close
                if len(rows) < 1000:
                    break
                cursor = int(rows[-1][0]) + 86_400_000
    except Exception as e:
        print(f"[binance] {symbol} FAILED: {e}")
        return pd.Series(dtype=float, name=symbol)
    return pd.Series(closes, name=symbol).sort_index()


def fetch_deribit_dvol_history(asset: str) -> pd.Series:
    """Daily Deribit DVOL history for BTC or ETH. Empty Series on failure."""
    if asset not in ("BTC", "ETH"):
        return pd.Series(dtype=float)
    currency = asset
    start_ts_ms = int(WINDOW_START.timestamp() * 1000)
    end_ts_ms = int(TODAY.timestamp() * 1000)
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    try:
        with httpx.Client(timeout=20.0) as cli:
            resp = cli.get(
                url,
                params={
                    "currency": currency,
                    "start_timestamp": start_ts_ms,
                    "end_timestamp": end_ts_ms,
                    "resolution": 86400,  # daily
                },
            )
            if resp.status_code != 200:
                print(f"[deribit] {asset} dvol HTTP {resp.status_code}: {resp.text[:120]}")
                return pd.Series(dtype=float)
            payload = resp.json()
            data = (payload.get("result") or {}).get("data") or []
            if not data:
                print(f"[deribit] {asset} dvol empty response")
                return pd.Series(dtype=float)
            rows: dict[pd.Timestamp, float] = {}
            # data rows: [timestamp_ms, open, high, low, close]
            for r in data:
                t = pd.Timestamp(int(r[0]), unit="ms", tz="UTC").normalize()
                close = float(r[4])
                rows[t] = close / 100.0  # decimal
            return pd.Series(rows, name=f"deribit_dvol_{asset}").sort_index()
    except Exception as e:
        print(f"[deribit] {asset} dvol FAILED: {e}")
        return pd.Series(dtype=float)


def realized_sigma_rolling(close: pd.Series, window: int, ann_factor: float) -> pd.Series:
    """Rolling annualised σ from log returns. ``ann_factor`` is bars/year."""
    if close.empty:
        return pd.Series(dtype=float)
    logret = np.log(close / close.shift(1))
    return logret.rolling(window).std() * math.sqrt(ann_factor)


def benchmark_for_asset(asset: str, binance_closes: dict[str, pd.Series]) -> tuple[pd.Series, str]:
    """Return (daily σ_benchmark series, source-label) for the asset."""
    a = asset.upper()
    if a == "SPX":
        return fetch_fred_bench("VIXCLS"), "FRED VIXCLS"
    if a == "WTI":
        return fetch_fred_bench("OVXCLS"), "FRED OVXCLS"
    if a == "GOLD":
        return fetch_fred_bench("GVZCLS"), "FRED GVZCLS"
    if a in ("BTC", "ETH"):
        ddv = fetch_deribit_dvol_history(a)
        if not ddv.empty and len(ddv) >= 30:
            return ddv, f"Deribit DVOL {a}"
        # Fallback: 30d rolling realized σ from Binance daily closes.
        sym = "BTCUSDT" if a == "BTC" else "ETHUSDT"
        closes = binance_closes.get(sym)
        if closes is None or closes.empty:
            return pd.Series(dtype=float), f"Binance {sym} unavailable"
        rs = realized_sigma_rolling(closes, window=30, ann_factor=365.0)
        return rs.rename(f"binance_rv30_{a}"), f"Binance 30d RV ({sym}, √365) [Deribit fallback]"
    return pd.Series(dtype=float), "unknown"


# ---------------------------------------------------------------------------
# Step 4 — forward realized σ
# ---------------------------------------------------------------------------
def yf_close_series(ticker: str) -> pd.Series:
    """Daily yfinance close series. Empty on failure."""
    try:
        import yfinance as yf
    except Exception as e:
        print(f"[yfinance] import FAILED: {e}")
        return pd.Series(dtype=float)
    try:
        # Use the explicit Ticker history API for reliability with futures.
        t = yf.Ticker(ticker)
        df = t.history(
            start=(WINDOW_START - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(TODAY + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
    except Exception as e:
        print(f"[yfinance] {ticker} FAILED: {e}")
        return pd.Series(dtype=float)
    if df is None or df.empty or "Close" not in df.columns:
        print(f"[yfinance] {ticker}: empty")
        return pd.Series(dtype=float)
    close = df["Close"].copy()
    idx = close.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    close.index = idx.normalize()
    return close.rename(ticker)


def forward_realized_sigma(close: pd.Series, horizon: int, ann_factor: float) -> pd.Series:
    """For each date t, compute annualised σ over the next ``horizon`` days.

    Uses forward log returns r_{t+1..t+horizon} and a sample std with
    ddof=1. Annualises by sqrt(ann_factor) (252 stocks/commodities, 365 crypto).
    """
    if close.empty:
        return pd.Series(dtype=float)
    logret = np.log(close / close.shift(1)).dropna()
    out: dict[pd.Timestamp, float] = {}
    arr = logret.values
    idx = logret.index
    n = len(arr)
    for i in range(n - horizon):
        window = arr[i + 1 : i + 1 + horizon]
        if len(window) < horizon // 2:  # require ≥50% of bars
            continue
        sd = float(np.std(window, ddof=1))
        out[idx[i]] = sd * math.sqrt(ann_factor)
    return pd.Series(out, name=f"fwd_sigma_{horizon}d").sort_index()


def underlying_forward_sigma(asset: str, binance_closes: dict[str, pd.Series]) -> pd.Series:
    a = asset.upper()
    if a == "SPX":
        return forward_realized_sigma(yf_close_series("^GSPC"), horizon=30, ann_factor=252.0)
    if a == "WTI":
        return forward_realized_sigma(yf_close_series("CL=F"), horizon=30, ann_factor=252.0)
    if a == "GOLD":
        return forward_realized_sigma(yf_close_series("GC=F"), horizon=30, ann_factor=252.0)
    if a == "BTC":
        c = binance_closes.get("BTCUSDT")
        if c is None or c.empty:
            return pd.Series(dtype=float)
        return forward_realized_sigma(c, horizon=30, ann_factor=365.0)
    if a == "ETH":
        c = binance_closes.get("ETHUSDT")
        if c is None or c.empty:
            return pd.Series(dtype=float)
        return forward_realized_sigma(c, horizon=30, ann_factor=365.0)
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Step 5 — per-asset metrics
# ---------------------------------------------------------------------------
def asset_metrics(
    asset: str,
    sigma_pm: pd.Series,
    sigma_bench: pd.Series,
    sigma_fwd: pd.Series,
    bench_label: str,
) -> dict[str, Any]:
    """Compute the correlation + signaled-strategy metrics for one asset."""
    df = pd.concat(
        [
            sigma_pm.rename("pm"),
            sigma_bench.rename("bench"),
            sigma_fwd.rename("fwd"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    n = len(df)
    out: dict[str, Any] = {
        "asset": asset,
        "bench_label": bench_label,
        "n_aligned": n,
        "date_min": df.index.min().date().isoformat() if n else None,
        "date_max": df.index.max().date().isoformat() if n else None,
        "sigma_pm_mean": float(df["pm"].mean()) if n else None,
        "sigma_bench_mean": float(df["bench"].mean()) if n else None,
        "sigma_fwd_mean": float(df["fwd"].mean()) if n else None,
        "pm_minus_bench_mean": float((df["pm"] - df["bench"]).mean()) if n else None,
        "pm_minus_bench_std": float((df["pm"] - df["bench"]).std()) if n else None,
    }
    if n < 10:
        out["status"] = "insufficient_data"
        return out

    pm = df["pm"].values
    bench = df["bench"].values
    fwd = df["fwd"].values

    # Correlation σ_PM vs σ_fwd
    out["corr_pm_fwd"] = float(np.corrcoef(pm, fwd)[0, 1])

    # KEY NUMBER — gap-vs-residual correlation (raw, level-sensitive)
    gap = pm - bench
    fwd_resid = fwd - bench
    if np.std(gap) > 1e-9 and np.std(fwd_resid) > 1e-9:
        out["corr_gap_residual"] = float(np.corrcoef(gap, fwd_resid)[0, 1])
    else:
        out["corr_gap_residual"] = None

    # HONESTY CHECK — first-difference correlation. The level-correlation
    # can be inflated by shared regime drift (both σ_PM and σ_fwd trend
    # together over a 6-month window even without true predictive power).
    # If Δgap doesn't predict Δresid, the "alpha" is just co-trending.
    if len(gap) > 5:
        dgap = np.diff(gap)
        dresid = np.diff(fwd_resid)
        if np.std(dgap) > 1e-9 and np.std(dresid) > 1e-9:
            out["corr_dgap_dresid"] = float(np.corrcoef(dgap, dresid)[0, 1])
        else:
            out["corr_dgap_dresid"] = None
    else:
        out["corr_dgap_dresid"] = None

    # Cross-sectional demeaned correlation isn't applicable here (per-asset),
    # but report the gap demeaned (still ok for direction).
    gap_dm = gap - gap.mean()
    resid_dm = fwd_resid - fwd_resid.mean()
    if np.std(gap_dm) > 1e-9 and np.std(resid_dm) > 1e-9:
        out["corr_demeaned"] = float(np.corrcoef(gap_dm, resid_dm)[0, 1])
    else:
        out["corr_demeaned"] = None

    # Strategy simulator: ±2pp threshold on the gap.
    threshold = 0.02
    signals = np.zeros(n, dtype=int)
    signals[gap > threshold] = +1  # long-vol
    signals[gap < -threshold] = -1  # short-vol
    pnl = signals * (fwd - bench)
    sig_mask = signals != 0
    n_signals = int(sig_mask.sum())
    out["n_signals"] = n_signals
    out["pct_signaled"] = float(n_signals / n) if n else 0.0

    if n_signals < 5:
        out["status"] = "insufficient_signals"
        return out

    signaled_pnl = pnl[sig_mask]
    n_long = int((signals[sig_mask] == 1).sum())
    n_short = int((signals[sig_mask] == -1).sum())
    out["n_long_signals"] = n_long
    out["n_short_signals"] = n_short

    # Hit rate: fraction of signaled days where pnl > 0.
    hits = int((signaled_pnl > 0).sum())
    out["hit_rate"] = float(hits / n_signals)

    mean_pnl = float(np.mean(signaled_pnl))
    std_pnl = float(np.std(signaled_pnl, ddof=1)) if n_signals > 1 else 0.0
    out["mean_pnl_per_signal_vol_units"] = mean_pnl
    out["std_pnl_per_signal_vol_units"] = std_pnl

    # Sharpe across ALL aligned days (signal=0 contributes 0 PnL), annualised.
    # Use 252 daily steps/year as the equity-style convention; crypto would
    # use 365 but for cross-asset comparability we standardize on 252.
    daily_std_all = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
    daily_mean_all = float(np.mean(pnl))
    if daily_std_all > 1e-9:
        sharpe = (daily_mean_all / daily_std_all) * math.sqrt(252.0)
    else:
        sharpe = 0.0
    out["sharpe_annualised"] = sharpe

    # Drop-top-3 Sharpe — zero out the 3 largest |PnL| days and recompute.
    pnl_trimmed = pnl.copy()
    abs_pnl = np.abs(pnl_trimmed)
    top3 = np.argsort(abs_pnl)[-3:]
    pnl_trimmed[top3] = 0.0
    dt_std = float(np.std(pnl_trimmed, ddof=1)) if n > 1 else 0.0
    dt_mean = float(np.mean(pnl_trimmed))
    out["sharpe_drop_top3"] = (dt_mean / dt_std) * math.sqrt(252.0) if dt_std > 1e-9 else 0.0

    out["status"] = "ok"
    return out


def pooled_metrics(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pool the aligned daily (gap, fwd-residual) rows across assets."""
    rows = []
    for asset, rec in records.items():
        df = rec.get("_df")
        if df is None or df.empty:
            continue
        for ts, r in df.iterrows():
            rows.append(
                {"asset": asset, "date": ts, "pm": r["pm"], "bench": r["bench"], "fwd": r["fwd"]}
            )
    if not rows:
        return {"status": "no_data", "n": 0}
    df = pd.DataFrame(rows)
    n = len(df)
    gap = (df["pm"] - df["bench"]).values
    resid = (df["fwd"] - df["bench"]).values
    out: dict[str, Any] = {"n_rows_pooled": n, "n_assets": df["asset"].nunique()}
    if np.std(gap) > 1e-9 and np.std(resid) > 1e-9:
        out["corr_gap_residual_pooled"] = float(np.corrcoef(gap, resid)[0, 1])
        # Approximate 95% CI on Pearson r via Fisher z transform.
        r = out["corr_gap_residual_pooled"]
        if n > 4 and abs(r) < 0.999:
            z = 0.5 * math.log((1 + r) / (1 - r))
            se = 1.0 / math.sqrt(n - 3)
            zlo, zhi = z - 1.96 * se, z + 1.96 * se
            out["ci95_lo"] = float((math.exp(2 * zlo) - 1) / (math.exp(2 * zlo) + 1))
            out["ci95_hi"] = float((math.exp(2 * zhi) - 1) / (math.exp(2 * zhi) + 1))

    # HONESTY CHECK — within-asset demeaned correlation. Removes the
    # σ_PM level bias (which can dominate the raw Pearson r when extraction
    # over-states σ by a constant offset). If this is near zero while the
    # raw r is positive, the signal is just a static offset artifact.
    df_dm = df.copy()
    df_dm["gap"] = df_dm["pm"] - df_dm["bench"]
    df_dm["resid"] = df_dm["fwd"] - df_dm["bench"]
    df_dm["gap_dm"] = df_dm.groupby("asset")["gap"].transform(lambda x: x - x.mean())
    df_dm["resid_dm"] = df_dm.groupby("asset")["resid"].transform(lambda x: x - x.mean())
    g_dm = df_dm["gap_dm"].values
    r_dm = df_dm["resid_dm"].values
    if np.std(g_dm) > 1e-9 and np.std(r_dm) > 1e-9:
        out["corr_demeaned_within_asset"] = float(np.corrcoef(g_dm, r_dm)[0, 1])
        rdm = out["corr_demeaned_within_asset"]
        if n > 4 and abs(rdm) < 0.999:
            z = 0.5 * math.log((1 + rdm) / (1 - rdm))
            se = 1.0 / math.sqrt(n - 3)
            zlo, zhi = z - 1.96 * se, z + 1.96 * se
            out["ci95_lo_demeaned"] = float((math.exp(2 * zlo) - 1) / (math.exp(2 * zlo) + 1))
            out["ci95_hi_demeaned"] = float((math.exp(2 * zhi) - 1) / (math.exp(2 * zhi) + 1))

    # First-difference correlation pooled. Reset the per-asset diff so we
    # don't bridge assets.
    diff_rows = []
    for asset_name, sub in df_dm.groupby("asset"):
        sub = sub.sort_values("date")
        dgap = np.diff(sub["gap"].values)
        dresid = np.diff(sub["resid"].values)
        for g, r2 in zip(dgap, dresid, strict=True):
            diff_rows.append((g, r2))
    if diff_rows:
        dg = np.array([r[0] for r in diff_rows])
        dr = np.array([r[1] for r in diff_rows])
        if np.std(dg) > 1e-9 and np.std(dr) > 1e-9:
            out["corr_first_diff_pooled"] = float(np.corrcoef(dg, dr)[0, 1])
    threshold = 0.02
    signals = np.where(gap > threshold, 1, np.where(gap < -threshold, -1, 0))
    pnl = signals * resid
    n_signals = int(np.sum(signals != 0))
    out["n_signals_pooled"] = n_signals
    if n_signals > 0:
        out["hit_rate_pooled"] = float(np.mean(pnl[signals != 0] > 0))
        out["mean_pnl_pooled"] = float(np.mean(pnl[signals != 0]))
    daily_std_all = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
    if daily_std_all > 1e-9:
        out["sharpe_pooled"] = (float(np.mean(pnl)) / daily_std_all) * math.sqrt(252.0)
    return out


# ---------------------------------------------------------------------------
# Step 6 — verdict + report
# ---------------------------------------------------------------------------
def classify_verdict(per_asset: dict[str, dict[str, Any]], pooled: dict[str, Any]) -> str:
    """Apply CLAUDE.md tier criteria — with explicit guard against level-bias artifacts.

    Honest classification requires:
        * pooled raw r > 0 with CI95 > 0
        * pooled demeaned-within-asset r > 0 with CI95 > 0   ← guards against
          σ_PM level bias (extraction over-states σ by a near-constant offset)
        * pooled first-difference r > 0                       ← guards against
          shared regime drift over a single window

    If the raw r is positive but the demeaned + Δ checks both fail, the
    "signal" is an artifact of the extraction's level bias and should be
    SHELVED — there's no actual day-to-day predictive content.
    """
    n_pooled = pooled.get("n_rows_pooled", 0) or 0
    if n_pooled < 50:
        return "INSUFFICIENT DATA"
    ci_lo = pooled.get("ci95_lo")
    pooled_corr = pooled.get("corr_gap_residual_pooled")
    demeaned_corr = pooled.get("corr_demeaned_within_asset")
    demeaned_ci_lo = pooled.get("ci95_lo_demeaned")
    diff_corr = pooled.get("corr_first_diff_pooled")

    # Hard sanity gate — if level r is positive but demeaned r is ~0 or
    # negative, this is a level-bias artifact. Shelve regardless of nominal
    # CI95 or Sharpe.
    if pooled_corr is not None and pooled_corr > 0:
        if demeaned_corr is None or demeaned_corr <= 0.05:
            return "SHELVE (level-bias artifact)"
        if diff_corr is None or diff_corr <= 0.0:
            return "SHELVE (no day-over-day predictive content)"
        if demeaned_ci_lo is None or demeaned_ci_lo <= 0:
            return "C_TENTATIVE"
        # Now we have: demeaned r > ~0.05, Δ r > 0, CI95 above zero on demeaned.
        if ci_lo is not None and ci_lo > 0 and demeaned_ci_lo > 0:
            # Still cannot be A_GOLD without 4Q OOS — call it B_VALIDATED at best.
            return "B_VALIDATED"
        return "C_TENTATIVE"
    return "SHELVE"


def fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.{digits}f}%"


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def write_report(per_asset: dict[str, dict[str, Any]], pooled: dict[str, Any], verdict: str) -> str:
    lines: list[str] = []
    lines.append("# Validation A4 — σ_PM gap vs. forward realized σ")
    lines.append("")
    lines.append(f"Run date: {datetime.now(tz=UTC).isoformat(timespec='seconds')}  ")
    lines.append(
        f"Analysis window: {WINDOW_START.date().isoformat()} → {TODAY.date().isoformat()}  "
    )
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "Does the gap **σ_PM − σ_benchmark** predict **σ_fwd_realized − σ_benchmark**? "
        "If yes, the prediction-market ladder is adding information beyond options-derived IV."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "1. σ_PM extracted daily from `LADDER_REGISTRY` via `fit_implied_sigma` "
        "(lognormal moment-matching on a Breeden-Litzenberger-style PMF). "
        "Day eligible only when ≥3 ladder slugs have a quote (≤1 day forward-fill per slug)."
    )
    lines.append(
        "2. σ_benchmark per asset: VIXCLS / OVXCLS / GVZCLS from FRED for SPX / WTI / GOLD; "
        "Deribit historical DVOL for BTC / ETH (with Binance 30d realised σ as fallback)."
    )
    lines.append(
        "3. σ_fwd_realized: forward 30-day realised σ on the underlying — yfinance for "
        "SPX (`^GSPC`), WTI (`CL=F`), GOLD (`GC=F`); Binance daily for BTCUSDT / ETHUSDT. "
        "Annualisation: √252 equities/commodities, √365 crypto."
    )
    lines.append(
        "4. Strategy simulator: long-vol if `σ_PM > σ_bench + 2pp`, short-vol if "
        "`σ_PM < σ_bench − 2pp`, else flat. PnL ≈ sign · (σ_fwd − σ_bench), expressed "
        "in vol-points. Sharpe annualised by √252 across **all** aligned days "
        "(non-signaled days = 0 PnL)."
    )
    lines.append("")
    lines.append("## Per-asset results")
    lines.append("")
    cols = [
        "Asset",
        "Bench",
        "N days",
        "Range",
        "⟨σ_PM⟩",
        "⟨σ_bench⟩",
        "⟨gap⟩",
        "ρ(gap, fwd−bench)",
        "ρ(Δgap, Δresid)",
        "N sig",
        "Hit",
        "Sharpe",
        "Sharpe ex-top3",
        "Status",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for asset, r in per_asset.items():
        rng = f"{r.get('date_min') or '—'}..{r.get('date_max') or '—'}"
        row = [
            asset,
            r.get("bench_label") or "—",
            str(r.get("n_aligned") or 0),
            rng,
            fmt_pct(r.get("sigma_pm_mean")),
            fmt_pct(r.get("sigma_bench_mean")),
            fmt_pct(r.get("pm_minus_bench_mean")),
            fmt(r.get("corr_gap_residual")),
            fmt(r.get("corr_dgap_dresid")),
            str(r.get("n_signals") or 0),
            fmt_pct(r.get("hit_rate")),
            fmt(r.get("sharpe_annualised")),
            fmt(r.get("sharpe_drop_top3")),
            r.get("status") or "n/a",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "> *Note:* per-asset Pearson r is translation-invariant, so a "
        'single-asset "demeaned" correlation is identical to the raw '
        "value. The within-asset demeaned check is only informative in "
        "the pooled sample, reported in the next section."
    )
    lines.append("")
    lines.append("## Pooled results")
    lines.append("")
    if pooled.get("n_rows_pooled"):
        lines.append(
            f"- Rows pooled: **{pooled.get('n_rows_pooled')}** across "
            f"{pooled.get('n_assets')} assets"
        )
        lines.append(
            f"- Pooled raw ρ(gap, fwd−bench): **{fmt(pooled.get('corr_gap_residual_pooled'))}** "
            f"(95% CI Fisher-z: "
            f"[{fmt(pooled.get('ci95_lo'))}, {fmt(pooled.get('ci95_hi'))}])"
        )
        lines.append(
            f"- **Demeaned-within-asset ρ**: **{fmt(pooled.get('corr_demeaned_within_asset'))}** "
            f"(95% CI: [{fmt(pooled.get('ci95_lo_demeaned'))}, "
            f"{fmt(pooled.get('ci95_hi_demeaned'))}]) — "
            f"removes the σ_PM level-bias offset. **This is the honest number.**"
        )
        lines.append(
            f"- **First-difference ρ(Δgap, Δresid)**: **{fmt(pooled.get('corr_first_diff_pooled'))}** — "
            f"does day-to-day *change* in the gap predict day-to-day change in the "
            f"forward-residual? If this is ≈0 the raw r is co-trending, not predictive."
        )
        lines.append(
            f"- Signals: {pooled.get('n_signals_pooled')} "
            f"(hit rate {fmt_pct(pooled.get('hit_rate_pooled'))}, "
            f"mean PnL {fmt_pct(pooled.get('mean_pnl_pooled'))})"
        )
        lines.append(
            f"- Pooled Sharpe (√252): **{fmt(pooled.get('sharpe_pooled'))}** "
            f"— **caveat:** this Sharpe is inflated by the level bias because "
            f"the gap exceeds the +2pp threshold on essentially every day for "
            f"WTI/GOLD, so the strategy is effectively a constant long-vol position "
            f"during a regime where σ_realized > σ_bench. The Sharpe is not a fair "
            f"estimate of out-of-regime performance."
        )
    else:
        lines.append("- No pooled rows.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **σ_PM extraction bias.** The lognormal-fit pipeline tends to *over*-state σ "
        "when the ladder is wide and the upper-tail mass is non-trivial — empirical "
        "moments inflate the implied std because the missing right-tail beyond the "
        "highest strike is treated as point mass. The **gap direction** can still be "
        "informative even if the absolute level is biased high, which is what this "
        "validation tests."
    )
    lines.append(
        "- **Calendar mismatch.** PM ladders are point-in-time risk-neutral views on a "
        "specific maturity (EoY-2026 for SPX/BTC/ETH, June-2026 for WTI/GOLD). The "
        "benchmark (VIX/OVX/GVZ/DVOL) is 30-day. We're comparing different tenors — "
        "this is a known apples-to-oranges issue and will shrink the realisable signal."
    )
    lines.append(
        "- **Single-window risk.** Only one ~6-month window of data is available. "
        "CLAUDE.md mandates a **4-quarter robustness check** before any A_GOLD claim; "
        "this validation cannot satisfy that on its own."
    )
    lines.append(
        "- **Survivorship in ladder coverage.** SPX/BTC `above` ladders are uncovered "
        "in the strat7 cache and require live fetches; if Polymarket has resolved or "
        "delisted any slug, the daily σ_PM for that asset will simply be missing for "
        "those dates."
    )
    lines.append(
        "- **Forward 30d realised σ alignment.** For dates within ~30 days of TODAY we "
        "have less than a full forward window; those rows are dropped automatically."
    )
    lines.append(
        "- **Bench fallback for BTC/ETH.** If Deribit's historical DVOL endpoint is "
        "unreachable, we fall back to Binance 30d realised σ — which is a poor proxy "
        "for forward IV (it is a backward-looking estimator). This pollutes the gap "
        "computation for those assets; flagged in the per-asset table by the `Bench` "
        "column."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    explain = {
        "B_VALIDATED": (
            "Both the level-r AND the within-asset demeaned r clear zero with "
            "CI95 > 0, AND the Δ-Δ correlation is positive. The σ_PM gap **does** "
            "carry day-over-day information about the forward residual, beyond a "
            "static offset. Wire as a **research-only telemetry** signal — still "
            "needs 4-quarter OOS before sizing."
        ),
        "C_TENTATIVE": (
            "Demeaned r is mildly positive but CI95 includes zero, or Δ-Δ is "
            "borderline. Suggestive but not decision-useful on a 6-month window."
        ),
        "SHELVE (level-bias artifact)": (
            "The raw correlation is positive **only because** σ_PM is systematically biased "
            "high (the lognormal-fit pipeline over-states σ by a near-constant offset for "
            "the wide WTI/GOLD ladders). Once we demean within asset — which removes that "
            "constant offset — the correlation collapses to near zero. The high Sharpe is "
            "spurious: with σ_PM permanently > σ_bench + 2pp, the strategy is effectively a "
            "constant long-vol position in a window where realised σ also happened to exceed "
            "implied. **Do not wire this into the UI as alpha.** The σ_PM display is still "
            "useful as a context number (tail-mass weight at extreme strikes), but the gap "
            "vs. options-IV is not a tradeable signal in its current form. Fix the extraction "
            "bias before re-validating."
        ),
        "SHELVE (no day-over-day predictive content)": (
            "Demeaned correlation is positive but the first-difference correlation is "
            "≤ 0 — Δgap does not predict Δresid. The level correlation is co-trending "
            "(both σ_PM and σ_fwd drift upward together over the window) without any "
            "actual lead-lag content. Shelve."
        ),
        "SHELVE": (
            "Pooled correlation is zero or negative — the σ_PM gap does not predict "
            "the forward realised-vol residual in this window. Shelve."
        ),
        "INSUFFICIENT DATA": (
            "Fewer than 50 pooled (asset × date) rows after all alignment. "
            "PM ladder coverage is too sparse (SPX/ETH `above` slugs do "
            "not exist on Polymarket; BTC `above` slugs also missing) to "
            "draw a robust conclusion. Re-run when ladder coverage expands."
        ),
    }.get(verdict, "")
    if explain:
        lines.append(explain)
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /Users/damiangallardoloya/Desktop/proyectofuentes")
    lines.append("PYTHONPATH=api/src api/.venv/bin/python api/scripts/validate_pm_iv_gap.py")
    lines.append("```")
    lines.append("")
    REPORT.write_text("\n".join(lines))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("Validation A4 — σ_PM gap predicts forward realised σ?")
    print(f"window: {WINDOW_START.date()} → {TODAY.date()}")
    print("=" * 72)

    # Collect all slugs
    all_slugs: list[str] = []
    for asset, cfg in LADDER_REGISTRY.items():
        for fam in cfg["families"]:
            for slug, _, _, _ in fam["entries"]:
                all_slugs.append(slug)
    # de-dup but preserve order
    seen: set[str] = set()
    unique_slugs = [s for s in all_slugs if not (s in seen or seen.add(s))]
    print(f"total ladder slugs: {len(unique_slugs)}")

    # Step 1 — load history
    use_live = os.environ.get("PM_IV_VAL_NO_LIVE", "") != "1"
    history = load_slug_history(unique_slugs, use_live=use_live, live_budget_s=180.0)
    print(f"history loaded: {len(history)}/{len(unique_slugs)} slugs")

    # Step 2 — σ_PM per asset
    sigma_pm_by_asset: dict[str, pd.Series] = {}
    for asset in LADDER_REGISTRY:
        s = build_sigma_pm_series(asset, history)
        sigma_pm_by_asset[asset] = s
        print(
            f"σ_PM {asset}: {len(s)} daily fits, "
            f"range={s.index.min().date() if len(s) else '—'}.."
            f"{s.index.max().date() if len(s) else '—'}, "
            f"mean={s.mean() * 100 if len(s) else float('nan'):.1f}%"
        )

    # Step 3 — Binance daily closes (used by both σ_bench fallback and forward σ)
    binance_closes: dict[str, pd.Series] = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        binance_closes[sym] = fetch_binance_daily_closes(sym)
        print(f"binance {sym} closes: n={len(binance_closes[sym])}")

    sigma_bench_by_asset: dict[str, pd.Series] = {}
    bench_labels: dict[str, str] = {}
    for asset in LADDER_REGISTRY:
        s, label = benchmark_for_asset(asset, binance_closes)
        sigma_bench_by_asset[asset] = s
        bench_labels[asset] = label
        print(
            f"σ_bench {asset}: {label} n={len(s)}, mean={s.mean() * 100 if len(s) else float('nan'):.1f}%"
        )

    # Step 4 — forward realised σ
    sigma_fwd_by_asset: dict[str, pd.Series] = {}
    for asset in LADDER_REGISTRY:
        fwd = underlying_forward_sigma(asset, binance_closes)
        sigma_fwd_by_asset[asset] = fwd
        print(
            f"σ_fwd {asset}: n={len(fwd)}, "
            f"mean={fwd.mean() * 100 if len(fwd) else float('nan'):.1f}%"
        )

    # Step 5 — per-asset metrics
    per_asset: dict[str, dict[str, Any]] = {}
    for asset in LADDER_REGISTRY:
        rec = asset_metrics(
            asset,
            sigma_pm_by_asset[asset],
            sigma_bench_by_asset[asset],
            sigma_fwd_by_asset[asset],
            bench_labels[asset],
        )
        # stash the aligned DataFrame for pooling
        df = pd.concat(
            [
                sigma_pm_by_asset[asset].rename("pm"),
                sigma_bench_by_asset[asset].rename("bench"),
                sigma_fwd_by_asset[asset].rename("fwd"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        rec["_df"] = df
        per_asset[asset] = rec
        print(f"\n--- {asset} ({rec.get('bench_label')}) ---")
        for k, v in rec.items():
            if k == "_df":
                continue
            print(f"   {k}: {v}")

    # Step 6 — pooled metrics + verdict + report
    pooled = pooled_metrics(per_asset)
    print("\n--- pooled ---")
    for k, v in pooled.items():
        print(f"   {k}: {v}")

    verdict = classify_verdict(per_asset, pooled)
    # Strip the DataFrame before writing
    for r in per_asset.values():
        r.pop("_df", None)
    write_report(per_asset, pooled, verdict)

    # One-page stdout summary
    print("\n" + "=" * 72)
    print(f"VERDICT: {verdict}")
    if pooled.get("n_rows_pooled"):
        print(
            f"  pooled ρ(gap, fwd−bench) = {pooled.get('corr_gap_residual_pooled')} "
            f"CI95=[{pooled.get('ci95_lo')}, {pooled.get('ci95_hi')}] "
            f"n={pooled.get('n_rows_pooled')}"
        )
        print(f"  pooled Sharpe = {pooled.get('sharpe_pooled')}")
    else:
        print("  no pooled data")
    print(f"  report: {REPORT}")
    print(f"  cache: {PM_CACHE}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[interrupt] aborting")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
