"""Closed-form spot-vs-market-implied probability machinery.

Compares a *live underlying* (e.g. BTC daily klines) to a Polymarket binary
outcome of the form "underlying ≥ K" or "underlying touches K". Returns a
model probability with a bootstrap CI and a clearly-labelled "edge" against
the market price.

Quant-rigour notes (the writeup in ``docs/strategies.md`` §6 has the full
discussion):

*   Vol via **Yang-Zhang OHLC** (Yang & Zhang 2000). Closed form, drift-
    independent, ~5× more efficient than close-to-close. Falls back to
    close-to-close when only closes are provided.
*   Two market geometries (caller's choice):
    - ``terminal``: P(S_T ≥ K) under risk-neutral GBM with drift μ.
    - ``one_touch_up`` / ``one_touch_down``: first-passage barrier
      probability under GBM via the reflection principle.
*   Drift μ defaults to **0** (risk-neutral, no carry on unfunded BTC spot).
    Caller can override (e.g. with the perp funding rate).
*   Bootstrap CI on the model probability uses **block bootstrap** to
    preserve daily-vol clustering (Kunsch 1989, Politis-Romano 1994).

This module deliberately does *not* fetch any data — caller passes a
DataFrame with OHLC. Fetchers live in :mod:`pfm.sources.binance` etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from math import log, sqrt
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

# ───────────────────────── data classes ───────────────────────────────


Geometry = Literal["terminal", "one_touch_up", "one_touch_down"]


@dataclass(frozen=True)
class VolEstimate:
    """Annualised σ̂ from a window of OHLC bars.

    Attributes:
        sigma_annual: annualised σ̂ (units: 1/√year).
        n_bars: number of bars used.
        method: ``"yang_zhang"`` or ``"close_to_close"``.
    """

    sigma_annual: float
    n_bars: int
    method: str


@dataclass(frozen=True)
class SpotVsImpliedResult:
    """Output of :func:`spot_vs_implied`.

    Attributes:
        spot: latest underlying price ``S_0``.
        strike: target ``K``.
        time_years: ``T`` in fractional years (act/365).
        geometry: ``terminal`` / ``one_touch_up`` / ``one_touch_down``.
        sigma_used: annualised σ̂ used (point estimate).
        drift_used: annualised μ used.
        model_prob: model probability of the outcome (point).
        ci_lo_90: 5th percentile of bootstrap distribution (90% CI lower).
        ci_hi_90: 95th percentile of bootstrap distribution.
        ci_lo_95: 2.5th percentile.
        ci_hi_95: 97.5th percentile.
        market_prob: market YES-price (caller-supplied).
        edge: ``market_prob - model_prob``. Positive = market sees more
            probability than model; negative = market under-weights.
        edge_significant_95: ``True`` if ``market_prob`` lies outside the
            95% bootstrap CI of ``model_prob``.
        n_bootstrap: number of bootstrap iterations completed.
    """

    spot: float
    strike: float
    time_years: float
    geometry: str
    sigma_used: float
    drift_used: float
    model_prob: float
    ci_lo_90: float
    ci_hi_90: float
    ci_lo_95: float
    ci_hi_95: float
    market_prob: float | None
    edge: float | None
    edge_significant_95: bool | None
    n_bootstrap: int


# ────────────────────── volatility estimators ─────────────────────────


def yang_zhang_volatility(
    ohlc: pd.DataFrame,
    *,
    annualisation: float = 365.0,
) -> VolEstimate:
    """Yang-Zhang OHLC volatility estimator (Yang & Zhang 2000).

    σ²_YZ = σ²_overnight + k · σ²_open_to_close + (1−k) · σ²_RS

    where σ²_RS is the Rogers-Satchell estimator and k minimises variance
    of the combined estimator (depends on ``n``):

        k = 0.34 / (1.34 + (n+1)/(n−1))

    Inputs:
        ohlc: DataFrame indexed by date with columns ``open, high, low, close``.
        annualisation: number of bars per year. For 24/7 crypto = 365; for
            equities use 252.

    Returns:
        :class:`VolEstimate` with annualised σ̂.

    Raises:
        ValueError: if ``ohlc`` has fewer than 3 rows or missing OHLC cols.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(ohlc.columns.str.lower()) - set(ohlc.columns)
    if missing:
        raise ValueError(f"yang_zhang_volatility: missing OHLC cols: {sorted(missing)}")
    df = ohlc.copy()
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

    if len(df) < 3:
        raise ValueError(f"yang_zhang_volatility: need ≥3 bars, got {len(df)}")

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    # Overnight: log(O_t / C_{t-1}). First entry is undefined.
    log_overnight = np.log(o[1:] / c[:-1])
    var_o = float(np.var(log_overnight, ddof=1))

    # Open-to-close.
    log_oc = np.log(c / o)
    var_oc = float(np.var(log_oc, ddof=1))

    # Rogers-Satchell (drift-independent intraday vol).
    rs = np.log(h / c) * np.log(h / o) + np.log(lo / c) * np.log(lo / o)
    var_rs = float(np.mean(rs))

    n = len(df)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    var_yz = var_o + k * var_oc + (1.0 - k) * var_rs
    if var_yz < 0.0:
        # Degenerate case (all bars flat) → fall back to close-to-close.
        var_yz = float(np.var(np.diff(np.log(c)), ddof=1))

    sigma_per_bar = sqrt(max(var_yz, 0.0))
    sigma_annual = sigma_per_bar * sqrt(annualisation)
    return VolEstimate(sigma_annual=sigma_annual, n_bars=n, method="yang_zhang")


def close_to_close_volatility(closes: pd.Series, *, annualisation: float = 365.0) -> VolEstimate:
    """Plain close-to-close σ̂. Use when OHLC isn't available."""
    if len(closes) < 3:
        raise ValueError(f"close_to_close_volatility: need ≥3 bars, got {len(closes)}")
    log_ret = np.diff(np.log(closes.to_numpy(dtype=float)))
    sigma_per_bar = float(np.std(log_ret, ddof=1))
    return VolEstimate(
        sigma_annual=sigma_per_bar * sqrt(annualisation),
        n_bars=len(closes),
        method="close_to_close",
    )


# ─────────────────────── GBM probabilities ────────────────────────────


def gbm_terminal_above(
    spot: float, strike: float, sigma_annual: float, drift_annual: float, t_years: float
) -> float:
    """P(S_T ≥ K) under risk-neutral GBM.

    Closed form (lognormal endpoint):
        P = Φ((ln(S₀/K) + (μ − σ²/2)·T) / (σ·√T))
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if t_years <= 0:
        return 1.0 if spot >= strike else 0.0
    if sigma_annual <= 0:
        # Degenerate: deterministic drift only.
        return 1.0 if spot * np.exp(drift_annual * t_years) >= strike else 0.0
    d = (log(spot / strike) + (drift_annual - 0.5 * sigma_annual**2) * t_years) / (
        sigma_annual * sqrt(t_years)
    )
    return float(norm.cdf(d))


def gbm_one_touch_up(
    spot: float, barrier: float, sigma_annual: float, drift_annual: float, t_years: float
) -> float:
    """P(max_{0≤t≤T} S_t ≥ H) under GBM, S_0 < H.

    Setting X_t = ln(S_t/S_0), the BM drift is μ_X = μ − σ²/2 and b = ln(H/S₀).
    By the reflection principle for first-passage of BM with drift to b > 0
    (Karatzas-Shreve §3.7):

        P(τ_b ≤ T) = Φ((μ_X·T − b)/(σ·√T)) + exp(2·μ_X·b/σ²) · Φ((−b − μ_X·T)/(σ·√T))

    If S₀ ≥ H the event has already occurred → return 1.
    """
    if spot <= 0 or barrier <= 0:
        raise ValueError("spot and barrier must be positive")
    if spot >= barrier:
        return 1.0
    if t_years <= 0:
        return 0.0
    if sigma_annual <= 0:
        return 1.0 if spot * np.exp(drift_annual * t_years) >= barrier else 0.0

    s_sqrt_t = sigma_annual * sqrt(t_years)
    mu_x = drift_annual - 0.5 * sigma_annual**2
    drift_term = mu_x * t_years
    b = log(barrier / spot)  # > 0 since barrier > spot
    d1 = (drift_term - b) / s_sqrt_t  # (μ_X·T − b)/(σ√T)
    d2 = -(b + drift_term) / s_sqrt_t  # (−b − μ_X·T)/(σ√T)
    refl = np.exp(2.0 * mu_x * b / (sigma_annual**2))
    p = norm.cdf(d1) + refl * norm.cdf(d2)
    return float(min(max(p, 0.0), 1.0))


def gbm_one_touch_down(
    spot: float, barrier: float, sigma_annual: float, drift_annual: float, t_years: float
) -> float:
    """P(min_{0≤t≤T} S_t ≤ L) under GBM, L < S₀.

    Symmetric to :func:`gbm_one_touch_up`. With X_t = ln(S_t/S_0),
    μ_X = μ − σ²/2 and b' = ln(L/S₀) < 0:

        P(τ_{b'} ≤ T) = Φ((b' − μ_X·T)/(σ·√T)) + exp(2·μ_X·b'/σ²) · Φ((b' + μ_X·T)/(σ·√T))
    """
    if spot <= 0 or barrier <= 0:
        raise ValueError("spot and barrier must be positive")
    if spot <= barrier:
        return 1.0
    if t_years <= 0:
        return 0.0
    if sigma_annual <= 0:
        return 1.0 if spot * np.exp(drift_annual * t_years) <= barrier else 0.0

    s_sqrt_t = sigma_annual * sqrt(t_years)
    mu_x = drift_annual - 0.5 * sigma_annual**2
    drift_term = mu_x * t_years
    bp = log(barrier / spot)  # < 0 since barrier < spot
    d1 = (bp - drift_term) / s_sqrt_t
    d2 = (bp + drift_term) / s_sqrt_t
    refl = np.exp(2.0 * mu_x * bp / (sigma_annual**2))
    p = norm.cdf(d1) + refl * norm.cdf(d2)
    return float(min(max(p, 0.0), 1.0))


# ────────────────────────── bootstrap CI ──────────────────────────────


def block_bootstrap_log_returns(
    log_returns: np.ndarray,
    *,
    block_size: int,
    n_iters: int,
    seed: int = 42,
) -> np.ndarray:
    """Stationary block bootstrap of a log-return series.

    Returns an ``(n_iters, len(log_returns))`` array of resampled series.
    Block size should be ≈ √n for vol-clustered data (rule of thumb).
    """
    rng = np.random.default_rng(seed)
    n = len(log_returns)
    out = np.empty((n_iters, n), dtype=float)
    for i in range(n_iters):
        # Concatenate consecutive blocks until we cover n bars.
        bars: list[float] = []
        while len(bars) < n:
            start = int(rng.integers(0, n))
            blk_n = min(block_size, n - len(bars))
            # Wrap-around: take from start to start+blk_n with mod n.
            idx = (np.arange(blk_n) + start) % n
            bars.extend(log_returns[idx].tolist())
        out[i, :] = bars[:n]
    return out


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return (5%, 95%, 2.5%, 97.5%) percentiles."""
    return (
        float(np.percentile(values, 5)),
        float(np.percentile(values, 95)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


# ───────────────────── top-level orchestrator ─────────────────────────


def _prob_for_geometry(
    geometry: Geometry,
    spot: float,
    strike: float,
    sigma: float,
    drift: float,
    t: float,
) -> float:
    if geometry == "terminal":
        return gbm_terminal_above(spot, strike, sigma, drift, t)
    if geometry == "one_touch_up":
        return gbm_one_touch_up(spot, strike, sigma, drift, t)
    if geometry == "one_touch_down":
        return gbm_one_touch_down(spot, strike, sigma, drift, t)
    raise ValueError(f"unknown geometry: {geometry!r}")


def spot_vs_implied(
    ohlc: pd.DataFrame,
    *,
    strike: float,
    expiry: _date,
    geometry: Geometry,
    market_prob: float | None = None,
    drift_annual: float = 0.0,
    annualisation: float = 365.0,
    n_bootstrap: int = 200,
    block_size: int | None = None,
    seed: int = 42,
    asof: _date | None = None,
    asof_ts: pd.Timestamp | None = None,
) -> SpotVsImpliedResult:
    """Compute model probability + bootstrap CI for a price-target market.

    Args:
        ohlc: DataFrame indexed by date with ``open, high, low, close``.
            Closes are used for spot/last-bar; OHLC for vol estimation.
        strike: target ``K`` (or barrier H/L for touch markets).
        expiry: market resolution date (UTC).
        geometry: ``"terminal"`` / ``"one_touch_up"`` / ``"one_touch_down"``.
        market_prob: market YES-price *today* (optional). When provided,
            ``edge`` and ``edge_significant_95`` are populated.
        drift_annual: μ. Default 0 (risk-neutral, no carry).
        annualisation: bars per year (365 for 24/7 crypto, 252 for equities).
        n_bootstrap: number of block-bootstrap iterations.
        block_size: stationary block-bootstrap block length. Default ≈ √n.
        seed: RNG seed.
        asof: pretend "today" is this date (defaults to the last bar's date).

    Returns:
        :class:`SpotVsImpliedResult`.
    """
    if len(ohlc) < 5:
        raise ValueError(f"spot_vs_implied: need ≥5 bars, got {len(ohlc)}")
    df = ohlc.sort_index()
    last_idx = df.index[-1]
    spot = float(df["close"].iloc[-1])

    # ``annualisation`` is bars-per-year. For daily data (365) the unit is
    # *calendar days*; for sub-daily (e.g. 5m → 105120) the unit is *bars*.
    # Prefer the precise sub-daily T calculation if the caller supplied
    # ``asof_ts`` (a UTC Timestamp); fall back to integer-day arithmetic
    # otherwise so daily users get the legacy behaviour.
    if asof_ts is not None or annualisation > 1000:
        # Sub-daily code path: compute T in bars to match the annualisation.
        ts_now = asof_ts if asof_ts is not None else last_idx
        ts_exp = pd.Timestamp(expiry, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
        if ts_exp < ts_now:
            raise ValueError(f"expiry {expiry} is before asof {ts_now}")
        seconds_per_year = 365.25 * 24 * 3600
        t_years = (ts_exp - ts_now).total_seconds() / seconds_per_year
    else:
        asof = asof or (last_idx.date() if hasattr(last_idx, "date") else last_idx)
        days_to_expiry = (expiry - asof).days
        if days_to_expiry < 0:
            raise ValueError(f"expiry {expiry} is before asof {asof}")
        t_years = days_to_expiry / annualisation

    # Vol estimate (point).
    vol = yang_zhang_volatility(df, annualisation=annualisation)
    sigma = vol.sigma_annual

    # Point model probability.
    p_model = _prob_for_geometry(geometry, spot, strike, sigma, drift_annual, t_years)

    # Bootstrap CI on the model probability — resample log-close-returns
    # and reconstruct an OHLC series for each iteration. To keep this fast
    # and avoid having to bootstrap the whole OHLC tensor, we approximate by
    # bootstrapping σ̂ via close-to-close on the resampled returns and
    # reusing the YZ point estimate's structural multiplier
    #   m = sigma_yz / sigma_cc.
    # This preserves the YZ efficiency ratio while allowing fast vol CI.
    closes = df["close"].to_numpy(dtype=float)
    log_ret = np.diff(np.log(closes))
    n_lr = len(log_ret)
    if n_lr < 5:
        raise ValueError(f"need ≥5 daily log-returns for bootstrap, got {n_lr}")

    cc_vol = close_to_close_volatility(df["close"], annualisation=annualisation)
    structural = sigma / cc_vol.sigma_annual if cc_vol.sigma_annual > 0 else 1.0

    bs_block = block_size or max(2, int(round(np.sqrt(n_lr))))
    boots = block_bootstrap_log_returns(
        log_ret,
        block_size=bs_block,
        n_iters=n_bootstrap,
        seed=seed,
    )
    # Per-iter σ̂ (close-to-close annualised), scaled by the structural ratio.
    sigma_iters = np.std(boots, axis=1, ddof=1) * np.sqrt(annualisation) * structural
    p_iters = np.array(
        [_prob_for_geometry(geometry, spot, strike, s, drift_annual, t_years) for s in sigma_iters],
        dtype=float,
    )
    ci_lo_90, ci_hi_90, ci_lo_95, ci_hi_95 = _bootstrap_ci(p_iters)

    edge = (market_prob - p_model) if market_prob is not None else None
    edge_sig = not (ci_lo_95 <= market_prob <= ci_hi_95) if market_prob is not None else None

    return SpotVsImpliedResult(
        spot=spot,
        strike=float(strike),
        time_years=float(t_years),
        geometry=geometry,
        sigma_used=float(sigma),
        drift_used=float(drift_annual),
        model_prob=float(p_model),
        ci_lo_90=ci_lo_90,
        ci_hi_90=ci_hi_90,
        ci_lo_95=ci_lo_95,
        ci_hi_95=ci_hi_95,
        market_prob=market_prob,
        edge=edge,
        edge_significant_95=edge_sig,
        n_bootstrap=int(boots.shape[0]),
    )


__all__ = [
    "Geometry",
    "SpotVsImpliedResult",
    "VolEstimate",
    "block_bootstrap_log_returns",
    "close_to_close_volatility",
    "gbm_one_touch_down",
    "gbm_one_touch_up",
    "gbm_terminal_above",
    "spot_vs_implied",
    "yang_zhang_volatility",
]
