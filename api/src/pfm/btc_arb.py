"""Fair-price overlay for the Polymarket BTC up/down latency-arb.

Polymarket's "BTC Up or Down" 5m / 15m markets resolve "Up" if the Chainlink
BTC/USD reference at the window's end is >= the reference at the window's
start. Under a Geometric Brownian Motion (GBM) model for BTC spot, the
fair Up-probability conditional on the *current* spot, the start spot,
the remaining time, and an annualized volatility is::

    P(BTC_T >= BTC_0 | BTC_t, t, T)
        = Phi( ( ln(BTC_t / BTC_0) + (mu - sigma^2/2) * (T - t) )
               / ( sigma * sqrt(T - t) ) )

with mu ~= 0 over short (sub-15-minute) windows. This module provides:

1. `compute_fair_up_prob` — the closed-form GBM Up-probability.
2. `arb_signal` — turns (poly_up_mid, fair_up) into BUY_UP / SELL_UP / HOLD.
3. `realized_volatility` — close-to-close annualized realized vol from a
   list of evenly-spaced prices.

All three are pure functions and have no I/O — they're intended to be unit
tested with synthetic data and then wired into the live monitor in
`api/scripts/btc_arb_live.py`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

# A year of "trading seconds" for crypto = 365 * 24 * 3600 (24/7 market).
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_fair_up_prob(
    btc_t: float,
    btc_0: float,
    seconds_remaining: float,
    vol_ann: float = 0.65,
    mu_ann: float = 0.0,
) -> float:
    """Closed-form GBM probability that BTC_T >= BTC_0.

    Args:
        btc_t: Current BTC spot (e.g. Binance bookTicker midpoint).
        btc_0: BTC spot at the window start (the resolution reference).
        seconds_remaining: Seconds left until window end (T - t). May be
            zero, in which case the probability collapses to 1{btc_t >= btc_0}.
        vol_ann: Annualized BTC volatility (sigma). Default 0.65 ~= 65%.
        mu_ann: Annualized drift. Default 0.0 — over <=15 min windows the
            drift contribution is negligible vs. diffusion.

    Returns:
        Probability in [0, 1] that the window resolves Up.

    Raises:
        ValueError: if any price is non-positive or vol is negative.
    """
    if btc_t <= 0 or btc_0 <= 0:
        raise ValueError("prices must be positive")
    if vol_ann < 0:
        raise ValueError("vol_ann must be non-negative")
    if seconds_remaining < 0:
        raise ValueError("seconds_remaining must be non-negative")

    log_ratio = math.log(btc_t / btc_0)

    # Edge case: at expiry the answer is deterministic.
    # (We treat exactly-equal as Up since Polymarket's convention is >=.)
    if seconds_remaining == 0.0 or vol_ann == 0.0:
        return 1.0 if log_ratio >= 0.0 else 0.0

    tau_years = seconds_remaining / SECONDS_PER_YEAR
    drift = (mu_ann - 0.5 * vol_ann * vol_ann) * tau_years
    denom = vol_ann * math.sqrt(tau_years)
    z = (log_ratio + drift) / denom
    return _norm_cdf(z)


def arb_signal(
    poly_up_mid: float,
    fair_up: float,
    edge_threshold: float = 0.03,
) -> str:
    """Convert a fair-vs-market gap into a discrete trade signal.

    If Polymarket's Up midpoint is materially below fair, the Up token is
    cheap -> BUY_UP. If it's materially above fair, Up is rich -> SELL_UP.
    Otherwise HOLD.

    Args:
        poly_up_mid: Polymarket CLOB midpoint of the Up token, in [0, 1].
        fair_up: Model-implied fair Up-probability, in [0, 1].
        edge_threshold: Absolute probability gap required to trade.
            Default 0.03 = 3 percentage points.

    Returns:
        "BUY_UP", "SELL_UP", or "HOLD".
    """
    if edge_threshold < 0:
        raise ValueError("edge_threshold must be non-negative")
    edge = fair_up - poly_up_mid
    if edge >= edge_threshold:
        return "BUY_UP"
    if edge <= -edge_threshold:
        return "SELL_UP"
    return "HOLD"


def realized_volatility(prices: Iterable[float], dt_seconds: float) -> float:
    """Annualized close-to-close realized volatility of a price series.

    Uses log returns r_i = ln(P_i / P_{i-1}), takes their sample std
    (ddof=1), and scales by sqrt(SECONDS_PER_YEAR / dt_seconds).

    Args:
        prices: Evenly-spaced prices (e.g. one Binance mid per second).
        dt_seconds: Spacing between consecutive prices, in seconds.

    Returns:
        Annualized vol (e.g. 0.65 for 65%/yr). Returns 0.0 when fewer
        than 2 returns are available.

    Raises:
        ValueError: if dt_seconds is non-positive or any price <= 0.
    """
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    px = list(prices)
    if len(px) < 3:
        return 0.0
    rets: list[float] = []
    for i in range(1, len(px)):
        if px[i] <= 0 or px[i - 1] <= 0:
            raise ValueError("prices must be positive")
        rets.append(math.log(px[i] / px[i - 1]))
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    sd = math.sqrt(var)
    return sd * math.sqrt(SECONDS_PER_YEAR / dt_seconds)
