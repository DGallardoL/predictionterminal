"""Brownian-bridge probability fan-chart for Polymarket binary markets.

Given a market that resolves at some future date T to either YES (1) or NO (0),
this module simulates plausible *paths* of the YES probability between today
and T, and returns five percentile paths (p10, p25, p50, p75, p90) suitable
for plotting a fan chart in the Terminal UI.

THEORY
------
We model the *logit* of the YES probability,

    L_t = log(p_t / (1 - p_t)),

as a continuous-time process. The market must resolve to a binary outcome at
time T, so conditional on the eventual outcome the path of L_s for t <= s < T
is a Brownian bridge whose endpoint is +inf (YES wins) or -inf (NO wins).
We approximate "+/- inf" with large finite logit caps (+/-6 ~ probability
0.9975 / 0.0025) so that the bridge stays numerically well-behaved.

The unconditional path is a mixture: with weight p_today the path bridges to
+CAP, with weight 1-p_today it bridges to -CAP. We sample N paths from this
mixture using a standard discrete Brownian-bridge construction:

    L_s = L_t + (s - t)/(T - t) * (L_T - L_t) + sigma * B_bridge(s)

where B_bridge is a standard Brownian bridge on [t, T] with B_bridge(t) =
B_bridge(T) = 0. We discretise on a daily grid.

sigma is estimated from the realised volatility of past 30 days of logit
returns (annualised then re-scaled to per-day-sqrt for the bridge formula).

OUTPUTS
-------
For each future day s in {1, 2, ..., min(T-t, 60)}, we take percentiles
across the N=1000 simulated paths to produce p10/p25/p50/p75/p90 bands.
Each band is returned as a list of {t: ISO-date, p: probability} points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from pfm.cache_utils import get_cache
from pfm.config import get_settings
from pfm.sources.polymarket import PolymarketClient, fetch_factor_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terminal", tags=["terminal"])

# Fan-chart inputs (45 d price history + endDate) update on hour-scale, and
# the Monte-Carlo simulation itself is deterministic (fixed seed). A 60-s
# cache cuts a 200-450 ms call to <1 ms for the dominant warm-path use
# case of "user re-opens the same market detail".
_FAN_CACHE = get_cache("terminal_prob_fan", ttl=60)

# Cap the logit endpoint at +/-LOGIT_CAP to keep the bridge finite.
# 6 corresponds to probability ~ 0.9975, well outside any sensible range
# but small enough that exp() doesn't overflow in float64.
LOGIT_CAP: float = 6.0

# Number of Monte Carlo paths used to estimate percentile bands.
N_PATHS: int = 1000

# Maximum forecast horizon (days) regardless of how far away resolution is —
# fan charts beyond ~2 months are visually useless because the bridge has
# barely moved relative to the noise.
MAX_HORIZON_DAYS: int = 60

# Trading-style annualisation factor — Polymarket trades 7 days a week, so
# we use 365 not 252.
DAYS_PER_YEAR: int = 365

# Percentiles surfaced to the UI.
PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)

# Floor on sigma (per-day stdev of logit returns) so that flat / illiquid
# markets still produce a visible fan instead of a flat line.
SIGMA_FLOOR: float = 0.05


@dataclass(frozen=True)
class FanChartResult:
    today_p: float
    vol_ann: float
    days_to_resolution: int
    paths: dict[str, list[dict[str, float | str]]]


def _logit(p: np.ndarray | float, eps: float = 1e-4) -> np.ndarray | float:
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _estimate_sigma_per_day(prices: pd.Series) -> float:
    """Per-day stdev of logit returns over the last 30 obs.

    Uses ``ddof=0`` so a flat series yields exactly 0 (and we then fall back
    to ``SIGMA_FLOOR``).
    """
    if len(prices) < 2:
        return SIGMA_FLOOR
    tail = prices.tail(31)  # 31 obs -> 30 returns
    logits = _logit(tail.to_numpy(dtype=float))
    rets = np.diff(logits)
    if rets.size == 0:
        return SIGMA_FLOOR
    sd = float(np.std(rets, ddof=0))
    return max(sd, SIGMA_FLOOR)


def _simulate_bridge_paths(
    l_today: float,
    sigma_per_day: float,
    horizon_days: int,
    p_yes_today: float,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Simulate ``n_paths`` Brownian-bridge logit paths to either +/-LOGIT_CAP.

    Returns an array of shape ``(n_paths, horizon_days)`` containing the logit
    *at each future day 1..horizon_days* (today is excluded — it's just l_today).
    """
    rng = np.random.default_rng(seed)
    # Each path independently lands at +CAP w.p. p_yes_today else -CAP.
    endpoints = np.where(rng.random(n_paths) < p_yes_today, LOGIT_CAP, -LOGIT_CAP)

    # Build the bridge on a daily grid s = 1, 2, ..., horizon_days, with the
    # bridge anchored at t=0 (l_today) and t_end=horizon_days+1 (endpoint).
    # We forecast strictly *between* today and resolution, never the endpoint
    # itself, so we use t_end = horizon_days + 1.
    t_end = horizon_days + 1
    s = np.arange(1, horizon_days + 1, dtype=float)  # (H,)
    # Drift component: linear interpolation between l_today and endpoint.
    drift = l_today + np.outer((s / t_end), (endpoints - l_today))  # (H, n_paths)

    # Bridge component: standard BB with B(0)=B(t_end)=0 has Var(B(s)) =
    # sigma^2 * s * (t_end - s) / t_end. We construct it from a Wiener
    # process w: B(s) = w(s) - (s/t_end) * w(t_end).
    increments = rng.normal(loc=0.0, scale=sigma_per_day, size=(t_end, n_paths))
    w = np.cumsum(increments, axis=0)  # (t_end, n_paths) — w(1)..w(t_end)
    w_end = w[-1, :]  # (n_paths,)
    # Bridge values at s=1..horizon_days (i.e. rows 0..horizon_days-1 of w).
    w_s = w[:horizon_days, :]  # (H, n_paths)
    bridge = w_s - np.outer(s / t_end, w_end)  # (H, n_paths)

    paths = drift + bridge  # (H, n_paths)
    return paths.T  # (n_paths, H)


def compute_fan_chart(
    prices: pd.Series,
    days_to_resolution: int,
    n_paths: int = N_PATHS,
    seed: int = 12345,
) -> FanChartResult:
    """Compute the fan chart from a price series and a horizon.

    Parameters
    ----------
    prices:
        Daily YES probabilities, index = UTC dates, values in (0, 1).
    days_to_resolution:
        Whole days from today until the market resolves. Clamped to
        ``[1, MAX_HORIZON_DAYS]`` for the simulation grid.
    """
    if prices.empty:
        raise ValueError("price history is empty — cannot compute fan chart")

    today_p = float(prices.iloc[-1])
    today_p = float(np.clip(today_p, 1e-4, 1 - 1e-4))
    l_today = float(_logit(today_p))

    sigma_per_day = _estimate_sigma_per_day(prices)
    vol_ann = sigma_per_day * float(np.sqrt(DAYS_PER_YEAR))

    horizon = max(1, min(int(days_to_resolution), MAX_HORIZON_DAYS))

    paths_logit = _simulate_bridge_paths(
        l_today=l_today,
        sigma_per_day=sigma_per_day,
        horizon_days=horizon,
        p_yes_today=today_p,
        n_paths=n_paths,
        seed=seed,
    )
    # Convert to probability space and clip back into [0, 1] for any
    # tiny floating-point excursions caused by sigmoid().
    paths_prob = np.clip(_sigmoid(paths_logit), 0.0, 1.0)  # (n_paths, H)

    # Percentile across paths at each step.
    pct_arr = np.percentile(paths_prob, PERCENTILES, axis=0)  # (5, H)

    today_utc = pd.Timestamp(datetime.now(tz=UTC)).normalize()
    future_dates = [
        (today_utc + pd.Timedelta(days=int(i))).strftime("%Y-%m-%d") for i in range(1, horizon + 1)
    ]

    bands: dict[str, list[dict[str, float | str]]] = {}
    for row, pct in enumerate(PERCENTILES):
        bands[f"p{pct}"] = [
            {"t": future_dates[i], "p": float(pct_arr[row, i])} for i in range(horizon)
        ]

    return FanChartResult(
        today_p=today_p,
        vol_ann=float(vol_ann),
        days_to_resolution=horizon,
        paths=bands,
    )


def _days_to_resolution(end_date: str | None) -> int:
    """Parse Gamma's ``endDate`` ISO string into whole days from today (UTC)."""
    if not end_date:
        # Default to ~30 days if Gamma doesn't expose endDate (some markets
        # don't). The fan chart is still informative on a fixed window.
        return 30
    try:
        end_ts = pd.Timestamp(end_date)
    except (ValueError, TypeError):
        return 30
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    today_ts = pd.Timestamp(datetime.now(tz=UTC)).normalize()
    delta = (end_ts - today_ts).total_seconds() / 86400.0
    return max(1, int(np.ceil(delta)))


@router.get("/prob-fan/{slug}")
def get_prob_fan(
    slug: str,
    n_paths: int = Query(default=N_PATHS, ge=50, le=10_000),
) -> dict:
    """Return percentile paths of the YES probability under a Brownian bridge.

    Note: ``n_paths`` is the number of *Monte Carlo* paths used to estimate the
    percentile bands. The response always contains exactly five percentile
    paths (p10/p25/p50/p75/p90) — the parameter name follows the project
    convention but doesn't affect the shape of the JSON.
    """
    cache_key = (slug, int(n_paths))
    cached = _FAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    with PolymarketClient(
        gamma_url=settings.polymarket_gamma_url,
        clob_url=settings.polymarket_clob_url,
        timeout=settings.request_timeout_seconds,
    ) as client:
        try:
            meta = client.get_market_metadata(slug)
        except Exception as e:  # surface upstream errors as 404 either way
            logger.warning("prob-fan: metadata fetch failed for %s: %s", slug, e)
            raise HTTPException(status_code=404, detail=f"market not found: {slug}") from e

        end = pd.Timestamp(datetime.now(tz=UTC))
        start = end - timedelta(days=45)
        try:
            prices_df = fetch_factor_history(client, slug, start=start, end=end)
        except Exception as e:
            logger.warning("prob-fan: price fetch failed for %s: %s", slug, e)
            # Include the slug AND a recovery hint — a generic "upstream
            # price history error" left users staring at a red toast with
            # no idea whether their slug was wrong or Polymarket was down.
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Could not load Polymarket price history for slug={slug!r}. "
                    "The market may have just resolved (history truncated) or "
                    "Polymarket's CLOB endpoint is briefly unreachable — retry "
                    "in a few seconds."
                ),
            ) from e

    if prices_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No price history available for slug={slug!r} in the last 45 days. "
                "Resolved markets older than that window return an empty series — "
                "the fan chart requires a live or recently-resolved contract."
            ),
        )

    prices = prices_df["price"].astype(float)

    days = _days_to_resolution(meta.end_date)
    try:
        result = compute_fan_chart(prices, days_to_resolution=days, n_paths=n_paths)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    out = {
        "slug": slug,
        "today_p": result.today_p,
        "vol_ann": result.vol_ann,
        "days_to_resolution": result.days_to_resolution,
        "paths": result.paths,
    }
    _FAN_CACHE.set(cache_key, out)
    return out
