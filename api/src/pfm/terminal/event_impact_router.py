"""Event-study analyzer for Polymarket contracts around a known event date.

Given a Polymarket ``slug`` and an ``event_date``, this router pulls the
contract's daily price history in a symmetric ±5-day window (configurable),
fetches a market-index daily return (SPY by default), and runs a textbook
**event study** in the Brown-Warner / MacKinlay tradition:

    1.  Compute log returns on the slug (the "asset") and on SPY (the
        "market proxy") and align them on the same UTC calendar dates.
    2.  Estimate the market model ``r_i = α + β · r_m + ε`` over an
        **estimation window** that ends just before the event window
        (default: 30 trading days, gap of 5 days from the event).
    3.  For each day in the event window ``[event_date - K, event_date + K]``
        compute the **abnormal return** ``AR_t = r_i,t - (α̂ + β̂ · r_m,t)``
        and the **cumulative abnormal return** ``CAR_t = Σ_{s ≤ t} AR_s``.
    4.  Test the null ``E[AR] = 0`` over the event window using the
        estimation-window residual σ as the scale. The test statistic is

            t = mean(AR_event) / (σ̂ / √N_event)

        i.e. a one-sample t-test under the standard event-study
        assumption that AR_t ~ iid N(0, σ²) under H₀.

The output payload mirrors the spec exactly::

    {slug, event_date, ar: [...], car: [...], t_stat, p_value, significant: bool}

Note: prediction-market prices aren't a market-cap-weighted asset, so
the market-model regression is admittedly weaker than for a stock.
We keep SPY as the proxy because the spec calls for it, and because for
issue-linked contracts (election, macro) SPY does capture the broad
macro state. The exposed schema includes the per-day vectors so the
caller can re-test under a different null if desired.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field
from scipy import stats

from pfm.sources.polymarket import PolymarketClient, fetch_factor_history

logger = logging.getLogger(__name__)


# --- Tunables ---------------------------------------------------------------

DEFAULT_WINDOW_DAYS: int = 5
"""Half-width of the event window in calendar days (±K around event_date)."""

MIN_WINDOW_DAYS: int = 1
MAX_WINDOW_DAYS: int = 30

DEFAULT_ESTIMATION_DAYS: int = 30
"""Length of the market-model estimation window in trading-day observations."""

MIN_ESTIMATION_DAYS: int = 10
MAX_ESTIMATION_DAYS: int = 252

DEFAULT_ESTIMATION_GAP_DAYS: int = 5
"""Buffer between estimation window end and event window start (calendar days).
Prevents pre-event leakage from contaminating the α/β estimate."""

DEFAULT_BENCHMARK_TICKER: str = "SPY"
"""Market index used as the systematic-risk proxy. Configurable per request."""

SIGNIFICANCE_ALPHA: float = 0.05
"""Two-sided test level used to set ``significant`` in the response."""

# Clipping range applied to raw Polymarket prices before taking log returns.
# Identical to the project default (CLAUDE.md "clipping epsilon"); a contract
# trading at 0 or 1 has resolved and yields ±∞ in log space — drop those days
# from the returns vector rather than poison the regression.
_CLIP_LOW: float = 0.005
_CLIP_HIGH: float = 0.995


# --- Pydantic schemas -------------------------------------------------------


class EventImpactDayPoint(BaseModel):
    """Per-day record over the event window."""

    date: str = Field(..., description="UTC calendar date, ISO YYYY-MM-DD.")
    offset_days: int = Field(
        ..., description="Trading-day offset from the event date (negative=before)."
    )
    asset_return: float | None = Field(
        None, description="Log return of the slug on this date (None if missing)."
    )
    market_return: float | None = Field(
        None, description="Log return of the benchmark on this date (None if missing)."
    )
    abnormal_return: float | None = Field(
        None,
        description="r_i - (α̂ + β̂·r_m). None when either input is missing.",
    )
    cumulative_abnormal_return: float | None = Field(
        None,
        description=(
            "Running sum of AR over the event window from the earliest "
            "available day to and including this one."
        ),
    )


class EventImpactResponse(BaseModel):
    slug: str
    event_date: str = Field(..., description="UTC calendar date, ISO YYYY-MM-DD.")
    benchmark: str = Field(..., description="Market-proxy ticker used (e.g. SPY).")
    window_days: int = Field(..., description="Half-width of the event window.")
    estimation_days: int = Field(..., description="Number of observations used for α/β estimation.")
    alpha: float = Field(..., description="Estimated market-model intercept α̂.")
    beta: float = Field(..., description="Estimated market-model slope β̂.")
    residual_sigma: float = Field(..., ge=0.0, description="Estimation-window residual σ̂.")
    n_event_days: int = Field(
        ..., ge=0, description="Number of usable AR observations in the event window."
    )

    # The flat vectors mandated by the spec — one entry per event-window day
    # that produced a usable AR. Indexed in chronological order. ``None``
    # entries are dropped from these top-level arrays but preserved in
    # ``per_day`` for the caller that wants full context.
    ar: list[float] = Field(..., description="Daily abnormal returns over the event window.")
    car: list[float] = Field(..., description="Cumulative abnormal returns over the event window.")

    t_stat: float = Field(..., description="One-sample t-statistic on mean(AR) under H₀: E[AR]=0.")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Two-sided p-value of the t-stat.")
    significant: bool = Field(..., description=f"True iff p < {SIGNIFICANCE_ALPHA}.")

    per_day: list[EventImpactDayPoint] = Field(
        default_factory=list,
        description="Per-day breakdown, including days with missing data.",
    )
    interpretation: str = Field(..., description="Plain-English summary of the event-study result.")


# --- pure-function core (testable without HTTP) -----------------------------


def _to_log_returns(prices: pd.Series) -> pd.Series:
    """Clip Polymarket-style prices to (0,1)-safe range and log-difference.

    Mirrors the rest of the codebase: prices ≤ 0 or ≥ 1 are clipped to
    ``[_CLIP_LOW, _CLIP_HIGH]`` before taking the log so a resolved-day
    print doesn't blow the series up. Non-finite entries are dropped.
    """
    if prices is None or len(prices) < 2:
        return pd.Series(dtype=float)
    s = prices.astype(float).copy()
    s = s.where(np.isfinite(s)).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s = s.clip(lower=_CLIP_LOW, upper=_CLIP_HIGH)
    lr = np.log(s / s.shift(1)).dropna()
    lr.name = prices.name or "r"
    return lr


def _normalise_dates(s: pd.Series) -> pd.Series:
    """Coerce the index to UTC-normalised calendar dates (drops time-of-day)."""
    if s.empty:
        return s
    idx = pd.to_datetime(s.index, utc=True).normalize()
    out = pd.Series(s.values, index=idx, name=s.name)
    # Collapse intra-day duplicates by keeping the last observation —
    # consistent with PolymarketClient.get_price_history.
    out = out.groupby(level=0).last().sort_index()
    return out


def estimate_market_model(
    asset_ret: pd.Series, market_ret: pd.Series
) -> tuple[float, float, float, int]:
    """Fit ``r_i = α + β·r_m + ε`` on aligned returns; return (α, β, σ̂, n).

    OLS is fine here — the estimation window is small (≤ 252 obs) and the
    autocorrelation is mild for daily returns. ``σ̂`` is the **residual**
    σ (ddof=2 since we estimate 2 params). Returns ``(0, 1, 0, 0)`` when
    the aligned series is too short (<3 obs).
    """
    common = asset_ret.index.intersection(market_ret.index)
    a = asset_ret.loc[common].astype(float)
    m = market_ret.loc[common].astype(float)
    if len(a) < 3:
        return 0.0, 1.0, 0.0, len(a)
    m_vals = m.values.astype(float)
    a_vals = a.values.astype(float)
    # Strip any rows where either input is non-finite — np.polyfit blows up
    # with an opaque LinAlgError on NaN/inf, and these can sneak in via
    # log-of-clipped-zero edge cases.
    mask = np.isfinite(m_vals) & np.isfinite(a_vals)
    m_vals = m_vals[mask]
    a_vals = a_vals[mask]
    if len(m_vals) < 3:
        return 0.0, 1.0, 0.0, len(m_vals)
    # If the market return has zero variance (e.g. holiday-only slice),
    # β is unidentified — fall back to α = mean(a), β = 0.
    m_var = float(np.var(m_vals, ddof=1)) if len(m_vals) > 1 else 0.0
    if m_var < 1e-18:
        alpha = float(np.mean(a_vals))
        beta = 0.0
    else:
        # np.polyfit returns highest-degree coeff first → [β, α]
        coeffs = np.polyfit(m_vals, a_vals, deg=1)
        beta, alpha = float(coeffs[0]), float(coeffs[1])
    resid = a_vals - (alpha + beta * m_vals)
    # Residual σ with the 2-parameter degrees-of-freedom correction.
    if len(resid) > 2:
        sigma = float(np.sqrt(np.sum(resid**2) / (len(resid) - 2)))
    else:
        sigma = float(np.std(resid, ddof=0))
    if not math.isfinite(sigma):
        sigma = 0.0
    if not math.isfinite(alpha):
        alpha = 0.0
    if not math.isfinite(beta):
        beta = 1.0
    return alpha, beta, sigma, len(m_vals)


def run_event_study(
    asset_prices: pd.Series,
    market_prices: pd.Series,
    event_date: pd.Timestamp,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    estimation_days: int = DEFAULT_ESTIMATION_DAYS,
    estimation_gap_days: int = DEFAULT_ESTIMATION_GAP_DAYS,
) -> dict:
    """Compute AR / CAR / t-stat for a single event.

    Pure function — takes already-fetched price series so we can test
    against synthetic DGPs without touching the network.

    Returns a dict with keys: ``alpha``, ``beta``, ``residual_sigma``,
    ``per_day`` (list of EventImpactDayPoint-compatible dicts), ``ar``,
    ``car``, ``t_stat``, ``p_value``, ``significant``, ``n_event_days``.

    Edge cases:
        * If no usable AR observations remain in the event window
          (e.g. prediction market hadn't started trading yet) the
          function returns zeros and ``significant=False`` rather than
          NaN, so the JSON payload stays valid.
        * If the residual σ is ~0 (asset perfectly tracks the model in
          the estimation window) the t-stat is set to 0 to avoid /0;
          this is the conservative behaviour because we have no
          variance against which to detect surprise.
    """
    # Coerce prices → returns and normalise indices.
    asset_ret = _normalise_dates(_to_log_returns(asset_prices))
    market_ret = _normalise_dates(_to_log_returns(market_prices))

    event_d = (
        pd.Timestamp(event_date).tz_localize("UTC")
        if (pd.Timestamp(event_date).tzinfo is None)
        else pd.Timestamp(event_date).tz_convert("UTC")
    )
    event_d = event_d.normalize()

    # Event-window calendar dates.
    window_start = event_d - pd.Timedelta(days=window_days)
    event_dates = [window_start + pd.Timedelta(days=i) for i in range(2 * window_days + 1)]

    # Estimation window: ``estimation_days`` trading observations ending
    # ``estimation_gap_days`` calendar days before the event window. We
    # take the closest available observations from the asset series.
    est_window_end = window_start - pd.Timedelta(days=estimation_gap_days)
    est_asset = asset_ret[asset_ret.index <= est_window_end].tail(estimation_days)
    est_market = market_ret[market_ret.index <= est_window_end].tail(estimation_days)
    alpha, beta, sigma, n_est = estimate_market_model(est_asset, est_market)

    # Per-day AR + CAR over the event window.
    per_day: list[dict] = []
    ar_values: list[float] = []
    car: float = 0.0
    car_values: list[float] = []
    for d in event_dates:
        d_norm = d.normalize()
        a_r = asset_ret.get(d_norm, None)
        m_r = market_ret.get(d_norm, None)
        a_val = float(a_r) if a_r is not None and np.isfinite(a_r) else None
        m_val = float(m_r) if m_r is not None and np.isfinite(m_r) else None
        ar_val: float | None
        if a_val is not None and m_val is not None:
            ar_val = a_val - (alpha + beta * m_val)
            if not np.isfinite(ar_val):
                ar_val = None
        else:
            ar_val = None
        if ar_val is not None:
            car += ar_val
            ar_values.append(ar_val)
            car_values.append(car)
            point_car = car
        else:
            point_car = car if car_values else None  # type: ignore[assignment]
        offset = (d_norm - event_d).days
        per_day.append(
            {
                "date": d_norm.strftime("%Y-%m-%d"),
                "offset_days": int(offset),
                "asset_return": a_val,
                "market_return": m_val,
                "abnormal_return": ar_val,
                "cumulative_abnormal_return": point_car,
            }
        )

    # Test statistic.
    n_event = len(ar_values)
    if n_event >= 1 and sigma > 0.0:
        mean_ar = float(np.mean(ar_values))
        t_stat = mean_ar / (sigma / math.sqrt(n_event))
        if not math.isfinite(t_stat):
            t_stat = 0.0
        # Use estimation-window dof for the null distribution; this is
        # the textbook event-study approach (MacKinlay 1997 eq. 14).
        dof = max(1, n_est - 2)
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof)))
        p_value = max(0.0, min(1.0, p_value))
    else:
        t_stat = 0.0
        p_value = 1.0
    significant = bool(p_value < SIGNIFICANCE_ALPHA and n_event >= 1)

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "residual_sigma": float(sigma),
        "n_event_days": int(n_event),
        "per_day": per_day,
        "ar": ar_values,
        "car": car_values,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": significant,
    }


def _interpret(result: dict) -> str:
    """One-sentence plain-English summary of the event-study outcome."""
    n = result["n_event_days"]
    if n == 0:
        return "No usable price observations in the event window — cannot test."
    car_final = result["car"][-1] if result["car"] else 0.0
    direction = "positive" if car_final > 0 else "negative" if car_final < 0 else "flat"
    sig = "statistically significant" if result["significant"] else "not statistically significant"
    return (
        f"Cumulative abnormal return over the event window was {direction} "
        f"({car_final:+.3f}); the mean-AR t-test is {sig} "
        f"(t={result['t_stat']:.2f}, p={result['p_value']:.3f}, n={n})."
    )


# --- Benchmark fetcher ------------------------------------------------------


def _fetch_benchmark_prices(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Fetch daily adjusted closes for the market-index ticker.

    Imported lazily so tests can stub the function without paying the
    yfinance import cost on every collection.
    """
    from pfm.equity_factors import fetch_equity_history

    return fetch_equity_history(ticker, start=start, end=end)


# --- routing ----------------------------------------------------------------


router = APIRouter(prefix="/terminal", tags=["terminal-event-impact"])


def _get_polymarket_client(request: Request) -> PolymarketClient:
    """Dependency: pull the shared PolymarketClient off ``app.state``."""
    poly: PolymarketClient | None = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


def _parse_event_date(s: str) -> pd.Timestamp:
    """Parse ``YYYY-MM-DD`` (or full ISO) and normalise to UTC midnight."""
    try:
        ts = pd.Timestamp(s)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"event_date must be ISO YYYY-MM-DD: {s!r} ({e})",
        ) from e
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


@router.get(
    "/event-impact/{slug}",
    response_model=EventImpactResponse,
    summary=(
        "Event-study analyzer: abnormal returns of a Polymarket slug around "
        "a known event date, with SPY as the market proxy."
    ),
)
def get_event_impact(
    request: Request,
    slug: Annotated[str, Path(min_length=1, max_length=160)],
    event_date: Annotated[str, Query(description="UTC date, ISO YYYY-MM-DD.")],
    window_days: Annotated[
        int, Query(ge=MIN_WINDOW_DAYS, le=MAX_WINDOW_DAYS)
    ] = DEFAULT_WINDOW_DAYS,
    estimation_days: Annotated[
        int, Query(ge=MIN_ESTIMATION_DAYS, le=MAX_ESTIMATION_DAYS)
    ] = DEFAULT_ESTIMATION_DAYS,
    benchmark: Annotated[str, Query(min_length=1, max_length=10)] = DEFAULT_BENCHMARK_TICKER,
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> EventImpactResponse:
    """Run an event study on the given slug centred at ``event_date``.

    Fetches the slug's daily price series spanning the estimation window +
    event window (plus padding), pulls the benchmark daily returns over
    the same horizon, estimates the market model on the pre-event sample,
    and computes AR / CAR / t-stat over ±``window_days`` around the event.
    """
    event_ts = _parse_event_date(event_date)

    # Horizon: estimation window + gap + event window, plus padding for
    # weekends / market holidays. ``estimation_days`` is in trading days
    # but we ask for ~1.6× calendar days to make sure we collect enough
    # observations even after weekend gaps.
    pad_before_days = int(estimation_days * 1.7 + DEFAULT_ESTIMATION_GAP_DAYS + window_days + 7)
    start_ts = event_ts - pd.Timedelta(days=pad_before_days)
    end_ts = event_ts + pd.Timedelta(days=window_days + 2)

    # --- asset prices (Polymarket) -----------------------------------------
    try:
        asset_df = fetch_factor_history(poly, slug, start=start_ts, end=end_ts)
    except Exception as e:  # surfaces httpx + PolymarketError + LookupError
        raise HTTPException(
            status_code=502 if "timeout" in str(e).lower() else 404,
            detail=f"polymarket history fetch failed for {slug!r}: {e}",
        ) from e
    if asset_df is None or asset_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"no price history for slug={slug!r} around {event_date}",
        )
    asset_prices = asset_df["price"] if "price" in asset_df.columns else asset_df.iloc[:, 0]
    asset_prices.name = slug

    # --- benchmark prices (yfinance / SPY) ---------------------------------
    try:
        market_prices = _fetch_benchmark_prices(benchmark, start=start_ts, end=end_ts)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"benchmark fetch failed for {benchmark!r}: {e}",
        ) from e
    if market_prices is None or len(market_prices) < 5:
        raise HTTPException(
            status_code=502,
            detail=f"benchmark {benchmark!r} returned too few observations",
        )

    result = run_event_study(
        asset_prices,
        market_prices,
        event_ts,
        window_days=window_days,
        estimation_days=estimation_days,
        estimation_gap_days=DEFAULT_ESTIMATION_GAP_DAYS,
    )

    return EventImpactResponse(
        slug=slug,
        event_date=event_ts.strftime("%Y-%m-%d"),
        benchmark=benchmark.upper(),
        window_days=window_days,
        estimation_days=estimation_days,
        alpha=result["alpha"],
        beta=result["beta"],
        residual_sigma=result["residual_sigma"],
        n_event_days=result["n_event_days"],
        ar=result["ar"],
        car=result["car"],
        t_stat=result["t_stat"],
        p_value=result["p_value"],
        significant=result["significant"],
        per_day=[EventImpactDayPoint(**p) for p in result["per_day"]],
        interpretation=_interpret(result),
    )


__all__ = [
    "DEFAULT_BENCHMARK_TICKER",
    "DEFAULT_ESTIMATION_DAYS",
    "DEFAULT_ESTIMATION_GAP_DAYS",
    "DEFAULT_WINDOW_DAYS",
    "SIGNIFICANCE_ALPHA",
    "EventImpactDayPoint",
    "EventImpactResponse",
    "estimate_market_model",
    "router",
    "run_event_study",
]
