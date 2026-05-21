"""Side-by-side comparison endpoint for N≤4 prediction-market contracts.

This module fills a gap in the Terminal data hub: while individual market
endpoints exist (``/terminal/market/{slug}``, ``/terminal/market/{slug}/history``)
there was no way to fetch live data, history, and stats for several
contracts in a single round trip — a must-have for a Yahoo-Finance-style
"compare tickers" panel.

What the endpoint does
----------------------
Given 2 to 4 slugs, in parallel:
  1. Fetch the live Gamma snapshot (best-bid, best-ask, midpoint, volume).
  2. Fetch ``days`` of CLOB daily price history.
  3. Compute per-leg stats: 24h / 7d change, 30-day realised vol of
     log-returns, half-life of mean reversion, current spread (cents).
  4. Cross-leg: Pearson correlation matrix on **Δlogit innovations**
     (not on raw prices — raw prices are non-stationary and would inflate
     correlations spuriously). Same convention as
     :mod:`pfm.terminal_correlations`.
  5. If exactly 2 slugs are passed, also compute a tiny pairs-trade card:
     β-hedge from OLS, current spread ``a − β·b``, and its z-score
     against its own mean / stdev.

History bars are normalised to ``t0 = 100`` so the frontend can plot
all legs on a shared y-axis without manually rebasing.

Routing
-------
This module owns its own :class:`fastapi.APIRouter`. Wire it explicitly
in ``main.py``::

    from pfm.terminal_compare import router as terminal_compare_router
    app.include_router(terminal_compare_router)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Annotated, Any

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pfm import terminal as terminal_mod
from pfm.config import Settings, get_settings

logger = logging.getLogger(__name__)


# --- limits / cache ---------------------------------------------------------

MIN_SLUGS: int = 2
MAX_SLUGS: int = 4
DEFAULT_DAYS: int = 90
MIN_DAYS: int = 7
MAX_DAYS: int = 365
DEFAULT_CLIP_EPS: float = 0.01
CACHE_TTL_SECONDS: int = 30
HTTP_TIMEOUT_SECONDS: float = 10.0

# Module-level TTL cache keyed by (sorted_slugs_tuple, days).
_CACHE: dict[tuple[tuple[str, ...], int], tuple[float, dict[str, Any]]] = {}


def _cache_get(key: tuple[tuple[str, ...], int]) -> dict[str, Any] | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    expiry, payload = entry
    if expiry < time.time():
        _CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: tuple[tuple[str, ...], int], payload: dict[str, Any]) -> None:
    _CACHE[key] = (time.time() + CACHE_TTL_SECONDS, payload)


def clear_cache() -> None:
    """Test/utility helper — drop all cached entries."""
    _CACHE.clear()


# --- Pydantic schemas -------------------------------------------------------


class CompareLive(BaseModel):
    """Live snapshot fields per leg (Gamma)."""

    best_bid: float | None = None
    best_ask: float | None = None
    midpoint: float | None = None
    last_trade_price: float | None = None
    spread_cents: float | None = None
    volume_24hr: float | None = None
    volume_total: float | None = None
    liquidity: float | None = None
    one_day_price_change: float | None = None
    one_week_price_change: float | None = None


class CompareMeta(BaseModel):
    """Static metadata per leg (Gamma)."""

    slug: str
    question: str
    description: str | None = None
    theme: str | None = None
    end_date: str | None = None
    days_to_resolve: int | None = None
    age_days: int | None = None
    active: bool = True
    closed: bool = False


class CompareStats(BaseModel):
    """Derived stats for a single leg over the requested window."""

    n_obs: int
    current_price: float | None = None
    change_24h: float | None = None  # absolute Δ in probability over 1 day
    change_7d: float | None = None  # absolute Δ over 7 days
    realized_vol_30d: float | None = None  # stdev of Δlogit on tail-30
    half_life_days: float | None = None  # AR(1) half-life on Δlogit
    spread_cents: float | None = None  # snapshot ask-bid in cents


class CompareBar(BaseModel):
    """One bar in a normalised history series (``t0 = 100``)."""

    t: int  # unix seconds, UTC midnight
    price: float  # raw probability in [0, 1]
    indexed: float  # rebased to 100 at the first observation


class CompareLeg(BaseModel):
    """One slug's full payload."""

    slug: str
    live: CompareLive
    meta: CompareMeta
    stats: CompareStats
    history: list[CompareBar]


class ComparePairs(BaseModel):
    """Pairs-trade card returned only when ``n == 2``."""

    a: str
    b: str
    beta_hedge: float | None
    intercept: float | None
    spread_now: float | None
    spread_mean: float | None
    spread_std: float | None
    z_score: float | None
    n_obs: int


class CompareResponse(BaseModel):
    """Full /terminal/compare response envelope."""

    slugs: list[str]
    days: int
    legs: list[CompareLeg]
    correlation_matrix: dict[str, dict[str, float | None]] = Field(
        ..., description="Pearson corr of Δlogit innovations, keyed by slug."
    )
    pairs_trade: ComparePairs | None = None


# --- helpers ----------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _logit(p: pd.Series, *, clip_eps: float = DEFAULT_CLIP_EPS) -> pd.Series:
    """Element-wise logit with explicit clipping to [eps, 1 - eps]."""
    s = p.astype(float).clip(lower=clip_eps, upper=1.0 - clip_eps)
    return np.log(s / (1.0 - s))


def _innovations(prob_series: pd.Series) -> pd.Series:
    """Δlogit innovations — the standard project-wide stationary transform."""
    return _logit(prob_series).diff()


def _half_life_ar1(innov: pd.Series) -> float | None:
    """Half-life from an AR(1) fit on the innovations series.

    Fit ``Δx_{t+1} = a + b·x_t + ε`` on the *level* (not the innovation),
    where the level is reconstructed by accumulating innovations from 0.
    Half-life = ``log(0.5) / log(1 + b)`` when ``-2 < b < 0``.
    """
    s = innov.dropna()
    if len(s) < 10:
        return None
    # Reconstruct a level series whose first differences are these innovations.
    lvl = s.cumsum()
    x = lvl.shift(1).dropna()
    dy = lvl.diff().dropna()
    common = x.index.intersection(dy.index)
    if len(common) < 10:
        return None
    x = x.loc[common].to_numpy()
    dy = dy.loc[common].to_numpy()
    if x.std() == 0:
        return None
    # OLS slope of dy on x (with intercept).
    X = np.column_stack([np.ones_like(x), x])
    try:
        coef, *_ = np.linalg.lstsq(X, dy, rcond=None)
    except np.linalg.LinAlgError:
        return None
    b = float(coef[1])
    if not math.isfinite(b) or b >= 0 or b <= -2:
        return None
    try:
        hl = math.log(0.5) / math.log(1.0 + b)
    except (ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(hl) or hl <= 0:
        return None
    return hl


def _pct_change(series: pd.Series, lookback_days: int) -> float | None:
    """Difference between the latest and ``lookback_days``-ago observation.

    Returned as an *absolute* probability change (not a percent), since
    YES probabilities live in [0, 1] and percent-changes there are
    misleading near 0 or 1.
    """
    s = series.dropna()
    if len(s) < 2:
        return None
    last = float(s.iloc[-1])
    target_date = s.index[-1] - pd.Timedelta(days=lookback_days)
    prior_obs = s[s.index <= target_date]
    if prior_obs.empty:
        return None
    prior = float(prior_obs.iloc[-1])
    return last - prior


def _normalise_history(series: pd.Series) -> list[CompareBar]:
    """Convert a price Series to ``CompareBar`` rows rebased to 100 at t0."""
    s = series.dropna()
    if s.empty:
        return []
    base = float(s.iloc[0]) or 1e-9
    bars: list[CompareBar] = []
    for ts, price in s.items():
        try:
            unix_t = int(pd.Timestamp(ts).timestamp())
        except (TypeError, ValueError, OSError):
            continue
        bars.append(
            CompareBar(
                t=unix_t,
                price=float(price),
                indexed=float(price) / base * 100.0,
            )
        )
    return bars


def _pairs_trade(
    a_series: pd.Series, b_series: pd.Series, a_slug: str, b_slug: str
) -> ComparePairs:
    """OLS β-hedge of ``a`` on ``b`` plus a z-score on the residual spread."""
    df = pd.concat([a_series.rename("a"), b_series.rename("b")], axis=1).dropna()
    if len(df) < 5 or df["b"].std(ddof=0) == 0:
        return ComparePairs(
            a=a_slug,
            b=b_slug,
            beta_hedge=None,
            intercept=None,
            spread_now=None,
            spread_mean=None,
            spread_std=None,
            z_score=None,
            n_obs=len(df),
        )
    x = df["b"].to_numpy()
    y = df["a"].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return ComparePairs(
            a=a_slug,
            b=b_slug,
            beta_hedge=None,
            intercept=None,
            spread_now=None,
            spread_mean=None,
            spread_std=None,
            z_score=None,
            n_obs=len(df),
        )
    intercept = float(coef[0])
    beta = float(coef[1])
    spread = df["a"] - beta * df["b"]
    s_mean = float(spread.mean())
    s_std = float(spread.std(ddof=1)) if len(spread) > 1 else 0.0
    s_now = float(spread.iloc[-1])
    z = (s_now - s_mean) / s_std if s_std > 0 else None
    return ComparePairs(
        a=a_slug,
        b=b_slug,
        beta_hedge=beta if math.isfinite(beta) else None,
        intercept=intercept if math.isfinite(intercept) else None,
        spread_now=s_now if math.isfinite(s_now) else None,
        spread_mean=s_mean if math.isfinite(s_mean) else None,
        spread_std=s_std if (s_std > 0 and math.isfinite(s_std)) else None,
        z_score=float(z) if (z is not None and math.isfinite(z)) else None,
        n_obs=len(df),
    )


# --- async fetchers ---------------------------------------------------------


async def _fetch_gamma_market_async(
    http: httpx.AsyncClient, gamma_url: str, slug: str
) -> dict[str, Any]:
    """Async equivalent of :func:`pfm.terminal.fetch_gamma_market`.

    Falls back to ``closed=true`` for resolved markets. Raises ``LookupError``
    when no market is found by either filter.
    """
    base = gamma_url.rstrip("/")
    r = await http.get(f"{base}/markets", params={"slug": slug}, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status()
    arr = r.json() or []
    if isinstance(arr, list) and arr:
        return arr[0]
    r2 = await http.get(
        f"{base}/markets",
        params={"slug": slug, "closed": "true"},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    if r2.status_code == 200:
        arr2 = r2.json() or []
        if isinstance(arr2, list) and arr2:
            return arr2[0]
    raise LookupError(f"no market for slug={slug!r}")


async def _fetch_clob_history_async(
    http: httpx.AsyncClient, clob_url: str, token_id: str, days: int
) -> pd.DataFrame:
    """Async fetch of CLOB ``/prices-history`` daily bars.

    Returns a DataFrame indexed by UTC-midnight Timestamp with a single
    ``price`` column. Empty DataFrame if there is no data.
    """
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days + 1)
    params: dict[str, str | int] = {
        "market": token_id,
        "fidelity": 1440,
        "interval": "max",
        "startTs": int(start.timestamp()),
    }
    r = await http.get(
        f"{clob_url.rstrip('/')}/prices-history",
        params=params,
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    payload = r.json() or {}
    history = payload.get("history", []) if isinstance(payload, dict) else []
    if not history:
        return pd.DataFrame(columns=["price"])
    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.normalize()
    df = df.rename(columns={"p": "price"})[["date", "price"]]
    df = df.groupby("date", as_index=False).last()
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= (end - pd.Timedelta(days=days))]
    return df.set_index("date")


def _yes_token_id(market: dict[str, Any]) -> str | None:
    """Pull the YES side ``clobTokenIds[0]`` out of a Gamma market dict.

    ``clobTokenIds`` arrives as a JSON-encoded string per Polymarket convention.
    """
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    try:
        token_ids = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(token_ids, list) or len(token_ids) < 1:
        return None
    return str(token_ids[0])


async def _fetch_one_leg(
    http: httpx.AsyncClient, gamma_url: str, clob_url: str, slug: str, days: int
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    """Fetch (gamma_market, history_df) for one slug. Raises ``LookupError``."""
    market = await _fetch_gamma_market_async(http, gamma_url, slug)
    token_id = _yes_token_id(market)
    if not token_id:
        return slug, market, pd.DataFrame(columns=["price"])
    try:
        hist = await _fetch_clob_history_async(http, clob_url, token_id, days)
    except httpx.HTTPError as e:
        logger.warning("CLOB history failed for slug=%s: %s", slug, e)
        hist = pd.DataFrame(columns=["price"])
    return slug, market, hist


# --- core compute -----------------------------------------------------------


def _build_leg(
    slug: str, market: dict[str, Any], hist_df: pd.DataFrame
) -> tuple[CompareLeg, pd.Series]:
    """Shape a single leg's response and return (leg, price_series)."""
    live_dict = terminal_mod.shape_live(market)
    meta_dict = terminal_mod.shape_meta(market)

    if "price" in hist_df.columns:
        price_series = hist_df["price"].astype(float)
    else:
        price_series = pd.Series(dtype=float)
    price_series = price_series.dropna()
    n_obs = len(price_series)

    # 24h / 7d change: prefer the on-disk window, fall back to Gamma's
    # one_day_price_change which is computed by Polymarket directly.
    change_24h = _pct_change(price_series, 1)
    if change_24h is None:
        change_24h = _safe_float(market.get("oneDayPriceChange"))
    change_7d = _pct_change(price_series, 7)
    if change_7d is None:
        change_7d = _safe_float(market.get("oneWeekPriceChange"))

    innov = _innovations(price_series).dropna()
    rv30: float | None = None
    if len(innov) >= 5:
        tail = innov.iloc[-30:] if len(innov) >= 30 else innov
        sd = float(tail.std(ddof=1)) if len(tail) > 1 else float("nan")
        if math.isfinite(sd):
            rv30 = sd

    half_life = _half_life_ar1(innov)

    stats = CompareStats(
        n_obs=n_obs,
        current_price=float(price_series.iloc[-1]) if n_obs > 0 else None,
        change_24h=change_24h,
        change_7d=change_7d,
        realized_vol_30d=rv30,
        half_life_days=half_life,
        spread_cents=live_dict.get("spread_cents"),
    )
    leg = CompareLeg(
        slug=slug,
        live=CompareLive(**live_dict),
        meta=CompareMeta(**meta_dict),
        stats=stats,
        history=_normalise_history(price_series),
    )
    return leg, price_series


def _build_correlation_matrix(
    series_by_slug: dict[str, pd.Series],
) -> dict[str, dict[str, float | None]]:
    """Pearson corr matrix on Δlogit innovations across all leg pairs."""
    innov_by_slug: dict[str, pd.Series] = {
        slug: _innovations(s).dropna() for slug, s in series_by_slug.items()
    }
    slugs = list(innov_by_slug.keys())
    matrix: dict[str, dict[str, float | None]] = {}
    for a in slugs:
        row: dict[str, float | None] = {}
        for b in slugs:
            if a == b:
                row[b] = 1.0
                continue
            joined = pd.concat(
                [innov_by_slug[a].rename("a"), innov_by_slug[b].rename("b")],
                axis=1,
            ).dropna()
            if len(joined) < 5 or joined["a"].std(ddof=0) == 0 or joined["b"].std(ddof=0) == 0:
                row[b] = None
                continue
            c = float(joined["a"].corr(joined["b"]))
            row[b] = c if math.isfinite(c) else None
        matrix[a] = row
    return matrix


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-compare"])


def _parse_slugs(raw: str) -> list[str]:
    """Split comma-separated slug arg, trim, drop empties, preserve order."""
    return [s.strip() for s in raw.split(",") if s.strip()]


async def _gather_legs(
    slugs: list[str], days: int, gamma_url: str, clob_url: str
) -> list[tuple[str, dict[str, Any], pd.DataFrame]]:
    """Run ``_fetch_one_leg`` in parallel for every slug.

    Uses one shared :class:`httpx.AsyncClient` so connections pool nicely.
    """
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as http:
        tasks = [_fetch_one_leg(http, gamma_url, clob_url, slug, days) for slug in slugs]
        return await asyncio.gather(*tasks)


@router.get("/compare", response_model=CompareResponse)
async def get_compare(
    slugs: Annotated[str, Query(min_length=1, description="Comma-separated slugs (2..4).")],
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
) -> CompareResponse:
    """Side-by-side comparison of N≤4 prediction-market contracts.

    Args:
        slugs: comma-separated list of Polymarket slugs.
        days: lookback window in days for history + stats.

    Returns:
        ``CompareResponse`` with per-leg payloads, a Δlogit-innovation
        correlation matrix, and (when ``n == 2``) a pairs-trade card.
    """
    parsed = _parse_slugs(slugs)
    if not (MIN_SLUGS <= len(parsed) <= MAX_SLUGS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"slugs must contain between {MIN_SLUGS} and {MAX_SLUGS} entries, got {len(parsed)}"
            ),
        )
    if len(set(parsed)) != len(parsed):
        raise HTTPException(status_code=400, detail="duplicate slugs are not allowed")

    cache_key = (tuple(sorted(parsed)), int(days))
    cached = _cache_get(cache_key)
    if cached is not None:
        return CompareResponse.model_validate(cached)

    settings: Settings = get_settings()
    gamma_url = settings.polymarket_gamma_url
    clob_url = settings.polymarket_clob_url

    try:
        results = await _gather_legs(parsed, days, gamma_url, clob_url)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e

    legs: list[CompareLeg] = []
    series_by_slug: dict[str, pd.Series] = {}
    # Preserve the user's original slug order in the response.
    by_slug = {slug: (market, hist) for slug, market, hist in results}
    for slug in parsed:
        market, hist = by_slug[slug]
        leg, series = _build_leg(slug, market, hist)
        legs.append(leg)
        series_by_slug[slug] = series

    corr_matrix = _build_correlation_matrix(series_by_slug)

    pairs: ComparePairs | None = None
    if len(parsed) == 2:
        a_slug, b_slug = parsed
        pairs = _pairs_trade(series_by_slug[a_slug], series_by_slug[b_slug], a_slug, b_slug)

    response = CompareResponse(
        slugs=parsed,
        days=int(days),
        legs=legs,
        correlation_matrix=corr_matrix,
        pairs_trade=pairs,
    )
    _cache_set(cache_key, response.model_dump())
    return response


__all__ = [
    "CACHE_TTL_SECONDS",
    "MAX_SLUGS",
    "MIN_SLUGS",
    "CompareBar",
    "CompareLeg",
    "CompareLive",
    "CompareMeta",
    "ComparePairs",
    "CompareResponse",
    "CompareStats",
    "clear_cache",
    "get_compare",
    "router",
]
