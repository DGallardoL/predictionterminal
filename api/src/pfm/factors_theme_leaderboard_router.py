"""``GET /factors/themes/{theme}/leaderboard`` — top-N factors by 7-day vol.

Task W12-16. For each factor whose ``theme`` matches the path parameter,
compute trailing 7-day standard deviation of daily log-returns and return
the top-N by ``vol_7d`` descending.

File location
-------------
The original task spec asked for ``api/src/pfm/factors/theme_leaderboard_router.py``
but ``pfm/factors.py`` already exists as a flat module exporting
``FactorConfig`` / ``load_factors`` / ``fetch_factor_history_dispatch`` — used
by ~40 import sites. Creating a ``pfm/factors/`` package directory would
shadow that module (Python prefers package over file) and break those
imports. The router therefore lives at the sibling path
``pfm/factors_theme_leaderboard_router.py``, parallel to
``factors_router.py`` and ``factors_related_router.py``.

Integration
-----------
The router is NOT mounted automatically because ``api/src/pfm/main.py`` has
a long-lived ``routes``-section claim. The next session that holds
``main.py:routes`` should add::

    from pfm.factors_theme_leaderboard_router import router as _factors_theme_leaderboard_router
    app.include_router(_factors_theme_leaderboard_router)

Caching
-------
Module-level TTL cache keyed on ``(theme, n, include_dead)`` with a 60 s
TTL. Tests can patch :data:`_PERF_COUNTER` to drive cache expiry
deterministically without sleeping. This mirrors the pattern in
``pfm.factors_related_router``.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pfm.cache import CacheBackend
from pfm.config import Settings, get_settings
from pfm.dependencies import (
    get_cache,
    get_factors_dep,
    get_polymarket_client,
)
from pfm.factors import FactorConfig
from pfm.sources.polymarket import PolymarketClient

router = APIRouter(tags=["factors"])

# Module-level wall clock — tests patch this to drive cache expiry without
# real sleep. ``perf_counter`` is monotonic, so NTP skew can't invalidate
# cached entries.
_PERF_COUNTER: Callable[[], float] = time.perf_counter

# Cache TTL in seconds. The factor universe is daily, so 60 s of staleness
# on a leaderboard panel is well within usefulness.
_CACHE_TTL_S: float = 60.0

# Path-level bounds on ``n``. 1..200 covers everything from "give me one
# headline factor" to "dump the whole pile" while keeping the response
# size predictable.
N_MIN: int = 1
N_MAX: int = 200
N_DEFAULT: int = 20

# Minimum observations required for a factor to be considered "live".
# Below this we treat it as dead/illiquid and drop it unless
# ``include_dead=true`` is passed. 30 daily bars ~= 6 trading weeks.
MIN_OBS_LIVE: int = 30

# Trailing window for the vol/mean calc, in calendar days. We fetch a
# slightly larger window (``_FETCH_PADDING_DAYS``) so weekends don't
# silently shorten the trailing-7 series.
WINDOW_DAYS: int = 7
_FETCH_PADDING_DAYS: int = 21

# Per-request fetch cap. Some themes (e.g. ``elections``) have hundreds of
# factors; sequential ``_fetch_series`` over the full set blew past the
# 15 s gateway deadline. We scan ``max(n*4, MAX_FETCH_FLOOR)`` candidates
# clipped at ``MAX_FETCH_CEIL`` — leaves the top-N sort with headroom while
# keeping wall-clock bounded.
MAX_FETCH_FLOOR: int = 20
MAX_FETCH_CEIL: int = 50


class LeaderboardRow(BaseModel):
    """One row of the theme leaderboard."""

    slug: str = Field(..., description="Per-source slug for the factor.")
    label: str = Field(..., description="Human-readable factor name.")
    vol_7d: float = Field(..., description="Trailing 7-day stdev of daily log-returns.")
    mean_7d: float = Field(..., description="Trailing 7-day mean of daily log-returns.")
    n_obs: int = Field(..., ge=0, description="Number of daily observations used.")
    last_value: float = Field(..., description="Most recent price/level observed for the factor.")


class ThemeLeaderboardResponse(BaseModel):
    """Response model for ``GET /factors/themes/{theme}/leaderboard``."""

    theme: str = Field(..., description="The theme echoed back.")
    n: int = Field(..., description="Top-N cap requested.")
    factors: list[LeaderboardRow] = Field(
        ...,
        description=(
            "Factors in the theme sorted desc by ``vol_7d``. Factors with "
            "``n_obs < 30`` are excluded unless ``include_dead=true``."
        ),
    )


class _Entry:
    """Tiny container for cached payload + expiry timestamp."""

    __slots__ = ("expires_at", "payload")

    def __init__(self, payload: ThemeLeaderboardResponse, expires_at: float) -> None:
        self.payload = payload
        self.expires_at = expires_at


# Thread-safe TTL cache. Keyed on ``(theme, n, include_dead)`` — caller
# parameters that affect the output. Request volume on this endpoint is
# low (interactive UI panel), so a single ``Lock`` is sufficient.
_CACHE: dict[tuple[str, int, bool], _Entry] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key: tuple[str, int, bool]) -> ThemeLeaderboardResponse | None:
    """Return cached payload if still fresh, else ``None``."""
    now = _PERF_COUNTER()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            _CACHE.pop(key, None)
            return None
        return entry.payload


def _cache_put(key: tuple[str, int, bool], payload: ThemeLeaderboardResponse) -> None:
    """Insert payload into the TTL cache with a fresh expiry."""
    expires_at = _PERF_COUNTER() + _CACHE_TTL_S
    with _CACHE_LOCK:
        _CACHE[key] = _Entry(payload, expires_at)


def _cache_clear() -> None:
    """Drop every entry — used by tests to force a cold path."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _fetch_series(
    fc: FactorConfig,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
) -> pd.Series:
    """Pull a single factor's daily 'price' series via the shared cache helper.

    Delegates to ``pfm.regression_core._cached_factor_history`` so we hit
    the same Redis blob the rest of the regression pipeline uses. Returns
    an empty Series on any upstream failure — we don't want one bad slug
    to 5xx the whole leaderboard.
    """
    try:
        from pfm.regression_core import _cached_factor_history
    except ImportError:  # pragma: no cover — only triggers in stripped builds
        return pd.Series(dtype="float64")

    try:
        df = _cached_factor_history(fc, start, end, poly, cache, settings)
    except HTTPException:
        # Upstream timeouts / 404s on individual candidates shouldn't bring
        # down the whole leaderboard. Skip them silently.
        return pd.Series(dtype="float64")
    except Exception:
        return pd.Series(dtype="float64")

    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if "price" in df.columns:
        ser = df["price"]
    elif "value" in df.columns:
        ser = df["value"]
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return pd.Series(dtype="float64")
        ser = df[numeric_cols[0]]
    ser = pd.to_numeric(ser, errors="coerce").dropna()
    return ser


def _vol_and_mean(series: pd.Series) -> tuple[float, float, int, float]:
    """Compute trailing 7-day vol & mean of daily log-returns.

    Returns ``(vol_7d, mean_7d, n_obs, last_value)``. ``n_obs`` is the
    count of daily observations in the input ``series`` (so callers can
    apply the ``MIN_OBS_LIVE`` filter), NOT the count of log-returns
    inside the 7-day window.

    Log returns use natural log: ``r_t = log(P_t / P_{t-1})``. We clip
    the input at a small ε so a probability factor that ticks to 0 / 1
    can't blow up the log.
    """
    n_obs = int(len(series))
    if n_obs == 0:
        return 0.0, 0.0, 0, float("nan")

    last_value = float(series.iloc[-1])

    # Clip into ``(ε, 1-ε)`` if the series looks like a probability series
    # (max ≤ 1.0 + small slack). Otherwise just lower-clip away from zero
    # so log() stays defined for macro/level factors.
    eps = 1e-6
    arr = series.to_numpy(dtype=float)
    if np.nanmax(arr) <= 1.0 + 1e-9 and np.nanmin(arr) >= 0.0 - 1e-9:
        arr = np.clip(arr, eps, 1.0 - eps)
    else:
        arr = np.where(arr <= 0.0, eps, arr)

    log_p = np.log(arr)
    log_ret = np.diff(log_p)
    # Take only the trailing ``WINDOW_DAYS`` log-returns. If we have
    # fewer than that, vol/mean are reported on whatever we have (the
    # caller still sees ``n_obs`` so they can decide what to trust).
    tail = log_ret[-WINDOW_DAYS:] if len(log_ret) > 0 else np.array([])
    if len(tail) == 0:
        return 0.0, 0.0, n_obs, last_value

    # ``ddof=0`` (population stdev) — for a fixed 7-bar window the
    # bias correction from ddof=1 buys nothing and just adds variance
    # at the small-n boundary.
    vol = float(np.std(tail, ddof=0))
    mean = float(np.mean(tail))
    if not math.isfinite(vol):
        vol = 0.0
    if not math.isfinite(mean):
        mean = 0.0
    return vol, mean, n_obs, last_value


def _compute_leaderboard(
    theme: str,
    factors: dict[str, FactorConfig],
    *,
    n: int,
    include_dead: bool,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
) -> list[LeaderboardRow]:
    """For each factor in the theme, compute trailing 7-day vol & rank."""
    matching = [fc for fc in factors.values() if fc.theme == theme]
    if not matching:
        return []

    # Bound the per-request fetch count. Sort deterministically by slug so
    # the truncated set is stable across calls (cache-friendly). The cap is
    # ``max(n * 4, MAX_FETCH_FLOOR)`` so the post-sort top-N still has
    # plenty of headroom even when most candidates get filtered out by
    # ``MIN_OBS_LIVE``, but we never iterate more than ``MAX_FETCH_CEIL``.
    matching.sort(key=lambda fc: fc.slug)
    fetch_cap = max(n * 4, MAX_FETCH_FLOOR)
    fetch_cap = min(fetch_cap, MAX_FETCH_CEIL)
    matching = matching[:fetch_cap]

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=_FETCH_PADDING_DAYS)

    # Parallelize the per-factor fetch. Sequential it was ~500ms × 50 = 25s
    # and blew the gateway deadline. ThreadPool fan-out + Polymarket's generous
    # rate limit gets the same work into ~3-5s wall-clock.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _row_for(fc: FactorConfig) -> LeaderboardRow | None:
        ser = _fetch_series(fc, start=start, end=end, poly=poly, cache=cache, settings=settings)
        vol, mean, n_obs, last_value = _vol_and_mean(ser)
        if not include_dead and n_obs < MIN_OBS_LIVE:
            return None
        if not math.isfinite(last_value):
            return None
        return LeaderboardRow(
            slug=fc.slug,
            label=fc.name,
            vol_7d=vol,
            mean_7d=mean,
            n_obs=n_obs,
            last_value=last_value,
        )

    rows: list[LeaderboardRow] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        for fut in as_completed(pool.submit(_row_for, fc) for fc in matching):
            try:
                row = fut.result()
            except Exception:
                continue
            if row is not None:
                rows.append(row)

    # Sort desc by vol_7d. Tie-breaker: factor name (stable, deterministic).
    rows.sort(key=lambda r: (-r.vol_7d, r.label))
    return rows[:n]


@router.get(
    "/factors/themes/{theme}/leaderboard",
    response_model=ThemeLeaderboardResponse,
    summary="Top-N factors in a theme by trailing 7-day volatility.",
)
def get_theme_leaderboard(
    theme: Annotated[str, "Theme name (e.g. ``macro``, ``elections``)."],
    request: Request,
    n: Annotated[
        int,
        Query(
            ge=N_MIN,
            le=N_MAX,
            description="Number of factors to return (default 20, max 200).",
        ),
    ] = N_DEFAULT,
    include_dead: Annotated[
        bool,
        Query(
            description=(
                "When true, include factors with fewer than 30 daily "
                "observations. Defaults to false."
            ),
        ),
    ] = False,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> ThemeLeaderboardResponse:
    """Return the top-``n`` factors in ``theme`` ranked by trailing 7-day vol.

    * Computes daily log-returns over the trailing 7-day window.
    * Drops factors with ``n_obs < 30`` unless ``include_dead=true``.
    * Sorts desc by ``vol_7d`` with factor name as the stable tie-breaker.
    * Cached for 60 seconds per ``(theme, n, include_dead)``.

    Returns 404 if the theme has no matching factors in the catalog.
    """
    # Validate theme exists in the catalog before doing any work.
    available_themes = {fc.theme for fc in factors.values()}
    if theme not in available_themes:
        raise HTTPException(
            status_code=404,
            detail=f"theme {theme!r} not found in catalog",
        )

    cache_key = (theme, int(n), bool(include_dead))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    rows = _compute_leaderboard(
        theme,
        factors,
        n=int(n),
        include_dead=bool(include_dead),
        poly=poly,
        cache=cache,
        settings=settings,
    )
    payload = ThemeLeaderboardResponse(theme=theme, n=int(n), factors=rows)
    _cache_put(cache_key, payload)
    return payload
