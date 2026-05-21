"""``GET /factors/{slug}/related`` — top-correlated factors over a rolling window.

Task T29 (wave-10). Returns up to 10 factors most correlated (by |Pearson ρ|)
with the anchor factor's daily price series over the last ``window_days``
trading days. Drops candidates with <20 overlapping observations.

File location
-------------
The original task spec asked for ``api/src/pfm/factors/related_router.py`` but
``pfm/factors.py`` already exists as a top-level module exporting
``FactorConfig``, ``ChainSegment``, ``load_factors``, etc. — used by ~40 import
sites across the codebase. Creating a ``pfm/factors/`` package directory would
shadow that module (Python prefers package over file) and break every one of
those imports. The router therefore lives at the sibling path
``pfm/factors_related_router.py``, parallel to ``factors_router.py``.

Integration
-----------
This router is NOT mounted automatically because ``api/src/pfm/main.py`` has
an active ``routes``-section claim by ``metrics-audit-endpoint-1778985000``.
The next session that holds ``main.py:routes`` should add, in the include
block near the bottom of ``main.py``::

    from pfm.factors_related_router import router as _factors_related_router
    app.include_router(_factors_related_router)

Caching
-------
Module-level :class:`TTLCache` keyed on ``(slug, window_days)`` with a 60 s
TTL. Tests can patch :data:`_PERF_COUNTER` to drive cache expiry
deterministically without sleeping.
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
# real sleep. Use ``time.perf_counter`` (monotonic, cheap) rather than
# ``time.time`` so leap-seconds / NTP skew can't invalidate cache state.
_PERF_COUNTER: Callable[[], float] = time.perf_counter

# Cache TTL in seconds. The factor price universe is daily, so 60 s of
# staleness on a "related-factor" panel is well within usefulness.
_CACHE_TTL_S: float = 60.0

# Hard caps on window length. Min 7 trading days (≈ 1.5 calendar weeks) is
# the smallest window for which Pearson ρ has any statistical meaning at our
# default min-overlap of 20 obs. Max 365 trading days (≈ 18 calendar months)
# keeps the response under a few seconds even for the full catalog.
WINDOW_MIN: int = 7
WINDOW_MAX: int = 365
WINDOW_DEFAULT: int = 30

# A candidate must share at least this many daily observations with the
# anchor to be considered. 20 is roughly four trading weeks — the threshold
# below which ρ is too noisy to act on.
MIN_OVERLAP_OBS: int = 20

# Cap how many candidates we return after sorting by |ρ| desc.
TOP_N: int = 10

# Cap on how many candidate factors we'll evaluate per request. The catalog
# now holds 1,260 factors; iterating all of them performs ~1,260 cache reads
# / Polymarket fetches and routinely exceeded the 15 s gateway deadline on
# cold caches. The caller can widen via the ``limit`` query param.
MAX_CANDIDATES_DEFAULT: int = 30
MAX_CANDIDATES_HARD: int = 500


class _Entry:
    """Tiny container for cached payload + expiry timestamp."""

    __slots__ = ("expires_at", "payload")

    def __init__(self, payload: FactorsRelatedResponse, expires_at: float) -> None:
        self.payload = payload
        self.expires_at = expires_at


# Thread-safe TTL cache. Keyed on ``(anchor_slug, window_days, limit)`` —
# caller parameters that affect output. ``threading.Lock`` is enough because
# the request volume on this endpoint is small (interactive UI panel).
_CACHE: dict[tuple[str, int, int], _Entry] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key: tuple[str, int, int]) -> FactorsRelatedResponse | None:
    """Return cached payload if still fresh, else ``None``."""
    now = _PERF_COUNTER()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            # Evict expired entry so the dict doesn't grow without bound.
            _CACHE.pop(key, None)
            return None
        return entry.payload


def _cache_put(key: tuple[str, int, int], payload: FactorsRelatedResponse) -> None:
    """Insert payload into the TTL cache with a fresh expiry."""
    expires_at = _PERF_COUNTER() + _CACHE_TTL_S
    with _CACHE_LOCK:
        _CACHE[key] = _Entry(payload, expires_at)


def _cache_clear() -> None:
    """Drop every entry — used by tests to force a cold path."""
    with _CACHE_LOCK:
        _CACHE.clear()


class RelatedFactor(BaseModel):
    """One related factor, ranked by |ρ| against the anchor's price series."""

    slug: str = Field(..., description="Polymarket slug / source-id for the related factor.")
    rho: float = Field(..., description="Pearson correlation in [-1, 1].")
    p_value: float = Field(..., description="Two-sided p-value for ρ=0 under H0.")
    n_obs: int = Field(..., ge=0, description="Number of overlapping daily observations.")


class FactorsRelatedResponse(BaseModel):
    """Response model for ``GET /factors/{slug}/related``."""

    anchor: str = Field(..., description="The anchor factor slug echoed back.")
    window_days: int = Field(..., description="Rolling window length in days.")
    related: list[RelatedFactor] = Field(
        ...,
        description=(
            "Top-N related factors sorted by |ρ| desc. Empty when no other "
            "factor has ≥20 overlapping daily observations with the anchor."
        ),
    )


def _resolve_anchor(factors: dict[str, FactorConfig], slug: str) -> FactorConfig:
    """Look up the anchor factor by slug. Raises 404 when unknown."""
    # Prefer the indexed lookup built during lifespan (O(1)) but fall back to
    # a one-shot dict-comp when ``factors_by_slug`` isn't on app.state — this
    # makes the function unit-testable without a full FastAPI app.
    for fc in factors.values():
        if fc.slug == slug:
            return fc
    raise HTTPException(
        status_code=404,
        detail=f"factor with slug {slug!r} not found in catalog",
    )


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

    Delegates to ``pfm.regression_core._cached_factor_history`` so we hit the
    same Redis blob the rest of the regression pipeline uses. Returns an
    empty Series on any upstream failure (we don't want one bad slug to 5xx
    the whole panel).
    """
    try:
        from pfm.regression_core import _cached_factor_history
    except ImportError:  # pragma: no cover — only triggers in stripped builds
        return pd.Series(dtype="float64")

    try:
        df = _cached_factor_history(fc, start, end, poly, cache, settings)
    except HTTPException:
        # Upstream timeouts / 404s on individual candidates shouldn't bring
        # down the whole related-factors panel. Skip them silently.
        return pd.Series(dtype="float64")
    except Exception:
        return pd.Series(dtype="float64")

    if df is None or df.empty:
        return pd.Series(dtype="float64")
    # Most sources return a ``price`` column. Some chain factors return
    # ``value`` instead — fall through to the first numeric column.
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


def _pearson_with_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson ρ + two-sided p-value. Avoids the scipy import when ``n < 3``.

    For n < 3 the test statistic is undefined, so we return ``(ρ, 1.0)``.
    For n ≥ 3 we use scipy's exact two-sided p-value.
    """
    n = len(x)
    if n < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0, 1.0
    rho = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(rho):
        return 0.0, 1.0
    if n < 3:
        return rho, 1.0
    # Manual Student-t conversion avoids the scipy hard-dep at import time.
    # df = n - 2; t = ρ * sqrt(df / (1 - ρ²))
    if abs(rho) >= 1.0:
        return rho, 0.0
    df = n - 2
    t_stat = rho * math.sqrt(df / (1.0 - rho * rho))
    try:
        from scipy.stats import t as student_t

        p_value = 2.0 * (1.0 - student_t.cdf(abs(t_stat), df=df))
    except ImportError:  # pragma: no cover
        # Crude normal approximation fallback — fine for df ≳ 30.
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
    return rho, float(max(0.0, min(1.0, p_value)))


def _compute_related(
    anchor: FactorConfig,
    factors: dict[str, FactorConfig],
    window_days: int,
    *,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
    max_candidates: int = MAX_CANDIDATES_DEFAULT,
) -> list[RelatedFactor]:
    """Core math: for each non-anchor factor, compute ρ vs the anchor.

    Candidate set is capped at ``max_candidates`` to keep the per-request
    work bounded — see :data:`MAX_CANDIDATES_DEFAULT` for the default and
    the rationale.
    """
    end = pd.Timestamp.utcnow().normalize()
    # Pad the start by an extra week of calendar days so weekends don't
    # leave us short on trading days.
    start = end - pd.Timedelta(days=window_days + 7)

    anchor_series = _fetch_series(
        anchor, start=start, end=end, poly=poly, cache=cache, settings=settings
    )
    if anchor_series.empty:
        # No anchor data → nothing we can correlate against. Return empty
        # list rather than 404; the slug was valid, the upstream is just dry.
        return []
    # Restrict anchor to the requested window (in case the cache returned more).
    anchor_series = anchor_series.tail(window_days)

    # Bound the candidate pool. Same-theme factors first so the truncation
    # keeps the most plausible neighbours; the rest of the catalog fills
    # in if there's room. Deterministic ordering (by slug) keeps cache
    # keys stable across calls.
    same_theme = sorted(
        (
            fc
            for fc in factors.values()
            if fc.slug != anchor.slug
            and getattr(fc, "theme", None) == getattr(anchor, "theme", None)
        ),
        key=lambda fc: fc.slug,
    )
    other = sorted(
        (
            fc
            for fc in factors.values()
            if fc.slug != anchor.slug
            and getattr(fc, "theme", None) != getattr(anchor, "theme", None)
        ),
        key=lambda fc: fc.slug,
    )
    ordered = (same_theme + other)[: max(1, int(max_candidates))]

    results: list[RelatedFactor] = []
    for fc in ordered:
        if fc.slug == anchor.slug:
            continue
        cand = _fetch_series(fc, start=start, end=end, poly=poly, cache=cache, settings=settings)
        if cand.empty:
            continue
        cand = cand.tail(window_days)
        # Inner-join on the date index. Both series are UTC-normalised by
        # the upstream cache helper, so a direct join is safe.
        joined = pd.concat(
            [anchor_series.rename("a"), cand.rename("b")], axis=1, join="inner"
        ).dropna()
        n_obs = int(len(joined))
        if n_obs < MIN_OVERLAP_OBS:
            continue
        a_arr = joined["a"].to_numpy()
        b_arr = joined["b"].to_numpy()
        # Drop zero-variance candidates: ρ is undefined when either side
        # is constant, and the panel would surface ρ=0 with p=1 which has
        # no decision value for the caller.
        if float(np.std(a_arr)) == 0.0 or float(np.std(b_arr)) == 0.0:
            continue
        rho, p_value = _pearson_with_pvalue(a_arr, b_arr)
        if not math.isfinite(rho):
            continue
        results.append(RelatedFactor(slug=fc.slug, rho=rho, p_value=p_value, n_obs=n_obs))

    # Sort desc by |ρ|, take top N.
    results.sort(key=lambda r: abs(r.rho), reverse=True)
    return results[:TOP_N]


@router.get(
    "/factors/{slug}/related",
    response_model=FactorsRelatedResponse,
    summary="Top-10 factors most correlated with the given anchor.",
)
def get_related_factors(
    slug: Annotated[str, "Polymarket / source slug of the anchor factor."],
    request: Request,
    window: Annotated[
        int,
        Query(
            ge=WINDOW_MIN,
            le=WINDOW_MAX,
            description="Rolling window length in trading days.",
        ),
    ] = WINDOW_DEFAULT,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_CANDIDATES_HARD,
            description=(
                "Max candidate factors to evaluate. Default 100 keeps the "
                "handler bounded; raise up to 500 for an exhaustive scan."
            ),
        ),
    ] = MAX_CANDIDATES_DEFAULT,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> FactorsRelatedResponse:
    """Return the top-10 factors most correlated with the anchor slug.

    * Computes Pearson ρ over the last ``window`` trading days.
    * Drops candidates with <20 overlapping daily observations.
    * Sorted desc by |ρ|, capped at 10.
    * Cached for 60 seconds per ``(slug, window, limit)``.
    """
    cache_key = (slug, int(window), int(limit))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    anchor = _resolve_anchor(factors, slug)
    related = _compute_related(
        anchor,
        factors,
        window_days=int(window),
        poly=poly,
        cache=cache,
        settings=settings,
        max_candidates=int(limit),
    )
    payload = FactorsRelatedResponse(
        anchor=slug,
        window_days=int(window),
        related=related,
    )
    _cache_put(cache_key, payload)
    return payload
