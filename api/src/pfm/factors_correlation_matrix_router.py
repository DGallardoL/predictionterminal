"""``GET /factors/{slug}/correlation-matrix`` — top-N peer correlation matrix.

Task W13-14. Given an *anchor* factor slug, find the top-N most correlated
peer factors over a rolling daily window and return a full ``(N+1) × (N+1)``
Pearson correlation matrix (anchor first, then peers in descending |ρ|
order).

The expensive bit — computing pair-correlations across ≤1228 factors — is
delegated to the W12-23 memoization layer
(:mod:`pfm.terminal.correlations_cache`). The router only ever asks for the
``(anchor + peers)`` block, so even on a cold cache the full-catalog cost
isn't paid here. Two callers requesting the same window for the same peer
basket get the *same* read-only :class:`CorrMatrix` instance back (the LRU
key is order-insensitive via ``frozenset``).

Why a separate router from ``factors_related_router``?
------------------------------------------------------
``/factors/{slug}/related`` already exists and returns a ranked *list* of
peers (ρ + p-value + n_obs). That endpoint is fine for the side-panel UI,
but a heatmap or block-diagonal cluster view needs the full *square*
correlation matrix — including peer-vs-peer entries, not just peer-vs-
anchor. Reusing the related-router by hand-stitching N+1 separate calls
would be O(N²) round-trips; the cached matrix path is one shot.

File location
-------------
Top-level ``pfm/factors_correlation_matrix_router.py`` (parallel to
``factors_router.py`` and ``factors_related_router.py``) for the same
reason documented in the related-router header: ``pfm.factors`` is a
module, so creating a ``pfm/factors/`` package would shadow it and break
~40 import sites.

Integration
-----------
This router is NOT wired into ``main.py`` automatically — the
``main.py:routes`` section is held by another active claim. The next
session that holds it should add::

    from pfm.factors_correlation_matrix_router import (
        router as _factors_corr_matrix_router,
    )
    app.include_router(_factors_corr_matrix_router)

The router declares no prefix; the path ``/factors/{slug}/correlation-matrix``
is set on the route decorator itself, matching the convention used by
``factors_related_router``.

Response shape
--------------
::

    {
      "anchor": "anchor-slug",
      "window_days": 30,
      "peers": ["peer-1-slug", "peer-2-slug", ...],
      "matrix": [
        [1.00, 0.85, 0.62, ...],   # anchor row
        [0.85, 1.00, 0.71, ...],   # peer-1 row
        ...
      ]
    }

Row/column ``i`` corresponds to the ``i``-th entry of
``[anchor] + peers``. Each entry is a Pearson ρ in ``[-1, 1]``;
non-finite values are coerced to ``0.0`` so the response is always
strict-JSON safe.
"""

from __future__ import annotations

import logging
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
from pfm.terminal.correlations_cache import (
    CorrMatrix,
    get_or_compute_corr,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["factors"])


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Wall-clock used for the response-level TTL cache. Tests monkeypatch this
# to drive expiry deterministically (same pattern as factors_related_router).
_PERF_COUNTER: Callable[[], float] = time.perf_counter

# Response cache TTL in seconds. Output is a function of the daily factor
# universe, so a minute of staleness is fine on an interactive panel.
_CACHE_TTL_S: float = 60.0

# Window bounds. ``window_days`` is the rolling lookback used both for
# selecting peers (via pair correlations against the anchor) and for the
# returned matrix. 7 is the smallest window where Pearson ρ is meaningful
# at the default 20-obs floor; 365 keeps the worst-case fetch under a few
# seconds on the full catalog.
WINDOW_MIN: int = 7
WINDOW_MAX: int = 365
WINDOW_DEFAULT: int = 30

# ``top_n`` bounds. 5 is the smallest peer set that yields a usable
# heatmap; 50 caps the matrix at ~51² ≈ 2600 cells — well within a JSON
# payload most browsers handle without re-rendering hiccups.
TOP_N_MIN: int = 1
TOP_N_MAX: int = 50
TOP_N_DEFAULT: int = 20

# Peer must share at least this many daily observations with the anchor
# to be considered. Same threshold the related-router uses.
MIN_OVERLAP_OBS: int = 20


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class CorrelationMatrixResponse(BaseModel):
    """JSON envelope for ``GET /factors/{slug}/correlation-matrix``."""

    anchor: str = Field(..., description="Anchor factor slug echoed back.")
    window_days: int = Field(
        ..., ge=WINDOW_MIN, le=WINDOW_MAX, description="Rolling window length in days."
    )
    peers: list[str] = Field(
        ...,
        description=(
            "Slugs of the top-N most correlated peers, sorted desc by |ρ| against "
            "the anchor. Empty when no peer has ≥20 overlapping daily observations."
        ),
    )
    matrix: list[list[float]] = Field(
        ...,
        description=(
            "Square correlation matrix indexed by ``[anchor] + peers``. Row i, "
            "column j is the Pearson ρ between series i and series j. Always "
            "symmetric with 1.0 on the diagonal."
        ),
    )


# ---------------------------------------------------------------------------
# Response-level TTL cache  (separate from the W12-23 matrix cache below)
# ---------------------------------------------------------------------------
# The W12-23 ``correlations_cache`` memoizes the *matrix* by ``(slug-set,
# window)``. We additionally memoize the *response* (anchor + peers + matrix)
# by ``(anchor, window, top_n)`` because peer-selection itself involves N
# pair correlations that we'd rather not redo on every keypress in the UI.


class _Entry:
    __slots__ = ("expires_at", "payload")

    def __init__(self, payload: CorrelationMatrixResponse, expires_at: float) -> None:
        self.payload = payload
        self.expires_at = expires_at


_RESPONSE_CACHE: dict[tuple[str, int, int], _Entry] = {}
_RESPONSE_CACHE_LOCK = threading.Lock()


def _response_cache_get(
    key: tuple[str, int, int],
) -> CorrelationMatrixResponse | None:
    now = _PERF_COUNTER()
    with _RESPONSE_CACHE_LOCK:
        entry = _RESPONSE_CACHE.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            _RESPONSE_CACHE.pop(key, None)
            return None
        return entry.payload


def _response_cache_put(key: tuple[str, int, int], payload: CorrelationMatrixResponse) -> None:
    expires_at = _PERF_COUNTER() + _CACHE_TTL_S
    with _RESPONSE_CACHE_LOCK:
        _RESPONSE_CACHE[key] = _Entry(payload, expires_at)


def _response_cache_clear() -> None:
    """Drop every cached response — used by tests to force a cold path."""
    with _RESPONSE_CACHE_LOCK:
        _RESPONSE_CACHE.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_anchor(factors: dict[str, FactorConfig], slug: str) -> FactorConfig:
    """Look up the anchor factor by slug. Raises 404 when unknown."""
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
    """Pull a single factor's daily price series via the shared cache helper.

    Mirrors the helper in ``factors_related_router`` so test monkeypatches
    against ``pfm.regression_core._cached_factor_history`` work for both
    routers without duplication of fixture wiring.
    """
    try:
        from pfm.regression_core import _cached_factor_history
    except ImportError:  # pragma: no cover - only triggers in stripped builds
        return pd.Series(dtype="float64")

    try:
        df = _cached_factor_history(fc, start, end, poly, cache, settings)
    except HTTPException:
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


def _safe_float(x: float) -> float:
    """Coerce a float for JSON output. NaN/inf collapse to 0.0."""
    if not math.isfinite(x):
        return 0.0
    # Clip into [-1, 1] to absorb floating-point overshoot at the boundaries.
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return float(x)


def _rank_peers_by_abs_rho(
    anchor_series: pd.Series,
    candidates: list[tuple[str, pd.Series]],
    *,
    top_n: int,
) -> list[str]:
    """Rank candidate slugs by |Pearson ρ| against the anchor (desc).

    Drops candidates with <``MIN_OVERLAP_OBS`` overlapping daily obs or
    zero variance on either side. Returns the top-``top_n`` slugs in
    descending |ρ| order.
    """
    if anchor_series.empty:
        return []
    a_std = float(anchor_series.std(ddof=0))
    if a_std == 0.0:
        return []

    ranked: list[tuple[str, float]] = []
    for slug, cand in candidates:
        if cand.empty:
            continue
        joined = pd.concat(
            [anchor_series.rename("a"), cand.rename("b")], axis=1, join="inner"
        ).dropna()
        if len(joined) < MIN_OVERLAP_OBS:
            continue
        a = joined["a"].to_numpy()
        b = joined["b"].to_numpy()
        if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        if not math.isfinite(rho):
            continue
        ranked.append((slug, rho))

    ranked.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [slug for slug, _ in ranked[:top_n]]


def _matrix_for_slugs(
    slug_series: dict[str, pd.Series],
    ordered_slugs: list[str],
) -> np.ndarray:
    """Compute a Pearson correlation matrix for the given series.

    Parameters
    ----------
    slug_series:
        ``{slug: pd.Series}`` covering at least every slug in ``ordered_slugs``.
    ordered_slugs:
        Output row/column order. Length ``n``.

    Returns
    -------
    np.ndarray
        ``(n, n)`` symmetric matrix with ``1.0`` on the diagonal. Cells
        where the two series share fewer than 2 obs (or where either has
        zero variance on the overlap) collapse to ``0.0``.
    """
    n = len(ordered_slugs)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.array([[1.0]], dtype=float)

    # Align every series onto the union of their indices so the joint dropna
    # per-pair below is cheap (vectorised slicing of an existing frame).
    frame = pd.DataFrame({s: slug_series.get(s, pd.Series(dtype=float)) for s in ordered_slugs})

    arr = np.zeros((n, n), dtype=float)
    for i in range(n):
        arr[i, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            pair = frame.iloc[:, [i, j]].dropna()
            if len(pair) < 2:
                continue
            x = pair.iloc[:, 0].to_numpy()
            y = pair.iloc[:, 1].to_numpy()
            if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            if not math.isfinite(c):
                continue
            arr[i, j] = c
            arr[j, i] = c
    return arr


def _build_matrix_via_cache(
    slug_series: dict[str, pd.Series],
    ordered_slugs: list[str],
    *,
    window_days: int,
) -> CorrMatrix:
    """Build the correlation matrix through the W12-23 memoization layer.

    The closure passed to :func:`get_or_compute_corr` does the heavy work
    using whatever series this request already has in hand; the cache layer
    handles single-flight, read-only sharing, and LRU eviction.
    """
    # Snapshot the series dict so the closure isn't bound to anything that
    # might mutate later in the request lifecycle.
    series_snapshot = {s: slug_series.get(s, pd.Series(dtype=float)) for s in ordered_slugs}

    def _compute(_sorted_slugs: list[str]) -> np.ndarray:
        # The cache passes us the sorted slug list; compute the matrix in
        # *that* order so the cached layout is canonical. Callers reorder
        # below.
        return _matrix_for_slugs(series_snapshot, _sorted_slugs)

    return get_or_compute_corr(
        ordered_slugs,
        _compute,
        window_days=window_days,
    )


def _reorder_matrix(
    cached: CorrMatrix,
    desired_order: list[str],
) -> list[list[float]]:
    """Reorder a cached square matrix from its canonical (sorted) layout to
    the caller's desired order, and emit JSON-safe nested lists."""
    # Index lookup: sorted-slug → row/col index in the cached matrix.
    index_by_slug = {s: i for i, s in enumerate(cached.slugs)}
    len(desired_order)
    out: list[list[float]] = []
    for slug_i in desired_order:
        row: list[float] = []
        i = index_by_slug.get(slug_i)
        for slug_j in desired_order:
            j = index_by_slug.get(slug_j)
            if i is None or j is None:
                # Shouldn't happen — desired_order is built from cached.slugs
                # — but stay defensive so a stale cache can't 5xx the route.
                row.append(0.0)
                continue
            row.append(_safe_float(float(cached.matrix[i, j])))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get(
    "/factors/{slug}/correlation-matrix",
    response_model=CorrelationMatrixResponse,
    summary="Top-N peer correlation matrix for the anchor factor.",
)
def get_correlation_matrix(
    slug: Annotated[str, "Polymarket / source slug of the anchor factor."],
    request: Request,
    top_n: Annotated[
        int,
        Query(
            ge=TOP_N_MIN,
            le=TOP_N_MAX,
            description="How many top-correlated peers to return alongside the anchor.",
        ),
    ] = TOP_N_DEFAULT,
    window_days: Annotated[
        int,
        Query(
            ge=WINDOW_MIN,
            le=WINDOW_MAX,
            description="Rolling window length in days.",
        ),
    ] = WINDOW_DEFAULT,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> CorrelationMatrixResponse:
    """Return the ``(N+1) × (N+1)`` correlation matrix for the anchor + peers.

    Selection
    ---------
    1. Resolve the anchor by slug; 404 if unknown.
    2. For every other factor in the catalogue with a non-empty history
       and ≥20 overlapping daily obs with the anchor, compute Pearson ρ
       vs the anchor.
    3. Sort by |ρ| desc; keep the top ``top_n``.
    4. Build the full ``(1 + len(peers))² `` Pearson correlation matrix
       through the W12-23 cache layer — same matrix shared with the
       Terminal correlation panels.

    Output
    ------
    See :class:`CorrelationMatrixResponse`.

    Caching
    -------
    Two-tier:

    * **W12-23 ``correlations_cache``** — caches the raw ``(slug-set,
      window)`` correlation matrix LRU-style. Shared with every other
      caller that asks for the same peer block in any order.
    * **Response cache** — a 60-second TTL on ``(anchor, window, top_n)``
      so paging through the UI doesn't re-rank peers on each open.
    """
    response_key = (slug, int(window_days), int(top_n))
    cached_payload = _response_cache_get(response_key)
    if cached_payload is not None:
        return cached_payload

    anchor = _resolve_anchor(factors, slug)

    # ── Fetch series for anchor + every candidate, restricted to the window
    end = pd.Timestamp.utcnow().normalize()
    # Pad start to leave slack for weekends / sparse days. 7 calendar days
    # is enough for the worst week-on-week gap our daily sources show.
    start = end - pd.Timedelta(days=int(window_days) + 7)

    anchor_full = _fetch_series(
        anchor, start=start, end=end, poly=poly, cache=cache, settings=settings
    )
    if anchor_full.empty:
        # Valid slug, empty upstream history: same convention as related-router
        # — return an empty matrix rather than 5xx-ing.
        payload = CorrelationMatrixResponse(
            anchor=slug,
            window_days=int(window_days),
            peers=[],
            matrix=[[1.0]],
        )
        _response_cache_put(response_key, payload)
        return payload
    anchor_series = anchor_full.tail(int(window_days))

    # Build the candidate list (anchor excluded). Fetch each in turn — the
    # underlying cache helper deduplicates network calls.
    candidates: list[tuple[str, pd.Series]] = []
    series_by_slug: dict[str, pd.Series] = {slug: anchor_series}
    for fc in factors.values():
        if fc.slug == anchor.slug:
            continue
        cand_full = _fetch_series(
            fc, start=start, end=end, poly=poly, cache=cache, settings=settings
        )
        if cand_full.empty:
            continue
        cand_series = cand_full.tail(int(window_days))
        candidates.append((fc.slug, cand_series))
        series_by_slug[fc.slug] = cand_series

    peer_slugs = _rank_peers_by_abs_rho(anchor_series, candidates, top_n=int(top_n))

    # ── Build the matrix through the W12-23 cache
    ordered = [slug, *peer_slugs]
    cached_matrix: CorrMatrix = _build_matrix_via_cache(
        series_by_slug, ordered, window_days=int(window_days)
    )
    matrix_payload = _reorder_matrix(cached_matrix, ordered)

    payload = CorrelationMatrixResponse(
        anchor=slug,
        window_days=int(window_days),
        peers=peer_slugs,
        matrix=matrix_payload,
    )
    _response_cache_put(response_key, payload)
    return payload


__all__ = [
    "MIN_OVERLAP_OBS",
    "TOP_N_DEFAULT",
    "TOP_N_MAX",
    "TOP_N_MIN",
    "WINDOW_DEFAULT",
    "WINDOW_MAX",
    "WINDOW_MIN",
    "CorrelationMatrixResponse",
    "_response_cache_clear",
    "get_correlation_matrix",
    "router",
]
