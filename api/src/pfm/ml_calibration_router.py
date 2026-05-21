"""ML Hub — calibration & favorite–longshot bias of prediction-market prices.

This module asks the most prediction-market-native ML question there is, and
answers it *honestly* because the label is ground truth: **are market prices
calibrated probabilities?** If a basket of contracts trades at 70¢, do they
resolve YES 70% of the time? And where they don't, *which way* does the bias
run — are longshots (cheap contracts) systematically overpriced and favourites
(expensive contracts) underpriced (the classic *favourite–longshot bias*)?

Why this is honest (no overfit risk)
-------------------------------------
There is **no forward prediction here on unlabelled data**. Every (price,
outcome) pair comes from a market that has already *resolved* — the outcome
``y ∈ {0,1}`` is the realised truth, not a model guess. We measure how well the
market's *own* prices matched reality. The isotonic recalibration is fit on the
same realised outcomes and is purely descriptive of historical mispricing; the
gap between raw price and the calibrated probability is a *candidate* signal the
user can inspect, not a backtested claim.

Data
----
Resolved markets come from :mod:`pfm.archive.polymarket_archive`. We take a
**representative pre-resolution price** for each market (the YES price a fixed
horizon of trading days before it closed, so we are not reading the degenerate
~0/1 settlement print) paired with the binary YES/NO outcome. Markets that
resolved ``AMBIGUOUS``/``PENDING`` are dropped. When a fresh container cannot
yield ``min_samples`` usable pairs the endpoint returns ``degraded_mode=true``
gracefully rather than emitting a meaningless curve.

Routing
-------
Owns its own :class:`fastapi.APIRouter`. Wire in ``main.py`` with::

    from pfm.ml_calibration_router import router as ml_calibration_router
    app.include_router(ml_calibration_router)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field
from sklearn.isotonic import IsotonicRegression

# Imported into *this* module's namespace on purpose: tests monkeypatch these
# names here (not on the source modules) so no network IO ever happens.
from pfm.archive.polymarket_archive import (
    fetch_archive_market_detail,
    fetch_resolved_markets,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml-hub"])

CAL_TTL_SECONDS: int = 1800  # resolved markets are immutable; 30 min is comfy
# Representative price = YES price this many trading days before the close, so
# we capture genuine market belief, not the degenerate settlement print.
PRE_RESOLUTION_HORIZON_DAYS: int = 3
# How many resolved markets to scan when gathering the dataset. Each costs one
# cached archive-detail call; bound it so a misuse can't pin a worker.
MAX_MARKETS_SCANNED: int = 600
# Curve grid resolution for the calibrated curve sent to the frontend.
_CURVE_GRID_POINTS: int = 41
# Default lookback window for "resolved recently" markets.
_LOOKBACK_DAYS: int = 365


# --- response schemas -------------------------------------------------------


class CalBin(BaseModel):
    """One bin of the reliability diagram (calibration curve)."""

    price_mid: float = Field(..., description="Bin midpoint on the price axis (0..1).")
    mean_pred: float = Field(..., description="Mean market price of samples in this bin.")
    empirical: float = Field(..., description="Empirical YES-rate of samples in this bin.")
    count: int = Field(..., description="Number of resolved markets in this bin.")
    ci_low: float = Field(..., description="Wilson 95% lower bound on the empirical YES-rate.")
    ci_high: float = Field(..., description="Wilson 95% upper bound on the empirical YES-rate.")


class CurvePoint(BaseModel):
    """One point of the fitted isotonic recalibration curve."""

    x: float = Field(..., description="Raw market price (grid point, 0..1).")
    y: float = Field(..., description="Isotonic-calibrated probability for that price.")


class CalibrationResponse(BaseModel):
    n_samples: int
    n_bins: int
    category: str | None
    # Brier score of raw prices vs realised outcomes (lower is better; 0=perfect).
    brier: float | None
    # Expected calibration error: count-weighted mean |empirical - mean_pred|.
    ece: float | None
    # Signed avg (empirical - price) gap in the longshot region (price<0.2).
    # Negative ⇒ longshots resolve YES *less* often than priced ⇒ overpriced.
    longshot_bias: float | None
    # Signed avg (empirical - price) gap in the favourite region (price>0.8).
    # Positive ⇒ favourites resolve YES *more* often than priced ⇒ underpriced.
    favorite_bias: float | None
    bins: list[CalBin]
    calibrated_curve: list[CurvePoint]
    degraded_mode: bool = False
    reason: str | None = None


# --- dataset gathering ------------------------------------------------------


def _representative_price(history: list[list[object]]) -> float | None:
    """Pick a YES price ``PRE_RESOLUTION_HORIZON_DAYS`` before the last point.

    ``history`` is the archive-detail ``[[date_iso, price, volume], ...]`` list,
    chronologically sorted. We step back a fixed horizon from the end so the
    price reflects genuine belief rather than the ~0/1 settlement print. Falls
    back to the earliest available point when the series is shorter than the
    horizon.

    Returns:
        A finite price in ``(0, 1)`` or ``None`` if no usable point exists.
    """
    if not history:
        return None
    n = len(history)
    idx = max(0, n - 1 - PRE_RESOLUTION_HORIZON_DAYS)
    for j in range(idx, -1, -1):
        try:
            p = float(history[j][1])  # type: ignore[index]
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(p) and 0.0 < p < 1.0:
            return p
    return None


def _gather_samples(
    *,
    category: str | None,
    max_markets: int,
) -> tuple[list[float], list[int]]:
    """Collect ``(price, outcome)`` pairs over recently-resolved markets.

    Walks the Polymarket archive for markets resolved in the last
    :data:`_LOOKBACK_DAYS` days, keeps those with a clean ``YES``/``NO``
    resolution, and pairs each with its representative pre-resolution YES price.

    This is the single network seam: tests monkeypatch
    :func:`fetch_resolved_markets` / :func:`fetch_archive_market_detail` on this
    module's namespace, so ``pytest`` never touches the network.

    Args:
        category: Optional theme/category filter forwarded to the archive.
        max_markets: Hard cap on markets scanned.

    Returns:
        ``(prices, outcomes)`` aligned lists; outcomes are ``1`` for YES,
        ``0`` for NO.
    """
    end = datetime.now(UTC).date()
    start = end - timedelta(days=_LOOKBACK_DAYS)
    try:
        rows = fetch_resolved_markets(start, end, theme=category, limit=int(max_markets), offset=0)
    except Exception as exc:  # archive/network errors degrade gracefully
        logger.warning("calibration: resolved-market fetch failed: %s", exc)
        return [], []

    prices: list[float] = []
    outcomes: list[int] = []
    for row in rows[:max_markets]:
        resolution = str(row.get("resolution") or "").upper()
        if resolution not in {"YES", "NO"}:
            continue
        slug = row.get("slug")
        if not slug:
            continue
        try:
            detail = fetch_archive_market_detail(str(slug))
        except Exception as exc:  # skip individual broken markets
            logger.debug("calibration: detail fetch failed for %s: %s", slug, exc)
            continue
        price = _representative_price(detail.get("history") or [])
        if price is None:
            continue
        prices.append(price)
        outcomes.append(1 if resolution == "YES" else 0)
    return prices, outcomes


# --- calibration maths ------------------------------------------------------


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k`` successes of ``n``.

    The Wilson interval behaves well for small ``n`` and extreme proportions
    (where the normal-approximation interval breaks down or escapes ``[0, 1]``),
    so a sparse reliability bin (e.g. ``n=2``) yields an honestly wide band
    rather than a falsely tight one.

    Args:
        k: Number of successes (YES resolutions) in the bin.
        n: Number of samples in the bin.
        z: Standard-normal quantile (``1.96`` for a 95% interval).

    Returns:
        ``(low, high)`` bounds, each clipped to ``[0, 1]``. Returns
        ``(0.0, 1.0)`` for an empty bin (``n <= 0``).
    """
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return float(low), float(high)


def _reliability_bins(prices: np.ndarray, outcomes: np.ndarray, n_bins: int) -> list[CalBin]:
    """Bin prices into ``n_bins`` equal-width buckets over ``[0, 1]``.

    Per non-empty bin: midpoint, mean predicted price, empirical YES-rate,
    count, and a Wilson 95% interval on the empirical YES-rate so sparse bins
    are not over-read. Empty bins are omitted so the frontend draws only
    supported points.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # ``digitize`` with the right edge inclusive so price==1.0 lands in the last bin.
    idx = np.clip(np.digitize(prices, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[CalBin] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        k = int(outcomes[mask].sum())
        ci_low, ci_high = _wilson_interval(k, count)
        bins.append(
            CalBin(
                price_mid=round(float((edges[b] + edges[b + 1]) / 2.0), 4),
                mean_pred=round(float(prices[mask].mean()), 4),
                empirical=round(float(outcomes[mask].mean()), 4),
                count=count,
                ci_low=round(ci_low, 4),
                ci_high=round(ci_high, 4),
            )
        )
    return bins


def _ece_from_bins(bins: list[CalBin], n_total: int) -> float | None:
    """Count-weighted expected calibration error over reliability bins."""
    if n_total <= 0 or not bins:
        return None
    weighted = sum(b.count * abs(b.empirical - b.mean_pred) for b in bins)
    return round(weighted / n_total, 4)


def _fit_isotonic_curve(
    prices: np.ndarray, outcomes: np.ndarray
) -> tuple[IsotonicRegression, list[CurvePoint]]:
    """Fit price→P(YES) isotonic regression and sample it on a fixed grid.

    Isotonic regression gives a monotone non-decreasing recalibration map,
    which is exactly the shape a favourite–longshot correction should take: it
    can pull overpriced longshots down and push underpriced favourites up while
    never inverting the price ordering.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(prices, outcomes)
    grid = np.linspace(0.0, 1.0, _CURVE_GRID_POINTS)
    calibrated = iso.predict(grid)
    curve = [
        CurvePoint(x=round(float(gx), 4), y=round(float(gy), 4))
        for gx, gy in zip(grid, calibrated, strict=True)
    ]
    return iso, curve


def _region_bias(prices: np.ndarray, outcomes: np.ndarray, mask: np.ndarray) -> float | None:
    """Signed mean ``(empirical - price)`` gap over a price region.

    Negative ⇒ region resolves YES less often than priced (overpriced);
    positive ⇒ resolves YES more often than priced (underpriced).
    """
    if not mask.any():
        return None
    gap = float((outcomes[mask] - prices[mask]).mean())
    return round(gap, 4)


# --- main endpoint ----------------------------------------------------------


@router.get(
    "/calibration",
    response_model=None,
    summary="Calibration & favourite–longshot bias of resolved-market prices.",
)
def calibration(
    category: Annotated[
        str | None,
        Query(description="Filter resolved markets by theme/category (e.g. 'politics')."),
    ] = None,
    n_bins: Annotated[
        int,
        Query(ge=2, le=50, description="Reliability-diagram bins over [0,1] (deciles=10)."),
    ] = 10,
    min_samples: Annotated[
        int,
        Query(ge=1, description="Minimum resolved (price, outcome) pairs to fit a curve."),
    ] = 40,
    request: Request = None,  # type: ignore[assignment]
) -> CalibrationResponse:
    """Measure whether resolved-market prices are calibrated probabilities.

    Pipeline: gather ``(representative pre-resolution YES price, binary
    outcome)`` over recently-resolved Polymarket markets → bin into a reliability
    diagram → fit a monotone isotonic recalibration → summarise the
    favourite–longshot bias in the cheap (``price<0.2``) and expensive
    (``price>0.8``) regions.

    Cached for :data:`CAL_TTL_SECONDS` via the shared L1/L2 TERMINAL_CACHE keyed
    on ``(category, n_bins, min_samples)``. Returns ``degraded_mode=true`` (empty
    curve) when fewer than ``min_samples`` usable pairs are available, mirroring
    the empty-state contract of ``/ml/factor-map``.
    """
    from pfm import terminal as _term_mod  # circular-safe lazy import

    cache_key = f"ml_calibration::{category or '*'}::{n_bins}::{min_samples}"
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return CalibrationResponse.model_validate(cached)

    prices_list, outcomes_list = _gather_samples(category=category, max_markets=MAX_MARKETS_SCANNED)
    n = len(prices_list)
    if n < min_samples:
        return CalibrationResponse(
            n_samples=n,
            n_bins=n_bins,
            category=category,
            brier=None,
            ece=None,
            longshot_bias=None,
            favorite_bias=None,
            bins=[],
            calibrated_curve=[],
            degraded_mode=True,
            reason=(
                f"Only {n} resolved (price, outcome) pairs reachable from the "
                f"archive (need {min_samples}). The archive cache may be cold in a "
                "fresh container, or no markets matched the category filter."
            ),
        )

    prices = np.asarray(prices_list, dtype=float)
    outcomes = np.asarray(outcomes_list, dtype=float)

    bins = _reliability_bins(prices, outcomes, n_bins)
    brier = round(float(np.mean((prices - outcomes) ** 2)), 5)
    ece = _ece_from_bins(bins, n)
    longshot_bias = _region_bias(prices, outcomes, prices < 0.2)
    favorite_bias = _region_bias(prices, outcomes, prices > 0.8)
    _iso, curve = _fit_isotonic_curve(prices, outcomes)

    resp = CalibrationResponse(
        n_samples=n,
        n_bins=n_bins,
        category=category,
        brier=brier,
        ece=ece,
        longshot_bias=longshot_bias,
        favorite_bias=favorite_bias,
        bins=bins,
        calibrated_curve=curve,
    )
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), CAL_TTL_SECONDS)
    return resp


__all__ = [
    "CalBin",
    "CalibrationResponse",
    "CurvePoint",
    "_wilson_interval",
    "calibration",
    "router",
]
