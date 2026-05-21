"""A3 — Polymarket-implied σ vs external benchmark gap signal.

Composes :mod:`pfm.vol.pm_iv_extractor` (A1) and :mod:`pfm.vol.vol_benchmarks`
(A2) into a single end-to-end snapshot. For a registered asset we:

1. Discover the ladder family (``discover_ladder_family``).
2. Fit a Polymarket-implied σ via the call-curve second derivative + lognormal recovery
   (``fit_implied_sigma``).
3. Pull every external annualized-σ benchmark we can produce for that asset
   (``get_benchmark_for_asset``).
4. Compute per-benchmark gaps in vol points (``σ_pm - σ_bench`` × 100).
5. Pick the *primary* benchmark (VIX/OVX/GVZ/DVOL by asset) and emit a
   directional signal (``pm_richer`` / ``benchmark_richer`` / ``flat``) plus
   a strength bucket (``weak`` / ``moderate`` / ``strong``).

Returns a :class:`PMIVGapSnapshot` that is also used as the response model on
``GET /vol/pm-iv/gap/{asset}``. When the asset is unknown (or every
benchmark fetcher fails and σ_pm cannot be produced) we return a sentinel
``signal="no_data"`` snapshot rather than raising — the router translates
that into a 404 only when the asset itself is not in ``LADDER_REGISTRY``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from pfm.vol.pm_iv_extractor import (
    LADDER_REGISTRY,
    discover_ladder_family,
    fit_implied_sigma,
)
from pfm.vol.vol_benchmarks import VolBenchmark, get_benchmark_for_asset

logger = logging.getLogger(__name__)


SignalLiteral = Literal["pm_richer", "benchmark_richer", "flat", "no_data"]
StrengthLiteral = Literal["weak", "moderate", "strong"]


# ---------------------------------------------------------------------------
# Constants — primary benchmark selection per asset
# ---------------------------------------------------------------------------


# Decision thresholds in *vol points* (e.g. +5.0 = σ_pm 5pp richer than VIX).
_FLAT_BAND_PP = 2.0
_MODERATE_PP = 3.0
_STRONG_PP = 5.0


# Each asset maps to an ordered list of preferred primary-benchmark keys.
# The first key that's present in the fetched benchmarks dict wins; any
# fall-through emits a warning so callers can see which fallback was used.
_PRIMARY_PREFERENCE: dict[str, list[str]] = {
    "SPX": ["vix"],
    "WTI": ["ovx"],
    "GOLD": ["gvz"],
    "BTC": ["dvol", "realized_30d"],
    "ETH": ["dvol", "realized_30d"],
}


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class PMIVGapSnapshot(BaseModel):
    """End-to-end Polymarket-vs-benchmark σ snapshot for one asset."""

    asset: str
    maturity_utc: datetime
    time_to_maturity_years: float
    sigma_pm: float = Field(..., ge=0)
    sigma_pm_method: str
    sigma_pm_ci_low: float | None
    sigma_pm_ci_high: float | None
    sigma_pm_n_strikes: int
    benchmarks: dict[str, float]
    gaps: dict[str, float]
    primary_benchmark: str | None
    primary_gap_pct_pts: float | None
    signal: SignalLiteral
    signal_strength: StrengthLiteral | None
    fitted_mean: float
    fitted_std: float
    implied_skew: float
    implied_kurtosis: float
    warnings: list[str]
    as_of_utc: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


def _empty_snapshot(asset: str, *, reason: str) -> PMIVGapSnapshot:
    """Build a ``signal='no_data'`` sentinel snapshot for an unknown/empty asset."""
    return PMIVGapSnapshot(
        asset=asset.upper(),
        maturity_utc=_now_utc(),
        time_to_maturity_years=0.0,
        sigma_pm=0.0,
        sigma_pm_method="lognormal_fit",
        sigma_pm_ci_low=None,
        sigma_pm_ci_high=None,
        sigma_pm_n_strikes=0,
        benchmarks={},
        gaps={},
        primary_benchmark=None,
        primary_gap_pct_pts=None,
        signal="no_data",
        signal_strength=None,
        fitted_mean=0.0,
        fitted_std=0.0,
        implied_skew=0.0,
        implied_kurtosis=0.0,
        warnings=[reason],
        as_of_utc=_now_utc(),
    )


def _classify_strength(gap_pp: float) -> StrengthLiteral:
    """Vol-pp gap → bucketed strength tag (matches docstring thresholds)."""
    mag = abs(gap_pp)
    if mag > _STRONG_PP:
        return "strong"
    if mag > _MODERATE_PP:
        return "moderate"
    return "weak"


def _classify_signal(gap_pp: float) -> SignalLiteral:
    """Vol-pp gap → directional signal with a ±2pp dead band around zero."""
    if gap_pp > _FLAT_BAND_PP:
        return "pm_richer"
    if gap_pp < -_FLAT_BAND_PP:
        return "benchmark_richer"
    return "flat"


def _pick_primary(
    asset: str,
    benchmarks: dict[str, VolBenchmark],
    warnings: list[str],
) -> str | None:
    """Walk the preference list and return the first available key.

    Mutates ``warnings`` in place with a ``primary_benchmark_fallback`` tag
    when the top preference is missing but a fallback succeeds.
    """
    prefs = _PRIMARY_PREFERENCE.get(asset.upper(), [])
    for i, key in enumerate(prefs):
        if key in benchmarks:
            if i > 0:
                warnings.append(f"primary_benchmark_fallback:{key}_used_instead_of_{prefs[0]}")
            return key
    if prefs:
        warnings.append("primary_benchmark_unavailable")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_gap_snapshot(
    asset: str,
    *,
    polymarket_client: Any,
    http: httpx.Client | None = None,
    maturity_filter: str | None = None,
) -> PMIVGapSnapshot:
    """Compose A1 + A2 into a single Polymarket-vs-benchmark σ gap snapshot.

    Pipeline:
      1. ``discover_ladder_family`` for the asset → ``LadderFamily`` or None.
         A missing family ⇒ ``signal="no_data"`` (no benchmark fetches).
      2. ``fit_implied_sigma(family)`` → ``PMIVResult`` (σ_pm + diagnostics).
      3. ``tenor_days = round(time_to_maturity_years * 365)``.
      4. ``get_benchmark_for_asset(asset, tenor_days, http=http)`` →
         ``dict[str, VolBenchmark]``.
      5. Gap per benchmark = (σ_pm - σ_bench) × 100 (in vol points).
      6. Primary-benchmark preference: SPX→vix, WTI→ovx, GOLD→gvz,
         BTC→dvol (fallback realized_30d), ETH→dvol (fallback realized_30d).
      7. Signal: pm_richer if primary_gap > +2pp, benchmark_richer if < -2pp,
         else flat. Strength: |gap|>5 strong, |gap|>3 moderate, else weak.
      8. Warnings merge ``PMIVResult.warnings`` + per-benchmark
         ``stale_warning`` flags + any primary-fallback notes.
    """
    asset_upper = asset.upper().strip()
    if asset_upper not in LADDER_REGISTRY:
        return _empty_snapshot(asset_upper, reason="unknown_asset")

    family = discover_ladder_family(
        asset_upper,
        polymarket_client=polymarket_client,
        maturity_filter=maturity_filter,
    )
    if family is None:
        return _empty_snapshot(asset_upper, reason="no_ladder_family")

    try:
        pm_result = fit_implied_sigma(family)
    except ValueError as exc:
        snap = _empty_snapshot(asset_upper, reason=f"sigma_fit_failed:{exc}")
        # Surface the maturity even on failure so downstream UIs can show
        # the contract under inspection.
        snap = snap.model_copy(update={"maturity_utc": family.maturity_utc})
        return snap

    tenor_days = max(1, round(pm_result.time_to_maturity_years * 365))

    benchmarks_raw = get_benchmark_for_asset(asset_upper, tenor_days, http=http)

    benchmarks: dict[str, float] = {}
    gaps: dict[str, float] = {}
    warnings: list[str] = list(pm_result.warnings)

    for key, bench in benchmarks_raw.items():
        benchmarks[key] = round(float(bench.sigma_annual), 6)
        gap_pp = (pm_result.sigma_annual - bench.sigma_annual) * 100.0
        gaps[key] = round(gap_pp, 4)
        if bench.stale_warning:
            warnings.append(f"benchmark_stale:{key}")

    primary_key = _pick_primary(asset_upper, benchmarks_raw, warnings)
    primary_gap_pp: float | None = None
    signal: SignalLiteral
    strength: StrengthLiteral | None

    if primary_key is None:
        signal = "no_data"
        strength = None
    else:
        primary_gap_pp = gaps[primary_key]
        signal = _classify_signal(primary_gap_pp)
        strength = _classify_strength(primary_gap_pp)

    return PMIVGapSnapshot(
        asset=asset_upper,
        maturity_utc=pm_result.maturity_utc,
        time_to_maturity_years=pm_result.time_to_maturity_years,
        sigma_pm=pm_result.sigma_annual,
        sigma_pm_method=pm_result.sigma_method,
        sigma_pm_ci_low=pm_result.sigma_ci_low,
        sigma_pm_ci_high=pm_result.sigma_ci_high,
        sigma_pm_n_strikes=pm_result.n_strikes,
        benchmarks=benchmarks,
        gaps=gaps,
        primary_benchmark=primary_key,
        primary_gap_pct_pts=round(primary_gap_pp, 4) if primary_gap_pp is not None else None,
        signal=signal,
        signal_strength=strength,
        fitted_mean=pm_result.fitted_mean,
        fitted_std=pm_result.fitted_std,
        implied_skew=pm_result.implied_skew,
        implied_kurtosis=pm_result.implied_kurtosis,
        warnings=warnings,
        as_of_utc=_now_utc(),
    )


__all__ = [
    "PMIVGapSnapshot",
    "compute_gap_snapshot",
]
