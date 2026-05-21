"""ML Hub — Regime Monitor: descriptive market-state classification.

This module answers the question that killed nearly every entry in this
project's anti-alpha graveyard: *what market regime are we in right now, and
when did it change?* It is **purely descriptive** — there is no return forecast
here. We classify the recent cross-section of prediction-market behaviour into a
small number of discrete states (calm / normal / stressed) and flag the current
one, so a trader can ask "is this strategy's pitch a single-regime artefact?"
before deploying.

Why this is honest (no overfit risk)
-------------------------------------
We never predict forward returns. We fit an unsupervised mixture model on
*contemporaneous* state features (cross-sectional dispersion, activity,
co-movement, aggregate volatility) and report the labelled regime path. Every
anti-alpha in this repo died because a backtest lived inside one regime and
flipped sign in the next; surfacing the regime timeline directly addresses that
failure mode without claiming predictive edge.

Pipeline
--------
1.  **History.** Reuse the strat7 daily-probability cache via
    ``_load_cached_history`` → Δlogit returns matrix (``_build_returns_matrix``).
2.  **State features (per UTC day).**
      * ``dispersion`` — std across factors of that day's Δlogit (how spread-out
        the cross-section is);
      * ``activity`` — mean |Δlogit| across factors (how much is moving);
      * ``comovement`` — average pairwise correlation in a trailing ``window``
        of the cross-section (do moves cluster together);
      * ``volatility`` — trailing-``window`` std of the equal-weight factor
        index (aggregate market vol).
3.  **Classifier.** ``sklearn.mixture.GaussianMixture`` with
    ``n_components=n_regimes`` and ``random_state=0`` (deterministic). We
    deliberately avoid ``statsmodels.MarkovRegression`` — it is heavier, noisier
    to test, and overkill for a descriptive state map.
4.  **Stable labels.** Regimes are re-indexed by ascending mean ``volatility`` so
    regime ``0`` is always the calmest and the last is the most stressed, then
    given a human label ("calm" / "normal" / … / "stressed").

Routing
-------
Owns its own :class:`fastapi.APIRouter`. Wire in ``main.py`` with::

    from pfm.ml_regime_router import router as ml_regime_router
    app.include_router(ml_regime_router)
"""

from __future__ import annotations

import logging
from itertools import pairwise
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

# Reuse the battle-tested factor-clusters primitives rather than reimplementing
# history loading / Δlogit / matrix assembly. Imported *by value* so tests can
# monkeypatch the loaders on this module's namespace (see test_ml_regime.py).
from pfm.terminal.factor_clusters import (
    _build_returns_matrix,
    _load_cached_history,
    _load_factor_meta,  # noqa: F401  (kept for namespace parity / future theme filter)
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml-hub"])

# State-feature column order — fixed so centroids and labels stay aligned.
FEATURE_NAMES: tuple[str, ...] = ("dispersion", "activity", "comovement", "volatility")

MIN_REGIME_OBS: int = 20  # need a reasonable run of daily features to fit a mixture
MAX_PATH_DAYS: int = 250  # cap the returned path so payloads stay sane
REGIME_TTL_SECONDS: int = 600  # regime geometry shifts slowly; 10 min is comfy

# Candidate component counts for the BIC model-selection hint.
BIC_CANDIDATES: tuple[int, ...] = (2, 3, 4, 5)

# One-line honesty note carried in every response.
SCOPE_NOTE: str = (
    "Regimes describe the prediction-market factor cross-section, not equity-market state."
)

# Human labels by regime rank (0 = calmest). Extra components beyond five fall
# back to "stressed" since they sit above the normal band by construction.
_LABEL_BY_RANK: tuple[str, ...] = ("calm", "quiet-normal", "normal", "elevated", "stressed")


# --- response schemas -------------------------------------------------------


class RegimePoint(BaseModel):
    """One day on the regime timeline."""

    date: str = Field(..., description="UTC date, ISO-8601 (YYYY-MM-DD).")
    regime: int = Field(..., description="Regime id (0 = calmest, last = most stressed).")


class RegimeSummary(BaseModel):
    """Aggregate stats for one detected regime."""

    regime: int
    label: str = Field(..., description="Human label derived from the volatility ordering.")
    n_days: int = Field(..., ge=0, description="Days assigned to this regime.")
    avg_duration_days: float = Field(
        ..., ge=0.0, description="Mean length of contiguous runs in this regime."
    )
    centroid: dict[str, float] = Field(
        ..., description="Mean of the daily state features for this regime."
    )


class FactorRegimeStats(BaseModel):
    """Per-regime Δlogit behaviour of a single requested factor."""

    regime: int
    label: str
    n_days: int = Field(..., ge=0, description="Factor observations falling in this regime.")
    mean_dlogit: float = Field(..., description="Mean Δlogit return within the regime.")
    vol_dlogit: float = Field(..., description="Std (ddof=0) of Δlogit within the regime.")
    hit_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of regime days with Δlogit > 0."
    )


class RegimeResponse(BaseModel):
    n_obs: int = Field(..., description="Daily state-feature rows fed to the classifier.")
    n_regimes: int
    window: int = Field(..., description="Rolling window (days) for co-movement / volatility.")
    current_regime: int = Field(..., description="Regime of the most recent day (-1 if degraded).")
    current_label: str
    degraded_mode: bool = False
    reason: str | None = None
    regimes: list[RegimeSummary]
    path: list[RegimePoint]
    # --- new, backward-compatible fields ---
    scope: str = Field(default=SCOPE_NOTE, description="Honesty note on what regimes describe.")
    transition_matrix: list[list[float]] = Field(
        default_factory=list,
        description="Row-normalized empirical Markov P(next=j | now=i), rounded 3dp.",
    )
    current_expected_remaining_days: float = Field(
        default=0.0,
        ge=0.0,
        description="Expected days remaining in the current regime = 1/(1 - P[i][i]).",
    )
    bic_by_n_regimes: dict[int, float] = Field(
        default_factory=dict,
        description="BIC of a GaussianMixture fit per candidate n_regimes (lower is better).",
    )
    recommended_n_regimes: int | None = Field(
        default=None, description="argmin BIC over the candidate n_regimes (None if undecidable)."
    )
    factor: str | None = Field(
        default=None, description="Requested factor slug for per-series stats (if any)."
    )
    factor_series_stats: dict[int, FactorRegimeStats] | None = Field(
        default=None,
        description="Per-regime Δlogit stats for the requested factor, keyed by regime id.",
    )


# --- feature engineering ----------------------------------------------------


def _state_features(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Build daily market-state features from the Δlogit returns matrix.

    Args:
        returns: wide ``[date x factor]`` Δlogit frame.
        window: rolling window (days) for the co-movement and volatility terms.

    Returns:
        A ``[date x feature]`` frame with the columns in :data:`FEATURE_NAMES`,
        rows with any NaN dropped (the leading ``window`` rows lack a rolling
        estimate).
    """
    # Per-day cross-sectional stats: dispersion (spread) and activity (magnitude).
    dispersion = returns.std(axis=1, ddof=0)
    activity = returns.abs().mean(axis=1)

    # Equal-weight factor index → its trailing std is aggregate market vol.
    index = returns.mean(axis=1)
    volatility = index.rolling(window, min_periods=window).std(ddof=0)

    # Rolling average pairwise correlation: mean off-diagonal of the trailing
    # window's correlation matrix. Computed row-by-row over the window so a
    # constant column inside a window degrades to 0 rather than NaN.
    comovement = _rolling_avg_pairwise_corr(returns, window)

    feats = pd.DataFrame(
        {
            "dispersion": dispersion,
            "activity": activity,
            "comovement": comovement,
            "volatility": volatility,
        }
    )
    return feats[list(FEATURE_NAMES)].dropna()


def _rolling_avg_pairwise_corr(returns: pd.DataFrame, window: int) -> pd.Series:
    """Average off-diagonal pairwise correlation over a trailing window.

    For each day ``t`` we take rows ``[t-window+1, t]``, compute the correlation
    matrix, and average its upper triangle. Days without a full window (or whose
    window has no estimable pair) get NaN and are dropped downstream.
    """
    vals = returns.to_numpy(dtype=float)
    dates = returns.index
    n_rows, n_cols = vals.shape
    out = np.full(n_rows, np.nan)
    if n_cols < 2:
        return pd.Series(out, index=dates)
    iu = np.triu_indices(n_cols, k=1)
    for t in range(window - 1, n_rows):
        block = vals[t - window + 1 : t + 1]
        # Need variation in each column to correlate; ddof handled by corrcoef.
        if block.shape[0] < 2:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            cmat = np.corrcoef(block, rowvar=False)
        pair = cmat[iu]
        pair = pair[np.isfinite(pair)]
        if pair.size:
            out[t] = float(np.mean(pair))
    return pd.Series(out, index=dates)


# --- classification ---------------------------------------------------------


def _fit_regimes(feats: pd.DataFrame, n_regimes: int) -> np.ndarray:
    """Fit a GaussianMixture and return volatility-sorted regime labels.

    Regimes are re-indexed by ascending mean ``volatility`` so label ``0`` is
    always the calmest segment and the last label the most stressed — a stable,
    interpretable ordering across re-runs and parameter choices.

    Args:
        feats: ``[date x feature]`` state-feature frame.
        n_regimes: number of mixture components to fit.

    Returns:
        Integer label per row of ``feats`` in volatility-sorted regime ids.
    """
    from sklearn.mixture import GaussianMixture

    x = feats.to_numpy(dtype=float)
    # Standardize so no single feature (e.g. volatility's scale) dominates the
    # Gaussian covariance; done in-numpy to stay warning-free and dependency-light.
    mu = x.mean(axis=0)
    sigma = x.std(axis=0, ddof=0)
    sigma[sigma == 0.0] = 1.0
    xz = (x - mu) / sigma

    model = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        random_state=0,
        n_init=1,
        reg_covar=1e-4,
    )
    raw_labels = model.fit_predict(xz)

    # Rank components by mean (raw, un-standardized) volatility, ascending.
    vol = feats["volatility"].to_numpy(dtype=float)
    present = sorted({int(label) for label in raw_labels})
    mean_vol = {comp: float(np.mean(vol[raw_labels == comp])) for comp in present}
    order = sorted(present, key=lambda c: mean_vol[c])
    remap = {comp: rank for rank, comp in enumerate(order)}
    return np.array([remap[int(label)] for label in raw_labels], dtype=int)


def _avg_run_length(labels: np.ndarray, regime: int) -> float:
    """Mean length of contiguous runs equal to ``regime`` (0.0 if absent)."""
    runs: list[int] = []
    current = 0
    for label in labels:
        if int(label) == regime:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return float(np.mean(runs)) if runs else 0.0


def _label_for_rank(rank: int, n_regimes: int) -> str:
    """Map a volatility rank to a human label, spanning calm…stressed."""
    if rank <= 0:
        return "calm"
    if rank >= n_regimes - 1:
        return "stressed"
    if rank < len(_LABEL_BY_RANK):
        return _LABEL_BY_RANK[rank]
    return "elevated"


def _transition_matrix(labels: np.ndarray, n_regimes: int) -> list[list[float]]:
    """Empirical row-normalized Markov transition matrix from a label path.

    Builds an ``n_regimes × n_regimes`` count matrix of consecutive
    ``(now, next)`` transitions, then row-normalizes to ``P(next=j | now=i)``.
    A regime that never appears as a "from" state yields an all-zero row (no
    transitions observed) rather than dividing by zero.

    Args:
        labels: integer regime id per day (volatility-sorted).
        n_regimes: requested number of regimes (matrix dimension).

    Returns:
        Row-normalized transition probabilities, each entry rounded to 3dp.
    """
    counts = np.zeros((n_regimes, n_regimes), dtype=float)
    for now, nxt in pairwise(labels):
        i, j = int(now), int(nxt)
        if 0 <= i < n_regimes and 0 <= j < n_regimes:
            counts[i, j] += 1.0
    out: list[list[float]] = []
    for i in range(n_regimes):
        row_sum = counts[i].sum()
        if row_sum > 0:
            row = counts[i] / row_sum
        else:
            row = counts[i]  # all zeros — never observed as a source state
        out.append([round(float(v), 3) for v in row])
    return out


def _expected_remaining_days(transition_matrix: list[list[float]], current: int) -> float:
    """Expected remaining days in ``current`` regime = 1 / (1 - P[i][i]).

    Models the regime as a geometric dwell time given the self-transition
    probability. Guards against ``P[i][i] >= 1`` (absorbing / single-observation
    states) by clamping to a large-but-finite horizon.
    """
    if not transition_matrix or not (0 <= current < len(transition_matrix)):
        return 0.0
    p_stay = transition_matrix[current][current]
    denom = 1.0 - p_stay
    if denom <= 0.0:
        return float(MAX_PATH_DAYS)  # effectively absorbing within our horizon
    return round(1.0 / denom, 3)


def _bic_by_n_regimes(feats: pd.DataFrame) -> dict[int, float]:
    """Fit a GaussianMixture per candidate n_regimes and record its BIC.

    Standardizes features identically to :func:`_fit_regimes`, fits on each
    candidate in :data:`BIC_CANDIDATES`, and records ``model.bic(xz)`` (lower is
    better). Candidates with fewer observations than components are skipped.

    Args:
        feats: ``[date x feature]`` state-feature frame.

    Returns:
        ``{n_regimes: bic}`` for every fittable candidate (rounded 3dp).
    """
    from sklearn.mixture import GaussianMixture

    x = feats.to_numpy(dtype=float)
    n_obs = x.shape[0]
    mu = x.mean(axis=0)
    sigma = x.std(axis=0, ddof=0)
    sigma[sigma == 0.0] = 1.0
    xz = (x - mu) / sigma

    bic: dict[int, float] = {}
    for k in BIC_CANDIDATES:
        if n_obs <= k:  # need more observations than components for a stable fit
            continue
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0,
            n_init=1,
            reg_covar=1e-4,
        )
        model.fit(xz)
        bic[k] = round(float(model.bic(xz)), 3)
    return bic


def _factor_stats_by_regime(
    factor_returns: pd.Series,
    feat_index: pd.Index,
    labels: np.ndarray,
    n_regimes: int,
) -> dict[int, FactorRegimeStats]:
    """Per-regime Δlogit stats for one factor, aligned to the regime path.

    Aligns the factor's Δlogit series to the state-feature dates (inner join on
    date), then for each regime present computes mean, std (ddof=0), positive
    hit-rate, and day count.

    Args:
        factor_returns: Δlogit series for the requested factor (date-indexed).
        feat_index: the state-feature frame's date index (one entry per label).
        labels: regime id per ``feat_index`` row.
        n_regimes: requested number of regimes (for labelling).

    Returns:
        ``{regime_id: FactorRegimeStats}`` for every regime with ≥1 aligned day.
    """
    label_series = pd.Series(labels, index=feat_index)
    aligned = pd.DataFrame({"ret": factor_returns, "regime": label_series}).dropna()
    out: dict[int, FactorRegimeStats] = {}
    for rank in sorted({int(r) for r in aligned["regime"].to_numpy()}):
        rets = aligned.loc[aligned["regime"] == rank, "ret"].to_numpy(dtype=float)
        if rets.size == 0:
            continue
        out[rank] = FactorRegimeStats(
            regime=rank,
            label=_label_for_rank(rank, n_regimes),
            n_days=int(rets.size),
            mean_dlogit=round(float(np.mean(rets)), 6),
            vol_dlogit=round(float(np.std(rets, ddof=0)), 6),
            hit_rate=round(float(np.mean(rets > 0.0)), 6),
        )
    return out


# --- main endpoint ----------------------------------------------------------


@router.get(
    "/regime",
    response_model=None,
    summary="Classify recent market state into discrete regimes (Regime Monitor).",
)
def regime(
    n_regimes: Annotated[
        int,
        Query(ge=2, le=5, description="Number of discrete regimes (GaussianMixture components)."),
    ] = 3,
    window: Annotated[
        int,
        Query(ge=2, le=120, description="Rolling window (days) for co-movement & volatility."),
    ] = 10,
    factor: Annotated[
        str | None,
        Query(description="Optional factor slug: attach per-regime Δlogit stats for it."),
    ] = None,
    request: Request = None,  # type: ignore[assignment]
) -> RegimeResponse:
    """Classify the recent prediction-market state into discrete regimes.

    Pipeline: cached daily-probability history → Δlogit returns → per-day state
    features (dispersion, activity, co-movement, aggregate volatility) →
    GaussianMixture with ``n_regimes`` components → volatility-sorted labels
    (regime 0 = calm, last = stressed). Reports the full per-day path (last
    ``MAX_PATH_DAYS`` days), the current regime, and per-regime persistence
    stats + feature centroids.

    Cached for ``REGIME_TTL_SECONDS`` via the shared L1/L2 TERMINAL_CACHE keyed
    on ``(n_regimes, window)``. Returns ``degraded_mode=true`` (empty path) when
    the factor-history cache is cold, mirroring ``/ml/factor-map``.
    """
    from pfm import terminal as _term_mod  # circular-safe lazy import

    cache_key = f"ml_regime::{n_regimes}::{window}::{factor or ''}"
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return RegimeResponse.model_validate(cached)

    def _degraded(reason: str) -> RegimeResponse:
        return RegimeResponse(
            n_obs=0,
            n_regimes=n_regimes,
            window=window,
            current_regime=-1,
            current_label="unknown",
            degraded_mode=True,
            reason=reason,
            regimes=[],
            path=[],
        )

    history = _load_cached_history()
    if not history:
        return _degraded(
            "Factor-history cache is empty (run the strat7 batch job or wait "
            "for the in-process prewarm to populate it)."
        )

    returns = _build_returns_matrix(history)
    if returns.empty or returns.shape[1] < 2:
        raise HTTPException(
            status_code=422,
            detail="Need at least 2 factors with sufficient history to detect regimes.",
        )

    feats = _state_features(returns, window=window)
    n_obs = int(feats.shape[0])
    if n_obs < MIN_REGIME_OBS:
        return _degraded(
            f"Only {n_obs} daily state-feature rows after a {window}-day rolling "
            f"window; need at least {MIN_REGIME_OBS} to fit {n_regimes} regimes."
        )
    if n_obs < n_regimes:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least n_regimes={n_regimes} observations; have {n_obs}.",
        )

    labels = _fit_regimes(feats, n_regimes=n_regimes)

    # Per-regime summaries (one row per id actually present; ids are 0..k-1
    # after the volatility remap, but a degenerate fit can drop a component).
    summaries: list[RegimeSummary] = []
    for rank in sorted({int(label) for label in labels}):
        mask = labels == rank
        centroid = {
            name: round(float(feats[name].to_numpy()[mask].mean()), 6) for name in FEATURE_NAMES
        }
        summaries.append(
            RegimeSummary(
                regime=rank,
                label=_label_for_rank(rank, n_regimes),
                n_days=int(mask.sum()),
                avg_duration_days=round(_avg_run_length(labels, rank), 3),
                centroid=centroid,
            )
        )

    current_regime = int(labels[-1])
    current_label = _label_for_rank(current_regime, n_regimes)

    # Empirical Markov transition matrix + expected dwell time for the current
    # regime (geometric model on the self-transition probability).
    transition_matrix = _transition_matrix(labels, n_regimes)
    current_expected_remaining_days = _expected_remaining_days(transition_matrix, current_regime)

    # BIC model-selection hint over candidate component counts on the same feats.
    bic = _bic_by_n_regimes(feats)
    recommended = min(bic, key=bic.get) if bic else None

    # Optional per-factor per-regime Δlogit stats. 404 if the slug isn't in the
    # cached returns matrix (column = factor slug).
    factor_series_stats: dict[int, FactorRegimeStats] | None = None
    if factor is not None:
        if factor not in returns.columns:
            raise HTTPException(
                status_code=404,
                detail=f"Factor {factor!r} not in cached history.",
            )
        factor_series_stats = _factor_stats_by_regime(
            returns[factor].dropna(), feats.index, labels, n_regimes
        )

    # Cap the path to the most recent MAX_PATH_DAYS rows.
    tail = feats.index[-MAX_PATH_DAYS:]
    tail_labels = labels[-MAX_PATH_DAYS:]
    path = [
        RegimePoint(date=pd.Timestamp(dt).date().isoformat(), regime=int(label))
        for dt, label in zip(tail, tail_labels, strict=True)
    ]

    resp = RegimeResponse(
        n_obs=n_obs,
        n_regimes=n_regimes,
        window=window,
        current_regime=current_regime,
        current_label=current_label,
        regimes=summaries,
        path=path,
        scope=SCOPE_NOTE,
        transition_matrix=transition_matrix,
        current_expected_remaining_days=current_expected_remaining_days,
        bic_by_n_regimes=bic,
        recommended_n_regimes=recommended,
        factor=factor,
        factor_series_stats=factor_series_stats,
    )
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), REGIME_TTL_SECONDS)
    return resp


__all__ = [
    "FactorRegimeStats",
    "RegimePoint",
    "RegimeResponse",
    "RegimeSummary",
    "regime",
    "router",
]
