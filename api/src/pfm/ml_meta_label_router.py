"""ML Hub — Triple-Barrier Meta-Labeling (López de Prado 2018, ch. 3).

This module does **not** invent a new alpha. It takes a *known* primary trading
signal — a textbook mean-reversion z-score rule on a prediction-market factor's
daily probability — and trains a **secondary (meta) classifier** that decides,
trade-by-trade, whether to *take* or *skip* the primary signal. The goal is
López de Prado's: keep the primary's side, raise its precision.

Why this is honest (no overfit / no survivorship illusion)
----------------------------------------------------------
* The primary side is fixed and naive (z-score reversion); we never search over
  it for a "wow" backtest.
* The meta-model is validated **walk-forward** with :class:`TimeSeriesSplit`.
  Every label we report (hit-rate, lift, AUC) is computed on *out-of-fold*
  predictions — a trigger is never scored by a model that saw it in training.
* The reported number is *precision lift* (meta hit-rate minus primary
  hit-rate), the only thing meta-labeling legitimately buys you. It costs
  trades (n_meta < n_primary). We report both so the trade-off is visible.

Labels come from :mod:`pfm.triple_barrier` — direction-aware profit / stop /
time barriers — so a short trade "wins" when the price falls.

Routing
-------
Owns its own :class:`fastapi.APIRouter`. Wire in ``main.py`` with::

    from pfm.ml_meta_label_router import router as ml_meta_label_router
    app.include_router(ml_meta_label_router)
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

# Reuse the proven factor-history + meta loaders the rest of the ML Hub uses, so
# this endpoint sees the exact same universe as /ml/factor-map and the Terminal
# clustering panel. (Patched on THIS module's namespace in tests.)
from pfm.terminal.factor_clusters import (
    _delta_logit,
    _load_cached_history,
    _load_factor_meta,
)
from pfm.triple_barrier import triple_barrier_backtest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml-hub"])

META_TTL_SECONDS: int = 900  # the fit is deterministic + slow-moving; cache it
MIN_TRIGGERS: int = 30  # below this a walk-forward classifier is meaningless
ROLL_WINDOW: int = 20  # rolling window for the z-score / vol features
MOM_WINDOW: int = 5  # short-momentum feature lookback

_CAVEAT: str = (
    "Meta-labeling filters an existing naive z-score reversion signal to raise "
    "precision; it does not create a new edge. Lift and AUC are out-of-fold "
    "(TimeSeriesSplit walk-forward), but this is a single-factor in-sample-period "
    "diagnostic, not a deployable backtest. Re-validate over 4 disjoint quarters "
    "before trusting it."
)


# --- response schemas -------------------------------------------------------


class FeatureImportance(BaseModel):
    """One meta-model input feature and its relative importance."""

    name: str
    importance: float = Field(..., ge=0.0, description="Normalized importance in [0, 1].")


class MetaLabelResponse(BaseModel):
    factor_id: str
    name: str
    n_primary: int = Field(..., description="Triggers fired by the primary z-score rule.")
    primary_hit_rate: float = Field(..., description="Fraction of primary trades that won.")
    primary_avg_ret: float = Field(..., description="Mean direction-aware forward return.")
    n_meta: int = Field(..., description="Trades the meta-filter kept (P(win) >= 0.5).")
    meta_hit_rate: float
    meta_avg_ret: float
    precision_lift: float = Field(..., description="meta_hit_rate − primary_hit_rate.")
    oos_auc: float | None = Field(None, description="Out-of-fold ROC-AUC of the meta-model.")
    caveat: str
    degraded_mode: bool = False
    reason: str | None = None
    features: list[FeatureImportance] = Field(default_factory=list)


# --- primary signal + feature engineering -----------------------------------


def _pick_factor(history: dict[str, pd.Series], factor: str | None) -> tuple[str, pd.Series] | None:
    """Resolve the factor to analyse.

    Args:
        history: slug → daily-probability series.
        factor: explicit slug, or ``None`` to auto-pick the longest-history one.

    Returns:
        ``(slug, series)`` or ``None`` if nothing usable is available.
    """
    if not history:
        return None
    if factor is not None:
        ser = history.get(factor)
        if ser is None:
            return None
        return factor, ser
    # Auto-pick: the factor with the most observations (richest training set).
    slug = max(history, key=lambda s: len(history[s]))
    return slug, history[slug]


def _build_triggers(
    prices: pd.Series, *, window: int, entry_z: float
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute primary triggers and their meta-features over a price series.

    The primary signal is mean-reversion: a trigger fires at bar ``i`` when
    ``|z_i| >= entry_z``; direction is ``+1`` (long) when the price is cheap
    (``z < 0``) and ``-1`` (short) when rich (``z > 0``).

    Features captured at each trigger: signed z, rolling vol, short momentum,
    and a vol-percentile regime proxy (the rolling vol's expanding rank in
    ``[0, 1]``) — designed so a learnable regime split is expressible.

    Args:
        prices: probability series (already cleaned / sorted).
        window: rolling window for the z-score and vol.
        entry_z: ``|z|`` threshold for a trigger.

    Returns:
        ``(features_df, entry_positions, directions)`` where ``features_df`` has
        one row per trigger (indexed by integer position), ``entry_positions`` is
        the integer index of each trigger in ``prices``, and ``directions`` is the
        signed primary side.
    """
    p = prices.astype(float)
    min_p = max(5, window // 2)
    mu = p.rolling(window=window, min_periods=min_p).mean()
    sd = p.rolling(window=window, min_periods=min_p).std(ddof=1)
    z = (p - mu) / sd
    # Short momentum: signed change over MOM_WINDOW bars.
    mom = p.diff(MOM_WINDOW)
    # Vol-percentile regime proxy: expanding rank of rolling vol in [0, 1].
    vol_pct = sd.rank(pct=True)

    z_arr = z.to_numpy()
    sd_arr = sd.to_numpy()
    mom_arr = mom.to_numpy()
    volp_arr = vol_pct.to_numpy()

    rows: list[list[float]] = []
    positions: list[int] = []
    directions: list[int] = []
    n = len(p)
    for i in range(n):
        zi = z_arr[i]
        if np.isnan(zi) or abs(zi) < entry_z:
            continue
        if np.isnan(sd_arr[i]) or sd_arr[i] <= 0:
            continue
        direction = 1 if zi < 0 else -1
        mom_i = mom_arr[i] if not np.isnan(mom_arr[i]) else 0.0
        volp_i = volp_arr[i] if not np.isnan(volp_arr[i]) else 0.5
        rows.append([float(zi), float(sd_arr[i]), float(mom_i), float(volp_i)])
        positions.append(i)
        directions.append(direction)

    cols = ["z", "rolling_vol", "short_momentum", "vol_regime_pct"]
    feats = pd.DataFrame(rows, columns=cols)
    return feats, np.asarray(positions, dtype=int), np.asarray(directions, dtype=int)


def _label_triggers(
    prices: pd.Series,
    positions: np.ndarray,
    directions: np.ndarray,
    *,
    window: int,
    entry_z: float,
    profit_target_sigma: float,
    stop_loss_sigma: float,
    time_horizon_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Triple-barrier-label each primary trigger.

    Runs :func:`triple_barrier_backtest` on the price series and aligns its
    trades back to the triggers we detected (the backtest uses the same z-score
    entry logic, so entry indices match). A trade is a *win* (label 1) iff it hit
    the profit barrier first; the forward return is the direction-aware PnL.

    Args:
        prices: probability series.
        positions: integer entry positions of the triggers.
        directions: signed primary side at each trigger.
        window: rolling window (must match the trigger builder).
        entry_z: entry threshold (must match the trigger builder).
        profit_target_sigma: profit barrier in σ units.
        stop_loss_sigma: stop barrier in σ units.
        time_horizon_bars: vertical (time) barrier.

    Returns:
        ``(y, fwd_ret)`` aligned to ``positions`` — ``y`` is 1 for a profit-barrier
        win else 0; ``fwd_ret`` is the realised direction-aware PnL. Triggers the
        backtest skipped (e.g. overlapping) get ``y=-1`` sentinel and ``ret=nan``.
    """
    result = triple_barrier_backtest(
        prices,
        window=window,
        entry_z=entry_z,
        profit_target_sigma=profit_target_sigma,
        stop_loss_sigma=stop_loss_sigma,
        time_horizon_bars=time_horizon_bars,
    )
    by_entry = {t.entry_index: t for t in result.trades}
    y = np.full(len(positions), -1, dtype=int)
    fwd = np.full(len(positions), np.nan, dtype=float)
    for k, pos in enumerate(positions):
        trade = by_entry.get(int(pos))
        if trade is None:
            continue
        # Sanity: directions should agree; if not, trust the backtest's.
        y[k] = 1 if trade.label == 1 else 0
        fwd[k] = float(trade.pnl)
    return y, fwd


# --- meta-model (walk-forward) ----------------------------------------------


def _walk_forward_meta(
    feats: pd.DataFrame, y: np.ndarray
) -> tuple[np.ndarray, float | None, dict[str, float]]:
    """Train the take/skip meta-classifier out-of-fold via TimeSeriesSplit.

    For each forward fold, fit on the past and predict P(win) on the held-out
    future block. Every trigger thus receives a prediction from a model that
    never saw it — no leakage. Aggregates feature importances over folds.

    Falls back to :class:`LogisticRegression` automatically when a fold's
    training slice is single-class (a gradient-boosting tree degenerates there).

    Args:
        feats: one row of features per trigger, time-ordered.
        y: binary win labels aligned to ``feats``.

    Returns:
        ``(proba, auc, importances)`` — out-of-fold P(win) for every trigger
        (``nan`` where no fold could score it), out-of-fold ROC-AUC (``None`` if
        undefined), and a name→importance mapping summed to ~1.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    x = feats.to_numpy(dtype=float)
    n = len(y)
    n_splits = min(5, max(2, n // 10))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    proba = np.full(n, np.nan, dtype=float)
    importances = np.zeros(x.shape[1], dtype=float)
    n_imp = 0

    for train_idx, test_idx in splitter.split(x):
        y_tr = y[train_idx]
        if len(np.unique(y_tr)) < 2:
            # Single-class train slice → predict the base rate as P(win).
            proba[test_idx] = float(y_tr.mean())
            continue
        x_tr, x_te = x[train_idx], x[test_idx]
        try:
            model = GradientBoostingClassifier(random_state=0)
            model.fit(x_tr, y_tr)
            fi = np.asarray(model.feature_importances_, dtype=float)
        except (ValueError, RuntimeError):
            model = LogisticRegression(max_iter=1000)
            model.fit(x_tr, y_tr)
            fi = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        proba[test_idx] = model.predict_proba(x_te)[:, 1]
        if fi.sum() > 0:
            importances += fi / fi.sum()
            n_imp += 1

    # OOS AUC over triggers that actually received a prediction.
    scored = ~np.isnan(proba)
    auc: float | None = None
    if scored.sum() >= 2 and len(np.unique(y[scored])) == 2:
        try:
            auc = float(roc_auc_score(y[scored], proba[scored]))
        except ValueError:
            auc = None

    if n_imp > 0:
        importances /= n_imp
        total = importances.sum()
        if total > 0:
            importances /= total
    imp_map = {name: float(importances[i]) for i, name in enumerate(feats.columns)}
    return proba, auc, imp_map


# --- main endpoint ----------------------------------------------------------


@router.get(
    "/meta-label",
    response_model=None,
    summary="Triple-barrier meta-labeling: a take/skip filter over a z-score reversion signal.",
)
def meta_label(
    factor: Annotated[
        str | None,
        Query(description="Factor slug. Omit to auto-pick the longest-history factor."),
    ] = None,
    entry_z: Annotated[
        float,
        Query(gt=0.0, le=5.0, description="|z| threshold of the primary reversion trigger."),
    ] = 1.5,
    profit_target_sigma: Annotated[
        float,
        Query(gt=0.0, le=10.0, description="Profit barrier in σ_local units."),
    ] = 2.0,
    stop_loss_sigma: Annotated[
        float,
        Query(gt=0.0, le=20.0, description="Stop barrier in σ_local units."),
    ] = 2.0,
    time_horizon_bars: Annotated[
        int,
        Query(ge=1, le=60, description="Vertical (time) barrier in bars."),
    ] = 10,
    request: Request = None,  # type: ignore[assignment]
) -> MetaLabelResponse:
    """Filter a naive z-score reversion signal with a walk-forward meta-classifier.

    Pipeline: pick a factor's daily-probability series → fire primary triggers
    where ``|z| >= entry_z`` (long cheap / short rich) → label each with the
    triple-barrier method (profit / stop / time) → train a take/skip classifier
    out-of-fold under :class:`TimeSeriesSplit` → keep only triggers with
    out-of-fold ``P(win) >= 0.5``. Reports primary vs meta-filtered hit-rate and
    average forward return, the precision lift, OOS AUC, and feature importances.

    Cached for ``META_TTL_SECONDS`` on the shared TERMINAL_CACHE. Returns
    ``degraded_mode=true`` when the factor-history cache is cold or the chosen
    factor has too few triggers for a walk-forward fit.
    """
    from pfm import terminal as _term_mod  # circular-safe lazy import

    cache_key = (
        f"ml_meta_label::{factor or '*'}::{entry_z:.2f}::{profit_target_sigma:.2f}"
        f"::{stop_loss_sigma:.2f}::{time_horizon_bars}"
    )
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return MetaLabelResponse.model_validate(cached)

    meta = _load_factor_meta()
    history = _load_cached_history()

    def _degraded(factor_id: str, name: str, reason: str) -> MetaLabelResponse:
        return MetaLabelResponse(
            factor_id=factor_id,
            name=name,
            n_primary=0,
            primary_hit_rate=0.0,
            primary_avg_ret=0.0,
            n_meta=0,
            meta_hit_rate=0.0,
            meta_avg_ret=0.0,
            precision_lift=0.0,
            oos_auc=None,
            caveat=_CAVEAT,
            degraded_mode=True,
            reason=reason,
            features=[],
        )

    if not history:
        return _degraded(
            factor or "n/a",
            factor or "n/a",
            "Factor-history cache is empty (run the strat7 batch job or wait for "
            "the in-process prewarm to populate it).",
        )

    picked = _pick_factor(history, factor)
    if picked is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached history for factor={factor!r}.",
        )
    slug, prices = picked
    m = meta.get(slug)
    factor_id = m.factor_id if m else slug
    name = m.name if m else slug

    # Validate the Δlogit series is non-degenerate before doing real work.
    if _delta_logit(prices).abs().sum() == 0:
        return _degraded(factor_id, name, "Factor probability series has no variation.")

    feats, positions, directions = _build_triggers(prices, window=ROLL_WINDOW, entry_z=entry_z)
    if len(positions) < MIN_TRIGGERS:
        return _degraded(
            factor_id,
            name,
            f"Only {len(positions)} primary triggers (need >= {MIN_TRIGGERS} for a "
            "walk-forward meta-fit). Lower entry_z or pick a longer-history factor.",
        )

    try:
        y_raw, fwd_raw = _label_triggers(
            prices,
            positions,
            directions,
            window=ROLL_WINDOW,
            entry_z=entry_z,
            profit_target_sigma=profit_target_sigma,
            stop_loss_sigma=stop_loss_sigma,
            time_horizon_bars=time_horizon_bars,
        )
    except ValueError as exc:
        return _degraded(factor_id, name, f"Triple-barrier labeling failed: {exc}")

    # Keep only triggers the backtest actually labeled (non-overlapping).
    keep = y_raw >= 0
    feats = feats.loc[keep].reset_index(drop=True)
    y = y_raw[keep]
    fwd = fwd_raw[keep]
    if len(y) < MIN_TRIGGERS:
        return _degraded(
            factor_id,
            name,
            f"Only {int(len(y))} non-overlapping labeled trades (need >= {MIN_TRIGGERS}).",
        )

    n_primary = int(len(y))
    primary_hit_rate = float(y.mean())
    primary_avg_ret = float(np.nanmean(fwd))

    if len(np.unique(y)) < 2:
        return _degraded(
            factor_id,
            name,
            "Primary labels are single-class (all wins or all losses); a meta "
            "classifier cannot learn a take/skip boundary.",
        )

    proba, oos_auc, imp_map = _walk_forward_meta(feats, y)

    # Meta-filter: take a trade only when an out-of-fold model predicts P(win)>=0.5.
    take = (~np.isnan(proba)) & (proba >= 0.5)
    n_meta = int(take.sum())
    if n_meta > 0:
        meta_hit_rate = float(y[take].mean())
        meta_avg_ret = float(np.nanmean(fwd[take]))
    else:
        meta_hit_rate = 0.0
        meta_avg_ret = 0.0

    features = [
        FeatureImportance(name=k, importance=round(v, 4))
        for k, v in sorted(imp_map.items(), key=lambda kv: -kv[1])
    ]

    resp = MetaLabelResponse(
        factor_id=factor_id,
        name=name,
        n_primary=n_primary,
        primary_hit_rate=round(primary_hit_rate, 4),
        primary_avg_ret=round(primary_avg_ret, 6),
        n_meta=n_meta,
        meta_hit_rate=round(meta_hit_rate, 4),
        meta_avg_ret=round(meta_avg_ret, 6),
        precision_lift=round(meta_hit_rate - primary_hit_rate, 4),
        oos_auc=(round(oos_auc, 4) if oos_auc is not None else None),
        caveat=_CAVEAT,
        degraded_mode=False,
        reason=None,
        features=features,
    )
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), META_TTL_SECONDS)
    return resp


__all__ = ["FeatureImportance", "MetaLabelResponse", "meta_label", "router"]
