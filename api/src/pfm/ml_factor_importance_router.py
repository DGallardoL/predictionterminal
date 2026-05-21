"""ML Hub — non-linear factor-importance diagnostic for a target equity.

This module answers a question the linear OLS factor model (``/fit``) cannot:
*which prediction-market factors carry the most information about a stock's
daily returns once we allow for non-linearities and interactions?* It fits a
gradient-boosted regressor of the target ticker's daily log returns on the
Δlogit returns of candidate factors and ranks each factor by **permutation
importance** evaluated under **walk-forward (time-series) cross-validation**.

Why this is honest (and explicitly NOT an alpha)
-------------------------------------------------
This project is allergic to overfit return-forecasters — every anti-alpha in
the catalogue died doing exactly that. So this endpoint is framed as a
*diagnostic*, not a tradeable signal:

* Permutation importance is computed **out-of-fold** under
  :class:`~sklearn.model_selection.TimeSeriesSplit`, never in-sample, so a
  factor that only helps the model memorise the training window scores ~0.
* We also report the **walk-forward out-of-sample R²** of the GBM against a
  naive train-mean predictor. A *negative* OOS R² (the common case) means the
  non-linear fit has **no exploitable signal** — and we say so plainly in the
  ``caveat`` field rather than burying it.
* Use it to *complement* the linear ``/fit``: a factor with high non-linear
  importance but a small OLS β is a hint of curvature/interaction worth a
  human look, not a green light to trade.

Pipeline
--------
1.  **Features.** Cached daily-probability history → Δlogit returns per factor
    (the same convention as ``model.py`` and ``/terminal/factor-clusters``,
    via the shared :func:`_delta_logit`).
2.  **Target.** The ticker's daily log returns via
    :func:`pfm.sources.equity.get_log_returns` over the factors' date span.
3.  **Align.** Inner-join features and target on the common UTC dates; require
    at least :data:`MIN_ALIGNED_OBS` rows or return 422.
4.  **Fit + score.** ``HistGradientBoostingRegressor`` with
    ``permutation_importance`` averaged across :data:`N_SPLITS` walk-forward
    folds; OOS R² accumulated over the same folds.
5.  **Rank.** Top-``n`` factors by mean permutation importance.

Routing
-------
Owns its own :class:`fastapi.APIRouter`. Per project convention ``main.py`` is
left untouched — wire it in there with::

    from pfm.ml_factor_importance_router import router as ml_factor_importance_router
    app.include_router(ml_factor_importance_router)
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

# Reuse the battle-tested factor-history / Δlogit primitives rather than
# reimplementing them; this keeps the feature convention identical to
# ``/terminal/factor-clusters``, the ML Hub factor map, and ``model.py``.
from pfm.sources.equity import get_log_returns
from pfm.terminal.factor_clusters import (
    _delta_logit,
    _FactorMeta,
    _load_cached_history,
    _load_factor_meta,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml-hub"])


def _resolve_factor_ids(
    factor_ids: str | None,
    meta: dict[str, _FactorMeta],
    history_keys: set[str],
) -> list[str] | None:
    """Parse a comma-separated ``factor_ids`` query into resolved history slugs.

    Each entry may be the public ``factor_id`` or the raw slug; entries are
    stripped, empties dropped, deduped (order-preserving), mapped id→slug via
    ``meta``, and intersected with the slugs present in the loaded history.
    Returns ``None`` when ``factor_ids`` is absent or only whitespace.
    """
    if factor_ids is None:
        return None
    id_to_slug = {m.factor_id: slug for slug, m in meta.items()}
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in factor_ids.split(","):
        token = raw.strip()
        if not token:
            continue
        slug = token if token in history_keys else id_to_slug.get(token)
        if slug is None or slug not in history_keys or slug in seen:
            continue
        seen.add(slug)
        resolved.append(slug)
    if not resolved and not factor_ids.strip():
        return None
    return resolved


# --- module constants -------------------------------------------------------

MIN_ALIGNED_OBS: int = 60  # below this a walk-forward fit is meaningless
MAX_CANDIDATE_FACTORS: int = 80  # keep the GBM fast; user can lower via query
DEFAULT_TOP_N: int = 15
N_SPLITS: int = 4  # walk-forward CV folds
N_PERM_REPEATS: int = 8  # permutation-importance shuffles per fold
IMPORTANCE_TTL_SECONDS: int = 1800  # importances drift slowly; 30 min is fine

#: Fixed honest framing returned on every successful response. The point of
#: this endpoint is the caveat, not the ranking.
CAVEAT: str = (
    "Diagnostic only — permutation importance over a non-linear fit; not a "
    "tradeable forecast. Negative OOS R² means no exploitable signal."
)


# --- response schemas -------------------------------------------------------


class FactorImportanceItem(BaseModel):
    """One factor's non-linear importance for the target ticker."""

    factor_id: str
    name: str
    theme: str
    importance: float = Field(
        ..., description="Mean walk-forward permutation importance (drop in R²)."
    )
    importance_std: float = Field(
        ..., ge=0.0, description="Std of permutation importance across folds/repeats."
    )
    is_significant: bool = Field(
        False,
        description=(
            "True when importance > 2*importance_std, i.e. distinguishable from "
            "zero. Even when True, only interpretable if oos_r2_interpretable."
        ),
    )


class FactorImportanceResponse(BaseModel):
    ticker: str
    n_aligned_obs: int
    oos_r2: float | None = Field(
        None,
        description=(
            "Walk-forward OOS R² of the GBM vs a naive mean predictor; can be "
            "negative (no signal), or null when not estimable / degraded."
        ),
    )
    model: str = "HistGBR walk-forward"
    oos_r2_interpretable: bool = Field(
        False,
        description=(
            "True only when oos_r2 is not None and > 0. When False the model "
            "has no out-of-sample signal and the importance rankings below are "
            "noise on an overfit model — the frontend should grey them out."
        ),
    )
    caveat: str = CAVEAT
    items: list[FactorImportanceItem]
    # Mirrors the ML Hub / Terminal pattern: an empty result + degraded_mode
    # when the upstream factor-history cache is cold, so the UI can render an
    # explanatory empty state instead of a 503.
    degraded_mode: bool = False
    reason: str | None = None


# --- math helpers -----------------------------------------------------------


def _build_feature_matrix(
    history: dict[str, pd.Series],
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Δlogit feature frame + aligned target on the common UTC dates.

    Args:
        history: ``{slug: daily-probability Series}`` for the candidate factors.
        target: the ticker's daily log returns, indexed by UTC dates.

    Returns:
        ``(X, y, slugs)`` where ``X`` is the inner-joined Δlogit feature frame,
        ``y`` is the aligned target, and ``slugs`` are ``X``'s columns in order.
        ``X`` is empty when no factor overlaps the target.
    """
    cols: dict[str, pd.Series] = {}
    for slug, ser in history.items():
        ret = _delta_logit(ser)
        if ret.empty:
            continue
        ret = ret[~ret.index.duplicated(keep="last")]
        cols[slug] = ret
    if not cols:
        return pd.DataFrame(), pd.Series(dtype=float), []

    feats = pd.DataFrame(cols).sort_index()
    feats.index = pd.to_datetime(feats.index, utc=True).normalize()
    feats = feats[~feats.index.duplicated(keep="last")]

    tgt = target.copy()
    tgt.index = pd.to_datetime(tgt.index, utc=True).normalize()
    tgt = tgt[~tgt.index.duplicated(keep="last")].sort_index()

    joined = feats.join(tgt.rename("__y__"), how="inner").dropna()
    if joined.empty:
        return pd.DataFrame(), pd.Series(dtype=float), []

    y = joined["__y__"]
    x = joined.drop(columns="__y__")
    # Drop constant columns — they can't carry importance and trip the GBM.
    nonconst = [c for c in x.columns if float(x[c].std()) > 0.0]
    x = x[nonconst]
    return x, y, list(x.columns)


def _walk_forward_importance(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Walk-forward permutation importance + OOS R² for a HistGBR.

    For each :class:`TimeSeriesSplit` fold we fit on the expanding train window
    and, on the *test* fold, (a) accumulate predictions for a pooled OOS R² and
    (b) compute permutation importance (drop in test-fold R² when each feature
    is shuffled). Per-feature importances are averaged across folds; their std
    captures fold-to-fold instability.

    Args:
        x: Δlogit feature frame (rows = dates, ordered ascending).
        y: aligned target returns.

    Returns:
        ``(mean_importance, std_importance, oos_r2)`` — the first two are
        per-feature arrays aligned to ``x.columns``; ``oos_r2`` is the pooled
        walk-forward R² of the GBM vs the naive train-mean predictor (can be
        negative when the model has no signal — the honest, common case).
    """
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    n_obs, n_feat = x_arr.shape

    # Fold count must leave a usable test window; clamp for short samples.
    n_splits = max(2, min(N_SPLITS, n_obs // 15))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_importances: list[np.ndarray] = []
    oos_pred: list[np.ndarray] = []
    oos_true: list[np.ndarray] = []
    oos_naive: list[np.ndarray] = []

    for train_idx, test_idx in splitter.split(x_arr):
        if len(test_idx) < 3 or len(train_idx) < 10:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=0,
        )
        model.fit(x_arr[train_idx], y_arr[train_idx])

        # Naive baseline: predict the training-window mean.
        train_mean = float(np.mean(y_arr[train_idx]))
        oos_pred.append(model.predict(x_arr[test_idx]))
        oos_true.append(y_arr[test_idx])
        oos_naive.append(np.full(len(test_idx), train_mean))

        # Permutation importance on the held-out fold only (out-of-sample).
        perm = permutation_importance(
            model,
            x_arr[test_idx],
            y_arr[test_idx],
            scoring="r2",
            n_repeats=N_PERM_REPEATS,
            random_state=0,
        )
        fold_importances.append(perm.importances_mean)

    if not fold_importances:
        # Degenerate: no fold produced a usable split. Report zeros + nan R².
        return np.zeros(n_feat), np.zeros(n_feat), float("nan")

    imp_stack = np.vstack(fold_importances)
    mean_imp = imp_stack.mean(axis=0)
    std_imp = imp_stack.std(axis=0)

    true = np.concatenate(oos_true)
    pred = np.concatenate(oos_pred)
    naive = np.concatenate(oos_naive)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_naive = float(np.sum((true - naive) ** 2))
    # R² relative to the naive mean predictor: 1 - SS_model / SS_naive.
    oos_r2 = 1.0 - ss_res / ss_naive if ss_naive > 0 else float("nan")
    return mean_imp, std_imp, oos_r2


# --- main endpoint ----------------------------------------------------------


@router.get(
    "/factor-importance",
    response_model=None,
    summary="Non-linear factor-importance diagnostic for a target equity (HistGBR + perm. importance).",
)
def factor_importance(
    ticker: Annotated[
        str,
        Query(description="Target equity symbol (e.g. 'NVDA')."),
    ],
    theme: Annotated[
        str | None,
        Query(description="Filter candidate factors by factors.yml theme tag."),
    ] = None,
    top_n: Annotated[
        int,
        Query(ge=1, le=100, description="Number of top-importance factors to return."),
    ] = DEFAULT_TOP_N,
    n_factors: Annotated[
        int,
        Query(ge=3, le=MAX_CANDIDATE_FACTORS, description="Cap on candidate factors (speed)."),
    ] = MAX_CANDIDATE_FACTORS,
    factor_ids: Annotated[
        str | None,
        Query(
            description="Comma-separated factor ids/slugs to use as the candidate pool (overrides theme + auto-select)."
        ),
    ] = None,
    request: Request = None,  # type: ignore[assignment]
) -> FactorImportanceResponse:
    """Rank prediction-market factors by non-linear importance for ``ticker``.

    Fits a gradient-boosted regressor of the ticker's daily log returns on the
    Δlogit of up to ``n_factors`` candidate factors and reports each factor's
    walk-forward permutation importance, plus the pooled out-of-sample R² so
    the caller can judge whether the non-linear fit has *any* signal (it often
    does not — see the ``caveat`` field).

    Cached for :data:`IMPORTANCE_TTL_SECONDS` via the shared L1/L2
    TERMINAL_CACHE keyed on ``(ticker, theme, top_n, n_factors)``. Returns
    ``degraded_mode=true`` (empty items) when the factor-history cache is cold,
    mirroring ``/ml/factor-map`` and ``/terminal/factor-clusters``.
    """
    from pfm import terminal as _term_mod  # circular-safe lazy import

    ticker_u = ticker.strip().upper()
    if not ticker_u:
        raise HTTPException(status_code=422, detail="ticker must be non-empty.")

    cache_key = (
        f"ml_factor_importance::{ticker_u}::{theme or '*'}::{top_n}::{n_factors}"
        f"::{factor_ids or '*'}"
    )
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return FactorImportanceResponse.model_validate(cached)

    meta = _load_factor_meta()
    history = _load_cached_history()
    if not history:
        return FactorImportanceResponse(
            ticker=ticker_u,
            n_aligned_obs=0,
            oos_r2=None,
            items=[],
            degraded_mode=True,
            reason=(
                "Factor-history cache is empty (run the strat7 batch job or wait "
                "for the in-process prewarm to populate it)."
            ),
        )

    selected = _resolve_factor_ids(factor_ids, meta, set(history.keys()))
    if selected is not None:
        # Use exactly the chosen factors as the candidate pool — overrides the
        # theme filter and the auto-select-by-history-length cap below.
        if len(selected) < 3:
            raise HTTPException(
                status_code=422,
                detail="Pick at least 3 factors with history.",
            )
        history = {slug: history[slug] for slug in selected}
    else:
        if theme is not None:
            wanted = {slug for slug, m in meta.items() if m.theme == theme}
            if not wanted:
                raise HTTPException(status_code=404, detail=f"No factors for theme={theme!r}.")
            history = {k: v for k, v in history.items() if k in wanted}
            if not history:
                raise HTTPException(
                    status_code=404,
                    detail=f"No cached history for any theme={theme!r} factor.",
                )

        # Cap candidates to the longest histories — they give the cleanest
        # signal and keep the GBM + permutation loop fast.
        if len(history) > n_factors:
            ranked = sorted(history.items(), key=lambda kv: -len(kv[1]))[:n_factors]
            history = dict(ranked)

    # Date span of the candidate factors → fetch the ticker over the same window.
    all_dates = pd.DatetimeIndex(sorted({d for ser in history.values() for d in ser.index}))
    if len(all_dates) < MIN_ALIGNED_OBS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Candidate factors span only {len(all_dates)} dates; "
                f"need >= {MIN_ALIGNED_OBS} for a walk-forward fit."
            ),
        )
    start = pd.Timestamp(all_dates.min())
    end = pd.Timestamp(all_dates.max())

    try:
        target = get_log_returns(ticker_u, start, end, return_type="log")
    except Exception as exc:  # surface any equity-source failure as a clean 422
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch returns for ticker {ticker_u!r}: {exc}",
        ) from exc

    if target is None or target.empty:
        raise HTTPException(
            status_code=422,
            detail=f"No return data for ticker {ticker_u!r}.",
        )

    x, y, slugs = _build_feature_matrix(history, target)
    if x.empty or len(y) < MIN_ALIGNED_OBS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Only {len(y)} aligned observations between {ticker_u!r} and the "
                f"candidate factors; need >= {MIN_ALIGNED_OBS}."
            ),
        )

    mean_imp, std_imp, oos_r2 = _walk_forward_importance(x, y)

    items: list[FactorImportanceItem] = []
    for slug, imp, std in zip(slugs, mean_imp, std_imp, strict=True):
        m = meta.get(slug)
        imp_r = round(float(imp), 6)
        std_r = round(float(abs(std)), 6)
        items.append(
            FactorImportanceItem(
                factor_id=(m.factor_id if m else slug),
                name=(m.name if m else slug),
                theme=(m.theme if m else "other"),
                importance=imp_r,
                importance_std=std_r,
                is_significant=bool(imp_r > 2.0 * std_r),
            )
        )
    # Rank by mean importance (descending) and keep the top-n.
    items.sort(key=lambda it: it.importance, reverse=True)
    items = items[:top_n]

    oos_r2_val = round(float(oos_r2), 6) if np.isfinite(oos_r2) else None
    interpretable = oos_r2_val is not None and oos_r2_val > 0
    caveat = CAVEAT
    if not interpretable:
        caveat = (
            f"{CAVEAT} OOS R² is non-positive here, so the rankings below are "
            "NOT interpretable — they are noise on an overfit model and must "
            "not be charted as meaningful factor importances."
        )

    resp = FactorImportanceResponse(
        ticker=ticker_u,
        n_aligned_obs=int(len(y)),
        oos_r2=oos_r2_val,
        oos_r2_interpretable=interpretable,
        caveat=caveat,
        items=items,
    )
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), IMPORTANCE_TTL_SECONDS)
    return resp


__all__ = [
    "CAVEAT",
    "FactorImportanceItem",
    "FactorImportanceResponse",
    "factor_importance",
    "router",
]
