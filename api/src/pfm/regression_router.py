"""``/fit`` and ``/attribution`` — the original regression endpoints.

Extracted from ``pfm.main`` so the monolith stops growing. The endpoints lean
on a handful of helpers (``_resolve_factor_specs``, ``_assemble_design``,
``_finite``, ``_jsafe``) that still live in ``pfm.main`` because three
endpoints share them (this one + ``/factors/best``). Those helpers are pulled
in via a lazy import inside each handler so this module can be imported by
``pfm.main`` at start-up without a circular dependency.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
import logging
import os as _os
import time as _time
from typing import Annotated
from urllib.parse import quote_plus

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response

from pfm.analyses import (
    apply_lag,
    apply_pca,
    bootstrap_betas,
    factor_stationarity,
    fit_lasso,
    fit_quantile,
    fit_ridge,
    granger_test,
    oos_split,
    permutation_test,
    regularized_to_estimates,
    rolling_betas,
)
from pfm.attribution import attribute
from pfm.cache import CacheBackend
from pfm.cache_utils import get_cache as get_terminal_cache
from pfm.config import Settings, get_settings
from pfm.dependencies import get_cache, get_factors_dep, get_polymarket_client
from pfm.factors import FactorConfig
from pfm.model import (
    DEFAULT_EPSILON,
    VIF_INF_SENTINEL,
    count_clipping_events,
    delta_logit,
    fit_ols_hac,
    stationarity_tests,
)
from pfm.regression_regime import detect_regime_changes
from pfm.schemas import (
    AttributionRequest,
    AttributionResponse,
    BacktestPoint,
    BootstrapCi,
    ContributionOut,
    DiagnosticsOut,
    FactorContributionOut,
    FactorCoverageOut,
    FactorEstimateOut,
    FactorMetadataOut,
    FactorStationarity,
    FactorTracePoint,
    FitPreviewRequest,
    FitPreviewResponse,
    FitRequest,
    FitResponse,
    GrangerLag,
    GrangerOut,
    LiveSignalOut,
    ModelStatsOut,
    MultitestHint,
    OosOut,
    OosRSquared,
    OosRSquaredSkipped,
    OverfitRiskFlag,
    PcaOut,
    PcaSummary,
    PermutationResult,
    PseudoBacktestOut,
    RegimeChangeOut,
    ResidualAnnotation,
    RollingBetaCiPoint,
    RollingBetaPoint,
    SuggestForTickerItem,
    SuggestForTickerRequest,
    SuggestForTickerResponse,
    TimeSeriesPoint,
)
from pfm.sources.polymarket import PolymarketClient, PolymarketError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["regression"])

# Threshold for "collinear" verdict + auto-prune. VIF >= 5 is the
# classical rule-of-thumb (Kutner et al. 2005, "moderate" cutoff); >=10 is
# the stricter "severe" cutoff. We pick 5 so the verdict surfaces problems
# early instead of waiting for them to become catastrophic.
_VIF_PRUNE_THRESHOLD: float = 5.0
# A regression is "underpowered" below this many obs OR when k factors
# leaves fewer than ~3x degrees of freedom per factor. Both rules from
# Greene "Econometric Analysis", 7e, §4.
_MIN_OBS_FOR_HAC: int = 30


def _build_summary(
    *,
    n_obs: int,
    k_factors: int,
    r2: float,
    n_significant: int,
    high_vif_count: int,
    auto_pruned_count: int,
) -> str:
    """One-line human readout summarising fit quality.

    Examples:
        "5 factors fit with R²=0.18; 2 significant at p<0.05."
        "3 factors fit with R²=0.04; no factor significant; 1 high-VIF warning."
        "Auto-pruned 2 collinear factors; 3 factors fit with R²=0.12; 1 significant."
    """
    parts: list[str] = []
    if auto_pruned_count > 0:
        parts.append(
            f"Auto-pruned {auto_pruned_count} collinear factor"
            f"{'s' if auto_pruned_count != 1 else ''}"
        )
    parts.append(
        f"{k_factors} factor{'s' if k_factors != 1 else ''} fit on {n_obs} obs with R²={r2:.2f}"
    )
    if n_significant > 0:
        parts.append(f"{n_significant} significant at p<0.05")
    else:
        parts.append("no factor significant at p<0.05")
    if high_vif_count > 0:
        parts.append(f"{high_vif_count} high-VIF warning{'s' if high_vif_count != 1 else ''}")
    return "; ".join(parts) + "."


def _classify_verdict(
    *,
    n_obs: int,
    k_factors: int,
    r2_adj: float,
    max_vif: float | None,
    n_significant: int,
) -> str:
    """Single-word fit-quality flag.

    Priority order (worst-first so the most actionable label wins):
      underpowered  : n_obs too small for the chosen k
      collinear     : max VIF >= 5 (interpretation contaminated)
      weak_fit      : R²adj < 0.02 (model explains essentially nothing)
      well_specified: R²adj >= 0.05 + >=1 significant factor + low VIF
      borderline    : middling fit with no major red flag
    """
    if n_obs < _MIN_OBS_FOR_HAC or n_obs <= 3 * max(1, k_factors):
        return "underpowered"
    if max_vif is not None and max_vif >= _VIF_PRUNE_THRESHOLD:
        return "collinear"
    if r2_adj < 0.02:
        return "weak_fit"
    if r2_adj >= 0.05 and n_significant >= 1:
        return "well_specified"
    return "borderline"


def _iterative_prune_collinear(
    X: pd.DataFrame,
    *,
    threshold: float = _VIF_PRUNE_THRESHOLD,
    min_keep: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop the highest-VIF factor iteratively until all VIF < threshold.

    Stops when (a) every remaining VIF is below ``threshold``, (b) only
    ``min_keep`` factors remain (we never prune to zero), or (c) the VIF
    computation degenerates (single column ⇒ VIF is mathematically 1.0).

    Returns the pruned X and the ordered list of dropped factor ids.
    """
    import statsmodels.api as sm  # local import to keep cold-start lean
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    dropped: list[str] = []
    cur = X.copy()
    # Safety bound: never iterate more than k times.
    for _ in range(max(0, X.shape[1] - min_keep)):
        if cur.shape[1] <= min_keep:
            break
        x_const = sm.add_constant(cur, has_constant="add")
        worst_col: str | None = None
        worst_vif = -np.inf
        for i, col in enumerate(x_const.columns):
            if col == "const":
                continue
            try:
                v = float(variance_inflation_factor(x_const.values, i))
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                v = float("inf")
            if not np.isfinite(v):
                v = VIF_INF_SENTINEL
            if v > worst_vif:
                worst_vif = v
                worst_col = col
        if worst_col is None or worst_vif < threshold:
            break
        dropped.append(worst_col)
        cur = cur.drop(columns=[worst_col])
    return cur, dropped


def _next_step_hint(verdict: str) -> str:
    """Return a one-sentence actionable hint for the verdict.

    The frontend displays the string verbatim, so phrasings are tuned to
    sound like a knowledgeable colleague nudging the user toward the most
    productive next move (not a generic stats lecture).
    """
    return {
        "well_specified": ("Try walk-forward OOS validation to confirm signal stability."),
        "weak_fit": ("Window >=120d and check the factor's coverage_pct in factor_metadata."),
        "collinear": ("Use ?prune_collinear=true or switch to ridge."),
        "underpowered": ("Need more observations -- extend date range or relax epsilon."),
        "borderline": ("Consider a longer window or a more targeted factor set."),
    }.get(verdict, "")


def _compute_rolling_betas_with_ci(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    window: int = 60,
    max_points_per_factor: int = 200,
) -> dict[str, list[RollingBetaCiPoint]]:
    """60-day rolling β with HAC 95% CIs per factor, downsampled.

    Walks an OLS+HAC over ``window`` business days and emits per-factor
    points keyed by the END date of each window. Skipped (returns empty
    dict) when n < 90 because a 60-day window leaves no headroom for HAC
    SEs to be stable. To keep the payload small we downsample to at most
    ``max_points_per_factor`` evenly spaced points per factor.

    Performance: each window is a small OLS + Newey-West, which is O(window
    * k^2). For typical (5 factors, 90 obs) the loop takes ~150-300 ms;
    for large fits we downsample by stepping the window stride, not the
    output, so the cost stays bounded.
    """
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import (
        InfeasibleTestError,
        MissingDataError,
    )

    n = len(y)
    if n < 90 or window <= X.shape[1] + 1 or window > n:
        return {}

    # Pick a stride so we emit roughly ``max_points_per_factor`` windows.
    n_windows = n - window + 1
    stride = max(1, int(np.ceil(n_windows / max_points_per_factor)))

    out: dict[str, list[RollingBetaCiPoint]] = {col: [] for col in X.columns}
    X_const = sm.add_constant(X, has_constant="add")

    # Cap the per-window HAC lag to a small constant so a 60-obs window
    # never gets a 50-lag SE (which would be numerically meaningless).
    win_hac_lag = min(5, max(1, int(np.floor(window ** (1 / 3)))))

    for end in range(window, n + 1, stride):
        start = end - window
        try:
            fit = sm.OLS(y.values[start:end], X_const.values[start:end]).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": win_hac_lag},
            )
        except (np.linalg.LinAlgError, ValueError, InfeasibleTestError, MissingDataError):
            continue
        ts_date = y.index[end - 1].date()
        for i, col in enumerate(X_const.columns):
            if col == "const":
                continue
            beta = float(fit.params[i])
            se = float(fit.bse[i]) if i < len(fit.bse) else float("nan")
            if not np.isfinite(beta):
                continue
            if not np.isfinite(se):
                ci_lo = ci_hi = beta
            else:
                ci_lo = beta - 1.96 * se
                ci_hi = beta + 1.96 * se
            out[col].append(
                RollingBetaCiPoint(
                    date=ts_date,
                    beta=beta,
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                )
            )
    # Drop empty factors (e.g. all windows degenerated).
    return {k: v for k, v in out.items() if v}


_MIN_N_FOR_WALK_FORWARD: int = 100


def _walk_forward_oos_r2(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    n_folds: int = 5,
) -> OosRSquared | OosRSquaredSkipped:
    """Walk-forward OOS R² across ``n_folds`` chronological folds.

    Splits the sample into ``n_folds`` contiguous chunks; for fold k the
    train set is the first ``k * fold_size`` obs and the test set is the
    next ``fold_size`` obs. Reports the median test R² to be robust to a
    single bad fold (e.g. an event-driven outlier window).

    Returns an :class:`OosRSquaredSkipped` block (with ``skipped_reason``)
    instead of ``None`` when the helper opts out — the user used to see a
    silent ``null`` when n_obs<100; now they see the explicit reason.
    """
    import statsmodels.api as sm

    n = len(y)
    if n < _MIN_N_FOR_WALK_FORWARD:
        return OosRSquaredSkipped(
            skipped_reason=(f"n_obs={n} < min_n_for_walk_forward={_MIN_N_FOR_WALK_FORWARD}"),
        )
    if X.shape[1] < 1:
        return OosRSquaredSkipped(
            skipped_reason="design matrix has no factor columns",
        )

    n_folds = min(n_folds, max(2, n // 25))
    fold_size = n // (n_folds + 1)
    if fold_size < 5:
        return OosRSquaredSkipped(
            skipped_reason=f"fold_size={fold_size} < 5 (n_obs={n} too small)",
        )

    X_const = sm.add_constant(X, has_constant="add")
    per_fold: list[float] = []
    train_sizes: list[int] = []
    test_sizes: list[int] = []

    for k in range(1, n_folds + 1):
        train_end = k * fold_size
        test_end = min(n, (k + 1) * fold_size)
        if test_end - train_end < 5 or train_end <= X_const.shape[1] + 1:
            continue
        X_train = X_const.iloc[:train_end].values
        y_train = y.iloc[:train_end].values
        X_test = X_const.iloc[train_end:test_end].values
        y_test = y.iloc[train_end:test_end].values
        try:
            fit = sm.OLS(y_train, X_train).fit()
        except (np.linalg.LinAlgError, ValueError):
            continue
        y_hat = X_test @ fit.params
        sst = float(np.sum((y_test - y_test.mean()) ** 2))
        sse = float(np.sum((y_test - y_hat) ** 2))
        if sst <= 0.0:
            continue
        r2 = 1.0 - sse / sst
        if not np.isfinite(r2):
            continue
        per_fold.append(float(r2))
        train_sizes.append(train_end)
        test_sizes.append(test_end - train_end)

    if not per_fold:
        return OosRSquaredSkipped(
            skipped_reason=("all folds degenerate (singular train, zero-variance test, or NaN R²)"),
        )
    return OosRSquared(
        value=float(np.median(per_fold)),
        n_train=int(np.mean(train_sizes)) if train_sizes else 0,
        n_test=int(np.mean(test_sizes)) if test_sizes else 0,
        fold_count=len(per_fold),
        per_fold=per_fold,
    )


def _build_news_links(ticker: str, ts_date: pd.Timestamp) -> list[str]:
    """Build deep-link search URLs for the (ticker, date) pair.

    The frontend renders these as clickable "find news for this day"
    chips under each residual annotation. We don't actually call any
    news API server-side — that would be slow, rate-limited, and the
    user typically wants to see the search results page anyway.
    """
    iso = ts_date.date().isoformat() if hasattr(ts_date, "date") else str(ts_date)
    sym = ticker.upper()
    q = quote_plus(f"{sym} {iso} stock")
    q_news = quote_plus(f'"{sym}" news {iso}')
    return [
        f"https://www.google.com/search?q={q}",
        f"https://www.google.com/search?q={q_news}&tbm=nws",
        f"https://duckduckgo.com/?q={q}&iar=news",
    ]


def _residual_annotations(
    y: pd.Series,
    X: pd.DataFrame,
    residual: pd.Series,
    factor_estimates: list[FactorEstimateOut],
    *,
    top_k: int = 5,
    ticker: str | None = None,
) -> list[ResidualAnnotation]:
    """Top ``top_k`` |residual| dates with per-factor Δlogit·β attribution.

    For each of the worst-fit dates we compute, for every factor in the
    fit, the contribution Δlogit_{i,t} · β_i — i.e. how much that factor
    pushed the prediction on that day. The factor with the largest
    |contribution| is most likely to explain the miss (e.g. a Fed-cuts
    factor on FOMC day). When ``ticker`` is supplied each annotation also
    carries a list of news-search deep-link URLs the UI can render.
    """
    if residual.empty or not factor_estimates:
        return []
    beta_by_id = {est.id: est.beta for est in factor_estimates}
    abs_resid = residual.abs().sort_values(ascending=False)
    annotations: list[ResidualAnnotation] = []
    for ts_date in abs_resid.index[:top_k]:
        if ts_date not in X.index:
            continue
        attribution: dict[str, float] = {}
        for col in X.columns:
            beta = beta_by_id.get(col)
            if beta is None or not np.isfinite(beta):
                continue
            dl = float(X.loc[ts_date, col])
            if not np.isfinite(dl):
                continue
            attribution[col] = float(dl * beta)
        top_factor = None
        if attribution:
            top_factor = max(attribution, key=lambda k: abs(attribution[k]))
        news_links: list[str] = []
        if ticker:
            news_links = _build_news_links(ticker, ts_date)
        annotations.append(
            ResidualAnnotation(
                date=ts_date.date(),
                residual=float(residual.loc[ts_date]),
                magnitude=float(abs_resid.loc[ts_date]),
                factor_attribution=attribution,
                top_factor=top_factor,
                news_links=news_links,
            )
        )
    return annotations


def _factor_correlation_matrix(
    X: pd.DataFrame,
    *,
    max_factors: int = 30,
) -> dict[str, dict[str, float]]:
    """Pearson r between every pair of factor Δlogit series.

    Capped at the first ``max_factors`` columns to keep the payload
    small (O(K^2) growth gets unfriendly past ~30). Returns a nested
    dict ``{fid_a: {fid_b: r, ...}, ...}`` so the UI can render it as a
    heatmap without re-pivoting.
    """
    if X.shape[1] < 2:
        return {}
    cols = list(X.columns[:max_factors])
    sub = X[cols]
    try:
        corr = sub.corr(method="pearson", min_periods=10)
    except (ValueError, TypeError):
        return {}
    out: dict[str, dict[str, float]] = {}
    for a in cols:
        out[a] = {}
        for b in cols:
            v = corr.loc[a, b]
            out[a][b] = float(v) if v is not None and np.isfinite(v) else 0.0
    return out


def _compute_pca_summary(
    X: pd.DataFrame,
    *,
    max_components: int = 5,
    top_loadings_per_pc: int = 5,
) -> PcaSummary | None:
    """Quick PCA on the design matrix using ``numpy.linalg.svd``.

    We center each column to mean 0 and then SVD; the squared singular
    values give the explained-variance ratio. This avoids a sklearn
    dependency and is fast for K <= 30.
    """
    if X.shape[1] < 2 or X.shape[0] < 3:
        return None
    M = X.values.astype(float)
    # Drop rows with NaNs (shouldn't be any after _assemble_design but
    # cheap insurance).
    M = M[~np.isnan(M).any(axis=1)]
    if M.shape[0] < 3:
        return None
    centered = M - M.mean(axis=0, keepdims=True)
    try:
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    var = (s**2) / max(1, M.shape[0] - 1)
    total = float(var.sum())
    if total <= 0.0:
        return None
    n_comp = min(max_components, len(s), X.shape[1])
    ratios = [float(v) for v in (var[:n_comp] / total)]
    cols = list(X.columns)
    top_loadings: dict[int, dict[str, float]] = {}
    for k in range(n_comp):
        loadings = vt[k]  # shape: (K,)
        # Order columns by |loading| desc, keep top N.
        order = np.argsort(-np.abs(loadings))[:top_loadings_per_pc]
        top_loadings[k] = {cols[i]: float(loadings[i]) for i in order if np.isfinite(loadings[i])}
    return PcaSummary(
        n_components=n_comp,
        explained_variance_ratio=ratios,
        top_loadings=top_loadings,
    )


def _compute_live_signal(
    y: pd.Series,
    X: pd.DataFrame,
    factor_estimates: list[FactorEstimateOut],
    alpha: float,
    *,
    low_confidence: bool,
) -> LiveSignalOut | None:
    """Forward-looking signal: ``α + Σ βᵢ · Δlogit_latest_i``.

    Uses the last row of ``X`` (the most recent Δlogit per factor) plus
    the just-fit coefficients to predict the next-period return. The
    standard error comes from the classical OLS prediction variance
    ``√(x* (XᵀX)⁻¹ x*ᵀ · σ²)`` where ``σ²`` is the residual variance and
    ``x*`` includes the intercept term.

    Returns ``None`` when the design matrix is empty, has no usable last
    row (all-NaN), or when ``XᵀX`` is too ill-conditioned to invert.

    Notes
    -----
    The SE captures parameter uncertainty only — it does NOT include the
    next-period innovation σ²(ε). Adding that term would inflate the CI
    by ``σ²`` which already shows up in the residual_std stat; we keep
    the narrower CI here to mirror what ``statsmodels.predict()`` returns
    when ``has_constant=True`` and ``cov_type=HAC``. Callers wanting a
    full prediction-interval should add ``residual_std`` themselves.
    """
    if X.empty or X.shape[1] == 0 or len(y) == 0:
        return None
    last_idx = X.index[-1]
    last_row = X.loc[last_idx]
    # Skip if any factor's last value is NaN — the prediction would be NaN.
    if not np.all(np.isfinite(last_row.values.astype(float))):
        return None

    beta_by_id = {est.id: est.beta for est in factor_estimates}
    cols = list(X.columns)
    contributions = 0.0
    latest_logits: dict[str, float] = {}
    for col in cols:
        v = float(last_row[col])
        b = beta_by_id.get(col, 0.0)
        if not (np.isfinite(v) and np.isfinite(b)):
            continue
        contributions += b * v
        latest_logits[col] = v
    if not np.isfinite(alpha):
        alpha = 0.0
    predicted = float(alpha + contributions)
    if not np.isfinite(predicted):
        return None

    # Classical OLS prediction SE: sqrt(x* (X'X)^-1 x*' · sigma2).
    se = float("nan")
    try:
        import statsmodels.api as sm  # local import keeps cold start lean

        X_const = sm.add_constant(X, has_constant="add")
        XtX = X_const.values.T @ X_const.values
        XtX_inv = np.linalg.pinv(XtX)
        x_star = np.concatenate([[1.0], last_row.values.astype(float)])
        # Residual variance: σ² = RSS / (n - k - 1). Use predict-style
        # residuals = y - Xβ_hat. We rebuild β_hat from the estimates we
        # already have to avoid re-fitting — equivalent for OLS.
        betas = np.array([alpha] + [beta_by_id.get(c, 0.0) for c in cols], dtype=float)
        y_hat = X_const.values @ betas
        resid = y.values - y_hat
        n_obs = len(y)
        dof = max(1, n_obs - X_const.shape[1])
        sigma2 = float(np.sum(resid**2)) / dof
        var_pred = float(x_star @ XtX_inv @ x_star) * sigma2
        if np.isfinite(var_pred) and var_pred >= 0:
            se = float(np.sqrt(var_pred))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        se = float("nan")

    if not np.isfinite(se) or se < 0:
        # Fall back to a degenerate-but-finite SE so the response stays
        # well-typed; UI can still color the prediction.
        se = 0.0
    ci_lo = predicted - 1.96 * se
    ci_hi = predicted + 1.96 * se

    return LiveSignalOut(
        predicted_return=predicted,
        std_err=float(se),
        ci_95_lo=float(ci_lo),
        ci_95_hi=float(ci_hi),
        edge_bp=float(abs(predicted) * 1e4),
        latest_date=last_idx.date(),
        latest_factor_logits={k: float(v) for k, v in latest_logits.items()},
        low_confidence=bool(low_confidence),
    )


def _compute_pseudo_backtest(
    y: pd.Series,
    predicted: pd.Series,
    *,
    transaction_cost_bp: float = 5.0,
    min_obs: int = 30,
) -> PseudoBacktestOut | None:
    """Daily-rebalanced replay of ``sign(predicted)`` over the fit window.

    On each day t, take ``position_t = sign(predicted_t)`` and earn
    ``actual_t · position_t``. When the position changes vs the previous
    day, deduct a flat ``transaction_cost_bp`` of edge as friction.

    Returns ``None`` when ``len(y) < min_obs``.

    Performance
    -----------
    Pure-numpy O(n) loop. For typical (5 factors × 90 obs) this is well
    under 5 ms — the helper does not refit the model.
    """
    if len(y) < min_obs or len(predicted) != len(y):
        return None

    y_vals = y.values.astype(float)
    p_vals = predicted.values.astype(float)
    # Replace NaN with 0 — equivalent to "no signal on that day".
    p_vals = np.where(np.isfinite(p_vals), p_vals, 0.0)
    y_vals = np.where(np.isfinite(y_vals), y_vals, 0.0)
    pos = np.sign(p_vals)
    # Per-side cost as a return-unit fraction. 5 bp -> 0.0005.
    cost = float(transaction_cost_bp) / 1e4

    equity = 1.0
    prev_pos = 0.0
    n_trades = 0
    hits = 0
    points: list[BacktestPoint] = []
    pnl_series = np.zeros(len(y_vals))
    for i, ts in enumerate(y.index):
        position = float(pos[i])
        traded = position != prev_pos
        if traded:
            n_trades += 1
        # Per-trade cost: charge full ``cost`` on a sign flip; charge ``cost``
        # on a flat→position open too. Closing to flat is also a trade.
        trade_cost = cost if traded else 0.0
        pnl = position * y_vals[i] - trade_cost
        pnl_series[i] = pnl
        equity *= 1.0 + pnl
        # Hit-rate counts only days the model took a stance (non-zero pos).
        if position != 0.0 and np.sign(y_vals[i]) == position:
            hits += 1
        points.append(
            BacktestPoint(
                date=ts.date(),
                predicted=float(p_vals[i]),
                actual=float(y_vals[i]),
                position=position,
                pnl=float(pnl),
                equity=float(equity),
            )
        )
        prev_pos = position

    total_return = float(equity - 1.0)
    std_pnl = float(np.std(pnl_series))
    mean_pnl = float(np.mean(pnl_series))
    sharpe = float(np.sqrt(252.0) * mean_pnl / std_pnl) if std_pnl > 0 else 0.0
    # Max drawdown: largest peak-to-trough drop on the equity curve.
    eq_arr = np.array([p.equity for p in points], dtype=float)
    if len(eq_arr) == 0:
        max_dd = 0.0
    else:
        running_max = np.maximum.accumulate(eq_arr)
        dd_series = (eq_arr - running_max) / np.where(running_max > 0, running_max, 1.0)
        max_dd = float(np.min(dd_series))
    # Hit rate denominator is "days the model took a stance".
    n_stance = int(np.sum(pos != 0.0))
    hit_rate = float(hits / n_stance) if n_stance > 0 else 0.0

    return PseudoBacktestOut(
        equity_curve=points,
        total_return=total_return,
        annualized_sharpe=sharpe,
        max_drawdown=max_dd,
        hit_rate=hit_rate,
        n_trades=int(n_trades),
        transaction_cost_bp=float(transaction_cost_bp),
    )


def _compute_factor_contributions(
    y: pd.Series,
    X: pd.DataFrame,
    full_r_squared: float,
) -> list[FactorContributionOut] | None:
    """Leave-one-out R² impact per factor.

    For each column ``i`` we re-fit OLS on ``X.drop(columns=[i])`` and
    record ``Δ_R² = R²(full) - R²(LOO_i)``. Returns the list sorted
    descending by ``delta_r_squared`` plus a ``share_of_explained_r_squared``
    normalised to sum to ≤ 1 across positive contributions.

    Skipped (returns ``None``) when ``X`` has fewer than 2 columns, since
    the LOO refit would be a degenerate constant-only model.

    Performance
    -----------
    O(k) plain OLS fits. For typical (5 factors × 90 obs) this completes
    in well under 25 ms. Uses plain OLS (not HAC) because we only need R²,
    which doesn't depend on the covariance estimator.
    """
    if X.shape[1] < 2 or len(y) < 5:
        return None
    import statsmodels.api as sm

    deltas: list[tuple[str, float]] = []
    for col in X.columns:
        X_loo = X.drop(columns=[col])
        if X_loo.shape[1] == 0:
            continue
        X_const = sm.add_constant(X_loo, has_constant="add")
        try:
            fit = sm.OLS(y.values, X_const.values).fit()
            r2_loo = float(fit.rsquared)
        except (np.linalg.LinAlgError, ValueError):
            r2_loo = float("nan")
        if not np.isfinite(r2_loo):
            r2_loo = 0.0
        delta = float(full_r_squared - r2_loo)
        deltas.append((col, delta))

    if not deltas:
        return None
    # Share = positive_delta_i / sum(positive_deltas). Negative deltas
    # (factor that hurts R²) get zero share so the bar reads cleanly.
    positives = [(c, max(0.0, d)) for c, d in deltas]
    total_pos = sum(d for _, d in positives)
    out: list[FactorContributionOut] = []
    for (col, delta), (_, pos_delta) in zip(deltas, positives, strict=True):
        share = (pos_delta / total_pos) if total_pos > 0 else 0.0
        out.append(
            FactorContributionOut(
                factor_id=col,
                delta_r_squared=float(delta),
                share_of_explained_r_squared=float(max(0.0, min(1.0, share))),
            )
        )
    out.sort(key=lambda c: c.delta_r_squared, reverse=True)
    return out


def _main_helpers() -> tuple:
    """Lazy import of helpers that still live in ``pfm.main``.

    The first call after module load resolves the import (Python caches it);
    subsequent calls are constant-time attribute lookups.
    """
    from pfm import main as _m

    return _m._resolve_factor_specs, _m._assemble_design, _m._finite, _m._jsafe


# ── Rigour pack helpers (2026-05-15) ────────────────────────────────────────

# Themes that we treat as "non-equity" — sports, entertainment, geopolitics.
# When a user picks one of these alongside a regular ticker (NVDA, SPY, …),
# we surface a low-severity "theme_mismatch" flag so they at least notice.
# The list is intentionally small and conservative; we only flag obvious
# mismatches, not cross-asset diversification (e.g. crypto on tech tickers
# is sometimes legitimate research).
_NON_EQUITY_THEMES: frozenset[str] = frozenset({"sports", "entertainment", "celebrity", "culture"})
# Tickers we treat as "equity-like" for the theme-mismatch heuristic. We
# keep this empty by default and apply the heuristic to *every* ticker —
# the override exists for futures/crypto symbols where the heuristic might
# misfire and which are typically passed through factor-themed strategies
# anyway. The heuristic falls back gracefully when the factor's theme
# field is absent or "other".
_BTC_LIKE_PREFIXES: tuple[str, ...] = ("BTC", "ETH", "SOL", "XBT")


def _compute_overfit_risk_flags(
    n_obs: int,
    factor_specs: list,
    factor_meta: dict[str, FactorMetadataOut],
    factor_estimates: list[FactorEstimateOut],
    ticker: str,
) -> list[OverfitRiskFlag]:
    """Build the structured overfit-risk advisories.

    Triggers (priority high → low):

      * ``low_dof``           n_obs / k < 10 (high severity)
      * ``moderate_dof``      n_obs / k < 20 (medium severity)
      * ``high_clipping``     >50 % of factors have clipping_events / n_obs > 20 %
      * ``sign_inconsistent`` 2+ factors with a same theme (e.g. both BTC-bullish)
                              have opposite β signs
      * ``theme_mismatch``    user has a non-sports ticker but a sports/celebrity
                              factor was selected
    """
    flags: list[OverfitRiskFlag] = []
    k = max(1, len(factor_specs))
    ratio = n_obs / k
    if ratio < 10.0:
        flags.append(
            OverfitRiskFlag(
                level="high",
                code="low_dof",
                message=(
                    f"Insufficient observations per factor (n/k={ratio:.1f}, "
                    "recommend ≥10). The model is at high risk of overfitting "
                    "— add more obs or drop factors."
                ),
            )
        )
    elif ratio < 20.0:
        flags.append(
            OverfitRiskFlag(
                level="medium",
                code="moderate_dof",
                message=(
                    f"Borderline observations per factor (n/k={ratio:.1f}, "
                    "recommend ≥20). Treat marginal coefficients with care."
                ),
            )
        )

    # Heavy clipping across many factors masks the Δlogit signal.
    heavy: list[tuple[str, int, int]] = []
    for fid, meta in factor_meta.items():
        n_meta = max(1, meta.n_obs)
        clip_pct = meta.clipping_events / n_meta
        if clip_pct > 0.20:
            heavy.append((fid, meta.clipping_events, meta.n_obs))
    if len(heavy) > 0 and len(heavy) >= max(1, int(0.5 * len(factor_meta))):
        worst = max(heavy, key=lambda t: t[1] / max(1, t[2]))
        worst_pct = int(round(100.0 * worst[1] / max(1, worst[2])))
        flags.append(
            OverfitRiskFlag(
                level="medium",
                code="high_clipping",
                message=(
                    f"Factor {worst[0]!r} has {worst_pct}% clipped — Δlogit "
                    f"signal is mostly noise. {len(heavy)} of {len(factor_meta)} "
                    "factors exceed 20% clipping; consider lowering epsilon."
                ),
            )
        )

    # Sign-inconsistency within a theme (e.g. two BTC-bullish factors with
    # opposite β signs is a red flag for spurious correlation).
    by_theme: dict[str, list[tuple[str, float]]] = {}
    theme_by_id = {fc.id: getattr(fc, "theme", "other") for fc in factor_specs}
    for est in factor_estimates:
        theme = theme_by_id.get(est.id, "other")
        if theme in {"other", "", None}:
            continue
        if not np.isfinite(est.beta):
            continue
        by_theme.setdefault(theme, []).append((est.id, est.beta))
    for theme, items in by_theme.items():
        if len(items) < 2:
            continue
        signs = {1 if b > 0 else -1 if b < 0 else 0 for _, b in items}
        signs.discard(0)
        if len(signs) > 1:
            ids = ", ".join(fid for fid, _ in items)
            flags.append(
                OverfitRiskFlag(
                    level="medium",
                    code="sign_inconsistent",
                    message=(
                        f"Factors in theme {theme!r} have opposite β signs "
                        f"({ids}). They should respond in the same economic "
                        "direction — check for spurious correlation."
                    ),
                )
            )

    # Theme mismatch: a non-sports ticker with a sports/entertainment factor.
    # Catalog is currently mis-themed for ~60% of sports markets (F1 drivers
    # are "other", Kansas Royals is "politics", Mayweather/Pacquiao is
    # "weather" — see audit 2026-05-15). Fall back to ID/name keyword
    # detection so the heuristic still fires when the theme field is wrong.
    _SPORTS_KW: tuple[str, ...] = (
        "nfl",
        "nba",
        "mlb",
        "nhl",
        "ncaa",
        "epl",
        "royals",
        "yankees",
        "dodgers",
        "athletics",
        "giants",
        "rangers",
        "champions_league",
        "champion_series",
        "champions_series",
        "world_series",
        "world_cup",
        "super_bowl",
        "superbowl",
        "playoff",
        "playoffs",
        "finals",
        "final",
        "wimbledon",
        "us_open",
        "french_open",
        "australian_open",
        "f1_drivers",
        "_f1_",
        "formula_1",
        "formula1",
        "psg",
        "barcelona",
        "real_madrid",
        "liverpool",
        "arsenal",
        "chelsea",
        "man_united",
        "manchester",
        "tottenham",
        "spurs",
        "soccer",
        "football",
        "basketball",
        "baseball",
        "hockey",
        "boxing",
        "ufc",
        "mayweather",
        "pacquiao",
        "tennis",
        "golf",
        "olympics",
        "olympic",
        "mvp",
        "_win_the_2026_",
        "_win_the_2027_",
        "_be_the_2026_",
        "_be_the_2027_",
    )
    _ENTERTAINMENT_KW: tuple[str, ...] = (
        "beyonce",
        "swift",
        "rihanna",
        "kanye",
        "musk_dating",
        "grammy",
        "oscar",
        "razzie",
        "emmy",
        "eurovision",
        "gta_vi",
        "gta_6",
        "fortnite",
        "minecraft",
        "netflix_show",
        "marvel",
        "pixar",
        "disney_film",
    )
    sym = (ticker or "").upper()
    is_crypto_like = any(sym.startswith(pfx) for pfx in _BTC_LIKE_PREFIXES)
    if not is_crypto_like:
        sport_factors = []
        for fc in factor_specs:
            fid = (getattr(fc, "id", "") or "").lower()
            name = (getattr(fc, "name", "") or "").lower()
            theme = (getattr(fc, "theme", "") or "").lower()
            haystack = f"{fid} {name}"
            kw_hit = any(kw in haystack for kw in _SPORTS_KW) or any(
                kw in haystack for kw in _ENTERTAINMENT_KW
            )
            theme_hit = theme in _NON_EQUITY_THEMES
            if kw_hit or theme_hit:
                sport_factors.append(fc.id)
        if sport_factors:
            flags.append(
                OverfitRiskFlag(
                    level="low",
                    code="theme_mismatch",
                    message=(
                        f"Non-equity-themed factors selected for {sym!r}: "
                        f"{', '.join(sport_factors)}. These themes "
                        "(sports/celebrity/entertainment) rarely have a "
                        "structural link to equity returns — likely overfit."
                    ),
                )
            )
    return flags


def _build_multitest_hint(tests_this_session: int) -> MultitestHint:
    """Bonferroni-style α/N readout from the X-Session-Test-Count header.

    See :class:`MultitestHint` for the rationale on Bonferroni vs BH-FDR
    in this context.
    """
    n = max(1, int(tests_this_session))
    threshold = 0.05 / n
    if n == 1:
        msg = (
            "First test of the session — α=0.05 applies as usual. The "
            "X-Session-Test-Count header lets the client tell the server "
            "how many fits have been run so far."
        )
    else:
        msg = (
            f"You've run {n} tests this session. Your BH-FDR adjusted "
            f"threshold is α/N = 0.05/{n} ≈ {threshold:.4f}. Treat any "
            f"p-value above {threshold:.4f} as not significant after "
            "multiple-testing correction."
        )
    return MultitestHint(
        tests_this_session=n,
        bh_q_threshold=float(threshold),
        message=msg,
    )


@router.post(
    "/fit",
    response_model=FitResponse,
    description=(
        "Fit OLS+HAC factor model of stock returns on prediction-market "
        "Δlogit factors. The optional ``X-Session-Test-Count`` request "
        "header (integer ≥1, default 1) lets the client tell the server "
        "how many fits have been run in this session — the server echoes "
        "back a Bonferroni-style α/N threshold in the ``multitest_hint`` "
        "field and an ``X-Session-Test-Hint`` response header."
    ),
)
def fit_endpoint(
    body: FitRequest,
    response: Response,
    epsilon: Annotated[float, Query(gt=0.0, lt=0.5)] = DEFAULT_EPSILON,
    prune_collinear: Annotated[
        bool,
        Query(
            description=(
                "If true, iteratively drop the factor with the highest VIF "
                "until every remaining VIF is < 5. Dropped ids are surfaced "
                "in the response under `auto_pruned`. Useful when the user "
                "throws 8+ correlated factors at the model and wants the "
                "server to keep only the identifiable subset."
            ),
        ),
    ] = False,
    x_session_test_count: Annotated[
        int | None,
        Header(
            alias="X-Session-Test-Count",
            description=(
                "Optional client-supplied counter for the number of /fit "
                "calls already made in this session. Used to compute the "
                "Bonferroni-style α/N threshold returned in "
                "``multitest_hint``. Defaults to 1 when absent."
            ),
        ),
    ] = None,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> FitResponse:
    _resolve_factor_specs, _assemble_design, _finite, _jsafe = _main_helpers()
    factor_specs = _resolve_factor_specs(body.factors, body.custom_factors, factors)
    if not factor_specs:
        raise HTTPException(status_code=400, detail="provide at least one factor")

    y, X, raw_prices = _assemble_design(
        body.ticker,
        factor_specs,
        body.start,
        body.end,
        epsilon,
        body.return_type,
        poly,
        cache,
        settings,
        alignment=body.alignment,
        residualize_market=getattr(body, "residualize_market", False),
    )
    if len(y) <= len(factor_specs) + 1:
        raise HTTPException(
            status_code=422,
            detail=f"too few overlapping observations ({len(y)}) to fit {len(factor_specs)} factors",
        )

    # ── Defensive diagnostics: per-factor metadata + warnings ────────────
    # Captured BEFORE preprocessing (lag/PCA) so the metadata reflects the
    # raw upstream series, not the (possibly empty) post-transform view.
    warnings_list: list[str] = []
    factor_meta: dict[str, FactorMetadataOut] = {}
    n_obs_pre_lag = len(y)
    total_clipping = 0
    for fc in factor_specs:
        raw = raw_prices.get(fc.id, pd.Series(dtype=float))
        clip_count = count_clipping_events(raw, epsilon=epsilon) if fc.is_probability else 0
        total_clipping += clip_count
        factor_meta[fc.id] = FactorMetadataOut(
            is_probability=fc.is_probability,
            source=fc.source,
            n_obs=int(len(raw)),
            clipping_events=int(clip_count),
        )
        if clip_count > max(5, int(0.10 * len(raw))):
            warnings_list.append(
                f"factor {fc.id!r}: {clip_count} clipping events at "
                f"epsilon={epsilon} ({clip_count / max(1, len(raw)):.0%} of obs); "
                "consider lowering epsilon or excluding the factor near resolution."
            )
    if n_obs_pre_lag < 30:
        warnings_list.append(
            f"only {n_obs_pre_lag} overlapping observations; "
            "regression statistics are unreliable for n < 30."
        )

    # ── Optional preprocessing: lag factors / PCA ──
    pca_out: PcaOut | None = None
    if body.lag and body.lag > 0:
        X = apply_lag(X, body.lag)
        common = X.index.intersection(y.index)
        y = y.loc[common]
        X = X.loc[common]
        if len(y) <= X.shape[1] + 1:
            raise HTTPException(
                status_code=422,
                detail=f"after lag={body.lag}, only {len(y)} obs remain — too few",
            )
    if body.pca_components:
        X, pca_info = apply_pca(X, body.pca_components)
        pca_out = PcaOut(
            components=pca_info.components,
            explained_variance=pca_info.explained_variance,
            loadings=pca_info.loadings,
        )

    # ── Optional auto-prune of collinear factors ────────────────────────
    # When the user sets ?prune_collinear=true we iteratively drop the
    # highest-VIF factor until every remaining VIF is below 5. The dropped
    # ids are surfaced via ``auto_pruned`` so the caller knows which
    # factors were removed (e.g. so the UI can grey them out in the
    # selected-factors chip list).
    auto_pruned: list[str] = []
    if prune_collinear and X.shape[1] >= 2:
        X_new, auto_pruned = _iterative_prune_collinear(X)
        if auto_pruned:
            X = X_new
            warnings_list.append(
                f"auto-pruned {len(auto_pruned)} collinear factor"
                f"{'s' if len(auto_pruned) != 1 else ''} "
                f"(VIF >= {_VIF_PRUNE_THRESHOLD:.0f}): {', '.join(auto_pruned)}. "
                "Re-run with prune_collinear=false to see the full set."
            )

    if body.hac_lag is not None and body.regression == "hac" and body.hac_lag >= len(y) - 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"hac_lag={body.hac_lag} too large for n_obs={len(y)}; "
                "must satisfy hac_lag < n_obs - 1"
            ),
        )

    # ── Core regression: OLS+HAC, OLS plain, ridge, lasso, or quantile ──
    factor_estimates_out: list[FactorEstimateOut]
    model_stats_out: ModelStatsOut
    diag_out: DiagnosticsOut
    predicted_arr: np.ndarray
    residual_arr: np.ndarray

    if body.regression in ("ols", "hac"):
        fit = fit_ols_hac(y, X, hac_lag=body.hac_lag, regression=body.regression)
        predicted_arr = fit.fitted.fittedvalues
        residual_arr = (y.values - predicted_arr).astype(float)
        factor_estimates_out = [
            FactorEstimateOut(
                id=est.factor_id,
                beta=est.beta,
                std_err=est.std_err,
                t_stat=est.t_stat,
                p_value=est.p_value,
                ci_low=est.ci_low,
                ci_high=est.ci_high,
            )
            for est in fit.factors
        ]
        model_stats_out = ModelStatsOut(
            alpha=fit.stats.alpha,
            r_squared=fit.stats.r_squared,
            r_squared_adj=fit.stats.r_squared_adj,
            f_stat=fit.stats.f_stat,
            f_pvalue=fit.stats.f_pvalue,
            residual_std=fit.stats.residual_std,
        )
        diag_out = DiagnosticsOut(
            vif=fit.diagnostics.vif,
            durbin_watson=fit.diagnostics.durbin_watson,
            hac_lag=fit.diagnostics.hac_lag,
            adf_stat=_finite(fit.diagnostics.adf_stat),
            adf_pvalue=_finite(fit.diagnostics.adf_pvalue),
            kpss_stat=_finite(fit.diagnostics.kpss_stat),
            kpss_pvalue=_finite(fit.diagnostics.kpss_pvalue),
        )
    elif body.regression in ("ridge", "lasso"):
        rfit = fit_ridge(y, X) if body.regression == "ridge" else fit_lasso(y, X)
        boot_iter = body.bootstrap_iters if body.bootstrap_iters else 300
        boot_ci = bootstrap_betas(y, X, n_iter=boot_iter)
        ests, stats = regularized_to_estimates(rfit, bootstrap_ci=boot_ci)
        factor_estimates_out = [
            FactorEstimateOut(
                id=e.factor_id,
                beta=_jsafe(e.beta),
                std_err=_jsafe(e.std_err),
                t_stat=_jsafe(e.t_stat),
                p_value=_jsafe(e.p_value),
                ci_low=_jsafe(e.ci_low),
                ci_high=_jsafe(e.ci_high),
            )
            for e in ests
        ]
        model_stats_out = ModelStatsOut(
            alpha=stats.alpha,
            r_squared=stats.r_squared,
            r_squared_adj=stats.r_squared_adj,
            f_stat=_jsafe(stats.f_stat),
            f_pvalue=_jsafe(stats.f_pvalue),
            residual_std=stats.residual_std,
        )
        predicted_arr = rfit.fitted_values
        residual_arr = rfit.residuals
        st = stationarity_tests(residual_arr)
        diag_out = DiagnosticsOut(
            vif=dict.fromkeys(X.columns, 1.0),
            durbin_watson=float("nan"),
            hac_lag=0,
            adf_pvalue=_finite(st["adf_pvalue"]),
            adf_stat=_finite(st["adf_stat"]),
            kpss_pvalue=_finite(st["kpss_pvalue"]),
            kpss_stat=_finite(st["kpss_stat"]),
        )
    else:  # quantile
        qfit = fit_quantile(y, X, q=body.quantile)
        factor_estimates_out = [
            FactorEstimateOut(
                id=col,
                beta=qfit.beta[col],
                std_err=qfit.std_err[col],
                t_stat=qfit.t_stat[col],
                p_value=qfit.p_value[col],
                ci_low=qfit.beta[col] - 1.96 * qfit.std_err[col],
                ci_high=qfit.beta[col] + 1.96 * qfit.std_err[col],
            )
            for col in X.columns
        ]
        n_q = len(y)
        k_q = X.shape[1]
        predicted_arr = qfit.fitted_values
        residual_arr = qfit.residuals
        sst = float(np.sum((y.values - y.values.mean()) ** 2))
        sse = float(np.sum(residual_arr**2))
        r2_q = 1 - sse / sst if sst > 0 else 0.0
        r2adj_q = 1 - (1 - r2_q) * (n_q - 1) / max(1, n_q - k_q - 1)
        model_stats_out = ModelStatsOut(
            alpha=qfit.intercept,
            r_squared=r2_q,
            r_squared_adj=r2adj_q,
            f_stat=float("nan"),
            f_pvalue=float("nan"),
            residual_std=float(np.sqrt(np.mean(residual_arr**2))),
        )
        st = stationarity_tests(residual_arr)
        diag_out = DiagnosticsOut(
            vif=dict.fromkeys(X.columns, 1.0),
            durbin_watson=float("nan"),
            hac_lag=0,
            adf_pvalue=_finite(st["adf_pvalue"]),
            adf_stat=_finite(st["adf_stat"]),
            kpss_pvalue=_finite(st["kpss_pvalue"]),
            kpss_stat=_finite(st["kpss_stat"]),
        )

    predicted = pd.Series(predicted_arr, index=y.index)
    residual = pd.Series(residual_arr, index=y.index)

    ts: list[TimeSeriesPoint] = []
    for ts_date in y.index:
        ts.append(
            TimeSeriesPoint(
                date=ts_date.date(),
                observed=float(y.loc[ts_date]),
                predicted=float(predicted.loc[ts_date]),
                residual=float(residual.loc[ts_date]),
                factor_prices={
                    fid: float(raw_prices[fid].get(ts_date, float("nan"))) for fid in X.columns
                },
                factor_delta_logits={fid: float(X.loc[ts_date, fid]) for fid in X.columns},
            )
        )

    factor_traces = {
        fid: [FactorTracePoint(date=d.date(), price=float(p)) for d, p in series.items()]
        for fid, series in raw_prices.items()
    }

    oos_out: OosOut | None = None
    if body.oos_test_fraction > 0:
        oos = oos_split(y, X, body.oos_test_fraction)
        if oos is not None:
            oos_out = OosOut(
                train_n=oos.train_n,
                test_n=oos.test_n,
                train_r2=oos.train_r2,
                test_r2=oos.test_r2,
                test_dates=[d.date() for d in oos.test_dates],
                test_observed=oos.test_observed,
                test_predicted=oos.test_predicted,
            )

    bootstrap_out: list[BootstrapCi] | None = None
    if body.bootstrap_iters > 0 and body.regression in ("ols", "hac"):
        boot = bootstrap_betas(y, X, n_iter=body.bootstrap_iters)
        bootstrap_out = [
            BootstrapCi(
                factor_id=fid,
                ci_low=v["ci_low"],
                ci_high=v["ci_high"],
                mean=v["mean"],
                std=v["std"],
            )
            for fid, v in boot.items()
        ]

    rolling_out: list[RollingBetaPoint] | None = None
    if body.rolling_window:
        rb = rolling_betas(y, X, window=body.rolling_window)
        rolling_out = [
            RollingBetaPoint(
                date=ts_date.date(),
                betas={c: float(rb.loc[ts_date, c]) for c in rb.columns},
            )
            for ts_date in rb.index
        ]

    granger_out: list[GrangerOut] | None = None
    if body.granger_max_lag > 0:
        granger_out = []
        for col in X.columns:
            res = granger_test(y, X[col], max_lag=body.granger_max_lag)
            granger_out.append(
                GrangerOut(
                    factor_id=col,
                    by_lag=[
                        GrangerLag(lag=lag, f_stat=v["f_stat"], p_value=v["p_value"])
                        for lag, v in sorted(res.items())
                    ],
                )
            )

    perm_out: PermutationResult | None = None
    if body.permutation_iters > 0:
        pr = permutation_test(y, X, n_iters=body.permutation_iters)
        if pr["n_iters_completed"] > 0:
            perm_out = PermutationResult(**pr)  # type: ignore[arg-type]

    fs = factor_stationarity(X)
    fac_stat_out: list[FactorStationarity] | None = [
        FactorStationarity(
            factor_id=fid,
            adf_pvalue=_finite(v.get("adf_pvalue")),
            kpss_pvalue=_finite(v.get("kpss_pvalue")),
        )
        for fid, v in fs.items()
    ]

    high_vif = [(fid, v) for fid, v in (diag_out.vif or {}).items() if v is not None and v >= 100.0]
    if high_vif:
        details = ", ".join(f"{fid} (VIF={v:.0f})" for fid, v in high_vif)
        if any(v >= VIF_INF_SENTINEL for _, v in high_vif):
            warnings_list.append(
                f"perfectly collinear factors detected: {details}. "
                "VIF clamped to sentinel; betas/SEs are unreliable. "
                "Drop or merge the duplicate factor before interpreting."
            )
        else:
            warnings_list.append(
                f"high collinearity (VIF >= 100): {details}. "
                "Confidence intervals are inflated; consider reducing the factor set."
            )

    # ── Summary / verdict / top-significant — server-side so all clients
    # share the same readout instead of re-computing it in JS.
    vif_values = [v for v in (diag_out.vif or {}).values() if v is not None and np.isfinite(v)]
    max_vif = max(vif_values) if vif_values else None
    high_vif_count = sum(1 for v in vif_values if v >= _VIF_PRUNE_THRESHOLD)
    significant_factors = [
        f for f in factor_estimates_out if f.p_value is not None and f.p_value < 0.05
    ]
    significant_factors.sort(
        key=lambda f: abs(f.t_stat) if f.t_stat is not None and np.isfinite(f.t_stat) else 0.0,
        reverse=True,
    )
    top_significant = [f.id for f in significant_factors]
    summary_text = _build_summary(
        n_obs=len(y),
        k_factors=len(factor_estimates_out),
        r2=model_stats_out.r_squared,
        n_significant=len(significant_factors),
        high_vif_count=high_vif_count,
        auto_pruned_count=len(auto_pruned),
    )
    verdict = _classify_verdict(
        n_obs=len(y),
        k_factors=len(factor_estimates_out),
        r2_adj=model_stats_out.r_squared_adj,
        max_vif=max_vif,
        n_significant=len(significant_factors),
    )

    n_obs_used = len(y)
    n_obs_dropped = max(0, n_obs_pre_lag - n_obs_used)

    # ── Always-on enrichments (additive). Each helper guards itself on
    # n_obs / k so the cost stays bounded for short windows.
    rolling_betas_ci = _compute_rolling_betas_with_ci(y, X)
    oos_r_squared = _walk_forward_oos_r2(y, X)
    residual_annotations = _residual_annotations(
        y,
        X,
        residual,
        factor_estimates_out,
        top_k=5,
        ticker=body.ticker,
    )
    factor_correlation_matrix = _factor_correlation_matrix(X)
    pca_summary = _compute_pca_summary(X)
    next_step_hint = _next_step_hint(verdict)

    # ── Rigour pack: overfit flags, multitest hint, regime changes ────────
    # All three are pure functions of the just-computed regression artifacts
    # so they cost a few ms total. The regime detector is the only one with
    # non-trivial work (extra OLS fits on sub-windows) — guarded by n>=60.
    overfit_risk_flags = _compute_overfit_risk_flags(
        n_obs=len(y),
        factor_specs=factor_specs,
        factor_meta=factor_meta,
        factor_estimates=factor_estimates_out,
        ticker=body.ticker,
    )
    n_session_tests = max(1, int(x_session_test_count or 1))
    multitest_hint = _build_multitest_hint(n_session_tests)
    # HTTP headers are ASCII-only (latin-1); use plain "alpha" instead of α.
    response.headers["X-Session-Test-Hint"] = (
        f"If you've run {n_session_tests} tests this session, your BH-FDR "
        f"adjusted threshold is alpha/N = {multitest_hint.bh_q_threshold:.4f}."
    )
    try:
        regime_changes_raw = detect_regime_changes(y, X)
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.debug("regime_changes skipped: %s", e)
        regime_changes_raw = []
    regime_changes_out = [
        RegimeChangeOut(
            factor_id=rc.factor_id,
            breakpoint_date=rc.breakpoint_date,
            pre_beta=rc.pre_beta,
            post_beta=rc.post_beta,
            sign_flipped=rc.sign_flipped,
            chow_stat=rc.chow_stat,
            p_value=rc.p_value,
        )
        for rc in regime_changes_raw
    ]

    # ── WOW features: tradeable signal, pseudo-backtest, LOO R² ──────────
    # All three are computed from the design matrix + just-fit coefficients
    # — no upstream fetches. Each helper returns ``None`` when its inputs
    # aren't sufficient (empty X, n_obs < 30, k < 2) so the response stays
    # well-typed for downstream JSON encoders.
    live_signal: LiveSignalOut | None = None
    pseudo_backtest: PseudoBacktestOut | None = None
    factor_contributions: list[FactorContributionOut] | None = None
    try:
        live_signal = _compute_live_signal(
            y,
            X,
            factor_estimates_out,
            model_stats_out.alpha,
            low_confidence=verdict in {"weak_fit", "underpowered"},
        )
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.debug("live_signal skipped: %s", e)
    try:
        pseudo_backtest = _compute_pseudo_backtest(y, predicted)
    except (ValueError, FloatingPointError) as e:
        logger.debug("pseudo_backtest skipped: %s", e)
    try:
        factor_contributions = _compute_factor_contributions(
            y,
            X,
            model_stats_out.r_squared,
        )
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.debug("factor_contributions skipped: %s", e)

    # Surface lag-induced row drops as a warning when they're material
    # (>=10% of the pre-lag sample). Without this the frontend showed
    # a smaller `n_obs` than the date window implied with no explanation
    # of where the rows went.
    if n_obs_dropped > 0 and n_obs_pre_lag > 0 and n_obs_dropped / n_obs_pre_lag >= 0.10:
        warnings_list.append(
            f"{n_obs_dropped} of {n_obs_pre_lag} observations "
            f"({n_obs_dropped / n_obs_pre_lag:.0%}) were dropped by the "
            f"lag={body.lag} transform. Consider reducing `lag` or widening "
            "the window if you need more samples."
        )

    return FitResponse(
        ticker=body.ticker,
        n_obs=len(y),
        start=body.start,
        end=body.end,
        epsilon=epsilon,
        return_type=body.return_type,
        regression=body.regression,
        alignment=body.alignment,
        lag=body.lag,
        pca=pca_out,
        model=model_stats_out,
        factors=factor_estimates_out,
        diagnostics=diag_out,
        time_series=ts,
        factor_traces=factor_traces,
        n_obs_used=n_obs_used,
        n_obs_dropped=n_obs_dropped,
        clipping_events=int(total_clipping),
        warnings=warnings_list,
        factor_metadata=factor_meta,
        summary=summary_text,
        verdict=verdict,
        top_significant=top_significant,
        auto_pruned=auto_pruned,
        oos=oos_out,
        bootstrap=bootstrap_out,
        rolling_betas=rolling_out,
        granger=granger_out,
        factor_stationarity=fac_stat_out,
        permutation=perm_out,
        rolling_betas_ci=rolling_betas_ci,
        oos_r_squared=oos_r_squared,
        residual_annotations=residual_annotations,
        factor_correlation_matrix=factor_correlation_matrix,
        pca_summary=pca_summary,
        next_step_hint=next_step_hint,
        live_signal=live_signal,
        pseudo_backtest=pseudo_backtest,
        factor_contributions=factor_contributions,
        overfit_risk_flags=overfit_risk_flags,
        multitest_hint=multitest_hint,
        regime_changes=regime_changes_out,
    )


@router.post("/fit/preview", response_model=FitPreviewResponse)
def fit_preview_endpoint(
    body: FitPreviewRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> FitPreviewResponse:
    """Fast pre-flight check for /fit.

    Resolves factors, fetches equity returns + factor histories, joins them,
    and reports per-factor coverage + joint n_obs. Does NOT run the regression
    — meant to be called as the user is composing a /fit request so they can
    see "you'll get 187 obs" before clicking Run.

    Same cache as /fit, so a preview followed by a full /fit pays the upstream
    cost once.
    """
    _resolve_factor_specs, _assemble_design, _, _ = _main_helpers()
    factor_specs = _resolve_factor_specs(body.factors, body.custom_factors, factors)
    if not factor_specs:
        raise HTTPException(status_code=400, detail="provide at least one factor")

    # Re-use the production design assembler so cache + alignment behave
    # identically to what /fit would see.
    y, _X, raw_prices = _assemble_design(
        body.ticker,
        factor_specs,
        body.start,
        body.end,
        DEFAULT_EPSILON,
        body.return_type,
        poly,
        cache,
        settings,
        alignment=body.alignment,
        residualize_market=False,
    )

    coverage: list[FactorCoverageOut] = []
    coverage_map: dict[str, FactorCoverageOut] = {}
    warnings_list: list[str] = []
    expected_days = max(1, (body.end - body.start).days)
    start_ts = pd.Timestamp(body.start, tz="UTC").normalize()
    end_ts = pd.Timestamp(body.end, tz="UTC").normalize()
    for fc in factor_specs:
        raw = raw_prices.get(fc.id, pd.Series(dtype=float))
        finite = raw.dropna()
        # n_obs_in_window = obs that survive the [start, end] filter — this
        # is what the /fit inner-join will see. n_obs_available is the raw
        # source coverage (could exceed the window if the source over-fetched).
        in_window = (
            finite[(finite.index >= start_ts) & (finite.index <= end_ts)] if len(finite) else finite
        )
        n_in_window = int(len(in_window))
        n_available = int(len(finite))
        first = in_window.index.min().date() if n_in_window else None
        last = in_window.index.max().date() if n_in_window else None
        # Coverage = in-window obs / requested days, capped at 1.
        cov = min(1.0, n_in_window / float(expected_days)) if expected_days else 0.0
        item = FactorCoverageOut(
            factor_id=fc.id,
            n_obs=n_in_window,
            n_obs_available=n_available,
            n_obs_in_window=n_in_window,
            first_date=first,
            last_date=last,
            coverage_pct=cov,
            is_probability=fc.is_probability,
            source=fc.source,
        )
        coverage.append(item)
        coverage_map[fc.id] = item
        if cov < 0.30 and n_in_window > 0:
            warnings_list.append(
                f"factor {fc.id!r}: only {cov:.0%} of the requested window has data "
                f"({first} → {last}). Consider tightening the window or dropping this factor."
            )
        if n_in_window == 0:
            warnings_list.append(
                f"factor {fc.id!r}: no data in the requested window. Will cause /fit to 502."
            )

    joint_n_obs = int(len(y))
    eq_first = y.index.min().date() if joint_n_obs else None
    eq_last = y.index.max().date() if joint_n_obs else None

    # Suggest a tighter window covering all factors + equity to maximise n.
    starts = [c.first_date for c in coverage if c.first_date is not None]
    ends = [c.last_date for c in coverage if c.last_date is not None]
    rec_start = max(starts) if starts else None
    rec_end = min(ends) if ends else None
    if eq_first is not None and rec_start is not None and rec_start < eq_first:
        rec_start = eq_first
    if eq_last is not None and rec_end is not None and rec_end > eq_last:
        rec_end = eq_last
    if rec_start is not None and rec_end is not None and rec_start >= rec_end:
        rec_start = rec_end = None

    if joint_n_obs < 30:
        warnings_list.append(
            f"only {joint_n_obs} joint observations after alignment. /fit will run "
            "but standard errors are unreliable for n < 30."
        )
    if joint_n_obs <= len(factor_specs) + 1:
        warnings_list.append(
            f"insufficient observations ({joint_n_obs}) for "
            f"{len(factor_specs)} factors. /fit will return 422 — reduce factors or "
            "widen the window."
        )

    return FitPreviewResponse(
        ticker=body.ticker,
        start=body.start,
        end=body.end,
        equity_n_obs=joint_n_obs,
        equity_first_date=eq_first,
        equity_last_date=eq_last,
        factor_coverage=coverage,
        joint_n_obs=joint_n_obs,
        joint_window_obs=joint_n_obs,
        factor_coverage_map=coverage_map,
        predicted_window_n_obs=joint_n_obs,
        min_recommended_obs=30,
        warnings=warnings_list,
        recommended_start=rec_start,
        recommended_end=rec_end,
    )


@router.post("/attribution", response_model=AttributionResponse)
def attribution_endpoint(
    body: AttributionRequest,
    epsilon: Annotated[float, Query(gt=0.0, lt=0.5)] = DEFAULT_EPSILON,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> AttributionResponse:
    _resolve_factor_specs, _assemble_design, _, _ = _main_helpers()
    factor_specs = _resolve_factor_specs(body.factors, body.custom_factors, factors)
    if not factor_specs:
        raise HTTPException(status_code=400, detail="provide at least one factor")

    target = pd.Timestamp(body.date, tz="UTC").normalize()
    y, X, _raw = _assemble_design(
        body.ticker,
        factor_specs,
        body.start,
        body.end,
        epsilon,
        body.return_type,
        poly,
        cache,
        settings,
        alignment=body.alignment,
        residualize_market=getattr(body, "residualize_market", False),
    )
    if target not in y.index:
        raise HTTPException(
            status_code=404,
            detail=f"date {body.date.isoformat()} not in fitted window after alignment",
        )
    if len(y) <= len(factor_specs) + 1:
        raise HTTPException(
            status_code=422,
            detail=f"too few overlapping observations ({len(y)}) to fit {len(factor_specs)} factors",
        )

    fit = fit_ols_hac(y, X, regression=body.regression)
    attr = attribute(fit, y, X, target)
    return AttributionResponse(
        date=body.date,
        observed_return=attr.observed_return,
        predicted_return=attr.predicted_return,
        residual=attr.residual,
        contributions=[
            ContributionOut(
                id=c.factor_id,
                delta_logit=c.delta_logit,
                beta=c.beta,
                contribution=c.contribution,
            )
            for c in attr.contributions
        ],
    )


# ── /factors/suggest-for-ticker — smart-factor-picker ────────────────────
# Cached for 1 h per (ticker, lookback_days, top_k, min_n_obs) so the UI
# can repeatedly open the picker without re-running the K-factor scan.
#
# Two-tier cache (post 2026-05-15 perf fix):
#   L1 = process-local TerminalCache (microsecond hits)
#   L2 = Redis (cross-worker so 4 gunicorn workers share the result)
#
# Stampede protection:
#   When L1+L2 are cold, only ONE worker (the SETNX winner) actually runs
#   the 1 200-factor scan. The other 3 workers poll L2 every 250 ms until
#   the leader writes the answer (or the scan budget expires).
#
# Polymarket 429 handling:
#   ``_cached_factor_history`` lifts the underlying request error directly,
#   so a single rate-limit blip can silently skip a factor. Inside the
#   scan we wrap the per-factor fetch with one retry after a 1.5 s sleep
#   on 429 / 503 / 504 — second 429 logs at INFO and skips that factor
#   only (the rest of the scan keeps going).
_SUGGEST_FOR_TICKER_TTL_S: int = 3600
# Cross-worker L2 key prefix — distinct from the in-process bucket key so
# Redis dumps stay greppable.
_SUGGEST_L2_PREFIX: str = "pfm:suggest_for_ticker"
# Stampede lock TTL: must comfortably exceed the longest cold scan we've
# observed (~30 s wall clock with concurrency 30 against 1 200 factors).
_SUGGEST_LOCK_TTL_S: int = 120
_SUGGEST_LOCK_PREFIX: str = "pfm:suggest_lock"
# Loser wait budget — slightly above expected cold-scan wall clock so the
# losers see the L2 write rather than fall back to a duplicate scan. With
# the prewarmed 200-factor warm-up done at app start, scans of un-warmed
# tickers still finish within this window.
_SUGGEST_LOCK_WAIT_S: float = 60.0
_SUGGEST_LOCK_POLL_S: float = 0.25
# Per-factor 429/503/504 retry inside the scan.
_SUGGEST_FETCH_RETRY_AFTER_S: float = 1.5


def _suggest_cache_key(ticker: str, lookback_days: int, top_k: int, min_n_obs: int) -> str:
    return f"factors_suggest_for_ticker::{ticker.upper()}::{lookback_days}::{top_k}::{min_n_obs}"


def _suggest_l2_key(cache_key: str) -> str:
    """Redis key for the L2 payload (distinct from the lock key)."""
    return f"{_SUGGEST_L2_PREFIX}:{cache_key}"


def _suggest_lock_key(cache_key: str) -> str:
    """Redis key for the SETNX stampede lock."""
    return f"{_SUGGEST_LOCK_PREFIX}:{cache_key}"


def _suggest_l2_get(cache: CacheBackend, cache_key: str) -> dict | None:
    """Cross-worker L2 read. ``None`` on miss / decode error / Redis hiccup."""
    if cache is None or not getattr(cache, "enabled", False):
        return None
    raw: bytes | None = None
    with contextlib.suppress(Exception):
        raw = cache.get(_suggest_l2_key(cache_key))
    if not raw:
        return None
    try:
        return _json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except (TypeError, ValueError, _json.JSONDecodeError):
        return None


def _suggest_l2_set(cache: CacheBackend, cache_key: str, payload: dict) -> None:
    """Cross-worker L2 write. Best-effort; silently skips on any failure."""
    if cache is None or not getattr(cache, "enabled", False):
        return
    try:
        blob = _json.dumps(payload, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return
    with contextlib.suppress(Exception):
        cache.set(_suggest_l2_key(cache_key), blob, _SUGGEST_FOR_TICKER_TTL_S)


def _suggest_try_lock(cache: CacheBackend, cache_key: str) -> bool:
    """SETNX lock — return ``True`` if we own the lock and should scan.

    When Redis is offline (NullCache in tests) ``setnx`` returns ``True``
    so single-process semantics apply. With Redis up, only one of N
    concurrent workers wins; the rest go into ``_suggest_wait_for_l2``.
    """
    if cache is None:
        return True
    try:
        return bool(cache.setnx(_suggest_lock_key(cache_key), b"1", _SUGGEST_LOCK_TTL_S))
    except Exception:  # pragma: no cover — defensive
        return True  # fail open: better double-fetch than to hang


def _suggest_wait_for_l2(
    cache: CacheBackend,
    bucket,
    cache_key: str,
    deadline_s: float = _SUGGEST_LOCK_WAIT_S,
) -> dict | None:
    """Poll L1 + L2 until the leader writes the answer or we time out.

    Returns the cached payload as soon as either layer materialises.
    Returns ``None`` once the wait budget is exhausted — caller falls
    back to scanning itself (preferred over hanging the request).
    """
    deadline = _time.monotonic() + deadline_s
    while _time.monotonic() < deadline:
        _time.sleep(_SUGGEST_LOCK_POLL_S)
        cached = bucket.get(cache_key)
        if cached is not None:
            return cached
        l2 = _suggest_l2_get(cache, cache_key)
        if l2 is not None:
            # Promote into L1 so we don't poll Redis on subsequent same-key
            # requests in this worker.
            with contextlib.suppress(TypeError, ValueError):
                bucket.set(cache_key, l2, _SUGGEST_FOR_TICKER_TTL_S)
            return l2
    return None


def _is_rate_limit_or_transient(e: BaseException) -> bool:
    """Return True for 429/503/504 — the cases worth a single retry.

    Polymarket's gamma + CLOB occasionally bursts 429 under fan-out load
    (we hit 30 concurrent requests for the cold scan). 503/504 cover
    upstream timeouts that are routinely transient. Other errors (404,
    422, 502 with a real failure) are NOT retried.
    """
    if isinstance(e, HTTPException):
        return e.status_code in (429, 503, 504)
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in (429, 503, 504)
    return False


def _fetch_factor_with_retry(
    fc: FactorConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
    cached_factor_history,  # injected to avoid the lazy-import in tight loop
) -> pd.DataFrame:
    """One-shot retry on 429/503/504 around ``_cached_factor_history``.

    On the first transient error we sleep ``_SUGGEST_FETCH_RETRY_AFTER_S``
    and retry once. A second transient error is raised so the caller skips
    this factor — we don't tank the whole scan because Polymarket throttled
    us on ONE slug.
    """
    try:
        return cached_factor_history(fc, start, end, poly, cache, settings)
    except (HTTPException, httpx.HTTPStatusError) as e:
        if not _is_rate_limit_or_transient(e):
            raise
        logger.info(
            "suggest-for-ticker: rate-limit on slug=%s — retrying in %.1fs",
            fc.slug,
            _SUGGEST_FETCH_RETRY_AFTER_S,
        )
        _time.sleep(_SUGGEST_FETCH_RETRY_AFTER_S)
        return cached_factor_history(fc, start, end, poly, cache, settings)


def _scan_factor_correlations_for_ticker(
    ticker: str,
    lookback_days: int,
    min_n_obs: int,
    *,
    factors: dict[str, FactorConfig],
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[list[SuggestForTickerItem], int, int]:
    """Compute |Pearson r| between the ticker and every curated factor.

    Returns (sorted_items, n_scanned, n_skipped). Items are sorted by
    ``abs_r`` descending. Skips factors that error out, have no data, or
    have fewer than ``min_n_obs`` overlapping observations after alignment.

    Uses the same parallel-fetch + alignment pipeline as ``/factors/rank``
    (just bypasses the OLS step) so factors tested here behave identically
    to factors fed to ``/fit``. Per-factor fetch retries once on 429/503/504
    — see ``_fetch_factor_with_retry``.
    """
    from pfm import main as _m  # lazy to avoid import cycle at startup

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=int(lookback_days))

    # 1. Equity returns for the ticker.
    try:
        y_full = _m._cached_log_returns(ticker, start, end, "log", cache, settings)
    except HTTPException:
        # Bubble up — caller turns this into a 502 response.
        raise

    candidates: list[FactorConfig] = list(factors.values())
    n_skipped = 0

    # 2. Parallel-fetch every factor's history. Bounded fan-out so we
    #    don't melt the Polymarket gateway. Override default 20 with the
    #    same env knob the prewarmer uses — we already proved the gamma
    #    gateway tolerates 30 in the prewarm path.
    from concurrent.futures import ThreadPoolExecutor

    default_fanout = getattr(_m, "_POLY_FANOUT_SEMAPHORE_SIZE", 20)
    try:
        env_fanout = int(_os.environ.get("PFM_FACTOR_PREWARM_CONCURRENCY", "0") or "0")
    except ValueError:
        env_fanout = 0
    fanout = max(default_fanout, env_fanout) if env_fanout > 0 else default_fanout
    max_workers = min(fanout, max(1, len(candidates)))
    fetched: dict[str, pd.DataFrame] = {}
    fetch_errors: dict[str, BaseException] = {}
    cached_factor_history = _m._cached_factor_history
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pfm-suggest") as ex:
        future_map = {
            ex.submit(
                _fetch_factor_with_retry,
                fc,
                start,
                end,
                poly,
                cache,
                settings,
                cached_factor_history,
            ): fc
            for fc in candidates
        }
        for fut, fc in future_map.items():
            try:
                fetched[fc.id] = fut.result()
            except (PolymarketError, ValueError, HTTPException, httpx.HTTPError) as e:
                fetch_errors[fc.id] = e

    items: list[SuggestForTickerItem] = []
    for fc in candidates:
        if fc.id in fetch_errors:
            n_skipped += 1
            continue
        prices = fetched.get(fc.id)
        if prices is None or prices.empty:
            n_skipped += 1
            continue
        try:
            prices = prices[(prices.index >= start) & (prices.index <= end)]
            aligned = _m._align_factor_prices(prices["price"], start, end, "strict")
            # Probability factors → Δlogit. Level factors (yields/indices) →
            # plain first difference (delta_level). For the smart-picker we
            # only care about probability factors since those are the ones
            # actually queryable by /fit. Skip non-probability factors to
            # keep the response laser-focused.
            if fc.is_probability:
                x_full = delta_logit(aligned, epsilon=epsilon).rename(fc.id).dropna()
            else:
                x_full = aligned.diff().rename(fc.id).dropna()
            x = _m._shift_to_stock_calendar(x_full, days=-1)
            common = x.index.intersection(y_full.index)
            n = len(common)
            if n < min_n_obs:
                n_skipped += 1
                continue
            y_sub = y_full.loc[common].astype(float).values
            x_sub = x.loc[common].astype(float).values
            # Guard against zero-variance series (would emit NaN).
            if float(np.std(y_sub)) <= 0.0 or float(np.std(x_sub)) <= 0.0:
                n_skipped += 1
                continue
            r = float(np.corrcoef(y_sub, x_sub)[0, 1])
            if not np.isfinite(r):
                n_skipped += 1
                continue
            items.append(
                SuggestForTickerItem(
                    factor_id=fc.id,
                    name=fc.name,
                    source=fc.source,
                    theme=fc.theme,
                    r=r,
                    abs_r=abs(r),
                    n_obs=int(n),
                )
            )
        except (PolymarketError, ValueError, HTTPException, httpx.HTTPError):
            n_skipped += 1
            continue

    # Sort by |r| desc, ties broken by larger n.
    items.sort(key=lambda it: (-it.abs_r, -it.n_obs))
    return items, len(candidates) - n_skipped, n_skipped


def _release_suggest_lock(cache: CacheBackend, cache_key: str) -> None:
    """Best-effort lock release — failure is fine, the TTL is a backstop."""
    if cache is None:
        return
    client = getattr(cache, "_client", None)
    if client is None:
        return
    with contextlib.suppress(Exception):
        client.delete(_suggest_lock_key(cache_key))


@router.post("/factors/suggest-for-ticker", response_model=SuggestForTickerResponse)
async def suggest_factors_for_ticker(
    body: SuggestForTickerRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> SuggestForTickerResponse:
    """Smart factor picker — top-K factors most correlated with a ticker.

    Computes |Pearson r| between the ticker's log returns and every
    curated factor's Δlogit (Δlevel for non-probability factors), ranked
    by |r|. Cached 1 h per (ticker, lookback_days, top_k, min_n_obs) at
    two layers (process-local + Redis L2) with SETNX stampede protection
    so 4 gunicorn workers share one cold scan instead of 4×.

    Powers the frontend's "factors most likely to explain ``TICKER``"
    panel in the Regression mode. Cold target: <60 s for 10 different
    tickers in parallel. Warm target: <5 s for 10 same-ticker calls.
    """
    cache_key = _suggest_cache_key(body.ticker, body.lookback_days, body.top_k, body.min_n_obs)
    bucket = get_terminal_cache("factors_suggest_for_ticker", ttl=_SUGGEST_FOR_TICKER_TTL_S)

    # --- L1 hit (this worker, microseconds) ----------------------------------
    cached = bucket.get(cache_key)
    if cached is not None:
        try:
            return SuggestForTickerResponse.model_validate(cached)
        except (ValueError, TypeError):
            # Corrupt cache (e.g. schema migration) — fall through and
            # rebuild from scratch.
            bucket.clear()

    # --- L2 hit (cross-worker, ~few ms Redis read) ---------------------------
    l2 = _suggest_l2_get(cache, cache_key)
    if l2 is not None:
        try:
            resp = SuggestForTickerResponse.model_validate(l2)
        except (ValueError, TypeError):
            resp = None  # poisoned L2; fall through and rescan
        if resp is not None:
            with contextlib.suppress(TypeError, ValueError):
                bucket.set(cache_key, l2, _SUGGEST_FOR_TICKER_TTL_S)
            return resp

    # --- Cold path: try to win the scan lock ---------------------------------
    is_leader = _suggest_try_lock(cache, cache_key)
    if not is_leader:
        # Another worker is scanning — wait for it to write the L2 entry.
        waited = await asyncio.to_thread(
            _suggest_wait_for_l2,
            cache,
            bucket,
            cache_key,
        )
        if waited is not None:
            try:
                return SuggestForTickerResponse.model_validate(waited)
            except (ValueError, TypeError):
                pass  # poisoned — fall through and scan ourselves
        # Wait budget exhausted (rare) — fall through to do our own scan.
        # We don't try to grab the lock again because the leader is
        # presumably still working; running concurrently is safe (we just
        # double-charge the upstream once).

    try:
        try:
            items, n_scanned, n_skipped = await asyncio.to_thread(
                _scan_factor_correlations_for_ticker,
                body.ticker,
                body.lookback_days,
                body.min_n_obs,
                factors=factors,
                poly=poly,
                cache=cache,
                settings=settings,
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("suggest-for-ticker failed for %s: %s", body.ticker, e)
            raise HTTPException(
                status_code=502,
                detail=f"factor scan failed: {type(e).__name__}",
            ) from e

        resp = SuggestForTickerResponse(
            ticker=body.ticker.upper(),
            lookback_days=body.lookback_days,
            n_factors_scanned=n_scanned,
            n_factors_skipped=n_skipped,
            top_factors=items[: body.top_k],
        )
        # JSON serialisation issues should not fail the request — both
        # cache layers are best-effort, the response is correct either way.
        payload = None
        with contextlib.suppress(TypeError, ValueError):
            payload = resp.model_dump()
        if payload is not None:
            with contextlib.suppress(TypeError, ValueError):
                bucket.set(cache_key, payload, _SUGGEST_FOR_TICKER_TTL_S)
            _suggest_l2_set(cache, cache_key, payload)
        return resp
    finally:
        if is_leader:
            _release_suggest_lock(cache, cache_key)
