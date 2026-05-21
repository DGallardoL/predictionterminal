"""Event-on-event factor model — regress one PM event on others, in Δlogit space.

This module is the *companion* to :mod:`pfm.model` (which regresses equity
returns on Δlogit factors): here both target and regressors are
**prediction-market probability series**. The point is to quantify how a
political/macro/sport event co-moves with other events, not how it co-moves
with stocks.

Five primitives are exposed:

* :func:`fit_event_on_event` — OLS-HAC regression of Δlogit(target) on
  Δlogit(predictors). Returns betas, t-stats, R², VIF, condition number,
  Durbin-Watson, residual diagnostics. Reuses :func:`pfm.model.fit_ols_hac`.
* :func:`event_correlation_matrix` — Pearson / Spearman / Kendall pairwise
  correlation among ``factor_ids``, with average off-diagonal and a
  hierarchical-cluster ordering for nicer heatmaps.
* :func:`event_lead_lag` — cross-correlation function over lags
  ``[-max_lag, +max_lag]``, plus bivariate Granger causality both ways.
* :func:`event_vector_autoregression` — VAR(p) on the Δlogit panel, with
  per-pair Granger p-values, the first 3 impulse-response horizons, and
  forecast-error-variance decomposition.
* :func:`event_pca_decomposition` — PCA on Δlogit innovations: loadings,
  explained-variance ratios, and a heuristic textual interpretation of
  the top three components.

All functions operate on **already-fetched** probability series (passed in
as :class:`pandas.Series` or via a ``fetch_history`` callable so tests can
mock cleanly). The router thinly wraps fetching + caching.

Theory: see ``docs/event_on_event_theory.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from pfm.model import DEFAULT_EPSILON, delta_logit, fit_ols_hac

# --- type aliases -----------------------------------------------------------

CorrelationMethod = Literal["pearson", "spearman", "kendall"]
CorrelationOn = Literal["delta_logit", "level"]
ReturnType = Literal["delta_logit", "level"]

# A history fetcher: ``(factor_id, start, end) -> pd.Series`` indexed by date,
# values in (0, 1). Tests inject synthetic series; the router injects a real
# Polymarket/Kalshi-backed fetcher.
HistoryFetcher = Callable[[str, date, date], pd.Series]


# --- helpers ---------------------------------------------------------------


def _to_aligned_panel(
    series_by_id: dict[str, pd.Series],
    *,
    return_type: ReturnType,
    epsilon: float,
) -> pd.DataFrame:
    """Stack probability series and apply Δlogit (or pass through level).

    Returns a DataFrame indexed by the *intersection* of dates, columns in
    the iteration order of ``series_by_id``. Drops any rows with NaN.
    """
    if not series_by_id:
        raise ValueError("at least one factor required")
    cols: dict[str, pd.Series] = {}
    for fid, s in series_by_id.items():
        if s is None or s.empty:
            raise ValueError(f"factor {fid!r}: empty history")
        if return_type == "delta_logit":
            cols[fid] = delta_logit(s.astype(float), epsilon=epsilon)
        else:
            cols[fid] = pd.Series(
                np.clip(s.astype(float).to_numpy(), epsilon, 1.0 - epsilon),
                index=s.index,
            )
    df = pd.concat(cols, axis=1).dropna()
    return df


def _residuals_summary(resid: np.ndarray) -> dict[str, float]:
    """Per-residual-vector descriptive stats useful for diagnosing the fit."""
    arr = np.asarray(resid, dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "skew": float("nan"),
            "kurtosis": float("nan"),
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    centred = arr - mean
    skew = float((centred**3).mean() / (std**3)) if std > 0 else 0.0
    kurt = float((centred**4).mean() / (std**4) - 3.0) if std > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "skew": skew,
        "kurtosis": kurt,
    }


def _condition_number(X: pd.DataFrame) -> float:
    """``cond(X)`` via SVD on the design matrix without the intercept."""
    if X.shape[1] == 0 or X.shape[0] == 0:
        return float("nan")
    try:
        sv = np.linalg.svd(X.values, compute_uv=False)
        if sv.min() <= 0:
            return float("inf")
        return float(sv.max() / sv.min())
    except np.linalg.LinAlgError:
        return float("inf")


def _hierarchical_order(corr: np.ndarray, labels: list[str]) -> list[str]:
    """Order labels by single-linkage clustering on the distance matrix.

    Falls back to the input order if the clustering library or the matrix
    is degenerate.
    """
    if corr.shape[0] != corr.shape[1] or corr.shape[0] < 2:
        return list(labels)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        # Distance ∈ [0, 2]: 0 perfectly correlated, 2 perfectly anti-correlated.
        dist = 1.0 - corr
        # Symmetrise + zero diagonal to satisfy ``squareform`` invariants.
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(np.clip(dist, 0.0, None), checks=False)
        z = linkage(condensed, method="average")
        order = leaves_list(z)
        return [labels[i] for i in order]
    except Exception:
        return list(labels)


# --- 1. event-on-event regression ------------------------------------------


def fit_event_on_event(
    target_factor_id: str,
    predictor_factor_ids: list[str],
    start: date,
    end: date,
    *,
    return_type: ReturnType = "delta_logit",
    epsilon: float = DEFAULT_EPSILON,
    fetch_history: HistoryFetcher,
    hac_lag: int | None = None,
) -> dict:
    """Fit ``Δlogit(target) ~ α + Σ β_j Δlogit(predictor_j)`` with HAC SEs.

    Args:
        target_factor_id: id of the dependent PM event.
        predictor_factor_ids: ids of the regressor PM events. Must be
            non-empty, and must NOT contain the target.
        start, end: UTC date window passed to ``fetch_history``.
        return_type: ``"delta_logit"`` (default — recommended) or
            ``"level"`` (regress probabilities directly; biased and less
            stationary, exposed for sensitivity checks).
        epsilon: clipping bound for the logit.
        fetch_history: ``(factor_id, start, end) -> pd.Series`` of probabilities.
        hac_lag: optional Newey-West truncation. ``None`` ⇒ Andrews bandwidth.

    Returns:
        dict with keys: ``target``, ``predictors``, ``n_obs``, ``alpha``,
        ``betas``, ``t_stats``, ``p_values``, ``r_squared``,
        ``adj_r_squared``, ``residuals_summary``, ``vif``, ``hac_lag``,
        ``condition_number``, ``durbin_watson``, ``return_type``.

    Raises:
        ValueError: if predictors are empty, contain the target, or after
            alignment the sample is too short.
    """
    if not predictor_factor_ids:
        raise ValueError("at least one predictor is required")
    if target_factor_id in predictor_factor_ids:
        raise ValueError(f"target {target_factor_id!r} cannot also appear in predictors")

    series: dict[str, pd.Series] = {target_factor_id: fetch_history(target_factor_id, start, end)}
    for fid in predictor_factor_ids:
        series[fid] = fetch_history(fid, start, end)

    panel = _to_aligned_panel(series, return_type=return_type, epsilon=epsilon)
    n_obs = len(panel)
    k = len(predictor_factor_ids)
    if n_obs < max(20, k + 5):
        raise ValueError(
            f"only {n_obs} jointly-observed dates for {k} predictors "
            f"(need >= max(20, k+5) = {max(20, k + 5)})"
        )

    y = panel[target_factor_id]
    X = panel[predictor_factor_ids]

    fit = fit_ols_hac(y, X, hac_lag=hac_lag, regression="hac")

    betas = {fe.factor_id: fe.beta for fe in fit.factors}
    t_stats = {fe.factor_id: fe.t_stat for fe in fit.factors}
    p_values = {fe.factor_id: fe.p_value for fe in fit.factors}

    return {
        "target": target_factor_id,
        "predictors": list(predictor_factor_ids),
        "n_obs": n_obs,
        "alpha": fit.stats.alpha,
        "betas": betas,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": fit.stats.r_squared,
        "adj_r_squared": fit.stats.r_squared_adj,
        "residuals_summary": _residuals_summary(np.asarray(fit.fitted.resid)),
        "vif": dict(fit.diagnostics.vif),
        "hac_lag": int(fit.diagnostics.hac_lag),
        "condition_number": _condition_number(X),
        "durbin_watson": float(fit.diagnostics.durbin_watson),
        "return_type": return_type,
    }


# --- 2. correlation matrix --------------------------------------------------


def event_correlation_matrix(
    factor_ids: list[str],
    start: date,
    end: date,
    *,
    method: CorrelationMethod = "pearson",
    on: CorrelationOn = "delta_logit",
    epsilon: float = DEFAULT_EPSILON,
    fetch_history: HistoryFetcher,
) -> dict:
    """Pairwise correlation of ``factor_ids`` (Pearson / Spearman / Kendall).

    Returns:
        dict with keys ``factor_ids``, ``method``, ``on``, ``matrix``
        (list of lists), ``avg_off_diagonal``, ``n_obs_min``,
        ``hierarchical_cluster_order``.
    """
    if len(factor_ids) < 2:
        raise ValueError("need at least 2 factor_ids for a correlation matrix")
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor_ids must be unique")

    series = {fid: fetch_history(fid, start, end) for fid in factor_ids}
    panel = _to_aligned_panel(series, return_type=on, epsilon=epsilon)
    if len(panel) < 5:
        raise ValueError(f"only {len(panel)} jointly-observed dates after alignment (need >= 5)")

    corr_df = panel.corr(method=method)
    # Reorder to user's input ordering (pandas may reshuffle on dropna).
    corr_df = corr_df.loc[factor_ids, factor_ids]
    matrix = corr_df.to_numpy()

    n = len(factor_ids)
    if n >= 2:
        off = matrix[~np.eye(n, dtype=bool)]
        avg_off = float(np.nanmean(off)) if off.size else float("nan")
    else:
        avg_off = float("nan")

    cluster_order = _hierarchical_order(matrix, factor_ids)

    return {
        "factor_ids": list(factor_ids),
        "method": method,
        "on": on,
        "matrix": [[float(matrix[i, j]) for j in range(n)] for i in range(n)],
        "avg_off_diagonal": avg_off,
        "n_obs_min": int(len(panel)),
        "hierarchical_cluster_order": cluster_order,
    }


# --- 3. lead-lag -----------------------------------------------------------


def _shifted_corr(a: pd.Series, b: pd.Series, lag: int) -> tuple[float, float, int]:
    """``corr(a_t, b_{t-lag})``: positive lag ⇒ b leads a.

    Returns ``(correlation, t_stat, n_used)``. NaN when the joint sample is
    too small or one side has zero variance.
    """
    if lag >= 0:
        a_s = a.iloc[lag:]
        b_s = b.iloc[: len(b) - lag] if lag > 0 else b
        # realign indices for pandas correlation
        n = min(len(a_s), len(b_s))
        if n < 5:
            return float("nan"), float("nan"), n
        x = a_s.iloc[:n].to_numpy()
        y = b_s.iloc[:n].to_numpy()
    else:
        # negative lag: a leads b
        a_s = a.iloc[: len(a) + lag]
        b_s = b.iloc[-lag:]
        n = min(len(a_s), len(b_s))
        if n < 5:
            return float("nan"), float("nan"), n
        x = a_s.iloc[:n].to_numpy()
        y = b_s.iloc[:n].to_numpy()

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x, y)[0, 1])
    # Asymptotic t-stat for Pearson r.
    t = r * np.sqrt(max(n - 2, 1) / max(1 - r**2, 1e-12)) if abs(r) < 1.0 else float("inf")
    return r, float(t), n


def event_lead_lag(
    target_id: str,
    predictor_id: str,
    start: date,
    end: date,
    *,
    max_lag: int = 5,
    epsilon: float = DEFAULT_EPSILON,
    fetch_history: HistoryFetcher,
) -> dict:
    """Cross-correlation function between Δlogit(target) and Δlogit(predictor).

    Sweeps ``lag ∈ [-max_lag, +max_lag]``. Sign convention:

    *   ``lag > 0`` ⇒ predictor *leads* target by ``lag`` days.
    *   ``lag < 0`` ⇒ target leads predictor.

    Also runs bivariate Granger causality both ways using
    :mod:`pfm.granger`.
    """
    if target_id == predictor_id:
        raise ValueError("target and predictor must differ")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    panel = _to_aligned_panel(
        {
            target_id: fetch_history(target_id, start, end),
            predictor_id: fetch_history(predictor_id, start, end),
        },
        return_type="delta_logit",
        epsilon=epsilon,
    )
    if len(panel) < max(20, 4 * max_lag + 2):
        raise ValueError(
            f"only {len(panel)} jointly-observed bars; need >= "
            f"max(20, 4*max_lag+2) = {max(20, 4 * max_lag + 2)}"
        )

    a = panel[target_id]
    b = panel[predictor_id]

    ccf: list[dict[str, float]] = []
    for lag in range(-max_lag, max_lag + 1):
        r, t, _ = _shifted_corr(a, b, lag)
        ccf.append({"lag": int(lag), "correlation": r, "t_stat": t})

    # Pick best by |corr|.
    finite = [row for row in ccf if not np.isnan(row["correlation"])]
    if finite:
        best = max(finite, key=lambda r: abs(r["correlation"]))
        best_lag = int(best["lag"])
        best_corr = float(best["correlation"])
    else:
        best_lag = 0
        best_corr = float("nan")

    # Granger (delegated). Catch failures so the endpoint stays usable.
    p_target_leads = float("nan")
    p_predictor_leads = float("nan")
    try:
        from pfm.granger import granger_test

        gran = granger_test(a, b, a_id=target_id, b_id=predictor_id, max_lag=max_lag)
        # In :func:`granger_test`, ``a`` is target and ``b`` is predictor.
        # ``best_pvalue_b_to_a`` = predictor → target ⇒ predictor leads target.
        # ``best_pvalue_a_to_b`` = target → predictor ⇒ target leads predictor.
        if gran.best_pvalue_b_to_a is not None:
            p_predictor_leads = float(gran.best_pvalue_b_to_a)
        if gran.best_pvalue_a_to_b is not None:
            p_target_leads = float(gran.best_pvalue_a_to_b)
    except Exception:
        # Granger failure (perfect-fit VAR, too short) is non-fatal.
        pass

    return {
        "target": target_id,
        "predictor": predictor_id,
        "n_obs": int(len(panel)),
        "max_lag": int(max_lag),
        "ccf": ccf,
        "best_lag": best_lag,
        "best_correlation": best_corr,
        "granger_p_target_leads": p_target_leads,
        "granger_p_predictor_leads": p_predictor_leads,
    }


# --- 4. VAR ----------------------------------------------------------------


def event_vector_autoregression(
    factor_ids: list[str],
    start: date,
    end: date,
    *,
    lags: int = 5,
    epsilon: float = DEFAULT_EPSILON,
    fetch_history: HistoryFetcher,
) -> dict:
    """Vector autoregression on the Δlogit panel of ``factor_ids``.

    Returns coefficient cubes, pairwise Granger p-values, the first three
    impulse-response horizons, and forecast-error-variance decomposition
    at horizon ``lags``.
    """
    if len(factor_ids) < 2:
        raise ValueError("VAR requires >= 2 factors")
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")

    series = {fid: fetch_history(fid, start, end) for fid in factor_ids}
    panel = _to_aligned_panel(series, return_type="delta_logit", epsilon=epsilon)
    n_obs = len(panel)
    k = len(factor_ids)
    # VAR(p) needs T > k*p + 1 to identify; require a comfortable margin.
    min_required = max(30, lags * (k + 2))
    if n_obs < min_required:
        raise ValueError(
            f"only {n_obs} jointly-observed bars for VAR(k={k}, p={lags}); need >= {min_required}"
        )

    panel = panel[factor_ids]  # enforce column order

    from statsmodels.tsa.api import VAR

    # statsmodels' VAR emits FutureWarning on some pandas/numpy combos —
    # silence to keep server logs clean. Test still asserts on numerical output.
    # Pass the DataFrame (not the raw array) so ``test_causality`` accepts
    # string column names downstream.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_res = VAR(panel).fit(maxlags=lags, trend="c")

    # Coefficient cube: shape (lags, k, k). var_res.coefs is (lags, k, k).
    coefs = np.asarray(var_res.coefs)  # (p, k, k)
    coef_matrix = [
        [[float(coefs[ell, i, j]) for j in range(k)] for i in range(k)]
        for ell in range(coefs.shape[0])
    ]

    # Granger causality pairwise: result[i][j] = p-value that j → i.
    granger_matrix: list[list[float]] = [[float("nan")] * k for _ in range(k)]
    for i, target in enumerate(factor_ids):
        for j, cause in enumerate(factor_ids):
            if i == j:
                granger_matrix[i][j] = float("nan")
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = var_res.test_causality(caused=target, causing=[cause], kind="f")
                granger_matrix[i][j] = float(g.pvalue)
            except Exception:
                granger_matrix[i][j] = float("nan")

    # Impulse response — first 3 periods.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            irf = var_res.irf(periods=3)
        # irf.irfs has shape (periods+1, k, k); index [h, response, shock].
        irfs = np.asarray(irf.irfs)
        impulse_response = [
            [[float(irfs[h, i, j]) for j in range(k)] for i in range(k)]
            for h in range(irfs.shape[0])
        ]
    except Exception:
        impulse_response = []

    # FEVD at horizon = lags.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fevd = var_res.fevd(periods=max(lags, 3))
        # fevd.decomp has shape (k, periods, k); we report the last horizon.
        decomp = np.asarray(fevd.decomp)
        last = decomp.shape[1] - 1
        fevd_matrix = [[float(decomp[i, last, j]) for j in range(k)] for i in range(k)]
    except Exception:
        fevd_matrix = [[float("nan")] * k for _ in range(k)]

    return {
        "factor_ids": list(factor_ids),
        "n_obs": int(n_obs),
        "lags": int(lags),
        "coefficients_matrix": coef_matrix,
        "granger_causality_matrix": granger_matrix,
        "impulse_response_first_3_periods": impulse_response,
        "forecast_error_variance_decomposition": fevd_matrix,
    }


# --- 5. PCA ----------------------------------------------------------------


def event_pca_decomposition(
    factor_ids: list[str],
    start: date,
    end: date,
    *,
    n_components: int = 5,
    epsilon: float = DEFAULT_EPSILON,
    fetch_history: HistoryFetcher,
) -> dict:
    """PCA on the Δlogit-innovation panel.

    Components are interpreted by inspecting the largest absolute loadings:
    if all loadings have the same sign, the component is labelled
    "broad_market"; otherwise it's a "spread" / "rotation" component
    between the heaviest positives and heaviest negatives.
    """
    if len(factor_ids) < 2:
        raise ValueError("PCA requires >= 2 factors")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    series = {fid: fetch_history(fid, start, end) for fid in factor_ids}
    panel = _to_aligned_panel(series, return_type="delta_logit", epsilon=epsilon)
    n_obs = len(panel)
    k = len(factor_ids)
    n_components = min(n_components, k, n_obs - 1)
    if n_components < 1:
        raise ValueError(f"insufficient data for PCA: n_obs={n_obs}, k={k}")

    from sklearn.decomposition import PCA

    panel = panel[factor_ids]
    # Centre-scale for numerical stability; PCA on raw Δlogit is fine but
    # the sklearn API works on the demeaned matrix internally regardless.
    mat = panel.to_numpy()
    pca = PCA(n_components=n_components)
    pca.fit(mat)

    explained = [float(v) for v in pca.explained_variance_ratio_]
    # components_ has shape (n_components, k); column j = loading on factor j.
    loadings = pca.components_

    loadings_matrix = [[float(loadings[c, j]) for j in range(k)] for c in range(n_components)]

    interpretations: list[dict[str, object]] = []
    for c in range(min(3, n_components)):
        v = loadings[c]
        signs = np.sign(v)
        same_sign = bool(np.all(signs[np.abs(v) > 1e-9] >= 0)) or bool(
            np.all(signs[np.abs(v) > 1e-9] <= 0)
        )
        order = np.argsort(-np.abs(v))
        top_factors = [factor_ids[idx] for idx in order[: min(3, k)]]
        if same_sign:
            kind = "broad_market"
        else:
            pos = [factor_ids[idx] for idx in order if v[idx] > 0][:2]
            neg = [factor_ids[idx] for idx in order if v[idx] < 0][:2]
            kind = "spread"
            top_factors = pos + neg
            interpretations.append(
                {
                    "component": c + 1,
                    "kind": kind,
                    "explained_variance_ratio": explained[c],
                    "top_positive": pos,
                    "top_negative": neg,
                }
            )
            continue
        interpretations.append(
            {
                "component": c + 1,
                "kind": kind,
                "explained_variance_ratio": explained[c],
                "top_factors": top_factors,
            }
        )

    return {
        "factor_ids": list(factor_ids),
        "n_obs": int(n_obs),
        "n_components": int(n_components),
        "explained_variance_ratio": explained,
        "loadings_matrix": loadings_matrix,
        "top_3_components_interpretation": interpretations,
    }


__all__ = [
    "HistoryFetcher",
    "event_correlation_matrix",
    "event_lead_lag",
    "event_pca_decomposition",
    "event_vector_autoregression",
    "fit_event_on_event",
]
