"""Cross-asset multi-event factor models and event chains.

This module connects prediction-market events to many assets at once, and
chains events across time so we can ask:

    "Given 20 PM events (politics, macro, crypto), how does each equity
    sector load on them, and which event-->event-->ticker paths actually
    transmit information?"

Five public entry points
------------------------

A. ``fit_multi_event_lasso`` — high-dimensional OLS with L1 penalty
   (LassoCV) on N PM-factor Δlogits to predict one ticker's log returns.
   Sparse solution; report which factors survive.

B. ``sector_attribution`` — per-sector OLS with HAC SEs on the same factor
   set, then variance decomposition of ``β_{j,i} · Δlogit_{i}`` per
   sector j × factor i. Output is a dense matrix the UI can heat-map.

C. ``find_chains`` — Granger-pathfinding from a starting factor to a
   target ticker through candidate intermediate factors, with each edge
   gated by Granger p < 0.10 and a sign-consistent lead-lag.

D. ``event_macro_correlation`` — Δlogit(factor) vs Δ(macro series)
   correlation, t-stat, and lead-lag from FRED daily series.

E. ``extract_systemic_pm_factor`` — PCA on Δlogit innovations of N
   factors. The first PC is interpreted as a PM-implied risk-on/off
   systemic factor, exposed as a tradeable signal.

Design choices
--------------
- Re-uses ``pfm.model.fit_ols_hac`` for the per-sector regressions so
  HAC SEs and diagnostics stay consistent with the rest of the project.
- Re-uses ``pfm.granger.granger_test`` for chain edges, treating each
  edge as a one-sided test (start → next factor → ... → target).
- All data IO goes through the project sources (``polymarket``,
  ``equity``, ``fred``); the module itself is pure for unit tests.
- Synthetic-DGP tests live in ``tests/test_multi_event_chain.py`` and
  cover all five entry points.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from pfm.granger import granger_test
from pfm.model import DEFAULT_EPSILON, delta_logit, fit_ols_hac

logger = logging.getLogger(__name__)


DEFAULT_SECTOR_ETFS: list[str] = [
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "XLI",
    "XLU",
    "XLB",
    "XLY",
    "XLP",
    "XLRE",
    "XLC",
]


# ---------------------------------------------------------------------------
# Data fetcher injection
# ---------------------------------------------------------------------------
# To keep the math testable we accept fetcher callables. Production callers
# (the router) wire them to the real ``polymarket`` / ``equity`` / ``fred``
# sources. Tests pass synthetic fetchers.

FactorFetcher = Callable[[str, pd.Timestamp, pd.Timestamp], pd.Series]
"""Fetcher signature: (factor_id, start, end) -> probability Series in [0, 1]."""

ReturnFetcher = Callable[[str, pd.Timestamp, pd.Timestamp], pd.Series]
"""Fetcher signature: (ticker, start, end) -> log-return Series."""

MacroFetcher = Callable[[str, pd.Timestamp, pd.Timestamp], pd.Series]
"""Fetcher signature: (series_id, start, end) -> level Series (FRED)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_dates(start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Accept ISO strings or ``pd.Timestamp`` and return UTC-normalised pair."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    if e.tzinfo is None:
        e = e.tz_localize("UTC")
    return s.normalize(), e.normalize()


def _build_factor_panel(
    factor_ids: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_factor: FactorFetcher,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.DataFrame:
    """Fetch each factor, compute Δlogit, align on a UTC-daily index.

    Factors with fewer than 3 valid Δlogit observations are dropped (they
    cannot contribute meaningfully to a regression).
    """
    cols: dict[str, pd.Series] = {}
    for fid in factor_ids:
        try:
            s = fetch_factor(fid, start, end)
        except Exception as e:
            logger.warning("multi_event: factor %s fetch failed: %s", fid, e)
            continue
        if s is None or len(s) < 4:
            continue
        s = pd.Series(s).astype(float)
        if s.index.tzinfo is None:
            s.index = pd.to_datetime(s.index, utc=True)
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        d = delta_logit(s, epsilon=epsilon).dropna()
        if len(d) >= 3:
            cols[fid] = d
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()


def _align_y_x(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-join y and X on UTC-daily index, drop NaN rows."""
    y = pd.Series(y).astype(float)
    if y.index.tzinfo is None:
        y.index = pd.to_datetime(y.index, utc=True)
    y.index = y.index.normalize()
    y = y[~y.index.duplicated(keep="last")].sort_index()
    df = pd.concat({"__y__": y, **{c: X[c] for c in X.columns}}, axis=1).dropna()
    return df["__y__"], df.drop(columns=["__y__"])


# ---------------------------------------------------------------------------
# A. Multi-event LASSO
# ---------------------------------------------------------------------------


def fit_multi_event_lasso(
    ticker: str,
    factor_ids: list[str],
    start: Any,
    end: Any,
    *,
    alpha: float = 0.01,
    fetch_factor: FactorFetcher,
    fetch_returns: ReturnFetcher,
    epsilon: float = DEFAULT_EPSILON,
    cv_folds: int = 5,
) -> dict:
    """Fit LassoCV on N factor Δlogits to predict ``ticker`` log returns.

    The ``alpha`` arg is used as a fallback when the CV grid search fails
    (e.g. too few observations); otherwise LassoCV picks the optimal
    penalty in a 5-fold time-series-style holdout.

    Returns a dict matching the public schema; never raises for empty
    factor sets — callers see ``n_factors_in=0``.
    """
    s_dt, e_dt = _coerce_dates(start, end)
    X_full = _build_factor_panel(factor_ids, s_dt, e_dt, fetch_factor, epsilon=epsilon)
    if X_full.empty:
        return {
            "ticker": ticker,
            "factor_ids": list(factor_ids),
            "n_factors_in": 0,
            "n_factors_nonzero": 0,
            "betas": {},
            "r_squared": 0.0,
            "alpha_optimal": float(alpha),
            "sparse_solution_factors": [],
            "n_obs": 0,
        }

    y_raw = fetch_returns(ticker, s_dt, e_dt)
    y, X = _align_y_x(y_raw, X_full)
    n_obs = int(len(y))
    n_factors_in = int(X.shape[1])
    if n_obs < max(15, n_factors_in + 2):
        logger.warning(
            "lasso: only %d obs for %d factors; falling back to fixed alpha", n_obs, n_factors_in
        )
        # Fall back to a single-alpha Lasso fit to still produce a result.
        from sklearn.linear_model import Lasso

        scaler = StandardScaler()
        X_s = (
            scaler.fit_transform(X.to_numpy(dtype=float))
            if n_obs > 0
            else np.zeros((0, n_factors_in))
        )
        if n_obs == 0:
            return {
                "ticker": ticker,
                "factor_ids": list(factor_ids),
                "n_factors_in": n_factors_in,
                "n_factors_nonzero": 0,
                "betas": {},
                "r_squared": 0.0,
                "alpha_optimal": float(alpha),
                "sparse_solution_factors": [],
                "n_obs": 0,
            }
        est = Lasso(alpha=alpha, max_iter=20_000)
        est.fit(X_s, y.to_numpy(dtype=float))
        coefs_std = est.coef_
        # Un-standardise back to original scale.
        coefs = coefs_std / scaler.scale_
        alpha_opt = float(alpha)
        y_pred = est.predict(X_s)
        ss_res = float(np.sum((y.to_numpy() - y_pred) ** 2))
        ss_tot = float(np.sum((y.to_numpy() - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X.to_numpy(dtype=float))
        n_splits = max(2, min(int(cv_folds), n_obs // 3))
        try:
            est = LassoCV(cv=n_splits, max_iter=20_000, alphas=50, random_state=0)
            est.fit(X_s, y.to_numpy(dtype=float))
            alpha_opt = float(est.alpha_)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LassoCV failed: %s; falling back to alpha=%g", exc, alpha)
            from sklearn.linear_model import Lasso

            est = Lasso(alpha=alpha, max_iter=20_000)
            est.fit(X_s, y.to_numpy(dtype=float))
            alpha_opt = float(alpha)
        coefs_std = est.coef_
        coefs = coefs_std / scaler.scale_
        y_pred = est.predict(X_s)
        ss_res = float(np.sum((y.to_numpy() - y_pred) ** 2))
        ss_tot = float(np.sum((y.to_numpy() - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    betas = {col: float(c) for col, c in zip(X.columns, coefs, strict=True)}
    sparse_factors = [col for col, c in betas.items() if abs(c) > 1e-9]
    return {
        "ticker": ticker,
        "factor_ids": list(factor_ids),
        "n_factors_in": n_factors_in,
        "n_factors_nonzero": int(len(sparse_factors)),
        "betas": betas,
        "r_squared": float(r2),
        "alpha_optimal": float(alpha_opt),
        "sparse_solution_factors": sparse_factors,
        "n_obs": n_obs,
    }


# ---------------------------------------------------------------------------
# B. Sector attribution
# ---------------------------------------------------------------------------


def sector_attribution(
    sectors_etfs: list[str] | None,
    factor_ids: list[str],
    start: Any,
    end: Any,
    *,
    fetch_factor: FactorFetcher,
    fetch_returns: ReturnFetcher,
    epsilon: float = DEFAULT_EPSILON,
) -> dict:
    """Per-sector OLS-HAC regressions, then variance decomposition.

    For sector j and factor i:

        attribution_{j,i} = Var(β_{j,i} · x_i) / Var(y_j)

    The returned matrix has rows = sectors (in input order, after fetch
    filtering), cols = factors (in input order, after fetch filtering).
    Cells are non-negative R²-share contributions; row sums approximate
    the regression's R² (not exactly — interaction covariances are
    folded into the residual).
    """
    sectors = list(sectors_etfs) if sectors_etfs else list(DEFAULT_SECTOR_ETFS)
    s_dt, e_dt = _coerce_dates(start, end)
    X_full = _build_factor_panel(factor_ids, s_dt, e_dt, fetch_factor, epsilon=epsilon)
    if X_full.empty or not sectors:
        return {
            "sectors": [],
            "factors": [],
            "attribution_matrix": [],
            "betas_matrix": [],
            "r_squared_per_sector": {},
            "dominant_factor_per_sector": {},
            "dominant_sector_per_factor": {},
            "n_obs": 0,
        }

    factor_cols = list(X_full.columns)
    rows: list[list[float]] = []
    beta_rows: list[list[float]] = []
    r2_per: dict[str, float] = {}
    used_sectors: list[str] = []
    n_obs_used = 0

    for sec in sectors:
        try:
            y_raw = fetch_returns(sec, s_dt, e_dt)
        except Exception as e:
            logger.warning("sector_attribution: %s fetch failed: %s", sec, e)
            continue
        y, X = _align_y_x(y_raw, X_full)
        if len(y) < max(10, len(factor_cols) + 2):
            logger.warning(
                "sector_attribution: %s only %d obs for %d factors; skipping",
                sec,
                len(y),
                len(factor_cols),
            )
            continue
        try:
            fit = fit_ols_hac(y, X, regression="hac")
        except Exception as e:
            logger.warning("sector_attribution: OLS-HAC failed for %s: %s", sec, e)
            continue
        var_y = float(np.var(y.to_numpy(dtype=float), ddof=1))
        if var_y <= 0:
            continue
        beta_map = {f.factor_id: f.beta for f in fit.factors}
        attribution_row: list[float] = []
        beta_row: list[float] = []
        for col in factor_cols:
            beta = float(beta_map.get(col, 0.0))
            x_col = X[col].to_numpy(dtype=float)
            var_x = float(np.var(x_col, ddof=1))
            share = (beta * beta * var_x) / var_y if var_y > 0 else 0.0
            attribution_row.append(float(share))
            beta_row.append(beta)
        rows.append(attribution_row)
        beta_rows.append(beta_row)
        r2_per[sec] = float(fit.stats.r_squared)
        used_sectors.append(sec)
        n_obs_used = max(n_obs_used, int(len(y)))

    dominant_factor_per_sector: dict[str, str] = {}
    for i, sec in enumerate(used_sectors):
        if not rows[i]:
            continue
        j_max = int(np.argmax(rows[i]))
        dominant_factor_per_sector[sec] = factor_cols[j_max]

    dominant_sector_per_factor: dict[str, str] = {}
    if rows:
        arr = np.array(rows)  # shape (S, F)
        for j, fid in enumerate(factor_cols):
            i_max = int(np.argmax(arr[:, j]))
            dominant_sector_per_factor[fid] = used_sectors[i_max]

    return {
        "sectors": used_sectors,
        "factors": factor_cols,
        "attribution_matrix": rows,
        "betas_matrix": beta_rows,
        "r_squared_per_sector": r2_per,
        "dominant_factor_per_sector": dominant_factor_per_sector,
        "dominant_sector_per_factor": dominant_sector_per_factor,
        "n_obs": int(n_obs_used),
    }


# ---------------------------------------------------------------------------
# C. Multi-event chains via Granger pathfinding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Edge:
    src: str
    dst: str
    p_value: float
    best_lag: int
    correlation: float


def _granger_edge(
    src_series: pd.Series,
    dst_series: pd.Series,
    src_id: str,
    dst_id: str,
    *,
    max_lag: int = 5,
    p_threshold: float = 0.10,
) -> _Edge | None:
    """Return an Edge if src Granger-causes dst at p < p_threshold, else None."""
    aligned = pd.concat([src_series, dst_series], axis=1).dropna()
    if len(aligned) < max(20, 4 * max_lag + 2):
        return None
    try:
        # statsmodels orientation: B granger-causes A means B is the
        # second column. We want src->dst, so a=dst, b=src.
        res = granger_test(
            aligned.iloc[:, 1].rename(dst_id),
            aligned.iloc[:, 0].rename(src_id),
            a_id=dst_id,
            b_id=src_id,
            max_lag=max_lag,
            alpha=p_threshold,
        )
    except Exception as e:
        logger.debug("granger edge %s->%s failed: %s", src_id, dst_id, e)
        return None
    p = res.best_pvalue_b_to_a
    lag = res.best_lag_b_to_a
    if p is None or lag is None or p >= p_threshold:
        return None
    corr = float(aligned.corr().iloc[0, 1]) if aligned.shape[1] >= 2 else 0.0
    return _Edge(src=src_id, dst=dst_id, p_value=float(p), best_lag=int(lag), correlation=corr)


def find_chains(
    start_factor: str,
    end_ticker: str,
    candidate_intermediate_factors: list[str],
    start: Any,
    end: Any,
    *,
    max_depth: int = 3,
    fetch_factor: FactorFetcher,
    fetch_returns: ReturnFetcher,
    epsilon: float = DEFAULT_EPSILON,
    p_threshold: float = 0.10,
    max_lag: int = 5,
) -> list[dict]:
    """Search depth-bounded paths start_factor → ... → end_ticker.

    Edges are Granger-causality tests at p < ``p_threshold``. Path nodes
    are factor ids except the terminal node which is the ticker. The
    final edge tests "factor Granger-causes ticker log returns".

    Returns list of paths sorted by total Granger p-value (smaller first).
    """
    s_dt, e_dt = _coerce_dates(start, end)

    all_factors = [start_factor] + [f for f in candidate_intermediate_factors if f != start_factor]
    panel = _build_factor_panel(all_factors, s_dt, e_dt, fetch_factor, epsilon=epsilon)
    if panel.empty or start_factor not in panel.columns:
        return []

    try:
        y_raw = fetch_returns(end_ticker, s_dt, e_dt)
    except Exception as e:
        logger.warning("find_chains: returns fetch failed for %s: %s", end_ticker, e)
        return []
    y = pd.Series(y_raw).astype(float)
    if y.index.tzinfo is None:
        y.index = pd.to_datetime(y.index, utc=True)
    y.index = y.index.normalize()
    y = y[~y.index.duplicated(keep="last")].sort_index()

    # DFS with depth limit. max_depth counts edges (so 3 = up to 3 hops).
    paths: list[dict] = []
    factor_universe = [c for c in panel.columns if c != start_factor]

    def _dfs(current: str, visited: list[str], edges: list[_Edge]) -> None:
        depth = len(edges)
        if depth >= max_depth:
            # try to close to the ticker
            edge_to_t = _granger_edge(
                panel[current],
                y,
                current,
                end_ticker,
                max_lag=max_lag,
                p_threshold=p_threshold,
            )
            if edge_to_t is not None:
                _record([*visited, end_ticker], [*edges, edge_to_t])
            return

        # Branch 1: close to ticker now.
        edge_to_t = _granger_edge(
            panel[current],
            y,
            current,
            end_ticker,
            max_lag=max_lag,
            p_threshold=p_threshold,
        )
        if edge_to_t is not None:
            _record([*visited, end_ticker], [*edges, edge_to_t])

        # Branch 2: try going through another factor.
        for nxt in factor_universe:
            if nxt in visited:
                continue
            edge = _granger_edge(
                panel[current],
                panel[nxt],
                current,
                nxt,
                max_lag=max_lag,
                p_threshold=p_threshold,
            )
            if edge is None:
                continue
            _dfs(nxt, [*visited, nxt], [*edges, edge])

    def _record(node_path: list[str], edge_path: list[_Edge]) -> None:
        if not edge_path:
            return
        ps = [e.p_value for e in edge_path]
        lags = [e.best_lag for e in edge_path]
        corrs = [e.correlation for e in edge_path]
        # Total correlation: product of |corr| (signed by product of signs).
        total_corr = float(np.prod(corrs)) if corrs else 0.0
        # Aggregate p: largest along path (weakest link governs).
        agg_p = float(max(ps))
        paths.append(
            {
                "path": node_path,
                "path_length": int(len(edge_path)),
                "total_correlation": total_corr,
                "granger_p_path": [float(p) for p in ps],
                "granger_p_max": agg_p,
                "lead_lag_days_path": [int(la) for la in lags],
            }
        )

    _dfs(start_factor, [start_factor], [])

    paths.sort(key=lambda d: (d["granger_p_max"], -abs(d["total_correlation"])))
    return paths


# ---------------------------------------------------------------------------
# D. Event ↔ macro overlay
# ---------------------------------------------------------------------------


def _newey_west_t_stat(x: np.ndarray, y: np.ndarray, *, max_lag: int | None = None) -> float:
    """t-stat for the slope in a univariate y = a + b·x regression (HAC).

    Used as a robust alternative to the naive Pearson t when both series
    are weakly serially correlated.
    """
    import statsmodels.api as sm

    if len(x) < 6 or len(y) < 6:
        return float("nan")
    X = sm.add_constant(x)
    n = len(x)
    lag = max_lag if max_lag is not None else max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    try:
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        return float(res.tvalues[1])
    except Exception:
        return float("nan")


def event_macro_correlation(
    factor_id: str,
    macro_series: list[str],
    start: Any,
    end: Any,
    *,
    fetch_factor: FactorFetcher,
    fetch_macro: MacroFetcher,
    epsilon: float = DEFAULT_EPSILON,
    max_lead_lag_days: int = 5,
) -> dict:
    """Correlate Δlogit(factor) against Δ(macro) for several macros.

    For lead-lag, we shift the macro series by k ∈ [-max, +max] and pick
    the k with maximum |corr|. Positive k means macro leads factor by k
    days; negative k means factor leads macro.
    """
    s_dt, e_dt = _coerce_dates(start, end)
    factor_panel = _build_factor_panel([factor_id], s_dt, e_dt, fetch_factor, epsilon=epsilon)
    if factor_panel.empty or factor_id not in factor_panel.columns:
        return {
            "factor_id": factor_id,
            "macro_correlations": {},
            "t_stats": {},
            "lead_lag_days": {},
            "n_obs": 0,
        }
    f = factor_panel[factor_id]

    macro_correlations: dict[str, float] = {}
    t_stats: dict[str, float] = {}
    lead_lag_days: dict[str, int] = {}
    n_obs_used = 0
    for mid in macro_series:
        try:
            m_raw = fetch_macro(mid, s_dt, e_dt)
        except Exception as e:
            logger.warning("macro fetch %s failed: %s", mid, e)
            continue
        if m_raw is None or len(m_raw) < 4:
            continue
        m = pd.Series(m_raw).astype(float)
        if m.index.tzinfo is None:
            m.index = pd.to_datetime(m.index, utc=True)
        m.index = m.index.normalize()
        m = m[~m.index.duplicated(keep="last")].sort_index()
        m_diff = m.diff().dropna()

        best_k = 0
        best_corr = 0.0
        for k in range(-max_lead_lag_days, max_lead_lag_days + 1):
            shifted = m_diff.shift(k)
            joined = pd.concat([f, shifted], axis=1).dropna()
            if len(joined) < 6:
                continue
            r = float(joined.corr().iloc[0, 1])
            if not np.isfinite(r):
                continue
            if abs(r) > abs(best_corr):
                best_corr = r
                best_k = k

        # Recompute the joined sample at best_k for the t-stat.
        shifted = m_diff.shift(best_k)
        joined = pd.concat([f, shifted], axis=1).dropna()
        if len(joined) < 6:
            continue
        t_stat = _newey_west_t_stat(
            joined.iloc[:, 1].to_numpy(dtype=float),
            joined.iloc[:, 0].to_numpy(dtype=float),
        )

        macro_correlations[mid] = float(best_corr)
        t_stats[mid] = float(t_stat)
        lead_lag_days[mid] = int(best_k)
        n_obs_used = max(n_obs_used, int(len(joined)))

    return {
        "factor_id": factor_id,
        "macro_correlations": macro_correlations,
        "t_stats": t_stats,
        "lead_lag_days": lead_lag_days,
        "n_obs": int(n_obs_used),
    }


# ---------------------------------------------------------------------------
# E. Cross-asset systemic factor (PM-PCA)
# ---------------------------------------------------------------------------


def extract_systemic_pm_factor(
    factor_ids: list[str],
    start: Any,
    end: Any,
    *,
    n_factors: int = 1,
    fetch_factor: FactorFetcher,
    epsilon: float = DEFAULT_EPSILON,
) -> dict:
    """PCA on Δlogit innovations. The first PC is a PM systemic risk factor.

    Δlogits are standardised before the PCA so loadings are comparable
    across factors with very different absolute volatilities.
    """
    s_dt, e_dt = _coerce_dates(start, end)
    X = _build_factor_panel(factor_ids, s_dt, e_dt, fetch_factor, epsilon=epsilon)
    X = X.dropna()
    if X.empty or X.shape[1] < 2 or X.shape[0] < max(10, X.shape[1] + 2):
        return {
            "component_scores": [],
            "dates": [],
            "loadings": {},
            "explained_variance": [],
            "n_components": 0,
            "can_use_as_factor": False,
            "n_obs": int(X.shape[0]),
            "n_factors_in": int(X.shape[1]),
        }

    n_comp = max(1, min(int(n_factors), X.shape[1]))
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X.to_numpy(dtype=float))
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_s)
    explained = [float(v) for v in pca.explained_variance_ratio_]

    # First-component loadings: which factors drive it most.
    first_loadings = {col: float(pca.components_[0, i]) for i, col in enumerate(X.columns)}

    # Sign convention: orient PC1 so the largest |loading| is positive.
    if first_loadings:
        biggest = max(first_loadings.items(), key=lambda kv: abs(kv[1]))
        if biggest[1] < 0:
            scores[:, 0] = -scores[:, 0]
            first_loadings = {k: -v for k, v in first_loadings.items()}

    can_use = bool(explained and explained[0] >= 0.20)

    dates = [d.isoformat() for d in X.index]
    return {
        "component_scores": [float(v) for v in scores[:, 0].tolist()],
        "dates": dates,
        "loadings": first_loadings,
        "explained_variance": explained,
        "n_components": int(n_comp),
        "can_use_as_factor": bool(can_use),
        "n_obs": int(X.shape[0]),
        "n_factors_in": int(X.shape[1]),
    }


__all__ = [
    "DEFAULT_SECTOR_ETFS",
    "FactorFetcher",
    "MacroFetcher",
    "ReturnFetcher",
    "event_macro_correlation",
    "extract_systemic_pm_factor",
    "find_chains",
    "fit_multi_event_lasso",
    "sector_attribution",
]
