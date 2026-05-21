"""Portfolio Optimizer — pure functions for selecting weights across alphas.

The hub holds 88 curated alpha strategies (see ``web/data/alpha_strategies.json``).
A user picks a subset of N pair-ids and this module suggests an *optimal* weight
vector under a chosen objective. All functions here are *pure* — no FastAPI,
no IO, no caching. They take a ``returns`` DataFrame (columns = pair_ids,
rows = aligned daily returns) and return a single dict with the same shape::

    {
        "weights":                       dict[str, float],   # sum = 1
        "expected_return":               float,              # annualised
        "expected_vol":                  float,              # annualised
        "sharpe":                        float,              # (μ - rf) / σ
        "marginal_risk_contribution":    dict[str, float],   # MRCᵢ = wᵢ·(Σw)ᵢ / σ²(p)
        "diversification_ratio":         float,              # (Σwᵢσᵢ) / σ(p)
        "effective_n":                   float,              # 1 / Σwᵢ²  (HHI inverse)
    }

Methods
-------
- ``equal_weight``: 1/N baseline, no optimisation.
- ``min_variance``: SLSQP min wᵀΣw  s.t.  Σw=1, min_w ≤ wᵢ ≤ max_w, w ≥ 0.
- ``mean_variance_max_sharpe``: SLSQP max (μᵀw - rf) / √(wᵀΣw) with shrinkage on μ.
- ``risk_parity_erc``: SLSQP minimise Var of risk contributions wᵢ·(Σw)ᵢ.
- ``hrp``: López de Prado (2016) Hierarchical Risk Parity. Does NOT invert Σ —
  uses correlation distance + single-linkage tree + recursive bisection on
  inverse-variance weights of clustered subsets. Robust to singular Σ.
- ``efficient_frontier``: 50 SLSQP solves at increasing target-vol levels.
- ``monte_carlo_drawdown``: stationary block-bootstrap of aggregate-portfolio
  daily PnL → distribution of max-drawdown over a ``horizon_days`` path.

Covariance estimator
--------------------
Default uses Ledoit–Wolf shrinkage (``sklearn.covariance.LedoitWolf``) which
shrinks the sample covariance toward a structured target (scaled identity).
Critical when N is comparable to T (the small-sample regime that breaks
Markowitz). Pass ``shrinkage='sample'`` to fall back to vanilla
``np.cov``.

Conventions
-----------
- Returns are interpreted as *daily simple returns* (not log).
- Annualisation factor is 252 (trading days).
- ``rf`` is supplied as an *annualised* rate; converted to per-period via
  ``rf_daily = rf / 252`` for the Sharpe objective.
- All weight constraints are box constraints + simplex constraint
  (no leverage, no shorts in the POC; ``min_w`` defaults to 0).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# covariance + diagnostics
# ---------------------------------------------------------------------------


def _estimate_cov(returns: pd.DataFrame, shrinkage: str = "ledoit_wolf") -> np.ndarray:
    """Return a ``(N, N)`` covariance matrix from a returns DataFrame.

    Daily-frequency cov (NOT annualised). Annualisation happens at the
    final summary-stats step so the optimiser objective stays in native
    daily units, which is numerically better-conditioned.
    """
    x = returns.to_numpy(dtype=float, copy=False)
    if shrinkage == "ledoit_wolf":
        if x.shape[0] < 2:
            # Degenerate sample; bail to a tiny-eps identity to keep math finite.
            n = x.shape[1]
            return np.eye(n) * 1e-8
        try:
            lw = LedoitWolf().fit(x)
            return np.asarray(lw.covariance_, dtype=float)
        except Exception:  # pragma: no cover — defensive
            logger.warning("LedoitWolf failed; falling back to sample cov.", exc_info=True)
    cov = np.cov(x, rowvar=False, ddof=1)
    if cov.ndim == 0:
        # Single-asset edge case: np.cov returns a 0-d array.
        cov = np.array([[float(cov)]])
    return np.asarray(cov, dtype=float)


def _portfolio_stats(
    weights: np.ndarray,
    mu_daily: np.ndarray,
    cov_daily: np.ndarray,
    rf_annual: float,
) -> tuple[float, float, float]:
    """Return (annualised return, annualised vol, annualised Sharpe)."""
    var_d = float(weights @ cov_daily @ weights)
    var_d = max(var_d, 0.0)
    vol_d = float(np.sqrt(var_d))
    ret_d = float(weights @ mu_daily)
    ret_a = ret_d * TRADING_DAYS_PER_YEAR
    vol_a = vol_d * float(np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (ret_a - rf_annual) / vol_a if vol_a > 0 else 0.0
    return ret_a, vol_a, sharpe


def _summarise(
    weights: np.ndarray,
    columns: list[str],
    returns: pd.DataFrame,
    cov_daily: np.ndarray,
    rf_annual: float,
) -> dict[str, Any]:
    """Build the canonical result-dict from a weight vector."""
    mu_daily = returns.mean(axis=0).to_numpy(dtype=float)
    ret_a, vol_a, sharpe = _portfolio_stats(weights, mu_daily, cov_daily, rf_annual)

    # Marginal risk contribution: MRCᵢ = wᵢ·(Σw)ᵢ / σ²(p), Σ = 1.
    sigma_w = cov_daily @ weights
    var_p = float(weights @ sigma_w)
    mrc = (weights * sigma_w) / var_p if var_p > 0 else np.zeros_like(weights)

    # Diversification ratio = Σ wᵢ σᵢ / σ(p).
    sigmas = np.sqrt(np.clip(np.diag(cov_daily), 0.0, None))
    weighted_sum_sigma = float(weights @ sigmas)
    sigma_p = float(np.sqrt(max(var_p, 0.0)))
    div_ratio = weighted_sum_sigma / sigma_p if sigma_p > 0 else 1.0

    # Effective N = 1 / HHI = 1 / Σ wᵢ²
    hhi = float((weights**2).sum())
    eff_n = 1.0 / hhi if hhi > 0 else float(len(columns))

    return {
        "weights": {c: float(w) for c, w in zip(columns, weights, strict=True)},
        "expected_return": float(ret_a),
        "expected_vol": float(vol_a),
        "sharpe": float(sharpe),
        "marginal_risk_contribution": {c: float(v) for c, v in zip(columns, mrc, strict=True)},
        "diversification_ratio": float(div_ratio),
        "effective_n": float(eff_n),
    }


# ---------------------------------------------------------------------------
# small SLSQP helper
# ---------------------------------------------------------------------------


def _box_constraints(n: int, min_w: float, max_w: float) -> list[tuple[float, float]]:
    return [(float(min_w), float(max_w))] * n


def _simplex_constraint() -> dict[str, Any]:
    return {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}


def _initial_weights(n: int, max_w: float, min_w: float) -> np.ndarray:
    """Feasible starting point for SLSQP."""
    base = np.full(n, 1.0 / n)
    base = np.clip(base, min_w, max_w)
    s = base.sum()
    if s > 0:
        base = base / s
    return base


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.shape[1] < 1:
        raise ValueError("returns must have at least one column")
    # Drop all-nan columns and rows; preserve order.
    cleaned = returns.dropna(how="all", axis=1).dropna(how="any", axis=0)
    if cleaned.shape[0] < 2:
        raise ValueError(
            f"returns has only {cleaned.shape[0]} usable rows after dropna; "
            "need at least 2 for covariance estimation."
        )
    if cleaned.shape[1] < 1:
        raise ValueError("no usable columns left after dropna")
    return cleaned


# ---------------------------------------------------------------------------
# allocators
# ---------------------------------------------------------------------------


def equal_weight(
    returns: pd.DataFrame,
    rf: float = 0.045,
    shrinkage: str = "ledoit_wolf",
) -> dict[str, Any]:
    """Naive 1/N baseline."""
    df = _validate_returns(returns)
    n = df.shape[1]
    w = np.full(n, 1.0 / n)
    cov = _estimate_cov(df, shrinkage=shrinkage)
    return _summarise(w, list(df.columns), df, cov, rf)


def min_variance(
    returns: pd.DataFrame,
    max_w: float = 0.30,
    min_w: float = 0.0,
    rf: float = 0.045,
    shrinkage: str = "ledoit_wolf",
) -> dict[str, Any]:
    """Min wᵀΣw subject to simplex + box constraints (long-only)."""
    df = _validate_returns(returns)
    n = df.shape[1]
    cov = _estimate_cov(df, shrinkage=shrinkage)

    def obj(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov @ w)

    res = minimize(
        obj,
        _initial_weights(n, max_w, min_w),
        jac=grad,
        method="SLSQP",
        bounds=_box_constraints(n, min_w, max_w),
        constraints=[_simplex_constraint()],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = _normalise_box(res.x, min_w, max_w)
    return _summarise(w, list(df.columns), df, cov, rf)


def mean_variance_max_sharpe(
    returns: pd.DataFrame,
    rf: float = 0.045,
    max_w: float = 0.30,
    min_w: float = 0.0,
    shrink_mu: float = 0.5,
    shrinkage: str = "ledoit_wolf",
) -> dict[str, Any]:
    """Max Sharpe (Markowitz tangency) with shrinkage on the mean estimate.

    ``shrink_mu`` ∈ [0, 1] interpolates the per-asset mean toward zero:
    ``μ_shrunk = (1 - shrink_mu) * μ_sample``. (Default 0.5 — a James-Stein-ish
    soft prior that the in-sample mean is half-noise.) A pure ``shrink_mu=1``
    collapses to a min-variance solve.
    """
    df = _validate_returns(returns)
    n = df.shape[1]
    cov = _estimate_cov(df, shrinkage=shrinkage)
    mu_d = df.mean(axis=0).to_numpy(dtype=float)
    mu_d_shrunk = (1.0 - float(shrink_mu)) * mu_d

    rf_d = float(rf) / TRADING_DAYS_PER_YEAR

    def neg_sharpe(w: np.ndarray) -> float:
        v = float(w @ cov @ w)
        if v <= 0:
            return 1e6
        ret = float(w @ mu_d_shrunk) - rf_d
        return -ret / float(np.sqrt(v))

    res = minimize(
        neg_sharpe,
        _initial_weights(n, max_w, min_w),
        method="SLSQP",
        bounds=_box_constraints(n, min_w, max_w),
        constraints=[_simplex_constraint()],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    w = _normalise_box(res.x, min_w, max_w)
    return _summarise(w, list(df.columns), df, cov, rf)


def risk_parity_erc(
    returns: pd.DataFrame,
    max_w: float = 0.30,
    min_w: float = 0.0,
    rf: float = 0.045,
    shrinkage: str = "ledoit_wolf",
) -> dict[str, Any]:
    """Equal Risk Contribution (Maillard, Roncalli, Teïletche 2010).

    Minimise Var(rcᵢ) where rcᵢ = wᵢ·(Σw)ᵢ. At the optimum, every asset
    contributes the same fraction of total portfolio variance — hence
    "equal risk contribution."
    """
    df = _validate_returns(returns)
    n = df.shape[1]
    cov = _estimate_cov(df, shrinkage=shrinkage)

    def obj(w: np.ndarray) -> float:
        sigma_w = cov @ w
        rc = w * sigma_w  # per-asset contribution to variance
        # Minimise pairwise squared differences (equivalent to variance of rc).
        target = rc.mean()
        return float(np.sum((rc - target) ** 2))

    res = minimize(
        obj,
        _initial_weights(n, max_w, min_w),
        method="SLSQP",
        bounds=_box_constraints(n, min_w, max_w),
        constraints=[_simplex_constraint()],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = _normalise_box(res.x, min_w, max_w)
    return _summarise(w, list(df.columns), df, cov, rf)


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (López de Prado 2016)
# ---------------------------------------------------------------------------


def _corr_distance(corr: np.ndarray) -> np.ndarray:
    """Distance metric: dᵢⱼ = √( ½ · (1 - ρᵢⱼ) ) ∈ [0, 1]."""
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Return the leaf-order from a single-linkage tree (López de Prado §4.3)."""
    link = link.astype(int)
    n_assets = link.shape[0] + 1
    # Each linkage row: [a, b, dist, n_leaves]. a/b are cluster ids; ids ≥ N
    # refer to internal nodes. Walk recursively to expand clusters.
    order: list[int] = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n_assets:
        new_order: list[int] = []
        for i in order:
            if i < n_assets:
                new_order.append(i)
            else:
                # Replace internal node with its two children.
                row = int(i - n_assets)
                new_order.extend([int(link[row, 0]), int(link[row, 1])])
        order = new_order
    return order


def _ivp_weights(cov_sub: np.ndarray) -> np.ndarray:
    """Inverse-variance portfolio for a sub-cluster (no Σ inversion)."""
    inv_var = 1.0 / np.clip(np.diag(cov_sub), 1e-12, None)
    return inv_var / inv_var.sum()


def _cluster_var(cov_sub: np.ndarray) -> float:
    """Variance of an IVP-weighted cluster — used for bisection split sizing."""
    w = _ivp_weights(cov_sub)
    return float(w @ cov_sub @ w)


def _recursive_bisection(cov: np.ndarray, sorted_ix: list[int]) -> np.ndarray:
    """López de Prado §4.4: top-down recursive bisection allocation."""
    n = cov.shape[0]
    weights = np.ones(n)
    clusters: list[list[int]] = [list(sorted_ix)]
    while clusters:
        new_clusters: list[list[int]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            # Split the cluster in half along the quasi-diagonal order.
            mid = len(c) // 2
            left = c[:mid]
            right = c[mid:]
            cov_left = cov[np.ix_(left, left)]
            cov_right = cov[np.ix_(right, right)]
            v_left = _cluster_var(cov_left)
            v_right = _cluster_var(cov_right)
            # Allocate inversely to cluster variance.
            total = v_left + v_right
            alpha = 0.5 if total <= 0 else 1.0 - v_left / total
            # alpha goes to LEFT, (1 - alpha) to RIGHT.
            for ix in left:
                weights[ix] *= alpha
            for ix in right:
                weights[ix] *= 1.0 - alpha
            new_clusters.append(left)
            new_clusters.append(right)
        clusters = new_clusters
    return weights


def hrp(
    returns: pd.DataFrame,
    rf: float = 0.045,
    shrinkage: str = "ledoit_wolf",
) -> dict[str, Any]:
    """Hierarchical Risk Parity — López de Prado (2016).

    Steps:
      1. Build the correlation matrix from ``returns``.
      2. Convert to distance dᵢⱼ = √( ½·(1-ρᵢⱼ) ).
      3. Single-linkage hierarchical clustering on the condensed distance.
      4. Quasi-diagonalise: reorder assets so similar pairs are adjacent.
      5. Recursive bisection: split into halves along the quasi-diagonal,
         allocate inversely-proportional to each half's IVP variance.

    No Σ inversion → numerically robust under near-singular covariance.
    """
    df = _validate_returns(returns)
    n = df.shape[1]
    cols = list(df.columns)

    if n == 1:
        w = np.array([1.0])
        cov = _estimate_cov(df, shrinkage=shrinkage)
        return _summarise(w, cols, df, cov, rf)

    cov = _estimate_cov(df, shrinkage=shrinkage)
    # Correlation from cov (numerically safer than re-computing on returns —
    # this way the LedoitWolf shrinkage propagates).
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    denom = np.outer(sd, sd)
    corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    dist = _corr_distance(corr)
    # squareform requires zero diagonal.
    np.fill_diagonal(dist, 0.0)
    try:
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
    except Exception:
        # Fallback: equal-weight if clustering fails (rare; e.g. all-equal corr).
        logger.warning("HRP linkage failed; falling back to equal-weight.", exc_info=True)
        w = np.full(n, 1.0 / n)
        return _summarise(w, cols, df, cov, rf)

    sorted_ix = _quasi_diag(link)
    weights = _recursive_bisection(cov, sorted_ix)
    # Normalise (numerical safety).
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return _summarise(weights, cols, df, cov, rf)


# ---------------------------------------------------------------------------
# efficient frontier
# ---------------------------------------------------------------------------


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    max_w: float = 0.30,
    min_w: float = 0.0,
    rf: float = 0.045,
    shrinkage: str = "ledoit_wolf",
    shrink_mu: float = 0.5,
) -> list[dict[str, float]]:
    """Trace ``n_points`` along the efficient frontier (annualised vol, return).

    Strategy: solve min-variance and max-sharpe to anchor the frontier, then
    sweep ``n_points`` target *returns* uniformly between the two and solve
    min-variance subject to ``μᵀw ≥ target_ret``. Skips infeasible targets.
    """
    df = _validate_returns(returns)
    n = df.shape[1]
    cov = _estimate_cov(df, shrinkage=shrinkage)
    mu_d = df.mean(axis=0).to_numpy(dtype=float)
    mu_d_shrunk = (1.0 - float(shrink_mu)) * mu_d

    # Anchor returns: equal-weight return is a safe min-bound; max single-asset
    # return is the upper bound (with cap respected).
    mv_w = min_variance(df, max_w=max_w, min_w=min_w, rf=rf, shrinkage=shrinkage)
    mv_ret_d = float(np.array([mv_w["weights"][c] for c in df.columns]) @ mu_d_shrunk)
    # Upper anchor: greedily concentrate weight on the highest-mu asset, capped by max_w.
    order = np.argsort(-mu_d_shrunk)
    upper_w = np.zeros(n)
    remaining = 1.0
    for idx in order:
        take = min(max_w, remaining)
        upper_w[idx] = take
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 0 and upper_w.sum() > 0:
        # Insufficient cap budget — re-normalise (shouldn't happen for max_w ≥ 1/n).
        upper_w = upper_w / upper_w.sum()
    upper_ret_d = float(upper_w @ mu_d_shrunk)

    if upper_ret_d <= mv_ret_d:
        # Degenerate: only one feasible portfolio. Return min-variance point.
        return [
            {
                "expected_return": mv_w["expected_return"],
                "expected_vol": mv_w["expected_vol"],
                "sharpe": mv_w["sharpe"],
            }
        ]

    targets = np.linspace(mv_ret_d, upper_ret_d, max(int(n_points), 2))
    points: list[dict[str, float]] = []

    def obj(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov @ w)

    for t in targets:
        cons = [
            _simplex_constraint(),
            {"type": "ineq", "fun": lambda w, t=t: float(w @ mu_d_shrunk - t)},
        ]
        res = minimize(
            obj,
            _initial_weights(n, max_w, min_w),
            jac=grad,
            method="SLSQP",
            bounds=_box_constraints(n, min_w, max_w),
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if not res.success:
            continue
        w = _normalise_box(res.x, min_w, max_w)
        ret_a, vol_a, sharpe = _portfolio_stats(w, mu_d, cov, rf)
        points.append(
            {
                "expected_return": float(ret_a),
                "expected_vol": float(vol_a),
                "sharpe": float(sharpe),
            }
        )
    # De-duplicate by vol and sort.
    points.sort(key=lambda p: p["expected_vol"])
    return points


# ---------------------------------------------------------------------------
# stationary block bootstrap drawdown
# ---------------------------------------------------------------------------


def monte_carlo_drawdown(
    weights: dict[str, float],
    returns: pd.DataFrame,
    n_paths: int = 10000,
    horizon_days: int = 252,
    block: int = 20,
    seed: int | None = 7,
) -> dict[str, Any]:
    """Stationary block-bootstrap drawdown distribution.

    For each path, draw blocks of length ``block`` (with random start) until
    the path is at least ``horizon_days`` long, truncate, compute the
    portfolio cumulative-return curve under ``weights``, then the maximum
    drawdown as ``max_t (peak_t - curve_t) / peak_t`` (positive number;
    fraction of equity).

    Returns p05/p50/p95 of the max-drawdown distribution plus the mean and
    stdev for plotting.
    """
    df = _validate_returns(returns)
    cols = list(df.columns)
    w = np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)
    s = w.sum()
    if s > 0:
        w = w / s
    # Portfolio daily returns (T,).
    port = df.to_numpy(dtype=float) @ w
    t = port.shape[0]
    if t < 2:
        return {
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "n_paths": 0,
            "horizon_days": int(horizon_days),
            "block": int(block),
        }
    block = max(1, min(int(block), t))
    horizon = max(1, int(horizon_days))
    n_paths = max(1, int(n_paths))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))
    # Generate ``n_paths × n_blocks`` random start indices in one shot.
    starts = rng.integers(0, t, size=(n_paths, n_blocks))
    # Build path-row index matrix shape (n_paths, n_blocks * block).
    offsets = np.arange(block)[None, None, :]
    raw_ix = (starts[:, :, None] + offsets) % t  # circular wrap
    paths_ix = raw_ix.reshape(n_paths, n_blocks * block)[:, :horizon]
    # Gather returns: shape (n_paths, horizon).
    paths = port[paths_ix]
    # Cumulative equity curve: (1 + r).cumprod
    equity = np.cumprod(1.0 + paths, axis=1)
    # Running peak per path.
    peaks = np.maximum.accumulate(equity, axis=1)
    drawdowns = (peaks - equity) / peaks  # ≥ 0
    max_dd = drawdowns.max(axis=1)

    return {
        "p05": float(np.percentile(max_dd, 5)),
        "p50": float(np.percentile(max_dd, 50)),
        "p95": float(np.percentile(max_dd, 95)),
        "mean": float(max_dd.mean()),
        "std": float(max_dd.std(ddof=1)) if max_dd.size > 1 else 0.0,
        "n_paths": int(n_paths),
        "horizon_days": int(horizon),
        "block": int(block),
    }


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def _normalise_box(w: np.ndarray, min_w: float, max_w: float) -> np.ndarray:
    """Project a near-feasible solution onto the simplex with box constraints.

    SLSQP can leave tiny constraint violations (∑w ≠ 1 by ε). We clip into
    [min_w, max_w] then renormalise. If clipping breaks the cap, do one
    pass of water-filling.
    """
    w = np.asarray(w, dtype=float)
    w = np.clip(w, min_w, max_w)
    s = w.sum()
    if s <= 0:
        n = len(w)
        return np.full(n, 1.0 / n)
    w = w / s
    # If renormalisation pushed any element above max_w, redistribute.
    for _ in range(5):
        over = w > max_w + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        free = ~over & (w < max_w - 1e-12)
        if not free.any():
            break
        w[free] += excess * (w[free] / w[free].sum())
    return w
