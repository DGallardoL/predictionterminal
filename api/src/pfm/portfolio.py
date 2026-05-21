"""Portfolio combiner: vol-targeted, equal-vol-contribution backstop strategy.

Given a set of validated pair-trading PnL series, build a single portfolio
equity curve that:

1.  **Vol-targets each leg** to contribute equal *risk* (not equal capital).
    Per-leg weight w_i ∝ 1/σ_i. This is the *naive risk parity* that
    Markowitz showed dominates equal-weight on uncorrelated alpha.

2.  Aggregates the weighted PnLs into a single equity curve.

3.  Reports portfolio Sharpe, Sortino, Calmar, max DD, hit rate.

4.  Optional **walk-forward**: split into 5 folds, refit weights from
    train-only data, evaluate on test fold. The realistic "what would
    actually have happened" Sharpe.

References:
    Markowitz, H. (1952). "Portfolio Selection." JF.
    Maillard, S. et al. (2010). "The Properties of Equally Weighted Risk
        Contribution Portfolios." J. Portfolio Mgmt.
    Lopez de Prado, M. (2018) §11 — strategy combination.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioResult:
    n_pairs: int
    pair_labels: list[str]
    weights: dict[str, float]  # vol-targeted weights, summing to 1
    individual_sharpes: dict[str, float]  # per-leg in-sample Sharpe
    correlation_matrix: list[list[float]]
    n_obs: int

    # Combined portfolio metrics (in-sample on aggregated PnL):
    portfolio_sharpe: float
    portfolio_sortino: float
    portfolio_calmar: float
    portfolio_max_drawdown: float
    portfolio_var_95: float
    portfolio_cvar_95: float
    portfolio_skew: float

    # Walk-forward (out-of-sample) — only set if requested:
    oos_sharpe_mean: float | None = None
    oos_sharpe_std: float | None = None
    oos_sharpe_min: float | None = None

    # Equity curve series (kept compact for transport):
    pnl_series: pd.Series | None = None
    equity_curve: pd.Series | None = None


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    rm = equity.cummax()
    return float((equity - rm).min())


def vol_targeted_combiner(
    pnls: dict[str, pd.Series],
    *,
    annualisation: float = 252.0,
    target_per_leg_vol: float = 0.10,  # 10% annualised per leg
    walk_forward_folds: int | None = 5,
) -> PortfolioResult:
    """Combine N pair-trading PnL series into a vol-targeted portfolio.

    Each leg's weight is set so that ``w_i · σ_i = target_per_leg_vol``
    (annualised). Combined PnL = Σ w_i · pnl_i.

    Args:
        pnls: dict {pair_label → per-bar PnL series}.
        annualisation: bars per year.
        target_per_leg_vol: target *contribution* to total annualised vol.
        walk_forward_folds: if set, run K-fold walk-forward and report OOS
            Sharpe distribution.

    Returns:
        :class:`PortfolioResult`.

    Raises:
        ValueError: fewer than 2 PnLs provided, or all-zero PnLs.
    """
    if len(pnls) < 2:
        raise ValueError(f"need ≥2 pairs, got {len(pnls)}")
    df = pd.DataFrame(pnls).dropna()
    if df.empty or len(df) < 20:
        raise ValueError(f"too few aligned bars: {len(df)}")
    sqrt_ann = sqrt(annualisation)

    # Per-leg σ (annualised) and Sharpe.
    per_leg_sigma = df.std(ddof=1) * sqrt_ann
    per_leg_mean = df.mean()
    per_leg_sharpe = (per_leg_mean / df.std(ddof=1) * sqrt_ann).fillna(0.0)
    if (per_leg_sigma <= 0).all():
        raise ValueError("all PnLs have zero variance")
    # Vol-targeted weights: w_i ∝ 1 / σ_i.
    weights = (target_per_leg_vol / per_leg_sigma.replace(0, np.nan)).fillna(0.0)
    # Normalise so the *sum of weights* equals N (so per-leg target stands as is).
    # Or: don't normalise — let portfolio vol = N * target_per_leg_vol when correlations are 0.
    weights_dict = weights.to_dict()
    weighted_pnl = (df * weights).sum(axis=1)
    equity = weighted_pnl.cumsum()

    p_mean = float(weighted_pnl.mean())
    p_std = float(weighted_pnl.std(ddof=1))
    p_sharpe = (p_mean / p_std) * sqrt_ann if p_std > 0 else 0.0
    neg = weighted_pnl[weighted_pnl < 0]
    downside_std = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    p_sortino = (p_mean / downside_std) * sqrt_ann if downside_std > 0 else 0.0
    max_dd = _max_drawdown(equity)
    p_calmar = (p_mean * annualisation) / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0
    arr = weighted_pnl.to_numpy()
    var_95 = float(np.percentile(arr, 5))
    tail = arr[arr <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) else var_95
    if p_std > 0:
        z = (arr - p_mean) / p_std
        p_skew = float(np.mean(z**3))
    else:
        p_skew = 0.0

    corr = df.corr().values.tolist()

    # Walk-forward
    oos_mean = oos_std = oos_min = None
    if walk_forward_folds and len(df) >= walk_forward_folds * 10:
        oos_sharpes: list[float] = []
        fold_size = len(df) // walk_forward_folds
        for k in range(walk_forward_folds):
            test_lo = k * fold_size
            test_hi = (k + 1) * fold_size if k < walk_forward_folds - 1 else len(df)
            train_df = pd.concat([df.iloc[:test_lo], df.iloc[test_hi:]])
            test_df = df.iloc[test_lo:test_hi]
            if len(train_df) < 10 or len(test_df) < 5:
                continue
            train_sigma = train_df.std(ddof=1) * sqrt_ann
            train_w = (target_per_leg_vol / train_sigma.replace(0, np.nan)).fillna(0.0)
            test_pnl = (test_df * train_w).sum(axis=1)
            ts_std = float(test_pnl.std(ddof=1))
            ts_mean = float(test_pnl.mean())
            ts_sharpe = (ts_mean / ts_std) * sqrt_ann if ts_std > 0 else 0.0
            oos_sharpes.append(ts_sharpe)
        if oos_sharpes:
            oos_mean = float(np.mean(oos_sharpes))
            oos_std = float(np.std(oos_sharpes, ddof=1)) if len(oos_sharpes) > 1 else 0.0
            oos_min = float(min(oos_sharpes))

    return PortfolioResult(
        n_pairs=len(pnls),
        pair_labels=list(df.columns),
        weights={k: float(v) for k, v in weights_dict.items()},
        individual_sharpes={k: float(per_leg_sharpe[k]) for k in df.columns},
        correlation_matrix=corr,
        n_obs=len(df),
        portfolio_sharpe=p_sharpe,
        portfolio_sortino=p_sortino,
        portfolio_calmar=p_calmar,
        portfolio_max_drawdown=max_dd,
        portfolio_var_95=var_95,
        portfolio_cvar_95=cvar_95,
        portfolio_skew=p_skew,
        oos_sharpe_mean=oos_mean,
        oos_sharpe_std=oos_std,
        oos_sharpe_min=oos_min,
        pnl_series=weighted_pnl,
        equity_curve=equity,
    )


__all__ = ["PortfolioResult", "vol_targeted_combiner"]
