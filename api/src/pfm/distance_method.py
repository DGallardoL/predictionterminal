"""Gatev-Goetzmann-Rouwenhorst (2006) Distance Method.

The classical pairs-trading benchmark every modern method must beat.
Method:

1.  **Formation period** (e.g., 12 months): for every pair of normalised
    cumulative-return series, compute the SSD (sum of squared
    differences). This is the "distance" metric.
2.  Rank pairs by smallest SSD (most "alike" historically).
3.  **Trading period** (e.g., 6 months): for the top-k closest pairs,
    open trade when normalised-price spread exceeds 2σ_formation,
    close when it crosses 0.

Intuition: pairs with low formation-period SSD have moved together
historically. Any large divergence in the trading period is "abnormal"
and likely to revert.

For our prediction-market setup, we adapt:
- Probability series replace returns (already comparable units, no
  normalisation needed beyond mean-centering).
- Formation period and trading period configurable.
- Uses just SSD on normalised levels (the classical method).

Reference:
    Gatev, E., Goetzmann, W. & Rouwenhorst, K. G. (2006). "Pairs Trading:
    Performance of a Relative-Value Arbitrage Rule." Review of Financial
    Studies 19(3), 797–827.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistanceMethodResult:
    pair_a: str
    pair_b: str
    formation_ssd: float
    formation_sigma: float
    n_trading_bars: int
    n_trades: int
    trade_pnl: float
    sharpe: float
    pnl_series: pd.Series
    positions: pd.Series


def _normalise_series(s: pd.Series) -> pd.Series:
    """Subtract mean, divide by std (z-score normalisation)."""
    sd = float(s.std(ddof=1))
    if sd <= 0:
        return s - s.mean()
    return (s - s.mean()) / sd


def distance_method(
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    a_id: str = "A",
    b_id: str = "B",
    formation_fraction: float = 0.5,
    entry_sigma: float = 2.0,
    exit_sigma: float = 0.0,
    annualisation: float = 252.0,
) -> DistanceMethodResult:
    """Run the GGR Distance Method on a single pair.

    Args:
        p_a / p_b: aligned probability series.
        a_id / b_id: labels.
        formation_fraction: first f% used for distance / σ estimation;
            remaining (1−f) is the trading period.
        entry_sigma: open trade when |spread − μ_formation| > entry_sigma · σ_formation.
        exit_sigma: close when |spread − μ_formation| < exit_sigma · σ_formation
            (default 0 = back to the formation mean).
        annualisation: bars per year for Sharpe.

    Returns:
        :class:`DistanceMethodResult`.
    """
    df = pd.concat({"a": p_a, "b": p_b}, axis=1).dropna()
    n = len(df)
    if n < 30:
        raise ValueError(f"distance_method: need ≥30 aligned bars, got {n}")
    n_form = int(round(n * formation_fraction))
    if n_form < 10 or n - n_form < 10:
        raise ValueError(
            f"distance_method: bad formation/trading split: n_form={n_form}, n_trade={n - n_form}"
        )

    # Formation period: normalise both legs, compute SSD and the spread σ.
    form = df.iloc[:n_form].copy()
    form["a_n"] = _normalise_series(form["a"])
    form["b_n"] = _normalise_series(form["b"])
    spread_form = form["a_n"] - form["b_n"]
    ssd = float(np.sum((form["a_n"] - form["b_n"]) ** 2))
    mu_f = float(spread_form.mean())
    sd_f = float(spread_form.std(ddof=1))
    if sd_f <= 0:
        raise ValueError("formation-period spread has zero variance")

    # Trading period: continue normalising via formation-period μ/σ.
    trade = df.iloc[n_form:].copy()
    # Use formation-period μ and σ to normalise (no peeking).
    mu_a = float(form["a"].mean())
    sd_a = float(form["a"].std(ddof=1))
    mu_b = float(form["b"].mean())
    sd_b = float(form["b"].std(ddof=1))
    if sd_a <= 0 or sd_b <= 0:
        raise ValueError("formation-period leg has zero variance")
    trade["a_n"] = (trade["a"] - mu_a) / sd_a
    trade["b_n"] = (trade["b"] - mu_b) / sd_b
    spread_trade = trade["a_n"] - trade["b_n"]
    z_trade = (spread_trade - mu_f) / sd_f

    # State machine
    state = 0
    pos = np.zeros(len(z_trade), dtype=int)
    for i, zi in enumerate(z_trade.values):
        if pd.isna(zi):
            pos[i] = state
            continue
        if state == 0:
            if zi <= -entry_sigma:
                state = 1  # spread below mean → long spread (long A, short B)
            elif zi >= entry_sigma:
                state = -1
        elif abs(zi) < exit_sigma + 1e-9:
            state = 0
        pos[i] = state

    # Per-bar PnL: position from t-1 × Δspread_trade
    dspread = spread_trade.diff().fillna(0).values
    pnl = np.concatenate([[0], pos[:-1].astype(float)]) * dspread
    pnl_series = pd.Series(pnl, index=trade.index, name="pnl")
    pos_series = pd.Series(pos, index=trade.index, name="position")

    n_trades = int((np.diff(np.concatenate([[0], pos])) != 0).sum() // 2)
    trade_pnl = float(pnl.sum())
    sd_pnl = float(pnl_series.std(ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = (
        (float(pnl_series.mean()) / sd_pnl) * float(np.sqrt(annualisation)) if sd_pnl > 0 else 0.0
    )

    return DistanceMethodResult(
        pair_a=a_id,
        pair_b=b_id,
        formation_ssd=ssd,
        formation_sigma=sd_f,
        n_trading_bars=len(trade),
        n_trades=n_trades,
        trade_pnl=trade_pnl,
        sharpe=float(sharpe),
        pnl_series=pnl_series,
        positions=pos_series,
    )


__all__ = ["DistanceMethodResult", "distance_method"]
