"""Alternative signal generators for the cointegration spread.

The default `pairs_backtest` uses fixed-z entry/exit (e.g., entry=2σ).
This module offers four alternative generators rooted in technical-
analysis literature, all returning the same ±1/0 position series so they
plug into `pairs_backtest`'s downstream PnL/equity machinery:

1.  **Bollinger Bands** (Bollinger 1983). Standard rolling μ ± k·σ
    where the spread is "outside the band" triggers entry. Differs from
    fixed-z by being more agile when vol regime shifts.

2.  **RSI on the spread** (Wilder 1978). Compute RSI on the spread's
    differences; enter LONG-spread when RSI < 30 (oversold), SHORT when
    RSI > 70. The "extremes" interpretation is a technical-analysis
    classic but works conceptually: an *oscillator* identifies temporary
    over/undershoots without requiring a fixed-vol assumption.

3.  **MACD on the spread** (Appel 1979). MACD = EMA(12) − EMA(26) on the
    spread; signal = EMA(9) of MACD. Cross-overs flag trend regime
    changes — useful as a *meta-filter*: don't enter on z if MACD is
    still trending against you.

4.  **Half-life-adaptive z-score**. Set the rolling window to round(K·½-life)
    where K is a multiplier (default 5). When half-life is 0.5d → window=3
    (very fast); when half-life is 10d → window=50. Adapts to the pair's
    intrinsic timescale automatically.

References:
    Bollinger, J. (1983). *Bollinger on Bollinger Bands*.
    Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*.
    Appel, G. (1979). *The Moving Average Convergence-Divergence Trading Method*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ──────────────────────── Bollinger Bands ─────────────────────────────


def bollinger_signals(
    spread: pd.Series,
    *,
    window: int = 20,
    k_entry: float = 2.0,
    k_exit: float = 0.0,
) -> pd.Series:
    """Position series from Bollinger-Bands rule.

    Bands = rolling_mean(window) ± k · rolling_std(window). Long-spread
    when spread is below lower band; short when above upper; flatten when
    spread crosses the middle band.

    Equivalent to fixed-z but adapts band width if rolling σ changes — so
    *fewer entries* in high-vol regimes (good for risk control).

    Args:
        spread: per-bar spread series.
        window: rolling window for μ and σ.
        k_entry: band multiplier for entry (default 2.0).
        k_exit: middle-band multiplier; 0.0 = exit at the rolling mean.

    Returns:
        Position series in {−1, 0, +1}.
    """
    mu = spread.rolling(window=window, min_periods=max(5, window // 2)).mean()
    sd = spread.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
    upper_e = mu + k_entry * sd
    lower_e = mu - k_entry * sd
    upper_x = mu + k_exit * sd
    lower_x = mu - k_exit * sd
    state = 0
    out = np.zeros(len(spread), dtype=int)
    sp = spread.values
    for i, s in enumerate(sp):
        ue, le = upper_e.iloc[i], lower_e.iloc[i]
        ux, lx = upper_x.iloc[i], lower_x.iloc[i]
        if any(pd.isna(v) for v in (ue, le, ux, lx)):
            out[i] = state
            continue
        if state == 0:
            if s <= le:
                state = 1
            elif s >= ue:
                state = -1
        elif state == 1:
            if s >= lx:
                state = 0  # crossed mean from below
        elif state == -1 and s <= ux:
            state = 0
        out[i] = state
    return pd.Series(out, index=spread.index, name="position")


# ──────────────────────────── RSI ─────────────────────────────────────


def rsi(spread: pd.Series, *, window: int = 14) -> pd.Series:
    """Wilder's RSI on the spread's first differences."""
    diff = spread.diff()
    up = diff.clip(lower=0.0)
    dn = (-diff).clip(lower=0.0)
    # Wilder smoothing = EMA with α = 1/window.
    avg_up = up.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def rsi_signals(
    spread: pd.Series,
    *,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    exit_band: float = 50.0,
) -> pd.Series:
    """Position from RSI extrema. Long-spread when oversold; short when
    overbought; exit when RSI crosses the centre line (default 50)."""
    r = rsi(spread, window=window)
    state = 0
    out = np.zeros(len(spread), dtype=int)
    rv = r.values
    for i, ri in enumerate(rv):
        if pd.isna(ri):
            out[i] = state
            continue
        if state == 0:
            if ri < oversold:
                state = 1
            elif ri > overbought:
                state = -1
        elif state == 1:
            if ri > exit_band:
                state = 0
        elif state == -1 and ri < exit_band:
            state = 0
        out[i] = state
    return pd.Series(out, index=spread.index, name="position")


# ──────────────────────────── MACD ────────────────────────────────────


def macd_signals(
    spread: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal_window: int = 9,
) -> pd.Series:
    """MACD-crossover position: long-spread on bullish cross (MACD > signal),
    short on bearish cross. Reverses on each cross — no separate exit.

    Best used as a *meta-filter* combined with Bollinger Bands: only take
    Bollinger entry signals when MACD is in the same direction.
    """
    ema_fast = spread.ewm(span=fast, adjust=False).mean()
    ema_slow = spread.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal_window, adjust=False).mean()
    diff = macd - sig
    state = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    return pd.Series(state, index=spread.index, name="position")


# ─────────────────── half-life-adaptive z-score ──────────────────────


def adaptive_zscore_signals(
    spread: pd.Series,
    half_life_bars: float,
    *,
    multiplier: float = 5.0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    min_window: int = 5,
    max_window: int = 80,
) -> pd.Series:
    """Z-score with rolling-window length = round(multiplier · half_life).

    When half-life is short, window is short (fast adaptation). When
    half-life is long, window is long (slow adaptation). Uses the same
    state-machine as the default backtest's z-score generator.
    """
    window = int(round(max(min_window, min(max_window, multiplier * half_life_bars))))
    mu = spread.rolling(window=window, min_periods=max(min_window, window // 2)).mean()
    sd = spread.rolling(window=window, min_periods=max(min_window, window // 2)).std(ddof=1)
    z = (spread - mu) / sd
    state = 0
    out = np.zeros(len(spread), dtype=int)
    for i, zi in enumerate(z.values):
        if pd.isna(zi):
            out[i] = state
            continue
        if state == 0:
            if zi <= -entry_z:
                state = 1
            elif zi >= entry_z:
                state = -1
        elif state == 1:
            if abs(zi) < exit_z or zi <= -stop_z:
                state = 0
        elif state == -1 and (abs(zi) < exit_z or zi >= stop_z):
            state = 0
        out[i] = state
    return pd.Series(out, index=spread.index, name="position")


# ──────────────────────── PnL evaluator ───────────────────────────────


def evaluate_signal(
    spread: pd.Series,
    positions: pd.Series,
    *,
    annualisation: float = 252.0,
) -> dict[str, float]:
    """Evaluate a position series's PnL on the spread.

    Returns:
        Dict with sharpe, n_trades, hit_rate, max_drawdown, sortino, calmar.
    """
    spread = spread.dropna().sort_index()
    pos = positions.reindex(spread.index).fillna(0).astype(int)
    dspread = spread.diff().fillna(0.0)
    pnl = pos.shift(1).fillna(0).astype(float) * dspread
    pnl_arr = pnl.to_numpy()
    n = len(pnl_arr)
    if n < 5:
        return {
            "sharpe": 0.0,
            "n_trades": 0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
        }
    sqrt_ann = float(np.sqrt(annualisation))
    pnl_std = float(pnl.std(ddof=1)) if n > 1 else 0.0
    pnl_mean = float(pnl.mean())
    sharpe = (pnl_mean / pnl_std) * sqrt_ann if pnl_std > 0 else 0.0
    neg = pnl_arr[pnl_arr < 0]
    downside_std = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
    sortino = (pnl_mean / downside_std) * sqrt_ann if downside_std > 0 else 0.0

    # Trade counting: count direction changes from 0 → ±1.
    n_trades = int(((pos != pos.shift(1)) & (pos != 0)).sum())
    # Hit rate: fraction of round trips with positive PnL.
    # Approximate: per-segment.
    equity = pnl.cumsum()
    running_max = equity.cummax()
    max_dd = float((equity - running_max).min())
    calmar = (pnl_mean * annualisation) / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0
    # Hit rate: rough via per-bar PnL.
    hit_rate = float((pnl_arr > 0).sum() / max((pnl_arr != 0).sum(), 1))
    return {
        "sharpe": float(sharpe),
        "n_trades": int(n_trades),
        "hit_rate": float(hit_rate),
        "max_drawdown": float(max_dd),
        "sortino": float(sortino),
        "calmar": float(calmar),
    }


__all__ = [
    "adaptive_zscore_signals",
    "bollinger_signals",
    "evaluate_signal",
    "macd_signals",
    "rsi",
    "rsi_signals",
]
