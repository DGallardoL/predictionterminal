"""Pairs-trading signal generator and walk-forward backtester.

Given a cointegrated probability pair (A, B), the spread ε_t = A_t − β·B_t
should mean-revert. We trade the *z-score* of the rolling spread:

*   Enter LONG-spread (buy A, sell βB) when z_t < −entry_z
*   Enter SHORT-spread (sell A, buy βB) when z_t > +entry_z
*   Exit when |z_t| < exit_z
*   Stop out when |z_t| ≥ stop_z (volatility regime change / pair breaks)

Position sign convention:
    +1 = long spread  (long A, short β B)  — bets on reversion *up*
    −1 = short spread (short A, long β B)  — bets on reversion *down*

Daily PnL with a position established at close of t-1 and held into t:
    pnl_t = position_{t-1} · (A_t − A_{t-1} − β · (B_t − B_{t-1}))
          = position_{t-1} · Δspread_t

We treat a "unit" of spread as one prediction-market contract on each leg
(notional ≈ $1 each at settlement). PnL is therefore in *dollars per unit
of spread*, and Sharpe / drawdown are computed on the dollar series.

This is an explicit pedagogical backtest — no transaction-cost model, no
funding cost, no slippage. The :func:`pairs_backtest` returns enough
diagnostics that the analyst can layer those costs in by eye (round-trip
spread cost on Polymarket ≈ 1-3¢; deduct from per-trade PnL).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeRecord:
    """A single round-trip in the pair."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int  # +1 long spread, -1 short spread
    entry_z: float
    exit_z: float
    pnl: float  # dollar PnL on a unit-of-spread position
    holding_days: int
    exit_reason: str  # "mean_reversion" | "stopped_out" | "time_stop" | "end_of_data"


@dataclass(frozen=True)
class BacktestResult:
    """Output of :func:`pairs_backtest`.

    Risk-adjusted return metrics:
    *   ``sharpe``  — μ/σ annualised. Penalises *all* volatility.
    *   ``sortino`` — μ/σ_downside annualised. Penalises only negative-PnL
        volatility (Sortino & Price 1994).
    *   ``calmar``  — annualised return / |max DD|. Pure drawdown-adjusted.

    Tail-risk metrics (computed on the per-bar PnL distribution):
    *   ``var_95``  — 5th percentile of PnL (loss threshold exceeded 5% of bars).
    *   ``cvar_95`` — mean of PnL below the VaR threshold (expected loss
        in the worst 5% of bars). Always ≤ ``var_95``.
    *   ``skew``    — Fisher skewness of PnL.
    *   ``kurtosis`` — excess kurtosis (Gaussian = 0).

    Out-of-sample split (when ``oos_fraction > 0``):
    *   ``sharpe_is`` / ``sharpe_oos`` — Sharpe on the in-sample (first
        ``1 − oos_fraction``) and out-of-sample (last ``oos_fraction``)
        slices respectively. The z-score is computed on a *single* rolling
        window over the whole spread (no look-ahead), so this is a
        legitimate walk-forward test. ``oos_to_is_ratio = sharpe_oos / sharpe_is``
        below ~0.5 flags overfit.
    """

    n_obs: int
    n_trades: int
    positions: pd.Series
    zscores: pd.Series
    spread: pd.Series
    pnl: pd.Series
    equity_curve: pd.Series
    sharpe: float
    sortino: float
    calmar: float
    hit_rate: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    skew: float
    kurtosis: float
    mean_holding_days: float
    sharpe_is: float
    sharpe_oos: float
    oos_to_is_ratio: float
    n_obs_is: int
    n_obs_oos: int
    trades: list[TradeRecord]


def _generate_signals(
    z: pd.Series,
    *,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold_bars: int | None = None,
) -> pd.Series:
    """State-machine signal generator returning a ±1/0 series.

    If ``max_hold_bars`` is set, force-close any position held longer
    than that. Useful as a half-life-aware time stop: don't sit on a
    pair that refuses to revert (regime broke).
    """
    state = 0
    held_bars = 0
    out = np.zeros(len(z), dtype=int)
    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            out[i] = state
            if state != 0:
                held_bars += 1
            continue
        if state == 0:
            if zi <= -entry_z:
                state = 1  # long spread (expect ε to rise back to mean)
                held_bars = 0
            elif zi >= entry_z:
                state = -1  # short spread
                held_bars = 0
        elif state == 1:
            if abs(zi) < exit_z:
                state = 0
                held_bars = 0
            elif zi <= -stop_z:
                state = 0
                held_bars = 0  # stop out (regime change)
            elif max_hold_bars is not None and held_bars >= max_hold_bars:
                state = 0
                held_bars = 0  # time stop
        elif state == -1 and (
            abs(zi) < exit_z
            or zi >= stop_z
            or (max_hold_bars is not None and held_bars >= max_hold_bars)
        ):
            state = 0
            held_bars = 0
        out[i] = state
        if state != 0:
            held_bars += 1
    return pd.Series(out, index=z.index, name="position")


def _max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough max drawdown (returned as a *negative* number)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity - running_max
    return float(dd.min())


def _build_trade_records(
    spread: pd.Series,
    positions: pd.Series,
    zscores: pd.Series,
    *,
    stop_z: float,
    exit_z: float,
    max_hold_bars: int | None = None,
) -> list[TradeRecord]:
    trades: list[TradeRecord] = []
    pos = positions.values
    spr = spread.values
    z = zscores.values
    idx = positions.index

    in_trade_dir = 0
    entry_i = -1
    for i in range(1, len(pos)):
        prev = pos[i - 1]
        curr = pos[i]
        if prev == 0 and curr != 0:
            in_trade_dir = curr
            entry_i = i
        elif prev != 0 and curr == 0:
            # Just closed a trade: position was prev, exited at i.
            ex_i = i
            zi = z[ex_i] if not np.isnan(z[ex_i]) else 0.0
            zi_e = z[entry_i] if entry_i >= 0 and not np.isnan(z[entry_i]) else 0.0
            ds = spr[ex_i] - spr[entry_i] if entry_i >= 0 else 0.0
            pnl = float(in_trade_dir * ds)
            held = ex_i - entry_i if entry_i >= 0 else 0
            if (
                max_hold_bars is not None
                and held >= max_hold_bars
                and abs(zi) >= exit_z
                and abs(zi) < stop_z
            ):
                reason = "time_stop"
            elif abs(zi) >= stop_z:
                reason = "stopped_out"
            else:
                reason = "mean_reversion"
            trades.append(
                TradeRecord(
                    entry_date=idx[entry_i] if entry_i >= 0 else idx[0],
                    exit_date=idx[ex_i],
                    direction=int(in_trade_dir),
                    entry_z=float(zi_e),
                    exit_z=float(zi),
                    pnl=pnl,
                    holding_days=int(ex_i - entry_i),
                    exit_reason=reason,
                )
            )
            in_trade_dir = 0
            entry_i = -1

    # Open trade at end of data → mark as end_of_data
    if in_trade_dir != 0 and entry_i >= 0:
        ex_i = len(pos) - 1
        ds = spr[ex_i] - spr[entry_i]
        pnl = float(in_trade_dir * ds)
        zi_e = z[entry_i] if not np.isnan(z[entry_i]) else 0.0
        zi = z[ex_i] if not np.isnan(z[ex_i]) else 0.0
        trades.append(
            TradeRecord(
                entry_date=idx[entry_i],
                exit_date=idx[ex_i],
                direction=int(in_trade_dir),
                entry_z=float(zi_e),
                exit_z=float(zi),
                pnl=pnl,
                holding_days=int(ex_i - entry_i),
                exit_reason="end_of_data",
            )
        )
    return trades


def pairs_backtest(
    spread: pd.Series,
    *,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    annualisation_factor: float = 252.0,
    oos_fraction: float = 0.30,
    max_hold_bars: int | None = None,
) -> BacktestResult:
    """Walk-forward backtest of a z-score-based pairs trade on a spread.

    Args:
        spread: per-date spread series ε_t = A_t − β·B_t.
        window: rolling window for z-score normalisation.
        entry_z: |z| threshold to open a position.
        exit_z: |z| threshold to flatten an open position.
        stop_z: |z| threshold beyond which we stop out.
        annualisation_factor: bars-per-year for the Sharpe calculation
            (252 daily-bar default).

    Returns:
        :class:`BacktestResult`.
    """
    if entry_z <= exit_z:
        raise ValueError("entry_z must be greater than exit_z")
    if stop_z <= entry_z:
        raise ValueError("stop_z must be greater than entry_z")
    if not 0.0 <= oos_fraction < 1.0:
        raise ValueError(f"oos_fraction must be in [0, 1), got {oos_fraction}")

    spread = spread.dropna().sort_index()
    if len(spread) < window + 5:
        raise ValueError(f"need at least window+5={window + 5} bars, got {len(spread)}")

    # Rolling z-score (matches `cointegration.spread_zscore` semantics).
    mu = spread.rolling(window=window, min_periods=max(5, window // 2)).mean()
    sd = spread.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
    z = ((spread - mu) / sd).rename("zscore")

    pos = _generate_signals(
        z, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z, max_hold_bars=max_hold_bars
    )

    # Daily PnL: position established at end of t-1, payoff over [t-1, t].
    dspread = spread.diff().fillna(0.0)
    pnl = (pos.shift(1).fillna(0).astype(float) * dspread).rename("pnl")
    equity = pnl.cumsum().rename("equity")

    pnl_arr = pnl.to_numpy()
    pnl_std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    pnl_mean = float(pnl.mean())
    sqrt_ann = float(np.sqrt(annualisation_factor))

    # Sharpe (annualised) from per-bar PnL.
    sharpe = (pnl_mean / pnl_std) * sqrt_ann if pnl_std > 0 else 0.0
    # Sortino: downside std (σ_d) on the negative-PnL leg only.
    neg = pnl_arr[pnl_arr < 0]
    downside_std = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
    sortino = (pnl_mean / downside_std) * sqrt_ann if downside_std > 0 else 0.0
    # VaR / CVaR at 95%
    if len(pnl_arr) >= 5:
        var_95 = float(np.percentile(pnl_arr, 5))
        tail = pnl_arr[pnl_arr <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) else var_95
    else:
        var_95, cvar_95 = 0.0, 0.0
    # Skew / kurtosis (Fisher / excess); skip when sample is degenerate.
    if pnl_std > 0 and len(pnl_arr) >= 4:
        z_scores = (pnl_arr - pnl_mean) / pnl_std
        skew = float(np.mean(z_scores**3))
        kurt = float(np.mean(z_scores**4) - 3.0)
    else:
        skew, kurt = 0.0, 0.0

    # Out-of-sample split: use the *same* rolling-window z-score over the
    # full spread (no look-ahead violation, since the rolling window only
    # uses past data) and compute Sharpe on the IS / OOS partitions.
    n_total = len(pnl_arr)
    n_oos = int(round(n_total * oos_fraction))
    n_is = n_total - n_oos
    if n_is >= 2 and n_oos >= 2 and oos_fraction > 0.0:
        is_arr = pnl_arr[:n_is]
        oos_arr = pnl_arr[n_is:]
        is_std = float(np.std(is_arr, ddof=1)) if len(is_arr) > 1 else 0.0
        oos_std = float(np.std(oos_arr, ddof=1)) if len(oos_arr) > 1 else 0.0
        sharpe_is = (float(np.mean(is_arr)) / is_std) * sqrt_ann if is_std > 0 else 0.0
        sharpe_oos = (float(np.mean(oos_arr)) / oos_std) * sqrt_ann if oos_std > 0 else 0.0
    else:
        sharpe_is, sharpe_oos = sharpe, 0.0
        n_is, n_oos = n_total, 0
    if abs(sharpe_is) > 1e-9:
        oos_to_is = sharpe_oos / sharpe_is
    else:
        oos_to_is = 0.0

    trades = _build_trade_records(
        spread,
        pos,
        z,
        stop_z=stop_z,
        exit_z=exit_z,
        max_hold_bars=max_hold_bars,
    )
    n_trades = len(trades)
    hit = sum(t.pnl > 0 for t in trades)
    hit_rate = hit / n_trades if n_trades else 0.0
    mean_hold = float(np.mean([t.holding_days for t in trades])) if trades else 0.0
    max_dd = _max_drawdown(equity)
    # Calmar: annualised return / |max DD|. Equity is cumulative PnL on a
    # unit-spread position; "annualised return" ≈ pnl_mean · annualisation.
    if abs(max_dd) > 1e-12:
        calmar = (pnl_mean * annualisation_factor) / abs(max_dd)
    else:
        calmar = 0.0

    return BacktestResult(
        n_obs=len(spread),
        n_trades=n_trades,
        positions=pos,
        zscores=z,
        spread=spread,
        pnl=pnl,
        equity_curve=equity,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        hit_rate=hit_rate,
        max_drawdown=max_dd,
        var_95=var_95,
        cvar_95=cvar_95,
        skew=skew,
        kurtosis=kurt,
        mean_holding_days=mean_hold,
        sharpe_is=sharpe_is,
        sharpe_oos=sharpe_oos,
        oos_to_is_ratio=oos_to_is,
        n_obs_is=n_is,
        n_obs_oos=n_oos,
        trades=trades,
    )


__all__ = ["BacktestResult", "TradeRecord", "pairs_backtest"]
