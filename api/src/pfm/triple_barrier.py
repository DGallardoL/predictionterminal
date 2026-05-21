"""Triple Barrier Method (López de Prado 2018, ch. 3).

The TBM is the modern industry-standard exit logic for trading signals.
Given a primary entry signal (e.g., z-score crossing −2σ), it sets three
*vol-scaled* barriers and exits the trade at the FIRST one touched:

1. **Upper barrier** (profit target): +pt · σ_local
2. **Lower barrier** (stop loss):     −sl · σ_local
3. **Vertical barrier** (time horizon): T bars after entry

σ_local is the rolling volatility estimate at entry — so barriers ADAPT
to the local vol regime. This is strictly better than fixed-z exits
because it sizes the profit target to the spread's own current volatility.

After labeling each trade with first-touched barrier (+1 / −1 / 0), the
realised PnL is captured. The advantage over fixed exit_z is that we
don't exit on a noise wiggle through the mean — we hold until a real
profit target or the time horizon expires.

References:
    López de Prado, M. (2018). *Advances in Financial Machine Learning*
        ch. 3 ("Labeling"), §3.3 ("The Triple-Barrier Method").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierTrade:
    entry_index: int
    exit_index: int
    direction: int  # +1 long spread, −1 short spread
    entry_value: float
    exit_value: float
    pnl: float
    label: int  # +1 = hit profit, −1 = stopped, 0 = time-stopped
    holding_bars: int


@dataclass(frozen=True)
class TripleBarrierResult:
    n_trades: int
    trades: list[TripleBarrierTrade]
    pnl_series: pd.Series
    total_pnl: float
    sharpe: float
    n_profit_hits: int
    n_stop_hits: int
    n_time_hits: int


def _detect_entries(
    spread: pd.Series,
    *,
    window: int = 20,
    entry_z: float = 2.0,
) -> list[tuple[int, int]]:
    """Identify entry indices and direction.

    Returns list of (idx, direction) where direction = +1 for long-spread
    (z below −entry_z) or −1 for short-spread (z above +entry_z). Skips
    bars where we're already in a trade (avoids overlapping entries).
    """
    mu = spread.rolling(window=window, min_periods=max(5, window // 2)).mean()
    sd = spread.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
    z = (spread - mu) / sd
    entries: list[tuple[int, int]] = []
    z_arr = z.values
    for i, zi in enumerate(z_arr):
        if pd.isna(zi):
            continue
        if zi <= -entry_z:
            entries.append((i, +1))
        elif zi >= entry_z:
            entries.append((i, -1))
    return entries


def triple_barrier_backtest(
    spread: pd.Series,
    *,
    window: int = 20,
    entry_z: float = 2.0,
    profit_target_sigma: float = 2.0,
    stop_loss_sigma: float = 4.0,
    time_horizon_bars: int = 10,
    annualisation: float = 252.0,
) -> TripleBarrierResult:
    """Run the Triple Barrier backtest.

    Args:
        spread: per-bar spread series.
        window: rolling window for σ_local estimation.
        entry_z: |z| threshold for entry (same as the z-score state machine).
        profit_target_sigma: profit target = +pt · σ_local.
        stop_loss_sigma: stop loss = −sl · σ_local.
        time_horizon_bars: vertical barrier — close after T bars regardless.
        annualisation: bars per year for Sharpe calc.

    Returns:
        :class:`TripleBarrierResult`.
    """
    if entry_z <= 0:
        raise ValueError(f"entry_z must be > 0, got {entry_z}")
    if time_horizon_bars < 1:
        raise ValueError(f"time_horizon_bars must be >= 1, got {time_horizon_bars}")

    spread = spread.dropna().sort_index()
    n = len(spread)
    if n < window + time_horizon_bars + 5:
        raise ValueError(
            f"need ≥ window+horizon+5 = {window + time_horizon_bars + 5} bars, got {n}"
        )

    sd = spread.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
    sd_arr = sd.values
    sp_arr = spread.values

    entries = _detect_entries(spread, window=window, entry_z=entry_z)

    # Build per-trade outputs and aggregate per-bar PnL.
    pnl_per_bar = np.zeros(n)
    trades: list[TripleBarrierTrade] = []
    in_trade_until = -1

    for entry_i, direction in entries:
        if entry_i <= in_trade_until:
            continue  # skip overlapping entry
        if pd.isna(sd_arr[entry_i]) or sd_arr[entry_i] <= 0:
            continue
        sigma = float(sd_arr[entry_i])
        entry_val = float(sp_arr[entry_i])
        # Set barriers in spread-units. For LONG spread (direction=+1):
        # profit when spread RISES by pt·σ, stop when spread FALLS by sl·σ.
        # Mirror for SHORT.
        if direction == 1:
            up = entry_val + profit_target_sigma * sigma
            dn = entry_val - stop_loss_sigma * sigma
        else:
            up = entry_val + stop_loss_sigma * sigma
            dn = entry_val - profit_target_sigma * sigma
        end_i = min(entry_i + time_horizon_bars, n - 1)
        # Walk forward to first barrier hit.
        exit_i = end_i
        label = 0
        for k in range(entry_i + 1, end_i + 1):
            sk = sp_arr[k]
            if direction == 1:
                if sk >= up:
                    exit_i = k
                    label = +1
                    break
                if sk <= dn:
                    exit_i = k
                    label = -1
                    break
            else:
                if sk <= dn:
                    exit_i = k
                    label = +1
                    break
                if sk >= up:
                    exit_i = k
                    label = -1
                    break
        exit_val = float(sp_arr[exit_i])
        pnl = direction * (exit_val - entry_val)
        # Distribute PnL into the bar of exit (cleanest accounting; trade
        # spans entry_i+1 .. exit_i, but we record a single P&L impact at exit).
        pnl_per_bar[exit_i] += pnl
        trades.append(
            TripleBarrierTrade(
                entry_index=entry_i,
                exit_index=exit_i,
                direction=direction,
                entry_value=entry_val,
                exit_value=exit_val,
                pnl=pnl,
                label=label,
                holding_bars=exit_i - entry_i,
            )
        )
        in_trade_until = exit_i

    pnl_series = pd.Series(pnl_per_bar, index=spread.index, name="pnl")
    total_pnl = float(pnl_per_bar.sum())
    sd_pnl = float(pnl_series.std(ddof=1)) if n > 1 else 0.0
    sharpe = (
        (float(pnl_series.mean()) / sd_pnl) * float(np.sqrt(annualisation)) if sd_pnl > 0 else 0.0
    )
    n_profit = sum(1 for t in trades if t.label == +1)
    n_stop = sum(1 for t in trades if t.label == -1)
    n_time = sum(1 for t in trades if t.label == 0)

    return TripleBarrierResult(
        n_trades=len(trades),
        trades=trades,
        pnl_series=pnl_series,
        total_pnl=total_pnl,
        sharpe=float(sharpe),
        n_profit_hits=n_profit,
        n_stop_hits=n_stop,
        n_time_hits=n_time,
    )


__all__ = ["TripleBarrierResult", "TripleBarrierTrade", "triple_barrier_backtest"]
