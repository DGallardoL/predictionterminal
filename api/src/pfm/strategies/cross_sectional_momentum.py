"""Cross-sectional momentum across binary prediction-market factors (W12-25).

Strategy summary
----------------
Each rebalance date ``t``:

1. For every factor in the cross-section, compute the trailing 14-day return
   ``r_{i,t} = log(P_{i,t} / P_{i,t-14})``. (Per CLAUDE.md: "Log returns, not
   simple returns.")
2. Rank factors by that trailing return.
3. **Long the top decile** (highest 14d returns) and **short the bottom
   decile** (lowest 14d returns), equal-weighted within each leg.
4. Net dollar-neutral: long-leg weights sum to +1, short-leg to −1; the
   strategy's gross exposure is 2 and net 0.
5. PnL on date ``t+1`` is the cross-sectional dot-product of the position
   vector with that day's realised log returns.

Anti-alpha gating
-----------------
Cross-sectional momentum is a heavily-studied factor in cash equities and
crypto. The CLAUDE.md memory ``Wave-5 stress tests killed 6 of 8 A_GOLD
claims`` is a stern reminder that what looks like a structural anomaly is
often regime-driven. Per the project's anti-alpha rule, every "wow" backtest
must clear 4 disjoint quarters of robustness before promotion past
``B_VALIDATED``. This module therefore ships with::

    SHOULD_DEPLOY = False
    TIER         = "B_VALIDATED"

and **does NOT auto-add itself to** ``web/data/alpha_strategies.json``.
Promotion requires (a) ``api/scripts/stress_test.py --strategy
cross-sectional-momentum --quarters 4`` returning PASS over four disjoint
quarters with Sharpe ≥ 0.5 in every quarter and no sign-flip vs full sample,
and (b) a human review flipping ``SHOULD_DEPLOY`` to ``True``.

Vectorisation contract
----------------------
The signal accepts a *factor-returns panel*: a ``pd.DataFrame`` with a
:class:`pd.DatetimeIndex` (rebalance dates, daily) and one column per
factor — each cell is the **trailing 14-day log-return** of that factor on
that date. The signal returns a long-form ``pd.Series`` indexed by a
``(date, factor)`` :class:`pd.MultiIndex` carrying the per-factor decile
weight (+1/N_long, 0, or −1/N_short).

The ``position(...)`` step rescales those weights by a Kelly cap (default
0.10 gross-exposure-equivalent) and the ``pnl(...)`` step takes the
dot-product against a realised-returns panel of identical shape.

References
----------
* Jegadeesh & Titman 1993, "Returns to buying winners and selling losers"
* Asness, Moskowitz, Pedersen 2013, "Value and momentum everywhere"
* CLAUDE.md anti-alpha rule (project root)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Deployment guard — per CLAUDE.md anti-alpha rule, default OFF.
# ---------------------------------------------------------------------------

#: Hard guard. Flip to ``True`` only after the 4-quarter robustness gate.
SHOULD_DEPLOY: bool = False

#: CLAUDE.md tier ceiling. Stays ``B_VALIDATED`` until 4+ disjoint quarters
#: of *live paper trading* confirm the alpha.
TIER: str = "B_VALIDATED"

#: Canonical strategy name (registry key + stress-test --strategy flag).
STRATEGY_NAME: str = "cross-sectional-momentum"

#: Trailing-return lookback (days). Hard-coded to 14 per the W12-25 spec.
LOOKBACK_DAYS: int = 14


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decile_threshold(n_factors: int, fraction: float = 0.10) -> int:
    """Return the integer count of factors per long (or short) leg.

    Implementation note: for tiny universes (``n < 10``) the "decile" reduces
    to a single factor on each end. We always pick at least ``1`` factor per
    leg whenever ``n_factors >= 2``; if ``n_factors < 2`` the strategy
    abstains (no long/short pair definable).
    """
    if n_factors < 2:
        return 0
    return max(1, int(math.floor(n_factors * fraction)))


def trailing_log_return(prices: pd.DataFrame, *, window: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Compute trailing ``window``-day log returns of each column.

    ``prices`` is a date-indexed frame; each column is a factor's price level.
    Returns a frame of identical shape, with the first ``window`` rows NaN.
    Per CLAUDE.md: log returns, not simple.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        # Be lenient — many callers use a sequential range index.
        pass
    log_prices = np.log(prices.astype(float))
    return log_prices.diff(periods=window)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


@dataclass
class CrossSectionalMomentum:
    """Long-top-decile / short-bottom-decile momentum across binary factors.

    Parameters
    ----------
    decile_fraction:
        Fraction of the cross-section on each leg (default 0.10 = decile).
    kelly_cap:
        Gross-exposure cap applied per leg in :meth:`position`. The raw decile
        weights are 1/N per leg (so the leg sums to ±1); the cap rescales them
        such that the **absolute value of each individual position never
        exceeds ``kelly_cap``**. For a typical universe of 50 factors and
        the default 0.10 fraction (5 names per leg, 0.20 raw weight each) the
        default ``kelly_cap=0.10`` ratchets each name down to 0.10.
    """

    name: str = STRATEGY_NAME
    tier: str = TIER
    decile_fraction: float = 0.10
    kelly_cap: float = 0.10

    def __post_init__(self) -> None:
        if not (0.0 < self.decile_fraction <= 0.5):
            raise ValueError(f"decile_fraction must be in (0, 0.5]; got {self.decile_fraction}")
        if not (0.0 < self.kelly_cap <= 1.0):
            raise ValueError(f"kelly_cap must be in (0, 1]; got {self.kelly_cap}")

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def signal(self, factor_returns_panel: pd.DataFrame) -> pd.Series:
        """Compute long-top / short-bottom decile signal.

        Parameters
        ----------
        factor_returns_panel:
            DataFrame whose index is the rebalance date and whose columns are
            factor identifiers. Each cell is the trailing 14-day **log
            return** of that factor on that date. Rows with all-NaN are
            skipped. NaN cells are excluded from ranking for that row.

        Returns
        -------
        pd.Series
            MultiIndex ``(date, factor)`` Series with values in
            ``{+1/N_long, 0.0, -1/N_short}``. Sums to ~0 per date
            (dollar-neutral) and absolute sums to ~2 per date (gross 2x).
        """
        if not isinstance(factor_returns_panel, pd.DataFrame):
            raise TypeError(
                "factor_returns_panel must be a pd.DataFrame "
                f"(got {type(factor_returns_panel).__name__})"
            )
        if factor_returns_panel.empty:
            return pd.Series(
                dtype=float,
                index=pd.MultiIndex.from_tuples([], names=["date", "factor"]),
                name="signal",
            )

        rows: list[pd.Series] = []
        for date, row in factor_returns_panel.iterrows():
            row_clean = row.dropna().astype(float)
            n = len(row_clean)
            k = _decile_threshold(n, fraction=self.decile_fraction)
            weights = pd.Series(0.0, index=row.index, dtype=float)
            if k == 0:
                weights.name = date
                rows.append(weights)
                continue
            # `sort_values` is stable — ties get the order of insertion.
            ranked = row_clean.sort_values(kind="mergesort")
            shorts = ranked.index[:k]
            longs = ranked.index[-k:]
            weights.loc[longs] = 1.0 / k
            weights.loc[shorts] = -1.0 / k
            weights.name = date
            rows.append(weights)

        wide = pd.DataFrame(rows)
        wide.index.name = "date"
        long = wide.stack(future_stack=True)
        long.index.names = ["date", "factor"]
        long.name = "signal"
        return long.astype(float)

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def position(
        self,
        signal: pd.Series,
        *,
        kelly_cap: float | None = None,
    ) -> pd.Series:
        """Kelly-capped sized position from a signal Series.

        Each per-name weight is clipped to ``[-kelly_cap, +kelly_cap]``. The
        sign and shape of the cross-section are preserved (no re-normalisation
        after the cap — this means a heavy clip will lower gross exposure
        symmetrically, which is the desired risk-control behaviour).
        """
        kcap = self.kelly_cap if kelly_cap is None else float(kelly_cap)
        if not (0.0 < kcap <= 1.0):
            raise ValueError(f"kelly_cap must be in (0, 1]; got {kcap}")
        if not isinstance(signal, pd.Series):
            raise TypeError(f"signal must be a pd.Series; got {type(signal).__name__}")
        if signal.empty:
            return signal.astype(float).rename("position")
        return signal.astype(float).clip(lower=-kcap, upper=kcap).rename("position")

    # ------------------------------------------------------------------
    # PnL
    # ------------------------------------------------------------------

    def pnl(
        self,
        position: pd.Series,
        realized: pd.DataFrame | pd.Series,
    ) -> pd.Series:
        """Realised PnL per rebalance date.

        Parameters
        ----------
        position:
            MultiIndex ``(date, factor)`` Series produced by :meth:`position`.
        realized:
            Either a wide ``DataFrame`` (date index × factor columns) of the
            realised log returns *for the holding period* (typically the
            single rebalance day, lagged by one day to avoid look-ahead), or
            a matching MultiIndex Series in long form.

        Returns
        -------
        pd.Series
            Daily PnL indexed by date. Missing factor returns are treated as
            zero (the position contributes nothing on that date).
        """
        if not isinstance(position, pd.Series):
            raise TypeError("position must be a pd.Series")
        if position.empty:
            return pd.Series(dtype=float, name="pnl")
        if not isinstance(position.index, pd.MultiIndex):
            raise ValueError("position must have a (date, factor) MultiIndex")

        if isinstance(realized, pd.DataFrame):
            realized_long = realized.stack(future_stack=True).astype(float)
            realized_long.index.names = ["date", "factor"]
        elif isinstance(realized, pd.Series):
            if not isinstance(realized.index, pd.MultiIndex):
                raise ValueError("realized Series must have a (date, factor) MultiIndex")
            realized_long = realized.astype(float)
            realized_long.index.names = ["date", "factor"]
        else:
            raise TypeError(
                f"realized must be pd.DataFrame or pd.Series; got {type(realized).__name__}"
            )

        # Align positions with realised returns; missing cells -> 0 PnL.
        aligned = realized_long.reindex(position.index).fillna(0.0)
        prod = position.astype(float) * aligned
        pnl_per_date = prod.groupby(level="date").sum()
        return pnl_per_date.astype(float).rename("pnl")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def compute_pnl(
        self,
        factor_returns_panel: pd.DataFrame,
        realized_returns_panel: pd.DataFrame,
    ) -> pd.Series:
        """Full signal -> position -> pnl pipeline (no shift applied here).

        Callers are responsible for ensuring ``realized_returns_panel`` is the
        **next-period** return (lag by 1 row vs the panel used to build the
        signal) so the strategy never peeks at the same-day realisation. This
        keeps the convenience method honest about look-ahead.
        """
        sig = self.signal(factor_returns_panel)
        pos = self.position(sig)
        return self.pnl(pos, realized_returns_panel)


# ---------------------------------------------------------------------------
# Registry hook — lazy, gated on SHOULD_DEPLOY.
# ---------------------------------------------------------------------------


def _signal_adapter(prices: pd.DataFrame) -> pd.Series:
    """Adapter so :mod:`pfm.strategies_registry` can consume this strategy.

    The registry's signal contract is ``signal(prices) -> pd.Series`` where
    ``prices`` is a daily-indexed frame with at least a ``close`` column.
    Cross-sectional momentum needs a *panel* of factors; if a single-asset
    frame is passed we degrade gracefully to a binary in/out trend signal
    based on the trailing 14-day log return sign.
    """
    alpha = CrossSectionalMomentum()
    if "close" in prices.columns and prices.shape[1] == 1:
        # Single-series fallback — sign of trailing 14d log return.
        log_ret = np.log(prices["close"].astype(float)).diff(LOOKBACK_DAYS)
        sig_wide = np.sign(log_ret).fillna(0.0).astype(float)
        return sig_wide.rename("signal")
    # Treat the frame columns as factors and the rows as dates.
    panel = trailing_log_return(prices, window=LOOKBACK_DAYS)
    long = alpha.signal(panel)
    # Reduce to per-date net exposure for the registry's 1D contract.
    return long.groupby(level="date").sum().rename("signal")


def _position_adapter(signal: pd.Series) -> pd.Series:
    alpha = CrossSectionalMomentum()
    return alpha.position(signal)


def register_if_ready(*, force: bool = False) -> Any:
    """Register the strategy in :mod:`pfm.strategies_registry` iff ready.

    Conditions:
        * :data:`SHOULD_DEPLOY` is ``True`` (or ``force=True`` for tests)

    Returns the registered :class:`Strategy` instance, else ``None``.
    """
    if not (force or SHOULD_DEPLOY):
        return None
    from pfm.strategies_registry import Strategy, register

    strat = Strategy(
        name=STRATEGY_NAME,
        signal=_signal_adapter,
        position=_position_adapter,
        pnl=None,
    )
    register(strat)
    return strat


# ---------------------------------------------------------------------------
# Catalog helper (NOT auto-invoked — see CLAUDE.md anti-alpha rule).
# ---------------------------------------------------------------------------


def alpha_catalog_entry(
    *,
    robustness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the dict to be APPENDED to ``web/data/alpha_strategies.json``.

    **Never** invoked automatically. Promotion requires a human review after
    the 4-quarter robustness gate is cleared.
    """
    return {
        "pair_id": STRATEGY_NAME,
        "tier": TIER,
        "label": "Cross-sectional momentum (binary prediction-market factors)",
        "deploy_params": {
            "lookback_days": LOOKBACK_DAYS,
            "decile_fraction": 0.10,
            "kelly_cap": 0.10,
        },
        "theory_ref": (
            "Jegadeesh & Titman 1993; Asness, Moskowitz, Pedersen 2013 "
            "('Value and momentum everywhere')."
        ),
        "robustness": dict(robustness or {}),
        "should_deploy_at_publish_time": SHOULD_DEPLOY,
        "anti_alpha_rule": (
            "CLAUDE.md anti-alpha rule: ceiling B_VALIDATED until 4 disjoint "
            "quarters of LIVE paper trading confirm Sharpe >= 0.5 with no "
            "sign-flip vs full sample. Auto-registration is OFF."
        ),
    }


__all__ = [
    "LOOKBACK_DAYS",
    "SHOULD_DEPLOY",
    "STRATEGY_NAME",
    "TIER",
    "CrossSectionalMomentum",
    "alpha_catalog_entry",
    "register_if_ready",
    "trailing_log_return",
]
