"""Minimal strategy registry for the stress-test harness.

CLAUDE.md's anti-alpha rule requires every "wow" backtest be cross-validated
against >=4 disjoint quarters. The stress-test CLI in
``scripts/stress_test.py`` needs a uniform way to load a strategy by name and
ask it for a PnL series — irrespective of whether the strategy lives in
:mod:`pfm.strategies` (classical diagnostics), in one of the
``strategies_*_router`` modules, or in a brand-new file under
``pfm/strategies/``.

To avoid leaking a heavy import graph into the script (and to keep this
unit-testable without spinning up FastAPI), strategies register themselves
via a small protocol:

* ``signal(prices: pd.DataFrame) -> pd.Series`` — a +1/-1/0 position signal.
* ``position(signal: pd.Series) -> pd.Series`` — sized position (default
  identity for the signal).
* ``pnl(prices, signal) -> pd.Series`` — daily PnL series indexed by date.

The registry only stores entries; it does NOT enumerate the historical
universe at import time. Tests register synthetic strategies inline so the
harness can be exercised hermetically.

If callers want a "real" strategy the canonical path is to add it to
``pfm/strategies/<name>.py`` and import-and-register from there (see
CLAUDE.md "Add a new alpha strategy"). For now the registry ships with two
demo entries — ``buy-and-hold`` and ``zero`` — which are useful both as
sanity checks and as fallbacks the CI can run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Protocol — kept as a dataclass rather than a typing.Protocol so the
# registry can store *instances* with mutable state (e.g. parameters).
# ---------------------------------------------------------------------------


SignalFn = Callable[[pd.DataFrame], pd.Series]
PositionFn = Callable[[pd.Series], pd.Series]
PnLFn = Callable[[pd.DataFrame, pd.Series], pd.Series]


@dataclass(frozen=True)
class Strategy:
    """A registered strategy with the three protocol callables.

    Strategies operate on a daily-indexed DataFrame with at least a ``close``
    column. The ``signal`` callable returns a position direction
    (typically -1/0/+1); ``position`` converts that to a sized position; and
    ``pnl`` returns the realised daily PnL series.

    The default ``position`` is the identity (no sizing); the default
    ``pnl`` is ``position.shift(1) * log_return(close)`` — the textbook
    one-day-lagged signal-times-log-return PnL.
    """

    name: str
    signal: SignalFn
    position: PositionFn | None = None
    pnl: PnLFn | None = None

    def compute_pnl(self, prices: pd.DataFrame) -> pd.Series:
        """Run the full signal -> position -> pnl pipeline."""
        sig = self.signal(prices)
        pos = self.position(sig) if self.position is not None else sig
        if self.pnl is not None:
            return self.pnl(prices, pos)
        return _default_pnl(prices, pos)


def _default_pnl(prices: pd.DataFrame, position: pd.Series) -> pd.Series:
    """Lagged-position * log-return PnL.

    Following CLAUDE.md: "Log returns, not simple returns.
    r_t = log(P_t / P_{t-1})".
    """
    if "close" not in prices.columns:
        msg = "prices DataFrame must contain a 'close' column"
        raise ValueError(msg)
    log_ret = np.log(prices["close"]).diff()
    aligned = position.reindex(log_ret.index).shift(1)
    return (aligned * log_ret).fillna(0.0).rename("pnl")


# ---------------------------------------------------------------------------
# Registry — module-global dict, simple by design.
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, Strategy] = {}


def register(strategy: Strategy) -> Strategy:
    """Register a strategy; idempotent on the same name+instance."""
    _REGISTRY[strategy.name] = strategy
    return strategy


def unregister(name: str) -> None:
    """Remove a strategy from the registry (used in tests for hygiene)."""
    _REGISTRY.pop(name, None)


def get(name: str) -> Strategy:
    """Look up a strategy by name. Raises ``KeyError`` if missing."""
    if name not in _REGISTRY:
        registered = ", ".join(sorted(_REGISTRY)) or "(none)"
        msg = f"Strategy {name!r} not registered. Known: {registered}"
        raise KeyError(msg)
    return _REGISTRY[name]


def names() -> list[str]:
    """Sorted list of registered strategy names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Default registrations — sanity-check strategies the CLI can always run.
# ---------------------------------------------------------------------------


def _buy_and_hold_signal(prices: pd.DataFrame) -> pd.Series:
    """Always-long, ignoring price action."""
    return pd.Series(1.0, index=prices.index, name="signal")


def _zero_signal(prices: pd.DataFrame) -> pd.Series:
    """Always-flat. Useful as a null-distribution sanity check."""
    return pd.Series(0.0, index=prices.index, name="signal")


register(Strategy(name="buy-and-hold", signal=_buy_and_hold_signal))
register(Strategy(name="zero", signal=_zero_signal))


__all__ = [
    "Strategy",
    "get",
    "names",
    "register",
    "unregister",
]
