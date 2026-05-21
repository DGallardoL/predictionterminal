"""Binary-pricing mispricing alpha (Track-L T84).

This module wraps a *fair-price* model for binary prediction markets (one of
the four candidate models built in T81 — risk-neutral logit, BS-digital,
Brownian bridge, beta-binomial) into a registrable
:class:`pfm.strategies_registry.Strategy`.

CLAUDE.md "Anti-alphas (DO NOT redeploy)" rule explicitly forbids shipping
a strategy that hasn't passed a 4-quarter robustness gate. T83 — the
empirical evaluation that picks a winning pricer and writes the verdict to
``docs/binary-pricing-results.md`` — has not yet landed at the time this
module was written, and T81's ``pfm.pricing.binary_models`` module body is
also not yet importable.

Therefore this module ships with::

    SHOULD_DEPLOY = False

``register_if_ready()`` consults this flag (and the existence of
``docs/binary-pricing-results.md``) before adding the strategy to
:mod:`pfm.strategies_registry`. The flag must be flipped to ``True`` by a
human follow-up review after:

1.  T81 lands and ``pfm.pricing.binary_models`` is importable.
2.  T83 writes a "winning model" verdict to
    ``docs/binary-pricing-results.md`` that satisfies all three deploy
    criteria (Brier vs market ≥10% better, calibration RMSE < 0.05,
    positive net PnL after 1% costs).
3.  This strategy passes ``api/scripts/stress_test.py
    --strategy binary-pricing-mispricing --quarters 4 --start 2024-01`` —
    *every* quarter Sharpe ≥ 0.5 and no sign-flip vs full sample.

Per the anti-alpha rule, the absolute tier ceiling for this strategy is
``B_VALIDATED``. Promotion to ``A_GOLD`` requires 4+ quarters of *positive*
confirmation in *live* paper trading, not just backtest survivorship.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Deployment guard — see module docstring.
# ---------------------------------------------------------------------------

#: Hard guard. Stays ``False`` until a human follow-up review confirms
#: (a) T81 pricer module is importable, (b) T83 wrote a winning-model
#: verdict, (c) ``api/scripts/stress_test.py`` returns PASS over 4 quarters.
SHOULD_DEPLOY: bool = False

#: CLAUDE.md ceiling. **NEVER** promote past this without 4+ quarters of
#: live paper-trade confirmation (not just backtest).
TIER: str = "B_VALIDATED"

#: Canonical strategy name (used for registry key, stress-test --strategy
#: flag, and the ``pair_id`` field in ``web/data/alpha_strategies.json``).
STRATEGY_NAME: str = "binary-pricing-mispricing"


# ---------------------------------------------------------------------------
# Lightweight Pricer protocol — kept independent of T81 so this file can be
# unit-tested in isolation (T81 is a separate active claim).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketState:
    """Snapshot of a binary prediction-market contract on a single date.

    Attributes
    ----------
    market_price:
        Mid quote of the YES contract on this date, in [0, 1].
    time_to_resolution_days:
        Calendar days until the market resolves. ``0`` means "resolves
        today"; positive values mean the future.
    features:
        Optional feature vector consumed by the pricer (e.g. news evidence,
        underlying spot ratio, drift estimate). Pricer-defined.
    """

    market_price: float
    time_to_resolution_days: float
    features: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Pricer(Protocol):
    """Minimal interface for any binary pricer used by this strategy.

    A pricer maps a :class:`MarketState` to a *fair* probability in [0, 1]
    representing the model's view of the true resolution probability.
    """

    def fair_price(self, state: MarketState) -> float:  # pragma: no cover - protocol
        """Return the model's fair YES probability in [0, 1]."""
        ...


# ---------------------------------------------------------------------------
# BinaryPricingAlpha — wraps a Pricer into the Strategy-protocol pipeline.
# ---------------------------------------------------------------------------


def _clip_prob(p: float, eps: float = 0.01) -> float:
    """Clip a probability into ``[eps, 1-eps]`` (CLAUDE.md default ε=0.01)."""
    if not math.isfinite(p):
        return 0.5
    return float(min(max(p, eps), 1.0 - eps))


def _zscore(values: pd.Series, window: int = 20) -> pd.Series:
    """Trailing-window z-score of ``values``; first ``window-1`` rows are NaN."""
    mu = values.rolling(window=window, min_periods=window).mean()
    sd = values.rolling(window=window, min_periods=window).std(ddof=1)
    z = (values - mu) / sd.replace(0.0, np.nan)
    return z


class BinaryPricingAlpha:
    """Mispricing-aware strategy on binary prediction-market contracts.

    Parameters
    ----------
    pricer:
        Any object satisfying the :class:`Pricer` protocol.
    kelly_cap:
        Hard cap on position size (in units of "fraction of bankroll"),
        applied AFTER computing Kelly. CLAUDE.md recommends ≤0.25 for
        speculative B_VALIDATED strategies; default ``0.20``.
    z_threshold:
        Minimum absolute signal z-score to take a position. Below this we
        sit on our hands. Default ``1.0`` (≈68% confidence interval).
    z_window:
        Rolling window for the trailing z-score normalisation. Default 20
        trading days, matching ``rule_window`` in the wider alpha catalog.
    clip_eps:
        Probability clipping epsilon (CLAUDE.md default 0.01).

    Notes
    -----
    * ``name`` and ``tier`` are class-level constants for ergonomics; do
      not mutate them at runtime.
    * The signal/position/pnl callables are designed to be invoked one-row
      at a time AND vectorised across a DataFrame; see the module tests.
    """

    name: str = STRATEGY_NAME
    tier: str = TIER

    def __init__(
        self,
        pricer: Pricer,
        *,
        kelly_cap: float = 0.20,
        z_threshold: float = 1.0,
        z_window: int = 20,
        clip_eps: float = 0.01,
    ) -> None:
        if not isinstance(pricer, Pricer):
            raise TypeError(
                "pricer must satisfy the Pricer protocol (implement fair_price(state) -> float)"
            )
        if not (0.0 < kelly_cap <= 1.0):
            raise ValueError(f"kelly_cap must be in (0, 1]; got {kelly_cap}")
        if z_threshold < 0.0:
            raise ValueError(f"z_threshold must be >= 0; got {z_threshold}")
        if z_window < 2:
            raise ValueError(f"z_window must be >= 2; got {z_window}")
        if not (0.0 < clip_eps < 0.5):
            raise ValueError(f"clip_eps must be in (0, 0.5); got {clip_eps}")
        self.pricer = pricer
        self.kelly_cap = float(kelly_cap)
        self.z_threshold = float(z_threshold)
        self.z_window = int(z_window)
        self.clip_eps = float(clip_eps)

    # ------------------------------------------------------------------
    # Public protocol — signal / position / pnl
    # ------------------------------------------------------------------

    def signal(self, state: MarketState | pd.DataFrame) -> float | pd.Series:
        """Mispricing signal.

        Scalar mode: returns a single float in roughly ``[-3, +3]``, the
        z-scored difference ``fair − market``. The sign carries the
        direction (positive = buy YES).

        DataFrame mode: vectorised across rows. Required columns:
        ``market_price`` (float), ``fair_price`` (precomputed by the
        caller — or this method runs ``self.pricer.fair_price`` per row if
        ``fair_price`` is absent). Returns a ``pd.Series`` of z-scores.
        """
        if isinstance(state, MarketState):
            fair = _clip_prob(self.pricer.fair_price(state), self.clip_eps)
            mkt = _clip_prob(state.market_price, self.clip_eps)
            # Single-state z-score has no trailing window, so we use the
            # *informational* SE of a Bernoulli at p=mkt: sqrt(p*(1-p)).
            se = math.sqrt(mkt * (1.0 - mkt))
            if se <= 0.0:
                return 0.0
            return float((fair - mkt) / se)

        if not isinstance(state, pd.DataFrame):
            raise TypeError(
                f"signal() accepts MarketState or pd.DataFrame; got {type(state).__name__}"
            )
        df = state.copy()
        if "market_price" not in df.columns:
            raise ValueError("DataFrame must have a 'market_price' column")
        if "fair_price" not in df.columns:
            # Caller didn't precompute — synthesise via the pricer.
            features = df["features"].tolist() if "features" in df.columns else [{}] * len(df)
            ttr = (
                df["time_to_resolution_days"].tolist()
                if "time_to_resolution_days" in df.columns
                else [30.0] * len(df)
            )
            fair: list[float] = []
            for mp, t, feat in zip(df["market_price"].tolist(), ttr, features, strict=False):
                st = MarketState(
                    market_price=float(mp),
                    time_to_resolution_days=float(t),
                    features=dict(feat) if isinstance(feat, dict) else {},
                )
                fair.append(_clip_prob(self.pricer.fair_price(st), self.clip_eps))
            df["fair_price"] = fair
        gap = df["fair_price"].astype(float).clip(self.clip_eps, 1.0 - self.clip_eps) - df[
            "market_price"
        ].astype(float).clip(self.clip_eps, 1.0 - self.clip_eps)
        z = _zscore(gap, window=self.z_window).fillna(0.0)
        return z.rename("signal")

    def position(self, signal: float | pd.Series) -> float | pd.Series:
        """Kelly-capped position from a signal value or Series.

        Logic:
            * Below the z_threshold we sit out (position = 0).
            * Otherwise Kelly fraction ∝ ``signal`` (clipped at kelly_cap).
            * Sign carries direction (positive = long YES).

        Kelly approximation: full Kelly for a 50/50 bet with edge ``e`` is
        ``2e``. We map ``signal/3`` to ``edge`` (so a ±3σ mispricing maps
        to ~100% edge), then cap by ``kelly_cap``.
        """
        if isinstance(signal, pd.Series):
            sig = signal.fillna(0.0).astype(float)
            mag = sig.abs()
            kept = (mag >= self.z_threshold).astype(float)
            # Map signal/3 -> edge approximation, then 2*edge -> Kelly.
            kelly = (2.0 * sig / 3.0).clip(-self.kelly_cap, self.kelly_cap)
            return (kelly * kept).rename("position")

        if not math.isfinite(float(signal)):
            return 0.0
        if abs(signal) < self.z_threshold:
            return 0.0
        kelly = 2.0 * float(signal) / 3.0
        if kelly > self.kelly_cap:
            return self.kelly_cap
        if kelly < -self.kelly_cap:
            return -self.kelly_cap
        return float(kelly)

    def pnl(
        self, position: float | pd.Series, realized: int | float | pd.Series
    ) -> float | pd.Series:
        """Realised PnL given a position and realised outcome.

        For an unresolved market with no outcome yet, callers should pass
        ``realized = market_price_t`` (a mark-to-market PnL). For a
        resolved market, ``realized`` is the 0/1 binary outcome.

        PnL ≈ ``position * (realized - entry_price)``. Because this is a
        protocol method (no entry price stored), callers can also pass
        ``realized = outcome - entry_price`` (i.e. pre-compute the return)
        and we'll simply multiply.

        To keep the API symmetric with other Strategies, the simplest case
        — ``realized`` is the return already — just multiplies through.
        """
        if isinstance(position, pd.Series) or isinstance(realized, pd.Series):
            p = (
                pd.Series(position).astype(float)
                if not isinstance(position, pd.Series)
                else position
            )
            r = (
                pd.Series(realized).astype(float)
                if not isinstance(realized, pd.Series)
                else realized
            )
            out = (p.reindex(r.index).fillna(0.0) * r.astype(float)).rename("pnl")
            return out
        return float(position) * float(realized)

    # ------------------------------------------------------------------
    # Convenience: compute a daily PnL series from a market-data frame
    # ------------------------------------------------------------------

    def compute_daily_pnl(
        self,
        frame: pd.DataFrame,
        *,
        outcome_col: str = "outcome",
    ) -> pd.Series:
        """Vectorised pipeline: signal -> position -> realised return -> pnl.

        Required columns:
            * ``market_price`` — YES price on each date
            * ``fair_price`` (optional — synthesised if absent)
            * ``outcome`` — realised resolution (0/1) repeated on every row
              of a given market, OR the next-day price for mark-to-market.

        Returns a daily PnL series.
        """
        sig = self.signal(frame)
        pos = self.position(sig)
        if outcome_col not in frame.columns:
            raise ValueError(f"frame is missing required column {outcome_col!r}")
        realized = frame[outcome_col].astype(float) - frame["market_price"].astype(float)
        return self.pnl(pos, realized)


# ---------------------------------------------------------------------------
# Registry hook — gated by SHOULD_DEPLOY + binary-pricing-results.md
# ---------------------------------------------------------------------------


def _binary_pricing_results_present() -> bool:
    """Check whether T83's verdict doc has landed."""
    here = os.path.dirname(os.path.abspath(__file__))
    # repo root = api/src/pfm/strategies/.. .. .. ..
    repo_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
    return os.path.exists(os.path.join(repo_root, "docs", "binary-pricing-results.md"))


def register_if_ready(*, pricer: Pricer | None = None, force: bool = False) -> Any:
    """Register the strategy in :mod:`pfm.strategies_registry` iff ready.

    Conditions:
        * :data:`SHOULD_DEPLOY` is ``True`` (or ``force=True`` for tests)
        * ``docs/binary-pricing-results.md`` exists
        * a ``pricer`` instance was supplied

    Returns the :class:`pfm.strategies_registry.Strategy` instance if
    registered, else ``None``.

    The default at module-import time is to NOT register; callers
    (e.g. a future ``pfm.main`` startup hook) must opt-in explicitly with a
    real pricer once the deployment review has flipped
    :data:`SHOULD_DEPLOY` to ``True``.
    """
    if pricer is None:
        return None
    if not (force or SHOULD_DEPLOY):
        return None
    if not (force or _binary_pricing_results_present()):
        return None

    # Lazy import — avoids a hard dep on the registry at module load.
    from pfm.strategies_registry import Strategy, register

    alpha = BinaryPricingAlpha(pricer=pricer)

    def _signal_fn(prices: pd.DataFrame) -> pd.Series:
        """Adapter: registry expects ``signal(prices) -> pd.Series``."""
        if "market_price" not in prices.columns:
            # Fall back: treat ``close`` as a probability series in [0,1].
            if "close" in prices.columns:
                norm = (prices["close"] - prices["close"].min()) / max(
                    prices["close"].max() - prices["close"].min(), 1e-9
                )
                frame = pd.DataFrame({"market_price": norm.clip(0.01, 0.99)})
            else:
                raise ValueError("prices needs 'market_price' or 'close' column")
        else:
            frame = prices
        return alpha.signal(frame)

    def _position_fn(signal: pd.Series) -> pd.Series:
        return alpha.position(signal)

    strat = Strategy(name=STRATEGY_NAME, signal=_signal_fn, position=_position_fn, pnl=None)
    register(strat)
    return strat


# ---------------------------------------------------------------------------
# alpha_strategies.json entry — the schema used for catalog publication.
# This helper is exported for use by a future deployment script (NOT
# auto-invoked here; flipping SHOULD_DEPLOY = True is a human decision).
# ---------------------------------------------------------------------------


def alpha_catalog_entry(
    *,
    pricer_name: str,
    robustness: dict[str, Any],
) -> dict[str, Any]:
    """Build the dict to APPEND to ``web/data/alpha_strategies.json``.

    Per Track-L T84 spec, the entry MUST include:
        ``pair_id``, ``tier``, ``label``, ``deploy_params``,
        ``theory_ref``, ``robustness``.
    """
    return {
        "pair_id": STRATEGY_NAME,
        "tier": TIER,
        "label": f"Binary-pricing mispricing ({pricer_name})",
        "deploy_params": {
            "kelly_cap": 0.20,
            "z_threshold": 1.0,
            "z_window": 20,
            "clip_eps": 0.01,
            "pricer": pricer_name,
        },
        "theory_ref": (
            "Risk-neutral pricing of binary digitals (digital payoff as the "
            "derivative of a call spread) + market-microstructure mispricing decay"
        ),
        "robustness": dict(robustness),
        "should_deploy_at_publish_time": SHOULD_DEPLOY,
        "anti_alpha_rule": (
            "CLAUDE.md anti-alpha rule: this entry is only legitimate when "
            "all 4 stress-test quarters returned Sharpe >= 0.5 with no "
            "sign-flip. Ceiling is B_VALIDATED; promotion to A_GOLD "
            "requires 4+ quarters of LIVE paper confirmation."
        ),
    }


__all__ = [
    "SHOULD_DEPLOY",
    "STRATEGY_NAME",
    "TIER",
    "BinaryPricingAlpha",
    "MarketState",
    "Pricer",
    "alpha_catalog_entry",
    "register_if_ready",
]
