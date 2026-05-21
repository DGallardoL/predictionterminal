"""Implied-vol vs realized-vol arbitrage in election / binary markets (W12-26).

This module materialises an *exploratory* alpha that compares the implied
probability of a binary event reported by a prediction market (e.g.
Polymarket) against the same event's probability backed-out from listed
options (a VIX-style overlay). When the two diverge by more than what the
realised volatility / time-to-event budget would justify, we have an
arbitrage *candidate* — long the cheaper side, short the richer one.

Theory
------
Let

    p_poly    = prediction-market implied probability (Polymarket YES mid)
    p_options = options-implied probability for the same binary event
                (e.g. probability of SPX < strike at expiry, derived from
                listed-options vol surface)
    σ̂_r      = annualised realised volatility of the underlying over
                a trailing window (e.g. 30d log-return σ)
    τ        = days to event (calendar) / 365

Define the **raw divergence**

    Δ_t = p_poly - p_options

If the binary event is genuinely the same and both markets are efficient,
``Δ_t`` should fluctuate around zero with a scale that grows with the
event's *uncertainty budget* — heuristically ``√(σ̂_r · τ)``. So we
normalise:

    raw_signal_t = Δ_t / √(σ̂_r · τ)

To remove slow regime drift (e.g. Polymarket structurally trades at a
premium for popular elections), we rolling-z-score ``raw_signal_t`` over
``z_window`` days. The final signal is

    s_t = (raw_signal_t - μ_window) / σ_window

When ``s_t`` is materially positive, ``p_poly`` is rich vs ``p_options`` →
**short YES on Polymarket** (and optionally hedge with options). When
``s_t`` is materially negative, ``p_poly`` is cheap → **long YES on
Polymarket**. The sign convention here is "trade the signal", not "fade".
The standard Kelly cap and z-threshold keep position sizing honest.

Tier — DO NOT DEPLOY
--------------------
Per the CLAUDE.md anti-alpha rule:

    *Don't deploy regime-driven alphas without a 4-quarter robustness
    check. Every "wow" backtest from a single window must be
    cross-validated against ≥4 disjoint quarters.*

This strategy has **not** been validated against 4Q robustness, BH-FDR
multiple-testing, or transaction-cost sensitivity. It MUST stay at tier
``B_VALIDATED`` with :data:`SHOULD_DEPLOY` ``False`` until those gates
clear. The module deliberately does **not** auto-register in
:mod:`pfm.strategies_registry` and is **not** added to
``web/data/alpha_strategies.json``.

References
----------
* CBOE 2020, "VIX Volatility Index Methodology" (binary event vol overlay)
* Hull 2018, "Options, Futures and Other Derivatives" (risk-neutral pricing)
* CLAUDE.md memory: ``user_hedge_fund_frame.md`` — favour cost-aware
  sizing with a defensible theoretical hook.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tier constants — B_VALIDATED, SHOULD_DEPLOY=False per anti-alpha rule.
# ---------------------------------------------------------------------------

#: Anti-alpha rule: pending 4Q robustness. Do NOT register or deploy.
SHOULD_DEPLOY: bool = False

#: CLAUDE.md tier label. ``B_VALIDATED`` = passed a single-window backtest
#: but not the 4-quarter stress harness.
TIER: str = "B_VALIDATED"

#: Canonical strategy name (used as registry key when/if promoted).
STRATEGY_NAME: str = "iv-realized-vol-arb"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IVRealizedVolState:
    """Snapshot of a binary event with both prediction-market & options views.

    Attributes
    ----------
    implied_prob_poly:
        Polymarket-implied YES probability in ``(0, 1)``.
    implied_prob_options:
        Same event's probability backed-out from listed options, in
        ``(0, 1)``.
    days_to_event:
        Calendar days to event resolution. Must be ``> 0`` for the signal
        to be defined; ``0`` (or negative) returns a 0 signal.
    realized_vol:
        Annualised realised volatility of the underlying over a trailing
        window (e.g. 30d log-return σ, in decimals — ``0.20`` = 20%/yr).
    features:
        Optional auxiliary features (liquidity, spread, etc.).
    """

    implied_prob_poly: float
    implied_prob_options: float
    days_to_event: float
    realized_vol: float
    features: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: float = 252.0
_CAL_DAYS_PER_YEAR: float = 365.0


def _clip_prob(p: float, eps: float = 0.01) -> float:
    """Clip a probability into ``[eps, 1 - eps]`` (CLAUDE.md ε default 0.01)."""
    if not math.isfinite(p):
        return 0.5
    return float(min(max(p, eps), 1.0 - eps))


def raw_divergence(
    p_poly: float,
    p_options: float,
    *,
    eps: float = 0.01,
) -> float:
    """Return ``p_poly - p_options`` after clipping both into ``[eps, 1-eps]``."""
    return _clip_prob(p_poly, eps) - _clip_prob(p_options, eps)


def vol_budget(realized_vol: float, days_to_event: float) -> float:
    """Compute the uncertainty-budget scaler ``√(σ̂_r · τ)``.

    Returns ``0.0`` for non-finite inputs, zero / negative time-to-event,
    or zero / negative realised vol. Callers must handle the ``0`` case
    (we do — the signal becomes 0 to avoid divide-by-zero).
    """
    if not math.isfinite(realized_vol) or not math.isfinite(days_to_event):
        return 0.0
    if realized_vol <= 0.0 or days_to_event <= 0.0:
        return 0.0
    tau = days_to_event / _CAL_DAYS_PER_YEAR
    return float(math.sqrt(realized_vol * tau))


def raw_signal(
    p_poly: float,
    p_options: float,
    realized_vol: float,
    days_to_event: float,
    *,
    eps: float = 0.01,
) -> float:
    """Compute ``Δ / √(σ̂_r · τ)`` with safe handling of edge cases.

    Returns ``0.0`` when the budget is zero (T=0, σ=0, non-finite inputs).
    """
    budget = vol_budget(realized_vol, days_to_event)
    if budget <= 0.0:
        return 0.0
    delta = raw_divergence(p_poly, p_options, eps=eps)
    if not math.isfinite(delta):
        return 0.0
    return float(delta / budget)


def _rolling_z(values: pd.Series, window: int) -> pd.Series:
    """Trailing rolling-window z-score with min_periods=window."""
    mu = values.rolling(window=window, min_periods=window).mean()
    sd = values.rolling(window=window, min_periods=window).std(ddof=1)
    z = (values - mu) / sd.replace(0.0, np.nan)
    return z.fillna(0.0)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class IVRealizedVolArb:
    """Implied-vol vs realised-vol arbitrage in binary / election markets.

    Parameters
    ----------
    kelly_cap:
        Hard upper bound on the absolute position size. CLAUDE.md
        recommends ≤0.25 for speculative alphas; default ``0.15`` here is
        conservative because the strategy is B_VALIDATED, not A_GOLD.
    z_threshold:
        Minimum ``|z|`` required to take a position. Below this, sit out.
        Default ``1.8`` (between the 90th and 95th percentile of standard
        normal — i.e. only modestly extreme divergences).
    z_window:
        Trailing window (in observations / days) for the z-score
        normalisation. Default ``30`` to mirror the realised-vol window.
    clip_eps:
        Probability-clipping epsilon (CLAUDE.md default 0.01).
    fade_sign:
        ``+1`` to **trade-with** the signal (when ``p_poly`` is rich,
        short YES). ``-1`` to **fade** the signal. Default ``+1``.

    Notes
    -----
    * ``name`` / ``tier`` are class-level constants matching the
      :mod:`pfm.strategies.binary_pricing_alpha` pattern.
    * ``signal`` / ``position`` / ``pnl`` accept either a single
      :class:`IVRealizedVolState` (scalar mode) or a ``pd.DataFrame``
      (vectorised mode).
    * Per CLAUDE.md, this strategy is **B_VALIDATED, SHOULD_DEPLOY=False**.
      Do **not** auto-register it in ``pfm.strategies_registry`` or add
      it to ``web/data/alpha_strategies.json``.
    """

    name: str = STRATEGY_NAME
    tier: str = TIER

    def __init__(
        self,
        *,
        kelly_cap: float = 0.15,
        z_threshold: float = 1.8,
        z_window: int = 30,
        clip_eps: float = 0.01,
        fade_sign: int = 1,
    ) -> None:
        if not (0.0 < kelly_cap <= 1.0):
            raise ValueError(f"kelly_cap must be in (0, 1]; got {kelly_cap}")
        if z_threshold < 0.0:
            raise ValueError(f"z_threshold must be >= 0; got {z_threshold}")
        if z_window < 2:
            raise ValueError(f"z_window must be >= 2; got {z_window}")
        if not (0.0 < clip_eps < 0.5):
            raise ValueError(f"clip_eps must be in (0, 0.5); got {clip_eps}")
        if fade_sign not in (-1, 1):
            raise ValueError(f"fade_sign must be -1 or +1; got {fade_sign}")
        self.kelly_cap = float(kelly_cap)
        self.z_threshold = float(z_threshold)
        self.z_window = int(z_window)
        self.clip_eps = float(clip_eps)
        self.fade_sign = int(fade_sign)

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def signal(
        self,
        state: IVRealizedVolState | pd.DataFrame,
    ) -> float | pd.Series:
        """Z-scored divergence signal.

        Scalar mode (``IVRealizedVolState``)
            Returns a single float. With no trailing window available,
            the "z-score" is approximated by the raw ratio ``Δ / √(σ̂_r·τ)``
            (so the magnitude is interpretable on a roughly unit scale).

        DataFrame mode
            Required columns: ``implied_prob_poly``, ``implied_prob_options``,
            ``days_to_event``, ``realized_vol``. Optional grouping column:
            ``market_id``. Returns a ``pd.Series`` of rolling-z-scored
            raw signals aligned to the frame index. The z-score is
            computed per market_id when that column is present.
        """
        if isinstance(state, IVRealizedVolState):
            return raw_signal(
                state.implied_prob_poly,
                state.implied_prob_options,
                state.realized_vol,
                state.days_to_event,
                eps=self.clip_eps,
            )

        if not isinstance(state, pd.DataFrame):
            raise TypeError(
                f"signal() accepts IVRealizedVolState or pd.DataFrame; got {type(state).__name__}"
            )

        df = state
        required = (
            "implied_prob_poly",
            "implied_prob_options",
            "days_to_event",
            "realized_vol",
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required column(s): {missing}")

        raw = df.apply(
            lambda r: raw_signal(
                float(r["implied_prob_poly"]),
                float(r["implied_prob_options"]),
                float(r["realized_vol"]),
                float(r["days_to_event"]),
                eps=self.clip_eps,
            ),
            axis=1,
        ).astype(float)

        if "market_id" in df.columns:
            # Per-market rolling z-score, preserving the original index.
            parts = []
            for _mid, g in df.groupby("market_id"):
                gz = _rolling_z(raw.loc[g.index], window=self.z_window)
                parts.append(gz)
            z = pd.concat(parts).reindex(df.index).fillna(0.0)
        else:
            z = _rolling_z(raw, window=self.z_window)

        return z.rename("signal")

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def position(
        self,
        signal: float | pd.Series,
        *,
        kelly_cap: float | None = None,
        z_threshold: float | None = None,
    ) -> float | pd.Series:
        """Kelly-capped position from a signal.

        Logic:
            * Below ``|z| < z_threshold`` → position = 0.
            * Else Kelly ≈ ``-fade_sign * z / 3`` (so a ±3σ deviation maps
              to ~33% Kelly weight before the cap). The negative sign in
              front of ``z`` reflects the trade-with-signal convention:
              when ``p_poly`` is rich (z > 0) we want to **short YES**
              (negative position).
            * Cap at ``±kelly_cap``.
        """
        kcap = self.kelly_cap if kelly_cap is None else float(kelly_cap)
        zthr = self.z_threshold if z_threshold is None else float(z_threshold)
        if kcap <= 0.0:
            raise ValueError(f"kelly_cap must be > 0; got {kcap}")
        if zthr < 0.0:
            raise ValueError(f"z_threshold must be >= 0; got {zthr}")

        if isinstance(signal, pd.Series):
            sig = signal.fillna(0.0).astype(float)
            mag = sig.abs()
            kept = (mag >= zthr).astype(float)
            kelly = (-self.fade_sign * sig / 3.0).clip(-kcap, kcap)
            return (kelly * kept).rename("position")

        s = float(signal)
        if not math.isfinite(s) or abs(s) < zthr:
            return 0.0
        kelly = -self.fade_sign * s / 3.0
        if kelly > kcap:
            return kcap
        if kelly < -kcap:
            return -kcap
        return float(kelly)

    # ------------------------------------------------------------------
    # PnL
    # ------------------------------------------------------------------

    def pnl(
        self,
        position: float | pd.Series,
        realized: int | float | pd.Series,
    ) -> float | pd.Series:
        """Realised PnL = position × realised return.

        ``realized`` semantics: callers may pass either a 0/1 resolution
        outcome (with the entry price subtracted externally to give a
        return), or a pre-computed return series.
        """
        if isinstance(position, pd.Series) or isinstance(realized, pd.Series):
            p = position if isinstance(position, pd.Series) else pd.Series(position).astype(float)
            r = realized if isinstance(realized, pd.Series) else pd.Series(realized).astype(float)
            out = (p.reindex(r.index).fillna(0.0) * r.astype(float)).rename("pnl")
            return out
        return float(position) * float(realized)

    # ------------------------------------------------------------------
    # Convenience: end-to-end pipeline
    # ------------------------------------------------------------------

    def compute_daily_pnl(
        self,
        frame: pd.DataFrame,
        *,
        outcome_col: str = "outcome",
    ) -> pd.Series:
        """Vectorised pipeline: signal -> position -> realised return -> pnl.

        Required columns: ``implied_prob_poly``, ``implied_prob_options``,
        ``days_to_event``, ``realized_vol`` and ``outcome`` (the realised
        resolution probability or 0/1).
        """
        if outcome_col not in frame.columns:
            raise ValueError(f"frame is missing required column {outcome_col!r}")
        sig = self.signal(frame)
        pos = self.position(sig)
        realized = frame[outcome_col].astype(float) - frame["implied_prob_poly"].astype(float)
        return self.pnl(pos, realized)


# ---------------------------------------------------------------------------
# Catalog entry helper (NOT auto-added; gated by SHOULD_DEPLOY).
# ---------------------------------------------------------------------------


def alpha_catalog_entry(
    *,
    robustness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a catalog-entry shape if/when this alpha clears 4Q robustness.

    Per the CLAUDE.md anti-alpha rule, this entry is **not** written to
    ``web/data/alpha_strategies.json`` automatically; a future sanitizer
    script must flip :data:`SHOULD_DEPLOY` to ``True`` and append this
    entry only after the 4-quarter stress + BH-FDR + deflated-Sharpe
    gates pass.
    """
    return {
        "pair_id": "polymarket_iv_realized_vol_arb_v1",
        "tier": TIER,
        "label": "IV vs Realized Vol arbitrage (election binaries)",
        "deploy_params": {
            "kelly_cap": 0.15,
            "z_threshold": 1.8,
            "z_window": 30,
            "clip_eps": 0.01,
            "fade_sign": 1,
        },
        "theory_ref": (
            "CBOE VIX methodology + Hull risk-neutral binary pricing; "
            "trade-with-signal when |z| > threshold."
        ),
        "robustness": dict(robustness or {}),
        "should_deploy_at_publish_time": SHOULD_DEPLOY,
        "anti_alpha_gate": ("Pending 4-quarter Sharpe stability + BH-FDR + deflated-Sharpe."),
    }


def register_if_ready(*, force: bool = False) -> Any:
    """Register in :mod:`pfm.strategies_registry` ONLY if cleared.

    Default behaviour: returns ``None`` because :data:`SHOULD_DEPLOY` is
    ``False`` (pending 4Q stress). ``force=True`` registers for testing
    purposes only.
    """
    if not (SHOULD_DEPLOY or force):
        return None
    # Lazy import to avoid pulling the registry at module import time.
    from pfm.strategies_registry import Strategy, register

    alpha = IVRealizedVolArb()

    def _signal_adapter(prices: pd.DataFrame) -> pd.Series:
        return alpha.signal(prices)

    def _position_adapter(signal: pd.Series) -> pd.Series:
        return alpha.position(signal)

    strat = Strategy(
        name=STRATEGY_NAME,
        signal=_signal_adapter,
        position=_position_adapter,
        pnl=None,
    )
    register(strat)
    return strat


__all__ = [
    "SHOULD_DEPLOY",
    "STRATEGY_NAME",
    "TIER",
    "IVRealizedVolArb",
    "IVRealizedVolState",
    "alpha_catalog_entry",
    "raw_divergence",
    "raw_signal",
    "register_if_ready",
    "vol_budget",
]
