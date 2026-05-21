"""Compare model probability vs market probability and emit a trade signal.

The signal layer is deliberately tiny:

* ``edge = model_prob - market_prob``
* ``|edge| >= edge_threshold`` ⇒ BUY YES (edge > 0) or BUY NO (edge < 0).
* otherwise WAIT.

We also derive a *Kelly stake fraction* assuming Polymarket pricing
``f* = (p * (1 - q) - (1 - p) * q) / ((1 - q) * q)`` where ``p`` is our model
probability and ``q`` is the market YES midpoint (i.e. the cost of YES). The
stake is clipped to [0, 0.20] so the UI doesn't suggest crazy sizes.

This module is pure — no I/O. The async router does the network calls and
hands the numbers in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pfm.crypto5min.predictor import ModelPrediction

DEFAULT_EDGE_THRESHOLD: float = 0.03
"""3 percentage-point gap. After we anchor the model to the market via
:data:`MARKET_ANCHOR_WEIGHT`, raw edges shrink ~5×, so the threshold
shrinks proportionally."""

MARKET_ANCHOR_WEIGHT: float = 0.50
# Was 0.90 — collapsed to market-tracker; 0.5 balances anchor vs raw GBM.
"""How much weight we give the *market's* probability when computing the
final model output, when the market price is available.

The Polymarket up-down 5m / 15m markets price implicit information our pure
GBM can't capture — short-horizon mean reversion, market-implied drift,
Chainlink lag expectations, etc. Pure GBM with realistic σ over-shoots:
80%+ model when market says 48% is *almost always* the model being wrong,
not the market.

A Bayesian-style anchor (final = 0.80·market + 0.20·gbm) keeps the model
close to the market consensus while still allowing microstructure (OFI,
whale flow, z-VWAP) to tilt the answer by a few percentage points. The
``edge`` then represents what we *uniquely* see beyond the market, capped
to a realistic ±5-10% by construction.

Set to 0.0 to use raw GBM (useful for backtests / model diagnostics)."""

_SIGNAL_BUY_YES = "BUY_YES"
_SIGNAL_BUY_NO = "BUY_NO"
_SIGNAL_WAIT = "WAIT"


def anchor_to_market(
    gbm_prob: float, market_prob: float | None, weight: float = MARKET_ANCHOR_WEIGHT
) -> float:
    """Return the market-anchored model probability.

    ``weight`` is the fraction of the final probability that comes from
    the market. When ``market_prob`` is ``None`` we return ``gbm_prob``
    unchanged so headless callers (no Polymarket available) still get the
    pure model signal.
    """
    if market_prob is None:
        return float(gbm_prob)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    return weight * float(market_prob) + (1.0 - weight) * float(gbm_prob)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Paired (model, market) probabilities with a discrete signal.

    ``model_prob_up`` is the *market-anchored* probability that the UI
    shows — close to the market most of the time, with small microstructure
    tilt. ``model_prob_gbm_raw`` is the pure GBM output before anchoring,
    kept for debug / backtest diagnostics.
    """

    slug: str
    asset: str
    window_minutes: int
    seconds_remaining: float
    model_prob_up: float
    market_prob_up: float
    edge: float
    signal: str
    edge_threshold: float
    kelly_fraction: float
    sigma_used_annual: float
    mu_used_annual: float
    model_prob_gbm_raw: float = 0.0
    market_anchor_weight: float = MARKET_ANCHOR_WEIGHT
    components: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "asset": self.asset,
            "window_minutes": self.window_minutes,
            "seconds_remaining": self.seconds_remaining,
            "model_prob_up": self.model_prob_up,
            "model_prob_gbm_raw": self.model_prob_gbm_raw,
            "market_anchor_weight": self.market_anchor_weight,
            "market_prob_up": self.market_prob_up,
            "edge": self.edge,
            "signal": self.signal,
            "edge_threshold": self.edge_threshold,
            "kelly_fraction": self.kelly_fraction,
            "sigma_used_annual": self.sigma_used_annual,
            "mu_used_annual": self.mu_used_annual,
            "components": dict(self.components),
        }


def decide_signal(edge: float, threshold: float = DEFAULT_EDGE_THRESHOLD) -> str:
    """Map ``edge = model_p - market_p`` to one of BUY_YES / BUY_NO / WAIT."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if edge >= threshold:
        return _SIGNAL_BUY_YES
    if edge <= -threshold:
        return _SIGNAL_BUY_NO
    return _SIGNAL_WAIT


def kelly_fraction(model_prob: float, market_prob: float, cap: float = 0.20) -> float:
    """Fractional Kelly stake for a YES/NO bet at Polymarket pricing.

    For a YES bet at price ``q``, payoff is ``(1 - q) / q`` on win and
    ``-1`` on loss. Kelly is::

        f* = (p * (1 - q) - (1 - p) * q) / ((1 - q) * q)

    For a NO bet we use the symmetric form with ``p_no = 1 - p`` and
    ``q_no = 1 - q``. Result is clipped to [0, ``cap``] so the UI never
    suggests >20% of bankroll on a single contract.

    Returns 0.0 when the edge has the wrong sign (you'd be betting against
    the model).
    """
    if cap < 0:
        raise ValueError("cap must be non-negative")
    p = max(1e-6, min(1.0 - 1e-6, float(model_prob)))
    q = max(1e-6, min(1.0 - 1e-6, float(market_prob)))
    if p > q:
        f = (p * (1.0 - q) - (1.0 - p) * q) / ((1.0 - q) * q)
    elif p < q:
        p_no = 1.0 - p
        q_no = 1.0 - q
        f = (p_no * (1.0 - q_no) - (1.0 - p_no) * q_no) / ((1.0 - q_no) * q_no)
    else:
        return 0.0
    return max(0.0, min(cap, f))


def compare_market_vs_model(
    *,
    slug: str,
    asset: str,
    window_minutes: int,
    market_prob_up: float,
    prediction: ModelPrediction,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    kelly_cap: float = 0.20,
    market_anchor_weight: float = MARKET_ANCHOR_WEIGHT,
) -> ComparisonResult:
    """Assemble a :class:`ComparisonResult` from a model prediction + a market mid.

    Applies :func:`anchor_to_market` so the final ``model_prob_up`` is a
    weighted blend of the market and our raw GBM. The raw GBM output is
    kept in ``model_prob_gbm_raw`` for diagnostics.
    """
    if not 0.0 <= market_prob_up <= 1.0:
        raise ValueError(f"market_prob_up must be in [0, 1], got {market_prob_up}")
    gbm_raw = prediction.prob_up
    model_prob = anchor_to_market(gbm_raw, market_prob_up, weight=market_anchor_weight)
    edge = model_prob - market_prob_up
    signal = decide_signal(edge, threshold=edge_threshold)
    stake = kelly_fraction(model_prob, market_prob_up, cap=kelly_cap) if signal != "WAIT" else 0.0
    return ComparisonResult(
        slug=slug,
        asset=asset,
        window_minutes=window_minutes,
        seconds_remaining=prediction.seconds_remaining,
        model_prob_up=model_prob,
        model_prob_gbm_raw=gbm_raw,
        market_anchor_weight=market_anchor_weight,
        market_prob_up=market_prob_up,
        edge=edge,
        signal=signal,
        edge_threshold=edge_threshold,
        kelly_fraction=stake,
        sigma_used_annual=prediction.sigma_used_annual,
        mu_used_annual=prediction.mu_used_annual,
        components=dict(prediction.components),
    )
