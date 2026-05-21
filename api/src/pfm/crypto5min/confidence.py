"""Confidence + z-score derivations for the crypto5min model.

Three quantities the UI cares about beyond the raw probabilities:

* ``z_model``      — the standardized GBM distance to the resolution
                     threshold. This is the same ``z`` that goes into
                     ``Φ(z)`` inside :func:`predict_up_prob`. Positive ⇒
                     model leans up; |z_model| > 1.5 = strong view.

* ``z_edge``       — the gap between model and market expressed in units
                     of *model uncertainty*, where uncertainty is a
                     σ-jackknife: rerun the predictor with σ shifted by
                     ±``SIGMA_PERTURB`` and use the half-spread as a
                     standard-error proxy. |z_edge| > 2 means the edge
                     survives reasonable σ misspecification.

* ``confidence`` — a 0-100 score combining buffer warmup, live-engine
                   availability, edge magnitude and time-decay. Used to
                   classify the discrete signal as STRONG / MEDIUM / WEAK.

All four functions are pure (no I/O, no state). The router glues them onto
:class:`ComparisonResult` before serializing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from pfm.crypto5min.predictor import (
    SECONDS_PER_YEAR,
    PredictorInputs,
    predict_up_prob,
)

#: How much we perturb σ for the jackknife. ±20% bands cover the
#: realistic uncertainty in our σ blend (long-σ from 30 daily closes has
#: ~15% sampling error, short-σ from tick RV ~25% in thin minutes).
SIGMA_PERTURB: float = 0.20

#: Confidence rubric weights — sum to 100 when all components saturate.
WEIGHT_DATA_QUALITY: float = 30.0  # n_samples ≥ N_FULL_CREDIT → full credit
WEIGHT_ENGINE_QUALITY: float = 20.0  # live cryptostuff WS engine on?
WEIGHT_EDGE_MAGNITUDE: float = 30.0  # |edge| ≥ EDGE_FULL_CREDIT → full credit
WEIGHT_TIME_DECAY: float = 20.0  # 1 - secs_remaining / window → linear ramp

#: Saturation thresholds for the rubric weights.
N_FULL_CREDIT: int = 60
EDGE_FULL_CREDIT: float = 0.20

#: Confidence → signal_strength buckets.
STRENGTH_STRONG_THRESHOLD: float = 65.0
STRENGTH_MEDIUM_THRESHOLD: float = 35.0


@dataclass(frozen=True, slots=True)
class ConfidenceBreakdown:
    """Per-component decomposition of the 0-100 confidence score."""

    data_quality: float
    engine_quality: float
    edge_magnitude: float
    time_decay: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "data_quality": self.data_quality,
            "engine_quality": self.engine_quality,
            "edge_magnitude": self.edge_magnitude,
            "time_decay": self.time_decay,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """Everything the UI needs to render confidence + z-stuff for one row."""

    confidence_score: float
    signal_strength: str
    z_model: float
    z_edge: float | None
    breakdown: ConfidenceBreakdown
    components: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence_score": self.confidence_score,
            "signal_strength": self.signal_strength,
            "z_model": self.z_model,
            "z_edge": self.z_edge,
            "confidence_breakdown": self.breakdown.as_dict(),
            "confidence_components": dict(self.components),
        }


def compute_z_model(
    spot_t: float,
    spot_0: float,
    seconds_remaining: float,
    sigma_used_annual: float,
    mu_used_annual: float,
) -> float:
    """Return the standardized z that goes into Φ(·) inside the GBM predictor.

    At ``seconds_remaining == 0`` or ``sigma_used == 0`` the GBM collapses
    to a step function — we return a signed infinity proxy (±1e9) so
    downstream UI can render "MAX" without dividing by 0.
    """
    if spot_t <= 0 or spot_0 <= 0:
        raise ValueError("spot prices must be positive")
    if seconds_remaining <= 0 or sigma_used_annual <= 0:
        log_ratio = math.log(spot_t / spot_0)
        return 1e9 if log_ratio >= 0 else -1e9
    tau = seconds_remaining / SECONDS_PER_YEAR
    log_ratio = math.log(spot_t / spot_0)
    drift = (mu_used_annual - 0.5 * sigma_used_annual**2) * tau
    denom = sigma_used_annual * math.sqrt(tau)
    return (log_ratio + drift) / denom


def compute_z_edge(
    base_inputs: PredictorInputs,
    base_model_prob: float,
    market_prob: float | None,
    *,
    perturb: float = SIGMA_PERTURB,
) -> float | None:
    """σ-jackknife z-score for the edge.

    Reruns :func:`predict_up_prob` with σ_long and σ_short scaled by
    ``(1 ± perturb)`` and uses the half-spread as a proxy for SE(model).
    Returns ``edge / SE`` clipped to ±50 for sanity (a huge value here is
    less informative than the edge itself).

    Returns ``None`` when:
    * ``market_prob`` is unavailable (no edge to test).
    * SE is exactly 0 (e.g. expiry edge case → trivially deterministic).
    """
    if market_prob is None:
        return None
    edge = base_model_prob - market_prob
    sigma_long = base_inputs.sigma_long_annual
    sigma_short = base_inputs.sigma_short_annual

    def _rebuild(scale: float) -> PredictorInputs:
        return PredictorInputs(
            spot_t=base_inputs.spot_t,
            spot_0=base_inputs.spot_0,
            seconds_remaining=base_inputs.seconds_remaining,
            sigma_long_annual=sigma_long * scale if sigma_long is not None else None,
            sigma_short_annual=sigma_short * scale if sigma_short is not None else None,
            ofi_1m=base_inputs.ofi_1m,
            z_vwap=base_inputs.z_vwap,
            whale_signed_notional_5m=base_inputs.whale_signed_notional_5m,
            notional_5m=base_inputs.notional_5m,
        )

    pred_low = predict_up_prob(_rebuild(1.0 - perturb))
    pred_high = predict_up_prob(_rebuild(1.0 + perturb))
    se = abs(pred_high.prob_up - pred_low.prob_up) / 2.0
    if se <= 0:
        # σ-perturbation didn't move the prob. Happens in two cases:
        #   (a) GBM collapsed to a step (T=0) — edge is trivially 0/1.
        #   (b) ATM regime (log_ratio≈0) where prob ≈ 0.5 for *any* σ.
        # In case (b) the edge is still informative — it reflects market
        # sentiment vs our drift-only contribution. Return a finite signed
        # value proportional to edge so the UI doesn't see "—".
        if base_inputs.seconds_remaining <= 0:
            return None
        # Sign of z_edge matches sign of edge; magnitude bounded.
        return max(-50.0, min(50.0, edge * 100.0))
    z = edge / se
    # Clip for sanity — beyond ±50 the number isn't meaningfully different
    # from infinity and just clutters the UI.
    return max(-50.0, min(50.0, z))


def _data_quality_score(n_samples: int) -> float:
    """0 → at 0 samples; full credit (WEIGHT_DATA_QUALITY) at N_FULL_CREDIT+."""
    if n_samples <= 0:
        return 0.0
    return min(WEIGHT_DATA_QUALITY, WEIGHT_DATA_QUALITY * n_samples / N_FULL_CREDIT)


def _engine_quality_score(live_engine_used: bool) -> float:
    return WEIGHT_ENGINE_QUALITY if live_engine_used else 0.0


def _edge_magnitude_score(edge: float | None) -> float:
    if edge is None:
        return 0.0
    return min(WEIGHT_EDGE_MAGNITUDE, WEIGHT_EDGE_MAGNITUDE * abs(edge) / EDGE_FULL_CREDIT)


def _time_decay_score(seconds_remaining: float, window_seconds: int) -> float:
    """Approaching expiry → more credit. Open-of-window = 0; expiry = full credit."""
    if window_seconds <= 0:
        return 0.0
    progress = 1.0 - max(0.0, min(1.0, seconds_remaining / window_seconds))
    return WEIGHT_TIME_DECAY * progress


def signal_strength_from_confidence(confidence: float) -> str:
    """Bucket confidence into STRONG / MEDIUM / WEAK."""
    if confidence >= STRENGTH_STRONG_THRESHOLD:
        return "STRONG"
    if confidence >= STRENGTH_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "WEAK"


def compute_confidence_score(
    *,
    n_samples: int,
    live_engine_used: bool,
    edge: float | None,
    seconds_remaining: float,
    window_seconds: int,
) -> ConfidenceBreakdown:
    """Combine the 4 rubric components into a 0-100 score.

    Returns the breakdown dataclass so the UI / debug surface can show
    *why* the score is what it is (e.g. "confidence 42 — engine off, low
    edge, mid-window").
    """
    dq = _data_quality_score(n_samples)
    eq = _engine_quality_score(live_engine_used)
    em = _edge_magnitude_score(edge)
    td = _time_decay_score(seconds_remaining, window_seconds)
    total = max(0.0, min(100.0, dq + eq + em + td))
    return ConfidenceBreakdown(
        data_quality=dq,
        engine_quality=eq,
        edge_magnitude=em,
        time_decay=td,
        total=total,
    )


def build_confidence_result(
    *,
    base_inputs: PredictorInputs,
    base_model_prob: float,
    market_prob: float | None,
    sigma_used_annual: float,
    mu_used_annual: float,
    n_samples: int,
    live_engine_used: bool,
    window_seconds: int,
) -> ConfidenceResult:
    """Top-level convenience: stitches z_model + z_edge + confidence into one."""
    edge = (base_model_prob - market_prob) if market_prob is not None else None
    z_model = compute_z_model(
        spot_t=base_inputs.spot_t,
        spot_0=base_inputs.spot_0,
        seconds_remaining=base_inputs.seconds_remaining,
        sigma_used_annual=sigma_used_annual,
        mu_used_annual=mu_used_annual,
    )
    z_edge = compute_z_edge(
        base_inputs=base_inputs,
        base_model_prob=base_model_prob,
        market_prob=market_prob,
    )
    breakdown = compute_confidence_score(
        n_samples=n_samples,
        live_engine_used=live_engine_used,
        edge=edge,
        seconds_remaining=base_inputs.seconds_remaining,
        window_seconds=window_seconds,
    )
    strength = signal_strength_from_confidence(breakdown.total)
    return ConfidenceResult(
        confidence_score=breakdown.total,
        signal_strength=strength,
        z_model=z_model,
        z_edge=z_edge,
        breakdown=breakdown,
        components={
            "sigma_perturb": SIGMA_PERTURB,
            "n_full_credit": N_FULL_CREDIT,
            "edge_full_credit": EDGE_FULL_CREDIT,
        },
    )
