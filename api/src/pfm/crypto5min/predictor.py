"""Model probability that a Binance spot price closes above its window-start
reference within a fixed 5m / 15m window.

We use **closed-form GBM with a microstructure-aware drift**:

    log_ratio   = ln(S_t / S_0)
    drift       = mu_eff * T
    variance    = sigma_eff^2 * T
    P(S_T > S_0) = Phi( (log_ratio + drift - 0.5 * variance) / sqrt(variance) )

where ``mu_eff`` and ``sigma_eff`` are **derived from the cryptostuff signal
engine** instead of being hard-coded constants:

* ``sigma_eff`` (annualized) blends a *long* historical estimate
  (Binance 30d daily-close σ — robust, slow-moving) with the *short* tick-
  derived σ from cryptostuff (responsive, can spike). We use the
  variance-weighted geometric mean so neither dominates.

* ``mu_eff`` (annualized) is a bounded function of the live order-flow
  imbalance (OFI ∈ [-1, +1]). At extreme imbalance ±1 the drift caps at
  ±30%/yr; at OFI=0 there is no drift. This is the **only** way the model
  expresses directional bias from microstructure — everything else flows
  through the variance / time-decay machinery.

Optional features the predictor consumes when available:

* ``z_vwap``        — mean-reversion overlay. When |z| > 2 the drift is
                      shrunk and a small *opposite-direction* term is added
                      (consistent with the fade-extremes literature).
* ``whale_flow``    — net signed whale notional in last 5 min (in USD). Adds
                      a tiny extra drift bias capped at ±15%/yr.

Everything is **pure**: no I/O, no clock, no shared state. The caller passes
the spot, the window start spot, the seconds remaining, and the
microstructure features; the function returns the probability plus a
structured breakdown for debugging / UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SECONDS_PER_YEAR: float = 365.0 * 24.0 * 3600.0
"""Crypto is 24/7 so a year is plain wall-clock seconds."""

#: Floor and ceiling for the effective annualized vol used in the GBM. The
#: floor stops a flat tape from collapsing the predictor into a step; the
#: ceiling catches pathological tick-σ spikes in thin minutes.
#: For very short windows (5min, 15min) the *effective* σ is intraday vol
#: which is much higher than daily σ. The 30-day daily-close σ
#: systematically under-estimates 5m intraday σ by ~3-4×. Without an
#: adaptive floor the model collapses to 0.5%/99.5% on tiny ±0.1% moves
#: while the market still prices 40-60% — a useless signal. The
#: ``adaptive_sigma_floor`` function below sets a window-aware minimum.
SIGMA_FLOOR: float = 0.10
SIGMA_CEILING: float = 3.00

#: Per-horizon minimum effective σ (annualized). Calibrated from observed
#: intraday RV vs daily-close σ for BTC + ETH (intraday is ~3-5× higher).
#: For windows shorter than these the GBM defaults to assuming reasonable
#: vol — well above the daily-σ value that would otherwise apply.
SIGMA_FLOOR_BY_SECONDS: tuple[tuple[float, float], ...] = (
    (60.0, 1.20),  # < 60s remaining → ≥120%/yr
    (300.0, 0.90),  # < 5 min → ≥90%/yr
    (900.0, 0.70),  # < 15 min → ≥70%/yr
    (3600.0, 0.50),  # < 1h → ≥50%/yr
)

#: Per-asset short-horizon dollar volatility, calibrated empirically against
#: Polymarket up-down 5m markets (see ``arbstuff/crypto_jump_arb.py``). Units:
#: USD / √second. So expected $-noise over T seconds = STD_PER_SEC × √T.
#:
#: This is the *signature* of intraday vol that the 30d daily-close σ
#: systematically under-estimates. We use it as a baseline σ-anchor for
#: short-window predictions — combined with the daily and tick-σ blends via
#: ``blend_sigma_with_calibrated`` below, the GBM produces values that
#: actually match the Polymarket midpoint instead of clipping to 0.5%/99.5%.
STD_PER_SEC: dict[str, float] = {
    "BTC": 4.0,
    "ETH": 0.30,
    "SOL": 0.05,
    "BNB": 0.10,
    "XRP": 0.002,
    "ADA": 0.001,
    "AVAX": 0.02,
    "MATIC": 0.001,
    "DOGE": 0.0003,
    "LINK": 0.01,
}

#: Seconds-per-year for converting per-second log returns into annualised σ.
SECONDS_PER_YEAR_FLOAT: float = SECONDS_PER_YEAR  # alias for clarity

#: Maximum |mu_eff| in /yr. At OFI=±1 + whale flow saturation we cap drift
#: at this so extreme imbalance can never override variance entirely.
MU_OFI_SCALE: float = 0.30
MU_WHALE_SCALE: float = 0.15
MU_CAP: float = 0.45

#: |z_vwap| above this triggers the mean-reversion shrink + opposite drift.
Z_REV_THRESHOLD: float = 2.0
Z_REV_SHRINK_AT: float = 4.0
"""At |z|>=4 the OFI drift is fully shrunk to 0 and replaced by a small
opposite-direction term (mean-reversion takeover)."""

#: Variance-weighting between long-σ and short-σ. ``LAMBDA_SHORT = 0.4``
#: means 40% of the variance comes from the short-horizon tick estimate,
#: which is responsive enough to register a regime shift but not enough to
#: be hijacked by a single jumpy minute.
LAMBDA_SHORT: float = 0.4


@dataclass(frozen=True, slots=True)
class PredictorInputs:
    """All inputs required to score one window.

    Only ``spot_t``, ``spot_0`` and ``seconds_remaining`` are mandatory;
    the rest default to "no information" so the predictor still works
    when the WS engine is off (it then collapses to the pure-GBM baseline
    used in :mod:`pfm.btc_arb`).

    ``asset`` (e.g. ``"BTC"``) lets ``blend_sigma`` pull the calibrated
    per-asset σ from :data:`STD_PER_SEC` — this is the intraday-vol anchor
    that prevents short-window predictions from collapsing to 0.5%/99.5%
    on tiny moves while the market still prices 40-60%.
    """

    spot_t: float
    spot_0: float
    seconds_remaining: float
    sigma_long_annual: float | None = None
    sigma_short_annual: float | None = None
    ofi_1m: float = 0.0
    z_vwap: float | None = None
    whale_signed_notional_5m: float | None = None
    notional_5m: float | None = None
    asset: str | None = None

    def __post_init__(self) -> None:
        if self.spot_t <= 0 or self.spot_0 <= 0:
            raise ValueError("spot prices must be positive")
        if self.seconds_remaining < 0:
            raise ValueError("seconds_remaining must be non-negative")
        if abs(self.ofi_1m) > 1.001:
            raise ValueError(f"OFI must be in [-1, +1], got {self.ofi_1m}")


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """Output of the predictor.

    Always returns a probability in [0, 1] — the breakdown is informational
    and lets the UI explain *why* the model is leaning a given way.
    """

    prob_up: float
    sigma_used_annual: float
    mu_used_annual: float
    seconds_remaining: float
    log_ratio: float
    components: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prob_up": self.prob_up,
            "sigma_used_annual": self.sigma_used_annual,
            "mu_used_annual": self.mu_used_annual,
            "seconds_remaining": self.seconds_remaining,
            "log_ratio": self.log_ratio,
            "components": dict(self.components),
        }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function — scipy-free."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def adaptive_sigma_floor(seconds_remaining: float) -> float:
    """Return the minimum σ for a window with this many seconds left.

    Daily-derived σ understates intraday vol by ~3-5× at sub-15min scales.
    This floor stops the GBM from collapsing to 0.5%/99.5% on tiny moves
    when the market still prices 40-60% — the same regime the
    ``arbstuff/crypto_jump_arb`` model gets right.
    """
    if seconds_remaining <= 0:
        return SIGMA_FLOOR
    for threshold, floor in SIGMA_FLOOR_BY_SECONDS:
        if seconds_remaining < threshold:
            return floor
    return SIGMA_FLOOR


def calibrated_sigma_annual(asset: str, spot_price: float) -> float | None:
    """Annualised σ derived from the per-asset STD_PER_SEC table.

    ``STD_PER_SEC`` is in USD/√sec. We convert via:
        σ_per_sec_log = STD_PER_SEC / spot_price
        σ_annual = σ_per_sec_log × √SECONDS_PER_YEAR

    Returns ``None`` for unknown assets or non-positive prices.
    """
    std = STD_PER_SEC.get(asset.upper())
    if std is None or spot_price <= 0:
        return None
    sigma_per_sec_log = std / spot_price
    return sigma_per_sec_log * math.sqrt(SECONDS_PER_YEAR)


def blend_sigma(
    sigma_long: float | None,
    sigma_short: float | None,
    lambda_short: float = LAMBDA_SHORT,
    *,
    seconds_remaining: float | None = None,
    asset: str | None = None,
    spot_price: float | None = None,
) -> float:
    """Variance-weighted blend of long-horizon and short-horizon σ.

    When ``asset`` + ``spot_price`` are provided, ``calibrated_sigma_annual``
    is folded in as an *additional* anchor — for short crypto windows this
    is the dominant signal because daily-σ under-estimates intraday vol.

    When ``seconds_remaining`` is provided we also apply the adaptive σ
    floor from :func:`adaptive_sigma_floor`. This is the critical fix for
    the "model says 0.5% when market says 46%" regime at the boundary.
    """
    candidates: list[float] = []
    if sigma_long is not None:
        candidates.append(float(sigma_long))
    if sigma_short is not None:
        candidates.append(float(sigma_short))
    calibrated: float | None = None
    if asset is not None and spot_price is not None and spot_price > 0:
        calibrated = calibrated_sigma_annual(asset, spot_price)
        if calibrated is not None:
            candidates.append(calibrated)

    if not candidates:
        sigma = 0.60
    elif len(candidates) == 1:
        sigma = candidates[0]
    elif sigma_long is not None and sigma_short is not None and calibrated is None:
        # Pure long/short blend (preserves backward compat for tests).
        var_blend = (1.0 - lambda_short) * sigma_long**2 + lambda_short * sigma_short**2
        sigma = math.sqrt(var_blend)
    else:
        # General case: variance-weighted blend so LAMBDA_SHORT actually
        # influences σ when calibrated is present. Was previously max(.)
        # which let the calibrated leg always win and silently ignored
        # both long and short. Weights:
        #   w_calibrated = 0.5  (strong, data-derived per-asset anchor)
        #   w_short      = lambda_short (typ. 0.4)
        #   w_long       = 1 - w_calibrated - w_short
        # If any leg is None its weight is redistributed to the rest.
        w_cal_default = 0.5
        w_short_default = float(lambda_short)
        w_long_default = max(0.0, 1.0 - w_cal_default - w_short_default)
        weights: list[float] = []
        variances: list[float] = []
        if sigma_long is not None:
            weights.append(w_long_default)
            variances.append(float(sigma_long) ** 2)
        if sigma_short is not None:
            weights.append(w_short_default)
            variances.append(float(sigma_short) ** 2)
        if calibrated is not None:
            weights.append(w_cal_default)
            variances.append(float(calibrated) ** 2)
        wsum = sum(weights)
        if wsum > 0:
            normalised = [w / wsum for w in weights]
            var_blend = sum(w * v for w, v in zip(normalised, variances, strict=True))
            sigma = math.sqrt(var_blend)
        else:
            sigma = max(candidates)

    sigma = max(SIGMA_FLOOR, min(SIGMA_CEILING, sigma))
    if seconds_remaining is not None:
        sigma = max(sigma, adaptive_sigma_floor(seconds_remaining))
    return min(SIGMA_CEILING, sigma)


def ofi_drift(ofi_1m: float) -> float:
    """Bounded linear mapping OFI ∈ [-1, +1] → drift ∈ [-MU_OFI_SCALE, +MU_OFI_SCALE]."""
    bounded = max(-1.0, min(1.0, ofi_1m))
    return bounded * MU_OFI_SCALE


def whale_drift(whale_signed_5m: float | None, notional_5m: float | None) -> float:
    """Whale-flow bias as a fraction of total 5m notional, scaled and capped.

    ``whale_signed_5m`` is the net signed whale notional (buy whales positive,
    sell whales negative). ``notional_5m`` is the total absolute notional in
    the same window. The contribution to drift is ``ratio * MU_WHALE_SCALE``
    with ``ratio`` clipped to [-1, +1]. Returns 0.0 when either input is None
    or notional is 0.
    """
    if whale_signed_5m is None or notional_5m is None or notional_5m <= 0:
        return 0.0
    ratio = whale_signed_5m / notional_5m
    bounded = max(-1.0, min(1.0, ratio))
    return bounded * MU_WHALE_SCALE


def reversion_overlay(
    z_vwap: float | None,
    base_drift: float,
) -> tuple[float, float]:
    """Shrink the OFI/whale drift when |z_vwap| is extreme.

    Returns ``(adjusted_drift, opposite_pull)``:

    * ``adjusted_drift`` — base drift shrunk linearly between |z|=2 and |z|=4.
      Above |z|=4 the OFI drift is fully zeroed (over-extension dominates).
    * ``opposite_pull`` — a small opposite-direction drift (mean-reversion).
      Magnitude grows linearly from 0 at |z|=2 to ±MU_OFI_SCALE/2 at |z|=4.
    """
    if z_vwap is None:
        return base_drift, 0.0
    abs_z = abs(z_vwap)
    if abs_z < Z_REV_THRESHOLD:
        return base_drift, 0.0
    # Linear shrink: 1.0 at |z|=2 → 0.0 at |z|>=4
    shrink = max(0.0, min(1.0, (Z_REV_SHRINK_AT - abs_z) / (Z_REV_SHRINK_AT - Z_REV_THRESHOLD)))
    adjusted = base_drift * shrink
    pull_magnitude = (1.0 - shrink) * (MU_OFI_SCALE * 0.5)
    opposite_pull = -math.copysign(pull_magnitude, z_vwap)
    return adjusted, opposite_pull


def predict_up_prob(inputs: PredictorInputs) -> ModelPrediction:
    """Closed-form GBM Up-probability with microstructure overlay.

    Decomposes the drift into:
        mu_eff = mu_ofi(ofi_1m) + mu_whale(whale_flow / notional)
                 (then shrunk + reversion-overlaid by z_vwap)

    The variance is the blended σ_long/σ_short, time-scaled to the window
    remaining. At ``seconds_remaining == 0`` the predictor collapses to
    ``1.0`` if ``spot_t >= spot_0`` else ``0.0`` (matches Polymarket's
    Up-resolution convention: tie counts as Up).
    """
    log_ratio = math.log(inputs.spot_t / inputs.spot_0)
    sigma_used = blend_sigma(
        inputs.sigma_long_annual,
        inputs.sigma_short_annual,
        seconds_remaining=inputs.seconds_remaining,
        asset=inputs.asset,
        spot_price=inputs.spot_t,
    )
    mu_ofi = ofi_drift(inputs.ofi_1m)
    mu_whale = whale_drift(inputs.whale_signed_notional_5m, inputs.notional_5m)
    base_drift = mu_ofi + mu_whale
    drift_after_z, opposite_pull = reversion_overlay(inputs.z_vwap, base_drift)
    mu_used = drift_after_z + opposite_pull
    mu_used = max(-MU_CAP, min(MU_CAP, mu_used))

    components = {
        "mu_ofi": mu_ofi,
        "mu_whale": mu_whale,
        "mu_base": base_drift,
        "mu_after_z_shrink": drift_after_z,
        "mu_reversion_pull": opposite_pull,
        "sigma_long_annual": inputs.sigma_long_annual,
        "sigma_short_annual": inputs.sigma_short_annual,
        "lambda_short": LAMBDA_SHORT,
        "z_vwap": inputs.z_vwap,
    }

    if inputs.seconds_remaining == 0.0 or sigma_used == 0.0:
        prob = 1.0 if log_ratio >= 0.0 else 0.0
        return ModelPrediction(
            prob_up=prob,
            sigma_used_annual=sigma_used,
            mu_used_annual=mu_used,
            seconds_remaining=inputs.seconds_remaining,
            log_ratio=log_ratio,
            components=components,
        )

    tau = inputs.seconds_remaining / SECONDS_PER_YEAR
    drift = (mu_used - 0.5 * sigma_used**2) * tau
    denom = sigma_used * math.sqrt(tau)
    z = (log_ratio + drift) / denom
    prob = _norm_cdf(z)
    # Clip away the 0/1 endpoints so downstream Kelly / edge math doesn't blow up.
    prob = max(0.005, min(0.995, prob))
    return ModelPrediction(
        prob_up=prob,
        sigma_used_annual=sigma_used,
        mu_used_annual=mu_used,
        seconds_remaining=inputs.seconds_remaining,
        log_ratio=log_ratio,
        components=components,
    )


def predict_for_window(
    spot_t: float,
    spot_0: float,
    seconds_remaining: float,
    *,
    sigma_long_annual: float | None = None,
    sigma_short_annual: float | None = None,
    ofi_1m: float = 0.0,
    z_vwap: float | None = None,
    whale_signed_notional_5m: float | None = None,
    notional_5m: float | None = None,
    asset: str | None = None,
) -> ModelPrediction:
    """Kwarg-friendly wrapper around ``predict_up_prob``."""
    return predict_up_prob(
        PredictorInputs(
            spot_t=spot_t,
            spot_0=spot_0,
            seconds_remaining=seconds_remaining,
            sigma_long_annual=sigma_long_annual,
            sigma_short_annual=sigma_short_annual,
            ofi_1m=ofi_1m,
            z_vwap=z_vwap,
            whale_signed_notional_5m=whale_signed_notional_5m,
            notional_5m=notional_5m,
            asset=asset,
        )
    )
