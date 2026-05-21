"""Event-driven Expected-Move (EM) engine.

Pure-math conversion of a discrete multinomial distribution over event
outcomes (e.g. an FOMC decision space {cut-50, cut-25, hold, hike-25,
hike-50}, a CPI YoY ladder {2.3 … 3.7}, an election binary, …) into an
Expected-Move forecast on an underlying ticker (SPY, TLT, CL=F, …).

This module fetches no data. Distributions flow IN, an
:class:`EventEMForecast` flows OUT. Data wiring is the responsibility of
the upstream router (module B3). The engine offers three estimation
modes:

1. **Calibrated** — linear projection of distribution features onto
   realised |Δ%| using an :class:`EMCalibration` previously fit by
   :func:`fit_em_calibration`.
2. **Entropy proxy** — a kind-specific scaling of Shannon entropy with
   a small tail-dispersion tilt. Used when no calibration is supplied.
3. **Single outcome** — distributions degenerate to one outcome (point
   mass) return EM=0 with a ``single_outcome`` warning.

The mathematical convention for the directional ``asymmetric_mass``
feature is **hawkish mass minus dovish mass** (positive ⇒ hawkish /
bearish-bonds, negative ⇒ dovish / bullish-bonds). For CPI we use
upside-surprise mass minus downside-surprise mass relative to the
probability-weighted mean. For elections we collapse to ``|2p-1|``.

The entropy-proxy scaling constants (k_kind) are deliberately
documented inline so consumers can audit them against the literature
on event-day implied moves.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

EventKind = Literal["fomc", "cpi", "nfp", "election", "opec", "geopolitical"]

# Default tail-dispersion lower threshold (uniform 5-outcome ladders have
# ~40 % mass on the outermost 20 % of the value range, so we subtract 0.4
# to keep uniform distributions at dispersion_factor=0).
_TAIL_REFERENCE = 0.4

# Kind-specific entropy-proxy multipliers. Values are %/entropy-nat and
# anchored so that a *uniform* 5-outcome distribution produces a roughly
# literature-consistent event-day EM on the typical underlying:
#
#   FOMC + SPY        :  ~0.5 % (literature: 0.4–0.7 % straddle)
#   CPI  + SPY/TLT    :  ~0.3 % (literature: 0.25–0.5 %)
#   NFP  + SPY        :  ~0.35 %
#   Election + SPY    :  ~1.2 % (binary, tails-of-outcome distribution)
#   OPEC + CL=F       :  ~0.8 % (oil reacts harder than equity to OPEC)
#   Geopolitical+SPY  :  ~0.6 % (broad fallback for ad-hoc events)
_K_KIND: dict[str, float] = {
    "fomc": 0.50,
    "cpi": 0.30,
    "nfp": 0.35,
    "election": 1.20,
    "opec": 0.80,
    "geopolitical": 0.60,
}


# =============================================================================
# Pydantic models
# =============================================================================


class Outcome(BaseModel):
    """One leg of a multinomial event distribution."""

    model_config = ConfigDict(extra="forbid")

    label: str
    probability: float = Field(..., ge=0.0, le=1.0)
    anchor_value: float


class EventDistribution(BaseModel):
    """Discrete probability distribution over an event's outcome space."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_kind: EventKind
    underlying_ticker: str
    scheduled_at_utc: datetime
    outcomes: list[Outcome]
    spot_at_lookup: float | None = None


class EventEMForecast(BaseModel):
    """Expected-Move forecast derived from an :class:`EventDistribution`."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    em_pct: float = Field(..., ge=0.0)
    em_pct_ci_low: float | None
    em_pct_ci_high: float | None
    em_method: Literal["calibrated", "entropy_proxy", "single_outcome"]
    distribution_features: dict[str, float]
    n_outcomes: int
    warnings: list[str]


class EMCalibration(BaseModel):
    """Linear projection ``EM_pct = β·features + α`` calibrated on history."""

    model_config = ConfigDict(extra="forbid")

    event_kind: str
    underlying_ticker: str
    coefficients: dict[str, float]
    intercept: float
    r_squared: float
    sample_size: int
    sigma_residual: float


# =============================================================================
# Distribution preprocessing
# =============================================================================


def normalize_outcomes(outcomes: list[Outcome]) -> list[Outcome]:
    """Renormalize a list of outcomes so probabilities sum to 1.

    Polymarket multi-leg books often sum to slightly more or less than 1
    because of bid/ask spreads on the YES legs. We:

    1. Drop outcomes with probability < 0.001 (numerical floor — keeps
       MLE stable and avoids log(0) in entropy).
    2. Rescale the remaining mass so it sums to 1 exactly.
    3. Raise :class:`ValueError` if the *pre-normalization* total mass is
       below 0.5 — the book is too thin to trust.

    Args:
        outcomes: Raw multinomial leg probabilities.

    Returns:
        New list with rescaled probabilities. Original order preserved
        for the surviving outcomes.

    Raises:
        ValueError: total mass below 0.5 or empty input.
    """
    if not outcomes:
        raise ValueError("normalize_outcomes: empty outcome list")
    total = sum(o.probability for o in outcomes)
    if total < 0.5:
        raise ValueError(
            f"normalize_outcomes: total probability mass {total:.3f} < 0.5; book too thin"
        )
    survivors = [o for o in outcomes if o.probability >= 0.001]
    if not survivors:
        raise ValueError("normalize_outcomes: every outcome below 0.001 floor")
    survivor_total = sum(o.probability for o in survivors)
    if survivor_total <= 0.0:
        raise ValueError("normalize_outcomes: survivor mass is zero")
    return [
        Outcome(
            label=o.label, probability=o.probability / survivor_total, anchor_value=o.anchor_value
        )
        for o in survivors
    ]


# =============================================================================
# Feature extraction
# =============================================================================


def _weighted_moments(probs: np.ndarray, anchors: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, skew, excess_kurt) of a probability-weighted set.

    Skew and excess kurtosis are the standard Fisher-Pearson definitions
    using *probability-weighted* central moments. When σ is zero we
    return ``(mean, 0, 0, 0)`` to keep downstream features finite.
    """
    mean = float(np.sum(probs * anchors))
    centred = anchors - mean
    var = float(np.sum(probs * centred * centred))
    std = math.sqrt(var) if var > 0 else 0.0
    if std <= 0.0:
        return mean, 0.0, 0.0, 0.0
    m3 = float(np.sum(probs * centred**3))
    m4 = float(np.sum(probs * centred**4))
    skew = m3 / (std**3)
    kurt_excess = m4 / (std**4) - 3.0
    return mean, std, skew, kurt_excess


def _asymmetric_mass(dist: EventDistribution, probs: np.ndarray, anchors: np.ndarray) -> float:
    """Directional asymmetry feature, kind-aware.

    Convention: ``positive = hawkish/bearish-bonds direction``.

    - **fomc**: hawkish mass (anchor>0, i.e. hike) minus dovish mass
      (anchor<0, i.e. cut). Mass at 0 ignored. So a 70%-dovish book
      returns ``-0.4``.
    - **cpi / nfp / opec**: probability-weighted mean is used as the
      pivot; mass above mean minus mass below.
    - **election**: ``|2p - 1|`` where p is the maximum-anchor outcome
      probability — magnitude of binary lead.
    - **geopolitical** (and fallback): same pivot-mean rule as CPI.
    """
    kind = dist.event_kind
    if kind == "fomc":
        hawkish = float(np.sum(probs[anchors > 0.0]))
        dovish = float(np.sum(probs[anchors < 0.0]))
        return hawkish - dovish
    if kind == "election":
        # Collapse to binary lead magnitude. For a multi-candidate vector
        # this is the deviation of the modal outcome from 50 %.
        p_max = float(np.max(probs))
        return abs(2.0 * p_max - 1.0)
    # CPI / NFP / OPEC / geopolitical → upside-vs-downside vs prob-mean
    mean = float(np.sum(probs * anchors))
    above = float(np.sum(probs[anchors > mean]))
    below = float(np.sum(probs[anchors < mean]))
    return above - below


def _tail_pct(probs: np.ndarray, anchors: np.ndarray, tail_fraction: float = 0.2) -> float:
    """Probability mass on the outermost ``tail_fraction`` of the anchor range.

    For a uniform 5-outcome ladder this is 2/5 = 0.4 (the outer two
    anchors fall outside the central 60 % of the range). For a binary
    distribution it is 1.0 by construction.
    """
    if len(anchors) <= 1:
        return float(np.sum(probs))
    a_min = float(np.min(anchors))
    a_max = float(np.max(anchors))
    span = a_max - a_min
    if span <= 0.0:
        return 0.0
    cutoff_low = a_min + tail_fraction * span
    cutoff_high = a_max - tail_fraction * span
    return float(np.sum(probs[(anchors <= cutoff_low) | (anchors >= cutoff_high)]))


def distribution_features(dist: EventDistribution) -> dict[str, float]:
    """Extract scalar features from a normalized :class:`EventDistribution`.

    The returned feature set is what calibration regresses on and what
    the entropy-proxy mode reads. Probabilities are renormalized
    defensively in case the caller passed a raw book.

    Returns:
        Dict with keys ``entropy``, ``entropy_normalized``,
        ``mean``, ``dispersion``, ``skew``, ``excess_kurt``,
        ``modal_mass``, ``asymmetric_mass``, ``tail_pct``,
        ``n_outcomes``.
    """
    if not dist.outcomes:
        # Degenerate empty distribution — emit all-zero features rather
        # than raising; the EM forecaster decides how to react.
        return {
            "entropy": 0.0,
            "entropy_normalized": 0.0,
            "mean": 0.0,
            "dispersion": 0.0,
            "skew": 0.0,
            "excess_kurt": 0.0,
            "modal_mass": 0.0,
            "asymmetric_mass": 0.0,
            "tail_pct": 0.0,
            "n_outcomes": 0.0,
        }

    probs_raw = np.array([o.probability for o in dist.outcomes], dtype=float)
    anchors = np.array([o.anchor_value for o in dist.outcomes], dtype=float)
    total = float(np.sum(probs_raw))
    probs = probs_raw / total if total > 0 else probs_raw

    # Shannon entropy in nats; ignore zero-mass outcomes to avoid 0·log0.
    positive = probs[probs > 0]
    entropy = float(-np.sum(positive * np.log(positive))) if positive.size > 0 else 0.0
    n = len(probs)
    entropy_max = math.log(n) if n > 1 else 1.0
    entropy_normalized = entropy / entropy_max if entropy_max > 0 else 0.0

    mean, std, skew, excess_kurt = _weighted_moments(probs, anchors)
    modal_mass = float(np.max(probs)) if probs.size > 0 else 0.0
    asymmetric_mass = _asymmetric_mass(dist, probs, anchors)
    tail_pct = _tail_pct(probs, anchors)

    return {
        "entropy": entropy,
        "entropy_normalized": entropy_normalized,
        "mean": mean,
        "dispersion": std,
        "skew": skew,
        "excess_kurt": excess_kurt,
        "modal_mass": modal_mass,
        "asymmetric_mass": asymmetric_mass,
        "tail_pct": tail_pct,
        "n_outcomes": float(n),
    }


# =============================================================================
# EM forecast — three modes
# =============================================================================


def _entropy_proxy_em(features: dict[str, float], k: float) -> float:
    """Compute EM% under the entropy-proxy heuristic.

    ``EM_pct = k * entropy_normalized * (1 + dispersion_factor)`` where
    ``dispersion_factor = max(0, tail_pct - 0.4)`` so uniform 5-outcome
    ladders yield ``dispersion_factor = 0`` and the leading term
    dominates.
    """
    entropy_normalized = float(features.get("entropy_normalized", 0.0))
    tail_pct = float(features.get("tail_pct", 0.0))
    dispersion_factor = max(0.0, tail_pct - _TAIL_REFERENCE)
    return float(k * entropy_normalized * (1.0 + dispersion_factor))


def expected_move_from_distribution(
    dist: EventDistribution,
    calibration: EMCalibration | None = None,
) -> EventEMForecast:
    """Convert a distribution to an EM forecast.

    Args:
        dist: Renormalized distribution. (We renormalize defensively in
            case the caller skipped :func:`normalize_outcomes`.)
        calibration: Optional historical fit. When supplied (and the
            kind/ticker matches; we do not enforce — caller's
            responsibility) we project features through the fitted
            linear model.

    Returns:
        :class:`EventEMForecast`. ``em_method`` discriminates the path
        taken. CIs are populated under the calibrated mode using the
        fitted residual σ at the 95 % normal level.
    """
    warnings: list[str] = []
    n = len(dist.outcomes)

    # Degenerate path — single outcome (or zero) — point-mass implies no
    # event-driven move under our model.
    if n <= 1:
        features = distribution_features(dist)
        warnings.append("single_outcome")
        return EventEMForecast(
            event_id=dist.event_id,
            em_pct=0.0,
            em_pct_ci_low=0.0,
            em_pct_ci_high=0.0,
            em_method="single_outcome",
            distribution_features=features,
            n_outcomes=n,
            warnings=warnings,
        )

    # Defensive renormalization. If the book is too thin we degrade to
    # entropy-proxy on raw probs but emit a warning.
    try:
        normalized = normalize_outcomes(dist.outcomes)
        dist_eff = dist.model_copy(update={"outcomes": normalized})
    except ValueError as exc:
        warnings.append(f"normalize_failed:{exc}")
        dist_eff = dist

    features = distribution_features(dist_eff)

    if calibration is not None:
        em_pct = float(calibration.intercept)
        for name, beta in calibration.coefficients.items():
            em_pct += float(beta) * float(features.get(name, 0.0))
        # Floor at 0 — EM is a magnitude.
        em_pct = max(em_pct, 0.0)
        if calibration.r_squared < 0.3:
            warnings.append("low_calibration_r2")
        # 95 % normal CI via residual σ.
        sigma = max(float(calibration.sigma_residual), 0.0)
        ci_low = max(em_pct - 1.96 * sigma, 0.0)
        ci_high = em_pct + 1.96 * sigma
        return EventEMForecast(
            event_id=dist.event_id,
            em_pct=em_pct,
            em_pct_ci_low=ci_low,
            em_pct_ci_high=ci_high,
            em_method="calibrated",
            distribution_features=features,
            n_outcomes=n,
            warnings=warnings,
        )

    # Entropy-proxy fallback.
    k = _K_KIND.get(dist_eff.event_kind, _K_KIND["geopolitical"])
    em_pct = _entropy_proxy_em(features, k)
    warnings.append("uncalibrated_entropy_proxy")
    return EventEMForecast(
        event_id=dist.event_id,
        em_pct=em_pct,
        em_pct_ci_low=None,
        em_pct_ci_high=None,
        em_method="entropy_proxy",
        distribution_features=features,
        n_outcomes=n,
        warnings=warnings,
    )


# =============================================================================
# Calibration — fit EM_pct = β·features + α from history
# =============================================================================


# Feature names entering the calibration regression. We deliberately
# exclude ``modal_mass`` here because it is almost a deterministic
# function of ``entropy_normalized`` (r ≈ −0.93 empirically on synthetic
# Dirichlet draws); including both makes the OLS unidentifiable and
# distributes the entropy loading across the colinear pair. ``modal_mass``
# is still exposed in ``distribution_features`` for downstream consumers,
# and consumers can override ``coefficients`` at construction time to
# include it if they have enough data to fit it stably (n>>5).
_CALIBRATION_FEATURES: tuple[str, ...] = (
    "entropy_normalized",
    "dispersion",
    "asymmetric_mass",
    "tail_pct",
)


def fit_em_calibration(
    historical_events: list[tuple[EventDistribution, float]],
    event_kind: str,
    underlying_ticker: str,
) -> EMCalibration:
    """Fit a linear regression of realized |Δ%| on distribution features.

    Uses the standard ``np.linalg.lstsq`` solver. Features are
    standardized internally for numerical conditioning; the resulting
    standardized betas are then back-transformed to unstandardized
    coefficients on the original feature scale, so consumers can plug
    them straight into :func:`expected_move_from_distribution` without
    re-standardizing.

    Args:
        historical_events: List of
            ``(distribution_at_event, realized_abs_pct_change)``
            tuples. The realized move is in percent (e.g. ``0.55`` for a
            55 bp move on SPY).
        event_kind: Tag for the resulting calibration; not used in the
            fit itself but stored on the result.
        underlying_ticker: Same — stored only.

    Returns:
        :class:`EMCalibration` carrying unstandardized coefficients,
        intercept, R², residual σ, and sample size.

    Raises:
        ValueError: fewer than 5 usable (non-NaN) observations.
    """
    if len(historical_events) < 5:
        raise ValueError(
            f"fit_em_calibration: need >=5 historical events, got {len(historical_events)}"
        )

    feature_rows: list[list[float]] = []
    y_values: list[float] = []
    for dist, realized in historical_events:
        feats = distribution_features(dist)
        row = [float(feats.get(name, 0.0)) for name in _CALIBRATION_FEATURES]
        if not all(np.isfinite(row)) or not np.isfinite(realized):
            continue
        feature_rows.append(row)
        y_values.append(float(realized))

    if len(y_values) < 5:
        raise ValueError(
            f"fit_em_calibration: only {len(y_values)} finite observations after NaN drop (need 5)"
        )

    x_raw = np.array(feature_rows, dtype=float)
    y = np.array(y_values, dtype=float)
    n_obs, n_feat = x_raw.shape

    # Standardize each column. Zero-variance columns are dropped from the
    # standardized fit (their unstandardized coefficient becomes 0).
    means = x_raw.mean(axis=0)
    stds = x_raw.std(axis=0, ddof=0)
    safe_stds = np.where(stds > 1e-12, stds, 1.0)
    x_std = (x_raw - means) / safe_stds

    # Design matrix with intercept column.
    design = np.column_stack([np.ones(n_obs), x_std])
    try:
        beta_std, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"fit_em_calibration: lstsq failed ({exc})") from exc

    intercept_std = float(beta_std[0])
    slopes_std = beta_std[1:]

    # Back-transform standardized betas to the original feature scale:
    # β_unstd = β_std / σ_x ;  α_unstd = α_std − Σ β_unstd · μ_x.
    coeffs: dict[str, float] = {}
    intercept_unstd = intercept_std
    for i, name in enumerate(_CALIBRATION_FEATURES):
        if stds[i] > 1e-12:
            b = float(slopes_std[i] / stds[i])
        else:
            b = 0.0
        coeffs[name] = b
        intercept_unstd -= b * float(means[i])

    # Diagnostics: in-sample R² and residual σ on the original scale.
    y_hat = design @ beta_std
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = max(n_obs - (n_feat + 1), 1)
    sigma_residual = math.sqrt(ss_res / dof) if ss_res > 0 else 0.0

    return EMCalibration(
        event_kind=event_kind,
        underlying_ticker=underlying_ticker,
        coefficients=coeffs,
        intercept=float(intercept_unstd),
        r_squared=float(r_squared),
        sample_size=int(n_obs),
        sigma_residual=float(sigma_residual),
    )


# =============================================================================
# Bootstrap CI on the calibrated EM
# =============================================================================


def bootstrap_em_ci(
    dist: EventDistribution,
    calibration: EMCalibration,
    n_iter: int = 500,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Dirichlet-resampling bootstrap CI for the calibrated EM.

    We treat the observed probability vector as a Dirichlet centre,
    resample ``n_iter`` synthetic books with concentration
    ``α = 50 · p + 1`` (50 effective pseudo-counts → mild perturbation),
    recompute features and EM under the calibration, and return the
    empirical lower/upper quantile bounds at the chosen confidence.

    Args:
        dist: Distribution to perturb. Must have ≥2 outcomes.
        calibration: Fitted EM calibration.
        n_iter: Bootstrap iterations. 500 is enough for stable 2-sigma
            quantiles.
        confidence: Two-sided coverage (default 95 %).

    Returns:
        ``(em_low, em_high)`` — empirical quantile bounds. Width may be
        zero when the distribution itself is degenerate.

    Raises:
        ValueError: invalid ``confidence`` or too few outcomes.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"bootstrap_em_ci: confidence must be in (0,1), got {confidence}")
    if len(dist.outcomes) < 2:
        raise ValueError("bootstrap_em_ci: need >=2 outcomes")

    rng = np.random.default_rng(seed=0)
    probs = np.array([o.probability for o in dist.outcomes], dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    alpha = 50.0 * probs + 1.0

    em_samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sampled = rng.dirichlet(alpha)
        new_outcomes = [
            Outcome(label=o.label, probability=float(p), anchor_value=o.anchor_value)
            for o, p in zip(dist.outcomes, sampled, strict=False)
        ]
        synthetic = dist.model_copy(update={"outcomes": new_outcomes})
        forecast = expected_move_from_distribution(synthetic, calibration=calibration)
        em_samples[i] = float(forecast.em_pct)

    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    em_low = float(np.quantile(em_samples, lower_q))
    em_high = float(np.quantile(em_samples, upper_q))
    return em_low, em_high


__all__ = [
    "EMCalibration",
    "EventDistribution",
    "EventEMForecast",
    "Outcome",
    "bootstrap_em_ci",
    "distribution_features",
    "expected_move_from_distribution",
    "fit_em_calibration",
    "normalize_outcomes",
]
