"""Tests for ``pfm.vol.event_vol_engine`` — synthetic-DGP first."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from pfm.vol.event_vol_engine import (
    EMCalibration,
    EventDistribution,
    Outcome,
    bootstrap_em_ci,
    distribution_features,
    expected_move_from_distribution,
    fit_em_calibration,
    normalize_outcomes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_T0 = datetime(2026, 6, 17, 18, 0, tzinfo=UTC)


def _fomc_uniform_dist(event_id: str = "fomc-uniform") -> EventDistribution:
    """Uniform FOMC distribution over 5 outcomes (cut-50 … hike-50)."""
    outcomes = [
        Outcome(label=lbl, probability=0.20, anchor_value=anchor)
        for lbl, anchor in zip(
            ["cut_50", "cut_25", "hold", "hike_25", "hike_50"],
            [-0.50, -0.25, 0.0, 0.25, 0.50],
            strict=False,
        )
    ]
    return EventDistribution(
        event_id=event_id,
        event_kind="fomc",
        underlying_ticker="SPY",
        scheduled_at_utc=_T0,
        outcomes=outcomes,
    )


def _fomc_dovish_dist() -> EventDistribution:
    """FOMC distribution with 70 % dovish (cut) mass, 30 % hawkish (hike)."""
    outcomes = [
        Outcome(label="cut_50", probability=0.30, anchor_value=-0.50),
        Outcome(label="cut_25", probability=0.40, anchor_value=-0.25),
        Outcome(label="hike_25", probability=0.20, anchor_value=0.25),
        Outcome(label="hike_50", probability=0.10, anchor_value=0.50),
    ]
    return EventDistribution(
        event_id="fomc-dovish",
        event_kind="fomc",
        underlying_ticker="SPY",
        scheduled_at_utc=_T0,
        outcomes=outcomes,
    )


# ---------------------------------------------------------------------------
# 1 & 2 — normalize_outcomes
# ---------------------------------------------------------------------------


def test_normalize_outcomes_rescales_to_unit() -> None:
    raw = [
        Outcome(label="a", probability=0.50, anchor_value=-0.25),
        Outcome(label="b", probability=0.55, anchor_value=0.25),
        Outcome(label="floor", probability=1e-4, anchor_value=0.0),
    ]
    out = normalize_outcomes(raw)
    assert len(out) == 2, "floor outcome should be dropped"
    total = sum(o.probability for o in out)
    assert math.isclose(total, 1.0, rel_tol=1e-9)
    # Surviving labels preserve order.
    assert [o.label for o in out] == ["a", "b"]


def test_normalize_outcomes_raises_on_thin_book() -> None:
    raw = [
        Outcome(label="a", probability=0.20, anchor_value=-0.25),
        Outcome(label="b", probability=0.20, anchor_value=0.25),
    ]
    with pytest.raises(ValueError, match="too thin"):
        normalize_outcomes(raw)


# ---------------------------------------------------------------------------
# 3, 4, 5 — distribution_features
# ---------------------------------------------------------------------------


def test_distribution_features_entropy_on_uniform_is_max() -> None:
    dist = _fomc_uniform_dist()
    feats = distribution_features(dist)
    assert math.isclose(feats["entropy"], math.log(5), abs_tol=1e-9)
    assert math.isclose(feats["entropy_normalized"], 1.0, abs_tol=1e-9)


def test_distribution_features_entropy_on_consensus_is_zero() -> None:
    outcomes = [
        Outcome(label="cut_50", probability=0.0025, anchor_value=-0.50),
        Outcome(label="cut_25", probability=0.0025, anchor_value=-0.25),
        Outcome(label="hold", probability=0.99, anchor_value=0.0),
        Outcome(label="hike_25", probability=0.0025, anchor_value=0.25),
        Outcome(label="hike_50", probability=0.0025, anchor_value=0.50),
    ]
    dist = EventDistribution(
        event_id="fomc-consensus",
        event_kind="fomc",
        underlying_ticker="SPY",
        scheduled_at_utc=_T0,
        outcomes=outcomes,
    )
    feats = distribution_features(dist)
    # 0.99·log(0.99) + 4·0.0025·log(0.0025) ≈ 0.06 nats — close to zero.
    assert feats["entropy"] < 0.10, f"expected near-zero entropy, got {feats['entropy']}"
    assert feats["modal_mass"] >= 0.98


def test_distribution_features_asymmetric_mass_fomc() -> None:
    dist = _fomc_dovish_dist()
    feats = distribution_features(dist)
    # Hawkish (0.30) − dovish (0.70) = −0.40 by convention.
    assert math.isclose(feats["asymmetric_mass"], -0.40, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 6, 7, 11 — expected_move_from_distribution
# ---------------------------------------------------------------------------


def test_expected_move_entropy_proxy_uniform_fomc_gives_meaningful_em() -> None:
    dist = _fomc_uniform_dist()
    forecast = expected_move_from_distribution(dist)
    assert forecast.em_method == "entropy_proxy"
    # k_fomc = 0.50, entropy_normalized = 1.0, tail_pct = 0.4 (uniform 5
    # outcomes → 2/5 in tails), dispersion_factor = max(0, 0.4-0.4) = 0.
    # So EM ≈ 0.50 % exactly.
    assert math.isclose(forecast.em_pct, 0.50, abs_tol=0.05)
    assert forecast.em_pct_ci_low is None
    assert forecast.em_pct_ci_high is None
    assert forecast.n_outcomes == 5


def test_expected_move_single_outcome_returns_zero() -> None:
    outcomes = [Outcome(label="only", probability=1.0, anchor_value=0.0)]
    dist = EventDistribution(
        event_id="degenerate",
        event_kind="fomc",
        underlying_ticker="SPY",
        scheduled_at_utc=_T0,
        outcomes=outcomes,
    )
    forecast = expected_move_from_distribution(dist)
    assert forecast.em_method == "single_outcome"
    assert forecast.em_pct == 0.0
    assert "single_outcome" in forecast.warnings


def test_expected_move_calibrated_uses_coefficients() -> None:
    dist = _fomc_uniform_dist()
    # Construct a distribution whose entropy_normalized=0.5 by mixing a
    # spike on one outcome with mass on a second.
    spiked = EventDistribution(
        event_id="spiked",
        event_kind="fomc",
        underlying_ticker="SPY",
        scheduled_at_utc=_T0,
        outcomes=[
            Outcome(label="hold", probability=0.89, anchor_value=0.0),
            Outcome(label="hike_25", probability=0.11, anchor_value=0.25),
        ],
    )
    feats = distribution_features(spiked)
    # Confirm we are roughly at entropy_normalized=0.5 before we test the
    # calibrated EM uses it.
    assert 0.4 < feats["entropy_normalized"] < 0.6

    calibration = EMCalibration(
        event_kind="fomc",
        underlying_ticker="SPY",
        coefficients={"entropy_normalized": 1.0},
        intercept=0.0,
        r_squared=0.9,
        sample_size=20,
        sigma_residual=0.10,
    )
    forecast = expected_move_from_distribution(spiked, calibration=calibration)
    assert forecast.em_method == "calibrated"
    # EM = 1.0 * entropy_normalized + 0  ≈ 0.5
    assert math.isclose(forecast.em_pct, feats["entropy_normalized"], abs_tol=1e-9)
    assert forecast.em_pct_ci_low is not None
    assert forecast.em_pct_ci_high is not None
    assert forecast.em_pct_ci_low <= forecast.em_pct <= forecast.em_pct_ci_high

    # Sanity check with a uniform distribution → entropy_normalized=1
    # → EM ≈ 1.0 under this β.
    uniform_forecast = expected_move_from_distribution(dist, calibration=calibration)
    assert math.isclose(uniform_forecast.em_pct, 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 8, 9 — fit_em_calibration
# ---------------------------------------------------------------------------


def _synthetic_event(
    rng: np.random.Generator,
    target_entropy: float,
    anchor_span: float,
    n_outcomes: int = 5,
) -> EventDistribution:
    """Build an FOMC-shaped event with controlled entropy + anchor span.

    The probabilities are a Dirichlet draw with concentration scaled so
    the entropy lands in the requested neighbourhood. ``anchor_span``
    independently scales the anchor range, decoupling entropy from
    dispersion so the OLS fit can identify both coefficients. Varying
    ``n_outcomes`` decorrelates modal_mass from entropy.
    """
    max_entropy = math.log(n_outcomes) if n_outcomes > 1 else 1.0
    concentration = max(0.05, 5.0 * (1.0 - target_entropy / max_entropy))
    probs = rng.dirichlet(np.full(n_outcomes, concentration))
    half = anchor_span / 2.0
    # Equally spaced anchors symmetric about 0.
    anchors = np.linspace(-half, half, n_outcomes)
    outcomes = [
        Outcome(label=f"out_{i}", probability=float(p), anchor_value=float(a))
        for i, (p, a) in enumerate(zip(probs, anchors, strict=True))
    ]
    return EventDistribution(
        event_id=f"synthetic-{rng.integers(0, 1_000_000)}",
        event_kind="fomc",
        underlying_ticker="SPY",
        scheduled_at_utc=_T0,
        outcomes=outcomes,
    )


def test_fit_em_calibration_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(seed=42)
    historical: list[tuple[EventDistribution, float]] = []
    for _ in range(50):
        target_ent = float(rng.uniform(0.05, math.log(5) * 0.95))
        # Wide anchor-span range fully decouples dispersion from entropy.
        anchor_span = float(rng.uniform(0.2, 8.0))
        dist = _synthetic_event(
            rng,
            target_entropy=target_ent,
            anchor_span=anchor_span,
            n_outcomes=5,
        )
        feats = distribution_features(dist)
        realized = 0.3 * feats["entropy"] + 0.1 * feats["dispersion"] + rng.normal(0.0, 0.05)
        historical.append((dist, float(realized)))

    calibration = fit_em_calibration(historical, event_kind="fomc", underlying_ticker="SPY")
    # All events have n=5, so entropy = entropy_normalized * log(5).
    beta_entropy_eff = calibration.coefficients["entropy_normalized"] / math.log(5)
    beta_dispersion = calibration.coefficients["dispersion"]

    assert calibration.r_squared > 0.7, f"R²={calibration.r_squared:.3f} below threshold"
    assert 0.225 <= beta_entropy_eff <= 0.375, (
        f"β_entropy_eff={beta_entropy_eff:.3f} not within ±25 % of 0.3"
    )
    assert 0.05 <= beta_dispersion <= 0.15, (
        f"β_dispersion={beta_dispersion:.3f} not within ±50 % of 0.1"
    )
    assert calibration.sample_size == 50
    assert calibration.sigma_residual > 0.0


def test_fit_em_calibration_raises_on_small_sample() -> None:
    rng = np.random.default_rng(seed=1)
    history = [
        (_synthetic_event(rng, target_entropy=1.0, anchor_span=1.0), 0.4),
        (_synthetic_event(rng, target_entropy=0.5, anchor_span=2.0), 0.2),
        (_synthetic_event(rng, target_entropy=0.8, anchor_span=1.5), 0.3),
    ]
    with pytest.raises(ValueError, match=">=5"):
        fit_em_calibration(history, event_kind="fomc", underlying_ticker="SPY")


# ---------------------------------------------------------------------------
# 10 — bootstrap_em_ci
# ---------------------------------------------------------------------------


def test_bootstrap_em_ci_brackets_point_estimate() -> None:
    dist = _fomc_uniform_dist()
    calibration = EMCalibration(
        event_kind="fomc",
        underlying_ticker="SPY",
        coefficients={"entropy_normalized": 1.0, "dispersion": 0.5},
        intercept=0.10,
        r_squared=0.8,
        sample_size=30,
        sigma_residual=0.15,
    )
    point = expected_move_from_distribution(dist, calibration=calibration).em_pct
    low, high = bootstrap_em_ci(dist, calibration, n_iter=300)
    assert low <= point <= high, f"CI [{low}, {high}] does not bracket point {point}"
    assert high - low > 0.0, "CI width must be positive for a non-degenerate book"
