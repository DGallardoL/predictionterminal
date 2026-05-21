"""Monte-Carlo calibration tests for ``pfm.crypto5min.predictor``.

These tests simulate spot paths under the **same** GBM the closed-form
predictor assumes and verify that the model's ``prob_up`` is within a
small absolute tolerance of the empirical Up-rate. This is the standard
synthetic-DGP recovery check — if a future refactor breaks the math the
calibration error will spike.

A couple of subtleties worth noting:

* The predictor enforces a *per-horizon* adaptive σ floor
  (``SIGMA_FLOOR_BY_SECONDS``) which boosts σ for short windows where
  the daily-σ would otherwise under-estimate intraday vol. We pick test
  parameters where the supplied σ is **already above** the floor so the
  closed-form uses our σ verbatim. Otherwise the model would use a
  larger σ than the Monte-Carlo path and calibration would diverge.
* The predictor returns ``P(spot_T > spot_0_arg)``. To test against an
  arbitrary strike ``K`` we pass ``spot_0_arg = K`` and ``spot_t = S``
  (the actual current spot). Monte-Carlo then simulates
  ``S_T = S * exp(-0.5 σ² τ + σ √τ Z)`` and we compare to ``K``.
* The clipping at ``[0.005, 0.995]`` inside ``predict_up_prob`` means the
  T=0 and extreme-drift edge tests use that value as the saturation
  bound (the model never returns exactly 0 or 1 except in the
  ``seconds_remaining == 0`` branch).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfm.crypto5min.predictor import (
    MU_CAP,
    MU_OFI_SCALE,
    SECONDS_PER_YEAR,
    Z_REV_THRESHOLD,
    predict_for_window,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SAMPLES: int = 5_000


def _simulate_terminal_spots(
    spot_now: float,
    sigma_annual: float,
    drift_annual: float,
    seconds: float,
    rng: np.random.Generator,
    n: int = N_SAMPLES,
) -> np.ndarray:
    """Vectorised single-step GBM. Returns terminal spots ``S_T``.

    ``S_T = S * exp((μ - 0.5 σ²) τ + σ √τ Z)`` with ``Z ~ N(0,1)``.
    """
    tau = seconds / SECONDS_PER_YEAR
    z = rng.standard_normal(n)
    log_ret = (drift_annual - 0.5 * sigma_annual**2) * tau + sigma_annual * math.sqrt(tau) * z
    return spot_now * np.exp(log_ret)


def _empirical_prob_above(spots: np.ndarray, strike: float) -> float:
    return float(np.mean(spots > strike))


# ---------------------------------------------------------------------------
# 1. Upside strike — model vs Monte-Carlo
# ---------------------------------------------------------------------------


def test_monte_carlo_calibration_upside_strike() -> None:
    """P(spot_T > 1.01 * spot_0) with σ=1.0/yr, T=5min, drift=0.

    σ=1.0/yr clears the 5min adaptive floor (0.90) so the model uses our
    σ as-is and the empirical Up-rate should match within 3% absolute.
    """
    spot_now = 60_000.0
    strike = spot_now * 1.01
    sigma = 1.0
    seconds = 300.0

    model = predict_for_window(
        spot_t=spot_now,
        spot_0=strike,
        seconds_remaining=seconds,
        sigma_long_annual=sigma,
    )
    rng = np.random.default_rng(42)
    spots = _simulate_terminal_spots(spot_now, sigma, 0.0, seconds, rng)
    empirical = _empirical_prob_above(spots, strike)

    assert model.sigma_used_annual == pytest.approx(sigma, abs=1e-9)
    assert abs(model.prob_up - empirical) < 0.03, (
        f"upside calibration off: model={model.prob_up:.4f} mc={empirical:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Downside strike — model vs Monte-Carlo
# ---------------------------------------------------------------------------


def test_monte_carlo_calibration_downside_strike() -> None:
    """P(spot_T > 0.99 * spot_0) — downside strike, should be well above 0.5."""
    spot_now = 60_000.0
    strike = spot_now * 0.99
    sigma = 1.0
    seconds = 300.0

    model = predict_for_window(
        spot_t=spot_now,
        spot_0=strike,
        seconds_remaining=seconds,
        sigma_long_annual=sigma,
    )
    rng = np.random.default_rng(42)
    spots = _simulate_terminal_spots(spot_now, sigma, 0.0, seconds, rng)
    empirical = _empirical_prob_above(spots, strike)

    # Downside strike — empirical should be > 0.5 (drift-free GBM tilts
    # slightly Down due to the -0.5 σ² term, but for 5min × σ=1.0 the
    # tilt is tiny so prob should still be solidly > 0.5).
    assert empirical > 0.5
    assert abs(model.prob_up - empirical) < 0.03, (
        f"downside calibration off: model={model.prob_up:.4f} mc={empirical:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. High-vol regime
# ---------------------------------------------------------------------------


def test_monte_carlo_calibration_high_vol() -> None:
    """σ=2.0/yr (well above any horizon floor), T=5min, ATM strike."""
    spot_now = 60_000.0
    strike = spot_now  # ATM
    sigma = 2.0
    seconds = 300.0

    model = predict_for_window(
        spot_t=spot_now,
        spot_0=strike,
        seconds_remaining=seconds,
        sigma_long_annual=sigma,
    )
    rng = np.random.default_rng(42)
    spots = _simulate_terminal_spots(spot_now, sigma, 0.0, seconds, rng)
    empirical = _empirical_prob_above(spots, strike)

    assert model.sigma_used_annual == pytest.approx(sigma, abs=1e-9)
    # ATM with drift=0 should sit near 0.5 (slightly below due to -½σ²τ).
    assert 0.40 < empirical < 0.55
    assert abs(model.prob_up - empirical) < 0.03


# ---------------------------------------------------------------------------
# 4. T == 0 edge case — deterministic step
# ---------------------------------------------------------------------------


def test_zero_time_is_deterministic_step() -> None:
    """At ``seconds_remaining == 0`` the predictor must return exactly 1.0
    when ``spot_t >= spot_0`` and exactly 0.0 otherwise (Polymarket's
    up-resolution convention: ties count as Up)."""
    out_above = predict_for_window(
        spot_t=60_001.0,
        spot_0=60_000.0,
        seconds_remaining=0.0,
        sigma_long_annual=1.0,
    )
    out_equal = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=0.0,
        sigma_long_annual=1.0,
    )
    out_below = predict_for_window(
        spot_t=59_999.0,
        spot_0=60_000.0,
        seconds_remaining=0.0,
        sigma_long_annual=1.0,
    )
    assert out_above.prob_up == 1.0
    assert out_equal.prob_up == 1.0  # tie -> Up
    assert out_below.prob_up == 0.0


# ---------------------------------------------------------------------------
# 5. ATM with tiny σ — prob hovers around 0.5 (subject to the clip floor)
# ---------------------------------------------------------------------------


def test_atm_with_small_sigma_is_near_half() -> None:
    """spot_t == spot_0 (log_ratio = 0) with the smallest σ allowed (=
    SIGMA_FLOOR 0.10) should land within an epsilon of 0.5.

    The closed-form gives ``Φ(-0.5 σ √τ)`` which for small σ √τ is ≈ 0.5
    with a tiny negative tilt. We allow a generous epsilon because the
    adaptive σ floor will push σ above the user-supplied value at short
    horizons.
    """
    pred = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=3600.0,
        sigma_long_annual=0.10,
    )
    # σ=0.10 at 1h-horizon clears the per-horizon floor (0.50 for <3600s
    # band; at exactly 3600s falls through to SIGMA_FLOOR=0.10), so the
    # closed-form runs with a small σ. Δ from 0.5 should be tiny.
    assert abs(pred.prob_up - 0.5) < 0.01


# ---------------------------------------------------------------------------
# 6. Drift dominates — large positive drift pushes prob_up toward 1
# ---------------------------------------------------------------------------


def test_large_positive_drift_pushes_prob_to_one() -> None:
    """Maxed-out positive drift must lift prob_up materially above the
    flat baseline. Direction and magnitude are what we care about — the
    closed-form clips at ``[0.005, 0.995]`` so we never see exact 1.0.

    With σ floored at ``SIGMA_FLOOR=0.10`` and μ at ``MU_CAP=0.45`` we
    need a horizon long enough for the drift to dominate the diffusion.
    """
    horizon = 7 * 86_400.0  # 7 days — drift-dominated regime at MU_CAP
    pred_pos = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=horizon,
        sigma_long_annual=0.10,
        ofi_1m=1.0,
        whale_signed_notional_5m=1_000_000.0,
        notional_5m=1_000_000.0,
    )
    pred_flat = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=horizon,
        sigma_long_annual=0.10,
        ofi_1m=0.0,
    )
    # Maxed-out positive drift must lift the prob materially above the
    # flat baseline (which sits near 0.5) and approach the clip ceiling.
    assert pred_pos.prob_up > pred_flat.prob_up + 0.10
    assert pred_pos.prob_up > 0.65
    # Symmetrically the negative case should depress prob_up.
    pred_neg = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=horizon,
        sigma_long_annual=0.10,
        ofi_1m=-1.0,
        whale_signed_notional_5m=-1_000_000.0,
        notional_5m=1_000_000.0,
    )
    assert pred_neg.prob_up < pred_flat.prob_up - 0.10
    assert pred_neg.prob_up < 0.35


# ---------------------------------------------------------------------------
# 7. Drift cap respected (|μ_eff| ≤ MU_CAP)
# ---------------------------------------------------------------------------


def test_drift_cap_respected() -> None:
    """Pushing OFI + whale_flow to saturation must not exceed ``MU_CAP``.

    CLAUDE.md states the OFI drift caps at ±30%/yr (``MU_OFI_SCALE``);
    the predictor additionally clamps the combined OFI+whale drift
    inside the wider ``[-MU_CAP, MU_CAP]`` band (``MU_CAP=0.45``).
    """
    pred = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=1.0,
        ofi_1m=1.0,
        whale_signed_notional_5m=10_000_000.0,
        notional_5m=10_000_000.0,
    )
    assert abs(pred.mu_used_annual) <= MU_CAP + 1e-9
    # And OFI-only drift must respect the tighter ±MU_OFI_SCALE band.
    pred_ofi_only = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=1.0,
        ofi_1m=1.0,
    )
    assert abs(pred_ofi_only.mu_used_annual) <= MU_OFI_SCALE + 1e-9


# ---------------------------------------------------------------------------
# 8. OFI shrinkage when |z_vwap| > 2
# ---------------------------------------------------------------------------


def test_ofi_shrinkage_on_extreme_z_vwap() -> None:
    """|z_vwap| > Z_REV_THRESHOLD shrinks OFI drift toward 0 and adds a
    small opposite-direction pull. At |z|=4 the OFI drift is fully zeroed
    and replaced by the reversion pull (sign-flipped against z).
    """
    base = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=1.0,
        ofi_1m=1.0,
    )
    # |z|=3 between the two reversion thresholds — drift should shrink
    # toward 0 (in magnitude) but not flip sign yet.
    shrunk = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=1.0,
        ofi_1m=1.0,
        z_vwap=3.0,
    )
    # |z|=5 (above Z_REV_SHRINK_AT=4) — OFI drift fully zeroed and a
    # reversion pull in the opposite direction kicks in.
    flipped = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=1.0,
        ofi_1m=1.0,
        z_vwap=5.0,
    )

    assert base.mu_used_annual > 0  # positive OFI drift baseline
    # |z|=3 shrinks the positive drift toward zero.
    assert 0.0 <= shrunk.mu_used_annual < base.mu_used_annual
    # |z|=5 flips effective drift to negative (reversion pull dominates).
    assert flipped.mu_used_annual < 0
    # Sanity: |z|>Z_REV_THRESHOLD must reduce prob_up vs the unshrunk
    # baseline because we started with a positive OFI bias.
    assert flipped.prob_up < base.prob_up
    # z_vwap below threshold should leave drift unchanged.
    untouched = predict_for_window(
        spot_t=60_000.0,
        spot_0=60_000.0,
        seconds_remaining=300.0,
        sigma_long_annual=1.0,
        ofi_1m=1.0,
        z_vwap=Z_REV_THRESHOLD - 0.5,
    )
    assert untouched.mu_used_annual == pytest.approx(base.mu_used_annual, abs=1e-9)


# ---------------------------------------------------------------------------
# 9. Parameterised Monte-Carlo grid — catches symmetric mis-specifications
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strike_ratio,sigma,seconds_remaining",
    [
        (1.00, 1.20, 600.0),  # ATM, 10min
        (1.005, 1.50, 900.0),  # slight upside, 15min
        (0.995, 1.50, 900.0),  # slight downside, 15min
        (1.00, 2.50, 1200.0),  # ATM, high σ, 20min
    ],
)
def test_parametric_monte_carlo_calibration(
    strike_ratio: float,
    sigma: float,
    seconds_remaining: float,
) -> None:
    spot_now = 60_000.0
    strike = spot_now * strike_ratio
    model = predict_for_window(
        spot_t=spot_now,
        spot_0=strike,
        seconds_remaining=seconds_remaining,
        sigma_long_annual=sigma,
    )
    rng = np.random.default_rng(42)
    spots = _simulate_terminal_spots(spot_now, sigma, 0.0, seconds_remaining, rng)
    empirical = _empirical_prob_above(spots, strike)
    assert abs(model.prob_up - empirical) < 0.03, (
        f"calibration off: ratio={strike_ratio} σ={sigma} T={seconds_remaining}s "
        f"model={model.prob_up:.4f} mc={empirical:.4f}"
    )
