"""Fast, seeded tests for the independent GBM barrier-formula oracle.

These exercise the closed forms in ``scripts/validate_implied_pdf_formulas.py``
(the independent oracle for ``pfm/vol/implied_pdf.py``) against a small,
seeded Monte-Carlo simulation, plus a pure-math fit-recovery check.

Conventions match the script: ``S_t = S0 * exp(sigma*W_t + nu*t)`` with log-drift
``nu = r - q - 0.5*sigma**2``; all quantities are probabilities (no discounting).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Import the oracle's reference implementation directly from the script so the
# test certifies the *actual* code, not a re-implementation.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from validate_implied_pdf_formulas import (
    fit_sigma_nu,
    max_touch_probability,
    reflection_term,
    running_max_survival,
    simulate_gbm,
    terminal_exceedance,
)

SEED = 20260519
N_PATHS = 20_000
N_STEPS = 250


def _mc(s0, sigma, nu, t, rng):
    """Helper: simulate and return (terminal, running_max)."""
    return simulate_gbm(s0, sigma, nu, t, N_PATHS, int(N_STEPS * max(1.0, t)), rng)


# --------------------------------------------------------------------------- #
# Item 1: running-max survival matches MC for several K and (sigma, nu) incl nu!=0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("sigma", "nu", "t", "s0"),
    [
        (0.20, 0.0, 1.0, 100.0),
        (0.30, 0.05, 1.0, 100.0),
        (0.25, -0.04, 0.5, 50.0),
    ],
)
def test_running_max_survival_matches_mc(sigma, nu, t, s0):
    rng = np.random.default_rng(SEED)
    steps = int(N_STEPS * max(1.0, t))
    for mult in (1.00, 1.05, 1.15, 1.30):
        k = s0 * mult
        # Brownian-bridge-corrected estimate (unbiased for the continuous max).
        mc = max_touch_probability(k, s0, sigma, nu, t, N_PATHS, steps, rng)
        cf = running_max_survival(k, s0, sigma, nu, t)
        assert abs(mc - cf) < 0.02, f"K={k}: MC={mc:.4f} CF={cf:.4f}"


# --------------------------------------------------------------------------- #
# Item 2: driftless factor-of-2  P(M_T>=K) == 2*P(S_T>=K)
# --------------------------------------------------------------------------- #


def test_driftless_factor_of_two_closed_form():
    # Pure-math identity (exact, no MC).
    sigma, nu, t, s0 = 0.20, 0.0, 1.0, 100.0
    for k in (101.0, 110.0, 130.0, 160.0):
        cf_max = running_max_survival(k, s0, sigma, nu, t)
        cf_term = terminal_exceedance(k, s0, sigma, nu, t)
        assert abs(cf_max - 2.0 * cf_term) < 1e-12


def test_driftless_factor_of_two_against_mc():
    rng = np.random.default_rng(SEED + 1)
    sigma, nu, t, s0 = 0.20, 0.0, 1.0, 100.0
    steps = int(N_STEPS * max(1.0, t))
    for k in (105.0, 120.0):
        mc_max = max_touch_probability(k, s0, sigma, nu, t, N_PATHS, steps, rng)
        cf_max = 2.0 * terminal_exceedance(k, s0, sigma, nu, t)
        assert abs(mc_max - cf_max) < 0.02


# --------------------------------------------------------------------------- #
# Item 3: terminal exceedance matches MC of terminal value
# --------------------------------------------------------------------------- #


def test_terminal_exceedance_matches_mc():
    rng = np.random.default_rng(SEED + 2)
    sigma, nu, t, s0 = 0.30, 0.05, 1.0, 100.0
    terminal, _ = _mc(s0, sigma, nu, t, rng)
    for mult in (0.95, 1.00, 1.10, 1.25):
        k = s0 * mult
        mc = float(np.mean(terminal >= k))
        cf = terminal_exceedance(k, s0, sigma, nu, t)
        assert abs(mc - cf) < 0.02, f"K={k}: MC={mc:.4f} CF={cf:.4f}"


# --------------------------------------------------------------------------- #
# Item 4: touch->terminal identity
#   P(S_T>=K) == P(M_T>=K) - reflection_term, and both match MC
# --------------------------------------------------------------------------- #


def test_touch_to_terminal_identity_closed_form():
    # Pure math: running-max survival minus reflection == terminal exceedance.
    for sigma, nu, t, s0 in [(0.25, 0.03, 1.0, 100.0), (0.40, -0.06, 1.5, 75.0)]:
        for mult in (1.0, 1.1, 1.3, 1.6):
            k = s0 * mult
            lhs = terminal_exceedance(k, s0, sigma, nu, t)
            rhs = running_max_survival(k, s0, sigma, nu, t) - reflection_term(k, s0, sigma, nu, t)
            assert abs(lhs - rhs) < 1e-12


def test_touch_to_terminal_identity_against_mc():
    rng = np.random.default_rng(SEED + 3)
    sigma, nu, t, s0 = 0.25, 0.03, 1.0, 100.0
    steps = int(N_STEPS * max(1.0, t))
    terminal, _ = _mc(s0, sigma, nu, t, rng)
    for mult in (1.0, 1.15, 1.30):
        k = s0 * mult
        mc_term = float(np.mean(terminal >= k))
        mc_max = max_touch_probability(k, s0, sigma, nu, t, N_PATHS, steps, rng)
        cf_term = terminal_exceedance(k, s0, sigma, nu, t)
        cf_max = running_max_survival(k, s0, sigma, nu, t)
        # CF identity matches MC on both legs.
        assert abs(mc_term - cf_term) < 0.02
        assert abs(mc_max - cf_max) < 0.02
        # The MC terminal also satisfies the identity via the CF reflection term.
        recon = cf_max - reflection_term(k, s0, sigma, nu, t)
        assert abs(mc_term - recon) < 0.02


# --------------------------------------------------------------------------- #
# Item 5: recovery / inverse problem (pure-math, deterministic, <1% on sigma)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("sigma_true", "nu_true", "t", "s0"),
    [
        (0.30, 0.05, 1.0, 100.0),
        (0.22, -0.03, 0.75, 80.0),
        (0.45, 0.0, 1.0, 100.0),
    ],
)
def test_fit_recovers_sigma_from_exact_ladder(sigma_true, nu_true, t, s0):
    # Build a touch ladder from the EXACT closed form (no MC noise).
    strikes = s0 * np.linspace(1.01, 1.60, 8)
    ladder = np.array([running_max_survival(float(k), s0, sigma_true, nu_true, t) for k in strikes])
    sigma_hat, nu_hat = fit_sigma_nu(strikes, ladder, s0, t)
    rel_err = abs(sigma_hat - sigma_true) / sigma_true
    assert rel_err < 0.01, f"sigma recovery {sigma_hat:.5f} vs {sigma_true} (rel {rel_err:.3%})"
    assert abs(nu_hat - nu_true) < 0.02


# --------------------------------------------------------------------------- #
# Heavier end-to-end MC cross-check (opt-in via -m slow)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_high_precision_mc_all_forms():
    rng = np.random.default_rng(SEED + 99)
    sigma, nu, t, s0 = 0.35, 0.07, 1.0, 100.0
    terminal, _ = simulate_gbm(s0, sigma, nu, t, 200_000, 500, rng)
    for mult in (1.0, 1.1, 1.25, 1.5):
        k = s0 * mult
        mc_max = max_touch_probability(k, s0, sigma, nu, t, 200_000, 500, rng)
        assert abs(mc_max - running_max_survival(k, s0, sigma, nu, t)) < 0.01
        assert abs(float(np.mean(terminal >= k)) - terminal_exceedance(k, s0, sigma, nu, t)) < 0.01
