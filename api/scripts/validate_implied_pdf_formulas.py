"""Independent numerical-validation oracle for the GBM barrier formulas.

This is a STANDALONE reference implementation of the closed-form geometric
Brownian motion (GBM) probabilities used by the implied-PDF feature in
``pfm/vol/implied_pdf.py``. Its sole purpose is to be an *independent* oracle:
it re-derives the running-max survival, terminal-exceedance, and touch->terminal
identities from first principles here (no import of the production module) and
cross-checks every closed form against a Monte-Carlo simulation of GBM paths.

Modelling convention
--------------------
We simulate ``S_t = S_0 * exp(sigma * W_t + nu * t)`` where ``nu = r - q - 0.5*sigma**2``
is the LOG-drift. All quantities here are *probabilities* under this measure;
there is no ``exp(-rT)`` discounting (these are touch / exceedance probabilities,
not option prices).

Closed forms (with ``a = ln(K/S0)``, ``Phi = norm.cdf``):

1. Running-max survival:
       P(M_T >= K) = Phi((-a + nu*T)/(sigma*sqrt(T)))
                     + (K/S0)**(2*nu/sigma**2) * Phi((-a - nu*T)/(sigma*sqrt(T)))
   where M_T = max_{t<=T} S_t (reflection principle for drifted BM).

2. Driftless special case (nu = 0):
       P(M_T >= K) = 2 * P(S_T >= K) = 2 * Phi(-a/(sigma*sqrt(T))).

3. Terminal exceedance:
       P(S_T >= K) = Phi((-a + nu*T)/(sigma*sqrt(T))).

4. Touch->terminal identity:
       P(S_T >= K) = P(M_T >= K) - (K/S0)**(2*nu/sigma**2)
                                    * Phi((-a - nu*T)/(sigma*sqrt(T))).

5. Recovery / inverse problem: build a synthetic touch ladder from form (1) for
   known (sigma, nu), then refit (sigma, nu) by least squares -> the math must
   be invertible.

Run with::

    python -m scripts.validate_implied_pdf_formulas
    # or
    python api/scripts/validate_implied_pdf_formulas.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# Closed forms (the independent reference)
# --------------------------------------------------------------------------- #


def running_max_survival(k: float, s0: float, sigma: float, nu: float, t: float) -> float:
    """Closed-form P(M_T >= K) for GBM with log-drift ``nu``.

    Args:
        k: Barrier / strike (absolute price level), K >= S0 expected.
        s0: Spot at t=0.
        sigma: Log-volatility (annualized, per unit of ``t``).
        nu: Log-drift, ``nu = r - q - 0.5*sigma**2``.
        t: Horizon.

    Returns:
        Probability that the running maximum reaches or exceeds ``k``.
    """
    a = np.log(k / s0)
    vol = sigma * np.sqrt(t)
    term1 = norm.cdf((-a + nu * t) / vol)
    term2 = (k / s0) ** (2.0 * nu / sigma**2) * norm.cdf((-a - nu * t) / vol)
    return float(term1 + term2)


def terminal_exceedance(k: float, s0: float, sigma: float, nu: float, t: float) -> float:
    """Closed-form P(S_T >= K) for GBM with log-drift ``nu``."""
    a = np.log(k / s0)
    vol = sigma * np.sqrt(t)
    return float(norm.cdf((-a + nu * t) / vol))


def reflection_term(k: float, s0: float, sigma: float, nu: float, t: float) -> float:
    """The reflection contribution (K/S0)**(2 nu/sigma^2) * Phi((-a - nu t)/(sigma sqrt t))."""
    a = np.log(k / s0)
    vol = sigma * np.sqrt(t)
    return float((k / s0) ** (2.0 * nu / sigma**2) * norm.cdf((-a - nu * t) / vol))


# --------------------------------------------------------------------------- #
# Monte-Carlo simulation of GBM
# --------------------------------------------------------------------------- #


def simulate_gbm(
    s0: float,
    sigma: float,
    nu: float,
    t: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GBM paths on a fine grid.

    Uses ``S_t = S0 * exp(sigma * W_t + nu * t)`` with exact Brownian increments,
    so there is no Euler discretisation bias. The discrete running max returned
    here still *under*-estimates the continuous-time maximum (the path can peak
    between grid points); for an unbiased touch probability use
    :func:`max_touch_probability` which applies the Brownian-bridge correction.

    Returns:
        Tuple ``(terminal, running_max)`` arrays of shape ``(n_paths,)``, where
        ``running_max`` is the *discrete-grid* maximum (including ``S0``).
    """
    dt = t / n_steps
    # Brownian increments.
    dw = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    w = np.cumsum(dw, axis=1)
    times = np.arange(1, n_steps + 1) * dt
    log_s = np.log(s0) + sigma * w + nu * times
    s = np.exp(log_s)
    # Include the starting point S0 in the running max.
    running_max = np.maximum(np.max(s, axis=1), s0)
    terminal = s[:, -1]
    return terminal, running_max


def max_touch_probability(
    k: float,
    s0: float,
    sigma: float,
    nu: float,
    t: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> float:
    """Unbiased MC estimate of ``P(M_T >= K)`` via the Brownian-bridge correction.

    A discrete-grid maximum systematically under-estimates the continuous max,
    which biases a naive ``mean(discrete_max >= K)`` *low* by ``O(1/sqrt(n_steps))``.
    We remove that bias exactly: between consecutive grid points the log-price is
    a Brownian bridge whose running-max-exceeds-barrier probability is

        p_i = exp( -2 * (b - x_i) * (b - x_{i+1}) / (sigma**2 * dt) )

    for ``x_{i+1} < b`` (and ``1`` if either endpoint already crosses ``b``),
    where ``x = log(S)`` and ``b = log(K)``. The path touches the barrier iff any
    sub-interval bridge crosses, so we accumulate the no-touch probability across
    steps and return ``1 - mean(prod_i (1 - p_i))``.

    Returns:
        Estimated continuous-time touch probability ``P(M_T >= K)``.
    """
    b = np.log(k)
    dt = t / n_steps
    dw = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    w = np.cumsum(dw, axis=1)
    times = np.arange(1, n_steps + 1) * dt
    log_s = np.log(s0) + sigma * w + nu * times  # x at grid points 1..n
    # Prepend the known start log(S0).
    x = np.empty((n_paths, n_steps + 1))
    x[:, 0] = np.log(s0)
    x[:, 1:] = log_s

    x0 = x[:, :-1]
    x1 = x[:, 1:]
    # Bridge crossing probability per sub-interval (vectorised).
    gap = (b - x0) * (b - x1)
    p_cross = np.exp(-2.0 * gap / (sigma**2 * dt))
    # If either endpoint is at/above the barrier, crossing is certain.
    endpoint_cross = (x0 >= b) | (x1 >= b)
    p_cross = np.where(endpoint_cross, 1.0, p_cross)
    # Probability of NO touch over the whole path = prod (1 - p_cross).
    no_touch = np.prod(1.0 - p_cross, axis=1)
    return float(1.0 - np.mean(no_touch))


# --------------------------------------------------------------------------- #
# Inverse problem (recovery of sigma, nu from a touch ladder)
# --------------------------------------------------------------------------- #


def fit_sigma_nu(
    strikes: np.ndarray,
    observed_survival: np.ndarray,
    s0: float,
    t: float,
    x0: tuple[float, float] = (0.20, 0.0),
) -> tuple[float, float]:
    """Recover (sigma, nu) from a touch ladder by least-squares on form (1).

    Args:
        strikes: Array of barrier levels (>= S0).
        observed_survival: Observed P(M_T >= K) at each strike.
        s0: Spot.
        t: Horizon.
        x0: Initial guess (sigma, nu).

    Returns:
        Recovered ``(sigma, nu)``.
    """

    def objective(params: np.ndarray) -> float:
        sigma, nu = params
        if sigma <= 1e-6:
            return 1e9
        model = np.array([running_max_survival(float(k), s0, sigma, nu, t) for k in strikes])
        return float(np.sum((model - observed_survival) ** 2))

    res = minimize(
        objective,
        np.array(x0),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 5000},
    )
    sigma_hat, nu_hat = res.x
    return float(sigma_hat), float(nu_hat)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the MC cross-checks and the fit-recovery, printing tidy tables."""
    rng = np.random.default_rng(20260519)
    n_paths = 200_000
    steps_per_period = 500

    # Grid of (sigma, nu, T, S0). Include nu != 0 cases.
    scenarios = [
        # (sigma, nu, T, S0)
        (0.20, 0.0, 1.0, 100.0),
        (0.30, 0.05, 1.0, 100.0),
        (0.25, -0.04, 0.5, 50.0),
        (0.40, 0.10, 2.0, 100.0),
    ]

    max_err_max = 0.0
    max_err_term = 0.0

    print("=" * 88)
    print("Independent GBM barrier-formula oracle  (oracle for pfm/vol/implied_pdf.py)")
    print(f"MC: n_paths={n_paths:,}  steps={int(steps_per_period)}/period  seed=20260519")
    print("=" * 88)

    for sigma, nu, t, s0 in scenarios:
        n_steps = int(steps_per_period * max(1.0, t))
        terminal, run_max = simulate_gbm(s0, sigma, nu, t, n_paths, n_steps, rng)

        # Strikes from spot upward (touch ladder is for K >= S0).
        strikes = s0 * np.array([1.00, 1.05, 1.10, 1.20, 1.35, 1.50])

        print(f"\nScenario: sigma={sigma}  nu={nu}  T={t}  S0={s0}")
        print("-" * 88)
        print(
            f"{'K':>8} | {'MC P(M>=K)':>11} {'CF P(M>=K)':>11} {'absErr':>8} | "
            f"{'MC P(S_T>=K)':>12} {'CF P(S_T>=K)':>12} {'absErr':>8} | {'identity':>9}"
        )
        for k in strikes:
            # Bridge-corrected (unbiased) touch probability; reuse the same rng
            # stream is not required for the table, fresh draws are fine.
            mc_max = max_touch_probability(float(k), s0, sigma, nu, t, n_paths, n_steps, rng)
            cf_max = running_max_survival(float(k), s0, sigma, nu, t)
            mc_term = float(np.mean(terminal >= k))
            cf_term = terminal_exceedance(float(k), s0, sigma, nu, t)
            # Touch->terminal identity (closed form): CF_max - reflection == CF_term
            ident = cf_max - reflection_term(float(k), s0, sigma, nu, t)

            e_max = abs(mc_max - cf_max)
            e_term = abs(mc_term - cf_term)
            max_err_max = max(max_err_max, e_max)
            max_err_term = max(max_err_term, e_term)

            print(
                f"{k:>8.2f} | {mc_max:>11.4f} {cf_max:>11.4f} {e_max:>8.4f} | "
                f"{mc_term:>12.4f} {cf_term:>12.4f} {e_term:>8.4f} | "
                f"{abs(ident - cf_term):>9.2e}"
            )

    # Driftless factor-of-2 demo (nu == 0).
    print("\n" + "=" * 88)
    print("Driftless special case  P(M_T>=K) == 2*P(S_T>=K)  (nu = 0)")
    print("=" * 88)
    sigma, nu, t, s0 = 0.20, 0.0, 1.0, 100.0
    for k in [105.0, 120.0, 140.0]:
        cf_max = running_max_survival(k, s0, sigma, nu, t)
        cf_term = terminal_exceedance(k, s0, sigma, nu, t)
        print(
            f"  K={k:>7.2f}  P(M>=K)={cf_max:.5f}  2*P(S_T>=K)={2 * cf_term:.5f}  "
            f"diff={abs(cf_max - 2 * cf_term):.2e}"
        )

    # Fit recovery (pure closed-form ladder -> recover sigma, nu).
    print("\n" + "=" * 88)
    print("Recovery / inverse problem  (synthetic ladder from exact CF -> refit)")
    print("=" * 88)
    for sigma_true, nu_true, t, s0 in [(0.30, 0.05, 1.0, 100.0), (0.22, -0.03, 0.75, 80.0)]:
        ladder_k = s0 * np.linspace(1.01, 1.60, 8)
        ladder = np.array(
            [running_max_survival(float(k), s0, sigma_true, nu_true, t) for k in ladder_k]
        )
        sigma_hat, nu_hat = fit_sigma_nu(ladder_k, ladder, s0, t)
        s_err = abs(sigma_hat - sigma_true) / sigma_true
        print(
            f"  true  sigma={sigma_true:.4f} nu={nu_true:+.4f}  ->  "
            f"recovered sigma={sigma_hat:.4f} nu={nu_hat:+.4f}  "
            f"(sigma rel.err={s_err:.2%})"
        )

    print("\n" + "=" * 88)
    print(f"MAX MC abs error  running-max: {max_err_max:.4f}   terminal: {max_err_term:.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
