"""Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR).

Canonical implementation in the :mod:`pfm.quant` namespace. Two older
implementations already live in the codebase
(:func:`pfm.robust_validation.deflated_sharpe_ratio` and
:func:`pfm.multitest.deflated_sharpe_full`); this module is the
**single-signature, paper-faithful** entry point that new code should use.
It is numerically consistent with ``pfm.multitest.deflated_sharpe_full``
when called with matching arguments.

Reference
---------
Bailey, D. H., & López de Prado, M. M. (2014).
*The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting, and Non-Normality.* Journal of Portfolio Management, 40(5),
94-107. (Working paper SSRN 2460551.)

Math
----
Given an observed Sharpe :math:`\\widehat{SR}` (per-period), number of
return observations :math:`T`, number of trials :math:`N`, return skew
:math:`\\gamma_3` and non-excess kurtosis :math:`\\gamma_4` (3 for
Gaussian):

1. Expected null maximum Sharpe across :math:`N` trials (eq. 5)::

       E[max SR] = sigma_SR * (
           (1 - gamma_em) * Phi^{-1}(1 - 1/N)
         +      gamma_em  * Phi^{-1}(1 - 1/(N*e))
       )

   where ``gamma_em = 0.5772156649...`` is the Euler-Mascheroni constant
   and ``sigma_SR`` is the cross-trial Sharpe dispersion (defaults to 1
   when only the trial count is known).

2. Edgeworth-expansion finite-sample SE of the Sharpe estimator (eq. 9)::

       SE(SR) = sqrt( (1 - gamma_3*SR + (gamma_4 - 1)/4 * SR^2) / (T - 1) )

3. Deflated Sharpe test statistic::

       z = (SR - E[max SR]) / SE(SR)
       DSR = Phi(z)          # cumulative
       p   = 1 - DSR         # one-sided p-value (H0: trader has zero skill)

Notes
-----
- The DSR returned from :func:`deflated_sharpe_ratio` is the *raw*
  studentised difference ``SR_per - E[max SR]`` on a per-period scale,
  matching ``pfm.multitest.deflated_sharpe_full``. Use :func:`dsr_pvalue`
  to convert to the one-sided p-value.
- ``deflated_sharpe_full`` accepts a raw returns array and computes
  sample Sharpe, skew, and kurtosis internally.
- An asymptotic Gumbel approximation
  ``E[max] ~ sqrt(2 ln N) - (gamma_em + ln ln N) / (2 sqrt(2 ln N))``
  is exposed via :func:`expected_max_sharpe_gumbel` for diagnostics; the
  BLDP eq. (5) form is preferred for finite N because it is exact for
  Gaussian-Sharpe IID trials at small N.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

# Euler-Mascheroni constant (BLDP 2014 eq. 5)
EULER_MASCHERONI: float = 0.5772156649015329

__all__ = [
    "EULER_MASCHERONI",
    "deflated_sharpe_full",
    "deflated_sharpe_ratio",
    "dsr_pvalue",
    "expected_max_sharpe_bldp",
    "expected_max_sharpe_gumbel",
    "sharpe_se_mertens",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def expected_max_sharpe_bldp(n_trials: int, sigma_sr: float = 1.0) -> float:
    """BLDP 2014 eq. (5) expected null-maximum Sharpe over ``n_trials``.

    Args:
        n_trials: number of strategies searched. ``n_trials <= 1`` returns 0.
        sigma_sr: cross-trial Sharpe dispersion. Use 1 when unknown.

    Returns:
        Non-negative expected maximum Sharpe under the null.
    """
    from scipy.stats import norm

    n = int(n_trials)
    if n <= 1:
        return 0.0
    z1 = float(norm.ppf(1.0 - 1.0 / n))
    z2 = float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    em = sigma_sr * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    return max(em, 0.0)


def expected_max_sharpe_gumbel(n_trials: int, sigma_sr: float = 1.0) -> float:
    """Asymptotic Gumbel approximation for ``E[max SR]``.

    ``E[max] ~ sigma_SR * ( sqrt(2 ln N) - (gamma + ln ln N) / (2 sqrt(2 ln N)) )``

    Useful as a sanity check against :func:`expected_max_sharpe_bldp`.
    For very small ``n_trials`` (<= e) ``ln ln N`` is non-positive and
    the formula is not meaningful; we clip to 0 in that regime.
    """
    n = int(n_trials)
    if n <= 2:
        return 0.0
    log_n = math.log(n)
    sqrt_2log = math.sqrt(2.0 * log_n)
    log_log_n = math.log(log_n)
    em = sigma_sr * (sqrt_2log - (EULER_MASCHERONI + log_log_n) / (2.0 * sqrt_2log))
    return max(em, 0.0)


def sharpe_se_mertens(
    sr_per_period: float, n_periods: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """Mertens / Bailey-LdP finite-sample SE of the Sharpe estimator.

    BLDP 2014 eq. (9). ``kurt`` is *non-excess* kurtosis (3 for Gaussian).

    Returns NaN if ``n_periods < 2`` and 0 if the under-the-square-root
    quantity is non-positive (numerical pathologies at very large |SR|).
    """
    t = int(n_periods)
    if t < 2:
        return float("nan")
    inner = 1.0 - skew * sr_per_period + ((kurt - 1.0) / 4.0) * sr_per_period * sr_per_period
    if inner <= 0.0:
        return 0.0
    return math.sqrt(inner / (t - 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    sr: float,
    n_trials: int,
    n_periods: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    ann_factor: float = 252.0,
    sigma_sr: float = 1.0,
) -> dict[str, float]:
    """Compute the Bailey-López de Prado Deflated Sharpe Ratio.

    Args:
        sr: observed Sharpe, *annualised* under ``ann_factor``.
            Pass the per-period Sharpe with ``ann_factor=1.0``.
        n_trials: number of strategies searched (data-mining budget).
            Must be >= 1.
        n_periods: number of return observations ``T``. Must be >= 2.
        skew: third standardised moment of returns. 0 = symmetric.
        kurt: fourth standardised moment (Pearson, NOT excess).
            3 = Gaussian.
        ann_factor: annualisation factor (252 daily, 12 monthly, 1 raw).
            Used to convert ``sr`` to per-period scale.
        sigma_sr: cross-trial Sharpe dispersion. Defaults to 1.

    Returns:
        Dict with keys ``dsr`` (raw per-period DSR = SR_per - E[max]),
        ``z`` (studentised test statistic), ``p_value`` (one-sided),
        ``expected_max_sharpe``, ``sigma_se``, plus echoed inputs.

    Raises:
        ValueError: if ``n_trials < 1`` or ``n_periods < 2`` or
            ``ann_factor <= 0`` or ``sigma_sr <= 0``.
    """
    if not math.isfinite(sr):
        raise ValueError(f"sr must be finite, got {sr!r}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_periods < 2:
        raise ValueError(f"n_periods must be >= 2, got {n_periods}")
    if ann_factor <= 0:
        raise ValueError(f"ann_factor must be > 0, got {ann_factor}")
    if sigma_sr <= 0:
        raise ValueError(f"sigma_sr must be > 0, got {sigma_sr}")
    if not math.isfinite(skew):
        raise ValueError(f"skew must be finite, got {skew!r}")
    if not math.isfinite(kurt):
        raise ValueError(f"kurt must be finite, got {kurt!r}")

    sr_per = float(sr) / math.sqrt(ann_factor)
    expected_max = expected_max_sharpe_bldp(n_trials, sigma_sr=sigma_sr)
    sigma_se = sharpe_se_mertens(sr_per, n_periods, skew=skew, kurt=kurt)

    if not math.isfinite(sigma_se) or sigma_se <= 0.0:
        z = 0.0
    else:
        z = (sr_per - expected_max) / sigma_se

    dsr = sr_per - expected_max

    return {
        "dsr": float(dsr),
        "z": float(z),
        "p_value": float(dsr_pvalue(z)),
        "expected_max_sharpe": float(expected_max),
        "sigma_se": float(sigma_se) if math.isfinite(sigma_se) else float("nan"),
        "sr_per_period": float(sr_per),
        "n_trials": int(n_trials),
        "n_periods": int(n_periods),
        "skew": float(skew),
        "kurt": float(kurt),
    }


def dsr_pvalue(z: float) -> float:
    """One-sided p-value for a DSR z-statistic.

    Under the null ``H0: true SR <= E[max SR]`` the z-statistic is
    approximately N(0, 1); the one-sided p-value is ``1 - Phi(z)``.

    Args:
        z: studentised test statistic from :func:`deflated_sharpe_ratio`.

    Returns:
        p-value in [0, 1]. Non-finite ``z`` (NaN/inf) maps to 1.0
        (i.e. "no evidence against the null").
    """
    from scipy.stats import norm

    if not math.isfinite(z):
        # Conservative: treat undefined statistic as "no evidence".
        return 1.0
    return float(1.0 - norm.cdf(z))


def deflated_sharpe_full(
    returns: Sequence[float] | Iterable[float] | np.ndarray,
    n_trials: int,
    *,
    ann_factor: float = 252.0,
    sigma_sr: float = 1.0,
) -> dict[str, float]:
    """All-in-one: compute Sharpe, skew, kurt, then DSR + p-value.

    Args:
        returns: per-period return series (any iterable of floats).
        n_trials: number of strategies searched.
        ann_factor: annualisation factor for the returned Sharpe.
        sigma_sr: cross-trial Sharpe dispersion.

    Returns:
        Same dict shape as :func:`deflated_sharpe_ratio` plus
        ``sharpe_annualised``, ``mean``, ``std``, ``skew``, ``kurt``.

    Raises:
        ValueError: empty returns, zero-variance returns, or n_trials < 1.
    """
    arr = np.asarray(list(returns), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("returns is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contains non-finite values (NaN/inf)")
    if arr.size < 2:
        raise ValueError(f"need at least 2 observations, got {arr.size}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # Reject near-zero variance: a series that is constant to within
    # 1e-12 of its mean is effectively degenerate and would produce a
    # nonsense Sharpe (mean / std blows up).
    scale = max(abs(mean), 1.0)
    if not math.isfinite(std) or std <= 1e-12 * scale:
        raise ValueError("returns has zero (or non-finite) variance")

    # Per-period sample moments. Skew & kurtosis are the standardised
    # third and fourth moments using the same population denominator that
    # BLDP 2014 assumes (i.e. divide by std^3, std^4 of the unbiased std).
    centered = arr - mean
    skew = float(np.mean(centered**3) / std**3)
    kurt = float(np.mean(centered**4) / std**4)  # non-excess

    sr_per = mean / std
    sr_ann = sr_per * math.sqrt(ann_factor)

    result = deflated_sharpe_ratio(
        sr_ann,
        n_trials,
        arr.size,
        skew=skew,
        kurt=kurt,
        ann_factor=ann_factor,
        sigma_sr=sigma_sr,
    )
    result.update(
        {
            "sharpe_annualised": float(sr_ann),
            "mean": mean,
            "std": std,
            "skew": skew,
            "kurt": kurt,
            "n_obs": int(arr.size),
        }
    )
    return result
