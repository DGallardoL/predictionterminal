"""Bootstrap confidence intervals for the Sharpe ratio.

Complements :mod:`pfm.quant.deflated_sharpe` (Bailey-LdP DSR) by giving a
distribution-free uncertainty estimate around an observed Sharpe.

Two CI methods are supported:

* **percentile**: simple :math:`\\alpha/2`, :math:`1-\\alpha/2` quantiles of
  the bootstrap distribution. Fast and unbiased when the bootstrap
  distribution is symmetric, but mis-covers under skew / heavy tails.
* **bca** (bias-corrected accelerated, Efron 1987): adjusts the
  percentile endpoints for *median bias* (``z0``) and for *acceleration*
  (``a``, a third-moment correction estimated via jackknife). Recovers
  the correct coverage on skewed return distributions (e.g. lognormal,
  trend-following series with long right tails) at the cost of one extra
  O(n) jackknife pass.

Reference
---------
Efron, B. (1987). *Better Bootstrap Confidence Intervals.* JASA 82(397),
171-185.

Notes
-----
The Sharpe ratio of a return series ``r`` is computed as
``mean(r) / std(r, ddof=1) * sqrt(ann_factor)``. Per-period scale is used
internally; ``ann_factor`` only rescales the final outputs.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "bootstrap_sharpe_ci",
    "sharpe_ratio",
]


def sharpe_ratio(returns: np.ndarray, ann_factor: float = 252.0) -> float:
    """Annualised sample Sharpe ratio.

    Returns NaN when the input is empty or has zero (sample) variance.
    Uses unbiased std (``ddof=1``).
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return float("nan")
    return mean / std * math.sqrt(ann_factor)


def _percentile_ci(boot_sharpes: np.ndarray, confidence: float) -> tuple[float, float]:
    """Simple percentile CI from a bootstrap distribution."""
    alpha = 1.0 - confidence
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    return (
        float(np.percentile(boot_sharpes, lo_q)),
        float(np.percentile(boot_sharpes, hi_q)),
    )


def _bca_ci(
    returns: np.ndarray,
    observed_sharpe: float,
    boot_sharpes: np.ndarray,
    confidence: float,
    ann_factor: float,
) -> tuple[float, float]:
    """Efron 1987 bias-corrected accelerated (BCa) CI.

    Falls back to percentile if the bias correction or acceleration is
    non-finite (e.g. degenerate jackknife with zero variance), so the
    function is always defined.
    """
    from scipy.stats import norm

    n = returns.size
    alpha = 1.0 - confidence

    # Bias-correction z0: standard-normal quantile of the proportion of
    # bootstrap replicates below the observed Sharpe.
    proportion_below = float(np.mean(boot_sharpes < observed_sharpe))
    # Clamp to (0, 1) so norm.ppf is finite.
    proportion_below = min(
        max(proportion_below, 1.0 / (boot_sharpes.size + 1)), 1.0 - 1.0 / (boot_sharpes.size + 1)
    )
    z0 = float(norm.ppf(proportion_below))

    # Acceleration via jackknife (Efron eq. 6.6).
    jack = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack[i] = sharpe_ratio(returns[mask], ann_factor=ann_factor)
    finite = np.isfinite(jack)
    if not np.any(finite):
        return _percentile_ci(boot_sharpes, confidence)
    jack = jack[finite]
    jack_mean = float(jack.mean())
    diff = jack_mean - jack  # note: mean - jack[i] is the canonical sign
    num = float(np.sum(diff**3))
    den = 6.0 * (float(np.sum(diff**2)) ** 1.5)
    if den <= 0.0 or not math.isfinite(num) or not math.isfinite(den):
        a = 0.0
    else:
        a = num / den

    z_lo = float(norm.ppf(alpha / 2.0))
    z_hi = float(norm.ppf(1.0 - alpha / 2.0))

    def _adjust(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if denom == 0.0 or not math.isfinite(denom):
            return float("nan")
        return float(norm.cdf(z0 + (z0 + z) / denom))

    p_lo = _adjust(z_lo)
    p_hi = _adjust(z_hi)
    if not (math.isfinite(p_lo) and math.isfinite(p_hi)):
        return _percentile_ci(boot_sharpes, confidence)
    # Clamp to [0, 100] percent.
    p_lo = min(max(p_lo, 0.0), 1.0)
    p_hi = min(max(p_hi, 0.0), 1.0)
    return (
        float(np.percentile(boot_sharpes, 100.0 * p_lo)),
        float(np.percentile(boot_sharpes, 100.0 * p_hi)),
    )


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    method: str = "percentile",
    ann_factor: float = 252.0,
    random_state: int = 42,
) -> dict:
    """Bootstrap CI for the (annualised) Sharpe ratio.

    Args:
        returns: 1-D array of per-period returns (e.g. daily log returns).
        n_resamples: number of bootstrap resamples. >= 100 recommended.
        confidence: nominal coverage (e.g. 0.95).
        method: ``"percentile"`` or ``"bca"`` (bias-corrected accelerated).
        ann_factor: annualisation factor (252 daily, 12 monthly, 1 raw).
        random_state: seed for reproducibility.

    Returns:
        Dict with keys:

        - ``sharpe_mean``: mean of the bootstrap Sharpe distribution.
        - ``sharpe_ci_low`` / ``sharpe_ci_high``: CI endpoints.
        - ``sharpe_std``: std of the bootstrap distribution.
        - ``n_resamples``: echoed.
        - ``method``: echoed.
        - ``observed_sharpe``: Sharpe of the original sample.
        - ``confidence``: echoed nominal coverage.

    Raises:
        ValueError: empty / non-1-D / non-finite returns, or invalid
            ``n_resamples``, ``confidence``, or ``method``.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        raise ValueError("returns is empty")
    if arr.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contains non-finite values (NaN/inf)")
    if arr.size < 2:
        raise ValueError(f"need at least 2 observations, got {arr.size}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if method not in ("percentile", "bca"):
        raise ValueError(f"method must be 'percentile' or 'bca', got {method!r}")
    if ann_factor <= 0:
        raise ValueError(f"ann_factor must be > 0, got {ann_factor}")

    observed = sharpe_ratio(arr, ann_factor=ann_factor)

    # Degenerate handling: a constant series (or all-zeros) has zero
    # variance, hence undefined Sharpe. Return NaN endpoints instead of
    # raising, so callers can ingest pipelines that produce occasional
    # flat windows. We still flag observed Sharpe as NaN.
    std = float(arr.std(ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return {
            "sharpe_mean": float("nan"),
            "sharpe_ci_low": float("nan"),
            "sharpe_ci_high": float("nan"),
            "sharpe_std": float("nan"),
            "n_resamples": int(n_resamples),
            "method": method,
            "observed_sharpe": float("nan"),
            "confidence": float(confidence),
            "degenerate": True,
        }

    rng = np.random.default_rng(int(random_state))
    n = arr.size
    # Vectorised resampling: draw an (n_resamples, n) integer index matrix.
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    resamples = arr[idx]
    means = resamples.mean(axis=1)
    stds = resamples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.where(stds > 0, means / stds * math.sqrt(ann_factor), np.nan)
    # Drop any degenerate resamples (zero-variance, unlikely but possible
    # for series with many ties). Keep at least 10 for the CI to make sense.
    boot = boot[np.isfinite(boot)]
    if boot.size < max(10, int(0.5 * n_resamples)):
        # Too many degenerate draws — likely a near-flat series.
        return {
            "sharpe_mean": float(observed),
            "sharpe_ci_low": float("nan"),
            "sharpe_ci_high": float("nan"),
            "sharpe_std": float("nan"),
            "n_resamples": int(boot.size),
            "method": method,
            "observed_sharpe": float(observed),
            "confidence": float(confidence),
            "degenerate": True,
        }

    if method == "percentile":
        ci_low, ci_high = _percentile_ci(boot, confidence)
    else:
        ci_low, ci_high = _bca_ci(arr, observed, boot, confidence, ann_factor)

    return {
        "sharpe_mean": float(boot.mean()),
        "sharpe_ci_low": float(ci_low),
        "sharpe_ci_high": float(ci_high),
        "sharpe_std": float(boot.std(ddof=1)) if boot.size > 1 else 0.0,
        "n_resamples": int(boot.size),
        "method": method,
        "observed_sharpe": float(observed),
        "confidence": float(confidence),
        "degenerate": False,
    }
