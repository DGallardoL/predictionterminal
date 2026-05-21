"""Politis-Romano stationary block bootstrap for autocorrelated time series.

Why
---
The classic IID bootstrap (sample-with-replacement on individual observations)
*destroys* the temporal dependence of a return series. If returns are
auto-correlated -- as is empirically true for prediction-market factor returns,
volatility, and many momentum/mean-reversion strategy PnLs -- the IID bootstrap
gives confidence intervals that are **too narrow**, because each resample
behaves as if the series were independent.

The **stationary bootstrap** (Politis & Romano 1994, JASA 89(428):1303-1313)
resamples contiguous *blocks* of random geometric length. The blocks preserve
local autocorrelation; the random length means the resampled series is itself
stationary (unlike the moving-block bootstrap of Kunsch 1989, where block
lengths are fixed). The expected block size ``L`` controls the bias/variance
trade-off:

* ``L = 1`` reduces to the IID bootstrap.
* ``L`` large preserves long-range dependence but increases variance.
* Common rule of thumb: ``L = O(n^{1/3})`` for series of length ``n``.

Block lengths are drawn from a geometric distribution with parameter
``p = 1/L`` (mean ``L``); starting indices are uniform over the original
series with **circular wrap-around** so every observation can start a block
with equal probability.

Reference
---------
Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap*.
Journal of the American Statistical Association, 89(428), 1303-1313.

Notes
-----
* The statistic callable receives a 1-D ``np.ndarray`` of length ``n`` (the
  same length as the input). If it returns NaN/inf for a particular resample
  that draw is dropped from the CI calculation -- callers that pass an
  always-NaN statistic will get NaN endpoints back rather than an exception.
* Reproducibility is per-call: passing the same ``random_state`` and inputs
  yields the same bootstrap distribution.
* This module is intentionally dependency-light (only numpy) so it can be
  imported from any router without pulling scipy at import time.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["stationary_block_bootstrap"]


def _generate_resample(
    arr: np.ndarray,
    rng: np.random.Generator,
    p: float,
) -> np.ndarray:
    """Generate one stationary-bootstrap resample of length ``n``.

    Geometric block lengths with mean ``1/p`` (so ``p = 1/avg_block_size``).
    Block start indices are uniform over [0, n) with circular wrap so the
    resample is stationary.
    """
    n = arr.size
    out = np.empty(n, dtype=arr.dtype)
    i = 0
    while i < n:
        # Uniform start.
        start = int(rng.integers(0, n))
        # Geometric block length: P(L=k) = (1-p)^{k-1} p, k=1,2,...
        # numpy's geometric matches this convention.
        if p >= 1.0:
            block_len = 1
        else:
            block_len = int(rng.geometric(p))
        # Trim to remaining slots.
        block_len = min(block_len, n - i)
        # Fill with circular wrap.
        end = start + block_len
        if end <= n:
            out[i : i + block_len] = arr[start:end]
        else:
            first = n - start
            out[i : i + first] = arr[start:]
            out[i + first : i + block_len] = arr[: block_len - first]
        i += block_len
    return out


def stationary_block_bootstrap(
    returns: np.ndarray,
    *,
    n_resamples: int = 2000,
    avg_block_size: int = 5,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Stationary block bootstrap CI for a user-supplied statistic.

    Args:
        returns: 1-D array of observations (e.g. daily returns). Must be
            non-empty; single-observation inputs are allowed but produce a
            degenerate distribution (every resample is identical).
        n_resamples: number of bootstrap resamples. Must be >= 1.
        avg_block_size: expected geometric block length L. ``L=1`` reduces
            to IID bootstrap. Must be >= 1.
        statistic: callable mapping 1-D ndarray -> float. Defaults to mean.
            May return NaN; such resamples are dropped from the CI.
        confidence: nominal coverage in (0, 1), e.g. 0.95.
        random_state: integer seed for reproducibility.

    Returns:
        Dict with keys:

        * ``mean``: mean of the bootstrap distribution of the statistic
          (NaN if every resample was NaN).
        * ``ci_low``, ``ci_high``: percentile CI endpoints at the requested
          confidence level.
        * ``std``: std of the bootstrap distribution (ddof=1; 0.0 with one
          finite sample, NaN if none).
        * ``n_resamples``: count of *finite* bootstrap samples retained.
        * ``avg_block_size``: echoed input (clamped to series length if
          larger than ``returns.size``).
        * ``observed``: statistic value on the original sample (NaN-safe).

    Raises:
        ValueError: empty / non-1-D / non-finite ``returns``; invalid
            ``n_resamples``, ``avg_block_size``, or ``confidence``.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        raise ValueError("returns is empty")
    if arr.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contains non-finite values (NaN/inf)")
    if int(n_resamples) < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if int(avg_block_size) < 1:
        raise ValueError(f"avg_block_size must be >= 1, got {avg_block_size}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if not callable(statistic):
        raise ValueError("statistic must be callable")

    n = arr.size
    L = min(int(avg_block_size), n)
    p = 1.0 / L if L > 0 else 1.0

    rng = np.random.default_rng(int(random_state))

    # Observed (original-sample) statistic, NaN-safe.
    try:
        observed = float(statistic(arr))
    except Exception:  # pragma: no cover - defensive
        observed = float("nan")

    # Generate resamples and apply the statistic.
    boot_vals = np.empty(int(n_resamples), dtype=float)
    for b in range(int(n_resamples)):
        sample = _generate_resample(arr, rng, p)
        try:
            val = float(statistic(sample))
        except Exception:
            val = float("nan")
        boot_vals[b] = val

    finite = boot_vals[np.isfinite(boot_vals)]
    alpha = 1.0 - float(confidence)

    if finite.size == 0:
        # All NaN: caller's statistic is unusable on this series.
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "std": float("nan"),
            "n_resamples": 0,
            "avg_block_size": int(L),
            "observed": observed,
        }

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    ci_low = float(np.percentile(finite, lo_q))
    ci_high = float(np.percentile(finite, hi_q))

    if finite.size > 1:
        std = float(finite.std(ddof=1))
    else:
        std = 0.0

    return {
        "mean": float(finite.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "std": std,
        "n_resamples": int(finite.size),
        "avg_block_size": int(L),
        "observed": observed,
    }
