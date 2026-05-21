"""Fractional differentiation (Hosking 1981, López de Prado 2018 §5).

The classical first-difference operator (1 − L) destroys all long-memory
information in a series — the resulting Δp_t is uncorrelated with anything
in the past beyond the immediate lag. For probability series this throws
out the baby with the bathwater: the level p_t carries genuine
information about *where the market thinks the world is*, and we want to
preserve that while making the series stationary enough for ML/regression.

The fractional operator (1 − L)^d for ``d ∈ (0, 1)`` interpolates:

- d = 0:  no transformation (carries full memory, but I(1) → spurious regression)
- d = 1:  classical first difference (kills memory, ensures stationarity)
- d = 0.5: half-difference, preserves significant long-memory structure
  while typically passing ADF stationarity test.

The transformed series is

    fd_t = Σ_{k=0}^∞ ω_k · p_{t-k}

with weights from the binomial expansion (Hosking 1981 eq. 1.2):

    ω_0 = 1
    ω_k = ω_{k-1} · −(d − k + 1) / k

López de Prado (2018) §5 popularised this for finance. The recommended
practice: for each series, find the *smallest* ``d`` that makes the
series pass an ADF test at α=0.05. That ``d`` retains the maximum amount
of memory while ensuring stationarity.

References:
    Hosking, J. R. M. (1981). "Fractional Differencing." Biometrika 68(1).
    López de Prado, M. (2018). *Advances in Financial Machine Learning* §5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _ffd_weights(d: float, threshold: float = 1e-3) -> np.ndarray:
    """Compute fractional-differentiation weights ω_k via the recursion
    ``ω_k = −ω_{k-1} · (d − k + 1) / k`` until ``|ω_k| < threshold``.
    """
    if d <= 0 or d >= 1:
        raise ValueError(f"d must be in (0, 1), got {d}")
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        if k > 10_000:  # hard safety
            break
    return np.array(weights[::-1], dtype=float)  # reversed so most-recent weight is last


def fractional_diff(
    series: pd.Series,
    *,
    d: float,
    threshold: float = 1e-3,
) -> pd.Series:
    """Apply the fixed-width fractional-differentiation filter.

    The output starts from the (n_weights)-th bar; earlier bars are NaN
    because we don't have enough history to apply the full filter window.

    Args:
        series: input level series (probabilities, prices, etc).
        d: differencing exponent in (0, 1).
        threshold: drop weights smaller than this in absolute value.

    Returns:
        Same-indexed series of fractionally-differenced values.
    """
    s = series.dropna()
    n = len(s)
    if n < 30:
        raise ValueError(f"fractional_diff: need ≥30 bars, got {n}")
    w = _ffd_weights(d, threshold=threshold)
    width = len(w)
    out = np.full(n, np.nan)
    arr = s.values
    for t in range(width - 1, n):
        out[t] = float(np.dot(w, arr[t - width + 1 : t + 1]))
    return pd.Series(out, index=s.index, name=f"fdiff_d={d}")


@dataclass(frozen=True)
class MinimalDResult:
    """Output of :func:`find_minimal_d`."""

    d: float | None  # minimum d that makes series stationary
    adf_p_at_d: float | None
    correlation_with_original: float | None  # (memory preservation metric)
    weights_width: int  # how many lags the filter spans
    grid_results: list[dict]  # per-d grid points (d, adf_p, corr)


def find_minimal_d(
    series: pd.Series,
    *,
    d_grid: list[float] | None = None,
    adf_threshold: float = 0.05,
    threshold: float = 1e-3,
) -> MinimalDResult:
    """Find the smallest ``d`` in (0, 1) that makes ``series`` stationary
    (ADF p < adf_threshold). The López de Prado (2018) recipe.

    Args:
        series: input series.
        d_grid: candidate ``d`` values; default = 0.05, 0.10, 0.15, ..., 0.95.
        adf_threshold: stationarity p-value cutoff.
        threshold: weight cutoff.

    Returns:
        :class:`MinimalDResult`. ``d=None`` if no grid value achieves
        stationarity (caller should expand the grid or increase threshold).
    """
    from statsmodels.tsa.stattools import adfuller

    if d_grid is None:
        d_grid = [round(0.05 * (i + 1), 2) for i in range(19)]  # 0.05..0.95
    s = series.dropna()
    if len(s) < 50:
        raise ValueError(f"find_minimal_d: need ≥50 bars, got {len(s)}")

    grid_results = []
    found_d = None
    found_p = None
    found_corr = None
    found_width = 0
    for d in d_grid:
        try:
            fd = fractional_diff(s, d=d, threshold=threshold)
        except ValueError:
            continue
        fd_clean = fd.dropna()
        if len(fd_clean) < 20:
            continue
        try:
            _adf_stat, adf_p, *_ = adfuller(fd_clean.values, autolag="AIC")
            adf_p = float(adf_p)
        except Exception:
            continue
        # Correlation with original (memory metric)
        s_align = s.loc[fd_clean.index]
        if s_align.std(ddof=1) > 0 and fd_clean.std(ddof=1) > 0:
            corr = float(np.corrcoef(s_align.values, fd_clean.values)[0, 1])
        else:
            corr = float("nan")
        grid_results.append(
            {
                "d": float(d),
                "adf_p": adf_p,
                "corr_with_original": corr,
                "n_after_filter": len(fd_clean),
            }
        )
        if adf_p < adf_threshold and found_d is None:
            found_d = float(d)
            found_p = adf_p
            found_corr = corr
            found_width = len(_ffd_weights(d, threshold=threshold))
    return MinimalDResult(
        d=found_d,
        adf_p_at_d=found_p,
        correlation_with_original=found_corr,
        weights_width=found_width,
        grid_results=grid_results,
    )


__all__ = ["MinimalDResult", "find_minimal_d", "fractional_diff"]
