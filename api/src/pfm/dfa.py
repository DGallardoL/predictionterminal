"""Detrended Fluctuation Analysis (Peng et al. 1994).

Robust alternative to R/S analysis for the Hurst exponent. DFA detrends
each subseries with a polynomial fit before computing the fluctuation,
which makes it less sensitive to non-stationary trends than classical R/S.

Algorithm:

1.  Integrate the (centered) series: Y_t = Σ_{i=1}^t (x_i − μ).
2.  Partition Y into non-overlapping segments of length n.
3.  Fit a polynomial (default degree 1: linear) within each segment;
    detrend.
4.  Compute the RMS of the detrended residuals: F(n).
5.  Fit log F(n) = log(c) + α · log(n) over a grid of n.

Interpretation of α (the DFA scaling exponent):
- α < 0.5: anti-persistent / mean-reverting
- α ≈ 0.5: white noise / random walk
- α > 0.5: persistent / trending
- α > 1.0: non-stationary

α is asymptotically equivalent to the Hurst exponent for stationary
processes but is more robust on real-world series with trends.

References:
    Peng, C.-K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E.,
    & Goldberger, A. L. (1994). "Mosaic organization of DNA nucleotides."
    Physical Review E 49, 1685.
    Kantelhardt, J. W., et al. (2001). "Detecting long-range correlations
    with detrended fluctuation analysis." Physica A 295, 441-454.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DFAResult:
    """Output of :func:`dfa`."""

    alpha: float  # the DFA scaling exponent
    n_obs: int
    log_n: list[float]
    log_f: list[float]
    intercept: float
    r_squared: float  # how well the log-log relation fits
    interpretation: str  # "mean_reverting" / "random_walk" / "persistent" / "non_stationary"


def dfa(
    series: pd.Series,
    *,
    min_n: int = 8,
    max_n: int | None = None,
    poly_order: int = 1,
    n_grid: list[int] | None = None,
) -> DFAResult:
    """Detrended Fluctuation Analysis.

    Args:
        series: input level series.
        min_n: smallest segment length.
        max_n: largest segment length; default = ``floor(N/4)``.
        poly_order: polynomial detrend order (1 = linear; 2 = quadratic).
        n_grid: explicit override of segment lengths; default
            geometric grid (powers of 2-ish).

    Returns:
        :class:`DFAResult`.
    """
    s = series.dropna()
    N = len(s)
    if 4 * min_n > N:
        return DFAResult(
            alpha=float("nan"),
            n_obs=N,
            log_n=[],
            log_f=[],
            intercept=float("nan"),
            r_squared=float("nan"),
            interpretation="insufficient-data",
        )
    if max_n is None:
        max_n = N // 4

    # Integrate centered series.
    x = s.values - s.values.mean()
    Y = np.cumsum(x)

    # Geometric grid of n.
    if n_grid is None:
        grid: list[int] = []
        n = min_n
        while n <= max_n:
            grid.append(n)
            n = max(n + 1, int(n * 1.5))
    else:
        grid = [int(g) for g in n_grid if min_n <= g <= max_n]

    log_n: list[float] = []
    log_f: list[float] = []
    for n in grid:
        n_segments = N // n
        if n_segments < 4:
            continue
        flucts = []
        for i in range(n_segments):
            seg = Y[i * n : (i + 1) * n]
            t = np.arange(n)
            # Polynomial detrend
            coef = np.polyfit(t, seg, deg=poly_order)
            trend = np.polyval(coef, t)
            resid = seg - trend
            flucts.append(np.mean(resid * resid))
        if not flucts:
            continue
        F = np.sqrt(np.mean(flucts))
        if F > 0:
            log_n.append(log(n))
            log_f.append(log(F))

    if len(log_n) < 4:
        return DFAResult(
            alpha=float("nan"),
            n_obs=N,
            log_n=log_n,
            log_f=log_f,
            intercept=float("nan"),
            r_squared=float("nan"),
            interpretation="insufficient-data",
        )

    # OLS log F(n) = intercept + α · log(n)
    a = np.array(log_n)
    b = np.array(log_f)
    A = np.column_stack([np.ones_like(a), a])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    intercept, alpha = float(coef[0]), float(coef[1])
    pred = intercept + alpha * a
    ss_res = float(np.sum((b - pred) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if alpha > 1.0:
        interp = "non_stationary"
    elif alpha > 0.55:
        interp = "persistent"
    elif alpha < 0.45:
        interp = "mean_reverting"
    else:
        interp = "random_walk"

    return DFAResult(
        alpha=alpha,
        n_obs=N,
        log_n=log_n,
        log_f=log_f,
        intercept=intercept,
        r_squared=r2,
        interpretation=interp,
    )


__all__ = ["DFAResult", "dfa"]
