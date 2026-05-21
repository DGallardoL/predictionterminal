"""Mean-reversion diagnostics: Hurst exponent + variance-ratio test.

Two complementary, model-free tests on a probability series:

1.  **Hurst exponent (R/S analysis, Mandelbrot 1968 / Peters 1994)**.
    Partition the series into non-overlapping subseries of length ``n``;
    for each subseries compute the rescaled range ``R/S`` (range of
    cumulative deviations from the mean, divided by the standard deviation).
    Average over subseries to get ``(R/S)_n``. Across multiple n, the
    expectation scales as

        E[(R/S)_n] ∝ n^H

    so the slope of ``log(R/S)_n`` against ``log n`` is the Hurst exponent.
    Interpretation:

        H < 0.5  → mean-reverting (anti-persistent)
        H ≈ 0.5  → random walk
        H > 0.5  → trending (persistent)

    **Caveat (Weron 2002)**: R/S has a well-documented upward bias at
    finite N. For N < 500 typical bias pushes H toward 0.55–0.60 even on
    a true random walk. Don't rely on Hurst alone for short series.

2.  **Variance-ratio test**. Define first differences
    ``Δp_t = p_t − p_{t-1}``. Under the random-walk null,

        Var(p_t − p_{t−q}) = q · Var(p_t − p_{t−1})

    so the variance ratio ``VR(q) = σ²(q) / σ²(1)`` should equal 1. The
    variance-ratio z-statistic with the *heteroscedasticity-consistent*
    variant tests H0: VR(q) = 1.

        VR(q) < 1  → mean reversion at horizon q
        VR(q) > 1  → momentum / persistent
        VR(q) ≈ 1  → fail to reject random walk

    VR is preferable to Hurst on short series because the asymptotic
    distribution of its z-stat is well-characterised.

Both tests assume *stationarity of differences*. Probability series near
resolution (p → 0 or 1) violate this; recommend running on the trailing
window before resolution and stating so.

References:
    Lo, A. & MacKinlay, A. (1988). "Stock Market Prices Do Not Follow
        Random Walks." Review of Financial Studies 1(1), 41-66.
    Mandelbrot, B. & Wallis, J. (1968). "Noah, Joseph, and Operational
        Hydrology." Water Resources Research 4(5), 909-918.
    Peters, E. (1994). *Fractal Market Analysis*.
    Weron, R. (2002). "Estimating Long-Range Dependence: Finite Sample
        Properties and Confidence Intervals." Physica A 312, 285-299.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class HurstResult:
    """Output of :func:`hurst_exponent`."""

    H: float
    n_obs: int
    log_n: list[float]
    log_rs: list[float]
    intercept: float
    r_squared: float
    interpretation: str  # "mean_reverting" / "random_walk" / "trending" / "insufficient-data"


@dataclass(frozen=True)
class VarianceRatioResult:
    """Output of :func:`variance_ratio_test`."""

    q: int
    n_obs: int
    vr: float
    z_stat: float
    p_value: float
    heteroscedastic: bool
    verdict: str  # "mean_reverting" / "random_walk" / "momentum" / "insufficient-data"


def _rs_for_n(x: np.ndarray, n: int) -> float:
    """Rescaled range averaged over non-overlapping subseries of length n."""
    T = len(x)
    n_subs = T // n
    if n_subs == 0:
        return float("nan")
    rs_vals: list[float] = []
    for i in range(n_subs):
        sub = x[i * n : (i + 1) * n]
        mean = sub.mean()
        cumdev = np.cumsum(sub - mean)
        R = float(cumdev.max() - cumdev.min())
        S = float(np.std(sub, ddof=1))
        if S > 0:
            rs_vals.append(R / S)
    if not rs_vals:
        return float("nan")
    return float(np.mean(rs_vals))


def hurst_exponent(
    series: pd.Series,
    *,
    min_n: int = 10,
    max_n: int | None = None,
    n_grid: Sequence[int] | None = None,
) -> HurstResult:
    """Hurst exponent via R/S analysis on the *first differences* of the input.

    Args:
        series: probability series (level, not differences).
        min_n: smallest sub-window length (default 10).
        max_n: largest sub-window length; default ``floor(N/2)``.
        n_grid: explicit override for the n-grid (defaults to powers of 2).

    Returns:
        :class:`HurstResult`.
    """
    s = series.dropna()
    diffs = s.diff().dropna().to_numpy(dtype=float)
    N = len(diffs)
    if 4 * min_n > N:
        return HurstResult(
            H=float("nan"),
            n_obs=N,
            log_n=[],
            log_rs=[],
            intercept=float("nan"),
            r_squared=float("nan"),
            interpretation="insufficient-data",
        )
    if max_n is None:
        max_n = N // 2

    if n_grid is None:
        # Powers-of-two grid (Peters 1994).
        grid: list[int] = []
        n = min_n
        while n <= max_n:
            grid.append(n)
            n *= 2
    else:
        grid = [int(g) for g in n_grid if min_n <= g <= max_n]

    log_n: list[float] = []
    log_rs: list[float] = []
    for n in grid:
        rs = _rs_for_n(diffs, n)
        if np.isfinite(rs) and rs > 0:
            log_n.append(log(n))
            log_rs.append(log(rs))

    if len(log_n) < 3:
        return HurstResult(
            H=float("nan"),
            n_obs=N,
            log_n=log_n,
            log_rs=log_rs,
            intercept=float("nan"),
            r_squared=float("nan"),
            interpretation="insufficient-data",
        )

    # OLS log(R/S) = intercept + H · log(n)
    a = np.array(log_n)
    b = np.array(log_rs)
    A = np.column_stack([np.ones_like(a), a])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    intercept, H = float(coef[0]), float(coef[1])
    pred = intercept + H * a
    ss_res = float(np.sum((b - pred) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if H < 0.45:
        interp = "mean_reverting"
    elif H > 0.55:
        interp = "trending"
    else:
        interp = "random_walk"

    return HurstResult(
        H=H,
        n_obs=N,
        log_n=log_n,
        log_rs=log_rs,
        intercept=intercept,
        r_squared=r2,
        interpretation=interp,
    )


def variance_ratio_test(
    series: pd.Series,
    *,
    q: int = 2,
    heteroscedastic: bool = True,
) -> VarianceRatioResult:
    """Variance-ratio test on first differences.

    Args:
        series: probability series (level, not differences).
        q: lag horizon ≥ 2.
        heteroscedastic: if True, use the heteroscedasticity-consistent
            z-stat (eq. 16 of the paper); if False, use the homoscedastic
            asymptotic variance (eq. 14).

    Returns:
        :class:`VarianceRatioResult`.
    """
    if q < 2:
        raise ValueError(f"variance_ratio_test: q must be ≥ 2, got {q}")
    s = series.dropna()
    diffs = s.diff().dropna().to_numpy(dtype=float)
    T = len(diffs)
    if 4 * q > T:
        return VarianceRatioResult(
            q=q,
            n_obs=T,
            vr=float("nan"),
            z_stat=float("nan"),
            p_value=float("nan"),
            heteroscedastic=heteroscedastic,
            verdict="insufficient-data",
        )

    mu = diffs.mean()
    # σ²(1): unbiased one-period variance.
    sig1 = float(np.sum((diffs - mu) ** 2) / (T - 1))
    # σ²(q): unbiased q-period variance using overlapping q-period sums.
    # X_t(q) = Σ_{j=0}^{q-1} (Δp_{t-j}) for t = q..T.
    cum = np.cumsum(np.insert(diffs, 0, 0.0))
    qsum = cum[q:] - cum[:-q]  # length T - q + 1
    m = q * (T - q + 1) * (1 - q / T)
    sig_q = float(np.sum((qsum - q * mu) ** 2) / m)
    if sig1 <= 0:
        return VarianceRatioResult(
            q=q,
            n_obs=T,
            vr=float("nan"),
            z_stat=float("nan"),
            p_value=float("nan"),
            heteroscedastic=heteroscedastic,
            verdict="insufficient-data",
        )
    vr = sig_q / sig1

    if heteroscedastic:
        # Heteroscedasticity-consistent variance estimator.
        # δ̂(j) = T · Σ_{t=j+1..T} (Δp_t − μ̂)² · (Δp_{t−j} − μ̂)² / [Σ (Δp − μ̂)²]²
        denom = np.sum((diffs - mu) ** 2) ** 2
        delta_sum = 0.0
        for j in range(1, q):
            d_j = diffs[j:] - mu
            d_lag = diffs[:-j] - mu
            num = T * float(np.sum((d_j**2) * (d_lag**2)))
            delta_j = num / denom if denom > 0 else 0.0
            weight = (2.0 * (q - j) / q) ** 2
            delta_sum += weight * delta_j
        theta = delta_sum
        z_stat = (vr - 1.0) / np.sqrt(theta / T) if theta > 0 else float("nan")
    else:
        # Homoscedastic version (eq. 14): VR ~ N(1, 2(2q−1)(q−1)/(3qT))
        var_homo = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * T)
        z_stat = (vr - 1.0) / np.sqrt(var_homo)

    p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat))) if np.isfinite(z_stat) else float("nan")
    if not np.isfinite(z_stat) or abs(z_stat) < 1.96:
        verdict = "random_walk"
    elif vr < 1:
        verdict = "mean_reverting"
    else:
        verdict = "momentum"

    return VarianceRatioResult(
        q=q,
        n_obs=T,
        vr=float(vr),
        z_stat=float(z_stat),
        p_value=float(p_value),
        heteroscedastic=heteroscedastic,
        verdict=verdict,
    )


__all__ = ["HurstResult", "VarianceRatioResult", "hurst_exponent", "variance_ratio_test"]
