"""Diebold-Mariano test for equal forecast accuracy with HLN correction.

Diebold & Mariano (1995), "Comparing Predictive Accuracy", *Journal of
Business & Economic Statistics* 13:253-263, propose a test for the null
of equal forecast accuracy between two competing forecasts based on the
loss differential

.. math::

    d_t = L(e_{1,t}) - L(e_{2,t})

where ``e_{i,t}`` are the (signed) forecast errors of model i. Under
H0: ``E[d_t] = 0``. The DM statistic is

.. math::

    DM = \\frac{\\bar{d}}{\\sqrt{\\hat{V}_{HAC}(\\bar{d})}}

with HAC long-run variance estimated using Newey-West with bandwidth
tuned to the forecast horizon ``h``. Under H0 the DM statistic is
asymptotically standard normal.

Harvey, Leybourne & Newbold (1997), "Testing the equality of prediction
mean squared errors", *International Journal of Forecasting* 13:281-291,
note that the DM test over-rejects in finite samples for ``h > 1`` and
provide a small-sample correction:

.. math::

    DM^* = DM \\cdot \\sqrt{\\frac{T + 1 - 2h + h(h-1)/T}{T}}

with reference distribution ``Student-t_{T-1}`` rather than standard
normal. We report both the asymptotic-normal and HLN-corrected p-values.

References
----------
Diebold, F. X. & Mariano, R. S. (1995). "Comparing Predictive Accuracy",
    *Journal of Business & Economic Statistics* 13:253-263.
Harvey, D., Leybourne, S., & Newbold, P. (1997). "Testing the equality of
    prediction mean squared errors",
    *International Journal of Forecasting* 13:281-291.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

LossKind = Literal["MSE", "MAE", "Quad", "Abs"]


def _loss(e: np.ndarray, kind: LossKind) -> np.ndarray:
    if kind in ("MSE", "Quad"):
        return e * e
    if kind in ("MAE", "Abs"):
        return np.abs(e)
    raise ValueError(f"unknown loss {kind!r}")


def _newey_west_long_run_variance(x: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance of ``x_t``.

    Returns the asymptotic variance of ``mean(x)`` divided by ``T`` would
    give the variance of the mean. We return the long-run variance (not
    divided by T) so callers can decide.
    """
    n = x.size
    if n <= 1:
        return float("nan")
    centered = x - x.mean()
    gamma_0 = float((centered * centered).mean())
    long_run = gamma_0
    for ell in range(1, lag + 1):
        if ell >= n:
            break
        cov = float((centered[ell:] * centered[:-ell]).mean())
        weight = 1.0 - ell / (lag + 1)
        long_run += 2.0 * weight * cov
    return max(long_run, 0.0)


def diebold_mariano(
    forecast_errors_1: np.ndarray | list[float],
    forecast_errors_2: np.ndarray | list[float],
    *,
    h: int = 1,
    loss: LossKind = "MSE",
    hac_lag: int | None = None,
) -> dict[str, float | int | str]:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold finite-sample fix.

    Args:
        forecast_errors_1: errors ``e_{1,t} = y_t - hat{y}_{1,t}`` of model 1.
        forecast_errors_2: errors of model 2, same length as model 1.
        h: forecast horizon. Default 1. Drives the HAC bandwidth and
            the HLN correction.
        loss: loss function applied to errors before differencing. ``MSE``
            squared, ``MAE`` absolute. ``Quad`` and ``Abs`` are aliases.
        hac_lag: explicit Newey-West lag. Default ``h - 1`` (Diebold-Mariano
            recommendation for h-step forecasts).

    Returns:
        Dict with::

            dm_stat                  asymptotic DM statistic (~ N(0, 1))
            p_value                  two-sided asymptotic p-value
            dm_stat_hln              HLN small-sample-corrected stat
            p_value_hln              two-sided p from Student-t_{T-1}
            mean_loss_diff           sample mean of L(e_1) - L(e_2)
            prefer_model             1 / 2 / "tie" at 5% asymptotic level
            n_obs                    T
            hac_lag                  bandwidth used
            loss                     loss kind
            h                        horizon
    """
    e1 = np.asarray(forecast_errors_1, dtype=float).ravel()
    e2 = np.asarray(forecast_errors_2, dtype=float).ravel()
    if e1.size != e2.size:
        raise ValueError(f"length mismatch: {e1.size} vs {e2.size}")
    n = e1.size
    if n < 5:
        raise ValueError(f"need >= 5 observations, got {n}")
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")

    d = _loss(e1, loss) - _loss(e2, loss)
    d_mean = float(d.mean())

    if hac_lag is None:
        hac_lag = max(0, h - 1)

    long_run_var = _newey_west_long_run_variance(d, lag=hac_lag)
    var_of_mean = long_run_var / n
    if not math.isfinite(var_of_mean) or var_of_mean <= 0:
        dm_stat = 0.0
        p_val = 1.0
        dm_stat_hln = 0.0
        p_val_hln = 1.0
    else:
        dm_stat = d_mean / math.sqrt(var_of_mean)
        p_val = float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))

        # Harvey-Leybourne-Newbold finite-sample correction.
        hln_factor_sq = (n + 1 - 2 * h + h * (h - 1) / n) / n
        hln_factor = math.sqrt(max(hln_factor_sq, 0.0))
        dm_stat_hln = dm_stat * hln_factor
        # Student-t_{T-1} reference distribution for HLN.
        p_val_hln = float(2.0 * (1.0 - student_t.cdf(abs(dm_stat_hln), df=n - 1)))

    if p_val < 0.05:
        prefer = 2 if d_mean > 0 else 1
    else:
        prefer = "tie"

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_val),
        "dm_stat_hln": float(dm_stat_hln),
        "p_value_hln": float(p_val_hln),
        "mean_loss_diff": float(d_mean),
        "prefer_model": prefer,
        "n_obs": int(n),
        "hac_lag": int(hac_lag),
        "loss": str(loss),
        "h": int(h),
    }


__all__ = ["diebold_mariano"]
