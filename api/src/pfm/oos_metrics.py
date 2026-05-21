"""Out-of-sample R-squared (Campbell & Thompson 2008) and Clark-West.

In-sample R^2 is famously optimistic for return-prediction regressions.
Campbell & Thompson (2008), "Predicting Excess Stock Returns Out of
Sample: Can Anything Beat the Historical Average?", *Review of Financial
Studies* 21:1509-1531, define an out-of-sample R^2 that compares the
mean squared forecast error of a candidate model to that of a simple
recursive-mean (historical-average) baseline:

.. math::

    R^2_{OOS} = 1 - \\frac{\\sum_t (y_t - \\hat{y}_t^{model})^2}
                          {\\sum_t (y_t - \\hat{y}_t^{baseline})^2}

Positive ``R^2_{OOS}`` means the model beats the baseline OOS. Negative
values are informative — they warn that the prediction adds noise.

For *nested* model comparisons (the common case where the model nests
the baseline as a special case, e.g. baseline = constant, model = constant +
factor) the standard ``DM`` test is biased toward the baseline because
the squared-error difference has nonzero mean under H0. Clark & West
(2007), "Approximately Normal Tests for Equal Predictive Accuracy in
Nested Models", *Journal of Econometrics* 138:291-311, derive a corrected
test statistic that adjusts for the bias term. Their adjusted-MSE
quantity is

.. math::

    \\hat{f}_t = (y_t - \\hat{y}_t^{base})^2 -
                 \\big[ (y_t - \\hat{y}_t^{model})^2 -
                        (\\hat{y}_t^{base} - \\hat{y}_t^{model})^2 \\big]

and the CW test statistic is the ``t``-statistic of ``mean(f) > 0`` with
HAC standard error.

References
----------
Campbell, J. Y. & Thompson, S. B. (2008). "Predicting Excess Stock Returns
    Out of Sample: Can Anything Beat the Historical Average?",
    *Review of Financial Studies* 21:1509-1531.
Clark, T. E. & West, K. D. (2007). "Approximately Normal Tests for Equal
    Predictive Accuracy in Nested Models", *Journal of Econometrics*
    138:291-311.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _newey_west_var(x: np.ndarray, lag: int) -> float:
    """Newey-West HAC variance of the sample mean of ``x``.

    Returns ``Var(mean(x))`` so the corresponding standard error is
    ``sqrt(Var)``. Uses the Bartlett kernel weights ``1 - l/(L+1)``.
    """
    n = x.size
    if n <= 1:
        return float("nan")
    centered = x - x.mean()
    # Lag 0 contribution is the sample variance (without ddof).
    gamma = [float((centered * centered).mean())]
    for ell in range(1, lag + 1):
        if ell >= n:
            break
        cov = float((centered[ell:] * centered[:-ell]).mean())
        gamma.append(cov)
    weights = [1.0 - ell / (lag + 1) for ell in range(len(gamma))]
    long_run = gamma[0] + 2.0 * sum(w * g for w, g in zip(weights[1:], gamma[1:], strict=False))
    long_run = max(long_run, 0.0)
    return long_run / n


def oos_r_squared_campbell_thompson(
    y_actual: np.ndarray | list[float],
    y_pred_model: np.ndarray | list[float],
    y_pred_baseline: np.ndarray | list[float],
    *,
    nested: bool = True,
    hac_lag: int | None = None,
) -> dict[str, float]:
    """Campbell-Thompson out-of-sample R^2 plus the Clark-West nested-test stat.

    Args:
        y_actual: realised values y_t.
        y_pred_model: candidate model forecasts \\hat{y}^{model}_t.
        y_pred_baseline: baseline forecasts \\hat{y}^{base}_t (typically
            the recursive sample mean of y up to t-1).
        nested: if True (default) compute Clark-West adjusted statistic;
            otherwise compute the plain Diebold-Mariano statistic on the
            squared-error differences.
        hac_lag: Newey-West truncation lag. Default = floor(T^{1/3}).

    Returns:
        Dict containing:

        - ``r_squared_oos``: 1 - SSE(model) / SSE(base).
        - ``mse_model``, ``mse_baseline``: realised MSEs.
        - ``n_obs``: sample size T.
        - ``hac_t_stat_clark_west``: Clark-West / DM statistic, asymptotic N(0,1).
        - ``hac_p_value``: one-sided p-value (H0: equal accuracy, H1: model better).
        - ``hac_lag``: lag used.
        - ``model_beats_baseline``: bool, True iff R^2_OOS > 0.
    """
    y = np.asarray(y_actual, dtype=float).ravel()
    m = np.asarray(y_pred_model, dtype=float).ravel()
    b = np.asarray(y_pred_baseline, dtype=float).ravel()
    if not (y.size == m.size == b.size):
        raise ValueError(f"length mismatch: y={y.size}, model={m.size}, baseline={b.size}")
    n = y.size
    if n < 5:
        raise ValueError(f"need >= 5 observations, got {n}")

    err_model = y - m
    err_base = y - b
    sse_model = float(np.sum(err_model * err_model))
    sse_base = float(np.sum(err_base * err_base))
    mse_model = sse_model / n
    mse_base = sse_base / n
    if sse_base <= 0:
        r2_oos = float("nan")
    else:
        r2_oos = 1.0 - sse_model / sse_base

    # Clark-West / DM loss-difference series.
    if nested:
        f_t = (err_base * err_base) - ((err_model * err_model) - (b - m) * (b - m))
    else:
        f_t = (err_base * err_base) - (err_model * err_model)

    if hac_lag is None:
        hac_lag = int(math.floor(n ** (1.0 / 3.0)))
        hac_lag = max(0, min(hac_lag, n - 1))

    f_mean = float(f_t.mean())
    var_mean = _newey_west_var(f_t, lag=hac_lag)
    if not math.isfinite(var_mean) or var_mean <= 0:
        t_stat = 0.0
        p_val = 0.5
    else:
        t_stat = f_mean / math.sqrt(var_mean)
        p_val = float(1.0 - norm.cdf(t_stat))

    return {
        "r_squared_oos": float(r2_oos),
        "mse_model": float(mse_model),
        "mse_baseline": float(mse_base),
        "n_obs": int(n),
        "hac_t_stat_clark_west": float(t_stat),
        "hac_p_value": float(p_val),
        "hac_lag": int(hac_lag),
        "model_beats_baseline": bool(math.isfinite(r2_oos) and r2_oos > 0.0),
    }


__all__ = ["oos_r_squared_campbell_thompson"]
