"""AR(1) mean-reversion half-life estimation for factor universes.

Given a time series :math:`y_t`, fit the AR(1) regression in first-difference
form (Engle-Granger style):

.. math::

    \\Delta y_t = \\alpha + \\beta\\, y_{t-1} + \\varepsilon_t

Under stationary mean reversion (-2 < β < 0), the implied half-life of a unit
shock decaying back to the long-run mean is

.. math::

    h = -\\frac{\\ln 2}{\\ln(1 + \\beta)}\\,.

Interpretation of β:

- ``β >= 0``  -> no mean reversion (random walk or explosive); half-life = +inf
- ``-2 < β < 0`` -> stationary AR(1); positive finite half-life
- ``β <= -2`` -> oscillating, undefined; return NaN

The regression is fit with OLS on the differenced equation. The reported
``p_value`` is the two-sided t-test against ``β = 0`` (i.e. "is there any
mean reversion at all"); small p-values mean reject the unit-root-like
null in favour of mean reversion.

Edge cases:

- ``n_obs < 3`` after dropping NaN/aligning lag -> all-NaN result
- zero variance / all-constant -> all-NaN result
- explosive (β > 0): half-life is +inf, p_value still reported
- pure oscillation (β <= -2): half-life is NaN

Reference
---------
Ornstein-Uhlenbeck discretisation; Engle & Granger (1987); also widely
used in pairs-trading literature (e.g. Chan, *Algorithmic Trading*, 2013,
§7.1 "Half-life of mean reversion").
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "estimate_half_life",
    "half_life_universe",
]


_NAN_RESULT: dict[str, float] = {
    "half_life_days": float("nan"),
    "ar1_coef": float("nan"),
    "p_value": float("nan"),
    "n_obs": 0,
}


def _half_life_from_beta(beta: float) -> float:
    """Convert AR(1) differenced-form β to half-life in periods.

    - β >= 0   -> +inf (no reversion / explosive)
    - -2 < β < 0 -> -log(2) / log(1 + β)
    - β <= -2  -> NaN (oscillating / undefined)
    """
    if not math.isfinite(beta):
        return float("nan")
    if beta >= 0.0:
        return float("inf")
    if beta <= -2.0:
        return float("nan")
    one_plus = 1.0 + beta
    # one_plus is in (-1, 1) for β in (-2, 0); but excludes 0 and 1.
    # log(one_plus) is negative for one_plus in (0, 1); positive imaginary
    # would occur for negative one_plus, so guard for β in (-2, -1].
    if one_plus <= 0.0:
        # β in (-2, -1] -> 1 + β in (-1, 0]; log undefined / oscillating.
        return float("nan")
    return float(-math.log(2.0) / math.log(one_plus))


def estimate_half_life(series: pd.Series) -> dict[str, Any]:
    """Estimate AR(1) mean-reversion half-life for a single series.

    Fits ``Δy_t = α + β·y_{t-1} + ε`` by OLS and converts β to half-life.

    Args:
        series: 1-D time series of (typically) prices, log-prices, spreads,
            or any quantity for which mean reversion is meaningful. NaN
            values are dropped before alignment.

    Returns:
        Dict with keys ``half_life_days`` (float), ``ar1_coef`` (β),
        ``p_value`` (two-sided t-test on β), ``n_obs`` (sample size after
        alignment). Degenerate inputs return an all-NaN result with
        ``n_obs = 0``.
    """
    if series is None or len(series) == 0:
        return dict(_NAN_RESULT)

    # Coerce to numeric, drop NaN, then differential alignment.
    try:
        s = pd.to_numeric(pd.Series(series).copy(), errors="coerce").dropna()
    except (TypeError, ValueError):
        return dict(_NAN_RESULT)

    # Need at least 3 obs total so that after the lag we have >= 2 rows for OLS.
    if len(s) < 3:
        return dict(_NAN_RESULT)

    s = s.astype(float).reset_index(drop=True)
    y_lag = s.iloc[:-1].to_numpy()
    dy = np.diff(s.to_numpy())
    n_obs = int(dy.size)

    if n_obs < 2:
        return dict(_NAN_RESULT)

    # Reject all-constant inputs (zero variance) -> regression undefined.
    # Use a relative tolerance to catch FP-rounded "constant" series like
    # ``[3.14] * 100`` whose numpy std is ~1e-15 rather than exactly 0.
    s_arr = s.to_numpy()
    s_scale = max(abs(float(np.mean(s_arr))), 1.0)
    if float(np.std(s_arr, ddof=0)) <= 1e-12 * s_scale:
        return {**_NAN_RESULT, "n_obs": n_obs}

    # Reject if the lag column itself has no variance (also degenerate).
    lag_scale = max(abs(float(np.mean(y_lag))), 1.0)
    if float(np.std(y_lag, ddof=0)) <= 1e-12 * lag_scale:
        return {**_NAN_RESULT, "n_obs": n_obs}

    # OLS on Δy_t = α + β·y_{t-1}. Use statsmodels for clean p-value.
    # statsmodels is already a hard dep of the project (see PLAN.md).
    try:
        import statsmodels.api as sm

        x = sm.add_constant(y_lag, has_constant="add")
        model = sm.OLS(dy, x, missing="drop").fit()
        # Parameters: [const, beta]
        beta = float(model.params[1])
        # p_value is two-sided t-test against 0 (statsmodels default).
        p_value = float(model.pvalues[1])
    except Exception:
        # Fall back to NaN if statsmodels chokes on something pathological.
        return {**_NAN_RESULT, "n_obs": n_obs}

    half_life = _half_life_from_beta(beta)

    # Sanitise non-finite p-value (e.g. zero residual variance).
    if not math.isfinite(p_value):
        p_value = float("nan")

    return {
        "half_life_days": float(half_life),
        "ar1_coef": float(beta),
        "p_value": float(p_value),
        "n_obs": int(n_obs),
    }


def half_life_universe(series_panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate AR(1) half-life for every column of a panel.

    Args:
        series_panel: wide DataFrame where each column is one factor /
            slug and rows are time observations. NaN handling is per-column
            (each column is dropped-NaN independently before fitting).

    Returns:
        DataFrame with one row per column of ``series_panel``, columns:
        ``slug``, ``half_life_days``, ``ar1_coef``, ``p_value``, ``n_obs``.
        The slug column carries the original DataFrame column label.
        Returned in the same column order as ``series_panel.columns``.
    """
    if series_panel is None or len(series_panel.columns) == 0:
        return pd.DataFrame(columns=["slug", "half_life_days", "ar1_coef", "p_value", "n_obs"])

    rows: list[dict[str, Any]] = []
    for col in series_panel.columns:
        res = estimate_half_life(series_panel[col])
        rows.append(
            {
                "slug": col,
                "half_life_days": res["half_life_days"],
                "ar1_coef": res["ar1_coef"],
                "p_value": res["p_value"],
                "n_obs": res["n_obs"],
            }
        )

    out = pd.DataFrame(
        rows,
        columns=["slug", "half_life_days", "ar1_coef", "p_value", "n_obs"],
    )
    # Preserve numeric dtype on n_obs even when all-NaN edge case occurs.
    out["n_obs"] = out["n_obs"].astype(int)
    return out
