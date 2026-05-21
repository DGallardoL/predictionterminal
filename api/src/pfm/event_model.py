"""Multi-factor regression of one event probability on N other events.

Distinct from :mod:`pfm.model` (which regresses *equity returns* on factor
Δlogits): here both the target ``y`` and the regressors ``X`` are
*probability series in [0, 1]*. The use case is "explain market A's
probability path as a linear combination of related markets":

    P_A_t = α + Σ_i β_i · P_B_i,t + ε_t          (HAC SE, lag = 5 default)

Interpretations:

*   β_i is the **co-move sensitivity** of A to factor i, holding the
    others fixed. Under jointly-stationary marginals it converges to the
    partial conditional risk difference.
*   The fitted ``ŷ`` is the **other-market-implied** probability of A.
    The residual is the slice of A's price *not explained* by the other
    markets — the "idiosyncratic" component that an analyst should
    inspect for news / sentiment / mispricing.

VIF, condition number, and HAC-CI per coefficient are reported so the
user can spot collinearity and unidentifiable factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass(frozen=True)
class EventCoefficient:
    """Per-factor HAC-OLS coefficient bundle."""

    factor_id: str
    beta: float
    hac_se: float
    t_stat: float
    p_value: float
    ci_lo: float
    ci_hi: float
    vif: float


@dataclass(frozen=True)
class EventModelResult:
    """Output of :func:`event_model`.

    Attributes:
        target_id: identifier for the dependent series.
        factor_ids: identifiers for the regressors, in fit order.
        n_obs: jointly-observed sample size after dropna.
        intercept: α.
        intercept_se: HAC standard error on the intercept.
        coefficients: list of :class:`EventCoefficient`, one per factor.
        r_squared: ordinary R².
        r_squared_adj: adjusted R² for the number of regressors.
        f_statistic: model F-stat.
        f_pvalue: F-stat p-value.
        condition_number: ``cond(X)`` — large ⇒ ill-posed.
        hac_lag: Newey-West lag truncation used.
        predicted: per-date ŷ_t series.
        residuals: per-date ε_t series.
        actual: per-date observed P_A_t series (passthrough).
    """

    target_id: str
    factor_ids: list[str]
    n_obs: int
    intercept: float
    intercept_se: float
    coefficients: list[EventCoefficient]
    r_squared: float
    r_squared_adj: float
    f_statistic: float
    f_pvalue: float
    condition_number: float
    hac_lag: int
    predicted: pd.Series
    residuals: pd.Series
    actual: pd.Series


def _vif_safe(X_with_const: np.ndarray, idx: int) -> float:
    try:
        return float(variance_inflation_factor(X_with_const, idx))
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def event_model(
    target: pd.Series,
    factors: pd.DataFrame,
    *,
    target_id: str = "target",
    hac_lag: int = 5,
) -> EventModelResult:
    """Fit ``target ~ α + Σ β_i · factors[i] + ε`` with HAC SEs.

    Args:
        target: probability series for the event being explained.
        factors: DataFrame whose columns are the explanatory probability
            series. Each column name will appear as ``factor_id`` in the
            output. Must NOT contain a constant column.
        target_id: identifier for the target (used only for output labelling).
        hac_lag: Newey-West lag (5 is reasonable for daily probability series;
            raise to 10–20 for very autocorrelated macro markets).

    Returns:
        :class:`EventModelResult`.

    Raises:
        ValueError: if alignment leaves <max(20, k+5) rows, or if any
            factor column has zero variance.
    """
    if factors.shape[1] == 0:
        raise ValueError("event_model requires at least one factor column")

    aligned = pd.concat({"y": target, **{c: factors[c] for c in factors.columns}}, axis=1).dropna()
    n = len(aligned)
    k = factors.shape[1]
    if n < max(20, k + 5):
        raise ValueError(
            f"event_model: only {n} jointly-observed dates for {k} factors "
            f"(need ≥ max(20, k+5) = {max(20, k + 5)})"
        )
    y = aligned["y"]
    X = aligned[list(factors.columns)]

    # Reject zero-variance columns (β unidentified, statsmodels silently drops).
    zero_var = [c for c in X.columns if float(np.var(X[c].values)) < 1e-12]
    if zero_var:
        raise ValueError(f"event_model: factor(s) {zero_var!r} have zero variance over the window")

    Xc = sm.add_constant(X)
    res = sm.OLS(y.values, Xc.values).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})

    columns = list(Xc.columns)
    params = res.params
    bse = res.bse
    tvals = res.tvalues
    pvals = res.pvalues
    ci = res.conf_int(alpha=0.05)

    intercept = float(params[0])
    intercept_se = float(bse[0])

    coefficients: list[EventCoefficient] = []
    Xc_arr = Xc.values
    for i, col in enumerate(columns[1:], start=1):
        coefficients.append(
            EventCoefficient(
                factor_id=col,
                beta=float(params[i]),
                hac_se=float(bse[i]),
                t_stat=float(tvals[i]),
                p_value=float(pvals[i]),
                ci_lo=float(ci[i, 0]),
                ci_hi=float(ci[i, 1]),
                vif=_vif_safe(Xc_arr, i),
            )
        )

    predicted = pd.Series(res.fittedvalues, index=aligned.index, name="predicted")
    residuals = pd.Series(res.resid, index=aligned.index, name="residual")

    # Condition number on the design matrix without the intercept (the
    # statsmodels' built-in includes the intercept column which inflates).
    try:
        sv = np.linalg.svd(X.values, compute_uv=False)
        cond_num = float(sv.max() / sv.min()) if sv.min() > 0 else float("inf")
    except np.linalg.LinAlgError:
        cond_num = float("inf")

    return EventModelResult(
        target_id=target_id,
        factor_ids=list(X.columns),
        n_obs=n,
        intercept=intercept,
        intercept_se=intercept_se,
        coefficients=coefficients,
        r_squared=float(res.rsquared),
        r_squared_adj=float(res.rsquared_adj),
        f_statistic=float(res.fvalue) if res.fvalue is not None else float("nan"),
        f_pvalue=float(res.f_pvalue) if res.f_pvalue is not None else float("nan"),
        condition_number=cond_num,
        hac_lag=hac_lag,
        predicted=predicted,
        residuals=residuals,
        actual=y.rename(target_id),
    )


__all__ = ["EventCoefficient", "EventModelResult", "event_model"]
