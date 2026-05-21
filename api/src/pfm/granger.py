"""Bivariate Granger causality between event probability series.

The Granger (1969) test asks: *do past values of B help predict A beyond
what past values of A alone provide?* Concretely, for each candidate lag
``L``, we run two regressions on series ``A``:

    Restricted:   A_t = α + Σ_{l=1}^L a_l · A_{t−l} + ε_t
    Unrestricted: A_t = α + Σ_{l=1}^L a_l · A_{t−l} + Σ_{l=1}^L b_l · B_{t−l} + ε_t

and compare via the F-statistic

    F = ((SSR_R − SSR_U) / L) / (SSR_U / (T − 2L − 1))

Reject H0 (B does not Granger-cause A) when F's p-value < α.

For prediction-market probability series, this is a *correlation-of-news*
test: does B's news lead A's news? Useful when picking the long leg of an
event-driven trade — go long the *follower*, hedge with the *leader*.

We use ``statsmodels.tsa.stattools.grangercausalitytests`` and surface
both the SSR-F and chi² p-values, plus the lag with minimum p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# statsmodels' verbose prints get squelched by passing verbose=False.


@dataclass(frozen=True)
class GrangerLagResult:
    lag: int
    ssr_f_stat: float
    ssr_f_pvalue: float
    ssr_chi2_pvalue: float


@dataclass(frozen=True)
class GrangerCausalityResult:
    a_id: str
    b_id: str
    n_obs: int
    direction: str  # "B_causes_A" | "A_causes_B" | "bidirectional" | "neither"
    best_lag_b_to_a: int | None
    best_pvalue_b_to_a: float | None
    best_lag_a_to_b: int | None
    best_pvalue_a_to_b: float | None
    lags: list[GrangerLagResult]  # B → A direction
    lags_reverse: list[GrangerLagResult]  # A → B direction


def granger_test(
    a: pd.Series,
    b: pd.Series,
    *,
    a_id: str = "a",
    b_id: str = "b",
    max_lag: int = 5,
    alpha: float = 0.05,
) -> GrangerCausalityResult:
    """Run bivariate Granger causality in both directions.

    Args:
        a, b: probability series.
        max_lag: test lags 1..max_lag.
        alpha: significance threshold for the verdict.

    Returns:
        :class:`GrangerCausalityResult` with both directions.

    Raises:
        ValueError: if joint sample is too short for the requested max_lag.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    aligned = pd.concat({a_id: a, b_id: b}, axis=1).dropna()
    n = len(aligned)
    # statsmodels requires T > 3·max_lag+1; be generous.
    if n < max(20, 4 * max_lag + 2):
        raise ValueError(
            f"granger_test: need ≥ max(20, 4·max_lag+2) = {max(20, 4 * max_lag + 2)} bars, got {n}"
        )

    # statsmodels expects [Y, X] columns for "X granger-causes Y"
    # B→A: Y=A, X=B.
    arr_b_to_a = aligned[[a_id, b_id]].to_numpy()
    arr_a_to_b = aligned[[b_id, a_id]].to_numpy()

    from statsmodels.tools.sm_exceptions import InfeasibleTestError

    try:
        out_b_to_a = grangercausalitytests(arr_b_to_a, maxlag=max_lag, verbose=False)
    except InfeasibleTestError:
        # Perfect-fit VAR (e.g. one series is a deterministic linear function
        # of the other). The test is undefined; report "neither".
        return GrangerCausalityResult(
            a_id=a_id,
            b_id=b_id,
            n_obs=n,
            direction="neither",
            best_lag_b_to_a=None,
            best_pvalue_b_to_a=None,
            best_lag_a_to_b=None,
            best_pvalue_a_to_b=None,
            lags=[],
            lags_reverse=[],
        )
    try:
        out_a_to_b = grangercausalitytests(arr_a_to_b, maxlag=max_lag, verbose=False)
    except InfeasibleTestError:
        return GrangerCausalityResult(
            a_id=a_id,
            b_id=b_id,
            n_obs=n,
            direction="neither",
            best_lag_b_to_a=None,
            best_pvalue_b_to_a=None,
            best_lag_a_to_b=None,
            best_pvalue_a_to_b=None,
            lags=[],
            lags_reverse=[],
        )

    def _to_lag_results(out: dict) -> list[GrangerLagResult]:
        rows: list[GrangerLagResult] = []
        for lag, (test, _models) in out.items():
            ssr_f = test.get("ssr_ftest", (float("nan"),) * 4)
            ssr_chi2 = test.get("ssr_chi2test", (float("nan"),) * 3)
            rows.append(
                GrangerLagResult(
                    lag=int(lag),
                    ssr_f_stat=float(ssr_f[0]),
                    ssr_f_pvalue=float(ssr_f[1]),
                    ssr_chi2_pvalue=float(ssr_chi2[1]),
                )
            )
        return rows

    lags_b_to_a = _to_lag_results(out_b_to_a)
    lags_a_to_b = _to_lag_results(out_a_to_b)

    def _best(rows: list[GrangerLagResult]) -> tuple[int | None, float | None]:
        if not rows:
            return None, None
        best = min(rows, key=lambda r: r.ssr_f_pvalue if not np.isnan(r.ssr_f_pvalue) else 1.0)
        return best.lag, best.ssr_f_pvalue

    lag_ba, p_ba = _best(lags_b_to_a)
    lag_ab, p_ab = _best(lags_a_to_b)

    b_causes_a = p_ba is not None and p_ba < alpha
    a_causes_b = p_ab is not None and p_ab < alpha
    if b_causes_a and a_causes_b:
        direction = "bidirectional"
    elif b_causes_a:
        direction = "B_causes_A"
    elif a_causes_b:
        direction = "A_causes_B"
    else:
        direction = "neither"

    return GrangerCausalityResult(
        a_id=a_id,
        b_id=b_id,
        n_obs=n,
        direction=direction,
        best_lag_b_to_a=lag_ba,
        best_pvalue_b_to_a=p_ba,
        best_lag_a_to_b=lag_ab,
        best_pvalue_a_to_b=p_ab,
        lags=lags_b_to_a,
        lags_reverse=lags_a_to_b,
    )


__all__ = ["GrangerCausalityResult", "GrangerLagResult", "granger_test"]
