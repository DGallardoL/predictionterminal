"""Probabilistic strategy detectors for prediction-market series.

This module operates on the *probability* level: each input is a daily series
of YES-prices in [0, 1] interpreted as the market-implied probability of the
underlying event on that date.

Implements three rigorous tools (no naive arbitrage detection — see the
``strategies.md`` writeup for why naive Σpᵢ-≠-1 arbitrage is rarely tradeable
under LMSR + non-zero spreads):

1.  **Logical implication test** — :func:`implication_test`.
    Given events A, B with the *known* logical relation A ⇒ B (i.e. A is
    strictly more specific), monotonicity of probability requires
    P(A) ≤ P(B) on every date. We flag any date where the empirical
    YES-prices reverse this and quantify the gap on both the linear and
    logit scales. Tolerance accounts for spread / quantization noise.

2.  **Conditional probability via co-move regression** — :func:`conditional_regression`.
    Regress P_A on P_B with HAC standard errors. Under jointly-stationary
    series, the slope β satisfies β = ρ · σ_A / σ_B where ρ is the
    Pearson correlation of the two probability series; under additional
    binarisation it converges to P(A|B=1) − P(A|B=0). We report β, its
    HAC-CI, and the empirical conditional means computed by binning
    P_B at 0.5.

3.  **Fréchet-Hoeffding bounds** — :func:`frechet_bounds`.
    For any bivariate distribution with marginals P(A), P(B):
        max(0, P(A) + P(B) − 1) ≤ P(A ∩ B) ≤ min(P(A), P(B)).
    These are *distribution-free*. The band width
        min(P(A), P(B)) − max(0, P(A) + P(B) − 1)
    measures how much joint information the marginals already pin down.

All functions are pure pandas/numpy and seed-deterministic where stochastic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ───────────────────────── numeric utilities ──────────────────────────


def _logit(p: pd.Series, *, eps: float = 1e-6) -> pd.Series:
    """Log-odds of a probability series, clipped to (eps, 1−eps)."""
    q = p.clip(lower=eps, upper=1.0 - eps)
    return np.log(q / (1.0 - q))


def _align_series(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join two date-indexed series and drop NaNs."""
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    return df["a"], df["b"]


# ─────────────────────────── implication ──────────────────────────────


@dataclass(frozen=True)
class ImplicationResult:
    """Output of :func:`implication_test`.

    Attributes:
        n_obs: number of jointly-observed dates after alignment.
        violation_dates: dates where ``P(A) − P(B) > tolerance``.
        max_gap: maximum positive ``P(A) − P(B)`` (linear scale).
        mean_gap: mean of ``P(A) − P(B)`` over all dates (signed).
        gap_series: full per-date gap series ``P(A) − P(B)``.
        logit_gap_series: per-date ``logit(P(A)) − logit(P(B))``.
        verdict: ``"consistent"`` / ``"borderline"`` / ``"violated"``.
    """

    n_obs: int
    violation_dates: list[date]
    max_gap: float
    mean_gap: float
    gap_series: pd.Series
    logit_gap_series: pd.Series
    verdict: str


def implication_test(
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    tolerance: float = 0.02,
    n_violations_borderline: int = 1,
    n_violations_violated: int = 5,
) -> ImplicationResult:
    """Test the logical-implication invariant ``A ⇒ B`` ⇒ ``P(A) ≤ P(B)``.

    A violation on a single date may be quantization noise or a momentary
    spread artefact; a *persistent* violation across many dates is evidence
    of a logical-pricing error.

    Args:
        p_a: daily YES-price series for event A (the *more specific* event).
        p_b: daily YES-price series for event B (the *broader* event).
        tolerance: ignore gaps with ``P(A) − P(B) ≤ tolerance``. Default 2%
            corresponds to typical Polymarket bid-ask half-spread.
        n_violations_borderline: lower threshold (in number of violating
            dates) for the ``"borderline"`` verdict.
        n_violations_violated: lower threshold for the ``"violated"`` verdict.

    Returns:
        :class:`ImplicationResult` with per-date diagnostics.
    """
    a, b = _align_series(p_a, p_b)
    if a.empty:
        return ImplicationResult(
            n_obs=0,
            violation_dates=[],
            max_gap=float("nan"),
            mean_gap=float("nan"),
            gap_series=pd.Series(dtype=float),
            logit_gap_series=pd.Series(dtype=float),
            verdict="insufficient-data",
        )

    gap = (a - b).rename("gap")
    logit_gap = (_logit(a) - _logit(b)).rename("logit_gap")
    viol_mask = gap > tolerance
    viol_dates = [ts.date() if hasattr(ts, "date") else ts for ts in gap.index[viol_mask]]
    n_viol = int(viol_mask.sum())

    if n_viol >= n_violations_violated:
        verdict = "violated"
    elif n_viol >= n_violations_borderline:
        verdict = "borderline"
    else:
        verdict = "consistent"

    return ImplicationResult(
        n_obs=len(a),
        violation_dates=viol_dates,
        max_gap=float(gap.max()),
        mean_gap=float(gap.mean()),
        gap_series=gap,
        logit_gap_series=logit_gap,
        verdict=verdict,
    )


# ───────────────────── conditional via co-move ────────────────────────


@dataclass(frozen=True)
class ConditionalRegressionResult:
    """Output of :func:`conditional_regression`.

    Attributes:
        n_obs: number of jointly-observed dates.
        beta: OLS slope β in P_A ~ α + β · P_B + ε. Interpretable as the
            change in P(A) for a 1-unit increase in P(B).
        beta_hac_se: HAC (Newey-West) standard error of β. Default lag = 5.
        beta_ci_lo: lower 95% CI on β.
        beta_ci_hi: upper 95% CI on β.
        intercept: α from the same regression.
        r_squared: R² of the regression.
        cond_mean_when_b_high: empirical mean of P_A on days where P_B > 0.5.
        cond_mean_when_b_low: empirical mean of P_A on days where P_B ≤ 0.5.
        n_b_high: count of days with P_B > 0.5.
    """

    n_obs: int
    beta: float
    beta_hac_se: float
    beta_ci_lo: float
    beta_ci_hi: float
    intercept: float
    r_squared: float
    cond_mean_when_b_high: float
    cond_mean_when_b_low: float
    n_b_high: int


def conditional_regression(
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    hac_lag: int = 5,
) -> ConditionalRegressionResult:
    """Estimate the directional dependence of P(A) on P(B) by HAC-OLS.

    Notes on interpretation:

    *   If both series are stationary, β is the linear projection coefficient.
        Under the additional binarisation A_t = 1{P_A > 0.5} and similarly
        for B, β converges to P(A=1|B=1) − P(A=1|B=0).

    *   HAC SEs are appropriate because daily probability series are
        autocorrelated by construction (overlapping news effects).

    Args:
        p_a: daily YES-price series for A.
        p_b: daily YES-price series for B.
        hac_lag: Newey-West lag truncation parameter.

    Returns:
        :class:`ConditionalRegressionResult`.
    """
    a, b = _align_series(p_a, p_b)
    if len(a) < 10:
        raise ValueError(f"need ≥10 jointly-observed dates, got {len(a)}")
    # Reject zero-variance predictors: with constant P_B, β is unidentified
    # and statsmodels silently drops the column (returning a 1-parameter fit).
    b_var = float(np.var(b.values))
    if b_var < 1e-12:
        raise ValueError(
            "conditioning series P_B has effectively zero variance over the window — "
            "cannot estimate β"
        )

    X = sm.add_constant(b.values)
    model = sm.OLS(a.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})
    beta = float(model.params[1])
    se = float(model.bse[1])
    ci = model.conf_int(alpha=0.05)
    ci_lo, ci_hi = float(ci[1, 0]), float(ci[1, 1])

    high_mask = b > 0.5
    n_high = int(high_mask.sum())
    cond_high = float(a[high_mask].mean()) if n_high else float("nan")
    cond_low = float(a[~high_mask].mean()) if (len(a) - n_high) else float("nan")

    return ConditionalRegressionResult(
        n_obs=len(a),
        beta=beta,
        beta_hac_se=se,
        beta_ci_lo=ci_lo,
        beta_ci_hi=ci_hi,
        intercept=float(model.params[0]),
        r_squared=float(model.rsquared),
        cond_mean_when_b_high=cond_high,
        cond_mean_when_b_low=cond_low,
        n_b_high=n_high,
    )


# ───────────────────────── Fréchet bounds ─────────────────────────────


@dataclass(frozen=True)
class FrechetBoundsResult:
    """Output of :func:`frechet_bounds`.

    Attributes:
        n_obs: number of jointly-observed dates.
        lower: per-date series ``max(0, P(A) + P(B) − 1)``.
        upper: per-date series ``min(P(A), P(B))``.
        width: per-date band width (upper − lower).
        independence_joint: per-date ``P(A) · P(B)`` (for reference; the
            joint under independence always lies inside the bounds).
        mean_lower: mean of the lower-bound series.
        mean_upper: mean of the upper-bound series.
        mean_width: mean band width.
    """

    n_obs: int
    lower: pd.Series
    upper: pd.Series
    width: pd.Series
    independence_joint: pd.Series
    mean_lower: float
    mean_upper: float
    mean_width: float


def frechet_bounds(p_a: pd.Series, p_b: pd.Series) -> FrechetBoundsResult:
    """Compute per-date Fréchet-Hoeffding bounds on the joint ``P(A ∩ B)``.

    No assumption on the dependence structure — these bounds hold for *any*
    joint distribution with the given marginals. Independence sits inside
    the band and is reported for reference.

    Args:
        p_a: daily YES-price series for A.
        p_b: daily YES-price series for B.

    Returns:
        :class:`FrechetBoundsResult`.
    """
    a, b = _align_series(p_a, p_b)
    if a.empty:
        empty = pd.Series(dtype=float)
        return FrechetBoundsResult(
            n_obs=0,
            lower=empty,
            upper=empty,
            width=empty,
            independence_joint=empty,
            mean_lower=float("nan"),
            mean_upper=float("nan"),
            mean_width=float("nan"),
        )

    lower = (a + b - 1.0).clip(lower=0.0).rename("lower")
    upper = pd.concat([a.rename("a"), b.rename("b")], axis=1).min(axis=1).rename("upper")
    width = (upper - lower).rename("width")
    indep = (a * b).rename("indep")

    return FrechetBoundsResult(
        n_obs=len(a),
        lower=lower,
        upper=upper,
        width=width,
        independence_joint=indep,
        mean_lower=float(lower.mean()),
        mean_upper=float(upper.mean()),
        mean_width=float(width.mean()),
    )


__all__ = [
    "ConditionalRegressionResult",
    "FrechetBoundsResult",
    "ImplicationResult",
    "conditional_regression",
    "frechet_bounds",
    "implication_test",
]
