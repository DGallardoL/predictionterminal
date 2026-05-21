"""Single-day attribution: decompose an observed return into per-factor parts.

For a fitted model

    r̂_t = α + Σ_i β_i · Δlogit(p_{i,t})

this module computes, for a target date ``t*``:

    contribution_i = β_i · Δlogit(p_{i,t*})
    contribution_α = α
    predicted = α + Σ_i contribution_i
    residual  = observed - predicted

If the date is not in the training window, an exception is raised — the API
layer surfaces that as a 404.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pfm.model import FitResult


@dataclass(frozen=True)
class Contribution:
    factor_id: str
    delta_logit: float | None  # None for the intercept row
    beta: float | None  # None for the intercept row
    contribution: float


@dataclass(frozen=True)
class AttributionResult:
    date: pd.Timestamp
    observed_return: float
    predicted_return: float
    residual: float
    contributions: list[Contribution]


def attribute(
    fit: FitResult,
    y: pd.Series,
    X: pd.DataFrame,
    target_date: pd.Timestamp,
) -> AttributionResult:
    """Decompose the observed return on ``target_date`` by factor.

    Args:
        fit: Output of ``fit_ols_hac``.
        y: Aligned dependent variable used for the fit.
        X: Aligned factor matrix used for the fit (same one passed to fit_ols_hac).
        target_date: Calendar date to attribute. Must be in ``y.index``.

    Returns:
        ``AttributionResult`` with intercept + per-factor contributions.
    """
    if target_date not in y.index:
        raise KeyError(
            f"date {target_date.date()} not in fitted window "
            f"[{y.index.min().date()}, {y.index.max().date()}]"
        )

    observed = float(y.loc[target_date])
    row = X.loc[target_date]

    beta_by_factor = {est.factor_id: est.beta for est in fit.factors}

    contributions: list[Contribution] = [
        Contribution(
            factor_id="alpha",
            delta_logit=None,
            beta=None,
            contribution=fit.stats.alpha,
        )
    ]
    for col in X.columns:
        beta = beta_by_factor[col]
        delta = float(row[col])
        contributions.append(
            Contribution(
                factor_id=col,
                delta_logit=delta,
                beta=beta,
                contribution=beta * delta,
            )
        )

    predicted = sum(c.contribution for c in contributions)
    residual = observed - predicted

    return AttributionResult(
        date=target_date,
        observed_return=observed,
        predicted_return=predicted,
        residual=residual,
        contributions=contributions,
    )
