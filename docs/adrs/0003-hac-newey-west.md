# ADR-0003: Inference via HAC standard errors with automatic bandwidth selection

- **Status:** Accepted
- **Date:** 2026-04-30
- **Authors:** Damian Gallardo

## Context

Daily financial returns exhibit two well-documented properties that would
break the Gauss-Markov assumptions of OLS standard errors:

- **Autocorrelation.** Momentum on short horizons, mean-reversion on
  longer ones. Residuals from a contemporaneous factor regression on
  returns will not be i.i.d.
- **Conditional heteroskedasticity.** Volatility clusters. Even when the
  conditional mean is correctly specified, $\text{Var}(\varepsilon_t \mid
  \mathcal{F}_{t-1})$ is far from constant.

Plain OLS point estimates remain unbiased, but **OLS standard errors are
not consistent** under these conditions. Reporting them would be a
correctness bug, not a stylistic preference. We need an estimator of
$\text{Var}(\hat\beta)$ that is robust to both.

## Considered alternatives

- **Heteroskedasticity-consistent (HC) robust SEs.** Fixes heteroskedasticity but ignores
  autocorrelation. Insufficient.
- **Bootstrap (block / circular-block).** Defensible and assumption-light
  but adds a tunable block-length parameter and runtime cost. Overkill for
  the POC and harder to defend in 15 minutes.
- **GLS with a parametric residual model.** Strong assumption on the
  residual process; brittle if mis-specified.

## Decision

Use **HAC** (heteroskedasticity- and autocorrelation-consistent)
covariance, via `statsmodels.OLS.fit(cov_type='HAC', cov_kwds={'maxlags': L})`.

Bandwidth $L$ is set by the **automatic bandwidth selection plug-in rule**:

$$
L = \left\lfloor 4 \cdot (T/100)^{2/9} \right\rfloor
$$

floored at 1. The chosen lag is returned in the response under
`diagnostics.hac_lag` so users can sanity-check it against the residual
ACF if they want.

We do **not** roll our own implementation: the statsmodels HAC code is
peer-reviewed and battle-tested. Reproducing it would be a footgun.

## Consequences

- Standard errors are larger than naive OLS would report — that's the
  whole point. Some borderline-significant factors will lose significance
  after HAC; that's correct.
- The bandwidth is data-dependent, so two fits over slightly different
  windows can use slightly different lags. We report the lag explicitly
  so this is transparent.
- We allow the caller to override the lag via `fit_ols_hac(..., hac_lag=L)`
  in unit tests, but the API does not expose that knob — users get the
  automatic-bandwidth default. Override is reserved for the future, behind an explicit
  request.
