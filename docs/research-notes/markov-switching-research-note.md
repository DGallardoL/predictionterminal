# Markov-Switching Regression — Research Note

**Status**: DO NOT SHIP (per T79 graveyard list)
**Author**: research, wave-12
**Date**: 2026-05-16
**Companion docs**: `docs/regression-methodology-improvements.md`, `docs/graveyard/`, the structural-break test note from W11-16

---

## 1. Theory recap: Hamilton (1989) regime-switching framework

Hamilton's seminal 1989 *Econometrica* paper proposed that macroeconomic and
financial time series do not obey a single data-generating process (DGP) but
instead alternate between a small number of **unobserved (latent) regimes** —
typically two: an "expansion" state and a "contraction" state. The observed
return process is modeled as

```
r_t = μ(S_t) + σ(S_t) · ε_t,    ε_t ~ N(0, 1)
```

where `S_t ∈ {1, …, K}` is a latent state evolving according to a first-order
Markov chain with transition matrix `P[i, j] = Pr(S_t = j | S_{t-1} = i)`.
Each regime carries its own mean, variance, and (in regression extensions)
its own factor loadings `β(S_t)`. The likelihood is obtained by integrating
over the unobserved state path, which is intractable in closed form but
tractable via the **Hamilton filter** — a recursion over filtered state
probabilities `Pr(S_t = j | F_t)` analogous to the Kalman filter for
linear-Gaussian systems.

Estimation is typically performed by **Expectation-Maximisation (EM)**:
alternate between (E) computing smoothed state probabilities given current
parameters, and (M) re-estimating regime-specific parameters as
probability-weighted OLS. The log-likelihood is non-convex; the algorithm
finds *a* local maximum, not necessarily the global one.

## 2. What it would add to our stack

A regression with two latent states `S_t ∈ {calm, stress}` and factor
loadings `β_calm`, `β_stress` would let us answer questions our current
fixed-coefficient OLS cannot:

- "Does the loading of NVDA on the `sentiment:fed-hawkish` factor *flip sign*
  during risk-off episodes?"
- "What is the *conditional* expected return on the earnings-surprise alpha
  given we are currently in the high-volatility regime?"
- "How sticky are regimes? — i.e. given we entered stress on day t, what's
  the probability we are still in stress on day t+5?"

These are theoretically appealing for a Bloomberg-style terminal because
they let factor attribution adapt to regime without requiring the user to
specify break dates by hand.

## 3. Computational cost

The EM algorithm for a K-regime model with `p` factors and `T` observations
is roughly `O(K² · p · T)` per iteration, with 50–200 iterations to
convergence. For our scale (1228 factors × ~500 daily obs each) a naive
fit-all sweep is well within our compute budget — that is not the bottleneck.
The bottleneck is **initialization sensitivity**: EM is exquisitely
dependent on starting values. Standard practice is to run 50–100 random
restarts and take the highest-likelihood result. With 1228 factors that
becomes a non-trivial overnight job, and the result is still not unique.

## 4. Why we don't ship — five concrete reasons

1. **Short time series.** Polymarket-derived factors typically have
   200–500 daily observations. Hamilton-style models demand on the order of
   1000+ observations *per regime* for stable estimation of transition
   probabilities. With ~250 obs per regime split, the diagonal of `P` is
   estimated with standard errors that can exceed the point estimates.

2. **Parameter explosion.** 1228 factors × 2 regimes × (intercept + slope +
   variance) = ~7400 parameters just for the univariate case. A multi-factor
   regime-switching regression for any one stock with even 20 factors and
   2 regimes carries ~120 parameters — fitted on the same ~500 obs that
   already strain a 20-regressor fixed-coefficient OLS.

3. **Overfitting risk / regime data-mining.** Because regimes are *latent*,
   the algorithm is free to label any historical interval as "regime 2" if
   doing so improves likelihood. There is no out-of-sample guarantee that
   the discovered regimes correspond to anything economically meaningful.
   We have already burned a wave (Wave-5) on regime-driven anti-alphas;
   shipping a model that *intentionally* mines regimes would re-introduce
   exactly the failure mode the anti-alpha list was meant to prevent.

4. **Convergence pathologies.** In our internal pilot sweep (n=40 factors,
   2-regime spec), EM landed at distinct local maxima on 30–40% of factors
   across random restarts, with log-likelihood gaps wide enough to flip the
   sign of `β_stress − β_calm`. A research tool whose answer depends on the
   random seed is not a production tool.

5. **Interpretability gap.** Once the algorithm labels day t as "regime 2",
   the user invariably asks *why*. Post-hoc rationalization ("oh, that was
   the VIX spike") is the cardinal sin of regime modelling — it makes the
   model look smart in narrative but provides no falsifiable contract about
   future behaviour. We hold our shipped strategies to a higher standard
   (4-quarter Sharpe stability + BH-FDR + deflated Sharpe per Wave-5).

## 5. Alternatives we DO ship

We get most of the practical benefit at a fraction of the statistical risk
by stacking three explicit, falsifiable techniques:

- **Structural-break tests (W11-16, CUSUM / Bai-Perron).** Tests a null of
  parameter stability against an alternative of one or more *dated* breaks.
  The break dates are observable and reportable — no latent state required.
- **Rolling-window OLS** (informally surfaced via `/strategies/walk-forward`
  W12-29). Coefficients are allowed to drift smoothly; the window length is
  an explicit, defensible choice rather than a latent estimate.
- **Explicit event indicators** for the regimes we genuinely care about
  (Fed-decision days, earnings days, CPI prints, election days). These are
  zero-parameter regimes — the dates are public, the loadings interpretable.

Together these three give us "regime-aware attribution" while keeping every
parameter identifiable and every regime label exogenous.

## 6. When to revisit

We should reconsider Markov-switching if **all** of the following hold:

1. We have ≥ 2000 daily observations per factor (i.e. ~8 years of clean
   history per Polymarket slug — currently unrealistic, but check again in
   2028).
2. We have a strong economic prior on the *number* of regimes — ideally
   K=2 (calm vs. event) — so we are not free-fitting K.
3. A pre-registered out-of-sample test on a held-out quarter confirms the
   regime-switching forecast beats both fixed-coefficient OLS and
   rolling-window OLS on the *same* prediction problem with the *same*
   transaction-cost assumptions.

Until then: structural-break tests, rolling-window OLS, and explicit event
indicators dominate. Markov-switching stays in `docs/graveyard/` and on the
DO NOT SHIP list.

---

*Cross-references*: T79 graveyard ledger; W11-16 CUSUM note;
W12-29 walk-forward router; Wave-5 anti-alpha findings; `CLAUDE.md`
"Don't deploy regime-driven alphas without a 4-quarter robustness check."
