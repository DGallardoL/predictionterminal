# ADR-0002: Use Δlogit(p) as the factor signal, with explicit clipping

- **Status:** Accepted
- **Date:** 2026-04-30
- **Authors:** Damian Gallardo

## Context

Polymarket prices are probabilities in $[0, 1]$. Treating raw $\Delta p$ as
a regressor has two well-known problems:

1. **Boundary heteroskedasticity.** A move from 0.95 → 0.99 carries far
   more Bayesian information than 0.50 → 0.54, even though both have
   $\Delta p = 0.04$. A regression that uses $\Delta p$ directly will
   under-weight tail moves and over-weight middle-of-the-distribution
   noise.
2. **Non-linear scale.** Bayes' rule composes multiplicatively in odds
   space. Probabilities don't.

The logit transform $\text{logit}(p) = \log(p / (1-p))$ maps $(0,1) \to \mathbb{R}$
and makes $\Delta \text{logit}(p)$ scale-invariant in the information sense.
This is the standard transform in proper-scoring-rule literature and in
prediction-market analyses (Wolfers & Zitzewitz, etc.).

A second concern: $\text{logit}(0)$ and $\text{logit}(1)$ are undefined, and
markets do trade arbitrarily close to those boundaries during the run-up
to resolution. We need a clipping policy.

## Decision

1. Use $\Delta \text{logit}(p_t)$ as the regressor for every factor.
2. Clip $p$ to $[\varepsilon, 1-\varepsilon]$ before applying the transform
   with default $\varepsilon = 0.01$.
3. Expose $\varepsilon$ as a query parameter `epsilon` on `/fit` and
   `/attribution` so users can probe sensitivity.
4. Document explicitly that clipping is **lossy**: if a market trades at
   $0.005$ then $0.002$, both clip to $0.01$ and the transform reports
   $\Delta \text{logit} = 0$ — the regression "doesn't see" the move. This
   is acceptable for the POC, but is a known modelling limitation.

## Consequences

- The user can dial $\varepsilon$ down for markets that spend time deep in
  the tails, at the cost of amplifying microstructure noise as $p \to 0$
  or $p \to 1$.
- The clipping bound shows up in the response body (`epsilon` field) so
  fits are reproducible.
- Future work could replace the hard clip with a soft transform (e.g.
  `arcsin(2p - 1)` or a Beta-prior smoothing) — left out of the POC.
