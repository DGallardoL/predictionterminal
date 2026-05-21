# Multi-source factors (Manifold / PredictIt / BLS / FRED)

The factor model originally accepted only Polymarket and Kalshi factors, plus
the chained-monthly composite. As of wave-23 the catalog supports four
additional sources: Manifold and PredictIt for prediction-market signals, and
BLS / FRED for macro level series. The dispatch lives in
`pfm.factors.fetch_factor_history_dispatch`; every fetcher returns a
`DataFrame` indexed by UTC date with a single `price` column so the
downstream regression pipeline does not have to branch on source.

## `is_probability` and the factor transform

Every `FactorConfig` carries an `is_probability` flag. The model reads it
when building the design matrix:

- `is_probability=True` (default for prediction-market sources): apply
  `Δlogit` — clip to `[ε, 1-ε]`, take `log(p / (1 - p))`, then first
  difference. Equation `r_t = α + Σ β_i · Δlogit(p_{i,t}) + ε_t`.
- `is_probability=False` (default for BLS / FRED): apply plain first
  differences via `pfm.model.delta_level`. Useful for yield spreads,
  jobless-claim counts, indices — anything that is not bounded to
  `[0, 1]`.

`pfm.model.delta_logit` carries a guardrail: if it sees a series outside
`[0, 1]` it emits a `UserWarning` and falls back to plain `diff()` rather
than silently returning all zeros after the clip.

## Mixed factor models

When a regression stacks both probability factors and level factors, the
columns of `X` end up on different scales (Δlogit values are typically
`O(0.1)`, while a Δyield-spread is `O(0.01)` and a Δclaim-count can be
`O(10⁴)`). For most fits this is fine because OLS is scale-invariant in
the betas, but condition numbers explode and the VIF report becomes
hard to read. We recommend standardising columns (`StandardScaler`-style
z-scoring) before fitting whenever you mix sources, or alternatively
running the level series through a percent-change transform first.
