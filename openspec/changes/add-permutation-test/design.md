## Context

The audit (`/tmp/perm_all.py`, `/tmp/nvda_cherry.py`) established that:
- 0/14 tickers have real signal under fully-automated stepwise selection
  with our 31-factor pool (permutation p > 0.20 for all)
- Cherry-picked thesis-driven baskets break this: NVDA + risk-on basket has
  p=0.013

The permutation runner in those scripts is ~30 lines of pandas/statsmodels.
We need to lift it into the production analyses module and wire it through
the API.

## Goals / Non-Goals

**Goals**
- Single-call permutation test that any user can request via the UI
- Honest verdict pill (real / marginal / noise) directly in stats row
- Optional histogram of null distribution in Diagnostics tab

**Non-Goals**
- Multiple-test correction (Bonferroni / FDR across factors) — single-model
  p only
- Block-bootstrap permutation (preserves autocorrelation) — start with
  iid shuffle of factor values
- Caching the null distribution — recompute each call (cheap enough)

## Decisions

- **Shuffle values per factor independently.** This breaks alignment with
  returns while preserving each factor's marginal distribution. Standard
  in practice. *Alternative*: shuffle returns instead — equivalent under
  exchangeability assumption, but breaks the "factor identity" reading.

- **Default `n_iters = 50`** when permutation enabled. 50 gives p resolution
  of 0.02 (one in 50). 100 gives 0.01 but doubles runtime. Users can override
  up to 500.

- **Reuse the same selected factors across permutations.** The selection
  step happens once on the real data; permutations re-fit with that fixed
  set. This isolates "are the selected coefficients real?" from "could
  selection have been lucky?" — a different (harder) question.

- **Run synchronously inside the request.** 50 perms × ~50ms per OLS = ~3s,
  fits within FastAPI request budget. If we ever exceed 10s we'll move to
  background jobs.

## Risks / Trade-offs

- **Risk**: Users will request permutation on huge datasets and time out.
  → Mitigation: hard-cap `n_iters` at 500 in the schema validator.

- **Risk**: Permutation test gives false confidence — passing p<.05 doesn't
  mean the model is correctly specified, only that the relationship is
  unlikely under iid shuffling. → Mitigation: pill text says "beats random"
  not "predicts" — semantically narrower.

- **Risk**: Null distribution shape is informative but a single p-value
  hides it. → Mitigation: also return `null_median`, `null_pct95`, and the
  full sample of null R²s so the UI can plot a histogram.

## Migration Plan

1. Lift `permutation_test` function from `/tmp/perm_all.py` into
   `pfm/analyses.py`.
2. Add new schemas `PermutationRequest`, `PermutationResult`.
3. New endpoint `POST /factors/permutation`.
4. Extend `FitRequest`/`BestModelRequest` with `permutation_iters: int = 0`.
5. UI: pill + histogram.
6. Test harness: synthetic data with known signal — verify p<0.05 reliably.

## Open Questions

- Should we add a "block-bootstrap" mode that preserves autocorrelation?
  Defer until users complain that residuals are autocorrelated.
- Cache null distributions? Probably not — input space is too large.
