## Why

Our 210-experiment audit established that R² OOS alone can't distinguish real
signal from random noise at our sample size — a model can have train R² 5%
and test R² 5% and still be indistinguishable from regressing on shuffled
factor data. The **permutation test** is the rigorous answer: shuffle factor
values, refit, repeat 50–100×, and compute `p = P(null R² ≥ real R²)`.

We've used this diagnostic in `/tmp` scripts to validate the cherry-picked
NVDA basket (p=0.013, REAL signal). It's the most honest test we have, and
it's not exposed in the API or UI. Users see test R² values without a way
to know whether they're signal or noise.

## What Changes

- Add **`POST /factors/permutation`** endpoint that takes (ticker, factors,
  start, end) and returns the real R² plus a null distribution from N
  permutations and a p-value.
- Add `permutation_iters` toggle to `FitRequest` and `BestModelRequest` —
  when set, runs the permutation test alongside the regression and returns
  `permutation` block in the response.
- Frontend: when permutation is requested, show a pill in the stats row:
  - `p_perm < .05` → green "real signal beats random"
  - `p_perm < .10` → orange "marginal"
  - `p_perm ≥ .20` → red "indistinguishable from noise"
- Display the null R² distribution as a small histogram with the real R²
  marker line, in the Diagnostics tab.

## Capabilities

- **Modified Capabilities**:
  - `factors-catalog` — adds a new endpoint `/factors/permutation` and a
    new optional field on `/fit` and `/factors/best` responses.

## Impact

- **Code**: `api/src/pfm/main.py` (new endpoint + body fields), `analyses.py`
  (extract permutation runner from `/tmp/perm_all.py`), `schemas.py`
  (new request/response types), frontend `index.html` (pill + histogram).
- **API**: additive; existing callers unaffected. New endpoint behind
  `permutation_iters > 0`.
- **Performance**: 50 permutations × full fit ≈ 5–15 s. Default off; user
  opt-in.
- **Tests**: add test for the permutation runner with a synthetic case
  (real signal in the data should give p<0.05 reliably across seeds).
