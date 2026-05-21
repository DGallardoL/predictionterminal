## ADDED Requirements

### Requirement: Permutation test endpoint
The API SHALL expose `POST /factors/permutation` that takes a regression
specification (ticker, factor ids, custom factors, window, return_type,
regression, alignment) plus `n_iters` and returns a real R² alongside the
null distribution from shuffling factor values within each series.

#### Scenario: Endpoint returns real and null distribution
- **WHEN** a client posts to `/factors/permutation` with a valid body and
  `n_iters=50`
- **THEN** the response SHALL include `real_test_r2` (float),
  `null_test_r2s` (list of `n_iters` floats), `null_median`,
  `null_pct95`, `null_max`, and `p_value` (fraction of null ≥ real).

#### Scenario: n_iters is bounded
- **WHEN** a client requests `n_iters` greater than 500 or less than 10
- **THEN** the request SHALL fail with HTTP 422 validation error.

#### Scenario: Honest p-value labelling
- **WHEN** the p-value from the permutation test is computed
- **THEN** clients SHALL be able to derive a verdict label using these
  thresholds: `< 0.05` → "real", `< 0.10` → "marginal", `≥ 0.20` → "noise".

### Requirement: Permutation alongside /fit
The `/fit` endpoint SHALL accept an optional `permutation_iters` field on
the request body (default 0). When greater than zero, the response SHALL
include a `permutation` block matching the structure of the dedicated
endpoint.

#### Scenario: Default behaviour unchanged
- **WHEN** a `/fit` request omits `permutation_iters` or sets it to 0
- **THEN** the response SHALL NOT include the `permutation` block and the
  fit SHALL be identical to current production behaviour.

#### Scenario: Permutation block populates when requested
- **WHEN** `/fit` is called with `permutation_iters=50` and 4 factors
- **THEN** the response SHALL contain a `permutation` object with
  `real_test_r2`, `p_value`, and the null summary.

### Requirement: Permutation runner is deterministic given seed
The internal `permutation_test` helper SHALL accept a `seed` argument and
produce the same null distribution for identical inputs.

#### Scenario: Same seed yields same nulls
- **WHEN** the runner is invoked twice with identical (y, X, n_iters,
  seed)
- **THEN** the returned `null_test_r2s` arrays SHALL be element-wise
  identical.
