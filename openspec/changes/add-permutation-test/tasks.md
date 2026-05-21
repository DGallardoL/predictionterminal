## Tasks

### 1. Backend: permutation runner

- [x] **Add `permutation_test` helper to `analyses.py`**
      Lift the implementation from `/tmp/perm_all.py`. Signature:
      ```python
      def permutation_test(
          y: pd.Series, X: pd.DataFrame,
          n_iters: int = 50, seed: int = 42,
          test_fraction: float = 0.20,
      ) -> dict[str, float | list[float]]:
          """Returns {real_test_r2, null_test_r2s, null_median, null_pct95,
                       null_max, p_value, n_iters_completed}."""
      ```
      Determinism: use `np.random.default_rng(seed)`. Permutations independent
      per factor.

- [x] **Add unit test** in `tests/test_permutation.py`:
      - Synthetic data with known β=0.5 signal → expect p<0.05 in ≥9/10 seeds.
      - Pure-noise data → expect uniform p across [0,1] (median p ≈ 0.5).

### 2. Schemas

- [x] **Add `PermutationRequest` and `PermutationResult`** in `schemas.py`.
      Constraints: `n_iters: int = Field(default=50, ge=10, le=500)`.

- [x] **Extend `FitRequest`** with `permutation_iters: int = Field(default=0, ge=0, le=500)`.

- [x] **Extend `FitResponse`** with `permutation: PermutationResult | None = None`.

### 3. API

- [x] **New endpoint** `POST /factors/permutation` in `main.py`. Reuses
      `_assemble_design` to build (y, X) then delegates to `permutation_test`.

- [x] **Update `/fit`** to optionally call `permutation_test` after the main
      fit when `permutation_iters > 0`. Reuse the same y, X.

### 4. Frontend

- [x] **Add `permutation_iters` slider** to the Validation accordion in
      sidebar (off / 50 / 100 / 200 / 500).

- [x] **Pill in stats row** when `permutation` block present:
      - p<.05 green "p_perm 0.013 — real"
      - p<.10 orange "p_perm 0.082 — marginal"
      - p≥.20 red "p_perm 0.34 — noise"
      - p in [.10, .20) gray "p_perm 0.15 — borderline"

- [x] **Histogram in Diagnostics tab**: `null_test_r2s` as bar chart with
      a vertical line at `real_test_r2`. Plotly histogram is one call.

### 5. Verification

- [x] **Unit tests pass**: `pytest tests/ -q` → 61/61 (51 prior + 10 new in `test_permutation.py`).

- [x] **Reproducibility check**: same seed across two calls returns
      identical `null_test_r2s`.

- [x] **Integration smoke**: `/fit` with `permutation_iters=50` for NVDA +
      cherry-pick basket — expect p<0.05 (matching the audit finding).

- [ ] **UI smoke**: select 3 factors, enable permutation, run fit, see pill
      and histogram render. _(awaiting manual verification in the browser — code is wired but I can't drive a browser from here)_

### 6. Cleanup

- [ ] **Archive** with `openspec archive add-permutation-test` (from project
      root) — applies the spec delta.
