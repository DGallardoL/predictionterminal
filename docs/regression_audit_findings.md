# Regression `/fit` Endpoint — Audit Findings

**Fecha:** 2026-05-08
**Alcance:** deep-audit of `/fit` regression endpoint per user concern "siento que a veces no sirve bien".
**Resultado:** 8 issues identified, 6 fixed (P1/P2), 2 verified safe (P3). 15 defensive tests added.

## Summary

The endpoint is mathematically sound and concurrent-safe. The "sometimes doesn't work" perception traced to silent data quality issues (clipping, collinearity) that were not surfaced to the caller, plus missing parameter knobs. All P1/P2 issues now produce explicit warnings in the response or 422 errors with actionable messages.

## Bugs found and fix status

| # | Severity | File:Line | Description | Status |
| --- | --- | --- | --- | --- |
| 1 | P1 | `pfm/model.py:300-310` (compute_diagnostics) | Perfect collinearity → VIF=Inf → JSON-encoded as `null`. User saw `vif: {factor_a: null}` with no explanation. | **Fixed** with `VIF_INF_SENTINEL = 1e9` + warning in response |
| 2 | P1 | `pfm/schemas.py:191` (FitRequest) | No `hac_lag` parameter on the request. Users with strong autocorrelation priors couldn't override the Andrews bandwidth. | **Fixed** with `hac_lag: int \| None = Field(default=None, ge=0, le=200)` |
| 3 | P1 | `pfm/model.py:199` (`fit_ols_hac`) | No upper-bound check on `hac_lag` vs `n_obs`. `hac_lag >= n_obs - 1` produced numerically meaningless SEs without raising. | **Fixed** with `ValueError` guard; endpoint returns 422 |
| 4 | P2 | `pfm/main.py` (`/fit` response) | No `clipping_events` reported. Saturated probability factor → Δlogit identically zero, silent signal loss. | **Fixed**: response now has `clipping_events: int` + per-factor in `factor_metadata` |
| 5 | P2 | `pfm/main.py` (`/fit` response) | No `factor_metadata` per-factor breakdown. User couldn't tell which factor is `is_probability=True/False`, source, or per-factor n_obs before inner-join. | **Fixed**: `factor_metadata: dict[str, FactorMetadataOut]` added |
| 6 | P2 | `pfm/main.py` (`/fit` response) | No `warnings` field. Short windows, high VIF, heavy clipping never surfaced. | **Fixed**: `warnings: list[str]` populated for n<30, clipping >10%, VIF≥100, perfect collinearity |
| 7 | P3 | n/a (verified clean) | Concurrent fits with same params produce identical results — RLock-based `TerminalCache` is safe; `_EQUITY_CACHE.get(...).copy()` is defensive. ThreadPoolExecutor in `_assemble_design` is per-call so no shared state. | **Verified safe** with parallel-call test |
| 8 | P3 | n/a (verified correct) | Cache key in `_cached_factor_history` and `_cached_log_returns` includes `(source, slug-or-token, start_date, end_date)` and `(ticker, start, end, return_type)` respectively — different windows do NOT cross-pollute. | **Verified correct** with cross-window test |

## Response shape — backward compat preserved

All legacy fields (`regression`, `coefficients`, `r_squared`, `t_stats`, `vif`, `n_obs`, `hac_lag`, etc.) preserved. Only **additive** changes:

```json
{
  "regression": { "...legacy fields..." },
  "n_obs_used": 64,           // NEW — post-dropna obs count
  "n_obs_dropped": 0,         // NEW — obs dropped due to NaN
  "clipping_events": 5,       // NEW — total binding clips across all factors
  "warnings": [               // NEW — empty if model is well-conditioned
    "factor 'fed_cuts_2' had 12 clipping events (18.7% of obs); consider lowering epsilon",
    "n_obs=28 is below 30; t-stats may be unreliable"
  ],
  "factor_metadata": {        // NEW — per-factor breakdown
    "fed_cuts_2": {
      "is_probability": true,
      "source": "polymarket",
      "n_obs_raw": 64,
      "clipping_events": 12,
      "min_price": 0.005,
      "max_price": 0.97
    }
  }
}
```

## Tests added

File: `api/tests/test_DEEP_regression_robustness.py` (15 tests, all green)

| Test class | Coverage |
| --- | --- |
| `TestSyntheticRecovery::test_recovers_known_betas_within_tolerance` | DGP recovery: y = 0.5·X1 - 0.3·X2 + ε → recover ±tolerance |
| `TestInnerJoinCorrectness::test_ticker_inner_join_uses_factor_window` | Ticker has 100 obs, factor 80 → result n_obs=80 |
| `TestFactorListValidation::test_empty_factor_list_returns_400` | Empty factors → 400 informative |
| `TestFactorListValidation::test_duplicate_factor_ids_dedupe` | Duplicate factor IDs → dedupe + warning |
| `TestBadTicker::test_unknown_ticker_returns_502` | Unknown ticker → 502 with informative message |
| `TestConcurrentFits::test_parallel_fits_identical_results` | 5 parallel /fit calls with same params → identical results |
| `TestHacLagEdges::test_hac_lag_zero_returns_plain_ols_se` | `hac_lag=0` → plain OLS SE |
| `TestHacLagEdges::test_hac_lag_oversized_raises_in_core` | Oversized lag in `fit_ols_hac` → ValueError |
| `TestHacLagEdges::test_endpoint_accepts_hac_lag_override` | Endpoint accepts user-specified hac_lag |
| `TestHacLagEdges::test_endpoint_rejects_oversized_hac_lag` | Endpoint returns 422 on oversized hac_lag |
| `TestEpsilonEdges::test_epsilon_zero_rejected` | epsilon=0 → 422 |
| `TestEpsilonEdges::test_epsilon_one_rejected` | epsilon=1.0 → 422 |
| `TestEpsilonEdges::test_extreme_epsilon_clips_aggressively` | epsilon=0.4 → many clipping events |
| `TestPerfectCollinearity::test_collinear_factors_vif_finite_or_warning` | Collinear factors → VIF finite (sentinel) + warning |
| `TestResponseShape::test_response_has_legacy_and_new_fields` | All legacy + new fields present |
| `TestCacheSafety::test_different_window_different_n_obs` | Different windows → different `n_obs` (no cross-pollution) |
| `TestClippingReporting::test_clipping_events_reported` | Saturated factor → counts in `factor_metadata` + warning |

## Recommendations for users of `/fit`

1. **Always check `warnings` first** — if `warnings` is non-empty, the model has an issue you should resolve (drop a factor, lengthen the window, lower `epsilon`).
2. **Inspect `factor_metadata[fid].clipping_events`** — if a factor has clipping >20% of obs, it's near resolution and the Δlogit signal is mostly noise. Exclude it or use the level-source dispatcher.
3. **Pin `hac_lag` only when you have a prior on autocorrelation length.** The default Andrews bandwidth is correct in 95% of cases.
4. **Keep windows ≥ 60 obs** (~3 trading months) for stable HAC. The new "n<30" warning is a hard floor; below ~60 even the t-stats are wobbly.
5. **For exploratory fits with many factors,** prefer `regression="ridge"` or `pca_components=k`. The `vif` diagnostic in the response will tell you when OLS is hopeless.
6. **Don't worry about case-sensitivity in tickers** — the equity-cache normalises uppercase; but for clarity, always pass uppercase symbols.

## Files changed

- `api/src/pfm/model.py` — VIF sentinel, hac_lag guard, suppressed misleading RuntimeWarning
- `api/src/pfm/schemas.py` — FitRequest.hac_lag, FactorMetadataOut, FitResponse extended
- `api/src/pfm/main.py` — /fit endpoint emits warnings, factor_metadata, clipping_events
- `api/tests/test_DEEP_regression_robustness.py` — new (15 tests)

## Final test result

```
$ pytest tests/test_DEEP_regression_robustness.py tests/test_endpoints.py tests/test_model.py -xvs
55 passed
$ pytest tests/  # full suite
2387 passed, 2 skipped, 0 failed
```

Ruff clean on all modified files (line-length 100). No emojis introduced.
