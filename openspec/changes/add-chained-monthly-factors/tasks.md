## Tasks

### 1. Loader & dataclasses

- [x] Extend `pfm.factors`:
      - new `CHAIN_SOURCE = "chain"` constant.
      - new `ChainSegment` frozen dataclass (source, slug, end, name).
      - extend `FactorConfig` with `segments: tuple[ChainSegment, ...]`.
      - validate ascending unique end dates; reject single-source carrying
        segments.
- [x] `load_factors` parses `segments:` list under chain factors.

### 2. Chain fetcher

- [x] `pfm.sources.chain.fetch_chained_history(segments, *, poly, kalshi,
      start, end, polymarket_fetch=…, kalshi_fetch=…)`:
      - per-segment fetch via the corresponding source helper
      - filter each to its active window: `(prev.end, end]` (exclusive of
        the previous segment's end, inclusive of its own)
      - concat, drop duplicate dates (keep earliest segment), sort by date
      - clamp the user's `[start, end]` window
- [x] `segments_signature(segments)` — stable cache-key fragment.

### 3. Dispatcher integration

- [x] Refactor `_cached_factor_history` to accept a `FactorConfig`
      directly instead of `(slug, source)` so chain dispatch is natural.
- [x] Branch on `fc.source == CHAIN_SOURCE` → `fetch_chained_history`,
      passing the Kalshi client from `app.state.kalshi`.
- [x] Cache key uses `segments_signature(fc.segments)` for chain entries
      so two chains with identical id but different composition don't
      collide.
- [x] Update `_resolve_factor_specs` and `_assemble_design` to work on
      `FactorConfig` values directly (fewer string-tuples in flight).

### 4. Tests

- [x] `tests/test_chain.py` (≥30 tests):
      - `ChainSegment` validation (bad source, empty slug, bad end type)
      - `FactorConfig` chain validation (no segments, unsorted, dup ends,
        single-source carrying segments, unknown source)
      - `_segment_window` lower-bound rules
      - `_filter_segment_bars` inclusive-upper / exclusive-lower
      - `fetch_chained_history` two-segment concatenation, mixed sources,
        empty-segment gap, all-empty returns empty, user-window clamp,
        missing client raises, empty/unsorted segments raise
      - `segments_signature` stable + distinguishes
      - Loader: parses chain from YAML, rejects missing fields
      - Dispatcher integration: chain dispatch concatenates segments,
        cache key distinguishes compositions
- [x] All existing tests still pass (no regressions).

### 5. Verification

- [x] `pytest tests/ -q` → 109+ passed (was 107).
- [x] `GET /factors` continues to list all factors including chains
      (verified via existing `test_factors_lists_loaded_factors`).

### 6. Cleanup

- [ ] Archive with `openspec archive add-chained-monthly-factors --yes`
      from project root once Damian confirms a chain factor renders in
      the UI and `/fit` accepts a chain factor in a real run.
