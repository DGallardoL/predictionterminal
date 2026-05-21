## Why

Many of the macro markets we want to use as factors (CPI, NFP, Fed funds
decisions, FOMC) settle every month. A *single* market only has a handful
of bars before resolution, so its time series is too short to be useful for
a 6-12 month equity-factor regression. The natural fix is to *chain*
consecutive monthly markets into one continuous probability series and
treat the chain as a single factor.

Concretely, each month a new "next-print CPI YoY > 3%" market opens, the
current one resolves, and the new one becomes the live source of the
forward-looking probability. By stitching these together — earliest
non-expired market is the active source on any date — we get a usable
multi-month history while preserving the "next-print" semantics on every
date.

This change adds the YAML shape and fetcher needed to express a chain.
**No quant logic changes**: chained factors flow through the same
Δlogit / HAC-OLS / VIF / permutation pipeline as single-source factors.

## What Changes

- **YAML loader (`pfm.factors`)** accepts a new `source: chain` shape with
  an ordered `segments:` list. Each segment carries its own
  (`source: polymarket | kalshi`, `slug`, `end`, optional `name`).
- **New module `pfm.sources.chain`** with `fetch_chained_history(...)`:
  per-segment fetch, slice each segment's bars to its active window
  (exclusive of the previous segment's end, inclusive of its own), then
  concatenate.
- **Dispatcher (`_cached_factor_history`)** dispatches `source=chain` →
  `fetch_chained_history`. Cache key includes a stable signature of the
  segment list so two chains with the same factor `id` but different
  composition don't collide in cache.
- **Validation**: segments must be non-empty and in strictly ascending
  unique `end` order. Single-source factors must NOT carry segments.

No schema changes, no public API changes: a chained factor surfaces in
`GET /factors` exactly like any other factor (with `source="chain"`), and
flows through `/fit`, `/attribution`, `/factors/best`, and the strategies
endpoints unchanged.

## Capabilities

- **Modified Capabilities**:
  - `factors-catalog` — adds the chained-factor source type and the
    requirement that the dispatcher handle it.

## Impact

- **Code**: `api/src/pfm/factors.py` (extended loader + `ChainSegment`
  dataclass), `api/src/pfm/sources/chain.py` (new file), `api/src/pfm/main.py`
  (dispatcher + import), `api/tests/test_chain.py` (new test file).
- **API**: additive — `source` discriminator can take the new value
  `"chain"`, but no existing factor changes shape.
- **Performance**: per-chain fetch is N segment-fetches; cache hit-rate
  should be excellent since most chain compositions are stable.
- **Tests**: 11 new tests covering segment validation, window slicing,
  chain concatenation across mixed sources, dispatcher integration,
  and cache-key uniqueness. All pass without external network IO.
