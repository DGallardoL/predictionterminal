## ADDED Requirements

### Requirement: Chained-factor source type
The factor catalog SHALL accept a new factor `source` value `"chain"`
that composes multiple underlying single-source markets into one
continuous daily probability series.

#### Scenario: Chain factor parses from YAML
- **WHEN** `factors.yml` contains a factor with `source: chain` and a
  non-empty `segments:` list, each segment having `source`, `slug`, and
  `end`
- **THEN** `load_factors` SHALL return a `FactorConfig` with `is_chained
  == True` and a `segments` tuple of `ChainSegment` instances in the
  same order as the YAML

#### Scenario: Chain segments must be ordered
- **WHEN** the YAML specifies chain segments whose `end` dates are not
  in strictly ascending order, or contain duplicates
- **THEN** `load_factors` SHALL raise `ValueError` mentioning the
  "ascending end-date order" or "unique end dates" rule

#### Scenario: Chain factor must have segments
- **WHEN** the YAML specifies `source: chain` without any `segments:`
- **THEN** `load_factors` SHALL raise `ValueError` mentioning
  "non-empty segments"

#### Scenario: Single-source factor cannot carry segments
- **WHEN** a YAML factor has `source: polymarket` (or `kalshi`) AND a
  non-empty `segments:` list
- **THEN** `FactorConfig` validation SHALL raise `ValueError` mentioning
  "only source=chain may carry segments"

### Requirement: Chain fetcher composes per-segment histories
The fetcher SHALL produce a single daily price series by stitching each
segment's bars on its active window: lower bound exclusive (the previous
segment's end), upper bound inclusive (its own end). The first segment's
lower bound is unbounded.

#### Scenario: Concatenation across two segments
- **WHEN** `fetch_chained_history` is called with two segments whose
  underlying fetchers return non-empty bars
- **THEN** the output SHALL contain bars from segment 1 with dates ≤
  `e_1`, immediately followed by bars from segment 2 with dates in
  `(e_1, e_2]`, sorted ascending, with no duplicate dates

#### Scenario: Mixed sources across segments
- **WHEN** one segment's `source` is `polymarket` and another's is
  `kalshi`
- **THEN** `fetch_chained_history` SHALL invoke both clients via their
  respective fetchers and concatenate the results in order

#### Scenario: Empty segment leaves a gap, chain still succeeds
- **WHEN** one segment in the list returns no bars
- **THEN** `fetch_chained_history` SHALL still return the bars from the
  other segments and SHALL NOT raise

#### Scenario: Missing client for a segment raises
- **WHEN** the segments list includes a Kalshi segment but `kalshi=None`
  is passed to `fetch_chained_history`
- **THEN** the function SHALL raise `RuntimeError` mentioning the
  segment index and the slug

### Requirement: Dispatcher routes chain factors and avoids cache collisions
`_cached_factor_history` SHALL dispatch `source=chain` factors through
`fetch_chained_history`, and SHALL include a stable signature of the
segment list in the cache key.

#### Scenario: Chain factor dispatches to chain fetcher
- **WHEN** `_cached_factor_history` is called with a `FactorConfig`
  whose `source == "chain"`
- **THEN** the dispatcher SHALL call `fetch_chained_history` (NOT the
  single-source Polymarket or Kalshi fetcher) with the segments list
  and BOTH the Polymarket and Kalshi clients

#### Scenario: Distinct compositions don't collide in cache
- **WHEN** two `FactorConfig` instances share the same `id` and `slug`
  but have different `segments`
- **THEN** their cache keys SHALL differ, and a fetch for one SHALL NOT
  return bars cached from the other
