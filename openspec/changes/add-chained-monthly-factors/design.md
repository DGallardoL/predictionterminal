## Decision: chain semantics = "earliest non-expired segment is active"

For an ordered list of segments `s_1 < s_2 < ... < s_n` with strictly
ascending end dates `e_1 < e_2 < ... < e_n`, the chained probability on
date `t` is

    chain(t) = p_i(t)   where i = min{j : e_j ≥ t}

i.e., the segment whose end has not yet passed and is closest to `t`.

This matches the "next-print probability" reading: on any date `t`, the
active market is the *earliest* one that still resolves to a future event.
When that market expires (`t > e_i`) the chain switches to `s_{i+1}`.

Edge of the segment is *inclusive on the right*: at exactly `t = e_i`,
the active segment is still `s_i` (the resolution day belongs to the
expiring market). On `t = e_i + 1` the next segment takes over.

## Decision: per-segment fetch covers the full window then we filter

The simpler alternative is "ask each segment for only its active window".
We don't do that, because:

1.  Polymarket's price endpoint takes (`startTs`, `endTs`) but the
    market may have a much shorter window than what we ask for; we'd be
    re-implementing the per-market start logic.
2.  Caching is more effective when the per-segment fetch parameters don't
    depend on chain composition (otherwise small composition tweaks
    invalidate every segment's cache).

Trade-off: extra bytes pulled. Acceptable given the small per-market
histories.

## Decision: cache key includes a stable segments signature

Two chained factors might share a factor `id` but differ in composition
(e.g. an analyst correcting one slug). To avoid stale cache hits, the
dispatcher's cache key uses

    "chain::{id}::{source|slug|end; ...}"

and `segments_signature` builds that string from the segment list. This
makes the cache safe across edits to `factors.yml`.

## Decision: chain is a YAML-level shape, not a runtime composition

We considered exposing chains as a runtime construct via a new endpoint
`POST /factors/chain` taking a list of slugs. Rejected because:

1.  Chained factors are *editorially curated* — getting the segment ends
    right requires a human looking at each underlying market.
2.  YAML keeps them auditable in version control alongside the rest of
    the catalog and makes them visible in `GET /factors`.

Custom (user-supplied) factors via `/fit?custom_factors=...` remain
single-source `polymarket` only. No chained custom factors.

## Validation rules

- ChainSegment: `source ∈ {polymarket, kalshi}`, non-empty `slug`,
  `end: datetime.date`.
- FactorConfig with `source=chain`: ≥1 segment, strictly ascending
  unique `end` dates. Single-source factors MUST NOT carry segments.
- Dispatcher raises `RuntimeError` if a segment requires a client that
  wasn't injected (e.g. a chain with a Kalshi segment but no Kalshi
  client in `app.state`).

## Out of scope

- Cross-source resolution-data normalisation (Kalshi resolves to Kalshi
  oracle, Polymarket resolves via UMA; we treat both as binary
  probabilities at the daily granularity and don't attempt to merge
  resolution semantics).
- Per-segment weighting (e.g. give 80% weight to the closest market and
  20% to the next). The current rule is hard switch at the boundary.
