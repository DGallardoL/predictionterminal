# ADR-0004: Redis cache with 1-hour TTL and graceful degradation

- **Status:** Accepted
- **Date:** 2026-04-30
- **Authors:** Damian Gallardo

## Context

Every `/fit` call would otherwise fan out into:

- 1 `GET /markets?slug=…` per factor (Polymarket Gamma)
- 1 `GET /prices-history?market=…` per factor (Polymarket CLOB)
- 1 `yfinance.download(ticker, start, end)` (Yahoo Finance via yfinance)

A user iterating on a regression — tweaking date range, swapping a factor —
will re-pull the same Polymarket histories repeatedly. We don't want each
iteration to hit upstream APIs from scratch:

- yfinance is **rate-limited unpredictably** by upstream Yahoo. Repeated
  pulls during a demo can produce empty DataFrames mid-presentation.
- Polymarket has a generous 1000 req / 10 s budget on `/prices-history`,
  but cache friendliness is also good citizenship.
- All three calls add latency to a demo.

## Considered alternatives

- **No cache.** Simplest; demos badly because the second `/fit` is as slow
  as the first.
- **In-process LRU.** Disappears across container restarts. With multiple
  uvicorn workers, would be per-worker — inconsistent.
- **SQLite on a volume.** Persistent but adds disk schema and migration
  surface for what is fundamentally a TTL'd KV store.
- **Postgres.** Vastly heavier than the problem. (See ADR-0005.)

## Decision

Use **Redis 7** as a TTL cache with `CACHE_TTL_SECONDS=3600` (1 hour).
Cache keys are namespaced as `pfm:<sha256(source,slug-or-ticker,start,end)>`.
Both factor histories and equity returns are cached, serialised as
pandas-JSON (`orient='split'`) — small enough to be fine in Redis.

The wrapper (`pfm.cache.RedisCache`) **degrades to a no-op** if Redis is
unreachable: it logs a warning once at startup and `get/set` return
`None`/`pass`. The API stays up. This matches the POC's "best effort"
posture — caching is an optimisation, not a correctness requirement.

The Redis container is internal-only (no host port mapped) to reduce blast
radius.

## Consequences

- 1-hour TTL is short enough that Polymarket data is never staler than an
  hour during interactive use, and long enough that consecutive demo runs
  are warm.
- Tests use `NullCache` to avoid taking a Redis dependency. Production
  uses `RedisCache`. Same API — `Protocol`-typed.
- A `flushdb` sweeps all caches; we accept that as fine for a POC.
- Promoting this to prod would require: per-key TTL based on whether the
  factor market is resolved (resolved → much longer TTL), and a cache-
  busting hook. Out of scope.
