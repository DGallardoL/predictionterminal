# ADR-0014: Cache Stampede Protection via Per-Key Single-Flight Locks

- **Status:** Accepted
- **Date:** 2026-05-16
- **Authors:** Damian Gallardo
- **References:** ADR-0008 (cache tiering), ADR-0010 (anti-alpha rule)

## Context

A cache stampede (a.k.a. dog-pile, thundering-herd) occurs when N
concurrent callers miss the cache for the same key at the same instant
and each independently triggers the expensive backing computation. On
`/terminal/jumps/{slug}`, the backing computation fans out to GDELT,
Reddit, Hacker News, and a curated RSS pool — four upstream APIs, each
with its own rate limit, latency variance, and politeness window.

The empirically observed pathology: a leaderboard rebuild kicked off
**10 concurrent requests** for the same slug. With no protection, each
request issued its own four-way fan-out, producing **40 upstream
calls** instead of 4, exhausting the Reddit token bucket and inflating
p99 latency from ~3 s to >25 s. The expensive work was identical across
all 10 callers; only the first result was needed.

The same pathology applied to `CachePool` (Tier-16, in-process LRU over
factor-prewarm) and `RedisLock` (Tier-22, cross-process shared cache).
Stampedes consistently dominated cold-start and after-eviction traffic.

## Decision

Adopt **per-key `asyncio.Lock` single-flight** as the standard cache
stampede mitigation across the API. Semantics:

1. On cache miss, the first caller acquires the per-key lock and
   becomes the *leader*; it runs the expensive compute.
2. Concurrent callers for the **same key** block on the lock.
3. After acquiring, each follower does a **double-check** of the cache.
   If the leader populated it, the follower returns the cached value
   without recomputing.
4. The lock is released in a `finally` so that compute failures do not
   wedge followers; on failure followers also raise (no silent retry).

## Implementation locations

- **`pfm.cache_pool.CachePool.get_or_compute_async`** — primary
  in-process single-flight. Locks are stored in a `WeakValueDictionary`
  keyed by namespace+key so they are GC'd once no caller waits.
- **`pfm.redis_lock.RedisLock`** — cross-process variant using
  `SET NX PX` with a 30 s safety TTL and Lua-CAS unlock for the small
  number of endpoints where two workers may race (currently only the
  prewarm-on-startup path and `/alpha-hub/leaderboard`).
- **`pfm.terminal.jumps_cluster._install_news_gather_cache`** — scoped
  via a bool-anchored `ContextVar` so the single-flight is installed
  per request-tree (cluster requests reuse the leader's news fan-out
  across many slugs within the same request).

## Tests

T16 ships a 100-concurrent stampede test (`tests/test_cache_pool.py`)
that asserts `compute_call_count == 1` after `asyncio.gather` of 100
calls for the same key. T22 has a multi-process variant using two
event loops in subprocesses. `jumps_cluster` has a synthetic 10-caller
test mirroring the production pathology.

## Consequences

**Positive:** the leaderboard rebuild now issues exactly 4 upstream
calls per slug regardless of concurrency; Reddit token-bucket
exhaustion is gone; p99 on warm-after-burst dropped from 25 s to 3.2 s.

**Negative:** one lock per cache key costs memory (~200 B). Mitigated
by `WeakValueDictionary` and the existing LRU eviction. Followers pay
the leader's compute latency, but this is strictly better than each
recomputing. A slow leader briefly stalls followers; we accept this.

## Alternatives considered

- **Probabilistic early refresh (XFetch / Beladi-style)** — refreshes
  values stochastically before TTL expiry. Smooths load but does not
  solve the *cold-miss* stampede that dominates our traffic.
- **External lock service (Redis Redlock only)** — rejected as the
  sole mechanism: adds a network hop per cache miss and an operational
  dependency for in-process coordination that `asyncio.Lock` handles
  trivially. We keep Redlock only for cross-process needs.
- **Refresh-ahead background workers** — rejected: requires keeping a
  worker pool warm for 1228 factors × many endpoints; operational
  complexity outweighs benefit at POC scale.
