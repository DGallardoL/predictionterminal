# Cache Architecture

This document describes the three-tier cache that fronts every hot endpoint in
`pfm.*`. It is the operational companion to two ADRs:

- **ADR-0008** (`docs/adrs/ADR-0008-cache-tiering.md`, landed W11-45) — the
  decision to split caching into an in-process L1, a shared L2 Redis layer,
  and a lifespan prewarm.
- **ADR-0011** (`docs/adrs/ADR-0011-cache-stampede-singleflight.md`, landed
  W12-43) — the single-flight stampede-protection pattern implemented via
  per-key `asyncio.Lock` / `threading.Lock` inside `CachePool`.

Read those ADRs for the *why*. This document covers the *how* and the *what
to do when something is wrong*.

---

## 1. Overview — the three tiers

```
                ┌────────────────────────────────────────────────────┐
                │  FastAPI worker (gunicorn, 1 of N)                 │
                │                                                    │
   request ───▶ │  L1: CachePool in-process dict (per worker)        │ ── hit ──▶ response
                │  ├── heap-by-expiry eviction (O(log n))            │
                │  ├── threading.Lock + per-key sync/async locks     │
                │  └── pickle envelope PFMTC1\x00 (Wave-3 origin)    │
                │           │                                        │
                │           ▼ miss                                   │
                │  L2: Redis (shared across all workers + boxes)     │ ── hit ──▶ promote to L1 + response
                │  ├── pickle envelope PFMCP1\x00 (W11-45)           │
                │  ├── TTLs 60 s … 3 600 s (capped)                  │
                │  └── SETNX-locked refresh (single-flight)          │
                │           │                                        │
                │           ▼ miss                                   │
                │  upstream (Polymarket Gamma / Kalshi / yfinance)   │ ── compute ──▶ backfill L2 + L1
                └────────────────────────────────────────────────────┘

      Prewarm (lifespan): fire-and-forget asyncio tasks populate L1+L2 for
      ~41 jump slugs, vol-distribution top-15, factor-clusters, earnings-
      whisper dashboard, and the 200-curated-factor history set, BEFORE
      the first user request arrives.
```

The model is intentionally simple. We do not cache at the HTTP layer
(no Varnish/CDN); per-tab auth and per-query parameters would defeat it.
We also do not have a write-through L3 to disk — restarts are accepted as a
warm-period event because lifespan prewarm hides it for the demo paths.

---

## 2. L1 — `CachePool` (T16, `api/src/pfm/cache_pool.py`)

L1 is a per-worker in-process dict managed by `pfm.cache_pool.CachePool`.

### Key properties

- **Eviction**: heap-by-expiry. The entry closest to `expires_at` is evicted
  when `len(_d) >= l1_maxsize`. Heap fix-ups also drop everything already
  expired in one sweep. This is **not LRU** — LRU would cost a parallel
  dict and pay for itself only under unusual access patterns. For our
  workload (predominantly TTL-bound rather than capacity-bound) heap-by-
  expiry wins on simplicity.
- **Single-flight per key**: every key gets its own `asyncio.Lock` (async
  callers) or `threading.Lock` (sync callers), lazily created and stored
  in a guarded dict. N concurrent callers asking for a missing key cause
  exactly one upstream compute; the other N-1 wait on the lock and read
  the populated value.
- **Threadsafe stats**: a separate `_stats_lock` guards four counters
  (`l1_hits`, `l2_hits`, `misses`, `set_count`). Without this lock two
  threads racing on `self._stat_l1_hits += 1` can lose increments under
  the right interleaving even with the GIL — the read+write is two
  bytecodes.
- **`l1_maxsize`** defaults to 1 024 entries per pool. Hot pools
  (jumps, leaderboard) override to 4 096; rarely-used metadata pools
  drop to 256.
- **Namespace required**: `CachePool(namespace="term", ...)`. The
  namespace becomes the Redis key prefix `pfm:{namespace}:{key}` to
  guarantee no collisions between pools sharing a Redis instance.

### Public API summary

```python
pool = CachePool(namespace="term", redis=redis_backend, l1_maxsize=4096)
pool.get(key, default=None)              # L1 -> L2 -> default
pool.set(key, value, ttl=60)             # writes both layers
pool.delete(key)                         # both layers
pool.clear(prefix="news:")               # scoped wipe; returns N removed
await pool.get_or_compute_async(key, fn, ttl=60)   # single-flight
pool.get_or_compute(key, fn, ttl=60)               # sync single-flight
pool.stats                                          # dict snapshot
```

`get_or_compute_async` is the workhorse: 95% of new callsites use it. The
sync variant exists for pre-async code paths inside ETL scripts and
synchronous test helpers.

---

## 3. L2 — Redis

L2 is the cross-worker layer. With 4 gunicorn workers, isolation in L1
alone gave a ~25% effective hit rate even on supposedly hot keys; L2
pushes that to >80% in steady state.

### Envelope

Every L2 value is wrapped in a **versioned pickle envelope**:

```
b"PFMCP1\x00" + pickle.dumps({"v": 1, "data": value}, protocol=HIGHEST_PROTOCOL)
```

- `PFMCP1\x00` (7 bytes) is the magic prefix introduced in W11-45. It
  distinguishes `CachePool` Redis entries from the older
  `pfm.terminal.TTLCache` (`PFMTC1\x00`) entries that share the Redis
  instance.
- `v: 1` is the payload version. Schema changes to cached models that
  break unpickling must bump this (or rotate the magic to `PFMCP2`) and
  pre-emptively flush the old TTL window.
- `pickle.HIGHEST_PROTOCOL` is required so `pd.Series`,
  `pd.DataFrame`, `np.ndarray`, and our Pydantic models round-trip
  exactly. The original Wave-3 JSON-with-`default=str` approach silently
  stringified Series values and we lost three days to that bug. Do **not**
  reintroduce JSON serialisation here.

Decoding (`CachePool._decode_l2`) sniffs the magic byte and refuses
anything that doesn't match — corrupt or legacy entries are treated as a
miss and left to expire naturally.

### TTLs

| Source / data class                              | TTL    | Rationale                                  |
|--------------------------------------------------|--------|---------------------------------------------|
| Live orderbooks, trade tape                      | 60 s   | Sub-minute freshness matters for arb        |
| Gamma market metadata, jump series               | 300 s  | Changes slowly, hot read path               |
| Factor history, alpha-hub leaderboard            | 900 s  | Bounded recomputation cost                  |
| Resolved-market metadata, factor catalog         | 3 600 s| Effectively static                          |
| OpenAPI JSON                                     | 3 600 s| Only changes on deploy                      |

`CachePool._l2_set` **caps every TTL at 3 600 s** so a runaway caller
cannot pollute Redis with month-long entries.

### SETNX-locked refresh

Per-key `asyncio.Lock` handles the *intra-worker* stampede. The
*inter-worker* stampede is handled by Redis itself: callers that intend
to do a long-running compute may take a `SETNX pfm:{namespace}:lock:{key}`
with a TTL slightly longer than the expected compute time. ADR-0014
documents the contract. In practice most callsites rely on the
asyncio lock and treat the brief multi-worker double-compute as
acceptable — it costs at most O(workers) upstream calls per expiry
event versus O(callers).

### Graceful degradation

If the Redis backend raises any exception during construction or any
get/set/delete, `_mark_degraded()` flips `_redis_degraded = True` for the
lifetime of the worker. The pool keeps serving from L1, logs one
structured `cache_pool.redis_degraded` warning, and never raises through
to the caller. The `redis_degraded` flag is exposed on `pool.stats` (see
§8 below).

---

## 4. Lifespan prewarm

The highest-traffic endpoints are warmed by `asyncio.create_task` calls
inside the FastAPI `lifespan()` (see `api/src/pfm/main.py` around
line 129). These run *after* the app has started serving `/health` but
*before* any user request hits a cold path. Failed prewarm logs a
warning and never blocks startup.

| Endpoint                                       | Module                                          | Count        |
|------------------------------------------------|-------------------------------------------------|--------------|
| `/terminal/jumps/{slug}` (T17)                 | `pfm.terminal.jumps_prewarm.CURATED_TOP_SLUGS`  | 41 slugs     |
| `/terminal/vol-distribution`                   | `pfm.prewarm.prewarm_voldist`                   | top-15       |
| `/terminal/factor-clusters`                    | `pfm.prewarm.prewarm_factor_clusters`           | 1 payload    |
| `/alpha/earnings-whisper-dashboard`            | `pfm.earnings_whisper.run_forever_dashboard_prewarm` | continuous |
| 200 curated factor history series              | factor-history prewarm (opt-in via env)         | 200          |
| PM-VIX                                         | `pfm.pm_vix.run_forever_prewarm` (opt-in)       | 1 payload    |
| OpenAPI JSON                                   | inline `_openapi_prewarm` task                  | 1            |

Concurrency is capped per-prewarmer (`asyncio.Semaphore(8)` is the
default) to keep us under Polymarket's 1 000-req/10 s budget even while
the server is otherwise serving traffic.

Opt-in flags (most prewarmers are ON in production, OFF by default in
dev/tests to keep `pytest` fast):

- `PFM_JUMPS_PREWARM_ENABLED=1`
- `PFM_FACTOR_PREWARM_ENABLED=1`
- `PFM_PMVIX_PREWARM_ENABLED=1`
- `PFM_EARNINGS_PREWARM_ENABLED=1`
- `PFM_CRYPTO_5MIN_ENABLED=1`

---

## 5. Migration status

`CachePool` was introduced in W11-14 to replace a half-dozen ad-hoc
caches. Currently migrated callsites:

- `pfm.sources.manifold` — `_SEARCH_CACHE`, `_MARKET_CACHE` (W11-14)
- `pfm.sources.kalshi` — `_MARKET_CACHE` (W11-14)
- `pfm.terminal.quote` — `_GAMMA_MARKET_CACHE` (W11-14)

Not yet migrated (carries its own hand-rolled TTL dict; safe but should
be unified):

- `pfm.terminal.__init__.TTLCache` — Wave-3 in-process cache used by
  the older terminal endpoints. Uses `PFMTC1\x00` envelope.
- `pfm.arb_scanner` — module-level dict keyed by `(venue, slug)`.
- `pfm.strategies_crypto_router` — short-TTL Binance snapshot cache.

The migration plan is captured in `pfm.cache_pool` module docstring; do
not migrate in the same commit as a feature change.

---

## 6. Operations

### Endpoints

- `GET /admin/cache-stats` (W12-17, `pfm.admin.cache_stats_router`) —
  aggregated per-pool stats. Walks `pfm.*` modules at request time,
  reflects every attribute of type `CachePool`, and returns
  `{namespace, l1_hits, l2_hits, misses, set_count, l1_size, redis_degraded}`.
- `GET /metrics/audit` (`pfm.metrics_router`) — request-level metrics
  including cache annotations (`X-Cache: hit-l1 | hit-l2 | miss`) and
  endpoint p50/p95 latencies.

### Manual invalidation

```python
from pfm.terminal.quote import _GAMMA_MARKET_CACHE
_GAMMA_MARKET_CACHE.delete("trump-2024-presidential-election")
# or scoped:
_GAMMA_MARKET_CACHE.clear(prefix="trump-")
```

`clear(prefix=...)` returns the number of L1 entries removed and
best-effort scans Redis with `redis.scan_iter(match="pfm:{ns}:{prefix}*")`.

### Invalidation triggers (automatic)

- **Resolved-market webhook** (when present): flushes the affected slug
  from `_GAMMA_MARKET_CACHE` + `_MARKET_CACHE` (kalshi/manifold).
- **Daily factor catalog refresh**: clears the factor-history prewarm
  pool with prefix `factor:`.
- **Lifespan teardown**: pools are dropped with the worker; Redis
  entries survive and the next worker hot-reads them.

### Restarting gunicorn

A clean `kill -HUP <gunicorn-master>` re-runs lifespan and re-prewarms.
**Do not restart `:8000` without writing to
`.coordination/restart-requests.txt`** — it is shared by every browser
tab.

---

## 7. Request flow diagram

```
Request
   │
   ▼
L1 lookup ─── hit ────────────────────────────────────────▶ return value
   │
   ▼ miss
single-flight gate (per-key asyncio/threading lock)
   │
   ▼
L2 lookup (Redis, PFMCP1 envelope) ─── hit ─── promote to L1 (30 s) ───▶ return value
   │
   ▼ miss
compute fn() ─── upstream call (Polymarket/Kalshi/yfinance/internal) ──▶ value
   │
   ▼
backfill L1 (TTL) + L2 (min(TTL, 3 600 s)) ──────────────▶ return value
```

The lock is released as soon as `set()` returns; subsequent waiters
re-read from L1 without recomputing.

---

## 8. Stats interpretation

`pool.stats` returns:

```python
{
  "l1_hits": int,        # served from in-process dict
  "l2_hits": int,        # served from Redis (then promoted to L1)
  "misses": int,         # had to recompute / call upstream
  "set_count": int,      # number of write operations
  "l1_size": int,        # current number of L1 entries
  "redis_degraded": bool,# True ⇒ L2 turned off for this worker
}
```

Interpretation guide:

- **`hit_rate = (l1_hits + l2_hits) / (l1_hits + l2_hits + misses)`**.
  Target: `>0.7` for hot pools, `>0.4` for cold/long-tail pools. Below
  0.3 suggests TTLs are too short or the key cardinality exceeds
  `l1_maxsize`.
- **`l2_hits` significance**: a high L2-to-L1 ratio (say `l2_hits >
  l1_hits / 3`) means the per-worker isolation is doing real work —
  cross-worker traffic is being saved. This validates the L2 layer
  pays for its serialization cost.
- **`redis_degraded: true`** is **never normal in production**. The flag
  is sticky for the worker's lifetime; restart the worker to retry. If
  it sets repeatedly, check Redis health (`redis-cli ping`, memory
  pressure, network).
- **`l1_size` near `l1_maxsize`** in a pool whose hit rate is also low
  means the working set exceeds the cap — bump `l1_maxsize` or shorten
  TTLs to allow eviction.
- **`set_count` rising while `l1_hits` flat** indicates a
  thundering herd has slipped past single-flight — check that callers
  are routing through `get_or_compute_async` and not `set()`-ing
  directly in parallel.

---

## 9. Common operations

```bash
# View aggregated cache stats
curl -s http://localhost:8000/admin/cache-stats | jq

# Per-pool hit rate (one-liner)
curl -s http://localhost:8000/admin/cache-stats \
  | jq '.pools[] | {ns: .namespace,
                    hit_rate: ((.l1_hits + .l2_hits) /
                              ((.l1_hits + .l2_hits + .misses) | if . == 0 then 1 else . end))}'

# Inspect Redis keys for a namespace
redis-cli --scan --pattern 'pfm:term:*' | head -20

# Manually evict a key from Redis
redis-cli DEL 'pfm:term:trump-2024-presidential-election'

# Clear an entire namespace (Redis side) — heavy, use sparingly
redis-cli --scan --pattern 'pfm:term:*' | xargs -L 1 redis-cli DEL
```

For Python-level invalidation use the targeted pool's `.delete()` /
`.clear()` rather than blowing away Redis directly — the L1 dict on
each worker will otherwise serve stale data until its own TTL fires.

---

## 10. Capacity planning

### L1

Per-pool footprint upper bound:

```
l1_maxsize × avg_entry_size_bytes
```

Empirically measured averages (May 2026):

| Pool                          | l1_maxsize | avg size (KB) | upper bound |
|-------------------------------|------------|---------------|-------------|
| `term.jumps`                  | 4 096      | 12            | ~48 MB      |
| `term.gamma_market`           | 4 096      | 3             | ~12 MB      |
| `term.voldist`                | 256        | 8             | ~2 MB       |
| `kalshi.market`               | 2 048      | 2             | ~4 MB       |
| `manifold.market`             | 2 048      | 2             | ~4 MB       |
| `manifold.search`             | 1 024      | 5             | ~5 MB       |
| All others (combined)         | n/a        | n/a           | <20 MB      |

Total per-worker L1 ceiling: ~100 MB. With 4 workers, ~400 MB of RSS is
attributable to caches in the worst case. Current production RSS per
worker hovers around 350–450 MB total, so caches are a meaningful
fraction but not dominant.

### L2 (Redis)

Each pool's L2 footprint is roughly:

```
N_distinct_keys × (avg_value_bytes + ~80 B overhead)
```

A 5 GB `maxmemory` configuration with `maxmemory-policy allkeys-lru` is
sufficient for current scale (~1.2 GB peak measured during demos). The
LRU eviction policy is a *fallback*; our explicit TTLs should normally
keep us well under the cap. Alert if Redis `used_memory` exceeds 80%
of `maxmemory` — that means TTL discipline has slipped somewhere.

### Scaling triggers

- **`l1_size == l1_maxsize` and `hit_rate < 0.5` for >1 h** on any hot
  pool: bump `l1_maxsize` (2× is the safe step).
- **Redis evictions rising** (`redis-cli info stats | grep evicted_keys`):
  shorten the longest TTLs first; resolved-market metadata at 3 600 s
  is usually the easiest to compress.
- **`misses` rising for a stable user population**: check upstream
  health — Polymarket Gamma flakiness shows up here first.
- **`redis_degraded: true` across all pools**: hard alert. Page the
  on-call. The system continues to serve from L1 but cross-worker
  consistency is lost and rate-limit pressure on upstreams rises.

---

## References

- `docs/adrs/ADR-0008-cache-tiering.md` — three-tier rationale (W11-45)
- `docs/adrs/ADR-0011-cache-stampede-singleflight.md` — single-flight
  pattern (W12-43)
- `docs/adrs/0004-redis-cache-ttl.md` — original Redis TTL choice
- `api/src/pfm/cache_pool.py` — `CachePool` implementation
- `api/src/pfm/admin/cache_stats_router.py` — `/admin/cache-stats`
- `api/src/pfm/metrics_router.py` — `/metrics/audit`
- `api/src/pfm/prewarm.py` — vol-distribution + factor-clusters prewarm
- `api/src/pfm/terminal/jumps_prewarm.py` — top-41 slugs prewarm
