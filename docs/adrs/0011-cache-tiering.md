# ADR-0011: Cache Tiering — L1 In-Process + L2 Redis + Lifespan Prewarm

## Status
Accepted (2026-05-16)

## Context
The API runs behind 4 gunicorn workers, each holding its own in-process state. Hot endpoints (`/terminal/jumps/{slug}`, `/alpha-hub/leaderboard`, `/terminal/vol-distribution`, `/terminal/cluster`, `/reverse-finder/stream`) compute expensive payloads (5k+ candles, factor cross-products, GBM Monte Carlo) and depend on shared upstreams (Polymarket, yfinance, Binance). Two failure modes dominated cold-start tail latency:

1. **Cache stampedes** — when a popular slug expired, N workers each refetched and recomputed the same payload concurrently, producing 10–13 s P99 spikes and burning Polymarket rate-limit budget.
2. **Per-worker isolation** — even with in-process TTLCache, worker A's cached payload was invisible to worker B; cold-pinned tabs got cold latency repeatedly.

Wave-3 introduced a pickle envelope (`PFMTC1\x00`) to make in-process entries discoverable; Wave-11 unifies it with Redis.

## Decision
Adopt a 3-tier strategy fronted by `pfm.cache_pool.CachePool` (T16):

1. **L1 in-process TTLCache per worker** — sub-millisecond hits, no serialization cost. Pickle envelope `PFMTC1\x00` (Wave-3 fix). Sized per-endpoint (typically 256–4096 entries).
2. **L2 Redis** — shared across all workers. Pickle envelope `PFMCP1\x00`. TTLs range 60 s (orderbooks, live signals) to 3600 s (resolved-market metadata, factor catalogs). Single-flight via `SETNX` lock plus a per-key `asyncio.Lock` to prevent intra-worker thundering herds.
3. **Lifespan prewarm** — for the highest-traffic endpoints (top-41 jump slugs, vol-distribution, factor-clusters, A-tier alpha cards, 200 curated factor series), fire-and-forget `asyncio` tasks during FastAPI lifespan populate L1 + L2 *before* the first request arrives.

## Consequences
+ Cold-cache latency on prewarmed endpoints drops from 3–13 s to <50 ms warm.
+ Cross-worker consistency for shared computed state (no more "this tab is fast, that one is slow").
+ Stampede protection: at most one upstream call per (key, TTL window).
+ Polymarket rate-limit headroom restored (peaks down ~70 %).
- Two pickle formats now coexist in storage (`PFMTC1` in-process, `PFMCP1` Redis); cross-tier reads must sniff the magic byte.
- Startup is 5–15 s longer (prewarm phase); failed prewarm logs warnings but never blocks `/health`.
- Pickle has version risk — any schema change in cached models needs a TTL flush or magic-byte bump.

## Alternatives Considered
- **Single-tier in-process only**: rejected — no cross-worker sharing; cache hit rate cratered with 4 workers.
- **Single-tier Redis only**: rejected — every read pays serialization + network roundtrip; hot paths regressed to 5–10 ms vs <1 ms L1.
- **Memcached instead of Redis**: rejected — Redis already deployed for queue/pubsub; second store adds ops surface.
- **ServiceWorker / CDN edge cache**: out of scope for the POC; would require auth/cookie story.
- **functools.lru_cache only**: insufficient — no TTL, no eviction policy, no shared state.

## References
- PR#W11-45 (this ADR) + W11-14 (CachePool migration)
- `api/src/pfm/cache_pool.py`, `api/src/pfm/cache_envelope.py`
- `docs/OVERNITE-RECAP.md` (perf measurements)
- ADR-0004 (original Redis TTL choice)
