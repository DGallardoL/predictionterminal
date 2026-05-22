# Performance Audit — W13-LUPA-3

**Date:** 2026-05-16
**Scope:** Production-readiness performance audit. Pure read-only; no code changes.
**Method:** Manual endpoint sampling (5 calls each), static scan of hot files, pattern-matching for unbounded state and sync-I/O-in-async.
**Server probed:** `http://localhost:8000` (gunicorn/uvicorn 4 workers per `Procfile`).

> Companion docs: [`docs/PERFORMANCE.md`](PERFORMANCE.md) (existing perf notes), [`docs/LAUNCH_AUDIT.md`](LAUNCH_AUDIT.md), [`docs/CODE_QUALITY_AUDIT.md`](CODE_QUALITY_AUDIT.md).

---

## 1. Endpoint latency sample

`/metrics/audit` and `/admin/cache-stats` were **not reachable on the live server** at audit time (the routes are wired in `api/src/pfm/main.py:2637-2693` but the long-running gunicorn instance pre-dates the wire-up and was deliberately not restarted per `.coordination/PROTOCOL-V2.md`). All numbers below are manual `curl` samples — 5 calls each, mean wall-clock in ms.

### 1a. Trivial / cached endpoints (fast)

| Endpoint | Avg | Notes |
|---|---|---|
| `/health` | 2.2 ms | constant-time JSON |
| `/factors` (paginated) | 2.8 ms | first-page cached |
| `/factors/all` | 5.1 ms | 524 KB body, gzipped to 78 KB by `GZipMiddleware` (`main.py:1471`) |
| `/openapi.json` | 2.4 ms | ETag + `_OPENAPI_CACHE` hit (`main.py:1486`) — correctly capped |
| `/terminal/overview` | 2.9 ms | static curated payload |
| `/strategies/arb/state` | 2.2 ms | served from `_FALLBACK_CACHE` / SSE tick buffer |
| `/alpha-hub/leaderboard` | 2.0 ms | reads `web/data/alpha_strategies.json` |
| `/news/feed` | 1.9 ms | RSS cache hit |
| `/alpha/strategies` | 7.6 ms | JSON file read with TTL |
| `/alpha-hub/strategy/{pair_id}` | 5.2 ms | embedded spread series |

### 1b. Live / network-bound endpoints

Cold = first call, warm = same URL re-hit immediately.

| Endpoint | Cold avg (5 calls) | Warm (2nd-hit) | Comment |
|---|---|---|---|
| `/terminal/quote/{slug}` | 171 ms | 7 ms | Polymarket Gamma + CLOB combine; TTL works |
| `/terminal/news/{slug}` | 376 ms | 2 ms | GDELT + RSS; warm extremely fast |
| `/terminal/orderbook/{slug}` | 97 ms | 20 ms | live CLOB book |
| `/terminal/news-impact/{slug}` | **1 119 ms** | 2 ms | heaviest cold path — see Finding **P1** |
| `/strategies/crypto/snapshot` | 189 ms | n/a (TTL=30s) | Binance 10-pair fan-out |
| `/strategies/arb/stream` | 5 005 ms | n/a | SSE — 5 s by design (2 s ticks ×2.5) — OK |
| `/sources/health` | **873 ms** | 755 ms warm | see Finding **P2** |

**Headline:** warm-cache responses dominate (sub-20 ms across the board). The pain is the **cold path on news-impact** and the **uncached `/sources/health`**.

---

## 2. Hot-file scan (files >1000 LOC)

Verified line counts:

| File | LOC | Verdict |
|---|---|---|
| `api/src/pfm/main.py` | 2 694 | larger than the CLAUDE.md-reported 1 534 — see **P3** |
| `api/src/pfm/strategies_router.py` | 2 280 | all 34 routes are sync `def` (FastAPI threadpool); CPU-bound `statsmodels` calls — acceptable |
| `api/src/pfm/regression_router.py` | 2 178 | not in original hot-file list but qualifies |
| `api/src/pfm/arb_scanner.py` | 1 450 | uses `ThreadPoolExecutor(max_workers=6)` in `compute_arb_spreads` (`arb_scanner.py:407-466`) — well-bounded |
| `api/src/pfm/schemas/strategies.py` | 1 195 | pure Pydantic; no perf risk |
| `api/src/pfm/decay_monitor.py` | 1 162 | per-call `httpx.Client(timeout=7.0)` construction (`decay_monitor.py:466`) — minor |
| `api/src/pfm/replay_mode.py` | 1 143 | calls `asyncio.run()` inside sync route (`replay_mode.py:651`) — see **P4** |
| `api/src/pfm/live_signals_job.py` | 1 111 | background only |
| `api/src/pfm/strategies_arb_router.py` | 1 083 | `_DETECTION_SEEN` IS bounded — `_DETECTION_MAX` enforced at line 480 (good) |

> The CLAUDE.md "Current state" section reports `main.py = 1534` but the actual file is **2 694 LOC**. Either the refactor regressed or the doc is stale. Update **CLAUDE.md** scale section after this audit.

---

## 3. Cache hit-rate estimation (live `/admin/cache-stats` unreachable)

The `CachePool` class (`api/src/pfm/cache_pool.py:95`) tracks `_stat_l1_hits / _stat_l2_hits / _stat_misses` (line 142-145). Without the live endpoint we estimate from warm-vs-cold timings:

- **Terminal quote / news / orderbook:** warm < 25 ms vs cold 100-1100 ms → cache hits in the 95-99 % range once a slug is touched twice.
- **Factor lookups:** `_OPENAPI_CACHE`, `factors_theme_leaderboard_router._CACHE`, `factors_related_router._CACHE`, `factors_correlation_matrix_router._RESPONSE_CACHE` — all have TTL-lazy eviction (read-time `expires_at` check) but **no size cap**. See **P5**.
- **News-search:** `news_search_router._CACHE` has TTL eviction but **no size cap** (`news_search_router.py:188`) — unique queries accumulate until process restart.

---

## 4. Memory: unbounded state

Findings from grepping module-level `dict[...]` declarations that look like caches/registries, then verifying eviction.

### P5 — Per-key cache dicts with no size cap (`severity: MEDIUM`)
- `api/src/pfm/news_search_router.py:188` `_CACHE: dict[(str,str,bool), _Entry]` — lazy TTL eviction only; user-controlled query string is part of the key. Worst case: many unique queries between restarts.
- `api/src/pfm/factors_theme_leaderboard_router.py:132` `_CACHE` — bounded by (`theme`, `n`, `flag`) cardinality (~ low hundreds), low risk.
- `api/src/pfm/factors_related_router.py:98` `_CACHE` — bounded by (`slug`, `n`) cardinality.
- `api/src/pfm/factors_correlation_matrix_router.py:181` `_RESPONSE_CACHE` — bounded by (`slug`, `window`, `top_n`) cardinality.
- `api/src/pfm/strategies_crypto_router.py:40` `_CACHE` and `:331` `_VOL_CACHE` — keyed by symbol; ≤10 entries because `PAIRS` is hard-coded. **Not a real leak.**

**Fix:** wrap each via `cachetools.TTLCache(maxsize=1024, ttl=...)` — keeps the lazy-TTL semantics but adds an LRU cap. ≈10 lines per file.

### P6 — `CachePool._async_locks` / `_sync_locks` grow forever (`severity: MEDIUM-HIGH`)
- `api/src/pfm/cache_pool.py:149` and `:151` — one `asyncio.Lock` (or `threading.Lock`) is created **per cache key** and **never removed** (no `pop`, no `clear` anywhere in the file).
- Confirmed by grep:
  ```
  149: self._async_locks: dict[str, asyncio.Lock] = {}
  151: self._sync_locks:  dict[str, threading.Lock] = {}
  381: self._async_locks[key] = lock      # only insert
  389: self._sync_locks[key]  = lock      # only insert
  ```
- For pools keyed by user-controlled strings (slug, query) this is a slow but real memory leak across long-lived workers. Lock objects are ~250 B each so 100 k unique keys → ~25 MB **per pool, per worker**.
- **Fix:** evict the per-key lock immediately after the single-flight completes, or back the lock dict with `weakref.WeakValueDictionary` (locks the GC can reclaim once the last waiter releases). 5-line change in `_get_async_lock` / `_get_sync_lock`.

### P7 — `_OPENAPI_CACHE` is bounded but `clear()`-before-`set` (`severity: LOW / NIT`)
- `api/src/pfm/main.py:1486` — bounded by design (`_OPENAPI_CACHE.clear()` at line 1500 before insert). OK.

---

## 5. Async boundaries (sync I/O inside `async def`)

Custom AST-style scan: looked for `requests.*`, `httpx.get/post/...`, `httpx.Client(`, `time.sleep(` appearing inside any `async def` body without an `await`. **Result: only 1 hit**, and it's a false-positive:

- `api/src/pfm/main.py:217` — `app.state.http = httpx.Client(...)` inside `async def lifespan(app)`. **Acceptable** — it is constructing the shared sync client at startup, not doing I/O.

Sync `httpx.get` calls in `main.py:1894` and `:1931` are inside **sync `def`** routes (`btc_arb_active_market`, `btc_arb_midpoint`) — FastAPI runs those in a threadpool, so they don't block the event loop. Acceptable, though see **P8** below.

### P8 — `asyncio.run()` / thread-bounded loop swap inside request paths (`severity: MEDIUM`)
- `api/src/pfm/factors.py:325-346` — when a sync helper is called inside a running event loop, it spawns a brand-new `threading.Thread` + `asyncio.new_event_loop()` per call. Allocation overhead is non-trivial (~3-5 ms) and exhausts threadpool slots under burst.
- Same pattern: `api/src/pfm/replay_mode.py:651`, `api/src/pfm/earnings_whisper.py:311`, `api/src/pfm/news_causal_chain.py:726`, `api/src/pfm/main.py:2361` (`_loop.run_until_complete(fut)`).
- **Fix:** make the calling sites `async def` and `await` the coroutine directly. Where that's infeasible, share one bg event-loop thread across the process (`asgiref.sync.async_to_sync` does this correctly).

---

## 6. Other observations

### P9 — `/sources/health` is uncached (`severity: LOW`)
- `api/src/pfm/sources/health_router.py:60-75` — every request hits **all 6 upstream probes** in parallel via `asyncio.gather` (correctly implemented, `sources/health.py:251-271`). Wall-clock ≈ slowest probe = ~750 ms.
- With 4 workers this is fine, but a status-page hitting it every 5 s wastes upstream rate-limit budget.
- **Fix:** wrap the route in a 15-30 s `TerminalCache` entry. 4 lines.

### P10 — Per-request `httpx.Client(...)` construction (`severity: LOW`)
- `api/src/pfm/decay_monitor.py:466`, `api/src/pfm/arb_scanner.py:426`, several other places.
- Each construction does SSL context setup. The app already exposes `app.state.http` (`main.py:217`); pass it down instead.

### P11 — `pd.read_csv(StringIO(text))` is sync and called from `async` paths (`severity: LOW`)
- `api/src/pfm/sources/stooq.py:115`, `api/src/pfm/sources/fred.py:240`.
- Read paths terminate in sync helpers; verified the async wrappers above them push the call into a thread (no event-loop block). **No action needed**, but flag if you add a high-QPS factor backed by either source.

### P12 — `_DETECTION_HISTORY` / `_DETECTION_SEEN` are correctly bounded
- `api/src/pfm/strategies_arb_router.py:480-484` enforces `_DETECTION_MAX`. Good defensive code — note as positive.

---

## 7. Top-10 perf wins, prioritised

| Rank | Fix | File:line | Severity | Effort | Expected win |
|---|---|---|---|---|---|
| 1 | Reload server to mount `/admin/cache-stats` + `/admin/cache-invalidate` so this audit can be re-run with real hit-rates (and ops can debug live) | `api/src/pfm/main.py:2637-2693` (already wired; just needs restart) | HIGH | 0 (ops) | unlocks observability |
| 2 | Fix `CachePool._async_locks` / `_sync_locks` leak with `WeakValueDictionary` or post-release `pop` | `api/src/pfm/cache_pool.py:149-151, 376-390` | MEDIUM-HIGH | S | bounded memory in long-running workers |
| 3 | Cap `news_search_router._CACHE` with `cachetools.TTLCache(maxsize=2048)` | `api/src/pfm/news_search_router.py:188` | MEDIUM | XS | bounded memory |
| 4 | Cache `/sources/health` for 15-30 s | `api/src/pfm/sources/health_router.py:60` | LOW | XS | -800 ms p50 on status-page polling, less upstream load |
| 5 | Investigate `/terminal/news-impact/{slug}` cold path (~1.1 s) — likely sequential GDELT + sentiment NLP — and parallelise | `api/src/pfm/terminal/news_impact*.py` | MEDIUM | M | cold p95 from 1.1 s → ~400 ms |
| 6 | Reuse `app.state.http` instead of constructing per-call `httpx.Client` | `api/src/pfm/decay_monitor.py:466`, `api/src/pfm/arb_scanner.py:426` | LOW | S | -3-10 ms/call + connection pooling |
| 7 | Replace `asyncio.run` / thread-loop swap pattern with `asgiref.sync.async_to_sync` or refactor callers to async | `api/src/pfm/factors.py:325-346` + 4 sibling sites | MEDIUM | M | reclaim threadpool slots under burst |
| 8 | Cap remaining size-uncapped factor caches (`factors_theme_leaderboard`, `factors_related`, `factors_correlation_matrix`) for safety | `factors_theme_leaderboard_router.py:132`, `factors_related_router.py:98`, `factors_correlation_matrix_router.py:181` | LOW | XS | defense-in-depth |
| 9 | Update CLAUDE.md "Current state" — `main.py` is now 2 694 LOC (was reported 1 534); audit shows split into routers is incomplete | `CLAUDE.md` | DOC | XS | accurate priors for next agents |
| 10 | Add a `slowest 10 routes (p50/p95)` panel to `/admin/cache-stats` or wire the existing `metrics_router` into a `/metrics/audit` view | `api/src/pfm/metrics_router.py`, `api/src/pfm/admin/cache_stats_router.py` | LOW | S | ongoing observability without `curl` benchmarks |

---

## 8. What was NOT audited (out of scope / blocked)

- Real cache hit-rates (server not reloaded — would have required violating coordination protocol).
- Background-job (`live_signals_job`, `decay_monitor.run_forever`, alpha-tier regen) memory and CPU over hours — needs a long-running profiling session.
- Plotly client-side bundle size on `web/` (frontend audit; covered by W13-LUPA-2 UX audit).
- Redis L2 hit-rate vs L1 (requires `/admin/cache-stats` live).
- Concurrent-load behaviour (no load-test run; `wrk`/`vegeta` was not invoked).

---

## 9. Methodology notes

1. `curl` micro-bench: 5 sequential calls, mean wall-clock. Single-client, no parallelism — measures **server-side** latency including JSON serialisation but excluding TLS handshake (HTTP on localhost).
2. Static scan: custom Python AST walker (regex-based) over all `async def` bodies in `api/src/pfm/**.py` checking for sync I/O patterns. Result: 1 false-positive, 0 real hits.
3. Module-level dict scan: regex `^(_[A-Z_]+)\s*[:=]` in 168 modules, cross-checked each suspect against grep for `del`, `pop`, `clear`, `maxsize`, `MAX_`, `TTL`.
4. No code was modified. No services were restarted. Coordination claim was appended to `.coordination/active-edits.json` as `agent-w13-lupa-3` (scope `W13-LUPA-3 perf-audit-readonly-creates-PERFORMANCE_AUDIT.md-only`).
