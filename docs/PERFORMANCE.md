# Performance Engineering Guide

**Owner:** Platform/Performance · **Wave 13 baseline (W13-48)** · **Last verified:** 2026-05-16

This document describes how Prediction Terminal measures, targets, and protects
performance across its 271-endpoint FastAPI surface. It is written for engineers
who need to (a) reason about whether a regression is real, (b) decide whether a
new endpoint can be merged given its latency profile, or (c) debug a
production-shaped slowdown on a developer M1 laptop. Numbers are conservative —
treat them as the contract, not the ceiling.

---

## 1. Benchmark Methodology

The canonical latency benchmarks live in `api/tests/test_perf_benchmarks.py`,
shipped in **W11-43**. The harness combines `pytest-benchmark` with an
in-process `httpx.AsyncClient` driving the FastAPI ASGI app — no network
round-trip, so wall-clock noise is bounded by the asyncio scheduler and the OS
timer resolution rather than localhost socket setup.

**Principles:**

1. **Warm runs only.** Each benchmark calls the endpoint three times before
   timing begins so JIT-able paths (Pydantic v2 model construction, `numpy`
   imports, statsmodels HAC factories) are paid up-front. Cold-start metrics
   live in a separate suite (`test_cold_start.py`) and are not part of the SLO.
2. **Mocked upstreams.** `respx` intercepts Polymarket and yfinance traffic
   with deterministic fixtures from `tests/fixtures/factors/`. Real-network
   benchmarks would fail the "no flake on CI" requirement; we keep them as an
   opt-in `--live` marker that only runs against staging.
3. **Threshold-based assertions.** Each benchmark asserts `p95 < threshold`
   where `threshold` is documented inline. A drift > 25 % on p95 fails the
   build under the `slow` pytest marker, matching the regression gate in
   `.github/workflows/perf.yml`.
4. **Reproducibility:** `PYTHONHASHSEED=0`, `numpy.random.seed(42)`, and a
   pinned `pytest-benchmark` calibration of 5 rounds × 20 iterations. Outliers
   beyond 3σ are discarded automatically.
5. **Comparability:** all numbers in this document were measured on a baseline
   M1 Pro (8P+2E) with 16 GB RAM running Python 3.12.4, uvloop disabled,
   gunicorn `--workers 4 --worker-class uvicorn.workers.UvicornWorker`. CI runs
   on `macos-14` GitHub-hosted runners, which produce p95s ~1.4× higher; the
   CI thresholds embed that headroom.

---

## 2. Current SLOs Per Endpoint Group

SLOs are taken from `OVERNIGHT-RECAP.md` and the W11-43 thresholds. They are
**warm-path** numbers (Redis L2 populated, in-proc L1 hot). Cold-start
allowances are 3× the warm number for the first request after a worker reload.

| Endpoint group              | p50      | p95      | p99      | Error budget |
|-----------------------------|----------|----------|----------|--------------|
| `/health`, `/metrics/audit` | < 5 ms   | < 15 ms  | < 25 ms  | 99.95 %      |
| `/factors`, `/factors/all`  | < 30 ms  | < 80 ms  | < 150 ms | 99.9 %       |
| `/fit` (≤3 factors, warm)   | < 250 ms | < 600 ms | < 900 ms | 99.5 %       |
| `/fit` (5–8 factors, warm)  | < 500 ms | < 1.2 s  | < 1.8 s  | 99.0 %       |
| `/alpha-hub/leaderboard`    | < 80 ms  | < 200 ms | < 350 ms | 99.9 %       |
| `/alpha-hub/strategy/{id}`  | < 120 ms | < 250 ms | < 400 ms | 99.5 %       |
| `/terminal/jumps/{slug}`    | < 150 ms | < 400 ms | < 700 ms | 99.5 %       |
| `/terminal/jumps/cluster`   | < 300 ms | < 750 ms | < 1.2 s  | 99.0 %       |
| `/terminal/news` (cached)   | < 60 ms  | < 180 ms | < 300 ms | 99.5 %       |
| `/strategies/arb/stream`    | first byte < 250 ms; tick interval 2 s ± 200 ms | — | — | 99.0 % |
| `POST /reverse-finder/stream` | first SSE event < 1.5 s warm; complete < 4 s | — | — | 99.0 % |

The error budget is expressed as a fraction of requests within the SLO over a
rolling 30-day window; breaches feed into the alerting pipeline described in §9.

---

## 3. Hot-Path Latency Breakdowns

### 3.1 `POST /fit`

`/fit` is the most expensive synchronous endpoint and the most heavily
exercised. Warm-path budget for a 3-factor, 252-day window:

| Stage                                            | Budget    | Typical |
|--------------------------------------------------|-----------|---------|
| Request parsing + Pydantic validation            | 5 ms      | 2 ms    |
| Cache lookup (L1 in-proc, then L2 Redis)         | 8 ms      | 1 ms (L1 hit) / 6 ms (L2 hit) |
| Polymarket fetch (concurrent, async, cached)     | 80 ms     | 12 ms (cache) / 110 ms (cold) |
| yfinance fetch (batched per ticker set)          | 60 ms     | 18 ms (cache) / 220 ms (cold) |
| Timezone normalize + log-return computation      | 15 ms     | 9 ms    |
| Clipping (ε), VIF computation                    | 12 ms     | 6 ms    |
| `statsmodels` OLS with HAC `cov_type='HAC'`      | 60 ms     | 28 ms   |
| Attribution (`attribution.py`)                   | 20 ms     | 8 ms    |
| Response serialization (Pydantic + orjson)       | 10 ms     | 4 ms    |
| **Total warm p95**                               | **≤ 600 ms** | **~250 ms** |

The dominant variable cost is `statsmodels` OLS, which grows roughly linearly
in `(observations × factors²)` because of the HAC sandwich. Above 8 factors we
recommend `/fit/regularized` (ridge fallback) which caps quadratic blow-up.

### 3.2 `GET /terminal/jumps/cluster`

The cluster endpoint groups recent jump events across factors and returns a
DBSCAN-like grouping with sentiment overlays.

| Stage                                | Budget   | Typical |
|--------------------------------------|----------|---------|
| Resolve candidate jumps (cached)     | 80 ms    | 30 ms   |
| Pairwise time-distance matrix        | 120 ms   | 70 ms   |
| Sentiment scoring per cluster        | 200 ms   | 90 ms   |
| Headline dedup + ranking             | 60 ms    | 25 ms   |
| Serialization                        | 30 ms    | 12 ms   |
| **Total warm p95**                   | **≤ 750 ms** | **~300 ms** |

The sentiment step is the practical bottleneck. The hybrid NLP scorer
(`pfm/terminal/sentiment_nlp.py`) is LRU-cached to 10 000 entries, which
absorbs repeated headlines but cold-cache cases dominate the tail.

### 3.3 `GET /alpha-hub/leaderboard`

The leaderboard is the WOW-hero surface; latency directly affects perceived
quality of the landing page.

| Stage                                       | Budget   | Typical |
|---------------------------------------------|----------|---------|
| Read `web/data/alpha_strategies.json`       | 15 ms    | 6 ms    |
| Apply tier filter + freshness filter        | 5 ms     | 1 ms    |
| Join with live signal counters              | 25 ms    | 10 ms   |
| Compute on-the-fly Sharpe ranking           | 60 ms    | 22 ms   |
| Render envelope (orjson)                    | 10 ms    | 4 ms    |
| **Total warm p95**                          | **≤ 200 ms** | **~80 ms** |

The endpoint is prewarmed at lifespan start (see §8) so first-call latency
matches steady-state.

---

## 4. Cache Hit-Rate Targets

Two-tier cache, both monitored on `/metrics/audit`:

| Layer            | Backend      | TTL       | Hit-rate target |
|------------------|--------------|-----------|-----------------|
| L1 (in-process)  | `cachetools.TTLCache(maxsize=1024)` | 60 s   | > 60 % on warm endpoints |
| L2 (cross-proc)  | Redis 7 (`pfm:*` prefix)            | 300 s  | > 80 % on warm endpoints |
| L3 (durable)     | SQLite-backed factor snapshots      | 24 h   | > 95 % on factor reads   |

**Aggregate target: warm-endpoint cache hit-rate > 80 %.** This is enforced by
the `cache_hit_rate_alerts` job in `pfm.observability` — a 1-hour rolling
average below 70 % triggers an alert and the prewarmer (§8) is automatically
rerun. The Redis prewarm of 200 curated factors at startup is what allows the
hot `/fit` paths to keep their p95 under 600 ms; without it, the cold tail
extends to roughly 3 s.

Per-endpoint targets:

- `/fit` factor sub-fetches: > 85 %
- `/alpha-hub/leaderboard`: > 95 % (effectively static)
- `/terminal/jumps/cluster`: > 70 % (sentiment churn dominates)
- `/terminal/news`: > 90 %
- `/factors/all`: > 99 % (read-mostly)

---

## 5. Concurrency Model

**Topology:** 4 gunicorn workers × Uvicorn ASGI × single async event loop per
worker. Each worker has its own L1 cache and shares the L2 Redis. No threading
is used inside endpoints; CPU-bound work (statsmodels OLS, numpy operations)
runs on the event loop and is bounded by the per-endpoint budgets above.

**Why 4 workers?**

- An M1 Pro has 8 performance cores. We reserve 4 for OS + Redis +
  background tasks and run 4 workers, leaving each worker with effectively one
  dedicated core during sustained load.
- Each request's CPU profile is < 60 ms (warm `/fit`), well under the 100 ms
  asyncio "blocking" rule of thumb, so the event loop does not stall.
- More workers (e.g. 8) doubled memory footprint without improving p95 in
  W11-43 measurements: the cache duplication penalty wiped out the
  parallelism win.

**Async discipline:**

- All upstream IO uses `httpx.AsyncClient` with a shared per-worker pool.
- CPU-bound sync code (OLS) is **not** offloaded to `run_in_executor` — the
  overhead exceeded the win in our profile.
- SSE endpoints (`/strategies/arb/stream`, `/reverse-finder/stream`) use
  cooperative `asyncio.sleep` between ticks and yield every record to avoid
  starving siblings.

**Background tasks** (crypto sampler, arb scanner autostart, leaderboard
prewarm) run on the lifespan loop with bounded queues; if a task falls behind
it drops the oldest sample rather than backing up the event loop.

---

## 6. Capacity

**Sustained throughput on the M1 Pro reference machine: ~50 req/s** across a
representative mix (60 % `/factors` & `/alpha-hub/*`, 25 % `/terminal/*`, 15 %
`/fit`). This is measured by `scripts/load_smoke.py` with a 2-minute warmup
followed by a 10-minute steady-state phase.

Burst capacity is higher (peaks of ~140 req/s for 5 s without SLO breach)
because the in-proc cache absorbs duplicate reads. Beyond ~50 req/s the
limiting factor is Polymarket's upstream — not the FastAPI stack itself.

For production sizing, scale linearly until the upstream rate-limit becomes
the bottleneck (§7). A 4-vCPU production VM (c7g.xlarge) sustains ~110 req/s
under the same workload — about 2.2× the dev box, consistent with core count
and clock differences.

---

## 7. Bottlenecks Identified

1. **Polymarket rate limit (1000 req / 10 s).** This is the hard ceiling on
   total upstream-touching traffic. Past ~80 concurrent users hitting
   uncached factor combinations, we expect 429s. The current dedup layer
   keeps us at < 10 % of this budget under nominal load, but a thundering
   herd on a fresh deployment can spike the rate.
2. **yfinance per-ticker fetch.** yfinance's HTTP backend is not designed for
   high concurrency. A single ticker fetch costs ~110 ms cold; without
   batching, an 8-ticker portfolio costs 880 ms serially. Our batched wrapper
   compresses this to ~220 ms total but still represents the second-largest
   tail contributor.
3. **statsmodels HAC OLS for high-factor fits.** Quadratic in factor count.
   Mitigated by the ridge fallback above 8 factors and by ADR-0014's
   "max factors" guardrail in the API layer.
4. **Sentiment NLP cold path.** VADER initialization + financial-lexicon
   blend costs ~140 ms on first call. Amortized via process-level lazy
   init at worker boot.
5. **JSON serialization for very large responses.** `/factors/all` returns
   ~1.2 MB; orjson keeps this under 30 ms but it shows in p99 tails.
6. **Redis network round-trip on cold L1.** Local Redis adds 0.4–0.8 ms per
   hit; not a problem alone but multiplies with sub-fetches inside `/fit`.

---

## 8. Mitigations In Place

- **HTTP/2 connection pool** — `httpx.AsyncClient(http2=True,
  limits=Limits(max_keepalive_connections=64, max_connections=128))` shared
  per worker. Cuts Polymarket TLS round-trip from ~45 ms to ~12 ms on
  keep-alive hits.
- **Batched yfinance** — `pfm.market.yf_batch` groups same-period ticker
  requests within a 50 ms window and issues a single `yf.download(tickers=...)`
  call. Reduces per-ticker effective cost from 110 ms to ~30 ms amortized.
- **News deduplication** — Hash-and-skip on `(source, normalized_url, day)`
  before scoring; cuts sentiment work by ~40 % on the cluster endpoint.
- **Lifespan prewarm** — At app start we fetch the top 200 curated factors
  and pre-populate the leaderboard envelope. This is what makes the first
  `/fit` request after a deploy complete in ~3 s instead of ~12 s.
- **Negative-result caching** — Failed Polymarket lookups (resolved markets,
  missing slugs) are cached for 60 s to avoid re-hammering on retries.
- **Per-endpoint concurrency limits** — `/fit` is bounded to 12 in-flight
  requests per worker via an `asyncio.Semaphore`; excess requests wait briefly
  rather than starving the event loop.
- **Pydantic v2 + orjson** — ~3× faster than v1 + stdlib `json` on the
  representative payloads.
- **Cache-key normalization** — Factor lists are sorted and lowercased before
  hashing so `["AAPL","MSFT"]` and `["msft","aapl"]` collide. Cheap but
  measurably moves the hit-rate from ~70 % to ~85 %.

---

## 9. Monitoring

**Primary surface:** `GET /metrics/audit` returns rolling p50/p95/p99 per
endpoint group, cache hit-rates per layer, in-flight counts, and the last 50
slow-request traces (> p95 threshold).

Sample response shape:

```json
{
  "window_seconds": 300,
  "endpoints": {
    "POST /fit": {
      "count": 412,
      "p50_ms": 248,
      "p95_ms": 587,
      "p99_ms": 880,
      "error_rate": 0.0024
    },
    "GET /alpha-hub/leaderboard": {
      "count": 1844,
      "p50_ms": 78,
      "p95_ms": 191,
      "p99_ms": 312,
      "error_rate": 0.0
    }
  },
  "cache": {
    "l1_hit_rate": 0.67,
    "l2_hit_rate": 0.84,
    "l3_hit_rate": 0.96
  }
}
```

Internally, latencies are recorded by a Starlette middleware
(`pfm.observability.LatencyMiddleware`) that pushes per-request samples into a
per-worker rolling reservoir. Aggregation across workers happens via a small
Redis sorted-set keyed by `pfm:metrics:{endpoint}:{worker}`.

A Prometheus exporter is available at `/metrics/prom` (disabled by default;
enable with `PFM_METRICS_PROM=1`) for production scraping. Grafana dashboards
live in `ops/grafana/` and are described in ADR-0015.

Alert thresholds:

- p95 over SLO for 5 consecutive minutes → page
- Cache hit-rate < 70 % rolling 1 h → warn + auto-rerun prewarm
- 429s from upstream > 1 % of requests → warn
- Event-loop lag > 200 ms p95 (per-worker) → page

---

## 10. Performance Regression CI

W11-43 ships `tests/test_perf_benchmarks.py` and the matching
`.github/workflows/perf.yml` workflow that runs the suite under the `slow`
pytest marker on every PR touching `api/src/pfm/**`. Default `pytest` runs
**skip** slow tests; CI runs them explicitly.

**Gate logic:**

1. Run benchmarks against the PR branch with `pytest-benchmark --benchmark-json=pr.json`.
2. Compare against the most recent passing main-branch baseline (cached in
   `gh-cache` keyed by `main:<sha>`).
3. Fail the job if any tracked endpoint regresses p95 by > 25 % or breaches
   its absolute SLO threshold.
4. On regression, the failure comment posts the diff table and links to the
   most-likely commit (heuristic: largest diff in `pfm/{module}` touched
   between baselines).

The `slow` marker is also used to gate long-running tests like
`test_memory_leak_fit.py` (W13-30) and SSE concurrent load tests (W13-28) so
they don't slow the default development loop but still protect main.

When you legitimately need to relax a threshold — for example after a feature
that intentionally adds 50 ms because it computes additional VIF metrics —
update the inline `threshold_ms` in the benchmark and add a one-line note to
this document under the relevant endpoint in §3. **Do not silently regress
the SLO table in §2 without an ADR.**

---

## Appendix A — Running Benchmarks Locally

```bash
cd api
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_perf_benchmarks.py \
  -m slow \
  --benchmark-only \
  --benchmark-warmup=on \
  --benchmark-min-rounds=5
```

To compare against a saved baseline:

```bash
.venv/bin/python -m pytest tests/test_perf_benchmarks.py \
  -m slow \
  --benchmark-compare=.benchmarks/main-baseline.json \
  --benchmark-compare-fail=median:25%
```

## Appendix B — Glossary

- **p50 / p95 / p99** — the 50th / 95th / 99th percentile request latency
  over the measurement window.
- **SLO** — service-level objective; the latency contract we commit to.
- **Error budget** — fraction of requests permitted to miss the SLO before
  the system is considered unhealthy.
- **Warm path** — request where all caches are populated. Most production
  traffic is warm-path because caches are aggressively prewarmed.
- **Cold path** — first request after a deploy or cache eviction; allowed
  3× the warm budget for one request, then must converge to warm SLO.
