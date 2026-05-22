# System Architecture — Prediction Terminal

> **Status:** Living document. Reflects post-Wave-9 + Wave-13 state as of 2026-05-16.
> **Audience:** New contributors, reviewers, and the next Claude Code session.
> **Companion documents:** [`docs/CACHE.md`](CACHE.md), [`.coordination/PROTOCOL-V2.md`](../.coordination/PROTOCOL-V2.md), [`docs/adrs/`](adrs/).

This document gives a top-down tour of how the **Prediction Terminal** product
is wired together. It is intentionally denser than the original POC
`architecture.md` (which described the three-service docker-compose stack);
that file remains valid as the deployment skeleton, but the system has grown
into a **three-mode product** with 280+ HTTP endpoints, two cache tiers, a
multi-session coordination protocol, and a small zoo of real-time streams.
Where this document and `architecture.md` disagree on a point of detail,
**this one is authoritative.**

---

## 1. Bird's-eye view

```
                                  ┌───────────────────────────────────────────┐
                                  │   UPSTREAM DATA SOURCES                   │
                                  │   Polymarket · Kalshi · yfinance · FRED   │
                                  │   GDELT · Reddit · HN · RSS · CoinGecko   │
                                  └───────────────────────────────────────────┘
                                                     ▲
                                                     │ HTTP/2 + REST
                                                     │
┌────────────────────────┐                ┌──────────┴────────────┐                ┌───────────────────┐
│ Browser (vanilla JS)   │                │ FastAPI app (:8000)   │                │ Redis (:6379)     │
│  - Regression tab      │  HTTPS / SSE   │  - 4 gunicorn workers │   pickle env   │  - L2 cache       │
│  - Strategies / α Hub  │◀──────────────▶│  - 280+ endpoints     │◀──────────────▶│  - SingleFlight   │
│  - Terminal data hub   │                │  - lifespan prewarm   │                │  - leaderboard ZSET│
│  - Plotly (CDN)        │                │  - async + sync mix   │                └───────────────────┘
└────────────────────────┘                └───────────┬───────────┘
        ▲                                             │
        │ static files                                │ background tasks
        │                                             ▼
┌───────┴───────────────┐                ┌────────────────────────┐
│ Frontend httpd (:8080)│                │ Schedulers / Samplers  │
│  nginx — index.html   │                │  - crypto5min sampler  │
│  + css/ + js/         │                │  - arb engine (opt-in) │
└───────────────────────┘                │  - decay monitor       │
                                         └────────────────────────┘
```

Three operational concerns separate cleanly:

1. **Frontend** is static: a single `web/index.html` (~1.6 MB), modular
   `web/css/<feature>.css` and `web/js/<feature>.js` files, all served by a
   thin nginx (`web:8080`). There is no build step, no bundler, no React.
2. **Backend** is one FastAPI process per worker (4 × gunicorn + `UvicornWorker`)
   that owns all upstream IO, caching, quant math, strategy registry, arb
   scoring, sentiment scoring, and SSE fan-out.
3. **Redis** is a shared L2 cache and a few specialised data structures
   (leaderboard ZSET, single-flight lock keys, prewarm fingerprints).

There is **no relational database** — by design (ADR-0005). State that
matters across runs lives in Redis (volatile) or on disk as JSON / Parquet
fixtures (`web/data/*.json`, `arbstuff/dashboard_state.json`,
`tests/fixtures/factors/*`).

---

## 2. Frontend

The frontend is **deliberately boring**: vanilla HTML + JS + Plotly from
CDN. The cumulative payload (including the largest single asset,
`index.html`) is ≤ 2 MB on a cold load.

### Layout

```
web/
├── index.html              # 1.6 MB single-page app: all 3 modes, all tabs
├── config.js               # API base URL, feature flags, demo defaults
├── plotly-theme.js         # shared Plotly layout/colour tokens
├── css/                    # per-feature stylesheets (tokens.css owns variables)
│   ├── tokens.css          # CSS variables (--ah-bg, --orange, ...)
│   ├── alpha-hub.css
│   ├── arb-monitor.css
│   ├── terminal.css
│   └── ... (≈30 files)
├── js/                     # per-feature behaviour modules
│   ├── alpha-hub.js
│   ├── arb-dashboard.js
│   ├── crypto-5min.js
│   ├── reverse-finder-stream.js
│   ├── sentiment-leaderboard.js
│   └── ... (≈40 files)
└── data/                   # cached JSON dropped by backend (alphas, signals)
    ├── alpha_strategies.json
    └── live_signals.json
```

### Three modes share one URL

`web/index.html` mounts three top-level tabs:

- **Regression** — the original `/fit` workbench. WOW hero auto-runs the
  reverse-finder SSE stream on page load.
- **Strategies (α Hub)** — Top Alphas / Calendar & Spreads / Cross-venue Arb /
  Crypto Micro. Two more sub-tabs (Live Edge, Research) are designed but
  not yet wired in `index.html`.
- **Terminal** — Bloomberg-style data hub with 58 endpoints; default-active
  landing tab.

### Hot-file discipline

`web/index.html`, `web/config.js`, and `api/src/pfm/main.py` are flagged as
**hot files**. Per `PROTOCOL-V2.md`, contributors **do not edit index.html
directly**; instead they ship a new `web/css/<name>.css` and/or
`web/js/<name>.js`, and the `index-html-owner` Claude session mounts them
with `<link>` and `<script>` tags. This keeps the merge surface narrow and
race-free across the up-to-60 concurrent agents.

---

## 3. Backend

### Process model

`gunicorn -w 4 -k uvicorn.workers.UvicornWorker pfm.main:app` runs four
worker processes behind a shared TCP socket on `:8000`. Each worker is a
full Python interpreter with its own L1 cache (see §4); the L2 cache in
Redis is what makes the workers feel like a single service.

### FastAPI app shape

`pfm.main` is the entrypoint. It is intentionally small (1534 lines, down
from a peak of 4774) and partitioned by section per `PROTOCOL-V2.md`:

- `main.py:lifespan` — startup prewarm (factor universe, leaderboard,
  alpha-hub strategy bundle, crypto5min sampler), shutdown drain.
- `main.py:cors` — CORS middleware for `:8080` and `localhost`.
- `main.py:routes` — `include_router(...)` calls into the per-area routers
  (see below).
- `main.py:exception-handlers` — `HTTPException` formatter, Pydantic
  validation envelope, upstream-timeout mapping.

### Router layout (280+ endpoints)

Routers live in `api/src/pfm/*_router.py` and are mounted by `main.py`:

| Prefix              | Router module                         | Count | Notes                                        |
|---------------------|---------------------------------------|-------|----------------------------------------------|
| `/terminal/*`       | `terminal/__init__.py` + sub-routers  |   61  | data hub; mounts 14+ terminal_* modules      |
| `/strategies/*`     | `strategies_router.py`                |   36  | α Hub list + detail (`/alpha-hub/...`)       |
| `/strategies/arb/*` | `strategies_arb_router.py`            |   12  | cross-venue live monitor + SSE stream        |
| `/archive/*`        | `archive_router.py`                   |   11  | resolved-market archive search               |
| `/factors/*`        | `factors_router.py`                   |    7  | factor catalog list + by-source              |
| `/alpha-hub/*`      | `alpha_hub_router.py`                 |    7  | leaderboard + per-strategy detail            |
| `/reverse-finder/*` | `reverse_finder_router.py`            |    4  | "Why is NVDA moving today?" SSE              |
| `/health`           | `main.py` directly                    |    2  | `/health` + `/health/deep`                   |
| `/metrics/*`        | `metrics_router.py`                   |    3  | audit, latency, cache hit-ratio              |
| `/ops/*`            | `ops_router.py`                       |    2  | `/ops/sessions` (active edits), `/ops/jobs`  |
| ...                 | regression, news, lab, signals, ...   |   ~140|                                              |

The **270+ paths** number from `CLAUDE.md` is verified by
`curl /openapi.json | jq '.paths | keys | length'` and should be kept in
sync after each wave. Current snapshot: **271 paths**.

### Async + sync mixing

FastAPI lets us mix sync and async route handlers. We use:

- **`async def`** for IO-bound endpoints that fan out to multiple upstream
  HTTP calls (e.g. `/strategies/arb/stream`, `/reverse-finder/stream`,
  `/terminal/peer-scan`).
- **`def`** for CPU-bound or single-call endpoints (e.g. `/fit`,
  `/quant/deflated-sharpe`, `/strategies/calendar/score`). `statsmodels`
  is sync; wrapping a single OLS call in `async` buys nothing.

Background work (samplers, prewarm) uses FastAPI's `BackgroundTasks` for
fire-and-forget and `asyncio.create_task` from inside `lifespan` for
long-lived workers.

### Lifespan prewarm

`pfm.main:lifespan` runs at worker startup:

1. Connects to Redis (or installs `NullCache` if unreachable).
2. Reads `factors.yml`, validates each slug, and prewarms the top **200
   curated factors** (by `tier` and `liquidity_score`) into L1 + L2. This
   is what lets the WOW hero finish a regression in ~3 s warm.
3. Loads the `alpha_strategies.json` bundle into L1.
4. Optionally starts the crypto5min background sampler if
   `PFM_CRYPTO_5MIN_ENABLED=1`.
5. Optionally starts the arb engine subprocess if
   `PFM_ARB_ENGINE_AUTOSTART=1`.
6. Records a startup fingerprint in Redis (`pfm:lifespan:fingerprint`) so
   `/health/deep` can confirm "this worker has prewarmed."

---

## 4. Caching tiers

See [`docs/CACHE.md`](CACHE.md) and
[`docs/adrs/ADR-0008-cache-tiering.md`](adrs/ADR-0008-cache-tiering.md)
for the full discussion. The summary:

```
        ┌──────────────────────────────────────────────────────────┐
        │  Request enters                                          │
        └────────────────────────────┬─────────────────────────────┘
                                     │
                       ┌─────────────▼─────────────┐
                       │  L1 — in-process CachePool│
                       │  LRU, ~10k entries/worker │
                       │  hit:  ~50 µs             │
                       └─────────┬─────────────────┘
                            miss │
                                 ▼
                       ┌───────────────────────────┐
                       │  L2 — Redis pickle envelope│
                       │  TTL: tiered by route      │
                       │  hit:  ~1–3 ms            │
                       │  pickle envelope: see     │
                       │  ADR-0016 (versioning)    │
                       └─────────┬─────────────────┘
                            miss │
                                 ▼
                       ┌───────────────────────────┐
                       │  SingleFlight gate        │
                       │  (ADR-0014): one fetch    │
                       │  per (key, ttl) across    │
                       │  workers, others wait     │
                       └─────────┬─────────────────┘
                                 ▼
                       ┌───────────────────────────┐
                       │  Upstream API call        │
                       │  (Polymarket, yfinance...)│
                       └───────────────────────────┘
```

- **L1** is `pfm.cache.CachePool`, an LRU keyed by SHA-256 of
  `(source, slug, start, end, args)`. Per-worker, so 4 × L1s coexist.
- **L2** is Redis, pickle-serialised with a 1-byte version prefix
  (ADR-0016). TTL is tiered: `factors` 1 h, `yfinance` 6 h, `news` 5 min,
  `arb` 30 s.
- **SingleFlight** (ADR-0014) wraps any upstream call that is expensive
  enough to be worth deduplicating across concurrent workers. The lock key
  is `pfm:sf:<sha>` with `SET NX PX 30000`.
- **Lifespan prewarm** (top of §3) is the cold-start mitigation.

---

## 5. Coordination

Up to **60 concurrent Claude Code sub-agents** plus 5 human sessions
operate on this repo. Coordination is enforced by file-based protocol, not
by tooling. The contract is in
[`.coordination/PROTOCOL-V2.md`](../.coordination/PROTOCOL-V2.md) and the
ADR is
[`docs/adrs/ADR-0007-multi-session-coordination.md`](adrs/ADR-0007-multi-session-coordination.md).

The relevant data structures are:

- **`.coordination/active-edits.json`** — append-only array of claims. Each
  claim names files, a kebab-tagged scope, an ISO8601 `expires_at` (default
  +30 min), the `task_id` from `TASK-BOARD.md`, and the wave.
- **`.coordination/TASK-BOARD.md`** — flat list of available tasks. Each
  is a single-owner unit.
- **`.coordination/issues.log`** and **`outcomes.log`** — append-only
  audit trail.

The single rule that prevents disaster: **never `Write` your single claim
into `active-edits.json`**. Always `Read` the full array, push, write the
merged array back. The protocol calls this out because we lost work to it
once.

`/ops/sessions` (mounted by `ops_router.py`) returns the current parsed
active-edits view for any humans-in-the-loop who want to see who's editing
what without `cat`-ing the file.

---

## 6. Data sources

```
                    ┌──────────────────────────────────────────┐
                    │  pfm.sources.*  (one module per upstream)│
                    └────────────────────┬─────────────────────┘
                                         │
   ┌────────────────┬─────────────────┬──┴──────────────┬─────────────────┬────────────┐
   │                │                 │                 │                 │            │
   ▼                ▼                 ▼                 ▼                 ▼            ▼
┌────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌────────┐
│Polymkt │    │ Kalshi   │    │ yfinance │    │ FRED     │    │ GDELT/RSS │    │CoinGecko│
│Gamma+  │    │ /api/v2  │    │ batch    │    │ /series  │    │ Reddit/HN │    │ /coins  │
│CLOB    │    │          │    │ download │    │          │    │           │    │         │
└────────┘    └──────────┘    └──────────┘    └──────────┘    └───────────┘    └────────┘
```

### Polymarket

The single most important upstream. Two distinct APIs:

- **Gamma** (`gamma-api.polymarket.com`) — market metadata, slugs, event
  tree, `clobTokenIds`. Note: `clobTokenIds` arrives as a **JSON string
  inside the JSON response** (`"clobTokenIds": "[\"123...\"]"`) — must
  `json.loads()` twice effectively. Documented in `PLAN.md` and
  `CLAUDE.md`.
- **CLOB** (`clob.polymarket.com`) — `/prices-history`, `/book`, `/trades`.
  Use `fidelity=1440` (daily) for `/prices-history` — sub-daily fails for
  resolved markets (ADR-0007: daily fidelity).

Both are accessed via a single shared **HTTP/2 pool** (`pfm.http.HTTP2Pool`,
backed by `httpx.AsyncClient(http2=True)`). The pool is created in
`lifespan`, has 50 max connections, a 10-s connect timeout, and a 30-s
read timeout. HTTP/2 lets us multiplex the typical "fan out 40 slugs"
fetch pattern over a single TCP connection.

Rate limit is generous (1000/10s), but the L1+L2 cache makes us issue
~5% of the requests we would otherwise need.

### Kalshi

Used by `pfm.strategies_arb_router` only. Symmetric counterpart to
Polymarket for cross-venue arb. Auth header lives in
`PFM_KALSHI_API_KEY`.

### yfinance

Equity closes for the regression `y`. We batch tickers in groups of 50 via
`yfinance.download(..., group_by='ticker')` and inner-join to the
Polymarket calendar at UTC-midnight (ADR-0006).

### FRED

Macro series used as control factors (TB3MS, DGS10, VIXCLS, …). One
request per series; aggressively cached (24 h TTL).

### News / sentiment

GDELT + Reddit (`/r/wallstreetbets`, `/r/stocks`, …) + Hacker News (top
stories) + a curated RSS list. All feed `pfm.terminal_news` and the
sentiment scorer (§10).

---

## 7. Quant pipeline

```
factor_ids ──▶ resolver ──▶ source fetchers ──▶ logit/Δlogit transform
                  │                                   │
                  ▼                                   ▼
              factors.yml                        UTC-midnight join
                                                      │
                                                      ▼
                                              y = log returns
                                                      │
                                                      ▼
                                          ┌───────────────────────────────┐
                                          │  /fit dispatcher              │
                                          │  default: OLS + HAC (Newey-W) │
                                          │  opt-in: elnet / quantile /   │
                                          │  bayes / pls / ridge          │
                                          └─────────────┬─────────────────┘
                                                        ▼
                                          ┌───────────────────────────────┐
                                          │  Diagnostics                  │
                                          │   - VIF per factor            │
                                          │   - residual autocorr (LB)    │
                                          │   - HAC SE table              │
                                          │   - R², adj-R², AIC, BIC      │
                                          └─────────────┬─────────────────┘
                                                        ▼
                                          ┌───────────────────────────────┐
                                          │  Deflated Sharpe gate         │
                                          │  (ADR-0013 anti-alpha rule)   │
                                          └───────────────────────────────┘
```

- **Default estimator:** `statsmodels.OLS(...).fit(cov_type='HAC',
  cov_kwds={'maxlags': lag})`. We do not roll our own.
- **Lag selection:** HAC rule of thumb,
  `lag = floor(4 * (T/100)**(2/9))`, overridable via `?hac_lag=N`.
- **Logit transform:** `logit(p) = log(p / (1-p))`, clipped at `ε` (default
  `0.01`, overridable via `?epsilon`). Returns are `Δlogit(p_t) =
  logit(p_t) − logit(p_{t-1})`. Log returns, not simple returns, for `y`.
- **Opt-in estimators:** elastic net (`?estimator=elnet`), quantile
  regression (`?estimator=quantile`), Bayesian linear (`?estimator=bayes`),
  PLS (`?estimator=pls`), ridge (`?estimator=ridge`). All return the same
  `FitResponse` envelope plus an `estimator_meta` block.
- **Diagnostics** are computed unconditionally and included in every
  `FitResponse`. VIF > 10 flags collinearity. Ljung-Box p < 0.05 flags
  residual autocorrelation (suggests increasing `hac_lag`).
- **Deflated Sharpe** is the gate for any strategy that wants to be marked
  `deployable`: see §8 and ADR-0013.

---

## 8. Strategies

Strategies live under `pfm/strategies/` and follow a **registry pattern**:

```python
# pfm/strategies/registry.py
@register("calendar_lambda_ratio")
class CalendarLambdaRatio(Strategy):
    def signal(self, t): ...
    def position(self, t): ...
    def pnl(self, t): ...
```

The four currently-shipped α Hub sections (Top Alphas, Calendar & Spreads,
Cross-venue Arb, Crypto Micro) are all built on this protocol.

### The 4-quarter stress test

Every candidate strategy runs through `scripts/robustness_check.py` against
**4 disjoint quarters**. If any quarter has Sharpe < 0.5 OR a sign flip
relative to the full sample, the strategy is **NOT** marked deployable.
The gate is enforced by ADR-0013.

### The anti-alpha rule

If a strategy fails the 4-quarter test, it goes on the **anti-alpha list**
in `CLAUDE.md`. Future Claude must NOT re-pitch these as wins. The
canonical list:

- Recession-odds → defensive-sector long (regime-only)
- Crypto-ETF approval drift (one-time event)
- Senate-control short-vol (single 2024 episode)
- Geopolitical-conflict oil long (TC > gross PnL)

Surviving strategies (currently 4) are listed in CLAUDE.md "Validated
alphas" and each has a caveat about capacity, regime, or liquidity.

---

## 9. Arb matching

The cross-venue arb scanner (`pfm.arb_scanner` + `pfm.strategies_arb_router`)
matches Polymarket markets against Kalshi markets to find equivalent
events trading at different odds. The matching is **not** a simple slug
compare: ADR-0012 documents the design.

### T76 + T77 + T78 multi-feature scorer

The pair-match score is a weighted blend of three signal families:

- **T76 — Semantic similarity** (sentence-transformers cosine on title +
  description + resolution criteria).
- **T77 — Entity/keyword overlap** (Jaccard on tagged named entities and
  curated keywords like "Fed", "rate cut", "election", …).
- **T78 — Temporal alignment** (overlap of `[resolution_open,
  resolution_close]` windows, penalised by mismatch in tick size and
  expected-settlement date).

The combined score is `s = 0.5·T76 + 0.3·T77 + 0.2·T78`. Pairs with
`s ≥ 0.72` are candidates.

### 4-tier rejection taxonomy

Each rejected pair is classified into one of four tiers so we can audit
why we're missing pairs:

1. **Tier 1 — Hard mismatch** (different resolution sources; e.g. one
   resolves to BLS, the other to a CNN call).
2. **Tier 2 — Soft mismatch** (different bucketing; e.g. ">3.0%" vs
   ">=3%"; flagged for human review).
3. **Tier 3 — Insufficient data** (one side has < 20 trades or wider
   than 200 bps spread).
4. **Tier 4 — Stale** (one side has not traded in > 4 h).

Rejections are written to `arbstuff/rejections.jsonl` for the offline
audit suite.

---

## 10. Sentiment

Sentiment scoring is **one module**: `pfm/terminal/sentiment_nlp.py`. Do
not roll your own scorer elsewhere — import `score_text()`.

The scorer is a **hybrid**:

- **VADER** for general polarity. Strong on social-media tone, weak on
  finance jargon.
- **Financial-lex overlay** — a curated dictionary of finance-specific
  terms with explicit weights. "Earnings beat" / "Fed hawkish" / "credit
  spread widening" / "guidance cut" etc. The overlay is what stops VADER
  from over-scoring generic positive words like "growth" or "rally".

The blend is `score = 0.6·vader + 0.4·finlex`, clipped to `[-1, 1]`. The
function is wrapped in `@functools.lru_cache(maxsize=10000)` because the
same headlines re-appear across jumps, leaderboards, and the
`sentiment:<query>` factor source.

`/terminal/sentiment-leaderboard` returns the top-N most positive and
most negative tickers over a rolling window, computed entirely from the
LRU.

---

## 11. Real-time streams

There are two flavours of "real-time" in the system: **SSE** (Server-Sent
Events to the browser) and **background samplers** (in-process loops that
keep a buffer fresh for synchronous endpoints).

### SSE

- **`GET /strategies/arb/stream`** — emits a JSON event every **2 s**
  with the current top arbs. The arb scanner publishes into an
  `asyncio.Queue` and the SSE handler drains it. Heartbeat every 15 s
  prevents proxy idle-timeouts.
- **`POST /reverse-finder/stream`** — single-shot SSE that streams the
  top factor contributors for a ticker as each one is computed (so the
  user sees the first card in ~500 ms instead of waiting for all 200).
- **`GET /alerts/stream`** — broadcasts user-configured price/odds
  alerts. Optional; off by default.

### Background samplers

- **`pfm.crypto5min.sampler`** — opt-in via `PFM_CRYPTO_5MIN_ENABLED=1`.
  Polls BTC/ETH spot every 2 s into a rolling buffer so the model can
  compute `log(spot_t / spot_0)` at the start of any 5-min window without
  waiting for a websocket connection. Lifecycle is managed in `lifespan`.
- **`pfm.arb_engine`** — opt-in via `PFM_ARB_ENGINE_AUTOSTART=1`. A
  subprocess that maintains `arbstuff/dashboard_state.json` for the
  Cross-venue Arb tab. If absent or stale (> 3 min), the router falls
  back to `pfm.arb_scanner.top_arbs()` synchronously.
- **`pfm.decay_monitor`** — tracks live-strategy decay and emits a
  Slack-style warning to `/ops/jobs` when realised Sharpe drifts > 1.5 σ
  from the backtest.

---

## 12. Observability

```
┌──────────────────────────────────────────────────────────────┐
│  /health           liveness — process up + Redis pingable    │
│  /health/deep      readiness — lifespan prewarm fingerprint  │
│                    present, Polymarket reachable, Kalshi OK  │
│  /metrics/audit    counter snapshot — per-route hits, misses,│
│                    p50/p95/p99 latency, error class counts   │
│  /metrics/cache    L1/L2 hit ratio per (route, source)       │
│  /ops/sessions     parsed view of active-edits.json          │
│  /ops/jobs         running background tasks + decay alerts   │
└──────────────────────────────────────────────────────────────┘
```

- **`/health`** is dumb: returns `{"status": "ok"}` if the process is
  alive and Redis (if configured) is pingable. Suitable for k8s/docker
  livenessProbe.
- **`/health/deep`** is for readiness: returns 503 if the worker has not
  finished `lifespan` prewarm or if a critical upstream is unreachable.
- **`/metrics/audit`** is read by the demo's status bar and by an offline
  CI step that catches latency regressions.
- **`/ops/sessions`** is the introspection endpoint that lets a human see
  the current `active-edits.json` as parsed JSON.

We deliberately do **not** ship Prometheus exposition — the project
remains a POC, and JSON snapshots are sufficient for the demo. This is
listed in `docs/future-work.md` as a production gap.

---

## 13. Operations

### Local stack

`docker-compose up` brings up three containers:

| Service | Image                 | Port  | Purpose                              |
|---------|-----------------------|-------|--------------------------------------|
| `api`   | built from `api/`     | 8000  | FastAPI, 4 × UvicornWorker gunicorn  |
| `web`   | `nginx:1.27-alpine`   | 8080  | static `index.html` + `css/` + `js/` |
| `redis` | `redis:7-alpine`      | 6379  | L2 cache (internal only)             |

The `api` container's entrypoint is

```
gunicorn pfm.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000 \
    --timeout 60 \
    --graceful-timeout 30
```

Four workers is the demo default; benchmarks (`scripts/bench_workers.py`)
show diminishing returns past 4 on a M-series MacBook and past 8 on a
modest cloud VM.

### Environment variables

- `PFM_REDIS_URL` — defaults to `redis://redis:6379/0`.
- `PFM_CACHE_TTL_FACTORS` / `..._YFINANCE` / `..._NEWS` — per-tier TTL
  overrides.
- `PFM_KALSHI_API_KEY` — Kalshi auth (optional; if absent the arb tab
  shows a banner and falls back to Polymarket-only signals).
- `PFM_CRYPTO_5MIN_ENABLED=1` — start the crypto sampler in `lifespan`.
- `PFM_ARB_ENGINE_AUTOSTART=1` — start the arb engine subprocess.
- `PFM_LOG_LEVEL` — `INFO` by default.

### Restart discipline

The shared gunicorn at `:8000` serves every active browser tab AND every
parallel Claude Code session. Restarting it mid-wave will break their
work. Per `PROTOCOL-V2.md`:

1. Append a row to `.coordination/restart-requests.txt` with reason +
   requested-time.
2. Wait for the `gunicorn-owner` session to ack.
3. Only the owner restarts.

In practice, most "I changed Python code, do I need to restart?" answers
are **no**: `uvicorn --reload` is **off** in this stack precisely because
the workers hold expensive state (prewarmed factors, HTTP/2 pool).
Instead, ship code that doesn't need a restart (new routes can be added
via include but require restart; data-only changes don't).

---

## Appendix A — Why these choices, in one paragraph each

- **FastAPI** (ADR-0001) — chosen for OpenAPI generation + Pydantic
  request validation + first-class async support. The course rubric
  rewards a generated `openapi.json` and FastAPI gives that for free.
- **Logit transform** (ADR-0002) — prediction-market probabilities are
  bounded `[0, 1]` and their differences are heteroscedastic; the logit
  transform fixes both, and `Δlogit` is approximately Gaussian for inner
  probabilities.
- **HAC** (ADR-0003) — daily probability changes have
  short-horizon autocorrelation and conditional heteroscedasticity; HAC
  is the cheapest robust SE that handles both.
- **Redis TTL 1 h** (ADR-0004) — long enough to make demo iteration
  cheap, short enough that re-running after a Polymarket fix gets fresh
  data within a coffee break.
- **No persistence** (ADR-0005) — the POC has no state worth keeping;
  adding Postgres would multiply deploy surface for zero demo benefit.
- **UTC alignment** (ADR-0006) — Polymarket timestamps (unix seconds) and
  yfinance closes both normalise to `pandas.Timestamp(date).normalize()`
  at UTC midnight. Anything else creates 1-day shift bugs that look like
  modelling errors.
- **Daily fidelity** (ADR-0007) — sub-daily `/prices-history` returns
  empty for resolved markets; `fidelity=1440` works for both active and
  resolved.
- **Vanilla HTML** (ADR-0009) — no build step, no bundle, no React.
- **Multi-session protocol** (ADR-0010) — race conditions on
  hot files were destroying work; file-based claims plus a single-owner
  rule fixed it.
- **Cache tiering** (ADR-0011) — L1 keeps per-worker hits at ~50 µs, L2
  keeps cross-worker hits at ~2 ms; lifespan prewarm makes cold starts
  feel warm.
- **SingleFlight** (ADR-0014) — without it, a thundering herd at cold
  start would 40×-amplify the upstream Polymarket call rate.
- **Anti-alpha rule** (ADR-0013) — the gate that prevents regime-driven
  flukes from being shipped as alpha.

---

## Appendix B — Pointers

- Plan: [`PLAN.md`](../PLAN.md)
- Project instructions: [`CLAUDE.md`](../CLAUDE.md)
- Cache details: [`docs/CACHE.md`](CACHE.md)
- Coordination protocol: [`.coordination/PROTOCOL-V2.md`](../.coordination/PROTOCOL-V2.md)
- ADR index: [`docs/adrs/`](adrs/)
- API reference: [`docs/API_REFERENCE.md`](API_REFERENCE.md)
- User guide: [`docs/USER_GUIDE.md`](USER_GUIDE.md)
- Runbook: [`docs/RUNBOOK.md`](RUNBOOK.md)
- Latest alpha report: [`docs/alpha-report-v19.md`](alpha-report-v19.md)

---

*Last updated: 2026-05-16 (Wave 13, task W13-46). When the system shape
changes materially — a new mode, a new cache tier, a new stream — update
this document and bump the date.*
