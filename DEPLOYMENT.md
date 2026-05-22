# Deployment guide

<!-- NOTE: see docs/operations/DEPLOYMENT.md (canonical). This root-level copy is kept for
discoverability from the repo home; the in-docs version is the source of truth
when the two drift. -->

This document explains how to put `prediction-factor-model` on the public
internet. The codebase ships with one-shot configs for **Fly.io**,
**Render**, **Heroku/Railway** (`Procfile`), and a **self-hosted
docker-compose** path. Pick whichever matches your hosting budget; the
backend image is identical in every case.

## Architecture summary

- **API**: FastAPI behind **gunicorn** (`uvicorn.workers.UvicornWorker`)
  with 4 workers, port 8000. The Dockerfile CMD enforces this; `uvicorn`
  alone is dev-only. Each worker has its own event loop + Python
  interpreter; they share state via Redis.
- **Compression**: `BrotliMiddleware` (quality 5) layered above
  `GZipMiddleware`. Browser advertising `br` gets ~250 KB on the wire for
  the 1.2 MB index.html; gzip clients fall back to ~270 KB.
- **Cross-worker caches** (all via Redis, set automatically when
  `REDIS_URL` is reachable):
  - `arb:gamma_prices` — Polymarket midpoints, TTL 300 s. SETNX lock so
    only one worker fans out to Gamma per refresh.
  - `arb:dashboard_state` — mirrored from the arb_engine sidecar's
    `dashboard_state.json` every 5 s. Survives container restarts.
  - `term:*` — L2 for the in-process `TERMINAL_CACHE` (overview,
    search, peers). Each worker has L1 in-memory + reads/writes L2 on miss.
  - factor-history pickle cache (the regression core's main hot path).
- **Web**: nginx-alpine static serving `web/index.html` (single-file
  vanilla JS + Plotly CDN). In a single-machine Fly setup the FastAPI
  process itself serves `/ui/*` via StaticFiles, so nginx is optional.
- **Redis**: 7-alpine, AOF persistence (`pfm-redis-data` volume in the
  self-hosted compose; managed Redis on PaaS).

All API URLs are environment-driven via `pfm.config.Settings`. In
production set `ENV=production` and the service warns at startup if
`CORS_ORIGINS=*` or if Redis still points at the docker-compose service
name `redis`.

### Performance env vars (Fly secrets are not needed for these)

| Var | Default | Effect |
| --- | --- | --- |
| `GUNICORN_WORKERS` | `4` | Number of UvicornWorker procs. Bump on bigger VMs. |
| `PFM_FACTOR_PREWARM_ENABLED` | `1` (in `fly.toml`) | Prewarms 200 curated factor histories at boot — `/reverse-finder` warm in 3 s instead of 30 s. |
| `PFM_FACTOR_PREWARM_TOP_N` | `200` | How many factors to prewarm. |
| `PFM_GAMMA_PRICE_TTL_S` | `300` | Redis TTL for the shared Polymarket price map. |
| `PFM_GAMMA_PRICE_REFRESH_S` | `60` | How often to refresh; the SETNX lock prevents thundering herd. |
| `PFM_ARB_FALLBACK_MIN_SPREAD` | `0.5` | Minimum spread % the in-process scanner reports. Lower → more (noisy) opps. |
| `PFM_ARB_STREAM_TICK_S` | `5.0` | SSE push interval to the ARB tab. |
| `PFM_ARB_STREAM_SCAN_LOG_MAX` | `30` | Trim scan_log on the wire (the full log stays on `/state`). |
| `PFM_ARB_MIRROR_INTERVAL_S` | `5` | How often to mirror `dashboard_state.json` to Redis. |
| `PFM_ARB_MIRROR_TTL_S` | `600` | Redis TTL of the mirrored state. |
| `PFM_CRYPTO_WS_ENABLED` | unset | Set `1` to enable Binance WebSocket signal capture. Adds ~100 MB / ~5% CPU; recommended off unless the demo uses Crypto Micro events. |

---

## Quick deploy: Fly.io (recommended)

1. **Install flyctl**
   ```sh
   curl -L https://fly.io/install.sh | sh
   ```
2. **Login**
   ```sh
   flyctl auth login
   ```
3. **Create the app** (keeps the existing `fly.toml`)
   ```sh
   flyctl launch --no-deploy --copy-config --name pfm-prod
   ```
4. **Set secrets** (CORS in particular must be set BEFORE first deploy)
   ```sh
   flyctl secrets set ENV=production
   flyctl secrets set CORS_ORIGINS="https://your-domain.com"
   flyctl secrets set SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
   flyctl secrets set SENTRY_DSN="https://...@sentry.io/..."   # optional
   ```
5. **Provision Redis**
   ```sh
   flyctl redis create --name pfm-redis --region iad
   # copy the connection string it prints, then:
   flyctl secrets set REDIS_URL="redis://default:...@fly-pfm-redis.upstash.io"
   ```
6. **Deploy**
   ```sh
   flyctl deploy
   ```
7. **Open**
   ```sh
   flyctl open
   ```

### CI auto-deploy

`.github/workflows/deploy.yml` runs `flyctl deploy --remote-only` on every
push to `main` and on `v*` tags. Add a repo secret named
**`FLY_API_TOKEN`** (generate with `flyctl auth token`) and the workflow
will take it from there. Concurrency group `deploy-group` prevents two
deploys racing.

---

## Render alternative

1. Push the repo (with `render.yaml` at root).
2. In Render: **New + → Blueprint → connect GitHub repo**. Render reads
   `render.yaml` and provisions:
   - `pfm-api` (Docker web service, healthcheck on `/health`)
   - `pfm-redis` (managed Redis, `allkeys-lru`)
3. Set additional env vars in the dashboard:
   - `CORS_ORIGINS` (your final domain)
   - `SLACK_WEBHOOK_URL` (optional)
   - `SENTRY_DSN` (optional)
4. First deploy auto-runs; subsequent pushes redeploy on commit.

The web frontend is NOT part of `render.yaml` — host it as a Render Static
Site pointing at `web/` (build command empty, publish dir `web`) or behind
a CDN. Update `web/config.js` so `PFM_API_BASE` points at your Render API
URL (or set up a reverse-proxy path of `/api`).

---

## Heroku / Railway / Fly Procfile mode

A `Procfile` at the repo root lets the API run on any PaaS that honors
buildpacks:

```
web: gunicorn pfm.main:app -k uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:$PORT --max-requests 5000 --max-requests-jitter 500 --timeout 90 --graceful-timeout 30 --keep-alive 30 --worker-tmp-dir /dev/shm --forwarded-allow-ips '*'
```

You will still need to provision Redis separately and set `REDIS_URL`,
`ENV=production`, and `CORS_ORIGINS`.

---

## DigitalOcean App Platform

DO can deploy directly from the Dockerfile:

1. Create a new App, point at the repo.
2. Component type: **Dockerfile**, source: `api/Dockerfile`.
3. HTTP port: `8000`. Health check path: `/health`.
4. Add a **Redis** component (managed) and bind its `REDIS_URL` to the API.
5. Set `ENV=production`, `CORS_ORIGINS=https://<your-app>.ondigitalocean.app`.

---

## Self-hosted (docker-compose)

The original dev compose still works for a single-VM deployment behind a
reverse proxy (Caddy / Traefik / nginx).

```sh
# Dev (default):
docker compose up -d

# Prod overrides (resource caps, ENV=production, CORS from env):
export CORS_ORIGINS="https://your-domain.com"
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Behind a reverse proxy, terminate TLS at the proxy and forward to
`api:8000` and `web:80`. Uvicorn already runs with `--proxy-headers
--forwarded-allow-ips '*'` so `X-Forwarded-*` is honored.

---

## Frontend config

`web/config.js` is loaded by `index.html` *before* any other script and
sets `window.PFM_API_BASE`:

- `localhost` / `127.0.0.1` → `http://localhost:8000`
- anything else → `/api` (assumes the deployment reverse-proxies `/api`
  to the FastAPI service)

> **Manual step required**: `web/index.html` must include
> `<script src="/config.js"></script>` as its first `<script>` tag. This
> guide does not modify `index.html`; do it once by hand and you're done.

If you deploy the API and the static site on different domains, edit
`web/config.js` to hardcode the API URL (e.g.
`window.PFM_API_BASE = 'https://pfm-prod.fly.dev';`) and rebuild the web
image.

---

## Activar Live Signals

The `pfm.live_signals_job` background task recomputes
`web/data/live_signals.json` every N minutes from the curated alpha
catalog. It is **off by default** so the test suite (which spins up
the app via `TestClient` repeatedly) never accumulates background
tasks. Two backends are available:

- `synthetic` (default, no network) — deterministic random walks per
  factor id. Good for CI, demos with no internet, and the existing
  test suite.
- `polymarket` (real data) — hits Gamma `/markets?slug=...` to resolve
  each leg's `clobTokenIds[0]`, then pulls daily history through
  `clob.polymarket.com/prices-history?fidelity=1440`. Both legs are
  inner-joined on UTC dates before the spread is computed. Fetches
  are cached in-process for 600 s to absorb re-runs that share legs.

Toggle it on by setting these environment variables before app start:

```sh
export PFM_LIVE_SIGNALS_ENABLED=1
export PFM_LIVE_SIGNALS_INTERVAL_S=900       # default 900 s (15 min)
export PFM_LIVE_SIGNALS_FETCHER=polymarket   # default 'synthetic'
# Optional override of the connectivity-probe slug:
# export PFM_CONNECTIVITY_SAMPLE_SLUG=will-bitcoin-hit-100k-by-end-of-2026
```

On Fly.io:

```sh
flyctl secrets set PFM_LIVE_SIGNALS_ENABLED=1
flyctl secrets set PFM_LIVE_SIGNALS_INTERVAL_S=900
flyctl secrets set PFM_LIVE_SIGNALS_FETCHER=polymarket
```

Pre-flight check before flipping `PFM_LIVE_SIGNALS_FETCHER=polymarket`
in production — `GET /signals/connectivity-check` (admin-gated when
`PFM_ADMIN_TOKEN` is set) confirms Gamma + CLOB are reachable and
returns the sample size + latency:

```sh
curl -s https://your-app.fly.dev/signals/connectivity-check \
  -H "X-Admin-Token: $PFM_ADMIN_TOKEN" | jq
# {"ok": true, "sample_size": 180, "error": null, "latency_ms": 312.4, ...}
```

Once enabled, watch the job through:

- `GET /signals/status` — last-run timestamp, duration, failure list
- `GET /signals/live` — the freshly computed signals (cached 30 s)
- Prometheus: `pfm_live_signals_last_run_age_seconds`,
  `pfm_live_signals_recompute_duration_seconds`

## Production checklist

- [ ] `ENV=production` set (auto-flips auth ON — see "Auth-by-default" below)
- [ ] `CORS_ORIGINS` set to your domain (no `*`)
- [ ] `REDIS_URL` points at managed Redis (not `redis://redis:6379/0`)
- [ ] Redis has persistence (AOF on self-host; managed providers default-on)
- [ ] HTTPS terminated (Fly.io does this automatically)
- [ ] `PFM_ADMIN_TOKEN` set to a long random string (recommended; see below)
- [ ] `SENTRY_DSN` configured (optional)
- [ ] `SLACK_WEBHOOK_URL` for alerts (optional; keep `PFM_ALERTS_DRY_RUN=1`
      until ready)
- [ ] `curl https://your-app.fly.dev/health` returns 200
- [ ] `curl https://your-app.fly.dev/health/detail | jq .auth_status` shows
      `enabled: true` and `admin_token_configured: true`
- [ ] OpenAPI docs accessible: `https://your-app.fly.dev/docs`
- [ ] Disclaimer footer visible in UI
- [ ] Rate limiting enabled (nginx + slowapi)
- [ ] Smoke test passes: `./scripts/smoke_test.sh https://your-app.fly.dev`

### Auth-by-default

The API auto-detects production-like environments and turns auth + rate
limiting **ON** without you needing to remember `PFM_AUTH_ENABLED=1`. Auth is
enabled when **any** of the following is true:

| Signal              | Typical setter                      |
|---------------------|-------------------------------------|
| `PFM_AUTH_ENABLED=1`| You, explicitly                     |
| `ENV=production`    | This guide / `flyctl secrets set`   |
| `FLY_APP_NAME` set  | Fly.io runtime, automatically       |
| `RENDER` set        | Render runtime, automatically       |
| `NODE_ENV=production`| Frontend stacks / Heroku       |

`PFM_AUTH_ENABLED=0` is a hard override that keeps auth OFF even in
production — use it only for an explicit reason (one-off debugging on a
private staging URL, etc.).

The active posture is reported (without leaking the token itself) at:

```sh
curl https://your-app.fly.dev/health/detail | jq .auth_status
# {
#   "enabled": true,
#   "autogen_token_in_use": false,
#   "admin_token_configured": true,
#   "env_detection": "fly"
# }
```

### Admin token: set it, or accept the autogen

When auth is ON the service refuses admin endpoints (`POST /auth/keys`,
`DELETE /auth/keys/...`, `GET /auth/usage/dashboard`, the alpha-tier-regen
admin routes, `/signals/connectivity-check`) unless an admin token is
configured. Resolution order:

1. **`PFM_ADMIN_TOKEN` set** (recommended) — used verbatim. Survives
   restarts. This is what you should ship.
   ```sh
   flyctl secrets set PFM_ADMIN_TOKEN="$(python -c 'import secrets;print(\"sk_admin_\"+secrets.token_urlsafe(32))')"
   ```
2. **Autogen on first boot** — if auth is on and `PFM_ADMIN_TOKEN` is unset,
   the service mints `sk_admin_<32 bytes urlsafe>` and persists it to
   `/tmp/pfm_admin_token.json` (chmod 0600). The token is regenerated on the
   **next** restart unless you set `PFM_ADMIN_TOKEN` (which is the warning
   the service logs on startup). Two ways to recover the value:
   - **Logs**: `flyctl logs | grep "Generated admin token"` (it's a single
     `WARNING` line at startup).
   - **One-shot endpoint**: `GET /auth/first-boot-info` returns the token
     once, marks `/tmp/pfm_first_boot_done.flag`, then 410s every subsequent
     call. The endpoint also returns 404 when auth is OFF, so it never
     leaks dev-mode posture from a prod URL by mistake.
   - **File**: `flyctl ssh console -C 'cat /tmp/pfm_admin_token.json'`.

### Disabling auth (not recommended in prod)

If you need to flip auth off temporarily — e.g. to run a load test against a
private hostname — set `PFM_AUTH_ENABLED=0`. This explicit override beats the
production auto-detect, so you don't have to clear `ENV` /
`FLY_APP_NAME` / etc.

```sh
flyctl secrets set PFM_AUTH_ENABLED=0
```

Re-enable by `flyctl secrets unset PFM_AUTH_ENABLED` (or set it back to `1`).

---

## HTTP performance tuning

The API ships with two complementary performance levers tuned for the
Polymarket fan-out workload (1090 factors x N parallel requests per page).
Defaults match the `prod` profile and are safe for most deploys.

### Async-http connection pool (Polymarket)

`pfm.main` sets up a shared `httpx.AsyncClient` on `app.state.async_http`
and reuses it across `/terminal/homepage`, `/terminal/quote`, the realtime
SSE hub, and the live-signals job. The tuned configuration is:

```python
limits = httpx.Limits(
    max_keepalive_connections=20,   # warm sockets per host
    max_connections=100,            # cap concurrent connections
    keepalive_expiry=30.0,          # seconds before idle socket closes
)
timeout = httpx.Timeout(
    connect=5.0,                    # TCP/TLS handshake budget
    read=30.0,                      # slow Polymarket bulk endpoints
    write=10.0,                     # POSTs are tiny but defensive
    pool=10.0,                      # wait time for a free connection
)
```

Why these numbers:

- **`max_keepalive_connections=20`**: enough warm sockets that 10
  parallel `/terminal/homepage` cold requests never serialise on a TLS
  handshake (measured: 1.3s → ~0.4s wall-clock with reuse).
- **`max_connections=100`**: under Polymarket's 1000-req/10s rate-limit
  envelope, leaves room for `/factors/rank` 1090-candidate fan-out
  without saturating.
- **`pool=10.0`**: callers wait at most 10s for a connection slot before
  raising. Set high enough that healthy traffic doesn't trip; low enough
  that a client that hangs upstream can't accumulate unbounded coroutines.
- **`keepalive_expiry=30.0`**: matches Polymarket's typical idle-close
  window so we drop sockets the server is about to close anyway.

If you scale to a larger frontend that fan-outs >100 in-flight requests
per worker, raise `max_connections` first. Don't raise
`max_keepalive_connections` past ~50 — Polymarket has been observed to
drop sockets idle longer than ~60s.

### OpenAPI compression and caching

`GET /openapi.json` is wrapped by `starlette.GZipMiddleware`
(`minimum_size=1024`) and a custom handler that emits a strong `ETag` plus
`Cache-Control: public, max-age=3600`. On the production app:

| Mode                                | Wire bytes |
|-------------------------------------|------------|
| Plain `GET /openapi.json`           | ~478 KiB   |
| With `Accept-Encoding: gzip`        | ~83 KiB    |
| `If-None-Match: <etag>` (304)       | 0          |

The ETag is `"<app.version>-<sha256[:16]>"`, so a code change (which
changes `app.version` or any path/schema) invalidates client caches
deterministically. Browsers add `Accept-Encoding: gzip,br` on every
request — no client-side change needed.

### Search-index lazy palette load

`GET /terminal/search-index` returns the full ~283 KiB factor catalogue
in one shot (kept for backward-compat). The new
`GET /terminal/search-index/chunked?chunk=N&size=200` slices the same
payload into 200-row pages (~47 KiB each, ~12 KiB gzipped) and emits
`X-Total-Chunks: N` so the frontend can prefetch idle.

---

## Monitoring

- **Prometheus metrics**: `GET /metrics`
- **Health detail**: `GET /health/detail`
- **Logs**:
  - Fly.io: `flyctl logs -a pfm-prod`
  - Render: dashboard → Logs tab
  - Self-hosted: `docker compose logs -f api`
- **Smoke test**: `./scripts/smoke_test.sh https://<host>`

---

## Observability

The API ships with a three-pillar observability stack: structured logs
(structlog), metrics (Prometheus), and error tracking (Sentry, optional).
Everything below is opt-in via environment variables; the API runs fine
with none of them set.

### Structured logging (structlog)

`pfm.logging_setup.configure_logging()` is called at import time of
`pfm.main`. Output format is controlled by:

| Var          | Default   | Values                                  |
|--------------|-----------|-----------------------------------------|
| `LOG_FORMAT` | `json`    | `json` (one JSON object per line) or `text` (coloured console) |
| `LOG_LEVEL`  | `INFO`    | `DEBUG` / `INFO` / `WARNING` / `ERROR`  |

In production keep the default `LOG_FORMAT=json` so Fly/Render/Loki/ELK
can index the fields. For local dev set `LOG_FORMAT=text` for readable
output. Each log line carries `timestamp`, `level`, `module`, `lineno`,
plus any structured kwargs the call site bound.

### Sentry error tracking

To enable Sentry, set:

```sh
flyctl secrets set SENTRY_DSN="https://<key>@<org>.ingest.sentry.io/<id>"
flyctl secrets set SENTRY_TRACES_SAMPLE_RATE=0.1     # optional, default 0.1
flyctl secrets set SENTRY_PROFILES_SAMPLE_RATE=0.1   # optional, default 0.1
flyctl secrets set ENV=production                    # tags events with env
flyctl secrets set GIT_SHA="$(git rev-parse HEAD)"   # tags release version
```

If `SENTRY_DSN` is unset (or empty), Sentry is not initialised and the app
behaves exactly as before. If `sentry-sdk` itself is missing from the
target image, the import is swallowed silently — no crash. The
`FastApiIntegration` is wired with `transaction_style="endpoint"` and
PII reporting is off (`send_default_pii=False`).

### Prometheus + Grafana (self-hosted)

The repo includes a turnkey monitoring stack under `monitoring/`:

- `monitoring/prometheus.yml` — scrapes `api:8000/metrics` every 15s
- `monitoring/grafana_dashboard.json` — 6-panel dashboard with templating
  for `$endpoint`, `$source`, `$tier`
- `monitoring/docker-compose.observability.yml` — Prometheus + Grafana
  containers, joined to the existing `pfm` network

Bring everything up:

```sh
docker compose -f docker-compose.yml -f monitoring/docker-compose.observability.yml up -d
```

Then open:

- Prometheus: <http://localhost:9090> (no auth)
- Grafana:    <http://localhost:3000>  (login `admin` / `admin`, change on
  first login)

The dashboard JSON is auto-mounted into Grafana's provisioning directory,
but Grafana provisions dashboards via a **dashboard-provider config**, not
just a JSON file. The first time Grafana boots, **manually import** the
dashboard:

1. In Grafana, **Dashboards → New → Import**
2. Upload `monitoring/grafana_dashboard.json` (or paste its contents)
3. Pick the **Prometheus** datasource (auto-discovered from
   `http://prometheus:9090`)
4. Click **Import**

The dashboard ships 6 panels:

1. Request rate per endpoint (`requests_total`)
2. p50 / p95 / p99 latency (`request_duration_seconds`)
3. 5xx error rate (overall)
4. Cache hit ratio by backend (Redis vs in-memory)
5. Upstream API health (Polymarket / Kalshi / Manifold / PredictIt)
6. Active SSE clients gauge (`pfm_realtime_clients`)

### Custom metrics catalogue

In addition to the built-in `requests_total` /
`request_duration_seconds`, the API exposes:

- `pfm_factor_history_fetches_total{source, status}`
- `pfm_factor_history_fetch_duration_seconds{source}`
- `pfm_alpha_lab_runs_total{status}`
- `pfm_alerts_fired_total{rule_kind, channel, status}`
- `pfm_realtime_clients`, `pfm_realtime_pollers` (gauges)
- `pfm_live_signals_last_run_age_seconds`,
  `pfm_live_signals_recompute_duration_seconds`
- `pfm_factor_model_fits_total{ticker, n_factors}`,
  `pfm_factor_model_fit_duration_seconds`

To instrument a new function, use the decorator:

```python
from pfm.observability import track_metric

@track_metric("pfm_my_feature", source="polymarket")
def my_function(...):
    ...
```

This auto-creates `pfm_my_feature_total` (Counter) and
`pfm_my_feature_duration_seconds` (Histogram) in the default registry.

---

## Required secrets (summary)

| Secret              | Required | Used by                                 |
|---------------------|----------|-----------------------------------------|
| `FLY_API_TOKEN`     | CI only  | GitHub Actions deploy workflow          |
| `ENV`               | yes      | `pfm.config.Settings.production` flag   |
| `CORS_ORIGINS`      | yes      | FastAPI CORS middleware                 |
| `REDIS_URL`         | yes      | Cache layer                             |
| `SENTRY_DSN`        | no       | Error reporting                         |
| `SENTRY_TRACES_SAMPLE_RATE`   | no | Sentry tracing sample rate (0.0–1.0, default 0.1) |
| `SENTRY_PROFILES_SAMPLE_RATE` | no | Sentry profiling sample rate (default 0.1) |
| `GIT_SHA`           | no       | Release tag for Sentry (12 chars used)  |
| `LOG_FORMAT`        | no       | `json` (default) or `text`              |
| `LOG_LEVEL`         | no       | `INFO` (default)                        |
| `SLACK_WEBHOOK_URL` | no       | Alerts router                           |
| `PFM_ALERTS_DRY_RUN`| no       | Default `1`; set `0` to actually post   |
| `PFM_LIVE_SIGNALS_ENABLED`  | no | Set to `1` to start the live-signals cron |
| `PFM_LIVE_SIGNALS_INTERVAL_S` | no | Recompute interval, seconds (default 900) |
| `PFM_LIVE_SIGNALS_FETCHER` | no | `synthetic` (default) or `polymarket`     |
| `PFM_CONNECTIVITY_SAMPLE_SLUG` | no | Slug used by `/signals/connectivity-check` |
| `PFM_ALPHA_STRATEGIES_PATH` | no | Override path to `alpha_strategies.json`   |
| `TIINGO_API_KEY`    | no       | Tiingo equity-prices fallback           |
| `POLYGON_API_KEY`   | no       | Live earnings consensus + calendar (whisper module) |

---

## Earnings whisper data source

The `pfm.earnings_whisper` module computes a "whisper EPS" from
Polymarket beat-ladder odds combined with sell-side consensus EPS. By
default it reads consensus from a hardcoded snapshot of 8 large-caps
(NVDA, TSLA, AAPL, AMZN, MSFT, META, GOOGL, AMD). Setting
`POLYGON_API_KEY` upgrades the module to a live feed.

- **Source**: Polygon.io `/vX/reference/financials` (free tier).
- **Caching**: 12 hours per ticker (`polygon_consensus` cache).
- **Rate limit**: free tier is **5 requests/minute**. `PolygonClient`
  serialises calls behind an `asyncio.Semaphore` and sleeps 13 s
  between successful requests, so a dashboard expansion never bursts
  past the limit. 429 responses are retried once before falling back.

### Configure the key

Local dev:
```sh
export POLYGON_API_KEY="poly_xxx"
.venv/bin/python -m uvicorn pfm.main:app --reload
```

Fly.io:
```sh
flyctl secrets set POLYGON_API_KEY="poly_xxx"
```

Render / DigitalOcean: add `POLYGON_API_KEY` in the dashboard env-vars
panel.

### Fallback behaviour

If `POLYGON_API_KEY` is unset, every Polygon call short-circuits and
the module emits a single `WARNING` log line on first use. Whisper
responses populate `consensus_source = "hardcoded_snapshot"` for
known tickers and `"unknown"` for everything else, so dashboards make
the data provenance explicit.

If `POLYGON_API_KEY` is set but Polygon returns 5xx or sustained
429s, the wrapper logs an `INFO` line and falls back to the same
hardcoded snapshot — the response stays a 200 and `consensus_source`
flips to `"hardcoded_snapshot"`.

### Endpoints affected

- `GET /alpha/earnings-whisper/{ticker}?source=cached|live|hardcoded`
- `GET /alpha/earnings-whisper-dashboard?days=14&source=cached`
  (cached 1 h; dashboard expands to up to 50 tickers when Polygon is
  configured, vs the 8 hardcoded baseline)
- `GET /alpha/earnings-calendar?days=30&source=cached`
