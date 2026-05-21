# Production Deployment Guide

This document describes how to deploy the `prediction-factor-model` (Prediction
Terminal) stack to a production environment. It assumes you have shell access
to a Linux host, a domain name with DNS pointing at it, and the standard
Unix toolchain available. The stack is small enough to run on a single
$10–$40/month VM and large enough that you will want monitoring, automated
backups, and a reproducible deploy pipeline. Everything below has been
verified against the repository contents on `main` as of wave-13.

---

## 1. Prerequisites

Before touching the host, confirm the following are installed and on `$PATH`:

| Component        | Minimum version | Why                                              |
| ---------------- | --------------- | ------------------------------------------------ |
| Python           | 3.12.x          | `pfm` uses 3.12-style type hints (`list[str]`).  |
| Docker Engine    | 24.x            | All services run in containers.                  |
| Docker Compose   | v2.20+          | `docker compose` (plugin) is required.           |
| Redis            | 7.x             | Cache + prewarm store. Run inside compose.       |
| gunicorn         | 21.x            | Sync ASGI/WSGI worker. Already in `requirements`. |
| nginx (optional) | 1.24+           | TLS termination + reverse proxy.                 |
| git              | 2.40+           | Pull releases.                                   |
| jq               | 1.7+            | Used by smoke tests and `scripts/monitor.sh`.    |
| curl             | any recent      | Healthchecks.                                    |

On a fresh Ubuntu 22.04 LTS host:

```bash
sudo apt update && sudo apt -y install \
  ca-certificates curl gnupg jq python3.12 python3.12-venv git
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker
```

A non-root user with `docker` group membership is strongly preferred — never
run the API stack as `root`.

### Host sizing

| Profile      | vCPU | RAM   | Disk  | Notes                              |
| ------------ | ---- | ----- | ----- | ---------------------------------- |
| Demo / POC   | 1    | 2 GB  | 20 GB | Cold latencies acceptable.         |
| Single-tenant prod | 2    | 4 GB  | 40 GB | Recommended starting point.   |
| Heavy backtest | 4  | 8 GB  | 80 GB | Needed if you enable the curated factor prewarm with `TOP_N > 200`. |

---

## 2. Build

The repository ships a multi-stage `api/Dockerfile`, a static-asset
`web/Dockerfile`, and two compose files: `docker-compose.yml` (base,
dev-friendly) and `docker-compose.prod.yml` (production overrides — resource
limits, log level, restart policy).

Clone the repo onto the host and build:

```bash
git clone https://github.com/<your-org>/proyectofuentes.git /opt/pfm
cd /opt/pfm
git checkout v$(cat VERSION 2>/dev/null || echo main)
docker compose -f docker-compose.yml -f docker-compose.prod.yml build
```

The base build takes ~3 minutes on a 2 vCPU host (pandas/numpy wheels are
the longest leg). Subsequent rebuilds use BuildKit caching and finish in
under 30 seconds when only Python source changes.

Verify the images:

```bash
docker images | grep pfm
# pfm-api    latest   <sha>   1 minute ago   780MB
# pfm-web    latest   <sha>   1 minute ago    25MB
```

If `pfm-api` exceeds 1.2 GB you probably installed dev dependencies; rerun
with `--no-cache` and ensure `requirements.txt` is the production list, not
`requirements-dev.txt`.

---

## 3. Configuration

All runtime configuration is driven by environment variables. Production
secrets live in `/opt/pfm/.env.prod` (mode `0600`, owner `pfm-deploy`). The
compose files reference them via `${VAR}` and `env_file:`.

### Required

| Variable        | Example                          | Purpose                                |
| --------------- | -------------------------------- | -------------------------------------- |
| `REDIS_URL`     | `redis://redis:6379/0`           | Cache + prewarm. Use container DNS.    |
| `CORS_ORIGINS`  | `https://terminal.example.com`   | Comma-separated, no spaces.            |
| `LOG_LEVEL`     | `INFO` (dev) / `WARNING` (prod)  | Set via prod override.                 |
| `ENV`           | `production`                     | Toggles strict pydantic + no debug UI. |

### `PFM_*` prewarm and feature flags

These are already declared in `docker-compose.yml`; tune them in `.env.prod`:

- `PFM_PMVIX_PREWARM_ENABLED=1`, `PFM_PMVIX_PREWARM_INTERVAL_S=300`
- `PFM_EARNINGS_PREWARM_ENABLED=1`, `PFM_EARNINGS_PREWARM_INTERVAL_S=3600`
- `PFM_LIVE_SIGNALS_ENABLED=1`, `PFM_LIVE_SIGNALS_INTERVAL_S=900`,
  `PFM_LIVE_SIGNALS_FETCHER=polymarket`
- `PFM_PM_VIX_AUTO_REFRESH_ENABLED=1`, `PFM_PM_VIX_AUTO_REFRESH_INTERVAL_S=21600`
- `PFM_DECAY_REFRESH_ENABLED=1`, `PFM_DECAY_REFRESH_INTERVAL_S=14400`
- `PFM_FACTOR_PREWARM_ENABLED=1`, `PFM_FACTOR_PREWARM_TOP_N=50`,
  `PFM_FACTOR_PREWARM_LOOKBACK_DAYS=180`,
  `PFM_FACTOR_PREWARM_CONCURRENCY=8`
- `PFM_ARB_ENGINE_AUTOSTART=0` — leave off in prod unless you run the
  cross-venue arb engine on the same host
- `PFM_CRYPTO_5MIN_ENABLED=0` — opt-in spot sampler; only enable if you
  intend to publish 5/15-minute crypto signals

### macOS-only escape hatch

If you ever run gunicorn directly on macOS (not in Docker) for debugging,
export `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` or `gunicorn --preload`
will crash on first request. This variable is **not needed** inside Linux
containers.

### Secrets you must not commit

- `SENTRY_DSN`
- `SLACK_WEBHOOK_URL`
- `POLYMARKET_API_KEY` (if you upgrade to authenticated CLOB calls)
- `ADMIN_BASIC_AUTH` (for `/admin/*` routes; see §12)

Generate strong values with `openssl rand -hex 32`, store them in your
secrets manager (1Password, Bitwarden CLI, AWS Secrets Manager, etc.) and
inject them at deploy time only.

---

## 4. Database / cache

The stack has no relational database. Persistent state lives in **Redis**
and a handful of JSON artefacts under `web/data/` and
`arbstuff/dashboard_state.json`. Redis is configured with append-only file
(AOF) plus a snapshot every 60 seconds on 1000+ writes:

```
command: ["redis-server", "--appendonly", "yes", "--save", "60", "1000"]
```

A named volume `pfm-redis-data` is mounted at `/data`. Back it up regularly
(see §7).

### Factor catalog seed

The 1228-entry factor catalog is shipped as `api/src/pfm/factors.yml`. On
first boot, the API loads it into memory; on prewarm-enabled deployments it
also fetches the top-N price histories into Redis. Confirm the seed worked:

```bash
docker compose exec api curl -fsS http://localhost:8000/factors/all \
  | jq '.factors | length'
# 1228
```

If you ever need to rebuild the cache from cold, restart the API with
`PFM_FACTOR_PREWARM_TOP_N=200` and watch `docker compose logs -f api` for
the `factor_prewarm.done` line (~3 minutes).

---

## 5. Deploy

Production launch is a single command:

```bash
cd /opt/pfm
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  --env-file .env.prod up -d
```

This brings up three containers (`pfm-redis`, `pfm-api`, `pfm-web`) on the
`pfm` bridge network. The API binds `:8000`, the static frontend binds
`:8080`. Redis is internal only — never expose `:6379`.

Resource limits from `docker-compose.prod.yml`:

| Container | CPU limit | Memory limit |
| --------- | --------- | ------------ |
| `api`     | 2         | 1 GiB        |
| `web`     | 1         | 256 MiB      |
| `redis`   | 1         | 512 MiB      |

Bring up sequence is enforced by `depends_on` + healthchecks: Redis must be
healthy before the API starts; the API must be healthy before nginx (web)
serves traffic.

### First-deploy seed

After `up -d`, prewarm the curated factors and confirm live signals are
populating:

```bash
docker compose exec api python -m pfm.scripts.warm_curated_factors --top-n 200
docker compose exec api curl -fsS http://localhost:8000/live-signals \
  | jq '.signals | length'
```

---

## 6. Verify

Three endpoints form the production smoke test:

```bash
# 1. Liveness
curl -fsS https://terminal.example.com/health | jq .
# {"status":"ok","version":"...","uptime_s":...}

# 2. Deep readiness (Redis, upstream APIs, prewarm staleness)
curl -fsS https://terminal.example.com/health/deep | jq .
# {"status":"ok","checks":{"redis":"ok","polymarket":"ok",...}}

# 3. End-to-end OLS fit
curl -fsS -X POST https://terminal.example.com/fit \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"NVDA","factors":["recession-2025","cpi-cool"],
       "start":"2025-01-01","end":"2025-05-01"}' \
  | jq '.r2, .betas'
```

A working deploy returns:
- `/health` → `200` in <50 ms
- `/health/deep` → `200` in <500 ms with all checks `"ok"`
- `/fit` → `200` in <8 s warm, <90 s cold (first call after restart)

If `/health/deep` reports any `degraded` field, consult
`docs/TROUBLESHOOTING.md` before declaring success.

OpenAPI surface verification:

```bash
curl -fsS https://terminal.example.com/openapi.json \
  | jq '.paths | keys | length'
# 271
```

---

## 7. Backup

`api/scripts/backup.sh` snapshots all critical state into a single
timestamped tarball under `/tmp/pfm-backup-<UTC>.tar.gz`:

```bash
bash /opt/pfm/api/scripts/backup.sh /var/backups/pfm/
```

Contents:

- `.coordination/active-edits.json` and `active-edits-archive.jsonl`
- `api/src/pfm/factors.yml`
- `web/data/alpha_strategies.json`
- `web/data/alpha_graveyard.json`
- `web/data/live_signals.json`
- `arbstuff/dashboard_state.json` (if present)
- `pfm-redis-data` volume snapshot via `redis-cli --rdb`

Recommended schedule:

```cron
# /etc/cron.d/pfm-backup
0 */6 * * * pfm-deploy bash /opt/pfm/api/scripts/backup.sh /var/backups/pfm/ \
            >> /var/log/pfm-backup.log 2>&1
0 3   * * * pfm-deploy find /var/backups/pfm/ -type f -mtime +14 -delete
```

Off-host: `rclone copy /var/backups/pfm/ b2:pfm-backups/` or equivalent S3
sync. The full backup is <50 MB so retention is cheap.

---

## 8. Rollback

`api/scripts/rollback.sh` reverses to a previous backup tarball:

```bash
bash /opt/pfm/api/scripts/rollback.sh /var/backups/pfm/pfm-backup-20260516T030001Z.tar.gz
```

Pipeline (from the script header):

1. Graceful SIGTERM to gunicorn master, escalate to SIGKILL after grace.
2. Invoke `api/scripts/restore.sh` to expand the tarball back into place.
3. Restart gunicorn (compose `up -d` or `${PFM_GUNICORN_CMD}` fallback).
4. Verify `GET /health` returns `"ok"`.
5. Append outcome to `.coordination/deploys.log`.

Exit codes: `0` success; `1` bad args; `2` restore failed; `3` restart
failed; `4` healthcheck failed. Always tail `.coordination/deploys.log`
after a rollback — the script records the outcome line so you have an
audit trail.

If a rollback fails mid-restore, the previous Redis AOF will still be
intact in the named volume; you can manually `docker compose restart api`
to fall back to the on-disk state.

---

## 9. Monitoring

`api/scripts/monitor.sh` is a polling watchdog:

```bash
nohup bash /opt/pfm/api/scripts/monitor.sh \
  https://terminal.example.com 10 >/var/log/pfm-monitor.log 2>&1 &
```

Per tick (default 10 s):

- `GET /health` — must return JSON `{"status":"ok"}`
- `GET /health/deep` — logs per-upstream latency
- `GET /metrics/audit` — extracts `err_rate`, alerts to stderr if >5%
- `ps gunicorn` — counts live workers
- `redis-cli ping` — if `REDIS_URL` is set

Alerts fire when the DOWN streak exceeds 3 consecutive checks (~30 s) and
when err_rate crosses the threshold. Pipe stderr into your alerting
sidecar:

```bash
bash monitor.sh ... 2> >(tee -a /var/log/pfm-alerts.log \
  | xargs -I{} curl -X POST "$SLACK_WEBHOOK_URL" -d "text={}")
```

Additional ad-hoc dashboards:

- `GET /metrics/audit` — request counts, error rate, p50/p95/p99 latency
- `GET /admin/cache-stats` — Redis hit ratio, prewarm staleness, key
  counts per namespace
- `GET /health/deep` — per-upstream status, last successful fetch age

Prometheus scraping is on the future-work list; for now, treat
`/metrics/audit` as the source of truth and shape it into your
observability platform of choice.

---

## 10. Scaling

The API is sync (statsmodels OLS, pandas) so gunicorn workers scale
horizontally on CPU. Rule of thumb:

```
workers = (2 * vCPU) + 1
```

Set via `PFM_GUNICORN_WORKERS` (read by the entrypoint):

| vCPU | Workers | Notes                                          |
| ---- | ------- | ---------------------------------------------- |
| 1    | 3       | Demo only.                                     |
| 2    | 5       | Production single-tenant default.              |
| 4    | 9       | Heavy use, plus enable factor prewarm `TOP_N=200`. |

Worker class: `uvicorn.workers.UvicornWorker` (FastAPI is async-capable but
the hot path stays sync). Each worker holds ~150 MB resident; size the
host RAM accordingly (workers × 150 MB + Redis 512 MB + 512 MB headroom).

### Redis scaling

`docker-compose.prod.yml` caps Redis at 512 MB. Enforce the same inside
Redis to avoid OOM-kill surprise:

```
maxmemory 480mb
maxmemory-policy allkeys-lru
```

Most factor histories are ~25 KB compressed; with the default TTL
(`CACHE_TTL_SECONDS=3600`) you fit ~15k entries comfortably. If you raise
`PFM_FACTOR_PREWARM_TOP_N` above 500, bump the Redis memory limit to 1 GB.

### Vertical first, horizontal later

A single 4 vCPU / 8 GB VM saturates somewhere around 80 RPS on `/fit`
(cold) or 500 RPS on `/health`. Past that, run two API replicas behind
nginx with `ip_hash` and a shared Redis. Do **not** shard Redis — the
working set is small and the prewarm logic assumes a single keyspace.

---

## 11. Logging

The API uses `structlog` configured for JSON output in production
(`ENV=production`). Each log line is a single JSON object so it pipes
cleanly into any aggregator. Sample:

```json
{"timestamp":"2026-05-16T21:07:01Z","level":"info","event":"fit.complete",
 "ticker":"NVDA","r2":0.34,"latency_ms":2841}
```

### Log rotation

Docker's default `json-file` driver grows without bound. Add to
`/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "5"}
}
```

Then `sudo systemctl restart docker`. For host-level scripts (monitor,
backup), use `logrotate`:

```
# /etc/logrotate.d/pfm
/var/log/pfm-*.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
    copytruncate
}
```

Sensitive fields (`api_key`, `auth_token`, `password`) are scrubbed by the
structlog processor before serialisation; do not disable this filter.

---

## 12. Security

### TLS via nginx

Terminate TLS at the host. `nginx` config (abridged):

```nginx
server {
    listen 443 ssl http2;
    server_name terminal.example.com;
    ssl_certificate     /etc/letsencrypt/live/terminal.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/terminal.example.com/privkey.pem;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;     # SSE streams need headroom
    }
    location / {
        proxy_pass http://127.0.0.1:8080/;
    }
}
server { listen 80; return 301 https://$host$request_uri; }
```

Use `certbot --nginx` for free certs and auto-renewal. Set HSTS once the
cert chain is stable.

### Admin authentication

All `/admin/*` routes require HTTP Basic auth (validated by FastAPI
dependency). Configure via:

```
ADMIN_BASIC_AUTH_USER=ops
ADMIN_BASIC_AUTH_PASS=$(openssl rand -hex 24)
```

Restrict the path further at the nginx layer if you want IP allowlisting:

```nginx
location /api/admin/ {
    allow 203.0.113.0/24;
    deny  all;
    proxy_pass http://127.0.0.1:8000/admin/;
}
```

### Other hardening

- `SECURE_PROXY_SSL_HEADER` is honoured when `X-Forwarded-Proto` is sent.
- CORS is allowlist-based (`CORS_ORIGINS`); never set `*` in production.
- Run `pip-audit` and `npm audit` on every release (see
  `docs/DEPS_AUDIT.md`).
- Container processes run as `app` UID 1000, not root.
- Redis is bind-mounted only on the internal bridge network.

---

## 13. CI/CD

`.github/workflows/ci.yml` defines three jobs that all must pass before a
release tag is created:

1. **lint** — `ruff check .` + `mypy api/src/pfm` (strict on `model.py` and
   `attribution.py`).
2. **test** — `pytest -q` against the full suite (~2700 tests, ~80 s).
   Coverage gate is 70 % on `model.py` and `attribution.py`.
3. **build** — `docker compose build` to ensure the image still assembles
   from a clean cache.

Release pipeline (`.github/workflows/release.yml`):

```yaml
on:
  push:
    tags: ['v*']
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - run: docker compose build
      - run: docker compose push
      - name: Deploy
        run: ssh deploy@prod 'cd /opt/pfm && git pull && \
             docker compose pull && \
             docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d'
```

Branch protection: `main` requires the `lint`, `test`, and `build` checks
plus one human review. Tags are signed (`git tag -s`) and verified by the
release workflow before push.

---

## 14. Hosting recommendations

The stack is small and stateful-but-simple, so cheap VPS hosts work well.

### Hetzner Cloud (recommended for cost)

- **CX22** (2 vCPU, 4 GB, 40 GB SSD) — €5.83/month. Fits the
  single-tenant prod profile comfortably.
- Use the **Falkenstein** or **Nuremberg** regions for the lowest latency
  to Polymarket / Kalshi servers (both are US east); for EU users the
  **Helsinki** region is faster.
- Attach a Hetzner volume (10 GB, €0.40/month) at `/var/backups/pfm` so
  backups survive instance rebuilds.

### Fly.io

- One `fly.toml` per service (`api`, `web`, `redis`). Use a **shared
  CPU 2x** machine with 1 GB RAM for the API and a `fly volumes create
  redis_data` for persistence.
- Pros: built-in TLS, anycast, zero-downtime deploys via `fly deploy
  --strategy rolling`.
- Cons: cold starts if you scale to zero. Pin `min_machines_running = 1`
  for any production tenant.

### DigitalOcean

- **Basic Droplet** 2 vCPU / 4 GB / 80 GB — $24/month. Add their **Managed
  Redis** ($15/month, 1 GB) if you want them to handle persistence and
  failover; otherwise stay with the in-cluster Redis container.
- Spaces ($5/month) doubles as your off-host backup target.
- DO Marketplace ships a Docker image that pre-installs the engine and
  `ufw` rules, saving ~20 minutes of setup.

### What to avoid

- Heroku / Render free tiers — they sleep idle dynos, which breaks
  prewarm and live-signals jobs.
- AWS Lambda — the OLS path is sync and uses pandas; cold-start latencies
  are punitive.
- Kubernetes — overkill for one API + one Redis. Revisit only if you need
  >2 API replicas or multi-region.

---

## Appendix: deploy day runbook

1. Tag release on `main` (`git tag -s v$(date +%Y.%m.%d) -m "..."`).
2. Wait for CI green on the tag (`gh run watch`).
3. SSH to host, `cd /opt/pfm`, `git fetch --tags && git checkout <tag>`.
4. `bash api/scripts/backup.sh /var/backups/pfm/` (pre-deploy snapshot).
5. `docker compose -f docker-compose.yml -f docker-compose.prod.yml \
   --env-file .env.prod pull && ... up -d`.
6. Run §6 smoke tests. If any fails, immediately
   `bash api/scripts/rollback.sh /var/backups/pfm/<latest>.tar.gz`.
7. Tail `docker compose logs -f api` for 5 minutes; watch
   `/var/log/pfm-monitor.log` for the same window.
8. Announce in `#ops` with the version, tag, and outcome.

A clean deploy round-trip takes about 8 minutes. Anything longer means
something is off — stop and investigate before continuing.
