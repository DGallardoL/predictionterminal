# Production checklist

Hard-gate this checklist before pushing the project to a public URL. Every
item is a real failure mode that has bitten an open-source project at the
"first 10k users" mark. Tick boxes top-to-bottom; do not skip ahead.

---

## Pre-launch — Security

- [ ] **No secrets in git.** Run `git ls-files | xargs grep -lE 'sk_|api_key|password|token' | grep -v test | grep -v fixture` and confirm zero hits.
- [ ] **`.env` is gitignored.** `git check-ignore .env` returns the path.
- [ ] **`.env.example` lists every required variable** with a placeholder
      and a one-line comment. No real values.
- [ ] **CORS is locked down.** `CORS_ORIGINS` set to your final domain
      (no `*`). Verify with `curl -H 'Origin: https://attacker.example' -I https://your-app/health` — should NOT return `Access-Control-Allow-Origin: *`.
- [ ] **HTTPS terminated.** `curl -I http://your-app` redirects to
      `https://`. HSTS header present. Fly.io / Render do this automatically.
- [ ] **Security headers set in nginx.** `X-Frame-Options: SAMEORIGIN`,
      `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, `Strict-Transport-Security`.
- [ ] **Rate limiting active.** nginx `limit_req` and FastAPI `slowapi`
      both engaged. Test: 200 requests in 10s from one IP returns 429s.
- [ ] **Webhook HMAC verified.** If alerts fire, the receiver checks
      `X-PFM-Signature: sha256=<hex>`. Do not deploy with `PFM_ALERTS_HMAC_SECRET` unset in production.
- [ ] **No debug toolbar / stack traces in responses.** `ENV=production`
      set, FastAPI exception handlers return `{"detail": "internal error"}` not the trace.
- [ ] **`pip-audit` clean.** No HIGH or CRITICAL CVEs in the dependency
      tree. Re-run on every deploy.
- [ ] **Disclaimer footer visible.** "Not investment advice · UTC
      timestamp" is in every page render.

## Pre-launch — Infrastructure

- [ ] **Redis has persistence.** AOF on self-host (`appendonly yes` in
      `redis.conf`), or use a managed Redis with backups enabled.
- [ ] **Redis has a password.** `REDIS_URL` includes a password
      (`redis://default:...@host:6379`), never `redis://host:6379`.
- [ ] **Database (SQLite for alerts) is on a persistent volume.** Not
      ephemeral container storage.
- [ ] **Backups configured.** Daily snapshot of the alerts SQLite +
      `web/data/*.json`. Keep 30 days.
- [ ] **uvicorn `--workers 4`** in production Dockerfile.
- [ ] **Healthcheck path `/health`** wired into the platform (Fly.io
      `[[checks]]`, Render `healthCheckPath`, etc.).
- [ ] **`/health/detail` returns 200** with `redis: ok`, `git_sha`
      populated, uptime > 0.

## Pre-launch — Code quality

- [ ] **CI is green on the deploy commit.** All three jobs (test, lint,
      build) pass on `main`.
- [ ] **`pytest` passes** (2547+ tests).
- [ ] **`ruff check .` clean.**
- [ ] **`mypy --strict` clean** on the `pfm/` package.
- [ ] **Coverage ≥ 70 %** on `model.py`, `attribution.py`, every strategy
      module.
- [ ] **No bare `except:`** in production code (`grep -rn 'except:' api/src/pfm/ | grep -v 'except [A-Z]'` returns zero).

## Pre-launch — Observability

- [ ] **Sentry DSN set** (or your equivalent error tracker). Send a test
      exception; confirm it lands in Sentry within 60 s.
- [ ] **Prometheus `/metrics` endpoint reachable** but not publicly
      indexed (rate-limited, optional auth).
- [ ] **Structured logs.** No `print()` statements; everything goes
      through `logging` / `structlog`. Logs include timestamp, level,
      and `git_sha`.
- [ ] **Request IDs propagated.** `X-Request-ID` set on every response.

---

## Launch day

- [ ] **Smoke test runs green.** `./scripts/smoke_test.sh https://your-app` returns all-green.
- [ ] **Manual click-through.** Open the live URL on a clean browser
      (no localStorage). Cmd-K works. Top movers loads. Quote page
      renders all eight panels. α Hub loads.
- [ ] **Three curls return expected shape:**
      ```bash
      curl -s https://your-app/health/detail | jq .status   # → "ok"
      curl -s https://your-app/factors | jq '.factors | length'  # → 1228
      curl -s -X POST https://your-app/reverse-finder \
        -H 'content-type: application/json' \
        -d '{"ticker":"NVDA","lookback_days":90}' | jq '.top_factors | length'  # → 5
      ```
- [ ] **OpenAPI docs accessible:** `https://your-app/docs` renders.
- [ ] **OpenAPI JSON exported:** `https://your-app/openapi.json | jq '.paths | length'` ≥ 260.
- [ ] **Disclaimer footer visible** in every mode.
- [ ] **First user notified.** Send the URL to one trusted user; ask
      them to break it. Wait 30 minutes before broader announcement.
- [ ] **Sentry receiving events.** Force a 500 with a known-broken
      input; confirm the alert arrives.
- [ ] **Slack alert tested.** If alerts engine is enabled, fire a
      synthetic alert with `PFM_ALERTS_DRY_RUN=0` and confirm receipt.
- [ ] **Status page** (if you have one) updated to "operational".

---

## Post-launch — first 24 hours

- [ ] **Watch logs for the first hour.** `flyctl logs -a pfm-prod` or
      equivalent. Look for repeated 500s, 429s, or upstream timeouts.
- [ ] **Check `/metrics`** for the request-rate, latency p50/p95/p99,
      and 5xx ratio. Alert if p95 > 2 s sustained.
- [ ] **Confirm redis hit-rate > 70 %.** If lower, the cache is not
      warming or the TTL is too short.
- [ ] **Sentry delta**: zero new exception types in the first 24h is
      the goal. Triage anything new within an hour.
- [ ] **Cost check.** Egress, compute, Redis. Confirm the daily run-rate
      is within budget.
- [ ] **Upstream health.** Confirm Polymarket, Kalshi, FRED, yfinance
      have not rate-limited the production IP. Log a daily count.

## Post-launch — first week

- [ ] **Weekly retrospective.** Note any incidents and their root cause
      in `docs/incidents/YYYY-MM-DD.md`.
- [ ] **Performance benchmark.** Re-run `tests/test_async_perf.py`
      against the production URL (read-only endpoints only). Confirm
      no regression vs local.
- [ ] **Backup verification.** Restore the alerts SQLite from yesterday's
      backup into a sandbox container; confirm row count matches.
- [ ] **Cache eviction sanity.** Confirm Redis `INFO memory` shows
      `used_memory_human` is under your `maxmemory` budget. If at the
      cap, raise it or shorten TTLs on cold keys.
- [ ] **Strategy tearsheets refreshed.** Re-run `scripts/robustness_check.py`
      and confirm the live α Hub matches the latest report.
- [ ] **Demote any decayed strategy.** `GET /alpha/decay` flags rolling-
      Sharpe drops. Move flagged ones to `C_TENTATIVE` and update
      `web/data/alpha_strategies.json`.
- [ ] **Update the changelog.** Cut a `v0.x.y` tag with the production
      diff.

---

## Rollback plan

If anything goes wrong in production:

1. **Fly.io**: `flyctl releases list -a pfm-prod`, then `flyctl deploy --image registry.fly.io/pfm-prod:<previous-sha>`.
2. **Render**: Dashboard → Deploys → click previous green deploy → Rollback.
3. **Self-hosted**: `docker compose pull && docker compose up -d` after a `git checkout <previous-tag>`.

For data corruption (alerts SQLite or `web/data/*.json`):

1. Stop the API: `flyctl scale count 0` (or compose `down`).
2. Restore the latest backup into the persistent volume.
3. Restart with `flyctl scale count 2` (or compose `up -d`).
4. Run the full smoke test before re-announcing.

Document every rollback in `docs/incidents/` so the next-time response
is faster.
