# Prediction Terminal Runbook

On-call reference for the Prediction Terminal stack. Practical recipes for
the most frequent operational situations — read top-to-bottom on your first
shift, then dip in by symptom.

## Quick Reference

- Gunicorn: `:8000`, 4 workers (UvicornWorker, `pfm.main:app`)
- Frontend: `:8080`, static httpd serving `web/`
- Redis: `:6379` (L2 cache + arb engine state)
- Logs: stdout + `structlog` (pipe to `pfm.main.log` under `api/`)
- Metrics: `GET /metrics/audit` (T-metrics ring buffer)
- Deep health: `GET /health/deep` (per-upstream probe)
- Coordination: `.coordination/active-edits.json`, `restart-requests.txt`

## Restart safely

Restarting the shared `:8000` worker pool affects every open browser tab.
Always coordinate first.

1. Check `.coordination/restart-requests.txt` for pending coordinator
   requests; append your reason BEFORE acting.
2. Drain: `kill -USR1 <gunicorn_pid>` (graceful reload, in-flight requests
   complete, new workers fork).
3. If full restart needed: `pkill -HUP -f "gunicorn pfm.main"`. HUP rereads
   config; SIGTERM is last resort.
4. Verify: `curl -fsS :8000/health` must return `{"status":"ok",...}` within
   10s. If not, tail logs immediately.
5. macOS fork-safety: workers SIGABRT during `requests`/`urllib3` calls
   unless `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` is exported. Put it in
   the gunicorn launch env, not your shell rc.
6. Document the restart reason in `.coordination/restart-requests.txt` so
   the next session knows why their cached state vanished.

## Reading /metrics/audit

- 10k-entry ring buffer since process start, per endpoint.
- Each row: `count`, `p50_ms`, `p95_ms`, `p99_ms`, `err_rate`, `last_ms`.
- `err_rate > 5%` on any endpoint → investigate immediately; pull the path
  into the triage flow below.
- `p99 > 10s` → upstream issue almost certain; cross-check `/health/deep`
  before blaming our code.
- Buffer resets on worker restart; if metrics look "too clean" after an
  incident, you may have just lost evidence.

## Triage 5xx

1. `GET /health/deep` — identifies which upstream is degraded (Polymarket
   CLOB, gamma, yfinance, Binance, Redis).
2. `GET /metrics/audit` — find the endpoint with non-zero `err_rate`.
   Confirm the timing aligns with the upstream from step 1.
3. Tail `api/pfm.main.log` for the traceback. Look for the request ID
   structlog emits at INFO when the request started.
4. Common causes: Polymarket 503 during odds reshuffle, yfinance timeout
   during US-open, Redis connection refused after a host reboot.
5. Each upstream has documented backoff in `pfm/sources/<name>.py`. If
   retries are not engaging, the source module is the bug, not the upstream.

## Cache invalidation

- L1 (per-process LRU + lifespan prewarm): restart the worker. There is no
  programmatic flush by design — the prewarm is the contract.
- L2 (Redis): `redis-cli SCAN 0 MATCH "pfm:*"` then `DEL` returned keys.
  For a single factor: `DEL "pfm:factor:<slug>"`.
- Lifespan prewarm (200 curated factors + arb seeds) runs automatically on
  process startup. Expect a 15–25s warm-up window where p95 is elevated.

## Common incidents

### Gunicorn worker crashed (SIGABRT on macOS)

Cause: fork-safety in `requests`/`grpc`/`urllib3` post-fork on Darwin. Fix:
export `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` in the worker env. Confirm
the dyld error in stderr matches this signature before applying.

### CORS errors from browser

Frontend at `:8080` calls API at `:8000`. Check middleware order in
`pfm/main.py` — `CORSMiddleware` must be the OUTERMOST middleware (the LAST
one added via `app.add_middleware`). If a new middleware was added after it,
preflight requests get rejected.

### Stale `active-edits.json` claims

A coordinator session may have crashed mid-edit, leaving claims that block
new work. Manually drop expired entries:

```
jq '[.[] | select(.expires_at > (now|todateiso8601))]' \
   .coordination/active-edits.json > /tmp/ae.json && \
mv /tmp/ae.json .coordination/active-edits.json
```

Never delete unexpired entries belonging to other sessions.

### Disk full (Redis or logs)

Check `/var/log/`, `/data/redis/`, and `api/*.log`. Rotate with
`logrotate -f /etc/logrotate.d/pfm` or manually truncate. Redis with
`maxmemory` unset will OOM the box — verify `redis-cli CONFIG GET maxmemory`
returns a non-zero value.

### Arb stream "stale" banner in UI

`arbstuff/dashboard_state.json` is older than 3 min. The router falls back
to `pfm.arb_scanner.top_arbs()` automatically; if banner persists, the
fallback is also failing — check Polymarket+Kalshi reachability.

## Useful commands

```
# Top 10 slowest endpoints
curl :8000/metrics/audit | jq '.endpoints | to_entries
  | sort_by(-.value.p95_ms) | .[0:10]'

# Live arb stream (Ctrl-C to exit)
curl -N :8000/strategies/arb/stream

# Force prewarm (admin endpoint, if mounted)
curl :8000/_admin/prewarm

# Pick out the noisy endpoint by error rate
curl :8000/metrics/audit | jq '.endpoints | to_entries
  | map(select(.value.err_rate > 0.05))'

# Confirm fork-safety env is set on the running worker
ps -E -p $(pgrep -f "gunicorn pfm.main" | head -1) | tr ' ' '\n' \
  | grep OBJC_DISABLE
```

## Escalation

- Backend (API, quant code, factor catalog): Damian
- Frontend (`web/index.html`, alpha hub, terminal panels): Damian
- Quant / model questions, alpha tier disputes: Damian
- Coordination protocol questions: this file plus
  `.coordination/PROTOCOL-V2.md`

When paging Damian, lead with: (a) the failing endpoint, (b) `err_rate` and
`p99` from `/metrics/audit`, (c) which `/health/deep` upstream is red. Do
not page without those three facts.
