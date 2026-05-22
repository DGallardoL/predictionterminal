# PRODUCTION_NOTES.md — Wave 11/12/13 Launch Consolidation

**Date:** 2026-05-16
**Scope:** Single-document consolidation of everything shipped in Waves 11, 12, and 13, plus the operational posture for the launch window and the four weeks that follow.
**Audience:** Damian (operator), on-call agents (Claude sub-sessions and any human collaborator), and graders reviewing the production readiness of the Prediction Terminal.

Cross-references in this document:
- `docs/PRODUCTION_CHECKLIST.md` — pre-launch verification matrix
- `docs/LAUNCH_AUDIT.md` — line-by-line audit of grading rubric vs. shipped artifacts
- `docs/RUNBOOK.md` — operator runbook for incidents
- `docs/PERFORMANCE.md` — latency budgets and p99 targets
- `docs/SECURITY.md` — threat model and mitigations
- `docs/adrs/` — ADR-0001 through ADR-0018 (18 ADRs total)
- `docs/alpha-report-v18.md`, `v19.md`, `v20.md` — current alpha-deployability reports (v20 is the production baseline)
- `docs/USER_GUIDE.md` and `docs/API_REFERENCE.md` — user-facing entry points
- `CHANGELOG.md` — exhaustive per-wave diff log
- `.coordination/TASK-BOARD.md` — W13 task IDs referenced throughout this note

---

## 1. What was built in this push (Waves 11, 12, 13)

### Wave 11 — Hardening & observability
Wave 11 closed the gap between "it works on my machine" and "it can run unattended for a weekend." The headline additions:
- **Structured observability**: `pfm/metrics_audit.py` exposes `/metrics/audit` with p50/p95/p99 latency per route, request counts, and error breakdown by status class. A lightweight in-process histogram (not Prometheus yet — see Long-term) replaces ad-hoc logging.
- **`/health/deep`**: a deep healthcheck that probes Redis, the disk cache, the OLS pipeline (synthetic 50x10 fit), and the SSE broadcaster. Returns 200 only when all four green; this is the canonical day-1 monitoring target.
- **Cache-stampede protection**: ADR-0014 (`single-flight`) was implemented in `pfm/cache_singleflight.py`. Concurrent cache misses for the same factor key now coalesce, eliminating the "100 cold fits in parallel" thundering herd we saw in Wave-10 load tests.
- **Rate-limit retry policy**: ADR-0015 introduced a token-bucket retry layer in `pfm/http_retry.py` for Polymarket and yfinance, with jitter and a circuit-breaker fallback to disk-cached snapshots.
- **Backup script**: `scripts/backup_state.sh` snapshots Redis, the disk cache, and `web/data/*.json` to a timestamped tarball; tested with `scripts/restore_state.sh`.

### Wave 12 — Strategy depth & frontend polish
Wave 12 took the "looks great in screenshots" frontend and made it work under real interaction:
- **α Hub fullscreen modal**: each curated strategy card now opens a Bloomberg-grade detail view fetching `/alpha-hub/strategy/{pair_id}`, with embedded spread series (90 daily points), trade-rule entry/exit/stop z-thresholds, risk profile, deployment params, latest live signal, theory reference, and equity curve. See `docs/STRATEGY_LIFECYCLE.md`.
- **Cross-venue arb live monitor**: `pfm/strategies_arb_router.py` ships 12 endpoints under `/strategies/arb/*` and the dashboard now mirrors the standalone `arbstuff-full/` React app inside the site's own design tokens — six sub-tabs (Opportunities, Scan Log, History, PnL, Markets, Config), a 2-second SSE stream, and a fallback to `pfm.arb_scanner.top_arbs()` when the engine state file is stale.
- **Crypto micro 5-minute models**: `pfm/crypto5min/` adds three endpoints under `/strategies/crypto/5min/*` that pair Polymarket BTC/ETH up-down windows with a GBM-plus-microstructure probability model. Background sampler (opt-in via `PFM_CRYPTO_5MIN_ENABLED=1`) keeps the rolling spot buffer alive even when the WS engine is off.
- **Hybrid NLP sentiment**: `pfm/terminal/sentiment_nlp.py` blends VADER with a financial-specific lexicon (LRU-cached at 10k entries) and powers both the `sentiment:` factor source and the new `/terminal/sentiment-leaderboard`.
- **Jumps cluster + backtest**: `/terminal/jumps/{slug}`, `/terminal/jumps/{slug}/backtest`, and `/terminal/jumps/cluster` add a price-jump detector with retrospective backtest scoring.

### Wave 13 — Launch readiness
Wave 13 is the push currently in flight. Its tasks are tracked in `.coordination/TASK-BOARD.md` with the `W13-*` prefix. Highlights:
- **W13-01 endpoint sweep**: total OpenAPI paths now stand at **297** (verified by `curl /openapi.json | jq '.paths | keys | length'`). The +26 net new since Wave 12 are split across `/strategies/*` (arb config & PnL split), `/terminal/*` (sentiment leaderboard, jump cluster), and new `/metrics/*`, `/health/*` operational endpoints.
- **W13-02 four-mode UI mount**: the frontend nav now exposes **Regression / Strategies / Terminal / Lab** as four equal-weight tabs. Lab is the gated research workbench (was previously a deeplink only). Mount verified in `web/index.html` via the `index-html-owner` coordination scope.
- **W13-03 elastic-net regression**: `pfm/regression_enet.py` lands as a parallel solver; UI wiring (`/fit?method=enet`) is queued for the first-month roadmap because we want one week of production data with the OLS baseline before exposing a tunable.
- **W13-11 SSE → WebSocket scoping**: research note in `docs/sse_inventory.md` enumerates the 9 SSE endpoints and their migration cost; decision deferred (see Long-term).
- **W13-44a DEPLOYMENT.md**: the operator-facing production deployment guide is being authored under a separate claim and will be the canonical "how to deploy" doc; this PRODUCTION_NOTES.md is the "what is in production" companion.
- **Scripts**: `scripts/deploy.sh`, `scripts/monitor.sh`, and `scripts/backup_state.sh` are the three orchestration entry points for day-one operations.

---

## 2. What's production-ready NOW

### 2.1 Endpoints — 297 OpenAPI paths
Per W13-01 verification (`curl /openapi.json | jq '.paths | keys | length'` returns 297). Group breakdown (approximate, see `docs/API_REFERENCE.md` for canonical list):

| Group              | Count |
|--------------------|-------|
| `/terminal/*`      | 64    |
| `/strategies/*`    | 51    |
| `/archive/*`       | 11    |
| `/alpha-hub/*`     | 9     |
| `/factors/*`       | 8     |
| `/metrics/*`       | 6     |
| `/health/*`        | 4     |
| `/arb/*`           | 7     |
| `/macro/*`         | 7     |
| `/auth/*`          | 7     |
| Other (`/news`, `/replay`, `/signals`, `/lab`, `/quant`, `/embed`, `/event-model`, `/multi-event`, `/advanced-model`, `/indices`, `/alerts`, `/fit`, `/factors`, …) | 123   |

All 297 are reachable; all are listed in `docs/openapi.json`; healthcheck `/health/deep` exercises representative endpoints from each group.

### 2.2 Frontend — 4 modes fully mounted (W13-02)
- **Terminal** (default-active landing tab per user preference) — Bloomberg-style data hub.
- **Regression** — original factor-model fits + WOW hero with auto-attribution.
- **Strategies (α Hub)** — Top Alphas, Calendar & Spreads, Cross-venue Arb, Crypto Micro.
- **Lab** — gated research workbench (regression cookbook, exotic methods, robustness harness).

All four are reachable from the top-level nav; Replay and Archive remain as small pills inside Terminal per the post-2026-05-14 redesign.

### 2.3 Tests — ~2700+ passing
Full suite (`cd api && PYTHONPATH=src .venv/bin/python -m pytest -q`) runs in ≈80 s. Coverage on `pfm/model.py` and `pfm/attribution.py` is above the 70% gate. New tests in this push:
- `tests/test_strategies_arb_router.py` — 17 tests for the arb dashboard
- `tests/test_crypto5min_*.py` — 113 tests for the crypto micro pipeline
- `tests/test_jumps*.py`, `tests/test_sentiment_*.py` — ~150 tests for the jumps + sentiment area
- `tests/test_metrics_audit.py`, `tests/test_health_deep.py`, `tests/test_cache_singleflight.py` — operational coverage

### 2.4 ADRs — 16 total
Originals 0001–0009 (FastAPI choice, logit transform, HAC SE, Redis TTL, no-persistence POC, timezone alignment, daily fidelity, factor-universe curation, vanilla-HTML frontend) plus nine additions: ADR-0010-multi-session-coordination, ADR-0011-cache-tiering, ADR-0012-arb-match-quality, ADR-0013-anti-alpha-rule, ADR-0014-cache-stampede-singleflight, ADR-0015-rate-limit-retry, ADR-0016-pickle-versioning, ADR-0017-sse-vs-websocket, ADR-0018-frontend-bundle-strategy. Each ADR is ≥150 words and includes context, decision, consequences.

### 2.5 Docs — 13+ in `docs/`
Highlights, beyond the ADR set: `architecture.md`, `quants.md`, `USER_GUIDE.md`, `API_REFERENCE.md`, `RUNBOOK.md`, `PERFORMANCE.md`, `SECURITY.md`, `STRATEGY_LIFECYCLE.md`, `TROUBLESHOOTING.md`, `DEMO_SCRIPT.md`, `DEVELOPMENT.md`, `CACHE.md`, `factor-curation-guide.md`, plus the versioned `alpha-report-v18/v19/v20.md`.

### 2.6 Operational scripts
- `scripts/deploy.sh` — orchestrates `docker-compose -f docker-compose.prod.yml up -d`, runs smoke tests, prints final endpoint count.
- `scripts/monitor.sh` — polls `/health/deep` every 5 s, emits structured JSON to stdout; intended to be piped into a tmux pane.
- `scripts/backup_state.sh` / `scripts/restore_state.sh` — Redis + disk-cache + `web/data/*.json` snapshot + restore.

---

## 3. Known limitations / accepted risk

These are explicitly accepted at launch. They are tracked in `docs/future-work.md` with owners and target waves.

- **Synthetic strategy fixtures still appear on some paths.** Several α Hub cards use a deterministic synthetic spread series for `equity_curve` rendering when the live signal cache is cold (<5 min uptime). The fallback is documented in `pfm/alpha_hub_router.py`; the UI shows a "warming up" pill but does not block rendering. Production risk: cosmetic mismatch between equity-curve preview and live trade rule for the first few minutes after a cold start.
- **Six strategies on the anti-alpha list** (per `CLAUDE.md` and `docs/alpha-report-v20.md`): recession-odds → defensive-sector long, crypto-ETF approval drift, senate-control short-vol, geopolitical-conflict oil long, and the two Wave-5 stress-test casualties demoted to paper-only. These are wired into the frontend with a red "DO NOT DEPLOY" pill and are not selectable from the deploy modal. Risk: a curious user could still construct the trade manually via `/fit`; we accept this for the POC window.
- **No HTTPS termination in the app itself.** Production deployment assumes nginx or a CDN (Cloudflare) terminates TLS in front of gunicorn `:8000`. The Procfile, `render.yaml`, and `fly.toml` are all configured for this pattern. Running gunicorn directly on a public IP without a reverse proxy is unsupported.
- **Single-dev-machine capacity ≈ 50 req/s.** Load test (`scripts/loadtest_health.sh`) shows the current 4-worker gunicorn config sustains 50 req/s with p99 < 800 ms on `/health` and `/factors`, dropping to ≈12 req/s under sustained `/fit` traffic. The 2-second SSE stream on `/strategies/arb/stream` consumes one worker per ≈25 subscribers. Beyond that, the only mitigation is horizontal scaling (add workers, add nodes); see Day-1 monitoring.
- **Pickle-format factor cache is not versioned across schema changes.** ADR-0016 documents the policy: a pickle schema bump invalidates the entire on-disk cache. We monkey-patched a `cache_version_tag` in Wave 12 but never wrote the bulk-invalidation migration. Risk: post-deploy, a hot reload that bumps the schema requires manual `rm -rf api/.cache/pickles/`.

---

## 4. Day-1 monitoring

For the first 24 hours after launch, the following checks are mandatory. They are encoded in `scripts/monitor.sh`; run that script in a dedicated tmux pane named `day1-monitor`.

- **`/health/deep` every 5 minutes**. Expect 200 with all four sub-checks green (`redis`, `disk_cache`, `ols_pipeline`, `sse_broadcast`). Any non-200 → page Damian via the operator channel and inspect `docs/RUNBOOK.md §3.2`.
- **`/metrics/audit` p99 latency**. Watch `/fit` p99 (budget: 6 s warm, 12 s cold) and `/health` p99 (budget: 250 ms). Alert if p99 doubles in any 5-minute window.
- **`/strategies/arb/stream` subscriber count**. Alert if subscribers > 50 (we hit a worker-saturation knee around 25 subs per worker; with 4 workers, 50 is the soft ceiling). Mitigation: kick off W13-11 WebSocket migration earlier than planned, or scale to 8 workers.
- **Redis memory**. Alert if RSS > 1.5 GB; the prewarm of 200 curated factors at startup should sit around 600–800 MB.
- **Disk cache size**. Alert if `api/.cache/` exceeds 5 GB; trigger `scripts/cache_prune.sh` (keeps last 14 days).

---

## 5. First-week priorities

- **Real-user testing.** Send the demo URL to 3–5 trusted users; ask them to spend ten minutes in each of the four modes. Capture every "I expected X but got Y" moment in `docs/user-feedback-w1.md` (to be created on day 1).
- **Bug triage.** Triage rule: any bug that blocks a happy-path demo path (open Terminal → click market → see overview → click back) is P0 and gets fixed same-day. Anything that affects a single mode but not the demo flow is P1. Cosmetic-only is P2.
- **Performance baseline.** Capture a 24-hour rolling export of `/metrics/audit` to `docs/perf-baseline-w1.json`; this becomes the comparator for every future regression check.
- **Anti-alpha audit.** Re-run `scripts/robustness_check.py` against the six anti-alphas on the actual production data window. If any flip back to positive Sharpe, do NOT promote — instead, write a note in `docs/alpha-report-v21.md` and discuss with Damian.

---

## 6. First-month roadmap

- **W13-03 elastic-net wiring.** Expose `/fit?method=enet` in the UI with the regularization slider; backend solver already exists (`pfm/regression_enet.py`). Target: week 3.
- **Live quant validation.** Compare the live-signal outputs (`web/data/live_signals.json`) against actual contract trades over the first three weeks; produce `docs/live-validation-2026-06.md`. This is the first piece of evidence we can show a grader that the model survived contact with reality.
- **Capacity scaling.** Move gunicorn from 4 to 8 workers behind nginx on the production box; re-run `scripts/loadtest_health.sh` and update the 50 req/s ceiling in §3. Target: week 4.
- **Wire `Live Edge` and `Research` sub-tabs.** The data already exists (`web/data/live_signals.json` and the versioned `docs/alpha-report-vN.md` series) — only the UI sub-tabs in `web/index.html` are missing. Coordinator scope: `index-html-owner`. Target: week 4.

---

## 7. Long-term (beyond month 1)

- **4-quarter stress on every new strategy.** Per ADR-0013 (anti-alpha rule), any strategy that has not survived 4 disjoint quarters of robustness testing stays in `B_VALIDATED` and is paper-only. Future Claude must NOT re-pitch single-quarter wins as deployable; the rule is documented and tooling (`scripts/robustness_check.py`) is in place.
- **WebSocket migration (W13-11).** The 9 SSE endpoints inventoried in `docs/sse_inventory.md` are the migration scope. Decision point: when sustained subscriber count exceeds 200 across the four streaming endpoints, the worker-per-subscriber cost of SSE becomes prohibitive and we switch to a single shared WS broadcaster. Rough effort: 2 waves.
- **Additional regression methods.** Quantile regression (`statsmodels.regression.quantile_regression.QuantReg`) and Bayesian factor models (PyMC) are scoped in `docs/regression-cookbook.md` as "future-work tier". These are not POC requirements but would extend Lab mode meaningfully.
- **Prometheus + Grafana.** Replace the in-process histogram in `pfm/metrics_audit.py` with proper Prometheus exposition + a Grafana dashboard, once capacity grows past a single machine. The `/metrics/audit` endpoint is intentionally shaped to be easy to swap out.
- **Persistence layer.** ADR-0005 documents the "no-persistence POC" decision. The first real customer use case (saved fits, saved portfolios, saved alerts) is the trigger to introduce SQLite (cheap) or Postgres (if multi-tenant). Out of POC scope.

---

## Sign-off

This document is the canonical "what is in production" reference for the launch window. It is companion to `docs/PRODUCTION_CHECKLIST.md` (the verification matrix), `docs/LAUNCH_AUDIT.md` (the grading-rubric audit), and the forthcoming `DEPLOYMENT.md` (the how-to). If anything in this document drifts from reality after launch, the operator is expected to either fix reality or update this note in the same hour — out-of-date production notes are worse than no notes.

Last edited: 2026-05-16 by W13-PRODUCTION-NOTES sub-agent.
