# Troubleshooting Guide — Prediction Terminal

This guide collects the most common failure modes encountered while running, demoing, or extending the **Prediction Terminal** stack (FastAPI backend on `:8000`, static frontend on `:8080`, optional Redis cache, optional arb engine). Each entry follows the same structure: the **error message** you'll see in the browser or terminal, the **symptoms** that distinguish it from neighbouring failures, and a **three-step fix** that resolves the underlying cause. If a fix doesn't work in three steps, the entry points you at the next escalation (RUNBOOK.md, the relevant ADR, or a diagnostic endpoint).

A general rule before diving in: **always check `GET /health/deep` first.** It returns the status of every upstream (Polymarket, Kalshi, yfinance, Redis, Binance, news) and the warm-cache state of the lifespan prewarm. Two thirds of the reports below resolve themselves once you see which downstream is the actual culprit.

---

## 1. Frontend won't load — "Connection error" on `:8080`

**Error message.** The browser shows a banner `Connection error: failed to reach API at http://localhost:8000`. Often accompanied by `net::ERR_CONNECTION_REFUSED` or a CORS preflight failure in the dev tools network tab.

**Symptoms.** `:8080` itself serves the HTML shell fine (you see the navbar and tab strip), but every panel shows the spinner indefinitely or shows the global error toast. `curl http://localhost:8000/health` from the same host may or may not work depending on which leg of the chain is broken.

**Three-step fix.**

1. **Confirm both servers are running.** Run `lsof -nP -iTCP:8000 -sTCP:LISTEN` and `lsof -nP -iTCP:8080 -sTCP:LISTEN`. You must see *two distinct* PIDs (FastAPI/gunicorn on 8000, a static server — Python `http.server`, `caddy`, or `npx serve` — on 8080). If 8000 is missing, start it with `cd api && PYTHONPATH=src .venv/bin/uvicorn pfm.main:app --port 8000 --reload`. If 8080 is missing, start it with `cd web && python3 -m http.server 8080`.

2. **Check `web/config.js` `PFM_API_BASE`.** Open `web/config.js` in the repo and verify the `PFM_API_BASE` constant points to `http://localhost:8000` (no trailing slash). The most common regression after a deploy is the value getting overwritten with a production URL or with `https://` against a non-TLS dev server. Reload the page with cache disabled (`Cmd-Shift-R`) after fixing.

3. **CORS / middleware ordering.** Inspect the failing request in the Network tab. If you see an `OPTIONS` preflight returning 400 or no `Access-Control-Allow-Origin` header, the CORSMiddleware has been mounted **after** another middleware that short-circuits the request. Open `api/src/pfm/main.py`, locate the `app.add_middleware(...)` block, and confirm `CORSMiddleware` is the **first** call in that section. If you recently added a custom middleware (auth, rate-limit, metrics), move it *below* the CORS line and restart the dev server. See ADR-0005 for the rationale.

If all three pass, escalate to RUNBOOK.md §"frontend-bringup".

---

## 2. All endpoints return 503 — upstream down

**Error message.** `503 Service Unavailable` with body `{"detail": "upstream temporarily unavailable", "source": "<name>"}`. The frontend overlays a yellow banner reading "Markets feed degraded".

**Symptoms.** *Every* request — not just `/fit` — comes back 503, including ostensibly local endpoints like `/factors`. Logs are full of `httpx.ConnectError`, `httpx.ReadTimeout`, or `429` responses from a single named upstream.

**Three-step fix.**

1. **Identify the source.** Hit `GET /health/deep`. It returns a JSON object keyed by upstream name (`polymarket`, `kalshi`, `yfinance`, `binance`, `redis`, `news`). The first one whose `ok` field is `false` is the culprit. The `last_error` field tells you whether it's a timeout, a 4xx, or a circuit-breaker trip.

2. **Check rate-limit headers in logs.** If the culprit is Polymarket or Kalshi, grep the API log for `x-ratelimit-remaining` or `retry-after`. Polymarket allows 1000 req / 10s but per-IP burst limits are lower. Kalshi's are roughly 10 req/s sustained. If you see `retry-after: <n>`, the upstream itself is throttling you and the only fix is to wait — see ADR-0015 on backoff strategy.

3. **Flip the circuit breaker.** If the upstream is healthy from `curl` but the app still 503s, the in-process breaker is stuck OPEN. Either wait for the half-open probe (60 s by default) or send `POST /admin/circuit/reset?source=<name>` if you have the admin token. As a last resort, restart only the API process (NOT gunicorn on `:8000` if it's shared — see `.coordination/restart-requests.txt`).

---

## 3. `/fit` returns 422 — "insufficient data"

**Error message.** `422 Unprocessable Entity` with body `{"detail":"insufficient overlapping observations for factor '<slug>' (got N, need >= 30)"}`.

**Symptoms.** A specific factor — usually a newly added Polymarket slug — fails consistently while older factors in the same request succeed. The N reported is typically between 0 and 25.

**Three-step fix.**

1. **Preview the factor.** Call `GET /factors/{slug}/preview`. The response contains `n_observations`, `first_date`, `last_date`, and a sparkline. If `n_observations < 30` the factor genuinely has too short a history (a freshly listed Polymarket market, or a resolved one with sparse trades). The fit cannot proceed with HAC-OLS below 30 observations.

2. **Widen the date range.** If the factor *does* have a long history but your `/fit` request used `start=` / `end=` parameters that exclude most of it, broaden them. Use the `first_date` from the preview as your new `start`. For non-overlapping ranges (factor ended before your range began) you must pick a different factor.

3. **Find an alternative.** `GET /factors/themes/{theme}/leaderboard` returns the top factors for the same theme (e.g. `fed`, `elections`, `earnings`) ranked by recent coverage and explanatory power. Pick one with `n_observations >= 60` for a robust fit. If no alternatives exist in the theme, broaden via `/factors/search?q=<keyword>`.

---

## 4. `/fit` returns NaN coefficients — numerical issue

**Error message.** The response is a 200 OK but `betas` contains `NaN` or `Infinity` values; `r_squared` is `null` or `1.0` exactly.

**Symptoms.** The fit completes "successfully" but the diagnostics are obviously broken. The VIF table in the response shows one or more factors with `vif > 100`, or `vif: null` (singular matrix).

**Three-step fix.**

1. **Read VIF for perfect collinearity.** Check the `vif` field in the response. Any VIF above 10 is suspicious; above 100 indicates near-perfect collinearity. If two factors have VIF in the hundreds, they're essentially the same signal — likely two slugs covering the same Polymarket event family, or a daily and weekly aggregate of the same series. Drop one.

2. **Re-run with `?prune_collinear=true`.** Add this query parameter to your `/fit` call. The backend will iteratively drop the highest-VIF factor until all remaining VIFs are below the configurable threshold (default 10). The pruned slugs are returned in the `dropped_factors` field so you know what was removed.

3. **Switch to elastic-net.** For wide regressions (≥ 8 factors) OLS becomes ill-conditioned even without strict collinearity. Use `?method=enet&l1_ratio=0.3` to switch to elastic-net regularisation. The response shape is identical but the coefficients are biased toward zero and the standard errors are not directly comparable to OLS. Document this in your write-up.

---

## 5. Empty α Hub leaderboard — source file missing

**Error message.** The Strategies → Top Alphas tab shows an empty grid with the text "No strategies available". The API returns `200 OK` with body `{"strategies": [], "tier_summary": {}}`.

**Symptoms.** Other Strategies tabs (Calendar & Spreads, Cross-venue Arb, Crypto Micro) work fine. Only Top Alphas is empty.

**Three-step fix.**

1. **Verify the source file exists.** Run `ls -lh web/data/alpha_strategies.json`. The file should be present and non-empty (typical size 80–250 KB). If it's missing or zero-bytes, the leaderboard router has nothing to read.

2. **Regenerate from canonical sources.** Run `cd api && PYTHONPATH=src .venv/bin/python scripts/alpha_tier_regen.py`. This script rebuilds `alpha_strategies.json` from the latest tier assignments, backtest snapshots, and validated-alpha registry. It takes ~15 seconds and writes atomically.

3. **Confirm tier-summary cache invalidation.** After regenerating, hit `POST /alpha-hub/cache/invalidate` (or restart the API process). The leaderboard caches the tier summary for 10 minutes; without invalidation you'll keep seeing the stale empty response. Reload the frontend with `Cmd-Shift-R`.

---

## 6. Slow first request — cold cache

**Error message.** No error per se; the first `/fit` or `/alpha-hub/leaderboard` call after boot takes 8–20 seconds. Subsequent calls are sub-second.

**Symptoms.** The browser spinner runs visibly longer on the first interaction than on every subsequent one. Logs show `prewarm: jumps starting` shortly after boot but no `prewarm: jumps complete` yet.

**Three-step fix.**

1. **Check lifespan progress.** Tail the API log: `tail -f api/logs/app.log | grep prewarm`. You should see entries like `prewarm: jumps starting`, `prewarm: factors loaded (1228)`, `prewarm: redis warm (200 curated)`. The full sequence finishes in 5–15 seconds on a warm machine, up to 30 seconds on a cold one.

2. **Wait for `prewarm: jumps complete`.** Until you see this line, the curated A-tier hero card is being assembled and the leaderboard will be slow. Hammering `/fit` during prewarm starves the warm-pool and makes the cold path even slower. Hold off.

3. **Validate Redis is reachable.** If you see `prewarm: redis unavailable, falling back to L1`, the prewarm completes but caches only in-process. That makes the *second* worker (if running with `--workers 2+`) also cold. Either start Redis (`redis-server`) or accept the L1-only mode for solo-dev.

---

## 7. Arb scanner shows false positives — T76b should have fixed

**Error message.** No explicit error; the Cross-venue Arb dashboard surfaces "opportunities" with negative or implausibly large edges (`edge_bps > 500`). Hovering on the detail pane shows mismatched market questions.

**Symptoms.** The opportunity card pairs a Kalshi market and a Polymarket market whose questions look superficially similar but resolve on different dates or with different criteria. Hide-arb-button clicks do not persist across refresh.

**Three-step fix.**

1. **Run the quality audit.** Hit `GET /arb/quality-audit`. The response lists every active pair, the matching score, the matcher version, and the audit verdict. A pair flagged `false_positive_risk: high` should not be surfaced. If you see such pairs in the live dashboard, the audit and the scanner have drifted apart.

2. **Verify the matching modules are loaded.** Run `cd api && PYTHONPATH=src .venv/bin/python -c "import pfm.arb_matching; print(pfm.arb_matching.__version__)"`. The version must be `>= 0.7.6` for T76b's fix to be active. If it's lower, your installed package is stale — reinstall with `pip install -e api/`.

3. **Force a scanner rebuild.** Issue `POST /arb/rebuild-pairs` (admin token required) to discard the cached pair list and re-run matching with the current matcher version. The next SSE tick (within 2 s) will reflect the rebuilt set. Pairs that were on the blacklist remain hidden.

---

## 8. Coordinator-agent conflicts — `active-edits.json`

**Error message.** Two agents simultaneously edit the same file, the second one's changes vanish. You see `git status` reporting unexpected reverts, or `outcomes.log` entries showing `clobbered` from earlier sessions.

**Symptoms.** Your edits to `web/index.html`, `web/config.js`, or `api/src/pfm/main.py` disappear minutes after you make them. `cat .coordination/active-edits.json | jq 'length'` returns a number well above expected (hundreds of entries, many overlapping).

**Three-step fix.**

1. **Dry-run cleanup.** Run `cd api && python scripts/coord_cleanup.py --dry-run`. The script prints every entry whose `expires_at` is in the past and every overlapping pair of active claims. Read the output carefully — do not skip to the apply step.

2. **Apply with `--apply`.** Once the dry-run output looks sane (no surprise live claims being removed), re-run with `--apply`. This rewrites `active-edits.json` in one atomic write, dropping expired entries and merging duplicate session IDs.

3. **Read the RUNBOOK section on stale claims.** Open `docs/RUNBOOK.md` and search for "stale claims". It describes how to triage a contested file (which session keeps the claim, how to negotiate a pivot via `.coordination/issues.log`). Do **not** delete another session's live claim without writing to `issues.log` first.

---

## 9. macOS gunicorn SIGABRT — fork safety

**Error message.** The gunicorn worker exits with `Abort trap: 6` or the macOS console shows `+[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called`. The API never finishes booting.

**Symptoms.** Crash happens on macOS only (Linux is unaffected). Frequently triggered by NumPy / SciPy / statsmodels imports that touch the macOS Accelerate framework before fork.

**Three-step fix.**

1. **Export the env var.** Run `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` in the shell that launches gunicorn. Add it to your `.envrc` / shell profile if you forget often. This disables Objective-C runtime's fork-safety assertion.

2. **Use `--preload`.** Add `--preload` to the gunicorn invocation so the app is loaded once in the master process before workers fork. This forces deterministic state and works around several other fork-related crashes.

3. **Fall back to uvicorn for dev.** For solo-dev on macOS the simplest fix is to skip gunicorn entirely and run `uvicorn pfm.main:app --port 8000 --reload`. Reserve gunicorn for the docker-compose stack, where it runs on Linux and the issue doesn't surface.

---

## 10. Redis connection refused — not running

**Error message.** Logs show `redis.exceptions.ConnectionError: Error 61 connecting to localhost:6379. Connection refused.` During boot you see `prewarm: redis unavailable, falling back to L1`.

**Symptoms.** The app still serves requests (L1 in-process cache is sufficient for solo-dev) but warm-second-request latency is noticeably worse, and any feature relying on cross-worker shared state (rate limiters, distributed locks) degrades to per-worker behaviour.

**Three-step fix.**

1. **Start Redis.** Run `redis-server` in a separate terminal (Homebrew: `brew services start redis`; Linux: `sudo systemctl start redis`). Confirm with `redis-cli ping` → `PONG`.

2. **Confirm the URL.** Open the `.env` file and verify `REDIS_URL=redis://localhost:6379/0`. If you're using a non-default port (e.g. `6380` for a second instance) update accordingly. Restart the API process for the env change to take effect.

3. **Or remove `REDIS_URL` to opt-in to L1.** If you don't need Redis (solo-dev, no cross-worker concerns, willing to accept slower second-requests), simply unset or comment-out `REDIS_URL`. The lifespan prewarm will detect the absence and skip Redis entirely without logging an error.

---

## When this guide doesn't help

- **Check `docs/RUNBOOK.md`** for the deeper operational playbook (process trees, port maps, restart sequences).
- **Check the relevant ADR** in `docs/adrs/` — every architectural choice with operational implications has one.
- **Search `.coordination/issues.log`** for the same symptom; another agent may have hit it already.
- **Ask Damian.** Per CLAUDE.md, when in doubt do not guess on API or infra behaviour — surface the question with reproduction steps and the failing `health/deep` payload.
