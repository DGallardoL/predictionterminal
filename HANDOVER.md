# HANDOVER — Prediction Terminal

Snapshot date: **2026-05-16**
Author: Claude session (continuation of 2026-05-14 → 2026-05-16; jumps + sentiment + premium-CSS overhauls)

This is the runbook + recent-work map for picking up the project. Read it before opening Claude / making changes.

---

## 1. Quick-start (local dev)

You need **two processes**: gunicorn (API) on `:8000`, plain static server (UI) on `:8080`. Redis on `:6379` is expected.

```bash
# 0) Redis (Homebrew)
brew services start redis

# 1) Backend
cd /Users/damiangallardoloya/Desktop/proyectofuentes/api
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
PYTHONPATH=src REDIS_URL=redis://127.0.0.1:6379/0 \
PFM_FACTOR_PREWARM_ENABLED=1 PFM_FACTOR_PREWARM_TOP_N=500 PFM_FACTOR_PREWARM_CONCURRENCY=30 \
PFM_ARB_FALLBACK_MIN_SPREAD=0.5 PFM_ARB_FALLBACK_TOP_N=50 \
PFM_CRYPTO_WS_ENABLED=1 \
PFM_CRYPTO_5MIN_ENABLED=1 \
PFM_CRYPTO_CLOB_WS_ENABLED=1 \
PFM_ARB_ENGINE_AUTOSTART=1 PFM_ARB_ENGINE_MODE=og \
nohup .venv/bin/gunicorn pfm.main:app \
  --worker-class uvicorn.workers.UvicornWorker --workers 4 \
  --bind 127.0.0.1:8000 --log-level info \
  >> /tmp/pfm_gunicorn.log 2>&1 &
disown

# 2) Frontend
cd /Users/damiangallardoloya/Desktop/proyectofuentes/web
nohup python3 -m http.server 8080 >> /tmp/pfm_static.log 2>&1 &
disown

# 3) Open browser
open http://127.0.0.1:8080/
```

**Critical: `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`** — without this, gunicorn workers crash-loop on macOS due to an Objective-C fork-safety abort. The symptom is `Worker (pid:N) was sent SIGABRT` repeating in the log.

### Stop everything

```bash
pkill -TERM -f "gunicorn pfm.main" ; sleep 4 ; pkill -9 -f "gunicorn pfm.main"
pkill -9 -f "arb_engine.py"
pkill -9 -f "http.server 8080"
```

---

## 2. Current state (live now)

- `:8000` — gunicorn master + 4 UvicornWorkers (PID master visible via `pgrep -fl gunicorn`)
- `:8080` — python http.server serving `/web/`
- `:6379` — Redis
- `arbstuff/arb_engine.py --mode og` running as subprocess (1 instance after leader-election fix; pre-fix it spawned 4)
- Crypto WS engine: opt-in via `PFM_CRYPTO_WS_ENABLED=1` — currently ON

Probe state:
```bash
curl -s http://127.0.0.1:8000/health
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" http://127.0.0.1:8000/terminal/themes
curl -s http://127.0.0.1:8000/strategies/arb/state | python3 -c "import json,sys; d=json.load(sys.stdin); print('_source:', d['_source'], 'opps:', len(d.get('opportunities',[])))"
```

Expected: `{"status":"ok"}`, `200 ~0.005s`, `_source: engine opps: 100+`.

---

## 3. Three modes (product surface)

The single SPA at `/web/index.html` ships **3 modes** + the α Hub has **4 sub-sections** (not 6 — earlier docs were wrong).

| Mode | What it does | Key endpoints |
|---|---|---|
| **Terminal** (default landing) | Bloomberg-style data hub. Gainers / losers / most-active / resolving-soon table, theme heatmap, market detail (orderbook, prob fan, fair price, equity curve, news, correlations, countdown). | `/terminal/homepage`, `/terminal/quote/{slug}`, `/terminal/gdelt/{slug}`, `/terminal/orderbook`, `/terminal/trades`, `/terminal/prob-fan/{slug}` |
| **Regression** | OLS fit of stock returns on prediction-market factor Δlogits. Coef bars + 95% CI, factor-correlation heatmap, interactive `(?)` coef interpretation popovers, rolling β, bootstrap CI, Granger, Q-Q, permutation null, PCA. | `POST /fit`, `POST /fit/preview`, `/factors`, `/factors/rank` |
| **Strategies (α Hub)** | 4 sub-sections: **Top Alphas** (88 curated, 4 A_STRUCTURAL), **Calendar & Spreads**, **Cross-venue Arb** (Kalshi×Polymarket live dashboard), **Crypto Micro** (10-pair model-vs-market). Live Edge & Research planned but not built — data sources exist at `web/data/live_signals.json` + `docs/alpha-report-vN.md`. | `/alpha-hub/leaderboard`, `/alpha-hub/strategy/{pair_id}`, `/strategies/arb/state`, `/strategies/crypto/snapshot`, `/strategies/crypto/5min/compare` |

---

## 4. Architecture pointers

- **Backend**: FastAPI + statsmodels + httpx, `/api/src/pfm/` (312 modules). **297 OpenAPI paths** (verified `curl /openapi.json | jq '.paths | keys | length'`).
- **Main router**: `pfm/main.py` (lifespan + middleware), endpoint groups split into `regression_router.py`, `factors_router.py`, `strategies_router.py`, `strategies_arb_router.py`, `strategies_crypto_router.py`, `alpha_hub_router.py`, `terminal/__init__.py` + `terminal/<feature>.py`.
- **Caching**: 2-tier — in-process TTLCache + Redis L2 (SETNX locks for stampede protection). See `pfm/cache_utils.py`.
- **Frontend**: single 1.2 MB `web/index.html` (vanilla JS + Plotly CDN). External small scripts: `config.js` (API base detection), `cmdk.js` (command palette), `realtime-tickers.js`, `onboarding.js`.
- **Static data**: `web/data/alpha_strategies.json` (88 curated alphas; 13 flagged `data_quality_warning`), `web/data/live_signals.json`.
- **Arb engine sidecar**: `arbstuff/arb_engine.py` (separate git repo) writes `dashboard_state.json` continuously; FastAPI's `/strategies/arb/state` reads it (with 3-min staleness threshold → falls back to `arb_scanner.top_arbs()`).

### Cross-origin in dev
- Static :8080 → API :8000.
- `web/config.js` detects this (`localhost + :8080 → PFM_API_BASE = "http://localhost:8000"`).
- A `fetch` shim in `index.html` rewrites paths matching 18 backend prefixes (`/api/`, `/health`, `/fit`, `/factors`, `/alpha-hub/`, `/terminal/`, `/strategies/`, etc.). EventSource (SSE) bypasses fetch — three explicit callsites prepend `PFM_API_BASE` manually.
- Connection-status pill (bottom right) waits for `detectApi()` before pinging; shows "Connecting…" then "Connected" / "Degraded" / "Disconnected".

---

## 5. What changed recently (work log)

### 2026-05-14 evening — initial polish
- `/terminal/homepage` gainers/losers now filter near-resolved markets (`price >= 0.95 or <= 0.05`) so the panel shows actionable trades, not 0.9995 noise.
- Breaking news word-boundary regex (was matching "ai" inside "MacBook **Ai**r").
- Modal focus trap via MutationObserver; A11y fixes: `--ink-3/4` darkened to WCAG AA, `prefers-reduced-motion` gated, tap targets to 44px.
- Factor-correlation heatmap added to Regression result (Pearson r between factor probability series).
- Interactive `(?)` coef popovers showing β-in-bp, p-value stars, VIF flag, rolling-β stability, `factor_metadata.coverage_pct`.

### 2026-05-14 night — 7 parallel agents
- **Performance**: `/terminal/themes` cache (480ms → 1.6ms), `/strategies/arb/state` async, `/terminal/rss/headlines` parallel fanout.
- **Errors**: clearer 502s, 429 with `Retry-After`, `redis_mirror` status in admin POST responses.
- **A11y**: tier-pill contrast, focus-ring, plotly chart aria.
- **α Hub**: STALE badge for stale `live_signals.json`, A_GOLD/A_STRUCTURAL split, View backtest fallback.
- **Cross-venue Arb**: PnL fallback shape support, Kalshi NO-side bid mapping fix, Hide-arb toast.
- **Crypto Micro**: `engine_running` real (not hardcoded False), alt-pair spot buffer passive fill from `/snapshot`, click-through detail modal.
- **Terminal market detail**: news panel empty-state, volume "|Δp|" honest title, correlations n≥30, countdown YES/NO P&L.

### 2026-05-15 overnight — 6 more agents
- **GDELT cache**: 8.6s cold → 5ms warm, SETNX lock.
- **Alpha-hunter data**: 13 strategies flagged `data_quality_warning`; root cause fixed in `scripts/backfill_ah_sweeps.py` (returns `None` instead of coercing to 0 when n<30).
- **Sentiment scraper**: switched from `mode=artlist` (no `tone` field) to `mode=timelinetone` + graceful `degraded_mode`.
- **docs/quants.md**: added crypto5min model math (LaTeX, ~55 lines).
- **main.py**: 5 silent `except Exception: pass` → `logger.debug/warning`.
- **Frontend polish 13 items**: focus ring 0.45 alpha, resolution-date vertical line on main chart, color-only TS chart pattern, `Math.max(0,…)` bootstrap whisker guard, recent-fits LRU dropdown, permalink URL hash, fit/preview factor_metadata pills, α Hub modal concurrent race fix, crypto detail σ honest fallback.
- **CLAUDE.md**: α Hub "six sections" → "four sections" with planned-but-not-built note.

### 2026-05-15 morning — bug hunt + new features
- **Dev split 404 storm fixed** (3 layers):
  1. `web/config.js`: when localhost on :8080, `PFM_API_BASE` now points to `:8000` directly (was assuming nginx `/api` proxy).
  2. Fetch shim now rewrites 18 backend prefixes (was only `/api/`).
  3. Three EventSource callsites manually prepend `PFM_API_BASE` (EventSource bypasses fetch).
- **Connection status pill**: waits for `detectApi()` before first ping (was flashing "Degraded" briefly).
- **Arb engine autostart**: leader election via SETNX so only one of 4 workers spawns the subprocess (was spawning 4× the rate-limit on Kalshi/Polymarket). Engine now running with 142 opportunities found, top tradeable +67% CA-41 primary.
- **Venue badges (K/P) in Terminal**: new column in the homepage table — K green (Kalshi) / P blue (Polymarket), clickable → external market page; small ↓ icon → raw JSON from `gamma-api.polymarket.com` or `api.elections.kalshi.com`. Helpers (`_pfmVenueOf`, `_pfmVenueBadge`, `_pfmVenueUrl`, `_pfmVenueDataUrl`) on `window` for reuse in search/watchlist later.

### 2026-05-15 afternoon — arb-engine dedup + prod-hardening of main.py
- **Duplicate arb engines fixed** (root cause: 3 latent bugs in the leader-election hook). (1) Refresh used `EXPIRE` without verifying ownership → a successor's lock could be extended; replaced with a Lua `GET == token → PEXPIRE` CAS. (2) Lifespan teardown never `DEL`'d the leader key (unlike the crypto-WS shutdown), so a successor batch had to wait the full 60 s TTL — when SIGKILL skipped teardown entirely, the orphan engine subprocess survived AND the lock stayed, so the next batch's leader (after expiry) spawned a 2nd engine on top of the orphan. Teardown now CAS-`DEL`s. (3) Added a pre-spawn `pgrep -f arb_engine.py` reaper that SIGTERMs any orphans before SETNX, plus a subprocess-health tick that releases the lock when `proc.poll()` is not None — failover happens inside one refresh interval instead of after 60 s. Lock token is now `pid|boot_ts|nonce_64bit` so it's effectively unique across pid recycles. New env: `PFM_ARB_ENGINE_LEADER_TTL_S` (default 60). Also added `PFM_CORS_ORIGINS` (default `127.0.0.1:8080,localhost:8080`; legacy `CORS_ORIGINS` honoured), `PFM_SECURITY_HEADERS_ENABLED` (default 1; X-Frame-Options downgraded to SAMEORIGIN, HSTS removed since nginx handles it), `PFM_METRICS_ENABLED` (default 0; /metrics returns 404 when off — inside pytest it defaults ON to keep existing observability tests green) and `PFM_METRICS_TOKEN` (Bearer-gates /metrics when non-empty). All 2647 tests passing; one observability test verified end-to-end via TestClient.

### 2026-05-15 evening — UX wave 3 (max-effort polish)

- **Regression UI overhaul.** Verdict pill (STRONG / MIXED / WEAK) + headline summary on top of the result card; VIF and Clip% columns inlined in the coef table; click-to-drill on any coef row opens the rolling-β + diagnostics popover. (`web/index.html` regression result renderer.)
- **Regression preset gallery + smart factor picker.** Two new tabs in the factor picker (Presets, Smart picker) plus a Save/Share/Compare row on the result card (LRU dropdown of last 10 fits in localStorage; Share copies a permalink with the payload encoded in the URL hash).
- **Auto-prune collinear factors.** New `?prune_collinear=true` query param on `/fit` drops VIF>10 factors before refitting; pruned slugs returned in `pruned_factors`. Wired to a UI checkbox. (`api/src/pfm/regression_router.py`.)
- **Browser back in Terminal.** Market detail pushes `#mode=terminal&market=<slug>` hash URLs; `popstate` restores both mode and open market. Shareable links now work end-to-end. (`web/index.html` near the existing permalink hash logic.)
- **α Hub modal nav.** ← Prev / Next → buttons + keyboard `←` / `→` / `Esc` cycle the fullscreen tearsheet through the *currently filtered + sorted* card list; data unified through `GET /alpha-hub/leaderboard?full=true` so modal/grid/live-pills agree.
- **News relevance scoring.** Anchor terms + topic match with NFKD accent normalization; sub-0.18 hits dropped so the per-market news panel shows empty state instead of off-topic noise. (`api/src/pfm/terminal_news.py` or sibling.)
- **Loading / empty / error states.** New shared `_termPanelState(el, kind, opts)` helper; auto-retry on `/fit/preview` token-id race (one retry after 600ms, masked).
- **Mobile + a11y sweep.** 768px and 375px breakpoints validated; ARIA roles on every clickable row; 4-step onboarding tour with `?tour=1` replay (persists `pfm:tour:done=1`).
- **Crypto Micro stripped down.** Section reduced to model-vs-market only; ~1 s freshness via prewarmer; CLOB WS push wired when `PFM_CRYPTO_CLOB_WS_ENABLED=1`. (`api/src/pfm/crypto5min/`.)
- **Resilience pass.** 502/503/504 across all upstream callsites soft-fail to empty state with retry; 429 honours `Retry-After` and surfaces a "rate limited" pill; every external call now caches through Redis L2 with a STALE badge when served past TTL.
- **Docs refresh.** `docs/USER_GUIDE.md` (new §0 Tour & shortcuts, refreshed §3 Regression workflow, new α Hub modal nav block, new §11 Mobile & a11y, new §12 Resilience). `docs/DEMO_SCRIPT.md` updated in-place at minutes 0:30-2:00 (URL share trick), 5:30-7:00 (verdict pill + auto-prune), 7:00-9:00 (modal Next nav).

### 2026-05-16 — jumps + sentiment + premium CSS

- **`/terminal/jumps/{slug}`** — algorithmic jump detection on prediction-market price series, with multi-source news attribution and hybrid NLP sentiment per jump. Defaults tightened from 3pp/2.5σ → **5pp/3σ** (BTC slug: 39 → 10 jumps; signal-to-noise massively improved). Pre-market article floor (`_articles_for_jump_with_floor(market_start_ts)`) strictly drops news dated before the Polymarket market was created. (`api/src/pfm/terminal/jumps.py`, ~28 KB.)
- **`/terminal/jumps/{slug}/backtest`** — paper-PnL simulation on each jump, scoring **disagrees** (price moved without supporting news) vs **agrees** (price moved with corroborating news). Returns Sharpe, hit-rate, max drawdown, equity curve. (`api/src/pfm/terminal/jumps_backtest.py`, ~20 KB.)
- **`/terminal/jumps/cluster`** — union-find clustering of co-occurring jumps across markets within a time window, surfacing shared news terms as cluster labels. (`api/src/pfm/terminal/jumps_cluster.py`, ~21 KB.)
- **`/terminal/sentiment-leaderboard`** — top markets ranked by `disagrees_pct` (jumps without news support), highlighting potential alpha candidates. (`api/src/pfm/terminal/sentiment_leaderboard.py`, ~11 KB.)
- **`pfm/terminal/sentiment_nlp.py`** — hybrid VADER + financial-lexicon sentiment scorer with LRU cache (10k entries). Used by jumps endpoints to label article tone per jump. (~11 KB.)
- **`pfm/sources/sentiment_factor.py`** — new factor source `sentiment`. 10 curated queries shipped (`sentiment:fed-hawkish`, `sentiment:earnings-beat-tech`, etc.) plus free-form `sentiment:<query>` syntax accepted in `/fit`. (~24 KB.)
- **`/terminal/search?theme=…` fallback** — when factor catalog filter returns 0 results, falls back to gamma-API classifier (so `theme=equities` / `theme=awards` always return something). (`main.py:2082-2096`.)
- **Premium CSS overhauls.** `web/terminal-premium.css` rewritten end-to-end (2486 lines). New `web/alphahub-premium.css` (1003 lines) scoped to `[data-mode-pane=strategies]`. `web/plotly-theme.js` rewritten (327 → 678 lines) with `inPremiumScope()` hook so theme applies to both Terminal and α Hub. The `arblive-*` region inside `web/index.html` got an in-place premium rewrite (no new file — kept inline to avoid scope conflicts with concurrent UX-audit session).
- **Test coverage.** Jumps + sentiment area added ~150 new tests (113 → 133 → ~150 passing across `test_jumps.py`, `test_jumps_backtest.py`, `test_jumps_cluster.py`, `test_sentiment_nlp.py`, `test_sentiment_factor.py`).
- **Multi-session coordination.** `.coordination/PROTOCOL.md` written (rules for 5-way concurrent Claude work); `.coordination/active-edits.json` is the append-only registry of in-flight scopes.

---

## 6. Known issues / pending

| Severity | Item | Notes |
|---|---|---|
| LOW | Crypto micro click-through modal uses hardcoded σ fallback (0.65/0.95) when `/model-state` 503s | Already partially fixed (shows "Model state unavailable") but should read live comparator state |
| LOW | α Hub modal: rapid double-click race condition fixed via `_alphaHubFsToken` but onboarding STALE → no re-fetch button | One-click refresh would help |
| LOW | Live Edge / Research sub-tabs in α Hub never built | Data sources exist; planned-but-not-built per CLAUDE.md |
| LOW | Sentiment trend `tone_series` per-day still returns 0 even when aggregate `current_tone` is real | Agent's fix only populated aggregate via `timelinetone`. To fix, plumb timeline overlay into each point in `_build_tone_series` |
| LOW | Onboarding tour "Step 1 of 3" reappears in fresh browser profiles | Skip button DOES persist `pfm:tour:done=1`. Headless probes get fresh profile each time so it looks broken — real users only see once |
| INFO | Venue badges only show **P** on the homepage table | Backend `/terminal/homepage` only fetches Polymarket gamma. If you want mixed venues, edit `pfm/terminal/homepage.py` to merge Kalshi events |
| INFO | Some α Hub strategies show `oos_sharpe=9.47` with `full_sharpe=0.00` | These are now flagged `data_quality_warning` and demoted to D_RAW. Hidden by default filter (`B_VALIDATED`). Root cause fixed in `backfill_ah_sweeps.py` — re-run the sweep to regenerate the JSON |
| INFO | Frontend `index.html` is 1.26 MB unminified | Compresses ~5-8× over gzip via nginx in prod. In dev (python http.server) it's larger but acceptable. No build step exists |

### Recently closed (2026-05-16)

- ~~Jumps over-trigger noise~~ — thresholds tightened 3pp/2.5σ → **5pp/3σ**; pre-market article floor (`market_start_ts`) added so news dated before the market existed is dropped.
- ~~`/terminal/search?theme=…` returns 0 when factor catalog has no entries for that theme~~ — gamma-API classifier fallback wired in `main.py:2082-2096`.
- ~~Sentiment scoring scattered across callsites~~ — consolidated in `pfm/terminal/sentiment_nlp.py` (hybrid VADER + financial-lex, LRU-cached).
- ~~News attribution limited to GDELT~~ — jumps endpoints now fan out across multiple sources (GDELT + RSS + ad-hoc) with sentiment per article.
- ~~No way to expose sentiment as a regression factor~~ — `sentiment:<query>` slugs accepted in `/fit`; 10 curated queries shipped (see `pfm/sources/sentiment_factor.py`).

---

## 6.5. Multi-session coordination

The project is regularly worked by **up to 5 concurrent Claude Code sessions**. Race conditions on hot files (`web/index.html`, `web/config.js`, `api/src/pfm/main.py`, the shared gunicorn at `:8000`) are real — earlier nights we lost edits to overwrites.

**Read `.coordination/PROTOCOL.md` before any edit.** TL;DR:

- **Announce intent** by appending an entry to `.coordination/active-edits.json` (JSON array of `{session_id, files, scope, started_at, expires_at}`). Append-only — don't rewrite the file.
- **Check for conflicts** with `jq '[.[] | select(.expires_at > now)]' .coordination/active-edits.json` before opening a hot file. Overlapping scope on the same file → wait or pivot.
- **Don't restart gunicorn unilaterally.** Write to `.coordination/restart-requests.txt` and let the owner session batch.
- **Don't kill processes you didn't start.** Other sessions may have subprocesses running.
- **Backups** live at `/tmp/pfm-race-backup/snapshots/` — timestamped, used to restore after a bad overwrite.
- **Issues / broken state** → append to `.coordination/issues.log` (one line: `<iso_ts> <session_id> <symptom> <what_you_tried>`).

If you are a solo session, the protocol is cheap — one JSON append at start, one removal at end — and prevents the next overnight batch from clobbering your work.

---

## 7. Critical files & where to look

- `api/src/pfm/main.py` — lifespan, prewarm hooks, arb engine autostart (with new SETNX leader election ~L380-460), middleware stack (brotli outer, gzip inner)
- `api/src/pfm/terminal/homepage.py` — `/terminal/homepage` + `/terminal/themes` (themes uses shared `_HOME_CACHE` — perf fix from this week). Near-resolved filter at L555-580.
- `api/src/pfm/terminal/sentiment_trend.py` — `_fetch_gdelt_tone_timeline` is the new fix path (artlist mode lacks tone)
- `api/src/pfm/terminal/gdelt_news.py` — Redis SETNX cache (10min TTL)
- `api/src/pfm/strategies_arb_router.py` — async state handler + admin gating + `redis_mirror` status in blacklist responses
- `api/src/pfm/regression_router.py` — `n_obs_dropped` warning at ≥10% (new)
- `web/config.js` — API_BASE detection (just rewritten — localhost+:8080 → :8000)
- `web/index.html`:
  - L11428+ `syncDetectApiBase` and the extended fetch shim with 18 prefixes
  - L12295+ modal focus trap (MutationObserver-driven)
  - L23260+ homepage table renderer (with new venue badge column)
  - L13565+ `drawRegFactorCorrChart` — factor correlation heatmap
  - L8469 `reg-chart-factor-corr` slot
  - L2429+ `.coef-info-btn` (?) popover CSS
  - L5294+ `.pfm-venue-chip` styles (new)
- `arbstuff/arb_engine.py` — subprocess sidecar, writes `dashboard_state.json` every cycle (~30 min full scan of 1351 events)
- `web/data/alpha_strategies.json` — 88 curated alpha pairs, `data_quality_sanitized_pass: sanitize_alpha_strategies.py`

---

## 8. Things to NOT do

- **Don't run gunicorn from project root** — `.venv` is inside `api/`. `cd api/` first or set the absolute path.
- **Don't kill arb engine subproc without leader-election context**: the new code uses Redis SETNX. If you `pkill -9` the leader before the lock expires (60s), no other worker will start one until expiry. Either `flushall` Redis or wait.
- **Don't commit `web/data/alpha_strategies.json`** without re-running `scripts/sanitize_alpha_strategies.py` — flags will be lost.
- **Don't trust headless screenshots of "Step 1 of 3" tour** — each fresh chrome profile resets `pfm:tour:done`. Real users see it once.
- **Don't auto-commit** — Damian runs git.
- **Don't `--no-verify` on commits** unless asked.

---

## 9. Operational commands cheatsheet

```bash
# Tail backend logs
tail -f /tmp/pfm_gunicorn.log

# Tail arb engine
tail -f /Users/damiangallardoloya/Desktop/proyectofuentes/arbstuff/arb_engine.log

# Check live arb opportunities
curl -s http://127.0.0.1:8000/strategies/arb/state | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('_source:', d['_source'])
opps=[o for o in d.get('opportunities',[]) if (o.get('volume') or 0) > 100]
for o in sorted(opps, key=lambda x: -x.get('profit_pct',0))[:5]:
    print(f\"  {o['name'][:55]:<55} {o['profit_pct']:+.2f}%  vol={o['volume']:.0f}\")
"

# Check redis hit rate
redis-cli info stats | grep -E "keyspace_(hits|misses)"

# Restart cleanly
pkill -TERM -f "gunicorn pfm.main"; sleep 5; pkill -9 -f "gunicorn pfm.main"
# then re-run the start command in §1

# Clear all caches (forces cold rebuild on next request)
redis-cli flushall

# Inspect what API_BASE the browser detected
# (open DevTools console on the running page)
window.PFM_API_BASE
```

---

## 10. If something is broken

1. **Workers crash-loop SIGABRT on macOS** → `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` was missed in the env.
2. **Conn-status pill stuck on "Degraded"** → likely `PFM_API_BASE` wrong. Check `window.PFM_API_BASE` in DevTools. Should be `"http://127.0.0.1:8000"` for dev split.
3. **404 storm on `/api/health`** etc. → `config.js` heuristic failed. Force-set in DevTools: `window.PFM_API_BASE = "http://127.0.0.1:8000"` and refresh.
4. **`gunicorn: No such file or directory`** → you launched from project root, not `api/`. The venv is inside `api/`.
5. **`_strategies/arb/state` returns `_source: live_fallback`** even though engine env is set → engine didn't acquire the SETNX leader lock, or no engine running. Check `pgrep -fla arb_engine.py`. Clear lock with `redis-cli del pfm:arb_engine:leader`.
6. **Onboarding tour keeps appearing** → either fresh browser profile or `localStorage.removeItem('pfm:tour:done')` was called somewhere. Set `localStorage.setItem('pfm:tour:done', '1')` in DevTools to dismiss permanently.

---

## 11. The two memory rules that matter

From persistent `MEMORY.md`:
- **No OpenSpec** — user dislikes ceremony; go straight to code + tests.
- **α Hub is the product surface** — curated alpha cards are the primary view; don't pivot to research-workbench framing.
- **User invokes "max effort / agentes"** — dispatch many parallel sub-agents in one shot, don't serialize.
- **BTC latency arb is dead** — don't re-explore unless rolling-σ or orderbook-imbalance angle is added.
- **Wave-5 demoted A_GOLD strategies** — anti-alpha list in `CLAUDE.md` is load-bearing.
- **Favorites-bias is paper-only until 2026 Q3** — regime-driven not structural.

If a future agent re-pitches recession-defensive, crypto-ETF, senate-vol, or geopolitical-oil as "wins", it's a hallucination — they're on the anti-alpha list.
