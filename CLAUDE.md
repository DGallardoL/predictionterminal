# CLAUDE.md — Instructions for Claude Code

This document is your instruction manual for building and extending the `prediction-factor-model` project. The original POC has expanded substantially (see "Current state (post-wave-9)" below) but the foundational priorities still apply.

## Context

You are building a Proof of Concept of a web service that fits factor models of stock returns on prediction-market-derived factors. **Read `PLAN.md` first** — it contains the full specification, API docs, architecture, and build order.

## Priorities (in order)

1. **Satisfy the course requirements** (see §2 in PLAN.md). This is graded by a professor who values engineering discipline as much as (or more than) the quant content. CI must be green. Docker must work with `docker-compose up`. OpenAPI must be generated. ADRs must be genuine.

2. **Get a working end-to-end slice quickly.** Before polishing any single module, have a minimal version that can fetch data, fit a dumb OLS, and return a response. Then iterate.

3. **Code quality.** Type hints everywhere (Python 3.12 style: `list[str]`, not `List[str]`). Pydantic for all request/response schemas. Ruff-clean. Tests with ≥70% coverage on `model.py` and `attribution.py`.

4. **Honest quant work.** The model isn't rigorous research — it's a POC. That's fine. But what IS there must be correct: use `statsmodels` OLS with `cov_type='HAC'` and `cov_kwds={'maxlags': lag}`, don't roll your own. Clipping must be explicit and configurable. VIF must be reported.

## How to approach the build

**Follow the phased plan in §10 of PLAN.md.** Do NOT jump ahead. Specifically:

- **Phase 2 (core quant) before Phase 3 (data sources).** Test the math with synthetic data where you control the DGP. Only then wire it to real APIs. This is the single biggest productivity win.
- **Write tests alongside code, not after.** For `model.py`, first write a test that generates data with known betas, then write the function to recover them.
- **Mock external APIs in tests.** Never hit real Polymarket or yfinance from `pytest`. Use `respx` or `httpx_mock`.

## Critical technical details (don't forget)

- **Polymarket `clobTokenIds`** arrives as a **JSON string inside the JSON response**. You must `json.loads()` it twice effectively. Example: the market response has `"clobTokenIds": "[\"123...\", \"456...\"]"`.
- **Always use `fidelity=1440`** (daily) for `/prices-history`. Sub-daily fails for resolved markets (see PLAN.md §5.3).
- **Rate limit is generous** (1000/10s), but cache aggressively anyway because users will re-run fits with the same factors often.
- **Timezone alignment is a trap.** Normalize everything to UTC date at the `pandas.Timestamp(date).normalize()` level. Both Polymarket timestamps (unix seconds) and yfinance closes must align to the same UTC calendar date. Document this in ADR-0006.
- **Log returns, not simple returns.** `r_t = log(P_t / P_{t-1})`. Add that to the quant docs.
- **The clipping epsilon is important.** Default `ε=0.01`. If a contract trades at 0.005 and then 0.002, clipping to 0.01 means Δlogit is 0 — the user should see this happening and be able to change ε. Expose it via a query parameter on `/fit`.
- **Hybrid NLP sentiment** (`pfm/terminal/sentiment_nlp.py`): VADER blended with a financial-specific lexicon (earnings/Fed/credit terms get explicit weights). LRU-cached (10k entries) because the same headlines reappear across jumps, leaderboards, and the `sentiment:` factor source. Don't roll your own scorer in a new module — import `score_text()` from here. VADER alone over-scores generic positive words ("growth", "rally"); the financial-lex overlay is what makes the score useful for prediction-market context.
- **Multi-session coordination.** Before editing hot files (`web/index.html`, `web/config.js`, `api/src/pfm/main.py`) read `.coordination/PROTOCOL.md` and append your scope to `.coordination/active-edits.json`. Up to 5 concurrent Claude Code sessions run on this repo; race conditions on these files have caused lost edits. Don't restart the shared uvicorn at `:8000` without writing to `.coordination/restart-requests.txt`.

## File conventions

- **Line length:** 100 chars (ruff default is 88; override in pyproject.toml to 100)
- **Imports:** absolute, sorted with isort rules (ruff does this)
- **Docstrings:** Google style for public functions
- **Error handling:** raise `HTTPException` with meaningful detail in API layer; domain layer raises plain exceptions
- **Logging:** `structlog` if you want, but `logging` stdlib is also fine. Don't `print()`.

## Current state (post-2026-05-14 night session)

The project has been rebranded **"Prediction Terminal"** in the UI (was "Prediction Factor Model"). Three coexisting modes share the same FastAPI app:

### Three modes

1. **Regression** (original) — factor-model fits of stock returns on prediction-market factors. `/fit`, `/factors`, `/health`.
   - **WOW Hero**: live curated A-tier alpha card (top of `/alpha-hub/leaderboard`) + auto-attribution prompt ("Why is NVDA moving today?") that streams the top contributors via `POST /reverse-finder/stream` (SSE). With Redis prewarm of 200 curated factors at startup, warm queries finish in ~3 s.
2. **Strategies (α Hub)** — four sections now: **Top Alphas** (rainbow tier-pill cards over `web/data/alpha_strategies.json`), **Calendar & Spreads**, **Cross-venue Arb** (now a **full Bloomberg-style multi-tab live monitor** — see below), **Crypto Micro** (live 10-pair Binance snapshot + 9-signal taxonomy via `pfm/strategies_crypto_router.py`).
   - Clicking any α Hub card opens a **fullscreen modal** that fetches `/alpha-hub/strategy/{pair_id}` — backend now embeds spread series (90 points), trade rule (entry/exit/stop z), risk profile, deployment params, recent live signal, theory reference, equity curve.
   - **Live Edge** (built — `web/index.html:11482-12215` and `_renderLiveSignalsFeed()` at line ~20488) consumes `web/data/live_signals.json` and shows real-time z-score signals. The α Hub strategy fullscreen modal also has an on-demand "↻ Refresh live signal" button that calls `GET /alpha-hub/strategy/{pair_id}/live-signal` for per-pair recompute (Kelly-sized, bankroll-aware, persists `pfm_alphahub_bankroll_usd` to localStorage).
   - **Research** (built — `_renderResearchReports()` at line ~20854) lists the versioned `docs/alpha-reports/alpha-report-vN.md` series. **v22 is current** (Wave-7 4Q-stability reckoning, 2026-05-19) — found 0 of 5 Wave-6 A_STRUCTURAL promotions clear the strict 4-quarter Sharpe-stability gate.
3. **Terminal (Bloomberg-style data hub)** — read-only market data, news, peer comparisons, portfolio simulation, volatility surfaces, quality scoring. **58 endpoints**. Replay + Archive modes were removed from the top nav and now live as small pills at the bottom of the Terminal pane. Market detail has a **back button** to return to overview without page reload.

**Cross-venue arb · live dashboard** (`pfm/strategies_arb_router.py`, 12 endpoints under `/strategies/arb/*`)
The Strategies → Cross-venue Arb tab is now a full Bloomberg-style live monitor mirroring the structure of `arbstuff-full/dashboard/` (the standalone React app) but rendered in the site's own design tokens. **Status bar** with pulsing dot, **5-metric strip** (Active Arbs · Best Profit · Volume · Scans · Engine source), **6 sub-tabs** (Opportunities · Scan Log · History · PnL · Markets · Config), and a **slide-in settings drawer**. The Opportunities tab has a 3-fr/2-fr split: sortable list on the left, detail pane on the right with price bars, fee breakdown table, side-by-side **live Kalshi+Polymarket orderbook** (fetched on selection), and a "Hide arb" button that POSTs to the blacklist. Live data comes from a **2 s SSE stream** at `GET /strategies/arb/stream`. When `arbstuff/dashboard_state.json` is missing or stale (>3 min), the router falls back to `pfm.arb_scanner.top_arbs()` so the panel always shows real opportunities — no need to run the separate `arb_engine.py`. Optional engine autostart via `PFM_ARB_ENGINE_AUTOSTART=1`. 17 new tests in `test_strategies_arb_router.py`.

**Crypto micro · model vs market** (`pfm/crypto5min/`, 3 endpoints under `/strategies/crypto/5min/*`)
A new section inside Strategies → Crypto Micro that pairs Polymarket BTC/ETH up-down 5m & 15m markets with our GBM-plus-microstructure model probability. The model blends a 30-day daily-close σ (long anchor) with cryptostuff's tick-derived σ_short (variance-weighted), and adds an OFI-derived drift bias capped at ±30%/yr plus a smaller whale-flow term. When |z_vwap|>2 the drift is shrunk toward 0 and a mean-reversion pull replaces it. A background sampler (opt-in via `PFM_CRYPTO_5MIN_ENABLED=1`) keeps a rolling spot buffer so we always have `spot_at_window_start` to compute `log(spot_t / spot_0)` — even without the WS engine running. 113 new tests cover synthetic-DGP recovery, Monte-Carlo calibration, mocked Polymarket discovery + CLOB, FastAPI TestClient flows, Kelly sizing edge cases, and the background sampler lifecycle.

### Scale (last verified 2026-05-19, post-v22 reckoning + auto-backtest 500 fix)

- **1,260 factors** loaded (verified via `GET /factors?limit=1`; 1,250 yaml entries + 10 curated sentiment auto-injected at lifespan). Factor catalog hot-reconciled with `web/data/alpha_strategies.json` — 22 previously-orphan slugs (`bp_acquired_before_2027`, `china_taiwan_before_gtavi`, `eurovision_winner_2026`, …) now resolve cleanly. Free-form `sentiment:<query>` accepted on `/fit`; set `PFM_SUPPRESS_CURATED_SENTIMENT=1` to skip the 10-factor injection (the test conftest does this).
- **69 curated alpha strategies** in `web/data/alpha_strategies.json` (was 88 pre-error-purge 2026-05-19 11:23). Tier breakdown post-v22 4Q reckoning: **A_STRUCTURAL=0, B_VALIDATED=27, C_TENTATIVE=13, D_RAW=29**. All 5 prior Wave-6 A_STRUCTURAL promotions reverted because none has `joint_days ≥ 360` (4 disjoint quarters required). `renan_santos / us_aliens` marked `B_VALIDATED++` — closest to passing on lenient 3-of-3 reading.
- **315 OpenAPI paths** (verified via `curl /openapi.json | jq '.paths | keys | length'`). Group breakdown:
  - `/terminal/*` — 76 (incl. `/terminal/implied-pdf/*` — SPX/NDX risk-neutral density from Kalshi ladders; and `/terminal/pricing-kernel/*` — cross-venue Kalshi-Q vs options-Q risk-neutral density (second derivative of the call-price curve, shape-restricted) + empirical pricing kernel M(S)=e^{-rτ}f_Q/f_P and implied risk aversion. Modules: `pfm/vol/{options_rn,physical_density,pricing_kernel,pricing_kernel_router}.py`. Both surface as cards in the Terminal → Tools panel.)
  - `/strategies/*` — 65 (was 61 — recent additions)
  - `/archive/*` — 12
  - `/factors/*` — 10
  - `/arb/*` — 9
  - `/alpha-hub/*` — 8 (includes new `/alpha-hub/strategy/{pair_id}/live-signal`)
  - `/macro/*` — 8
  - `/auth/*`, `/alpha/*`, `/news/*`, `/embed/*`, `/replay/*` — 7 each
  - `/alerts/*`, `/advanced-model/*` — 6 each
  - remainder: `/quant`, `/event-model`, `/indices`, `/multi-event`, `/lab`, `/signals`, `/ops`, `/health`, etc.
- **5,668 tests** collected (2026-05-19; was 5,635 pre-session). +33 new tests this session: 16 endpoint live-signal, 4 promotion gate (`JOINT_DAYS_4Q_GATE = 360`), 1 cointegration regression (`test_constant_leg_returns_insufficient_variation_not_indexerror` — the auto-backtest 500 fix), tier-filter test split, snapshot regenerated. 0 lint errors (ruff). 5 skips are `playwright`/`python-multipart`-optional. Slow tests (`-m slow`) opt-in.
- **312 Python modules** in `src/pfm/`. The 5 largest: `main.py` (2701), `strategies_router.py` (2318), `regression_router.py` (2178), `arb_scanner.py` (1450), `strategies_arb_router.py` (1341).
- Wave-1 through Wave-13 + launch-readiness + Wave-7 v22 4Q reckoning (2026-05-19).

> **Live calc (2026-05-19):** `GET /alpha-hub/strategy/{pair_id}/live-signal` wraps `_compute_signal_for_alpha()` for on-demand recompute (no waiting for hourly batch). Returns current z, action, Kelly fraction (capped at quarter-Kelly), edge in bps, recommended USD size, and `data_source` (live/cached_batch/stale_fallback). UI exposes a "↻ Refresh live signal" button + bankroll input on every strategy modal — bankroll persists to `localStorage` (`pfm_alphahub_bankroll_usd`).

> **Maintenance note**: when scale numbers change materially, update this section. Cifras stale → next-Claude takes bad decisions. The α Hub strategy fullscreen modal (Live Edge sub-tab, Research sub-tab, Reports tab with v22 marker) is the UI source of truth — keep docs in sync with what the UI actually renders, not the other way around.

> **Maintenance note**: when scale numbers change materially, update this section. Cifras stale → next-Claude takes bad decisions.

### Validated alphas (deployable, with caveats)

These passed 4-quarter robustness, OOS holdout, and transaction-cost sensitivity. **Always re-run robustness before claiming a new strategy is deployable.** See `docs/alpha-reports/alpha-report-v22.md` for the 2026-05-19 honest re-test that produced the current caveats. **Note on Wave-6:** the five A_STRUCTURAL promotions documented in v21 are **revertible** — 0 of 5 cleared the strict 4-quarter Sharpe-stability gate because none has > 174 days of joint Polymarket history. Treat as B_VALIDATED until Aug 2026 at earliest.

- **Election-binary momentum** — long the leading binary contract on resolution-decay; capacity-limited (~$50k notional). Caveat: only validated on **long-dated `_out_2027` cross-sections** (`putin_out_2027 / xi_out_2027` is the one pair that passes a lenient 3-of-3-valid-quarters reading per v22 §3.1); other pairs flip sign or fail. Only works in elections with ≥3 months to resolution.
- **Fed-decision straddle proxy** `PENDING_4Q` — VIX-overlay using Polymarket FOMC odds vs implied move. **Signal is real where data exists (Sharpe 2.6–4.1) but no pair has 4 quarters of joint history** because Polymarket FOMC strike contracts were minted ~Jan 2026. Caveat: degrades when realized vol < 12. Re-test monthly; expected to clear the gate by Oct 2026 if liquidity persists.
- **Earnings-surprise odds vs IV** — **moved to future-work / aspirational.** Zero matching factors (`earnings`/`beats_eps`/`eps_surprise`) in the 1,260-factor catalog as of 2026-05-19. Revisit only when Polymarket lists liquid quarterly-EPS binaries on ≥ 6 large-cap names; track in `docs/future-work.md`.

### Anti-alphas (DO NOT redeploy)

These looked promising in a single quarter but failed regime-robustness. Future Claude must NOT re-pitch these as wins:

- **Recession-odds → defensive-sector long.** Worked Q4-2024; reversed sign in Q1-2025. Pure regime trade.
- **Crypto-ETF approval drift.** One-time event, no repeatable signal. Backtest is a survivorship illusion.
- **Senate-control short-vol.** Dominated by a single 2024 episode; OOS Sharpe < 0.2.
- **Geopolitical-conflict oil long.** Direction-correct but transaction costs eat ≥110% of gross PnL.
- **Sports mean-reversion in NBA-finals same-game contracts.** v22 4Q test (2026-05-19) found a clean sign-flip on both tested pairs: Cavaliers/Wolves went +1.0, +1.7, **-2.2** across Q3-25/Q4-25/Q1-26; Spurs/Pistons went +3.5, **-1.9**, **-1.5**. Single-regime trade — works in summer/early-fall when contracts are far from resolution, breaks down once playoffs approach. The "final-hour overreaction" framing was never instantiated on the binaries this project actually carries. See `docs/alpha-reports/alpha-report-v22.md` §3.3.

## Recent file additions (Wave-2 through Wave-9)

### Terminal core and feature modules (`pfm/terminal_*`)

- `pfm.terminal` — core router, shared response envelopes, caching layer
- `pfm.terminal_peer_scanner` — comparable-contract discovery and ranking
- `pfm.terminal_portfolio_sim` — multi-leg portfolio simulator with PnL attribution
- `pfm.terminal_vol_distribution` — implied-vol surface from prediction-market odds
- `pfm.terminal_quality_score` — composite contract quality (liquidity × resolution × spread)
- `pfm.terminal_news` — news ingestion and tagging to factors
- `pfm.terminal_trades` — recent-trades tape and aggregation
- `pfm.terminal_orderbook`, `pfm.terminal_resolution`, `pfm.terminal_calendar`, `pfm.terminal_alerts`, `pfm.terminal_screener`, `pfm.terminal_correlations`, `pfm.terminal_history` — additional data-hub modules (14+ in total)

### Frontend

- `web/plotly-theme.js` — shared Plotly theme used across Regression / Strategies / Terminal panels

### Documentation

- `docs/USER_GUIDE.md` — end-user walkthrough of all three modes
- `docs/alpha-report-v15.md`, `docs/alpha-report-v16.md`, `docs/alpha-report-v17.md` — versioned alpha-deployability reports (v17 is current)

## How to extend

### Add a new alpha strategy

1. Create `pfm/strategies/<name>.py` implementing the `Strategy` protocol (`signal()`, `position()`, `pnl()`).
2. Register it in `pfm/strategies/registry.py`.
3. Add a synthetic-DGP test in `tests/strategies/test_<name>.py` that recovers a known signal-to-PnL relationship.
4. Run the 4-quarter robustness harness (`scripts/robustness_check.py`). **If any quarter has Sharpe < 0.5 OR sign-flip vs full-sample, do NOT mark deployable.**
5. Add an entry to the next `docs/alpha-report-vN.md`. Bump N; do not edit older reports.

### Add a new Terminal feature

1. Create `pfm/terminal_<feature>.py` with a router and Pydantic response models.
2. Mount the router in `pfm/terminal/__init__.py`.
3. Add a golden-file test under `tests/terminal/test_<feature>.py` (mock the upstream API, snapshot the response).
4. Add a frontend panel in `web/terminal/` reusing `plotly-theme.js`.
5. Update the endpoint count in this file's "Current state" section.

### Expand the factor catalog (Wave N pattern)

1. Branch a new wave: `wave-N-<theme>` (e.g. `wave-10-commodities`).
2. Add slugs to `factors.yml` under a new section. Keep counts in this file in sync.
3. Run `scripts/validate_factors.py` to confirm each slug resolves and has ≥30 daily observations.
4. Add no-network tests using cached fixtures from `tests/fixtures/factors/`.
5. Bump the totals in the "Scale" subsection above.

## What not to do

- Don't use async for things that don't need it (the OLS fit is sync; no point wrapping it)
- Don't add features not in PLAN.md §3 (scope) **unless** they fit the Strategies or Terminal modes already shipped. If tempted, write a note in `docs/future-work.md` and move on.
- Don't skip the ADRs. They're a grading criterion. 6–7 genuine ADRs of 1 page each.
- Don't auto-commit or push. Damian runs git commands.
- Don't hardcode real Polymarket slugs that might be resolved/gone by demo time. Keep `factors.yml` minimal where possible and let Damian update with live slugs when he's ready to demo.
- Don't build a beautiful React frontend. Plain HTML + Plotly from CDN is the target.
- **Don't deploy regime-driven alphas without a 4-quarter robustness check.** Every "wow" backtest from a single window must be cross-validated against ≥4 disjoint quarters. If sign flips or Sharpe collapses in any quarter, it goes on the anti-alpha list, not the deployable list.

## Verification checklist before handing off

Before declaring "done," verify:

- [ ] `docker-compose up` starts all three services and they pass healthchecks
- [ ] `curl http://localhost:8000/health` returns `{"status":"ok",...}`
- [ ] `curl http://localhost:8000/factors` returns the factor list
- [ ] Opening `http://localhost:8080` shows the frontend form (Regression / Strategies / Terminal tabs)
- [ ] `curl http://localhost:8000/openapi.json | jq '.paths | keys | length'` returns ≥271 endpoints
- [ ] `pytest` passes (5,635 tests) with coverage report
- [ ] `ruff check .` is clean
- [ ] `.github/workflows/ci.yml` has all three jobs and looks correct
- [ ] README has: badges section, quickstart, example curls, link to docs/
- [ ] All 6–7 ADRs exist and are non-trivial (≥150 words each)
- [ ] `docs/quants.md` has the full math with LaTeX
- [ ] `docs/USER_GUIDE.md` and the latest `docs/alpha-report-vN.md` are current
- [ ] `factors.yml` totals match the "Scale" subsection (1,260 factors)

## When in doubt

Ask Damian. Don't guess on API behavior — if you're unsure, use a mock and leave a `# TODO: verify with live call` comment.

Good luck. The goal is a **clean, professional product that demos well in 15 minutes** across all three modes and gives Damian a foundation to extend.
