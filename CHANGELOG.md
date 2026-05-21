# Changelog

All notable changes to **Prediction Terminal** (formerly *prediction-factor-model*).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Dates use ISO 8601 (`YYYY-MM-DD`). Newest entries on top.

Cross-reference index:

- ADRs live in [`docs/adrs/`](docs/adrs/) (0001–0009 are the foundation ADRs;
  0010–0018 cover the Wave-10/11/12 hardening — renamed from `ADR-NNNN-…`
  to lowercase `NNNN-…` on 2026-05-19 to remove an 0007/0008/0009 collision
  with the foundation set).
- Versioned alpha reports live in [`docs/alpha-reports/`](docs/alpha-reports/)
  (`alpha-report-v2.md` … `alpha-report-v22.md`); v22 is current.
- Coordination protocol: [`.coordination/PROTOCOL-V2.md`](.coordination/PROTOCOL-V2.md).
- Overnight recap: [`.coordination/OVERNIGHT-RECAP.md`](.coordination/OVERNIGHT-RECAP.md).

---

## [Unreleased] — 2026-05-19

Wave-7 v22 reckoning session: drove `alpha_strategies.json` back into
consistency with `docs/alpha-reports/alpha-report-v22.md`, hard-failed the
promotion script's 4Q gate, and added an on-demand live-signal endpoint so
strategies can be *computed* live (not just shown as static cards).

### Added

- **`GET /alpha-hub/strategy/{pair_id}/live-signal`** — on-demand live signal
  for one pair. Query params: `bankroll_usd` (default 10000), `force_refresh`
  (default false), `kelly_cap` (default 0.25 = quarter-Kelly). Returns
  `current_z`, `action`, `kelly_fraction`, `edge_bps`, `recommended_size_usd`,
  `decay_status`, `data_source` (`live` / `cached_batch` / `stale_fallback`)
  + warnings. Reuses `_compute_signal_for_alpha()` from
  `pfm.live_signals_job` so behaviour matches the hourly batch exactly.
  16 new tests in `api/tests/test_alpha_hub_live_signal.py`.
- **Refresh-live-signal button** in the α Hub fullscreen modal
  (`web/index.html` lines ~22897–23177). Bankroll input persists to
  `localStorage` (`pfm_alphahub_bankroll_usd`). Auto-fires once on modal open;
  manual refresh thereafter. Action color-coded (green LONG, red SHORT,
  gray HOLD, orange STOP). Inline SVG sparkline of recent z-history when
  available.
- **Module-level constant `JOINT_DAYS_4Q_GATE = 360`** in
  `api/src/pfm/alpha_tier_regen.py:78`, with the gate enforced in
  `_final_tier()` before any of strike-family / BH-q05 / BH-q10 / OOS-Sharpe
  branches run. This catches the structural issue at the data-availability
  layer — pairs without 4 disjoint quarters of joint Polymarket history can
  no longer slip to A/B tier via statistical gates alone. 4 new tests
  pin the boundary (`n_obs=359` → C_TENTATIVE, `n_obs=360` → A/B).

### Changed

- **`web/data/alpha_strategies.json`** — re-tiered per v22 §1 table.
  - A_STRUCTURAL: 5 → **0** (all reverted; none had `joint_days ≥ 360`)
  - B_VALIDATED: 22 → **27** (+5 demotions)
  - C_TENTATIVE: 13 (unchanged)
  - D_RAW: 29 (unchanged)
  - Each demoted record carries `tier_before_v22`, `tier_change_reason`
    (with the quarterly-Sharpe pattern from v22 §2), `v22_lenient_pass`,
    `v22_marker` (`B_VALIDATED++` for `renan_santos / us_aliens` —
    the only pair that scored 3/3 valid quarters with no sign-flip).
  - File metadata: `pipeline_version` bumped to include
    `wave-7 v22 4Q reckoning (2026-05-19)`, `generated: 2026-05-19`,
    `v22_reckoning_applied: true`. Backup retained at
    `web/data/alpha_strategies.json.bak-pre-v22-demotion-20260519`.
- **`api/tests/test_alpha_hub_leaderboard.py`** — split the legacy
  `test_leaderboard_tier_filter_a_structural` into:
  - `test_leaderboard_tier_filter_b_validated` (positive case on the new
    top deployable tier),
  - `test_leaderboard_tier_filter_a_structural_post_v22` (asserts the
    A_STRUCTURAL filter still returns 200 with `total == 0`, not 5xx).
  - `test_leaderboard_combined_filters` switched from A_STRUCTURAL to
    B_VALIDATED so it still exercises the combined-filter code path.
- **`api/tests/test_regenerate_alpha_tiers.py`** — bumped the synthetic
  `test_recovers_real_signals_from_5_pair_synthetic` series from `n=260`
  → `n=400` so cointegrated pairs still clear the new 360-day gate
  (otherwise the existing `tier >= B_VALIDATED` assertion would be
  impossible by design).

### Operational

- **Future-work entries** (`docs/future-work.md`): earnings-surprise
  aspirational alpha, monthly re-test calendar for Wave-6 pairs
  (2026-06-19 / 07-19 / 08-19), sub-quarter Sharpe stability stretch goal.

---

## [Unreleased] — 2026-05-17

Launch-readiness session: drove the test suite to **5,635 green / 0 failed / 0
lint errors**, reactivated 35 SSE + WebSocket tests that had been
``skipif``'d under Python 3.14, and reconciled factor catalog drift end-to-end.

### Fixed

- **`reset_caches()` orphaning module-level singletons** — the prior
  implementation cleared the singleton index, so module-level captures
  (``_FOO_CACHE = get_cache("foo")``) kept writing to a now-unreferenced
  instance while ``get_cache("foo")`` returned a fresh empty cache. This
  produced reproducible cross-test state-bleed in ~13 terminal tests.
  Fixed to clear contents without dropping identity (`pfm/cache_utils.py`).
- **Mode switch (Regression / Strategies) rendered blank** — ``mode-router.js``
  applies ``hidden=""`` on inactive panes at boot; the inline click handler
  in ``web/index.html`` only toggled ``.active`` and never cleared the
  attribute, so the HTML ``hidden`` flag continued to win over CSS
  ``display: block``. Handler now manages both.
- **"50 CONTRACTS" eyebrow** — Terminal hero counter called ``/factors``
  (default ``limit=50``) and used ``factors.length``. Switched to
  ``/factors/all`` + ``body.total`` so the count reflects the real
  catalogue (1,260 today).
- **Stale onboarding stats** — onboarding cover stats and hero text bumped
  from 1,228 → 1,250+ and 266 → 297 endpoints.
- **`scripts/aggregate_alpha_history.py`** — iterated ``(headers, cells)``
  table rows as if a single list; now unpacks properly and feeds
  ``_parse_row_sharpe`` / ``_parse_row_allocation``.
- **Polymarket fixture: missing ``clobTokenIds``** — 7 fixtures in
  ``test_terminal_news_relevance.py`` updated so ``poly.get_market_metadata``
  no longer falls into the degraded-payload branch.
- **Bid/ask inversion in `test_terminal_live_stream.py`** — live-verified
  CLOB convention is ``side=BUY → best bid``; mocks now match.
- **Hypothesis flake in `test_logit_strictly_monotonic_in_open_interval`** —
  ``assume`` now requires ``(b - a) > 1e-9`` so logit's monotonicity is
  observable in float64 (was failing on sub-ULP distances).
- **`A_STRUCTURAL` tier missing from deployable allowlist** in
  ``test_scenario_integration.py``.
- **22 orphan factor IDs renamed** — ``orphan_will_bp_be_acquired_before_2027_549``
  → ``bp_acquired_before_2027`` etc., with proper names and descriptions.

### Added

- **`live_server_factory` fixture** (`tests/conftest.py`) — boots uvicorn
  on a free port in a daemon thread, returns a base URL. Used by the SSE
  + WebSocket tests that can't go through `httpx.ASGITransport` under
  Python 3.14.
- **`_LiveWSClient` adapter** in `tests/test_ws_live.py` — small synchronous
  facade over `websockets` that mirrors the slice of `TestClient.websocket_connect`
  the tests use.
- **`PFM_SUPPRESS_CURATED_SENTIMENT=1`** — lifespan-level skip for the
  +10 curated-sentiment factor injection; the conftest autouse fixture sets
  this so tests with a 2-factor fixture catalogue don't see drift.
- **Autouse `_disable_background_prewarms`** in conftest — sets every
  ``PFM_*_PREWARM_ENABLED`` / ``PFM_*_AUTOSTART`` env var to "0" so the
  TestClient lifespan can't kick off real network fan-out.
- **22 reconciled factor entries** in `factors.yml` covering live alpha
  strategies that had been pointing at non-existent slugs (acquisition
  M&A tail, geopolitics, commodities, calendar-strike variants).

### Changed

- **Ruff config**: gracefully ignored pathlib stylistic preferences
  (`PTH100/105/108/110/118/120/123`), `SIM105` (try/except/pass),
  `E741` (single-letter math var names), `N806` (uppercase quant
  notation). All 363+ auto-fixable issues applied. Per-file overrides
  for ``main.py``, ``sources/*.py`` (lazy imports), tests (``B017``,
  ``PLW0127``, etc.).
- **Bumped scale notes**: 1,250 yaml factors (+22 orphans recovered),
  297 OpenAPI paths, 5,635 tests.
- **`.gitignore`**: added `*.rdb` / `dump.rdb` so local Redis snapshots
  don't pollute commits.

---

## [Unreleased] — 2026-05-16

Wave-11 and Wave-12 (two 60-agent bursts on 2026-05-16) consolidated everything
built in Wave-10 into a single coherent product surface, added a binary
prediction-market pricing pipeline end-to-end (theory → models → empirical
calibration → strategy), and shipped seven new ADRs that document the
non-obvious operational decisions.

### Added (Wave 11–12)

- **13 ADRs total** — `docs/adrs/0001-use-fastapi.md` through `0009-frontend-vanilla-html.md`
  (foundation, unchanged) **plus seven new operational ADRs**:
  - `ADR-0007-multi-session-coordination.md` — append-only claim ledger for up
    to 60 concurrent Claude Code agents.
  - `ADR-0008-cache-tiering.md` — L1 in-process + L2 Redis + lifespan-prewarm,
    rationale + invalidation contract.
  - `ADR-0009-arb-match-quality.md` — half-open `ResolutionWindow` semantics
    that fixed the "Trump 2024 vs 2028" false-positive class (see T76b below).
  - `ADR-0010-anti-alpha-rule.md` — the 4-quarter Sharpe-stability gate; an
    alpha is *demoted* (not deleted) when any quarter sign-flips.
  - `ADR-0011-cache-stampede-singleflight.md` — per-key locks on hot
    `/terminal/jumps/{slug}` and `/arb/auto-discover` paths.
  - `ADR-0012-rate-limit-retry.md` — token-bucket + exponential backoff for
    Polymarket / Kalshi / GDELT.
  - `ADR-0013-pickle-versioning.md` — magic-byte `PFMTC1\x00` envelope on
    TTLCache values so legacy entries fail-soft and migrate on next set.
- **~30 new endpoints** mounted under existing namespaces:
  - `GET /alerts/digest?since=24h` (T28) — rolled-up multi-channel alerts.
  - `GET /news/search?q=&since=&factors=true` (T32) — semantic news search with
    factor cross-links.
  - `GET /strategies/anti-alpha-list` + `/strategies/deployable-list` (W11-23/24)
    — symmetric, both backed by `web/data/alpha_strategies.json`.
  - `GET /pricing/binary/{slug}?model=logit|bsd|brownian|beta` (W11-25) — four
    binary-pricing models exposed via HTTP (logit / Black-Scholes-digital /
    Brownian-bridge / Beta-binomial).
  - `GET /arb/quality-audit` (W11-26) — live confusion-matrix run of the T76+T77
    matcher against current arb pairs.
  - `GET /regression/methods` (W11-27) + `?method=enet|quantile|bayes` query on
    `POST /fit` (W11-28) — three new estimators on the canonical endpoint.
  - `GET /health/deep` (T26) — concurrent upstream pings (Polymarket / Kalshi /
    yfinance) with per-source latency + last-error.
  - `GET /ops/sessions` + `GET /ops/config` (T27) — env flags, cache stats,
    Redis status; used by the connection pill.
  - `GET /factors/{slug}/related` (T29) — top-10 correlated factors with
    rolling-30d ρ.
  - `GET /terminal/jumps/compare?slugs=a,b,c` (T30) — aligned jump timelines.
  - `GET /research/reports` (T31) — `docs/alpha-report-vN.md` parsed as JSON
    cards; backs the planned Research sub-tab.
  - `POST /portfolio/import` (T33) — CSV (ticker,shares,cost) → simulation
    handle.
- **4 new strategies** registered via `pfm/strategies/registry.py`:
  - `binary-pricing-mispricing` (T84) — long/short the residual between model
    fair price and market price, Kelly-sized with a cap, gated on Sharpe>0.5
    in each of 4 quarters.
  - `calendar-lambda-ratio` (T55) — materialized the structural survivor of the
    Wave-5 stress test (see `docs/alpha-reports/alpha-report-v18.md`).
  - `cross-sectional-momentum` — z-scored across the 1,228-factor universe,
    BH-FDR gated.
  - `iv-realized-vol-arb` — Polymarket binary IV vs realized for the same
    horizon; degrades below σ_realized = 12 (paper-only flag).
- **Three new regression estimators** behind `?method=` on `/fit`
  (Track J, T79 picks): Elastic Net (sklearn `ElasticNetCV`, default for
  high-collinearity factor sets), **Quantile regression** (`statsmodels`
  `QuantReg`, exposes τ via query), **Bayesian linear** (PyMC NUTS, returns
  posterior intervals not just CIs). Math sketches and "when to use which"
  guidance in `docs/regression-methodology-improvements.md`.
- **Deflated Sharpe ratio** (T53) — `pfm/quant/deflated_sharpe.py`,
  consumed by `stress_test` and the new `/strategies/{id}/deflated-sharpe`
  endpoint. **Bootstrap-Sharpe** confidence intervals and **block-bootstrap**
  (stationary block, expected length tuned per series) for serial-dependence-
  honest CIs on strategy returns.
- **Frontend** (mounted by index-html-owner via W11-01 in cascade order):
  - Command palette `⌘K` (T03) — fuzzy-searches endpoints, slugs, jumps;
    recents in localStorage; keyboard nav.
  - Dark-mode toggle (T05) — persists; re-themes Plotly via the `PFM_THEME`
    custom event; 200 ms transition.
  - 7-step onboarding tour (T13) — covers jumps, backtest, clusters,
    sentiment-leaderboard; skippable + replayable from the menu.
  - Skeletons / error banners / empty states (T04/T07/T08) — friendly,
    actionable copy, inline retry, copy-trace-ID button. Replaced every
    `alert(...)` in `web/index.html` (W11-04).
  - Regression UX overhaul (T61–T63, T69, T72): sticky 3-metric summary card
    appears *before* you scroll, plain-language verdict ("NVDA moves 0.62σ
    per unit BTC factor shock, statistically significant"), real-time
    progress steps, off-screen "Fit complete" toast.
  - Result pinner (T70) — save fits to a pinboard panel, side-by-side
    compare any two, localStorage-backed.
  - Connection pill (T60, W11-30 wiring) — heartbeats `/health/deep`
    every 30 s, debounced tri-state.
  - Copy-as-cURL (T73), keyboard shortcuts (T15), responsive mobile (T75),
    print stylesheet (T14).
- **Unified `CachePool`** (T16) — `pfm/cache_pool.py` exposes
  `get/set/get_or_compute_async(key, fn, ttl)` with L1 in-process + L2 Redis,
  per-key single-flight locks (ADR-0014), and pickle-versioned envelope
  (ADR-0016). Three ad-hoc caches migrated (W11-14).
- **Coordination tooling**: `.coordination/PROTOCOL-V2.md` (hardened from V1
  after the 2026-05-16 morning `Write`-clobber incident), append-only
  `active-edits.json` discipline, `OVERNIGHT-RECAP.md`, `TASK-BOARD.md` with
  T01–T84 + W11-/W12- expansions. Race monitor armed for 5 h with zero
  corruption events.
- **~600 new tests** across Wave-11 and Wave-12 (sentiment factor unit,
  arb-matching live pipeline, OpenAPI snapshot, CachePool concurrent
  stampede, crypto-5min calibration Monte Carlo, binary-pricing strategy
  stress). Coverage on new modules ≥95 % per the project policy.

### Fixed

- **Half-open `ResolutionWindow` bug (T76b)** in arb matching — previously the
  matcher used closed-closed intervals so a market resolving on Dec 31 2024
  matched a market resolving on Jan 1 2025. Fixed by switching to
  `[earliest, latest)` half-open semantics and documenting in ADR-0009.
  Eliminates the "Trump 2024 win" ↔ "Trump 2028 win" class of false positives.
- **CORS middleware ordering** (Wave-1) — re-ordered to outermost, plus three
  exception handlers; 4xx/5xx now carry CORS headers correctly.
- **`TTLCache` pickling `pd.Series`** (Wave-3) — TTLCache was JSON-stringifying
  pandas Series and breaking on round-trip; switched to pickled-bytes envelope
  with magic header (ADR-0013). Legacy entries decode-fail and migrate on
  next set.
- **`factor_clusters.py` FastAPI response-model issue** (T80) — Pydantic-v2
  incompatible field type at ~line 372 broke `python -c "import pfm.main"`;
  fixed minimally without touching other files.
- **`/arb/scanner` 15 s timeout** → 0.80 s — single-flight lock + asyncio.gather
  fanout on inner CLOB calls (Wave-1 Track-Perf, Wave-4 follow-up).
- **`/alpha/earnings-whisper-dashboard` 13.55 s** → 2.42 s (warm) / 0.65 s
  (lifespan-prewarmed) — `ThreadPoolExecutor(8)` over per-name fetches.
- **`/terminal/sentiment-trend/spike-alerts` 15 s timeout** → 4.27 s — nested
  `ThreadPoolExecutor(2)` for parallel GDELT round-trips.

### Changed

- **Terminal is now the default mode** (was Regression). α Hub is a sub-tab
  inside Terminal mode; Regression is the third pill. Matches the user's
  "Terminal is the landing page" framing.
- **Connection pill is now tri-state** (live / slow / offline) with a 5 s
  debounce; previously it flickered Degraded on every cold cache miss.
- **41 curated slugs prewarmed at startup** via `pfm/terminal/jumps_prewarm.py`
  (T17) — `/terminal/jumps/{slug}` warm path now p50 ~50 ms.
- **Frontend mount order** consolidated in W11-01: `tokens.css` →
  `global-refinement.css` → typography → primitives → mode sheets → feature
  sheets → dark-mode → print(media=print). Documented in
  `web/css/_cascade-order.md`.
- **`/fit` accepts `method=` query parameter** (W11-28) — `ols` (default),
  `enet`, `quantile`, `bayes`. Existing callers unaffected.
- All Pydantic models that grew during Wave-11 were **appended to the end of
  `api/src/pfm/schemas.py`** per PROTOCOL-V2; no inline reorderings.

### Removed

- **6 strategies demoted** from A_GOLD to B_VALIDATED / C_PAPER tier after the
  Wave-5 stress test (4-quarter Sharpe-stability + BH-FDR + deflated-Sharpe
  triple-gate). Anti-alpha entries materialized in
  `docs/alpha-reports/alpha-report-v18.md` with signed death-certificates.
  Only the **calendar λ-ratio** survived as a structural alpha.
- **BTC latency arb** — investigated 2026-05-02 by an 8-agent burst; no
  exploitable midpoint lag exists at current spreads. Removed from the
  α Hub and the Research tab; documented as dead in the graveyard.
- Replay + Archive modes removed from the **top nav** (still reachable from
  small pills at the bottom of the Terminal pane).

---

## [Wave 10] — 2026-05-15 (60 agents, overnight)

The first 60-agent burst. The `/loop` self-paced dispatcher ran for ~5 hours
between 2026-05-15T23:30Z and 2026-05-16T04:30Z with a 28-minute heartbeat,
producing the entire `T01`–`T84` task board (see
[`.coordination/TASK-BOARD.md`](.coordination/TASK-BOARD.md)) and the
infrastructure that made Wave-11 possible. Full detail in
[`.coordination/OVERNIGHT-RECAP.md`](.coordination/OVERNIGHT-RECAP.md).

### Added

- **15 frontend primitive CSS files** (Track A, T01–T15) — `tokens.css`,
  `typography.css`, `data-cards.css`, `error-states.css`, `empty-states.css`,
  `skeletons.css`, `charts.css`, `buttons.css`, `forms.css`, `modal.css`,
  `cmdk.css`, `tour.css`, `print.css`, `shortcuts-help.css`. Single source
  of truth for design tokens; no more duplicated CSS variables.
- **10 backend perf modules** (Track B, T16–T25) — unified `CachePool`,
  jumps prewarm, shared `PolymarketHTTPPool` with HTTP/2 and keep-alive,
  `yfinance_batch` (8-worker concurrent), news SimHash deduper,
  `redis_lock.py` SETNX leader election, correlations LRU memoizer,
  orderbook pool, alpha-hub gathered detail fetcher.
- **8 new endpoints** (Track C, T26–T33), 13 ADRs (Track E, T44–T51) including
  the operational ones consumed by Wave-11.
- **4 binary-pricing models** (Track L, T81) — logit, Black-Scholes-digital,
  Brownian-bridge, Beta-binomial bayesian. Each implements
  `theoretical_price`, `calibrate`, `confidence_interval`. 60+ unit tests
  via synthetic DGP (recover known params; edge cases p→0, p→1, T→0).
- **Empirical pricing harness** (T82) — pulls 50+ resolved Polymarket markets
  (mocked with `respx`), computes Brier / log-loss / calibration-RMSE /
  early-warning lead-time / economic PnL per model. Output written to
  `/tmp/binary-pricing-report-<date>.json`.
- **Arb matching quality overhaul** (Track I, T76–T78) — robust date
  extractor (`ResolutionWindow(earliest, latest, confidence)`, ≥40 fixture
  cases), refined event-similarity scorer (jaccard + numeric threshold +
  jurisdiction + NER), audit script that produces a confusion-matrix CSV
  at `/tmp/arb-match-audit.csv`. Decision contract for shipping a pair:
  matched-score ≥ 0.5.
- **Crypto-5min calibration Monte Carlo** (T41) — model probabilities
  calibrated within 3 % under known DGP. 113 total tests in `pfm/crypto5min/`.
- **Sentiment factor unit tests** (T35) — `respx`-mocked GDELT/Reddit/HN/RSS,
  ≥85 % coverage on `pfm/sources/sentiment_factor.py`.
- **OpenAPI completeness gate** (T43) — every router mounted in `main.py`
  must appear in `/openapi.json` with summary + response model.
- **Anti-alpha rule** (T54, ADR-0010) — `sentiment-regression alert` fires
  when >40 % of >5 markets agree on a model-vs-market disagreement that
  would-have-been-alpha if shipped. Pre-emptive guardrail against
  redeploying Wave-5 demotions.

### Changed

- **Race monitor** armed for 5 h continuous. **Zero corruption events** across
  15+ writes per 30-min window. Validated that PROTOCOL-V2's append-only
  discipline survives concurrent agents.
- One coordination metadata-loss incident — the `alphahub-premium` session ran
  `Write` instead of read-merge-write on `active-edits.json`. Caught and
  recovered; the forbidden pattern is now front-and-center in PROTOCOL-V2 §
  "Forbidden Patterns That Look Innocent".
- Ruff warnings in `api/src/pfm/`: 34 → 4 (Track A retro). Tests + scripts
  remain at 51 (out-of-scope per project policy).

### Fixed

- 9 timeout endpoints reduced to 1 (`/strategies/arb/stream`, SSE-only,
  by-design). Net +10 endpoints returning 2xx, 5xx count 1 → 0.

---

## [0.2.x — pre-Wave-10] — 2026-05-08 → 2026-05-14

The "audit + features" sprint and the subsequent re-org that took the project
from a 469-test POC to a ~2,700-test data hub. Driven by `AUDITORIA_2026-05-08.md`
and the post-audit "max effort" dispatches.

### Added

- **Reverse Factor Finder** (`POST /reverse-finder`, SSE variant) — given a
  ticker, return the top-5 Polymarket markets that explain its return. Demo
  runs in <800 ms warm.
- **Prediction-Driven Alpha Scanner** (`POST /alpha/prediction-driven`) — given
  a Polymarket slug, return a ranked equity basket tracking that probability.
- **Alpha Graveyard** with signed death-certificates (`GET /alpha-hub/graveyard`).
- **Comparison tool** (`GET /terminal/compare?slugs=a,b,c[,d]`) — side-by-side
  N≤4 contracts with correlation matrix and pairs-trade z-score.
- **Universal export** — CSV/JSON on `/terminal/market`, PDF via WeasyPrint,
  bulk via `POST /terminal/export/bulk`.
- **Portfolio Optimizer** (`POST /strategies/optimize`) — HRP / mean-variance /
  min-var / risk-parity / ERC plus efficient frontier and Monte-Carlo drawdown.
- **Alert engine** (8 endpoints under `/alerts/*`) — Slack, Discord,
  HMAC-signed Webhook, In-app; SQLite-backed.
- **Decay tracking** (`GET /alpha/decay`, `/alpha/{id}/rolling-sharpe`) —
  auto-demote when rolling Sharpe drops below threshold.
- **Unified calendar** (`GET /terminal/calendar`) — resolution + earnings +
  macro events merged.
- **News Causal Chain** — connects headlines to specific factor moves.
- **Resolution P&L Tree** — Monte-Carlo over branching outcome trees.
- **Embed widgets** — iframe / `<script>` snippets / Open-Graph images.
- **Replay Mode** — 4 historical scenarios (2024 election night, 2025 BTC ATH,
  Fed-pivot day, Eurovision settlement).
- **Auto-Generated Alpha Lab** — proposes new strategy candidates from raw
  factor cross-products.
- **Cross-venue arb scanner** (`pfm/strategies_arb_router.py`, 12 endpoints
  under `/strategies/arb/*`) — full Bloomberg-style multi-tab live monitor;
  2 s SSE stream at `GET /strategies/arb/stream`; settings drawer; per-pair
  fee breakdown and live Kalshi+Polymarket orderbook.
- **PM-VIX composite risk index** — single attention-weighted dispersion
  index over the (then-)1,090 factors.
- **Quant validation primitives** — BH-FDR multi-test correction
  (`POST /quant/multitest/bh`), 4-quarter Sharpe stability gate
  (`POST /quant/quarterly-stability`), embargo walk-forward (Lopez-de-Prado)
  baked into `pfm/advanced.py`.
- **SSE multiplexed live stream** (`/terminal/live-stream`) — aggregates
  midpoint/bid/ask across up to 30 slugs.
- **Hybrid NLP sentiment** (`pfm/terminal/sentiment_nlp.py`) — VADER blended
  with a financial lexicon (earnings/Fed/credit terms get explicit weights);
  LRU-cached 10k entries.
- **Frontend**: global `⌘+K` search modal precursor, deep-linking URL state
  (`?mode=…&market=…&compare=…`), inline glossary tooltips, share button,
  permanent disclaimer footer.
- **Observability**: Prometheus `/metrics`, `/health/detail` with redis-ping
  and `git_sha`.
- **Testing infrastructure**: Hypothesis property-based tests, golden-file
  regression on terminal responses, `mypy --strict` on touched modules,
  `py.typed` marker shipped.
- **Three coexisting modes** unified into a single FastAPI app and the
  rebrand from *prediction-factor-model* to *Prediction Terminal*. α Hub
  expanded to four sections: Top Alphas, Calendar & Spreads, Cross-venue
  Arb, Crypto Micro.
- **Crypto-micro model-vs-market** (`pfm/crypto5min/`, 3 endpoints) — pairs
  Polymarket BTC/ETH 5m & 15m markets with a GBM+microstructure model
  blending 30-day σ_daily with tick-derived σ_short via variance-weighting,
  plus OFI-derived drift bias (capped ±30 %/yr) and a smaller whale-flow
  term. Background sampler is opt-in via `PFM_CRYPTO_5MIN_ENABLED=1`.

### Changed

- Refactored httpx **sync → async** in `/factors/rank`, `/fit`,
  `/factors/discover`, and the `/terminal/live-stream` SSE generator — 5–10×
  latency reduction. Bounded concurrency via `asyncio.Semaphore(20)` to stay
  under Polymarket's 1000/10 s rate limit.
- Centralized cache utilities in `pfm/cache_utils.py` — DRY refactor across
  8 modules; later subsumed by Wave-10's `CachePool`.
- CORS configurable via `CORS_ORIGINS` env var (no longer hardcoded `*`).
- nginx gains gzip, security headers (HSTS, X-Frame-Options, Referrer-Policy),
  and rate limiting.
- `uvicorn --workers 4` in the production image; Redis with persistence
  (docker volume + `appendonly yes`).
- 17 endpoints gained explicit `response_model=` declarations.

### Fixed

- Embargoed walk-forward was documented but not coded —
  now implemented and enforced.
- BH-FDR multi-test correction now blocks promotion to `B_VALIDATED` per
  ADR-0010.
- Bare `except:` handlers in 5 modules replaced with typed catches.
- Cleanup of 7 orphaned `factors.yml.bak.*` files (−2.3 MB).

### Stats snapshot at end of 0.2.x

| metric            | 0.1.0 (POC) | 0.2.x (pre Wave-10) | Δ      |
|-------------------|-------------|----------------------|--------|
| Tests passing     | 469         | ~2,550               | +2,081 |
| OpenAPI paths     | 61          | 248                  | +187   |
| Frontend LOC      | ~600        | ~1,400               | +800   |
| Factors loaded    | 1,090       | 1,228                | +138   |

---

## [0.1.0] — 2026-04-15 (initial POC)

The course-required minimum-viable surface.

### Added

- **Three modes** sharing the same FastAPI app and `web/index.html`:
  Regression / Strategies / Terminal.
- **Regression mode** — `POST /fit`, `POST /attribution`, `GET /factors`,
  `GET /health`. HAC standard errors with automatic bandwidth selection,
  configurable clipping `ε`, VIF reporting.
- **Strategies mode** — calendar λ-ratio carry, equity-cointegrated tech
  basket, China-Taiwan PCA-residual cluster.
- **Terminal mode** — 19 read-only endpoints (orderbook, tape, calendar,
  movers, heatmap, news, macro, spread, volume, liquidity, resolved, search,
  stats, categories, leaders, whales, funding, skew, health).
- **1,090 factors** loaded from `factors.yml` (944 Polymarket + 146 Kalshi).
- **Test suite** — 469 tests, ≥70 % coverage on `model.py` and
  `attribution.py`, all external IO mocked via `respx` / `httpx_mock`.
- **Foundation ADRs 0001–0009**:
  - `0001-use-fastapi.md`
  - `0002-logit-transform.md`
  - `0003-hac-newey-west.md`
  - `0004-redis-cache-ttl.md`
  - `0005-no-persistence-poc.md`
  - `0006-timezone-alignment.md`
  - `0007-daily-fidelity.md`
  - `0008-factor-universe-curation.md`
  - `0009-frontend-vanilla-html.md`
- **Docker compose** — three services (api, web, redis); `docker-compose up`
  works out of the box.
- **CI** — GitHub Actions: test, lint, build.

---

## Legend

- **Wave** = a single dispatch of N parallel sub-agents on a shared task board.
- **Tier** = `A_GOLD` (deployable, full size) → `B_VALIDATED` (deployable,
  half size) → `C_PAPER` (paper-trade only) → `D_GRAVEYARD` (do not redeploy;
  see anti-alpha list in [`docs/alpha-reports/alpha-report-v18.md`](docs/alpha-reports/alpha-report-v18.md)).
- **Anti-alpha rule** (ADR-0010): any 4-quarter sign-flip OR Sharpe-collapse
  demotes one tier. Decisions are *demotions*, never deletions — the
  graveyard exists to prevent re-pitching.
