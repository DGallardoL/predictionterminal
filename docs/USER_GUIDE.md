# User Guide — Prediction Markets Quant Hub

A friendly walkthrough of every feature in this POC: a web service that turns
live Polymarket data into factor models, statistical-arb strategies, and a
Bloomberg-style data terminal. If you have never used the app before, start
at §1 and skim. If you want to jump in, §3 (Regression), §4 (Strategies) and
§5 (Terminal) are independent and can be read in any order.

---

## 0. Tour & shortcuts

- **First-run tour** — a 4-step guided tour fires once per browser; replay
  any time with `?tour=1` (e.g. `http://localhost:8080/?tour=1`). Skip is
  persisted via `pfm:tour:done=1` in localStorage.
- **Cmd-K / Ctrl-K** — global search across markets, tickers, strategies,
  and recents (see §10.2).
- **Browser back / forward** — Terminal market detail pushes a hash URL
  (`#mode=terminal&market=<slug>`); `popstate` restores both the active
  mode and the open market, so back returns to the overview without a
  page reload.
- **Shareable links** — copy the address bar at any time; opening the
  same URL in another tab lands directly on that mode + market + filters.

---

## 1. Getting started

### Run it locally

The single moving part is a FastAPI server on port 8000. The HTML frontend is
served from the same process at `/ui/`. From the project root:

```bash
cd api
source .venv/bin/activate          # the project's virtualenv
uvicorn pfm.main:app --reload      # serves API + UI on :8000
```

Then open:

- **<http://localhost:8000/ui/>** — the full web UI (the thing you actually use)
- **<http://localhost:8000/docs>** — Swagger/OpenAPI explorer (every endpoint, click "Try it out")
- **<http://localhost:8000/redoc>** — read-only ReDoc view of the same OpenAPI

A `docker-compose up` from the project root achieves the same thing plus a
Redis cache and the static-site frontend on port 8080. Either workflow is
fine for local use.

Smoke-test the API before opening the UI:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/factors | jq '.factors | length'
# → 1228+
```

### What the UI looks like

A single HTML page with a **mode switcher** at the top (Regression · Strategies
· Terminal). The three modes share the API but otherwise have no overlap — pick
the one that matches your task and ignore the others.

---

## 2. The three modes, side by side

| You want to… | Use this mode | Why |
|---|---|---|
| Fit `r_AAPL = α + β · Δlogit(market) + ε` and read coefficients. | **Regression** | Classical factor-model workflow with HAC standard errors. |
| Find tradeable alpha (cointegrated pairs, OU bands, Kalman hedges, BTC arb). | **Strategies** | 13 sub-tabs of stat-arb plumbing. Start in α Hub. |
| Browse Polymarket like Bloomberg — heatmap, charts, orderbook, fair-price. | **Terminal** | Read-only data hub. No model fitting; just look. |

You can switch modes at any time without losing state — each mode keeps its
own form values while you flick between them.

---

## 3. Regression mode — fit a factor model

This is the original POC: project equity returns onto Δlogit of one or more
prediction-market probabilities and read off the coefficients with proper
HAC-corrected standard errors. The workflow has four steps.

### 3.1 Pick a target

Top of the panel, "Target ticker": any yfinance-compatible symbol
(`AAPL`, `^GSPC`, `BTC-USD`, …). Pick the date window with the date pickers —
default is "last 6 months ending today". Polymarket history is roughly
Sept 2025 → present, so 2025-09-01 is your effective floor.

### 3.2 Discover & select factors

Five tabs in the factor picker:

- **Presets** — curated preset gallery (e.g. "AI / NVDA basket",
  "Fed-cut strikes", "Election binaries"); one click loads the target +
  factor set together.
- **Smart picker** — type a ticker or theme; the picker ranks factors by
  univariate |t-stat| over the chosen window and pre-fills the top-K.
- **Curated** — the ~1228 factors in `factors.yml`, hand-named with themes.
- **Discover** — call `GET /factors/discover` to surface live high-volume
  Polymarket markets that are *not* in the curated list.
- **Custom / Selected** — paste a slug or `clobTokenId`; the Selected tab
  is your shortlist sent to `/fit`.

**Tip:** start with 1–3 factors. The Verdict pill (§3.4) and auto-prune
(§3.4) make collinearity safe to experiment with — but readability still
suffers above 5 factors.

### 3.3 Run the fit

- **Window**: how many trading days. The HAC `maxlags` is set to
  `floor(N^{1/3})` by default (HAC rule of thumb), so 100 obs ⇒ 4 lags.
- **Clipping ε**: probabilities are clipped to `[ε, 1−ε]` before the logit
  transform. Default `0.01`. If you see Δlogit columns that look stuck at 0,
  lower ε to `0.005` or `0.001` and refit.
- Click **Run fit**. The frontend POSTs to `/fit` and renders the coefficient
  table, residual plot, and diagnostics.

### 3.4 Read the output

The response leads with a **headline summary** + **verdict pill**:

- **STRONG** (green) — at least one factor with |t| ≥ 3, R² ≥ 0.10, no
  red-flag VIFs.
- **MIXED** (amber) — significant factor but VIF or DW warning attached.
- **WEAK** (grey) — no factor clears |t| ≥ 2; R² < 0.05.

Below the pill, the coefficient table now includes **VIF** and **Clip%**
columns inline (Clip% = % of observations where Δlogit was zeroed by the
ε-clip, exposing dead factors). Click any coefficient row to drill into
its rolling-β chart and per-factor diagnostic popover.

- **β̂** with HAC standard error and a t-stat (significance stars).
- **R²** / adj. R² and **Durbin-Watson** in the header card.
- **Residual series** — if autocorrelated, raise the HAC lag.

Β̂ ≈ 0.20 with t = 3.5 says: a one-unit Δlogit (≈ a flip from 0.27 → 0.50)
moves the equity 20 bps on the day, with strong-enough significance under HAC.

**Auto-prune.** Pass `?prune_collinear=true` on `/fit` (or tick the
"Auto-prune collinear factors" box in the UI) to drop factors whose VIF
exceeds 10 before re-fitting; pruned slugs come back in the response under
`pruned_factors` with their VIFs.

**Save / Share / Compare.** The result card has three buttons: **Save**
(LRU dropdown of the last 10 fits, persisted in localStorage), **Share**
(copies a permalink with the full payload encoded in the URL hash), and
**Compare** (re-runs the same target across two factor sets side by side).

### 3.5 Companion endpoint: `/attribution`

Once you have a fit, POST the same payload to `/attribution` to decompose any
realised return into per-factor and residual components. Useful for "why did
NVDA move 4% on Wednesday."

---

## 4. Strategies mode — 13 sub-tabs

### 4.1 The two top-of-stack tabs ("Trade now")

#### α Hub `data-stab="alphahub"` — start here

The hub of curated, robustness-tested strategies. Loads
`web/data/alpha_strategies.json` and `live_signals.json`. After the wave-5
robustness gate (see `docs/alpha-reports/alpha-report-v17.md`), the **88 curated strategies**
are tier-classified:

- **A_STRUCTURAL** (4) — mechanically cointegrated strike-family pairs
  (Fed-target strikes, BTC dip strikes). Half-size deploy.
- **B_VALIDATED** (30) — bootstrap CI95 lower bound > 0, paper → small live.
- **C_TENTATIVE** (27) — regime-driven; paper only until wave-6 confirms.
- **D_RAW** (27) — failed gates; do not deploy.
- **A_GOLD** (0) — no strategy yet has the 4-quarter shelf life to deserve it.

Use the filter bar (Min tier · Category · Theme · Sort) to drill in. Click any
card to open a fullscreen tearsheet modal.

**Modal navigation.** Inside the tearsheet, **← Prev / Next →** buttons
(and keyboard `←` / `→` / `Esc`) cycle through the *currently filtered
and sorted* card list — narrowing the filter narrows the cycle, so you
can ladder through "B_VALIDATED, calendar theme" without dropping back
to the grid. Card data is unified through
`GET /alpha-hub/leaderboard?full=true` (single source of truth) so
modal, grid, and live-signal pills always agree.

#### BTC Arb `data-stab="btcarb"` — live

The Binance-vs-Polymarket Chainlink-lag latency arb on BTC Up/Down 5m and 15m
markets. Runs continuously against `/btc-arb/active-market` and
`/btc-arb/midpoint`. Shows the live edge in basis points.

### 4.2 Research workflows

- **Auto-Backtest** `data-stab="autobt"` — one-click pipeline: scan → backtest
  → leaderboard. The "press here for alpha" button. POSTs to
  `/strategies/auto-backtest`.
- **Scanner** `data-stab="scanner"` — Cartesian scan across the whole catalog
  on three tracks (implication, conditional, cointegration). Calls
  `/strategies/scan`.
- **Pair Explorer** `data-stab="pair"` — distribution-free single-pair
  diagnostics: implication test, HAC-OLS, Fréchet-Hoeffding bounds.

### 4.3 Quant tools (9 tabs)

Each is a thin UI on a single endpoint. One or two sentences each:

- **Cointegration** `coint` → `/strategies/cointegration`. Engle-Granger 2-step
  with ADF p-value and AR(1) half-life. Use to confirm a pair is mean-reverting
  before you trade it.
- **Kalman Hedge** `kalman` → `/strategies/kalman-hedge`. Time-varying β̂ₜ via
  Kalman filter — adapts when the cointegrating relationship drifts.
- **Pairs Trading** `pairs` → `/strategies/pairs-backtest`. Walk-forward
  z-score backtest with Sharpe, hit-rate, max drawdown.
- **OU Bands** `ou` → `/strategies/ou-bands`. Continuous-time OU optimal
  entry/exit thresholds from Ornstein-Uhlenbeck calibration.
- **Mean-Rev** `mr` → `/strategies/mean-reversion`. Hurst R/S exponent and
  the variance-ratio test — model-free MR tests.
- **Granger** `granger` → `/strategies/granger`. Bidirectional causality test
  in both directions — answers "leader vs follower."
- **Event Model** `event` → `/strategies/event-model`. Multi-factor explanatory
  model with HAC standard errors. Like Regression mode but with a
  rich preset library for events.
- **Basket** `basket` → `/strategies/basket-stat-arb`. Avellaneda-Lee
  PCA-residual basket stat-arb — generalises pairs trading to k > 2.
- **Spot vs Implied (SVI)** `svi` → `/strategies/spot-vs-implied`. Compares the
  underlying's spot-implied probability (Yang-Zhang vol + closed-form GBM)
  against Polymarket's traded mid.

---

## 5. Terminal mode — the data hub

A read-only Bloomberg-style browser of every Polymarket market. Three regions:
top strip, sidebar, detail panel.

### 5.1 Top strip (overview)

- **Theme heatmap · 24h** — coloured cells per theme (politics, crypto, macro,
  sports, tech, culture, science). Cell colour = aggregate 24h Δprob, intensity
  scales with theme volume. Click a cell to filter the list pane.
- **Top movers** — biggest |Δprob| over the last 24h with traded volume sanity
  filter. Click a row to drill into the detail pane.
- **Upcoming resolutions** — markets resolving in the next ~14 days. Click to
  drill in.

Backed by `GET /terminal/overview`.

### 5.2 Sidebar

- **Search** — debounced typeahead on `GET /terminal/search?q=…`. Matches names,
  slugs and themes.
- **Watchlist** — pin markets from the detail-pane "★" button; persists in
  localStorage.
- **Themes** — explicit theme links (All / Politics / Crypto / Macro / Sports /
  Tech / Culture / Science).

### 5.3 Detail panel

When you pick a market, the right column populates with:

- **Hero** — name, slug, current YES price, 24h Δ, volume, resolves-on date,
  resolution-source link, and the ★ pin button. Backed by
  `GET /terminal/market/{slug}`.
- **Main chart** — price line with togglable overlays (Volume · Spread (z) ·
  Fair · Bid-Ask). Timeframe tabs: 1H / 1D / 7D / 30D / MAX. Backed by
  `GET /terminal/market/{slug}/history`.
- **Multi-panel chart grid** — eight Bloomberg-style mini-cards:
  1. **Volume** · 30-bar daily.
  2. **Spread vs peer** — z-score against the highest-correlation cointegrated
     peer (auto-discovered).
  3. **Realized vol cone** — quantile bands (5/25/50/75/95) of rolling
     realised vol vs the current realised vol.
  4. **Prob fan to resolution** — forward-looking probability cone using the
     calibrated GBM in logit space.
  5. **Orderbook depth ladder** — current bid/ask ladder from the CLOB.
  6. **Calendar λ-ratio surface** — only renders for markets that participate
     in a calendar pair (e.g. "Fed-cut by Jul" vs "Fed-cut by Sep").
  7. **Recent trades · Lee-Ready** — tape of recent fills classified
     buyer/seller-initiated.
  8. **Fair price · 4 models** — gauge comparing market mid against four
     reference fair-price models (TWAP, Bayes, GBM, peer-cointegration).
- **Stats card** — full quantile/half-life/spread-cost/Hurst diagnostics.
- **Related** — peer markets in the same theme or with significant
  cointegration.
- **Resolution panel** — the resolution source URL and rule text.

---

## 6. Strategies the user should actually know about

After the wave-5 audit (May 2026, see `docs/alpha-reports/alpha-report-v17.md`), exactly four
clusters survive at a tier you'd put live capital on. In priority order:

### 6.1 PCA-residual china/taiwan basket — B_VALIDATED, the strongest signal

Two pairs in production: `pca_residual_china_taiwan__us_invade_cuba` and
siblings. Walk-forward OOS Sharpe 4.59 across 6 pairs (n_obs = 67 days),
drop-top-3 robustness 3.66 with CI95 [0.02, 7.37] — survives even with the
three best pairs zeroed out. Quarterly trajectory is *rising* (early third
−0.04 → mid +6.44 → late +8.90), not regime-decay. **Recommended weight: 15%.**

### 6.2 Calendar λ-ratio — B_VALIDATED with stem cap, the only structural alpha

`polymarket_calendar_lambda_v1`. Pooled Sharpe 1.19 (n=35, 8.5 months),
bootstrap CI95 [0.55, 2.05]. The catch: 74% of trades are in 2026Q1, 43% on a
single stem (the "gold-ATH calendar" trade). **Hard rule for live deployment:
max 3 trades per stem in any 30-day window**, and re-check that
drop-dominant-stem Sharpe stays > 0.5 quarterly. Weight: 10%.

### 6.3 Equity-coint tech basket — flagged, watch only

`polymarket_equity_coint_tech_v1`. The headline pair `AAPL ↔ apple-largest-jun`
posts an OOS Sharpe of **6.79** (half-life 3.1d), with siblings
`MSFT ↔ msft-largest` (3.09), `GTLB ↔ gitlab-acquired` (5.99) and
`TSLA ↔ spacex-highest-ipo` (5.01). Wave-5 demoted it to **C_TENTATIVE** — all
14 input pairs had n < 125 obs, so the basket can't yet form. Watch for
wave-6 once we cross the data threshold.

### 6.4 Bundle arbs — opportunistic only

`polymarket_event_bundle_v1`. Within-event mutually-exclusive bundles
(non-negRisk events) where Σ(YES asks) < 1 lock a gross profit. Live examples:
**Eurovision 2026** (vol $124M, +1.2% net), **MLS Cup** (+1.2% net, $16M vol),
Israel-strikes count (+0.7% net, $6.5M vol). Filter: only non-negRisk events
with ≥85% candidate exhaustiveness; **assume 1.8% taker fee per leg**. Run
opportunistically — these expire when someone else takes them.

### 6.5 Live cross-exchange arb (BTC)

The Binance-vs-Polymarket Chainlink-lag arb on BTC Up/Down 5m & 15m markets.
Live signals stream through the BTC Arb tab. Manual cross-exchange arb on
specific FOMC contracts is also tracked when surfaced.

---

## 7. Reading the data

### 7.1 What is a Polymarket probability?

The mid-quote of a binary YES contract on Polymarket. A 0.27 mid means the
market thinks there's a 27% chance the event resolves YES. We pull these
daily at UTC close via `/prices-history?fidelity=1440`. **Always use daily
fidelity** — sub-daily silently fails for resolved markets.

### 7.2 The logit transform

Probabilities live in [0, 1] which is the wrong scale for linear regression.
We apply

\[ \text{logit}(p) = \log\frac{p}{1 - p} \]

so a unit change in logit ≈ a unit of Bayesian information regardless of where
on the [0, 1] scale you started. That's the lever Δlogit pulls in the factor
model. To avoid `log 0` / `log ∞` we clip to `[ε, 1−ε]` with default
**ε = 0.01** (configurable on `/fit`).

### 7.3 Why we assume 1.8% taker fee

Polymarket charges takers a small fee (currently around 1.8% per side under
the relayer model we use for sizing). All bundle-arb and pairs-backtest net
numbers in the alpha reports deduct **1.8% per side per round trip**, so
gross-Sharpe ÷ ~3 to get a sane net estimate at small size. If you trade
maker-only the cost goes near zero, but assume taker for sizing — it's the
honest case.

### 7.4 Log returns, not simple returns

The factor model uses `r_t = log(P_t / P_{t-1})`. Both equity and
prediction-market returns. This makes them additive across time and symmetric.

### 7.5 Timezone alignment

Both Polymarket timestamps (unix seconds) and yfinance closes are normalised
to **UTC date** at `pandas.Timestamp(date).normalize()`. Don't mix tz-aware
and tz-naive timestamps — see ADR-0006.

---

## 8. What is NOT alpha (the cautionary list)

These strategies looked attractive in v15/v16 but failed wave-5 robustness.
Do not deploy them — they're regime-driven, not structural.

| Strategy | Failure mode |
|---|---|
| `polymarket_favorites_bias_v1` | Negative Sharpe pre-2026Q1, +1-4 in Q1, fades after. Classic regime emergence. |
| `polymarket_sparse_trade_v1` | Same pattern: dead → Q1 spike → fade. Cannot rule out luck. |
| `polymarket_var_ratio_mr_v1` | Sharpe series [-0.89, +4.89, -2.05] — only 17% of MR pairs persist across windows. |
| `polymarket_fresh_consensus_v1` | Live signal active but no quarterly stability data; B_FDR_ONLY only. |
| `polymarket_prelec_skewness_v1` | Gamma drift 0.70 → 1.08 (passes through 1, where the bias mechanism inverts). |
| `regime_aware_macro_composite` | The HMM regime gate adds **zero** vs naive pair selection across all folds. The regime feature is a no-op. |
| 17× `B_FDR_ONLY` cross-theme pairs | BH-FDR pass alone is not enough without bootstrap CI confirmation. |

The unifying lesson: **2026Q1 was an unusually high-dispersion regime** (gold
ATH, BTC vol, Fed-pivot anticipation). Anything that monetises price
dispersion or attention deficits lit up simultaneously and has been fading.
None of these belong on live capital until they have ≥4 quarters of out-of-
sample evidence with CI95 lower bound > 0.

---

## 9. Glossary

- **ADF p-value** — Augmented Dickey-Fuller test for unit root. p < 0.05
  means the series is stationary (good for cointegration legs and spreads).
- **AR(1) half-life** — for a mean-reverting series with AR(1) coefficient
  ρ, the half-life is `−log(2)/log(ρ)` days. Pairs with half-life < 5 days
  are tradeable; > 30 days are too slow for size.
- **OOS Sharpe** — out-of-sample annualised Sharpe ratio. Anything ≥ 1.5
  net of costs is interesting; ≥ 3 is rare and worth re-checking for bugs.
- **IS Sharpe** — in-sample. Always higher than OOS. The IS/OOS ratio is
  a quick over-fit sniff test (we want it close to 1).
- **BH-FDR** — Benjamini-Hochberg false-discovery-rate correction. With
  4499 pair candidates, naive p < 0.05 lets ~225 false positives through.
  We require BH-q ≤ 0.05.
- **DSR** — Deflated Sharpe Ratio. Adjusts a
  reported Sharpe down for the number of trials and skew/kurt of returns.
- **Bootstrap CI95** — stationary block bootstrap 95% confidence interval
  on the Sharpe. We require the **lower bound > 0** for B_VALIDATED.
- **Drop-top-N robustness** — recompute the basket Sharpe with the top-N
  pairs zeroed out. If it stays positive, the alpha isn't concentrated in
  a handful of lucky pairs.
- **HAC** — heteroskedasticity- and autocorrelation-consistent
  standard errors. We use `statsmodels.OLS(...).fit(cov_type='HAC',
  cov_kwds={'maxlags': L})` with `L = floor(N^{1/3})`.
- **VIF** — Variance Inflation Factor. > 10 ⇒ multicollinearity, drop a
  factor or use ridge.
- **Δlogit** — first difference of the logit-transformed probability. The
  regressor in our factor model.
- **negRisk event** — Polymarket bundle where the legs are *not* mutually
  exclusive (Σ ≠ 1). Bundle arb only works on **non**-negRisk events.
- **λ-ratio (calendar)** — short-leg-to-long-leg probability ratio across
  resolution dates of the same event family. Mean-reverts by no-arbitrage.
- **PCA-residual** — residual after stripping the first k principal
  components (k=3 in our setup: risk-on/off, theme, vol). The leftover is
  the idiosyncratic stat-arb signal.
- **Stem (calendar)** — the underlying event that spawns multiple
  resolution-date contracts (e.g. "gold ATH" is a stem with Jun/Jul/Aug
  contracts). Stem caps prevent over-concentration.
- **Lee-Ready** — Lee & Ready (1991) trade-classification rule. Marks
  each fill buyer- or seller-initiated using the prevailing midpoint.
- **Yang-Zhang vol** — drift-independent realised volatility estimator
  using OHLC. We use it as the spot-vs-implied input.

---

## 10. New features (post-audit, 2026-05-08)

Everything below shipped after the May-2026 audit and is *not* covered in
sections 3–5. Each lives behind its own endpoint(s) and frontend surface.

### 10.1 Reverse Factor Finder

"I have a ticker. Tell me which prediction-market contracts explain its
return." The inverse direction of Regression mode.

```bash
curl -s -X POST http://localhost:8000/reverse-finder \
  -H 'content-type: application/json' \
  -d '{"ticker": "NVDA", "lookback_days": 90, "top_k": 5}' | jq
```

The response ranks the top-K factors by the absolute t-stat of a univariate
HAC-OLS, with the join window aligned to UTC dates. The frontend exposes
this as a search-style box on the regression panel — type a ticker, hit
**Find factors**, and the top-5 are pre-loaded as a `selected` set you can
fit immediately.

Use cases: discovery on a new equity, post-earnings forensics, sanity-check
on whether a thematic pair (e.g. NVDA vs the AI-mentions Polymarket
contract) actually correlates.

### 10.2 Cmd-K global search

Press **`Cmd+K`** (macOS) or **`Ctrl+K`** (Linux/Windows) anywhere in the
UI. A modal opens with:

- **Markets** — typeahead across the 1228 curated factors.
- **Tickers** — yfinance symbols (`NVDA`, `^GSPC`, `BTC-USD`, …).
- **Strategies** — α Hub strategy names.
- **Recents** — your last 10 picks, persisted in `localStorage`.

Hit Enter on any result to navigate (preserves your URL state — see §10.4).

### 10.3 Alert engine

Programmatic alerts on factor moves, strategy regime drift, calendar
events, or threshold crossings on any endpoint that returns a numeric
field. Channels: **Slack**, **Discord**, **Webhook with HMAC**, and
**In-app**. Backed by SQLite so alerts survive an API restart.

```bash
# Create a Slack alert when NVDA-ai-mentions Δprob > 5pp in 1h
curl -s -X POST http://localhost:8000/alerts \
  -H 'content-type: application/json' \
  -d '{
    "name": "NVDA AI mentions spike",
    "channel": "slack",
    "webhook_url": "https://hooks.slack.com/services/...",
    "trigger": {
      "kind": "factor_move",
      "slug": "nvda-ai-mentions-by-2026q3",
      "metric": "delta_prob_1h",
      "op": ">",
      "threshold": 0.05
    }
  }'

# List alerts
curl -s http://localhost:8000/alerts | jq

# Mute / delete
curl -s -X PATCH http://localhost:8000/alerts/{id}/mute
curl -s -X DELETE http://localhost:8000/alerts/{id}
```

Webhook channel signs the payload with `X-PFM-Signature: sha256=<hex>` so
the receiver can verify authenticity. Set `PFM_ALERTS_DRY_RUN=1` (default
in dev) to log instead of actually firing.

### 10.4 Embed widgets

Embed any market or strategy card on a blog post or in a Twitter card.
Three flavours:

- **iframe** — copy the `<iframe>` from the **Embed** button on any quote
  page. Renders the full quote-card panel (chart + stats + resolution).
- **`<script>` snippet** — pastes a small client-side initializer that
  fetches `/terminal/embed/{slug}.json` and renders a Plotly chart inline.
  Better for SPAs where iframes are awkward.
- **OG images** — `GET /terminal/og/{slug}.png` returns a 1200×630 PNG
  with the latest price, sparkline, and resolves-on date for use in
  `<meta property="og:image">`. Cached for 5 minutes.

The auto-generated HTML/Markdown snippet on the **Embed** button is the
shortest path; the API exists for power users.

### 10.5 Replay Mode

A historical sandbox with **four pre-baked scenarios**. Each replays a
fixed-frame timeline at user-controllable speed (0.25× / 1× / 4× / max),
exactly like a video scrubber:

| Scenario | What happens |
|---|---|
| **2024-11-05 election night** | Watch the presidential and Senate Polymarket contracts re-price as state calls land. Six headline contracts streamed minute-by-minute. |
| **2025 BTC ATH (Mar 2025)** | BTC 5m / 15m Polymarket contracts during the parabolic run; Binance overlay shows the Chainlink-lag arb in action. |
| **Fed-pivot day (Sep 2025)** | FOMC strikes contract family + S&P / Treasury moves in 1-second slices. |
| **Eurovision settlement** | Bundle-arb scenario; watch Σ(YES) collapse from 1.04 to ~1.00 as the venue clears. |

Open via **Strategies → Replay** or `GET /replay/scenarios`. Each scenario
has a "what to look for" tooltip describing the teachable moment.

### 10.6 Portfolio Optimizer

Five methods on the same set of strategies. Pick a subset of validated
alphas, choose an objective, get back weights and an efficient-frontier
curve.

| Method | When to use it |
|---|---|
| **Equal-Weight (EW)** | Baseline; what an unsophisticated user would do. |
| **Mean-Variance (MV)** | Classical Markowitz. Sensitive to mean estimates; use with shrinkage. |
| **Min-Variance** | Ignores expected returns, minimises portfolio variance. Robust. |
| **Risk Parity** | Equal risk contribution per asset. Good when expected returns are similar. |
| **Equal Risk Contribution (ERC)** | Same family as risk parity, with a different solver — converges where naive RP doesn't. |
| **Hierarchical Risk Parity (HRP)** | Clustering-based RP. Doesn't need to invert the covariance matrix; robust to small samples. |

Endpoint:

```bash
curl -s -X POST http://localhost:8000/strategies/optimize \
  -H 'content-type: application/json' \
  -d '{
    "strategy_ids": ["pca_residual_china_taiwan", "polymarket_calendar_lambda_v1"],
    "method": "hrp",
    "lookback_days": 180,
    "monte_carlo": true
  }'
```

The response includes the weight vector, the efficient frontier (50 points),
and a Monte-Carlo drawdown distribution (10,000 sims) so the user can see
the 95th-percentile drawdown they're signing up for.

### 10.7 Comparison tool (side-by-side N≤4)

Compare up to 4 contracts head-to-head: aligned price chart, correlation
matrix, pairs-trade z-scores between every pair.

```bash
curl -s 'http://localhost:8000/terminal/compare?slugs=nvda-ai-mentions,gpt5-by-eoy,openai-funding-2026'
```

Frontend access via the **Compare** button on any quote page (multi-select
up to 3 more contracts).

---

## 11. Mobile & accessibility

The UI was swept at **768 px (tablet)** and **375 px (phone)**. What works
on small screens:

- Terminal homepage table, theme heatmap, market detail (vertically
  stacked chart grid), α Hub card grid, Crypto Micro snapshot.
- ARIA roles on every clickable row, focus ring at 0.45 alpha for
  keyboard nav, `prefers-reduced-motion` honoured, 44 px tap targets.
- The 4-step onboarding tour replays via `?tour=1`.

**Deferred to desktop** for now: the live cross-venue Arb dashboard
(needs the 3-fr/2-fr split), the Regression rolling-β chart grid, and
the multi-leg Portfolio Optimizer table. These render but require
horizontal scrolling on phone.

---

## 12. Resilience & loading states

Every panel uses a shared `_termPanelState(el, kind, opts)` helper to
render **loading**, **empty**, and **error** states with consistent
copy. Behaviour:

- Upstream **502 / 503 / 504** soft-fail to an empty state with a
  one-click retry — the page never goes blank.
- **429** responses honour `Retry-After` and back off; a small "rate
  limited" pill appears next to the affected panel.
- Token-id race on `/fit/preview` (factor not yet resolved) auto-retries
  once after 600 ms, masked from the user.
- All upstream callsites cache through Redis L2; a STALE badge appears
  when the cache served data older than its TTL.
- The Terminal news panel scores headlines for **per-market relevance**
  (anchor terms + topic match, NFKD-normalized to handle accents) and
  hides anything below a 0.18 floor — the panel shows an empty state
  rather than off-topic noise.

---

## Where to go next

- `docs/quants.md` — full math with LaTeX (factor model, HAC, logit clipping).
- `docs/alpha-reports/alpha-report-v22.md` — the current robustness verdict.
  Wave-7 4Q reckoning (2026-05-19) re-tested every Wave-6 promotion against the
  strict 4-quarter Sharpe-stability gate (`joint_days ≥ 360`): **0 of 5 prior
  `A_STRUCTURAL` cards cleared**, so the tier is empty until Aug 2026 at the
  earliest and every prior promotion was reverted to `B_VALIDATED`.
- `docs/strategies.md` — every endpoint catalogued with sample inputs.
- `docs/DEMO_SCRIPT.md` — minute-by-minute 15-minute demo plan.
- `docs/PRODUCTION_CHECKLIST.md` — pre-launch / launch / post-launch checklist.
- `docs/adrs/` — nine architecture decision records.
- <http://localhost:8000/docs> — every endpoint, with "Try it out".
