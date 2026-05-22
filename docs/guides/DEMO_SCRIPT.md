# 15-minute live demo — minute-by-minute script

**Audience:** professor / quant interview panel / general technical viewer.
**Goal:** demonstrate that the project is a credible "Yahoo Finance of
prediction markets + quant research workbench", not a toy POC.
**Length:** 15 minutes hard cap. Cushion of 2 minutes for Q&A.

---

## Pre-flight checklist (T-30 minutes)

Run this checklist **30 minutes before** the demo. Do not skip steps;
the failure modes below are all things that have actually happened.

- [ ] **Boot the stack.** `docker-compose up -d` from the repo root.
      Watch for "API healthy" / "Web up" / "Redis ready" in the logs.
- [ ] **Warm the cache.** Hit each surface once so live demos render
      instantly:
      ```bash
      ./scripts/smoke_test.sh http://localhost:8000
      curl -s http://localhost:8000/factors > /dev/null
      curl -s 'http://localhost:8000/terminal/movers?window=24h' > /dev/null
      curl -s 'http://localhost:8000/terminal/heatmap' > /dev/null
      curl -s 'http://localhost:8000/alpha-hub/graveyard' > /dev/null
      ```
- [ ] **Pre-load a watchlist** with 2 markets you have rehearsed:
      `nvda-ai-mentions-by-2026q3` and `polymarket_calendar_lambda_v1` (or
      whichever survive in `docs/alpha-reports/alpha-report-v18.md` on demo day).
- [ ] **Confirm fallback cassette.** If Polymarket is down,
      `PFM_USE_CASSETTES=1 docker-compose up` replays from
      `tests/cassettes/`. Have this command pasted in your terminal,
      ready to swap in.
- [ ] **Check the disclaimer footer is visible.** If a reviewer screenshots
      it, "Not investment advice · UTC timestamp" must be in frame.
- [ ] **Open three tabs in the browser**, all on `http://localhost:8080`:
      1. Terminal mode — landing.
      2. Quote page for the rehearsed market.
      3. Strategies → α Hub.
- [ ] **Have these CLI snippets ready** in a separate terminal pane:
      ```bash
      curl -s http://localhost:8000/health/detail | jq
      curl -s -X POST http://localhost:8000/reverse-finder \
        -H 'content-type: application/json' \
        -d '{"ticker": "NVDA", "lookback_days": 90}' | jq '.top_factors[:3]'
      ```
- [ ] **Silence Slack and notifications.**
- [ ] **One-liner intro practiced** — see minute 0:00.

---

## Minute-by-minute

### 0:00 — 0:30 · Hook (30 s)

Say:

> "This is the Yahoo Finance of prediction markets, plus a quant
> strategies hub. It pulls 1,228 factors from Polymarket and Kalshi,
> fits HAC-corrected factor models against equity returns, and ships
> three validated strategies live. Today: a five-minute tour, a five-
> minute deep-dive into one alpha, and five minutes on engineering."

Click: do nothing. Just the landing page.

### 0:30 — 2:00 · Terminal panorama (90 s)

Click through, **read the panel name out loud each time**:

1. **Top movers** — point at one row with a sparkline trending up.
   "Each row is a 24h Δprob, sparkline is hourly. Click anywhere → quote
   page."
2. **Theme heatmap** — "Coloured cell = aggregate 24h Δprob, intensity =
   theme volume. Click → filter."
3. **Calendar** — "Resolutions, earnings, macro, all merged. Next 14
   days."
4. **PM-VIX index** — "Composite dispersion. Single number for the
   prediction-market volatility regime."

**Wow moment #1**: press `Cmd+K`. The global search modal pops. Type
`nvda` — autocomplete finds three markets. Hit Enter. Point at the URL
bar: it now reads `…/#mode=terminal&market=nvda-…`. Copy it, open in a
second tab — the same market detail loads instantly. "Every view is a
shareable URL; back/forward restores state without a reload."

### 2:00 — 4:30 · Quote page deep dive (150 s)

You are now on a quote page (NVDA-AI-mentions or your rehearsed pick).

Walk the reviewer through the eight chart grid:

1. **Volume** — "30-bar daily, sanity check on liquidity."
2. **Spread vs peer** — "z-score against the highest-correlation
   cointegrated peer. Auto-discovered, not hand-picked."
3. **Realized vol cone** — "Quantile bands, current vol overlaid. Tells
   you whether realised is in or out of distribution."
4. **Prob fan to resolution** — "Forward GBM in logit space. Not a
   forecast — a sanity envelope."
5. **Orderbook depth ladder** — live from CLOB.
6. **Calendar λ-ratio** — "If this market has a sibling resolution-date
   contract, the ratio surface shows up here. Mean-reverting by
   no-arbitrage."
7. **Recent trades · Lee-Ready** — "Buyer / seller initiated tape."
8. **Fair price · 4 models** — "TWAP, Bayes, GBM, peer-cointegration.
   Gauge against current mid."

Click **Compare**. Add two sibling markets. The side-by-side fills with
correlation matrix and pairs-trade z-scores.

### 4:30 — 5:30 · Reverse Factor Finder (60 s)

Switch tab to your CLI pane. Run:

```bash
curl -s -X POST http://localhost:8000/reverse-finder \
  -H 'content-type: application/json' \
  -d '{"ticker": "NVDA", "lookback_days": 90}' | jq '.top_factors[:3]'
```

Read the slug names out loud. Then say:

> "Sub-second response, warm cache. The inverse direction is what makes
> this useful for a quant — most platforms only let you go ticker → news,
> not ticker → which prediction-market is explaining its return."

**Wow moment #2**: paste the top slug into the regression panel via
Cmd-K, hit **Run fit**, β and t-stat appear.

### 5:30 — 7:00 · Regression mode (90 s)

Already pre-filled with the slug from minute 5:30. Add a second factor
manually (rehearsed: `gpt5-released-by-eoy-2026`).

Click **Run fit**.

Lead with the **Verdict pill** at the top of the result card — STRONG /
MIXED / WEAK plus a one-line headline summary. "This is the answer in
two seconds; the rest is the audit trail." Then walk through:
- β̂ with HAC SE; **VIF** and **Clip%** now inline in the table.
- Durbin-Watson — note distance from 2.
- Residual plot — should look stationary; if it doesn't, raise lag.
- Tick **Auto-prune collinear factors** and re-fit to show pruned slugs
  surfacing in the response.

> "The HAC standard errors come from `statsmodels.OLS(...).fit(cov_type=
> 'HAC', cov_kwds={'maxlags': L})` — we don't roll our own. ADR-0003
> documents the choice."

### 7:00 — 9:00 · α Hub + one validated alpha (120 s)

Switch tab to **Strategies → α Hub**.

Filter: tier `B_VALIDATED`. Show the cards:

- **PCA-residual china/taiwan basket** — click the card.
  - Fullscreen tearsheet loads. Read out: OOS Sharpe 4.59, drop-top-3
    Sharpe 3.66, CI95 lower bound > 0. "Survives even with the three
    best pairs zeroed out — that's what robustness looks like."
  - Press **→** (or click **Next**) — the modal flips to the next
    B_VALIDATED card without closing. "Filter-aware cycling: I'm only
    paging through what the filter bar matched."
- **Calendar λ-ratio** — explain the stem cap rule (max 3 trades per
  stem per 30 days). "We had to cap because 43% of trades were on a
  single stem; without the cap it's a single-stem play, not a strategy."

**Wow moment #3**: switch to the **Alpha Graveyard** tab. Read out two
death certificates:

- `polymarket_favorites_bias_v1` — "Negative Sharpe pre-2026Q1, +1.4
  in Q1, fades. Classic regime emergence. Not on live capital."
- `regime_aware_macro_composite` — "The HMM regime gate adds zero alpha
  vs naive pair selection across all folds. The feature is a no-op.
  Documented so future me doesn't re-pitch it."

### 9:00 — 10:30 · Replay Mode (90 s)

Click **Strategies → Replay → 2024-11-05 election night**.

Hit Play at 4× speed. Watch the presidential and Senate contracts
re-price as state calls land. Pause at a dramatic moment. Say:

> "This is a teaching tool and a sanity check. Backtests are abstract;
> Replay puts you in the actual minute the market repriced. Useful for
> debugging strategies that depend on event-time alignment."

### 10:30 — 12:00 · Engineering surface (90 s)

Switch to the API tab. Walk through:

- `http://localhost:8000/docs` — Swagger. Scroll. **266+ endpoints**,
  every request/response Pydantic-typed.
- `http://localhost:8000/health/detail` — uptime, redis ping, git_sha.
- `http://localhost:8000/metrics` — Prometheus output.
- Show the test count: `cd api && pytest --collect-only -q | tail -3`.
  2547+ tests, all external IO mocked.
- Briefly mention CI: "GitHub Actions, ruff + pytest + pip-audit +
  Codecov + Docker build, all green."

### 12:00 — 13:30 · Production posture (90 s)

Open `DEPLOYMENT.md` in the browser (if you have it on a static host)
or in the terminal. Walk through:

- Fly.io deploy in 6 commands.
- Render via blueprint.
- nginx with gzip, security headers, rate limiting.
- Redis with AOF persistence.
- CORS env-driven (no `*` in production).
- 4-worker uvicorn.

Mention the production checklist (`docs/PRODUCTION_CHECKLIST.md`).

### 13:30 — 15:00 · Wrap + Q&A invite (90 s)

> "To recap: 1,228 factors, 266 endpoints, 2547 tests, three validated
> alphas, one hub. Built in a month. Everything I've shown is in this
> one repo, no external services beyond data feeds, deployable in six
> commands on Fly.io. Happy to take questions."

---

## Q&A bait — pre-cooked answers

These are the most likely questions, with the answers you should give
verbatim. Memorise them.

### "How do you know your alphas are real and not p-hacked?"

> "Three gates. (1) Bootstrap CI95 lower bound on Sharpe must be > 0,
> stationary block bootstrap. (2) Drop-top-N robustness — recompute the
> Sharpe with the N best pairs zeroed out; it has to stay positive.
> (3) 4-quarter Sharpe stability — if any single quarter has Sharpe
> below 0.5 or sign-flips vs full sample, the strategy stays C_TENTATIVE.
> Also BH-FDR multi-test correction since we screen ~4,500 candidates.
> Wave-5 killed six of eight A_GOLD claims under those gates."

### "What's the alpha graveyard?"

> "A public ledger of strategies that *failed* validation, with signed
> death certificates explaining the failure mode. Six entries today.
> The pedagogical point: any quant project that doesn't show its
> graveyard is hiding the file drawer. Mine's open."

### "Why no React?"

> "ADR-0009. The interaction model is read-mostly: fetch, redraw, click.
> A reactive framework's overhead doesn't earn its keep here. Plain
> HTML + Plotly via CDN means zero build step, instant deploys, no node
> version drift. If we ever grow a write-heavy surface I'll revisit."

### "What's next?"

> "Three things. (1) Wave-6 robustness pass next quarter to either
> promote C_TENTATIVE strategies to B_VALIDATED or retire them. (2) Two
> new factor sources: Manifold and PredictIt. (3) A market-making harness
> on the validated calendar λ-ratio strategy, post-paper."

### "How does this differ from Kalshi's own dashboard or Polymarket's UI?"

> "Three ways. (1) Cross-venue: Polymarket, Kalshi, FRED, yfinance unified.
> (2) Quant primitives: HAC regression, BH-FDR, embargo walk-forward,
> bootstrap CI — none of which the venue UIs surface. (3) The
> graveyard. The venues won't tell you which strategies *don't* work.
> I will."

### "Where do you store user state?"

> "Server-side state is in Redis (cache only) and SQLite (alerts).
> Client-side: localStorage for watchlist and recents. ADR-0005
> documents the no-database POC stance — strategies and factor universe
> are version-controlled YAML, not a database, so the entire
> reproducibility story is `git checkout`."

### "What's the catch?"

> "Capacity. Every alpha I've shown is < $200k notional before
> slippage eats half the edge. This is a research tool that scales
> *down* to a small allocation, not a fund-scale platform. I'd rather
> ship one honest alpha at $50k than a fake one at $50M."

---

## Failure-mode playbook (if something goes wrong live)

| Symptom | Recovery |
|---|---|
| Polymarket 429 / 5xx mid-demo | `PFM_USE_CASSETTES=1 docker-compose restart api` (already pasted in your terminal). |
| Plotly chart blank | Hard reload `Cmd+Shift+R`. The CDN occasionally serves stale. |
| `/fit` slow (> 3s) | Switch to a pre-warmed factor combo in the rehearsed list. Avoid live-discovery during demo. |
| α Hub card missing | Refresh; `web/data/alpha_strategies.json` is loaded once at page open. |
| Reviewer asks for a feature you don't have | "It's in `docs/future-work.md` — happy to walk through the roadmap after." |
