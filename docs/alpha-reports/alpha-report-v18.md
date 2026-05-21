# Alpha Report v18 — The Final Hedge-Fund Book (Wave 1 → Wave 9)

**Generated**: 2026-05-02. End-of-cycle synthesis.
**Predecessors**: v14 (gauntlet), v15 (20-agent expansion), v16 (A_GOLD claims), v17 (wave-5 stress reckoning).
**Purpose**: Lock the production book. Tell the truth about what survives. Hand off to capital with eyes open.

---

## 1 · Executive summary

After nine waves of research, twenty parallel agents, 1090 factors ingested, 469 tests written, and a wave-5 stress-test that publicly demolished six of eight v16 A_GOLD claims, the project lands here:

- **Three strategies survive into the production book.** They are not spectacular — they are honest. Combined expected net Sharpe is **~1.2 – 1.5** on a $10k starter book, putting realistic year-1 P&L at **+$3,000 – $5,000**, not the +$50k v16 fantasy.
- **Three watchlist names** stay in paper-only status. Each has a testable regime trigger that would re-qualify it.
- **Six prior alphas archived** for reasons ranging from "tautologically efficient" (calendar arb, monotone strikes) to "data window too short" (HMM regime aware) to "BTC latency window closed in 2024" (latency arb).
- **The real durable asset is the infrastructure**: 1090-factor universe, 61-endpoint API, 19-endpoint Bloomberg-grade Terminal, three usage modes, fully containerised, CI-green. This is what compounds even if every signal decays.

**Honest verdict**: this is a respectable lower-tail hedge-fund book on a college-sized capital base. It is not a money-printer. It is the product of nine waves of self-criticism, and most of those waves *removed* alpha rather than added it. That is the correct direction of travel.

### 1.1 Where we started vs where we land

| Wave | Headline claim | What survived to v18 |
|---|---|---|
| Wave 1–2 | Retail MM + latency arb + regime classifier | None (all archived in §4) |
| Wave 3–4 | Hong-Stein, monotone strikes, calendar arb | None (efficient or non-portable) |
| Wave 5 | 8 A_GOLD strategies, +$50k/yr claim | 2 demoted to B_VALIDATED, 6 to C_TENTATIVE |
| Wave 6 | Re-validation of B tier | 3 confirmed, watchlist created |
| Wave 7–9 | Terminal + 14 backend modules + QA hardening | Infrastructure deliverable (§5) |

This table is the project in one glance: we generated, we tested, we discarded, we kept the platform.

### 1.2 Reading guide

Section 2 is the production book — read this first if you're allocating capital today. Section 3 is the watchlist (paper only). Section 4 is the graveyard with lessons. Section 5 is why the infra matters more than any signal. Sections 6–8 are forward-looking: allocation, deploy roadmap, and what we'd do next.

---

## 2 · The three-strategy production book

All three strategies have cleared the v17 gate at B_VALIDATED or better and have been re-validated under wave-7 → wave-9 audits.

### 2.1 Calendar λ-ratio (`polymarket_calendar_lambda_v1`)

| Field | Value |
|---|---|
| Tier | **B_VALIDATED with stem cap** |
| Edge thesis | Front-month vs back-month decay coefficients λ on the same Polymarket question stem are mis-priced; the term-structure of decay implies a long-front / short-back trade when |λ_front − λ_back| > 0.18 |
| Pooled Sharpe (OOS) | 1.19 |
| Net Sharpe est. | ~0.6 after fees, slippage, stem cap |
| Quarters positive | 2 of 4 |
| Bootstrap CI95 | [0.55, 2.05] |
| **Allocation** | **15% of book ($1,500 of $10k)** |
| Capacity | $5–10k notional per stem; ~$50k total before slippage explodes |
| Caveats | 74% of historical trades concentrated in 2026Q1; 43% on the single "gold high hit" stem. Stem-concentration cap of 30% per stem is mandatory. |
| Monitoring rule | Halt if 4-week rolling Sharpe < 0 OR if any single stem exceeds 30% of position-weighted exposure for two consecutive weeks. |

### 2.2 Equity-coint tech basket (`polymarket_equity_coint_tech_v1`, AAPL/AMZN/MSFT/GTLB)

| Field | Value |
|---|---|
| Tier | **B_VALIDATED** (rebuilt post wave-5 with longer history) |
| Edge thesis | Polymarket "AI capex / tech earnings" stems cointegrate with a basket of mega-cap tech equities; residual mean-reversion at z > 1.8 generates a market-neutral entry |
| OOS Sharpe (gross) | 2.1 (rebuilt 4-pair basket, n=180) |
| Net Sharpe est. | ~1.0 after equity commissions and Polymarket spreads |
| Quarters positive | 3 of 3 |
| Bootstrap CI95 | [0.6, 3.4] |
| **Allocation** | **10% of book ($1,000 of $10k)** |
| Capacity | $20–40k per leg before equity-side market impact dominates |
| Caveats | Original 14-pair version failed wave-5 (n<125 per pair). Surviving 4-pair basket clears the data-sufficiency bar. |
| Monitoring rule | Re-fit cointegration vector every 30 trading days; halt if Engle-Granger p > 0.10 on rolling window. |

### 2.3 China-Taiwan cluster (`pca_residual_china_taiwan__iran_regime + iran_coup`)

| Field | Value |
|---|---|
| Tier | **B_VALIDATED with tail-event halt** |
| Edge thesis | Geopolitical "China-Taiwan" PCA residual loads on Iran-regime / Iran-coup stems; cross-cluster residual mean-reversion fires when sentiment overshoots |
| Basket Sharpe | 4.59 IS / ~1.5 net |
| Quarters positive | 2 of 2 |
| Bootstrap CI95 | [2.28, 7.03] (IS); [0.4, 2.6] (net) |
| **Allocation** | **5% of book ($500 of $10k)** |
| Capacity | Hard ceiling $5k notional — these are thin geopolitical stems |
| Caveats | Wave-5 reduced the surviving sub-cluster from 5 stems to 2 (`iran_regime`, `iran_coup`). The original `us_invade_cuba` leg failed cross-quarter stability. |
| **Tail-event halt rules** | (1) Auto-halt if Iran sovereign CDS moves >150bp in a session. (2) Auto-halt on any confirmed kinetic strike Iran↔Israel↔US. (3) Manual review after any UN Security Council emergency session on Iran or Taiwan. (4) Maximum 48h holding period during halt. |

### 2.4 Aggregate book metrics (production three)

| Metric | Value |
|---|---|
| Total deployed | 30% of $10k = **$3,000** |
| Cash buffer | 70% = **$7,000** |
| Expected gross P&L (year-1) | +$4,500 – $7,000 |
| Expected net P&L (year-1) | **+$3,000 – $5,000** |
| Combined net Sharpe | **1.2 – 1.5** |
| Worst plausible drawdown | -$1,200 (12% of book) under simultaneous regime break |

---

## 3 · Watchlist (paper-only)

These three remain instrumented but unallocated. Each has a *specific testable trigger* that would promote it.

| Strategy | Why paper | Promotion trigger |
|---|---|---|
| `polymarket_favorites_bias_v1` | Wave-5 showed the bias is regime-conditional: Sharpe positive only when VIX < 18 and Polymarket aggregate volume > $40M/day. | If VIX < 18 *and* PM volume > $40M/day persists for 30 consecutive sessions *and* paper Sharpe > 1.0 in that window → graduate to B_VALIDATED at 5%. |
| `polymarket_sparse_trade_v1` | Classic alpha emergence — negative pre-2026Q1, +1 to +4 inside Q1. Could be regime, could be data drift. | Need ≥3 of 4 forward quarters with positive Sharpe and bootstrap CI95 LB > 0 → then promote at 5%. |
| `polymarket_fresh_consensus_v1` | Strong proxy Sharpe (3.88) but only 2 quarters of clean data. Live signal currently active. | Two more quarters at Sharpe > 1.5 with stable gamma exponent (no drift through 1.0) → promote at 7.5%. |

All three run continuously in paper mode in the Strategies tab. Their P&L is logged but not capitalised.

---

## 4 · Archived strategies

Six names, six different reasons to walk away. Each archive entry is a lesson.

| Strategy | Archive reason | Lesson |
|---|---|---|
| **BTC latency arbitrage** | Cross-venue BTC latency window collapsed in 2024 after Coinbase→CME relay upgrades. Wave-3 measurements showed median spread < 1bp, below any retail-feasible cost. | Latency arb is a perpetual race; you cannot enter post-hoc. |
| **Hong-Stein momentum** | Wave-4 confirmed the original 1999 cross-section result does not survive in prediction markets (different attention dynamics, no analyst coverage variable). | Equity-style behavioural anomalies do not auto-port to PM. |
| **Regime-aware HMM composite** | Wave-5 strat43: Δ(aware vs blind) = 0.0 across all folds. The HMM gate added literally nothing. Pair selection alone explained the result. | A model that tests as no-op is a no-op. Ship it dead. |
| **Retail market-making** | Wave-2 showed positive gross but negative net after Polymarket's 2% taker fee and adverse selection. Capacity also tiny (<$2k). | If your gross Sharpe doesn't survive realistic fees, archive without sentiment. |
| **Monotone-strikes basket** | Tautologically efficient — strike monotonicity is enforced by Polymarket's matching engine to within rounding. The "alpha" was clipping noise. | Always ask: is this constraint mechanical or behavioural? Mechanical = no alpha. |
| **Calendar arbitrage (cross-resolution)** | Same: Polymarket's resolution UMA flow forces calendar consistency at expiry. Pre-expiry there's noise; at expiry there's no edge to harvest. | Don't confuse settlement noise for tradeable inefficiency. |

---

## 5 · Infrastructure as the real asset

Even if every alpha decays to zero tomorrow, the platform itself is the durable output of this project.

### 5.1 By the numbers

| Asset | Count | Notes |
|---|---|---|
| Factor universe | **1090 factors** | Polymarket (~860) + Kalshi (~230), all daily-aligned to UTC |
| Backend API endpoints | **61 endpoints** | OpenAPI-generated, fully typed |
| Terminal endpoints | **19 endpoints** | Multi-chart Bloomberg-style surfacing |
| `pfm.terminal_*` modules | **14+ modules** | term-structure, cointegration matrix, residual decomposer, vol-cone, λ-surface, regime classifier, etc. |
| Test count | **469 tests** | ≥70% line coverage on `model.py` and `attribution.py` (course requirement met) |
| Frontend modes | **3 modes** | Regression / Strategies / Terminal |
| ADRs | 7 | All ≥150 words, all genuine decisions |
| Containers | 3 | api / frontend / cache, single `docker-compose up` |

### 5.2 The three modes

1. **Regression mode**: original course deliverable. Fit a factor model of stock returns on PM-derived factors. HAC OLS, VIF reporting, configurable clipping ε. This is what gets graded.
2. **Strategies mode**: live P&L tracker for the production three + watchlist three. Daily mark, drawdown, regime overlay.
3. **Terminal mode**: 19-endpoint research desk with multi-chart panels for term structure, cointegration matrices, λ-surfaces, residual decomposition, vol cones, regime classification, and cross-cluster PCA.

### 5.3 The 14+ terminal modules in detail

| Module | What it does | Wave introduced |
|---|---|---|
| `terminal_term_structure` | Plots λ(τ) decay curves across stems | Wave 7 |
| `terminal_cointegration_matrix` | NxN Engle-Granger / Johansen heatmap | Wave 7 |
| `terminal_residual_decomposer` | PCA / factor-residual breakdown for any basket | Wave 7 |
| `terminal_vol_cone` | Rolling realised vol vs IV-implied cone | Wave 8 |
| `terminal_lambda_surface` | 3D surface of decay vs strike vs tenor | Wave 8 |
| `terminal_regime_classifier` | HMM state probabilities per session | Wave 8 |
| `terminal_cross_cluster_pca` | PCA across geopolitical / macro / single-name clusters | Wave 8 |
| `terminal_orderflow_imbalance` | Buy-vs-sell pressure on top stems | Wave 9 |
| `terminal_calendar_spread` | Front-back λ ratio explorer (drives §2.1) | Wave 9 |
| `terminal_basket_builder` | Ad-hoc basket construction with live PCA | Wave 9 |
| `terminal_event_overlay` | News/event markers over price series | Wave 9 |
| `terminal_factor_loader` | Browse / filter the 1090-factor universe | Wave 7 |
| `terminal_qa_dashboard` | Coverage, test counts, CI status | Wave 9 |
| `terminal_pnl_attribution` | Per-strategy P&L decomposition | Wave 9 |

### 5.4 Why infra > alpha

Alpha decays. Infra compounds. The 14+ terminal modules each took 2–6 hours to build and will keep generating research questions for as long as Damian wants to point them at new data. The factor universe (1090 names) plus the regression pipeline is, in itself, a usable research desk independent of any single strategy. A future Damian who returns in six months with new data will find the entire desk still operational and instantly useful — that is not true of any of the archived signals.

---

## 6 · Capital allocation recommendation

For a $10k starter book today (2026-05-02):

| Bucket | Allocation | $ | Rationale |
|---|---|---|---|
| Calendar λ-ratio | 15% | $1,500 | Highest-confidence structural edge, capacity-bound |
| Equity-coint tech basket | 10% | $1,000 | Market-neutral, longest clean history |
| China-Taiwan cluster | 5% | $500 | Highest gross Sharpe but thinnest stems and tail-event risk |
| Cash buffer | **70%** | $7,000 | Reserve for paper→live promotions, drawdown absorption, optionality |
| **Total deployed** | **30%** | **$3,000** | |

**Expected outcome**: +$3,000 to +$5,000 net P&L over 12 months at combined Sharpe 1.2 – 1.5. Worst plausible drawdown ~12% of book.

The 70% cash is not laziness — it is capital reserved for (a) absorbing drawdowns without forced liquidation of correlated positions, (b) doubling into watchlist promotions when their triggers fire, and (c) optionality to deploy into wave-10 strategies.

---

## 7 · 90-day deploy roadmap

| Window | Milestone | Pass/Fail criterion |
|---|---|---|
| **Q2 2026 (May–Jun)** | Paper-trade all three strategies in the Strategies tab. Daily reconciliation against Terminal mid prices. Log every entry/exit decision. | Net paper Sharpe ≥ 1.0 across all three combined; no execution-feasibility surprises. |
| **Q3 2026 (Jul–Sep)** | Live deploy at half-size ($1,500 total exposure). Pre-funded Polymarket account. Manual entry, daily re-mark. | Live Sharpe ≥ 0.8 (allowing for slippage haircut). Drawdown < 8% of deployed. |
| **Q4 2026 (Oct–Dec)** | If Q3 passes, scale to full $3,000 allocation. Begin watchlist trigger monitoring. | **The 4-quarter rule**: if any of the production three has now seen 4 consecutive positive quarters, it graduates to A_GOLD and gets a 50% allocation bump from cash buffer. |

**Hard kill switches** (any one triggers full halt of the offending strategy):
- Single-stem concentration > 30% for 2 consecutive weeks
- 4-week rolling Sharpe < 0
- Drawdown > 15% on the strategy's deployed capital
- Tail event in geopolitical cluster (see §2.3 halt rules)

### 7.1 Pre-deploy checklist (must clear before any live capital)

- [ ] Polymarket account funded with USDC, KYC complete, withdrawal limits known
- [ ] Kalshi account active for cross-venue arb research (wave-10)
- [ ] Daily reconciliation script running against Terminal mid prices
- [ ] All three strategies generating paper P&L for ≥ 4 consecutive weeks
- [ ] Drawdown alarms wired to email/SMS (not just dashboard)
- [ ] Manual entry runbook written and dry-run executed twice
- [ ] Tax tracking: 1099 / W-9 implications for prediction-market gains documented
- [ ] Backup data feed: secondary mid-price source identified per strategy

### 7.2 What "passes" Q3 actually means

Live Sharpe ≥ 0.8 sounds modest, but it is the right bar. Paper-to-live Sharpe haircut is empirically 30–50% in retail PM trading (slippage, fee surprises, missed entries). A paper Sharpe of 1.2 should haircut to ~0.7–0.85 live. If we land below 0.6, something structural is wrong and we halt rather than scale.

---

## 8 · What we'd do next — wave-10 priorities

If continuing past v18, in priority order:

1. **Cross-venue arb (Polymarket ↔ Kalshi)**. The factor universe spans both venues but no strategy yet exploits the cross-venue basis. Wave-9 spot checks suggested 2–4% standing spreads on overlapping geopolitical stems. Highest expected new alpha for one wave of effort.
2. **Calendar λ-surface modelling**. The current calendar strategy is two-point (front vs back). Fitting the full λ(t) surface and trading curvature instead of slope is the natural extension. Requires more data per stem; may need 6 more months.
3. **Order-book microstructure**. We've ignored the LOB entirely. Wave-2 retail-MM was a coarse take; a real book-imbalance / queue-position model is the next frontier and would also feed back into execution quality for the production three.
4. **News-event regime tagger**. Hand-tag major events (Fed meetings, elections, geopolitical breaks) and re-run all archived strategies with regime-conditional re-evaluation. Some of the archived names may be alive in specific regimes.
5. **Monte Carlo allocation optimiser**. Currently allocations are eyeball-set. A proper Black-Litterman or risk-parity layer over the production three would reduce allocation arbitrariness.
6. **Multi-account execution harness**. If live deploy passes Q3, build an automated executor with circuit breakers tied to the §7 kill switches.

---

### 8.1 Sequencing rationale

Why cross-venue arb (#1) first? Two reasons. (a) The infra is already in place — both venues are ingested into the 1090-factor universe. The marginal cost of running cross-venue residual scans is one wave of analyst time. (b) Cross-venue spreads are the single edge least likely to be "in" the existing strategies — they tap a fundamentally different inefficiency (fragmented liquidity) than the time-series mean-reversion that drives the production three. So the diversification benefit is real, not redundant.

Why news-event regime tagger (#4) before MC allocation (#5)? Because regime-conditional re-evaluation might *resurrect* archived strategies, which would change the universe MC has to optimise over. Always do the universe-expansion work before the optimisation work.

## 9 · The honest postmortem in three lines

1. **We started believing we had ten alphas. We end with three.** That is the correct ratio for retail-edge research; anything higher would be self-deception.
2. **The wave-5 stress test was the most valuable single piece of work in the entire project.** It cost us paper-Sharpe but bought us calibration.
3. **The infrastructure is the asset.** The 1090-factor universe, the 61-endpoint API, the 19-endpoint Terminal, the 469 tests — these outlive any single signal and constitute the actual deliverable to a quant employer or investor.

End of v18. End of cycle. Ship it.
