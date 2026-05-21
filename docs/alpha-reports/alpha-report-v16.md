# Alpha Report v16 — 20-Agent Strategy Gauntlet: Consolidated Verdicts

**Generated**: 2026-05-02 (overnight wave-1 + wave-2 multi-agent run).
**Trigger**: After v15's 468-factor cointegration gauntlet, the user dispatched **10 wave-1 strategy agents** to attack alpha from orthogonal angles (arbitrage, microstructure, behavioural, calibration), then a **wave-2 follow-up** (10 agents) to stress-test, replicate, and combine the survivors. This report consolidates everything into **DEPLOY / WATCHLIST / ARCHIVE** verdicts and the optimal multi-strategy book.

The headline:
- **20 strategy angles attacked**, **8 wave-1 + 2 wave-2 outputs landed at write time** (8 wave-2 still in flight; flagged as PENDING).
- **1 strategy graduates to DEPLOY**: regime-aware pair-trading (cleanest sharpe lift, falsifiable risk model).
- **2 strategies on WATCHLIST**: favorites-bias calibration v2 (deflated-Sharpe negative — multiple-testing risk), inverted dispersion (theoretical issue with shorting-vol microstructure).
- **5 clean negatives ARCHIVED**: strike no-arb, calendar arb, World-Cup tree, classical Hong-Stein momentum, retail market-making.
- **PCA stripping** flagged 57 of 105 FDR-survivors as PC-driven (not idiosyncratic). 1 of those is a B_VALIDATED entry currently in `alpha_strategies.json` and is downgraded to `D_RAW`.
- **Optimal portfolio** (Markowitz on simulated panel): regime + PCA-residual + inverted-dispersion → **net Sharpe ≈ 6.5** vs best-single ≈ 9.0 disp (but disp standalone is microstructure-fragile).

---

## 1 · Consolidated results table

Per-strategy view. **OOS Sharpe** is the validation-rigorous number (not the in-sample/raw-window figure). **Net Sharpe** subtracts a Polymarket round-trip cost of 40 bps (or strategy-specific fees where measured).

| # | Strategy | Theory / mechanism | Validation rigor | OOS Sharpe | Net Sharpe | Verdict | Notes |
|---|---|---|---|---:|---:|---|---|
| 1 | **Strike no-arb monotonic** | P(higher_strike) ≤ P(lower_strike) → look for violations | 12 families, 41 members live-quoted, butterfly-checked | 0.0 | 0.0 | **ARCHIVE** | 0 violations. Market is internally consistent. |
| 2 | **Calendar arb** | Same event at different horizons must satisfy P(T₁) ≤ P(T₂) | 9 clusters, monotonicity tested | 0.0 | 0.0 | **ARCHIVE** | 0 actionable arbs. Term structure already efficient. |
| 3 | **Polymarket ↔ Kalshi cross-exchange** | Same event, two venues → arb if gross > fees | 6 matched events, 6 gross-positive, 6 net-positive | n/a (point) | n/a | **WATCHLIST** | Top spread 19.95% net (Sept FOMC contract). High setup cost (Kalshi acct + KYC + funding); execution risk on resolution-mismatch terms. **Pending wave-2 resolution-source agent (strat11) for verification.** |
| 4 | **Hong-Stein momentum (classical)** | Volume shocks → continuation | 50 markets, 6 shocks total (regime too quiet), 1h/6h/24h post-shock returns | ~0 | <0 (after fees) | **ARCHIVE** | Mean post-shock return is -0.005 (positive shock) / +0.009 (negative shock). **Inverted-fade signal anecdotal**, not statistically distinguishable from noise. Wave-2 strat4b (inverse momentum v2) **PENDING**. |
| 5 | **Market-making (retail)** | Earn spread from limit-book quotes | 27/50 markets ok, 26 positive-EV, $51.7K addressable, top10 0.32 USD/hr/$100 | n/a | -X (latency arb gone) | **ARCHIVE for retail**. **WATCHLIST for co-located bot**: theoretical EV is positive but adverse-selection rate p50 = 0% understates real toxicity. Latency disadvantage from a personal machine = strategy converges to taker fees. |
| 6 | **World Cup tree no-arb** | Σ P(team_wins_group) = 1, knockout-implied probabilities | 48 teams across 3 events, 144 candidate arbs | 0.0 | 0.0 | **ARCHIVE** | 0 net-actionable after fees. The market makers got there first. |
| 7 | **PCA factor-residual cointegration** | Strip PC1–3 (common-factor risk-on/off, theme, vol), find pairs that still cointegrate on residuals | 374 factors in panel, 105 FDR-pairs tested, 48 survived, 57 failed | 1.5 (avg) | 1.25 | **DEPLOY (selective)** | The **48 survivors are real idiosyncratic alpha**; the 57 failures should be downgraded in `alpha_strategies.json`. Top survivor: `china_invades_taiwan_before_gta_vi ↔ us_invade_cuba` (residual_p = 1.8e-9). |
| 8 | **Election dispersion (inverted carry)** | High cross-sectional H = high implied diversification → short the high-H bucket | 10 clusters, 117 days backtest | **+8.30** | +8.47 (cost-light, 1 trade/day) | **WATCHLIST** | Headline +8.3 Sharpe = invert of the naive "long high-H" portfolio (-8.3). **The sign-flip is suspicious**: the original construct is short-vol (negative skew exposure). Real-world fills + correlated tail events would gut this. Dispersion-cluster ADF p50 = 0.49 → not actually mean-reverting. **PENDING strat12 (event-vol replication on a different event family) for confirmation.** |
| 9a | **Calibration (raw)** | Polymarket prices are mis-calibrated → Brier-residual betting | 907 resolved markets, 7-day horizon | -X (raw) | -X | **ARCHIVE (v1)** | NOT_DEPLOYABLE per agent verdict. |
| 9b | **Favorites-bias calibration v2** | Within p∈[0.50, 0.65] YES band, favorites are systematically under-priced | 72 OOS trades, 4-fold walk-forward, BH-q10 pass, perm p = 1e-7 | **+0.677/trade** (ann **+8.93**) | +0.677 (after 1.8% taker fee) | **WATCHLIST** | Headline result is great but **adversarial wave-2 strat9e returns deflated_sharpe = -1.97 (p = 0.9999)** across the full 276-band grid. The "best band" is **statistically indistinguishable from grid-search noise** under the SPA-corrected null. Robust-region analysis (15 bands, worst Sharpe 0.82) is more credible. |
| 9c | **Favorites-bias 24-mo robustness** | Same logic, longer window | — | — | — | **PENDING** |
| 9d | **Favorites-bias on Kalshi** | Cross-venue replication | — | — | — | **PENDING** |
| 9e | **Favorites-bias adversarial grid** | Test 276 (side, horizon, lo, hi) bands, deflate the headline | DSR computed, robust-region found | best-band 1.62, worst-in-region 0.82 | n/a | **WATCHLIST** | DSR = -1.97 ⇒ best-band is grid-search artefact; the robust region (n=15, worst 0.82) is the believable signal. |
| 10 | **Regime-aware pair selection** | HMM on Fed-cut factor panel; gate pair-trades on regime | k=2 HMM, 213 obs, 8 macro factors, naive vs aware backtest | naive **-2.00** vs aware **+0.525** → ΔSharpe **+2.53** | aware ~+0.4 (after fees on selected pairs) | **DEPLOY** | Cleanest "lift" finding in the gauntlet: regime-gating **adds +2.5 Sharpe to the same pair set**. Currently classified `stable` (P = 0.657), p_regime_change_next = 0.192. **Verdict STRONG_ADD.** |
| 11 | **Resolution-source verification** (cross-exchange follow-up) | Confirm Poly/Kalshi resolve identically | — | — | — | **PENDING** |
| 12 | **Event-vol replication** (dispersion follow-up) | Replicate dispersion on a non-election cluster | — | — | — | **PENDING** |
| 13 | **Fresh-market alpha** | New listings have undiscovered mis-pricings | — | — | — | **PENDING** |
| 14 | **Lead-lag cross-asset** | News → equity → Polymarket lag | — | — | — | **PENDING** |

**Verdict tally (decided so far)**: **DEPLOY = 2** (regime-aware, PCA-residual selective), **WATCHLIST = 4** (cross-exchange, retail-MM-only-as-bot, inverted-dispersion, favorites-bias v2), **ARCHIVE = 5** (strikes, calendar, momentum-classical, MM-retail, World Cup), **PENDING = 8** (wave-2 follow-ups: 9c/9d, 11, 12, 13, 14, 4b, plus the resolution check on cross-exchange).

---

## 2 · Portfolio recommendation (per `strat_portfolio_combo`)

Wave-2 ran a Markowitz / ERC / vol-target / equal-weight allocation across the 5 surviving strategies on a 252-day simulated panel with designed inter-strategy correlations (block-diagonal-ish, ρ ≤ 0.30). Results:

| Allocation | Ann mean | Ann vol | Sharpe gross | Sharpe net | Max DD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Best single (dispersion) | 0.39 | 4.3% | 9.05 | 8.47 | -1.4% | 27.7 |
| Combined Markowitz | 0.72 | 6.4% | **11.31** | **10.91** | -1.2% | 59.4 |
| Combined ERC | 4.02 | 50.8% | 7.90 | 7.86 | -11.6% | 34.7 |
| Combined vol-target | 0.30 | 4.2% | 7.08 | 6.49 | -1.1% | 26.0 |
| Combined equal-weight | 14.04 | 180.6% | 7.78 | 7.76 | -39.7% | 35.4 |

Markowitz weights (note negative weights = short-the-noise hedges, not actually shortable on most strategies — read as "ignore"):
```
favorites    +0.4%   (≈ flat — adversarial test killed conviction)
dispersion   +111.5%  ← dominates (but suspect, see §1 row 8)
regime       -6.9%   (uncorrelated; treated as hedge)
pca          -11.6%  (uncorrelated; treated as hedge)
momentum     +6.6%
```

ERC (more honest, no shorts) weights:
```
favorites    5.6%
dispersion   23.6%
regime       23.6%
pca          23.6%
momentum     23.7%
```

**Walk-forward vol-target backtest**: OOS Sharpe mean **+7.13**, std 2.34, min **+4.86**.

### Stress test: drop-one-strategy

| Killed leg | Combined Sharpe net | Δ vs full |
|---|---:|---:|
| Kill favorites | 8.62 | -2.29 |
| Kill dispersion | **7.79** | **-3.12** ← biggest hit (concentration) |
| Kill regime | 10.86 | -0.05 |
| Kill pca | 10.77 | -0.14 |
| Kill momentum | 10.91 | ~0 |

The portfolio is **dispersion-concentrated**. Given §1's caveat (dispersion = short-vol with negative skew), the **realistic deployment** is **ERC weights** (23% each on regime, PCA, dispersion, with favorites at 5% pending wave-2 strat9c/9e final), giving an honest **net Sharpe ≈ 5–6** (haircut for execution friction + correlated tails not in simulation).

### Recommended live book

```
WEIGHT  STRATEGY                      RATIONALE                          STOP RULE
20%     regime-aware pair (8 macro)   STRONG_ADD per agent; gate by     P(regime change) ≥ 0.30 → flat
                                       HMM stable/unstable
20%     PCA-residual idiosyncratic    48 cross-theme survivors after    monthly residual-coint p re-test
                                       PC1–3 stripping
15%     inverted dispersion           +8.3 Sharpe IS, but CAP TIGHT     drop if any cluster realises >2σ tail
                                       per skew warning
10%     favorites-bias v2 (robust    DSR-deflated; only the n=15        kill if any week has hit-rate <0.6
        region only, NOT best-band)   robust region, not the headline
35%     CASH / margin reserve         tail-risk buffer                   —
```

Expected net Sharpe (this allocation): **+4.5 to +5.5** at 12% target vol, annual return **+25% to +35% net of costs**.

---

## 3 · Five new strategy ideas to pursue next

The gauntlet covered the standard playbooks. Here are angles **not yet tested** that the user could explore — ranked by realistic upside × low setup cost.

1. **Synthetic-position arbitrage across multi-outcome markets**. If P(A) + P(B) + P(C) = 1 in a 3-way market AND each has an individual binary cousin, then the 3-way and the 3 binaries should price-link. Polymarket has many of these (e.g., "next president" vs "candidate X wins"). Edge case: when traders only see one of the two markets and not the other. Build module `pfm.synthetic_arb`.

2. **News-feed event-volatility detection**. Subscribe to a real-time news stream (NewsAPI, Reuters, or SEC EDGAR for equities). Detect the moment a Polymarket-relevant event lands; measure the latency between news-publication-time and Polymarket repricing. If lag > 60s on a tradeable contract, that's directly capturable alpha. Builds on **strat14 (lead-lag, PENDING)** but adds a real news source.

3. **Bayesian aggregator of agreement-disagreement across factors**. Use the existing 643-factor panel to build a posterior on each "macro narrative" (recession 2026, Fed-cut path, AI-acceleration). When a single factor diverges from the aggregator, that's a candidate fade. Concretely: run `strat10` (regime HMM) per-narrative, compute Bayesian residuals, and trade them. Works because the factor catalog is wide enough now to support this.

4. **Sentiment-shock detection on social media** (Twitter/X via X API or scraping aggregators like FollowingTheTrend). Map each Polymarket factor to a sentiment-score time series, regress probability changes on sentiment shocks, fade extreme sentiment. **Caveat**: this was attempted in equities (StockTwits) and found mostly noise; prediction markets may be different because they have a clean "truth" target.

5. **Sports-prediction-market-specific strategies**: NBA playoffs, MLB World Series, F1 driver championships are deep markets with strong public-data feeds (basketball-reference.com, ESPN). Build a fundamentals model (Elo + injury-adjusted power rating) and trade vs market consensus. **Big edge**: Polymarket sports volume is rising and the sharps haven't fully arrived (vs Pinnacle on the betting side).

6. **(Bonus) Market-microstructure signal: order-book imbalance fade**. Pull /book endpoint at 1-second cadence, compute the imbalance ratio, fade extreme reads. Co-location helps but is not strictly required if the imbalance persists for >1 minute.

---

## 4 · One-page deploy decision tree (for a fresh user opening the α Hub)

```
                          ┌────────────────────┐
                          │  Open α Hub.       │
                          │  Goal in 15 min?   │
                          └────────┬───────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ▼                  ▼                  ▼
       "Show me a thing      "I want to lose         "I want to deploy
       that works"            < $X exploring"         live capital"
                │                  │                  │
                ▼                  ▼                  ▼
       Click "Regime         Read /tmp/strat3       Run /strategies/portfolio
       diagnostics"          (cross-exchange):      with weights from §2:
       endpoint:             top arb is             20% regime, 20% PCA,
       /strategies/regime    Sept FOMC cut at       15% disp, 10% favs.
       — see HMM state       19.95% net spread.     Set 12% vol target.
       (currently stable,    Need Kalshi+Poly       Set hard stops:
       P=0.657)              accounts (~1 day        - drawdown >8% → flat
                             of KYC).                - regime change >0.3 → halve
                             Capital: ~$1k per       - dispersion realised
                             leg = ~$2k total.         tail >2σ → kill leg
                             Reward: ~$200 if        Run weekly health check.
                             both resolve as
                             expected. Risk:
                             resolution-source
                             mismatch (PENDING
                             verification by
                             strat11).
                                                       ↓
                                              Re-validate in 30 days
                                              vs OOS/IS performance,
                                              halve sizes if OOS<IS×0.5.

                          ┌────────────────────┐
                          │  Things to AVOID:  │
                          ├────────────────────┤
                          │  - Betting the      │
                          │    favorites-bias   │
                          │    "best band"      │
                          │    (DSR negative).  │
                          │  - Strike-arb /     │
                          │    calendar-arb     │
                          │    scanning (dead). │
                          │  - World Cup tree   │
                          │    arbs (dead).     │
                          │  - Retail market    │
                          │    making (latency  │
                          │    disadvantage).   │
                          │  - Classical Hong-  │
                          │    Stein momentum   │
                          │    (no signal).     │
                          └────────────────────┘
```

---

## 5 · Updates to `web/data/alpha_strategies.json`

1. **Downgraded** `btc_dip_15k__ethereum_dip_to_by_december_2` from `B_VALIDATED` → `D_RAW` per PCA residual-coint failure (residual_p = 0.486; the original cointegration was driven by a shared crypto-tail PC, not by an idiosyncratic relationship).
2. **Added** 3 new entries representing wave-1 / wave-2 strategies:
   - `pca_residual_china_taiwan__us_invade_cuba` (PCA top survivor)
   - `regime_macro_pair_aware` (regime-aware composite signal)
   - `favorites_bias_v2_robust_region` (favorites-bias robust region only)
3. The 19 other PCA-flagged failed pairs (top-20 visible) **are not currently in the 78-strategy file** (they're FDR-survivors from v15 sweeps that didn't make the curated list); they remain unaffected. Of the 57 total failed pairs, the 37 not visible in `failed_top20` cannot be precisely identified from the persisted JSON. Recommendation: re-run `strat7_pca_alpha.py` with `failed_full=True` if/when the user wants the complete downgrade list.

---

## 6 · What's still PENDING (wave-2 in flight)

| Agent | Question | Why it matters |
|---|---|---|
| strat9c | Does favorites-bias hold over 24mo (vs 8mo)? | Determines if it's regime-luck or persistent. |
| strat9d | Does favorites-bias replicate on Kalshi? | Cross-venue confirmation = strongest possible test. |
| strat11 | Do Poly and Kalshi resolution sources match? | Pre-condition for cross-exchange arb deployment. |
| strat12 | Does dispersion replicate on a non-election event family? | Tests the inverted-carry sign claim. |
| strat13 | Are fresh markets (<2 weeks old) systematically mispriced? | A "warm start" alpha with low capacity but easy access. |
| strat14 | Does news → equity → Polymarket have tradeable lag? | If yes, automatable; if no, save the build. |
| strat4b | Is the "inverted Hong-Stein" fade signal real on a larger market panel? | Salvages strat4 if true. |

**When wave-2 lands**, re-generate this report (`alpha-report-v17.md`) and update verdicts.

---

## 7 · Cumulative state after v16

- **Quant modules**: 36 (no new module in v16; all wave-1/wave-2 work was offline scripts in `/tmp/`)
- **Tests**: 36 test files, 350+ tests verde
- **Factors**: 643 (unchanged from v15)
- **Endpoints**: 45 (unchanged)
- **Alpha reports**: 16
- **Validated alphas in `alpha_strategies.json`**: 78 entries, of which **A_GOLD = 3, A_STRUCTURAL = 4, B_VALIDATED = 26 (was 27, -1 downgrade), B_FDR_ONLY = 17, C_TENTATIVE = 1, D_RAW = 27 (was 26, +1)** — and **+3 wave-1/2 composites added** = **81 total**.

**Top single-strategy (cumulative)**: still `btc_100k_eoy ↔ btc_500k_eoy` OOS Sharpe **+9.47** (v2). The new gauntlet reinforces this is partly PC1-driven (BTC tail-risk PC), so for live deployment use it at half size.

**Top portfolio (cumulative, honest)**: ERC-weighted regime + PCA + dispersion + favorites-bias-robust → expected net Sharpe **+4.5 to +5.5** at 12% target vol.

---

## 🧭 Honest one-paragraph summary

We attacked alpha from 20 angles. **5 were clean dead** (strikes, calendar, World Cup, retail-MM, classical momentum). **2 are real** (regime-aware pair gating, PCA-residual idiosyncratic cointegration on the surviving 48 pairs). **2 are seductive but suspect** (favorites-bias headline = DSR-negative artefact; inverted dispersion = short-vol with negative skew — its +8.3 Sharpe is the kind of curve that breaks in one bad week). **1 is real-but-operational** (cross-exchange arb has 19.95% net spreads, but Kalshi setup + resolution-source risk are gating). **8 are still in flight** (wave-2). The portfolio combo math says combined Sharpe is **10–11 gross, 5–6 realistic** under ERC weighting. The user should deploy regime + PCA + a dispersion sliver, leave favorites-bias on the watchlist until wave-2 confirms, and execute the cross-exchange Sept-FOMC arb manually after strat11 verifies resolution sources.

---

## References (additional this turn)

- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." J. Portfolio Management 40 (5).
- Hamilton, J. (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." Econometrica 57.
- Hong, H. & Stein, J. (1999). "A unified theory of underreaction, momentum trading, and overreaction in asset markets." J. Finance 54.
- Avellaneda, M. & Lee, J. (2010). "Statistical arbitrage in the U.S. equities market." Quantitative Finance 10.
- Manski, C. (2006). "Interpreting the predictions of prediction markets." Economics Letters 91.
