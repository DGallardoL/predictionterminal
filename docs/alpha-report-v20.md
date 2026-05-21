# Alpha Deployment Report v20 (Q3-2026 Preview)

**Date**: 2026-05-16
**Author**: Auto-generated, Wave-13 sweep (forward-looking Q3-2026 preview)
**Prior report**: v19 (2026-05-16, Wave-11 + Wave-12 sweep)
**Cadence**: forward-looking, intra-cycle preview between v19 and the formal post-Q3 v21 review
**Next formal review**: 2026-09-01 (Q3-2026 stress test cycle)

---

## 1. Summary

Report v20 is the first forward-looking preview written *into* the Wave-13 build window rather than after it. It is explicitly **not a promotion document**: no strategy is upgraded, demoted, or reallocated in v20. Its purpose is to fix the gates, the watch list, and the methodology constraints that the **2026-09-01 Q3 stress cycle** will run against, and to give the desk an honest read on which Wave-13 candidates have a credible path through that cycle versus which are likely to drop straight to the anti-alpha list.

Two new strategies entered the Wave-13 pipeline in the four-week window since v19. **Cross-sectional momentum**, originally introduced as a B_VALIDATED candidate in W12-25 and re-stated in v19 §2.3, has been re-instrumented under the v20 walk-forward and deflated-Sharpe gates and is now the strongest candidate for promotion to live B_VALIDATED allocation on 2026-09-01. **IV-realized-vol arbitrage** (W13-21), which trades Polymarket binary-implied volatility against same-window options-implied realized volatility, enters the pipeline at **CONDITIONAL** with `SHOULD_DEPLOY=False`. It depends on three Wave-13 verdicts (W13-31 fee normalisation across CBOE and Polymarket, W13-32 cross-resolution time alignment between options expiry and Polymarket resolution, and W13-33 realised-vol windowing audit) that are all pending. Methodology-side, **elastic-net** is now production-ready and is the default non-OLS method on `/regression/fit`. **Quantile** and **Bayesian** regression remain pending live test (W12-23, W12-24 still in progress) and will not gate the Q3 cycle. The walk-forward framework (W12-29) and the deflated Sharpe gate (W11-53) become **hard requirements** for any v20+ promotion, as flagged in v19 §6.

The live book composition is therefore **unchanged** in v20: one A_STRUCTURAL (calendar lambda-ratio) plus four B_VALIDATED legs, aggregate target Sharpe +1.2 to +1.5 on an 8% target-vol risk budget. Aggregate capacity sits at **$220k to $260k notional** before marginal slippage degrades the next dollar, consistent with v18 and v19. The Q3-2026 stress cycle is the first cycle in the project's history that will run against four full quarters of Polymarket data without the data-span caveat that has constrained every prior report.

---

## 2. Wave-13 Strategies Entering Pipeline

### 2.1 Cross-sectional prediction-market momentum (W12-25, re-stated)

Cross-sectional momentum was introduced in v19 §2.3 at initial Sharpe ~1.05 across three quarters with a marginal-but-acceptable DSR p ~ 0.04. In v20 it is re-instrumented under the strict v20 gates and re-stated as the **leading B_VALIDATED candidate** for live promotion on 2026-09-01.

The signal ranks the prediction-market universe by trailing 30-day log-odds drift, longs the top quintile, shorts the bottom quintile, and rebalances weekly. Wave-13 added two improvements over the v19 specification:

- **Universe filter (W13-12)**: contracts must have minimum aggregate $5k notional traded in the trailing 14 days and a resolution date at least 21 days forward. This drops the universe from approximately 2,400 active contracts to approximately 850 and removes the survivorship-illusion tail.
- **Stem-concentration cap (W13-13)**: borrowing the calendar lambda-ratio rule, no single resolution stem may contribute more than 30% of either quintile. Pre-cap, the long quintile concentrated 47% on election stems during 2025Q4, which biased the Sharpe high.

Under the W13-13 cap and the W13-12 universe filter, the re-run Sharpe over the same three quarters is **0.92** (down from the pre-cap 1.05). DSR p moves to **0.06**, fractionally above the 0.05 gate. This is the central uncertainty for Q3-2026: a fourth quarter of positive walk-forward performance would pull the pooled DSR back under 0.05 and clear the gate; a flat or negative fourth quarter would keep the strategy at CONDITIONAL.

**Walk-forward picture**: under the W12-29 anchored walk-forward harness with a 60-day training window and a 21-day step, the strategy posts non-negative Sharpe in 11 of the 13 windows currently available, with the two negative windows clustered in 2025Q3 (a known regime-change period). The Q3-2026 cycle will add at least 4 more windows.

**Promotion verdict for 2026-09-01**: probable B_VALIDATED at 5% to 7% allocation, subject to the four-quarter gate clearing without sign flip and DSR p falling below 0.05.

### 2.2 IV-realized-vol arbitrage (W13-21)

The IV-realized-vol-arb strategy enters at **CONDITIONAL**. It is the first Wave-13 strategy that explicitly couples Polymarket prediction-market data to listed options data; every prior strategy has been pure Polymarket plus stock cross-section.

The construction: for each binary contract with a resolution date within 14 to 45 days, the Polymarket-implied volatility is derived from the contract's clipped log-odds time series under the standard Bachelier-style approximation (clipping epsilon = 0.01, log-returns, HAC standard errors with maxlags = 5 as per ADR-0006). That implied volatility is compared against the same-window options-implied realized volatility on the underlying. When the gap exceeds 1.5 vol points, the strategy longs the cheaper side and shorts the richer side, sized by the Kelly fraction of the gap divided by the variance of the gap distribution.

Why CONDITIONAL: three open verdicts gate the strategy.

- **W13-31 fee normalisation**: the CBOE bid-ask quoted in cents per contract is not directly comparable to the Polymarket fee structure quoted in basis points of notional. Until W13-31 closes with a reconciled per-trade cost model, the apparent edge is partly an accounting artefact.
- **W13-32 cross-resolution time alignment**: Polymarket resolutions fire at the resolution-source-feed timestamp; options expire at 16:00 ET. These differ by hours to days. Until W13-32 closes, the trade has an unbounded basis-risk leg.
- **W13-33 realised-vol windowing audit**: the realized-vol window choice (5-day, 10-day, 21-day) materially affects which side appears mispriced. Without a windowing audit the Sharpe is not stable across reasonable specifications.

Pre-verdict naive Sharpe estimate sits at approximately 1.6 over the limited 9-month back-test window, but this is not a deployable number until the three verdicts close. The strategy will not be evaluated for promotion in the Q3-2026 cycle; the earliest realistic promotion window is the v22 / 2026Q4 cycle.

---

## 3. Quant Methodology Expansion

### 3.1 Elastic-net (W11-27, W12-13) — production-ready

Elastic-net is now the default non-OLS method on `/regression/fit`. The `method=enet` path is wired through `pfm.regression.fit_enet` with cross-validated alpha and l1_ratio selection (5-fold time-series CV, no leakage). The endpoint returns the same response schema as OLS plus an `enet_params` block reporting the selected alpha, l1_ratio, and the per-fold validation R-squared. Promotion-pipeline rule: any new candidate strategy that reports OLS coefficients must also report the elastic-net comparison, and any factor with elastic-net coefficient shrunk to zero must be flagged in the response.

### 3.2 Quantile regression (W12-23) — pending live test

Quantile regression remains under development. The module `pfm.regression.fit_quantile` exists but is not yet exposed via the public endpoint. The intended use is robustness checks on the calendar lambda-ratio fit, where conditional-median behaviour around the resolution-date elbow differs materially from conditional-mean behaviour. Earliest production date: 2026-08-15, pre-Q3 cycle.

### 3.3 Bayesian regression (W12-24) — pending live test

Bayesian regression (Normal-Inverse-Gamma conjugate prior plus weakly informative ridge) is similarly pending. The intent is to surface posterior credible intervals on factor loadings rather than the frequentist HAC CIs that are currently the only option. Earliest production date: 2026-09-30, post-Q3 cycle. Bayesian regression will not gate the Q3 cycle.

### 3.4 Walk-forward + deflated Sharpe — hard gates

Both `pfm.quant.walk_forward` (W12-29) and `pfm.quant.deflated_sharpe` (W11-53) are now **hard requirements** for any v20-and-later promotion. A candidate that fails either gate cannot advance past CONDITIONAL regardless of its naive in-sample Sharpe. The walk-forward requirement is a minimum of 5 disjoint windows with non-negative Sharpe; the deflated Sharpe requirement is p < 0.05 under the Bailey-Lopez de Prado deflation that corrects for the current candidate-set cardinality of 4,499 pairs.

---

## 4. Capacity Ladder Review per Deployable

The capacity figures in v19 §8 are re-verified under Wave-13 order-book conditions and are unchanged except where flagged:

- **Calendar lambda-ratio**: hard cap **$80k notional aggregate**. Verified under Wave-13 depth audit (W13-08). No change.
- **Election binary momentum**: hard cap **$50k notional**. The 2026 election cycle re-opened depth on this strategy; if 2026 election-cycle depth holds through Q3, capacity may revise up to $60k in v21.
- **Fed-decision straddle proxy**: soft cap **$40k notional per FOMC cycle**. No change.
- **Sports-event mean reversion**: hard cap **$30k notional aggregate**. The Wave-13 sports-feed audit (W13-09) confirmed that final-hour actionable depth remains in the $3k to $5k range per contract; limit-only execution remains mandatory.
- **Earnings-surprise odds vs IV**: soft cap **$25k notional**. The liquid-name set expanded to **8 names** from 6 in v19 (W13-10 added TSLA and AAPL); per-name cap holds at $4k to $5k. Aggregate cap revises up to **$32k** in v21 if the 8-name set holds through Q3.

**Aggregate live capacity** is therefore approximately **$220k to $260k** at v20 publication, with a credible path to **$240k to $280k** at v21 if the election-cycle and earnings-name expansions both hold through Q3.

---

## 5. Q3-2026 Stress Test Cycle (scheduled 2026-09-01)

The Q3 cycle is the first cycle that operates entirely inside the strict v17 / v18 / v19 gate without the data-span caveat. The four-quarter window covers 2025Q3 through 2026Q2 inclusive. The cycle scope:

1. **Re-stress the A_STRUCTURAL survivor** (calendar lambda-ratio) under the v20 walk-forward gate. Expectation: hold A_STRUCTURAL. Kill rule: drop-dominant-stem Sharpe below 0.5 in any 30-day re-check.
2. **Re-stress all four B_VALIDATED legs** (election binary momentum, Fed-decision straddle proxy, sports-event mean reversion, earnings-surprise odds vs IV). Each must clear the 4-quarter Sharpe stability test, BH-FDR correction, and DSR p < 0.05.
3. **Run the 4-quarter gate on cross-sectional momentum (W12-25)**. Verdict expected on 2026-09-01.
4. **Run the PCA-residual cross-theme basket re-check** at the 30-day mandatory cadence flagged in v19 §9 item 6. Promote to A_GOLD if OOS Sharpe holds above 1.5 with drop-top-3 above 0.5; demote to C_TENTATIVE otherwise.
5. **Re-confirm favorites bias** at the 2026Q3 paper-only verdict. Promote back to A_GOLD if structural; archive to anti-alpha if the regime-driven hypothesis holds.

The cycle output will be `docs/alpha-report-v21.md`, scheduled for publication 2026-09-08.

---

## 6. Anti-Alpha Addition Watch List

The following are currently deployable or candidate strategies that warrant **close monitoring** for Q3 demotion. These are **not** demotions in v20; they are flagged so the Q3 cycle's audit team knows which strategies have the weakest evidence and where to allocate audit time.

- **Sports-event mean reversion (B_VALIDATED)**: net Sharpe 0.8 is the lowest of the B_VALIDATED set. Liquidity windows are narrow and slippage assumptions are the most fragile in the book. If 2026Q2 OOS Sharpe falls below 0.5 net, this is a demotion candidate.
- **Earnings-surprise odds vs IV (B_VALIDATED)**: sample remains thin even after the W13-10 expansion to 8 names. A 2026Q2 OOS Sharpe collapse on any subset of the new names would warrant a re-fit, not a demotion, but persistent failure across both new names triggers demotion.
- **Favorites bias (B_VALIDATED paper-only)**: the central case is that this is regime-driven (see `memory/project_favorites_bias_alpha.md`). 2026Q3 is the verdict cycle; allocation is already halved.
- **PCA-residual cross-theme basket (B_VALIDATED 15%)**: held at 15% in v18 / v19 pending the 30-day re-check. The single largest cycle-over-cycle allocation in the book; therefore the highest single-strategy concentration risk.
- **Cross-sectional momentum (CONDITIONAL until 2026-09-01)**: the post-cap Sharpe of 0.92 and DSR p of 0.06 are at the edge of the gate. A flat or negative Q3 keeps it CONDITIONAL.
- **IV-realized-vol arbitrage (CONDITIONAL)**: gated by three open verdicts. Not in the Q3 cycle; flagged so audit time is not wasted on it.

The v18 / v19 anti-alpha list (recession-odds defensive long, crypto-ETF approval drift, senate-control short-vol, geopolitical-conflict oil long, BTC midpoint-latency arb) carries forward unchanged. The strict rule still applies: **do not re-pitch any of these as wins.** Any positive single-window backtest on those themes is assumed regime-driven until proven otherwise via the 4-quarter gate.

---

## 7. Tooling Updates Since v19

The following endpoints landed in Wave-13 and are now live on the FastAPI app:

- `GET /regression/methods` (W11-27, W12-13) — now reports `ols` and `enet` as production; `quantile` and `bayes` are exposed as `status: pending` with target dates.
- `GET /strategies/watch-list` (W13-19) — new programmatic source of the Q3 watch list documented in v20 §6. Returns strategy name, tier, watch-flag reason, and Q3 verdict cadence.
- `GET /strategies/walk-forward/{name}` (W13-20) — returns the walk-forward Sharpe history for a named strategy under the W12-29 framework. Currently supports calendar lambda-ratio, cross-sectional momentum, and the four B_VALIDATED legs.

---

## 8. Outlook for v21 (post-Q3 cycle)

The v21 report is the first that will be written entirely **inside** the strict gate. Three plausible outcomes, in rough probability order:

1. **Central case (~55%)**: cross-sectional momentum promotes to live B_VALIDATED at 5% to 7%, PCA-residual basket holds B_VALIDATED, favorites bias archives to anti-alpha, sports-event mean reversion holds. Book composition: 1 A_STRUCTURAL + 5 B_VALIDATED + several anti-alphas.
2. **Bull case (~25%)**: cross-sectional momentum promotes, PCA-residual basket promotes to A_GOLD on a strong Q3, favorites bias holds B_VALIDATED structural. Book composition: 1 A_STRUCTURAL + 1 A_GOLD + 4 to 5 B_VALIDATED.
3. **Bear case (~20%)**: cross-sectional momentum stays CONDITIONAL, PCA-residual basket demotes to C_TENTATIVE on a Sharpe collapse, sports-event mean reversion demotes. Book composition: 1 A_STRUCTURAL + 3 B_VALIDATED.

The IV-realized-vol-arb strategy is not in any v21 scenario; earliest plausible promotion window is v22.

---

## References

- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management* 40 (5).
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate." *J. R. Stat. Soc. B* 57.
- Politis, D. & Romano, J. (1994). Stationary block bootstrap.
- Harvey, C., Liu, Y., Zhu, H. (2016). "...and the cross-section of expected returns." *Rev. Financial Studies* 29 (1).
- Zou, H. & Hastie, T. (2005). "Regularization and variable selection via the elastic net." *J. R. Stat. Soc. B* 67 (2).
- Internal: `docs/alpha-report-v19.md`, `docs/alpha-report-v18.md`, ADR-0010 (anti-alpha rule), ADR-0011 (cache stampede single-flight), `memory/project_wave5_stress_test_findings.md`, `memory/project_favorites_bias_alpha.md`, `memory/project_btc_latency_arb_dead.md`.
