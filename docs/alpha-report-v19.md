# Alpha Deployment Report v19

**Date**: 2026-05-16
**Author**: Auto-generated, Wave-11 + Wave-12 sweep (forward-looking)
**Prior report**: v18 (2026-05-16, Wave-10/11 sweep)
**Cadence**: 4-week delta over v18
**Next review**: 2026-09-01

---

## 1. Summary

Report v19 is a forward-looking successor to v18. The book composition is **unchanged in live allocation** since v18, but the infrastructure under the book has hardened materially: Wave-11 closed the deflated-Sharpe enforcement gap (W11-53), Wave-12 shipped walk-forward and bootstrap-CI tooling (W12-29, W12-27), and the calendar lambda-ratio strategy (`polymarket_calendar_lambda_v1`) was re-confirmed as the sole **A_STRUCTURAL** survivor under a synthetic-DGP 4-regime test (W11-55) that recovers expected directionality across each of the four constructed regimes. A new **B_VALIDATED candidate** entered the pipeline this cycle, the cross-sectional prediction-market momentum strategy (W12-25), and now awaits the standard 4-quarter stress gate. No new anti-alphas were demoted this cycle; the binary pricing mispricing strategy (T84) remains CONDITIONAL with `SHOULD_DEPLOY=False` pending T81/T82/T83 verdicts.

The headline operational improvement is the **arb matching quality stress audit** (T76b follow-up): zero false positives in the v19 stress audit versus roughly five in v18. That removes the largest known source of noise from the live α Hub Cross-venue Arb tab and improves trust in the live signal feed that backs `web/data/live_signals.json`.

The net live-deploy book continues to carry an expected aggregate Sharpe of **+1.2 to +1.5** on an 8% target-vol budget, translating to roughly **+$3,000 to +$5,000 per year on $10k notional after Polymarket fees**. Aggregate capacity across the five deployable strategies sums to approximately **$220k to $260k notional** before slippage degrades the marginal trade.

---

## 2. Changes Since v18 (4 weeks delta)

### 2.1 Binary pricing mispricing (T84)

T84 remains **CONDITIONAL** with the `SHOULD_DEPLOY=False` flag intact. The strategy depends on the T81 (resolution-event-id reconciliation), T82 (cross-venue fee normalisation), and T83 (cross-resolution match audit) verdicts. As of v19 all three are **pending**, so the binary pricing engine is held at infrastructure-ready / capital-not-deployed status. The live signal feed continues to populate `web/data/live_signals.json` but is not surfaced on the α Hub strategy cards.

### 2.2 Calendar lambda-ratio (W11-55)

`polymarket_calendar_lambda_v1` was **re-confirmed as A_STRUCTURAL**. The W11-55 work added a synthetic-DGP 4-regime stress test that constructs four distinct decay-curve regimes (steep-near, flat, hump-shape, inverse-hump) and verifies the signal recovers the **expected directionality** under each regime. The strategy passed in all four. This is the strongest piece of structural evidence the book has and is the reason the calendar lambda-ratio remains the only A-tier deployable.

### 2.3 Cross-sectional prediction-market momentum (W12-25)

A **NEW B_VALIDATED candidate** entered the pipeline. Cross-sectional momentum on the prediction-market universe ranks contracts by trailing 30-day log-odds drift, longs the top quintile, shorts the bottom, and rebalances weekly. Initial Sharpe estimate is approximately 1.05 on the back-test window covered by Polymarket data, with no sign flip across the three quarters of available data. The strategy is **awaiting the 4-quarter stress gate** before any allocation. Deflated-Sharpe pre-screen passes at p ~ 0.04, marginal but acceptable.

### 2.4 Arb matching quality (T76b follow-up)

The v19 stress audit produced **0 false positives** across the same audit harness that flagged ~5 false positives in v18. This is a direct consequence of the resolution-event-id tightening shipped in T76b, plus a secondary filter on disjoint resolution-date windows. The Cross-venue Arb live feed is now trusted enough that the planned Live Edge sub-tab (currently designed, not built) becomes a reasonable next UI investment.

---

## 3. Currently Deployable

Five strategies remain in the live or paper-live book. Aggregate target allocation across all five sits at approximately **35-40% of risk budget**; the remainder is cash or held in strike-family A_STRUCTURAL reserve.

| Strategy | Tier | Net Sharpe | Capacity ($ notional) | Allocation |
|---|---|---|---|---|
| Calendar lambda-ratio (`polymarket_calendar_lambda_v1`) | A_STRUCTURAL | 1.19 (pooled), CI95 [0.55, 2.05] | ~$80k | 12% |
| Election-binary momentum | B_VALIDATED | ~1.4 (4Q stable) | ~$50k | 8% |
| Fed-decision straddle proxy | B_VALIDATED | ~1.1 | ~$40k | 7% |
| Sports-event mean reversion | B_VALIDATED | ~0.8 net | ~$30k | 5% |
| Earnings-surprise odds vs IV | B_VALIDATED | ~1.3 | ~$25k | 4% |

**Caveats summary** (full caveats live in v18, repeated in shorter form here):

- **Calendar lambda-ratio (A_STRUCTURAL):** stem-concentration cap holds, drop-dominant-stem Sharpe remains 0.80. Deploy rule: max 3 trades per stem in any 30-day window. Kill rule: if drop-dominant-stem Sharpe falls below 0.5 in any 30-day re-check.
- **Election binary momentum (B_VALIDATED):** only valid with at least 3 months to resolution. Below that, gamma dominates and the trade flips sign.
- **Fed-decision straddle proxy (B_VALIDATED):** degrades when realised vol is below 12. Edge collapses in low-vol regimes.
- **Sports event mean reversion (B_VALIDATED):** narrow liquidity window of typically less than $5k actionable depth. Limit-only execution mandatory; market orders forbidden.
- **Earnings surprise odds vs IV (B_VALIDATED):** approximately 6 names only. Treat allocation as exploratory until 2026Q3 doubles the sample.

---

## 4. Anti-Alpha (additions)

**None added this cycle.** The v18 anti-alpha list (recession-odds defensive long, crypto-ETF approval drift, senate-control short-vol, geopolitical-conflict oil long, plus the BTC midpoint-latency arb thesis killed on 2026-05-02) carries forward unchanged. Favorites bias remains downgraded to B_VALIDATED paper-only at 5% allocation pending 2026Q3 confirmation.

The strict v17/v18 rule still applies: **do not re-pitch any of the demoted A_GOLD strategies from v15 to v16 as wins.** Any positive single-window backtest on those themes should be assumed regime-driven until proven otherwise via the 4-quarter gate.

---

## 5. Pending Stress (CONDITIONAL)

Two strategies are in the conditional bucket:

- **Binary pricing mispricing (T84):** gated by `SHOULD_DEPLOY=False` until T81 / T82 / T83 verdicts close. Highest-probability future promotion if Q3 stress passes.
- **Cross-sectional momentum (W12-25):** B_VALIDATED on the initial sweep, but the 4-quarter stress harness has not yet run because the W11-52 `validate_alphas_4q.py` script needs a wider universe definition file before it can be pointed at this strategy. Target: cleared by 2026-06-30.

Neither strategy contributes to live capital allocation in v19.

---

## 6. Methodology Updates

### 6.1 Deflated Sharpe enforcement (W11-53)

Every claimed result in v19 and forward must satisfy **DSR p < 0.05** under the Bailey-Lopez de Prado deflation that corrects for the cardinality of the candidate set (currently 4499 pairs). The enforcement is wired into the v19 promotion pipeline via `pfm.quant.deflated_sharpe`. Strategies that pass naive Sharpe but fail DSR are demoted to B_VALIDATED at best, never A-tier.

### 6.2 Walk-forward (W12-29)

The walk-forward framework `pfm.quant.walk_forward` is now available. It supports both anchored and rolling-origin schemes, with configurable training window, step size, and minimum out-of-sample bar count. The framework is not yet a hard requirement for v19 promotions but will be enforced from v20 onwards. The expectation is that every B_VALIDATED claim must show non-negative walk-forward Sharpe across at least 5 disjoint windows before A-tier consideration.

### 6.3 Bootstrap CIs (W12-27)

Stationary block-bootstrap confidence intervals are now available via `pfm.quant.bootstrap_sharpe`. The block length defaults to the Politis-Romano-recommended geometric draw with mean block length proportional to n^(1/3). CI95 LB > 0 remains the operational filter for B_VALIDATED status.

---

## 7. Tooling Updates

The following endpoints landed in Wave-11 to Wave-12 and are now live on the FastAPI app:

- `GET /strategies/deployable-list` (W11-24) — programmatic source of the deployable book. Returns tier, Sharpe estimate, capacity, allocation, and caveats. Now consumed by the α Hub cards.
- `GET /strategies/anti-alpha-list` (W11-23) — programmatic source of the anti-alpha set. Guards against re-pitching demoted ideas.
- `GET /regression/methods` (W11-27, W12-13) — lists available regression methods. As of v19: **ols** and **enet** are supported and exposed via the endpoint; **quantile** and **bayes** are pending (W12-23 and W12-24 are in progress in the active-edits ledger). The methods endpoint returns capability metadata, not just names, so the frontend can render only the supported subset.

---

## 8. Capacity Review per Deployable

Capacity figures below are post-fee, slippage-adjusted on Polymarket order-book depth as observed during 2026Q1 and partial Q2.

- **Calendar lambda-ratio**: hard cap **$80,000 notional aggregate**, with the 3-trades-per-stem-per-30-days rule limiting any single stem to roughly $12,000 to $15,000. Beyond $80k the dominant-stem concentration risk reasserts and drop-dominant-stem Sharpe collapses.
- **Election binary momentum**: hard cap **$50,000 notional**. Above this level the limit-order fill rate drops sharply and the term-structure edge is eaten by slippage.
- **Fed-decision straddle proxy**: soft cap **$40,000 notional** per FOMC cycle. The SPY-options leg is uncapped; the Polymarket FOMC-odds leg is the binding constraint.
- **Sports-event mean reversion**: hard cap **$30,000 notional aggregate across active games**, with the final-hour liquidity window typically supporting no more than $3,000 to $5,000 actionable per contract. Limit-only execution is mandatory.
- **Earnings-surprise odds vs IV**: soft cap **$25,000 notional** spread across the approximately six liquid earnings names. Per-name cap of about $4,000 to $5,000 to avoid sweeping the book.

**Aggregate live capacity** is therefore approximately **$220k to $260k** before marginal slippage degrades the next dollar. This is consistent with the v18 number and supports the same 8% target-vol risk budget.

---

## 9. Q2-2026 Plan

The plan for 2026Q2, in priority order:

1. **Close T81 / T82 / T83.** These are the gating verdicts for T84 binary pricing mispricing. If they pass, T84 advances to a 4-quarter stress run.
2. **Stress cross-sectional momentum (W12-25)** through the 4-quarter harness. Promote to B_VALIDATED live allocation if Sharpe remains positive across the three available quarters and DSR p < 0.05 with bootstrap CI95 LB > 0.
3. **Complete W12-23 (quantile regression)** and **W12-24 (Bayesian regression)** so that `/regression/methods` exposes the full method set. Re-run the calendar lambda-ratio fit under quantile regression as a robustness check.
4. **Wire walk-forward (W12-29) into the v20 promotion pipeline** as a hard requirement rather than an optional check.
5. **Build the Live Edge sub-tab** in `web/index.html` (currently designed, not built) now that the arb feed has 0 false positives in stress audit. Connect to the `/strategies/arb/stream` SSE.
6. **Re-run the PCA-residual cross-theme basket** (held at B_VALIDATED 15% allocation in v18) at the 30-day mandatory re-check. If 2026Q2 OOS Sharpe holds above 1.5 with drop-top-3 above 0.5, advance to A_GOLD candidacy for the v20 cycle.
7. **Refresh the factor catalog totals** in `CLAUDE.md` and `factors.yml` if Wave-12 adds new factor slugs.

The single biggest unknown remains whether 2026Q3 will be the first quarter that lets the strict 4-quarter gate run without caveats. The data-span constraint is structural and cannot be accelerated.

---

## 10. Next Review

**2026-09-01.** This is one month after the 2026Q3 close, giving enough lag for trade settlement and resolution to clean up before the v20 sweep. At that point Polymarket data will span 4 full quarters for the first time, and the strict v17/v18/v19 gate will be runnable without the data-span caveat. Expect either:

- The PCA-residual basket promotes to A_GOLD with up to 25% allocation, the calendar lambda-ratio retains A_STRUCTURAL, and cross-sectional momentum enters live B_VALIDATED allocation, OR
- The PCA-residual basket demotes to C_TENTATIVE on a Sharpe collapse, and the live book contracts toward a single A_STRUCTURAL plus the existing four B_VALIDATED legs.

Either outcome resolves the central uncertainty of the v17 / v18 / v19 era and clears the way for v20 to be the first report that operates entirely inside the strict gate without data-span apologies.

---

## References

- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management* 40 (5).
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate." *J. R. Stat. Soc. B* 57.
- Politis, D. & Romano, J. (1994). Stationary block bootstrap.
- Harvey, C., Liu, Y., Zhu, H. (2016). "...and the cross-section of expected returns." *Rev. Financial Studies* 29 (1).
- Internal: `docs/alpha-report-v18.md`, ADR-0010 (anti-alpha rule), ADR-0011 (cache stampede single-flight), `memory/project_wave5_stress_test_findings.md`, `memory/project_favorites_bias_alpha.md`, `memory/project_btc_latency_arb_dead.md`.
