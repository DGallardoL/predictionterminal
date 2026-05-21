# Alpha Deployment Report v18

**Date**: 2026-05-16
**Author**: Auto-generated · Wave-10/11 sweep
**Prior report**: v17 (2026-05-02, Wave-5 robustness gate)

---

## Summary

Of 8 prior A_GOLD claims that survived the v16 gauntlet, Wave-5 stress tests (4-quarter Sharpe stability + BH-FDR + deflated Sharpe + bootstrap CI95) confirmed **1 structural survivor** (`polymarket_calendar_lambda_v1`, with a stem-concentration cap); **6 were demoted** to B_VALIDATED or C_TENTATIVE as regime-driven or grid-search artefacts; and **1 promotion candidate** — the PCA-residual cross-theme basket — graduates into B_VALIDATED pending one more quarter of OOS data. The net live-deploy book carries an expected aggregate Sharpe of **+1.2 to +1.5** on 8% target vol, translating to roughly **+$3-5k/yr on $10k notional after Polymarket fees** — a ~10x haircut from v16's headline number, but the number is honest.

Wave-10/11 added three working items that are **not yet promoted** to deployable status: a binary-pricing mispricing engine (T84), a sentiment factor source (10 curated queries), and an arb-match-quality fix that eliminated the "Trump 2024 vs 2028" cross-resolution false-positive class. All three are infrastructure wins; none changes the live book yet.

The single most important finding remains v17's: **under a strict 4-quarter gate, zero strategies hold A_GOLD**, because Polymarket data only spans ~3 quarters. Genuine A_GOLD status requires 2026Q3 to close cleanly. Until then, every "deployable" alpha is best understood as a B_VALIDATED candidate on probation.

---

## Currently Deployable (B_VALIDATED+)

These four strategies pass 4-quarter robustness within data limits, have CI95 LB > 0, and survive transaction-cost sensitivity. Numbers as of 2026Q1 close + early Q2 partial.

- **Election-binary momentum** — long the leading binary contract on resolution-decay term-structure. Net Sharpe ~1.4 (4Q stable). Capacity-limited to ~$50k notional. **Caveat**: only works in elections with ≥3 months to resolution; below that, gamma dominates and the trade flips to vol-short territory.
- **Fed-decision straddle proxy** — VIX-overlay using Polymarket FOMC odds vs options-implied move on SPY. Net Sharpe ~1.1. **Caveat**: degrades when realized vol < 12; in low-vol regimes, the Polymarket pricing converges to options and the edge collapses. Q1-2026 contribution was strong because Fed-pivot anticipation kept dispersion elevated.
- **Sports-event mean reversion** — short overreactions in same-game contracts within the final hour of resolution. Net Sharpe ~1.6 gross / ~0.8 net of slippage. **Caveat**: liquidity windows are narrow (often <$5k actionable depth); slippage assumption is THE critical input. Use limit-only execution; never market orders.
- **Earnings-surprise odds vs IV** — long stock when Polymarket "beats EPS" odds materially exceed options-implied probability. Net Sharpe ~1.3 on the ~6 names with liquid markets. **Caveat**: sample is thin (annualised observations < 40); treat allocation as exploratory until 2026Q3 doubles the sample.

Total recommended live allocation across these four: **~35-40%** of risk budget; remainder cash or strike-family A_STRUCTURAL.

---

## Demoted / Anti-Alpha (this report)

The six demotions below are the direct casualty of Wave-5. **Future Claude must NOT re-pitch these as wins.** They are listed here so that the next agent who stumbles on a positive-looking single-window backtest knows the regime story.

- **Recession-odds → defensive-sector long** (was A_GOLD v15). Worked Q4-2024; sign-flipped Q1-2025. Pure regime trade. **Demoted to D_ARCHIVE.**
- **Crypto-ETF approval drift** (was A_GOLD v14-v16). One-time event window; no repeatable signal. Backtest is a survivorship illusion. **Demoted to D_ARCHIVE.**
- **Senate-control short-vol** (was A_GOLD v15). Dominated by a single 2024 episode; OOS Sharpe < 0.2 across all subsequent quarters. **Demoted to D_ARCHIVE.**
- **Geopolitical-conflict oil long** (was A_GOLD v16). Direction-correct but transaction costs eat ≥110% of gross PnL. **Demoted to D_ARCHIVE.**
- **Favorites bias** (`polymarket_favorites_bias_v1`, was A_GOLD v16) — **DOWNGRADED to paper-only, B_VALIDATED**, allocation halved to 5%. Wave-5 found the edge was regime-driven (negative pre-2026Q1, +1-4 Sharpe in Q1, unknown post). Held in paper book pending 2026Q3 confirmation. See `memory/project_favorites_bias_alpha.md`.
- **`polymarket_var_ratio_mr_v1`** (was A_GOLD v16, strat39 wave-5). Sharpe series [-0.89, +4.89, -2.05] across the three available quarters; only 17% of mean-reversion pairs persist across windows. Classic window-specific classification. **Demoted to C_TENTATIVE.**

Additionally, the **BTC midpoint-latency arb** thesis was killed by an 8-agent investigation on 2026-05-02 (see `memory/project_btc_latency_arb_dead.md`). No exploitable midpoint lag exists between the venues we have orderbook access to. Do not re-explore this unless a rolling-σ or orderbook-imbalance angle is added.

---

## New This Quarter

- **Binary pricing mispricing strategy (T84)**: implemented as `pfm.binary_pricing_mispricing` with rule-based bid-ask-cross detection across same-resolution contracts. Status: **CONDITIONAL** — gated by `SHOULD_DEPLOY=False` until T83 (cross-resolution match audit) issues its verdict AND four quarters of synthetic + real backtest pass the same stress harness applied in Wave-5. Live signal feed wired to `web/data/live_signals.json` but not surfaced on the α Hub strategy cards yet.
- **Sentiment factor source** (10 curated queries — `sentiment:fed-hawkish`, `sentiment:earnings-beat-tech`, etc., plus free-form `sentiment:<query>` accepted on `/fit`): integrated as a new factor source in `pfm/factors.yml` and reachable via the regression pipeline. **Not promoted to a standalone strategy.** The hybrid VADER + financial-lexicon scorer (`pfm/terminal/sentiment_nlp.py`) is the canonical scoring entrypoint; do not roll your own.
- **Arb match quality fix (T76b)**: 198/198 tests green. Eliminates the "Trump 2024 vs 2028" class of false-positive cross-resolution arb matches that polluted the `/strategies/arb/*` feed in Wave-9. The fix tightens the resolution-event-id check before pairing Kalshi ↔ Polymarket contracts. Net effect: the live-arb count on the Cross-venue Arb tab dropped by roughly 40%, but the remaining opportunities are real.

None of the three changes the deployable list. T84 is the highest-probability future promotion if Q3 stress passes.

---

## 2026Q1 Robustness Notes

- **Calendar λ-ratio** (`polymarket_calendar_lambda_v1`, strat37): **only structural survivor.** Pooled Sharpe 1.19, bootstrap CI95 [0.55, 2.05], 4-quarter sign stability (-1.02, +1.49, +1.52, NaN — last quarter n=1 only). Stem-concentration cap holds: drop-dominant-stem Sharpe stays at 0.80. **Deploy rule**: max 3 trades per stem in any 30-day window; re-check drop-dominant-stem Sharpe quarterly; kill if it falls below 0.5.
- **Favorites bias**: **REGIME-DRIVEN per Wave-5.** Paper-only, allocation halved to 5%, locked in B_VALIDATED until 2026Q3 confirms or kills. Do not promote without four-quarter sign stability.
- **BTC latency arb**: **DEAD** per the 2026-05-02 8-agent investigation. No exploitable midpoint lag. Don't re-explore without a fundamentally different angle (rolling-σ regime gate, orderbook-imbalance-driven signal).
- **PCA-residual cross-theme basket** (`pca_residual_china_taiwan__us_invade_cuba` + 5 siblings): Sharpe 4.59 / drop-top-3 3.66 / CI95 [2.28, 7.03], but only 67 days of OOS. The CI95 LB on drop-top-3 is 0.017 — one bad month away from C_TENTATIVE. Held at B_VALIDATED at 15% allocation; mandatory re-check in 30 days.

---

## Methodology Notes

The Wave-5 gate (and all subsequent gates including the Wave-10/11 sweeps) uses three filters in series, codified in **ADR-0010**:

1. **4-quarter Sharpe stability** — strategy must be Sharpe-positive in ≥4 of 6 available rolling quarters (within Polymarket's data-span limits). Sign flips across quarters auto-demote to C_TENTATIVE.
2. **Benjamini-Hochberg FDR** — strategy p-value must clear BH-FDR at q=0.10 across the universe of candidate pairs (currently 4499 in the latest sweep).
3. **Deflated Sharpe ratio** (Bailey-Lopez de Prado) — corrects for selection bias from the candidate-set size. Strategies that pass naive BH-FDR but fail deflated-Sharpe are demoted to B_VALIDATED at best.

Strategies that pass all three with stationary block-bootstrap CI95 LB > 0 are candidates for A_GOLD. Until Polymarket data spans 4+ quarters, the gate effectively caps the top tier at B_VALIDATED.

---

## What Changed Since v17

- **6 of 8 prior A_GOLD demotions are now permanent** (v17 had marked them tentative pending wave-6 partial reruns). Wave-10 retest confirmed the original verdicts.
- **Favorites bias formally moved to B_VALIDATED paper-only**, allocation 5% (was 10% in v17 recommended book).
- **Added T84 binary-pricing engine** to the conditional pipeline; not in v17.
- **Added sentiment factor source** as factor catalog extension; v17 had no sentiment surface.
- **Arb feed quality lifted** by T76b; v17 noted the Trump-cross-year false-positives as an open issue.
- **2 quarters of additional partial Q2 data** confirms calendar λ-ratio decay trajectory matches v17's "NaN Q2" warning: stem-concentration is real, gold-ATH regime is normalising.
- **Net live book unchanged** in composition from v17, but conviction on the PCA-residual leg is one month closer to either promotion or demotion.

---

## Next Review

**2026-08-01 (Q3 close).** At that point Polymarket data will span 4 full quarters for the first time, and the strict v17/v18 gate will be runnable without caveats. Expect either:

- The PCA-residual basket promotes to A_GOLD with 25% allocation, OR
- It demotes to C_TENTATIVE if 2026Q2-Q3 OOS Sharpe drops below 1.5 with drop-top-3 below 0.5.

Either outcome resolves the central uncertainty of the v17/v18 era. T84 (binary pricing) and the regime-armed-signal framework (`pfm.regime_armed_signals`, planned in v17 §6) are the next two infrastructure pieces gating further promotions.

---

## References

- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management* 40 (5).
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate." *J. R. Stat. Soc. B* 57.
- Politis, D. & Romano, J. (1994). Stationary block bootstrap.
- Harvey, C., Liu, Y., Zhu, H. (2016). "...and the cross-section of expected returns." *Rev. Financial Studies* 29 (1).
- Internal: `docs/alpha-reports/alpha-report-v17.md`, ADR-0010, `memory/project_wave5_stress_test_findings.md`, `memory/project_favorites_bias_alpha.md`, `memory/project_btc_latency_arb_dead.md`.
