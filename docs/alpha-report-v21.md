# Alpha Report v21 — Wave-6 Robustness & Microstructure Reckoning

**Date:** 2026-05-19
**Prior reports:** v20 (Q3-2026 forward preview, Wave-13), v17 (Wave-5 honest reckoning)
**Purpose:** Document what survived Wave-6 stress tests, what got demoted, and what was killed. Replace tier labels that didn't reflect the actual statistical evidence.

---

## 1 · Executive summary

This wave applied three new gates to every existing alpha and every newly-proposed microstructure strategy:

1. **BH-FDR** at q=0.05/0.10/0.20 on stored `perm_p` (caveat: current resolution ~50-150 perms; re-run with ≥10k recommended).
2. **Bootstrap-CI** — require `sharpe_ci_lo > 0 AND ci_width > 0.5`.
3. **Deflated Sharpe** (Bailey-Lopez de Prado 2014) — haircut for multiple-trial bias.

Plus, for the new microstructure strategies, **4-window stability** (β stable in sign across 4 disjoint 24h test periods).

### Headline outcomes

- **5 strategies promoted to A_STRUCTURAL** (3 anti-correlated French/Brazilian elections, 1 Fed-target pair, 1 Iran/Grenell).
- **4 strategies demoted from A_STRUCTURAL → B_VALIDATED** — all failed the bootstrap-CI gate (three never computed, one with `lo=0.0`). Three of the four are strike-family pairs the user might want to defend on theory grounds.
- **2 microstructure strategies deployed** (Roll filter, VPIN gate), but **VPIN demoted from A → B_VALIDATED** because its β is significant in only 1/4 windows — it's a real signal but level-thresholds are regime-driven.
- **Roll filter caught a self-inflicted bug** — using `interval=max` (117-152d) blacklisted markets whose noisy regime is historical, not current. Switched to hourly fidelity + 48h freshness gate. 2 false-positive pairs (SpaceX→MS, Viking) were un-blacklisted.
- **3 microstructure strategies confirmed dead** — kline-OFI (S1), 1Hz OBI mean-reversion (S5), single-factor VR (S6). Documented in memory so future Claude doesn't re-explore.

### Tier deltas

| Tier | Before | After | Δ |
|---|---:|---:|---:|
| A_STRUCTURAL | 4 | 5 | +1 net (5 in, 4 out) |
| B_VALIDATED | 23 | 22 | -1 net |
| C_TENTATIVE | 13 | 13 | 0 |
| D_RAW | 29 | 29 | 0 |
| **Total** | **69** | **69** | |

---

## 2 · The 5 new A_STRUCTURAL strategies

All five pass: BH-FDR q=0.05, bootstrap-CI strictly positive with width > 0.5, deflated-Sharpe ≥ 0.5.

| Pair | OOS Sharpe | Deflated | CI95 | Category |
|---|---:|---:|---|---|
| Clémence Guetté ↔ Tom Steyer | 8.17 | 7.86 | [4.95, 12.48] | Election anti-corr |
| Fed target 4.5% EOY ↔ no_fed_cuts_2026 | 6.91 | 6.58 | [3.31, 11.97] | **Macro strike family** |
| François Asselineau ↔ Roberto Sánchez | 6.64 | 6.36 | [2.93, 9.96] | Election anti-corr |
| Renan Santos ↔ US aliens | 6.34 | 6.07 | [4.07, 9.06] | Cross-theme |
| Richard Grenell ↔ US-Iran nuclear | 5.59 | 5.29 | [3.59, 8.15] | Geopolitical |

**Caveats:**
- 4 of 5 are intra-theme election anti-correlations — Sharpes this high almost certainly reflect a structural anti-correlation, not realised PnL after costs. Capacity is microscopic (<$50k notional). Treat as **paper-deployable** until the 4-quarter Sharpe stability check runs.
- The Fed target ↔ no_fed_cuts pair is the only one with a clear macro narrative and is the primary live-deployment candidate.

## 3 · Microstructure strategies tested (5 ideas, 2 deployable)

| # | Theory | Test | Status |
|---|---|---|---|
| S1 | Cont-Kukanov-Stoikov 2014 OFI | kline-1m OFI → r_{t+1m/5m/15m} | ✗ killed (β≈0, t<1, R²<0.1%) |
| S2 | Hasbrouck 1995 info-share | Kalshi vs Polymarket Fed twins | ✗ bidirectional Granger (no leader) |
| S3 | **Roll 1984 effective spread** | Per-leg ROll on 65 active arb pairs | ✓ **deployed**, 3 pairs blacklisted as bid-ask bounce illusions |
| S4 | **Easley-LdP-O'Hara 2012 VPIN** | VPIN_bulk → RV_{t+30m} on BTCUSDT | ✓ **deployed** as gate, but B_VALIDATED (regime-driven) |
| S5 | Cartea-Jaimungal OBI MR | OBI z @ 1Hz → r at 1/5/15/30s | ✗ MOMENTUM not MR at this cadence |
| S6 | Lo-MacKinlay 1988 VR | Single-factor VR on Fed factor | ✗ random walk; needs pairs |

### S3 Roll filter — deployment detail

- **Method:** for each Polymarket leg in current opportunities, pull hourly midpoint history (last ~1 month). Roll spread `s = 2·√(-Cov(Δp_t, Δp_{t-1}))`. Blacklist when `s > displayed_cross-venue_gap`.
- **Wave-6 bug fix:** original implementation used `fidelity=1440&interval=max` (117-152d daily history). Old regimes blacklisted markets that are clean today — 2 of 4 entries were false positives. Switched to `fidelity=60&interval=1m` (~700 hourly points) + 48h freshness gate on the cache.
- **Current blacklist (3 pairs):** Trump-Somaliland NR, CA-07 Mai Vang, FL-09 Chalifoux. Persisted across multiple fidelity tests.
- **Sidecar:** `arbstuff/research/microstructure_sidecar.py` rebuilds the Roll cache every 30 min.

### S4 VPIN — deployment + caveats

- **Bulk-volume classifier** (Φ(Δp/σ_Δp)) is the right method — direct `taker_buy_quote / quote_volume` does not predict on BTC (HFT/MM noise).
- **24h baseline:** β=+0.0028, t=+2.71, R²=5.8% predicting RV_{t+30m}.
- **Wave-6 4-window check:** β > 0 in 4/4 (no sign flip). |t| > 2 in **only 1/4** — fails A_STRUCTURAL. Tier set to B_VALIDATED.
- **Threshold revision:** WARN 0.45 kept (cross-window p90 median), HALT bumped 0.50 → 0.55 (cross-window p99 conservative ceiling — 0.50 was meaningless: 14.6% trigger rate in vol window vs 0% in calm windows).
- **Future work:** rolling 7d-p90/p99 thresholds so the gate measures deviation from current regime, not absolute volatility.

## 4 · A_STRUCTURAL demotions

These were labeled A_STRUCTURAL but fail bootstrap-CI:

| Pair | Reason |
|---|---|
| fed_target_40_eoy ↔ fed_target_45_eoy | `ci_lo = 0.0` — touches zero, fails strict gate |
| fed_cuts_10_2026 ↔ fed_cuts_7_2026 | no bootstrap CI computed |
| bitcoin_reach_by_december ↔ bitcoin_reach_by_december_2 | no bootstrap CI computed |
| btc_dip_15k ↔ btc_dip_35k | no bootstrap CI computed |

Three of four are **strike-family pairs**. They may be defensible on no-arbitrage / Carr-Madan theory grounds, but they haven't been bootstrap-validated and one explicitly has `lo=0.0`. **Compute CIs and re-evaluate before any sizing decision.**

## 5 · BH-FDR resolution caveat

The stored `perm_p` distribution is heavily right-truncated: max p=0.067, 22 of 46 are exactly 0.0, granularity ~0.007-0.02 (≈50-150 perms). With p-values pinned this low, BH-FDR is **non-discriminating at q=0.05** — almost everything passes. The BH gate alone is therefore weak; bootstrap-CI is doing the load-bearing work.

**Action:** re-run permutation tests with ≥10 000 perms before the next wave-stress check. Until then, only trust the bootstrap-CI signal.

## 6 · Dead strategies (documented to prevent re-exploration)

Per `memory/project_microstructure_findings.md`:

- **S1 kline-OFI** — CKS 2014 lives at tick level (ms cadence, L1 depth deltas). 1-min aggregates wash out the signal. Re-test only with `PFM_CRYPTO_WS_ENABLED=1` capturing true tick OFI.
- **S5 1Hz OBI mean-reversion** — at 1-second polling, OBI is MOMENTUM not MR. Theory's absorption effect is at μs scale. Re-test only with <100ms matching-engine feed.
- **S6 single-factor VR** — single factor isn't mean-reverting on its own. MR lives in pairs — this is why the project uses cointegration spreads, not single factors.

## 7 · Operational follow-ups

1. Re-run permutations on the full alpha catalog with ≥10 000 perms to give BH-FDR real discriminating power.
2. Compute bootstrap CIs for the 45 strategies missing them, especially the 4 demoted A_STRUCTURALs.
3. Wave-6 4-quarter Sharpe stability check is **pending** (agent was rejected). Run before promoting any new A_STRUCTURAL to live trading.
4. VPIN gate: switch to rolling p90/p99 thresholds. Filed as B_VALIDATED follow-up.
5. Cap each intra-theme election-pair at 2% portfolio weight — five anti-correlated French/Brazilian pairs dominate the deployable set and cannot all be sized independently.

## 8 · How to reproduce

- α Hub audit: `/tmp/wave6_alpha_fdr.md` (agent report) + `web/data/alpha_strategies.json.bak-pre-wave6-20260519` (pre-patch backup).
- VPIN 4-window test: `arbstuff/research/s4_vpin_btc_test.py` + `/tmp/wave6_vpin_*.md`.
- Roll persistence test: `/tmp/wave6_roll_persistence.md` + `/tmp/roll_persistence_test.py`.
- OBI test (raw tape): `arbstuff/research/s5_obi_meanrev_test.py` + `s5_obi_results_20260519.json`.

— end of report —
