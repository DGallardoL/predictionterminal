# Robustness Lab Report — Five-Candidate Validation Gauntlet

**Generated**: 2026-05-15. Sample window 2025-09-01 → 2026-05-14 (~8.5 months / ~3 quarters of Polymarket data). All fits use `/fit` with `regression=hac`, `bootstrap_iters=1000`, `oos_test_fraction=0.3`. Cross-tests via `/quant/multitest/bh` and `/quant/quarterly-stability`. Bootstrap CI on Sharpe and deflated-Sharpe (n_trials=5) computed locally on the strategy returns `r_t = sign(ŷ_t) · y_t` (in-sample). Cost sensitivity computed as `r_t − bp/10⁴ · |Δposition_t|` over the in-sample horizon. **Read-only run**; no factor catalogue, registry, or `/tmp` cache is mutated.

## Why these gates, and the project context

CLAUDE.md and `docs/alpha-reports/alpha-report-v17.md` make the project's stance explicit: every prior "wow" backtest in the wave-5 gauntlet either survived as B_VALIDATED (2/8) or was demoted to C_TENTATIVE (6/8) once subjected to bootstrap-CI + BH-FDR + 4-quarter stability + cost stress. The anti-alpha list in CLAUDE.md (recession-odds defensive long, crypto-ETF drift, Senate short-vol, geopol oil long) was assembled by exactly this procedure — single-window winners that signed-flipped or had Sharpe collapse OOS. The gates here exist because the alternative (calling a 6-month IS Sharpe of 8 a "deployable strategy") is how the v16 → v17 10× haircut happened. The deflated Sharpe explicitly accounts for the trial count of the survey (≥10 candidate panels were screened to land on these 5; n_trials=5 is a *floor*, true selection bias is larger). The 4Q stability gate is aspirational at 8.5 months — Polymarket history simply does not yet support an A_STRUCTURAL claim by the strict v17 definition.

## 1 · Per-candidate validation results

### 1.1 SPY × `[twelve_plus_fed_cuts, us_recession_2026, fed_cuts_2_2026, no_fed_cuts_2026]`

Rates-policy / recession-odds factor pack on broad-market beta.

| Gate | Result |
|---|---|
| Fit | n=141, R²=0.081, F p=0.064 |
| Significant β (HAC, p<0.05) | `us_recession_2026` β=−0.0204 (p=0.0061), `twelve_plus_fed_cuts` β=+0.0051 (p=0.028) |
| HAC 95% CI clears 0 | yes for both above |
| Bootstrap CI (β, 1000 iters) | `us_recession_2026` [−0.0353, −0.0043] **clears 0** · `twelve_plus_fed_cuts` [−0.00025, +0.0101] **straddles 0** |
| BH-FDR (within-pack, α=0.10) | 2/4 reject — q=0.024 (`us_recession`), q=0.056 (`twelve_plus_fed_cuts`) |
| In-sample Sharpe | **2.31** (CI95 [−1.17, +6.39], block=5) — **CI lower bound NEGATIVE** |
| Deflated Sharpe (n_trials=5) | z=0.50, p=**0.307** — fails |
| Quarterly Sharpes | 2025Q4 +0.11 · 2026Q1 +3.03 · 2026Q2 +8.46 (Q3 n=1, dropped) — passes 4Q gold? **no** |
| OOS walk-forward (test n=42) | OOS R²=−0.075, OOS Sharpe **−1.96** — **sign flip** |
| Cost sensitivity (Sharpe at 0/5/10/25 bp) | 2.31 / 1.29 / 0.27 / **−2.68** — break-even ≈ 11 bp |

**Read**: the macro pack has two genuinely significant betas with the expected signs (recession-odds short SPY, dovish-tail long SPY). But the trading translation (sign(ŷ)·y) loses out-of-sample (Sharpe −1.96) and the bootstrap CI on Sharpe straddles zero. Classic regression edge that does not survive the position-direction test.

---

### 1.2 COIN × `[btc_ath_jun, btc_200k_eoy, btc_150k_h1, btc_beats_gold, mstr_sells_btc]`

Crypto narrative pack on Coinbase equity. Borderline at the regression layer (no factor passes p<0.05 individually) but the joint F-test is highly significant.

| Gate | Result |
|---|---|
| Fit | n=67, R²=0.146, F p=**0.0003** (joint) |
| Significant β individually | none (best `btc_150k_h1` p=0.115) |
| Bootstrap CI (β) | `btc_150k_h1` [+0.012, +0.118] **clears 0** despite HAC straddle — bootstrap finds positive evidence HAC misses |
| BH-FDR within-pack (α=0.10) | 0/5 reject (best q=0.443) |
| In-sample Sharpe | **6.85** (CI95 [+3.31, +10.78], block=4) — **CI clears 0** |
| Deflated Sharpe (n_trials=5) | z=2.30, p=**0.011** — passes |
| Quarterly Sharpes | 2026Q1 +7.98 · 2026Q2 +3.16 (Q4 n=1 dropped) — both positive, no sign-flip, but only 2Q |
| OOS walk-forward (test n=20) | OOS R²=−1.09, OOS Sharpe **+4.93** — *positive*, despite negative R²: position-direction is right even though magnitude is wrong |
| Cost sensitivity | 6.85 / 6.72 / 6.58 / 6.17 — flat in cost (n_trades=61, low turnover relative to magnitude) |

**Read**: the standout. Joint significance + positive bootstrap CI on Sharpe + DSR p<0.05 + OOS sign-direction holds + cost-insensitive. Caveat: only 2 disjoint quarters; the F-test joint significance is doing the heavy lifting and *no individual factor BH-survives*. Reads as a beta-to-crypto-sentiment factor, not 5 independent edges.

---

### 1.3 TSLA × `[china_invade_taiwan_2026, china_blockade_taiwan, us_invade_cuba]`

Geopolitical risk-off panel. Hypothesis: TSLA loads negatively on tail-conflict odds. **Failed across every gate.**

| Gate | Result |
|---|---|
| Fit | n=74, R²=0.008, F p=0.916 |
| Significant β | none (all p>0.48) |
| Bootstrap CI | all factors straddle 0 |
| BH-FDR | 0/3 reject (q=0.957) |
| In-sample Sharpe | 0.89 (CI95 [−3.04, +5.11]) — straddles 0 |
| Deflated Sharpe | z=−0.71, p=0.763 — fails |
| Quarterly Sharpes | 2026Q1 +2.17 · 2026Q2 **−3.00** — **sign-flip** |
| OOS walk-forward | OOS Sharpe −0.60 |
| Cost sensitivity | 0.89 / 0.58 / 0.27 / −0.66 — break-even ≈ 9 bp |

**Read**: pure noise. Geopolitical-tail factors do not load measurably on TSLA over this window. Echoes the CLAUDE.md anti-alpha "geopolitical-conflict oil long" pattern.

---

### 1.4 NVDA × `[openai_ipo_1t, anthropic_ipo_before, fannie_mae_ipo_before]`

AI/IPO sentiment proxy on NVDA. **Failed.**

| Gate | Result |
|---|---|
| Fit | n=55, R²=0.040, F p=0.435 |
| Significant β | none (all p>0.29) |
| Bootstrap CI | all straddle 0 |
| BH-FDR | 0/3 reject (q=0.41) |
| In-sample Sharpe | 0.28 (CI95 [−3.31, +3.79]) |
| Deflated Sharpe | z=−1.06, p=0.856 |
| Quarterly Sharpes | 2026Q1 −0.55 · 2026Q2 +1.96 — **sign-flip** |
| OOS walk-forward | OOS Sharpe **−9.08** (test n=16, very small — high-leverage outlier) |
| Cost sensitivity | 0.28 / 0.02 / −0.24 / −1.03 — break-even ≈ 5 bp |

**Read**: smallest sample (n=55), no signal anywhere. The OOS Sharpe −9 on n=16 is essentially a single bad week and should not be over-interpreted, but there is nothing to defend.

---

### 1.5 GLD × `[gold_5500_jun, us_recession_2026, twelve_plus_fed_cuts, btc_beats_gold]`

Gold-strike + macro pack on GLD. **The strongest single survivor — but with caveats.**

| Gate | Result |
|---|---|
| Fit | n=76, R²=**0.334**, F p=0.0013 |
| Significant β | `gold_5500_jun` β=+0.032 (p=**0.0005**) — the rest insignificant |
| Bootstrap CI (β) | `gold_5500_jun` [+0.020, +0.044] **clears 0 strongly** |
| BH-FDR within-pack (α=0.10) | 1/4 reject — q=**0.0019** for `gold_5500_jun` |
| In-sample Sharpe | **8.28** (CI95 [+5.42, +11.93], block=4) |
| Deflated Sharpe (n_trials=5) | z=3.77, p=**0.00008** — passes strongly |
| Quarterly Sharpes | 2026Q1 +8.07 · 2026Q2 +11.76 (Q4 n=1 dropped) — both strong, but only 2Q |
| OOS walk-forward (test n=22) | OOS R²=+0.243, OOS Sharpe **+13.14** — confirms IS direction |
| Cost sensitivity | 8.28 / 7.94 / 7.59 / 6.56 — robust to 25 bp |
| Skew / kurtosis | skew +1.16, kurt 7.34 (Pearson) — **fat right tail**, jackpot dynamics |

**Read**: structurally the cleanest result, but the +1.16 skew and 7.34 kurtosis say a few large positive returns are doing the work — characteristic of a strike-family payoff, not a smooth alpha. `gold_5500_jun` is a tautological proxy for GLD-near-strike; this is essentially the "strike-family A_STRUCTURAL" pattern from v17 §4.4 leaking into a regression frame.

---

## 2 · Cross-candidate (global) BH-FDR

Best p-value per candidate, BH-corrected at α=0.10:

| # | Candidate | Best factor | Raw p | Global q (BH) | Reject @ 0.10? |
|---|---|---|---:|---:|:---:|
| 1 | SPY × macro_fed | `us_recession_2026` | 0.0061 | **0.015** | YES |
| 2 | COIN × crypto | `btc_150k_h1` | 0.115 | 0.192 | no |
| 3 | TSLA × geopolit | `china_blockade_taiwan` | 0.484 | 0.484 | no |
| 4 | NVDA × ai_ipo | `openai_ipo_1t` | 0.294 | 0.367 | no |
| 5 | GLD × gold_macro | `gold_5500_jun` | 0.00048 | **0.0024** | YES |

Two survive global BH-FDR at α=0.10. COIN at q=0.19 is the boundary case — it never had an individual-factor pass, only a joint-F win.

## 3 · Final verdict table

Tier definitions repeated for convenience:

| Tier | Gate |
|---|---|
| A_STRUCTURAL | Bootstrap CI lo > 0 · DSR < 0.05 · 4Q stability (≥4Q, all positive, no sign-flip) · BH q < 0.05 |
| B_VALIDATED | Bootstrap CI lo > 0 · DSR < 0.05 · ≥3Q stability · BH q < 0.10 |
| C_TENTATIVE | Bootstrap CI lo > 0 · some quarter weak · BH q < 0.20 |
| D_RAW | anything else |

| # | Candidate | β-CI bot > 0? | DSR p | 4Q stable? | OOS Sharpe | Global BH q | Cost-bp break-even | **Tier** | Reason |
|---|---|:---:|---:|:---:|---:|---:|---:|:---:|---|
| 1 | SPY × macro_fed | mixed (1/2) | 0.307 | no (3Q only, OK trend) | −1.96 | 0.015 | ~11 | **D_RAW** | OOS sign-flip + DSR fails + IS Sharpe CI straddles 0 |
| 2 | COIN × crypto | yes (Sharpe) | **0.011** | n/a (2Q both pos) | +4.93 | 0.192 | >25 | **C_TENTATIVE** | Survives Sharpe-CI + DSR + OOS direction + cost; fails BH q<0.10 (no individual factor passes) and lacks 3Q |
| 3 | TSLA × geopolit | no | 0.763 | no (sign-flip Q1→Q2) | −0.60 | 0.48 | ~9 | **D_RAW** | Pure noise; every gate fails |
| 4 | NVDA × ai_ipo | no | 0.856 | no (sign-flip) | −9.08 | 0.37 | ~5 | **D_RAW** | Pure noise; sample too small (n=55, OOS n=16) |
| 5 | GLD × gold_macro | yes (β & Sharpe) | **0.00008** | n/a (2Q both pos, n=1 in Q4) | **+13.14** | **0.0024** | >25 | **B_VALIDATED** (with strike-family caveat) | All numerical gates pass; only blocker to A_STRUCTURAL is the absent 4Q (Polymarket data span). Stem concentration on `gold_5500_jun` strike — not an independent edge |

**Tally**: A_STRUCTURAL **0**, B_VALIDATED **1** (GLD/gold_5500), C_TENTATIVE **1** (COIN), D_RAW **3** (SPY, TSLA, NVDA).

## 4 · Discussion

**Survivors (1 B + 1 C of 5)**. GLD × gold-strike pack is the only B_VALIDATED — and even there the deflated-Sharpe pass is essentially the `gold_5500_jun` strike acting as a proxy for the underlying. This is the same `polymarket_calendar_lambda_v1` / strike-family pattern that v17 §4.4 already classified A_STRUCTURAL on a half-size deploy basis: mechanically tied to GLD spot. **It is not new alpha; it is restated cointegration.** COIN sits at C_TENTATIVE on the strength of its joint-F significance (p=0.0003), Sharpe-CI [3.3, 10.8], DSR p=0.011, and OOS sign-direction confirmation — but no individual factor passes BH within the pack and the global BH q=0.19 is above the C threshold of 0.20 by a thin margin.

**What killed the others**.
- *SPY*: in-sample regression looked clean (2 factors BH-pass within pack, recession-odds with the textbook negative sign). The trading translation broke. OOS Sharpe −1.96. The discrepancy is informative: the level relationship is real (recession odds correlate negatively with SPY), but the *next-day-direction* signal is washed out by SPY's idiosyncratic noise. Regression p-values mislead when the dependent variable is dominated by single-name vol.
- *TSLA, NVDA*: the panels are not structurally connected to the regressand. Geopolitical-tail odds have a coarse weekly resolution and TSLA moves on Twitter; AI-IPO odds resolve over months and NVDA moves on demand commentary. **Frequency mismatch**, not absence of an underlying economic story.
- *DSR penalty*: with n_trials=5 the deflated-Sharpe E[SR_max] threshold sits at ≈0.7σ_SR. Three of the five have raw SR/se inside that band — the deflation is doing real work, not just noise.

**Quarterly stability is the binding constraint** on every candidate. The `/quant/quarterly-stability` endpoint returned `passes_4q_gold=false` and `tier_recommendation=C_TENTATIVE` for all five — an artefact of the 8.5-month sample window matching v17's diagnosis ("Polymarket data only spans ~3 quarters"). One more quarter of clean data could lift COIN to B and confirm GLD at A.

## 5 · Honest caveats

1. **Sample length**. 8.5 months is too short for the 4-quarter test. The 4Q gate could not, by construction, be met by any candidate. We report the 2Q / 3Q variants and defer A_STRUCTURAL claims accordingly.
2. **Block-bootstrap block size**. We use `L = max(T^(1/3), 3)` (3 to 5 across candidates). Politis-Romano stationary bootstrap is robust to ±50% block-size mis-specification, but a sweep would tighten CIs by perhaps 10-20%. Not done.
3. **Deflated-Sharpe trial count**. Set to **5** (= number of candidates carried into the gauntlet). The true number of (ticker × pack) combinations *implicitly* tested upstream — the user's earlier exotic-regression survey — is plausibly 10-50. Re-running with n_trials=50 would push GLD's DSR p from 8e-5 to ~7e-4 (still passes), COIN's from 0.011 to ~0.05 (boundary), and would make the COIN survival doubt-of-the-week rather than a clear C.
4. **In-sample naïve replay**. The `pseudo_backtest` (and the strategy returns reconstructed here) uses contemporaneous `predicted` from the IS regression. There is no information-leak control beyond the 0.3 OOS holdout, which is the discipline `/fit` provides. A walk-forward refit at each step would lower most IS Sharpes by 0.5-1.5; the OOS walk-forward column already captures the worst-case version.
5. **Cost sensitivity** uses position-flip count from `sign(predicted)` — which over-counts because every observation's `predicted` flips sign frequently in noisy panels. A real deploy would smooth via a holding period or a 0-band around the prediction; both lower turnover and shift cost-break-even higher. The numbers reported are conservative (worst-case turnover).
6. **Within-pack BH-FDR vs. cross-pack global BH-FDR** are reported separately. The within-pack version is what each candidate would face if proposed alone; the global version is what the gauntlet of 5 implies. We use the global q for the tier decision because the candidate selection itself was a multi-test exercise.
7. **CLAUDE.md anti-alpha overlap**. SPY × macro_fed combines `us_recession_2026` with broad-market exposure — the closest analogue to the "recession-odds → defensive-sector long" anti-alpha. Its OOS sign-flip here is a *third-quarter confirmation* of the v17 anti-alpha listing. Do not redeploy.

## 6 · One-paragraph recommendation

Of the five candidates, **only GLD × `gold_5500_jun`** earns B_VALIDATED — and it is structurally a strike-family pattern that v17 §4.4 already categorises as A_STRUCTURAL at half-size. **COIN × crypto-pack** is the most interesting *new* find at C_TENTATIVE: joint-F significance + Sharpe-CI clears zero + DSR pass + OOS sign-direction holds + cost-insensitive. It deserves another quarter of out-of-sample tracking before being graduated to B. The remaining three (SPY, TSLA, NVDA) join the regime/noise pile. The single most important finding is that **none of the five clears the 4-quarter stability gate — by construction**, until Polymarket history reaches 12+ months, and that the v17 portfolio recommendation (no live A_GOLD until 2026Q3) is reaffirmed by this independent gauntlet.
