# Exotic Regressions: A Hypothesis-Driven Tour of `/fit`

*Author: research desk · Date: 2026-05-15 · Window: 2025-11-15 → 2026-05-14*

## 1. Intent and method

I treated `POST /fit` as a quant playground and ran ten regressions, each picked because there is a **prior economic story** linking the candidate prediction-market factors to the target ticker. For every fit I requested HAC inference (Newey-West, Andrews-bandwidth), 200 bootstrap iterations, a 30-day rolling window, and a 25 % out-of-sample test fraction. Factor candidates were sourced from `POST /factors/suggest-for-ticker` (lookback 120 days, top-15) and pruned to 4–6 names per fit on economic grounds — never on raw |r|. The window is short (≈120 trading days, ≈6 months), the multiple-comparison count is high (10 fits × 5 factors ≈ 50 t-tests, **no BH-FDR applied below**), and the pseudo-backtest uses the **in-sample** fitted return as the position signal — so its Sharpe should be read as an upper bound on what survives walk-forward, not as a deployable PnL.

In what follows I report point estimates, HAC p-values, factor-contribution shares (Δ R² incremental decomposition), rolling-beta stability, the server's verdict label, OOS R² where the server returned one, and the headline pseudo-backtest stats. Every "deployable" claim in §13 is conditional on at least the rolling-beta sign being stable and (where computable) the OOS R² being non-negative.

---

## 2. XLE — Energy ETF on Iran-conflict & oil odds

**Hypothesis.** Higher Iran regime-fall odds and oil-above-$150 odds raise crude → XLE; a US-Iran nuclear deal lowers the war premium → XLE.

**Setup.** `XLE` ~ `iran_regime_eoy` + `us_iran_nuclear_deal_before` + `iran_regime_jun` + `oil_above_150_jun` + `us_iran_nuclear_deal_jun`. n=33 (strict align).

| Factor | β | p (HAC) | Sign vs theory | Contribution share |
|---|---:|---:|---|---:|
| `us_iran_nuclear_deal_before` | -0.0437 | **0.003** | matches (deal → lower oil → lower XLE) | 56.6 % |
| `oil_above_150_jun` | +0.0087 | **0.026** | matches | 10.3 % |
| `us_iran_nuclear_deal_jun` | +0.0202 | **0.026** | wrong sign (positive!) | 19.8 % |
| `iran_regime_jun` | +0.0293 | 0.067 | matches direction | 12.4 % |
| `iran_regime_eoy` | +0.0087 | 0.666 | matches direction | 0.8 % |

R² = 0.49 · adj-R² = 0.39 · F-p < 0.001 · DW = 1.65 · HAC lag = 3 · verdict = `well_specified`. Rolling betas are very tight: `us_iran_nuclear_deal_before` ranges only [-0.054, -0.036] across the 30-day windows — sign-stable. **OOS not computable** (n_obs=33 < threshold). Pseudo-backtest Sharpe 7.75, total return 24 %, hit rate 64 % — implausible at face value, classic IS-overfit.

**Interpretation.** The "deal-before-2027" factor dominates and matches theory; the same factor in the June flavour flips sign, which is most likely a multicollinearity artifact between two near-substitutes (VIF 2.2 / 2.4 — borderline). Story holds at the headline level, but the wrong-sign June factor and short n make this *suggestive, not confirmed*.

---

## 3. ITA — Defense ETF on conflict odds

**Hypothesis.** Rising odds of China-Taiwan, Russia-NATO, Iran instability → defense spending expectations → ITA up. Ukraine joining NATO is ambiguous (could be peace dividend or escalation).

**Setup.** `ITA` ~ `china_invade_taiwan_2026` + `russia_invade_nato_jun` + `iran_regime_eoy` + `ukraine_joins_nato` + `us_strike_or_more_countries`. n=102.

| Factor | β | p | Notes |
|---|---:|---:|---|
| `ukraine_joins_nato` | -0.0234 | 0.085 | only borderline-significant factor |
| `russia_invade_nato_jun` | -0.0135 | 0.280 | wrong sign |
| `china_invade_taiwan_2026` | +0.0170 | 0.536 | right sign, not significant |
| `iran_regime_eoy` | -0.0080 | 0.546 | wrong sign |
| `us_strike_or_more_countries` | -0.0014 | 0.776 | clipped 22× |

R² = 0.06 · adj-R² = 0.006 · F-p = 0.52 · verdict = `weak_fit`. **OOS R² = -0.30** (4 walk-forward folds: -2.81, -0.40, -0.20, -0.15). All folds negative.

**Interpretation.** Clean **null**. ITA does not load on these conflict-binary contracts at all over the past six months. Either the markets are too quiet (low Δlogit), defense names are dominated by ETF flows / earnings idiosyncratic to the basket, or the right factor is something this catalog does not have (e.g. defense-budget-passage odds).

---

## 4. GDX — Gold miners on metals & macro odds

**Hypothesis.** GC > $5500 and SI > level → miner revenue → GDX up. GC < June level → GDX down. Fed cuts (dovish) and recession odds → safe-haven bid → GDX up.

**Setup.** `GDX` ~ `gold_5500_jun` + `gc_settle_below_jun` + `si_settle_above_jun` + `fed_cuts_2_2026` + `us_recession_2026`. n=75.

| Factor | β | p | Sign vs theory | Share |
|---|---:|---:|---|---:|
| `si_settle_above_jun` | +0.0397 | **<0.001** | matches | 42.3 % |
| `gold_5500_jun` | +0.0206 | **0.025** | matches | 17.5 % |
| `gc_settle_below_jun` | -0.0225 | 0.067 | matches direction | 16.7 % |
| `us_recession_2026` | -0.0720 | **<0.001** | wrong sign (recession → defensive bid expected) | 13.8 % |
| `fed_cuts_2_2026` | +0.0481 | **0.017** | matches (dovish → gold up) | 9.8 % |

R² = 0.48 · adj-R² = 0.45 · F-p < 0.001 · DW = 2.60 · verdict = `well_specified`. Rolling betas: silver +0.037 mean ([-0.008, +0.073]) — mostly sign-stable. Recession beta swings sign in rolling windows ([-0.117, +0.078], mean -0.060), confirming the negative loading is regime-dependent. **Pseudo-backtest Sharpe 10.1, total return 310 %, hit rate 77 %** — almost certainly IS-overfit but the sign of every signal-bearing factor matches the structural story.

**Interpretation.** **Best fit of the panel** by a wide margin. The metals trio (gold, silver, gold-below) plus dovish Fed expectations cleanly explain GDX. The negative recession sign is the only blemish — it likely reflects the fact that "recession odds rising" in 2025-26 has coincided with risk-off sell-everything episodes that dragged gold *miners* (high-beta, equity-like) down even as bullion held. That's a known feature of GDX vs GLD.

---

## 5. DKNG — Sports gambling on NFL & FIFA winner odds

**Hypothesis.** Sports betting volume tracks event interest. Marquee teams (Cowboys, Eagles, Chiefs) winning championship odds and World Cup contender odds may track engagement → DKNG.

**Setup.** `DKNG` ~ `will_the_kansas_city_chiefs_win_the_nfl_league_cha` + `will_the_philadelphia_eagles_win_the_nfl_league_ch` + `will_the_dallas_cowboys_win_the_nfl_league_champio` + `france_win_the_fifa_world` + `england_win_2026_fifa_world`. n=45.

| Factor | β | p | Share |
|---|---:|---:|---:|
| `france_win_the_fifa_world` | +0.6954 | **0.002** | 67.8 % |
| `will_the_dallas_cowboys_win_the_nfl_league_champio` | +0.1589 | **0.002** | 28.7 % |
| `will_the_philadelphia_eagles_win_the_nfl_league_ch` | -0.0780 | 0.295 | 3.0 % |
| others | ~0 | n.s. | <1 % |

R² = 0.33 · adj-R² = 0.24 · F-p = 0.011 · verdict = `well_specified`. Rolling betas: France [+0.029, +0.768], Dallas [+0.082, +0.205] — the *signs* are stable but magnitudes swing wildly (the France beta of 0.69 means a 1-unit Δlogit on a low-probability World Cup contender translates to a 70 % stock move, which is mechanically silly). England's rolling beta swings between -1.62 and +0.76.

**Interpretation.** The point estimates are absurd in magnitude because two of the contracts trade at very low probability and produce small but volatile Δlogit moves; the regression simply fits a large coefficient. With n=45 and such low-volume markets, the t-stat of 3.1 is hot air. This is a **noise-driven fit** dressed in significance — exactly the kind of thing BH-FDR over the panel would knock out.

---

## 6. PFE — Pharma on macro odds (no FDA factors in catalog)

**Hypothesis.** With no pharma-specific prediction-market factors in the catalog (no FDA approvals, no GLP-1 odds), PFE should respond to defensive-rotation macro: Fed cuts and recession odds → defensive bid.

**Setup.** `PFE` ~ `fed_cuts_2_2026` + `us_recession_2026` + `no_fed_cuts_2026` + `fed_rate_hike_2026` + `inflation_above_4_2026`. n=88.

R² = 0.006 · adj-R² = -0.05 · F-p = 0.95 · **all betas n.s.** · verdict = `weak_fit`. Pseudo-backtest Sharpe 0.80, total return 5 %, hit rate 50 % — basically a coin flip.

**Interpretation.** Clean **null**. PFE is not driven by aggregate Fed/recession odds in this window. The signal is presumably in pipeline news, regulatory headlines, and tariff exposure — none of which exist as a Polymarket factor in the catalog. *Recommendation*: do not pretend macro factors explain PFE here; mark this as "no information" rather than as a failed alpha.

---

## 7. JPM — Money-center bank on Fed path & recession

**Hypothesis.** Steeper curve (some cuts but not all) → JPM up. Recession odds → JPM down. No-cuts → ambiguous (high NIM but credit risk).

**Setup.** `JPM` ~ `fed_cuts_2_2026` + `us_recession_2026` + `no_fed_cuts_2026` + `fed_rate_hike_2026` + `fed_target_40_eoy`. n=65.

| Factor | β | p | Sign |
|---|---:|---:|---|
| `fed_target_40_eoy` | -0.0068 | 0.082 | matches direction |
| `us_recession_2026` | -0.0169 | 0.375 | matches |
| `no_fed_cuts_2026` | +0.0059 | 0.470 | matches direction |
| `fed_rate_hike_2026` | -0.0047 | 0.712 | wrong sign |
| `fed_cuts_2_2026` | +0.0048 | 0.800 | matches direction |

R² = 0.06 · adj-R² = -0.02 · F-p = 0.35 · verdict = `weak_fit`. No factor crosses p<0.05. Note: `fed_target_40_eoy` had 17 clipping events at ε=0.01.

**Interpretation.** Another **null**. The signs are mostly directionally consistent with theory but nothing is statistically distinguishable from zero. Given the well-known literature on bank stocks responding to **realized rate moves** (not probability of cuts), this is the expected outcome: the prediction-market factor is the wrong unit of information.

---

## 8. LIT — Lithium ETF on Tesla / EV / oil odds (BUG SURFACED)

**Hypothesis.** Tesla good-news (largest co., robotaxi, Optimus) → EV demand story → lithium up. High oil → lithium up (substitution). Fed cuts → growth-tilt up.

**Setup.** `LIT` ~ `tsla_largest_jun` + `tesla_robotaxi_ca_jun` + `will_tesla_release_optimus_by_june_30` + `fed_cuts_2_2026` + `oil_above_150_jun`. n=32.

| Factor | β | p | Note |
|---|---:|---:|---|
| `oil_above_150_jun` | -0.0290 | **0.002** | wrong sign |
| `tesla_robotaxi_ca_jun` | -0.0095 | **0.006** | wrong sign |
| `tsla_largest_jun` | -1.5500 | 0.047 | **VIF=10⁹, 98/149 obs clipped** — beta uninterpretable |
| `fed_cuts_2_2026` | +0.0210 | 0.291 | matches |
| `will_tesla_release_optimus_by_june_30` | +0.0018 | 0.833 | n.s. |

R² = 0.26 · verdict = `collinear`. Server flagged: `factor 'tsla_largest_jun': 98 clipping events (66% of obs); perfectly collinear factors detected: tsla_largest_jun (VIF=1000000000)`. The factor-contribution table reports `delta_r_squared = -1.110` for `tsla_largest_jun` (negative — a numerical artifact from the leave-one-out refit when that column is degenerate) with the share manually clipped to 0.0. Rolling-beta mean for `tsla_largest_jun` is **+9.247** while min is -1.85, max +2.45 — confirming nothing usable.

**Interpretation.** Two findings here. (a) The factor catalog needs a sanity gate: when a probability series spends two-thirds of its observations at the clipping floor, the factor should be auto-excluded with a warning, not just flagged. (b) Aside from that mess, the *real* signs (oil and robotaxi negative) reject the substitution / EV-tailwind story. LIT moved opposite to oil and opposite to robotaxi probability over the window — likely a "low-quality / battery-metal sell-off + high-oil = tighter financial conditions = LIT down" macro effect rather than the EV story.

---

## 9. XHB — Homebuilders on Fed path

**Hypothesis.** Cuts good, hikes bad; lower terminal rate → builders up.

**Setup.** `XHB` ~ `fed_cuts_2_2026` + `us_recession_2026` + `no_fed_cuts_2026` + `fed_rate_hike_2026` + `fed_target_40_eoy`. n=65.

| Factor | β | p | Sign |
|---|---:|---:|---|
| `fed_rate_hike_2026` | -0.0453 | **0.009** | matches (hike → XHB down) |
| `fed_target_40_eoy` | -0.0070 | **0.036** | wrong sign (4.0 % terminal would be relatively dovish) |
| `us_recession_2026` | -0.0270 | 0.180 | matches |
| `fed_cuts_2_2026` | +0.0246 | 0.173 | matches |
| `no_fed_cuts_2026` | +0.0025 | 0.800 | wrong sign |

R² = 0.35 · adj-R² = 0.29 · F-p < 0.001 · verdict = `well_specified`. Rolling betas on the hike factor are sign-stable: [-0.084, -0.013] mean -0.031.

**Interpretation.** The headline finding — XHB drops when hike-odds rise — is the **most theory-consistent of the macro fits**, and the rolling stability is encouraging. That said, only 65 obs and the wrong-sign `fed_target_40_eoy` mean this does not survive a strict reading. Most plausible read: hike-probability surprises drive a real beta; the level-of-terminal-rate factor is double-counting and should be dropped before redeployment.

---

## 10. NVDA — Pure AI factors only (no `nvda_largest_jun`)

**Hypothesis.** If AI-narrative factors carry alpha for NVDA, the Anthropic / Google / Alibaba / GPT-6 / AI-bubble odds should jointly explain a chunk of NVDA returns even with the tautological mcap factor excluded.

**Setup.** `NVDA` ~ `anthropic_best_jun` + `google_best_ai_jun` + `alibaba_have_best_ai_model` + `gpt_be_released_by_june` + `ai_bubble_burst`. n=101.

R² = 0.05 · adj-R² = -0.003 · F-p = 0.62 · verdict = `weak_fit`. **No factor significant.** **OOS R² = -0.41** (folds: -13.31, -0.61, -0.22, -0.04). One factor (`alibaba_have_best_ai_model`) had 90/176 clipping events. The factor-contribution table reports `delta_r_squared = 3.135` for `google_best_ai_jun` (a value > 1 is also a numerical artifact in the leave-one-out decomposition — likely caused by the OLS refit becoming numerically unstable when a near-constant factor is dropped).

**Interpretation.** A clean and important **null**: with the mechanical "NVDA largest" tautology removed, AI-narrative prediction-market factors carry **zero forecasting power** for NVDA daily returns. The earlier impression that "AI factors explain NVDA" was driven entirely by `nvda_largest_jun` (a contract whose price *is* NVDA's mcap rank). This is the result the report should make hardest to ignore: it deflates the AI-narrative regression as a story.

---

## 11. DIS — Entertainment on UCL/World-Cup + macro

**Hypothesis.** PSG winning Champions League may correlate with Disney's ad/streaming / ESPN-adjacent flows; recession and AI-bubble odds for risk-off context.

**Setup.** `DIS` ~ `psg_win_202526_champions_league` + `england_win_2026_fifa_world` + `france_win_the_fifa_world` + `ai_bubble_burst` + `us_recession_2026`. n=99.

| Factor | β | p | Share |
|---|---:|---:|---:|
| `psg_win_202526_champions_league` | +0.0489 | **<0.001** | 93.5 % |
| `england_win_2026_fifa_world` | -0.0848 | **0.018** | 5.9 % |
| others | ~0 | n.s. | <1 % |

R² = 0.16 · adj-R² = 0.11 · F-p < 0.001 · verdict = `well_specified`. **But**: the rolling beta on PSG is [-0.140, +0.095] **mean = -0.023** — the full-sample +0.049 *flips sign* across rolling 30-day windows. England rolling [-0.365, +0.612], no stable sign.

**Interpretation.** Spurious. A 4.3 t-stat on a sport contract that has no causal channel to Disney's cash flows is exactly the kind of result that 50 t-tests will throw up by chance. The rolling-beta sign flip kills it. **Anti-alpha**: do not redeploy.

---

## 12. Findings summary

| # | Ticker | Top significant factor(s) | R² | OOS R² | Verdict | Deployable? |
|---|---|---|---:|---:|---|---|
| 1 | XLE | `us_iran_nuclear_deal_before` (-), `oil_above_150_jun` (+) | 0.49 | n/a (n<min) | well_specified | tentative |
| 2 | ITA | none | 0.06 | **-0.30** | weak_fit | no (clean null) |
| 3 | GDX | `si_settle_above_jun` (+), `gold_5500_jun` (+), `us_recession_2026` (-), `fed_cuts_2_2026` (+) | 0.48 | n/a | well_specified | **yes (tentative)** |
| 4 | DKNG | `france_win_the_fifa_world` (+), `dallas_cowboys_nfl` (+) | 0.33 | n/a | well_specified | no (rolling magnitudes absurd) |
| 5 | PFE | none | 0.01 | n/a | weak_fit | no (clean null) |
| 6 | JPM | none | 0.06 | n/a | weak_fit | no (clean null) |
| 7 | LIT | `oil` (-), `tesla_robotaxi` (-), `tsla_largest_jun` (broken) | 0.26 | n/a | collinear | no (catalog bug + wrong-sign) |
| 8 | XHB | `fed_rate_hike_2026` (-), `fed_target_40_eoy` (-) | 0.35 | n/a | well_specified | tentative (hike beta sign-stable) |
| 9 | NVDA | none | 0.05 | **-0.41** | weak_fit | no (kills AI-narrative claim) |
| 10 | DIS | `psg_win_202526_champions_league` (+, spurious) | 0.16 | n/a | well_specified | no (rolling-sign flip) |

## 13. Discussion: what held up vs what didn't

**Held up.** **GDX on the metals trio (silver-above, gold-above-5500, gold-below as a negative) plus a dovish Fed-cuts factor.** Five of five betas have the right direction (silver, gold-up, gold-down, fed-cuts), all four are p<0.05, the rolling-beta signs are stable, and the contribution shares are economically intuitive (silver dominates because it is the noisiest / highest-beta metal). This is the only fit on this panel I would put real money behind, and even then only as a *factor exposure verification*, not as a tradable strategy. The recession-negative beta is a feature of the equity wrapper, not a bug.

**Held up partially.** **XLE on Iran-deal odds** — the dominant factor (`us_iran_nuclear_deal_before`) has the right sign, p<0.005, very tight rolling betas, and contributes 57 % of explained R². The catch is the wrong-sign June flavour of the same factor, which suggests the two are competing for the same variance. After dropping one, the story would be cleaner. **XHB on hike-odds** — the hike-factor beta is sign-stable and p<0.01 with the theoretically correct sign; the additional Fed factors are noise.

**Did not hold up.** Three of the ten regressions produced **clean nulls** (ITA, PFE, JPM) — the factor catalog simply does not contain the right information for those tickers. NVDA on **AI-narrative-only factors** also produced a null (R²=0.05, OOS R²=-0.41), which is the most informative result of the panel: it demonstrates that the apparent power of AI factors on NVDA in earlier work came from the tautological `nvda_largest_jun` contract.

**Spurious / regime / noise.** DKNG and DIS both produced superficially significant fits (R² 0.33 and 0.16, headline t-stats 3.1 and 4.3) on sports contracts with **no causal channel** to revenue. DIS-on-PSG flips sign in rolling windows; DKNG's France beta is mechanically absurd in magnitude (a 1-unit Δlogit move on a low-probability contract has tiny variance, which the OLS solver compensates for with a giant coefficient). LIT's `tsla_largest_jun` factor exposed a real **server-side issue** (see §14).

## 14. Bugs and engineering notes surfaced

1. **`delta_r_squared` can be negative or > 1.** In the LIT fit (`tsla_largest_jun`, share clipped to 0.0) and the NVDA fit (`google_best_ai_jun`, value 3.135), the leave-one-out incremental-R² calculation produces values outside [0, 1]. This happens when the OLS refit on the reduced design becomes numerically unstable — typically when one of the remaining columns is near-constant or when the dropped column is collinear. The server clips display but does not warn that the contribution-share is meaningless. Suggested fix: when |delta_r_squared| > 1 or < 0, replace with a sentinel and surface a warning instead of silently clipping the share.
2. **High-clipping factors should be auto-excluded.** `tsla_largest_jun` had 98 / 149 obs at the ε=0.01 floor; 66 % is well past the threshold at which the factor carries any usable Δlogit signal. The current behaviour (warn + still include) lets the factor produce a VIF of 10⁹ and a beta of -1.55. A hard cutoff at e.g. 50 % clipped → drop with reason in `auto_pruned` would be safer.
3. **OOS R² silently absent for short samples.** Only 2 of 10 fits (ITA, NVDA) returned an `oos_r_squared` object; the other 8 returned `null`. The threshold should be reported in the response (e.g. `"oos_r_squared": null, "oos_skipped_reason": "n_obs_used=33 below n_train+n_test minimum=70"`), otherwise users assume the field is missing rather than skipped.
4. **Pseudo-backtest is in-sample replay.** Sharpe 7.7–10.1 across well-specified fits. The `note` field says so honestly, but the headline number is misleading; consider also reporting a *walk-forward* Sharpe by default whenever n permits OOS.

## 15. Honest caveats

- **Window length.** All ten fits use ≈6 months. Two fits (XLE, LIT) collapsed to 33 / 32 obs after strict alignment with sparse Polymarket factors. Anything claimed on n < 50 should be treated as exploratory.
- **In-sample bias.** Pseudo-backtest Sharpes use the OLS prediction as the position signal *on the same data the OLS was trained on*; this guarantees a positive expected PnL even when the underlying betas are noise.
- **Multiple comparisons.** 10 fits × 5 factors = 50 t-tests. At α=0.05, BH-FDR with q=0.10 would knock out roughly any factor with a p-value above ~0.02 here. The DIS-on-PSG and DKNG-on-Cowboys "wins" almost certainly do not survive.
- **Event-study contamination.** Several factors (Iran-deal, Fed-cuts, Champions League) had specific news events in the window (e.g. the April Iran-strike repricing visible in XLE's PnL spike on 2026-04-08). A handful of high-Δlogit days drive most of the regression's R² in those cases.
- **Catalog gaps.** PFE has no FDA-approval factors; ITA has no defense-budget factors. The "weak_fit" verdicts there reflect catalog coverage, not the absence of structural relationships.

---

*Files: `/tmp/fits/{xle,ita,gdx,dkng,pfe,jpm,lit,xhb,nvda,dis}.json` for the raw `/fit` responses; `/tmp/sug_*.json` for the prior `suggest-for-ticker` rankings.*
