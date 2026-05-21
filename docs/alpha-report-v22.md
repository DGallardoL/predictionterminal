# Alpha Report v22 — 4Q Stability Reckoning · When the Statistical Gates Aren't Enough

**Date:** 2026-05-19
**Prior reports:** v21 (Wave-6 robustness + microstructure), v17 (Wave-5 honest reckoning)
**Purpose:** Re-test every Wave-6 A_STRUCTURAL promotion against the CLAUDE.md 4-quarter Sharpe-stability rule, and audit the four "Validated alphas" claimed in CLAUDE.md against the catalog we actually have. Be honest about what the data supports.

---

## 1 · Executive summary

> **0 of 5 Wave-6 A_STRUCTURAL promotions survive the strict 4-quarter Sharpe-stability gate.** Every newly-promoted pair has fewer than 6 months of joint Polymarket history — the underlying contracts were minted Q4-2025 / Q1-2026. Bootstrap-CI + BH-FDR + deflated Sharpe are necessary but not sufficient: a strategy can pass three sophisticated statistical gates and still have **zero out-of-regime evidence**. The lesson from Wave-5 ("default to B_VALIDATED until 4 quarters confirm") was the right one and was not respected in Wave-6. All five promotions should revert to B_VALIDATED; one (`renan_santos / us_aliens`) is the closest to passing on a lenient 3-of-3 reading and earns a B_VALIDATED++ marker. Separately, an honest re-test of CLAUDE.md's four "validated alphas" finds **only one (election-binary, 1 of 3 pairs) is loosely supported**; Sports mean-reversion fails with a clean sign-flip and moves to the anti-alpha list, Earnings-surprise has no underlying factors in the 1,260-factor catalog, and Fed-decision straddle works where data exists but has no pair with 4 quarters of joint history.

### Headline numbers

- Wave-6 A_STRUCTURAL pass rate against 4Q gate: **0 / 5**
- Max joint history of any Wave-6 promotion: **174 days** (`renan_santos / us_aliens`, 1.93 Q)
- CLAUDE.md "validated alphas" honest pass rate: **1 / 4** (Election-binary, partial — 1 of 3 pairs)
- New anti-alphas added: **1** (Sports mean-reversion in NBA-finals same-game contracts)
- Strategies moved to future-work / aspirational: **1** (Earnings-surprise odds vs IV — zero matching factors)

### Tier deltas (relative to v21)

| Tier | v21 | v22 (recommended) | Δ |
|---|---:|---:|---:|
| A_STRUCTURAL | 5 | **0** | -5 (all revert pending 4Q data) |
| B_VALIDATED | 22 | 27 | +5 |
| C_TENTATIVE | 13 | 13 | 0 |
| D_RAW | 29 | 29 | 0 |
| **Total** | **69** | **69** | |

The catalog totals stay flat; what moves is **tier confidence**. Net effect: zero A-tier deployables. That is the correct answer until the data exists.

---

## 2 · The 5 Wave-6 promotions vs the 4-quarter gate

Method: per CLAUDE.md Wave-5 rule — split the available joint history into four contiguous sub-windows, run `/strategies/pairs-backtest` (window=20, entry_z=2, exit_z=0.5, stop_z=4, ann=252) on each, and require **≥ 3 of 4 sub-quarters with Sharpe > 0.5 and no sign-flip**. Insufficient-data sub-quarters (rolling-z window=20 leaves too few bars) are counted as failures of the gate, not free passes.

| Pair | Joint days | Q1 | Q2 | Q3 | Q4 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `clmence_guett / tom_steyer` | 133 | 2.61 | 2.96 | INSUF | INSUF | **NO_DATA → revert** |
| `fed_target_45_eoy / no_fed_cuts_2026` | 126 | 0.00 (no signal) | INSUF | INSUF | INSUF | **NO_DATA → revert** |
| `asselineau / palomino` | 153 | 0.00 (no signal) | 3.24 | INSUF | INSUF | **FAIL → revert** (Q1 sh=0) |
| `renan_santos / us_aliens` | 174 | 3.42 | 3.14 | 6.26 | INSUF | **NO_DATA, but lenient 3/3 = B_VALIDATED++** |
| `grenell / us_iran_deal_jun` | 134 | 4.52 | 2.72 | INSUF | INSUF | **NO_DATA → revert** |

**Root cause:** the underlying Polymarket binaries (French 2027 presidential, Brazilian 2026 presidential, US alien-disclosure, Iran nuclear deal, dovish-Fed strike grid) were all minted Q4-2025 to Q1-2026. There is no way to construct 4 disjoint quarters from < 6 months of joint coverage. The earliest these can be re-tested honestly is **late August 2026** (≥ 180 trading days per leg), assuming markets stay liquid.

### The 4Q gate is doing real work

Three of the five pairs above produced Sharpes between 2.6 and 6.3 in the windows where data exists. Bootstrap-CI was strictly positive. BH-FDR cleared. Deflated Sharpe stayed high. Yet none of them have shown the regime variation needed to claim structural robustness. **This is exactly the failure mode Wave-5 warned about**: dazzling in-sample numbers from a single regime, no out-of-regime evidence, premature A-tier label.

---

## 3 · CLAUDE.md "Validated alphas" — independent honest check

Same `/strategies/pairs-backtest` method, but split by **calendar quarters** Q3-25 · Q4-25 · Q1-26 · Q2-26-partial, on the factor IDs each strategy actually maps to in the 1,260-factor catalog.

### 3.1 Election-binary momentum

| Pair | Q3-25 | Q4-25 | Q1-26 | Q2-26p | Verdict |
|---|---|---|---|---|---|
| `trump_out_2027 / xi_out_2027` | 502 a-leg | n=56, sh=0.0 | n=77, sh=2.21, hit=0.67 | INSUF | **FAIL** (Q4 sh<0.5) |
| `putin_out_2027 / xi_out_2027` | n=86, sh=2.94, hit=1.00 | n=92, sh=3.85, hit=0.75 | n=77, sh=2.16, hit=0.67 | INSUF | **PASS-3Q** (all same sign, all > 2) |

**Verdict:** 1 of 3 tested pairs (`putin / xi`) passes a lenient 3-of-3-valid-quarters reading. Capacity capped to one liquid cross-section. Keep the strategy on the validated list with the explicit caveat that it works on **long-dated `_out_2027` cross-sections only** and has not yet been tested across 4 full quarters (Q3-25 valid data emerges only for `xi_out_2027`-bearing pairs).

### 3.2 Fed-decision straddle proxy

| Pair | Q3-25 | Q4-25 | Q1-26 | Q2-26p | Verdict |
|---|---|---|---|---|---|
| `fed_target_45_eoy / no_fed_cuts_2026` | 502 | 502 | n=65, sh=4.10, hit=1.00 | n=30, sh=3.37, hit=1.00 | **NO_DATA** (only 2 valid Q) |
| `fed_no_change_jun / fed_cut_25_jun` | 502 | INSUF | n=81, sh=2.64, hit=1.00 | INSUF | **NO_DATA** (1 Q) |
| Kalshi `k_fed_jul_cut25 / k_fed_sep_cut25` | 502 | 502 | INSUF | n=35, sh=2.68, hit=1.00 | **NO_DATA** (1 Q) |

**Verdict:** the signal is real where it exists — Sharpe 2.6 to 4.1 with perfect hit rates — but **no pair has 4 quarters of joint history**. Polymarket FOMC strike contracts were minted ~Jan 2026 when the 2026 FOMC year began trading. CLAUDE.md's "VIX-overlay using Polymarket FOMC odds vs implied move" presumes a depth of history we do not have. Status downgraded from "Validated" to **`PENDING_4Q`** with a deployment-allowed annotation conditional on monthly re-test. Reassess October 2026.

### 3.3 Sports-event mean reversion

| Pair | Q3-25 | Q4-25 | Q1-26 | Q2-26p | Verdict |
|---|---|---|---|---|---|
| `cleveland_cavaliers_2026 / minnesota_timberwolves` | sh=1.00, hit=0.67 | sh=1.74, hit=0.80 | **sh=-2.24, hit=0.00** | INSUF | **FAIL — sign flip Q1-26** |
| `san_antonio_spurs / detroit_pistons` | sh=3.48, hit=1.00 | **sh=-1.94, hit=0.33** | **sh=-1.15, hit=0.50** | INSUF | **FAIL — sign flip Q4-25, persistent neg** |

**Verdict:** **Move to the anti-alpha list.** Both tested pairs flip sign within four quarters — this is a single-regime trade (works in summer / early-fall when contracts are far from resolution, fails as playoffs approach). CLAUDE.md's framing — "short overreactions in same-game contracts within the final hour" — was never instantiated on the NBA-finals binaries the project actually carries. The honest reading is: the alpha as described has no validated pair in the catalog, and the pairs that look like its closest cousins fail the gate cleanly.

### 3.4 Earnings-surprise odds vs IV

Searched the 1,260-factor catalog via `/factors?limit=2000` for slugs containing `earnings`, `beats_eps`, `eps_surprise`, `quarterly_earnings`. **Zero matches.** The CLAUDE.md "validated alpha" #4 has no underlying data source in our current catalog. It is aspirational, not validated. **Move to future-work** (revisit only if Polymarket lists liquid quarterly-EPS binaries on a meaningful number of names).

---

## 4 · Demotion recommendations (concrete)

### 4.1 `web/data/alpha_strategies.json` (handled by separate lane)

Revert the 5 Wave-6 promotions back to B_VALIDATED with `tier_change_reason="wave-7 (2026-05-19): fails 4Q stability gate per docs/alpha-reports/alpha-report-v22.md — insufficient joint Polymarket history for 4 disjoint quarters"`. Mark `renan_santos / us_aliens` with `notes="closest to passing — lenient 3/3 valid; re-test once Q4 data exists (target Aug 2026)"`. **A separate lane is handling this edit; v22 does not touch the JSON directly.**

### 4.2 `CLAUDE.md` edits (this report drives)

- **Validated alphas** section: remove Sports-event mean reversion, downgrade Earnings-surprise to future-work, tag Fed-decision straddle as `PENDING_4Q`, qualify Election-binary momentum to long-dated `_out_2027` cross-sections only.
- **Anti-alphas** section: add Sports mean-reversion in NBA-finals same-game contracts with citation to v22 §3.3.
- Add a one-line note acknowledging that the v21 Wave-6 A_STRUCTURAL promotions are revertible pending 4Q data.

### 4.3 `web/index.html` research-reports list

Promote v22 to `current: true`, demote v21 to a non-current marker. Update the fallback summary block.

---

## 5 · Operational follow-ups

1. **Re-test Wave-6 promotions monthly.** Set a calendar nudge for 2026-06-19, 2026-07-19, 2026-08-19. The first month where every Wave-6 pair has ≥ 4 disjoint 30-day sub-windows of joint coverage with rolling-z window=20 producing trades, re-run the gate. If any pair clears, promote it back to A_STRUCTURAL with a fresh report.
2. **Stop trusting "promotion-ready" labels that don't include a quarter count.** Every future A_STRUCTURAL ticket must include the joint-coverage day count and the number of disjoint quarters with valid trades. The Wave-6 audit shows the label can pass three statistical gates and still be premature.
3. **Patch the strict Wave-5 rule into the promotion script.** Whoever runs alpha-tier-regen next should hard-fail any pair with `joint_days < 360` from A-tier candidacy. This catches the structural issue at the data-availability layer, before bootstrap-CI and deflated Sharpe are even computed.
4. **Earnings-surprise: file in `docs/future-work.md`.** Note that the alpha requires Polymarket to list liquid quarterly-EPS binaries on ≥ 6 large-cap names. Track which tickers gain coverage. Until then, no point in re-running the test.
5. **Sports MR: keep the anti-alpha citation visible.** Future Claude must NOT re-pitch NBA-finals mean reversion based on a single summer's Sharpe. The sign-flip is the structural feature here.

---

## 6 · How to reproduce

- **4Q gate output** (5 Wave-6 pairs × 4 sub-windows + the 4 CLAUDE.md alphas × 4 calendar quarters): `/tmp/wave6_4q_alphas_final.md` (earlier agent today).
- **Raw API responses:** `/tmp/wave6_4q_raw.pkl`, `/tmp/wave6_validated_alphas.json`, `/tmp/wave6_validated.log`.
- **API:** `POST http://127.0.0.1:8000/strategies/pairs-backtest` with `{"a": "<slug_a>", "b": "<slug_b>", "window": 20, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "annualisation": 252, "start": "<YYYY-MM-DD>", "end": "<YYYY-MM-DD>"}`. 502 responses on the a-leg indicate the leg's market did not exist or was illiquid in that calendar quarter — a real data gap, not a network error.
- **Catalog check for earnings:** `curl -s 'http://127.0.0.1:8000/factors?limit=2000' | jq '.factors | map(select(.id | test("earnings|beats_eps|eps_surprise"; "i"))) | length'` returns 0.

---

## 7 · Closing note

This report is uncomfortable to write. v21 spent significant agent time promoting five pairs to A_STRUCTURAL on three sophisticated statistical gates, and v22 has to acknowledge that none of them clears the simplest gate of all: "did this work in more than one quarter?" The right response is not to lower the bar. It is to remember that **alpha is regime-survival, not in-sample sharpness**, and to wait until the data exists. We will be in the same place in three months — minus four false promises.

— end of report —
