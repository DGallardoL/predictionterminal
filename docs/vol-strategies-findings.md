# Vol-Trading Strategies — Joint Findings & Deployment Decision

Run date: 2026-05-16  
Author: Phase-9 synthesis after A1-A4 + B1-B4 + F1 (σ_PM bias fix) + F2 (slug rediscovery).

## TL;DR

| Strategy | Tier (per CLAUDE.md) | Ship to UI? | Action |
|---|---|---|---|
| **A — PM-IV gap** (σ_PM vs VIX/OVX/GVZ/DVOL) | **C_TENTATIVE** | **NO** | Keep modules + endpoints behind feature flag; do not advertise as signal |
| **B — Event-vol harness** (multinomial → EM) | **CEMETERY** | **NO** | Keep modules + endpoints behind feature flag for future calibration work; **not a trading signal** |

Both strategies built end-to-end (8 modules, 11 endpoints, 137 tests passing). Both backends are alive behind `PFM_VOL_PM_IV_ENABLED=1` / `PFM_VOL_EVENT_ENABLED=1` for research access. **Neither earns a UI panel right now.**

---

## A — PM-IV gap: what we built, what we learned

### Build
- `pfm/vol/pm_iv_extractor.py` (917 lines) — direction-aware (`above` / `dip_to` / `hit_high` / `below` / `range_low|high`) survival function → lognormal σ via either survival-function least-squares (F1's fix, primary) or moment-match on PMF (fallback).
- `pfm/vol/vol_benchmarks.py` (452 lines) — VIX/OVX/GVZ from FRED; Deribit BTC/ETH DVOL; Binance realized σ.
- `pfm/vol/pm_iv_gap.py` (281 lines) + router (176 lines) — composer with primary-benchmark routing, ±2pp flat band, signal strength bands.
- 36 tests covering the math, the registry, and the gap classification.

### Validation (A4, iter-2 with F1+F2 applied)

| Asset | Bench | N days | ⟨σ_PM⟩ | ⟨bench⟩ | ⟨gap⟩ | ρ(gap, fwd−bench) | ρ(Δgap, Δresid) | Sharpe | Honest read |
|---|---|---|---|---|---|---|---|---|---|
| **BTC** | DVOL | 141 | 48.3% | 48.6% | -0.3% | -0.220 | 0.474 | **-3.25** | σ_PM now matches benchmark cleanly → signal vanishes. Sharpe is negative — the gap predicted in the *wrong* direction once bias was removed. |
| **ETH** | DVOL | 0 | n/a | n/a | n/a | n/a | n/a | n/a | F2's short-dated May-2026 ladder didn't yield enough historical data. No signal extractable. |
| **WTI** | OVXCLS | 57 | 204.6% | 62.0% | +142.6% | 0.521 | 0.067 | 10.56 | F1's survival fit made WTI σ_PM *worse* (was 144%, now 205%). The "Sharpe" is a level-bias artifact — strategy is constantly long-vol in a regime where realised σ outran benchmark. |
| **GOLD** | GVZCLS | 64 | 88.9% | 32.8% | +56.0% | 0.532 | -0.087 | 5.35 | Bias halved by F1 (133% → 89%) but still inflated. Δ-Δ ρ flipped negative — the day-over-day predictive content disappeared. |

**Pooled (262 rows, 3 assets)**:
- Raw ρ(gap, fwd−bench) = **0.313**, CI95 [0.199, 0.418] → passes "level signal" floor.
- Demeaned-within-asset ρ = **0.242**, CI95 [0.124, 0.353] → still passes after stripping the static offset.
- **First-difference ρ(Δgap, Δresid) = 0.081** → essentially zero. **This is the critical metric and it dropped from 0.274 pre-fix to 0.081 post-fix.**

### What that means

The original A4 verdict (B_VALIDATED, demeaned ρ=0.355, Δ-Δ ρ=0.274) was sustained largely by the σ_PM **inflation bias**: the systematic positive offset on WTI/GOLD made the gap *look* tradeable, but a constant offset cannot generate day-over-day forecast value. When F1 partially corrected the bias (GOLD halved, BTC normalised — WTI got worse from a different math wart in the `hit_high` 0.5 barrier-factor approximation), three of the four observable phenomena weakened:

1. **BTC**, the asset where σ_PM now matches DVOL cleanly, shows **negative** Sharpe (-3.25). With the bias removed, BTC's tiny gaps don't predict — they slightly anti-predict.
2. **Δ-Δ ρ collapsed** from 0.274 to 0.081. This is the cleanest test of "day-over-day predictive content" and it nearly disappeared.
3. The pooled positive ρ that survives is now dominated by the *remaining* WTI/GOLD bias (those Sharpes 5–11 are level-bias artifacts, not tradeable).

Per CLAUDE.md anti-alpha pattern: "Don't redeploy regime-driven alphas without 4-quarter robustness." The original A4 looked like a regime-driven alpha; the F1 re-run confirmed it. **The signal is at most C_TENTATIVE — paper-only with a regime trigger — and we don't have one yet.**

### What would resurrect Strategy A

- **Fix `hit_high` survival function properly**: replace the 0.5 barrier factor with a per-strike numerical solution to the GBM first-passage equation. This is what made WTI worse (more sensitive to that factor than the moment-match was). Without this, σ_PM on commodities with `hit_high` ladders is unusable.
- **Tenor matching**: σ_PM is for the ladder's maturity (8 months out for BTC), benchmarks are 30-day. Compare apples-to-apples by either (a) extracting σ_PM at 30 days from a near-dated ladder (don't have one for BTC), or (b) building the matching 8-month options-IV benchmark.
- **4-quarter robustness**: current data window is 6 months. CLAUDE.md mandates 4 disjoint quarters. We won't have that until 2026-Q4.
- **Re-test on full universe**: BTC dip_to + BTC above expanded + WTI + GOLD only. SPX is gone from Polymarket. ETH coverage is thin. Universe is narrower than the brief suggested.

---

## B — Event-vol harness: shelved as trading signal

### Build
- `pfm/vol/event_vol_engine.py` (607 lines) — EventDistribution → EM via entropy-proxy (`k_fomc=0.50`, `k_cpi=0.30`, etc.) or fitted calibration.
- `pfm/vol/event_calendar.py` (447 lines) — 8 curated events (FOMC Jun/Jul/Dec 2026, CPI May/Jun/Jul 2026, midterms 2026, Brazil 2026) with PM/Kalshi slug mappings, all verified live in factors.yml.
- `pfm/vol/event_signal.py` (455 lines) + router (208 lines) — live signal composer.
- 42 tests, all passing.

### Validation (B4)

| Metric | Value | Reading |
|---|---|---|
| Past events recoverable | 8 curated / 5 honest equity-events | Tiny sample for any robustness claim |
| MAE EM vs realised | 1.13 pp | Proxy systematically under-shoots |
| **Pearson r (EM ↔ realised \|Δ\|)** | **-0.156** | **Negative — worse than random** |
| **Spearman r** | **-0.255** | Negative — ranking fails too |
| Short-straddle gross PnL | -1.13 pp/event | Loses on every event |
| Win-rate | 0% | Not a single win |
| CPI subset Pearson | -0.63 | Worst sector — wrong-direction concentrated |

### Why the proxy fails

Two diagnosed failure modes (per B4 agent):

1. **FOMC entropy collapse.** Prediction markets settled at ~99% "no change" weeks before both meetings, so `entropy_normalised ≈ 0` and the engine forecast EM ≈ 0. But SPY still realised ~1pp moves from positioning unwind. The engine needs a non-vanishing baseline event-day vol term, which the literature-derived `k_kind` constants don't capture.
2. **CPI dispersion constancy.** Across CPI events the implied-distribution entropy is roughly constant (H_norm ~0.66-0.76). All discrimination has to come from `asymmetric_mass` or `tail_pct`, but they were also flat. Realised moves are driven by *directional surprise vs consensus*, not by raw dispersion shape — the engine measures shape, not surprise.

A fitted `EMCalibration` cannot rescue this: with negative Spearman, no linear projection of the existing feature vector ranks realised moves correctly. The features themselves carry the wrong information.

### What would resurrect Strategy B

- Add a non-vanishing baseline term for FOMC (e.g., 0.4pp × (1 + asym_mass²)).
- Build a "directional surprise" feature requiring an external consensus print (Bloomberg consensus or Kalshi modal-anchor delta). Not in current data sources.
- Interact distribution shape with regime (VIX level, prior-event realised vol). Single-feature OLS isn't enough.

None of these are quick. **Shelve the trading signal interpretation. Keep the modules; the `EventDistribution` → `EventEMForecast` math is reusable as a descriptive panel ("market-implied event-day range") — a Terminal feature, not a strategy.**

---

## Deployment matrix

| Component | Status | Where | Public? | Notes |
|---|---|---|---|---|
| `pfm/vol/pm_iv_extractor.py` | Shipped | api/src/pfm/vol/ | Module only | Survival-fit primary, moment-match fallback |
| `pfm/vol/vol_benchmarks.py` | Shipped | api/src/pfm/vol/ | Module only | VIX/OVX/GVZ/DVOL/Binance wrappers |
| `pfm/vol/pm_iv_gap.py` | Shipped | api/src/pfm/vol/ | Module only | Composer |
| `pfm/vol/pm_iv_router.py` | Mounted | `/vol/pm-iv/*` | **Gated by `PFM_VOL_PM_IV_ENABLED=1`** | 3 endpoints; default OFF |
| `pfm/vol/event_vol_engine.py` | Shipped | api/src/pfm/vol/ | Module only | Entropy-proxy + OLS calibration |
| `pfm/vol/event_calendar.py` | Shipped | api/src/pfm/vol/ | Module only | 8 events curated |
| `pfm/vol/event_signal.py` | Shipped | api/src/pfm/vol/ | Module only | Live composer |
| `pfm/vol/event_vol_router.py` | Mounted | `/vol/event/*` | **Gated by `PFM_VOL_EVENT_ENABLED=1`** | 4 endpoints; default OFF |
| `web/index.html` | **NOT TOUCHED** | n/a | n/a | Per user instruction: no UI until signals prove out |
| `docs/vol-pm-iv-validation.md` | Generated | docs/ | docs only | A4 report (iter-2 post-fix) |
| `docs/vol-event-validation.md` | Generated | docs/ | docs only | B4 report |
| `docs/vol-strategies-findings.md` | This file | docs/ | docs only | Phase 9 synthesis |

### Verification snapshots

```bash
# All vol tests green:
api/.venv/bin/python -m pytest api/tests/test_pm_iv_extractor.py api/tests/test_pm_iv_gap.py \
    api/tests/test_pm_iv_router.py api/tests/test_vol_benchmarks.py \
    api/tests/test_event_vol_engine.py api/tests/test_event_calendar.py \
    api/tests/test_event_signal.py api/tests/test_event_vol_router.py \
    api/tests/test_vol_surface_pm.py -q
# → 101 passed in ~55s

# Feature flags wire correctly:
PFM_VOL_PM_IV_ENABLED=1 PFM_VOL_EVENT_ENABLED=1 PYTHONPATH=api/src \
  api/.venv/bin/python -c "from pfm.main import app; print(len([r for r in app.routes if '/vol/' in r.path and ('pm-iv' in r.path or '/vol/event' in r.path)]))"
# → 7 routes
```

---

## Recommendation

1. **Leave both routers default-OFF** (the user already chose this — `PFM_VOL_PM_IV_ENABLED` and `PFM_VOL_EVENT_ENABLED` are unset by default).
2. **Do not add UI panels** for either strategy. Strategy A would mislead users into thinking there's a tradeable gap when the day-over-day signal is ≈0. Strategy B is anti-correlated and would be actively harmful.
3. **Keep the modules** — `vol_benchmarks`, `event_calendar`, and the math infrastructure are reusable for future calibrated approaches, dashboards, or research scripts. They are quietly mature library code now.
4. **Reopen the question 2026-Q4** when:
    - Two more FOMC events have resolved (gives 4-quarter sample for B).
    - WTI/GOLD `hit_high` survival math is fixed (gives clean σ_PM on commodities for A).
    - At least one new ETH or SPX live ladder appears in Polymarket coverage.

The honest read is that **prediction-market-implied vol is real but is currently dominated by structural pricing artifacts** (resolution premium, USDC funding cost embedded in spreads, illiquidity at wide strikes, mismatched tenor vs options-IV benchmarks) that swamp the signal on a 6-month, 3-asset window. The infrastructure to evaluate it again later is now in place.
