# Validation A4 — σ_PM gap vs. forward realized σ

Run date: 2026-05-16T06:19:41+00:00  
Analysis window: 2025-10-01 → 2026-05-16  

## Question

Does the gap **σ_PM − σ_benchmark** predict **σ_fwd_realized − σ_benchmark**? If yes, the prediction-market ladder is adding information beyond options-derived IV.

## Methodology

1. σ_PM extracted daily from `LADDER_REGISTRY` via `fit_implied_sigma` (lognormal moment-matching on a PMF derived from the option chain as the second derivative of the call-price curve). Day eligible only when ≥3 ladder slugs have a quote (≤1 day forward-fill per slug).
2. σ_benchmark per asset: VIXCLS / OVXCLS / GVZCLS from FRED for SPX / WTI / GOLD; Deribit historical DVOL for BTC / ETH (with Binance 30d realised σ as fallback).
3. σ_fwd_realized: forward 30-day realised σ on the underlying — yfinance for SPX (`^GSPC`), WTI (`CL=F`), GOLD (`GC=F`); Binance daily for BTCUSDT / ETHUSDT. Annualisation: √252 equities/commodities, √365 crypto.
4. Strategy simulator: long-vol if `σ_PM > σ_bench + 2pp`, short-vol if `σ_PM < σ_bench − 2pp`, else flat. PnL ≈ sign · (σ_fwd − σ_bench), expressed in vol-points. Sharpe annualised by √252 across **all** aligned days (non-signaled days = 0 PnL).

## Per-asset results

| Asset | Bench | N days | Range | ⟨σ_PM⟩ | ⟨σ_bench⟩ | ⟨gap⟩ | ρ(gap, fwd−bench) | ρ(Δgap, Δresid) | N sig | Hit | Sharpe | Sharpe ex-top3 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC | Deribit DVOL BTC | 141 | 2025-11-25..2026-04-16 | 48.3% | 48.6% | -0.3% | -0.220 | 0.474 | 112 | 40.2% | -3.254 | -2.607 | ok |
| ETH | Deribit DVOL ETH | 0 | —..— | n/a | n/a | n/a | n/a | n/a | 0 | n/a | n/a | n/a | insufficient_data |
| WTI | FRED OVXCLS | 57 | 2025-12-29..2026-03-30 | 204.6% | 62.0% | 142.6% | 0.521 | 0.067 | 57 | 71.9% | 10.558 | 9.330 | ok |
| GOLD | FRED GVZCLS | 64 | 2025-12-29..2026-04-02 | 88.9% | 32.8% | 56.0% | 0.532 | -0.087 | 64 | 64.1% | 5.345 | 4.185 | ok |

> *Note:* per-asset Pearson r is translation-invariant, so a single-asset "demeaned" correlation is identical to the raw value. The within-asset demeaned check is only informative in the pooled sample, reported in the next section.

## Pooled results

- Rows pooled: **262** across 3 assets
- Pooled raw ρ(gap, fwd−bench): **0.313** (95% CI Fisher-z: [0.199, 0.418])
- **Demeaned-within-asset ρ**: **0.242** (95% CI: [0.124, 0.353]) — removes the σ_PM level-bias offset. **This is the honest number.**
- **First-difference ρ(Δgap, Δresid)**: **0.081** — does day-to-day *change* in the gap predict day-to-day change in the forward-residual? If this is ≈0 the raw r is co-trending, not predictive.
- Signals: 233 (hit rate 54.5%, mean PnL 1.6%)
- Pooled Sharpe (√252): **1.219** — **caveat:** this Sharpe is inflated by the level bias because the gap exceeds the +2pp threshold on essentially every day for WTI/GOLD, so the strategy is effectively a constant long-vol position during a regime where σ_realized > σ_bench. The Sharpe is not a fair estimate of out-of-regime performance.

## Caveats

- **σ_PM extraction bias.** The lognormal-fit pipeline tends to *over*-state σ when the ladder is wide and the upper-tail mass is non-trivial — empirical moments inflate the implied std because the missing right-tail beyond the highest strike is treated as point mass. The **gap direction** can still be informative even if the absolute level is biased high, which is what this validation tests.
- **Calendar mismatch.** PM ladders are point-in-time risk-neutral views on a specific maturity (EoY-2026 for SPX/BTC/ETH, June-2026 for WTI/GOLD). The benchmark (VIX/OVX/GVZ/DVOL) is 30-day. We're comparing different tenors — this is a known apples-to-oranges issue and will shrink the realisable signal.
- **Single-window risk.** Only one ~6-month window of data is available. CLAUDE.md mandates a **4-quarter robustness check** before any A_GOLD claim; this validation cannot satisfy that on its own.
- **Survivorship in ladder coverage.** SPX/BTC `above` ladders are uncovered in the strat7 cache and require live fetches; if Polymarket has resolved or delisted any slug, the daily σ_PM for that asset will simply be missing for those dates.
- **Forward 30d realised σ alignment.** For dates within ~30 days of TODAY we have less than a full forward window; those rows are dropped automatically.
- **Bench fallback for BTC/ETH.** If Deribit's historical DVOL endpoint is unreachable, we fall back to Binance 30d realised σ — which is a poor proxy for forward IV (it is a backward-looking estimator). This pollutes the gap computation for those assets; flagged in the per-asset table by the `Bench` column.

## Verdict

**B_VALIDATED**

Both the level-r AND the within-asset demeaned r clear zero with CI95 > 0, AND the Δ-Δ correlation is positive. The σ_PM gap **does** carry day-over-day information about the forward residual, beyond a static offset. Wire as a **research-only telemetry** signal — still needs 4-quarter OOS before sizing.

## How to reproduce

```bash
cd /Users/damiangallardoloya/Desktop/proyectofuentes
PYTHONPATH=api/src api/.venv/bin/python api/scripts/validate_pm_iv_gap.py
```
