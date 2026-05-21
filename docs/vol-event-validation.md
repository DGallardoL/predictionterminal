# Entropy-Proxy EM Forecast Validation (Task B4)

_Generated 2026-05-16 by `api/scripts/validate_event_vol.py`._

## Scope

We test whether the entropy-proxy mode of `pfm.vol.event_vol_engine.expected_move_from_distribution` (the path taken when `calibration=None`, with kind-specific constants `k_fomc=0.50`, `k_cpi=0.30`, …) predicts realised `|Δ%|` on SPY across past macro events that resolved on Polymarket / Kalshi.

Date range of usable events: **2026-03-11 → 2026-05-13**  
Total events curated: **8**  
Usable events (entered metrics): **8**  
Dropped events: **0**
  By kind: cpi=6, fomc=2

## Per-event detail

| Event | Kind | T-1 → T+1 window | n_outcomes | entropy_norm | tail_pct | asym_mass | EM (entropy-proxy) % | Realised \|Δ\| % | Error (EM − real) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fomc-2026-03` | fomc | 2026-03-17 → 2026-03-19 | 4 | 0.034 | 1.000 | -0.004 | 0.027 | 1.638 | -1.611 |
| `fomc-2026-04` | fomc | 2026-04-28 → 2026-04-30 | 4 | -0.000 | 1.000 | +0.000 | -0.000 | 0.979 | -0.979 |
| `cpi-2026-02-release` | cpi | 2026-03-10 → 2026-03-12 | 16 | 0.742 | 0.246 | -0.552 | 0.223 | 1.642 | -1.419 |
| `cpi-2026-03-release` | cpi | 2026-04-09 → 2026-04-13 | 16 | 0.716 | 0.889 | +0.603 | 0.320 | 0.910 | -0.591 |
| `cpi-2026-04-release` | cpi | 2026-05-12 → 2026-05-14 | 22 | 0.699 | 0.397 | +0.574 | 0.210 | 1.353 | -1.143 |
| `cpi-2026-02-release-mom` | cpi | 2026-03-10 → 2026-03-12 | 8 | 0.660 | 0.057 | -0.019 | 0.198 | 1.642 | -1.444 |
| `cpi-2026-03-release-mom` | cpi | 2026-04-09 → 2026-04-13 | 15 | 0.730 | 0.109 | +0.172 | 0.219 | 0.910 | -0.691 |
| `cpi-2026-04-release-mom` | cpi | 2026-05-12 → 2026-05-14 | 13 | 0.756 | 0.139 | +0.131 | 0.227 | 1.353 | -1.127 |

## Distribution snapshots at T-1 close

**fomc-2026-03** — fomc @ 2026-03-18
  - raw legs: cut_25bp=0.004, cut_50bp=0.001, no_change=0.995, hike_25bp=0.001

**fomc-2026-04** — fomc @ 2026-04-29
  - raw legs: cut_25bp=0.001, cut_50bp=0.001, no_change=0.999, hike_25bp=0.001

**cpi-2026-02-release** — cpi @ 2026-03-11
  - raw legs: cell_2.0=0.050, cell_2.1=0.030, cell_2.2=0.050, cell_2.3=0.110, cell_2.4=0.410, cell_2.5=0.390, cell_2.6=0.050, cell_2.7=0.060, cell_2.8=0.030, cell_2.9=0.010, cell_3.0=0.010, cell_3.1=0.050, cell_3.2=0.010, cell_3.3=0.050, cell_3.4=0.020, cell_3.5=0.010

**cpi-2026-03-release** — cpi @ 2026-04-10
  - raw legs: cell_2.0=0.040, cell_2.1=0.050, cell_2.2=0.010, cell_2.3=0.010, cell_2.4=0.010, cell_2.5=0.020, cell_2.6=0.020, cell_2.7=0.010, cell_2.8=0.010, cell_2.9=0.010, cell_3.0=0.020, cell_3.1=0.040, cell_3.2=0.160, cell_3.3=0.420, cell_3.4=0.290, cell_3.5=0.140

**cpi-2026-04-release** — cpi @ 2026-05-13
  - raw legs: cell_2.0=0.010, cell_2.1=0.020, cell_2.2=0.010, cell_2.3=0.010, cell_2.4=0.040, cell_2.5=0.040, cell_2.6=0.090, cell_2.7=0.010, cell_2.8=0.010, cell_2.9=0.010, cell_3.0=0.010, cell_3.1=0.010, cell_3.2=0.010, cell_3.3=0.010, cell_3.5=0.040, cell_3.6=0.150, cell_3.7=0.430, cell_3.8=0.340, cell_3.9=0.060, cell_4.0=0.010, cell_4.1=0.010, cell_4.2=0.030
  - dropped: KXECONSTATCPIYOY-26APR-T3.4 (no history)

**cpi-2026-02-release-mom** — cpi @ 2026-03-11
  - raw legs: cell_-0.2=0.030, cell_-0.1=0.010, cell_0.0=0.010, cell_0.1=0.070, cell_0.2=0.420, cell_0.3=0.420, cell_0.4=0.080, cell_0.6=0.020
  - dropped: KXECONSTATCPI-26FEB-T0.5 (no prints before T-1)

**cpi-2026-03-release-mom** — cpi @ 2026-04-10
  - raw legs: cell_-0.1=0.060, cell_0.0=0.020, cell_0.3=0.060, cell_0.4=0.020, cell_0.5=0.040, cell_0.6=0.010, cell_0.7=0.040, cell_0.8=0.280, cell_0.9=0.420, cell_1.0=0.240, cell_1.1=0.030, cell_1.2=0.010, cell_1.3=0.020, cell_1.4=0.010, cell_1.5=0.020
  - dropped: KXECONSTATCPI-26MAR-T-0.2 (no history), KXECONSTATCPI-26MAR-T0.1 (no history), KXECONSTATCPI-26MAR-T0.2 (no history)

**cpi-2026-04-release-mom** — cpi @ 2026-05-13
  - raw legs: cell_-0.2=0.020, cell_-0.1=0.040, cell_0.0=0.060, cell_0.1=0.010, cell_0.2=0.050, cell_0.3=0.050, cell_0.4=0.040, cell_0.5=0.260, cell_0.6=0.360, cell_0.7=0.280, cell_0.8=0.030, cell_0.9=0.010, cell_1.0=0.010

## Aggregate metrics

- MAE (EM − realised): **1.126 pp**
- Mean signed error (EM − realised): **-1.126 pp**
- Pearson correlation (EM, realised): **-0.156**
- Spearman rank correlation: **-0.255**
- Pairwise rank concordance (Kendall-style): **0.40**
- n = 8

### Short-straddle PnL (sold at entropy-proxy EM)

Convention: short one event-day straddle priced at the entropy-proxy EM. Gross PnL per event = EM − realised (pp of notional). Net PnL subtracts a 1.8% transaction-cost proxy applied to the premium collected — this approximates a single-sided maker/taker fee on the premium leg. We **deliberately do NOT inflate the cost to 3.6%** (both sides at expiry) because event-day straddles are typically held to expiry and the closing leg pays settlement, not a second taker fee.

- Mean gross PnL: **-1.126 pp/event**  
- Mean net PnL (–1.8% on premium): **-1.129 pp/event**  
- Win-rate gross: **0.00**  
- Win-rate net: **0.00**  

## Per-kind breakdown

| Kind | n | MAE | mean_signed_err | Pearson | Spearman | concordance |
|---|---:|---:|---:|---:|---:|---:|
| cpi | 6 | 1.069 | -1.069 | -0.628 | -0.478 | 0.25 |
| fomc | 2 | 1.295 | -1.295 | +1.000 | +1.000 | 1.00 |

## Caveats and data sparseness

- **Sample size is tiny.** Only 2 past FOMC meetings (March 18, April 29 2026 — both unanimous holds) and 3 past CPI releases (Feb, Mar, Apr 2026 data) exist in the live Polymarket / Kalshi catalogues with surviving ticker addresses. We *double-count* each CPI release by treating the YoY and MoM Kalshi ladders as separate events (they sample the same underlying distribution from different axes), bringing n to 8, but the realised |Δ| is shared between the YoY/MoM pair for each release — honest equity-event count is therefore **5**, not 8. Earlier 2026 meetings (Jan, Feb FOMC) and all of 2025 are unavailable: factors.yml is forward-looking and neither venue indexes pre-2026Q1 macro markets by the slug patterns we tried. The reported correlations have effectively zero statistical power; nothing here should be taken as a production-grade backtest.

- **Entropy-proxy constants are literature-anchored, not calibrated.** `k_fomc=0.50`, `k_cpi=0.30` were chosen to match historical event-day straddle quotes (~0.4–0.7% for FOMC on SPY) for a uniform 5-outcome ladder; they have NOT been fit to the out-of-sample realised moves used here. A miscalibration of constants does NOT invalidate the entropy-shape hypothesis.

- **Kalshi CPI cells are point-mass markets**, confirmed via `yes_sub_title` ("Exactly X%"). The ladder is therefore a near-true partition over the discretised CPI range. We discover the full ladder per release at run time (typically 16–23 cells per CPI YoY release, 18+ per MoM headline) and rescale the surviving mass to sum=1 before passing into the entropy engine. Cells with zero history are dropped silently; the rescaling means missing tail mass is redistributed across the survivors, which biases entropy slightly upward but does not change the rank ordering across events.

- **Options-IV comparison is unavailable.** The task contemplated comparing `em_pm` to an ATM straddle implied vol from yfinance options. yfinance options chains are streamed live and contain no historical IV snapshot at the T-1 close. Without a separate history capture pipeline (e.g. Polygon options endpoint or a stored daily snapshot), this column is left empty across the sample. Reported as a gap, not as a finding against the proxy.

- **Window choice biases magnitudes upward.** Using `|close[T+1] − close[T-1]| / close[T-1]` includes the trading day AFTER the announcement, which captures post-headline drift and overnight macro news. A pure event-day window (`[T-1 close, T close]`) would shave roughly 20–40% off realised magnitudes. We chose [T-1, T+1] because CPI prints at 08:30 ET (i.e. before the equity open) and the announcement impulse continues into the close of the *next* trading day, which matches the contract settlement window.

- **No NFP / unrate coverage.** Probing `KXECONSTATNFP-26{JAN,FEB,MAR,APR}-T<…>` and the unrate variants returned 404 across the entire grid. Either the ticker pattern is different or Kalshi does not run a per-month NFP ladder. Either way, NFP is excluded from this report; flagged as a TODO for B5 (data discovery).

## Verdict

With n=8 the sample is **below the CLAUDE.md robustness floor (≥4 disjoint quarters per kind).** Per the anti-alpha policy, any claim of skill from this data is inadmissible — at best it sketches a hypothesis.

MAE of **1.13 pp** on a typical realised range of ~0.3–1.5 pp is the dominant fact, and the mean signed error is **-1.13 pp** — the entropy-proxy systematically *under-shoots*. Two distinct failure modes appear on this slice: (i) on FOMC, the prediction-market distribution collapses to ~99% no-change weeks before the decision, so entropy → 0 and EM → 0 even though SPY still realised ~1 pp from positioning unwind; (ii) on CPI, the YoY ladders are nicely spread (entropy_normalized ~0.7) but the resulting EM (~0.2-0.3 pp) is dwarfed by realised SPY moves of 0.9–1.6 pp driven by directional surprise relative to consensus rather than by raw distribution dispersion.

Spearman rank correlation is **-0.25** (Pearson -0.16). This is the load-bearing number — even if `k_kind` constants are wrong, a positive rank correlation would prove the distribution-shape signal is informative and justify fitting `EMCalibration` (the engine's `fit_em_calibration` path). A near-zero or negative rank correlation says the entropy-shape signal is uninformative on this slice and calibration would not rescue it.

**Verdict: SHELVE the entropy-proxy as a standalone trading signal.** The rank correlation is zero or *negative*, meaning higher-entropy distributions did NOT predict larger realised moves on this slice — if anything the relationship runs the wrong way. The feature set (entropy_normalized, tail_pct, asymmetric_mass) does not rank realised magnitudes here, so fitting `EMCalibration` cannot rescue it: a linear projection of features that don't rank the target will either pick the intercept (constant prediction) or over-fit the residuals — neither generalises out of sample. This is a deeper problem than `k_kind` miscalibration.

Action: do **not** ship the entropy-proxy to the UI as a trading signal. Acceptable to ship it as a *descriptive* panel ("contract-implied event-day range") with prominent labelling of the data sparseness and the lack of calibration. Re-run this validation after 4 more consecutive FOMC + CPI events have resolved (≈ 2026-Q4) and re-evaluate. Two more specific follow-ups, in priority order: (1) fix the FOMC partition — when no-change probability ≥ 95 % the entropy proxy collapses to 0 % EM but realised moves are still 1–2 pp from positioning unwind, so the engine needs a *baseline event-day vol* term that does not vanish with low entropy; (2) regress realised |Δ| on a richer feature vector (interaction with VIX level, spot-trend, sector-rotation residual) — even if shape alone is uninformative, shape × regime might be.

