# Cross-Sectional Regression Sweep

**Date generated:** 2026-05-15  
**API:** `http://127.0.0.1:8000` (local Prediction Terminal)  
**Window:** 2025-10-01 → 2026-05-01 (~6 months / ~130 trading obs)  
**Regression:** HAC (Newey-West, Andrews automatic bandwidth), log returns, strict alignment  
**Fits attempted:** 93 (3 packs × 31 tickers).  
**Successes:** 93  
**Failures:** 0  

## 1. Setup & methodology

Three economically themed factor packs were each regressed against 31 cross-sectional tickers spanning Tech, Financials, Energy, Materials, Cyclicals, and Crypto-adjacent names. Each `(pack, ticker)` request POSTs all four pack factors jointly to `/fit` so the resulting betas are partial — i.e. each factor's coefficient is the marginal contribution after controlling for the other three. We then capture (a) joint R² and adjusted R², (b) per-factor t-stats and p-values under HAC SE, (c) the model's built-in 5-fold CV `oos_r_squared`, (d) a naive sign-of-prediction `pseudo_backtest` (annualized Sharpe / total return / hit rate), and (e) the dominant factor by `share_of_explained_r_squared`. We requested `oos_test_fraction=0.2` to force a single 80/20 holdout `oos.test_r2` for every fit (the auto 5-fold `oos_r_squared` is only populated for some packs; we report it separately when present).

**Caveats.** With ~130 daily obs and 4 collinear-leaning factors, a single fit's adjusted-R² is a noisy estimate; many specifications are technically `weak_fit` per the API verdict. We are running 93 hypothesis tests at α=0.05 — under the null we'd expect ~5 false positives, so any individual `top_significant` should not be taken at face value (see §4 multiple-comparisons note). The `pseudo_backtest` is in-sample, no transaction cost, daily rebalanced — useful only as a directional sanity check, never as evidence of deployable alpha.

## 2. Pack A — Macro Fed / Recession

Factors: `no_fed_cuts_2026, fed_cuts_2_2026, twelve_plus_fed_cuts, us_recession_2026`  
Mean in-sample R² across the 31 tickers: **0.053**. Tickers with ≥1 significant factor (p<0.05): **16/31**. Tickers with verdict `well_specified`: **6**. Tickers with positive OOS R²: **11/31**.

| Ticker | Sector | n | R² | OOS R² (holdout) | Sig (n) | Top |t| factor (β, t, p) | Top contrib (share) | Sharpe | Hit | Verdict |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---|
| FCX | Materials | 130 | 0.151 | -0.052 | 2 | `us_recession_2026` (-0.0834, t=-3.49, p=0.000) | `twelve_plus_fed_cuts` (43.7%) | 6.18 | 0.631 | `well_specified` |
| COP | Energy | 130 | 0.102 | -0.589 | 1 | `no_fed_cuts_2026` (0.0258, t=2.97, p=0.003) | `no_fed_cuts_2026` (74.4%) | 3.77 | 0.600 | `well_specified` |
| BA | Cyclical | 130 | 0.093 | 0.189 | 2 | `no_fed_cuts_2026` (-0.0206, t=-3.07, p=0.002) | `no_fed_cuts_2026` (48.6%) | 3.68 | 0.569 | `well_specified` |
| MARA | Crypto | 130 | 0.093 | 0.194 | 2 | `twelve_plus_fed_cuts` (0.0573, t=2.99, p=0.003) | `twelve_plus_fed_cuts` (52.1%) | 5.01 | 0.662 | `well_specified` |
| HD | Cyclical | 130 | 0.081 | -0.046 | 1 | `fed_cuts_2_2026` (0.0351, t=2.35, p=0.019) | `fed_cuts_2_2026` (74.8%) | 3.01 | 0.562 | `well_specified` |
| XOM | Energy | 130 | 0.081 | -0.443 | 1 | `no_fed_cuts_2026` (0.0191, t=2.66, p=0.008) | `no_fed_cuts_2026` (82.5%) | 6.06 | 0.631 | `well_specified` |
| GOOGL | Tech | 130 | 0.079 | 0.049 | 1 | `us_recession_2026` (-0.0361, t=-2.27, p=0.024) | `no_fed_cuts_2026` (56.7%) | 3.65 | 0.554 | `borderline` |
| COIN | Crypto | 130 | 0.078 | 0.010 | 1 | `twelve_plus_fed_cuts` (0.0577, t=3.65, p=0.000) | `twelve_plus_fed_cuts` (92.1%) | 4.20 | 0.623 | `borderline` |
| BLK | Fin | 130 | 0.077 | -0.175 | 0 | `us_recession_2026` (-0.0440, t=-1.64, p=0.101) | `us_recession_2026` (62.7%) | 3.81 | 0.577 | `borderline` |
| AAPL | Tech | 130 | 0.073 | 0.179 | 1 | `us_recession_2026` (-0.0227, t=-2.35, p=0.019) | `fed_cuts_2_2026` (35.3%) | 0.83 | 0.538 | `borderline` |
| MSTR | Crypto | 130 | 0.065 | 0.110 | 2 | `twelve_plus_fed_cuts` (0.0441, t=2.93, p=0.003) | `twelve_plus_fed_cuts` (56.6%) | 3.45 | 0.623 | `borderline` |
| AMD | Tech | 130 | 0.062 | -0.159 | 1 | `twelve_plus_fed_cuts` (0.0476, t=2.11, p=0.035) | `twelve_plus_fed_cuts` (79.1%) | 3.48 | 0.554 | `borderline` |
| CVX | Energy | 130 | 0.055 | -0.516 | 0 | `no_fed_cuts_2026` (0.0133, t=1.76, p=0.078) | `no_fed_cuts_2026` (69.5%) | 3.67 | 0.623 | `borderline` |
| AMZN | Tech | 130 | 0.054 | 0.117 | 1 | `twelve_plus_fed_cuts` (0.0153, t=3.05, p=0.002) | `twelve_plus_fed_cuts` (49.2%) | 4.09 | 0.608 | `borderline` |
| TSLA | Tech | 130 | 0.053 | 0.081 | 1 | `us_recession_2026` (-0.0507, t=-2.65, p=0.008) | `us_recession_2026` (57.3%) | 3.39 | 0.608 | `borderline` |
| F | Cyclical | 130 | 0.053 | 0.058 | 1 | `fed_cuts_2_2026` (0.0309, t=2.05, p=0.040) | `fed_cuts_2_2026` (48.9%) | 2.36 | 0.569 | `borderline` |
| META | Tech | 130 | 0.052 | -0.040 | 1 | `twelve_plus_fed_cuts` (0.0161, t=2.93, p=0.003) | `twelve_plus_fed_cuts` (41.6%) | 0.18 | 0.492 | `borderline` |
| NVDA | Tech | 130 | 0.046 | -0.045 | 1 | `twelve_plus_fed_cuts` (0.0169, t=2.89, p=0.004) | `twelve_plus_fed_cuts` (51.3%) | 1.43 | 0.569 | `weak_fit` |
| GDX | Materials | 130 | 0.042 | -0.428 | 0 | `us_recession_2026` (-0.0566, t=-1.90, p=0.058) | `us_recession_2026` (56.9%) | 1.52 | 0.531 | `weak_fit` |
| MCD | Cyclical | 130 | 0.039 | -0.293 | 0 | `fed_cuts_2_2026` (0.0178, t=1.54, p=0.122) | `fed_cuts_2_2026` (82.9%) | 2.79 | 0.592 | `weak_fit` |
| NEM | Materials | 130 | 0.033 | -0.134 | 0 | `us_recession_2026` (-0.0494, t=-1.62, p=0.105) | `us_recession_2026` (53.3%) | 0.95 | 0.538 | `weak_fit` |
| ORCL | Tech | 130 | 0.030 | -0.085 | 0 | `twelve_plus_fed_cuts` (0.0175, t=1.85, p=0.065) | `twelve_plus_fed_cuts` (40.5%) | 2.69 | 0.554 | `weak_fit` |
| WFC | Fin | 130 | 0.026 | 0.090 | 0 | `us_recession_2026` (-0.0236, t=-1.45, p=0.148) | `us_recession_2026` (58.9%) | 1.17 | 0.585 | `weak_fit` |
| NFLX | Tech | 130 | 0.024 | 0.160 | 0 | `twelve_plus_fed_cuts` (0.0138, t=1.24, p=0.217) | `twelve_plus_fed_cuts` (76.0%) | 3.81 | 0.569 | `weak_fit` |
| MS | Fin | 130 | 0.022 | -0.363 | 0 | `us_recession_2026` (-0.0217, t=-1.31, p=0.190) | `us_recession_2026` (54.9%) | 1.60 | 0.562 | `weak_fit` |
| MSFT | Tech | 130 | 0.020 | -0.109 | 0 | `no_fed_cuts_2026` (-0.0116, t=-1.51, p=0.131) | `no_fed_cuts_2026` (76.4%) | 2.60 | 0.554 | `weak_fit` |
| BAC | Fin | 130 | 0.018 | -0.067 | 0 | `us_recession_2026` (-0.0204, t=-1.54, p=0.124) | `us_recession_2026` (91.3%) | 1.37 | 0.562 | `weak_fit` |
| JPM | Fin | 130 | 0.016 | -0.487 | 0 | `twelve_plus_fed_cuts` (0.0065, t=1.05, p=0.292) | `twelve_plus_fed_cuts` (51.3%) | 0.27 | 0.523 | `weak_fit` |
| DIS | Cyclical | 130 | 0.012 | -0.440 | 0 | `fed_cuts_2_2026` (0.0174, t=0.84, p=0.402) | `fed_cuts_2_2026` (84.9%) | 1.83 | 0.623 | `weak_fit` |
| GS | Fin | 130 | 0.009 | -1.070 | 0 | `twelve_plus_fed_cuts` (0.0048, t=0.75, p=0.451) | `us_recession_2026` (40.1%) | 0.26 | 0.508 | `weak_fit` |
| NKE | Cyclical | 130 | 0.006 | -0.799 | 0 | `fed_cuts_2_2026` (0.0129, t=0.69, p=0.493) | `fed_cuts_2_2026` (59.4%) | 2.62 | 0.592 | `weak_fit` |

## 3. Pack B — Geopolitical Risk

Factors: `china_invade_taiwan_2026, china_blockade_taiwan, trump_putin_meet_europe, trump_acquire_greenland`  
Mean in-sample R² across the 31 tickers: **0.051**. Tickers with ≥1 significant factor (p<0.05): **12/31**. Tickers with verdict `well_specified`: **3**. Tickers with positive OOS R²: **6/31**.

| Ticker | Sector | n | R² | OOS R² (holdout) | Sig (n) | Top |t| factor (β, t, p) | Top contrib (share) | Sharpe | Hit | Verdict |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---|
| XOM | Energy | 71 | 0.115 | -0.053 | 1 | `trump_acquire_greenland` (-0.0519, t=-2.58, p=0.010) | `trump_acquire_greenland` (77.1%) | 6.57 | 0.648 | `well_specified` |
| GS | Fin | 71 | 0.110 | -0.284 | 1 | `trump_putin_meet_europe` (0.0332, t=2.07, p=0.039) | `trump_putin_meet_europe` (59.0%) | 3.91 | 0.563 | `well_specified` |
| ORCL | Tech | 71 | 0.107 | -0.012 | 1 | `trump_putin_meet_europe` (0.0636, t=2.49, p=0.013) | `trump_putin_meet_europe` (80.4%) | 7.86 | 0.732 | `well_specified` |
| MS | Fin | 71 | 0.099 | -0.332 | 1 | `trump_putin_meet_europe` (0.0379, t=2.67, p=0.008) | `trump_putin_meet_europe` (89.5%) | 4.52 | 0.606 | `borderline` |
| COP | Energy | 71 | 0.084 | 0.027 | 1 | `trump_acquire_greenland` (-0.0557, t=-2.13, p=0.033) | `trump_acquire_greenland` (91.3%) | 5.70 | 0.634 | `borderline` |
| CVX | Energy | 71 | 0.084 | 0.031 | 0 | `trump_acquire_greenland` (-0.0434, t=-1.40, p=0.162) | `trump_acquire_greenland` (97.2%) | 6.12 | 0.676 | `borderline` |
| FCX | Materials | 71 | 0.078 | 0.096 | 0 | `trump_putin_meet_europe` (0.0427, t=1.66, p=0.098) | `china_blockade_taiwan` (46.2%) | 4.37 | 0.606 | `borderline` |
| BAC | Fin | 71 | 0.075 | -0.141 | 1 | `trump_putin_meet_europe` (0.0264, t=2.86, p=0.004) | `trump_putin_meet_europe` (95.6%) | 2.50 | 0.507 | `weak_fit` |
| JPM | Fin | 71 | 0.072 | -0.278 | 1 | `trump_putin_meet_europe` (0.0205, t=2.82, p=0.005) | `trump_putin_meet_europe` (60.2%) | 2.92 | 0.592 | `weak_fit` |
| WFC | Fin | 71 | 0.066 | -0.401 | 1 | `trump_putin_meet_europe` (0.0280, t=2.36, p=0.018) | `trump_putin_meet_europe` (88.5%) | 3.48 | 0.535 | `weak_fit` |
| BA | Cyclical | 71 | 0.057 | 0.047 | 1 | `trump_putin_meet_europe` (0.0289, t=2.86, p=0.004) | `trump_putin_meet_europe` (89.3%) | 4.47 | 0.563 | `weak_fit` |
| AMD | Tech | 71 | 0.049 | -0.200 | 1 | `trump_acquire_greenland` (-0.0914, t=-4.19, p=0.000) | `trump_acquire_greenland` (92.5%) | 2.27 | 0.592 | `weak_fit` |
| META | Tech | 71 | 0.046 | -0.003 | 1 | `trump_putin_meet_europe` (0.0218, t=2.27, p=0.023) | `trump_putin_meet_europe` (39.2%) | 1.90 | 0.577 | `weak_fit` |
| BLK | Fin | 71 | 0.046 | -0.334 | 0 | `trump_acquire_greenland` (-0.0247, t=-1.77, p=0.077) | `trump_putin_meet_europe` (52.4%) | -0.05 | 0.493 | `weak_fit` |
| MSFT | Tech | 71 | 0.045 | 0.002 | 1 | `trump_acquire_greenland` (0.0432, t=3.53, p=0.000) | `trump_acquire_greenland` (66.9%) | 3.31 | 0.620 | `weak_fit` |
| F | Cyclical | 71 | 0.043 | -0.227 | 0 | `trump_acquire_greenland` (-0.0264, t=-1.00, p=0.319) | `trump_acquire_greenland` (40.8%) | 2.35 | 0.620 | `weak_fit` |
| HD | Cyclical | 71 | 0.039 | 0.026 | 0 | `china_invade_taiwan_2026` (-0.0275, t=-1.24, p=0.217) | `trump_putin_meet_europe` (42.9%) | 2.95 | 0.577 | `weak_fit` |
| TSLA | Tech | 71 | 0.036 | -0.139 | 0 | `china_invade_taiwan_2026` (-0.0297, t=-0.95, p=0.342) | `china_blockade_taiwan` (35.4%) | 4.71 | 0.563 | `weak_fit` |
| NFLX | Tech | 71 | 0.033 | -0.078 | 0 | `trump_acquire_greenland` (0.0226, t=1.27, p=0.205) | `china_blockade_taiwan` (56.5%) | 1.09 | 0.563 | `weak_fit` |
| DIS | Cyclical | 71 | 0.033 | -0.435 | 0 | `trump_acquire_greenland` (-0.0310, t=-1.61, p=0.108) | `trump_acquire_greenland` (66.1%) | 2.88 | 0.634 | `weak_fit` |
| MSTR | Crypto | 71 | 0.029 | -0.009 | 0 | `china_invade_taiwan_2026` (-0.0853, t=-1.52, p=0.129) | `trump_putin_meet_europe` (32.1%) | 1.65 | 0.549 | `weak_fit` |
| MCD | Cyclical | 71 | 0.029 | -0.134 | 0 | `china_blockade_taiwan` (-0.0062, t=-1.01, p=0.311) | `china_blockade_taiwan` (35.1%) | 3.74 | 0.592 | `weak_fit` |
| GOOGL | Tech | 71 | 0.028 | -0.063 | 0 | `china_invade_taiwan_2026` (-0.0459, t=-0.96, p=0.338) | `china_invade_taiwan_2026` (91.6%) | -0.27 | 0.437 | `weak_fit` |
| AMZN | Tech | 71 | 0.027 | -0.199 | 0 | `trump_acquire_greenland` (0.0204, t=1.44, p=0.150) | `china_blockade_taiwan` (46.7%) | 1.52 | 0.563 | `weak_fit` |
| GDX | Materials | 71 | 0.027 | -0.063 | 0 | `trump_acquire_greenland` (0.0495, t=1.91, p=0.056) | `trump_acquire_greenland` (49.0%) | 2.03 | 0.620 | `weak_fit` |
| MARA | Crypto | 71 | 0.027 | -0.086 | 0 | `china_invade_taiwan_2026` (-0.1070, t=-1.13, p=0.258) | `china_invade_taiwan_2026` (55.6%) | 2.89 | 0.577 | `weak_fit` |
| NVDA | Tech | 71 | 0.027 | -0.063 | 0 | `trump_acquire_greenland` (-0.0232, t=-1.34, p=0.181) | `trump_acquire_greenland` (37.9%) | 2.62 | 0.592 | `weak_fit` |
| NEM | Materials | 71 | 0.026 | -0.033 | 0 | `trump_acquire_greenland` (0.0420, t=1.38, p=0.168) | `trump_acquire_greenland` (43.4%) | 1.25 | 0.563 | `weak_fit` |
| COIN | Crypto | 71 | 0.016 | -0.234 | 0 | `trump_putin_meet_europe` (0.0255, t=1.08, p=0.280) | `china_invade_taiwan_2026` (45.9%) | 2.46 | 0.634 | `weak_fit` |
| NKE | Cyclical | 71 | 0.016 | -0.266 | 0 | `trump_acquire_greenland` (-0.0214, t=-1.22, p=0.221) | `trump_acquire_greenland` (83.9%) | -0.45 | 0.521 | `weak_fit` |
| AAPL | Tech | 71 | 0.016 | -0.042 | 0 | `trump_putin_meet_europe` (0.0107, t=1.27, p=0.205) | `trump_putin_meet_europe` (65.2%) | 1.19 | 0.521 | `weak_fit` |

## 4. Pack C — Crypto Bullish

Factors: `btc_ath_jun, btc_200k_eoy, btc_150k_h1, btc_beats_gold`  
Mean in-sample R² across the 31 tickers: **0.121**. Tickers with ≥1 significant factor (p<0.05): **18/31**. Tickers with verdict `well_specified`: **12**. Tickers with positive OOS R²: **12/31**.

| Ticker | Sector | n | R² | OOS R² (holdout) | Sig (n) | Top |t| factor (β, t, p) | Top contrib (share) | Sharpe | Hit | Verdict |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---|
| MCD | Cyclical | 58 | 0.351 | 0.100 | 2 | `btc_ath_jun` (0.0207, t=4.57, p=0.000) | `btc_150k_h1` (65.1%) | 5.08 | 0.621 | `well_specified` |
| GDX | Materials | 58 | 0.236 | 0.216 | 2 | `btc_beats_gold` (-0.1018, t=-5.16, p=0.000) | `btc_beats_gold` (62.2%) | 6.32 | 0.672 | `well_specified` |
| MSFT | Tech | 58 | 0.233 | 0.206 | 2 | `btc_150k_h1` (0.0257, t=4.51, p=0.000) | `btc_150k_h1` (39.6%) | 7.05 | 0.690 | `well_specified` |
| NEM | Materials | 58 | 0.208 | 0.192 | 1 | `btc_beats_gold` (-0.0818, t=-3.81, p=0.000) | `btc_beats_gold` (49.1%) | 5.61 | 0.621 | `well_specified` |
| NKE | Cyclical | 58 | 0.194 | -0.500 | 2 | `btc_200k_eoy` (-0.0871, t=-3.32, p=0.001) | `btc_200k_eoy` (56.1%) | 4.38 | 0.586 | `well_specified` |
| COIN | Crypto | 58 | 0.185 | -0.034 | 0 | `btc_150k_h1` (0.0760, t=1.83, p=0.067) | `btc_150k_h1` (72.8%) | 7.91 | 0.672 | `borderline` |
| NVDA | Tech | 58 | 0.169 | 0.048 | 1 | `btc_150k_h1` (0.0304, t=2.23, p=0.026) | `btc_150k_h1` (64.4%) | 4.38 | 0.672 | `well_specified` |
| AMZN | Tech | 58 | 0.149 | -0.069 | 1 | `btc_ath_jun` (0.0277, t=2.41, p=0.016) | `btc_ath_jun` (57.4%) | 6.26 | 0.603 | `well_specified` |
| COP | Energy | 58 | 0.145 | 0.172 | 1 | `btc_beats_gold` (0.0417, t=2.83, p=0.005) | `btc_beats_gold` (45.7%) | 2.87 | 0.621 | `well_specified` |
| MS | Fin | 58 | 0.139 | -0.031 | 1 | `btc_beats_gold` (0.0307, t=2.44, p=0.015) | `btc_150k_h1` (52.8%) | 3.86 | 0.638 | `well_specified` |
| BLK | Fin | 58 | 0.126 | -0.037 | 1 | `btc_beats_gold` (0.0322, t=2.11, p=0.035) | `btc_beats_gold` (51.7%) | 4.88 | 0.586 | `well_specified` |
| ORCL | Tech | 58 | 0.126 | -0.092 | 1 | `btc_150k_h1` (0.0378, t=4.82, p=0.000) | `btc_150k_h1` (63.4%) | 4.41 | 0.621 | `well_specified` |
| GOOGL | Tech | 58 | 0.122 | -0.096 | 2 | `btc_200k_eoy` (0.0747, t=3.05, p=0.002) | `btc_200k_eoy` (64.0%) | 7.10 | 0.672 | `well_specified` |
| WFC | Fin | 58 | 0.108 | -0.043 | 1 | `btc_150k_h1` (0.0225, t=2.13, p=0.033) | `btc_150k_h1` (71.9%) | 3.59 | 0.586 | `borderline` |
| BAC | Fin | 58 | 0.104 | 0.028 | 1 | `btc_150k_h1` (0.0196, t=2.32, p=0.020) | `btc_150k_h1` (69.4%) | 3.97 | 0.534 | `borderline` |
| JPM | Fin | 58 | 0.102 | -0.278 | 0 | `btc_200k_eoy` (-0.0610, t=-1.66, p=0.097) | `btc_200k_eoy` (58.8%) | 0.93 | 0.448 | `borderline` |
| FCX | Materials | 58 | 0.102 | -0.131 | 2 | `btc_beats_gold` (-0.0494, t=-2.46, p=0.014) | `btc_beats_gold` (37.2%) | 5.54 | 0.655 | `borderline` |
| TSLA | Tech | 58 | 0.099 | -0.145 | 0 | `btc_150k_h1` (0.0128, t=1.73, p=0.084) | `btc_ath_jun` (50.9%) | 3.06 | 0.569 | `borderline` |
| MSTR | Crypto | 58 | 0.094 | 0.050 | 0 | `btc_150k_h1` (0.0491, t=1.52, p=0.128) | `btc_150k_h1` (54.3%) | 3.13 | 0.569 | `borderline` |
| NFLX | Tech | 58 | 0.090 | -0.868 | 1 | `btc_ath_jun` (0.0277, t=1.99, p=0.047) | `btc_ath_jun` (41.1%) | 3.41 | 0.534 | `borderline` |
| CVX | Energy | 58 | 0.086 | -0.254 | 1 | `btc_beats_gold` (0.0218, t=2.81, p=0.005) | `btc_ath_jun` (53.1%) | 5.68 | 0.707 | `weak_fit` |
| GS | Fin | 58 | 0.082 | 0.055 | 0 | `btc_150k_h1` (0.0226, t=1.51, p=0.131) | `btc_150k_h1` (84.6%) | 5.15 | 0.690 | `weak_fit` |
| XOM | Energy | 58 | 0.080 | -0.000 | 0 | `btc_150k_h1` (-0.0112, t=-1.69, p=0.090) | `btc_200k_eoy` (43.2%) | 1.89 | 0.552 | `weak_fit` |
| AMD | Tech | 58 | 0.075 | 0.012 | 0 | `btc_200k_eoy` (0.1528, t=1.91, p=0.056) | `btc_200k_eoy` (74.2%) | 4.17 | 0.569 | `weak_fit` |
| BA | Cyclical | 58 | 0.071 | 0.059 | 1 | `btc_beats_gold` (-0.0221, t=-2.29, p=0.022) | `btc_ath_jun` (44.7%) | 3.07 | 0.586 | `weak_fit` |
| AAPL | Tech | 58 | 0.068 | 0.047 | 0 | `btc_ath_jun` (0.0164, t=1.69, p=0.091) | `btc_ath_jun` (75.5%) | 2.71 | 0.500 | `weak_fit` |
| DIS | Cyclical | 58 | 0.064 | -0.286 | 0 | `btc_ath_jun` (-0.0209, t=-1.45, p=0.146) | `btc_ath_jun` (52.6%) | 1.44 | 0.500 | `weak_fit` |
| META | Tech | 58 | 0.060 | -0.169 | 0 | `btc_150k_h1` (0.0272, t=1.82, p=0.068) | `btc_150k_h1` (95.5%) | 3.21 | 0.586 | `weak_fit` |
| F | Cyclical | 58 | 0.037 | -0.371 | 0 | `btc_ath_jun` (0.0124, t=1.34, p=0.181) | `btc_ath_jun` (55.9%) | 1.81 | 0.586 | `weak_fit` |
| HD | Cyclical | 58 | 0.030 | -0.311 | 0 | `btc_150k_h1` (-0.0094, t=-1.05, p=0.292) | `btc_150k_h1` (58.6%) | 1.50 | 0.569 | `weak_fit` |
| MARA | Crypto | 58 | 0.004 | -0.066 | 0 | `btc_ath_jun` (0.0137, t=0.37, p=0.709) | `btc_ath_jun` (61.2%) | -3.08 | 0.466 | `weak_fit` |

## 5. Cross-pack ticker analysis

**Tickers with ≥1 significant factor across ALL 3 packs (broad-beta names — likely confounded):**

- **BA** (Cyclical)
- **COP** (Energy)

**Tickers significant in exactly 1 pack (cleanest sector / single-narrative exposure):**

- **AAPL** (Tech) → `macro_fed_recession`
- **BLK** (Fin) → `crypto_bullish`
- **COIN** (Crypto) → `macro_fed_recession`
- **CVX** (Energy) → `crypto_bullish`
- **F** (Cyclical) → `macro_fed_recession`
- **GDX** (Materials) → `crypto_bullish`
- **GS** (Fin) → `geopolitical_risk`
- **HD** (Cyclical) → `macro_fed_recession`
- **JPM** (Fin) → `geopolitical_risk`
- **MARA** (Crypto) → `macro_fed_recession`
- **MCD** (Cyclical) → `crypto_bullish`
- **MSTR** (Crypto) → `macro_fed_recession`
- **NEM** (Materials) → `crypto_bullish`
- **NFLX** (Tech) → `crypto_bullish`
- **NKE** (Cyclical) → `crypto_bullish`
- **TSLA** (Tech) → `macro_fed_recession`

**Tickers significant in exactly 2 packs (mixed exposure):**

- AMD (Tech) → `geopolitical_risk`, `macro_fed_recession`
- AMZN (Tech) → `crypto_bullish`, `macro_fed_recession`
- BAC (Fin) → `crypto_bullish`, `geopolitical_risk`
- FCX (Materials) → `crypto_bullish`, `macro_fed_recession`
- GOOGL (Tech) → `crypto_bullish`, `macro_fed_recession`
- META (Tech) → `geopolitical_risk`, `macro_fed_recession`
- MS (Fin) → `crypto_bullish`, `geopolitical_risk`
- MSFT (Tech) → `crypto_bullish`, `geopolitical_risk`
- NVDA (Tech) → `crypto_bullish`, `macro_fed_recession`
- ORCL (Tech) → `crypto_bullish`, `geopolitical_risk`
- WFC (Fin) → `crypto_bullish`, `geopolitical_risk`
- XOM (Energy) → `geopolitical_risk`, `macro_fed_recession`

**Tickers with NO significant factor in any pack:**

DIS

### Sector-level mean R² by pack

| Sector | Macro Fed/Rec | Geopolitical | Crypto Bullish |
|---|---:|---:|---:|
| Crypto | 0.079 | 0.024 | 0.094 |
| Cyclical | 0.047 | 0.036 | 0.125 |
| Energy | 0.079 | 0.094 | 0.104 |
| Fin | 0.028 | 0.078 | 0.110 |
| Materials | 0.075 | 0.044 | 0.182 |
| Tech | 0.049 | 0.042 | 0.119 |

## 6. Honest discussion

### 6.1 Which pack had the highest hit rate of significant fits?

| Pack | Tickers w/ ≥1 sig (p<0.05) | Mean R² | Mean OOS R² | `well_specified` count |
|---|---:|---:|---:|---:|
| macro_fed_recession | 16/31 | 0.053 | -0.165 | 6 |
| geopolitical_risk | 12/31 | 0.051 | -0.125 | 3 |
| crypto_bullish | 18/31 | 0.121 | -0.084 | 12 |

### 6.2 Sign consistency across tickers

For each headline factor, count tickers with negative vs positive β (sign of partial exposure). Sign should be consistent within a sector if the factor really is a macro shock.

**macro_fed_recession**

| Factor | β<0 | β>0 | mean β | mean |t| | n p<0.10 |
|---|---:|---:|---:|---:|---:|
| `no_fed_cuts_2026` | 17 | 14 | -0.000 | 0.923 | 5 |
| `fed_cuts_2_2026` | 8 | 23 | 0.010 | 0.896 | 4 |
| `twelve_plus_fed_cuts` | 1 | 30 | 0.014 | 1.382 | 10 |
| `us_recession_2026` | 28 | 3 | -0.030 | 1.356 | 9 |

**geopolitical_risk**

| Factor | β<0 | β>0 | mean β | mean |t| | n p<0.10 |
|---|---:|---:|---:|---:|---:|
| `china_invade_taiwan_2026` | 19 | 12 | -0.005 | 0.637 | 1 |
| `china_blockade_taiwan` | 8 | 23 | 0.006 | 0.606 | 1 |
| `trump_putin_meet_europe` | 7 | 24 | 0.016 | 1.279 | 9 |
| `trump_acquire_greenland` | 17 | 14 | -0.006 | 1.225 | 6 |

**crypto_bullish**

| Factor | β<0 | β>0 | mean β | mean |t| | n p<0.10 |
|---|---:|---:|---:|---:|---:|
| `btc_ath_jun` | 9 | 22 | 0.013 | 1.458 | 11 |
| `btc_200k_eoy` | 23 | 8 | -0.020 | 1.038 | 6 |
| `btc_150k_h1` | 7 | 24 | 0.013 | 1.530 | 12 |
| `btc_beats_gold` | 7 | 24 | 0.005 | 1.337 | 11 |

### 6.3 OOS survivors (in-sample R² > 0 AND OOS R² > 0)

| Pack | Ticker | R² | OOS R² | Sharpe (in-sample) |
|---|---|---:|---:|---:|
| crypto_bullish | GDX | 0.236 | 0.216 | 6.32 |
| crypto_bullish | MSFT | 0.233 | 0.206 | 7.05 |
| macro_fed_recession | MARA | 0.093 | 0.194 | 5.01 |
| crypto_bullish | NEM | 0.208 | 0.192 | 5.61 |
| macro_fed_recession | BA | 0.093 | 0.189 | 3.68 |
| macro_fed_recession | AAPL | 0.073 | 0.179 | 0.83 |
| crypto_bullish | COP | 0.145 | 0.172 | 2.87 |
| macro_fed_recession | NFLX | 0.024 | 0.160 | 3.81 |
| macro_fed_recession | AMZN | 0.054 | 0.117 | 4.09 |
| macro_fed_recession | MSTR | 0.065 | 0.110 | 3.45 |
| crypto_bullish | MCD | 0.351 | 0.100 | 5.08 |
| geopolitical_risk | FCX | 0.078 | 0.096 | 4.37 |
| macro_fed_recession | WFC | 0.026 | 0.090 | 1.17 |
| macro_fed_recession | TSLA | 0.053 | 0.081 | 3.39 |
| crypto_bullish | BA | 0.071 | 0.059 | 3.07 |
| macro_fed_recession | F | 0.053 | 0.058 | 2.36 |
| crypto_bullish | GS | 0.082 | 0.055 | 5.15 |
| crypto_bullish | MSTR | 0.094 | 0.050 | 3.13 |
| macro_fed_recession | GOOGL | 0.079 | 0.049 | 3.65 |
| crypto_bullish | NVDA | 0.169 | 0.048 | 4.38 |
| geopolitical_risk | BA | 0.057 | 0.047 | 4.47 |
| crypto_bullish | AAPL | 0.068 | 0.047 | 2.71 |
| geopolitical_risk | CVX | 0.084 | 0.031 | 6.12 |
| crypto_bullish | BAC | 0.104 | 0.028 | 3.97 |
| geopolitical_risk | COP | 0.084 | 0.027 | 5.70 |
| geopolitical_risk | HD | 0.039 | 0.026 | 2.95 |
| crypto_bullish | AMD | 0.075 | 0.012 | 4.17 |
| macro_fed_recession | COIN | 0.078 | 0.010 | 4.20 |
| geopolitical_risk | MSFT | 0.045 | 0.002 | 3.31 |

**Count:** 29 of 93 fits cleared the (in-sample R²>0.02 AND OOS R²>0) bar.

### 6.4 Multiple-comparisons concern

With 93 joint regressions × 4 factors per regression = 372 individual coefficient tests, and a nominal α=0.05, the expected false-positive count under a global null is **~18.6**. Counting actual `top_significant` hits (p<0.05 from `top_significant`) across all 93 fits:

- **Observed coefficient hits at p<0.05:** 56
- **Expected under null (372 × 0.05):** ~18.6
- **Inflation factor:** 3.01× nominal

If the inflation factor is close to 1, the entire result set is consistent with noise. A more honest threshold is BH-FDR at q=0.10 within each pack (4 factors × 31 tickers = 124 tests per pack), or — better — a Politis-Romano stationary bootstrap p-value before claiming any single fit as 'real'. **None of the table entries below are claimed as deployable alpha; this is exploratory cross-section.**

### 6.5 Sample-size disparity across packs

The three packs have **very different effective sample sizes** because each pack's window is bounded by the shortest factor's price history (strict alignment).

| Pack | Median n_obs | Implication |
|---|---:|---|
| macro_fed_recession | 130 | Full window. Standard errors trustworthy; pseudo-bt has ~6 mo of trades. |
| geopolitical_risk | 71 | Partial window. ~3 mo of trades; t-stats slightly inflated by small-sample HAC. |
| crypto_bullish | 58 | Short window. <3 mo; treat all p-values with extra skepticism, OOS holdout is n≈11. |

**Consequence:** Pack C (Crypto Bullish) has the highest mean R² (0.121) partly because n is smallest — 4 factors against 58 obs explains more by chance. The same regression on Pack C with 130 obs would likely shrink R² toward 0.05. This is the single biggest reason the §7 winners deserve only tentative validation.

### 6.6 Surprising findings (cherry-picked)

- **MCD × `btc_ath_jun`** has β=+0.021, t=+4.57, p<0.001 with R²=0.351 — the highest R² of any fit. There is no obvious economic channel from BTC reaching an all-time high in June to McDonald's returns; this is almost certainly a confound with the broader risk-on regime when the crypto pack lifts. The high t-stat is real, the causal story is not.
- **GDX × `btc_beats_gold`** has β=-0.102, t=-5.16 (the largest |t| in the entire sweep). The sign is economically intuitive — if the market prices BTC outperforming gold, gold-miner ETFs should weaken — and OOS R² survives at +0.216. This is the cleanest factor-narrative match in the sweep.
- **AMD × `trump_acquire_greenland`** registers β=-0.091, t=-4.19, p<0.001 in the geopolitical pack. There's no plausible AMD–Greenland channel; the 'Greenland' contract is likely acting as a generic Trump-policy-uncertainty proxy that happened to correlate with semiconductor risk-off days. Classic data-mining artifact and a useful caution.

**Three cleanest sector-narrative signals (intuitive sign, p<0.05, OOS holdout > 0):**

- **GDX × `btc_beats_gold`** (β=-0.102) — bullish-BTC-vs-gold contract loads negatively on gold miners.
- **NEM × `btc_beats_gold`** (β=-0.082) — same story; cross-confirms the BTC-gold seesaw.
- **TSLA × `us_recession_2026`** (β=-0.051, t=-2.65) — recession odds rise → high-beta consumer cyclical falls. Sign matches macro intuition and survives OOS holdout (+0.081).

## 7. Top-3 candidate (pack, ticker) pairs for further validation

Ranked by composite score: positive OOS R² (4×), in-sample R² (1×), positive Sharpe (0.05×). Only fits with **OOS R² > 0** are eligible.

### #1 — GDX × crypto_bullish  (score 1.415)

- **R²** = 0.236, **OOS R²** = 0.216, **Adj R²** = 0.178
- **n_obs_used:** 58
- **Verdict:** `well_specified` — _4 factors fit on 58 obs with R²=0.24; 2 significant at p<0.05._
- **Significant factors (p<0.05):** `btc_beats_gold`, `btc_ath_jun`
- **Top |t| factor:** `btc_beats_gold` (β=-0.1018, t=-5.16, p=0.000)
- **Dominant contributor (Δ-R² share):** `btc_beats_gold` (62.2%)
- **Pseudo-backtest (in-sample, no costs):** Sharpe=6.32, hit-rate=0.672

**Why interesting:** the pack's narrative (`crypto_bullish`) is most strongly anchored on `btc_beats_gold` for GDX, the 80/20 holdout OOS R² is positive (only 29 of 93 fits clear OOS>0), and the sign of the dominant factor is economically interpretable. **Next step:** re-run with HAC + bootstrap (`bootstrap_iters=1000`), 4-quarter robustness, and Newey-West permutation null before flagging as deployable. Note that all three winners are in Pack C with n=58 obs only — a smaller sample makes the OOS holdout (n_test≈11) noisier than it looks.

### #2 — MSFT × crypto_bullish  (score 1.408)

- **R²** = 0.233, **OOS R²** = 0.206, **Adj R²** = 0.175
- **n_obs_used:** 58
- **Verdict:** `well_specified` — _4 factors fit on 58 obs with R²=0.23; 2 significant at p<0.05._
- **Significant factors (p<0.05):** `btc_150k_h1`, `btc_200k_eoy`
- **Top |t| factor:** `btc_150k_h1` (β=0.0257, t=4.51, p=0.000)
- **Dominant contributor (Δ-R² share):** `btc_150k_h1` (39.6%)
- **Pseudo-backtest (in-sample, no costs):** Sharpe=7.05, hit-rate=0.690

**Why interesting:** the pack's narrative (`crypto_bullish`) is most strongly anchored on `btc_150k_h1` for MSFT, the 80/20 holdout OOS R² is positive (only 29 of 93 fits clear OOS>0), and the sign of the dominant factor is economically interpretable. **Next step:** re-run with HAC + bootstrap (`bootstrap_iters=1000`), 4-quarter robustness, and Newey-West permutation null before flagging as deployable. Note that all three winners are in Pack C with n=58 obs only — a smaller sample makes the OOS holdout (n_test≈11) noisier than it looks.

### #3 — NEM × crypto_bullish  (score 1.255)

- **R²** = 0.208, **OOS R²** = 0.192, **Adj R²** = 0.148
- **n_obs_used:** 58
- **Verdict:** `well_specified` — _4 factors fit on 58 obs with R²=0.21; 1 significant at p<0.05._
- **Significant factors (p<0.05):** `btc_beats_gold`
- **Top |t| factor:** `btc_beats_gold` (β=-0.0818, t=-3.81, p=0.000)
- **Dominant contributor (Δ-R² share):** `btc_beats_gold` (49.1%)
- **Pseudo-backtest (in-sample, no costs):** Sharpe=5.61, hit-rate=0.621

**Why interesting:** the pack's narrative (`crypto_bullish`) is most strongly anchored on `btc_beats_gold` for NEM, the 80/20 holdout OOS R² is positive (only 29 of 93 fits clear OOS>0), and the sign of the dominant factor is economically interpretable. **Next step:** re-run with HAC + bootstrap (`bootstrap_iters=1000`), 4-quarter robustness, and Newey-West permutation null before flagging as deployable. Note that all three winners are in Pack C with n=58 obs only — a smaller sample makes the OOS holdout (n_test≈11) noisier than it looks.

## 8. Files

- Per-fit JSON dumps: `/tmp/cross-section/{pack}_{ticker}.json`
- Lightweight summary: `/tmp/cross-section/_all_results.json`
- Sweep script: `/tmp/cross-section/sweep.py`
- Report builder: `/tmp/cross-section/build_report.py`

**Failures:** _none — all 93 fits returned 200 OK._
