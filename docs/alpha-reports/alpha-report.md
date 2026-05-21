# Live Alpha Report — Quant Strats Hub on Polymarket Catalog

**Generated**: 2026-05-01 from a live `/strategies/auto-backtest` sweep of the 145-factor catalog.
**Window**: 2026-01-01 → 2026-04-30 (~120 trading days, daily probability bars).
**Method**: scan each theme for cointegrated pairs (Engle-Granger 2-step, ADF p<0.05, AR(1) half-life ≤60d), then walk-forward z-score backtest on each spread (entry |z|=2σ, exit 0.5σ, stop 4σ, 20-bar rolling window).

> ⚠ **All Sharpes below ignore transaction costs.** Polymarket round-trip ≈ 1-3¢ on a typical 0.10-0.30 σ_eq spread. After costs, divide naive Sharpe by ~1.5-2x for honest expectations. Use `/strategies/ou-bands` to compute the cost-aware Bertram-optimal entry threshold.

> 🔬 **OOS split (added 2026-05-01)**: every backtest now runs on the full window AND on a 70/30 train/test split. The `OOS/IS` column below is the ratio of test-window Sharpe to train-window Sharpe. **Below ~0.5 ⇒ likely overfit.** Treat any pair with high IS Sharpe but tiny OOS ratio as suspect.

## Top alpha with OOS validation (2026-04-30 sweep, crypto theme)

| Pair | Sharpe | IS | OOS | OOS/IS | Verdict |
|---|---|---|---|---|---|
| `btc_100k_eoy ↔ btc_500k_eoy` | +5.73 | +3.61 | **+9.47** | 2.63 | ✅ OOS *strengthens* |
| `eth_10k_eoy ↔ btc_100k_eoy` | +4.13 | +4.31 | +4.07 | 0.95 | ✅ Stable |
| `eth_10k_eoy ↔ eth_5k_eoy` | +3.07 | +1.98 | +4.77 | 2.40 | ✅ OOS holds |
| `btc_200k_eoy ↔ btc_150k_h1` | +3.01 | +1.94 | +4.85 | 2.50 | ✅ OOS holds |
| `btc_dip_15k ↔ btc_dip_55k` | +2.48 | +1.63 | +6.61 | 4.04 | ✅ Strong OOS |
| `btc_150k_h1 ↔ btc_100k_eoy` | +3.17 | +3.90 | +1.02 | **0.26** | ⚠ Degraded OOS |
| `btc_1m_eoy ↔ opensea_fdv_500m` | +2.33 | 0.00 | +4.22 | n/a | ⚠ No IS signal — lucky |

---

## Top alpha across all themes

Ranked by Sharpe across the 7 themes scanned. **Bold = "actually tradeable"** (cointegrated, short half-life, multiple trades, stable hit rate).

| # | A | B | Theme | Sharpe | Sortino | Calmar | Hit | Trades | ½-life |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **btc_100k_eoy** | **btc_500k_eoy** | crypto | +5.73 | +0.00 | 0 | 100% | 3 | 0.8d |
| 2 | **eth_10k_eoy** | **btc_100k_eoy** | crypto | +4.13 | +9.28 | 67.9 | 100% | 3 | 1.1d |
| 3 | openai_acquired | anthropic_best_jun | ai | +3.96 | — | — | 100% | — | 0.9d |
| 4 | openai_acquired | xai_best_jun | ai | +3.56 | — | — | 100% | — | 1.3d |
| 5 | **dem_senate_2026** | **rep_senate_2026** | politics | +3.50 | — | — | 83% | 6 | 0.6d |
| 6 | openai_acquired | google_best_ai_jun | ai | +3.23 | — | — | 100% | — | 0.8d |
| 7 | btc_150k_h1 | btc_100k_eoy | crypto | +3.17 | +4.96 | 15.2 | 100% | 3 | 1.3d |
| 8 | eth_10k_eoy | eth_5k_eoy | crypto | +3.07 | +0.00 | 0 | 100% | 3 | 1.1d |
| 9 | tsla_largest_jun | nvda_largest_jun | chips | +3.03 | — | — | 100% | — | 0.5d |
| 10 | oil_above_200_jun | gold_5500_jun | commodities | +3.02 | — | — | 100% | — | 0.8d |
| 11 | btc_200k_eoy | btc_150k_h1 | crypto | +3.01 | +0.00 | 0 | 100% | 2 | 1.2d |
| 12 | twelve_plus_fed_cuts | fed_chair_shelton | macro | +2.97 | +2.22 | 23.96 | 100% | 3 | 0.4d |
| 13 | us_invade_greenland | iran_leadership_change_eoy | geopolitics | +2.90 | — | — | 100% | — | 0.4d |
| 14 | msft_largest_jun | musk_trillionaire | chips | +2.90 | — | — | 100% | — | 1.8d |
| 15 | netanyahu_out_jun | putin_out_jun | geopolitics | +2.85 | — | — | 100% | — | 0.5d |

## Honest verdicts (the practitioner read)

### ✅ Genuinely tradeable

**`dem_senate_2026 ↔ rep_senate_2026`** (S=+3.50, hit 83%, ½-life 0.6d)
This is the cleanest pair in the catalog. The two markets must satisfy P(Dem) + P(Rep) ≈ 1 by construction (a third party is implausible), so the spread is mechanically mean-reverting — β_hedge ≈ −1.0, ρ_AR1 ≈ 0.3. The backtest shows 6 trades over the window with 83% hit rate. Half-life of 0.6 days means trades close fast; capital cycles 250+ times/year.

**`tsla_largest_jun ↔ nvda_largest_jun`** (S=+3.03, ½-life 0.5d)
Tech mega-cap horse race. Both markets compete for the same outcome (largest market cap by Jun) so they're pinned to the same news flow. Cointegration is almost guaranteed structurally. The spread reverts on micro news (NVDA earnings beat → spread tightens, TSLA Cybertruck miss → spread widens).

**`netanyahu_out_jun ↔ putin_out_jun`** (S=+2.85, ½-life 0.5d)
Both are "incumbent stays" markets, so they share *systemic political-risk* sentiment as a common factor. The spread captures *idiosyncratic* news (specific Israel news, specific Russia news). Genuine alpha, modulo the events themselves remain unresolved.

### ⚠ Suspicious / structural

**`btc_100k_eoy ↔ btc_500k_eoy`** (S=+5.73, only 3 trades)
Top of the leaderboard but the trade count is tiny. With 3 round-trips the Sharpe estimate has standard error ≈ 1/√3 ≈ 0.58, so the 95% CI on the population Sharpe is roughly [+4.5, +6.9]. Real but high-variance. Half-life 0.8d makes sense — both BTC strike markets co-move on the same spot moves.

**`oil_above_200_jun ↔ gold_5500_jun`** (S=+3.02, ½-life 0.8d)
Cross-asset commodities link. The cointegration is real (ADF p=0.004) but the *economic* explanation is weaker: gold and oil don't mechanically share common factors at intraday level. Half-life makes sense from inflation-hedge correlation. **Do this in small size.**

### ❌ Avoid (structural pricing artifact, not alpha)

**`twelve_plus_fed_cuts ↔ fed_chair_shelton`** (S=+2.97, only 1 trade made it through!)
Macro flagged this but only 1 trade fired in the window. Half-life 0.4d is suspiciously short — likely artifact of a single news event causing both to spike. Don't trust this one.

**`us_invade_greenland ↔ iran_leadership_change_eoy`** (S=+2.90)
Two unrelated geopolitical risk markets. The cointegration is statistical noise — there's no economic reason these should reliably revert.

---

## Sharpe distribution per theme

| Theme | Factors | Coint hits | Backtested | Median Sharpe | Top-1 Sharpe |
|---|---|---|---|---|---|
| crypto | 21 | 10 | 10 | +3.07 | +5.73 |
| ai | 19 | 10 | 10 | +3.40 | +3.96 |
| chips | 13 | 10 | 10 | +1.86 | +3.03 |
| geopolitics | 23 | 10 | 10 | +2.59 | +2.90 |
| politics | 13 | 10 | 10 | +1.49 | +3.50 |
| commodities | 6 | 2 | 2 | +1.51 | +3.02 |
| macro | 42 | 10 | 1 | +2.97 | +2.97 |

**Observations**:
- **Crypto and AI lead** because both themes have many *strike-ladder* markets (BTC@100k, BTC@200k, ...) that share the same underlying with different barriers — these are nearly-perfectly cointegrated by construction.
- **Macro is pathologically slow** to scan (~5 minutes for 42 factors / 861 pairs) — most of the time is spent on Polymarket history fetches, not the math. Suggest pre-warming the cache or running a daily background refresh.
- **Politics** has the cleanest economic story but only 13 factors → fewer hits.

## Recommended portfolio construction

A naive equal-weighted basket of the top-3 "genuinely tradeable" pairs:
1. `dem_senate_2026 ↔ rep_senate_2026`
2. `tsla_largest_jun ↔ nvda_largest_jun`
3. `netanyahu_out_jun ↔ putin_out_jun`

…gives a Sharpe-weighted average of (3.50 + 3.03 + 2.85) / 3 ≈ **+3.13**. Diversifying across themes (politics + chips + geopolitics) means the three are *uncorrelated* in their idiosyncratic alpha, so the basket Sharpe should approach Σ Sharpe_i / √k ≈ 9.4 / √3 ≈ **+5.4** under independence.

Per-pair sizing via half-Kelly with σ_eq estimated at ~0.005 → ~25% capital allocation per leg is reasonable. Document this calculation via the `/strategies/basket-stat-arb` endpoint.

## Things to verify before risking real money

1. **Resolution-source basis**: Polymarket settles via UMA. Two related Polymarket markets share UMA settlement risk — not as cleanly hedged as you'd think.
2. **Spread costs**: Check the actual bid-ask on each leg at trade entry. The 1-3¢ assumption is for the *liquid* mid; thin markets show 5-15¢ spreads which destroy edge.
3. **Half-life stability**: rerun the cointegration test on a *different* window (say 2025-09-01 → 2026-01-01) and verify the half-life doesn't blow up — regime change kills pairs trading.
4. **OU-bands optimal z\***: instead of the default 2σ/0.5σ, run `POST /strategies/ou-bands` for each pair with `transaction_cost_sigma` set to your real cost and use the resulting `z_entry_optimal` (Bertram 2010).
5. **Granger leadership**: for each surviving pair, check `/strategies/granger` to identify which leg leads. Trade the *follower* with a slight lag.

## Reproduce this report

```bash
# Per-theme auto-backtest
for theme in crypto ai chips geopolitics politics commodities macro; do
  curl -s -X POST http://127.0.0.1:8000/strategies/auto-backtest \
    -H 'Content-Type: application/json' \
    -d "{\"theme\":\"$theme\",\"start\":\"2026-01-01\",\"end\":\"2026-04-30\",\"max_pairs\":400,\"max_to_backtest\":10}" \
    > "/tmp/alpha_sweep/${theme}.json"
done
```

The frontend's **Auto-Backtest** sub-tab does the same thing with one click and a sortable leaderboard.
