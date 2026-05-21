# Death Certificate — Recession-odds → Defensive-sector long

**Killed:** 2025-04-15 (Wave 3) · **Cause:** regime · **Claimed Sharpe:** 1.8 → **Post-mortem Sharpe:** -0.3

## Original thesis

When Polymarket's aggregate recession-probability climbs over 3 daily sessions, institutional flows rotate from cyclicals into defensives (XLU, XLP, XLV) ahead of the price reaction. The strategy went long XLU and short SPY at signal, sized to dollar-neutral, with a 5-day holding window. The signal was attractive precisely because PM odds appeared to lead Bloomberg consensus by 1-2 sessions during late 2024.

## Test results

In-sample (Aug-2024 → Mar-2025) the strategy posted a Sharpe of 1.8, hit-rate of 58%, and a max drawdown of -2.1%. Wave-3 robustness ran the same rules across four disjoint quarters. Q4-2024 alone delivered a Sharpe of 3.2 and accounted for 89% of cumulative PnL. Q1-2025 flipped sign — Sharpe -1.4 — as recession-odds kept rising while defensives underperformed in a broad melt-up. Q2-2025 and Q3-2025 were near-flat.

## Why it died

The signal had no structural carry. It was a coincident regime trade: when risk-off and recession-odds rose together, the trade made money; when they decoupled (Q1-2025: recession-odds up, equities up), the trade lost. The PM signal added zero alpha after controlling for SPX trend; a vanilla risk-off filter would have produced the same PnL.

## Lesson

Single-quarter dominance is a red flag. Always demand sign stability across ≥4 disjoint quarters before promoting from B_VALIDATED. Macro-regime correlation must be regressed out before claiming PM-derived alpha.

## Resurrection

Need ≥4 quarters of stable risk-off regime AND a residualized signal that controls for SPX trend; otherwise treat as a macro overlay only.
