# Death Certificate — Geopolitical-conflict → Oil long

**Killed:** 2025-10-10 (Wave 6) · **Cause:** TC · **Claimed Sharpe:** 1.1 → **Post-mortem Sharpe:** -0.05

## Original thesis

When Polymarket conflict-escalation odds (Middle East, Russia-Ukraine, Taiwan Strait families) rise sharply over a 48-hour window, front-month WTI futures should reprice with a measurable lag as physical-side participants update slower than headline-driven PM bettors. The strategy went long WTI whenever any escalation contract jumped ≥5pp in 48h, holding for 5 sessions or until the contract resolved.

## Test results

Across the full Wave-6 sample (Jan-2024 → Sep-2025) the strategy traded 11 distinct escalation episodes with a frictionless gross Sharpe of 1.1 and a positive PnL in 4 of 4 quarters — the direction was correct. Once realistic transaction costs were modeled (5bp slippage on entry, 5bp on exit, plus widened bid-ask during news windows), net Sharpe fell to -0.05. Sensitivity analysis showed that even at $5M notional the modeled TC eats ≥110% of gross PnL.

## Why it died

The signal exists, but execution is the wrong instrument. WTI futures bid-ask widens dramatically during exactly the news windows that trigger the strategy, and shallow size at the touch means realistic fills are several ticks away from the mid. The PM signal correctly anticipated oil moves; the bottleneck is friction, not alpha.

## Lesson

A direction-correct signal is not a deployable strategy. Always model TC under stressed conditions matching the signal-firing regime, not annualized averages. Strategies that trigger during high-vol windows must be benchmarked against high-vol TC, which is typically 3-5x normal.

## Resurrection

Only resurrect with execution via a brent-WTI calendar spread or an oil-equity proxy (XOP) that has lower TC; never with outright WTI futures.
