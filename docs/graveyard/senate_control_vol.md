# Death Certificate — Senate-control short-vol

**Killed:** 2025-08-20 (Wave 5) · **Cause:** single-episode · **Claimed Sharpe:** 1.5 → **Post-mortem Sharpe:** 0.18

## Original thesis

Once Polymarket Senate-control odds become decisively one-sided (>80% on either side), the political-uncertainty risk premium embedded in SPX implied vol should compress. The strategy shorted front-month VIX futures whenever PM Senate-control concentration crossed 0.80 in either direction, holding to the resolution date or the next FOMC, whichever came first.

## Test results

In-sample (Jul-2024 → Jan-2025) the strategy posted Sharpe 1.5 with a clean equity curve over the November 2024 election. Wave-5 stress tests broke the sample into four disjoint quarters and ran the same rules on the 2025 mid-cycle Senate odds. Q4-2024 (the actual election) drove ~95% of the gross PnL. The 2025 quarters showed essentially no realized-vol response to PM Senate odds, with OOS Sharpe collapsing to 0.18 across the three out-of-sample quarters.

## Why it died

A single election cycle is not a sample. The 2024 cycle had a unique combination of features — late-breaking Senate races, a contested presidential race driving cross-correlated vol bid, and a particular Fed-cut path — that made vol compress as Senate-control concentrated. Mid-cycle 2025 odds tightened on routine political news without any associated VIX move. The "edge" was a one-off macro coincidence, not a reusable signal.

## Lesson

Vol-compression strategies that ride one election need at least three distinct election cycles to be credible. Wave-5 stress tests killed 6 of 8 A_GOLD claims for exactly this reason — single-episode dominance that disappears the moment the regime changes.

## Resurrection

Need at least three distinct election cycles with comparable Senate-control concentration AND a vol-regime control; one cycle is not a sample.
