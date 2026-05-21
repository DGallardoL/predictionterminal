# Death Certificate — Favorites-bias (heavy-favorite under-pricing)

**Killed:** 2026-04-28 (Wave 5) · **Cause:** regime · **Claimed Sharpe:** 1.4 → **Post-mortem Sharpe:** 0.4

> **Note:** This entry records a *downgrade*, not a hard archive. The favorites-bias strategy is currently paper-trading at B_VALIDATED with 5% allocation, pending 2026Q3 confirmation. It is listed here so the public record reflects that the original A_GOLD claim did not hold up.

## Original thesis

Heavy favorites in PM binary contracts (probability ≥85%) were claimed to systematically resolve at a higher rate than implied — a longshot-bias mirror image. The strategy went long the favorite at entry and held to resolution, equivalent to selling longshot premium at scale across thousands of binary contracts. Gold-tier deployment with 10% allocation was proposed in early 2026.

## Test results

Wave-5 stress tests broke the original sample into four disjoint quarters. Q4-2024 delivered Sharpe 2.1 and drove most of the headline 1.4 figure. Q1-2025 collapsed to Sharpe 0.3 as PM market-makers tightened heavy-favorite spreads in response to mounting professional flow. Q2-2025 fell further to 0.1. The cross-quarter Sharpe stability score — a Wave-5 promotion gate — failed by a wide margin, and the BH-FDR-corrected p-value on the 2025 sub-sample was 0.31.

## Why it died (in its A_GOLD form)

The effect was real but transient. As recently as Q4-2024, heavy-favorite contracts were systematically mispriced because PM liquidity providers had not yet calibrated to professional flow. By Q1-2025, market-makers had re-priced the favorite tail and the edge largely disappeared. It was a function of PM market-microstructure immaturity, not a structural mispricing.

## Lesson

Mispricings driven by venue immaturity are dated assets. Tier them as B_VALIDATED at most, allocate small, and re-test every quarter. Wave-5 stress tests killed 6 of 8 A_GOLD claims for exactly this kind of cross-quarter Sharpe collapse.

## Resurrection

Reinstate to A_GOLD only after 2026Q3 paper-trading shows live OOS Sharpe ≥ 0.8 across ≥3 disjoint contract-categories.
