# Death Certificate — Crypto-ETF approval drift

**Killed:** 2025-06-02 (Wave 4) · **Cause:** single-episode · **Claimed Sharpe:** 2.4 → **Post-mortem Sharpe:** 0.1

## Original thesis

Polymarket's spot-ETF-approval contracts were posited to lead the SEC's actual decision by 3-7 sessions during the late-2023 / early-2024 BTC ETF cycle. The strategy went long BTC (and later ETH) spot whenever the rolling 5-day change in PM approval probability exceeded +5pp, exiting on either resolution or a -5pp reversal. Front-running institutional inflows into the new ETF wrappers was the implied edge.

## Test results

Naive backtest (Sep-2023 → Apr-2024) showed a Sharpe of 2.4 with hit-rate 63% and a single-trade max winner of +18%. Wave-4 broke the sample into pre/post January-2024 windows. The pre-Jan-2024 window — covering the BTC spot ETF approval — delivered Sharpe 4.1. Every other window (May-2024 ETH approval, the 2025 ETF-extension dates, the bitcoin-futures-ETF-conversion events) had Sharpe below 0.3 with no statistically significant drift in PM odds preceding the SEC decision.

## Why it died

The full backtest PnL was a single event: the January-2024 BTC ETF approval. By construction, the PM market and the spot price moved together during that approval window, so any strategy that triggered on rising PM odds also bought rising spot. There was no leading information — the apparent edge was contemporaneous correlation. The ETH approval, in particular, came as a surprise to PM bettors and the strategy generated zero PnL there.

## Lesson

A "backtest" with one decision event is not a backtest. Refuse to deploy any strategy whose gross PnL is concentrated in a single event window, even if the in-sample Sharpe is gaudy. Demand ≥3 independent decision episodes before accepting an event-driven thesis.

## Resurrection

Only revisit if a future ETF-decision cycle produces a fresh, fully out-of-sample sample with ≥3 distinct decision events.
