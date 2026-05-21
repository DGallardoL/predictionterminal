# Death Certificate — BTC latency arbitrage (PM ↔ spot)

**Killed:** 2026-05-02 (Wave 7) · **Cause:** non-portable · **Claimed Sharpe:** 3.6 → **Post-mortem Sharpe:** 0.2

## Original thesis

An earlier study claimed Polymarket's BTC-price binary contracts lagged the BTC spot midpoint by 200-800ms during high-flow periods. The proposed strategy leaned against the PM mid whenever Coinbase spot deviated by ≥1bp over a 60-second window, expecting convergence within seconds and clipping the spread. Annualized Sharpe of 3.6 with hit-rate 71% looked structural — a microstructure edge tied to PM's slower price-discovery.

## Test results

An 8-agent investigation on 2026-05-02 re-ran the full study against 11 weeks of synchronized PM and Coinbase tick data, with timestamps re-aligned to NTP-disciplined references on both sides. The previously reported lag of 200-800ms collapsed to a median of 12ms with no exploitable persistence: autocorrelation of the lag at 1s+ horizons was indistinguishable from zero. PM-side fill probability for the implied trade size was below 40% even at the touch, and gross Sharpe net of the 2% PM fee fell to 0.2.

## Why it died

The original Sharpe was a clock-alignment artifact. The two data feeds had been logged on different machines with drifting NTP, producing an apparent 400ms PM lag that was actually clock skew. Once timestamps were aligned, no exploitable midpoint lag remained. Vanilla midpoint latency arb between PM and Coinbase is not a real edge.

## Lesson

Microstructure studies live or die on clock discipline. Always verify NTP synchronization and assume zero edge until clock drift is bounded below 10ms. A 400ms "discovery" delay across venues is, in 2026, almost always an instrumentation bug.

## Resurrection

Do not re-explore unless a new angle is added — rolling-σ-conditioned latency, orderbook-imbalance lead, or a different venue pair. Vanilla midpoint lag is dead.
