# ADR-0013: The Anti-Alpha Rule — 4-Quarter Robustness + Deflated Sharpe Gate

- **Status:** Accepted
- **Date:** 2026-05-16
- **Authors:** Damian Gallardo
- **Supersedes / amends:** None (extends the deployability claims first sketched in `docs/alpha-reports/alpha-report-v15.md` and hardened through v17).

## Context

This project's most expensive failure mode is **publishing a strategy as
"deployable" that turns out to be a one-quarter regime trade**. The cost
is not only intellectual — every reader of the alpha-hub tier badges,
every downstream backtest that cites a Sharpe number, and every demo
narrative inherits the false claim. Once it is on the public surface it
is very hard to retract without looking dishonest.

The trigger for this ADR is the **Wave-5 stress-test exercise**
(2026-04 → 2026-05) documented in the auto-memory note
`project_wave5_stress_test_findings.md` and the
`docs/alpha-reports/alpha-report-v15.md` → `…-v17.md` series. Wave-5
applied a uniform robustness harness to the eight A_GOLD strategies
inherited from earlier alpha reports. The result was brutal:

- **Six of eight A_GOLD claims failed** the 4-quarter Sharpe-stability
  test. Five flipped sign in at least one disjoint quarter; one
  collapsed to Sharpe < 0.2 OOS after deflation.
- The only structural survivor was the **calendar λ-ratio** strategy
  (Fed-decision straddle proxy via VIX overlay on Polymarket FOMC odds),
  which kept Sharpe ≥ 0.5 across all four quarters and survived
  deflation against the multiple-testing budget.
- The favorites-bias alpha that was previously A_GOLD was downgraded
  to **B_VALIDATED, paper-only**, with allocation halved to 5% until
  2026 Q3 confirms or kills it (see `project_favorites_bias_alpha.md`).

The recurring pattern across the killed claims was the same: a *single*
quarterly window with an unusually clean regime (Q4-2024 election cycle,
Q1-2025 crypto-ETF approval, the 2024 Senate-control vote, the
geopolitical-conflict oil window) produced a backtest Sharpe of 1.5–2.5
on three to six months of in-sample data, with no out-of-sample period
because the event itself was unique. The reported Sharpe was real; the
**generalisation** was fictional. Each of these is now on the anti-alpha
list in `CLAUDE.md`.

The CLAUDE.md "What not to do" section captures the rule in one line:
> Don't deploy regime-driven alphas without a 4-quarter robustness check.

This ADR turns that one-liner into a binding gate.

## Decision

A strategy may be claimed **deployable** (tier ≥ B_VALIDATED) only if it
passes a three-part robustness gate, evaluated by `api/scripts/stress_test.py`
and enforced in CI:

**(a) 4-quarter Sharpe-stability.** Split the full sample into four
disjoint quarterly windows of approximately equal length. Compute the
Sharpe ratio of the strategy's daily PnL in each window. Require
`Sharpe_q ≥ 0.5` in every window. A single window below 0.5 fails the
gate, even if the full-sample Sharpe is high.

**(b) No sign-flip vs full-sample.** The sign of each quarter's mean
return must match the sign of the full-sample mean return. A strategy
whose direction flips across regimes is by definition not a structural
edge, regardless of magnitude.

**(c) Deflated Sharpe ≥ 0** (one-sided, α = 0.05). Apply the
deflated-Sharpe correction in
`pfm/quant/deflated_sharpe.py` using:
- the number of independent strategies considered during the wave's
  search (`N_trials`),
- the empirical skewness and kurtosis of daily returns,
- the sample length in trading days.

This corrects the headline Sharpe for the multiple-testing inflation
inherent in scanning thousands of factor combinations and reports a
*deflated* Sharpe and its p-value. A strategy whose deflated Sharpe is
indistinguishable from zero at α = 0.05 fails the gate, even if it
passed (a) and (b).

### Tier ceiling

Even when a strategy passes all three gates on **historical** data, it
is capped at **B_VALIDATED**. Promotion to **A_GOLD** requires **four
quarters of LIVE confirmation** (paper or real) of the gated metrics,
i.e. roughly one year of forward evidence. This is enforced by the tier
field in `web/data/alpha_strategies.json` and surfaced as a rainbow
tier-pill badge on every α Hub card.

## Implementation

- **`api/scripts/stress_test.py`** — the T-stress harness. Loads a
  strategy from `pfm/strategies/registry.py`, fetches its PnL series,
  partitions into four quarters, computes per-quarter Sharpe and sign,
  and calls `deflated_sharpe()` for the multiple-testing correction.
  Exits non-zero on any gate failure so CI fails the build.
- **`pfm/quant/deflated_sharpe.py`** — deflated-Sharpe-ratio
  implementation. Takes the raw Sharpe, sample length, skewness,
  kurtosis, and `N_trials`; returns the deflated Sharpe plus the
  p-value under the null of zero true Sharpe.
- **CI gate.** The `.github/workflows/ci.yml` `stress-test` job runs
  `python api/scripts/stress_test.py --all` against every strategy
  marked `tier: B_VALIDATED` or higher in `alpha_strategies.json`. A
  red gate blocks merge.
- **`docs/alpha-reports/alpha-report-vN.md` versioning.** Each wave's
  robustness pass produces a new alpha-report. **Older reports are
  never edited.** The current report is v17; the next wave bumps to
  v18. This preserves the audit trail of *what was believed
  deployable at version N*, which is essential when a future quarter
  invalidates a claim.
- **Anti-alpha list.** Strategies that fail the gate are not silently
  removed. They are written into the "Anti-alphas (DO NOT redeploy)"
  section of `CLAUDE.md` with a one-line cause-of-death so the next
  Claude does not re-pitch them. Current entries:
  - **recession-defensive (Q4-2024)** — sign flipped Q1-2025; pure
    regime trade.
  - **crypto-ETF approval drift** — single one-time event; backtest
    was a survivorship illusion.
  - **senate-control short-vol** — dominated by a single 2024
    episode; OOS Sharpe < 0.2 after deflation.
  - **geopolitical-conflict oil long** — direction-correct but
    transaction costs eat ≥ 110% of gross PnL.

## Consequences

- **Slower deploy cadence.** Most strategies that "look great" in one
  quarter sit at C_RESEARCH or D_DEAD until four disjoint quarters
  exist. Wave-5 itself took ≈ 6 weeks to gate eight candidates. This
  is the intended cost.
- **Trustworthy claims.** A B_VALIDATED tier on an α Hub card now means
  something concrete: four quarterly windows of Sharpe ≥ 0.5, no sign
  flip, deflated p < 0.05. A_GOLD means that **plus** a year of live
  forward confirmation. Users can read tier badges as load-bearing
  information rather than marketing.
- **Honest scoreboard.** The anti-alpha list is part of the
  product. Showing what we tried and killed is a stronger signal of
  rigor than showing only winners, and it prevents the team from
  re-running dead ideas under new names.
- **Multiple-testing discipline.** Every wave declares its `N_trials`
  search budget up front. A wave that scanned 2,000 factor
  combinations has a much higher deflation hurdle than a wave that
  pre-registered one hypothesis. This pushes the project toward
  fewer, more theory-grounded candidates per wave.
- **Calendar λ-ratio is the only A_GOLD on the board.** That is the
  correct state of the world after Wave-5. Everything else is
  B_VALIDATED-or-below until the live record exists. Future waves
  will widen the A_GOLD list at the pace that real time provides.
