# Strategy Lifecycle — Research to Live

> **Audience**: anyone (human or Claude) proposing a new alpha for the Prediction Terminal.
> **Companion docs**: `docs/adrs/ADR-0010-anti-alpha-rule.md` (W11-47), `docs/alpha-report-vN.md` series, `CLAUDE.md`.
> **Status**: canonical. Any deviation must be justified in a new ADR.

## 1. Overview — the 5-stage funnel

Every strategy that touches real capital must pass through five sequential gates. The funnel is deliberately punitive: roughly 1 in 20 ideas survives to full-live deployment, and that ratio is healthy. Stages cannot be skipped; a tier promotion needs evidence from the preceding stage on file.

```
  IDEA  →  BACKTEST  →  PAPER  →  SMALL-LIVE (≤5%)  →  FULL-LIVE (tier cap)
   |          |           |             |                       |
   100%      ~30%        ~10%          ~5%                    ~1-2%
```

Survival counts are illustrative (Wave-5 stress tests pruned 6 of 8 prior A_GOLD claims), but the shape is real: the cheapest place to kill a strategy is at the top.

## 2. Stage 1 — Idea

Ideas enter the funnel from three sources, and we log the source for every candidate.

- **Literature**: peer-reviewed quant journals (JPM, JoF, RFS), working papers (SSRN, arXiv-qfin), and prediction-market-specific research (Wolfers/Zitzewitz on accuracy, Manski on extracting probabilities). When a paper is cited, link the DOI in the proposal.
- **Observed market dislocations**: anomalies surfaced by Terminal mode (jumps, calendar λ-ratio outliers, cross-venue arb persistence, sentiment-vs-price divergence). The Terminal exists partly to feed this pipeline.
- **Intuition / structural priors**: e.g. resolution-decay on binary contracts, election-binary momentum, Fed-decision overreaction. Intuition-sourced ideas face a *higher* burden in Stage 2 because there is no external prior.

Output of Stage 1: a one-page hypothesis stub in `docs/proposals/<slug>.md` with the predicted sign, expected horizon, expected Sharpe band, and a falsifiable failure condition.

## 3. Stage 2 — Backtest

Backtest is performed in four ordered sub-steps. Skipping the first two is the single most common reason an alpha later blows up.

1. **Synthetic DGP first**. Generate data with known betas / signal-to-noise / regime structure and confirm the estimator recovers them. This catches sign errors, scaling bugs, and lookahead bias before any real data touches the pipeline. Every strategy module ships a `test_<name>_synthetic_dgp.py`.
2. **Historical data backtest**. Run on the full available history with realistic frictions: bid-ask, fees per venue, slippage proportional to depth. No assuming the close print is fillable.
3. **In-sample / OOS split**. 70/30 chronological split. Tune parameters in-sample only; report OOS numbers as the headline. If OOS Sharpe < 0.5 × IS Sharpe, the parameter set is overfit; refit with stronger regularization or kill the idea.
4. **Walk-forward (W12-29)** for regime sensitivity. Rolling 12-month train / 3-month test, stepped monthly. The walk-forward Sharpe is the number we trust; a single OOS slice can still be lucky.

Output of Stage 2: a backtest notebook + a row in `docs/alpha-report-vN.md` under "Candidate" tier with all four numbers (DGP recovery error, full-history Sharpe, OOS Sharpe, walk-forward median Sharpe).

## 4. Stage 3 — Paper trading

Paper trading is where most "obviously good" ideas die. Three gates run in parallel.

- **4-quarter stress test** (the `CLAUDE.md` anti-alpha rule). Disjoint quarters across at least one regime change. If any quarter has Sharpe < 0.5 *or* the sign flips vs. full-sample, the strategy is moved to the anti-alpha graveyard and may not be re-pitched without new structural justification (see ADR-0013).
- **Deflated Sharpe gate (W11-53)**. Bailey-López de Prado deflation against the number of trials we ran. If the deflated Sharpe is not statistically positive at 5% (BH-FDR adjusted across the candidate batch), the strategy stays at `B_FDR_ONLY` and is not eligible for live capital.
- **Tier classification**:
  - `B_FDR_ONLY` — survives FDR but fails one robustness leg. Paper only.
  - `B_VALIDATED` — survives FDR + 4-quarter stress; no structural story strong enough to promise persistence. Eligible for Stage 4 at 5%.
  - `A_STRUCTURAL` — survives both above and has an articulated micro-/macrostructural reason (e.g. calendar λ-ratio bias from real settlement asymmetry). Eligible for Stage 4 at 5%; can be promoted to `A_GOLD` only after Stage 5 confirmation.

Output of Stage 3: tier assignment recorded in `web/data/alpha_strategies.json`.

## 5. Stage 4 — Small live (5% allocation cap)

Real capital, real fills, but capped at 5% of book NAV per strategy regardless of tier. This is the only place we see true slippage, queue position, and adverse selection.

- **Live signal pipeline** writes to `web/data/live_signals.json` daily; the α Hub "Live Edge" tab consumes it.
- **Daily verdict review**: a one-line note in `docs/live-journal-<yyyy-mm>.md`: signal fired? filled? PnL vs. expected? Anomalies?
- **Monthly Sharpe re-check**: rolling 60-trading-day Sharpe vs. the paper-trading Sharpe. A persistent gap > 1.0 (paper >> live) signals modeled-vs-realized friction mismatch; halt and recalibrate.
- Minimum 6 weeks at this stage before promotion is even considered.

## 6. Stage 5 — Full live (up to tier cap)

Promotion to full live raises the allocation cap to the tier limit (currently: `A_STRUCTURAL` 10%, `A_GOLD` 20%, `B_VALIDATED` 5%).

- **`A_GOLD` requires 4+ quarters of live confirmation** at Stage 4, with no quarter Sharpe < 0.5 and no sign flip. This is non-negotiable; the 2026 Wave-5 audit reclassified six prior A_GOLD claims because they had not cleared this bar.
- **Quarterly review** appears in the next `docs/alpha-report-vN.md`. The review must address: live-vs-paper Sharpe gap, capacity utilization, slippage realization, any regime change observed.

## 7. Demotion paths

Demotion is faster than promotion by design. Triggers, evaluated quarterly:

- Any quarter Sharpe < 0.5 → demote 1 tier (`A_GOLD` → `A_STRUCTURAL` → `B_VALIDATED` → `B_FDR_ONLY`).
- Sign flip vs. full-history → demote 1 tier and freeze new entries for the quarter.
- Two consecutive demotion triggers → move to anti-alpha graveyard.
- Mechanism collapse (the structural story stops holding, e.g. the venue closes a calendar window) → straight to graveyard regardless of Sharpe.

Demotions are not punishments; they are accounting. The favorites-bias case (`B_VALIDATED`, paper-only as of Wave-5) is the canonical example.

## 8. Audit trail

Every strategy carries a persistent audit log queryable at `GET /strategies/{pair_id}/audit-trail` (W12-14). The endpoint returns the full chronological tier history, the test runs that triggered each transition, and any human/Claude overrides with justification. No tier change is valid without a corresponding audit-trail entry.

## 9. Capacity awareness

When scaling notional, we monitor three numbers continuously:

- **Slippage realization**: live fills vs. modeled fills. If realized > 1.5 × modeled for two weeks, cut size in half.
- **Market impact**: post-trade reversion within 30 minutes. If consistent reversion > 30% of edge, we are the marginal mover and must shrink.
- **Depth utilization**: notional traded / top-3-level depth. Cap at 25%; above that the queue-position model breaks.

Capacity is per-venue and per-contract; aggregating across markets hides the constraint.

## 10. Anti-alpha graveyard

The "Demoted" section of every `docs/alpha-report-vN.md` is the graveyard. Entries there must NOT be re-pitched as wins by any future Claude or human contributor without:

1. A new structural mechanism (not the old one resurrected),
2. A fresh Stage 2 backtest including the post-demotion period, and
3. An explicit reference in the proposal acknowledging the prior demotion.

The graveyard is the institutional memory that prevents the same mistake from being re-litigated every six months. Treat it as load-bearing.

---

*See ADR-0010 (`docs/adrs/ADR-0010-anti-alpha-rule.md`, W11-47) for the formal decision record that codifies the demotion rule referenced in §7.*
