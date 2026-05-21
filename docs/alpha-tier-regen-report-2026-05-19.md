# Alpha-tier regeneration report — 2026-05-19

Pipeline: cointegration -> walk-forward (k=4, embargo) -> permutation Sharpe -> BH-FDR -> 4Q stability -> alpha card verdict.

## Summary

- Pairs in input: **5**
- Pairs processed: **5**
- Pairs with errors: **3**
- Runtime: **0.095s**
- Timed out: **False**
- Output mode: `dry-run`

## Tier distribution — before vs after

| Tier | Before | After |
|------|--------|-------|
| B_VALIDATED | 0 | 2 |
| C_TENTATIVE | 5 | 0 |
| D_RAW | 0 | 3 |

## Pairs that gained A_GOLD

_None._

## Errored pairs (top 25)

| pair_id | error |
|---------|-------|
| `n0_a__n0_b` | not_cointegrated |
| `n2_a__n2_b` | not_cointegrated |
| `n1_a__n1_b` | not_cointegrated |
