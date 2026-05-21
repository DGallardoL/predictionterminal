## Context

Existing macro factors cover Fed decisions (5 Kalshi + 5 Polymarket variants),
recession (1 Kalshi + 1 Polymarket), and tariff policy. Inflation prints —
which the Fed's policy decisions are downstream of — are not directly
represented. The Kalshi `KXLCPIMAXYOY-27` series tracks the maximum CPI YoY
during 2027 with daily volume per bar.

## Goals / Non-Goals

**Goals**
- Add one canonical CPI factor with verified history depth.
- Document its thesis in `factors.yml` so users see economic linkage in the UI
  description.
- Validate that it loads, sparkline renders, regression runs end-to-end.

**Non-Goals**
- Adding multiple inflation strikes (5%, 6%, 7% etc.). Start with one.
- Refactoring the Kalshi client.
- Adding back-tested validation — that's a separate audit change.

## Decisions

- **Use `KXLCPIMAXYOY-27` (year 2027, max YoY)** rather than monthly print
  markets — has 200+ bars of history vs the monthly markets which are
  shorter-dated.
  *Alternative considered*: `KXLCPIMAXYOY-26` — but it resolves end of 2026
  so the trading window is shorter. 2027 gives more runway.

- **Theme `macro`** — same theme bucket as Fed factors. Inflation belongs
  with the rates regime cluster economically.

- **Skip resolving filter override** — the `is_resolving_factor` heuristic
  already handles the 14-day-tail check; this market is far from resolution
  so won't be filtered.

## Risks / Trade-offs

- **Risk**: One CPI factor is a single read on a multi-dimensional thing
  (headline vs core, monthly vs annual). → Mitigation: documentation makes
  the limitation explicit; future change can add core/headline split.

- **Risk**: Kalshi markets sometimes have low volume on early bars, which
  amplifies Δlogit noise. → Mitigation: `is_resolving_factor` + zscore
  pipeline already in place. Verify n_bars ≥ 100 before merging.

## Migration Plan

1. Verify the slug returns history (`fetch_factor_history`).
2. Add yaml entry.
3. Restart server; confirm `/factors` lists the new entry with source=kalshi.
4. Click through UI: sparkline renders, regression with single factor works.
5. Archive change after smoke-test.

## Open Questions

- Should we also add `KXLCPIMAXYOY-26` for the tighter window? Defer until
  this one is validated.
