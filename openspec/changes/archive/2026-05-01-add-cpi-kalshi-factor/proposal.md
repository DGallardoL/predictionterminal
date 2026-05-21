## Why

Our 47 curated factors are heavy on Fed-decision and recession markets but
have **zero direct inflation reads**. CPI surprises are one of the biggest
drivers of equity returns (rate-sensitive sectors, growth multiples, energy).
Kalshi has high-volume CPI markets with clean per-bar volume data — adding
one closes a major gap in the macro coverage.

## What Changes

- Add **`k_cpi_above_4_27`** factor (Kalshi `KXLCPIMAXYOY-27-P4` market — "max CPI YoY exceeds 4% in 2027") tracking
  the highest YoY CPI print expected during 2027.
- Document the thesis in `factors.yml` with explicit equity linkage.
- No backend code changes required — the Kalshi client introduced in the
  prior change already handles arbitrary KX series tickers.

## Capabilities

- **Modified Capabilities**:
  - `factors-catalog` — adding a new entry to the curated factor list.

## Impact

- **Code**: only `api/src/pfm/factors.yml` (config, not code).
- **API**: no schema changes; existing `/factors` returns the new entry.
- **UI**: green "K" badge automatically picked up via the source field.
- **Tests**: no new tests; the `test_factors_yaml.py` smoke test will
  validate parsing.
