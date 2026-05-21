# factors-catalog Specification

## Purpose
TBD - created by archiving change add-cpi-kalshi-factor. Update Purpose after archive.
## Requirements
### Requirement: CPI inflation factor available in catalog
The factor catalog (`factors.yml`) SHALL include a Kalshi-sourced market
tracking the maximum YoY US CPI print expected during 2027.

#### Scenario: Factor appears in /factors response
- **WHEN** a client calls `GET /factors`
- **THEN** the response SHALL include an entry with `id="k_cpi_above_4_27"`,
  `source="kalshi"`, and `theme="macro"`

#### Scenario: Factor history pulls successfully
- **WHEN** the server requests history for slug `KXLCPIMAXYOY-27-P4`
- **THEN** the Kalshi candlesticks endpoint SHALL return at least 100 daily
  bars within the 2026-01-01 to 2026-04-30 window
- **AND** each bar SHALL contain `price` (close in [0,1]), `volume`, and
  `open_interest` fields

#### Scenario: Factor description names equity linkage
- **WHEN** a user views the factor in the UI
- **THEN** the description text SHALL state which sectors the factor links
  to economically (e.g., rate-sensitive growth multiples, energy)

