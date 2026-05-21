## ADDED Requirements

### Requirement: New themes — commodities and climate
The factor catalog SHALL support two new theme groups, `commodities` and
`climate`, surfaced through the existing `theme` field of `FactorMetadata`
and the front-end Curated tab. No code changes are required — the loader
and UI accept arbitrary theme strings.

#### Scenario: Commodities theme has at least four oil-strike factors
- **WHEN** a client calls `GET /factors`
- **THEN** the response SHALL include factors with theme `commodities`
  spanning a high-strike oil tail (≥$150), a near-the-money strike
  (~$115), an extreme strike (≥$200), and a downside strike (≤$70).

#### Scenario: Climate theme covers hurricane and earthquake tails
- **WHEN** a client calls `GET /factors`
- **THEN** the response SHALL include factors with theme `climate`
  covering Cat-4-hurricane US-landfall probability, magnitude-7+
  earthquake count, and pre-season named-storm formation.

### Requirement: Macro Fed-Chair confirmation slate
The factor catalog SHALL include a categorical Fed Chair confirmation
race covering at least the six top-volume named candidates as separate
binary factors.

#### Scenario: Six chair candidates are independently tradable factors
- **WHEN** a client calls `GET /factors` and filters by `theme=macro`
- **THEN** the response SHALL include `fed_chair_warsh`,
  `fed_chair_shelton`, `fed_chair_bessent`, `fed_chair_bowman`,
  `fed_chair_powell_keep`, and `fed_chair_waller` as distinct factor
  IDs, each pointing to a verified-active Polymarket market.

### Requirement: Crypto coverage beyond Bitcoin
The factor catalog SHALL include Ethereum-specific tail markets and at
least one Solana market, complementing the existing BTC-only crypto
coverage.

#### Scenario: ETH and SOL factors are present
- **WHEN** a client calls `GET /factors`
- **THEN** the response SHALL include at least three Ethereum factors
  (`eth_ath_eoy`, `eth_5k_eoy`, `eth_dip_1500`) and at least one
  Solana factor (`sol_ath_jun`).

### Requirement: Full Fed-cut cardinal distribution
The catalog SHALL provide the full cardinal distribution of 2026 Fed-cut
counts, allowing analysts to regress against the empirical mass at each
specific count rather than only on tails.

#### Scenario: Counts 0 through 12+ are present
- **WHEN** a client calls `GET /factors` and filters by `theme=macro`
- **THEN** the response SHALL include `no_fed_cuts_2026`,
  `fed_cuts_1_2026`, `fed_cuts_2_2026`, `fed_cuts_3_2026`,
  `fed_cuts_4_2026`, `five_fed_cuts`, `fed_cuts_6_2026`,
  `fed_cuts_7_2026`, `fed_cuts_8_2026`, `fed_cuts_9_2026`,
  `fed_cuts_10_2026`, `eleven_fed_cuts`, and `twelve_plus_fed_cuts`.

### Requirement: Direct Fed-funds-rate level markets
The catalog SHALL include at least two end-of-2026 Fed-funds-rate
upper-bound level markets, providing a level-based regressor that ties
to a single number rather than to cumulative-cut counts.

#### Scenario: 4.0% and 4.5% level markets are present
- **WHEN** a client calls `GET /factors` and filters by `theme=macro`
- **THEN** the response SHALL include `fed_target_45_eoy` and
  `fed_target_40_eoy` as live verified factors.

### Requirement: Crypto BTC dip/reach surface
The catalog SHALL provide a multi-strike BTC dip-and-reach surface
covering 15K / 35K / 45K / 55K dips and 100K / 200K / 250K / 500K / 1M
upside strikes for end-2026, enabling regression on the implied
distribution rather than a single point.

#### Scenario: Multi-strike BTC surface
- **WHEN** a client calls `GET /factors` and filters by `theme=crypto`
- **THEN** the response SHALL include all of: `btc_dip_15k`,
  `btc_dip_35k`, `btc_dip_45k`, `btc_dip_55k`, `btc_100k_eoy`,
  `btc_200k_eoy`, `btc_250k_eoy`, `btc_500k_eoy`, `btc_1m_eoy`.

### Requirement: Gold strike market
The catalog SHALL include at least one direct gold-price-level
Polymarket market, providing the first non-oil commodity factor.

#### Scenario: Gold $5500 strike is present
- **WHEN** a client calls `GET /factors` and filters by `theme=commodities`
- **THEN** the response SHALL include `gold_5500_jun` as a live
  verified factor.

### Requirement: Health, politics, and geopolitics expansion
The catalog SHALL include 2026 midterm chamber-control markets, US
public-health tail markets, and additional Iran-detail and Russia-NATO
escalation markets to broaden the curated theme baskets.

#### Scenario: Midterm and pandemic factors present
- **WHEN** a client calls `GET /factors`
- **THEN** the response SHALL include `dem_house_2026`, `rep_house_2026`,
  `dem_senate_2026`, `rep_senate_2026`, `measles_10k_us`,
  `new_pandemic_2026`, and `russia_invade_nato_jun` as live factors.

### Requirement: Verification gate for added factors
Every factor newly added in this change SHALL have been verified live
against the Polymarket Gamma API on the catalog-update date with
`active=true`, `closed=false`, and at least 20 daily price bars in the
preceding 180-day window.

#### Scenario: All new factors pass the verification gate
- **GIVEN** the catalog is updated as part of `expand-factor-catalog`
- **WHEN** the verification probe runs
  (`fetch_factor_history` with start = today−180d)
- **THEN** every newly-added factor SHALL return ≥20 bars and the
  current price SHALL be in (0, 1).
