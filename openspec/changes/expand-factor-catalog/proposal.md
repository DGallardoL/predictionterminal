## Why

The catalog has 48 factors but is biased toward AI / crypto / Iran headlines
and missing several economically-relevant Polymarket markets that the
discovery probe surfaced as high-volume on 2026-04-30:

- No direct US oil price tail markets (crude-200/150/115/70 — millions in volume)
- Single Tesla / Apple / Amazon catalysts despite their weight in the SPX
- No Ethereum / Solana history (only Bitcoin)
- Fed Chair confirmation slate is a cleaner political-economy proxy than the
  current `powell_out_may` alone
- 2026 midterm control markets (House/Senate) are a known equity-vol driver
- Climate/pandemic tail markets exist with $300k–$6M volume and are absent

The cherry-pick experiments showed that thesis-driven baskets (3–4 factors
selected by an analyst) beat 30-factor stepwise. More raw material per theme
is what makes good baskets possible. We are explicitly NOT changing logic —
only widening the input universe.

## What Changes

- Add 94 verified-live Polymarket factors to `api/src/pfm/factors.yml`
  (51 → 145), in two rounds:
  - **Round 1** (52 factors): macro/Fed-Chair (6) + macro/FOMC-detail (6) +
    crypto-ETH/SOL (8) + commodities (4 new theme) + tech mega-cap (7) +
    AI race (5) + M&A (2) + Iran-detail (2) + midterms-control (5) +
    climate (3 new theme) + health (3) + Trump (1).
  - **Round 2** (42 factors): full Fed-cut cardinal distribution (1-10
    cuts) + Fed-funds-rate level (4.0/4.5%) + any-hike (1) + 3 more
    Fed-Chair candidates + 6 crypto BTC dip/reach tails + 4 balance-of-
    power combos + Saudi Aramco mcap + crude $175 + GOLD $5500
    (first non-oil commodity) + 12 escalation/de-escalation
    geopolitics (Greenland, Cuba, NATO, Iran-deal, Netanyahu out,
    Putin near-term, Iran leadership, Trump-Putin meet).
- Add two new themes: `commodities` and `climate` (the YAML loader already
  accepts arbitrary theme strings — the front-end groups by `theme` for
  the composite reducer, so new themes just appear as new groups).
- Pre-flight verify each slug via the Polymarket Gamma API and only keep
  ones that are still `active=true, closed=false` and have a tradeable
  YES token.

No code changes. No schema changes. No tests required beyond the existing
`test_factors_loaded` which iterates `factors.yml` and checks shape.

## Impact

- Affected specs: `factors-catalog` — adds new requirements describing the
  expanded coverage (new themes, list of new IDs).
- Affected code: `api/src/pfm/factors.yml` (single-file edit, ~600 new lines).
- No breaking change. Existing factor IDs are untouched.
- The `/factors/best` stepwise will see more candidates — users may want to
  combine with the new permutation gate to filter spurious additions.
