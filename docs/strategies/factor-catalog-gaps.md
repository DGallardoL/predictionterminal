# Factor Catalog Gaps — Audit 2026-05-15

## Intro

The curated catalog (`api/src/pfm/factors.yml`, 1228 factors as of 2026-05-13) is heavily skewed toward
politics (365), sports (128), geopolitics (117) and macro (89), with thin coverage of the
sector-specific factors most likely to drive single-stock and ETF returns. Several recent
regression audits failed not because the math was wrong but because the *relevant* contract
was never curated. This document inventories the gaps by sector, proposes specific
Polymarket slugs that either exist today or could be added when liquidity arrives, and
ranks the highest-impact additions.

Verification methodology: each candidate was probed via
`GET /factors/discover?min_volume=…&keyword=…` against the live Polymarket gamma cache.
That endpoint walks the top ~500 markets by 24h-volume; markets in the long tail are not
returned, so a "no" verification result means "not in the high-volume tier today" rather
than "does not exist on Polymarket." Where I have residual doubt the entry is marked
**maybe** or **speculative**.

Existing single-keyword counts in `factors.yml` (sanity check before proposing additions):

| keyword | matches in catalog |
|---|---|
| `tariff` | 0 |
| `shutdown` | 0 |
| `supreme court` / `scotus` | 0 |
| `Trump approval` | 1 (one `hit-35-in-2026` strike) |
| `boeing` / `lockheed` / `raytheon` / `northrop` | 0 |
| `wegovy` / `ozempic` / `mounjaro` / `keytruda` | 0 |
| `dkng` / `sportsbook` / `state legalization` | 0 |
| `coinbase` / `marathon` / `riot platforms` | 0 |
| `cybertruck` / `fsd` / `robotaxi` (TSLA) | 1 (Tesla CA-robotaxi only) |
| `etf flows` / `gbtc` / `fbtc` / `ibit` | 0 |

---

## 1. TSLA / Tesla

The catalog has the largest-by-mcap meta-bet, the optimus/robovan/cybercab launch markets,
and the SpaceX/xAI merger probability. What is **missing** is the operational/sales-cadence
factor set that actually moves TSLA week-to-week: deliveries, FSD subscriber count, China
EV competition, tariffs.

- **`tesla-deliveries-q2-2026-above-450k`**: TSLA price reacts violently to quarterly
  delivery prints; an event-clock contract would dominate Δlogit on print day.
  Found via discover: **no** (likely speculative — would need Polymarket to list).
- **`tesla-fsd-unsupervised-rollout-by-jun-30`**: FSD unsupervised release is the largest
  single thesis driver post-2025. Found via discover: **no — speculative**.
- **`china-raises-ev-tariffs-on-us-by-jun-30`**: TSLA Shanghai is exposure-locked to
  US-China auto tariffs; a binary on a tariff-hike materially Δlogit-moves the stock.
  Found via discover: **no — speculative, similar markets exist for Trump-era tariffs**.
- **`musk-out-as-tesla-ceo-before-2027`**: catalog already has this (`musk_out_as_tesla_ceo_before`).
  No gap.
- **`will-byd-overtake-tesla-deliveries-2026-q2`**: cross-listed competitor pressure
  metric; cleaner than the largest-by-mcap meta-bet. Found via discover: **no — speculative**.

The catalog also has 30+ tweet-count strikes for Musk (May-2026 buckets) which are noise,
not signal. Recommend deprioritising those over the deliverables/FSD set above.

---

## 2. Defense (LMT, ITA, RTX, NOC, GD)

The catalog has 117 geopolitics factors (Russia-Ukraine, Iran, China-Taiwan, NATO, etc.)
but **zero** ticker-specific defense factors. The geopolitics binaries already drive
defense-stock Δlogits via narrative; a few additions would close the gap.

- **`china-blockade-taiwan-by-june-30`**: already in catalog (`china_blockade_taiwan`).
  Verified: **yes**, $1.4M volume.
- **`russia-invade-a-nato-country-by-june-30-2026`**: already in catalog (`russia_invade_nato_jun`).
  Verified: **yes**, exists.
- **`will-the-us-officially-declare-war-on-iran-by-december-31-2026-746`**: already in
  catalog (`us_officially_declare_war_iran`). Verified: **yes**.
- **`russia-ukraine-ceasefire-by-jun-30-2026`**: catalog has Ukraine-peace-referendum but
  not a clean ceasefire binary. Found via discover: keyword `ceasefire` returns 0 in
  high-volume tier, **maybe** exists in tail.
- **`will-france-uk-or-germany-strike-iran-by-june-30-259`**: already in catalog. Verified: **yes**.
- **`us-defense-budget-above-1t-fy27`**: a budget-vote contract would be the cleanest
  defense-spending factor. Found via discover: **no — speculative**.
- **`nato-spending-pledge-3pct-of-gdp-by-2027`**: cleaner exposure than war binaries.
  Found via discover: **no — speculative**.
- **`anduril-ipo-before-2027`**: catalog has this. Verified: **yes**.

**Verdict for defense**: gap is in *positive-narrative* contracts (budgets, peace
deals); the war/escalation side is over-curated.

---

## 3. Energy (XLE, USO, CVX, XOM, OXY)

The catalog has 14 oil-price strike contracts (`cl-above-{42,50,63,70,75,90}-jun`,
`will-crude-oil-cl-hit-high-{115,140,150,175,200}`, `wti-reach-{110,120,140,150}-may`),
which is solid for level-targeting but missing the *flow* factors that drive XLE:
OPEC supply decisions, Saudi production targets, US SPR.

- **`opec-cuts-output-by-jun-meeting-above-1mbpd`**: OPEC meeting binaries materially
  re-price energy on the day. Found via discover: keyword `opec` returns 0 in
  high-volume tier; **maybe — likely exists in long tail**.
- **`saudi-arabia-oil-production-above-10mbpd-jun-2026`**: cleaner than meta-mcap markets.
  Found via discover: **no — speculative**.
- **`us-spr-refilled-above-400m-barrels-by-eoy`**: SPR draw/refill is a directional
  XLE driver. Found via discover: **no — speculative**.
- **`will-mohammed-bin-salman-cease-to-be-de-facto-leader-saudi-by-jun-30-2026`**: already
  in catalog (`p21_will_mohammed_bin_salman_cease...`). Verified: **yes**.
- **`will-trump-impose-gas-tax-by-eoy-2026`**: speculative; gas-tax binary would directly
  hit XLE/USO. Found via discover: **no — speculative**.
- **`gasoline-national-average-above-5-by-aug`**: pump-price binary aligns with election
  cycle attention. Found via discover: keyword `gasoline` returns 0; **maybe** in tail.

**Verdict for energy**: oil-price strikes are saturated. Gap is in
production/policy/strategic-reserve binaries.

---

## 4. Healthcare / Pharma (PFE, MRNA, NVO, LLY, JNJ)

This is the **most under-covered sector** in the catalog: only 5 health-themed factors,
0 of which are FDA-approval / drug-launch binaries. PFE/MRNA/NVO/LLY price action is
dominated by FDA decisions and GLP-1 share.

- **`fda-approves-novo-cagrisema-by-eoy-2026`**: NVO's GLP-1 follow-on; would directly
  re-price NVO and competitor LLY on approval. Found via discover: **no — speculative**.
- **`fda-approves-lly-orforglipron-oral-glp1-by-q3-2026`**: oral-GLP-1 launch is the LLY
  thesis. Found via discover: **no — speculative**.
- **`medicare-negotiation-list-2027-includes-keytruda`**: Medicare price-negotiation list
  expansion is a measurable PFE/MRK risk event. Found via discover: **no — speculative**.
- **`mrna-flu-vaccine-fda-approval-by-eoy`**: MRNA cash-flow story rests on flu/CMV/RSV
  pipeline. Found via discover: **no — speculative**.
- **`ozempic-shortage-resolved-by-jun-30`**: shortage-status binary tied to NVO/LLY
  capacity. Found via discover: **no — speculative**.
- **`gpt5-by-eoy`**: catalog already covers some "AI helps drug discovery" narrative
  via the AI theme (47 factors). No gap there.

**Verdict for healthcare**: catalog is essentially empty on FDA / Medicare events.
Adding even 5-10 FDA-approval binaries would make `/reverse-finder?ticker=LLY` and
`?ticker=NVO` produce useful results for the first time.

---

## 5. Sports betting (DKNG, PENN, MGM, FLUT)

**Zero coverage** of the operating drivers for sports-betting equities. The catalog
has 128 sports factors but they're **outcome** binaries (who wins the World Cup), not
**handle / state-legalization** binaries.

- **`california-legalizes-sports-betting-by-eoy-2027`**: CA legalization is a multi-year
  TAM catalyst; would re-price DKNG/FLUT 5-10% on signature. Found via discover:
  **no — speculative**.
- **`texas-legalizes-sports-betting-by-eoy-2027`**: same logic, second-largest unbet
  state. Found via discover: **no — speculative**.
- **`super-bowl-2027-handle-above-2b`**: Super Bowl prop activity drives Q1 prints.
  Found via discover: **no — speculative**.
- **`dkng-q2-2026-revenue-above-1.4b`**: clean earnings-print binary. Found via discover:
  **no — speculative**.
- **`penn-divests-espn-bet-by-eoy-2026`**: ESPN-Bet wind-down speculation already moves
  PENN. Found via discover: **no — speculative**.

**Verdict for sports betting**: catalog is functionally absent. Even one
state-legalization binary would close the largest sector gap by ticker-impact.

---

## 6. Crypto-equity (COIN, MSTR, MARA, RIOT, CLSK)

The catalog has 107 crypto factors but they're almost entirely **price-strike** contracts
on BTC ($5k through $1M, plus dip levels). Missing: the operational binaries that
differentiate COIN from MARA from MSTR.

- **`coinbase-staking-suspended-by-eoy-2026`**: a staking-injunction binary directly hits
  COIN's revenue mix. Found via discover: keyword `coinbase` returns 0 in high-volume tier;
  **maybe** in tail.
- **`microstrategy-sells-any-bitcoin-by-december-31-2026`**: catalog already has this
  (`microstrategy_sells_any_bitcoin_december` and `mstr_sells_btc`). Verified: **yes**.
- **`mara-hashrate-above-50eh-by-eoy-2026`**: hashrate is the cleanest MARA fundamental.
  Found via discover: **no — speculative**.
- **`bitcoin-difficulty-adjustment-above-110t-by-jun`**: difficulty adjustments hit miner
  margins immediately. Found via discover: **no — speculative**.
- **`spot-btc-etf-net-flows-positive-may-2026`**: monthly ETF-flow binary is the cleanest
  COIN/IBIT/FBIT factor. Found via discover: **no — speculative**.
- **`spot-eth-etf-staking-approved-by-sec-by-eoy-2026`**: ETH-spot-ETF staking is a clean
  ETHE/ETHA driver. Found via discover: **no — speculative**.
- **`circle-ipo-by-eoy-2026`**: stablecoin IPO would re-price COIN and crypto-banks.
  Found via discover: **no — speculative**.

**Verdict for crypto-equity**: BTC strikes are over-curated; operational binaries
(staking, hashrate, ETF flows) are absent.

---

## 7. Banks (JPM, GS, BAC, WFC, regional)

Catalog has 4 bank-tagged factors, 3 of which are SpaceX-IPO underwriter contests
(noise) and one is `will-jpmorgan-chase-fail-by-june-30-2026` (extreme tail). Missing
the meaningful bank factors.

- **`fed-bank-stress-test-2026-jpm-passes`**: stress-test pass/fail binary is the cleanest
  JPM/GS event-clock factor. Found via discover: **no — speculative**.
- **`fdic-takes-over-any-us-bank-by-june-30-2026`**: regional-bank stress proxy.
  Found via discover: **no — speculative**.
- **`us-deposit-flight-above-100b-q2-2026`**: deposit-flight binary tied to regional banks
  (KRE). Found via discover: **no — speculative**.
- **`jamie-dimon-out-as-jpm-ceo-before-2027`**: succession risk premium. Found via discover:
  Jamie Dimon appears only as a 2028 presidential candidate ($8.9M volume), which is
  the **inverse** signal (his political ambition could hint at exit). **maybe — adjacent
  market exists**.
- **`silicon-valley-bank-style-failure-2026`**: SVB-style event binary. Found via discover:
  **no — speculative**.
- **`fed-policy-error-recession-narrative-q3-2026`**: too vague; better captured by the
  existing `us_recession_2026`. **No gap**.

**Verdict for banks**: stress-test binary is the highest-priority single addition.

---

## 8. Climate / commodities

Catalog has 15 climate + 9 weather factors but they're earthquake/named-storm-formation
binaries, not the *frequency / intensity* binaries that drive insurance and commodity
ETFs.

- **`atlantic-hurricane-season-2026-named-storms-above-20`**: NOAA-style season-total
  binary; drives reinsurer pricing. Found via discover: keyword `hurricane` returns 0
  but `hurricane_cat4_us` exists in catalog. **partial coverage — gap remains for
  season-totals**.
- **`southwest-us-drought-classification-d4-by-aug-2026`**: drought-state binary affects
  agricultural and water utilities. Found via discover: **no — speculative**.
- **`lithium-carbonate-spot-above-15k-usd-per-ton-by-eoy`**: lithium spot driver of
  ALB/SQM. Found via discover: keyword `lithium` returns 0 in high-volume; **speculative
  — would need Polymarket to list**.
- **`copper-price-above-12k-by-eoy-2026`**: catalog has gold strikes but no copper.
  Found via discover: keyword `copper` returns 0; **speculative**.
- **`eu-carbon-permit-above-100-eur-by-jun-30`**: EU ETS price binary. Found via discover:
  **no — speculative**.
- **`pacific-typhoon-season-2026-above-30-named`**: insurance/reinsurance driver.
  Found via discover: **no — speculative**.

**Verdict for climate/commodities**: hurricane-season-total and drought-state binaries
are the cheapest two additions with clear ticker linkage (RE, AGRI, WEAT).

---

## 9. Macro/policy (orthogonal gaps not covered above)

Catalog has 89 macro factors, dominated by Fed-rate strikes (38). Notably absent:

- **`us-government-shutdown-by-oct-1-2026`**: shutdown binaries materially move USD,
  Treasuries and defense. Catalog: 0 mentions of `shutdown`. Found via discover:
  **no — speculative but historically liquid on Polymarket**.
- **`scotus-rules-on-tariff-authority-by-eoy-2026`**: SCOTUS-tariff-authority would
  re-price the entire industrial complex. Catalog: 0 mentions of `supreme court`.
  Found via discover: **no — speculative**.
- **`trump-imposes-25pct-tariff-on-mexico-by-jun-30`**: tariff binaries are the largest
  missing macro factor — 0 mentions of `tariff` in the catalog. Found via discover:
  **no — speculative but very high-impact**.
- **`trump-approval-rating-above-45-on-jun-30-2026`**: catalog has only one strike
  (`will-trumps-approval-rating-hit-35-in-2026`); a >45 strike captures the upside.
  Found via discover: **no — speculative**.
- **`debt-ceiling-raised-by-eoy-2026`**: speculative but historically liquid binary.
  Found via discover: **no — speculative**.

---

## 10. Prioritized list — top 20 highest-impact additions

Ranked by `(ticker_demand × discoverability)`. Discoverability scored Y/M/N where Y =
verified or near-verified on Polymarket, M = adjacent market exists, N = speculative.

| # | Proposed slug / theme | Ticker(s) impacted | Discoverability | Why it matters |
|---|---|---|---|---|
| 1 | `us-government-shutdown-by-oct-1-2026` | DXY, TLT, defense, GS | M | Catalog has 0 shutdown factors; historically Polymarket lists these. |
| 2 | `trump-imposes-25pct-tariff-on-mexico-by-jun-30` | F, GM, retailers, agri | M | Tariff is the single most-mentioned macro driver in 2026 news; catalog has 0 tariff factors. |
| 3 | `fda-approves-lly-orforglipron-oral-glp1-by-q3-2026` | LLY, NVO | N | Catalog has 0 FDA-approval binaries; LLY is a $700B mcap with no curated factor. |
| 4 | `california-legalizes-sports-betting-by-eoy-2027` | DKNG, FLUT, PENN, MGM | N | Single biggest TAM expansion in US gaming; would 5-10% re-price DKNG. |
| 5 | `spot-btc-etf-net-flows-positive-may-2026` | COIN, IBIT, FBTC, MSTR | N | Cleanest crypto-equity discriminator; catalog over-relies on price strikes. |
| 6 | `tesla-deliveries-q2-2026-above-450k` | TSLA | N | TSLA prints quarterly; print-day Δlogit dominates other factors. |
| 7 | `fed-bank-stress-test-2026-jpm-passes` | JPM, GS, BAC, KRE | N | Catalog has 4 bank factors, none stress-test. |
| 8 | `opec-cuts-output-by-jun-meeting-above-1mbpd` | XLE, USO, CVX, XOM | M | Catalog over-indexed on price strikes; supply-side binary is the cleaner factor. |
| 9 | `scotus-rules-on-tariff-authority-by-eoy-2026` | broad market | N | 0 SCOTUS factors in catalog; ruling would re-price the industrial complex. |
| 10 | `russia-ukraine-ceasefire-by-jun-30-2026` | LMT, ITA, NOC, RTX, GS | M | Catalog has the war side, missing the peace side; defense-stock asymmetry. |
| 11 | `tesla-fsd-unsupervised-rollout-by-jun-30` | TSLA | N | Largest single TSLA thesis; catalog has CA-robotaxi but not FSD-unsupervised. |
| 12 | `medicare-negotiation-list-2027-includes-keytruda` | MRK, PFE | N | Direct revenue hit binary; catalog has 0 Medicare factors. |
| 13 | `mara-hashrate-above-50eh-by-eoy-2026` | MARA, RIOT, CLSK | N | Catalog has 0 miner-fundamental factors. |
| 14 | `gasoline-national-average-above-5-by-aug` | XLE, retail, AAA | M | Pump-price binary aligns with consumer-spending narrative. |
| 15 | `china-raises-ev-tariffs-on-us-by-jun-30` | TSLA, F, GM, BYDDY | N | Direct TSLA Shanghai exposure; speculative but high-impact. |
| 16 | `atlantic-hurricane-season-2026-named-storms-above-20` | RE, BRK.B, P&C | N | Insurance/reinsurance pricing factor; catalog has only formation binaries. |
| 17 | `texas-legalizes-sports-betting-by-eoy-2027` | DKNG, FLUT | N | Second-largest unbet state; mirrors CA but more politically tractable. |
| 18 | `coinbase-staking-suspended-by-eoy-2026` | COIN | M | SEC enforcement risk; cleanest COIN-specific factor. |
| 19 | `us-spr-refilled-above-400m-barrels-by-eoy` | XLE, USO | N | SPR is a directional flow that retail-equity catalog ignores. |
| 20 | `lithium-carbonate-spot-above-15k-usd-per-ton-by-eoy` | ALB, SQM, LIT | N | 0 lithium/copper factors; would close the battery-metals gap entirely. |

---

## What is already saturated (no gap)

- **Fed rate path**: 38 FOMC strike factors across June/July/Sep/Dec 2026 plus
  Polymarket "no cuts" / "11+ cuts" / "12+ cuts" tail bets — exhaustive.
- **BTC price strikes**: 50 BTC-strike and BTC-dip factors covering $5k → $1M; further
  strikes are noise.
- **Crude oil price strikes**: 14 strikes covering $42 → $200; saturated.
- **2028 presidential primary**: politics theme is over-saturated with 50+ "Will X
  win the 2028 nomination" candidates; further additions are diminishing-returns noise.
- **Iran regime change / US-Iran nuclear deal**: 8+ overlapping binaries; saturated.
- **AI lab leaderboard / IPO**: 26 AI-tagged factors covering Anthropic/OpenAI/DeepSeek
  IPO and capability binaries — adequate.
- **Mcap-leadership meta-bets** ("Will TSLA / Aramco / Apple be largest by mcap on
  Dec 31"): 8+ exist; further additions are noise.

---

## Methodology caveats

1. The `/factors/discover` endpoint walks only the top ~500 markets by 24h volume.
   A "no" result therefore means "not in the high-volume tier," not "does not exist."
   Many proposed slugs marked **speculative** likely exist on Polymarket in the long
   tail and could be verified by querying the gamma API directly with `search=` param.
2. Slug names above are normalized to Polymarket's kebab-case convention but are not
   guaranteed to be the exact slug Polymarket uses. Verify against
   `https://polymarket.com/event/<question-fragment>` before adding to `factors.yml`.
3. Adding a factor to `factors.yml` only makes sense if the underlying market has ≥30
   daily observations; for new launches, wait until 30 days of `/prices-history`
   data accumulates (per `scripts/validate_factors.py`).
4. This audit does not consider Kalshi-side coverage. Several gaps (FOMC, CPI,
   recession) already have Kalshi coverage that the catalog absorbs as `KX*` slugs;
   Kalshi may be the right venue for several **N**-rated proposals (FDA approvals,
   bank stress-test outcomes).
