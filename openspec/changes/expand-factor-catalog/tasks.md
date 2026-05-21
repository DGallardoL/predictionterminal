## Tasks

### 1. Discovery & verification

- [x] Probe Polymarket Gamma for high-volume active markets across
      themes (~3000 candidates fetched, then filtered by volume,
      end-date, and an opinionated skip list of sports / entertainment /
      foreign-elections / 2028-primary-personality markets).
- [x] **Round 1**: Live-verify 58 candidates → 52 survivors (≥20 bars
      in 180d). Drops were May-31 markets just-listed.
- [x] **Round 2**: Live-verify 42 high-volume candidates → 42 survivors
      after slug-correction for 5 markets that had longer numeric
      suffixes than expected.

### 2. Catalog update

- [x] Append round-1 entries (52 factors, ~340 lines) to
      `api/src/pfm/factors.yml`.
- [x] Append round-2 entries (42 factors, ~270 lines) to
      `api/src/pfm/factors.yml`.
- [x] Total catalog: 51 → 145 (Polymarket 139 + Kalshi 6).
      Themes: macro 42 / geopolitics 23 / crypto 21 / ai 19 / chips 13 /
      politics 13 / commodities 6 / health 4 / climate 3 / energy 1.

### 3. Verification

- [x] `pytest tests/ -q` → 61/61 (no regressions).
- [x] `GET /factors` returns 145 entries with all required fields.
- [x] **Smoke fit (round 1)**: XOM × `[oil_above_115_jun,
      oil_above_150_jun, oil_below_70_jun]` → n=32, R²=0.16,
      `oil_below_70_jun` p=0.048.
- [x] **Smoke fit (round 2)**: GLD × `[gold_5500_jun, oil_above_175_jun,
      us_invade_cuba, ukraine_joins_nato, fed_cuts_3_2026]` → n=65,
      R²=0.24, `gold_5500_jun` **t=+3.77 p<0.001** (strong real signal),
      permutation p=0.10 (marginal-signal verdict).

### 4. Cleanup

- [ ] Archive with `openspec archive expand-factor-catalog --yes`
      from project root once Damian confirms the new factors render
      in the Curated tab in the browser.
