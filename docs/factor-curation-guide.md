# Factor Curation Guide

This document is the operational manual for adding, validating, auditing, and
pruning factor entries inside `api/src/pfm/factors.yml`. The catalog currently
holds **1,228 active factors** (verified 2026-05-16, post jumps + sentiment
session) spread across nineteen themes and six source types. The catalog is the
single most important asset of the Regression mode: every `/fit` call resolves
its `factors=[...]` query against this file, and the α Hub research surface
treats validated slugs here as the universe over which all curated alphas are
discovered. Sloppy edits cascade — a typo in a slug, a duplicated id, or a
silently-dead Polymarket market all cause `/fit` to fail in subtle ways that
look like model bugs but are actually data-layer bugs. This guide exists so
that future Claude agents and human contributors converge on the same
discipline.

A note on file ownership: per `.coordination/PROTOCOL-V2.md`, **no agent may
edit `factors.yml` directly**. The risk of YAML corruption on a 1,360-entry
file with many concurrent writers is unacceptable. Instead, propose additions
in a side file (`factors_wave_N.yml` or `factors_wave_N_<theme>.yml`) and ask
Damian to merge during a quiet window. The rules below describe the *content*
of those proposals, not the merge mechanics.

## 1. Factor lifecycle: discovery → vetting → catalog → audit → prune

Every factor moves through five phases. Skipping any phase produces a stale
catalog within a quarter.

1. **Discovery.** A new market or macro series surfaces — a Polymarket election
   slug, a fresh FRED indicator, a Manifold question with deep liquidity, or
   an analyst-curated `sentiment:` query. Discovery channels include
   `pfm.terminal_calendar` (upcoming resolution events), `arb_scanner` cross-
   venue scans, and ad-hoc requests from the user.
2. **Vetting.** Before adding to the catalog, the candidate must pass three
   filters: (a) at least 30 daily observations over the past 90 days; (b)
   non-degenerate variance (σ_log_return > 1e-4); (c) no slug-collision with
   an existing entry. Vetting is automated by
   `api/scripts/validate_factors.py`, which mocks upstream calls in CI and
   uses cached fixtures from `tests/fixtures/factors/`.
3. **Catalog.** The proposal is appended to the wave-specific YAML file with
   all required and applicable optional fields filled in. Wave numbering
   follows the convention `wave-N-<theme>` (e.g. `wave-10-commodities`).
4. **Audit.** Each Sunday a CI job (`audit_factors`) re-runs the vetting
   filters against the live catalog. Factors that drop below the observation
   floor are tagged `degraded` in `docs/factor-audit-log.md`.
5. **Prune.** After two consecutive weekly audits in `degraded` state, a
   factor is either *migrated* (slug replaced; observations re-checked) or
   *retired* (entry deleted, retirement note added to the audit log). See
   §10 for the migrate-vs-retire decision tree.

## 2. Source types

Six source types are supported. Each has its own resolver in `pfm.factors` and
its own fixture format.

- **`polymarket`** — Resolves a market slug to a clobTokenId via the gamma API
  (`/markets`), then pulls `/prices-history?fidelity=1440` to build a daily
  log-odds series. Polymarket slugs are case-sensitive and may be re-mapped
  by Polymarket without notice; always cross-check at vetting time.
- **`kalshi`** — Uses the Kalshi v2 events API. Tickers are upper-snake (e.g.
  `KXFEDDECISION-25DEC-T425`). Auth is required for some endpoints; CI uses
  recorded fixtures.
- **`manifold`** — Liquidity is thin outside the top ~200 markets. Only use
  Manifold for niche themes (academic, AI-progress) where Polymarket has no
  coverage.
- **`predictit`** — Legacy support only. PredictIt's API is rate-limited to
  60 req/minute and frequently has resolved-market gaps; vetting threshold is
  raised to ≥45 obs/90d to compensate.
- **`bls`** — Bureau of Labor Statistics; requires a `series_id` (e.g.
  `CES0000000001` for total nonfarm employment). Monthly frequency means BLS
  factors are forward-filled inside the resolver; document this in the
  factor description.
- **`fred`** — Federal Reserve Economic Data; requires a `series_id` (e.g.
  `DGS10` for 10Y Treasury yield). Daily or weekly frequency depending on
  series; weekly series are forward-filled like BLS.

## 3. Required fields

Every factor entry MUST carry these six keys:

- **`id`** — unique stable identifier inside the catalog, lowercase-kebab.
  Used as the dictionary key in factor responses and as the column header in
  the regression design matrix.
- **`name`** — human-readable label shown in the frontend factor picker.
- **`slug`** — the upstream identifier (Polymarket slug, Kalshi ticker,
  FRED/BLS series id). Resolver chooses the right adapter by `source`.
- **`source`** — one of the six types above.
- **`theme`** — one of the 19 active themes (see §7).
- **`description`** — one to three sentences explaining what the factor
  measures and which direction is bullish. The description is surfaced in
  the `/factors/all` endpoint and read by users when they pick a factor.

If any required field is missing, the W11-42 schema test fails the build.

## 4. Optional fields

Four optional fields cover edge cases.

- **`is_probability`** (bool, default `true` for prediction-market sources,
  `false` for FRED/BLS). When true, the resolver applies the
  configurable clipping epsilon (default `ε = 0.01`) before computing
  log-odds: `clip(p, ε, 1-ε)`. When false, the raw level is used and the
  resolver computes log-returns instead.
- **`series_id`** — required for `bls` and `fred`, otherwise omitted.
- **`chain`** — present only for **chain factors** (see §9). Indicates this
  factor is the concatenation of multiple slugs across time, used when a
  single market resolves before its theme is exhausted.
- **`segments`** — a list of `{slug, start, end}` dicts. Required when
  `chain: true`; forbidden otherwise.

A handful of legacy entries also carry `notes` (free text) and `quality`
(`A`, `B`, or `C` per the validated-alphas tiering). New entries should NOT
introduce additional optional fields without a corresponding schema-test
update.

## 5. Validation: the W12-09 dead-slug detector

A slug is *dead* when fewer than 30 daily observations exist in the trailing
90-day window. Dead slugs degrade the regression silently — the OLS still
runs but the factor column is mostly NaN, the HAC standard errors blow up,
and the user sees an enormous beta with a meaningless p-value.

The W12-09 detector lives in `api/scripts/detect_dead_slugs.py`. It runs
nightly in CI and produces a JSON report under `tests/fixtures/factor_audit/
dead_slugs_<date>.json`. Two failure modes are distinguished:

- **`resolved`** — market resolved within the window; expected.
- **`stale`** — market still listed as active upstream but no new prints;
  this is the actionable mode.

`stale` slugs must be migrated within two weekly audits or retired. The
detector also flags factors whose σ_log_return collapses below 1e-4 (a
"frozen" market) — these are functionally dead even if observation counts
look healthy.

## 6. Naming conventions

- IDs are **lowercase-kebab**: `fed-hawkish-2026q1`, not `FedHawkish2026Q1`.
- No spaces, no underscores, no leading digits.
- Theme prefix is **optional but encouraged** for new entries:
  `politics-house-control-2026` reads better than `house-control-2026`.
- Source prefix is **forbidden** in IDs (the `source` field already encodes
  it). `polymarket-fed-hike` is wrong; `fed-hike-march-2026` is right.
- Avoid version suffixes (`-v2`, `-final`) — if a slug changes, migrate and
  log the old slug in the audit file rather than minting a new id.
- Sentiment factors are a special case: their id is **always**
  `sentiment:<query>` with a single colon (see §8).

## 7. Themes (19 active, per W11-42)

The W11-42 schema test enforces this exact list:

`politics`, `macro`, `crypto`, `equities`, `commodities`, `geopolitics`,
`fed`, `elections`, `sports`, `weather`, `tech`, `ai`, `health`, `energy`,
`rates`, `inflation`, `housing`, `labor`, `sentiment`.

Adding a new theme requires (a) a one-line justification in the wave PR, (b)
an update to the W11-42 test, and (c) coordination with the frontend
maintainer so the factor picker's grouping UI doesn't show an empty bucket.

## 8. Adding sentiment factors

Sentiment factors are NLP-derived signals; they do not have an upstream
market or series. The id pattern is `sentiment:<query>` where `<query>` is a
free-form lowercase-kebab phrase: `sentiment:fed-hawkish`,
`sentiment:earnings-beat-tech`, `sentiment:china-tariffs`.

At resolve time, the factor module forwards the query to
`pfm.terminal.sentiment_nlp.score_text` over a rolling 90-day headline
corpus and returns a daily VADER+financial-lex blended score in `[-1, 1]`.
LRU caching (10k entries) is automatic. The catalog ships 10 curated
sentiment factors; users can also pass arbitrary `sentiment:<query>` slugs
directly on `/fit` without registering them — they just won't appear in
`/factors/all`.

When adding a curated sentiment entry, set `source: sentiment`,
`is_probability: false`, and use the `sentiment` theme. Description should
note which corpus (default: Reuters + Bloomberg headline cache) and the
expected sign (positive score → bullish stocks).

## 9. Adding chain factors

Chain factors are the workaround for markets that resolve mid-window. If you
want a continuous 12-month series for "Fed cuts rates in next FOMC", you
cannot use one slug — each meeting's market resolves and the next one opens
afterwards. Chain factors solve this by declaring `chain: true` and listing
explicit segments.

Each segment is a `{slug, start, end}` dict where `start` and `end` are ISO
dates. Segments must be contiguous (`end` of segment N equals `start` of
segment N+1) and non-overlapping. The resolver fetches each segment, builds
its log-odds series, and concatenates them at the segment boundaries.

Chain factors must use a **single source** — you cannot chain Polymarket
into Kalshi inside one entry. Cross-venue equivalence belongs in the
`arb_scanner`, not in the factor catalog.

## 10. Removal: migrate vs prune

When a factor goes `stale` for two consecutive audits:

- **Migrate** if the underlying theme is still active and a near-equivalent
  upstream slug exists (e.g. the same Polymarket market reopened with a new
  slug after a question revision). Update the `slug` field, keep the `id`
  stable so historical fits remain reproducible, and log the swap in
  `docs/factor-audit-log.md` with the old slug, the new slug, and the
  observed date of the changeover.
- **Prune** if the theme has expired (post-election markets, resolved
  one-shot events) or no equivalent exists. Delete the entry, add a
  retirement note to the audit log with the final observation date, and
  decrement the catalog total in `CLAUDE.md`'s Scale section.

Never prune without an audit-log entry — historical α Hub reports cite
factor ids, and a silent deletion turns a reproducible report into a
mystery.

## 11. CI checks

Two CI jobs guard the catalog:

- **W11-42 schema test** (`tests/factors/test_schema_w1142.py`) validates
  every entry against the required/optional field rules in §3–4, enforces
  the 19-theme allowlist in §7, and verifies id uniqueness. Runs on every
  PR.
- **W12-09 dead-slug sweep** (`tests/factors/test_dead_slugs_w1209.py` plus
  `scripts/detect_dead_slugs.py`) runs nightly against fixtures and weekly
  against the live catalog. It produces the audit report consumed by §5 and
  §10.

A green PR on `factors_wave_N.yml` means schema is valid, but it does not
mean upstream is alive — only the nightly sweep proves liveness. Treat the
weekly audit as the source of truth for catalog health.

## 12. Common mistakes

- **Typos in slugs.** Polymarket slugs are long and easy to mistype. Always
  copy from the gamma API response, never from the URL bar.
- **Duplicate ids.** Two entries with the same `id` will silently shadow each
  other inside the YAML loader — the second wins. W11-42 catches this.
- **Mismatched source.** Pointing a `kalshi` source at a Polymarket slug
  fails at resolve time with a confusing 404; double-check the source field
  matches the slug format.
- **Forgotten `series_id`** for FRED/BLS entries — the resolver will fall
  back to treating the `slug` as the series id and may "work" but return
  the wrong series.
- **Theme drift.** Re-using `politics` for a Fed-decision market because
  "it's political" violates the W11-42 allowlist semantics; use `fed`.
- **Probability flag wrong.** Setting `is_probability: true` on a FRED yield
  series produces nonsensical log-odds; setting it `false` on a Polymarket
  contract skips the clipping step and lets ε=0 propagate.
- **Editing `factors.yml` directly.** Per PROTOCOL-V2, this is forbidden.
  Always propose in a wave file and let Damian merge.

Following this guide keeps the catalog reproducible, the audit log honest,
and the α Hub backtests trustworthy.
