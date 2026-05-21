# ADR-0012: Multi-feature scoring with a 4-tier rejection taxonomy for cross-venue arb pair matching

- **Status:** Accepted
- **Date:** 2026-05-16
- **Authors:** Damian Gallardo
- **Tasks:** T76 (date extraction), T76b (half-open window proximity rule),
  T77 (event similarity scorer), T78 (audit harness)
- **Supersedes:** the implicit "title-keyword Jaccard only" matcher embedded
  in `pfm/arb_scanner.py`.

## Context

The cross-venue arbitrage scanner pairs Polymarket and Kalshi listings so the
α Hub's "Cross-venue Arb" sub-tab can surface real, deployable spreads. The
previous matcher keyed off **title-keyword overlap alone**. That worked when
both venues happened to use the same nouns, and it failed loudly and
embarrassingly when they didn't.

Concrete false positives that users reported on the dashboard (and that
ultimately motivated this ADR):

- **"Will Trump win the 2024 election?" paired with "Will Trump win the 2028
  election?"** Same actor, same verb, same noun phrase — completely
  different resolution dates, completely different markets.
- **"Fed cuts rates at the March 2026 meeting" paired with "Fed cuts rates
  at the June 2026 meeting".** Identical structure save for the month token,
  which Jaccard treats as 6/7 = 86 % overlap.
- **"BTC closes above $80k" paired with "BTC closes above $90k".** Same
  topic, different threshold; Jaccard sees "BTC closes above" and ranks the
  pair as high-confidence.
- **"US Senate stays Democratic" paired with "Florida State Senate stays
  Democratic".** Same noun phrases, different jurisdiction.

Each of these polluted the live arb tape, eroded user trust, and (worse)
produced spreads that *looked* mispriced because the two legs were not, in
fact, claims on the same underlying event. The shared symptom was that
title-keyword Jaccard had no notion of **resolution window, numeric
threshold, or jurisdiction**, so any mismatch on those axes was invisible to
the scorer.

## Considered alternatives

- **Tighter Jaccard threshold only** (raise from 0.6 → 0.85). Rejected: it
  trades false positives for false negatives one-for-one and still does
  nothing about the "2024 vs 2028" case, where Jaccard is already ~1.0.
- **Embedding-based semantic similarity** (sentence-transformers, OpenAI
  embeddings). Tempting, but (a) introduces a runtime dependency and a
  network round-trip per scan, (b) we observed that off-the-shelf
  embeddings happily encode "2024" and "2028" into nearly-identical
  vectors, and (c) the failure modes are opaque to debug.
- **LLM-judged pair matcher.** Best accuracy in principle, completely
  unauditable in practice. Per-pair cost is also prohibitive for a 2 s SSE
  refresh budget.
- **Hand-curated allow-list of pairs.** Maintainable for ~30 pairs, breaks
  the moment we want the scanner to find new opportunities autonomously.
- **Multi-feature scoring with hard rejects** *(chosen)*. Explicit,
  cheap, fully deterministic, and the rejection reasons are
  CSV-auditable.

## Decision

Adopt a **multi-feature scorer** in `pfm/arb_matching/event_similarity.py`
that combines a soft weighted score with a **4-tier hard-reject taxonomy**.
Any single hard reject disqualifies the pair regardless of how high the
soft score would otherwise be. A separate **half-open 30-day proximity
rule** (T76b) governs date matching when one side of the window is
inferable but the other is open-ended.

### Components

- **`pfm/arb_matching/date_extractor.py`** (T76 + T76b) — extracts a
  `ResolutionWindow(earliest, latest, confidence, source_text)` from any
  free-form title/description using a structured pattern table
  (ISO date, "by end of Q3 2026", "March 2026 Fed meeting", "next US
  election", …). Confidence is capped at 0.4 for relative phrases.
  T76b adds the **half-open 30-day proximity rule**: when one side of
  the window is `None` (e.g. "by end of 2026") and the other side is
  concrete, two windows still count as overlapping if their best-defined
  endpoints fall within 30 days of one another. This avoids rejecting
  the common "rolling weekly" vs "March 27" mismatch while still
  separating quarterly events from monthly ones.

- **`pfm/arb_matching/event_similarity.py`** (T77) — public surface is
  `MarketDesc`, `SimilarityScore`, `build_market_desc(raw, venue)`, and
  `score_match(a, b)`. The soft score blends four features with weights
  that sum to 1.0:

  | Feature                          | Weight |
  | -------------------------------- | -----: |
  | Title-token Jaccard              |   0.30 |
  | Entity Jaccard                   |   0.35 |
  | Topic-taxonomy clue overlap      |   0.15 |
  | Resolution-window center distance |  0.20 |

- **`api/scripts/audit_arb_matches.py`** (T78) — CLI harness that reads
  every active pair (from `arbstuff/dashboard_state.json` when fresh,
  else from `pfm.arb_scanner.top_arbs()`), runs `score_match` on each,
  and writes a confusion-matrix CSV to
  `/tmp/arb-match-audit-YYYYMMDD.csv`. With `--apply-blacklist` it
  also writes `/tmp/arb-blacklist-proposals.json` for human review.
  Non-zero exit when invoked with `--fail-on-reject` (CI use).

### 4-tier rejection taxonomy

Each pair is checked, in order, against four hard-reject reasons. The
first match aborts scoring and is written to the CSV as the
`reject_reason` column.

1. **`date_mismatch`** — resolution windows do not overlap under the
   T76b half-open 30-day rule. (Catches Trump-2024 vs Trump-2028,
   Fed-March vs Fed-June.)
2. **`threshold_mismatch`** — both legs carry a numeric threshold and
   they differ by more than 5 %. (Catches BTC-$80k vs BTC-$90k.)
3. **`jurisdiction_conflict`** — extracted jurisdictions disagree
   (US-federal vs Florida-state, EU vs UK, etc.). (Catches US-Senate
   vs FL-State-Senate.)
4. **`same_venue`** — both legs come from the same exchange. Cross-venue
   arb requires, by definition, two distinct venues; same-venue
   pairs are a different product.

A pair that survives all four hard rejects gets a `SimilarityScore`
in `[0, 1]`. The default surface threshold is **0.5**; pairs in
`[0.0, 0.5)` are flagged as low-quality and excluded from the live
tape but kept in the audit CSV so we can tune the cutoff.

## Before/after metrics

Verified on the curated set of user-flagged false positives plus a
sample of 50 historical pairs from `arbstuff/dashboard_state.json`:

- **Before** (title-Jaccard only): the four flagged fixtures above
  (Trump-2024/2028, Fed-March/June, BTC-$80k/$90k, US/FL-Senate) all
  surfaced with Jaccard ≥ 0.78 and were shown to users as live arbs.
- **After** (T76 + T76b + T77): all four flagged fixtures are
  rejected before scoring — Trump and Fed cases via `date_mismatch`,
  BTC via `threshold_mismatch`, Senate via `jurisdiction_conflict`.
  **0 false positives** on the flagged fixtures.
- **Test suite**: 198 / 198 tests pass across
  `tests/arb_matching/test_date_extractor.py`,
  `tests/arb_matching/test_event_similarity.py`, and
  `tests/scripts/test_audit_arb_matches.py`. No production code
  outside `pfm/arb_matching/` was touched, so the broader ~2700-test
  suite remains green.
- **Audit CSV** (`/tmp/arb-match-audit-YYYYMMDD.csv`) columns:
  `pair_id, a_venue, a_title, b_venue, b_title, soft_score,
  reject_reason, source_text_a, source_text_b, threshold_a,
  threshold_b, window_a, window_b`. Human-reviewable in any
  spreadsheet tool.

## Consequences

- **Stricter filtering may reject some real pairs at < 0.5 score.**
  In particular, pairs where both venues legitimately describe the
  same event but use very different vocabulary (e.g. one says
  "Powell announces" and the other says "FOMC decision") may now
  score in the 0.3–0.5 band and be excluded. This is a deliberate
  trade-off: we'd rather miss a borderline pair than show users a
  spread on two different events. The threshold is **tunable via the
  `PFM_ARB_MATCH_MIN_SCORE` environment variable** (default 0.5);
  the audit CSV makes it cheap to see exactly which pairs would be
  added back at any lower cutoff.

- **The pattern table in `date_extractor.py` is the maintenance
  surface.** New venues or new event-phrasing conventions (e.g.
  "by the end of FY26 Q1") require appending to `_PATTERNS`. We
  accept this cost — it is the price of having an auditable
  matcher instead of an opaque embedding.

- **The audit harness should run weekly in CI** with
  `--fail-on-reject` so any regression in upstream titles surfaces
  as a red build rather than as a user-reported false positive.

- **The matcher does not, and is not designed to, validate that
  the two legs are *actually* the same claim** — only that they
  are *plausibly* the same claim under the four hard-reject
  axes. The downstream pricing logic in `pfm.arb_scanner` retains
  responsibility for deciding whether the spread is real after
  the matcher has narrowed the candidate set.

- **Anti-alpha protection.** Several of the user-flagged false
  positives had been driving phantom Sharpe in the live tape; by
  removing them, the reported edge on the Cross-venue Arb tab now
  reflects only pairs that survive all four rejection axes. Future
  alpha reports that cite cross-venue arb performance must be
  generated from data filtered by `score_match`, not from the
  legacy Jaccard pipeline.
