# ADR-0008: Curate the 1,090-factor universe by hand instead of scraping live

- **Status:** Accepted
- **Date:** 2026-05-08
- **Authors:** Damian Gallardo

## Context

The factor universe powers every mode of the application: regression
matrices, terminal heatmaps, alpha-hub strategies, the Reverse Factor
Finder, and the SSE live-stream. Choosing how that universe is *defined*
is therefore the single biggest reproducibility decision in the project.

Polymarket's `/markets` endpoint returns roughly **42,000 markets** at any
time, the vast majority of which have one of three problems that disqualify
them as factor candidates: (a) less than 30 daily observations of price
history, (b) liquidity below the $50k weekly volume floor we treat as the
"is this number trustworthy?" cutoff, or (c) duplicates of the same event
(e.g. 47 different "will Trump tweet today?" daily-resolving variants).
Kalshi's universe is an order of magnitude smaller but has the same shape.

The grading criteria for this project value reproducibility, and several
downstream artefacts (`docs/alpha-reports/alpha-report-v18.md`, the death certificates,
the wave-5 robustness check) **must** reference a stable factor list to
mean anything. A factor that disappears from the live API a week before the
demo invalidates every backtest that depended on it.

## Considered alternatives

- **Full live scrape on every request.** Most reactive, zero maintenance.
  Rejected: makes every Sharpe number unauditable. A market-listing change
  on the upstream silently changes every reported alpha. The audit trail
  becomes "ask the upstream what it returned that minute," which is not
  an audit trail.
- **Hybrid: curate a small core + scrape the long tail at runtime.**
  Tempting, and we partially do this via `/factors/discover` (which surfaces
  live high-volume markets *not* in the curated list as a UX hint). But the
  core regression / strategies / graveyard pipeline can only depend on the
  curated list, otherwise the hybrid is just the live-scrape path with
  extra steps.
- **Snapshot-once-at-deploy, write the snapshot to `factors.yml`.**
  Variant of the chosen approach but loses the human curation pass that
  catches duplicates and renames events into stable theme/sub-theme
  buckets.
- **Hand-curated `factors.yml` (chosen).** Two-track expansion via numbered
  "waves" (wave-1 through wave-9 to date), each adding a coherent block of
  factors keyed by theme. `scripts/validate_factors.py` walks the file and
  asserts every slug resolves and has ≥30 daily observations.

## Decision

Maintain `factors.yml` as the **single source of truth for the factor
universe**. Each entry has a stable slug, theme tag, sub-theme, and an
optional curator note explaining why the factor was added. The current
total is 1,090 factors (944 Polymarket + 146 Kalshi).

Expansion follows the **Wave N pattern** documented in `CLAUDE.md`:

1. Branch `wave-N-<theme>`.
2. Add slugs grouped by theme.
3. Run `scripts/validate_factors.py`.
4. Add no-network tests under `tests/fixtures/factors/`.
5. Bump the totals in `CLAUDE.md` and `README.md`.

Live discovery is **strictly UX-only**: `/factors/discover` surfaces
candidates the user can manually paste into the regression panel via the
"Custom" tab, but no scoring / strategies / alpha-hub artefact pulls from
discovery output. Discovery never enters `factors.yml` automatically.

## Consequences

- **Reproducibility wins.** Every Sharpe in `docs/alpha-reports/alpha-report-v18.md`
  ties back to a factor whose price history we can pin down to a snapshot
  date. Re-running the wave-5 robustness check next quarter is meaningful.
- **Maintenance cost is real but bounded.** A wave is roughly an
  afternoon of curation per ~100 factors. Nine waves so far; we expect
  one wave per month going forward.
- **Slug rot is the known failure mode.** Polymarket sometimes archives
  resolved markets and re-issues a related event under a new slug. The
  validator catches this on every PR; the response is a `wave-N-fixups`
  rename in the YAML. We do **not** auto-resolve to "current best match"
  — that would silently change historical Sharpes.
- **Coverage is intentionally narrower than the live universe.** Users
  who want a factor we don't carry can paste the slug into the Custom
  tab. They lose history-window guarantees and the strategies/graveyard
  pipeline ignores their pick, but the regression mode handles ad-hoc
  factors fine.
- **Demo cushion.** Damian can hot-patch `factors.yml` immediately
  before a demo if a curated slug got resolved upstream that morning,
  without having to re-run any quant validation gate.
