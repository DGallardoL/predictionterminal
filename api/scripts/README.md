# `api/scripts/`

Operational scripts that live alongside the FastAPI service. Most are
one-shot maintenance utilities (slug audits, factor catalogue pulls,
backtest harnesses); a few power scheduled CI jobs.

## `validate_factors.py`

Walks every entry in `src/pfm/factors.yml` and confirms the slug still
resolves at its upstream venue. Polymarket factors are checked against
`https://gamma-api.polymarket.com/markets?slug=<slug>` (empty list ⇒ DEAD),
Kalshi factors against `https://api.elections.kalshi.com/trade-api/v2/markets/<slug>`
(404 ⇒ DEAD). Sources without a cheap slug→exists endpoint (fred, bls,
manifold, predictit, chain) are reported as SKIPPED.

Run from the `api/` directory:

```bash
.venv/bin/python scripts/validate_factors.py \
    --source polymarket,kalshi \
    --limit 200 \
    --out factor_validation_report.json
```

### Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--source a,b,...` | `kalshi,polymarket` | Comma-separated source filter. Sources outside the live-checkable set are reported as SKIPPED. |
| `--limit N` | (none) | Process only the first N factors after filtering. CI uses 200 to stay under ~5 min. |
| `--strict` | off | Exit 1 on a single dead factor. Default mode exits 1 only when dead share > 5% of live-checked. |
| `--out PATH` | `./factor_validation_<UTC-date>.json` | JSON report destination. |
| `--workers N` | 20 | `ThreadPoolExecutor` max workers — matches the `/factors/rank` fan-out. |
| `--factors-yml PATH` | `src/pfm/factors.yml` | Override for tests. |

### Output

- **Stdout** — per-factor `[i/N] STATUS source slug` lines and a summary tail.
- **JSON report** — `{meta, ok[], dead[], skipped[]}`; arrays are sorted by
  factor id so the file diffs cleanly week-over-week.
- **Exit code** — `0` on a healthy catalog (dead share ≤ 5% in default mode,
  zero dead in `--strict`), `1` otherwise. A run that produces only SKIPPED
  entries (e.g. `--source fred`) exits 0 — nothing was actually live-checked.

### CI integration

The `validate-factors` job in `.github/workflows/ci.yml` runs this script
weekly (Monday 06:00 UTC) and on manual `workflow_dispatch`. It uploads the
JSON report as the `factor-validation-report` artifact. The job is skipped on
PR/push runs because the live HTTP calls would make per-PR CI flaky.

## Other scripts (reference only)

| Script | Purpose |
| --- | --- |
| `audit_dead_factors.py` | Heavier, write-back-to-yaml janitor that classifies polymarket slugs as ACTIVE/RESOLVED/DEAD and prunes the catalog in place. Run manually before a demo, not on CI. |
| `compute_live_signals.py` | Recomputes `web/data/live_signals.json` from current factor closes. |
| `regenerate_alpha_tiers.py` | Re-runs the 4-quarter robustness harness and writes `web/data/alpha_strategies.json`. |
| `run_alpha_hunter.py` | Batch alpha-search driver. |
| `btc_arb_*.py`, `chainlink_lag.py`, `poly_book_depth.py` | Investigation scripts kept for reproducibility; see commit history for context. |
| `wave21_pull.py`, `wave22_pull.py`, `build_factor_tags.py` | Catalog-expansion helpers from Wave-21/22. |
| `probe_polymarket.py` | Ad-hoc Polymarket gamma probe. |
