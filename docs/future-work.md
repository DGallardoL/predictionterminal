# Future work — out of POC scope

Items intentionally deferred. Each belongs in its own follow-up issue/PR.

## Quant

- **Lag structure.** Replace contemporaneous Δlogit with $\Delta\text{logit}_{t-k}$
  for $k \in \{0, 1, 2\}$ and use the lead-lag pattern as evidence of
  information flow direction.
- **Rolling betas.** Window the regression (e.g. 60-day rolling) to
  surface time-variation. Plot $\hat\beta_t$ as a fan chart.
- **Cross-sectional pooling.** Fit one model per industry rather than one
  per ticker, exploiting common factor exposure.
- **Backtest residuals.** Build a long/short strategy on the residuals
  and report Sharpe, with realistic transaction costs.
- **Soft transform.** Replace hard clipping with `arcsin(2p - 1)` or a
  Beta-prior smoothing to recover information near the 0/1 boundary.
- **VIF auto-pruning.** When `VIF > 5` on multiple factors, drop the most
  collinear and re-fit, surfacing the dropped factors in the response.

## Data sources

- **Kalshi.** Add a second `sources/kalshi.py` and let factors declare
  `source: kalshi` in `factors.yml`.
- **Volume / liquidity weighting.** Use Polymarket's `volumeClob` to
  down-weight low-liquidity days inside the regression.
- **Trade-level data via `warproxxx/poly_data`.** Pipeline from the
  Goldsky subgraph for tick-level analysis (see research notes).

## Engineering

- **Persistence (Postgres).** Save fits and let users reference them by
  id.
- **Auth.** Per-user API keys + a saved-runs UI.
- **Multi-replica.** Move Redis to a managed service; deploy multiple API
  replicas behind a load balancer.
- **Structured logs.** `structlog` JSON output to stdout, scraped by a
  log aggregator.
- **Metrics.** Prometheus `/metrics` endpoint with counters per endpoint
  and a histogram of fit durations.

## Frontend

- React or HTMX rewrite of the single HTML page.
- Save fits, compare side-by-side.
- Plot the residual time series and flag dates with $|e_t| > 2\sigma$.

## Alpha-tier follow-ups (Wave-7 v22 reckoning, 2026-05-19)

- **Earnings-surprise odds vs IV** — aspirational, no underlying factors. Revisit
  only when Polymarket lists liquid quarterly-EPS binaries on ≥ 6 large-cap
  names (search `factors.yml` for `earnings` / `beats_eps` / `eps_surprise`).
  Track which tickers gain coverage. Until then no point re-running the test.
  See `docs/alpha-reports/alpha-report-v22.md` §3.4.

- **Wave-6 promotion re-test calendar.** The 5 pairs reverted to B_VALIDATED
  (one marked `B_VALIDATED++`) per v22 §2 must be re-tested monthly. Schedule
  the re-runs of `/strategies/pairs-backtest` (window=20, entry_z=2, exit_z=0.5,
  stop_z=4, ann=252) split by 4 disjoint quarters on:
  - **2026-06-19** — first checkpoint
  - **2026-07-19** — second checkpoint
  - **2026-08-19** — earliest legitimate promotion window (renan_santos /
    us_aliens crosses 4Q on this date if Polymarket coverage persists)
  The promotion script (`api/src/pfm/alpha_tier_regen.py:_final_tier`) now
  hard-fails any pair with `joint_days < 360` at the data-availability layer,
  so manual override is required to re-promote. See v22 §5.

- **Sub-quarter Sharpe stability inside a quarter.** Current
  `quarterly_stability_test()` checks per-quarter Sharpes only. A finer-grained
  intra-quarter check (e.g. 13-week sub-windows) could catch single-event
  blow-ups that a quarterly average smooths over. Low priority — wait until
  enough Wave-6 pairs accumulate 4Q of data to make this useful.
