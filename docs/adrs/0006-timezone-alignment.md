# ADR-0006: Align all dates to UTC midnight

- **Status:** Accepted
- **Date:** 2026-04-30
- **Authors:** Damian Gallardo

## Context

We join three time series:

1. **Polymarket** `prices-history` returns unix-second timestamps. With
   `fidelity=1440` (daily) the bucket boundaries are typically aligned to
   UTC midnight, but the API does not formally guarantee this.
2. **yfinance** returns a `DatetimeIndex` whose interpretation depends on
   the exchange (US equity closes at 16:00 ET; yfinance reports the date
   only, dropping the intra-day time but in the exchange's local zone).
3. **User input** (`start`, `end`, `date` query parameters) is naïve calendar
   dates with no timezone information.

If we don't pin a single convention, the inner-join across three sources
will silently drop rows or, worse, produce off-by-one-day artefacts where
$\Delta\text{logit}$ from day $t$ regresses against the return of day $t+1$.

The trap is real. During development of an earlier prototype I observed
~10% data loss on a 6-month window because `pd.Timestamp("2025-03-15")` (no
tz) and `pd.Timestamp("2025-03-15", tz="UTC")` did not compare equal in a
join.

## Considered alternatives

- **Local exchange time.** Plausible for US equity-only, but breaks for any
  cross-market study (and the project may extend to non-US markets).
- **Naive Timestamps everywhere.** Simpler but masks the fact that
  Polymarket bars are unambiguously UTC. Mixing tz-aware and tz-naive in
  pandas is itself a footgun.
- **Per-source timezone, then reconcile at fit-time.** Defers the problem
  rather than solving it — every transform downstream has to remember
  which series is in which zone.

## Decision

**All dates are normalised to UTC midnight** (`pandas.Timestamp(...,
tz="UTC").normalize()`) **as soon as data enters the application boundary**.

Specifically:

- `polymarket.get_price_history` converts unix seconds with
  `pd.to_datetime(unit="s", utc=True).dt.normalize()`.
- `equity.get_log_returns` converts the yfinance index with
  `pd.to_datetime(idx, utc=True).normalize()`.
- The API layer parses `start` / `end` / `date` request fields with
  `pd.Timestamp(value, tz="UTC")`.

The inner-join is then over a homogeneous `DatetimeIndex` of UTC midnights.

## Consequences

- The fit is, strictly speaking, a regression of "calendar-day return on
  calendar-day Δlogit" — not "16:00 ET return on the Polymarket close at
  16:00 ET". For most demo purposes the two are indistinguishable, but it
  is documented here so the demo can answer the question honestly.
- DST does not affect us because we never use a tz with DST.
- A future cross-market extension can keep this convention; no rework.
