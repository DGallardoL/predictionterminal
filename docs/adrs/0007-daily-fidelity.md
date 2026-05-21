# ADR-0007: Always request `fidelity=1440` (daily) from Polymarket CLOB

- **Status:** Accepted
- **Date:** 2026-04-30
- **Authors:** Damian Gallardo

## Context

The Polymarket CLOB endpoint
`GET clob.polymarket.com/prices-history?market=…&fidelity=…` accepts a
`fidelity` parameter in **minutes** controlling the bucket size of the
returned series.

A known limitation is documented in
[py-clob-client issue #216](https://github.com/Polymarket/py-clob-client/issues/216):
**resolved (closed) markets only return data for `fidelity ≥ 720` (12 h)**.
Sub-12h requests against resolved markets return an **empty `history`
array**. The endpoint does not 4xx; it just silently returns nothing.

This is a footgun: a developer testing on an active market with
`fidelity=60` sees data, then the same code returns nothing once the
market resolves a few weeks later. We saw this exact regression while
prototyping.

## Considered alternatives

- **`fidelity=60` (hourly) for active markets, `1440` for resolved.**
  Doubles the code paths and conditional logic; needs to know the market
  state at fetch time. Adds complexity for no POC benefit (we use only
  daily returns anyway).
- **`fidelity=1` (minute).** Maximum granularity but huge response payloads
  and unusable for resolved markets. We don't even need minute-level data
  for daily-return regressions.
- **Don't pass `fidelity` and rely on the default (= 1).** Hits the
  resolved-market footgun.

## Decision

**Always send `fidelity=1440`.** The constant lives in
`pfm.sources.polymarket.DAILY_FIDELITY` and is used unconditionally by
`get_price_history`. There is no API-level knob to override it — users
of `/fit` cannot accidentally select a sub-daily fidelity and silently get
empty data.

The integration test asserts this:

```python
def test_get_price_history_uses_daily_fidelity(client):
    ... # respx route
    client.get_price_history("token-id")
    assert request.url.params["fidelity"] == "1440"
```

so a future refactor can't quietly break it.

## Consequences

- The API does not support intraday returns regressions. That is the
  intended scope (see PLAN.md §3 — "rolling betas / time-varying" is
  future work).
- The model only ever sees one observation per UTC day per factor, which
  matches yfinance's daily granularity and removes the need for any
  intraday alignment logic.
- If we ever do want intraday data, it would need a separate code path
  that detects market state first and chooses fidelity accordingly. We
  defer that decision and document the constraint here.
