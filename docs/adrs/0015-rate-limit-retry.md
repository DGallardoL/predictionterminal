# ADR-0015: API Rate-Limit and Retry Strategy

- **Status**: Accepted
- **Date**: 2026-05-16
- **Wave**: 12
- **Supersedes**: none
- **Related**: ADR-0004 (Redis cache TTL), ADR-0008 (cache tiering), ADR-0011 (single-flight stampede prevention)

## Context

The Prediction Terminal aggregates data from four upstream HTTP sources whose
rate-limit budgets and failure modes differ by an order of magnitude:

| Source         | Documented budget       | Observed failure modes                     |
|----------------|-------------------------|--------------------------------------------|
| Polymarket CLOB / Gamma | ~1000 req / 10s (generous) | 429 with `Retry-After`; sporadic 502/503 during resolution events |
| Kalshi         | Stricter (≈60 req/min unauth, token-bucket auth) | 429 without `Retry-After`; long-tail latency on `/markets` |
| yfinance       | Undocumented; empirically ~5 req/s safe | Silent throttling, occasional `ConnectionError`, surprise HTML payloads |
| GDELT          | ~10 req/min               | Timeout under load, malformed JSON when over budget |

Naive `httpx` calls without retry logic were producing visible UI failures
during the demo path (Terminal landing → α Hub leaderboard → reverse-finder
SSE). A single transient 502 from Polymarket cascaded into a failed `/fit`
because we re-derived 200 curated factors serially. We need a uniform,
testable retry policy that respects per-source budgets, bounds tail latency,
and prevents thundering-herd retries from melting an already-stressed upstream.

The existing code references that motivate centralizing this policy:
- `pfm.sources.polymarket_pool` — HTTP/2 connection pool from T18
- `pfm.sources.kalshi_ratelimit` — token-bucket implementation
- `pfm.sources.yfinance_batch` — bounded-concurrency batcher from T19

Each currently handles failures with ad-hoc `try/except`, which is a clear DRY
violation and makes observability inconsistent (some log structured, some
silently swallow).

## Decision

Adopt a **per-source retry policy** built on `tenacity` (already a dependency
via `requirements.txt`) combined with a lightweight **per-source circuit
breaker**. The shared helper lives in `pfm.sources.retry` and is consumed by
every source-specific client.

Policy by exception class / status code:

1. **HTTP 429 (Too Many Requests)** — respect `Retry-After` header when
   present (Polymarket sends it; Kalshi sometimes does); otherwise fall back
   to exponential backoff with jitter starting at the source's minimum
   inter-request interval. Max 5 retries.
2. **HTTP 502 / 503 / 504** — exponential backoff `1s, 2s, 4s` with full
   jitter, max 3 retries. After 3, raise to caller (caller decides whether
   to serve stale cache via the ADR-0008 tiering).
3. **`httpx.ConnectError` / `RemoteProtocolError`** — 3 retries with
   exponential backoff + jitter (200ms base). DNS/TLS issues are usually
   transient on residential networks.
4. **`httpx.ReadTimeout` / `WriteTimeout`** — 1 retry with the per-call
   timeout doubled. A second timeout is almost certainly the upstream, not
   the network, so escalating wastes the budget.
5. **`json.JSONDecodeError` / malformed payload** — 1 retry, then raise.
   GDELT and yfinance occasionally return HTML error pages.

Circuit breaker (per source, per process):
- Window: rolling 60 seconds
- Trip threshold: 5 consecutive failures **after** the retry policy has been
  exhausted (so transient 502s do not trip; only sustained outage does)
- Open state: 30 seconds, during which all calls fast-fail with
  `UpstreamUnavailable` and the caller falls back to cache (ADR-0008) or a
  degraded response
- Half-open probe: a single request after the open window; success closes
  the breaker, another failure reopens for 60 s (doubling, capped at 5 min)

## Implementation

```python
# api/src/pfm/sources/retry.py
from tenacity import (
    retry, stop_after_attempt, wait_exponential_jitter,
    retry_if_exception, RetryCallState,
)

def make_policy(source: str) -> Callable:
    # ... maps status codes to wait strategies; honors Retry-After
```

Each source client wraps its low-level `httpx` call with the decorator
returned by `make_policy("polymarket" | "kalshi" | "yfinance" | "gdelt")`.
The circuit breaker is a small in-memory `CircuitState` dataclass keyed by
source, updated atomically via `threading.Lock` (we are single-process
gunicorn at `:8000`; multi-worker rollout is out of scope for the POC and
tracked in `docs/future-work.md`).

Observability: every retry emits a `structlog` event
`source.retry attempt=<n> source=<s> reason=<class>` and every breaker
state transition emits `source.circuit_breaker source=<s> state=<open|closed|half_open>`.
These will feed a Terminal sub-tile in a later wave.

## Consequences

**Positive**
- Tail latencies are bounded: worst case for the user is the sum of the
  retry budget (Polymarket: ≈ 1 + 2 + 4 ≈ 7 s for 5xx; Kalshi: respects
  server `Retry-After` so usually < 2 s).
- No cascading failures: when an upstream is genuinely down, the breaker
  trips in ~5 s and we fall through to cache for 30 s. Previously the
  reverse-finder SSE would spin for 30+ s before any data appeared.
- DRY: all source clients share one policy module, one logger, one set
  of metrics. Adding source #5 (FRED, planned) is ~5 lines.
- Testability: `tenacity` retry attempts are deterministic when we patch
  `time.sleep`, so the existing `respx`/`httpx_mock` tests can assert
  exact attempt counts.

**Negative**
- Per-source circuit-breaker state is in-process; a multi-worker gunicorn
  deployment would need a shared store (Redis). Acceptable for the POC.
- Slight memory overhead (~1 KB per source) for the breaker state.
- Retries can extend an already slow call. We mitigate by setting
  per-call `httpx` timeouts ≤ the retry budget so the total stays under
  the SSE keepalive window (10 s).

## Alternatives Considered

- **Custom retry decorator per source.** Rejected: DRY violation, and we
  saw three subtly different exponential-backoff implementations during
  audit (Polymarket pool, Kalshi limiter, GDELT helper).
- **Per-call back-pressure via a semaphore.** Operationally complex — we
  would need to tune per-source concurrency separately, and starvation
  is hard to reason about. Bounded concurrency at the batch layer
  (T19 `yfinance_batch`) already covers the worst offender.
- **Pure exponential backoff without a breaker.** Would still cascade
  when an upstream is fully down; the breaker is what turns a 30 s
  outage into a fast-fail.
- **External library (e.g. `pybreaker`).** Adds a dependency for ~80
  lines of logic and the project already standardizes on `tenacity`.

## Verification

- New tests in `api/tests/test_sources_retry.py` cover: 429 with and
  without `Retry-After`, 5xx exponential schedule, `ConnectError` jitter,
  timeout single-retry, breaker open after 5 consecutive failures,
  breaker half-open recovery.
- Existing source-client tests are updated to assert no behavioural
  regression (`respx` mocks still see the same first call).
