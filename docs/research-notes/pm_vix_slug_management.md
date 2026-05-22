# PM-VIX Slug Management

This note documents the auto-refresh subsystem that keeps the PM-VIX
composite index grounded in *live* Polymarket contracts rather than the
hardcoded slugs that ship in `pfm/pm_vix.py`.

## Why slugs go dead

The headline PM-VIX score is built from five buckets (`recession`,
`geopolitical`, `election`, `macro`, `crypto`), each of which lists a
handful of Polymarket *slugs* whose YES probability we want.
Hardcoded slugs decay for three reasons:

1. **Markets resolve.** A "US recession by end of 2026" market closes
   on 2026-12-31; on day 366 the slug returns an empty list from
   `gamma-api.polymarket.com/markets?slug=…`. Polymarket keeps resolved
   markets reachable for a few weeks via `?closed=true`, but the data
   is stale and the spread is meaningless.
2. **Markets get renamed.** Polymarket curators occasionally rewrite
   slugs to canonicalise wording (`"will-trump-…"` → `"trump-…"`),
   and the old slug becomes a 404.
3. **Markets never existed.** A handful of slugs in the original
   bucket map were guesses that never matched anything live (see the
   pre-refresh logs: ~50% of the original 18 slugs returned
   `no market for slug=…`).

A score driven half by missing markets is not financially meaningful.
The auto-refresh fixes that without checking dynamic slugs into the
hardcoded map.

## Architecture

`pfm/pm_vix.py` exposes:

- **`validate_and_refresh_buckets(http)`** — async function that, for
  each bucket, probes every hardcoded slug against Gamma in parallel,
  marks the dead ones, and replaces them with the top-N most-active
  live markets surfaced by a per-bucket keyword search. Persists the
  refreshed map atomically to `/tmp/pfm_pm_vix_slugs.json`.
- **`_get_active_slugs()`** — returns the persisted map if it exists
  and is fresh (≤24h); otherwise returns `{}`. `compute_pm_vix` then
  falls back to the hardcoded `BUCKET_SLUGS` bucket-by-bucket.
- **`run_forever_slug_refresh(interval_seconds=21600)`** — background
  task that runs `validate_and_refresh_buckets` every 6 hours.
  Opt-in via `PFM_PM_VIX_AUTO_REFRESH_ENABLED=1`.

### TTL: why 24h?

Gamma's `markets` endpoint typically reflects renames and resolutions
within minutes, but most users hit `/indices/pm-vix` from the frontend
many times per minute. A 24h TTL means:

- One refresh per day fully covers normal market churn.
- Frontend reads stay sub-millisecond on the in-process memory cache.
- A refresh failure (Gamma down, search returns nothing) doesn't
  immediately demote the score — we keep yesterday's live map for
  another 24h before falling back to the hardcoded list.
- Running the refresh on the 6h cron gives us 4 chances per day to
  recover from a transient failure before the cache expires.

Tune via `SLUG_CACHE_TTL_SECONDS` in `pfm/pm_vix.py` (or override the
disk path with `PFM_PM_VIX_SLUG_CACHE_PATH=/some/other/path` for
multi-tenant deployments).

## Endpoints

```
POST /indices/pm-vix/refresh-slugs
GET  /indices/pm-vix/slugs
```

`POST /indices/pm-vix/refresh-slugs` synchronously runs the validate +
search + persist cycle. Returns:

```json
{
  "as_of": "2026-05-08T18:30:00+00:00",
  "n_kept": 12,
  "n_dead_replaced": 5,
  "buckets": {
    "recession": ["us-recession-by-end-of-2026", "..."],
    "..."
  }
}
```

`GET /indices/pm-vix/slugs` returns the slug map currently driving the
score, with a `source` field of `"live"`, `"fallback"`, or `"mixed"`
so the caller can tell at a glance whether the headline number is
backed by live contracts.

### Admin gating

When `PFM_ADMIN_TOKEN` is set in the environment, the refresh endpoint
requires the matching `X-Admin-Token` header. When the env var is
unset, the endpoint is open — convenient for local dev / demos but
**not** something you want in production.

## How to manually refresh

```bash
# Open mode (no admin token configured).
curl -X POST http://localhost:8000/indices/pm-vix/refresh-slugs

# Admin-gated.
curl -X POST http://localhost:8000/indices/pm-vix/refresh-slugs \
  -H "X-Admin-Token: $PFM_ADMIN_TOKEN"

# Inspect the live map.
curl http://localhost:8000/indices/pm-vix/slugs | jq
```

## Background task

To run the refresh on a 6-hour cron inside the FastAPI lifespan, set:

```
PFM_PM_VIX_AUTO_REFRESH_ENABLED=1
PFM_PM_VIX_AUTO_REFRESH_INTERVAL_S=21600   # default
```

The task is **off by default** so the existing test suite — which
spins up the FastAPI app via `TestClient` once per test module —
doesn't accumulate background tasks or fire spurious HTTP requests.

## Recovery

- **Corrupt cache file.** A garbled `/tmp/pfm_pm_vix_slugs.json` is
  detected at read time; we log a warning and fall back to the
  hardcoded map. The next successful refresh overwrites the file.
- **Gamma unreachable during refresh.** The endpoint surfaces
  `502 Polymarket Gamma unreachable: …`. The persisted map (if any)
  is left untouched, so subsequent reads continue to use the previous
  live map.
- **Search returns nothing for a bucket.** If every hardcoded slug
  for that bucket also dies, `validate_and_refresh_buckets` falls back
  to the hardcoded list bucket-by-bucket so the bucket still
  contributes a defined (if zero) sub-score rather than vanishing.
