# ADR-0005: No persistent database in the POC

- **Status:** Accepted
- **Date:** 2026-04-30
- **Authors:** Damian Gallardo

## Context

A natural extension would be to persist fitted models, named "saved
regressions", per-user history, comparison runs, etc. Each of those
implies a real database (Postgres being the obvious choice).

The course requirements explicitly say *Docker, OpenAPI, CI, ADRs*; they
do **not** require persistence. The grading criterion is engineering
quality, not feature breadth. The POC's value proposition is the
end-to-end flow from a Polymarket slug to a HAC-corrected coefficient.

Adding a database costs:

- A migration story (Alembic or hand-rolled).
- A 4th container in `docker-compose.yml` and a 4th healthcheck.
- An ORM dependency (SQLAlchemy) and 1–2k LoC of model/repo plumbing.
- A whole extra dimension of failure modes during demo (connection
  pools, schema drift between dev and CI, container start order).

The user-visible benefit: zero, for the POC.

## Considered alternatives

- **Postgres + SQLAlchemy.** Full-featured but heavy. Detailed above.
- **SQLite on a bind-mounted volume.** Lightweight but solves only "save a
  fit", which the POC does not need.
- **Redis as a quasi-database.** Considered for cache only. Treating Redis
  as the source-of-truth for model artefacts would be a mis-use. We have
  cache there (ADR-0004) and stop.

## Decision

The POC has **no persistent database**. State is:

- **Configuration** in YAML (`factors.yml`) — version-controlled.
- **Cache** in Redis (TTL 1 h) — explicitly ephemeral.
- **No fit history.** Each `/fit` call is independent.

If a user wants to keep a fit, they save the JSON response themselves.

## Consequences

- The `docker-compose.yml` stays at three services (`api`, `web`, `redis`).
- Tests don't need a database fixture or migration step.
- The CI pipeline doesn't have to spin up Postgres.
- `factors.yml` is the only persistent artefact; updating factors means a
  config change + restart, which is fine for the demo cadence.
- Adding persistence later is a clean delta: introduce a `db` service,
  introduce a `repository.py` layer, hide `model.py` behind it. The
  current architecture does not preclude this; it just doesn't pay for it.
