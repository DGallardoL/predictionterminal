# ADR-0001: Use FastAPI for the API service

- **Status:** Accepted
- **Date:** 2026-04-30
- **Authors:** Damian Gallardo

## Context

The POC needs a small HTTP service that:

1. Exposes 4 endpoints (`/health`, `/factors`, `/fit`, `/attribution`).
2. Validates request and response bodies with strict typing — the model is
   numerical and silently coercing wrong types would produce confusing
   errors during a demo.
3. Auto-generates an OpenAPI schema, since that is an explicit course
   requirement.
4. Plays well with `pydantic` (because we already use it for settings) and
   with synchronous code (statsmodels, yfinance are sync).

## Considered alternatives

- **Flask + apispec.** Mature but the OpenAPI generation is bolt-on,
  validation is manual, and the Pydantic integration is poor.
- **Django REST Framework.** Heavy: a full ORM, admin, migrations and
  auth scaffold for a stateless POC.
- **Starlette directly.** Lower-level than FastAPI but loses dependency
  injection and Pydantic-driven schema generation.

## Decision

Use **FastAPI** (with Uvicorn ASGI server). It gives us:

- First-class Pydantic v2 request/response models with automatic 422 on
  validation errors and rich field constraints (`min_length`, `gt`, etc.).
- An OpenAPI 3.1 schema served at `/openapi.json` and a Swagger UI at
  `/docs` with zero extra code — directly satisfying the OpenAPI grading
  criterion.
- Dependency injection via `Depends(...)`, which we use to share the
  Polymarket client, settings, and cache across handlers without globals.
- A lifespan context manager that we use to construct the HTTP and Redis
  clients once at startup and tear them down cleanly on shutdown.
- Synchronous handler support via `def` (not `async def`); endpoints run on
  the threadpool and we don't pay async-tax for sync libraries.

## Consequences

- Adds Pydantic v2 and FastAPI to the dependency tree (~25 MB image
  delta), well within the docker-compose target.
- Tests use `fastapi.testclient.TestClient`, which exercises the full
  ASGI stack including lifespan hooks — so unit tests are also smoke tests.
- We're locked into Pydantic v2's slight schema-export quirks (e.g. needing
  `protected_namespaces=()` on responses that have a `model` field, which
  this project does on `FitResponse`).
