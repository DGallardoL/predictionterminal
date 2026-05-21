# DEVELOPMENT.md — Contributor Guide

> **Audience.** Anyone (human or Claude Code sub-agent) extending the
> Prediction Terminal monorepo. If you are about to touch code, read this
> end-to-end at least once. The conventions below are not aesthetic — they
> are how we keep ~2700 tests green, 271 OpenAPI paths in sync, and up to
> 60 concurrent agents from clobbering each other's work.

This document complements `CLAUDE.md` (high-level rules of engagement),
`PLAN.md` (product specification), `docs/architecture.md` (system layout),
and `.coordination/PROTOCOL-V2.md` (race-condition discipline). When
those disagree with this file, the more specific document wins; flag the
drift in `.coordination/issues.log` so the next contributor can reconcile.

---

## 1. Setup

The repository ships with two coordinated runtimes: a Python FastAPI
backend in `api/` and a static frontend served from `web/`. For demo and
CI we use `docker-compose`; for day-to-day development a local virtualenv
gives the fastest feedback loop.

### 1.1 One-command bring-up (Docker)

```bash
docker-compose up --build
```

This builds three services — `api` (uvicorn on `:8000`), `web` (nginx on
`:8080`), and `redis` (cache on `:6379`) — and runs healthchecks until
all are green. The compose file mounts `api/src/` and `web/` as volumes,
so edits picked up on the host are reflected without rebuilding.

Smoke-test the stack:

```bash
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8000/factors | jq '.factors | length'
open http://localhost:8080
```

### 1.2 Local Python environment

For a faster inner loop (no container rebuilds, native debugger):

```bash
cd api
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `[dev]` extras pull in `pytest`, `pytest-cov`, `pytest-asyncio`,
`respx`, `httpx_mock`, `ruff`, `mypy`, and the documentation toolchain.
Run the API directly:

```bash
PYTHONPATH=src uvicorn pfm.main:app --reload --port 8000
```

For the frontend, any static file server works; the simplest is:

```bash
cd web && python3 -m http.server 8080
```

### 1.3 Optional services

- **Redis.** The cache layer degrades gracefully if Redis is absent
  (everything falls back to an in-process LRU), but for parity with
  production set `REDIS_URL=redis://localhost:6379/0`.
- **Polymarket / yfinance.** All upstream calls are mocked in tests.
  For ad-hoc exploration set `PFM_ALLOW_LIVE_FETCH=1`; never enable this
  in CI.

---

## 2. Running tests

`pytest` is the single test driver. The full suite is roughly 2700 cases
and finishes in ~80 seconds on an M-series laptop. Most contributors
should run a scoped subset during development and the full suite before
opening a PR.

### 2.1 The basics

```bash
cd api
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Speed tips:

- `-x` — stop on first failure.
- `-k <expr>` — filter by name (`-k "strategies and arb"`).
- `--lf` — re-run only the last failures.
- `-n auto` (via `pytest-xdist`) — run in parallel; useful for the full
  suite, occasionally flaky for fixtures that race on Redis.

### 2.2 Coverage

```bash
PYTHONPATH=src .venv/bin/python -m pytest --cov=pfm --cov-report=term-missing
```

The contractual coverage floor is **70 %** on `pfm.model` and
`pfm.attribution`. CI will fail if either dips below. Aim higher on new
modules — most current routers sit at 85–95 %.

### 2.3 The `slow` marker

Stress tests, 4-quarter robustness checks, and Monte-Carlo calibrations
are decorated with `@pytest.mark.slow`. They are skipped by default in
the pre-commit smoke and in the inner-loop runs:

```bash
# Default (skips slow):
pytest -m "not slow"

# Slow only:
pytest -m slow

# Everything:
pytest
```

Mark a test as slow when its wall-clock cost exceeds ~2 seconds *or* it
depends on long synthetic-DGP simulations. Slow tests still run in CI
via the dedicated `slow` job.

---

## 3. Code style

We standardise on **ruff** for both formatting and linting; there is no
separate `black` or `isort`. The single source of truth is
`api/pyproject.toml` — see `[tool.ruff]` where `line-length = 100` is
declared (overriding ruff's 88-char default).

### 3.1 Day-to-day workflow

```bash
ruff format .       # apply formatter
ruff check . --fix  # autofix lints
ruff check .        # strict pass; must be clean before commit
```

### 3.2 Conventions

- **Type hints everywhere.** Python 3.12 style: `list[str]`, `dict[str,
  int]`, `X | None`, never `typing.List` / `Optional`.
- **Pydantic v2** for all request/response schemas. Models live in
  `api/src/pfm/schemas.py` and must be **appended to the end** — see
  PROTOCOL-V2 for why.
- **Google-style docstrings** on public functions. One-liners are fine
  for private helpers.
- **Imports** are absolute, sorted by ruff's isort rules. Never inline
  `import` statements inside functions unless dodging a circular import.
- **Errors.** Routers raise `fastapi.HTTPException` with a meaningful
  `detail`. The domain layer raises plain exceptions; the router maps
  them at the boundary.
- **Logging.** Prefer `structlog` (already configured in
  `pfm.logging_setup`) for structured fields; vanilla `logging` is
  acceptable when context is purely local. **Never** `print()` —
  pre-commit will not catch this, but reviewers will.

---

## 4. Pre-commit hooks (W11-50)

Hooks live in `.pre-commit-config.yaml` at the repo root and were
hardened by task W11-50 on 2026-05-16.

### 4.1 Install once per clone

```bash
pre-commit install
```

### 4.2 What runs on every `git commit`

1. **ruff-format** — autoformat the staged Python.
2. **ruff --fix** — autofix safe lints.
3. **ruff (strict)** — fails if anything is still dirty.
4. **end-of-file-fixer** / **trailing-whitespace** — POSIX hygiene.
5. **check-merge-conflict** — bails on `<<<<<<<` markers.
6. **check-yaml** / **check-json** — syntactic validation; this catches
   broken `active-edits.json` claims before they hit the ledger.
7. **check-added-large-files** — `>1 MB` is rejected, with
   `web/index.html` excluded because it is intentionally ~1.6 MB.
8. **check-jsonschema** — validates `web/data/alpha_strategies.json`
   against its schema when present.
9. **validate-active-edits** — runs
   `api/scripts/validate_active_edits.py` to confirm no expired or
   malformed claim slipped in.
10. **pytest-smoke** — a ~5 s targeted run of `test_health_deep.py` and
    `test_factors_yml_schema.py`. Only triggers on `.py` or
    `factors.yml` changes.

### 4.3 Escape hatches

- Skip the slow smoke: `SKIP=pytest-smoke git commit ...`
- Skip everything (last resort): `git commit --no-verify`. Justify the
  bypass in the PR description.

Run the whole stack manually before opening a PR:

```bash
pre-commit run --all-files
```

---

## 5. Adding a new endpoint

The canonical pattern is a **standalone router module** mounted from
`pfm/main.py`. Do not add new endpoint functions directly inside
`main.py`.

### 5.1 Create the router

```python
# api/src/pfm/<feature>_router.py
from fastapi import APIRouter, Depends, HTTPException

from pfm.dependencies import get_cache, get_logger
from pfm.schemas import MyFeatureResponse  # appended at end of schemas.py

router = APIRouter(prefix="/<feature>", tags=["<feature>"])


@router.get("/", response_model=MyFeatureResponse)
async def list_things(cache=Depends(get_cache)) -> MyFeatureResponse:
    cached = await cache.get("feature:list")
    if cached:
        return cached
    ...
```

Shared dependencies (Redis client, logger, settings, rate limiter) live
in `pfm/dependencies.py`. Reuse them; do not re-instantiate.

### 5.2 Wire it up via `main.py:routes`

`api/src/pfm/main.py` is partitioned by PROTOCOL-V2 into sections —
`lifespan`, `cors`, `routes`, `exception-handlers`. Only the holder of
the `main.py:routes` claim may add an `app.include_router(...)` line.
If you are the endpoint author *and* the routes coordinator, do both;
otherwise file a wire-up request in
`.coordination/main-py-wire-up.md`.

### 5.3 Tests and OpenAPI

- Add `api/tests/test_<feature>_router.py` with at least one happy-path
  case and one error case. Mock upstreams with `respx` /
  `httpx_mock`.
- Regenerate the OpenAPI snapshot:
  ```bash
  python api/scripts/dump_openapi.py > docs/openapi.json
  ```
  CI compares this against `app.openapi()`; drift fails the build.
- Bump the endpoint count in `CLAUDE.md` → "Current state" if your
  addition crosses a group boundary.

---

## 6. Adding a new factor

Factors are declared in `api/src/pfm/factors.yml` (1228+ entries). Do
**not** rewrite the file end-to-end; corruption risk is real. The flow
below avoids whole-file rewrites entirely for wave additions.

### 6.1 Schema

Each entry has `slug`, `source`, `description`, `created`, plus
source-specific fields (e.g. `polymarket_slug`, `clob_token_ids`). The
authoritative schema lives in `pfm/factor_schema.py` and is enforced by
`tests/test_factors_yml_schema.py`.

### 6.2 Wave pattern

1. Branch `wave-N-<theme>` (per `CLAUDE.md` — "Expand the factor
   catalog").
2. **For small additions (≤5 slugs).** Append carefully under the
   appropriate section; run `python api/scripts/validate_factors.py`.
3. **For wave-scale additions.** Create
   `api/src/pfm/factors_wave<N>.yml`, then ask Damian to merge — never
   batch-mutate `factors.yml` from a sub-agent.
4. Verify each slug resolves and yields ≥30 daily observations.
5. Add no-network tests using cached fixtures from
   `api/tests/fixtures/factors/`.
6. Bump the totals in `CLAUDE.md` → "Scale".

---

## 7. Adding a new strategy

Strategies live under `api/src/pfm/strategies/` and implement the
`Strategy` protocol declared in `pfm/strategies/base.py`
(`signal()`, `position()`, `pnl()`).

### 7.1 Skeleton

```python
# api/src/pfm/strategies/<name>.py
from pfm.strategies.base import Strategy, StrategySignal


class MyStrategy(Strategy):
    name = "my_strategy"

    def signal(self, df) -> StrategySignal:
        ...

    def position(self, signal: StrategySignal) -> float:
        ...

    def pnl(self, positions, returns):
        ...
```

Register the class in `pfm/strategies/registry.py`. Add a synthetic-DGP
test in `api/tests/strategies/test_<name>.py` that recovers a known
signal-to-PnL relationship; this is the same discipline as the model
tests in §2 of `CLAUDE.md`.

### 7.2 Four-quarter stress test

Before tagging a strategy as **deployable**:

```bash
python api/scripts/robustness_check.py --strategy <name> --quarters 4
```

Acceptance criteria (per `CLAUDE.md` and the anti-alpha rule, ADR-0013):

- Sharpe ≥ 0.5 in **every** quarter.
- No sign flip versus the full-sample backtest.
- Transaction-cost sensitivity ≤ 30 % of gross PnL.
- BH-FDR-adjusted p < 0.05 and deflated Sharpe > 0.

If any quarter fails, file the strategy under `docs/alpha-reports/`
graveyard and add it to the anti-alphas section. Do not lobby for
re-promotion without new orthogonal evidence (cf. memory note
"Wave-5 stress tests killed 6 of 8 A_GOLD claims").

---

## 8. Adding a new ADR

Architecture Decision Records live in `docs/adrs/`. The current set
runs `0001-use-fastapi.md` through `0018-frontend-bundle-strategy.md`.

### 8.1 Template

```markdown
# ADR-NNNN: <Short imperative title>

- **Status:** Proposed | Accepted | Superseded by ADR-XXXX
- **Date:** YYYY-MM-DD
- **Deciders:** Damian, <agent-id if relevant>

## Context

What problem are we solving? What constraints apply? Cite prior ADRs.

## Decision

The choice we are making, stated as a single sentence followed by
detail.

## Consequences

Positive, negative, and any follow-up work or migration cost.

## Alternatives considered

Bullets, each with one-line rejection rationale.

## References

Links to PRs, issues, related ADRs, external docs.
```

### 8.2 Numbering and references

- Pick the **next free integer**. Do not reuse numbers from superseded
  ADRs; mark them `Superseded by ADR-XXXX` instead.
- Use the `NNNN-kebab-title.md` filename pattern (all lowercase, no
  `ADR-` prefix — the legacy uppercase `ADR-NNNN-…` files were renamed
  on 2026-05-19 to harmonise the catalogue).
- Cross-reference from any code or doc that depends on the decision —
  e.g. `pfm/cache.py` cites ADR-0004, ADR-0011, and ADR-0014.
- Genuine ADRs only. Each must be ≥150 words and describe a real
  trade-off (CLAUDE.md grading criterion).

---

## 9. Multi-session coordination

This repo is edited concurrently by **up to 60 Claude Code sub-agents +
5 human-coordinated sessions**. The only thing standing between us and
silent data loss is `.coordination/PROTOCOL-V2.md`. **Read it before any
write.**

Cross-reference: **ADR-0007 — Multi-Session Coordination**.

The non-negotiables:

1. **Read** `.coordination/active-edits.json` and look for unexpired
   claims on files you intend to touch.
2. **Append** your claim to that JSON array — never `Write` the file
   with only your entry. The clobbering incident on 2026-05-16 was
   caused exactly by this.
3. **Single-owner files.** `web/index.html`, `web/config.js`,
   `api/src/pfm/main.py`, and `api/src/pfm/schemas.py` have section-level
   ownership; pivot if claimed.
4. **Append-only growth.** New CSS goes into `web/css/<feature>.css`,
   new JS into `web/js/<feature>.js`. The `index-html-owner` mounts
   them.
5. **Release** your claim when done (set `expires_at` to the past or
   delete your entry).

On conflict, stop and log to `.coordination/issues.log`. On failure,
log to `.coordination/outcomes.log` and abandon the approach.

---

## 10. Debugging

### 10.1 Structured logs

`pfm.logging_setup.configure()` wires `structlog` with JSON output in
production and a pretty console renderer in dev. Bind request-scoped
fields with `log = log.bind(request_id=..., factor=...)`. Logs flow to
stdout; the Docker compose stack tails them in `docker-compose logs -f
api`.

### 10.2 `/metrics/audit`

`GET /metrics/audit` returns a snapshot of cache hit/miss rates, rate
limiter buckets, upstream call counts, and the last 50 slow requests.
It is the first place to look when a user reports "the dashboard is
slow."

### 10.3 `/health/deep`

`GET /health/deep` does what `/health` does plus exercises Redis,
verifies that `factors.yml` loads, and round-trips a synthetic OLS fit.
Used by docker-compose healthchecks and the pre-commit smoke. A green
`/health/deep` is the minimum proof your change did not break import
or wiring.

### 10.4 Useful one-liners

```bash
# Tail just the audit logger
docker-compose logs -f api | jq 'select(.logger=="pfm.audit")'

# Profile a slow endpoint
PYTHONPATH=src python -X importtime -m pfm.main 2> import-time.log

# Reproduce a failing test with verbose output
pytest tests/test_<x>.py::TestY::test_z -vv -s
```

---

## 11. CI workflow

`.github/workflows/ci.yml` runs on every push and PR. The pipeline
mirrors the local pre-commit + test invariants:

1. **Lint** — `ruff format --check .` and `ruff check .`.
2. **Tests (fast)** — `pytest -m "not slow" --cov=pfm` with the 70 %
   floor on `pfm.model` and `pfm.attribution`.
3. **Tests (slow)** — `pytest -m slow` on a separate job so a flaky
   Monte-Carlo run does not block the fast feedback loop.
4. **OpenAPI snapshot** — runs `api/scripts/dump_openapi.py` and
   compares against the committed `docs/openapi.json`. Drift fails the
   build; regenerate locally and commit.
5. **Docker build** — `docker-compose build` to keep the demo image
   honest.
6. **Healthcheck** — `docker-compose up -d` followed by `curl
   /health/deep` to catch wiring regressions that pass unit tests but
   break the live process.

Green CI is the gate. There are no overrides for non-emergency work.

---

## 12. Release process

Releases are lightweight — we are not yet running multi-tenant
production — but the discipline matters for the demo and for the
grading rubric.

### 12.1 Changelog

`CHANGELOG.md` follows the *Keep a Changelog* convention. For every
user-visible change add a line under `## [Unreleased]` in the
appropriate subsection (`Added`, `Changed`, `Fixed`, `Removed`,
`Security`). Wave-scale additions to `factors.yml` collapse to one
line ("Added 142 factors under wave-9: commodities and rates").

### 12.2 Cutting a release

```bash
# 1. Bump the Unreleased heading to vX.Y.Z dated today.
# 2. Update the version in api/src/pfm/__init__.py.
# 3. Commit with message "release: vX.Y.Z".
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin main --tags
```

The tag push triggers `.github/workflows/deploy.yml`, which:

1. Re-runs the full CI matrix on the tagged commit.
2. Builds and pushes the `api` and `web` Docker images.
3. Uploads the `docs/` tree and `docs/openapi.json` as workflow
   artefacts.
4. Posts a summary comment with the diff of endpoint counts, test
   counts, and factor totals versus the prior tag.

### 12.3 Hotfixes

For an urgent fix on a deployed tag:

1. Branch from the tag: `git checkout -b hotfix/<issue> vX.Y.Z`.
2. Apply the minimal change plus a regression test.
3. Bump to `vX.Y.Z+1`, repeat §12.2.
4. Merge the hotfix branch back to `main` to avoid divergence.

---

## Appendix: When in doubt

Ask Damian. Do not guess at upstream API behaviour — if you are
unsure, mock it and leave a `# TODO: verify with live call` comment.
Do not auto-commit or push. Do not silently widen scope; if the work
balloons, surface it on the `TASK-BOARD.md` and split the task.

Welcome aboard.
