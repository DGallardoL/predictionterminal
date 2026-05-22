# syntax=docker/dockerfile:1.7
# ---------- Stage 1: builder ----------
# NOTE: no `--platform=$BUILDPLATFORM` — that var is only set by BuildKit/buildx,
# so it breaks `docker-compose build` on the classic builder (empty platform =
# parse error). Building for the host platform is exactly what we want for a
# local `docker-compose up`; CI/Fly still build for their native/target arch.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Build context is the REPO ROOT (fly.toml lives there), so paths are api/…
COPY api/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Extra deps for the in-image cross-venue arb engine (arbstuff/arb_engine.py).
# These are imported unconditionally at the top of arb_engine.py but are NOT
# in api/requirements.txt (the API itself doesn't use requests/dotenv/crypto).
# Without them the autostarted engine subprocess crashes on import. Kept here
# rather than in requirements.txt so the API runtime deps stay minimal and the
# arb engine's footprint is explicit. (py_clob_client / web3 / websockets are
# optional in arb_engine.py via try/except and only needed for --live trading,
# which prod never runs — autostart uses test mode + --mode og.)
RUN pip install \
        "requests>=2.31,<3.0" \
        "python-dotenv>=1.0,<2.0" \
        "cryptography>=42.0"

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# Build-time git SHA. The GitHub Actions deploy workflow passes
# `--build-arg GIT_SHA=$(git rev-parse --short HEAD)` so the running
# container can tag Sentry events with the exact release. Falls back to
# "unknown" when built locally without the arg.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app api/src ./src
# Bundle the static frontend so a single machine serves the API (/*) AND the
# UI (/ui). main.py mounts it from ``…/main.py``.parent×4/"web" → ``/web``.
COPY --chown=app:app web /web

# Cross-venue arb engine (opt-in via PFM_ARB_ENGINE_AUTOSTART=1). main.py's
# lifespan computes the engine dir as ``…/main.py``.parents[3]/"arbstuff" →
# ``/arbstuff`` and spawns ``python /arbstuff/arb_engine.py --mode og`` with
# ``cwd=/arbstuff``. strategies_arb_router.py reads dashboard_state.json from
# the SAME ``/arbstuff`` (its _ARB_DIR = parents[3]/"arbstuff"), so engine
# writes and router reads agree.
#
# Only the engine + the four config JSONs it loads are baked in. Everything
# else in arbstuff/ (logs, *.bak, backups/, the 18 MB of runtime state) is
# excluded by .dockerignore. The engine creates its runtime files
# (dashboard_state.json, arb_engine.log, arb_pnl_log.json, …) at startup, so
# they must NOT be baked. ``--chown=app:app`` makes /arbstuff owned by the
# unprivileged ``app`` user so the engine subprocess can write there.
COPY --chown=app:app arbstuff/arb_engine.py /arbstuff/arb_engine.py
COPY --chown=app:app arbstuff/markets_config_reviewed.json /arbstuff/markets_config_reviewed.json
COPY --chown=app:app arbstuff/markets_config.json /arbstuff/markets_config.json
COPY --chown=app:app arbstuff/markets_config_politics.json /arbstuff/markets_config_politics.json
COPY --chown=app:app arbstuff/markets_config_discovered.json /arbstuff/markets_config_discovered.json

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Production server: gunicorn pre-forks N UvicornWorker processes. Each
# worker has its own event loop + Python interpreter; they share Redis via
# REDIS_URL so cached prewarms only run once on the lucky worker that warms
# the key first. ``--worker-tmp-dir /dev/shm`` keeps tmpfile heartbeats off
# of slow container disks. ``--max-requests`` cycles workers gradually to
# bound memory growth from long-lived in-process caches.
#
# Worker count: keep --workers = (CPU * 2) on small containers; Fly.io
# 2-vCPU shared instances generally land best at 4. Override per-deployment
# with ``GUNICORN_WORKERS`` env var.
CMD ["sh", "-c", "exec gunicorn pfm.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers ${GUNICORN_WORKERS:-4} \
  --bind 0.0.0.0:8000 \
  --max-requests 5000 \
  --max-requests-jitter 500 \
  --timeout 90 \
  --graceful-timeout 30 \
  --keep-alive 30 \
  --worker-tmp-dir /dev/shm \
  --forwarded-allow-ips '*' \
  --proxy-allow-from '*' \
  --access-logfile -"]
