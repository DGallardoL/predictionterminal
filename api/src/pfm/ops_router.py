"""``GET /ops/sessions`` and ``GET /ops/config`` — operations endpoints.

Two read-only endpoints that surface operational state for on-call
engineers and the project's "multi-session coordination" workflow:

* ``/ops/sessions`` — lists currently-unexpired entries from
  ``.coordination/active-edits.json``. Up to ~60 concurrent Claude
  Code sub-agents can be in-flight against this repo at any time and
  the JSON file is the source of truth for who's writing what. This
  endpoint surfaces the same data over HTTP so dashboards can render
  it without needing filesystem access.
* ``/ops/config`` — non-secret runtime configuration: relevant
  ``PFM_*`` env vars (with secrets masked), Redis URL (password masked),
  worker count, factor count, OpenAPI path count, process uptime, and
  the cache-stats snapshot from :mod:`pfm.metrics` (T16) when available.

Secret-masking rules
--------------------

* Any environment variable name matching the regex
  ``(PASSWORD|TOKEN|SECRET|KEY)`` is masked entirely (``"***"``).
* URLs of the form ``scheme://user:password@host[/...]`` have their
  password component rewritten to ``***`` while preserving everything
  else. This is applied to ``REDIS_URL`` and any value that looks
  url-shaped.

Integration note (when ``main.py:routes`` is unclaimed)::

    from pfm.ops_router import router as _ops_router
    app.include_router(_ops_router)

This module is intentionally tolerant — missing state attributes,
unreadable files, and malformed JSON degrade to empty / null fields
rather than 5xx-ing. Operations endpoints must not panic.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter()


# Module-load timestamp for the ``uptime_s`` reading. Close enough to
# the gunicorn worker boot time that ops dashboards treat the two as
# interchangeable; same approach as :mod:`pfm.health_router`.
_PROCESS_START = time.time()


# ---------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------

# ``ops_router.py`` lives at ``api/src/pfm/ops_router.py``; the
# coordination file lives at ``<repo_root>/.coordination/active-edits.json``.
# Resolve relative to this file so the lookup works from any CWD (gunicorn
# under /opt/app, pytest from api/, etc.).
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent.parent  # api/src/pfm/ → repo/
_DEFAULT_ACTIVE_EDITS_PATH = _REPO_ROOT / ".coordination" / "active-edits.json"


def _active_edits_path() -> Path:
    """Resolve the path to ``active-edits.json``.

    Test override via ``PFM_OPS_ACTIVE_EDITS_PATH`` so unit tests can
    point at a fixture without touching the real coordination file.
    """
    override = os.environ.get("PFM_OPS_ACTIVE_EDITS_PATH")
    if override:
        return Path(override)
    return _DEFAULT_ACTIVE_EDITS_PATH


# ---------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------

# Env var names matching this regex are masked entirely. Case-insensitive.
_SECRET_NAME_PATTERN = re.compile(r"(PASSWORD|TOKEN|SECRET|KEY)", re.IGNORECASE)

# Mask the password component of ``scheme://user:password@host[/...]`` URLs.
# Constructed to be permissive: any character that isn't ``@`` or ``/`` is
# valid inside the password segment (handles ``+``, ``-``, ``=``, ``%``).
_URL_PASSWORD_PATTERN = re.compile(
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+\-.]*://)"
    r"(?P<user>[^:@/\s]+)"
    r":(?P<pw>[^@/\s]+)"
    r"@"
)


def _mask_url_password(url: str) -> str:
    """Replace ``:password@`` with ``:***@`` in a URL.

    Idempotent (running it twice yields the same string) and a no-op
    when the URL has no embedded password.
    """
    if not isinstance(url, str) or "@" not in url:
        return url
    return _URL_PASSWORD_PATTERN.sub(
        lambda m: f"{m.group('scheme')}{m.group('user')}:***@",
        url,
    )


def _mask_env_value(name: str, value: str) -> str:
    """Apply the masking rules to a single env var.

    * Name matches the secret-name regex → ``"***"``
    * Value is URL-shaped → mask the password segment
    * Otherwise → return unchanged
    """
    if _SECRET_NAME_PATTERN.search(name):
        return "***"
    return _mask_url_password(value)


# Env vars whitelisted for the ``/ops/config`` "env" block. Anything
# starting with ``PFM_`` is included by definition; we explicitly add
# a few well-known infra vars (``REDIS_URL``, ``ENV``, ``GIT_SHA``)
# that ops cares about even though they don't carry the prefix.
_EXTRA_ENV_KEYS = ("REDIS_URL", "ENV", "GIT_SHA")


def _collect_env() -> dict[str, str]:
    """Snapshot env vars relevant for ops, with masking applied."""
    out: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith("PFM_") or key in _EXTRA_ENV_KEYS:
            out[key] = _mask_env_value(key, value)
    return out


# ---------------------------------------------------------------------
# Active-sessions helpers
# ---------------------------------------------------------------------


def _parse_iso_utc(s: str) -> datetime | None:
    """Lenient ISO8601 parser. Returns ``None`` if the string isn't valid.

    Accepts both ``...Z`` and ``...+00:00`` suffixes, plus optional
    fractional seconds (the coordination file uses both forms).
    """
    if not isinstance(s, str):
        return None
    raw = s.strip().replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    # Naive datetimes are treated as UTC — the schema requires UTC, and
    # writers that forget the suffix shouldn't be silently treated as
    # local time.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _load_active_sessions(now: datetime | None = None) -> list[dict[str, Any]]:
    """Read the coordination file and return unexpired entries.

    Returns an empty list (never raises) when:

    * the file does not exist
    * the file isn't valid JSON
    * the file isn't a JSON array

    Each returned entry preserves the original schema and adds nothing —
    callers get exactly what's on disk, modulo the expiry filter.
    """
    path = _active_edits_path()
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("ops: cannot read active-edits.json: %s", exc)
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("ops: active-edits.json is not valid JSON: %s", exc)
        return []
    if not isinstance(data, list):
        return []

    now = now or datetime.now(UTC)
    active: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        # Sessions that explicitly mark themselves COMPLETED are filtered
        # out even if their TTL hasn't elapsed yet — the coordination
        # protocol uses both signals interchangeably (see
        # ``.coordination/PROTOCOL-V2.md``).
        status = str(entry.get("status", "")).upper()
        if status == "COMPLETED":
            continue
        expires_at = _parse_iso_utc(entry.get("expires_at", ""))
        if expires_at is None:
            # Conservatively treat unparseable timestamps as active — we'd
            # rather show a stale-looking entry than silently hide it from
            # ops dashboards.
            active.append(entry)
            continue
        if expires_at > now:
            active.append(entry)
    return active


# ---------------------------------------------------------------------
# /ops/sessions
# ---------------------------------------------------------------------


@router.get(
    "/ops/sessions",
    summary="List unexpired multi-session coordination claims",
    tags=["ops"],
)
def ops_sessions() -> dict[str, Any]:
    """Return the unexpired claims from ``.coordination/active-edits.json``.

    Response shape::

        {
            "active_sessions": [
                {
                    "session_id": "T27-ops-router-...",
                    "scope": "T27-ops-router",
                    "files": ["api/src/pfm/ops_router.py", ...],
                    "expires_at": "2026-05-16T16:00:00Z",
                    ...
                },
                ...
            ],
            "count": 1,
            "checked_at": "2026-05-16T15:45:00Z"
        }
    """
    sessions = _load_active_sessions()
    return {
        "active_sessions": sessions,
        "count": len(sessions),
        "checked_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


# ---------------------------------------------------------------------
# /ops/config
# ---------------------------------------------------------------------


def _worker_count() -> int:
    """Best-effort gunicorn worker count.

    Reads ``WEB_CONCURRENCY`` (the conventional 12-factor knob) first,
    then ``GUNICORN_WORKERS``, then falls back to ``1`` (the TestClient
    case and any deployment that doesn't set the var).
    """
    for key in ("WEB_CONCURRENCY", "GUNICORN_WORKERS"):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            continue
    return 1


def _factor_count(request: Request) -> int:
    """Length of ``app.state.factors``; ``0`` when state isn't initialised."""
    factors = getattr(request.app.state, "factors", None)
    try:
        return len(factors) if factors is not None else 0
    except TypeError:
        return 0


def _openapi_path_count(request: Request) -> int:
    """Number of OpenAPI paths the running app exposes.

    Calling ``app.openapi()`` is cached by FastAPI after the first call,
    so this is cheap on the warm path. We swallow any error during the
    initial build so the ops endpoint never 500s.
    """
    try:
        spec = request.app.openapi()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ops: openapi build failed: %s", exc)
        return 0
    paths = (spec or {}).get("paths", {}) if isinstance(spec, dict) else {}
    return len(paths) if isinstance(paths, dict) else 0


def _cache_stats() -> dict[str, Any] | None:
    """Pull the per-endpoint latency snapshot from :mod:`pfm.metrics`.

    Returns ``None`` when the T16 metrics module is unavailable or
    has no observations yet; this keeps the response shape stable
    while leaving room for richer aggregation later.
    """
    try:
        from pfm.metrics import get_tracker
    except ImportError:
        return None
    try:
        tracker = get_tracker()
        endpoints = tracker.snapshot()
        return {
            "endpoints_tracked": len(endpoints) if isinstance(endpoints, dict) else 0,
            "total_requests": tracker.total_requests(),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ops: cache_stats fetch failed: %s", exc)
        return None


def _redis_url_masked(request: Request) -> str | None:
    """Pull the configured Redis URL (with password masked) from
    settings, env, or app state — whichever resolves first.
    """
    settings = getattr(request.app.state, "settings", None)
    candidate = getattr(settings, "redis_url", None) if settings else None
    if not candidate:
        candidate = os.environ.get("REDIS_URL")
    if not candidate:
        return None
    return _mask_url_password(candidate)


@router.get(
    "/ops/config",
    summary="Non-secret runtime configuration snapshot",
    tags=["ops"],
)
def ops_config(request: Request) -> dict[str, Any]:
    """Return a snapshot of relevant runtime config with secrets masked.

    Response shape::

        {
            "env": { "PFM_ENV": "dev", "REDIS_URL": "redis://***@localhost:6379/0", ... },
            "runtime": { "workers": 4, "uptime_s": 12345.6, "factor_count": 1360 },
            "openapi": { "path_count": 271 },
            "cache_stats": { ... } | null
        }
    """
    env = _collect_env()
    # Ensure REDIS_URL appears with its password masked even if the
    # actual env var isn't named ``REDIS_URL`` (e.g. an older deployment
    # that wired it through ``settings.redis_url`` from a ``.env`` file).
    redis_url = _redis_url_masked(request)
    if redis_url and "REDIS_URL" not in env:
        env["REDIS_URL"] = redis_url

    return {
        "env": env,
        "runtime": {
            "workers": _worker_count(),
            "uptime_s": round(time.time() - _PROCESS_START, 2),
            "factor_count": _factor_count(request),
        },
        "openapi": {
            "path_count": _openapi_path_count(request),
        },
        "cache_stats": _cache_stats(),
    }
