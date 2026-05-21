"""Detailed health endpoint exposed at `/health/detail`.

The simple `/health` (in `pfm.main`) is the fast liveness probe used by Docker
healthchecks. This router adds a richer readiness/diagnostics endpoint with
Redis connectivity, uptime, and the deployed git SHA.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any

from fastapi import APIRouter, Request

from pfm import __version__
from pfm.auth.production import (
    admin_token_configured,
    detect_env_reason,
    is_admin_token_autogen,
    is_auth_enabled,
)

router = APIRouter()

_PROCESS_START = time.time()


def _git_sha() -> str:
    """Best-effort git SHA. Reads $GIT_SHA first, then falls back to `git`."""
    sha = os.environ.get("GIT_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _redis_status(request: Request) -> dict[str, Any]:
    """Ping the cache backend if it exposes a redis client. Never raises."""
    cache = getattr(request.app.state, "cache", None)
    client = getattr(cache, "_client", None) or getattr(cache, "client", None)
    if client is None:
        return {"connected": False, "latency_ms": None}
    try:
        start = time.perf_counter()
        client.ping()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"connected": True, "latency_ms": latency_ms}
    except Exception:
        return {"connected": False, "latency_ms": None}


def _auth_status() -> dict[str, Any]:
    """Public snapshot of the auth posture — never includes the token itself."""
    return {
        "enabled": is_auth_enabled(),
        "autogen_token_in_use": is_admin_token_autogen(),
        "admin_token_configured": admin_token_configured(),
        "env_detection": detect_env_reason(),
    }


@router.get("/health/detail", tags=["health"])
def health_detail(request: Request) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "redis": _redis_status(request),
        "uptime_seconds": round(time.time() - _PROCESS_START, 2),
        "git_sha": _git_sha(),
        "auth_status": _auth_status(),
    }
