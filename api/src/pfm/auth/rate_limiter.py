"""Token-bucket-ish rate limiter backed by the auth SQLite counters.

We model two buckets per key:

- per-minute (``rate_limit_per_minute``)
- per-day (``daily_quota``; 0 ⇒ unlimited)

Each request increments both counters atomically (single sqlite write per
bucket via ``ON CONFLICT … DO UPDATE``). When either counter exceeds its
quota we refuse the request and surface ``Retry-After`` set to the lesser
of the two reset windows.

Bypassed paths
==============
``/health``, ``/health/*``, ``/embed/*``, ``/metrics``, ``/openapi.json``,
``/docs``, ``/redoc``, ``/ui/*`` skip the limiter entirely. (Static +
observability shouldn't fail because a customer is over-quota.)

``/auth/*`` is **not** bypassed — `/auth/demo-key` in particular must be
rate-limited so a single anonymous client can't drain the demo-key quota
table by spamming requests.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from pfm.auth.dependencies import _extract_key
from pfm.auth.models import (
    ANON_DAILY_QUOTA,
    ANON_RATE_PER_MIN,
    APIKey,
    Tier,
)
from pfm.auth.production import is_auth_enabled
from pfm.auth.storage import APIKeyStore, get_api_key_store

BYPASS_PREFIXES: tuple[str, ...] = (
    "/health",
    "/embed/",
    "/metrics",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/ui/",
)


def _is_bypass(path: str) -> bool:
    return path == "/health" or any(path.startswith(p) for p in BYPASS_PREFIXES)


def check_and_increment(
    api_key: str,
    tier: Tier,
    *,
    rate_limit_per_minute: int,
    daily_quota: int,
    store: APIKeyStore | None = None,
    endpoint: str = "*",
    now: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Bump the counters, return ``(ok, info)``.

    ``info`` always contains:
      ``minute_count, day_count, minute_remaining, day_remaining, reset_at, retry_after, tier``.

    The minute window resets on the next minute boundary; the day window on
    the next UTC midnight. ``daily_quota=0`` means unlimited (``day_remaining
    = -1``) and never trips the daily check.
    """
    s = store or get_api_key_store()
    t = now if now is not None else time.time()
    minute_count, day_count = s.increment(api_key, endpoint=endpoint, now=t)

    minute_reset = APIKeyStore.next_minute_reset(t)
    day_reset = APIKeyStore.next_day_reset(t)

    minute_remaining = max(rate_limit_per_minute - minute_count, 0)
    if daily_quota <= 0:
        day_remaining = -1
        day_ok = True
    else:
        day_remaining = max(daily_quota - day_count, 0)
        day_ok = day_count <= daily_quota

    minute_ok = minute_count <= rate_limit_per_minute
    ok = minute_ok and day_ok

    # Pick the tighter reset for the Retry-After header.
    if not minute_ok and not day_ok:
        reset_at = min(minute_reset, day_reset)
    elif not minute_ok:
        reset_at = minute_reset
    elif not day_ok:
        reset_at = day_reset
    else:
        reset_at = minute_reset

    retry_after = max(int(reset_at - t), 1) if not ok else 0

    return ok, {
        "minute_count": minute_count,
        "day_count": day_count,
        "minute_remaining": minute_remaining,
        "day_remaining": day_remaining,
        "minute_reset": minute_reset,
        "day_reset": day_reset,
        "reset_at": reset_at,
        "retry_after": retry_after,
        "tier": tier,
    }


# --------------------------------------------------------------------- middleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply the token bucket on every request when ``PFM_AUTH_ENABLED=1``.

    The middleware never raises — over-quota requests get a clean 429 JSON
    body with ``retry_after``. Headers ``X-RateLimit-Remaining``,
    ``X-RateLimit-Reset``, ``X-RateLimit-Tier`` are stamped on every
    non-bypassed response.
    """

    def __init__(self, app, store_factory=get_api_key_store) -> None:
        super().__init__(app)
        self._store_factory = store_factory

    async def dispatch(self, request: Request, call_next):
        if not is_auth_enabled() or _is_bypass(request.url.path):
            return await call_next(request)

        store = self._store_factory()
        key_obj: APIKey | None = None
        raw = _extract_key(request.headers.get("authorization")) or (
            request.headers.get("x-api-key")
            if (request.headers.get("x-api-key") or "").startswith("sk_pfm_")
            else None
        )
        if raw:
            key_obj = store.get_key(raw)
            if key_obj and not key_obj.enabled:
                key_obj = None

        if key_obj is None:
            # Anonymous: identify by client IP for fairness.
            client_ip = (request.client.host if request.client else "unknown") or "unknown"
            bucket_key = f"anon:{client_ip}"
            tier: Tier = "free"
            rpm = ANON_RATE_PER_MIN
            quota = ANON_DAILY_QUOTA
        else:
            bucket_key = key_obj.key
            tier = key_obj.tier
            rpm = key_obj.rate_limit_per_minute
            quota = key_obj.daily_quota

        ok, info = check_and_increment(
            bucket_key,
            tier,
            rate_limit_per_minute=rpm,
            daily_quota=quota,
            store=store,
            endpoint=request.url.path,
        )

        if not ok:
            resp = JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": info["retry_after"],
                    "tier": tier,
                    "limit_per_minute": rpm,
                    "daily_quota": quota,
                },
            )
            _stamp_headers(resp, info, tier)
            resp.headers["Retry-After"] = str(info["retry_after"])
            return resp

        response: Response = await call_next(request)
        _stamp_headers(response, info, tier)
        return response


def _stamp_headers(response: Response, info: dict[str, Any], tier: Tier) -> None:
    response.headers["X-RateLimit-Tier"] = tier
    response.headers["X-RateLimit-Remaining"] = str(info["minute_remaining"])
    response.headers["X-RateLimit-Reset"] = str(int(info["reset_at"]))
    response.headers["X-RateLimit-Daily-Remaining"] = str(info["day_remaining"])


def install(app, store_factory=get_api_key_store) -> None:
    """Attach the middleware. No-op when ``PFM_AUTH_ENABLED`` is unset.

    We always install it (so flipping the env var doesn't require a code
    change), but the dispatch fast-paths to the next handler when auth is
    off, so the overhead is one ``os.environ.get`` per request.
    """
    if os.environ.get("PFM_RATE_LIMIT_DISABLED") == "1":
        return
    app.add_middleware(RateLimitMiddleware, store_factory=store_factory)
