"""FastAPI router for API-key management + usage analytics.

Endpoints (tagged ``auth``)::

    POST   /auth/keys                  -> create key (admin via X-Admin-Token)
    GET    /auth/keys/me               -> current key info
    GET    /auth/keys/me/usage         -> per-key usage stats
    DELETE /auth/keys/{key}            -> revoke a key (admin)
    POST   /auth/demo-key              -> mint a 24h Free demo key (open)
    GET    /auth/usage/dashboard       -> aggregated org-wide usage (admin)

The router only owns the persistence + introspection surface. The actual
gating of *other* endpoints is enforced by:

- :class:`pfm.auth.rate_limiter.RateLimitMiddleware` (per-request)
- :func:`pfm.auth.dependencies.require_tier` (per-endpoint)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from pfm.auth.dependencies import require_admin, require_api_key
from pfm.auth.models import (
    TIER_DEFAULTS,
    APIKey,
    APIKeyCreateRequest,
    APIKeyPublic,
    UsageStats,
)
from pfm.auth.production import (
    first_boot_done,
    get_or_generate_admin_token,
    is_auth_enabled,
    mark_first_boot_done,
)
from pfm.auth.storage import APIKeyStore, get_api_key_store

router = APIRouter(prefix="/auth", tags=["auth"])

DEMO_KEY_DAILY_CAP_PER_IP = 5


def _client_ip(request: Request) -> str:
    """Best-effort client IP. Honors ``X-Forwarded-For`` first hop."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip() or "unknown"
    return (request.client.host if request.client else "unknown") or "unknown"


# ---------------------------------------------------------------------- keys


@router.post(
    "/keys",
    summary="Create a new API key (admin only)",
    dependencies=[Depends(require_admin)],
)
async def create_key(
    body: APIKeyCreateRequest,
    store: APIKeyStore = Depends(get_api_key_store),
) -> dict:
    """Mint a new key. The plaintext is returned exactly once."""
    key = APIKey.new(user_id=body.user_id, tier=body.tier)
    store.save_key(key)
    return {
        "key": key.key,  # plaintext shown ONLY here
        "user_id": key.user_id,
        "tier": key.tier,
        "rate_limit_per_minute": key.rate_limit_per_minute,
        "daily_quota": key.daily_quota,
        "created_at": key.created_at.isoformat(),
        "warning": "Store this key now — it cannot be retrieved again.",
    }


@router.get(
    "/keys/me",
    response_model=APIKeyPublic,
    summary="Inspect the API key in use on this request",
)
async def get_my_key(key: APIKey = Depends(require_api_key)) -> APIKeyPublic:
    return APIKeyPublic(
        key_masked=key.masked(),
        user_id=key.user_id,
        tier=key.tier,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        enabled=key.enabled,
        rate_limit_per_minute=key.rate_limit_per_minute,
        daily_quota=key.daily_quota,
    )


@router.get(
    "/keys/me/usage",
    response_model=UsageStats,
    summary="Usage stats for the API key in use",
)
async def get_my_usage(
    key: APIKey = Depends(require_api_key),
    store: APIKeyStore = Depends(get_api_key_store),
) -> UsageStats:
    minute, today = store.get_counts(key.key, endpoint="*")
    if key.daily_quota <= 0:
        daily_remaining = -1
    else:
        daily_remaining = max(key.daily_quota - today, 0)
    return UsageStats(
        user_id=key.user_id,
        tier=key.tier,
        requests_this_minute=minute,
        requests_today=today,
        rate_limit_per_minute=key.rate_limit_per_minute,
        daily_quota=key.daily_quota,
        daily_remaining=daily_remaining,
        minute_remaining=max(key.rate_limit_per_minute - minute, 0),
    )


@router.delete(
    "/keys/{key}",
    summary="Revoke a key (admin only)",
    dependencies=[Depends(require_admin)],
)
async def revoke_key_endpoint(
    key: str,
    store: APIKeyStore = Depends(get_api_key_store),
) -> dict:
    if not key.startswith("sk_pfm_"):
        raise HTTPException(status_code=400, detail="Not an API key.")
    ok = store.revoke_key(key)
    if not ok:
        raise HTTPException(status_code=404, detail="Key not found.")
    return {"key": key[:11] + "…", "revoked": True}


# ----------------------------------------------------------------- demo / open


@router.post(
    "/demo-key",
    summary="Mint a 24h Free-tier demo key (open, no admin token required)",
)
async def mint_demo_key(
    request: Request,
    store: APIKeyStore = Depends(get_api_key_store),
) -> dict:
    """Hands out a short-lived Free key for in-browser demos.

    Capped to ``DEMO_KEY_DAILY_CAP_PER_IP`` (5) issuances per client IP per
    UTC day. Excess requests get HTTP 429 with ``Retry-After`` set to the
    seconds until the next UTC midnight. The surrounding
    :class:`RateLimitMiddleware` further bounds abuse.
    """
    ip = _client_ip(request)
    current = store.get_demo_quota_count(ip)
    if current >= DEMO_KEY_DAILY_CAP_PER_IP:
        retry_after = max(int(APIKeyStore.next_day_reset() - time.time()), 1)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Demo-key daily cap reached "
                f"({DEMO_KEY_DAILY_CAP_PER_IP}/day per IP). "
                f"Try again after UTC midnight."
            ),
            headers={"Retry-After": str(retry_after)},
        )
    key = APIKey.new(user_id="demo", tier="free")
    expires_at = time.time() + 24 * 60 * 60
    store.save_key(key, expires_at=expires_at)
    new_count = store.increment_demo_quota(ip)
    return {
        "key": key.key,
        "tier": key.tier,
        "expires_at": datetime.fromtimestamp(expires_at, tz=UTC).isoformat(),
        "rate_limit_per_minute": key.rate_limit_per_minute,
        "daily_quota": key.daily_quota,
        "demo_keys_issued_today": new_count,
        "demo_keys_remaining_today": max(DEMO_KEY_DAILY_CAP_PER_IP - new_count, 0),
    }


# ----------------------------------------------------------------- analytics


@router.get(
    "/usage/dashboard",
    summary="Aggregated org-wide usage (admin only)",
    dependencies=[Depends(require_admin)],
)
async def usage_dashboard(
    store: APIKeyStore = Depends(get_api_key_store),
) -> dict:
    agg = store.aggregate()
    agg["tier_defaults"] = {
        t: {"per_minute": rpm, "daily": quota} for t, (rpm, quota) in TIER_DEFAULTS.items()
    }
    return agg


# ---------------------------------------------------------- first-boot info


@router.get(
    "/first-boot-info",
    summary="One-shot retrieval of the autogenerated admin token (prod only)",
)
async def first_boot_info() -> dict:
    """Return the active admin token exactly once after a fresh boot.

    Only registered as a real endpoint when auth is ON — when auth is OFF the
    handler returns 404 to avoid leaking the dev-time bypass posture. After the
    first successful read the marker file is created and subsequent calls
    return 410 Gone; the operator must read the persisted token file or
    consult the startup logs to recover it after that.
    """
    if not is_auth_enabled():
        # Don't reveal that the endpoint exists in dev — the bypass posture is
        # not something a leak from a misconfigured prod replica should expose.
        raise HTTPException(status_code=404, detail="Not Found")
    if first_boot_done():
        raise HTTPException(
            status_code=410,
            detail=(
                "First-boot info already retrieved. "
                "Read /tmp/pfm_admin_token.json on the host or set PFM_ADMIN_TOKEN."
            ),
        )
    token = get_or_generate_admin_token()
    if not token:
        # Defensive: is_auth_enabled() said yes but token is empty.
        raise HTTPException(status_code=500, detail="Admin token unavailable.")
    mark_first_boot_done()
    return {
        "admin_token": token,
        "message": "Save this token. Set PFM_ADMIN_TOKEN env var to persist.",
        "warning": "endpoint disabled after first call",
    }
