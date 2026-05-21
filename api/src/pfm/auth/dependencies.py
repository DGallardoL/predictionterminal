"""FastAPI dependencies for API key auth + tier gating.

Auth is automatically ON in production-like environments (``ENV=production``,
``FLY_APP_NAME``, ``RENDER``, ``NODE_ENV=production``) and OFF in dev. The
explicit override ``PFM_AUTH_ENABLED=1``/``=0`` always wins. When auth is
disabled, the dependencies short-circuit to a synthetic "system" key so
existing tests + dev curls keep working.

See :mod:`pfm.auth.production` for the detection logic and admin-token
autogeneration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from pfm.auth.models import APIKey, Tier, tier_at_least
from pfm.auth.production import (
    admin_token_configured as _admin_token_configured,
)
from pfm.auth.production import (
    get_or_generate_admin_token,
    is_auth_enabled,
)
from pfm.auth.storage import APIKeyStore, get_api_key_store


def auth_enabled() -> bool:
    """Backward-compat alias for :func:`pfm.auth.production.is_auth_enabled`."""
    return is_auth_enabled()


def _system_key() -> APIKey:
    """Synthetic key used when auth is disabled (dev / tests)."""
    return APIKey(
        key="sk_pfm_system_bypass",
        user_id="system",
        tier="enterprise",
        created_at=datetime.now(UTC),
        rate_limit_per_minute=30_000,
        daily_quota=0,
    )


def _extract_key(header: str | None) -> str | None:
    """Pull the bearer key out of an ``Authorization`` header."""
    if not header:
        return None
    h = header.strip()
    if h.lower().startswith("bearer "):
        h = h[7:].strip()
    if not h.startswith("sk_pfm_"):
        return None
    return h


async def require_api_key(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
    store: APIKeyStore = Depends(get_api_key_store),
) -> APIKey:
    """Require a valid, enabled API key.

    Accepts either ``Authorization: Bearer sk_pfm_…`` or ``X-API-Key: sk_pfm_…``.
    """
    if not auth_enabled():
        return _system_key()

    raw = _extract_key(authorization) or (
        x_api_key if x_api_key and x_api_key.startswith("sk_pfm_") else None
    )
    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed API key (use 'Authorization: Bearer sk_pfm_…').",
            headers={"WWW-Authenticate": "Bearer"},
        )
    key = store.get_key(raw)
    if key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not key.enabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is disabled.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    store.touch(key.key)
    return key


async def optional_api_key(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
    store: APIKeyStore = Depends(get_api_key_store),
) -> APIKey | None:
    """Same as :func:`require_api_key` but returns ``None`` instead of 401."""
    if not auth_enabled():
        return _system_key()

    raw = _extract_key(authorization) or (
        x_api_key if x_api_key and x_api_key.startswith("sk_pfm_") else None
    )
    if raw is None:
        return None
    key = store.get_key(raw)
    if key is None or not key.enabled:
        return None
    store.touch(key.key)
    return key


def require_tier(min_tier: Tier):
    """Build a FastAPI dependency that enforces ``key.tier >= min_tier``.

    Usage::

        @app.get("/strategies/optimize",
                 dependencies=[Depends(require_tier("pro"))])
        def optimize(...): ...

    The dependency itself takes care of pulling the key out of the request,
    so callers don't need to thread a key argument through their handler.
    """

    async def _dep(key: APIKey = Depends(require_api_key)) -> APIKey:
        if not auth_enabled():
            return key
        if not tier_at_least(key.tier, min_tier):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"This endpoint requires tier '{min_tier}' or higher; "
                    f"your key is on '{key.tier}'."
                ),
            )
        return key

    return _dep


# ----------------------------------------------------------------------- admin


def require_admin(
    x_admin_token: Annotated[str | None, Header()] = None,
) -> None:
    """Cheap shared-secret gate for admin-only endpoints.

    The token resolution defers to :func:`pfm.auth.production.get_or_generate_admin_token`
    so production environments without an explicit ``PFM_ADMIN_TOKEN`` get a
    generated one (rather than silently leaving admin endpoints open). If no
    token is configured *and* auth is off, admin endpoints stay disabled
    (always 403) — fail-closed by design.
    """
    if not _admin_token_configured():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin endpoints are disabled (PFM_ADMIN_TOKEN unset).",
        )
    expected = get_or_generate_admin_token()
    if not expected or x_admin_token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token.",
        )
