"""API key auth, rate limiting, and tier gating for the PFM service.

Auto-detects production environments: ``ENV=production``, ``FLY_APP_NAME``,
``RENDER``, or ``NODE_ENV=production`` flips auth ON by default. The explicit
``PFM_AUTH_ENABLED=1``/``=0`` always wins. When auth is disabled, the
dependencies + rate-limit middleware fast-path so dev workflows + the existing
test suite stay unaffected.

Public surface:

- :class:`pfm.auth.models.APIKey`
- :class:`pfm.auth.storage.APIKeyStore`
- :func:`pfm.auth.dependencies.require_api_key`
- :func:`pfm.auth.dependencies.optional_api_key`
- :func:`pfm.auth.dependencies.require_tier`
- :func:`pfm.auth.rate_limiter.check_and_increment`
- :func:`pfm.auth.rate_limiter.RateLimitMiddleware`
- :func:`pfm.auth.production.is_auth_enabled`
- :func:`pfm.auth.production.get_or_generate_admin_token`
- :data:`pfm.auth.router.router`
"""

from __future__ import annotations

from pfm.auth.dependencies import (
    optional_api_key,
    require_api_key,
    require_tier,
)
from pfm.auth.models import TIER_DEFAULTS, TIER_ORDER, APIKey
from pfm.auth.production import (
    detect_env_reason,
    get_or_generate_admin_token,
    is_admin_token_autogen,
    is_auth_enabled,
)
from pfm.auth.rate_limiter import RateLimitMiddleware, check_and_increment
from pfm.auth.router import router
from pfm.auth.storage import APIKeyStore, get_api_key_store

__all__ = [
    "TIER_DEFAULTS",
    "TIER_ORDER",
    "APIKey",
    "APIKeyStore",
    "RateLimitMiddleware",
    "check_and_increment",
    "detect_env_reason",
    "get_api_key_store",
    "get_or_generate_admin_token",
    "is_admin_token_autogen",
    "is_auth_enabled",
    "optional_api_key",
    "require_api_key",
    "require_tier",
    "router",
]
