"""Shared FastAPI dependencies and small request-scoped helpers.

Centralising these here lets feature routers live in their own module without
importing from ``pfm.main`` (which would create a circular import). Every
helper resolves its state via ``request.app.state``, so any FastAPI ``app``
that ran the startup lifespan in ``pfm.main`` exposes the same objects.

Why this exists
---------------
Historically ``pfm/main.py`` defined ``get_cache`` / ``get_factors_dep`` /
``get_polymarket_client`` as closures over a module-level ``app``. That made
the routers physically inseparable from ``main.py`` (4700+ lines). Moving the
helpers to read ``request.app.state`` removes that coupling and lets us
gradually carve ``/factors/*``, ``/strategies/*``, ``/fit`` etc. into their
own files.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from fastapi import Request

from pfm.cache import CacheBackend, NullCache

if TYPE_CHECKING:
    from pfm.factors import FactorConfig
    from pfm.sources.kalshi import KalshiClient
    from pfm.sources.polymarket import PolymarketClient


# ---------------------------------------------------------------------------
# Request-scoped dependencies. Each reads from ``request.app.state`` populated
# during the FastAPI lifespan in ``pfm.main``. Callers wire them with
# ``Annotated[T, Depends(get_*)]`` — FastAPI injects ``Request`` for us.
# ---------------------------------------------------------------------------


def _app_state() -> object:
    """Resolve ``app.state`` even when called outside the FastAPI request cycle.

    Legacy call sites (e.g. ``reverse_finder_router.py``) call these helpers
    without a ``Request``. To stay backward-compatible we lazy-import the
    process-wide ``app`` from ``pfm.main`` — the import is deferred so this
    module can be imported by ``pfm.main`` itself at start-up without cycle.
    """
    from pfm import main as _main_mod  # local to avoid circular import at module load

    return _main_mod.app.state


# FastAPI dependency functions. When mounted via ``Depends(get_cache)`` FastAPI
# inspects the signature and injects the ``Request``. Legacy callers that hit
# these without a Request (e.g. background tasks, helper utilities) instead use
# the matching ``current_*`` accessors below.


def get_cache(request: Request) -> CacheBackend:
    """Return the process-wide cache backend (Redis in prod, Null in tests)."""
    return getattr(request.app.state, "cache", NullCache())


def get_factors_dep(request: Request) -> dict[str, FactorConfig]:
    """Return the loaded factor catalogue keyed by factor id."""
    return request.app.state.factors


def get_polymarket_client(request: Request) -> PolymarketClient:
    """Return the singleton Polymarket HTTP client wired during startup."""
    return request.app.state.poly


def get_kalshi_client(request: Request) -> KalshiClient:
    """Return the singleton Kalshi HTTP client wired during startup."""
    existing = getattr(request.app.state, "kalshi", None)
    if existing is not None:
        return existing
    from pfm.sources.kalshi import KalshiClient as _KalshiClient

    return _KalshiClient()


# ---------------------------------------------------------------------------
# Direct accessors for non-DI call sites (background tasks, legacy helpers).
# These read the singleton ``app`` from ``pfm.main`` via lazy import.
# ---------------------------------------------------------------------------


def current_cache() -> CacheBackend:
    """Return the cache backend without needing a request."""
    return getattr(_app_state(), "cache", NullCache())


def current_factors() -> dict[str, FactorConfig]:
    """Return the loaded factor catalogue without needing a request."""
    return _app_state().factors


def current_polymarket() -> PolymarketClient:
    """Return the Polymarket client singleton without needing a request."""
    return _app_state().poly


def current_kalshi() -> KalshiClient:
    """Return the Kalshi client singleton without needing a request."""
    existing = getattr(_app_state(), "kalshi", None)
    if existing is not None:
        return existing
    from pfm.sources.kalshi import KalshiClient as _KalshiClient

    return _KalshiClient()


# ---------------------------------------------------------------------------
# Stateless helpers that callers reach for from many routers.
# ---------------------------------------------------------------------------


def cache_key(*parts: object) -> str:
    """Build a deterministic cache key from arbitrary serialisable parts.

    The legacy alias ``_cache_key`` is preserved for ``pfm.main`` while the
    extraction is in flight; new code should import ``cache_key`` directly.
    """
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return "pfm:" + hashlib.sha256(blob).hexdigest()
