"""``POST /admin/cache/invalidate`` — manual cache invalidation (W12-18).

Operational counterpart to :mod:`pfm.admin.cache_stats_router`. The stats
router lets operators *see* which caches exist; this one lets them
*flush* slices of them on demand — useful when a stale factor catalog is
serving wrong answers and the natural TTL is hours away.

Behaviour
---------

Body: ``{"prefix": "factors:", "namespace": "manifold-search"}`` — both
fields optional but the request must specify at least one (a body where
both are empty/missing is a 422). The handler then:

1. Discovers every loaded ``CachePool`` instance by walking ``sys.modules``
   for ``pfm.*``  — same introspection helper used by the stats router
   (kept private in that module, reimplemented here as a local fallback
   so the two routers stay decoupled).
2. Filters by ``namespace`` if provided (exact match on
   ``pool._namespace``); otherwise every pool is in scope.
3. Calls ``pool.clear(prefix=...)`` on each matched pool — ``prefix=None``
   (i.e. no prefix in the body) wipes the entire pool, which is what the
   ``CachePool.clear`` contract already supports.
4. Returns a per-pool ``{namespace, removed, remaining}`` breakdown plus
   the ISO-8601 ``invalidated_at`` and the grand-total ``total_removed``.

Authentication
--------------

Belt-and-braces: the *route* declares an optional dependency
``_check_admin_token`` that reads ``Authorization: Bearer <token>`` and
compares against ``os.environ['PFM_ADMIN_TOKEN']``.

* ``PFM_ADMIN_TOKEN`` unset → dev mode, no auth check. This mirrors the
  pattern used by :func:`pfm.pm_vix._admin_dep_if_enabled` and the
  signals-recompute endpoint.
* ``PFM_ADMIN_TOKEN`` set, header missing or mismatched → 403.
* ``PFM_ADMIN_TOKEN`` set, header matches → 200.

The token is read **per-request** (not at module import) so the operator
can flip the env var without restarting gunicorn.

Why a router-level token check and not the ``pfm.auth`` framework
-----------------------------------------------------------------

``pfm.auth.dependencies.require_admin`` is built around DB-backed user
sessions with rate-limited login. That's the right tool for end-user
endpoints. For ops endpoints we want **out-of-band** access that does
not depend on the auth tables existing (think bootstrapping, recovery,
or a worker container with no DB). A pre-shared env-var token is the
standard approach for that — it's used in two other places already
(``pm_vix.refresh-slugs``, ``live_signals_job.recompute-now``).

This router does NOT mount itself on the app — ``pfm.main`` includes
admin routers explicitly. Keeping the mount external lets callers wrap
the routes in additional middleware (rate limit, IP allow-list, …) per
environment.
"""

from __future__ import annotations

import importlib
import os
import sys
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from pfm.cache_pool import CachePool

router = APIRouter(tags=["admin"])


# ─────────────────────── request / response models ────────────────────────


class CacheInvalidateRequest(BaseModel):
    """Body for ``POST /admin/cache/invalidate``.

    At least one of ``prefix`` / ``namespace`` must be a non-empty string.
    A body with both fields missing or empty fails Pydantic validation
    with a 422 from FastAPI before the handler runs.
    """

    prefix: str | None = Field(
        default=None,
        description=(
            "Key prefix to clear within each matched pool. Passed straight "
            "to ``CachePool.clear(prefix=...)``. ``None`` or empty wipes "
            "the whole pool."
        ),
    )
    namespace: str | None = Field(
        default=None,
        description=(
            "Restrict the operation to the pool whose ``CachePool.namespace`` "
            "matches exactly. ``None`` or empty means 'every discovered pool'."
        ),
    )

    @model_validator(mode="after")
    def _at_least_one(self) -> CacheInvalidateRequest:
        # Empty strings count as "not provided" — otherwise a JS frontend
        # sending `""` for an optional field would silently widen the
        # invalidation scope to "wipe everything", which is dangerous.
        prefix_provided = bool(self.prefix and self.prefix.strip())
        namespace_provided = bool(self.namespace and self.namespace.strip())
        if not prefix_provided and not namespace_provided:
            raise ValueError("body must include at least one non-empty 'prefix' or 'namespace'")
        return self


class _PoolResult(BaseModel):
    namespace: str
    removed: int
    remaining: int


class CacheInvalidateResponse(BaseModel):
    invalidated_at: str
    results: list[_PoolResult]
    total_removed: int


# ─────────────────────────── introspection ────────────────────────────


def _iter_cache_pools() -> list[tuple[str, str, CachePool]]:
    """Return ``[(module_name, attr_name, pool), ...]`` across loaded ``pfm.*``.

    This is the same discovery pattern documented in W12-17's
    :func:`pfm.admin.cache_stats_router._iter_cache_pools`. Re-implemented
    here (rather than imported) so the two routers stay independently
    testable; their shared assumption is "module-level CachePool attrs",
    which is enforced in code review, not at runtime.
    """
    found: list[tuple[str, str, CachePool]] = []
    seen: set[int] = set()
    # Snapshot the dict to avoid ``RuntimeError: dictionary changed size
    # during iteration`` if a request handler races with an import.
    snapshot = list(sys.modules.items())
    for mod_name, mod in snapshot:
        if mod is None:
            continue
        if not mod_name.startswith("pfm."):
            continue
        if mod_name.startswith("pfm.admin"):
            # Skip the admin package itself — pools accidentally stashed
            # on a router module during tests must not become reachable.
            continue
        try:
            attr_names = dir(mod)
        except Exception:
            continue
        for attr_name in attr_names:
            try:
                value = getattr(mod, attr_name)
            except Exception:
                continue
            if not isinstance(value, CachePool):
                continue
            pool_id = id(value)
            if pool_id in seen:
                # Re-exports of the same instance from multiple modules
                # would otherwise be cleared twice (harmless) and counted
                # twice (misleading). De-dup by identity.
                continue
            seen.add(pool_id)
            found.append((mod_name, attr_name, value))
    found.sort(key=lambda row: (row[0], row[1]))
    return found


def _eager_import_known_pool_modules() -> None:
    """Import known pool-bearing modules so a cold worker still finds them.

    Mirrors W12-17. The invalidation endpoint is more often invoked from a
    fresh ops shell than the stats endpoint, so cold-start coverage matters
    even more. Failures are swallowed — never crash an admin endpoint on
    an unrelated import bug.
    """
    known = (
        "pfm.sources.manifold",
        "pfm.sources.kalshi",
        "pfm.terminal.quote",
    )
    for name in known:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            continue


# ─────────────────────────── auth dependency ──────────────────────────


def _check_admin_token(
    authorization: str | None = Header(default=None),
) -> None:
    """Token-gate the route when ``PFM_ADMIN_TOKEN`` is set.

    Parsing rules:

    * Header missing AND env unset → allow (dev mode).
    * Env unset → allow regardless of header.
    * Env set, header missing → 403.
    * Env set, header malformed (no ``Bearer`` prefix) → 403.
    * Env set, header token != env token → 403.
    * Env set, tokens match → allow.

    The comparison uses Python's ``==`` rather than a constant-time
    helper. For a self-hosted admin endpoint behind an IP allow-list the
    timing-attack surface is negligible; if this ever becomes
    internet-exposed, swap to ``hmac.compare_digest``.
    """
    expected = os.environ.get("PFM_ADMIN_TOKEN")
    if not expected:
        # Dev mode — no env, no gate. Matches the pm_vix / live_signals_job
        # convention so operators have one mental model for admin auth.
        return
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin token required",
        )
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="expected Authorization: Bearer <token>",
        )
    if parts[1] != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin token mismatch",
        )


# ─────────────────────────── core operation ───────────────────────────


def perform_invalidation(req: CacheInvalidateRequest) -> dict[str, Any]:
    """Execute the invalidation. Exposed for non-HTTP callers (CLI, tests).

    Algorithm
    ---------

    1. Discover all pools via :func:`_iter_cache_pools`.
    2. If ``namespace`` is set, drop pools whose namespace doesn't match.
       This is exact-match — substring matching would be too generous
       (``"foo"`` would unintentionally match ``"foo-extras"``).
    3. For each matched pool, call ``pool.clear(prefix=...)`` and record
       the count returned (L1 removed) plus ``len(pool._d)`` afterwards
       as the "remaining" figure. Note that ``remaining`` is L1-only —
       L2 (Redis) state is best-effort and the pool doesn't expose a
       cheap remaining-count for it.
    """
    pools = _iter_cache_pools()

    target_namespace = (req.namespace or "").strip() or None
    target_prefix = (req.prefix or "").strip() or None

    if target_namespace is not None:
        pools = [(m, a, p) for (m, a, p) in pools if p._namespace == target_namespace]

    results: list[dict[str, Any]] = []
    total_removed = 0
    for _module_name, _attr_name, pool in pools:
        try:
            removed = pool.clear(prefix=target_prefix)
        except Exception:
            # Skip pools whose ``.clear`` raises. Continuing is preferable
            # to a 500 — partial invalidation is more useful than none.
            continue
        with pool._lock:
            remaining = len(pool._d)
        results.append(
            {
                "namespace": pool._namespace,
                "removed": int(removed),
                "remaining": int(remaining),
            }
        )
        total_removed += int(removed)

    return {
        "invalidated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "results": results,
        "total_removed": total_removed,
    }


# ─────────────────────────── HTTP endpoint ────────────────────────────


@router.post(
    "/admin/cache/invalidate",
    response_model=CacheInvalidateResponse,
    tags=["admin"],
)
def invalidate_cache(
    body: CacheInvalidateRequest,
    authorization: str | None = Header(default=None),
) -> CacheInvalidateResponse:
    """Invalidate cache entries by ``prefix`` and/or ``namespace``.

    Returns a per-pool breakdown plus the grand total. See module docstring
    for the auth contract and full behaviour.
    """
    _check_admin_token(authorization)
    _eager_import_known_pool_modules()
    payload = perform_invalidation(body)
    return CacheInvalidateResponse.model_validate(payload)
