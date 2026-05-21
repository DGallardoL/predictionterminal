"""``GET /admin/cache-stats`` — combined L1+L2 cache statistics.

Walks every loaded ``pfm.*`` module, finds module-level attributes that
are :class:`pfm.cache_pool.CachePool` instances (the pattern established
in W11-14: ``_SEARCH_CACHE``, ``_MARKET_CACHE``, ``_GAMMA_MARKET_CACHE``,
…), and returns a per-pool breakdown plus aggregate totals.

Why introspection and not a registry
------------------------------------

A registry would require every callsite that constructs a ``CachePool``
to remember to ``register()`` itself. Introspection is forgiving — pools
declared anywhere in ``pfm.*`` show up automatically, including ones
added in future waves. The cost is a one-time walk over ``sys.modules``
plus a ``dir()`` per module on each request, which is microseconds for a
~170-module package.

Aggregation
-----------

Per-pool fields come straight from ``CachePool.stats`` (a dict snapshot
guarded by the pool's stats lock). The ``hit_rate`` and ``totals`` are
computed here:

* ``hit_rate = (l1_hits + l2_hits) / (l1_hits + l2_hits + misses)`` —
  zero requests ⇒ ``0.0`` rather than NaN (the cache is technically 0/0
  effective on a fresh worker; calling it "perfect" or "broken" is
  equally wrong, so default to neutral 0.0).
* ``totals`` is a sum across all pools, computed once for the response.

Endpoint is intentionally *not* gated by admin auth at the router level
— mounting code in ``main.py`` can wrap it (or not) per environment.
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from pfm.cache_pool import CachePool

router = APIRouter(tags=["admin"])


def _iter_cache_pools() -> list[tuple[str, str, CachePool]]:
    """Return ``[(module_name, attr_name, pool), ...]`` across loaded ``pfm.*``.

    Iterates a *snapshot* of ``sys.modules`` to avoid ``RuntimeError:
    dictionary changed size during iteration`` if a request handler races
    with an import. Skips modules that failed to load (``None`` value)
    and the ``pfm.admin.*`` package itself (otherwise this router's own
    test fixtures, if they install pools onto the admin module, would be
    counted).
    """
    found: list[tuple[str, str, CachePool]] = []
    seen: set[int] = set()
    snapshot = list(sys.modules.items())
    for mod_name, mod in snapshot:
        if mod is None:
            continue
        if not mod_name.startswith("pfm."):
            continue
        if mod_name.startswith("pfm.admin"):
            continue
        # ``dir()`` is preferable to ``vars()`` so we get inherited names
        # too — pools live as plain module attributes today but a future
        # mixin could wrap them. Tolerate any AttributeError (lazy attrs
        # implemented via ``__getattr__`` can raise on probe).
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
                # The same pool instance may be re-exported by multiple
                # modules (e.g. via ``from foo import _CACHE``). Only
                # count it once — duplicates would inflate totals.
                continue
            seen.add(pool_id)
            found.append((mod_name, attr_name, value))
    # Sort for determinism — tests rely on stable ordering for snapshot
    # comparisons and humans reading the JSON appreciate it too.
    found.sort(key=lambda row: (row[0], row[1]))
    return found


def _hit_rate(l1_hits: int, l2_hits: int, misses: int) -> float:
    """Compute hit rate; returns 0.0 when there have been zero requests."""
    total = l1_hits + l2_hits + misses
    if total <= 0:
        return 0.0
    return round((l1_hits + l2_hits) / total, 4)


def _pool_row(module_name: str, attr_name: str, pool: CachePool) -> dict[str, Any]:
    """Build a single per-pool dict for the response."""
    stats = pool.stats
    l1_hits = int(stats.get("l1_hits", 0))
    l2_hits = int(stats.get("l2_hits", 0))
    misses = int(stats.get("misses", 0))
    set_count = int(stats.get("set_count", 0))
    l1_size = int(stats.get("l1_size", 0))
    return {
        "namespace": pool._namespace,
        # doesn't expose a getter and rewriting the attribute is overkill.
        "module": module_name,
        "attr": attr_name,
        "l1_hits": l1_hits,
        "l1_misses": misses,
        "l2_hits": l2_hits,
        "set_count": set_count,
        "l1_size": l1_size,
        "redis_degraded": bool(stats.get("redis_degraded", False)),
        "hit_rate": _hit_rate(l1_hits, l2_hits, misses),
    }


def collect_cache_stats() -> dict[str, Any]:
    """Build the full ``/admin/cache-stats`` response payload.

    Exposed as a standalone function (not just a route handler) so other
    callers — a CLI dump, a Prometheus exporter, an admin notebook —
    can grab the same structure without going through HTTP.
    """
    pools = _iter_cache_pools()
    rows = [_pool_row(m, a, p) for (m, a, p) in pools]

    totals_l1_hits = sum(r["l1_hits"] for r in rows)
    totals_l2_hits = sum(r["l2_hits"] for r in rows)
    totals_misses = sum(r["l1_misses"] for r in rows)
    totals_set_count = sum(r["set_count"] for r in rows)
    totals_l1_size = sum(r["l1_size"] for r in rows)

    return {
        "checked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "pool_count": len(rows),
        "pools": rows,
        "totals": {
            "l1_hits": totals_l1_hits,
            "l2_hits": totals_l2_hits,
            "misses": totals_misses,
            "set_count": totals_set_count,
            "l1_size": totals_l1_size,
            "hit_rate": _hit_rate(totals_l1_hits, totals_l2_hits, totals_misses),
        },
    }


def _eager_import_known_pool_modules() -> None:
    """Import known cache-pool-bearing modules so introspection finds them.

    On a fresh worker (especially during tests that build a minimal
    FastAPI app), only modules touched by the request path are loaded.
    ``/admin/cache-stats`` is metadata, not part of any hot path, so the
    pools in ``pfm.sources.manifold`` etc. may not be imported yet.

    We import a short list of known modules best-effort. Import failure
    is logged-and-swallowed — admin endpoints must never crash because
    an unrelated module has a bug.
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


@router.get("/admin/cache-stats", tags=["admin"])
def cache_stats() -> dict[str, Any]:
    """Return aggregated L1+L2 cache stats for every ``CachePool`` in ``pfm.*``."""
    _eager_import_known_pool_modules()
    return collect_cache_stats()
