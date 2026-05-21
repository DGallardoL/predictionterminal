"""Unified thread-safe TTL cache used across the terminal_* modules.

Several ``terminal_*`` modules grew their own near-identical pair of
``_cache_get`` / ``_cache_set`` helpers backed by a module-level dict of
``key -> (expiry_unix, payload)``. This module consolidates that pattern
so new modules don't have to reinvent it.

Backward compatibility
----------------------
:class:`TerminalCache` accepts an optional ``store`` argument that lets
callers expose the underlying dict at module scope (e.g. as ``_CACHE``).
Existing tests that assert ``module._CACHE == {}`` or call
``module._CACHE.clear()`` continue to pass when the backing store is the
same dict object — see ``pfm.terminal_correlations`` and
``pfm.terminal_gdelt_news`` for the migration pattern.

Public API
----------
- :class:`TerminalCache` — get/set/clear/get_or_compute/stats.
- :func:`get_cache` — process-wide named-instance factory.
- :func:`cached` — decorator that fronts a pure function with a
  :class:`TerminalCache` keyed by the call's args.

Threading model
---------------
Every public method takes the cache's lock. The lock is re-entrant so
``get_or_compute`` can ``set`` from within its own critical section
without deadlocking. The compute callback runs *outside* the lock to
avoid holding it during slow IO; we accept the well-known thundering
herd cost on first miss in exchange for not blocking other callers.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Hashable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Stored value shape: (expiry_unix_seconds, payload).
_Entry = tuple[float, Any]


class TerminalCache:
    """Thread-safe TTL cache for terminal-style endpoints.

    Args:
        default_ttl: Seconds entries live for unless overridden per-call.
        store: Optional external dict to use as the backing store. Pass
            this when a module needs to keep ``_CACHE`` exposed at module
            scope for legacy tests; in that case the same dict is shared
            between this instance and any ``module._CACHE`` reference.
    """

    def __init__(
        self,
        default_ttl: int = 300,
        *,
        store: dict[Any, _Entry] | None = None,
    ) -> None:
        self._store: dict[Any, _Entry] = store if store is not None else {}
        self._ttl = int(default_ttl)
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    # --- core ---------------------------------------------------------------

    def get(self, key: Hashable) -> Any | None:
        """Return cached value or ``None`` if missing / expired.

        Expired entries are removed lazily on access.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            expiry, payload = entry
            if expiry < time.time():
                self._store.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return payload

    def set(
        self,
        key: Hashable,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Insert ``value`` for ``key`` with an explicit or default TTL."""
        ttl_seconds = int(ttl) if ttl is not None else self._ttl
        with self._lock:
            self._store[key] = (time.time() + ttl_seconds, value)

    def get_or_compute(
        self,
        key: Hashable,
        fn: Callable[[], T],
        ttl: int | None = None,
    ) -> T:
        """Return the cached value for ``key`` or compute, store, and return it.

        ``fn`` runs *outside* the lock. Concurrent first-miss callers may
        all run ``fn``; whichever finishes last wins the ``set``. This is
        deliberate — the alternative (single-flight) would require a
        per-key lock and the IO is rarely hot-pathed enough to justify it.
        """
        cached_val = self.get(key)
        if cached_val is not None:
            return cached_val
        computed = fn()
        self.set(key, computed, ttl=ttl)
        return computed

    def clear(self) -> None:
        """Drop every entry. Counters are reset too."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int]:
        """Return ``{hits, misses, size}`` snapshot.

        Note: ``size`` is the raw store length and may include expired
        entries that haven't been touched yet. Callers that care can run
        :meth:`prune` first.
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._store),
            }

    def prune(self) -> int:
        """Remove expired entries; return the number of entries dropped."""
        now = time.time()
        with self._lock:
            doomed = [k for k, (exp, _) in self._store.items() if exp < now]
            for k in doomed:
                self._store.pop(k, None)
            return len(doomed)


# --- process-wide named instances -------------------------------------------
# Modules call ``get_cache("namespace")`` to share a single instance keyed
# by name. The first call sets the TTL; subsequent calls return the same
# object regardless of the ``ttl`` arg (so passing different TTLs from
# different call sites does not silently mutate behaviour).

_instances: dict[str, TerminalCache] = {}
_instances_lock = threading.Lock()


def get_cache(namespace: str = "default", ttl: int = 300) -> TerminalCache:
    """Return the process-wide :class:`TerminalCache` named ``namespace``.

    If the namespace doesn't exist yet a new cache is created with
    ``default_ttl=ttl``; otherwise the existing instance is returned and
    ``ttl`` is ignored. Use :func:`reset_caches` in tests if you need a
    fresh slate.
    """
    with _instances_lock:
        cache = _instances.get(namespace)
        if cache is None:
            cache = TerminalCache(default_ttl=ttl)
            _instances[namespace] = cache
        return cache


def reset_caches() -> None:
    """Clear every named instance. Test-only.

    Clears each cache's contents but PRESERVES the singleton identity so
    module-level captures (``_FOO_CACHE = get_cache("foo")``) remain
    referring to the same TerminalCache object after reset. Dropping
    ``_instances`` here used to orphan those captures, which then leaked
    state across tests because subsequent ``get_cache("foo")`` calls
    returned a NEW empty instance while the module kept writing to the
    OLD (now-uncleared) one.
    """
    with _instances_lock:
        for c in _instances.values():
            c.clear()


# --- decorator ---------------------------------------------------------------


def _default_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
    """Build a deterministic hashable key from ``(args, kwargs)``.

    Falls back to ``repr`` for any non-hashable argument so the decorator
    doesn't blow up on lists/dicts; the resulting key is still
    deterministic but slower to construct.
    """
    try:
        return (args, tuple(sorted(kwargs.items())))
    except TypeError:
        return (repr(args), repr(sorted(kwargs.items())))


def cached(
    namespace: str,
    ttl: int = 300,
    key_fn: Callable[..., Hashable] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that fronts a pure function with a named cache.

    Preferred over the explicit ``cache = get_cache("ns"); cache.get(...)``
    pattern when the function's args fully determine the cache key. For
    cases where you need conditional caching (e.g. don't cache empty
    responses) or where the key depends on something other than the args
    (e.g. an env var), drop down to the explicit form.

    Migration examples
    ------------------
    Before::

        _CACHE: dict[str, tuple[float, dict]] = {}
        _CACHE_TTL = 600

        def get_peers(slug: str, top: int = 10) -> dict:
            key = (slug, top)
            now = time.time()
            if key in _CACHE and _CACHE[key][0] > now:
                return _CACHE[key][1]
            result = _compute_peers(slug, top)
            _CACHE[key] = (now + _CACHE_TTL, result)
            return result

    After::

        @cached(namespace="peers", ttl=600)
        def get_peers(slug: str, top: int = 10) -> dict:
            return _compute_peers(slug, top)

    ``key_fn(*args, **kwargs)`` overrides the default tuple-based key
    builder when callers need a custom hash (e.g. dropping a ``Request``
    arg from the key)::

        @cached(
            namespace="quote",
            ttl=30,
            key_fn=lambda request, slug: slug,  # ignore the Request
        )
        def get_quote(request: Request, slug: str) -> dict: ...
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        cache = get_cache(namespace, ttl=ttl)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                key = _default_key(args, kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            cache.set(key, result, ttl=ttl)
            return result

        # Expose the backing cache for tests / debugging.
        wrapper.__wrapped_cache__ = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator


__all__ = [
    "TerminalCache",
    "cached",
    "get_cache",
    "reset_caches",
]
