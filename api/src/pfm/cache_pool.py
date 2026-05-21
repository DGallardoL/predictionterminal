"""Unified L1+L2 cache abstraction (``CachePool``).

This module wraps the two caching layers used throughout the codebase
(in-process TTL dict and an optional Redis backend) behind a single,
small API so endpoint handlers and source modules can stop hand-rolling
their own caches.

Two layers
----------

L1 — in-process dict with per-key TTL and a hard ``maxsize`` cap. Eviction
is heap-based (oldest-expiry first), no LRU.  This is per-gunicorn-worker,
so a hit here is fast (~microseconds).

L2 — optional Redis. Same magic-byte pickle envelope as
``pfm.terminal.__init__.TTLCache`` (the pattern OVERNIGHT-RECAP wave-3
established): ``b"PFMCP1\\x00"`` + pickle of ``{"v": 1, "data": value}``.
Pickle handles arbitrary Python objects faithfully (``pd.Series``,
``np.ndarray``, dataclasses, …) — the older
``json.dumps(default=str)`` approach silently stringified Series values.

Single-flight protection
------------------------

``get_or_compute_async`` keeps a per-key ``asyncio.Lock``. Ten concurrent
callers asking for the same missing key cause exactly one ``fn()``
invocation; the other nine get the cached value. The synchronous
``get_or_compute`` uses a per-key ``threading.Lock`` for the same reason.

Graceful degradation
--------------------

If the Redis backend raises any error on construction or during a get/set
the pool flips to L1-only mode and emits a single structured warning. The
caller never sees the exception.

Migration note
--------------

The existing ad-hoc caches in ``pfm/terminal/__init__.py`` (``TTLCache``)
and ``pfm/sources/scanner.py`` (module-level dicts keyed by slug) can be
replaced with::

    from pfm.cache_pool import CachePool

    TERMINAL_CACHE = CachePool(namespace="term", redis=redis_backend)
    # then everywhere:
    val = TERMINAL_CACHE.get(key)
    TERMINAL_CACHE.set(key, val, ttl=300)
    # or:
    val = await TERMINAL_CACHE.get_or_compute_async(
        key, lambda: _fetch_from_gamma(slug), ttl=60
    )

Do NOT migrate them in this commit — separate task. The current callers
keep working until the migration ticket lands.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import heapq
import hmac
import logging
import os
import pickle
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Magic prefix for the L2 pickle envelope — matches the wave-3 pattern in
# ``pfm.terminal.TTLCache`` but with a distinct tag so the two caches can
# coexist on the same Redis without mistaking each other's entries.
#
# Two magics so HMAC-signed envelopes are distinguishable from legacy ones:
# - ``_L2_MAGIC`` (v1) — unsigned pickle (legacy). Accepted on decode only in
#   dev or when ``PFM_CACHE_ALLOW_UNSIGNED_PICKLE=1``.
# - ``_L2_MAGIC_SIGNED`` (v2) — HMAC-SHA256 prefix + pickle. The 32-byte HMAC
#   is computed over the pickle bytes with the secret in
#   ``PFM_CACHE_HMAC_SECRET``. Tamper-proof; what we write by default when a
#   secret is configured.
_L2_MAGIC: bytes = b"PFMCP1\x00"
_L2_MAGIC_SIGNED: bytes = b"PFMCP2\x00"
_L2_PAYLOAD_VERSION: int = 1
_L2_HMAC_LEN: int = 32  # sha256 digest

_HMAC_SECRET_ENV = "PFM_CACHE_HMAC_SECRET"
_ALLOW_UNSIGNED_ENV = "PFM_CACHE_ALLOW_UNSIGNED_PICKLE"


def _hmac_secret() -> bytes | None:
    """Return the configured HMAC secret, or None if unset."""
    raw = os.environ.get(_HMAC_SECRET_ENV)
    if not raw:
        return None
    return raw.encode("utf-8")


def _is_production() -> bool:
    """Best-effort production detection. Mirrors :mod:`pfm.auth.production`
    so we don't have an import cycle for a one-line check."""
    if (os.environ.get("ENV") or "").lower() == "production":
        return True
    if (os.environ.get("PFM_ENV") or "").lower() == "production":
        return True
    if (os.environ.get("NODE_ENV") or "").lower() == "production":
        return True
    if os.environ.get("FLY_APP_NAME"):
        return True
    return bool(os.environ.get("RENDER"))


def _allow_unsigned() -> bool:
    return os.environ.get(_ALLOW_UNSIGNED_ENV, "").lower() in {"1", "true", "yes"}


@dataclass(order=True)
class _HeapEntry:
    """Heap node sorted by ``expires_at``; payload kept on the side."""

    expires_at: float
    key: str = field(compare=False)


@dataclass
class _Entry:
    value: Any
    expires_at: float


class CachePool:
    """Two-tier cache with stampede protection and graceful Redis degradation.

    Parameters
    ----------
    namespace:
        Prefix for L2 keys; final Redis key is ``pfm:{namespace}:{key}``.
        Required so multiple ``CachePool`` instances can share the same
        Redis without collision.
    redis:
        Optional Redis-like backend. Anything with ``get(key)``,
        ``set(key, value, ex=ttl)`` (or ``ttl_seconds=``), and a truthy
        ``enabled`` attribute. ``None`` ⇒ L1-only.
    l1_maxsize:
        Hard cap on L1 entries. Eviction is heap-based by expiry time —
        the entry closest to expiry leaves first. Not LRU; that would
        cost an extra dict.

    Thread-safety
    -------------
    All public methods take a single ``threading.Lock``. The async
    ``get_or_compute_async`` additionally keeps per-key ``asyncio.Lock``
    instances to single-flight concurrent callers.
    """

    def __init__(
        self,
        *,
        namespace: str,
        redis: Any | None = None,
        l1_maxsize: int = 1024,
    ) -> None:
        if not namespace or ":" in namespace:
            raise ValueError(f"namespace must be non-empty and contain no ':' (got {namespace!r})")
        self._namespace = namespace
        self._redis = redis
        self._l1_maxsize = max(1, int(l1_maxsize))

        self._d: dict[str, _Entry] = {}
        self._heap: list[_HeapEntry] = []
        self._lock = threading.Lock()

        # Stats counters guarded by ``_stats_lock`` to avoid the GIL-vs-counter
        # foot-gun (two threads reading the same int and both incrementing).
        self._stats_lock = threading.Lock()
        self._stat_l1_hits = 0
        self._stat_l2_hits = 0
        self._stat_misses = 0
        self._stat_set_count = 0

        # Per-key single-flight locks. Two flavours so sync and async callers
        # don't share a lock type (asyncio.Lock isn't safe across threads).
        self._async_locks: dict[str, asyncio.Lock] = {}
        self._async_locks_guard = threading.Lock()
        self._sync_locks: dict[str, threading.Lock] = {}
        self._sync_locks_guard = threading.Lock()

        # Track whether the Redis backend is healthy. The first error flips
        # ``_redis_degraded`` to ``True`` and we stop trying for the rest
        # of the process lifetime. Reset is intentionally not exposed.
        self._redis_degraded = False
        if self._redis is not None and not self._redis_is_enabled():
            self._redis_degraded = True
            logger.warning(
                "cache_pool.redis_disabled",
                extra={
                    "namespace": self._namespace,
                    "reason": "backend reports not enabled at construction",
                },
            )

    # -- internals -----------------------------------------------------------

    def _redis_key(self, key: str) -> str:
        return f"pfm:{self._namespace}:{key}"

    def _redis_is_enabled(self) -> bool:
        """Best-effort 'is the Redis backend usable?' check.

        We accept either a truthy ``enabled`` attribute (matches the in-tree
        ``RedisCache``/``NullCache`` protocol) or, if absent, assume the
        backend is live and let the first call fail-open.
        """
        if self._redis is None:
            return False
        enabled = getattr(self._redis, "enabled", None)
        if enabled is None:
            return True
        return bool(enabled)

    def _redis_available(self) -> bool:
        if self._redis is None or self._redis_degraded:
            return False
        return self._redis_is_enabled()

    @staticmethod
    def _encode_l2(value: Any) -> bytes:
        envelope = {"v": _L2_PAYLOAD_VERSION, "data": value}
        body = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        secret = _hmac_secret()
        if secret is not None:
            # Signed envelope: PFMCP2 + 32-byte HMAC + pickle body.
            mac = hmac.new(secret, body, hashlib.sha256).digest()
            return _L2_MAGIC_SIGNED + mac + body
        # No secret configured. Refuse to write *unsigned* pickles in
        # production unless the operator opted in explicitly — accepting an
        # unsigned read elsewhere on the cluster could let an attacker plant
        # a malicious payload through any RCE-on-Redis vector.
        if _is_production() and not _allow_unsigned():
            raise RuntimeError(
                "cache_pool: refusing to write unsigned pickle to L2 in production. "
                "Set PFM_CACHE_HMAC_SECRET (recommended) or "
                "PFM_CACHE_ALLOW_UNSIGNED_PICKLE=1 to override (NOT recommended)."
            )
        return _L2_MAGIC + body

    @staticmethod
    def _decode_l2(raw: bytes | str) -> Any:
        if isinstance(raw, str):
            raw = raw.encode()
        if raw.startswith(_L2_MAGIC_SIGNED):
            secret = _hmac_secret()
            if secret is None:
                raise ValueError("signed L2 payload but PFM_CACHE_HMAC_SECRET is not set")
            body_offset = len(_L2_MAGIC_SIGNED) + _L2_HMAC_LEN
            if len(raw) < body_offset:
                raise ValueError("truncated signed L2 payload")
            mac_got = raw[len(_L2_MAGIC_SIGNED) : body_offset]
            body = raw[body_offset:]
            mac_want = hmac.new(secret, body, hashlib.sha256).digest()
            if not hmac.compare_digest(mac_got, mac_want):
                # Compare-digest failure → treat as a corrupted/tampered entry.
                # Never unpickle in this branch — that's the whole point.
                raise ValueError("L2 payload HMAC mismatch")
            envelope = pickle.loads(body)
            if not isinstance(envelope, dict) or envelope.get("v") != _L2_PAYLOAD_VERSION:
                raise ValueError(f"unsupported L2 payload version: {envelope!r}")
            return envelope["data"]
        if raw.startswith(_L2_MAGIC):
            # Legacy unsigned envelope. Accept only when:
            #   - not running in production, OR
            #   - operator explicitly opted in via PFM_CACHE_ALLOW_UNSIGNED_PICKLE=1
            # Otherwise refuse — better to bust the cache than to RCE.
            if _is_production() and not _allow_unsigned():
                raise ValueError(
                    "legacy unsigned L2 payload rejected in production "
                    "(set PFM_CACHE_HMAC_SECRET or PFM_CACHE_ALLOW_UNSIGNED_PICKLE=1)"
                )
            envelope = pickle.loads(raw[len(_L2_MAGIC) :])
            if not isinstance(envelope, dict) or envelope.get("v") != _L2_PAYLOAD_VERSION:
                raise ValueError(f"unsupported L2 payload version: {envelope!r}")
            return envelope["data"]
        raise ValueError("legacy or unknown L2 payload (missing PFMCP1/PFMCP2 magic)")

    def _evict_if_full(self, now: float) -> None:
        """Heap-based eviction. Caller MUST hold ``self._lock``."""
        # Drop everything that has already expired in one sweep so we don't
        # over-evict still-fresh entries when the cap is tight.
        while self._heap and self._heap[0].expires_at <= now:
            stale = heapq.heappop(self._heap)
            existing = self._d.get(stale.key)
            if existing is not None and existing.expires_at <= now:
                self._d.pop(stale.key, None)
        # Still over the cap? Evict the closest-to-expiry entry.
        while len(self._d) >= self._l1_maxsize and self._heap:
            victim = heapq.heappop(self._heap)
            # The heap can contain stale references (a previous set() for the
            # same key pushed a newer entry without removing the old one); only
            # evict when the popped expiry actually matches the live entry.
            live = self._d.get(victim.key)
            if live is not None and live.expires_at == victim.expires_at:
                self._d.pop(victim.key, None)

    def _l1_get(self, key: str, now: float) -> tuple[bool, Any]:
        """Return ``(hit, value)`` for the L1 layer. Caller holds no locks."""
        with self._lock:
            entry = self._d.get(key)
            if entry is None:
                return (False, None)
            if entry.expires_at < now:
                self._d.pop(key, None)
                return (False, None)
            return (True, entry.value)

    def _l1_set(self, key: str, value: Any, ttl: int) -> None:
        now = time.time()
        expires_at = now + max(0, int(ttl))
        with self._lock:
            self._evict_if_full(now)
            self._d[key] = _Entry(value=value, expires_at=expires_at)
            heapq.heappush(self._heap, _HeapEntry(expires_at=expires_at, key=key))

    def _l2_get(self, key: str) -> tuple[bool, Any]:
        if not self._redis_available():
            return (False, None)
        try:
            raw = self._redis.get(self._redis_key(key))
        except Exception as e:
            self._mark_degraded("get", e)
            return (False, None)
        if not raw:
            return (False, None)
        try:
            return (True, self._decode_l2(raw))
        except Exception:
            # Legacy/corrupt entry — treat as a miss, let it expire naturally.
            return (False, None)

    def _l2_set(self, key: str, value: Any, ttl: int) -> None:
        if not self._redis_available():
            return
        try:
            blob = self._encode_l2(value)
        except (pickle.PicklingError, TypeError, AttributeError):
            # Not picklable (open file handle, lock, …). L1-only is fine.
            return
        try:
            # Cap at 1h so we don't pollute Redis with month-long entries.
            capped = min(max(1, int(ttl)), 3600)
            # Most clients accept ``ex=``; the in-tree ``RedisCache`` uses a
            # positional 3rd arg. Try both forms so the pool works with either.
            try:
                self._redis.set(self._redis_key(key), blob, ex=capped)
            except TypeError:
                self._redis.set(self._redis_key(key), blob, capped)
        except Exception as e:
            self._mark_degraded("set", e)

    def _l2_delete(self, key: str) -> None:
        if not self._redis_available():
            return
        try:
            delete = getattr(self._redis, "delete", None)
            if delete is not None:
                delete(self._redis_key(key))
        except Exception as e:
            self._mark_degraded("delete", e)

    def _mark_degraded(self, op: str, exc: BaseException) -> None:
        if self._redis_degraded:
            return
        self._redis_degraded = True
        logger.warning(
            "cache_pool.redis_degraded",
            extra={
                "namespace": self._namespace,
                "op": op,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

    def _bump(self, attr: str) -> None:
        with self._stats_lock:
            setattr(self, attr, getattr(self, attr) + 1)

    # -- public API ----------------------------------------------------------

    def get(self, key: str, *, default: Any = None) -> Any:
        """Return the cached value or ``default`` on miss.

        Order: L1 → L2 → ``default``. An L2 hit is promoted into L1 with a
        short TTL (30s) so subsequent reads in the same worker skip Redis.
        """
        now = time.time()
        hit, value = self._l1_get(key, now)
        if hit:
            self._bump("_stat_l1_hits")
            return value
        hit, value = self._l2_get(key)
        if hit:
            self._bump("_stat_l2_hits")
            # Promote into L1 with a small TTL — the L2 TTL handles real expiry.
            self._l1_set(key, value, 30)
            return value
        self._bump("_stat_misses")
        return default

    def set(self, key: str, value: Any, *, ttl: int = 60) -> None:
        """Store ``value`` in both L1 and L2 with a ``ttl``-second lifetime."""
        self._bump("_stat_set_count")
        self._l1_set(key, value, ttl)
        self._l2_set(key, value, ttl)

    def delete(self, key: str) -> None:
        """Remove ``key`` from both layers. Silent on miss."""
        with self._lock:
            self._d.pop(key, None)
        self._l2_delete(key)

    def clear(self, *, prefix: str | None = None) -> int:
        """Drop all entries whose key starts with ``prefix`` (or all if ``None``).

        Returns the number of L1 entries removed. L2 entries with the same
        prefix are best-effort-deleted via ``redis.scan_iter`` if available.
        """
        with self._lock:
            if prefix is None:
                count = len(self._d)
                self._d.clear()
                self._heap.clear()
            else:
                victims = [k for k in self._d if k.startswith(prefix)]
                for k in victims:
                    self._d.pop(k, None)
                count = len(victims)
        # L2 best-effort scan_iter — only some Redis clients expose it.
        if self._redis_available():
            scan = getattr(self._redis, "scan_iter", None)
            if scan is not None:
                try:
                    pattern_prefix = "" if prefix is None else prefix
                    redis_pattern = f"pfm:{self._namespace}:{pattern_prefix}*"
                    for raw_key in scan(match=redis_pattern):
                        with contextlib.suppress(Exception):
                            self._redis.delete(raw_key)
                except Exception as e:
                    self._mark_degraded("scan_iter", e)
        return count

    def _get_async_lock(self, key: str) -> asyncio.Lock:
        with self._async_locks_guard:
            lock = self._async_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._async_locks[key] = lock
            return lock

    def _get_sync_lock(self, key: str) -> threading.Lock:
        with self._sync_locks_guard:
            lock = self._sync_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._sync_locks[key] = lock
            return lock

    async def get_or_compute_async(
        self,
        key: str,
        fn: Callable[[], Awaitable[Any]],
        *,
        ttl: int = 60,
    ) -> Any:
        """Async single-flight cache fetch.

        If ``key`` is not in the cache, exactly one concurrent caller
        invokes ``fn()`` — the others wait on the per-key ``asyncio.Lock``
        and then read the populated value out of L1.
        """
        sentinel = object()
        cached = self.get(key, default=sentinel)
        if cached is not sentinel:
            return cached
        lock = self._get_async_lock(key)
        async with lock:
            cached = self.get(key, default=sentinel)
            if cached is not sentinel:
                return cached
            value = await fn()
            self.set(key, value, ttl=ttl)
            return value

    def get_or_compute(
        self,
        key: str,
        fn: Callable[[], Any],
        *,
        ttl: int = 60,
    ) -> Any:
        """Synchronous single-flight cache fetch (per-key ``threading.Lock``)."""
        sentinel = object()
        cached = self.get(key, default=sentinel)
        if cached is not sentinel:
            return cached
        lock = self._get_sync_lock(key)
        with lock:
            cached = self.get(key, default=sentinel)
            if cached is not sentinel:
                return cached
            value = fn()
            self.set(key, value, ttl=ttl)
            return value

    @property
    def stats(self) -> dict[str, int]:
        """Snapshot of hit/miss counters. Read under the stats lock."""
        with self._stats_lock:
            return {
                "l1_hits": self._stat_l1_hits,
                "l2_hits": self._stat_l2_hits,
                "misses": self._stat_misses,
                "set_count": self._stat_set_count,
                "l1_size": len(self._d),
                "redis_degraded": self._redis_degraded,
            }

    def __repr__(self) -> str:
        return (
            f"CachePool(namespace={self._namespace!r}, "
            f"l1_size={len(self._d)}, l1_maxsize={self._l1_maxsize}, "
            f"redis={'on' if self._redis_available() else 'off'})"
        )
