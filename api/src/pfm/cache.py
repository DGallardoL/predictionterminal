"""Tiny Redis cache wrapper with a no-op fallback.

If Redis is unreachable the wrapper logs once at WARNING and silently bypasses
caching for the rest of the process lifetime. The API still serves requests,
just without cache amortisation. This avoids the POC failing closed when
Redis is down — fine for a POC; revisit if we ever go to prod.
"""

from __future__ import annotations

import logging
from typing import Protocol

import redis

logger = logging.getLogger(__name__)


class CacheBackend(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes, ttl_seconds: int) -> None: ...
    def setnx(self, key: str, value: bytes, ttl_seconds: int) -> bool: ...


class RedisCache:
    """Thin wrapper around ``redis.Redis`` with graceful degradation."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: redis.Redis | None = None
        self._disabled = False
        try:
            self._client = redis.Redis.from_url(url, socket_timeout=2)
            self._client.ping()
        except (redis.RedisError, ValueError) as e:
            # RedisError = reachable URL but connection/auth failed.
            # ValueError = missing/invalid scheme (e.g. REDIS_URL="" or a bare
            # "host:6379"), which from_url() raises BEFORE any connection. Both
            # must degrade to no-cache, not crash app startup — a first deploy
            # may run before Redis is provisioned.
            logger.warning("redis unavailable at %r (%s) — caching disabled", url, e)
            self._disabled = True
            self._client = None

    def get(self, key: str) -> bytes | None:
        if self._disabled or self._client is None:
            return None
        try:
            return self._client.get(key)
        except redis.RedisError as e:
            logger.warning("redis GET failed for %s: %s", key, e)
            return None

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        if self._disabled or self._client is None:
            return
        try:
            self._client.set(key, value, ex=ttl_seconds)
        except redis.RedisError as e:
            logger.warning("redis SET failed for %s: %s", key, e)

    def setnx(self, key: str, value: bytes, ttl_seconds: int) -> bool:
        """Atomic SET if NOT EXISTS with TTL. Used as a cross-worker lock so
        only one gunicorn worker fans out to slow upstreams (Gamma, Kalshi)
        per refresh interval. Returns ``True`` when the lock was acquired."""
        if self._disabled or self._client is None:
            return True  # cache offline — caller proceeds directly
        try:
            return bool(self._client.set(key, value, ex=ttl_seconds, nx=True))
        except redis.RedisError as e:
            logger.warning("redis SETNX failed for %s: %s", key, e)
            return True  # fail open

    @property
    def enabled(self) -> bool:
        return not self._disabled


class NullCache:
    """Cache that does nothing — used in tests and as a safe default."""

    def get(self, key: str) -> bytes | None:
        return None

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        return None

    def setnx(self, key: str, value: bytes, ttl_seconds: int) -> bool:
        return True  # always claim the lock — single-process semantics

    @property
    def enabled(self) -> bool:
        return False
