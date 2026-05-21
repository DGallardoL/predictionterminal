"""SETNX-based distributed lock / leader-election helper.

Centralises the ``SET key value NX EX ttl`` + atomic compare-and-delete /
compare-and-renew pattern that was previously inlined in at least four
places (arb engine autostart, CLOB cache refresh, crypto5min sampler,
gdelt-news prewarm). Each call-site re-implemented the unique-token
fingerprint + Lua release script inconsistently; this module unifies it.

Design notes
------------
- ``SET key value NX EX ttl`` is the atomic primitive Redis exposes for
  this pattern. It is preferred over the legacy ``SETNX`` + separate
  ``EXPIRE`` because the latter is non-atomic (a crash between the two
  commands leaves the key without a TTL).
- ``release()`` MUST only delete the key if we still own it. Otherwise a
  process whose lock already expired could DEL the lock that a newer
  leader just acquired. The standard solution is a Lua script that
  performs ``GET == owner_id`` then ``DEL`` in one atomic step (see
  https://redis.io/docs/latest/develop/use/patterns/distributed-locks/).
- ``renew()`` extends the TTL only when we still own the key (same Lua
  guard). Renewal allows long-running leaders to outlive the initial TTL
  without giving up leadership.
- ``owner_id`` defaults to ``hostname:pid:<random8>`` so it is unique
  across machines, processes, and re-acquires within the same process.

Tests live at ``tests/test_redis_lock.py`` and use a tiny in-memory
fake redis client (no external dep) that implements ``set/get/delete``
+ ``eval`` semantics needed by this module.
"""

from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


# Lua: only delete the key if its current value matches the token we
# supplied. Returns 1 on delete, 0 on no-op. Atomic on the Redis server.
_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

# Lua: only extend the TTL if we still own the key. Returns 1 on
# success, 0 on no-op (key gone or owned by someone else).
_RENEW_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
else
    return 0
end
"""


def _default_owner_id() -> str:
    """Hostname:pid:<random8> — collision-free across machines/forks."""
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _to_str(value: Any) -> str:
    """Normalise a redis return value (bytes or str) to a str, or '' if None."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return ""
    return str(value)


class RedisLock:
    """A single SETNX-style distributed lock.

    Use as a context manager (``with lock: ...``) — ``__enter__`` calls
    ``acquire`` and ``__exit__`` calls ``release`` unconditionally. The
    enter returns the lock itself; callers must check ``lock.acquired``
    if they need to know whether they actually won the election.

    Parameters
    ----------
    redis_client : Any
        A ``redis.Redis`` client (or anything with ``set``/``get``/
        ``delete``/``eval`` matching that signature). May be ``None`` in
        tests — every method then becomes a single-process no-op that
        claims success.
    key : str
        The lock key. Convention: ``pfm:lock:<feature>``.
    ttl_s : int, default 60
        Lock TTL in seconds. Pick a value > expected critical section
        duration; call ``renew()`` if you need longer.
    owner_id : str | None
        Unique token for this lock instance. Defaults to
        ``hostname:pid:<random8>``.
    """

    def __init__(
        self,
        redis_client: Any,
        key: str,
        *,
        ttl_s: int = 60,
        owner_id: str | None = None,
    ) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        if not isinstance(ttl_s, int) or ttl_s <= 0:
            raise ValueError("ttl_s must be a positive integer")
        self._redis = redis_client
        self.key = key
        self.ttl_s = ttl_s
        self.owner_id = owner_id or _default_owner_id()
        self.acquired: bool = False

    # ------------------------------------------------------------------
    # core API

    def acquire(self) -> bool:
        """Attempt to acquire the lock. Returns True on success.

        Idempotent within a single instance: re-calling after a
        successful acquire returns True without contacting Redis. To
        re-acquire after release, create a new ``RedisLock``.
        """
        if self.acquired:
            return True
        if self._redis is None:
            # No redis backend — single-process semantics. We always win.
            self.acquired = True
            return True
        try:
            ok = bool(
                self._redis.set(
                    self.key,
                    self.owner_id,
                    nx=True,
                    ex=self.ttl_s,
                )
            )
        except Exception as e:
            logger.warning("RedisLock acquire failed for %s: %s", self.key, e)
            # Fail open — caller can decide based on the False return
            # what to do; we do NOT silently claim the lock on Redis
            # errors because that could yield split-brain leadership.
            return False
        self.acquired = ok
        return ok

    def release(self) -> bool:
        """Release the lock if (and only if) we still own it.

        Returns True on a successful release, False if the key was
        already gone or owned by someone else (e.g. our TTL expired and
        another worker re-acquired it).
        """
        if not self.acquired:
            return False
        if self._redis is None:
            self.acquired = False
            return True
        try:
            res = self._redis.eval(_RELEASE_LUA, 1, self.key, self.owner_id)
        except Exception as e:
            logger.warning("RedisLock release failed for %s: %s", self.key, e)
            return False
        finally:
            # Always mark this instance as no-longer-owner so future
            # release() calls become no-ops even if Redis is flaky.
            self.acquired = False
        return int(res or 0) == 1

    def renew(self) -> bool:
        """Extend the lock TTL by ``ttl_s`` seconds (atomic).

        Returns True if we still own the lock and TTL was extended,
        False otherwise (someone else owns it, or it has expired).
        """
        if not self.acquired:
            return False
        if self._redis is None:
            return True
        try:
            res = self._redis.eval(_RENEW_LUA, 1, self.key, self.owner_id, str(self.ttl_s))
        except Exception as e:
            logger.warning("RedisLock renew failed for %s: %s", self.key, e)
            return False
        ok = int(res or 0) == 1
        if not ok:
            # We lost the lock (expired or stolen). Reflect that in
            # state so subsequent release() is a no-op.
            self.acquired = False
        return ok

    def is_held_by_me(self) -> bool:
        """Return True iff the key in Redis currently equals our owner_id.

        Cheap GET; does not modify state. Useful for diagnostics or to
        gate work that should only run as the leader.
        """
        if not self.acquired:
            return False
        if self._redis is None:
            return True
        try:
            current = _to_str(self._redis.get(self.key))
        except Exception as e:
            logger.warning("RedisLock get failed for %s: %s", self.key, e)
            return False
        return current == self.owner_id

    # ------------------------------------------------------------------
    # context manager

    def __enter__(self) -> RedisLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.release()
        except Exception:  # pragma: no cover — defensive
            logger.exception("RedisLock __exit__ release raised")


@contextmanager
def leader_election(
    redis_client: Any,
    lock_name: str,
    *,
    ttl_s: int = 60,
    owner_id: str | None = None,
) -> Iterator[bool]:
    """Yields True if this process is the leader, False otherwise.

    Auto-releases the lock on exit if we acquired it. Designed for the
    "exactly one worker fans out" pattern, e.g.::

        with leader_election(redis, "pfm:lock:arb-engine", ttl_s=120) as is_leader:
            if is_leader:
                _start_arb_engine()
            else:
                logger.info("arb engine: leader held elsewhere")

    Note ``yield`` is OUTSIDE the try/finally on a single line so callers
    who suppress exceptions still get release().
    """
    lock = RedisLock(redis_client, lock_name, ttl_s=ttl_s, owner_id=owner_id)
    acquired = lock.acquire()
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()


__all__ = ["RedisLock", "leader_election"]


# Re-exported for tests + power users that want to script their own
# locking variants. Considered semi-private.
_RELEASE_LUA_SCRIPT = _RELEASE_LUA
_RENEW_LUA_SCRIPT = _RENEW_LUA


def _now() -> float:  # pragma: no cover — trivial wrapper, swap point for tests
    return time.time()
