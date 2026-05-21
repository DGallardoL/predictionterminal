"""Tests for ``pfm.redis_lock``.

Runs with ``pytest tests/test_redis_lock.py -q --noconftest`` so we do
NOT rely on the project conftest (which imports FastAPI / DataFrames).

We ship a tiny in-memory ``FakeRedis`` here rather than depend on the
``fakeredis`` PyPI package (which is not installed in this repo's
.venv). The fake implements the subset of Redis semantics used by
``pfm.redis_lock``:

- ``set(key, value, nx=bool, ex=int)`` returning truthy on success
- ``get(key)`` returning bytes-or-None
- ``delete(key)`` returning count
- ``eval(script, numkeys, *args)`` recognising the release / renew Lua
  scripts shipped in pfm.redis_lock
- Manual ``advance_time(seconds)`` to deterministically expire keys

This is more than mocking: it's a behavioural reimplementation of the
exact Redis primitives the module under test relies on. That keeps the
tests honest while still being hermetic.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

# Make ``src/`` importable when running with --noconftest (no rootdir hook).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pfm.redis_lock import (
    _RELEASE_LUA_SCRIPT,
    _RENEW_LUA_SCRIPT,
    RedisLock,
    _default_owner_id,
    leader_election,
)

# ---------------------------------------------------------------------------
# In-memory fake redis


class FakeRedis:
    """Subset of redis.Redis sufficient for RedisLock tests."""

    def __init__(self) -> None:
        # key -> (value_bytes, expiry_monotonic_seconds_or_None)
        self._store: dict[str, tuple[bytes, float | None]] = {}
        self._now = 1000.0  # virtual clock; .advance_time() bumps it
        self._lock = threading.Lock()  # protects _store + _now under threading test
        self.set_calls = 0  # observability for race tests

    # ---- helpers -----------------------------------------------------

    def advance_time(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds
            # Lazily clear expired keys to keep state realistic.
            expired = [
                k for k, (_v, exp) in self._store.items() if exp is not None and exp <= self._now
            ]
            for k in expired:
                del self._store[k]

    def _get_live(self, key: str) -> bytes | None:
        v = self._store.get(key)
        if v is None:
            return None
        val, exp = v
        if exp is not None and exp <= self._now:
            del self._store[key]
            return None
        return val

    def ttl(self, key: str) -> int:
        """Return remaining TTL in whole seconds, or -2 if missing, -1 if no TTL."""
        with self._lock:
            v = self._store.get(key)
            if v is None:
                return -2
            _val, exp = v
            if exp is None:
                return -1
            if exp <= self._now:
                del self._store[key]
                return -2
            return int(exp - self._now)

    # ---- redis-client surface ---------------------------------------

    def set(
        self,
        key: str,
        value,
        *,
        nx: bool = False,
        ex: int | None = None,
        **_unused,
    ) -> bool | None:
        with self._lock:
            self.set_calls += 1
            existing = self._get_live(key)
            if nx and existing is not None:
                return None  # redis-py returns None on NX failure
            if isinstance(value, str):
                value_b = value.encode("utf-8")
            elif isinstance(value, bytes):
                value_b = value
            else:
                value_b = str(value).encode("utf-8")
            exp: float | None = (self._now + ex) if ex is not None else None
            self._store[key] = (value_b, exp)
            return True

    def get(self, key: str):
        with self._lock:
            return self._get_live(key)

    def delete(self, *keys: str) -> int:
        with self._lock:
            n = 0
            for k in keys:
                if k in self._store:
                    # Treat expired-but-still-present as gone.
                    if self._get_live(k) is None:
                        continue
                    del self._store[k]
                    n += 1
            return n

    def eval(self, script: str, numkeys: int, *args):
        """Interpret the two specific Lua scripts shipped by redis_lock.

        We don't run real Lua — we pattern-match the canonical scripts
        and execute their behaviour atomically under self._lock. This is
        the standard fakeredis approach for tightly-scoped tests.
        """
        keys = list(args[:numkeys])
        argv = list(args[numkeys:])
        with self._lock:
            if script == _RELEASE_LUA_SCRIPT:
                # if GET == ARGV[1] then DEL else 0
                key = keys[0]
                want = argv[0]
                if isinstance(want, str):
                    want_b = want.encode("utf-8")
                else:
                    want_b = want
                cur = self._get_live(key)
                if cur is None:
                    return 0
                if cur == want_b:
                    del self._store[key]
                    return 1
                return 0
            if script == _RENEW_LUA_SCRIPT:
                key = keys[0]
                want = argv[0]
                new_ttl = int(argv[1])
                if isinstance(want, str):
                    want_b = want.encode("utf-8")
                else:
                    want_b = want
                cur = self._get_live(key)
                if cur is None:
                    return 0
                if cur == want_b:
                    self._store[key] = (cur, self._now + new_ttl)
                    return 1
                return 0
            raise NotImplementedError(f"FakeRedis.eval: unknown script: {script[:40]!r}")


# ---------------------------------------------------------------------------
# fixtures


@pytest.fixture
def fake() -> FakeRedis:
    return FakeRedis()


# ---------------------------------------------------------------------------
# Tests


def test_default_owner_id_is_unique_per_call():
    a = _default_owner_id()
    b = _default_owner_id()
    assert a != b
    # Sanity: format hostname:pid:hex8 — at least two colons.
    assert a.count(":") >= 2
    assert b.count(":") >= 2


def test_init_rejects_empty_key(fake):
    with pytest.raises(ValueError):
        RedisLock(fake, "")
    with pytest.raises(ValueError):
        RedisLock(fake, "x", ttl_s=0)
    with pytest.raises(ValueError):
        RedisLock(fake, "x", ttl_s=-1)


def test_single_acquire_then_second_blocks(fake):
    lock_a = RedisLock(fake, "pfm:test:lock1", ttl_s=30)
    lock_b = RedisLock(fake, "pfm:test:lock1", ttl_s=30)
    assert lock_a.acquire() is True
    assert lock_a.acquired is True
    # Second process tries -> blocked.
    assert lock_b.acquire() is False
    assert lock_b.acquired is False


def test_acquire_is_idempotent_within_instance(fake):
    lock = RedisLock(fake, "pfm:test:idem", ttl_s=10)
    assert lock.acquire() is True
    # Should not contact redis again on second call.
    before = fake.set_calls
    assert lock.acquire() is True
    assert fake.set_calls == before  # no second SET


def test_ttl_expiry_releases_implicitly(fake):
    lock_a = RedisLock(fake, "pfm:test:ttl", ttl_s=5)
    assert lock_a.acquire() is True
    # Fast-forward past TTL.
    fake.advance_time(6)
    # New process should now be able to acquire.
    lock_b = RedisLock(fake, "pfm:test:ttl", ttl_s=5)
    assert lock_b.acquire() is True


def test_release_returns_true_when_owner(fake):
    lock = RedisLock(fake, "pfm:test:rel-ok", ttl_s=30)
    assert lock.acquire() is True
    assert lock.release() is True
    assert lock.acquired is False
    # Key gone.
    assert fake.get("pfm:test:rel-ok") is None


def test_release_by_non_owner_fails(fake):
    # Process A acquires.
    lock_a = RedisLock(fake, "pfm:test:rel-bad", ttl_s=30, owner_id="alice")
    assert lock_a.acquire() is True

    # Process B fabricates ownership and attempts release; it should
    # NOT be able to delete the key. We simulate this by constructing
    # a lock with a different owner_id and forcing .acquired=True (the
    # only realistic way to even *try* to call release() under our API).
    lock_b = RedisLock(fake, "pfm:test:rel-bad", ttl_s=30, owner_id="bob")
    lock_b.acquired = True  # pretend we own it
    assert lock_b.release() is False
    # The legitimate owner's key is still there.
    assert fake.get("pfm:test:rel-bad") == b"alice"

    # Cleanup: real owner can still release.
    assert lock_a.release() is True


def test_release_no_op_if_never_acquired(fake):
    lock = RedisLock(fake, "pfm:test:rel-noop", ttl_s=10)
    # Never acquired.
    assert lock.release() is False


def test_renew_extends_ttl(fake):
    lock = RedisLock(fake, "pfm:test:renew", ttl_s=10)
    assert lock.acquire() is True
    ttl_initial = fake.ttl("pfm:test:renew")
    assert ttl_initial == 10
    # Time passes.
    fake.advance_time(7)
    assert fake.ttl("pfm:test:renew") == 3
    # Renew resets the TTL to ttl_s (10).
    assert lock.renew() is True
    ttl_after = fake.ttl("pfm:test:renew")
    assert ttl_after == 10
    assert ttl_after > 3  # strictly increased


def test_renew_by_non_owner_fails(fake):
    lock_a = RedisLock(fake, "pfm:test:renew-bad", ttl_s=10, owner_id="alice")
    assert lock_a.acquire() is True
    lock_b = RedisLock(fake, "pfm:test:renew-bad", ttl_s=10, owner_id="mallory")
    lock_b.acquired = True
    assert lock_b.renew() is False
    # alice's TTL was NOT extended (still ~10).
    assert fake.ttl("pfm:test:renew-bad") == 10
    # mallory's renew should have flipped her instance state.
    assert lock_b.acquired is False


def test_renew_after_expiry_returns_false(fake):
    lock = RedisLock(fake, "pfm:test:renew-expired", ttl_s=5)
    assert lock.acquire() is True
    fake.advance_time(6)  # past TTL — Redis evicted the key
    assert lock.renew() is False
    # State was reconciled.
    assert lock.acquired is False


def test_is_held_by_me_truth_table(fake):
    lock = RedisLock(fake, "pfm:test:held", ttl_s=10, owner_id="me")
    # Not acquired -> False.
    assert lock.is_held_by_me() is False
    lock.acquire()
    assert lock.is_held_by_me() is True
    # Somebody else stomps the key (only possible if our TTL expired
    # and a new leader took over — simulate by force-overwriting).
    fake._store["pfm:test:held"] = (b"someone-else", fake._now + 5)
    assert lock.is_held_by_me() is False


def test_context_manager_auto_releases(fake):
    key = "pfm:test:ctx"
    with RedisLock(fake, key, ttl_s=30) as lock:
        assert lock.acquired is True
        assert fake.get(key) is not None
    # On exit, key released.
    assert fake.get(key) is None


def test_context_manager_releases_on_exception(fake):
    key = "pfm:test:ctx-exc"
    with pytest.raises(RuntimeError), RedisLock(fake, key, ttl_s=30):
        assert fake.get(key) is not None
        raise RuntimeError("boom")
    assert fake.get(key) is None


def test_leader_election_yields_true_for_winner(fake):
    with leader_election(fake, "pfm:test:le", ttl_s=10) as is_leader:
        assert is_leader is True
        assert fake.get("pfm:test:le") is not None
    # Auto-release on exit.
    assert fake.get("pfm:test:le") is None


def test_leader_election_yields_false_for_loser(fake):
    # First call wins.
    with leader_election(fake, "pfm:test:le2", ttl_s=30) as winner:
        assert winner is True
        # Inside the winner's region, second caller MUST get False.
        with leader_election(fake, "pfm:test:le2", ttl_s=30) as loser:
            assert loser is False
        # Loser must NOT have released the winner's key on exit.
        assert fake.get("pfm:test:le2") is not None
    # Now that winner has released, key is gone.
    assert fake.get("pfm:test:le2") is None


def test_concurrent_acquire_exactly_one_wins(fake):
    """10 threads race; exactly one must come out with .acquired=True."""
    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(10)

    def worker():
        lock = RedisLock(fake, "pfm:test:race", ttl_s=30)
        barrier.wait()  # release all at once for max contention
        ok = lock.acquire()
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for r in results if r) == 1
    assert sum(1 for r in results if not r) == 9


def test_release_then_second_can_acquire(fake):
    a = RedisLock(fake, "pfm:test:reacq", ttl_s=30)
    b = RedisLock(fake, "pfm:test:reacq", ttl_s=30)
    assert a.acquire() is True
    assert b.acquire() is False
    assert a.release() is True
    assert b.acquire() is True
    assert b.release() is True


def test_none_client_single_process_mode():
    """With redis_client=None, the lock behaves as a single-process no-op."""
    lock = RedisLock(None, "pfm:test:nop", ttl_s=5)
    assert lock.acquire() is True
    assert lock.is_held_by_me() is True
    assert lock.renew() is True
    assert lock.release() is True


def test_acquire_failure_does_not_set_acquired_on_redis_error():
    """If redis.set raises, acquire() returns False and state stays clean."""

    class FlakyClient:
        def set(self, *a, **kw):
            raise ConnectionError("redis down")

    lock = RedisLock(FlakyClient(), "pfm:test:flaky", ttl_s=5)
    assert lock.acquire() is False
    assert lock.acquired is False
    # Subsequent release must be a no-op (not raise, not return True).
    assert lock.release() is False


def test_release_swallows_redis_errors():
    class FlakyAfterAcquire:
        def __init__(self):
            self.set_ok = True

        def set(self, *a, **kw):
            return True  # acquire succeeds

        def eval(self, *a, **kw):
            raise ConnectionError("redis down mid-release")

        def get(self, key):
            return None

    lock = RedisLock(FlakyAfterAcquire(), "pfm:test:flaky-rel", ttl_s=5)
    assert lock.acquire() is True
    # Release must NOT propagate the error; returns False but clears state.
    assert lock.release() is False
    assert lock.acquired is False


def test_custom_owner_id_round_trips(fake):
    lock = RedisLock(fake, "pfm:test:custom-owner", ttl_s=10, owner_id="my-custom-id")
    assert lock.acquire() is True
    assert fake.get("pfm:test:custom-owner") == b"my-custom-id"
    assert lock.is_held_by_me() is True


def test_leader_election_with_explicit_owner_id(fake):
    with leader_election(fake, "pfm:test:le-owner", ttl_s=10, owner_id="agent-7") as ok:
        assert ok is True
        assert fake.get("pfm:test:le-owner") == b"agent-7"
