"""Integration-style tests for ``pfm.redis_lock`` — full leader-election scenarios.

These tests are deliberately distinct from the unit tests in
``tests/test_redis_lock.py`` (T22). The unit-test suite exercises individual
methods (``acquire``, ``release``, ``renew``, ``is_held_by_me``,
``leader_election`` happy-path) against a tiny hand-rolled fake. THIS file
instead exercises **end-to-end scenarios** that span multiple leaders,
multiple threads, and TTL expiry — the situations the lock is meant to
protect against in production (arb-engine autostart, gdelt-news prewarm,
crypto5min sampler, CLOB cache refresh). We use ``fakeredis`` (with ``lupa``
backing Lua ``EVAL``) so the same ``SET NX EX`` + Lua compare-and-delete
primitives the prod path uses are exercised by the test.

If either ``fakeredis`` or ``pfm.redis_lock`` is unavailable, every test in
this module skips with a clear reason — so the file is runnable even before
T22 lands.

Scenarios covered:

1. **Process restart** — A holds with 60 s TTL, dies (we simulate TTL by
   fast-forwarding the fakeredis ``expireat``); B then acquires.
2. **Failover mid-task** — A acquires, work proceeds, TTL elapses while A is
   still "doing work"; B takes over. We assert no double-execution beyond
   the TTL window: A's ``is_held_by_me`` flips False, and any guarded
   side-effect (counter increment) only fires once per leader.
3. **Multiple-leader prevention** — 50 threads race for the same lock;
   exactly one wins. Winner releases; the next acquire wave returns a single
   new winner.
4. **Renew during long work** — process holds with ttl=30 s, renews every
   10 simulated seconds for 90 s of "work", finishes still holding the lock.
5. **Wrong-owner release attempt** — A holds, B (different ``owner_id`` on
   the SAME key) calls release → returns False, key still belongs to A.
6. **Namespace isolation** — ``pfm:lock:arb-engine`` and
   ``pfm:lock:news-pipeline`` do not interfere with each other.
7. **Reentrant attempt by same owner instance** — the same ``RedisLock``
   instance's second ``acquire()`` returns True without re-contacting Redis
   (idempotent within a single instance). Note: a *different* ``RedisLock``
   with the SAME ``owner_id`` is NOT supported as reentrant — Redis ``SET
   NX`` will refuse the second SET because the key already exists. This
   matches the documented module semantics ("Idempotent within a single
   instance ... To re-acquire after release, create a new RedisLock").
"""

from __future__ import annotations

import threading
import time

import pytest

# ---------------------------------------------------------------------------
# Soft imports so the file is runnable before T22 lands and before fakeredis
# is in the dev-deps lockfile.
# ---------------------------------------------------------------------------

try:
    import fakeredis  # type: ignore

    _HAS_FAKEREDIS = True
except Exception:  # pragma: no cover — import guard
    fakeredis = None  # type: ignore
    _HAS_FAKEREDIS = False

try:
    import lupa  # noqa: F401  type: ignore

    _HAS_LUPA = True
except Exception:  # pragma: no cover — import guard
    _HAS_LUPA = False

try:
    from pfm.redis_lock import RedisLock, leader_election

    _HAS_REDIS_LOCK = True
except Exception:  # pragma: no cover — import guard
    RedisLock = None  # type: ignore
    leader_election = None  # type: ignore
    _HAS_REDIS_LOCK = False


pytestmark = [
    pytest.mark.skipif(
        not _HAS_FAKEREDIS,
        reason=(
            "fakeredis not installed — run "
            "`pip install fakeredis lupa` in api/.venv to enable lock integration tests"
        ),
    ),
    pytest.mark.skipif(
        not _HAS_LUPA,
        reason=(
            "lupa not installed — fakeredis cannot execute Lua EVAL without it; "
            "run `pip install lupa`"
        ),
    ),
    pytest.mark.skipif(
        not _HAS_REDIS_LOCK,
        reason="pfm.redis_lock not importable yet (T22 not landed)",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server():
    """One shared FakeServer per test → multiple clients see the same db."""
    return fakeredis.FakeServer()


@pytest.fixture
def make_client(server):
    """Factory that produces independent FakeRedis clients sharing one server.

    Mirrors the prod topology where multiple worker processes / threads each
    own a ``redis.Redis`` connection but point at the same Redis instance.
    """

    def _make():
        return fakeredis.FakeRedis(server=server)

    return _make


def _force_ttl_expiry(server, key: str) -> None:
    """Simulate "TTL elapsed" by rewriting the fakeredis ``expireat`` to the past.

    This is the deterministic equivalent of ``time.sleep(ttl_s + 1)`` but
    free. Validates that the lock's ``SET NX EX`` + Lua guard correctly
    refuse to act on a key that has lapsed.
    """
    db = server.dbs[0]
    raw = key.encode("utf-8") if isinstance(key, str) else key
    item = db.get(raw)
    if item is not None:
        item.expireat = time.time() - 1.0


# ---------------------------------------------------------------------------
# Scenario 1 — process restart: TTL must allow re-acquisition
# ---------------------------------------------------------------------------


def test_process_restart_after_ttl_other_process_acquires(server, make_client):
    """Process A acquires; we kill it (no release); after TTL, B succeeds."""
    a = RedisLock(make_client(), "pfm:lock:restart-test", ttl_s=60, owner_id="proc-A")
    b = RedisLock(make_client(), "pfm:lock:restart-test", ttl_s=60, owner_id="proc-B")

    assert a.acquire() is True
    # While A is alive, B cannot get the lock.
    assert b.acquire() is False

    # Simulate "process A killed" — no release() call. Force TTL elapsed.
    _force_ttl_expiry(server, "pfm:lock:restart-test")

    # Now B (and only B) can acquire.
    assert b.acquire() is True
    assert b.is_held_by_me() is True

    # A still thinks it holds the lock (its in-memory flag), but Redis says
    # otherwise. This is the documented "fail-open" behaviour: A's
    # is_held_by_me() must reflect the truth.
    assert a.is_held_by_me() is False


# ---------------------------------------------------------------------------
# Scenario 2 — failover mid-task: no double-execution beyond TTL
# ---------------------------------------------------------------------------


def test_failover_during_long_task_no_double_execution(server, make_client):
    """A holds, TTL lapses, B takes over. Guarded work runs once per leader.

    "No double-execution beyond TTL" here means: while only ONE of (A, B)
    is the live leader at any wall-clock instant, the guarded counter
    increments at most once per *leadership tenure*. We pin this by
    checking ``is_held_by_me`` *before* each increment — exactly the
    pattern the gdelt-news prewarm uses.
    """
    a = RedisLock(make_client(), "pfm:lock:failover", ttl_s=30, owner_id="A")
    b = RedisLock(make_client(), "pfm:lock:failover", ttl_s=30, owner_id="B")

    counter = {"runs_by_A": 0, "runs_by_B": 0}

    # --- A's tenure ---
    assert a.acquire() is True
    if a.is_held_by_me():
        counter["runs_by_A"] += 1

    # A starts a long task. Halfway through, A's TTL lapses (network
    # partition, GC pause, …). B notices and takes over.
    _force_ttl_expiry(server, "pfm:lock:failover")

    # B acquires; A's guard should now correctly refuse further work.
    assert b.acquire() is True
    if a.is_held_by_me():  # must be False — A's leadership ended
        counter["runs_by_A"] += 1  # pragma: no cover — should not execute

    if b.is_held_by_me():
        counter["runs_by_B"] += 1

    assert counter == {"runs_by_A": 1, "runs_by_B": 1}

    # Belt-and-braces: A trying to release now must fail (key belongs to B).
    assert a.release() is False
    # B still owns the key.
    assert b.is_held_by_me() is True


# ---------------------------------------------------------------------------
# Scenario 3 — multiple-leader prevention under 50-thread contention
# ---------------------------------------------------------------------------


def test_fifty_threads_race_exactly_one_winner(server, make_client):
    """50 threads call acquire on the same key → exactly one returns True.

    After the winner releases, a second wave produces exactly one *new*
    winner. This is the core leader-election invariant.
    """
    key = "pfm:lock:race"
    wave1_wins: list[str] = []
    wave2_wins: list[str] = []

    def worker(wave: list[str], oid: str) -> None:
        lock = RedisLock(make_client(), key, ttl_s=60, owner_id=oid)
        if lock.acquire():
            wave.append(oid)

    threads = [threading.Thread(target=worker, args=(wave1_wins, f"w1-{i}")) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(wave1_wins) == 1, (
        f"expected exactly 1 leader from 50-thread race, got {len(wave1_wins)}: {wave1_wins}"
    )

    # Winner releases. NOTE: we cannot use the original RedisLock instance
    # from inside the thread (it's gone), so we mimic the release using the
    # same owner_id — exactly what a restarted leader handoff would do.
    winner_oid = wave1_wins[0]
    releaser = RedisLock(make_client(), key, ttl_s=60, owner_id=winner_oid)
    releaser.acquired = True  # type: ignore[attr-defined]  reflect prior ownership
    assert releaser.release() is True

    # Wave 2: another 50-thread race. Exactly one new winner.
    threads2 = [threading.Thread(target=worker, args=(wave2_wins, f"w2-{i}")) for i in range(50)]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join()

    assert len(wave2_wins) == 1, (
        f"expected exactly 1 leader in second wave, got {len(wave2_wins)}: {wave2_wins}"
    )


# ---------------------------------------------------------------------------
# Scenario 4 — renew keeps long-running leader alive past initial TTL
# ---------------------------------------------------------------------------


def test_renew_during_long_work_keeps_lock_held(server, make_client):
    """30 s TTL, renewed three times across simulated 90 s of work."""
    lock = RedisLock(make_client(), "pfm:lock:renew", ttl_s=30, owner_id="long-runner")
    contender = RedisLock(make_client(), "pfm:lock:renew", ttl_s=30, owner_id="contender")

    assert lock.acquire() is True
    start = time.monotonic()

    # Simulate 90 s of work in 3 chunks of 30 simulated s. We do not actually
    # sleep — we just bump the expireat backwards to "drain" some TTL, then
    # renew(). After each renew the contender must still be locked out.
    for _ in range(3):
        # "Drain" the TTL down towards (but not past) expiry. We knock 25 s
        # off, leaving 5 s remaining — exactly the situation where a smart
        # leader renews preemptively.
        db = server.dbs[0]
        item = db.get(b"pfm:lock:renew")
        assert item is not None, "lock key disappeared mid-renew loop"
        item.expireat -= 25.0
        assert lock.renew() is True
        # Contender still cannot break in.
        assert contender.acquire() is False
        assert lock.is_held_by_me() is True

    elapsed = time.monotonic() - start
    # Elapsed wall-clock here is ~0 ms (we never slept), but the *logical*
    # work simulated > TTL. The point: the lock survived.
    assert lock.is_held_by_me() is True
    assert lock.release() is True
    # Now the contender wins.
    assert contender.acquire() is True

    # Sanity: the test itself ran fast — we never actually slept 90 s.
    assert elapsed < 5.0


# ---------------------------------------------------------------------------
# Scenario 5 — wrong-owner release attempt is a no-op
# ---------------------------------------------------------------------------


def test_wrong_owner_release_fails_original_owner_retains(server, make_client):
    """Imposter B with different owner_id cannot release A's lock."""
    a = RedisLock(make_client(), "pfm:lock:wrong-owner", ttl_s=60, owner_id="A")
    b = RedisLock(make_client(), "pfm:lock:wrong-owner", ttl_s=60, owner_id="B-imposter")

    assert a.acquire() is True
    assert b.acquire() is False  # B never won the lock

    # B fakes "having acquired" to coax release into calling Redis. The
    # Lua compare-and-delete must still refuse.
    b.acquired = True  # type: ignore[attr-defined]
    assert b.release() is False
    # After a failed release, b.acquired is reset to False (per module
    # contract: "Always mark this instance as no-longer-owner so future
    # release() calls become no-ops even if Redis is flaky").
    assert b.acquired is False

    # A still holds the lock — that's the invariant that matters.
    assert a.is_held_by_me() is True
    assert a.release() is True
    assert a.is_held_by_me() is False


# ---------------------------------------------------------------------------
# Scenario 6 — namespace isolation: different lock keys do not interfere
# ---------------------------------------------------------------------------


def test_namespace_isolation_arb_vs_news(server, make_client):
    """``pfm:lock:arb-engine`` and ``pfm:lock:news-pipeline`` are independent."""
    arb = RedisLock(make_client(), "pfm:lock:arb-engine", ttl_s=60, owner_id="arb-1")
    news = RedisLock(make_client(), "pfm:lock:news-pipeline", ttl_s=60, owner_id="news-1")

    assert arb.acquire() is True
    assert news.acquire() is True  # different key — must succeed
    assert arb.is_held_by_me() is True
    assert news.is_held_by_me() is True

    # Releasing one must NOT release the other.
    assert arb.release() is True
    assert news.is_held_by_me() is True
    assert news.release() is True


def test_leader_election_helper_isolated_per_lock_name(server, make_client):
    """The ``leader_election`` ctx-manager also respects namespace separation."""
    leader_flags: dict[str, bool] = {}

    with leader_election(make_client(), "pfm:lock:helper-a", ttl_s=30, owner_id="a") as is_a:
        leader_flags["a"] = is_a
        with leader_election(make_client(), "pfm:lock:helper-b", ttl_s=30, owner_id="b") as is_b:
            leader_flags["b"] = is_b

    # Both should have been leader of their respective lock.
    assert leader_flags == {"a": True, "b": True}

    # And after the ``with``-blocks exit, both keys must be released.
    client = make_client()
    assert client.get("pfm:lock:helper-a") is None
    assert client.get("pfm:lock:helper-b") is None


# ---------------------------------------------------------------------------
# Scenario 7 — reentrant acquire by the same owner instance
# ---------------------------------------------------------------------------


def test_reentrant_acquire_same_instance_returns_true_no_redis_roundtrip(server, make_client):
    """Same instance calling acquire() twice → returns True both times.

    Documented semantics chosen (per pfm.redis_lock docstring):
      "Idempotent within a single instance: re-calling after a successful
       acquire returns True without contacting Redis."

    So we verify the SECOND acquire does NOT cause a Redis SET. We pin this
    by snapshotting the key's TTL after the first acquire — if a second
    acquire had hit Redis with ``SET NX EX``, TTL would have been refreshed
    (or rejected). We instead expect TTL to be untouched.
    """
    client = make_client()
    lock = RedisLock(client, "pfm:lock:reentrant", ttl_s=60, owner_id="solo")

    assert lock.acquire() is True
    # Knock the TTL down so we'd notice a second SET (which would reset it).
    db = server.dbs[0]
    item = db.get(b"pfm:lock:reentrant")
    assert item is not None
    item.expireat -= 30.0  # ~30 s drained off
    ttl_after_first = client.ttl("pfm:lock:reentrant")

    # Second acquire — must be True (idempotent) AND must not refresh TTL.
    assert lock.acquire() is True
    ttl_after_second = client.ttl("pfm:lock:reentrant")
    assert ttl_after_second <= ttl_after_first, (
        "Reentrant acquire should NOT refresh TTL — module contract says "
        "the second call returns True without contacting Redis. Got TTL "
        f"{ttl_after_second} vs {ttl_after_first} before."
    )
    assert lock.is_held_by_me() is True
    assert lock.release() is True


def test_reentrant_different_instance_same_owner_is_not_supported(server, make_client):
    """A *different* RedisLock with the SAME owner_id is NOT reentrant.

    This pins the documented semantics: reentrancy is per-instance, not
    per-owner_id. If you spin up a fresh RedisLock with the same owner_id
    you will collide with your own outstanding lock (Redis ``SET NX`` will
    refuse) — at which point you should ``release()`` first, then re-acquire.
    """
    first = RedisLock(make_client(), "pfm:lock:reentrant-2", ttl_s=60, owner_id="dup")
    second = RedisLock(make_client(), "pfm:lock:reentrant-2", ttl_s=60, owner_id="dup")

    assert first.acquire() is True
    # Second instance can NOT acquire even though owner_id matches — Redis
    # SET NX doesn't care about token equality at acquire time.
    assert second.acquire() is False

    # But because the owner_id matches, ``second`` *could* release ``first``'s
    # lock — that's a footgun callers must be aware of. We pin it so the
    # behaviour is intentional, not accidental.
    second.acquired = True  # type: ignore[attr-defined]
    assert second.release() is True
    # Now ``first``'s internal flag is stale but Redis is empty.
    assert first.is_held_by_me() is False
