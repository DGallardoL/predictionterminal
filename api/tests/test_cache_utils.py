"""Tests for ``pfm.cache_utils`` — TerminalCache, get_cache, cached."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from pfm.cache_utils import TerminalCache, cached, get_cache, reset_caches


@pytest.fixture(autouse=True)
def _isolate_named_instances() -> Iterator[None]:
    """Each test starts and ends with an empty named-instance registry."""
    reset_caches()
    yield
    reset_caches()


# ---------------------------------------------------------------------------
# Core get/set semantics
# ---------------------------------------------------------------------------


class TestGetSet:
    def test_set_then_get_returns_value(self) -> None:
        cache = TerminalCache(default_ttl=60)
        cache.set("k", {"v": 1})
        assert cache.get("k") == {"v": 1}

    def test_get_missing_returns_none(self) -> None:
        cache = TerminalCache(default_ttl=60)
        assert cache.get("never-set") is None

    def test_tuple_keys_work(self) -> None:
        cache = TerminalCache(default_ttl=60)
        cache.set(("slug", 90), "payload")
        assert cache.get(("slug", 90)) == "payload"

    def test_set_overwrites_existing(self) -> None:
        cache = TerminalCache(default_ttl=60)
        cache.set("k", "v1")
        cache.set("k", "v2")
        assert cache.get("k") == "v2"

    def test_external_store_is_shared(self) -> None:
        """Backing dict supplied by caller stays in sync with the cache."""
        store: dict = {}
        cache = TerminalCache(default_ttl=60, store=store)
        cache.set("a", 1)
        # Same dict, same entry.
        assert "a" in store
        assert store["a"][1] == 1  # (expiry, value) shape
        store.clear()
        assert cache.get("a") is None


# ---------------------------------------------------------------------------
# TTL expiration (with mocked time)
# ---------------------------------------------------------------------------


class TestTTLExpiration:
    def test_entry_expires_after_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [1_000_000.0]
        monkeypatch.setattr("pfm.cache_utils.time.time", lambda: now[0])

        cache = TerminalCache(default_ttl=10)
        cache.set("k", "v")

        # Within window — still alive.
        now[0] += 5
        assert cache.get("k") == "v"

        # Past window — expired and removed.
        now[0] += 10
        assert cache.get("k") is None

    def test_per_call_ttl_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [2_000_000.0]
        monkeypatch.setattr("pfm.cache_utils.time.time", lambda: now[0])

        cache = TerminalCache(default_ttl=10)
        cache.set("short", "x")  # default 10s
        cache.set("long", "y", ttl=100)  # explicit 100s

        now[0] += 50
        assert cache.get("short") is None
        assert cache.get("long") == "y"

    def test_expired_entry_is_purged_on_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [3_000.0]
        monkeypatch.setattr("pfm.cache_utils.time.time", lambda: now[0])

        store: dict = {}
        cache = TerminalCache(default_ttl=5, store=store)
        cache.set("k", 1)
        assert "k" in store
        now[0] += 10
        cache.get("k")  # triggers eviction
        assert "k" not in store


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_writes_do_not_corrupt_store(self) -> None:
        cache = TerminalCache(default_ttl=60)
        n_threads = 16
        n_writes = 200

        def worker(tid: int) -> None:
            for i in range(n_writes):
                cache.set((tid, i), f"v-{tid}-{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every (tid, i) pair must have been recorded exactly.
        for tid in range(n_threads):
            for i in range(n_writes):
                assert cache.get((tid, i)) == f"v-{tid}-{i}"

        assert cache.stats()["size"] == n_threads * n_writes

    def test_concurrent_get_or_compute_does_not_deadlock(self) -> None:
        """Lock is reentrant; no deadlock under contention."""
        cache = TerminalCache(default_ttl=60)
        results: list[int] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            v = cache.get_or_compute(f"key-{i % 8}", lambda i=i: i)
            with lock:
                results.append(v)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(64)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert all(not t.is_alive() for t in threads)
        assert len(results) == 64


# ---------------------------------------------------------------------------
# get_or_compute
# ---------------------------------------------------------------------------


class TestGetOrCompute:
    def test_computes_only_on_first_miss(self) -> None:
        cache = TerminalCache(default_ttl=60)
        calls = {"n": 0}

        def expensive() -> str:
            calls["n"] += 1
            return "answer"

        assert cache.get_or_compute("k", expensive) == "answer"
        assert cache.get_or_compute("k", expensive) == "answer"
        assert cache.get_or_compute("k", expensive) == "answer"
        assert calls["n"] == 1

    def test_recomputes_after_expiry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [10_000.0]
        monkeypatch.setattr("pfm.cache_utils.time.time", lambda: now[0])

        cache = TerminalCache(default_ttl=5)
        calls = {"n": 0}

        def producer() -> int:
            calls["n"] += 1
            return calls["n"]

        cache.get_or_compute("k", producer)
        cache.get_or_compute("k", producer)
        assert calls["n"] == 1
        now[0] += 10
        cache.get_or_compute("k", producer)
        assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


class TestCachedDecorator:
    def test_decorator_caches_by_args(self) -> None:
        calls = {"n": 0}

        @cached(namespace="t-decorator-args", ttl=60)
        def slow_double(x: int) -> int:
            calls["n"] += 1
            return x * 2

        assert slow_double(3) == 6
        assert slow_double(3) == 6
        assert calls["n"] == 1
        # Different arg → fresh compute.
        assert slow_double(4) == 8
        assert calls["n"] == 2

    def test_decorator_respects_kwargs(self) -> None:
        calls = {"n": 0}

        @cached(namespace="t-decorator-kw", ttl=60)
        def f(*, name: str) -> str:
            calls["n"] += 1
            return name.upper()

        assert f(name="alice") == "ALICE"
        assert f(name="alice") == "ALICE"
        assert f(name="bob") == "BOB"
        assert calls["n"] == 2

    def test_decorator_custom_key_fn(self) -> None:
        calls = {"n": 0}

        @cached(
            namespace="t-decorator-keyfn",
            ttl=60,
            key_fn=lambda *a, **kw: kw.get("slug", ""),
        )
        def fetch(slug: str, request_id: int) -> str:
            calls["n"] += 1
            return f"data-{slug}"

        # Different request_id but same slug → cache hit.
        assert fetch(slug="abc", request_id=1) == "data-abc"
        assert fetch(slug="abc", request_id=2) == "data-abc"
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_hits_and_misses_counted(self) -> None:
        cache = TerminalCache(default_ttl=60)
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("a")  # hit
        cache.get("missing")  # miss
        cache.get("missing-2")  # miss

        s = cache.stats()
        assert s["hits"] == 2
        assert s["misses"] == 2
        assert s["size"] == 1

    def test_clear_resets_counters(self) -> None:
        cache = TerminalCache(default_ttl=60)
        cache.set("a", 1)
        cache.get("a")
        cache.get("missing")
        cache.clear()
        s = cache.stats()
        assert s == {"hits": 0, "misses": 0, "size": 0}

    def test_expired_entry_counts_as_miss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [50.0]
        monkeypatch.setattr("pfm.cache_utils.time.time", lambda: now[0])
        cache = TerminalCache(default_ttl=5)
        cache.set("k", 1)
        now[0] += 10
        assert cache.get("k") is None
        assert cache.stats()["misses"] == 1
        assert cache.stats()["hits"] == 0


# ---------------------------------------------------------------------------
# Named instances
# ---------------------------------------------------------------------------


class TestNamedInstances:
    def test_get_cache_returns_same_instance(self) -> None:
        a = get_cache("foo", ttl=100)
        b = get_cache("foo", ttl=999)
        assert a is b

    def test_distinct_namespaces_isolated(self) -> None:
        a = get_cache("ns-a")
        b = get_cache("ns-b")
        assert a is not b
        a.set("k", 1)
        assert b.get("k") is None

    def test_reset_caches_clears_all(self) -> None:
        a = get_cache("rs-a")
        b = get_cache("rs-b")
        a.set("k", 1)
        b.set("k", 2)
        reset_caches()
        # Contents are dropped but singleton identity is preserved so
        # module-level captures (``_FOO_CACHE = get_cache("foo")``) keep
        # referring to the same TerminalCache after reset. Dropping the
        # singletons would orphan those captures and leak state across
        # tests.
        a2 = get_cache("rs-a")
        assert a2 is a
        assert a2.get("k") is None
        assert b.get("k") is None


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------


class TestPrune:
    def test_prune_drops_expired_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [0.0]
        monkeypatch.setattr("pfm.cache_utils.time.time", lambda: now[0])
        cache = TerminalCache(default_ttl=5)
        cache.set("a", 1, ttl=5)
        cache.set("b", 2, ttl=100)
        now[0] += 10
        dropped = cache.prune()
        assert dropped == 1
        assert cache.get("b") == 2
