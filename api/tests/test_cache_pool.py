"""Tests for ``pfm.cache_pool.CachePool``.

Run with::

    pytest tests/test_cache_pool.py -q --noconftest

The tests deliberately avoid the project ``conftest.py`` because that
pulls in the full app fixture chain (Polymarket mocks, factor catalog
warmup, Redis attach). A cache test should not require any of that.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make ``pfm`` importable without conftest sys.path tweaks.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pfm.cache_pool import _L2_MAGIC, CachePool

# ---------------------------------------------------------------------------
# Helpers — mock Redis backend
# ---------------------------------------------------------------------------


class _MockRedis:
    """Minimal in-memory Redis stand-in matching the ``RedisCache`` protocol."""

    def __init__(self, *, broken: bool = False) -> None:
        self._d: dict[str, bytes] = {}
        self._broken = broken
        self.get_calls = 0
        self.set_calls = 0
        self.enabled = not broken

    def get(self, key: str) -> bytes | None:
        if self._broken:
            raise ConnectionError("mock redis offline")
        self.get_calls += 1
        return self._d.get(key)

    def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        if self._broken:
            raise ConnectionError("mock redis offline")
        self.set_calls += 1
        self._d[key] = value

    def delete(self, key: str) -> None:
        if self._broken:
            return
        self._d.pop(key, None)

    def scan_iter(self, *, match: str = "*"):
        # Naive glob — only supports trailing ``*`` which is all we use.
        if match.endswith("*"):
            prefix = match[:-1]
            return [k for k in list(self._d) if k.startswith(prefix)]
        return [k for k in list(self._d) if k == match]


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_set_get_basic():
    pool = CachePool(namespace="test")
    pool.set("a", 1, ttl=60)
    assert pool.get("a") == 1


def test_get_default_on_miss():
    pool = CachePool(namespace="test")
    assert pool.get("missing") is None
    assert pool.get("missing", default="fallback") == "fallback"


def test_delete_removes_entry():
    pool = CachePool(namespace="test")
    pool.set("a", 1, ttl=60)
    pool.delete("a")
    assert pool.get("a") is None
    # delete on missing key is a no-op
    pool.delete("never-set")


def test_namespace_validation():
    with pytest.raises(ValueError):
        CachePool(namespace="")
    with pytest.raises(ValueError):
        CachePool(namespace="bad:colon")


# ---------------------------------------------------------------------------
# TTL expiry + heap eviction
# ---------------------------------------------------------------------------


def test_ttl_expiry():
    pool = CachePool(namespace="test")
    pool.set("a", "value", ttl=0)
    # ttl=0 means immediate expiry; one tick later it must be gone.
    time.sleep(0.01)
    assert pool.get("a") is None


def test_l1_maxsize_evicts_when_full():
    pool = CachePool(namespace="test", l1_maxsize=3)
    pool.set("a", 1, ttl=60)
    pool.set("b", 2, ttl=120)
    pool.set("c", 3, ttl=180)
    pool.set("d", 4, ttl=240)  # forces eviction of "a" (earliest expiry)
    stats = pool.stats
    assert stats["l1_size"] <= 3
    # "a" should be gone, "d" should be present
    assert pool.get("d") == 4


# ---------------------------------------------------------------------------
# L1 vs L2 hits vs miss
# ---------------------------------------------------------------------------


def test_l1_hit_does_not_touch_redis():
    mock = _MockRedis()
    pool = CachePool(namespace="test", redis=mock)
    pool.set("k", "v", ttl=60)
    mock.get_calls = 0  # reset (set above promoted via _l2_set)
    pool.get("k")
    pool.get("k")
    assert mock.get_calls == 0
    assert pool.stats["l1_hits"] == 2


def test_l2_hit_when_l1_misses():
    mock = _MockRedis()
    pool_a = CachePool(namespace="ns", redis=mock)
    pool_a.set("k", "v", ttl=60)
    # Simulate a new worker: fresh pool, same Redis backend.
    pool_b = CachePool(namespace="ns", redis=mock)
    val = pool_b.get("k")
    assert val == "v"
    assert pool_b.stats["l2_hits"] == 1
    assert pool_b.stats["l1_hits"] == 0
    # After the L2 hit it should be promoted to L1.
    val2 = pool_b.get("k")
    assert val2 == "v"
    assert pool_b.stats["l1_hits"] == 1


def test_miss_counter_increments():
    pool = CachePool(namespace="test")
    pool.get("never")
    pool.get("never2")
    assert pool.stats["misses"] == 2


# ---------------------------------------------------------------------------
# Stampede / single-flight protection
# ---------------------------------------------------------------------------


def test_async_stampede_protection_calls_fn_once():
    """10 concurrent get_or_compute_async on the same missing key → fn() called once."""

    call_count = 0
    call_lock = threading.Lock()

    async def expensive():
        nonlocal call_count
        with call_lock:
            call_count += 1
        await asyncio.sleep(0.05)
        return "computed"

    async def run():
        pool = CachePool(namespace="test")
        tasks = [
            asyncio.create_task(pool.get_or_compute_async("k", expensive, ttl=60))
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)
        return results, pool

    results, pool = asyncio.run(run())
    assert call_count == 1, f"expected 1 fn() call, got {call_count}"
    assert all(r == "computed" for r in results)
    assert pool.get("k") == "computed"


def test_sync_get_or_compute_basic():
    pool = CachePool(namespace="test")
    calls = [0]

    def compute():
        calls[0] += 1
        return 42

    assert pool.get_or_compute("k", compute, ttl=60) == 42
    assert pool.get_or_compute("k", compute, ttl=60) == 42
    assert calls[0] == 1  # second call hit the cache


def test_sync_stampede_protection():
    """Concurrent threads calling get_or_compute on the same missing key
    should still only invoke fn once."""
    pool = CachePool(namespace="test")
    calls = [0]
    call_lock = threading.Lock()
    barrier = threading.Barrier(8)
    results: list = []
    results_lock = threading.Lock()

    def slow():
        with call_lock:
            calls[0] += 1
        time.sleep(0.05)
        return "v"

    def worker():
        barrier.wait()
        v = pool.get_or_compute("k", slow, ttl=60)
        with results_lock:
            results.append(v)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls[0] == 1
    assert all(r == "v" for r in results)
    assert len(results) == 8


# ---------------------------------------------------------------------------
# Pickle envelope: round-trip non-JSON-able types
# ---------------------------------------------------------------------------


def test_pickle_envelope_roundtrips_pd_series():
    mock = _MockRedis()
    pool_writer = CachePool(namespace="test", redis=mock)
    series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2025-01-01", periods=3))
    pool_writer.set("ser", series, ttl=60)
    # Cross-worker readback: fresh pool, same Redis.
    pool_reader = CachePool(namespace="test", redis=mock)
    fetched = pool_reader.get("ser")
    assert isinstance(fetched, pd.Series), f"expected pd.Series, got {type(fetched)}"
    pd.testing.assert_series_equal(fetched, series)


def test_pickle_envelope_roundtrips_numpy_array():
    mock = _MockRedis()
    pool_writer = CachePool(namespace="test", redis=mock)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    pool_writer.set("arr", arr, ttl=60)
    pool_reader = CachePool(namespace="test", redis=mock)
    fetched = pool_reader.get("arr")
    assert isinstance(fetched, np.ndarray)
    np.testing.assert_array_equal(fetched, arr)


def test_pickle_envelope_has_magic_prefix():
    mock = _MockRedis()
    pool = CachePool(namespace="test", redis=mock)
    pool.set("k", {"a": 1}, ttl=60)
    raw = mock._d["pfm:test:k"]
    assert raw.startswith(_L2_MAGIC)


def test_pickle_envelope_rejects_legacy_payload():
    """A non-magic-prefixed Redis value must be treated as a miss, not crash."""
    mock = _MockRedis()
    mock._d["pfm:test:legacy"] = b'{"json": "stringified"}'
    pool = CachePool(namespace="test", redis=mock)
    val = pool.get("legacy")
    assert val is None  # decoded → ValueError → treated as miss
    assert pool.stats["misses"] == 1


# ---------------------------------------------------------------------------
# Stats counters
# ---------------------------------------------------------------------------


def test_stats_counters_increment():
    pool = CachePool(namespace="test")
    pool.set("a", 1, ttl=60)
    pool.set("b", 2, ttl=60)
    pool.get("a")  # l1_hit
    pool.get("a")  # l1_hit
    pool.get("missing")  # miss
    stats = pool.stats
    assert stats["set_count"] == 2
    assert stats["l1_hits"] == 2
    assert stats["misses"] == 1
    assert stats["l2_hits"] == 0


# ---------------------------------------------------------------------------
# Graceful Redis degradation
# ---------------------------------------------------------------------------


def test_redis_unavailable_at_construction_degrades_to_l1(caplog):
    broken = _MockRedis(broken=True)
    with caplog.at_level("WARNING"):
        pool = CachePool(namespace="test", redis=broken)
    pool.set("k", "v", ttl=60)
    assert pool.get("k") == "v"  # L1 still works
    assert pool.stats["redis_degraded"] is True
    # set should not have been forwarded to the broken backend
    assert broken.set_calls == 0


def test_redis_failure_mid_op_degrades_quietly():
    class _Flaky(_MockRedis):
        def __init__(self):
            super().__init__()
            self._ok = True

        def get(self, key):
            if not self._ok:
                raise ConnectionError("flaky")
            return super().get(key)

    flaky = _Flaky()
    pool = CachePool(namespace="test", redis=flaky)
    pool.set("k", "v", ttl=60)
    # Flip Redis offline mid-flight
    flaky._ok = False
    # L1 still has the value — no Redis call needed
    assert pool.get("k") == "v"
    # New pool same Redis backend: first get triggers a Redis failure
    pool2 = CachePool(namespace="test", redis=flaky)
    assert pool2.get("k") is None  # degrades, returns default
    assert pool2.stats["redis_degraded"] is True


# ---------------------------------------------------------------------------
# clear(prefix=)
# ---------------------------------------------------------------------------


def test_clear_all():
    pool = CachePool(namespace="test")
    pool.set("a:1", 1, ttl=60)
    pool.set("a:2", 2, ttl=60)
    pool.set("b:1", 3, ttl=60)
    removed = pool.clear()
    assert removed == 3
    assert pool.get("a:1") is None
    assert pool.get("b:1") is None


def test_clear_with_prefix_removes_matching_only():
    pool = CachePool(namespace="test")
    pool.set("user:1", "alice", ttl=60)
    pool.set("user:2", "bob", ttl=60)
    pool.set("session:1", "abc", ttl=60)
    removed = pool.clear(prefix="user:")
    assert removed == 2
    assert pool.get("user:1") is None
    assert pool.get("user:2") is None
    assert pool.get("session:1") == "abc"


def test_clear_prefix_also_clears_l2():
    mock = _MockRedis()
    pool = CachePool(namespace="test", redis=mock)
    pool.set("user:1", "alice", ttl=60)
    pool.set("user:2", "bob", ttl=60)
    pool.set("other:1", "x", ttl=60)
    pool.clear(prefix="user:")
    # L2 keys should be gone for "user:*" but "other:1" remains
    assert "pfm:test:user:1" not in mock._d
    assert "pfm:test:user:2" not in mock._d
    assert "pfm:test:other:1" in mock._d


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------


def test_namespaces_are_isolated_in_l2():
    mock = _MockRedis()
    pool_a = CachePool(namespace="alpha", redis=mock)
    pool_b = CachePool(namespace="beta", redis=mock)
    pool_a.set("k", "from-alpha", ttl=60)
    pool_b.set("k", "from-beta", ttl=60)
    # Fresh pools (cold L1) reading same Redis
    pool_a2 = CachePool(namespace="alpha", redis=mock)
    pool_b2 = CachePool(namespace="beta", redis=mock)
    assert pool_a2.get("k") == "from-alpha"
    assert pool_b2.get("k") == "from-beta"


# ---------------------------------------------------------------------------
# Repr smoke
# ---------------------------------------------------------------------------


def test_repr_contains_namespace_and_state():
    pool = CachePool(namespace="abc", l1_maxsize=4)
    pool.set("x", 1, ttl=60)
    r = repr(pool)
    assert "abc" in r
    assert "l1_maxsize=4" in r
