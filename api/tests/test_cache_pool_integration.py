"""Integration tests for ``pfm.cache_pool.CachePool`` wired into FastAPI.

Run with::

    pytest tests/test_cache_pool_integration.py -q --noconftest

These tests construct a minimal FastAPI app exposing one endpoint
``GET /cached-thing/{key}`` whose handler is
``CachePool.get_or_compute_async`` over a deliberately slow async
``compute_fn`` (``await asyncio.sleep(0.5)``). The point is to verify
the pool's behaviour under realistic ASGI concurrency — not the in-tree
single-flight unit tests in ``test_cache_pool.py``.

Why ``--noconftest``: the project ``conftest.py`` pulls in the full app
fixture chain (Polymarket mocks, factor warmup, Redis attach). A
two-tier cache test needs none of that and runs ~50x faster on its own.

Tests that need a real Redis-like backend use ``fakeredis``. When the
package is unavailable they are skipped with a reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

# Make ``pfm`` importable without the project's conftest sys.path tweaks.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pfm.cache_pool import CachePool

try:
    import fakeredis  # type: ignore

    _HAS_FAKEREDIS = True
except ImportError:  # pragma: no cover — env-dependent
    _HAS_FAKEREDIS = False


# ---------------------------------------------------------------------------
# App factory + compute_fn
# ---------------------------------------------------------------------------


_COMPUTE_DELAY = 0.5  # seconds; matches the spec ("~500ms compute")


def _build_app(pool: CachePool, *, compute_counter: dict[str, int]) -> FastAPI:
    """Mount a minimal FastAPI app with one cached endpoint.

    ``compute_counter`` is mutated in place every time the slow compute
    actually runs, so tests can assert "exactly N computes happened".
    """
    app = FastAPI()

    async def _slow_compute(key: str) -> dict[str, object]:
        compute_counter["count"] = compute_counter.get("count", 0) + 1
        await asyncio.sleep(_COMPUTE_DELAY)
        # ``random_value`` so we can verify all concurrent callers got the
        # SAME cached value (i.e. compute ran once) — if it ran twice they'd
        # see different numbers.
        return {
            "key": key,
            "random_value": random.random(),
            "nonce": uuid.uuid4().hex,
        }

    @app.get("/cached-thing/{key}")
    async def cached_thing(key: str, ttl: int = 60) -> dict[str, object]:
        return await pool.get_or_compute_async(key, lambda: _slow_compute(key), ttl=ttl)

    return app


@contextlib.asynccontextmanager
async def _client_for(pool: CachePool, *, compute_counter: dict[str, int]):
    app = _build_app(pool, compute_counter=compute_counter)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# 1 & 2 — cold miss latency + warm hit latency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_request_is_slow_and_populates_cache() -> None:
    """First request must take roughly ``_COMPUTE_DELAY`` (a cold miss)."""
    pool = CachePool(namespace="t1")
    counter: dict[str, int] = {}
    async with _client_for(pool, compute_counter=counter) as client:
        t0 = time.perf_counter()
        r = await client.get("/cached-thing/abc")
        elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert r.json()["key"] == "abc"
    # The compute is 500ms; allow a generous upper bound for slow CI.
    assert 0.45 <= elapsed <= 2.0, f"cold miss elapsed={elapsed:.3f}s"
    assert counter["count"] == 1
    # Pool now has exactly one entry.
    assert pool.stats["l1_size"] == 1


@pytest.mark.asyncio
async def test_second_request_is_a_fast_cache_hit() -> None:
    """Warm hit must be orders of magnitude faster than the cold miss."""
    pool = CachePool(namespace="t2")
    counter: dict[str, int] = {}
    async with _client_for(pool, compute_counter=counter) as client:
        first = await client.get("/cached-thing/abc")
        t0 = time.perf_counter()
        second = await client.get("/cached-thing/abc")
        warm_elapsed = time.perf_counter() - t0
    assert first.json() == second.json(), "warm hit must return cached payload"
    # The spec says ~5ms; in-process httpx ASGI calls are ~1ms, allow up to 100ms.
    assert warm_elapsed < 0.1, f"warm hit elapsed={warm_elapsed:.3f}s"
    assert counter["count"] == 1, "compute should NOT run on warm hit"


# ---------------------------------------------------------------------------
# 3 — concurrent stampede on a single key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ten_concurrent_requests_compute_runs_once() -> None:
    pool = CachePool(namespace="t3")
    counter: dict[str, int] = {}
    async with _client_for(pool, compute_counter=counter) as client:
        results = await asyncio.gather(*(client.get("/cached-thing/same") for _ in range(10)))
    assert counter["count"] == 1, f"compute ran {counter['count']} times"
    # All ten responses identical (proves they share the cached value).
    payloads = [r.json() for r in results]
    first = payloads[0]
    for p in payloads[1:]:
        assert p == first, "all concurrent callers must see the same cached value"


# ---------------------------------------------------------------------------
# 4 — concurrent requests across many distinct keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_hundred_requests_over_ten_keys_computes_ten_times() -> None:
    pool = CachePool(namespace="t4")
    counter: dict[str, int] = {}
    keys = [f"k{i}" for i in range(10)]
    async with _client_for(pool, compute_counter=counter) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(client.get(f"/cached-thing/{random.choice(keys)}") for _ in range(100))
        )
        elapsed = time.perf_counter() - t0
    assert all(r.status_code == 200 for r in results)
    # Each distinct key triggers exactly one compute regardless of fan-in.
    assert counter["count"] == 10, f"expected 10 computes, got {counter['count']}"
    # Parallel completion: 10 distinct computes of 0.5s each must finish
    # well under the serial 5s; on modern hardware ~0.7s is typical. Allow
    # 4s as the upper bound so a loaded CI box still passes.
    assert elapsed < 4.0, f"100 reqs / 10 keys elapsed={elapsed:.2f}s"
    assert pool.stats["l1_size"] == 10


# ---------------------------------------------------------------------------
# 5 — TTL expiry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttl_expiry_triggers_recompute() -> None:
    pool = CachePool(namespace="t5")
    counter: dict[str, int] = {}
    async with _client_for(pool, compute_counter=counter) as client:
        # ttl=1s, then wait 1.3s so the entry definitely expired.
        await client.get("/cached-thing/x?ttl=1")
        assert counter["count"] == 1
        await asyncio.sleep(1.3)
        await client.get("/cached-thing/x?ttl=1")
    assert counter["count"] == 2, "TTL expiry should force a recompute"


# ---------------------------------------------------------------------------
# 6 — Redis L2 promotion
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_FAKEREDIS, reason="fakeredis not installed")
@pytest.mark.asyncio
async def test_l2_redis_hit_when_l1_evicted() -> None:
    """When L1 is wiped but L2 still has the entry, the next call hits L2."""
    redis = fakeredis.FakeRedis()
    # CachePool's protocol expects a truthy ``enabled`` attribute; attach it.
    redis.enabled = True  # type: ignore[attr-defined]
    pool = CachePool(namespace="t6", redis=redis, l1_maxsize=1024)
    counter: dict[str, int] = {}
    async with _client_for(pool, compute_counter=counter) as client:
        await client.get("/cached-thing/foo")
        assert counter["count"] == 1
        # Wipe L1 only — leave L2 untouched.
        pool._d.clear()
        pool._heap.clear()
        # Reset stats so we can see the L2 hit cleanly.
        before = pool.stats
        r = await client.get("/cached-thing/foo")
    assert counter["count"] == 1, "L2 hit must NOT trigger a recompute"
    assert r.status_code == 200
    after = pool.stats
    assert after["l2_hits"] > before["l2_hits"], "L2 hit counter must increment"


# ---------------------------------------------------------------------------
# 7 — Restart simulation: new pool, same Redis
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_FAKEREDIS, reason="fakeredis not installed")
@pytest.mark.asyncio
async def test_restart_simulation_l2_survives_new_pool_instance() -> None:
    redis = fakeredis.FakeRedis()
    redis.enabled = True  # type: ignore[attr-defined]

    counter_a: dict[str, int] = {}
    pool_a = CachePool(namespace="t7", redis=redis)
    async with _client_for(pool_a, compute_counter=counter_a) as client_a:
        await client_a.get("/cached-thing/restart-key")
    assert counter_a["count"] == 1

    # Simulate a restart: brand-new CachePool, same Redis instance.
    counter_b: dict[str, int] = {}
    pool_b = CachePool(namespace="t7", redis=redis)
    async with _client_for(pool_b, compute_counter=counter_b) as client_b:
        r = await client_b.get("/cached-thing/restart-key")
    # The new pool has empty L1 but must fetch from L2 — no recompute.
    # ``counter_b`` is mutated only when ``_slow_compute`` runs; an empty
    # dict therefore proves the L2 hit short-circuited compute entirely.
    assert counter_b.get("count", 0) == 0, (
        "new CachePool instance must read from shared Redis without recomputing"
    )
    assert r.status_code == 200
    assert pool_b.stats["l2_hits"] >= 1


# ---------------------------------------------------------------------------
# 8 — 50-request concurrent stampede
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fifty_concurrent_stampede_exactly_one_compute() -> None:
    pool = CachePool(namespace="t8")
    counter: dict[str, int] = {}
    async with _client_for(pool, compute_counter=counter) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(client.get("/cached-thing/stampede") for _ in range(50)))
        elapsed = time.perf_counter() - t0
    assert all(r.status_code == 200 for r in results)
    assert counter["count"] == 1, f"expected 1 compute, got {counter['count']}"
    # 50 callers waiting on a 0.5s single-flight should finish in ~0.5s — not 25s.
    assert elapsed < 2.0, f"stampede elapsed={elapsed:.2f}s suggests serialization"


# ---------------------------------------------------------------------------
# 9 — L1 maxsize eviction
# ---------------------------------------------------------------------------


def test_l1_maxsize_eviction_drops_oldest_entries() -> None:
    """Insert 1100 entries into a pool with ``l1_maxsize=1024``; oldest evicted.

    Note: the pool's eviction policy is heap-based by *expiry time*, not
    LRU. Entries inserted earliest have the earliest ``expires_at`` (for a
    fixed TTL), so they are the first victims — which matches the spec's
    "oldest evicted" expectation.
    """
    pool = CachePool(namespace="t9", l1_maxsize=1024)
    # Stagger TTLs slightly so the first inserts have the smallest expires_at.
    for i in range(1100):
        # ttl grows monotonically so the heap order matches insertion order.
        pool.set(f"k{i:04d}", i, ttl=60 + i)
    assert pool.stats["l1_size"] <= 1024, "L1 must not exceed l1_maxsize"
    # The first ~76 keys (the oldest, smallest TTL) should be gone.
    sentinel = object()
    early_misses = sum(1 for i in range(80) if pool.get(f"k{i:04d}", default=sentinel) is sentinel)
    assert early_misses >= 50, (
        f"expected most of the first 80 keys evicted, got only {early_misses} misses"
    )
    # The most-recent keys must still be present.
    assert pool.get("k1099") == 1099
    assert pool.get("k1050") == 1050


# ---------------------------------------------------------------------------
# 10 — Stats counters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_counters_track_hits_and_misses() -> None:
    pool = CachePool(namespace="t10")
    counter: dict[str, int] = {}
    async with _client_for(pool, compute_counter=counter) as client:
        # Three distinct keys — three misses, three computes, three sets.
        for k in ("a", "b", "c"):
            await client.get(f"/cached-thing/{k}")
        # Re-fetch each twice — six L1 hits.
        for k in ("a", "b", "c"):
            await client.get(f"/cached-thing/{k}")
            await client.get(f"/cached-thing/{k}")
    stats = pool.stats
    # ``get_or_compute_async`` calls ``get`` once per request *plus* once
    # under the lock on a miss. So a miss costs 2 misses, a hit costs 1 hit.
    # Three misses → 6 miss-bumps. Six hits → 6 l1_hits.
    assert stats["misses"] == 6, f"misses={stats['misses']}"
    assert stats["l1_hits"] == 6, f"l1_hits={stats['l1_hits']}"
    assert stats["set_count"] == 3
    assert stats["l1_size"] == 3
    assert counter["count"] == 3


# ---------------------------------------------------------------------------
# Bonus — p50/p95 warm-hit latency probe (documented, not strict)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_hit_p50_under_50ms() -> None:
    """Quick latency probe so we have a real number to report.

    Runs 50 warm hits on the same populated key and asserts the median
    is well under 50ms; this is a smoke check, not a benchmark.
    """
    pool = CachePool(namespace="t11")
    counter: dict[str, int] = {}
    async with _client_for(pool, compute_counter=counter) as client:
        await client.get("/cached-thing/warm")  # populate
        timings: list[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            await client.get("/cached-thing/warm")
            timings.append(time.perf_counter() - t0)
    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[int(len(timings) * 0.95)]
    assert p50 < 0.05, f"warm-hit p50={p50 * 1000:.2f}ms exceeds 50ms"
    # p95 sanity bound — generous for slow CI.
    assert p95 < 0.2, f"warm-hit p95={p95 * 1000:.2f}ms exceeds 200ms"
