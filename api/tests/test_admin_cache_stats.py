"""Tests for :mod:`pfm.admin.cache_stats_router` (W12-17).

Strategy
--------

We install **fake** modules into ``sys.modules`` whose names look like
``pfm.<x>`` so the router's introspection picks them up. Each fake
module is a ``types.ModuleType`` carrying one or more real
``CachePool`` instances. We never need to touch the live source modules
(``pfm.sources.manifold`` etc.) — those keep their actual pools, which
shows up in the response but is verified separately via fixtures that
zero them out.

Fixture hygiene
---------------

A ``_clean_modules`` fixture snapshots ``sys.modules`` before each test
and restores it afterwards. This keeps test isolation watertight even
if a test forgets to clean up an injected ``pfm.fake_*`` module.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.admin.cache_stats_router import (
    _hit_rate,
    _iter_cache_pools,
    _pool_row,
    collect_cache_stats,
    router,
)
from pfm.cache_pool import CachePool

# ─────────────────────────── fixtures ───────────────────────────


@pytest.fixture()
def _clean_modules() -> Iterator[None]:
    """Snapshot ``sys.modules`` and restore after the test.

    Any ``pfm.fake_*`` modules injected by the test are wiped on exit so
    introspection in subsequent tests doesn't see them.
    """
    before = set(sys.modules.keys())
    try:
        yield
    finally:
        # Remove anything the test added; don't touch pre-existing names
        # (some tests may legitimately import new pfm.* modules).
        added = set(sys.modules.keys()) - before
        for name in added:
            if name.startswith("pfm.fake") or name.startswith("pfm.t_admin_"):
                sys.modules.pop(name, None)


@pytest.fixture()
def client() -> TestClient:
    """FastAPI client with just the cache-stats router mounted."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _install_fake_module(name: str, **pool_attrs: CachePool) -> types.ModuleType:
    """Create a ``pfm.<name>`` fake module with the given pool attributes."""
    mod_name = f"pfm.{name}"
    mod = types.ModuleType(mod_name)
    for attr, pool in pool_attrs.items():
        setattr(mod, attr, pool)
    sys.modules[mod_name] = mod
    return mod


# ───────────────────────── unit helpers ─────────────────────────


def test_hit_rate_zero_requests_returns_zero() -> None:
    """Zero requests must not raise ZeroDivisionError or return NaN."""
    assert _hit_rate(0, 0, 0) == 0.0


def test_hit_rate_basic_math() -> None:
    """Simple math sanity: 4 hits over 5 requests ⇒ 0.8."""
    # 3 L1 hits + 1 L2 hit + 1 miss = 5 total, 4 hits ⇒ 0.8.
    assert _hit_rate(3, 1, 1) == 0.8


def test_hit_rate_perfect() -> None:
    """All hits ⇒ 1.0."""
    assert _hit_rate(10, 5, 0) == 1.0


def test_hit_rate_all_misses() -> None:
    """All misses ⇒ 0.0."""
    assert _hit_rate(0, 0, 7) == 0.0


# ─────────────────────────── introspection ──────────────────────


def test_introspection_finds_injected_pool(_clean_modules: None) -> None:
    """A CachePool stuck on a fake pfm.* module must show up."""
    pool = CachePool(namespace="t-admin-find", l1_maxsize=4)
    _install_fake_module("fake_one", _CACHE=pool)

    found = _iter_cache_pools()
    namespaces = [p._namespace for (_m, _a, p) in found]
    assert "t-admin-find" in namespaces


def test_introspection_skips_non_pfm_modules(_clean_modules: None) -> None:
    """Pools on non-``pfm.*`` modules must be ignored."""
    pool = CachePool(namespace="t-admin-skip", l1_maxsize=4)
    # Put it under a top-level name — should be skipped.
    mod = types.ModuleType("not_pfm_module")
    mod._CACHE = pool  # type: ignore[attr-defined]
    sys.modules["not_pfm_module"] = mod
    try:
        found = _iter_cache_pools()
        namespaces = [p._namespace for (_m, _a, p) in found]
        assert "t-admin-skip" not in namespaces
    finally:
        sys.modules.pop("not_pfm_module", None)


def test_introspection_skips_admin_package_itself(_clean_modules: None) -> None:
    """Pools registered on ``pfm.admin.*`` modules must not be counted.

    Otherwise a test that imports the router and accidentally sets a pool
    attribute on it would pollute the stats endpoint forever.
    """
    pool = CachePool(namespace="t-admin-self", l1_maxsize=4)
    _install_fake_module("admin.fake_admin_sub", _CACHE=pool)
    found = _iter_cache_pools()
    namespaces = [p._namespace for (_m, _a, p) in found]
    assert "t-admin-self" not in namespaces


def test_introspection_deduplicates_reexports(_clean_modules: None) -> None:
    """The same pool re-exported under two modules counts once."""
    pool = CachePool(namespace="t-admin-dup", l1_maxsize=4)
    _install_fake_module("fake_dup_a", _CACHE=pool)
    _install_fake_module("fake_dup_b", _CACHE=pool, _ALIAS=pool)

    found = _iter_cache_pools()
    matches = [(m, a) for (m, a, p) in found if p._namespace == "t-admin-dup"]
    assert len(matches) == 1, f"expected exactly 1 row for dedup pool, got {matches}"


# ─────────────────────────── pool row ───────────────────────────


def test_pool_row_reflects_live_stats(_clean_modules: None) -> None:
    """After a few real ops the row's counters must match the pool's stats."""
    pool = CachePool(namespace="t-admin-rows", l1_maxsize=8)
    pool.set("a", 1, ttl=60)
    pool.set("b", 2, ttl=60)
    pool.get("a")  # L1 hit
    pool.get("a")  # L1 hit
    pool.get("missing-key")  # miss

    row = _pool_row("pfm.fake_rows", "_CACHE", pool)

    assert row["namespace"] == "t-admin-rows"
    assert row["module"] == "pfm.fake_rows"
    assert row["attr"] == "_CACHE"
    assert row["l1_hits"] == 2
    assert row["l1_misses"] == 1
    assert row["l2_hits"] == 0
    assert row["set_count"] == 2
    assert row["l1_size"] == 2
    # 2 hits / 3 requests ⇒ 0.6667
    assert row["hit_rate"] == pytest.approx(2 / 3, abs=1e-3)


# ─────────────────────────── aggregation ────────────────────────


def test_collect_cache_stats_aggregates_totals(_clean_modules: None) -> None:
    """Totals must sum across all discovered pools."""
    p1 = CachePool(namespace="t-admin-agg-1", l1_maxsize=8)
    p2 = CachePool(namespace="t-admin-agg-2", l1_maxsize=8)

    p1.set("x", 1, ttl=60)
    p1.get("x")  # 1 L1 hit, 1 set on p1
    p2.set("y", 2, ttl=60)
    p2.get("y")  # 1 L1 hit, 1 set on p2
    p2.get("missing")  # 1 miss on p2

    _install_fake_module("fake_agg_a", _CACHE=p1)
    _install_fake_module("fake_agg_b", _CACHE=p2)

    payload = collect_cache_stats()
    # Filter to our two pools to ignore the live ones picked up
    # incidentally from pfm.sources.*.
    ours = [r for r in payload["pools"] if r["namespace"].startswith("t-admin-agg-")]
    assert len(ours) == 2
    sub_total_hits = sum(r["l1_hits"] for r in ours)
    sub_total_misses = sum(r["l1_misses"] for r in ours)
    sub_total_sets = sum(r["set_count"] for r in ours)
    assert sub_total_hits == 2
    assert sub_total_misses == 1
    assert sub_total_sets == 2

    # The grand totals must be at least our contribution (real pools may add).
    assert payload["totals"]["l1_hits"] >= 2
    assert payload["totals"]["misses"] >= 1
    assert payload["totals"]["set_count"] >= 2


def test_checked_at_is_iso8601_z(_clean_modules: None) -> None:
    """``checked_at`` must be a UTC ISO8601 string ending in ``Z``."""
    payload = collect_cache_stats()
    assert isinstance(payload["checked_at"], str)
    assert payload["checked_at"].endswith("Z")
    # Roundtrip-parse it to make sure it's actually ISO8601.
    from datetime import datetime

    parsed = datetime.fromisoformat(payload["checked_at"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


# ─────────────────────────── HTTP endpoint ──────────────────────


def test_endpoint_returns_expected_top_level_shape(
    client: TestClient, _clean_modules: None
) -> None:
    """The HTTP response must have the documented top-level keys."""
    pool = CachePool(namespace="t-admin-http", l1_maxsize=4)
    pool.set("k", "v", ttl=60)
    pool.get("k")
    _install_fake_module("fake_http", _CACHE=pool)

    resp = client.get("/admin/cache-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) >= {"checked_at", "pools", "totals", "pool_count"}
    assert isinstance(body["pools"], list)
    assert isinstance(body["totals"], dict)
    # Required total fields
    assert set(body["totals"].keys()) >= {
        "l1_hits",
        "l2_hits",
        "misses",
        "set_count",
        "l1_size",
        "hit_rate",
    }
    namespaces = [p["namespace"] for p in body["pools"]]
    assert "t-admin-http" in namespaces


def test_endpoint_with_no_pools(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force introspection to return empty; totals must all be zero."""
    import pfm.admin.cache_stats_router as mod

    monkeypatch.setattr(mod, "_iter_cache_pools", lambda: [])
    # Also short-circuit the eager-import helper so it doesn't repopulate.
    monkeypatch.setattr(mod, "_eager_import_known_pool_modules", lambda: None)

    resp = client.get("/admin/cache-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pools"] == []
    assert body["pool_count"] == 0
    assert body["totals"]["l1_hits"] == 0
    assert body["totals"]["l2_hits"] == 0
    assert body["totals"]["misses"] == 0
    assert body["totals"]["hit_rate"] == 0.0


def test_endpoint_tolerates_broken_module(client: TestClient, _clean_modules: None) -> None:
    """A module whose ``__getattr__`` raises must not crash the endpoint."""

    class _BrokenModule(types.ModuleType):
        def __getattribute__(self, name: str) -> object:
            # dir() must still work for the introspector to walk us.
            if name in ("__class__", "__dict__", "__dir__"):
                return super().__getattribute__(name)
            if name.startswith("_BROKEN"):
                raise RuntimeError(f"boom on {name}")
            return super().__getattribute__(name)

    broken = _BrokenModule("pfm.fake_broken")
    # Stash a real attribute so dir() lists at least one ``_BROKEN_*`` name
    # that, when getattr'd, raises.
    broken.__dict__["_BROKEN_X"] = "would-explode"
    sys.modules["pfm.fake_broken"] = broken

    # Should not raise — the introspector swallows getattr errors.
    resp = client.get("/admin/cache-stats")
    assert resp.status_code == 200


def test_pool_with_zero_activity_yields_zero_hit_rate(
    client: TestClient, _clean_modules: None
) -> None:
    """A brand-new pool with no get/set has hit_rate 0.0, not NaN/error."""
    pool = CachePool(namespace="t-admin-fresh", l1_maxsize=4)
    _install_fake_module("fake_fresh", _CACHE=pool)
    resp = client.get("/admin/cache-stats")
    assert resp.status_code == 200
    body = resp.json()
    rows = [r for r in body["pools"] if r["namespace"] == "t-admin-fresh"]
    assert len(rows) == 1
    assert rows[0]["hit_rate"] == 0.0
    assert rows[0]["l1_hits"] == 0
    assert rows[0]["l1_misses"] == 0


def test_endpoint_includes_l2_hits_when_promoted(client: TestClient, _clean_modules: None) -> None:
    """Simulate an L2 hit by stuffing a fake redis backend, then verify."""

    class _FakeRedis:
        enabled = True

        def __init__(self) -> None:
            self._d: dict[str, bytes] = {}

        def get(self, key: str) -> bytes | None:
            return self._d.get(key)

        def set(self, key: str, value: bytes, ex: int | None = None) -> None:
            self._d[key] = value

        def delete(self, key: str) -> None:
            self._d.pop(key, None)

    redis = _FakeRedis()
    pool = CachePool(namespace="t-admin-l2", redis=redis, l1_maxsize=4)
    # ``set`` populates both L1 and L2.
    pool.set("k", 42, ttl=60)
    # Wipe L1 directly so the next get falls through to L2.
    with pool._lock:
        pool._d.clear()
        pool._heap.clear()
    # Now this read is an L2 hit.
    assert pool.get("k") == 42

    _install_fake_module("fake_l2", _CACHE=pool)
    resp = client.get("/admin/cache-stats")
    body = resp.json()
    rows = [r for r in body["pools"] if r["namespace"] == "t-admin-l2"]
    assert len(rows) == 1
    assert rows[0]["l2_hits"] >= 1
