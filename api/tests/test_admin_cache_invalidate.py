"""Tests for :mod:`pfm.admin.cache_invalidate_router` (W12-18).

Strategy
--------

Mirrors ``test_admin_cache_stats``: install fake ``pfm.<x>`` modules
into ``sys.modules`` carrying real ``CachePool`` instances, then drive
the router via FastAPI's ``TestClient``. A ``_clean_modules`` fixture
restores ``sys.modules`` after each test so introspection in later
tests doesn't see leftover fake modules.

Auth tests use ``monkeypatch.setenv`` / ``delenv`` to flip
``PFM_ADMIN_TOKEN`` per test — the dependency reads the env var
per-request so this works without restarting anything.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.admin.cache_invalidate_router import (
    CacheInvalidateRequest,
    _iter_cache_pools,
    perform_invalidation,
    router,
)
from pfm.cache_pool import CachePool

# ─────────────────────────── fixtures ───────────────────────────


@pytest.fixture()
def _clean_modules() -> Iterator[None]:
    """Snapshot ``sys.modules`` and clean up any test-injected ``pfm.fake_*``."""
    before = set(sys.modules.keys())
    try:
        yield
    finally:
        added = set(sys.modules.keys()) - before
        for name in added:
            # Be conservative: only remove names we know we injected.
            if name.startswith("pfm.fake") or name.startswith("pfm.t_inv_"):
                sys.modules.pop(name, None)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """FastAPI client with just the cache-invalidate router mounted.

    ``PFM_ADMIN_TOKEN`` is removed by default so each test starts in
    "dev mode" (auth disabled). Tests that need auth gating explicitly
    call ``monkeypatch.setenv``.
    """
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)
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


# ─────────────────────── body validation ────────────────────────


def test_body_missing_both_fields_returns_422(client: TestClient) -> None:
    """Empty body must fail validation — preventing accidental wipe-all."""
    resp = client.post("/admin/cache/invalidate", json={})
    assert resp.status_code == 422


def test_body_both_empty_strings_returns_422(client: TestClient) -> None:
    """``{"prefix": "", "namespace": ""}`` is functionally empty → 422.

    Without this, a JS frontend defaulting unset fields to ``""`` would
    silently flush every pool. The validator treats blank strings as
    "not provided".
    """
    resp = client.post("/admin/cache/invalidate", json={"prefix": "", "namespace": ""})
    assert resp.status_code == 422


def test_body_whitespace_only_returns_422(client: TestClient) -> None:
    """Whitespace-only fields must also be rejected."""
    resp = client.post("/admin/cache/invalidate", json={"prefix": "   ", "namespace": "\t"})
    assert resp.status_code == 422


def test_request_model_validator_directly() -> None:
    """Unit-test the Pydantic validator without HTTP."""
    with pytest.raises(ValueError):
        CacheInvalidateRequest(prefix=None, namespace=None)
    with pytest.raises(ValueError):
        CacheInvalidateRequest(prefix="", namespace="")
    # These must not raise.
    CacheInvalidateRequest(prefix="factors:")
    CacheInvalidateRequest(namespace="manifold-search")
    CacheInvalidateRequest(prefix="factors:", namespace="manifold-search")


# ─────────────────────── basic invalidation ─────────────────────


def test_valid_prefix_invalidates_correctly(client: TestClient, _clean_modules: None) -> None:
    """A prefix-scoped call must remove only matching keys."""
    pool = CachePool(namespace="t-inv-prefix", l1_maxsize=16)
    pool.set("factors:abc", 1, ttl=60)
    pool.set("factors:def", 2, ttl=60)
    pool.set("other:xyz", 3, ttl=60)
    _install_fake_module("fake_inv_prefix", _CACHE=pool)

    resp = client.post(
        "/admin/cache/invalidate",
        json={"prefix": "factors:", "namespace": "t-inv-prefix"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_removed"] == 2
    assert len(body["results"]) == 1
    row = body["results"][0]
    assert row["namespace"] == "t-inv-prefix"
    assert row["removed"] == 2
    assert row["remaining"] == 1  # the "other:xyz" key survives
    # Sanity-check the live pool agrees.
    assert pool.get("other:xyz") == 3
    assert pool.get("factors:abc") is None


def test_namespace_only_wipes_entire_pool(client: TestClient, _clean_modules: None) -> None:
    """No prefix + namespace match → clear() with prefix=None drops everything."""
    pool = CachePool(namespace="t-inv-ns", l1_maxsize=16)
    pool.set("a", 1, ttl=60)
    pool.set("b", 2, ttl=60)
    pool.set("c", 3, ttl=60)
    _install_fake_module("fake_inv_ns", _CACHE=pool)

    resp = client.post("/admin/cache/invalidate", json={"namespace": "t-inv-ns"})
    assert resp.status_code == 200
    body = resp.json()
    rows = [r for r in body["results"] if r["namespace"] == "t-inv-ns"]
    assert len(rows) == 1
    assert rows[0]["removed"] == 3
    assert rows[0]["remaining"] == 0


def test_prefix_only_applies_across_all_pools(client: TestClient, _clean_modules: None) -> None:
    """Without a namespace filter, every discovered pool gets the prefix wipe."""
    p1 = CachePool(namespace="t-inv-all-1", l1_maxsize=8)
    p2 = CachePool(namespace="t-inv-all-2", l1_maxsize=8)
    p1.set("factors:1", 1, ttl=60)
    p1.set("keep:1", 2, ttl=60)
    p2.set("factors:2", 3, ttl=60)
    p2.set("keep:2", 4, ttl=60)
    _install_fake_module("fake_inv_all_a", _CACHE=p1)
    _install_fake_module("fake_inv_all_b", _CACHE=p2)

    resp = client.post("/admin/cache/invalidate", json={"prefix": "factors:"})
    assert resp.status_code == 200
    body = resp.json()
    # Our two pools must both appear; total_removed >= 2 (real pools may add 0).
    ours = [r for r in body["results"] if r["namespace"].startswith("t-inv-all-")]
    assert len(ours) == 2
    assert sum(r["removed"] for r in ours) == 2
    # Sanity: kept keys survived.
    assert p1.get("keep:1") == 2
    assert p2.get("keep:2") == 4


def test_empty_match_returns_zero_removed(client: TestClient, _clean_modules: None) -> None:
    """A prefix that matches nothing returns 200 with removed=0."""
    pool = CachePool(namespace="t-inv-empty", l1_maxsize=8)
    pool.set("alpha", 1, ttl=60)
    _install_fake_module("fake_inv_empty", _CACHE=pool)

    resp = client.post(
        "/admin/cache/invalidate",
        json={"prefix": "no-such-prefix:", "namespace": "t-inv-empty"},
    )
    assert resp.status_code == 200
    body = resp.json()
    rows = [r for r in body["results"] if r["namespace"] == "t-inv-empty"]
    assert len(rows) == 1
    assert rows[0]["removed"] == 0
    assert rows[0]["remaining"] == 1  # original key still there


def test_unknown_namespace_yields_empty_results(client: TestClient, _clean_modules: None) -> None:
    """If no pool matches the namespace filter, results is empty."""
    pool = CachePool(namespace="t-inv-real", l1_maxsize=8)
    pool.set("k", 1, ttl=60)
    _install_fake_module("fake_inv_unknown", _CACHE=pool)

    resp = client.post("/admin/cache/invalidate", json={"namespace": "does-not-exist-xyz"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"] == []
    assert body["total_removed"] == 0


# ─────────────────────── response shape ─────────────────────────


def test_response_has_required_top_level_keys(client: TestClient, _clean_modules: None) -> None:
    """Documented shape: invalidated_at, results, total_removed."""
    pool = CachePool(namespace="t-inv-shape", l1_maxsize=4)
    pool.set("x", 1, ttl=60)
    _install_fake_module("fake_inv_shape", _CACHE=pool)

    resp = client.post("/admin/cache/invalidate", json={"namespace": "t-inv-shape"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"invalidated_at", "results", "total_removed"}
    assert isinstance(body["invalidated_at"], str)
    assert body["invalidated_at"].endswith("Z")
    assert isinstance(body["results"], list)
    assert isinstance(body["total_removed"], int)


def test_pool_row_has_required_fields(client: TestClient, _clean_modules: None) -> None:
    """Each row must include namespace, removed, remaining."""
    pool = CachePool(namespace="t-inv-row", l1_maxsize=4)
    pool.set("a", 1, ttl=60)
    pool.set("b", 2, ttl=60)
    _install_fake_module("fake_inv_row", _CACHE=pool)

    resp = client.post("/admin/cache/invalidate", json={"namespace": "t-inv-row"})
    body = resp.json()
    row = next(r for r in body["results"] if r["namespace"] == "t-inv-row")
    assert set(row.keys()) == {"namespace", "removed", "remaining"}


# ─────────────────────── auth gating ────────────────────────────


def test_no_token_env_unset_allows_request(
    client: TestClient, _clean_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env unset + no Authorization header → 200 (dev mode)."""
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)
    resp = client.post("/admin/cache/invalidate", json={"namespace": "anything-here"})
    assert resp.status_code == 200


def test_token_mismatch_returns_403(
    client: TestClient, _clean_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env set + wrong bearer token → 403."""
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "correct-token")
    resp = client.post(
        "/admin/cache/invalidate",
        json={"namespace": "x"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 403


def test_token_match_returns_200(
    client: TestClient, _clean_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env set + matching bearer token → 200."""
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "secret-abc")
    resp = client.post(
        "/admin/cache/invalidate",
        json={"namespace": "anything"},
        headers={"Authorization": "Bearer secret-abc"},
    )
    assert resp.status_code == 200


def test_token_env_set_no_header_returns_403(
    client: TestClient, _clean_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env set + no Authorization header → 403."""
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "secret-abc")
    resp = client.post("/admin/cache/invalidate", json={"namespace": "anything"})
    assert resp.status_code == 403


def test_token_env_set_malformed_header_returns_403(
    client: TestClient, _clean_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authorization header without ``Bearer`` prefix → 403."""
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "secret-abc")
    resp = client.post(
        "/admin/cache/invalidate",
        json={"namespace": "x"},
        headers={"Authorization": "secret-abc"},  # missing "Bearer "
    )
    assert resp.status_code == 403


def test_token_env_set_wrong_scheme_returns_403(
    client: TestClient, _clean_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Basic ...`` instead of ``Bearer ...`` → 403."""
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "secret-abc")
    resp = client.post(
        "/admin/cache/invalidate",
        json={"namespace": "x"},
        headers={"Authorization": "Basic secret-abc"},
    )
    assert resp.status_code == 403


def test_empty_token_env_var_is_dev_mode(
    client: TestClient, _clean_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``PFM_ADMIN_TOKEN=""`` is treated as unset (dev mode)."""
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "")
    resp = client.post("/admin/cache/invalidate", json={"namespace": "anything"})
    assert resp.status_code == 200


# ─────────────────────── introspection / unit ──────────────────


def test_iter_cache_pools_finds_injected_pool(_clean_modules: None) -> None:
    """Discovery helper must surface CachePool attributes on pfm.* modules."""
    pool = CachePool(namespace="t-inv-iter", l1_maxsize=4)
    _install_fake_module("fake_inv_iter", _CACHE=pool)
    found = _iter_cache_pools()
    namespaces = [p._namespace for (_m, _a, p) in found]
    assert "t-inv-iter" in namespaces


def test_iter_cache_pools_skips_admin_modules(_clean_modules: None) -> None:
    """Pools registered on ``pfm.admin.*`` must NOT be discovered."""
    pool = CachePool(namespace="t-inv-admin-skip", l1_maxsize=4)
    _install_fake_module("admin.fake_admin_skip", _CACHE=pool)
    found = _iter_cache_pools()
    namespaces = [p._namespace for (_m, _a, p) in found]
    assert "t-inv-admin-skip" not in namespaces


def test_iter_cache_pools_deduplicates_reexports(_clean_modules: None) -> None:
    """A pool re-exported from two modules counts once."""
    pool = CachePool(namespace="t-inv-dedup", l1_maxsize=4)
    _install_fake_module("fake_inv_dedup_a", _CACHE=pool)
    _install_fake_module("fake_inv_dedup_b", _CACHE=pool, _ALIAS=pool)
    found = _iter_cache_pools()
    matches = [(m, a) for (m, a, p) in found if p._namespace == "t-inv-dedup"]
    assert len(matches) == 1


def test_perform_invalidation_direct(_clean_modules: None) -> None:
    """Drive the core function without HTTP — same outcome shape."""
    pool = CachePool(namespace="t-inv-direct", l1_maxsize=8)
    pool.set("p:1", 1, ttl=60)
    pool.set("p:2", 2, ttl=60)
    pool.set("q:1", 3, ttl=60)
    _install_fake_module("fake_inv_direct", _CACHE=pool)

    payload = perform_invalidation(CacheInvalidateRequest(prefix="p:", namespace="t-inv-direct"))
    assert payload["total_removed"] == 2
    row = next(r for r in payload["results"] if r["namespace"] == "t-inv-direct")
    assert row["removed"] == 2
    assert row["remaining"] == 1


def test_perform_invalidation_tolerates_broken_pool(
    _clean_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a pool's ``.clear`` raises, skip it but continue with the others.

    Partial invalidation is more useful than a 500: an operator firefighting
    a stale cache wants the *other* pools cleared even if one is buggy.
    """
    good = CachePool(namespace="t-inv-good", l1_maxsize=4)
    bad = CachePool(namespace="t-inv-bad", l1_maxsize=4)
    good.set("p:1", 1, ttl=60)
    bad.set("p:1", 1, ttl=60)

    def _explode(*_a: object, **_kw: object) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(bad, "clear", _explode)
    _install_fake_module("fake_inv_good", _CACHE=good)
    _install_fake_module("fake_inv_bad", _CACHE=bad)

    payload = perform_invalidation(CacheInvalidateRequest(prefix="p:"))
    namespaces = [r["namespace"] for r in payload["results"]]
    assert "t-inv-good" in namespaces
    assert "t-inv-bad" not in namespaces  # skipped due to exception
    good_row = next(r for r in payload["results"] if r["namespace"] == "t-inv-good")
    assert good_row["removed"] == 1
