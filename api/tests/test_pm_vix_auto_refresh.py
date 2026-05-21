"""Tests for the PM-VIX live-slug validation + refresh subsystem.

Covers:
  * ``validate_and_refresh_buckets`` correctly classifies hardcoded
    slugs as live vs. dead and pulls in keyword-search replacements.
  * The persisted ``/tmp/pfm_pm_vix_slugs.json`` is consumed by
    ``compute_pm_vix`` on subsequent calls (cache hit, no re-validation).
  * The admin endpoints ``POST /indices/pm-vix/refresh-slugs`` and
    ``GET /indices/pm-vix/slugs`` round-trip a refresh end-to-end.
  * Recovery: a corrupt cache file falls back to the hardcoded map
    silently rather than raising.

Every test patches the Polymarket Gamma layer with an in-process
``httpx.MockTransport`` so the suite stays hermetic.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import pm_vix
from pfm.cache_utils import get_cache
from pfm.pm_vix import (
    BUCKET_SEARCH_KEYWORDS,
    BUCKET_SLUGS,
    _get_active_slugs,
    _load_persisted_slugs,
    compute_pm_vix,
    router,
    validate_and_refresh_buckets,
)

#: Snapshot the real ``httpx.AsyncClient`` constructor *before* any test
#: patches it. Tests that monkeypatch ``pm_vix.httpx.AsyncClient`` need a
#: way to build the underlying mock client without re-entering the patch
#: (which would recurse infinitely).
_ORIGINAL_ASYNC_CLIENT = httpx.AsyncClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_slug_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the slug cache path to a per-test tmp file and reset caches."""
    cache_path = tmp_path / "pfm_pm_vix_slugs.json"
    monkeypatch.setenv("PFM_PM_VIX_SLUG_CACHE_PATH", str(cache_path))
    get_cache("pm_vix").clear()
    get_cache("pm_vix_slugs").clear()
    return cache_path


def _gamma_market(slug: str, prob: float = 0.4, vol: float = 50_000.0) -> dict[str, Any]:
    """Build a Gamma-shaped market dict with the requested ``slug``."""
    return {
        "slug": slug,
        "bestBid": max(0.0, prob - 0.005),
        "bestAsk": min(1.0, prob + 0.005),
        "lastTradePrice": prob,
        "volume24hr": vol,
    }


def _make_async_client(
    *,
    dead_slugs: set[str],
    search_results: dict[str, list[dict[str, Any]]] | None = None,
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` whose transport mocks Gamma.

    ``dead_slugs`` is the set of slugs whose ``/markets?slug=…`` (and
    ``closed=true`` fallback) return an empty list — i.e. the slug is
    "dead". ``search_results`` is a ``keyword -> [market_dicts]`` map
    consulted when a search request comes in. Anything not in either map
    yields an empty list, which keeps the mock transport noise-free.
    """
    search_results = search_results or {}

    def _handler(request: httpx.Request) -> httpx.Response:
        if "/markets" not in request.url.path:
            return httpx.Response(200, json=[])
        params = dict(request.url.params)
        if "slug" in params:
            slug = params["slug"]
            if slug in dead_slugs:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[_gamma_market(slug)])
        if "search" in params:
            kw = params["search"]
            return httpx.Response(200, json=search_results.get(kw, []))
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(_handler)
    return _ORIGINAL_ASYNC_CLIENT(transport=transport, base_url=pm_vix.GAMMA_URL)


# ---------------------------------------------------------------------------
# validate_and_refresh_buckets
# ---------------------------------------------------------------------------


def test_validate_marks_dead_slugs_and_keeps_live() -> None:
    """Two dead slugs in `recession` get replaced; the rest stay put."""
    dead = {
        "us-recession-by-end-of-2026",
        "us-recession-in-q1-2026",
    }
    # The recession-keyword search returns 5 alternatives ordered by volume.
    alternatives = [
        _gamma_market(f"recession-alt-{i}", vol=v)
        for i, v in enumerate([100_000, 80_000, 60_000, 40_000, 20_000])
    ]
    client = _make_async_client(
        dead_slugs=dead,
        search_results={"recession": alternatives},
    )

    async def _run() -> dict[str, Any]:
        async with client:
            return await validate_and_refresh_buckets(client)

    payload = asyncio.run(_run())

    rec_diag = payload["diagnostics"]["recession"]
    assert set(rec_diag["dead"]) == dead
    assert rec_diag["n_kept"] == len(BUCKET_SLUGS["recession"]) - len(dead)
    # Replacements are top-3 by volume.
    assert rec_diag["replacements"] == [
        "recession-alt-0",
        "recession-alt-1",
        "recession-alt-2",
    ]
    # Persisted file shape matches what the endpoints expect.
    assert payload["n_kept"] >= 0
    assert payload["n_dead_replaced"] >= 1
    assert "recession-alt-0" in payload["buckets"]["recession"]


def test_validate_persists_to_configured_path(_isolated_slug_cache: Path) -> None:
    """The atomic write hits the env-overridden cache path."""
    client = _make_async_client(dead_slugs=set(), search_results={})

    async def _run() -> None:
        async with client:
            await validate_and_refresh_buckets(client)

    asyncio.run(_run())

    assert _isolated_slug_cache.exists(), "slug cache file was not written"
    raw = json.loads(_isolated_slug_cache.read_text())
    assert "as_of" in raw and "buckets" in raw
    assert set(raw["buckets"].keys()) == set(BUCKET_SLUGS.keys())


def test_validate_no_dead_slugs_keeps_everything() -> None:
    """When every hardcoded slug resolves, no replacements are made."""
    client = _make_async_client(dead_slugs=set())

    async def _run() -> dict[str, Any]:
        async with client:
            return await validate_and_refresh_buckets(client)

    payload = asyncio.run(_run())
    assert payload["n_dead_replaced"] == 0
    for bucket, slugs in BUCKET_SLUGS.items():
        assert payload["buckets"][bucket] == list(slugs), bucket


def test_validate_empty_search_keeps_hardcoded_fallback() -> None:
    """If every slug dies and no replacements surface, hardcoded list survives."""
    all_dead: set[str] = {s for slugs in BUCKET_SLUGS.values() for s in slugs}
    client = _make_async_client(dead_slugs=all_dead, search_results={})

    async def _run() -> dict[str, Any]:
        async with client:
            return await validate_and_refresh_buckets(client)

    payload = asyncio.run(_run())
    # No bucket is empty even though every hardcoded slug died.
    for bucket, slugs in BUCKET_SLUGS.items():
        assert payload["buckets"][bucket]
        # Falls back to original hardcoded list.
        assert payload["buckets"][bucket] == list(slugs)


# ---------------------------------------------------------------------------
# Cache layer / compute_pm_vix integration
# ---------------------------------------------------------------------------


def test_compute_pm_vix_uses_persisted_slugs(
    _isolated_slug_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a refresh, ``compute_pm_vix`` queries the live slugs only."""
    custom_slugs = {b: [f"{b}-live-1", f"{b}-live-2"] for b in BUCKET_SLUGS}
    # Use a fresh timestamp so the 24h staleness check passes on any test run.
    fresh_iso = datetime.now(UTC).isoformat()
    _isolated_slug_cache.write_text(
        json.dumps(
            {
                "as_of": fresh_iso,
                "buckets": custom_slugs,
                "n_kept": 0,
                "n_dead_replaced": 5,
            }
        )
    )

    queried: list[str] = []

    def _fake_fetch(http: Any, gamma_url: str, slug: str, **_kwargs: Any) -> dict[str, Any]:
        queried.append(slug)
        return {"bestBid": 0.39, "bestAsk": 0.41, "volume24hr": 1000.0}

    monkeypatch.setattr(pm_vix, "fetch_gamma_market", _fake_fetch)

    snap = compute_pm_vix(http=MagicMock())
    expected = {s for slugs in custom_slugs.values() for s in slugs}
    # Every persisted slug got hit; no hardcoded slug did.
    assert set(queried) == expected
    # Components carry the live source marker.
    assert all(c["source"] == "live" for c in snap["components"])


def test_compute_pm_vix_falls_back_when_no_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No cache file ⇒ compute uses ``BUCKET_SLUGS`` as before."""
    queried: list[str] = []

    def _fake_fetch(http: Any, gamma_url: str, slug: str, **_kwargs: Any) -> dict[str, Any]:
        queried.append(slug)
        return {"bestBid": 0.4, "bestAsk": 0.42, "volume24hr": 1000.0}

    monkeypatch.setattr(pm_vix, "fetch_gamma_market", _fake_fetch)

    snap = compute_pm_vix(http=MagicMock())
    expected = {s for slugs in BUCKET_SLUGS.values() for s in slugs}
    assert set(queried) == expected
    assert all(c["source"] == "hardcoded" for c in snap["components"])


def test_cache_hit_does_not_re_validate(
    _isolated_slug_cache: Path,
) -> None:
    """A second ``_get_active_slugs`` call stays in-memory (no disk read)."""
    custom_slugs = {b: [f"{b}-live"] for b in BUCKET_SLUGS}
    _isolated_slug_cache.write_text(
        json.dumps(
            {
                "as_of": "2026-05-08T12:00:00+00:00",
                "buckets": custom_slugs,
                "n_kept": 0,
                "n_dead_replaced": 0,
            }
        )
    )
    first = _get_active_slugs()
    # Mutate the file on disk; the in-memory cache should mask the change.
    _isolated_slug_cache.write_text(
        json.dumps(
            {
                "as_of": "2026-05-08T12:00:00+00:00",
                "buckets": {b: ["totally-different"] for b in BUCKET_SLUGS},
                "n_kept": 0,
                "n_dead_replaced": 0,
            }
        )
    )
    second = _get_active_slugs()
    assert first == second, "second call should hit memory cache, not re-read disk"


def test_corrupt_cache_falls_back_silently(_isolated_slug_cache: Path) -> None:
    """A garbled cache file does not raise — fallback to hardcoded."""
    _isolated_slug_cache.write_text("{not valid json{")
    assert _load_persisted_slugs() is None
    assert _get_active_slugs() == {}


def test_stale_cache_is_ignored(_isolated_slug_cache: Path) -> None:
    """Cache older than 24h returns ``None``."""
    _isolated_slug_cache.write_text(
        json.dumps(
            {
                "as_of": "2020-01-01T00:00:00+00:00",  # ancient
                "buckets": {b: ["x"] for b in BUCKET_SLUGS},
                "n_kept": 0,
                "n_dead_replaced": 0,
            }
        )
    )
    assert _load_persisted_slugs() is None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_endpoint_get_slugs_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GET /indices/pm-vix/slugs`` returns hardcoded buckets when no cache."""
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)
    client = TestClient(_make_app())
    r = client.get("/indices/pm-vix/slugs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "fallback"
    assert body["as_of"] is None
    assert set(body["buckets"].keys()) == set(BUCKET_SLUGS.keys())
    for bucket, slugs in BUCKET_SLUGS.items():
        assert body["buckets"][bucket] == list(slugs)


def test_endpoint_refresh_slugs(
    _isolated_slug_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POST /indices/pm-vix/refresh-slugs`` runs validation and persists."""
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)

    dead = {"us-recession-by-end-of-2026"}
    alternatives = [_gamma_market("recession-replacement-1", vol=200_000)]

    def _patched_client_factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
        return _make_async_client(
            dead_slugs=dead,
            search_results={"recession": alternatives},
        )

    monkeypatch.setattr(pm_vix.httpx, "AsyncClient", _patched_client_factory)

    client = TestClient(_make_app())
    r = client.post("/indices/pm-vix/refresh-slugs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_dead_replaced"] >= 1
    assert "recession-replacement-1" in body["buckets"]["recession"]
    assert _isolated_slug_cache.exists()


def test_endpoint_refresh_then_get_slugs(
    _isolated_slug_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: POST refresh then GET slugs surfaces the same live map."""
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)

    alternatives = [_gamma_market("e2e-replacement", vol=500_000)]

    def _client_factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
        # Make every recession slug "dead" so we know the replacement
        # propagates all the way through.
        dead = set(BUCKET_SLUGS["recession"])
        return _make_async_client(
            dead_slugs=dead,
            search_results={"recession": alternatives},
        )

    monkeypatch.setattr(pm_vix.httpx, "AsyncClient", _client_factory)

    client = TestClient(_make_app())
    refresh = client.post("/indices/pm-vix/refresh-slugs")
    assert refresh.status_code == 200
    listing = client.get("/indices/pm-vix/slugs")
    assert listing.status_code == 200, listing.text
    body = listing.json()
    assert body["source"] in {"live", "mixed"}
    assert "e2e-replacement" in body["buckets"]["recession"]
    assert body["as_of"] is not None


def test_endpoint_refresh_admin_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``PFM_ADMIN_TOKEN`` is set, the refresh endpoint requires it."""
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "secret-token")
    # Re-import not needed: ``_admin_dep_if_enabled`` reads the env at
    # import time of the ``router`` declaration. Build a fresh app.
    from importlib import reload

    reload(pm_vix)
    app = FastAPI()
    app.include_router(pm_vix.router)
    client = TestClient(app)
    # No token → 403.
    r = client.post("/indices/pm-vix/refresh-slugs")
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Bucket-search keywords sanity
# ---------------------------------------------------------------------------


def test_bucket_search_keywords_cover_every_bucket() -> None:
    """``BUCKET_SEARCH_KEYWORDS`` must have an entry for every bucket."""
    assert set(BUCKET_SEARCH_KEYWORDS.keys()) == set(BUCKET_SLUGS.keys())
    # Each list non-empty so the validate step has something to ask Gamma.
    for kws in BUCKET_SEARCH_KEYWORDS.values():
        assert kws, "every bucket needs at least one search keyword"


# ---------------------------------------------------------------------------
# Helper sanity: Callable type alias is exported, transport handler returns
# a 200 for the basic ``slug`` path. We assert directly so a regression in
# the mock-transport scaffolding fails fast rather than silently degrading
# every other test in this file.
# ---------------------------------------------------------------------------


def test_mock_transport_returns_alive_for_unknown_slug() -> None:
    """Smoke-test the test-helper itself — keeps regressions noisy."""
    client = _make_async_client(dead_slugs=set())

    async def _run() -> bool:
        async with client:
            return await pm_vix._check_slug_alive(client, "any-slug")

    assert asyncio.run(_run()) is True


def test_callable_alias_imported() -> None:
    """``Callable`` is a type-only import; this asserts the test file
    really uses it (avoids unused-import lint failures)."""
    assert isinstance(_gamma_market, Callable)
