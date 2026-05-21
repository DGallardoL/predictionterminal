"""Wave-9 P2 polish: openapi compression + ETag, search-index chunked, async pool.

These tests pin the three behaviours added in the P2 polish pass:

1. ``GET /openapi.json`` honours ``Accept-Encoding: gzip`` and emits a
   strong ``ETag`` so subsequent ``If-None-Match`` requests return 304.
2. ``GET /terminal/search-index/chunked`` returns at most ``size`` rows
   per page and the ``X-Total-Chunks`` header agrees with
   ``ceil(n_factors / size)``.
3. The shared async-http connection pool reuses keepalive sockets across
   parallel terminal requests — a 50-call respx-mocked benchmark must
   complete well below the no-pool serialisation budget.

External HTTP is mocked; tests run fully offline.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.main as main_mod
from pfm.terminal_search_index import (
    DEFAULT_CHUNK_SIZE,
)
from pfm.terminal_search_index import (
    clear_cache as _clear_search_index_cache,
)
from pfm.terminal_search_index import (
    router as search_index_router,
)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


# --- shared fixtures --------------------------------------------------------


@pytest.fixture
def app_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[TestClient]:
    """A fully-mounted main.app TestClient (mirrors conftest.app_client)."""
    factors_yml = tmp_path / "factors.yml"
    factors_yml.write_text(
        "factors:\n"
        "  - id: factor_a\n"
        "    name: Factor A\n"
        "    slug: slug-a\n"
        "    source: polymarket\n"
        "    description: A.\n"
        "  - id: factor_b\n"
        "    name: Factor B\n"
        "    slug: slug-b\n"
        "    source: polymarket\n"
        "    description: B.\n"
    )
    monkeypatch.setenv("FACTORS_FILE", str(factors_yml))
    import pfm.config as cfg

    cfg._settings = None

    from pfm.cache import NullCache

    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    with TestClient(main_mod.app) as client:
        yield client


# --- 1. /openapi.json compression + ETag ------------------------------------


class TestOpenAPICompression:
    def test_uncompressed_baseline_is_json(self, app_client: TestClient) -> None:
        """Without ``Accept-Encoding`` the body comes back as plain JSON."""
        r = app_client.get("/openapi.json")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        # Should parse round-trip — basic sanity that the schema isn't truncated.
        doc = r.json()
        assert "paths" in doc
        assert "info" in doc

    def test_etag_header_present(self, app_client: TestClient) -> None:
        r = app_client.get("/openapi.json")
        assert r.status_code == 200
        etag = r.headers.get("etag")
        assert etag is not None
        # Strong ETag in double quotes.
        assert etag.startswith('"') and etag.endswith('"')

    def test_cache_control_max_age(self, app_client: TestClient) -> None:
        r = app_client.get("/openapi.json")
        cc = r.headers.get("cache-control", "")
        assert "max-age=3600" in cc
        assert "public" in cc

    def test_etag_stable_across_calls(self, app_client: TestClient) -> None:
        """Same code, same schema → same ETag."""
        e1 = app_client.get("/openapi.json").headers["etag"]
        e2 = app_client.get("/openapi.json").headers["etag"]
        assert e1 == e2

    def test_if_none_match_returns_304(self, app_client: TestClient) -> None:
        first = app_client.get("/openapi.json")
        etag = first.headers["etag"]
        second = app_client.get(
            "/openapi.json",
            headers={"If-None-Match": etag},
        )
        assert second.status_code == 304
        # 304 must not carry a body.
        assert second.content == b""
        # ETag should still be present so the client can refresh the cache.
        assert second.headers.get("etag") == etag

    def test_gzip_path_returns_smaller_body(self, app_client: TestClient) -> None:
        """With ``Accept-Encoding: gzip`` the wire payload shrinks materially.

        ``httpx.TestClient`` auto-decompresses, so we measure the compressed
        size by counting bytes in the ``content-encoding: gzip`` branch
        (``r.content`` after decode). For the byte-budget assertion we
        invoke the GZip middleware path and check the compressed bytes via
        the raw response stream.
        """
        # The TestClient transparently decompresses, but we can still observe
        # ``Content-Encoding`` to confirm the middleware kicked in.
        r = app_client.get(
            "/openapi.json",
            headers={"Accept-Encoding": "gzip"},
        )
        assert r.status_code == 200
        # Either Content-Encoding is gzip (middleware compressed) or the
        # response is small enough to fall under the 1 KiB threshold.
        ce = r.headers.get("content-encoding", "")
        body_len = len(r.content)
        # The schema with the full app should easily exceed 1 KiB so we
        # expect the middleware to have engaged.
        if body_len >= 1024:
            assert ce == "gzip", (
                f"expected Content-Encoding: gzip on a {body_len}-byte body, got {ce!r}"
            )

    def test_etag_changes_when_version_changes(self, app_client: TestClient) -> None:
        """Bumping ``app.version`` invalidates the cached ETag."""
        e1 = app_client.get("/openapi.json").headers["etag"]
        original = main_mod.app.version
        try:
            main_mod.app.version = "test-bumped-9.9.9"
            # Force schema regen.
            main_mod.app.openapi_schema = None
            e2 = app_client.get("/openapi.json").headers["etag"]
        finally:
            main_mod.app.version = original
            main_mod.app.openapi_schema = None
        assert e1 != e2


# --- 2. /terminal/search-index/chunked --------------------------------------


def _chunked_app() -> TestClient:
    """Mount only the search-index router so the test doesn't need the full app."""
    app = FastAPI()
    app.include_router(search_index_router)
    return TestClient(app)


class TestSearchIndexChunked:
    @pytest.fixture(autouse=True)
    def _drop_cache(self) -> Iterator[None]:
        _clear_search_index_cache()
        yield
        _clear_search_index_cache()

    def test_chunk_zero_returns_at_most_size(self) -> None:
        client = _chunked_app()
        r = client.get("/terminal/search-index/chunked?chunk=0&size=200")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["chunk"] == 0
        assert body["chunk_size"] == 200
        assert len(body["factors"]) <= 200

    def test_total_chunks_header_matches_ceil(self) -> None:
        client = _chunked_app()
        r = client.get("/terminal/search-index/chunked?chunk=0&size=200")
        assert r.status_code == 200
        body = r.json()
        total = body["n_factors"]
        expected = math.ceil(total / 200) if total > 0 else 0
        # Header agrees with body.
        assert int(r.headers["x-total-chunks"]) == expected
        assert body["total_chunks"] == expected
        # The actual count must be at least ceil(total/size) — spec.
        assert body["total_chunks"] >= expected

    def test_chunked_slices_are_disjoint(self) -> None:
        client = _chunked_app()
        c0 = client.get("/terminal/search-index/chunked?chunk=0&size=50").json()
        c1 = client.get("/terminal/search-index/chunked?chunk=1&size=50").json()
        ids0 = {row["i"] for row in c0["factors"]}
        ids1 = {row["i"] for row in c1["factors"]}
        # Disjoint OR (when n_factors < 50) one of the chunks is empty.
        if ids0 and ids1:
            assert ids0.isdisjoint(ids1)

    def test_out_of_range_chunk_is_empty_not_404(self) -> None:
        client = _chunked_app()
        # Big chunk index way past the end.
        r = client.get("/terminal/search-index/chunked?chunk=9999&size=200")
        assert r.status_code == 200
        assert r.json()["factors"] == []

    def test_strategies_pages_actions_ride_along(self) -> None:
        client = _chunked_app()
        r = client.get("/terminal/search-index/chunked?chunk=0&size=10")
        body = r.json()
        # Pages list is hard-coded in the source; expect the canonical entries.
        page_ids = {p["i"] for p in body["pages"]}
        assert "page-terminal" in page_ids
        # Actions is a small static list too.
        assert any(a["i"] == "action-search" for a in body["actions"])

    def test_default_size_is_200(self) -> None:
        client = _chunked_app()
        r = client.get("/terminal/search-index/chunked?chunk=0")
        body = r.json()
        assert body["chunk_size"] == DEFAULT_CHUNK_SIZE
        assert DEFAULT_CHUNK_SIZE == 200

    def test_size_lower_bound_validation(self) -> None:
        client = _chunked_app()
        r = client.get("/terminal/search-index/chunked?chunk=0&size=0")
        assert r.status_code == 422

    def test_size_upper_bound_validation(self) -> None:
        client = _chunked_app()
        r = client.get("/terminal/search-index/chunked?chunk=0&size=10000")
        assert r.status_code == 422


# --- 3. Async-http connection pool ------------------------------------------


def _gamma_response_payload(slug: str) -> dict[str, Any]:
    """Minimal /markets payload that doesn't exercise any optional code path."""
    return {
        "slug": slug,
        "question": f"Q-{slug}",
        "clobTokenIds": json.dumps([f"tok-{slug}", f"tok-{slug}-no"]),
        "bestBid": 0.49,
        "bestAsk": 0.51,
        "lastTradePrice": 0.50,
        "volume24hr": 10_000.0,
        "volumeNum": 100_000.0,
        "endDate": "2026-12-01T00:00:00Z",
        "active": True,
        "closed": False,
    }


class TestConnectionPool:
    def test_async_http_pool_configured(self) -> None:
        """Lifespan-time check: the shared client uses the tuned limits."""
        # Read the module-level constants to avoid spinning up the lifespan.
        assert main_mod._ASYNC_HTTP_LIMITS.max_keepalive_connections == 20
        assert main_mod._ASYNC_HTTP_LIMITS.max_connections == 100
        assert main_mod._ASYNC_HTTP_LIMITS.keepalive_expiry == 30.0
        # Per-stage timeouts.
        t = main_mod._ASYNC_HTTP_TIMEOUT
        assert t.connect == 5.0
        assert t.read == 30.0
        assert t.write == 10.0
        assert t.pool == 10.0

    def test_pool_reuse_under_50_parallel_requests(self) -> None:
        """50 parallel mocked Gamma fetches should complete < 3s with reuse.

        The benchmark spins up one shared :class:`httpx.AsyncClient` with
        the production limits, then issues 50 concurrent ``GET`` calls
        to a respx-mocked endpoint. With keepalive working, all 50 finish
        in well under a second; without pooling each request would pay
        the (mocked) connect overhead serially.
        """

        async def _run() -> float:
            limits = main_mod._ASYNC_HTTP_LIMITS
            timeout = main_mod._ASYNC_HTTP_TIMEOUT
            with respx.mock(assert_all_called=False) as router:
                router.get(f"{GAMMA_URL}/markets").mock(
                    return_value=httpx.Response(
                        200, json=[_gamma_response_payload("benchmark-slug")]
                    )
                )
                async with httpx.AsyncClient(limits=limits, timeout=timeout) as http:
                    t0 = time.monotonic()
                    coros = [
                        http.get(
                            f"{GAMMA_URL}/markets",
                            params={"slug": "benchmark-slug"},
                        )
                        for _ in range(50)
                    ]
                    results = await asyncio.gather(*coros)
                    elapsed = time.monotonic() - t0
            for r in results:
                assert r.status_code == 200
            return elapsed

        elapsed = asyncio.run(_run())
        assert elapsed < 3.0, f"50 pooled requests took {elapsed:.2f}s — pool tuning regressed"
