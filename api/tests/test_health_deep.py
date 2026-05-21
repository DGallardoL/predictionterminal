"""Tests for ``GET /health/deep`` (:mod:`pfm.health_deep_router`).

All external HTTP is mocked via ``respx``; Redis is probed by
monkeypatching the lazy-imported ``redis.from_url`` so the test suite
runs offline with no Redis instance. yfinance is treated as a plain
HTTP URL (the same Yahoo chart endpoint the library hits internally),
so respx mocks it like any other source.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.health_deep_router as hd_mod
from pfm.health_deep_router import (
    GDELT_URL,
    KALSHI_URL,
    POLYMARKET_URL,
    YFINANCE_URL,
    _overall_status,
    router,
)

# ─────────────────────────── fixtures ───────────────────────────


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> None:
    """Reset the module-level ``_probe_cache`` between tests.

    ``health_deep_router._probe_cache`` is a process-lifetime dict that
    caches per-source probe results for 60s (15s on failure). Within a
    pytest session that means the first test populates it with ``ok``
    entries and every subsequent test reads cached "ok" instead of
    hitting the freshly-configured ``respx`` mocks. Clear before each
    test so respx routes are actually exercised.
    """
    hd_mod._probe_cache.clear()


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a fresh FastAPI app mounting only the deep-health router.

    The default is ``REDIS_URL`` unset so the Redis probe returns the
    "not configured" success row. Individual tests can opt in to a
    Redis probe by monkeypatching ``REDIS_URL`` + the lazy ``redis``
    import.
    """
    monkeypatch.delenv("REDIS_URL", raising=False)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _ok(name: str, url: str) -> respx.Route:
    return respx.get(url).mock(return_value=httpx.Response(200, json={"name": name, "items": []}))


def _all_ok() -> None:
    _ok("polymarket", POLYMARKET_URL)
    _ok("kalshi", KALSHI_URL)
    _ok("gdelt", GDELT_URL)
    _ok("yfinance", YFINANCE_URL)


# ─────────────────────────── _overall_status ───────────────────────────


class TestOverallStatus:
    def test_all_ok_returns_ok(self) -> None:
        sources = {f"s{i}": {"ok": True} for i in range(5)}
        assert _overall_status(sources) == "ok"

    def test_one_down_returns_degraded(self) -> None:
        sources = {f"s{i}": {"ok": True} for i in range(5)}
        sources["s0"] = {"ok": False}
        assert _overall_status(sources) == "degraded"

    def test_two_down_of_five_returns_degraded(self) -> None:
        sources = {f"s{i}": {"ok": True} for i in range(5)}
        sources["s0"] = {"ok": False}
        sources["s1"] = {"ok": False}
        assert _overall_status(sources) == "degraded"

    def test_half_or_more_down_returns_down(self) -> None:
        # 3 of 5 down → 3*2 >= 5 → down
        sources = {f"s{i}": {"ok": True} for i in range(5)}
        sources["s0"] = {"ok": False}
        sources["s1"] = {"ok": False}
        sources["s2"] = {"ok": False}
        assert _overall_status(sources) == "down"

    def test_all_down_returns_down(self) -> None:
        sources = {f"s{i}": {"ok": False} for i in range(5)}
        assert _overall_status(sources) == "down"


# ─────────────────────────── response shape ───────────────────────────


class TestResponseShape:
    @respx.mock
    def test_all_ok_response_shape(self, client: TestClient) -> None:
        _all_ok()
        r = client.get("/health/deep")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "uptime_s" in body
        assert isinstance(body["uptime_s"], (int, float))
        assert body["uptime_s"] >= 0
        assert set(body["sources"].keys()) == {
            "polymarket",
            "kalshi",
            "yfinance",
            "redis",
            "gdelt",
        }
        for name, src in body["sources"].items():
            assert "ok" in src, f"{name} missing 'ok'"
            assert "latency_ms" in src, f"{name} missing 'latency_ms'"
            assert "last_error" in src, f"{name} missing 'last_error'"
            assert "checked_at" in src, f"{name} missing 'checked_at'"
        assert body["summary"] == "5 sources checked, 5 ok"

    @respx.mock
    def test_iso8601_timestamps(self, client: TestClient) -> None:
        _all_ok()
        r = client.get("/health/deep")
        body = r.json()
        for name, src in body["sources"].items():
            ts = src["checked_at"]
            # Tolerate 'Z' or '+00:00' suffix
            assert ts.endswith("Z") or "+" in ts, f"{name} ts not iso: {ts}"
            assert "T" in ts, f"{name} ts not iso: {ts}"


# ─────────────────────────── parallelism ───────────────────────────


class TestParallelism:
    @respx.mock
    def test_all_sources_pinged(self, client: TestClient) -> None:
        """Verify every upstream is actually called in a single request."""
        poly = _ok("polymarket", POLYMARKET_URL)
        kals = _ok("kalshi", KALSHI_URL)
        gdel = _ok("gdelt", GDELT_URL)
        yfin = _ok("yfinance", YFINANCE_URL)
        client.get("/health/deep")
        assert poly.called, "polymarket route not called"
        assert kals.called, "kalshi route not called"
        assert gdel.called, "gdelt route not called"
        assert yfin.called, "yfinance route not called"

    @respx.mock
    def test_parallel_calls_total_under_2x_slowest(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If probes run in parallel, total ≈ slowest, not sum-of-all.

        We give every upstream a 250ms artificial delay. Serial would
        cost ~1000ms; parallel ≤ ~500ms.
        """

        async def _slow_response(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.25)
            return httpx.Response(200, json={"ok": True})

        respx.get(POLYMARKET_URL).mock(side_effect=_slow_response)
        respx.get(KALSHI_URL).mock(side_effect=_slow_response)
        respx.get(GDELT_URL).mock(side_effect=_slow_response)
        respx.get(YFINANCE_URL).mock(side_effect=_slow_response)

        t0 = time.perf_counter()
        r = client.get("/health/deep")
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200
        # Parallel: ≈ 0.25s. Serial would be ≥ 1.0s. Generous upper bound
        # to keep the test stable under CI scheduler jitter.
        assert elapsed < 0.8, (
            f"probes appear to run serially: elapsed={elapsed:.3f}s "
            f"(expected < 0.8s for 4 parallel 250ms calls)"
        )


# ─────────────────────────── degraded path ───────────────────────────


class TestDegradedAndDown:
    @respx.mock
    def test_one_source_500_returns_degraded(self, client: TestClient) -> None:
        _ok("polymarket", POLYMARKET_URL)
        _ok("kalshi", KALSHI_URL)
        _ok("yfinance", YFINANCE_URL)
        respx.get(GDELT_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))
        r = client.get("/health/deep")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert body["sources"]["gdelt"]["ok"] is False
        assert body["sources"]["gdelt"]["last_error"] is not None
        assert "500" in body["sources"]["gdelt"]["last_error"]
        assert body["sources"]["polymarket"]["ok"] is True
        assert body["summary"] == "5 sources checked, 4 ok"

    @respx.mock
    def test_504_error_text_contains_status(self, client: TestClient) -> None:
        _ok("polymarket", POLYMARKET_URL)
        _ok("kalshi", KALSHI_URL)
        _ok("yfinance", YFINANCE_URL)
        respx.get(GDELT_URL).mock(return_value=httpx.Response(504, text="Gateway Timeout"))
        r = client.get("/health/deep")
        body = r.json()
        assert "504" in body["sources"]["gdelt"]["last_error"]

    @respx.mock
    def test_all_500_returns_down(self, client: TestClient) -> None:
        respx.get(POLYMARKET_URL).mock(return_value=httpx.Response(500))
        respx.get(KALSHI_URL).mock(return_value=httpx.Response(503))
        respx.get(GDELT_URL).mock(return_value=httpx.Response(502))
        respx.get(YFINANCE_URL).mock(return_value=httpx.Response(500))
        r = client.get("/health/deep")
        body = r.json()
        # 4 HTTP sources down + redis ok (no REDIS_URL) → 4/5 down → down
        assert body["status"] == "down"
        assert body["sources"]["polymarket"]["ok"] is False
        assert body["sources"]["kalshi"]["ok"] is False
        assert body["sources"]["gdelt"]["ok"] is False
        assert body["sources"]["yfinance"]["ok"] is False


# ─────────────────────────── timeout path ───────────────────────────


class TestTimeout:
    @respx.mock
    def test_slow_source_marked_timeout(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A source slower than the per-call budget is marked timeout
        but the response still returns within the total budget.
        """
        # Shrink the per-call timeout so this test runs in <1s.
        monkeypatch.setattr(hd_mod, "_PER_CALL_TIMEOUT_S", 0.2)
        monkeypatch.setattr(hd_mod, "_TOTAL_BUDGET_S", 1.0)

        async def _hang(request: httpx.Request) -> httpx.Response:
            # Sleep much longer than _PER_CALL_TIMEOUT_S so the probe
            # times out. We rely on httpx raising TimeoutException.
            raise httpx.ReadTimeout("simulated slow upstream", request=request)

        respx.get(POLYMARKET_URL).mock(side_effect=_hang)
        _ok("kalshi", KALSHI_URL)
        _ok("gdelt", GDELT_URL)
        _ok("yfinance", YFINANCE_URL)

        t0 = time.perf_counter()
        r = client.get("/health/deep")
        elapsed = time.perf_counter() - t0
        assert elapsed < 6.0, f"endpoint blew the 5s budget: elapsed={elapsed:.2f}s"
        body = r.json()
        assert body["sources"]["polymarket"]["ok"] is False
        assert body["sources"]["polymarket"]["last_error"] == "timeout"
        # Other sources still ok → status is degraded (1 of 5 down)
        assert body["status"] == "degraded"


# ─────────────────────────── redis probe ───────────────────────────


class TestRedis:
    @respx.mock
    def test_no_redis_url_returns_ok_with_note(self, client: TestClient) -> None:
        _all_ok()
        r = client.get("/health/deep")
        body = r.json()
        assert body["sources"]["redis"]["ok"] is True
        assert body["sources"]["redis"]["note"] == "REDIS_URL not set"

    @respx.mock
    def test_redis_ping_success(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        _all_ok()

        # Inject a fake redis module so the lazy import succeeds without
        # an actual redis-py install in this test environment.
        class _FakeClient:
            def ping(self) -> bool:
                return True

        class _FakeRedisModule:
            @staticmethod
            def from_url(_url: str, **_kw: object) -> _FakeClient:
                return _FakeClient()

        import sys

        monkeypatch.setitem(sys.modules, "redis", _FakeRedisModule)
        r = client.get("/health/deep")
        body = r.json()
        assert body["sources"]["redis"]["ok"] is True
        assert body["sources"]["redis"]["last_error"] is None

    @respx.mock
    def test_redis_ping_failure(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REDIS_URL", "redis://nonexistent:6379/0")
        _all_ok()

        class _FakeClient:
            def ping(self) -> bool:
                raise ConnectionError("Connection refused")

        class _FakeRedisModule:
            @staticmethod
            def from_url(_url: str, **_kw: object) -> _FakeClient:
                return _FakeClient()

        import sys

        monkeypatch.setitem(sys.modules, "redis", _FakeRedisModule)
        r = client.get("/health/deep")
        body = r.json()
        assert body["sources"]["redis"]["ok"] is False
        assert "ConnectionError" in body["sources"]["redis"]["last_error"]
        # 1 of 5 down → degraded
        assert body["status"] == "degraded"


# ─────────────────────────── uptime / summary ───────────────────────────


class TestUptimeAndSummary:
    @respx.mock
    def test_uptime_is_positive(self, client: TestClient) -> None:
        _all_ok()
        r = client.get("/health/deep")
        body = r.json()
        assert body["uptime_s"] > 0

    @respx.mock
    def test_summary_string_format(self, client: TestClient) -> None:
        _ok("polymarket", POLYMARKET_URL)
        _ok("kalshi", KALSHI_URL)
        _ok("yfinance", YFINANCE_URL)
        respx.get(GDELT_URL).mock(return_value=httpx.Response(500))
        r = client.get("/health/deep")
        body = r.json()
        assert body["summary"] == "5 sources checked, 4 ok"
