"""Tests for the ``GET /metrics/audit`` operations endpoint.

These tests pin the contract of :mod:`pfm.metrics` + :mod:`pfm.metrics_router`:

* The endpoint is itself excluded from the latency-audit sample (no recursion).
* Per-path stats keyed by templated route, never raw paths-with-slugs.
* ``p50 <= p95 <= p99`` invariant holds.
* Hammering one endpoint 100x does not create new keys.
* Empty buffer returns the documented shape ``{"endpoints": {}, "total_requests": 0}``.

Driven entirely by ``TestClient`` so no gunicorn restart is required.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import pfm.main as main_mod
from pfm.metrics import LatencyTracker, _percentiles, get_tracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_tracker() -> Iterator[None]:
    """Reset the process-wide tracker before AND after each test."""
    get_tracker().clear()
    yield
    get_tracker().clear()


@pytest.fixture
def client(_clean_tracker: None) -> Iterator[TestClient]:
    with TestClient(main_mod.app) as c:
        yield c


# ---------------------------------------------------------------------------
# LatencyTracker unit tests
# ---------------------------------------------------------------------------


def test_tracker_empty_snapshot_is_empty_dict() -> None:
    """A fresh tracker yields ``{}`` and reports zero total requests."""
    t = LatencyTracker()
    assert t.snapshot() == {}
    assert t.total_requests() == 0


def test_tracker_record_groups_by_path() -> None:
    t = LatencyTracker()
    t.record("/a", 1.0, 200)
    t.record("/a", 2.0, 200)
    t.record("/b", 5.0, 500)
    snap = t.snapshot()
    assert set(snap) == {"/a", "/b"}
    assert snap["/a"]["count"] == 2
    assert snap["/b"]["count"] == 1
    assert snap["/b"]["err_rate"] == 1.0
    assert snap["/b"]["errors_5xx"] == 1
    assert snap["/a"]["err_rate"] == 0.0


def test_tracker_percentile_ordering_holds() -> None:
    """``p50 <= p95 <= p99`` must hold for any non-empty sample."""
    t = LatencyTracker()
    # Intentionally noisy distribution (one fat-tail outlier at 1000).
    for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]:
        t.record("/x", float(ms), 200)
    s = t.snapshot()["/x"]
    assert s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"]


def test_tracker_known_percentiles_match_numpy_definition() -> None:
    """Sanity-check that we use linear interpolation (numpy default)."""
    t = LatencyTracker()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        t.record("/p", v, 200)
    s = t.snapshot()["/p"]
    # For [1..5] with linear interp: p50=3, p95=4.8, p99=4.96.
    assert s["p50_ms"] == pytest.approx(3.0, abs=1e-3)
    assert s["p95_ms"] == pytest.approx(4.8, abs=1e-3)
    assert s["p99_ms"] == pytest.approx(4.96, abs=1e-3)


def test_tracker_err_rate_5xx_only() -> None:
    """Only ``code >= 500`` counts as an error; 4xx does not."""
    t = LatencyTracker()
    t.record("/x", 1.0, 200)
    t.record("/x", 1.0, 404)
    t.record("/x", 1.0, 429)
    t.record("/x", 1.0, 500)
    t.record("/x", 1.0, 502)
    s = t.snapshot()["/x"]
    assert s["errors_5xx"] == 2
    assert s["err_rate"] == pytest.approx(0.4, abs=1e-4)


def test_tracker_maxlen_caps_buffer() -> None:
    """The ring buffer drops the oldest observation once at maxlen."""
    t = LatencyTracker(maxlen=5)
    for i in range(20):
        t.record("/q", float(i), 200)
    assert t.total_requests() == 5
    s = t.snapshot()["/q"]
    # Only the last five (15..19) should be considered.
    assert s["count"] == 5
    # min of remaining is 15, max is 19; p50 is the median 17.
    assert s["p50_ms"] == pytest.approx(17.0, abs=1e-3)


def test_percentiles_empty_returns_zeros() -> None:
    assert _percentiles([], (50.0, 95.0, 99.0)) == (0.0, 0.0, 0.0)


def test_percentiles_single_value_is_constant() -> None:
    assert _percentiles([42.0], (50.0, 95.0, 99.0)) == (42.0, 42.0, 42.0)


# ---------------------------------------------------------------------------
# /metrics/audit endpoint integration tests
# ---------------------------------------------------------------------------


def test_audit_endpoint_returns_200_and_empty_shape(client: TestClient) -> None:
    """Endpoint must return the documented shape even when nothing was sampled.

    NOTE: we call the audit endpoint FIRST so no other endpoints have populated
    the tracker, then we expect ``endpoints={}, total_requests=0``.
    """
    r = client.get("/metrics/audit")
    assert r.status_code == 200
    body = r.json()
    assert body == {"endpoints": {}, "total_requests": 0}


def test_audit_endpoint_self_is_not_tracked(client: TestClient) -> None:
    """Hitting ``/metrics/audit`` itself must not populate its own snapshot."""
    for _ in range(5):
        client.get("/metrics/audit")
    body = client.get("/metrics/audit").json()
    assert "/metrics/audit" not in body["endpoints"]
    assert body["total_requests"] == 0


def test_audit_endpoint_records_other_endpoints(client: TestClient) -> None:
    """A call to ``/health`` must show up in the next audit snapshot."""
    r = client.get("/health")
    assert r.status_code == 200
    body = client.get("/metrics/audit").json()
    assert "/health" in body["endpoints"]
    stats = body["endpoints"]["/health"]
    assert stats["count"] == 1
    assert stats["err_rate"] == 0.0
    assert stats["errors_5xx"] == 0
    # All percentiles non-negative.
    for k in ("p50_ms", "p95_ms", "p99_ms"):
        assert stats[k] >= 0.0


def test_audit_repeated_calls_no_key_collision(client: TestClient) -> None:
    """100 hits to the same endpoint must produce exactly one key with count=100."""
    for _ in range(100):
        client.get("/health")
    body = client.get("/metrics/audit").json()
    # Only /health should appear (audit itself is skipped).
    keys = list(body["endpoints"].keys())
    assert keys == ["/health"], f"unexpected extra keys: {keys}"
    stats = body["endpoints"]["/health"]
    assert stats["count"] == 100
    # Percentile invariant.
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
    assert body["total_requests"] == 100


def test_audit_groups_templated_paths(client: TestClient) -> None:
    """``/health`` repeated and a 404 path map to distinct keys."""
    client.get("/health")
    client.get("/health")
    # Some path that won't match any FastAPI route — falls through 404.
    client.get("/definitely-not-a-route-abc-xyz")
    body = client.get("/metrics/audit").json()
    eps = body["endpoints"]
    assert "/health" in eps
    assert eps["/health"]["count"] == 2
    # 404 paths still get tracked (under the raw URL path since no route
    # matched and no template exists).
    assert any("definitely-not-a-route" in k for k in eps)


def test_audit_percentile_invariant_under_mixed_latencies(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthesise varying latencies via direct tracker calls; verify p50<=p95<=p99."""
    tracker = get_tracker()
    # Hit 5 distinct templated endpoints with 20 observations each, where
    # each endpoint has a different latency distribution.
    distributions: dict[str, list[float]] = {
        "/a": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 50.0, 100.0],
        "/b": [10.0] * 10,
        "/c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "/d": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        "/e": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    for path, latencies in distributions.items():
        for ms in latencies:
            tracker.record(path, ms, 200)
    body = client.get("/metrics/audit").json()
    eps = body["endpoints"]
    for path in distributions:
        s = eps[path]
        assert s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"], (
            f"percentile invariant violated for {path}: {s}"
        )
        assert s["count"] == 10
        assert s["err_rate"] == 0.0


def test_audit_records_5xx_status(client: TestClient) -> None:
    """Synthetic 5xx tracker observations should drive ``err_rate``."""
    tracker = get_tracker()
    for _ in range(7):
        tracker.record("/oops", 5.0, 503)
    for _ in range(3):
        tracker.record("/oops", 5.0, 200)
    body = client.get("/metrics/audit").json()
    s = body["endpoints"]["/oops"]
    assert s["count"] == 10
    assert s["errors_5xx"] == 7
    assert s["err_rate"] == pytest.approx(0.7, abs=1e-4)


def test_audit_endpoint_appears_in_openapi(client: TestClient) -> None:
    """The new route must show up under the OpenAPI schema."""
    spec: dict[str, Any] = client.get("/openapi.json").json()
    assert "/metrics/audit" in spec["paths"]
    op = spec["paths"]["/metrics/audit"]["get"]
    assert op["responses"]["200"]


def test_audit_after_clear_returns_empty(client: TestClient) -> None:
    """Clearing the tracker mid-flight resets snapshot to empty."""
    client.get("/health")
    assert client.get("/metrics/audit").json()["total_requests"] >= 1
    get_tracker().clear()
    body = client.get("/metrics/audit").json()
    assert body == {"endpoints": {}, "total_requests": 0}
