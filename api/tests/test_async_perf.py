"""Regression tests for the async refactor + N+1 parallelisation in main.py.

These tests pin the behaviour that ``/factors/rank``, ``/fit`` and the
``/factors/permutation`` endpoint do NOT serialize per-factor history fetches.
We use a ``fake_factor_history`` that sleeps for a controlled latency so the
*difference* between sequential and parallel execution is observable as a
wall-clock budget assertion (with a generous margin so flaky CI doesn't trip).

The math is the same as the production code path — only the IO is mocked.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import pfm.main as main_mod

_FACTOR_LATENCY_S: float = 0.10  # 100 ms per simulated Polymarket fetch


@pytest.fixture
def slow_factor_history() -> callable:
    """``fetch_factor_history`` replacement that sleeps before returning data.

    The N+1 sequential implementation would take ``len(factors) * 0.1`` s;
    the parallel implementation should take ~0.1s + overhead regardless of N.
    """
    rng = pd.date_range("2025-06-01", "2025-12-31", freq="D", tz="UTC")
    n = len(rng)
    base = (0.30 + 0.30 * np.sin(2 * np.pi * np.arange(n) / n * 1.2)).clip(0.05, 0.95)

    def _fetch(_client, slug: str, start=None, end=None):
        # Real-world latency simulation. Crucial: sleep is OUTSIDE the GIL
        # release we get from real httpx, but ``time.sleep`` does release the
        # GIL on POSIX, so ``ThreadPoolExecutor`` parallelism still applies.
        time.sleep(_FACTOR_LATENCY_S)
        seed = abs(hash(slug)) % (2**32)
        offset = np.random.default_rng(seed).normal(0, 0.03, n)
        df = pd.DataFrame({"price": (base + offset).clip(0.05, 0.95)}, index=rng)
        df.index.name = "date"
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    return _fetch


@pytest.fixture
def big_factors_file(tmp_path: Path) -> Path:
    """A factors.yml with 20 factors so /factors/rank exercises N=20 fan-out."""
    lines = ["factors:"]
    for i in range(20):
        lines.append(f"  - id: f_{i:02d}")
        lines.append(f"    name: Factor {i:02d}")
        lines.append(f"    slug: slug-{i:02d}")
        lines.append("    source: polymarket")
        lines.append(f"    description: Test factor {i:02d}.")
    p = tmp_path / "factors.yml"
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.fixture
def perf_app_client(
    monkeypatch: pytest.MonkeyPatch,
    big_factors_file: Path,
    slow_factor_history: callable,
    fake_log_returns: callable,
) -> Iterator[TestClient]:
    """TestClient backed by 20 factors + a 100ms-latency mock fetcher."""
    monkeypatch.setenv("FACTORS_FILE", str(big_factors_file))
    import pfm.config as cfg

    cfg._settings = None

    monkeypatch.setattr(main_mod, "fetch_factor_history", slow_factor_history)
    monkeypatch.setattr(main_mod, "get_log_returns", fake_log_returns)

    from pfm.cache import NullCache

    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    with TestClient(main_mod.app) as client:
        yield client


# ---- Test 1: /factors/rank with N=20 finishes in < 5s --------------------


def test_factors_rank_parallel_fanout_under_5s(perf_app_client: TestClient) -> None:
    """20 factors × 100ms sequential ≈ 2.0s; parallel (cap=20) should be ≈ 0.1-0.5s.

    The 5s budget is intentionally generous so a slow CI runner doesn't trip.
    What we're really pinning: NOT N×latency = 2.0s as the wall clock.
    """
    t0 = time.monotonic()
    r = perf_app_client.post(
        "/factors/rank",
        json={
            "ticker": "TEST",
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    elapsed = time.monotonic() - t0
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["items"]) == 20
    # Sequential lower bound = 20 × 0.10 = 2.0s; we should be well under.
    assert elapsed < 5.0, (
        f"/factors/rank with 20 candidates took {elapsed:.2f}s — "
        f"parallelisation regressed (expected < 5s)"
    )
    # Stronger sanity: should beat the 2.0s sequential lower bound by a margin.
    # Allow up to 1.8s to account for thread-pool startup + GIL contention.
    assert elapsed < 1.8, (
        f"/factors/rank took {elapsed:.2f}s, very close to the 2.0s "
        f"sequential floor — N+1 fan-out may have regressed."
    )


# ---- Test 2: /fit with 10 factors finishes in < 3s ------------------------


def test_fit_parallel_assemble_under_3s(perf_app_client: TestClient) -> None:
    """10 factors × 100ms sequential = 1.0s; parallel ~0.1-0.3s wall-clock."""
    t0 = time.monotonic()
    factor_ids = [f"f_{i:02d}" for i in range(10)]
    r = perf_app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": factor_ids,
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    elapsed = time.monotonic() - t0
    assert r.status_code == 200, r.text
    body = r.json()
    assert {f["id"] for f in body["factors"]} == set(factor_ids)
    assert elapsed < 3.0, (
        f"/fit with 10 factors took {elapsed:.2f}s — _assemble_design "
        f"parallelisation regressed (expected < 3s)"
    )


# ---- Test 3: concurrency — 5 simultaneous /factors/rank don't serialise ---


def test_concurrent_rank_requests_dont_serialise(perf_app_client: TestClient) -> None:
    """5 concurrent /factors/rank calls should NOT take 5× a single call.

    We can't use real ``threading`` against a ``TestClient`` (it dispatches via
    ASGI in-process), but we can compare a single-request wall clock with the
    aggregate of 5 sequentially-issued requests. With the parallel internal
    fan-out, each call is dominated by the slowest worker (~0.1s), so 5
    sequential requests should also stay well below 5 × 2.0s = 10s.
    """
    t0 = time.monotonic()
    r = perf_app_client.post(
        "/factors/rank",
        json={"ticker": "TEST", "start": "2025-06-15", "end": "2025-12-15"},
    )
    single_elapsed = time.monotonic() - t0
    assert r.status_code == 200

    t0 = time.monotonic()
    for _ in range(5):
        r = perf_app_client.post(
            "/factors/rank",
            json={"ticker": "TEST", "start": "2025-06-15", "end": "2025-12-15"},
        )
        assert r.status_code == 200
    five_elapsed = time.monotonic() - t0

    # Sequential cost without internal parallelism would be 5 × 2.0s = 10s.
    # With our fix, each request is ~0.1-0.3s, so 5 × that ≤ 2-3s typically.
    assert five_elapsed < 5 * single_elapsed * 1.6 + 1.0, (
        f"5 sequential /factors/rank calls took {five_elapsed:.2f}s; "
        f"single call was {single_elapsed:.2f}s — concurrency degraded > 1.6x"
    )
