"""Verify sources-health probes run in parallel via asyncio.gather.

Each mocked probe sleeps for 1.0–1.5 s. With 6 probes:

- sequential   → ~7+ s wall-clock
- parallelized → ~max(per-probe) ≈ ~1.5 s

We patch the async probe registry to deterministic sleepers and check
the overall ``acheck_all_sources`` wall-clock against a tight bound.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from pfm.sources import health as health_mod


def _make_sleeper(delay: float, ok: bool = True):
    async def probe() -> dict:
        await asyncio.sleep(delay)
        return {
            "ok": ok,
            "latency_ms": round(delay * 1000, 2),
            "detail": None if ok else "down",
            "configured": True,
        }

    return probe


def test_acheck_all_sources_runs_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = {
        "yfinance": 1.0,
        "tiingo": 1.2,
        "stooq": 1.5,
        "polymarket": 1.1,
        "kalshi": 1.3,
        "fred": 1.4,
    }
    fake_registry = {name: _make_sleeper(d) for name, d in delays.items()}
    monkeypatch.setattr(health_mod, "ASYNC_SOURCE_CHECKS", fake_registry)

    start = time.perf_counter()
    out = asyncio.run(health_mod.acheck_all_sources())
    elapsed = time.perf_counter() - start

    # Sum of delays is 7.5s; max is 1.5s. Allow 2.5s slack for scheduler
    # / venv overhead but well under the sequential floor.
    assert elapsed < 4.0, f"too slow ({elapsed:.2f}s) — probes not parallel?"
    assert elapsed >= 1.4, f"unexpectedly fast ({elapsed:.2f}s)"

    assert set(out.keys()) == set(delays.keys())
    for payload in out.values():
        assert payload["ok"] is True
        assert payload["latency_ms"] is not None


def test_acheck_all_sources_isolates_probe_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If one probe raises, the others still report normally."""

    async def boom() -> dict:
        raise RuntimeError("upstream blew up")

    fake_registry = {
        "yfinance": _make_sleeper(0.1),
        "tiingo": _make_sleeper(0.1),
        "stooq": boom,
        "polymarket": _make_sleeper(0.1),
        "kalshi": _make_sleeper(0.1),
        "fred": _make_sleeper(0.1),
    }
    monkeypatch.setattr(health_mod, "ASYNC_SOURCE_CHECKS", fake_registry)

    out = asyncio.run(health_mod.acheck_all_sources())
    assert out["stooq"]["ok"] is False
    assert "RuntimeError" in (out["stooq"]["detail"] or "")
    for name in ("yfinance", "tiingo", "polymarket", "kalshi", "fred"):
        assert out[name]["ok"] is True
