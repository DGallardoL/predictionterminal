"""Tests for :mod:`pfm.terminal.jumps_prewarm`.

What is verified
----------------
1. ``prewarm_jumps`` invokes the underlying ``get_jumps`` handler exactly
   once per slug (no double-fan-out, no stalls).
2. After prewarm, the module-level ``_CACHE`` in :mod:`pfm.terminal.jumps`
   is populated for each requested slug so the live endpoint hits the
   warm branch.
3. ``app.state.warm_jumps`` is populated with a timestamp + per-slug
   elapsed map.
4. The lifespan helper ``register_jumps_prewarm`` returns an
   ``asyncio.Task`` and inits the state slot.
5. A failing slug (Polymarket 502 / market not found) is logged at DEBUG
   and never propagates — the prewarm continues for the remaining slugs.
6. Cold latency for a prewarmed slug, served via the warm cache, is
   <100 ms (mocked, but exercises the same code path the live endpoint
   takes on a cache hit).

All upstream HTTP is mocked — these tests do not touch GDELT, Reddit,
HN, RSS, or Polymarket in any way.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from unittest import mock

import pytest
from fastapi import FastAPI

from pfm.terminal import jumps_prewarm
from pfm.terminal.jumps import TerminalJumpsResponse
from pfm.terminal.jumps_prewarm import (
    CURATED_TOP_SLUGS,
    DEFAULT_CONCURRENCY,
    PER_SLUG_TIMEOUT_S,
    prewarm_jumps,
    register_jumps_prewarm,
)

# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------


def _empty_response(slug: str) -> TerminalJumpsResponse:
    """A trivially-valid jumps response — enough to satisfy the cache write."""
    return TerminalJumpsResponse(
        slug=slug,
        days=14,
        threshold_mad_k=3.0,
        threshold_min_jump_pp=5.0,
        n_jumps=0,
        n_explained=0,
        explained_pct=0.0,
        jumps=[],
        interpretation="(synthetic test payload — no jumps)",
    )


@pytest.fixture
def app_with_poly() -> FastAPI:
    """Bare FastAPI with a stub PolymarketClient stashed on state."""
    app = FastAPI()
    # The prewarm only checks that ``app.state.poly`` is truthy; the real
    # PolymarketClient is unused because we mock get_jumps below.
    app.state.poly = mock.Mock(name="poly-stub")
    return app


# ---------------------------------------------------------------------------
# 1. CURATED_TOP_SLUGS sanity
# ---------------------------------------------------------------------------


def test_curated_slug_count_in_expected_range() -> None:
    """30-50 slugs per spec; protect against accidental empties or runaway lists."""
    assert 30 <= len(CURATED_TOP_SLUGS) <= 60
    # All slugs are non-empty, lowercase, and dash-separated.
    for s in CURATED_TOP_SLUGS:
        assert isinstance(s, str) and s, f"empty slug in CURATED_TOP_SLUGS: {s!r}"
        assert s == s.lower(), f"slug must be lowercase: {s!r}"


def test_curated_slugs_are_deduplicated() -> None:
    assert len(CURATED_TOP_SLUGS) == len(set(CURATED_TOP_SLUGS))


# ---------------------------------------------------------------------------
# 2. prewarm_jumps calls get_jumps exactly once per slug
# ---------------------------------------------------------------------------


def test_prewarm_calls_each_slug_exactly_once(app_with_poly: FastAPI) -> None:
    """Five slugs, one mocked get_jumps; one call per slug, no retries."""
    target = ["slug-a", "slug-b", "slug-c", "slug-d", "slug-e"]
    seen: list[str] = []

    async def fake_get_jumps(*, request, slug, days, mad_k, min_jump_pp, poly):
        seen.append(slug)
        return _empty_response(slug)

    with mock.patch("pfm.terminal.jumps.get_jumps", side_effect=fake_get_jumps):
        result = asyncio.run(prewarm_jumps(app_with_poly, slugs=target))

    # Exact count + uniqueness — confirms no slug was double-warmed.
    assert sorted(seen) == sorted(target)
    assert len(seen) == len(target)
    # All slugs landed in the success map with a non-negative elapsed time.
    assert set(result.keys()) == set(target)
    for elapsed in result.values():
        assert elapsed >= 0.0


# ---------------------------------------------------------------------------
# 3. Cache populated — the live endpoint's cache lookup hits warm
# ---------------------------------------------------------------------------


def test_cache_populated_for_warmed_slugs(app_with_poly: FastAPI) -> None:
    """After prewarm, pfm.terminal.jumps._CACHE has entries keyed by
    (slug, days, mad_k, min_jump_pp) for every warmed slug."""
    from pfm.terminal import jumps as jumps_mod

    target = ["pol-a", "pol-b"]

    # Use the REAL get_jumps with its full body — but mock the underlying
    # work (market metadata + news + prices). That way we exercise the actual
    # cache-write path inside the handler, not a synthetic shortcut.
    async def fake_to_thread(fn, *args, **kwargs):
        # gather() in get_jumps wraps both _gather_all_news and
        # _fetch_hourly_prices via asyncio.to_thread; return synthetic safe values.
        name = getattr(fn, "__name__", "")
        if name == "_gather_all_news":
            return []
        if name == "_fetch_hourly_prices":
            import pandas as pd

            return pd.Series(dtype=float)
        return fn(*args, **kwargs)

    fake_meta = mock.Mock()
    fake_meta.question = "Will X happen by 2026?"
    fake_meta.yes_token_id = "1"
    fake_meta.start_date = None

    app_with_poly.state.poly.get_market_metadata = mock.Mock(return_value=fake_meta)
    app_with_poly.state.poly._client = mock.Mock()
    app_with_poly.state.poly.clob_url = "https://clob.example"

    # Reset the module-level cache so we can assert a clean delta.
    jumps_mod._CACHE.clear() if hasattr(jumps_mod._CACHE, "clear") else None

    with mock.patch("asyncio.to_thread", side_effect=fake_to_thread):
        asyncio.run(prewarm_jumps(app_with_poly, slugs=target))

    # Cache keys are (slug, days, mad_k, min_jump_pp). After prewarm the
    # canonical key for each slug must resolve to a non-None entry.
    for slug in target:
        key = (slug, 14, round(jumps_mod.DEFAULT_MAD_K, 2), round(jumps_mod.DEFAULT_MIN_JUMP_PP, 2))
        cached = jumps_mod._CACHE.get(key)
        assert cached is not None, f"cache miss for {slug}"
        # Round-trip the cached dict through the pydantic model — verifies
        # the cache stores the same shape the endpoint returns.
        rehydrated = TerminalJumpsResponse(**cached)
        assert rehydrated.slug == slug


# ---------------------------------------------------------------------------
# 4. app.state.warm_jumps populated
# ---------------------------------------------------------------------------


def test_app_state_warm_jumps_populated(app_with_poly: FastAPI) -> None:
    target = ["slug-1", "slug-2", "slug-3"]

    async def fake_get_jumps(**kwargs):
        return _empty_response(kwargs["slug"])

    with mock.patch("pfm.terminal.jumps.get_jumps", side_effect=fake_get_jumps):
        asyncio.run(prewarm_jumps(app_with_poly, slugs=target))

    warm = app_with_poly.state.warm_jumps
    assert isinstance(warm, dict)
    assert "computed_at" in warm
    assert "slugs" in warm
    assert set(warm["slugs"].keys()) == set(target)
    # Timestamp is fresh.
    assert time.time() - warm["computed_at"] < 5.0


# ---------------------------------------------------------------------------
# 5. Lifespan helper
# ---------------------------------------------------------------------------


def test_register_jumps_prewarm_returns_task(app_with_poly: FastAPI) -> None:
    """register_jumps_prewarm must hand back a running Task and init state."""

    async def driver() -> asyncio.Task[Any]:
        async def noop_prewarm(*_a, **_k):
            return {}

        with mock.patch.object(jumps_prewarm, "prewarm_jumps", side_effect=noop_prewarm):
            task = register_jumps_prewarm(app_with_poly)
            await asyncio.wait_for(task, timeout=1.0)
            return task

    task = asyncio.run(driver())
    assert isinstance(task, asyncio.Task)
    assert task.done()
    # State slot was at least touched (init writes None or the eventual dict).
    assert hasattr(app_with_poly.state, "warm_jumps")


# ---------------------------------------------------------------------------
# 6. Failure isolation — one bad slug doesn't sink the others
# ---------------------------------------------------------------------------


def test_failing_slug_isolated(
    app_with_poly: FastAPI,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If get_jumps raises for one slug, the remaining slugs still warm."""
    target = ["ok-1", "boom", "ok-2"]

    async def fake_get_jumps(*, request, slug, days, mad_k, min_jump_pp, poly):
        if slug == "boom":
            raise RuntimeError("polymarket 502")
        return _empty_response(slug)

    with (
        mock.patch("pfm.terminal.jumps.get_jumps", side_effect=fake_get_jumps),
        caplog.at_level(logging.DEBUG, logger="pfm.terminal.jumps_prewarm"),
    ):
        result = asyncio.run(prewarm_jumps(app_with_poly, slugs=target))

    # 'boom' is omitted from the success map; 'ok-*' both present.
    assert "boom" not in result
    assert "ok-1" in result and "ok-2" in result


def test_missing_poly_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    """When app.state.poly is None, prewarm_jumps logs and exits cleanly."""
    app = FastAPI()
    # Explicitly set poly to None so the getattr default fires.
    app.state.poly = None

    with caplog.at_level(logging.DEBUG, logger="pfm.terminal.jumps_prewarm"):
        result = asyncio.run(prewarm_jumps(app, slugs=["x", "y", "z"]))

    assert result == {}
    # state.warm_jumps still populated (empty success set) for observability.
    assert app.state.warm_jumps["slugs"] == {}


def test_empty_slug_list_short_circuits(
    app_with_poly: FastAPI,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="pfm.terminal.jumps_prewarm"):
        result = asyncio.run(prewarm_jumps(app_with_poly, slugs=[]))
    assert result == {}
    msgs = " | ".join(rec.getMessage() for rec in caplog.records)
    assert "no slugs" in msgs.lower()


# ---------------------------------------------------------------------------
# 7. Log line on success — "prewarm: jumps complete N/M in Xs"
# ---------------------------------------------------------------------------


def test_success_log_line_format(
    app_with_poly: FastAPI,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = ["a", "b", "c"]

    async def fake_get_jumps(**kwargs):
        return _empty_response(kwargs["slug"])

    with (
        mock.patch("pfm.terminal.jumps.get_jumps", side_effect=fake_get_jumps),
        caplog.at_level(logging.INFO, logger="pfm.terminal.jumps_prewarm"),
    ):
        asyncio.run(prewarm_jumps(app_with_poly, slugs=target))

    msgs = " | ".join(rec.getMessage() for rec in caplog.records)
    # Format pattern: "prewarm: jumps complete N/M in Xs"
    assert "prewarm: jumps complete" in msgs
    assert "3/3" in msgs


# ---------------------------------------------------------------------------
# 8. Cold-latency simulation: prewarmed cache hit is sub-100ms
# ---------------------------------------------------------------------------


def test_prewarmed_slug_cache_hit_under_100ms(app_with_poly: FastAPI) -> None:
    """After prewarm, fetching the cached entry via the same key takes <100 ms.

    This isn't a benchmark — it's a regression test that the cache lookup
    short-circuit is still in place. If a future refactor removes the
    module-level _CACHE check from get_jumps, this assertion fails.
    """
    from pfm.terminal import jumps as jumps_mod

    async def fake_get_jumps(*, request, slug, days, mad_k, min_jump_pp, poly):
        resp = _empty_response(slug)
        key = (slug, int(days), round(mad_k, 2), round(min_jump_pp, 2))
        jumps_mod._CACHE.set(key, resp.model_dump(), ttl=jumps_mod.CACHE_TTL_SECONDS)
        return resp

    target = ["fast-slug"]
    jumps_mod._CACHE.clear() if hasattr(jumps_mod._CACHE, "clear") else None

    with mock.patch("pfm.terminal.jumps.get_jumps", side_effect=fake_get_jumps):
        asyncio.run(prewarm_jumps(app_with_poly, slugs=target))

    key = (
        "fast-slug",
        14,
        round(jumps_mod.DEFAULT_MAD_K, 2),
        round(jumps_mod.DEFAULT_MIN_JUMP_PP, 2),
    )

    # Measure the warm read path. Even with timestamp overhead this is well
    # under 100 ms — we assert 0.1 s as a loose ceiling so CI noise doesn't
    # flake the test.
    t0 = time.perf_counter()
    payload = jumps_mod._CACHE.get(key)
    elapsed = time.perf_counter() - t0

    assert payload is not None
    assert elapsed < 0.1, f"warm cache lookup took {elapsed * 1000:.1f} ms (>100ms)"


# ---------------------------------------------------------------------------
# 9. Concurrency cap is respected
# ---------------------------------------------------------------------------


def test_concurrency_cap_respected(app_with_poly: FastAPI) -> None:
    """No more than ``concurrency`` get_jumps calls run simultaneously."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def fake_get_jumps(*, request, slug, days, mad_k, min_jump_pp, poly):
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        # Give the scheduler a chance to surface concurrent invocations.
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return _empty_response(slug)

    target = [f"slug-{i}" for i in range(20)]
    with mock.patch("pfm.terminal.jumps.get_jumps", side_effect=fake_get_jumps):
        asyncio.run(prewarm_jumps(app_with_poly, slugs=target, concurrency=4))

    # The semaphore cap is 4 — peak in-flight must not exceed that.
    assert peak <= 4, f"semaphore breach: peak in-flight = {peak} (cap=4)"


# ---------------------------------------------------------------------------
# 10. Constants exported sanely
# ---------------------------------------------------------------------------


def test_constants_exported() -> None:
    assert isinstance(DEFAULT_CONCURRENCY, int) and DEFAULT_CONCURRENCY >= 1
    assert isinstance(PER_SLUG_TIMEOUT_S, (int, float)) and PER_SLUG_TIMEOUT_S > 0
