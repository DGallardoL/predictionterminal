"""Background workers that keep the crypto5min surface hot.

Two coroutines live here:

* ``run_sampler``     — polls Binance REST every ``poll_seconds`` to keep the
  spot buffer warm even when no user is hitting the API. Without this the
  ``/compare`` endpoint would need to fetch a fresh mid on every call and
  the σ-jackknife would have nothing to anchor against.

* ``run_compare_prewarmer`` — rebuilds the ``/compare`` payload (default
  ``BTC,ETH × 5,15`` combos) every ``compare_seconds`` so user-facing
  requests always land on a hot in-process cache. Brings p95 response
  time from ~1.5s (cold) to ~5ms (cache hit). Idle CPU cost is ~1
  parallel-fanout of 24 gamma probes + 2 kline + 2 ticker calls every 3s.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import httpx

from pfm.crypto5min.market_fetcher import SUPPORTED_ASSETS, fetch_binance_mid
from pfm.crypto5min.state import CryptoFiveMinState, get_state

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS: float = 1.0
DEFAULT_SYMBOLS: tuple[str, ...] = tuple(m["binance_symbol"] for m in SUPPORTED_ASSETS.values())


async def _poll_one(client: httpx.AsyncClient, symbol: str, state: CryptoFiveMinState) -> None:
    mid = await fetch_binance_mid(client, symbol)
    if mid is not None:
        state.record_spot(symbol, time.time(), mid)


async def run_sampler(
    *,
    client: httpx.AsyncClient | None = None,
    symbols: list[str] | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Long-running coroutine. Cancellable via ``stop_event`` or task.cancel()."""
    owned = client is None
    if owned:
        client = httpx.AsyncClient(timeout=4.0)
    state = get_state()
    symbols = list(symbols or DEFAULT_SYMBOLS)
    stop = stop_event or asyncio.Event()
    logger.info("crypto5min sampler started · %d symbols · %.1fs", len(symbols), poll_seconds)
    try:
        while not stop.is_set():
            t0 = time.time()
            try:
                await asyncio.gather(
                    *(_poll_one(client, s, state) for s in symbols),
                    return_exceptions=True,
                )
            except Exception as exc:
                logger.warning("crypto5min poll iteration error: %s", exc)
            elapsed = time.time() - t0
            sleep_for = max(0.05, poll_seconds - elapsed)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=sleep_for)
    except asyncio.CancelledError:
        raise
    finally:
        if owned:
            with contextlib.suppress(Exception):
                await client.aclose()


DEFAULT_COMPARE_SECONDS: float = 1.0
URGENT_COMPARE_SECONDS: float = 0.4
"""Sleep cadence when any market is within ``URGENT_AT_SECONDS`` of expiry.
Faster refresh near boundary so the cached payload doesn't go stale and
collapse the model into the 0/1 step-function regime.

Reduced from 1.5s/0.5s to 1.0s/0.4s — at this cadence every Polymarket CLOB
midpoint refresh + GBM-prob rebuild lands inside one frontend poll window
(also 1s), so the UI sees a fresh number on every tick. Each iteration costs
roughly: 2 Binance mid calls + 2-4 Polymarket CLOB midpoint calls + 2-4 gamma
probes (cached) ≈ 6-10 HTTP requests/s, well within both rate limits."""

URGENT_AT_SECONDS: float = 20.0


async def run_compare_prewarmer(
    *,
    client: httpx.AsyncClient | None = None,
    assets: list[str] | None = None,
    windows: list[int] | None = None,
    edge_threshold: float | None = None,
    compare_seconds: float = DEFAULT_COMPARE_SECONDS,
    stop_event: asyncio.Event | None = None,
    redis_client: Any | None = None,
) -> None:
    """Continuously rebuild the ``/compare`` payload and stash it in the cache.

    With this running, every user-facing ``GET /strategies/crypto/5min/compare``
    call hits a hot cache (response time ~5ms) regardless of when the user
    first lands on the page. Default args mirror the UI default — refreshes
    ``BTC,ETH × 5,15`` at default edge threshold.
    """
    # Lazy import — avoid an import cycle (router → background) and let the
    # prewarmer skip cleanly if the router didn't load (e.g. partial install).
    from pfm.crypto5min.comparator import DEFAULT_EDGE_THRESHOLD
    from pfm.crypto5min.market_fetcher import SUPPORTED_ASSETS as _SUPPORTED
    from pfm.crypto5min.router import (
        _build_compare_cache_key,
        _compare_cache,
        build_compare_payload,
    )

    owned = client is None
    if owned:
        client = httpx.AsyncClient(timeout=5.0)
    state = get_state()
    asset_list = [a.upper() for a in (assets or list(_SUPPORTED.keys()))]
    asset_list = [a for a in asset_list if a in _SUPPORTED]
    win_list = list(windows or [5])
    edge = float(edge_threshold) if edge_threshold is not None else DEFAULT_EDGE_THRESHOLD
    cache_key = _build_compare_cache_key(asset_list, win_list, edge)
    stop = stop_event or asyncio.Event()
    logger.info(
        "crypto5min compare prewarmer started · %s × %s · every %.1fs",
        ",".join(asset_list),
        ",".join(map(str, win_list)),
        compare_seconds,
    )
    try:
        while not stop.is_set():
            t0 = time.time()
            urgent = False
            try:
                payload = await build_compare_payload(
                    client,
                    state,
                    assets=asset_list,
                    windows=win_list,
                    edge_threshold=edge,
                    redis_client=redis_client,
                )
                _compare_cache[cache_key] = (time.time(), payload)
                # If any market is within URGENT_AT_SECONDS of expiry, sleep
                # for URGENT_COMPARE_SECONDS so users near the boundary see
                # fresh data + the model doesn't collapse into the step regime.
                for row in payload.get("rows", []):
                    secs = row.get("seconds_remaining")
                    if secs is not None and secs <= URGENT_AT_SECONDS:
                        urgent = True
                        break
            except Exception as exc:
                logger.warning("crypto5min compare prewarmer iter failed: %s", exc)
            elapsed = time.time() - t0
            cadence = URGENT_COMPARE_SECONDS if urgent else compare_seconds
            sleep_for = max(0.1, cadence - elapsed)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=sleep_for)
    except asyncio.CancelledError:
        raise
    finally:
        if owned:
            with contextlib.suppress(Exception):
                await client.aclose()


def start_in_lifespan(
    app,
    *,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    compare_seconds: float = DEFAULT_COMPARE_SECONDS,
) -> tuple[asyncio.Task, asyncio.Task]:
    """Spawn both background tasks. Returns (sampler_task, prewarmer_task)."""
    stop_sampler = asyncio.Event()
    stop_prewarmer = asyncio.Event()
    shared = getattr(app.state, "async_http", None)
    shared = shared if isinstance(shared, httpx.AsyncClient) else None
    # Optional follower path: when the prewarmer runs on a non-leader
    # worker, ``redis_client`` lets ``fetch_clob_midpoint`` consult the
    # leader's published midpoints instead of paying REST every tick.
    cache = getattr(app.state, "cache", None)
    redis_client = getattr(cache, "_client", None) if cache is not None else None

    sampler_task = asyncio.create_task(
        run_sampler(
            client=shared,
            poll_seconds=poll_seconds,
            stop_event=stop_sampler,
        ),
        name="pfm-crypto5min-sampler",
    )
    prewarmer_task = asyncio.create_task(
        run_compare_prewarmer(
            client=shared,
            compare_seconds=compare_seconds,
            stop_event=stop_prewarmer,
            redis_client=redis_client,
        ),
        name="pfm-crypto5min-compare-prewarmer",
    )
    app.state.crypto5min_sampler_task = sampler_task
    app.state.crypto5min_sampler_stop = stop_sampler
    app.state.crypto5min_prewarmer_task = prewarmer_task
    app.state.crypto5min_prewarmer_stop = stop_prewarmer
    return sampler_task, prewarmer_task


async def stop_in_lifespan(app) -> None:
    """Tear down both background tasks cleanly."""
    for stop_attr, task_attr in [
        ("crypto5min_sampler_stop", "crypto5min_sampler_task"),
        ("crypto5min_prewarmer_stop", "crypto5min_prewarmer_task"),
    ]:
        stop = getattr(app.state, stop_attr, None)
        task = getattr(app.state, task_attr, None)
        if stop is not None:
            stop.set()
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
