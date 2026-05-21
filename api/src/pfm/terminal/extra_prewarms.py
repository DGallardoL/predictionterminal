"""Lifespan warm-cache prewarmers for four high-latency Terminal endpoints.

Mirrors :mod:`pfm.prewarm` (vol-distribution + factor-clusters) and
:mod:`pfm.terminal.jumps_prewarm` (per-slug jumps) but targets the next
tier of cold-start offenders measured 2026-05-19 against a freshly-booted
worker:

    GET /terminal/jumps/cluster              ~11.3 s cold
    GET /terminal/sentiment-leaderboard      ~ 8.1 s cold
    GET /terminal/sentiment-trend/spike-alerts ~ 5.1 s cold
    GET /terminal/calendar-curated/clusters  ~ 1.9 s cold

All four populate their *existing* module-level response caches as a side
effect of calling the handler with default query params, so the live
endpoint short-circuits on the next request without any handler change.
Each prewarm is fire-and-forget, wrapped in a try/except, and bounded by
a per-task timeout — none of them can delay startup beyond the liveness
probe, and a single upstream failure can't cascade.

The :func:`register_extra_prewarms` helper schedules all four as one
unit so ``main.py:lifespan`` adds a single block. Pattern mirrors
:func:`pfm.terminal.jumps_prewarm.register_jumps_prewarm`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


#: Hard ceiling on any single prewarm task. Spike-alerts fans out to
#: GDELT (2 calls × 8 candidates) and can blow past 30 s when GDELT is
#: degraded; cluster fan-out chains 20 jump computations. Cap each so a
#: stuck upstream can't tie up the lifespan watcher indefinitely.
PER_TASK_TIMEOUT_S: float = 60.0


def _now_unix() -> float:
    """Indirection point so tests can monkeypatch wall-clock."""
    return time.time()


class _ReqShim:
    """Minimal stand-in for :class:`fastapi.Request`.

    The four handlers we prewarm only ever dereference ``request.app.state.*``
    (verified 2026-05-19: ``async_http`` and ``poly`` are the only
    attributes touched). A shim with a single ``.app`` attribute is
    enough — we avoid spinning up an ASGI transport just to populate a
    cache.
    """

    __slots__ = ("app",)

    def __init__(self, app: FastAPI) -> None:
        self.app = app


async def _run_one(
    name: str,
    coro_factory: Any,
) -> None:
    """Run a single prewarm coroutine factory under timeout + logging.

    ``coro_factory`` is a zero-arg callable returning a coroutine (NOT
    the coroutine itself) so the timeout/log start tick happens at the
    same moment we kick off the work — important for accurate timing
    when many prewarms launch concurrently.
    """
    start = _now_unix()
    try:
        await asyncio.wait_for(coro_factory(), timeout=PER_TASK_TIMEOUT_S)
        logger.info("prewarm: %s ok in %.2fs", name, _now_unix() - start)
    except TimeoutError:
        logger.warning(
            "prewarm: %s timed out after %.1fs",
            name,
            PER_TASK_TIMEOUT_S,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "prewarm: %s failed after %.2fs: %s",
            name,
            _now_unix() - start,
            exc,
        )


# ---------------------------------------------------------------------------
# Per-endpoint prewarm coroutines
# ---------------------------------------------------------------------------


async def prewarm_sentiment_leaderboard(app: FastAPI) -> None:
    """Populate :data:`pfm.terminal.sentiment_leaderboard._CACHE`.

    Hits the canonical query (``days=DEFAULT_DAYS``,
    ``min_jumps=DEFAULT_MIN_JUMPS``). Subsequent live requests with the
    default params short-circuit on the cache in <2 ms.
    """
    from pfm.terminal.sentiment_leaderboard import (
        DEFAULT_DAYS,
        DEFAULT_MIN_JUMPS,
        get_sentiment_leaderboard,
    )

    req = _ReqShim(app)
    await get_sentiment_leaderboard(  # type: ignore[arg-type]
        request=req,  # type: ignore[arg-type]
        days=DEFAULT_DAYS,
        min_jumps=DEFAULT_MIN_JUMPS,
    )


async def prewarm_spike_alerts(app: FastAPI) -> None:
    """Populate :data:`pfm.terminal.sentiment_trend._CACHE` for spike-alerts.

    The handler is sync — wrap in :func:`asyncio.to_thread` so we don't
    block the event loop on its GDELT fan-out. Uses the canonical query
    (``days=7``, ``min_n_articles=3``).
    """
    poly = getattr(app.state, "poly", None)
    if poly is None:
        logger.debug("prewarm: spike-alerts skipped — app.state.poly missing")
        return

    from pfm.terminal.sentiment_trend import get_spike_alerts

    req = _ReqShim(app)

    def _call() -> Any:
        return get_spike_alerts(  # type: ignore[call-arg]
            request=req,  # type: ignore[arg-type]
            days=7,
            min_n_articles=3,
            poly=poly,
        )

    await asyncio.to_thread(_call)


async def prewarm_jumps_cluster(app: FastAPI) -> None:
    """Populate :data:`pfm.terminal.jumps_cluster._CACHE`.

    Calls with the empty-slug default path which fans out to top-N slugs
    by 24h volume — exactly what the homepage cards land on. Subsequent
    live requests with the same params return from cache in <10 ms.
    """
    from pfm.terminal.jumps import DEFAULT_MAD_K, DEFAULT_MIN_JUMP_PP
    from pfm.terminal.jumps_cluster import (
        DEFAULT_DAYS,
        DEFAULT_KW_MIN_JACCARD,
        DEFAULT_TIME_TOL_MINUTES,
        get_jumps_clusters,
    )

    req = _ReqShim(app)
    await get_jumps_clusters(  # type: ignore[arg-type]
        request=req,  # type: ignore[arg-type]
        slugs="",
        days=DEFAULT_DAYS,
        time_tol_minutes=DEFAULT_TIME_TOL_MINUTES,
        kw_min_jaccard=DEFAULT_KW_MIN_JACCARD,
        mad_k=DEFAULT_MAD_K,
        min_jump_pp=DEFAULT_MIN_JUMP_PP,
    )


async def prewarm_calendar_curated_clusters(app: FastAPI) -> None:
    """Populate :data:`pfm.terminal.calendar_curated._CLUSTERS_CACHE`.

    The handler is sync and takes ``settings`` + ``poly`` via Depends.
    We resolve both from the running app and call directly through
    :func:`asyncio.to_thread`.
    """
    poly = getattr(app.state, "poly", None)
    if poly is None:
        logger.debug("prewarm: calendar-curated-clusters skipped — app.state.poly missing")
        return

    from pfm.config import get_settings
    from pfm.terminal.calendar_curated import list_clusters

    settings = get_settings()

    def _call() -> Any:
        # list_clusters is kw-only (``*,`` in signature).
        return list_clusters(settings=settings, poly=poly)

    await asyncio.to_thread(_call)


# ---------------------------------------------------------------------------
# Lifespan entry-point
# ---------------------------------------------------------------------------


async def _run_all(app: FastAPI) -> None:
    """Schedule the four prewarms concurrently.

    Each runs under :data:`PER_TASK_TIMEOUT_S` and never raises out of
    the coroutine — failures are logged and swallowed individually so
    one slow upstream doesn't poison the others.
    """
    await asyncio.gather(
        _run_one("sentiment-leaderboard", lambda: prewarm_sentiment_leaderboard(app)),
        _run_one("spike-alerts", lambda: prewarm_spike_alerts(app)),
        _run_one("jumps-cluster", lambda: prewarm_jumps_cluster(app)),
        _run_one(
            "calendar-curated-clusters",
            lambda: prewarm_calendar_curated_clusters(app),
        ),
        return_exceptions=False,
    )


def register_extra_prewarms(app: FastAPI) -> asyncio.Task[None]:
    """Schedule all four prewarms as one fire-and-forget lifespan task.

    Usage in ``pfm.main.lifespan``::

        from pfm.terminal.extra_prewarms import register_extra_prewarms
        app.state.extra_prewarms_task = register_extra_prewarms(app)

    Returns the created Task so the caller can stash it on ``app.state``
    for graceful-shutdown cancellation (mirrors
    :mod:`pfm.terminal.jumps_prewarm`).
    """
    return asyncio.create_task(_run_all(app))


__all__ = [
    "PER_TASK_TIMEOUT_S",
    "prewarm_calendar_curated_clusters",
    "prewarm_jumps_cluster",
    "prewarm_sentiment_leaderboard",
    "prewarm_spike_alerts",
    "register_extra_prewarms",
]
