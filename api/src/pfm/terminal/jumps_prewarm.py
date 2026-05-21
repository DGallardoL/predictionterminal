"""Lifespan warm-cache prewarmer for ``/terminal/jumps/{slug}``.

Mirrors :mod:`pfm.prewarm` (which prewarms vol-distribution + factor-clusters)
but targets the **jumps** endpoint, which is the second-most-expensive
Terminal route on cold cache: each call fans out to four news sources
(GDELT + Reddit + HN + RSS), fetches an hourly Polymarket price history,
and scores every article against the market question. p50 cold latency
is ~3-5 s; p95 worst-case (full RSS fan-out + slow Polymarket) is 8-10 s.

By precomputing the canonical query (``days=14``, ``mad_k=3.0``,
``min_jump_pp=5.0``) for the ~30-50 curated headline slugs at lifespan
start, the warm path hits the in-process :data:`pfm.terminal.jumps._CACHE`
TTLCache in single-digit milliseconds.

Pattern
-------
* Called fire-and-forget from ``pfm.main.lifespan`` via
  ``asyncio.create_task(prewarm_jumps(app))`` right after the
  voldist + factor-clusters prewarm tasks.
* Cache target is the *existing* module-level TTLCache (``_CACHE``) in
  :mod:`pfm.terminal.jumps`, so the endpoint handler already short-circuits
  via its existing cache lookup — no handler change required. We also
  keep a parallel ``app.state.warm_jumps[slug] = elapsed_s`` map for
  observability and for callers that want a fast existence check.
* Each per-slug call is wrapped in a try/except so a single 502 from
  Polymarket can't take the whole prewarm down. Failures are logged at
  WARNING but never raise into the lifespan.
* Concurrency is capped via an ``asyncio.Semaphore`` (default 4) — the
  underlying news fan-out is already parallel within a single jump call,
  so blasting all 50 slugs simultaneously would blow Polymarket's rate
  limit and saturate the upstream connection pool.

Failure mode
------------
The prewarm logs and swallows. The endpoint always falls back to its
existing live-compute branch when the cache misses. If the prewarm
crashes wholesale (e.g. import error), the lifespan still completes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


#: Curated set of slugs prewarmed at lifespan start. These are the
#: top-traffic markets users land on from the homepage cards and the
#: WOW Hero panel. Adjust this list when the demo headliners change —
#: the prewarm is cheap to extend (each slug adds ~3 s amortised, run
#: under the concurrency cap below).
CURATED_TOP_SLUGS: list[str] = [
    # --- politics (highest expected traffic) ---
    "trump-2024-presidential-election",
    "fed-rate-cut-march-2026",
    "us-government-shutdown-2026",
    "us-recession-2026",
    "recession-2026",
    "biden-finishes-term-2024",
    "harris-2024-presidential-election",
    "senate-control-2026",
    "house-control-2026",
    "trump-impeachment-2026",
    # --- macro ---
    "fed-rate-cut-june-2026",
    "fed-rate-cut-2026",
    "us-cpi-above-3-march-2026",
    "unemployment-above-5-2026",
    "us-gdp-recession-q2-2026",
    # --- crypto ---
    "bitcoin-100k-2026",
    "btc-150k-2026",
    "bitcoin-200k-2026",
    "ethereum-5k-2026",
    "btc-all-time-high-2026",
    # --- geopolitics ---
    "russia-ukraine-ceasefire-2026",
    "china-taiwan-2026",
    "israel-hamas-ceasefire-2026",
    "iran-nuclear-deal-2026",
    "north-korea-nuclear-test-2026",
    # --- earnings / single-name equities ---
    "nvda-earnings-q1-2026",
    "tesla-100b-revenue-2026",
    "tsla-1trillion-marketcap-2026",
    "apple-earnings-beat-q1-2026",
    "microsoft-earnings-beat-q1-2026",
    "amzn-earnings-beat-q1-2026",
    "meta-earnings-beat-q1-2026",
    "google-earnings-beat-q1-2026",
    # --- tech / AI ---
    "openai-ipo-2026",
    "agi-2026",
    "ai-stocks-bubble-2026",
    # --- sports / pop-culture (still hot on Polymarket) ---
    "super-bowl-2026",
    "world-cup-2026-winner",
    "nba-finals-2026-winner",
    "oscar-best-picture-2026",
    "taylor-swift-engagement-2026",
]

#: How many slugs to prewarm in parallel. The underlying jumps handler
#: already fans out 4 news sources per slug — a semaphore of 4 gives us
#: ~16 outbound requests in flight, well under the Polymarket 1000/10s cap
#: and below the GDELT rate limit.
DEFAULT_CONCURRENCY: int = 4

#: How long any single prewarm call may run before we abandon it. Some
#: slugs are degenerate (resolved, no history) and would otherwise sit
#: forever on a stuck connection.
PER_SLUG_TIMEOUT_S: float = 30.0


def _now_unix() -> float:
    """Indirection point so tests can monkeypatch the wall-clock."""
    return time.time()


async def _prewarm_one(
    app: FastAPI,
    slug: str,
    *,
    semaphore: asyncio.Semaphore,
) -> tuple[str, float | None]:
    """Compute the canonical ``/terminal/jumps/{slug}`` payload for one slug.

    Returns ``(slug, elapsed_s)`` on success or ``(slug, None)`` on
    failure (Polymarket 404, GDELT 5xx, timeout, anything). The endpoint
    handler is invoked directly so its in-module TTLCache (``_CACHE``)
    gets populated as a side effect — that is the cache the live endpoint
    consults on the next request.

    Args:
        app: FastAPI instance holding ``state.poly`` (PolymarketClient).
        slug: Polymarket condition slug to prewarm.
        semaphore: Concurrency limiter shared across the fan-out.

    Returns:
        Tuple of ``(slug, elapsed_seconds | None)``.
    """
    poly = getattr(app.state, "poly", None)
    if poly is None:
        logger.debug("jumps prewarm: app.state.poly missing for slug=%s", slug)
        return slug, None

    # Import lazily — pfm.terminal.jumps pulls in numpy/pandas/httpx and
    # we want ``import pfm.terminal.jumps_prewarm`` to stay cheap.
    from pfm.terminal.jumps import DEFAULT_DAYS as _DEFAULT_DAYS
    from pfm.terminal.jumps import (
        DEFAULT_MAD_K,
        DEFAULT_MIN_JUMP_PP,
        get_jumps,
    )

    async with semaphore:
        start = _now_unix()
        try:
            # Build a minimal stand-in for the Request dependency. The handler
            # only touches ``request.app.state.poly`` via ``_get_polymarket_client``
            # but we bypass that by passing ``poly`` explicitly. The ``request``
            # parameter itself isn't otherwise dereferenced, so a tiny shim
            # with a ``.app`` attribute is enough.
            class _Req:
                pass

            fake_req = _Req()
            fake_req.app = app  # type: ignore[attr-defined]

            await asyncio.wait_for(
                get_jumps(  # type: ignore[call-arg]
                    request=fake_req,  # type: ignore[arg-type]
                    slug=slug,
                    days=_DEFAULT_DAYS,
                    mad_k=DEFAULT_MAD_K,
                    min_jump_pp=DEFAULT_MIN_JUMP_PP,
                    poly=poly,
                ),
                timeout=PER_SLUG_TIMEOUT_S,
            )
            elapsed = _now_unix() - start
            return slug, elapsed
        except TimeoutError:
            logger.warning(
                "jumps prewarm: slug=%s timed out after %.1fs",
                slug,
                PER_SLUG_TIMEOUT_S,
            )
            return slug, None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Anything else — Polymarket 502, missing market, schema error
            # in degenerate data — is logged and swallowed. The endpoint will
            # surface a real 404/502 to actual callers; the prewarm just
            # doesn't get to populate the cache for this slug.
            logger.debug(
                "jumps prewarm: slug=%s failed: %s",
                slug,
                exc,
            )
            return slug, None


async def prewarm_jumps(
    app: FastAPI,
    *,
    slugs: list[str] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, float]:
    """Background task: precompute ``/terminal/jumps/{slug}`` for top slugs.

    Populates two places:
        * The existing module-level TTLCache in :mod:`pfm.terminal.jumps`
          (``_CACHE``) — this is what the live endpoint already consults.
        * ``app.state.warm_jumps`` — a ``{slug: elapsed_seconds}`` map for
          observability and existence checks.

    Args:
        app: FastAPI app whose ``state.poly`` PolymarketClient drives the
            underlying fetches.
        slugs: Optional explicit slug list. Defaults to
            :data:`CURATED_TOP_SLUGS`.
        concurrency: Max in-flight prewarm calls. Defaults to
            :data:`DEFAULT_CONCURRENCY`.

    Returns:
        A ``{slug: elapsed_s}`` map of successfully-prewarmed slugs.
        Failures are omitted (not represented with ``None`` to keep
        the contract clean). Empty dict on full failure.
    """
    target = list(slugs) if slugs is not None else list(CURATED_TOP_SLUGS)
    if not target:
        logger.info("prewarm: jumps no slugs to warm")
        return {}

    start = _now_unix()
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    results: dict[str, float] = {}
    try:
        gathered = await asyncio.gather(
            *(_prewarm_one(app, s, semaphore=semaphore) for s in target),
            return_exceptions=True,
        )
        for item in gathered:
            if isinstance(item, BaseException):
                # gather already swallowed via return_exceptions=True,
                # but cancellation must still propagate up.
                if isinstance(item, asyncio.CancelledError):
                    raise item
                continue
            slug, elapsed = item
            if elapsed is not None:
                results[slug] = elapsed

        # Surface results on app.state for observability / tests.
        app.state.warm_jumps = {
            "computed_at": _now_unix(),
            "slugs": results,
        }
        logger.info(
            "prewarm: jumps complete %d/%d in %.2fs",
            len(results),
            len(target),
            _now_unix() - start,
        )
        return results
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "prewarm: jumps failed after %.2fs: %s",
            _now_unix() - start,
            exc,
        )
        return results


def register_jumps_prewarm(app: FastAPI) -> asyncio.Task[dict[str, float]]:
    """Schedule :func:`prewarm_jumps` as a fire-and-forget lifespan task.

    Helper for the ``main.py:lifespan`` coordinator: a single call after
    the factors_by_slug index is built does everything (slot init +
    background launch). Returns the created Task so the caller can stash
    it on ``app.state`` for graceful-shutdown cancellation, mirroring how
    :mod:`pfm.prewarm` is wired.

    Usage in ``lifespan(app)``::

        from pfm.terminal.jumps_prewarm import register_jumps_prewarm
        app.state.jumps_prewarm_task = register_jumps_prewarm(app)
    """
    # Initialise the slot so endpoint callers / tests can read it
    # unconditionally without an ``hasattr`` dance.
    if not hasattr(app.state, "warm_jumps") or app.state.warm_jumps is None:
        app.state.warm_jumps = None
    return asyncio.create_task(prewarm_jumps(app))


__all__ = [
    "CURATED_TOP_SLUGS",
    "DEFAULT_CONCURRENCY",
    "PER_SLUG_TIMEOUT_S",
    "prewarm_jumps",
    "register_jumps_prewarm",
]
