"""Lifespan warm-cache prewarmers for /terminal/vol-distribution and
/terminal/factor-clusters.

These are fire-and-forget background coroutines launched from
``pfm.main.lifespan`` immediately after ``app.state.factors_by_slug`` is
built. The pattern mirrors the earnings-whisper dashboard prewarm
(OVERNIGHT-RECAP: cold 13.5s → 0.65s post lifespan prewarm) but keeps the
state on ``app.state.warm_voldist`` / ``app.state.warm_clusters`` so it is
trivially testable via ``app.dependency_overrides`` + a TestClient.

Why hoist into lifespan
-----------------------
* ``/terminal/factor-clusters`` performs hierarchical clustering on up to
  ~600 factor return series — ~300-600 ms p50 once the pickle is hot, but
  the very first request after worker boot pays the pickle-deserialise tax
  plus the AgglomerativeClustering fit (~3 s combined cold).
* ``/terminal/vol-distribution/{slug}`` is per-slug and called eagerly by
  the Terminal panel for the curated headliners. Without prewarm each card
  paid ~250 ms on first touch (peer-σ scan over the same theme).

We precompute the top-15 most-eagerly-requested slugs (the same heuristic
the homepage panel uses: the highest-MOM "alpha-tier" factors that ship in
``factors.yml`` under ``theme=politics`` / ``earnings``) and the default
factor-clusters payload (theme=None, min_corr=0.5).

Freshness
---------
Both warm entries carry a unix-timestamp. The endpoint handlers fall back
to live compute when ``now - computed_at > WARM_TTL_SECONDS`` (default 60 s).
That ceiling is intentionally tight: the underlying pickle is rewritten
every few hours by the strat-7 batch job, so a 1-minute warm horizon is the
sweet spot between cold-tail elimination and serving stale snapshots.

Failure mode
------------
Both prewarm tasks log + swallow exceptions. A failing prewarm never blocks
startup and never raises into the endpoint hot path — the handler falls
back to its existing live-compute branch.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

    from pfm.factors import FactorConfig

logger = logging.getLogger(__name__)


#: How long (seconds) a warm entry is considered fresh enough to short-circuit
#: the endpoint's live-compute branch. Above this the handler recomputes.
WARM_TTL_SECONDS: int = 60

#: How many top slugs to precompute vol-distribution snapshots for.
TOP_N_VOLDIST: int = 15


def _now_unix() -> float:
    """Indirection point so tests can monkeypatch wall-clock."""
    return time.time()


# ---------------------------------------------------------------------------
# Top-slug selection
# ---------------------------------------------------------------------------


def _top_slugs_for_voldist(
    factors: dict[str, FactorConfig],
    n: int = TOP_N_VOLDIST,
) -> list[str]:
    """Pick the N slugs we expect /terminal/vol-distribution to be hit for.

    Heuristic: prefer the headline themes (politics → earnings → macro →
    crypto → sentiment), pull the first slugs found within each — these
    are what the homepage and α-Hub cards bind to. We don't have a live
    request-count source available at boot, so theme-order is the next-best
    proxy.

    Args:
        factors: ``{factor_id: FactorConfig}`` registry.
        n: Cap on returned slug count.

    Returns:
        A deduplicated list of up to ``n`` slugs, ordered by theme priority.
    """
    if n <= 0 or not factors:
        return []
    theme_priority = ("politics", "earnings", "macro", "crypto", "sentiment")
    seen: set[str] = set()
    out: list[str] = []
    for theme in theme_priority:
        for fc in factors.values():
            if getattr(fc, "theme", None) != theme:
                continue
            slug = getattr(fc, "slug", None)
            if not slug or slug in seen:
                continue
            seen.add(slug)
            out.append(slug)
            if len(out) >= n:
                return out
    # If theme-bucketing didn't yield enough (e.g. tiny test catalog), pad
    # from whatever else is available.
    if len(out) < n:
        for fc in factors.values():
            slug = getattr(fc, "slug", None)
            if not slug or slug in seen:
                continue
            seen.add(slug)
            out.append(slug)
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# Sync compute helpers (run inside asyncio.to_thread so they don't block loop)
# ---------------------------------------------------------------------------


def _compute_voldist_snapshots(
    factors: dict[str, FactorConfig],
    slugs: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute one vol-distribution payload per slug.

    Returns a ``{slug: payload}`` map. Slugs that error (missing history,
    insufficient observations, etc.) are silently skipped — the endpoint
    handler will fall through to its existing 404/422 branch.
    """
    # Lazy import: ``pfm.terminal.vol_distribution`` pulls scientific deps
    # (numpy/pandas/sklearn under the hood via delta_logit) which are slow
    # at module import. Defer until the background task actually runs so
    # ``import pfm.prewarm`` stays cheap.
    from pfm.terminal.vol_distribution import (
        DEFAULT_FACTOR_HISTORY_PATH,
        _load_history_pickle,
        compute_vol_distribution,
    )

    history = _load_history_pickle(DEFAULT_FACTOR_HISTORY_PATH)
    if not history:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for slug in slugs:
        try:
            result = compute_vol_distribution(slug, factors=factors, history=history)
        except (KeyError, ValueError) as exc:
            logger.debug("voldist prewarm skip slug=%s: %s", slug, exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("voldist prewarm slug=%s raised: %s", slug, exc)
            continue
        out[slug] = {
            "slug": result.slug,
            "current_vol": result.current_vol,
            "theme": result.theme,
            "n_peers": result.n_peers,
            "percentile_in_theme": result.percentile_in_theme,
            "vol_distribution": result.vol_distribution,
            "current_z_score": result.current_z_score,
            "peers_higher_vol": result.peers_higher_vol,
            "peers_lower_vol": result.peers_lower_vol,
        }
    return out


def _compute_factor_clusters_default() -> dict[str, Any] | None:
    """Compute the default ``/terminal/factor-clusters`` payload (theme=None).

    Returns the response as a plain dict so it can be JSON-encoded straight
    back from the cached path in the endpoint handler. ``None`` on failure.
    """
    try:
        from pfm.terminal.factor_clusters import factor_clusters as _compute
    except ImportError as exc:  # pragma: no cover - upstream missing dep
        logger.warning("factor-clusters prewarm import failed: %s", exc)
        return None

    try:
        # ``factor_clusters`` is the FastAPI handler but it has no Request
        # arg — it reads the on-disk pickle directly. Safe to call inline.
        resp = _compute(theme=None, min_corr=0.5)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("factor-clusters prewarm raised: %s", exc)
        return None
    # ``FactorClustersResponse`` is a pydantic BaseModel — dump it.
    try:
        return resp.model_dump()
    except AttributeError:
        # Already a dict (e.g. in a test stub).
        return dict(resp) if not isinstance(resp, dict) else resp


# ---------------------------------------------------------------------------
# Async fire-and-forget wrappers (call from lifespan)
# ---------------------------------------------------------------------------


async def prewarm_voldist(app: FastAPI) -> None:
    """Background task: precompute top-N vol-distribution snapshots.

    Stores the result on ``app.state.warm_voldist`` as::

        {
            "computed_at": <unix ts>,
            "snapshots": {slug: payload, ...},
        }

    The endpoint handler reads this and short-circuits if fresh. Exceptions
    are logged and swallowed.
    """
    import asyncio

    start = _now_unix()
    try:
        factors = getattr(app.state, "factors", {}) or {}
        slugs = _top_slugs_for_voldist(factors)
        snapshots = await asyncio.to_thread(
            _compute_voldist_snapshots,
            factors,
            slugs,
        )
        app.state.warm_voldist = {
            "computed_at": _now_unix(),
            "snapshots": snapshots,
        }
        logger.info(
            "prewarm: voldist ok in %.2fs (%d/%d snapshots)",
            _now_unix() - start,
            len(snapshots),
            len(slugs),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("prewarm: voldist failed after %.2fs: %s", _now_unix() - start, exc)


async def prewarm_factor_clusters(app: FastAPI) -> None:
    """Background task: precompute the default factor-clusters payload.

    Stores the result on ``app.state.warm_clusters`` as::

        {"computed_at": <unix ts>, "payload": <dict>}

    The endpoint handler reads this and short-circuits if fresh. Exceptions
    are logged and swallowed.
    """
    import asyncio

    start = _now_unix()
    try:
        payload = await asyncio.to_thread(_compute_factor_clusters_default)
        if payload is None:
            logger.info("prewarm: factor-clusters skipped (no history)")
            return
        # If the pickle didn't exist when prewarm ran (it races the
        # ``_factor_prewarm`` task in ``main.py`` that *writes* the pickle),
        # ``payload`` will be a ``degraded_mode=True`` stub. Storing that as
        # ``warm_clusters`` would make every subsequent /terminal/factor-clusters
        # request short-circuit on the bad payload until WARM_TTL_SECONDS
        # elapses. Refuse to cache a degraded payload so the endpoint falls
        # through to live compute (which by then will see the freshly-written
        # pickle and serve the real clusters).
        if isinstance(payload, dict) and payload.get("degraded_mode"):
            logger.info(
                "prewarm: factor-clusters degraded (pickle not yet written); "
                "leaving warm_clusters unset so handler retries live."
            )
            return
        app.state.warm_clusters = {
            "computed_at": _now_unix(),
            "payload": payload,
        }
        logger.info(
            "prewarm: factor-clusters ok in %.2fs",
            _now_unix() - start,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "prewarm: factor-clusters failed after %.2fs: %s",
            _now_unix() - start,
            exc,
        )


# ---------------------------------------------------------------------------
# Freshness helpers for endpoint handlers
# ---------------------------------------------------------------------------


def warm_voldist_lookup(
    app: FastAPI,
    slug: str,
    *,
    ttl_seconds: int = WARM_TTL_SECONDS,
) -> dict[str, Any] | None:
    """Return a prewarmed vol-distribution payload for ``slug`` if fresh.

    Returns ``None`` when there is no warm entry, the slug isn't in the
    prewarmed set, or the entry is older than ``ttl_seconds``. The endpoint
    handler treats ``None`` as "compute live".
    """
    state = getattr(app, "state", None)
    if state is None:
        return None
    warm = getattr(state, "warm_voldist", None)
    if not isinstance(warm, dict):
        return None
    computed_at = warm.get("computed_at")
    if computed_at is None or _now_unix() - float(computed_at) > ttl_seconds:
        return None
    snapshots = warm.get("snapshots") or {}
    payload = snapshots.get(slug)
    if not isinstance(payload, dict):
        return None
    return payload


def warm_clusters_lookup(
    app: FastAPI,
    *,
    theme: str | None,
    min_corr: float,
    ttl_seconds: int = WARM_TTL_SECONDS,
) -> dict[str, Any] | None:
    """Return the prewarmed factor-clusters payload if fresh AND matching.

    The prewarm only covers the default query (``theme=None``,
    ``min_corr=0.5``); any other parameterisation falls through to compute.
    """
    if theme is not None or abs(min_corr - 0.5) > 1e-9:
        return None
    state = getattr(app, "state", None)
    if state is None:
        return None
    warm = getattr(state, "warm_clusters", None)
    if not isinstance(warm, dict):
        return None
    computed_at = warm.get("computed_at")
    if computed_at is None or _now_unix() - float(computed_at) > ttl_seconds:
        return None
    payload = warm.get("payload")
    if not isinstance(payload, dict):
        return None
    return payload


__all__ = [
    "TOP_N_VOLDIST",
    "WARM_TTL_SECONDS",
    "_compute_factor_clusters_default",
    "_compute_voldist_snapshots",
    "_top_slugs_for_voldist",
    "prewarm_factor_clusters",
    "prewarm_voldist",
    "warm_clusters_lookup",
    "warm_voldist_lookup",
]
