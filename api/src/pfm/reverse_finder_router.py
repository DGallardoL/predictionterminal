"""FastAPI router exposing :mod:`pfm.reverse_finder` as two endpoints.

- ``POST /reverse-finder``               — top-k factors that explain a ticker
- ``POST /alpha/prediction-driven``      — basket of equities loading on a factor

Per project convention :mod:`pfm.main` is left untouched; wire this router
explicitly on app startup::

    from pfm.reverse_finder_router import router as reverse_finder_router
    app.include_router(reverse_finder_router)

Production fetchers (yfinance + Polymarket/Kalshi) are reused via the helpers
in ``pfm.main``. Results are cached for 600 seconds via ``get_cache()``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from datetime import date
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pfm.cache import CacheBackend, NullCache
from pfm.cache_utils import get_cache as _get_terminal_cache
from pfm.factors import FactorConfig
from pfm.model import DEFAULT_EPSILON
from pfm.reverse_finder import (
    DEFAULT_TICKERS,
    iter_reverse_find_factors,
    prediction_driven_alpha,
    reverse_find_factors,
)

logger = logging.getLogger(__name__)

# Cap the candidate pool so /reverse-finder doesn't iterate over all ~1360
# factors when the user passes none — keep it under a few seconds end-to-end.
# ``DEFAULT_CANDIDATE_LIMIT`` is the legacy cap for the volume-ranked path.
DEFAULT_CANDIDATE_LIMIT: int = 100
# ``CURATED_POOL_LIMIT`` is the (larger but still tractable) cap for the
# curated discovery pool used as the default for /reverse-finder.
CURATED_POOL_LIMIT: int = 200
# Up to this many factors per ``theme`` in the curated pool so a single
# busy theme (e.g. AI race) doesn't drown the rest.
CURATED_PER_THEME_CAP: int = 30
TTL_SECONDS: int = 600

# SSE keep-alive cadence for ``/reverse-finder/stream``. Each forward-
# selection step in ``iter_reverse_find_factors`` runs an OLS over the
# candidate pool and can take several seconds when ``pool=all`` (~1.3k
# factors). Without comment-line pings, idle proxies (nginx, cloud LBs)
# close the connection at 60 s and the browser sees a torn stream.
HEARTBEAT_INTERVAL_S: float = float(os.environ.get("PFM_REVERSE_FINDER_HEARTBEAT_S", "15.0"))
# Self-terminate after this many seconds so the server doesn't run an
# unbounded selection on a misconfigured request. 10 minutes is plenty for
# even ``pool=all`` at k=10.
MAX_STREAM_SECONDS: float = float(os.environ.get("PFM_REVERSE_FINDER_MAX_STREAM_S", "600.0"))

# Pool selection modes:
# - ``curated``   (default) — source-prioritised, theme-balanced ≤200 ids.
# - ``top_volume``           — top-N by 24h volume from homepage cache.
# - ``all``                  — every factor in factors.yml (~1360, slow).
# - ``theme``                — reserved for future per-theme cohorts.
PoolMode = Literal["curated", "top_volume", "all", "theme"]


def _top_volume_candidate_ids(
    factors: dict[str, FactorConfig], limit: int = DEFAULT_CANDIDATE_LIMIT
) -> tuple[list[str], str]:
    """Return ``(candidate_ids, source_label)`` for ``pool="top_volume"``.

    Tries (1) the terminal homepage cache (``most_active`` rows have
    24h volume), then (2) the search-index cache (per-factor ``v`` field),
    finally falls back to the alphabetic first-N. The second return value
    documents which path won so the response can surface it.
    """
    slug_to_id: dict[str, str] = {fc.slug: fid for fid, fc in factors.items() if fc.slug}

    # Path 1: homepage payload (newest, freshest volume snapshot).
    home_cache = _get_terminal_cache("terminal_homepage")
    for entry in home_cache._store.values():
        try:
            payload = entry[1]
        except (IndexError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        rows = payload.get("most_active") or []
        ranked: list[tuple[float, str]] = []
        for row in rows:
            slug = (row or {}).get("slug")
            vol = (row or {}).get("volume_24h")
            if not slug or vol is None:
                continue
            fid = slug_to_id.get(str(slug))
            if fid is None:
                continue
            try:
                ranked.append((float(vol), fid))
            except (TypeError, ValueError):
                continue
        if ranked:
            ranked.sort(key=lambda r: -r[0])
            picked = [fid for _, fid in ranked[:limit]]
            if len(picked) >= min(limit, 5):
                return picked, "terminal_homepage"

    # Path 2: search-index cache.
    idx_cache = _get_terminal_cache("terminal_search_index")
    for entry in idx_cache._store.values():
        try:
            payload = entry[1]
        except (IndexError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        rows = payload.get("factors") or []
        ranked2: list[tuple[float, str]] = []
        for row in rows:
            fid = (row or {}).get("i")
            v = (row or {}).get("v")
            if not fid or v is None or fid not in factors:
                continue
            try:
                ranked2.append((float(v), str(fid)))
            except (TypeError, ValueError):
                continue
        if ranked2:
            ranked2.sort(key=lambda r: -r[0])
            picked = [fid for _, fid in ranked2[:limit]]
            if len(picked) >= min(limit, 5):
                return picked, "terminal_search_index"

    # Fallback: alphabetic first-N.
    return sorted(factors.keys())[:limit], "alphabetic_fallback"


# Source priority for the curated pool: polymarket has the deepest daily
# history so it's the only tier that's reliably regressable. Kalshi is
# second; the level / macro sources (bls, fred) and predictit / manifold
# are excluded so we don't waste candidate slots on series the user can't
# meaningfully load on with a single equity factor model.
_CURATED_SOURCE_PRIORITY: tuple[str, ...] = ("polymarket", "kalshi")


def _curated_candidate_ids(
    factors: dict[str, FactorConfig],
    limit: int = CURATED_POOL_LIMIT,
    per_theme_cap: int = CURATED_PER_THEME_CAP,
) -> list[str]:
    """Return up to ``limit`` factor ids — source-prioritised, theme-balanced.

    Algorithm:

    1.  If any factor carries ``theme == "_curated"`` (sentinel for an
        explicit hand-picked subset), use those first — they're the
        editorial gold standard.
    2.  Iterate sources in priority order (``polymarket`` → ``kalshi``).
        Within each source bucket, group by ``theme`` and take up to
        ``per_theme_cap`` ids, ordered alphabetically by ``id`` for
        determinism. This stops a single high-volume theme (e.g.
        ``ai``) from monopolising the pool.
    3.  Stop once the cumulative count reaches ``limit``.

    Other sources (``bls``, ``fred``, ``manifold``, ``predictit``,
    ``chain``) are excluded. We rely on ``FactorConfig.theme`` (already
    present in the dataclass) and ``FactorConfig.source``; there is no
    ``resolved`` field on ``FactorConfig`` today so the resolved-filter
    is a no-op.
    """
    if limit < 1:
        return []

    picked: list[str] = []
    seen: set[str] = set()

    # Step 1: explicit ``_curated`` opt-in subset, if any.
    curated_explicit = sorted(fid for fid, fc in factors.items() if fc.theme == "_curated")
    for fid in curated_explicit:
        if fid in seen:
            continue
        picked.append(fid)
        seen.add(fid)
        if len(picked) >= limit:
            return picked[:limit]

    # Step 2: priority-ordered, theme-balanced walk.
    for src in _CURATED_SOURCE_PRIORITY:
        if len(picked) >= limit:
            break
        # group ids by theme for this source.
        by_theme: dict[str, list[str]] = {}
        for fid, fc in factors.items():
            if fid in seen or fc.source != src:
                continue
            by_theme.setdefault(fc.theme, []).append(fid)
        for theme_ids in by_theme.values():
            theme_ids.sort()

        # Round-robin across themes so no single theme dominates the
        # head of the list. Each round takes one id from each non-empty
        # theme until either (a) the per-theme cap is hit for that theme
        # or (b) ``limit`` is reached.
        offsets: dict[str, int] = dict.fromkeys(by_theme, 0)
        taken_per_theme: dict[str, int] = dict.fromkeys(by_theme, 0)
        while len(picked) < limit:
            advanced = False
            for theme, ids in by_theme.items():
                if taken_per_theme[theme] >= per_theme_cap:
                    continue
                idx = offsets[theme]
                if idx >= len(ids):
                    continue
                fid = ids[idx]
                offsets[theme] = idx + 1
                if fid in seen:
                    continue
                picked.append(fid)
                seen.add(fid)
                taken_per_theme[theme] += 1
                advanced = True
                if len(picked) >= limit:
                    break
            if not advanced:
                break

    return picked[:limit]


# --- Pydantic request/response models --------------------------------------


class ReverseFinderRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, examples=["NVDA"])
    start: date
    end: date
    candidate_factor_ids: list[str] | None = Field(
        default=None,
        description=(
            "Pool of factor ids to consider. If null, the candidate pool is "
            f"derived from the ``pool`` query param (default: top-{DEFAULT_CANDIDATE_LIMIT} "
            "by 24h volume; opt-in ``pool=all`` for the full ~1360 sweep)."
        ),
    )
    k: int = Field(default=5, ge=1, le=10)
    return_type: str = Field(default="log", pattern=r"^(log|simple)$")
    epsilon: float = Field(default=DEFAULT_EPSILON, gt=0.0, lt=0.5)
    min_obs: int = Field(default=30, ge=10)


class ReverseFinderTopFactor(BaseModel):
    factor_id: str
    factor_name: str | None = None
    delta_r_squared: float
    beta: float
    t_stat: float
    vif: float


class ReverseFinderResponse(BaseModel):
    ticker: str
    top_factors: list[ReverseFinderTopFactor]
    total_r_squared: float
    n_obs: int
    rejected: list[str] = Field(default_factory=list)
    note: str | None = None
    pool_used: str = Field(
        default="explicit",
        description=(
            "Which candidate pool produced the iterated factor list: "
            "``top_volume`` / ``all`` / ``theme`` / ``explicit``. The "
            "``top_volume`` mode also surfaces the source via the trailing "
            "label (e.g. ``top_volume:terminal_homepage``)."
        ),
    )
    n_candidates_evaluated: int = Field(
        default=0,
        description="Number of factor ids actually iterated by the stepwise scan.",
    )


class PredictionAlphaRequest(BaseModel):
    factor_id: str = Field(min_length=1, max_length=120)
    candidate_tickers: list[str] | None = Field(
        default=None,
        description="Defaults to DEFAULT_TICKERS (broad-market + sector ETFs).",
    )
    window_days: int = Field(default=252, ge=60, le=1500)
    top_n: int = Field(default=12, ge=1, le=50)
    delta_logit_assumed: float | None = Field(
        default=None,
        description=(
            "Optional hypothetical Δlogit. If set, response includes "
            "expected_return_pct = β · Δlogit · 100 for each ticker."
        ),
    )
    return_type: str = Field(default="log", pattern=r"^(log|simple)$")
    epsilon: float = Field(default=DEFAULT_EPSILON, gt=0.0, lt=0.5)


class PredictionAlphaTickerRow(BaseModel):
    ticker: str
    beta: float
    r_squared: float
    t_stat: float
    n_obs: int
    expected_return_pct: float | None = None


class PredictionAlphaResponse(BaseModel):
    factor_id: str
    factor_name: str
    tickers: list[PredictionAlphaTickerRow]
    ranked_by: str
    delta_logit_assumed: float | None = None
    window_days: int | None = None
    note: str | None = None


# --- internal helpers -------------------------------------------------------


def _cache_key(*parts: object) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return "pfm:rff:" + hashlib.sha256(blob).hexdigest()


def _build_returns_fetcher() -> Any:
    """Adapter: wrap ``main._cached_log_returns`` to match :class:`ReturnsFetcher`."""
    from pfm import main as main_mod

    def _fetch(
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        return_type: str = "log",
    ) -> pd.Series:
        return main_mod._cached_log_returns(
            ticker,
            start,
            end,
            return_type,
            main_mod.get_cache(),
            main_mod.get_settings(),
        )

    return _fetch


def _build_factor_fetcher(factors: dict[str, FactorConfig]) -> Any:
    """Adapter: wrap ``main._cached_factor_history`` indexing by factor id."""
    from pfm import main as main_mod

    def _fetch(
        factor_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        fc = factors.get(factor_id)
        if fc is None:
            raise KeyError(f"unknown factor id: {factor_id!r}")
        return main_mod._cached_factor_history(
            fc,
            start,
            end,
            main_mod.app.state.poly,
            main_mod.get_cache(),
            main_mod.get_settings(),
        )

    return _fetch


# --- router -----------------------------------------------------------------

router = APIRouter(tags=["alpha-discovery"])


def _get_factors_dep() -> dict[str, FactorConfig]:
    from pfm import main as main_mod

    return main_mod.app.state.factors


def _get_cache_dep() -> CacheBackend:
    from pfm import main as main_mod

    return getattr(main_mod.app.state, "cache", NullCache())


@router.post("/reverse-finder", response_model=ReverseFinderResponse)
def reverse_finder_endpoint(
    body: ReverseFinderRequest,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)],
    cache: Annotated[CacheBackend, Depends(_get_cache_dep)],
    pool: Annotated[
        PoolMode,
        Query(
            description=(
                "Candidate-pool selection mode when ``candidate_factor_ids`` "
                "is null. ``curated`` (default) uses a source-prioritised, "
                "theme-balanced ≤200-factor pool (~3s end-to-end). "
                "``top_volume`` uses the top-N factors by 24h volume from "
                "the homepage cache. ``all`` iterates every factor in "
                "factors.yml (~1360, slow). ``theme`` is reserved for "
                "future per-theme pools."
            ),
        ),
    ] = "curated",
) -> ReverseFinderResponse:
    """Top-k Polymarket / Kalshi markets that best explain a ticker's returns."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")

    pool_used: str
    if body.candidate_factor_ids:
        # Accept id / slug / name in candidate_factor_ids and normalise
        # to the canonical id (so the downstream fetcher, which keys on
        # id, keeps working). Unknown queries are reported with
        # did_you_mean suggestions instead of being silently dropped.
        from pfm.factor_resolver import (
            resolve_factor,
            suggest_factors_with_meta,
        )

        candidate_ids = []
        unknown_with_hints: list[dict[str, object]] = []
        for q in body.candidate_factor_ids:
            fc = resolve_factor(q, factors)
            if fc is None:
                unknown_with_hints.append(
                    {
                        "query": q,
                        "did_you_mean": suggest_factors_with_meta(
                            q,
                            factors,
                            top_k=3,
                        ),
                    }
                )
            else:
                candidate_ids.append(fc.id)
        if not candidate_ids:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "none of candidate_factor_ids matches a known factor",
                    "unknown": unknown_with_hints,
                },
            )
        pool_used = "explicit"
    elif pool == "all":
        # Explicit opt-in for the full sweep. Slow (~1360 factors) but
        # available when the caller really wants exhaustive coverage.
        candidate_ids = sorted(factors.keys())
        pool_used = f"all_{len(candidate_ids)}"
    elif pool == "theme":
        # Reserved for future per-theme cohorts; today it falls through
        # to the same top-volume default so the contract is stable.
        candidate_ids, src = _top_volume_candidate_ids(factors, DEFAULT_CANDIDATE_LIMIT)
        pool_used = f"theme:{src}"
    elif pool == "top_volume":
        # Legacy: top-N by 24h volume from the homepage cache.
        candidate_ids, src = _top_volume_candidate_ids(factors, DEFAULT_CANDIDATE_LIMIT)
        pool_used = f"top_volume:{src}"
    else:
        # ``curated`` (default): source-prioritised, theme-balanced pool
        # of at most CURATED_POOL_LIMIT (200) ids.
        candidate_ids = _curated_candidate_ids(factors, CURATED_POOL_LIMIT)
        if not candidate_ids:
            # Edge case: tiny factor catalogue (tests) — fall back to all.
            candidate_ids = sorted(factors.keys())
        pool_used = f"curated_{len(candidate_ids)}"

    cache_key = _cache_key(
        "reverse-finder",
        body.ticker,
        body.start.isoformat(),
        body.end.isoformat(),
        body.k,
        body.return_type,
        body.epsilon,
        body.min_obs,
        tuple(candidate_ids),
        pool_used,
    )
    blob = cache.get(cache_key)
    if blob:
        return ReverseFinderResponse.model_validate_json(blob)

    try:
        result = reverse_find_factors(
            ticker=body.ticker,
            candidate_factor_ids=candidate_ids,
            start=body.start,
            end=body.end,
            k=body.k,
            return_type=body.return_type,  # type: ignore[arg-type]
            epsilon=body.epsilon,
            min_obs=body.min_obs,
            returns_fetcher=_build_returns_fetcher(),
            factor_fetcher=_build_factor_fetcher(factors),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("reverse_finder failed")
        raise HTTPException(status_code=500, detail=f"reverse_finder error: {e!r}") from e

    # Enrich with display names from the catalogue.
    for row in result.get("top_factors", []):
        fc = factors.get(row["factor_id"])
        row["factor_name"] = fc.name if fc is not None else None

    result["pool_used"] = pool_used
    result["n_candidates_evaluated"] = len(candidate_ids)

    resp = ReverseFinderResponse.model_validate(result)
    cache.set(cache_key, resp.model_dump_json().encode("utf-8"), TTL_SECONDS)
    return resp


# --- streaming variant ------------------------------------------------------


def _resolve_pool(
    body: ReverseFinderRequest,
    factors: dict[str, FactorConfig],
    pool: PoolMode,
) -> tuple[list[str], str]:
    """Resolve ``body.candidate_factor_ids`` + ``pool`` to ``(ids, pool_used)``.

    Centralises the branching logic shared by ``/reverse-finder`` and
    ``/reverse-finder/stream`` so the two endpoints can't drift.
    Raises ``HTTPException`` for the same conditions as the original
    endpoint (422 on entirely-unknown candidate_factor_ids).
    """
    if body.candidate_factor_ids:
        from pfm.factor_resolver import (
            resolve_factor,
            suggest_factors_with_meta,
        )

        candidate_ids: list[str] = []
        unknown_with_hints: list[dict[str, object]] = []
        for q in body.candidate_factor_ids:
            fc = resolve_factor(q, factors)
            if fc is None:
                unknown_with_hints.append(
                    {
                        "query": q,
                        "did_you_mean": suggest_factors_with_meta(
                            q,
                            factors,
                            top_k=3,
                        ),
                    }
                )
            else:
                candidate_ids.append(fc.id)
        if not candidate_ids:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "none of candidate_factor_ids matches a known factor",
                    "unknown": unknown_with_hints,
                },
            )
        return candidate_ids, "explicit"

    if pool == "all":
        ids = sorted(factors.keys())
        return ids, f"all_{len(ids)}"
    if pool == "theme":
        ids, src = _top_volume_candidate_ids(factors, DEFAULT_CANDIDATE_LIMIT)
        return ids, f"theme:{src}"
    if pool == "top_volume":
        ids, src = _top_volume_candidate_ids(factors, DEFAULT_CANDIDATE_LIMIT)
        return ids, f"top_volume:{src}"

    # curated (default)
    ids = _curated_candidate_ids(factors, CURATED_POOL_LIMIT)
    if not ids:
        ids = sorted(factors.keys())
    return ids, f"curated_{len(ids)}"


def _sse_format(event: str, data: dict) -> bytes:
    """Encode one Server-Sent Events frame."""
    payload = json.dumps(data, default=str, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode()


def _stream_reverse_finder(
    body: ReverseFinderRequest,
    factors: dict[str, FactorConfig],
    pool: PoolMode,
) -> Iterator[bytes]:
    """SSE generator: emits meta, factor*, done (or error on failure).

    All exceptions raised inside the underlying generator are caught and
    surfaced as a final ``event: error`` frame so the browser EventSource
    sees a clean stream-close instead of a torn TCP connection.
    """
    try:
        candidate_ids, pool_used = _resolve_pool(body, factors, pool)
    except HTTPException as he:
        yield _sse_format(
            "error",
            {"error": "pool_resolution_failed", "status": he.status_code, "detail": he.detail},
        )
        return

    # Emit a router-level meta frame first so the client knows ticker /
    # pool / candidate count before any expensive work runs. The
    # generator's own ``"meta"`` event (with the post-filter
    # ``n_candidates``) is forwarded as a second frame below.
    yield _sse_format(
        "meta",
        {
            "ticker": body.ticker,
            "pool_used": pool_used,
            "n_candidates": len(candidate_ids),
            "start": body.start.isoformat(),
            "end": body.end.isoformat(),
            "k": body.k,
        },
    )

    # Deadline + heartbeat bookkeeping. SSE comment lines (``: ping``) keep
    # the connection warm through idle-timeout proxies; the timeout event
    # gives the client a clean reconnect path on stalled selections.
    start_t = time.monotonic()
    last_ping = start_t

    try:
        gen = iter_reverse_find_factors(
            ticker=body.ticker,
            candidate_factor_ids=candidate_ids,
            start=body.start,
            end=body.end,
            k=body.k,
            return_type=body.return_type,  # type: ignore[arg-type]
            epsilon=body.epsilon,
            min_obs=body.min_obs,
            returns_fetcher=_build_returns_fetcher(),
            factor_fetcher=_build_factor_fetcher(factors),
        )
        factor_events_emitted = 0
        for step in gen:
            # Hard deadline. Self-terminate so the browser EventSource
            # auto-reconnects instead of hanging on a half-open stream.
            if time.monotonic() - start_t > MAX_STREAM_SECONDS:
                yield b"event: timeout\ndata: max stream duration reached\n\n"
                return
            if step.kind == "meta":
                meta_extra = dict(step.extra or {})
                meta_extra.setdefault("pool_used", pool_used)
                yield _sse_format("meta", meta_extra)
            elif step.kind == "factor":
                if factor_events_emitted >= body.k:
                    # Defensive: the generator already caps at k.
                    continue
                fc = factors.get(step.factor_id or "")
                yield _sse_format(
                    "factor",
                    {
                        "rank": step.rank,
                        "factor_id": step.factor_id,
                        "factor_name": fc.name if fc is not None else None,
                        "delta_r_squared": step.delta_r2,
                        "beta": step.beta,
                        "t_stat": step.t_stat,
                        "vif": step.vif,
                        "cumulative_r2": step.cumulative_r2,
                    },
                )
                factor_events_emitted += 1
            elif step.kind == "done":
                done_payload = dict(step.extra or {})
                done_payload.setdefault("pool_used", pool_used)
                yield _sse_format("done", done_payload)
            # Inject a keep-alive ping if we haven't emitted a real frame
            # within ``HEARTBEAT_INTERVAL_S``. Cheap (~10 bytes) and silently
            # ignored by EventSource clients per the SSE spec.
            now = time.monotonic()
            if now - last_ping > HEARTBEAT_INTERVAL_S:
                yield b": ping\n\n"
            last_ping = now
    except ValueError as e:
        yield _sse_format("error", {"error": str(e)})
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("reverse_finder stream failed")
        yield _sse_format("error", {"error": f"reverse_finder error: {e!r}"})


@router.post("/reverse-finder/stream")
def reverse_finder_stream_endpoint(
    body: ReverseFinderRequest,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)],
    pool: Annotated[
        PoolMode,
        Query(
            description=(
                "Same as ``/reverse-finder``. ``curated`` (default) is "
                "recommended for interactive use."
            ),
        ),
    ] = "curated",
) -> StreamingResponse:
    """Server-Sent Events variant of ``/reverse-finder``.

    Emits one ``event: factor`` frame per forward-selection step so the
    frontend can animate bars as picks land. Always closes with an
    ``event: done`` (or ``event: error``) frame.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")

    return StreamingResponse(
        _stream_reverse_finder(body, factors, pool),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/alpha/prediction-driven", response_model=PredictionAlphaResponse)
def prediction_driven_endpoint(
    body: PredictionAlphaRequest,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)],
    cache: Annotated[CacheBackend, Depends(_get_cache_dep)],
) -> PredictionAlphaResponse:
    """Univariate β / R² scan: which equities load on a single PM factor."""
    from pfm.factor_resolver import resolve_or_404

    fc = resolve_or_404(body.factor_id, factors, status_code=404)

    tickers = list(body.candidate_tickers) if body.candidate_tickers else list(DEFAULT_TICKERS)

    cache_key = _cache_key(
        "prediction-alpha",
        body.factor_id,
        tuple(tickers),
        body.window_days,
        body.top_n,
        body.delta_logit_assumed,
        body.return_type,
        body.epsilon,
    )
    blob = cache.get(cache_key)
    if blob:
        return PredictionAlphaResponse.model_validate_json(blob)

    try:
        result = prediction_driven_alpha(
            factor_id=body.factor_id,
            candidate_tickers=tickers,
            window_days=body.window_days,
            top_n=body.top_n,
            delta_logit_assumed=body.delta_logit_assumed,
            return_type=body.return_type,  # type: ignore[arg-type]
            epsilon=body.epsilon,
            factor_name=fc.name,
            returns_fetcher=_build_returns_fetcher(),
            factor_fetcher=_build_factor_fetcher(factors),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("prediction_driven_alpha failed")
        raise HTTPException(status_code=500, detail=f"prediction-alpha error: {e!r}") from e

    resp = PredictionAlphaResponse.model_validate(result)
    cache.set(cache_key, resp.model_dump_json().encode("utf-8"), TTL_SECONDS)
    return resp


__all__ = [
    "CURATED_PER_THEME_CAP",
    "CURATED_POOL_LIMIT",
    "DEFAULT_CANDIDATE_LIMIT",
    "PredictionAlphaRequest",
    "PredictionAlphaResponse",
    "ReverseFinderRequest",
    "ReverseFinderResponse",
    "router",
]
