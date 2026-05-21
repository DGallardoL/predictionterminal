"""Multi-market jump cluster detector.

A single jump on one prediction market is an event. **Many jumps across many
markets within a few minutes of each other, sharing the same news terms**, is
a *macro event* — an FOMC decision, a Trump tweet, an earnings beat, a
geopolitical headline. Surfacing those clusters lets the Terminal answer the
question "what just moved the board?" rather than "what just moved this one
slug?".

This module is the post-processor that consumes ``/terminal/jumps/{slug}``
output for many slugs and groups them.

Algorithm (greedy union-find)
-----------------------------
For each pair ``(slug_a / jump_a, slug_b / jump_b)`` we evaluate two gates::

    1. |ts_a - ts_b| <= time_tol_minutes      (temporal proximity)
    2. jaccard(terms_a, terms_b) >= kw_min_jaccard   (semantic overlap)

If both pass, the two jumps are linked. Linked sets are unioned via a small
disjoint-set forest (path-compression union-find). The result is the set of
connected components — same algorithm Kruskal uses to build an MST.

We chose **greedy union-find over hierarchical clustering** for three reasons:

* **Interpretability.** A single dual-gate rule ("close in time AND shares
  terms") is something a quant can argue with. A linkage matrix is not.
* **No need for a global distance.** The pair predicate isn't a true metric —
  Jaccard is, but the time gate is a hard cutoff. Hierarchical clustering
  with a custom distance would over-engineer this.
* **Determinism + linear behaviour.** Union-find is O(N α(N)) where N is the
  total jumps; for a Terminal with ~20 slugs and ~30 jumps each this is
  trivially fast and the cluster ids are stable across runs.

A jump that has no qualifying partner stays in its own singleton component;
the response filters singletons out by default (a "cluster of one" isn't a
cluster — it's just a jump).

Representative selection
------------------------
For each cluster we pick:

* ``ts_iso``: **median** member timestamp. Robust to one anchor jump being
  much earlier than the others (e.g. an EU close ahead of a US tweet).
* ``dominant_terms``: top-5 terms by frequency across *all* member-jump
  articles (not the strict intersection — for a 5-market FOMC cluster the
  strict intersection is often empty because each market's article uses
  slightly different anchor words).
* ``representative_headline``: the article whose ``relevance_score`` is
  highest across the whole cluster, falling back to the lowest
  ``|seconds_from_jump|`` to break ties.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from collections import Counter
from statistics import median
from typing import Annotated

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.config import Settings, get_settings
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal import jumps as _jumps_module
from pfm.terminal.jumps import (
    DEFAULT_MAD_K,
    DEFAULT_MIN_JUMP_PP,
    Jump,
    TerminalJumpsResponse,
    get_jumps,
)

logger = logging.getLogger(__name__)


# --- knobs / cache ----------------------------------------------------------

DEFAULT_TIME_TOL_MINUTES: float = 5.0
DEFAULT_KW_MIN_JACCARD: float = 0.20
DEFAULT_DAYS: int = 14
MIN_DAYS: int = 1
MAX_DAYS: int = 90
MAX_SLUGS: int = 40  # hard cap to keep the fan-out manageable
# Lowered 2026-05-18 from 20 to 8: even with bounded concurrency, 20 slugs ×
# ~5-10 s gather() per slug blew past the 15 s gateway deadline on cold
# caches. Eight slugs is enough for a "what just moved the board" panel
# and keeps wall-clock under ~10 s warm-pool.
DEFAULT_TOP_N_SLUGS: int = 8
# Bumped 2026-05-16 from 10 to 20: with the default top-20-slug fan-out, the
# old cap of 10 forced two serial waves of ~5s GDELT/Reddit/HN gathers
# (observed wall-clock ~9.6s). Upstreams have plenty of headroom:
#   * Polymarket Gamma: 1000 req / 10 s
#   * GDELT 2.0: no documented hard limit, soft-throttles at ~1 rps but each
#     unique query is hit at most once per cluster window thanks to
#     _NEWS_GATHER_CACHE below
#   * Reddit / HN Algolia: generous, single-query each
# Single-batch firing collapses the wall-clock to ~5 s and the in-process
# news-gather cache deduplicates queries that overlap across slugs.
CONCURRENCY: int = 20  # bounded fan-out for the per-slug detection
DOMINANT_TERMS_K: int = 5
CACHE_TTL_SECONDS: int = 300

# Shared in-process news-gather cache. Multiple slugs in the same cluster
# request often build the **same** GDELT query (e.g. all the "Trump 2024" or
# "Fed rate" markets distill to the same anchor terms). Without this cache
# the per-slug call to ``_gather_all_news`` re-pays the full
# GDELT+Reddit+HN+RSS round-trip even when the result is identical. We
# install a thin caching wrapper around ``pfm.terminal.jumps._gather_all_news``
# at module import. The wrapper is idempotent (guarded by a sentinel attr)
# so re-importing this module — e.g. under ``pytest`` — doesn't double-wrap.
NEWS_GATHER_CACHE_TTL_SECONDS: int = 300
_NEWS_GATHER_CACHE: dict[tuple[str, str, int], tuple[float, list]] = {}
_NEWS_GATHER_CACHE_LOCK = threading.Lock()
# Per-cluster-request scope token. The wrapper consults this; if the current
# task isn't inside a cluster fan-out the wrapper is a no-op pass-through.
# This keeps the cache invisible to direct callers of ``/terminal/jumps/{slug}``
# and to unit tests that exercise ``get_jumps`` in isolation — they continue to
# see the real underlying fetcher, so per-test mocks aren't poisoned by a
# sibling test's cached payload.
_CLUSTER_SCOPE: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "pfm_jumps_cluster_scope", default=None
)

_CACHE = get_cache("terminal_jumps_cluster", ttl=CACHE_TTL_SECONDS)


def _install_news_gather_cache() -> None:
    """Monkey-patch ``jumps._gather_all_news`` with a query-level cache.

    Idempotent: subsequent calls become no-ops thanks to the sentinel
    ``_pfm_cluster_cached`` attribute on the installed wrapper.
    """
    original = getattr(_jumps_module, "_gather_all_news", None)
    if original is None:
        return
    if getattr(original, "_pfm_cluster_cached", False):
        return

    def _cached_gather(http_client, query: str, timespan: str):
        # Only consult the cache when we're inside an active cluster
        # fan-out (scope token set by ``get_jumps_clusters``). Direct callers
        # of ``/terminal/jumps/{slug}`` and unit tests that exercise
        # ``get_jumps`` in isolation get the original pass-through behaviour
        # so per-test mocks aren't poisoned by a sibling test's payload.
        scope = _CLUSTER_SCOPE.get()
        if scope is None:
            return original(http_client, query, timespan)
        # Scope-scoped key so concurrent cluster requests with different
        # mocks (in tests) don't collide. In production every cluster
        # request gets a fresh scope, so the cache lives for one fan-out
        # only — exactly the lifetime we want.
        key = (str(query or ""), str(timespan or ""), scope)
        now = time.time()
        with _NEWS_GATHER_CACHE_LOCK:
            entry = _NEWS_GATHER_CACHE.get(key)
            if entry is not None and entry[0] > now:
                return list(entry[1])
        result = original(http_client, query, timespan)
        # Only cache non-empty results: an empty payload usually means a
        # transient upstream failure (GDELT throttled, network blip) and
        # we'd rather refetch on the next call than poison the cache for
        # the full scope.
        if result:
            with _NEWS_GATHER_CACHE_LOCK:
                _NEWS_GATHER_CACHE[key] = (
                    now + NEWS_GATHER_CACHE_TTL_SECONDS,
                    list(result),
                )
        return result

    _cached_gather._pfm_cluster_cached = True  # type: ignore[attr-defined]
    _cached_gather._pfm_cluster_original = original  # type: ignore[attr-defined]
    _jumps_module._gather_all_news = _cached_gather  # type: ignore[assignment]


_install_news_gather_cache()


# --- Pydantic schemas -------------------------------------------------------


class ClusterMember(BaseModel):
    """One jump that's part of a cluster — slim reference back to the source."""

    slug: str
    ts_iso: str
    delta_pp: float
    sentiment_alignment: str = Field(
        "neutral",
        description="Inherited from the underlying Jump: agrees / disagrees / neutral.",
    )


class Cluster(BaseModel):
    """A group of ≥2 jumps tied together by time + keyword overlap."""

    cluster_id: int = Field(..., description="1-indexed; stable within a single response.")
    ts_iso: str = Field(
        ...,
        description="Representative timestamp (median of the member-jump timestamps).",
    )
    n_markets: int = Field(..., description="Distinct slugs participating.")
    n_articles: int = Field(..., description="Total matched articles across all member jumps.")
    dominant_terms: list[str] = Field(
        default_factory=list,
        description="Top-K terms by frequency across the cluster (K=5).",
    )
    representative_headline: str | None = Field(
        None,
        description=(
            "Single best headline — highest relevance × proximity inside the "
            "cluster — meant to be the human-readable "
            "label for the macro event."
        ),
    )
    member_jumps: list[ClusterMember] = Field(default_factory=list)


class JumpsClusterResponse(BaseModel):
    slugs: list[str]
    days: int
    time_tol_minutes: float
    kw_min_jaccard: float
    n_jumps_total: int
    n_clusters: int
    clusters: list[Cluster]


# --- pure helpers -----------------------------------------------------------


def _parse_ts(ts_iso: str) -> pd.Timestamp | None:
    """Parse an ISO-8601 timestamp into a UTC ``pd.Timestamp`` or None."""
    if not ts_iso:
        return None
    try:
        ts = pd.Timestamp(ts_iso)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _terms_for_jump(jump: Jump) -> set[str]:
    """Union of ``matched_terms`` across all top articles for one jump.

    Casefolded so 'Trump' and 'trump' collide. Empty strings dropped.
    """
    out: set[str] = set()
    for art in jump.top_articles or []:
        for t in art.matched_terms or []:
            t_norm = (t or "").strip().lower()
            if t_norm:
                out.add(t_norm)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    """Standard Jaccard. Empty ∪ empty returns 0 (we want no merge)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


class _UnionFind:
    """Tiny disjoint-set forest with path compression + union-by-rank.

    Why inline a class for ~10 nodes? Keeps the cluster logic readable —
    callers see ``uf.union(i, j)`` instead of dict gymnastics, and the
    behaviour is correct regardless of input order (which a naïve merge
    loop is *not*).
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def find_clusters(
    jumps_by_slug: dict[str, list[Jump]],
    *,
    time_tol_minutes: float = DEFAULT_TIME_TOL_MINUTES,
    kw_min_jaccard: float = DEFAULT_KW_MIN_JACCARD,
) -> list[Cluster]:
    """Group jumps across multiple slugs by time proximity AND keyword overlap.

    Pure function — no IO, deterministic. The unit tests in
    ``tests/terminal/test_jumps_cluster.py`` cover it directly with synthetic
    Jump objects.

    Args:
        jumps_by_slug: ``{slug -> [Jump, ...]}``. Order within each list
            doesn't matter; we sort the flat node list by timestamp before
            pair iteration so adjacency in the response feels natural.
        time_tol_minutes: max gap between two jumps to consider linkage.
        kw_min_jaccard: min Jaccard on the union of ``matched_terms`` of the
            two jumps' top articles. ``0`` would make this a pure-time
            cluster; ``1`` would require identical term sets.

    Returns:
        Sorted list of clusters (descending ``n_markets``). Singletons —
        components of size 1 — are filtered out.
    """
    # Flatten to a single list of nodes; remember (slug, jump, terms, ts).
    nodes: list[tuple[str, Jump, set[str], pd.Timestamp]] = []
    for slug, jumps in jumps_by_slug.items():
        for j in jumps:
            ts = _parse_ts(j.ts_iso)
            if ts is None:
                continue
            nodes.append((slug, j, _terms_for_jump(j), ts))
    n = len(nodes)
    if n < 2:
        return []

    # Sort by ts so the pair loop short-circuits cleanly when the time window
    # is exceeded. Sorting by ts also stabilises cluster ids across runs.
    nodes.sort(key=lambda r: r[3])

    uf = _UnionFind(n)
    tol_sec = float(time_tol_minutes) * 60.0
    for i in range(n):
        slug_i, _j_i, terms_i, ts_i = nodes[i]
        for k in range(i + 1, n):
            slug_k, _j_k, terms_k, ts_k = nodes[k]
            dt = (ts_k - ts_i).total_seconds()
            if dt > tol_sec:
                # Sorted order: no later node can satisfy the time window.
                break
            if slug_i == slug_k:
                # Don't merge two jumps on the *same* slug — a single market
                # ringing twice in 5 min is one event, not a macro cluster.
                continue
            jac = _jaccard(terms_i, terms_k)
            if jac >= kw_min_jaccard:
                uf.union(i, k)

    # Collect components
    components: dict[int, list[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        components.setdefault(root, []).append(idx)

    clusters: list[Cluster] = []
    cluster_seq = 0
    for member_idxs in components.values():
        if len(member_idxs) < 2:
            continue
        slugs_in_cluster = {nodes[i][0] for i in member_idxs}
        if len(slugs_in_cluster) < 2:
            # All hits were on the same slug (shouldn't happen given the
            # same-slug guard above, but a sanity check costs us nothing).
            continue
        cluster_seq += 1

        # Representative ts = median of member timestamps. Stay in UTC.
        ts_list = sorted(nodes[i][3] for i in member_idxs)
        mid = len(ts_list) // 2
        if len(ts_list) % 2 == 1:
            rep_ts = ts_list[mid]
        else:
            # pandas' Timestamp doesn't directly average; use unix seconds.
            # Build a fresh UTC Timestamp from the median epoch — careful
            # because ``pd.Timestamp(unit='s')`` is tz-naive on some pandas
            # versions and tz-aware on others; ``pd.to_datetime`` with
            # explicit ``utc=True`` is the portable spelling.
            lo = ts_list[mid - 1].timestamp()
            hi = ts_list[mid].timestamp()
            rep_ts = pd.to_datetime(median([lo, hi]), unit="s", utc=True)

        # Dominant terms = top-K by frequency across every member-jump article.
        term_counter: Counter[str] = Counter()
        article_total = 0
        best_article = None  # (rank_score, headline)
        for i in member_idxs:
            jump = nodes[i][1]
            for art in jump.top_articles or []:
                article_total += 1
                for t in art.matched_terms or []:
                    t_norm = (t or "").strip().lower()
                    if t_norm:
                        term_counter[t_norm] += 1
                # Headline ranking: higher relevance wins, then proximity.
                rank_score = (
                    float(art.relevance_score or 0.0),
                    -abs(int(art.seconds_from_jump or 0)),
                )
                if best_article is None or rank_score > best_article[0]:
                    best_article = (rank_score, art.headline)

        dominant_terms = [t for t, _c in term_counter.most_common(DOMINANT_TERMS_K)]
        rep_headline = best_article[1] if best_article else None

        members: list[ClusterMember] = []
        for i in member_idxs:
            slug_m, j_m, _terms_m, _ts_m = nodes[i]
            members.append(
                ClusterMember(
                    slug=slug_m,
                    ts_iso=j_m.ts_iso,
                    delta_pp=float(j_m.delta_pp),
                    sentiment_alignment=str(j_m.sentiment_alignment or "neutral"),
                )
            )
        # Sort members by ts so the response reads chronologically.
        members.sort(key=lambda m: m.ts_iso)

        clusters.append(
            Cluster(
                cluster_id=cluster_seq,
                ts_iso=rep_ts.isoformat().replace("+00:00", "Z"),
                n_markets=len(slugs_in_cluster),
                n_articles=article_total,
                dominant_terms=dominant_terms,
                representative_headline=rep_headline,
                member_jumps=members,
            )
        )

    # Sort by n_markets desc then by representative ts desc (newest big
    # events surface first — a 6-market FOMC cluster outranks a 2-market
    # earnings cluster).
    clusters.sort(key=lambda c: (-c.n_markets, c.ts_iso), reverse=False)
    clusters.sort(key=lambda c: c.n_markets, reverse=True)
    # Re-number cluster_id to match final sort order.
    for new_id, c in enumerate(clusters, start=1):
        c.cluster_id = new_id
    return clusters


# --- helpers for slug discovery + fan-out -----------------------------------


async def _fetch_default_slugs(
    http: httpx.AsyncClient | None,
    gamma_url: str,
    *,
    top_n: int = DEFAULT_TOP_N_SLUGS,
) -> list[str]:
    """Pull the top N slugs by 24h volume from Gamma.

    We could re-use ``_fetch_top_markets_async`` from ``terminal.homepage``,
    but importing it directly would create a circular-ish dep when the
    homepage router itself is being composed. The duplicated 8-line call
    is the smaller evil.
    """
    if http is None:
        return []
    base = gamma_url.rstrip("/")
    params: dict[str, str | int] = {
        "active": "true",
        "closed": "false",
        "limit": max(top_n * 2, 50),  # over-fetch to account for filtering
        "order": "volume24hr",
        "ascending": "false",
    }
    try:
        r = await http.get(f"{base}/markets", params=params, timeout=10.0)
        r.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("jumps_cluster: gamma slug discovery failed: %s", e)
        return []
    page = r.json() or []
    if not isinstance(page, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in page:
        slug = str(m.get("slug") or "").strip()
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
        if len(out) >= top_n:
            break
    return out


async def _safe_get_jumps_for_slug(
    request: Request,
    slug: str,
    days: int,
    mad_k: float,
    min_jump_pp: float,
    poly: PolymarketClient,
    sem: asyncio.Semaphore,
) -> tuple[str, list[Jump]]:
    """Call ``get_jumps`` for a single slug under a concurrency bound."""
    async with sem:
        try:
            resp: TerminalJumpsResponse = await get_jumps(  # type: ignore[misc]
                request=request,
                slug=slug,
                days=days,
                mad_k=mad_k,
                min_jump_pp=min_jump_pp,
                poly=poly,
            )
            return slug, list(resp.jumps)
        except HTTPException as e:
            # 404 (slug gone) and 502 (gamma flaky) should not nuke the whole
            # cluster response — degrade to "no jumps for this slug".
            logger.info("jumps_cluster: slug %s skipped (%s)", slug, e.detail)
            return slug, []
        except Exception as e:
            logger.warning("jumps_cluster: slug %s failed: %s", slug, e)
            return slug, []


# --- router -----------------------------------------------------------------


router = APIRouter(prefix="/terminal", tags=["terminal-jumps-cluster"])


def _get_polymarket_client(request: Request) -> PolymarketClient:
    poly: PolymarketClient | None = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


@router.get(
    "/jumps/cluster",
    response_model=JumpsClusterResponse,
    summary="Group jumps across many slugs into macro-event clusters.",
)
async def get_jumps_clusters(
    request: Request,
    slugs: Annotated[
        str,
        Query(
            description=(
                "Comma-separated Polymarket slugs. Omit to default to the top-20 by 24h volume."
            ),
            max_length=4000,
        ),
    ] = "",
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
    time_tol_minutes: Annotated[float, Query(ge=0.1, le=120.0)] = DEFAULT_TIME_TOL_MINUTES,
    kw_min_jaccard: Annotated[float, Query(ge=0.0, le=1.0)] = DEFAULT_KW_MIN_JACCARD,
    mad_k: Annotated[float, Query(ge=1.0, le=10.0)] = DEFAULT_MAD_K,
    min_jump_pp: Annotated[float, Query(ge=0.5, le=50.0)] = DEFAULT_MIN_JUMP_PP,
) -> JumpsClusterResponse:
    """For a list of slugs, run jump detection then cluster the results.

    The per-slug fan-out uses the existing cached ``/terminal/jumps/{slug}``
    so a second cluster call within ``CACHE_TTL_SECONDS=600`` hits warm.
    Concurrency is bounded to :data:`CONCURRENCY` to avoid stampeding GDELT
    + the Gamma API.
    """
    # 1. Resolve the slug list
    requested = [s.strip() for s in (slugs or "").split(",") if s.strip()]
    if requested:
        if len(requested) > MAX_SLUGS:
            raise HTTPException(
                status_code=400,
                detail=f"too many slugs (max {MAX_SLUGS}, got {len(requested)})",
            )
        slug_list = requested
    else:
        settings: Settings = get_settings()
        shared_http: httpx.AsyncClient | None = getattr(request.app.state, "async_http", None)
        slug_list = await _fetch_default_slugs(
            shared_http, settings.polymarket_gamma_url, top_n=DEFAULT_TOP_N_SLUGS
        )
        if not slug_list:
            # Nothing to do — return an empty envelope rather than 5xx.
            return JumpsClusterResponse(
                slugs=[],
                days=int(days),
                time_tol_minutes=float(time_tol_minutes),
                kw_min_jaccard=float(kw_min_jaccard),
                n_jumps_total=0,
                n_clusters=0,
                clusters=[],
            )

    # Cache key — round floats so 0.20 vs 0.2 collide.
    cache_key = (
        tuple(slug_list),
        int(days),
        round(float(time_tol_minutes), 2),
        round(float(kw_min_jaccard), 3),
        round(float(mad_k), 2),
        round(float(min_jump_pp), 2),
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return JumpsClusterResponse(**cached)

    poly = _get_polymarket_client(request)

    # 2. Fan out, bounded concurrency. Activate the shared news-gather cache
    # for the duration of this fan-out so two slugs that resolve to the same
    # GDELT/Reddit/HN query share one network round-trip. The scope token is
    # ``id(slug_list)`` so concurrent cluster requests each get their own
    # bucket (no cross-request leakage), and we clear the bucket in a
    # ``finally`` so the cache doesn't grow unbounded under load.
    scope_token = id(slug_list)
    scope_handle = _CLUSTER_SCOPE.set(scope_token)
    try:
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [
            _safe_get_jumps_for_slug(
                request=request,
                slug=s,
                days=int(days),
                mad_k=float(mad_k),
                min_jump_pp=float(min_jump_pp),
                poly=poly,
                sem=sem,
            )
            for s in slug_list
        ]
        pairs: list[tuple[str, list[Jump]]] = await asyncio.gather(*tasks)
    finally:
        _CLUSTER_SCOPE.reset(scope_handle)
        # Drop this request's bucket from the shared cache so a long-running
        # process doesn't accumulate stale entries from past requests.
        with _NEWS_GATHER_CACHE_LOCK:
            stale = [k for k in _NEWS_GATHER_CACHE if k[2] == scope_token]
            for k in stale:
                _NEWS_GATHER_CACHE.pop(k, None)
    jumps_by_slug: dict[str, list[Jump]] = dict(pairs)

    n_jumps_total = sum(len(v) for v in jumps_by_slug.values())

    # 3. Run the pure clusterer
    clusters = find_clusters(
        jumps_by_slug,
        time_tol_minutes=float(time_tol_minutes),
        kw_min_jaccard=float(kw_min_jaccard),
    )

    resp = JumpsClusterResponse(
        slugs=slug_list,
        days=int(days),
        time_tol_minutes=float(time_tol_minutes),
        kw_min_jaccard=float(kw_min_jaccard),
        n_jumps_total=n_jumps_total,
        n_clusters=len(clusters),
        clusters=clusters,
    )
    _CACHE.set(cache_key, resp.model_dump(), ttl=CACHE_TTL_SECONDS)
    return resp


__all__ = [
    "DEFAULT_KW_MIN_JACCARD",
    "DEFAULT_TIME_TOL_MINUTES",
    "Cluster",
    "ClusterMember",
    "JumpsClusterResponse",
    "find_clusters",
    "get_jumps_clusters",
    "router",
]
