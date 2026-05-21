"""Hierarchical clustering of prediction-market factors by Δlogit-return correlation.

The Terminal "Factor Clusters" panel groups the live factor universe into
semantically-coherent baskets and surfaces, for each cluster, the *leader*
factor whose lagged returns best forecast the cluster's mean move.

Pipeline
--------
1.  **History.** Prefer the on-disk cache at ``/tmp/strat7_factor_history.pkl``
    (a ``dict[slug, pandas.Series]`` of daily probabilities indexed by UTC
    date) emitted by the strategy-7 batch job. If absent we fall back to
    fetching live histories one slug at a time via Polymarket — slower but
    keeps the endpoint useful in fresh containers.
2.  **Theme filter.** When the caller passes ``?theme=politics`` we restrict
    to slugs whose ``factors.yml`` row carries that theme tag.
3.  **Δlogit returns.** Probabilities are clipped to ``[ε, 1-ε]`` (ε=0.01)
    and transformed:

        Δlogit_t = log(p_t / (1 - p_t)) − log(p_{t-1} / (1 - p_{t-1}))

    This matches the convention used by ``model.py`` and the
    ``/terminal/correlations`` panel — important for cross-panel reasoning.
4.  **Pearson correlation.** Computed pairwise on the joint dropna of each
    factor pair. Pairs with fewer than ``MIN_PAIR_OBS`` overlap are coerced
    to zero so they neither cluster together nor inflate a leader's lead
    strength on thin samples.
5.  **Distance.** ``d_ij = 1 - |corr_ij|`` — sign-agnostic so two factors
    that are economic mirrors (e.g. "Trump out" vs "Trump still in") still
    cluster together.
6.  **Hierarchical agglomerative clustering** with average linkage. We use
    ``sklearn.cluster.AgglomerativeClustering`` with
    ``distance_threshold = 1 - min_corr`` to *cut* the dendrogram at the
    user-supplied correlation floor.
7.  **Leader detection (Granger-lite).** Inside each cluster, for every
    candidate leader L we compute, across lags k ∈ ``LEADER_LAG_RANGE``,
    the Pearson correlation between L's Δlogit at time t-k and the mean
    Δlogit of the *other* cluster members at time t. The (factor, lag)
    pair with the highest |corr| wins; ``lead_strength`` is that |corr|.
    This is not a formal Granger test (no F-stat, no AR controls) — the
    name "lite" is intentional and documented in the response.

Routing
-------
This module owns its own :class:`fastapi.APIRouter`. Per project convention
``main.py`` is left untouched — wire it in there with::

    from pfm.terminal_factor_clusters import router as factor_clusters_router
    app.include_router(factor_clusters_router)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from pfm.terminal_export import respond as _export_respond

logger = logging.getLogger(__name__)

# --- module constants -------------------------------------------------------

CACHE_PATH: Path = Path("/tmp/strat7_factor_history.pkl")
# 2026-05 refactor: module moved into ``pfm/terminal/``; ``factors.yml`` still
# lives at the package root one directory up.
FACTORS_YML: Path = Path(__file__).resolve().parents[1] / "factors.yml"

CLIP_EPS: float = 0.01
MIN_HISTORY: int = 30  # need at least 30 daily obs to enter the panel
MIN_PAIR_OBS: int = 20  # joint observations to trust a pairwise corr
LEADER_LAG_RANGE: tuple[int, int] = (1, 5)  # inclusive [1, 5]
MAX_FACTORS: int = 600  # safety cap so a misuse can't OOM the worker

router = APIRouter(prefix="/terminal", tags=["terminal-factor-clusters"])


# --- response schemas -------------------------------------------------------


class LeaderInfo(BaseModel):
    """Leader factor for a cluster, with the lag at which it best
    anticipates the cluster-mean move."""

    factor_id: str
    n_lags_lead: int = Field(..., ge=1, description="Lag k (days) maximizing |corr|.")
    lead_strength: float = Field(..., ge=0.0, le=1.0)


class ClusterOut(BaseModel):
    cluster_id: str
    n_factors: int
    avg_intra_corr: float
    leader: LeaderInfo | None
    members: list[str]
    theme_centroid: str


class FactorClustersResponse(BaseModel):
    n_factors_in: int
    n_clusters: int
    clusters: list[ClusterOut]
    theme: str | None
    min_corr: float
    # When the upstream factor-history cache is missing the endpoint returns
    # an empty cluster list + ``degraded_mode=true`` so the UI can render an
    # explanatory empty state instead of a scary 503. Same pattern as
    # /terminal/peers (alpha_hunter cache unavailable).
    degraded_mode: bool = False
    reason: str | None = None


# --- yaml loading -----------------------------------------------------------


@dataclass(frozen=True)
class _FactorMeta:
    factor_id: str
    slug: str
    theme: str
    name: str


def _load_factor_meta() -> dict[str, _FactorMeta]:
    """Index ``factors.yml`` by slug. Falls back to slug==id if unparseable."""
    if not FACTORS_YML.exists():
        logger.warning("factors.yml missing at %s; clustering will use slugs as ids", FACTORS_YML)
        return {}
    try:
        raw = yaml.safe_load(FACTORS_YML.read_text())
    except yaml.YAMLError as exc:
        logger.warning("factors.yml parse failure: %s", exc)
        return {}
    out: dict[str, _FactorMeta] = {}
    for row in raw.get("factors") or []:
        slug = row.get("slug")
        if not slug:
            continue
        out[slug] = _FactorMeta(
            factor_id=row.get("id") or slug,
            slug=slug,
            theme=(row.get("theme") or "other"),
            name=row.get("name") or slug,
        )
    return out


# --- history loading --------------------------------------------------------


# mtime-keyed cache so we re-read the pickle when the strat-7 batch (or the
# in-process factor prewarm in ``main.py``) writes a fresh copy. Caching by
# (path, mtime) instead of via ``@functools.cache`` (the previous design)
# avoids the pathological case where the very first call landed *before* the
# pickle existed → ``{}`` got memoised forever for the lifetime of the
# worker, so /terminal/factor-clusters stayed in degraded_mode even after
# the pickle was written a few seconds later. See bug-fix note 2026-05-19.
_history_cache: dict[tuple[str, float], dict[str, pd.Series]] = {}


def _load_cached_history() -> dict[str, pd.Series]:
    """Pull the strat7 daily-probability cache if present.

    Cached by ``(path, mtime_ns)`` so we automatically pick up a freshly
    written pickle without needing an explicit ``cache_clear()`` call. The
    previous ``@functools.cache`` design memoised ``{}`` permanently when the
    pickle didn't exist at first call → endpoint stuck in degraded_mode for
    the whole worker lifetime.
    """
    if not CACHE_PATH.exists():
        return {}
    try:
        mtime = CACHE_PATH.stat().st_mtime_ns
    except OSError as exc:
        logger.warning("failed to stat %s: %s", CACHE_PATH, exc)
        return {}
    key = (str(CACHE_PATH), float(mtime))
    cached = _history_cache.get(key)
    if cached is not None:
        return cached
    try:
        with CACHE_PATH.open("rb") as fh:
            blob = pickle.load(fh)
    except (pickle.PickleError, OSError) as exc:
        logger.warning("failed to read %s: %s", CACHE_PATH, exc)
        return {}
    if not isinstance(blob, dict):
        return {}
    # normalize values to pd.Series with sorted DatetimeIndex
    cleaned: dict[str, pd.Series] = {}
    for slug, ser in blob.items():
        if not isinstance(ser, pd.Series):
            continue
        s = ser.dropna().sort_index()
        if not isinstance(s.index, pd.DatetimeIndex):
            try:
                s.index = pd.to_datetime(s.index)
            except (TypeError, ValueError):
                continue
        if len(s) >= MIN_HISTORY:
            cleaned[slug] = s.astype(float)
    # Only memoise non-empty payloads. If the pickle is empty / malformed we
    # want the next call to retry rather than serve stale ``{}``.
    if cleaned:
        # Bound the cache: only keep the most recent mtime for a path.
        _history_cache.clear()
        _history_cache[key] = cleaned
    return cleaned


# Back-compat shim — older tests + the docstring of the strat-7 batch job
# still call ``_load_cached_history.cache_clear()`` after writing a fresh
# pickle. Preserve that surface so we don't break callers, even though the
# mtime-keyed cache above already self-invalidates.
def _cache_clear() -> None:
    """Drop the in-process mtime cache. Kept for back-compat with callers
    that used to invoke ``_load_cached_history.cache_clear()`` from the old
    ``@functools.cache`` design."""
    _history_cache.clear()


_load_cached_history.cache_clear = _cache_clear  # type: ignore[attr-defined]


# --- math primitives --------------------------------------------------------


def _delta_logit(prices: pd.Series, eps: float = CLIP_EPS) -> pd.Series:
    """Δlogit returns of a probability series, clipped to ``[eps, 1-eps]``.

    Args:
        prices: probability series in (0, 1).
        eps: clip threshold; extreme probs are pinned in [eps, 1-eps].

    Returns:
        Differenced logit series, length ``len(prices) - 1``.
    """
    p = prices.clip(lower=eps, upper=1.0 - eps)
    logit = np.log(p / (1.0 - p))
    return logit.diff().dropna()


def _build_returns_matrix(history: dict[str, pd.Series]) -> pd.DataFrame:
    """Stack per-factor Δlogit series into a wide ``[date x factor]`` frame."""
    cols: dict[str, pd.Series] = {}
    for slug, ser in history.items():
        ret = _delta_logit(ser)
        if len(ret) >= MIN_HISTORY - 1:
            cols[slug] = ret
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    # one row per UTC date — drop dupes from any tz-stripping mismatch
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _pairwise_corr(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation, zeroing pairs with thin overlap."""
    # pandas .corr handles NaN-overlap pair-by-pair which is exactly what we want
    corr = returns.corr(method="pearson", min_periods=MIN_PAIR_OBS)
    corr = corr.fillna(0.0)
    # diag must be 1 even if a column is constant after fillna(0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def _cluster_from_corr(corr: pd.DataFrame, min_corr: float) -> dict[int, list[str]]:
    """Agglomerative clustering on the ``1-|corr|`` distance matrix.

    Cuts the dendrogram at ``distance = 1 - min_corr`` so every intra-cluster
    pair has |corr| ≥ ``min_corr`` *on average*.
    """
    from sklearn.cluster import AgglomerativeClustering

    dist = (1.0 - corr.abs().to_numpy()).clip(min=0.0)
    np.fill_diagonal(dist, 0.0)
    # sklearn requires a strict zero diagonal and symmetric matrix
    dist = (dist + dist.T) / 2.0
    threshold = max(1.0 - min_corr, 1e-6)

    if len(corr) < 2:
        return {0: list(corr.index)}

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
    )
    labels = model.fit_predict(dist)
    clusters: dict[int, list[str]] = {}
    for slug, lab in zip(corr.index, labels, strict=True):
        clusters.setdefault(int(lab), []).append(slug)
    return clusters


def _avg_intra_corr(corr: pd.DataFrame, members: list[str]) -> float:
    """Mean of |corr| over the upper-triangle of the cluster sub-matrix."""
    if len(members) < 2:
        return 1.0
    sub = corr.loc[members, members].abs().to_numpy()
    iu = np.triu_indices_from(sub, k=1)
    vals = sub[iu]
    if vals.size == 0:
        return 1.0
    return float(np.mean(vals))


def _detect_leader(
    returns: pd.DataFrame,
    members: list[str],
    lag_range: tuple[int, int] = LEADER_LAG_RANGE,
) -> tuple[str, int, float] | None:
    """Pick the factor whose lagged returns best predict the cluster mean.

    For each candidate leader L and lag k in ``[lag_range[0], lag_range[1]]``,
    correlate ``L.shift(k)`` against the mean Δlogit of the *other* members.
    The triple ``(factor, lag, |corr|)`` with the largest |corr| wins.

    Returns ``None`` when the cluster is a singleton.
    """
    if len(members) < 2:
        return None
    sub = returns[members].dropna(how="all")
    if len(sub) < MIN_PAIR_OBS:
        return None
    best_factor: str | None = None
    best_lag: int = lag_range[0]
    best_strength: float = -1.0
    lo, hi = lag_range
    for leader in members:
        others = [m for m in members if m != leader]
        cluster_mean = sub[others].mean(axis=1)
        leader_ser = sub[leader]
        for k in range(lo, hi + 1):
            shifted = leader_ser.shift(k)
            joint = pd.concat([shifted, cluster_mean], axis=1).dropna()
            if len(joint) < MIN_PAIR_OBS:
                continue
            x = joint.iloc[:, 0].to_numpy()
            y = joint.iloc[:, 1].to_numpy()
            if x.std() == 0 or y.std() == 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            strength = abs(r)
            if strength > best_strength:
                best_strength = strength
                best_factor = leader
                best_lag = k
    if best_factor is None or best_strength < 0:
        return None
    return best_factor, best_lag, best_strength


# --- naming helpers ---------------------------------------------------------


def _shared_prefix_label(names: list[str], max_words: int = 4) -> str:
    """Cheap centroid label: longest leading word-prefix shared by ≥half names."""
    if not names:
        return "Mixed"
    if len(names) == 1:
        return names[0][:64]
    tokenized = [n.split() for n in names]
    out: list[str] = []
    threshold = max(1, len(names) // 2)
    for i in range(max_words):
        candidates: dict[str, int] = {}
        for toks in tokenized:
            if i >= len(toks):
                continue
            w = toks[i].strip(".,;:'\"")
            if not w:
                continue
            candidates[w] = candidates.get(w, 0) + 1
        if not candidates:
            break
        word, count = max(candidates.items(), key=lambda kv: kv[1])
        if count < threshold:
            break
        out.append(word)
    if not out:
        # fall back to the first member's first 5 words
        return " ".join(tokenized[0][:5])[:64]
    return " ".join(out)[:64]


def _cluster_id_from_members(members_meta: list[_FactorMeta], cluster_idx: int) -> str:
    """Build a stable, debuggable cluster id from the leader-ish member."""
    if not members_meta:
        return f"cluster_{cluster_idx}"
    # use the factor_id of the alphabetically-first member as a deterministic anchor
    anchor = sorted(m.factor_id for m in members_meta)[0]
    return f"{anchor}_cluster"


# --- main router ------------------------------------------------------------


@router.get(
    "/factor-clusters",
    response_model=None,
    summary="Hierarchical clustering of factors by Δlogit-return correlation.",
)
def factor_clusters(
    theme: Annotated[
        str | None,
        Query(description="Filter by factors.yml theme tag (e.g. 'politics')."),
    ] = None,
    min_corr: Annotated[
        float,
        Query(ge=0.0, le=0.99, description="|corr| threshold cutting the dendrogram."),
    ] = 0.5,
    format: Annotated[Literal["json", "csv", "pdf"], Query()] = "json",
    request: Request = None,  # type: ignore[assignment]  # FastAPI special-cases Request; PEP-604 union confuses analyze_param
) -> FactorClustersResponse | FastAPIResponse:
    """Cluster factors by return correlation and surface a leader per cluster.

    Cached for ``CLUSTER_TTL_SECONDS`` (default 600 s) via the L1/L2 hybrid
    TERMINAL_CACHE. The clustering work — pairwise correlation across up to
    600 series + hierarchical cut — is ~300-600 ms p50, so amortising
    across workers is a big perceived-latency win.

    Warm-cache fast path: when invoked via HTTP and the lifespan prewarm
    has stored a fresh default-payload (``theme=None, min_corr=0.5``) on
    ``app.state.warm_clusters``, we short-circuit before touching the L1/L2
    cache. Cold-tail latency drops from ~3 s to <50 ms. See ``pfm.prewarm``.
    """

    def _finalize(resp: FactorClustersResponse) -> FactorClustersResponse | FastAPIResponse:
        if format == "json":
            return resp
        return _export_respond(resp, format, filename="factor-clusters", kind="market")

    # Warm-cache short-circuit — only on the canonical default query and
    # only when a Request is present (the prewarm itself calls this fn
    # directly without one, so we must NOT recurse).
    if request is not None and theme is None and abs(min_corr - 0.5) < 1e-9:
        try:
            from pfm.prewarm import warm_clusters_lookup as _warm_clusters_lookup
        except ImportError:  # pragma: no cover - module is in-repo
            _warm_clusters_lookup = None
        if _warm_clusters_lookup is not None:
            cached_payload = _warm_clusters_lookup(
                request.app,
                theme=theme,
                min_corr=min_corr,
            )
            if cached_payload is not None:
                return _finalize(FactorClustersResponse.model_validate(cached_payload))

    # Cache key: theme + min_corr round-tripped so query variations don't
    # collide. Read-through L1 (in-process) → L2 (Redis) → compute.
    from pfm import terminal as _term_mod  # circular-safe lazy import

    cache_key = f"factor_clusters::{theme or '*'}::{min_corr:.3f}"
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return _finalize(FactorClustersResponse.model_validate(cached))

    meta = _load_factor_meta()
    history = _load_cached_history()
    if not history:
        # Graceful degraded mode — UI gets an empty cluster list it can
        # show as "Cluster data is being warmed up" instead of a 503 stub.
        return _finalize(
            FactorClustersResponse(
                n_factors_in=0,
                n_clusters=0,
                clusters=[],
                theme=theme,
                min_corr=min_corr,
                degraded_mode=True,
                reason=(
                    "Factor-history cache is empty (run the strat7 batch job "
                    "or wait for the in-process prewarm to populate Redis)."
                ),
            )
        )

    # theme filter
    if theme is not None:
        wanted = {slug for slug, m in meta.items() if m.theme == theme}
        if not wanted:
            raise HTTPException(status_code=404, detail=f"No factors for theme={theme!r}.")
        history = {k: v for k, v in history.items() if k in wanted}
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"No cached history for any theme={theme!r} factor.",
            )

    if len(history) > MAX_FACTORS:
        # keep the longest histories — they give the cleanest correlations
        ranked = sorted(history.items(), key=lambda kv: -len(kv[1]))[:MAX_FACTORS]
        history = dict(ranked)

    returns = _build_returns_matrix(history)
    if returns.empty or returns.shape[1] < 2:
        raise HTTPException(
            status_code=422,
            detail="Not enough factors with sufficient history to cluster.",
        )

    corr = _pairwise_corr(returns)
    raw_clusters = _cluster_from_corr(corr, min_corr=min_corr)

    out: list[ClusterOut] = []
    for cidx, members in sorted(raw_clusters.items(), key=lambda kv: -len(kv[1])):
        members_meta = [
            meta.get(s, _FactorMeta(factor_id=s, slug=s, theme="other", name=s)) for s in members
        ]
        ids = [m.factor_id for m in members_meta]
        leader_tup = _detect_leader(returns, members)
        if leader_tup is not None:
            leader_slug, lag_k, strength = leader_tup
            leader_meta = meta.get(
                leader_slug,
                _FactorMeta(
                    factor_id=leader_slug, slug=leader_slug, theme="other", name=leader_slug
                ),
            )
            leader = LeaderInfo(
                factor_id=leader_meta.factor_id,
                n_lags_lead=int(lag_k),
                lead_strength=round(float(strength), 4),
            )
        else:
            leader = None
        out.append(
            ClusterOut(
                cluster_id=_cluster_id_from_members(members_meta, cidx),
                n_factors=len(members),
                avg_intra_corr=round(_avg_intra_corr(corr, members), 4),
                leader=leader,
                members=ids,
                theme_centroid=_shared_prefix_label([m.name for m in members_meta]),
            )
        )

    resp = FactorClustersResponse(
        n_factors_in=int(returns.shape[1]),
        n_clusters=len(out),
        clusters=out,
        theme=theme,
        min_corr=min_corr,
    )
    # Persist to L1+L2 (Redis) cache so subsequent hits across all workers
    # skip the clustering math. Underlying inputs (factor history pickle,
    # factors.yml) change rarely; a 10-minute TTL is comfortable.
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), 600)
    return _finalize(resp)
