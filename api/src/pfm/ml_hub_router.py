"""ML Hub — unsupervised structure of the prediction-market factor universe.

The ML Hub layers *machine-learning-flavoured insight* on top of the data the
platform already brings in. This first module — **Factor Galaxy** — answers a
question the flat factor catalogue cannot: *which prediction markets actually
move together, and how is the universe shaped?*

It deliberately reuses the proven pipeline behind ``/terminal/factor-clusters``
(daily-probability history → Δlogit returns → ``1-|corr|`` distance →
agglomerative clusters) and adds the missing layer: a **2-D embedding** of the
distance matrix via classical/metric MDS so the universe can be plotted as a
navigable scatter. Distance in the plot is honest — it is the dissimilarity
``1-|corr|``, not a t-SNE neighbourhood distortion — which matters for a project
that grades quant honesty. ``method=tsne`` is offered as a (clearly-labelled)
prettier-but-distorting alternative.

Why this is honest (no overfit risk)
-------------------------------------
There is **no forward prediction here**. We do not forecast returns — every
anti-alpha in this project died doing exactly that. This is purely descriptive
unsupervised geometry over historical co-movement: it surfaces latent themes,
redundant factors, and bridges between baskets. Those are insights you can act
on (de-dup a basket, find a hedge, spot a lone-wolf factor) without claiming
predictive edge.

Routing
-------
Owns its own :class:`fastapi.APIRouter`. Wire in ``main.py`` with::

    from pfm.ml_hub_router import router as ml_hub_router
    app.include_router(ml_hub_router)
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

# Reuse the battle-tested factor-clusters pipeline rather than reimplementing
# history loading / Δlogit / correlation / clustering. These are the same
# primitives the Terminal "Factor Clusters" panel relies on, so the ML Hub map
# and that panel stay consistent by construction.
from pfm.terminal.factor_clusters import (
    _build_returns_matrix,
    _cluster_from_corr,
    _detect_leader,
    _FactorMeta,
    _load_cached_history,
    _load_factor_meta,
    _pairwise_corr,
    _shared_prefix_label,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml-hub"])


def _resolve_factor_ids(
    factor_ids: str | None,
    meta: dict[str, _FactorMeta],
    history_keys: set[str],
) -> list[str] | None:
    """Parse a comma-separated ``factor_ids`` query into resolved history slugs.

    Each entry may be the public ``factor_id`` or the raw slug. Entries are
    stripped, empties dropped, deduped (order-preserving), mapped id→slug via
    ``meta``, and finally intersected with the slugs that actually have cached
    history.

    Args:
        factor_ids: raw comma-separated query value (or ``None``).
        meta: ``{slug: _FactorMeta}`` lookup for id→slug resolution.
        history_keys: the set of slugs present in the loaded history index.

    Returns:
        The resolved slugs in request order, or ``None`` when ``factor_ids`` is
        absent or only whitespace (caller should keep default behaviour).
    """
    if factor_ids is None:
        return None
    id_to_slug = {m.factor_id: slug for slug, m in meta.items()}
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in factor_ids.split(","):
        token = raw.strip()
        if not token:
            continue
        slug = token if token in history_keys else id_to_slug.get(token)
        if slug is None or slug not in history_keys or slug in seen:
            continue
        seen.add(slug)
        resolved.append(slug)
    if not resolved and not factor_ids.strip():
        return None
    return resolved


# The embedding eigendecomposition is O(n³); cap so a misuse can't pin a worker.
MAX_MAP_FACTORS: int = 500
MAP_TTL_SECONDS: int = 900  # universe geometry shifts slowly; 15 min is comfy

# The neighbours scan is O(n) once the correlation matrix exists; same TTL as
# the map since both ride on the same slowly-shifting co-movement structure.
NEIGHBORS_TTL_SECONDS: int = 900


# --- response schemas -------------------------------------------------------


class FactorPoint(BaseModel):
    """One prediction-market factor placed in the 2-D galaxy."""

    factor_id: str
    name: str
    theme: str
    x: float = Field(..., description="MDS / t-SNE coordinate 1.")
    y: float = Field(..., description="MDS / t-SNE coordinate 2.")
    cluster: int = Field(..., description="Cluster index (color group).")
    vol: float = Field(..., ge=0.0, description="Δlogit daily volatility (point size).")
    n_obs: int = Field(..., description="Daily observations backing this factor.")


class ClusterSummary(BaseModel):
    cluster: int
    label: str
    n_factors: int
    avg_intra_corr: float
    leader: str | None = Field(None, description="Granger-lite leading factor, if any.")
    leader_lag: int | None = None


class FactorMapResponse(BaseModel):
    n_factors: int
    n_clusters: int
    method: Literal["mds", "tsne"]
    min_corr: float
    theme: str | None
    # Kruskal stress-1 of the embedding (MDS only): <0.1 excellent, <0.2 fair.
    # None for t-SNE, which optimises a different (KL) objective.
    stress: float | None
    points: list[FactorPoint]
    clusters: list[ClusterSummary]
    selected_factor_ids: list[str] | None = Field(
        None,
        description="Slugs actually used when restricted via ?factor_ids; null otherwise.",
    )
    degraded_mode: bool = False
    reason: str | None = None


class NeighborItem(BaseModel):
    """One peer of the target factor, with its signed Δlogit correlation."""

    factor_id: str
    name: str
    theme: str
    corr: float = Field(..., description="Signed Pearson corr on Δlogit vs the target.")
    latest_price: float | None = Field(None, description="Last cached probability, if any.")
    n_obs: int = Field(..., description="Δlogit observations backing the peer.")


class FactorNeighborsResponse(BaseModel):
    """Hedge / de-dup view of one factor's closest co-movers and opposites."""

    factor_id: str
    name: str
    theme: str
    latest_price: float | None
    n_obs: int
    vol: float = Field(..., ge=0.0, description="Target factor Δlogit daily volatility.")
    # Most positively-correlated peers — redundant exposures to drop when
    # de-duping a basket.
    duplicates: list[NeighborItem]
    # Most negatively-correlated peers — natural hedges against the target.
    hedges: list[NeighborItem]
    degraded_mode: bool = False
    reason: str | None = None


# --- embedding --------------------------------------------------------------


def _embed_mds(dist: np.ndarray) -> tuple[np.ndarray, float]:
    """Classical (Torgerson) MDS on a precomputed distance matrix.

    Returns 2-D coordinates and the normalized Kruskal stress-1, a standard
    goodness-of-fit (lower is better; <0.1 excellent, 0.1-0.2 fair).

    We implement classical MDS / PCoA directly in numpy rather than calling
    ``sklearn.manifold.MDS``: it is deterministic (no SMACOF random restarts),
    fast (a single eigendecomposition), warning-free, and — importantly for a
    repo edited by several sessions on different sklearn versions — immune to
    the in-flight ``dissimilarity``→``metric`` constructor-API migration.

    Method: double-centre the squared-distance matrix
    ``B = -½ · J D² J`` with ``J = I - 11ᵀ/n``; the top-2 positive
    eigenpairs of ``B`` give the embedding ``X = V₂ · √Λ₂``.
    """
    n = dist.shape[0]
    d2 = dist**2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    # Symmetric eigendecomposition; take the two largest eigenvalues.
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1][:2]
    top_vals = np.clip(eigvals[order], a_min=0.0, a_max=None)
    coords = eigvecs[:, order] * np.sqrt(top_vals)

    # Kruskal stress-1 = sqrt(Σ(d_ij - d̂_ij)² / Σ d_ij²) over the upper triangle,
    # where d̂ is the Euclidean distance in the 2-D embedding.
    iu = np.triu_indices(n, k=1)
    d_orig = dist[iu]
    diff = coords[iu[0]] - coords[iu[1]]
    d_emb = np.sqrt((diff**2).sum(axis=1))
    denom = float(np.sum(d_orig**2))
    stress1 = float(np.sqrt(np.sum((d_orig - d_emb) ** 2) / denom)) if denom > 0 else 0.0
    return coords, stress1


def _embed_tsne(dist: np.ndarray) -> tuple[np.ndarray, float | None]:
    """t-SNE on a precomputed distance matrix → ``(coords, None)``.

    Perplexity is clamped to ``< n_samples`` (sklearn requirement). t-SNE has
    no stress analogue we report, so the second element is ``None``.
    """
    from sklearn.manifold import TSNE

    n = dist.shape[0]
    perplexity = float(min(30, max(5, (n - 1) // 3)))
    model = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        random_state=0,
    )
    return model.fit_transform(dist), None


# --- main endpoint ----------------------------------------------------------


@router.get(
    "/factor-map",
    response_model=None,
    summary="2-D embedding of the prediction-market factor universe (Factor Galaxy).",
)
def factor_map(
    theme: Annotated[
        str | None,
        Query(description="Filter by factors.yml theme tag (e.g. 'politics')."),
    ] = None,
    min_corr: Annotated[
        float,
        Query(ge=0.0, le=0.99, description="|corr| threshold cutting the dendrogram."),
    ] = 0.5,
    method: Annotated[
        Literal["mds", "tsne"],
        Query(description="Embedding: 'mds' (honest distances) or 'tsne' (prettier)."),
    ] = "mds",
    factor_ids: Annotated[
        str | None,
        Query(
            description="Comma-separated factor ids/slugs to restrict analysis to (overrides theme)."
        ),
    ] = None,
    request: Request = None,  # type: ignore[assignment]
) -> FactorMapResponse:
    """Embed the factor universe into 2-D and colour it by co-movement cluster.

    Pipeline: cached daily-probability history → Δlogit returns → pairwise
    Pearson correlation → distance ``d = 1-|corr|`` → MDS/t-SNE embedding +
    agglomerative clustering for colours. Point size encodes each factor's
    Δlogit volatility; clusters report their Granger-lite leader.

    Cached for ``MAP_TTL_SECONDS`` via the shared L1/L2 TERMINAL_CACHE keyed on
    ``(theme, min_corr, method)``. Returns ``degraded_mode=true`` (empty map)
    when the factor-history cache is cold, mirroring ``/terminal/factor-clusters``.
    """
    from pfm import terminal as _term_mod  # circular-safe lazy import

    cache_key = f"ml_factor_map::{theme or '*'}::{min_corr:.3f}::{method}::{factor_ids or '*'}"
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return FactorMapResponse.model_validate(cached)

    meta = _load_factor_meta()
    history = _load_cached_history()
    if not history:
        return FactorMapResponse(
            n_factors=0,
            n_clusters=0,
            method=method,
            min_corr=min_corr,
            theme=theme,
            stress=None,
            points=[],
            clusters=[],
            selected_factor_ids=None,
            degraded_mode=True,
            reason=(
                "Factor-history cache is empty (run the strat7 batch job or wait "
                "for the in-process prewarm to populate it)."
            ),
        )

    selected = _resolve_factor_ids(factor_ids, meta, set(history.keys()))
    if selected is not None:
        # An explicit factor set overrides the theme filter entirely.
        if len(selected) < 3:
            raise HTTPException(
                status_code=422,
                detail="Pick at least 3 factors with history.",
            )
        history = {slug: history[slug] for slug in selected}
    elif theme is not None:
        wanted = {slug for slug, m in meta.items() if m.theme == theme}
        if not wanted:
            raise HTTPException(status_code=404, detail=f"No factors for theme={theme!r}.")
        history = {k: v for k, v in history.items() if k in wanted}
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"No cached history for any theme={theme!r} factor.",
            )

    if len(history) > MAX_MAP_FACTORS:
        ranked = sorted(history.items(), key=lambda kv: -len(kv[1]))[:MAX_MAP_FACTORS]
        history = dict(ranked)

    returns = _build_returns_matrix(history)
    if returns.empty or returns.shape[1] < 3:
        if selected is not None:
            raise HTTPException(
                status_code=422,
                detail="Pick at least 3 factors with history.",
            )
        raise HTTPException(
            status_code=422,
            detail="Need at least 3 factors with sufficient history to build a map.",
        )

    corr = _pairwise_corr(returns)
    slugs = list(corr.index)

    # Distance matrix: sign-agnostic so economic mirrors sit together.
    dist = (1.0 - corr.abs().to_numpy()).clip(min=0.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0

    if method == "tsne":
        coords, stress = _embed_tsne(dist)
    else:
        coords, stress = _embed_mds(dist)

    clusters = _cluster_from_corr(corr, min_corr=min_corr)
    slug_to_cluster: dict[str, int] = {}
    for cidx, members in clusters.items():
        for s in members:
            slug_to_cluster[s] = cidx

    # Per-factor Δlogit volatility → point size.
    vol = returns.std(axis=0)

    points: list[FactorPoint] = []
    for i, slug in enumerate(slugs):
        m = meta.get(slug)
        points.append(
            FactorPoint(
                factor_id=(m.factor_id if m else slug),
                name=(m.name if m else slug),
                theme=(m.theme if m else "other"),
                x=round(float(coords[i, 0]), 4),
                y=round(float(coords[i, 1]), 4),
                cluster=int(slug_to_cluster.get(slug, -1)),
                vol=round(float(vol.get(slug, 0.0) or 0.0), 5),
                n_obs=int(returns[slug].dropna().shape[0]),
            )
        )

    cluster_summaries: list[ClusterSummary] = []
    for cidx, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        names = [(meta[s].name if s in meta else s) for s in members]
        sub = corr.loc[members, members].abs().to_numpy()
        iu = np.triu_indices_from(sub, k=1)
        avg_intra = float(np.mean(sub[iu])) if sub[iu].size else 1.0
        leader_tup = _detect_leader(returns, members)
        leader_id = leader_lag = None
        if leader_tup is not None:
            lslug, lag_k, _ = leader_tup
            leader_id = meta[lslug].factor_id if lslug in meta else lslug
            leader_lag = int(lag_k)
        cluster_summaries.append(
            ClusterSummary(
                cluster=int(cidx),
                label=_shared_prefix_label(names),
                n_factors=len(members),
                avg_intra_corr=round(avg_intra, 4),
                leader=leader_id,
                leader_lag=leader_lag,
            )
        )

    resp = FactorMapResponse(
        n_factors=len(slugs),
        n_clusters=len(clusters),
        method=method,
        min_corr=min_corr,
        theme=theme,
        stress=(round(stress, 4) if stress is not None else None),
        points=points,
        clusters=cluster_summaries,
        selected_factor_ids=selected,
    )
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), MAP_TTL_SECONDS)
    return resp


# --- hedge / de-dup finder --------------------------------------------------


@router.get(
    "/factor-neighbors",
    response_model=None,
    summary="Top co-movers (de-dup candidates) and opposites (hedges) for a factor.",
)
def factor_neighbors(
    factor_id: Annotated[
        str,
        Query(description="Slug or factor_id of the target factor."),
    ],
    k: Annotated[
        int,
        Query(ge=1, le=25, description="How many duplicates / hedges to return each side."),
    ] = 8,
    min_obs: Annotated[
        int,
        Query(ge=1, description="Minimum Δlogit observations for the target to qualify."),
    ] = 30,
    request: Request = None,  # type: ignore[assignment]
) -> FactorNeighborsResponse:
    """Surface a factor's most positively- and negatively-correlated peers.

    Pipeline mirrors ``/ml/factor-map``: cached daily-probability history →
    Δlogit returns → pairwise Pearson correlation (reusing ``_pairwise_corr``).
    For the requested factor we split its peer row into the top-``k`` most
    *positively* correlated peers (``duplicates`` — redundant exposures a basket
    could drop) and the top-``k`` most *negatively* correlated peers (``hedges``
    — natural offsets). ``corr`` is the signed Pearson on Δlogit; this is purely
    descriptive co-movement, not a forecast.

    Returns HTTP 404 when ``factor_id`` is not in the correlation index, and
    ``degraded_mode=true`` (empty lists) when the factor-history cache is cold,
    mirroring ``/ml/factor-map``. Cached for ``NEIGHBORS_TTL_SECONDS`` via the
    shared L1/L2 TERMINAL_CACHE keyed on ``(factor_id, k, min_obs)``.
    """
    from pfm import terminal as _term_mod  # circular-safe lazy import

    cache_key = f"ml_factor_neighbors::{factor_id}::{k}::{min_obs}"
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return FactorNeighborsResponse.model_validate(cached)

    meta = _load_factor_meta()
    history = _load_cached_history()
    if not history:
        return FactorNeighborsResponse(
            factor_id=factor_id,
            name=factor_id,
            theme="other",
            latest_price=None,
            n_obs=0,
            vol=0.0,
            duplicates=[],
            hedges=[],
            degraded_mode=True,
            reason=(
                "Factor-history cache is empty (run the strat7 batch job or wait "
                "for the in-process prewarm to populate it)."
            ),
        )

    if len(history) > MAX_MAP_FACTORS:
        ranked = sorted(history.items(), key=lambda kv: -len(kv[1]))[:MAX_MAP_FACTORS]
        history = dict(ranked)

    # Map factor_id -> slug so callers can pass either the public id or the slug.
    id_to_slug = {m.factor_id: slug for slug, m in meta.items()}
    target_slug = factor_id if factor_id in history else id_to_slug.get(factor_id)
    if target_slug is None or target_slug not in history:
        raise HTTPException(
            status_code=404,
            detail=f"factor_id={factor_id!r} not found in cached factor history.",
        )

    returns = _build_returns_matrix(history)
    if returns.empty or target_slug not in returns.columns:
        raise HTTPException(
            status_code=404,
            detail=f"factor_id={factor_id!r} has no usable Δlogit returns.",
        )

    target_n_obs = int(returns[target_slug].dropna().shape[0])
    if target_n_obs < min_obs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"factor_id={factor_id!r} has only {target_n_obs} Δlogit obs (< min_obs={min_obs})."
            ),
        )

    corr = _pairwise_corr(returns)
    if target_slug not in corr.index:
        raise HTTPException(
            status_code=404,
            detail=f"factor_id={factor_id!r} not present in the correlation index.",
        )

    def _latest_price(slug: str) -> float | None:
        ser = history.get(slug)
        if ser is None or ser.empty:
            return None
        return round(float(ser.iloc[-1]), 4)

    def _to_item(slug: str, c: float) -> NeighborItem:
        m = meta.get(slug)
        return NeighborItem(
            factor_id=(m.factor_id if m else slug),
            name=(m.name if m else slug),
            theme=(m.theme if m else "other"),
            corr=round(float(c), 4),
            latest_price=_latest_price(slug),
            n_obs=int(returns[slug].dropna().shape[0]),
        )

    # Signed peer row (drop self). Sort descending → most-positive first for
    # duplicates; ascending → most-negative first for hedges.
    peer_corr = corr[target_slug].drop(labels=[target_slug])
    pos_sorted = peer_corr.sort_values(ascending=False)
    duplicates = [_to_item(slug, c) for slug, c in pos_sorted.head(k).items()]
    neg_sorted = peer_corr.sort_values(ascending=True)
    hedges = [_to_item(slug, c) for slug, c in neg_sorted.head(k).items()]

    m = meta.get(target_slug)
    vol = float(returns[target_slug].std() or 0.0)
    resp = FactorNeighborsResponse(
        factor_id=(m.factor_id if m else target_slug),
        name=(m.name if m else target_slug),
        theme=(m.theme if m else "other"),
        latest_price=_latest_price(target_slug),
        n_obs=target_n_obs,
        vol=round(vol, 5),
        duplicates=duplicates,
        hedges=hedges,
    )
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), NEIGHBORS_TTL_SECONDS)
    return resp


__all__ = [
    "ClusterSummary",
    "FactorMapResponse",
    "FactorNeighborsResponse",
    "FactorPoint",
    "NeighborItem",
    "factor_map",
    "factor_neighbors",
    "router",
]
