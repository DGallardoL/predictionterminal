"""ML Hub — latent-factor decomposition of the prediction-market universe.

This module adds a **Latent Factor Model** to the ML Hub. Where
``/ml/factor-map`` answers *"what is shaped like what?"* via a distance
embedding, this endpoint answers *"how many independent drivers move the whole
universe, and how exposed is each contract to them?"* by running a principal
component analysis (PCA) on the standardized Δlogit-returns matrix.

What it returns
---------------
- **Variance explained** per principal component (and cumulative), so the user
  can see how low-rank the universe actually is.
- **Loadings** of every contract on the top-``k`` components — the exposure of
  each prediction market to each latent driver.
- **Interpretable labels**: for each component we list the contracts with the
  largest +/- loadings, so PC1 reads as e.g. "politics bloc vs macro bloc".
- **Idiosyncratic residual signal**: reconstruct each contract from the top-``k``
  PCs, take ``residual = actual − reconstructed``, and report the latest
  *standardized* residual ``z``. A large ``|z|`` means the contract is moving
  out of line with the common factors today — a relative-value signal that is
  richer than any single pairwise spread because it nets out *all* common
  structure at once.

Why this is honest
------------------
This is **descriptive dimensionality reduction, not forecasting**. PCA is fit
on the historical co-movement matrix; we make no claim that today's residual
predicts tomorrow's return. The residual ``z`` is a *relative-value* read
("rich/cheap vs the common factors right now"), and we say so. No forward
target is ever regressed — consistent with this project's anti-overfit stance.

Routing
-------
Owns its own :class:`fastapi.APIRouter`. Wire in ``main.py`` with::

    from pfm.ml_latent_factors_router import router as ml_latent_factors_router
    app.include_router(ml_latent_factors_router)
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

# Reuse the same proven history/returns primitives that back the Factor Galaxy
# and the Terminal "Factor Clusters" panel, so all three stay consistent.
from pfm.terminal.factor_clusters import (
    _build_returns_matrix,
    _FactorMeta,
    _load_cached_history,
    _load_factor_meta,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml-hub"])


def _resolve_factor_ids(
    factor_ids: str | None,
    meta: dict[str, _FactorMeta],
    history_keys: set[str],
) -> list[str] | None:
    """Parse a comma-separated ``factor_ids`` query into resolved history slugs.

    Each entry may be the public ``factor_id`` or the raw slug; entries are
    stripped, empties dropped, deduped (order-preserving), mapped id→slug via
    ``meta``, and intersected with the slugs present in the loaded history.
    Returns ``None`` when ``factor_ids`` is absent or only whitespace.
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


# PCA is O(n·m²); cap factors so a misuse can't pin a worker.
MAX_LATENT_FACTORS: int = 500
LATENT_TTL_SECONDS: int = 900  # universe structure shifts slowly; 15 min is comfy
TOP_LOADING_NAMES: int = 5  # names listed per pole when labelling a component
MAX_RESIDUAL_ITEMS: int = 40  # cap the RV-signal table to the most extreme |z|


# --- response schemas -------------------------------------------------------


class ComponentSummary(BaseModel):
    """One principal component (latent driver) of the factor universe."""

    pc: int = Field(..., description="1-based component index (PC1 is the largest).")
    explained_var: float = Field(..., ge=0.0, le=1.0, description="Variance-explained ratio.")
    cum_var: float = Field(..., ge=0.0, le=1.0, description="Cumulative variance explained.")
    top_positive: list[str] = Field(..., description="Contract names with the largest + loading.")
    top_negative: list[str] = Field(..., description="Contract names with the largest - loading.")


class ResidualItem(BaseModel):
    """A contract's idiosyncratic relative-value signal vs the latent factors."""

    factor_id: str
    name: str
    theme: str
    resid_z: float = Field(..., description="Latest standardized residual (rich/cheap RV signal).")
    loadings: list[float] = Field(..., description="Loadings on the top-k PCs (length k).")


class LatentFactorsResponse(BaseModel):
    n_factors: int
    n_obs: int
    k: int = Field(..., description="Number of principal components retained.")
    components: list[ComponentSummary]
    residuals: list[ResidualItem]
    selected_factor_ids: list[str] | None = Field(
        None,
        description="Slugs actually used when restricted via ?factor_ids; null otherwise.",
    )
    degraded_mode: bool = False
    reason: str | None = None


# --- main endpoint ----------------------------------------------------------


@router.get(
    "/latent-factors",
    response_model=None,
    summary="PCA latent-factor decomposition of the prediction-market universe.",
)
def latent_factors(
    theme: Annotated[
        str | None,
        Query(description="Filter by factors.yml theme tag (e.g. 'politics')."),
    ] = None,
    k: Annotated[
        int,
        Query(ge=1, le=20, description="Number of principal components to retain."),
    ] = 5,
    factor_ids: Annotated[
        str | None,
        Query(
            description="Comma-separated factor ids/slugs to restrict analysis to (overrides theme)."
        ),
    ] = None,
    request: Request = None,  # type: ignore[assignment]
) -> LatentFactorsResponse:
    """Decompose the Δlogit-returns matrix into latent factors via PCA.

    Pipeline: cached daily-probability history → Δlogit returns → z-score each
    factor's column → ``PCA(n_components=k)``. ``k`` is capped at
    ``min(k, n_factors-1, n_obs-1)``. We report per-component variance
    explained, every contract's loadings on the top-``k`` PCs, an interpretable
    label per component (its largest +/- loading contracts), and each
    contract's latest standardized reconstruction residual — a relative-value
    "rich/cheap vs the common factors" signal, ranked by ``|z|``.

    Cached for ``LATENT_TTL_SECONDS`` via the shared L1/L2 TERMINAL_CACHE keyed
    on ``(theme, k)``. Returns ``degraded_mode=true`` (empty payload) when the
    factor-history cache is cold, mirroring ``/ml/factor-map``.
    """
    from pfm import terminal as _term_mod  # circular-safe lazy import

    cache_key = f"ml_latent_factors::{theme or '*'}::{k}::{factor_ids or '*'}"
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return LatentFactorsResponse.model_validate(cached)

    meta = _load_factor_meta()
    history = _load_cached_history()
    if not history:
        return LatentFactorsResponse(
            n_factors=0,
            n_obs=0,
            k=0,
            components=[],
            residuals=[],
            selected_factor_ids=None,
            degraded_mode=True,
            reason=(
                "Factor-history cache is empty (run the strat7 batch job or wait "
                "for the in-process prewarm to populate it)."
            ),
        )

    selected = _resolve_factor_ids(factor_ids, meta, set(history.keys()))
    if selected is not None:
        # An explicit factor set overrides the theme filter; PCA needs >= k+1.
        if len(selected) < k + 1:
            raise HTTPException(
                status_code=422,
                detail=f"Pick at least {k + 1} factors with history for k={k} components.",
            )
        history = {slug: history[slug] for slug in selected}
    elif theme is not None:
        wanted = {slug for slug, m in meta.items() if m.theme == theme}
        if not wanted:
            raise HTTPException(status_code=404, detail=f"No factors for theme={theme!r}.")
        history = {key: val for key, val in history.items() if key in wanted}
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"No cached history for any theme={theme!r} factor.",
            )

    if len(history) > MAX_LATENT_FACTORS:
        ranked = sorted(history.items(), key=lambda kv: -len(kv[1]))[:MAX_LATENT_FACTORS]
        history = dict(ranked)

    returns = _build_returns_matrix(history)
    if returns.empty or returns.shape[1] < 3:
        raise HTTPException(
            status_code=422,
            detail="Need at least 3 factors with sufficient history to fit a latent model.",
        )

    # Align on common dates and fill any residual pairwise gaps with 0 (the
    # mean of a centred return) so PCA sees a dense rectangular matrix.
    returns = returns.sort_index().fillna(0.0)
    slugs = list(returns.columns)
    raw = returns.to_numpy(dtype=float)
    n_obs = raw.shape[0]

    # Standardize each column (z-score). Constant columns (zero variance) carry
    # no co-movement information and would divide by zero — drop them.
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    keep = std > 1e-12
    if int(keep.sum()) < 3:
        raise HTTPException(
            status_code=422,
            detail="Need at least 3 non-constant factors to fit a latent model.",
        )
    slugs = [s for s, kept in zip(slugs, keep, strict=True) if kept]
    z = (raw[:, keep] - mean[keep]) / std[keep]
    n_factors = z.shape[1]

    # Cap k: a PCA can recover at most min(n_obs, n_features) - 1 informative
    # components after centring.
    k_eff = max(1, min(k, n_factors - 1, n_obs - 1))

    from sklearn.decomposition import PCA

    pca = PCA(n_components=k_eff, random_state=0)
    scores = pca.fit_transform(z)  # [n_obs x k]
    components = pca.components_  # [k x n_factors]
    evr = pca.explained_variance_ratio_

    # Loadings = components_ transposed: row per factor, column per PC.
    loadings = components.T  # [n_factors x k]

    # --- component summaries (interpretable labels) -------------------------
    name_of = {s: (meta[s].name if s in meta else s) for s in slugs}
    component_summaries: list[ComponentSummary] = []
    cum = 0.0
    for j in range(k_eff):
        col = loadings[:, j]
        cum += float(evr[j])
        order = np.argsort(col)  # ascending: most negative first
        pos_idx = order[::-1][:TOP_LOADING_NAMES]
        neg_idx = order[:TOP_LOADING_NAMES]
        top_positive = [name_of[slugs[i]] for i in pos_idx if col[i] > 0]
        top_negative = [name_of[slugs[i]] for i in neg_idx if col[i] < 0]
        component_summaries.append(
            ComponentSummary(
                pc=j + 1,
                explained_var=round(float(evr[j]), 4),
                cum_var=round(min(cum, 1.0), 4),
                top_positive=top_positive,
                top_negative=top_negative,
            )
        )

    # --- idiosyncratic residual RV signal -----------------------------------
    # Reconstruct each factor from the top-k PCs; residual = actual - recon.
    recon = scores @ components  # [n_obs x n_factors], in z-space
    resid = z - recon
    resid_std = resid.std(axis=0)
    resid_std_safe = np.where(resid_std > 1e-12, resid_std, 1.0)
    latest_z = resid[-1, :] / resid_std_safe  # standardized latest residual

    rv_order = np.argsort(-np.abs(latest_z))
    residual_items: list[ResidualItem] = []
    for i in rv_order[:MAX_RESIDUAL_ITEMS]:
        slug = slugs[i]
        m = meta.get(slug)
        residual_items.append(
            ResidualItem(
                factor_id=(m.factor_id if m else slug),
                name=(m.name if m else slug),
                theme=(m.theme if m else "other"),
                resid_z=round(float(latest_z[i]), 4),
                loadings=[round(float(v), 4) for v in loadings[i, :]],
            )
        )

    resp = LatentFactorsResponse(
        n_factors=n_factors,
        n_obs=n_obs,
        k=k_eff,
        components=component_summaries,
        residuals=residual_items,
        selected_factor_ids=selected,
    )
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), LATENT_TTL_SECONDS)
    return resp


__all__ = [
    "ComponentSummary",
    "LatentFactorsResponse",
    "ResidualItem",
    "latent_factors",
    "router",
]
