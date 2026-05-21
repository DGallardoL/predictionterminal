"""ML Hub — relative-value **Mispricing Scanner** over the factor universe.

This module answers a different question from the Factor Galaxy
(``/ml/factor-map``): *given how a prediction-market factor normally co-moves
with its closest peers, is its current price abnormally rich or cheap relative
to what those peers imply right now?*

Honest framing (no return forecasting)
--------------------------------------
This is **descriptive cross-sectional residual analysis**, not a forecast.
Every anti-alpha in this project died forecasting forward returns. Here we make
no claim about where any price is going. We fit, *in-sample*, each factor's
Δlogit return against its most-correlated neighbours' contemporaneous Δlogit
returns, look at the residual, and report whether the *latest* residual is an
outlier (a large standardized residual). A large residual just says "today this
factor moved much more/less than its usual peer relationship explains" — a
relative-value flag for a human to investigate, not a tradeable signal.

Algorithm
---------
1.  **History → returns.** Reuse the proven factor-clusters pipeline:
    cached daily-probability history → Δlogit returns matrix → pairwise
    Pearson correlation (same primitives as ``/terminal/factor-clusters`` and
    ``/ml/factor-map``, so the three panels stay consistent by construction).
2.  **Neighbours.** For each target factor, take its top-``K`` peers by
    ``|corr|`` (excluding self), keeping only peers with ``|corr| >= min_corr``.
3.  **Contemporaneous OLS.** Fit ``target_Δlogit ~ 1 + neighbours_Δlogit`` over
    the joint dropna sample via ``numpy.linalg.lstsq`` (ordinary least squares;
    no HAC — we are not doing inference on the coefficients, only inspecting
    residuals). Residual std is the in-sample dispersion of the relationship.
4.  **Mispricing z-score.** ``z = latest_residual / residual_std``. Positive z
    means the factor's *last* Δlogit move was larger than peers implied — the
    contract drifted **rich** relative to its block; negative ⇒ **cheap**.
    ``|z| < Z_FAIR`` is reported as ``"fair"``.
5.  **Rank & return** the top-``limit`` factors by ``|z|``.

A note on the "expected price" approximation
--------------------------------------------
The residual lives in Δlogit space. We report the factor's actual
``latest_price`` and a ``direction`` label rather than reconstructing a synthetic
"fair price": inverting one day's residual back through the logit to a price is
only an approximation (it ignores the path and the clip at ε), and dressing it
up as a precise fair-value number would overstate the rigour. The honest,
auditable signal is the residual z plus the rich/cheap/fair direction.

Routing
-------
Owns its own :class:`fastapi.APIRouter`. Per project convention ``main.py`` is
left untouched — wire it in there with::

    from pfm.ml_mispricing_router import router as ml_mispricing_router
    app.include_router(ml_mispricing_router)
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

# Reuse the battle-tested factor-clusters primitives (history loading, Δlogit,
# pairwise correlation) so this scanner stays consistent with the Terminal
# clusters panel and the ML Hub factor map. Imported by value into this
# namespace: tests monkeypatch the loaders *here*, not on the source module.
from pfm.terminal.factor_clusters import (
    _build_returns_matrix,
    _delta_logit,  # noqa: F401  # re-exported for callers/tests that expect it here
    _FactorMeta,
    _load_cached_history,
    _load_factor_meta,
    _pairwise_corr,
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


# Pairwise correlation across the universe is O(n²); cap so a misuse can't pin
# a worker. Mirrors the factor-clusters / factor-map safety caps.
MAX_SCAN_FACTORS: int = 500
SCAN_TTL_SECONDS: int = 600  # residual structure shifts slowly; 10 min is fine

# Need a meaningful joint sample before a residual std is trustworthy.
MIN_FIT_OBS: int = 25
# |z| below this is reported as "fair" rather than rich/cheap.
Z_FAIR: float = 1.5
# Liquidity / clip gate: a probability this close to 0 or 1 lives in the
# clipped tail (ε=0.01), so its Δlogit moves are clipping noise, not signal.
# We skip such factors entirely. Symmetric upper bound is ``1 - this``.
MISPRICING_MIN_PRICE: float = 0.02
# Above this in-sample R² we flag the fit as ``suspect`` (overfit / spurious
# near-perfect fit) without dropping it — see the docstring.
SUSPECT_R2: float = 0.97


# --- response schemas -------------------------------------------------------


class MispricingItem(BaseModel):
    """One factor flagged by the relative-value residual scan."""

    factor_id: str
    name: str
    theme: str
    z_score: float = Field(..., description="Latest residual / residual std. >0 rich, <0 cheap.")
    direction: Literal["rich", "cheap", "fair"] = Field(
        ..., description="rich = moved more than peers imply; cheap = less; fair = within band."
    )
    r_squared: float = Field(..., ge=0.0, le=1.0, description="OLS fit quality vs neighbours.")
    neighbors: list[str] = Field(..., description="factor_ids used as regressors.")
    neighbor_corrs: list[float] = Field(
        ...,
        description="Signed Δlogit corr of each listed neighbour with the target, aligned to "
        "``neighbors``.",
    )
    n_obs: int = Field(..., description="Joint observations backing the fit.")
    latest_price: float = Field(..., description="Actual most-recent probability of the factor.")
    suspect: bool = Field(
        False,
        description="True when r_squared > SUSPECT_R2 (overfit / spurious near-perfect fit). "
        "Not dropped, but ranked after non-suspect items.",
    )


class MispricingResponse(BaseModel):
    n_factors: int = Field(..., description="Factors that produced a usable fit.")
    min_corr: float
    top_k: int
    items: list[MispricingItem]
    # When the upstream factor-history cache is missing the endpoint returns an
    # empty item list + ``degraded_mode=true`` so the UI can render an
    # explanatory empty state instead of a 503. Same pattern as
    # /terminal/factor-clusters and /ml/factor-map.
    degraded_mode: bool = False
    reason: str | None = None


# --- core scan --------------------------------------------------------------


def _fit_residual_z(target: np.ndarray, neighbors: np.ndarray) -> tuple[float, float, int] | None:
    """OLS of ``target ~ 1 + neighbors``; return ``(latest_z, r_squared, n_obs)``.

    Args:
        target: target factor Δlogit returns, shape ``(T,)``.
        neighbors: neighbour Δlogit returns, shape ``(T, K)``.

    Returns:
        ``(latest_z, r_squared, n_obs)`` where ``latest_z`` is the final
        residual standardized by the residual standard deviation, or ``None``
        when the joint sample is too small or degenerate (constant target /
        zero residual variance).
    """
    # Joint dropna across target + every neighbour column.
    stacked = np.column_stack([target, neighbors])
    mask = np.isfinite(stacked).all(axis=1)
    stacked = stacked[mask]
    n_obs = int(stacked.shape[0])
    if n_obs < MIN_FIT_OBS:
        return None

    y = stacked[:, 0]
    x = stacked[:, 1:]
    if float(np.std(y)) == 0.0:
        return None

    # Design matrix with intercept.
    design = np.column_stack([np.ones(n_obs), x])
    coef, _res, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    resid = y - fitted

    resid_std = float(np.std(resid, ddof=1)) if n_obs > 1 else 0.0
    if resid_std == 0.0:
        return None

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Clip for the response model bound; an in-sample OLS keeps R² in [0, 1] but
    # floating error can nudge it a hair outside.
    r_squared = float(np.clip(r_squared, 0.0, 1.0))

    latest_z = float(resid[-1] / resid_std)
    return latest_z, r_squared, n_obs


# --- main endpoint ----------------------------------------------------------


@router.get(
    "/mispricing",
    response_model=None,
    summary="Relative-value residual scan: factors abnormally rich/cheap vs their peers.",
)
def mispricing(
    min_corr: Annotated[
        float,
        Query(ge=0.0, le=0.99, description="Minimum |corr| for a factor to count as a neighbour."),
    ] = 0.3,
    top_k: Annotated[
        int,
        Query(ge=1, le=20, alias="k", description="Number of neighbours to regress against."),
    ] = 5,
    limit: Annotated[
        int,
        Query(ge=1, le=200, description="Return the top-N factors by |z|."),
    ] = 30,
    factor_ids: Annotated[
        str | None,
        Query(
            description="Comma-separated factor ids/slugs to restrict the scan to (target side)."
        ),
    ] = None,
    request: Request = None,  # type: ignore[assignment]
) -> MispricingResponse:
    """Flag factors whose latest Δlogit move is an outlier vs their peer block.

    For each factor we fit (in-sample, contemporaneously) its Δlogit returns on
    its top-``k`` most-correlated neighbours (``|corr| >= min_corr``) and report
    the standardized *latest* residual as a **mispricing z-score**. Large ``|z|``
    means today the contract moved much more (``rich``) or less (``cheap``) than
    its usual peer relationship explains — a relative-value flag, **not** a
    forecast. See the module docstring for why this is honest.

    Cached for ``SCAN_TTL_SECONDS`` via the shared L1/L2 TERMINAL_CACHE keyed on
    ``(min_corr, top_k, limit)``. Returns ``degraded_mode=true`` (empty items)
    when the factor-history cache is cold, mirroring ``/ml/factor-map``.
    """
    from pfm import terminal as _term_mod  # circular-safe lazy import

    cache_key = f"ml_mispricing::{min_corr:.3f}::{top_k}::{limit}::{factor_ids or '*'}"
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return MispricingResponse.model_validate(cached)

    meta = _load_factor_meta()
    history = _load_cached_history()
    if not history:
        return MispricingResponse(
            n_factors=0,
            min_corr=min_corr,
            top_k=top_k,
            items=[],
            degraded_mode=True,
            reason=(
                "Factor-history cache is empty (run the strat7 batch job or wait "
                "for the in-process prewarm to populate it)."
            ),
        )

    if len(history) > MAX_SCAN_FACTORS:
        ranked = sorted(history.items(), key=lambda kv: -len(kv[1]))[:MAX_SCAN_FACTORS]
        history = dict(ranked)

    returns = _build_returns_matrix(history)
    if returns.empty or returns.shape[1] < 3:
        raise HTTPException(
            status_code=422,
            detail="Need at least 3 factors with sufficient history to scan for mispricing.",
        )

    corr = _pairwise_corr(returns)
    slugs = list(corr.index)

    # When an explicit factor set is given, only score those as *targets*;
    # neighbours still come from the full correlation universe.
    selected = _resolve_factor_ids(factor_ids, meta, set(corr.index))
    target_slugs = [s for s in slugs if s in set(selected)] if selected is not None else slugs

    items: list[MispricingItem] = []
    for slug in target_slugs:
        # Liquidity / clip gate: probabilities in the clipped tail produce
        # Δlogit moves that are clipping noise, not mispricing — skip them.
        latest_price = float(history[slug].iloc[-1])
        if latest_price < MISPRICING_MIN_PRICE or latest_price > 1.0 - MISPRICING_MIN_PRICE:
            continue

        # Rank candidate neighbours by |corr|, drop self, keep those over floor.
        abs_peer_corr = corr[slug].drop(labels=[slug]).abs()
        abs_peer_corr = abs_peer_corr[abs_peer_corr >= min_corr].sort_values(ascending=False)
        if abs_peer_corr.empty:
            continue
        peers = list(abs_peer_corr.index[:top_k])

        target_ret = returns[slug].to_numpy(dtype=float)
        neighbor_ret = returns[peers].to_numpy(dtype=float)
        fit = _fit_residual_z(target_ret, neighbor_ret)
        if fit is None:
            continue
        z_score, r_squared, n_obs = fit

        # Degrees-of-freedom guard: a k-regressor OLS (plus intercept) needs
        # enough joint observations or the residual std is unreliable.
        k_reg = len(peers)
        if n_obs < MIN_FIT_OBS + 3 * k_reg:
            continue

        # Signed corr of each listed neighbour with the target, aligned to peers.
        neighbor_corrs = [round(float(corr.loc[slug, p]), 4) for p in peers]
        suspect = bool(r_squared > SUSPECT_R2)

        if z_score > Z_FAIR:
            direction: Literal["rich", "cheap", "fair"] = "rich"
        elif z_score < -Z_FAIR:
            direction = "cheap"
        else:
            direction = "fair"

        m = meta.get(slug)
        items.append(
            MispricingItem(
                factor_id=(m.factor_id if m else slug),
                name=(m.name if m else slug),
                theme=(m.theme if m else "other"),
                z_score=round(z_score, 3),
                direction=direction,
                r_squared=round(r_squared, 4),
                neighbors=[(meta[p].factor_id if p in meta else p) for p in peers],
                neighbor_corrs=neighbor_corrs,
                n_obs=n_obs,
                latest_price=round(latest_price, 4),
                suspect=suspect,
            )
        )

    # Rank non-suspect items first, then by absolute mispricing, and keep top-N.
    items.sort(key=lambda it: (it.suspect, -abs(it.z_score)))
    items = items[:limit]

    resp = MispricingResponse(
        n_factors=len(items),
        min_corr=min_corr,
        top_k=top_k,
        items=items,
    )
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), SCAN_TTL_SECONDS)
    return resp


__all__ = ["MispricingItem", "MispricingResponse", "mispricing", "router"]
