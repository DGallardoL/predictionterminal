"""ML Hub — Event Relationship Graph via vector autoregression (VAR).

The Factor Galaxy (``/ml/factor-map``) answers *which markets sit near each
other*. This module answers the next two questions, both grounded in a single
fitted **VAR(p)** over the same Δlogit return panel:

1. **Contemporaneous co-movement** — undirected edges between factors whose
   same-day returns are correlated (``|corr| ≥ edge_threshold``). "These two
   move at the same time."
2. **Directed lead-lag** — directed edges ``X → Y`` when a VAR Granger-causality
   test says past values of *X* help predict *Y* (p < 0.05). "X drives Y."

On top of the network we attach, per node, the VAR's **one-step-ahead implied
next price** and a **mispricing z-score** versus a flat (no-move) expectation.

Honest framing (read before trusting the mispricing)
----------------------------------------------------
This is a *descriptive network* plus a *clearly-caveated one-step forecast*. A
VAR on prediction-market Δlogits is weakly identified and prone to overfit — so
we ship a **walk-forward out-of-sample R²** of the VAR's one-step forecast
against a naive last-value/zero predictor on a held-out tail. The response
states plainly: the mispricing is only worth acting on when ``oos_r2 > 0``.
When the OOS R² is negative the VAR adds nothing the naive forecast didn't, and
the implied prices should be read as illustrative, not tradable.

Routing
-------
Owns its own :class:`fastapi.APIRouter`. Wire in ``main.py`` with::

    from pfm.ml_event_graph_router import router as ml_event_graph_router
    app.include_router(ml_event_graph_router)
"""

from __future__ import annotations

import logging
import warnings
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

# Reuse the proven factor-clusters pipeline (history → Δlogit → corr → clusters)
# and the ML-Hub MDS embedding so this graph's node layout agrees with the
# Factor Galaxy by construction.
from pfm.ml_hub_router import _embed_mds
from pfm.terminal.factor_clusters import (
    _build_returns_matrix,
    _cluster_from_corr,
    _load_cached_history,
    _load_factor_meta,
    _pairwise_corr,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml-hub"])

# VAR needs T >> n_vars · lags to be well-posed, so we cap the node count hard.
DEFAULT_MAX_NODES: int = 25
NODE_HARD_CAP: int = 40
DEFAULT_EDGE_THRESHOLD: float = 0.4
DEFAULT_MAXLAGS: int = 2
MAXLAGS_HARD_CAP: int = 3
GRANGER_ALPHA: float = 0.05
GRAPH_TTL_SECONDS: int = 600
CLIP_EPS: float = 0.01  # keep in step with factor_clusters._delta_logit


# --- response schemas -------------------------------------------------------


class GraphNode(BaseModel):
    """One prediction-market event placed in the relationship graph."""

    factor_id: str
    name: str
    theme: str
    x: float = Field(..., description="MDS coordinate 1 (shared with Factor Galaxy).")
    y: float = Field(..., description="MDS coordinate 2 (shared with Factor Galaxy).")
    community: int = Field(..., description="Co-movement community index (colour group).")
    centrality: int = Field(..., description="Out-degree of directed lead-lag edges.")
    market_price: float = Field(..., description="Latest observed probability.")
    model_price: float = Field(..., description="VAR one-step-ahead implied next probability.")
    mispricing_z: float = Field(
        ..., description="Standardized implied Δlogit move vs a flat (no-move) expectation."
    )


class GraphEdge(BaseModel):
    """A relationship between two events."""

    source: str
    target: str
    kind: Literal["comove", "lead"]
    weight: float = Field(..., description="|corr| for comove; -log10(p) for lead.")
    lag: int | None = Field(None, description="VAR lag order for 'lead' edges; None for comove.")


class EventGraphResponse(BaseModel):
    n_nodes: int
    lag_order: int
    n_comove_edges: int
    n_lead_edges: int
    # Walk-forward OOS R² of the VAR one-step forecast vs a naive predictor.
    # >0 means the VAR beats naive on the holdout; mispricing is only trustworthy then.
    oos_r2: float | None
    caveat: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    degraded_mode: bool = False
    reason: str | None = None


# --- VAR helpers ------------------------------------------------------------


def _select_lag_order(returns: pd.DataFrame, maxlags: int) -> int:
    """Pick the VAR lag order by AIC, capped at ``maxlags`` (≥1).

    Falls back to lag 1 when statsmodels cannot select an order (e.g. too few
    observations for the requested ``maxlags``).
    """
    from statsmodels.tsa.api import VAR

    if maxlags < 1:
        return 1
    try:
        sel = VAR(returns).select_order(maxlags=maxlags)
        order = int(sel.aic)
    except (ValueError, np.linalg.LinAlgError):
        return 1
    return max(1, min(order, maxlags))


def _fit_var(returns: pd.DataFrame, lag_order: int):  # statsmodels VARResults type
    """Fit ``VAR(lag_order)``; raise HTTP 422 if it is ill-posed."""
    from statsmodels.tsa.api import VAR

    try:
        # statsmodels can emit a benign ValueWarning when the panel is short
        # relative to the parameter count; we keep series long in practice, so
        # scope the filter narrowly to that one category here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
            return VAR(returns).fit(lag_order)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"VAR({lag_order}) could not be fit on the selected panel: {exc}",
        ) from exc


def _walk_forward_oos_r2(returns: pd.DataFrame, lag_order: int, holdout: int) -> float | None:
    """One-step walk-forward OOS R² of the VAR vs a naive last-value forecast.

    Refits the VAR on an expanding window and forecasts one step at each of the
    final ``holdout`` dates, scoring out-of-sample SSE against the naive
    predictor (here: the panel's training-window mean, ≈0 for Δlogits). Returns
    ``1 - SSE_var / SSE_naive`` pooled across all series, or ``None`` when there
    is not enough history to run even a single honest step.
    """
    from statsmodels.tsa.api import VAR

    n = len(returns)
    min_train = max(lag_order + 2, 5 * returns.shape[1])
    start = n - holdout
    if start < min_train or holdout < 2:
        return None

    sse_var = 0.0
    sse_naive = 0.0
    values = returns.to_numpy()
    for t in range(start, n):
        train = values[:t]
        actual = values[t]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                res = VAR(train).fit(lag_order)
                pred = res.forecast(train[-lag_order:], steps=1)[0]
        except (ValueError, np.linalg.LinAlgError):
            return None
        naive = train.mean(axis=0)  # flat / no-move baseline
        sse_var += float(np.sum((actual - pred) ** 2))
        sse_naive += float(np.sum((actual - naive) ** 2))

    if sse_naive <= 0:
        return None
    return 1.0 - sse_var / sse_naive


def _inv_logit(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# --- main endpoint ----------------------------------------------------------


@router.get(
    "/event-graph",
    response_model=None,
    summary="VAR-powered event relationship graph (co-movement + directed lead-lag).",
)
def event_graph(
    theme: Annotated[
        str | None,
        Query(description="Filter by factors.yml theme tag (e.g. 'politics')."),
    ] = None,
    max_nodes: Annotated[
        int,
        Query(ge=3, le=NODE_HARD_CAP, description="Cap on nodes (longest-history factors)."),
    ] = DEFAULT_MAX_NODES,
    edge_threshold: Annotated[
        float,
        Query(ge=0.0, le=0.99, description="|corr| cutoff for contemporaneous comove edges."),
    ] = DEFAULT_EDGE_THRESHOLD,
    maxlags: Annotated[
        int,
        Query(ge=1, le=MAXLAGS_HARD_CAP, description="Max VAR lag order (AIC selects ≤ this)."),
    ] = DEFAULT_MAXLAGS,
    request: Request = None,  # type: ignore[assignment]
) -> EventGraphResponse:
    """Build the event relationship graph from a single fitted VAR.

    Pipeline: cached daily-probability history → Δlogit returns → (cap to the
    ``max_nodes`` longest-history factors) → pairwise correlation for
    contemporaneous *comove* edges + MDS layout → ``VAR(p)`` (p chosen by AIC,
    capped at ``maxlags``) → Granger causality for directed *lead* edges →
    one-step forecast for per-node implied price and mispricing. A walk-forward
    OOS R² is attached so callers can judge whether the mispricing is tradable.

    Cached for ``GRAPH_TTL_SECONDS`` via the shared TERMINAL_CACHE keyed on
    ``(theme, max_nodes, edge_threshold, maxlags)``. Returns
    ``degraded_mode=true`` when the factor-history cache is cold, mirroring
    ``/terminal/factor-clusters`` and ``/ml/factor-map``.
    """
    from pfm import terminal as _term_mod  # circular-safe lazy import

    cache_key = f"ml_event_graph::{theme or '*'}::{max_nodes}::{edge_threshold:.3f}::{maxlags}"
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return EventGraphResponse.model_validate(cached)

    meta = _load_factor_meta()
    history = _load_cached_history()
    if not history:
        return EventGraphResponse(
            n_nodes=0,
            lag_order=0,
            n_comove_edges=0,
            n_lead_edges=0,
            oos_r2=None,
            caveat="Factor-history cache is empty; no graph could be built.",
            nodes=[],
            edges=[],
            degraded_mode=True,
            reason=(
                "Factor-history cache is empty (run the strat7 batch job or wait "
                "for the in-process prewarm to populate it)."
            ),
        )

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

    # Keep the VAR well-posed: cap to the longest-history factors.
    if len(history) > max_nodes:
        ranked = sorted(history.items(), key=lambda kv: -len(kv[1]))[:max_nodes]
        history = dict(ranked)

    returns = _build_returns_matrix(history)
    if returns.empty or returns.shape[1] < 3:
        raise HTTPException(
            status_code=422,
            detail="Need at least 3 factors with sufficient history to build the graph.",
        )

    # VAR cannot ingest NaNs / constant columns — align to a common dense panel.
    returns = returns.dropna(axis=0, how="any")
    nonconstant = [c for c in returns.columns if returns[c].std() > 0]
    returns = returns[nonconstant]
    if returns.shape[1] < 3 or returns.shape[0] < 5 * returns.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=(
                "Insufficient overlapping observations for a well-posed VAR "
                f"({returns.shape[0]} rows × {returns.shape[1]} factors)."
            ),
        )

    slugs = list(returns.columns)
    # Edges reference the same id space as nodes (factor_id), not raw slugs.
    fid = {s: (meta[s].factor_id if s in meta else s) for s in slugs}

    # --- contemporaneous comove edges + MDS layout --------------------------
    corr = _pairwise_corr(returns)
    corr = corr.loc[slugs, slugs]
    dist = (1.0 - corr.abs().to_numpy()).clip(min=0.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    coords, _ = _embed_mds(dist)

    comove_edges: list[GraphEdge] = []
    comove_pairs: set[tuple[int, int]] = set()
    n = len(slugs)
    for i in range(n):
        for jx in range(i + 1, n):
            c = abs(float(corr.iat[i, jx]))
            if c >= edge_threshold:
                comove_edges.append(
                    GraphEdge(
                        source=fid[slugs[i]],
                        target=fid[slugs[jx]],
                        kind="comove",
                        weight=round(c, 4),
                        lag=None,
                    )
                )
                comove_pairs.add((i, jx))

    # --- VAR fit + directed lead-lag (Granger) edges ------------------------
    lag_order = _select_lag_order(returns, maxlags)
    var_res = _fit_var(returns, lag_order)

    lead_edges: list[GraphEdge] = []
    out_degree = dict.fromkeys(slugs, 0)
    # Restrict the n² Granger sweep to ordered pairs that share a comove edge;
    # falls back to all ordered pairs only when the comove set is empty.
    if comove_pairs:
        candidate_pairs = [(i, j) for (i, j) in comove_pairs] + [(j, i) for (i, j) in comove_pairs]
    else:
        candidate_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    for i, j in candidate_pairs:
        causing, caused = slugs[i], slugs[j]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                test = var_res.test_causality(caused, causing, kind="f")
            pval = float(test.pvalue)
        except (ValueError, KeyError, np.linalg.LinAlgError):
            continue
        if pval < GRANGER_ALPHA:
            lead_edges.append(
                GraphEdge(
                    source=fid[causing],
                    target=fid[caused],
                    kind="lead",
                    weight=round(float(-np.log10(max(pval, 1e-12))), 4),
                    lag=lag_order,
                )
            )
            out_degree[causing] += 1

    # --- communities --------------------------------------------------------
    communities = _cluster_from_corr(corr, min_corr=edge_threshold)
    slug_to_comm: dict[str, int] = {}
    for cidx, members in communities.items():
        for s in members:
            slug_to_comm[s] = cidx

    # --- one-step forecast → implied price + mispricing ---------------------
    last_window = returns.to_numpy()[-lag_order:]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
            forecast = var_res.forecast(last_window, steps=1)[0]
    except (ValueError, np.linalg.LinAlgError):
        forecast = np.zeros(n)
    resid_std = returns.std(axis=0).to_numpy()

    holdout = max(2, min(20, len(returns) // 4))
    oos_r2 = _walk_forward_oos_r2(returns, lag_order, holdout)

    nodes: list[GraphNode] = []
    for i, slug in enumerate(slugs):
        m = meta.get(slug)
        market_price = float(history[slug].iloc[-1])
        market_price = min(max(market_price, CLIP_EPS), 1.0 - CLIP_EPS)
        cur_logit = float(np.log(market_price / (1.0 - market_price)))
        implied_logit = cur_logit + float(forecast[i])
        model_price = float(_inv_logit(implied_logit))
        sd = float(resid_std[i])
        misp_z = float(forecast[i] / sd) if sd > 0 else 0.0
        nodes.append(
            GraphNode(
                factor_id=(m.factor_id if m else slug),
                name=(m.name if m else slug),
                theme=(m.theme if m else "other"),
                x=round(float(coords[i, 0]), 4),
                y=round(float(coords[i, 1]), 4),
                community=int(slug_to_comm.get(slug, -1)),
                centrality=int(out_degree[slug]),
                market_price=round(market_price, 4),
                model_price=round(model_price, 4),
                mispricing_z=round(misp_z, 4),
            )
        )

    if oos_r2 is None:
        caveat = (
            "Walk-forward OOS R² could not be computed (too little holdout history); "
            "treat all model prices as illustrative, not tradable."
        )
    elif oos_r2 > 0:
        caveat = (
            f"VAR beats the naive forecast on the holdout (OOS R²={oos_r2:.3f}); the "
            "one-step mispricing is weakly informative but capacity- and cost-sensitive."
        )
    else:
        caveat = (
            f"VAR does NOT beat a naive no-move forecast on the holdout (OOS R²={oos_r2:.3f}); "
            "the mispricing is descriptive only — do NOT trade it."
        )

    resp = EventGraphResponse(
        n_nodes=n,
        lag_order=lag_order,
        n_comove_edges=len(comove_edges),
        n_lead_edges=len(lead_edges),
        oos_r2=(round(oos_r2, 4) if oos_r2 is not None else None),
        caveat=caveat,
        nodes=nodes,
        edges=comove_edges + lead_edges,
    )
    _term_mod.TERMINAL_CACHE.set(cache_key, resp.model_dump(), GRAPH_TTL_SECONDS)
    return resp


__all__ = ["EventGraphResponse", "GraphEdge", "GraphNode", "event_graph", "router"]
