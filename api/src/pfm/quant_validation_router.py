"""FastAPI router for the quant-validation endpoints.

Two endpoints, both POST and JSON-in/JSON-out:

* ``POST /quant/multitest/bh`` — Benjamini-Hochberg FDR over a vector of
  marginal p-values.
* ``POST /quant/quarterly-stability`` — 4-quarter Sharpe stability gate.

The router is **not** auto-mounted.  Add the following to ``main.py`` to
register it::

    from pfm.quant_validation_router import router as qv_router
    app.include_router(qv_router)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from pfm.multitest import benjamini_hochberg_fdr
from pfm.strategy_verdict import quarterly_stability_test

router = APIRouter(prefix="/quant", tags=["quant-validation"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class BHRequest(BaseModel):
    """Body of ``POST /quant/multitest/bh``."""

    p_values: list[float] = Field(
        ...,
        description="Marginal p-values to correct, each in [0, 1].",
    )
    alpha: float = Field(
        0.05,
        gt=0.0,
        lt=1.0,
        description="Target false-discovery rate (default 0.05).",
    )


class BHResponse(BaseModel):
    rejected_idx: list[int]
    q_values: list[float]
    n_significant: int


class QuarterlyStabilityRequest(BaseModel):
    """Body of ``POST /quant/quarterly-stability``."""

    quarterly_sharpes: list[float] = Field(
        ...,
        description="Per-quarter Sharpe ratios in chronological order.",
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        description="Minimum per-quarter Sharpe to count as 'positive'.",
    )


class QuarterlyStabilityResponse(BaseModel):
    n_quarters: int
    n_positive: int
    sign_flips: int
    passes_4q_gold: bool
    passes_4q_silver: bool
    tier_recommendation: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/multitest/bh", response_model=BHResponse)
def post_bh_multitest(body: BHRequest) -> dict[str, Any]:
    """Apply Benjamini-Hochberg FDR to the supplied p-values."""
    return benjamini_hochberg_fdr(body.p_values, alpha=body.alpha)


@router.post("/quarterly-stability", response_model=QuarterlyStabilityResponse)
def post_quarterly_stability(body: QuarterlyStabilityRequest) -> dict[str, Any]:
    """Score a strategy's per-quarter Sharpe record for tier promotion."""
    return quarterly_stability_test(
        body.quarterly_sharpes,
        threshold=body.threshold,
    )
