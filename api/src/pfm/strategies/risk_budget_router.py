"""``GET /strategies/risk-budget`` — capital allocation across deployables (W12-20).

Wires the curated deployable enumeration from
:mod:`pfm.strategies.deployable_router` into a risk-parity sizing engine
with tier caps and a concentration limit:

1. Pull the deployable list (same logic / fallback as W11-24).
2. Estimate per-strategy daily volatility from whatever JSON fields exist
   (``vol``, ``sigma``, ``daily_vol``, ``ann_vol``), falling back to a tier-
   conditional default when none is present.
3. Risk-parity raw weights ``w_i ∝ 1 / σ_i``.
4. Apply tier caps: ``A_GOLD ≤ 25%``, ``A_STRUCTURAL ≤ 20%``,
   ``B_VALIDATED ≤ 10%``.
5. Apply a hard 30% concentration limit on any single strategy.
6. Renormalize the remaining headroom **back into** under-capped strategies
   until the caps bind or no slack remains.  Leftover capital is reported
   as ``remaining_cash`` (the spec explicitly tolerates < 100% deployed).

The router is mounted standalone so the ``main.py:routes`` owner can
include it via::

    from pfm.strategies.risk_budget_router import router as _rb_router
    app.include_router(_rb_router)
"""

from __future__ import annotations

import math
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from pfm.strategies import deployable_router as _dr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum weight per tier.  Per CLAUDE.md "Wave-5 stress test" defaults:
#: A_GOLD 25%, A_STRUCTURAL 20%, B_VALIDATED 10%.
TIER_CAPS: dict[str, float] = {
    "A_GOLD": 0.25,
    "A_STRUCTURAL": 0.20,
    "B_VALIDATED": 0.10,
}

#: Concentration limit — no individual strategy gets more than this share.
CONCENTRATION_LIMIT: float = 0.30

#: Fallback daily volatility per tier when the JSON row exposes none.
#: Lower-tier strategies get a higher prior vol so risk-parity does not
#: over-weight them when data is missing.
_FALLBACK_VOL_BY_TIER: dict[str, float] = {
    "A_GOLD": 0.012,
    "A_STRUCTURAL": 0.015,
    "B_VALIDATED": 0.020,
}

_DEFAULT_FALLBACK_VOL: float = 0.020


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class RiskBudgetAllocation(BaseModel):
    """A single line item in the risk-budget table."""

    pair_id: str = Field(..., min_length=1)
    tier: str = Field(..., min_length=1)
    label: str = Field(default="")
    weight: float = Field(..., ge=0.0, le=1.0)
    notional: float = Field(..., ge=0.0)
    vol: float = Field(..., gt=0.0)
    rationale: str = Field(default="")


class RiskBudgetResponse(BaseModel):
    """Top-level response for ``GET /strategies/risk-budget``."""

    total_capital: float = Field(..., gt=0.0)
    allocations: list[RiskBudgetAllocation]
    remaining_cash: float = Field(..., ge=0.0)
    total_active_capital: float = Field(..., ge=0.0)
    method: str = Field(default="risk-parity + tier-caps + 30% concentration")
    source: str = Field(default="json")


# ---------------------------------------------------------------------------
# Vol estimation
# ---------------------------------------------------------------------------


def _vol_for_item(item: _dr.DeployableItem) -> float:
    """Derive a positive daily-vol estimate for one deployable item.

    Strategy:

    1. If the underlying JSON row exposes any of ``vol``/``sigma``/
       ``daily_vol``/``ann_vol``, coerce + normalise (ann -> daily ≈ /16).
    2. Otherwise pick a tier-conditional fallback (lower-tier → higher vol).
    """
    # ``DeployableItem`` doesn't carry raw JSON, so the only way to recover
    # explicit vol is via tier fallback.  Callers can override by passing
    # rows directly to ``build_risk_budget`` (see ``_vol_from_row``).
    return _FALLBACK_VOL_BY_TIER.get(item.tier, _DEFAULT_FALLBACK_VOL)


def _vol_from_row(row: dict[str, Any]) -> float:
    """Pull a vol estimate directly from the curated JSON row when possible."""
    for key in ("daily_vol", "vol", "sigma"):
        candidate = _dr._coerce_float(row.get(key), default=float("nan"))
        if not math.isnan(candidate) and candidate > 0:
            return candidate
    ann = _dr._coerce_float(row.get("ann_vol"), default=float("nan"))
    if not math.isnan(ann) and ann > 0:
        # sqrt(252) ≈ 15.87 ≈ 16
        return ann / 16.0
    tier = str(row.get("tier") or "")
    return _FALLBACK_VOL_BY_TIER.get(tier, _DEFAULT_FALLBACK_VOL)


# ---------------------------------------------------------------------------
# Core allocator
# ---------------------------------------------------------------------------


def _risk_parity_raw(vols: list[float]) -> list[float]:
    """Compute risk-parity weights ``w_i ∝ 1/σ_i`` summing to 1."""
    inv = [1.0 / v for v in vols]
    total = sum(inv)
    if total <= 0:
        n = len(vols)
        return [1.0 / n] * n if n else []
    return [x / total for x in inv]


def _apply_caps_and_redistribute(
    weights: list[float],
    caps: list[float],
) -> list[float]:
    """Clip to caps, then redistribute the trimmed mass to under-capped names.

    Iterates until either every weight is at its cap or every excess has
    been redistributed.  Guarantees ``sum(weights) ≤ 1`` (with the slack
    surfacing as cash) and ``weights[i] ≤ caps[i]`` for every i.
    """
    if not weights:
        return []
    w = list(weights)
    n = len(w)
    # Safety: hard-cap once up-front.
    for i in range(n):
        w[i] = min(w[i], caps[i])
    # Iteratively redistribute slack into under-capped strategies.
    for _ in range(64):  # bounded fixed-point loop
        total = sum(w)
        slack = 1.0 - total
        if slack <= 1e-9:
            break
        # Strategies that still have headroom under their cap.
        headroom = [max(0.0, caps[i] - w[i]) for i in range(n)]
        total_headroom = sum(headroom)
        if total_headroom <= 1e-12:
            break
        # Distribute proportional to remaining headroom (i.e. names that
        # were already favoured by risk-parity get scaled up first).
        for i in range(n):
            if total_headroom > 0:
                w[i] += slack * (headroom[i] / total_headroom)
            w[i] = min(w[i], caps[i])
    return w


def build_risk_budget(
    items: list[_dr.DeployableItem],
    total_capital: float,
    *,
    vols: list[float] | None = None,
) -> RiskBudgetResponse:
    """Construct a :class:`RiskBudgetResponse` from a deployable list.

    Pure function, no I/O — used by both the FastAPI handler and tests.

    ``vols`` may be supplied directly (one positive float per item) to
    override the tier-conditional fallback.
    """
    if total_capital <= 0:
        raise ValueError("total_capital must be positive")
    if not items:
        return RiskBudgetResponse(
            total_capital=total_capital,
            allocations=[],
            remaining_cash=total_capital,
            total_active_capital=0.0,
            source="empty",
        )

    if vols is None:
        vols = [_vol_for_item(it) for it in items]
    if len(vols) != len(items):
        raise ValueError("vols and items must be same length")
    if any(v <= 0 for v in vols):
        raise ValueError("all vols must be > 0")

    raw = _risk_parity_raw(vols)

    # Per-item cap is min(tier-cap, concentration-limit).
    per_item_caps = [
        min(TIER_CAPS.get(it.tier, CONCENTRATION_LIMIT), CONCENTRATION_LIMIT) for it in items
    ]
    capped = _apply_caps_and_redistribute(raw, per_item_caps)

    allocations: list[RiskBudgetAllocation] = []
    for it, w, v, raw_w, cap in zip(items, capped, vols, raw, per_item_caps, strict=True):
        w_round = round(max(0.0, w), 6)
        notional = round(w_round * total_capital, 2)
        rationale = _rationale_for(it.tier, raw_w, w_round, cap)
        allocations.append(
            RiskBudgetAllocation(
                pair_id=it.pair_id,
                tier=it.tier,
                label=it.label,
                weight=w_round,
                notional=notional,
                vol=round(v, 6),
                rationale=rationale,
            )
        )

    # Sort highest-weight first for human-readable output.
    allocations.sort(key=lambda a: a.weight, reverse=True)

    total_weight = sum(a.weight for a in allocations)
    total_active = round(total_weight * total_capital, 2)
    remaining = round(max(0.0, total_capital - total_active), 2)

    return RiskBudgetResponse(
        total_capital=total_capital,
        allocations=allocations,
        remaining_cash=remaining,
        total_active_capital=total_active,
    )


def _rationale_for(tier: str, raw_w: float, final_w: float, cap: float) -> str:
    """Human-readable explanation of how the final weight was derived."""
    cap_pct = round(cap * 100, 1)
    raw_pct = round(raw_w * 100, 1)
    final_pct = round(final_w * 100, 1)
    if final_w >= cap - 1e-6:
        return (
            f"{tier} capped at {cap_pct}% "
            f"(risk-parity raw {raw_pct}% bound by tier/concentration cap)."
        )
    if final_w > raw_w + 1e-6:
        return (
            f"{tier} risk-parity raw {raw_pct}% scaled up to {final_pct}% "
            f"to absorb cap-trimmed mass from other strategies."
        )
    return f"{tier} risk-parity weight {final_pct}% (cap {cap_pct}%, raw {raw_pct}%)."


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/strategies", tags=["strategies-risk-budget"])


def _load_items_with_vols() -> tuple[list[_dr.DeployableItem], list[float], str]:
    """Pull deployable items + per-item vol estimates from the curated JSON.

    Mirrors the loader in :mod:`deployable_router` so we can grab the raw
    rows (which carry vol fields) instead of round-tripping through the
    cached HTTP response.
    """
    rows = _dr._load_strategies()
    deployable = _dr._filter_deployable(rows)
    if deployable:
        items = [_dr._build_item(r) for r in deployable]
        vols = [_vol_from_row(r) for r in deployable]
        return items, vols, "json"
    # Fallback path mirrors deployable_router exactly.
    items = list(_dr._FALLBACK_DEPLOYABLE)
    vols = [_vol_for_item(it) for it in items]
    return items, vols, "fallback"


@router.get(
    "/risk-budget",
    response_model=RiskBudgetResponse,
    summary="Risk-parity capital allocation across deployable strategies.",
)
def get_risk_budget(
    total_capital: Annotated[
        float,
        Query(
            gt=0.0,
            le=1e12,
            description="Total capital in account currency (e.g. USD).",
        ),
    ] = 100_000.0,
) -> RiskBudgetResponse:
    """Allocate ``total_capital`` across the deployable list.

    Returns weights, notionals, and a human-readable rationale per
    strategy plus the leftover cash.
    """
    items, vols, source = _load_items_with_vols()
    response = build_risk_budget(items, total_capital, vols=vols)
    response.source = source
    return response


__all__ = [
    "CONCENTRATION_LIMIT",
    "TIER_CAPS",
    "RiskBudgetAllocation",
    "RiskBudgetResponse",
    "build_risk_budget",
    "get_risk_budget",
    "router",
]
