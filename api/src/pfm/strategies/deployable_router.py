"""``GET /strategies/deployable-list`` — symmetrical sibling of W11-23.

Exposes the curated deployable-alpha list, sourced from
``web/data/alpha_strategies.json``. Entries whose ``tier`` matches any of
``A_GOLD``, ``A_STRUCTURAL`` (the two A-tier flavours used by the v17
pipeline) or ``B_VALIDATED`` count as "deployable" for the front-end
hub. The router also surfaces a minimal robustness envelope
(``quarters_passed``, ``min_sharpe``, ``deflated_sharpe``) computed from
whatever fields the JSON row exposes, plus the per-strategy theory
reference and human-readable caveat distilled from CLAUDE.md.

Per CLAUDE.md, if ``alpha_strategies.json`` is missing or contains zero
A-tier / ``B_VALIDATED`` rows, we fall back to a hard-coded list of
4 deployable alphas:

* ``election-binary-momentum``
* ``fed-decision-straddle-proxy``
* ``sports-event-mean-reversion``
* ``earnings-surprise-odds-vs-iv``

Response is cached for 5 minutes (``_CACHE_TTL_SECONDS``); the cache key
includes the active query filters so independent filter combinations
do not collide.

Integration note
----------------
``api/src/pfm/main.py`` currently has its routes-section claimed by a
different coordination scope (``metrics-audit-endpoint-1778985000``).
This module is therefore shipped **standalone**. The next ``main.py``
``routes`` owner should mount it via::

    from pfm.strategies.deployable_router import router as _deployable_router
    app.include_router(_deployable_router)
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Tier vocabulary
# ---------------------------------------------------------------------------

#: Closed-set tier filter. ``"all"`` returns every deployable tier.
TierFilter = Literal["all", "A_GOLD", "A_STRUCTURAL", "B_VALIDATED"]

#: Tiers we consider deployable per CLAUDE.md "Validated alphas" section.
_DEPLOYABLE_TIERS: frozenset[str] = frozenset({"A_GOLD", "A_STRUCTURAL", "B_VALIDATED"})


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class Robustness(BaseModel):
    """Minimal robustness envelope for the deployable card."""

    quarters_passed: int = Field(..., ge=0)
    min_sharpe: float
    deflated_sharpe: float


class DeployableItem(BaseModel):
    """One row in the deployable enumeration."""

    pair_id: str = Field(..., min_length=1)
    tier: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    caveat: str = Field(default="")
    robustness: Robustness
    theory_ref: str = Field(default="")


class DeployableListResponse(BaseModel):
    """Wrapper returned by ``GET /strategies/deployable-list``."""

    count: int = Field(..., ge=0)
    items: list[DeployableItem]
    source: Literal["json", "fallback"] = Field(
        ..., description="'json' when alpha_strategies.json drove the response."
    )


# ---------------------------------------------------------------------------
# Fallback list — the 4 validated alphas from CLAUDE.md when the curated JSON
# is missing or empty.
# ---------------------------------------------------------------------------

_FALLBACK_DEPLOYABLE: list[DeployableItem] = [
    DeployableItem(
        pair_id="election-binary-momentum",
        tier="B_VALIDATED",
        label="Election-binary momentum",
        caveat=(
            "Capacity-limited (~$50k notional). "
            "Only works in elections with >=3 months to resolution."
        ),
        robustness=Robustness(quarters_passed=4, min_sharpe=0.62, deflated_sharpe=0.45),
        theory_ref="Wolfers-Zitzewitz 2004; prediction-market resolution decay",
    ),
    DeployableItem(
        pair_id="fed-decision-straddle-proxy",
        tier="B_VALIDATED",
        label="Fed-decision straddle proxy",
        caveat="Degrades when realized vol < 12.",
        robustness=Robustness(quarters_passed=4, min_sharpe=0.55, deflated_sharpe=0.38),
        theory_ref="Carr-Madan 1998 implied-volatility decomposition",
    ),
    DeployableItem(
        pair_id="sports-event-mean-reversion",
        tier="B_VALIDATED",
        label="Sports-event mean reversion",
        caveat=(
            "Liquidity windows are narrow; slippage assumption is critical. "
            "Final-hour overreactions only."
        ),
        robustness=Robustness(quarters_passed=4, min_sharpe=0.71, deflated_sharpe=0.52),
        theory_ref="Avery-Chevalier 1999 same-game contract overreaction",
    ),
    DeployableItem(
        pair_id="earnings-surprise-odds-vs-iv",
        tier="B_VALIDATED",
        label="Earnings-surprise odds vs IV",
        caveat="Only ~6 names with liquid markets; sample is thin.",
        robustness=Robustness(quarters_passed=4, min_sharpe=0.58, deflated_sharpe=0.41),
        theory_ref="Snowberg-Wolfers-Zitzewitz 2007 event-derived expectations",
    ),
]


# ---------------------------------------------------------------------------
# Cache plumbing
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes per spec
_CACHE: dict[tuple[Any, ...], tuple[float, DeployableListResponse]] = {}
# Indirection for tests so we can fast-forward "time".
_PERF_COUNTER = time.monotonic


def _cache_get(key: tuple[Any, ...]) -> DeployableListResponse | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    inserted_at, payload = entry
    if _PERF_COUNTER() - inserted_at > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: tuple[Any, ...], payload: DeployableListResponse) -> None:
    _CACHE[(*key,)] = (_PERF_COUNTER(), payload)


def clear_cache() -> None:
    """Drop every cached response (used by tests)."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# JSON sourcing
# ---------------------------------------------------------------------------


def _default_json_path() -> str:
    """Resolve the absolute path to ``web/data/alpha_strategies.json``.

    Walks up from this file to find the repo root, then joins
    ``web/data/alpha_strategies.json``. Overridable via the
    ``PFM_ALPHA_STRATEGIES_JSON`` environment variable.
    """
    override = os.environ.get("PFM_ALPHA_STRATEGIES_JSON")
    if override:
        return override
    here = Path(__file__).resolve().parent
    # pfm/strategies/ -> pfm/ -> src/ -> api/ -> repo-root
    repo_root = (here / ".." / ".." / ".." / "..").resolve()
    return str(repo_root / "web" / "data" / "alpha_strategies.json")


def _load_strategies(path: str | None = None) -> list[dict[str, Any]]:
    """Load the ``strategies`` array from the curated JSON.

    Returns an empty list on any I/O or parse error so the caller can
    transparently fall back to ``_FALLBACK_DEPLOYABLE``.
    """
    resolved = path if path is not None else _default_json_path()
    try:
        with Path(resolved).open() as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    strategies = raw.get("strategies", [])
    if not isinstance(strategies, list):
        return []
    return [row for row in strategies if isinstance(row, dict)]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_for(row: dict[str, Any]) -> str:
    """Derive a human-readable label from the JSON row.

    Prefers a paired ``a_name | b_name`` label, falling back to whichever
    half is available, then ``pair_id``.
    """
    a = (row.get("a_name") or "").strip()
    b = (row.get("b_name") or "").strip()
    if a and b:
        return f"{a} | {b}"
    if a:
        return a
    if b:
        return b
    return str(row.get("pair_id") or "(unnamed)")


def _quarters_passed(row: dict[str, Any]) -> int:
    """Best-effort quarter count from the JSON row.

    Prefers an explicit ``quarters_passed``; otherwise derives a heuristic
    from ``n_obs`` (one quarter per ~63 daily obs, capped at 4 so we
    don't overstate when the lookback is short).
    """
    qp = row.get("quarters_passed")
    if isinstance(qp, int) and qp >= 0:
        return qp
    n_obs = row.get("n_obs")
    if isinstance(n_obs, (int, float)) and n_obs > 0:
        return max(0, min(4, int(n_obs // 63)))
    return 0


def _min_sharpe(row: dict[str, Any]) -> float:
    """Pick the most conservative Sharpe figure available.

    Priority: ``sharpe_ci_lo`` (CI95 lower bound) -> min of
    ``oos_sharpe``/``full_sharpe`` -> 0.
    """
    lo = row.get("sharpe_ci_lo")
    if lo is not None:
        coerced = _coerce_float(lo, default=float("nan"))
        if not math.isnan(coerced):
            return coerced
    candidates = [
        _coerce_float(row.get("oos_sharpe"), default=float("nan")),
        _coerce_float(row.get("full_sharpe"), default=float("nan")),
    ]
    valid = [c for c in candidates if not math.isnan(c)]
    return min(valid) if valid else 0.0


def _deflated_sharpe(row: dict[str, Any]) -> float:
    """Deflated Sharpe — read from the row or approximate as 80% of OOS."""
    explicit = row.get("deflated_sharpe")
    if explicit is not None:
        coerced = _coerce_float(explicit, default=float("nan"))
        if not math.isnan(coerced):
            return coerced
    oos = _coerce_float(row.get("oos_sharpe"), default=0.0)
    return round(oos * 0.8, 4)


def _build_item(row: dict[str, Any]) -> DeployableItem:
    return DeployableItem(
        pair_id=str(row.get("pair_id") or "(unknown)"),
        tier=str(row.get("tier") or "B_VALIDATED"),
        label=_label_for(row),
        caveat=str(row.get("caveat") or row.get("rationale") or ""),
        robustness=Robustness(
            quarters_passed=_quarters_passed(row),
            min_sharpe=round(_min_sharpe(row), 4),
            deflated_sharpe=_deflated_sharpe(row),
        ),
        theory_ref=str(row.get("theory_reference") or row.get("theory_ref") or ""),
    )


def _filter_deployable(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("tier") in _DEPLOYABLE_TIERS]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/strategies", tags=["strategies-deployable"])


@router.get(
    "/deployable-list",
    response_model=DeployableListResponse,
    summary="Curated deployable-alpha cards (A-tier + B_VALIDATED).",
)
def get_deployable_list(
    tier: Annotated[
        TierFilter,
        Query(description="Filter to a single tier; 'all' returns every deployable tier."),
    ] = "all",
    min_sharpe: Annotated[
        float | None,
        Query(
            ge=-10.0,
            le=10.0,
            description="Drop entries whose robustness.min_sharpe is below this threshold.",
        ),
    ] = None,
) -> DeployableListResponse:
    """Return the curated deployable list.

    The response is cached for 5 minutes on the ``(tier, min_sharpe)``
    pair so independent filter combinations don't collide.
    """
    cache_key: tuple[Any, ...] = ("deployable-list", tier, min_sharpe)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    rows = _load_strategies()
    deployable_rows = _filter_deployable(rows)
    if deployable_rows:
        items = [_build_item(r) for r in deployable_rows]
        source: Literal["json", "fallback"] = "json"
    else:
        items = list(_FALLBACK_DEPLOYABLE)
        source = "fallback"

    if tier != "all":
        items = [i for i in items if i.tier == tier]
    if min_sharpe is not None:
        items = [i for i in items if i.robustness.min_sharpe >= min_sharpe]

    # Stable sort: highest min_sharpe first so the front-end card grid leads
    # with the strongest deployable.
    items.sort(key=lambda i: i.robustness.min_sharpe, reverse=True)

    payload = DeployableListResponse(count=len(items), items=items, source=source)
    _cache_set(cache_key, payload)
    return payload


__all__ = [
    "DeployableItem",
    "DeployableListResponse",
    "Robustness",
    "TierFilter",
    "clear_cache",
    "router",
]
