"""Whale Mirror Portfolio — surface top whales and replicate their books.

This module sits one level above :mod:`pfm.terminal_whale_tracker`, which only
exposes per-market whale aggregation. Here we want a *cohort-level* view:

  - Identify the top-N whales by absolute 7d PnL.
  - For each whale, summarise current positions value, win rate, and
    number of active positions.
  - Given a user's capital budget, propose a proportional mirror portfolio
    over the whale's currently-held contracts and translate that to an
    estimated *equity-equivalent beta* (so a stock-only PM, looking at this
    UI, can sanity-check exposure against their existing book).

The live whale-positions API is rate-limited and per-market. Building a true
"all-positions for wallet X" view would need indexed wallet activity that
Polymarket doesn't expose for free. So for the POC we lean on a deterministic
synthetic generator (seeded by the wallet address) when live data isn't
available — the API contract stays identical, and the UI can show the
"synthetic / live" badge in the response envelope.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)

CACHE = get_cache("whale_mirror", ttl=300)

# Determinism knobs — this is a POC; live wallet-history isn't accessible
# free-tier, so we synthesise plausible per-whale books seeded by address.
_SYNTH_WHALE_POOL: tuple[str, ...] = (
    "0xWHALE000000000000000000000000000000A001",
    "0xWHALE000000000000000000000000000000A002",
    "0xWHALE000000000000000000000000000000A003",
    "0xWHALE000000000000000000000000000000A004",
    "0xWHALE000000000000000000000000000000A005",
    "0xWHALE000000000000000000000000000000A006",
    "0xWHALE000000000000000000000000000000A007",
    "0xWHALE000000000000000000000000000000A008",
    "0xWHALE000000000000000000000000000000A009",
    "0xWHALE000000000000000000000000000000A010",
    "0xWHALE000000000000000000000000000000A011",
    "0xWHALE000000000000000000000000000000A012",
)

# A fixed factor universe used to synthesise whale books. Real factor slugs
# from factors.yml would substitute these in production; for the POC the
# names need only be plausible and stable.
_SYNTH_SLUG_POOL: tuple[str, ...] = (
    "trump-wins-2024",
    "fed-cut-march-2026",
    "btc-150k-by-eoy",
    "spx-6500-by-eoy",
    "recession-2026",
    "nvda-eps-beat-q1",
    "tsla-1tn-by-eoy",
    "election-senate-control",
    "oil-100-by-eoy",
    "nfp-positive-may",
    "vix-25-by-jun",
    "earnings-beat-aapl",
    "ai-capex-cut-q2",
    "cpi-below-3pct",
    "geopolitics-mideast",
)

# Slug -> equity-beta proxy. Used when the user mirrors a whale to translate
# the PM book into an estimated equity-equivalent β. Numbers are illustrative
# (a fed-cut market is rate-sensitive; recession-odds shorts equities).
_SLUG_EQUITY_BETA: dict[str, float] = {
    "trump-wins-2024": 0.10,
    "fed-cut-march-2026": -0.30,
    "btc-150k-by-eoy": 1.40,
    "spx-6500-by-eoy": 1.00,
    "recession-2026": -0.85,
    "nvda-eps-beat-q1": 1.20,
    "tsla-1tn-by-eoy": 1.55,
    "election-senate-control": 0.05,
    "oil-100-by-eoy": 0.45,
    "nfp-positive-may": 0.20,
    "vix-25-by-jun": -0.55,
    "earnings-beat-aapl": 1.10,
    "ai-capex-cut-q2": -0.65,
    "cpi-below-3pct": 0.35,
    "geopolitics-mideast": -0.40,
}


# --- schemas ----------------------------------------------------------------


class WhaleSummary(BaseModel):
    address: str
    pnl_7d_usd: float
    positions_value_usd: float
    win_rate: float = Field(..., ge=0.0, le=1.0)
    num_active_positions: int = Field(..., ge=0)
    last_active_iso: str | None = None


class TopWhalesResponse(BaseModel):
    window_days: int
    min_pnl_usd: float
    n_whales: int
    whales: list[WhaleSummary]
    source: Literal["live", "synthetic"] = "synthetic"


class MirrorPosition(BaseModel):
    slug: str
    side: Literal["YES", "NO"] = "YES"
    size_usd: float = Field(..., ge=0.0)
    current_price: float = Field(..., ge=0.0, le=1.0)
    target_price: float = Field(..., ge=0.0, le=1.0)
    equity_beta: float = Field(
        ..., description="Estimated equity-equivalent β contribution per $1 long."
    )


class MirrorRequest(BaseModel):
    whale_address: str = Field(..., min_length=4)
    capital_usd: float = Field(..., gt=0.0)
    max_positions: int = Field(10, ge=1, le=30)


class MirrorResponse(BaseModel):
    whale_address: str
    capital_usd: float
    suggested_positions: list[MirrorPosition]
    total_exposure: float
    equivalent_equity_beta_estimate: float
    source: Literal["live", "synthetic"] = "synthetic"
    notes: str


class WhaleHistoryPoint(BaseModel):
    date_iso: str
    cumulative_pnl_usd: float
    positions_value_usd: float


class WhaleHistoryResponse(BaseModel):
    whale_address: str
    days: int
    trace: list[WhaleHistoryPoint]
    source: Literal["live", "synthetic"] = "synthetic"


# --- core logic -------------------------------------------------------------


def _addr_seed(address: str) -> int:
    """Stable 32-bit seed derived from a wallet address.

    Same address → same synthetic book across calls. We hash with sha256
    rather than ``hash()`` because Python's builtin hash is salted per
    process and would break determinism across restarts.
    """
    digest = hashlib.sha256(address.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _synth_whale_summary(address: str, window_days: int) -> WhaleSummary:
    """Generate a deterministic plausible WhaleSummary for ``address``.

    The distributions are loosely calibrated to look like a "real" whale:
    PnL roughly uniform in [-$120k, +$400k] (skewed positive because we're
    looking at the *top* cohort), positions value $50k–$1M, 4–14 positions,
    win rate clustered around 0.55.
    """
    rng = np.random.default_rng(_addr_seed(address))
    pnl_scale = 50_000.0 + (window_days / 7.0) * 30_000.0
    pnl = float(rng.normal(loc=80_000.0, scale=pnl_scale))
    pos_value = float(50_000.0 + rng.gamma(shape=2.0, scale=180_000.0))
    n_pos = int(rng.integers(low=4, high=15))
    win_rate = float(np.clip(rng.normal(0.55, 0.08), 0.30, 0.85))
    return WhaleSummary(
        address=address,
        pnl_7d_usd=round(pnl, 2),
        positions_value_usd=round(pos_value, 2),
        win_rate=round(win_rate, 3),
        num_active_positions=n_pos,
        last_active_iso=None,
    )


def top_whales(window_days: int = 7, min_pnl_usd: float = 10_000.0) -> list[dict]:
    """Return the top whales over ``window_days``, sorted by ``|PnL|`` desc.

    Args:
        window_days: Lookback for PnL accounting. Synthetic in this POC.
        min_pnl_usd: Minimum absolute PnL to include a wallet.

    Returns:
        A list of plain dicts (one per whale) shaped like
        :class:`WhaleSummary`. Sorted by absolute PnL descending.
    """
    cache_key = ("top_whales", int(window_days), float(min_pnl_usd))
    cached = CACHE.get(cache_key)
    if cached is not None:
        return cached

    summaries = [_synth_whale_summary(addr, window_days) for addr in _SYNTH_WHALE_POOL]
    summaries = [s for s in summaries if abs(s.pnl_7d_usd) >= min_pnl_usd]
    summaries.sort(key=lambda s: abs(s.pnl_7d_usd), reverse=True)

    result = [s.model_dump() for s in summaries]
    CACHE.set(cache_key, result, ttl=300)
    return result


def _synth_whale_positions(address: str, max_positions: int) -> list[dict]:
    """Build a deterministic per-whale set of current positions.

    Each position is a dict ``{slug, side, current_price, target_price,
    weight}``. Weights sum to 1.0 over the chosen positions so the caller
    can scale by capital.
    """
    rng = np.random.default_rng(_addr_seed(address))
    # Pick at least 1 position; ceiling is min(max_positions, 12).
    upper = max(1, min(int(max_positions), 12))
    lower = min(4, upper)  # avoid low >= high when caller passes max_positions < 4
    if lower == upper:
        n = upper
    else:
        n = int(rng.integers(low=lower, high=upper + 1))
    chosen_idx = rng.choice(len(_SYNTH_SLUG_POOL), size=n, replace=False)
    raw_weights = rng.dirichlet(np.ones(n) * 1.5)

    positions: list[dict] = []
    for i, w in zip(chosen_idx, raw_weights, strict=True):
        slug = _SYNTH_SLUG_POOL[int(i)]
        side: Literal["YES", "NO"] = "YES" if rng.random() > 0.35 else "NO"
        current_price = float(np.clip(rng.beta(2.0, 2.0), 0.05, 0.95))
        # Target price: drift upward if YES (whale is presumably long-thesis),
        # downward if NO; clipped into [0.02, 0.98].
        drift = 0.10 if side == "YES" else -0.08
        target_price = float(np.clip(current_price + drift + rng.normal(0, 0.05), 0.02, 0.98))
        positions.append(
            {
                "slug": slug,
                "side": side,
                "current_price": round(current_price, 4),
                "target_price": round(target_price, 4),
                "weight": float(w),
            }
        )
    return positions


def mirror_whale(
    whale_address: str,
    capital_usd: float,
    max_positions: int = 10,
) -> dict:
    """Construct a proportionally-sized mirror portfolio for ``whale_address``.

    The whale's current positions are pulled (synthetic in POC), then sized
    proportionally so that ``Σ size_usd ≤ capital_usd``. The equity-equivalent
    β is the dollar-weighted sum of per-slug β proxies (see
    :data:`_SLUG_EQUITY_BETA`), with NO positions inverting the sign.

    Args:
        whale_address: Whale wallet to mirror.
        capital_usd: User's total budget in USD. All sizes scale to this.
        max_positions: Cap on the number of positions surfaced.

    Returns:
        A dict shaped like :class:`MirrorResponse` with `suggested_positions`,
        total exposure, and the estimated equity-equivalent β.
    """
    if capital_usd <= 0:
        raise ValueError("capital_usd must be positive")

    raw_positions = _synth_whale_positions(whale_address, max_positions=max_positions)
    raw_positions.sort(key=lambda p: p["weight"], reverse=True)
    raw_positions = raw_positions[:max_positions]

    total_weight = sum(p["weight"] for p in raw_positions) or 1.0
    suggested: list[MirrorPosition] = []
    beta_dollars = 0.0

    for p in raw_positions:
        size = capital_usd * (p["weight"] / total_weight)
        slug_beta = _SLUG_EQUITY_BETA.get(p["slug"], 0.20)
        # NO position inverts the directional exposure.
        signed_beta = slug_beta if p["side"] == "YES" else -slug_beta
        beta_dollars += signed_beta * size
        suggested.append(
            MirrorPosition(
                slug=p["slug"],
                side=p["side"],
                size_usd=round(size, 2),
                current_price=p["current_price"],
                target_price=p["target_price"],
                equity_beta=signed_beta,
            )
        )

    total_exposure = round(sum(s.size_usd for s in suggested), 2)
    equivalent_beta = beta_dollars / capital_usd if capital_usd > 0 else 0.0

    return MirrorResponse(
        whale_address=whale_address,
        capital_usd=capital_usd,
        suggested_positions=suggested,
        total_exposure=total_exposure,
        equivalent_equity_beta_estimate=round(equivalent_beta, 4),
        source="synthetic",
        notes=(
            "POC mirror: positions synthesised from a deterministic seed of "
            "the whale address. β is the $-weighted sum of per-slug equity-β "
            "proxies; treat it as a sanity check, not a tradable hedge ratio."
        ),
    ).model_dump()


def whale_history(address: str, days: int = 30) -> dict:
    """Return a 30-day cumulative-PnL trace for a single whale.

    The path is a simple seeded random walk anchored to the whale's
    snapshot PnL; deterministic for the same ``(address, days)`` pair.
    """
    rng = np.random.default_rng(_addr_seed(address) ^ (days & 0xFFFF))
    summary = _synth_whale_summary(address, window_days=days)
    daily_drift = summary.pnl_7d_usd / max(days, 1)
    daily_vol = max(abs(summary.pnl_7d_usd) * 0.05, 2_000.0)

    cum = 0.0
    pos_value = summary.positions_value_usd
    trace: list[WhaleHistoryPoint] = []
    for i in range(days):
        cum += float(rng.normal(daily_drift, daily_vol))
        # Positions value drifts within ±20% of the current snapshot.
        pos_value = float(
            np.clip(pos_value + rng.normal(0, 0.05) * pos_value, 1_000.0, 5_000_000.0)
        )
        trace.append(
            WhaleHistoryPoint(
                date_iso=f"2026-04-{(i % 30) + 1:02d}",
                cumulative_pnl_usd=round(cum, 2),
                positions_value_usd=round(pos_value, 2),
            )
        )

    return WhaleHistoryResponse(
        whale_address=address,
        days=days,
        trace=trace,
        source="synthetic",
    ).model_dump()


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/whales", tags=["whale-mirror"])


@router.get(
    "/top",
    response_model=TopWhalesResponse,
    summary="Top whales by absolute 7d PnL.",
)
def get_top_whales(
    window_days: Annotated[int, Query(ge=1, le=90)] = 7,
    min_pnl_usd: Annotated[float, Query(ge=0.0)] = 10_000.0,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> TopWhalesResponse:
    rows = top_whales(window_days=window_days, min_pnl_usd=min_pnl_usd)
    rows = rows[:limit]
    return TopWhalesResponse(
        window_days=window_days,
        min_pnl_usd=min_pnl_usd,
        n_whales=len(rows),
        whales=[WhaleSummary(**r) for r in rows],
        source="synthetic",
    )


@router.post(
    "/mirror",
    response_model=MirrorResponse,
    summary="Build a mirror portfolio over a whale's current positions.",
)
def post_mirror(body: MirrorRequest) -> MirrorResponse:
    try:
        payload = mirror_whale(
            whale_address=body.whale_address,
            capital_usd=body.capital_usd,
            max_positions=body.max_positions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return MirrorResponse(**payload)


@router.get(
    "/{address}/history",
    response_model=WhaleHistoryResponse,
    summary="Cumulative-PnL trace for a single whale over N days.",
)
def get_whale_history(
    address: Annotated[str, Path(min_length=4)],
    days: Annotated[int, Query(ge=1, le=180)] = 30,
) -> WhaleHistoryResponse:
    payload = whale_history(address=address, days=days)
    return WhaleHistoryResponse(**payload)


__all__ = [
    "MirrorPosition",
    "MirrorRequest",
    "MirrorResponse",
    "TopWhalesResponse",
    "WhaleHistoryResponse",
    "WhaleSummary",
    "mirror_whale",
    "router",
    "top_whales",
    "whale_history",
]
