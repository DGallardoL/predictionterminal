"""Schemas for /factors/* endpoints (catalog, discover, preview, rank, permutation, best)."""

from __future__ import annotations

from datetime import date as _date
from typing import Literal

from pydantic import BaseModel, Field

from pfm.schemas.common import (
    TICKER_PATTERN,
    AlignmentLit,
    CustomFactor,
    RegressionLit,
    ReturnTypeLit,
)

# ---- /factors ---------------------------------------------------------------


class FactorMetadata(BaseModel):
    id: str
    name: str
    slug: str
    source: str
    description: str
    theme: str = "other"


class FactorList(BaseModel):
    """Paginated factor list.

    The ``total`` / ``limit`` / ``offset`` / ``next_offset`` fields were
    added once the catalog grew past 1 000 entries; the legacy ``factors``
    array is preserved so existing clients ``r.json()["factors"]`` keep
    working. ``next_offset`` is ``None`` when there are no more pages.
    """

    factors: list[FactorMetadata]
    total: int = 0
    limit: int = 0
    offset: int = 0
    next_offset: int | None = None


# ---- /factors/discover ------------------------------------------------------


class DiscoveredMarket(BaseModel):
    slug: str
    question: str
    volume: float
    end_date: str | None = None
    active: bool
    closed: bool


class DiscoverResponse(BaseModel):
    markets: list[DiscoveredMarket]


# ---- /factors/preview -------------------------------------------------------


class PreviewRequest(BaseModel):
    slug: str = Field(min_length=1, max_length=200)
    source: Literal["polymarket", "kalshi"] = "polymarket"


class PriceBar(BaseModel):
    date: _date
    price: float


class PreviewResponse(BaseModel):
    slug: str
    question: str
    yes_token_id: str
    active: bool
    closed: bool
    n_bars: int
    first_date: _date | None = None
    last_date: _date | None = None
    current_price: float | None = None
    history: list[PriceBar]


# ---- /factors/rank ----------------------------------------------------------


class RankRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, examples=["NVDA"], pattern=TICKER_PATTERN)
    start: _date
    end: _date
    return_type: ReturnTypeLit = "log"
    regression: RegressionLit = "hac"
    alignment: AlignmentLit = "strict"
    custom_factors: list[CustomFactor] = Field(default_factory=list)
    # Anything below ``min_n_for_ranking`` overlapping observations is demoted
    # below "fully-ranked" results so a tiny-sample R²=0.64 (n=11) doesn't
    # out-rank a 100-obs R²=0.15. The thin-sample factors are still returned
    # so callers can inspect them — just sorted after the reliable ones.
    min_n_for_ranking: int = Field(
        default=30,
        ge=5,
        le=10_000,
        description=(
            "Minimum overlapping observations required to enter the primary "
            "ranking by R². Factors below the threshold are returned but sorted "
            "after the well-sampled ones (and after errors)."
        ),
    )


class RankItem(BaseModel):
    factor_id: str
    name: str
    slug: str
    theme: str
    n_obs: int
    r_squared: float
    beta: float
    t_stat: float
    p_value: float
    sample_first_date: _date | None = None
    sample_last_date: _date | None = None
    error: str | None = None


class RankResponse(BaseModel):
    ticker: str
    start: _date
    end: _date
    return_type: ReturnTypeLit
    regression: RegressionLit
    items: list[RankItem]


# ---- /factors/best (forward stepwise) --------------------------------------


class BestModelRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, examples=["NVDA"], pattern=TICKER_PATTERN)
    start: _date
    end: _date
    return_type: ReturnTypeLit = "log"
    regression: RegressionLit = "hac"
    alignment: AlignmentLit = "strict"
    custom_factors: list[CustomFactor] = Field(default_factory=list)
    max_factors: int = Field(default=5, ge=1, le=10)
    min_obs: int = Field(default=30, ge=10)
    # ── Predictive-quality controls (defaults aimed at OOS, not in-sample) ──
    criterion: Literal["r2_adj", "oos_r2"] = Field(
        default="oos_r2",
        description="Selection criterion: in-sample R² adj, or chronological 80/20 OOS R².",
    )
    filter_resolving: bool = Field(
        default=True,
        description="Skip factors in resolution-collapse phase (extreme moves in last 14 days).",
    )
    zscore: bool = Field(
        default=True,
        description="Z-score each Δlogit so factors have unit variance during selection.",
    )
    residualize_market: bool = Field(
        default=False,
        description="Predict α (residual after SPY β) instead of raw return.",
    )
    theme_composites: bool = Field(
        default=False,
        description="Average factors within each theme into composites — reduces 30+ factors to ~7, fights overfitting.",
    )
    max_per_theme: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Constrain stepwise to at most N factors per theme. Forces diversification.",
    )


class StepwiseStep(BaseModel):
    step: int
    added: str
    r_squared: float
    r_squared_adj: float
    n_obs: int
    candidates_considered: int


class BestModelResponse(BaseModel):
    ticker: str
    start: _date
    end: _date
    selected: list[str]
    final_r_squared: float
    final_r_squared_adj: float
    final_n_obs: int
    log: list[StepwiseStep]
    rejected: list[str] = Field(default_factory=list)
