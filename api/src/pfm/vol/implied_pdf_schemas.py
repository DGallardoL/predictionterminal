"""Shared contract for the implied-PDF feature (Phase 1+2).

This module is the **integration seam**. The math engine
(:mod:`pfm.vol.implied_pdf`), the Kalshi ladder-discovery
(:func:`pfm.sources.kalshi.discover_index_ladder`), the router
(:mod:`pfm.vol.implied_pdf_router`) and all the tests build against the types
and signatures defined *here*.

Do **not** change a public field name or a function signature without updating
every consumer in lock-step — several modules are authored in parallel against
this file.

Background
----------
A set of prediction-market binary contracts at a common maturity encodes a
risk-neutral distribution of an underlying. We support three input *shapes*:

``terminal_buckets``
    Range markets ("S&P close between X and Y on date D", Kalshi ``KXINX``).
    Each market's YES price **is** the probability mass over ``[floor, cap]``.
    No differentiation needed — just normalise and smooth.

``terminal_ladder``
    Above/below threshold markets at a fixed maturity ("close > K on D",
    Kalshi ``KXINXU``). YES("above K") = survival ``S(K) = P(S_T > K)``; the
    PDF is ``f(K) = -dS/dK`` (the derivative of the survival curve, for digitals).

``barrier_touch``
    Touch/one-touch markets ("reach K by D", Polymarket ``reach``/``hit_high``).
    YES("touch K") = ``P(M_T >= K)`` where ``M_T = max_{t<=T} S_t``. Differencing
    recovers the law of the **running maximum**, NOT the terminal price. This is
    model-free and honest. A terminal density is only recoverable under a GBM
    (reflection-principle) assumption — see :mod:`pfm.vol.implied_pdf` and
    ``docs/quants.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DataShape = Literal["terminal_buckets", "terminal_ladder", "barrier_touch"]
SmoothMethod = Literal["pchip_monotone", "lognormal", "empirical"]
Direction = Literal["above", "below", "between", "touch_above", "touch_below"]
AssetClass = Literal["equity_index", "crypto", "commodity_energy", "commodity_metal"]
DistributionOf = Literal["terminal_price", "running_max", "running_min"]

#: Default clipping epsilon for survival/CDF probabilities (mirrors the
#: project-wide convention; exposed as a query param on the endpoint).
DEFAULT_EPSILON: float = 0.01

#: Default number of points in the dense output grid.
DEFAULT_GRID_SIZE: int = 256

#: Annualisation factor for daily-σ → per-year σ (calendar-day convention used
#: for short-dated PM maturities; document any deviation in the result warnings).
TRADING_DAYS_PER_YEAR: float = 252.0


# ---------------------------------------------------------------------------
# Input types (plain dataclasses — cheap to construct in discovery + tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderEntry:
    """One market in a strike ladder.

    Field usage by ``direction``:

    * ``between`` (range bucket): set ``floor`` and ``cap``; ``strike`` ignored.
      A half-open tail bucket sets only one of ``floor``/``cap`` (the other
      ``None``) — the engine treats it as an open tail.
    * ``above`` / ``touch_above``: set ``strike`` (YES if value/peak > strike).
    * ``below`` / ``touch_below``: set ``strike`` (YES if value/trough < strike).

    ``prob`` is the market-implied YES probability in [0, 1] (mid of bid/ask,
    or last trade). The engine clips/monotonises as needed.
    """

    direction: Direction
    prob: float
    strike: float | None = None
    floor: float | None = None
    cap: float | None = None
    slug: str | None = None  # polymarket slug or kalshi market ticker
    venue: str | None = None
    # Executability metadata (Kalshi): the raw two-sided quote + activity, so a
    # fair-value scanner can trade the executable side and skip dead markets.
    yes_bid: float | None = None
    yes_ask: float | None = None
    volume: float | None = None
    open_interest: float | None = None


@dataclass
class LadderFamily:
    """A coherent set of same-maturity markets on one underlying."""

    asset: str
    asset_class: AssetClass
    data_shape: DataShape
    maturity_utc: datetime
    spot: float | None
    entries: list[LadderEntry]
    source: str = ""  # provenance, e.g. "kalshi:KXINX-26MAY15H1600"
    extra: dict = field(default_factory=dict)  # free-form discovery metadata


# ---------------------------------------------------------------------------
# Output types (Pydantic — these are the JSON response shapes)
# ---------------------------------------------------------------------------


class Moments(BaseModel):
    mean: float
    median: float
    mode: float
    std: float
    skew: float
    kurtosis: float  # excess kurtosis (normal == 0)


class Quantiles(BaseModel):
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


class MarketPoint(BaseModel):
    """A raw market observation, surfaced for transparency / overlay."""

    k: float  # representative strike (bucket midpoint or threshold)
    prob: float  # raw YES probability as quoted
    kind: Literal["mass", "survival", "cdf"]  # how to read `prob`
    floor: float | None = None
    cap: float | None = None
    slug: str | None = None


class GBMFit(BaseModel):
    """Result of fitting GBM (ν, σ) to a barrier (running-max) survival ladder."""

    sigma_annual: float = Field(..., ge=0.0)
    nu_annual: float  # log-drift ν = r - q - σ²/2
    rmse: float = Field(..., ge=0.0)  # survival-ladder fit residual
    converted_to_terminal: bool  # whether a touch→terminal overlay was produced


class ImpliedPDFResult(BaseModel):
    """Dense, smoothed implied distribution + summary stats."""

    asset: str
    data_shape: DataShape
    distribution_of: DistributionOf
    maturity_utc: datetime
    time_to_maturity_years: float = Field(..., ge=0.0)
    spot: float | None

    grid: list[float]  # dense K grid (length grid_size)
    pdf: list[float]  # primary density f(K) on grid, integrates ~1
    cdf: list[float]  # F(K) on grid, monotone non-decreasing in [0, 1]

    market_points: list[MarketPoint]  # raw inputs for transparency
    lognormal_overlay: list[float] | None = None  # smooth reference PDF on grid
    gbm_terminal_overlay: list[float] | None = None  # touch→terminal PDF (barrier)
    gbm_fit: GBMFit | None = None

    moments: Moments
    quantiles: Quantiles

    method: SmoothMethod
    eps: float
    n_strikes: int = Field(..., ge=1)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine entry-point signature (DOCUMENTATION ONLY — implemented in
# pfm.vol.implied_pdf; reproduced here so parallel authors agree on it).
# ---------------------------------------------------------------------------
#
#   def compute_implied_pdf(
#       family: LadderFamily,
#       *,
#       method: SmoothMethod = "pchip_monotone",
#       eps: float = DEFAULT_EPSILON,
#       grid_size: int = DEFAULT_GRID_SIZE,
#       barrier_to_terminal: bool = False,
#       tail_model: Literal["lognormal", "linear", "none"] = "lognormal",
#       now_utc: datetime | None = None,
#   ) -> ImpliedPDFResult: ...
#
# Discovery entry-point signature (implemented in pfm.sources.kalshi):
#
#   def discover_index_ladder(
#       series_ticker: str,
#       client: KalshiClient,
#       *,
#       maturity_filter: str | None = None,   # ISO-date prefix, e.g. "2026-05-15"
#       prefer_shape: DataShape | None = None,
#   ) -> LadderFamily: ...

__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_GRID_SIZE",
    "TRADING_DAYS_PER_YEAR",
    "AssetClass",
    "DataShape",
    "Direction",
    "DistributionOf",
    "GBMFit",
    "ImpliedPDFResult",
    "LadderEntry",
    "LadderFamily",
    "MarketPoint",
    "Moments",
    "Quantiles",
    "SmoothMethod",
]
