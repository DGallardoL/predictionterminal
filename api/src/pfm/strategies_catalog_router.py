"""Catalog / discovery endpoint over the `/strategies/*` POST endpoints.

The Strategies mode in the API has grown well past 30 endpoints; the
frontend (and downstream API consumers) need an enumeration so they can
render a discovery panel without hard-coding the list.

Endpoints
---------
* ``GET /strategies/list`` — every strategy endpoint with a slug, a
  human description and a category tag.
* ``GET /strategies/discovery?tag=<tag>`` — same payload filtered by tag.

The catalog is hand-maintained in :data:`STRATEGY_CATALOG`. Adding a new
strategy means appending one row here. Tests assert the count is sane.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Tag vocabulary — closed set so the tag filter is type-checked at the door.
# ---------------------------------------------------------------------------

#: Closed-set tag — extra ``"all"`` makes the filter idiomatic.
StrategyTag = Literal[
    "all",
    "classical",
    "stat-arb",
    "diagnostic",
    "ml",
    "pattern",
    "execution",
    "risk",
    "validation",
    "regime",
    "scanner",
]


class StrategyEntry(BaseModel):
    """One row in the catalog enumeration."""

    id: str = Field(..., min_length=1)
    endpoint: str = Field(..., min_length=1)
    method: Literal["GET", "POST"] = "POST"
    description: str = Field(..., min_length=1)
    tag: str = Field(..., min_length=1)


class StrategyListResponse(BaseModel):
    """Wrapper returned by ``GET /strategies/list``."""

    total: int = Field(..., ge=0)
    items: list[StrategyEntry]


# ---------------------------------------------------------------------------
# Catalog — the registered ``@app.post("/strategies/...")`` routes in main.py.
# Keep this in sync with main.py when adding new strategy endpoints.
# ---------------------------------------------------------------------------

STRATEGY_CATALOG: list[StrategyEntry] = [
    StrategyEntry(
        id="implication",
        endpoint="/strategies/implication",
        description="Logical implication / inclusion bound between two markets.",
        tag="classical",
    ),
    StrategyEntry(
        id="conditional",
        endpoint="/strategies/conditional",
        description="Conditional probability / Bayes-coherence test.",
        tag="classical",
    ),
    StrategyEntry(
        id="bounds",
        endpoint="/strategies/bounds",
        description="Static no-arbitrage bounds across related contracts.",
        tag="classical",
    ),
    StrategyEntry(
        id="spot-vs-implied",
        endpoint="/strategies/spot-vs-implied",
        description="Spot price vs. implied probability cross-check.",
        tag="classical",
    ),
    StrategyEntry(
        id="cointegration",
        endpoint="/strategies/cointegration",
        description="Engle-Granger pairs cointegration with HAC standard errors.",
        tag="classical",
    ),
    StrategyEntry(
        id="pairs-backtest",
        endpoint="/strategies/pairs-backtest",
        description="Walk-forward pairs trade with z-score signals + costs.",
        tag="classical",
    ),
    StrategyEntry(
        id="event-model",
        endpoint="/strategies/event-model",
        description="Event-window factor regression around macro releases.",
        tag="regime",
    ),
    StrategyEntry(
        id="basket-stat-arb",
        endpoint="/strategies/basket-stat-arb",
        description="Multi-leg basket stat-arb with PCA residual signal.",
        tag="stat-arb",
    ),
    StrategyEntry(
        id="ou-bands",
        endpoint="/strategies/ou-bands",
        description="Bertram optimal entry/exit bands on an OU process.",
        tag="stat-arb",
    ),
    StrategyEntry(
        id="granger",
        endpoint="/strategies/granger",
        description="Granger causality test between two series.",
        tag="diagnostic",
    ),
    StrategyEntry(
        id="kalman-hedge",
        endpoint="/strategies/kalman-hedge",
        description="Time-varying hedge ratio via Kalman filter.",
        tag="stat-arb",
    ),
    StrategyEntry(
        id="mean-reversion",
        endpoint="/strategies/mean-reversion",
        description="Half-life-based mean reversion with z-score gating.",
        tag="stat-arb",
    ),
    StrategyEntry(
        id="auto-backtest",
        endpoint="/strategies/auto-backtest",
        description="One-click auto-pipeline: discover -> fit -> backtest.",
        tag="scanner",
    ),
    StrategyEntry(
        id="patterns",
        endpoint="/strategies/patterns",
        description="Chart-pattern detector (head-shoulders / triangles / etc.).",
        tag="pattern",
    ),
    StrategyEntry(
        id="ml-predictor",
        endpoint="/strategies/ml-predictor",
        description="Gradient-boosted ML signal with walk-forward validation.",
        tag="ml",
    ),
    StrategyEntry(
        id="info-share",
        endpoint="/strategies/info-share",
        description="Hasbrouck information share between two price series.",
        tag="diagnostic",
    ),
    StrategyEntry(
        id="regime-switching",
        endpoint="/strategies/regime-switching",
        description="Markov-switching mean / variance regime model.",
        tag="regime",
    ),
    StrategyEntry(
        id="almgren-chriss",
        endpoint="/strategies/almgren-chriss",
        description="Optimal execution schedule (Almgren-Chriss).",
        tag="execution",
    ),
    StrategyEntry(
        id="fractional-diff",
        endpoint="/strategies/fractional-diff",
        description="Fractional differentiation for stationarity preservation.",
        tag="diagnostic",
    ),
    StrategyEntry(
        id="garch",
        endpoint="/strategies/garch",
        description="GARCH(1,1) / asymmetric GJR-GARCH volatility fit.",
        tag="risk",
    ),
    StrategyEntry(
        id="dfa",
        endpoint="/strategies/dfa",
        description="Detrended Fluctuation Analysis (long-memory / Hurst).",
        tag="diagnostic",
    ),
    StrategyEntry(
        id="triple-barrier",
        endpoint="/strategies/triple-barrier",
        description="Lopez de Prado triple-barrier labeling for signals.",
        tag="ml",
    ),
    StrategyEntry(
        id="distance-method",
        endpoint="/strategies/distance-method",
        description="Gatev-Goetzmann-Rouwenhorst distance pairs trade.",
        tag="stat-arb",
    ),
    StrategyEntry(
        id="robust-validation",
        endpoint="/strategies/robust-validation",
        description="Combined permutation + bootstrap robustness gate.",
        tag="validation",
    ),
    StrategyEntry(
        id="portfolio",
        endpoint="/strategies/portfolio",
        description="Mean-variance portfolio over alpha-hub strategies.",
        tag="risk",
    ),
    StrategyEntry(
        id="factor-model-pro",
        endpoint="/strategies/factor-model-pro",
        description="Multi-factor regression with HAC + VIF diagnostics.",
        tag="classical",
    ),
    StrategyEntry(
        id="cusum",
        endpoint="/strategies/cusum",
        description="CUSUM structural-break detector on residuals.",
        tag="diagnostic",
    ),
    StrategyEntry(
        id="walk-forward",
        endpoint="/strategies/walk-forward",
        description="Walk-forward optimization with IS/OOS split.",
        tag="validation",
    ),
    StrategyEntry(
        id="sharpe-bootstrap",
        endpoint="/strategies/sharpe-bootstrap",
        description="Stationary-bootstrap Sharpe-ratio confidence interval.",
        tag="validation",
    ),
    StrategyEntry(
        id="sharpe-permutation",
        endpoint="/strategies/sharpe-permutation",
        description="Permutation test against random-trade null Sharpe.",
        tag="validation",
    ),
    StrategyEntry(
        id="presets",
        endpoint="/strategies/presets",
        method="GET",
        description="Curated example inputs for every strategies sub-tool.",
        tag="scanner",
    ),
    StrategyEntry(
        id="scan",
        endpoint="/strategies/scan",
        description="Cartesian inefficiency scanner across the factor catalog.",
        tag="scanner",
    ),
]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/strategies", tags=["strategies-catalog"])


@router.get(
    "/list",
    response_model=StrategyListResponse,
    summary="Enumerate every /strategies/* endpoint with metadata.",
)
def list_strategies() -> StrategyListResponse:
    """Return the full catalog enumeration."""
    return StrategyListResponse(total=len(STRATEGY_CATALOG), items=list(STRATEGY_CATALOG))


@router.get(
    "/discovery",
    response_model=StrategyListResponse,
    summary="Filter the strategies catalog by tag.",
)
def discover_strategies(
    tag: Annotated[StrategyTag, Query(description="Tag filter; 'all' disables.")] = "all",
) -> StrategyListResponse:
    """Return the catalog filtered by ``tag``."""
    if tag == "all":
        items = list(STRATEGY_CATALOG)
    else:
        items = [s for s in STRATEGY_CATALOG if s.tag == tag]
    return StrategyListResponse(total=len(items), items=items)


__all__ = [
    "STRATEGY_CATALOG",
    "StrategyEntry",
    "StrategyListResponse",
    "StrategyTag",
    "router",
]
