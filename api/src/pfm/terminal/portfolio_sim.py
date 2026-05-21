"""Portfolio-simulator endpoint for the Terminal panel.

A user pins a basket of Polymarket positions — each is a ``(slug, side,
size_usd)`` triple where ``side`` is ``"YES"`` or ``"NO"``. The endpoint
fetches the daily YES-token price history for every slug, builds the
position-level daily PnL series, aggregates them into a single portfolio
PnL series, and returns a small bundle of risk and concentration
diagnostics.

Methodology
-----------
- For each position, daily PnL is::

      pnl_t = size_usd * (p_t - p_{t-1})       if side == YES
      pnl_t = size_usd * -(p_t - p_{t-1})      if side == NO

  (Polymarket binary contracts settle in [0, 1], so ``size_usd × Δp``
  is the dollar mark-to-market change for one full $-of-notional unit.)

- Portfolio PnL is the cross-sectional sum across positions on each
  date (after aligning indices to the union of dates and forward-filling
  gaps in the underlying prices).

- ``sharpe_estimate_via_history`` is the *annualised* Sharpe of the
  portfolio's daily PnL series (normalised by gross notional so it is
  comparable across portfolios of different sizes), using the standard
  ``√252`` scale factor.

- ``max_drawdown`` is computed on the cumulative-PnL equity curve as a
  fraction of the gross notional: ``min_t (cum_t - running_max_t) /
  gross_notional``.

- ``expected_payoff_usd_at_resolution`` assumes resolution in the
  direction implied by the *latest* observed price: a YES position at
  ``p`` has expected payoff ``size_usd * p`` if ``p ≥ 0.5`` else
  ``size_usd * (1 - p)`` for the NO case (this is a deliberately naive
  marker — the user should read it as "if today's price is the true
  probability, what does the book pay out at resolution").

- ``current_book_pnl_usd`` is the unrealised mark-to-market PnL: for
  each position, ``size_usd * (latest_price - first_observed_price)``
  with the YES/NO sign.

- ``position_correlation_matrix`` is the Pearson correlation between
  position daily-PnL series, computed on the joint dropna so signals
  with different histories degrade gracefully.

- ``recommended_hedge`` runs a tiny heuristic: if the gross sum of
  absolute position betas (proxied by absolute PnL contribution share)
  exceeds ``HEDGE_NOTIONAL_LIMIT`` of gross notional, suggest the
  most negatively-correlated peer slug already in the book as a hedge
  candidate. If no negative correlation exists, the list is empty.

Routing
-------
This module owns its :class:`fastapi.APIRouter`. ``main.py`` is left
untouched — wire it in explicitly via::

    from pfm.terminal_portfolio_sim import router as portfolio_sim_router
    app.include_router(portfolio_sim_router)
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Literal

import httpx
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,
)
from pfm.terminal_export import respond as _export_respond

logger = logging.getLogger(__name__)


# Annualisation factor for daily-frequency Sharpe.
TRADING_DAYS_PER_YEAR: int = 252

# If gross |contribution| share of any single position exceeds this fraction
# of the book, the hedge heuristic kicks in.
HEDGE_NOTIONAL_LIMIT: float = 0.50

# Maximum positions a single request may carry (defensive — prevents a user
# from accidentally fanning out 1000 Polymarket fetches in one call).
MAX_POSITIONS: int = 25


# --- request / response schemas --------------------------------------------


class Position(BaseModel):
    """One pinned book entry: a directional bet on a Polymarket slug."""

    slug: str = Field(..., min_length=1, max_length=200)
    side: Literal["YES", "NO"]
    size_usd: float = Field(..., gt=0.0, description="Notional in USD; must be > 0.")


class PortfolioSimRequest(BaseModel):
    positions: list[Position] = Field(..., min_length=1, max_length=MAX_POSITIONS)
    days: int = Field(180, ge=20, le=730)


class HedgeSuggestion(BaseModel):
    slug: str
    size: float


# --- helpers ----------------------------------------------------------------


def _coerce_finite(x: object) -> float | None:
    """JSON-safe float coercion (``None`` for NaN / inf / non-numeric)."""
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _position_key(idx: int, pos: Position) -> str:
    """Stable label for matrix / hedge output that disambiguates duplicates."""
    return f"{idx}:{pos.side}:{pos.slug}"


def _fetch_price_series(
    poly: PolymarketClient,
    slug: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Pull one slug's price column and return a UTC-normalised pd.Series."""
    df = fetch_factor_history(poly, slug, start=start, end=end)
    if df is None or df.empty or "price" not in df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"no price history for slug={slug!r}",
        )
    s = df["price"].astype(float)
    s.index = pd.to_datetime(s.index, utc=True).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _position_pnl(
    price: pd.Series,
    size_usd: float,
    side: str,
) -> pd.Series:
    """Daily mark-to-market PnL series for one position."""
    sign = 1.0 if side == "YES" else -1.0
    return sign * size_usd * price.diff()


def _max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline of a cumulative-PnL series.

    Returns a non-positive number (or 0 for monotone curves / empties).
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity - running_max
    val = float(dd.min())
    return val if math.isfinite(val) else 0.0


def _correlation_matrix(pnl_frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Pairwise Pearson correlation, JSON-safe."""
    if pnl_frame.shape[1] == 0:
        return {}
    if pnl_frame.shape[1] == 1:
        col = pnl_frame.columns[0]
        return {col: {col: 1.0}}
    corr = pnl_frame.corr(method="pearson", min_periods=3)
    out: dict[str, dict[str, float | None]] = {}
    for c in corr.columns:
        row: dict[str, float | None] = {}
        for r in corr.index:
            v = corr.at[r, c]
            row[r] = _coerce_finite(v)
        out[c] = row
    return out


def _recommend_hedge(
    positions: list[Position],
    pnl_frame: pd.DataFrame,
    gross_notional: float,
) -> list[dict[str, Any]]:
    """Heuristic hedge suggestion.

    If any single position's |PnL contribution| exceeds
    ``HEDGE_NOTIONAL_LIMIT`` of the book's gross |contribution|, look
    inside the book for the peer with the *most negative* correlation
    to that dominant position. If no correlation < 0 exists, return [].

    The returned size is a soft target: the dominant position's notional
    scaled by ``|corr|`` — i.e. a beta-matched offset.
    """
    if pnl_frame.empty or pnl_frame.shape[1] < 2:
        return []

    abs_contrib = pnl_frame.abs().sum()
    total = float(abs_contrib.sum())
    if total <= 0 or not math.isfinite(total):
        return []

    share = abs_contrib / total
    dominant_label = str(share.idxmax())
    if float(share.max()) < HEDGE_NOTIONAL_LIMIT:
        return []

    # Find the dominant position's index (label is "idx:side:slug").
    try:
        dom_idx = int(dominant_label.split(":", 1)[0])
    except ValueError:
        return []
    dom_pos = positions[dom_idx]

    corr = pnl_frame.corr(method="pearson", min_periods=3)
    if dominant_label not in corr.columns:
        return []
    row = corr[dominant_label].drop(labels=[dominant_label], errors="ignore")
    if row.empty:
        return []

    most_neg_label = str(row.idxmin())
    most_neg_corr = _coerce_finite(row.min())
    if most_neg_corr is None or most_neg_corr >= 0.0:
        return []

    # Beta-matched hedge size: dominant notional * |corr|.
    hedge_size = float(dom_pos.size_usd) * abs(most_neg_corr)
    try:
        hedge_idx = int(most_neg_label.split(":", 1)[0])
    except ValueError:
        return []
    hedge_pos = positions[hedge_idx]

    return [
        {
            "slug": hedge_pos.slug,
            "size": _coerce_finite(hedge_size) or 0.0,
        }
    ]


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-portfolio-sim"])


def get_polymarket_client() -> PolymarketClient:
    """Resolve the shared client from app state. Lazy import avoids cycles."""
    from pfm.main import app  # local import to avoid circulars

    return app.state.poly


@router.post("/portfolio-sim", response_model=None)
def post_portfolio_sim(
    body: Annotated[PortfolioSimRequest, Body(...)],
    *,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    format: Annotated[Literal["json", "csv", "pdf"], Query()] = "json",
) -> dict[str, Any] | FastAPIResponse:
    """Simulate a multi-position Polymarket book over a daily history window.

    Aggregates the daily mark-to-market PnL across all pinned positions
    and surfaces book-level risk metrics: Sharpe, max drawdown, the
    pairwise position-PnL correlation matrix, and an optional hedge
    suggestion if any single position dominates the gross book.
    """
    if not body.positions:
        # Pydantic min_length=1 already enforces this, but fail loud.
        raise HTTPException(status_code=400, detail="positions list is empty")

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=body.days + 2)

    # --- fetch all price series --------------------------------------------
    price_series_by_label: dict[str, pd.Series] = {}
    for idx, pos in enumerate(body.positions):
        label = _position_key(idx, pos)
        try:
            series = _fetch_price_series(poly, pos.slug, start=start, end=end)
        except HTTPException:
            raise
        except PolymarketError as e:
            raise HTTPException(status_code=404, detail=f"unknown slug: {e}") from e
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e
        # Trim to the requested window.
        series = series[series.index >= (end - pd.Timedelta(days=body.days))]
        price_series_by_label[label] = series

    # --- align on the union of dates (forward-fill within each series) -----
    all_dates = sorted({d for s in price_series_by_label.values() for d in s.index})
    if not all_dates:
        raise HTTPException(
            status_code=422,
            detail="no overlapping price history across positions",
        )
    union_idx = pd.DatetimeIndex(all_dates)

    pnl_columns: dict[str, pd.Series] = {}
    current_pnl_per_pos: dict[str, float] = {}
    expected_payoff_per_pos: dict[str, float] = {}
    latest_price_per_pos: dict[str, float] = {}

    for idx, pos in enumerate(body.positions):
        label = _position_key(idx, pos)
        s = price_series_by_label[label].reindex(union_idx).ffill()
        pnl_columns[label] = _position_pnl(s, pos.size_usd, pos.side)

        first = s.dropna()
        if first.empty:
            current_pnl_per_pos[label] = 0.0
            expected_payoff_per_pos[label] = 0.0
            latest_price_per_pos[label] = float("nan")
            continue
        p_first = float(first.iloc[0])
        p_last = float(first.iloc[-1])
        latest_price_per_pos[label] = p_last
        sign = 1.0 if pos.side == "YES" else -1.0
        current_pnl_per_pos[label] = sign * pos.size_usd * (p_last - p_first)
        # Naive expected-payoff at resolution: assume the side wins iff its
        # current implied probability >= 0.5; payout is size_usd × prob_of_win.
        prob_win = p_last if pos.side == "YES" else (1.0 - p_last)
        expected_payoff_per_pos[label] = pos.size_usd * prob_win

    pnl_frame = pd.DataFrame(pnl_columns)
    portfolio_pnl = pnl_frame.sum(axis=1, skipna=True)

    # --- aggregate metrics --------------------------------------------------
    gross_notional = float(sum(p.size_usd for p in body.positions))

    daily = portfolio_pnl.dropna()
    if len(daily) >= 2 and daily.std(ddof=1) > 0:
        mean_daily = float(daily.mean())
        sd_daily = float(daily.std(ddof=1))
        sharpe = (mean_daily / sd_daily) * math.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = float("nan")

    cum_pnl = portfolio_pnl.fillna(0.0).cumsum()
    raw_dd = _max_drawdown(cum_pnl)
    # Express as fraction of gross notional (negative number).
    max_dd = raw_dd / gross_notional if gross_notional > 0 else raw_dd

    corr_matrix = _correlation_matrix(pnl_frame)
    hedge = _recommend_hedge(body.positions, pnl_frame, gross_notional)

    response: dict[str, Any] = {
        "n_positions": len(body.positions),
        "gross_notional_usd": _coerce_finite(gross_notional) or 0.0,
        "expected_payoff_usd_at_resolution": _coerce_finite(sum(expected_payoff_per_pos.values()))
        or 0.0,
        "current_book_pnl_usd": _coerce_finite(sum(current_pnl_per_pos.values())) or 0.0,
        "sharpe_estimate_via_history": _coerce_finite(sharpe),
        "max_drawdown": _coerce_finite(max_dd) or 0.0,
        "position_correlation_matrix": corr_matrix,
        "recommended_hedge": hedge,
        "lookback_days": int(body.days),
        "n_observations": len(daily),
    }
    if format == "json":
        return response
    return _export_respond(
        response,
        format,
        filename="portfolio-sim",
        kind="portfolio",
    )


__all__ = [
    "HEDGE_NOTIONAL_LIMIT",
    "MAX_POSITIONS",
    "TRADING_DAYS_PER_YEAR",
    "HedgeSuggestion",
    "PortfolioSimRequest",
    "Position",
    "get_polymarket_client",
    "post_portfolio_sim",
    "router",
]
