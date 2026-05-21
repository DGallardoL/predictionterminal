"""Side-by-side pairs-trade backtest comparison for the Terminal panel.

Where ``terminal_inline_backtest`` runs *one* slug against its dominant
peer, this module runs *N* strategies — each with its own slug, side,
and z-score / hold-day parameters — over the same time window and
returns:

*   per-strategy diagnostics (Sharpe, hit rate, max DD, equity curve)
*   the pairwise PnL correlation matrix (so the analyst can see which
    strategies actually decorrelate, vs. just look different)
*   a naive equal-weighted combined-portfolio Sharpe + drawdown, which
    is a quick sanity check on whether stacking the strategies helps
    risk-adjusted returns at all (vs. each one solo)

The walk-forward semantics are inherited from :func:`pfm.pairs.pairs_backtest`:
the rolling z-score uses only past data, so there is no look-ahead.

This module owns its own :class:`fastapi.APIRouter` so ``main.py``
only needs the conventional one-line import + ``include_router``::

    from pfm.terminal_backtest_compare import router as terminal_backtest_compare_router
    app.include_router(terminal_backtest_compare_router)

The Polymarket client and the alpha-hunter hits-file path are wired
via the *same* DI dependencies that ``terminal_inline_backtest`` uses,
so a test can patch one set of overrides and have both endpoints
honour them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from pfm.pairs import pairs_backtest
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,
)
from pfm.terminal_inline_backtest import (
    _filter_side,
    _find_dominant_peer,
    _id_to_slug_candidate,
    _load_hits,
    get_hits_path,
    get_polymarket_client,
)

logger = logging.getLogger(__name__)


# --- schemas ----------------------------------------------------------------


class StrategySpec(BaseModel):
    """One strategy in the comparison set.

    Mirrors :class:`pfm.terminal_inline_backtest.InlineBacktestRequest`
    plus the slug-to-test, so each strategy is fully self-describing.
    """

    slug: str = Field(..., min_length=1, max_length=160, description="Polymarket slug.")
    side: Literal["both", "long", "short"] = Field(
        "both", description="Direction filter: long-spread, short-spread, or both."
    )
    entry_z: float = Field(2.0, gt=0.0, le=10.0, description="|z| to open a position.")
    exit_z: float = Field(0.5, ge=0.0, lt=10.0, description="|z| to flatten.")
    stop_z: float = Field(4.0, gt=0.0, le=20.0, description="|z| stop-out.")
    window: int = Field(20, ge=5, le=252, description="Rolling-window for z-score.")
    hold_days: int | None = Field(None, ge=1, le=365, description="Optional time-stop in bars.")


class CompareRequest(BaseModel):
    """Body of POST /terminal/backtest-compare."""

    strategies: list[StrategySpec] = Field(
        ...,
        min_length=2,
        max_length=8,
        description="Two-to-eight strategy specs to backtest side-by-side.",
    )
    days: int = Field(
        180,
        ge=30,
        le=2000,
        description=(
            "Lookback window in calendar days. Each strategy is restricted "
            "to the last `days` of overlapping history."
        ),
    )


class EquityPoint(BaseModel):
    """One {date, equity} point for the comparison sparkline."""

    t: str
    equity: float


class StrategyResult(BaseModel):
    """Per-strategy diagnostics for the compare panel."""

    spec: StrategySpec
    peer_slug: str
    beta_hedge: float
    n_obs: int
    n_trades: int
    sharpe: float
    hit_rate: float
    max_dd: float
    equity_curve: list[EquityPoint]


class CombinedPortfolio(BaseModel):
    """Equal-weighted combined PnL diagnostics."""

    sharpe: float
    dd: float = Field(..., description="Max drawdown of the combined equity curve (≤ 0).")


class CompareResponse(BaseModel):
    """Top-level response: per-strategy + pairwise correlation + combo."""

    strategies: list[StrategyResult]
    correlation: list[list[float]] = Field(
        ..., description="N-by-N pairwise PnL correlation matrix (Pearson)."
    )
    combined_portfolio: CombinedPortfolio


# --- router -----------------------------------------------------------------


router = APIRouter(prefix="/terminal", tags=["terminal-backtest"])


# --- helpers ----------------------------------------------------------------


def _max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough max drawdown (returned as a *negative* number)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    return float((equity - running_max).min())


def _trim_to_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Restrict a date-indexed DataFrame to the last ``days`` calendar days."""
    if df.empty:
        return df
    end = df.index.max()
    start = end - pd.Timedelta(days=days)
    return df.loc[df.index >= start]


def _run_one(
    spec: StrategySpec,
    *,
    poly: PolymarketClient,
    hits: list[dict[str, Any]],
    days: int,
) -> tuple[StrategyResult, pd.Series]:
    """Backtest a single strategy and return its result + per-bar PnL.

    The PnL series is needed to assemble the correlation matrix and the
    combined-portfolio Sharpe; we keep the series in pandas so the join
    on dates handles ragged calendars cleanly.
    """
    if spec.exit_z >= spec.entry_z:
        raise HTTPException(
            status_code=400,
            detail=f"strategy {spec.slug!r}: entry_z must exceed exit_z",
        )
    if spec.stop_z <= spec.entry_z:
        raise HTTPException(
            status_code=400,
            detail=f"strategy {spec.slug!r}: stop_z must exceed entry_z",
        )

    match = _find_dominant_peer(spec.slug, hits)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"no cointegrated peer for slug={spec.slug!r}",
        )
    peer_id, beta = match
    peer_slug = _id_to_slug_candidate(peer_id)

    try:
        df_a = fetch_factor_history(poly, spec.slug)
        df_b = fetch_factor_history(poly, peer_slug)
    except PolymarketError as e:
        raise HTTPException(status_code=502, detail=f"polymarket error: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e

    if df_a.empty or df_b.empty:
        raise HTTPException(
            status_code=502,
            detail=f"empty history for {spec.slug!r} or peer {peer_slug!r}",
        )

    a = df_a["price"].rename("a")
    b = df_b["price"].rename("b")
    a.index = pd.to_datetime(a.index, utc=True).normalize()
    b.index = pd.to_datetime(b.index, utc=True).normalize()
    joined = pd.concat([a, b], axis=1).dropna()
    joined = _trim_to_days(joined, days)
    if len(joined) < spec.window + 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"strategy {spec.slug!r}: too little overlap in last {days}d "
                f"({len(joined)} bars, need ≥ window+5 = {spec.window + 5})"
            ),
        )

    spread = (joined["a"] - beta * joined["b"]).rename("spread")
    try:
        result = pairs_backtest(
            spread,
            window=spec.window,
            entry_z=spec.entry_z,
            exit_z=spec.exit_z,
            stop_z=spec.stop_z,
            max_hold_bars=spec.hold_days,
            oos_fraction=0.0,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"strategy {spec.slug!r}: {e}") from e

    if spec.side != "both":
        positions = _filter_side(result.positions, spec.side)
        dspread = spread.diff().fillna(0.0)
        pnl = positions.shift(1).fillna(0).astype(float) * dspread
        equity = pnl.cumsum()
        trade_pnls = [
            t.pnl
            for t in result.trades
            if (spec.side == "long" and t.direction > 0)
            or (spec.side == "short" and t.direction < 0)
        ]
        n_trades = len(trade_pnls)
        hits_n = sum(1 for p in trade_pnls if p > 0)
        hit_rate = hits_n / n_trades if n_trades else 0.0
        std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
        sharpe = float(pnl.mean()) / std * float(np.sqrt(252.0)) if std > 0 else 0.0
        max_dd = _max_drawdown(equity)
    else:
        pnl = result.pnl
        equity = result.equity_curve
        n_trades = result.n_trades
        sharpe = result.sharpe
        hit_rate = result.hit_rate
        max_dd = result.max_drawdown

    equity_payload = [
        EquityPoint(t=ts.date().isoformat(), equity=float(v))
        for ts, v in equity.items()
        if np.isfinite(v)
    ]

    return (
        StrategyResult(
            spec=spec,
            peer_slug=peer_slug,
            beta_hedge=float(beta),
            n_obs=int(result.n_obs),
            n_trades=int(n_trades),
            sharpe=float(sharpe),
            hit_rate=float(hit_rate),
            max_dd=float(max_dd),
            equity_curve=equity_payload,
        ),
        pnl.rename(spec.slug),
    )


def _correlation_matrix(pnls: list[pd.Series]) -> list[list[float]]:
    """Pairwise Pearson correlation of per-bar PnL series.

    Series are aligned on the union of their dates with NaNs forward-filled
    from zero (no PnL on a missing bar). Diagonal is 1.0; degenerate cases
    (zero-variance leg) yield 0.0 so the JSON stays well-formed.
    """
    if not pnls:
        return []
    df = pd.concat(pnls, axis=1).fillna(0.0)
    n = df.shape[1]
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        out[i][i] = 1.0
        xi = df.iloc[:, i].to_numpy()
        sxi = float(np.std(xi, ddof=1)) if len(xi) > 1 else 0.0
        for j in range(i + 1, n):
            xj = df.iloc[:, j].to_numpy()
            sxj = float(np.std(xj, ddof=1)) if len(xj) > 1 else 0.0
            if sxi <= 0 or sxj <= 0:
                rho = 0.0
            else:
                rho_arr = np.corrcoef(xi, xj)
                rho = float(rho_arr[0, 1]) if np.isfinite(rho_arr[0, 1]) else 0.0
            out[i][j] = rho
            out[j][i] = rho
    return out


def _combined(pnls: list[pd.Series]) -> CombinedPortfolio:
    """Equal-weighted combo Sharpe + max DD on the date-aligned PnL sum."""
    if not pnls:
        return CombinedPortfolio(sharpe=0.0, dd=0.0)
    df = pd.concat(pnls, axis=1).fillna(0.0)
    combo = df.sum(axis=1)
    std = float(combo.std(ddof=1)) if len(combo) > 1 else 0.0
    sharpe = float(combo.mean()) / std * float(np.sqrt(252.0)) if std > 0 else 0.0
    equity = combo.cumsum()
    return CombinedPortfolio(sharpe=sharpe, dd=_max_drawdown(equity))


# --- endpoint ---------------------------------------------------------------


@router.post(
    "/backtest-compare",
    response_model=CompareResponse,
    summary="Compare N pairs-trading strategies side-by-side on the same data.",
)
def run_backtest_compare(
    body: Annotated[CompareRequest, Body()],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
    hits_path: Annotated[Path, Depends(get_hits_path)] = ...,  # type: ignore[assignment]
) -> CompareResponse:
    """Run each strategy through ``pairs_backtest`` and compare them.

    The walk-forward backtest is sequential (each strategy is independent
    of the others), so we just loop. Correlation and combined Sharpe are
    computed on the date-aligned per-bar PnL series.
    """
    hits = _load_hits(hits_path)

    results: list[StrategyResult] = []
    pnl_series: list[pd.Series] = []
    for spec in body.strategies:
        res, pnl = _run_one(spec, poly=poly, hits=hits, days=body.days)
        results.append(res)
        pnl_series.append(pnl)

    correlation = _correlation_matrix(pnl_series)
    combined = _combined(pnl_series)

    return CompareResponse(
        strategies=results,
        correlation=correlation,
        combined_portfolio=combined,
    )


__all__ = [
    "CombinedPortfolio",
    "CompareRequest",
    "CompareResponse",
    "EquityPoint",
    "StrategyResult",
    "StrategySpec",
    "router",
    "run_backtest_compare",
]
