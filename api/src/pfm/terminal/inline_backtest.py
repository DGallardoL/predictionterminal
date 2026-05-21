"""Inline backtester for the Terminal panel.

The Terminal lets the analyst pick *any* Polymarket slug and pre-cooked
``entry_z / exit_z / stop_z / window / hold_days`` thresholds, then
returns a Sharpe / PnL / equity-curve in one round-trip.

Three trading modes are supported:

*   ``pair`` — the original pipeline. Looks up the slug's *dominant
    cointegrated peer* in ``/tmp/ah_sweeps/all_unique_hits.json`` and
    trades the OLS-hedged spread :math:`\\varepsilon = P_A - \\beta P_B`.
    404s if no peer is available.
*   ``rolling_z`` — self-referential mean-reversion on the slug's own
    logit series. The spread is :math:`\\text{logit}(p_t)` minus its
    rolling mean over ``window``; the standard z-score backtest is then
    run on this single-asset series. No peer required.
*   ``bollinger`` — Bollinger-band mean-reversion on the raw probability:
    long when :math:`p_t < \\mu_t - k\\sigma_t`, short when
    :math:`p_t > \\mu_t + k\\sigma_t`, exit on a cross of :math:`\\mu_t`.
    PnL is :math:`\\Delta p_t \\cdot \\text{position}_{t-1}`.
*   ``auto`` (default) — try ``pair`` first; fall back to ``rolling_z``
    if no peer is registered for the slug.

Routing note: this module owns its own :class:`fastapi.APIRouter` so
``main.py`` is left untouched (per project convention).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from pfm.pairs import pairs_backtest
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,
)

logger = logging.getLogger(__name__)

DEFAULT_HITS_PATH: Path = Path("/tmp/ah_sweeps/all_unique_hits.json")

Mode = Literal["pair", "rolling_z", "bollinger", "auto"]


# --- schemas ----------------------------------------------------------------


class InlineBacktestRequest(BaseModel):
    """User-supplied thresholds for the inline backtester.

    Defaults match the Terminal preset: a 20-bar rolling window, ±2σ
    entry, ±0.5σ exit, ±4σ stop, and no time stop. ``side`` filters
    the trade tape to long-only or short-only spreads — useful when
    the analyst has a directional view on the dominant peer.

    ``mode`` selects the strategy. ``auto`` tries the cointegrated
    pair-trade first and silently falls back to ``rolling_z`` when the
    slug has no peer in the alpha-hunter sweep.
    """

    entry_z: float = Field(2.0, gt=0.0, le=10.0, description="|z| to open a position.")
    exit_z: float = Field(0.5, ge=0.0, lt=10.0, description="|z| to flatten.")
    stop_z: float = Field(4.0, gt=0.0, le=20.0, description="|z| stop-out.")
    window: int = Field(20, ge=5, le=252, description="Rolling-window for z-score / band.")
    hold_days: int | None = Field(
        None,
        ge=1,
        le=365,
        description="Optional time stop in bars (force-close after N days).",
    )
    side: Literal["both", "long", "short"] = Field(
        "both",
        description="Restrict trades to long-spread, short-spread, or both.",
    )
    mode: Mode = Field(
        "auto",
        description=(
            "Strategy selector. ``pair`` requires a cointegrated peer "
            "in the alpha-hunter sweep; ``rolling_z`` and ``bollinger`` "
            "are self-referential fallbacks; ``auto`` tries pair first."
        ),
    )
    bollinger_k: float = Field(
        2.0,
        gt=0.0,
        le=10.0,
        description="Bollinger-band width in sigmas (only used when mode='bollinger').",
    )


class EquityPoint(BaseModel):
    """One {date, equity} point for the frontend sparkline."""

    t: str
    equity: float


class InlineBacktestResponse(BaseModel):
    slug: str
    mode_used: Mode = Field(
        ...,
        description="The strategy that actually ran (pair / rolling_z / bollinger).",
    )
    peer_slug: str | None = Field(
        None,
        description="Dominant cointegrated peer (only set in pair mode).",
    )
    beta_hedge: float | None = Field(
        None,
        description="OLS hedge ratio reused from the sweep (only set in pair mode).",
    )
    n_obs: int
    n_trades: int
    sharpe: float
    hit_rate: float
    max_dd: float
    calmar: float
    side: Literal["both", "long", "short"]
    equity_curve: list[EquityPoint]
    trade_pnls: list[float] = Field(
        ...,
        description="Per-trade PnL series (chronological, dollar PnL on a unit spread).",
    )
    note: str | None = Field(
        None,
        description="Human-readable annotation (e.g. why a fallback mode was used).",
    )


# --- router -----------------------------------------------------------------


router = APIRouter(prefix="/terminal", tags=["terminal-backtest"])

# Module-level default so the endpoint signature avoids B008
# (function-call in argument default). The Pydantic model is treated
# as frozen by the handlers — no in-place mutation.
_DEFAULT_REQUEST = InlineBacktestRequest()


def get_polymarket_client() -> PolymarketClient:
    """Resolve the shared :class:`PolymarketClient` from app state.

    Imported lazily so this module doesn't drag ``pfm.main`` into its
    import graph (would create a cycle: main → terminal_inline_backtest
    → main).
    """
    from pfm.main import app  # local import to break the cycle

    return app.state.poly


def get_hits_path() -> Path:
    """Return the path to the alpha-hunter unique-hits JSON.

    Wrapped in a dependency so tests can override it with a temp file.
    """
    return DEFAULT_HITS_PATH


# --- helpers ----------------------------------------------------------------


_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LOGIT_EPS = 1e-4  # safety clip for logit() to avoid ±inf at p∈{0,1}.


def _slug_to_id(slug: str) -> str:
    """Approximate the alpha-hunter sweep ``a_id`` form for a Polymarket slug."""
    s = _NON_ALNUM.sub("_", slug.lower()).strip("_")
    return s


def _find_dominant_peer(slug: str, hits: list[dict[str, Any]]) -> tuple[str, float] | None:
    """Find the highest-Sharpe peer for ``slug`` in the sweep."""
    target = _slug_to_id(slug)
    if not target:
        return None

    best: tuple[str, float, float] | None = None  # (peer, beta, score)
    for row in hits:
        a_id = str(row.get("a_id", ""))
        b_id = str(row.get("b_id", ""))
        score = float(row.get("oos_sharpe", 0.0) or 0.0)
        beta = float(row.get("beta_hedge", 0.0) or 0.0)

        if a_id == target or a_id.startswith(target + "_") or target.startswith(a_id + "_"):
            peer, peer_beta = b_id, beta
        elif b_id == target or b_id.startswith(target + "_") or target.startswith(b_id + "_"):
            peer = a_id
            peer_beta = 1.0 / beta if abs(beta) > 1e-6 else 0.0
        else:
            continue

        if peer and (best is None or score > best[2]):
            best = (peer, peer_beta, score)

    if best is None:
        return None
    return best[0], best[1]


def _id_to_slug_candidate(peer_id: str) -> str:
    """Convert a sweep ``a_id`` back to a Polymarket-slug guess."""
    return peer_id.replace("_", "-").lower()


def _filter_side(positions: pd.Series, side: str) -> pd.Series:
    """Mask the position series to enforce a one-sided trade rule."""
    if side == "long":
        return positions.where(positions > 0, 0).astype(int)
    if side == "short":
        return positions.where(positions < 0, 0).astype(int)
    return positions


def _logit(p: pd.Series) -> pd.Series:
    """Stable logit transform with a fixed clip at ε=1e-4."""
    pc = p.clip(lower=_LOGIT_EPS, upper=1.0 - _LOGIT_EPS)
    return np.log(pc / (1.0 - pc))


def _load_hits(path: Path) -> list[dict[str, Any]]:
    """Read the alpha-hunter unique-hits JSON.

    Returns an empty list (instead of raising 503) when the sweep artifact is
    missing or malformed; callers in ``auto`` mode will then degrade to
    ``rolling_z`` and ``pair`` mode will surface a 404 ("no cointegrated peer
    in sweep"). Run ``scripts/alpha_hunter_sweep.py`` to populate.
    """
    if not path.exists():
        logger.warning(
            "alpha-hunter hits file not found at %s; degrading pair-mode to no-peer",
            path,
        )
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("could not load alpha-hunter hits at %s: %s", path, e)
        return []
    if not isinstance(data, list):
        logger.warning(
            "alpha-hunter hits file at %s is not a JSON list; treating as empty",
            path,
        )
        return []
    return data


# --- shared metric helpers --------------------------------------------------


def _equity_to_payload(equity: pd.Series) -> list[EquityPoint]:
    """Serialise a {Timestamp → float} equity series for the frontend."""
    return [
        EquityPoint(t=ts.date().isoformat(), equity=float(v))
        for ts, v in equity.items()
        if np.isfinite(v)
    ]


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    return float((equity - running_max).min())


# --- mode runners -----------------------------------------------------------


def _run_pair_mode(
    slug: str,
    body: InlineBacktestRequest,
    poly: PolymarketClient,
    hits: list[dict[str, Any]],
) -> InlineBacktestResponse:
    """Run the original cointegrated-peer pairs backtest. Raises 404 if
    no peer exists in the sweep for ``slug``."""
    match = _find_dominant_peer(slug, hits)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"no cointegrated peer for slug={slug!r} in alpha-hunter sweep",
        )
    peer_id, beta = match
    peer_slug = _id_to_slug_candidate(peer_id)

    try:
        df_a = fetch_factor_history(poly, slug)
        df_b = fetch_factor_history(poly, peer_slug)
    except PolymarketError as e:
        raise HTTPException(status_code=502, detail=f"polymarket error: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e

    if df_a.empty or df_b.empty:
        raise HTTPException(
            status_code=502,
            detail=f"empty history for {slug!r} or peer {peer_slug!r}",
        )

    a = df_a["price"].rename("a")
    b = df_b["price"].rename("b")
    a.index = pd.to_datetime(a.index, utc=True).normalize()
    b.index = pd.to_datetime(b.index, utc=True).normalize()
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < body.window + 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"too little overlapping history: have {len(joined)} bars, "
                f"need at least window+5 = {body.window + 5}"
            ),
        )

    spread = (joined["a"] - beta * joined["b"]).rename("spread")

    try:
        result = pairs_backtest(
            spread,
            window=body.window,
            entry_z=body.entry_z,
            exit_z=body.exit_z,
            stop_z=body.stop_z,
            max_hold_bars=body.hold_days,
            oos_fraction=0.0,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    equity, n_trades, trade_pnls, sharpe, hit_rate, max_dd, calmar = _apply_side_filter(
        body.side, result, spread
    )

    return InlineBacktestResponse(
        slug=slug,
        mode_used="pair",
        peer_slug=peer_slug,
        beta_hedge=float(beta),
        n_obs=int(result.n_obs),
        n_trades=int(n_trades),
        sharpe=float(sharpe),
        hit_rate=float(hit_rate),
        max_dd=float(max_dd),
        calmar=float(calmar),
        side=body.side,
        equity_curve=_equity_to_payload(equity),
        trade_pnls=[float(p) for p in trade_pnls],
        note=None,
    )


def _apply_side_filter(
    side: str,
    result: Any,
    spread: pd.Series,
) -> tuple[pd.Series, int, list[float], float, float, float, float]:
    """Recompute equity / metrics when ``side`` masks one direction.

    Returns ``(equity, n_trades, trade_pnls, sharpe, hit_rate, max_dd, calmar)``.
    """
    if side == "both":
        return (
            result.equity_curve,
            result.n_trades,
            [t.pnl for t in result.trades],
            result.sharpe,
            result.hit_rate,
            result.max_drawdown,
            result.calmar,
        )

    positions = _filter_side(result.positions, side)
    dspread = spread.diff().fillna(0.0)
    pnl = positions.shift(1).fillna(0).astype(float) * dspread
    equity = pnl.cumsum()
    trade_pnls = [
        t.pnl
        for t in result.trades
        if ((side == "long" and t.direction > 0) or (side == "short" and t.direction < 0))
    ]
    n_trades = len(trade_pnls)
    hits_n = sum(1 for p in trade_pnls if p > 0)
    hit_rate = hits_n / n_trades if n_trades else 0.0
    std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = float(pnl.mean()) / std * float(np.sqrt(252.0)) if std > 0 else 0.0
    max_dd = _max_drawdown(equity)
    calmar = float(pnl.mean()) * 252.0 / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0
    return equity, n_trades, trade_pnls, sharpe, hit_rate, max_dd, calmar


def _run_rolling_z_mode(
    slug: str,
    body: InlineBacktestRequest,
    poly: PolymarketClient,
    *,
    note: str | None = None,
) -> InlineBacktestResponse:
    """Self-referential mean-reversion on logit(p) − rolling-mean(logit(p)).

    The "spread" passed to :func:`pairs_backtest` is the demeaned logit
    of the slug's own probability series. The same ``window`` is used
    for the inner z-score that drives entry/exit — this gives an
    interpretable z-of-z (mean-reversion strength relative to recent
    volatility), which is the conventional "rolling z" strategy.
    """
    try:
        df = fetch_factor_history(poly, slug)
    except PolymarketError as e:
        raise HTTPException(status_code=502, detail=f"polymarket error: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e

    if df.empty:
        raise HTTPException(status_code=502, detail=f"empty history for {slug!r}")

    p = df["price"].copy()
    p.index = pd.to_datetime(p.index, utc=True).normalize()
    p = p.dropna().sort_index()
    if len(p) < body.window + 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"too little history: have {len(p)} bars, "
                f"need at least window+5 = {body.window + 5}"
            ),
        )

    lp = _logit(p)
    mu = lp.rolling(window=body.window, min_periods=max(5, body.window // 2)).mean()
    spread = (lp - mu).rename("spread").dropna()

    if len(spread) < body.window + 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"too little post-rolling-mean history: have {len(spread)} bars, "
                f"need at least window+5 = {body.window + 5}"
            ),
        )

    try:
        result = pairs_backtest(
            spread,
            window=body.window,
            entry_z=body.entry_z,
            exit_z=body.exit_z,
            stop_z=body.stop_z,
            max_hold_bars=body.hold_days,
            oos_fraction=0.0,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    equity, n_trades, trade_pnls, sharpe, hit_rate, max_dd, calmar = _apply_side_filter(
        body.side, result, spread
    )

    return InlineBacktestResponse(
        slug=slug,
        mode_used="rolling_z",
        peer_slug=None,
        beta_hedge=None,
        n_obs=int(result.n_obs),
        n_trades=int(n_trades),
        sharpe=float(sharpe),
        hit_rate=float(hit_rate),
        max_dd=float(max_dd),
        calmar=float(calmar),
        side=body.side,
        equity_curve=_equity_to_payload(equity),
        trade_pnls=[float(p) for p in trade_pnls],
        note=note,
    )


def _run_bollinger_mode(
    slug: str,
    body: InlineBacktestRequest,
    poly: PolymarketClient,
    *,
    note: str | None = None,
) -> InlineBacktestResponse:
    """Bollinger-band mean-reversion on the raw probability series.

    State machine:
        flat → long  when p < μ − k·σ  (expect bounce up to μ)
        flat → short when p > μ + k·σ  (expect drift down to μ)
        long  → flat when p ≥ μ        (target hit)
        short → flat when p ≤ μ        (target hit)
        any   → flat when ``hold_days`` bars elapse (optional time stop).

    Per-bar PnL is :math:`\\Delta p_t \\cdot \\text{position}_{t-1}`, which
    matches the unit-of-spread convention used by the pair backtest so
    the equity curves are directly comparable.
    """
    try:
        df = fetch_factor_history(poly, slug)
    except PolymarketError as e:
        raise HTTPException(status_code=502, detail=f"polymarket error: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e

    if df.empty:
        raise HTTPException(status_code=502, detail=f"empty history for {slug!r}")

    p = df["price"].copy()
    p.index = pd.to_datetime(p.index, utc=True).normalize()
    p = p.dropna().sort_index()
    if len(p) < body.window + 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"too little history: have {len(p)} bars, "
                f"need at least window+5 = {body.window + 5}"
            ),
        )

    minp = max(5, body.window // 2)
    mu = p.rolling(window=body.window, min_periods=minp).mean()
    sd = p.rolling(window=body.window, min_periods=minp).std(ddof=1)
    upper = mu + body.bollinger_k * sd
    lower = mu - body.bollinger_k * sd

    n = len(p)
    pos_arr = np.zeros(n, dtype=int)
    state = 0
    held = 0
    entry_idx = -1
    trades: list[tuple[int, int, int]] = []  # (entry_i, exit_i, direction)
    p_arr = p.to_numpy()
    mu_arr = mu.to_numpy()
    upper_arr = upper.to_numpy()
    lower_arr = lower.to_numpy()
    for i in range(n):
        pi = p_arr[i]
        mi = mu_arr[i]
        ui = upper_arr[i]
        li = lower_arr[i]
        # If bands not yet defined, stay flat.
        if not (np.isfinite(mi) and np.isfinite(ui) and np.isfinite(li)):
            pos_arr[i] = state
            if state != 0:
                held += 1
            continue
        if state == 0:
            if pi < li:
                state = 1
                entry_idx = i
                held = 0
            elif pi > ui:
                state = -1
                entry_idx = i
                held = 0
        elif state == 1:
            # exit on cross of mean upward, or time stop
            if pi >= mi or (body.hold_days is not None and held >= body.hold_days):
                trades.append((entry_idx, i, 1))
                state = 0
                held = 0
                entry_idx = -1
        elif state == -1 and (pi <= mi or (body.hold_days is not None and held >= body.hold_days)):
            trades.append((entry_idx, i, -1))
            state = 0
            held = 0
            entry_idx = -1
        pos_arr[i] = state
        if state != 0:
            held += 1

    # Close any open trade at end of data so trade_pnls are complete.
    if state != 0 and entry_idx >= 0:
        trades.append((entry_idx, n - 1, state))

    pos = pd.Series(pos_arr, index=p.index, name="position")
    if body.side != "both":
        pos = _filter_side(pos, body.side)

    dprice = p.diff().fillna(0.0)
    pnl = (pos.shift(1).fillna(0).astype(float) * dprice).rename("pnl")
    equity = pnl.cumsum().rename("equity")

    # Per-trade PnL using Δprice × direction over the holding window.
    trade_pnls: list[float] = []
    for ei, xi, direction in trades:
        if body.side == "long" and direction <= 0:
            continue
        if body.side == "short" and direction >= 0:
            continue
        trade_pnls.append(float(direction * (p_arr[xi] - p_arr[ei])))
    n_trades = len(trade_pnls)
    hit_rate = sum(1 for x in trade_pnls if x > 0) / n_trades if n_trades else 0.0
    std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = float(pnl.mean()) / std * float(np.sqrt(252.0)) if std > 0 else 0.0
    max_dd = _max_drawdown(equity)
    calmar = float(pnl.mean()) * 252.0 / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    return InlineBacktestResponse(
        slug=slug,
        mode_used="bollinger",
        peer_slug=None,
        beta_hedge=None,
        n_obs=len(p),
        n_trades=int(n_trades),
        sharpe=float(sharpe),
        hit_rate=float(hit_rate),
        max_dd=float(max_dd),
        calmar=float(calmar),
        side=body.side,
        equity_curve=_equity_to_payload(equity),
        trade_pnls=trade_pnls,
        note=note,
    )


# --- endpoint ---------------------------------------------------------------


@router.post(
    "/backtest/{slug}",
    response_model=InlineBacktestResponse,
    summary="Inline mean-reversion backtest (pair / rolling-z / bollinger).",
)
def run_inline_backtest(
    slug: Annotated[str, PathParam(min_length=1, max_length=160)],
    body: Annotated[InlineBacktestRequest, Body()] = _DEFAULT_REQUEST,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
    hits_path: Annotated[Path, Depends(get_hits_path)] = ...,  # type: ignore[assignment]
) -> InlineBacktestResponse:
    """Run a backtest of ``slug`` in one of three modes.

    *   ``pair`` — cointegrated-peer spread (404 if no peer registered).
    *   ``rolling_z`` — self-referential logit-rolling-mean reversion.
    *   ``bollinger`` — Bollinger-band reversion on raw probability.
    *   ``auto`` (default) — pair if available, else ``rolling_z``.
    """
    if body.exit_z >= body.entry_z:
        raise HTTPException(status_code=400, detail="entry_z must exceed exit_z")
    if body.stop_z <= body.entry_z:
        raise HTTPException(status_code=400, detail="stop_z must exceed entry_z")

    if body.mode == "bollinger":
        return _run_bollinger_mode(slug, body, poly)

    if body.mode == "rolling_z":
        return _run_rolling_z_mode(slug, body, poly)

    # pair / auto: both need the hits file.
    hits = _load_hits(hits_path)
    match = _find_dominant_peer(slug, hits)

    if body.mode == "pair":
        if match is None:
            raise HTTPException(
                status_code=404,
                detail=f"no cointegrated peer for slug={slug!r} in alpha-hunter sweep",
            )
        return _run_pair_mode(slug, body, poly, hits)

    # auto
    if match is not None:
        return _run_pair_mode(slug, body, poly, hits)
    return _run_rolling_z_mode(
        slug,
        body,
        poly,
        note="No cointegrated peer; using rolling-z mean-reversion on self.",
    )


__all__ = [
    "EquityPoint",
    "InlineBacktestRequest",
    "InlineBacktestResponse",
    "Mode",
    "get_hits_path",
    "get_polymarket_client",
    "router",
]
