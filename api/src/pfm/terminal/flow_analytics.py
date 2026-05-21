"""Trade-flow analytics extending the Polymarket terminal trade tape.

This module exposes a single endpoint:

    GET /terminal/flow/{slug}?window_minutes=60

It reuses :mod:`pfm.terminal_trades` to fetch and Lee-Ready-classify the
trade tape, then folds the classified trades into aggressive-flow /
informed-flow style metrics over a configurable lookback window.

Definitions (all evaluated over the trailing ``window_minutes`` window
ending at the most recent trade):

  - ``buy_ratio``: fraction of trades classified BUY (excluding AT_MID).
  - ``aggressive_ratio``: fraction of trades that crossed the spread —
    i.e. price ≥ ask (BUY) or price ≤ bid (SELL). When quotes are
    missing we fall back to "moved the tape" (price strictly higher
    than the prior print for BUY, strictly lower for SELL).
  - ``net_flow_usd``: notional_buy - notional_sell, with notional
    approximated as ``price * size`` (both in USDC for Polymarket).
  - ``bursts``: V-PIN-style activity bumps. We count trades in
    one-minute buckets, take the rolling mean over the window, and
    flag any bucket with count > 2 × that mean.
  - ``informed_flow_signal``: ``BUY`` / ``SELL`` / ``NEUTRAL``. Trips
    only when ``aggressive_ratio > 0.6`` AND the net flow agrees
    with the dominant aggressive side (so we don't fire on noisy
    two-sided action).

The route is intentionally read-only and side-effect free; the heavy
lifting (HTTP + classification) is delegated to
:func:`pfm.terminal_trades.get_terminal_trades`.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_trades import (
    TradeTick,
    get_polymarket_client,
    get_terminal_trades,
)

# Flow metrics are pure aggregates over the trade tape, which already has a
# 5 s cache in trades.py. Caching the assembled flow response on
# (slug, window_minutes) for 5 s eliminates the pandas resampling +
# bucketing work on rapid re-polls from the market-detail UI.
_FLOW_CACHE = get_cache("terminal_flow", ttl=5)

AGGRESSIVE_THRESHOLD: float = 0.6
BURST_MULTIPLIER: float = 2.0
BURST_BUCKET: str = "1min"

InformedSignal = Literal["BUY", "SELL", "NEUTRAL"]

router = APIRouter(prefix="/terminal", tags=["terminal"])


# --- schemas ----------------------------------------------------------------


class TopTrade(BaseModel):
    """One of the top-N largest trades by notional USD."""

    timestamp: str
    price: float = Field(..., ge=0.0, le=1.0)
    size: float = Field(..., ge=0.0)
    notional_usd: float = Field(..., ge=0.0)
    side: Literal["BUY", "SELL", "AT_MID"]


class Burst(BaseModel):
    """A one-minute bucket flagged as a trade-count burst."""

    timestamp: str
    n_trades: int = Field(..., ge=0)
    magnitude: float = Field(
        ...,
        description="Bucket count divided by the rolling mean count.",
    )


class TerminalFlowResponse(BaseModel):
    slug: str
    window_minutes: int
    n_trades_total: int
    n_trades_buy: int
    n_trades_sell: int
    buy_ratio: float = Field(..., ge=0.0, le=1.0)
    aggressive_ratio: float = Field(..., ge=0.0, le=1.0)
    notional_buy_usd: float = Field(..., ge=0.0)
    notional_sell_usd: float = Field(..., ge=0.0)
    net_flow_usd: float
    largest_trade_usd: float = Field(..., ge=0.0)
    top_5_trades: list[TopTrade]
    bursts: list[Burst]
    informed_flow_signal: InformedSignal


# --- pure helpers -----------------------------------------------------------


def _filter_window(trades: list[TradeTick], window_minutes: int) -> list[TradeTick]:
    """Keep only trades within the trailing ``window_minutes`` ending at the last print."""
    if not trades:
        return []
    ts = pd.to_datetime([t.timestamp for t in trades], utc=True)
    end = ts.max()
    start = end - pd.Timedelta(minutes=window_minutes)
    return [t for t, x in zip(trades, ts, strict=True) if x >= start]


def _is_aggressive(t: TradeTick, prev_price: float | None) -> bool:
    """Did this trade cross the spread / take the tape?

    Polymarket's data-api rarely gives us bid/ask on historical trades, so
    we approximate aggression with the tick relative to the prior print:
    a BUY that printed strictly higher is lifting the offer, a SELL that
    printed strictly lower is hitting the bid.
    """
    if prev_price is None:
        return False
    if t.side == "BUY":
        return t.price > prev_price
    if t.side == "SELL":
        return t.price < prev_price
    return False


def detect_bursts(trades: list[TradeTick]) -> list[Burst]:
    """Bucket trades into 1-min bins; flag bins where count > 2× rolling mean.

    Easley-O'Hara V-PIN flags "toxic" intervals by abnormal trade
    intensity. We approximate that here by counting trades per minute,
    taking a rolling mean across the full sample, and emitting a burst
    whenever a bucket exceeds ``BURST_MULTIPLIER`` times that baseline.
    """
    if not trades:
        return []
    df = (
        pd.DataFrame({"ts": pd.to_datetime([t.timestamp for t in trades], utc=True)})
        .set_index("ts")
        .sort_index()
    )
    df["count"] = 1
    counts = df["count"].resample(BURST_BUCKET).sum()
    if counts.empty:
        return []
    mean_count = float(counts.mean())
    if mean_count <= 0:
        return []
    threshold = BURST_MULTIPLIER * mean_count
    out: list[Burst] = []
    for ts, n in counts.items():
        n_int = int(n)
        if n_int > threshold:
            out.append(
                Burst(
                    timestamp=ts.isoformat().replace("+00:00", "Z"),
                    n_trades=n_int,
                    magnitude=n_int / mean_count,
                )
            )
    return out


def _informed_signal(
    aggressive_ratio: float,
    net_flow_usd: float,
    notional_buy: float,
    notional_sell: float,
) -> InformedSignal:
    """Direction-agreement gate on the aggressive-flow ratio."""
    if aggressive_ratio <= AGGRESSIVE_THRESHOLD:
        return "NEUTRAL"
    if net_flow_usd > 0 and notional_buy > notional_sell:
        return "BUY"
    if net_flow_usd < 0 and notional_sell > notional_buy:
        return "SELL"
    return "NEUTRAL"


def compute_flow_metrics(
    slug: str,
    trades: list[TradeTick],
    window_minutes: int,
) -> TerminalFlowResponse:
    """Fold a classified tape into the flow-analytics response.

    Pure function so it can be unit-tested without HTTP. ``trades`` must
    arrive in chronological order (oldest → newest); this is what the
    upstream :func:`pfm.terminal_trades.get_terminal_trades` already
    guarantees.
    """
    windowed = _filter_window(trades, window_minutes)

    n_total = len(windowed)
    n_buy = sum(1 for t in windowed if t.side == "BUY")
    n_sell = sum(1 for t in windowed if t.side == "SELL")
    classified = n_buy + n_sell
    buy_ratio = (n_buy / classified) if classified else 0.0

    notional_buy = 0.0
    notional_sell = 0.0
    aggressive_count = 0
    prev_price: float | None = None
    enriched: list[tuple[TradeTick, float]] = []  # (trade, notional)
    for t in windowed:
        notional = float(t.price) * float(t.size)
        enriched.append((t, notional))
        if t.side == "BUY":
            notional_buy += notional
        elif t.side == "SELL":
            notional_sell += notional
        if _is_aggressive(t, prev_price):
            aggressive_count += 1
        prev_price = float(t.price)

    aggressive_ratio = (aggressive_count / n_total) if n_total else 0.0
    net_flow = notional_buy - notional_sell
    largest = max((n for _, n in enriched), default=0.0)
    top5_sorted = sorted(enriched, key=lambda pair: pair[1], reverse=True)[:5]
    top5 = [
        TopTrade(
            timestamp=t.timestamp,
            price=t.price,
            size=t.size,
            notional_usd=n,
            side=t.side,
        )
        for t, n in top5_sorted
    ]
    bursts = detect_bursts(windowed)
    signal = _informed_signal(aggressive_ratio, net_flow, notional_buy, notional_sell)

    return TerminalFlowResponse(
        slug=slug,
        window_minutes=window_minutes,
        n_trades_total=n_total,
        n_trades_buy=n_buy,
        n_trades_sell=n_sell,
        buy_ratio=buy_ratio,
        aggressive_ratio=aggressive_ratio,
        notional_buy_usd=notional_buy,
        notional_sell_usd=notional_sell,
        net_flow_usd=net_flow,
        largest_trade_usd=largest,
        top_5_trades=top5,
        bursts=bursts,
        informed_flow_signal=signal,
    )


# --- endpoint ---------------------------------------------------------------


@router.get(
    "/flow/{slug}",
    response_model=TerminalFlowResponse,
    summary="Trade-flow analytics (informed/aggressive flow) for a Polymarket market.",
)
def get_terminal_flow(
    slug: Annotated[str, Path(min_length=1, description="Polymarket market slug.")],
    window_minutes: Annotated[int, Query(ge=1, le=24 * 60)] = 60,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> TerminalFlowResponse:
    """Return flow analytics for the trailing ``window_minutes`` of trades."""
    cache_key = (slug, int(window_minutes))
    cached_resp = _FLOW_CACHE.get(cache_key)
    if cached_resp is not None:
        return cached_resp
    # Reuse the existing trades endpoint so we get Lee-Ready classification
    # (and the same Gamma + data-api error handling) for free.
    tape = get_terminal_trades(slug=slug, limit=500, poly=poly)
    resp = compute_flow_metrics(slug, tape.trades, window_minutes)
    _FLOW_CACHE.set(cache_key, resp)
    return resp
