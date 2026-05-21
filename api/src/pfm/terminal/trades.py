"""Polymarket terminal trade tape with Lee-Ready buy/sell inference.

Exposes a single endpoint:

    GET /terminal/trades/{slug}?limit=50

That endpoint resolves the market's ``conditionId`` via the Gamma metadata
endpoint (``/markets?slug=X``), then pulls the most recent trades from the
public ``data-api/trades`` endpoint and tags each trade with a side
(BUY / SELL / AT_MID) using the Lee & Ready (1991) classification:

    - If price > mid → BUY (lifted the offer)
    - If price < mid → SELL (hit the bid)
    - If price == mid → tick test against the previous trade
        - higher than prev trade → BUY
        - lower than prev trade  → SELL
        - equal to prev trade    → AT_MID (zero-tick: cannot classify)

When the API does not provide best bid / best ask (which it usually does
not for historical trade tapes), we fall back to a pure tick test —
equivalent to Lee-Ready when bid/ask are unavailable.

We also compute a 5-minute rolling buy ratio (buy_volume / total_volume)
and flag windows where it deviates from 0.5 by ≥ 0.15 — a crude informed
flow indicator.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Literal

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_export import respond as _export_respond

logger = logging.getLogger(__name__)

POLY_DATA_API: str = "https://data-api.polymarket.com"
ROLLING_WINDOW: str = "5min"
INFORMED_FLOW_THRESHOLD: float = 0.15

# Single 429-retry backoff — matches the polymarket.py source convention.
_RETRY_BACKOFF_S: float = 1.5

Side = Literal["BUY", "SELL", "AT_MID"]

router = APIRouter(prefix="/terminal", tags=["terminal"])

# Trade tape + slug→conditionId cache. The conditionId never changes for a
# given market; cache it for an hour. The tape itself moves on a seconds
# scale but a 5 s cache lets the trades + flow + (parallel) quality fanout
# from a single market-detail open share one upstream fetch.
_CID_CACHE = get_cache("terminal_trades_cid", ttl=3600)
_TRADES_CACHE = get_cache("terminal_trades_tape", ttl=5)


# --- schemas ----------------------------------------------------------------


class TradeTick(BaseModel):
    """One classified trade in the tape."""

    timestamp: str = Field(..., description="ISO-8601 UTC trade timestamp.")
    price: float = Field(..., ge=0.0, le=1.0)
    size: float = Field(..., ge=0.0, description="Trade size in shares/contracts.")
    side: Side = Field(..., description="Lee-Ready inferred side.")


class FlowWindow(BaseModel):
    """One 5-minute rolling buy-ratio observation."""

    timestamp: str
    buy_ratio: float = Field(..., ge=0.0, le=1.0)
    informed: bool = Field(
        ...,
        description=f"True when |buy_ratio - 0.5| ≥ {INFORMED_FLOW_THRESHOLD}.",
    )


class TerminalTradesResponse(BaseModel):
    slug: str
    condition_id: str
    n_trades: int
    trades: list[TradeTick]
    rolling_buy_ratio: list[FlowWindow]
    informed_flow_alert: bool = Field(
        ...,
        description="True if any 5-min window crossed the informed-flow threshold.",
    )


# --- dependencies -----------------------------------------------------------


def get_polymarket_client(request: Request) -> PolymarketClient:
    """Resolve the shared ``PolymarketClient`` from app state.

    Reading off ``request.app.state`` (rather than importing ``pfm.main``)
    keeps this module decoupled from ``main.py`` so it can be wired up by
    a single ``include_router`` call without circular imports.
    """
    poly = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


# --- core algorithm ---------------------------------------------------------


def _classify_lee_ready(
    price: float,
    bid: float | None,
    ask: float | None,
    prev_price: float | None,
    prev_side: Side | None,
) -> Side:
    """Classify a single trade per Lee-Ready (1991).

    Quote rule first; tick rule if price sits at the mid (or quotes are
    missing). Zero-tick reuses the previous trade's classification — we
    return ``AT_MID`` only when the very first trade is unclassifiable.
    """
    if bid is not None and ask is not None and ask > bid:
        mid = 0.5 * (bid + ask)
        if price > mid:
            return "BUY"
        if price < mid:
            return "SELL"
        # price == mid → fall through to tick test
    if prev_price is None:
        return "AT_MID"
    if price > prev_price:
        return "BUY"
    if price < prev_price:
        return "SELL"
    # zero-tick: inherit previous side, else AT_MID
    return prev_side if prev_side is not None else "AT_MID"


def classify_trades(raw_trades: list[dict]) -> list[TradeTick]:
    """Apply Lee-Ready classification to a list of raw trade dicts.

    ``raw_trades`` is expected to be sorted oldest → newest. Each entry
    must have ``timestamp`` (unix seconds or ISO string), ``price``, and
    ``size``; ``bid`` / ``ask`` are optional.
    """
    out: list[TradeTick] = []
    prev_price: float | None = None
    prev_side: Side | None = None
    for t in raw_trades:
        price = float(t["price"])
        size = float(t.get("size", t.get("amount", 0.0)) or 0.0)
        bid = _maybe_float(t.get("bid") or t.get("bestBid"))
        ask = _maybe_float(t.get("ask") or t.get("bestAsk"))
        side = _classify_lee_ready(price, bid, ask, prev_price, prev_side)
        ts_raw = t.get("timestamp") or t.get("ts") or t.get("time")
        ts = _to_iso_utc(ts_raw)
        out.append(TradeTick(timestamp=ts, price=price, size=size, side=side))
        prev_price = price
        if side != "AT_MID":
            prev_side = side
    return out


def rolling_buy_ratio(trades: list[TradeTick]) -> list[FlowWindow]:
    """Compute 5-minute rolling buy-volume / total-volume.

    Returns one window per trade timestamp. ``buy_ratio`` is 0.5 when no
    classified volume sits in the window (so the metric is well-defined
    even at the edges).
    """
    if not trades:
        return []
    df = (
        pd.DataFrame(
            {
                "ts": pd.to_datetime([t.timestamp for t in trades], utc=True),
                "size": [t.size for t in trades],
                "side": [t.side for t in trades],
            }
        )
        .set_index("ts")
        .sort_index()
    )
    df["buy_vol"] = df["size"].where(df["side"] == "BUY", 0.0)
    df["sell_vol"] = df["size"].where(df["side"] == "SELL", 0.0)
    rolled = df[["buy_vol", "sell_vol"]].rolling(ROLLING_WINDOW).sum()
    total = rolled["buy_vol"] + rolled["sell_vol"]
    ratio = (rolled["buy_vol"] / total).where(total > 0, 0.5).fillna(0.5)
    out: list[FlowWindow] = []
    for ts, r in ratio.items():
        r_f = float(r)
        out.append(
            FlowWindow(
                timestamp=ts.isoformat().replace("+00:00", "Z"),
                buy_ratio=r_f,
                informed=abs(r_f - 0.5) >= INFORMED_FLOW_THRESHOLD,
            )
        )
    return out


# --- endpoint ---------------------------------------------------------------


@router.get(
    "/trades/{slug}",
    response_model=None,
    summary="Recent classified trades for a Polymarket market.",
)
@router.get(
    "/volume-tape/{slug}",
    response_model=None,
    summary="Recent classified trades (alias of /trades/{slug}).",
    include_in_schema=True,
)  # UX-audit 2026-05-14: front-end calls /volume-tape — keep an alias.
def get_terminal_trades(
    slug: Annotated[str, Path(min_length=1, description="Polymarket market slug.")],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    format: Annotated[Literal["json", "csv", "pdf"], Query()] = "json",
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> TerminalTradesResponse | FastAPIResponse:
    """Return the last ``limit`` trades for ``slug`` with Lee-Ready sides.

    Short-cached (5 s) on ``(slug, limit)``. This lets the trades and flow
    endpoints — which the market-detail UI fires together — share one
    upstream data-api call without re-running Lee-Ready classification.
    """
    cache_key = (slug, int(limit))
    cached_resp = _TRADES_CACHE.get(cache_key)
    if cached_resp is not None:
        if format == "json":
            return cached_resp
        return _export_respond(cached_resp, format, filename=f"trades-{slug}", kind="market")

    condition_id = _resolve_condition_id(poly, slug)

    try:
        r = poly._client.get(
            f"{POLY_DATA_API}/trades",
            params={"market": condition_id, "limit": int(limit)},
        )
        if r.status_code == 429:
            logger.warning(
                "terminal/trades data-api 429 on slug=%s — retrying in %.1fs",
                slug,
                _RETRY_BACKOFF_S,
            )
            time.sleep(_RETRY_BACKOFF_S)
            r = poly._client.get(
                f"{POLY_DATA_API}/trades",
                params={"market": condition_id, "limit": int(limit)},
            )
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket data-api error: {e}") from e

    raw = payload if isinstance(payload, list) else payload.get("trades", [])
    # data-api returns newest → oldest; flip so classify_trades sees chronological order.
    raw_sorted = sorted(raw, key=_sort_key)
    trades = classify_trades(raw_sorted)
    flow = rolling_buy_ratio(trades)
    informed_alert = any(w.informed for w in flow)

    resp = TerminalTradesResponse(
        slug=slug,
        condition_id=condition_id,
        n_trades=len(trades),
        trades=trades,
        rolling_buy_ratio=flow,
        informed_flow_alert=informed_alert,
    )
    _TRADES_CACHE.set(cache_key, resp)
    if format == "json":
        return resp
    return _export_respond(resp, format, filename=f"trades-{slug}", kind="market")


# --- helpers ----------------------------------------------------------------


def _resolve_condition_id(poly: PolymarketClient, slug: str) -> str:
    """Fetch ``conditionId`` for a slug via Gamma /markets?slug=...

    We can't reuse ``get_market_metadata`` because it doesn't surface the
    conditionId. We hit Gamma directly with the same client.

    Cached for 1 h: slug→conditionId is immutable per market.
    """
    cached_cid = _CID_CACHE.get(slug)
    if cached_cid is not None:
        return cached_cid
    try:
        r = poly._client.get(f"{poly.gamma_url}/markets", params={"slug": slug})
        if r.status_code == 429:
            logger.warning(
                "terminal/trades gamma 429 on slug=%s — retrying in %.1fs",
                slug,
                _RETRY_BACKOFF_S,
            )
            time.sleep(_RETRY_BACKOFF_S)
            r = poly._client.get(f"{poly.gamma_url}/markets", params={"slug": slug})
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e

    market = payload[0] if isinstance(payload, list) and payload else None
    if not market:
        raise HTTPException(status_code=404, detail=f"no market found for slug={slug!r}")
    cid = market.get("conditionId") or market.get("condition_id")
    if not cid:
        raise HTTPException(
            status_code=502,
            detail=f"market {slug!r} missing conditionId in gamma payload",
        )
    cid_s = str(cid)
    _CID_CACHE.set(slug, cid_s)
    return cid_s


def _maybe_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_iso_utc(ts: object) -> str:
    """Normalize a timestamp (unix seconds or ISO string) to ISO-8601 UTC."""
    if ts is None:
        return ""
    if isinstance(ts, (int, float)):
        return pd.Timestamp(ts, unit="s", tz="UTC").isoformat().replace("+00:00", "Z")
    try:
        return pd.Timestamp(str(ts)).tz_convert("UTC").isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        # Fall back to naive parse, then assume UTC.
        try:
            return pd.Timestamp(str(ts)).tz_localize("UTC").isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError):
            return str(ts)


def _sort_key(t: dict) -> float:
    raw = t.get("timestamp") or t.get("ts") or t.get("time") or 0
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(pd.Timestamp(str(raw)).timestamp())
    except (TypeError, ValueError):
        return 0.0
