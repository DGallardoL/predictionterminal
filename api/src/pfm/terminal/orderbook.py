"""Terminal orderbook ladder endpoint.

Exposes ``GET /terminal/book/{slug}`` which returns a 10-level orderbook
ladder for the YES side of a Polymarket market, plus depth metrics, fill
costs for $50/$200/$1000 buy/sell orders, and a top-5 imbalance metric.

External calls:
    - Gamma  ``/markets?slug={slug}`` for the YES ``clobTokenIds``.
    - CLOB   ``/book?token_id={token_id}`` for the live orderbook snapshot.

All HTTP is done via httpx so it can be respx-mocked in tests.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.terminal_export import respond as _export_respond

logger = logging.getLogger(__name__)

GAMMA_URL: str = "https://gamma-api.polymarket.com"
CLOB_URL: str = "https://clob.polymarket.com"

# Single 429-retry backoff. Matches the polymarket.py source convention.
_RETRY_BACKOFF_S: float = 1.5

# Public router — main.py is responsible for ``app.include_router(...)``.
router = APIRouter(prefix="/terminal", tags=["terminal"])

# slug → YES token_id never changes for a given market, but resolving it
# requires a Gamma /markets?slug=... call that gets rate-limited (429) very
# easily during a busy market-detail open. Cache for an hour.
_TOKEN_CACHE = get_cache("terminal_orderbook_tokens", ttl=3600)
# The raw book changes constantly but the level structure shifts on a
# multi-second scale for most Polymarket markets. 2 s TTL collapses
# concurrent quote+orderbook+quality fanout into one /book call without
# introducing user-visible staleness (UI polls at 5-10 s).
_BOOK_CACHE = get_cache("terminal_orderbook_book", ttl=2)


# ---- Schemas ---------------------------------------------------------------


class BookLevel(BaseModel):
    """One side of one level of the ladder, with running cumulative size."""

    price: float = Field(..., ge=0.0, le=1.0)
    size: float = Field(..., ge=0.0)
    cumulative: float = Field(..., ge=0.0)


class FillCost(BaseModel):
    """Fill cost in cents (notional) for a buy and sell of the given USD size."""

    buy: float | None
    sell: float | None


class OrderbookResponse(BaseModel):
    """Top-of-book ladder + depth/imbalance metrics."""

    slug: str
    token_id: str
    bid_levels: list[BookLevel]
    ask_levels: list[BookLevel]
    mid: float | None
    spread_cents: float | None
    depth_at_1c_mid: float
    depth_at_3c_mid: float
    depth_at_10c_mid: float
    fill_cost: dict[str, FillCost]
    imbalance_top5: float | None
    imbalance_signal: Literal["bullish", "bearish", "neutral"]


# ---- Pure helpers (kept module-private; tested via the endpoint) -----------


def _coerce_levels(raw: list[dict], side: Literal["bid", "ask"]) -> list[BookLevel]:
    """Sort + clip + cumulate the top 10 levels of one side."""
    parsed: list[tuple[float, float]] = []
    for entry in raw:
        try:
            price = float(entry["price"])
            size = float(entry["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if size <= 0:
            continue
        parsed.append((price, size))

    # Polymarket sometimes returns bids ascending — make ordering explicit.
    parsed.sort(key=lambda t: t[0], reverse=(side == "bid"))
    parsed = parsed[:10]

    out: list[BookLevel] = []
    cum = 0.0
    for price, size in parsed:
        cum += size
        out.append(BookLevel(price=price, size=size, cumulative=cum))
    return out


def _depth_within(levels: list[BookLevel], anchor: float, band: float) -> float:
    """Sum size on levels with price within ``band`` of ``anchor`` (in dollars)."""
    return sum(lvl.size for lvl in levels if abs(lvl.price - anchor) <= band + 1e-12)


def _walk_book(levels: list[BookLevel], notional_usd: float) -> float | None:
    """Walk levels (already side-sorted) and return average fill price in cents.

    Returns ``None`` if the book can't fill the requested size.
    """
    if notional_usd <= 0 or not levels:
        return None
    remaining = notional_usd
    spend = 0.0  # in dollars
    shares = 0.0
    for lvl in levels:
        if lvl.price <= 0:
            continue
        # Each "share" of a binary contract has notional == price (in [0, 1]).
        level_notional = lvl.price * lvl.size
        take = min(level_notional, remaining)
        take_shares = take / lvl.price
        spend += take
        shares += take_shares
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9 or shares <= 0:
        return None
    avg_price = spend / shares  # dollars per share, in [0, 1]
    return round(avg_price * 100.0, 4)  # convert to cents


def _imbalance(bids: list[BookLevel], asks: list[BookLevel]) -> float | None:
    """Top-5 size imbalance: bids / (bids + asks). ``None`` if both empty."""
    bid_sz = sum(b.size for b in bids[:5])
    ask_sz = sum(a.size for a in asks[:5])
    total = bid_sz + ask_sz
    if total <= 0:
        return None
    return bid_sz / total


def _imbalance_signal(value: float | None) -> Literal["bullish", "bearish", "neutral"]:
    if value is None:
        return "neutral"
    if value > 0.6:
        return "bullish"
    if value < 0.4:
        return "bearish"
    return "neutral"


# ---- HTTP fetchers ---------------------------------------------------------


def _fetch_yes_token_id(slug: str, client: httpx.Client) -> str:
    """Resolve a slug to the YES (first) ``clobTokenIds`` entry.

    Cached for 1 h: the slug→token_id mapping is immutable for the
    lifetime of a market. This collapses the per-request Gamma round
    trip (and its 429-prone rate budget) into a one-time lookup per slug.
    On a cache miss we additionally retry once on a 429 with a 1.5 s
    backoff so a transient rate-limit doesn't surface as a 502.
    """
    cached = _TOKEN_CACHE.get(slug)
    if cached is not None:
        return cached
    r = client.get(f"{GAMMA_URL}/markets", params={"slug": slug})
    if r.status_code == 429:
        logger.warning(
            "terminal/book gamma 429 on slug=%s — retrying in %.1fs", slug, _RETRY_BACKOFF_S
        )
        time.sleep(_RETRY_BACKOFF_S)
        r = client.get(f"{GAMMA_URL}/markets", params={"slug": slug})
    r.raise_for_status()
    payload = r.json()
    market = payload[0] if isinstance(payload, list) and payload else None
    if not market:
        raise HTTPException(status_code=404, detail=f"no market found for slug={slug!r}")
    raw = market.get("clobTokenIds")
    if not raw:
        raise HTTPException(status_code=502, detail=f"market {slug!r} has no clobTokenIds")
    try:
        token_ids = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"clobTokenIds for {slug!r} is not valid JSON"
        ) from exc
    if not isinstance(token_ids, list) or not token_ids:
        raise HTTPException(status_code=502, detail=f"empty clobTokenIds for {slug!r}")
    token_id = str(token_ids[0])
    _TOKEN_CACHE.set(slug, token_id)
    return token_id


def _fetch_book(token_id: str, client: httpx.Client) -> dict:
    """Fetch the live CLOB book; 2 s in-process cache to coalesce burst
    requests when the UI fires quote+orderbook+quality back-to-back.

    Retries once on 429 with a 1.5 s backoff. The TTL is deliberately
    short (≈2 s) because orderbook depth shifts fast — too-long caching
    would mislead fill-cost users.
    """
    cached = _BOOK_CACHE.get(token_id)
    if cached is not None:
        return cached
    r = client.get(f"{CLOB_URL}/book", params={"token_id": token_id})
    if r.status_code == 429:
        logger.warning(
            "terminal/book clob 429 on token=%s — retrying in %.1fs", token_id, _RETRY_BACKOFF_S
        )
        time.sleep(_RETRY_BACKOFF_S)
        r = client.get(f"{CLOB_URL}/book", params={"token_id": token_id})
    r.raise_for_status()
    book = r.json()
    _BOOK_CACHE.set(token_id, book)
    return book


# ---- Endpoint --------------------------------------------------------------


@router.get("/book/{slug}", response_model=None)
@router.get("/orderbook/{slug}", response_model=None)
def get_book_ladder(
    slug: str,
    timeout: float = Query(default=10.0, gt=0.0, le=30.0),
    format: Literal["json", "csv", "pdf"] = Query(default="json"),
) -> OrderbookResponse | FastAPIResponse:
    """Return a 10-level ladder, depth bands, fill costs, and top-5 imbalance.

    Mounted under both ``/terminal/book/{slug}`` (legacy) and
    ``/terminal/orderbook/{slug}`` (newer, more discoverable). The two paths
    share the same handler so behaviour and response shape are identical.
    """
    with httpx.Client(timeout=timeout) as client:
        token_id = _fetch_yes_token_id(slug, client)
        try:
            raw = _fetch_book(token_id, client)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502, detail=f"CLOB /book failed: {exc.response.status_code}"
            ) from exc

    bids = _coerce_levels(raw.get("bids", []) or [], side="bid")
    asks = _coerce_levels(raw.get("asks", []) or [], side="ask")

    best_bid = bids[0].price if bids else None
    best_ask = asks[0].price if asks else None
    if best_bid is not None and best_ask is not None:
        mid: float | None = (best_bid + best_ask) / 2.0
        spread_cents: float | None = round((best_ask - best_bid) * 100.0, 4)
    else:
        mid, spread_cents = None, None

    if mid is not None:
        depth_1c = _depth_within(bids, mid, 0.01) + _depth_within(asks, mid, 0.01)
        depth_3c = _depth_within(bids, mid, 0.03) + _depth_within(asks, mid, 0.03)
        depth_10c = _depth_within(bids, mid, 0.10) + _depth_within(asks, mid, 0.10)
    else:
        depth_1c = depth_3c = depth_10c = 0.0

    fill_cost: dict[str, FillCost] = {}
    for usd in (50, 200, 1000):
        fill_cost[str(usd)] = FillCost(
            buy=_walk_book(asks, float(usd)),  # buyer lifts the offer
            sell=_walk_book(bids, float(usd)),  # seller hits the bid
        )

    imb = _imbalance(bids, asks)

    resp = OrderbookResponse(
        slug=slug,
        token_id=token_id,
        bid_levels=bids,
        ask_levels=asks,
        mid=mid,
        spread_cents=spread_cents,
        depth_at_1c_mid=depth_1c,
        depth_at_3c_mid=depth_3c,
        depth_at_10c_mid=depth_10c,
        fill_cost=fill_cost,
        imbalance_top5=imb,
        imbalance_signal=_imbalance_signal(imb),
    )
    if format == "json":
        return resp
    return _export_respond(resp, format, filename=f"orderbook-{slug}", kind="market")
