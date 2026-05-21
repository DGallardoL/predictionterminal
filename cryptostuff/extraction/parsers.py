"""Parse raw Binance WebSocket events into typed dataclasses.

Each parser takes the `data` dict from a combined stream message and returns
a typed event object. Use these to normalize the raw JSON into clean Python
objects before feeding them to models or storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any


def _dec(v: str | float | Decimal) -> Decimal:
    return Decimal(str(v))


def _ts(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _bucket(ms: int) -> str:
    return _ts(ms).strftime("%Y-%m-%d-%H")


@dataclass(frozen=True, slots=True)
class TradeEvent:
    symbol: str
    trade_id: int
    price: Decimal
    quantity: Decimal
    notional: Decimal
    is_buyer_maker: bool
    trade_time: datetime
    event_time: datetime
    ingest_time: datetime
    bucket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": float(self.price),
            "quantity": float(self.quantity),
            "notional": float(self.notional),
            "is_buyer_maker": self.is_buyer_maker,
            "trade_time": self.trade_time,
            "event_time": self.event_time,
            "ingest_time": self.ingest_time,
        }


@dataclass(frozen=True, slots=True)
class BookTickerEvent:
    symbol: str
    update_id: int
    best_bid_price: Decimal
    best_bid_qty: Decimal
    best_ask_price: Decimal
    best_ask_qty: Decimal
    midprice: Decimal
    spread: Decimal
    spread_bps: Decimal
    ingest_time: datetime
    bucket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "update_id": self.update_id,
            "best_bid_price": float(self.best_bid_price),
            "best_bid_qty": float(self.best_bid_qty),
            "best_ask_price": float(self.best_ask_price),
            "best_ask_qty": float(self.best_ask_qty),
            "midprice": float(self.midprice),
            "spread": float(self.spread),
            "spread_bps": float(self.spread_bps),
            "ingest_time": self.ingest_time,
        }


@dataclass(frozen=True, slots=True)
class DepthEvent:
    symbol: str
    event_time: datetime
    first_update_id: int
    last_update_id: int
    bids: list[tuple[Decimal, Decimal]]
    asks: list[tuple[Decimal, Decimal]]
    bucket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "event_time": self.event_time,
            "first_update_id": self.first_update_id,
            "last_update_id": self.last_update_id,
            "n_bids": len(self.bids),
            "n_asks": len(self.asks),
        }


@dataclass(frozen=True, slots=True)
class KlineEvent:
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open_price: Decimal
    close_price: Decimal
    high_price: Decimal
    low_price: Decimal
    base_volume: Decimal
    quote_volume: Decimal
    n_trades: int
    is_closed: bool
    taker_buy_base: Decimal
    taker_buy_quote: Decimal
    ingest_time: datetime
    bucket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open": float(self.open_price),
            "close": float(self.close_price),
            "high": float(self.high_price),
            "low": float(self.low_price),
            "volume": float(self.base_volume),
            "n_trades": self.n_trades,
            "is_closed": self.is_closed,
        }


def parse_trade(data: dict[str, Any]) -> TradeEvent:
    price = _dec(data["p"])
    qty = _dec(data["q"])
    trade_time_ms = int(data["T"])
    return TradeEvent(
        symbol=data["s"],
        trade_id=int(data["t"]),
        price=price,
        quantity=qty,
        notional=price * qty,
        is_buyer_maker=bool(data["m"]),
        trade_time=_ts(trade_time_ms),
        event_time=_ts(int(data["E"])),
        ingest_time=datetime.now(UTC),
        bucket=_bucket(trade_time_ms),
    )


def parse_book_ticker(data: dict[str, Any]) -> BookTickerEvent:
    now = datetime.now(UTC)
    bid_p = _dec(data["b"])
    ask_p = _dec(data["a"])
    midprice = (bid_p + ask_p) / 2
    spread = ask_p - bid_p
    spread_bps = (spread / midprice * 10000) if midprice > 0 else Decimal(0)
    return BookTickerEvent(
        symbol=data["s"],
        update_id=int(data["u"]),
        best_bid_price=bid_p,
        best_bid_qty=_dec(data["B"]),
        best_ask_price=ask_p,
        best_ask_qty=_dec(data["A"]),
        midprice=midprice,
        spread=spread,
        spread_bps=spread_bps,
        ingest_time=now,
        bucket=_bucket(int(now.timestamp() * 1000)),
    )


def parse_depth(data: dict[str, Any]) -> DepthEvent:
    event_time_ms = int(data["E"])
    return DepthEvent(
        symbol=data["s"],
        event_time=_ts(event_time_ms),
        first_update_id=int(data["U"]),
        last_update_id=int(data["u"]),
        bids=[(_dec(p), _dec(q)) for p, q in data.get("b", [])],
        asks=[(_dec(p), _dec(q)) for p, q in data.get("a", [])],
        bucket=_bucket(event_time_ms),
    )


def parse_kline(data: dict[str, Any]) -> KlineEvent:
    k = data["k"]
    open_time_ms = int(k["t"])
    return KlineEvent(
        symbol=data["s"],
        interval=str(k["i"]),
        open_time=_ts(open_time_ms),
        close_time=_ts(int(k["T"])),
        open_price=_dec(k["o"]),
        close_price=_dec(k["c"]),
        high_price=_dec(k["h"]),
        low_price=_dec(k["l"]),
        base_volume=_dec(k["v"]),
        quote_volume=_dec(k["q"]),
        n_trades=int(k["n"]),
        is_closed=bool(k["x"]),
        taker_buy_base=_dec(k["V"]),
        taker_buy_quote=_dec(k["Q"]),
        ingest_time=datetime.now(UTC),
        bucket=_bucket(open_time_ms),
    )
