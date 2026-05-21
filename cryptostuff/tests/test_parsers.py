"""Tests for Binance event parsers."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from extraction.parsers import (
    parse_book_ticker,
    parse_depth,
    parse_kline,
    parse_trade,
)


@pytest.fixture
def trade_payload() -> dict:
    return {
        "e": "trade",
        "E": 1672515782136,
        "s": "BTCUSDT",
        "t": 12345,
        "p": "16800.50",
        "q": "0.0500",
        "T": 1672515782128,
        "m": False,
        "M": True,
    }


@pytest.fixture
def book_ticker_payload() -> dict:
    return {
        "u": 400900217,
        "s": "BNBUSDT",
        "b": "25.35190000",
        "B": "31.21000000",
        "a": "25.36520000",
        "A": "40.66000000",
    }


@pytest.fixture
def depth_payload() -> dict:
    return {
        "e": "depthUpdate",
        "E": 1672515782200,
        "s": "BTCUSDT",
        "U": 100,
        "u": 105,
        "b": [["16800.0", "1.5"], ["16799.5", "0.0"]],
        "a": [["16801.0", "0.8"]],
    }


@pytest.fixture
def kline_payload() -> dict:
    return {
        "e": "kline",
        "E": 1672515782200,
        "s": "BTCUSDT",
        "k": {
            "t": 1672515781000,
            "T": 1672515781999,
            "s": "BTCUSDT",
            "i": "1s",
            "f": 1000,
            "L": 1010,
            "o": "16800.0",
            "c": "16800.5",
            "h": "16801.0",
            "l": "16799.5",
            "v": "1.5",
            "n": 11,
            "x": True,
            "q": "25201.0",
            "V": "0.7",
            "Q": "11760.5",
            "B": "0",
        },
    }


def test_parse_trade(trade_payload: dict) -> None:
    evt = parse_trade(trade_payload)
    assert evt.symbol == "BTCUSDT"
    assert evt.trade_id == 12345
    assert evt.price == Decimal("16800.50")
    assert evt.quantity == Decimal("0.0500")
    assert evt.notional == Decimal("840.025000")
    assert evt.is_buyer_maker is False
    assert isinstance(evt.trade_time, datetime)
    assert isinstance(evt.event_time, datetime)
    assert evt.bucket.startswith("20")


def test_parse_book_ticker(book_ticker_payload: dict) -> None:
    evt = parse_book_ticker(book_ticker_payload)
    assert evt.symbol == "BNBUSDT"
    assert evt.update_id == 400900217
    assert evt.best_bid_price == Decimal("25.35190000")
    assert evt.best_ask_price == Decimal("25.36520000")
    assert evt.midprice == (evt.best_bid_price + evt.best_ask_price) / 2
    assert evt.spread == evt.best_ask_price - evt.best_bid_price
    assert evt.spread_bps > 0


def test_parse_depth(depth_payload: dict) -> None:
    evt = parse_depth(depth_payload)
    assert evt.symbol == "BTCUSDT"
    assert evt.first_update_id == 100
    assert evt.last_update_id == 105
    assert len(evt.bids) == 2
    assert len(evt.asks) == 1
    assert evt.bids[0] == (Decimal("16800.0"), Decimal("1.5"))


def test_parse_kline(kline_payload: dict) -> None:
    evt = parse_kline(kline_payload)
    assert evt.symbol == "BTCUSDT"
    assert evt.interval == "1s"
    assert evt.open_price == Decimal("16800.0")
    assert evt.n_trades == 11
    assert evt.is_closed is True
    assert evt.open_time < evt.close_time


def test_trade_to_dict(trade_payload: dict) -> None:
    evt = parse_trade(trade_payload)
    d = evt.to_dict()
    assert d["symbol"] == "BTCUSDT"
    assert d["price"] == 16800.50
    assert d["notional"] == 840.025
