"""Real-time quant signal engine over streaming Binance data.

Computes signals incrementally as events arrive -- no batch processing needed.
All state is kept in memory using rolling windows.

Signals produced:
    - VWAP (1m, 5m, 30m) + Z-score mean reversion
    - Realized volatility (5m, 15m)
    - Signed volume / order flow imbalance
    - Whale detection (notional > P99 threshold)
    - Spread and midprice tracking
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Signal:
    """A computed signal at a point in time."""

    symbol: str
    timestamp: datetime
    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SymbolState:
    """Rolling state for a single symbol."""

    trades_1m: deque = field(default_factory=lambda: deque(maxlen=10_000))
    trades_5m: deque = field(default_factory=lambda: deque(maxlen=50_000))
    trades_30m: deque = field(default_factory=lambda: deque(maxlen=300_000))
    returns_5m: deque = field(default_factory=lambda: deque(maxlen=300))
    returns_15m: deque = field(default_factory=lambda: deque(maxlen=900))
    notionals: deque = field(default_factory=lambda: deque(maxlen=100_000))
    last_price: float = 0.0
    prev_price: float = 0.0
    whale_threshold: float = 0.0
    last_bid: float = 0.0
    last_ask: float = 0.0
    last_bid_qty: float = 0.0
    last_ask_qty: float = 0.0
    total_trades: int = 0


class SignalEngine:
    """Stateful signal engine that processes events and emits signals.

    Usage:
        engine = SignalEngine()

        # Feed trade events
        signals = engine.on_trade(symbol, price, qty, is_buyer_maker, timestamp)

        # Feed book ticker events
        signals = engine.on_book_ticker(symbol, bid_p, bid_q, ask_p, ask_q, timestamp)

        # Each call returns a list of Signal objects (may be empty if not enough data)
    """

    def __init__(self, whale_warmup: int = 1000) -> None:
        self._states: dict[str, _SymbolState] = {}
        self._whale_warmup = whale_warmup

    def _get(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    def on_trade(
        self,
        symbol: str,
        price: float,
        quantity: float,
        is_buyer_maker: bool,
        timestamp: datetime,
    ) -> list[Signal]:
        s = self._get(symbol)
        ts_epoch = timestamp.timestamp()
        notional = price * quantity
        signed_qty = -quantity if is_buyer_maker else quantity

        trade_record = (ts_epoch, price, quantity, notional, signed_qty)
        s.trades_1m.append(trade_record)
        s.trades_5m.append(trade_record)
        s.trades_30m.append(trade_record)
        s.notionals.append(notional)
        s.total_trades += 1

        s.prev_price = s.last_price
        s.last_price = price

        if s.prev_price > 0:
            ret = math.log(price / s.prev_price) if price > 0 and s.prev_price > 0 else 0.0
            s.returns_5m.append(ret)
            s.returns_15m.append(ret)

        # Prune old trades from rolling windows
        cutoff_1m = ts_epoch - 60
        cutoff_5m = ts_epoch - 300
        cutoff_30m = ts_epoch - 1800
        while s.trades_1m and s.trades_1m[0][0] < cutoff_1m:
            s.trades_1m.popleft()
        while s.trades_5m and s.trades_5m[0][0] < cutoff_5m:
            s.trades_5m.popleft()
        while s.trades_30m and s.trades_30m[0][0] < cutoff_30m:
            s.trades_30m.popleft()

        # Update whale threshold periodically
        if s.total_trades % 500 == 0 and len(s.notionals) >= self._whale_warmup:
            sorted_n = sorted(s.notionals)
            idx = int(len(sorted_n) * 0.99)
            s.whale_threshold = sorted_n[min(idx, len(sorted_n) - 1)]

        signals: list[Signal] = []

        # VWAP signals
        for label, window in [("vwap_1m", s.trades_1m), ("vwap_5m", s.trades_5m), ("vwap_30m", s.trades_30m)]:
            if len(window) < 2:
                continue
            total_notional = sum(t[3] for t in window)
            total_qty = sum(t[2] for t in window)
            if total_qty > 0:
                vwap = total_notional / total_qty
                signals.append(Signal(symbol, timestamp, label, vwap))

                # Z-score for mean reversion (only 30m)
                if label == "vwap_30m" and price > 0 and vwap > 0:
                    prices_in_window = [t[1] for t in window]
                    if len(prices_in_window) > 10:
                        mean_p = sum(prices_in_window) / len(prices_in_window)
                        var_p = sum((p - mean_p) ** 2 for p in prices_in_window) / len(prices_in_window)
                        std_p = math.sqrt(var_p) if var_p > 0 else 1e-10
                        z_score = (price - vwap) / std_p
                        signals.append(Signal(
                            symbol, timestamp, "vwap_zscore_30m", z_score,
                            {"mean_reversion": abs(z_score) > 2.0},
                        ))

        # Signed volume (order flow)
        if len(s.trades_1m) > 0:
            sv = sum(t[4] for t in s.trades_1m)
            total_v = sum(abs(t[4]) for t in s.trades_1m)
            ofi = sv / total_v if total_v > 0 else 0.0
            signals.append(Signal(symbol, timestamp, "signed_volume_1m", sv))
            signals.append(Signal(symbol, timestamp, "order_flow_imbalance_1m", ofi))

        # Realized volatility
        for label, returns in [("rv_5m", s.returns_5m), ("rv_15m", s.returns_15m)]:
            if len(returns) < 10:
                continue
            rets = list(returns)
            n = len(rets)
            mean_r = sum(rets) / n
            var_r = sum((r - mean_r) ** 2 for r in rets) / n
            rv = math.sqrt(var_r) if var_r > 0 else 0.0
            # Annualize: assume ~500 trades/sec, so scale factor depends on trade freq
            signals.append(Signal(symbol, timestamp, label, rv))

        # Whale detection
        if s.whale_threshold > 0 and notional >= s.whale_threshold:
            side = "sell" if is_buyer_maker else "buy"
            signals.append(Signal(
                symbol, timestamp, "whale_detected", notional,
                {"side": side, "threshold": s.whale_threshold, "size_vs_threshold": notional / s.whale_threshold},
            ))

        return signals

    def on_book_ticker(
        self,
        symbol: str,
        bid_price: float,
        bid_qty: float,
        ask_price: float,
        ask_qty: float,
        timestamp: datetime,
    ) -> list[Signal]:
        s = self._get(symbol)
        s.last_bid = bid_price
        s.last_ask = ask_price
        s.last_bid_qty = bid_qty
        s.last_ask_qty = ask_qty

        signals: list[Signal] = []

        if bid_price > 0 and ask_price > 0:
            midprice = (bid_price + ask_price) / 2
            spread = ask_price - bid_price
            spread_bps = (spread / midprice * 10000) if midprice > 0 else 0.0

            signals.append(Signal(symbol, timestamp, "midprice", midprice))
            signals.append(Signal(symbol, timestamp, "spread_bps", spread_bps))

        total_qty = bid_qty + ask_qty
        if total_qty > 0:
            obi = (bid_qty - ask_qty) / total_qty
            signals.append(Signal(symbol, timestamp, "obi_top1", obi))

        return signals

    def get_state_summary(self, symbol: str) -> dict[str, Any] | None:
        if symbol not in self._states:
            return None
        s = self._states[symbol]
        return {
            "last_price": s.last_price,
            "trades_in_1m_window": len(s.trades_1m),
            "trades_in_5m_window": len(s.trades_5m),
            "whale_threshold": s.whale_threshold,
            "total_trades": s.total_trades,
            "last_bid": s.last_bid,
            "last_ask": s.last_ask,
        }
