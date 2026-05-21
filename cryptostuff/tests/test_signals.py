"""Tests for the real-time signal engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from models.signals import SignalEngine


def _ts(offset_seconds: int = 0) -> datetime:
    return datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=offset_seconds)


def test_vwap_from_trades() -> None:
    engine = SignalEngine()
    signals = []
    for i in range(20):
        sigs = engine.on_trade("BTCUSDT", 100.0 + i * 0.1, 1.0, False, _ts(i))
        signals.extend(sigs)

    vwap_signals = [s for s in signals if s.name == "vwap_1m"]
    assert len(vwap_signals) > 0
    last_vwap = vwap_signals[-1]
    assert 100.0 < last_vwap.value < 102.0


def test_signed_volume_direction() -> None:
    engine = SignalEngine()
    # All buyer-taker trades (is_buyer_maker=False -> positive signed volume)
    for i in range(10):
        engine.on_trade("ETHUSDT", 3000.0, 1.0, False, _ts(i))
    sigs = engine.on_trade("ETHUSDT", 3000.0, 1.0, False, _ts(11))
    sv = [s for s in sigs if s.name == "signed_volume_1m"]
    assert len(sv) == 1
    assert sv[0].value > 0

    # All seller-taker trades (is_buyer_maker=True -> negative signed volume)
    engine2 = SignalEngine()
    for i in range(10):
        engine2.on_trade("ETHUSDT", 3000.0, 1.0, True, _ts(i))
    sigs2 = engine2.on_trade("ETHUSDT", 3000.0, 1.0, True, _ts(11))
    sv2 = [s for s in sigs2 if s.name == "signed_volume_1m"]
    assert len(sv2) == 1
    assert sv2[0].value < 0


def test_whale_detection() -> None:
    engine = SignalEngine(whale_warmup=50)
    # Warmup with small trades
    for i in range(500):
        engine.on_trade("BTCUSDT", 100.0, 0.01, False, _ts(i))

    # Big trade should trigger whale
    sigs = engine.on_trade("BTCUSDT", 100.0, 1000.0, False, _ts(501))
    whales = [s for s in sigs if s.name == "whale_detected"]
    assert len(whales) == 1
    assert whales[0].metadata["side"] == "buy"


def test_realized_volatility_increases_with_price_swings() -> None:
    engine_stable = SignalEngine()
    engine_volatile = SignalEngine()

    for i in range(100):
        engine_stable.on_trade("BTC", 100.0, 1.0, False, _ts(i))
        # Alternating price swings
        price = 100.0 + (5.0 if i % 2 == 0 else -5.0)
        engine_volatile.on_trade("BTC", price, 1.0, False, _ts(i))

    sigs_stable = engine_stable.on_trade("BTC", 100.0, 1.0, False, _ts(101))
    sigs_volatile = engine_volatile.on_trade("BTC", 105.0, 1.0, False, _ts(101))

    rv_stable = [s for s in sigs_stable if s.name == "rv_5m"]
    rv_volatile = [s for s in sigs_volatile if s.name == "rv_5m"]

    if rv_stable and rv_volatile:
        assert rv_volatile[0].value > rv_stable[0].value


def test_book_ticker_signals() -> None:
    engine = SignalEngine()
    sigs = engine.on_book_ticker("BTCUSDT", 100.0, 5.0, 100.05, 3.0, _ts(0))

    names = {s.name for s in sigs}
    assert "midprice" in names
    assert "spread_bps" in names
    assert "obi_top1" in names

    midprice = [s for s in sigs if s.name == "midprice"][0]
    assert midprice.value == 100.025

    obi = [s for s in sigs if s.name == "obi_top1"][0]
    assert obi.value > 0  # more bids than asks


def test_order_flow_imbalance() -> None:
    engine = SignalEngine()
    for i in range(20):
        engine.on_trade("SOL", 150.0, 1.0, False, _ts(i))
    sigs = engine.on_trade("SOL", 150.0, 1.0, False, _ts(21))
    ofi = [s for s in sigs if s.name == "order_flow_imbalance_1m"]
    assert len(ofi) == 1
    assert ofi[0].value == 1.0  # all buyer-taker


def test_state_summary() -> None:
    engine = SignalEngine()
    assert engine.get_state_summary("NOPE") is None

    engine.on_trade("X", 10.0, 1.0, False, _ts(0))
    summary = engine.get_state_summary("X")
    assert summary is not None
    assert summary["last_price"] == 10.0
    assert summary["total_trades"] == 1
