# CLAUDE.md

> Context file for Claude Code. Read this before touching anything.

---

## What this is

A **lightweight, self-sustained real-time crypto microstructure signal engine**. It connects to the Binance public WebSocket, streams trade and order book data for multiple pairs simultaneously, and computes quant finance signals in real-time with zero external dependencies beyond a Python process (no databases, no Docker, no Spark).

This module is designed to be imported into a larger **Bloomberg terminal-style project** for prediction markets and trading strategies. It is the crypto data + signals component.

---

## How to run

```bash
# Install deps
pip install -e .

# Run the signal engine (connects to Binance, prints signals live)
python run.py

# Specific pairs
python run.py --symbols btcusdt,ethusdt,solusdt

# Only whale + mean reversion alerts
python run.py --quiet

# Run for N seconds
python run.py --duration 60

# Tests
pytest -v
```

---

## Architecture

```
Binance WebSocket (public, no API key needed)
       │
       │  wss://stream.binance.com:9443/stream?streams=...
       │  Combined stream: N symbols x M stream types over 1 connection
       ▼
extraction/binance_ws.py  ─── BinanceStreamClient
       │  Auto-reconnect with exponential backoff + jitter (1s-30s)
       │  Yields raw JSON dicts: {"stream": "btcusdt@trade", "data": {...}}
       ▼
extraction/parsers.py  ─── parse_trade(), parse_book_ticker(), etc.
       │  Raw JSON -> typed frozen dataclasses (TradeEvent, BookTickerEvent, ...)
       │  Decimal precision preserved from Binance string prices
       ▼
models/signals.py  ─── SignalEngine
       │  Stateful engine with rolling windows per symbol (deques)
       │  on_trade() and on_book_ticker() return list[Signal]
       ▼
Signal objects  ─── name, value, symbol, timestamp, metadata
```

**Optional (requires Redis):**
```
extraction/orderbook.py  ─── OrderBookState
  Reconstructs full order book in Redis following official Binance algorithm
  Provides deep OBI (Order Book Imbalance) across N levels
```

---

## File map

| File | Purpose | Key classes/functions |
|---|---|---|
| `run.py` | CLI entry point. Connects to Binance, feeds events into SignalEngine, prints signals. | `main()` |
| `extraction/__init__.py` | Public API re-exports. | All extraction types and functions |
| `extraction/binance_ws.py` | WebSocket client. Single combined-stream connection for all symbols. Exponential backoff reconnection. | `BinanceStreamClient`, `build_combined_stream_url()`, `StreamMetrics` |
| `extraction/parsers.py` | Converts raw Binance JSON payloads into typed frozen dataclasses. | `parse_trade()`, `parse_book_ticker()`, `parse_depth()`, `parse_kline()`, `TradeEvent`, `BookTickerEvent`, `DepthEvent`, `KlineEvent` |
| `extraction/orderbook.py` | Full order book reconstruction in Redis (ZSET + HASH). Follows Binance's official algorithm for incremental updates with gap detection and auto-resync. | `OrderBookState`, `OrderBookMetrics` |
| `models/__init__.py` | Re-exports SignalEngine. | |
| `models/signals.py` | Core signal computation engine. Stateful per-symbol, uses rolling deque windows. No batch -- computes on every event. | `SignalEngine`, `Signal`, `_SymbolState` |
| `tests/test_ws.py` | URL building tests. | |
| `tests/test_parsers.py` | Parser correctness (Decimal precision, timestamps, all 4 event types). | |
| `tests/test_orderbook.py` | Order book reconstruction (uses fakeredis). | |
| `tests/test_signals.py` | Signal engine logic (VWAP, whale detection, RV, signed volume, OBI). | |

---

## Signals reference

### From `on_trade()` (every trade event):

| Signal name | Type | Description |
|---|---|---|
| `vwap_1m` | float | Volume-weighted average price, 1-minute rolling window |
| `vwap_5m` | float | VWAP, 5-minute rolling |
| `vwap_30m` | float | VWAP, 30-minute rolling |
| `vwap_zscore_30m` | float | Z-score of current price vs 30m VWAP. `metadata.mean_reversion=True` when \|z\|>2 |
| `signed_volume_1m` | float | Net signed volume in 1m window. Positive = buyer-taker dominance |
| `order_flow_imbalance_1m` | float | `signed_volume / abs(total_volume)`. Range [-1, +1] |
| `rv_5m` | float | Realized volatility (std of log returns) over 5-min trade window |
| `rv_15m` | float | Realized volatility over 15-min trade window |
| `whale_detected` | float | Notional value of the whale trade. `metadata: {side, threshold, size_vs_threshold}` |

### From `on_book_ticker()` (every best bid/ask update):

| Signal name | Type | Description |
|---|---|---|
| `midprice` | float | `(best_bid + best_ask) / 2` |
| `spread_bps` | float | `(ask - bid) / midprice * 10000` |
| `obi_top1` | float | Order Book Imbalance from top-of-book: `(bid_qty - ask_qty) / (bid_qty + ask_qty)` |

---

## Binance WebSocket details

- **Endpoint:** `wss://stream.binance.com:9443/stream?streams=...`
- **Auth:** None needed. Public market data.
- **Protocol:** Combined stream -- one connection for all symbol+stream combos.
- **Rate limits:** Binance allows 5 connections per IP. We use 1.
- **Message format:** `{"stream": "btcusdt@trade", "data": {<payload>}}`

### Stream types

| Stream | Meaning | Typical frequency |
|---|---|---|
| `trade` | Every matched order (aggressor vs maker) | 10-500/s per pair (BTC highest) |
| `bookTicker` | Best bid/ask update (top of book) | 1-10/s per pair |
| `depth` | Incremental order book changes (1s push) | 1/s per pair |
| `kline_1s` | 1-second OHLCV candle | 1/s per pair |

### Default pairs

```
btcusdt, ethusdt, solusdt, bnbusdt, xrpusdt,
adausdt, avaxusdt, maticusdt, dogeusdt, linkusdt
```

Any Binance Spot pair ending in `usdt` can be added. Just pass the lowercase symbol.

---

## Key design decisions

### Why rolling deques, not pandas DataFrames
The signal engine processes events one at a time as they arrive. Deques with maxlen give O(1) append/popleft and bounded memory. Rebuilding a DataFrame on every tick would be orders of magnitude slower.

### Why Decimal in parsers, float in signals
Parsers preserve Binance's string-based Decimal precision (no floating point drift in price representation). The signal engine uses float because the math (log returns, std, etc.) needs it and the precision loss is irrelevant at the signal level.

### Why the order book needs Redis
A full L2 order book for 10 symbols has ~10k price levels each. Keeping it in-process memory works but Redis provides persistence across restarts and lets other services (like a UI) query it. The `OrderBookState` class is optional -- `SignalEngine` works fine with just `trade` + `bookTicker` streams.

### Whale threshold
Computed as P99 of the notional distribution per symbol. Recalculated every 500 trades after a warmup period (default 1000 trades). This means whales are contextual -- a $10k trade on DOGE is a whale, but not on BTC.

### Reconnection
Exponential backoff: `min(1s * 2^attempts + jitter, 30s)`. Attempt counter resets after 60s of stable connection. This handles both brief hiccups and extended outages.

---

## How to integrate into the parent project

### As an imported module

```python
from extraction import BinanceStreamClient, parse_trade, parse_book_ticker
from models import SignalEngine

engine = SignalEngine()

async def on_message(msg: dict):
    stream = msg["stream"]
    data = msg["data"]
    kind = stream.split("@")[1]

    if kind == "trade":
        evt = parse_trade(data)
        signals = engine.on_trade(
            evt.symbol, float(evt.price), float(evt.quantity),
            evt.is_buyer_maker, evt.trade_time,
        )
        for sig in signals:
            # Feed into your terminal's signal bus, database, UI, etc.
            your_signal_handler(sig)
```

### Getting a snapshot of current state

```python
summary = engine.get_state_summary("BTCUSDT")
# Returns: {last_price, trades_in_1m_window, trades_in_5m_window,
#           whale_threshold, total_trades, last_bid, last_ask}
```

### Adding new symbols at runtime

Instantiate a new `BinanceStreamClient` with the updated symbol list. The `SignalEngine` automatically creates state for any new symbol it sees.

### Adding new signal types

Add a method to `SignalEngine` or extend `on_trade()`/`on_book_ticker()`. The pattern is:
1. Add state to `_SymbolState` if needed (usually a deque).
2. Update state in the event handler.
3. Compute signal value.
4. Append `Signal(symbol, timestamp, "your_signal_name", value, metadata)`.

---

## Performance characteristics

- **Throughput:** Tested at 504 events/sec sustained over 60s (10 pairs, 3 streams). Signal engine adds negligible latency.
- **Memory:** ~50MB for 10 pairs with 30m rolling windows. Bounded by deque maxlen.
- **Latency:** Sub-millisecond from WebSocket message receipt to signal emission.
- **Startup:** Connects in <1s. First signals appear immediately. Whale detection needs ~1000 trades warmup (~2-5 minutes at normal volume).

---

## Dependencies

| Package | Why |
|---|---|
| `websockets` | Binance WebSocket connection |
| `httpx` | REST API calls (order book snapshots) |
| `redis[hiredis]` | Order book state storage (optional, only if using OrderBookState) |
| `structlog` | Structured JSON logging |
| `pandas` | Available for downstream analysis (not used in core engine) |
| `numpy` | Available for downstream analysis (not used in core engine) |

Dev: `ruff`, `pytest`, `pytest-asyncio`, `fakeredis`

---

## Testing

22 unit tests covering:
- WebSocket URL construction (4 tests)
- Event parsing for all 4 stream types + Decimal precision (5 tests)
- Order book reconstruction with fakeredis (6 tests)
- Signal engine: VWAP, signed volume, whale detection, RV, OBI, state (7 tests)

```bash
pytest -v
```

No network or Docker needed for tests. Order book tests use `fakeredis` as an in-memory Redis mock.

---

## Code style

- Python 3.11+ with type hints everywhere.
- `from __future__ import annotations` at top of every module.
- Frozen dataclasses with `slots=True` for event types (immutable, memory-efficient).
- `ruff` for linting and formatting.
- No comments unless explaining a non-obvious why.
- English for code; Spanish OK in user-facing text.
