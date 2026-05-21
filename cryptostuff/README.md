# crypto-microstructure

Real-time crypto market microstructure signal engine. Connects to Binance WebSocket, streams trade + order book data for 10 pairs, and computes quant signals live.

## Quick start

```bash
pip install -e .
python run.py
```

That's it. It connects to Binance, streams data, and prints signals in real time.

## Signals

| Signal | Method | Output |
|---|---|---|
| VWAP (1m, 5m, 30m) | `sum(price * qty) / sum(qty)` rolling | Price level |
| VWAP Z-score | `(price - vwap) / std(prices)` 30m window | Mean reversion alert when \|z\| > 2 |
| Realized Volatility | `std(log returns)` over 5m/15m | Volatility level |
| Signed Volume | `+qty` if buyer-taker, `-qty` if seller-taker | Order flow pressure |
| Order Flow Imbalance | `signed_volume / total_volume` 1m window | -1 to +1 |
| OBI (Order Book Imbalance) | `(bid_qty - ask_qty) / (bid_qty + ask_qty)` | -1 to +1 |
| Whale Detection | `notional >= P99` threshold per symbol | Alert with side + magnitude |
| Spread (bps) | `(ask - bid) / midprice * 10000` | Liquidity measure |
| Midprice | `(bid + ask) / 2` | Fair value |

## Usage

```bash
# All 10 pairs, trade + bookTicker
python run.py

# Specific pairs
python run.py --symbols btcusdt,ethusdt,solusdt

# Only show whale trades and mean reversion alerts
python run.py --quiet

# Run for 60 seconds then stop
python run.py --duration 60
```

## As a library

```python
from extraction import BinanceStreamClient, parse_trade, parse_book_ticker
from models import SignalEngine

engine = SignalEngine()

async with BinanceStreamClient(symbols=["btcusdt"]) as client:
    async for msg in client.iter_messages():
        kind = msg["stream"].split("@")[1]
        data = msg["data"]

        if kind == "trade":
            evt = parse_trade(data)
            signals = engine.on_trade(
                evt.symbol, float(evt.price), float(evt.quantity),
                evt.is_buyer_maker, evt.trade_time,
            )
            for sig in signals:
                print(f"{sig.name}: {sig.value}")
```

## Order book (requires Redis)

```python
from extraction import OrderBookState

ob = OrderBookState(redis_url="redis://localhost:6379/0")
await ob.connect()
await ob.load_snapshot("BTCUSDT")
# Then feed depth updates from the stream...
obi = await ob.get_obi("BTCUSDT", levels=5)
```

## Structure

```
├── run.py                  # Self-sustained entry point
├── extraction/
│   ├── binance_ws.py       # WebSocket client (auto-reconnect)
│   ├── parsers.py          # Raw JSON -> typed dataclasses
│   └── orderbook.py        # Order book reconstruction (Redis)
├── models/
│   └── signals.py          # Real-time signal engine (VWAP, RV, whales, OBI)
└── tests/                  # 25 unit tests
```

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```
