"""Self-sustained real-time signal engine.

Run this and it connects to Binance, streams data, and prints signals live.

    python run.py
    python run.py --symbols btcusdt,ethusdt --streams trade,bookTicker
"""

from __future__ import annotations

import argparse
import asyncio
import signal as signal_mod
import sys
from datetime import UTC, datetime

import structlog

from extraction.binance_ws import BinanceStreamClient, DEFAULT_SYMBOLS, DEFAULT_STREAMS
from extraction.parsers import parse_book_ticker, parse_trade
from models.signals import SignalEngine

logger = structlog.get_logger("runner")


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time crypto signal engine")
    parser.add_argument("--symbols", type=str, default=None, help="CSV of symbols")
    parser.add_argument("--streams", type=str, default=None, help="CSV of streams")
    parser.add_argument("--duration", type=float, default=None, help="Run for N seconds then stop")
    parser.add_argument("--quiet", action="store_true", help="Only print whale/mean-reversion signals")
    args = parser.parse_args()

    setup_logging()

    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS
    streams = args.streams.split(",") if args.streams else ["trade", "bookTicker"]

    engine = SignalEngine()
    shutdown = asyncio.Event()

    loop = asyncio.get_event_loop()
    for sig in (signal_mod.SIGTERM, signal_mod.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            pass

    client = BinanceStreamClient(symbols=symbols, streams=streams)
    msg_count = 0
    signal_count = 0
    start_time = asyncio.get_event_loop().time()

    logger.info("starting", symbols=len(symbols), streams=streams)

    async def watchdog() -> None:
        if args.duration:
            await asyncio.sleep(args.duration)
            shutdown.set()

    async def stats_printer() -> None:
        while not shutdown.is_set():
            await asyncio.sleep(10)
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(
                "stats",
                messages=msg_count,
                signals=signal_count,
                evt_per_sec=round(msg_count / max(elapsed, 1), 1),
                elapsed=round(elapsed, 1),
            )

    async def consume() -> None:
        nonlocal msg_count, signal_count
        async with client:
            async for envelope in client.iter_messages():
                if shutdown.is_set():
                    break

                stream_name = envelope.get("stream", "")
                data = envelope.get("data", {})
                if not data:
                    continue

                msg_count += 1
                kind = stream_name.split("@", 1)[1] if "@" in stream_name else stream_name
                now = datetime.now(UTC)

                signals = []
                if kind == "trade":
                    evt = parse_trade(data)
                    signals = engine.on_trade(
                        evt.symbol,
                        float(evt.price),
                        float(evt.quantity),
                        evt.is_buyer_maker,
                        now,
                    )
                elif kind == "bookTicker":
                    evt = parse_book_ticker(data)
                    signals = engine.on_book_ticker(
                        evt.symbol,
                        float(evt.best_bid_price),
                        float(evt.best_bid_qty),
                        float(evt.best_ask_price),
                        float(evt.best_ask_qty),
                        now,
                    )

                for sig in signals:
                    signal_count += 1
                    if args.quiet and sig.name not in ("whale_detected", "vwap_zscore_30m"):
                        continue
                    if sig.name == "whale_detected":
                        print(
                            f"  WHALE  {sig.symbol:>10s}  "
                            f"${sig.value:>12,.2f}  "
                            f"{sig.metadata.get('side', '?'):>4s}  "
                            f"{sig.metadata.get('size_vs_threshold', 0):.1f}x threshold"
                        )
                    elif sig.name == "vwap_zscore_30m" and sig.metadata.get("mean_reversion"):
                        direction = "ABOVE" if sig.value > 0 else "BELOW"
                        print(
                            f"  REVERT {sig.symbol:>10s}  "
                            f"z={sig.value:>+7.2f}  {direction} VWAP"
                        )
                    elif not args.quiet and kind == "trade" and sig.name in ("vwap_1m", "rv_5m", "order_flow_imbalance_1m"):
                        print(
                            f"  {sig.name:>28s}  {sig.symbol:>10s}  {sig.value:>14.6f}"
                        )

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(consume(), name="consumer")
            tg.create_task(stats_printer(), name="stats")
            if args.duration:
                tg.create_task(watchdog(), name="watchdog")
    except* asyncio.CancelledError:
        pass

    await client.close()
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(
        "stopped",
        messages=msg_count,
        signals=signal_count,
        elapsed=round(elapsed, 1),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
