"""Binance Spot WebSocket client with automatic reconnection.

Reconnects with exponential backoff + jitter. Yields parsed JSON dicts
from the combined stream endpoint.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = structlog.get_logger(__name__)

DEFAULT_WS_URL = "wss://stream.binance.com:9443"
DEFAULT_SYMBOLS = [
    "btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
    "adausdt", "avaxusdt", "maticusdt", "dogeusdt", "linkusdt",
]
DEFAULT_STREAMS = ["trade", "depth", "bookTicker", "kline_1s"]


def build_combined_stream_url(
    symbols: list[str],
    streams: list[str],
    base_url: str = DEFAULT_WS_URL,
) -> str:
    base = base_url.rstrip("/")
    parts: list[str] = []
    for symbol in symbols:
        sym = symbol.lower()
        for stream in streams:
            parts.append(f"{sym}@{stream}")
    streams_str = "/".join(parts)
    return f"{base}/stream?streams={streams_str}"


@dataclass
class StreamMetrics:
    messages_received: int = 0
    reconnects: int = 0
    last_message_ts: float = 0.0
    bytes_received: int = 0
    decode_errors: int = 0
    by_stream_type: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "messages_received": self.messages_received,
            "reconnects": self.reconnects,
            "bytes_received": self.bytes_received,
            "decode_errors": self.decode_errors,
            "by_stream_type": dict(self.by_stream_type),
        }


class BinanceStreamClient:
    """Async WebSocket client with exponential backoff reconnection.

    Usage:
        async with BinanceStreamClient(symbols, streams) as client:
            async for msg in client.iter_messages():
                # msg = {"stream": "btcusdt@trade", "data": {...}}
                ...
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        streams: list[str] | None = None,
        base_url: str = DEFAULT_WS_URL,
        min_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.streams = streams or DEFAULT_STREAMS
        self.url = build_combined_stream_url(self.symbols, self.streams, base_url)
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff
        self.metrics = StreamMetrics()
        self._closed = False
        self._connection: websockets.WebSocketClientProtocol | None = None

    async def __aenter__(self) -> BinanceStreamClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._closed = True
        if self._connection is not None:
            await self._connection.close()

    async def iter_messages(self) -> AsyncIterator[dict[str, Any]]:
        attempts = 0
        while not self._closed:
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=180.0,
                    ping_timeout=60.0,
                    max_size=10 * 1024 * 1024,
                    close_timeout=5,
                ) as ws:
                    self._connection = ws
                    open_at = asyncio.get_event_loop().time()
                    logger.info(
                        "ws_connected",
                        symbols=len(self.symbols),
                        streams=self.streams,
                    )

                    async for raw in ws:
                        if attempts > 0 and (asyncio.get_event_loop().time() - open_at > 60):
                            attempts = 0

                        self.metrics.bytes_received += (
                            len(raw) if isinstance(raw, bytes | str) else 0
                        )
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            self.metrics.decode_errors += 1
                            continue

                        self.metrics.messages_received += 1
                        self.metrics.last_message_ts = asyncio.get_event_loop().time()

                        stream_name = msg.get("stream", "unknown")
                        kind = stream_name.split("@", 1)[1] if "@" in stream_name else stream_name
                        self.metrics.by_stream_type[kind] = (
                            self.metrics.by_stream_type.get(kind, 0) + 1
                        )

                        yield msg

            except (TimeoutError, ConnectionClosed, WebSocketException, OSError) as exc:
                if self._closed:
                    return
                attempts += 1
                self.metrics.reconnects += 1
                delay = min(
                    self.min_backoff * (2 ** (attempts - 1)) + random.uniform(0, 1.0),
                    self.max_backoff,
                )
                logger.warning(
                    "ws_reconnecting",
                    error=str(exc),
                    attempt=attempts,
                    delay=round(delay, 2),
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise
            except asyncio.CancelledError:
                raise
