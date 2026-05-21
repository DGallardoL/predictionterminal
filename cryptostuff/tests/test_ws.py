"""Tests for WebSocket URL building."""

from __future__ import annotations

from extraction.binance_ws import build_combined_stream_url


def test_single_stream_single_symbol() -> None:
    url = build_combined_stream_url(["btcusdt"], ["trade"], base_url="wss://x")
    assert url == "wss://x/stream?streams=btcusdt@trade"


def test_multi_symbol_multi_stream() -> None:
    url = build_combined_stream_url(
        ["btcusdt", "ethusdt"],
        ["trade", "bookTicker"],
        base_url="wss://x",
    )
    parts = url.split("?streams=", 1)[1].split("/")
    assert "btcusdt@trade" in parts
    assert "btcusdt@bookTicker" in parts
    assert "ethusdt@trade" in parts
    assert "ethusdt@bookTicker" in parts
    assert len(parts) == 4


def test_lowercases_symbols() -> None:
    url = build_combined_stream_url(["BTCUSDT"], ["trade"], base_url="wss://x")
    assert "btcusdt@trade" in url


def test_strips_trailing_slash() -> None:
    url = build_combined_stream_url(["btcusdt"], ["trade"], base_url="wss://x/")
    assert url == "wss://x/stream?streams=btcusdt@trade"
