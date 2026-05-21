"""Tests for ``pfm.sources.binance`` — REST klines fetcher.

Uses ``respx`` to mock the public ``/api/v3/klines`` endpoint. No real
network calls; deterministic responses verify pagination, timestamp
parsing, retry-on-429, and error handling.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx
import pandas as pd
import pytest
import respx

from pfm.sources.binance import (
    BINANCE_API_BASE,
    BinanceClient,
    BinanceError,
)


def _make_kline(open_time_ms: int, o: float, h: float, lo: float, c: float, v: float) -> list:
    """Build a single Binance-shaped kline row."""
    close_time_ms = open_time_ms + 86_399_999  # daily bar
    return [
        open_time_ms,
        str(o),
        str(h),
        str(lo),
        str(c),
        str(v),
        close_time_ms,
        "0",
        0,
        "0",
        "0",
        "0",
    ]


@pytest.fixture
def http_client() -> Iterator[httpx.Client]:
    with httpx.Client(timeout=5.0) as c:
        yield c


@respx.mock
def test_get_klines_basic_parses_ohlc(http_client: httpx.Client) -> None:
    bar1 = _make_kline(1_700_000_000_000, 100, 110, 95, 105, 12.3)
    bar2 = _make_kline(1_700_086_400_000, 105, 115, 100, 112, 8.4)
    respx.get(BINANCE_API_BASE + "/api/v3/klines").mock(
        return_value=httpx.Response(200, json=[bar1, bar2])
    )
    cli = BinanceClient(client=http_client)
    df = cli.get_klines("BTCUSDT", limit=10)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[0] == 105.0
    assert df["high"].iloc[1] == 115.0
    # Index must be UTC, normalised to midnight.
    assert df.index.tz is not None
    assert df.index[0].hour == 0


@respx.mock
def test_paginates_when_page_full(http_client: httpx.Client) -> None:
    """When a page returns exactly `limit` bars, the client paginates forward."""
    page1 = [_make_kline(1_700_000_000_000 + i * 86_400_000, 1, 2, 0.5, 1.5, 1) for i in range(3)]
    page2 = [_make_kline(1_700_259_200_000 + i * 86_400_000, 2, 3, 1.5, 2.5, 1) for i in range(2)]
    route = respx.get(BINANCE_API_BASE + "/api/v3/klines").mock(
        side_effect=[
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]
    )
    cli = BinanceClient(client=http_client)
    df = cli.get_klines("BTCUSDT", limit=3)
    assert route.call_count == 2
    assert len(df) == 5


@respx.mock
def test_window_filtering(http_client: httpx.Client) -> None:
    bars = [
        _make_kline(1_700_000_000_000, 1, 1, 1, 1, 1),  # 2023-11-14
        _make_kline(1_700_086_400_000, 2, 2, 2, 2, 1),  # 2023-11-15
        _make_kline(1_700_172_800_000, 3, 3, 3, 3, 1),  # 2023-11-16
    ]
    respx.get(BINANCE_API_BASE + "/api/v3/klines").mock(return_value=httpx.Response(200, json=bars))
    cli = BinanceClient(client=http_client)
    df = cli.get_klines(
        "BTCUSDT",
        start=pd.Timestamp("2023-11-15", tz="UTC"),
        end=pd.Timestamp("2023-11-16", tz="UTC"),
        limit=100,
    )
    assert len(df) == 2
    assert df["close"].iloc[0] == 2.0
    assert df["close"].iloc[1] == 3.0


@respx.mock
def test_429_retries_then_succeeds(http_client: httpx.Client, monkeypatch) -> None:
    # Avoid actual sleeps in the retry loop.
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    bar = _make_kline(1_700_000_000_000, 1, 2, 0.5, 1.5, 1)
    route = respx.get(BINANCE_API_BASE + "/api/v3/klines").mock(
        side_effect=[
            httpx.Response(429, text="rate"),
            httpx.Response(429, text="rate"),
            httpx.Response(200, json=[bar]),
        ]
    )
    cli = BinanceClient(client=http_client, max_retries=5)
    df = cli.get_klines("BTCUSDT", limit=10)
    assert route.call_count == 3
    assert len(df) == 1


@respx.mock
def test_persistent_429_raises(http_client: httpx.Client, monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    respx.get(BINANCE_API_BASE + "/api/v3/klines").mock(
        return_value=httpx.Response(429, text="rate"),
    )
    cli = BinanceClient(client=http_client, max_retries=3)
    with pytest.raises(BinanceError, match="rate-limit/server error"):
        cli.get_klines("BTCUSDT", limit=10)


@respx.mock
def test_400_status_raises(http_client: httpx.Client) -> None:
    respx.get(BINANCE_API_BASE + "/api/v3/klines").mock(
        return_value=httpx.Response(400, text='{"code":-1100,"msg":"bad symbol"}'),
    )
    cli = BinanceClient(client=http_client)
    with pytest.raises(BinanceError, match="400"):
        cli.get_klines("BTCUSDT", limit=10)


def test_invalid_limit_raises(http_client: httpx.Client) -> None:
    cli = BinanceClient(client=http_client)
    with pytest.raises(ValueError, match=r"limit must be in"):
        cli.get_klines("BTCUSDT", limit=0)
    with pytest.raises(ValueError, match=r"limit must be in"):
        cli.get_klines("BTCUSDT", limit=2000)


@respx.mock
def test_5min_interval_preserves_time_of_day(http_client: httpx.Client) -> None:
    """Sub-daily intervals must NOT normalise the index to midnight."""
    from pfm.sources.binance import annualisation_for_interval

    # Three 5-minute bars on 2025-04-01 starting at 12:00 UTC.
    base = 1_743_508_800_000  # 2025-04-01T12:00:00 UTC in ms
    bars = [
        _make_kline(base + 0 * 300_000, 100, 101, 99, 100.5, 1),
        _make_kline(base + 1 * 300_000, 100.5, 102, 100, 101.5, 1),
        _make_kline(base + 2 * 300_000, 101.5, 103, 101, 102.5, 1),
    ]
    respx.get(BINANCE_API_BASE + "/api/v3/klines").mock(
        return_value=httpx.Response(200, json=bars),
    )
    cli = BinanceClient(client=http_client)
    df = cli.get_klines("BTCUSDT", interval="5m", limit=10)
    assert len(df) == 3
    # Indexes must differ by 5 minutes — no normalisation.
    assert (df.index[1] - df.index[0]).total_seconds() == 300
    assert df.index[0].hour == 12
    # Annualisation table must agree.
    assert annualisation_for_interval("5m") == 105_120.0
    assert annualisation_for_interval("1d") == 365.0


@respx.mock
def test_unknown_interval_raises(http_client: httpx.Client) -> None:
    cli = BinanceClient(client=http_client)
    with pytest.raises(ValueError, match=r"unknown interval"):
        cli.get_klines("BTCUSDT", interval="42s", limit=10)


@respx.mock
def test_empty_response_returns_empty_df(http_client: httpx.Client) -> None:
    respx.get(BINANCE_API_BASE + "/api/v3/klines").mock(
        return_value=httpx.Response(200, json=[]),
    )
    cli = BinanceClient(client=http_client)
    df = cli.get_klines("BTCUSDT", limit=10)
    assert df.empty


# ---------------------------------------------------------------------------
# 2026-05-15 upstream-hardening: process-local klines cache (TTL 5 min)
# ---------------------------------------------------------------------------


@respx.mock
def test_get_klines_second_identical_call_hits_cache(http_client: httpx.Client) -> None:
    """Two get_klines calls with the same key share one upstream call.

    The 5-min TTL is intentional: re-running a Terminal/crypto-micro fit
    on the same symbol+window should not re-hit Binance.
    """
    bar = _make_kline(1_700_000_000_000, 1, 2, 0.5, 1.5, 1)
    route = respx.get(BINANCE_API_BASE + "/api/v3/klines").mock(
        return_value=httpx.Response(200, json=[bar])
    )
    cli = BinanceClient(client=http_client)
    df1 = cli.get_klines("BTCUSDT", limit=10)
    df2 = cli.get_klines("BTCUSDT", limit=10)
    assert route.call_count == 1, f"expected cache hit; got {route.call_count} upstream calls"
    assert len(df1) == 1 and len(df2) == 1
    # Returned frames must be independent copies (caller can mutate freely).
    assert df1 is not df2
