"""Tests for ``pfm.sources.tiingo``."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx

from pfm.sources.tiingo import TIINGO_BASE, TiingoError, fetch_daily_prices

SAMPLE_PAYLOAD = [
    {
        "date": "2025-01-02T00:00:00.000Z",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "adjClose": 100.5,
        "volume": 1_000_000,
    },
    {
        "date": "2025-01-03T00:00:00.000Z",
        "open": 100.5,
        "high": 102.0,
        "low": 100.0,
        "close": 101.7,
        "adjClose": 101.7,
        "volume": 1_200_000,
    },
]


@respx.mock
def test_fetch_daily_prices_basic() -> None:
    route = respx.get(f"{TIINGO_BASE}/NVDA/prices").mock(
        return_value=httpx.Response(200, json=SAMPLE_PAYLOAD)
    )
    df = fetch_daily_prices(
        "NVDA",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-05", tz="UTC"),
        api_key="fake-token",
    )
    assert route.called
    # Confirm auth header.
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Token fake-token"

    assert list(df.columns) == ["open", "high", "low", "close", "adjClose", "volume"]
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2025-01-02", tz="UTC")
    assert df["close"].iloc[0] == 100.5


@respx.mock
def test_missing_api_key_raises() -> None:
    with pytest.raises(TiingoError, match="api_key"):
        fetch_daily_prices(
            "NVDA",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-05", tz="UTC"),
            api_key="",
        )


@respx.mock
def test_http_error_raises() -> None:
    respx.get(f"{TIINGO_BASE}/BADTKR/prices").mock(
        return_value=httpx.Response(404, text="Not found")
    )
    with pytest.raises(TiingoError, match="404"):
        fetch_daily_prices(
            "BADTKR",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-05", tz="UTC"),
            api_key="fake-token",
        )


@respx.mock
def test_empty_payload_raises() -> None:
    respx.get(f"{TIINGO_BASE}/NVDA/prices").mock(return_value=httpx.Response(200, json=[]))
    with pytest.raises(TiingoError, match="empty"):
        fetch_daily_prices(
            "NVDA",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-05", tz="UTC"),
            api_key="fake-token",
        )


@respx.mock
def test_missing_date_field_raises() -> None:
    respx.get(f"{TIINGO_BASE}/NVDA/prices").mock(
        return_value=httpx.Response(200, json=[{"close": 100.0}])
    )
    with pytest.raises(TiingoError, match="date"):
        fetch_daily_prices(
            "NVDA",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-05", tz="UTC"),
            api_key="fake-token",
        )


@respx.mock
def test_uppercases_ticker_in_url() -> None:
    """Tiingo URL paths use upper-case tickers."""
    route = respx.get(f"{TIINGO_BASE}/AAPL/prices").mock(
        return_value=httpx.Response(200, json=SAMPLE_PAYLOAD)
    )
    fetch_daily_prices(
        "aapl",  # lowercase in
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-05", tz="UTC"),
        api_key="fake-token",
    )
    assert route.called


# ---------------------------------------------------------------------------
# 2026-05-15 upstream-hardening: single 429 retry
# ---------------------------------------------------------------------------


@respx.mock
def test_retries_once_on_429_then_succeeds(monkeypatch) -> None:
    """A 429 followed by a 200 surfaces as a 200, not a TiingoError."""
    import pfm.sources.tiingo as tg

    monkeypatch.setattr(tg, "_RETRY_BACKOFF_S", 0.01)

    route = respx.get(f"{TIINGO_BASE}/NVDA/prices").mock(
        side_effect=[
            httpx.Response(429, text="rate-limit"),
            httpx.Response(200, json=SAMPLE_PAYLOAD),
        ]
    )
    df = fetch_daily_prices(
        "NVDA",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-05", tz="UTC"),
        api_key="fake-token",
    )
    assert route.call_count == 2
    assert not df.empty
