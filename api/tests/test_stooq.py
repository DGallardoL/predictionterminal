"""Tests for ``pfm.sources.stooq``."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx

from pfm.sources.stooq import STOOQ_BASE, StooqError, fetch_daily_prices

SAMPLE_CSV = """Date,Open,High,Low,Close,Volume
2025-01-02,100.00,101.00,99.50,100.50,1000000
2025-01-03,100.50,102.00,100.00,101.70,1200000
2025-01-06,101.70,103.00,101.00,102.50,1100000
"""


@respx.mock
def test_fetch_daily_prices_basic() -> None:
    route = respx.get(STOOQ_BASE).mock(return_value=httpx.Response(200, text=SAMPLE_CSV))
    df = fetch_daily_prices(
        "AAPL",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-10", tz="UTC"),
    )
    assert route.called
    # Confirm symbol normalisation appended .us.
    sent = route.calls[0].request
    assert "s=aapl.us" in sent.url.query.decode()
    assert "i=d" in sent.url.query.decode()

    assert "close" in df.columns
    assert len(df) == 3
    assert df["close"].iloc[0] == 100.50
    assert df.index[0] == pd.Timestamp("2025-01-02", tz="UTC")


@respx.mock
def test_passthrough_when_dot_in_ticker() -> None:
    """Already-suffixed ticker (e.g. 'aapl.us', 'eurusd') passes through."""
    route = respx.get(STOOQ_BASE).mock(return_value=httpx.Response(200, text=SAMPLE_CSV))
    fetch_daily_prices(
        "aapl.us",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-10", tz="UTC"),
    )
    sent = route.calls[0].request
    assert "s=aapl.us" in sent.url.query.decode()


@respx.mock
def test_no_data_response_raises() -> None:
    """Stooq sends ``"No data"`` body with 200 OK for unknown tickers."""
    respx.get(STOOQ_BASE).mock(return_value=httpx.Response(200, text="No data found\n"))
    with pytest.raises(StooqError, match="no data"):
        fetch_daily_prices(
            "ZZZZ",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-10", tz="UTC"),
        )


@respx.mock
def test_http_error_raises() -> None:
    respx.get(STOOQ_BASE).mock(return_value=httpx.Response(503, text="Down"))
    with pytest.raises(StooqError, match="503"):
        fetch_daily_prices(
            "AAPL",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-10", tz="UTC"),
        )


@respx.mock
def test_malformed_csv_raises() -> None:
    """CSV without required columns raises."""
    respx.get(STOOQ_BASE).mock(return_value=httpx.Response(200, text="Foo,Bar\n1,2\n"))
    with pytest.raises(StooqError, match="missing required"):
        fetch_daily_prices(
            "AAPL",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-10", tz="UTC"),
        )


# ---------------------------------------------------------------------------
# 2026-05-15 upstream-hardening: single retry on transient 429 / 5xx
# ---------------------------------------------------------------------------


@respx.mock
def test_retries_once_on_429_then_succeeds(monkeypatch) -> None:
    """A transient 429 followed by a 200 surfaces as a 200, not a StooqError.

    Stooq is auth-free; the retry covers the "two parallel requests collided"
    case rather than a sustained rate-limit budget.
    """
    import pfm.sources.stooq as st

    monkeypatch.setattr(st, "_RETRY_BACKOFF_S", 0.01)

    route = respx.get(STOOQ_BASE).mock(
        side_effect=[
            httpx.Response(429, text="rate-limit"),
            httpx.Response(200, text=SAMPLE_CSV),
        ]
    )
    df = fetch_daily_prices(
        "AAPL",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-10", tz="UTC"),
    )
    assert route.call_count == 2
    assert not df.empty
