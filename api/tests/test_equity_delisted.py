"""Delisted-ticker detection and registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pfm.sources import equity as equity_mod
from pfm.sources.equity import (
    EquityDelistedError,
    get_log_returns,
    is_delisted,
    list_delisted,
    mark_delisted,
)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    monkeypatch.setattr(
        equity_mod,
        "DELISTED_REGISTRY_PATH",
        tmp_path / "delisted.json",
    )
    equity_mod._EQUITY_CACHE.clear()
    yield
    equity_mod._EQUITY_CACHE.clear()


def test_empty_yfinance_with_no_market_price_raises_delisted() -> None:
    """yfinance returns empty + ``regularMarketPrice=None`` → EquityDelistedError."""
    fake_info = {"regularMarketPrice": None, "shortName": "Dead Co"}
    fake_ticker = MagicMock(info=fake_info)

    with (
        patch.object(
            equity_mod,
            "yf",
            download=MagicMock(return_value=pd.DataFrame()),
            Ticker=MagicMock(return_value=fake_ticker),
        ),
        pytest.raises(EquityDelistedError) as exc_info,
    ):
        get_log_returns(
            "DEADCO",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-10", tz="UTC"),
        )
    assert exc_info.value.ticker == "DEADCO"


def test_delisted_persists_to_registry() -> None:
    """A confirmed delist is appended to the on-disk JSON registry."""
    fake_info = {"regularMarketPrice": None}
    fake_ticker = MagicMock(info=fake_info)

    with (
        patch.object(
            equity_mod,
            "yf",
            download=MagicMock(return_value=pd.DataFrame()),
            Ticker=MagicMock(return_value=fake_ticker),
        ),
        pytest.raises(EquityDelistedError),
    ):
        get_log_returns(
            "ZOMBIE",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-10", tz="UTC"),
        )

    assert is_delisted("ZOMBIE")
    assert "ZOMBIE" in list_delisted()


def test_delisted_short_circuits_on_subsequent_call() -> None:
    """Once registered, subsequent calls don't even hit yfinance."""
    mark_delisted("GHOST")

    download_mock = MagicMock()
    with (
        patch.object(equity_mod, "yf", download=download_mock, Ticker=MagicMock()),
        pytest.raises(EquityDelistedError),
    ):
        get_log_returns(
            "GHOST",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-10", tz="UTC"),
        )

    assert not download_mock.called  # short-circuited via registry


def test_mark_delisted_idempotent() -> None:
    mark_delisted("AAPL")
    mark_delisted("AAPL")
    mark_delisted("aapl")  # case-insensitive
    assert list_delisted().count("AAPL") == 1


def test_empty_yfinance_without_delist_signal_falls_through(monkeypatch) -> None:
    """Empty yfinance + ``regularMarketPrice`` present → falls through to next source."""
    monkeypatch.setenv("TIINGO_API_KEY", "fake-token")

    fake_info = {"regularMarketPrice": 250.0}
    fake_ticker = MagicMock(info=fake_info)

    idx = pd.date_range("2025-01-02", periods=5, freq="B", tz="UTC").normalize()
    tiingo_df = pd.DataFrame(
        {
            "open": [1.0] * 5,
            "high": [1.0] * 5,
            "low": [1.0] * 5,
            "close": [100.0, 101.0, 102.0, 101.5, 103.0],
            "adjClose": [100.0, 101.0, 102.0, 101.5, 103.0],
            "volume": [1] * 5,
        },
        index=idx,
    )

    with (
        patch.object(
            equity_mod,
            "yf",
            download=MagicMock(return_value=pd.DataFrame()),
            Ticker=MagicMock(return_value=fake_ticker),
        ),
        patch(
            "pfm.sources.equity.tiingo_src.fetch_daily_prices",
            return_value=tiingo_df,
        ) as tiingo_mock,
    ):
        ret = get_log_returns(
            "ALIVE",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-10", tz="UTC"),
        )

    assert tiingo_mock.called
    assert len(ret) == 4
    assert not is_delisted("ALIVE")
