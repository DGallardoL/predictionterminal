"""Cascade fallback for ``pfm.sources.equity.get_log_returns``.

Yfinance fails → Tiingo is tried; Tiingo fails → Stooq is tried.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from pfm.sources import equity as equity_mod
from pfm.sources.equity import (
    EquityDataError,
    get_log_returns,
)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """Each test gets a fresh delisted registry and a wiped equity cache."""
    monkeypatch.setattr(
        equity_mod,
        "DELISTED_REGISTRY_PATH",
        tmp_path / "delisted.json",
    )
    equity_mod._EQUITY_CACHE.clear()
    yield
    equity_mod._EQUITY_CACHE.clear()


def _sample_closes() -> pd.Series:
    idx = pd.date_range("2025-01-02", periods=5, freq="B", tz="UTC")
    return pd.Series([100.0, 101.0, 102.0, 101.5, 103.0], index=idx, name="Close")


def _sample_tiingo_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-02", periods=5, freq="B", tz="UTC").normalize()
    return pd.DataFrame(
        {
            "open": [100, 100.5, 101, 101.2, 101.7],
            "high": [101, 101.5, 102.5, 102.0, 103.5],
            "low": [99, 100.0, 100.5, 101.0, 101.5],
            "close": [100.0, 101.0, 102.0, 101.5, 103.0],
            "adjClose": [100.0, 101.0, 102.0, 101.5, 103.0],
            "volume": [1, 2, 3, 4, 5],
        },
        index=idx,
    )


def _sample_stooq_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-02", periods=5, freq="B", tz="UTC").normalize()
    return pd.DataFrame(
        {
            "open": [100, 100.5, 101, 101.2, 101.7],
            "high": [101, 101.5, 102.5, 102.0, 103.5],
            "low": [99, 100.0, 100.5, 101.0, 101.5],
            "close": [100.0, 101.0, 102.0, 101.5, 103.0],
            "volume": [1, 2, 3, 4, 5],
        },
        index=idx,
    )


def test_yfinance_succeeds_no_fallback() -> None:
    """Happy path: yfinance returns data, neither fallback is invoked."""
    closes = _sample_closes()

    with (
        patch.object(equity_mod, "_yfinance_closes", return_value=closes) as yf_mock,
        patch.object(equity_mod, "_try_tiingo") as tiingo_mock,
        patch.object(equity_mod, "_try_stooq") as stooq_mock,
    ):
        ret = get_log_returns(
            "NVDA",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-10", tz="UTC"),
        )

    assert yf_mock.called
    assert not tiingo_mock.called
    assert not stooq_mock.called
    assert len(ret) == 4  # 5 closes → 4 returns
    assert ret.name == "r"


def test_falls_back_to_tiingo_when_yfinance_fails(monkeypatch) -> None:
    """yfinance raises → Tiingo fetches successfully."""
    monkeypatch.setenv("TIINGO_API_KEY", "fake-token")

    with (
        patch.object(
            equity_mod,
            "_yfinance_closes",
            side_effect=EquityDataError("yf down"),
        ) as yf_mock,
        patch.object(
            equity_mod,
            "_check_delisted_via_yf_info",
            return_value=False,
        ),
        patch(
            "pfm.sources.equity.tiingo_src.fetch_daily_prices",
            return_value=_sample_tiingo_df(),
        ) as tiingo_mock,
        patch.object(equity_mod, "_try_stooq") as stooq_mock,
    ):
        ret = get_log_returns(
            "NVDA",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-10", tz="UTC"),
        )

    assert yf_mock.called
    assert tiingo_mock.called
    assert not stooq_mock.called
    assert len(ret) == 4


def test_falls_back_to_stooq_when_yfinance_and_tiingo_fail(monkeypatch) -> None:
    """yfinance + Tiingo fail → Stooq fetches successfully."""
    monkeypatch.setenv("TIINGO_API_KEY", "fake-token")

    with (
        patch.object(
            equity_mod,
            "_yfinance_closes",
            side_effect=EquityDataError("yf down"),
        ),
        patch.object(
            equity_mod,
            "_check_delisted_via_yf_info",
            return_value=False,
        ),
        patch(
            "pfm.sources.equity.tiingo_src.fetch_daily_prices",
            side_effect=RuntimeError("tiingo down"),
        ) as tiingo_mock,
        patch(
            "pfm.sources.equity.stooq_src.fetch_daily_prices",
            return_value=_sample_stooq_df(),
        ) as stooq_mock,
    ):
        ret = get_log_returns(
            "NVDA",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-10", tz="UTC"),
        )

    assert tiingo_mock.called
    assert stooq_mock.called
    assert len(ret) == 4


def test_skip_tiingo_when_no_api_key(monkeypatch) -> None:
    """Without TIINGO_API_KEY the Tiingo fallback is silently skipped."""
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)

    with (
        patch.object(
            equity_mod,
            "_yfinance_closes",
            side_effect=EquityDataError("yf down"),
        ),
        patch.object(
            equity_mod,
            "_check_delisted_via_yf_info",
            return_value=False,
        ),
        patch(
            "pfm.sources.equity.tiingo_src.fetch_daily_prices",
        ) as tiingo_inner_mock,
        patch(
            "pfm.sources.equity.stooq_src.fetch_daily_prices",
            return_value=_sample_stooq_df(),
        ) as stooq_mock,
    ):
        ret = get_log_returns(
            "NVDA",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-10", tz="UTC"),
        )

    # Tiingo HTTP function must NOT be called when no key.
    assert not tiingo_inner_mock.called
    assert stooq_mock.called
    assert len(ret) == 4


def test_all_sources_fail_raises_with_breakdown(monkeypatch) -> None:
    """Every source fails → EquityDataError with per-source detail."""
    monkeypatch.setenv("TIINGO_API_KEY", "fake-token")

    with (
        patch.object(
            equity_mod,
            "_yfinance_closes",
            side_effect=EquityDataError("yf down"),
        ),
        patch.object(
            equity_mod,
            "_check_delisted_via_yf_info",
            return_value=False,
        ),
        patch(
            "pfm.sources.equity.tiingo_src.fetch_daily_prices",
            side_effect=RuntimeError("tiingo down"),
        ),
        patch(
            "pfm.sources.equity.stooq_src.fetch_daily_prices",
            side_effect=RuntimeError("stooq down"),
        ),
        pytest.raises(EquityDataError) as exc_info,
    ):
        get_log_returns(
            "ZZZZ",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-10", tz="UTC"),
        )

    msg = str(exc_info.value)
    assert "yfinance" in msg
    assert "tiingo" in msg
    assert "stooq" in msg
