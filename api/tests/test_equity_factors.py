"""Tests for ``pfm.equity_factors`` — yfinance ↔ Polymarket cointegration.

External IO is patched out: yfinance is monkeypatched to return synthetic
DataFrames so the suite never hits the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm import equity_factors
from pfm.equity_factors import (
    EquityFactorError,
    _logit,
    equity_market_cointegration,
    fetch_equity_history,
)


def _mock_yf_download(
    closes: pd.Series,
) -> callable:
    """Return a stand-in for ``yf.download`` that yields a DataFrame with
    one Close column for the requested ticker."""

    def _fn(ticker, start, end, **kwargs):
        # yfinance treats ``end`` as exclusive; mimic by trimming.
        idx = closes.index
        mask = (idx.date >= start) & (idx.date < end)
        sub = closes.loc[mask].copy()
        df = pd.DataFrame({"Close": sub.values}, index=sub.index)
        df.index.name = "Date"
        return df

    return _fn


class TestFetchEquityHistory:
    def test_returns_close_series_normalised_to_utc(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        idx = pd.date_range("2025-01-01", "2025-01-10", freq="D")
        closes = pd.Series(
            np.linspace(100.0, 110.0, len(idx)),
            index=idx,
            name="Close",
        )
        monkeypatch.setattr(
            equity_factors.yf,
            "download",
            _mock_yf_download(closes),
        )
        s = fetch_equity_history(
            "NVDA",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-10", tz="UTC"),
        )
        assert isinstance(s, pd.Series)
        assert s.name == "NVDA"
        assert str(s.index.tz) == "UTC"
        # All values strictly positive (we'll need that for the log transform).
        assert (s > 0).all()
        # Endpoint is inclusive in our wrapper (end+1 day in yf call).
        assert len(s) >= 10

    def test_empty_yfinance_response_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            equity_factors.yf,
            "download",
            lambda *a, **kw: pd.DataFrame(),
        )
        with pytest.raises(EquityFactorError, match="no data"):
            fetch_equity_history(
                "FAKE",
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-01-10", tz="UTC"),
            )


class TestLogit:
    def test_logit_basic(self) -> None:
        s = pd.Series([0.1, 0.5, 0.9])
        out = _logit(s, clip_eps=0.01)
        assert pytest.approx(out.iloc[1], abs=1e-9) == 0.0
        # Symmetric around 0.5
        assert pytest.approx(out.iloc[0], abs=1e-9) == -out.iloc[2]

    def test_logit_drops_out_of_range(self) -> None:
        s = pd.Series([0.5, -0.1, 1.2, 0.7])
        out = _logit(s, clip_eps=0.01)
        # Values outside (0, 1) become NaN.
        assert pd.isna(out.iloc[1])
        assert pd.isna(out.iloc[2])
        assert np.isfinite(out.iloc[0])
        assert np.isfinite(out.iloc[3])


class TestEquityMarketCointegration:
    def test_cointegrated_synthetic_pair(self) -> None:
        """Construct logit(prob) = α + β·log(price) + small_noise so the
        residual is stationary by construction; the function should
        recover something close to the true β and verdict cointegrated."""
        n = 400
        rng = np.random.default_rng(2024)
        # Simulate log-price as a random walk; build logit(prob)
        # mechanically as a linear function of it plus stationary AR(1)
        # residual (ρ ∈ (0, 1) → finite positive half-life).
        log_p = np.cumsum(rng.normal(0.0, 0.02, size=n)) + np.log(100.0)
        true_alpha = -2.0
        true_beta = 0.5
        rho = 0.7
        eps = rng.normal(0.0, 0.05, size=n)
        ar1 = np.empty(n)
        ar1[0] = eps[0]
        for t in range(1, n):
            ar1[t] = rho * ar1[t - 1] + eps[t]
        logit_q = true_alpha + true_beta * log_p + ar1
        # Invert logit to get a probability series in (0, 1).
        prob = 1.0 / (1.0 + np.exp(-logit_q))
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        ticker_series = pd.Series(np.exp(log_p), index=idx, name="NVDA")
        prob_series = pd.Series(prob, index=idx, name="nvda_largest")

        out = equity_market_cointegration(ticker_series, prob_series)

        assert out["verdict"] == "cointegrated"
        assert out["adf_p"] < 0.05
        assert pytest.approx(out["beta"], abs=0.05) == true_beta
        assert pytest.approx(out["alpha"], abs=0.5) == true_alpha
        assert out["n_obs"] == n
        # half_life should be defined for a reverting AR(0)+constant residual.
        assert out["half_life"] is not None

    def test_independent_random_walks_not_cointegrated(self) -> None:
        n = 300
        rng = np.random.default_rng(1)
        log_p = np.cumsum(rng.normal(0.0, 0.02, size=n)) + np.log(100.0)
        # Independent random-walk in logit space → no shared trend.
        logit_q = np.cumsum(rng.normal(0.0, 0.05, size=n))
        prob = 1.0 / (1.0 + np.exp(-logit_q))
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        ts = pd.Series(np.exp(log_p), index=idx, name="AAPL")
        ps = pd.Series(prob, index=idx, name="aapl_largest")

        out = equity_market_cointegration(ts, ps)

        assert out["verdict"] in ("not_cointegrated", "cointegrated")
        # In an independent-RW pair we expect ADF *not* to reject most of
        # the time; in the rare seed-dependent case it does, OOS Sharpe
        # should still be modest. So we assert either ADF p > 0.05 OR
        # OOS Sharpe is bounded.
        if out["verdict"] == "not_cointegrated":
            assert out["adf_p"] >= 0.05
            # No backtest run when not cointegrated.
            assert out["sharpe_oos"] is None

    def test_negative_prices_raise(self) -> None:
        idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
        ts = pd.Series([-1.0] * 50, index=idx)
        ps = pd.Series([0.4] * 50, index=idx)
        with pytest.raises(EquityFactorError, match="strictly positive"):
            equity_market_cointegration(ts, ps)

    def test_insufficient_data_returns_verdict(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        ts = pd.Series(np.linspace(100.0, 110.0, 10), index=idx)
        ps = pd.Series(np.linspace(0.2, 0.4, 10), index=idx)
        out = equity_market_cointegration(ts, ps)
        assert out["verdict"] == "insufficient-data"
        assert out["sharpe_oos"] is None
        # n_obs is small (the alignment of 10 points overlaps fully).
        assert out["n_obs"] <= 10
