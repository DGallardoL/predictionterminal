"""Tests for ``pfm.terminal_macro_overlay`` — the /terminal/macro-overlay/{slug}.

External dependencies (FRED HTTP, yfinance, Polymarket) are patched out
so the suite never touches the network. The router is mounted on a
fresh :class:`FastAPI` app to avoid the full ``pfm.main`` lifespan.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_macro_overlay
from pfm.terminal_macro_overlay import (
    best_lag,
    get_polymarket_client,
    router,
)

# --- fakes ------------------------------------------------------------------


class _FakePoly:
    """Sentinel — the router only passes the client through to
    :func:`fetch_factor_history`, which is monkeypatched wholesale."""


def _make_macro_series(
    days: int = 180,
    start_value: float = 4.0,
    slope: float = 0.005,
    name: str = "MACRO",
) -> pd.Series:
    """A monotonically rising synthetic macro series indexed by UTC date."""
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    values = start_value + slope * np.arange(days)
    return pd.Series(values, index=idx, name=name)


def _make_prob_history(days: int = 180, low: float = 0.20, high: float = 0.80) -> pd.DataFrame:
    """A factor-history DataFrame matching ``fetch_factor_history``'s contract."""
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    idx.name = "date"
    prices = np.linspace(low, high, days)
    return pd.DataFrame({"price": prices}, index=idx)


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Mount the router on a bare FastAPI app with all IO patched out."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_polymarket_client] = _FakePoly

    monkeypatch.setattr(
        terminal_macro_overlay,
        "fetch_fred_series",
        lambda series_id, start, end, **_: _make_macro_series(name=series_id),
    )
    monkeypatch.setattr(
        terminal_macro_overlay,
        "fetch_equity_history",
        lambda ticker, start, end, **_: _make_macro_series(
            start_value=400.0,
            slope=0.5,
            name=ticker,
        ),
    )
    monkeypatch.setattr(
        terminal_macro_overlay,
        "fetch_factor_history",
        lambda _client, _slug, start=None, end=None: _make_prob_history(),
    )

    with TestClient(app) as client:
        yield client


# --- tests ------------------------------------------------------------------


class TestMacroOverlayEndpoint:
    def test_fed_market_routes_to_dgs10(self, app_client: TestClient) -> None:
        """``fed_cuts_2026`` should overlay DGS10 (FRED) as the primary
        ticker, with SPY listed as an additional toggle option."""
        r = app_client.get("/terminal/macro-overlay/fed_cuts_2026?days=180")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["polymarket_slug"] == "fed_cuts_2026"
        assert body["macro_ticker"] == "DGS10"
        assert body["macro_source"] == "fred"
        assert isinstance(body["macro_series"], list)
        assert len(body["macro_series"]) > 50
        first = body["macro_series"][0]
        assert set(first) == {"t", "value"}
        assert isinstance(first["value"], float)
        # SPY surfaces as the secondary overlay.
        assert {"ticker": "SPY", "source": "yf"} in body["additional_tickers"]
        # Polymarket leg is also returned.
        assert isinstance(body["polymarket_series"], list)
        assert {"t", "p"} == set(body["polymarket_series"][0])
        # Synthetic series are both monotone-rising → strong + correlation.
        assert body["correlation"] is not None
        assert body["correlation"] > 0.95

    def test_btc_market_routes_to_btc_usd(self, app_client: TestClient) -> None:
        """``btc_*`` slug should overlay BTC-USD via yfinance."""
        r = app_client.get("/terminal/macro-overlay/btc_ath_jun?days=120")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["macro_ticker"] == "BTC-USD"
        assert body["macro_source"] == "yf"
        assert body["additional_tickers"] == []
        # Stats keys must always be present.
        for key in ("correlation", "beta", "lag_days", "best_lag_corr"):
            assert key in body

    def test_unknown_slug_returns_null_macro(self, app_client: TestClient) -> None:
        """Unmapped slugs answer 200 with ``macro_ticker: null`` plus the
        explanatory message — the chart still has the polymarket leg."""
        r = app_client.get("/terminal/macro-overlay/some-unrelated-market")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["macro_ticker"] is None
        assert body["macro_source"] is None
        assert body["macro_series"] == []
        assert body["message"] == "no macro overlay"
        # The polymarket leg is still populated.
        assert len(body["polymarket_series"]) > 0


class TestCrossCorrelationLag:
    def test_best_lag_recovers_known_shift(self) -> None:
        """If macro_t = prob_{t+5} (macro is the future of prob, i.e. prob
        leads macro by 5 days), the best lag in our convention — where
        ``shifted = macro.shift(lag)`` — is +5: macro.shift(+5)_t =
        macro_{t-5} = prob_t."""
        rng = np.random.default_rng(42)
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
        # Smooth-ish AR(1) prob series so a correlation has signal.
        noise = rng.standard_normal(n)
        prob_vals = np.zeros(n)
        for i in range(1, n):
            prob_vals[i] = 0.9 * prob_vals[i - 1] + 0.1 * noise[i]
        prob = pd.Series(0.5 + 0.05 * prob_vals, index=idx, name="prob")

        shift = 5
        # macro_t = prob_{t+shift} → prob leads macro by `shift` days.
        macro = prob.shift(-shift).rename("macro")

        lag, corr = best_lag(prob, macro, max_lag=15)
        assert lag == shift
        assert corr is not None
        assert corr > 0.95

    def test_best_lag_returns_none_when_too_short(self) -> None:
        """Too few observations → ``(None, None)`` rather than crashing."""
        idx = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
        prob = pd.Series(np.linspace(0.2, 0.8, 20), index=idx)
        macro = pd.Series(np.linspace(4.0, 4.5, 20), index=idx)
        lag, corr = best_lag(prob, macro, max_lag=30)
        assert lag is None
        assert corr is None


class TestMappingTable:
    def test_mapping_covers_required_prefixes(self) -> None:
        """Sanity-check the curated mapping covers the headline categories."""
        prefixes = {p for p, _ in terminal_macro_overlay.PREFIX_MAP}
        for required in (
            "fed_cuts_",
            "no_fed_cuts_",
            "twelve_plus_fed_cuts",
            "inflation_above_",
            "k_cpi_",
            "us_recession_2026",
            "k_recession_2026",
            "btc_",
            "bitcoin_",
            "eth_",
            "ethereum_",
            "oil_",
            "crude_",
            "gold_",
            "silver_",
            "powell_out_",
            "taiwan_",
            "china_",
            "iran_",
            "ipo_",
        ):
            assert required in prefixes, required
        suffixes = {s for s, _ in terminal_macro_overlay.SUFFIX_MAP}
        assert "_acquired" in suffixes
