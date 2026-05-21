"""Extra coverage for ``pfm.terminal_macro_overlay``.

Targets the error and edge-case paths not exercised in
``test_terminal_macro_overlay.py``:

  * polymarket upstream errors → 502
  * macro fetcher errors → 502
  * empty polymarket history is non-fatal (overlay stat null, series empty)
  * suffix mapping (`*_acquired`) routes to SPY
  * helper functions (_correlation, _beta, _resolve_overlay)
  * day-bound query validation
  * known prefix matrix (BTC/ETH/oil/gold/silver)
"""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_macro_overlay
from pfm.equity_factors import EquityFactorError
from pfm.sources.fred import FredDataError
from pfm.sources.polymarket import PolymarketError
from pfm.terminal_macro_overlay import (
    _beta,
    _correlation,
    _resolve_overlay,
    best_lag,
    get_polymarket_client,
    router,
)


class _FakePoly:
    pass


def _make_macro_series(days: int = 180, name: str = "X") -> pd.Series:
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    return pd.Series(np.linspace(4.0, 5.0, days), index=idx, name=name)


def _make_prob_history(days: int = 180) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    return pd.DataFrame({"price": np.linspace(0.2, 0.8, days)}, index=idx)


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch):
    def _build(
        *,
        prob_df: pd.DataFrame | None = None,
        prob_raises: BaseException | None = None,
        macro_raises: BaseException | None = None,
    ) -> TestClient:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_polymarket_client] = _FakePoly

        def _factor_history(_c, _s, start=None, end=None):
            if prob_raises is not None:
                raise prob_raises
            return prob_df if prob_df is not None else _make_prob_history()

        def _fred(series_id, start, end, **_):
            if macro_raises is not None:
                raise macro_raises
            return _make_macro_series(name=series_id)

        def _equity(ticker, start, end, **_):
            if macro_raises is not None:
                raise macro_raises
            return _make_macro_series(name=ticker)

        monkeypatch.setattr(terminal_macro_overlay, "fetch_factor_history", _factor_history)
        monkeypatch.setattr(terminal_macro_overlay, "fetch_fred_series", _fred)
        monkeypatch.setattr(terminal_macro_overlay, "fetch_equity_history", _equity)
        return TestClient(app)

    return _build


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_polymarket_error_yields_502(self, make_client) -> None:
        client = make_client(prob_raises=PolymarketError("market not found"))
        r = client.get("/terminal/macro-overlay/btc_ath_jun?days=120")
        assert r.status_code == 502
        assert "polymarket" in r.json()["detail"]

    def test_polymarket_http_error_yields_502(self, make_client) -> None:
        client = make_client(prob_raises=httpx.ConnectError("net down"))
        r = client.get("/terminal/macro-overlay/fed_cuts_2026?days=120")
        assert r.status_code == 502
        assert "polymarket http error" in r.json()["detail"]

    def test_macro_fetch_failure_yields_502(self, make_client) -> None:
        client = make_client(macro_raises=FredDataError("FRED 503"))
        r = client.get("/terminal/macro-overlay/fed_cuts_2026?days=120")
        assert r.status_code == 502
        assert "macro fetch failed" in r.json()["detail"]

    def test_equity_fetch_failure_yields_502(self, make_client) -> None:
        client = make_client(macro_raises=EquityFactorError("yfinance down"))
        r = client.get("/terminal/macro-overlay/btc_ath_jun?days=120")
        assert r.status_code == 502


# ---------------------------------------------------------------------------
# Empty / boundary inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_empty_polymarket_history_emits_empty_series_with_macro(self, make_client) -> None:
        empty = pd.DataFrame({"price": []}, index=pd.DatetimeIndex([], tz="UTC", name="date"))
        client = make_client(prob_df=empty)
        r = client.get("/terminal/macro-overlay/btc_ath_jun?days=120")
        # No prob data ⇒ no correlation/beta, but the macro series is still served.
        assert r.status_code == 200
        body = r.json()
        assert body["polymarket_series"] == []
        assert body["correlation"] is None
        assert body["beta"] is None
        assert body["macro_ticker"] == "BTC-USD"
        assert len(body["macro_series"]) > 0


# ---------------------------------------------------------------------------
# Mapping / overlay resolution
# ---------------------------------------------------------------------------


class TestMappingResolution:
    def test_suffix_acquired_routes_to_spy(self, make_client) -> None:
        client = make_client()
        r = client.get("/terminal/macro-overlay/some-company_acquired?days=120")
        assert r.status_code == 200
        assert r.json()["macro_ticker"] == "SPY"

    @pytest.mark.parametrize(
        "slug, expected_ticker",
        [
            ("btc_ath_jun", "BTC-USD"),
            ("bitcoin_above_100k", "BTC-USD"),
            ("eth_above_4k", "ETH-USD"),
            ("ethereum_flippening", "ETH-USD"),
            ("oil_below_50", "USO"),
            ("crude_above_100", "USO"),
            ("gold_above_3000", "GLD"),
            ("silver_above_50", "SLV"),
            ("twelve_plus_fed_cuts", "DGS10"),
            ("ipo_databricks", "SPY"),
        ],
    )
    def test_prefix_routing_matrix(self, make_client, slug: str, expected_ticker: str) -> None:
        client = make_client()
        r = client.get(f"/terminal/macro-overlay/{slug}?days=60")
        assert r.status_code == 200, r.text
        assert r.json()["macro_ticker"] == expected_ticker

    def test_resolve_overlay_pure_helper(self) -> None:
        # Known prefix.
        assert _resolve_overlay("btc_high")[0][0] == "BTC-USD"
        # Suffix path.
        assert _resolve_overlay("xyz_acquired")[0][0] == "SPY"
        # No match.
        assert _resolve_overlay("totally-unrelated") is None


# ---------------------------------------------------------------------------
# Pure stat helpers
# ---------------------------------------------------------------------------


class TestStatHelpers:
    def test_correlation_returns_none_on_short_input(self) -> None:
        idx = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        assert _correlation(s, s) is None

    def test_correlation_returns_none_on_zero_variance(self) -> None:
        idx = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
        prob = pd.Series(np.linspace(0.2, 0.8, 20), index=idx)
        flat = pd.Series(np.zeros(20), index=idx)
        assert _correlation(prob, flat) is None

    def test_correlation_recovers_perfect_pos(self) -> None:
        idx = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
        s = pd.Series(np.arange(30, dtype=float), index=idx)
        # Linear function ⇒ corr = 1.0.
        c = _correlation(s, 2 * s + 5)
        assert c == pytest.approx(1.0)

    def test_beta_zero_variance_returns_none(self) -> None:
        idx = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
        prob = pd.Series(np.linspace(0.2, 0.8, 10), index=idx)
        flat = pd.Series(np.ones(10), index=idx)
        assert _beta(prob, flat) is None

    def test_beta_recovers_known_slope(self) -> None:
        idx = pd.date_range("2025-01-01", periods=50, freq="D", tz="UTC")
        x = pd.Series(np.linspace(1.0, 10.0, 50), index=idx)
        y = 0.5 * x  # exact linear
        # β of y on x = cov(y,x)/var(x) = 0.5
        b = _beta(y, x)
        assert b == pytest.approx(0.5, rel=1e-9)

    def test_best_lag_returns_none_when_max_lag_exceeds_window(self) -> None:
        idx = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
        s = pd.Series(np.arange(10, dtype=float), index=idx)
        lag, corr = best_lag(s, s, max_lag=20)
        assert lag is None
        assert corr is None


# ---------------------------------------------------------------------------
# Endpoint validation
# ---------------------------------------------------------------------------


class TestQueryValidation:
    def test_days_lower_bound(self, make_client) -> None:
        client = make_client()
        r = client.get("/terminal/macro-overlay/btc_x?days=5")
        assert r.status_code == 422

    def test_days_upper_bound(self, make_client) -> None:
        client = make_client()
        r = client.get("/terminal/macro-overlay/btc_x?days=999999")
        assert r.status_code == 422

    def test_default_days_180(self, make_client) -> None:
        client = make_client()
        r = client.get("/terminal/macro-overlay/btc_x")
        assert r.status_code == 200
