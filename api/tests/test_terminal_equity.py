"""Tests for ``pfm.terminal_equity`` — the /terminal/equity/{slug} overlay.

Both external dependencies (yfinance and Polymarket) are patched out
so the suite never touches the network. The router is mounted on a
fresh :class:`FastAPI` app to avoid spinning up the full ``pfm.main``
lifespan (Redis, factors.yml, …) just for an isolated unit test.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_equity
from pfm.terminal_equity import SLUG_TO_TICKER, get_polymarket_client, router

# --- fakes ------------------------------------------------------------------


class _FakePoly:
    """Stand-in for :class:`pfm.sources.polymarket.PolymarketClient`.

    The router only ever passes the client through to
    :func:`fetch_factor_history`, which we monkeypatch wholesale, so the
    instance only needs to be a non-None sentinel.
    """


def _make_equity_series(days: int = 180, start_price: float = 150.0) -> pd.Series:
    """A monotonically rising synthetic equity series indexed by UTC date."""
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    values = start_price + np.linspace(0.0, 30.0, days)
    return pd.Series(values, index=idx, name="EQ")


def _make_prob_history(days: int = 180, low: float = 0.20, high: float = 0.80) -> pd.DataFrame:
    """A factor-history-style DataFrame matching ``fetch_factor_history``'s contract."""
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    idx.name = "date"
    # Linear sweep low → high so it is strongly correlated with the equity
    # series (also monotone) and cointegration is well-posed.
    prices = np.linspace(low, high, days)
    return pd.DataFrame({"price": prices}, index=idx)


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Mount the router on a bare FastAPI app with all IO patched out."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_polymarket_client] = _FakePoly

    # Default fakes — individual tests can re-patch as needed.
    monkeypatch.setattr(
        terminal_equity,
        "fetch_equity_history",
        lambda ticker, start, end, **_: _make_equity_series(),
    )
    monkeypatch.setattr(
        terminal_equity,
        "fetch_factor_history",
        lambda _client, _slug, start=None, end=None: _make_prob_history(),
    )

    with TestClient(app) as client:
        yield client


# --- tests ------------------------------------------------------------------


class TestTerminalEquity:
    def test_known_slug_returns_full_payload(self, app_client: TestClient) -> None:
        r = app_client.get("/terminal/equity/apple_largest_jun?days=180")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ticker"] == "AAPL"
        assert isinstance(body["equity_series"], list)
        assert len(body["equity_series"]) > 50
        first = body["equity_series"][0]
        assert set(first) == {"t", "close"}
        assert isinstance(first["close"], float)
        # Diagnostics are present (may be None if cointegration fails on
        # the synthetic data, but the keys must always be there).
        for key in ("correlation_with_prob", "beta", "intercept", "half_life"):
            assert key in body
        # Synthetic series are monotone-rising → strong positive correlation.
        assert body["correlation_with_prob"] is not None
        assert body["correlation_with_prob"] > 0.95

    def test_prefix_match_resolves_ticker(self, app_client: TestClient) -> None:
        """``nvda_*`` prefix should map to NVDA even when the literal slug
        isn't in the table."""
        r = app_client.get("/terminal/equity/nvda_largest_dec?days=120")
        assert r.status_code == 200, r.text
        assert r.json()["ticker"] == "NVDA"

    def test_known_slug_without_public_ticker_returns_null(
        self,
        app_client: TestClient,
    ) -> None:
        r = app_client.get("/terminal/equity/spacex_ipo")
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] is None
        assert body["message"] == "no equity counterpart"

    def test_unknown_slug_returns_404(self, app_client: TestClient) -> None:
        """Unknown slug → 200 with ``ticker: null`` (post-2026-05 UX revision).

        The endpoint used to return a 404 for unrecognised slugs but every
        browser logged it to console for sports / non-equity markets that
        fire this endpoint defensively. Per ``src/pfm/terminal/equity.py``
        (~line 217) the endpoint now returns 200 with an empty payload —
        same UX as the "known slug, no public counterpart" case — and the
        frontend hides the card silently when ``ticker is None``.

        Test renamed conceptually (kept the function name for git-blame
        continuity) and now asserts the documented quiet-200 contract.
        """
        r = app_client.get("/terminal/equity/some-completely-unrelated-market")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ticker"] is None
        assert body["slug"] == "some-completely-unrelated-market"
        # Message must be non-empty so the UI has something to log/render.
        assert body.get("message")

    def test_polymarket_empty_history_returns_502(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty = pd.DataFrame(columns=["price"])
        empty.index = pd.DatetimeIndex([], tz="UTC", name="date")
        monkeypatch.setattr(
            terminal_equity,
            "fetch_factor_history",
            lambda _c, _s, start=None, end=None: empty,
        )
        r = app_client.get("/terminal/equity/apple_largest_jun")
        assert r.status_code == 502
        assert "no history" in r.json()["detail"]

    def test_mapping_table_is_non_trivial(self) -> None:
        """Sanity-check the curated mapping covers the headline tickers."""
        for required in (
            "apple_largest_jun",
            "msft_largest",
            "gitlab_acquired",
            "jpmorgan_chase_fail",
            "bp_acquired",
        ):
            assert required in SLUG_TO_TICKER, required

    def test_response_schema_for_matched_slug(self, app_client: TestClient) -> None:
        """The matched-ticker payload exposes exactly the documented keys."""
        body = app_client.get("/terminal/equity/apple_largest_jun?days=120").json()
        expected = {
            "ticker",
            "equity_series",
            "correlation_with_prob",
            "beta",
            "intercept",
            "half_life",
        }
        assert set(body.keys()) == expected
        # Each equity_series point is exactly {t, close} with float close.
        for pt in body["equity_series"][:5]:
            assert set(pt.keys()) == {"t", "close"}
            assert isinstance(pt["close"], float)

    def test_days_query_validator_bounds(self, app_client: TestClient) -> None:
        """`days` is constrained to [10, 3650]."""
        # below minimum
        r1 = app_client.get("/terminal/equity/apple_largest_jun?days=5")
        assert r1.status_code == 422
        # above maximum
        r2 = app_client.get("/terminal/equity/apple_largest_jun?days=10000")
        assert r2.status_code == 422
        # boundary minimum is accepted
        r3 = app_client.get("/terminal/equity/apple_largest_jun?days=10")
        assert r3.status_code == 200
