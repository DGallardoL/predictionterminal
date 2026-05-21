"""Tests for ``pfm.terminal_correlations`` — /terminal/correlations/{slug}.

External dependencies (FRED, yfinance, Polymarket) are patched out so
the suite is fully offline. The router is mounted on a fresh
:class:`FastAPI` app to avoid the full ``pfm.main`` lifespan.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_correlations
from pfm.sources.polymarket import PolymarketError
from pfm.terminal_correlations import (
    BENCHMARKS,
    best_lag_corr,
    clear_cache,
    get_polymarket_client,
    router,
)

# --- fakes ------------------------------------------------------------------


class _FakePoly:
    """Sentinel — the router only forwards this through to
    :func:`fetch_factor_history`, which is monkeypatched wholesale."""


def _ar1_walk(n: int, *, phi: float, sigma: float, seed: int) -> np.ndarray:
    """Deterministic AR(1) innovation walk used to build synthetic series."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * sigma
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _make_prob_history(
    days: int = 120,
    base: float = 0.50,
    seed: int = 1,
) -> pd.DataFrame:
    """A factor-history DataFrame matching ``fetch_factor_history``'s contract."""
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    idx.name = "date"
    walk = _ar1_walk(days, phi=0.85, sigma=0.04, seed=seed)
    prices = (base + walk).clip(0.05, 0.95)
    return pd.DataFrame({"price": prices}, index=idx)


def _make_correlated_price(
    prob_df: pd.DataFrame,
    *,
    correlation: float,
    base: float = 100.0,
    seed: int = 7,
    name: str = "ASSET",
) -> pd.Series:
    """Build a price series whose log-returns are ~``correlation`` with the
    market's logit-prob innovations.

    Construction: take Δlogit(p) as the driver, mix with independent
    noise at the requested ratio, then accumulate to a price level.
    """
    rng = np.random.default_rng(seed)
    p = prob_df["price"].astype(float).clip(0.01, 0.99)
    logit_p = np.log(p / (1 - p))
    driver = logit_p.diff().fillna(0.0).to_numpy()
    # Standardise the driver so we can mix linearly.
    driver_std = driver.std() or 1.0
    driver_n = driver / driver_std

    rho = float(correlation)
    independent = rng.standard_normal(len(driver_n))
    mixed = rho * driver_n + np.sqrt(max(1.0 - rho * rho, 0.0)) * independent
    # Scale to a typical daily equity log-return magnitude.
    log_returns = 0.01 * mixed
    log_price = np.cumsum(log_returns)
    prices = base * np.exp(log_price)
    return pd.Series(prices, index=prob_df.index, name=name)


def _make_uncorrelated_price(
    days: int,
    *,
    seed: int,
    base: float = 100.0,
    name: str = "ASSET",
) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    log_returns = 0.01 * rng.standard_normal(days)
    prices = base * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=idx, name=name)


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _drop_cache() -> Iterator[None]:
    """Reset the in-memory cache between tests."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def app_factory(monkeypatch: pytest.MonkeyPatch):
    """Mount the router on a bare FastAPI app and return a builder.

    The builder accepts a ``benchmark_factory`` callable that produces a
    benchmark price series given ``(symbol, prob_df)`` so individual
    tests can dial up / down each asset's correlation with the market.
    """

    def _build(
        prob_df: pd.DataFrame | None = None,
        *,
        benchmark_factory=None,
        slug_raises: BaseException | None = None,
    ) -> TestClient:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_polymarket_client] = _FakePoly

        captured = prob_df if prob_df is not None else _make_prob_history()

        def _fake_factor_history(_client, _slug, start=None, end=None):
            if slug_raises is not None:
                raise slug_raises
            return captured

        monkeypatch.setattr(terminal_correlations, "fetch_factor_history", _fake_factor_history)

        # Default benchmark factory: uncorrelated noise per ticker.
        def _default_factory(symbol: str, prob: pd.DataFrame) -> pd.Series:
            base = _make_uncorrelated_price(
                len(prob),
                seed=hash(symbol) % (2**32),
                name=symbol,
            )
            return base.reindex(prob.index).ffill().bfill()

        factory = benchmark_factory or _default_factory

        def _fake_equity_history(ticker, start, end, **_):
            s = factory(ticker, captured)
            s.name = ticker
            return s

        def _fake_fred_series(series_id, start, end, **_):
            s = factory(series_id, captured)
            s.name = series_id
            return s

        monkeypatch.setattr(terminal_correlations, "fetch_equity_history", _fake_equity_history)
        monkeypatch.setattr(terminal_correlations, "fetch_fred_series", _fake_fred_series)

        return TestClient(app)

    return _build


# --- tests ------------------------------------------------------------------


class TestCorrelationEndpoint:
    def test_high_correlation_pair_recovered(self, app_factory) -> None:
        """A benchmark engineered to be highly correlated (ρ≈0.85) with the
        market's logit innovations should surface in the response with a
        large positive ``corr`` and a tiny p-value."""
        prob = _make_prob_history(days=180, seed=11)

        def factory(symbol: str, prob_df: pd.DataFrame) -> pd.Series:
            if symbol == "BTC-USD":
                return _make_correlated_price(
                    prob_df,
                    correlation=0.85,
                    seed=21,
                    name=symbol,
                )
            # Other benchmarks remain uncorrelated noise.
            return _make_uncorrelated_price(
                len(prob_df),
                seed=hash(symbol) % (2**32),
                name=symbol,
            ).reindex(prob_df.index)

        client = app_factory(prob_df=prob, benchmark_factory=factory)
        r = client.get("/terminal/correlations/btc_ath_jun?days=120")
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["slug"] == "btc_ath_jun"
        assert body["lookback_days"] == 120
        # All 7 benchmarks should be present in the response.
        assert set(body["correlations"]) == {n for n, _, _ in BENCHMARKS}

        btc = body["correlations"]["BTC-USD"]
        assert btc["corr"] is not None
        assert btc["corr"] > 0.6, f"expected strong + corr, got {btc['corr']}"
        assert btc["p_value"] is not None
        assert btc["p_value"] < 0.01
        assert btc["lag_days"] in range(-7, 8)

        # BTC should be the strongest (ranked first).
        assert body["strongest"][0]["asset"] == "BTC-USD"
        # Interpretation should mention BTC by name.
        assert "BTC-USD" in body["interpretation"]

    def test_low_correlation_pair_reported(self, app_factory) -> None:
        """When all benchmarks are independent noise, all corrs should be
        small in magnitude and p-values mostly above 0.05."""
        prob = _make_prob_history(days=200, seed=42)
        client = app_factory(prob_df=prob)

        r = client.get("/terminal/correlations/some_indie_market?days=120")
        assert r.status_code == 200, r.text
        body = r.json()

        for asset, info in body["correlations"].items():
            assert info["corr"] is not None, asset
            # With ~120 obs of independent noise we expect |corr| < ~0.4
            # at the strongest lag in [-7, 7].
            assert abs(info["corr"]) < 0.5, f"{asset}: {info['corr']}"
        # And the strongest of these noise-corrs should still be < 0.5.
        if body["strongest"]:
            assert abs(body["strongest"][0]["corr"]) < 0.5

    def test_unknown_slug_returns_404(self, app_factory) -> None:
        """A :class:`PolymarketError` from the data layer should surface as
        an HTTP 404 with a helpful detail rather than a 500."""
        client = app_factory(slug_raises=PolymarketError("no market found for slug='nope'"))
        r = client.get("/terminal/correlations/nope?days=90")
        assert r.status_code == 404, r.text
        assert "nope" in r.json()["detail"]


class TestLagDetection:
    def test_best_lag_recovers_known_shift(self) -> None:
        """If asset_innov_t = prob_innov_{t+3} (prob leads asset by 3 days),
        ``shifted = asset.shift(lag)`` is maximised at ``lag = +3``:
        asset.shift(+3)_t = asset_{t-3} = prob_t."""
        rng = np.random.default_rng(2024)
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
        prob_innov = pd.Series(rng.standard_normal(n) * 0.05, index=idx, name="p")

        shift = 3
        # asset_t = prob_{t+shift}  →  prob leads asset by `shift` days.
        asset_innov = prob_innov.shift(-shift).rename("a")

        lag, corr, n_obs = best_lag_corr(prob_innov, asset_innov, max_lag=7)
        assert lag == shift
        assert corr is not None and corr > 0.95
        assert n_obs is not None and n_obs > 100
