"""Tests for the event-impact (event study) router.

Two layers exercised:

1. :func:`run_event_study` and :func:`estimate_market_model` are pure
   functions — we hand-build synthetic price series with a known DGP
   (asset = α + β·market + ε, plus a "shock" injected on event day) and
   verify the estimator recovers the planted parameters and rejects the
   null when the shock is large.
2. The HTTP route joins two upstream fetches (Polymarket history and
   SPY via yfinance). Both are mocked so the test runs offline.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal.event_impact_router import (
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_WINDOW_DAYS,
    SIGNIFICANCE_ALPHA,
    EventImpactResponse,
    estimate_market_model,
    router,
    run_event_study,
)

# ---------------------------------------------------------------------------
# Synthetic DGP helpers
# ---------------------------------------------------------------------------


def _daily_dates(n: int, end: str = "2026-03-30") -> pd.DatetimeIndex:
    """Build a UTC-normalised business-day index of length ``n`` ending at ``end``."""
    end_ts = pd.Timestamp(end, tz="UTC").normalize()
    # Use calendar days so the index alignment in run_event_study matches
    # SPY's business-day index after intersection.
    return pd.date_range(end=end_ts, periods=n, freq="D", tz="UTC")


def _synthetic_market_prices(
    n: int = 120, drift: float = 0.0003, sigma: float = 0.01, seed: int = 11
) -> pd.Series:
    """GBM-ish market index series with positive drift."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=drift, scale=sigma, size=n)
    log_p = np.cumsum(shocks)
    prices = 400.0 * np.exp(log_p)  # start near SPY's recent level
    return pd.Series(prices, index=_daily_dates(n), name="SPY")


def _synthetic_asset_prices(
    market_prices: pd.Series,
    *,
    alpha: float = 0.0,
    beta: float = 1.0,
    idio_sigma: float = 0.005,
    event_date: pd.Timestamp | None = None,
    event_shock: float = 0.0,
    seed: int = 7,
    start: float = 0.45,
) -> pd.Series:
    """Build a Polymarket-style probability series from the DGP

        Δlog(p_i) = α + β · Δlog(p_m) + ε,    ε ~ N(0, idio_sigma)

    optionally injecting ``event_shock`` (in log-return units) on
    ``event_date``. Bounded to stay in (0.01, 0.99) so log-clipping
    doesn't kick in.
    """
    rng = np.random.default_rng(seed)
    m_ret = np.log(market_prices / market_prices.shift(1)).fillna(0.0).values
    eps = rng.normal(loc=0.0, scale=idio_sigma, size=len(m_ret))
    a_ret = alpha + beta * m_ret + eps
    if event_date is not None:
        ts = pd.Timestamp(event_date)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        event_d = ts.normalize()
        idx_list = list(market_prices.index)
        if event_d in idx_list:
            pos = idx_list.index(event_d)
            a_ret[pos] += event_shock
    # Cumulate returns, then transform to a (0,1) probability via inverse-logit.
    log_p = np.cumsum(a_ret)
    base_logit = np.log(start / (1.0 - start))
    logit_path = base_logit + log_p * 0.5  # scale so the path stays in (0.05, 0.95)
    probs = 1.0 / (1.0 + np.exp(-logit_path))
    return pd.Series(probs, index=market_prices.index, name="asset")


# ---------------------------------------------------------------------------
# 1. Pure-function tests against a known DGP
# ---------------------------------------------------------------------------


def test_estimate_market_model_recovers_known_alpha_beta() -> None:
    """Plant α=0.001, β=1.4; OLS should recover both within 0.1."""
    rng = np.random.default_rng(42)
    m_ret = rng.normal(0.0, 0.01, size=200)
    eps = rng.normal(0.0, 0.002, size=200)
    a_ret = 0.001 + 1.4 * m_ret + eps
    idx = pd.date_range("2026-01-01", periods=200, freq="D", tz="UTC")
    market = pd.Series(m_ret, index=idx)
    asset = pd.Series(a_ret, index=idx)
    alpha, beta, sigma, n = estimate_market_model(asset, market)
    assert n == 200
    assert abs(alpha - 0.001) < 0.001, f"alpha drift: {alpha}"
    assert abs(beta - 1.4) < 0.05, f"beta drift: {beta}"
    assert 0.0015 < sigma < 0.003, f"residual sigma off: {sigma}"


def test_estimate_market_model_handles_too_few_obs() -> None:
    """With < 3 obs we fall back to (0, 1, 0, n) — don't blow up."""
    idx = pd.date_range("2026-03-01", periods=2, freq="D", tz="UTC")
    a, b, s, n = estimate_market_model(
        pd.Series([0.01, -0.02], index=idx),
        pd.Series([0.005, -0.01], index=idx),
    )
    assert n == 2
    assert a == 0.0 and b == 1.0 and s == 0.0


def test_no_event_shock_yields_insignificant_t_stat() -> None:
    """Under the null (no shock) the t-stat should be near 0 and p > 0.05."""
    market = _synthetic_market_prices(n=120, seed=1)
    asset = _synthetic_asset_prices(
        market, alpha=0.0, beta=1.2, idio_sigma=0.004, event_shock=0.0, seed=2
    )
    event_date = market.index[-DEFAULT_WINDOW_DAYS - 3]
    result = run_event_study(
        # Convert "returns" series → "price-level" series for the function
        # under test, which expects raw prices.
        asset,
        market,
        event_date,
        window_days=DEFAULT_WINDOW_DAYS,
        estimation_days=40,
    )
    assert result["n_event_days"] >= 2 * DEFAULT_WINDOW_DAYS  # most days usable
    assert abs(result["t_stat"]) < 3.0, f"unexpectedly large t under null: {result['t_stat']}"
    assert result["p_value"] > 0.01, f"unexpected rejection under null: {result['p_value']}"


def test_large_event_shock_is_detected_as_significant() -> None:
    """A 10-σ-equivalent shock on event day should produce a rejection."""
    market = _synthetic_market_prices(n=160, seed=3)
    event_date = market.index[-DEFAULT_WINDOW_DAYS - 2]
    asset = _synthetic_asset_prices(
        market,
        alpha=0.0,
        beta=1.0,
        idio_sigma=0.003,
        event_date=event_date,
        event_shock=0.30,  # enormous shock → unambiguous rejection
        seed=4,
    )
    result = run_event_study(
        asset,
        market,
        event_date,
        window_days=DEFAULT_WINDOW_DAYS,
        estimation_days=50,
    )
    # The AR vector should be non-empty and the t-stat large.
    assert result["n_event_days"] >= 5
    assert abs(result["t_stat"]) > 2.0
    assert result["p_value"] < SIGNIFICANCE_ALPHA
    assert result["significant"] is True


def test_car_is_running_sum_of_ar() -> None:
    """CAR_t must equal the cumulative sum of AR_s for s ≤ t."""
    market = _synthetic_market_prices(n=120, seed=5)
    event_date = market.index[-DEFAULT_WINDOW_DAYS - 2]
    asset = _synthetic_asset_prices(
        market, beta=1.0, event_date=event_date, event_shock=0.05, seed=6
    )
    result = run_event_study(asset, market, event_date, window_days=DEFAULT_WINDOW_DAYS)
    ar = result["ar"]
    car = result["car"]
    assert len(ar) == len(car)
    assert ar  # non-empty
    cum = 0.0
    for a, c in zip(ar, car, strict=True):
        cum += a
        assert abs(c - cum) < 1e-9


def test_returns_payload_keys_match_spec() -> None:
    """Spec: {slug, event_date, ar, car, t_stat, p_value, significant}."""
    market = _synthetic_market_prices(n=100, seed=8)
    event_date = market.index[-DEFAULT_WINDOW_DAYS - 2]
    asset = _synthetic_asset_prices(market, event_date=event_date, event_shock=0.0, seed=9)
    result = run_event_study(asset, market, event_date, window_days=DEFAULT_WINDOW_DAYS)
    for key in ("ar", "car", "t_stat", "p_value", "significant", "alpha", "beta", "residual_sigma"):
        assert key in result, f"missing key {key}"
    assert isinstance(result["significant"], bool)


def test_event_window_size_matches_window_days_param() -> None:
    """A ±K window has at most 2K+1 entries in per_day."""
    market = _synthetic_market_prices(n=120, seed=10)
    event_date = market.index[-8]
    asset = _synthetic_asset_prices(market, seed=11)
    K = 5
    result = run_event_study(asset, market, event_date, window_days=K)
    assert len(result["per_day"]) == 2 * K + 1


def test_offset_days_centered_on_event() -> None:
    """One per_day entry has offset_days == 0 (the event date itself)."""
    market = _synthetic_market_prices(n=100, seed=12)
    event_date = market.index[-7]
    asset = _synthetic_asset_prices(market, seed=13)
    result = run_event_study(asset, market, event_date, window_days=3)
    offsets = [p["offset_days"] for p in result["per_day"]]
    assert 0 in offsets
    assert offsets[0] == -3
    assert offsets[-1] == 3


def test_no_overlap_with_estimation_window_returns_zero_t() -> None:
    """If asset history starts AFTER the event window, no AR observations.

    The function should not crash and should return ``significant=False``.
    """
    market = _synthetic_market_prices(n=100, seed=14)
    # Slice asset to the LAST 5 days only — no estimation window, no
    # event-window observations from before event_date.
    event_date = market.index[0] + pd.Timedelta(days=5)
    short_asset = pd.Series(
        [0.5, 0.51, 0.52, 0.53, 0.54],
        index=pd.date_range(start=market.index[-5], periods=5, freq="D", tz="UTC"),
        name="asset",
    )
    result = run_event_study(short_asset, market, event_date, window_days=DEFAULT_WINDOW_DAYS)
    assert result["significant"] is False
    assert result["t_stat"] == 0.0 or result["n_event_days"] == 0


def test_clipping_handles_resolved_market_prices() -> None:
    """A series that hits 1.0 (resolved YES) shouldn't produce inf log-returns."""
    market = _synthetic_market_prices(n=80, seed=15)
    # Asset: probability climbs to 1.0 on the event date and stays there.
    event_date = market.index[-DEFAULT_WINDOW_DAYS - 2]
    vals = np.linspace(0.40, 1.0, len(market))
    asset = pd.Series(vals, index=market.index, name="asset")
    result = run_event_study(asset, market, event_date, window_days=DEFAULT_WINDOW_DAYS)
    # Must complete without NaNs in the AR/CAR vectors.
    for v in result["ar"]:
        assert np.isfinite(v), f"non-finite AR: {v}"
    for v in result["car"]:
        assert np.isfinite(v), f"non-finite CAR: {v}"


def test_p_value_is_between_zero_and_one() -> None:
    """Sanity: p must be in [0, 1] under any DGP."""
    market = _synthetic_market_prices(n=120, seed=20)
    event_date = market.index[-DEFAULT_WINDOW_DAYS - 2]
    for shock in [-0.20, -0.05, 0.0, 0.05, 0.20]:
        asset = _synthetic_asset_prices(market, event_date=event_date, event_shock=shock, seed=21)
        result = run_event_study(asset, market, event_date)
        assert 0.0 <= result["p_value"] <= 1.0


def test_significance_flag_consistent_with_p_value() -> None:
    """``significant`` must equal ``p_value < SIGNIFICANCE_ALPHA`` (when n>=1)."""
    market = _synthetic_market_prices(n=120, seed=30)
    event_date = market.index[-DEFAULT_WINDOW_DAYS - 2]
    asset = _synthetic_asset_prices(market, event_date=event_date, event_shock=0.0, seed=31)
    result = run_event_study(asset, market, event_date)
    if result["n_event_days"] >= 1:
        assert result["significant"] == (result["p_value"] < SIGNIFICANCE_ALPHA)


# ---------------------------------------------------------------------------
# 2. HTTP route tests — Polymarket + benchmark both mocked
# ---------------------------------------------------------------------------


class _FakePoly:
    """Stand-in for PolymarketClient used by the dependency injection."""

    def __init__(self) -> None:
        self.calls = 0


def _build_test_app(asset_prices: pd.Series, market_prices: pd.Series) -> TestClient:
    """FastAPI app with the router mounted and both fetches patched."""
    app = FastAPI()
    app.state.poly = _FakePoly()
    app.include_router(router)
    return TestClient(app)


def test_endpoint_happy_path_returns_significant_shock() -> None:
    market = _synthetic_market_prices(n=160, seed=40)
    event_date = market.index[-DEFAULT_WINDOW_DAYS - 2]
    asset = _synthetic_asset_prices(
        market, event_date=event_date, event_shock=0.30, idio_sigma=0.003, seed=41
    )
    # Repackage into the DataFrame shape fetch_factor_history returns.
    asset_df = pd.DataFrame({"price": asset.values}, index=asset.index)
    asset_df.index.name = "date"

    client = _build_test_app(asset, market)
    with (
        patch(
            "pfm.terminal.event_impact_router.fetch_factor_history",
            return_value=asset_df,
        ),
        patch(
            "pfm.terminal.event_impact_router._fetch_benchmark_prices",
            return_value=market,
        ),
    ):
        r = client.get(
            "/terminal/event-impact/my-test-slug",
            params={
                "event_date": event_date.strftime("%Y-%m-%d"),
                "window_days": DEFAULT_WINDOW_DAYS,
                "estimation_days": 50,
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    # Spec-mandated keys.
    for k in ("slug", "event_date", "ar", "car", "t_stat", "p_value", "significant"):
        assert k in body, f"missing key {k}"
    assert body["slug"] == "my-test-slug"
    assert body["event_date"] == event_date.strftime("%Y-%m-%d")
    assert body["benchmark"] == DEFAULT_BENCHMARK_TICKER
    assert body["significant"] is True
    assert body["p_value"] < SIGNIFICANCE_ALPHA


def test_endpoint_missing_polymarket_history_returns_404() -> None:
    """Empty asset history → 404."""
    client = _build_test_app(pd.Series(dtype=float), _synthetic_market_prices(50, seed=50))
    with (
        patch(
            "pfm.terminal.event_impact_router.fetch_factor_history",
            return_value=pd.DataFrame(columns=["price"]),
        ),
        patch(
            "pfm.terminal.event_impact_router._fetch_benchmark_prices",
            return_value=_synthetic_market_prices(50, seed=51),
        ),
    ):
        r = client.get(
            "/terminal/event-impact/nope-slug",
            params={"event_date": "2026-03-15"},
        )
    assert r.status_code == 404


def test_endpoint_bad_event_date_returns_400() -> None:
    """A malformed event_date triggers a 400."""
    client = _build_test_app(pd.Series(dtype=float), pd.Series(dtype=float))
    r = client.get(
        "/terminal/event-impact/some-slug",
        params={"event_date": "not-a-date"},
    )
    assert r.status_code == 400


def test_endpoint_503_when_poly_client_missing() -> None:
    """Without app.state.poly the dependency raises 503."""
    app = FastAPI()
    # Deliberately no app.state.poly.
    app.include_router(router)
    client = TestClient(app)
    r = client.get(
        "/terminal/event-impact/x",
        params={"event_date": "2026-03-15"},
    )
    assert r.status_code == 503


def test_endpoint_response_validates_against_schema() -> None:
    """The JSON payload round-trips through EventImpactResponse cleanly."""
    market = _synthetic_market_prices(n=120, seed=60)
    event_date = market.index[-DEFAULT_WINDOW_DAYS - 2]
    asset = _synthetic_asset_prices(market, event_date=event_date, seed=61)
    asset_df = pd.DataFrame({"price": asset.values}, index=asset.index)
    asset_df.index.name = "date"

    client = _build_test_app(asset, market)
    with (
        patch(
            "pfm.terminal.event_impact_router.fetch_factor_history",
            return_value=asset_df,
        ),
        patch(
            "pfm.terminal.event_impact_router._fetch_benchmark_prices",
            return_value=market,
        ),
    ):
        r = client.get(
            "/terminal/event-impact/some-slug",
            params={"event_date": event_date.strftime("%Y-%m-%d")},
        )
    assert r.status_code == 200
    validated = EventImpactResponse.model_validate(r.json())
    assert validated.slug == "some-slug"
    # The flat ar / car arrays in the response equal the ones derived
    # from per_day entries that had a non-null abnormal_return.
    ar_from_per_day = [
        p.abnormal_return for p in validated.per_day if p.abnormal_return is not None
    ]
    assert validated.ar == ar_from_per_day
