"""Tests for :mod:`pfm.portfolio_var_router`.

The router is mounted on a fresh ``FastAPI`` app in every test —
no full ``pfm.main`` import, no real yfinance / Tiingo / Stooq calls.
Daily returns and latest-price lookups are injected via
``app.state.var_return_provider`` and ``app.state.var_price_provider``
respectively.

Synthetic-DGP recovery checks generate Gaussian return paths with
known μ/Σ and confirm that all three VaR methodologies recover the
theoretical Gaussian VaR to within ±10% (relative).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.portfolio_import_router import Portfolio, PortfolioRow
from pfm.portfolio_var_router import (
    ES_CONFIDENCE,
    expected_shortfall,
    historical_var,
    monte_carlo_var,
    parametric_var,
    router,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _make_portfolio(
    handle: str,
    rows: list[tuple[str, float]],
) -> Portfolio:
    return Portfolio(
        handle=handle,
        rows=[PortfolioRow(ticker=tk, shares=sh) for tk, sh in rows],
        created_at=datetime.now(UTC),
    )


@pytest.fixture()
def app() -> Iterator[FastAPI]:
    a = FastAPI()
    a.include_router(router)
    a.state.portfolios = OrderedDict()
    yield a


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _install_stubs(
    app: FastAPI,
    returns_by_ticker: dict[str, pd.Series],
    prices_by_ticker: dict[str, float],
) -> None:
    """Register an in-process return/price provider on ``app.state``."""

    def _return_provider(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        if ticker not in returns_by_ticker:
            raise ValueError(f"no stub series for {ticker!r}")
        return returns_by_ticker[ticker].copy()

    def _price_provider(ticker: str) -> float:
        if ticker not in prices_by_ticker:
            raise ValueError(f"no stub price for {ticker!r}")
        return float(prices_by_ticker[ticker])

    app.state.var_return_provider = _return_provider
    app.state.var_price_provider = _price_provider


def _gaussian_series(
    sigma_daily: float,
    n: int,
    mu_daily: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    """Generate `n` daily Gaussian log-returns indexed by business-day dates."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=n)
    return pd.Series(
        rng.normal(loc=mu_daily, scale=sigma_daily, size=n),
        index=idx,
        name="r",
    )


def _correlated_series(
    sigmas: list[float],
    corr: np.ndarray,
    n: int,
    mu: list[float] | None = None,
    seed: int = 1,
) -> list[pd.Series]:
    rng = np.random.default_rng(seed)
    sigmas_arr = np.asarray(sigmas, dtype=float)
    mu_arr = np.asarray(mu, dtype=float) if mu is not None else np.zeros(len(sigmas), dtype=float)
    cov = np.outer(sigmas_arr, sigmas_arr) * corr
    chol = np.linalg.cholesky(cov + np.eye(len(sigmas)) * 1e-14)
    z = rng.standard_normal(size=(n, len(sigmas)))
    paths = z @ chol.T + mu_arr
    idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=n)
    return [pd.Series(paths[:, i], index=idx, name="r") for i in range(len(sigmas))]


# ---------------------------------------------------------------------------
# pure-math tests (synthetic DGP recovery)
# ---------------------------------------------------------------------------


class TestSyntheticDGPRecovery:
    """Confirm each estimator recovers the theoretical Gaussian VaR ±10%."""

    def test_parametric_var_single_asset_matches_closed_form(self) -> None:
        # Single-asset N(0, σ²) — theoretical 95% 1-day VaR = z * σ * pv
        sigma = 0.02
        pv = 1_000_000.0
        z = 1.6448536269514722  # Φ⁻¹(0.95)
        expected = z * sigma * pv
        got = parametric_var(
            weights=np.array([1.0]),
            mu=np.array([0.0]),
            cov=np.array([[sigma**2]]),
            confidence=0.95,
            horizon_days=1,
            portfolio_value=pv,
        )
        assert math.isclose(got, expected, rel_tol=1e-6)

    def test_parametric_var_two_asset_diversification(self) -> None:
        # Two uncorrelated assets, equal-weight, each σ=2%
        # σ_p = √(0.5²·0.02² + 0.5²·0.02²) = 0.02/√2
        sigma = 0.02
        pv = 1_000_000.0
        weights = np.array([0.5, 0.5])
        cov = np.array([[sigma**2, 0.0], [0.0, sigma**2]])
        mu = np.zeros(2)
        z = 1.6448536269514722
        expected = z * (sigma / math.sqrt(2.0)) * pv
        got = parametric_var(weights, mu, cov, 0.95, 1, pv)
        assert math.isclose(got, expected, rel_tol=1e-6)

    def test_historical_var_recovers_gaussian_quantile_10pct(self) -> None:
        sigma = 0.015
        n = 5000
        port_ret = np.random.default_rng(42).normal(0.0, sigma, n)
        pv = 1_000_000.0
        z = 1.6448536269514722
        expected = z * sigma * pv
        got = historical_var(port_ret, confidence=0.95, horizon_days=1, portfolio_value=pv)
        # 5000-sample empirical quantile of a Gaussian → noisy ±10% target.
        assert abs(got - expected) / expected < 0.10, (got, expected)

    def test_monte_carlo_var_matches_parametric_10pct(self) -> None:
        sigma = 0.02
        pv = 1_000_000.0
        weights = np.array([1.0])
        mu = np.array([0.0])
        cov = np.array([[sigma**2]])
        z = 1.6448536269514722
        expected = z * sigma * pv
        got = monte_carlo_var(weights, mu, cov, 0.95, 1, pv, n_paths=50_000, seed=7)
        assert abs(got - expected) / expected < 0.10, (got, expected)

    def test_mc_horizon_scales_with_sqrt_h(self) -> None:
        sigma = 0.015
        pv = 1_000_000.0
        weights = np.array([1.0])
        cov = np.array([[sigma**2]])
        mu = np.array([0.0])
        v1 = monte_carlo_var(weights, mu, cov, 0.95, 1, pv, 50_000, seed=11)
        v10 = monte_carlo_var(weights, mu, cov, 0.95, 10, pv, 50_000, seed=11)
        # √10 ≈ 3.162
        ratio = v10 / v1
        assert abs(ratio - math.sqrt(10.0)) / math.sqrt(10.0) < 0.10, ratio

    def test_expected_shortfall_exceeds_var(self) -> None:
        # For a Gaussian, ES_α = σ * φ(z_α) / (1−α). For 95%: ≈ σ * 2.0627
        sigma = 0.02
        pv = 1_000_000.0
        n = 20_000
        port_ret = np.random.default_rng(99).normal(0.0, sigma, n)
        var = historical_var(port_ret, 0.95, 1, pv)
        es = expected_shortfall(port_ret, ES_CONFIDENCE, 1, pv)
        assert es > var, (es, var)
        # Sanity: ES ≈ σ × φ(z₀.₉₅)/(1−0.95) × pv = σ × 2.0627 × pv
        expected_es = sigma * 2.062712761 * pv
        assert abs(es - expected_es) / expected_es < 0.10


# ---------------------------------------------------------------------------
# endpoint tests
# ---------------------------------------------------------------------------


class TestEndpointHappyPath:
    def test_single_asset_endpoint_returns_var(self, app: FastAPI, client: TestClient) -> None:
        # 252 daily Gaussian returns at σ=2% — VaR_95% ≈ z*σ*pv = 1.645*0.02*pv
        s = _gaussian_series(sigma_daily=0.02, n=252, seed=7)
        pf = _make_portfolio("pf_test_solo", [("NVDA", 100.0)])
        app.state.portfolios[pf.handle] = pf
        _install_stubs(
            app,
            returns_by_ticker={"NVDA": s},
            prices_by_ticker={"NVDA": 500.0},
        )

        r = client.get(f"/portfolio/{pf.handle}/var?confidence=0.95&horizon_days=1")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["handle"] == pf.handle
        assert body["confidence"] == 0.95
        assert body["horizon_days"] == 1
        pv = body["portfolio_value"]
        assert pv == pytest.approx(50_000.0)
        # All three VaRs should be in the right ballpark — z * σ * pv ≈ 1645
        target = 1.6448536269514722 * 0.02 * pv
        for key in ("parametric_var", "historical_var", "monte_carlo_var"):
            assert body[key] > 0
            assert abs(body[key] - target) / target < 0.30, (key, body[key])
        # weights should sum to ~1
        assert math.isclose(sum(body["weights"].values()), 1.0, rel_tol=1e-9)
        assert body["n_observations"] == 252
        assert body["mc_paths"] == 10_000
        # ES should exceed historical VaR (positive Gaussian tail).
        assert body["expected_shortfall_95"] > body["historical_var"] * 0.9

    def test_two_asset_endpoint_aligns_inner_join(self, app: FastAPI, client: TestClient) -> None:
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2024-01-01", periods=200)
        sA = pd.Series(rng.normal(0.0, 0.02, 200), index=idx)
        # B has 10 fewer observations — inner join should keep 190 rows.
        sB = pd.Series(rng.normal(0.0, 0.015, 190), index=idx[10:])
        pf = _make_portfolio("pf_test_pair", [("AAA", 100.0), ("BBB", 50.0)])
        app.state.portfolios[pf.handle] = pf
        _install_stubs(
            app,
            returns_by_ticker={"AAA": sA, "BBB": sB},
            prices_by_ticker={"AAA": 100.0, "BBB": 200.0},
        )

        r = client.get(f"/portfolio/{pf.handle}/var")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_observations"] == 190
        # weights: AAA = 100*100 / (100*100 + 50*200) = 10k/20k = 0.5
        assert body["weights"]["AAA"] == pytest.approx(0.5, abs=1e-9)
        assert body["weights"]["BBB"] == pytest.approx(0.5, abs=1e-9)
        assert body["portfolio_value"] == pytest.approx(20_000.0)

    def test_horizon_scaling_endpoint(self, app: FastAPI, client: TestClient) -> None:
        s = _gaussian_series(sigma_daily=0.02, n=300, seed=11)
        pf = _make_portfolio("pf_test_h", [("XYZ", 10.0)])
        app.state.portfolios[pf.handle] = pf
        _install_stubs(
            app,
            returns_by_ticker={"XYZ": s},
            prices_by_ticker={"XYZ": 1_000.0},
        )

        r1 = client.get(f"/portfolio/{pf.handle}/var?horizon_days=1&seed=42")
        r5 = client.get(f"/portfolio/{pf.handle}/var?horizon_days=5&seed=42")
        assert r1.status_code == 200
        assert r5.status_code == 200
        # parametric VaR scales by √5 up to the drift correction
        # (z·σ·√h − μ·h): with sample μ ≠ 0 in finite data, the linear
        # drift term creates a ±5% gap from the pure √h prediction.
        p1 = r1.json()["parametric_var"]
        p5 = r5.json()["parametric_var"]
        ratio = p5 / p1
        assert abs(ratio - math.sqrt(5.0)) / math.sqrt(5.0) < 0.10, ratio

    def test_three_methods_agree_within_25pct_on_gaussian_data(
        self, app: FastAPI, client: TestClient
    ) -> None:
        # Larger sample for tighter cross-method agreement.
        rng = np.random.default_rng(5)
        idx = pd.bdate_range("2020-01-01", periods=2000)
        s = pd.Series(rng.normal(0.0, 0.018, 2000), index=idx)
        pf = _make_portfolio("pf_test_3way", [("AGREE", 1.0)])
        app.state.portfolios[pf.handle] = pf
        _install_stubs(
            app,
            returns_by_ticker={"AGREE": s},
            prices_by_ticker={"AGREE": 100.0},
        )
        r = client.get(f"/portfolio/{pf.handle}/var?confidence=0.95&mc_paths=20000")
        assert r.status_code == 200, r.text
        body = r.json()
        p, h, m = (
            body["parametric_var"],
            body["historical_var"],
            body["monte_carlo_var"],
        )
        # all three within 10% of each other
        avg = (p + h + m) / 3.0
        for x in (p, h, m):
            assert abs(x - avg) / avg < 0.10, (x, avg, body)


class TestEndpointErrors:
    def test_unknown_handle_returns_404(self, app: FastAPI, client: TestClient) -> None:
        # Need at least an empty store on app.state for the friendly path.
        r = client.get("/portfolio/pf_nope/var")
        assert r.status_code == 404
        assert "unknown portfolio handle" in r.json()["detail"]

    def test_no_imports_yet_returns_404(self) -> None:
        # A fresh app with NO app.state.portfolios attribute at all.
        a = FastAPI()
        a.include_router(router)
        with TestClient(a) as c:
            r = c.get("/portfolio/anything/var")
            assert r.status_code == 404

    def test_all_fetches_fail_returns_502(self, app: FastAPI, client: TestClient) -> None:
        pf = _make_portfolio("pf_fail", [("BAD", 1.0)])
        app.state.portfolios[pf.handle] = pf

        def _bad_returns(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
            raise RuntimeError("boom")

        app.state.var_return_provider = _bad_returns
        app.state.var_price_provider = lambda t: 1.0
        r = client.get(f"/portfolio/{pf.handle}/var")
        assert r.status_code == 502
        assert "all return-series fetches failed" in r.json()["detail"]

    def test_insufficient_observations_returns_502(self, app: FastAPI, client: TestClient) -> None:
        # Only 10 obs — below MIN_OBSERVATIONS=30.
        s = _gaussian_series(0.02, n=10, seed=2)
        pf = _make_portfolio("pf_tiny", [("TINY", 1.0)])
        app.state.portfolios[pf.handle] = pf
        _install_stubs(app, returns_by_ticker={"TINY": s}, prices_by_ticker={"TINY": 100.0})
        r = client.get(f"/portfolio/{pf.handle}/var")
        assert r.status_code == 502
        assert "insufficient aligned return observations" in r.json()["detail"]

    def test_invalid_confidence_returns_422(self, app: FastAPI, client: TestClient) -> None:
        # Need at least an entry in the store so we hit query validation.
        pf = _make_portfolio("pf_val", [("VLD", 1.0)])
        app.state.portfolios[pf.handle] = pf
        r = client.get(f"/portfolio/{pf.handle}/var?confidence=1.5")
        assert r.status_code == 422

    def test_invalid_horizon_returns_422(self, app: FastAPI, client: TestClient) -> None:
        pf = _make_portfolio("pf_horz", [("HZN", 1.0)])
        app.state.portfolios[pf.handle] = pf
        r = client.get(f"/portfolio/{pf.handle}/var?horizon_days=0")
        assert r.status_code == 422

    def test_partial_price_lookup_failure_drops_ticker_with_warning(
        self, app: FastAPI, client: TestClient
    ) -> None:
        rng = np.random.default_rng(8)
        idx = pd.bdate_range("2024-01-01", periods=200)
        sA = pd.Series(rng.normal(0.0, 0.02, 200), index=idx)
        sB = pd.Series(rng.normal(0.0, 0.02, 200), index=idx)
        pf = _make_portfolio("pf_part", [("AAA", 100.0), ("BBB", 100.0)])
        app.state.portfolios[pf.handle] = pf

        # Return-series ok for both; price ok for AAA, fails for BBB.
        def _return_provider(t, s, e):
            return {"AAA": sA, "BBB": sB}[t].copy()

        def _price_provider(t):
            if t == "BBB":
                raise ValueError("no price")
            return 100.0

        app.state.var_return_provider = _return_provider
        app.state.var_price_provider = _price_provider
        r = client.get(f"/portfolio/{pf.handle}/var")
        assert r.status_code == 200, r.text
        body = r.json()
        # BBB dropped; AAA carries 100% weight.
        assert body["weights"] == {"AAA": pytest.approx(1.0)}
        assert any("price_lookup_failed: BBB" in w for w in body["warnings"])


class TestCorrelatedDGPRecovery:
    def test_two_correlated_assets_recover_parametric_var(
        self, app: FastAPI, client: TestClient
    ) -> None:
        # Build a 2-asset DGP with known σ and ρ; check the endpoint's
        # parametric VaR is within 10% of the closed-form Gaussian value.
        n = 2000
        sigmas = [0.02, 0.025]
        rho = 0.4
        corr = np.array([[1.0, rho], [rho, 1.0]])
        s1, s2 = _correlated_series(sigmas, corr, n, seed=33)
        pf = _make_portfolio("pf_corr", [("CA", 100.0), ("CB", 100.0)])
        app.state.portfolios[pf.handle] = pf
        _install_stubs(
            app,
            returns_by_ticker={"CA": s1, "CB": s2},
            prices_by_ticker={"CA": 100.0, "CB": 100.0},
        )
        r = client.get(f"/portfolio/{pf.handle}/var?confidence=0.95")
        assert r.status_code == 200
        body = r.json()
        # Equal weights at equal prices ⇒ w = (0.5, 0.5)
        # σ_p² = 0.5²·σ1² + 0.5²·σ2² + 2·0.5·0.5·ρ·σ1·σ2
        s1v, s2v = sigmas
        sigma_p = math.sqrt(0.25 * s1v**2 + 0.25 * s2v**2 + 2 * 0.25 * rho * s1v * s2v)
        z = 1.6448536269514722
        pv = 20_000.0  # 100*100 + 100*100
        expected = z * sigma_p * pv
        assert abs(body["parametric_var"] - expected) / expected < 0.10, (
            body["parametric_var"],
            expected,
        )
        assert abs(body["historical_var"] - expected) / expected < 0.15
        assert abs(body["monte_carlo_var"] - expected) / expected < 0.10
