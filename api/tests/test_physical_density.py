"""Tests for the physical (P-measure) terminal density estimator.

All tests are no-network: yfinance is mocked. Synthetic GARCH-like returns
are used to verify σ recovery and density properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfm.vol.physical_density import (
    PhysicalDensityResult,
    estimate_physical_density,
    fetch_index_history,
    physical_density_from_returns,
)

# NumPy 2.0 trapezoid compat for the tests too.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _simulate_garch_log_prices(
    n: int = 600,
    *,
    omega: float = 1e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    mu: float = 0.0003,
    seed: int = 7,
    start_price: float = 5000.0,
) -> pd.Series:
    """Simulate a GARCH(1,1) log-return path and return the log-price series."""
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    eps = np.empty(n)
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-6)
    eps[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    log_rets = mu + eps
    log_prices = np.log(start_price) + np.cumsum(log_rets)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(log_prices, index=idx, name="log_close")


# ---------------------------------------------------------------------------
# (a) synthetic GARCH recovers a sigma in the right ballpark
# ---------------------------------------------------------------------------


def test_garch_recovers_ballpark_sigma() -> None:
    # Unconditional daily σ ≈ sqrt(omega / (1 - alpha - beta)).
    omega, alpha, beta = 1e-6, 0.08, 0.90
    target_sigma = np.sqrt(omega / (1.0 - alpha - beta))  # ~0.00707
    lp = _simulate_garch_log_prices(n=800, omega=omega, alpha=alpha, beta=beta)
    res = physical_density_from_returns(lp, spot=5000.0, t_years=0.05, horizon_days=12.6)
    # σ_1d should be in the same ballpark (within a factor of ~3).
    assert 0.3 * target_sigma < res.sigma_1d < 3.0 * target_sigma
    assert res.garch_converged in (True, False)
    assert res.n_obs == 800


# ---------------------------------------------------------------------------
# (b) pdf integrates to ~1, cdf monotone in [0, 1], endpoints ~0 and ~1
# ---------------------------------------------------------------------------


def test_pdf_integrates_to_one_and_cdf_monotone() -> None:
    lp = _simulate_garch_log_prices(n=500)
    res = physical_density_from_returns(lp, spot=5000.0, t_years=0.04, horizon_days=10.0)

    area = float(_trapz(res.pdf, res.grid))
    assert area == pytest.approx(1.0, abs=1e-3)

    assert np.all(res.pdf >= 0.0)
    assert np.all(np.diff(res.cdf) >= -1e-12)  # monotone non-decreasing
    assert res.cdf.min() >= 0.0 and res.cdf.max() <= 1.0
    assert res.cdf[0] < 1e-3
    assert res.cdf[-1] > 1.0 - 1e-3


# ---------------------------------------------------------------------------
# (c) higher annual_drift shifts the mean up
# ---------------------------------------------------------------------------


def _density_mean(res: PhysicalDensityResult) -> float:
    return float(_trapz(res.grid * res.pdf, res.grid))


def test_higher_drift_shifts_mean_up() -> None:
    lp = _simulate_garch_log_prices(n=500)
    common = {"spot": 5000.0, "t_years": 0.5, "horizon_days": 126.0}
    low = physical_density_from_returns(lp, annual_drift=0.0, **common)
    high = physical_density_from_returns(lp, annual_drift=0.20, **common)
    # Same grid construction differs by drift; compare distribution means.
    assert _density_mean(high) > _density_mean(low)


# ---------------------------------------------------------------------------
# (d) horizon scaling: sigma_T grows ~sqrt(horizon_days)
# ---------------------------------------------------------------------------


def test_sigma_T_scales_with_sqrt_horizon() -> None:
    lp = _simulate_garch_log_prices(n=500)
    r1 = physical_density_from_returns(lp, spot=5000.0, t_years=0.04, horizon_days=4.0)
    r2 = physical_density_from_returns(lp, spot=5000.0, t_years=0.16, horizon_days=16.0)
    # 4x horizon_days -> 2x sigma_T; sigma_1d identical across calls.
    assert r1.sigma_1d == pytest.approx(r2.sigma_1d, rel=1e-9)
    assert r2.sigma_T / r1.sigma_T == pytest.approx(2.0, rel=1e-6)
    # sigma_ann is horizon-independent.
    assert r1.sigma_ann == pytest.approx(r2.sigma_ann, rel=1e-9)


def test_sigma_ann_annualisation() -> None:
    lp = _simulate_garch_log_prices(n=500)
    res = physical_density_from_returns(lp, spot=5000.0, t_years=0.04, horizon_days=10.0)
    assert res.sigma_ann == pytest.approx(res.sigma_1d * np.sqrt(252.0), rel=1e-9)


# ---------------------------------------------------------------------------
# (e) GARCH-failure / too-few-obs fallback path adds a warning
# ---------------------------------------------------------------------------


def test_few_obs_fallback_adds_warning() -> None:
    # Below the GARCH minimum but enough for a sample-σ.
    lp = _simulate_garch_log_prices(n=40)
    res = physical_density_from_returns(lp, spot=5000.0, t_years=0.04, horizon_days=10.0)
    assert any("sample-σ" in w or "obs" in w for w in res.warnings)
    assert res.garch_converged is False
    assert res.sigma_1d > 0.0
    # Still a valid density.
    assert float(_trapz(res.pdf, res.grid)) == pytest.approx(1.0, abs=1e-3)


def test_garch_exception_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    lp = _simulate_garch_log_prices(n=300)

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic GARCH explosion")

    monkeypatch.setattr("pfm.vol.physical_density.fit_garch_11", _boom)
    res = physical_density_from_returns(lp, spot=5000.0, t_years=0.04, horizon_days=10.0)
    assert any("GARCH fit failed" in w for w in res.warnings)
    assert res.garch_converged is False
    assert res.sigma_1d > 0.0
    assert float(_trapz(res.pdf, res.grid)) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_invalid_spot_raises() -> None:
    lp = _simulate_garch_log_prices(n=200)
    with pytest.raises(ValueError, match="spot"):
        physical_density_from_returns(lp, spot=0.0, t_years=0.04, horizon_days=10.0)


def test_invalid_horizon_raises() -> None:
    lp = _simulate_garch_log_prices(n=200)
    with pytest.raises(ValueError, match="horizon_days"):
        physical_density_from_returns(lp, spot=5000.0, t_years=0.04, horizon_days=0.0)


def test_explicit_grid_used() -> None:
    lp = _simulate_garch_log_prices(n=300)
    grid = np.linspace(4000.0, 6000.0, 250)
    res = physical_density_from_returns(lp, spot=5000.0, t_years=0.04, horizon_days=10.0, grid=grid)
    assert np.array_equal(res.grid, grid)


def test_bad_grid_rejected() -> None:
    lp = _simulate_garch_log_prices(n=300)
    with pytest.raises(ValueError, match="strictly increasing"):
        physical_density_from_returns(
            lp,
            spot=5000.0,
            t_years=0.04,
            horizon_days=10.0,
            grid=np.array([5000.0, 4000.0, 6000.0]),
        )


# ---------------------------------------------------------------------------
# (f) mocked fetch + orchestrator flow
# ---------------------------------------------------------------------------


class _FakeTicker:
    """Minimal stand-in for ``yfinance.Ticker``."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def history(self, period: str = "400d", interval: str = "1d") -> pd.DataFrame:
        n = 400
        lp = _simulate_garch_log_prices(n=n, start_price=7000.0)
        close = np.exp(lp.to_numpy())
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": close}, index=idx)


def test_fetch_index_history_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", _FakeTicker)
    series = fetch_index_history("SPX", lookback_days=400)
    assert isinstance(series, pd.Series)
    assert len(series) == 400
    assert series.name == "log_close_SPX"
    # log-prices are finite and roughly log(7000) scale.
    assert np.all(np.isfinite(series.to_numpy()))
    assert 8.0 < float(series.iloc[0]) < 10.0


def test_fetch_index_history_unknown_asset() -> None:
    with pytest.raises(ValueError, match="unknown asset"):
        fetch_index_history("DOGECOIN")


def test_fetch_index_history_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import yfinance as yf

    class _EmptyTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, period: str = "400d", interval: str = "1d") -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(yf, "Ticker", _EmptyTicker)
    with pytest.raises(ValueError, match="no data"):
        fetch_index_history("NDX")


def test_estimate_physical_density_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", _FakeTicker)
    res = estimate_physical_density(
        "SPX", spot=7000.0, t_years=0.0018, horizon_days=0.67, label="next-close"
    )
    assert isinstance(res, PhysicalDensityResult)
    assert res.asset == "SPX"
    assert res.label == "next-close"
    assert res.spot == 7000.0
    assert res.sigma_1d > 0.0
    assert res.sigma_ann > 0.0
    assert float(_trapz(res.pdf, res.grid)) == pytest.approx(1.0, abs=1e-3)
    assert res.cdf[0] < 1e-3 and res.cdf[-1] > 1.0 - 1e-3
