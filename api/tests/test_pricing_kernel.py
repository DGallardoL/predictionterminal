"""Tests for the cross-venue / pricing-kernel orchestration.

The three density sources (Kalshi-Q, options-Q, physical-P) are mocked with
synthetic Gaussians of known mean/σ so the divergence, kernel and risk-aversion
outputs can be asserted analytically.
"""

from __future__ import annotations

import types
from datetime import UTC, datetime

import numpy as np
import pytest

from pfm.vol.options_rn import OptionsRNResult
from pfm.vol.physical_density import PhysicalDensityResult
from pfm.vol.pricing_kernel import compute_pricing_kernel

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _gauss(grid: np.ndarray, mean: float, std: float) -> np.ndarray:
    pdf = np.exp(-0.5 * ((grid - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    return pdf / float(_trapz(pdf, grid))


def _cdf(grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    return c / c[-1]


def _patch_sources(
    monkeypatch,
    *,
    spot=7350.0,
    kalshi_std=110.0,
    options_std=60.0,
    phys_std=50.0,
    options_mean=7350.0,
    kalshi_mean=7350.0,
    sigma_ann=0.13,
):
    expiry = datetime(2026, 5, 20, 20, 0, 0, tzinfo=UTC)
    t_years = 0.0021

    kg = np.linspace(spot - 800, spot + 800, 400)
    kalshi = types.SimpleNamespace(
        grid=kg.tolist(),
        pdf=_gauss(kg, kalshi_mean, kalshi_std).tolist(),
        cdf=_cdf(kg, _gauss(kg, kalshi_mean, kalshi_std)).tolist(),
        maturity_utc=expiry,
        time_to_maturity_years=t_years,
        spot=spot,
        n_strikes=30,
    )

    og = np.linspace(spot - 500, spot + 500, 400)
    opt = OptionsRNResult(
        asset="SPX",
        forward=spot,
        t_years=t_years,
        risk_free=0.045,
        grid=og,
        pdf=_gauss(og, options_mean, options_std),
        cdf=_cdf(og, _gauss(og, options_mean, options_std)),
        smile_k=np.array([-0.02, 0.0, 0.02]),
        smile_iv=np.array([0.19, 0.18, 0.185]),
        n_options=120,
        atm_iv=0.18,
        warnings=[],
    )

    pg = np.linspace(spot - 500, spot + 500, 400)
    phys = PhysicalDensityResult(
        spot=spot,
        t_years=t_years,
        horizon_days=0.77,
        sigma_1d=0.008,
        sigma_ann=sigma_ann,
        sigma_T=0.007,
        annual_drift=0.06,
        risk_free=0.045,
        grid=pg,
        pdf=_gauss(pg, spot, phys_std),
        cdf=_cdf(pg, _gauss(pg, spot, phys_std)),
        garch_persistence=0.93,
        garch_converged=True,
        n_obs=400,
        warnings=[],
    )

    monkeypatch.setattr(
        "pfm.sources.kalshi.discover_index_ladder",
        lambda key, client=None, maturity_filter=None: types.SimpleNamespace(
            spot=spot, maturity_utc=expiry, entries=[]
        ),
    )
    monkeypatch.setattr("pfm.vol.implied_pdf.compute_implied_pdf", lambda family: kalshi)
    monkeypatch.setattr(
        "pfm.vol.options_rn.extract_options_rn",
        lambda asset, target_expiry=None, risk_free=0.045, now_utc=None, price_t_years=None: opt,
    )
    monkeypatch.setattr(
        "pfm.vol.physical_density.estimate_physical_density",
        lambda asset, *, spot, t_years, horizon_days, annual_drift, risk_free, lookback_days: phys,
    )


def test_three_densities_and_moments(monkeypatch):
    _patch_sources(monkeypatch)
    r = compute_pricing_kernel("SPX", kalshi_client=None)
    venues = {d.venue for d in r.densities}
    assert venues == {"kalshi", "options", "garch"}
    by = {d.venue: d for d in r.densities}
    assert by["kalshi"].std > by["options"].std > by["garch"].std  # 110 > 60 > 50
    assert by["kalshi"].mean == pytest.approx(7350, abs=8)


def test_cross_venue_divergence_nonnegative_and_std_ratio(monkeypatch):
    _patch_sources(monkeypatch, kalshi_std=110.0, options_std=60.0)
    r = compute_pricing_kernel("SPX", kalshi_client=None)
    cv = r.cross_venue
    assert cv.kl_kalshi_given_options >= 0
    assert cv.kl_options_given_kalshi >= 0
    assert cv.jensen_shannon >= 0
    assert cv.std_ratio == pytest.approx(110.0 / 60.0, rel=0.15)
    assert len(cv.ratio) == len(r.grid)


def test_identical_rn_measures_have_zero_divergence(monkeypatch):
    _patch_sources(
        monkeypatch, kalshi_std=60.0, options_std=60.0, kalshi_mean=7350.0, options_mean=7350.0
    )
    r = compute_pricing_kernel("SPX", kalshi_client=None)
    assert r.cross_venue.jensen_shannon == pytest.approx(0.0, abs=1e-3)
    assert r.cross_venue.std_ratio == pytest.approx(1.0, rel=0.05)


def test_pricing_kernel_and_risk_aversion_present_central(monkeypatch):
    _patch_sources(monkeypatch)
    r = compute_pricing_kernel("SPX", kalshi_client=None)
    km = [x for x in r.pricing_kernel if x is not None]
    assert len(km) >= 20
    assert all(m > 0 for m in km)
    # central relative risk aversion should be O(1-50), not exploding
    rra = [x for x in r.relative_risk_aversion if x is not None]
    median = sorted(rra)[len(rra) // 2]
    assert -50 < median < 80


def test_variance_risk_premium_positive_when_iv_above_realized(monkeypatch):
    # options σ_ann (from std/spot/√T) > physical σ_ann → positive VRP
    _patch_sources(monkeypatch, options_std=70.0, sigma_ann=0.10)
    r = compute_pricing_kernel("SPX", kalshi_client=None)
    assert r.variance_risk_premium > 0


def test_grid_and_kernel_arrays_aligned(monkeypatch):
    _patch_sources(monkeypatch)
    r = compute_pricing_kernel("SPX", kalshi_client=None)
    n = len(r.grid)
    assert len(r.pricing_kernel) == n
    assert len(r.implied_risk_aversion) == n
    assert len(r.relative_risk_aversion) == n
    assert len(r.cross_venue.ratio) == n
    for d in r.densities:
        assert len(d.pdf) == n
        assert len(d.cdf) == n
