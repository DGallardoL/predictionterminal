"""Tests for the options-implied risk-neutral density (Breeden-Litzenberger).

The math entry ``rn_density_from_quotes`` is driven with synthetic Black-Scholes
quotes where the true ``σ`` is known, so we can assert exact recovery. The
network fetch is mocked.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from pfm.vol.options_rn import (
    _bl_density_shape_restricted,
    _bs_undisc_call,
    _bs_undisc_put,
    _pava_increasing,
    extract_options_rn,
    implied_vol,
    rn_density_from_quotes,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _synthetic_quotes(forward: float, t_years: float, sigma: float, risk_free: float = 0.045):
    """Flat-vol Black-76 call/put mid quotes (discounted) on a strike grid."""
    disc = float(np.exp(-risk_free * t_years))
    strikes = np.linspace(forward * 0.94, forward * 1.06, 61)
    call_mid = np.array([_bs_undisc_call(forward, k, t_years, sigma) * disc for k in strikes])
    put_mid = np.array([_bs_undisc_put(forward, k, t_years, sigma) * disc for k in strikes])
    return strikes, call_mid, put_mid


# ---------------------------------------------------------------------------
# Black-76 + IV inversion
# ---------------------------------------------------------------------------


def test_implied_vol_roundtrip_call_and_put():
    f, t, sig = 7350.0, 1.0 / 365.0, 0.12
    c = _bs_undisc_call(f, 7400.0, t, sig)
    p = _bs_undisc_put(f, 7300.0, t, sig)
    assert implied_vol(c, f, 7400.0, t, is_call=True) == pytest.approx(sig, abs=1e-3)
    assert implied_vol(p, f, 7300.0, t, is_call=False) == pytest.approx(sig, abs=1e-3)


def test_implied_vol_rejects_out_of_bounds():
    f, t = 100.0, 0.1
    # price below intrinsic / above upper bound → None
    assert implied_vol(-1.0, f, 90.0, t, is_call=True) is None
    assert implied_vol(f + 1, f, 90.0, t, is_call=True) is None  # >= forward upper bound


def test_putcall_parity_holds_undiscounted():
    f, t, sig = 500.0, 0.05, 0.2
    for k in (450.0, 500.0, 560.0):
        c = _bs_undisc_call(f, k, t, sig)
        p = _bs_undisc_put(f, k, t, sig)
        assert c - p == pytest.approx(f - k, abs=1e-6)


# ---------------------------------------------------------------------------
# Shape-restricted density primitives (Aït-Sahalia & Duarte)
# ---------------------------------------------------------------------------


def test_pava_increasing_projects_to_monotone():
    y = np.array([3.0, 1.0, 2.0, 0.5, 4.0])
    out = _pava_increasing(y, np.ones_like(y))
    assert np.all(np.diff(out) >= -1e-12)  # non-decreasing
    # weighted-mean preserving on the whole block
    assert out.mean() == pytest.approx(y.mean(), rel=1e-9)


def test_shape_restricted_density_nonnegative_on_noisy_convex_curve():
    grid = np.linspace(90.0, 110.0, 200)
    # convex decreasing call curve + noise → projection must still be >= 0
    call = np.maximum(100.0 - grid, 0.0) + 5.0 * np.exp(-((grid - 100) ** 2) / 50.0)
    rng = np.random.default_rng(0)
    call_noisy = call + rng.normal(0, 0.02, size=grid.size)
    pdf = _bl_density_shape_restricted(grid, call_noisy)
    assert np.all(pdf >= 0.0)


# ---------------------------------------------------------------------------
# Density recovery from synthetic quotes
# ---------------------------------------------------------------------------


def test_recovers_lognormal_moments_from_flat_vol_quotes():
    f, t, sig = 7350.0, 1.0 / 365.0, 0.11
    strikes, cm, pm = _synthetic_quotes(f, t, sig)
    res = rn_density_from_quotes(strikes, cm, pm, t, spot_hint=f)
    g, pdf = res.grid, res.pdf
    mean = float(_trapz(g * pdf, g))
    std = float(np.sqrt(_trapz((g - mean) ** 2 * pdf, g)))
    theo_std = f * np.sqrt(np.exp(sig**2 * t) - 1)
    assert res.forward == pytest.approx(f, rel=1e-3)
    assert mean == pytest.approx(f, rel=2e-3)
    assert std == pytest.approx(theo_std, rel=0.05)
    assert res.atm_iv == pytest.approx(sig, abs=5e-3)


def test_density_is_valid_distribution():
    f, t, sig = 500.0, 0.08, 0.25
    strikes, cm, pm = _synthetic_quotes(f, t, sig)
    res = rn_density_from_quotes(strikes, cm, pm, t, spot_hint=f)
    assert np.all(res.pdf >= 0.0)
    assert _trapz(res.pdf, res.grid) == pytest.approx(1.0, abs=1e-6)
    assert res.cdf[0] == pytest.approx(0.0, abs=1e-6)
    assert res.cdf[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(res.cdf) >= -1e-9)  # monotone non-decreasing


def test_put_skew_yields_left_skewed_density():
    # Build a smile with richer downside vol (negative skew) and check the
    # recovered density has negative skewness.
    f, t = 1000.0, 0.1
    strikes = np.linspace(f * 0.85, f * 1.15, 61)
    disc = np.exp(-0.045 * t)
    cm, pm = [], []
    for k in strikes:
        kk = np.log(k / f)
        sig = 0.20 - 0.6 * kk  # downside (kk<0) → higher vol
        sig = float(np.clip(sig, 0.05, 0.8))
        cm.append(_bs_undisc_call(f, k, t, sig) * disc)
        pm.append(_bs_undisc_put(f, k, t, sig) * disc)
    res = rn_density_from_quotes(strikes, np.array(cm), np.array(pm), t, spot_hint=f)
    g, pdf = res.grid, res.pdf
    mean = float(_trapz(g * pdf, g))
    std = float(np.sqrt(_trapz((g - mean) ** 2 * pdf, g)))
    skew = float(_trapz(((g - mean) / std) ** 3 * pdf, g))
    assert skew < 0.0


def test_too_few_strikes_raises():
    with pytest.raises(ValueError):
        rn_density_from_quotes(
            np.array([100.0, 101.0]),
            np.array([2.0, 1.5]),
            np.array([1.0, 1.5]),
            0.1,
            spot_hint=100.0,
        )


def test_density_on_supplied_grid_matches_grid():
    f, t, sig = 7350.0, 1.0 / 365.0, 0.11
    strikes, cm, pm = _synthetic_quotes(f, t, sig)
    custom = np.linspace(7100.0, 7600.0, 256)
    res = rn_density_from_quotes(strikes, cm, pm, t, spot_hint=f, grid=custom)
    assert np.array_equal(res.grid, custom)


# ---------------------------------------------------------------------------
# Mocked network fetch + orchestrator
# ---------------------------------------------------------------------------


class _FakeChain:
    def __init__(self, calls, puts):
        self.calls = calls
        self.puts = puts


class _FakeTicker:
    """Minimal yfinance.Ticker stand-in returning flat-vol BS quotes.

    Uses a 1-week horizon and a tight proportional bid/ask so option values are
    large enough that the spread does not corrupt the recovered implied vols.
    """

    def __init__(self, ticker):
        import pandas as pd

        self._pd = pd
        # T must match what fetch_option_chain computes from (expiry - now_utc);
        # the test pins now_utc 7 days before the 2026-05-20 expiry.
        self.f, self.t, self.sig = 7350.0, 7.0 / 365.25, 0.12

    @property
    def options(self):
        return ("2026-05-20", "2026-05-21")

    def _quotes(self, fn):
        disc = float(np.exp(-0.045 * self.t))
        ks = np.linspace(self.f * 0.93, self.f * 1.07, 57)
        mids = np.array([fn(self.f, float(k), self.t, self.sig) * disc for k in ks])
        return self._pd.DataFrame(
            {"strike": ks, "bid": mids * 0.999, "ask": mids * 1.001, "lastPrice": mids}
        )

    def option_chain(self, expiry):
        return _FakeChain(self._quotes(_bs_undisc_call), self._quotes(_bs_undisc_put))

    def history(self, period="1d"):
        return self._pd.DataFrame({"Close": [self.f]})


def test_extract_options_rn_with_mocked_yfinance(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", _FakeTicker)
    now = datetime(2026, 5, 20, 20, 0, 0, tzinfo=UTC) - timedelta(days=7)
    res = extract_options_rn("SPX", target_expiry="2026-05-20", now_utc=now)
    assert res.n_options >= 8
    assert res.forward == pytest.approx(7350.0, rel=2e-3)
    assert res.atm_iv == pytest.approx(0.12, abs=1.5e-2)
    assert _trapz(res.pdf, res.grid) == pytest.approx(1.0, abs=1e-6)
