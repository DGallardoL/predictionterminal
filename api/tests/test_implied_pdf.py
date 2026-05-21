"""Synthetic-DGP recovery tests for the implied-PDF engine.

Pattern (per CLAUDE.md): generate data from a known data-generating process,
run the engine, and assert recovery within tolerance. No network.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
import pytest
from scipy.stats import norm

from pfm.vol.implied_pdf import compute_implied_pdf
from pfm.vol.implied_pdf_schemas import ImpliedPDFResult, LadderEntry, LadderFamily

_trapz = getattr(np, "trapezoid", None) or np.trapz

# A maturity ~1 year out (relative to a fixed now) for deterministic T.
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_MATURITY_1Y = datetime(2026, 12, 31, 18, tzinfo=UTC)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _lognormal_stats(mu: float, sigma: float) -> tuple[float, float, float]:
    """Return analytic (mean, median, std) of LogNormal(mu, sigma)."""
    mean = math.exp(mu + 0.5 * sigma * sigma)
    median = math.exp(mu)
    var = (math.exp(sigma * sigma) - 1.0) * math.exp(2.0 * mu + sigma * sigma)
    return mean, median, math.sqrt(var)


def _assert_arbitrage_free(res: ImpliedPDFResult) -> None:
    grid = np.asarray(res.grid)
    pdf = np.asarray(res.pdf)
    cdf = np.asarray(res.cdf)
    assert np.all(np.diff(grid) > 0), "grid must be strictly increasing"
    assert np.all(pdf >= -1e-9), "pdf must be >= 0"
    assert np.all(cdf >= -1e-9) and np.all(cdf <= 1.0 + 1e-9), "cdf in [0,1]"
    assert np.all(np.diff(cdf) >= -1e-9), "cdf monotone non-decreasing"
    area = float(_trapz(pdf, grid))
    assert abs(area - 1.0) < 0.02, f"pdf integrates to {area}"


# ---------------------------------------------------------------------------
# 1. terminal_ladder recovery
# ---------------------------------------------------------------------------


def test_terminal_ladder_recovers_lognormal_moments() -> None:
    mu, sigma = math.log(100.0), 0.25
    mean, median, std = _lognormal_stats(mu, sigma)
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 125.0, 140.0, 160.0]
    entries = [
        LadderEntry(
            direction="above",
            strike=k,
            prob=float(1.0 - norm.cdf((math.log(k) - mu) / sigma)),
        )
        for k in strikes
    ]
    fam = LadderFamily(
        asset="SYNTH",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=_MATURITY_1Y,
        spot=100.0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, now_utc=_NOW)

    assert res.distribution_of == "terminal_price"
    assert res.n_strikes == 8
    _assert_arbitrage_free(res)
    assert res.moments.mean == pytest.approx(mean, rel=0.06)
    assert res.moments.median == pytest.approx(median, rel=0.06)
    # std is mildly under-recovered: a model-free PCHIP CDF truncates the
    # tails beyond the observed strike span, dropping their variance.
    assert res.moments.std == pytest.approx(std, rel=0.30)


def test_terminal_ladder_below_direction() -> None:
    mu, sigma = math.log(50.0), 0.3
    strikes = [30.0, 40.0, 50.0, 60.0, 75.0, 90.0]
    entries = [
        LadderEntry(
            direction="below",
            strike=k,
            prob=float(norm.cdf((math.log(k) - mu) / sigma)),
        )
        for k in strikes
    ]
    fam = LadderFamily(
        asset="SYNTH",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=_MATURITY_1Y,
        spot=50.0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, now_utc=_NOW)
    _assert_arbitrage_free(res)
    mean, median, _ = _lognormal_stats(mu, sigma)
    assert res.moments.median == pytest.approx(median, rel=0.10)
    assert res.moments.mean == pytest.approx(mean, rel=0.10)


# ---------------------------------------------------------------------------
# 2. terminal_buckets recovery + overround
# ---------------------------------------------------------------------------


def _bucket_family(masses_scale: float = 1.0) -> LadderFamily:
    mu, sigma = math.log(100.0), 0.2
    edges = [60, 70, 80, 90, 100, 110, 120, 135, 150]

    def cdf(x: float) -> float:
        return float(norm.cdf((math.log(x) - mu) / sigma))

    entries: list[LadderEntry] = []
    # Open lower tail.
    entries.append(
        LadderEntry(direction="between", floor=None, cap=float(edges[0]), prob=cdf(edges[0]))
    )
    for lo, hi in pairwise(edges):
        entries.append(
            LadderEntry(
                direction="between",
                floor=float(lo),
                cap=float(hi),
                prob=cdf(hi) - cdf(lo),
            )
        )
    # Open upper tail.
    entries.append(
        LadderEntry(
            direction="between", floor=float(edges[-1]), cap=None, prob=1.0 - cdf(edges[-1])
        )
    )
    # Apply overround scale.
    entries = [
        LadderEntry(
            direction=e.direction,
            prob=e.prob * masses_scale,
            floor=e.floor,
            cap=e.cap,
        )
        for e in entries
    ]
    return LadderFamily(
        asset="SYNTH",
        asset_class="equity_index",
        data_shape="terminal_buckets",
        maturity_utc=_MATURITY_1Y,
        spot=100.0,
        entries=entries,
    )


def test_terminal_buckets_recovers_moments() -> None:
    fam = _bucket_family(masses_scale=1.0)
    res = compute_implied_pdf(fam, now_utc=_NOW)
    assert res.distribution_of == "terminal_price"
    _assert_arbitrage_free(res)
    mean, median, std = _lognormal_stats(math.log(100.0), 0.2)
    assert res.moments.mean == pytest.approx(mean, rel=0.05)
    assert res.moments.median == pytest.approx(median, rel=0.05)
    assert res.moments.std == pytest.approx(std, rel=0.25)


def test_terminal_buckets_overround_renormalised() -> None:
    fam = _bucket_family(masses_scale=1.05)
    res = compute_implied_pdf(fam, now_utc=_NOW)
    _assert_arbitrage_free(res)
    assert any("overround" in w for w in res.warnings)
    mean, _median, _std = _lognormal_stats(math.log(100.0), 0.2)
    assert res.moments.mean == pytest.approx(mean, rel=0.05)


# ---------------------------------------------------------------------------
# 3. barrier_touch / running-max
# ---------------------------------------------------------------------------


def _driftless_running_max_survival(k: float, s0: float, sigma: float, t: float) -> float:
    """P(M_T >= K) for driftless GBM = 2 Φ(-a/(σ√T)), a=ln(K/S0), K>S0."""
    a = math.log(k / s0)
    if a <= 0:
        return 1.0
    return float(2.0 * norm.cdf(-a / (sigma * math.sqrt(t))))


def test_barrier_touch_is_running_max() -> None:
    s0, sigma, t = 100.0, 0.4, 1.0
    strikes = [105.0, 115.0, 130.0, 150.0, 175.0, 210.0]
    entries = [
        LadderEntry(
            direction="touch_above",
            strike=k,
            prob=_driftless_running_max_survival(k, s0, sigma, t),
        )
        for k in strikes
    ]
    fam = LadderFamily(
        asset="SYNTH",
        asset_class="crypto",
        data_shape="barrier_touch",
        maturity_utc=_MATURITY_1Y,
        spot=s0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, now_utc=_NOW)
    assert res.distribution_of == "running_max"
    assert any("running_max" in w for w in res.warnings)
    _assert_arbitrage_free(res)
    assert res.gbm_fit is None
    assert res.gbm_terminal_overlay is None


def test_barrier_to_terminal_recovers_sigma() -> None:
    s0, sigma, t = 100.0, 0.4, 1.0
    strikes = [105.0, 115.0, 130.0, 150.0, 175.0, 210.0, 260.0]
    entries = [
        LadderEntry(
            direction="touch_above",
            strike=k,
            prob=_driftless_running_max_survival(k, s0, sigma, t),
        )
        for k in strikes
    ]
    fam = LadderFamily(
        asset="SYNTH",
        asset_class="crypto",
        data_shape="barrier_touch",
        maturity_utc=_MATURITY_1Y,
        spot=s0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, barrier_to_terminal=True, now_utc=_NOW)
    assert res.gbm_fit is not None
    assert res.gbm_fit.converted_to_terminal is True
    assert res.gbm_fit.sigma_annual == pytest.approx(sigma, rel=0.20)
    assert res.gbm_terminal_overlay is not None
    overlay = np.asarray(res.gbm_terminal_overlay)
    assert float(_trapz(overlay, np.asarray(res.grid))) == pytest.approx(1.0, abs=0.05)


def test_barrier_to_terminal_estimates_spot_when_missing() -> None:
    s0, sigma, t = 100.0, 0.4, 1.0
    strikes = [105.0, 115.0, 130.0, 150.0, 175.0, 210.0]
    entries = [
        LadderEntry(
            direction="touch_above",
            strike=k,
            prob=_driftless_running_max_survival(k, s0, sigma, t),
        )
        for k in strikes
    ]
    fam = LadderFamily(
        asset="SYNTH",
        asset_class="crypto",
        data_shape="barrier_touch",
        maturity_utc=_MATURITY_1Y,
        spot=None,
        entries=entries,
    )
    res = compute_implied_pdf(fam, barrier_to_terminal=True, now_utc=_NOW)
    assert any("spot estimated" in w for w in res.warnings)
    assert res.spot is not None


def test_barrier_touch_below_running_min() -> None:
    s0, sigma, t = 100.0, 0.4, 1.0

    def touch_below(k: float) -> float:
        a = math.log(k / s0)
        if a >= 0:
            return 1.0
        return float(2.0 * norm.cdf(a / (sigma * math.sqrt(t))))

    strikes = [95.0, 85.0, 70.0, 55.0, 40.0]
    entries = [LadderEntry(direction="touch_below", strike=k, prob=touch_below(k)) for k in strikes]
    fam = LadderFamily(
        asset="SYNTH",
        asset_class="crypto",
        data_shape="barrier_touch",
        maturity_utc=_MATURITY_1Y,
        spot=s0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, now_utc=_NOW)
    assert res.distribution_of == "running_min"
    _assert_arbitrage_free(res)


# ---------------------------------------------------------------------------
# 4. arbitrage-free invariants (parametrised)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid_size", [64, 128, 256])
def test_invariants_ladder(grid_size: int) -> None:
    mu, sigma = math.log(100.0), 0.3
    strikes = [60.0, 80.0, 100.0, 120.0, 150.0]
    entries = [
        LadderEntry(
            direction="above",
            strike=k,
            prob=float(1.0 - norm.cdf((math.log(k) - mu) / sigma)),
        )
        for k in strikes
    ]
    fam = LadderFamily(
        asset="X",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=_MATURITY_1Y,
        spot=100.0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, grid_size=grid_size, now_utc=_NOW)
    assert len(res.grid) == grid_size
    _assert_arbitrage_free(res)


# ---------------------------------------------------------------------------
# 5. edge cases
# ---------------------------------------------------------------------------


def test_non_monotone_survival_is_monotonised() -> None:
    # Deliberately broken: survival increases at one strike.
    entries = [
        LadderEntry(direction="above", strike=80.0, prob=0.7),
        LadderEntry(direction="above", strike=100.0, prob=0.8),  # impossible bump
        LadderEntry(direction="above", strike=120.0, prob=0.3),
        LadderEntry(direction="above", strike=150.0, prob=0.1),
    ]
    fam = LadderFamily(
        asset="X",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=_MATURITY_1Y,
        spot=100.0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, now_utc=_NOW)
    _assert_arbitrage_free(res)


def test_eps_clipping_exposed() -> None:
    entries = [
        LadderEntry(direction="above", strike=80.0, prob=0.999),
        LadderEntry(direction="above", strike=100.0, prob=0.5),
        LadderEntry(direction="above", strike=120.0, prob=0.001),
    ]
    fam = LadderFamily(
        asset="X",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=_MATURITY_1Y,
        spot=100.0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, eps=0.05, now_utc=_NOW)
    assert res.eps == 0.05
    _assert_arbitrage_free(res)


def test_few_strikes_warns_but_returns() -> None:
    entries = [
        LadderEntry(direction="above", strike=90.0, prob=0.6),
        LadderEntry(direction="above", strike=110.0, prob=0.3),
    ]
    fam = LadderFamily(
        asset="X",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=_MATURITY_1Y,
        spot=100.0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, now_utc=_NOW)
    assert any("few_strikes" in w for w in res.warnings)
    _assert_arbitrage_free(res)


def test_empty_input_raises() -> None:
    fam = LadderFamily(
        asset="X",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=_MATURITY_1Y,
        spot=100.0,
        entries=[],
    )
    with pytest.raises(ValueError):
        compute_implied_pdf(fam, now_utc=_NOW)


def test_degenerate_flat_probs_raises() -> None:
    entries = [
        LadderEntry(direction="above", strike=80.0, prob=0.5),
        LadderEntry(direction="above", strike=100.0, prob=0.5),
        LadderEntry(direction="above", strike=120.0, prob=0.5),
    ]
    fam = LadderFamily(
        asset="X",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=_MATURITY_1Y,
        spot=100.0,
        entries=entries,
    )
    with pytest.raises(ValueError):
        compute_implied_pdf(fam, now_utc=_NOW)


def test_past_maturity_warns() -> None:
    mu, sigma = math.log(100.0), 0.3
    strikes = [70.0, 100.0, 130.0, 160.0]
    entries = [
        LadderEntry(
            direction="above",
            strike=k,
            prob=float(1.0 - norm.cdf((math.log(k) - mu) / sigma)),
        )
        for k in strikes
    ]
    fam = LadderFamily(
        asset="X",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=datetime(2020, 1, 1, tzinfo=UTC),
        spot=100.0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, now_utc=_NOW)
    assert any("past_maturity" in w for w in res.warnings)
    assert res.time_to_maturity_years > 0


# ---------------------------------------------------------------------------
# 6. method + tail_model variants
# ---------------------------------------------------------------------------


def test_method_lognormal_primary() -> None:
    mu, sigma = math.log(100.0), 0.25
    strikes = [70.0, 85.0, 100.0, 115.0, 130.0, 150.0]
    entries = [
        LadderEntry(
            direction="above",
            strike=k,
            prob=float(1.0 - norm.cdf((math.log(k) - mu) / sigma)),
        )
        for k in strikes
    ]
    fam = LadderFamily(
        asset="X",
        asset_class="equity_index",
        data_shape="terminal_ladder",
        maturity_utc=_MATURITY_1Y,
        spot=100.0,
        entries=entries,
    )
    res = compute_implied_pdf(fam, method="lognormal", now_utc=_NOW)
    assert res.method == "lognormal"
    _assert_arbitrage_free(res)
    mean, _median, _std = _lognormal_stats(mu, sigma)
    assert res.moments.mean == pytest.approx(mean, rel=0.10)


@pytest.mark.parametrize("tail_model", ["lognormal", "linear", "none"])
def test_tail_model_variants(tail_model: str) -> None:
    fam = _bucket_family(masses_scale=1.0)
    res = compute_implied_pdf(fam, tail_model=tail_model, now_utc=_NOW)  # type: ignore[arg-type]
    _assert_arbitrage_free(res)


def test_market_points_and_quantile_order() -> None:
    fam = _bucket_family(masses_scale=1.0)
    res = compute_implied_pdf(fam, now_utc=_NOW)
    assert len(res.market_points) == len(fam.entries)
    assert all(mp.kind == "mass" for mp in res.market_points)
    q = res.quantiles
    assert q.p5 < q.p25 < q.p50 < q.p75 < q.p95
