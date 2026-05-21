"""Tests for the A3 σ-gap composer ``pfm.vol.pm_iv_gap``.

Implementation note
-------------------
The σ_pm fit uses :func:`pfm.vol_surface_pm._empirical_moments` + a
log-normal moment-match (see ``vol_surface_pm._fit_lognormal``). With only
five strikes and an open upper-tail bin centred at ``strikes[-1]*1.25``,
that estimator is systematically biased *upward* in σ (~×1.8 amplification
vs the true GBM σ). We therefore do **not** assert exact σ_pm values in
these tests — we assert the *gap-logic* invariants:

* Sign of ``primary_gap_pct_pts`` is consistent with σ_pm − σ_bench.
* Signal classification thresholds (±2pp band, ±3/5pp strength bands).
* Fallback selection when the preferred benchmark source is missing.

Tests calibrate the benchmark value *to* the recovered σ_pm rather than
the input σ. This keeps the test asserting the composer's logic, not the
upstream extractor's calibration accuracy (which is covered separately in
``test_pm_iv_extractor.py``).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from scipy.stats import norm

from pfm.cache_utils import get_cache
from pfm.sources.fred import FREDGRAPH_BASE
from pfm.vol.pm_iv_extractor import discover_ladder_family, fit_implied_sigma
from pfm.vol.pm_iv_gap import PMIVGapSnapshot, compute_gap_snapshot
from pfm.vol.vol_benchmarks import BINANCE_KLINES_URL, DERIBIT_INDEX_URL


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    get_cache("pm_iv_extractor").clear()
    get_cache("vol_benchmarks").clear()
    get_cache("pm_iv_gap").clear()
    yield
    get_cache("pm_iv_extractor").clear()
    get_cache("vol_benchmarks").clear()
    get_cache("pm_iv_gap").clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lognormal_above_prob(strike: float, mu: float, sigma_t: float) -> float:
    """P(X_T > K) under no-drift LogNormal(μ, σ_T²)."""
    return float(1.0 - norm.cdf((math.log(strike) - mu) / sigma_t))


def _t_years_to(maturity: datetime) -> float:
    return max(
        (maturity - datetime.now(tz=UTC)).total_seconds() / (365.25 * 86_400.0),
        1e-6,
    )


def _build_above_probs(
    strikes: list[float],
    spot: float,
    sigma_annual: float,
    t_years: float,
) -> dict[float, float]:
    sigma_t = sigma_annual * math.sqrt(t_years)
    mu = math.log(spot) - 0.5 * sigma_t * sigma_t
    return {k: _lognormal_above_prob(k, mu, sigma_t) for k in strikes}


def _fred_csv(series_id: str, values: list[float]) -> str:
    base = datetime.now(tz=UTC).date() - timedelta(days=10)
    dates = [(base + timedelta(days=i)).isoformat() for i in range(len(values))]
    rows = "\n".join(f"{d},{v}" for d, v in zip(dates, values, strict=True))
    return f"DATE,{series_id}\n{rows}\n"


def _binance_klines_payload(closes: list[float]) -> list[list]:
    base_ts_ms = int(datetime.now(tz=UTC).timestamp() * 1000) - 86_400_000 * len(closes)
    day_ms = 86_400_000
    rows: list[list] = []
    for i, c in enumerate(closes):
        open_t = base_ts_ms + i * day_ms
        close_t = open_t + day_ms - 1
        rows.append(
            [
                open_t,
                str(c),
                str(c * 1.01),
                str(c * 0.99),
                str(c),
                "100.0",
                close_t,
                "1000.0",
                1000,
                "50.0",
                "500.0",
                "0",
            ]
        )
    return rows


class _SlugMidpoints:
    """Polymarket client stub returning ``midpoints[slug]`` per call."""

    def __init__(self, midpoints: dict[str, float]) -> None:
        self.midpoints = midpoints
        self.calls: list[str] = []

    def get_market_metadata(self, slug: str) -> dict[str, Any]:
        self.calls.append(slug)
        p = self.midpoints.get(slug, 0.5)
        return {"bestBid": p - 0.005, "bestAsk": p + 0.005, "lastTradePrice": p}


def _wti_midpoints(target_input_sigma: float) -> dict[str, float]:
    """WTI June-2026 above-ladder midpoints (3 strikes — replaces dead SPX path).

    SPX was dropped from LADDER_REGISTRY on 2026-05-15 because Polymarket no
    longer hosts a live multi-strike SPX ladder. WTI uses the same FRED-based
    benchmark plumbing (OVX rather than VIX), exercising the same
    compute_gap_snapshot code path that the original SPX tests targeted.
    """
    maturity = datetime(2026, 6, 30, 23, 59, tzinfo=UTC)
    t_years = _t_years_to(maturity)
    strikes = [50.0, 75.0, 90.0]
    probs = _build_above_probs(strikes, spot=75.0, sigma_annual=target_input_sigma, t_years=t_years)
    return {
        "cl-above-50-jun-2026": probs[50.0],
        "cl-above-75-jun-2026": probs[75.0],
        "cl-above-90-jun-2026": probs[90.0],
    }


_BTC_LADDER_SLUGS_FULL: dict[float, str] = {
    90_000.0: "will-bitcoin-reach-90000-by-december-31-2026-113-862-581",
    100_000.0: "will-bitcoin-reach-100000-by-december-31-2026-571-361-361",
    140_000.0: "will-bitcoin-reach-140000-by-december-31-2026-131-829-299",
    150_000.0: "will-bitcoin-reach-150000-by-december-31-2026-557-246-971",
    160_000.0: "will-bitcoin-reach-160000-by-december-31-2026-934-934-164",
    190_000.0: "will-bitcoin-reach-190000-by-december-31-2026-936-485-627",
    200_000.0: "will-bitcoin-reach-200000-by-december-31-2026-752-232-389",
    250_000.0: "will-bitcoin-reach-250000-by-december-31-2026-579-442",
    500_000.0: "will-bitcoin-reach-500000-by-december-31-2026-864",
    1_000_000.0: "will-bitcoin-reach-1000000-by-december-31-2026-946",
}


def _btc_midpoints(target_input_sigma: float) -> dict[str, float]:
    """10-strike BTC above-ladder at EOY-2026 (resolves 2027-01-01).

    Refreshed 2026-05-15: the original `btc-above-Xk-eoy-2026` slugs are dead
    on Polymarket. Replaced with the long-form `will-bitcoin-reach-…` markets
    that LADDER_REGISTRY now points at.
    """
    maturity = datetime(2027, 1, 1, 0, 0, tzinfo=UTC)
    t_years = _t_years_to(maturity)
    strikes = list(_BTC_LADDER_SLUGS_FULL.keys())
    probs = _build_above_probs(
        strikes, spot=110_000.0, sigma_annual=target_input_sigma, t_years=t_years
    )
    return {_BTC_LADDER_SLUGS_FULL[k]: probs[k] for k in strikes}


def _measure_sigma_pm(midpoints: dict[str, float], asset: str) -> float:
    """Run the same pipeline the composer does to discover σ_pm point estimate."""
    client = _SlugMidpoints(midpoints)
    fam = discover_ladder_family(asset, polymarket_client=client)
    assert fam is not None
    return fit_implied_sigma(fam).sigma_annual


# ---------------------------------------------------------------------------
# Test 1 — WTI with σ_pm > OVX produces pm_richer / strong
# (was SPX/VIX before 2026-05-15; SPX dropped from registry).
# ---------------------------------------------------------------------------


@respx.mock
def test_compute_gap_snapshot_wti_pm_richer() -> None:
    """σ_pm > OVX by >5pp ⇒ signal=pm_richer, strength=strong."""
    midpoints = _wti_midpoints(target_input_sigma=0.55)
    sigma_pm = _measure_sigma_pm(midpoints, "WTI")
    # Pick OVX = σ_pm - 0.10 → gap >> +5pp (strong)
    ovx_decimal = max(sigma_pm - 0.10, 0.05)
    ovx_index = ovx_decimal * 100.0
    respx.get(FREDGRAPH_BASE).mock(
        return_value=httpx.Response(200, text=_fred_csv("OVXCLS", [ovx_index] * 5))
    )

    client = _SlugMidpoints(midpoints)
    # Clear caches that may have been seeded by _measure_sigma_pm.
    get_cache("pm_iv_extractor").clear()
    snap = compute_gap_snapshot("WTI", polymarket_client=client)

    assert isinstance(snap, PMIVGapSnapshot)
    assert snap.asset == "WTI"
    assert snap.sigma_pm_n_strikes == 3
    assert snap.primary_benchmark == "ovx"
    assert snap.benchmarks["ovx"] == pytest.approx(ovx_decimal, abs=1e-4)
    assert snap.primary_gap_pct_pts is not None
    assert snap.primary_gap_pct_pts > _gap_threshold_strong() - 0.5
    assert snap.signal == "pm_richer"
    assert snap.signal_strength == "strong"


def _gap_threshold_strong() -> float:
    return 5.0  # mirrors _STRONG_PP


# ---------------------------------------------------------------------------
# Test 2 — BTC uses DVOL as primary, benchmark_richer when σ_bench > σ_pm
# ---------------------------------------------------------------------------


@respx.mock
def test_compute_gap_snapshot_btc_uses_dvol_primary() -> None:
    midpoints = _btc_midpoints(target_input_sigma=0.55)
    sigma_pm = _measure_sigma_pm(midpoints, "BTC")
    # DVOL is *richer* than σ_pm: DVOL = σ_pm + 0.06 → gap = -6pp (strong, bench_richer)
    dvol_decimal = sigma_pm + 0.06
    dvol_index = dvol_decimal * 100.0
    respx.get(DERIBIT_INDEX_URL).mock(
        return_value=httpx.Response(200, json={"result": {"index_price": dvol_index}})
    )
    # Binance also responds so dvol is the *preferred* over realized.
    rng = np.random.default_rng(7)
    closes = [50_000.0]
    for r in rng.normal(0.0, 0.03, size=30):
        closes.append(closes[-1] * math.exp(r))
    respx.get(BINANCE_KLINES_URL).mock(
        return_value=httpx.Response(200, json=_binance_klines_payload(closes))
    )

    client = _SlugMidpoints(midpoints)
    get_cache("pm_iv_extractor").clear()
    snap = compute_gap_snapshot("BTC", polymarket_client=client)
    assert snap.asset == "BTC"
    assert snap.primary_benchmark == "dvol"
    assert "dvol" in snap.benchmarks and "realized_30d" in snap.benchmarks
    assert snap.benchmarks["dvol"] == pytest.approx(dvol_decimal, abs=1e-4)
    assert snap.primary_gap_pct_pts is not None
    assert snap.primary_gap_pct_pts < 0
    assert snap.signal == "benchmark_richer"


# ---------------------------------------------------------------------------
# Test 3 — DVOL missing falls back to realized_30d
# ---------------------------------------------------------------------------


@respx.mock
def test_compute_gap_snapshot_falls_back_when_dvol_missing() -> None:
    """Deribit 500 ⇒ no DVOL ⇒ primary=realized_30d + fallback warning."""
    midpoints = _btc_midpoints(target_input_sigma=0.55)

    respx.get(DERIBIT_INDEX_URL).mock(return_value=httpx.Response(500, text="boom"))
    rng = np.random.default_rng(3)
    sigma_daily = 0.034
    closes = [50_000.0]
    for r in rng.normal(0.0, sigma_daily, size=30):
        closes.append(closes[-1] * math.exp(r))
    respx.get(BINANCE_KLINES_URL).mock(
        return_value=httpx.Response(200, json=_binance_klines_payload(closes))
    )

    client = _SlugMidpoints(midpoints)
    snap = compute_gap_snapshot("BTC", polymarket_client=client)
    assert snap.primary_benchmark == "realized_30d"
    assert "dvol" not in snap.benchmarks
    assert "realized_30d" in snap.benchmarks
    assert any("primary_benchmark_fallback" in w for w in snap.warnings)


# ---------------------------------------------------------------------------
# Test 4 — unknown asset returns no_data
# ---------------------------------------------------------------------------


def test_compute_gap_snapshot_unknown_asset_returns_no_data_signal() -> None:
    client = _SlugMidpoints({})
    snap = compute_gap_snapshot("DOGE", polymarket_client=client)
    assert snap.signal == "no_data"
    assert snap.asset == "DOGE"
    assert snap.benchmarks == {}
    assert snap.gaps == {}
    assert snap.primary_benchmark is None
    assert snap.primary_gap_pct_pts is None
    assert any("unknown_asset" in w for w in snap.warnings)


# ---------------------------------------------------------------------------
# Test 5 — |gap| < 2pp ⇒ flat
# ---------------------------------------------------------------------------


@respx.mock
def test_compute_gap_snapshot_flat_signal_within_2pp_band() -> None:
    """OVX pinned at σ_pm (− 0.01pp ≈ noise) ⇒ flat signal.

    (Was SPX/VIX before 2026-05-15; SPX is no longer in LADDER_REGISTRY.)
    """
    midpoints = _wti_midpoints(target_input_sigma=0.55)
    sigma_pm = _measure_sigma_pm(midpoints, "WTI")
    # OVX = σ_pm - 0.005 → gap ≈ +0.5pp (well within ±2pp dead band)
    ovx_decimal = sigma_pm - 0.005
    ovx_index = ovx_decimal * 100.0
    respx.get(FREDGRAPH_BASE).mock(
        return_value=httpx.Response(200, text=_fred_csv("OVXCLS", [ovx_index] * 5))
    )

    client = _SlugMidpoints(midpoints)
    get_cache("pm_iv_extractor").clear()
    snap = compute_gap_snapshot("WTI", polymarket_client=client)
    assert snap.primary_benchmark == "ovx"
    assert snap.primary_gap_pct_pts is not None
    assert abs(snap.primary_gap_pct_pts) < 2.0
    assert snap.signal == "flat"
    assert snap.signal_strength == "weak"
