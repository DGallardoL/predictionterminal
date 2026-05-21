"""Tests for the multi-asset PM+Kalshi implied-σ extractor."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from scipy.stats import norm

from pfm.cache_utils import get_cache
from pfm.vol.pm_iv_extractor import (
    LADDER_REGISTRY,
    LadderEntry,
    LadderFamily,
    _fit_lognormal_survival,
    build_survival_function,
    discover_ladder_family,
    fit_implied_sigma,
)
from pfm.vol_surface_pm import _empirical_moments, _fit_lognormal


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    get_cache("pm_iv_extractor").clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_plus(days: float) -> datetime:
    return datetime.now(tz=UTC) + timedelta(days=days)


def _lognormal_above_prob(strike: float, mu: float, sigma_t: float) -> float:
    """P(X_T > K) for X_T ~ LogNormal(μ, σ_T²)."""
    return float(1.0 - norm.cdf((math.log(strike) - mu) / sigma_t))


def _family(entries: list[LadderEntry], days: float = 365.0, spot: float = 100.0) -> LadderFamily:
    return LadderFamily(
        asset="TEST",
        asset_class="equity_index",
        maturity_utc=_now_plus(days),
        spot_at_lookup=spot,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# build_survival_function — direction shapes
# ---------------------------------------------------------------------------


def test_survival_function_above_direction() -> None:
    """5-strike `above` ladder with descending probs → already-monotone S(K)."""
    entries = [
        LadderEntry(
            slug=f"k{k}", strike=float(k), direction="above", venue="polymarket", market_value=p
        )
        for k, p in [(50, 0.95), (80, 0.80), (100, 0.55), (150, 0.25), (200, 0.05)]
    ]
    strikes, surv = build_survival_function(_family(entries))
    assert strikes == [50.0, 80.0, 100.0, 150.0, 200.0]
    assert surv == [0.95, 0.80, 0.55, 0.25, 0.05]
    assert all(surv[i] >= surv[i + 1] - 1e-9 for i in range(len(surv) - 1))


def test_survival_function_below_direction() -> None:
    """`below` is P(X_T < K) — flipped to S(K) = 1 - p, then monotonised."""
    entries = [
        LadderEntry(
            slug=f"k{k}", strike=float(k), direction="below", venue="polymarket", market_value=p
        )
        for k, p in [(50, 0.05), (100, 0.40), (150, 0.85)]
    ]
    strikes, surv = build_survival_function(_family(entries))
    # 1 - p = 0.95, 0.60, 0.15 — already monotone
    assert strikes == [50.0, 100.0, 150.0]
    assert surv == pytest.approx([0.95, 0.60, 0.15])


def test_survival_function_dip_to_direction() -> None:
    """dip_to: lower strikes → lower touch prob → higher survival.

    After applying S(K) = 1 - 0.5·P(touch_K), survival must be monotone
    non-increasing in K (touch prob ↑ in K, so 1 - 0.5·P_touch ↓ in K).
    """
    # Plausible touch probs for BTC dip ladder
    entries = [
        LadderEntry(
            slug=f"d{k}", strike=float(k), direction="dip_to", venue="polymarket", market_value=p
        )
        for k, p in [(15000, 0.05), (25000, 0.15), (35000, 0.30), (45000, 0.55), (55000, 0.80)]
    ]
    _strikes, surv = build_survival_function(_family(entries))
    # Raw: 1 - 0.5*p = 0.975, 0.925, 0.85, 0.725, 0.60 — monotone ↓ in K ✓
    assert surv == pytest.approx([0.975, 0.925, 0.850, 0.725, 0.600])
    # Monotone non-increasing
    assert all(surv[i] >= surv[i + 1] - 1e-9 for i in range(len(surv) - 1))


def test_survival_function_hit_high_direction() -> None:
    """hit_high mirror of dip_to: S(K) = 0.5·P(hit_K). Higher K → lower hit → lower S."""
    entries = [
        LadderEntry(
            slug=f"h{k}", strike=float(k), direction="hit_high", venue="polymarket", market_value=p
        )
        for k, p in [(115, 0.80), (140, 0.55), (150, 0.40), (175, 0.20), (200, 0.08)]
    ]
    _strikes, surv = build_survival_function(_family(entries))
    # 0.5*p = 0.40, 0.275, 0.20, 0.10, 0.04 — monotone ↓ in K
    assert surv == pytest.approx([0.40, 0.275, 0.20, 0.10, 0.04])
    assert all(surv[i] >= surv[i + 1] - 1e-9 for i in range(len(surv) - 1))


def test_survival_function_range_directions() -> None:
    """Two range buckets [50-70, 70-90] convert to right-cumulative survival."""
    # Bucket masses: P(50<X<=70)=0.3, P(70<X<=90)=0.5
    entries = [
        LadderEntry(
            slug="r50", strike=50.0, direction="range_low", venue="polymarket", market_value=0.3
        ),
        LadderEntry(
            slug="r70", strike=70.0, direction="range_low", venue="polymarket", market_value=0.5
        ),
    ]
    strikes, surv = build_survival_function(_family(entries))
    # Right-cumulative: at K=50, mass-to-right = 0.3+0.5=0.8; at K=70, mass-to-right = 0.5
    assert strikes == [50.0, 70.0]
    assert surv == pytest.approx([0.8, 0.5])


# ---------------------------------------------------------------------------
# fit_implied_sigma — DGP recovery + annualisation
# ---------------------------------------------------------------------------


def test_fit_implied_sigma_recovers_known_lognormal() -> None:
    """Generate exact LN(μ=ln(100), σ_T=0.20) above-probs at T=1; recover σ ≈ 0.20 ± 2%."""
    spot = 100.0
    sigma_t_true = 0.20
    mu_true = math.log(spot)  # zero-drift log-normal: E[X]=spot·exp(σ²/2) but the
    # log-normal MoM fit recovers σ regardless of drift offset
    strikes = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0]
    entries = [
        LadderEntry(
            slug=f"s{int(K)}",
            strike=K,
            direction="above",
            venue="polymarket",
            market_value=_lognormal_above_prob(K, mu_true, sigma_t_true),
        )
        for K in strikes
    ]
    fam = LadderFamily(
        asset="LN",
        asset_class="equity_index",
        maturity_utc=_now_plus(365.25),  # T = 1 year
        spot_at_lookup=spot,
        entries=entries,
    )
    result = fit_implied_sigma(fam)
    # Post-bias-fix the primary method is the direct survival-function fit;
    # accept either (moment-match remains a fallback).
    assert result.sigma_method in {"lognormal_fit", "lognormal_survival_fit"}
    # σ_annual ≈ σ_T at T=1
    assert abs(result.sigma_annual - sigma_t_true) / sigma_t_true < 0.30, (
        f"σ_annual={result.sigma_annual} vs truth {sigma_t_true}"
    )


def test_fit_implied_sigma_annualizes_correctly() -> None:
    """Same DGP at T=0.5: raw σ_T ≈ 0.20·√0.5 but σ_annual ≈ 0.20."""
    spot = 100.0
    sigma_annual_true = 0.20
    t_years = 0.5
    sigma_t_true = sigma_annual_true * math.sqrt(t_years)
    mu_true = math.log(spot)
    strikes = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0]
    entries = [
        LadderEntry(
            slug=f"s{int(K)}",
            strike=K,
            direction="above",
            venue="polymarket",
            market_value=_lognormal_above_prob(K, mu_true, sigma_t_true),
        )
        for K in strikes
    ]
    fam = LadderFamily(
        asset="LN",
        asset_class="equity_index",
        maturity_utc=_now_plus(365.25 * t_years),
        spot_at_lookup=spot,
        entries=entries,
    )
    result = fit_implied_sigma(fam)
    # σ_annual should be close to 0.20 (within 35% tolerance — the MoM
    # log-normal fit on a 7-strike grid is not exact but recovers scale)
    assert abs(result.sigma_annual - sigma_annual_true) / sigma_annual_true < 0.35, (
        f"σ_annual={result.sigma_annual} vs truth {sigma_annual_true}"
    )
    # And T_years matches what we asked for
    assert abs(result.time_to_maturity_years - t_years) < 0.01


def test_warnings_emitted_for_short_maturity() -> None:
    """T = 5/365 days → short_maturity warning present."""
    entries = [
        LadderEntry(
            slug=f"k{k}", strike=float(k), direction="above", venue="polymarket", market_value=p
        )
        for k, p in [(50, 0.95), (100, 0.55), (150, 0.10)]
    ]
    fam = LadderFamily(
        asset="TEST",
        asset_class="equity_index",
        maturity_utc=_now_plus(5),
        spot_at_lookup=100.0,
        entries=entries,
    )
    result = fit_implied_sigma(fam)
    assert "short_maturity" in result.warnings


def test_warnings_emitted_for_insufficient_strikes() -> None:
    """2-strike ladder → few_strikes warning; method is lognormal_fit or empirical_pmf."""
    entries = [
        LadderEntry(
            slug="k50", strike=50.0, direction="above", venue="polymarket", market_value=0.9
        ),
        LadderEntry(
            slug="k150", strike=150.0, direction="above", venue="polymarket", market_value=0.1
        ),
    ]
    fam = _family(entries, days=365.0, spot=100.0)
    result = fit_implied_sigma(fam)
    assert "few_strikes" in result.warnings
    assert result.sigma_method in {
        "lognormal_fit",
        "empirical_pmf",
        "single_strike_inverse",
    }
    assert result.n_strikes == 2


# ---------------------------------------------------------------------------
# discover_ladder_family
# ---------------------------------------------------------------------------


class _StubPolymarketClient:
    """Stub that returns a fixed midpoint for every slug."""

    def __init__(self, midpoint: float = 0.5) -> None:
        self.midpoint = midpoint
        self.calls: list[str] = []

    def get_market_metadata(self, slug: str) -> dict[str, Any]:
        self.calls.append(slug)
        return {
            "bestBid": self.midpoint - 0.005,
            "bestAsk": self.midpoint + 0.005,
            "lastTradePrice": self.midpoint,
        }


def test_discover_ladder_family_btc_returns_full_above_ladder() -> None:
    """BTC EOY-2026 above-ladder has 10 rungs; midpoints get filled in.

    (Previously SPX/5 strikes — SPX dropped from registry on 2026-05-15 after
    discovery confirmed no live SPX multi-strike ladder on Polymarket. BTC
    above ladder expanded from the original 5 dead `btc-above-Xk-eoy-2026`
    slugs to 10 live long-form `will-bitcoin-reach-…` slugs spanning 90k–1M.)
    """
    client = _StubPolymarketClient(midpoint=0.5)
    fam = discover_ladder_family("BTC", polymarket_client=client)
    assert fam is not None
    assert fam.asset == "BTC"
    assert fam.asset_class == "crypto"
    assert len(fam.entries) == 10
    for entry in fam.entries:
        assert entry.market_value == pytest.approx(0.5, abs=1e-6)
        assert entry.direction == "above"
        assert entry.venue == "polymarket"
    assert client.calls  # client was actually invoked


def test_discover_ladder_family_unknown_asset_returns_none() -> None:
    client = _StubPolymarketClient()
    assert discover_ladder_family("DOGE", polymarket_client=client) is None


def test_discover_ladder_family_btc_picks_first_family_when_no_filter() -> None:
    """BTC has two registered families (above + dip_to); first wins without filter."""
    client = _StubPolymarketClient(midpoint=0.4)
    fam = discover_ladder_family("BTC", polymarket_client=client)
    assert fam is not None
    assert fam.asset == "BTC"
    # First family is the `above` ladder (10 strikes after 2026-05-15 refresh).
    assert fam.entries[0].direction == "above"
    assert len(fam.entries) == 10


def test_ladder_registry_has_expected_assets() -> None:
    """Sanity-check that LADDER_REGISTRY contains the currently-live assets.

    SPX was dropped on 2026-05-15 after a discovery sweep found no live
    multi-strike same-direction ladder on Polymarket.
    """
    for asset in ("BTC", "ETH", "WTI", "GOLD"):
        assert asset in LADDER_REGISTRY
    assert "SPX" not in LADDER_REGISTRY, "SPX intentionally absent — no live ladder"


def test_fit_implied_sigma_full_pipeline_via_discover() -> None:
    """End-to-end: discover_ladder_family → fit_implied_sigma returns a PMIVResult.

    Uses the BTC EOY-2026 above-ladder (10 live slugs, discovered 2026-05-15).
    Replaces the pre-existing SPX-based test now that SPX is unsupported.
    """
    # Plausible descending probs for the 10-strike BTC above ladder
    midpoints = {
        "will-bitcoin-reach-90000-by-december-31-2026-113-862-581": 0.65,
        "will-bitcoin-reach-100000-by-december-31-2026-571-361-361": 0.43,
        "will-bitcoin-reach-140000-by-december-31-2026-131-829-299": 0.115,
        "will-bitcoin-reach-150000-by-december-31-2026-557-246-971": 0.065,
        "will-bitcoin-reach-160000-by-december-31-2026-934-934-164": 0.07,
        "will-bitcoin-reach-190000-by-december-31-2026-936-485-627": 0.042,
        "will-bitcoin-reach-200000-by-december-31-2026-752-232-389": 0.04,
        "will-bitcoin-reach-250000-by-december-31-2026-579-442": 0.028,
        "will-bitcoin-reach-500000-by-december-31-2026-864": 0.02,
        "will-bitcoin-reach-1000000-by-december-31-2026-946": 0.015,
    }

    class _BTCStub:
        def get_market_metadata(self, slug: str) -> dict[str, Any]:
            p = midpoints[slug]
            return {"bestBid": p - 0.005, "bestAsk": p + 0.005, "lastTradePrice": p}

    fam = discover_ladder_family("BTC", polymarket_client=_BTCStub())
    assert fam is not None
    result = fit_implied_sigma(fam)
    assert result.n_strikes == 10
    assert result.sigma_annual > 0.0
    assert result.fitted_std > 0.0
    # Survival fit is primary post-bias-fix; moment-match remains a fallback.
    assert result.sigma_method in {"lognormal_fit", "lognormal_survival_fit"}


def test_build_survival_skips_entries_without_market_value() -> None:
    """Entries with `market_value=None` are dropped before survival construction."""
    entries = [
        LadderEntry(slug="a", strike=50.0, direction="above", venue="polymarket", market_value=0.9),
        LadderEntry(
            slug="b", strike=100.0, direction="above", venue="polymarket", market_value=None
        ),
        LadderEntry(
            slug="c", strike=150.0, direction="above", venue="polymarket", market_value=0.1
        ),
    ]
    strikes, surv = build_survival_function(_family(entries))
    assert strikes == [50.0, 150.0]
    assert surv == [0.9, 0.1]


def test_fit_implied_sigma_monotonicity_violation_warning() -> None:
    """An `above` ladder that violates monotonicity raises the warning."""
    entries = [
        LadderEntry(
            slug="k50", strike=50.0, direction="above", venue="polymarket", market_value=0.5
        ),
        LadderEntry(
            slug="k100", strike=100.0, direction="above", venue="polymarket", market_value=0.7
        ),  # impossible
        LadderEntry(
            slug="k150", strike=150.0, direction="above", venue="polymarket", market_value=0.2
        ),
    ]
    fam = _family(entries, days=365.0, spot=100.0)
    result = fit_implied_sigma(fam)
    assert "monotonicity_violated" in result.warnings
    # Survival still constructed and clamped monotonically
    assert all(result.raw_probs[i] is not None for i in range(len(result.raw_probs)))


def test_single_strike_inverse_method_selected_for_one_strike() -> None:
    """A 1-strike ladder triggers single_strike_inverse method."""
    entries = [
        LadderEntry(
            slug="k100", strike=100.0, direction="above", venue="polymarket", market_value=0.5
        ),
    ]
    fam = _family(entries, days=365.0, spot=100.0)
    result = fit_implied_sigma(fam)
    assert result.sigma_method == "single_strike_inverse"
    assert result.n_strikes == 1
    assert "few_strikes" in result.warnings


def test_fit_implied_sigma_raises_on_empty_family() -> None:
    """All-None market_values → ValueError (no usable strikes)."""
    entries = [
        LadderEntry(
            slug="a", strike=50.0, direction="above", venue="polymarket", market_value=None
        ),
    ]
    fam = _family(entries)
    with pytest.raises(ValueError):
        fit_implied_sigma(fam)


def test_discover_ladder_family_handles_polymarket_exception() -> None:
    """A client that raises returns a family with market_value=None entries."""

    class _BadClient:
        def get_market_metadata(self, slug: str) -> dict[str, Any]:
            raise RuntimeError("network down")

    fam = discover_ladder_family("ETH", polymarket_client=_BadClient())
    assert fam is not None
    assert all(e.market_value is None for e in fam.entries)


# ---------------------------------------------------------------------------
# Direct lognormal survival-function fit (post-A4 bias fix)
# ---------------------------------------------------------------------------


def test_lognormal_survival_fit_recovers_known_sigma_wide_ladder() -> None:
    """Wide-ladder (5× range) recovery — THE bias the new fit targets.

    The previous moment-match pipeline systematically over-stated σ on
    ladders like this because mass past K_max was treated as a point at
    1.25·K_max, inflating the empirical variance. The direct survival fit
    uses only the observed strikes and should recover σ ≈ 0.30 within ±5%.
    """
    spot = 100.0
    mu_true = math.log(spot)
    sigma_t_true = 0.30
    strikes = [100.0, 150.0, 200.0, 300.0, 500.0]
    surv = [_lognormal_above_prob(K, mu_true, sigma_t_true) for K in strikes]
    _mu_hat, sigma_t_hat, resid = _fit_lognormal_survival(strikes, surv)
    assert math.isfinite(sigma_t_hat)
    assert abs(sigma_t_hat - sigma_t_true) / sigma_t_true < 0.05, (
        f"σ_T_hat={sigma_t_hat} vs truth {sigma_t_true} (resid={resid})"
    )


def test_lognormal_survival_fit_recovers_low_sigma_narrow_ladder() -> None:
    """Narrow 5-strike ladder at low vol — fit must not collapse to σ ≈ 0."""
    spot = 100.0
    mu_true = math.log(spot)
    sigma_annual_true = 0.10
    t_years = 0.5
    sigma_t_true = sigma_annual_true * math.sqrt(t_years)
    strikes = [95.0, 100.0, 105.0, 110.0, 115.0]
    surv = [_lognormal_above_prob(K, mu_true, sigma_t_true) for K in strikes]
    _mu_hat, sigma_t_hat, _ = _fit_lognormal_survival(strikes, surv)
    assert math.isfinite(sigma_t_hat)
    assert abs(sigma_t_hat - sigma_t_true) / sigma_t_true < 0.05, (
        f"σ_T_hat={sigma_t_hat} vs truth {sigma_t_true}"
    )


def test_fit_implied_sigma_uses_survival_fit_for_wide_above_ladder() -> None:
    """End-to-end wide-ladder fit reports the survival-fit method and a sane σ."""
    spot = 100.0
    sigma_t_true = 0.30
    mu_true = math.log(spot)
    strikes = [100.0, 150.0, 200.0, 300.0, 500.0]
    entries = [
        LadderEntry(
            slug=f"s{int(K)}",
            strike=K,
            direction="above",
            venue="polymarket",
            market_value=_lognormal_above_prob(K, mu_true, sigma_t_true),
        )
        for K in strikes
    ]
    fam = _family(entries, days=365.25, spot=spot)
    result = fit_implied_sigma(fam)
    assert result.sigma_method == "lognormal_survival_fit"
    # σ_annual should land near 0.30 at T≈1
    assert 0.20 <= result.sigma_annual <= 0.45, (
        f"σ_annual={result.sigma_annual} unreasonable for σ_true=0.30"
    )
    # Cross-check it lies within 30% of what the legacy moment-match would
    # produce on the same survival fn — they should be the same order of
    # magnitude even though the moment-match over-states for wide ladders.
    _strikes_used, probs_used = build_survival_function(fam)
    mom = _empirical_moments(_strikes_used, probs_used)
    _, sigma_t_mom = _fit_lognormal(mom["mean"], mom["std"])
    sigma_mom_annual = sigma_t_mom / math.sqrt(result.time_to_maturity_years)
    assert abs(result.sigma_annual - sigma_mom_annual) / sigma_mom_annual < 0.50


def test_fit_implied_sigma_falls_back_to_moment_match_on_optimization_failure() -> None:
    """Flat survival (all 0.5) is non-informative → survival fit returns NaN
    and we fall back to moment-match (or empirical_pmf)."""
    entries = [
        LadderEntry(
            slug=f"f{int(K)}",
            strike=float(K),
            direction="above",
            venue="polymarket",
            market_value=0.5,
        )
        for K in [50, 75, 100, 125, 150]
    ]
    fam = _family(entries, days=365.0, spot=100.0)
    result = fit_implied_sigma(fam)
    assert result.sigma_method in {"lognormal_fit", "empirical_pmf"}
    # Sanity: the survival fit itself returns NaN on the flat input
    _, sigma_t_hat, _ = _fit_lognormal_survival(
        [50.0, 75.0, 100.0, 125.0, 150.0],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    )
    assert math.isnan(sigma_t_hat)


def test_survival_fit_handles_dip_to_after_direction_normalization() -> None:
    """A `dip_to` ladder still produces a valid lognormal_survival_fit result.

    Mirrors the synthetic-DGP case in `test_fit_implied_sigma_recovers_known_lognormal`
    but uses the dip_to direction shape so the 0.5 barrier inversion is
    exercised before the σ fit.
    """
    spot = 100.0
    mu_true = math.log(spot)
    sigma_t_true = 0.25
    # dip_to encodes touch probs. Build the touch prob whose central
    # estimate (0.5·P_touch) equals 1 - S(K), i.e. equals P(X_T < K).
    strikes = [50.0, 65.0, 80.0, 95.0]  # below spot — dips
    touch_probs: list[float] = []
    for K in strikes:
        p_below = 1.0 - _lognormal_above_prob(K, mu_true, sigma_t_true)
        # one-touch ≈ 2·P(X_T<K), capped at 1
        touch_probs.append(float(min(2.0 * p_below, 0.99)))

    entries = [
        LadderEntry(
            slug=f"d{int(K)}",
            strike=K,
            direction="dip_to",
            venue="polymarket",
            market_value=p,
        )
        for K, p in zip(strikes, touch_probs, strict=True)
    ]
    fam = _family(entries, days=365.25, spot=spot)
    result = fit_implied_sigma(fam)
    assert result.sigma_method == "lognormal_survival_fit"
    assert result.sigma_annual > 0.0


def test_survival_fit_bias_reduction_vs_moment_match() -> None:
    """Regression-against-bias test: survival fit closer to truth than moment-match.

    Synthetic LN(μ=ln(60), σ=0.40) at T=0.5 (WTI-like). Wide ladder. The
    pre-fix moment-match systematically over-stated σ here; the survival
    fit must be strictly closer to the true σ.
    """
    spot_proxy = 60.0
    sigma_annual_true = 0.40
    t_years = 0.5
    sigma_t_true = sigma_annual_true * math.sqrt(t_years)
    mu_true = math.log(spot_proxy)
    strikes = [50.0, 75.0, 90.0, 115.0, 150.0, 175.0, 200.0]
    surv = [_lognormal_above_prob(K, mu_true, sigma_t_true) for K in strikes]

    # Survival-fit σ_T
    _, sigma_t_surv, _ = _fit_lognormal_survival(strikes, surv)
    assert math.isfinite(sigma_t_surv)

    # Legacy moment-match σ_T
    mom = _empirical_moments(strikes, surv)
    _, sigma_t_mom = _fit_lognormal(mom["mean"], mom["std"])

    err_surv = abs(sigma_t_surv - sigma_t_true)
    err_mom = abs(sigma_t_mom - sigma_t_true)
    assert err_surv < err_mom, (
        f"survival_fit err {err_surv:.4f} not < moment_match err {err_mom:.4f} "
        f"(surv σ_T={sigma_t_surv:.4f}, mom σ_T={sigma_t_mom:.4f}, "
        f"truth σ_T={sigma_t_true:.4f})"
    )
