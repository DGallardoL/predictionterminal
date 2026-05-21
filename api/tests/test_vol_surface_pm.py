"""Tests for the PM-derived volatility surface module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import vol_surface_pm
from pfm.cache_utils import get_cache
from pfm.vol_surface_pm import (
    KNOWN_LADDERS,
    compare_pm_vs_options_iv,
    extract_implied_distribution,
    router,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    get_cache("vol_surface_pm").clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _market(prob: float) -> dict[str, Any]:
    return {
        "bestBid": max(0.0, prob - 0.005),
        "bestAsk": min(1.0, prob + 0.005),
        "lastTradePrice": prob,
    }


# ---------------------------------------------------------------------------
# extract_implied_distribution
# ---------------------------------------------------------------------------


# Slug map for the BTC EOY-2026 above-ladder, refreshed 2026-05-15.
# The original `btc-above-Xk-eoy-2026` slugs are dead on Polymarket; replaced
# with the long-form `will-bitcoin-reach-…-by-december-31-2026` markets that
# KNOWN_LADDERS["btc-price-eoy-2026"] now points at (10 strikes).
_BTC_LADDER_SLUGS: dict[float, str] = {
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


def test_extract_distribution_from_btc_ladder() -> None:
    """Survival fn across the live 10-strike BTC EOY-2026 above-ladder → coherent PMF."""
    # Monotone-descending survival across strikes 90k..1M
    survival = [0.65, 0.43, 0.115, 0.065, 0.06, 0.042, 0.04, 0.028, 0.02, 0.015]
    strikes = list(_BTC_LADDER_SLUGS.keys())
    overrides = {_BTC_LADDER_SLUGS[k]: _market(p) for k, p in zip(strikes, survival, strict=True)}
    out = extract_implied_distribution(
        "btc-price-eoy-2026", market_value=110_000.0, http=MagicMock(), overrides=overrides
    )
    assert out["n_strikes"] == 10
    assert out["strikes"] == strikes
    # After monotone-enforcement the survival stays the same when raw is already monotone.
    assert out["implied_probs"] == survival
    assert out["fitted_std"] > 0.0
    assert out["fitted_distribution_type"] == "lognormal"
    assert out["lognormal_sigma"] > 0.0


def test_fitted_mean_within_tolerance_of_market_value() -> None:
    """A ladder anchored at spot=110k should produce a fitted mean within 150% of 110k.

    The current live BTC ladder skews far right (highest rung 1M) — the
    log-normal moment-match recovers a mean that pulls toward the upper tail
    by construction. We relax the tolerance vs. the pre-refresh test to
    reflect the wider strike grid (90k..1M instead of 50k..200k).
    """
    survival = [0.60, 0.40, 0.10, 0.07, 0.06, 0.04, 0.035, 0.025, 0.018, 0.012]
    strikes = list(_BTC_LADDER_SLUGS.keys())
    overrides = {_BTC_LADDER_SLUGS[k]: _market(p) for k, p in zip(strikes, survival, strict=True)}
    out = extract_implied_distribution(
        "btc-price-eoy-2026", market_value=110_000.0, http=MagicMock(), overrides=overrides
    )
    # Wider tolerance reflects the right-skewed 10-strike grid (top rung 1M).
    assert abs(out["fitted_mean"] - 110_000.0) / 110_000.0 < 1.5, (
        f"fitted mean {out['fitted_mean']} too far from 110k"
    )


def test_extract_distribution_enforces_monotonicity() -> None:
    """A non-monotone ladder must still produce a coherent CDF."""
    # Bumpy ladder: 200k probability higher than 160k breaks monotonicity.
    raw = {
        90_000.0: 0.65,
        100_000.0: 0.43,
        140_000.0: 0.115,
        150_000.0: 0.10,
        160_000.0: 0.06,
        190_000.0: 0.04,
        200_000.0: 0.30,  # would violate (higher than 160k/190k)
        250_000.0: 0.028,
        500_000.0: 0.02,
        1_000_000.0: 0.015,
    }
    overrides = {_BTC_LADDER_SLUGS[k]: _market(p) for k, p in raw.items()}
    out = extract_implied_distribution("btc-price-eoy-2026", http=MagicMock(), overrides=overrides)
    probs = out["implied_probs"]
    assert all(probs[i] >= probs[i + 1] - 1e-9 for i in range(len(probs) - 1)), probs


def test_extract_distribution_unknown_pattern() -> None:
    with pytest.raises(KeyError):
        extract_implied_distribution("unknown-foo-2099", http=MagicMock(), overrides={})


def test_extract_distribution_too_few_rungs_raises() -> None:
    overrides = {_BTC_LADDER_SLUGS[90_000.0]: _market(0.5)}
    with pytest.raises(ValueError):
        extract_implied_distribution("btc-price-eoy-2026", http=MagicMock(), overrides=overrides)


# ---------------------------------------------------------------------------
# compare_pm_vs_options_iv
# ---------------------------------------------------------------------------


def test_compare_pm_vs_options_iv_pm_richer() -> None:
    pm_data = {"lognormal_sigma": 1.0, "strikes": [1, 2], "implied_probs": [0.5, 0.1]}
    out = compare_pm_vs_options_iv(
        "BTC", current_price=100.0, pm_strikes_data=pm_data, options_iv_annual=0.50
    )
    assert out["direction"] == "pm_richer"
    assert out["spread_sigma"] > 0


def test_compare_pm_vs_options_iv_options_richer() -> None:
    pm_data = {"lognormal_sigma": 0.10, "strikes": [], "implied_probs": []}
    out = compare_pm_vs_options_iv(
        "BTC", current_price=100.0, pm_strikes_data=pm_data, options_iv_annual=0.80
    )
    assert out["direction"] == "options_richer"


def test_compare_pm_vs_options_iv_default_iv_fallback() -> None:
    pm_data = {"lognormal_sigma": 0.40, "strikes": [], "implied_probs": []}
    out = compare_pm_vs_options_iv("BTC", current_price=100.0, pm_strikes_data=pm_data)
    # Default fallback IV for BTC = 0.65 * 1.2 = 0.78
    assert out["options_iv_annual"] == pytest.approx(0.78, abs=0.001)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_get_pm_distribution_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    survival = [0.65, 0.43, 0.115, 0.065, 0.06, 0.042, 0.04, 0.028, 0.02, 0.015]
    probs = {
        _BTC_LADDER_SLUGS[k]: p for k, p in zip(_BTC_LADDER_SLUGS.keys(), survival, strict=True)
    }

    def _fake(http, gamma_url, slug, **_kwargs):
        return _market(probs[slug])

    monkeypatch.setattr(vol_surface_pm, "fetch_gamma_market", _fake)

    r = client.get("/vol-surface/pm/btc-price-eoy-2026")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_strikes"] == 10
    assert body["fitted_distribution_type"] == "lognormal"


def test_get_pm_distribution_endpoint_unknown_pattern(client: TestClient) -> None:
    r = client.get("/vol-surface/pm/totally-fake-pattern")
    assert r.status_code == 404


def test_compare_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    survival = [0.65, 0.43, 0.115, 0.065, 0.06, 0.042, 0.04, 0.028, 0.02, 0.015]
    probs = {
        _BTC_LADDER_SLUGS[k]: p for k, p in zip(_BTC_LADDER_SLUGS.keys(), survival, strict=True)
    }
    monkeypatch.setattr(
        vol_surface_pm,
        "fetch_gamma_market",
        lambda http, gamma_url, slug, **_k: _market(probs[slug]),
    )
    r = client.get(
        "/vol-surface/compare",
        params={
            "ticker": "BTC",
            "pm_pattern": "btc-price-eoy-2026",
            "current_price": 110_000.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "BTC"
    assert body["direction"] in {"pm_richer", "options_richer", "flat"}
    assert len(body["pm_strikes"]) == 10


def test_known_ladders_have_min_2_rungs() -> None:
    for pattern, ladder in KNOWN_LADDERS.items():
        assert len(ladder) >= 2, f"{pattern} has fewer than 2 rungs"
