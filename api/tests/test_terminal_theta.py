"""Tests for ``pfm.terminal_theta`` — /terminal/theta/{slug} and /cluster.

External IO (Polymarket fetch + gamma metadata) is monkeypatched on the
module under test so the suite is fully offline.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_theta
from pfm.factors import FactorConfig
from pfm.terminal_theta import (
    EPSILON,
    aggregate_cluster,
    annualised_sigma_logit,
    compute_market_theta,
    empirical_theta,
    get_factors_dep,
    get_polymarket_client,
    parse_resolution_period,
    router,
    theoretical_theta_brownian_bridge,
)

# --- helpers ---------------------------------------------------------------


class _FakePoly:
    """Sentinel — :func:`fetch_factor_history` is monkey-patched, so this
    object is just forwarded as the client argument."""


def _make_prices(
    *,
    days: int = 90,
    base: float = 0.40,
    drift: float = 0.0,
    sigma: float = 0.04,
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic factor-history DataFrame matching ``fetch_factor_history``."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2026-05-01", tz="UTC").normalize()
    idx = pd.date_range(end=end, periods=days, freq="D", tz="UTC")
    idx.name = "date"
    # Random walk in logit space, drifted, then squashed back to a probability.
    increments = rng.normal(drift, sigma, days)
    logits = np.cumsum(increments) + math.log(base / (1.0 - base))
    p = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(p, 0.05, 0.95)
    return pd.DataFrame({"price": p}, index=idx)


# --- 1. Single-market empirical theta --------------------------------------


def test_empirical_theta_recovers_known_median_abs_delta() -> None:
    """A series whose daily diffs have known median |Δp| should recover it."""
    # Build a deterministic price path with known absolute jumps.
    idx = pd.date_range("2026-01-01", periods=11, freq="D", tz="UTC")
    # Differences (in order): +0.05, -0.03, +0.10, -0.02, +0.04, -0.05, +0.01, -0.06, +0.03, -0.04
    # |Δ| = [0.05, 0.03, 0.10, 0.02, 0.04, 0.05, 0.01, 0.06, 0.03, 0.04] → median = 0.04
    diffs = [0.05, -0.03, 0.10, -0.02, 0.04, -0.05, 0.01, -0.06, 0.03, -0.04]
    levels = np.cumsum([0.50, *diffs])
    prices = pd.Series(levels, index=idx)

    theta = empirical_theta(prices)
    assert math.isclose(theta, 0.04, abs_tol=1e-9)


def test_single_market_endpoint_returns_well_formed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hitting /terminal/theta/{slug} returns the documented schema and
    sensible numeric ranges for a synthetic price series."""
    df = _make_prices(days=120, base=0.42, sigma=0.05, seed=11)

    def fake_fetch(_client, slug: str, start=None, end=None):
        assert slug == "putin-out-before-2027"
        return df

    end_meta = (datetime.now(UTC) + timedelta(days=240)).isoformat()

    def fake_gamma(slug: str, _client):
        return {"endDate": end_meta}

    monkeypatch.setattr(terminal_theta, "fetch_factor_history", fake_fetch)
    monkeypatch.setattr(terminal_theta, "_fetch_gamma_metadata", fake_gamma)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_polymarket_client] = _FakePoly

    client = TestClient(app)
    r = client.get("/terminal/theta/putin-out-before-2027?days=30")
    assert r.status_code == 200, r.text
    body = r.json()

    # Schema: every documented top-level key must exist.
    expected_keys = {
        "slug",
        "current_p",
        "days_to_resolve",
        "empirical_theta_per_day",
        "theoretical_theta_per_day",
        "theta_acceleration",
        "historical_decay_curve",
        "interpretation",
    }
    assert expected_keys.issubset(body.keys())

    assert body["slug"] == "putin-out-before-2027"
    assert 0.0 < body["current_p"] < 1.0
    assert body["days_to_resolve"] > 0.0

    # Sanity: empirical theta is non-negative and small (median |Δp| of a
    # bounded probability series is at most 1).
    assert 0.0 <= body["empirical_theta_per_day"] <= 1.0
    # Theoretical theta is non-negative and bounded by p(1-p) ≤ 0.25.
    assert 0.0 <= body["theoretical_theta_per_day"] <= 0.25
    # Decay curve is non-empty.
    assert isinstance(body["historical_decay_curve"], list)
    assert len(body["historical_decay_curve"]) > 0
    assert {"days_to_res", "abs_delta"} <= set(body["historical_decay_curve"][0].keys())
    # Interpretation mentions either 'pp' or 'resolved'.
    assert "pp" in body["interpretation"] or "resolved" in body["interpretation"]


# --- 2. Single-market theoretical theta + interpretation -------------------


def test_theoretical_theta_brownian_bridge_matches_closed_form() -> None:
    """The closed-form half-normal step matches the implementation.

    For p=0.5, σ_annual=1.0 ⇒ σ_d = 1/√252, expected |Δp| in prob-space:

        p(1-p) · σ_d · √(2/π)
    """
    p = 0.5
    sigma_annual = 1.0
    sigma_d = 1.0 / math.sqrt(252.0)
    expected = p * (1.0 - p) * sigma_d * math.sqrt(2.0 / math.pi)

    got = theoretical_theta_brownian_bridge(p, days_to_resolve=30.0, sigma_annual=sigma_annual)
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_theta_card_interpretation_mentions_resolution_horizon() -> None:
    """Interpretation contains the resolution date and a per-day pp move."""
    df = _make_prices(days=80, base=0.55, sigma=0.04, seed=3)
    end_date = (datetime.now(UTC) + timedelta(days=45)).date()
    card = compute_market_theta(
        slug="test-slug",
        prices=df["price"],
        end_date=end_date,
        days_lookback=30,
        now=datetime.now(UTC),
    )
    assert card.days_to_resolve > 0.0
    # Theoretical theta non-negative, finite.
    assert math.isfinite(card.theoretical_theta_per_day)
    assert card.theoretical_theta_per_day >= 0.0
    # Acceleration is finite.
    assert math.isfinite(card.theta_acceleration)
    assert "pp" in card.interpretation
    assert end_date.isoformat() in card.interpretation


def test_annualised_sigma_logit_finite_and_positive() -> None:
    """σ from a noisy synthetic walk is positive and finite."""
    df = _make_prices(days=100, sigma=0.05, seed=42)
    sigma = annualised_sigma_logit(df["price"])
    assert math.isfinite(sigma) and sigma > 0.0


# --- 3. Cluster aggregation ------------------------------------------------


def test_cluster_endpoint_aggregates_across_multiple_markets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cluster endpoint should pull theme-matching markets from
    ``factors.yml``, fetch each, and report aggregate statistics."""

    factors = {
        "btc_a": FactorConfig(
            id="btc_a",
            name="BTC market A",
            slug="btc-strike-a",
            source="polymarket",
            description="x",
            theme="crypto",
        ),
        "btc_b": FactorConfig(
            id="btc_b",
            name="BTC market B",
            slug="btc-strike-b",
            source="polymarket",
            description="x",
            theme="crypto",
        ),
        "btc_c": FactorConfig(
            id="btc_c",
            name="BTC market C",
            slug="btc-strike-c",
            source="polymarket",
            description="x",
            theme="crypto",
        ),
        "macro_one": FactorConfig(
            id="macro_one",
            name="Macro 1",
            slug="macro-1",
            source="polymarket",
            description="x",
            theme="macro",
        ),
    }

    bank = {
        "btc-strike-a": _make_prices(days=80, base=0.30, sigma=0.04, seed=1),
        "btc-strike-b": _make_prices(days=80, base=0.55, sigma=0.06, seed=2),
        "btc-strike-c": _make_prices(days=80, base=0.45, sigma=0.05, seed=3),
    }

    def fake_fetch(_client, slug: str, start=None, end=None):
        return bank[slug]

    monkeypatch.setattr(terminal_theta, "fetch_factor_history", fake_fetch)
    # Cluster endpoint hits gamma only when resolution_period is set; we
    # filter only by theme here, so the gamma call is skipped.

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_polymarket_client] = _FakePoly
    app.dependency_overrides[get_factors_dep] = lambda: factors

    client = TestClient(app)
    r = client.get("/terminal/theta/cluster?theme=crypto&days=30")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["theme"] == "crypto"
    assert body["n_markets"] == 3, body
    assert math.isfinite(body["mean_theta"])
    assert math.isfinite(body["median_theta"])
    # p10 ≤ median ≤ p90.
    assert body["p10_theta"] <= body["median_theta"] <= body["p90_theta"]
    # The macro factor must NOT appear in the aggregate.
    assert "macro" not in body["interpretation"]
    # Curve is bucketed and finite.
    assert isinstance(body["theta_curve"], list)
    if body["theta_curve"]:
        row = body["theta_curve"][0]
        assert {"days_to_res", "n_markets", "median_abs_delta"} <= set(row.keys())


def test_aggregate_cluster_handles_empty_card_list() -> None:
    """An empty cluster yields nan stats, no curve, and zero markets."""
    agg = aggregate_cluster([])
    assert agg["n_markets"] == 0
    assert math.isnan(agg["mean_theta"])
    assert math.isnan(agg["median_theta"])
    assert agg["theta_curve"] == []


def test_parse_resolution_period_quarter_strings() -> None:
    """``2026Q3`` parses to (Jul 1, Sep 30); junk returns None."""
    span = parse_resolution_period("2026Q3")
    assert span is not None
    qstart, qend = span
    assert qstart.isoformat() == "2026-07-01"
    assert qend.isoformat() == "2026-09-30"

    assert parse_resolution_period(None) is None
    assert parse_resolution_period("not-a-quarter") is None


# --- 4. Edge: resolved market (days_to_resolve = 0) ------------------------


def test_resolved_market_has_zero_theoretical_theta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A market whose end date is in the past must report
    ``days_to_resolve == 0`` and ``theoretical_theta_per_day == 0``."""
    df = _make_prices(days=60, base=0.10, sigma=0.03, seed=5)

    def fake_fetch(_client, slug: str, start=None, end=None):
        return df

    # End date is 5 days *in the past*.
    past_end = (datetime.now(UTC) - timedelta(days=5)).isoformat()

    def fake_gamma(slug: str, _client):
        return {"endDate": past_end}

    monkeypatch.setattr(terminal_theta, "fetch_factor_history", fake_fetch)
    monkeypatch.setattr(terminal_theta, "_fetch_gamma_metadata", fake_gamma)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_polymarket_client] = _FakePoly

    client = TestClient(app)
    r = client.get("/terminal/theta/some-resolved-slug?days=20")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["days_to_resolve"] == 0.0
    assert body["theoretical_theta_per_day"] == 0.0
    assert body["theta_acceleration"] == 0.0
    assert "resolved" in body["interpretation"].lower()
    # Empirical theta still defined (we have plenty of history).
    assert math.isfinite(body["empirical_theta_per_day"])


def test_epsilon_constant_matches_module_default() -> None:
    """Sanity check that the module-level ε is the documented 0.01."""
    assert EPSILON == 0.01
