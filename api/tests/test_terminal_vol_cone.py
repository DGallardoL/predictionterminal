"""Tests for the realized-volatility cone Terminal endpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.terminal_vol_cone as vc_mod
from pfm.terminal_vol_cone import (
    ANNUALISATION,
    HORIZONS,
    VolConeResult,
    _get_polymarket_client_dep,
    compute_vol_cone,
    router,
)


def _synthetic_prices(n: int = 240, seed: int = 7) -> pd.Series:
    """Bounded oscillating probability series — never saturates the clip."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    base = 0.50 + 0.20 * np.sin(np.linspace(0, 6 * np.pi, n))
    noise = rng.normal(0, 0.01, n)
    prices = pd.Series((base + noise).clip(0.05, 0.95), index=idx, name="price")
    return prices


# ---------- 1. Pure compute layer ------------------------------------------


def test_compute_vol_cone_returns_well_formed_payload() -> None:
    """Bands and current_vol arrays are aligned to HORIZONS and contain finite values."""
    prices = _synthetic_prices(n=240)

    result = compute_vol_cone(prices)

    assert isinstance(result, VolConeResult)
    assert result.horizons == list(HORIZONS)

    # Each band has one entry per horizon.
    for key in ("p10", "p25", "p50", "p75", "p90"):
        assert key in result.percentile_bands
        assert len(result.percentile_bands[key]) == len(HORIZONS)

    assert len(result.current_vol) == len(HORIZONS)
    assert len(result.current_percentile) == len(HORIZONS)

    # Band ordering must hold per horizon: p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90.
    for i in range(len(HORIZONS)):
        p10 = result.percentile_bands["p10"][i]
        p25 = result.percentile_bands["p25"][i]
        p50 = result.percentile_bands["p50"][i]
        p75 = result.percentile_bands["p75"][i]
        p90 = result.percentile_bands["p90"][i]
        assert p10 <= p25 <= p50 <= p75 <= p90, f"band order violated at horizon idx {i}"

    # Annualised σ should be strictly positive on synthetic noisy data.
    assert all(v > 0 for v in result.current_vol)
    # Percentiles are in [0, 100].
    assert all(0.0 <= p <= 100.0 for p in result.current_percentile)


# ---------- 2. Annualisation correctness -----------------------------------


def test_compute_vol_cone_annualises_with_sqrt_252() -> None:
    """A constant-σ Δlogit series must scale by √252 in the cone output."""
    # Construct prices whose Δlogit returns are exactly ±0.05 alternating —
    # std of ±0.05 is 0.05 (population)/0.05 (sample-ish on a long sequence).
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    # Build logits as a triangle wave so diffs alternate ±0.05.
    logits = np.cumsum(np.tile([0.05, -0.05], n // 2 + 1)[:n])
    # Recover prices = sigmoid(logit), bounded away from 0/1.
    probs = 1.0 / (1.0 + np.exp(-logits)) * 0.6 + 0.2  # squashed into [0.2, 0.8]
    prices = pd.Series(probs, index=idx, name="price")

    result = compute_vol_cone(prices, horizons=(30,))

    # σ_30 should be ~ std(Δlogit) * √252. Use a generous tolerance — the
    # squashing into [0.2, 0.8] perturbs the diffs slightly via the clip.
    rolling_std = (
        pd.Series(probs)
        .pipe(lambda s: np.log((s.clip(0.01, 0.99)) / (1 - s.clip(0.01, 0.99))).diff())
        .rolling(30)
        .std(ddof=1)
        .dropna()
        .iloc[-1]
    )
    expected = float(rolling_std * ANNUALISATION)
    assert result.current_vol[0] == pytest.approx(expected, rel=1e-6)


# ---------- 3. End-to-end through FastAPI ----------------------------------


def test_endpoint_returns_cone_with_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The router serves the cone shape over HTTP using a patched data layer."""
    prices_df = _synthetic_prices(n=240).to_frame()

    def _fake_fetch(_client, _slug: str, start=None, end=None):
        return prices_df

    # Patch the module-level reference used by the route handler.
    monkeypatch.setattr(vc_mod, "fetch_factor_history", _fake_fetch)

    app = FastAPI()
    app.include_router(router)
    # The DI dependency normally pulls a real PolymarketClient off app.state;
    # for tests we override it with a sentinel — _fake_fetch ignores it anyway.
    app.dependency_overrides[_get_polymarket_client_dep] = object

    with TestClient(app) as client:
        r = client.get("/terminal/vol-cone/some-slug")

    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["horizons"] == list(HORIZONS)
    assert set(payload["percentile_bands"]) == {"p10", "p25", "p50", "p75", "p90"}
    assert len(payload["current_vol"]) == len(HORIZONS)
    assert len(payload["current_percentile"]) == len(HORIZONS)
    # Sanity: at the longest horizon the current σ should sit in [p10/2, p90*2].
    p10 = payload["percentile_bands"]["p10"][-1]
    p90 = payload["percentile_bands"]["p90"][-1]
    cur = payload["current_vol"][-1]
    assert p10 / 2 <= cur <= p90 * 2 + 1e-9


# ---------- 4. Boundary / error coverage -----------------------------------


def test_compute_vol_cone_raises_on_too_few_observations() -> None:
    """A 1-element series cannot form even a single Δlogit return."""
    s = pd.Series([0.5], index=pd.date_range("2025-01-01", periods=1, tz="UTC"))
    with pytest.raises(ValueError):
        compute_vol_cone(s)


def test_compute_vol_cone_rejects_non_positive_horizons() -> None:
    """horizon=0 or negative is a programming error."""
    prices = _synthetic_prices(n=120)
    with pytest.raises(ValueError):
        compute_vol_cone(prices, horizons=(0,))
    with pytest.raises(ValueError):
        compute_vol_cone(prices, horizons=(7, -1))


def test_endpoint_404_on_empty_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Polymarket returning no rows → 404 with explanatory detail."""

    def _empty(_client, _slug, start=None, end=None):
        return pd.DataFrame(columns=["price"])

    monkeypatch.setattr(vc_mod, "fetch_factor_history", _empty)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_get_polymarket_client_dep] = object
    with TestClient(app) as client:
        r = client.get("/terminal/vol-cone/no-data-slug")
    assert r.status_code == 404
    assert "no price history" in r.json()["detail"]


def test_endpoint_502_on_upstream_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raised exception inside the fetcher is wrapped into 502."""

    def _boom(*_a, **_kw):
        raise RuntimeError("polymarket down")

    monkeypatch.setattr(vc_mod, "fetch_factor_history", _boom)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_get_polymarket_client_dep] = object
    with TestClient(app) as client:
        r = client.get("/terminal/vol-cone/any-slug")
    assert r.status_code == 502
    assert "polymarket fetch failed" in r.json()["detail"]


def test_endpoint_422_when_history_is_too_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """Need ≥ max(HORIZONS)+5 = 95 bars; 50 bars must be rejected as 422."""
    short = _synthetic_prices(n=50).to_frame()
    monkeypatch.setattr(
        vc_mod,
        "fetch_factor_history",
        lambda *_a, **_kw: short,
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_get_polymarket_client_dep] = object
    with TestClient(app) as client:
        r = client.get("/terminal/vol-cone/short-slug")
    assert r.status_code == 422
    assert "insufficient history" in r.json()["detail"]
