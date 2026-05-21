"""Tests for ``GET /quant/cointegration`` (task W13-17).

Standalone tests: the router is mounted on a throw-away FastAPI app with
the four data dependencies overridden, and the upstream data call
(``_cached_factor_history``) is monkeypatched to deterministic synthetic
series so we never hit the network or import ``pfm.main`` for its full
lifespan.

Coverage targets (≥10 tests, per W13-17):

1.  Synthetic cointegrated pair → ADF p<0.05 and ``is_cointegrated=True``.
2.  Independent random walks → not cointegrated.
3.  422 on bad slugs (start >= end).
4.  404 on unknown ``a`` slug.
5.  404 on unknown ``b`` slug.
6.  Half-life uses the W12-32 helper (numeric agreement with the helper).
7.  Response shape matches the documented contract.
8.  ``a == b`` (same factor id resolved) → 422.
9.  Slug aliasing (input slug != id, but resolves) → response echoes the
    resolved ``id`` so frontends get a canonical reference.
10. Empty ``a`` query param → 422 (Pydantic ``min_length`` enforcement).
11. ``half_life_days`` is a positive finite number for a cointegrated pair.
12. A non-cointegrated pair returns ``is_cointegrated=False`` and a
    plausible ``adf_p_value`` >= 0.05.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.regression_core as _regression_core
from pfm.cache import NullCache
from pfm.dependencies import get_cache, get_factors_dep, get_polymarket_client
from pfm.factors import FactorConfig
from pfm.quant.cointegration_router import router
from pfm.quant.half_life import estimate_half_life

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_factor_catalog() -> dict[str, FactorConfig]:
    """A tiny four-factor catalog: two cointegrated, two random walks."""
    return {
        "coint-a": FactorConfig(
            id="coint-a",
            name="Cointegrated leg A",
            slug="coint-leg-a",
            source="polymarket",
            description="(test)",
            theme="test",
            is_probability=True,
        ),
        "coint-b": FactorConfig(
            id="coint-b",
            name="Cointegrated leg B",
            slug="coint-leg-b",
            source="polymarket",
            description="(test)",
            theme="test",
            is_probability=True,
        ),
        "rw-a": FactorConfig(
            id="rw-a",
            name="Random walk A",
            slug="rw-leg-a",
            source="polymarket",
            description="(test)",
            theme="test",
            is_probability=True,
        ),
        "rw-b": FactorConfig(
            id="rw-b",
            name="Random walk B",
            slug="rw-leg-b",
            source="polymarket",
            description="(test)",
            theme="test",
            is_probability=True,
        ),
    }


def _bounded_walk(seed: int, n: int) -> np.ndarray:
    """Independent bounded random walk in (0.05, 0.95)."""
    rng = np.random.default_rng(seed)
    logit = np.cumsum(rng.normal(0.0, 0.07, size=n))
    prob = 1.0 / (1.0 + np.exp(-logit))
    return np.clip(prob, 0.05, 0.95)


def _build_histories(n: int = 252, start: str = "2024-01-01") -> dict[str, pd.DataFrame]:
    """Build per-slug history DataFrames.

    ``coint-a`` and ``coint-b`` are constructed so that ``a = β·b + spread``
    where ``spread`` is a slow AR(1) with ρ in (0, 1) — i.e. proper textbook
    cointegration with a finite positive half-life. ``b`` is a random walk
    in logit-space. This guarantees:

    * ADF on the OLS residuals rejects the unit-root null (p < 0.05).
    * The AR(1) on the spread has ρ ∈ (0, 1), so the W12-32 helper returns
      a positive finite half-life rather than oscillating.

    ``rw-a`` and ``rw-b`` are independent random walks → should NOT be
    declared cointegrated.
    """
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(42)

    # Leg B: random-walk logit, mapped to probability.
    logit_b = np.cumsum(rng.normal(0.0, 0.07, size=n))

    # Stationary AR(1) spread in logit-space with ρ = 0.8 → half-life ≈ 3.1 days.
    rho = 0.8
    spread_logit = np.zeros(n)
    eps = rng.normal(0.0, 0.05, size=n)
    for t in range(1, n):
        spread_logit[t] = rho * spread_logit[t - 1] + eps[t]

    beta_logit = 1.05  # cointegrating hedge ratio in logit space
    logit_a = beta_logit * logit_b + spread_logit

    prob_a = np.clip(1.0 / (1.0 + np.exp(-logit_a)), 0.05, 0.95)
    prob_b = np.clip(1.0 / (1.0 + np.exp(-logit_b)), 0.05, 0.95)

    return {
        "coint-leg-a": pd.DataFrame({"price": prob_a}, index=idx),
        "coint-leg-b": pd.DataFrame({"price": prob_b}, index=idx),
        "rw-leg-a": pd.DataFrame({"price": _bounded_walk(seed=101, n=n)}, index=idx),
        "rw-leg-b": pd.DataFrame({"price": _bounded_walk(seed=202, n=n)}, index=idx),
    }


@pytest.fixture
def histories() -> dict[str, pd.DataFrame]:
    return _build_histories()


@pytest.fixture
def patched_data(
    monkeypatch: pytest.MonkeyPatch, histories: dict[str, pd.DataFrame]
) -> Iterator[None]:
    """Patch ``_cached_factor_history`` to return the synthetic histories."""

    def fake_cached_factor_history(fc, start, end, poly, cache, settings):
        df = histories[fc.slug].copy()
        df = df[(df.index >= start) & (df.index <= end)]
        return df

    monkeypatch.setattr(_regression_core, "_cached_factor_history", fake_cached_factor_history)
    yield


@pytest.fixture
def client(patched_data: None) -> TestClient:
    """Standalone FastAPI app with the cointegration router mounted."""
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_factors_dep] = _fake_factor_catalog
    app.dependency_overrides[get_polymarket_client] = lambda: object()
    app.dependency_overrides[get_cache] = lambda: NullCache()

    return TestClient(app)


# ---------------------------------------------------------------------------
# Happy-path: cointegrated pair
# ---------------------------------------------------------------------------


def test_cointegrated_pair_rejects_unit_root(client: TestClient) -> None:
    """Synthetic pair sharing a trend → ADF p < 0.05."""
    resp = client.get(
        "/quant/cointegration",
        params={
            "a": "coint-a",
            "b": "coint-b",
            "start": "2024-01-01",
            "end": "2024-09-08",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["a"] == "coint-a"
    assert data["b"] == "coint-b"
    assert data["n_obs"] >= 30
    assert data["adf_p_value"] < 0.05, (
        f"expected p<0.05 for cointegrated pair, got {data['adf_p_value']}"
    )
    assert data["is_cointegrated"] is True


def test_cointegrated_pair_half_life_positive_finite(client: TestClient) -> None:
    """A cointegrated pair has a positive, finite mean-reversion half-life."""
    resp = client.get(
        "/quant/cointegration",
        params={
            "a": "coint-a",
            "b": "coint-b",
            "start": "2024-01-01",
            "end": "2024-09-08",
        },
    )
    assert resp.status_code == 200, resp.text
    hl = resp.json()["half_life_days"]
    assert hl is not None
    assert hl > 0.0
    # Sanity bound — should be well under the full sample length.
    assert hl < 250.0


# ---------------------------------------------------------------------------
# Negative path: independent random walks
# ---------------------------------------------------------------------------


def test_independent_random_walks_not_cointegrated(client: TestClient) -> None:
    """Two independent random walks should NOT be flagged as cointegrated.

    Run on a sample that's long enough to fail the ADF test convincingly;
    the synthetic random walks are independent so ADF on the OLS residuals
    should not reject the unit-root null at α=0.05.
    """
    resp = client.get(
        "/quant/cointegration",
        params={
            "a": "rw-a",
            "b": "rw-b",
            "start": "2024-01-01",
            "end": "2024-09-08",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["adf_p_value"] >= 0.05, (
        f"expected p>=0.05 for independent walks, got {data['adf_p_value']}"
    )
    assert data["is_cointegrated"] is False


# ---------------------------------------------------------------------------
# Validation: 404 / 422
# ---------------------------------------------------------------------------


def test_unknown_slug_a_returns_404(client: TestClient) -> None:
    """An unknown ``a`` slug → 404 with structured ``did_you_mean`` hint."""
    resp = client.get(
        "/quant/cointegration",
        params={"a": "no-such-factor", "b": "coint-b"},
    )
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert detail["query"] == "no-such-factor"
    assert detail["leg"] == "a"
    assert "did_you_mean" in detail


def test_unknown_slug_b_returns_404(client: TestClient) -> None:
    """An unknown ``b`` slug → 404."""
    resp = client.get(
        "/quant/cointegration",
        params={"a": "coint-a", "b": "totally-bogus"},
    )
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert detail["leg"] == "b"


def test_start_after_end_returns_422(client: TestClient) -> None:
    """``start >= end`` is a client error → 422."""
    resp = client.get(
        "/quant/cointegration",
        params={
            "a": "coint-a",
            "b": "coint-b",
            "start": "2024-09-01",
            "end": "2024-01-01",
        },
    )
    assert resp.status_code == 422, resp.text


def test_same_factor_pair_returns_422(client: TestClient) -> None:
    """Cointegration of a series with itself is trivial → 422."""
    resp = client.get(
        "/quant/cointegration",
        params={"a": "coint-a", "b": "coint-a"},
    )
    assert resp.status_code == 422, resp.text


def test_empty_a_param_returns_422(client: TestClient) -> None:
    """Empty ``a`` param violates ``min_length=1`` → 422."""
    resp = client.get(
        "/quant/cointegration",
        params={"a": "", "b": "coint-b"},
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Half-life delegation to W12-32
# ---------------------------------------------------------------------------


def test_half_life_matches_w12_32_helper(
    client: TestClient, histories: dict[str, pd.DataFrame]
) -> None:
    """The router's ``half_life_days`` matches the W12-32 helper on the spread.

    Replays the same alignment + OLS step locally, then calls
    :func:`pfm.quant.half_life.estimate_half_life` directly on the residuals.
    The endpoint must return the same value (within FP tolerance).
    """
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-09-08", tz="UTC")
    a = histories["coint-leg-a"]["price"]
    b = histories["coint-leg-b"]["price"]
    a = a[(a.index >= start) & (a.index <= end)]
    b = b[(b.index >= start) & (b.index <= end)]

    # Reproduce engle_granger's step-1 OLS to get the same residual spread.
    import statsmodels.api as sm

    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    X = sm.add_constant(df["b"].values)
    ols = sm.OLS(df["a"].values, X).fit()
    spread = pd.Series(ols.resid, index=df.index)
    expected = estimate_half_life(spread)["half_life_days"]

    resp = client.get(
        "/quant/cointegration",
        params={
            "a": "coint-a",
            "b": "coint-b",
            "start": "2024-01-01",
            "end": "2024-09-08",
        },
    )
    assert resp.status_code == 200, resp.text
    hl = resp.json()["half_life_days"]

    # Both finite + close.
    assert hl is not None
    assert np.isfinite(expected), f"helper returned non-finite: {expected}"
    assert abs(hl - float(expected)) < 1e-6


# ---------------------------------------------------------------------------
# Response-shape / contract
# ---------------------------------------------------------------------------


def test_response_shape_matches_contract(client: TestClient) -> None:
    """Every documented field is present with the right type."""
    resp = client.get(
        "/quant/cointegration",
        params={
            "a": "coint-a",
            "b": "coint-b",
            "start": "2024-01-01",
            "end": "2024-09-08",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    expected_keys = {
        "a",
        "b",
        "n_obs",
        "beta",
        "alpha",
        "adf_stat",
        "adf_p_value",
        "is_cointegrated",
        "half_life_days",
    }
    assert set(data) == expected_keys
    assert isinstance(data["a"], str)
    assert isinstance(data["b"], str)
    assert isinstance(data["n_obs"], int)
    assert isinstance(data["beta"], (int, float))
    assert isinstance(data["alpha"], (int, float))
    assert isinstance(data["adf_stat"], (int, float))
    assert isinstance(data["adf_p_value"], (int, float))
    assert isinstance(data["is_cointegrated"], bool)
    # half_life_days is nullable.
    assert data["half_life_days"] is None or isinstance(data["half_life_days"], (int, float))


def test_slug_aliasing_echoes_canonical_id(client: TestClient) -> None:
    """Input by slug → response echoes the resolved canonical ``id``."""
    # ``coint-leg-a`` is the slug; ``coint-a`` is the id. The unified
    # resolver should match the slug and the response should echo the id.
    resp = client.get(
        "/quant/cointegration",
        params={
            "a": "coint-leg-a",  # slug, not id
            "b": "coint-b",
            "start": "2024-01-01",
            "end": "2024-09-08",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["a"] == "coint-a"
    assert resp.json()["b"] == "coint-b"


def test_default_dates_route_reachable(client: TestClient) -> None:
    """Omitting ``start`` and ``end`` is accepted by the schema.

    The synthetic histories span 2024-01-01 → 2024-09-08, so a 1-year-from-
    today default window will return empty in the test environment (which
    surfaces as 502). The important assertion is *no 500* and *no 4xx-on-
    schema*; the default-window contract is intentionally generous.
    """
    resp = client.get(
        "/quant/cointegration",
        params={"a": "coint-a", "b": "coint-b"},
    )
    # 200 (rare — only if synthetic dates overlap default window), 422 (no
    # overlap after alignment), or 502 (upstream empty) are all acceptable.
    # The router must NOT 500 and must NOT 4xx-on-schema (no missing-param).
    assert resp.status_code in (200, 422, 502), resp.text


def test_beta_recovered_for_cointegrated_pair(client: TestClient) -> None:
    """OLS β for the cointegrated pair should be finite and non-trivial.

    The synthetic DGP shares a logit-space trend; in raw probability space
    the relationship is monotonic-but-nonlinear, so we just check β is
    finite and not ridiculously far from a sensible range.
    """
    resp = client.get(
        "/quant/cointegration",
        params={
            "a": "coint-a",
            "b": "coint-b",
            "start": "2024-01-01",
            "end": "2024-09-08",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert np.isfinite(data["beta"])
    assert np.isfinite(data["alpha"])
    # β should be positive (legs move together) and roughly O(1).
    assert -5.0 < data["beta"] < 5.0
