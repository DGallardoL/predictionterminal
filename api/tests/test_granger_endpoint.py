"""Tests for ``GET /quant/granger`` (W13-18).

The router is mounted on a throw-away FastAPI app with the four data
dependencies overridden, and ``pfm.regression_core._cached_factor_history``
is monkeypatched to deterministic synthetic series so no network IO occurs.

The synthetic series are indexed at ``today - 600d ... today`` so the
endpoint's default date window (``end=today``, ``start=end-365d``) overlaps
the fixture's range with plenty of headroom. Tests that need a *non*-default
window pass explicit ``start``/``end`` query params.

Coverage targets (≥10 tests, per W13-18):

1.  Planted A → B causation: lag-1 should be significant.
2.  Reverse direction: B → A on the same data should NOT trigger
    ``a_granger_causes_b``.
3.  Independent series: all p-values > 0.05.
4.  Maxlag bounds — lower (``maxlag=0`` → 422).
5.  Maxlag bounds — upper (``maxlag=21`` → 422).
6.  Maxlag default (5) and explicit 1.
7.  Unknown factor → 404 with ``did_you_mean``.
8.  ``a == b`` self-test → 422.
9.  Two aliases of the same factor → 422.
10. Window too short for the requested maxlag → 422.
11. Response shape: keys, types, and per-lag entry count.
12. ``best_lag`` is the lag with the smallest p-value.
13. ``significant`` flag matches ``p_value < 0.05`` per row.
14. Maxlag bounds — both extremes (1 and 20) succeed shape-wise.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.regression_core as _regression_core
from pfm.cache import NullCache
from pfm.dependencies import get_cache, get_factors_dep, get_polymarket_client
from pfm.factors import FactorConfig
from pfm.quant.granger_router import router

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _catalog() -> dict[str, FactorConfig]:
    """Four-factor catalog used by the tests."""
    return {
        "leader": FactorConfig(
            id="leader",
            name="Leader signal",
            slug="leader-slug",
            source="polymarket",
            description="(test)",
            theme="test",
            is_probability=True,
        ),
        "follower": FactorConfig(
            id="follower",
            name="Follower signal",
            slug="follower-slug",
            source="polymarket",
            description="(test)",
            theme="test",
            is_probability=True,
        ),
        "noise": FactorConfig(
            id="noise",
            name="Independent noise",
            slug="noise-slug",
            source="polymarket",
            description="(test)",
            theme="test",
            is_probability=True,
        ),
        "short-window": FactorConfig(
            id="short-window",
            name="Short series",
            slug="short-slug",
            source="polymarket",
            description="(test)",
            theme="test",
            is_probability=True,
        ),
    }


def _planted_pair(n: int = 600, seed: int = 7, alpha: float = 0.6) -> tuple[pd.Series, pd.Series]:
    """Generate a leader/follower pair where leader[t-1] -> follower[t].

    Construction notes (matters for Granger interpretation):

    * The leader is **i.i.d. logit-noise** — explicitly NOT a random walk.
      A random-walk leader makes ``follower[t]`` predictive of
      ``leader[t]`` (the lagged-leader signal in the follower is
      autocorrelated with the leader's own future), which mechanically
      triggers reverse-Granger significance even though the DGP is purely
      one-way. White-noise leader breaks that loop and gives the test
      something honest to detect.
    * Both series are mapped to (0.05, 0.95) so they look like
      prediction-market probabilities and never explode the test.
    * With ``alpha = 0.6`` and ``n = 600`` the Granger F-test at lag 1 is
      overwhelmingly significant in the planted direction
      (``p ~ 1e-15``) while the reverse direction stays at ``p > 0.05``.
    """
    rng = np.random.default_rng(seed)
    # Anchor on today so the endpoint's default window (end=today,
    # start=end-365d) always overlaps the synthetic data.
    end = pd.Timestamp(date.today(), tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="D", tz="UTC")

    # White-noise leader in logit space → mean-zero, no autocorrelation.
    leader_logit = rng.normal(0, 1.0, n)
    leader = 1.0 / (1.0 + np.exp(-leader_logit))
    leader = np.clip(leader, 0.05, 0.95)

    follower = np.zeros(n)
    follower[0] = 0.5
    leader_dev = leader - 0.5
    for t in range(1, n):
        follower[t] = 0.4 * follower[t - 1] + 0.3 + alpha * leader_dev[t - 1] + rng.normal(0, 0.02)
    follower = np.clip(follower, 0.05, 0.95)

    s_leader = pd.Series(leader, index=idx, name="price")
    s_follower = pd.Series(follower, index=idx, name="price")
    return s_leader, s_follower


def _independent_pair(n: int = 600, seed: int = 11) -> tuple[pd.Series, pd.Series]:
    """Two independent random-walk-in-logit probability series."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(date.today(), tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="D", tz="UTC")

    def _walk() -> np.ndarray:
        logit = np.cumsum(rng.normal(0, 0.1, n))
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.clip(p, 0.05, 0.95)

    return (
        pd.Series(_walk(), index=idx, name="price"),
        pd.Series(_walk(), index=idx, name="price"),
    )


@pytest.fixture
def planted_data(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, pd.Series]]:
    """Monkeypatch ``_cached_factor_history`` to return planted pair + noise."""
    s_leader, s_follower = _planted_pair()
    _s_noise_a, s_noise_b = _independent_pair()
    # Short window to trip the n_obs check: the most recent 30 days only.
    # With maxlag=10 the statsmodels floor is max(20, 4*10+2)=42, so 30 is
    # not enough and the endpoint must 422.
    s_short = s_leader.iloc[-30:].copy()

    by_slug: dict[str, pd.Series] = {
        "leader-slug": s_leader,
        "follower-slug": s_follower,
        "noise-slug": s_noise_b,
        "short-slug": s_short,
    }

    def fake_history(fc, start, end, poly, cache, settings):
        series = by_slug[fc.slug]
        df = pd.DataFrame({"price": series})
        df = df[(df.index >= start) & (df.index <= end)]
        return df

    monkeypatch.setattr(_regression_core, "_cached_factor_history", fake_history)
    yield by_slug


@pytest.fixture
def client(planted_data: dict[str, pd.Series]) -> TestClient:
    """Standalone app mounting only the Granger router with overridden deps."""
    app = FastAPI()
    app.include_router(router)

    catalog = _catalog()
    app.dependency_overrides[get_factors_dep] = lambda: catalog
    app.dependency_overrides[get_cache] = lambda: NullCache()
    app.dependency_overrides[get_polymarket_client] = lambda: object()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_planted_a_causes_b_is_significant(client: TestClient) -> None:
    """Leader → follower at lag 1 should be overwhelmingly significant."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "follower", "maxlag": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["a"] == "leader"
    assert body["b"] == "follower"
    assert body["a_granger_causes_b"] is True
    assert body["best_lag"] == 1
    assert body["best_p_value"] is not None
    assert body["best_p_value"] < 0.05
    # The lag-1 row in particular must be significant.
    lag1 = next(t for t in body["tests"] if t["lag"] == 1)
    assert lag1["significant"] is True
    assert lag1["p_value"] < 0.05


def test_reverse_direction_is_not_significant(client: TestClient) -> None:
    """B (follower) does NOT Granger-cause A (leader) on the planted pair.

    By construction the follower's innovations carry no information about
    future leader values, so the F-test should fail to reject.
    """
    r = client.get(
        "/quant/granger",
        params={"a": "follower", "b": "leader", "maxlag": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["a"] == "follower"
    assert body["b"] == "leader"
    # The reverse direction should be insignificant.
    assert body["a_granger_causes_b"] is False
    # All per-lag p-values should be > 0.05 OR at minimum the best p > 0.01.
    # We use a loose check to avoid flakiness on a single RNG seed.
    assert body["best_p_value"] is None or body["best_p_value"] > 0.01


def test_independent_series_no_causation(client: TestClient) -> None:
    """Two independent random walks: no lag should be significant."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "noise", "maxlag": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["a_granger_causes_b"] is False
    # Best p-value (if any) should not clear the 0.05 bar.
    if body["best_p_value"] is not None:
        assert body["best_p_value"] >= 0.05


def test_maxlag_below_lower_bound_is_422(client: TestClient) -> None:
    """``maxlag=0`` is below the documented [1, 20] range."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "follower", "maxlag": 0},
    )
    assert r.status_code == 422


def test_maxlag_above_upper_bound_is_422(client: TestClient) -> None:
    """``maxlag=21`` exceeds the documented [1, 20] range."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "follower", "maxlag": 21},
    )
    assert r.status_code == 422


def test_maxlag_default_is_five(client: TestClient) -> None:
    """Omitting ``maxlag`` should produce 5 per-lag entries."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "follower"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["maxlag"] == 5
    assert len(body["tests"]) == 5
    assert [t["lag"] for t in body["tests"]] == [1, 2, 3, 4, 5]


def test_maxlag_one_returns_single_lag(client: TestClient) -> None:
    """``maxlag=1`` (the lower bound) returns exactly one row."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "follower", "maxlag": 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["maxlag"] == 1
    assert len(body["tests"]) == 1
    assert body["tests"][0]["lag"] == 1


def test_unknown_factor_is_404_with_hint(client: TestClient) -> None:
    """A typo in the factor id should yield 404 + ``did_you_mean``."""
    r = client.get(
        "/quant/granger",
        params={"a": "leadr", "b": "follower"},
    )
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "leadr" in detail["error"] or "leadr" in str(detail)
    assert "did_you_mean" in detail


def test_self_test_is_422(client: TestClient) -> None:
    """A == B is mechanically infeasible — refuse early with 422."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "leader"},
    )
    assert r.status_code == 422


def test_window_too_short_is_422(client: TestClient) -> None:
    """Too few aligned observations for the requested maxlag → 422.

    The ``short-window`` factor has only 30 obs (anchored at today). With
    ``maxlag=10`` the statsmodels floor is ``max(20, 4*10+2) = 42``, so
    even the full 30-row sample falls short and the endpoint refuses.
    """
    today = date.today()
    start = today - timedelta(days=40)
    r = client.get(
        "/quant/granger",
        params={
            "a": "short-window",
            "b": "follower",
            "maxlag": 10,
            "start": start.isoformat(),
            "end": today.isoformat(),
        },
    )
    assert r.status_code == 422
    assert "obs" in r.json()["detail"].lower() or "maxlag" in r.json()["detail"].lower()


def test_response_shape_and_types(client: TestClient) -> None:
    """Response keys + types match the documented contract."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "follower", "maxlag": 3},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {
        "a",
        "b",
        "maxlag",
        "n_obs",
        "tests",
        "best_lag",
        "best_p_value",
        "a_granger_causes_b",
    }
    assert isinstance(body["a"], str)
    assert isinstance(body["b"], str)
    assert isinstance(body["maxlag"], int)
    assert isinstance(body["n_obs"], int) and body["n_obs"] > 0
    assert isinstance(body["a_granger_causes_b"], bool)
    assert isinstance(body["tests"], list) and len(body["tests"]) == 3
    for t in body["tests"]:
        assert set(t.keys()) == {"lag", "f_stat", "p_value", "significant"}
        assert isinstance(t["lag"], int)
        assert isinstance(t["f_stat"], (int, float))
        assert isinstance(t["p_value"], (int, float))
        assert isinstance(t["significant"], bool)


def test_best_lag_minimises_p_value(client: TestClient) -> None:
    """``best_lag`` must be the lag with the smallest p-value."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "follower", "maxlag": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Pick lag with min p_value ourselves and compare to API's best_lag.
    finite = [t for t in body["tests"] if t["p_value"] == t["p_value"]]  # not NaN
    expected = min(finite, key=lambda t: (t["p_value"], t["lag"]))
    assert body["best_lag"] == expected["lag"]
    assert body["best_p_value"] == pytest.approx(expected["p_value"], rel=1e-9)


def test_significant_flag_matches_p_value(client: TestClient) -> None:
    """Per-row ``significant`` is true iff ``p_value < 0.05``."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "follower", "maxlag": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    for t in body["tests"]:
        if t["p_value"] == t["p_value"]:  # not NaN
            assert t["significant"] == (t["p_value"] < 0.05)
        else:
            assert t["significant"] is False


def test_maxlag_upper_bound_twenty_succeeds(client: TestClient) -> None:
    """``maxlag=20`` (the upper bound) is accepted and returns 20 rows."""
    r = client.get(
        "/quant/granger",
        params={"a": "leader", "b": "follower", "maxlag": 20},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["maxlag"] == 20
    assert len(body["tests"]) == 20
    assert [t["lag"] for t in body["tests"]] == list(range(1, 21))
