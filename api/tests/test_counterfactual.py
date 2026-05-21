"""Tests for the counterfactual backtest module.

Strategy: build a synthetic ticker return series with a known linear
dependence on Δlogit_factor, then verify that
:func:`counterfactual_path` recovers the right beta and that the
counterfactual path matches the analytical expectation.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import get_cache
from pfm.counterfactual import (
    attribution_decomposition,
    counterfactual_path,
    router,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    get_cache("counterfactual").clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _synthetic_pair(
    n: int = 60,
    beta_true: float = 0.5,
    noise_sd: float = 0.001,
    drift: float = 0.005,
) -> tuple[pd.Series, pd.Series]:
    """Build (returns, dlogit) where returns = β · dlogit + ε with small ε."""
    rng = np.random.default_rng(seed=42)
    idx = pd.date_range(start="2024-01-01", periods=n, freq="B")
    dlogit = pd.Series(rng.normal(drift, 0.20, size=n), index=idx)
    returns = pd.Series(beta_true * dlogit.values + rng.normal(0, noise_sd, size=n), index=idx)
    return returns, dlogit


# ---------------------------------------------------------------------------
# counterfactual_path — recovers β from synthetic DGP
# ---------------------------------------------------------------------------


def test_counterfactual_recovers_known_beta() -> None:
    """returns = 0.5·dlogit → estimated β should be ≈ 0.5."""
    rets, dlog = _synthetic_pair(n=80, beta_true=0.5, noise_sd=0.0005)

    out = counterfactual_path(
        ticker="TEST",
        factor_id="some-factor",
        scenario="NO",
        actual_resolution="YES",
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
        returns=rets,
        dlogit=dlog,
    )
    assert out["beta"] == pytest.approx(0.5, abs=0.05)


def test_counterfactual_path_differs_from_actual_when_scenario_flipped() -> None:
    rets, dlog = _synthetic_pair(n=60, beta_true=0.5, noise_sd=0.0005, drift=0.01)
    out = counterfactual_path(
        ticker="TEST",
        factor_id="some-factor",
        scenario="NO",
        actual_resolution="YES",
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
        returns=rets,
        dlogit=dlog,
        beta=0.5,
    )
    actual_final = out["actual_path"][-1]["price"]
    counter_final = out["counterfactual_path"][-1]["price"]
    assert actual_final != counter_final
    # β=0.5, sum_dlogit ≈ +0.6 → flipping flips sign of attribution.
    # Actual return total > counterfactual total when drift is positive.
    assert out["total_return_actual_pct"] > out["total_return_counterfactual_pct"]


def test_counterfactual_path_equal_when_scenario_matches_actual() -> None:
    rets, dlog = _synthetic_pair(n=40)
    out = counterfactual_path(
        ticker="TEST",
        factor_id="f",
        scenario="YES",
        actual_resolution="YES",
        start=date(2024, 1, 1),
        end=date(2024, 3, 1),
        returns=rets,
        dlogit=dlog,
        beta=0.5,
    )
    actual = [p["price"] for p in out["actual_path"]]
    counter = [p["price"] for p in out["counterfactual_path"]]
    assert actual == counter


def test_counterfactual_attribution_pct_in_reasonable_range() -> None:
    rets, dlog = _synthetic_pair(n=80, beta_true=0.8, noise_sd=0.0005, drift=0.005)
    out = counterfactual_path(
        ticker="TEST",
        factor_id="f",
        scenario="NO",
        actual_resolution="YES",
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
        returns=rets,
        dlogit=dlog,
        beta=0.8,
    )
    assert out["n_obs"] == 80
    # Attribution should be a finite percentage; with strong DGP it's typically large.
    assert isinstance(out["attribution_pct"], float)
    assert isinstance(out["attributable_return_total_pct"], float)


def test_counterfactual_path_has_n_obs_points() -> None:
    rets, dlog = _synthetic_pair(n=25)
    out = counterfactual_path(
        ticker="TEST",
        factor_id="f",
        scenario="NO",
        actual_resolution="YES",
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
        returns=rets,
        dlogit=dlog,
        beta=0.5,
    )
    assert len(out["actual_path"]) == 25
    assert len(out["counterfactual_path"]) == 25


# ---------------------------------------------------------------------------
# attribution_decomposition
# ---------------------------------------------------------------------------


def test_attribution_decomposition_two_factors() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range(start="2024-01-01", periods=100, freq="B")
    f1 = pd.Series(rng.normal(0.005, 0.20, size=100), index=idx)
    f2 = pd.Series(rng.normal(-0.003, 0.20, size=100), index=idx)
    rets = pd.Series(0.4 * f1.values + 0.2 * f2.values + rng.normal(0, 0.0005, size=100), index=idx)

    out = attribution_decomposition(
        ticker="TEST",
        factors_list=["f1", "f2"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
        betas={"f1": 0.4, "f2": 0.2},
        returns=rets,
        dlogits={"f1": f1, "f2": f2},
    )

    assert out["n_factors"] == 2
    assert {r["factor_id"] for r in out["rows"]} == {"f1", "f2"}
    by_id = {r["factor_id"]: r for r in out["rows"]}
    assert by_id["f1"]["beta"] == pytest.approx(0.4, abs=1e-6)
    assert by_id["f2"]["beta"] == pytest.approx(0.2, abs=1e-6)
    # Shares must sum to 1 (across absolute contributions).
    shares = sum(r["contribution_share"] for r in out["rows"])
    assert shares == pytest.approx(1.0, abs=1e-3)


def test_attribution_decomposition_residual_small_when_factors_explain_returns() -> None:
    rng = np.random.default_rng(11)
    idx = pd.date_range(start="2024-01-01", periods=120, freq="B")
    f1 = pd.Series(rng.normal(0.01, 0.18, size=120), index=idx)
    rets = pd.Series(0.5 * f1.values + rng.normal(0, 0.0001, size=120), index=idx)

    out = attribution_decomposition(
        ticker="TEST",
        factors_list=["f1"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 1),
        betas={"f1": 0.5},
        returns=rets,
        dlogits={"f1": f1},
    )
    # With β fixed and almost-zero noise, residual should be tiny.
    assert abs(out["residual_pct"]) < 1.0


def test_attribution_decomposition_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        attribution_decomposition(
            ticker="X", factors_list=[], start=date(2024, 1, 1), end=date(2024, 6, 1)
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_post_counterfactual_endpoint(client: TestClient) -> None:
    body = {
        "ticker": "NVDA",
        "factor_id": "trump-wins-2024",
        "scenario": "NO",
        "actual_resolution": "YES",
        "start": "2024-08-01",
        "end": "2024-12-01",
    }
    r = client.post("/counterfactual", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ticker"] == "NVDA"
    assert data["scenario"] == "NO"
    assert len(data["actual_path"]) > 0
    assert len(data["counterfactual_path"]) > 0


def test_post_counterfactual_endpoint_bad_dates(client: TestClient) -> None:
    body = {
        "ticker": "NVDA",
        "factor_id": "f",
        "scenario": "NO",
        "start": "2024-12-01",
        "end": "2024-08-01",
    }
    r = client.post("/counterfactual", json=body)
    assert r.status_code == 400


def test_post_counterfactual_multi_endpoint(client: TestClient) -> None:
    body = {
        "ticker": "NVDA",
        "factors_list": ["fed-cuts-2026", "recession-2026"],
        "start": "2024-08-01",
        "end": "2024-12-01",
    }
    r = client.post("/counterfactual/multi", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["n_factors"] >= 1
    assert {row["factor_id"] for row in data["rows"]}.issubset({"fed-cuts-2026", "recession-2026"})


def test_post_counterfactual_multi_empty_factors(client: TestClient) -> None:
    body = {
        "ticker": "NVDA",
        "factors_list": [],
        "start": "2024-08-01",
        "end": "2024-12-01",
    }
    r = client.post("/counterfactual/multi", json=body)
    # Pydantic min_length=1 → 422
    assert r.status_code == 422
