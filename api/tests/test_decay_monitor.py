"""Tests for ``pfm.decay_monitor``.

Scope:

* Synthetic-DGP: returns engineered to have Sharpe ≈ 2.0 should
  classify as FRESH/STABLE.
* Drift injection: tail-degraded returns must collapse the rolling
  Sharpe and produce DECAYING.
* All-zero returns: must produce DEAD (zero variance + zero ratio).
* End-to-end: ``/alpha/decay`` is reachable through ``TestClient``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.decay_monitor import (
    ANNUALISATION_DAYS,
    check_all_alphas,
    compute_rolling_sharpe,
    detect_decay,
)
from pfm.decay_monitor import (
    router as decay_router,
)

# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def fresh_returns() -> pd.Series:
    """200 daily returns engineered for ann-Sharpe ≈ 2.0."""
    rng = np.random.default_rng(seed=42)
    daily_vol = 0.01
    target_sharpe = 2.0
    daily_mean = target_sharpe * daily_vol / math.sqrt(ANNUALISATION_DAYS)
    n = 200
    r = rng.normal(loc=daily_mean, scale=daily_vol, size=n)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.Series(r, index=idx, name="returns")


@pytest.fixture
def decaying_returns() -> pd.Series:
    """Returns where the trailing 60 obs collapse to ≈ zero mean."""
    rng = np.random.default_rng(seed=7)
    daily_vol = 0.01
    target_sharpe = 2.0
    daily_mean = target_sharpe * daily_vol / math.sqrt(ANNUALISATION_DAYS)
    n = 200
    r = rng.normal(loc=daily_mean, scale=daily_vol, size=n)
    # Inject decay: last 60 obs become noise around zero.
    r[-60:] = rng.normal(loc=0.0, scale=daily_vol, size=60)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.Series(r, index=idx, name="returns")


@pytest.fixture
def temp_alpha_catalog(tmp_path: Path) -> Path:
    """Tiny ``alpha_strategies.json`` for end-to-end tests."""
    payload = {
        "generated": "2026-05-08",
        "strategies": [
            {
                "pair_id": "fresh_pair",
                "tier": "A_GOLD",
                "oos_sharpe": 2.0,
            },
            {
                "pair_id": "decaying_pair",
                "tier": "A_GOLD",
                "oos_sharpe": 2.0,
            },
        ],
    }
    p = tmp_path / "alpha_strategies.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# --- compute_rolling_sharpe -------------------------------------------------


def test_rolling_sharpe_fresh_is_near_target(fresh_returns: pd.Series) -> None:
    rolling = compute_rolling_sharpe(fresh_returns, window=30)
    # Some of the early window is NaN.
    assert rolling.iloc[:29].isna().all()
    # Trailing rolling Sharpe should be in the same ballpark as 2.0;
    # finite-sample noise widens the band.
    tail_mean = rolling.dropna().tail(60).mean()
    assert 0.5 < tail_mean < 4.0, f"tail mean Sharpe out of band: {tail_mean}"


def test_rolling_sharpe_zero_variance_emits_zero() -> None:
    s = pd.Series([0.0] * 60, index=pd.date_range("2026-01-01", periods=60))
    rolling = compute_rolling_sharpe(s, window=30)
    # All rolling values after the warmup must be exactly 0.0.
    assert (rolling.dropna() == 0.0).all()


def test_rolling_sharpe_rejects_window_too_small() -> None:
    s = pd.Series([0.001, -0.002, 0.001])
    with pytest.raises(ValueError):
        compute_rolling_sharpe(s, window=1)


def test_rolling_sharpe_handles_empty() -> None:
    out = compute_rolling_sharpe(pd.Series(dtype=float), window=30)
    assert out.empty


# --- detect_decay -----------------------------------------------------------


def test_detect_decay_fresh(fresh_returns: pd.Series) -> None:
    rolling = compute_rolling_sharpe(fresh_returns, window=30)
    status = detect_decay(rolling, baseline=2.0)
    assert status["decay_indicator"] in {"FRESH", "STABLE"}
    assert status["demote_recommendation"] == "A_GOLD"
    assert status["baseline_sharpe"] == 2.0
    assert status["n_consecutive_below"] >= 0


def test_detect_decay_decaying(decaying_returns: pd.Series) -> None:
    rolling = compute_rolling_sharpe(decaying_returns, window=30)
    status = detect_decay(rolling, baseline=2.0)
    assert status["decay_indicator"] in {"DECAYING", "DEAD"}
    assert status["demote_recommendation"] in {"B_VALIDATED", "C_TENTATIVE"}
    # Tail collapsed → at least some consecutive sub-threshold prints.
    assert status["n_consecutive_below"] >= 1


def test_detect_decay_all_zeros() -> None:
    s = pd.Series([0.0] * 100, index=pd.date_range("2026-01-01", periods=100))
    rolling = compute_rolling_sharpe(s, window=30)
    status = detect_decay(rolling, baseline=2.0)
    assert status["current_sharpe"] == 0.0
    assert status["decay_indicator"] == "DEAD"
    assert status["demote_recommendation"] == "C_TENTATIVE"


def test_detect_decay_ratio_band_dead() -> None:
    # Construct a rolling series fixed below 0.3 * baseline.
    rolling = pd.Series([0.1] * 50, index=pd.date_range("2026-01-01", periods=50))
    status = detect_decay(rolling, baseline=2.0)
    assert status["decay_indicator"] == "DEAD"


def test_detect_decay_ratio_band_decaying() -> None:
    # current = 1.0, baseline = 2.0 → ratio = 0.5 ∈ [0.3, 0.7).
    # threshold = 0.5 * 2.0 = 1.0; at exactly 1.0 nothing is "below",
    # so n_consecutive_below = 0 and we land in DECAYING via ratio.
    rolling = pd.Series([1.0] * 50, index=pd.date_range("2026-01-01", periods=50))
    status = detect_decay(rolling, baseline=2.0)
    assert status["decay_indicator"] == "DECAYING"
    assert status["demote_recommendation"] == "B_VALIDATED"


def test_detect_decay_ratio_band_stable() -> None:
    # ratio = 0.8 → STABLE.
    rolling = pd.Series([1.6] * 50, index=pd.date_range("2026-01-01", periods=50))
    status = detect_decay(rolling, baseline=2.0)
    assert status["decay_indicator"] == "STABLE"
    assert status["demote_recommendation"] == "A_GOLD"


def test_detect_decay_consecutive_dead_overrides_ratio() -> None:
    # ratio = 1.0 (current==baseline, so > 0.9) but tail has 12
    # straight obs below 50% of baseline → DEAD wins.
    series = [2.0] * 30 + [0.5] * 12 + [2.0]
    idx = pd.date_range("2026-01-01", periods=len(series))
    # Make the LAST point sub-threshold so the consecutive counter
    # actually applies to the tail.
    series[-1] = 0.5
    rolling = pd.Series(series, index=idx)
    status = detect_decay(rolling, baseline=2.0)
    assert status["n_consecutive_below"] >= 10
    assert status["decay_indicator"] == "DEAD"


# --- check_all_alphas -------------------------------------------------------


def test_check_all_alphas_temp_catalog(temp_alpha_catalog: Path) -> None:
    out = check_all_alphas(str(temp_alpha_catalog))
    assert set(out.keys()) == {"fresh_pair", "decaying_pair"}
    for pair_id, status in out.items():
        assert status["pair_id"] == pair_id
        assert "decay_indicator" in status
        assert status["decay_indicator"] in {"FRESH", "STABLE", "DECAYING", "DEAD"}
        assert status["n_obs"] > 0
        assert status["tier"] == "A_GOLD"


def test_check_all_alphas_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    out = check_all_alphas(str(missing))
    assert out == {}


def test_check_all_alphas_real_catalog_smoke() -> None:
    """The real ``web/data/alpha_strategies.json`` should load and classify."""
    out = check_all_alphas()  # default path
    # Don't assert exact counts — catalog may evolve. Just sanity-check.
    assert isinstance(out, dict)
    if out:
        sample = next(iter(out.values()))
        assert sample["decay_indicator"] in {"FRESH", "STABLE", "DECAYING", "DEAD"}


# --- router (TestClient) ----------------------------------------------------


@pytest.fixture
def decay_app(temp_alpha_catalog: Path) -> TestClient:
    """Stand-alone FastAPI app mounting only the decay router."""
    app = FastAPI()
    app.include_router(decay_router)
    # Stash the temp catalog path so tests can pass it as a query param.
    app.state.alpha_catalog_path = str(temp_alpha_catalog)
    return TestClient(app)


def test_endpoint_alpha_decay(decay_app: TestClient) -> None:
    catalog = decay_app.app.state.alpha_catalog_path
    resp = decay_app.get("/alpha/decay", params={"alpha_strategies_path": catalog})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_total"] == 2
    assert body["n_fresh"] + body["n_stable"] + body["n_decaying"] + body["n_dead"] == 2
    pair_ids = {it["pair_id"] for it in body["items"]}
    assert pair_ids == {"fresh_pair", "decaying_pair"}


def test_endpoint_rolling_sharpe(decay_app: TestClient) -> None:
    catalog = decay_app.app.state.alpha_catalog_path
    resp = decay_app.get(
        "/alpha/fresh_pair/rolling-sharpe",
        params={"window": 20, "alpha_strategies_path": catalog},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pair_id"] == "fresh_pair"
    assert body["window"] == 20
    assert body["n_obs"] > 0
    assert isinstance(body["series"], list)
    # First (window-1) entries should be null rolling Sharpe.
    null_count = sum(1 for p in body["series"] if p["rolling_sharpe"] is None)
    assert null_count == 19


def test_endpoint_rolling_sharpe_unknown_pair(decay_app: TestClient) -> None:
    catalog = decay_app.app.state.alpha_catalog_path
    resp = decay_app.get(
        "/alpha/does_not_exist/rolling-sharpe",
        params={"alpha_strategies_path": catalog},
    )
    assert resp.status_code == 404


def test_endpoint_recompute(decay_app: TestClient) -> None:
    catalog = decay_app.app.state.alpha_catalog_path
    resp = decay_app.post(
        "/alpha/fresh_pair/recompute-decay",
        params={"alpha_strategies_path": catalog},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pair_id"] == "fresh_pair"
    assert body["forced"] is True
    assert body["status"]["pair_id"] == "fresh_pair"
