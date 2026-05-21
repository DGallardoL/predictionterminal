"""Tests for ``pfm.portfolio_optimizer`` and the /strategies/optimize endpoint.

Coverage:
  * synthetic-DGP recovery: min_variance loads on the lowest-σ asset
  * ERC invariant: risk-parity equalises wᵢ·(Σw)ᵢ across assets
  * HRP robustness: works on a singular Σ (rank-deficient sample) without raising
  * Equal-weight baseline: a real optimiser beats 1/N in-sample
  * Box constraints: SLSQP respects max_w
  * Frontier monotonicity: as vol increases, return is non-decreasing
  * Endpoint smoke test: /strategies/optimize returns a valid response

All synthetic data uses ``np.random.default_rng(seed=...)`` for determinism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.portfolio_optimizer import (
    efficient_frontier,
    equal_weight,
    hrp,
    mean_variance_max_sharpe,
    min_variance,
    monte_carlo_drawdown,
    risk_parity_erc,
)
from pfm.portfolio_optimizer_router import router as optimizer_router

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _gaussian_returns(
    sigmas: list[float],
    rho: float = 0.30,
    mu: list[float] | None = None,
    n: int = 500,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate correlated Gaussian daily returns with prescribed σ's and ρ."""
    rng = np.random.default_rng(seed)
    k = len(sigmas)
    sigmas_arr = np.asarray(sigmas, dtype=float)
    mu_arr = np.zeros(k) if mu is None else np.asarray(mu, dtype=float)
    # Daily-equivalent params (caller supplies daily σ).
    corr = np.full((k, k), rho)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(sigmas_arr, sigmas_arr) * corr
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=(n, k))
    paths = z @ chol.T + mu_arr
    cols = [f"a{i}" for i in range(k)]
    return pd.DataFrame(paths, columns=cols)


# ---------------------------------------------------------------------------
# 1. min-variance loads on the lowest-σ asset
# ---------------------------------------------------------------------------


def test_min_variance_prefers_low_vol_asset() -> None:
    """Three assets with σ = [0.10, 0.20, 0.15] (daily-equiv): lowest σ wins."""
    df = _gaussian_returns(sigmas=[0.10, 0.20, 0.15], rho=0.3, n=750, seed=11)
    res = min_variance(df, max_w=0.80, min_w=0.0, shrinkage="sample")
    w = res["weights"]
    # Asset a0 (σ=0.10) should have the largest weight by ≥ 5pp over the others.
    assert w["a0"] > w["a1"] + 0.05
    assert w["a0"] > w["a2"] + 0.05
    # Weights sum to 1 (within numerical tolerance).
    assert abs(sum(w.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. ERC invariant — equal risk contributions
# ---------------------------------------------------------------------------


def test_risk_parity_equalises_risk_contributions() -> None:
    """At ERC optimum, wᵢ·(Σw)ᵢ should be approximately constant (CV<5%)."""
    df = _gaussian_returns(sigmas=[0.08, 0.18, 0.12, 0.22], rho=0.25, n=1000, seed=3)
    res = risk_parity_erc(df, max_w=0.50, min_w=0.0, shrinkage="sample")
    w = np.array([res["weights"][c] for c in df.columns])
    cov = np.cov(df.to_numpy(), rowvar=False, ddof=1)
    sigma_w = cov @ w
    rc = w * sigma_w
    cv = float(rc.std(ddof=1) / abs(rc.mean()))
    assert cv < 0.05, f"ERC CV={cv:.4f} should be < 0.05"


# ---------------------------------------------------------------------------
# 3. HRP handles singular covariance
# ---------------------------------------------------------------------------


def test_hrp_handles_singular_covariance() -> None:
    """Build a deliberately rank-deficient sample (duplicate column) — HRP must
    not raise and weights must sum to 1."""
    df = _gaussian_returns(sigmas=[0.10, 0.20, 0.15], rho=0.2, n=400, seed=42)
    # Force singularity: duplicate a0 into a3.
    df["a3"] = df["a0"]
    res = hrp(df, shrinkage="sample")
    w = res["weights"]
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-6, f"weights sum {total} ≠ 1"
    # All weights should be in [0, 1].
    for v in w.values():
        assert 0.0 <= v <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 4. Optimiser methods beat the equal-weight baseline in-sample
# ---------------------------------------------------------------------------


def test_max_sharpe_beats_equal_weight_in_sample() -> None:
    """In-sample, max-sharpe should achieve a Sharpe ≥ equal-weight (with no
    shrinkage on μ so the optimiser actually exploits sample-mean differences)."""
    # Asymmetric mu: a0 is clearly the best alpha.
    df = _gaussian_returns(
        sigmas=[0.10, 0.18, 0.14],
        rho=0.20,
        mu=[0.0010, 0.0001, 0.0003],  # daily means
        n=1000,
        seed=99,
    )
    eq = equal_weight(df, rf=0.0, shrinkage="sample")
    ms = mean_variance_max_sharpe(
        df, rf=0.0, max_w=0.99, min_w=0.0, shrink_mu=0.0, shrinkage="sample"
    )
    assert ms["sharpe"] >= eq["sharpe"] - 1e-6


# ---------------------------------------------------------------------------
# 5. Constraints binding — max_w respected
# ---------------------------------------------------------------------------


def test_max_weight_constraint_is_respected() -> None:
    """SLSQP must honour max_w in the *feasible* regime (max_w × N ≥ 1)."""
    df = _gaussian_returns(sigmas=[0.10, 0.20, 0.15], rho=0.3, n=600, seed=21)
    # 3 assets, max_w=0.50 → cap × N = 1.5 ≥ 1 (feasible). Cap MUST hold tightly.
    res = min_variance(df, max_w=0.50, min_w=0.0, shrinkage="sample")
    weights = list(res["weights"].values())
    assert abs(sum(weights) - 1.0) < 1e-6
    for v in weights:
        assert v <= 0.50 + 1e-6, f"weight {v} exceeded cap 0.50"

    # 4 assets, max_w=0.30: feasible (0.30×4=1.2). Cap should hold.
    df4 = _gaussian_returns(sigmas=[0.08, 0.12, 0.16, 0.20], rho=0.2, n=600, seed=22)
    res4 = min_variance(df4, max_w=0.30, min_w=0.0, shrinkage="sample")
    weights4 = list(res4["weights"].values())
    assert abs(sum(weights4) - 1.0) < 1e-6
    for v in weights4:
        assert v <= 0.30 + 1e-6, f"weight {v} exceeded cap 0.30"


# ---------------------------------------------------------------------------
# 6. Efficient frontier monotonicity
# ---------------------------------------------------------------------------


def test_efficient_frontier_return_monotonic_in_vol() -> None:
    """As frontier vol increases, expected return should be non-decreasing."""
    df = _gaussian_returns(
        sigmas=[0.08, 0.16, 0.12, 0.20],
        rho=0.15,
        mu=[0.0002, 0.0008, 0.0004, 0.0010],
        n=800,
        seed=55,
    )
    pts = efficient_frontier(df, n_points=20, max_w=0.60, min_w=0.0, rf=0.0, shrinkage="sample")
    assert len(pts) >= 2
    rets = [p["expected_return"] for p in pts]
    # Allow small SLSQP wobble: each step ≥ -ε.
    eps = 1e-3
    for i in range(1, len(rets)):
        assert rets[i] >= rets[i - 1] - eps, (
            f"non-monotonic frontier at i={i}: {rets[i - 1]} → {rets[i]}"
        )


# ---------------------------------------------------------------------------
# 7. Equal-weight returns + structure
# ---------------------------------------------------------------------------


def test_equal_weight_basic_shape() -> None:
    df = _gaussian_returns(sigmas=[0.10, 0.20, 0.15], rho=0.3, n=300, seed=1)
    res = equal_weight(df, rf=0.0, shrinkage="sample")
    w = res["weights"]
    for v in w.values():
        assert abs(v - 1 / 3) < 1e-9
    assert "marginal_risk_contribution" in res
    assert "diversification_ratio" in res
    assert "effective_n" in res
    # Effective N should be exactly 3 for equal-weight.
    assert abs(res["effective_n"] - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# 8. MC drawdown returns sensible quantiles
# ---------------------------------------------------------------------------


def test_monte_carlo_drawdown_quantiles_ordered() -> None:
    df = _gaussian_returns(sigmas=[0.10, 0.20, 0.15], rho=0.3, n=300, seed=4)
    weights = {"a0": 0.5, "a1": 0.25, "a2": 0.25}
    out = monte_carlo_drawdown(weights, df, n_paths=2000, horizon_days=126, block=10)
    assert out["p05"] <= out["p50"] <= out["p95"]
    assert out["p05"] >= 0.0  # drawdown is non-negative as fraction
    assert out["n_paths"] == 2000
    assert out["horizon_days"] == 126


# ---------------------------------------------------------------------------
# 9. Endpoint smoke test
# ---------------------------------------------------------------------------


@pytest.fixture
def optimizer_app() -> TestClient:
    """Tiny app that mounts ONLY the optimizer router (no main.py imports)."""
    app = FastAPI()
    app.include_router(optimizer_router)
    return TestClient(app)


def test_endpoint_optimize_hrp_smoke(optimizer_app: TestClient) -> None:
    """POST /strategies/optimize with 3 dummy pair_ids and method=hrp."""
    body = {
        "pair_ids": ["dummy_a__dummy_b", "dummy_c__dummy_d", "dummy_e__dummy_f"],
        "method": "hrp",
        "lookback_days": 252,
        "risk_free_rate": 0.045,
        "max_weight": 0.50,
        "min_weight": 0.0,
        "shrinkage": "ledoit_wolf",
        "mc_paths": 500,
        "mc_horizon_days": 126,
        "return_frontier": True,
        "seed": 42,
    }
    r = optimizer_app.post("/strategies/optimize", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["method"] == "hrp"
    assert set(payload["weights"].keys()) == set(body["pair_ids"])
    total_w = sum(payload["weights"].values())
    assert abs(total_w - 1.0) < 1e-6
    assert payload["frontier"] is not None
    assert len(payload["frontier"]) >= 2
    assert payload["mc_drawdown"]["p05"] <= payload["mc_drawdown"]["p95"]
    # synthetic_returns warning should be present.
    assert any("synthetic_returns" in w for w in payload["warnings"])


def test_endpoint_rejects_duplicate_pair_ids(optimizer_app: TestClient) -> None:
    body = {
        "pair_ids": ["x__y", "x__y"],
        "method": "hrp",
    }
    r = optimizer_app.post("/strategies/optimize", json=body)
    assert r.status_code == 422


def test_endpoint_rejects_too_few_pair_ids(optimizer_app: TestClient) -> None:
    body = {"pair_ids": ["only_one"], "method": "hrp"}
    r = optimizer_app.post("/strategies/optimize", json=body)
    assert r.status_code == 422
