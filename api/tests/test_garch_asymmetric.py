"""Tests for asymmetric GARCH models (GJR-GARCH, EGARCH) and the
model-selection helper, plus a router smoke test under TestClient.

The math tests use synthetic DGPs where the true parameters are known so
recovery can be measured. The endpoint test uses a stand-alone FastAPI app
with only :mod:`pfm.garch_router` mounted (we do not touch ``main.py``)
and patches the equity fetcher so no network IO is required.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.garch import compare_garch_models, fit_egarch_11, fit_gjr_garch_11

# ---------------------------------------------------------------------------
# Synthetic DGPs
# ---------------------------------------------------------------------------


def _gjr_dgp(
    n: int,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    seed: int = 42,
) -> pd.Series:
    """Generate a GJR-GARCH(1,1) return path. Returns are demeaned by
    construction (ε_t innovation series)."""
    rng = np.random.default_rng(seed)
    eps = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / max(1.0 - alpha - 0.5 * gamma - beta, 0.05)
    eps[0] = math.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        e_prev = eps[t - 1]
        ind = 1.0 if e_prev < 0.0 else 0.0
        sigma2[t] = (
            omega + alpha * e_prev * e_prev + gamma * ind * e_prev * e_prev + beta * sigma2[t - 1]
        )
        eps[t] = math.sqrt(max(sigma2[t], 1e-12)) * rng.standard_normal()
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(eps, index=idx, name="r")


def _egarch_dgp(
    n: int,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    seed: int = 7,
) -> pd.Series:
    """Generate an EGARCH(1,1) return path."""
    rng = np.random.default_rng(seed)
    eps = np.empty(n)
    log_sigma2 = np.empty(n)
    log_sigma2[0] = math.log(0.0001)  # initial unconditional log-var
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    sigma_prev = math.sqrt(math.exp(log_sigma2[0]))
    eps[0] = sigma_prev * rng.standard_normal()
    for t in range(1, n):
        z_prev = eps[t - 1] / max(sigma_prev, 1e-9)
        log_sigma2[t] = (
            omega + alpha * (abs(z_prev) - sqrt_2_pi) + gamma * z_prev + beta * log_sigma2[t - 1]
        )
        sigma_prev = math.sqrt(math.exp(log_sigma2[t]))
        eps[t] = sigma_prev * rng.standard_normal()
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(eps, index=idx, name="r")


# ---------------------------------------------------------------------------
# GJR-GARCH recovery tests
# ---------------------------------------------------------------------------


class TestGJRGARCH:
    def test_gjr_recovery_on_asymmetric_dgp(self) -> None:
        """Simulate γ=0.10 GJR DGP and check recovery within 50% tolerance."""
        returns = _gjr_dgp(
            n=4000,
            omega=2e-5,
            alpha=0.05,
            gamma=0.10,
            beta=0.85,
            seed=11,
        )
        out = fit_gjr_garch_11(returns)
        assert out["converged"] in (True, False)
        # γ recovered with the right sign and within 50%.
        gamma_est = out["gamma_leverage"]
        assert gamma_est > 0.02, f"γ collapsed to ~0: {gamma_est}"
        assert abs(gamma_est - 0.10) / 0.10 < 0.50, f"γ={gamma_est} far from 0.10"
        # Persistence < 1
        assert out["persistence"] < 1.0
        # Half-life sane
        assert out["half_life_vol_days"] > 0.0
        # Diagnostics keys are present and have right types.
        for key in (
            "omega",
            "alpha",
            "gamma_leverage",
            "beta",
            "log_likelihood",
            "aic",
            "bic",
            "ljung_box_p",
            "leverage_effect_significant",
            "asymmetry_t_stat",
        ):
            assert key in out
        assert isinstance(out["leverage_effect_significant"], bool)
        assert len(out["conditional_variance"]) == out["n_obs"]
        assert len(out["standardized_residuals"]) == out["n_obs"]

    def test_gjr_collapses_to_zero_on_symmetric_dgp(self) -> None:
        """Symmetric GARCH(1,1) DGP — γ should be ≈ 0 (within 0.06)."""
        # Symmetric is gamma=0
        returns = _gjr_dgp(
            n=3000,
            omega=2e-5,
            alpha=0.10,
            gamma=0.0,
            beta=0.85,
            seed=21,
        )
        out = fit_gjr_garch_11(returns)
        assert abs(out["gamma_leverage"]) < 0.06
        # Symmetric DGP — leverage should not be flagged as significant.
        assert out["leverage_effect_significant"] is False

    def test_gjr_persistence_below_one(self) -> None:
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.normal(0, 0.01, 600))
        out = fit_gjr_garch_11(returns)
        assert out["persistence"] < 1.0

    def test_gjr_zero_variance_raises(self) -> None:
        flat = pd.Series([0.0] * 200)
        with pytest.raises(ValueError, match="zero variance"):
            fit_gjr_garch_11(flat)

    def test_gjr_too_few_obs_raises(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 30))
        with pytest.raises(ValueError, match=">=50"):
            fit_gjr_garch_11(returns)

    def test_gjr_student_t_distribution_runs(self) -> None:
        returns = _gjr_dgp(n=1500, omega=2e-5, alpha=0.05, gamma=0.08, beta=0.86, seed=33)
        out = fit_gjr_garch_11(returns, distribution="t")
        assert out["distribution"] == "t"
        assert out["persistence"] < 1.0
        assert out["half_life_vol_days"] > 0.0


# ---------------------------------------------------------------------------
# EGARCH recovery tests
# ---------------------------------------------------------------------------


class TestEGARCH:
    def test_egarch_negative_gamma_recovery(self) -> None:
        """Equity-style DGP with γ<0 — recover the negative sign."""
        returns = _egarch_dgp(
            n=3000,
            omega=-0.10,
            alpha=0.15,
            gamma=-0.10,
            beta=0.96,
            seed=55,
        )
        out = fit_egarch_11(returns)
        assert out["gamma_leverage"] < 0.0, f"expected γ<0, got {out['gamma_leverage']}"
        assert out["leverage_negative"] is True
        assert out["persistence"] < 1.0
        assert out["half_life_log_vol_days"] > 0.0
        for key in (
            "omega",
            "alpha",
            "gamma_leverage",
            "beta",
            "log_likelihood",
            "aic",
            "bic",
            "leverage_negative",
        ):
            assert key in out
        assert len(out["conditional_variance"]) == out["n_obs"]

    def test_egarch_positive_gamma_dgp(self) -> None:
        """Symmetric / mildly positive γ DGP — sign should not be forced negative."""
        returns = _egarch_dgp(
            n=2000,
            omega=-0.05,
            alpha=0.10,
            gamma=0.05,
            beta=0.95,
            seed=77,
        )
        out = fit_egarch_11(returns)
        # Should recover positive sign or at least not anchor at -0.10.
        assert out["gamma_leverage"] > -0.03

    def test_egarch_zero_variance_raises(self) -> None:
        flat = pd.Series([0.0] * 200)
        with pytest.raises(ValueError, match="zero variance"):
            fit_egarch_11(flat)

    def test_egarch_persistence_below_one(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 600))
        out = fit_egarch_11(returns)
        assert out["persistence"] < 1.0
        assert out["half_life_log_vol_days"] > 0.0


# ---------------------------------------------------------------------------
# Compare-models helper
# ---------------------------------------------------------------------------


class TestCompareGarchModels:
    def test_best_aic_picks_asymmetric_on_asymmetric_dgp(self) -> None:
        """A strong leverage-effect DGP should reward GJR or EGARCH over GARCH."""
        returns = _gjr_dgp(
            n=4000,
            omega=2e-5,
            alpha=0.04,
            gamma=0.15,
            beta=0.85,
            seed=101,
        )
        out = compare_garch_models(returns)
        assert out["best_model_aic"] in ("gjr11", "egarch11")
        # 3 entries
        models = {c["model"] for c in out["comparisons"]}
        assert models == {"garch11", "gjr11", "egarch11"}
        # All AIC/BIC finite for these well-conditioned data.
        for c in out["comparisons"]:
            assert math.isfinite(c["aic"])
            assert math.isfinite(c["bic"])
            assert math.isfinite(c["persistence"])
            assert c["persistence"] < 1.0

    def test_unknown_model_rejected(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 200))
        with pytest.raises(ValueError, match="unknown models"):
            compare_garch_models(returns, models=["garch11", "foo_bar"])

    def test_subset_of_models_runs(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 400))
        out = compare_garch_models(returns, models=["garch11", "egarch11"])
        assert len(out["comparisons"]) == 2
        assert out["best_model_aic"] in ("garch11", "egarch11")
        assert out["best_model_bic"] in ("garch11", "egarch11")


# ---------------------------------------------------------------------------
# Router smoke test (no real network IO; equity fetcher is patched)
# ---------------------------------------------------------------------------


@pytest.fixture
def vol_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient with only `pfm.garch_router` mounted and equity patched."""
    from pfm import garch_router as gr

    rng = np.random.default_rng(2024)
    n = 500
    fake = pd.Series(
        rng.normal(0, 0.01, n),
        index=pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC"),
        name="r",
    )

    def _fake_fetch(
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        return fake.copy()

    monkeypatch.setattr(gr, "_fetch_returns", _fake_fetch)

    reset_caches()
    app = FastAPI()
    app.include_router(gr.router)
    with TestClient(app) as client:
        yield client
    reset_caches()


def _post(client: TestClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    return r.json()


class TestEndpoints:
    def test_gjr_endpoint(self, vol_client: TestClient) -> None:
        out = _post(
            vol_client,
            "/vol/gjr-garch",
            {"ticker": "SPY", "start": "2022-01-03", "end": "2024-01-03"},
        )
        assert out["ticker"] == "SPY"
        assert "gamma_leverage" in out
        assert "asymmetry_t_stat" in out
        assert isinstance(out["leverage_effect_significant"], bool)
        assert out["persistence"] < 1.0

    def test_egarch_endpoint(self, vol_client: TestClient) -> None:
        out = _post(
            vol_client,
            "/vol/egarch",
            {"ticker": "AAPL", "start": "2022-01-03", "end": "2024-01-03"},
        )
        assert out["ticker"] == "AAPL"
        assert "leverage_negative" in out
        assert isinstance(out["leverage_negative"], bool)
        assert out["persistence"] < 1.0

    def test_compare_endpoint(self, vol_client: TestClient) -> None:
        out = _post(
            vol_client,
            "/vol/garch-compare",
            {
                "ticker": "QQQ",
                "start": "2022-01-03",
                "end": "2024-01-03",
                "models": ["garch11", "gjr11", "egarch11"],
            },
        )
        assert out["ticker"] == "QQQ"
        assert out["best_model_aic"] in ("garch11", "gjr11", "egarch11")
        assert out["best_model_bic"] in ("garch11", "gjr11", "egarch11")
        assert len(out["comparisons"]) == 3

    def test_compare_endpoint_rejects_bad_dates(self, vol_client: TestClient) -> None:
        r = vol_client.post(
            "/vol/garch-compare",
            json={
                "ticker": "QQQ",
                "start": "2024-01-03",
                "end": "2022-01-03",
                "models": ["garch11"],
            },
        )
        assert r.status_code == 422
