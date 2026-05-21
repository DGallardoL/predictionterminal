"""Tests for the enriched ``/fit`` response + ``/factors/suggest-for-ticker``.

Covers the additive fields shipped 2026-05-15:
    rolling_betas_ci, oos_r_squared, residual_annotations,
    factor_correlation_matrix, pca_summary, next_step_hint.

Plus the new smart-factor-picker endpoint
``POST /factors/suggest-for-ticker``.

Each test uses a synthetic DGP where we control the answer end-to-end
(no live Polymarket / yfinance calls).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import pfm.main as main_mod
from pfm.regression_router import (
    _compute_pca_summary,
    _compute_rolling_betas_with_ci,
    _factor_correlation_matrix,
    _next_step_hint,
    _residual_annotations,
    _walk_forward_oos_r2,
)
from pfm.schemas import FactorEstimateOut

# ---------------------------------------------------------------------------
# next_step_hint — every verdict bucket maps to a non-empty actionable string.
# ---------------------------------------------------------------------------


class TestNextStepHint:
    def test_every_verdict_has_a_hint(self) -> None:
        for v in (
            "well_specified",
            "weak_fit",
            "collinear",
            "underpowered",
            "borderline",
        ):
            hint = _next_step_hint(v)
            assert isinstance(hint, str) and len(hint) >= 10, v

    def test_unknown_verdict_returns_empty(self) -> None:
        assert _next_step_hint("nonsense_value") == ""


# ---------------------------------------------------------------------------
# rolling_betas_ci — synthetic DGP with time-varying β must show variation.
# ---------------------------------------------------------------------------


class TestRollingBetasCi:
    def test_skipped_when_n_below_floor(self) -> None:
        idx = pd.date_range("2025-01-01", periods=80, freq="D")
        X = pd.DataFrame({"f": np.linspace(-1, 1, 80)}, index=idx)
        y = pd.Series(np.linspace(-1, 1, 80), index=idx)
        # n=80 < 90 floor.
        assert _compute_rolling_betas_with_ci(y, X) == {}

    def test_recovers_time_varying_beta(self) -> None:
        # β switches from +0.5 to -0.5 at the midpoint. Rolling β must
        # show that swing in its time series.
        rng = np.random.default_rng(seed=11)
        n = 240
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = rng.normal(0, 0.30, n)
        beta_t = np.where(np.arange(n) < n // 2, 0.50, -0.50)
        eps = rng.normal(0, 0.005, n)
        y = pd.Series(beta_t * x + eps, index=idx, name="r")
        X = pd.DataFrame({"f": x}, index=idx)
        out = _compute_rolling_betas_with_ci(y, X, window=60)
        assert "f" in out
        betas = [p.beta for p in out["f"]]
        assert max(betas) > 0.30
        assert min(betas) < -0.30

    def test_downsamples_to_max_points_per_factor(self) -> None:
        rng = np.random.default_rng(seed=3)
        n = 600
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        x = rng.normal(0, 0.2, n)
        y = pd.Series(0.3 * x + rng.normal(0, 0.01, n), index=idx, name="r")
        X = pd.DataFrame({"f": x}, index=idx)
        out = _compute_rolling_betas_with_ci(y, X, window=60, max_points_per_factor=50)
        # Stride enforcement keeps the count below the cap (with some slack).
        assert len(out["f"]) <= 60

    def test_ci_lo_lt_beta_lt_ci_hi(self) -> None:
        rng = np.random.default_rng(seed=2)
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = rng.normal(0, 0.20, n)
        y = pd.Series(0.40 * x + rng.normal(0, 0.02, n), index=idx, name="r")
        X = pd.DataFrame({"f": x}, index=idx)
        out = _compute_rolling_betas_with_ci(y, X)
        assert out["f"], "expected non-empty rolling betas"
        for p in out["f"]:
            assert p.ci_lo <= p.beta <= p.ci_hi


# ---------------------------------------------------------------------------
# oos_r_squared — overfit DGP must show OOS R² < in-sample R².
# ---------------------------------------------------------------------------


class TestWalkForwardOos:
    def test_returns_skipped_block_for_short_window(self) -> None:
        # 2026-05-15 rigour pack: the helper now returns an explicit
        # OosRSquaredSkipped block carrying ``skipped_reason`` instead of
        # silently returning None when n_obs < 100. Old behaviour was a
        # silent null which was invisible in the UI.
        from pfm.schemas import OosRSquaredSkipped

        idx = pd.date_range("2025-01-01", periods=80, freq="D")
        X = pd.DataFrame({"f": np.linspace(-1, 1, 80)}, index=idx)
        y = pd.Series(np.linspace(-1, 1, 80), index=idx)
        out = _walk_forward_oos_r2(y, X)
        assert isinstance(out, OosRSquaredSkipped)
        assert out.value is None
        assert out.skipped is True
        assert "n_obs=80" in out.skipped_reason
        assert "min_n_for_walk_forward=100" in out.skipped_reason

    def test_structural_break_oos_collapses(self) -> None:
        # In-sample fit is great because the model averages over both
        # regimes. Walk-forward OOS R² should be much lower because each
        # train fold sees regime A, test fold sees regime B (or vice
        # versa).
        rng = np.random.default_rng(seed=29)
        n = 250
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = rng.normal(0, 0.30, n)
        beta_t = np.where(np.arange(n) < n // 2, 0.80, -0.80)
        y = pd.Series(beta_t * x + rng.normal(0, 0.005, n), index=idx, name="r")
        X = pd.DataFrame({"f": x}, index=idx)
        from pfm.model import fit_ols_hac

        in_sample = fit_ols_hac(y, X)
        oos = _walk_forward_oos_r2(y, X)
        assert oos is not None
        # The structural break tanks OOS — value is well below in-sample.
        assert oos.value < in_sample.stats.r_squared
        assert oos.fold_count >= 2
        assert len(oos.per_fold) == oos.fold_count

    def test_clean_dgp_oos_close_to_in_sample(self) -> None:
        rng = np.random.default_rng(seed=4)
        n = 250
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x1 = rng.normal(0, 0.20, n)
        x2 = rng.normal(0, 0.30, n)
        y = pd.Series(
            0.30 * x1 - 0.20 * x2 + rng.normal(0, 0.005, n),
            index=idx,
            name="r",
        )
        X = pd.DataFrame({"f1": x1, "f2": x2}, index=idx)
        oos = _walk_forward_oos_r2(y, X)
        assert oos is not None
        # Clean DGP → OOS R² should be high (>= 0.5).
        assert oos.value > 0.5


# ---------------------------------------------------------------------------
# residual_annotations — top-|e_t| dates must include the planted outlier.
# ---------------------------------------------------------------------------


class TestResidualAnnotations:
    def test_flags_planted_outlier(self) -> None:
        rng = np.random.default_rng(seed=44)
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = rng.normal(0, 0.25, n)
        y_clean = 0.40 * x + rng.normal(0, 0.005, n)
        # Plant a single 50-sigma residual on day 60.
        y_clean[60] += 0.50
        y = pd.Series(y_clean, index=idx, name="r")
        X = pd.DataFrame({"f": x}, index=idx)
        # Residual = y - beta·x where beta ≈ 0.40
        residual = y - 0.40 * x
        ests = [
            FactorEstimateOut(
                id="f",
                beta=0.40,
                std_err=0.01,
                t_stat=40.0,
                p_value=0.0,
                ci_low=0.38,
                ci_high=0.42,
            )
        ]
        annos = _residual_annotations(y, X, residual, ests, top_k=5)
        # The outlier date must be among the top-5.
        outlier_date = idx[60].date()
        assert any(a.date == outlier_date for a in annos)

    def test_top_factor_is_largest_abs_contribution(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        X = pd.DataFrame(
            {"a": [0.1] * 10, "b": [-1.0] * 10},  # b dominates in magnitude
            index=idx,
        )
        y = pd.Series([0.0] * 10, index=idx)
        residual = pd.Series([0.5] * 10, index=idx)
        ests = [
            FactorEstimateOut(
                id="a",
                beta=0.5,
                std_err=0.0,
                t_stat=0.0,
                p_value=0.0,
                ci_low=0.0,
                ci_high=0.0,
            ),
            FactorEstimateOut(
                id="b",
                beta=0.3,
                std_err=0.0,
                t_stat=0.0,
                p_value=0.0,
                ci_low=0.0,
                ci_high=0.0,
            ),
        ]
        annos = _residual_annotations(y, X, residual, ests, top_k=1)
        assert annos
        # |0.1*0.5|=0.05 vs |-1.0*0.3|=0.30 → 'b' is the larger contributor.
        assert annos[0].top_factor == "b"


# ---------------------------------------------------------------------------
# factor_correlation_matrix — collinear factors must show high r.
# ---------------------------------------------------------------------------


class TestFactorCorrelationMatrix:
    def test_collinear_factors_show_high_r(self) -> None:
        rng = np.random.default_rng(seed=5)
        n = 200
        x1 = rng.normal(0, 1, n)
        x2 = x1 + rng.normal(0, 0.05, n)  # near-perfect copy
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        X = pd.DataFrame({"a": x1, "b": x2}, index=idx)
        m = _factor_correlation_matrix(X)
        assert "a" in m and "b" in m
        assert m["a"]["b"] > 0.95
        assert m["a"]["a"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_empty_for_single_factor(self) -> None:
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        X = pd.DataFrame({"a": np.linspace(0, 1, 50)}, index=idx)
        assert _factor_correlation_matrix(X) == {}

    def test_caps_at_max_factors(self) -> None:
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {f"f{i}": rng.normal(0, 1, 50) for i in range(40)},
            index=idx,
        )
        m = _factor_correlation_matrix(X, max_factors=10)
        assert len(m) == 10


# ---------------------------------------------------------------------------
# pca_summary — explained-variance-ratio sums to <=1, top loadings present.
# ---------------------------------------------------------------------------


class TestPcaSummary:
    def test_returns_none_for_single_factor(self) -> None:
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        X = pd.DataFrame({"a": np.linspace(0, 1, 50)}, index=idx)
        assert _compute_pca_summary(X) is None

    def test_explained_variance_sums_to_one(self) -> None:
        rng = np.random.default_rng(seed=7)
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        X = pd.DataFrame(
            {"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)},
            index=idx,
        )
        pca = _compute_pca_summary(X, max_components=3)
        assert pca is not None
        assert pca.n_components == 3
        assert sum(pca.explained_variance_ratio) == pytest.approx(1.0, abs=1e-3)
        for k in range(pca.n_components):
            assert k in pca.top_loadings
            assert pca.top_loadings[k]


# ---------------------------------------------------------------------------
# /fit response — new fields must be present and correctly shaped.
# ---------------------------------------------------------------------------


class TestFitEnrichedResponseShape:
    def _post(self, app_client: TestClient, **overrides: Any) -> dict[str, Any]:
        payload = {
            "ticker": "TEST",
            "factors": ["factor_a", "factor_b"],
            "start": "2025-06-01",
            "end": "2025-12-31",
        }
        payload.update(overrides)
        r = app_client.post("/fit", json=payload)
        assert r.status_code == 200, r.text
        return r.json()

    def test_all_new_fields_present(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        for key in (
            "rolling_betas_ci",
            "oos_r_squared",
            "residual_annotations",
            "factor_correlation_matrix",
            "pca_summary",
            "next_step_hint",
        ):
            assert key in body, key

    def test_next_step_hint_matches_verdict_table(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        verdict = body["verdict"]
        if verdict in {"well_specified", "weak_fit", "collinear", "underpowered", "borderline"}:
            assert body["next_step_hint"]
            # well_specified → mentions OOS validation.
            if verdict == "well_specified":
                assert "OOS" in body["next_step_hint"] or "walk-forward" in body["next_step_hint"]

    def test_factor_correlation_matrix_is_symmetric(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        m = body["factor_correlation_matrix"]
        if not m:
            pytest.skip("only one factor in fit — matrix is empty by design")
        keys = list(m.keys())
        for a in keys:
            for b in keys:
                assert m[a][b] == pytest.approx(m[b][a], abs=1e-9)

    def test_residual_annotations_capped_at_5(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        assert len(body["residual_annotations"]) <= 5

    def test_oos_field_only_when_n_obs_sufficient(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        if body["n_obs"] >= 100:
            assert body["oos_r_squared"] is not None
            assert body["oos_r_squared"]["fold_count"] >= 2

    def test_rolling_betas_ci_shape(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        rb = body["rolling_betas_ci"]
        if body["n_obs"] >= 90:
            # Either populated per factor (dict of lists) or empty if HAC failed.
            assert isinstance(rb, dict)
            for points in rb.values():
                for p in points:
                    assert {"date", "beta", "ci_lo", "ci_hi"} <= set(p.keys())
                    assert p["ci_lo"] <= p["beta"] <= p["ci_hi"]


# ---------------------------------------------------------------------------
# /factors/suggest-for-ticker — DGP with planted correlation must rank.
# ---------------------------------------------------------------------------


class TestSuggestForTicker:
    def test_returns_top_k_sorted_by_abs_r(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        # Build a hand-tuned ticker series correlated with factor_a but
        # independent of factor_b. The picker should rank factor_a above
        # factor_b.
        rng_idx = pd.date_range("2025-06-01", "2025-12-31", freq="B", tz="UTC")
        n = len(rng_idx)
        t = np.arange(n) / n
        # factor_a fixture is ~ sin(2π·1.2·t); make returns oppose it
        # (so |r| is large) and add small noise.
        rng = np.random.default_rng(seed=2026)
        factor_a_series = (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)
        # Approximate the post-Δlogit Δ-like signal with the price differences;
        # the picker computes r vs Δlogit so we just ensure a strong
        # monotonic relationship.
        synthetic_returns = -np.diff(factor_a_series, prepend=factor_a_series[0]) * 5.0
        synthetic_returns += rng.normal(0, 0.001, n)

        def _fake_returns(
            ticker: str,
            start,
            end,
            return_type: str = "log",
        ) -> pd.Series:
            idx = pd.date_range(start, end, freq="B", tz="UTC")
            # Build deterministic series that overlaps with the factor window.
            n_local = len(idx)
            tt = np.arange(n_local) / max(1, n_local)
            fa = (0.30 + 0.30 * np.sin(2 * np.pi * tt * 1.2)).clip(0.05, 0.95)
            r = -np.diff(fa, prepend=fa[0]) * 5.0
            r += rng.normal(0, 0.001, n_local)
            s = pd.Series(r, index=idx, name="r")
            s.index = pd.to_datetime(s.index, utc=True).normalize()
            return s

        monkeypatch.setattr(main_mod, "get_log_returns", _fake_returns)

        r = app_client.post(
            "/factors/suggest-for-ticker",
            json={
                "ticker": "TEST",
                "lookback_days": 90,
                "top_k": 5,
                "min_n_obs": 20,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ticker"] == "TEST"
        assert body["lookback_days"] == 90
        assert isinstance(body["top_factors"], list)
        # |r| sorted descending.
        abs_rs = [it["abs_r"] for it in body["top_factors"]]
        assert abs_rs == sorted(abs_rs, reverse=True)
        # factor_a should be ranked above factor_b given the planted
        # correlation. Both should appear (only 2 in the test fixture).
        ids = [it["factor_id"] for it in body["top_factors"]]
        if "factor_a" in ids and "factor_b" in ids:
            assert ids.index("factor_a") < ids.index("factor_b")

    def test_caches_per_ticker_lookback(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        # Two identical requests should hit the cache on the second call.
        # We instrument the underlying scan function and assert it runs
        # exactly once.
        from pfm import regression_router as rr

        original = rr._scan_factor_correlations_for_ticker
        call_count = {"n": 0}

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            rr,
            "_scan_factor_correlations_for_ticker",
            _wrapped,
        )

        body = {
            "ticker": "CACHE",
            "lookback_days": 90,
            "top_k": 3,
            "min_n_obs": 10,
        }
        r1 = app_client.post("/factors/suggest-for-ticker", json=body)
        r2 = app_client.post("/factors/suggest-for-ticker", json=body)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert call_count["n"] == 1, "second call should hit the cache"
        assert r1.json() == r2.json()

    def test_invalid_ticker_returns_502(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        from pfm.sources.equity import EquityDataError

        def _fail(ticker: str, start, end, return_type: str = "log") -> pd.Series:
            raise EquityDataError(f"all equity sources failed for {ticker!r}")

        monkeypatch.setattr(main_mod, "get_log_returns", _fail)
        r = app_client.post(
            "/factors/suggest-for-ticker",
            json={"ticker": "XXXXXX", "lookback_days": 60, "top_k": 5},
        )
        assert r.status_code == 502

    def test_top_k_bounds_are_enforced(self, app_client: TestClient) -> None:
        # top_k=0 must fail validation.
        r = app_client.post(
            "/factors/suggest-for-ticker",
            json={"ticker": "TEST", "lookback_days": 90, "top_k": 0},
        )
        assert r.status_code == 422
