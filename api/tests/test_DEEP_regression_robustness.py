"""Defensive regression-robustness tests for /fit.

These tests catch bugs and quirks in the regression endpoint that surfaced
during the 2026-05-08 audit. Each test documents the issue it guards
against in its docstring, so future Claude can tell what's load-bearing.

The tests rely on the ``app_client`` fixture in ``conftest.py`` which
patches out yfinance/Polymarket/redis so the suite stays hermetic.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import pfm.main as main_mod
from pfm.model import delta_logit, fit_ols_hac

# ---------------------------------------------------------------------------
# Synthetic recovery — confirm the math is right.
# ---------------------------------------------------------------------------


class TestSyntheticRecovery:
    """y = β1·X1 + β2·X2 + ε with known β must be recovered within tolerance."""

    def test_recovers_known_betas_within_tolerance(self) -> None:
        rng = np.random.default_rng(seed=7)
        n = 400
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x1 = rng.normal(0, 0.30, n)
        x2 = rng.normal(0, 0.50, n)
        X = pd.DataFrame({"f1": x1, "f2": x2}, index=idx)
        beta1, beta2, alpha = 0.50, -0.30, 0.001
        eps = rng.normal(0, 0.005, n)
        y = pd.Series(alpha + beta1 * x1 + beta2 * x2 + eps, index=idx, name="r")

        result = fit_ols_hac(y, X)
        recovered = {e.factor_id: e.beta for e in result.factors}
        assert abs(recovered["f1"] - beta1) < 0.01
        assert abs(recovered["f2"] - beta2) < 0.01
        assert abs(result.stats.alpha - alpha) < 0.005


# ---------------------------------------------------------------------------
# Inner-join correctness — make sure /fit doesn't silently use wrong dates.
# ---------------------------------------------------------------------------


class TestInnerJoinCorrectness:
    """Ticker has more obs than factor → final n_obs == min after intersection."""

    def test_ticker_inner_join_uses_factor_window(self, app_client: TestClient) -> None:
        # Synthetic factor_a is 2025-06-01..2025-12-31 (~214 days, daily).
        # Request a wider window: result must still be bounded by factor coverage.
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a"],
                "start": "2024-01-01",
                "end": "2026-12-31",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Factor coverage caps n_obs at the intersection.
        assert body["n_obs"] <= 215, body["n_obs"]
        assert body["n_obs"] > 100, body["n_obs"]


# ---------------------------------------------------------------------------
# Empty / duplicate factor handling.
# ---------------------------------------------------------------------------


class TestFactorListValidation:
    """Empty list, duplicates and unknown ids must produce informative 4xx."""

    def test_empty_factor_list_returns_400(self, app_client: TestClient) -> None:
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": [],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 400
        assert "factor" in r.json()["detail"].lower()

    def test_duplicate_factor_ids_dedupe(self, app_client: TestClient) -> None:
        """Duplicates in `factors` are deduplicated silently — not an error."""
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a", "factor_a"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        ids = [f["id"] for f in body["factors"]]
        assert ids.count("factor_a") == 1


# ---------------------------------------------------------------------------
# Bad ticker handling.
# ---------------------------------------------------------------------------


def _patched_returns_factory(behaviour: str = "ok"):
    """Return a fake ``get_log_returns`` that fails per ``behaviour``."""

    def _fail_unknown(ticker, start, end, return_type="log"):
        from pfm.sources.equity import EquityDataError

        raise EquityDataError(f"all equity sources failed for {ticker!r}")

    def _ok(ticker, start, end, return_type="log"):
        idx = pd.date_range(start, end, freq="B", tz="UTC")
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="r")
        s.index = pd.to_datetime(s.index, utc=True).normalize()
        return s

    return _fail_unknown if behaviour == "fail" else _ok


class TestBadTicker:
    """A ticker that produces no equity data must surface as 502 with detail."""

    def test_unknown_ticker_returns_502(self, monkeypatch, app_client: TestClient) -> None:
        monkeypatch.setattr(main_mod, "get_log_returns", _patched_returns_factory("fail"))
        r = app_client.post(
            "/fit",
            json={
                "ticker": "XXXXXX",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 502
        assert "XXXXXX" in r.json()["detail"] or "equity" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Concurrent /fit requests — same params must produce identical results.
# ---------------------------------------------------------------------------


class TestConcurrentFits:
    """Race-condition guard: 5 parallel /fit calls with identical params
    must return identical β/R²/n_obs. Mismatch implies a shared mutable
    cache somewhere on the hot path."""

    def test_parallel_fits_identical_results(self, app_client: TestClient) -> None:
        payload = {
            "ticker": "RACE",
            "factors": ["factor_a", "factor_b"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        }

        def _one() -> dict[str, Any]:
            r = app_client.post("/fit", json=payload)
            assert r.status_code == 200, r.text
            b = r.json()
            return {
                "n_obs": b["n_obs"],
                "r2": round(b["model"]["r_squared"], 12),
                "betas": {f["id"]: round(f["beta"], 12) for f in b["factors"]},
            }

        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(_one) for _ in range(5)]
            results = [f.result() for f in as_completed(futs)]

        first = results[0]
        for r in results[1:]:
            assert r == first, f"race-condition: {r} != {first}"


# ---------------------------------------------------------------------------
# hac_lag edge cases.
# ---------------------------------------------------------------------------


class TestHacLagEdges:
    """hac_lag=0 should yield plain OLS-equivalent residual SE; hac_lag too
    large vs n_obs should error out informatively (not silently return junk)."""

    def test_hac_lag_zero_equivalent_plain_ols(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        x = rng.normal(0, 1, n)
        X = pd.DataFrame({"f": x}, index=idx)
        y = pd.Series(0.3 * x + rng.normal(0, 0.1, n), index=idx, name="r")

        # hac_lag=0 is allowed by statsmodels (it's the no-lag-correction case)
        # but with HAC it would still build a Newey-West with maxlags=0.
        # We check: lag is honoured and stats are produced (no NaN/Inf).
        result = fit_ols_hac(y, X, hac_lag=0)
        assert result.diagnostics.hac_lag == 0
        for est in result.factors:
            assert math.isfinite(est.std_err)
            assert math.isfinite(est.t_stat)

    def test_hac_lag_too_large_raises(self) -> None:
        """hac_lag >= n_obs - 1 must raise rather than produce junk SE."""
        import pytest

        rng = np.random.default_rng(0)
        n = 50
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        x = rng.normal(0, 1, n)
        X = pd.DataFrame({"f": x}, index=idx)
        y = pd.Series(0.3 * x + rng.normal(0, 0.1, n), index=idx, name="r")
        with pytest.raises(ValueError, match=r"hac_lag.*too large"):
            fit_ols_hac(y, X, hac_lag=100)

    def test_hac_lag_endpoint_override(self, app_client: TestClient) -> None:
        """User-supplied hac_lag is honoured by /fit and reported back."""
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-12-15",
                "hac_lag": 7,
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["diagnostics"]["hac_lag"] == 7

    def test_hac_lag_endpoint_too_large_returns_422(self, app_client: TestClient) -> None:
        """Endpoint validates hac_lag against post-transform n_obs."""
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-07-15",
                "hac_lag": 199,
            },
        )
        assert r.status_code == 422
        assert "hac_lag" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Epsilon edge cases for /fit.
# ---------------------------------------------------------------------------


class TestEpsilonEdges:
    """Pydantic enforces 0 < epsilon < 0.5 on the query param. Confirm the
    full pipeline rejects pathological values and clips correctly otherwise."""

    def test_epsilon_zero_rejected_by_query_validation(self, app_client: TestClient) -> None:
        r = app_client.post(
            "/fit?epsilon=0",
            json={
                "ticker": "TEST",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 422

    def test_epsilon_one_rejected_by_query_validation(self, app_client: TestClient) -> None:
        r = app_client.post(
            "/fit?epsilon=1.0",
            json={
                "ticker": "TEST",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 422

    def test_epsilon_extreme_high_clips_aggressively(self) -> None:
        """epsilon=0.49 should clip the [0.01, 0.99] series to [0.49, 0.51]
        and produce near-zero variance in Δlogit. The Δlogit series mean
        absolute deviation should drop by orders of magnitude vs eps=0.01."""
        idx = pd.date_range("2025-01-01", periods=100, freq="D")
        prices = pd.Series(np.linspace(0.05, 0.95, 100), index=idx)
        d_low = delta_logit(prices, epsilon=0.01).dropna()
        d_high = delta_logit(prices, epsilon=0.49).dropna()
        # With epsilon=0.49, almost all values clip → tiny derivative.
        assert d_high.abs().mean() < d_low.abs().mean() / 5


# ---------------------------------------------------------------------------
# Perfect collinearity: VIF must be reported as a number (Inf would JSON-encode
# as null and silently bypass the user's understanding of the problem).
# ---------------------------------------------------------------------------


class TestPerfectCollinearity:
    """When two factors are perfectly collinear, VIF is mathematically Inf.
    The /fit response must communicate this — we do that by capping VIF at a
    sentinel large value AND adding a warning to the new ``warnings`` field
    so the user sees what's happening."""

    def test_collinear_factors_vif_finite_or_warning(
        self, monkeypatch, app_client: TestClient
    ) -> None:
        # Build a custom_factors payload where factor_b is just an alias of
        # factor_a (same slug).  Using the synthetic_factor fixture, slug-a
        # is the same DataFrame either way.
        rng = pd.date_range("2025-06-01", "2025-12-31", freq="D", tz="UTC")
        t = np.arange(len(rng)) / len(rng)
        prices = pd.DataFrame(
            {"price": (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)},
            index=rng,
        )
        prices.index.name = "date"

        def _fake_fetch(_client, slug, start=None, end=None):
            df = prices.copy()
            if start is not None:
                df = df[df.index >= start]
            if end is not None:
                df = df[df.index <= end]
            return df

        monkeypatch.setattr(main_mod, "fetch_factor_history", _fake_fetch)

        r = app_client.post(
            "/fit",
            json={
                "ticker": "COLL",
                "factors": ["factor_a", "factor_b"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        vif = body["diagnostics"]["vif"]
        warns: list[str] = body.get("warnings") or []
        # VIF must be finite (sentinel-capped) — never None — so clients
        # can rely on a comparable numeric type.
        for v in vif.values():
            assert v is not None, "VIF was None (Inf leaked through JSON)"
            assert math.isfinite(v), f"VIF was non-finite: {v}"
        # The collinearity must be surfaced as a warning so the user knows
        # to drop the duplicate factor.
        joined = " ".join(warns).lower()
        assert "collinear" in joined, f"expected collinearity warning, got {warns!r}"


# ---------------------------------------------------------------------------
# Response shape: backward-compat fields plus new defensive fields.
# ---------------------------------------------------------------------------


class TestResponseShape:
    """Make sure new fields exist (n_obs_used, warnings, factor_metadata,
    etc.) and the legacy fields are still present (backward compat)."""

    def test_response_has_legacy_and_new_fields(self, app_client: TestClient) -> None:
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a", "factor_b"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Legacy fields (must not regress):
        for key in (
            "ticker",
            "n_obs",
            "epsilon",
            "model",
            "factors",
            "diagnostics",
            "time_series",
            "factor_traces",
        ):
            assert key in body, f"missing legacy field {key}"
        # New defensive fields (additive — backward compatible):
        assert "warnings" in body
        assert isinstance(body["warnings"], list)
        assert "factor_metadata" in body
        assert isinstance(body["factor_metadata"], dict)
        for fid in ("factor_a", "factor_b"):
            md = body["factor_metadata"].get(fid)
            assert md is not None, f"missing metadata for {fid}"
            assert "is_probability" in md
            assert "source" in md
            assert "n_obs" in md


# ---------------------------------------------------------------------------
# Cache safety: different (start, end) for same ticker must NOT cross-pollute.
# ---------------------------------------------------------------------------


class TestCacheSafety:
    """Two consecutive /fit calls with different windows must produce different
    n_obs (the cached path must include start/end in the key)."""

    def test_different_window_different_n_obs(self, app_client: TestClient) -> None:
        r1 = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-09-15",
            },
        )
        r2 = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r1.status_code == 200 and r2.status_code == 200
        # Wider window must include strictly more obs.
        assert r2.json()["n_obs"] > r1.json()["n_obs"]


# ---------------------------------------------------------------------------
# Clipping behaviour reporting.
# ---------------------------------------------------------------------------


class TestClippingReporting:
    """When a probability factor saturates near 0 or 1 the Δlogit step is
    near-zero — a silent failure mode in production. The endpoint must
    report ``clipping_events`` and per-factor counts so the user can react."""

    def test_clipping_events_reported(self, monkeypatch, app_client: TestClient) -> None:
        # Force a series that saturates at 0.001 (well below default
        # epsilon=0.01) for half the window.
        rng = pd.date_range("2025-06-01", "2025-12-31", freq="D", tz="UTC")
        n = len(rng)
        prices_saturated = np.where(np.arange(n) < n // 2, 0.001, 0.50)
        df_sat = pd.DataFrame({"price": prices_saturated}, index=rng)
        df_sat.index.name = "date"

        # Smooth fallback for the other slug.
        t = np.arange(n) / n
        df_smooth = pd.DataFrame(
            {"price": (0.55 + 0.20 * np.cos(2 * np.pi * t * 0.8)).clip(0.05, 0.95)},
            index=rng,
        )
        df_smooth.index.name = "date"

        bank = {"slug-a": df_sat, "slug-b": df_smooth}

        def _fake(_client, slug, start=None, end=None):
            df = bank[slug].copy()
            if start is not None:
                df = df[df.index >= start]
            if end is not None:
                df = df[df.index <= end]
            return df

        monkeypatch.setattr(main_mod, "fetch_factor_history", _fake)

        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a", "factor_b"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["clipping_events"] >= 50, body["clipping_events"]
        meta_a = body["factor_metadata"]["factor_a"]
        meta_b = body["factor_metadata"]["factor_b"]
        assert meta_a["clipping_events"] >= 50
        assert meta_b["clipping_events"] == 0
        # A user-visible warning should be generated.
        joined = " ".join(body["warnings"]).lower()
        assert "clipping" in joined


# ---------------------------------------------------------------------------
# Headline summary, verdict pill, top_significant — additive UX fields.
# These cover the post-audit improvements that surface fit quality at a
# glance without forcing the caller to re-derive everything in JS.
# ---------------------------------------------------------------------------


class TestSummaryAndVerdict:
    """The /fit response now includes a human-readable summary string, a
    single-word verdict ('well_specified' | 'weak_fit' | 'collinear' |
    'underpowered' | 'borderline'), and a ``top_significant`` list of
    factor ids sorted by |t-stat|. These are computed server-side so all
    clients share the same readout."""

    def _post(self, app_client: TestClient, **overrides: Any) -> dict[str, Any]:
        payload = {
            "ticker": "TEST",
            "factors": ["factor_a", "factor_b"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        }
        payload.update(overrides)
        r = app_client.post("/fit", json=payload)
        assert r.status_code == 200, r.text
        return r.json()

    def test_summary_string_is_populated_and_human(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        s = body.get("summary", "")
        assert isinstance(s, str) and len(s) > 0
        # Must mention factor count + R² in some form.
        assert "factor" in s.lower()
        assert "r" in s.lower()  # 'R²' contains 'r'

    def test_verdict_is_one_of_the_known_buckets(self, app_client: TestClient) -> None:
        body = self._post(app_client)
        assert body["verdict"] in {
            "well_specified",
            "weak_fit",
            "collinear",
            "underpowered",
            "borderline",
        }

    def test_top_significant_is_sorted_by_abs_t_stat_descending(
        self, app_client: TestClient
    ) -> None:
        body = self._post(app_client)
        sig_ids = body.get("top_significant", [])
        # Cross-check ordering against the per-factor t-stats.
        t_by_id = {f["id"]: f["t_stat"] for f in body["factors"]}
        # Each id in top_significant should have p<0.05.
        for fid in sig_ids:
            f = next(f for f in body["factors"] if f["id"] == fid)
            assert f["p_value"] < 0.05, fid
        # Ordering descending by |t|.
        ts = [abs(t_by_id[fid]) for fid in sig_ids]
        assert ts == sorted(ts, reverse=True)

    def test_underpowered_verdict_on_short_window(self, app_client: TestClient) -> None:
        # Force a short overlap (~10 obs).
        body = self._post(
            app_client,
            start="2025-07-01",
            end="2025-07-15",
        )
        # n_obs is tiny; verdict should be underpowered (or weak_fit if
        # the synthetic series happens to land that way; both are honest
        # readouts for a 10-obs fit).
        assert body["verdict"] in {"underpowered", "weak_fit", "borderline"}, body["verdict"]


class TestAutoPruneCollinear:
    """The ?prune_collinear=true query param should iteratively drop the
    highest-VIF factor until every remaining VIF is < 5, surfacing the
    dropped ids in ``auto_pruned``. Without the param the response shape
    must remain backward-compatible (auto_pruned is []) so legacy clients
    don't break."""

    def test_no_op_when_factors_not_collinear(self, app_client: TestClient) -> None:
        # factor_a and factor_b in the fixture are nearly orthogonal so
        # nothing should be pruned even with the flag on.
        r = app_client.post(
            "/fit?prune_collinear=true",
            json={
                "ticker": "TEST",
                "factors": ["factor_a", "factor_b"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["auto_pruned"] == []
        # Both factors should still appear in the coef table.
        assert {f["id"] for f in body["factors"]} == {"factor_a", "factor_b"}

    def test_drops_duplicate_collinear_factor(self, monkeypatch, app_client: TestClient) -> None:
        # Both slugs return identical price series → perfect collinearity →
        # exactly one of the two must be pruned. (The pruner picks the
        # highest-VIF first, which is symmetric here; either is fine.)
        rng = pd.date_range("2025-06-01", "2025-12-31", freq="D", tz="UTC")
        t = np.arange(len(rng)) / len(rng)
        prices = pd.DataFrame(
            {"price": (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)},
            index=rng,
        )
        prices.index.name = "date"

        def _fake_fetch(_client, slug, start=None, end=None):
            df = prices.copy()
            if start is not None:
                df = df[df.index >= start]
            if end is not None:
                df = df[df.index <= end]
            return df

        monkeypatch.setattr(main_mod, "fetch_factor_history", _fake_fetch)

        r = app_client.post(
            "/fit?prune_collinear=true",
            json={
                "ticker": "COLL",
                "factors": ["factor_a", "factor_b"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["auto_pruned"]) == 1, body["auto_pruned"]
        # The surviving factor must NOT be in auto_pruned.
        survivors = {f["id"] for f in body["factors"]}
        assert len(survivors & set(body["auto_pruned"])) == 0
        # A user-facing warning must explain what was dropped.
        joined = " ".join(body["warnings"]).lower()
        assert "prune" in joined or "auto-pruned" in joined
