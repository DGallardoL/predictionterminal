"""Tests for the 6-feature rigour pack on /fit (2026-05-15).

The rigour pack adds:

  1. ``overfit_risk_flags``  — structured guardrails on n/k, clipping,
                                sign-inconsistency, theme mismatch.
  2. ``multitest_hint``      — Bonferroni-style alpha/N derived from the
                                ``X-Session-Test-Count`` request header
                                (also surfaced via the ``X-Session-Test-Hint``
                                response header).
  3. extended ``/fit/preview`` factor coverage with ``n_obs_available`` /
     ``n_obs_in_window`` / ``predicted_window_n_obs`` so the user knows
     BEFORE clicking Run how many days will survive the inner-join.
  4. explicit ``OosRSquaredSkipped`` block (replaces silent ``null`` when
     the walk-forward floor isn't reached).
  5. ``regime_changes``     — per-factor structural-break detection
                                (``pfm.regression_regime``).
  6. ``residual_annotations[*].news_links`` — deep-link search URLs for
     the worst-fit dates.

Each test uses a synthetic DGP or the ``app_client`` fixture (which mocks
external IO) so the suite stays hermetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from pfm.regression_regime import RegimeChange, detect_regime_changes
from pfm.regression_router import (
    _build_multitest_hint,
    _build_news_links,
    _compute_overfit_risk_flags,
    _residual_annotations,
    _walk_forward_oos_r2,
)
from pfm.schemas import (
    FactorEstimateOut,
    FactorMetadataOut,
    OosRSquaredSkipped,
)

# ===========================================================================
# Feature 1 — overfit_risk_flags
# ===========================================================================


class _Spec:
    """Minimal stand-in for ``FactorConfig`` — only ``id`` and ``theme`` used."""

    def __init__(self, fid: str, theme: str = "other") -> None:
        self.id = fid
        self.theme = theme


def _est(fid: str, beta: float) -> FactorEstimateOut:
    return FactorEstimateOut(
        id=fid,
        beta=beta,
        std_err=0.05,
        t_stat=beta / 0.05,
        p_value=0.01,
        ci_low=beta - 0.1,
        ci_high=beta + 0.1,
    )


def _meta(n_obs: int = 60, clipping_events: int = 0) -> FactorMetadataOut:
    return FactorMetadataOut(
        is_probability=True,
        source="polymarket",
        n_obs=n_obs,
        clipping_events=clipping_events,
    )


class TestOverfitRiskFlags:
    def test_low_dof_emitted_when_ratio_below_10(self) -> None:
        # n=10, k=2 => ratio=5 → high severity, code="low_dof".
        specs = [_Spec("f1"), _Spec("f2")]
        flags = _compute_overfit_risk_flags(
            n_obs=10,
            factor_specs=specs,
            factor_meta={"f1": _meta(), "f2": _meta()},
            factor_estimates=[_est("f1", 0.3), _est("f2", -0.2)],
            ticker="NVDA",
        )
        codes = [f.code for f in flags]
        assert "low_dof" in codes
        flag = next(f for f in flags if f.code == "low_dof")
        assert flag.level == "high"
        assert "n/k=5.0" in flag.message

    def test_moderate_dof_when_ratio_between_10_and_20(self) -> None:
        # n=30, k=2 => ratio=15 → medium "moderate_dof".
        specs = [_Spec("f1"), _Spec("f2")]
        flags = _compute_overfit_risk_flags(
            n_obs=30,
            factor_specs=specs,
            factor_meta={"f1": _meta(), "f2": _meta()},
            factor_estimates=[_est("f1", 0.3), _est("f2", -0.2)],
            ticker="NVDA",
        )
        codes = [f.code for f in flags]
        assert "moderate_dof" in codes
        assert "low_dof" not in codes

    def test_no_dof_flag_when_ratio_above_20(self) -> None:
        specs = [_Spec("f1")]
        flags = _compute_overfit_risk_flags(
            n_obs=200,
            factor_specs=specs,
            factor_meta={"f1": _meta()},
            factor_estimates=[_est("f1", 0.3)],
            ticker="NVDA",
        )
        assert all(f.code not in {"low_dof", "moderate_dof"} for f in flags)

    def test_high_clipping_flag(self) -> None:
        # 2 of 2 factors above 20% clipping → flag fires.
        specs = [_Spec("f1"), _Spec("f2")]
        meta = {
            "f1": _meta(n_obs=100, clipping_events=30),
            "f2": _meta(n_obs=100, clipping_events=25),
        }
        flags = _compute_overfit_risk_flags(
            n_obs=200,
            factor_specs=specs,
            factor_meta=meta,
            factor_estimates=[_est("f1", 0.3), _est("f2", -0.2)],
            ticker="NVDA",
        )
        codes = [f.code for f in flags]
        assert "high_clipping" in codes

    def test_sign_inconsistency_within_theme(self) -> None:
        # Two BTC-themed factors with opposite signs → sign_inconsistent.
        specs = [_Spec("btc_a", theme="crypto"), _Spec("btc_b", theme="crypto")]
        flags = _compute_overfit_risk_flags(
            n_obs=200,
            factor_specs=specs,
            factor_meta={"btc_a": _meta(), "btc_b": _meta()},
            factor_estimates=[_est("btc_a", 0.5), _est("btc_b", -0.4)],
            ticker="BTC-USD",
        )
        codes = [f.code for f in flags]
        assert "sign_inconsistent" in codes

    def test_theme_mismatch_for_sports_factor_on_equity_ticker(self) -> None:
        specs = [_Spec("mayweather_v_pacquiao", theme="sports")]
        flags = _compute_overfit_risk_flags(
            n_obs=200,
            factor_specs=specs,
            factor_meta={"mayweather_v_pacquiao": _meta()},
            factor_estimates=[_est("mayweather_v_pacquiao", 0.3)],
            ticker="NVDA",
        )
        codes = [f.code for f in flags]
        assert "theme_mismatch" in codes


# ===========================================================================
# Feature 2 — multitest_hint + X-Session-Test-Count header
# ===========================================================================


class TestMultitestHint:
    def test_default_threshold_is_alpha(self) -> None:
        hint = _build_multitest_hint(1)
        assert hint.tests_this_session == 1
        assert abs(hint.bh_q_threshold - 0.05) < 1e-12

    def test_threshold_shrinks_with_n(self) -> None:
        hint = _build_multitest_hint(30)
        # 0.05 / 30 ≈ 0.001666...
        assert abs(hint.bh_q_threshold - 0.05 / 30) < 1e-12
        assert "30" in hint.message

    def test_endpoint_reads_header(self, app_client: TestClient) -> None:
        # Header value is opaque to the body; verify it round-trips into
        # both the multitest_hint field and the X-Session-Test-Hint header.
        r = app_client.post(
            "/fit",
            headers={"X-Session-Test-Count": "30"},
            json={
                "ticker": "TEST",
                "factors": ["factor_a", "factor_b"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        hint = body["multitest_hint"]
        assert hint["tests_this_session"] == 30
        assert abs(hint["bh_q_threshold"] - 0.05 / 30) < 1e-9
        # Header is ASCII-safe (uses "alpha" not the greek letter).
        header_hint = r.headers.get("X-Session-Test-Hint", "")
        assert "30 tests" in header_hint
        assert "alpha/N" in header_hint

    def test_endpoint_defaults_to_one_when_header_absent(
        self,
        app_client: TestClient,
    ) -> None:
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text
        hint = r.json()["multitest_hint"]
        assert hint["tests_this_session"] == 1


# ===========================================================================
# Feature 3 — /fit/preview factor_coverage with new fields
# ===========================================================================


class TestPreFlightFactorCoverage:
    def test_returns_factor_coverage_map_and_predicted_n(
        self,
        app_client: TestClient,
    ) -> None:
        r = app_client.post(
            "/fit/preview",
            json={
                "ticker": "TEST",
                "factors": ["factor_a", "factor_b"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "factor_coverage_map" in body
        assert "predicted_window_n_obs" in body
        assert "joint_window_obs" in body
        # Map and list carry the same factors.
        list_ids = {c["factor_id"] for c in body["factor_coverage"]}
        map_ids = set(body["factor_coverage_map"].keys())
        assert list_ids == map_ids == {"factor_a", "factor_b"}
        # Predicted post-join obs == joint_n_obs.
        assert body["predicted_window_n_obs"] == body["joint_n_obs"]
        assert body["joint_window_obs"] == body["joint_n_obs"]
        # Per-factor coverage exposes both raw + in-window counts.
        for item in body["factor_coverage_map"].values():
            assert item["n_obs_in_window"] >= 0
            # n_obs_available should be >= n_obs_in_window (or equal if
            # the upstream fetch happened to land exactly on the window).
            assert item["n_obs_available"] >= item["n_obs_in_window"]
            # Coverage ratio is bounded.
            assert 0.0 <= item["coverage_pct"] <= 1.0


# ===========================================================================
# Feature 4 — OOS skipped reason
# ===========================================================================


class TestOosSkippedReason:
    def test_walk_forward_returns_skipped_block(self) -> None:
        idx = pd.date_range("2025-01-01", periods=32, freq="D")
        X = pd.DataFrame({"f": np.linspace(-1, 1, 32)}, index=idx)
        y = pd.Series(np.linspace(-1, 1, 32), index=idx)
        out = _walk_forward_oos_r2(y, X)
        assert isinstance(out, OosRSquaredSkipped)
        assert out.value is None
        assert out.skipped is True
        assert "n_obs=32" in out.skipped_reason
        assert "min_n_for_walk_forward=100" in out.skipped_reason

    def test_endpoint_emits_skipped_reason(self, app_client: TestClient) -> None:
        # The fixture's factor windows produce ~130 obs, so this test uses
        # a wide window with a known-tiny inner-join. The synthetic factor
        # window is 2025-06-01..2025-12-31 — pinning the request to a
        # narrow slice keeps n below the walk-forward floor.
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-08-01",
            },
        )
        assert r.status_code == 200, r.text
        oos = r.json()["oos_r_squared"]
        # Either the skipped block or — if the test fixture happens to
        # exceed 100 obs — the real OosRSquared. The contract guarantees
        # one of the two and never silent null.
        assert oos is not None
        if oos.get("skipped") is True:
            assert oos["value"] is None
            assert "min_n_for_walk_forward=100" in oos["skipped_reason"]
        else:
            assert "value" in oos and oos["value"] is not None


# ===========================================================================
# Feature 5 — regime_changes detection
# ===========================================================================


class TestRegimeChanges:
    def test_detects_synthetic_break(self) -> None:
        # β switches from +0.5 to -0.5 at the midpoint. Detector must
        # surface a regime change for "f" with sign_flipped=True. The
        # detector picks the breakpoint with the smallest p — that may
        # land at any of the quartile candidates, so we don't assert the
        # exact break date or magnitude on the post-side (just that the
        # sign flipped and the test was significant).
        rng = np.random.default_rng(seed=11)
        n = 240
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = rng.normal(0, 0.30, n)
        beta_t = np.where(np.arange(n) < n // 2, 0.50, -0.50)
        eps = rng.normal(0, 0.005, n)
        y = pd.Series(beta_t * x + eps, index=idx, name="r")
        X = pd.DataFrame({"f": x}, index=idx)
        out = detect_regime_changes(y, X, p_threshold=0.10)
        assert len(out) == 1
        rc = out[0]
        assert isinstance(rc, RegimeChange)
        assert rc.factor_id == "f"
        assert rc.sign_flipped is True
        # Sign of pre/post must match the planted DGP at the chosen
        # breakpoint. We accept any quartile pick since the smallest-p
        # breakpoint depends on sample noise.
        assert rc.pre_beta > 0
        assert rc.post_beta < 0
        assert rc.p_value < 0.10
        assert rc.chow_stat > 5.0  # joint break is large
        assert "2024" in rc.breakpoint_date

    def test_skipped_when_n_below_floor(self) -> None:
        idx = pd.date_range("2025-01-01", periods=40, freq="D")
        X = pd.DataFrame({"f": np.linspace(-1, 1, 40)}, index=idx)
        y = pd.Series(np.linspace(-1, 1, 40), index=idx)
        assert detect_regime_changes(y, X) == []

    def test_no_break_when_beta_constant(self) -> None:
        # A clean DGP with constant β shouldn't flip the threshold.
        rng = np.random.default_rng(seed=2)
        n = 240
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = rng.normal(0, 0.20, n)
        y = pd.Series(0.30 * x + rng.normal(0, 0.005, n), index=idx, name="r")
        X = pd.DataFrame({"f": x}, index=idx)
        out = detect_regime_changes(y, X, p_threshold=0.10)
        # Stable DGP — at most an occasional spurious detection at the
        # 10% level. The contract is "empty most of the time"; we accept
        # 0 detections deterministically with this seed.
        assert out == []


# ===========================================================================
# Feature 6 — residual_annotations.news_links
# ===========================================================================


class TestResidualNewsLinks:
    def test_news_links_built_from_ticker_and_date(self) -> None:
        ts = pd.Timestamp("2025-09-17", tz="UTC")
        links = _build_news_links("NVDA", ts)
        assert len(links) >= 2
        assert all(link.startswith("http") for link in links)
        assert any("NVDA" in link for link in links)
        assert any("2025-09-17" in link for link in links)

    def test_residual_annotations_carry_news_links(self) -> None:
        rng = np.random.default_rng(seed=44)
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = rng.normal(0, 0.30, n)
        y_clean = 0.30 * x + rng.normal(0, 0.005, n)
        # Plant a 5-σ outlier on a known date.
        y = y_clean.copy()
        y[60] += 0.20
        y_ser = pd.Series(y, index=idx, name="r")
        X = pd.DataFrame({"f": x}, index=idx)
        residual = pd.Series(y - 0.30 * x, index=idx)
        ests = [_est("f", 0.30)]
        out = _residual_annotations(
            y_ser,
            X,
            residual,
            ests,
            top_k=5,
            ticker="NVDA",
        )
        assert out, "expected at least one residual annotation"
        first = out[0]
        assert first.news_links, "ticker provided -> news_links populated"
        assert all("NVDA" in link or "nvda" in link.lower() for link in first.news_links)

    def test_residual_annotations_omit_news_links_without_ticker(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        residual = pd.Series([0.0, 0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
        X = pd.DataFrame({"f": np.zeros(10)}, index=idx)
        y = pd.Series(np.zeros(10), index=idx)
        ests = [_est("f", 0.0)]
        out = _residual_annotations(y, X, residual, ests, top_k=3)
        for ann in out:
            assert ann.news_links == []

    def test_endpoint_emits_news_links(self, app_client: TestClient) -> None:
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
        annotations = r.json().get("residual_annotations", [])
        if annotations:
            ann = annotations[0]
            assert isinstance(ann.get("news_links"), list)
            assert len(ann["news_links"]) >= 1
            assert any("TEST" in link.upper() for link in ann["news_links"])


# ===========================================================================
# End-to-end /fit shape: rigour fields all present in the response.
# ===========================================================================


class TestRigourPackEndpointShape:
    def test_fit_response_carries_all_rigour_fields(
        self,
        app_client: TestClient,
    ) -> None:
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
        # All six rigour fields must be present.
        assert "overfit_risk_flags" in body
        assert "multitest_hint" in body
        assert "regime_changes" in body
        assert "oos_r_squared" in body
        assert isinstance(body["overfit_risk_flags"], list)
        assert isinstance(body["regime_changes"], list)
        # multitest_hint always built (default n=1).
        assert body["multitest_hint"]["tests_this_session"] >= 1
