"""Tests for ``pfm.ml_regime_router`` — /ml/regime (Regime Monitor).

The router is mounted on a fresh :class:`FastAPI` app (no full ``pfm.main``
lifespan). The history loader is monkeypatched on the *ml_regime_router module
namespace* (it imports the loader by value from
``pfm.terminal.factor_clusters``, so patching the source module would not take
effect) with a synthetic DGP that has a deliberate regime change partway
through: the first half is driven by low-volatility innovations and the second
half by high-volatility innovations.

We assert the classifier recovers that structure — at least two regimes, the
calm regime dominating the first half and the stressed regime the second — and
exercise the degraded-mode and ``n_regimes`` param paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import ml_regime_router as mr


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(mr.router)
    return TestClient(app)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_regime_history(seed: int = 11) -> dict[str, pd.Series]:
    """Six factors with a sharp vol regime change at the midpoint.

    First half: small daily innovations (calm). Second half: large innovations
    (stressed). Both halves share a latent factor so co-movement is non-trivial.
    """
    rng = np.random.default_rng(seed)
    half = 130
    n = 2 * half
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    # Per-day innovation scale: low for the first half, high for the second.
    scale = np.concatenate([np.full(half, 0.05), np.full(half, 0.45)])

    series: dict[str, pd.Series] = {}
    latent = rng.standard_normal(n) * scale
    for i in range(6):
        innov = latent + rng.standard_normal(n) * scale * 0.5
        prob = _logistic(np.cumsum(innov))
        series[f"factor-{i}"] = pd.Series(prob, index=idx)
    return series


def _make_calm_then_calm_history(seed: int = 3) -> dict[str, pd.Series]:
    """Uniformly calm history — used to check the current regime is the last
    segment's level. Here the whole tail is low-vol, so the current regime must
    be the calmest (regime 0)."""
    rng = np.random.default_rng(seed)
    half = 130
    n = 2 * half
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    # High first, low (calm) second → current should be calm.
    scale = np.concatenate([np.full(half, 0.45), np.full(half, 0.05)])
    series: dict[str, pd.Series] = {}
    latent = rng.standard_normal(n) * scale
    for i in range(6):
        innov = latent + rng.standard_normal(n) * scale * 0.5
        series[f"factor-{i}"] = pd.Series(_logistic(np.cumsum(innov)), index=idx)
    return series


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch on the consuming module's namespace, not the source module.
    monkeypatch.setattr(mr, "_load_cached_history", _make_regime_history)
    # Clear the shared TTL cache between tests so query variants (and the
    # degraded-mode test's empty history) don't collide on a cached payload.
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()


# --- endpoint tests ---------------------------------------------------------


def test_regime_basic_shape(client: TestClient) -> None:
    body = client.get("/ml/regime").json()
    assert body["degraded_mode"] is False
    assert body["n_regimes"] == 3
    assert body["window"] == 10
    assert body["n_obs"] > 0
    # path is one point per state-feature row, capped at MAX_PATH_DAYS.
    assert len(body["path"]) == min(body["n_obs"], mr.MAX_PATH_DAYS)
    for pt in body["path"]:
        assert pt["regime"] >= 0
        # date parses as ISO.
        pd.Timestamp(pt["date"])
    # Every regime in the path has a summary.
    summarized = {s["regime"] for s in body["regimes"]}
    assert {pt["regime"] for pt in body["path"]}.issubset(summarized)


def test_at_least_two_regimes_detected(client: TestClient) -> None:
    body = client.get("/ml/regime").json()
    distinct = {pt["regime"] for pt in body["path"]}
    assert len(distinct) >= 2


def test_calm_dominates_first_half_stressed_second(client: TestClient) -> None:
    """With n_regimes=2 matching the 2-state DGP, the calm regime (0) owns the
    low-vol first half and the stressed regime (1) the high-vol second half."""
    body = client.get("/ml/regime?n_regimes=2").json()
    path = body["path"]
    n = len(path)
    mid = n // 2
    first = [p["regime"] for p in path[:mid]]
    second = [p["regime"] for p in path[mid:]]
    stressed = body["n_regimes"] - 1  # == 1

    # First half is overwhelmingly the calmest regime.
    assert first.count(0) > 0.7 * len(first)
    # Second half is overwhelmingly the most-stressed regime.
    assert second.count(stressed) > 0.7 * len(second)


def test_centroid_volatility_increases_with_regime(client: TestClient) -> None:
    """Volatility-sorted labels mean centroid vol is non-decreasing in id."""
    body = client.get("/ml/regime").json()
    by_id = {s["regime"]: s for s in body["regimes"]}
    vols = [by_id[i]["centroid"]["volatility"] for i in sorted(by_id)]
    assert vols == sorted(vols)
    # Labels span calm → stressed.
    assert by_id[min(by_id)]["label"] == "calm"
    assert by_id[max(by_id)]["label"] == "stressed"


def test_current_regime_matches_last_segment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the tail of history is the calm segment, current_regime is the
    calmest (0)."""
    monkeypatch.setattr(mr, "_load_cached_history", _make_calm_then_calm_history)
    body = client.get("/ml/regime").json()
    assert body["current_regime"] == 0
    assert body["current_label"] == "calm"
    assert body["path"][-1]["regime"] == 0


def test_persistence_stats_present(client: TestClient) -> None:
    body = client.get("/ml/regime").json()
    for s in body["regimes"]:
        assert s["n_days"] >= 0
        assert s["avg_duration_days"] >= 0.0
        assert set(s["centroid"]) == set(mr.FEATURE_NAMES)
    # Total assigned days equals the number of feature rows.
    assert sum(s["n_days"] for s in body["regimes"]) == body["n_obs"]


def test_n_regimes_param_plumbing(client: TestClient) -> None:
    body = client.get("/ml/regime?n_regimes=4").json()
    assert body["n_regimes"] == 4
    # At most 4 distinct labels can appear.
    assert len({pt["regime"] for pt in body["path"]}) <= 4
    # current_label is one of the labelled bands.
    assert body["current_label"] in {"calm", "quiet-normal", "normal", "elevated", "stressed"}


def test_window_param_plumbing(client: TestClient) -> None:
    body = client.get("/ml/regime?window=20").json()
    assert body["window"] == 20
    assert body["degraded_mode"] is False


def test_degraded_mode_when_history_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mr, "_load_cached_history", lambda: {})
    body = client.get("/ml/regime").json()
    assert body["degraded_mode"] is True
    assert body["path"] == []
    assert body["regimes"] == []
    assert body["current_regime"] == -1
    assert body["reason"]


def test_invalid_n_regimes_rejected(client: TestClient) -> None:
    # n_regimes outside [2, 5] is a 422 from query validation.
    assert client.get("/ml/regime?n_regimes=1").status_code == 422
    assert client.get("/ml/regime?n_regimes=9").status_code == 422


# --- new capabilities -------------------------------------------------------


def _make_planted_factor_history(seed: int = 7) -> dict[str, pd.Series]:
    """Six co-moving factors with a vol regime change, plus one planted factor.

    The planted factor (``planted-up``) has *positive-drift* Δlogit in the
    high-vol second half and roughly flat Δlogit in the calm first half, so its
    up-day hit-rate must be higher in the stressed regime than the calm one.
    """
    rng = np.random.default_rng(seed)
    half = 130
    n = 2 * half
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    scale = np.concatenate([np.full(half, 0.05), np.full(half, 0.45)])

    series: dict[str, pd.Series] = {}
    latent = rng.standard_normal(n) * scale
    for i in range(6):
        innov = latent + rng.standard_normal(n) * scale * 0.5
        series[f"factor-{i}"] = pd.Series(_logistic(np.cumsum(innov)), index=idx)

    # Planted: tiny zero-mean noise in calm half, modest positive drift in the
    # stressed half. We offset the random-walk level low (-3.0 in logit space)
    # so the probability climbs from ~0.04 toward ~0.33 without ever hitting the
    # clip band — keeping Δlogit > 0 on (almost) every stressed day.
    drift = np.concatenate([np.zeros(half), np.full(half, 0.02)])
    noise = np.concatenate([rng.standard_normal(half) * 0.01, rng.standard_normal(half) * 0.005])
    planted_walk = np.cumsum(drift + noise) - 3.0
    series["planted-up"] = pd.Series(_logistic(planted_walk), index=idx)
    return series


def test_transition_matrix_shape_and_rows_sum_to_one(client: TestClient) -> None:
    body = client.get("/ml/regime?n_regimes=3").json()
    tm = body["transition_matrix"]
    assert len(tm) == 3
    assert all(len(row) == 3 for row in tm)  # shape == n_regimes^2
    # Every row that has observed transitions sums to ~1 (all-zero rows allowed
    # for never-sourced states).
    for row in tm:
        total = sum(row)
        assert abs(total - 1.0) < 1e-6 or total == 0.0


def test_current_expected_remaining_days_positive(client: TestClient) -> None:
    body = client.get("/ml/regime").json()
    assert body["current_expected_remaining_days"] > 0.0


def test_factor_series_stats_returned(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mr, "_load_cached_history", _make_planted_factor_history)
    body = client.get("/ml/regime?n_regimes=2&factor=planted-up").json()
    assert body["factor"] == "planted-up"
    stats = body["factor_series_stats"]
    assert stats is not None
    for regime_id, s in stats.items():
        assert s["regime"] == int(regime_id)
        assert s["n_days"] >= 1
        assert 0.0 <= s["hit_rate"] <= 1.0


def test_planted_factor_higher_hit_rate_in_stressed_regime(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planted factor trends up in the high-vol (stressed) regime, so its
    up-day hit-rate must be higher there than in the calm regime."""
    monkeypatch.setattr(mr, "_load_cached_history", _make_planted_factor_history)
    body = client.get("/ml/regime?n_regimes=2&factor=planted-up").json()
    stats = {int(k): v for k, v in body["factor_series_stats"].items()}
    calm = stats[0]["hit_rate"]
    stressed = stats[max(stats)]["hit_rate"]
    assert stressed > calm
    # Stressed half is a strong up-drift, so its hit-rate should be high.
    assert stressed > 0.7


def test_bic_hint_present(client: TestClient) -> None:
    body = client.get("/ml/regime").json()
    bic = body["bic_by_n_regimes"]
    # 4 candidates {2,3,4,5}; the synthetic history is long enough for all.
    assert len(bic) == 4
    assert set(bic) == {"2", "3", "4", "5"}
    rec = body["recommended_n_regimes"]
    assert rec in {int(k) for k in bic}


def test_unknown_factor_404(client: TestClient) -> None:
    resp = client.get("/ml/regime?factor=does-not-exist")
    assert resp.status_code == 404


def test_scope_note_present(client: TestClient) -> None:
    body = client.get("/ml/regime").json()
    assert "prediction-market factor cross-section" in body["scope"]
