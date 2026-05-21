"""Tests for ``pfm.terminal_factor_clusters`` — /terminal/factor-clusters.

The router is mounted on a fresh :class:`FastAPI` app to avoid the full
``pfm.main`` lifespan, and the on-disk cache + factors.yml loaders are
monkeypatched with synthetic series whose correlation structure we
control. That lets us assert on cluster shapes, leader detection, and
filter behaviour without touching real Polymarket data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_factor_clusters as tfc
from pfm.terminal_factor_clusters import (
    _cluster_from_corr,
    _delta_logit,
    _detect_leader,
    _FactorMeta,
    _pairwise_corr,
    router,
)

# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_history(seed: int = 7) -> dict[str, pd.Series]:
    """Three correlated 'trump' factors + three correlated 'fed' factors
    + one uncorrelated noise factor.

    Cluster construction: within each block all three slugs share the
    same contemporaneous latent innovation (so they correlate strongly
    on Δlogit). On top of that, the leader 'a' contributes a private
    pulse that feeds into followers 'b' and 'c' at lag 2 — that's the
    Granger-lite signal we want the leader detector to recover.
    """
    rng = np.random.default_rng(seed)
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    def _shifted(arr: np.ndarray, k: int) -> np.ndarray:
        return np.concatenate([np.zeros(k), arr[:-k]]) if k > 0 else arr

    def _block(seed_offset: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        latent = rng.standard_normal(n) * 0.5  # shared contemp signal
        leader_pulse = rng.standard_normal(n) * 0.4  # leader's private innovation
        noise_a = rng.standard_normal(n) * 0.05
        noise_b = rng.standard_normal(n) * 0.05
        noise_c = rng.standard_normal(n) * 0.05
        innov_a = latent + leader_pulse + noise_a
        innov_b = latent + _shifted(leader_pulse, 2) + noise_b
        innov_c = latent + _shifted(leader_pulse, 2) + noise_c

        def _series(innov: np.ndarray) -> pd.Series:
            logit = np.cumsum(innov) * 0.10  # damp so probs stay in (0.1, 0.9)
            return pd.Series(_logistic(logit), index=idx)

        return _series(innov_a), _series(innov_b), _series(innov_c)

    t_a, t_b, t_c = _block(0)
    f_a, f_b, f_c = _block(1)

    lone_innov = rng.standard_normal(n) * 0.5
    lone = pd.Series(_logistic(np.cumsum(lone_innov) * 0.10), index=idx)

    return {
        "trump-a": t_a,
        "trump-b": t_b,
        "trump-c": t_c,
        "fed-a": f_a,
        "fed-b": f_b,
        "fed-c": f_c,
        "lone": lone,
    }


def _make_meta() -> dict[str, _FactorMeta]:
    return {
        "trump-a": _FactorMeta("trump_a", "trump-a", "politics", "Trump out by June"),
        "trump-b": _FactorMeta("trump_b", "trump-b", "politics", "Trump impeached 2026"),
        "trump-c": _FactorMeta("trump_c", "trump-c", "politics", "Trump removed by Senate"),
        "fed-a": _FactorMeta("fed_a", "fed-a", "macro", "Fed cuts 25bps July"),
        "fed-b": _FactorMeta("fed_b", "fed-b", "macro", "Fed cuts 25bps Sept"),
        "fed-c": _FactorMeta("fed_c", "fed-c", "macro", "Fed cuts 50bps July"),
        "lone": _FactorMeta("lone", "lone", "other", "Unrelated noise factor"),
    }


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tfc, "_load_cached_history", _make_history)
    monkeypatch.setattr(tfc, "_load_factor_meta", _make_meta)


# --- pure-math unit tests ---------------------------------------------------


def test_delta_logit_clips_and_diffs() -> None:
    """ε-clipping pins prob=0 to logit(ε), and diff produces n-1 obs."""
    s = pd.Series(
        [0.0, 0.5, 0.9, 1.0],
        index=pd.date_range("2025-01-01", periods=4),
    )
    out = _delta_logit(s, eps=0.01)
    assert len(out) == 3
    # first transition: logit(0.01) -> logit(0.5) -> Δ ≈ +log(99) ≈ 4.595
    assert pytest.approx(out.iloc[0], rel=1e-3) == np.log(0.5 / 0.5) - np.log(0.01 / 0.99)
    assert np.isfinite(out).all()


def test_pairwise_corr_recovers_two_clusters() -> None:
    """Synthetic blocks should exhibit high intra-block, low inter-block |corr|."""
    history = _make_history()
    returns = pd.DataFrame({k: _delta_logit(v) for k, v in history.items()})
    corr = _pairwise_corr(returns)
    # intra-trump avg |corr| should comfortably exceed inter-cluster
    intra = np.mean(
        [
            abs(corr.loc[a, b])
            for a in ["trump-a", "trump-b", "trump-c"]
            for b in ["trump-a", "trump-b", "trump-c"]
            if a != b
        ]
    )
    inter = np.mean(
        [
            abs(corr.loc[a, b])
            for a in ["trump-a", "trump-b", "trump-c"]
            for b in ["fed-a", "fed-b", "fed-c"]
        ]
    )
    assert intra > 0.30
    assert intra > inter + 0.10


def test_cluster_and_leader_detection() -> None:
    """Cut at min_corr=0.3 -> trump and fed cluster separately; lead lag>=1."""
    history = _make_history()
    returns = pd.DataFrame({k: _delta_logit(v) for k, v in history.items()})
    corr = _pairwise_corr(returns)
    clusters = _cluster_from_corr(corr, min_corr=0.3)
    # locate the cluster containing trump-a; it should also hold trump-b and trump-c
    trump_cluster = next(members for members in clusters.values() if "trump-a" in members)
    assert {"trump-a", "trump-b", "trump-c"}.issubset(set(trump_cluster))
    # leader should be detected with a positive lag, strength > 0
    lead = _detect_leader(returns, trump_cluster)
    assert lead is not None
    _leader, lag, strength = lead
    assert lag >= 1
    assert 0.0 < strength <= 1.0


def test_endpoint_returns_clusters_and_filters_by_theme(client: TestClient) -> None:
    """Full HTTP path: theme=politics restricts the universe to 3 trump factors."""
    # full universe
    resp = client.get("/terminal/factor-clusters", params={"min_corr": 0.3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_factors_in"] == 7
    assert body["n_clusters"] >= 2
    # one cluster should hold all three trump factors
    cluster_sets = [set(c["members"]) for c in body["clusters"]]
    assert any({"trump_a", "trump_b", "trump_c"}.issubset(s) for s in cluster_sets)

    # theme=politics filter
    resp_p = client.get("/terminal/factor-clusters", params={"theme": "politics", "min_corr": 0.3})
    assert resp_p.status_code == 200
    body_p = resp_p.json()
    assert body_p["n_factors_in"] == 3
    assert body_p["theme"] == "politics"
    # all returned member ids should belong to the politics block
    politics_ids = {"trump_a", "trump_b", "trump_c"}
    seen: set[str] = set()
    for c in body_p["clusters"]:
        seen.update(c["members"])
        # leader, when present, must be one of the cluster members
        if c["leader"] is not None:
            assert c["leader"]["factor_id"] in c["members"]
            assert 1 <= c["leader"]["n_lags_lead"] <= 5
    assert seen.issubset(politics_ids)
