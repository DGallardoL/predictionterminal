"""Tests for ``pfm.ml_hub_router`` — /ml/factor-map (Factor Galaxy).

The router is mounted on a fresh :class:`FastAPI` app (no full ``pfm.main``
lifespan). History + factors.yml loaders are monkeypatched on the *ml_hub_router
module namespace* (it imports those names by value from
``pfm.terminal.factor_clusters``, so patching the source module would not take
effect) with synthetic series whose correlation structure we control: two
strongly-correlated blocks ('trump', 'fed') plus one lone noise factor.

We assert the embedding is well-formed (one point per factor, finite coords,
low MDS stress) and that points in the same synthetic block land in the same
cluster — the geometry must recover the structure we baked into the DGP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import ml_hub_router as mh
from pfm.terminal.factor_clusters import _FactorMeta


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(mh.router)
    return TestClient(app)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_history(seed: int = 7) -> dict[str, pd.Series]:
    """Two correlated 3-factor blocks + one uncorrelated lone factor."""
    rng = np.random.default_rng(seed)
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    def _block() -> tuple[pd.Series, pd.Series, pd.Series]:
        latent = rng.standard_normal(n) * 0.5
        out = []
        for _ in range(3):
            innov = latent + rng.standard_normal(n) * 0.05
            out.append(pd.Series(_logistic(np.cumsum(innov) * 0.10), index=idx))
        return out[0], out[1], out[2]

    t_a, t_b, t_c = _block()
    f_a, f_b, f_c = _block()
    lone = pd.Series(_logistic(np.cumsum(rng.standard_normal(n) * 0.5) * 0.10), index=idx)
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
    # Patch on the consuming module's namespace, not the source module.
    monkeypatch.setattr(mh, "_load_cached_history", _make_history)
    monkeypatch.setattr(mh, "_load_factor_meta", _make_meta)
    # Clear the shared TTL cache between tests so query variants (and the
    # degraded-mode test's empty history) don't collide on a cached payload.
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()


# --- endpoint tests ---------------------------------------------------------


def test_factor_map_returns_one_point_per_factor(client: TestClient) -> None:
    r = client.get("/ml/factor-map")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded_mode"] is False
    assert body["n_factors"] == 7
    assert len(body["points"]) == 7
    ids = {p["factor_id"] for p in body["points"]}
    assert ids == {"trump_a", "trump_b", "trump_c", "fed_a", "fed_b", "fed_c", "lone"}


def test_coords_are_finite_and_stress_is_low(client: TestClient) -> None:
    body = client.get("/ml/factor-map").json()
    for p in body["points"]:
        assert np.isfinite(p["x"]) and np.isfinite(p["y"])
        assert p["vol"] >= 0.0
        assert p["n_obs"] > 0
    # MDS on a clean 2-block structure should fit well.
    assert body["method"] == "mds"
    assert body["stress"] is not None
    assert body["stress"] < 0.3


def test_blocks_land_in_same_cluster(client: TestClient) -> None:
    body = client.get("/ml/factor-map?min_corr=0.5").json()
    cluster_of = {p["factor_id"]: p["cluster"] for p in body["points"]}
    # The three 'trump' factors share a cluster; likewise the three 'fed' ones.
    assert cluster_of["trump_a"] == cluster_of["trump_b"] == cluster_of["trump_c"]
    assert cluster_of["fed_a"] == cluster_of["fed_b"] == cluster_of["fed_c"]
    # The two blocks are distinct, and the lone factor is off on its own.
    assert cluster_of["trump_a"] != cluster_of["fed_a"]
    assert cluster_of["lone"] not in {cluster_of["trump_a"], cluster_of["fed_a"]}


def test_cluster_summaries_present(client: TestClient) -> None:
    body = client.get("/ml/factor-map").json()
    assert body["n_clusters"] >= 2
    labels = {c["cluster"]: c for c in body["clusters"]}
    # Every point's cluster id appears in the summary list.
    for p in body["points"]:
        assert p["cluster"] in labels
    for c in body["clusters"]:
        assert 0.0 <= c["avg_intra_corr"] <= 1.0


def test_theme_filter(client: TestClient) -> None:
    body = client.get("/ml/factor-map?theme=macro").json()
    # Only the three 'fed' macro factors survive — but the map needs >=3.
    assert body["n_factors"] == 3
    assert {p["factor_id"] for p in body["points"]} == {"fed_a", "fed_b", "fed_c"}


def test_unknown_theme_404(client: TestClient) -> None:
    assert client.get("/ml/factor-map?theme=nonexistent").status_code == 404


def test_tsne_method(client: TestClient) -> None:
    body = client.get("/ml/factor-map?method=tsne").json()
    assert body["method"] == "tsne"
    assert body["stress"] is None  # t-SNE reports no Kruskal stress
    assert len(body["points"]) == 7


def test_degraded_mode_when_history_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mh, "_load_cached_history", lambda: {})
    body = client.get("/ml/factor-map").json()
    assert body["degraded_mode"] is True
    assert body["points"] == []
    assert body["reason"]


# --- explicit factor_ids selection ------------------------------------------


def test_factor_ids_restricts_to_chosen_subset(client: TestClient) -> None:
    # Pass a 3-factor subset by public factor_id; only those should appear.
    body = client.get("/ml/factor-map?factor_ids=trump_a,fed_a,lone").json()
    assert body["n_factors"] == 3
    assert {p["factor_id"] for p in body["points"]} == {"trump_a", "fed_a", "lone"}
    assert set(body["selected_factor_ids"]) == {"trump-a", "fed-a", "lone"}


def test_factor_ids_accepts_raw_slugs(client: TestClient) -> None:
    body = client.get("/ml/factor-map?factor_ids=trump-a,trump-b,trump-c").json()
    assert body["n_factors"] == 3
    assert {p["factor_id"] for p in body["points"]} == {"trump_a", "trump_b", "trump_c"}


def test_factor_ids_overrides_theme(client: TestClient) -> None:
    # theme=macro would normally restrict to fed_*; factor_ids wins.
    body = client.get("/ml/factor-map?theme=macro&factor_ids=trump_a,trump_b,trump_c").json()
    assert {p["factor_id"] for p in body["points"]} == {"trump_a", "trump_b", "trump_c"}


def test_factor_ids_fewer_than_three_is_422(client: TestClient) -> None:
    r = client.get("/ml/factor-map?factor_ids=trump_a,fed_a")
    assert r.status_code == 422
    assert "at least 3" in r.json()["detail"]


def test_factor_ids_unknown_entries_dropped_then_422(client: TestClient) -> None:
    # Only one resolves → fewer than 3 → 422.
    r = client.get("/ml/factor-map?factor_ids=trump_a,does_not_exist,nope")
    assert r.status_code == 422


def test_absent_factor_ids_keeps_prior_behaviour(client: TestClient) -> None:
    body = client.get("/ml/factor-map").json()
    assert body["n_factors"] == 7
    assert body["selected_factor_ids"] is None
