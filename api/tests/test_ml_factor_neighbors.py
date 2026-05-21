"""Tests for ``pfm.ml_hub_router`` — /ml/factor-neighbors (hedge / de-dup finder).

The router is mounted on a fresh :class:`FastAPI` app (no full ``pfm.main``
lifespan). History + factors.yml loaders are monkeypatched on the *ml_hub_router
module namespace* (it imports those names by value from
``pfm.terminal.factor_clusters``, so patching the source module would not take
effect) with synthetic series whose correlation structure we control.

The DGP plants known signs: ``trump-a``/``trump-b``/``trump-c`` move together
(positive corr → duplicates of one another), while ``mirror-a`` is the logit
mirror of ``trump-a`` (strong negative corr → a hedge). A lone noise factor is
uncorrelated with everything.
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
    """One positively-correlated 3-factor block + an anti-correlated mirror.

    ``mirror-a`` is built from the *negated* latent path of the trump block, so
    its Δlogit returns are strongly negatively correlated with ``trump-a``.
    """
    rng = np.random.default_rng(seed)
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    latent = rng.standard_normal(n) * 0.5

    def _from_latent(sign: float) -> pd.Series:
        innov = sign * latent + rng.standard_normal(n) * 0.05
        return pd.Series(_logistic(np.cumsum(innov) * 0.10), index=idx)

    lone = pd.Series(_logistic(np.cumsum(rng.standard_normal(n) * 0.5) * 0.10), index=idx)
    return {
        "trump-a": _from_latent(1.0),
        "trump-b": _from_latent(1.0),
        "trump-c": _from_latent(1.0),
        "mirror-a": _from_latent(-1.0),
        "lone": lone,
    }


def _make_meta() -> dict[str, _FactorMeta]:
    return {
        "trump-a": _FactorMeta("trump_a", "trump-a", "politics", "Trump out by June"),
        "trump-b": _FactorMeta("trump_b", "trump-b", "politics", "Trump impeached 2026"),
        "trump-c": _FactorMeta("trump_c", "trump-c", "politics", "Trump removed by Senate"),
        "mirror-a": _FactorMeta("mirror_a", "mirror-a", "politics", "Trump stays in office"),
        "lone": _FactorMeta("lone", "lone", "other", "Unrelated noise factor"),
    }


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mh, "_load_cached_history", _make_history)
    monkeypatch.setattr(mh, "_load_factor_meta", _make_meta)
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()


# --- endpoint tests ---------------------------------------------------------


def test_neighbors_happy_path_recovers_planted_signs(client: TestClient) -> None:
    r = client.get("/ml/factor-neighbors?factor_id=trump_a")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded_mode"] is False
    assert body["factor_id"] == "trump_a"
    assert body["n_obs"] > 0
    assert body["vol"] >= 0.0

    dup_ids = {d["factor_id"] for d in body["duplicates"]}
    hedge_ids = {h["factor_id"] for h in body["hedges"]}
    # The positively-correlated block-mates show up as duplicates.
    assert {"trump_b", "trump_c"} <= dup_ids
    # The logit mirror shows up as a hedge.
    assert "mirror_a" in hedge_ids
    # A factor never appears as its own neighbour.
    assert "trump_a" not in dup_ids and "trump_a" not in hedge_ids


def test_duplicates_positive_hedges_negative_and_sorted(client: TestClient) -> None:
    body = client.get("/ml/factor-neighbors?factor_id=trump_a").json()
    dup_corrs = [d["corr"] for d in body["duplicates"]]
    hedge_corrs = [h["corr"] for h in body["hedges"]]
    # duplicates sorted descending (most-positive first).
    assert dup_corrs == sorted(dup_corrs, reverse=True)
    assert dup_corrs[0] > 0.5
    # hedges sorted ascending (most-negative first).
    assert hedge_corrs == sorted(hedge_corrs)
    assert hedge_corrs[0] < -0.5


def test_accepts_slug_as_well_as_factor_id(client: TestClient) -> None:
    # Passing the raw slug also resolves.
    body = client.get("/ml/factor-neighbors?factor_id=trump-a").json()
    assert body["factor_id"] == "trump_a"


def test_k_caps_each_side(client: TestClient) -> None:
    body = client.get("/ml/factor-neighbors?factor_id=trump_a&k=2").json()
    assert len(body["duplicates"]) <= 2
    assert len(body["hedges"]) <= 2


def test_latest_price_populated(client: TestClient) -> None:
    body = client.get("/ml/factor-neighbors?factor_id=trump_a").json()
    assert body["latest_price"] is not None
    assert 0.0 <= body["latest_price"] <= 1.0
    for item in body["duplicates"] + body["hedges"]:
        assert item["latest_price"] is None or 0.0 <= item["latest_price"] <= 1.0
        assert item["n_obs"] > 0


def test_unknown_factor_404(client: TestClient) -> None:
    assert client.get("/ml/factor-neighbors?factor_id=does_not_exist").status_code == 404


def test_min_obs_too_high_404(client: TestClient) -> None:
    # 200 daily probs → ~199 Δlogit obs; demand more than that.
    assert client.get("/ml/factor-neighbors?factor_id=trump_a&min_obs=500").status_code == 404


def test_degraded_mode_when_history_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mh, "_load_cached_history", lambda: {})
    body = client.get("/ml/factor-neighbors?factor_id=trump_a").json()
    assert body["degraded_mode"] is True
    assert body["duplicates"] == []
    assert body["hedges"] == []
    assert body["reason"]
