"""Tests for ``pfm.ml_latent_factors_router`` — /ml/latent-factors (PCA).

The router is mounted on a fresh :class:`FastAPI` app (no full ``pfm.main``
lifespan). History + factors.yml loaders are monkeypatched on the
*ml_latent_factors_router module namespace* (it imports those names by value
from ``pfm.terminal.factor_clusters``, so patching the source module would not
take effect) with a synthetic series whose *low-rank* structure we control: two
latent drivers, each shared by a block of three factors, plus a small amount of
idiosyncratic noise.

We assert the PCA recovers that structure: the top-2 PCs explain the large
majority of variance, factors in the same block share the sign of their loading
on the dominant component, and a factor with an injected idiosyncratic spike in
the last observation surfaces with a large ``|resid_z|``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import ml_latent_factors_router as lf
from pfm.terminal.factor_clusters import _FactorMeta


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(lf.router)
    return TestClient(app)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_history(seed: int = 11, spike: bool = False) -> dict[str, pd.Series]:
    """Two latent drivers, each shared by a 3-factor block + idiosyncratic noise.

    The factor returns are built directly in Δlogit space (cumsum of returns
    becomes a logit path, then mapped back through the logistic so the loaders
    see realistic probability series). ``spike`` injects a large idiosyncratic
    jump into the final observation of ``trump-a`` so its residual ``z`` blows
    up while the common-factor reconstruction stays put.
    """
    rng = np.random.default_rng(seed)
    n = 240
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    f1 = rng.standard_normal(n) * 0.20  # latent driver 1 (politics bloc)
    f2 = rng.standard_normal(n) * 0.20  # latent driver 2 (macro bloc)

    def _series(driver: np.ndarray, sign: float, spike_last: bool = False) -> pd.Series:
        innov = sign * driver + rng.standard_normal(n) * 0.02
        if spike_last:
            innov = innov.copy()
            innov[-1] += 1.5  # big idiosyncratic move on the last day
        return pd.Series(_logistic(np.cumsum(innov)), index=idx)

    return {
        "trump-a": _series(f1, 1.0, spike_last=spike),
        "trump-b": _series(f1, 1.0),
        "trump-c": _series(f1, 1.0),
        "fed-a": _series(f2, 1.0),
        "fed-b": _series(f2, 1.0),
        "fed-c": _series(f2, 1.0),
    }


def _make_meta() -> dict[str, _FactorMeta]:
    return {
        "trump-a": _FactorMeta("trump_a", "trump-a", "politics", "Trump out by June"),
        "trump-b": _FactorMeta("trump_b", "trump-b", "politics", "Trump impeached 2026"),
        "trump-c": _FactorMeta("trump_c", "trump-c", "politics", "Trump removed by Senate"),
        "fed-a": _FactorMeta("fed_a", "fed-a", "macro", "Fed cuts 25bps July"),
        "fed-b": _FactorMeta("fed_b", "fed-b", "macro", "Fed cuts 25bps Sept"),
        "fed-c": _FactorMeta("fed_c", "fed-c", "macro", "Fed cuts 50bps July"),
    }


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch on the consuming module's namespace, not the source module.
    monkeypatch.setattr(lf, "_load_cached_history", _make_history)
    monkeypatch.setattr(lf, "_load_factor_meta", _make_meta)
    # Clear the shared TTL cache between tests so query variants (and the
    # degraded-mode test's empty history) don't collide on a cached payload.
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()


# --- endpoint tests ---------------------------------------------------------


def test_basic_shape(client: TestClient) -> None:
    body = client.get("/ml/latent-factors?k=5").json()
    assert body["degraded_mode"] is False
    assert body["n_factors"] == 6
    assert body["n_obs"] > 0
    # k capped at n_factors - 1 = 5 here.
    assert body["k"] == 5
    assert len(body["components"]) == 5
    # One residual item per factor (well under the 40-item cap).
    ids = {r["factor_id"] for r in body["residuals"]}
    assert ids == {"trump_a", "trump_b", "trump_c", "fed_a", "fed_b", "fed_c"}
    for r in body["residuals"]:
        assert len(r["loadings"]) == body["k"]


def test_top_two_pcs_dominate_variance(client: TestClient) -> None:
    body = client.get("/ml/latent-factors?k=5").json()
    comps = body["components"]
    # Two latent drivers → PC1+PC2 should explain the large majority of variance.
    assert comps[1]["cum_var"] > 0.8
    # Variance is reported in descending order.
    evrs = [c["explained_var"] for c in comps]
    assert evrs == sorted(evrs, reverse=True)
    # Cumulative is monotone non-decreasing and bounded by 1.
    cums = [c["cum_var"] for c in comps]
    assert cums == sorted(cums)
    assert cums[-1] <= 1.0


def test_block_members_share_loading_sign(client: TestClient) -> None:
    body = client.get("/ml/latent-factors?k=5").json()
    load = {r["factor_id"]: r["loadings"] for r in body["residuals"]}
    # PC1 loadings of the trump block agree in sign with each other; likewise fed.
    trump_pc1 = [load["trump_a"][0], load["trump_b"][0], load["trump_c"][0]]
    fed_pc1 = [load["fed_a"][0], load["fed_b"][0], load["fed_c"][0]]

    # Each block is internally consistent on at least one of the top-2 PCs.
    def _same_sign(vals: list[float]) -> bool:
        return all(v > 0 for v in vals) or all(v < 0 for v in vals)

    trump_pc2 = [load["trump_a"][1], load["trump_b"][1], load["trump_c"][1]]
    fed_pc2 = [load["fed_a"][1], load["fed_b"][1], load["fed_c"][1]]
    assert _same_sign(trump_pc1) or _same_sign(trump_pc2)
    assert _same_sign(fed_pc1) or _same_sign(fed_pc2)


def test_component_labels_have_names(client: TestClient) -> None:
    body = client.get("/ml/latent-factors?k=2").json()
    pc1 = body["components"][0]
    # Dominant component must split the universe into named +/- poles.
    assert pc1["pc"] == 1
    assert pc1["top_positive"] or pc1["top_negative"]
    all_names = set(pc1["top_positive"]) | set(pc1["top_negative"])
    assert all_names  # at least one labelled contract


def test_idiosyncratic_spike_surfaces(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lf, "_load_cached_history", lambda: _make_history(spike=True))
    # Keep k at the true rank (2 latent drivers): the residual then captures the
    # idiosyncratic part, where the injected spike must dominate.
    body = client.get("/ml/latent-factors?k=2").json()
    resid = {r["factor_id"]: abs(r["resid_z"]) for r in body["residuals"]}
    # The spiked factor should have the most extreme standardized residual and
    # rank first in the |z|-sorted list.
    assert body["residuals"][0]["factor_id"] == "trump_a"
    assert resid["trump_a"] > 3.0
    assert len(body["residuals"][0]["loadings"]) == 2
    assert resid["trump_a"] > max(v for fid, v in resid.items() if fid != "trump_a")


def test_k_capping_when_k_exceeds_n_factors(client: TestClient) -> None:
    # k=20 but only 6 factors → capped at n_factors - 1 = 5.
    body = client.get("/ml/latent-factors?k=20").json()
    assert body["k"] == 5
    assert len(body["components"]) == 5
    for r in body["residuals"]:
        assert len(r["loadings"]) == 5


def test_theme_filter(client: TestClient) -> None:
    # Macro theme has only 3 factors → k capped at 2.
    body = client.get("/ml/latent-factors?theme=macro&k=5").json()
    assert body["n_factors"] == 3
    assert {r["factor_id"] for r in body["residuals"]} == {"fed_a", "fed_b", "fed_c"}
    assert body["k"] == 2


def test_unknown_theme_404(client: TestClient) -> None:
    assert client.get("/ml/latent-factors?theme=nonexistent").status_code == 404


def test_degraded_mode_when_history_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lf, "_load_cached_history", lambda: {})
    body = client.get("/ml/latent-factors").json()
    assert body["degraded_mode"] is True
    assert body["components"] == []
    assert body["residuals"] == []
    assert body["k"] == 0
    assert body["reason"]


# --- explicit factor_ids selection ------------------------------------------


def test_factor_ids_restricts_pca_to_subset(client: TestClient) -> None:
    body = client.get("/ml/latent-factors?k=2&factor_ids=trump_a,trump_b,fed_a").json()
    assert body["n_factors"] == 3
    assert {r["factor_id"] for r in body["residuals"]} == {"trump_a", "trump_b", "fed_a"}
    assert set(body["selected_factor_ids"]) == {"trump-a", "trump-b", "fed-a"}


def test_factor_ids_accepts_raw_slugs(client: TestClient) -> None:
    body = client.get("/ml/latent-factors?k=2&factor_ids=trump-a,trump-b,fed-a").json()
    assert {r["factor_id"] for r in body["residuals"]} == {"trump_a", "trump_b", "fed_a"}


def test_factor_ids_overrides_theme(client: TestClient) -> None:
    body = client.get(
        "/ml/latent-factors?k=2&theme=macro&factor_ids=trump_a,trump_b,trump_c"
    ).json()
    assert {r["factor_id"] for r in body["residuals"]} == {"trump_a", "trump_b", "trump_c"}


def test_factor_ids_fewer_than_k_plus_one_is_422(client: TestClient) -> None:
    # k=2 requires >= 3 factors; only 2 supplied.
    r = client.get("/ml/latent-factors?k=2&factor_ids=trump_a,trump_b")
    assert r.status_code == 422
    assert "at least 3" in r.json()["detail"]


def test_absent_factor_ids_keeps_prior_behaviour(client: TestClient) -> None:
    body = client.get("/ml/latent-factors?k=5").json()
    assert body["n_factors"] == 6
    assert body["selected_factor_ids"] is None
