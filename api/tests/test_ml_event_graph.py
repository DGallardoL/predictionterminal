"""Tests for ``pfm.ml_event_graph_router`` — /ml/event-graph (Event Graph).

The router is mounted on a fresh :class:`FastAPI` app (no full ``pfm.main``
lifespan). History + factors.yml loaders are monkeypatched on the
*ml_event_graph_router module namespace* (it imports those names by value, so
patching the source module would not take effect) with a synthetic DGP whose
structure we control:

  * a **contemporaneous block** (``trump-a/b/c``) driven by a shared latent so
    they co-move within-day → comove edges,
  * a **known causal link**: ``B_t``'s innovation is ``0.7 · A_{t-1} + noise``
    so ``A`` Granger-causes ``B`` (a directed lead edge A→B, ideally not B→A),
  * a lone noise factor.

We assert the directed/undirected edges, finite MDS coords, and that the OOS R²
is a finite float or None. statsmodels VAR is well-conditioned here because we
use n≥240 observations and deterministic seeds.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import ml_event_graph_router as eg
from pfm.terminal.factor_clusters import _FactorMeta


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(eg.router)
    return TestClient(app)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_history(seed: int = 11) -> dict[str, pd.Series]:
    """Contemporaneous block + a known A→B lead link + a lone factor.

    Returns are constructed in Δlogit space, then integrated and squashed back
    to a probability series so the router's Δlogit pipeline recovers the DGP.
    """
    rng = np.random.default_rng(seed)
    n = 260
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    # Contemporaneous block: shared latent → same-day co-movement.
    latent = rng.standard_normal(n) * 0.30
    blk = []
    for _ in range(3):
        r = latent + rng.standard_normal(n) * 0.05
        blk.append(r)

    # Causal pair: A is exogenous noise; B_t depends on A_{t-1}.
    a = rng.standard_normal(n) * 0.30
    b = np.zeros(n)
    for t in range(1, n):
        b[t] = 0.7 * a[t - 1] + rng.standard_normal() * 0.10

    lone = rng.standard_normal(n) * 0.30

    def _to_prices(r: np.ndarray) -> pd.Series:
        # integrate Δlogit then inverse-logit back to a probability series
        return pd.Series(_logistic(np.cumsum(r) * 0.10), index=idx)

    return {
        "trump-a": _to_prices(blk[0]),
        "trump-b": _to_prices(blk[1]),
        "trump-c": _to_prices(blk[2]),
        "drive-a": _to_prices(a),
        "drive-b": _to_prices(b),
        "lone": _to_prices(lone),
    }


def _make_meta() -> dict[str, _FactorMeta]:
    return {
        "trump-a": _FactorMeta("trump_a", "trump-a", "politics", "Trump out by June"),
        "trump-b": _FactorMeta("trump_b", "trump-b", "politics", "Trump impeached 2026"),
        "trump-c": _FactorMeta("trump_c", "trump-c", "politics", "Trump removed by Senate"),
        "drive-a": _FactorMeta("drive_a", "drive-a", "macro", "Leading driver A"),
        "drive-b": _FactorMeta("drive_b", "drive-b", "macro", "Lagging follower B"),
        "lone": _FactorMeta("lone", "lone", "other", "Unrelated noise factor"),
    }


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eg, "_load_cached_history", _make_history)
    monkeypatch.setattr(eg, "_load_factor_meta", _make_meta)
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()


# --- endpoint tests ---------------------------------------------------------


def test_basic_shape_and_nodes(client: TestClient) -> None:
    r = client.get("/ml/event-graph")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded_mode"] is False
    assert body["n_nodes"] == 6
    assert len(body["nodes"]) == 6
    assert body["lag_order"] >= 1
    ids = {nd["factor_id"] for nd in body["nodes"]}
    assert ids == {"trump_a", "trump_b", "trump_c", "drive_a", "drive_b", "lone"}


def test_nodes_carry_finite_coords_and_prices(client: TestClient) -> None:
    body = client.get("/ml/event-graph").json()
    for nd in body["nodes"]:
        assert math.isfinite(nd["x"]) and math.isfinite(nd["y"])
        assert 0.0 < nd["market_price"] < 1.0
        assert 0.0 < nd["model_price"] < 1.0
        assert math.isfinite(nd["mispricing_z"])
        assert nd["centrality"] >= 0


def test_directed_lead_edge_a_drives_b(client: TestClient) -> None:
    body = client.get("/ml/event-graph?edge_threshold=0.0").json()
    lead = {(e["source"], e["target"]) for e in body["edges"] if e["kind"] == "lead"}
    # The DGP wires B_t = 0.7*A_{t-1}+noise → A Granger-causes B.
    assert ("drive_a", "drive_b") in lead
    # Ideally the reverse link is absent (no feedback from B to A in the DGP).
    assert ("drive_b", "drive_a") not in lead
    # All lead edges carry the VAR lag order, comove edges carry None.
    for e in body["edges"]:
        if e["kind"] == "lead":
            assert e["lag"] == body["lag_order"]
        else:
            assert e["lag"] is None
    assert body["n_lead_edges"] >= 1


def test_comove_edges_within_block(client: TestClient) -> None:
    body = client.get("/ml/event-graph?edge_threshold=0.4").json()
    comove = {frozenset((e["source"], e["target"])) for e in body["edges"] if e["kind"] == "comove"}
    # The three trump factors share a contemporaneous latent → mutual comove.
    assert frozenset(("trump_a", "trump_b")) in comove
    assert frozenset(("trump_a", "trump_c")) in comove
    assert frozenset(("trump_b", "trump_c")) in comove
    assert body["n_comove_edges"] == len([e for e in body["edges"] if e["kind"] == "comove"])


def test_oos_r2_is_finite_or_none(client: TestClient) -> None:
    body = client.get("/ml/event-graph").json()
    assert (body["oos_r2"] is None) or math.isfinite(body["oos_r2"])
    assert body["caveat"]


def test_communities_assigned(client: TestClient) -> None:
    body = client.get("/ml/event-graph").json()
    comm_of = {nd["factor_id"]: nd["community"] for nd in body["nodes"]}
    assert comm_of["trump_a"] == comm_of["trump_b"] == comm_of["trump_c"]


def test_max_nodes_capping(client: TestClient) -> None:
    body = client.get("/ml/event-graph?max_nodes=4").json()
    assert body["n_nodes"] == 4
    assert len(body["nodes"]) == 4


def test_max_nodes_below_three_rejected(client: TestClient) -> None:
    # max_nodes < 3 is rejected by the Query validator (422).
    assert client.get("/ml/event-graph?max_nodes=2").status_code == 422


def test_theme_filter_and_unknown(client: TestClient) -> None:
    body = client.get("/ml/event-graph?theme=politics").json()
    assert body["n_nodes"] == 3
    assert {nd["factor_id"] for nd in body["nodes"]} == {"trump_a", "trump_b", "trump_c"}
    assert client.get("/ml/event-graph?theme=nonexistent").status_code == 404


def test_degraded_mode_when_history_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eg, "_load_cached_history", lambda: {})
    body = client.get("/ml/event-graph").json()
    assert body["degraded_mode"] is True
    assert body["nodes"] == []
    assert body["edges"] == []
    assert body["reason"]
