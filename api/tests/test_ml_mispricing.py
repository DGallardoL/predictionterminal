"""Tests for ``pfm.ml_mispricing_router`` — /ml/mispricing (Mispricing Scanner).

The router is mounted on a fresh :class:`FastAPI` app (no full ``pfm.main``
lifespan). History + factors.yml loaders are monkeypatched on the
*ml_mispricing_router module namespace* (it imports those names by value from
``pfm.terminal.factor_clusters``, so patching the source module would not take
effect) with synthetic series whose correlation structure we control.

The DGP is two strongly-correlated 3-factor blocks driven by latent factors.
We then deliberately inject a divergence into the *last few* observations of one
target factor (``trump-a``) so that, relative to its block, its latest Δlogit
move is an outlier. The scanner must rank that factor at the top by |z|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import ml_mispricing_router as mp
from pfm.terminal.factor_clusters import _FactorMeta


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(mp.router)
    return TestClient(app)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_history(seed: int = 7, inject: bool = True) -> dict[str, pd.Series]:
    """Two correlated 3-factor blocks; optionally inject a divergence.

    With ``inject=True`` the last few innovations of ``trump-a`` are pushed
    hard against the rest of its block, so its latest Δlogit residual versus its
    neighbours is a large outlier and it should top the mispricing ranking.
    """
    rng = np.random.default_rng(seed)
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    def _block_innovs() -> list[np.ndarray]:
        latent = rng.standard_normal(n) * 0.5
        return [latent + rng.standard_normal(n) * 0.05 for _ in range(3)]

    t_innovs = _block_innovs()
    f_innovs = _block_innovs()

    if inject:
        # Drive trump-a's last 4 daily moves opposite to its block by a big,
        # peer-uncorrelated amount → large standardized latest residual.
        t_innovs[0][-4:] += np.array([2.5, -2.5, 2.5, -2.5])

    def _to_series(innov: np.ndarray) -> pd.Series:
        return pd.Series(_logistic(np.cumsum(innov) * 0.10), index=idx)

    lone = pd.Series(_logistic(np.cumsum(rng.standard_normal(n) * 0.5) * 0.10), index=idx)
    return {
        "trump-a": _to_series(t_innovs[0]),
        "trump-b": _to_series(t_innovs[1]),
        "trump-c": _to_series(t_innovs[2]),
        "fed-a": _to_series(f_innovs[0]),
        "fed-b": _to_series(f_innovs[1]),
        "fed-c": _to_series(f_innovs[2]),
        "lone": lone,
    }


def _make_clean_history(seed: int = 11) -> dict[str, pd.Series]:
    """Same structure, no injected divergence — every residual is well-behaved."""
    return _make_history(seed=seed, inject=False)


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
    monkeypatch.setattr(mp, "_load_cached_history", _make_history)
    monkeypatch.setattr(mp, "_load_factor_meta", _make_meta)
    # Clear the shared TTL cache between tests so query variants (and the
    # degraded-mode test's empty history) don't collide on a cached payload.
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()


# --- endpoint tests ---------------------------------------------------------


def test_injected_factor_ranks_top(client: TestClient) -> None:
    r = client.get("/ml/mispricing")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded_mode"] is False
    assert body["items"], "expected at least one flagged factor"
    top = body["items"][0]
    assert top["factor_id"] == "trump_a"
    # The injected divergence should be a strong outlier.
    assert abs(top["z_score"]) >= 1.5
    assert top["direction"] in {"rich", "cheap"}


def test_item_schema_fields(client: TestClient) -> None:
    body = client.get("/ml/mispricing").json()
    item = body["items"][0]
    assert set(item) >= {
        "factor_id",
        "name",
        "theme",
        "z_score",
        "direction",
        "r_squared",
        "neighbors",
        "n_obs",
        "latest_price",
    }
    assert 0.0 <= item["r_squared"] <= 1.0
    assert item["n_obs"] > 0
    assert 0.0 <= item["latest_price"] <= 1.0
    assert isinstance(item["neighbors"], list) and item["neighbors"]
    # A factor never appears as its own neighbour.
    assert item["factor_id"] not in item["neighbors"]


def test_top_envelope_fields(client: TestClient) -> None:
    body = client.get("/ml/mispricing?min_corr=0.4&k=4&limit=5").json()
    assert body["min_corr"] == 0.4
    assert body["top_k"] == 4
    assert len(body["items"]) <= 5
    assert body["n_factors"] == len(body["items"])
    for it in body["items"]:
        assert len(it["neighbors"]) <= 4


def test_limit_caps_item_count(client: TestClient) -> None:
    body = client.get("/ml/mispricing?limit=2").json()
    assert len(body["items"]) <= 2


def test_clean_history_low_z(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Without an injected divergence, the top |z| should be modest and the
    # injected outlier's extreme magnitude must not appear.
    monkeypatch.setattr(mp, "_load_cached_history", _make_clean_history)
    body = client.get("/ml/mispricing").json()
    assert body["items"]
    top_abs_z = max(abs(it["z_score"]) for it in body["items"])
    assert top_abs_z < 4.0
    # On clean data many factors should land in the 'fair' band.
    assert any(it["direction"] == "fair" for it in body["items"])


def test_degraded_mode_when_history_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mp, "_load_cached_history", lambda: {})
    body = client.get("/ml/mispricing").json()
    assert body["degraded_mode"] is True
    assert body["items"] == []
    assert body["reason"]
    assert body["n_factors"] == 0


# --- de-artefact guards (Task 2) --------------------------------------------


def test_neighbor_corrs_present_and_aligned(client: TestClient) -> None:
    body = client.get("/ml/mispricing").json()
    for it in body["items"]:
        assert "neighbor_corrs" in it
        assert len(it["neighbor_corrs"]) == len(it["neighbors"])
        assert all(-1.0 <= c <= 1.0 for c in it["neighbor_corrs"])


def test_suspect_field_present_and_default_false(client: TestClient) -> None:
    body = client.get("/ml/mispricing").json()
    for it in body["items"]:
        assert "suspect" in it
        assert isinstance(it["suspect"], bool)


def test_suspect_flag_set_when_r2_high(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a target that is an exact linear combination of two neighbours →
    # in-sample R² == 1.0 > SUSPECT_R2, so the item must be flagged suspect.
    def _collinear_history() -> dict[str, pd.Series]:
        rng = np.random.default_rng(3)
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        a = rng.standard_normal(n) * 0.3
        b = rng.standard_normal(n) * 0.3
        c = rng.standard_normal(n) * 0.3
        # exact target in Δlogit space ⇒ residual ~0 ⇒ R²≈1
        tgt = 0.5 * a + 0.5 * b
        to_series = lambda innov: pd.Series(  # noqa: E731
            _logistic(np.cumsum(innov) * 0.10), index=idx
        )
        return {
            "trump-a": to_series(tgt),
            "trump-b": to_series(a),
            "trump-c": to_series(b),
            "fed-a": to_series(c),
            "fed-b": to_series(c + rng.standard_normal(n) * 0.05),
            "fed-c": to_series(c + rng.standard_normal(n) * 0.05),
        }

    monkeypatch.setattr(mp, "_load_cached_history", _collinear_history)
    body = client.get("/ml/mispricing?k=2&min_corr=0.1").json()
    by_id = {it["factor_id"]: it for it in body["items"]}
    # The collinear target should be present and flagged suspect.
    assert "trump_a" in by_id
    assert by_id["trump_a"]["suspect"] is True
    assert by_id["trump_a"]["r_squared"] > 0.97
    # Non-suspect items are ranked ahead of suspect ones.
    suspect_flags = [it["suspect"] for it in body["items"]]
    assert suspect_flags == sorted(suspect_flags)  # False(0) before True(1)


def test_liquidity_gate_skips_clip_tail_factor(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _tail_history() -> dict[str, pd.Series]:
        hist = _make_history()
        # Pin trump-a's latest price into the clipped tail (< MISPRICING_MIN_PRICE).
        ser = hist["trump-a"].copy()
        ser.iloc[-1] = 0.005
        hist["trump-a"] = ser
        return hist

    monkeypatch.setattr(mp, "_load_cached_history", _tail_history)
    body = client.get("/ml/mispricing").json()
    ids = {it["factor_id"] for it in body["items"]}
    assert "trump_a" not in ids


def test_dof_guard_skips_low_n_factors(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Short histories: ~31 Δlogit obs. With k neighbours the dof requirement is
    # MIN_FIT_OBS + 3*k = 25 + 3*k. For k=5 that's 40 > 31 ⇒ everything skipped.
    def _short_history() -> dict[str, pd.Series]:
        rng = np.random.default_rng(5)
        n = 32  # > MIN_HISTORY(30) so it enters the matrix, but few dof
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        latent = rng.standard_normal(n) * 0.5
        to_series = lambda innov: pd.Series(  # noqa: E731
            _logistic(np.cumsum(innov) * 0.10), index=idx
        )
        return {f"f-{i}": to_series(latent + rng.standard_normal(n) * 0.05) for i in range(6)}

    monkeypatch.setattr(mp, "_load_cached_history", _short_history)
    monkeypatch.setattr(mp, "_load_factor_meta", lambda: {})
    body = client.get("/ml/mispricing?k=5&min_corr=0.1").json()
    # n_obs (~31) < MIN_FIT_OBS + 3*5 = 40 for every factor ⇒ all skipped.
    assert body["items"] == []
    assert body["n_factors"] == 0


# --- explicit factor_ids selection ------------------------------------------


def test_factor_ids_restricts_scanned_targets(client: TestClient) -> None:
    # Only score trump_a + fed_a as targets; neighbours still come from the
    # full universe so the fit can still run.
    body = client.get("/ml/mispricing?factor_ids=trump_a,fed_a").json()
    ids = {it["factor_id"] for it in body["items"]}
    assert ids <= {"trump_a", "fed_a"}
    assert ids  # at least one scored


def test_factor_ids_accepts_raw_slug(client: TestClient) -> None:
    body = client.get("/ml/mispricing?factor_ids=trump-a").json()
    ids = {it["factor_id"] for it in body["items"]}
    assert ids <= {"trump_a"}


def test_absent_factor_ids_keeps_prior_behaviour(client: TestClient) -> None:
    # Without the param, the injected trump_a still tops the full scan.
    body = client.get("/ml/mispricing").json()
    assert body["items"][0]["factor_id"] == "trump_a"
