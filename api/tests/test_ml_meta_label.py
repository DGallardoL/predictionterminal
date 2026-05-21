"""Tests for ``pfm.ml_meta_label_router`` — /ml/meta-label (triple-barrier meta-labeling).

The router is mounted on a fresh :class:`FastAPI` app (no full ``pfm.main``
lifespan). History + factors.yml loaders are monkeypatched on the
*ml_meta_label_router module namespace* (it imports those names by value, so
patching the source module would not take effect).

The core DGP is engineered so the naive z-score reversion signal is profitable
ONLY in a learnable sub-condition: **mean-reversion works in low-vol regimes and
fails in high-vol regimes**. We build a price series of alternating low-vol and
high-vol blocks where a dislocation in a low-vol block reliably reverts (the
primary trade wins) while a dislocation in a high-vol block keeps drifting (the
primary trade is stopped out). The meta-model should learn the regime split,
so meta_hit_rate > primary_hit_rate, n_meta < n_primary, and oos_auc > 0.5.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import ml_meta_label_router as ml
from pfm.terminal.factor_clusters import _FactorMeta


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(ml.router)
    return TestClient(app)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_regime_history(seed: int = 11) -> dict[str, pd.Series]:
    """Build a probability series where reversion wins in low-vol, fails in high-vol.

    The series is a long sequence of blocks. Each block holds a stable baseline
    for ~25 bars, then injects a dislocation (a z-score trigger). In a *low-vol*
    block the dislocation reverts to baseline within the horizon (primary win); in
    a *high-vol* block the dislocation is the start of a sustained drift, so the
    primary reversion trade is stopped out (primary loss). The block's noise level
    (rolling vol) is the learnable discriminator.
    """
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    anchor = 0.0  # the calm baseline mean-reverts to this anchor
    block = 0
    while len(vals) < 4000:
        low_vol = block % 2 == 0
        noise = 0.010 if low_vol else 0.060
        level = anchor
        # Calm baseline: Ornstein-Uhlenbeck pull to the anchor so it does NOT
        # random-walk away and fire spurious triggers between dislocations.
        for _ in range(22):
            level += 0.6 * (anchor - level) + rng.normal(0.0, noise)
            vals.append(level)
        # Dislocation: a sharp jump that fires a high-|z| reversion trigger.
        # Alternate the jump sign across dislocations so the series stays bounded.
        jump = 0.45 if (block % 4 < 2) else -0.45
        level += jump
        vals.append(level)
        if low_vol:
            # Revert toward the anchor → the reversion trade hits its profit barrier.
            for _ in range(9):
                level += 0.55 * (anchor - level) + rng.normal(0.0, noise)
                vals.append(level)
        else:
            # Keep drifting in the dislocation direction → reversion is stopped out.
            for _ in range(9):
                level += 0.06 * np.sign(jump) + rng.normal(0.0, noise)
                vals.append(level)
            anchor = level  # the high-vol regime relevels the baseline
        block += 1

    idx = pd.date_range("2022-01-01", periods=len(vals), freq="D")
    prob = pd.Series(_logistic(np.asarray(vals) * 0.5), index=idx)
    # Keep probabilities in a sane interior band so Δlogit stays well-defined.
    prob = prob.clip(0.05, 0.95)
    return {"regime": prob}


def _make_meta() -> dict[str, _FactorMeta]:
    return {"regime": _FactorMeta("regime_factor", "regime", "macro", "Regime reversion test")}


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml, "_load_cached_history", _make_regime_history)
    monkeypatch.setattr(ml, "_load_factor_meta", _make_meta)
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()


# --- core lift test ---------------------------------------------------------


def test_meta_filter_lifts_precision_and_skips_bad_trades(client: TestClient) -> None:
    # sklearn can emit narrow convergence/data warnings; scope them out here only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = client.get("/ml/meta-label?entry_z=1.5&profit_target_sigma=2&stop_loss_sigma=2")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded_mode"] is False
    assert body["factor_id"] == "regime_factor"

    # (a) the meta-filter raises the hit-rate over the naive primary.
    assert body["meta_hit_rate"] > body["primary_hit_rate"]
    assert body["precision_lift"] > 0.0
    # (b) it takes strictly fewer trades (it skips the bad regime).
    assert 0 < body["n_meta"] < body["n_primary"]
    # (c) the meta-model has out-of-fold discriminative power.
    assert body["oos_auc"] is not None
    assert body["oos_auc"] > 0.5


def test_features_and_caveat_present(client: TestClient) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        body = client.get("/ml/meta-label").json()
    names = {f["name"] for f in body["features"]}
    assert names == {"z", "rolling_vol", "short_momentum", "vol_regime_pct"}
    for f in body["features"]:
        assert 0.0 <= f["importance"] <= 1.0
    assert "meta-labeling" in body["caveat"].lower()


def test_explicit_factor_slug(client: TestClient) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        body = client.get("/ml/meta-label?factor=regime").json()
    assert body["factor_id"] == "regime_factor"
    assert body["n_primary"] > 0


def test_unknown_factor_404(client: TestClient) -> None:
    assert client.get("/ml/meta-label?factor=does-not-exist").status_code == 404


# --- degraded / graceful paths ----------------------------------------------


def test_degraded_mode_when_history_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ml, "_load_cached_history", lambda: {})
    body = client.get("/ml/meta-label").json()
    assert body["degraded_mode"] is True
    assert body["n_primary"] == 0
    assert body["reason"]
    assert body["features"] == []


def test_too_few_triggers_is_graceful_degraded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A short, calm series fires almost no |z|>=1.5 triggers → graceful degrade.
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=80, freq="D")
    flat = pd.Series(_logistic(np.cumsum(rng.normal(0, 0.01, 80))), index=idx)
    monkeypatch.setattr(ml, "_load_cached_history", lambda: {"regime": flat})
    body = client.get("/ml/meta-label?entry_z=3.5").json()
    assert body["degraded_mode"] is True
    assert body["n_meta"] == 0
    assert "trigger" in body["reason"].lower() or "trade" in body["reason"].lower()
