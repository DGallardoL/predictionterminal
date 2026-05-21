"""Tests for ``pfm.ml_factor_importance_router`` — /ml/factor-importance.

The router is mounted on a fresh :class:`FastAPI` app (no full ``pfm.main``
lifespan). The factor-history loader, factors.yml loader, AND the equity-return
fetch are monkeypatched on the *router module's namespace* (it imports those
names by value, so patching the source modules would not take effect) with a
synthetic DGP we control: the target ``y`` is a known linear-plus-non-linear
function of exactly ONE factor (``f3``) — ``y = 1.5*f3 + 0.5*f3² + small noise``
— while every other factor is pure noise. We assert ``f3`` ranks #1 in
permutation importance and that the walk-forward OOS R² is clearly positive.

We never hit the network: ``get_log_returns`` is replaced with a closure that
returns synthetic returns aligned to the synthetic factor dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import ml_factor_importance_router as fi
from pfm.terminal.factor_clusters import _FactorMeta

N_DAYS = 260
SIGNAL_SLUG = "f3"


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _dates() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=N_DAYS, freq="D", tz="UTC")


def _make_history(seed: int = 11) -> dict[str, pd.Series]:
    """Six factors as random-walk probability series; only ``f3`` drives y."""
    rng = np.random.default_rng(seed)
    idx = _dates()
    hist: dict[str, pd.Series] = {}
    for i in range(6):
        innov = rng.standard_normal(N_DAYS) * 0.30
        probs = _logistic(np.cumsum(innov) * 0.15)
        hist[f"f{i}"] = pd.Series(probs, index=idx)
    return hist


def _delta_logit_of(slug: str, history: dict[str, pd.Series]) -> pd.Series:
    return fi._delta_logit(history[slug])


def _make_target(history: dict[str, pd.Series], seed: int = 23) -> pd.Series:
    """y = 1.5*Δlogit(f3) + 0.5*Δlogit(f3)^2 + small noise, on the f3 dates."""
    rng = np.random.default_rng(seed)
    f3 = _delta_logit_of(SIGNAL_SLUG, history)
    noise = pd.Series(rng.standard_normal(len(f3)) * 0.01, index=f3.index)
    y = 1.5 * f3 + 0.5 * (f3**2) + noise
    y.index = pd.to_datetime(y.index, utc=True).normalize()
    y.name = "r"
    return y


def _make_meta() -> dict[str, _FactorMeta]:
    return {
        f"f{i}": _FactorMeta(f"factor_{i}", f"f{i}", "macro", f"Synthetic factor {i}")
        for i in range(6)
    }


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(fi.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    history = _make_history()
    target = _make_target(history)

    monkeypatch.setattr(fi, "_load_cached_history", lambda: history)
    monkeypatch.setattr(fi, "_load_factor_meta", _make_meta)
    # Patch the equity fetch on the router namespace so no network is touched.
    monkeypatch.setattr(
        fi,
        "get_log_returns",
        lambda ticker, start, end, return_type="log": target.copy(),
    )
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()


# --- endpoint tests ---------------------------------------------------------


def test_signal_factor_ranks_first(client: TestClient) -> None:
    r = client.get("/ml/factor-importance?ticker=NVDA")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded_mode"] is False
    assert body["ticker"] == "NVDA"
    assert body["n_aligned_obs"] >= fi.MIN_ALIGNED_OBS
    assert body["model"] == "HistGBR walk-forward"
    assert body["caveat"] == fi.CAVEAT
    assert body["items"], "expected ranked items"
    # f3 is the one factor that drives y → it must top the importance ranking.
    assert body["items"][0]["factor_id"] == "factor_3"
    # Its importance should clearly exceed the runner-up noise factor.
    assert body["items"][0]["importance"] > body["items"][1]["importance"]


def test_oos_r2_clearly_positive_for_real_signal(client: TestClient) -> None:
    body = client.get("/ml/factor-importance?ticker=NVDA").json()
    # The GBM should beat the naive mean predictor handsomely on this DGP.
    assert body["oos_r2"] > 0.2


def test_items_have_expected_schema(client: TestClient) -> None:
    body = client.get("/ml/factor-importance?ticker=NVDA").json()
    for it in body["items"]:
        assert set(it) == {
            "factor_id",
            "name",
            "theme",
            "importance",
            "importance_std",
            "is_significant",
        }
        assert isinstance(it["importance"], float)
        assert it["importance_std"] >= 0.0
        assert it["theme"] == "macro"
        # is_significant must equal importance > 2*std.
        assert it["is_significant"] == (it["importance"] > 2.0 * it["importance_std"])


def test_oos_r2_interpretable_true_for_real_signal(client: TestClient) -> None:
    """On the f3-driven DGP the OOS R² is positive ⇒ interpretable is True."""
    body = client.get("/ml/factor-importance?ticker=NVDA").json()
    assert body["oos_r2"] > 0
    assert body["oos_r2_interpretable"] is True


def test_oos_r2_interpretable_false_for_pure_noise(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pure-noise target ⇒ negative OOS R², not interpretable, caveat warns."""
    history = fi._load_cached_history()
    idx = next(iter(history.values())).index
    rng = np.random.default_rng(99)
    noise = pd.Series(rng.standard_normal(len(idx)), index=idx, name="r")
    noise.index = pd.to_datetime(noise.index, utc=True).normalize()
    monkeypatch.setattr(
        fi, "get_log_returns", lambda ticker, start, end, return_type="log": noise.copy()
    )
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()
    body = client.get("/ml/factor-importance?ticker=NVDA").json()
    assert body["oos_r2"] is None or body["oos_r2"] <= 0
    assert body["oos_r2_interpretable"] is False
    assert "NOT interpretable" in body["caveat"]


def test_top_n_caps_result_length(client: TestClient) -> None:
    body = client.get("/ml/factor-importance?ticker=NVDA&top_n=3").json()
    assert len(body["items"]) == 3


def test_unknown_ticker_no_aligned_data_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Equity fetch returns data on dates that do not overlap the factor span.
    disjoint = pd.Series(
        np.zeros(80),
        index=pd.date_range("2030-01-01", periods=80, freq="D", tz="UTC"),
        name="r",
    )
    monkeypatch.setattr(
        fi, "get_log_returns", lambda ticker, start, end, return_type="log": disjoint
    )
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()
    r = client.get("/ml/factor-importance?ticker=ZZZZ")
    assert r.status_code == 422


def test_equity_source_failure_is_422(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ticker, start, end, return_type="log"):
        raise RuntimeError("all equity sources failed")

    monkeypatch.setattr(fi, "get_log_returns", _boom)
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()
    r = client.get("/ml/factor-importance?ticker=NOPE")
    assert r.status_code == 422


def test_degraded_mode_when_history_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fi, "_load_cached_history", lambda: {})
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()
    body = client.get("/ml/factor-importance?ticker=NVDA").json()
    assert body["degraded_mode"] is True
    assert body["items"] == []
    assert body["reason"]


def test_unknown_theme_404(client: TestClient) -> None:
    assert client.get("/ml/factor-importance?ticker=NVDA&theme=nonexistent").status_code == 404


# --- explicit factor_ids candidate pool -------------------------------------


def test_factor_ids_restricts_candidate_pool(client: TestClient) -> None:
    # Pass exactly three factors (including the signal f3) as the candidate pool.
    body = client.get(
        "/ml/factor-importance?ticker=NVDA&factor_ids=factor_1,factor_3,factor_5"
    ).json()
    ids = {it["factor_id"] for it in body["items"]}
    assert ids <= {"factor_1", "factor_3", "factor_5"}
    # The signal factor must still rank first within the restricted pool.
    assert body["items"][0]["factor_id"] == "factor_3"


def test_factor_ids_accepts_raw_slugs(client: TestClient) -> None:
    body = client.get("/ml/factor-importance?ticker=NVDA&factor_ids=f1,f3,f5").json()
    ids = {it["factor_id"] for it in body["items"]}
    assert ids <= {"factor_1", "factor_3", "factor_5"}


def test_factor_ids_overrides_theme(client: TestClient) -> None:
    # A nonexistent theme would normally 404; factor_ids overrides it.
    body = client.get(
        "/ml/factor-importance?ticker=NVDA&theme=nonexistent&factor_ids=f1,f3,f5"
    ).json()
    ids = {it["factor_id"] for it in body["items"]}
    assert ids <= {"factor_1", "factor_3", "factor_5"}


def test_factor_ids_fewer_than_three_is_422(client: TestClient) -> None:
    r = client.get("/ml/factor-importance?ticker=NVDA&factor_ids=f3,f1")
    assert r.status_code == 422
    assert "at least 3" in r.json()["detail"]


def test_absent_factor_ids_keeps_prior_behaviour(client: TestClient) -> None:
    body = client.get("/ml/factor-importance?ticker=NVDA").json()
    assert body["items"][0]["factor_id"] == "factor_3"
