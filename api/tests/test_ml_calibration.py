"""Tests for ``pfm.ml_calibration_router`` — /ml/calibration.

The router is mounted on a fresh :class:`FastAPI` app (no full ``pfm.main``
lifespan). The resolved-data accessors (``fetch_resolved_markets`` and
``fetch_archive_market_detail``) are monkeypatched on the *ml_calibration_router
module namespace* — the module imports those names by value, so patching the
source ``pfm.archive.polymarket_archive`` would not take effect. No network IO
ever happens.

We inject a known favourite–longshot bias into the synthetic dataset: outcomes
are drawn with ``true_prob = price**1.5`` over a price grid. Because ``p**1.5``
sits *below* ``p`` for every ``p∈(0,1)``, longshots (cheap contracts) resolve
YES *less* often than priced — i.e. they are overpriced — which is exactly the
classic bias we want the endpoint to surface. We assert the reliability curve,
the monotone isotonic map (which pulls longshot prices down), and that the
Brier/ECE summaries are finite.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import ml_calibration_router as mc

# Price grid + samples-per-price for the synthetic DGP. Enough to clear the
# default min_samples=100 gate and to make the empirical rates stable.
_GRID = np.linspace(0.02, 0.98, 49)
_PER_PRICE = 200


def _synthetic_samples(seed: int = 0) -> tuple[list[float], list[int]]:
    """``(prices, outcomes)`` with an injected favourite–longshot bias.

    ``true_prob = price**1.5`` < ``price`` everywhere ⇒ markets resolve YES
    less often than their price implies, strongest at the cheap end.
    """
    rng = np.random.default_rng(seed)
    prices = np.repeat(_GRID, _PER_PRICE)
    true_prob = prices**1.5
    outcomes = (rng.random(prices.size) < true_prob).astype(int)
    return prices.tolist(), outcomes.tolist()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(mc.router)
    return TestClient(app)


# Captured before any patching so the accessor-seam tests can exercise the
# real ``_gather_samples`` (the autouse fixture stubs it for endpoint tests).
_REAL_GATHER = mc._gather_samples


@pytest.fixture(autouse=True)
def _patch_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: feed the synthetic biased dataset; clear the TTL cache."""
    monkeypatch.setattr(
        mc, "_gather_samples", lambda *, category, max_markets: _synthetic_samples()
    )
    from pfm import terminal as term

    term.TERMINAL_CACHE.clear()


# --- happy-path / algorithm tests -------------------------------------------


def test_returns_calibration_payload(client: TestClient) -> None:
    body = client.get("/ml/calibration").json()
    assert body["degraded_mode"] is False
    assert body["n_samples"] == len(_GRID) * _PER_PRICE
    assert body["n_bins"] == 10
    assert len(body["bins"]) >= 5
    for b in body["bins"]:
        assert 0.0 <= b["empirical"] <= 1.0
        assert 0.0 <= b["mean_pred"] <= 1.0
        assert b["count"] > 0
        # Wilson 95% interval present and brackets the empirical rate.
        assert 0.0 <= b["ci_low"] <= b["empirical"] <= b["ci_high"] <= 1.0


def test_default_min_samples_is_40(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """~50 resolved pairs must clear the (new) default gate, not degrade."""
    rng = np.random.default_rng(7)
    prices = rng.uniform(0.05, 0.95, 50)
    outcomes = (rng.random(50) < prices).astype(int)
    monkeypatch.setattr(
        mc,
        "_gather_samples",
        lambda *, category, max_markets: (prices.tolist(), outcomes.tolist()),
    )
    body = client.get("/ml/calibration").json()
    assert body["degraded_mode"] is False
    assert body["n_samples"] == 50


def test_wilson_interval_bounds_sane() -> None:
    # phat must always sit inside the interval, bounds within [0, 1].
    for k, n in [(0, 5), (3, 10), (10, 10), (1, 100)]:
        low, high = mc._wilson_interval(k, n)
        phat = k / n
        assert 0.0 <= low <= phat <= high <= 1.0


def test_wilson_interval_wide_for_sparse_bin() -> None:
    # k=1, n=2 is maximally uninformative — the interval must be very wide.
    low, high = mc._wilson_interval(1, 2)
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert high - low > 0.7
    # Empty bin degrades to the full [0, 1] range.
    assert mc._wilson_interval(0, 0) == (0.0, 1.0)


def test_reliability_longshot_bin_empirical_below_price(client: TestClient) -> None:
    """In the cheapest bin, the YES-rate must sit below the mean price."""
    body = client.get("/ml/calibration").json()
    cheap = min(body["bins"], key=lambda b: b["price_mid"])
    assert cheap["empirical"] < cheap["mean_pred"]


def test_brier_and_ece_finite(client: TestClient) -> None:
    body = client.get("/ml/calibration").json()
    assert body["brier"] is not None and np.isfinite(body["brier"])
    assert 0.0 <= body["brier"] <= 1.0
    assert body["ece"] is not None and np.isfinite(body["ece"])
    assert body["ece"] >= 0.0


def test_longshot_overpriced_favorite_signs(client: TestClient) -> None:
    body = client.get("/ml/calibration").json()
    # Longshots (price<0.2) resolve YES less often than priced ⇒ negative gap.
    assert body["longshot_bias"] is not None
    assert body["longshot_bias"] < 0.0
    # p**1.5 < p everywhere, so the favourite region is also below its price.
    assert body["favorite_bias"] is not None
    assert body["favorite_bias"] < 0.0


def test_calibrated_curve_is_monotone_and_pulls_longshots_down(
    client: TestClient,
) -> None:
    body = client.get("/ml/calibration").json()
    curve = body["calibrated_curve"]
    assert len(curve) >= 2
    xs = [pt["x"] for pt in curve]
    ys = [pt["y"] for pt in curve]
    assert xs == sorted(xs)
    # Isotonic regression is monotone non-decreasing.
    assert all(ys[i + 1] >= ys[i] - 1e-9 for i in range(len(ys) - 1))
    # Calibrated probability for a cheap price is pulled below the raw price.
    longshot = next(pt for pt in curve if abs(pt["x"] - 0.1) < 1e-6)
    assert longshot["y"] < longshot["x"]


def test_n_bins_param_respected(client: TestClient) -> None:
    body = client.get("/ml/calibration?n_bins=20").json()
    assert body["n_bins"] == 20
    # Every bin midpoint must fall on the 20-bin grid and be unique.
    mids = [b["price_mid"] for b in body["bins"]]
    assert len(mids) == len(set(mids))


# --- gating / degraded-mode tests -------------------------------------------


def test_min_samples_gating(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fewer pairs than ``min_samples`` ⇒ degraded mode, empty curve."""
    monkeypatch.setattr(
        mc,
        "_gather_samples",
        lambda *, category, max_markets: ([0.3, 0.6, 0.9], [0, 1, 1]),
    )
    body = client.get("/ml/calibration?min_samples=100").json()
    assert body["degraded_mode"] is True
    assert body["n_samples"] == 3
    assert body["bins"] == []
    assert body["calibrated_curve"] == []
    assert body["brier"] is None
    assert body["reason"]


def test_degraded_mode_when_no_resolved_data(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mc, "_gather_samples", lambda *, category, max_markets: ([], []))
    body = client.get("/ml/calibration").json()
    assert body["degraded_mode"] is True
    assert body["n_samples"] == 0
    assert body["calibrated_curve"] == []
    assert body["reason"]


# --- accessor-seam tests (exercise _gather_samples via patched archive) -----


def test_gather_samples_uses_patched_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_gather_samples`` must read the archive accessors on this namespace.

    We patch ``fetch_resolved_markets`` / ``fetch_archive_market_detail`` (not
    ``_gather_samples`` itself) and confirm clean YES/NO rows with a
    pre-resolution price are paired correctly while AMBIGUOUS rows are dropped.
    """
    rows = [
        {"slug": "mkt-yes", "resolution": "YES"},
        {"slug": "mkt-no", "resolution": "NO"},
        {"slug": "mkt-amb", "resolution": "AMBIGUOUS"},
        {"slug": None, "resolution": "YES"},
    ]
    details = {
        # history rows are [date, price, volume]; final ~1.0 is the settlement
        # print and must be skipped in favour of the pre-resolution price.
        "mkt-yes": {
            "history": [
                ["2025-01-01", 0.40, 1.0],
                ["2025-01-02", 0.55, 1.0],
                ["2025-01-03", 0.60, 1.0],
                ["2025-01-04", 0.70, 1.0],
                ["2025-01-05", 0.99, 1.0],
            ],
        },
        "mkt-no": {
            "history": [
                ["2025-01-01", 0.30, 1.0],
                ["2025-01-02", 0.25, 1.0],
                ["2025-01-03", 0.20, 1.0],
                ["2025-01-04", 0.15, 1.0],
                ["2025-01-05", 0.01, 1.0],
            ],
        },
    }
    monkeypatch.setattr(
        mc,
        "fetch_resolved_markets",
        lambda start, end, theme, limit, offset: rows,
    )
    monkeypatch.setattr(mc, "fetch_archive_market_detail", lambda slug: details[slug])
    monkeypatch.setattr(mc, "_gather_samples", _REAL_GATHER)

    prices, outcomes = mc._gather_samples(category=None, max_markets=600)
    assert len(prices) == 2
    assert outcomes == [1, 0]
    # 5-point series, horizon=3 ⇒ index max(0, 5-1-3)=1 ⇒ second price.
    assert prices == [0.55, 0.25]


def test_gather_samples_degrades_on_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a: object, **k: object) -> list[dict[str, object]]:
        raise RuntimeError("upstream down")

    monkeypatch.setattr(mc, "fetch_resolved_markets", _boom)
    monkeypatch.setattr(mc, "_gather_samples", _REAL_GATHER)
    prices, outcomes = mc._gather_samples(category=None, max_markets=600)
    assert prices == []
    assert outcomes == []
