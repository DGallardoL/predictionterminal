"""Tests for the Brownian-bridge probability fan-chart endpoint.

The router is mounted on an ad-hoc FastAPI app so we don't have to touch
``pfm.main``. External IO (Polymarket Gamma + CLOB) is patched out via
``monkeypatch``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.terminal_prob_fan as tpf
from pfm.sources.polymarket import MarketMetadata


def _make_prices(p_today: float, n: int = 35, seed: int = 0) -> pd.DataFrame:
    """Build a fake daily YES-probability series ending today (UTC).

    Walks geometrically toward ``p_today`` from a slightly different start
    so realised vol is non-zero.
    """
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(datetime.now(tz=UTC)).normalize()
    idx = pd.date_range(end=today, periods=n, freq="D", tz="UTC")
    idx.name = "date"
    # Random walk in logit space, anchored at p_today on the last day.
    target_logit = float(np.log(p_today / (1 - p_today)))
    noise = rng.normal(0, 0.05, size=n).cumsum()
    noise = noise - noise[-1]  # zero out at the end
    logits = target_logit + noise
    probs = 1.0 / (1.0 + np.exp(-logits))
    return pd.DataFrame({"price": probs}, index=idx)


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Mount the router on a fresh FastAPI app with all IO patched."""
    app = FastAPI()
    app.include_router(tpf.router)

    # Patch the Polymarket client constructor to a no-op object — none of
    # its methods will actually be called because we patch the two consumers
    # below.
    class _FakeClient:
        def __init__(self, *_a: object, **_kw: object) -> None:
            pass

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *_exc_info: object) -> None:
            return None

        # Real-call shim; overridden per-test via monkeypatch on the module.
        def get_market_metadata(self, slug: str) -> MarketMetadata:
            raise NotImplementedError

    monkeypatch.setattr(tpf, "PolymarketClient", _FakeClient)
    return TestClient(app)


def _patch_market(
    monkeypatch: pytest.MonkeyPatch,
    *,
    end_date: str,
    prices: pd.DataFrame,
) -> None:
    def fake_metadata(self: object, slug: str) -> MarketMetadata:
        return MarketMetadata(
            slug=slug,
            question="?",
            yes_token_id="111",
            no_token_id="222",
            start_date="2026-01-01T00:00:00Z",
            end_date=end_date,
            closed=False,
            active=True,
        )

    def fake_history(
        _client: object,
        _slug: str,
        start: object = None,
        end: object = None,
    ) -> pd.DataFrame:
        return prices

    monkeypatch.setattr(tpf.PolymarketClient, "get_market_metadata", fake_metadata, raising=False)
    monkeypatch.setattr(tpf, "fetch_factor_history", fake_history)


def test_middle_prob_market_returns_well_formed_fan(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 50/50 market 30 days from resolution should produce ordered bands
    that fan out in the middle and re-converge near today's probability."""
    prices = _make_prices(p_today=0.50, seed=1)
    end_date = (datetime.now(tz=UTC) + timedelta(days=30)).isoformat()
    _patch_market(monkeypatch, end_date=end_date, prices=prices)

    r = app_client.get("/terminal/prob-fan/midprob-mkt?n_paths=300")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["slug"] == "midprob-mkt"
    assert 0.45 < body["today_p"] < 0.55
    assert body["vol_ann"] > 0
    assert body["days_to_resolution"] >= 1
    paths = body["paths"]
    assert set(paths.keys()) == {"p10", "p25", "p50", "p75", "p90"}
    # Each band has the same length, equal to days_to_resolution.
    h = body["days_to_resolution"]
    for k in ("p10", "p25", "p50", "p75", "p90"):
        assert len(paths[k]) == h
        # Probabilities are valid floats in [0, 1].
        for pt in paths[k]:
            assert 0.0 <= pt["p"] <= 1.0
            assert "t" in pt
    # Percentiles must be monotonic at every time-step.
    for i in range(h):
        assert paths["p10"][i]["p"] <= paths["p25"][i]["p"] <= paths["p50"][i]["p"]
        assert paths["p50"][i]["p"] <= paths["p75"][i]["p"] <= paths["p90"][i]["p"]
    # In a 50/50 market the median path stays in (0, 1) and starts near 0.5.
    medians = [pt["p"] for pt in paths["p50"]]
    assert 0.0 < float(np.mean(medians)) < 1.0
    # First-step median is anchored to today's probability — the bridge can't
    # have moved far in one day.
    assert abs(medians[0] - body["today_p"]) < 0.20


def test_low_prob_market_median_drifts_down(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 5% YES market should have its median path below today's probability
    at the far end of the horizon (the bridge collapses to NO with prob 0.95)."""
    prices = _make_prices(p_today=0.05, seed=2)
    end_date = (datetime.now(tz=UTC) + timedelta(days=20)).isoformat()
    _patch_market(monkeypatch, end_date=end_date, prices=prices)

    r = app_client.get("/terminal/prob-fan/lowprob-mkt?n_paths=500")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["today_p"] < 0.10
    median_far = body["paths"]["p50"][-1]["p"]
    # Median collapses *toward* 0 (NO is overwhelmingly likely), so the last
    # median should be no greater than today's probability.
    assert median_far <= body["today_p"] + 0.05
    # And p90 (upper band) should still be above the median — fan didn't
    # degenerate to a line.
    assert body["paths"]["p90"][-1]["p"] >= median_far


def test_near_resolution_market_has_short_horizon(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A market resolving in 2 days should produce a 2-step fan chart."""
    prices = _make_prices(p_today=0.40, seed=3)
    end_date = (datetime.now(tz=UTC) + timedelta(days=2)).isoformat()
    _patch_market(monkeypatch, end_date=end_date, prices=prices)

    r = app_client.get("/terminal/prob-fan/near-res-mkt")
    assert r.status_code == 200, r.text
    body = r.json()

    # Horizon is clamped at the short end to days_to_resolution.
    assert 1 <= body["days_to_resolution"] <= 3
    h = body["days_to_resolution"]
    for k in ("p10", "p25", "p50", "p75", "p90"):
        assert len(body["paths"][k]) == h
    # The endpoint mixture (p_today=0.40 → 60% land at NO, 40% at YES) makes
    # p10/p90 wider than for a balanced bridge, but they should still be
    # ordered and within [0, 1].
    spread_first = body["paths"]["p90"][0]["p"] - body["paths"]["p10"][0]["p"]
    assert 0.0 < spread_first < 1.0


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------


def test_metadata_failure_returns_404(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any failure resolving the market metadata is funneled to 404."""

    def failing_metadata(self: object, slug: str) -> object:
        raise RuntimeError("upstream gone")

    monkeypatch.setattr(
        tpf.PolymarketClient,
        "get_market_metadata",
        failing_metadata,
        raising=False,
    )
    r = app_client.get("/terminal/prob-fan/missing-mkt")
    assert r.status_code == 404
    assert "market not found" in r.json()["detail"]


def test_history_fetch_failure_returns_502(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata succeeds but price-history fetch fails → 502."""
    end_date = (datetime.now(tz=UTC) + timedelta(days=10)).isoformat()
    # Patch metadata to succeed.
    from pfm.sources.polymarket import MarketMetadata

    def fake_meta(self: object, slug: str) -> MarketMetadata:
        return MarketMetadata(
            slug=slug,
            question="?",
            yes_token_id="111",
            no_token_id="222",
            start_date="2026-01-01T00:00:00Z",
            end_date=end_date,
            closed=False,
            active=True,
        )

    monkeypatch.setattr(
        tpf.PolymarketClient,
        "get_market_metadata",
        fake_meta,
        raising=False,
    )

    def boom_history(*_a: object, **_kw: object) -> pd.DataFrame:
        raise RuntimeError("price api down")

    monkeypatch.setattr(tpf, "fetch_factor_history", boom_history)
    r = app_client.get("/terminal/prob-fan/some-mkt")
    assert r.status_code == 502


def test_n_paths_query_validator_bounds(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`n_paths` must be in [50, 10000]."""
    prices = _make_prices(p_today=0.50, seed=4)
    end_date = (datetime.now(tz=UTC) + timedelta(days=10)).isoformat()
    _patch_market(monkeypatch, end_date=end_date, prices=prices)
    # Below the minimum.
    assert app_client.get("/terminal/prob-fan/x?n_paths=10").status_code == 422
    # Above the maximum.
    assert app_client.get("/terminal/prob-fan/x?n_paths=20000").status_code == 422
    # Boundary minimum is accepted.
    assert app_client.get("/terminal/prob-fan/x?n_paths=50").status_code == 200


def test_response_schema_keys_match_contract(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Top-level response keys are exactly the four documented fields."""
    prices = _make_prices(p_today=0.30, seed=5)
    end_date = (datetime.now(tz=UTC) + timedelta(days=15)).isoformat()
    _patch_market(monkeypatch, end_date=end_date, prices=prices)
    body = app_client.get("/terminal/prob-fan/schema-mkt?n_paths=200").json()
    assert set(body.keys()) == {"slug", "today_p", "vol_ann", "days_to_resolution", "paths"}
    assert set(body["paths"].keys()) == {"p10", "p25", "p50", "p75", "p90"}
    for k in body["paths"]:
        # Each path entry is exactly {t, p}.
        for pt in body["paths"][k]:
            assert set(pt.keys()) == {"t", "p"}


def test_prob_fan_response_is_cached(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second hit to /prob-fan/{slug} must skip the upstream history fetch
    and the 1k-path Monte Carlo simulation.

    The MC sim is deterministic (fixed seed) and inputs only refresh on the
    hour scale; the cache turns a 200-450 ms cold call into ~1 ms warm.
    """
    from pfm.cache_utils import get_cache

    get_cache("terminal_prob_fan").clear()

    prices = _make_prices(p_today=0.40, seed=42)
    end_date = (datetime.now(tz=UTC) + timedelta(days=20)).isoformat()

    n_history_calls = {"n": 0}

    def fake_metadata(self: object, slug: str) -> MarketMetadata:
        return MarketMetadata(
            slug=slug,
            question="?",
            yes_token_id="111",
            no_token_id="222",
            start_date="2026-01-01T00:00:00Z",
            end_date=end_date,
            closed=False,
            active=True,
        )

    def counting_history(_client, _slug, start=None, end=None):
        n_history_calls["n"] += 1
        return prices

    monkeypatch.setattr(tpf.PolymarketClient, "get_market_metadata", fake_metadata, raising=False)
    monkeypatch.setattr(tpf, "fetch_factor_history", counting_history)

    r1 = app_client.get("/terminal/prob-fan/cached-mkt?n_paths=300")
    r2 = app_client.get("/terminal/prob-fan/cached-mkt?n_paths=300")
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    assert n_history_calls["n"] == 1, (
        f"fetch_factor_history should be called once across two warm hits, "
        f"got {n_history_calls['n']}"
    )
