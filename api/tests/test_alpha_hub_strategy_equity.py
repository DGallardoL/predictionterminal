"""Tests for the equity-curve preview embedded in ``GET /alpha-hub/strategy/{pair_id}``.

The handler embeds a 30-point ``equity_curve`` series so the frontend can
render a sparkline on first paint without firing a second request to
``/terminal/backtest/{slug}``. The curve source today is a synthetic
linear ramp derived from ``oos_sharpe``; these tests assert the SHAPE
and contract, not specific magnitudes (since the ramp is deliberately
cheap and informational).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.alpha_hub_router import _load_strategies
from pfm.alpha_hub_router import router as alpha_hub_router
from pfm.cache_utils import reset_caches


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()
    yield
    reset_caches()


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(alpha_hub_router)
    return TestClient(app)


def _first_pair_id() -> str:
    """Pull a known-good pair_id from the live catalog (not hardcoded)."""
    strategies = _load_strategies()
    assert strategies, "alpha_strategies.json is empty — cannot pick a pair_id"
    return str(strategies[0]["pair_id"])


def test_strategy_detail_embeds_equity_curve_field() -> None:
    client = _make_app()
    pair_id = _first_pair_id()
    r = client.get(f"/alpha-hub/strategy/{pair_id}")
    assert r.status_code == 200
    body = r.json()
    assert "equity_curve" in body, "equity_curve must be embedded for first-paint UX"


def test_strategy_detail_equity_curve_is_non_empty_dict_list() -> None:
    client = _make_app()
    pair_id = _first_pair_id()
    body = client.get(f"/alpha-hub/strategy/{pair_id}").json()
    curve = body["equity_curve"]
    assert isinstance(curve, list)
    assert len(curve) > 0
    for point in curve:
        assert isinstance(point, dict)
        assert "date" in point and "equity" in point


def test_strategy_detail_equity_curve_has_at_least_20_points() -> None:
    client = _make_app()
    pair_id = _first_pair_id()
    body = client.get(f"/alpha-hub/strategy/{pair_id}").json()
    curve = body["equity_curve"]
    assert len(curve) >= 20, f"expected >=20 points, got {len(curve)}"


def test_strategy_detail_equity_curve_starts_at_one() -> None:
    client = _make_app()
    pair_id = _first_pair_id()
    body = client.get(f"/alpha-hub/strategy/{pair_id}").json()
    curve = body["equity_curve"]
    start_equity = float(curve[0]["equity"])
    assert start_equity == pytest.approx(1.0, abs=1e-6), (
        f"equity curve must start at 1.0, got {start_equity}"
    )


def test_strategy_detail_metadata_preserved_alongside_equity_curve() -> None:
    """Embedding equity_curve must not strip the existing detail fields."""
    client = _make_app()
    pair_id = _first_pair_id()
    body = client.get(f"/alpha-hub/strategy/{pair_id}").json()
    # The original detail response had these keys; they must survive.
    assert body["pair_id"] == pair_id
    assert "tier" in body
    assert "oos_sharpe" in body
    # A full record has either of these descriptive fields.
    assert "rationale" in body or "deploy_signal_logic" in body


def test_strategy_detail_unknown_pair_id_returns_404() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/strategy/__not_a_real_pair_id__")
    assert r.status_code == 404


def test_strategy_detail_includes_spread_series() -> None:
    """Spread series must be embedded for the fullscreen detail view."""
    client = _make_app()
    pair_id = _first_pair_id()
    body = client.get(f"/alpha-hub/strategy/{pair_id}").json()
    assert "spread_series" in body
    series = body["spread_series"]
    assert isinstance(series, list)
    assert len(series) >= 30, f"expected >=30 points, got {len(series)}"
    required_keys = {"date", "z_score", "p_a", "p_b", "spread"}
    for point in series:
        assert isinstance(point, dict)
        assert required_keys.issubset(point.keys()), f"missing keys: {required_keys - point.keys()}"
    # Dates must be sorted ascending.
    dates = [p["date"] for p in series]
    assert dates == sorted(dates), "spread_series dates must be ascending"


def test_strategy_detail_includes_rule_and_risk() -> None:
    """Detail must surface the rule + risk sub-dicts for the fullscreen view."""
    client = _make_app()
    pair_id = _first_pair_id()
    body = client.get(f"/alpha-hub/strategy/{pair_id}").json()
    assert "rule" in body, "rule sub-dict key must be present"
    assert "risk" in body, "risk sub-dict key must be present"
    # risk is always a dict (sparse but never absent).
    assert isinstance(body["risk"], dict)
    assert {"grade", "max_dd", "best_conditions", "worst_conditions"}.issubset(body["risk"].keys())


def test_strategy_detail_recent_signal_optional_404_safe() -> None:
    """Unknown pair_id is 404, but a known pair without a live signal returns 200.

    For any pair_id in the catalog that's not present in ``live_signals.json``,
    the response should be 200 and ``recent_signal`` should be ``None`` — not
    a 500. We assert the contract for an arbitrary catalog pair: ``recent_signal``
    is either ``None`` or a dict — and the request never 500s.
    """
    client = _make_app()
    # Iterate from the tail of the catalog to find a pair_id more likely
    # to be absent from live_signals.json (which today only covers ~88).
    from pfm.alpha_hub_router import _load_strategies as _ls

    strategies = _ls()
    assert strategies
    # Try last entry first; fall back to first if needed.
    for s in reversed(strategies):
        pid = str(s["pair_id"])
        r = client.get(f"/alpha-hub/strategy/{pid}")
        assert r.status_code == 200, f"unexpected {r.status_code} for {pid}"
        body = r.json()
        assert "recent_signal" in body
        rs = body["recent_signal"]
        assert rs is None or isinstance(rs, dict), (
            f"recent_signal must be None or dict, got {type(rs).__name__}"
        )
        # We only need one positive check that the contract holds.
        if rs is None:
            return
    # If every pair had a signal, the contract still holds (dicts only).
    # The test passes as long as we didn't 500.
