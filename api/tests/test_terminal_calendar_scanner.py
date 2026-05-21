"""Tests for ``pfm.terminal_calendar_scanner`` — calendar-arb scanner.

The scanner consumes the curated cluster set produced (in parallel) by
``pfm.terminal_calendar_curated.get_clusters()``. These tests stub that
producer with in-memory fixtures so the suite is hermetic and does not
require live Polymarket access.
"""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_calendar_scanner
from pfm.terminal_calendar_scanner import (
    ACTIONABLE_THRESHOLD,
    _build_signal,
    _classify_conviction,
    _expected_ev_pct,
    _implied_lambda,
    router,
)


def _install_curated_stub(clusters: list[dict[str, object]]) -> tuple[object, bool]:
    """Inject a fake ``pfm.terminal_calendar_curated`` module.

    Returns ``(prev_module_or_None, prev_attr_existed)`` so the original
    state can be restored. We must shim BOTH ``sys.modules`` *and* the
    attribute on the ``pfm`` package — when the real curated module has
    already been imported elsewhere (e.g. via ``pfm.main``), the
    statement ``from pfm import terminal_calendar_curated`` resolves the
    attribute on the package namespace rather than re-reading
    ``sys.modules``.
    """
    import pfm

    prev_module = sys.modules.get("pfm.terminal_calendar_curated")
    prev_attr_existed = hasattr(pfm, "terminal_calendar_curated")

    mod = types.ModuleType("pfm.terminal_calendar_curated")
    mod.get_clusters = lambda: clusters  # type: ignore[attr-defined]
    sys.modules["pfm.terminal_calendar_curated"] = mod
    pfm.terminal_calendar_curated = mod  # type: ignore[attr-defined]
    return prev_module, prev_attr_existed


def _uninstall_curated_stub(prev_module: object, prev_attr_existed: bool) -> None:
    import pfm

    if prev_module is not None:
        sys.modules["pfm.terminal_calendar_curated"] = prev_module  # type: ignore[assignment]
        if prev_attr_existed:
            pfm.terminal_calendar_curated = prev_module  # type: ignore[attr-defined]
        elif hasattr(pfm, "terminal_calendar_curated"):
            delattr(pfm, "terminal_calendar_curated")
    else:
        sys.modules.pop("pfm.terminal_calendar_curated", None)
        if not prev_attr_existed and hasattr(pfm, "terminal_calendar_curated"):
            delattr(pfm, "terminal_calendar_curated")


@pytest.fixture
def actionable_clusters() -> list[dict[str, object]]:
    """Two clusters: one well above the 0.75 threshold, one benign."""
    return [
        {
            # Strong STEEPEN signal: λ_far ≫ λ_near ⇒ log_ratio > 0.75.
            # near: p=0.02, T=60   ⇒ λ ≈ 0.000337
            # far:  p=0.50, T=240  ⇒ λ ≈ 0.002888
            # ln(λ_far / λ_near)   ≈ 2.15
            "cluster_id": "trump_out_president",
            "title": "Trump out as president",
            "legs": [
                {
                    "slug": "trump-out-by-jun30",
                    "name": "Trump out by Jun 30",
                    "current_p": 0.02,
                    "dtr": 60,
                },
                {
                    "slug": "trump-out-before-2027",
                    "name": "Trump out before 2027",
                    "current_p": 0.50,
                    "dtr": 240,
                },
            ],
        },
        {
            # Benign cluster: |log λ-ratio| ≪ 0.75.
            # near: p=0.10, T=30   ⇒ λ ≈ 0.003512
            # far:  p=0.18, T=60   ⇒ λ ≈ 0.003307
            # ln-ratio ≈ -0.06.
            "cluster_id": "benign_cluster",
            "title": "Benign no-trade cluster",
            "legs": [
                {
                    "slug": "benign-near",
                    "name": "Benign near",
                    "current_p": 0.10,
                    "dtr": 30,
                },
                {
                    "slug": "benign-far",
                    "name": "Benign far",
                    "current_p": 0.18,
                    "dtr": 60,
                },
            ],
        },
    ]


@pytest.fixture
def client(
    actionable_clusters: list[dict[str, object]],
) -> Iterator[TestClient]:
    prev_module, prev_attr_existed = _install_curated_stub(actionable_clusters)
    app = FastAPI()
    app.include_router(router)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        _uninstall_curated_stub(prev_module, prev_attr_existed)


# --- tests ------------------------------------------------------------------


def test_active_endpoint_emits_only_threshold_crossing_signals(
    client: TestClient,
) -> None:
    """The /active endpoint returns the actionable cluster but not the benign one."""
    r = client.get("/terminal/calendar-scanner/active")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 1, f"expected 1 actionable signal, got {body!r}"

    signal = body[0]
    assert signal["cluster_id"] == "trump_out_president"
    assert abs(signal["log_lambda_ratio"]) >= ACTIONABLE_THRESHOLD
    # Strong λ_far / λ_near ⇒ STEEPEN_CURVE (long near, short far).
    assert signal["trade_type"] == "STEEPEN_CURVE"
    assert signal["long_leg"]["slug"] == "trump-out-by-jun30"
    assert signal["short_leg"]["slug"] == "trump-out-before-2027"
    assert signal["entry_signal"] == ("Long trump-out-by-jun30 + Short trump-out-before-2027")
    assert "reverts below 0.30" in signal["exit_rule"]
    assert signal["conviction"] == "high"
    assert signal["hold_window_days"] == 110
    # EV = 2.15 × 4% − 3.6% ≈ 4–5 %.
    assert signal["expected_ev_pct"] > 0.0


def test_historical_endpoint_returns_404_for_unknown_cluster(
    client: TestClient,
) -> None:
    """Unknown cluster_id yields a 404 with a helpful detail."""
    r = client.get(
        "/terminal/calendar-scanner/historical",
        params={"cluster_id": "no-such-cluster"},
    )
    assert r.status_code == 404
    assert "unknown cluster_id" in r.json()["detail"]


def test_historical_endpoint_succeeds_with_empty_history_when_data_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When live price history can't be fetched the endpoint still returns 200.

    We force the fetcher to return an empty triple — the scanner must
    degrade gracefully to a zero-trade backtest rather than 500ing.
    """
    monkeypatch.setattr(
        terminal_calendar_scanner,
        "_fetch_pair_history",
        lambda *_args, **_kwargs: ([], [], []),
    )
    r = client.get(
        "/terminal/calendar-scanner/historical",
        params={"cluster_id": "trump_out_president"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cluster_id"] == "trump_out_president"
    assert body["n_days"] == 0
    assert body["n_trades"] == 0
    assert body["cum_pnl"] == 0.0
    assert body["points"] == []


def test_signal_helpers_classify_and_price_correctly() -> None:
    """Pure-function helpers behave as documented."""
    # implied_lambda: monotone in p, zero on degenerate inputs.
    assert _implied_lambda(0.0, 30) == 0.0
    assert _implied_lambda(0.5, 0) == 0.0
    assert _implied_lambda(0.1, 30) < _implied_lambda(0.5, 30)

    # Conviction tiers.
    assert _classify_conviction(1.5) == "high"
    assert _classify_conviction(0.85) == "medium"
    assert _classify_conviction(0.40) == "low"

    # EV: gross = abs_ratio × 0.04, net = gross − 0.036, in percent.
    expected_ev = (1.0 * 0.04 - 0.036) * 100.0
    assert math.isclose(_expected_ev_pct(1.0), round(expected_ev, 2))

    # _build_signal: a strong negative log-ratio → FLATTEN_CURVE.
    near = {"slug": "n", "name": "near", "current_p": 0.50, "dtr": 30}
    far = {"slug": "f", "name": "far", "current_p": 0.02, "dtr": 240}
    sig = _build_signal("c1", "title", near, far)
    assert sig is not None
    assert sig.trade_type == "FLATTEN_CURVE"
    assert sig.long_leg.slug == "f"  # long the cold (far) leg
    assert sig.short_leg.slug == "n"
    assert sig.log_lambda_ratio < -ACTIONABLE_THRESHOLD

    # Below-threshold pair → no signal.
    weak_near = {"slug": "n", "name": "near", "current_p": 0.10, "dtr": 30}
    weak_far = {"slug": "f", "name": "far", "current_p": 0.18, "dtr": 60}
    assert _build_signal("c2", "title", weak_near, weak_far) is None
