"""Tests for ``GET /alpha-hub/strategy/{pair_id}/live-signal``.

On-demand single-pair signal recompute that wraps
``pfm.live_signals_job._compute_signal_for_alpha`` and adds Kelly-scaled
position sizing on top of the existing batch output.

Tests cover:
  * happy path (live compute → 200 with all fields, sized > 0)
  * unknown ``pair_id`` → 404
  * non-trading actions zero out ``recommended_size_usd``
  * ``kelly_cap`` is enforced
  * cached-batch path returned when ``live_signals.json`` is fresh
  * stale-fallback path when live compute fails but cache exists
  * 503 when ``force_refresh=True`` and live compute fails
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.alpha_hub_router as ah
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


def _first_strategy() -> dict[str, Any]:
    strategies = _load_strategies()
    assert strategies, "alpha_strategies.json must be non-empty"
    return strategies[0]


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _fake_signal(
    pair_id: str,
    *,
    action: str = "LONG_SPREAD",
    z: float = 2.3,
    prev_z: float = 1.4,
    sigma: float = 0.013,
    n_obs: int = 87,
    as_of: str | None = None,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "a_id": "leg_a",
        "b_id": "leg_b",
        "as_of": as_of or _now_iso(),
        "n_obs": n_obs,
        "beta_hedge": 0.012,
        "current_spread": 0.025,
        "current_z": z,
        "previous_z": prev_z,
        "current_a_price": 0.42,
        "current_b_price": 0.38,
        "action": action,
        "reason": f"z={z:+.2f} ≥ entry → {action}",
        "mu_window": 0.001,
        "sigma_window": sigma,
        "decay_status": "ACTIVE",
    }


# --- helper-unit tests ------------------------------------------------------


def test_kelly_zero_when_action_is_hold() -> None:
    assert ah._kelly_for_action(action="HOLD", oos_sharpe=3.0, kelly_cap=0.25) == 0.0


def test_kelly_zero_when_action_is_flat() -> None:
    assert ah._kelly_for_action(action="FLAT", oos_sharpe=5.0, kelly_cap=0.25) == 0.0


def test_kelly_proxy_value_for_long_spread() -> None:
    k = ah._kelly_for_action(action="LONG_SPREAD", oos_sharpe=4.0, kelly_cap=0.25)
    expected = min(4.0 / math.sqrt(252.0), 0.25)
    assert k == pytest.approx(expected, abs=1e-9)


def test_kelly_cap_clamps_high_sharpe() -> None:
    k = ah._kelly_for_action(action="OPEN_LONG", oos_sharpe=99.0, kelly_cap=0.10)
    assert k == 0.10


def test_kelly_zero_when_sharpe_missing() -> None:
    assert ah._kelly_for_action(action="LONG_SPREAD", oos_sharpe=None, kelly_cap=0.25) == 0.0


def test_edge_bps_basic() -> None:
    # |2.0| * 0.01 * 10_000 = 200 bps
    assert ah._edge_bps(2.0, 0.01) == pytest.approx(200.0)


def test_edge_bps_none_when_missing_inputs() -> None:
    assert ah._edge_bps(None, 0.01) is None
    assert ah._edge_bps(1.5, None) is None


# --- happy path / live compute ----------------------------------------------


def test_live_signal_happy_path_returns_full_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compute is mocked; verify 200 + all contract fields + sized > 0."""
    src = _first_strategy()
    pair_id = str(src["pair_id"])
    fake = _fake_signal(pair_id, action="LONG_SPREAD", z=2.3)

    async def _stub(s: dict[str, Any], *, as_of_iso: str) -> dict[str, Any]:
        # Echo the as_of from the caller so we can assert the field passthrough.
        fake["as_of"] = as_of_iso
        return fake

    monkeypatch.setattr(ah, "_compute_live_signal_now", _stub)

    client = _make_app()
    r = client.get(
        f"/alpha-hub/strategy/{pair_id}/live-signal",
        params={"bankroll_usd": 10000.0, "kelly_cap": 0.25},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # Required fields from the contract are all present.
    required = {
        "pair_id",
        "as_of",
        "computed_at",
        "data_source",
        "n_obs",
        "beta_hedge",
        "current_a_price",
        "current_b_price",
        "current_spread",
        "previous_z",
        "current_z",
        "mu_window",
        "sigma_window",
        "action",
        "reason",
        "decay_status",
        "rule_window",
        "rule_entry_z",
        "rule_exit_z",
        "rule_stop_z",
        "tier",
        "kelly_fraction",
        "edge_bps",
        "recommended_size_usd",
        "bankroll_usd",
        "warnings",
    }
    assert required.issubset(body.keys()), required - body.keys()

    assert body["pair_id"] == pair_id
    assert body["data_source"] == "live"
    assert body["action"] == "LONG_SPREAD"
    assert body["current_z"] == pytest.approx(2.3)
    assert body["previous_z"] == pytest.approx(1.4)
    assert body["bankroll_usd"] == 10000.0
    # Sizing must be strictly positive for a trading action with non-zero
    # Sharpe and a non-zero suggested_allocation.
    assert body["kelly_fraction"] > 0
    assert body["recommended_size_usd"] > 0
    # edge_bps = |z| * sigma * 10_000 = 2.3 * 0.013 * 10000 ≈ 299
    assert body["edge_bps"] == pytest.approx(2.3 * 0.013 * 10_000, rel=1e-6)
    # Warnings array exists and is empty on the live happy-path.
    assert body["warnings"] == []


def test_live_signal_unknown_pair_id_returns_404() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/strategy/__not_a_real_pair_id__/live-signal")
    assert r.status_code == 404


def test_live_signal_hold_action_zeros_recommended_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = _first_strategy()
    pair_id = str(src["pair_id"])
    fake = _fake_signal(pair_id, action="HOLD", z=0.4)

    async def _stub(s: dict[str, Any], *, as_of_iso: str) -> dict[str, Any]:
        fake["as_of"] = as_of_iso
        return fake

    monkeypatch.setattr(ah, "_compute_live_signal_now", _stub)

    client = _make_app()
    r = client.get(f"/alpha-hub/strategy/{pair_id}/live-signal")
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "HOLD"
    assert body["kelly_fraction"] == 0.0
    assert body["recommended_size_usd"] == 0.0


def test_live_signal_kelly_cap_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High Sharpe + tight cap → kelly_fraction == cap (not raw proxy)."""
    # Pick a strategy with high oos_sharpe so the uncapped proxy
    # 'sharpe / sqrt(252)' beats a 0.05 cap. The catalog has alphas with
    # Sharpe > 1.0; 1.0 / sqrt(252) ≈ 0.063 > 0.05.
    strategies = _load_strategies()
    src = next(
        (s for s in strategies if (s.get("oos_sharpe") or 0) >= 1.0),
        strategies[0],
    )
    pair_id = str(src["pair_id"])
    fake = _fake_signal(pair_id, action="LONG_SPREAD", z=2.5)

    async def _stub(s: dict[str, Any], *, as_of_iso: str) -> dict[str, Any]:
        fake["as_of"] = as_of_iso
        return fake

    monkeypatch.setattr(ah, "_compute_live_signal_now", _stub)

    client = _make_app()
    r = client.get(
        f"/alpha-hub/strategy/{pair_id}/live-signal",
        params={"kelly_cap": 0.05},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kelly_fraction"] <= 0.05 + 1e-9


# --- cached-batch path ------------------------------------------------------


def test_live_signal_returns_cached_batch_when_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh ``live_signals.json`` entry must be returned without recompute."""
    src = _first_strategy()
    pair_id = str(src["pair_id"])

    # Build a synthetic fresh cache (~5 minutes old).
    recent_iso = (datetime.now(tz=UTC) - timedelta(minutes=5)).isoformat()
    cached = _fake_signal(pair_id, action="LONG_SPREAD", z=1.95, as_of=recent_iso, n_obs=42)
    monkeypatch.setattr(ah, "_cached_live_signals", lambda: {pair_id: cached})

    # Sentinel: if compute is invoked we explode the test.
    async def _no_compute(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise AssertionError("live compute must NOT be called when cached_batch is fresh")

    monkeypatch.setattr(ah, "_compute_live_signal_now", _no_compute)

    client = _make_app()
    r = client.get(f"/alpha-hub/strategy/{pair_id}/live-signal")
    assert r.status_code == 200
    body = r.json()
    assert body["data_source"] == "cached_batch"
    assert body["n_obs"] == 42
    assert body["current_z"] == pytest.approx(1.95)


def test_live_signal_force_refresh_bypasses_fresh_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``force_refresh=True`` must recompute even if cache is fresh."""
    src = _first_strategy()
    pair_id = str(src["pair_id"])

    recent_iso = (datetime.now(tz=UTC) - timedelta(minutes=2)).isoformat()
    cached = _fake_signal(pair_id, action="SHORT_SPREAD", z=-2.6, as_of=recent_iso, n_obs=10)
    monkeypatch.setattr(ah, "_cached_live_signals", lambda: {pair_id: cached})

    fresh = _fake_signal(pair_id, action="LONG_SPREAD", z=2.1, n_obs=99)

    async def _stub(s: dict[str, Any], *, as_of_iso: str) -> dict[str, Any]:
        fresh["as_of"] = as_of_iso
        return fresh

    monkeypatch.setattr(ah, "_compute_live_signal_now", _stub)

    client = _make_app()
    r = client.get(
        f"/alpha-hub/strategy/{pair_id}/live-signal",
        params={"force_refresh": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data_source"] == "live"
    assert body["n_obs"] == 99
    assert body["current_z"] == pytest.approx(2.1)


# --- stale-fallback path ----------------------------------------------------


def test_live_signal_stale_fallback_when_compute_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compute fails + cache exists (stale) → return cached with warning."""
    src = _first_strategy()
    pair_id = str(src["pair_id"])

    stale_iso = (datetime.now(tz=UTC) - timedelta(hours=6)).isoformat()
    cached = _fake_signal(pair_id, action="LONG_SPREAD", z=2.1, as_of=stale_iso, n_obs=55)
    monkeypatch.setattr(ah, "_cached_live_signals", lambda: {pair_id: cached})

    async def _boom(s: dict[str, Any], *, as_of_iso: str) -> dict[str, Any]:
        raise RuntimeError("simulated upstream Polymarket failure")

    monkeypatch.setattr(ah, "_compute_live_signal_now", _boom)

    client = _make_app()
    r = client.get(f"/alpha-hub/strategy/{pair_id}/live-signal")
    assert r.status_code == 200
    body = r.json()
    assert body["data_source"] == "stale_fallback"
    assert body["n_obs"] == 55
    assert body["warnings"], "stale_fallback must include a warning"
    assert any("recompute failed" in w for w in body["warnings"])


def test_live_signal_force_refresh_503_when_live_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``force_refresh=True`` + compute fails → 503 (never falls back)."""
    src = _first_strategy()
    pair_id = str(src["pair_id"])

    # Even if a cached entry exists, force_refresh must not fall back.
    monkeypatch.setattr(
        ah,
        "_cached_live_signals",
        lambda: {pair_id: _fake_signal(pair_id, action="LONG_SPREAD")},
    )

    async def _boom(s: dict[str, Any], *, as_of_iso: str) -> dict[str, Any]:
        raise RuntimeError("upstream-down")

    monkeypatch.setattr(ah, "_compute_live_signal_now", _boom)

    client = _make_app()
    r = client.get(
        f"/alpha-hub/strategy/{pair_id}/live-signal",
        params={"force_refresh": "true"},
    )
    assert r.status_code == 503
    body = r.json()
    assert "live recompute failed" in body.get("detail", "")


def test_live_signal_503_when_compute_fails_and_no_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No cache + compute fails (no force) → 503 too (nothing to return)."""
    src = _first_strategy()
    pair_id = str(src["pair_id"])

    monkeypatch.setattr(ah, "_cached_live_signals", lambda: {})

    async def _boom(s: dict[str, Any], *, as_of_iso: str) -> dict[str, Any]:
        raise RuntimeError("upstream-down")

    monkeypatch.setattr(ah, "_compute_live_signal_now", _boom)

    client = _make_app()
    r = client.get(f"/alpha-hub/strategy/{pair_id}/live-signal")
    assert r.status_code == 503
