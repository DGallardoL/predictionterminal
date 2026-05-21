"""Tests for the Earnings Whisper module.

The Polymarket Gamma fetch is short-circuited via the ``overrides`` map
on :func:`compute_whisper`. Endpoint integration tests patch
``earnings_whisper.fetch_gamma_market`` module-wide.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import earnings_whisper
from pfm.cache_utils import get_cache
from pfm.earnings_whisper import (
    BEAT_LADDERS,
    CONSENSUS_EPS,
    NEXT_EARNINGS,
    compute_whisper,
    router,
    whisper_dashboard,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    get_cache("earnings_whisper").clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _market(prob: float, vol: float = 1_000.0) -> dict[str, Any]:
    return {
        "bestBid": max(0.0, prob - 0.005),
        "bestAsk": min(1.0, prob + 0.005),
        "lastTradePrice": prob,
        "volume24hr": vol,
    }


# ---------------------------------------------------------------------------
# compute_whisper
# ---------------------------------------------------------------------------


def test_compute_whisper_nvda_strong_beat() -> None:
    """High beat probabilities should push whisper above consensus."""
    overrides: dict[str, dict[str, Any]] = {}
    # Strong ladder: 0.85 base beat, 0.55 by 5%, 0.30 by 10%, 0.10 by 20%.
    probs = {0.0: 0.85, 0.05: 0.55, 0.10: 0.30, 0.20: 0.10}
    for slug, t in BEAT_LADDERS["NVDA"]:
        overrides[slug] = _market(probs[t])

    out = compute_whisper("NVDA", date(2026, 5, 22), http=MagicMock(), overrides=overrides)

    assert out["ticker"] == "NVDA"
    assert out["consensus_eps"] == CONSENSUS_EPS["NVDA"]
    assert out["pm_beat_prob"] == pytest.approx(0.85, abs=1e-6)
    assert out["whisper_eps"] > out["consensus_eps"]
    assert out["edge_vs_consensus_pct"] > 0
    assert out["recommendation"] == "long_pre_print"
    assert out["n_ladder_rungs_used"] == 4


def test_compute_whisper_weak_beat_short_recommendation() -> None:
    """Low beat probabilities → expected beat is negative → short."""
    overrides: dict[str, dict[str, Any]] = {}
    # Very weak beat: only 5% chance of any beat at all → big mass below 0.
    probs = {0.0: 0.05, 0.05: 0.01, 0.10: 0.005, 0.20: 0.001}
    for slug, t in BEAT_LADDERS["NVDA"]:
        overrides[slug] = _market(probs[t])

    out = compute_whisper("NVDA", date(2026, 5, 22), http=MagicMock(), overrides=overrides)
    assert out["whisper_eps"] < out["consensus_eps"]
    assert out["edge_vs_consensus_pct"] < 0
    assert out["recommendation"] == "short_pre_print"


def test_compute_whisper_in_band_holds() -> None:
    """Edge inside ±2% → 'hold'."""
    overrides: dict[str, dict[str, Any]] = {}
    # Tightly balanced: 50% chance of beating, all higher rungs near 0.
    probs = {0.0: 0.50, 0.05: 0.06, 0.10: 0.02, 0.20: 0.01}
    for slug, t in BEAT_LADDERS["NVDA"]:
        overrides[slug] = _market(probs[t])
    out = compute_whisper("NVDA", date(2026, 5, 22), http=MagicMock(), overrides=overrides)
    assert -2.0 <= out["edge_vs_consensus_pct"] <= 2.0
    assert out["recommendation"] == "hold"


def test_compute_whisper_unknown_ticker_raises() -> None:
    with pytest.raises(KeyError):
        compute_whisper("ZZZZ", date(2026, 5, 22), http=MagicMock(), overrides={})


def test_compute_whisper_iv_implied_move_positive() -> None:
    out = compute_whisper("NVDA", date(2026, 5, 22), http=MagicMock(), overrides={})
    assert out["iv_implied_move_pct"] > 0.0


def test_edge_calculation_matches_formula() -> None:
    """edge_vs_consensus_pct must equal (whisper-consensus)/consensus * 100."""
    overrides: dict[str, dict[str, Any]] = {}
    for slug, _t in BEAT_LADDERS["TSLA"]:
        overrides[slug] = _market(0.40)
    out = compute_whisper("TSLA", date(2026, 5, 19), http=MagicMock(), overrides=overrides)
    expected = (out["whisper_eps"] - out["consensus_eps"]) / out["consensus_eps"] * 100.0
    assert out["edge_vs_consensus_pct"] == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_whisper_dashboard_returns_sorted_by_abs_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    today = date.today()
    # Force every NEXT_EARNINGS entry into the next 14 days so all show up.
    horizon = today + timedelta(days=10)
    fake_calendar = dict.fromkeys(NEXT_EARNINGS.keys(), horizon)
    monkeypatch.setattr(earnings_whisper, "NEXT_EARNINGS", fake_calendar)

    # Build per-ticker overrides with varying intensity so edges differ.
    overrides: dict[str, dict[str, Any]] = {}
    for i, (_tk, ladder) in enumerate(BEAT_LADDERS.items()):
        intensity = 0.05 + 0.10 * i
        for slug, t in ladder:
            overrides[slug] = _market(max(0.01, 0.85 - 1.5 * t - intensity))

    rows = whisper_dashboard(days=14, http=MagicMock(), overrides=overrides)
    assert rows, "expected at least one whisper row"
    edges = [abs(r["edge_vs_consensus_pct"]) for r in rows]
    assert edges == sorted(edges, reverse=True), f"dashboard not sorted by |edge| desc: {edges}"


def test_whisper_dashboard_horizon_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only tickers with earnings within ``days`` should appear."""
    today = date.today()
    fake_calendar = {
        "NVDA": today + timedelta(days=3),
        "TSLA": today + timedelta(days=20),  # outside default 14-day window
    }
    monkeypatch.setattr(earnings_whisper, "NEXT_EARNINGS", fake_calendar)
    rows = whisper_dashboard(days=14, http=MagicMock(), overrides={})
    tickers = {r["ticker"] for r in rows}
    assert "NVDA" in tickers
    assert "TSLA" not in tickers


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_get_whisper_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(earnings_whisper, "fetch_gamma_market", lambda *a, **k: _market(0.45))
    r = client.get("/alpha/earnings-whisper/NVDA", params={"date": "2026-05-22"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "NVDA"
    assert body["recommendation"] in {"long_pre_print", "short_pre_print", "hold"}


def test_get_whisper_endpoint_unknown_ticker(client: TestClient) -> None:
    r = client.get("/alpha/earnings-whisper/ZZZZ", params={"date": "2026-05-22"})
    assert r.status_code == 404


def test_get_whisper_endpoint_invalid_date(client: TestClient) -> None:
    r = client.get("/alpha/earnings-whisper/NVDA", params={"date": "not-a-date"})
    assert r.status_code == 400


def test_get_whisper_dashboard_endpoint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = date.today()
    monkeypatch.setattr(
        earnings_whisper,
        "NEXT_EARNINGS",
        {"NVDA": today + timedelta(days=2), "TSLA": today + timedelta(days=5)},
    )
    monkeypatch.setattr(earnings_whisper, "fetch_gamma_market", lambda *a, **k: _market(0.50))
    r = client.get("/alpha/earnings-whisper-dashboard", params={"days": 14})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["horizon_days"] == 14
    assert body["n"] == len(body["rows"])


def test_whisper_dashboard_parallelizes_per_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-ticker compute fans out — wall time ≪ Σ per-ticker latency.

    Regression: probe found ``/alpha/earnings-whisper-dashboard`` at
    13.55 s warm because the loop walked tickers serially while each
    ticker required N synchronous ``fetch_gamma_market`` HTTP calls.
    We patch ``fetch_gamma_market`` with a 0.10 s sleeper and check
    that 6 tickers complete in well under the serial bound.
    """
    import time

    today = date.today()
    fake_calendar = {
        tk: today + timedelta(days=3) for tk in ("NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "META")
    }
    monkeypatch.setattr(earnings_whisper, "NEXT_EARNINGS", fake_calendar)

    def _slow_fetch(*_a: Any, **_k: Any) -> dict[str, Any]:
        time.sleep(0.10)
        return _market(0.50)

    monkeypatch.setattr(earnings_whisper, "fetch_gamma_market", _slow_fetch)

    # 6 tickers × ~3 rungs × 0.10 s sleep = ~1.8 s serial. With pool of
    # 8 workers the wall clock should drop to roughly one ticker's
    # latency (~0.3-0.5 s). Use a generous bound to avoid CI flake.
    started = time.perf_counter()
    rows = whisper_dashboard(days=14, overrides=None)
    elapsed = time.perf_counter() - started

    assert rows, "expected at least one row"
    assert elapsed < 1.2, (
        f"dashboard wall time {elapsed:.2f}s suggests per-ticker compute "
        f"is still serialised (serial baseline ≈1.8s)"
    )
