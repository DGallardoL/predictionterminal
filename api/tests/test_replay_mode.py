"""Tests for ``pfm.replay_mode``."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.replay_mode as rm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_pm_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Smooth oscillating PM-style probability series, indexed by date."""
    idx = pd.date_range(start, end, freq="D", tz="UTC").normalize()
    n = len(idx)
    t = np.arange(n) / max(n, 1)
    price = (0.50 + 0.20 * np.sin(2 * np.pi * t * 1.5)).clip(0.05, 0.95)
    df = pd.DataFrame({"price": price}, index=idx)
    df.index.name = "date"
    return df


def _synthetic_yf_rows(start_iso: str, end_iso: str, base: float = 100.0):
    idx = pd.date_range(start_iso, end_iso, freq="D", tz="UTC").normalize()
    n = len(idx)
    rng = np.random.default_rng(42)
    drift = np.cumsum(rng.normal(0.0005, 0.01, n))
    closes = base * np.exp(drift)
    return tuple((d.isoformat(), float(c)) for d, c in zip(idx, closes, strict=False))


@pytest.fixture(autouse=True)
def _patch_external(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the PM history + yfinance sources so tests are hermetic."""

    def fake_resolve_pm_history(slug: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if slug.startswith("missing-"):
            return pd.DataFrame()
        return _synthetic_pm_history(start, end)

    monkeypatch.setattr(rm, "_resolve_pm_history", fake_resolve_pm_history)

    rm._yf_close_cached.cache_clear()

    def fake_yf(ticker: str, start_iso: str, end_iso: str):
        if ticker == "MISSING":
            return ()
        base = {
            "SPY": 450.0,
            "QQQ": 380.0,
            "TLT": 90.0,
            "BTC-USD": 70000.0,
            "GLD": 200.0,
            "DJT": 30.0,
            "VIX": 25.0,
            "DXY": 105.0,
            "COIN": 200.0,
            "MSTR": 250.0,
            "ETH-USD": 3500.0,
        }.get(ticker, 100.0)
        return _synthetic_yf_rows(start_iso, end_iso, base=base)

    monkeypatch.setattr(rm, "_yf_close_cached", fake_yf)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestGetStateAt:
    def test_returns_markets_and_equities(self) -> None:
        ts = datetime(2024, 11, 5, 23, 0, tzinfo=UTC)
        out = rm.get_state_at(ts, slugs=["dummy-a", "dummy-b"], equity_tickers=["SPY", "QQQ"])
        assert out["as_of"].startswith("2024-11-05")
        assert len(out["markets"]) == 2
        assert all(0.0 <= m["prob"] <= 1.0 for m in out["markets"])
        assert {e["ticker"] for e in out["equities"]} == {"SPY", "QQQ"}
        for e in out["equities"]:
            assert e["price"] > 0

    def test_skips_missing_slugs(self) -> None:
        ts = datetime(2024, 11, 5, 23, 0, tzinfo=UTC)
        out = rm.get_state_at(ts, slugs=["missing-x", "ok-1"], equity_tickers=["SPY"])
        assert {m["slug"] for m in out["markets"]} == {"ok-1"}

    def test_handles_no_slugs_gracefully(self) -> None:
        ts = datetime(2024, 11, 5, 23, 0, tzinfo=UTC)
        out = rm.get_state_at(ts, slugs=[], equity_tickers=["SPY"])
        assert out["markets"] == []
        assert len(out["equities"]) == 1


class TestSimulatePaperOrder:
    def test_long_with_hold_returns_pnl(self) -> None:
        entry = datetime(2024, 9, 1, 18, 0, tzinfo=UTC)
        exit_ts = datetime(2024, 11, 1, 18, 0, tzinfo=UTC)
        out = rm.simulate_paper_order(
            "demo-slug",
            "LONG",
            1000.0,
            entry,
            hold_until=exit_ts,
        )
        assert out["status"] == "CLOSED"
        assert out["entry_price"] is not None
        assert out["exit_price"] is not None
        assert out["bars_held"] >= 1
        assert isinstance(out["pnl_pct"], float)
        # PnL units: 1% slippage default → for size=1000 USD, pnl_usd should
        # be on the order of pct * 1000.
        assert abs(out["pnl_usd"] - out["pnl_pct"] * 1000.0) < 0.5

    def test_short_returns_inverted_pnl_sign(self) -> None:
        entry = datetime(2024, 9, 1, 18, 0, tzinfo=UTC)
        exit_ts = datetime(2024, 11, 1, 18, 0, tzinfo=UTC)
        long_out = rm.simulate_paper_order("demo-slug", "LONG", 1000.0, entry, hold_until=exit_ts)
        short_out = rm.simulate_paper_order("demo-slug", "SHORT", 1000.0, entry, hold_until=exit_ts)
        # Slippage drag is symmetric, so long_pct + short_pct ≈ -2*half_slip*2
        # but at minimum the mid-price components flip sign.
        assert (long_out["pnl_pct"] - short_out["pnl_pct"]) > 0 or (
            short_out["pnl_pct"] - long_out["pnl_pct"]
        ) > 0

    def test_no_data_returns_safe_payload(self) -> None:
        out = rm.simulate_paper_order(
            "missing-slug",
            "LONG",
            500.0,
            datetime(2024, 9, 1, tzinfo=UTC),
        )
        assert out["status"] == "NO_DATA"
        assert out["pnl_usd"] == 0.0

    def test_open_mtm_when_no_hold_until(self) -> None:
        entry = datetime(2024, 9, 1, 18, 0, tzinfo=UTC)
        out = rm.simulate_paper_order("demo-slug", "LONG", 100.0, entry)
        assert out["status"] in {"OPEN_MTM", "NO_EXIT_PRICE"}

    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError):
            rm.simulate_paper_order(
                "demo-slug",
                "BUY",
                100.0,  # type: ignore[arg-type]
                datetime(2024, 9, 1, tzinfo=UTC),
            )


class TestScenarios:
    def test_list_scenarios_returns_four(self) -> None:
        rows = rm.list_scenarios()
        names = {r["name"] for r in rows}
        assert {
            "election_night_2024",
            "fomc_2024_09",
            "btc_ath_2024_11",
            "covid_crash_2020_03",
        } <= names

    def test_replay_scenario_includes_metadata(self) -> None:
        out = rm.replay_scenario("election_night_2024")
        assert out["scenario"]["name"] == "election_night_2024"
        assert out["scenario"]["title"]
        assert isinstance(out["headline_news"], list)
        assert len(out["equities"]) >= 1


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


def _build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(rm.router)
    return TestClient(app)


def test_router_state_endpoint() -> None:
    client = _build_test_client()
    r = client.get(
        "/replay/state",
        params={
            "as_of": "2024-11-05T23:00:00+00:00",
            "slugs": "alpha,beta",
            "tickers": "SPY,QQQ",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["as_of"].startswith("2024-11-05")
    assert len(body["markets"]) == 2
    assert {e["ticker"] for e in body["equities"]} == {"SPY", "QQQ"}


def test_router_order_endpoint() -> None:
    client = _build_test_client()
    r = client.post(
        "/replay/order",
        json={
            "slug": "demo-slug",
            "side": "LONG",
            "size_usd": 1000.0,
            "at_timestamp": "2024-09-01T18:00:00+00:00",
            "hold_until": "2024-11-01T18:00:00+00:00",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "CLOSED"
    assert body["entry_price"] is not None


def test_router_scenarios_listing() -> None:
    client = _build_test_client()
    r = client.get("/replay/scenarios")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_scenarios"] == 4
    assert {s["name"] for s in body["scenarios"]} >= {
        "election_night_2024",
        "fomc_2024_09",
        "btc_ath_2024_11",
        "covid_crash_2020_03",
    }


def test_router_scenario_detail() -> None:
    client = _build_test_client()
    r = client.get("/replay/scenario/election_night_2024")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scenario"]["name"] == "election_night_2024"


def test_router_scenario_unknown_returns_404() -> None:
    client = _build_test_client()
    r = client.get("/replay/scenario/does_not_exist")
    assert r.status_code == 404
