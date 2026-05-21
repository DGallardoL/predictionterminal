"""Tests for the populated replay scenarios.

Verifies that each pre-baked scenario in ``pfm.replay_mode`` carries real
slugs and tickers (no more empty ``slugs:0 tickers:0`` outputs), that the
preflight endpoint correctly classifies live/resolved/missing slugs against
a respx-mocked Gamma, and that the PnL endpoint returns coherent numbers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.replay_mode as rm

GAMMA_URL = "https://gamma-api.polymarket.com"


# ---------------------------------------------------------------------------
# Hermetic patching for the existing replay machinery (PM history + yfinance)
# ---------------------------------------------------------------------------


def _synthetic_pm_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D", tz="UTC").normalize()
    n = len(idx)
    t = np.arange(n) / max(n, 1)
    price = (0.50 + 0.20 * np.sin(2 * np.pi * t * 1.5)).clip(0.05, 0.95)
    df = pd.DataFrame({"price": price}, index=idx)
    df.index.name = "date"
    return df


def _synthetic_yf_rows(
    start_iso: str, end_iso: str, base: float = 100.0
) -> tuple[tuple[str, float], ...]:
    idx = pd.date_range(start_iso, end_iso, freq="D", tz="UTC").normalize()
    n = len(idx)
    rng = np.random.default_rng(42)
    drift = np.cumsum(rng.normal(0.0005, 0.01, n))
    closes = base * np.exp(drift)
    return tuple((d.isoformat(), float(c)) for d, c in zip(idx, closes, strict=False))


@pytest.fixture(autouse=True)
def _patch_external(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Patch PM history + yfinance + Gamma URL so tests are hermetic."""

    def fake_resolve_pm_history(slug: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return _synthetic_pm_history(start, end)

    monkeypatch.setattr(rm, "_resolve_pm_history", fake_resolve_pm_history)

    rm._yf_close_cached.cache_clear()

    def fake_yf(ticker: str, start_iso: str, end_iso: str):
        return _synthetic_yf_rows(start_iso, end_iso, base=100.0 + len(ticker))

    monkeypatch.setattr(rm, "_yf_close_cached", fake_yf)
    monkeypatch.setattr(rm, "_gamma_url", lambda: GAMMA_URL)
    yield


def _build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(rm.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1) Each scenario has real slugs + tickers
# ---------------------------------------------------------------------------


SCENARIO_IDS = ["election_night_2024", "fomc_2024_09", "btc_ath_2024_11", "covid_crash_2020_03"]


class TestScenariosPopulated:
    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    def test_scenario_has_min_slugs_and_tickers(self, scenario_id: str) -> None:
        sc = rm.SCENARIOS[scenario_id]
        if scenario_id == "covid_crash_2020_03":
            # COVID-era PM coverage was thin; keep a realistic floor of 2.
            assert len(sc.pm_slugs) >= 2, f"{scenario_id} slugs: {sc.pm_slugs}"
        else:
            assert len(sc.pm_slugs) >= 4, f"{scenario_id} slugs: {sc.pm_slugs}"
        assert len(sc.equity_tickers) >= 4, f"{scenario_id} tickers: {sc.equity_tickers}"

    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    def test_scenario_has_narrative_and_description(self, scenario_id: str) -> None:
        sc = rm.SCENARIOS[scenario_id]
        assert sc.description, f"{scenario_id} missing description"
        assert sc.narrative, f"{scenario_id} missing narrative"
        # Narrative should be non-trivial — at least 200 chars / multiple paras.
        assert len(sc.narrative) >= 200, f"{scenario_id} narrative too short: {len(sc.narrative)}"

    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    def test_scenario_has_expected_pnl_per_dollar_long(self, scenario_id: str) -> None:
        sc = rm.SCENARIOS[scenario_id]
        assert sc.expected_pnl_per_dollar_long, (
            f"{scenario_id} missing expected_pnl_per_dollar_long"
        )
        # Every ticker should have a corresponding return entry.
        for tkr in sc.equity_tickers:
            assert tkr in sc.expected_pnl_per_dollar_long, (
                f"{scenario_id}: missing return for {tkr}"
            )


# ---------------------------------------------------------------------------
# 2) replay_scenario(scenario_id) returns the full struct
# ---------------------------------------------------------------------------


class TestScenarioStruct:
    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    def test_replay_scenario_returns_full_payload(self, scenario_id: str) -> None:
        out = rm.replay_scenario(scenario_id)  # type: ignore[arg-type]
        sc_block = out["scenario"]
        assert sc_block["id"] == scenario_id
        assert sc_block["title"]
        assert sc_block["description"]
        assert sc_block["narrative"]
        assert sc_block["as_of_iso"]
        assert isinstance(sc_block["slugs"], list)
        assert isinstance(sc_block["tickers"], list)
        # slugs / tickers must mirror the curated values
        sc_obj = rm.SCENARIOS[scenario_id]
        assert sc_block["slugs"] == list(sc_obj.pm_slugs)
        assert sc_block["tickers"] == list(sc_obj.equity_tickers)
        assert sc_block["expected_pnl_per_dollar_long"] is not None

    def test_list_scenarios_includes_full_curated_data(self) -> None:
        rows = rm.list_scenarios()
        assert len(rows) == 4
        for r in rows:
            assert r["id"] in SCENARIO_IDS
            assert r["slugs"], f"{r['id']} has empty slugs"
            assert r["tickers"], f"{r['id']} has empty tickers"
            assert r["narrative"], f"{r['id']} has empty narrative"
            assert r["n_markets"] == len(r["slugs"])
            assert r["n_equities"] == len(r["tickers"])


# ---------------------------------------------------------------------------
# 3) Preflight endpoint with respx-mocked Gamma
# ---------------------------------------------------------------------------


def _gamma_market(slug: str, *, closed: bool = False, active: bool = True) -> dict[str, Any]:
    return {
        "slug": slug,
        "question": f"Will the prediction-market {slug} resolve YES?",
        "clobTokenIds": json.dumps(["tok-a", "tok-b"]),
        "closed": closed,
        "active": active,
        "endDate": "2024-12-31T00:00:00Z",
    }


class TestPreflight:
    @respx.mock
    def test_preflight_all_resolved(self) -> None:
        # Election scenario — historic markets should all be `closed=True`.
        sc = rm.SCENARIOS["election_night_2024"]

        def _handler(req: httpx.Request) -> httpx.Response:
            slug = req.url.params.get("slug", "")
            if slug in sc.pm_slugs:
                return httpx.Response(200, json=[_gamma_market(slug, closed=True, active=False)])
            return httpx.Response(200, json=[])

        respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_handler)

        out = rm.preflight_scenario("election_night_2024")
        assert out["scenario_id"] == "election_night_2024"
        assert len(out["slugs_status"]) == len(sc.pm_slugs)
        for entry in out["slugs_status"]:
            assert entry["status"] == "resolved"
        assert out["can_replay"] is True

    @respx.mock
    def test_preflight_marks_missing_and_suggests_substitutes(self) -> None:
        # FOMC scenario — make one slug missing on Gamma.
        sc = rm.SCENARIOS["fomc_2024_09"]
        missing_slug = "recession-2024"

        def _handler(req: httpx.Request) -> httpx.Response:
            slug = req.url.params.get("slug", "")
            if slug == missing_slug:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[_gamma_market(slug, closed=True, active=False)])

        respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_handler)

        out = rm.preflight_scenario("fomc_2024_09")
        statuses = {e["slug"]: e["status"] for e in out["slugs_status"]}
        assert statuses[missing_slug] == "missing"
        # All other FOMC slugs should be resolved
        for s in sc.pm_slugs:
            if s != missing_slug:
                assert statuses[s] == "resolved"

    @respx.mock
    def test_preflight_router_endpoint(self) -> None:
        sc = rm.SCENARIOS["btc_ath_2024_11"]

        def _handler(req: httpx.Request) -> httpx.Response:
            slug = req.url.params.get("slug", "")
            if slug in sc.pm_slugs:
                return httpx.Response(200, json=[_gamma_market(slug, closed=False, active=True)])
            return httpx.Response(200, json=[])

        respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_handler)

        client = _build_test_client()
        r = client.get("/replay/scenario/btc_ath_2024_11/preflight")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["scenario_id"] == "btc_ath_2024_11"
        assert all(e["status"] == "live" for e in body["slugs_status"])
        assert body["can_replay"] is True

    def test_preflight_unknown_scenario_returns_404(self) -> None:
        client = _build_test_client()
        r = client.get("/replay/scenario/does_not_exist/preflight")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 4) PnL endpoint
# ---------------------------------------------------------------------------


class TestScenarioPnL:
    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    def test_compute_scenario_pnl_returns_finite_values(self, scenario_id: str) -> None:
        out = rm.compute_scenario_pnl(scenario_id, capital_usd=10_000.0)
        assert out["scenario_id"] == scenario_id
        assert out["capital_usd"] == 10_000.0
        assert isinstance(out["ticker_returns"], dict)
        assert len(out["ticker_returns"]) >= 4 if scenario_id != "covid_crash_2020_03" else True
        assert np.isfinite(out["basket_pnl_long_only"])
        assert np.isfinite(out["basket_pnl_equal_weighted"])

    def test_pnl_scales_linearly_with_capital(self) -> None:
        a = rm.compute_scenario_pnl("election_night_2024", capital_usd=1_000.0)
        b = rm.compute_scenario_pnl("election_night_2024", capital_usd=10_000.0)
        # Both basket measures should scale 10x within rounding tolerance.
        assert abs(b["basket_pnl_long_only"] - 10 * a["basket_pnl_long_only"]) < 0.5
        assert abs(b["basket_pnl_equal_weighted"] - 10 * a["basket_pnl_equal_weighted"]) < 0.5

    def test_pnl_router_endpoint(self) -> None:
        client = _build_test_client()
        r = client.get("/replay/scenario/fomc_2024_09/pnl", params={"capital": 25000})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["scenario_id"] == "fomc_2024_09"
        assert body["capital_usd"] == 25000.0
        assert isinstance(body["ticker_returns"], dict)
        assert len(body["ticker_returns"]) == len(rm.SCENARIOS["fomc_2024_09"].equity_tickers)

    def test_pnl_invalid_capital_rejected(self) -> None:
        client = _build_test_client()
        r = client.get("/replay/scenario/election_night_2024/pnl", params={"capital": -5})
        assert r.status_code in (400, 422)

    def test_pnl_unknown_scenario_returns_404(self) -> None:
        client = _build_test_client()
        r = client.get("/replay/scenario/does_not_exist/pnl")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 5) Integration: GET /replay/scenarios returns 4 with full data
# ---------------------------------------------------------------------------


class TestScenariosListingIntegration:
    def test_full_listing_returns_four_with_full_payload(self) -> None:
        client = _build_test_client()
        r = client.get("/replay/scenarios")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_scenarios"] == 4
        ids = {s["id"] for s in body["scenarios"]}
        assert ids == set(SCENARIO_IDS)
        for s in body["scenarios"]:
            assert s["slugs"], f"{s['id']} returned with empty slugs"
            assert s["tickers"], f"{s['id']} returned with empty tickers"
            assert s["narrative"]
            assert s["n_markets"] == len(s["slugs"])
            assert s["n_equities"] == len(s["tickers"])

    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    def test_scenario_detail_endpoint_returns_curated_metadata(self, scenario_id: str) -> None:
        client = _build_test_client()
        r = client.get(f"/replay/scenario/{scenario_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["scenario"]["id"] == scenario_id
        assert body["scenario"]["narrative"]
        assert body["scenario"]["slugs"]
        assert body["scenario"]["tickers"]
