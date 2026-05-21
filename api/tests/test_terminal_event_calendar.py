"""Unit tests for the Terminal macro-event calendar endpoint."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal_event_calendar import (
    EVENTS,
    _load_factor_ids,
    filter_upcoming,
    router,
)


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _write_factors_yml(tmp: Path) -> Path:
    p = tmp / "factors.yml"
    p.write_text(
        """
factors:
  - id: no_fed_cuts_2026
    name: No Fed cuts 2026
    slug: a
    source: polymarket
  - id: k_fed_jul_cut25
    name: K fed jul
    slug: b
    source: kalshi
  - id: k_cpi_above_4_27
    name: K CPI
    slug: c
    source: kalshi
  - id: us_recession_2026
    name: Recession
    slug: d
    source: polymarket
  - id: btc_ath_jun
    name: BTC ATH
    slug: e
    source: polymarket
  - id: dem_house_2026
    name: Dem House
    slug: f
    source: polymarket
  - id: openai_ipo_1t
    name: OpenAI IPO
    slug: g
    source: polymarket
  - id: random_unrelated_factor
    name: Random
    slug: h
    source: polymarket
"""
    )
    return p


def test_endpoint_returns_well_formed_response_within_horizon() -> None:
    """GET /terminal/calendar/upcoming returns valid schema and respects `days` filter."""
    client = _make_client()
    r = client.get("/terminal/calendar/upcoming?days=60")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) >= {"as_of", "horizon_days", "n_events", "events"}
    assert body["horizon_days"] == 60
    assert body["n_events"] == len(body["events"])
    for ev in body["events"]:
        assert set(ev.keys()) >= {
            "date",
            "time_et",
            "name",
            "category",
            "expected_impact_themes",
            "related_markets",
            "days_until",
        }
        assert 0 <= ev["days_until"] <= 60
        assert ev["category"] in {
            "fomc",
            "cpi",
            "nfp",
            "election",
            "crypto_expiry",
            "crypto_resolution",
            "debate",
            "ipo",
        }


def test_filter_upcoming_window_and_related_markets(tmp_path: Path) -> None:
    """`filter_upcoming` clamps to the horizon window and tags related markets via patterns."""
    factor_ids = _load_factor_ids(_write_factors_yml(tmp_path))
    assert "no_fed_cuts_2026" in factor_ids

    # Pick a today right before the March 2026 FOMC so it lands inside a 30-day window.
    today = date(2026, 3, 1)
    events = filter_upcoming(EVENTS, today=today, days=30, factor_ids=factor_ids)
    assert len(events) > 0

    # Every event must be in [today, today+30].
    for ev in events:
        ev_date = date.fromisoformat(ev.date)
        assert today <= ev_date <= date(2026, 3, 31)

    # FOMC March 18 must appear and tag fed-related factor IDs.
    fomc = [e for e in events if e.category == "fomc"]
    assert any(e.date == "2026-03-18" for e in fomc)
    fomc_mar = next(e for e in fomc if e.date == "2026-03-18")
    assert "no_fed_cuts_2026" in fomc_mar.related_markets
    assert "k_fed_jul_cut25" in fomc_mar.related_markets
    # Unrelated factors must NOT be tagged on FOMC.
    assert "random_unrelated_factor" not in fomc_mar.related_markets
    assert "btc_ath_jun" not in fomc_mar.related_markets


def test_categories_tag_correct_market_buckets(tmp_path: Path) -> None:
    """CPI events tag inflation+fed factors; election events tag balance-of-power factors;
    crypto events tag btc/eth factors. Confirms the regex tagging logic doesn't bleed."""
    factor_ids = _load_factor_ids(_write_factors_yml(tmp_path))

    # Wide horizon so we capture CPI, election, crypto windows in a single call.
    today = date(2026, 1, 1)
    events = filter_upcoming(EVENTS, today=today, days=400, factor_ids=factor_ids)
    by_cat: dict[str, list] = {}
    for e in events:
        by_cat.setdefault(e.category, []).append(e)

    # CPI release should tag k_cpi_above_4_27 and fed-related factors.
    cpi = by_cat["cpi"][0]
    assert "k_cpi_above_4_27" in cpi.related_markets
    assert "no_fed_cuts_2026" in cpi.related_markets
    assert "btc_ath_jun" not in cpi.related_markets

    # Midterm election should tag dem_house_2026 (balance-of-power) but NOT BTC.
    midterms = next(e for e in by_cat["election"] if e.date == "2026-11-03")
    assert "dem_house_2026" in midterms.related_markets
    assert "btc_ath_jun" not in midterms.related_markets

    # Crypto expiry should tag btc_ath_jun but NOT fed factors.
    crypto = by_cat["crypto_expiry"][0]
    assert "btc_ath_jun" in crypto.related_markets
    assert "no_fed_cuts_2026" not in crypto.related_markets

    # IPO event tags openai_ipo_1t.
    ipo_evs = by_cat["ipo"]
    assert any("openai_ipo_1t" in e.related_markets for e in ipo_evs)


# --- additional coverage ----------------------------------------------------


def test_endpoint_rejects_out_of_range_days() -> None:
    """`days` is constrained to [1, 730] by the Query validator."""
    client = _make_client()
    assert client.get("/terminal/calendar/upcoming?days=0").status_code == 422
    assert client.get("/terminal/calendar/upcoming?days=731").status_code == 422
    # A negative value also short-circuits at the Query validator.
    assert client.get("/terminal/calendar/upcoming?days=-1").status_code == 422


def test_filter_upcoming_returns_empty_outside_horizon() -> None:
    """A horizon strictly before any scheduled event should return [] cleanly."""
    # 2030 is past every event in EVENTS — must produce zero results.
    today = date(2030, 1, 1)
    out = filter_upcoming(EVENTS, today=today, days=10, factor_ids=[])
    assert out == []


def test_load_factor_ids_handles_missing_and_malformed_files(tmp_path: Path) -> None:
    """Both FileNotFound and bad YAML must produce an empty list (never raise)."""
    # Missing file.
    missing = tmp_path / "nope.yml"
    assert _load_factor_ids(missing) == []

    # Malformed YAML.
    bad = tmp_path / "bad.yml"
    bad.write_text(": : : not: valid:\n  - [")
    assert _load_factor_ids(bad) == []

    # File present but no factors block — also empty.
    empty = tmp_path / "empty.yml"
    empty.write_text("other_root:\n  - foo\n")
    assert _load_factor_ids(empty) == []


def test_endpoint_response_keys_and_n_events_consistency() -> None:
    """Response shape matches the CalendarResponse model exactly."""
    client = _make_client()
    body = client.get("/terminal/calendar/upcoming?days=180").json()
    assert set(body.keys()) == {"as_of", "horizon_days", "n_events", "events"}
    assert body["n_events"] == len(body["events"])
    # Events are returned sorted by (date, time_et).
    sort_keys = [(e["date"], e["time_et"]) for e in body["events"]]
    assert sort_keys == sorted(sort_keys)


def test_endpoint_caches_response_across_calls(monkeypatch) -> None:
    """Second call to /calendar/upcoming should bypass yaml parse + filter.

    The 1 s warm-call cost on the original endpoint was dominated by parsing
    10800-line factors.yml + regex-compiling on every request. The fix is a
    response cache keyed on (today, days); confirm it's wired by counting
    yaml.safe_load invocations across two identical hits.
    """
    import pfm.terminal_event_calendar as ec_mod
    from pfm.cache_utils import get_cache

    # Clear the namespace cache so this test is hermetic regardless of order.
    get_cache("terminal_calendar").clear()

    n_yaml_loads = {"n": 0}
    real_safe_load = ec_mod.yaml.safe_load

    def _counting_load(*args, **kwargs):
        n_yaml_loads["n"] += 1
        return real_safe_load(*args, **kwargs)

    monkeypatch.setattr(ec_mod.yaml, "safe_load", _counting_load)

    client = _make_client()
    r1 = client.get("/terminal/calendar/upcoming?days=60")
    r2 = client.get("/terminal/calendar/upcoming?days=60")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()
    # First call may parse yaml once; second call must NOT re-parse.
    # (The factor_ids cache and the response cache both protect this.)
    assert n_yaml_loads["n"] <= 1, (
        f"factors.yml should be parsed at most once per warm cycle, got {n_yaml_loads['n']}"
    )


def test_load_factor_ids_is_cached(tmp_path: Path) -> None:
    """_load_factor_ids must memoise on the resolved path."""
    from pfm.cache_utils import get_cache

    get_cache("terminal_calendar").clear()

    p = _write_factors_yml(tmp_path)
    ids_1 = _load_factor_ids(p)
    # Mutate the file under the cache — cached call should still return old.
    p.write_text("factors: []\n")
    ids_2 = _load_factor_ids(p)
    assert ids_1 == ids_2, "cache should pin the first load until TTL expires"
