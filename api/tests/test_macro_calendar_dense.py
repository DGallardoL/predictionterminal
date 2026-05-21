"""Tests for the Wave-10 dense macro calendar expansion."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.macro_calendar import (
    _ALL_EVENTS_DENSE,
    _BOJ_2026,
    _CHINA_PMI_2026,
    _CPI_EUROZONE_2026,
    _CPI_JAPAN_2026,
    _ECB_2026,
    _JOBLESS_CLAIMS_2026,
    _OPEC_2026,
    next_releases,
    render_ics,
)
from pfm.macro_calendar import router as macro_calendar_router


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()
    yield
    reset_caches()


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(macro_calendar_router)
    return TestClient(app)


# --- raw schedule shape -----------------------------------------------------


def test_jobless_claims_52_weekly() -> None:
    """One Thursday per week for the calendar year."""
    assert 52 <= len(_JOBLESS_CLAIMS_2026) <= 53
    for d in _JOBLESS_CLAIMS_2026:
        assert d.weekday() == 3  # Thursday
        assert d.year == 2026


def test_ecb_8_meetings() -> None:
    assert len(_ECB_2026) == 8
    assert sorted(_ECB_2026) == _ECB_2026


def test_boj_8_meetings() -> None:
    assert len(_BOJ_2026) == 8


def test_opec_12_monthly() -> None:
    assert len(_OPEC_2026) == 12
    assert {d.month for d in _OPEC_2026} == set(range(1, 13))


def test_cpi_eurozone_12_monthly() -> None:
    assert len(_CPI_EUROZONE_2026) == 12


def test_cpi_japan_12_monthly() -> None:
    assert len(_CPI_JAPAN_2026) == 12


def test_china_pmi_12_monthly() -> None:
    assert len(_CHINA_PMI_2026) == 12


def test_dense_total_at_least_150() -> None:
    """Goal: ~150 events. Allow some slack for ±1 jobless-claims weeks."""
    assert len(_ALL_EVENTS_DENSE) >= 150


# --- next_releases windows --------------------------------------------------


def test_30_day_window_returns_at_least_15() -> None:
    out = next_releases(days_ahead=30, today=date(2026, 1, 1))
    assert len(out) >= 15


def test_90_day_window_returns_at_least_40() -> None:
    out = next_releases(days_ahead=90, today=date(2026, 1, 1))
    assert len(out) >= 40


def test_365_day_window_returns_at_least_150() -> None:
    out = next_releases(days_ahead=365, today=date(2026, 1, 1))
    assert len(out) >= 150


def test_event_has_dense_keys() -> None:
    out = next_releases(days_ahead=30, today=date(2026, 1, 1))
    for ev in out:
        assert "kind" in ev
        assert "title" in ev
        assert "importance" in ev
        assert "region" in ev
        assert "expected_impact" in ev
        assert ev["importance"] in {1, 2, 3}


def test_importance_filter_min_3() -> None:
    out = next_releases(days_ahead=365, today=date(2026, 1, 1), importance_min=3)
    for ev in out:
        assert ev["importance"] >= 3
    # Should still contain the major US prints.
    kinds = {ev["kind"] for ev in out}
    assert "fomc" in kinds
    assert "cpi" in kinds
    assert "nfp" in kinds


def test_importance_filter_min_2_drops_jobless() -> None:
    out = next_releases(days_ahead=365, today=date(2026, 1, 1), importance_min=2)
    kinds = {ev["kind"] for ev in out}
    assert "jobless_claims" not in kinds


def test_region_filter_us_excludes_eu() -> None:
    out = next_releases(days_ahead=365, today=date(2026, 1, 1), region="US")
    regions = {ev["region"] for ev in out}
    assert regions == {"US"}


def test_region_filter_eu_includes_ecb() -> None:
    out = next_releases(days_ahead=365, today=date(2026, 1, 1), region="EU")
    kinds = {ev["kind"] for ev in out}
    assert "ecb" in kinds
    assert "cpi_eurozone" in kinds


def test_region_filter_jp() -> None:
    out = next_releases(days_ahead=365, today=date(2026, 1, 1), region="JP")
    kinds = {ev["kind"] for ev in out}
    assert "boj" in kinds


def test_kind_filter() -> None:
    out = next_releases(days_ahead=365, today=date(2026, 1, 1), kind="opec")
    assert all(ev["kind"] == "opec" for ev in out)
    assert len(out) == 12


# --- /macro/upcoming endpoint ----------------------------------------------


def test_router_upcoming_with_filters() -> None:
    client = _make_app()
    r = client.get("/macro/upcoming?days=180&region=US&importance_min=3")
    assert r.status_code == 200
    body = r.json()
    assert body["filters"]["region"] == "US"
    assert body["filters"]["importance_min"] == 3
    for ev in body["events"]:
        assert ev["region"] == "US"
        assert ev["importance"] >= 3


def test_router_upcoming_kind_filter() -> None:
    client = _make_app()
    r = client.get("/macro/upcoming?days=365&kind=ecb")
    assert r.status_code == 200
    body = r.json()
    assert all(e["kind"] == "ecb" for e in body["events"])


def test_router_upcoming_importance_validation() -> None:
    client = _make_app()
    r = client.get("/macro/upcoming?importance_min=4")
    assert r.status_code == 422


# --- ICS export --------------------------------------------------------------


def test_render_ics_basic_envelope() -> None:
    out = next_releases(days_ahead=30, today=date(2026, 1, 1))
    body = render_ics(out)
    assert body.startswith("BEGIN:VCALENDAR")
    assert body.rstrip().endswith("END:VCALENDAR")
    assert body.count("BEGIN:VEVENT") == len(out)


def test_router_ics_returns_text_calendar() -> None:
    client = _make_app()
    r = client.get("/macro/calendar/export.ics?days=30")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/calendar")
    assert r.text.startswith("BEGIN:VCALENDAR")


def test_router_ics_filters_apply() -> None:
    client = _make_app()
    r = client.get("/macro/calendar/export.ics?days=365&kind=ecb")
    assert r.status_code == 200
    body = r.text
    # Number of ECB meetings remaining in 2026 depends on today's date —
    # cap at 8 (full year) and require at least 1.
    n = body.count("BEGIN:VEVENT")
    assert 1 <= n <= 8
    assert "CATEGORIES:ecb" in body
