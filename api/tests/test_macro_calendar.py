"""Tests for ``pfm.macro_calendar`` — hardcoded 2026 macro release dates."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.macro_calendar import (
    _ALL_EVENTS,
    _CPI_2026,
    _FOMC_2026,
    _GDP_2026,
    _NFP_2026,
    _PPI_2026,
    _RETAIL_SALES_2026,
    next_releases,
)
from pfm.macro_calendar import (
    router as macro_calendar_router,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    reset_caches()
    yield
    reset_caches()


# --- raw schedule -----------------------------------------------------------


def test_fomc_has_8_meetings_2026() -> None:
    """The FOMC schedules 8 meetings per year."""
    assert len(_FOMC_2026) == 8
    # All in 2026
    assert all(d.year == 2026 for d in _FOMC_2026)
    # Strictly increasing
    assert sorted(_FOMC_2026) == _FOMC_2026


def test_cpi_has_12_releases_2026() -> None:
    assert len(_CPI_2026) == 12
    assert all(d.year == 2026 for d in _CPI_2026)


def test_nfp_first_friday_pattern() -> None:
    """NFP fires on the first Friday — weekday=4, day <=7."""
    for d in _NFP_2026:
        assert d.weekday() == 4, f"{d} is not a Friday"
        assert d.day <= 7, f"{d} is not the first Friday"


def test_ppi_has_12_releases_2026() -> None:
    assert len(_PPI_2026) == 12


def test_retail_sales_has_12_releases_2026() -> None:
    assert len(_RETAIL_SALES_2026) == 12


def test_gdp_has_4_releases_2026() -> None:
    """One advance estimate per quarter — Jan/Apr/Jul/Oct."""
    assert len(_GDP_2026) == 4
    assert {d.month for d in _GDP_2026} == {1, 4, 7, 10}


def test_all_events_aggregate_count() -> None:
    """8 FOMC + 12 CPI + 12 NFP + 12 PPI + 12 retail + 4 GDP = 60."""
    assert len(_ALL_EVENTS) == 8 + 12 + 12 + 12 + 12 + 4


# --- next_releases ----------------------------------------------------------


def test_next_releases_window_filtering() -> None:
    """A 30-day window from 2026-01-01 must include the Jan FOMC + CPI."""
    out = next_releases(days_ahead=30, today=date(2026, 1, 1))
    types = {e["type"] for e in out}
    assert "fomc" in types  # 2026-01-28
    assert "cpi" in types  # 2026-01-14
    assert "nfp" in types  # 2026-01-09


def test_next_releases_excludes_past_events() -> None:
    """Anchored at 2026-02-01, the Jan releases must be gone."""
    out = next_releases(days_ahead=30, today=date(2026, 2, 1))
    for e in out:
        assert e["date"] >= "2026-02-01"


def test_next_releases_returns_days_until() -> None:
    out = next_releases(days_ahead=14, today=date(2026, 1, 1))
    for e in out:
        assert e["days_until"] >= 0
        assert e["days_until"] <= 14


def test_next_releases_zero_days_returns_today_only() -> None:
    """``days_ahead=0`` matches just today (no events match exactly)."""
    out = next_releases(days_ahead=0, today=date(2026, 1, 1))
    # 2026-01-01 has no event, so the result is empty.
    assert out == []


def test_next_releases_includes_fomc_jan_28() -> None:
    """Look 60 days from 2026-01-01 — must surface the Jan-28 FOMC."""
    out = next_releases(days_ahead=60, today=date(2026, 1, 1))
    fomc_dates = {e["date"] for e in out if e["type"] == "fomc"}
    assert "2026-01-28" in fomc_dates


def test_next_releases_sorted() -> None:
    out = next_releases(days_ahead=365, today=date(2026, 1, 1))
    dates = [e["date"] for e in out]
    assert dates == sorted(dates)


def test_next_releases_event_shape() -> None:
    out = next_releases(days_ahead=30, today=date(2026, 1, 1))
    for e in out:
        assert {"date", "event", "type", "source", "impact", "days_until"} <= e.keys()
        assert e["impact"] in {"high", "medium", "low"}


# --- router -----------------------------------------------------------------


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(macro_calendar_router)
    return TestClient(app)


def test_router_upcoming_default() -> None:
    client = _make_app()
    r = client.get("/macro/upcoming")
    assert r.status_code == 200
    body = r.json()
    assert "events" in body
    assert "as_of" in body
    assert body["days_ahead"] == 30


def test_router_upcoming_custom_window() -> None:
    client = _make_app()
    r = client.get("/macro/upcoming?days=180")
    assert r.status_code == 200
    body = r.json()
    assert body["days_ahead"] == 180


def test_router_rejects_zero_days() -> None:
    """Validation: ``days >= 1``."""
    client = _make_app()
    r = client.get("/macro/upcoming?days=0")
    assert r.status_code == 422
