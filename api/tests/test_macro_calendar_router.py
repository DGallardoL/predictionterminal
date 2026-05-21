"""W12-15 — Tests for ``pfm.macro_calendar_router``.

Distinct from ``test_macro_calendar.py`` which targets the older dense
multi-region calendar at ``/macro/upcoming``. This suite exercises the
new ``/macro/calendar`` endpoint with hardcoded 2026 cadence.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.macro_calendar_router import (
    _EVENTS_2026,
    _FOMC_2026,
    upcoming_events,
)
from pfm.macro_calendar_router import (
    router as macro_calendar_router,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(macro_calendar_router)
    return TestClient(app)


# --- Event-list sanity checks ------------------------------------------------


def test_events_include_all_fomc_2026() -> None:
    """Every spec'd FOMC decision date appears as an event."""
    fomc_dates = {e["date"] for e in _EVENTS_2026 if e["event"] == "FOMC Decision"}
    expected = {d.isoformat() for d in _FOMC_2026}
    assert expected.issubset(fomc_dates)
    assert len(_FOMC_2026) == 8


def test_events_include_fomc_minutes_three_weeks_after() -> None:
    """Minutes land ~3 weeks (21 days) after each FOMC decision."""
    minutes_dates = {e["date"] for e in _EVENTS_2026 if e["event"] == "FOMC Minutes"}
    for d in _FOMC_2026:
        assert (d + timedelta(days=21)).isoformat() in minutes_dates


def test_nfp_is_first_friday_each_month() -> None:
    """All 12 NFP rows fall on the first Friday of the month."""
    nfp = [e for e in _EVENTS_2026 if e["event"] == "Nonfarm Payrolls"]
    assert len(nfp) == 12
    for e in nfp:
        d = date.fromisoformat(e["date"])
        assert d.weekday() == calendar.FRIDAY
        assert d.day <= 7


def test_cpi_dates_are_in_12_to_15_range() -> None:
    """CPI representative day is within the 12-15 window per spec."""
    cpi = [e for e in _EVENTS_2026 if e["event"] == "CPI"]
    assert len(cpi) == 12
    for e in cpi:
        d = date.fromisoformat(e["date"])
        assert 12 <= d.day <= 15


def test_ppi_dates_are_in_13_to_16_range() -> None:
    """PPI representative day is within the 13-16 window per spec."""
    ppi = [e for e in _EVENTS_2026 if e["event"] == "PPI"]
    assert len(ppi) == 12
    for e in ppi:
        d = date.fromisoformat(e["date"])
        assert 13 <= d.day <= 16


def test_retail_sales_dates_are_in_14_to_17_range() -> None:
    """Retail Sales representative day is within the 14-17 window."""
    rs = [e for e in _EVENTS_2026 if e["event"] == "Retail Sales"]
    assert len(rs) == 12
    for e in rs:
        d = date.fromisoformat(e["date"])
        assert 14 <= d.day <= 17


def test_gdp_has_four_quarterly_entries() -> None:
    gdp = [e for e in _EVENTS_2026 if e["event"].startswith("GDP")]
    assert len(gdp) == 4


# --- upcoming_events() unit tests --------------------------------------------


def test_upcoming_events_window_filters_correctly() -> None:
    """Only events between today and today+days are returned, inclusive."""
    today = date(2026, 5, 1)
    out = upcoming_events(today=today, window_days=14)
    for e in out:
        d = date.fromisoformat(e["date"])
        assert today <= d <= today + timedelta(days=14)


def test_upcoming_events_no_results_in_far_future() -> None:
    """Far-future window past 2026 has zero events (hardcoded scope)."""
    out = upcoming_events(today=date(2030, 1, 1), window_days=30)
    assert out == []


def test_upcoming_events_ordering_ascending() -> None:
    """Returned events are sorted ascending by date."""
    out = upcoming_events(today=date(2026, 1, 1), window_days=365)
    dates = [date.fromisoformat(e["date"]) for e in out]
    assert dates == sorted(dates)
    assert len(out) >= 50  # full year should yield many events


def test_upcoming_events_category_filter() -> None:
    out = upcoming_events(today=date(2026, 1, 1), window_days=365, category="fed")
    assert len(out) > 0
    for e in out:
        assert e["category"] == "fed"


def test_upcoming_events_importance_filter() -> None:
    out = upcoming_events(today=date(2026, 1, 1), window_days=365, importance="high")
    assert len(out) > 0
    for e in out:
        assert e["importance"] == "high"


# --- HTTP endpoint tests ------------------------------------------------------


def test_get_calendar_default_30_days(client: TestClient) -> None:
    """Default request returns window_days=30 and the canonical shape."""
    resp = client.get("/macro/calendar")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"checked_at", "window_days", "events"}
    assert body["window_days"] == 30
    assert isinstance(body["events"], list)
    # Each event has the documented shape
    for e in body["events"]:
        assert set(e.keys()) == {"date", "event", "category", "importance"}


def test_get_calendar_custom_days_param(client: TestClient) -> None:
    """``days=90`` produces a strictly larger (or equal) result set."""
    a = client.get("/macro/calendar?days=7").json()
    b = client.get("/macro/calendar?days=90").json()
    assert a["window_days"] == 7
    assert b["window_days"] == 90
    assert len(b["events"]) >= len(a["events"])


def test_get_calendar_category_filter_fed(client: TestClient) -> None:
    """``?category=fed`` keeps only Fed-tagged events."""
    resp = client.get("/macro/calendar?days=365&category=fed")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["events"]) > 0
    for e in body["events"]:
        assert e["category"] == "fed"


def test_get_calendar_importance_high(client: TestClient) -> None:
    resp = client.get("/macro/calendar?days=365&importance=high")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["events"]) > 0
    for e in body["events"]:
        assert e["importance"] == "high"


def test_get_calendar_invalid_category_returns_400(client: TestClient) -> None:
    resp = client.get("/macro/calendar?category=bogus")
    assert resp.status_code == 400


def test_get_calendar_invalid_importance_returns_400(client: TestClient) -> None:
    resp = client.get("/macro/calendar?importance=critical")
    assert resp.status_code == 400


def test_get_calendar_days_out_of_range_returns_422(client: TestClient) -> None:
    """FastAPI rejects ``days=0`` and ``days>365`` per the Query constraints."""
    assert client.get("/macro/calendar?days=0").status_code == 422
    assert client.get("/macro/calendar?days=999").status_code == 422


def test_get_calendar_combined_filters(client: TestClient) -> None:
    """Category + importance compose correctly."""
    resp = client.get("/macro/calendar?days=365&category=inflation&importance=high")
    assert resp.status_code == 200
    body = resp.json()
    # CPI is the inflation/high candidate (PPI is medium); should yield 12 in 2026
    assert all(e["category"] == "inflation" for e in body["events"])
    assert all(e["importance"] == "high" for e in body["events"])


def test_get_calendar_results_are_date_sorted(client: TestClient) -> None:
    body = client.get("/macro/calendar?days=365").json()
    dates = [e["date"] for e in body["events"]]
    assert dates == sorted(dates)


def test_get_calendar_checked_at_is_iso_utc(client: TestClient) -> None:
    body = client.get("/macro/calendar").json()
    s = body["checked_at"]
    # Must be parseable as ISO 8601 with a UTC marker
    assert s.endswith("Z") or "+00:00" in s
    # round-trip parse
    from datetime import datetime

    datetime.fromisoformat(s.replace("Z", "+00:00"))
