"""Tests for ``pfm.terminal_calendar_unified``.

Scope:

* ``GET /terminal/calendar`` returns a mix of resolution + earnings +
  macro items when no kind filter is set.
* ``kinds=earnings`` (and ``kinds=macro``) restrict the response.
* The ``start`` / ``end`` window is honoured in both directions.
* Items are emitted in chronological order.
* A bad date / unknown ``kinds`` produces a 422.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal_calendar_unified import router as calendar_router


@pytest.fixture
def calendar_client() -> TestClient:
    app = FastAPI()
    app.include_router(calendar_router)
    return TestClient(app)


# --- mix-of-kinds -----------------------------------------------------------


def test_calendar_mix_default(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={"start": "2026-01-01", "end": "2026-12-31"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["start"] == "2026-01-01"
    assert body["end"] == "2026-12-31"
    assert body["total"] == len(body["items"])
    kinds = {it["kind"] for it in body["items"]}
    # 2026 has lots of macro and earnings entries; resolutions are
    # cluster-driven so they may or may not land in this window —
    # but the macro + earnings union must be non-empty.
    assert "earnings" in kinds
    assert "macro" in kinds


# --- single-kind filters ----------------------------------------------------


def test_calendar_filter_earnings_only(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={
            "start": "2026-04-01",
            "end": "2026-08-31",
            "kinds": "earnings",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] > 0
    assert all(it["kind"] == "earnings" for it in body["items"])
    # Tickers must be set on earnings rows.
    assert all(it["ticker"] for it in body["items"])
    assert all(it["slug"] is None for it in body["items"])


def test_calendar_filter_macro_only(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={
            "start": "2026-01-01",
            "end": "2026-06-30",
            "kinds": "macro",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] > 0
    assert all(it["kind"] == "macro" for it in body["items"])
    # Macro rows have neither slug nor ticker.
    assert all(it["slug"] is None and it["ticker"] is None for it in body["items"])


def test_calendar_filter_two_kinds(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={
            "start": "2026-01-01",
            "end": "2026-12-31",
            "kinds": "earnings,macro",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    kinds = {it["kind"] for it in body["items"]}
    assert kinds == {"earnings", "macro"}
    assert "resolution" not in kinds


# --- date range -------------------------------------------------------------


def test_calendar_date_range_excludes_outside(calendar_client: TestClient) -> None:
    # Tight window in May 2026.
    resp = calendar_client.get(
        "/terminal/calendar",
        params={
            "start": "2026-05-01",
            "end": "2026-05-31",
            "kinds": "macro",
        },
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    for it in items:
        assert "2026-05-01" <= it["date"] <= "2026-05-31"


def test_calendar_empty_window(calendar_client: TestClient) -> None:
    # Far-future window with no hardcoded items.
    resp = calendar_client.get(
        "/terminal/calendar",
        params={"start": "2030-01-01", "end": "2030-12-31"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 0
    assert body["items"] == []


def test_calendar_start_after_end_is_422(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={"start": "2026-12-31", "end": "2026-01-01"},
    )
    assert resp.status_code == 422


# --- sort order -------------------------------------------------------------


def test_calendar_sorted_by_date(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    dates = [it["date"] for it in items]
    assert dates == sorted(dates)


# --- input validation ------------------------------------------------------


def test_calendar_bad_iso_date(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={"start": "not-a-date", "end": "2026-12-31"},
    )
    assert resp.status_code == 422


def test_calendar_unknown_kind(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={
            "start": "2026-01-01",
            "end": "2026-12-31",
            "kinds": "not_a_kind",
        },
    )
    assert resp.status_code == 422


# --- theme filter ----------------------------------------------------------


def test_calendar_theme_filter_nvda(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={
            "start": "2026-01-01",
            "end": "2026-12-31",
            "kinds": "earnings",
            "theme": "nvda",
        },
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert len(items) > 0
    assert all(it["ticker"] == "NVDA" for it in items)


def test_calendar_theme_filter_inflation(calendar_client: TestClient) -> None:
    resp = calendar_client.get(
        "/terminal/calendar",
        params={
            "start": "2026-01-01",
            "end": "2026-12-31",
            "kinds": "macro",
            "theme": "inflation",
        },
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    # All CPI entries have the "inflation" theme.
    assert len(items) > 0
    assert all("CPI" in it["title"] for it in items)
