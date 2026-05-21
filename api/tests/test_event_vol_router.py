"""Tests for the ``pfm.vol.event_vol_router`` (B3).

The router is feature-gated in ``pfm.main`` behind
``PFM_VOL_EVENT_ENABLED=1``. Tests here mount the router on a fresh
``FastAPI()`` directly to avoid the global lifespan and prove the
endpoints work in isolation.
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import get_cache
from pfm.dependencies import get_kalshi_client, get_polymarket_client
from pfm.vol.event_calendar import CALENDAR
from pfm.vol.event_vol_router import router as event_vol_router

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
FOMC_JUN = "fomc-2026-06"


# --------------------------------------------------------------------------- #
# Stubs                                                                        #
# --------------------------------------------------------------------------- #


class _StubPolymarketClient:
    """Minimal stub: matches the attributes ``_polymarket_midpoint`` reads."""

    def __init__(self, midpoints: dict[str, float]) -> None:
        # ``_polymarket_midpoint`` reaches through ``._client`` so we need
        # a real httpx.Client wired to a MockTransport that returns the
        # midpoints encoded as Gamma market dicts.
        self.gamma_url = GAMMA
        self._midpoints = midpoints
        self._client = httpx.Client(transport=httpx.MockTransport(self._respond))

    def _respond(self, request: httpx.Request) -> httpx.Response:
        slug = request.url.params.get("slug")
        if slug is None or slug not in self._midpoints:
            return httpx.Response(200, json=[])
        mid = self._midpoints[slug]
        return httpx.Response(
            200,
            json=[
                {
                    "slug": slug,
                    "question": f"Q: {slug}?",
                    "clobTokenIds": json.dumps(["111", "222"]),
                    "bestBid": max(0.0, mid - 0.005),
                    "bestAsk": min(1.0, mid + 0.005),
                    "lastTradePrice": mid,
                    "closed": False,
                    "active": True,
                    "startDate": "2025-12-01T00:00:00Z",
                    "endDate": "2026-12-31T00:00:00Z",
                }
            ],
        )


class _StubKalshiClient:
    """Stub Kalshi client; satisfies any ticker passed to it."""

    def get_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1440,
        series_ticker: str | None = None,
    ) -> Any:
        import pandas as pd

        return pd.DataFrame(
            {
                "price": [0.30],
                "volume": [1000.0],
                "open_interest": [500.0],
                "yes_bid": [0.295],
                "yes_ask": [0.305],
                "spread": [0.01],
            },
            index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC").normalize()], name="date"),
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_app() -> tuple[FastAPI, _StubPolymarketClient, _StubKalshiClient]:
    """Mount the router on a fresh FastAPI with stub DI."""
    # Populate the stub with every Polymarket slug in the curated calendar.
    midpoints: dict[str, float] = {}
    for entry in CALENDAR:
        for slug_def in entry.outcome_slugs:
            if slug_def.venue == "polymarket":
                midpoints[slug_def.slug] = 0.20
    poly = _StubPolymarketClient(midpoints)
    kalshi = _StubKalshiClient()

    app = FastAPI()
    app.include_router(event_vol_router)
    app.dependency_overrides[get_polymarket_client] = lambda: poly
    app.dependency_overrides[get_kalshi_client] = lambda: kalshi
    return app, poly, kalshi


@pytest.fixture(autouse=True)
def _clear_event_signal_cache() -> None:
    """The router caches signals per 5-min bucket; flush between tests."""
    get_cache("event_signal").clear()
    yield
    get_cache("event_signal").clear()


@pytest.fixture
def fake_now() -> Any:
    """Freeze ``datetime.now`` inside the router and signal modules to 2026-05-15."""
    from datetime import UTC, datetime

    frozen = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> _FrozenDateTime:  # type: ignore[override]
            return frozen if tz is None else frozen.astimezone(tz)

    return frozen


# --------------------------------------------------------------------------- #
# 8 — calendar endpoint                                                        #
# --------------------------------------------------------------------------- #


def test_get_calendar_returns_upcoming_events(fake_now: Any) -> None:
    app, _poly, _kalshi = _make_app()
    from datetime import datetime

    with patch("pfm.vol.event_vol_router.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = datetime
        with TestClient(app) as client:
            r = client.get("/vol/event/calendar", params={"lookahead_days": 120})

    assert r.status_code == 200, r.text
    payload = r.json()
    assert isinstance(payload, list)
    ids = {e["event_id"] for e in payload}
    # FOMC June 2026 sits 33 days after our frozen-now → must be in [0, 120].
    assert FOMC_JUN in ids


def test_get_calendar_kind_filter(fake_now: Any) -> None:
    app, _poly, _kalshi = _make_app()
    from datetime import datetime

    with patch("pfm.vol.event_vol_router.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = datetime
        with TestClient(app) as client:
            r = client.get(
                "/vol/event/calendar",
                params={"lookahead_days": 180, "kind": "fomc"},
            )

    assert r.status_code == 200
    payload = r.json()
    assert all(e["event_kind"] == "fomc" for e in payload)


# --------------------------------------------------------------------------- #
# 9 — per-event signal endpoint                                                #
# --------------------------------------------------------------------------- #


def test_get_event_signal_returns_200_for_known_event() -> None:
    app, _poly, _kalshi = _make_app()
    with TestClient(app) as client:
        r = client.get(f"/vol/event/{FOMC_JUN}/signal")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["event_id"] == FOMC_JUN
    assert payload["fetch_completeness"] == pytest.approx(1.0)
    assert payload["forecast"]["em_method"] == "entropy_proxy"
    assert payload["options_em_pct"] is None
    assert payload["em_gap_pct"] is None


# --------------------------------------------------------------------------- #
# 10 — unknown event_id → 404                                                  #
# --------------------------------------------------------------------------- #


def test_get_event_signal_returns_404_for_unknown() -> None:
    app, _poly, _kalshi = _make_app()
    with TestClient(app) as client:
        r = client.get("/vol/event/not-a-real-event-xyz/signal")
    assert r.status_code == 404
    assert "unknown event_id" in r.text


# --------------------------------------------------------------------------- #
# 11 — /signals returns a list                                                 #
# --------------------------------------------------------------------------- #


def test_get_signals_returns_list(fake_now: Any) -> None:
    app, _poly, _kalshi = _make_app()
    from datetime import datetime

    with patch("pfm.vol.event_vol_router.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = datetime
        with TestClient(app) as client:
            r = client.get("/vol/event/signals", params={"lookahead_days": 120})

    assert r.status_code == 200, r.text
    payload = r.json()
    assert isinstance(payload, list)
    # At minimum the June 2026 FOMC should be in the result.
    ids = {item["event_id"] for item in payload}
    assert FOMC_JUN in ids


# --------------------------------------------------------------------------- #
# 12 — /kinds endpoint                                                         #
# --------------------------------------------------------------------------- #


def test_get_kinds_returns_kind_list() -> None:
    app, _poly, _kalshi = _make_app()
    with TestClient(app) as client:
        r = client.get("/vol/event/kinds")
    assert r.status_code == 200
    payload = r.json()
    assert "supported" in payload and "present" in payload
    assert "fomc" in payload["supported"]
    assert "fomc" in payload["present"]
    assert set(payload["present"]).issubset(set(payload["supported"]))


# --------------------------------------------------------------------------- #
# 13 — lookahead validation: out-of-range → 422                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_value", [0, 500, -1])
def test_lookahead_days_validation_422(bad_value: int) -> None:
    app, _poly, _kalshi = _make_app()
    with TestClient(app) as client:
        r = client.get("/vol/event/calendar", params={"lookahead_days": bad_value})
    assert r.status_code == 422, r.text


# --------------------------------------------------------------------------- #
# 14 — feature-flag gates router mount in main.py                              #
# --------------------------------------------------------------------------- #


def test_feature_flag_gates_router_mount() -> None:
    """Compare which paths /vol/event/* exist on a fresh main.app per env."""
    import pfm.main as main_mod

    # Flag OFF — the canonical state of the test suite.
    with patch.dict(os.environ, {"PFM_VOL_EVENT_ENABLED": ""}, clear=False):
        os.environ.pop("PFM_VOL_EVENT_ENABLED", None)
        reloaded = importlib.reload(main_mod)
        paths_off = {r.path for r in reloaded.app.routes if "/vol/event" in r.path}

    # Flag ON — should expose the four router endpoints.
    with patch.dict(os.environ, {"PFM_VOL_EVENT_ENABLED": "1"}, clear=False):
        reloaded = importlib.reload(main_mod)
        paths_on = {r.path for r in reloaded.app.routes if "/vol/event" in r.path}

    # Restore the canonical OFF state to keep other tests deterministic.
    os.environ.pop("PFM_VOL_EVENT_ENABLED", None)
    importlib.reload(main_mod)

    assert paths_off == set(), f"Expected no /vol/event routes with flag OFF, got {paths_off}"
    expected = {
        "/vol/event/calendar",
        "/vol/event/kinds",
        "/vol/event/signals",
        "/vol/event/{event_id}/signal",
    }
    assert expected <= paths_on, f"Missing routes: {expected - paths_on}"
