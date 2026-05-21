"""Tests for `pfm.strategies_arb_router` — live arb dashboard endpoints.

The router has two operating modes:

1. **Engine path** — when ``arbstuff/dashboard_state.json`` exists and is fresh,
   it's read verbatim and surfaced through ``/state`` (and the SSE stream).
2. **Fallback path** — when the state file is missing or stale, the router
   synthesises a state object from :func:`pfm.arb_scanner.top_arbs`. This is
   on by default but disabled here so tests don't hit the real network.

Strategy:
- Point ``_ARB_DIR`` at a temp directory per test via monkeypatch.
- Disable ``_LIVE_FALLBACK_ENABLED`` so we get deterministic empty-state
  responses without scanner calls.
- Write synthetic JSON fixtures and assert the endpoint shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def arb_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the router at an isolated temp dir + disable live fallback."""
    from pfm import strategies_arb_router as r

    monkeypatch.setattr(r, "_ARB_DIR", tmp_path)
    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", False)
    # Drop the fallback cache between tests so prior runs don't leak.
    r._FALLBACK_CACHE["t"] = 0
    r._FALLBACK_CACHE["value"] = None
    return tmp_path


@pytest.fixture
def client(arb_dir: Path) -> TestClient:
    from pfm.strategies_arb_router import router

    app = FastAPI()
    app.include_router(router)
    # Tests run with auth fully disabled — the production app gates blacklist
    # / settings behind ``require_admin``, but in unit tests we want to
    # exercise the underlying logic without setting an admin token. Override
    # the dependency to a no-op so the gate doesn't 403.
    from pfm.auth.dependencies import require_admin

    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app)


def _write(p: Path, data: object) -> None:
    p.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# /state — engine path + offline fallback
# ---------------------------------------------------------------------------


def test_state_returns_offline_envelope_when_engine_missing(client: TestClient) -> None:
    """No ``dashboard_state.json`` + fallback disabled → ``bot_status='offline'``."""
    r = client.get("/strategies/arb/state")
    assert r.status_code == 200
    body = r.json()
    assert body["bot_status"] == "offline"
    assert body["opportunities"] == []
    assert body["scan_log"] == []
    assert "hint" in body
    assert body["_source"] == "empty"


def test_state_passes_through_engine_file(client: TestClient, arb_dir: Path) -> None:
    """When the engine wrote a fresh state file, /state must surface every field."""
    payload = {
        "timestamp": "2026-05-14T17:42:11",
        "scan_count": 873,
        "cycle_time_s": 3.4,
        "balances": {"kalshi": 124.55, "polymarket": 482.12},
        "config": {
            "poll_interval": 8,
            "threshold": 0.94,
            "min_alert_profit": 1.0,
            "event_count": 412,
        },
        "bot_status": "running",
        "test_mode": True,
        "scan_mode": "WS",
        "candidates_count": 7,
        "opportunities": [
            {
                "name": "TestEvent",
                "type": "Buy K_YES+P_NO",
                "profit_pct": 2.41,
                "volume": 312.0,
                "cost": 0.9351,
                "kalshi_price": 0.61,
                "poly_price": 0.3251,
                "kalshi_ticker": "KXTEST-01",
                "poly_token_id": "0xabc",
                "arb_key": "KXTEST-01_yes_0xabc",
                "source": "main",
            }
        ],
        "scan_log": [
            {
                "t": "17:42:09",
                "event": "TestEvent",
                "outcome": "yes",
                "k_yes": 0.61,
                "p_no": 0.32,
                "pass_yes": True,
                "pass_no": False,
            }
        ],
    }
    _write(arb_dir / "dashboard_state.json", payload)
    r = client.get("/strategies/arb/state")
    assert r.status_code == 200
    body = r.json()
    assert body["bot_status"] == "running"
    assert body["scan_count"] == 873
    assert body["balances"]["kalshi"] == 124.55
    assert len(body["opportunities"]) == 1
    assert body["opportunities"][0]["arb_key"] == "KXTEST-01_yes_0xabc"
    assert body["_source"] == "engine"


def test_state_fills_missing_keys_from_partial_engine_state(
    client: TestClient, arb_dir: Path
) -> None:
    """Engine writes opportunities but not scan_log/balances — endpoint must fill defaults."""
    _write(arb_dir / "dashboard_state.json", {"opportunities": []})
    body = client.get("/strategies/arb/state").json()
    assert body["scan_log"] == []
    assert body["balances"] == {"kalshi": 0.0, "polymarket": 0.0}
    assert body["config"]["threshold"] == 0.94


def test_state_sets_browser_cache_control(client: TestClient) -> None:
    """``/state`` returns ``Cache-Control: public, max-age=2`` so the browser
    revalidates the (135 KB) body only every 2 s. The SSE stream still pushes
    every tick — ``/state`` is the initial-paint + manual-refresh path."""
    r = client.get("/strategies/arb/state")
    assert r.status_code == 200
    cc = r.headers.get("cache-control", "")
    assert "public" in cc
    assert "max-age=2" in cc


# ---------------------------------------------------------------------------
# /pnl, /detection-history, /config-stats
# ---------------------------------------------------------------------------


def test_pnl_returns_empty_when_file_missing(client: TestClient) -> None:
    body = client.get("/strategies/arb/pnl").json()
    # Engine file missing + in-process detection buffer empty → no trades.
    # We tolerate the extra ``_source`` discriminator the fallback adds so the
    # UI can show "synthetic" vs "engine" provenance.
    assert body["trades"] == []
    assert body["total_pnl"] == 0
    assert body["count"] == 0


def test_pnl_aggregates_total(client: TestClient, arb_dir: Path) -> None:
    _write(
        arb_dir / "arb_pnl_log.json",
        [
            {
                "timestamp": "2026-05-14T15:01:33",
                "event": "A",
                "outcome": "yes",
                "side": "yes",
                "volume": 10,
                "k_price": 0.5,
                "p_price": 0.45,
                "total_cost": 9.50,
                "guaranteed_profit": 0.50,
            },
            {
                "timestamp": "2026-05-14T15:05:00",
                "event": "B",
                "outcome": "no",
                "side": "no",
                "volume": 20,
                "k_price": 0.4,
                "p_price": 0.55,
                "total_cost": 19.00,
                "guaranteed_profit": 1.00,
            },
        ],
    )
    body = client.get("/strategies/arb/pnl").json()
    assert body["count"] == 2
    assert body["total_pnl"] == 1.5
    assert len(body["trades"]) == 2


def test_detection_history_reverses_order(client: TestClient, arb_dir: Path) -> None:
    _write(
        arb_dir / "arb_detection_history.json",
        [
            {"ts": "2026-05-14T10:00:00", "name": "First"},
            {"ts": "2026-05-14T10:01:00", "name": "Second"},
            {"ts": "2026-05-14T10:02:00", "name": "Third"},
        ],
    )
    body = client.get("/strategies/arb/detection-history").json()
    assert body["count"] == 3
    # Newest first.
    assert body["items"][0]["name"] == "Third"
    assert body["items"][-1]["name"] == "First"


def test_config_stats_counts_mappings(client: TestClient, arb_dir: Path) -> None:
    _write(
        arb_dir / "markets_config_reviewed.json",
        {
            "events": [
                {"kalshi_ticker": "A", "mapping": {"x": "y"}},
                {"kalshi_ticker": "B", "mapping": {}},  # not mapped
            ]
        },
    )
    _write(
        arb_dir / "markets_config.json", {"events": [{"kalshi_ticker": "C", "mapping": {"a": "b"}}]}
    )
    body = client.get("/strategies/arb/config-stats").json()
    assert body["reviewed"] == {"total": 2, "mapped": 1}
    assert body["main"] == {"total": 1, "mapped": 1}
    assert body["combined_mapped"] == 2


# ---------------------------------------------------------------------------
# /config-events — merged universe
# ---------------------------------------------------------------------------


def test_config_events_dedupes_by_ticker_with_priority(client: TestClient, arb_dir: Path) -> None:
    _write(
        arb_dir / "markets_config_reviewed.json",
        {
            "events": [
                {
                    "name": "Reviewed-A",
                    "kalshi_ticker": "A",
                    "poly_slug": "a-poly",
                    "mapping": {"x": "y"},
                }
            ]
        },
    )
    _write(
        arb_dir / "markets_config.json",
        {
            "events": [
                # Same ticker — must be dropped (reviewed wins).
                {
                    "name": "Main-A",
                    "kalshi_ticker": "A",
                    "poly_slug": "a-poly",
                    "mapping": {"x": "y"},
                },
                {
                    "name": "Main-B",
                    "kalshi_ticker": "B",
                    "poly_slug": "b-poly",
                    "mapping": {"x": "y"},
                },
                # Empty mapping — must be excluded.
                {"name": "Main-C", "kalshi_ticker": "C", "poly_slug": "c-poly", "mapping": {}},
            ]
        },
    )
    body = client.get("/strategies/arb/config-events").json()
    tickers = {e["kalshi_ticker"]: e["source"] for e in body["events"]}
    assert tickers == {"A": "reviewed", "B": "main"}


# ---------------------------------------------------------------------------
# /politics-events
# ---------------------------------------------------------------------------


def test_politics_events_parses_state_and_office(client: TestClient, arb_dir: Path) -> None:
    _write(
        arb_dir / "markets_config_politics.json",
        {
            "events": [
                {
                    "name": "Texas District 33 HOUSE (primary) [D] 2026",
                    "kalshi_ticker": "K1",
                    "poly_slug": "p1",
                    "mapping": {"x": "y"},
                },
                {
                    "name": "California GOV [R] 2026",
                    "kalshi_ticker": "K2",
                    "poly_slug": "p2",
                    "mapping": {"x": "y"},
                },
            ]
        },
    )
    body = client.get("/strategies/arb/politics-events").json()
    assert body["total"] == 2
    tx = next(e for e in body["events"] if e["kalshi_ticker"] == "K1")
    assert tx["state"] == "TX"
    assert tx["office"] == "HOUSE"
    assert tx["race_type"] == "primary"
    assert tx["party"] == "D"
    assert tx["district"] == 33
    assert tx["year"] == 2026
    assert body["stats"]["by_state"]["TX"] == 1
    assert body["stats"]["by_office"]["GOV"] == 1


# ---------------------------------------------------------------------------
# /blacklist (POST + DELETE)
# ---------------------------------------------------------------------------


def test_blacklist_add_is_idempotent(client: TestClient, arb_dir: Path) -> None:
    r1 = client.post("/strategies/arb/blacklist", json={"arb_key": "KXFOO_yes_0xabc"})
    assert r1.status_code == 200
    assert r1.json()["added"] is True
    r2 = client.post("/strategies/arb/blacklist", json={"arb_key": "KXFOO_yes_0xabc"})
    assert r2.json()["added"] is False
    # Two distinct keys.
    r3 = client.post("/strategies/arb/blacklist", json={"arb_key": "KXBAR_no_0xdef"})
    assert r3.json()["blacklisted"] == 2
    # File on disk has both.
    data = json.loads((arb_dir / "arb_blacklist.json").read_text())
    assert set(data) == {"KXFOO_yes_0xabc", "KXBAR_no_0xdef"}


def test_blacklist_delete_clears(client: TestClient, arb_dir: Path) -> None:
    _write(arb_dir / "arb_blacklist.json", ["one", "two", "three"])
    r = client.delete("/strategies/arb/blacklist")
    assert r.status_code == 200
    assert json.loads((arb_dir / "arb_blacklist.json").read_text()) == []


# ---------------------------------------------------------------------------
# /settings (POST)
# ---------------------------------------------------------------------------


def test_settings_merges_keys(client: TestClient, arb_dir: Path) -> None:
    _write(arb_dir / "dashboard_control.json", {"email_enabled": False, "threshold": 0.94})
    r = client.post("/strategies/arb/settings", json={"threshold": 0.96, "scan_mode": "WS"})
    assert r.status_code == 200
    on_disk = json.loads((arb_dir / "dashboard_control.json").read_text())
    assert on_disk["threshold"] == 0.96
    assert on_disk["scan_mode"] == "WS"
    # Untouched key preserved.
    assert on_disk["email_enabled"] is False


def test_settings_rejects_invalid_scan_mode(client: TestClient) -> None:
    r = client.post("/strategies/arb/settings", json={"scan_mode": "TURBO"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /markets — existing endpoint, sanity-check
# ---------------------------------------------------------------------------


def test_markets_paginates_and_dedupes(client: TestClient, arb_dir: Path) -> None:
    _write(
        arb_dir / "markets_config_reviewed.json",
        {
            "events": [
                {
                    "name": "Reviewed-A",
                    "kalshi_ticker": "A",
                    "poly_slug": "a-poly",
                    "mapping": {"x": "y", "z": "w"},
                },
            ]
        },
    )
    _write(
        arb_dir / "markets_config.json",
        {
            "events": [
                {
                    "name": "Main-A",
                    "kalshi_ticker": "A",
                    "poly_slug": "a-poly",
                    "mapping": {"x": "y"},
                },  # dedupes
                {
                    "name": "Main-B",
                    "kalshi_ticker": "B",
                    "poly_slug": "b-poly",
                    "mapping": {"x": "y"},
                },
            ]
        },
    )
    r = client.get("/strategies/arb/markets?limit=10&source=all")
    body = r.json()
    assert body["total"] == 2
    tickers = {e["kalshi_ticker"] for e in body["events"]}
    assert tickers == {"A", "B"}
    # n_outcomes for the deduped 'A' must come from the *first* (reviewed) source.
    a = next(e for e in body["events"] if e["kalshi_ticker"] == "A")
    assert a["n_outcomes"] == 2


def test_markets_rejects_unknown_source(client: TestClient) -> None:
    r = client.get("/strategies/arb/markets?source=notarealsource")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /stream — SSE smoke test
# ---------------------------------------------------------------------------


def test_stream_emits_initial_data_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSE generator must emit a well-formed ``data:`` frame on first tick.

    We don't go through ``TestClient.stream`` here — Starlette's test client
    buffers the response until the generator exits, so a long-lived SSE
    handler would block the test indefinitely. Instead drive the async
    generator directly: build a fake ``Request`` whose ``is_disconnected``
    returns ``True`` after one frame, then collect the bytes.
    """
    import asyncio

    from pfm import strategies_arb_router as r

    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", False)
    monkeypatch.setattr(r, "_STREAM_TICK_SECONDS", 0.01)

    class FakeRequest:
        def __init__(self) -> None:
            self._first = True

        async def is_disconnected(self) -> bool:
            # Allow exactly one frame, then signal disconnect.
            if self._first:
                self._first = False
                return False
            return True

    async def _collect() -> list[bytes]:
        out: list[bytes] = []
        async for chunk in r._state_event_generator(FakeRequest(), 0.01):
            out.append(chunk)
            # First chunk is the keep-alive comment; we need the data frame
            # that follows it to validate the payload.
            if len(out) >= 2:
                break
        return out

    frames = asyncio.run(_collect())
    assert len(frames) >= 2
    # First frame: SSE comment that fires onopen immediately.
    assert frames[0].startswith(b":")
    # Second frame: the data payload.
    body = frames[1].decode("utf-8")
    assert body.startswith("data: ")
    assert body.endswith("\n\n")
    payload = body[len("data: ") :].rstrip()
    parsed = json.loads(payload)
    assert "bot_status" in parsed
    assert "opportunities" in parsed


# ---------------------------------------------------------------------------
# Live fallback (smoke — relies on arb_scanner.top_arbs)
# ---------------------------------------------------------------------------


def test_fallback_synthesises_state_when_enabled(
    arb_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``_LIVE_FALLBACK_ENABLED`` is on, /state must include opps from arb_scanner.

    We monkeypatch ``top_arbs`` to a known list — no network in this test.
    """
    from pfm import strategies_arb_router as r

    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", True)
    # Bust the cache + replace top_arbs with a tiny synthetic.
    r._FALLBACK_CACHE["t"] = 0
    r._FALLBACK_CACHE["value"] = None

    fake_arbs = [
        {
            "pm_slug": "fake-event",
            "kalshi_slug": "KXFAKE-01",
            "label": "Synthetic Test Arb",
            "pm_price": 0.40,
            "kalshi_price": 0.30,
            "spread_pct": 30.0,
            "direction": "buy_pm_yes_kalshi_no",
            "tradeable_size_usd": 250.0,
            "half_life_minutes": 12.5,
            "last_seen_iso": "2026-05-14T17:00:00",
            "confirmed": True,
            "confirmation_window_min": 5,
        }
    ]

    def _fake_top_arbs(**kwargs: object) -> list[dict]:
        return fake_arbs

    # Replace the actual top_arbs function inside arb_scanner.
    from pfm import arb_scanner

    monkeypatch.setattr(arb_scanner, "top_arbs", _fake_top_arbs)

    app = FastAPI()
    from pfm.strategies_arb_router import router

    app.include_router(router)
    cli = TestClient(app)
    body = cli.get("/strategies/arb/state").json()
    assert body["_source"] == "live_fallback"
    assert body["candidates_count"] == 1
    opp = body["opportunities"][0]
    assert opp["name"] == "Synthetic Test Arb"
    assert opp["profit_pct"] == 30.0
    assert opp["kalshi_price"] == 0.30
    assert opp["poly_price"] == 0.40
