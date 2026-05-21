"""W12-08 — tests for the 24h arb history surface.

The arb router exposes two endpoints that, together, provide the
"24h arb history" view consumed by the dashboard's Opportunities
and History sub-tabs:

* ``GET /strategies/arb/state`` — current snapshot with
  ``opportunities[]`` and ``scan_log[]``.
* ``GET /strategies/arb/detection-history`` — newest-first rolling
  list of arbs that have been detected.

These tests pin the wire shape, the source-priority order
(engine → fallback → empty), the TTL behaviour of the live-fallback
cache, the dedup behaviour when the scanner is called concurrently,
and the ordering guarantees the UI relies on.

All upstream is mocked: ``arb_scanner.top_arbs`` is patched with
synthetic dicts and ``_ARB_DIR`` is pointed at a tmp directory so
the real ``arbstuff/`` files never matter.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures — isolate _ARB_DIR + clear caches per test
# ---------------------------------------------------------------------------


@pytest.fixture
def arb_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the router at an isolated temp dir, disable live fallback.

    Live fallback is opt-in per test (some tests turn it back on) so the
    default state of every test is "engine missing, no scanner calls".
    """
    from pfm import strategies_arb_router as r

    monkeypatch.setattr(r, "_ARB_DIR", tmp_path)
    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", False)
    # Clear the in-memory detection ring so prior tests don't leak.
    r._DETECTION_HISTORY.clear()
    r._DETECTION_SEEN.clear()
    # Drop the fallback cache between tests.
    r._FALLBACK_CACHE["t"] = 0
    r._FALLBACK_CACHE["value"] = None
    # Also clear Redis-mirror reads (test app has no cache attr → already None).
    return tmp_path


@pytest.fixture
def client(arb_dir: Path) -> Iterator[TestClient]:
    from pfm.auth.dependencies import require_admin
    from pfm.strategies_arb_router import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: None
    with TestClient(app) as cli:
        yield cli


def _write_json(p: Path, data: object) -> None:
    p.write_text(json.dumps(data), encoding="utf-8")


def _sample_engine_state() -> dict:
    """A small but complete engine-shaped state with two opps."""
    return {
        "timestamp": "2026-05-16T12:00:00",
        "scan_count": 42,
        "cycle_time_s": 2.1,
        "balances": {"kalshi": 100.0, "polymarket": 200.0},
        "config": {
            "poll_interval": 8,
            "threshold": 0.94,
            "min_alert_profit": 1.0,
            "event_count": 2,
        },
        "bot_status": "running",
        "test_mode": True,
        "scan_mode": "WS",
        "candidates_count": 2,
        "opportunities": [
            {
                "name": "Older Arb",
                "type": "Buy K_YES+P_NO",
                "side": "yes",
                "profit_pct": 1.2,
                "volume": 100.0,
                "cost": 0.95,
                "kalshi_price": 0.55,
                "poly_price": 0.40,
                "kalshi_ticker": "KXOLD-01",
                "poly_slug": "older-slug",
                "poly_token_id": "0xold",
                "arb_key": "KXOLD-01__older-slug",
                "source": "engine",
                "spread": 0.15,
                "timestamp": "2026-05-16T10:00:00",
            },
            {
                "name": "Newer Arb",
                "type": "Buy K_NO+P_YES",
                "side": "no",
                "profit_pct": 3.4,
                "volume": 250.0,
                "cost": 0.92,
                "kalshi_price": 0.45,
                "poly_price": 0.47,
                "kalshi_ticker": "KXNEW-02",
                "poly_slug": "newer-slug",
                "poly_token_id": "0xnew",
                "arb_key": "KXNEW-02__newer-slug",
                "source": "engine",
                "spread": 0.02,
                "timestamp": "2026-05-16T11:55:00",
            },
        ],
        "scan_log": [],
    }


# ---------------------------------------------------------------------------
# 1. 200 with default — no engine file, no fallback → offline envelope
# ---------------------------------------------------------------------------


def test_state_200_with_default_params(client: TestClient) -> None:
    """A bare ``GET /strategies/arb/state`` must respond 200 with the well-known
    offline envelope (no query params required)."""
    r = client.get("/strategies/arb/state")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    # Required envelope keys every consumer relies on:
    for key in (
        "timestamp",
        "bot_status",
        "opportunities",
        "scan_log",
        "balances",
        "config",
        "candidates_count",
        "scan_mode",
    ):
        assert key in body, f"missing top-level key: {key}"
    assert body["bot_status"] == "offline"
    assert body["opportunities"] == []


# ---------------------------------------------------------------------------
# 2. List of arb opportunities with timestamps
# ---------------------------------------------------------------------------


def test_state_returns_list_of_arb_opportunities_with_timestamps(
    client: TestClient, arb_dir: Path
) -> None:
    """Engine-path /state must include the ``opportunities`` array, each entry
    a dict with a ``timestamp`` (for the 24h history view), and a top-level
    ``timestamp`` for the snapshot itself."""
    _write_json(arb_dir / "dashboard_state.json", _sample_engine_state())
    body = client.get("/strategies/arb/state").json()

    assert isinstance(body["opportunities"], list)
    assert len(body["opportunities"]) == 2
    # Snapshot wall-clock timestamp.
    assert isinstance(body["timestamp"], str)
    assert body["timestamp"].startswith("2026-")
    # Every opp must carry an arb_key (dedup key) + a per-opp timestamp.
    for opp in body["opportunities"]:
        assert "arb_key" in opp
        assert "timestamp" in opp
        assert isinstance(opp["timestamp"], str)


# ---------------------------------------------------------------------------
# 3. Filter by `since` — endpoint does not implement server-side `since`
#    filtering, so the param must be accepted (FastAPI ignores extras) and
#    the full history returned. Future work: add explicit filtering.
# ---------------------------------------------------------------------------


def test_detection_history_accepts_since_query_param(client: TestClient, arb_dir: Path) -> None:
    """The history endpoint does not yet implement server-side `since=1h`
    filtering — but the param must NOT 422 (FastAPI silently ignores
    extra query params on endpoints that don't declare them)."""
    _write_json(
        arb_dir / "arb_detection_history.json",
        [
            {"ts": "2026-05-15T10:00:00", "name": "Old", "arb_key": "a__1"},
            {"ts": "2026-05-16T11:00:00", "name": "Mid", "arb_key": "b__2"},
            {"ts": "2026-05-16T11:55:00", "name": "Fresh", "arb_key": "c__3"},
        ],
    )
    r = client.get("/strategies/arb/detection-history", params={"since": "1h"})
    assert r.status_code == 200
    body = r.json()
    # All three returned (no server-side filtering today).
    assert body["count"] == 3
    # Newest-first ordering preserved.
    assert body["items"][0]["name"] == "Fresh"


# ---------------------------------------------------------------------------
# 4. Empty result when no arbs (engine file present but empty list)
# ---------------------------------------------------------------------------


def test_state_empty_opportunities_when_engine_has_none(client: TestClient, arb_dir: Path) -> None:
    """Engine ran but found nothing — opps must be `[]` and `_source==engine`."""
    payload = _sample_engine_state()
    payload["opportunities"] = []
    payload["candidates_count"] = 0
    _write_json(arb_dir / "dashboard_state.json", payload)
    body = client.get("/strategies/arb/state").json()
    assert body["opportunities"] == []
    assert body["candidates_count"] == 0
    assert body["_source"] == "engine"


def test_detection_history_empty_when_no_engine_no_buffer(
    client: TestClient,
) -> None:
    """No engine file + empty in-process ring → ``count=0`` + ``items=[]``."""
    body = client.get("/strategies/arb/detection-history").json()
    assert body["count"] == 0
    assert body["items"] == []
    assert body["_source"] == "fallback_buffer"


# ---------------------------------------------------------------------------
# 5. TTL cache — second fallback call within TTL reuses the cached value
# ---------------------------------------------------------------------------


def test_state_fallback_cache_ttl_reuses_value(
    arb_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Within ``_FALLBACK_TTL`` seconds, the live-fallback scanner is called
    only once. Two GETs should hit ``top_arbs`` exactly once."""
    from pfm import arb_scanner
    from pfm import strategies_arb_router as r
    from pfm.auth.dependencies import require_admin
    from pfm.strategies_arb_router import router

    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", True)
    monkeypatch.setattr(r, "_FALLBACK_TTL", 60.0)
    r._FALLBACK_CACHE["t"] = 0
    r._FALLBACK_CACHE["value"] = None

    call_count = {"n": 0}

    def _spying_top_arbs(**kwargs: object) -> list[dict]:
        call_count["n"] += 1
        return [
            {
                "pm_slug": "x",
                "kalshi_slug": "K-X",
                "label": "X",
                "pm_price": 0.4,
                "kalshi_price": 0.3,
                "spread_pct": 12.5,
                "direction": "buy_pm_yes_kalshi_no",
                "tradeable_size_usd": 100.0,
                "half_life_minutes": 5.0,
                "last_seen_iso": "2026-05-16T12:00:00",
            }
        ]

    monkeypatch.setattr(arb_scanner, "top_arbs", _spying_top_arbs)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: None

    with TestClient(app) as cli:
        b1 = cli.get("/strategies/arb/state").json()
        b2 = cli.get("/strategies/arb/state").json()

    assert call_count["n"] == 1, "fallback cache failed to suppress 2nd call"
    assert b1["candidates_count"] == 1
    assert b2["candidates_count"] == 1
    # Timestamps may differ because the envelope wraps the cached opps with
    # a fresh ``timestamp`` — but the underlying opp arb_keys must match.
    assert b1["opportunities"][0]["arb_key"] == b2["opportunities"][0]["arb_key"]


# ---------------------------------------------------------------------------
# 6. Concurrent calls — detection-history must dedupe by arb_key
# ---------------------------------------------------------------------------


def test_concurrent_state_calls_dedup_detection_buffer(
    arb_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When many concurrent /state requests trigger the fallback (after the
    TTL is bypassed), the detection buffer must not grow unbounded — it's
    deduped by ``arb_key``. Drives the scanner concurrently and asserts
    only one entry lands in the ring."""
    from pfm import arb_scanner
    from pfm import strategies_arb_router as r
    from pfm.auth.dependencies import require_admin
    from pfm.strategies_arb_router import router

    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", True)
    # Use a tiny TTL so each parallel call re-enters the scanner. Must be
    # nonzero (the envelope's scan_count divides by TTL).
    monkeypatch.setattr(r, "_FALLBACK_TTL", 0.001)
    r._FALLBACK_CACHE["t"] = 0
    r._FALLBACK_CACHE["value"] = None
    r._DETECTION_HISTORY.clear()
    r._DETECTION_SEEN.clear()

    def _const_top_arbs(**kwargs: object) -> list[dict]:
        return [
            {
                "pm_slug": "dup-slug",
                "kalshi_slug": "KXDUP",
                "label": "Dup Arb",
                "pm_price": 0.4,
                "kalshi_price": 0.3,
                "spread_pct": 11.1,
                "direction": "buy_pm_yes_kalshi_no",
                "tradeable_size_usd": 50.0,
                "half_life_minutes": 5.0,
                "last_seen_iso": "2026-05-16T12:00:00",
            }
        ]

    monkeypatch.setattr(arb_scanner, "top_arbs", _const_top_arbs)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: None

    cli = TestClient(app)
    results: list[int] = []
    lock = threading.Lock()

    def _hit() -> None:
        resp = cli.get("/strategies/arb/state")
        with lock:
            results.append(resp.status_code)

    threads = [threading.Thread(target=_hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [200] * 8
    # Detection ring deduped → exactly one entry for the single arb_key.
    assert len(r._DETECTION_HISTORY) == 1
    assert r._DETECTION_HISTORY[0]["arb_key"] == "KXDUP__dup-slug"


# ---------------------------------------------------------------------------
# 7. Schema validation — each engine-state opp has price, spread, sources
# ---------------------------------------------------------------------------


def test_state_opportunity_schema_has_price_spread_sources(
    client: TestClient, arb_dir: Path
) -> None:
    """Each opportunity in the /state response must carry the fields the
    Bloomberg-style UI binds to: kalshi+poly price, derived spread,
    and a source attribution (kalshi_ticker / poly_slug)."""
    _write_json(arb_dir / "dashboard_state.json", _sample_engine_state())
    body = client.get("/strategies/arb/state").json()
    assert body["opportunities"], "engine path must surface opps"
    for opp in body["opportunities"]:
        # Prices on both venues
        assert isinstance(opp["kalshi_price"], (int, float))
        assert isinstance(opp["poly_price"], (int, float))
        # Spread (or cost — both are present in the engine shape)
        assert "spread" in opp or "cost" in opp
        # Source attribution — these identify the underlying venues.
        assert opp.get("kalshi_ticker"), "missing kalshi source ticker"
        assert opp.get("poly_slug") or opp.get("poly_token_id"), (
            "missing polymarket source identifier"
        )
        # Profit signal must be present.
        assert "profit_pct" in opp


# ---------------------------------------------------------------------------
# 8. Unknown filter — FastAPI ignores undeclared query params by design
#    (no validation declared on these endpoints), so we assert the
#    permissive contract: extra ``filter=...`` => 200, not 422.
# ---------------------------------------------------------------------------


def test_state_unknown_filter_param_does_not_422(client: TestClient) -> None:
    """Extra params are tolerated. (Documenting current contract so a future
    refactor that adds Pydantic Query() validators must update this test.)"""
    r = client.get("/strategies/arb/state", params={"sort": "nope", "x": "y"})
    assert r.status_code == 200
    body = r.json()
    assert "opportunities" in body


def test_detection_history_unknown_param_does_not_422(client: TestClient) -> None:
    r = client.get("/strategies/arb/detection-history", params={"foo": "bar"})
    assert r.status_code == 200
    assert "items" in r.json()


# ---------------------------------------------------------------------------
# 9. Sort order — detection history is newest-first
# ---------------------------------------------------------------------------


def test_detection_history_engine_file_is_newest_first(client: TestClient, arb_dir: Path) -> None:
    """The engine writes the file oldest-first (append log); the router
    reverses it so the dashboard sees newest at index 0."""
    _write_json(
        arb_dir / "arb_detection_history.json",
        [
            {"first_seen_iso": "2026-05-16T08:00:00", "name": "A", "arb_key": "a"},
            {"first_seen_iso": "2026-05-16T09:00:00", "name": "B", "arb_key": "b"},
            {"first_seen_iso": "2026-05-16T10:00:00", "name": "C", "arb_key": "c"},
        ],
    )
    body = client.get("/strategies/arb/detection-history").json()
    assert body["count"] == 3
    names = [item["name"] for item in body["items"]]
    assert names == ["C", "B", "A"]
    assert body["_source"] == "engine"


def test_detection_history_in_process_buffer_sorted_by_first_seen_unix(
    arb_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an engine file, the in-process buffer is sorted by
    ``first_seen_unix`` descending."""
    from pfm import strategies_arb_router as r
    from pfm.auth.dependencies import require_admin
    from pfm.strategies_arb_router import router

    r._DETECTION_HISTORY.clear()
    r._DETECTION_SEEN.clear()
    r._DETECTION_HISTORY.extend(
        [
            {"first_seen_unix": 1000.0, "name": "old", "arb_key": "k1"},
            {"first_seen_unix": 3000.0, "name": "new", "arb_key": "k2"},
            {"first_seen_unix": 2000.0, "name": "mid", "arb_key": "k3"},
        ]
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: None
    with TestClient(app) as cli:
        body = cli.get("/strategies/arb/detection-history").json()

    assert body["_source"] == "fallback_buffer"
    names = [item["name"] for item in body["items"]]
    assert names == ["new", "mid", "old"]


# ---------------------------------------------------------------------------
# 10. Pagination — endpoint does not support it today. Document the
#     current contract: extra ``limit``/``offset`` params are silently
#     ignored (FastAPI undeclared-param semantics) and the full list is
#     returned. A future explicit implementation should bump and update
#     this test.
# ---------------------------------------------------------------------------


def test_detection_history_pagination_not_supported_returns_full_list(
    client: TestClient, arb_dir: Path
) -> None:
    """``limit`` + ``offset`` are not (yet) declared query params on
    detection-history; they must be silently ignored and the full list
    must be returned. The dashboard does client-side pagination today."""
    _write_json(
        arb_dir / "arb_detection_history.json",
        [
            {"first_seen_iso": f"2026-05-16T{i:02d}:00:00", "name": f"E{i}", "arb_key": f"k{i}"}
            for i in range(10)
        ],
    )
    body = client.get(
        "/strategies/arb/detection-history",
        params={"limit": 3, "offset": 2},
    ).json()
    # Full list returned — pagination flags ignored.
    assert body["count"] == 10
    assert len(body["items"]) == 10
    # Still newest-first.
    assert body["items"][0]["name"] == "E9"


# ---------------------------------------------------------------------------
# Bonus: provenance discriminator must distinguish engine vs fallback
# vs empty. The dashboard footer renders this to the user as
# "live engine" / "live scanner" / "engine offline".
# ---------------------------------------------------------------------------


def test_state_source_discriminator_offline_engine_fallback(
    arb_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pfm import arb_scanner
    from pfm import strategies_arb_router as r
    from pfm.auth.dependencies import require_admin
    from pfm.strategies_arb_router import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: None

    # 1. No file, no fallback → empty.
    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", False)
    r._FALLBACK_CACHE["t"] = 0
    r._FALLBACK_CACHE["value"] = None
    with TestClient(app) as cli:
        body = cli.get("/strategies/arb/state").json()
    assert body["_source"] == "empty"

    # 2. Engine file present → engine.
    _write_json(arb_dir / "dashboard_state.json", _sample_engine_state())
    with TestClient(app) as cli:
        body = cli.get("/strategies/arb/state").json()
    assert body["_source"] == "engine"

    # 3. Engine file gone, fallback on, scanner mocked → live_fallback.
    (arb_dir / "dashboard_state.json").unlink()
    monkeypatch.setattr(r, "_LIVE_FALLBACK_ENABLED", True)
    r._FALLBACK_CACHE["t"] = 0
    r._FALLBACK_CACHE["value"] = None
    monkeypatch.setattr(arb_scanner, "top_arbs", lambda **_: [])
    with TestClient(app) as cli:
        body = cli.get("/strategies/arb/state").json()
    assert body["_source"] == "live_fallback"
