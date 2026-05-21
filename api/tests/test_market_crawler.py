"""No-network tests for :mod:`pfm.arb.market_crawler`.

All HTTP is mocked by monkeypatching the module-internal ``_get`` seam (and,
for the backoff path, a fake session). No real Polymarket / Kalshi calls.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from pfm.arb import market_crawler as mc

# ---------------------------------------------------------------------------
# Helpers / fakes.
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeResp:
    """Minimal requests-like response for the backoff/timeout path tests."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self) -> Any:
        return self._json


class FakeSession:
    """Session whose ``.get`` replays a queue of :class:`FakeResp`."""

    def __init__(self, responses: list[FakeResp]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params: Any = None, timeout: float | None = None) -> FakeResp:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Kalshi cursor pagination.
# ---------------------------------------------------------------------------


def test_kalshi_cursor_pagination_collects_all_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    base = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    pages = {
        None: {
            "markets": [{"ticker": "A", "open_time": _iso(base)}],
            "cursor": "C1",
        },
        "C1": {
            "markets": [{"ticker": "B", "open_time": _iso(base + timedelta(hours=1))}],
            "cursor": "C2",
        },
        "C2": {"markets": [{"ticker": "C", "open_time": _iso(base)}], "cursor": ""},
    }

    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        return pages[params.get("cursor")]

    monkeypatch.setattr(mc, "_get", fake_get)

    page = mc.crawl_kalshi_markets(max_pages=10, pace_s=0.0)

    assert page.n_pages == 3
    assert page.done is True
    assert {m["ticker"] for m in page.markets} == {"A", "B", "C"}
    # Newest-first by open_time: B (base+1h) leads.
    assert page.markets[0]["ticker"] == "B"


def test_kalshi_max_pages_limits_step(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        # Never-ending feed: always returns a fresh cursor.
        return {"markets": [{"ticker": "X", "open_time": "2026-05-20T00:00:00Z"}], "cursor": "next"}

    monkeypatch.setattr(mc, "_get", fake_get)

    page = mc.crawl_kalshi_markets(max_pages=2, pace_s=0.0)

    assert page.n_pages == 2
    assert page.done is False
    assert page.next_cursor == "next"


def test_kalshi_empty_markets_marks_done(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mc,
        "_get",
        lambda *a, **k: {"markets": [], "cursor": "still-here"},
    )
    page = mc.crawl_kalshi_markets(max_pages=5, pace_s=0.0)
    assert page.done is True
    assert page.markets == []


# ---------------------------------------------------------------------------
# Kalshi freshness.
# ---------------------------------------------------------------------------


def test_new_kalshi_markets_filters_by_open_time(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    recent = {"ticker": "FRESH", "open_time": _iso(now - timedelta(hours=2))}
    stale = {"ticker": "OLD", "open_time": _iso(now - timedelta(hours=48))}
    # created_time fallback when open_time missing.
    fallback = {"ticker": "FB", "created_time": _iso(now - timedelta(hours=1))}

    monkeypatch.setattr(
        mc,
        "_get",
        lambda *a, **k: {"markets": [stale, recent, fallback], "cursor": ""},
    )

    fresh = mc.new_kalshi_markets(within_hours=24.0, now=now)
    tickers = [m["ticker"] for m in fresh]
    assert "OLD" not in tickers
    assert set(tickers) == {"FRESH", "FB"}
    # Newest-first: FB (1h) before FRESH (2h).
    assert tickers[0] == "FB"


# ---------------------------------------------------------------------------
# Kalshi events crawl (real titles + nested markets).
# ---------------------------------------------------------------------------


def test_kalshi_events_cursor_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    base = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    pages = {
        None: {
            "events": [
                {
                    "event_ticker": "E1",
                    "title": "Will Trump win 2028?",
                    "markets": [{"ticker": "E1-Y", "open_time": _iso(base)}],
                }
            ],
            "cursor": "C1",
        },
        "C1": {
            "events": [
                {
                    "event_ticker": "E2",
                    "title": "Fed rate decision June 2026",
                    "markets": [{"ticker": "E2-Y", "open_time": _iso(base)}],
                }
            ],
            "cursor": "",
        },
    }

    captured: list[dict[str, Any]] = []

    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        captured.append(params)
        return pages[params.get("cursor")]

    monkeypatch.setattr(mc, "_get", fake_get)

    page = mc.crawl_kalshi_events(max_pages=10, pace_s=0.0)

    assert page.n_pages == 2
    assert page.done is True
    assert {e["event_ticker"] for e in page.events} == {"E1", "E2"}
    # Hits the events endpoint with nested markets requested.
    assert captured[0]["with_nested_markets"] == "true"
    assert captured[0]["status"] == "open"


def test_kalshi_events_max_pages_limits_step(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        return {"events": [{"event_ticker": "X", "title": "T", "markets": []}], "cursor": "next"}

    monkeypatch.setattr(mc, "_get", fake_get)

    page = mc.crawl_kalshi_events(max_pages=2, pace_s=0.0)
    assert page.n_pages == 2
    assert page.done is False
    assert page.next_cursor == "next"


def test_new_kalshi_events_derives_freshness_from_nested_markets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    # Event freshness = MAX open_time across nested markets.
    fresh_evt = {
        "event_ticker": "FRESH",
        "title": "Bitcoin above $200k by Dec 31 2026",
        "markets": [
            {"ticker": "F-OLD", "open_time": _iso(now - timedelta(hours=72))},
            {"ticker": "F-NEW", "open_time": _iso(now - timedelta(hours=2))},
        ],
    }
    stale_evt = {
        "event_ticker": "STALE",
        "title": "Senate control 2026",
        "markets": [{"ticker": "S-Y", "open_time": _iso(now - timedelta(hours=96))}],
    }
    ephemeral_evt = {
        "event_ticker": "SOL",
        "title": "Solana Up or Down - May 22 3:15PM-3:30PM ET",
        "markets": [{"ticker": "SOL-Y", "open_time": _iso(now - timedelta(hours=1))}],
    }

    monkeypatch.setattr(
        mc,
        "_get",
        lambda *a, **k: {"events": [stale_evt, fresh_evt, ephemeral_evt], "cursor": ""},
    )

    fresh = mc.new_kalshi_events(within_hours=24.0, now=now)
    tickers = [e["event_ticker"] for e in fresh]
    # Stale dropped (oldest market 96h ago); ephemeral Solana up/down dropped.
    assert tickers == ["FRESH"]


# ---------------------------------------------------------------------------
# Ephemeral-series filter.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Solana Up or Down - May 22 3:15PM-3:30PM ET", True),
        ("Bitcoin Up/Down 15m", True),
        ("ETH up or down", True),
        ("Highest temperature in New York today", True),
        ("High temp in Chicago", True),
        ("Will it rain in Seattle tomorrow", True),
        ("BTC price window 3:00pm", True),
        ("Ethereum above ___ on May 21, 4PM ET?", True),
        ("Bitcoin above ___ on May 21, 4PM ET?", True),
        ("Will Trump win 2028?", False),
        ("Bitcoin above $200k by Dec 31 2026", False),
        ("Fed rate decision June 2026", False),
        ("Lakers to win the 2026 NBA Finals", False),
        ("", False),
    ],
)
def test_is_ephemeral_market_table(text: str, expected: bool) -> None:
    assert mc.is_ephemeral_market(text) is expected


# ---------------------------------------------------------------------------
# Polymarket offset pagination.
# ---------------------------------------------------------------------------


def test_poly_offset_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    full = [{"slug": f"e{i}", "startDate": "2026-05-20T00:00:00Z"} for i in range(100)]
    short = [{"slug": "tail", "startDate": "2026-05-19T00:00:00Z"}]

    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        return full if params["offset"] == 0 else short

    monkeypatch.setattr(mc, "_get", fake_get)

    page = mc.crawl_poly_events(offset=0, max_pages=5, pace_s=0.0)

    assert page.n_pages == 2
    assert page.done is True  # short page ends the sweep
    assert len(page.events) == 101
    assert page.next_offset == 200


def test_poly_422_at_cap_returns_done_no_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        raise mc.CrawlHTTPError(422, "offset exceeds maximum")

    monkeypatch.setattr(mc, "_get", fake_get)

    # Below the cap so the request is actually attempted -> 422 -> graceful done.
    page = mc.crawl_poly_events(offset=10000, max_pages=5, pace_s=0.0)
    assert page.done is True
    assert page.events == []


def test_poly_offset_at_or_above_cap_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    def fake_get(*a: Any, **k: Any):
        called["n"] += 1
        return []

    monkeypatch.setattr(mc, "_get", fake_get)

    page = mc.crawl_poly_events(offset=mc.POLY_OFFSET_CAP, max_pages=5, pace_s=0.0)
    assert page.done is True
    assert called["n"] == 0  # never hit the network past the cap


def test_poly_non_422_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*a: Any, **k: Any):
        raise mc.CrawlHTTPError(500, "boom")

    monkeypatch.setattr(mc, "_get", fake_get)

    with pytest.raises(mc.CrawlHTTPError):
        mc.crawl_poly_events(offset=0, max_pages=2, pace_s=0.0)


# ---------------------------------------------------------------------------
# Polymarket volume-sorted (liquid) crawl.
# ---------------------------------------------------------------------------


def test_poly_by_volume_pagination_and_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two full pages then a short page -> done. Verify the volume-sort params.
    full_a = [{"slug": f"a{i}", "volumeNum": 1e6 - i} for i in range(100)]
    full_b = [{"slug": f"b{i}", "volumeNum": 5e5 - i} for i in range(100)]
    short = [{"slug": "tail", "volumeNum": 1.0}]
    seen_params: list[dict[str, Any]] = []

    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        assert url == mc.POLY_MARKETS_URL
        seen_params.append(dict(params))
        off = params["offset"]
        return full_a if off == 0 else (full_b if off == 100 else short)

    monkeypatch.setattr(mc, "_get", fake_get)

    page = mc.crawl_poly_by_volume(offset=0, max_pages=5, pace_s=0.0)

    assert page.n_pages == 3
    assert page.done is True  # short page ends the sweep
    assert len(page.events) == 201
    assert page.next_offset == 300
    # Volume-descending order param is what makes this the substantive feed.
    assert seen_params[0]["order"] == "volumeNum"
    assert seen_params[0]["ascending"] == "false"
    assert seen_params[0]["closed"] == "false"
    assert seen_params[0]["active"] == "true"
    # First item is highest-volume (server-sorted, returned verbatim).
    assert page.events[0]["slug"] == "a0"


def test_poly_by_volume_422_at_cap_returns_done_no_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        raise mc.CrawlHTTPError(422, "offset exceeds maximum")

    monkeypatch.setattr(mc, "_get", fake_get)

    page = mc.crawl_poly_by_volume(offset=10000, max_pages=5, pace_s=0.0)
    assert page.done is True
    assert page.events == []


def test_poly_by_volume_does_not_filter_liquid_markets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The volume crawl is RAW: it must NOT drop substantive (even sports-final
    # or vaguely templated-looking) liquid markets. The pipeline deliberately
    # skips the ephemeral filter on this feed; the crawler never applies it.
    page0 = [
        {"slug": "trump-2028", "question": "Will Trump win 2028?", "volumeNum": 9e7},
        {
            "slug": "iran-regime",
            "question": "Will the Iranian regime fall by June 30?",
            "volumeNum": 4e7,
        },
        # A title that the ephemeral regex WOULD flag ("up or down") but is a
        # high-volume market — it must still survive the volume crawl untouched.
        {"slug": "btc-eoy-band", "question": "Bitcoin up or down by year end?", "volumeNum": 2e7},
    ]

    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        return page0 if params["offset"] == 0 else []

    monkeypatch.setattr(mc, "_get", fake_get)

    page = mc.crawl_poly_by_volume(offset=0, max_pages=2, pace_s=0.0)
    slugs = {m["slug"] for m in page.events}
    # ALL three retained — nothing filtered, even the "up or down" title.
    assert slugs == {"trump-2028", "iran-regime", "btc-eoy-band"}


# ---------------------------------------------------------------------------
# Polymarket freshness early-stop.
# ---------------------------------------------------------------------------


def test_new_poly_events_early_stops_on_old_startdate(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    page0 = [
        {"slug": "new1", "startDate": _iso(now - timedelta(hours=1))},
        {"slug": "new2", "startDate": _iso(now - timedelta(hours=5))},
        {"slug": "old1", "startDate": _iso(now - timedelta(hours=48))},
        {"slug": "new3", "startDate": _iso(now - timedelta(hours=2))},  # after the old one
    ]
    calls = {"n": 0}

    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        calls["n"] += 1
        return page0 if params["offset"] == 0 else []

    monkeypatch.setattr(mc, "_get", fake_get)

    fresh = mc.new_poly_events(within_hours=24.0, now=now, max_pages=10)
    slugs = [e["slug"] for e in fresh]
    # Early-stop at old1 -> new3 (which comes after it) is NOT collected.
    assert slugs == ["new1", "new2"]
    # Only one page fetched (early-stop), well clear of the offset cap.
    assert calls["n"] == 1


def test_new_poly_events_drops_ephemeral(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    page0 = [
        {
            "slug": "btc-eoy",
            "title": "Bitcoin above $200k by Dec 31 2026",
            "startDate": _iso(now - timedelta(hours=1)),
        },
        {
            "slug": "sol-updown",
            "title": "Solana Up or Down 15m",
            "startDate": _iso(now - timedelta(hours=1)),
        },
        {
            "slug": "trump-2028",
            "title": "Will Trump win 2028?",
            "startDate": _iso(now - timedelta(hours=2)),
        },
    ]

    def fake_get(url: str, *, params: Any = None, session: Any = None, timeout: float = 15.0):
        return page0 if params["offset"] == 0 else []

    monkeypatch.setattr(mc, "_get", fake_get)

    fresh = mc.new_poly_events(within_hours=24.0, now=now, max_pages=10)
    slugs = {e["slug"] for e in fresh}
    assert "sol-updown" not in slugs
    assert slugs == {"btc-eoy", "trump-2028"}


# ---------------------------------------------------------------------------
# clobTokenIds double-decode.
# ---------------------------------------------------------------------------


def test_parse_clob_token_ids_double_json() -> None:
    market = {"clobTokenIds": '["123", "456"]'}
    assert mc.parse_clob_token_ids(market) == ["123", "456"]
    assert mc.parse_clob_token_ids({"clobTokenIds": ["7", 8]}) == ["7", "8"]
    assert mc.parse_clob_token_ids({}) == []
    assert mc.parse_clob_token_ids({"clobTokenIds": "not-json"}) == []


# ---------------------------------------------------------------------------
# Checkpoint round-trip + reset logic.
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path: Any) -> None:
    path = tmp_path / "state" / "crawl_state.json"
    ckpt = mc.CrawlCheckpoint(
        kalshi_cursor="CUR", poly_offset=300, last_seen_poly_start_iso="2026-05-20T00:00:00Z"
    )
    mc.save_checkpoint(path, ckpt)
    loaded = mc.load_checkpoint(path)
    assert loaded == ckpt


def test_load_checkpoint_missing_returns_fresh(tmp_path: Any) -> None:
    loaded = mc.load_checkpoint(tmp_path / "nope.json")
    assert loaded == mc.CrawlCheckpoint()


def test_load_checkpoint_corrupt_returns_fresh(tmp_path: Any) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert mc.load_checkpoint(p) == mc.CrawlCheckpoint()


def test_advance_checkpoint_steps_forward() -> None:
    ckpt = mc.CrawlCheckpoint(kalshi_cursor="C0", poly_offset=0)
    kp = mc.KalshiCrawlPage(markets=[], next_cursor="C1", done=False, n_pages=1)
    pp = mc.PolyCrawlPage(
        events=[{"slug": "e", "startDate": "2026-05-20T00:00:00Z"}],
        next_offset=100,
        done=False,
        n_pages=1,
    )
    out = mc.advance_checkpoint(ckpt, kalshi_page=kp, poly_page=pp)
    assert out.kalshi_cursor == "C1"
    assert out.poly_offset == 100
    assert out.last_seen_poly_start_iso == "2026-05-20T00:00:00Z"


def test_advance_checkpoint_resets_on_kalshi_exhaust() -> None:
    ckpt = mc.CrawlCheckpoint(kalshi_cursor="C5")
    kp = mc.KalshiCrawlPage(markets=[], next_cursor=None, done=True, n_pages=1)
    out = mc.advance_checkpoint(ckpt, kalshi_page=kp)
    assert out.kalshi_cursor is None  # fresh sweep next cycle


def test_advance_checkpoint_resets_on_poly_cap() -> None:
    ckpt = mc.CrawlCheckpoint(poly_offset=mc.POLY_OFFSET_CAP - 100)
    pp = mc.PolyCrawlPage(events=[], next_offset=mc.POLY_OFFSET_CAP, done=True, n_pages=1)
    out = mc.advance_checkpoint(ckpt, poly_page=pp)
    assert out.poly_offset == 0


# ---------------------------------------------------------------------------
# HTTP layer: 429 backoff + non-retryable error.
# ---------------------------------------------------------------------------


def test_get_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mc.time, "sleep", lambda *_a, **_k: None)
    session = FakeSession(
        [
            FakeResp(status_code=429, headers={"Retry-After": "0"}),
            FakeResp(status_code=200, json_body={"markets": [], "cursor": ""}),
        ]
    )
    body = mc._get("http://x", params={}, session=session)
    assert body == {"markets": [], "cursor": ""}
    assert len(session.calls) == 2


def test_get_raises_on_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession([FakeResp(status_code=422, text="offset exceeds maximum")])
    with pytest.raises(mc.CrawlHTTPError) as exc:
        mc._get("http://x", params={}, session=session)
    assert exc.value.status_code == 422


def test_get_429_exhausts_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mc.time, "sleep", lambda *_a, **_k: None)
    session = FakeSession([FakeResp(status_code=429) for _ in range(mc.MAX_BACKOFF_RETRIES + 1)])
    with pytest.raises(mc.CrawlHTTPError) as exc:
        mc._get("http://x", params={}, session=session)
    assert exc.value.status_code == 429
