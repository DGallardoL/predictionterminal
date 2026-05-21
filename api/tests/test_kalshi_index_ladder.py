"""Tests for Kalshi index-ladder discovery (S&P 500 / Nasdaq-100).

All HTTP is mocked with ``respx`` — no network. Mirrors the mocking pattern
in ``test_kalshi_archive.py`` / ``test_kalshi_ratelimit.py``: a deterministic
:class:`KalshiClient` (``min_interval_s=0``, ``max_retries=0``) plus
``respx.get(url).mock(...)``.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pfm.sources.kalshi import (
    INDEX_SERIES,
    KalshiClient,
    KalshiError,
    discover_index_ladder,
)
from pfm.vol.implied_pdf_schemas import LadderFamily

BASE = KalshiClient.BASE_URL


def _client() -> KalshiClient:
    return KalshiClient(min_interval_s=0.0, max_retries=0, sleep=lambda *_a: None)


# ─────────────────────────── payload builders ──────────────────────────────


def _between(
    ticker: str,
    floor: float | None,
    cap: float | None,
    *,
    bid: float | None = None,
    ask: float | None = None,
    legacy: bool = False,
    last: float | None = None,
) -> dict:
    mkt: dict = {
        "ticker": ticker,
        "event_ticker": "KXINX-26MAY15H1600",
        "status": "open",
        "strike_type": "between",
        "floor_strike": floor,
        "cap_strike": cap,
        "expected_expiration_time": "2026-05-15T20:00:00Z",
    }
    if legacy:
        if bid is not None:
            mkt["yes_bid"] = bid  # integer cents
        if ask is not None:
            mkt["yes_ask"] = ask
        if last is not None:
            mkt["last_price"] = last
    else:
        if bid is not None:
            mkt["yes_bid_dollars"] = bid
        if ask is not None:
            mkt["yes_ask_dollars"] = ask
        if last is not None:
            mkt["last_price_dollars"] = last
    return mkt


def _threshold(
    ticker: str,
    strike_type: str,
    *,
    floor: float | None = None,
    cap: float | None = None,
    bid: float,
    ask: float,
    event_ticker: str = "KXINXU-26MAY15H1600",
) -> dict:
    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "status": "open",
        "strike_type": strike_type,
        "floor_strike": floor,
        "cap_strike": cap,
        "yes_bid_dollars": bid,
        "yes_ask_dollars": ask,
        "expected_expiration_time": "2026-05-15T20:00:00Z",
    }


def _events_payload(markets: list[dict], event_ticker: str = "KXINX-26MAY15H1600") -> dict:
    return {
        "events": [
            {
                "event_ticker": event_ticker,
                "series_ticker": event_ticker.split("-", 1)[0],
                "status": "open",
                "markets": markets,
            }
        ],
        "cursor": "",
    }


# ──────────────────────────── 1. between buckets ───────────────────────────


@respx.mock
def test_between_buckets_terminal_buckets_shape() -> None:
    markets = [
        # half-open tail bucket (no floor)
        _between("KXINX-...-B5200", None, 5200.0, bid=0.04, ask=0.06),
        _between("KXINX-...-B5300", 5200.0, 5300.0, bid=0.10, ask=0.12),
        _between("KXINX-...-B5400", 5300.0, 5400.0, bid=0.28, ask=0.32),
        _between("KXINX-...-B5500", 5400.0, 5500.0, bid=0.24, ask=0.26),
        _between("KXINX-...-B5600", 5500.0, 5600.0, bid=0.13, ask=0.15),
        # half-open tail bucket (no cap)
        _between("KXINX-...-B5600U", 5600.0, None, bid=0.03, ask=0.05),
    ]
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(200, json=_events_payload(markets))
    )

    fam = discover_index_ladder("SPX", _client())

    assert isinstance(fam, LadderFamily)
    assert fam.data_shape == "terminal_buckets"
    assert fam.asset == "SPX"
    assert fam.asset_class == "equity_index"
    assert fam.spot is None
    assert fam.source == "kalshi:KXINX-26MAY15H1600"
    assert len(fam.entries) == 6
    assert all(e.direction == "between" for e in fam.entries)

    # half-open tail: floor None, cap 5200, mid(0.04,0.06)=0.05
    tail = fam.entries[0]
    assert tail.floor is None
    assert tail.cap == 5200.0
    assert tail.prob == pytest.approx(0.05)

    mid_bucket = fam.entries[2]
    assert mid_bucket.floor == 5300.0
    assert mid_bucket.cap == 5400.0
    assert mid_bucket.prob == pytest.approx(0.30)  # mid(0.28, 0.32)

    # maturity parsed from ISO close
    assert fam.maturity_utc.year == 2026
    assert fam.maturity_utc.month == 5
    assert fam.maturity_utc.day == 15
    assert fam.maturity_utc.tzinfo is not None


# ──────────────────────────── 2. legacy cent prices ────────────────────────


@respx.mock
def test_legacy_integer_cent_prices_divided_by_100() -> None:
    markets = [
        _between("KXINX-...-A", 5300.0, 5400.0, bid=28, ask=32, legacy=True),
        _between("KXINX-...-B", 5400.0, 5500.0, bid=24, ask=26, legacy=True),
        # only a last_price (no bid/ask)
        _between("KXINX-...-C", 5500.0, 5600.0, last=15, legacy=True),
    ]
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(200, json=_events_payload(markets))
    )

    fam = discover_index_ladder("SPX", _client())

    assert fam.entries[0].prob == pytest.approx(0.30)  # mid(0.28, 0.32)
    assert fam.entries[1].prob == pytest.approx(0.25)
    assert fam.entries[2].prob == pytest.approx(0.15)  # last 15c → 0.15


# ──────────────────────────── 3. above/below ladder ────────────────────────


@respx.mock
def test_threshold_ladder_terminal_ladder_shape() -> None:
    markets = [
        _threshold("KXINXU-...-G5300", "greater", floor=5300.0, bid=0.80, ask=0.82),
        _threshold("KXINXU-...-G5400", "greater_or_equal", floor=5400.0, bid=0.55, ask=0.57),
        _threshold("KXINXU-...-L5500", "less", cap=5500.0, bid=0.60, ask=0.62),
        _threshold("KXINXU-...-L5600", "less_or_equal", cap=5600.0, bid=0.75, ask=0.77),
    ]
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(
            200, json=_events_payload(markets, event_ticker="KXINXU-26MAY15H1600")
        )
    )

    fam = discover_index_ladder("SPX", _client(), prefer_shape="terminal_ladder")

    assert fam.data_shape == "terminal_ladder"
    assert fam.source == "kalshi:KXINXU-26MAY15H1600"

    above = [e for e in fam.entries if e.direction == "above"]
    below = [e for e in fam.entries if e.direction == "below"]
    assert len(above) == 2
    assert len(below) == 2

    # above → strike from floor_strike
    assert above[0].strike == 5300.0
    assert above[0].prob == pytest.approx(0.81)
    # below → strike from cap_strike
    assert below[0].strike == 5500.0
    assert below[0].prob == pytest.approx(0.61)


@respx.mock
def test_prefer_shape_ladder_targets_the_U_series_in_query() -> None:
    captured: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.url.params))
        return httpx.Response(
            200,
            json=_events_payload(
                [_threshold("KXINXU-...-G5300", "greater", floor=5300.0, bid=0.8, ask=0.82)],
                event_ticker="KXINXU-26MAY15H1600",
            ),
        )

    respx.get(f"{BASE}/events").mock(side_effect=_handler)
    discover_index_ladder("SPX", _client(), prefer_shape="terminal_ladder")
    assert captured[0]["series_ticker"] == INDEX_SERIES["SPX"]["ladder"]


# ──────────────────────────── 4. maturity_filter ───────────────────────────


@respx.mock
def test_maturity_filter_selects_matching_event() -> None:
    payload = {
        "events": [
            {
                "event_ticker": "KXINX-26MAY15H1600",
                "status": "open",
                "markets": [_between("KXINX-15-A", 5300.0, 5400.0, bid=0.3, ask=0.32)],
            },
            {
                "event_ticker": "KXINX-26MAY16H1600",
                "status": "open",
                "markets": [_between("KXINX-16-A", 5300.0, 5400.0, bid=0.4, ask=0.42)],
            },
            {
                "event_ticker": "KXINX-26MAY17H1600",
                "status": "open",
                "markets": [_between("KXINX-17-A", 5300.0, 5400.0, bid=0.5, ask=0.52)],
            },
        ],
        "cursor": "",
    }
    # Fix ISO close so maturity comes from the ticker date code path too;
    # here we drop the ISO field to force ticker-date parsing.
    for ev in payload["events"]:
        for m in ev["markets"]:
            m.pop("expected_expiration_time", None)

    respx.get(f"{BASE}/events").mock(return_value=httpx.Response(200, json=payload))

    fam = discover_index_ladder("SPX", _client(), maturity_filter="2026-05-16")

    assert fam.source == "kalshi:KXINX-26MAY16H1600"
    assert fam.maturity_utc.day == 16
    assert fam.entries[0].prob == pytest.approx(0.41)


@respx.mock
def test_maturity_filter_no_match_raises() -> None:
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(
            200,
            json=_events_payload([_between("KXINX-A", 5300.0, 5400.0, bid=0.3, ask=0.32)]),
        )
    )
    with pytest.raises(KalshiError):
        discover_index_ladder("SPX", _client(), maturity_filter="2099-01-01")


# ──────────────────────────── 5. empty / no events ─────────────────────────


@respx.mock
def test_no_open_events_raises() -> None:
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(200, json={"events": [], "cursor": ""})
    )
    with pytest.raises(KalshiError):
        discover_index_ladder("SPX", _client())


@respx.mock
def test_event_with_empty_markets_raises() -> None:
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(
            200,
            json={
                "events": [{"event_ticker": "KXINX-26MAY15H1600", "status": "open", "markets": []}],
                "cursor": "",
            },
        )
    )
    with pytest.raises(KalshiError):
        discover_index_ladder("SPX", _client())


@respx.mock
def test_only_functional_markets_raises_and_is_skipped() -> None:
    markets = [
        {
            "ticker": "KXINX-FN",
            "event_ticker": "KXINX-26MAY15H1600",
            "status": "open",
            "strike_type": "functional",
            "yes_bid_dollars": 0.5,
            "yes_ask_dollars": 0.52,
            "expected_expiration_time": "2026-05-15T20:00:00Z",
        }
    ]
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(200, json=_events_payload(markets))
    )
    with pytest.raises(KalshiError):
        discover_index_ladder("SPX", _client())


# ──────────────────────────── 6. get_events plumbing ───────────────────────


@respx.mock
def test_get_events_builds_url_and_params_and_parses_list() -> None:
    captured: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.url.params))
        assert str(request.url).startswith(f"{BASE}/events")
        return httpx.Response(
            200,
            json=_events_payload([_between("KXINX-A", 5300.0, 5400.0, bid=0.3, ask=0.32)]),
        )

    respx.get(f"{BASE}/events").mock(side_effect=_handler)

    events = _client().get_events("KXINX", status="open", limit=50)

    assert isinstance(events, list)
    assert len(events) == 1
    assert events[0]["event_ticker"] == "KXINX-26MAY15H1600"

    params = captured[0]
    assert params["series_ticker"] == "KXINX"
    assert params["status"] == "open"
    assert params["limit"] == "50"
    assert params["with_nested_markets"] == "true"


@respx.mock
def test_get_event_single_fetch() -> None:
    respx.get(f"{BASE}/events/KXINX-26MAY15H1600").mock(
        return_value=httpx.Response(
            200,
            json={
                "event": {
                    "event_ticker": "KXINX-26MAY15H1600",
                    "status": "open",
                    "markets": [_between("KXINX-A", 5300.0, 5400.0, bid=0.3, ask=0.32)],
                }
            },
        )
    )
    event = _client().get_event("KXINX-26MAY15H1600")
    assert event["event_ticker"] == "KXINX-26MAY15H1600"
    assert len(event["markets"]) == 1


@respx.mock
def test_get_event_missing_raises() -> None:
    respx.get(f"{BASE}/events/KXMISSING").mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(KalshiError):
        _client().get_event("KXMISSING")


# ──────────────────────── raw-series-ticker acceptance ─────────────────────


@respx.mock
def test_accepts_raw_series_ticker() -> None:
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(
            200,
            json=_events_payload([_between("KXINX-A", 5300.0, 5400.0, bid=0.3, ask=0.32)]),
        )
    )
    fam = discover_index_ladder("KXINX", _client())
    # Raw series ticker resolves back to the friendly asset key.
    assert fam.asset == "SPX"
    assert fam.data_shape == "terminal_buckets"
