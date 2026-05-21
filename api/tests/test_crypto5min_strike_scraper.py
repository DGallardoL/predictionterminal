"""Tests for the polymarket.com priceToBeat scraper."""

from __future__ import annotations

import re

import httpx
import pytest
import respx

from pfm.crypto5min.strike_scraper import (
    POLYMARKET_BASE,
    SLUG_PREFIX_BY_ASSET,
    _iso_to_unix,
    _reset_cache,
    _series_url,
    fetch_strikes,
    get_strike_for_market,
    parse_html_for_strikes,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    _reset_cache()
    yield
    _reset_cache()


# ---------------------------------------------------------------------------
# Pure parser
# ---------------------------------------------------------------------------


def test_iso_to_unix_basic() -> None:
    assert _iso_to_unix("2026-05-14T15:25:00Z") == 1_778_772_300


def test_iso_to_unix_handles_fractional_seconds() -> None:
    # Fractional seconds get stripped, only second-resolution preserved.
    assert _iso_to_unix("2026-05-14T15:25:00.387Z") == 1_778_772_300


def test_iso_to_unix_rejects_garbage() -> None:
    assert _iso_to_unix("not-a-date") is None
    assert _iso_to_unix("") is None


def test_parse_html_empty_returns_empty() -> None:
    assert parse_html_for_strikes("") == {}
    assert parse_html_for_strikes("<html><body>nothing</body></html>") == {}


def test_parse_html_single_event_with_both_prices() -> None:
    html = (
        '... lots of context "startTime":"2026-05-14T15:20:00Z",'
        '"endDate":"2026-05-14T15:25:00Z",'
        '"seriesSlug":"btc-up-or-down-5m","eventMetadata":'
        '{"priceToBeat":80994.33,"finalPrice":81005.50} ...'
    )
    out = parse_html_for_strikes(html)
    # priceToBeat → start_unix; finalPrice → end_unix
    assert out[1_778_772_000] == pytest.approx(80994.33)
    assert out[1_778_772_300] == pytest.approx(81005.50)


def test_parse_html_priceToBeat_only_no_finalPrice() -> None:
    """Most-recently-active event has priceToBeat but no finalPrice yet."""
    html = (
        '"startTime":"2026-05-14T15:20:00Z","endDate":"2026-05-14T15:25:00Z",'
        '"seriesSlug":"btc","eventMetadata":{"priceToBeat":80994.33}'
    )
    out = parse_html_for_strikes(html)
    assert out == {1_778_772_000: pytest.approx(80994.33)}


def test_parse_html_skips_malformed_metadata() -> None:
    html = (
        '"startTime":"2026-05-14T15:20:00Z",'
        '"endDate":"2026-05-14T15:25:00Z",'
        '"eventMetadata":{this is not JSON}'
    )
    assert parse_html_for_strikes(html) == {}


def test_parse_html_skips_empty_metadata() -> None:
    html = '"startTime":"2026-05-14T15:20:00Z","eventMetadata":{}'
    assert parse_html_for_strikes(html) == {}


def test_parse_html_skips_metadata_without_startTime() -> None:
    # No startTime before the eventMetadata block → can't anchor the price.
    html = '"eventMetadata":{"priceToBeat":80994.33}'
    assert parse_html_for_strikes(html) == {}


def test_parse_html_keeps_first_value_when_duplicates() -> None:
    """Polymarket sometimes lists the same event twice on a page (e.g. in
    series cards + main detail). We use ``setdefault`` so the first hit wins."""
    html = (
        '"startTime":"2026-05-14T15:20:00Z","eventMetadata":{"priceToBeat":80994.33}'
        " ... lots of stuff ... "
        '"startTime":"2026-05-14T15:20:00Z","eventMetadata":{"priceToBeat":99999.99}'
    )
    out = parse_html_for_strikes(html)
    assert out[1_778_772_000] == pytest.approx(80994.33)


def test_parse_html_multiple_events_chronological() -> None:
    html = (
        '"startTime":"2026-05-14T15:00:00Z","endDate":"2026-05-14T15:05:00Z",'
        '"eventMetadata":{"finalPrice":80100.0,"priceToBeat":80000.0}'
        ' "startTime":"2026-05-14T15:05:00Z","endDate":"2026-05-14T15:10:00Z",'
        '"eventMetadata":{"finalPrice":80200.0,"priceToBeat":80100.0}'
        ' "startTime":"2026-05-14T15:10:00Z","endDate":"2026-05-14T15:15:00Z",'
        '"eventMetadata":{"finalPrice":80300.0,"priceToBeat":80200.0}'
    )
    out = parse_html_for_strikes(html)
    assert out[_iso_to_unix("2026-05-14T15:00:00Z")] == pytest.approx(80000.0)
    assert out[_iso_to_unix("2026-05-14T15:05:00Z")] == pytest.approx(80100.0)
    assert out[_iso_to_unix("2026-05-14T15:10:00Z")] == pytest.approx(80200.0)
    assert out[_iso_to_unix("2026-05-14T15:15:00Z")] == pytest.approx(80300.0)


def test_parse_html_ignores_non_numeric_prices() -> None:
    html = (
        '"startTime":"2026-05-14T15:00:00Z",'
        '"eventMetadata":{"priceToBeat":"oops","finalPrice":null}'
    )
    assert parse_html_for_strikes(html) == {}


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def test_series_url_handles_known_assets() -> None:
    url = _series_url("BTC", 5, now_unix=1_778_772_900)
    # last_end = (1778772900 // 300) * 300 = 1778772900 (exactly on boundary)
    assert url == f"{POLYMARKET_BASE}/event/btc-updown-5m-1778772900"


def test_series_url_unknown_asset_returns_none() -> None:
    assert _series_url("DOGE", 5) is None


def test_series_url_unknown_window_returns_none() -> None:
    assert _series_url("BTC", 7) is None


def test_slug_prefixes_known() -> None:
    assert SLUG_PREFIX_BY_ASSET["BTC"][5] == "btc-updown-5m"
    assert SLUG_PREFIX_BY_ASSET["ETH"][15] == "eth-updown-15m"


# ---------------------------------------------------------------------------
# fetch_strikes (mocked HTTP)
# ---------------------------------------------------------------------------

_FAKE_HTML = (
    '"startTime":"2026-05-14T15:15:00Z","endDate":"2026-05-14T15:20:00Z",'
    '"eventMetadata":{"finalPrice":80994.33,"priceToBeat":80971.37}'
    ' "startTime":"2026-05-14T15:10:00Z","endDate":"2026-05-14T15:15:00Z",'
    '"eventMetadata":{"finalPrice":80971.37,"priceToBeat":80870.43}'
)


#: Match any series URL — the actual end_unix in the URL depends on the
#: clock at test run-time which we don't pin.
_BTC_5M_URL = re.compile(r"https://polymarket\.com/event/btc-updown-5m-\d+")
_BTC_15M_URL = re.compile(r"https://polymarket\.com/event/btc-updown-15m-\d+")


@pytest.mark.asyncio
async def test_fetch_strikes_happy() -> None:
    async with respx.mock:
        respx.get(url__regex=_BTC_5M_URL).mock(return_value=httpx.Response(200, text=_FAKE_HTML))
        async with httpx.AsyncClient() as client:
            snap = await fetch_strikes(client, "BTC", 5, timeout=2.0)
    assert snap is not None
    assert snap.asset == "BTC"
    assert snap.window_minutes == 5
    assert snap.get(_iso_to_unix("2026-05-14T15:15:00Z")) == pytest.approx(80971.37)
    assert snap.get(_iso_to_unix("2026-05-14T15:20:00Z")) == pytest.approx(80994.33)


@pytest.mark.asyncio
async def test_fetch_strikes_caches_result() -> None:
    call_count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, text=_FAKE_HTML)

    async with respx.mock:
        respx.get(url__regex=_BTC_5M_URL).mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            await fetch_strikes(client, "BTC", 5, timeout=2.0)
            await fetch_strikes(client, "BTC", 5, timeout=2.0)
            await fetch_strikes(client, "BTC", 5, timeout=2.0)
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_fetch_strikes_returns_none_on_404() -> None:
    async with respx.mock:
        respx.get(url__regex=_BTC_5M_URL).mock(return_value=httpx.Response(404))
        async with httpx.AsyncClient() as client:
            snap = await fetch_strikes(client, "BTC", 5, timeout=2.0)
    assert snap is None


@pytest.mark.asyncio
async def test_fetch_strikes_returns_none_on_network_error() -> None:
    async with respx.mock:
        respx.get(url__regex=_BTC_5M_URL).mock(side_effect=httpx.ConnectError("boom"))
        async with httpx.AsyncClient() as client:
            snap = await fetch_strikes(client, "BTC", 5, timeout=2.0)
    assert snap is None


@pytest.mark.asyncio
async def test_fetch_strikes_returns_none_on_empty_html() -> None:
    async with respx.mock:
        respx.get(url__regex=_BTC_5M_URL).mock(
            return_value=httpx.Response(200, text="<html>no events</html>")
        )
        async with httpx.AsyncClient() as client:
            snap = await fetch_strikes(client, "BTC", 5, timeout=2.0)
    assert snap is None


# ---------------------------------------------------------------------------
# get_strike_for_market — fallback chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_strike_falls_through_to_5m_scrape_for_15m_market() -> None:
    async with respx.mock:
        respx.get(url__regex=_BTC_15M_URL).mock(
            return_value=httpx.Response(200, text="<html>no events</html>")
        )
        respx.get(url__regex=_BTC_5M_URL).mock(return_value=httpx.Response(200, text=_FAKE_HTML))
        async with httpx.AsyncClient() as client:
            target_start = _iso_to_unix("2026-05-14T15:15:00Z")
            price, source = await get_strike_for_market(
                client,
                asset="BTC",
                window_minutes=15,
                start_unix=target_start,
            )
    assert price == pytest.approx(80971.37)
    assert source == "polymarket-scrape"


@pytest.mark.asyncio
async def test_get_strike_returns_unavailable_when_no_page_matches() -> None:
    async with respx.mock:
        respx.get(url__regex=_BTC_5M_URL).mock(
            return_value=httpx.Response(200, text="<html>no events</html>")
        )
        async with httpx.AsyncClient() as client:
            price, source = await get_strike_for_market(
                client,
                asset="BTC",
                window_minutes=5,
                start_unix=_iso_to_unix("2026-05-14T15:15:00Z"),
            )
    assert price is None
    assert source == "unavailable"


@pytest.mark.asyncio
async def test_get_strike_returns_unavailable_for_unknown_asset() -> None:
    async with httpx.AsyncClient() as client:
        price, source = await get_strike_for_market(
            client,
            asset="DOGE",
            window_minutes=5,
            start_unix=1_778_772_300,
        )
    assert price is None
    assert source == "unavailable"


@pytest.mark.asyncio
async def test_get_strike_misses_when_requested_start_not_in_scrape() -> None:
    """The scrape has 15:15 and 15:20 but we ask for 15:25 — should miss."""
    async with respx.mock:
        respx.get(url__regex=_BTC_5M_URL).mock(return_value=httpx.Response(200, text=_FAKE_HTML))
        async with httpx.AsyncClient() as client:
            price, source = await get_strike_for_market(
                client,
                asset="BTC",
                window_minutes=5,
                start_unix=_iso_to_unix("2026-05-14T15:25:00Z"),
            )
    assert price is None
    assert source == "unavailable"
