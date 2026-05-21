"""Tests for the Polymarket discovery + CLOB midpoint fetch."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from pfm.crypto5min.market_fetcher import (
    DEFAULT_CLOB_URL,
    DEFAULT_GAMMA_URL,
    ActiveMarket,
    discover_active_markets,
    fetch_binance_mid,
    fetch_clob_midpoint,
    parse_active_market,
)

# ---------------------------------------------------------------------------
# parse_active_market (pure, no IO)
# ---------------------------------------------------------------------------


def _gamma_payload(
    slug: str,
    *,
    up: str = "tok_up",
    down: str = "tok_down",
    closed: bool = False,
) -> dict:
    return {
        "slug": slug,
        "id": "12345",
        "closed": closed,
        "active": not closed,
        "clobTokenIds": json.dumps([up, down]),
    }


def test_parse_active_market_minimal_ok() -> None:
    payload = _gamma_payload("btc-updown-5m-1700000000")
    out = parse_active_market(
        payload,
        asset="BTC",
        window_minutes=5,
        binance_symbol="BTCUSDT",
        end_unix=1_700_000_000,
        now_unix=1_700_000_000 - 100,  # 100s remaining
    )
    assert out is not None
    assert out.asset == "BTC"
    assert out.window_minutes == 5
    assert out.up_token_id == "tok_up"
    assert out.down_token_id == "tok_down"
    assert out.start_unix == 1_700_000_000 - 300
    assert out.end_unix == 1_700_000_000
    assert out.seconds_remaining == pytest.approx(100.0)


def test_parse_active_market_returns_none_on_closed() -> None:
    payload = _gamma_payload("btc-updown-5m-1700000000", closed=True)
    out = parse_active_market(
        payload,
        asset="BTC",
        window_minutes=5,
        binance_symbol="BTCUSDT",
        end_unix=1_700_000_000,
        now_unix=1_700_000_000 - 100,
    )
    assert out is None


def test_parse_active_market_returns_none_on_missing_tokens() -> None:
    bad = {"slug": "btc-updown-5m-1700000000", "id": "x"}
    out = parse_active_market(
        bad,
        asset="BTC",
        window_minutes=5,
        binance_symbol="BTCUSDT",
        end_unix=1_700_000_000,
        now_unix=1_700_000_000 - 100,
    )
    assert out is None


def test_parse_active_market_returns_none_on_bad_json_tokens() -> None:
    bad = {
        "slug": "btc-updown-5m-1700000000",
        "id": "x",
        "clobTokenIds": "not-json",
    }
    out = parse_active_market(
        bad,
        asset="BTC",
        window_minutes=5,
        binance_symbol="BTCUSDT",
        end_unix=1_700_000_000,
        now_unix=1_700_000_000 - 100,
    )
    assert out is None


def test_parse_active_market_accepts_list_tokens() -> None:
    payload = {
        "slug": "btc-updown-5m-1700000000",
        "id": "x",
        "clobTokenIds": ["a", "b"],
    }
    out = parse_active_market(
        payload,
        asset="BTC",
        window_minutes=5,
        binance_symbol="BTCUSDT",
        end_unix=1_700_000_000,
        now_unix=1_700_000_000 - 100,
    )
    assert out is not None
    assert out.up_token_id == "a"


def test_parse_active_market_seconds_remaining_clamped_to_zero() -> None:
    payload = _gamma_payload("btc-updown-5m-1700000000")
    out = parse_active_market(
        payload,
        asset="BTC",
        window_minutes=5,
        binance_symbol="BTCUSDT",
        end_unix=1_700_000_000,
        now_unix=1_700_000_001,  # past
    )
    assert out is not None
    assert out.seconds_remaining == 0.0


def test_active_market_as_dict_round_trip() -> None:
    m = ActiveMarket(
        asset="BTC",
        binance_symbol="BTCUSDT",
        window_minutes=5,
        slug="x",
        market_id="1",
        up_token_id="a",
        down_token_id="b",
        start_unix=100,
        end_unix=400,
        seconds_remaining=200.0,
    )
    d = m.as_dict()
    assert d["asset"] == "BTC"
    assert d["seconds_remaining"] == 200.0


# ---------------------------------------------------------------------------
# discover_active_markets (with respx)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_finds_btc_5m_at_next_boundary() -> None:
    """The function tries (period boundary + 0/1/2 offsets) for each asset+window.
    We mock just the BTC-5m hit and confirm we get exactly one market back."""
    now = 1_700_000_000  # arbitrary unix
    period = 300
    next_end = ((now // period) + 1) * period
    slug = f"btc-updown-5m-{next_end}"
    payload = [_gamma_payload(slug)]
    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=payload if req.url.params.get("slug") == slug else [],
            )
        )
        async with httpx.AsyncClient() as client:
            out = await discover_active_markets(
                client,
                assets=["BTC"],
                window_minutes_list=[5],
                now_unix=now,
                timeout=1.0,
            )
    assert len(out) == 1
    assert out[0].asset == "BTC"
    assert out[0].slug == slug
    assert out[0].window_minutes == 5


@pytest.mark.asyncio
async def test_discover_skips_when_no_market_exists() -> None:
    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=[]))
        async with httpx.AsyncClient() as client:
            out = await discover_active_markets(
                client,
                assets=["BTC"],
                window_minutes_list=[5],
                now_unix=1_700_000_000,
                lookahead=2,
                timeout=1.0,
            )
    assert out == []


@pytest.mark.asyncio
async def test_discover_handles_legacy_slug_prefix() -> None:
    """Falls back to ``btc-up-or-down-5m-...`` if ``btc-updown-5m-...`` 404s."""
    now = 1_700_000_000
    period = 300
    next_end = ((now // period) + 1) * period
    legacy_slug = f"btc-up-or-down-5m-{next_end}"

    def handler(req: httpx.Request) -> httpx.Response:
        slug = req.url.params.get("slug")
        if slug == legacy_slug:
            return httpx.Response(200, json=[_gamma_payload(legacy_slug)])
        return httpx.Response(200, json=[])

    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            out = await discover_active_markets(
                client,
                assets=["BTC"],
                window_minutes_list=[5],
                now_unix=now,
                timeout=1.0,
            )
    assert len(out) == 1
    assert out[0].slug == legacy_slug


@pytest.mark.asyncio
async def test_discover_handles_network_error() -> None:
    async with respx.mock:
        respx.get(f"{DEFAULT_GAMMA_URL}/markets").mock(side_effect=httpx.ConnectError("boom"))
        async with httpx.AsyncClient() as client:
            out = await discover_active_markets(
                client,
                assets=["BTC"],
                window_minutes_list=[5],
                now_unix=1_700_000_000,
                lookahead=1,
                timeout=0.5,
            )
    assert out == []


# ---------------------------------------------------------------------------
# fetch_clob_midpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_clob_midpoint_happy() -> None:
    async with respx.mock:
        respx.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.62"})
        )
        async with httpx.AsyncClient() as client:
            mid = await fetch_clob_midpoint(client, "tok_abc")
    assert mid == pytest.approx(0.62)


@pytest.mark.asyncio
async def test_fetch_clob_midpoint_handles_5xx() -> None:
    async with respx.mock:
        respx.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(return_value=httpx.Response(503))
        async with httpx.AsyncClient() as client:
            mid = await fetch_clob_midpoint(client, "tok_abc")
    assert mid is None


@pytest.mark.asyncio
async def test_fetch_clob_midpoint_rejects_out_of_range_mid() -> None:
    async with respx.mock:
        respx.get(f"{DEFAULT_CLOB_URL}/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "1.5"})
        )
        async with httpx.AsyncClient() as client:
            mid = await fetch_clob_midpoint(client, "tok_abc")
    assert mid is None


@pytest.mark.asyncio
async def test_fetch_clob_midpoint_empty_token_id_returns_none() -> None:
    async with httpx.AsyncClient() as client:
        out = await fetch_clob_midpoint(client, "")
    assert out is None


# ---------------------------------------------------------------------------
# fetch_binance_mid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_binance_mid_happy() -> None:
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=httpx.Response(
                200,
                json={
                    "symbol": "BTCUSDT",
                    "bidPrice": "60000.0",
                    "askPrice": "60010.0",
                    "bidQty": "1.0",
                    "askQty": "1.0",
                },
            )
        )
        async with httpx.AsyncClient() as client:
            mid = await fetch_binance_mid(client, "BTCUSDT")
    assert mid == pytest.approx(60_005.0)


@pytest.mark.asyncio
async def test_fetch_binance_mid_handles_5xx() -> None:
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=httpx.Response(503)
        )
        async with httpx.AsyncClient() as client:
            mid = await fetch_binance_mid(client, "BTCUSDT")
    assert mid is None


@pytest.mark.asyncio
async def test_fetch_binance_mid_rejects_bad_payload() -> None:
    async with respx.mock:
        respx.get("https://api.binance.com/api/v3/ticker/bookTicker").mock(
            return_value=httpx.Response(200, json={"oops": "no prices"})
        )
        async with httpx.AsyncClient() as client:
            mid = await fetch_binance_mid(client, "BTCUSDT")
    assert mid is None
