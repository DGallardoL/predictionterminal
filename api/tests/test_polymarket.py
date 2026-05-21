"""Tests for the Polymarket Gamma + CLOB client. All HTTP is mocked via respx."""

from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest
import respx

from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,
)

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"


@pytest.fixture
def client() -> PolymarketClient:
    return PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())


@respx.mock
def test_get_market_metadata_parses_clob_token_ids(client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets", params={"slug": "fed-decision"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "abc",
                    "question": "Will the Fed cut rates?",
                    "slug": "fed-decision",
                    "clobTokenIds": json.dumps(["111", "222"]),
                    "startDate": "2025-07-15T00:00:00Z",
                    "endDate": "2026-11-01T00:00:00Z",
                    "closed": False,
                    "active": True,
                }
            ],
        )
    )

    meta = client.get_market_metadata("fed-decision")
    assert meta.yes_token_id == "111"
    assert meta.no_token_id == "222"
    assert meta.active is True
    assert meta.closed is False


@respx.mock
def test_get_market_metadata_empty_list_raises(client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json=[]))
    with pytest.raises(PolymarketError, match="no market found"):
        client.get_market_metadata("missing-slug")


@respx.mock
def test_get_market_metadata_bad_clob_token_ids_raises(client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[{"slug": "x", "clobTokenIds": "not-json", "question": "?"}],
        )
    )
    with pytest.raises(PolymarketError, match="not valid JSON"):
        client.get_market_metadata("x")


@respx.mock
def test_get_price_history_parses_response(client: PolymarketClient) -> None:
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(
            200,
            json={
                "history": [
                    {"t": 1706745600, "p": 0.42},
                    {"t": 1706832000, "p": 0.45},
                ]
            },
        )
    )
    df = client.get_price_history("111")
    assert list(df.columns) == ["date", "price"]
    assert len(df) == 2
    assert df["price"].tolist() == [0.42, 0.45]
    assert all(d.tzinfo is not None for d in df["date"])


@respx.mock
def test_get_price_history_empty_returns_empty_df(client: PolymarketClient) -> None:
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(200, json={"history": []}))
    df = client.get_price_history("111")
    assert df.empty
    assert list(df.columns) == ["date", "price"]


@respx.mock
def test_get_price_history_uses_daily_fidelity(client: PolymarketClient) -> None:
    route = respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": []})
    )
    client.get_price_history("111")
    request = route.calls.last.request
    assert request.url.params["fidelity"] == "1440"


@respx.mock
def test_get_price_history_passes_unix_timestamps(client: PolymarketClient) -> None:
    route = respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": []})
    )
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2025-12-31", tz="UTC")
    client.get_price_history("111", start=start, end=end)
    params = route.calls.last.request.url.params
    # startTs is sent; endTs is NOT (CLOB 400s on it — see polymarket.py).
    assert int(params["startTs"]) == int(start.timestamp())
    assert "endTs" not in params
    # interval=max is always sent — required by CLOB whenever start is given.
    assert params["interval"] == "max"


@respx.mock
def test_get_price_history_filters_end_client_side(client: PolymarketClient) -> None:
    # CLOB returns bars beyond ``end``; the client must trim them.
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(
            200,
            json={
                "history": [
                    {"t": 1735689600, "p": 0.30},  # 2025-01-01
                    {"t": 1735776000, "p": 0.31},  # 2025-01-02
                    {"t": 1735862400, "p": 0.32},  # 2025-01-03
                    {"t": 1735948800, "p": 0.33},  # 2025-01-04
                ]
            },
        )
    )
    df = client.get_price_history(
        "111",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-02", tz="UTC"),
    )
    assert len(df) == 2  # only Jan 1 and Jan 2 survive
    assert df["price"].tolist() == [0.30, 0.31]


@respx.mock
def test_fetch_factor_history_end_to_end(client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "x",
                    "question": "?",
                    "clobTokenIds": json.dumps(["111", "222"]),
                    "active": True,
                    "closed": False,
                }
            ],
        )
    )
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(
            200,
            json={"history": [{"t": 1706745600, "p": 0.5}]},
        )
    )
    df = fetch_factor_history(client, "x")
    assert "price" in df.columns
    assert df.index.name == "date"
    assert df["price"].iloc[0] == 0.5


@respx.mock
def test_clob_5xx_propagates(client: PolymarketClient) -> None:
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        client.get_price_history("111")
