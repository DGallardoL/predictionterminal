"""Coverage-targeted tests for ``pfm.sources.polymarket``.

These tests exercise branches the existing ``test_polymarket.py`` skips:

  - process-local metadata cache (hit + LRU-style eviction)
  - 429 retry path for ``get_market_metadata`` (both branches)
  - 429 retry path for ``discover_markets`` per-page
  - Gamma fallback to ``?closed=true`` (success + still-missing)
  - clobTokenIds list-too-short validation branch
  - context manager + ``close()`` ownership semantics
  - ``fetch_factor_history`` httpx-timeout retry path (retry-succeeds + retry-fails)
  - ``discover_markets`` keyword filter / volume gate / cache / pagination break
  - ``utc_now_unix`` smoke

All HTTP is mocked via respx — never hits the network.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import httpx
import pandas as pd
import pytest
import respx

from pfm.sources import polymarket as pm
from pfm.sources.polymarket import (
    MarketCandidate,
    PolymarketClient,
    PolymarketError,
    discover_markets,
    fetch_factor_history,
    utc_now_unix,
)

GAMMA = "https://gamma-cov.test"
CLOB = "https://clob-cov.test"


@pytest.fixture(autouse=True)
def _clear_module_caches() -> Iterator[None]:
    """Process-local caches leak state across tests; reset before AND after each."""
    pm._METADATA_CACHE.clear()
    pm._DISCOVER_CACHE.clear()
    yield
    pm._METADATA_CACHE.clear()
    pm._DISCOVER_CACHE.clear()


@pytest.fixture
def client() -> PolymarketClient:
    return PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())


def _market_payload(slug: str = "x", yes: str = "111", no: str = "222") -> dict:
    return {
        "slug": slug,
        "question": "?",
        "clobTokenIds": json.dumps([yes, no]),
        "startDate": "2025-01-01T00:00:00Z",
        "endDate": "2026-01-01T00:00:00Z",
        "closed": False,
        "active": True,
    }


# ---------------------------------------------------------------------------
# Construction + lifecycle
# ---------------------------------------------------------------------------


def test_owns_client_when_none_passed_closes_on_exit() -> None:
    c = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB)
    assert c._owns_client is True
    # context manager exits and triggers close()
    with c as ctx:
        assert ctx is c
    assert c._client.is_closed is True


def test_does_not_own_injected_client() -> None:
    injected = httpx.Client()
    c = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=injected)
    assert c._owns_client is False
    c.close()  # no-op on injected client
    assert injected.is_closed is False
    injected.close()


def test_strips_trailing_slashes_on_urls() -> None:
    c = PolymarketClient(gamma_url=GAMMA + "/", clob_url=CLOB + "//")
    assert c.gamma_url == GAMMA
    # rstrip strips ALL trailing slashes, both are removed
    assert c.clob_url == CLOB
    c.close()


# ---------------------------------------------------------------------------
# Metadata cache (hit + eviction)
# ---------------------------------------------------------------------------


@respx.mock
def test_metadata_cache_returns_cached_on_second_call(client: PolymarketClient) -> None:
    route = respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(200, json=[_market_payload("cached-slug")])
    )
    m1 = client.get_market_metadata("cached-slug")
    m2 = client.get_market_metadata("cached-slug")
    assert m1 is m2  # cache returns the same object
    assert route.call_count == 1  # second call did not refetch


@respx.mock
def test_metadata_cache_evicts_when_full(
    client: PolymarketClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Shrink the cap so eviction triggers without 4096 inserts.
    monkeypatch.setattr(pm, "_METADATA_CACHE_MAX_ENTRIES", 4)
    respx.get(f"{GAMMA}/markets").mock(
        side_effect=lambda req: httpx.Response(200, json=[_market_payload(req.url.params["slug"])])
    )
    # Pre-load 4 entries with monotonically increasing cached-at timestamps.
    for i in range(4):
        client.get_market_metadata(f"slug-{i}")
        time.sleep(0.001)  # ensure distinct timestamps for victim sort
    assert len(pm._METADATA_CACHE) == 4
    # Adding a 5th triggers eviction of oldest 25% (= 1 entry).
    client.get_market_metadata("slug-new")
    # Original oldest should be gone; newest should be present.
    assert "slug-0" not in pm._METADATA_CACHE
    assert "slug-new" in pm._METADATA_CACHE


# ---------------------------------------------------------------------------
# 429 retry paths
# ---------------------------------------------------------------------------


@respx.mock
def test_get_market_metadata_retries_once_on_429(
    client: PolymarketClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Skip the 1.5s sleep so the test is fast.
    monkeypatch.setattr(pm.time, "sleep", lambda _s: None)
    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(200, json=[_market_payload("retry-slug")]),
        ]
    )
    route = respx.get(f"{GAMMA}/markets").mock(side_effect=lambda req: next(responses))
    meta = client.get_market_metadata("retry-slug")
    assert meta.yes_token_id == "111"
    assert route.call_count == 2


@respx.mock
def test_get_market_metadata_two_429s_in_a_row_raises(
    client: PolymarketClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pm.time, "sleep", lambda _s: None)
    respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(429))
    with pytest.raises(httpx.HTTPStatusError):
        client.get_market_metadata("fail-slug")


# ---------------------------------------------------------------------------
# Closed-market fallback branch
# ---------------------------------------------------------------------------


@respx.mock
def test_metadata_falls_back_to_closed_true_when_empty(client: PolymarketClient) -> None:
    # First call (without ?closed=true) returns []; fallback succeeds.
    responses = iter(
        [
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[_market_payload("closed-mkt")]),
        ]
    )
    route = respx.get(f"{GAMMA}/markets").mock(side_effect=lambda req: next(responses))
    meta = client.get_market_metadata("closed-mkt")
    assert meta.slug == "closed-mkt"
    assert route.call_count == 2


@respx.mock
def test_metadata_fallback_still_missing_raises(client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json=[]))
    with pytest.raises(PolymarketError, match="no market found"):
        client.get_market_metadata("ghost-slug")


@respx.mock
def test_metadata_fallback_non_200_then_raises(client: PolymarketClient) -> None:
    # First call returns []; the ?closed=true retry returns a non-200, so the
    # function falls through and raises the "no market found" error.
    responses = iter([httpx.Response(200, json=[]), httpx.Response(503)])
    respx.get(f"{GAMMA}/markets").mock(side_effect=lambda req: next(responses))
    with pytest.raises(PolymarketError, match="no market found"):
        client.get_market_metadata("ghost-slug")


# ---------------------------------------------------------------------------
# clobTokenIds validation branches
# ---------------------------------------------------------------------------


@respx.mock
def test_metadata_missing_clob_token_ids_raises(client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(200, json=[{"slug": "x", "question": "?"}])
    )
    with pytest.raises(PolymarketError, match="no clobTokenIds"):
        client.get_market_metadata("x")


@respx.mock
def test_metadata_clob_token_ids_too_short_raises(client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[{"slug": "x", "question": "?", "clobTokenIds": json.dumps(["only-one"])}],
        )
    )
    with pytest.raises(PolymarketError, match=r"≥2 entries"):
        client.get_market_metadata("x")


@respx.mock
def test_metadata_clob_token_ids_not_a_list_raises(client: PolymarketClient) -> None:
    # JSON-valid but wrong shape (object, not list) — exercises the
    # `isinstance(token_ids, list)` arm.
    respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "x",
                    "question": "?",
                    "clobTokenIds": json.dumps({"yes": "111", "no": "222"}),
                }
            ],
        )
    )
    with pytest.raises(PolymarketError, match=r"≥2 entries"):
        client.get_market_metadata("x")


# ---------------------------------------------------------------------------
# fetch_factor_history retry-on-timeout
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_factor_history_retries_on_read_timeout(client: PolymarketClient) -> None:
    # First gamma call raises ReadTimeout; second succeeds. CLOB returns one bar.
    responses = iter(
        [
            httpx.ReadTimeout("simulated"),
            httpx.Response(200, json=[_market_payload("x")]),
        ]
    )
    respx.get(f"{GAMMA}/markets").mock(
        side_effect=lambda req: (
            lambda r: r if not isinstance(r, Exception) else (_ for _ in ()).throw(r)
        )(next(responses))
    )
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": [{"t": 1706745600, "p": 0.5}]})
    )
    df = fetch_factor_history(client, "x")
    assert df["price"].iloc[0] == 0.5


@respx.mock
def test_fetch_factor_history_raises_after_two_timeouts(client: PolymarketClient) -> None:
    respx.get(f"{GAMMA}/markets").mock(side_effect=httpx.ConnectTimeout("simulated"))
    with pytest.raises(httpx.ConnectTimeout):
        fetch_factor_history(client, "x")


@respx.mock
def test_fetch_factor_history_returns_indexed_df_when_empty(client: PolymarketClient) -> None:
    # Gamma returns metadata; CLOB returns no history. Function returns empty
    # DataFrame; we explicitly take the `df.empty` branch.
    respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(200, json=[_market_payload("x")])
    )
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(200, json={"history": []}))
    df = fetch_factor_history(client, "x")
    assert df.empty
    assert df.index.name == "date"


# ---------------------------------------------------------------------------
# get_price_history — start without tz (covers _to_unix tzinfo=None branch)
# ---------------------------------------------------------------------------


@respx.mock
def test_get_price_history_accepts_naive_start_timestamp(client: PolymarketClient) -> None:
    route = respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": []})
    )
    naive = pd.Timestamp("2025-06-01")  # no tz
    client.get_price_history("111", start=naive)
    sent = route.calls.last.request.url.params
    expected = int(naive.tz_localize("UTC").timestamp())
    assert int(sent["startTs"]) == expected


@respx.mock
def test_get_price_history_end_naive_filters_client_side(client: PolymarketClient) -> None:
    # Naive end timestamp exercises the `end.tz_localize("UTC")` branch.
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(
            200,
            json={
                "history": [
                    {"t": 1735689600, "p": 0.30},  # 2025-01-01
                    {"t": 1735862400, "p": 0.32},  # 2025-01-03
                ]
            },
        )
    )
    df = client.get_price_history("111", end=pd.Timestamp("2025-01-02"))
    assert len(df) == 1
    assert df["price"].iloc[0] == 0.30


# ---------------------------------------------------------------------------
# discover_markets — full path: keyword, gate, pagination, cache, 429
# ---------------------------------------------------------------------------


@respx.mock
def test_discover_markets_basic_keyword_and_volume_filter(client: PolymarketClient) -> None:
    page1 = [
        {
            "slug": "fed-rate-cut",
            "question": "Will Fed cut rates?",
            "volume": "5000000",
            "endDate": "2026-12-01T00:00:00Z",
            "active": True,
            "closed": False,
        },
        {
            "slug": "small-market",
            "question": "Small",
            "volume": "10",  # below min_volume
            "endDate": "2026-09-15T00:00:00Z",
        },
        {
            "slug": "election-2028",
            "question": "Who wins?",
            "volume": "3000000",
            "endDate": None,
        },
    ]
    respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json=page1))
    out = discover_markets(client, min_volume=1_000_000, limit=10, keyword="fed", pages=1)
    assert len(out) == 1
    assert out[0].slug == "fed-rate-cut"
    assert out[0].volume == 5_000_000
    assert out[0].end_date == "2026-12-01"


@respx.mock
def test_discover_markets_breaks_on_empty_page(client: PolymarketClient) -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json=[{"slug": "a", "question": "?", "volume": "5000000"}],
            ),
            httpx.Response(200, json=[]),  # break
        ]
    )
    respx.get(f"{GAMMA}/markets").mock(side_effect=lambda req: next(responses))
    out = discover_markets(client, min_volume=1_000_000, limit=10, pages=5)
    assert len(out) == 1


@respx.mock
def test_discover_markets_429_retries_once_per_page(
    client: PolymarketClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pm.time, "sleep", lambda _s: None)
    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(
                200,
                json=[{"slug": "a", "question": "?", "volume": "5000000"}],
            ),
            httpx.Response(200, json=[]),
        ]
    )
    route = respx.get(f"{GAMMA}/markets").mock(side_effect=lambda req: next(responses))
    out = discover_markets(client, min_volume=1_000_000, limit=10, pages=2)
    assert len(out) == 1
    assert route.call_count == 3  # 429 + retry + empty page


@respx.mock
def test_discover_markets_stops_when_limit_reached(client: PolymarketClient) -> None:
    page1 = [{"slug": f"s-{i}", "question": "q", "volume": "5000000"} for i in range(5)]
    respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json=page1))
    out = discover_markets(client, min_volume=1_000_000, limit=2, pages=1)
    assert len(out) == 2


@respx.mock
def test_discover_markets_skips_duplicate_slugs_and_bad_volume(client: PolymarketClient) -> None:
    page1 = [
        {"slug": "dup", "question": "q", "volume": "5000000"},
        {"slug": "dup", "question": "q", "volume": "6000000"},  # dedup
        {"slug": "no-slug-key", "question": "q"},  # skipped: slug present but volume default 0
        {"slug": "bad-vol", "question": "q", "volume": "not-a-number"},  # volume → 0.0
    ]
    respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json=page1))
    out = discover_markets(client, min_volume=1_000_000, limit=10, pages=1)
    assert [c.slug for c in out] == ["dup"]


@respx.mock
def test_discover_markets_skips_when_slug_missing(client: PolymarketClient) -> None:
    page1 = [
        {"question": "q", "volume": "5000000"},  # no slug key
        {"slug": "real", "question": "q", "volume": "5000000"},
    ]
    respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json=page1))
    out = discover_markets(client, min_volume=1_000_000, limit=10, pages=1)
    assert [c.slug for c in out] == ["real"]


@respx.mock
def test_discover_markets_cache_returns_same_result(client: PolymarketClient) -> None:
    route = respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(200, json=[{"slug": "a", "question": "?", "volume": "5000000"}])
    )
    first = discover_markets(client, min_volume=1_000_000, limit=10, pages=1)
    second = discover_markets(client, min_volume=1_000_000, limit=10, pages=1)
    assert [c.slug for c in first] == [c.slug for c in second]
    # The router walks one page then exits the loop normally; cache key
    # matches so a second call short-circuits.
    assert route.call_count <= 1


@respx.mock
def test_discover_markets_keyword_matches_question(client: PolymarketClient) -> None:
    page1 = [
        {
            "slug": "abc-2028",
            "question": "Will Fed hike?",
            "volume": "5000000",
        },
        {"slug": "irrelevant", "question": "?", "volume": "5000000"},
    ]
    respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json=page1))
    out = discover_markets(client, min_volume=1_000_000, limit=10, keyword="fed", pages=1)
    assert [c.slug for c in out] == ["abc-2028"]


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def test_utc_now_unix_is_recent_int() -> None:
    now = utc_now_unix()
    assert isinstance(now, int)
    assert abs(now - int(time.time())) < 5


def test_market_candidate_is_frozen() -> None:
    c = MarketCandidate(
        slug="s", question="q", volume=1.0, end_date=None, active=True, closed=False
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        c.slug = "x"  # type: ignore[misc]
