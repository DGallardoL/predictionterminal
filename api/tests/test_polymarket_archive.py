"""Tests for the Polymarket archive.

All upstream HTTP is mocked via :mod:`respx` — no real Gamma / CLOB calls.
The module-level cache lives in the ``archive_polymarket`` namespace; we
clear it around every test so cache hits don't mask call-count regressions.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.archive.polymarket_archive import (
    CLOB_URL,
    GAMMA_URL,
    archive_themes_distribution,
    fetch_archive_market_detail,
    fetch_resolved_markets,
)
from pfm.archive.resolutions import get_resolution
from pfm.archive.router import router as archive_router
from pfm.cache_utils import get_cache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_archive_cache() -> None:
    """Wipe the shared archive cache around every test."""
    get_cache("archive_polymarket").clear()
    yield
    get_cache("archive_polymarket").clear()


@pytest.fixture
def app_client() -> TestClient:
    """Throw-away FastAPI app with just the archive router mounted."""
    app = FastAPI()
    app.include_router(archive_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Sample payloads (factory functions so each test gets fresh dicts)
# ---------------------------------------------------------------------------


def _gamma_market(
    *,
    slug: str,
    question: str,
    end_date: str = "2024-11-06T00:00:00Z",
    closed: bool = True,
    yes_price: float = 1.0,
    no_price: float = 0.0,
    theme: str = "politics",
    volume: float = 1_500_000.0,
    traders: int = 4200,
    token_yes: str = "111",
    token_no: str = "222",
    extra: dict | None = None,
) -> dict:
    base = {
        "id": f"id-{slug}",
        "slug": slug,
        "question": question,
        "endDate": end_date,
        "startDate": "2024-01-15T00:00:00Z",
        "closed": closed,
        "active": False,
        "outcomePrices": json.dumps([yes_price, no_price]),
        "volume": volume,
        "traders": traders,
        "category": theme,
        "clobTokenIds": json.dumps([token_yes, token_no]),
        "lastTradePrice": yes_price,
        "topWalletsShare": 0.42,
    }
    if extra:
        base.update(extra)
    return base


def _clob_history(prices: list[float], start_unix: int = 1_700_000_000) -> dict:
    """Daily history payload with one bucket per price (86400s apart)."""
    return {"history": [{"t": start_unix + i * 86400, "p": p} for i, p in enumerate(prices)]}


# ---------------------------------------------------------------------------
# fetch_resolved_markets
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_resolved_markets_returns_normalized_rows() -> None:
    page = [
        _gamma_market(
            slug="trump-2024",
            question="Will Trump win 2024?",
            yes_price=1.0,
            no_price=0.0,
            theme="politics",
        ),
        _gamma_market(
            slug="btc-100k",
            question="Will BTC hit 100k?",
            yes_price=0.0,
            no_price=1.0,
            theme="crypto",
        ),
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=page))

    rows = fetch_resolved_markets(
        start_date=__import__("datetime").date(2024, 1, 1),
        end_date=__import__("datetime").date(2024, 12, 31),
        limit=10,
        offset=0,
    )

    assert len(rows) == 2
    by_slug = {r["slug"]: r for r in rows}
    assert by_slug["trump-2024"]["resolution"] == "YES"
    assert by_slug["btc-100k"]["resolution"] == "NO"
    assert by_slug["trump-2024"]["theme"] == "politics"
    assert by_slug["btc-100k"]["theme"] == "crypto"
    assert by_slug["trump-2024"]["total_volume"] == 1_500_000.0


@respx.mock
def test_fetch_resolved_markets_pagination_passes_offset_and_limit() -> None:
    """Verify limit + offset are forwarded to Gamma."""
    captured: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=[])

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_handler)

    fetch_resolved_markets(
        start_date=__import__("datetime").date(2024, 6, 1),
        end_date=__import__("datetime").date(2024, 12, 31),
        limit=25,
        offset=50,
    )

    assert len(captured) == 1
    qs = dict(captured[0].url.params)
    assert qs["limit"] == "25"
    assert qs["offset"] == "50"
    assert qs["closed"] == "true"
    assert qs["date_end_min"] == "2024-06-01"
    assert qs["date_end_max"] == "2024-12-31"


@respx.mock
def test_fetch_resolved_markets_theme_filters_client_side() -> None:
    """Theme filter is applied after Gamma returns; non-matching rows are dropped."""
    page = [
        _gamma_market(slug="m-pol-1", question="A", theme="politics"),
        _gamma_market(slug="m-cry-1", question="B", theme="crypto"),
        _gamma_market(slug="m-pol-2", question="C", theme="politics"),
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=page))

    rows = fetch_resolved_markets(
        start_date=__import__("datetime").date(2024, 1, 1),
        end_date=__import__("datetime").date(2024, 12, 31),
        theme="politics",
        limit=10,
    )
    assert {r["slug"] for r in rows} == {"m-pol-1", "m-pol-2"}
    assert all(r["theme"] == "politics" for r in rows)


# ---------------------------------------------------------------------------
# fetch_archive_market_detail
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_archive_market_detail_computes_stats() -> None:
    market = _gamma_market(
        slug="trump-2024",
        question="Will Trump win the 2024 election?",
        yes_price=1.0,
    )
    # Build a 60-day series that climbs from 0.4 to 0.95 — should give us
    # a peak at the end, a trough at the start, and a non-None half-life.
    prices = [0.40 + (0.55 * i / 59) for i in range(60)]
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "trump-2024"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )

    detail = fetch_archive_market_detail("trump-2024")

    assert detail["slug"] == "trump-2024"
    assert detail["resolution"] == "YES"
    assert detail["theme"] == "politics"
    assert detail["final_price"] == 1.0
    assert len(detail["history"]) == 60
    stats = detail["stats"]
    assert stats["peak_price"] == pytest.approx(0.95, abs=1e-6)
    assert stats["trough_price"] == pytest.approx(0.40, abs=1e-6)
    assert stats["half_life_to_resolution"] is not None
    assert stats["half_life_to_resolution"] >= 0
    assert stats["volatility_realized"] is not None
    assert stats["total_volume"] == 1_500_000.0
    assert stats["n_unique_traders"] == 4200
    assert stats["whale_concentration"] == pytest.approx(0.42, abs=1e-6)


@respx.mock
def test_fetch_archive_market_detail_404_when_missing() -> None:
    """No Gamma match (default + closed=true) → LookupError."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost", "closed": "true"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    with pytest.raises(LookupError):
        fetch_archive_market_detail("ghost")


# ---------------------------------------------------------------------------
# Resolution lookup
# ---------------------------------------------------------------------------


@respx.mock
def test_get_resolution_record_yes() -> None:
    market = _gamma_market(
        slug="abc",
        question="Will X happen?",
        yes_price=1.0,
        extra={"resolutionSource": "https://example.com/uma"},
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "abc"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    rec = get_resolution("abc")
    assert rec["resolution"] == "YES"
    assert rec["resolution_source"] == "https://example.com/uma"
    assert rec["payout_per_share"] == 1.0


@respx.mock
def test_get_resolution_record_ambiguous() -> None:
    market = _gamma_market(
        slug="disp",
        question="Was it disputed?",
        yes_price=0.5,
        no_price=0.5,
        extra={"umaResolutionStatuses": "disputed by signer"},
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "disp"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    rec = get_resolution("disp")
    assert rec["resolution"] == "AMBIGUOUS"
    assert any(d.get("kind") == "uma" for d in rec["dispute_history"])


# ---------------------------------------------------------------------------
# archive_themes_distribution
# ---------------------------------------------------------------------------


@respx.mock
def test_archive_themes_distribution_aggregates_by_theme() -> None:
    pages = [
        [
            _gamma_market(slug="p1", question="Q1", theme="politics", yes_price=1.0),
            _gamma_market(slug="p2", question="Q2", theme="politics", yes_price=0.0, no_price=1.0),
            _gamma_market(slug="c1", question="Q3", theme="crypto", yes_price=1.0),
        ],
        [],  # second page empty stops the walk
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=[httpx.Response(200, json=p) for p in pages])

    out = archive_themes_distribution(pages=2)

    by_theme = {row["theme"]: row for row in out["themes"]}
    assert by_theme["politics"]["n_markets"] == 2
    assert by_theme["politics"]["pct_yes"] == 0.5
    assert by_theme["politics"]["pct_no"] == 0.5
    assert by_theme["crypto"]["n_markets"] == 1
    assert by_theme["crypto"]["pct_yes"] == 1.0
    assert out["n_markets_total"] == 3


# ---------------------------------------------------------------------------
# Router endpoints
# ---------------------------------------------------------------------------


@respx.mock
def test_router_list_endpoint(app_client: TestClient) -> None:
    page = [_gamma_market(slug="a", question="A?", yes_price=1.0, theme="politics")]
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=page))
    resp = app_client.get("/archive/polymarket/markets?start=2024-01-01&end=2024-12-31&limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_markets"] == 1
    assert body["markets"][0]["slug"] == "a"


@respx.mock
def test_router_detail_endpoint_csv_export(app_client: TestClient) -> None:
    market = _gamma_market(slug="x-csv", question="X?", yes_price=1.0)
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "x-csv"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.4, 0.6, 0.8, 1.0]))
    )

    resp = app_client.get("/archive/polymarket/market/x-csv?format=csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    body = resp.text
    assert body.splitlines()[0] == "date,price,volume,sentiment"
    # 4 history rows + header
    assert len(body.strip().splitlines()) == 5


@respx.mock
def test_router_detail_endpoint_json(app_client: TestClient) -> None:
    market = _gamma_market(slug="x-json", question="X?", yes_price=1.0)
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "x-json"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.5, 0.7, 1.0]))
    )

    resp = app_client.get("/archive/polymarket/market/x-json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == "x-json"
    assert body["resolution"] == "YES"
    assert isinstance(body["history"], list)
    assert "stats" in body


@respx.mock
def test_router_detail_endpoint_404(app_client: TestClient) -> None:
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "nope"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "nope", "closed": "true"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    resp = app_client.get("/archive/polymarket/market/nope")
    assert resp.status_code == 404


@respx.mock
def test_router_themes_endpoint(app_client: TestClient) -> None:
    pages = [
        [_gamma_market(slug="p1", question="Q1", theme="politics", yes_price=1.0)],
        [],
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=[httpx.Response(200, json=p) for p in pages])
    resp = app_client.get("/archive/polymarket/themes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_markets_total"] >= 1
    assert any(t["theme"] == "politics" for t in body["themes"])


@respx.mock
def test_router_resolution_endpoint(app_client: TestClient) -> None:
    market = _gamma_market(slug="r1", question="R?", yes_price=1.0)
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "r1"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    resp = app_client.get("/archive/polymarket/resolutions/r1")
    assert resp.status_code == 200
    assert resp.json()["resolution"] == "YES"


@respx.mock
def test_router_search_endpoint(app_client: TestClient) -> None:
    pages = [
        [
            _gamma_market(slug="trump-2024", question="Will Trump win?", yes_price=1.0),
            _gamma_market(
                slug="btc-100k", question="Will BTC hit 100k?", yes_price=0.0, no_price=1.0
            ),
        ],
        [],
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=[httpx.Response(200, json=p) for p in pages])
    resp = app_client.get("/archive/polymarket/search?q=trump&limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_results"] == 1
    assert body["results"][0]["slug"] == "trump-2024"


@respx.mock
def test_router_export_bulk_csv_zip(app_client: TestClient) -> None:
    """POST /export-bulk returns a ZIP with one CSV per slug."""
    market_a = _gamma_market(slug="a", question="A?", yes_price=1.0, token_yes="aa", token_no="ab")
    market_b = _gamma_market(
        slug="b", question="B?", yes_price=0.0, no_price=1.0, token_yes="ba", token_no="bb"
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "a"}).mock(
        return_value=httpx.Response(200, json=[market_a])
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "b"}).mock(
        return_value=httpx.Response(200, json=[market_b])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.5, 0.7, 1.0]))
    )

    resp = app_client.post(
        "/archive/polymarket/export-bulk",
        json={"slugs": ["a", "b"], "format": "csv"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        names = sorted(zf.namelist())
        assert names == ["a.csv", "b.csv"]
        body_a = zf.read("a.csv").decode()
        assert body_a.splitlines()[0] == "date,price,volume"
