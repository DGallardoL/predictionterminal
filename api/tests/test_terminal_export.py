"""Tests for the universal CSV/JSON export helpers + bulk-export endpoint.

Coverage targets:
    * :mod:`pfm.terminal_export` — ``to_csv`` / ``to_json`` / ``respond`` for
      flat dicts, dicts with nested ``history``, plain lists and BaseModels.
    * The single-endpoint integration (``GET /terminal/market/{slug}?format=csv``)
      via the shared ``app_client`` fixture and ``respx`` for Gamma.
    * The bulk endpoint (``POST /terminal/export/bulk``) for CSV multi-section
      output, JSON output, and the PDF 501 stub.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import BaseModel

import pfm.terminal as terminal_mod
import pfm.terminal_bulk_export as bulk_mod
from pfm.terminal_export import PDF_AVAILABLE, respond, to_csv, to_json

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


# ──────────────────────────────────────────────────────────────────────────
# Pure-function tests for to_csv / to_json
# ──────────────────────────────────────────────────────────────────────────


def test_to_csv_simple_dict() -> None:
    payload = {"slug": "abc", "price": 0.42, "volume": 12345.6}
    csv = to_csv(payload)
    lines = csv.strip().splitlines()
    assert lines[0].split(",") == ["slug", "price", "volume"]
    assert lines[1].split(",") == ["abc", "0.42", "12345.6"]


def test_to_csv_dict_with_history_section() -> None:
    payload = {
        "slug": "abc",
        "fidelity": 1440,
        "n_bars": 2,
        "history": [
            {"t": 1735689600, "p": 0.40},
            {"t": 1735776000, "p": 0.42},
        ],
    }
    csv = to_csv(payload)
    # The scalar header should NOT include a "history" column — it was split.
    first_section, _, second_section = csv.partition("\n\n# section: history\n")
    assert "history" not in first_section.splitlines()[0]
    assert "slug" in first_section.splitlines()[0]
    # Second section is the time-series.
    sec_lines = second_section.strip().splitlines()
    assert sec_lines[0].split(",") == ["t", "p"]
    assert sec_lines[1].split(",") == ["1735689600", "0.4"]


def test_to_csv_list_of_dicts() -> None:
    payload = [
        {"slug": "a", "price": 0.1},
        {"slug": "b", "price": 0.9},
    ]
    csv = to_csv(payload)
    lines = csv.strip().splitlines()
    assert lines[0].split(",") == ["slug", "price"]
    assert len(lines) == 3  # header + 2 rows


def test_to_csv_basemodel() -> None:
    class Sample(BaseModel):
        slug: str
        price: float
        nested: dict

    obj = Sample(slug="abc", price=0.5, nested={"x": 1, "y": 2})
    csv = to_csv(obj)
    header = csv.strip().splitlines()[0].split(",")
    # json_normalize flattens nested dicts with dotted keys.
    assert "nested.x" in header
    assert "nested.y" in header
    assert "slug" in header


def test_to_csv_basemodel_with_history() -> None:
    class Bar(BaseModel):
        t: int
        p: float

    class Resp(BaseModel):
        slug: str
        n_bars: int
        history: list[Bar]

    obj = Resp(slug="abc", n_bars=2, history=[Bar(t=1, p=0.1), Bar(t=2, p=0.2)])
    csv = to_csv(obj)
    assert "# section: history" in csv
    # Scalar section came first.
    head = csv.split("\n\n# section: history")[0]
    assert "slug" in head and "n_bars" in head


def test_to_json_pretty_prints() -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    out = to_json(payload)
    # Indent=2 means the second line starts with two spaces.
    lines = out.splitlines()
    assert any(line.startswith("  ") for line in lines)
    # Round-trippable.
    assert json.loads(out) == payload


def test_to_json_basemodel() -> None:
    class Sample(BaseModel):
        slug: str
        price: float

    out = to_json(Sample(slug="abc", price=0.5))
    parsed = json.loads(out)
    assert parsed == {"slug": "abc", "price": 0.5}


def test_respond_pdf_when_unavailable_returns_501() -> None:
    """When the WeasyPrint native deps are missing, ``respond`` falls back to 501.

    The contract changed in v0.2: instead of a "Coming soon" stub,
    ``respond`` now actually tries to render and only returns 501 when
    the toolchain is unavailable.
    """
    if PDF_AVAILABLE:
        pytest.skip("PDF stack installed — 501 fallback not exercisable here")
    resp = respond({"x": 1}, "pdf", filename="thing", kind="market")
    assert resp.status_code == 501
    body = json.loads(resp.body.decode())
    assert "PDF export unavailable" in body["detail"]
    assert "weasyprint" in body["detail"].lower()
    assert body["kind"] == "market"


def test_respond_pdf_when_available_returns_pdf() -> None:
    """End-to-end: a real PDF comes back when the toolchain is present."""
    if not PDF_AVAILABLE:
        pytest.skip("WeasyPrint stack unavailable (install cairo, pango)")
    resp = respond(
        {"slug": "abc", "live": {"midpoint": 0.5}},
        "pdf",
        filename="thing",
        kind="market",
    )
    assert resp.status_code == 200
    assert resp.media_type == "application/pdf"
    assert resp.headers["content-disposition"] == 'attachment; filename="thing.pdf"'
    assert resp.body[:4] == b"%PDF"


def test_to_pdf_renders_market_payload() -> None:
    """Direct ``to_pdf`` smoke test (skipped without weasyprint native deps).

    ``importorskip`` triggers the import which itself raises ``OSError``
    on hosts missing libpango / libcairo, so we gate on ``PDF_AVAILABLE``
    *first* — that flag was set at module import time and is the only
    safe way to detect "stack present" without re-tripping the OSError.
    """
    if not PDF_AVAILABLE:
        pytest.skip("WeasyPrint stack unavailable (install cairo, pango)")
    from pfm.terminal_export import to_pdf

    payload = {
        "slug": "abc",
        "live": {"midpoint": 0.42, "best_bid": 0.40, "best_ask": 0.44},
        "stats": {"n_obs": 100, "mean": 0.41, "std": 0.05},
        "history": [{"t": 1, "p": 0.40}, {"t": 2, "p": 0.42}, {"t": 3, "p": 0.41}],
    }
    out = to_pdf(payload, "market", filename="market-abc")
    assert isinstance(out, bytes)
    assert out[:4] == b"%PDF"
    assert len(out) > 1000  # any real PDF is well above this


def test_to_pdf_unknown_kind_falls_back_to_market() -> None:
    if not PDF_AVAILABLE:
        pytest.skip("WeasyPrint stack unavailable (install cairo, pango)")
    from pfm.terminal_export import to_pdf

    out = to_pdf({"slug": "x"}, "totally-unknown-kind", filename="x")
    assert out[:4] == b"%PDF"


def test_respond_csv_sets_attachment_header() -> None:
    resp = respond({"x": 1}, "csv", filename="thing", kind="market")
    assert resp.status_code == 200
    assert resp.media_type == "text/csv"
    assert resp.headers["content-disposition"] == 'attachment; filename="thing.csv"'


def test_respond_json_returns_jsonresponse() -> None:
    resp = respond({"x": 1}, "json", filename="thing")
    assert resp.status_code == 200
    assert resp.media_type == "application/json"
    assert json.loads(resp.body.decode()) == {"x": 1}


# ──────────────────────────────────────────────────────────────────────────
# Integration: /terminal/market/{slug}?format=csv
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_terminal_cache() -> None:
    """Force every test to hit the real handler, not stale cache."""
    terminal_mod.TERMINAL_CACHE.clear()


@pytest.fixture
def patched_terminal_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the on-disk pickle + AH JSON with empty/in-memory fixtures."""
    monkeypatch.setattr(terminal_mod, "_load_factor_history_cache", lambda _path: {})
    monkeypatch.setattr(terminal_mod, "_load_ah_hits", lambda _path: [])


@respx.mock
def test_terminal_market_format_csv(app_client: TestClient, patched_terminal_data: None) -> None:
    respx.get(f"{GAMMA}/markets", params={"slug": "slug-a"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "slug-a",
                    "question": "Will A happen?",
                    "bestBid": 0.42,
                    "bestAsk": 0.46,
                    "lastTradePrice": 0.44,
                    "volume24hr": 12345.6,
                    "volumeNum": 99999.9,
                    "liquidity": 5000.0,
                    "liquidityNum": 5000.0,
                    "active": True,
                    "closed": False,
                    "outcomePrices": json.dumps(["0.44", "0.56"]),
                    "clobTokenIds": json.dumps(["111", "222"]),
                }
            ],
        )
    )
    r = app_client.get("/terminal/market/slug-a?format=csv")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    assert "market-slug-a.csv" in r.headers["content-disposition"]
    body = r.text
    # First line is a header containing slug and at least one nested live.* col.
    header = body.splitlines()[0]
    assert "slug" in header
    assert "live.midpoint" in header or "live.best_bid" in header


@respx.mock
def test_terminal_market_format_json_default_unchanged(
    app_client: TestClient, patched_terminal_data: None
) -> None:
    """Without ?format, the endpoint still returns JSON with the usual shape."""
    respx.get(f"{GAMMA}/markets", params={"slug": "slug-a"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "slug-a",
                    "question": "?",
                    "bestBid": 0.42,
                    "bestAsk": 0.46,
                    "lastTradePrice": 0.44,
                    "active": True,
                    "closed": False,
                    "outcomePrices": json.dumps(["0.44", "0.56"]),
                    "clobTokenIds": json.dumps(["111", "222"]),
                }
            ],
        )
    )
    r = app_client.get("/terminal/market/slug-a")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slug"] == "slug-a"
    assert "live" in body and "meta" in body


# ──────────────────────────────────────────────────────────────────────────
# Bulk export endpoint
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def patched_bulk_fetchers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace bulk_export's HTTP fetchers with deterministic in-memory data."""

    bank: dict[str, dict] = {
        "slug-a": {
            "slug": "slug-a",
            "bestBid": 0.40,
            "bestAsk": 0.42,
            "lastTradePrice": 0.41,
            "volume24hr": 1000.0,
            "liquidity": 500.0,
            "oneDayPriceChange": 0.01,
            "active": True,
            "closed": False,
            "clobTokenIds": json.dumps(["t1", "t1n"]),
        },
        "slug-b": {
            "slug": "slug-b",
            "bestBid": 0.60,
            "bestAsk": 0.62,
            "lastTradePrice": 0.61,
            "volume24hr": 2000.0,
            "liquidity": 800.0,
            "oneDayPriceChange": -0.02,
            "active": True,
            "closed": False,
            "clobTokenIds": json.dumps(["t2", "t2n"]),
        },
        "slug-c": {
            "slug": "slug-c",
            "bestBid": 0.10,
            "bestAsk": 0.12,
            "lastTradePrice": 0.11,
            "volume24hr": 50.0,
            "liquidity": 20.0,
            "oneDayPriceChange": 0.0,
            "active": True,
            "closed": False,
            "clobTokenIds": json.dumps(["t3", "t3n"]),
        },
    }
    history_bank: dict[str, list[dict]] = {
        "slug-a": [{"slug": "slug-a", "t": 1, "p": 0.40}, {"slug": "slug-a", "t": 2, "p": 0.41}],
        "slug-b": [{"slug": "slug-b", "t": 1, "p": 0.60}],
        "slug-c": [],
    }

    async def fake_gamma(_client, slug: str):
        return bank.get(slug)

    async def fake_history(_client, slug: str, _market):
        return history_bank.get(slug, [])

    monkeypatch.setattr(bulk_mod, "_fetch_gamma", fake_gamma)
    monkeypatch.setattr(bulk_mod, "_fetch_history", fake_history)


def test_bulk_csv_three_slugs_live_only(
    app_client: TestClient, patched_bulk_fetchers: None
) -> None:
    r = app_client.post(
        "/terminal/export/bulk",
        json={
            "slugs": ["slug-a", "slug-b", "slug-c"],
            "format": "csv",
            "scope": ["live"],
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert "bulk-export.csv" in r.headers["content-disposition"]
    body = r.text
    # Section header present, all three slugs listed.
    assert "# section: live" in body
    for slug in ("slug-a", "slug-b", "slug-c"):
        assert slug in body


def test_bulk_csv_full_scope_emits_all_sections(
    app_client: TestClient, patched_bulk_fetchers: None
) -> None:
    r = app_client.post(
        "/terminal/export/bulk",
        json={
            "slugs": ["slug-a", "slug-b"],
            "format": "csv",
            "scope": ["live", "stats", "history"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.text
    for header in ("# section: live", "# section: stats", "# section: history"):
        assert header in body, f"missing {header} in body"


def test_bulk_json_includes_results(app_client: TestClient, patched_bulk_fetchers: None) -> None:
    r = app_client.post(
        "/terminal/export/bulk",
        json={
            "slugs": ["slug-a", "slug-b"],
            "format": "json",
            "scope": ["live", "stats"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scope"] == ["live", "stats"]
    assert {entry["slug"] for entry in body["results"]} == {"slug-a", "slug-b"}
    for entry in body["results"]:
        assert "live" in entry
        assert "stats" in entry


def test_bulk_pdf_returns_501_when_unavailable(
    app_client: TestClient, patched_bulk_fetchers: None
) -> None:
    """If the WeasyPrint native deps are missing, bulk export falls back to 501."""
    if PDF_AVAILABLE:
        pytest.skip("PDF stack installed — 501 fallback not exercisable here")
    r = app_client.post(
        "/terminal/export/bulk",
        json={"slugs": ["slug-a"], "format": "pdf", "scope": ["live"]},
    )
    assert r.status_code == 501
    body = r.json()
    assert "PDF export unavailable" in body["detail"]


def test_bulk_pdf_returns_multipage_pdf(
    app_client: TestClient, patched_bulk_fetchers: None
) -> None:
    """Bulk ``format=pdf`` produces a real multi-page PDF (one section per slug)."""
    if not PDF_AVAILABLE:
        pytest.skip("WeasyPrint stack unavailable (install cairo, pango)")
    r = app_client.post(
        "/terminal/export/bulk",
        json={
            "slugs": ["slug-a", "slug-b", "slug-c"],
            "format": "pdf",
            "scope": ["live", "stats", "history"],
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/pdf")
    assert "bulk-export.pdf" in r.headers["content-disposition"]
    body = r.content
    assert body[:4] == b"%PDF"
    # PDF streams are compressed by WeasyPrint so the literal ``/Page`` byte
    # marker isn't visible. A multi-section render is meaningfully larger
    # than a single-section one — 5 KB is well above the empty-PDF floor of
    # ~2 KB while staying tolerant of the renderer compression ratio.
    assert len(body) > 5_000
    # And the trailer must mark a complete document.
    assert b"%%EOF" in body[-32:]


# ──────────────────────────────────────────────────────────────────────────
# Single-endpoint PDF integration via /terminal/market
# ──────────────────────────────────────────────────────────────────────────


@respx.mock
def test_terminal_market_format_pdf(app_client: TestClient, patched_terminal_data: None) -> None:
    """``GET /terminal/market/{slug}?format=pdf`` returns a real PDF body.

    Skipped if WeasyPrint's native deps aren't installed on this host.
    """
    if not PDF_AVAILABLE:
        pytest.skip("WeasyPrint stack unavailable (install cairo, pango)")
    respx.get(f"{GAMMA}/markets", params={"slug": "slug-a"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "slug-a",
                    "question": "Will A happen?",
                    "bestBid": 0.42,
                    "bestAsk": 0.46,
                    "lastTradePrice": 0.44,
                    "active": True,
                    "closed": False,
                    "outcomePrices": json.dumps(["0.44", "0.56"]),
                    "clobTokenIds": json.dumps(["111", "222"]),
                }
            ],
        )
    )
    r = app_client.get("/terminal/market/slug-a?format=pdf")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/pdf")
    assert "market-slug-a.pdf" in r.headers["content-disposition"]
    assert r.content[:4] == b"%PDF"


# ──────────────────────────────────────────────────────────────────────────
# Chart-as-PNG endpoint (/export/chart-png)
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def chart_app_client() -> TestClient:
    """Mini FastAPI app that mounts only the chart_export router.

    main.py is intentionally untouched per project convention, so the
    chart router is wired at test time. This also keeps the chart cache
    isolated from the main app's lifespan startup.
    """
    from fastapi import FastAPI

    from pfm.chart_export import _clear_cache_for_tests
    from pfm.chart_export import router as chart_router

    _clear_cache_for_tests()
    mini_app = FastAPI()
    mini_app.include_router(chart_router)
    return TestClient(mini_app)


def test_chart_png_line_returns_image(chart_app_client: TestClient) -> None:
    pytest.importorskip("matplotlib")
    body = {
        "title": "Test line chart",
        "x": list(range(10)),
        "y": [v * 0.1 for v in range(10)],
        "kind": "line",
        "width": 800,
        "height": 400,
    }
    r = chart_app_client.post("/export/chart-png", json=body)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/png")
    # PNG magic: 89 50 4e 47 0d 0a 1a 0a
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert r.headers.get("X-Cache") == "MISS"


def test_chart_png_bar_and_scatter(chart_app_client: TestClient) -> None:
    pytest.importorskip("matplotlib")
    base = {
        "title": "t",
        "x": [1, 2, 3, 4],
        "y": [0.1, 0.2, 0.3, 0.4],
        "width": 400,
        "height": 300,
    }
    for kind in ("bar", "scatter"):
        r = chart_app_client.post("/export/chart-png", json={**base, "kind": kind})
        assert r.status_code == 200, (kind, r.text)
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_png_cache_hit(chart_app_client: TestClient) -> None:
    """Two identical POSTs — second response advertises ``X-Cache: HIT``."""
    pytest.importorskip("matplotlib")
    body = {
        "title": "cached",
        "x": [0, 1, 2],
        "y": [0.0, 0.5, 1.0],
        "kind": "line",
    }
    r1 = chart_app_client.post("/export/chart-png", json=body)
    r2 = chart_app_client.post("/export/chart-png", json=body)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.headers.get("X-Cache") == "MISS"
    assert r2.headers.get("X-Cache") == "HIT"
    # Bytes are identical because they came from the cache.
    assert r1.content == r2.content


def test_chart_png_rejects_too_many_points(chart_app_client: TestClient) -> None:
    body = {
        "x": list(range(1001)),
        "y": [0.0] * 1001,
        "kind": "line",
    }
    r = chart_app_client.post("/export/chart-png", json=body)
    assert r.status_code == 422


def test_chart_png_rejects_oversized_canvas(chart_app_client: TestClient) -> None:
    body = {
        "x": [0, 1],
        "y": [0.0, 1.0],
        "kind": "line",
        "width": 10000,  # > 4096
        "height": 600,
    }
    r = chart_app_client.post("/export/chart-png", json=body)
    assert r.status_code == 422


def test_chart_png_xy_length_mismatch(chart_app_client: TestClient) -> None:
    body = {"x": [1, 2, 3], "y": [0.1, 0.2], "kind": "line"}
    r = chart_app_client.post("/export/chart-png", json=body)
    assert r.status_code == 422
