"""Tests for the Polygon-backed earnings whisper integration.

The Polygon HTTP layer is mocked with respx; the Polymarket Gamma fetch
is short-circuited via ``earnings_whisper.fetch_gamma_market``.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import earnings_whisper
from pfm.cache_utils import get_cache
from pfm.earnings_whisper import (
    BEAT_LADDERS,
    CONSENSUS_EPS,
    _get_consensus_eps,
    compute_whisper,
    earnings_calendar,
    router,
)
from pfm.sources import polygon as polygon_src
from pfm.sources.polygon import (
    POLYGON_BASE_URL,
    PolygonClient,
    PolygonError,
    _reset_warning_latch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Reset every cache touched by these tests."""
    for ns in (
        "earnings_whisper",
        "earnings_whisper_dashboard",
        "earnings_calendar",
        "polygon_consensus",
        "polygon_calendar",
    ):
        get_cache(ns).clear()
    _reset_warning_latch()


@pytest.fixture
def fast_client() -> PolygonClient:
    """Polygon client with the rate-limit gap disabled for fast tests."""
    return PolygonClient(api_key="test-key", rate_limit_sleep=0.0)


@pytest.fixture
def with_polygon_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")


@pytest.fixture
def without_polygon_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)


@pytest.fixture
def fast_polygon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the module-level rate-limit constant so async paths stay fast."""
    monkeypatch.setattr(polygon_src, "_RATE_LIMIT_SLEEP", 0.0)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _market(prob: float) -> dict[str, Any]:
    return {
        "bestBid": max(0.0, prob - 0.005),
        "bestAsk": min(1.0, prob + 0.005),
        "lastTradePrice": prob,
        "volume24hr": 1_000.0,
    }


def _financials_payload(
    ticker: str = "NVDA",
    diluted_eps: float = 0.91,
    history: list[tuple[str, float, float | None]] | None = None,
) -> dict[str, Any]:
    """Return a minimal-but-realistic Polygon financials response.

    ``history`` is ``[(period, actual_eps, estimated_eps), ...]``. The
    first entry is what ``current_estimate`` resolves to.
    """
    if history is None:
        history = [
            ("2026Q1", diluted_eps, 0.85),
            ("2025Q4", 0.78, 0.75),
            ("2025Q3", 0.62, 0.60),
        ]
    results: list[dict[str, Any]] = []
    for i, (period, actual, est) in enumerate(history):
        results.append(
            {
                "tickers": [ticker],
                "fiscal_period": period,
                "filing_date": f"2026-04-{15 - i:02d}",
                "period_of_report_date": f"2026-04-{30 - i:02d}",
                "end_date": f"2026-04-{30 - i:02d}",
                "estimated_eps": est,
                "financials": {
                    "income_statement": {
                        "diluted_earnings_per_share": {"value": actual},
                    },
                },
            }
        )
    return {"status": "OK", "results": results}


# ---------------------------------------------------------------------------
# PolygonClient — direct unit tests
# ---------------------------------------------------------------------------


@respx.mock
def test_polygon_consensus_basic(fast_client: PolygonClient) -> None:
    route = respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(200, json=_financials_payload("NVDA", 0.91))
    )
    out = asyncio.run(fast_client.fetch_consensus_eps("NVDA"))
    asyncio.run(fast_client.close())

    assert route.called
    sent = route.calls[0].request
    assert "apiKey=test-key" in str(sent.url)
    assert "ticker=NVDA" in str(sent.url)
    assert out["ticker"] == "NVDA"
    assert out["current_estimate"] == pytest.approx(0.91)
    assert len(out["surprise_history"]) == 3
    first = out["surprise_history"][0]
    assert first["actual_eps"] == pytest.approx(0.91)
    assert first["estimated_eps"] == pytest.approx(0.85)
    assert first["surprise_pct"] == pytest.approx((0.91 - 0.85) / 0.85 * 100.0, abs=1e-2)


@respx.mock
def test_polygon_eps_history_limit(fast_client: PolygonClient) -> None:
    history = [(f"2026Q{i}", 0.5 + i * 0.1, 0.5) for i in range(8)]
    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(200, json=_financials_payload("AAPL", history=history))
    )
    rows = asyncio.run(fast_client.fetch_eps_history("AAPL", limit=4))
    asyncio.run(fast_client.close())
    assert len(rows) == 4


@respx.mock
def test_polygon_calendar_filters_window(fast_client: PolygonClient) -> None:
    today = date(2026, 5, 1)
    payload = {
        "status": "OK",
        "results": [
            {
                "tickers": ["NVDA"],
                "period_of_report_date": "2026-05-22",
                "estimated_eps": 0.85,
                "financials": {
                    "income_statement": {
                        "diluted_earnings_per_share": {"value": 0.91},
                    }
                },
            },
            {
                "tickers": ["AAPL"],
                "period_of_report_date": "2026-05-28",
                "estimated_eps": 1.50,
                "financials": {
                    "income_statement": {
                        "diluted_earnings_per_share": {"value": 1.55},
                    }
                },
            },
        ],
    }
    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(200, json=payload)
    )
    rows = asyncio.run(fast_client.fetch_earnings_calendar(today, today + timedelta(days=30)))
    asyncio.run(fast_client.close())
    tickers = {r["ticker"] for r in rows}
    assert tickers == {"NVDA", "AAPL"}
    nvda = next(r for r in rows if r["ticker"] == "NVDA")
    assert nvda["earnings_date"] == "2026-05-22"
    assert nvda["consensus_eps"] == pytest.approx(0.91)


@respx.mock
def test_polygon_5xx_then_success(fast_client: PolygonClient) -> None:
    """A single 5xx is retried; second attempt succeeds."""
    route = respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        side_effect=[
            httpx.Response(503, text="upstream"),
            httpx.Response(200, json=_financials_payload("NVDA", 0.91)),
        ]
    )
    out = asyncio.run(fast_client.fetch_consensus_eps("NVDA"))
    asyncio.run(fast_client.close())
    assert route.call_count == 2
    assert out["current_estimate"] == pytest.approx(0.91)


@respx.mock
def test_polygon_5xx_persistent_raises(fast_client: PolygonClient) -> None:
    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(503, text="upstream")
    )
    with pytest.raises(PolygonError):
        asyncio.run(fast_client.fetch_consensus_eps("NVDA"))
    asyncio.run(fast_client.close())


@respx.mock
def test_polygon_429_then_success(fast_client: PolygonClient) -> None:
    route = respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        side_effect=[
            httpx.Response(429, text="rate limited"),
            httpx.Response(200, json=_financials_payload("NVDA", 0.91)),
        ]
    )
    out = asyncio.run(fast_client.fetch_consensus_eps("NVDA"))
    asyncio.run(fast_client.close())
    assert route.call_count == 2
    assert out["current_estimate"] == pytest.approx(0.91)


@respx.mock
def test_polygon_429_persistent_raises(fast_client: PolygonClient) -> None:
    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(429, text="rate limited")
    )
    with pytest.raises(PolygonError, match="rate-limited"):
        asyncio.run(fast_client.fetch_consensus_eps("NVDA"))
    asyncio.run(fast_client.close())


def test_polygon_missing_key_raises() -> None:
    cli = PolygonClient(api_key=None, rate_limit_sleep=0.0)
    with pytest.raises(PolygonError, match="POLYGON_API_KEY"):
        asyncio.run(cli.fetch_consensus_eps("NVDA"))
    asyncio.run(cli.close())


@respx.mock
def test_polygon_consensus_caches_within_ttl(fast_client: PolygonClient) -> None:
    route = respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(200, json=_financials_payload("NVDA", 0.91))
    )
    asyncio.run(fast_client.fetch_consensus_eps("NVDA"))
    asyncio.run(fast_client.fetch_consensus_eps("NVDA"))
    asyncio.run(fast_client.close())
    # Cache must short-circuit the second call.
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# _get_consensus_eps — provenance routing
# ---------------------------------------------------------------------------


def test_consensus_falls_back_when_key_missing(without_polygon_key: None) -> None:
    eps, src = _get_consensus_eps("NVDA")
    assert eps == CONSENSUS_EPS["NVDA"]
    assert src == "hardcoded_snapshot"


def test_consensus_unknown_ticker_no_polygon(without_polygon_key: None) -> None:
    eps, src = _get_consensus_eps("ZZZZ")
    assert eps is None
    assert src == "unknown"


@respx.mock
def test_consensus_polygon_live_when_key_present(
    with_polygon_key: None, fast_polygon: None
) -> None:
    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(200, json=_financials_payload("NVDA", 0.93))
    )
    eps, src = _get_consensus_eps("NVDA", source="cached")
    assert eps == pytest.approx(0.93)
    assert src == "polygon_live"


@respx.mock
def test_consensus_polygon_5xx_falls_back(with_polygon_key: None, fast_polygon: None) -> None:
    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(503, text="boom")
    )
    eps, src = _get_consensus_eps("NVDA", source="cached")
    assert eps == CONSENSUS_EPS["NVDA"]
    assert src == "hardcoded_snapshot"


@respx.mock
def test_consensus_polygon_429_then_falls_back(with_polygon_key: None, fast_polygon: None) -> None:
    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(429, text="rate limited")
    )
    eps, src = _get_consensus_eps("NVDA", source="cached")
    # After retry exhaustion the wrapper swallows the error and falls back.
    assert eps == CONSENSUS_EPS["NVDA"]
    assert src == "hardcoded_snapshot"


def test_consensus_hardcoded_source_skips_polygon(
    with_polygon_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """source='hardcoded' must not hit Polygon even when key configured."""
    called: dict[str, int] = {"n": 0}

    async def _spy(*_a: Any, **_kw: Any) -> None:
        called["n"] += 1

    monkeypatch.setattr(polygon_src, "fetch_consensus_eps_or_none", _spy)
    eps, src = _get_consensus_eps("NVDA", source="hardcoded")
    assert eps == CONSENSUS_EPS["NVDA"]
    assert src == "hardcoded_snapshot"
    assert called["n"] == 0


# ---------------------------------------------------------------------------
# compute_whisper / dashboard backward-compat + new fields
# ---------------------------------------------------------------------------


def test_compute_whisper_returns_consensus_source_field(
    without_polygon_key: None,
) -> None:
    overrides: dict[str, dict[str, Any]] = {}
    for slug, _t in BEAT_LADDERS["NVDA"]:
        overrides[slug] = _market(0.40)
    out = compute_whisper("NVDA", date(2026, 5, 22), overrides=overrides)
    assert "consensus_source" in out
    assert out["consensus_source"] == "hardcoded_snapshot"


@respx.mock
def test_compute_whisper_polygon_live_provenance(
    with_polygon_key: None, fast_polygon: None
) -> None:
    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(200, json=_financials_payload("NVDA", 0.97))
    )
    overrides: dict[str, dict[str, Any]] = {}
    for slug, _t in BEAT_LADDERS["NVDA"]:
        overrides[slug] = _market(0.50)
    out = compute_whisper("NVDA", date(2026, 5, 22), overrides=overrides)
    assert out["consensus_source"] == "polygon_live"
    assert out["consensus_eps"] == pytest.approx(0.97, abs=1e-3)


# ---------------------------------------------------------------------------
# earnings_calendar helper + endpoint
# ---------------------------------------------------------------------------


def test_earnings_calendar_hardcoded_only(without_polygon_key: None) -> None:
    rows = earnings_calendar(days=365, source="hardcoded")
    tickers = {r["ticker"] for r in rows}
    # Every hardcoded ticker should appear with a populated consensus.
    assert tickers >= {"NVDA", "AAPL", "MSFT"}
    nvda_row = next(r for r in rows if r["ticker"] == "NVDA")
    assert nvda_row["consensus_eps"] == CONSENSUS_EPS["NVDA"]


@respx.mock
def test_earnings_calendar_endpoint_live(
    client: TestClient,
    with_polygon_key: None,
    fast_polygon: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = date.today()
    fwd = today + timedelta(days=10)
    payload = {
        "status": "OK",
        "results": [
            {
                "tickers": ["AVGO"],
                "period_of_report_date": fwd.isoformat(),
                "estimated_eps": 1.20,
                "financials": {
                    "income_statement": {
                        "diluted_earnings_per_share": {"value": 1.25},
                    }
                },
            }
        ],
    }
    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(
        return_value=httpx.Response(200, json=payload)
    )

    r = client.get("/alpha/earnings-calendar", params={"days": 30, "source": "live"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "live"
    tickers = {row["ticker"] for row in body["rows"]}
    assert "AVGO" in tickers  # came from Polygon
    avgo = next(row for row in body["rows"] if row["ticker"] == "AVGO")
    assert avgo["earnings_date"] == fwd.isoformat()
    assert avgo["consensus_eps"] == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Endpoint integration
# ---------------------------------------------------------------------------


def test_dashboard_endpoint_source_param_propagates(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    without_polygon_key: None,
) -> None:
    today = date.today()
    monkeypatch.setattr(
        earnings_whisper,
        "NEXT_EARNINGS",
        {"NVDA": today + timedelta(days=2)},
    )
    monkeypatch.setattr(earnings_whisper, "fetch_gamma_market", lambda *a, **k: _market(0.50))
    r = client.get(
        "/alpha/earnings-whisper-dashboard",
        params={"days": 14, "source": "hardcoded"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "hardcoded"
    assert body["rows"][0]["consensus_source"] == "hardcoded_snapshot"


@respx.mock
def test_dashboard_expansion_includes_polygon_tickers(
    client: TestClient,
    with_polygon_key: None,
    fast_polygon: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Polygon returns extra tickers, the dashboard must expand beyond the static 8."""
    today = date.today()

    # The dashboard makes two distinct Polygon calls per ticker requested
    # (calendar + per-ticker consensus). Mock both shapes with one route.
    extra_ticker = "AVGO"
    extra_date = today + timedelta(days=3)

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "ticker=AVGO" in url:
            return httpx.Response(200, json=_financials_payload(extra_ticker, 1.40))
        if "ticker=" in url and "ticker=AVGO" not in url:
            # Per-ticker consensus for hardcoded tickers.
            return httpx.Response(200, json=_financials_payload("X", 1.0))
        # Calendar query.
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "results": [
                    {
                        "tickers": [extra_ticker],
                        "period_of_report_date": extra_date.isoformat(),
                        "estimated_eps": 1.30,
                        "financials": {
                            "income_statement": {
                                "diluted_earnings_per_share": {"value": 1.40},
                            }
                        },
                    }
                ],
            },
        )

    respx.get(f"{POLYGON_BASE_URL}/vX/reference/financials").mock(side_effect=_handler)

    monkeypatch.setattr(
        earnings_whisper,
        "NEXT_EARNINGS",
        {"NVDA": today + timedelta(days=2)},
    )
    # Wire up a beat ladder for AVGO so compute_whisper has rungs to evaluate.
    avgo_ladder = [(f"avgo-beats-eps-q1-2026-{int(t * 100)}", t) for t in (0.0, 0.05)]
    monkeypatch.setitem(earnings_whisper.BEAT_LADDERS, extra_ticker, avgo_ladder)
    monkeypatch.setattr(earnings_whisper, "fetch_gamma_market", lambda *a, **k: _market(0.55))

    r = client.get(
        "/alpha/earnings-whisper-dashboard",
        params={"days": 30, "source": "cached"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    tickers = {row["ticker"] for row in body["rows"]}
    assert extra_ticker in tickers
    avgo_row = next(row for row in body["rows"] if row["ticker"] == extra_ticker)
    assert avgo_row["consensus_source"] == "polygon_live"
