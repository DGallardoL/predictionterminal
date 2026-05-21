"""DEEP exhaustive tests for the past-events archive subsystem.

Covers:
- Polymarket archive listing / detail / themes / search / bulk export / CSV
- Kalshi settled-markets listing / detail / series distribution
- Cross-venue PM-vs-Kalshi divergence comparator
- Stats sanity: peak/trough, log-vol, Hurst, half-life
- Edge cases: empty history, single point, missing volume, disjoint dates

All upstream IO is mocked with respx; nothing hits the live network.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import zipfile
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.archive import kalshi_archive as ka
from pfm.archive.cross_venue_archive import (
    CACHE_NS as CV_CACHE_NS,
)
from pfm.archive.cross_venue_archive import (
    cross_venue_resolved_pairs,
    list_concepts,
)
from pfm.archive.kalshi_archive import (
    KALSHI_BASE_URL,
    fetch_archive_kalshi_detail,
    fetch_settled_markets,
)
from pfm.archive.kalshi_router import router as kalshi_router
from pfm.archive.polymarket_archive import (
    CLOB_URL,
    GAMMA_URL,
    archive_themes_distribution,
    fetch_archive_market_detail,
    fetch_resolved_markets,
    search_archive,
)
from pfm.archive.resolutions import get_resolution
from pfm.archive.router import router as polymarket_router
from pfm.cache_utils import get_cache
from pfm.sources.kalshi import KalshiClient

# ---------------------------------------------------------------------------
# Cache reset around every test to avoid bleed-over.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_archive_caches():
    for ns in ("archive_polymarket", ka.ARCHIVE_CACHE_NS, CV_CACHE_NS):
        get_cache(ns).clear()
    yield
    for ns in ("archive_polymarket", ka.ARCHIVE_CACHE_NS, CV_CACHE_NS):
        get_cache(ns).clear()


# ---------------------------------------------------------------------------
# Fixtures: factory builders for Gamma + CLOB + Kalshi payloads.
# ---------------------------------------------------------------------------


def _gamma_market(
    *,
    slug: str,
    question: str = "Q?",
    end_date: str = "2024-11-06T00:00:00Z",
    start_date: str = "2024-01-15T00:00:00Z",
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
        "startDate": start_date,
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
    return {"history": [{"t": start_unix + i * 86400, "p": p} for i, p in enumerate(prices)]}


def _kalshi_market_row(ticker: str, *, series: str | None = None, **extra) -> dict:
    return {
        "ticker": ticker,
        "event_ticker": series or ticker.split("-", 1)[0],
        "title": extra.get("title", f"Will {ticker}?"),
        "settle_time": extra.get("settle_time", "2024-11-06T00:00:00Z"),
        "close_time": extra.get("close_time", "2024-11-06T00:00:00Z"),
        "result": extra.get("result", "yes"),
        "open_interest": extra.get("open_interest", 1000),
        "volume": extra.get("volume", 5000),
        "last_price": extra.get("last_price", 78),
    }


def _kalshi_candles(prices: list[tuple[int, float, float]]) -> dict:
    return {
        "candlesticks": [
            {
                "end_period_ts": ts,
                "price": {"close_dollars": close},
                "yes_bid": {"close_dollars": max(0.0, close - 0.01)},
                "yes_ask": {"close_dollars": min(1.0, close + 0.01)},
                "volume_fp": vol,
                "open_interest_fp": 100.0,
            }
            for ts, close, vol in prices
        ]
    }


@pytest.fixture
def poly_app() -> TestClient:
    app = FastAPI()
    app.include_router(polymarket_router)
    return TestClient(app)


@pytest.fixture
def kalshi_app() -> TestClient:
    app = FastAPI()
    app.include_router(kalshi_router)
    return TestClient(app)


# ===========================================================================
# 1) Polymarket archive listing
# ===========================================================================


@respx.mock
def test_pm_listing_returns_paginated_rows() -> None:
    page = [_gamma_market(slug=f"m-{i}", question=f"Q{i}") for i in range(20)]
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=page))
    rows = fetch_resolved_markets(date(2024, 1, 1), date(2024, 12, 31), limit=20)
    assert len(rows) == 20
    assert all(r["slug"].startswith("m-") for r in rows)


@respx.mock
def test_pm_listing_offset_passed_through() -> None:
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.url.params))
        return httpx.Response(200, json=[])

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=handler)
    fetch_resolved_markets(date(2024, 1, 1), date(2024, 12, 31), limit=20, offset=20)
    assert captured[0]["offset"] == "20"
    assert captured[0]["limit"] == "20"


@respx.mock
def test_pm_listing_theme_filter() -> None:
    page = [
        _gamma_market(slug="p1", theme="politics"),
        _gamma_market(slug="c1", theme="crypto"),
        _gamma_market(slug="p2", theme="politics"),
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=page))
    rows = fetch_resolved_markets(date(2024, 1, 1), date(2024, 12, 31), theme="politics", limit=10)
    assert {r["slug"] for r in rows} == {"p1", "p2"}
    assert all(r["theme"] == "politics" for r in rows)


@respx.mock
def test_pm_listing_sort_order_descending_default() -> None:
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.url.params))
        return httpx.Response(200, json=[])

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=handler)
    fetch_resolved_markets(date(2024, 1, 1), date(2024, 12, 31))
    assert captured[0]["order"] == "endDate"
    assert captured[0]["ascending"] == "false"


@respx.mock
def test_pm_listing_cache_hit_on_second_call() -> None:
    page = [_gamma_market(slug="cached")]
    route = respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=page))
    out1 = fetch_resolved_markets(date(2024, 1, 1), date(2024, 12, 31), limit=10)
    out2 = fetch_resolved_markets(date(2024, 1, 1), date(2024, 12, 31), limit=10)
    assert out1 == out2
    assert route.call_count == 1


@respx.mock
def test_pm_listing_resolution_distribution_sensible() -> None:
    page = [
        _gamma_market(slug="y1", yes_price=1.0, no_price=0.0),
        _gamma_market(slug="y2", yes_price=0.99, no_price=0.01),
        _gamma_market(slug="n1", yes_price=0.0, no_price=1.0),
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=page))
    rows = fetch_resolved_markets(date(2024, 1, 1), date(2024, 12, 31))
    n_yes = sum(1 for r in rows if r["resolution"] == "YES")
    n_no = sum(1 for r in rows if r["resolution"] == "NO")
    assert n_yes == 2 and n_no == 1


@respx.mock
def test_pm_listing_market_without_close_is_pending() -> None:
    page = [_gamma_market(slug="pending", closed=False)]
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=page))
    rows = fetch_resolved_markets(date(2024, 1, 1), date(2024, 12, 31))
    assert rows[0]["resolution"] == "PENDING"


# ===========================================================================
# 2) Polymarket detail + stats
# ===========================================================================


@respx.mock
def test_pm_detail_full_stats_yes_resolution() -> None:
    market = _gamma_market(slug="trump-2024", yes_price=1.0)
    # Monotonic series 0.40 -> 0.99: triggers half_life path (>=0.5 then >=0.9).
    prices = [0.40 + (0.59 * i / 59) for i in range(60)]
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "trump-2024"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )
    detail = fetch_archive_market_detail("trump-2024")

    assert detail["resolution"] == "YES"
    s = detail["stats"]
    assert s["peak_price"] == pytest.approx(0.99, abs=1e-6)
    assert s["trough_price"] == pytest.approx(0.40, abs=1e-6)
    assert s["peak_price"] > s["trough_price"]
    assert s["half_life_to_resolution"] is not None
    assert s["half_life_to_resolution"] >= 0
    assert s["volatility_realized"] is not None and s["volatility_realized"] > 0
    # whale_concentration is the topWalletsShare we set (0.42).
    assert 0.0 <= s["whale_concentration"] <= 1.0
    assert s["n_unique_traders"] == 4200

    # Verify history list of [date, price, vol], sorted ascending.
    history = detail["history"]
    assert len(history) == 60
    dates = [h[0] for h in history]
    assert dates == sorted(dates)


@respx.mock
def test_pm_detail_404_for_missing_slug() -> None:
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost", "closed": "true"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    with pytest.raises(LookupError) as exc:
        fetch_archive_market_detail("ghost")
    assert "ghost" in str(exc.value)


@respx.mock
def test_pm_detail_volatility_matches_numpy_std() -> None:
    """The realized vol the archive returns must equal numpy stdev of log
    returns times sqrt(365), within reasonable rounding tolerance."""
    market = _gamma_market(slug="vol-check")
    rng = np.random.default_rng(42)
    # Smooth random walk in (0,1) so log returns are stable.
    noise = rng.normal(0, 0.02, 60)
    prices = list(np.clip(0.5 + np.cumsum(noise) * 0.05, 0.05, 0.95))

    respx.get(f"{GAMMA_URL}/markets", params={"slug": "vol-check"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )
    detail = fetch_archive_market_detail("vol-check")

    px = pd.Series(prices).clip(lower=1e-4)
    expected = float(np.log(px / px.shift(1)).dropna().std(ddof=1) * math.sqrt(365))
    got = detail["stats"]["volatility_realized"]
    assert got == pytest.approx(expected, rel=1e-6)


@respx.mock
def test_pm_detail_hurst_defined_and_finite() -> None:
    """Hurst exponent should be a finite float, returned for series ≥30 obs.

    Note: ``_hurst`` operates on first-differences. Strictly linear price
    series produce zero-variance diffs and a degenerate R/S, so we use a
    noisy random walk — what actually tests the estimator's domain.
    """
    market = _gamma_market(slug="hurst-mon")
    rng = np.random.default_rng(7)
    # Random walk with moderate volatility — gives well-defined R/S.
    steps = rng.normal(0, 0.02, 80)
    prices = list(np.clip(0.5 + np.cumsum(steps), 0.05, 0.95))
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "hurst-mon"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )
    detail = fetch_archive_market_detail("hurst-mon")
    h = detail["stats"]["hurst_exponent"]
    assert h is not None
    assert math.isfinite(h)
    # R/S Hurst on small samples lives roughly in [0, 1.5].
    assert 0.0 <= h <= 1.5


@respx.mock
def test_pm_detail_peak_trough_within_history() -> None:
    market = _gamma_market(slug="peak-check")
    prices = [0.30, 0.45, 0.20, 0.85, 0.60, 0.99, 0.15, 0.50]
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "peak-check"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )
    detail = fetch_archive_market_detail("peak-check")
    s = detail["stats"]
    assert s["peak_price"] == pytest.approx(max(prices))
    assert s["trough_price"] == pytest.approx(min(prices))
    assert s["peak_price"] >= s["trough_price"]


@respx.mock
def test_pm_detail_half_life_only_for_yes_resolution() -> None:
    """If resolution != YES the half_life field is None."""
    market = _gamma_market(slug="half-no", yes_price=0.0, no_price=1.0)
    prices = [0.50 + 0.01 * i for i in range(40)]
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "half-no"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )
    detail = fetch_archive_market_detail("half-no")
    assert detail["resolution"] == "NO"
    assert detail["stats"]["half_life_to_resolution"] is None


@respx.mock
def test_pm_detail_history_is_list_of_tuples_sorted() -> None:
    market = _gamma_market(slug="sorted")
    # Provide unsorted timestamps so we can verify sort happens.
    payload = {
        "history": [{"t": 1_700_000_000 + 86400 * i, "p": 0.4 + 0.01 * i} for i in [3, 1, 0, 2, 4]]
    }
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "sorted"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(return_value=httpx.Response(200, json=payload))
    detail = fetch_archive_market_detail("sorted")
    dates = [row[0] for row in detail["history"]]
    assert dates == sorted(dates)


# ===========================================================================
# 3) Polymarket themes distribution
# ===========================================================================


@respx.mock
def test_pm_themes_distribution_groups_and_pcts_sum() -> None:
    page = [
        _gamma_market(slug="p1", theme="politics", yes_price=1.0),
        _gamma_market(slug="p2", theme="politics", yes_price=0.0, no_price=1.0),
        _gamma_market(slug="p3", theme="politics", extra={"umaResolutionStatuses": "disputed"}),
        _gamma_market(slug="c1", theme="crypto", yes_price=1.0),
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=[
            httpx.Response(200, json=page),
            httpx.Response(200, json=[]),
        ]
    )
    out = archive_themes_distribution(pages=2)
    by_theme = {t["theme"]: t for t in out["themes"]}
    pol = by_theme["politics"]
    # 1 YES + 1 NO + 1 AMBIGUOUS over 3 → pcts ≈ 1/3 each, sum ≤ 1.
    assert pol["n_markets"] == 3
    total_pct = pol["pct_yes"] + pol["pct_no"] + pol["pct_ambiguous"]
    assert total_pct == pytest.approx(1.0, abs=0.01)
    assert pol["avg_duration_days"] is not None and pol["avg_duration_days"] > 0
    assert out["n_markets_total"] == 4
    # Sum of n_markets across themes equals total resolved
    assert sum(t["n_markets"] for t in out["themes"]) == out["n_markets_total"]


# ===========================================================================
# 4) Kalshi settled markets
# ===========================================================================


@respx.mock
def test_kalshi_listing_paginates_and_normalizes() -> None:
    page1 = {
        "markets": [
            _kalshi_market_row("KXFEDDECISION-24SEP-C50", series="KXFEDDECISION"),
            _kalshi_market_row("KXFEDDECISION-24NOV-C25", series="KXFEDDECISION", result="no"),
        ],
        "cursor": "page2",
    }
    page2 = {"markets": [_kalshi_market_row("PRES-2024-DJT", series="PRES")], "cursor": ""}
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(
        side_effect=[
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]
    )
    rows = asyncio.run(fetch_settled_markets(limit=10))
    assert len(rows) == 3
    assert rows[0]["settle_value"] == "YES"
    assert rows[1]["settle_value"] == "NO"
    assert rows[0]["last_trade_price"] == 0.78  # 78c → $0.78


@respx.mock
def test_kalshi_listing_series_filter() -> None:
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.url.params))
        return httpx.Response(
            200,
            json={
                "markets": [
                    _kalshi_market_row("KXFOMC-24SEP", series="KXFOMC"),
                    _kalshi_market_row("KXCPI-24DEC", series="KXCPI"),
                ],
                "cursor": "",
            },
        )

    respx.get(f"{KALSHI_BASE_URL}/markets").mock(side_effect=handler)
    rows = asyncio.run(fetch_settled_markets(series_ticker="KXFOMC", limit=10))
    assert len(rows) == 1
    assert rows[0]["series"] == "KXFOMC"
    assert captured[0]["series_ticker"] == "KXFOMC"


@respx.mock
def test_kalshi_listing_empty_when_no_markets() -> None:
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(
        return_value=httpx.Response(200, json={"markets": [], "cursor": ""})
    )
    rows = asyncio.run(
        fetch_settled_markets(start_date=date(2030, 1, 1), end_date=date(2030, 12, 31))
    )
    assert rows == []


@respx.mock
def test_kalshi_detail_stats_consistent() -> None:
    ticker = "KXFEDDECISION-24SEP-C50"
    series = "KXFEDDECISION"
    respx.get(f"{KALSHI_BASE_URL}/markets/{ticker}").mock(
        return_value=httpx.Response(
            200,
            json={
                "market": {
                    "ticker": ticker,
                    "event_ticker": series + "-24SEP",
                    "title": "Fed >=50bps Sep 2024?",
                    "status": "settled",
                    "open_time": "2024-06-01T00:00:00Z",
                    "close_time": "2024-09-18T18:00:00Z",
                    "settle_time": "2024-09-18T18:00:00Z",
                    "result": "yes",
                    "n_traders": 432,
                    "top_wallets": ["0xaaa", "0xbbb"],
                    "volume": 999,
                }
            },
        )
    )
    candles = [
        (1717200000, 0.40, 200.0),
        (1717286400, 0.55, 300.0),
        (1726617600, 0.90, 500.0),
        (1726704000, 0.85, 400.0),
        (1726790400, 0.95, 600.0),
    ]
    respx.get(f"{KALSHI_BASE_URL}/series/{series}/markets/{ticker}/candlesticks").mock(
        return_value=httpx.Response(200, json=_kalshi_candles(candles))
    )

    http = httpx.Client()
    kc = KalshiClient(client=http, min_interval_s=0.0, max_retries=0)
    detail = fetch_archive_kalshi_detail(ticker, kalshi_client=kc, http_client=http)

    s = detail["stats"]
    assert s["peak_price"] == 0.95
    assert s["trough_price"] == 0.40
    assert s["peak_price"] >= s["trough_price"]
    assert s["realized_vol"] is not None and s["realized_vol"] > 0
    assert s["n_days"] == 5
    assert detail["settle_value"] == "YES"


# ===========================================================================
# 5) Cross-venue resolved pairs
# ===========================================================================


def _frame(prices: list[tuple[str, float]]) -> pd.DataFrame:
    idx = pd.to_datetime([d for d, _ in prices], utc=True).normalize()
    df = pd.DataFrame({"price": [p for _, p in prices]}, index=idx)
    df.index.name = "date"
    return df


def _make_election_pair() -> tuple:
    """PM > Kalshi throughout. Days with spread > 0.05: indices 2 and 3."""
    poly_df = _frame(
        [
            ("2024-09-01", 0.53),
            ("2024-09-02", 0.58),
            ("2024-09-03", 0.70),  # spread 0.10
            ("2024-09-04", 0.65),  # spread 0.10
            ("2024-11-05", 0.83),
            ("2024-11-06", 0.99),
        ]
    )
    kalshi_df = _frame(
        [
            ("2024-09-01", 0.50),
            ("2024-09-02", 0.55),
            ("2024-09-03", 0.60),
            ("2024-09-04", 0.55),
            ("2024-11-05", 0.80),
            ("2024-11-06", 0.98),
        ]
    )
    return (
        lambda *_a, **_k: poly_df,
        lambda *_a, **_k: kalshi_df,
    )


def test_cv_list_concepts_returns_5() -> None:
    items = list_concepts()
    assert len(items) == 5
    expected = {
        "presidential_election_2024",
        "recession_2024",
        "fed_first_cut_2024",
        "btc_70k_2024",
        "cpi_above_3_2024",
    }
    assert {i["concept"] for i in items} == expected
    for it in items:
        assert it["polymarket_slug"]
        assert it["kalshi_ticker"]
        assert it["resolved_outcome"] in {"YES", "NO"}


def test_cv_pairs_election_metrics_sane() -> None:
    poly, kalshi = _make_election_pair()
    out = cross_venue_resolved_pairs(
        "presidential_election_2024",
        polymarket_history=poly,
        kalshi_history_fn=kalshi,
    )
    assert out["error"] is None
    assert out["n_overlap_days"] == 6
    assert out["spread_at_resolution"] == pytest.approx(0.01)
    assert out["max_spread_observed"] == pytest.approx(0.10)
    assert out["days_diverged"] == 2  # only 2 days where |spread|>0.05
    assert 0.0 <= out["pct_time_pm_higher"] <= 1.0
    assert out["pct_time_pm_higher"] == pytest.approx(1.0)


def test_cv_pairs_unknown_concept_raises() -> None:
    with pytest.raises(KeyError):
        cross_venue_resolved_pairs("non-existent-concept-123")


def test_cv_pairs_empty_history_returns_error_payload() -> None:
    empty = pd.DataFrame(columns=["price"])
    out = cross_venue_resolved_pairs(
        "recession_2024",
        polymarket_history=lambda *_a, **_k: empty,
        kalshi_history_fn=lambda *_a, **_k: empty,
    )
    assert out["error"] is not None
    assert out["n_overlap_days"] == 0


# ===========================================================================
# 6) CSV export (single market)
# ===========================================================================


@respx.mock
def test_csv_export_header_and_parseable(poly_app: TestClient) -> None:
    market = _gamma_market(slug="csv-test")
    prices = [0.5, 0.6, 0.8, 1.0]
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "csv-test"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )
    resp = poly_app.get("/archive/polymarket/market/csv-test?format=csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    lines = resp.text.splitlines()
    assert lines[0] == "date,price,volume,sentiment"
    # Verify every column present and parseable
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = list(reader)
    assert len(rows) == len(prices)
    for r in rows:
        assert "date" in r and "price" in r
        assert "volume" in r and "sentiment" in r


# ===========================================================================
# 7) Bulk export ZIP
# ===========================================================================


@respx.mock
def test_bulk_export_csv_zip_three_slugs(poly_app: TestClient) -> None:
    slugs = ["a", "b", "c"]
    for s in slugs:
        respx.get(f"{GAMMA_URL}/markets", params={"slug": s}).mock(
            return_value=httpx.Response(
                200, json=[_gamma_market(slug=s, token_yes=f"{s}1", token_no=f"{s}2")]
            )
        )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.4, 0.6, 0.8]))
    )

    resp = poly_app.post(
        "/archive/polymarket/export-bulk",
        json={"slugs": slugs, "format": "csv"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = sorted(zf.namelist())
        assert names == ["a.csv", "b.csv", "c.csv"]
        for n in names:
            body = zf.read(n).decode()
            assert body.splitlines()[0] == "date,price,volume"
            # Parseable
            list(csv.DictReader(io.StringIO(body)))


@respx.mock
def test_bulk_export_json_format(poly_app: TestClient) -> None:
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "j1"}).mock(
        return_value=httpx.Response(200, json=[_gamma_market(slug="j1")])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.5, 0.7]))
    )
    resp = poly_app.post(
        "/archive/polymarket/export-bulk",
        json={"slugs": ["j1"], "format": "json"},
    )
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        assert names == ["j1.json"]
        body = json.loads(zf.read("j1.json"))
        assert body["slug"] == "j1"
        assert "history" in body


@respx.mock
def test_bulk_export_parquet_falls_back_or_succeeds(poly_app: TestClient) -> None:
    """parquet should not crash even when pyarrow is missing — the route
    falls back to CSV silently per docstring."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "pq1"}).mock(
        return_value=httpx.Response(200, json=[_gamma_market(slug="pq1")])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.5, 0.7]))
    )
    resp = poly_app.post(
        "/archive/polymarket/export-bulk",
        json={"slugs": ["pq1"], "format": "parquet"},
    )
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        assert len(names) == 1
        # Either pq1.parquet or pq1.csv depending on pyarrow availability.
        assert names[0] in {"pq1.parquet", "pq1.csv"}


@respx.mock
def test_bulk_export_failure_isolation(poly_app: TestClient) -> None:
    """One bad slug must NOT prevent successful exports for the others."""
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "good1"}).mock(
        return_value=httpx.Response(200, json=[_gamma_market(slug="good1")])
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "good2"}).mock(
        return_value=httpx.Response(200, json=[_gamma_market(slug="good2")])
    )
    # Bad slug: both queries return [].
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "bad"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "bad", "closed": "true"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.5]))
    )
    resp = poly_app.post(
        "/archive/polymarket/export-bulk",
        json={"slugs": ["good1", "bad", "good2"], "format": "csv"},
    )
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = sorted(zf.namelist())
        assert "good1.csv" in names
        assert "good2.csv" in names
        assert "ERROR-bad.txt" in names


# ===========================================================================
# 8) Search
# ===========================================================================


@respx.mock
def test_search_archive_substring_case_insensitive() -> None:
    page = [
        _gamma_market(slug="trump-2024", question="Will TRUMP win?"),
        _gamma_market(slug="biden-2024", question="Will Biden win?"),
        _gamma_market(slug="btc-100k", question="Will BTC hit 100k?"),
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=[
            httpx.Response(200, json=page),
            httpx.Response(200, json=[]),
        ]
    )
    rows = search_archive("trump", limit=25)
    assert len(rows) == 1
    assert rows[0]["slug"] == "trump-2024"


@respx.mock
def test_search_archive_empty_query_returns_empty() -> None:
    rows = search_archive("", limit=10)
    assert rows == []


@respx.mock
def test_search_archive_limit_respected() -> None:
    page = [_gamma_market(slug=f"trump-{i}", question=f"Trump scenario {i}") for i in range(10)]
    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=[
            httpx.Response(200, json=page),
            httpx.Response(200, json=[]),
        ]
    )
    rows = search_archive("trump", limit=3)
    assert len(rows) == 3


@respx.mock
def test_search_archive_special_characters_handled() -> None:
    page = [_gamma_market(slug="abc-2024", question="Will it happen?")]
    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=[
            httpx.Response(200, json=page),
            httpx.Response(200, json=[]),
        ]
    )
    # Special chars shouldn't crash; just no match expected.
    rows = search_archive("!@#$%", limit=10)
    assert rows == []


# ===========================================================================
# 9) Resolutions endpoint
# ===========================================================================


@respx.mock
def test_resolutions_yes_with_payout() -> None:
    market = _gamma_market(
        slug="res-yes",
        yes_price=1.0,
        extra={"resolutionSource": "https://example.com/uma"},
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "res-yes"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    rec = get_resolution("res-yes")
    assert rec["resolution"] == "YES"
    assert rec["payout_per_share"] == 1.0
    assert rec["resolution_source"] == "https://example.com/uma"


@respx.mock
def test_resolutions_pending_for_open_market() -> None:
    market = _gamma_market(slug="res-pending", closed=False)
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "res-pending"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    rec = get_resolution("res-pending")
    assert rec["resolution"] == "PENDING"


@respx.mock
def test_resolutions_dispute_history_populated() -> None:
    market = _gamma_market(
        slug="res-disp",
        extra={
            "umaResolutionStatuses": "disputed by signer",
            "disputes": [
                {
                    "timestamp": "2024-09-10T00:00:00Z",
                    "kind": "dispute",
                    "reason": "ambiguous wording",
                }
            ],
        },
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "res-disp"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    rec = get_resolution("res-disp")
    assert len(rec["dispute_history"]) >= 1
    assert any(d.get("kind") == "uma" for d in rec["dispute_history"])


# ===========================================================================
# 10) API endpoint smoke tests
# ===========================================================================


@respx.mock
def test_api_pm_markets_paginated(poly_app: TestClient) -> None:
    page = [_gamma_market(slug=f"m{i}") for i in range(3)]
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=page))
    r = poly_app.get("/archive/polymarket/markets?start=2024-01-01&end=2024-12-31")
    assert r.status_code == 200
    body = r.json()
    assert body["n_markets"] == 3
    assert "limit" in body and "offset" in body


@respx.mock
def test_api_pm_market_detail(poly_app: TestClient) -> None:
    market = _gamma_market(slug="api-detail")
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "api-detail"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.5, 0.7, 0.9]))
    )
    r = poly_app.get("/archive/polymarket/market/api-detail")
    assert r.status_code == 200
    body = r.json()
    assert body["slug"] == "api-detail"
    assert "stats" in body
    assert "history" in body


@respx.mock
def test_api_pm_themes(poly_app: TestClient) -> None:
    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=[
            httpx.Response(200, json=[_gamma_market(slug="m1", theme="politics")]),
            httpx.Response(200, json=[]),
        ]
    )
    r = poly_app.get("/archive/polymarket/themes")
    assert r.status_code == 200
    assert "themes" in r.json()


@respx.mock
def test_api_pm_resolutions(poly_app: TestClient) -> None:
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "rx"}).mock(
        return_value=httpx.Response(200, json=[_gamma_market(slug="rx")])
    )
    r = poly_app.get("/archive/polymarket/resolutions/rx")
    assert r.status_code == 200
    assert r.json()["resolution"] == "YES"


@respx.mock
def test_api_pm_search(poly_app: TestClient) -> None:
    page = [
        _gamma_market(slug="trump-q", question="Trump?"),
        _gamma_market(slug="other", question="Other?"),
    ]
    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=[
            httpx.Response(200, json=page),
            httpx.Response(200, json=[]),
        ]
    )
    r = poly_app.get("/archive/polymarket/search?q=trump&limit=5")
    assert r.status_code == 200
    body = r.json()
    assert body["n_results"] == 1


@respx.mock
def test_api_kalshi_markets(kalshi_app: TestClient) -> None:
    payload = {"markets": [_kalshi_market_row("KX-1", series="KX")], "cursor": ""}
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(return_value=httpx.Response(200, json=payload))
    r = kalshi_app.get("/archive/kalshi/markets?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 1
    assert body["items"][0]["ticker"] == "KX-1"


@respx.mock
def test_api_kalshi_series(kalshi_app: TestClient) -> None:
    payload = {
        "markets": [_kalshi_market_row("KX-1", series="KX", result="yes", volume=10)],
        "cursor": "",
    }
    respx.get(f"{KALSHI_BASE_URL}/markets").mock(return_value=httpx.Response(200, json=payload))
    r = kalshi_app.get("/archive/kalshi/series")
    assert r.status_code == 200
    assert r.json()["n_series"] == 1


def test_api_cross_venue_concepts(kalshi_app: TestClient) -> None:
    r = kalshi_app.get("/archive/cross-venue/concepts")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 5


def test_api_cross_venue_unknown_concept_404(kalshi_app: TestClient) -> None:
    r = kalshi_app.get("/archive/cross-venue/totally-not-a-concept")
    assert r.status_code == 404


def test_api_cross_venue_concept_via_cache(kalshi_app: TestClient) -> None:
    """Pre-populate the function cache so the route doesn't hit the network."""
    poly, kalshi = _make_election_pair()
    payload = cross_venue_resolved_pairs(
        "presidential_election_2024",
        polymarket_history=poly,
        kalshi_history_fn=kalshi,
    )
    assert payload["error"] is None
    r = kalshi_app.get("/archive/cross-venue/presidential_election_2024")
    assert r.status_code == 200
    body = r.json()
    assert body["concept"] == "presidential_election_2024"
    assert body["n_overlap_days"] == 6


# ===========================================================================
# 11) Stats sanity checks
# ===========================================================================


@respx.mock
def test_stats_peak_geq_max_history() -> None:
    market = _gamma_market(slug="peak-sanity")
    prices = [0.20, 0.55, 0.30, 0.99, 0.40]
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "peak-sanity"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )
    detail = fetch_archive_market_detail("peak-sanity")
    history_prices = [row[1] for row in detail["history"]]
    s = detail["stats"]
    assert s["peak_price"] == pytest.approx(max(history_prices))
    assert s["trough_price"] == pytest.approx(min(history_prices))
    assert s["peak_price"] >= s["trough_price"]


@respx.mock
def test_stats_whale_concentration_in_unit_interval() -> None:
    market = _gamma_market(slug="whale-sanity", extra={"topWalletsShare": 0.78})
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "whale-sanity"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.5, 0.6]))
    )
    detail = fetch_archive_market_detail("whale-sanity")
    wc = detail["stats"]["whale_concentration"]
    assert wc is not None
    assert 0.0 <= wc <= 1.0


@respx.mock
def test_stats_extreme_monotonic_market_short_half_life() -> None:
    """Market climbing 0.5 → 0.99 monotonic over 60d: half_life is bounded."""
    market = _gamma_market(slug="extreme-mono", yes_price=0.99)
    prices = list(np.linspace(0.50, 0.99, 60))
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "extreme-mono"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )
    detail = fetch_archive_market_detail("extreme-mono")
    s = detail["stats"]
    assert s["half_life_to_resolution"] is not None
    # Series climbs 0.5→0.9 in ~50d, so half_life ≤ 60 days.
    assert s["half_life_to_resolution"] <= 60
    # Hurst defined and finite (estimator applied to first-differences).
    assert s["hurst_exponent"] is not None
    assert math.isfinite(s["hurst_exponent"])


# ===========================================================================
# 12) Edge cases
# ===========================================================================


@respx.mock
def test_edge_single_point_history_does_not_crash() -> None:
    """One-point history should produce stats but realized_vol may be None
    because we need >= 3 points to compute log-returns ddof=1."""
    market = _gamma_market(slug="single-pt")
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "single-pt"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history([0.5]))
    )
    detail = fetch_archive_market_detail("single-pt")
    assert detail["resolution"] == "YES"
    s = detail["stats"]
    # Peak == trough on a single point.
    assert s["peak_price"] == pytest.approx(0.5)
    assert s["trough_price"] == pytest.approx(0.5)
    # Realized vol may be None on too-short series.
    assert s["volatility_realized"] is None or s["volatility_realized"] >= 0


@respx.mock
def test_edge_empty_history_stats_nullable() -> None:
    market = _gamma_market(slug="empty-hist", token_yes="", token_no="")
    market["clobTokenIds"] = json.dumps([])
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "empty-hist"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    detail = fetch_archive_market_detail("empty-hist")
    s = detail["stats"]
    assert s["peak_price"] is None
    assert s["trough_price"] is None
    assert s["volatility_realized"] is None
    assert s["hurst_exponent"] is None
    assert detail["history"] == []


@respx.mock
def test_edge_short_history_no_half_life() -> None:
    """If price never crosses 0.9, half_life_to_resolution must be None
    even when YES side won."""
    market = _gamma_market(slug="no-cross", yes_price=1.0)
    # Prices stuck at 0.5 then jump to 1.0 only on resolution day; CLOB
    # series here intentionally never reaches 0.9 until last bucket.
    prices = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]  # max < 0.9
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "no-cross"}).mock(
        return_value=httpx.Response(200, json=[market])
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history(prices))
    )
    detail = fetch_archive_market_detail("no-cross")
    assert detail["resolution"] == "YES"
    assert detail["stats"]["half_life_to_resolution"] is None


def test_edge_pm_listing_invalid_date_range_422(poly_app: TestClient) -> None:
    r = poly_app.get("/archive/polymarket/markets?start=2024-12-31&end=2024-01-01")
    assert r.status_code == 422


def test_edge_pm_listing_bad_iso_date_422(poly_app: TestClient) -> None:
    r = poly_app.get("/archive/polymarket/markets?start=not-a-date")
    assert r.status_code == 422


@respx.mock
def test_edge_resolutions_404_for_missing_slug(poly_app: TestClient) -> None:
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost-r"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost-r", "closed": "true"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    r = poly_app.get("/archive/polymarket/resolutions/ghost-r")
    assert r.status_code == 404


def test_edge_search_endpoint_empty_q_422(poly_app: TestClient) -> None:
    """The search endpoint constrains q to min_length=1 → 422 on empty."""
    r = poly_app.get("/archive/polymarket/search?q=&limit=5")
    assert r.status_code == 422


# Helpful: verify the cross-venue endpoint range check works.
def test_cv_pct_pm_higher_in_unit_interval() -> None:
    poly, kalshi = _make_election_pair()
    out = cross_venue_resolved_pairs(
        "fed_first_cut_2024",
        polymarket_history=poly,
        kalshi_history_fn=kalshi,
    )
    assert 0.0 <= out["pct_time_pm_higher"] <= 1.0


# Date-range filter: end_date is included in [start, end]
@respx.mock
def test_pm_date_range_filter_passed_to_gamma() -> None:
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.url.params))
        return httpx.Response(200, json=[])

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=handler)
    s = date(2024, 6, 1)
    e = date(2024, 12, 31)
    fetch_resolved_markets(s, e, limit=10)
    assert captured[0]["date_end_min"] == s.isoformat()
    assert captured[0]["date_end_max"] == e.isoformat()


# Verify default date window when start/end omitted (should be ~1 year)
@respx.mock
def test_pm_listing_default_dates_one_year_window(poly_app: TestClient) -> None:
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.url.params))
        return httpx.Response(200, json=[])

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=handler)
    poly_app.get("/archive/polymarket/markets")
    qs = captured[0]
    s = date.fromisoformat(qs["date_end_min"])
    e = date.fromisoformat(qs["date_end_max"])
    delta = e - s
    # Default is 365 days; allow ±2 days slack.
    assert abs(delta - timedelta(days=365)) <= timedelta(days=2)
