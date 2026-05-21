"""Tests for the cross-venue arb-scanner module.

Mounts ``router`` on a throw-away FastAPI app so we don't touch
``main.py``. External calls (Polymarket Gamma + Kalshi candlesticks) are
mocked at module level — no real-network hits.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import arb_scanner
from pfm.arb_scanner import (
    PRE_MATCHED_PAIRS,
    compute_arb_spreads,
    match_markets,
    router,
    top_arbs,
)
from pfm.cache_utils import get_cache


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Drop the scanner + match caches between tests so they don't bleed."""
    get_cache("arb_scanner").clear()
    get_cache("arb_matched").clear()
    # Also wipe the manual-pair registry so each test starts clean.
    arb_scanner._MANUAL_PAIRS.clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# match_markets
# ---------------------------------------------------------------------------


def test_match_markets_pairs_two_synthetic_markets() -> None:
    pm_markets = [
        {
            "slug": "us-recession-2026",
            "title": "Will the US enter a recession in 2026?",
            "theme": "macro",
            "end_date": "2026-12-31T00:00:00Z",
        },
        {
            "slug": "btc-100k",
            "title": "BTC above $100k by year end",
            "theme": "crypto",
            "end_date": "2026-12-31T00:00:00Z",
        },
    ]
    kalshi_markets = [
        {
            "ticker": "KXRECSSNBER-26",
            "title": "US recession declared by NBER in 2026",
            "theme": "macro",
            "end_date": "2026-12-31T00:00:00Z",
        },
        # Mismatched: different theme + topic.
        {
            "ticker": "KX-FOO-99",
            "title": "Will AI surpass humans by 2099?",
            "theme": "tech",
            "end_date": "2099-12-31T00:00:00Z",
        },
    ]
    pairs = match_markets(pm_markets, kalshi_markets, min_similarity=0.5)
    assert len(pairs) == 1, pairs
    pair = pairs[0]
    assert pair["pm_slug"] == "us-recession-2026"
    assert pair["kalshi_slug"] == "KXRECSSNBER-26"
    assert pair["similarity_score"] >= 0.5
    assert isinstance(pair["suggested"], bool)


def test_match_markets_empty_inputs() -> None:
    assert match_markets([], []) == []
    assert match_markets([{"slug": "x", "title": "Foo"}], []) == []


def test_match_markets_below_threshold_returns_empty() -> None:
    pm = [{"slug": "a", "title": "Apples", "theme": "food", "end_date": None}]
    kalshi = [{"ticker": "B", "title": "Zebras racing", "theme": "sports", "end_date": None}]
    assert match_markets(pm, kalshi, min_similarity=0.7) == []


# ---------------------------------------------------------------------------
# compute_arb_spreads
# ---------------------------------------------------------------------------


def _fake_pm_market(slug: str, mid: float, vol: float = 50_000.0) -> dict[str, Any]:
    """Build a minimal Gamma market dict that ``_pm_mid`` can parse."""
    return {
        "slug": slug,
        "bestBid": mid - 0.005,
        "bestAsk": mid + 0.005,
        "lastTradePrice": mid,
        "volume24hr": vol,
    }


def _fake_kalshi_df(mid: float, vol: float = 30_000.0) -> pd.DataFrame:
    """One-row candlestick DataFrame matching ``KalshiClient.get_candlesticks``."""
    idx = pd.DatetimeIndex([pd.Timestamp("2026-05-08", tz="UTC")], name="date")
    return pd.DataFrame(
        {
            "price": [mid],
            "volume": [vol],
            "open_interest": [vol * 2],
            "yes_bid": [mid - 0.01],
            "yes_ask": [mid + 0.01],
            "spread": [0.02],
        },
        index=idx,
    )


def _stub_clients(
    pm_returns: dict[str, float],
    kalshi_returns: dict[str, float],
    *,
    pm_vol: float = 50_000.0,
    kalshi_vol: float = 30_000.0,
) -> tuple[MagicMock, MagicMock]:
    """Build a (httpx-like, KalshiClient-like) pair patched to return mids."""
    http = MagicMock()
    kalshi_client = MagicMock()

    # Patch the internal mid fetchers via ``arb_scanner`` directly so we
    # don't have to reach into the Polymarket Gamma HTTP shape.
    return http, kalshi_client


def test_compute_arb_spreads_filters_by_min_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 1% spread must be filtered out at min_spread_pct=2.0."""
    pairs = [
        {"pm_slug": "p1", "kalshi_slug": "K1", "label": "Pair 1"},
        {"pm_slug": "p2", "kalshi_slug": "K2", "label": "Pair 2"},
    ]
    pm_mids = {"p1": (0.50, 50_000.0), "p2": (0.60, 50_000.0)}
    kalshi_mids = {"K1": (0.51, 30_000.0), "K2": (0.55, 30_000.0)}

    monkeypatch.setattr(arb_scanner, "_pm_mid", lambda slug, http: pm_mids[slug])
    monkeypatch.setattr(arb_scanner, "_kalshi_mid", lambda ticker, client: kalshi_mids[ticker])

    arbs = compute_arb_spreads(
        pairs,
        min_spread_pct=2.0,
        min_volume_usd=1_000.0,
        http=MagicMock(),
        kalshi_client=MagicMock(),
    )
    # Pair 1: 1% spread → filtered. Pair 2: 5% spread → kept.
    assert len(arbs) == 1
    assert arbs[0]["pm_slug"] == "p2"
    assert arbs[0]["spread_pct"] >= 2.0
    assert arbs[0]["confirmed"] is True
    assert arbs[0]["direction"] in {"buy_kalshi_sell_pm", "buy_pm_sell_kalshi"}


def test_compute_arb_spreads_filters_by_min_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even a wide spread must be filtered when tradeable size is too small."""
    pairs = [{"pm_slug": "p1", "kalshi_slug": "K1", "label": "Tiny"}]
    monkeypatch.setattr(arb_scanner, "_pm_mid", lambda slug, http: (0.50, 100.0))
    monkeypatch.setattr(arb_scanner, "_kalshi_mid", lambda ticker, c: (0.60, 100.0))
    arbs = compute_arb_spreads(
        pairs,
        min_spread_pct=2.0,
        min_volume_usd=5_000.0,
        http=MagicMock(),
        kalshi_client=MagicMock(),
    )
    assert arbs == []


# ---------------------------------------------------------------------------
# top_arbs
# ---------------------------------------------------------------------------


def test_top_arbs_respects_n_and_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    """top_arbs(n=2) must return at most 2, ranked by spread × size."""
    fake_pairs = [
        {"pm_slug": "a", "kalshi_slug": "A", "label": "A"},
        {"pm_slug": "b", "kalshi_slug": "B", "label": "B"},
        {"pm_slug": "c", "kalshi_slug": "C", "label": "C"},
    ]
    pm_mids = {
        "a": (0.50, 100_000.0),  # 5% spread × 30k = high
        "b": (0.50, 50_000.0),  # 3% spread × 50k = medium
        "c": (0.50, 200_000.0),  # 10% spread × 200k = highest
    }
    kalshi_mids = {
        "A": (0.55, 30_000.0),
        "B": (0.53, 50_000.0),
        "C": (0.60, 200_000.0),
    }

    monkeypatch.setattr(arb_scanner, "all_matched_pairs", lambda: fake_pairs)
    monkeypatch.setattr(arb_scanner, "_pm_mid", lambda slug, http: pm_mids[slug])
    monkeypatch.setattr(arb_scanner, "_kalshi_mid", lambda ticker, client: kalshi_mids[ticker])

    out = top_arbs(min_spread_pct=2.0, n=2, http=MagicMock(), kalshi_client=MagicMock())
    assert len(out) == 2
    # "c" has the largest spread × size, must rank first.
    assert out[0]["pm_slug"] == "c"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_get_matched_includes_hardcoded_pairs(client: TestClient) -> None:
    r = client.get("/arb/matched")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] >= len(PRE_MATCHED_PAIRS)
    sources = {p["source"] for p in body["pairs"]}
    assert "hardcoded" in sources


def test_post_match_then_get_matched_includes_manual(client: TestClient) -> None:
    payload = {
        "pm_slug": "btc-100k",
        "kalshi_slug": "KXBTC100K-26",
        "label": "BTC 100k",
        "theme": "crypto",
    }
    r = client.post("/arb/match", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pm_slug"] == "btc-100k"
    assert body["source"] == "manual"

    r2 = client.get("/arb/matched")
    assert r2.status_code == 200
    body2 = r2.json()
    manual_pairs = [p for p in body2["pairs"] if p["source"] == "manual"]
    assert any(p["pm_slug"] == "btc-100k" for p in manual_pairs)


def test_post_match_idempotent(client: TestClient) -> None:
    payload = {"pm_slug": "x", "kalshi_slug": "Y"}
    client.post("/arb/match", json=payload)
    client.post("/arb/match", json=payload)
    r = client.get("/arb/matched")
    body = r.json()
    matches = [p for p in body["pairs"] if p["pm_slug"] == "x" and p["kalshi_slug"] == "Y"]
    assert len(matches) == 1


def test_get_scanner_with_mocked_pricing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Endpoint integration: stub mid fetchers, hit /arb/scanner, expect arbs."""
    pm_mids = {p["pm_slug"]: (0.50, 50_000.0) for p in PRE_MATCHED_PAIRS}
    kalshi_mids = {p["kalshi_slug"]: (0.60, 30_000.0) for p in PRE_MATCHED_PAIRS}

    monkeypatch.setattr(arb_scanner, "_pm_mid", lambda slug, http: pm_mids.get(slug, (None, None)))
    monkeypatch.setattr(
        arb_scanner,
        "_kalshi_mid",
        lambda ticker, c: kalshi_mids.get(ticker, (None, None)),
    )

    r = client.get("/arb/scanner", params={"min_spread_pct": 2.0, "n": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == min(5, len(PRE_MATCHED_PAIRS))
    assert body["arbs"], "expected at least one arb with the stubbed 10% spread"
    for a in body["arbs"]:
        assert a["spread_pct"] >= 2.0
        assert a["tradeable_size_usd"] >= 0
        assert a["confirmed"] is True
        assert a["direction"] in {"buy_kalshi_sell_pm", "buy_pm_sell_kalshi"}


def test_auto_discover_singleflight_dedupes_cold_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent cold-cache callers share one expensive upstream walk.

    Regression: probe found ``/arb/auto-discover`` at 10.31 s cold → 2 s
    warm, signalling that each first-caller pays the full venue
    fan-out. The ``_auto_discover_lock`` single-flight makes the second
    caller wait for the first to populate the cache.
    """
    import asyncio as _asyncio

    get_cache("arb_dynamic").clear()
    call_count = {"n": 0}

    async def _slow_pairs(**_k: Any) -> list[dict[str, Any]]:
        call_count["n"] += 1
        # Simulate a non-trivial upstream so the second caller races in.
        await _asyncio.sleep(0.10)
        return [{"venue_a": "pm", "venue_b": "kalshi", "label": "synthetic"}]

    monkeypatch.setattr(arb_scanner, "auto_discover_arb_pairs", _slow_pairs)

    async def _both() -> tuple[dict, dict]:
        return await _asyncio.gather(
            arb_scanner.get_auto_discover(0.65, 1000.0, 50),
            arb_scanner.get_auto_discover(0.65, 1000.0, 50),
        )

    a, b = _asyncio.run(_both())
    assert a == b
    assert call_count["n"] == 1, (
        f"expected single-flight to dedupe concurrent first-callers, "
        f"got {call_count['n']} upstream invocations"
    )
