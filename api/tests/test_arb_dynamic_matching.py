"""Tests for the dynamic-matching engine and 4-venue arb extensions.

Covers ``auto_discover_arb_pairs``, ``compute_4way_arbs``, the persistent
confirmed-match registry, and the new ``/arb/auto-discover`` /
``/arb/4way-arbs`` / ``/arb/confirmed-matches`` endpoints. External
fetchers are stubbed at the module level so no real network calls fire.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import arb_scanner
from pfm.arb_scanner import (
    PRE_MATCHED_PAIRS,
    auto_discover_arb_pairs,
    compute_4way_arbs,
    list_confirmed_matches,
    record_match_observation,
    router,
)
from pfm.cache_utils import get_cache


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path) -> None:
    """Drop caches + redirect the persistent-store path so tests don't bleed."""
    get_cache("arb_scanner").clear()
    get_cache("arb_matched").clear()
    get_cache("arb_dynamic").clear()
    arb_scanner._MANUAL_PAIRS.clear()
    arb_scanner.CONFIRMED_MATCHES_PATH = tmp_path / "confirmed.json"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _pm_market(
    slug: str,
    title: str,
    price: float = 0.5,
    vol: float = 50_000.0,
    end_date: str = "2026-12-31T00:00:00Z",
    theme: str = "macro",
) -> dict[str, Any]:
    return {
        "venue": "polymarket",
        "id": slug,
        "slug": slug,
        "title": title,
        "theme": theme,
        "end_date": end_date,
        "price": price,
        "volume_24h_usd": vol,
    }


def _kalshi_market(
    ticker: str,
    title: str,
    price: float = 0.55,
    vol: float = 30_000.0,
    end_date: str = "2026-12-31T00:00:00Z",
    theme: str = "macro",
) -> dict[str, Any]:
    return {
        "venue": "kalshi",
        "id": ticker,
        "ticker": ticker,
        "slug": ticker,
        "title": title,
        "theme": theme,
        "end_date": end_date,
        "price": price,
        "volume_24h_usd": vol,
    }


def _manifold_market(
    slug: str,
    title: str,
    price: float = 0.53,
    vol: float = 4_000.0,
) -> dict[str, Any]:
    return {
        "venue": "manifold",
        "id": slug,
        "slug": slug,
        "title": title,
        "theme": None,
        "end_date": "2026-12-31T00:00:00Z",
        "price": price,
        "volume_24h_usd": vol,
    }


def _predictit_market(
    mid: str,
    title: str,
    price: float = 0.60,
    vol: float = 1_500.0,
) -> dict[str, Any]:
    return {
        "venue": "predictit",
        "id": mid,
        "slug": mid,
        "title": title,
        "theme": None,
        "end_date": "2026-12-31T00:00:00Z",
        "price": price,
        "volume_24h_usd": vol,
    }


def _make_pm_universe(n: int = 100) -> list[dict[str, Any]]:
    """100 PM markets, the first one matching the recession theme."""
    out = [
        _pm_market(
            "us-recession-2026",
            "Will the US enter a recession in 2026?",
            price=0.50,
        )
    ]
    for i in range(1, n):
        out.append(
            _pm_market(
                f"unrelated-pm-{i}",
                f"Unrelated PM market number {i} about widgets",
                vol=10_000.0 + i,
            )
        )
    return out


def _make_kalshi_universe(n: int = 100) -> list[dict[str, Any]]:
    out = [
        _kalshi_market(
            "KXRECSSNBER-26",
            "US recession declared by NBER in 2026",
            price=0.55,
        )
    ]
    for i in range(1, n):
        out.append(
            _kalshi_market(
                f"K-NOISE-{i}",
                f"Sports event number {i} which has nothing in common",
                theme="sports",
                end_date="2099-01-01T00:00:00Z",
                vol=8_000.0 + i,
            )
        )
    return out


# ---------------------------------------------------------------------------
# auto_discover_arb_pairs
# ---------------------------------------------------------------------------


def _patch_fetchers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pm: list[dict[str, Any]] | None = None,
    kalshi: list[dict[str, Any]] | None = None,
    manifold: list[dict[str, Any]] | None = None,
    predictit: list[dict[str, Any]] | None = None,
) -> None:
    async def _fake_pm(http: Any, *, limit: int = 100) -> list[dict[str, Any]]:
        return list(pm or [])

    async def _fake_kalshi(http: Any, *, limit: int = 100) -> list[dict[str, Any]]:
        return list(kalshi or [])

    async def _fake_manifold(http: Any, *, limit: int = 100) -> list[dict[str, Any]]:
        return list(manifold or [])

    async def _fake_predictit(http: Any, *, limit: int = 100) -> list[dict[str, Any]]:
        return list(predictit or [])

    monkeypatch.setattr(arb_scanner, "_fetch_active_polymarket", _fake_pm)
    monkeypatch.setattr(arb_scanner, "_fetch_active_kalshi", _fake_kalshi)
    monkeypatch.setattr(arb_scanner, "_fetch_active_manifold", _fake_manifold)
    monkeypatch.setattr(arb_scanner, "_fetch_active_predictit", _fake_predictit)
    monkeypatch.setattr(
        arb_scanner,
        "_VENUE_FETCHERS",
        {
            "polymarket": _fake_pm,
            "kalshi": _fake_kalshi,
            "manifold": _fake_manifold,
            "predictit": _fake_predictit,
        },
    )


def test_auto_discover_recovers_recession_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 PM x 100 Kalshi: only the recession pair survives the threshold."""
    _patch_fetchers(
        monkeypatch,
        pm=_make_pm_universe(100),
        kalshi=_make_kalshi_universe(100),
    )
    pairs = asyncio.run(
        auto_discover_arb_pairs(
            min_similarity=0.5,
            min_volume_usd_per_venue=1_000.0,
            max_pairs=50,
            venues=["polymarket", "kalshi"],
        )
    )
    assert pairs, "expected at least the recession pair to be discovered"
    top = pairs[0]
    assert top["polymarket_slug"] == "us-recession-2026"
    assert top["kalshi_slug"] == "KXRECSSNBER-26"
    assert top["similarity_score"] >= 0.5
    # Spread = |0.50 - 0.55| * 100 = 5%
    assert top["spread_pct"] == pytest.approx(5.0, abs=0.01)
    # Tradeable size = min(50_000, 30_000) = 30_000
    assert top["tradeable_size_usd"] == pytest.approx(30_000.0, abs=0.01)


def test_auto_discover_threshold_changes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raising min_similarity past the recession-pair score eliminates it."""
    _patch_fetchers(
        monkeypatch,
        pm=_make_pm_universe(100),
        kalshi=_make_kalshi_universe(100),
    )
    loose = asyncio.run(
        auto_discover_arb_pairs(
            min_similarity=0.4,
            min_volume_usd_per_venue=1_000.0,
            max_pairs=50,
            venues=["polymarket", "kalshi"],
        )
    )
    strict = asyncio.run(
        auto_discover_arb_pairs(
            min_similarity=0.999,
            min_volume_usd_per_venue=1_000.0,
            max_pairs=50,
            venues=["polymarket", "kalshi"],
        )
    )
    assert len(loose) >= 1
    assert len(strict) == 0


def test_auto_discover_volume_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markets below ``min_volume_usd_per_venue`` are excluded before matching."""
    pm_low = [
        _pm_market(
            "us-recession-2026",
            "Will the US enter a recession in 2026?",
            vol=200.0,  # under threshold
        )
    ]
    kalshi_ok = [
        _kalshi_market(
            "KXRECSSNBER-26",
            "US recession declared by NBER in 2026",
            vol=30_000.0,
        )
    ]
    _patch_fetchers(monkeypatch, pm=pm_low, kalshi=kalshi_ok)
    pairs = asyncio.run(
        auto_discover_arb_pairs(
            min_similarity=0.4,
            min_volume_usd_per_venue=1_000.0,
            max_pairs=50,
            venues=["polymarket", "kalshi"],
        )
    )
    assert pairs == []


def test_auto_discover_max_pairs_caps_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing ``max_pairs=3`` returns at most 3 pairs even when more match."""
    pm = [_pm_market(f"recession-{i}", "US recession 2026 NBER declared") for i in range(10)]
    kalshi = [_kalshi_market(f"K-REC-{i}", "US recession declared NBER 2026") for i in range(10)]
    _patch_fetchers(monkeypatch, pm=pm, kalshi=kalshi)
    pairs = asyncio.run(
        auto_discover_arb_pairs(
            min_similarity=0.5,
            min_volume_usd_per_venue=1_000.0,
            max_pairs=3,
            venues=["polymarket", "kalshi"],
        )
    )
    assert len(pairs) == 3


# ---------------------------------------------------------------------------
# compute_4way_arbs
# ---------------------------------------------------------------------------


def test_compute_4way_arbs_max_spread_10pct() -> None:
    """PM 50c, K 55c, Manifold 53c, PredictIt 60c => 10c max spread."""
    fns = {
        "polymarket": lambda _ident: (0.50, 5_000.0),
        "kalshi": lambda _ident: (0.55, 4_000.0),
        "manifold": lambda _ident: (0.53, 3_000.0),
        "predictit": lambda _ident: (0.60, 2_000.0),
    }
    arbs = compute_4way_arbs(price_fns=fns, min_spread_pct=0.0)
    assert arbs, "expected at least one 4-way arb"
    rec = arbs[0]
    assert rec["max_spread_pct"] == pytest.approx(10.0, abs=0.01)
    assert rec["low_venue"] == "polymarket"
    assert rec["high_venue"] == "predictit"
    # Tradeable size is the minimum across venues = 2_000
    assert rec["tradeable_size_usd"] == pytest.approx(2_000.0, abs=0.01)
    assert set(rec["legs_present"]) == {
        "polymarket",
        "kalshi",
        "manifold",
        "predictit",
    }


def test_compute_4way_arbs_skips_concept_with_one_leg() -> None:
    """A concept with only one venue priced is dropped (no spread possible)."""
    fns = {
        "polymarket": lambda _ident: (0.50, 5_000.0),
        # No other venue functions => single leg per concept.
    }
    arbs = compute_4way_arbs(price_fns=fns, min_spread_pct=0.0)
    assert arbs == []


def test_compute_4way_arbs_min_spread_filter() -> None:
    """Concepts whose max-pairwise spread is below ``min_spread_pct`` are dropped."""
    fns = {
        "polymarket": lambda _ident: (0.50, 5_000.0),
        "kalshi": lambda _ident: (0.505, 4_000.0),  # 0.5% spread
    }
    arbs = compute_4way_arbs(price_fns=fns, min_spread_pct=2.0)
    assert arbs == []


# ---------------------------------------------------------------------------
# Persistent matching
# ---------------------------------------------------------------------------


def test_persistent_matching_promotes_after_seven_fetches() -> None:
    """Seven consecutive observations promote the pair to ``confirmed=True``."""
    for _ in range(arb_scanner.CONFIRMED_FETCHES_REQUIRED):
        rec = record_match_observation(
            "polymarket",
            "us-recession-2026",
            "kalshi",
            "KXRECSSNBER-26",
            similarity=0.82,
            label="US recession 2026",
        )
    assert rec["confirmed"] is True
    confirmed = list_confirmed_matches(only_confirmed=True)
    assert len(confirmed) == 1
    assert confirmed[0]["fetches"] == arb_scanner.CONFIRMED_FETCHES_REQUIRED
    assert confirmed[0]["label"] == "US recession 2026"


def test_persistent_matching_below_threshold_not_confirmed() -> None:
    """Six observations are below the threshold; pair stays unconfirmed."""
    for _ in range(arb_scanner.CONFIRMED_FETCHES_REQUIRED - 1):
        rec = record_match_observation(
            "polymarket",
            "btc-100k-2026",
            "kalshi",
            "KXBTC100K-26",
            similarity=0.71,
            label="BTC 100k 2026",
        )
    assert rec["confirmed"] is False
    assert list_confirmed_matches(only_confirmed=True) == []
    all_matches = list_confirmed_matches(only_confirmed=False)
    assert len(all_matches) == 1
    assert all_matches[0]["fetches"] == arb_scanner.CONFIRMED_FETCHES_REQUIRED - 1


def test_persistent_matching_pair_key_is_order_independent() -> None:
    """The same pair recorded with venues swapped maps to one entry."""
    record_match_observation(
        "polymarket",
        "x",
        "kalshi",
        "Y",
        similarity=0.9,
        label="forward",
    )
    rec = record_match_observation(
        "kalshi",
        "Y",
        "polymarket",
        "x",
        similarity=0.9,
        label="reversed",
    )
    assert rec["fetches"] == 2
    assert len(list_confirmed_matches(only_confirmed=False)) == 1


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_pre_matched_pairs_still_present() -> None:
    """The 5 hardcoded pairs MUST remain available (backward compat)."""
    assert len(PRE_MATCHED_PAIRS) == 5
    for pair in PRE_MATCHED_PAIRS:
        assert "pm_slug" in pair and "kalshi_slug" in pair


def test_concept_maps_still_have_five_entries() -> None:
    assert len(arb_scanner.CONCEPT_MAPS) == 5
    for concept in arb_scanner.CONCEPT_MAPS:
        assert "concept_id" in concept
        assert "polymarket" in concept and "kalshi" in concept


def test_find_4way_arb_still_works() -> None:
    """The legacy 4-way snapshot helper still resolves a known concept."""
    out = arb_scanner.find_4way_arb(
        "fed_cuts_2026",
        pm_price_fn=lambda _ident: (0.50, 1_000.0),
        kalshi_price_fn=lambda _ident: (0.55, 1_000.0),
    )
    assert out["concept_id"] == "fed_cuts_2026"
    assert out["max_spread_pct"] == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# Endpoint smoke tests
# ---------------------------------------------------------------------------


def test_get_auto_discover_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetchers(
        monkeypatch,
        pm=_make_pm_universe(20),
        kalshi=_make_kalshi_universe(20),
    )
    r = client.get(
        "/arb/auto-discover",
        params={"min_similarity": 0.5, "min_volume": 1000.0, "max_pairs": 10},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] >= 1
    assert "pairs" in body
    assert body["min_similarity"] == 0.5
    assert any(
        p.get("polymarket_slug") == "us-recession-2026" and p.get("kalshi_slug") == "KXRECSSNBER-26"
        for p in body["pairs"]
    )


def test_get_4way_arbs_endpoint(client: TestClient) -> None:
    """Endpoint returns 200 with an empty arbs list when no price fns wired."""
    r = client.get("/arb/4way-arbs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "arbs" in body
    assert body["n"] == len(body["arbs"])


def test_get_confirmed_matches_endpoint(client: TestClient) -> None:
    # Seed the registry to seven fetches so we have one confirmed entry.
    for _ in range(arb_scanner.CONFIRMED_FETCHES_REQUIRED):
        record_match_observation(
            "polymarket",
            "test-confirmed",
            "kalshi",
            "K-CONF",
            similarity=0.91,
            label="endpoint smoke",
        )
    r = client.get("/arb/confirmed-matches")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fetches_required"] == arb_scanner.CONFIRMED_FETCHES_REQUIRED
    assert body["n"] >= 1
    assert any(m.get("label") == "endpoint smoke" for m in body["matches"])


def test_legacy_endpoints_unchanged(client: TestClient) -> None:
    """``/arb/concepts`` and ``/arb/matched`` keep working with no regressions."""
    r1 = client.get("/arb/concepts")
    assert r1.status_code == 200
    assert r1.json()["n"] == 5

    r2 = client.get("/arb/matched")
    assert r2.status_code == 200
    body = r2.json()
    assert body["n"] >= 5
    assert any(p["source"] == "hardcoded" for p in body["pairs"])
