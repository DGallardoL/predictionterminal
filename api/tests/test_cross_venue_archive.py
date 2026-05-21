"""Tests for the cross-venue archive comparator (Polymarket vs Kalshi).

Both venue history fetchers are injected, so these tests need no
``respx`` for HTTP — we just hand in pre-built DataFrames.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.archive import cross_venue_archive as cva
from pfm.archive.cross_venue_archive import (
    CROSS_VENUE_CONCEPTS,
    cross_venue_resolved_pairs,
    list_concepts,
)
from pfm.archive.kalshi_router import router as kalshi_router
from pfm.cache_utils import get_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    get_cache(cva.CACHE_NS).clear()
    yield
    get_cache(cva.CACHE_NS).clear()


def _frame(prices: list[tuple[str, float]]) -> pd.DataFrame:
    idx = pd.to_datetime([d for d, _ in prices], utc=True).normalize()
    df = pd.DataFrame({"price": [p for _, p in prices]}, index=idx)
    df.index.name = "date"
    return df


def _make_history_pair():
    """Return matched (poly, kalshi) history fetchers for the election concept.

    Polymarket sits above Kalshi throughout; only two days breach the
    0.05 absolute-divergence threshold (the days where the spread is
    0.10). The last day they nearly converge.
    """
    poly_df = _frame(
        [
            ("2024-09-01", 0.53),  # spread 0.03 — within threshold
            ("2024-09-02", 0.58),  # spread 0.03
            ("2024-09-03", 0.70),  # spread 0.10 — diverged
            ("2024-09-04", 0.65),  # spread 0.10 — diverged
            ("2024-11-05", 0.83),  # spread 0.03
            ("2024-11-06", 0.99),  # spread 0.01
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

    def _poly(_client, _slug, start=None, end=None):
        return poly_df

    def _kalshi(_client, _ticker, start=None, end=None, series_ticker=None):
        return kalshi_df

    return _poly, _kalshi


def test_concepts_catalog_has_five_entries() -> None:
    items = list_concepts()
    assert len(items) == 5
    assert {i["concept"] for i in items} == {
        "presidential_election_2024",
        "recession_2024",
        "fed_first_cut_2024",
        "btc_70k_2024",
        "cpi_above_3_2024",
    }
    for entry in items:
        assert entry["polymarket_slug"]
        assert entry["kalshi_ticker"]
        assert entry["resolved_outcome"] in {"YES", "NO"}


def test_cross_venue_resolved_pairs_election_metrics() -> None:
    poly, kalshi = _make_history_pair()
    out = cross_venue_resolved_pairs(
        "presidential_election_2024",
        polymarket_history=poly,
        kalshi_history_fn=kalshi,
    )

    assert out["error"] is None
    assert out["concept"] == "presidential_election_2024"
    assert out["resolved_outcome"] == "YES"
    assert out["n_overlap_days"] == 6
    assert out["spread_at_resolution"] == pytest.approx(0.01)
    assert out["max_spread_observed"] == pytest.approx(0.10)
    # 0.05 threshold: only 0.10-spread days count → 2 days.
    assert out["days_diverged"] == 2
    # PM > Kalshi on ALL days in the fixture.
    assert out["pct_time_pm_higher"] == pytest.approx(1.0)


def test_cross_venue_resolved_pairs_unknown_concept_raises() -> None:
    with pytest.raises(KeyError):
        cross_venue_resolved_pairs("not-a-concept")


def test_cross_venue_resolved_pairs_handles_empty_history() -> None:
    def _empty(_c, *_a, **_k):
        return pd.DataFrame(columns=["price"])

    out = cross_venue_resolved_pairs(
        "recession_2024",
        polymarket_history=_empty,
        kalshi_history_fn=_empty,
    )
    assert out["error"] is not None
    assert out["n_overlap_days"] == 0
    assert out["spread_at_resolution"] is None


def test_cross_venue_resolved_pairs_disjoint_dates() -> None:
    """No overlap between PM and Kalshi calendars → error payload."""
    pm = _frame([("2024-01-01", 0.5), ("2024-01-02", 0.6)])
    ks = _frame([("2024-06-01", 0.5), ("2024-06-02", 0.6)])

    out = cross_venue_resolved_pairs(
        "btc_70k_2024",
        polymarket_history=lambda *_a, **_k: pm,
        kalshi_history_fn=lambda *_a, **_k: ks,
    )
    assert "no overlapping" in (out["error"] or "")


def test_cross_venue_resolved_pairs_caches_result() -> None:
    poly, kalshi = _make_history_pair()
    calls = {"poly": 0, "ks": 0}

    def _poly(*a, **k):
        calls["poly"] += 1
        return poly(*a, **k)

    def _ks(*a, **k):
        calls["ks"] += 1
        return kalshi(*a, **k)

    cross_venue_resolved_pairs(
        "presidential_election_2024", polymarket_history=_poly, kalshi_history_fn=_ks
    )
    cross_venue_resolved_pairs(
        "presidential_election_2024", polymarket_history=_poly, kalshi_history_fn=_ks
    )
    assert calls == {"poly": 1, "ks": 1}


# ────────────────────────────── router smoke ───────────────────────────────


def test_router_concepts_endpoint() -> None:
    app = FastAPI()
    app.include_router(kalshi_router)
    client = TestClient(app)

    r = client.get("/archive/cross-venue/concepts")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 5
    assert any(c["concept"] == "presidential_election_2024" for c in body["concepts"])


def test_router_unknown_concept_returns_404() -> None:
    app = FastAPI()
    app.include_router(kalshi_router)
    client = TestClient(app)
    r = client.get("/archive/cross-venue/does-not-exist")
    assert r.status_code == 404


def test_router_concept_endpoint_uses_cached_payload() -> None:
    """Pre-populate the cache so the router doesn't make network calls."""
    poly, kalshi = _make_history_pair()
    payload = cross_venue_resolved_pairs(
        "presidential_election_2024",
        polymarket_history=poly,
        kalshi_history_fn=kalshi,
    )
    assert payload["error"] is None

    app = FastAPI()
    app.include_router(kalshi_router)
    client = TestClient(app)
    r = client.get("/archive/cross-venue/presidential_election_2024")
    assert r.status_code == 200
    body = r.json()
    assert body["concept"] == "presidential_election_2024"
    assert body["n_overlap_days"] == payload["n_overlap_days"]
    assert body["max_spread_observed"] == pytest.approx(payload["max_spread_observed"])


def test_concept_dict_keys_are_complete() -> None:
    """Every concept entry must carry the four mapping fields."""
    required = {"polymarket_slug", "kalshi_ticker", "description", "resolved_outcome"}
    for v in CROSS_VENUE_CONCEPTS.values():
        assert required <= set(v.keys())
