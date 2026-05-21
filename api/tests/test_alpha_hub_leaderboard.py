"""Tests for ``pfm.alpha_hub_router`` — leaderboard / detail / live-panel."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.alpha_hub_router import _cached_strategies, _load_strategies
from pfm.alpha_hub_router import router as alpha_hub_router
from pfm.cache_utils import reset_caches


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()
    yield
    reset_caches()


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(alpha_hub_router)
    return TestClient(app)


# --- loader -----------------------------------------------------------------


def test_loader_returns_non_empty_strategies_array() -> None:
    s = _load_strategies()
    assert isinstance(s, list)
    assert len(s) > 0
    assert "pair_id" in s[0]


def test_cached_strategies_round_trip() -> None:
    a = _cached_strategies()
    b = _cached_strategies()
    # Same object returned by the cache (no recomputation).
    assert a is b


# --- /alpha-hub/leaderboard --------------------------------------------------


def test_leaderboard_default_returns_paginated() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard")
    assert r.status_code == 200
    body = r.json()
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert body["total"] >= len(body["items"])
    assert body["n_returned"] == len(body["items"])


def test_leaderboard_pagination_offset_limit() -> None:
    client = _make_app()
    r1 = client.get("/alpha-hub/leaderboard?limit=5&offset=0")
    r2 = client.get("/alpha-hub/leaderboard?limit=5&offset=5")
    assert r1.status_code == 200
    assert r2.status_code == 200
    b1 = r1.json()
    b2 = r2.json()
    assert len(b1["items"]) == 5
    assert len(b2["items"]) == 5
    ids1 = {it["pair_id"] for it in b1["items"]}
    ids2 = {it["pair_id"] for it in b2["items"]}
    assert ids1.isdisjoint(ids2)


def test_leaderboard_sort_by_oos_sharpe_desc() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?sort=oos_sharpe&order=desc&limit=20")
    body = r.json()
    sharpes = [it["oos_sharpe"] for it in body["items"] if it["oos_sharpe"] is not None]
    assert sharpes == sorted(sharpes, reverse=True)


def test_leaderboard_sort_by_oos_sharpe_asc() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?sort=oos_sharpe&order=asc&limit=20")
    body = r.json()
    sharpes = [it["oos_sharpe"] for it in body["items"] if it["oos_sharpe"] is not None]
    assert sharpes == sorted(sharpes)


def test_leaderboard_tier_filter_b_validated() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?tier=B_VALIDATED&limit=200")
    body = r.json()
    assert body["total"] >= 1
    for item in body["items"]:
        assert item["tier"] == "B_VALIDATED"


def test_leaderboard_tier_filter_a_structural_post_v22() -> None:
    # Per docs/alpha-reports/alpha-report-v22.md (2026-05-19): all 5 Wave-6
    # A_STRUCTURAL promotions were reverted (joint_days < 360). Catalog has 0
    # A_STRUCTURAL until late Aug 2026 at earliest. The filter must still
    # return 200 with an empty list — not error out.
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?tier=A_STRUCTURAL&limit=200")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["items"] == []


def test_leaderboard_theme_filter_macro() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?theme=macro&limit=200")
    body = r.json()
    assert body["total"] >= 1
    for item in body["items"]:
        assert item["theme_a"] == "macro" or item["theme_b"] == "macro"


def test_leaderboard_min_sharpe_filter() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?min_sharpe=2.0&limit=200")
    body = r.json()
    for item in body["items"]:
        if item["oos_sharpe"] is not None:
            assert item["oos_sharpe"] >= 2.0


def test_leaderboard_combined_filters() -> None:
    # Post-v22 catalog has 0 A_STRUCTURAL; switch the positive filter to
    # B_VALIDATED (the top deployable tier today). This also exercises the
    # combined-filter code path, which was the actual purpose of the test.
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?tier=B_VALIDATED&theme=macro&min_sharpe=1.0&limit=50")
    assert r.status_code == 200
    body = r.json()
    for item in body["items"]:
        assert item["tier"] == "B_VALIDATED"
        assert item["theme_a"] == "macro" or item["theme_b"] == "macro"
        if item["oos_sharpe"] is not None:
            assert item["oos_sharpe"] >= 1.0


def test_leaderboard_item_has_required_slim_fields() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?limit=1")
    item = r.json()["items"][0]
    for k in (
        "pair_id",
        "tier",
        "theme_a",
        "theme_b",
        "oos_sharpe",
        "max_dd",
        "half_life_days",
        "beta_hedge",
    ):
        assert k in item


# --- /alpha-hub/strategy/{pair_id} ------------------------------------------


def test_strategy_detail_returns_full_record() -> None:
    client = _make_app()
    pool = client.get("/alpha-hub/leaderboard?limit=1").json()["items"]
    pair_id = pool[0]["pair_id"]
    r = client.get(f"/alpha-hub/strategy/{pair_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["pair_id"] == pair_id
    # Full record has additional keys not present in slim leaderboard view.
    assert "rationale" in body or "deploy_signal_logic" in body


def test_strategy_detail_unknown_returns_404() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/strategy/__definitely_not_a_pair__")
    assert r.status_code == 404


# --- /alpha-hub/live-panel --------------------------------------------------


def test_live_panel_returns_three_buckets() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/live-panel")
    assert r.status_code == 200
    body = r.json()
    assert "production" in body
    assert "watchlist" in body
    assert "graveyard" in body
    # Production capped at 3.
    assert len(body["production"]) <= 3
    # Watchlist capped at 10.
    assert len(body["watchlist"]) <= 10
    # Graveyard capped at 5.
    assert len(body["graveyard"]) <= 5


def test_live_panel_production_only_a_tiers() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/live-panel")
    body = r.json()
    for it in body["production"]:
        assert it["tier"] in {"A_STRUCTURAL", "A_GOLD"}


def test_live_panel_watchlist_only_b_validated() -> None:
    client = _make_app()
    r = client.get("/alpha-hub/live-panel")
    body = r.json()
    for it in body["watchlist"]:
        assert it["tier"] == "B_VALIDATED"


# --- response cache ----------------------------------------------------------


def test_leaderboard_response_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical query params must hit the response cache on the second call.

    Spies on ``_cached_strategies`` (the inner data loader) and asserts
    the handler invokes it once, not twice. The response cache TTL is 5
    minutes so two back-to-back calls within a test always overlap.
    """
    import pfm.alpha_hub_router as router_mod

    call_count = {"n": 0}
    real_loader = router_mod._cached_strategies

    def _spy() -> list[dict]:
        call_count["n"] += 1
        return real_loader()

    monkeypatch.setattr(router_mod, "_cached_strategies", _spy)

    client = _make_app()
    params = "?tier=A_STRUCTURAL&min_sharpe=1.0&sort=oos_sharpe&order=desc&limit=10&offset=0"
    r1 = client.get(f"/alpha-hub/leaderboard{params}")
    r2 = client.get(f"/alpha-hub/leaderboard{params}")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()
    assert call_count["n"] == 1, (
        f"loader should run once for identical params; ran {call_count['n']} times"
    )


def test_leaderboard_response_cache_keys_on_query_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different query params must NOT share a cache entry."""
    import pfm.alpha_hub_router as router_mod

    call_count = {"n": 0}
    real_loader = router_mod._cached_strategies

    def _spy() -> list[dict]:
        call_count["n"] += 1
        return real_loader()

    monkeypatch.setattr(router_mod, "_cached_strategies", _spy)

    client = _make_app()
    client.get("/alpha-hub/leaderboard?tier=A_STRUCTURAL&limit=5")
    client.get("/alpha-hub/leaderboard?tier=B_VALIDATED&limit=5")
    # Two distinct param sets => two distinct cache keys => two loader hits.
    assert call_count["n"] == 2


# --- /alpha-hub/leaderboard?full=true (single source of truth path) ----------


def test_leaderboard_full_returns_every_strategy() -> None:
    """``full=true`` with a generous limit must return every catalog row.

    This is the core anti-regression for the dual-source bug — the
    frontend now pulls the catalog through the API, so any strategy
    that's in ``alpha_strategies.json`` must appear in the response
    (otherwise ``selectAlpha`` lookups by ``pair_id`` will silently
    fail again).
    """
    client = _make_app()
    on_disk = _load_strategies()
    r = client.get("/alpha-hub/leaderboard?full=true&limit=500&sort=oos_sharpe&order=desc")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == len(on_disk)
    assert body["n_returned"] == len(on_disk)
    assert len(body["items"]) == len(on_disk)
    disk_ids = {s["pair_id"] for s in on_disk}
    resp_ids = {it["pair_id"] for it in body["items"]}
    assert disk_ids == resp_ids


def test_leaderboard_full_preserves_raw_fields() -> None:
    """``full=true`` items must include rich fields the slim view drops.

    The slim ``LeaderboardItem`` projection strips ``a_name``,
    ``b_name``, ``sharpe_ci_lo``, ``rationale``, ``data_quality_warning``
    and many others. The frontend relies on these for card rendering
    and lookups — losing them was half of the dual-source bug.
    """
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?full=true&limit=500")
    body = r.json()
    items = body["items"]
    # At least one item should carry richer keys the slim view drops.
    rich_keys = ("a_name", "b_name", "rationale", "sharpe_ci_lo", "perm_p")
    matches = [it for it in items if any(k in it for k in rich_keys)]
    assert matches, "expected some items to carry the rich-catalog fields"


def test_leaderboard_full_includes_meta_block() -> None:
    """The ``meta`` block must carry the top-level summary counts.

    The frontend's hero strip and meta line read ``n_curated``,
    ``n_factors_in_catalog``, ``lookback_start`` etc. directly off the
    response, so the meta block is the only way the API can fully
    replace the static-JSON fetch.
    """
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?full=true&limit=500")
    body = r.json()
    meta = body.get("meta")
    assert isinstance(meta, dict)
    for k in ("n_curated", "n_factors_in_catalog", "lookback_start", "lookback_end"):
        assert k in meta, f"meta block missing {k}"


def test_leaderboard_full_preserves_data_quality_warning_flag() -> None:
    """Sanitized rows must keep their ``data_quality_warning`` flag.

    Previously ``data_quality_warning`` only existed on the static JSON;
    the slim API view stripped it. With ``full=true`` we re-emit the
    raw catalog dicts so warning rows surface to the UI.
    """
    on_disk = _load_strategies()
    warned = [s for s in on_disk if s.get("data_quality_warning")]
    if not warned:
        pytest.skip("no data_quality_warning rows in current catalog")
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?full=true&limit=500")
    body = r.json()
    by_id = {it["pair_id"]: it for it in body["items"]}
    for s in warned:
        item = by_id.get(s["pair_id"])
        assert item is not None
        assert item.get("data_quality_warning") == s["data_quality_warning"]


def test_leaderboard_full_default_off_keeps_slim_envelope() -> None:
    """Calling without ``full`` must still return the slim envelope."""
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?limit=3")
    assert r.status_code == 200
    body = r.json()
    assert body.get("meta") in (None,)  # null in slim mode
    item = body["items"][0]
    # Slim items only carry the fields declared on LeaderboardItem.
    assert "pair_id" in item
    assert "rationale" not in item
    assert "a_name" not in item


def test_leaderboard_full_pagination_and_filters_still_apply() -> None:
    """``full=true`` must obey limit/offset/tier filters like the slim view."""
    client = _make_app()
    r = client.get("/alpha-hub/leaderboard?full=true&tier=B_VALIDATED&limit=3")
    body = r.json()
    assert body["limit"] == 3
    assert len(body["items"]) <= 3
    for it in body["items"]:
        assert it["tier"] == "B_VALIDATED"
