"""Tests for ``pfm.terminal.sentiment_leaderboard`` — the
``/terminal/sentiment-leaderboard`` endpoint.

We mock the two upstream entry points so no network is required:

* ``_fetch_top_markets_async`` — returns a fixed list of gamma-shaped market
  dicts.
* ``_get_jumps_endpoint`` — a lightweight async stub that returns canned
  ``TerminalJumpsResponse`` payloads per slug. We can therefore control
  ``n_jumps``, ``n_explained``, and the per-jump ``sentiment_alignment``
  field that drives ``disagrees_pct``.

Test coverage focuses on:

* HTTP 200 happy path + envelope shape
* Sort order (``disagrees_pct`` desc, tie-break on ``n_disagrees``, then
  ``volume_24h``)
* ``min_jumps`` filter excludes thin markets
* Rank field is 1-indexed and dense
* 422 on out-of-range query params
* 503 when no Polymarket client wired into ``app.state``
* 502 when the upstream gamma listing raises ``httpx.HTTPError``
* Cache short-circuits a second call with the same key
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal import sentiment_leaderboard as sl_mod
from pfm.terminal.jumps import TerminalJumpsResponse

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_leaderboard_cache() -> Iterator[None]:
    sl_mod.clear_cache()
    yield
    sl_mod.clear_cache()


class _StubPoly:
    """Empty stub — the real `get_jumps` is mocked out, so the polymarket
    client never gets called in these tests; we just need a truthy value
    to satisfy the ``_get_polymarket_client`` guard."""


def _build_client(*, with_poly: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(sl_mod.router)
    if with_poly:
        app.state.poly = _StubPoly()
    return TestClient(app)


def _mk_market(slug: str, volume_24h: float, question: str | None = None) -> dict[str, Any]:
    """A gamma-listing-shaped market dict — only the fields the endpoint reads."""
    return {
        "slug": slug,
        "volume24hr": volume_24h,
        "question": question or f"Question for {slug}",
    }


def _mk_jump(
    *,
    delta_pp: float = 5.0,
    alignment: str = "disagrees",
    explained: bool = True,
    n_articles: int = 1,
) -> dict[str, Any]:
    """Build a single ``Jump`` dict that can roundtrip through pydantic."""
    return {
        "ts_iso": "2026-05-10T12:00:00+00:00",
        "price_before": 0.40,
        "price_after": 0.40 + delta_pp / 100.0,
        "delta_pp": delta_pp,
        "delta_logit": 0.5 if delta_pp > 0 else -0.5,
        "z_score": 4.2,
        "direction": "up" if delta_pp > 0 else "down",
        "explained": explained,
        "n_articles": n_articles,
        "top_articles": [],
        "news_sentiment_score": -0.3,
        "news_sentiment_label": "negative",
        "sentiment_alignment": alignment,
    }


def _mk_jumps_response(
    slug: str,
    *,
    jumps: list[dict[str, Any]],
) -> TerminalJumpsResponse:
    n = len(jumps)
    n_explained = sum(1 for j in jumps if j.get("explained"))
    return TerminalJumpsResponse(
        slug=slug,
        days=7,
        threshold_mad_k=3.0,
        threshold_min_jump_pp=5.0,
        n_jumps=n,
        n_explained=n_explained,
        explained_pct=round(100.0 * n_explained / max(n, 1), 1),
        jumps=jumps,  # pydantic coerces dicts into Jump models
        interpretation="stub",
    )


def _make_jumps_stub(payloads_by_slug: dict[str, TerminalJumpsResponse]):
    """Build an async stub for `_get_jumps_endpoint` that returns a
    pre-canned ``TerminalJumpsResponse`` per slug. Slugs not in the map
    raise ``KeyError`` — useful to surface accidental fan-outs."""

    async def _stub(*, slug: str, **_kwargs: Any) -> TerminalJumpsResponse:
        return payloads_by_slug[slug]

    return _stub


# ---------------------------------------------------------------------------
# Happy path: shape + sort + tie-break
# ---------------------------------------------------------------------------


def test_endpoint_returns_200_and_correct_envelope() -> None:
    markets = [
        _mk_market("alpha", 100_000.0),
        _mk_market("bravo", 50_000.0),
        _mk_market("charlie", 25_000.0),
    ]
    payloads = {
        "alpha": _mk_jumps_response(
            "alpha",
            jumps=[
                _mk_jump(alignment="disagrees"),
                _mk_jump(alignment="disagrees"),
                _mk_jump(alignment="agrees"),
            ],
        ),
        "bravo": _mk_jumps_response(
            "bravo",
            jumps=[
                _mk_jump(alignment="agrees"),
                _mk_jump(alignment="agrees"),
                _mk_jump(alignment="agrees"),
            ],
        ),
        "charlie": _mk_jumps_response(
            "charlie",
            jumps=[
                _mk_jump(alignment="disagrees"),
                _mk_jump(alignment="disagrees"),
                _mk_jump(alignment="disagrees"),
                _mk_jump(alignment="disagrees"),
            ],
        ),
    }

    fetch_stub = AsyncMock(return_value=markets)
    with (
        patch.object(sl_mod, "_fetch_top_markets_async", fetch_stub),
        patch.object(sl_mod, "_get_jumps_endpoint", _make_jumps_stub(payloads)),
    ):
        c = _build_client()
        r = c.get("/terminal/sentiment-leaderboard?days=7&min_jumps=3")

    assert r.status_code == 200, r.text
    body = r.json()
    # Top-level envelope
    assert body["days"] == 7
    assert body["min_jumps"] == 3
    assert body["n_markets_considered"] == 3
    assert body["n_markets_qualified"] == 3
    assert isinstance(body["rows"], list) and len(body["rows"]) == 3
    assert "interpretation" in body and isinstance(body["interpretation"], str)

    # Row shape — required fields present on every row.
    for row in body["rows"]:
        for field in (
            "rank",
            "slug",
            "name",
            "volume_24h",
            "n_jumps",
            "n_explained",
            "n_disagrees",
            "disagrees_pct",
        ):
            assert field in row, f"missing {field} in {row}"


def test_rows_sorted_by_disagrees_pct_desc() -> None:
    """Charlie has 100% disagrees (4/4); alpha has 66.7% (2/3); bravo has 0%.
    Output must be [charlie, alpha, bravo]."""
    markets = [
        _mk_market("alpha", 100_000.0),
        _mk_market("bravo", 50_000.0),
        _mk_market("charlie", 25_000.0),
    ]
    payloads = {
        "alpha": _mk_jumps_response(
            "alpha",
            jumps=[
                _mk_jump(alignment="disagrees"),
                _mk_jump(alignment="disagrees"),
                _mk_jump(alignment="agrees"),
            ],
        ),
        "bravo": _mk_jumps_response(
            "bravo",
            jumps=[_mk_jump(alignment="agrees")] * 3,
        ),
        "charlie": _mk_jumps_response(
            "charlie",
            jumps=[_mk_jump(alignment="disagrees")] * 4,
        ),
    }
    with (
        patch.object(sl_mod, "_fetch_top_markets_async", AsyncMock(return_value=markets)),
        patch.object(sl_mod, "_get_jumps_endpoint", _make_jumps_stub(payloads)),
    ):
        c = _build_client()
        body = c.get("/terminal/sentiment-leaderboard?min_jumps=3").json()

    slugs_in_order = [r["slug"] for r in body["rows"]]
    assert slugs_in_order == ["charlie", "alpha", "bravo"]
    # 100, 66.7, 0
    pcts = [r["disagrees_pct"] for r in body["rows"]]
    assert pcts == sorted(pcts, reverse=True)


def test_sort_tie_breaks_on_n_disagrees_then_volume() -> None:
    """Two markets tied at 100% disagrees: the one with more disagreements
    (denominator) ranks first. If both are tied on disagrees_pct AND
    n_disagrees, higher 24h volume wins."""
    markets = [
        _mk_market("low_vol_same", 10_000.0),
        _mk_market("high_vol_same", 90_000.0),
        _mk_market("high_vol_more", 50_000.0),
    ]
    payloads = {
        # All three sit at 100% disagrees.
        "low_vol_same": _mk_jumps_response(
            "low_vol_same",
            jumps=[_mk_jump(alignment="disagrees")] * 3,
        ),
        "high_vol_same": _mk_jumps_response(
            "high_vol_same",
            jumps=[_mk_jump(alignment="disagrees")] * 3,
        ),
        "high_vol_more": _mk_jumps_response(
            "high_vol_more",
            jumps=[_mk_jump(alignment="disagrees")] * 5,
        ),
    }
    with (
        patch.object(sl_mod, "_fetch_top_markets_async", AsyncMock(return_value=markets)),
        patch.object(sl_mod, "_get_jumps_endpoint", _make_jumps_stub(payloads)),
    ):
        c = _build_client()
        body = c.get("/terminal/sentiment-leaderboard?min_jumps=3").json()

    slugs_in_order = [r["slug"] for r in body["rows"]]
    # high_vol_more wins on n_disagrees (5 > 3). The two 3-jump rows tie on
    # pct AND n_disagrees; volume tie-break elevates high_vol_same above
    # low_vol_same.
    assert slugs_in_order == ["high_vol_more", "high_vol_same", "low_vol_same"]


# ---------------------------------------------------------------------------
# min_jumps filter
# ---------------------------------------------------------------------------


def test_min_jumps_filter_excludes_thin_markets() -> None:
    markets = [
        _mk_market("thick", 100_000.0),
        _mk_market("thin", 50_000.0),
    ]
    payloads = {
        "thick": _mk_jumps_response("thick", jumps=[_mk_jump(alignment="disagrees")] * 5),
        "thin": _mk_jumps_response("thin", jumps=[_mk_jump(alignment="disagrees")] * 2),
    }
    with (
        patch.object(sl_mod, "_fetch_top_markets_async", AsyncMock(return_value=markets)),
        patch.object(sl_mod, "_get_jumps_endpoint", _make_jumps_stub(payloads)),
    ):
        c = _build_client()
        body = c.get("/terminal/sentiment-leaderboard?min_jumps=3").json()

    assert body["n_markets_considered"] == 2
    assert body["n_markets_qualified"] == 1
    assert [r["slug"] for r in body["rows"]] == ["thick"]


def test_min_jumps_zero_keeps_all_markets() -> None:
    markets = [_mk_market("any", 1_000.0)]
    payloads = {"any": _mk_jumps_response("any", jumps=[])}
    with (
        patch.object(sl_mod, "_fetch_top_markets_async", AsyncMock(return_value=markets)),
        patch.object(sl_mod, "_get_jumps_endpoint", _make_jumps_stub(payloads)),
    ):
        c = _build_client()
        body = c.get("/terminal/sentiment-leaderboard?min_jumps=0").json()
    assert body["n_markets_qualified"] == 1
    assert body["rows"][0]["n_jumps"] == 0
    assert body["rows"][0]["disagrees_pct"] == 0.0


# ---------------------------------------------------------------------------
# Rank field
# ---------------------------------------------------------------------------


def test_ranks_are_dense_and_one_indexed() -> None:
    markets = [
        _mk_market("a", 100.0),
        _mk_market("b", 90.0),
        _mk_market("c", 80.0),
    ]
    payloads = {
        slug: _mk_jumps_response(slug, jumps=[_mk_jump(alignment="disagrees")] * (5 - i))
        for i, slug in enumerate(["a", "b", "c"])
    }
    with (
        patch.object(sl_mod, "_fetch_top_markets_async", AsyncMock(return_value=markets)),
        patch.object(sl_mod, "_get_jumps_endpoint", _make_jumps_stub(payloads)),
    ):
        c = _build_client()
        body = c.get("/terminal/sentiment-leaderboard?min_jumps=3").json()

    ranks = [r["rank"] for r in body["rows"]]
    assert ranks == [1, 2, 3]


# ---------------------------------------------------------------------------
# Validation: 422 on out-of-range query params
# ---------------------------------------------------------------------------


def test_invalid_days_below_min_returns_422() -> None:
    c = _build_client()
    r = c.get("/terminal/sentiment-leaderboard?days=0")
    assert r.status_code == 422


def test_invalid_min_jumps_above_max_returns_422() -> None:
    c = _build_client()
    r = c.get("/terminal/sentiment-leaderboard?min_jumps=51")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 503 when no polymarket client wired in
# ---------------------------------------------------------------------------


def test_returns_503_when_polymarket_client_missing() -> None:
    markets = [_mk_market("alpha", 100.0)]
    with patch.object(sl_mod, "_fetch_top_markets_async", AsyncMock(return_value=markets)):
        c = _build_client(with_poly=False)
        r = c.get("/terminal/sentiment-leaderboard")
    assert r.status_code == 503
    assert "polymarket client not initialized" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 502 on upstream gamma error
# ---------------------------------------------------------------------------


def test_returns_502_when_gamma_upstream_errors() -> None:
    async def _boom(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise httpx.ConnectError("gamma timeout")

    with patch.object(sl_mod, "_fetch_top_markets_async", _boom):
        c = _build_client()
        r = c.get("/terminal/sentiment-leaderboard")
    assert r.status_code == 502
    assert "upstream gamma error" in r.json()["detail"]


# ---------------------------------------------------------------------------
# A failed per-slug fan-out call must not poison the leaderboard.
# ---------------------------------------------------------------------------


def test_one_slug_failure_is_isolated() -> None:
    markets = [
        _mk_market("good", 100_000.0),
        _mk_market("bad", 50_000.0),
    ]
    good_payload = _mk_jumps_response("good", jumps=[_mk_jump(alignment="disagrees")] * 3)

    async def _stub(*, slug: str, **_kwargs: Any) -> TerminalJumpsResponse:
        if slug == "bad":
            raise RuntimeError("boom for bad slug")
        return good_payload

    with (
        patch.object(sl_mod, "_fetch_top_markets_async", AsyncMock(return_value=markets)),
        patch.object(sl_mod, "_get_jumps_endpoint", _stub),
    ):
        c = _build_client()
        r = c.get("/terminal/sentiment-leaderboard?min_jumps=3")

    assert r.status_code == 200
    body = r.json()
    # Both probed; only one qualified (bad failed silently).
    assert body["n_markets_considered"] == 2
    assert body["n_markets_qualified"] == 1
    assert body["rows"][0]["slug"] == "good"


# ---------------------------------------------------------------------------
# Cache: second call same key short-circuits.
# ---------------------------------------------------------------------------


def test_second_call_with_same_key_serves_from_cache() -> None:
    markets = [_mk_market("alpha", 100.0)]
    payloads = {"alpha": _mk_jumps_response("alpha", jumps=[_mk_jump(alignment="disagrees")] * 3)}
    fetch_stub = AsyncMock(return_value=markets)
    with (
        patch.object(sl_mod, "_fetch_top_markets_async", fetch_stub),
        patch.object(sl_mod, "_get_jumps_endpoint", _make_jumps_stub(payloads)),
    ):
        c = _build_client()
        r1 = c.get("/terminal/sentiment-leaderboard?days=7&min_jumps=3")
        r2 = c.get("/terminal/sentiment-leaderboard?days=7&min_jumps=3")
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Only the first call should have hit the gamma listing.
    assert fetch_stub.await_count == 1
    # Same payload returned both times.
    assert r1.json() == r2.json()


# ---------------------------------------------------------------------------
# Empty considered set → interpretation text steers the user.
# ---------------------------------------------------------------------------


def test_zero_qualified_markets_returns_helpful_interpretation() -> None:
    with patch.object(sl_mod, "_fetch_top_markets_async", AsyncMock(return_value=[])):
        c = _build_client()
        body = c.get("/terminal/sentiment-leaderboard").json()
    assert body["n_markets_considered"] == 0
    assert body["n_markets_qualified"] == 0
    assert body["rows"] == []
    assert "Lower min_jumps" in body["interpretation"]
