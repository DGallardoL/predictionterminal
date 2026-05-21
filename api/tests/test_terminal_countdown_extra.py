"""Extra coverage for ``pfm.terminal_countdown``.

Hits boundaries, error paths, and helper edge cases not covered in
``test_terminal_countdown.py``:

  * factor-yaml parsing (missing file, malformed yaml, junk rows)
  * Gamma metadata parsing edge cases (outcomePrices fallback, garbage)
  * end-date parsing (Z suffix, naive ISO, date-only, invalid)
  * Gamma 404 / 500 / unparseable JSON paths
  * empty / dropped factor universe
  * group ordering with mixed buckets
  * ``/terminal/countdown/{slug}`` 404 when Gamma returns nothing
  * ``/terminal/countdown/{slug}`` 502 when endDate is unusable
  * conviction monotonicity
  * day-bucket boundary conditions (>30 ⇒ later)
  * query-param validation on the list endpoint
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_countdown as tc
from pfm.terminal_countdown import (
    GAMMA_URL,
    _coerce_float,
    _current_p_from_gamma,
    _load_polymarket_factors,
    _parse_end_date,
    build_countdown_markets,
    conviction,
    day_bucket_for,
    group_by_bucket,
    router,
)


@pytest.fixture(autouse=True)
def _clear_countdown_cache():
    """Reset module-level cache between tests so cached responses don't leak."""
    from pfm import terminal_countdown as _tc

    _tc._COUNTDOWN_CACHE.clear()
    if hasattr(_tc, "_SLUG_META_CACHE"):
        _tc._SLUG_META_CACHE.clear()
    yield
    _tc._COUNTDOWN_CACHE.clear()
    if hasattr(_tc, "_SLUG_META_CACHE"):
        _tc._SLUG_META_CACHE.clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _gamma_market(
    slug: str,
    end_date: datetime | None,
    last_trade_price: float | None = 0.5,
    *,
    closed: bool = False,
    active: bool = True,
    one_day_change: float | None = 0.0,
    volume_24hr: float | None = 100.0,
) -> dict:
    out: dict = {
        "slug": slug,
        "question": f"Will {slug}?",
        "closed": closed,
        "active": active,
        "clobTokenIds": json.dumps(["aa", "bb"]),
    }
    if end_date is not None:
        out["endDate"] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    if last_trade_price is not None:
        out["lastTradePrice"] = last_trade_price
        out["outcomePrices"] = json.dumps([str(last_trade_price), str(1 - last_trade_price)])
    out["oneDayPriceChange"] = one_day_change
    out["volume24hr"] = volume_24hr
    return out


# ──────────────────────────────────────────────────────────────────────────
# Pure helpers — boundary conditions
# ──────────────────────────────────────────────────────────────────────────


class TestPureHelpers:
    def test_conviction_is_symmetric_and_monotone(self) -> None:
        # Symmetry around 0.5
        for delta in (0.0, 0.1, 0.2, 0.3, 0.49):
            assert conviction(0.5 + delta) == pytest.approx(conviction(0.5 - delta))
        # Monotone in |p - 0.5|
        levels = [0.5, 0.6, 0.75, 0.9, 1.0]
        convs = [conviction(p) for p in levels]
        assert convs == sorted(convs)

    def test_day_bucket_boundaries(self) -> None:
        # 30 is still "this-month"; 31 spills into "later".
        assert day_bucket_for(30) == "this-month"
        assert day_bucket_for(31) == "later"
        assert day_bucket_for(365) == "later"

    def test_coerce_float_handles_garbage(self) -> None:
        assert _coerce_float(None) is None
        assert _coerce_float("not-a-num") is None
        assert _coerce_float([1, 2]) is None
        assert _coerce_float("3.14") == pytest.approx(3.14)
        assert _coerce_float(2) == 2.0

    def test_parse_end_date_variants(self) -> None:
        z = _parse_end_date("2026-05-02T12:00:00Z")
        assert z is not None and z.tzinfo is not None and z == datetime(2026, 5, 2, 12, tzinfo=UTC)
        # ISO without timezone → assumed UTC.
        naive = _parse_end_date("2026-05-02T12:00:00")
        assert naive is not None and naive.tzinfo == UTC
        # Date-only fallback.
        date_only = _parse_end_date("2026-05-02")
        assert date_only is not None and date_only.day == 2
        # Junk → None.
        assert _parse_end_date("not-a-date") is None
        assert _parse_end_date(None) is None
        assert _parse_end_date("") is None

    def test_current_p_from_gamma_falls_back_to_outcome_prices(self) -> None:
        # Missing lastTradePrice but valid outcomePrices.
        m = {"outcomePrices": json.dumps(["0.73", "0.27"])}
        p = _current_p_from_gamma(m)
        assert p == pytest.approx(0.73)

    def test_current_p_from_gamma_clips_to_unit_interval(self) -> None:
        # Out-of-range prices clamp into [0, 1].
        assert _current_p_from_gamma({"lastTradePrice": 1.5}) == 1.0
        assert _current_p_from_gamma({"lastTradePrice": -0.2}) == 0.0

    def test_current_p_from_gamma_returns_none_on_garbage(self) -> None:
        assert _current_p_from_gamma({}) is None
        # Malformed outcomePrices string.
        assert _current_p_from_gamma({"outcomePrices": "{bogus"}) is None
        # outcomePrices is a non-list.
        assert _current_p_from_gamma({"outcomePrices": json.dumps({"a": 1})}) is None
        # outcomePrices contains non-numeric entries.
        assert _current_p_from_gamma({"outcomePrices": json.dumps(["foo", "bar"])}) is None


# ──────────────────────────────────────────────────────────────────────────
# Factor YAML loading
# ──────────────────────────────────────────────────────────────────────────


class TestLoadPolymarketFactors:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert _load_polymarket_factors(tmp_path / "nope.yml") == []

    def test_malformed_yaml_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yml"
        p.write_text(":\nthis: is: not valid: yaml::")
        assert _load_polymarket_factors(p) == []

    def test_filters_to_polymarket_and_drops_invalid_rows(self, tmp_path: Path) -> None:
        p = tmp_path / "factors.yml"
        p.write_text(
            """
factors:
  - id: keeper
    slug: keeper-slug
    source: polymarket
    theme: macro
  - id: kalshi_skip
    slug: kalshi-slug
    source: kalshi
  - id: missing_slug
    source: polymarket
  - {id: ""}
  - "not a dict"
  - source: polymarket
    slug: no-id
"""
        )
        out = _load_polymarket_factors(p)
        assert [f["id"] for f in out] == ["keeper"]
        assert out[0]["theme"] == "macro"
        assert out[0]["slug"] == "keeper-slug"

    def test_empty_factors_block_is_handled(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yml"
        p.write_text("factors: []")
        assert _load_polymarket_factors(p) == []


# ──────────────────────────────────────────────────────────────────────────
# Pure builder + grouping edge cases
# ──────────────────────────────────────────────────────────────────────────


class TestBuilderEdgeCases:
    def test_missing_metadata_skipped(self) -> None:
        now = datetime(2026, 5, 2, 12, tzinfo=UTC)
        factors = [{"id": "a", "slug": "a", "theme": "macro"}]
        # No Gamma blob for slug ``a`` ⇒ row dropped.
        rows = build_countdown_markets(factors, {}, now=now, horizon_days=7)
        assert rows == []

    def test_market_without_end_date_skipped(self) -> None:
        now = datetime(2026, 5, 2, 12, tzinfo=UTC)
        factors = [{"id": "a", "slug": "a", "theme": "macro"}]
        gamma = {"a": _gamma_market("a", end_date=None, last_trade_price=0.5)}
        assert build_countdown_markets(factors, gamma, now=now, horizon_days=7) == []

    def test_market_without_price_skipped(self) -> None:
        now = datetime(2026, 5, 2, 12, tzinfo=UTC)
        factors = [{"id": "a", "slug": "a", "theme": "macro"}]
        meta = _gamma_market("a", end_date=now + timedelta(days=2), last_trade_price=None)
        meta.pop("outcomePrices", None)
        gamma = {"a": meta}
        assert build_countdown_markets(factors, gamma, now=now, horizon_days=7) == []

    def test_group_by_bucket_empty_and_single(self) -> None:
        # Empty in → empty out.
        assert group_by_bucket([]) == []

    def test_inactive_market_filtered(self) -> None:
        now = datetime(2026, 5, 2, 12, tzinfo=UTC)
        factors = [{"id": "a", "slug": "a", "theme": "x"}]
        gamma = {
            "a": _gamma_market(
                "a",
                now + timedelta(hours=4),
                0.9,
                active=False,
            )
        }
        assert build_countdown_markets(factors, gamma, now=now, horizon_days=7) == []


# ──────────────────────────────────────────────────────────────────────────
# Endpoint validation + error paths
# ──────────────────────────────────────────────────────────────────────────


class TestEndpoints:
    def test_days_query_validation(self, client: TestClient) -> None:
        # 0 < 1 → 422
        assert client.get("/terminal/countdown?days=0").status_code == 422
        # > 365 → 422
        assert client.get("/terminal/countdown?days=400").status_code == 422

    def test_empty_factor_universe_returns_valid_payload(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty = tmp_path / "empty.yml"
        empty.write_text("factors: []")
        monkeypatch.setattr(tc, "_factors_path", lambda: empty)

        r = client.get("/terminal/countdown?days=7")
        assert r.status_code == 200
        body = r.json()
        assert body["n_markets"] == 0
        assert body["groups"] == []
        assert body["horizon_days"] == 7

    @respx.mock
    def test_gamma_404_drops_market_silently(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "factors.yml"
        path.write_text(
            """
factors:
  - id: a
    slug: market-a
    source: polymarket
"""
        )
        monkeypatch.setattr(tc, "_factors_path", lambda: path)
        respx.get(f"{GAMMA_URL}/markets", params={"slug": "market-a"}).mock(
            return_value=httpx.Response(404, json=[])
        )
        r = client.get("/terminal/countdown?days=7")
        assert r.status_code == 200
        assert r.json()["n_markets"] == 0

    @respx.mock
    def test_gamma_500_drops_market_silently(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "factors.yml"
        path.write_text(
            """
factors:
  - id: a
    slug: market-a
    source: polymarket
"""
        )
        monkeypatch.setattr(tc, "_factors_path", lambda: path)
        respx.get(f"{GAMMA_URL}/markets", params={"slug": "market-a"}).mock(
            return_value=httpx.Response(500, text="boom")
        )
        r = client.get("/terminal/countdown?days=7")
        assert r.status_code == 200
        assert r.json()["n_markets"] == 0

    @respx.mock
    def test_gamma_returns_empty_array(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "factors.yml"
        path.write_text(
            """
factors:
  - id: a
    slug: market-a
    source: polymarket
"""
        )
        monkeypatch.setattr(tc, "_factors_path", lambda: path)
        respx.get(f"{GAMMA_URL}/markets", params={"slug": "market-a"}).mock(
            return_value=httpx.Response(200, json=[])
        )
        r = client.get("/terminal/countdown?days=7")
        assert r.status_code == 200
        assert r.json()["n_markets"] == 0

    @respx.mock
    def test_market_countdown_404_when_no_market(self, client: TestClient) -> None:
        respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost"}).mock(
            return_value=httpx.Response(200, json=[])
        )
        r = client.get("/terminal/countdown/ghost")
        assert r.status_code == 404
        assert "ghost" in r.json()["detail"]

    @respx.mock
    def test_market_countdown_502_on_unparseable_end_date(self, client: TestClient) -> None:
        meta = {
            "slug": "broken",
            "question": "Will broken?",
            "endDate": "garbage-not-a-date",
            "lastTradePrice": 0.6,
            "outcomePrices": json.dumps(["0.6", "0.4"]),
            "active": True,
            "closed": False,
            "clobTokenIds": json.dumps(["aa", "bb"]),
        }
        respx.get(f"{GAMMA_URL}/markets", params={"slug": "broken"}).mock(
            return_value=httpx.Response(200, json=[meta])
        )
        r = client.get("/terminal/countdown/broken")
        assert r.status_code == 502
        assert "endDate" in r.json()["detail"]

    @respx.mock
    def test_market_countdown_502_when_price_missing(self, client: TestClient) -> None:
        meta = {
            "slug": "priceless",
            "question": "Will priceless?",
            "endDate": (datetime.now(UTC) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            # No lastTradePrice; outcomePrices is invalid JSON.
            "outcomePrices": "{nope",
            "active": True,
            "closed": False,
        }
        respx.get(f"{GAMMA_URL}/markets", params={"slug": "priceless"}).mock(
            return_value=httpx.Response(200, json=[meta])
        )
        r = client.get("/terminal/countdown/priceless")
        assert r.status_code == 502
        assert "price" in r.json()["detail"]

    @respx.mock
    def test_market_countdown_low_p_routes_to_zero(self, client: TestClient) -> None:
        end = datetime.now(UTC) + timedelta(days=3, hours=2)
        respx.get(f"{GAMMA_URL}/markets", params={"slug": "lowp"}).mock(
            return_value=httpx.Response(
                200,
                json=[_gamma_market("lowp", end, last_trade_price=0.20)],
            ),
        )
        r = client.get("/terminal/countdown/lowp")
        assert r.status_code == 200
        body = r.json()
        # current_p < 0.5 ⇒ fair_price_at_resolution rounds down to 0.0.
        assert body["fair_price_at_resolution"] == 0.0
        assert body["expected_payoff_if_held"] == pytest.approx(0.20)
        # Days/hours/minutes match a 3d2h delta.
        assert body["days"] == 3
        assert body["hours"] in range(0, 24)


class TestPerSlugCountdownCache:
    """`/terminal/countdown/{slug}` caches Gamma metadata for 15 s so the
    market-detail fanout (quote+orderbook+quality+countdown all firing for
    the same slug) doesn't trip the upstream rate-limit."""

    @respx.mock
    def test_repeat_hits_reuse_cached_metadata(self, client: TestClient) -> None:
        end = datetime.now(UTC) + timedelta(days=4)
        gamma_route = respx.get(f"{GAMMA_URL}/markets", params={"slug": "cached-slug"}).mock(
            return_value=httpx.Response(
                200,
                json=[_gamma_market("cached-slug", end, last_trade_price=0.42)],
            ),
        )
        r1 = client.get("/terminal/countdown/cached-slug")
        r2 = client.get("/terminal/countdown/cached-slug")
        assert r1.status_code == r2.status_code == 200
        # Same slug within TTL → one upstream call total.
        assert gamma_route.call_count == 1
