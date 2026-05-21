"""Offline tests for :mod:`pfm.arb.live_pricing`.

All upstream HTTP is faked by injecting a ``kalshi_client`` (object exposing
``BASE_URL`` + ``_request``) and a ``poly_http`` (object exposing ``.get``).
No network is ever touched.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pfm.arb.live_pricing import (
    compute_binary_arb,
    kalshi_taker_fee,
    make_price_fn,
    poly_taker_fee,
)

# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class FakeKalshiClient:
    """Minimal stand-in: exposes ``BASE_URL`` + ``_request`` like the real one."""

    BASE_URL = "https://kalshi.test/v2"

    def __init__(self, market: dict[str, Any] | None, *, raise_on_call: bool = False):
        self._market = market
        self._raise = raise_on_call
        self.calls: list[str] = []

    def _request(self, method: str, url: str, **_kw: Any) -> _FakeResp:
        self.calls.append(url)
        if self._raise:
            raise RuntimeError("boom")
        return _FakeResp({"market": self._market})


class FakePolyHttp:
    """Routes ``/events`` to a gamma payload and ``/book`` to per-token books."""

    def __init__(
        self,
        *,
        event: list | None,
        books: dict[str, dict] | None = None,
        raise_on: str | None = None,
    ):
        self._event = event
        self._books = books or {}
        self._raise_on = raise_on
        self.calls: list[str] = []

    def get(self, url: str, *, params: dict | None = None, timeout: float | None = None):
        self.calls.append(url)
        if self._raise_on and self._raise_on in url:
            raise RuntimeError("network down")
        if url.endswith("/events"):
            return _FakeResp(self._event)
        if url.endswith("/book"):
            tok = (params or {}).get("token_id", "")
            return _FakeResp(self._books.get(tok, {}))
        return _FakeResp(None)


def _gamma_event(yes_tok: str = "YES_T", no_tok: str = "NO_T") -> list:
    """A one-market gamma event with YES/NO tokens (clobTokenIds as a string)."""
    return [
        {
            "markets": [
                {
                    "closed": False,
                    "active": True,
                    "clobTokenIds": f'["{yes_tok}", "{no_tok}"]',
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.55", "0.45"]',
                    "bestAsk": "0.55",
                    "bestBid": "0.54",
                }
            ]
        }
    ]


def _book(price: float, size: float = 100.0) -> dict:
    return {"asks": [{"price": str(price), "size": str(size)}]}


# ---------------------------------------------------------------------------
# Fee helpers.
# ---------------------------------------------------------------------------


def test_kalshi_fee_ceils_to_cents() -> None:
    # 0.07 * 0.5 * 0.5 = 0.0175 -> ceil to 0.02.
    assert kalshi_taker_fee(0.5) == pytest.approx(0.02)
    # Endpoints have zero fee.
    assert kalshi_taker_fee(0.0) == 0.0
    assert kalshi_taker_fee(1.0) == 0.0


def test_poly_fee_quadratic() -> None:
    assert poly_taker_fee(0.5) == pytest.approx(0.04 * 0.25)
    assert poly_taker_fee(0.5, fee_rate=0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_binary_arb.
# ---------------------------------------------------------------------------


def test_compute_binary_arb_detects_arb() -> None:
    prices = {"kalshi_yes_ask": 0.45, "poly_no_price": 0.45}
    out = compute_binary_arb(prices)
    assert out["has_arb"] is True
    assert out["best_side"] == "kalshi_yes_poly_no"
    assert out["cost"] == pytest.approx(0.90)
    assert out["profit_pct"] == pytest.approx(10.0)


def test_compute_binary_arb_no_arb_when_sum_ge_one() -> None:
    prices = {"kalshi_yes_ask": 0.55, "poly_no_price": 0.50}
    out = compute_binary_arb(prices)
    assert out["has_arb"] is False
    assert out["cost"] == pytest.approx(1.05)
    assert out["profit_pct"] < 0


def test_compute_binary_arb_picks_cheaper_leg() -> None:
    # Leg A (yes+no) = 0.98; Leg B (no+yes) = 0.90 -> picks B.
    prices = {
        "kalshi_yes_ask": 0.50,
        "poly_no_price": 0.48,
        "kalshi_no_ask": 0.40,
        "poly_yes_price": 0.50,
    }
    out = compute_binary_arb(prices)
    assert out["best_side"] == "kalshi_no_poly_yes"
    assert out["cost"] == pytest.approx(0.90)
    assert out["has_arb"] is True


def test_compute_binary_arb_no_legs_quoted() -> None:
    out = compute_binary_arb({"kalshi_yes_ask": 0.4})  # missing poly_no_price
    assert out["has_arb"] is False
    assert out["best_side"] == ""
    assert math.isinf(out["cost"])


# ---------------------------------------------------------------------------
# make_price_fn — happy paths.
# ---------------------------------------------------------------------------


def test_price_fn_returns_arb_quote_no_fees() -> None:
    # Kalshi yes_ask 0.45, no_ask 0.55 (from dollars fields).
    kalshi = FakeKalshiClient(
        {
            "yes_ask_dollars": 0.45,
            "no_ask_dollars": 0.55,
            "yes_bid_dollars": 0.44,
            "no_bid_dollars": 0.54,
        }
    )
    # Poly YES book best ask 0.55, NO book best ask 0.45.
    http = FakePolyHttp(
        event=_gamma_event(),
        books={"YES_T": _book(0.55), "NO_T": _book(0.45)},
    )
    price_fn = make_price_fn(kalshi_client=kalshi, poly_http=http, fee_aware=False)
    quote = price_fn("KX-TEST", "test-slug")

    assert quote is not None
    assert quote["kalshi_yes_ask"] == pytest.approx(0.45)
    assert quote["poly_no_price"] == pytest.approx(0.45)
    # Leg A: 0.45 + 0.45 = 0.90 < 1 -> arb.
    out = compute_binary_arb(quote)
    assert out["has_arb"] is True
    assert out["profit_pct"] == pytest.approx(10.0)


def test_price_fn_no_arb_quote() -> None:
    kalshi = FakeKalshiClient(
        {
            "yes_ask_dollars": 0.55,
            "no_ask_dollars": 0.50,
            "yes_bid_dollars": 0.50,
            "no_bid_dollars": 0.45,
        }
    )
    http = FakePolyHttp(
        event=_gamma_event(),
        books={"YES_T": _book(0.55), "NO_T": _book(0.50)},
    )
    price_fn = make_price_fn(kalshi_client=kalshi, poly_http=http, fee_aware=False)
    quote = price_fn("KX-TEST", "test-slug")
    assert quote is not None
    # Both legs >= 1: 0.55+0.50=1.05 and 0.50+0.55=1.05.
    assert compute_binary_arb(quote)["has_arb"] is False


def test_price_fn_fee_aware_raises_net_cost() -> None:
    kalshi = FakeKalshiClient(
        {
            "yes_ask_dollars": 0.45,
            "no_ask_dollars": 0.55,
            "yes_bid_dollars": 0.44,
            "no_bid_dollars": 0.54,
        }
    )
    http = FakePolyHttp(
        event=_gamma_event(),
        books={"YES_T": _book(0.55), "NO_T": _book(0.45)},
    )
    raw = make_price_fn(kalshi_client=kalshi, poly_http=http, fee_aware=False)("KX", "slug")
    netd = make_price_fn(kalshi_client=kalshi, poly_http=http, fee_aware=True)("KX", "slug")
    assert raw is not None and netd is not None
    # Fees are non-negative and strictly raise each interior-priced ask.
    assert netd["kalshi_yes_ask"] > raw["kalshi_yes_ask"]
    assert netd["poly_no_price"] > raw["poly_no_price"]
    # Net cost of the arb leg is higher than raw.
    assert compute_binary_arb(netd)["cost"] > compute_binary_arb(raw)["cost"]
    # Specifically, kalshi 0.45 -> +ceil(0.07*0.45*0.55*100)/100 = +0.02.
    assert netd["kalshi_yes_ask"] == pytest.approx(0.45 + kalshi_taker_fee(0.45))


def test_price_fn_falls_back_to_gamma_prices_when_book_empty() -> None:
    kalshi = FakeKalshiClient(
        {
            "yes_ask_dollars": 0.45,
            "no_ask_dollars": 0.55,
            "yes_bid_dollars": 0.44,
            "no_bid_dollars": 0.54,
        }
    )
    # No books supplied -> _book returns {} -> fall back to outcomePrices.
    http = FakePolyHttp(event=_gamma_event(), books={})
    quote = make_price_fn(kalshi_client=kalshi, poly_http=http, fee_aware=False)("KX", "slug")
    assert quote is not None
    # outcomePrices = ["0.55"(yes), "0.45"(no)].
    assert quote["poly_yes_price"] == pytest.approx(0.55)
    assert quote["poly_no_price"] == pytest.approx(0.45)


def test_price_fn_uses_cent_fields_when_no_dollars() -> None:
    # Legacy integer-cent fields only.
    kalshi = FakeKalshiClient({"yes_ask": 45, "no_ask": 55, "yes_bid": 44, "no_bid": 54})
    http = FakePolyHttp(event=_gamma_event(), books={"YES_T": _book(0.55), "NO_T": _book(0.45)})
    quote = make_price_fn(kalshi_client=kalshi, poly_http=http, fee_aware=False)("KX", "slug")
    assert quote is not None
    assert quote["kalshi_yes_ask"] == pytest.approx(0.45)
    assert quote["kalshi_no_ask"] == pytest.approx(0.55)


def test_price_fn_derives_no_ask_from_complement() -> None:
    # Only YES side quoted; NO ask derived as 1 - yes_bid.
    kalshi = FakeKalshiClient({"yes_ask_dollars": 0.45, "yes_bid_dollars": 0.44})
    http = FakePolyHttp(event=_gamma_event(), books={"YES_T": _book(0.55), "NO_T": _book(0.45)})
    quote = make_price_fn(kalshi_client=kalshi, poly_http=http, fee_aware=False)("KX", "slug")
    assert quote is not None
    assert quote["kalshi_no_ask"] == pytest.approx(1.0 - 0.44)


# ---------------------------------------------------------------------------
# make_price_fn — graceful degradation (never raise -> None).
# ---------------------------------------------------------------------------


def test_price_fn_none_on_kalshi_failure() -> None:
    kalshi = FakeKalshiClient(None, raise_on_call=True)
    http = FakePolyHttp(event=_gamma_event(), books={"YES_T": _book(0.45), "NO_T": _book(0.45)})
    assert make_price_fn(kalshi_client=kalshi, poly_http=http)("KX", "slug") is None


def test_price_fn_none_on_missing_kalshi_market() -> None:
    kalshi = FakeKalshiClient(None)  # {"market": None}
    http = FakePolyHttp(event=_gamma_event())
    assert make_price_fn(kalshi_client=kalshi, poly_http=http)("KX", "slug") is None


def test_price_fn_none_on_poly_event_missing() -> None:
    kalshi = FakeKalshiClient({"yes_ask_dollars": 0.45, "no_ask_dollars": 0.55})
    http = FakePolyHttp(event=[])  # empty gamma response
    assert make_price_fn(kalshi_client=kalshi, poly_http=http)("KX", "slug") is None


def test_price_fn_none_on_poly_network_error() -> None:
    kalshi = FakeKalshiClient({"yes_ask_dollars": 0.45, "no_ask_dollars": 0.55})
    http = FakePolyHttp(event=_gamma_event(), raise_on="/events")
    assert make_price_fn(kalshi_client=kalshi, poly_http=http)("KX", "slug") is None


def test_price_fn_none_when_no_poly_price_anywhere() -> None:
    kalshi = FakeKalshiClient({"yes_ask_dollars": 0.45, "no_ask_dollars": 0.55})
    # No books, and gamma market has no prices/bestAsk/bestBid.
    event = [
        {
            "markets": [
                {
                    "closed": False,
                    "active": True,
                    "clobTokenIds": '["YES_T", "NO_T"]',
                    "outcomes": '["Yes", "No"]',
                }
            ]
        }
    ]
    http = FakePolyHttp(event=event, books={})
    assert make_price_fn(kalshi_client=kalshi, poly_http=http)("KX", "slug") is None


# ---------------------------------------------------------------------------
# Integration with the discovery pipeline gate.
# ---------------------------------------------------------------------------


def test_quote_matches_pipeline_arb_gate() -> None:
    """The pipeline's gate (min of both legs < 1) agrees with compute_binary_arb."""
    from pfm.arb.discovery_pipeline import _arb_cost

    kalshi = FakeKalshiClient(
        {
            "yes_ask_dollars": 0.45,
            "no_ask_dollars": 0.55,
            "yes_bid_dollars": 0.44,
            "no_bid_dollars": 0.54,
        }
    )
    http = FakePolyHttp(event=_gamma_event(), books={"YES_T": _book(0.55), "NO_T": _book(0.45)})
    quote = make_price_fn(kalshi_client=kalshi, poly_http=http, fee_aware=False)("KX", "slug")
    assert quote is not None
    assert _arb_cost(quote) == pytest.approx(compute_binary_arb(quote)["cost"])
