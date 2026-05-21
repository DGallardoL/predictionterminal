"""Live cross-venue binary quote fetcher for the discovery pipeline.

This module supplies a ``price_fn`` for
:func:`pfm.arb.discovery_pipeline.run_discovery_step`. The pipeline calls
``price_fn(kalshi_ticker, poly_slug) -> dict | None`` for each surviving
candidate and flags an arbitrage when::

    min(kalshi_yes_ask + poly_no_price, kalshi_no_ask + poly_yes_price) < 1.0

i.e. one of the two complementary legs (buy YES on Kalshi + NO on Polymarket,
or buy NO on Kalshi + YES on Polymarket) costs less than the guaranteed \\$1
payout. The returned prices are *net cost to take* each side, in ``[0, 1]``.

Design
------
* **Kalshi** is fetched via :class:`pfm.sources.kalshi.KalshiClient` — the
  cleanest pfm-importable, sync, rate-limited path. The market detail endpoint
  exposes ``yes_bid``/``yes_ask`` (and ``*_dollars`` variants); the *ask to buy
  YES* is ``yes_ask`` and the *ask to buy NO* is ``no_ask`` (≈ ``1 - yes_bid``
  when no explicit NO side is quoted, since YES + NO = 1 on a binary).
* **Polymarket** is fetched with a sync ``httpx.Client`` against Gamma
  (``/events?slug=``) to resolve the binary's YES/NO ``clobTokenIds`` (the
  double-``json.loads`` trap — ``clobTokenIds`` arrives as a JSON *string*
  inside the JSON), then CLOB ``/book?token_id=`` for the best ASK on each
  token. Gamma ``outcomePrices`` / ``bestAsk`` are used as a fallback when the
  book is empty.

Fees
----
When ``fee_aware=True`` (default) the *taker* fee is added to each ask so the
pipeline's ``< 1.0`` test compares **net** cost:

* Kalshi: ``ceil(0.07 * p * (1 - p) * 100) / 100`` (rounded up to whole cents,
  matching the exchange's per-contract rounding — see ``arbstuff/arb_engine``).
* Polymarket: ``poly_fee_rate * p * (1 - p)`` (default rate ``0.04``; set the
  rate to ``0.0`` to disable, e.g. for zero-fee markets).

Robustness
----------
The closure **never raises**: any network error, missing market, empty book or
malformed payload degrades to ``None`` so the production discovery loop keeps
running. Each upstream call is bounded by ``timeout_s``. Imports of the heavy
HTTP/source modules are lazy so importing this module is cheap (and test-only
callers can monkeypatch the fetch seams without opening sockets).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

__all__ = [
    "CLOB_BASE_URL",
    "DEFAULT_KALSHI_FEE_RATE",
    "DEFAULT_POLY_FEE_RATE",
    "GAMMA_BASE_URL",
    "compute_binary_arb",
    "kalshi_taker_fee",
    "make_price_fn",
    "poly_taker_fee",
]

#: Gamma + CLOB read endpoints (sync REST; no auth required).
GAMMA_BASE_URL: str = "https://gamma-api.polymarket.com"
CLOB_BASE_URL: str = "https://clob.polymarket.com"

#: Kalshi taker fee rate (7% of p*(1-p), ceil'd to cents).
DEFAULT_KALSHI_FEE_RATE: float = 0.07
#: Polymarket taker fee rate (4% of p*(1-p)); 0.0 disables.
DEFAULT_POLY_FEE_RATE: float = 0.04


# ---------------------------------------------------------------------------
# Fee math (mirrors arbstuff/arb_engine.py — single source of truth here).
# ---------------------------------------------------------------------------


def kalshi_taker_fee(price: float, fee_rate: float = DEFAULT_KALSHI_FEE_RATE) -> float:
    """Kalshi per-contract taker fee, ceil'd up to whole cents.

    Args:
        price: Contract price in ``[0, 1]``.
        fee_rate: Fee rate (default 7%).

    Returns:
        ``ceil(fee_rate * p * (1 - p) * 100) / 100`` — the exchange rounds the
        per-contract fee *up* to the nearest cent.
    """
    raw = fee_rate * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0


def poly_taker_fee(price: float, fee_rate: float = DEFAULT_POLY_FEE_RATE) -> float:
    """Polymarket per-contract taker fee (no cent-rounding).

    Args:
        price: Contract price in ``[0, 1]``.
        fee_rate: Fee rate (default 4%; pass ``0.0`` for zero-fee markets).

    Returns:
        ``fee_rate * p * (1 - p)``.
    """
    return fee_rate * price * (1.0 - price)


# ---------------------------------------------------------------------------
# Pure arb math — exposed for testing/reuse, identical to the pipeline's gate.
# ---------------------------------------------------------------------------


def compute_binary_arb(prices: dict[str, Any]) -> dict[str, Any]:
    """Compute the cheaper cross-venue binary arb leg from a quote dict.

    The two complementary legs are:

    * ``"kalshi_yes_poly_no"`` — ``kalshi_yes_ask + poly_no_price``.
    * ``"kalshi_no_poly_yes"`` — ``kalshi_no_ask + poly_yes_price``.

    A combined cost below ``1.0`` is an arbitrage (guaranteed \\$1 payout for a
    sub-\\$1 stake). This is the exact gate the discovery pipeline applies.

    Args:
        prices: Mapping with any of the four leg keys (prices in ``[0, 1]``).

    Returns:
        ``{has_arb, best_side, cost, profit_pct}``. When neither leg is fully
        quoted, ``has_arb=False``, ``best_side=""``, ``cost=float("inf")`` and
        ``profit_pct=0.0``.
    """
    legs: list[tuple[str, float]] = []

    yes_ask = _as_float(prices.get("kalshi_yes_ask"))
    no_price = _as_float(prices.get("poly_no_price"))
    if yes_ask is not None and no_price is not None:
        legs.append(("kalshi_yes_poly_no", yes_ask + no_price))

    no_ask = _as_float(prices.get("kalshi_no_ask"))
    yes_price = _as_float(prices.get("poly_yes_price"))
    if no_ask is not None and yes_price is not None:
        legs.append(("kalshi_no_poly_yes", no_ask + yes_price))

    if not legs:
        return {
            "has_arb": False,
            "best_side": "",
            "cost": float("inf"),
            "profit_pct": 0.0,
        }

    best_side, cost = min(legs, key=lambda lc: lc[1])
    has_arb = cost < 1.0
    profit_pct = (1.0 - cost) * 100.0
    return {
        "has_arb": has_arb,
        "best_side": best_side,
        "cost": cost,
        "profit_pct": profit_pct,
    }


def _as_float(value: Any) -> float | None:
    """Best-effort float coercion; ``None``/garbage -> ``None``."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Kalshi quote extraction.
# ---------------------------------------------------------------------------


def _kalshi_yes_no_ask(market: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract (yes_ask, no_ask) in ``[0, 1]`` from a Kalshi market dict.

    Prefers explicit ``*_dollars`` fields (already in ``[0, 1]``), falls back
    to integer-cent fields (divide by 100). When the NO side is not quoted, it
    is derived from the binary complement of the YES bid (``no_ask ≈ 1 -
    yes_bid``); likewise ``yes_ask ≈ 1 - no_bid`` if YES ask is missing.

    Args:
        market: Raw Kalshi market dict.

    Returns:
        ``(yes_ask, no_ask)``; either element may be ``None`` if unquoted.
    """

    def _field(*keys: str) -> float | None:
        for k in keys:
            v = _as_float(market.get(k))
            if v is not None:
                # Cent fields (no "_dollars" suffix) are integers 0..100.
                return v / 100.0 if not k.endswith("_dollars") else v
        return None

    yes_ask = _field("yes_ask_dollars", "yes_ask")
    no_ask = _field("no_ask_dollars", "no_ask")
    yes_bid = _field("yes_bid_dollars", "yes_bid")
    no_bid = _field("no_bid_dollars", "no_bid")

    # Derive missing side from the binary complement of the opposite bid.
    if no_ask is None and yes_bid is not None:
        no_ask = max(0.0, min(1.0, 1.0 - yes_bid))
    if yes_ask is None and no_bid is not None:
        yes_ask = max(0.0, min(1.0, 1.0 - no_bid))
    return yes_ask, no_ask


def _fetch_kalshi_market(
    client: Any, kalshi_ticker: str, *, timeout_s: float
) -> dict[str, Any] | None:
    """Fetch a single Kalshi market dict (with yes/no bid/ask), or ``None``.

    Uses the client's private ``_request`` (the rate-limited, 429-retrying
    layer) to hit ``GET /markets/{ticker}``, which carries the quote fields the
    cached ``get_market`` metadata view drops. Swallows all errors -> ``None``.
    """
    try:
        url = f"{client.BASE_URL}/markets/{kalshi_ticker}"
        resp = client._request("GET", url)  # rate-limited 429-retrying seam
        market = resp.json().get("market")
        if not market:
            return None
        return dict(market)
    except Exception:
        return None


def _default_kalshi_client(timeout_s: float) -> Any:
    """Lazily construct a default :class:`pfm.sources.kalshi.KalshiClient`."""
    from pfm.sources.kalshi import KalshiClient  # lazy import

    return KalshiClient(timeout=timeout_s)


# ---------------------------------------------------------------------------
# Polymarket quote extraction.
# ---------------------------------------------------------------------------


def _loads_maybe(value: Any) -> Any:
    """``json.loads`` a value if it's a JSON string, else return as-is.

    Polymarket's ``clobTokenIds`` / ``outcomes`` / ``outcomePrices`` arrive as
    JSON *strings* embedded in the JSON response — the double-loads trap.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _poly_yes_no_tokens(
    market: dict[str, Any],
) -> tuple[str | None, str | None, float | None, float | None]:
    """Resolve (yes_token, no_token, gamma_yes_ask, gamma_no_ask) from a market.

    Maps each ``clobTokenIds`` entry to its ``outcomes`` label so YES/NO are
    correctly assigned regardless of ordering. Also extracts gamma's
    ``outcomePrices`` (or ``bestAsk``) as a no-book fallback ASK estimate.

    Args:
        market: Raw gamma market dict.

    Returns:
        ``(yes_token, no_token, gamma_yes_ask, gamma_no_ask)`` — any may be
        ``None``.
    """
    token_ids = _loads_maybe(market.get("clobTokenIds")) or []
    outcomes = _loads_maybe(market.get("outcomes")) or []
    prices = _loads_maybe(market.get("outcomePrices")) or []

    yes_token: str | None = None
    no_token: str | None = None
    gamma_yes: float | None = None
    gamma_no: float | None = None

    if isinstance(token_ids, list) and isinstance(outcomes, list):
        for i, outcome in enumerate(outcomes):
            label = str(outcome).strip().lower()
            tok = str(token_ids[i]) if i < len(token_ids) else None
            px = _as_float(prices[i]) if i < len(prices) else None
            if label == "yes":
                yes_token, gamma_yes = tok, px
            elif label == "no":
                no_token, gamma_no = tok, px

    # bestAsk on gamma is the YES best ask; NO ≈ 1 - bestBid.
    if gamma_yes is None:
        gamma_yes = _as_float(market.get("bestAsk"))
    if gamma_no is None:
        best_bid = _as_float(market.get("bestBid"))
        if best_bid is not None:
            gamma_no = max(0.0, min(1.0, 1.0 - best_bid))
    return yes_token, no_token, gamma_yes, gamma_no


def _best_ask_from_book(book: dict[str, Any]) -> float | None:
    """Return the lowest ask price from a CLOB ``/book`` payload, or ``None``.

    The CLOB book lists ``asks`` as ``[{price, size}, ...]`` (string values).
    The best ask to *buy* is the minimum price level.
    """
    asks = book.get("asks") if isinstance(book, dict) else None
    if not asks:
        return None
    best: float | None = None
    for level in asks:
        px = _as_float(level.get("price") if isinstance(level, dict) else None)
        sz = _as_float(level.get("size") if isinstance(level, dict) else None)
        if px is None or (sz is not None and sz <= 0):
            continue
        if best is None or px < best:
            best = px
    return best


def _fetch_poly_event(http: Any, poly_slug: str, *, timeout_s: float) -> dict[str, Any] | None:
    """Fetch the first gamma event for ``slug`` (its first market), or ``None``."""
    try:
        resp = http.get(
            f"{GAMMA_BASE_URL}/events",
            params={"slug": poly_slug},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        event = data[0]
        markets = event.get("markets") or []
        if not markets:
            return None
        # First active, non-closed market is the binary we price.
        for m in markets:
            if not m.get("closed") and m.get("active", True):
                return dict(m)
        return dict(markets[0])
    except Exception:
        return None


def _fetch_poly_book(http: Any, token_id: str, *, timeout_s: float) -> dict[str, Any]:
    """Fetch a CLOB order book for ``token_id``; ``{}`` on any failure."""
    if not token_id:
        return {}
    try:
        resp = http.get(
            f"{CLOB_BASE_URL}/book",
            params={"token_id": token_id},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        return dict(body) if isinstance(body, dict) else {}
    except Exception:
        return {}


def _default_http_client(timeout_s: float) -> Any:
    """Lazily construct a default sync ``httpx.Client``."""
    import httpx  # lazy import

    return httpx.Client(timeout=timeout_s)


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


def make_price_fn(
    *,
    kalshi_client: Any = None,
    poly_pool: Any = None,
    poly_http: Any = None,
    timeout_s: float = 4.0,
    fee_aware: bool = True,
    kalshi_fee_rate: float = DEFAULT_KALSHI_FEE_RATE,
    poly_fee_rate: float = DEFAULT_POLY_FEE_RATE,
) -> Callable[[str, str], dict[str, Any] | None]:
    """Build a ``price_fn(kalshi_ticker, poly_slug) -> dict | None`` closure.

    The returned closure fetches the best YES/NO *ask* on each venue, optionally
    adds taker fees so the cost is net, and returns the four keys the discovery
    pipeline reads (``kalshi_yes_ask``, ``poly_no_price``, ``kalshi_no_ask``,
    ``poly_yes_price``) plus a ``raw`` provenance block. It returns ``None`` on
    any fetch failure / missing book and **never raises**.

    Args:
        kalshi_client: A :class:`pfm.sources.kalshi.KalshiClient` (or any object
            exposing ``BASE_URL`` + ``_request``). Lazily constructed if ``None``.
        poly_pool: Accepted for API symmetry with the async pool; ignored when a
            sync ``poly_http`` client is available (the closure is sync because
            the pipeline calls it synchronously).
        poly_http: A sync ``httpx.Client`` (or any object exposing ``.get``) for
            gamma + CLOB. Lazily constructed if ``None``.
        timeout_s: Per-call HTTP timeout in seconds.
        fee_aware: When ``True`` (default), add taker fees to each ask so the
            arb test uses net cost.
        kalshi_fee_rate: Kalshi taker fee rate (default 7%).
        poly_fee_rate: Polymarket taker fee rate (default 4%; ``0.0`` disables).

    Returns:
        The ``price_fn`` closure.
    """
    _kalshi = kalshi_client
    _http = poly_http

    def _ensure_kalshi() -> Any:
        nonlocal _kalshi
        if _kalshi is None:
            _kalshi = _default_kalshi_client(timeout_s)
        return _kalshi

    def _ensure_http() -> Any:
        nonlocal _http
        if _http is None:
            _http = _default_http_client(timeout_s)
        return _http

    def price_fn(kalshi_ticker: str, poly_slug: str) -> dict[str, Any] | None:
        try:
            # --- Kalshi leg ------------------------------------------------
            market = _fetch_kalshi_market(_ensure_kalshi(), kalshi_ticker, timeout_s=timeout_s)
            if not market:
                return None
            kalshi_yes_ask, kalshi_no_ask = _kalshi_yes_no_ask(market)
            if kalshi_yes_ask is None and kalshi_no_ask is None:
                return None

            # --- Polymarket leg --------------------------------------------
            http = _ensure_http()
            pm = _fetch_poly_event(http, poly_slug, timeout_s=timeout_s)
            if not pm:
                return None
            yes_tok, no_tok, gamma_yes, gamma_no = _poly_yes_no_tokens(pm)

            # Prefer the live CLOB book best-ask; fall back to gamma price.
            yes_book = _fetch_poly_book(http, yes_tok or "", timeout_s=timeout_s)
            no_book = _fetch_poly_book(http, no_tok or "", timeout_s=timeout_s)
            poly_yes_price = _best_ask_from_book(yes_book)
            poly_no_price = _best_ask_from_book(no_book)
            if poly_yes_price is None:
                poly_yes_price = gamma_yes
            if poly_no_price is None:
                poly_no_price = gamma_no

            if poly_yes_price is None and poly_no_price is None:
                return None

            # --- Fees ------------------------------------------------------
            if fee_aware:
                if kalshi_yes_ask is not None:
                    kalshi_yes_ask += kalshi_taker_fee(kalshi_yes_ask, kalshi_fee_rate)
                if kalshi_no_ask is not None:
                    kalshi_no_ask += kalshi_taker_fee(kalshi_no_ask, kalshi_fee_rate)
                if poly_yes_price is not None:
                    poly_yes_price += poly_taker_fee(poly_yes_price, poly_fee_rate)
                if poly_no_price is not None:
                    poly_no_price += poly_taker_fee(poly_no_price, poly_fee_rate)

            return {
                "kalshi_yes_ask": kalshi_yes_ask,
                "poly_no_price": poly_no_price,
                "kalshi_no_ask": kalshi_no_ask,
                "poly_yes_price": poly_yes_price,
                "raw": {
                    "kalshi_ticker": kalshi_ticker,
                    "poly_slug": poly_slug,
                    "fee_aware": fee_aware,
                    "kalshi_fee_rate": kalshi_fee_rate,
                    "poly_fee_rate": poly_fee_rate,
                    "poly_yes_token": yes_tok,
                    "poly_no_token": no_tok,
                    "poly_book_used": {
                        "yes": bool(_best_ask_from_book(yes_book) is not None),
                        "no": bool(_best_ask_from_book(no_book) is not None),
                    },
                },
            }
        except Exception:
            # Production must degrade gracefully — never raise from the closure.
            return None

    return price_fn
