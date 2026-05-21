"""Related-stocks endpoint for the Terminal panel.

``GET /terminal/related-stocks/{ticker}`` — given a ticker, return the
top-10 factor-correlated peer tickers based on factor-exposure overlap.

Methodology (POC level — see PLAN.md §10 for the phased plan)
-------------------------------------------------------------

For every ticker in a hardcoded universe of ~100 liquid US equities we
fit log-returns against the same top-20 prediction-market factors and
keep the slope coefficients as a "factor exposure vector".  Two
tickers' similarity is then the cosine of their exposure vectors:

    sim(a, b) = (β_a · β_b) / (‖β_a‖ · ‖β_b‖)

The endpoint returns the top-10 peers sorted descending, with the
anchor itself excluded.  Per-peer we surface the ``shared_factors`` —
the factor slugs where BOTH the anchor and the peer have a notable
(``|β| ≥ EXPOSURE_THRESHOLD``) loading.  This intersection is the
quick "why are they similar" hint a user can scan.

The actual regression fit lives behind a swap-able hook
:func:`_default_exposure_fn`. Tests monkeypatch this to a synthetic
mapping so we don't hit the real ``/fit`` machinery (which would need
the Polymarket CLOB + yfinance, slow and non-hermetic). In production
the hook can be wired to ``pfm.regression_core.run_fit`` — the function
signature ``(ticker, factors) -> dict[slug, beta]`` is what matters.

Routing
-------
This module owns its own :class:`fastapi.APIRouter`. Per project
convention ``main.py`` is left untouched here — wire it explicitly
elsewhere via::

    from pfm.terminal.related_stocks_router import router as related_stocks_router
    app.include_router(related_stocks_router)
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from fastapi import Path as FPath

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)


# --- universe ----------------------------------------------------------------
# Hardcoded universe of ~100 liquid US equities. The set spans the major
# sector ETFs plus the largest single-name stocks across tech, financials,
# energy, healthcare, consumer, industrials, and a handful of crypto-adjacent
# names. It's deliberately a static list — the POC doesn't need a survivorship-
# bias-free CRSP universe; what it needs is "names a demo viewer recognises".

UNIVERSE_TICKERS: tuple[str, ...] = (
    # Mega-cap index/sector ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "VOO",
    "VEA",
    "VWO",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
    "GLD",
    "SLV",
    "USO",
    "TLT",
    "HYG",
    # Mega-cap tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "GOOG",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "AVGO",
    "ORCL",
    "ADBE",
    "CRM",
    "AMD",
    "INTC",
    "QCOM",
    "TXN",
    "CSCO",
    "IBM",
    "MU",
    "AMAT",
    "LRCX",
    "KLAC",
    "ASML",
    "TSM",
    # Financials
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "BLK",
    "SCHW",
    "AXP",
    "V",
    "MA",
    "PYPL",
    "COIN",
    "SQ",
    # Healthcare
    "UNH",
    "JNJ",
    "LLY",
    "PFE",
    "ABBV",
    "MRK",
    "TMO",
    "DHR",
    "ABT",
    "BMY",
    "AMGN",
    "GILD",
    "CVS",
    # Consumer
    "WMT",
    "HD",
    "PG",
    "KO",
    "PEP",
    "MCD",
    "NKE",
    "SBUX",
    "COST",
    "DIS",
    "NFLX",
    "TGT",
    "LOW",
    "BABA",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
    "OXY",
    "PSX",
    "MPC",
    # Industrials / Misc
    "BA",
    "CAT",
    "GE",
    "HON",
    "UPS",
    "RTX",
    "LMT",
    "DE",
    # Crypto-adjacent / volatile growth
    "MSTR",
    "MARA",
    "RIOT",
    "PLTR",
    "SHOP",
    "UBER",
    "ABNB",
    "RBLX",
    "SNAP",
)

# How many factors to fit per ticker. Keeping it at 20 keeps the cosine
# stable while not blowing up the wall-clock for a fresh /fit call.
TOP_FACTORS_N: int = 20

# Default top-20 factor slugs. These are well-populated, theme-spanning
# factors from the catalog (politics, macro, ai, crypto, sports, etc.) —
# stable enough to serve as a canonical "factor basis" without us having
# to read factors.yml on each request. Production wiring can pass a custom
# list to :func:`compute_related_stocks` if needed.
DEFAULT_FACTOR_SLUGS: tuple[str, ...] = (
    "fed-cut-2026q2",
    "fed-hike-2026q2",
    "ai-boom",
    "ai-regulation",
    "election-gop-2026",
    "election-dem-2026",
    "recession-2026",
    "oil-100-2026",
    "btc-100k-2026",
    "btc-200k-2026",
    "eth-5k-2026",
    "china-tariffs-2026",
    "ukraine-ceasefire",
    "middle-east-escalation",
    "inflation-cpi-above-3",
    "unemployment-above-5",
    "spy-above-6000",
    "vix-above-30",
    "earnings-beat-tech",
    "credit-spreads-widen",
)

# A peer's "shared_factors" list contains every factor where both the
# anchor and the peer have an absolute loading above this threshold.
# 0.15 is roughly "more than rounding-noise" for the synthetic-fixture
# generator below; production callers with real exposures should tune.
EXPOSURE_THRESHOLD: float = 0.15

# Top-N peers to return.
DEFAULT_TOP_PEERS: int = 10


# --- cache -------------------------------------------------------------------
# Process-wide cache shared across requests; 10-min TTL.
_RS_CACHE = get_cache("related_stocks", ttl=600)


def clear_cache() -> None:
    """Drop every cached related-stocks payload. Used by tests."""
    _RS_CACHE.clear()


# --- exposure hook (test-overrideable) ---------------------------------------

# A "exposure" is just dict[factor_slug, beta_coefficient]. Tests
# monkeypatch this module-level callable to return a synthetic mapping so
# the endpoint stays hermetic (no real /fit, no Polymarket, no yfinance).
ExposureFn = Callable[[str, tuple[str, ...]], dict[str, float]]


def _default_exposure_fn(ticker: str, factors: tuple[str, ...]) -> dict[str, float]:
    """Deterministic synthetic exposures keyed on ``(ticker, factor_slug)``.

    The default implementation hashes ``ticker || factor`` to produce a
    seeded float in ``[-1, 1]``. It's NOT a real regression — it's a
    placeholder so the endpoint returns sensible-looking data even when
    the production ``/fit`` pipeline isn't wired in. Callers that want
    real exposures should monkeypatch this module's ``_default_exposure_fn``
    or pass ``exposure_fn=`` explicitly to :func:`compute_related_stocks`.

    The hash-based generator is deterministic across processes, which is
    a useful property: the same ticker always produces the same exposure
    vector, so cosine similarities are repeatable across restarts.
    """
    out: dict[str, float] = {}
    for slug in factors:
        # SHA-1 of "<ticker>::<slug>" → first 8 bytes interpreted as
        # uint64, mapped to [-1, 1] via /MAX*2-1. Avoids the bias that
        # a naive ``hash(...) / MAX`` has on 64-bit signed ints.
        h = hashlib.sha1(f"{ticker.upper()}::{slug}".encode()).digest()
        val = int.from_bytes(h[:8], "big") / 0xFFFFFFFFFFFFFFFF
        out[slug] = 2.0 * val - 1.0
    return out


# --- math --------------------------------------------------------------------


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two exposure dicts.

    Treats missing keys in either dict as zero. Returns ``0.0`` if either
    vector has zero norm (avoids divide-by-zero); negative cosines are
    preserved (they represent inversely-loaded peers).
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k in keys:
        ai = float(a.get(k, 0.0))
        bi = float(b.get(k, 0.0))
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def shared_factors(
    a: dict[str, float],
    b: dict[str, float],
    threshold: float = EXPOSURE_THRESHOLD,
) -> list[str]:
    """Slugs where both ``a`` and ``b`` have ``|β| >= threshold``.

    Returns the intersection sorted by combined exposure magnitude
    (``|β_a| + |β_b|``) descending so the highest-conviction shared
    factors come first.
    """
    out: list[tuple[str, float]] = []
    for slug, va in a.items():
        if slug not in b:
            continue
        ma = abs(float(va))
        mb = abs(float(b[slug]))
        if ma >= threshold and mb >= threshold:
            out.append((slug, ma + mb))
    out.sort(key=lambda kv: kv[1], reverse=True)
    return [slug for slug, _ in out]


# --- core compute ------------------------------------------------------------


def compute_related_stocks(
    ticker: str,
    *,
    universe: tuple[str, ...] | None = None,
    factors: tuple[str, ...] | None = None,
    top_n: int = DEFAULT_TOP_PEERS,
    exposure_fn: ExposureFn | None = None,
    threshold: float = EXPOSURE_THRESHOLD,
) -> dict[str, Any]:
    """Compute the top-N factor-correlated peers for ``ticker``.

    Args:
        ticker: Anchor ticker (case-insensitive). Must be in ``universe``.
        universe: Candidate peer set (tuple of upper-case tickers).
        factors: Factor slugs to fit each ticker against. Length defines
            the dimensionality of the cosine.
        top_n: How many peers to return.
        exposure_fn: ``(ticker, factors) -> {slug: β}`` callable. When
            ``None``, uses the module-level :func:`_default_exposure_fn`
            (which tests can monkeypatch).
        threshold: Minimum ``|β|`` to count a factor as "shared" between
            anchor and peer.

    Returns:
        Dict with keys ``anchor`` and ``peers``. Each peer entry has
        ``ticker``, ``similarity`` (float in ``[-1, 1]``), and
        ``shared_factors`` (list of factor slugs).

    Raises:
        ValueError: if ``ticker`` is not in ``universe``.
    """
    # Resolve module-level defaults LAZILY so tests can monkeypatch the
    # ``UNIVERSE_TICKERS`` / ``DEFAULT_FACTOR_SLUGS`` module attributes and
    # have the new values reflected. Default-arg evaluation happens once
    # at def-time, which would otherwise capture the pre-patch tuples.
    universe_resolved: tuple[str, ...] = universe if universe is not None else UNIVERSE_TICKERS
    factors_resolved: tuple[str, ...] = factors if factors is not None else DEFAULT_FACTOR_SLUGS

    anchor = ticker.upper()
    universe_upper = tuple(t.upper() for t in universe_resolved)
    if anchor not in universe_upper:
        raise ValueError(f"unknown ticker: {ticker!r} not in universe of {len(universe_resolved)}")

    fn: ExposureFn = exposure_fn if exposure_fn is not None else _default_exposure_fn
    anchor_exp = fn(anchor, factors_resolved)

    scored: list[dict[str, Any]] = []
    for peer in universe_upper:
        if peer == anchor:
            continue
        peer_exp = fn(peer, factors_resolved)
        sim = cosine_similarity(anchor_exp, peer_exp)
        shared = shared_factors(anchor_exp, peer_exp, threshold=threshold)
        scored.append(
            {
                "ticker": peer,
                "similarity": round(float(sim), 4),
                "shared_factors": shared,
            }
        )

    # Sort desc by similarity; ties broken alphabetically by ticker for
    # determinism (test suite asserts on exact ordering).
    scored.sort(key=lambda d: (-d["similarity"], d["ticker"]))
    top = scored[:top_n]

    return {
        "anchor": anchor,
        "peers": top,
    }


# --- router ------------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-related-stocks"])


@router.get("/related-stocks/{ticker}", response_model=None)
def get_related_stocks(
    ticker: Annotated[str, FPath(min_length=1, max_length=12)],
) -> dict[str, Any]:
    """Return top-10 factor-correlated peer tickers for ``ticker``.

    The peer list is sorted descending by cosine similarity between
    factor-exposure vectors. ``shared_factors`` lists every factor where
    both anchor and peer have a notable loading (``|β| >= 0.15``).

    Returns a 404 if the ticker isn't in the supported universe. Use
    ``/terminal/related-stocks/_universe`` (TODO) for the full list if
    you need to validate client-side.
    """
    anchor = ticker.upper()
    cache_key = f"related::{anchor}"
    cached = _RS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        payload = compute_related_stocks(anchor)
    except ValueError as exc:
        # Unknown ticker — 404 per spec.
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    _RS_CACHE.set(cache_key, payload)
    return payload


__all__ = [
    "DEFAULT_FACTOR_SLUGS",
    "DEFAULT_TOP_PEERS",
    "EXPOSURE_THRESHOLD",
    "TOP_FACTORS_N",
    "UNIVERSE_TICKERS",
    "ExposureFn",
    "_default_exposure_fn",
    "clear_cache",
    "compute_related_stocks",
    "cosine_similarity",
    "get_related_stocks",
    "router",
    "shared_factors",
]
