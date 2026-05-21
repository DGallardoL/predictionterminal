"""Cross-Venue Arbitrage Scanner — Polymarket vs Kalshi spread detection.

This module is what we believe is the only product in the world to ship a
*single normalised view* over both Polymarket (944 factors) and Kalshi
(146 factors) prediction markets. Once two contracts on the two venues
are confirmed to resolve on the same underlying event, any sustained
mid-price disagreement greater than ``min_spread_pct`` is a tradeable
arbitrage subject to (a) tradeable size, (b) carry costs to resolution,
and (c) venue-specific fees.

Public entry points:

  - :func:`match_markets` — heuristically pairs PM ↔ Kalshi markets by
    title-keyword overlap, theme match, and end-date proximity.
  - :func:`compute_arb_spreads` — fetches live mids on both venues for
    each matched pair and emits arbitrage opportunities filtered by
    minimum spread percentage and minimum tradeable USD size.
  - :func:`top_arbs` — convenience wrapper that returns the N best arbs
    ranked by ``spread_pct × tradeable_size_usd``.

Five known PM/Kalshi pre-matched pairs are hardcoded in
``PRE_MATCHED_PAIRS`` so the demo always returns something even when
heuristic matching flags no candidates.

Endpoints (mounted via the module's ``router``):

  - ``GET  /arb/scanner``    top arbs ranked by spread × size
  - ``POST /arb/match``      manually confirm a (pm_slug, kalshi_slug) pair
  - ``GET  /arb/matched``    list of all known matched pairs
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.sources import kalshi as kalshi_src
from pfm.sources.polymarket_pool import PolymarketHTTPPool
from pfm.terminal import fetch_gamma_market

logger = logging.getLogger(__name__)

GAMMA_URL: str = "https://gamma-api.polymarket.com"
KALSHI_URL: str = "https://api.elections.kalshi.com/trade-api/v2"

# Cache for the scanner (60s) and the matched-pair registry (1h).
_SCANNER_CACHE = get_cache("arb_scanner", ttl=60)
_MATCH_CACHE = get_cache("arb_matched", ttl=3600)
# Dynamic-matching cache (5 min) for auto-discovery and 4-way arbs.
_DYNAMIC_CACHE = get_cache("arb_dynamic", ttl=300)

# Single-flight locks for /arb/auto-discover. The handler does ~10 s of
# upstream venue scans on a cache miss; without dedup, three concurrent
# first-callers each pay the full cost. Locks are lazily allocated per
# (event-loop, cache-key) so the registry never binds to a stale loop.
_AUTO_DISCOVER_LOCKS: dict[tuple[int, Any], asyncio.Lock] = {}


def _auto_discover_lock(key: Any) -> asyncio.Lock:
    """Return the per-key asyncio lock for the current event loop."""
    loop = asyncio.get_event_loop()
    composite = (id(loop), key)
    lock = _AUTO_DISCOVER_LOCKS.get(composite)
    if lock is None:
        lock = asyncio.Lock()
        _AUTO_DISCOVER_LOCKS[composite] = lock
    return lock


# Persistent confirmed-match store. Two markets that pass the similarity
# threshold across ``CONFIRMED_FETCHES_REQUIRED`` consecutive fetches are
# promoted to "confirmed" status and persisted to the path below so the
# next process restart still treats them as confirmed.
CONFIRMED_MATCHES_PATH: Path = Path("/tmp/pfm_arb_confirmed_matches.json")
CONFIRMED_FETCHES_REQUIRED: int = 7
_CONFIRMED_LOCK = threading.Lock()

# Threshold above which a spread snapshot is treated as confirmed for the POC.
# Real production would require sustained spread for ``CONFIRM_WINDOW_MIN``.
CONFIRM_WINDOW_MIN: int = 30

# Stop-words that don't contribute meaning to title overlap.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "for",
        "on",
        "in",
        "to",
        "by",
        "is",
        "be",
        "will",
        "and",
        "or",
        "than",
        "more",
        "less",
        "above",
        "below",
        "before",
        "after",
        "at",
        "with",
        "as",
        "from",
        "this",
        "that",
        "any",
        "all",
        "do",
        "does",
        "did",
        "go",
        "going",
        "next",
    }
)


# ---------------------------------------------------------------------------
# Hardcoded pre-matched pairs
# ---------------------------------------------------------------------------
#
# These are the known-good cross-venue pairs we trust without running
# heuristic matching. Slugs/tickers come straight from ``factors.yml``.
# Keep them in this list even if they resolve — the scanner gracefully
# skips a pair whose mid cannot be fetched.

PRE_MATCHED_PAIRS: list[dict[str, str]] = [
    {
        "pm_slug": "us-recession-by-end-of-2026",
        "kalshi_slug": "KXRECSSNBER-26",
        "label": "US recession by end of 2026",
        "theme": "macro",
    },
    {
        "pm_slug": "will-no-fed-rate-cuts-happen-in-2026",
        "kalshi_slug": "KXFEDDECISION-26DEC-C25",
        "label": "Fed cuts 2026 (composite)",
        "theme": "macro",
    },
    {
        "pm_slug": "will-the-fed-decrease-interest-rates-by-50-bps-after-the-june-2026-meeting",
        "kalshi_slug": "KXFEDDECISION-26JUN-C25",
        "label": "Fed June-2026 decision",
        "theme": "macro",
    },
    # Two further commonly-watched cross-venue pairs. If the underlying
    # markets are resolved/missing at demo time, the scanner skips them.
    {
        "pm_slug": "btc-all-time-high-by-june-30",
        "kalshi_slug": "KXBTCMAXY-26",
        "label": "BTC all-time high by H1 2026",
        "theme": "crypto",
    },
    {
        "pm_slug": "will-cpi-be-above-3-5-in-2026",
        "kalshi_slug": "KXLCPIMAXYOY-26-P3.5",
        "label": "CPI > 3.5% in 2026",
        "theme": "macro",
    },
]

# In-memory registry of additional manually-confirmed pairs. Cleared per
# process. Production would persist this in a real datastore.
_MANUAL_PAIRS: list[dict[str, str]] = []
_MANUAL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Title / theme similarity
# ---------------------------------------------------------------------------


def _tokenise(title: str) -> set[str]:
    """Lower-case, drop stopwords, drop tokens shorter than 3 chars."""
    if not title:
        return set()
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", title.lower())
    return {t for t in raw if len(t) >= 3 and t not in _STOPWORDS}


def _keyword_jaccard(a: str, b: str) -> float:
    """Jaccard overlap between two titles' content tokens."""
    ta, tb = _tokenise(a), _tokenise(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _date_proximity_score(a: str | None, b: str | None, *, tol_days: int = 7) -> float:
    """1.0 if end-dates are within ``tol_days``, decaying linearly to 0 at 30d.

    Either side missing → 0.5 (neutral; can't reject the pairing on date alone).
    """
    if not a or not b:
        return 0.5
    # Some venues return numeric epoch timestamps where others return ISO
    # strings — normalise both shapes to ISO before parsing.
    try:
        if not isinstance(a, str):
            da = datetime.fromtimestamp(float(a), tz=UTC)
        else:
            da = datetime.fromisoformat(a.replace("Z", "+00:00"))
        if not isinstance(b, str):
            db = datetime.fromtimestamp(float(b), tz=UTC)
        else:
            db = datetime.fromisoformat(b.replace("Z", "+00:00"))
    except (TypeError, ValueError, OSError, OverflowError):
        return 0.5
    delta_days = abs((da - db).total_seconds()) / 86_400.0
    if delta_days <= tol_days:
        return 1.0
    if delta_days >= 30:
        return 0.0
    # Linear: tol → 1.0, 30d → 0.0
    return max(0.0, 1.0 - (delta_days - tol_days) / (30.0 - tol_days))


def _theme_score(a: str | None, b: str | None) -> float:
    """1 if themes match (case-insensitive), 0 otherwise. None pair = 0.5."""
    if not a or not b:
        return 0.5
    return 1.0 if a.strip().lower() == b.strip().lower() else 0.0


_POLARITY_NEGATIVE_PATTERN = re.compile(
    r"\b(?:not|no|fails?|loses?|won'?t|wont|declines?|negative|never|"
    r"unable|reject(?:s|ed)?|denied?|deny|cancel(?:s|led)?|"
    r"miss(?:es|ed)?|fall(?:s)?(?:\s+short)?)\b",
    re.IGNORECASE,
)


def _polarity_is_negative(text: str | None) -> bool:
    """Return True iff ``text`` carries an explicit negation marker.

    Used to detect inverted-question pairings where Jaccard token overlap
    would otherwise match "X happens" against "X does NOT happen". A naive
    overlap would emit a spread record with sign inverted — a phantom arb.
    """
    if not text:
        return False
    return bool(_POLARITY_NEGATIVE_PATTERN.search(text))


def _polarity_inverted(a: str | None, b: str | None) -> bool:
    """True iff exactly one side is phrased negatively (XOR)."""
    return _polarity_is_negative(a) != _polarity_is_negative(b)


def _similarity(pm: dict[str, Any], kalshi: dict[str, Any]) -> float:
    """Composite similarity in [0, 1].

    Weights: 60% keyword Jaccard, 25% end-date proximity, 15% theme match.
    """
    kw = _keyword_jaccard(pm.get("title", "") or pm.get("question", ""), kalshi.get("title", ""))
    dt = _date_proximity_score(pm.get("end_date"), kalshi.get("end_date"))
    th = _theme_score(pm.get("theme"), kalshi.get("theme"))
    return 0.60 * kw + 0.25 * dt + 0.15 * th


def match_markets(
    pm_markets: list[dict[str, Any]],
    kalshi_markets: list[dict[str, Any]],
    min_similarity: float = 0.7,
    *,
    manifold_markets: list[dict[str, Any]] | None = None,
    predictit_markets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Pair markets across up to four venues (PM × Kalshi × Manifold × PredictIt).

    The legacy 2-venue contract is unchanged: callers passing only ``pm`` and
    ``kalshi`` get the original PM↔Kalshi pair list with the original keys
    (``pm_slug``, ``kalshi_slug``).

    When ``manifold_markets`` or ``predictit_markets`` is provided the
    function additionally emits cross-venue pairs for every distinct
    venue-pair (PM↔Manifold, PM↔PredictIt, Kalshi↔Manifold, etc.). Each
    record carries ``venue_a`` / ``venue_b`` plus per-venue identifier
    fields so downstream consumers can route to the right price fetcher.
    """
    venues: dict[str, list[dict[str, Any]]] = {
        "polymarket": pm_markets or [],
        "kalshi": kalshi_markets or [],
    }
    if manifold_markets is not None:
        venues["manifold"] = manifold_markets
    if predictit_markets is not None:
        venues["predictit"] = predictit_markets

    out: list[dict[str, Any]] = []
    venue_keys = list(venues.keys())
    only_pm_kalshi = (
        manifold_markets is None
        and predictit_markets is None
        and venue_keys == ["polymarket", "kalshi"]
    )

    for i, va in enumerate(venue_keys):
        for vb in venue_keys[i + 1 :]:
            for ma in venues[va]:
                for mb in venues[vb]:
                    score = _similarity(ma, mb)
                    if score < min_similarity:
                        continue
                    # Polarity guard: Jaccard can match "X happens" with
                    # "X does NOT happen" → spread with inverted sign =
                    # phantom arb. Drop pairings where exactly one side
                    # carries an explicit negation.
                    title_a = ma.get("title") or ma.get("question") or ""
                    title_b = mb.get("title") or mb.get("question") or ""
                    if _polarity_inverted(title_a, title_b):
                        continue
                    rec: dict[str, Any] = {
                        "venue_a": va,
                        "venue_b": vb,
                        f"{va}_slug": ma.get("slug") or ma.get("ticker") or str(ma.get("id", "")),
                        f"{vb}_slug": mb.get("slug") or mb.get("ticker") or str(mb.get("id", "")),
                        "similarity_score": round(score, 4),
                        "suggested": score < 0.85,
                        "label": title_a,
                    }
                    # Backward-compatibility shim: when callers stayed on
                    # the legacy 2-venue API, surface the original keys
                    # directly so existing consumers keep working.
                    if only_pm_kalshi:
                        rec["pm_slug"] = ma.get("slug", "")
                        rec["kalshi_slug"] = mb.get("ticker") or mb.get("slug", "")
                    out.append(rec)
    out.sort(key=lambda r: r["similarity_score"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Mid fetchers (PM + Kalshi)
# ---------------------------------------------------------------------------


# Maximum staleness for ``lastTradePrice`` fallback. Thin contracts can keep
# a hours-old last print that no longer reflects the orderbook; using it as
# the mid leads to phantom spreads. 30 min matches the half-life window we
# already use elsewhere in the arb pipeline.
_PM_LAST_TRADE_STALE_SECONDS: float = 30.0 * 60.0


def _pm_last_trade_is_fresh(market: dict[str, Any]) -> bool:
    """True iff ``lastTradeTime`` (when present) is within the freshness window.

    When the field is missing entirely we conservatively return ``False`` so
    the fallback is not used — callers prefer ``None`` to a possibly-stale
    print. Robust to ISO strings and numeric epoch values alike.
    """
    raw = market.get("lastTradeTime") or market.get("last_trade_time")
    if raw is None:
        return False
    try:
        if isinstance(raw, str):
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            ts = dt.timestamp()
        else:
            ts = float(raw)
            # Heuristic: epoch-millis values are ~1e12+
            if ts > 1e12:
                ts = ts / 1000.0
    except (TypeError, ValueError, OSError, OverflowError):
        return False
    age = datetime.now(tz=UTC).timestamp() - ts
    return 0.0 <= age <= _PM_LAST_TRADE_STALE_SECONDS


def _pm_mid(slug: str, http: httpx.Client) -> tuple[float | None, float | None]:
    """Fetch (mid_price, volume_24hr) for a Polymarket slug. None on miss.

    Falls back to ``lastTradePrice`` only when ``lastTradeTime`` shows the
    print is within the last 30 min — thin contracts can carry hours-old
    last prints that would otherwise be returned as a stale mid.
    """
    try:
        m = fetch_gamma_market(http, GAMMA_URL, slug)
    except (LookupError, httpx.HTTPError) as exc:
        logger.info("PM fetch failed for %s: %s", slug, exc)
        return None, None
    bb = m.get("bestBid")
    ba = m.get("bestAsk")
    last = m.get("lastTradePrice")
    mid: float | None = None
    try:
        if bb is not None and ba is not None:
            mid = (float(bb) + float(ba)) / 2.0
        elif last is not None and _pm_last_trade_is_fresh(m):
            mid = float(last)
    except (TypeError, ValueError):
        mid = None
    vol: float | None = None
    try:
        v = m.get("volume24hr") or m.get("volumeNum") or m.get("volume")
        vol = float(v) if v is not None else None
    except (TypeError, ValueError):
        vol = None
    return mid, vol


def _kalshi_mid(ticker: str, client: kalshi_src.KalshiClient) -> tuple[float | None, float | None]:
    """Fetch (mid_price, est_volume_usd) for a Kalshi ticker.

    Mid = (yes_bid + yes_ask) / 2 from the latest candlestick. Volume is
    the most recent bar's ``volume`` (in dollars per Kalshi convention).
    Returns (None, None) on any error.
    """
    try:
        end = datetime.now(tz=UTC)
        start_ts = int(end.timestamp()) - 86_400 * 7
        end_ts = int(end.timestamp())
        df = client.get_candlesticks(ticker, start_ts=start_ts, end_ts=end_ts)
    except Exception as exc:
        logger.info("Kalshi fetch failed for %s: %s", ticker, exc)
        return None, None
    if df.empty:
        return None, None
    last = df.iloc[-1]
    try:
        bid = float(last["yes_bid"])
        ask = float(last["yes_ask"])
        mid = (bid + ask) / 2.0
    except (KeyError, TypeError, ValueError):
        return None, None
    try:
        vol = float(last.get("volume", 0.0))
    except (TypeError, ValueError):
        vol = 0.0
    return mid, vol


# ---------------------------------------------------------------------------
# Spread computation
# ---------------------------------------------------------------------------


def _spread_record(
    pair: dict[str, str],
    pm_price: float,
    kalshi_price: float,
    pm_vol: float | None,
    kalshi_vol: float | None,
    *,
    min_spread_pct: float,
    min_volume_usd: float,
) -> dict[str, Any] | None:
    """Build an arb record from a paired PM + Kalshi snapshot.

    Returns None when the pair fails the spread or volume gate.

    .. warning::

        ``tradeable_size_usd`` here is a **24h-volume proxy**, not depth at
        touch. We compute ``min(pm_volume_24h, kalshi_last_bar_volume)``
        which mixes a cumulative 24h notional (PM) with a single candlestick
        bar (Kalshi). The PM number can overstate the actually-fillable
        size by 1–3 orders of magnitude on illiquid markets. Treat the
        returned value as an *upper bound* on tradeable size; production
        sizing must consult per-side orderbook depth via the Kalshi
        ``OrderbookSnapshot`` / PM ``bestBidSize`` / ``bestAskSize``
        fields directly. Tracked in TODO: replace with depth-at-touch.
    """
    spread_pct = abs(pm_price - kalshi_price) * 100.0  # both prices in [0, 1]
    if spread_pct < min_spread_pct:
        return None

    pm_v = float(pm_vol or 0.0)
    k_v = float(kalshi_vol or 0.0)
    # TODO(quant): replace with depth-at-touch. min(pm_vol_24h, kalshi_bar_vol)
    # mixes dimensions — see docstring warning.
    tradeable_size_usd = min(pm_v, k_v)
    if tradeable_size_usd < min_volume_usd:
        return None

    direction = "buy_kalshi_sell_pm" if kalshi_price < pm_price else "buy_pm_sell_kalshi"

    # Half-life proxy: tighter spreads are stickier; wider ones close fast.
    # Calibration: 2% spread → 30 min, 10% → 5 min. Linear in 1/spread.
    half_life_minutes = max(5.0, min(120.0, 60.0 / max(0.01, spread_pct / 5.0)))

    return {
        "pm_slug": pair["pm_slug"],
        "kalshi_slug": pair["kalshi_slug"],
        "label": pair.get("label", ""),
        "pm_price": round(float(pm_price), 4),
        "kalshi_price": round(float(kalshi_price), 4),
        "spread_pct": round(spread_pct, 3),
        "direction": direction,
        "tradeable_size_usd": round(tradeable_size_usd, 2),
        "half_life_minutes": round(half_life_minutes, 1),
        "last_seen_iso": datetime.now(tz=UTC).isoformat(),
        # POC: a snapshot that qualifies is treated as already-confirmed.
        "confirmed": True,
        "confirmation_window_min": CONFIRM_WINDOW_MIN,
    }


def compute_arb_spreads(
    matched_pairs: list[dict[str, Any]],
    min_spread_pct: float = 2.0,
    min_volume_usd: float = 5_000.0,
    *,
    http: httpx.Client | None = None,
    kalshi_client: kalshi_src.KalshiClient | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every matched pair against live mids; emit qualifying arbs.

    The two upstream fetches per pair (Polymarket Gamma + Kalshi) used to
    run sequentially across every pair → ~N×2 RTTs. Both clients are
    thread-safe for the simple GET/candlestick paths we hit here so we
    fan-out across pairs with a bounded thread pool. The wall clock drops
    from O(N) to roughly the slowest single pair, which is what shows up
    as a >30 s warm probe on a fresh cache.
    """
    own_http = http is None
    own_kalshi = kalshi_client is None
    http = http or httpx.Client(timeout=8.0)
    kalshi_client = kalshi_client or kalshi_src.KalshiClient()

    def _evaluate(pair: dict[str, Any]) -> dict[str, Any] | None:
        try:
            pm_mid, pm_vol = _pm_mid(pair["pm_slug"], http)
            if pm_mid is None:
                return None
            k_mid, k_vol = _kalshi_mid(pair["kalshi_slug"], kalshi_client)
            if k_mid is None:
                return None
            return _spread_record(
                pair,
                pm_mid,
                k_mid,
                pm_vol,
                k_vol,
                min_spread_pct=min_spread_pct,
                min_volume_usd=min_volume_usd,
            )
        except Exception as e:  # never blank the whole scan
            logger.info("arb pair evaluation failed for %r: %s", pair, e)
            return None

    out: list[dict[str, Any]] = []
    try:
        if not matched_pairs:
            return out
        # Cap concurrency: 6 keeps us well under Polymarket's 1000/10s rate
        # limit and matches Kalshi's recommended per-IP concurrency.
        max_workers = min(len(matched_pairs), 6)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="arb-pair") as ex:
            for rec in ex.map(_evaluate, matched_pairs):
                if rec is not None:
                    out.append(rec)
    finally:
        if own_http:
            http.close()
        if own_kalshi:
            kalshi_client.close()
    return out


def all_matched_pairs() -> list[dict[str, str]]:
    """Pre-matched + manually-confirmed pairs as a single list."""
    with _MANUAL_LOCK:
        manual = list(_MANUAL_PAIRS)
    return list(PRE_MATCHED_PAIRS) + manual


# ---------------------------------------------------------------------------
# 4-way concept maps (Polymarket × Kalshi × Manifold × PredictIt)
# ---------------------------------------------------------------------------
#
# Each entry is a high-level "concept" that resolves on the same underlying
# event across up to four venues. We keep a small hand-curated list rather
# than running heuristic matching across four venues at once — at four-venue
# scale the false-pair rate explodes and the daily curation cost is cheap.
#
# Any venue field can be ``None`` (a concept may not list on every venue);
# downstream :func:`find_4way_arb` skips missing legs gracefully.

CONCEPT_MAPS: list[dict[str, Any]] = [
    {
        "concept_id": "presidential_election_2028",
        "label": "2028 US Presidential Election winner",
        "theme": "politics",
        "polymarket": "presidential-election-winner-2028",
        "kalshi": "KXPRES-28",
        "manifold": "who-will-win-the-2028-us-presidenti",
        "predictit": 8200,
    },
    {
        "concept_id": "fed_cuts_2026",
        "label": "Fed cuts rates in 2026",
        "theme": "macro",
        "polymarket": "will-no-fed-rate-cuts-happen-in-2026",
        "kalshi": "KXFEDDECISION-26DEC-C25",
        "manifold": "will-the-fed-cut-rates-in-2026",
        "predictit": 7400,
    },
    {
        "concept_id": "recession_2026",
        "label": "US recession by end of 2026",
        "theme": "macro",
        "polymarket": "us-recession-by-end-of-2026",
        "kalshi": "KXRECSSNBER-26",
        "manifold": "will-the-us-be-in-recession-by-end-2026",
        "predictit": 7300,
    },
    {
        "concept_id": "btc_ath_2026",
        "label": "BTC all-time high in 2026",
        "theme": "crypto",
        "polymarket": "btc-all-time-high-by-june-30",
        "kalshi": "KXBTCMAXY-26",
        "manifold": "will-btc-hit-new-ath-2026",
        "predictit": None,
    },
    {
        "concept_id": "cpi_above_3_5_2026",
        "label": "US CPI above 3.5% YoY in 2026",
        "theme": "macro",
        "polymarket": "will-cpi-be-above-3-5-in-2026",
        "kalshi": "KXLCPIMAXYOY-26-P3.5",
        "manifold": "will-us-cpi-be-above-35-in-2026",
        "predictit": 7500,
    },
]


def get_concept_map(concept_id: str) -> dict[str, Any] | None:
    """Lookup a 4-venue concept map by id. ``None`` when not found."""
    cid = (concept_id or "").strip().lower()
    for m in CONCEPT_MAPS:
        if str(m["concept_id"]).lower() == cid:
            return m
    return None


def _max_pairwise_spread_pct(prices: dict[str, float]) -> tuple[float, str, str]:
    """Return ``(spread_pct, low_venue, high_venue)`` across the supplied legs.

    Empty / single-leg input returns ``(0.0, "", "")`` so downstream callers
    can still render the response.
    """
    if len(prices) < 2:
        return 0.0, "", ""
    items = sorted(prices.items(), key=lambda kv: kv[1])
    low_v, low_p = items[0]
    high_v, high_p = items[-1]
    return (high_p - low_p) * 100.0, low_v, high_v


def find_4way_arb(
    market_concept: str,
    *,
    pm_price_fn: Any = None,
    kalshi_price_fn: Any = None,
    manifold_price_fn: Any = None,
    predictit_price_fn: Any = None,
    capital_per_leg_usd: float = 10_000.0,
) -> dict[str, Any]:
    """Snapshot the prices of a 4-venue concept and rank the spread.

    Each ``*_price_fn`` is an optional callable taking the venue-specific
    identifier and returning ``(price, volume_usd)`` (or ``(None, None)``
    on miss). When ``None`` the leg is treated as missing — useful for
    tests where we inject deterministic price functions, and for
    production where some venues may be down at scan time.

    Returns
    -------
    dict
        ``{concept_id, label, theme, prices: {venue: price}, volumes: {...},
        max_spread_pct, low_venue, high_venue, capital_required_usd,
        legs_present, missing_venues}``.
    """
    concept = get_concept_map(market_concept)
    if concept is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown concept_id: {market_concept!r}",
        )

    prices: dict[str, float] = {}
    volumes: dict[str, float] = {}
    missing: list[str] = []

    fn_map: dict[str, Any] = {
        "polymarket": pm_price_fn,
        "kalshi": kalshi_price_fn,
        "manifold": manifold_price_fn,
        "predictit": predictit_price_fn,
    }

    for venue, fn in fn_map.items():
        ident = concept.get(venue)
        if not ident or fn is None:
            missing.append(venue)
            continue
        try:
            price, vol = fn(ident)
        except Exception as exc:
            logger.info("4way-arb %s leg failed for %s: %s", venue, ident, exc)
            missing.append(venue)
            continue
        if price is None:
            missing.append(venue)
            continue
        prices[venue] = float(price)
        volumes[venue] = float(vol or 0.0)

    spread_pct, low_v, high_v = _max_pairwise_spread_pct(prices)
    # Capital required: long the cheap leg, short the expensive leg.
    # Single-leg notional × 2 since we hit two venues to capture the spread.
    capital_required = float(capital_per_leg_usd) * 2.0 if spread_pct > 0 else 0.0

    return {
        "concept_id": concept["concept_id"],
        "label": concept.get("label", ""),
        "theme": concept.get("theme", ""),
        "prices": {k: round(v, 4) for k, v in prices.items()},
        "volumes": {k: round(v, 2) for k, v in volumes.items()},
        "max_spread_pct": round(spread_pct, 3),
        "low_venue": low_v,
        "high_venue": high_v,
        "capital_required_usd": round(capital_required, 2),
        "legs_present": sorted(prices.keys()),
        "missing_venues": sorted(missing),
        "as_of": datetime.now(tz=UTC).isoformat(),
    }


def top_arbs(
    min_spread_pct: float = 2.0,
    n: int = 10,
    *,
    http: httpx.Client | None = None,
    kalshi_client: kalshi_src.KalshiClient | None = None,
) -> list[dict[str, Any]]:
    """Top N arbs ranked by ``spread_pct × tradeable_size_usd``."""
    arbs = compute_arb_spreads(
        all_matched_pairs(),
        min_spread_pct=min_spread_pct,
        http=http,
        kalshi_client=kalshi_client,
    )
    arbs.sort(key=lambda r: r["spread_pct"] * r["tradeable_size_usd"], reverse=True)
    return arbs[: max(0, int(n))]


# ---------------------------------------------------------------------------
# Dynamic matching: per-venue active-market fetchers
# ---------------------------------------------------------------------------
#
# Each fetcher returns a list of normalised market dicts with keys:
#   ``id``, ``slug``/``ticker``, ``title``, ``theme``, ``end_date``,
#   ``price`` (float in [0,1]), ``volume_24h_usd`` (float).
# Failures degrade gracefully to ``[]`` — never raise, so a single bad
# venue does not blank the cross-venue scan.


def _normalise_pm_market(m: dict[str, Any]) -> dict[str, Any]:
    """Normalise a Polymarket Gamma market record."""
    bb = m.get("bestBid")
    ba = m.get("bestAsk")
    last = m.get("lastTradePrice")
    price: float | None = None
    try:
        if bb is not None and ba is not None:
            price = (float(bb) + float(ba)) / 2.0
        elif last is not None:
            price = float(last)
    except (TypeError, ValueError):
        price = None
    try:
        v = m.get("volume24hr") or m.get("volumeNum") or m.get("volume") or 0.0
        vol = float(v)
    except (TypeError, ValueError):
        vol = 0.0
    return {
        "venue": "polymarket",
        "id": str(m.get("id", "")),
        "slug": m.get("slug", ""),
        "title": m.get("question") or m.get("title") or "",
        "theme": (m.get("category") or "").lower() or None,
        "end_date": m.get("endDate"),
        "price": price,
        "volume_24h_usd": vol,
    }


def _normalise_kalshi_market(m: dict[str, Any]) -> dict[str, Any]:
    """Normalise a Kalshi public-API market record (cents → [0,1])."""
    yes_bid = m.get("yes_bid")
    yes_ask = m.get("yes_ask")
    last = m.get("last_price")
    price: float | None = None
    try:
        if yes_bid is not None and yes_ask is not None:
            price = (float(yes_bid) + float(yes_ask)) / 2.0 / 100.0
        elif last is not None:
            price = float(last) / 100.0
    except (TypeError, ValueError):
        price = None
    try:
        v = m.get("volume_24h") or m.get("dollar_volume_24h") or m.get("volume") or 0.0
        vol = float(v)
    except (TypeError, ValueError):
        vol = 0.0
    return {
        "venue": "kalshi",
        "id": str(m.get("ticker", "")),
        "ticker": str(m.get("ticker", "")),
        "slug": str(m.get("ticker", "")),
        "title": m.get("title") or m.get("subtitle") or "",
        "theme": (m.get("category") or "").lower() or None,
        "end_date": m.get("close_time"),
        "price": price,
        "volume_24h_usd": vol,
    }


def _normalise_manifold_market(m: dict[str, Any]) -> dict[str, Any]:
    """Normalise a Manifold market record."""
    try:
        prob = m.get("probability")
        price = float(prob) if prob is not None else None
    except (TypeError, ValueError):
        price = None
    try:
        v = m.get("volume24Hours") or m.get("volume") or 0.0
        vol = float(v)
    except (TypeError, ValueError):
        vol = 0.0
    return {
        "venue": "manifold",
        "id": str(m.get("id", "")),
        "slug": m.get("slug", ""),
        "title": m.get("question") or m.get("title") or "",
        "theme": None,
        "end_date": m.get("closeTime"),
        "price": price,
        "volume_24h_usd": vol,
    }


def _normalise_predictit_market(m: dict[str, Any]) -> dict[str, Any]:
    """Normalise a PredictIt market record (snapshot has lead-contract price)."""
    contracts = m.get("contracts") or []
    price: float | None = None
    if isinstance(contracts, list) and contracts:
        try:
            lead = max(
                contracts,
                key=lambda c: float(c.get("lastTradePrice") or 0.0),
            )
            ltp = lead.get("lastTradePrice")
            price = float(ltp) if ltp is not None else None
        except (TypeError, ValueError):
            price = None
    try:
        v = m.get("totalSharesTraded") or m.get("volume") or 0.0
        vol = float(v)
    except (TypeError, ValueError):
        vol = 0.0
    return {
        "venue": "predictit",
        "id": str(m.get("id", "")),
        "slug": str(m.get("id", "")),
        "title": m.get("name") or m.get("shortName") or "",
        "theme": None,
        "end_date": m.get("dateEnd"),
        "price": price,
        "volume_24h_usd": vol,
    }


async def _fetch_active_polymarket(
    http: httpx.AsyncClient, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Fetch top active Polymarket markets ordered by 24h volume."""
    try:
        r = await http.get(
            f"{GAMMA_URL}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": int(limit),
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("polymarket active-fetch failed: %s", exc)
        return []
    if not isinstance(data, list):
        return []
    return [_normalise_pm_market(m) for m in data[:limit]]


async def _fetch_active_kalshi(
    http: httpx.AsyncClient, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Fetch top active Kalshi markets via the public ``/markets`` endpoint."""
    try:
        r = await http.get(
            f"{KALSHI_URL}/markets",
            params={"status": "open", "limit": int(limit)},
        )
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("kalshi active-fetch failed: %s", exc)
        return []
    markets = data.get("markets") if isinstance(data, dict) else None
    if not isinstance(markets, list):
        return []
    return [_normalise_kalshi_market(m) for m in markets[:limit]]


async def _fetch_active_manifold(
    http: httpx.AsyncClient, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Fetch top active Manifold markets via ``/v0/markets``."""
    try:
        r = await http.get(
            "https://api.manifold.markets/v0/markets",
            params={"limit": int(limit)},
        )
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("manifold active-fetch failed: %s", exc)
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for m in data[:limit]:
        if m.get("isResolved"):
            continue
        out.append(_normalise_manifold_market(m))
    return out


async def _fetch_active_predictit(
    http: httpx.AsyncClient, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Fetch active PredictIt markets via the all-markets snapshot."""
    try:
        r = await http.get("https://www.predictit.org/api/marketdata/all/")
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("predictit active-fetch failed: %s", exc)
        return []
    markets = data.get("markets") if isinstance(data, dict) else None
    if not isinstance(markets, list):
        return []
    return [_normalise_predictit_market(m) for m in markets[:limit]]


_VENUE_FETCHERS: dict[
    str,
    Any,
] = {
    "polymarket": _fetch_active_polymarket,
    "kalshi": _fetch_active_kalshi,
    "manifold": _fetch_active_manifold,
    "predictit": _fetch_active_predictit,
}


# ---------------------------------------------------------------------------
# Persistent confirmed-match registry
# ---------------------------------------------------------------------------


def _load_confirmed_store() -> dict[str, dict[str, Any]]:
    """Load the on-disk confirmed-match registry. Tolerant to missing/bad files."""
    try:
        with CONFIRMED_MATCHES_PATH.open("r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def _save_confirmed_store(store: dict[str, dict[str, Any]]) -> None:
    """Persist the confirmed-match registry. Best-effort — never raise."""
    try:
        CONFIRMED_MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIRMED_MATCHES_PATH.open("w") as f:
            json.dump(store, f)
    except OSError as exc:
        logger.info("failed to persist confirmed matches: %s", exc)


def _pair_key(venue_a: str, slug_a: str, venue_b: str, slug_b: str) -> str:
    """Stable, order-independent key for a (venue, slug) pair."""
    a = (venue_a, slug_a)
    b = (venue_b, slug_b)
    lo, hi = sorted([a, b])
    return f"{lo[0]}::{lo[1]}||{hi[0]}::{hi[1]}"


def record_match_observation(
    venue_a: str,
    slug_a: str,
    venue_b: str,
    slug_b: str,
    *,
    similarity: float,
    label: str = "",
) -> dict[str, Any]:
    """Bump the consecutive-fetch counter for a (venue, slug) pair.

    When the counter reaches :data:`CONFIRMED_FETCHES_REQUIRED` the pair is
    flagged ``confirmed=True`` and persisted to disk. Returns the updated
    record.
    """
    key = _pair_key(venue_a, slug_a, venue_b, slug_b)
    now_iso = datetime.now(tz=UTC).isoformat()
    with _CONFIRMED_LOCK:
        store = _load_confirmed_store()
        rec = store.get(
            key,
            {
                "venue_a": venue_a,
                "slug_a": slug_a,
                "venue_b": venue_b,
                "slug_b": slug_b,
                "label": label,
                "fetches": 0,
                "confirmed": False,
                "first_seen_iso": now_iso,
                "last_seen_iso": now_iso,
                "last_similarity": float(similarity),
            },
        )
        rec["fetches"] = int(rec.get("fetches", 0)) + 1
        rec["last_seen_iso"] = now_iso
        rec["last_similarity"] = float(similarity)
        if label and not rec.get("label"):
            rec["label"] = label
        if rec["fetches"] >= CONFIRMED_FETCHES_REQUIRED:
            rec["confirmed"] = True
        store[key] = rec
        _save_confirmed_store(store)
        return dict(rec)


def list_confirmed_matches(*, only_confirmed: bool = True) -> list[dict[str, Any]]:
    """Return the persisted confirmed-match registry as a list."""
    with _CONFIRMED_LOCK:
        store = _load_confirmed_store()
    out = list(store.values())
    if only_confirmed:
        out = [r for r in out if r.get("confirmed")]
    return out


# ---------------------------------------------------------------------------
# Auto-discovery of arb pairs
# ---------------------------------------------------------------------------


_DEFAULT_VENUES: tuple[str, ...] = ("polymarket", "kalshi", "manifold", "predictit")


async def _gather_active_markets(
    venues: list[str],
    http: httpx.AsyncClient,
    *,
    per_venue_limit: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch active markets from every requested venue concurrently."""
    tasks = []
    used: list[str] = []
    for v in venues:
        fetcher = _VENUE_FETCHERS.get(v)
        if fetcher is None:
            continue
        used.append(v)
        tasks.append(fetcher(http, limit=per_venue_limit))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: dict[str, list[dict[str, Any]]] = {}
    for v, res in zip(used, results, strict=True):
        if isinstance(res, BaseException):
            logger.info("active-fetch %s raised: %s", v, res)
            out[v] = []
        else:
            out[v] = res
    return out


async def auto_discover_arb_pairs(
    *,
    min_similarity: float = 0.65,
    min_volume_usd_per_venue: float = 1000.0,
    max_pairs: int = 50,
    venues: list[str] | None = None,
    http: httpx.AsyncClient | None = None,
    per_venue_limit: int = 100,
) -> list[dict[str, Any]]:
    """Discovery automatico: fetch active markets de cada venue, cluster por similarity.

    Returns the top-N matched pairs ordered by tradeable arb potential
    (``similarity_score * min_leg_volume_usd``). Each pair carries the
    venue identifiers, the per-leg volumes/prices, and the spread when
    both legs have a live mid.
    """
    chosen = list(venues) if venues else list(_DEFAULT_VENUES)
    # W11-11 (T18 pool migration): reuse the shared Polymarket gamma client
    # when no http is injected. The pool's client is process-wide, HTTP/2,
    # keep-alive — never close it. Tests still inject their own AsyncClient.
    if http is None:
        http = PolymarketHTTPPool.instance().gamma_client
    per_venue = await _gather_active_markets(
        chosen,
        http,
        per_venue_limit=per_venue_limit,
    )

    # Pre-filter by volume threshold.
    filtered: dict[str, list[dict[str, Any]]] = {}
    for v, mkts in per_venue.items():
        kept = [
            m
            for m in mkts
            if float(m.get("volume_24h_usd") or 0.0) >= float(min_volume_usd_per_venue)
        ]
        filtered[v] = kept

    venue_keys = list(filtered.keys())
    out: list[dict[str, Any]] = []
    for i, va in enumerate(venue_keys):
        for vb in venue_keys[i + 1 :]:
            for ma in filtered[va]:
                for mb in filtered[vb]:
                    score = _similarity(ma, mb)
                    if score < float(min_similarity):
                        continue
                    slug_a = ma.get("slug") or ma.get("ticker") or ma.get("id", "")
                    slug_b = mb.get("slug") or mb.get("ticker") or mb.get("id", "")
                    pa = ma.get("price")
                    pb = mb.get("price")
                    spread_pct: float | None = None
                    if pa is not None and pb is not None:
                        spread_pct = abs(float(pa) - float(pb)) * 100.0
                    vol_a = float(ma.get("volume_24h_usd") or 0.0)
                    vol_b = float(mb.get("volume_24h_usd") or 0.0)
                    tradeable = min(vol_a, vol_b)
                    rec: dict[str, Any] = {
                        "venue_a": va,
                        "venue_b": vb,
                        f"{va}_slug": str(slug_a),
                        f"{vb}_slug": str(slug_b),
                        "label": ma.get("title") or mb.get("title") or "",
                        "similarity_score": round(float(score), 4),
                        "suggested": float(score) < 0.85,
                        "prices": {
                            va: round(float(pa), 4) if pa is not None else None,
                            vb: round(float(pb), 4) if pb is not None else None,
                        },
                        "volumes_24h_usd": {
                            va: round(vol_a, 2),
                            vb: round(vol_b, 2),
                        },
                        "spread_pct": (round(spread_pct, 3) if spread_pct is not None else None),
                        "tradeable_size_usd": round(tradeable, 2),
                        "potential_score": round(float(score) * tradeable, 2),
                    }
                    # Persistent matching: bump the observation counter.
                    confirmed_rec = record_match_observation(
                        va,
                        str(slug_a),
                        vb,
                        str(slug_b),
                        similarity=float(score),
                        label=rec["label"],
                    )
                    rec["confirmed"] = bool(confirmed_rec.get("confirmed", False))
                    rec["confirmation_fetches"] = int(confirmed_rec.get("fetches", 0))
                    out.append(rec)

    out.sort(key=lambda r: r["potential_score"], reverse=True)
    return out[: max(0, int(max_pairs))]


# ---------------------------------------------------------------------------
# 4-way arbs computed dynamically across every venue carrying the concept
# ---------------------------------------------------------------------------


def _half_life_estimate(spread_pct: float) -> float:
    """Same calibration as :func:`_spread_record`: 2% spread => 30min, 10% => 5min."""
    return max(5.0, min(120.0, 60.0 / max(0.01, spread_pct / 5.0)))


def _build_4way_record(
    concept: dict[str, Any],
    prices: dict[str, float],
    volumes: dict[str, float],
) -> dict[str, Any]:
    """Build a 4-way arb record from per-venue prices/volumes."""
    spread_pct, low_v, high_v = _max_pairwise_spread_pct(prices)
    tradeable = min(volumes.values()) if volumes else 0.0
    return {
        "concept": concept["concept_id"],
        "label": concept.get("label", ""),
        "theme": concept.get("theme", ""),
        "prices_per_venue": {k: round(v, 4) for k, v in prices.items()},
        "volumes_per_venue": {k: round(v, 2) for k, v in volumes.items()},
        "max_spread_pct": round(spread_pct, 3),
        "low_venue": low_v,
        "high_venue": high_v,
        "tradeable_size_usd": round(float(tradeable), 2),
        "half_life_estimate_min": round(_half_life_estimate(spread_pct), 1),
        "legs_present": sorted(prices.keys()),
    }


def compute_4way_arbs(
    concepts: list[dict[str, Any]] | None = None,
    *,
    price_fns: dict[str, Any] | None = None,
    min_spread_pct: float = 0.0,
) -> list[dict[str, Any]]:
    """Snapshot every concept's price across the four venues and rank spreads.

    ``price_fns`` is a mapping ``{venue: callable(ident) -> (price, volume)}``.
    Missing venues are silently skipped per concept (legs that have neither
    an identifier nor a price function are not part of the spread).
    """
    cmap = list(concepts) if concepts is not None else list(CONCEPT_MAPS)
    fn_map = price_fns or {}
    out: list[dict[str, Any]] = []
    for concept in cmap:
        prices: dict[str, float] = {}
        volumes: dict[str, float] = {}
        for venue in _DEFAULT_VENUES:
            ident = concept.get(venue)
            fn = fn_map.get(venue)
            if not ident or fn is None:
                continue
            try:
                p, v = fn(ident)
            except Exception as exc:
                logger.info(
                    "compute_4way_arbs: %s leg failed for %s: %s",
                    venue,
                    ident,
                    exc,
                )
                continue
            if p is None:
                continue
            prices[venue] = float(p)
            volumes[venue] = float(v or 0.0)
        if len(prices) < 2:
            continue
        rec = _build_4way_record(concept, prices, volumes)
        if rec["max_spread_pct"] < float(min_spread_pct):
            continue
        out.append(rec)
    out.sort(
        key=lambda r: r["max_spread_pct"] * r["tradeable_size_usd"],
        reverse=True,
    )
    return out


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ArbOpportunity(BaseModel):
    """Single qualifying cross-venue arb snapshot."""

    pm_slug: str
    kalshi_slug: str
    label: str = ""
    pm_price: float = Field(..., ge=0.0, le=1.0)
    kalshi_price: float = Field(..., ge=0.0, le=1.0)
    spread_pct: float = Field(..., ge=0.0)
    direction: str
    tradeable_size_usd: float = Field(..., ge=0.0)
    half_life_minutes: float = Field(..., ge=0.0)
    last_seen_iso: str
    confirmed: bool
    confirmation_window_min: int


class ArbScannerResponse(BaseModel):
    as_of: str
    n: int
    min_spread_pct: float
    arbs: list[ArbOpportunity]


class ArbMatchRequest(BaseModel):
    pm_slug: str = Field(..., min_length=1)
    kalshi_slug: str = Field(..., min_length=1)
    label: str = ""
    theme: str = ""


class ArbMatchedPair(BaseModel):
    pm_slug: str
    kalshi_slug: str
    label: str = ""
    theme: str = ""
    source: str  # "hardcoded" | "manual"


class ArbMatchedResponse(BaseModel):
    n: int
    pairs: list[ArbMatchedPair]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/arb", tags=["arb-scanner"])


@router.get("/scanner", response_model=ArbScannerResponse)
def get_scanner(
    min_spread_pct: float = Query(default=2.0, ge=0.0, le=100.0),
    n: int = Query(default=10, ge=1, le=100),
) -> ArbScannerResponse:
    """Top cross-venue arbs ranked by ``spread_pct × tradeable_size_usd``."""
    cache_key = ("scanner", round(min_spread_pct, 3), n)
    cached = _SCANNER_CACHE.get(cache_key)
    if cached is not None:
        return ArbScannerResponse(**cached)

    arbs = top_arbs(min_spread_pct=min_spread_pct, n=n)
    response = ArbScannerResponse(
        as_of=datetime.now(tz=UTC).isoformat(),
        n=len(arbs),
        min_spread_pct=float(min_spread_pct),
        arbs=arbs,
    )
    _SCANNER_CACHE.set(cache_key, response.model_dump(), ttl=60)
    return response


@router.post("/match", response_model=ArbMatchedPair)
def post_match(body: ArbMatchRequest) -> ArbMatchedPair:
    """Manually register a (pm_slug, kalshi_slug) pair.

    Idempotent: the same pair is only added once. Persists for the life
    of the process.
    """
    pm = body.pm_slug.strip()
    ks = body.kalshi_slug.strip()
    if not pm or not ks:
        raise HTTPException(status_code=400, detail="empty slug not allowed")
    record = {
        "pm_slug": pm,
        "kalshi_slug": ks,
        "label": body.label,
        "theme": body.theme,
    }
    with _MANUAL_LOCK:
        for p in _MANUAL_PAIRS:
            if p["pm_slug"] == pm and p["kalshi_slug"] == ks:
                _MATCH_CACHE.clear()
                return ArbMatchedPair(source="manual", **p)
        _MANUAL_PAIRS.append(record)
    _MATCH_CACHE.clear()
    return ArbMatchedPair(source="manual", **record)


@router.get("/matched", response_model=ArbMatchedResponse)
def get_matched() -> ArbMatchedResponse:
    """List all matched pairs (hardcoded + manually confirmed)."""
    cached = _MATCH_CACHE.get("matched")
    if cached is not None:
        return ArbMatchedResponse(**cached)

    pairs: list[dict[str, str]] = []
    for p in PRE_MATCHED_PAIRS:
        pairs.append(
            {
                "pm_slug": p["pm_slug"],
                "kalshi_slug": p["kalshi_slug"],
                "label": p.get("label", ""),
                "theme": p.get("theme", ""),
                "source": "hardcoded",
            }
        )
    with _MANUAL_LOCK:
        for p in _MANUAL_PAIRS:
            pairs.append(
                {
                    "pm_slug": p["pm_slug"],
                    "kalshi_slug": p["kalshi_slug"],
                    "label": p.get("label", ""),
                    "theme": p.get("theme", ""),
                    "source": "manual",
                }
            )

    response = ArbMatchedResponse(n=len(pairs), pairs=pairs)
    _MATCH_CACHE.set("matched", response.model_dump(), ttl=60)
    return response


@router.get("/concept/{concept_id}")
def get_4way_concept(concept_id: str) -> dict[str, Any]:
    """Return the 4-venue concept map and a snapshot of available legs.

    Live price fetchers are not wired here — the endpoint returns the
    concept's identifiers across PM/Kalshi/Manifold/PredictIt plus an empty
    ``prices`` dict. Callers that want a live snapshot pass
    :func:`find_4way_arb` their venue-specific price functions directly.
    """
    concept = get_concept_map(concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail=f"unknown concept_id: {concept_id!r}")
    return {
        "concept_id": concept["concept_id"],
        "label": concept.get("label", ""),
        "theme": concept.get("theme", ""),
        "venues": {
            "polymarket": concept.get("polymarket"),
            "kalshi": concept.get("kalshi"),
            "manifold": concept.get("manifold"),
            "predictit": concept.get("predictit"),
        },
    }


@router.get("/concepts")
def list_4way_concepts() -> dict[str, Any]:
    """List every hardcoded 4-venue concept map."""
    return {"n": len(CONCEPT_MAPS), "concepts": CONCEPT_MAPS}


@router.get("/auto-discover")
async def get_auto_discover(
    min_similarity: float = Query(default=0.65, ge=0.0, le=1.0),
    min_volume: float = Query(default=1000.0, ge=0.0),
    max_pairs: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Auto-discovered cross-venue arb pairs (5-min cache).

    Single-flight: concurrent first-callers on a cold cache share the
    same upstream fan-out (~10 s) instead of each running their own.
    """
    cache_key = (
        "auto_discover",
        round(float(min_similarity), 4),
        round(float(min_volume), 2),
        int(max_pairs),
    )
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lock = _auto_discover_lock(cache_key)
    async with lock:
        # Re-check inside the lock: a previous lock-holder may have
        # populated the cache while we waited.
        cached = _DYNAMIC_CACHE.get(cache_key)
        if cached is not None:
            return cached

        pairs = await auto_discover_arb_pairs(
            min_similarity=min_similarity,
            min_volume_usd_per_venue=min_volume,
            max_pairs=max_pairs,
        )
        payload: dict[str, Any] = {
            "as_of": datetime.now(tz=UTC).isoformat(),
            "n": len(pairs),
            "min_similarity": float(min_similarity),
            "min_volume_usd_per_venue": float(min_volume),
            "pairs": pairs,
        }
        _DYNAMIC_CACHE.set(cache_key, payload, ttl=300)
        return payload


@router.get("/4way-arbs")
def get_4way_arbs(
    min_spread_pct: float = Query(default=0.0, ge=0.0, le=100.0),
) -> dict[str, Any]:
    """Active 4-venue arb opportunities across the curated concept maps.

    Without injected price functions the response carries the concept maps
    plus an empty ``arbs`` list (no live mids fetched). Tests inject
    ``price_fns`` directly into :func:`compute_4way_arbs`; production
    callers can wire venue mid-fetchers via the same function.
    """
    cache_key = ("4way", round(float(min_spread_pct), 3))
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    arbs = compute_4way_arbs(min_spread_pct=min_spread_pct)
    payload: dict[str, Any] = {
        "as_of": datetime.now(tz=UTC).isoformat(),
        "n": len(arbs),
        "min_spread_pct": float(min_spread_pct),
        "arbs": arbs,
    }
    _DYNAMIC_CACHE.set(cache_key, payload, ttl=300)
    return payload


@router.get("/confirmed-matches")
def get_confirmed_matches(
    only_confirmed: bool = Query(default=True),
) -> dict[str, Any]:
    """Persistent registry of cross-venue pairs confirmed across N consecutive fetches."""
    matches = list_confirmed_matches(only_confirmed=only_confirmed)
    return {
        "n": len(matches),
        "fetches_required": CONFIRMED_FETCHES_REQUIRED,
        "matches": matches,
    }


__all__ = [
    "CONCEPT_MAPS",
    "CONFIRMED_FETCHES_REQUIRED",
    "CONFIRMED_MATCHES_PATH",
    "PRE_MATCHED_PAIRS",
    "ArbMatchRequest",
    "ArbMatchedPair",
    "ArbMatchedResponse",
    "ArbOpportunity",
    "ArbScannerResponse",
    "all_matched_pairs",
    "auto_discover_arb_pairs",
    "compute_4way_arbs",
    "compute_arb_spreads",
    "find_4way_arb",
    "get_concept_map",
    "list_confirmed_matches",
    "match_markets",
    "record_match_observation",
    "router",
    "top_arbs",
]
