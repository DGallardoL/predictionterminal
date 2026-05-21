"""Cross-venue archive comparator — Polymarket vs Kalshi for the same event.

For a small set of pre-mapped, already-resolved concepts we know the
canonical contract on each venue. Given a concept slug we fetch the
final-quote price series from both venues, align them on a common UTC
date index, and compute:

- ``spread_at_resolution``: |PM_price − Kalshi_price| on the last
  observed day before settle
- ``days_diverged``: number of days where |PM − Kalshi| > 0.05
- ``max_spread_observed``: max |PM − Kalshi| across the joint window
- ``pct_time_pm_higher``: fraction of joint days with PM > Kalshi

The five concepts are deliberately hard-coded for the demo — they are
all known historical events with stable URLs and tickers. New concepts
should be added by editing :data:`CROSS_VENUE_CONCEPTS` below; the
function takes the concept name as input and looks the mapping up.
"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from pfm.cache_utils import get_cache
from pfm.sources.kalshi import KalshiClient
from pfm.sources.kalshi import fetch_factor_history as kalshi_history
from pfm.sources.polymarket import (
    PolymarketClient,
)
from pfm.sources.polymarket import (
    fetch_factor_history as poly_history,
)

# ─── pre-mapped concepts ────────────────────────────────────────────────
# Each entry maps a concept slug to (polymarket_slug, kalshi_ticker,
# human-readable description). All five are settled events with known
# outcomes — kept as a fixture so the demo doesn't break when live
# contracts roll off.
CROSS_VENUE_CONCEPTS: dict[str, dict[str, str]] = {
    "presidential_election_2024": {
        "polymarket_slug": "presidential-election-winner-2024",
        "kalshi_ticker": "PRES-2024-DJT",
        "description": "2024 US presidential election outcome (Trump wins).",
        "resolved_outcome": "YES",
    },
    "recession_2024": {
        "polymarket_slug": "us-recession-in-2024",
        "kalshi_ticker": "RECESSION-2024-Y",
        "description": "US enters NBER-defined recession in 2024.",
        "resolved_outcome": "NO",
    },
    "fed_first_cut_2024": {
        "polymarket_slug": "fed-decision-in-september",
        "kalshi_ticker": "FEDDECISION-24SEP-C50",
        "description": "Fed cuts >=50bps at Sep-2024 FOMC.",
        "resolved_outcome": "YES",
    },
    "btc_70k_2024": {
        "polymarket_slug": "will-bitcoin-reach-70000-in-2024",
        "kalshi_ticker": "BTCMAX-24DEC31-B70000",
        "description": "BTC closes above $70,000 at any point in 2024.",
        "resolved_outcome": "YES",
    },
    "cpi_above_3_2024": {
        "polymarket_slug": "cpi-above-3-percent-in-2024",
        "kalshi_ticker": "CPIYOY-24DEC-T3.0",
        "description": "Headline CPI YoY above 3.0% for any 2024 release.",
        "resolved_outcome": "YES",
    },
}

CACHE_NS = "archive_kalshi"
CACHE_TTL_S = 3600

DIVERGENCE_THRESHOLD = 0.05  # absolute price gap that counts as "diverged"


# ─── public API ─────────────────────────────────────────────────────────


def cross_venue_resolved_pairs(
    concept: str,
    *,
    polymarket_client: PolymarketClient | None = None,
    kalshi_client: KalshiClient | None = None,
    polymarket_history: Any = poly_history,
    kalshi_history_fn: Any = kalshi_history,
) -> dict[str, Any]:
    """Compare same-event Polymarket vs Kalshi prices for ``concept``.

    Args:
        concept: Key from :data:`CROSS_VENUE_CONCEPTS`.
        polymarket_client: Optional :class:`PolymarketClient` to reuse;
            a default one is built when ``None``.
        kalshi_client: Optional :class:`KalshiClient` to reuse.
        polymarket_history: Function with the same signature as
            :func:`pfm.sources.polymarket.fetch_factor_history`. Test
            hook so callers can inject a mocked history fetcher.
        kalshi_history_fn: Same idea but for the Kalshi side.

    Returns:
        Dict with concept metadata and the four divergence metrics.
        ``error`` is set when one of the two venues lacks data.
    """
    if concept not in CROSS_VENUE_CONCEPTS:
        raise KeyError(f"unknown concept {concept!r}; choose one of {sorted(CROSS_VENUE_CONCEPTS)}")

    cache = get_cache(CACHE_NS, ttl=CACHE_TTL_S)
    hit = cache.get(("cross_venue", concept))
    if hit is not None:
        return hit

    mapping = CROSS_VENUE_CONCEPTS[concept]

    owns_poly = polymarket_client is None
    pc = polymarket_client or PolymarketClient(
        gamma_url="https://gamma-api.polymarket.com",
        clob_url="https://clob.polymarket.com",
    )
    owns_kalshi = kalshi_client is None
    kc = kalshi_client or KalshiClient(client=httpx.Client(timeout=15.0))

    try:
        try:
            pm_df = polymarket_history(pc, mapping["polymarket_slug"])
        except Exception as e:
            return _error_payload(concept, mapping, f"polymarket fetch failed: {e}")

        try:
            ks_df = kalshi_history_fn(kc, mapping["kalshi_ticker"])
        except Exception as e:
            return _error_payload(concept, mapping, f"kalshi fetch failed: {e}")

        result = _compute_divergence(concept, mapping, pm_df, ks_df)
        cache.set(("cross_venue", concept), result)
        return result
    finally:
        if owns_poly:
            pc.close()
        if owns_kalshi:
            kc.close()


def list_concepts() -> list[dict[str, str]]:
    """Return the catalog of concepts as a list of metadata dicts."""
    return [{"concept": k, **dict(v)} for k, v in CROSS_VENUE_CONCEPTS.items()]


# ─── helpers ────────────────────────────────────────────────────────────


def _compute_divergence(
    concept: str,
    mapping: dict[str, str],
    pm_df: pd.DataFrame,
    ks_df: pd.DataFrame,
) -> dict[str, Any]:
    if pm_df is None or pm_df.empty or ks_df is None or ks_df.empty:
        return _error_payload(concept, mapping, "one venue returned empty history")

    pm_series = _price_series(pm_df).rename("pm")
    ks_series = _price_series(ks_df).rename("ks")

    # Align on UTC normalized dates.
    pm_series.index = pd.to_datetime(pm_series.index, utc=True).normalize()
    ks_series.index = pd.to_datetime(ks_series.index, utc=True).normalize()
    joined = pd.concat([pm_series, ks_series], axis=1, join="inner").dropna()

    if joined.empty:
        return _error_payload(concept, mapping, "no overlapping days between venues")

    spread = (joined["pm"] - joined["ks"]).abs()
    spread_at_resolution = float(spread.iloc[-1])
    max_spread = float(spread.max())
    days_diverged = int((spread > DIVERGENCE_THRESHOLD).sum())
    pct_pm_higher = float((joined["pm"] > joined["ks"]).mean())

    first_day = joined.index[0]
    last_day = joined.index[-1]

    return {
        "concept": concept,
        "description": mapping.get("description"),
        "polymarket_slug": mapping.get("polymarket_slug"),
        "kalshi_ticker": mapping.get("kalshi_ticker"),
        "resolved_outcome": mapping.get("resolved_outcome"),
        "n_overlap_days": int(len(joined)),
        "first_overlap_day": first_day.strftime("%Y-%m-%d"),
        "last_overlap_day": last_day.strftime("%Y-%m-%d"),
        "spread_at_resolution": round(spread_at_resolution, 6),
        "max_spread_observed": round(max_spread, 6),
        "days_diverged": days_diverged,
        "divergence_threshold": DIVERGENCE_THRESHOLD,
        "pct_time_pm_higher": round(pct_pm_higher, 6),
        "error": None,
    }


def _error_payload(concept: str, mapping: dict[str, str], message: str) -> dict[str, Any]:
    return {
        "concept": concept,
        "description": mapping.get("description"),
        "polymarket_slug": mapping.get("polymarket_slug"),
        "kalshi_ticker": mapping.get("kalshi_ticker"),
        "resolved_outcome": mapping.get("resolved_outcome"),
        "n_overlap_days": 0,
        "first_overlap_day": None,
        "last_overlap_day": None,
        "spread_at_resolution": None,
        "max_spread_observed": None,
        "days_diverged": 0,
        "divergence_threshold": DIVERGENCE_THRESHOLD,
        "pct_time_pm_higher": None,
        "error": message,
    }


def _price_series(df: pd.DataFrame) -> pd.Series:
    """Pull a single price column out of either Polymarket or Kalshi
    history shape. Both expose a ``price`` column on the index."""
    if "price" in df.columns:
        return df["price"].astype(float)
    if "p" in df.columns:
        return df["p"].astype(float).rename("price")
    # Fallback: first numeric column.
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[c].astype(float).rename("price")
    raise ValueError("no price column found in history dataframe")


__all__ = [
    "CROSS_VENUE_CONCEPTS",
    "DIVERGENCE_THRESHOLD",
    "cross_venue_resolved_pairs",
    "list_concepts",
]
