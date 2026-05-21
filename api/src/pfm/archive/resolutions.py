"""Resolution-outcome lookup for resolved Polymarket markets.

A small wrapper that, given a slug, returns the canonical resolution
record:

    {
      "slug": str,
      "resolution": "YES" | "NO" | "AMBIGUOUS" | "PENDING",
      "resolution_date": ISO date | None,
      "resolution_source": str | None,
      "payout_per_share": float | None,
      "dispute_history": list[dict],
    }

We deliberately keep this separate from
:mod:`pfm.archive.polymarket_archive` because some callers (e.g. the
strategy backtester) only need the outcome, not the full price history.
Caching + TTL is shared via the ``archive_polymarket`` cache namespace.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from pfm.archive.polymarket_archive import (
    ARCHIVE_CACHE_TTL,
    _gamma_fetch_market,
    _resolution_label,
    _safe_float,
)
from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)


def _payout_per_share(market: dict[str, Any]) -> float | None:
    """Final per-share payout for the YES side.

    Polymarket pays $1 per resolved-true share, $0 per resolved-false. The
    YES holder's payout = the YES element of ``outcomePrices`` (which Gamma
    pins to 1.0 / 0.0 on resolution).
    """
    raw = market.get("outcomePrices")
    arr: list[Any] | None = None
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            arr = None
    elif isinstance(raw, list):
        arr = raw
    if arr is not None and arr:
        return _safe_float(arr[0])
    return None


def _dispute_history(market: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort dispute timeline.

    Gamma exposes a few overlapping fields depending on whether UMA was
    pinged: ``umaResolutionStatuses`` (string), ``disputes`` (list of
    dicts), and occasionally ``resolutionEvents`` (list of dicts). We
    flatten whichever is present into a uniform list.
    """
    out: list[dict[str, Any]] = []
    disputes = market.get("disputes")
    if isinstance(disputes, list):
        for d in disputes:
            if isinstance(d, dict):
                out.append(
                    {
                        "ts": d.get("timestamp") or d.get("ts") or "",
                        "kind": d.get("kind") or d.get("type") or "dispute",
                        "detail": d.get("detail") or d.get("reason") or "",
                    }
                )
    events = market.get("resolutionEvents")
    if isinstance(events, list):
        for e in events:
            if isinstance(e, dict):
                out.append(
                    {
                        "ts": e.get("timestamp") or e.get("ts") or "",
                        "kind": e.get("kind") or e.get("type") or "event",
                        "detail": e.get("detail") or "",
                    }
                )
    statuses = market.get("umaResolutionStatuses")
    if isinstance(statuses, str) and statuses.strip():
        out.append({"ts": "", "kind": "uma", "detail": statuses})
    return out


def get_resolution(
    slug: str,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Return the canonical resolution record for ``slug``.

    Raises:
        LookupError: if no Gamma market matches the slug.
    """
    cache = get_cache("archive_polymarket", ttl=ARCHIVE_CACHE_TTL)
    cache_key = ("resolution", slug)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    owns = client is None
    http = client or httpx.Client(timeout=15.0)
    try:
        market = _gamma_fetch_market(http, slug)
        if market is None:
            raise LookupError(f"no archive market for slug={slug!r}")

        resolution = _resolution_label(market)
        end_date = (market.get("endDate") or "")[:10] or None
        # Some Gamma markets expose a separate ``resolvedDate`` distinct from
        # ``endDate`` (UMA resolution can lag the trading-end timestamp).
        resolved_date = (market.get("resolvedDate") or "")[:10] or end_date
        source = (
            market.get("resolutionSource") or market.get("resolutionUrl") or market.get("oracle")
        )

        record = {
            "slug": slug,
            "resolution": resolution,
            "resolution_date": resolved_date,
            "resolution_source": (source if isinstance(source, str) and source.strip() else None),
            "payout_per_share": _payout_per_share(market),
            "dispute_history": _dispute_history(market),
        }
    finally:
        if owns:
            http.close()

    # Resolved markets are immutable, so the 1h TTL is just to refresh on
    # the rare retroactive UMA dispute. PENDING records re-fetch sooner
    # (they may flip to YES/NO/AMBIGUOUS within the day).
    ttl = 60 if resolution == "PENDING" else ARCHIVE_CACHE_TTL
    cache.set(cache_key, record, ttl=ttl)
    return record


__all__ = ["get_resolution"]
