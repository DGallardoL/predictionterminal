"""Terminal trending-news router.

Endpoint
--------

    GET /terminal/news/trending?limit=20&hours=24

Aggregates the last ``hours`` worth of news from the existing 4-source
pipeline (GDELT + Reddit + Hacker News + curated RSS wires), deduplicates
near-identical headlines via :mod:`pfm.terminal.news_dedupe`'s SimHash
clustering (re-used from task T20), and scores each cluster on three
axes:

* **Recency** — fresher headlines dominate (``1 / hours_since``).
* **Cross-source corroboration** — a story carried by GDELT *and*
  Reddit *and* HN is far more trending than one only on RSS
  (``log(1 + n_sources)``).
* **Sentiment intensity** — strongly-positive or strongly-negative
  headlines outrank neutral filler (``1 + |compound|``).

The composite score is the product of the three factors and a higher
value is more trending. The endpoint returns the top-``limit``
clusters sorted by score descending.

Design notes
------------

* This module **never imports from `main.py`** — it owns its own
  :class:`fastapi.APIRouter` and is wired up by callers via
  ``app.include_router(...)``. Per CLAUDE.md this lets us extend the
  Terminal mode without touching the hot ``main.py`` file.
* The fetch layer is intentionally simple: it calls into the existing
  modules' public/private fetch helpers and is best-effort — if any
  upstream is down the score still works on the surviving sources
  (the n_sources factor naturally penalises single-source items).
* All upstream HTTP calls are wrapped in try/except so a failure in
  one source never kills the endpoint. The unit tests verify this by
  raising from every source pipe.
* A small TTL cache keeps the response warm for 60 s — this endpoint
  is meant to drive a UI ticker that polls every minute.

Routing note
------------

To activate the endpoint::

    from pfm.terminal.news_trending_router import router as terminal_news_trending_router
    app.include_router(terminal_news_trending_router)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from pfm.terminal.news_dedupe import NewsItem, dedupe_news

logger = logging.getLogger(__name__)

# Window of news to consider — the default. Bounded at the query layer.
DEFAULT_LOOKBACK_HOURS: int = 24
MAX_LOOKBACK_HOURS: int = 168  # one week, hard cap
DEFAULT_LIMIT: int = 20
MAX_LIMIT: int = 100

# SimHash threshold reused from news_dedupe defaults.
SIMHASH_THRESHOLD_BITS: int = 4

# Tiny floor on hours_since so a brand-new article (≤1 min old) doesn't
# blow up the recency factor. 1 min = 0.0167 h.
MIN_HOURS_SINCE: float = 1.0 / 60.0

# Response cache TTL — short on purpose because the UI ticker polls.
CACHE_TTL_SECONDS: int = 60


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TrendingItem(BaseModel):
    """One trending news cluster."""

    title: str = Field(..., description="Representative headline for the cluster.")
    url: str = Field(..., description="Canonical URL — first source to publish.")
    n_sources: int = Field(..., ge=1, description="Number of distinct sources carrying the story.")
    first_seen: str = Field(
        ..., description="ISO-8601 UTC timestamp of the earliest article in the cluster."
    )
    sentiment: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Signed compound sentiment of the cluster representative title.",
    )
    score: float = Field(
        ..., ge=0.0, description="Composite trending score (higher = more trending)."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Distinct source identifiers (e.g. ``gdelt``, ``reddit``, ``hn``).",
    )
    hours_since: float = Field(
        ..., ge=0.0, description="Hours since the earliest article in the cluster."
    )


class TrendingResponse(BaseModel):
    checked_at: str = Field(..., description="ISO-8601 UTC timestamp of this response.")
    lookback_hours: int = Field(..., ge=1, le=MAX_LOOKBACK_HOURS)
    n_clusters: int = Field(..., ge=0, description="Total clusters before truncation.")
    trending: list[TrendingItem]


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------
#
# Each ``fetch_<source>`` returns ``list[NewsItem]`` with the per-source tag
# populated on ``NewsItem.source``. They are best-effort: any exception is
# caught and logged, returning an empty list so one bad upstream cannot
# poison the aggregate. The tests monkey-patch these directly.


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


def _safe_parse_ts(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 (or ISO with Z) timestamp; return ``None`` on failure."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def fetch_gdelt_items(lookback_hours: int) -> list[NewsItem]:
    """Pull recent global headlines from GDELT 2.0 'breaking'.

    Re-uses the helpers exposed by :mod:`pfm.terminal.gdelt_news` rather
    than re-implementing the HTTP plumbing. Best-effort — any error in
    the upstream module yields an empty list.
    """
    try:
        # Lazy import: keep this module light at import time and let tests
        # patch the module path without triggering the heavy chain.
        from pfm.terminal import gdelt_news as _gn

        client = _build_http_client()
        articles = _gn._fetch_gdelt(  # type: ignore[attr-defined]
            client, "sourcelang:english", 50, timespan=f"{lookback_hours}h"
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("trending: gdelt fetch failed: %s", exc)
        return []

    out: list[NewsItem] = []
    cutoff = _now_utc() - timedelta(hours=lookback_hours)
    for a in articles or []:
        ts = _safe_parse_ts(getattr(a, "ts", None))
        if ts is None or ts < cutoff:
            continue
        title = (getattr(a, "title", "") or "").strip()
        url = (getattr(a, "url", "") or "").strip()
        if not title or not url:
            continue
        tone = getattr(a, "tone", None)
        try:
            tone_f = float(tone) if tone is not None else None
        except (TypeError, ValueError):
            tone_f = None
        out.append(
            NewsItem(
                title=title,
                url=url,
                source="gdelt",
                published_at=ts,
                tone=tone_f,
            )
        )
    return out


def fetch_rss_items(lookback_hours: int) -> list[NewsItem]:
    """Pull recent headlines from the curated RSS wire pool."""
    try:
        from pfm.terminal import rss_news as _rss

        client = _build_http_client()
        items: list[NewsItem] = []
        cutoff = _now_utc() - timedelta(hours=lookback_hours)
        for src in getattr(_rss, "SOURCES", []) or []:
            try:
                headlines, _err = _rss._fetch_source(client, src)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                logger.info("trending: rss source %r failed: %s", src, exc)
                continue
            for h in headlines or []:
                ts = _safe_parse_ts(getattr(h, "pub_date", None))
                if ts is None or ts < cutoff:
                    continue
                title = (getattr(h, "title", "") or "").strip()
                url = (getattr(h, "link", "") or "").strip()
                if not title or not url:
                    continue
                tone = None
                raw_score = getattr(h, "sentiment_score", None)
                if raw_score is not None:
                    try:
                        tone = float(raw_score) * 10.0  # rescale to ~GDELT range
                    except (TypeError, ValueError):
                        tone = None
                items.append(
                    NewsItem(
                        title=title,
                        url=url,
                        source="rss",
                        published_at=ts,
                        tone=tone,
                    )
                )
        return items
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("trending: rss fetch failed: %s", exc)
        return []


def fetch_reddit_items(lookback_hours: int) -> list[NewsItem]:
    """Pull recent Reddit posts from the existing ``terminal/news.py`` helpers."""
    try:
        from pfm.terminal import news as _news

        client = _build_http_client()
        items, _ok = _news._fetch_reddit(  # type: ignore[attr-defined]
            client, "news", 50
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("trending: reddit fetch failed: %s", exc)
        return []

    out: list[NewsItem] = []
    cutoff = _now_utc() - timedelta(hours=lookback_hours)
    for it in items or []:
        ts = _safe_parse_ts(getattr(it, "ts", None))
        if ts is None or ts < cutoff:
            continue
        title = (getattr(it, "title", "") or "").strip()
        url = (getattr(it, "url", "") or "").strip()
        if not title or not url:
            continue
        out.append(
            NewsItem(
                title=title,
                url=url,
                source="reddit",
                published_at=ts,
                tone=None,
            )
        )
    return out


def fetch_hn_items(lookback_hours: int) -> list[NewsItem]:
    """Pull recent Hacker News stories from the existing helpers."""
    try:
        from pfm.terminal import news as _news

        client = _build_http_client()
        items, _ok = _news._fetch_hn(  # type: ignore[attr-defined]
            client, "news", 50
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("trending: hn fetch failed: %s", exc)
        return []

    out: list[NewsItem] = []
    cutoff = _now_utc() - timedelta(hours=lookback_hours)
    for it in items or []:
        ts = _safe_parse_ts(getattr(it, "ts", None))
        if ts is None or ts < cutoff:
            continue
        title = (getattr(it, "title", "") or "").strip()
        url = (getattr(it, "url", "") or "").strip()
        if not title or not url:
            continue
        out.append(
            NewsItem(
                title=title,
                url=url,
                source="hn",
                published_at=ts,
                tone=None,
            )
        )
    return out


def _build_http_client():
    """Build a short-lived httpx client. Kept inside a helper for monkey-patching."""
    import httpx

    return httpx.Client(timeout=10.0)


# Aggregator wiring — exposed at module level so tests can patch the list.
SOURCE_FETCHERS: dict[str, Callable[[int], list[NewsItem]]] = {
    "gdelt": fetch_gdelt_items,
    "reddit": fetch_reddit_items,
    "hn": fetch_hn_items,
    "rss": fetch_rss_items,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _sentiment_compound(title: str, tone: float | None) -> float:
    """Return signed compound sentiment for ``title`` in ``[-1, +1]``.

    Falls back to a no-op zero on any error so a broken scorer cannot
    bring the whole endpoint down. When ``tone`` is provided it is
    used as the GDELT external signal (the scorer rescales 10:1).
    """
    try:
        from pfm.terminal.sentiment_nlp import score_headline as _score

        compound, _ = _score(title, external_tone=tone if tone is not None else None)
        return float(compound)
    except Exception:  # pragma: no cover - defensive
        return 0.0


def compute_score(*, hours_since: float, n_sources: int, compound: float) -> float:
    """Composite trending score.

    ``score = (1 / hours_since) * log(1 + n_sources) * (1 + |compound|)``

    ``hours_since`` is clamped to a tiny positive floor so a freshly-
    published article doesn't divide by zero. ``n_sources`` is clamped
    to ``>= 1`` (every cluster has at least one source by construction).
    ``compound`` is clipped to ``[-1, +1]`` defensively.
    """
    h = max(MIN_HOURS_SINCE, float(hours_since))
    n = max(1, int(n_sources))
    c = max(-1.0, min(1.0, float(compound)))
    return (1.0 / h) * math.log(1.0 + n) * (1.0 + abs(c))


def rank_trending(
    items: list[NewsItem],
    *,
    now: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
    threshold_bits: int = SIMHASH_THRESHOLD_BITS,
) -> tuple[list[TrendingItem], int]:
    """Dedupe + score + rank a flat list of ``NewsItem``.

    Returns ``(top_k, total_clusters)`` where ``total_clusters`` is the
    pre-truncation cluster count. The cluster representative is the
    earliest article (matches ``dedupe_news`` semantics).
    """
    if not items:
        return [], 0
    clusters = dedupe_news(items, threshold_bits=threshold_bits)
    now = now or _now_utc()

    ranked: list[TrendingItem] = []
    for c in clusters:
        # ``dedupe_news`` populates ``sources`` with all merged source tags
        # and falls back to the single ``source`` when no merge happened.
        srcs = list(c.sources) if c.sources else [c.source]
        first_seen = c.published_at
        if first_seen.tzinfo is None:
            first_seen = first_seen.replace(tzinfo=UTC)
        hours_since = max(0.0, (now - first_seen).total_seconds() / 3600.0)
        compound = _sentiment_compound(c.title, c.tone)
        score = compute_score(
            hours_since=hours_since,
            n_sources=len(set(srcs)),
            compound=compound,
        )
        ranked.append(
            TrendingItem(
                title=c.title,
                url=c.url,
                n_sources=len(set(srcs)),
                first_seen=first_seen.isoformat(),
                sentiment=round(compound, 4),
                score=round(score, 4),
                sources=sorted(set(srcs)),
                hours_since=round(hours_since, 4),
            )
        )

    ranked.sort(key=lambda r: -r.score)
    total = len(ranked)
    return ranked[: max(1, int(limit))], total


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    expires_at: float
    payload: dict[str, Any]


@dataclass
class _ResponseCache:
    """Tiny single-key TTL cache. Thread-safe."""

    ttl_seconds: int = CACHE_TTL_SECONDS
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _store: dict[str, _CacheEntry] = field(default_factory=dict)

    def get(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < now:
                self._store.pop(key, None)
                return None
            return entry.payload

    def set(self, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = _CacheEntry(
                expires_at=time.time() + self.ttl_seconds,
                payload=payload,
            )

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


RESPONSE_CACHE = _ResponseCache()


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def build_trending(
    *,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    limit: int = DEFAULT_LIMIT,
) -> TrendingResponse:
    """End-to-end aggregator. Used by both the endpoint and the tests."""
    lookback_hours = max(1, min(MAX_LOOKBACK_HOURS, int(lookback_hours)))
    limit = max(1, min(MAX_LIMIT, int(limit)))

    all_items: list[NewsItem] = []
    for tag, fetcher in SOURCE_FETCHERS.items():
        try:
            fetched = fetcher(lookback_hours)
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("trending: source %s failed: %s", tag, exc)
            fetched = []
        if fetched:
            all_items.extend(fetched)

    ranked, total = rank_trending(all_items, limit=limit)
    return TrendingResponse(
        checked_at=_now_utc().isoformat(),
        lookback_hours=lookback_hours,
        n_clusters=total,
        trending=ranked,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/terminal/news", tags=["terminal"])


@router.get(
    "/trending",
    response_model=TrendingResponse,
    summary="Cross-source trending headlines ranked by recency, corroboration, and sentiment.",
)
def get_news_trending(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    hours: Annotated[
        int, Query(ge=1, le=MAX_LOOKBACK_HOURS, description="Lookback window in hours.")
    ] = DEFAULT_LOOKBACK_HOURS,
) -> TrendingResponse:
    """Return the top trending news clusters across our 4-source pipeline.

    Score = ``(1 / hours_since) * log(1 + n_sources) * (1 + |sentiment_compound|)``

    Cross-source corroboration (a story appearing on multiple feeds within
    the same time window) is the strongest signal — that is what
    distinguishes "actually trending" from "one source's filler".
    """
    cache_key = f"trending:{hours}:{limit}"
    cached = RESPONSE_CACHE.get(cache_key)
    if cached is not None:
        return TrendingResponse(**cached)

    response = build_trending(lookback_hours=hours, limit=limit)
    RESPONSE_CACHE.set(cache_key, response.model_dump())
    return response


__all__ = [
    "CACHE_TTL_SECONDS",
    "DEFAULT_LIMIT",
    "DEFAULT_LOOKBACK_HOURS",
    "MAX_LIMIT",
    "MAX_LOOKBACK_HOURS",
    "RESPONSE_CACHE",
    "SIMHASH_THRESHOLD_BITS",
    "SOURCE_FETCHERS",
    "TrendingItem",
    "TrendingResponse",
    "build_trending",
    "compute_score",
    "fetch_gdelt_items",
    "fetch_hn_items",
    "fetch_reddit_items",
    "fetch_rss_items",
    "rank_trending",
    "router",
]
