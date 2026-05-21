"""Terminal GDELT 2.0 global news feed.

GDELT 2.0 (https://api.gdeltproject.org/api/v2/doc/doc) is a free, no-auth,
real-time global news monitoring API updated every 15 minutes. It ingests
articles in 65+ languages from worldwide sources and is an excellent feed
for macro / political prediction-market context.

Two endpoints are exposed:

    GET /terminal/gdelt/{slug}?limit=20
    GET /terminal/gdelt/breaking?limit=10

The per-slug endpoint resolves the Polymarket question for ``slug``, extracts
2-3 keywords (reusing :func:`pfm.terminal_news.extract_keywords`), builds a
GDELT query string, fetches up to 20 articles, and aggregates per-source /
per-country counts plus a mean tone (sentiment in [-10, +10]).

The /breaking endpoint returns the top headlines globally over the last 6
hours, sorted by GDELT's ``hybridrel`` (recency + relevance).

Routing note: this module owns its :class:`fastapi.APIRouter`; ``main.py``
is left untouched (per CLAUDE.md). To activate the endpoint::

    from pfm.terminal_gdelt_news import router as terminal_gdelt_router
    app.include_router(terminal_gdelt_router)
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections import Counter
from typing import Annotated
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal.news_relevance import (
    RELEVANCE_MIN,
    QuestionTerms,
    build_phrase_query,
    build_terms,
    score_relevance,
)
from pfm.terminal_news import MAX_KEYWORDS, extract_keywords

logger = logging.getLogger(__name__)

GDELT_DOC_URL: str = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT: str = "polymarket-terminal/1.0 (gdelt-news)"
CACHE_TTL_SECONDS: int = 600  # 10 minutes — matches other terminal_* modules
DEFAULT_MAX_RECORDS: int = 20
HARD_CAP_RECORDS: int = 250  # GDELT-side hard limit
BREAKING_TIMESPAN: str = "6h"

# Redis SETNX single-flight lock: when a slug/breaking key is cold, only the
# first concurrent caller actually hits GDELT (~8.6 s upstream). Subsequent
# callers within ``_LOCK_TTL_SECONDS`` wait briefly and re-read the cache.
_LOCK_TTL_SECONDS: int = 30
_LOCK_WAIT_SECONDS: float = 8.5  # slightly above the 8.6 s observed cold latency
_LOCK_POLL_INTERVAL: float = 0.25
# Cross-worker L2 cache. The L1 dict is per-process so a single hot worker
# can't share its result; promoting through Redis lets a cold worker do an
# ~5 ms Redis read instead of an ~8.3 s GDELT fetch.
_REDIS_TTL_SECONDS: int = CACHE_TTL_SECONDS  # match L1 so both layers age together
_REDIS_PAYLOAD_MAX_BYTES: int = 256 * 1024  # 256 KB cap — articles can be chunky


# --- schemas ----------------------------------------------------------------


class GDELTArticle(BaseModel):
    """One GDELT 2.0 article record."""

    url: str
    title: str
    source: str = Field(..., description="Domain of the publisher (e.g. ``bbc.com``).")
    country: str = Field(..., description="GDELT-reported source country.")
    ts: str = Field(..., description="ISO-8601 UTC timestamp.")
    tone: float = Field(0.0, description="GDELT tone in [-10, +10]; 0 if not provided.")
    language: str
    image_url: str | None = None
    relevance_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Question-relevance in [0, 1]; 0 for non-slug endpoints.",
    )
    matched_terms: list[str] = Field(
        default_factory=list,
        description="Anchor/topic terms that contributed to the relevance score.",
    )
    nlp_sentiment_score: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Hybrid VADER + finance-lex sentiment, signed [-1,+1]. Distinct from `tone` which is GDELT-native.",
    )
    nlp_sentiment_label: str = Field(
        "neutral",
        description="Bucketed label: positive / negative / neutral (±0.15 deadband).",
    )


class SourceCount(BaseModel):
    source: str
    n_articles: int


class TerminalGDELTResponse(BaseModel):
    slug: str
    query_used: str
    n_articles: int
    articles: list[GDELTArticle]
    mean_tone: float = Field(..., description="Average article tone; 0 if all missing.")
    dominant_topic: str = Field(..., description="Most frequent keyword across titles.")
    top_sources: list[SourceCount]
    anchors: list[str] = Field(
        default_factory=list,
        description="Question anchor terms (entities) used for relevance scoring.",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Question topic terms (content words) used for relevance scoring.",
    )
    relevance_min: float = Field(
        RELEVANCE_MIN,
        ge=0.0,
        le=1.0,
        description="Score threshold below which articles are dropped.",
    )


class TerminalGDELTBreakingResponse(BaseModel):
    timespan: str = Field(..., description="GDELT timespan window (e.g. ``6h``).")
    n_articles: int
    articles: list[GDELTArticle]


# --- in-memory cache --------------------------------------------------------
# Keyed by ``"slug:<slug>:<limit>"`` or ``"breaking:<limit>"``.
# ``_CACHE`` stays exposed as a module-level dict so test fixtures that
# call ``_CACHE.clear()`` keep working; the TerminalCache wrapper adds
# the TTL + thread-safety logic on top.
#
# We use the process-wide ``get_cache("terminal_gdelt")`` factory to match
# the pattern in pfm/terminal/{search_index,quote,homepage,peer_scanner}.py,
# and back it with our exposed ``_CACHE`` dict so test fixtures that import
# and ``.clear()`` it continue to work unchanged.

_CACHE: dict[str, tuple[float, dict]] = {}
_cache = get_cache("terminal_gdelt", ttl=CACHE_TTL_SECONDS)
# Repoint the shared cache's backing store to our exposed dict so the named
# singleton and the module-level ``_CACHE`` reference the same object. This
# is idempotent: subsequent imports of the module re-bind to the same dict.
_cache._store = _CACHE


def _cache_get(key: str) -> dict | None:
    return _cache.get(key)


def _cache_set(key: str, payload: dict) -> None:
    _cache.set(key, payload)


def _redis_lock_key(cache_key: str) -> str:
    """Redis key for the single-flight SETNX lock around a cache key."""
    return f"terminal_gdelt:lock:{cache_key}"


def _try_acquire_lock(request: Request, cache_key: str) -> bool:
    """Acquire a cross-worker SETNX lock; return True if we should fetch.

    Mirrors the pattern in ``main.py``'s ``_gamma_price_prewarm`` — only the
    worker that wins the SETNX call hits GDELT; others wait + re-read the
    L1 cache that the winner populated. If Redis is offline (NullCache),
    ``setnx`` returns True so we degrade to single-process semantics.
    """
    cache = getattr(request.app.state, "cache", None)
    if cache is None:
        return True  # no Redis wired (tests) — single-process, just fetch.
    try:
        return bool(cache.setnx(_redis_lock_key(cache_key), b"1", _LOCK_TTL_SECONDS))
    except Exception:  # pragma: no cover - defensive
        return True  # fail open: better to double-fetch than to hang


def _wait_for_other_worker(request: Request, cache_key: str) -> dict | None:
    """Poll BOTH the in-process L1 and the Redis L2 while another worker fetches.

    Returns the cached payload as soon as one of the layers materialises
    (whichever wins). With the L2 promotion this works cross-worker too:
    worker A wins the SETNX lock and writes both L1 + L2; worker B was
    waiting and sees the L2 entry on the next poll.

    Returns ``None`` if the wait budget runs out — caller falls back to
    fetching itself.
    """
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(_LOCK_POLL_INTERVAL)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        l2 = _redis_payload_get(request, cache_key)
        if l2 is not None:
            # Promote into our local L1 so we don't keep polling Redis on
            # subsequent requests within this worker.
            _cache_set(cache_key, l2)
            return l2
    return None


def _redis_payload_key(cache_key: str) -> str:
    """Redis key for the cached response payload (distinct from the lock key)."""
    return f"terminal_gdelt:payload:{cache_key}"


def _redis_payload_get(request: Request, cache_key: str) -> dict | None:
    """L2 read. ``None`` on miss, decode error, or any Redis hiccup."""
    cache = getattr(request.app.state, "cache", None)
    if cache is None or not getattr(cache, "enabled", False):
        return None
    raw: bytes | None = None
    with contextlib.suppress(Exception):  # defensive: never break on cache I/O
        raw = cache.get(_redis_payload_key(cache_key))
    if not raw:
        return None
    try:
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _redis_payload_set(request: Request, cache_key: str, payload: dict) -> None:
    """L2 write, skipping silently when the payload exceeds the size cap."""
    cache = getattr(request.app.state, "cache", None)
    if cache is None or not getattr(cache, "enabled", False):
        return
    try:
        blob = json.dumps(payload, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return
    if len(blob) > _REDIS_PAYLOAD_MAX_BYTES:
        logger.info(
            "terminal_gdelt: skipping Redis SET — payload %d B > %d B cap",
            len(blob),
            _REDIS_PAYLOAD_MAX_BYTES,
        )
        return
    with contextlib.suppress(Exception):  # defensive
        cache.set(_redis_payload_key(cache_key), blob, _REDIS_TTL_SECONDS)


# --- helpers ----------------------------------------------------------------


# Tiny stop-words list for dominant_topic; we don't need the full set from
# terminal_news because article titles tend to be punchier than market
# questions. Kept inline so the dependency surface is small.
_TITLE_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "after",
        "before",
        "over",
        "under",
        "about",
        "says",
        "said",
        "will",
        "would",
        "could",
        "should",
        "have",
        "has",
        "had",
        "are",
        "was",
        "were",
        "been",
        "being",
        "but",
        "not",
        "all",
        "new",
        "amid",
        "what",
        "when",
        "why",
        "how",
        "who",
    }
)


def _seendate_to_iso(seendate: str) -> str:
    """Convert GDELT ``seendate`` (``YYYYMMDDTHHMMSSZ``) to ISO-8601."""
    s = seendate.strip()
    if len(s) < 15 or "T" not in s:
        return s
    # "20260304T171500Z" → "2026-03-04T17:15:00Z"
    date, _, rest = s.partition("T")
    if len(date) != 8 or len(rest) < 7:
        return s
    yyyy, mm, dd = date[0:4], date[4:6], date[6:8]
    hh, mi, ss = rest[0:2], rest[2:4], rest[4:6]
    return f"{yyyy}-{mm}-{dd}T{hh}:{mi}:{ss}Z"


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host.lower().removeprefix("www.")


def _coerce_tone(raw: object) -> float:
    """GDELT may return tone under several keys — best-effort float coerce."""
    if raw is None:
        return 0.0
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _parse_articles(payload: dict) -> list[GDELTArticle]:
    """Map a raw GDELT JSON response → list[GDELTArticle]."""
    out: list[GDELTArticle] = []
    for art in payload.get("articles") or []:
        if not isinstance(art, dict):
            continue
        url = str(art.get("url") or "").strip()
        title = str(art.get("title") or "").strip()
        if not url or not title:
            continue
        domain = str(art.get("domain") or "").strip().lower() or _domain_of(url)
        country = str(art.get("sourcecountry") or "").strip() or "Unknown"
        language = str(art.get("language") or "").strip() or "Unknown"
        ts = _seendate_to_iso(str(art.get("seendate") or ""))
        # Tone may live under "tone" (preferred) or "avgtone".
        tone = _coerce_tone(art.get("tone") if "tone" in art else art.get("avgtone"))
        image = str(art.get("socialimage") or "").strip() or None
        # NLP sentiment — boosts headlines vanilla VADER misses on finance
        # vocabulary, and incorporates GDELT's own tone as a 10% factor.
        # Import inside the loop to avoid a circular import at module load.
        try:
            from pfm.terminal.sentiment_nlp import score_headline as _score

            nlp_score, nlp_label = _score(title, external_tone=tone)
        except Exception:  # never break the fetch on NLP error
            nlp_score, nlp_label = 0.0, "neutral"
        out.append(
            GDELTArticle(
                url=url,
                title=title,
                source=domain,
                country=country,
                ts=ts,
                tone=tone,
                language=language,
                image_url=image,
                nlp_sentiment_score=nlp_score,
                nlp_sentiment_label=nlp_label,
            )
        )
    return out


def _dominant_topic(articles: list[GDELTArticle], fallback: str = "") -> str:
    """Most-frequent non-stopword token across article titles."""
    counter: Counter[str] = Counter()
    for art in articles:
        for tok in re.findall(r"[A-Za-z]{3,}", art.title.lower()):
            if tok in _TITLE_STOP_WORDS:
                continue
            counter[tok] += 1
    if not counter:
        return fallback
    return counter.most_common(1)[0][0]


def _top_sources(articles: list[GDELTArticle], n: int = 5) -> list[SourceCount]:
    counter: Counter[str] = Counter(a.source for a in articles if a.source)
    return [SourceCount(source=src, n_articles=cnt) for src, cnt in counter.most_common(n)]


def _build_query(keywords: list[str]) -> str:
    """GDELT takes space-separated terms (URL-encoded by httpx). Quote multi-words."""
    parts: list[str] = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        parts.append(f'"{kw}"' if " " in kw else kw)
    return " ".join(parts)


def _fetch_gdelt(
    client: httpx.Client,
    query: str,
    max_records: int,
    timespan: str | None = None,
) -> list[GDELTArticle]:
    """Hit GDELT 2.0 doc API and return parsed articles. Empty list on failure."""
    params: dict[str, str | int] = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(int(max_records), HARD_CAP_RECORDS),
        "sort": "hybridrel",
    }
    if timespan:
        params["timespan"] = timespan
    try:
        r = client.get(
            GDELT_DOC_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=5.0,
        )
    except httpx.HTTPError as e:
        logger.warning("gdelt fetch failed: %s", e)
        return []
    if r.status_code >= 400:
        logger.warning("gdelt non-2xx: %s body=%s", r.status_code, r.text[:200])
        return []
    body = r.text or ""
    # GDELT throttles with a plaintext "Please limit requests..." message.
    if body.lstrip().startswith("Please limit"):
        logger.warning("gdelt throttled this caller")
        return []
    try:
        payload = r.json()
    except ValueError:
        logger.warning("gdelt returned non-JSON: %s", body[:200])
        return []
    if not isinstance(payload, dict):
        return []
    return _parse_articles(payload)


# --- dependency -------------------------------------------------------------


router = APIRouter(prefix="/terminal/gdelt", tags=["terminal"])


def get_polymarket_client(request: Request) -> PolymarketClient:
    poly = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


# --- endpoints --------------------------------------------------------------


@router.get(
    "/breaking",
    response_model=TerminalGDELTBreakingResponse,
    summary="Top global breaking-news headlines from GDELT (last 6 hours).",
)
def get_gdelt_breaking(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> TerminalGDELTBreakingResponse:
    """Return up to ``limit`` recent global headlines (English, hybridrel-sorted)."""
    cache_key = f"breaking:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return TerminalGDELTBreakingResponse(**cached)

    # L2 cross-worker cache — saves the ~8.3 s cold cross-worker round-trip.
    l2 = _redis_payload_get(request, cache_key)
    if l2 is not None:
        _cache_set(cache_key, l2)  # promote into L1
        return TerminalGDELTBreakingResponse(**l2)

    # Single-flight: only the SETNX winner pays the ~8.6 s GDELT round-trip.
    # Other concurrent first-hits wait briefly for the winner's L1+L2 set.
    if not _try_acquire_lock(request, cache_key):
        waited = _wait_for_other_worker(request, cache_key)
        if waited is not None:
            return TerminalGDELTBreakingResponse(**waited)
        # Wait budget elapsed — fall through and fetch ourselves.

    # We need an httpx.Client; reuse the one on the polymarket client if it's
    # there, but fall back to a fresh client so /breaking works even if
    # app.state.poly is not configured.
    poly = getattr(request.app.state, "poly", None)
    client: httpx.Client = poly._client if poly is not None else httpx.Client(timeout=15.0)

    # "sourcelang:english" keeps the feed consumable; users can compose their
    # own multi-lingual query via the per-slug endpoint.
    query = "sourcelang:english"
    articles = _fetch_gdelt(client, query, limit, timespan=BREAKING_TIMESPAN)

    # GDELT enforces ~1 req / 5s. When throttled, fall back to curated RSS
    # pulls so the breaking-ticker never goes blank. RssHeadline has fields
    # title/link/pub_date/source_name (NOT url/ts/published — different schema).
    if not articles:
        try:
            from pfm.terminal_rss_news import SOURCES as _RSS_SOURCES
            from pfm.terminal_rss_news import _fetch_source as _rss_fetch_source

            for src in _RSS_SOURCES:
                if len(articles) >= limit:
                    break
                try:
                    items, _err = _rss_fetch_source(client, src)
                except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                    logger.warning("rss fallback source %r failed: %s", src, exc)
                    continue
                for h in items[: max(2, limit - len(articles))]:
                    title = (getattr(h, "title", "") or "").strip()
                    if not title:
                        continue
                    _ext_tone = float(getattr(h, "sentiment_score", 0.0) or 0.0)
                    try:
                        from pfm.terminal.sentiment_nlp import score_headline as _score

                        _nlp_s, _nlp_l = _score(title, external_tone=_ext_tone * 10.0)
                    except Exception:
                        _nlp_s, _nlp_l = 0.0, "neutral"
                    articles.append(
                        GDELTArticle(
                            url=getattr(h, "link", "") or "",
                            title=title,
                            source=getattr(h, "source_name", None) or getattr(src, "name", "wire"),
                            country="Unknown",
                            ts=getattr(h, "pub_date", None),
                            tone=_ext_tone,
                            language="English",
                            image_url=None,
                            nlp_sentiment_score=_nlp_s,
                            nlp_sentiment_label=_nlp_l,
                        )
                    )
                    if len(articles) >= limit:
                        break
            logger.info("breaking-fallback: rss yielded %d articles", len(articles))
        except (ImportError, ValueError, RuntimeError, httpx.HTTPError) as exc:
            # Defensive: rss module unavailable, malformed feed, or
            # pydantic validation. Fallback is best-effort so swallow.
            logger.warning("rss fallback failed: %s", exc)

    response = TerminalGDELTBreakingResponse(
        timespan=BREAKING_TIMESPAN,
        n_articles=len(articles),
        articles=articles,
    )
    payload = response.model_dump()
    _cache_set(cache_key, payload)
    _redis_payload_set(request, cache_key, payload)
    return response


@router.get(
    "/{slug}",
    response_model=TerminalGDELTResponse,
    summary="GDELT 2.0 global news for a Polymarket market's topic.",
)
def get_gdelt_news(
    request: Request,
    slug: Annotated[str, Path(min_length=1, description="Polymarket market slug.")],
    limit: Annotated[int, Query(ge=1, le=HARD_CAP_RECORDS)] = DEFAULT_MAX_RECORDS,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> TerminalGDELTResponse:
    """Return GDELT articles relevant to the topic of ``slug`` plus aggregates."""
    cache_key = f"slug:{slug}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return TerminalGDELTResponse(**cached)

    # L2 cross-worker read — cold-worker hits return in ~5 ms instead of ~8.3 s.
    l2 = _redis_payload_get(request, cache_key)
    if l2 is not None:
        _cache_set(cache_key, l2)
        return TerminalGDELTResponse(**l2)

    # Single-flight: only the SETNX winner pays the ~8.6 s GDELT round-trip
    # for this slug. Other concurrent first-hits wait for the winner's set.
    if not _try_acquire_lock(request, cache_key):
        waited = _wait_for_other_worker(request, cache_key)
        if waited is not None:
            return TerminalGDELTResponse(**waited)
        # Wait budget elapsed — fall through and fetch ourselves.

    # Resolve question via Gamma. Explicit 5 s timeout so a slow gamma
    # response can't push the handler past the 15 s gateway deadline.
    try:
        r = poly._client.get(
            f"{poly.gamma_url}/markets",
            params={"slug": slug},
            timeout=5.0,
        )
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e

    market = payload[0] if isinstance(payload, list) and payload else None
    if not market:
        raise HTTPException(status_code=404, detail=f"no market found for slug={slug!r}")
    question = str(market.get("question") or "").strip()
    if not question:
        raise HTTPException(
            status_code=502, detail=f"market {slug!r} missing question in gamma payload"
        )

    keywords = extract_keywords(question, max_n=MAX_KEYWORDS)
    if not keywords:
        # Fallback: tokenise the slug itself.
        keywords = [t for t in re.split(r"[-_]+", slug) if len(t) >= 3][:MAX_KEYWORDS]

    # Build proper anchor/topic terms — these tighten the GDELT query
    # (quoted anchors) and drive a post-fetch relevance filter so an
    # NVDA-AI question never returns generic "AI" stories.
    terms: QuestionTerms = build_terms(question)
    query = build_phrase_query(terms) or _build_query(keywords)
    # Fetch ~3x what the caller asked for so the filter has headroom.
    fetch_n = min(HARD_CAP_RECORDS, max(int(limit) * 3, 25))
    articles = _fetch_gdelt(poly._client, query, fetch_n)

    # Score + filter. If no terms could be extracted (rare, e.g. only
    # stopwords), keep all articles so the panel doesn't go blank.
    if terms.anchors or terms.topics:
        kept: list[GDELTArticle] = []
        for art in articles:
            score, matched = score_relevance(art.title, terms)
            if score < RELEVANCE_MIN:
                continue
            kept.append(
                art.model_copy(
                    update={"relevance_score": round(score, 4), "matched_terms": matched}
                )
            )
        kept.sort(key=lambda a: (a.relevance_score, a.ts), reverse=True)
        articles = kept[: int(limit)]
    else:
        articles = articles[: int(limit)]

    mean_tone = sum(a.tone for a in articles) / len(articles) if articles else 0.0
    fallback_topic = keywords[0] if keywords else slug
    response = TerminalGDELTResponse(
        slug=slug,
        query_used=query,
        n_articles=len(articles),
        articles=articles,
        mean_tone=round(mean_tone, 4),
        dominant_topic=_dominant_topic(articles, fallback=fallback_topic),
        top_sources=_top_sources(articles),
        anchors=list(terms.anchors),
        topics=list(terms.topics),
        relevance_min=RELEVANCE_MIN,
    )
    payload = response.model_dump()
    _cache_set(cache_key, payload)
    _redis_payload_set(request, cache_key, payload)
    return response
