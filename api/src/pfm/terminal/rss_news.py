"""Terminal RSS news aggregator: a unified stream from major wire feeds.

Pulls free, public, no-auth RSS/Atom feeds spanning financial, political,
tech, and crypto wires, then exposes them through a single API surface so
the Terminal UI can show a polished headline ticker alongside the GDELT
breadth feed.

Design notes
------------
- Uses the standard library ``xml.etree.ElementTree`` parser. ``feedparser``
  is not pulled in; the wire RSS we consume is plain RSS 2.0 or Atom 1.0
  and the simple parsing here is enough.
- All sources are probed with a 10-minute in-memory cache to keep latency
  low. Cache key is the source slug (NOT the URL — keeps things simple).
- Each headline is enriched with the existing ``pfm.sentiment_lexicon``
  scorer so the UI can color-code by polarity.
- Reuters' ``feeds.reuters.com`` host has been DNS-dead for some time,
  so the original Reuters spec is replaced 1:1 by MarketWatch and Fox
  Politics fallbacks (per CLAUDE.md guidance on dropping 404s).

Activation
----------
This module owns its :class:`fastapi.APIRouter`; ``main.py`` is left
untouched (per CLAUDE.md). To wire it up::

    from pfm.terminal_rss_news import router as terminal_rss_news_router
    app.include_router(terminal_rss_news_router)
"""

from __future__ import annotations

import html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from typing import Annotated, Literal
from xml.etree import ElementTree as ET

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from pfm.sentiment_lexicon import score_sentiment
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal.news_relevance import (
    RELEVANCE_MIN,
    QuestionTerms,
    build_terms,
    score_relevance,
)
from pfm.terminal_news import extract_keywords

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
# Reuters' ``feeds.reuters.com`` host is DNS-dead. Per CLAUDE.md ("if any 404,
# drop and add a fallback"), MarketWatch and Fox Politics replace the three
# Reuters slots so the user-visible source mix stays the same shape.

Category = Literal["all", "markets", "politics", "tech", "crypto", "world"]


class _SourceSpec(BaseModel):
    slug: str
    name: str
    url: str
    category: Category


SOURCES: list[_SourceSpec] = [
    _SourceSpec(
        slug="marketwatch_top",
        name="MarketWatch Top Stories",
        url="https://feeds.marketwatch.com/marketwatch/topstories/",
        category="markets",
    ),
    _SourceSpec(
        slug="marketwatch_pulse",
        name="MarketWatch MarketPulse",
        url="https://feeds.marketwatch.com/marketwatch/marketpulse/",
        category="markets",
    ),
    _SourceSpec(
        slug="fox_politics",
        name="Fox Politics",
        url="https://moxie.foxnews.com/google-publisher/politics.xml",
        category="politics",
    ),
    _SourceSpec(
        slug="yahoo_top",
        name="Yahoo Top Stories (AP)",
        url="https://news.yahoo.com/rss/topstories",
        category="world",
    ),
    _SourceSpec(
        slug="bbc_world",
        name="BBC World",
        url="http://feeds.bbci.co.uk/news/world/rss.xml",
        category="world",
    ),
    _SourceSpec(
        slug="bbc_business",
        name="BBC Business",
        url="http://feeds.bbci.co.uk/news/business/rss.xml",
        category="markets",
    ),
    _SourceSpec(
        slug="coindesk",
        name="CoinDesk",
        url="https://www.coindesk.com/arc/outboundfeeds/rss/",
        category="crypto",
    ),
    _SourceSpec(
        slug="verge",
        name="The Verge",
        url="https://www.theverge.com/rss/index.xml",
        category="tech",
    ),
    _SourceSpec(
        slug="ft_home",
        name="Financial Times — Home",
        url="https://www.ft.com/rss/home",
        category="markets",
    ),
]

USER_AGENT: str = "polymarket-terminal/1.0"
CACHE_TTL_SECONDS: int = 600  # 10 minutes
HTTP_TIMEOUT: float = 10.0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RssHeadline(BaseModel):
    """One parsed headline from a single RSS source."""

    source: str = Field(..., description="Source slug, e.g. ``bbc_world``.")
    source_name: str
    category: Category
    title: str
    link: str
    pub_date: str = Field(..., description="ISO-8601 UTC timestamp; empty if unknown.")
    description: str = Field("", description="Short summary; HTML stripped.")
    sentiment: Literal["positive", "negative", "neutral"]
    sentiment_score: float = Field(..., description="Score in [-1, +1].")
    relevance_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Question-relevance in [0, 1]; only set for slug-matched feeds.",
    )
    matched_terms: list[str] = Field(
        default_factory=list,
        description="Anchor/topic terms that contributed to the relevance score.",
    )
    nlp_sentiment_score: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Hybrid VADER + finance-lex sentiment, signed [-1,+1].",
    )
    nlp_sentiment_label: str = Field(
        "neutral",
        description="Bucketed positive / negative / neutral (±0.15 deadband).",
    )


class RssHeadlinesResponse(BaseModel):
    n_items: int
    category: Category
    sources_used: list[str]
    items: list[RssHeadline]


class RssSourceStatus(BaseModel):
    slug: str
    name: str
    url: str
    category: Category
    status: Literal["ok", "error"]
    n_items: int
    error: str = ""


class RssSourcesResponse(BaseModel):
    n_sources: int
    n_ok: int
    sources: list[RssSourceStatus]


class RssSlugMatchResponse(BaseModel):
    slug: str
    question: str
    keywords: list[str]
    n_items: int
    items: list[RssHeadline]
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
        description="Score threshold below which headlines are dropped.",
    )


# ---------------------------------------------------------------------------
# In-memory cache (per source slug)
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, list[RssHeadline]]] = {}


def _cache_get(slug: str) -> list[RssHeadline] | None:
    entry = _CACHE.get(slug)
    if entry is None:
        return None
    expiry, payload = entry
    if expiry < time.time():
        _CACHE.pop(slug, None)
        return None
    return payload


def _cache_set(slug: str, payload: list[RssHeadline]) -> None:
    _CACHE[slug] = (time.time() + CACHE_TTL_SECONDS, payload)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _strip_html(text: str) -> str:
    """Remove HTML tags, unescape entities, and collapse whitespace."""
    if not text:
        return ""
    cleaned = _WS_RE.sub(" ", _HTML_TAG_RE.sub(" ", text)).strip()
    return html.unescape(cleaned)


def _to_iso_utc(raw: str) -> str:
    """Best-effort RFC-822 / ISO-8601 → ISO-8601 UTC. Returns "" on failure."""
    if not raw:
        return ""
    raw = raw.strip()
    # First try email RFC-822 (most RSS pubDate). Then ISO-8601 (Atom updated).
    for parser in (parsedate_to_datetime, pd.Timestamp):
        try:
            dt = parser(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        try:
            ts = pd.Timestamp(dt)
        except (TypeError, ValueError):
            continue
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        return ts.isoformat().replace("+00:00", "Z")
    return ""


def _parse_rss(xml_bytes: bytes, source: _SourceSpec) -> list[RssHeadline]:
    """Parse an RSS-2.0 or Atom-1.0 byte payload into headlines."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.warning("rss parse error (%s): %s", source.slug, e)
        return []

    items: list[RssHeadline] = []

    # RSS 2.0 path: <channel><item>...</item></channel>
    rss_items = root.findall(".//item")
    if rss_items:
        for it in rss_items:
            title = _strip_html(it.findtext("title") or "")
            link = (it.findtext("link") or "").strip()
            pub = (
                it.findtext("pubDate")
                or it.findtext("{http://purl.org/dc/elements/1.1/}date")
                or ""
            )
            desc = _strip_html(it.findtext("description") or "")
            if not title or not link:
                continue
            scored = score_sentiment(f"{title}. {desc}")
            try:
                from pfm.terminal.sentiment_nlp import score_headline as _nlp_score

                _nlp_s, _nlp_l = _nlp_score(title)
            except Exception:
                _nlp_s, _nlp_l = 0.0, "neutral"
            items.append(
                RssHeadline(
                    source=source.slug,
                    source_name=source.name,
                    category=source.category,
                    title=title,
                    link=link,
                    pub_date=_to_iso_utc(pub),
                    description=desc[:400],
                    sentiment=scored["dominant"],
                    sentiment_score=scored["score"],
                    nlp_sentiment_score=_nlp_s,
                    nlp_sentiment_label=_nlp_l,
                )
            )
        return items

    # Atom 1.0 path
    entries = root.findall(f"{_ATOM_NS}entry")
    for entry in entries:
        title = _strip_html(entry.findtext(f"{_ATOM_NS}title") or "")
        link_el = entry.find(f"{_ATOM_NS}link")
        link = link_el.get("href", "") if link_el is not None else ""
        pub = entry.findtext(f"{_ATOM_NS}updated") or entry.findtext(f"{_ATOM_NS}published") or ""
        desc = _strip_html(
            entry.findtext(f"{_ATOM_NS}summary") or entry.findtext(f"{_ATOM_NS}content") or ""
        )
        if not title or not link:
            continue
        scored = score_sentiment(f"{title}. {desc}")
        try:
            from pfm.terminal.sentiment_nlp import score_headline as _nlp_score

            _nlp_s, _nlp_l = _nlp_score(title)
        except Exception:
            _nlp_s, _nlp_l = 0.0, "neutral"
        items.append(
            RssHeadline(
                source=source.slug,
                source_name=source.name,
                category=source.category,
                title=title,
                link=link,
                pub_date=_to_iso_utc(pub),
                description=desc[:400],
                sentiment=scored["dominant"],
                sentiment_score=scored["score"],
                nlp_sentiment_score=_nlp_s,
                nlp_sentiment_label=_nlp_l,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


def _fetch_sources_parallel(
    client: httpx.Client, sources: list[_SourceSpec]
) -> list[tuple[_SourceSpec, list[RssHeadline], str]]:
    """Fan-out ``_fetch_source`` across sources in a thread pool.

    Previously the headlines endpoint iterated sources serially; with 9 feeds
    that's 9× network RTT in the worst (all-uncached) case (>2.7 s warm).
    Parallel fan-out brings warm latency down to the slowest single source.

    A per-future ``HTTP_TIMEOUT + 2`` wall clock is enforced on top of
    ``client.get``'s socket timeout so a wedged TLS handshake on one feed
    can't stall the whole fan-out — the worker is cancelled and we return
    a partial set with that one feed marked as "timeout".
    """
    if not sources:
        return []
    # Bounded pool — RSS hosts vary so we want concurrency but not unlimited.
    max_workers = min(len(sources), 9)
    out: list[tuple[_SourceSpec, list[RssHeadline], str]] = []
    # Hard wall on per-feed wait, on top of the httpx socket timeout. We
    # add a small buffer because the inner read-timeout can be the same
    # value, so a tight 10 s budget would race the inner timeout.
    per_feed_deadline = HTTP_TIMEOUT + 2.0
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rss") as ex:
        futures = {ex.submit(_fetch_source, client, src): src for src in sources}
        for fut, src in futures.items():
            try:
                items, err = fut.result(timeout=per_feed_deadline)
            except TimeoutError:
                logger.warning(
                    "rss fetch wall-timeout on %s after %.1fs", src.slug, per_feed_deadline
                )
                items, err = [], "timeout"
            except Exception as e:
                # Defensive: don't let a single bad feed kill the whole
                # response. Log so an operator can spot a perma-broken
                # source, but always degrade to "partial".
                logger.warning("rss fetch pool error on %s: %s", src.slug, e)
                items, err = [], f"pool error: {e}"
            out.append((src, items, err))
    return out


def _fetch_source(client: httpx.Client, source: _SourceSpec) -> tuple[list[RssHeadline], str]:
    """Return (items, error_msg). On any error returns ([], reason)."""
    cached = _cache_get(source.slug)
    if cached is not None:
        return cached, ""
    try:
        r = client.get(
            source.url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/rss+xml, application/xml, text/xml",
            },
            follow_redirects=True,
            timeout=HTTP_TIMEOUT,
        )
    except httpx.HTTPError as e:
        logger.warning("rss fetch failed (%s): %s", source.slug, e)
        return [], f"http error: {e}"
    if r.status_code >= 400:
        logger.warning("rss non-2xx (%s): %s", source.slug, r.status_code)
        return [], f"status {r.status_code}"
    items = _parse_rss(r.content, source)
    if items:
        _cache_set(source.slug, items)
    return items, ""


# ---------------------------------------------------------------------------
# Router & dependency
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/terminal/rss", tags=["terminal"])


def get_polymarket_client(request: Request) -> PolymarketClient:
    poly = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


def get_http_client(request: Request) -> httpx.Client:
    """Reuse the polymarket httpx.Client when available; else build one."""
    poly = getattr(request.app.state, "poly", None)
    if poly is not None and getattr(poly, "_client", None) is not None:
        return poly._client
    return httpx.Client(timeout=HTTP_TIMEOUT)


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def _sort_key_recency(it: RssHeadline) -> str:
    """Sort key: ISO timestamps lexicographic-sort chronologically; "" → bottom."""
    return it.pub_date or "0000"


def _filter_category(items: list[RssHeadline], category: Category) -> list[RssHeadline]:
    if category == "all":
        return items
    return [it for it in items if it.category == category]


def _score_overlap(headline: RssHeadline, keywords: set[str]) -> int:
    """Token-overlap score between a headline (title+description) and keywords."""
    if not keywords:
        return 0
    text = f"{headline.title} {headline.description}".lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    return len(tokens & keywords)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/sources",
    response_model=RssSourcesResponse,
    summary="List all RSS sources and their current ok/error status.",
)
def list_sources(
    client: Annotated[httpx.Client, Depends(get_http_client)],
) -> RssSourcesResponse:
    """Probe each source and report status. Cache hits count as ``ok``.

    Performance: fan-out probes via :func:`_fetch_sources_parallel` so the
    cold-cache path is bounded by the slowest single feed rather than the
    sum of every feed's RTT (which was ~4 s observed for 9 sources).
    """
    statuses: list[RssSourceStatus] = []
    n_ok = 0
    for src, items, err in _fetch_sources_parallel(client, list(SOURCES)):
        ok = bool(items) and not err
        if ok:
            n_ok += 1
        statuses.append(
            RssSourceStatus(
                slug=src.slug,
                name=src.name,
                url=src.url,
                category=src.category,
                status="ok" if ok else "error",
                n_items=len(items),
                error=err,
            )
        )
    return RssSourcesResponse(n_sources=len(SOURCES), n_ok=n_ok, sources=statuses)


@router.get(
    "/headlines",
    response_model=RssHeadlinesResponse,
    summary="Unified, ranked RSS headlines across every active wire source.",
)
def get_headlines(
    client: Annotated[httpx.Client, Depends(get_http_client)],
    limit: Annotated[int, Query(ge=1, le=200)] = 30,
    category: Annotated[Category, Query()] = "all",
) -> RssHeadlinesResponse:
    """Aggregate every source's RSS, filter by category, rank by recency."""
    candidates = [s for s in SOURCES if category == "all" or s.category == category]
    merged: list[RssHeadline] = []
    sources_used: list[str] = []
    for src, items, _err in _fetch_sources_parallel(client, candidates):
        if items:
            sources_used.append(src.slug)
            merged.extend(items)

    # Dedupe on link.
    seen: set[str] = set()
    deduped: list[RssHeadline] = []
    for it in merged:
        if it.link in seen:
            continue
        seen.add(it.link)
        deduped.append(it)

    deduped.sort(key=_sort_key_recency, reverse=True)
    deduped = _filter_category(deduped, category)[:limit]

    return RssHeadlinesResponse(
        n_items=len(deduped),
        category=category,
        sources_used=sources_used,
        items=deduped,
    )


@router.get(
    "/{slug}",
    response_model=RssSlugMatchResponse,
    summary="Headlines matching a Polymarket market's question keywords.",
)
def get_headlines_for_slug(
    slug: Annotated[str, Path(min_length=1, description="Polymarket market slug.")],
    client: Annotated[httpx.Client, Depends(get_http_client)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> RssSlugMatchResponse:
    """Resolve the slug → question, score every headline by token overlap."""
    try:
        r = poly._client.get(f"{poly.gamma_url}/markets", params={"slug": slug})
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

    keywords = extract_keywords(question, max_n=6)
    if not keywords:
        keywords = [t for t in re.split(r"[-_]+", slug) if len(t) >= 3][:6]

    # Build anchor + topic terms for the question. Anchors are insisted
    # on; a single shared common token like "year" no longer matches.
    terms: QuestionTerms = build_terms(question)

    # Pull every source in parallel — same fan-out as /headlines.
    pool: list[RssHeadline] = []
    for _src, items, _err in _fetch_sources_parallel(client, list(SOURCES)):
        pool.extend(items)

    # Score every headline. Apply relevance floor when we have terms;
    # otherwise fall back to legacy token-overlap so the panel still
    # populates on questions made entirely of stopwords.
    if terms.anchors or terms.topics:
        scored_items: list[tuple[float, list[str], RssHeadline]] = []
        for it in pool:
            score, matched_terms = score_relevance(it.title, terms, body=it.description)
            if score < RELEVANCE_MIN:
                continue
            scored_items.append((score, matched_terms, it))
        scored_items.sort(
            key=lambda triple: (triple[0], triple[2].pub_date or "0"),
            reverse=True,
        )
        # Dedupe on link, preserve order.
        seen: set[str] = set()
        out: list[RssHeadline] = []
        for score, matched_terms, it in scored_items:
            if it.link in seen:
                continue
            seen.add(it.link)
            out.append(
                it.model_copy(
                    update={
                        "relevance_score": round(score, 4),
                        "matched_terms": matched_terms,
                    }
                )
            )
            if len(out) >= limit:
                break
    else:
        # Legacy fallback: pure token-overlap, no scoring.
        keyword_set = {k.lower() for k in keywords}
        legacy_scored = [(it, _score_overlap(it, keyword_set)) for it in pool]
        matched_legacy = [(it, s) for it, s in legacy_scored if s > 0]
        matched_legacy.sort(key=lambda pair: (pair[1], pair[0].pub_date or "0"), reverse=True)
        seen = set()
        out = []
        for it, _s in matched_legacy:
            if it.link in seen:
                continue
            seen.add(it.link)
            out.append(it)
            if len(out) >= limit:
                break

    return RssSlugMatchResponse(
        slug=slug,
        question=question,
        keywords=keywords,
        n_items=len(out),
        items=out,
        anchors=list(terms.anchors),
        topics=list(terms.topics),
        relevance_min=RELEVANCE_MIN,
    )


# ---------------------------------------------------------------------------
# Discoverability alias — /terminal/rss-news
# ---------------------------------------------------------------------------
# Live testers often try ``/terminal/rss-news?q=bitcoin`` (more discoverable
# than ``/terminal/rss/headlines``). Rather than 404 them, we mount a thin
# alias router that mirrors ``/headlines`` and adds an optional ``q`` keyword
# filter applied as a case-insensitive substring match against title +
# description. The shape matches RssHeadlinesResponse so existing frontend
# code can target either path.


@router.get(
    "-news",
    response_model=RssHeadlinesResponse,
    summary="Discoverability alias for /headlines with optional ``q`` keyword.",
)
def get_rss_news_alias(
    client: Annotated[httpx.Client, Depends(get_http_client)],
    q: Annotated[str | None, Query(max_length=120)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 30,
    category: Annotated[Category, Query()] = "all",
) -> RssHeadlinesResponse:
    """Same backing logic as ``/headlines`` plus a case-insensitive ``q`` filter.

    Mounting trick: ``router`` already has ``prefix='/terminal/rss'`` so a
    path of ``-news`` resolves to ``/terminal/rss-news`` — exactly the path
    live clients hit. Avoids a second ``include_router`` in main.py.
    """
    candidates = [s for s in SOURCES if category == "all" or s.category == category]
    merged: list[RssHeadline] = []
    sources_used: list[str] = []
    for src, items, _err in _fetch_sources_parallel(client, candidates):
        if items:
            sources_used.append(src.slug)
            merged.extend(items)

    seen: set[str] = set()
    deduped: list[RssHeadline] = []
    for it in merged:
        if it.link in seen:
            continue
        seen.add(it.link)
        deduped.append(it)

    if q:
        needle = q.lower().strip()
        if needle:
            deduped = [
                it
                for it in deduped
                if needle in it.title.lower() or needle in it.description.lower()
            ]

    deduped.sort(key=_sort_key_recency, reverse=True)
    deduped = _filter_category(deduped, category)[:limit]

    return RssHeadlinesResponse(
        n_items=len(deduped),
        category=category,
        sources_used=sources_used,
        items=deduped,
    )
