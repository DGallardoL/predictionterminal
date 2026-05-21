"""Terminal news/sentiment feed: Reddit + Hacker News discussion for a market.

Exposes a single endpoint:

    GET /terminal/news/{slug}?limit=20

Resolves the Polymarket question for ``slug`` via Gamma ``/markets?slug=X``,
extracts 2-3 keywords from the question (simple stop-word filter), then
queries two public no-auth endpoints:

  - Reddit:  https://www.reddit.com/search.json?q=...&sort=new
             (rate-limited; requires User-Agent header).
  - HN:      https://hn.algolia.com/api/v1/search?query=...&tags=story
             (no auth, no documented rate limit).

Posts are merged, deduped on URL, sorted newest-first, and tagged with a
naive sentiment label (positive / negative / neutral) using a small
word-list lookup. A 10-minute in-memory cache fronts the upstream calls
because users tend to flip between markets in the Terminal UI.

Routing note: this module owns its :class:`fastapi.APIRouter`; ``main.py``
is left untouched (per CLAUDE.md). To activate the endpoint::

    from pfm.terminal_news import router as terminal_news_router
    app.include_router(terminal_news_router)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal.news_relevance import (
    RELEVANCE_MIN,
    QuestionTerms,
    build_reddit_query,
    build_terms,
    score_relevance,
)

logger = logging.getLogger(__name__)

REDDIT_SEARCH_URL: str = "https://www.reddit.com/search.json"
HN_SEARCH_URL: str = "https://hn.algolia.com/api/v1/search"
USER_AGENT: str = "polymarket-terminal/1.0"
CACHE_TTL_SECONDS: int = 600  # 10 minutes
MAX_KEYWORDS: int = 3
# Single retry on 429 from Reddit/HN. Aligns with pfm.sources.polymarket's
# pattern so the whole stack has uniform back-off semantics.
_RETRY_BACKOFF_S: float = 1.5

Source = Literal["reddit", "hn"]
Sentiment = Literal["positive", "negative", "neutral"]


# --- schemas ----------------------------------------------------------------


class NewsItem(BaseModel):
    """One Reddit or HN post mentioning the market topic."""

    source: Source = Field(..., description="``reddit`` or ``hn``.")
    title: str
    url: str
    ts: str = Field(..., description="ISO-8601 UTC timestamp.")
    score: int = Field(..., description="Upvotes / points (0 if missing).")
    sentiment: Sentiment = Field(..., description="Naive word-list sentiment label.")
    relevance_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Question-relevance in [0, 1]; higher = closer match.",
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


class TerminalNewsResponse(BaseModel):
    slug: str
    question: str
    keywords: list[str]
    n_items: int
    items: list[NewsItem]
    anchors: list[str] = Field(
        default_factory=list,
        description="Question anchor terms (entities) used for matching.",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Question topic terms (content words) used for matching.",
    )
    relevance_min: float = Field(
        RELEVANCE_MIN,
        ge=0.0,
        le=1.0,
        description="Score threshold below which items are dropped.",
    )
    degraded_mode: bool = Field(
        False,
        description=(
            "True when every upstream news source (Reddit + HN) failed; the "
            "payload still validates (empty ``items``) so the UI renders a "
            "graceful 'news unavailable' state instead of a 502."
        ),
    )


# --- stop-words & sentiment word lists -------------------------------------
# Kept tiny on purpose — this is a POC, not an NLP pipeline.

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "than",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "any",
        "all",
        "each",
        "every",
        "no",
        "not",
        "yes",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "over",
        "under",
        "again",
        "further",
        "once",
        # Polymarket question-specific noise
        "win",
        "wins",
        "winner",
        "lose",
        "loses",
        "happen",
        "happens",
        "occur",
        "occurs",
        "say",
        "says",
        "said",
        "vs",
        # Verbs / nouns that show up in nearly every question and surface
        # off-topic results when fed to upstream search APIs.
        "end",
        "ends",
        "ended",
        "ending",
        "draw",
        "draws",
        "drawn",
        "reach",
        "reaches",
        "reached",
        "hit",
        "hits",
        "hitting",
        "year",
        "years",
        "month",
        "months",
        "day",
        "days",
        "high",
        "low",
        # numbers / quantifiers that aren't useful as keywords
        "more",
        "most",
        "less",
        "least",
    }
)

_POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "good",
        "great",
        "excellent",
        "positive",
        "strong",
        "bullish",
        "rally",
        "surge",
        "soar",
        "boom",
        "growth",
        "gain",
        "gains",
        "win",
        "wins",
        "winning",
        "success",
        "successful",
        "hope",
        "optimistic",
        "breakthrough",
        "record",
        "best",
        "love",
        "amazing",
    }
)

_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "bad",
        "terrible",
        "awful",
        "negative",
        "weak",
        "bearish",
        "crash",
        "plunge",
        "tumble",
        "collapse",
        "loss",
        "losses",
        "lose",
        "lost",
        "fail",
        "failure",
        "fear",
        "panic",
        "fraud",
        "scandal",
        "worst",
        "hate",
        "disaster",
        "recession",
        "decline",
    }
)


# --- in-memory cache --------------------------------------------------------
# Keyed by (slug, limit). Value is (expiry_unix_seconds, response_dict).

_CACHE: dict[tuple[str, int], tuple[float, dict]] = {}


def _cache_get(key: tuple[str, int]) -> dict | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    expiry, payload = entry
    if expiry < time.time():
        _CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: tuple[str, int], payload: dict) -> None:
    _CACHE[key] = (time.time() + CACHE_TTL_SECONDS, payload)


# --- keyword extraction & sentiment ----------------------------------------


def extract_keywords(question: str, max_n: int = MAX_KEYWORDS) -> list[str]:
    """Return up to ``max_n`` keywords from a Polymarket question.

    Tokenises on non-alphanumeric, lowercases, drops stop-words and tokens
    shorter than 3 chars, and preserves original order while de-duping.

    Accents are NFKD-normalized BEFORE tokenisation so non-ASCII proper
    nouns survive intact: "Castellón" → "castellon", "Cádiz" → "cadiz",
    "Müller" → "muller". Without this the ``[A-Za-z0-9]+`` regex would
    split "castellón" into ["castell", "n"] and the news lookup would
    surface CD-collection results instead of the actual football club.
    """
    import unicodedata

    # Decompose accented chars (NFKD) then strip combining marks so the
    # ASCII regex below matches the base letter.
    folded = unicodedata.normalize("NFKD", question)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    tokens = re.findall(r"[A-Za-z0-9]+", folded.lower())
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        if len(tok) < 3 or tok in _STOP_WORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_n:
            break
    return out


def classify_sentiment(text: str) -> Sentiment:
    """Three-way sentiment label for a single headline.

    Delegates to :func:`pfm.terminal.sentiment_nlp.score_headline`, the
    hybrid VADER + financial-lex scorer used elsewhere in the Terminal.
    This keeps every callsite (``quote.py``, ``news.py``, etc.) on the
    same scoring pipeline so a headline like "NVDA surges to record high"
    no longer reads as ``neutral`` just because the word-list missed it.

    The ``_POSITIVE_WORDS`` / ``_NEGATIVE_WORDS`` module constants are
    retained — other modules may still import them for unrelated heuristics
    — but the classifier itself no longer uses them.
    """
    # Import locally to avoid a circular import at module-load time
    # (sentiment_nlp is light, but it pulls vaderSentiment which we want
    # to defer until classify_sentiment is actually invoked).
    from pfm.terminal.sentiment_nlp import score_headline

    _compound, label = score_headline(text)
    return label  # type: ignore[return-value]  # Label literal matches Sentiment


# --- upstream fetchers ------------------------------------------------------


def _fetch_reddit(client: httpx.Client, query: str, limit: int) -> tuple[list[NewsItem], bool]:
    """Hit Reddit's public search.json with a single 429-retry.

    Returns ``(items, ok)``. ``ok`` is ``True`` if the upstream returned a
    2xx (even with zero items). ``False`` means the call was rate-limited
    or errored out — the caller uses that to decide whether to mark the
    response ``degraded_mode``.

    Sort is ``relevance`` (Reddit's TF-IDF-ish ranking) — we found
    ``sort=new`` returned chronologically-fresh-but-off-topic posts.
    The post-fetch relevance filter still applies on top of this.
    """
    params = {
        "q": query,
        "sort": "relevance",
        # Fetch up to 3x the caller's limit so the relevance
        # filter has room to drop off-topic hits before we
        # truncate. Reddit caps at 100 internally.
        "limit": int(min(100, max(limit * 3, 25))),
        "t": "month",  # last month only — keeps cold ancient threads out
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        r = client.get(REDDIT_SEARCH_URL, params=params, headers=headers)
    except httpx.HTTPError as e:
        logger.warning("reddit search failed: %s", e)
        return [], False
    # Single retry on 429 — Reddit's free endpoint is bursty but a short
    # back-off usually clears the bucket without us having to give up.
    if r.status_code == 429:
        logger.warning("reddit 429 — retrying in %.1fs", _RETRY_BACKOFF_S)
        time.sleep(_RETRY_BACKOFF_S)
        try:
            r = client.get(REDDIT_SEARCH_URL, params=params, headers=headers)
        except httpx.HTTPError as e:
            logger.warning("reddit retry failed: %s", e)
            return [], False
    if r.status_code == 429:
        logger.warning("reddit rate limited (429 after retry); degraded")
        return [], False
    if r.status_code >= 400:
        logger.warning("reddit search non-2xx: %s", r.status_code)
        return [], False
    try:
        payload = r.json()
    except ValueError:
        return [], False

    out: list[NewsItem] = []
    for child in payload.get("data", {}).get("children", []):
        d = child.get("data", {}) if isinstance(child, dict) else {}
        title = str(d.get("title") or "").strip()
        permalink = d.get("permalink") or ""
        url = f"https://www.reddit.com{permalink}" if permalink else str(d.get("url") or "")
        if not title or not url:
            continue
        created = d.get("created_utc")
        ts = _unix_to_iso(created)
        score = int(d.get("score") or 0)
        try:
            from pfm.terminal.sentiment_nlp import score_headline as _score

            _nlp_s, _nlp_l = _score(title)
        except Exception:
            _nlp_s, _nlp_l = 0.0, "neutral"
        out.append(
            NewsItem(
                source="reddit",
                title=title,
                url=url,
                ts=ts,
                score=score,
                sentiment=classify_sentiment(title),
                nlp_sentiment_score=_nlp_s,
                nlp_sentiment_label=_nlp_l,
            )
        )
    return out, True


def _fetch_hn(client: httpx.Client, query: str, limit: int) -> tuple[list[NewsItem], bool]:
    """Hit HN Algolia /search?tags=story with a single 429-retry.

    Returns ``(items, ok)`` — see :func:`_fetch_reddit` for the contract.
    HN doesn't normally rate-limit but we keep the retry path symmetric
    so a Cloudflare interstitial doesn't kill the panel.
    """
    # Last 90 days = 90 * 86400 = 7,776,000 seconds.
    ninety_days_ago = int(time.time()) - 90 * 86400
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": int(min(50, max(limit * 3, 20))),
        "numericFilters": f"created_at_i>{ninety_days_ago}",
    }
    try:
        r = client.get(HN_SEARCH_URL, params=params)
    except httpx.HTTPError as e:
        logger.warning("hn search failed: %s", e)
        return [], False
    if r.status_code == 429:
        logger.warning("hn 429 — retrying in %.1fs", _RETRY_BACKOFF_S)
        time.sleep(_RETRY_BACKOFF_S)
        try:
            r = client.get(HN_SEARCH_URL, params=params)
        except httpx.HTTPError as e:
            logger.warning("hn retry failed: %s", e)
            return [], False
    if r.status_code >= 400:
        logger.warning("hn search non-2xx: %s", r.status_code)
        return [], False
    try:
        payload = r.json()
    except ValueError:
        return [], False

    out: list[NewsItem] = []
    for hit in payload.get("hits", []):
        title = str(hit.get("title") or hit.get("story_title") or "").strip()
        url = str(
            hit.get("url")
            or hit.get("story_url")
            or (
                f"https://news.ycombinator.com/item?id={hit['objectID']}"
                if hit.get("objectID")
                else ""
            )
        )
        if not title or not url:
            continue
        ts = str(hit.get("created_at") or "")
        if ts and not ts.endswith("Z"):
            # Algolia returns "2025-09-01T12:34:56.000Z" already; defensive.
            ts = ts.replace("+00:00", "Z")
        score = int(hit.get("points") or 0)
        try:
            from pfm.terminal.sentiment_nlp import score_headline as _score

            _nlp_s, _nlp_l = _score(title)
        except Exception:
            _nlp_s, _nlp_l = 0.0, "neutral"
        out.append(
            NewsItem(
                source="hn",
                title=title,
                url=url,
                ts=ts,
                score=score,
                sentiment=classify_sentiment(title),
                nlp_sentiment_score=_nlp_s,
                nlp_sentiment_label=_nlp_l,
            )
        )
    return out, True


def _unix_to_iso(t: object) -> str:
    if t is None:
        return ""
    try:
        secs = float(t)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(t)
    import pandas as pd

    return pd.Timestamp(secs, unit="s", tz="UTC").isoformat().replace("+00:00", "Z")


# --- dependency -------------------------------------------------------------


router = APIRouter(prefix="/terminal", tags=["terminal"])


def get_polymarket_client(request: Request) -> PolymarketClient:
    poly = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


# --- endpoint ---------------------------------------------------------------


@router.get(
    "/news/{slug}",
    response_model=TerminalNewsResponse,
    summary="Recent Reddit + HN posts mentioning a Polymarket market's topic.",
)
def get_terminal_news(
    slug: Annotated[str, Path(min_length=1, description="Polymarket market slug.")],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> TerminalNewsResponse:
    """Return recent Reddit + HN posts for the topic of ``slug``."""
    cached = _cache_get((slug, limit))
    if cached is not None:
        return TerminalNewsResponse(**cached)

    # Resolve question from Gamma. We use the existing PolymarketClient's
    # underlying httpx.Client to inherit any timeouts / retries it has.
    # We re-route through the cached `poly.get_market_metadata` so we benefit
    # from polymarket.py's 1h slug→meta cache + 1.5s 429-retry. Falling all the
    # way through to a 502 here triggers a red error card in the UI; instead
    # we degrade to an empty payload with `degraded_mode=True` so the panel
    # renders a friendly empty state.
    try:
        meta = poly.get_market_metadata(slug)
        question = (meta.question or "").strip()
        if not question:
            return TerminalNewsResponse(
                slug=slug,
                question="",
                keywords=[],
                n_items=0,
                items=[],
                anchors=[],
                topics=[],
                relevance_min=RELEVANCE_MIN,
                degraded_mode=True,
            )
    except httpx.HTTPError as e:
        # Upstream gamma 429/5xx — return degraded payload, NOT 502.
        logger.info("news: gamma unavailable for %s (%s) — returning degraded payload", slug, e)
        return TerminalNewsResponse(
            slug=slug,
            question="",
            keywords=[],
            n_items=0,
            items=[],
            anchors=[],
            topics=[],
            relevance_min=RELEVANCE_MIN,
            degraded_mode=True,
        )
    except Exception as e:
        # Catches PolymarketError ("no market found", "no clobTokenIds") etc.
        # 404 is the right code for an unknown slug, but everything else
        # should still return a degraded payload.
        msg = str(e)
        if "no market found" in msg:
            raise HTTPException(status_code=404, detail=msg) from e
        logger.warning("news: unexpected error for %s: %s — returning degraded payload", slug, e)
        return TerminalNewsResponse(
            slug=slug,
            question="",
            keywords=[],
            n_items=0,
            items=[],
            anchors=[],
            topics=[],
            relevance_min=RELEVANCE_MIN,
            degraded_mode=True,
        )

    keywords = extract_keywords(question)
    if not keywords:
        # Fallback to the slug itself, broken on hyphens.
        keywords = [t for t in re.split(r"[-_]+", slug) if len(t) >= 3][:MAX_KEYWORDS]

    # Anchor/topic terms drive the relevance filter. ``build_terms``
    # uses the same question as ``extract_keywords`` but separates
    # entities ("NVDA", "Joe Biden") from generic content words.
    terms: QuestionTerms = build_terms(question)
    reddit_query = build_reddit_query(terms) or " ".join(keywords)
    hn_query = build_reddit_query(terms) or " ".join(keywords)

    reddit_items, reddit_ok = _fetch_reddit(poly._client, reddit_query, limit)
    hn_items, hn_ok = _fetch_hn(poly._client, hn_query, limit)
    # Degraded only when BOTH upstreams failed (rather than just one
    # returning zero hits). Caller still gets a valid envelope with
    # empty ``items`` so the UI can render "news unavailable" cleanly.
    degraded = not (reddit_ok or hn_ok)

    merged: list[NewsItem] = reddit_items + hn_items
    # Dedupe on URL (preserve first occurrence).
    seen_urls: set[str] = set()
    deduped: list[NewsItem] = []
    for it in merged:
        if it.url in seen_urls:
            continue
        seen_urls.add(it.url)
        deduped.append(it)

    # Apply relevance filter + score. If anchors/topics are empty
    # (very rare — e.g. all-stopword questions) we degrade to "keep
    # everything" so the endpoint never silently goes blank.
    if terms.anchors or terms.topics:
        scored: list[NewsItem] = []
        for it in deduped:
            score, matched = score_relevance(it.title, terms)
            if score < RELEVANCE_MIN:
                continue
            scored.append(
                it.model_copy(update={"relevance_score": round(score, 4), "matched_terms": matched})
            )
        # Sort by relevance desc, then recency desc as tie-break.
        scored.sort(key=lambda it: (it.relevance_score, it.ts), reverse=True)
        filtered = scored[:limit]
    else:
        # No usable terms — keep legacy newest-first behaviour.
        deduped.sort(key=lambda it: it.ts, reverse=True)
        filtered = deduped[:limit]

    response = TerminalNewsResponse(
        slug=slug,
        question=question,
        keywords=keywords,
        n_items=len(filtered),
        items=filtered,
        anchors=list(terms.anchors),
        topics=list(terms.topics),
        relevance_min=RELEVANCE_MIN,
        degraded_mode=degraded,
    )
    # Don't cache degraded responses — if both upstreams just hiccuped we
    # want the next request to retry rather than serve "news unavailable"
    # for the full 10-min TTL.
    if not degraded:
        _cache_set((slug, limit), response.model_dump())
    return response
