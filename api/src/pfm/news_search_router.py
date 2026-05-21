"""``GET /news/search`` — semantic news search with factor-tag linking.

Task W11-22 (T32). A lightweight search-style news endpoint that complements
the per-slug ``/terminal/gdelt/{slug}`` and ``/terminal/news/{slug}`` panels
by letting the user query the GDELT 2.0 doc API directly with an arbitrary
free-form string, scored by token-Jaccard against the query and (optionally)
linked back to matching factor catalog entries.

The router intentionally lives at top-level ``pfm/news_search_router.py``
rather than inside ``pfm/terminal/`` because the path is ``/news/search``
(not ``/terminal/news/...``) and the response shape doesn't carry any of
the per-slug context (no Polymarket question, no anchors / topics).

Integration
-----------
Add to ``main.py`` under the ``main.py:routes`` claim::

    from pfm.news_search_router import router as _news_search_router
    app.include_router(_news_search_router)

Caching
-------
Module-level TTL cache keyed on ``(q_normalised, since, factors)`` with a
60 s TTL. The query is normalised (lowercase + whitespace-collapsed) before
keying so trivial casing variants share a cache slot.

Scoring
-------
Token-Jaccard between the query and each article title, after lowercasing
and stop-word stripping. Results are sorted by descending score and capped
at 50.

Factor linking
--------------
When ``factors=true`` (default), each article is matched against
``app.state.factors_by_slug`` by intersecting tokens in the article title
with tokens in ``FactorConfig.name``. Slugs whose name shares ≥1 non-stop
token with the title are attached as ``matched_factors``. The match is
case-insensitive and never raises if ``factors_by_slug`` is missing.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pfm.factors import FactorConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["news"])

# --- constants --------------------------------------------------------------

GDELT_DOC_URL: str = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT: str = "polymarket-terminal/1.0 (news-search)"

# Wall-clock used for cache expiry. Tests patch this to drive expiry without
# real sleep. ``time.monotonic`` is monotonic, cheap, and immune to NTP jumps.
_MONOTONIC: Callable[[], float] = time.monotonic

# Cache TTL in seconds. News is fast-moving; 60 s of staleness on a search
# panel is well within usefulness and matches the ``factors_related_router``
# TTL pattern.
CACHE_TTL_SECONDS: float = 60.0

# Hard cap on result count. GDELT returns up to ~250 articles but the UI
# table is meant to be skim-read so anything above 50 is wasteful.
MAX_RESULTS: int = 50

# Allowed ``since`` window values and their mapping to GDELT timespan
# parameters. GDELT accepts ``1h``/``24h``/``7days``/``30days`` (NOT ``7d``).
SinceWindow = Literal["1h", "24h", "7d", "30d"]
_GDELT_TIMESPAN: dict[str, str] = {
    "1h": "1h",
    "24h": "24h",
    "7d": "7days",
    "30d": "30days",
}

# Minimal English stop-word set tuned for news headlines + market questions.
# Kept small on purpose: aggressive stopword removal would collapse queries
# that genuinely differ on a function word. Reused from ``news_dedupe`` but
# duplicated here so the router doesn't take a dependency on that module's
# import side-effects.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "are",
        "been",
        "being",
        "do",
        "does",
        "did",
        "had",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "can",
        "if",
        "then",
        "else",
        "than",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "each",
        "every",
        "no",
        "not",
        "yes",
        "vs",
        "via",
        "amid",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --- schemas ----------------------------------------------------------------


class NewsSearchResult(BaseModel):
    """One scored news article matching the query."""

    title: str
    url: str
    source: str = Field(..., description="Publisher domain (e.g. ``bbc.com``).")
    published_at: str = Field(..., description="ISO-8601 UTC timestamp from GDELT ``seendate``.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Token-Jaccard similarity between query and title, in [0,1].",
    )
    matched_factors: list[str] = Field(
        default_factory=list,
        description=(
            "Factor slugs whose ``name`` shares ≥1 non-stop token with the "
            "article title. Empty when ``factors=false`` or no factor matches."
        ),
    )


class NewsSearchResponse(BaseModel):
    """Envelope for ``GET /news/search``."""

    q: str = Field(..., description="The original query string, echoed back.")
    since: SinceWindow = Field(..., description="Time window applied to the upstream news search.")
    results: list[NewsSearchResult]
    count: int = Field(..., ge=0, description="Length of ``results`` (after cap of 50).")


# --- token helpers ----------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, drop stop-words, return token *set*.

    A set (not a list) is the right shape for Jaccard scoring and factor
    intersection. We drop tokens shorter than 2 characters to suppress
    noisy single-letter matches (e.g. "S" in "S&P 500").
    """
    return {
        tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= 2 and tok not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard index of two token sets. Returns 0.0 when both are empty."""
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    union = a | b
    return len(inter) / len(union)


# --- cache ------------------------------------------------------------------


class _Entry:
    __slots__ = ("expires_at", "payload")

    def __init__(self, payload: NewsSearchResponse, expires_at: float) -> None:
        self.payload = payload
        self.expires_at = expires_at


_CACHE: dict[tuple[str, str, bool], _Entry] = {}
_CACHE_LOCK = threading.Lock()


def _cache_key(q: str, since: str, factors: bool) -> tuple[str, str, bool]:
    """Normalise the query (lowercase + collapse whitespace) and form the key."""
    q_norm = " ".join(q.lower().split())
    return (q_norm, since, factors)


def _cache_get(key: tuple[str, str, bool]) -> NewsSearchResponse | None:
    now = _MONOTONIC()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            _CACHE.pop(key, None)
            return None
        return entry.payload


def _cache_put(key: tuple[str, str, bool], payload: NewsSearchResponse) -> None:
    expires_at = _MONOTONIC() + CACHE_TTL_SECONDS
    with _CACHE_LOCK:
        _CACHE[key] = _Entry(payload, expires_at)


def _cache_clear() -> None:
    """Drop every cache entry — used by tests to force cold paths."""
    with _CACHE_LOCK:
        _CACHE.clear()


# --- GDELT timestamp helper -------------------------------------------------


def _seendate_to_iso(seendate: str) -> str:
    """``YYYYMMDDTHHMMSSZ`` → ISO-8601. Mirrors ``terminal.gdelt_news``.

    Returns the input string unchanged on a malformed value so the router
    never 5xx's on an odd GDELT response — we'd rather render a slightly
    ugly timestamp than fail the whole panel.
    """
    s = seendate.strip()
    if len(s) < 15 or "T" not in s:
        return s
    date, _, rest = s.partition("T")
    if len(date) != 8 or len(rest) < 7:
        return s
    yyyy, mm, dd = date[0:4], date[4:6], date[6:8]
    hh, mi, ss = rest[0:2], rest[2:4], rest[4:6]
    return f"{yyyy}-{mm}-{dd}T{hh}:{mi}:{ss}Z"


def _domain_of(url: str) -> str:
    """Best-effort host extraction. Returns ``""`` on a malformed URL."""
    from urllib.parse import urlparse

    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host.lower().removeprefix("www.")


# --- GDELT fetch ------------------------------------------------------------


def _fetch_gdelt_articles(
    client: httpx.Client, query: str, timespan: str, max_records: int = 75
) -> list[dict]:
    """Fetch raw GDELT articles for the search query.

    Returns a list of raw article dicts (NOT parsed). Empty on any upstream
    failure — we degrade gracefully rather than 5xx the user.
    """
    params: dict[str, str | int] = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": int(max_records),
        "sort": "hybridrel",
        "timespan": timespan,
    }
    try:
        r = client.get(
            GDELT_DOC_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
        )
    except httpx.HTTPError as e:
        logger.warning("news_search: gdelt fetch failed: %s", e)
        return []
    if r.status_code >= 400:
        logger.warning("news_search: gdelt non-2xx: %s", r.status_code)
        return []
    body = r.text or ""
    if body.lstrip().startswith("Please limit"):
        logger.warning("news_search: gdelt throttled this caller")
        return []
    try:
        payload = r.json()
    except ValueError:
        logger.warning("news_search: gdelt returned non-JSON")
        return []
    if not isinstance(payload, dict):
        return []
    arts = payload.get("articles") or []
    return [a for a in arts if isinstance(a, dict)]


# --- factor-match helper ----------------------------------------------------


def _match_factors(
    title_tokens: set[str],
    factors_by_slug: dict[str, FactorConfig] | None,
    *,
    max_attached: int = 5,
) -> list[str]:
    """Return factor slugs whose ``name`` shares ≥1 non-stop token with the title.

    Cap at ``max_attached`` per article so a generic title doesn't drag in
    every macro factor. The cap is applied in catalog-iteration order; the
    UI only needs a handful of pills per row.
    """
    if not factors_by_slug or not title_tokens:
        return []
    out: list[str] = []
    for slug, fc in factors_by_slug.items():
        name_tokens = _tokenize(fc.name)
        if not name_tokens:
            continue
        if name_tokens & title_tokens:
            out.append(slug)
            if len(out) >= max_attached:
                break
    return out


# --- endpoint ---------------------------------------------------------------


@router.get(
    "/news/search",
    response_model=NewsSearchResponse,
    summary="Free-form news search scored by token-Jaccard, optionally factor-tagged.",
)
def news_search(
    request: Request,
    q: Annotated[
        str,
        Query(
            min_length=1,
            max_length=200,
            description="Free-form query string (required).",
        ),
    ],
    since: Annotated[
        SinceWindow,
        Query(description="Time window: 1h / 24h / 7d / 30d."),
    ] = "7d",
    factors: Annotated[
        bool,
        Query(description="When true, attach matching factor slugs per result."),
    ] = True,
) -> NewsSearchResponse:
    """Semantic news search with optional factor-tag linking.

    Returns up to 50 results sorted by descending token-Jaccard score.
    Empty result lists are valid (``count: 0``) — the upstream returning
    nothing or being unavailable both surface as ``[]`` rather than 5xx.
    """
    q_stripped = q.strip()
    if not q_stripped:
        # Empty after strip is treated as a 422 to match the FastAPI Query
        # min_length=1 semantics (which rejects "" but not "   ").
        raise HTTPException(
            status_code=422, detail="q must contain at least one non-whitespace character"
        )

    key = _cache_key(q_stripped, since, factors)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    timespan = _GDELT_TIMESPAN[since]

    # httpx.Client: reuse the one on the polymarket client if present so we
    # benefit from the shared connection pool. Fall back to a one-shot client
    # otherwise (tests without app.state.poly hit this path via respx).
    client: httpx.Client | None = None
    poly = getattr(request.app.state, "poly", None)
    if poly is not None:
        client = getattr(poly, "_client", None)
    owns_client = False
    if client is None:
        client = httpx.Client(timeout=httpx.Timeout(10.0))
        owns_client = True

    try:
        raw_articles = _fetch_gdelt_articles(client, q_stripped, timespan)
    finally:
        if owns_client:
            client.close()

    factors_by_slug: dict[str, FactorConfig] | None = getattr(
        request.app.state, "factors_by_slug", None
    )

    q_tokens = _tokenize(q_stripped)

    scored: list[NewsSearchResult] = []
    for art in raw_articles:
        title = str(art.get("title") or "").strip()
        url = str(art.get("url") or "").strip()
        if not title or not url:
            continue
        title_tokens = _tokenize(title)
        score = _jaccard(q_tokens, title_tokens)
        published_at = _seendate_to_iso(str(art.get("seendate") or ""))
        domain = str(art.get("domain") or "").strip().lower() or _domain_of(url)
        matched = _match_factors(title_tokens, factors_by_slug) if factors else []
        scored.append(
            NewsSearchResult(
                title=title,
                url=url,
                source=domain or "gdelt",
                published_at=published_at,
                score=score,
                matched_factors=matched,
            )
        )

    # Sort by score desc, then by published_at desc (lexicographic on ISO-8601
    # is monotone in time) so equally-scored items present newest-first.
    scored.sort(key=lambda r: (-r.score, _negated_ts(r.published_at)))
    capped = scored[:MAX_RESULTS]

    response = NewsSearchResponse(
        q=q_stripped,
        since=since,
        results=capped,
        count=len(capped),
    )
    _cache_put(key, response)
    return response


def _negated_ts(ts: str) -> str:
    """Tie-break helper: invert ISO-8601 lexicographic order via char negation.

    We can't return a tuple of (-score, ts_desc) directly because Python's
    sort is stable and strings have no natural reverse comparator inline
    with a numeric. This builds a string that sorts in REVERSE chronological
    order so we can keep the sort key as ``(-score, negated_ts)``.
    """
    # Map each char c to chr(0x7F - ord(c)); inverts lexicographic order
    # for printable ASCII. Non-ASCII (rare in ISO-8601) falls back to ord 0.
    return "".join(chr(0x7F - (ord(c) if ord(c) < 0x7F else 0x7E)) for c in ts)
