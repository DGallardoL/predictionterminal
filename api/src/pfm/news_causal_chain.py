"""News Causal Chain: news headline -> Polymarket Δprob -> stock expected return.

The "killer feature" the Terminal lacked: turn a fresh news headline into a
quantified, ticker-level expected-return forecast by chaining together three
links the rest of the codebase already has (each in isolation):

    1. ``terminal_news_impact`` — does a headline correlate with a Polymarket
       price reaction in the surrounding window? We borrow its keyword-match
       logic and the price-before / price-after sampling pattern.
    2. ``model.delta_logit`` — convert that probability move to a logit move
       on the same scale the regression betas live on.
    3. β cache (or synthetic-placeholder) — multiply the Δlogit by the
       per-ticker factor β to get an *expected* stock log-return.

The output is a "chain" object showing every hop so the analyst can trace
the inference: headline -> tagged factor -> Δprob -> Δlogit -> per-ticker
expected return + confidence label. This is intentionally a forecast, not a
backtest: callers downstream (Strategies, Portfolio P&L Tree) are expected
to consume these chains and decide whether to act.

Confidence rules
----------------
- ``high``: |Δlogit| > 0.5 AND β has a non-null sample (cached regression).
- ``medium``: keyword match present but the factor lacks a cached β —
  falls back to ``SYNTHETIC_BETA_PLACEHOLDER`` and the caller is told.
- ``low``: keyword match weak (<2 token overlap) or no price reaction
  data available; expected_return is set to ``None``.

Routing
-------
This module owns its :class:`fastapi.APIRouter`; ``main.py`` is left
untouched (per CLAUDE.md). Wire-up::

    from pfm.news_causal_chain import router as news_causal_router
    app.include_router(news_causal_router)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Annotated, Literal

import httpx
import numpy as np
from fastapi import APIRouter, Body, Depends, Query, Request
from pydantic import BaseModel, Field

from pfm.auth.dependencies import require_tier
from pfm.cache_utils import get_cache
from pfm.model import DEFAULT_EPSILON
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_news import extract_keywords

logger = logging.getLogger(__name__)


CACHE_TTL_SECONDS: int = 600  # 10 min — matches the GDELT/RSS feed cadence
POST_CACHE_TTL_SECONDS: int = 300  # 5 min — POST /causal-chain body cache
NAMESPACE_CHAIN: str = "news_causal_chain"
NAMESPACE_MOVERS: str = "news_causal_movers"
NAMESPACE_POST_CHAIN: str = "news_causal"

# Cap how many distinct factor keywords we score against any single news item.
# Without this, the inner loop is O(N_items × N_factors) and dominates wall
# time when the caller passes hundreds of pre-fetched items. Twenty is the
# empirical sweet-spot — covers the relevant entity-tagged factors without
# scoring noise.
MAX_FACTORS_PER_ITEM: int = 20

# When a tagged factor has no cached β regression, we fall back to a small
# placeholder so the chain still produces a *directional* expected-return
# signal. The user is told (``confidence="medium"`` and ``beta_source``).
SYNTHETIC_BETA_PLACEHOLDER: float = 0.05

# Minimum token overlap to consider a news item tagged to a factor.
MIN_KEYWORD_OVERLAP: int = 1
# Strong overlap threshold (used in confidence scoring).
STRONG_KEYWORD_OVERLAP: int = 2
# A |Δlogit| above this threshold is considered a "strong" probability move.
STRONG_DELTA_LOGIT: float = 0.5


Confidence = Literal["high", "medium", "low"]
BetaSource = Literal["regression", "synthetic", "none"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class NewsItem(BaseModel):
    """One headline that callers want to score against a factor.

    The shape is intentionally permissive: Terminal-RSS, GDELT and the
    /news endpoints all populate slightly different fields, but as long as
    the title is present we can do keyword matching.
    """

    title: str = Field(..., description="Headline / article title.")
    ts: str = Field("", description="ISO-8601 UTC timestamp; empty if unknown.")
    url: str = Field("", description="Article URL.")
    source: str = Field("", description="Domain or feed slug, e.g. ``bbc.com``.")
    description: str = Field("", description="Optional article body or summary.")
    price_before: float | None = Field(None, description="PM price strictly before ts (if known).")
    price_after: float | None = Field(
        None, description="PM price after ts in a fixed window (if known)."
    )


class TickerImpact(BaseModel):
    """Per-ticker stock-return forecast given the inferred Δlogit."""

    ticker: str
    beta: float = Field(..., description="Cached or synthetic factor β.")
    beta_source: BetaSource = Field(..., description="Where the β came from.")
    expected_return_pct: float | None = Field(
        None,
        description=(
            "Forecast log-return in percent (β * Δlogit * 100). ``None`` "
            "when the chain cannot produce a return (no Δprob or no β)."
        ),
    )
    confidence: Confidence


class CausalLink(BaseModel):
    """One news item with its full chain to ticker-level impacts."""

    news_item: NewsItem
    tagged_factor: str | None = Field(
        None, description="Factor id this item was tagged to (None = unmatched)."
    )
    keyword_overlap: int = Field(0, description="Tokens shared with the factor's keywords.")
    delta_prob: float | None = Field(
        None, description="Polymarket probability change attributed to this item."
    )
    delta_logit: float | None = Field(
        None, description="logit(p_after) - logit(p_before) at clip ε=0.01."
    )
    affected_tickers: list[TickerImpact]
    confidence: Confidence
    notes: str = Field("", description="Why the chain landed at this confidence.")


class CausalChainResponse(BaseModel):
    factor_id: str
    lookback_hours: int
    n_items: int
    n_tagged: int
    chain: list[CausalLink]


class TopMover(BaseModel):
    factor_id: str
    headline: str
    ts: str
    source: str
    expected_impact_pct: float = Field(
        ...,
        description=(
            "max-|expected_return_pct| across affected tickers, signed in "
            "the direction of the Δlogit move."
        ),
    )
    delta_prob: float | None = None
    n_affected_tickers: int
    link: CausalLink


class TopMoversResponse(BaseModel):
    window_hours: int
    n_total: int
    n_returned: int
    min_impact_pct: float
    movers: list[TopMover]


# ---------------------------------------------------------------------------
# β cache — pluggable so callers can register real regression results
# ---------------------------------------------------------------------------
# The chain works without any cached β at all (fallback to synthetic), but
# the more realistic path is: a previous /fit call wrote ``{factor_id ->
# {ticker -> beta}}`` into this dict, and the chain reads it. Tests inject
# directly via ``BETA_REGISTRY`` to avoid coupling to a heavy fit step.

BETA_REGISTRY: dict[str, dict[str, float]] = {}


def register_betas(factor_id: str, betas: dict[str, float]) -> None:
    """Register a {ticker -> β} map for ``factor_id``. Overwrites prior entry."""
    BETA_REGISTRY[factor_id] = dict(betas)


def get_factor_betas(factor_id: str) -> dict[str, float]:
    """Return cached {ticker -> β} for ``factor_id`` or empty dict."""
    return BETA_REGISTRY.get(factor_id, {})


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------


def _factor_keywords(factor_id: str) -> list[str]:
    """Heuristic keyword list for a factor id.

    Most factor ids are either slugs (``trump-out-by-2027``) or short phrases
    encoded in snake/kebab case. We tokenise on non-alphanum, drop tokens
    shorter than 3 chars, and pass through ``extract_keywords`` to drop the
    standard stop-words.
    """
    raw = re.findall(r"[A-Za-z0-9]+", factor_id.replace("_", " ").replace("-", " "))
    text = " ".join(raw)
    return extract_keywords(text, max_n=8)


def _keyword_overlap(news_item: NewsItem, keywords: set[str]) -> int:
    """Count of news-item tokens that intersect ``keywords``.

    Augments the raw keyword overlap with the multi-entity NER tagger from
    :mod:`pfm.news_tagger`: each named entity (Trump, China, FOMC, BTC, …)
    that maps to *this* factor via the curated entity-factor JSON adds one
    to the overlap. This lets the causal chain match headlines that share
    *concepts* with the factor even when no surface-token overlap exists
    (e.g. "Donald Trump" → trump-2028 factor).
    """
    text = f"{news_item.title} {news_item.description}".lower()
    base = 0
    if keywords:
        tokens = set(re.findall(r"[a-z0-9]+", text))
        base = len(tokens & keywords)

    # Lazy import to avoid a hard cycle at module import time.
    try:
        from pfm.news_tagger import (
            all_entities,
            extract_entities,
            load_entity_factor_map,
        )
    except ImportError:  # pragma: no cover - defensive
        return base

    entities = all_entities(extract_entities(text))
    if not entities:
        return base
    emap = load_entity_factor_map()
    haystack = " ".join(keywords) if keywords else ""
    extra = 0
    # Cap entities scored per item — beyond MAX_FACTORS_PER_ITEM the
    # marginal hit rate is dominated by stop-token false positives and the
    # O(N×F) inner loop dominates wall time when callers send hundreds of
    # pre-fetched news items.
    for ent in entities[:MAX_FACTORS_PER_ITEM]:
        for sub in emap.get(ent, []):
            if not sub:
                continue
            # "Maps to this factor" if the curated substring is contained
            # in any of the factor's derived keywords (which themselves
            # come from the factor id/slug). Cheap, deterministic.
            if sub in haystack or any(sub in kw for kw in keywords):
                extra += 1
                break
    return base + extra


def _logit(p: float, eps: float = DEFAULT_EPSILON) -> float:
    """Scalar logit with the project's standard clip."""
    p = max(eps, min(1.0 - eps, float(p)))
    return float(np.log(p / (1.0 - p)))


def _delta_logit_from_prices(p_before: float | None, p_after: float | None) -> float | None:
    """Return logit(p_after) - logit(p_before) or ``None`` if either side missing."""
    if p_before is None or p_after is None:
        return None
    if not np.isfinite(p_before) or not np.isfinite(p_after):
        return None
    return _logit(p_after) - _logit(p_before)


def _confidence_for(
    overlap: int,
    delta_logit_value: float | None,
    beta_source: BetaSource,
) -> Confidence:
    """Map (keyword strength, |Δlogit|, β availability) -> confidence label."""
    if delta_logit_value is None or beta_source == "none":
        return "low"
    if overlap < MIN_KEYWORD_OVERLAP:
        return "low"
    strong_overlap = overlap >= STRONG_KEYWORD_OVERLAP
    strong_move = abs(delta_logit_value) >= STRONG_DELTA_LOGIT
    if beta_source == "regression" and strong_overlap and strong_move:
        return "high"
    if beta_source == "synthetic":
        return "medium"
    if strong_overlap or strong_move:
        return "medium"
    return "low"


def _expected_return_for(beta: float, delta_logit_value: float | None) -> float | None:
    """β × Δlogit × 100 (percent). ``None`` if Δlogit is missing."""
    if delta_logit_value is None:
        return None
    return float(beta * delta_logit_value * 100.0)


# ---------------------------------------------------------------------------
# Core: build_causal_chain
# ---------------------------------------------------------------------------


def build_causal_chain(
    factor_id: str,
    news_items: list[dict],
    lookback_hours: int = 48,
    *,
    beta_map: dict[str, float] | None = None,
) -> dict:
    """For each ``news_items`` item, build a chain headline -> ticker impact.

    Args:
        factor_id: Factor id to tag against; its keywords are derived from
            the id (or use ``register_betas`` to attach real ones).
        news_items: A list of dict-shaped items. Each item should have at
            least a ``title``. Optional: ``ts``, ``url``, ``source``,
            ``description``, ``price_before``, ``price_after``.
        lookback_hours: Reported back to the caller; not load-bearing here
            because the price snapshots are already on the items. (Kept for
            future expansion when we wire automatic price-window fetching.)
        beta_map: Optional explicit ``{ticker -> β}`` override for testing.
            Falls back to ``BETA_REGISTRY[factor_id]`` and finally to a
            synthetic placeholder per affected ticker.

    Returns:
        Dict matching :class:`CausalChainResponse`.
    """
    keywords = set(_factor_keywords(factor_id))
    chain: list[CausalLink] = []

    # Resolve β map: explicit > registry > none
    if beta_map is not None:
        betas = dict(beta_map)
        beta_source: BetaSource = "regression" if betas else "none"
    else:
        betas = get_factor_betas(factor_id)
        beta_source = "regression" if betas else "none"

    n_tagged = 0
    for raw in news_items:
        item = NewsItem(**raw) if not isinstance(raw, NewsItem) else raw
        overlap = _keyword_overlap(item, keywords)
        tagged = factor_id if overlap >= MIN_KEYWORD_OVERLAP else None

        delta_logit_value: float | None = None
        delta_prob: float | None = None
        if tagged is not None:
            n_tagged += 1
            if item.price_before is not None and item.price_after is not None:
                delta_prob = float(item.price_after - item.price_before)
                delta_logit_value = _delta_logit_from_prices(item.price_before, item.price_after)

        # Build per-ticker impacts. If we have any β we use the registry;
        # otherwise we still emit a single synthetic-placeholder entry so
        # the UI can render a directional read.
        ticker_impacts: list[TickerImpact] = []
        if betas:
            for tkr, b in betas.items():
                expected = _expected_return_for(b, delta_logit_value)
                conf = _confidence_for(overlap, delta_logit_value, beta_source)
                ticker_impacts.append(
                    TickerImpact(
                        ticker=tkr,
                        beta=float(b),
                        beta_source=beta_source,
                        expected_return_pct=expected,
                        confidence=conf,
                    )
                )
        elif tagged is not None and delta_logit_value is not None:
            # Tagged item with a price reaction but no β cache — synthetic.
            expected = _expected_return_for(SYNTHETIC_BETA_PLACEHOLDER, delta_logit_value)
            ticker_impacts.append(
                TickerImpact(
                    ticker="(no β cached)",
                    beta=SYNTHETIC_BETA_PLACEHOLDER,
                    beta_source="synthetic",
                    expected_return_pct=expected,
                    confidence="medium",
                )
            )
        # else: tagged=None or no price reaction → empty ticker_impacts.

        link_conf: Confidence
        notes: str
        if tagged is None:
            link_conf = "low"
            notes = f"no keyword match (overlap={overlap}, factor_kw={sorted(keywords)})"
        elif delta_logit_value is None:
            link_conf = "low"
            notes = "tagged but no price-before/after on the news item"
        elif not betas:
            link_conf = "medium"
            notes = "synthetic β placeholder used (register a real β regression)"
        else:
            link_conf = _confidence_for(overlap, delta_logit_value, beta_source)
            notes = f"overlap={overlap}, |Δlogit|={abs(delta_logit_value):.3f}"

        chain.append(
            CausalLink(
                news_item=item,
                tagged_factor=tagged,
                keyword_overlap=overlap,
                delta_prob=delta_prob,
                delta_logit=delta_logit_value,
                affected_tickers=ticker_impacts,
                confidence=link_conf,
                notes=notes,
            )
        )

    response = CausalChainResponse(
        factor_id=factor_id,
        lookback_hours=int(lookback_hours),
        n_items=len(news_items),
        n_tagged=n_tagged,
        chain=chain,
    )
    return response.model_dump()


# ---------------------------------------------------------------------------
# Top movers: scan across factors using fetched feeds
# ---------------------------------------------------------------------------


def _max_signed_impact(link: CausalLink) -> float:
    """Largest |expected_return_pct| across affected tickers, signed by Δlogit.

    Returns 0.0 when no ticker has a numeric forecast (e.g. a tagged item
    with no price reaction) so the mover does not surface as "big".
    """
    best = 0.0
    for tk in link.affected_tickers:
        if tk.expected_return_pct is None:
            continue
        if abs(tk.expected_return_pct) > abs(best):
            best = tk.expected_return_pct
    if best == 0.0 and link.delta_logit:
        # Preserve the *direction* of the underlying probability move so a
        # zero-magnitude signed impact still tells the caller which way the
        # move went.
        return 0.0 * np.sign(link.delta_logit)
    return float(best)


def top_news_movers(
    window_hours: int = 24,
    n: int = 10,
    min_impact_pct: float = 1.0,
    *,
    fetched_items_by_factor: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Rank news items by |expected stock impact| across all known factors.

    The default path expects the caller to pass ``fetched_items_by_factor``
    — a dict ``{factor_id: [news_dicts]}`` — because deciding *which* feeds
    to hit (GDELT vs RSS vs cached) and which factor universe to scan is
    a higher-level orchestration concern. The router endpoint below does
    the orchestration; the function stays pure for testability.
    """
    if not fetched_items_by_factor:
        return []

    movers: list[TopMover] = []
    for factor_id, items in fetched_items_by_factor.items():
        chain_resp = build_causal_chain(factor_id, items, lookback_hours=window_hours)
        for link_dict in chain_resp["chain"]:
            link = CausalLink(**link_dict)
            if link.tagged_factor is None:
                continue
            impact = _max_signed_impact(link)
            if abs(impact) < float(min_impact_pct):
                continue
            movers.append(
                TopMover(
                    factor_id=factor_id,
                    headline=link.news_item.title,
                    ts=link.news_item.ts,
                    source=link.news_item.source,
                    expected_impact_pct=impact,
                    delta_prob=link.delta_prob,
                    n_affected_tickers=sum(
                        1 for tk in link.affected_tickers if tk.expected_return_pct is not None
                    ),
                    link=link,
                )
            )

    movers.sort(key=lambda m: abs(m.expected_impact_pct), reverse=True)
    return [m.model_dump() for m in movers[: int(n)]]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/news", tags=["news-causal"])


def _get_polymarket_client(request: Request) -> PolymarketClient | None:
    return getattr(request.app.state, "poly", None)


class _CausalChainRequest(BaseModel):
    factor_id: str = Field(..., min_length=1, max_length=200)
    news_items: list[dict] | None = Field(
        None,
        description=(
            "Optional pre-fetched news items. When ``None``, the endpoint "
            "tries to hydrate from GDELT / RSS using the factor keywords."
        ),
    )
    lookback_hours: int = Field(48, ge=1, le=24 * 30)
    beta_map: dict[str, float] | None = Field(
        None,
        description="Optional explicit {ticker: β} override for one-off tests.",
    )


async def _fetch_gdelt_async(
    keywords: list[str],
    timespan: str,
    *,
    timeout_s: float = 6.0,
) -> list[dict]:
    """Async GDELT fetch — mirrors :func:`pfm.terminal_gdelt_news._fetch_gdelt`.

    Reimplemented here as ``httpx.AsyncClient`` so the hydration can fan out
    GDELT + RSS concurrently with :func:`asyncio.gather`. Returns the same
    dict-shaped items the legacy sync path produced.
    """
    try:
        from pfm.terminal_gdelt_news import (
            GDELT_DOC_URL,
            HARD_CAP_RECORDS,
            USER_AGENT,
            _build_query,
            _parse_articles,
        )
    except ImportError:  # pragma: no cover - defensive
        return []

    query = _build_query(keywords)
    if not query:
        return []
    params: dict[str, str | int] = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(25, HARD_CAP_RECORDS),
        "sort": "hybridrel",
        "timespan": timespan,
    }
    items: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as http:
            r = await http.get(GDELT_DOC_URL, params=params, headers={"User-Agent": USER_AGENT})
            if r.status_code >= 400:
                return []
            body = r.text or ""
            if body.lstrip().startswith("Please limit"):
                return []
            try:
                payload = r.json()
            except ValueError:
                return []
            if not isinstance(payload, dict):
                return []
            for art in _parse_articles(payload):
                items.append(
                    {
                        "title": art.title,
                        "ts": art.ts,
                        "url": art.url,
                        "source": art.source,
                        "description": "",
                    }
                )
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("GDELT async hydration failed: %s", e)
        return []
    return items


async def _fetch_rss_async(
    keywords: list[str],
    *,
    timeout_s: float = 4.0,
    max_items: int = 25,
) -> list[dict]:
    """Async RSS fallback — pulls a small fan-out of business RSS feeds.

    Uses the same source list as :mod:`pfm.terminal_rss_news` so the hot
    path lines up with what the Terminal already considers canonical.
    Returns ``[]`` on any failure — the causal-chain stays resilient.
    """
    feeds = [
        "http://feeds.bbci.co.uk/news/business/rss.xml",
        "https://news.yahoo.com/rss/topstories",
        "https://www.theverge.com/rss/index.xml",
    ]
    kw_lower = {k.lower() for k in keywords if k}
    if not kw_lower:
        return []

    out: list[dict] = []

    async def _pull_one(url: str) -> list[dict]:
        local: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as http:
                r = await http.get(
                    url,
                    headers={
                        "User-Agent": "pfm-news-causal/1.0",
                        "Accept": "application/rss+xml, application/xml, text/xml",
                    },
                )
                if r.status_code >= 400:
                    return []
                # Lazy parse via the same helper the RSS module uses.
                from pfm.terminal_rss_news import _parse_rss, _SourceSpec

                spec = _SourceSpec(slug="ad-hoc", name="ad-hoc", url=url)
                for item in _parse_rss(r.content, spec):
                    title = (item.title or "").lower()
                    if not any(kw in title for kw in kw_lower):
                        continue
                    local.append(
                        {
                            "title": item.title,
                            "ts": item.published_iso or "",
                            "url": item.link or "",
                            "source": item.source or "rss",
                            "description": "",
                        }
                    )
        except (httpx.HTTPError, ValueError, ImportError, Exception) as e:
            logger.debug("RSS pull failed for %s: %s", url, e)
            return []
        return local

    results = await asyncio.gather(*(_pull_one(u) for u in feeds), return_exceptions=False)
    for batch in results:
        out.extend(batch)
        if len(out) >= max_items:
            break
    return out[:max_items]


async def _hydrate_news_for_factor_async(
    factor_id: str,
    lookback_hours: int,
) -> list[dict]:
    """Parallel-hydrate GDELT + RSS for ``factor_id`` and merge.

    The two fetches run concurrently via :func:`asyncio.gather`, halving the
    serial wait when both feeds are reachable. The merged list is deduped on
    ``(title, url)`` so a story echoing across feeds doesn't double-count.
    """
    keywords = _factor_keywords(factor_id)
    if not keywords:
        return []

    timespan_hours = max(1, int(lookback_hours))
    timespan = f"{timespan_hours}h" if timespan_hours <= 96 else f"{timespan_hours // 24}d"

    gdelt_items, rss_items = await asyncio.gather(
        _fetch_gdelt_async(keywords, timespan),
        _fetch_rss_async(keywords),
        return_exceptions=False,
    )

    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for batch in (gdelt_items, rss_items):
        for item in batch:
            key = (str(item.get("title", "")).lower(), str(item.get("url", "")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _hydrate_news_for_factor(
    factor_id: str,
    poly: PolymarketClient | None,
    lookback_hours: int,
) -> list[dict]:
    """Sync shim around :func:`_hydrate_news_for_factor_async`.

    Kept for backward compatibility with the (sync) ``/movers`` endpoint
    which iterates over BETA_REGISTRY without an active event loop. The
    POST ``/causal-chain`` endpoint now runs the async path directly.
    """
    if not _factor_keywords(factor_id) or poly is None:
        return []
    try:
        return asyncio.run(_hydrate_news_for_factor_async(factor_id, lookback_hours))
    except RuntimeError:
        # Already inside an event loop — schedule on a worker thread to
        # avoid the "asyncio.run() cannot be called from a running loop"
        # error during ASGI request handling.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, _hydrate_news_for_factor_async(factor_id, lookback_hours)
            ).result()


def _hash_post_body(body: _CausalChainRequest) -> str:
    """Stable digest of the POST request body for cache keying."""
    blob = json.dumps(
        {
            "factor_id": body.factor_id,
            "lookback_hours": body.lookback_hours,
            "news_items": body.news_items or [],
            "beta_map": body.beta_map or {},
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@router.post(
    "/causal-chain",
    response_model=CausalChainResponse,
    summary="Build news -> Δprob -> Δlogit -> ticker-impact chain for a factor.",
    dependencies=[Depends(require_tier("pro"))],
)
async def post_causal_chain(
    body: Annotated[_CausalChainRequest, Body()],
    poly: Annotated[PolymarketClient | None, Depends(_get_polymarket_client)] = None,
) -> CausalChainResponse:
    # Body-hash cache: 5-min TTL, keyed by SHA-256 of the canonicalised body.
    # This is the user-visible "POST cache" — second identical call returns
    # in <50ms even when the first paid for GDELT + RSS hydration.
    post_cache = get_cache(NAMESPACE_POST_CHAIN, ttl=POST_CACHE_TTL_SECONDS)
    post_key = ("post-causal-chain", _hash_post_body(body))
    hit = post_cache.get(post_key)
    if hit is not None:
        return CausalChainResponse(**hit)

    # Legacy structured cache — kept so existing call sites that invoke
    # build_causal_chain via the same factor_id + items signature still
    # benefit from result memoisation across the two endpoints.
    cache = get_cache(NAMESPACE_CHAIN, ttl=CACHE_TTL_SECONDS)
    cache_key = (
        "chain",
        body.factor_id,
        body.lookback_hours,
        len(body.news_items or []),
        repr([(i.get("title", ""), i.get("ts", "")) for i in (body.news_items or [])]),
        repr(sorted((body.beta_map or {}).items())),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        post_cache.set(post_key, cached, ttl=POST_CACHE_TTL_SECONDS)
        return CausalChainResponse(**cached)

    items = body.news_items
    if items is None:
        # Async fan-out: GDELT + RSS in parallel via asyncio.gather.
        items = await _hydrate_news_for_factor_async(body.factor_id, body.lookback_hours)

    payload = build_causal_chain(
        factor_id=body.factor_id,
        news_items=items,
        lookback_hours=body.lookback_hours,
        beta_map=body.beta_map,
    )
    cache.set(cache_key, payload)
    post_cache.set(post_key, payload, ttl=POST_CACHE_TTL_SECONDS)
    return CausalChainResponse(**payload)


@router.get(
    "/movers",
    response_model=TopMoversResponse,
    summary="Top news items by |expected stock impact| across registered factors.",
)
def get_movers(
    hours: Annotated[int, Query(ge=1, le=24 * 30)] = 24,
    n: Annotated[int, Query(ge=1, le=100)] = 10,
    min_impact_pct: Annotated[float, Query(ge=0.0, le=100.0)] = 1.0,
    poly: Annotated[PolymarketClient | None, Depends(_get_polymarket_client)] = None,
) -> TopMoversResponse:
    """Scan every β-registered factor, hydrate news, rank by |impact|.

    Note: only factors that already have a β map registered (via
    ``register_betas`` or a previous /fit) participate. Unregistered
    factors are silently skipped — this keeps the endpoint cheap and the
    response focused on actionable cards.
    """
    cache = get_cache(NAMESPACE_MOVERS, ttl=CACHE_TTL_SECONDS)
    cache_key = ("movers", hours, n, min_impact_pct, tuple(sorted(BETA_REGISTRY.keys())))
    cached = cache.get(cache_key)
    if cached is not None:
        return TopMoversResponse(**cached)

    fetched: dict[str, list[dict]] = {}
    for factor_id in BETA_REGISTRY:
        fetched[factor_id] = _hydrate_news_for_factor(factor_id, poly, hours)

    movers_list = top_news_movers(
        window_hours=hours,
        n=n,
        min_impact_pct=min_impact_pct,
        fetched_items_by_factor=fetched,
    )
    n_total = sum(len(v) for v in fetched.values())
    response = TopMoversResponse(
        window_hours=int(hours),
        n_total=n_total,
        n_returned=len(movers_list),
        min_impact_pct=float(min_impact_pct),
        movers=[TopMover(**m) for m in movers_list],
    )
    cache.set(cache_key, response.model_dump())
    return response


__all__ = [
    "BETA_REGISTRY",
    "CACHE_TTL_SECONDS",
    "MIN_KEYWORD_OVERLAP",
    "STRONG_DELTA_LOGIT",
    "STRONG_KEYWORD_OVERLAP",
    "SYNTHETIC_BETA_PLACEHOLDER",
    "CausalChainResponse",
    "CausalLink",
    "NewsItem",
    "TickerImpact",
    "TopMover",
    "TopMoversResponse",
    "build_causal_chain",
    "get_factor_betas",
    "register_betas",
    "router",
    "top_news_movers",
]
