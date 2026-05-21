"""Jump-to-news mapping for a Polymarket slug.

Inverse of ``/terminal/news-impact``. There the pipeline starts from
articles and asks "did the market react?"; here it starts from the
**price series itself**, detects statistically-significant jumps, and
for each jump looks back into a [-2h, +1h] window for GDELT articles
that might explain it.

Why both: an article-first view misses jumps that had no news coverage
in GDELT (information asymmetry, insider flow, or just an uncovered
event). A jump-first view surfaces those as **unexplained** rows that
deserve a human read — they are the rows a quant analyst actually wants
to look at first.

Algorithm (jump detection)
--------------------------
Operate on Δlogit, not raw Δp. A 5pp move at p=0.5 is mild but the
same 5pp move at p=0.05 is a doubling of odds — Δlogit treats them
symmetrically and matches how the market itself prices information.

For each consecutive pair of hourly observations we compute::

    Δlogit_t = logit(p_t) - logit(p_{t-1})

We flag a jump when BOTH conditions hold (logical AND):
    1. ``|Δlogit_t| >= k * MAD_24h`` (default k=2.5, robust to outliers)
    2. ``|Δp_t| >= min_jump_pp / 100`` (absolute floor so a tiny series
        doesn't drown the user in micro-jumps)

Using AND (not OR) gives much better signal: the floor kills
white-noise outliers that pass the z-test purely because MAD is
tiny, and the z-test kills market-wide regime shifts where an
absolute 3pp threshold would be spammed.

The 24h rolling MAD is robust to a single fat-tailed jump and avoids
the classic problem of σ-based detectors that get inflated by the
very outlier you're trying to find.

Article matching
----------------
For each jump at time ``t`` we collect articles with ``ts ∈ [t-2h, t+1h]``.
The asymmetric window reflects market microstructure: news typically
*precedes* the move (2h lookback), but on prediction markets the market
sometimes leads the wires (~1h leeway). Each article is scored by
``score_relevance`` against the market question; we keep the top-K by
score per jump.

A jump with zero matching articles is returned with ``explained=False``
so the frontend can render it in a different colour ("info-asymmetric
jump — no public news"). This is the row most worth investigating.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Literal

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal.news import _fetch_hn, _fetch_reddit
from pfm.terminal.news_dedupe import NewsItem as _DedupeNewsItem
from pfm.terminal.news_dedupe import dedupe_news
from pfm.terminal.news_impact import (
    _fetch_hourly_prices,
    _hourly_log_return_sigma,  # noqa: F401  (kept for parity / future use)
)
from pfm.terminal.news_relevance import (
    RELEVANCE_MIN,
    build_phrase_query,
    build_terms,
    score_relevance,
)
from pfm.terminal.rss_news import (
    SOURCES as _RSS_SOURCES,
)
from pfm.terminal.rss_news import (
    _fetch_sources_parallel as _rss_fetch_parallel,
)
from pfm.terminal.sentiment_nlp import aggregate_sentiment, score_headline
from pfm.terminal_export import respond as _export_respond
from pfm.terminal_gdelt_news import GDELTArticle, _build_query, _fetch_gdelt
from pfm.terminal_news import MAX_KEYWORDS, extract_keywords

logger = logging.getLogger(__name__)

DEFAULT_DAYS: int = 14
MIN_DAYS: int = 1
MAX_DAYS: int = 90
# Defaults tightened 2026-05-16 — user feedback: "solo saltos bruscos, no
# spamear". A 3pp / 2.5σ jump on a quiet market was triggering 30+ rows per
# slug. 5pp + 3σ keeps only moves that are both economically meaningful
# AND statistically loud. Callers can still override via query params.
DEFAULT_MIN_JUMP_PP: float = 5.0  # absolute Δp floor in percentage points
DEFAULT_MAD_K: float = 3.0  # threshold multiplier on rolling MAD
ROLLING_HOURS: int = 24  # window for the rolling MAD
LOOKBACK_HOURS: int = 2  # news window left of jump
LOOKAHEAD_HOURS: int = 1  # news window right of jump
TOP_K_ARTICLES_PER_JUMP: int = 5
MAX_JUMPS_RETURNED: int = 60  # hard cap so the response stays small
EPS_LOGIT: float = 0.005  # logit clipping to keep values finite

# Cache mirrors news_impact's TTLs so a slug's news + jumps go stale together.
CACHE_TTL_SECONDS: int = 600
_CACHE = get_cache("terminal_jumps", ttl=CACHE_TTL_SECONDS)


# --- schemas ----------------------------------------------------------------


Direction = Literal["up", "down", "flat"]


class JumpArticle(BaseModel):
    """One GDELT article matched to a jump's [-2h, +1h] window."""

    ts_iso: str = Field(..., description="ISO-8601 UTC of the article.")
    seconds_from_jump: int = Field(
        ..., description="Article ts minus jump ts, in seconds. Negative = before."
    )
    headline: str
    source: str
    url: str | None = None
    tone: float = 0.0
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    sentiment_score: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Hybrid VADER + financial-lexicon score, signed [-1,+1].",
    )
    sentiment_label: Literal["positive", "negative", "neutral"] = Field(
        "neutral",
        description="Bucketed sentiment using a ±0.15 deadband.",
    )


class Jump(BaseModel):
    """One detected price-jump with its (possibly empty) news context."""

    ts_iso: str = Field(..., description="ISO-8601 UTC of the jump.")
    price_before: float = Field(..., description="Price at t-1 (just before the jump).")
    price_after: float = Field(..., description="Price at t (after the jump).")
    delta_pp: float = Field(..., description="(price_after - price_before) * 100, signed.")
    delta_logit: float = Field(..., description="logit(price_after) - logit(price_before), signed.")
    z_score: float = Field(
        ..., description="|delta_logit| divided by rolling 24h MAD of |delta_logit|."
    )
    direction: Direction
    explained: bool = Field(
        ...,
        description="True iff ≥1 GDELT article with relevance≥floor in the window.",
    )
    n_articles: int = Field(..., description="GDELT articles in [-2h, +1h], any relevance.")
    top_articles: list[JumpArticle] = Field(
        default_factory=list,
        description="Top-K articles by relevance × time-proximity.",
    )
    news_sentiment_score: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Mean sentiment across matched articles, in [-1, +1].",
    )
    news_sentiment_label: Literal["positive", "negative", "neutral"] = "neutral"
    sentiment_alignment: Literal["agrees", "disagrees", "neutral"] = Field(
        "neutral",
        description=(
            "Does aggregate news sentiment match the price jump direction? "
            "'agrees' = same sign; 'disagrees' = opposite (interesting!); "
            "'neutral' = either side too close to 0 to call."
        ),
    )


class TerminalJumpsResponse(BaseModel):
    slug: str
    days: int
    threshold_mad_k: float
    threshold_min_jump_pp: float
    n_jumps: int
    n_explained: int
    explained_pct: float = Field(..., description="100 * n_explained / max(n_jumps, 1).")
    jumps: list[Jump]
    interpretation: str


# --- helpers ----------------------------------------------------------------


def _logit(p: float) -> float:
    """Numerically-safe logit with the module's EPS_LOGIT clip.

    Clipping is symmetric and explicit (not the default ε from
    :mod:`pfm.model`) because we want the jump magnitudes to be
    comparable across different markets.
    """
    p = max(min(float(p), 1.0 - EPS_LOGIT), EPS_LOGIT)
    return float(np.log(p / (1.0 - p)))


def _direction(x: float) -> Direction:
    if x > 1e-6:
        return "up"
    if x < -1e-6:
        return "down"
    return "flat"


def detect_jumps(
    prices: pd.Series,
    *,
    mad_k: float = DEFAULT_MAD_K,
    min_jump_pp: float = DEFAULT_MIN_JUMP_PP,
    rolling_hours: int = ROLLING_HOURS,
) -> list[dict]:
    """Detect significant jumps in a UTC-indexed hourly probability series.

    Returns a list of dicts (one per jump) with the fields needed to
    build :class:`Jump` after the news join. Pure function — no IO,
    deterministic, fast — so the unit tests below cover it directly.

    Args:
        prices: hourly UTC-indexed series of probabilities in [0, 1].
            Sparse / non-uniform indexing is tolerated; we use successive
            observations rather than a fixed grid.
        mad_k: jump fires when ``|Δlogit| >= mad_k × rolling_mad``.
        min_jump_pp: absolute floor on ``|Δp| * 100`` so micro-noise on
            quiet markets doesn't dominate the list.
        rolling_hours: window for the trailing MAD estimate.

    Returns:
        ``[{ts, price_before, price_after, delta_pp, delta_logit,
            mad, z_score}, ...]`` sorted by descending ``|delta_logit|``.
    """
    if prices.empty or len(prices) < 3:
        return []
    s = prices.sort_index().astype(float)
    # Compute successive Δlogit on the *raw* observations. This avoids
    # introducing NaNs from a fixed-grid resample on irregular series.
    logits = s.apply(_logit)
    d_logit = logits.diff()
    d_price = s.diff()

    # Rolling MAD on |Δlogit|. We use the trailing rolling_hours of
    # observations as a count cap (not a literal time-window slice)
    # because the series cadence is hourly by construction. ``min_periods``
    # ≥ 6 ensures the MAD has something to chew on early in the series.
    abs_dl = d_logit.abs()
    mad = abs_dl.rolling(rolling_hours, min_periods=min(6, rolling_hours)).apply(
        lambda x: float(np.nanmedian(np.abs(x - np.nanmedian(x)))), raw=True
    )
    # MAD can be 0 (a flat stretch); guard against div-by-zero by using
    # a tiny epsilon. The min_jump_pp floor catches degenerate cases.
    mad_eff = mad.fillna(0.0).clip(lower=1e-6)

    jumps: list[dict] = []
    for ts, dl in d_logit.items():
        if pd.isna(dl):
            continue
        dp = float(d_price.loc[ts])
        if pd.isna(dp):
            continue
        m = float(mad_eff.loc[ts])
        z = abs(dl) / m if m > 0 else 0.0
        # Both gates must pass: z-score kills market-wide drift, the pp
        # floor kills noise. See module docstring.
        passes_z = z >= mad_k
        passes_floor = abs(dp) * 100.0 >= min_jump_pp
        if not (passes_z and passes_floor):
            continue
        # Resolve price_before from the row at t-1 — use the actual prior
        # observation in the (possibly non-uniform) series.
        prev_idx = logits.index[logits.index < ts]
        if len(prev_idx) == 0:
            continue
        p_before = float(s.loc[prev_idx[-1]])
        p_after = float(s.loc[ts])
        jumps.append(
            {
                "ts": ts,
                "price_before": p_before,
                "price_after": p_after,
                "delta_pp": round((p_after - p_before) * 100.0, 3),
                "delta_logit": float(dl),
                "mad": float(m),
                "z_score": float(z),
            }
        )

    jumps.sort(key=lambda j: -abs(j["delta_logit"]))
    return jumps[:MAX_JUMPS_RETURNED]


def _to_gdelt_shape(
    *,
    ts: str,
    title: str,
    source: str,
    url: str | None,
    country: str = "us",
    language: str = "english",
    tone: float = 0.0,
) -> GDELTArticle | None:
    """Wrap a Reddit/HN/RSS item into the GDELT envelope used by the matcher.

    Returns ``None`` if the timestamp can't be parsed — the downstream
    window match would silently drop it anyway, so we filter here.
    """
    if not ts or not title:
        return None
    try:
        # Validate parseability; the matcher pd.Timestamps it later.
        pd.Timestamp(ts)
    except (ValueError, TypeError):
        return None
    try:
        return GDELTArticle(
            ts=str(ts),
            title=str(title),
            source=str(source) or "unknown",
            country=country,
            language=language,
            tone=float(tone) or 0.0,
            url=url or f"https://{source}",
        )
    except Exception:  # pydantic validation can blow up on weird inputs
        return None


def _gather_all_news(
    http_client,
    query: str,
    timespan: str,
) -> list[GDELTArticle]:
    """Fan-out fetch from every news source we have and merge into one list.

    Sources (best-effort — each is wrapped so a single failure can't take
    the whole gather down):

    1. **GDELT 2.0** ``DOC`` API — global, multilingual, tone-scored.
    2. **Reddit** search — community-discussion signal.
    3. **HN** Algolia — tech / finance leaning frontpage.
    4. **RSS feeds** — curated set (BBC, Reuters, MarketWatch, etc.) from
       :mod:`pfm.terminal.rss_news`. Cast a wide net so even when GDELT
       is throttled we still surface explained jumps.

    Returns a single list of GDELT-shaped items (so the existing scorer
    + window matcher work unchanged). Deduplication is by ``url``
    when present, else by ``(source, title)``.
    """
    out: list[GDELTArticle] = []

    # 1. GDELT
    try:
        gd = _fetch_gdelt(http_client, query, 100, timespan)
        if gd:
            out.extend(gd)
    except Exception as e:
        logger.debug("jumps: gdelt fetch failed: %s", e)

    # 2. Reddit
    try:
        reddit_items, _ok = _fetch_reddit(http_client, query, 40)
        for it in reddit_items or []:
            wrapped = _to_gdelt_shape(
                ts=it.ts,
                title=it.title,
                source=f"reddit:{it.source}" if hasattr(it, "source") else "reddit",
                url=it.url,
            )
            if wrapped is not None:
                out.append(wrapped)
    except Exception as e:
        logger.debug("jumps: reddit fetch failed: %s", e)

    # 3. Hacker News
    try:
        hn_items, _ok = _fetch_hn(http_client, query, 40)
        for it in hn_items or []:
            wrapped = _to_gdelt_shape(
                ts=it.ts,
                title=it.title,
                source="hn",
                url=it.url,
            )
            if wrapped is not None:
                out.append(wrapped)
    except Exception as e:
        logger.debug("jumps: hn fetch failed: %s", e)

    # 4. RSS feeds — these don't take a query; they're broad-coverage,
    #    we rely on the relevance scorer to filter.
    try:
        rss_results = _rss_fetch_parallel(http_client, _RSS_SOURCES)
        for _src, items, _err in rss_results:
            for it in items or []:
                wrapped = _to_gdelt_shape(
                    ts=it.pub_date,
                    title=it.title,
                    source=it.source,
                    url=it.link,
                )
                if wrapped is not None:
                    out.append(wrapped)
    except Exception as e:
        logger.debug("jumps: rss fetch failed: %s", e)

    # Dedup — same article often appears in multiple feeds, and the same
    # headline reappears across GDELT/Reddit/HN/RSS with minor wording
    # shifts. Run SimHash-based dedupe (T20) on top of the URL-key pass so
    # both exact-URL duplicates and near-duplicate titles collapse.
    seen: set[str] = set()
    url_deduped: list[GDELTArticle] = []
    for art in out:
        key = (art.url or "").strip() or f"{art.source}|{art.title}"
        if key in seen:
            continue
        seen.add(key)
        url_deduped.append(art)

    # SimHash-aware cross-source dedupe at the merge boundary. We adapt
    # GDELTArticle into the dedupe module's NewsItem (which keys on title)
    # and map the survivors back to the original GDELTArticle instances by
    # url so we preserve the response shape exactly.
    by_url: dict[str, GDELTArticle] = {art.url: art for art in url_deduped}
    dedupe_items: list[_DedupeNewsItem] = []
    for art in url_deduped:
        try:
            published = pd.Timestamp(art.ts).to_pydatetime()
        except (ValueError, TypeError):
            # Unparseable timestamps were filtered upstream; defensive.
            continue
        dedupe_items.append(
            _DedupeNewsItem(
                title=art.title,
                url=art.url,
                source=art.source,
                published_at=published,
                tone=art.tone,
            )
        )
    merged = dedupe_news(dedupe_items, threshold_bits=4)
    deduped: list[GDELTArticle] = [by_url[m.url] for m in merged if m.url in by_url]
    logger.info(
        "jumps: news gather got %d total / %d after dedup (gdelt+reddit+hn+rss)",
        len(out),
        len(deduped),
    )
    return deduped


def _articles_for_jump_with_floor(
    jump_ts: pd.Timestamp,
    scored_articles: list[tuple[GDELTArticle, float, list[str]]],
    *,
    market_start_ts: pd.Timestamp | None = None,
    lookback_h: int = LOOKBACK_HOURS,
    lookahead_h: int = LOOKAHEAD_HOURS,
    top_k: int = TOP_K_ARTICLES_PER_JUMP,
) -> tuple[list[JumpArticle], int]:
    """Like :func:`_articles_for_jump` but with a hard left-edge floor.

    Articles with ``ts < market_start_ts`` are excluded entirely — the
    market did not yet exist for the news to react to, so attributing
    an old wire story to a market that launched after it is misleading.
    This honours the user's rule "estrictamente no pongas news before el
    evento empieza".
    """
    lo = jump_ts - pd.Timedelta(hours=lookback_h)
    if market_start_ts is not None and market_start_ts > lo:
        # Clamp the window's left edge to the market's creation date.
        lo = market_start_ts
    hi = jump_ts + pd.Timedelta(hours=lookahead_h)
    picked: list[tuple[float, JumpArticle]] = []
    n_in_window = 0
    for art, score, matched in scored_articles:
        try:
            art_ts = pd.Timestamp(art.ts)
            if art_ts.tzinfo is None:
                art_ts = art_ts.tz_localize("UTC")
            else:
                art_ts = art_ts.tz_convert("UTC")
        except (ValueError, TypeError):
            continue
        if art_ts < lo or art_ts > hi:
            continue
        # Belt-and-braces hard floor: even if lookback_h would put us
        # ahead of market_start_ts in some future refactor, never emit
        # an article that pre-dates the market itself.
        if market_start_ts is not None and art_ts < market_start_ts:
            continue
        n_in_window += 1
        secs = int((art_ts - jump_ts).total_seconds())
        proximity = float(np.exp(-(abs(secs) / 3600.0) / 2.0))
        rank = score * proximity
        tone = float(getattr(art, "tone", 0.0) or 0.0)
        sent_score, sent_label = score_headline(art.title, external_tone=tone)
        picked.append(
            (
                rank,
                JumpArticle(
                    ts_iso=art.ts,
                    seconds_from_jump=secs,
                    headline=art.title,
                    source=art.source,
                    url=getattr(art, "url", None),
                    tone=tone,
                    relevance_score=round(float(score), 4),
                    matched_terms=list(matched or []),
                    sentiment_score=sent_score,
                    sentiment_label=sent_label,
                ),
            )
        )
    picked.sort(key=lambda r: -r[0])
    return [p[1] for p in picked[:top_k]], n_in_window


def _articles_for_jump(
    jump_ts: pd.Timestamp,
    scored_articles: list[tuple[GDELTArticle, float, list[str]]],
    lookback_h: int = LOOKBACK_HOURS,
    lookahead_h: int = LOOKAHEAD_HOURS,
    top_k: int = TOP_K_ARTICLES_PER_JUMP,
) -> tuple[list[JumpArticle], int]:
    """Pick articles in ``[jump_ts - lookback_h, jump_ts + lookahead_h]``.

    Returns ``(top_k_picked, n_in_window)``.

    Ranking score = ``relevance × proximity_decay``. Proximity decay is
    exp(-(|seconds|/3600) / 2) — articles 2 hours away count for ~37%
    of the same article right at the jump.
    """
    lo = jump_ts - pd.Timedelta(hours=lookback_h)
    hi = jump_ts + pd.Timedelta(hours=lookahead_h)
    picked: list[tuple[float, JumpArticle]] = []
    n_in_window = 0
    for art, score, matched in scored_articles:
        try:
            art_ts = pd.Timestamp(art.ts)
            if art_ts.tzinfo is None:
                art_ts = art_ts.tz_localize("UTC")
            else:
                art_ts = art_ts.tz_convert("UTC")
        except (ValueError, TypeError):
            continue
        if art_ts < lo or art_ts > hi:
            continue
        n_in_window += 1
        secs = int((art_ts - jump_ts).total_seconds())
        proximity = float(np.exp(-(abs(secs) / 3600.0) / 2.0))
        rank = score * proximity
        tone = float(getattr(art, "tone", 0.0) or 0.0)
        # Hybrid sentiment per article: VADER + financial lex + external tone.
        sent_score, sent_label = score_headline(art.title, external_tone=tone)
        picked.append(
            (
                rank,
                JumpArticle(
                    ts_iso=art.ts,
                    seconds_from_jump=secs,
                    headline=art.title,
                    source=art.source,
                    url=getattr(art, "url", None),
                    tone=tone,
                    relevance_score=round(float(score), 4),
                    matched_terms=list(matched or []),
                    sentiment_score=sent_score,
                    sentiment_label=sent_label,
                ),
            )
        )
    picked.sort(key=lambda r: -r[0])
    return [p[1] for p in picked[:top_k]], n_in_window


# --- routing ---------------------------------------------------------------


router = APIRouter(prefix="/terminal", tags=["terminal-jumps"])


def _get_polymarket_client(request: Request) -> PolymarketClient:
    # Lifespan stores the singleton as `app.state.poly` — match
    # news_impact.get_polymarket_client so we share the cached HTTP pool.
    poly: PolymarketClient | None = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


@router.get(
    "/jumps/{slug}",
    response_model=None,
    summary="Detect price-series jumps and attach matching GDELT articles.",
)
async def get_jumps(
    request: Request,
    slug: Annotated[str, Path(min_length=1, max_length=120)],
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
    mad_k: Annotated[float, Query(ge=1.0, le=10.0)] = DEFAULT_MAD_K,
    min_jump_pp: Annotated[float, Query(ge=0.5, le=50.0)] = DEFAULT_MIN_JUMP_PP,
    format: Annotated[Literal["json", "csv", "pdf"], Query()] = "json",
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> TerminalJumpsResponse | FastAPIResponse:
    """For a Polymarket slug, return jumps (∆logit outliers) with matching news.

    A jump fires when ``|Δlogit| ≥ mad_k × rolling_MAD`` OR
    ``|Δp| ≥ min_jump_pp / 100``. For each jump we look ±1h around the
    timestamp in GDELT for articles relevant to the market question.
    Jumps with no matching article are returned with ``explained=False``
    — these are the rows worth a human read.
    """
    cache_key = (slug, int(days), round(mad_k, 2), round(min_jump_pp, 2))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        cached_payload = TerminalJumpsResponse(**cached)
        if format == "json":
            return cached_payload
        return _export_respond(cached_payload, format, filename=f"jumps-{slug}", kind="market")

    # 1. Market metadata
    try:
        meta = poly.get_market_metadata(slug)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"market not found: {e}") from e

    question = meta.question or slug
    keywords = (
        extract_keywords(question, max_n=MAX_KEYWORDS)
        or [t for t in slug.replace("_", "-").split("-") if len(t) >= 3][:MAX_KEYWORDS]
    )
    terms = build_terms(question)
    query = build_phrase_query(terms) or _build_query(keywords) or slug

    # 2. Fetch GDELT + prices in parallel
    timespan = f"{days}d"
    end_ts = pd.Timestamp.utcnow()
    start_ts = end_ts - pd.Timedelta(days=days)
    pad_start_unix = int((start_ts - pd.Timedelta(hours=24)).timestamp())

    raw_articles, prices = await asyncio.gather(
        asyncio.to_thread(_gather_all_news, poly._client, query, timespan),
        asyncio.to_thread(
            _fetch_hourly_prices,
            poly._client,
            poly.clob_url,
            meta.yes_token_id,
            pad_start_unix,
        ),
    )

    # 3. Pre-score every article once so per-jump matching is O(n_articles)
    #    per jump (with cheap scoring) rather than a re-score per pair.
    scored: list[tuple[GDELTArticle, float, list[str]]] = []
    has_terms = bool(terms.anchors or terms.topics)
    for art in raw_articles:
        if has_terms:
            s_score, s_matched = score_relevance(art.title, terms)
            if s_score < RELEVANCE_MIN:
                continue
            scored.append((art, s_score, s_matched))
        else:
            # No anchor terms — fall back to a uniform 0.5 score so every
            # article is still eligible. This matches news_impact's behavior.
            scored.append((art, 0.5, []))

    # 4. Detect jumps from the price series itself
    raw_jumps = detect_jumps(
        prices, mad_k=mad_k, min_jump_pp=min_jump_pp, rolling_hours=ROLLING_HOURS
    )

    # 5. Pair each jump with its window articles.
    # Hard floor: never attribute an article published BEFORE the
    # market itself started trading. Without this, a 2024 wire story
    # could "explain" a 2026 market just because the question text
    # overlapped — misleading the analyst.
    market_start_ts: pd.Timestamp | None = None
    raw_start = getattr(meta, "start_date", None)
    if raw_start:
        try:
            ts_parsed = pd.Timestamp(raw_start)
            market_start_ts = (
                ts_parsed.tz_convert("UTC") if ts_parsed.tzinfo else ts_parsed.tz_localize("UTC")
            )
        except (ValueError, TypeError):
            market_start_ts = None

    jumps_out: list[Jump] = []
    for j in raw_jumps:
        top, n_window = _articles_for_jump_with_floor(
            j["ts"],
            scored,
            market_start_ts=market_start_ts,
        )
        direction = _direction(j["delta_logit"])
        # Aggregate sentiment across the matched articles for this jump.
        sent_mean, sent_label, sent_align = aggregate_sentiment(
            [a.sentiment_score for a in top],
            jump_direction=direction,
        )
        jumps_out.append(
            Jump(
                ts_iso=j["ts"].isoformat().replace("+00:00", "Z"),
                price_before=round(j["price_before"], 4),
                price_after=round(j["price_after"], 4),
                delta_pp=j["delta_pp"],
                delta_logit=round(j["delta_logit"], 4),
                # Cap display z-score: very flat stretches make MAD→0 which
                # blows the divisor up to thousands. Anything above ~50 is
                # already "extreme outlier"; the cap keeps the UI honest.
                z_score=round(min(j["z_score"], 99.0), 2),
                direction=direction,
                explained=len(top) > 0,
                n_articles=n_window,
                top_articles=top,
                news_sentiment_score=sent_mean,
                news_sentiment_label=sent_label,
                sentiment_alignment=sent_align,
            )
        )

    # Sort chronologically for chart overlay (descending magnitude was for
    # the truncation cap inside detect_jumps; here we want time order).
    jumps_out.sort(key=lambda x: x.ts_iso)

    n_jumps = len(jumps_out)
    n_explained = sum(1 for j in jumps_out if j.explained)
    explained_pct = round(100.0 * n_explained / max(n_jumps, 1), 1)
    if n_jumps == 0:
        interpretation = (
            f"No jumps above the threshold ({mad_k}×MAD or {min_jump_pp}pp) in the last {days}d."
        )
    else:
        interpretation = (
            f"{n_jumps} jumps detected · {n_explained} explained by news "
            f"(GDELT + Reddit + HN + RSS) ({explained_pct}%) · "
            f"{n_jumps - n_explained} unexplained ({100 - explained_pct:.1f}%) "
            "— those rows are the ones worth a read."
        )

    payload = TerminalJumpsResponse(
        slug=slug,
        days=int(days),
        threshold_mad_k=float(mad_k),
        threshold_min_jump_pp=float(min_jump_pp),
        n_jumps=n_jumps,
        n_explained=n_explained,
        explained_pct=explained_pct,
        jumps=jumps_out,
        interpretation=interpretation,
    )
    _CACHE.set(cache_key, payload.model_dump(), ttl=CACHE_TTL_SECONDS)
    if format == "json":
        return payload
    return _export_respond(payload, format, filename=f"jumps-{slug}", kind="market")


__all__ = [
    "DEFAULT_MAD_K",
    "DEFAULT_MIN_JUMP_PP",
    "Jump",
    "JumpArticle",
    "TerminalJumpsResponse",
    "detect_jumps",
    "get_jumps",
    "router",
]
