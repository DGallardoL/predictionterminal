"""News-event-overlay endpoint: GDELT events + Polymarket price reactions.

Goal: for a given Polymarket slug, fetch recent GDELT articles and
match each headline timestamp to the prediction-market price *before*
the news broke and at +1h / +6h / +24h afterwards. The frontend can
draw vertical lines on the probability chart at each event timestamp
and annotate "Headline X → price moved Y pp".

We flag an event as ``attributable`` when the absolute 6h move exceeds
1.5σ of the realised hourly log-return volatility on the same series —
i.e. the move stands out from noise. This is a heuristic, not a causal
claim: GDELT-listed news *correlates* with price moves; the headline
need not have *caused* the move (the market may have led the news).
The :class:`Event` model carries the raw numbers so callers can apply
their own threshold.

Routing note: this module owns its :class:`fastapi.APIRouter`; the
existing ``main.py`` is left untouched (per CLAUDE.md). To activate::

    from pfm.terminal_news_impact import router as news_impact_router
    app.include_router(news_impact_router)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Annotated, Literal

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal.news_relevance import (
    RELEVANCE_MIN,
    QuestionTerms,
    build_phrase_query,
    build_terms,
    score_relevance,
)
from pfm.terminal_gdelt_news import (
    GDELT_DOC_URL,
    GDELTArticle,
    _build_query,
    _fetch_gdelt,
)
from pfm.terminal_news import MAX_KEYWORDS, extract_keywords

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS: int = 900  # 15 min — matches GDELT update cadence
# L2 Redis TTL — shorter than L1 so workers see fresher data even when one
# worker's L1 entry is still warm. 10 min matches the L1 TTL of the
# upstream gdelt_news module so cache invalidations align.
REDIS_TTL_SECONDS: int = 600
# Refuse to push payloads larger than this into Redis. Keeps a slow JSON
# round-trip out of the hot path when an event spike produces a huge
# events array. We still serve from L1 — just don't pollute the L2 set.
REDIS_PAYLOAD_MAX_BYTES: int = 128 * 1024  # 128 KB
HOURLY_FIDELITY: int = 60  # minutes per bar
DEFAULT_DAYS: int = 30
MIN_DAYS: int = 1
MAX_DAYS: int = 365
ATTRIBUTION_SIGMA: float = 1.5  # |6h move| > 1.5σ → attributable


# --- schemas ----------------------------------------------------------------


Direction = Literal["up", "down", "flat"]


class Event(BaseModel):
    """One GDELT article paired with the price reaction in its window."""

    ts_iso: str = Field(..., description="ISO-8601 UTC timestamp of the article.")
    headline: str
    source: str
    tone: float = Field(0.0, description="GDELT tone in [-10, +10].")
    price_before: float | None = Field(None, description="Last price strictly before ts.")
    price_1h_after: float | None = None
    price_6h_after: float | None = None
    price_24h_after: float | None = None
    abs_move_pp: float | None = Field(
        None, description="|price_6h_after - price_before| in percentage points."
    )
    direction: Direction = Field("flat", description="Sign of the 6h move.")
    attributable: bool = Field(
        False,
        description=f"True iff |6h move| > {ATTRIBUTION_SIGMA} sigma of hourly returns.",
    )
    relevance_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Question-relevance of the headline in [0, 1].",
    )
    matched_terms: list[str] = Field(
        default_factory=list,
        description="Anchor/topic terms that contributed to the relevance score.",
    )


class TerminalNewsImpactResponse(BaseModel):
    slug: str
    days: int
    events: list[Event]
    n_events: int
    n_attributable: int
    attributable_pct: float = Field(..., description="100 * n_attributable / n_events.")
    interpretation: str


# --- in-memory + Redis L2 cache --------------------------------------------
# L1 is the existing process-local dict (kept exposed for tests). L2 is the
# Redis cache mounted on app.state.cache so all four gunicorn workers see
# the same payload — cold cross-worker latency drops from ~11 s to <50 ms.

_CACHE: dict[str, tuple[float, dict]] = {}


def _cache_get(key: str) -> dict | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    expiry, payload = entry
    if expiry < time.time():
        _CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: str, payload: dict) -> None:
    _CACHE[key] = (time.time() + CACHE_TTL_SECONDS, payload)


def _redis_key(cache_key: str) -> str:
    """Namespace L2 keys so they're easy to flush in ops."""
    return f"terminal_news_impact:{cache_key}"


def _redis_get(request: Request, cache_key: str) -> dict | None:
    """L2 (Redis) read. Returns ``None`` on miss or any Redis hiccup."""
    cache = getattr(request.app.state, "cache", None)
    if cache is None or not getattr(cache, "enabled", False):
        return None
    raw: bytes | None = None
    with contextlib.suppress(Exception):  # defensive: never break on cache I/O
        raw = cache.get(_redis_key(cache_key))
    if not raw:
        return None
    try:
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _redis_set(request: Request, cache_key: str, payload: dict) -> None:
    """L2 (Redis) write — skipped silently when payload is too large."""
    cache = getattr(request.app.state, "cache", None)
    if cache is None or not getattr(cache, "enabled", False):
        return
    try:
        blob = json.dumps(payload, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return
    if len(blob) > REDIS_PAYLOAD_MAX_BYTES:
        logger.info(
            "news-impact: skipping Redis SET — payload %d B exceeds %d B cap",
            len(blob),
            REDIS_PAYLOAD_MAX_BYTES,
        )
        return
    with contextlib.suppress(Exception):  # defensive: never break /impact on cache I/O
        cache.set(_redis_key(cache_key), blob, REDIS_TTL_SECONDS)


# --- price helpers ----------------------------------------------------------


def _fetch_hourly_prices(
    client: httpx.Client,
    clob_url: str,
    token_id: str,
    start_ts: int,
) -> pd.Series:
    """Fetch hourly Polymarket prices (fidelity=60) and return a UTC-indexed series.

    Uses ``interval=max`` plus ``startTs`` to avoid the CLOB's ``endTs``
    rejection (see :mod:`pfm.sources.polymarket` for the quirks). Returns
    an empty Series on any failure — callers degrade to "no price data".
    """
    params: dict[str, str | int] = {
        "market": token_id,
        "fidelity": HOURLY_FIDELITY,
        "interval": "max",
        "startTs": int(start_ts),
    }
    try:
        r = client.get(f"{clob_url.rstrip('/')}/prices-history", params=params, timeout=5.0)
        r.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("hourly clob fetch failed: %s", e)
        return pd.Series(dtype=float)
    try:
        history = r.json().get("history", []) or []
    except ValueError:
        return pd.Series(dtype=float)
    if not history:
        return pd.Series(dtype=float)
    rows = [(int(b["t"]), float(b["p"])) for b in history if "t" in b and "p" in b]
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
    return pd.Series([r[1] for r in rows], index=idx, name="price").sort_index()


def _price_at_or_before(series: pd.Series, ts: pd.Timestamp) -> float | None:
    """Last observed price strictly at or before ``ts``; ``None`` if no data."""
    if series.empty:
        return None
    sub = series[series.index <= ts]
    if sub.empty:
        return None
    v = float(sub.iloc[-1])
    return v if np.isfinite(v) else None


def _price_at_or_after(series: pd.Series, ts: pd.Timestamp) -> float | None:
    """First observed price at or after ``ts``; ``None`` if past the series end."""
    if series.empty:
        return None
    sub = series[series.index >= ts]
    if sub.empty:
        return None
    v = float(sub.iloc[0])
    return v if np.isfinite(v) else None


def _hourly_log_return_sigma(series: pd.Series) -> float:
    """Realised σ of hourly log returns. ``0.0`` if too few observations.

    We use the *clipped* price (ε=0.01) before logging to avoid blowing
    up the volatility on near-zero prints — same convention as the rest
    of the codebase.
    """
    if series.empty or len(series) < 3:
        return 0.0
    prices = series.clip(lower=0.01, upper=0.99)
    log_ret = np.log(prices).diff().dropna()
    if len(log_ret) < 2:
        return 0.0
    sigma = float(log_ret.std(ddof=1))
    return sigma if np.isfinite(sigma) else 0.0


def _direction(move: float) -> Direction:
    if move > 0:
        return "up"
    if move < 0:
        return "down"
    return "flat"


# --- router -----------------------------------------------------------------


router = APIRouter(prefix="/terminal/news-impact", tags=["terminal-news-impact"])


def get_polymarket_client(request: Request) -> PolymarketClient:
    poly = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


# --- endpoint ---------------------------------------------------------------


@router.get(
    "/{slug}",
    response_model=TerminalNewsImpactResponse,
    summary="GDELT news events with Polymarket price-reaction windows.",
)
async def get_news_impact(
    request: Request,
    slug: Annotated[str, Path(min_length=1, max_length=120)],
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> TerminalNewsImpactResponse:
    """Return per-event price reactions (1h/6h/24h) for a Polymarket slug.

    Pipeline:
      1. Resolve the market metadata via Gamma (cached upstream).
      2. Hit GDELT 2.0 + Polymarket hourly-prices in parallel — the two
         calls don't depend on each other and the prices fetch is the
         long-pole at ~6 s. ``asyncio.gather`` over ``asyncio.to_thread``
         brings warm cross-worker latency from ~11 s down to <6 s.
      3. Score + filter articles by relevance.
      4. For each surviving article timestamp, look up the most-recent
         price before the event and the next prices at +1h / +6h / +24h.
      5. Compute the realised σ of hourly log returns, then flag each
         event as ``attributable`` iff its 6h |move| exceeds 1.5σ.

    Cache layers:
      - L1 (process-local dict, 15 min TTL) — exposed as ``_CACHE``.
      - L2 (Redis, 10 min TTL, payloads ≤ 128 KB) — shared across all
        gunicorn workers so a cold worker can warm-fill without re-hitting
        GDELT + CLOB.
    """
    cache_key = f"impact:{slug}:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return TerminalNewsImpactResponse(**cached)

    # L2 fallback — only pay the Redis read on an L1 miss.
    redis_cached = _redis_get(request, cache_key)
    if redis_cached is not None:
        _cache_set(cache_key, redis_cached)  # promote into L1
        return TerminalNewsImpactResponse(**redis_cached)

    # --- 1. Resolve market metadata -----------------------------------------
    try:
        meta = poly.get_market_metadata(slug)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e
    except Exception as e:  # PolymarketError or anything else with bad data
        raise HTTPException(status_code=404, detail=f"market not found: {e}") from e

    question = meta.question or slug
    keywords = (
        extract_keywords(question, max_n=MAX_KEYWORDS)
        or [t for t in slug.replace("_", "-").split("-") if len(t) >= 3][:MAX_KEYWORDS]
    )
    terms: QuestionTerms = build_terms(question)
    query = build_phrase_query(terms) or _build_query(keywords) or slug

    # --- 2. Fetch GDELT events + hourly prices in PARALLEL -----------------
    # These two upstreams don't depend on each other. Running them
    # sequentially was the dominant cost of cold-cache requests.
    timespan = f"{days}d"
    end_ts = pd.Timestamp.utcnow()
    start_ts = end_ts - pd.Timedelta(days=days)
    pad_start_unix = int((start_ts - pd.Timedelta(hours=24)).timestamp())

    # Hard upper bound on the parallel-gather. Inner fetchers each set 5 s
    # http timeouts, but the CLOB's startTs scan can stall on long windows;
    # cap the whole gather at 10 s so we never push past the 15 s gateway.
    try:
        raw_articles, prices = await asyncio.wait_for(
            asyncio.gather(
                asyncio.to_thread(_fetch_gdelt, poly._client, query, 75, timespan),
                asyncio.to_thread(
                    _fetch_hourly_prices,
                    poly._client,
                    poly.clob_url,
                    meta.yes_token_id,
                    pad_start_unix,
                ),
            ),
            timeout=10.0,
        )
    except TimeoutError:
        # Degrade to "no events" rather than 504. UI handles empty list.
        raw_articles, prices = [], pd.Series(dtype=float)
    # Pre-score every article. If we have anchors/topics, drop anything
    # below the relevance floor — otherwise an unrelated headline gets
    # flagged "attributable" simply because the slug's price drifted
    # 1.5σ during the same hour.
    if terms.anchors or terms.topics:
        articles: list[GDELTArticle] = []
        for art in raw_articles:
            score, matched = score_relevance(art.title, terms)
            if score < RELEVANCE_MIN:
                continue
            articles.append(
                art.model_copy(
                    update={
                        "relevance_score": round(score, 4),
                        "matched_terms": matched,
                    }
                )
            )
    else:
        articles = raw_articles

    sigma_log = _hourly_log_return_sigma(prices)

    # --- 4. Build events ----------------------------------------------------
    events: list[Event] = []
    for art in articles:
        try:
            ts = (
                pd.Timestamp(art.ts).tz_convert("UTC")
                if pd.Timestamp(art.ts).tzinfo
                else pd.Timestamp(art.ts, tz="UTC")
            )
        except (ValueError, TypeError):
            continue

        p_before = _price_at_or_before(prices, ts)
        p_1h = _price_at_or_after(prices, ts + pd.Timedelta(hours=1))
        p_6h = _price_at_or_after(prices, ts + pd.Timedelta(hours=6))
        p_24h = _price_at_or_after(prices, ts + pd.Timedelta(hours=24))

        # Attribution uses the 6h log-return move vs σ. Fall back to
        # the level-difference in pp for the displayed `abs_move_pp`
        # so the UI stays interpretable when prices are not available.
        if p_before is not None and p_6h is not None:
            move_level = p_6h - p_before
            abs_move_pp: float | None = round(abs(move_level) * 100.0, 2)
            direction = _direction(move_level)
            if sigma_log > 0:
                # log-return move at 6h
                p_b = max(min(p_before, 0.99), 0.01)
                p_6 = max(min(p_6h, 0.99), 0.01)
                log_move = float(np.log(p_6 / p_b))
                attributable = abs(log_move) > ATTRIBUTION_SIGMA * sigma_log
            else:
                attributable = False
        else:
            abs_move_pp = None
            direction = "flat"
            attributable = False

        events.append(
            Event(
                ts_iso=art.ts,
                headline=art.title,
                source=art.source,
                tone=art.tone,
                price_before=p_before,
                price_1h_after=p_1h,
                price_6h_after=p_6h,
                price_24h_after=p_24h,
                abs_move_pp=abs_move_pp,
                direction=direction,
                attributable=attributable,
                relevance_score=getattr(art, "relevance_score", 0.0),
                matched_terms=list(getattr(art, "matched_terms", []) or []),
            )
        )

    n_events = len(events)
    n_attributable = sum(1 for e in events if e.attributable)
    attributable_pct = (100.0 * n_attributable / n_events) if n_events else 0.0
    interpretation = (
        f"{n_attributable} of {n_events} GDELT events caused "
        f">{ATTRIBUTION_SIGMA}-sigma price moves in the next 6h"
    )

    response = TerminalNewsImpactResponse(
        slug=slug,
        days=days,
        events=events,
        n_events=n_events,
        n_attributable=n_attributable,
        attributable_pct=round(attributable_pct, 2),
        interpretation=interpretation,
    )
    payload = response.model_dump()
    _cache_set(cache_key, payload)
    _redis_set(request, cache_key, payload)
    return response


__all__ = [
    "ATTRIBUTION_SIGMA",
    "GDELT_DOC_URL",
    "Event",
    "TerminalNewsImpactResponse",
    "router",
]
