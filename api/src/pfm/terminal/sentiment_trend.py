"""Terminal sentiment-trend backend.

Tracks GDELT 2.0 news tone over time per Polymarket market and correlates
the daily mean tone with the YES-token price series. The hypothesis is that
sentiment regime shifts often *precede* price moves, so we explicitly search
for the lag ``k ∈ [-7, +7]`` that maximises ``corr(tone[t], price[t+k])``.

Two endpoints are exposed under :data:`router`::

    GET /terminal/sentiment-trend/{slug}?days=30
    GET /terminal/sentiment-trend/spike-alerts?days=7&min_n_articles=3

Routing note: this module owns its :class:`fastapi.APIRouter`; ``main.py``
is left untouched (per CLAUDE.md). To activate::

    from pfm.terminal_sentiment_trend import router as sentiment_trend_router
    app.include_router(sentiment_trend_router)

Implementation notes
--------------------
* No scipy — Pearson r and the lag scan are written by hand on plain
  ``list[float]`` to keep the dependency surface tight (see CLAUDE.md
  "What not to do").
* Article timestamps are normalised to UTC dates via the same convention
  used elsewhere in the codebase (``pandas.Timestamp(...).normalize()``).
* GDELT is fetched once with ``timespan=<days>d`` and bucketed locally by
  date — that's both cheaper and more reliable than ``days`` separate
  queries.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Annotated

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from pfm.sources.polymarket import (
    PolymarketClient,
    discover_markets,
    fetch_factor_history,
)
from pfm.terminal_gdelt_news import (
    GDELT_DOC_URL,
    HARD_CAP_RECORDS,
    USER_AGENT,
    GDELTArticle,
    _build_query,
    _dominant_topic,
    _parse_articles,
    _seendate_to_iso,
)
from pfm.terminal_news import MAX_KEYWORDS, extract_keywords

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS: int = 900  # match GDELT's 15-minute update cadence
# Cross-worker L2 cache (Redis). Shorter than L1 so a stale single worker
# can't stick to a 15-min-old answer when the other workers have refreshed.
REDIS_TTL_SECONDS: int = 300  # 5 min
REDIS_PAYLOAD_MAX_BYTES: int = 128 * 1024  # 128 KB
MAX_LAG_DAYS: int = 7
DEFAULT_DAYS: int = 30
MIN_DAYS: int = 3
MAX_DAYS: int = 90
GDELT_FETCH_RECORDS: int = 250  # cap is HARD_CAP_RECORDS; pull as much as we can
BULL_THRESHOLD: float = 2.0
BEAR_THRESHOLD: float = -2.0
SPIKE_THRESHOLD: float = 3.0


# --- schemas ----------------------------------------------------------------


class TonePoint(BaseModel):
    """One daily bucket of tone."""

    date: str = Field(..., description="UTC date in ``YYYY-MM-DD`` form.")
    mean_tone: float = Field(..., description="Mean GDELT tone in [-10, +10].")
    n_articles: int
    dominant_topic: str = Field(
        ..., description="Most-frequent non-stopword across the day's titles."
    )


class SentimentTrendResponse(BaseModel):
    slug: str
    current_tone: float = Field(
        ...,
        description="Mean tone over the last 24h (range [-10, +10]); 0 if no articles.",
    )
    tone_series: list[TonePoint]
    sentiment_regime: str = Field(..., description="``bullish`` | ``bearish`` | ``neutral``.")
    correlation_with_price: float = Field(
        ...,
        description=("Best-lag Pearson correlation between daily mean tone and YES-token price."),
    )
    lead_lag_days: int = Field(
        ...,
        description=(
            "Lag k in [-7, +7] at which the correlation peaks. Positive means news "
            "leads price by k days; 0 means contemporaneous; negative means price "
            "leads news."
        ),
    )
    interpretation: str = Field(..., description="Human-readable summary of regime + correlation.")
    # When the upstream sentiment source (GDELT) is unreachable / throttled /
    # returns no tone signal we surface a graceful degraded payload rather
    # than a flat-zero chart that misleads the user. The frontend can then
    # render "Sentiment unavailable for this market" instead.
    degraded_mode: bool = Field(
        False,
        description=(
            "True when no usable tone signal was retrieved from the upstream "
            "GDELT API (e.g. throttling, network error, or no tone returned)."
        ),
    )
    reason: str | None = Field(
        None,
        description="Human-readable explanation when ``degraded_mode`` is True.",
    )


class SpikeAlert(BaseModel):
    slug: str
    question: str
    tone_start: float
    tone_end: float
    tone_shift: float = Field(..., description="``tone_end - tone_start``.")
    n_articles: int
    direction: str = Field(..., description="``up`` (positive shift) or ``down``.")


class SpikeAlertsResponse(BaseModel):
    days: int
    min_n_articles: int
    n_alerts: int
    alerts: list[SpikeAlert]


# --- in-memory + Redis L2 cache --------------------------------------------

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
    return f"terminal_sentiment_trend:{cache_key}"


def _redis_get(request: Request, cache_key: str) -> dict | None:
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
    cache = getattr(request.app.state, "cache", None)
    if cache is None or not getattr(cache, "enabled", False):
        return
    try:
        blob = json.dumps(payload, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return
    if len(blob) > REDIS_PAYLOAD_MAX_BYTES:
        logger.info(
            "sentiment-trend: skipping Redis SET — payload %d B > %d B cap",
            len(blob),
            REDIS_PAYLOAD_MAX_BYTES,
        )
        return
    with contextlib.suppress(Exception):  # defensive
        cache.set(_redis_key(cache_key), blob, REDIS_TTL_SECONDS)


# --- helpers ----------------------------------------------------------------


def _utc_date(ts: str) -> str | None:
    """Map a GDELT ISO-8601 timestamp → ``YYYY-MM-DD`` UTC date or ``None``."""
    if not ts:
        return None
    try:
        # GDELT ISO has trailing "Z"; pandas handles it.
        return pd.Timestamp(ts).tz_convert("UTC").normalize().strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        try:
            # Fallback: maybe the timestamp is naive.
            return pd.Timestamp(ts).normalize().strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return None


def _bucket_articles_by_date(
    articles: list[GDELTArticle],
) -> dict[str, list[GDELTArticle]]:
    """Group articles into UTC-date buckets, dropping records with bad timestamps."""
    out: dict[str, list[GDELTArticle]] = defaultdict(list)
    for art in articles:
        d = _utc_date(art.ts)
        if d is None:
            continue
        out[d].append(art)
    return out


def _normalize_timeline_keys(
    tone_timeline: dict[str, float] | None,
) -> dict[str, float]:
    """Coerce timeline keys to ``YYYY-MM-DD`` UTC date strings.

    The upstream ``_fetch_gdelt_tone_timeline`` already returns date-only
    keys, but defensively re-normalising here means the function stays
    correct even when a caller (or a future refactor) hands us keys that
    still carry the hour/minute suffix from GDELT's raw ``seendate`` —
    which is exactly the failure mode that caused per-day tone points to
    stay at 0.0 even when ``current_tone`` was populated correctly.

    Keys that fail to parse are silently dropped (better than poisoning
    the alignment with garbage).
    """
    if not tone_timeline:
        return {}
    out: dict[str, float] = {}
    for raw_key, val in tone_timeline.items():
        key = str(raw_key)
        if len(key) == 10 and key[4] == "-" and key[7] == "-":
            # Already YYYY-MM-DD — keep as-is.
            out[key] = val
            continue
        try:
            ts = pd.Timestamp(_seendate_to_iso(key))
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            out[ts.normalize().strftime("%Y-%m-%d")] = val
        except (ValueError, TypeError):
            continue
    return out


def _headline_tone_for_bucket(bucket: list[GDELTArticle]) -> float:
    """Score the article titles in a day's bucket via the hybrid NLP scorer.

    Returns the mean compound score rescaled from ``[-1, +1]`` to the
    GDELT tone range ``[-10, +10]`` so it composes cleanly with native
    article tone and the ``timelinetone`` overlay (both of which already
    live on the GDELT scale). When the bucket is empty or every headline
    is sentiment-neutral, returns ``0.0`` and the caller treats that as
    "no signal".
    """
    if not bucket:
        return 0.0
    try:
        from pfm.terminal.sentiment_nlp import score_headline
    except Exception:  # never break the trend pipeline on NLP import
        return 0.0
    scores: list[float] = []
    for art in bucket:
        title = (art.title or "").strip()
        if not title:
            continue
        try:
            compound, _label = score_headline(title)
        except Exception:  # defensive: skip a bad row
            continue
        if abs(compound) > 1e-6:
            scores.append(compound)
    if not scores:
        return 0.0
    # Rescale compound ∈ [-1, +1] → GDELT-style tone ∈ [-10, +10].
    return (sum(scores) / len(scores)) * 10.0


def _build_tone_series(
    articles: list[GDELTArticle],
    fallback_topic: str,
    days: int,
    end_date: pd.Timestamp,
    tone_timeline: dict[str, float] | None = None,
) -> list[TonePoint]:
    """Build a daily tone series for the last ``days`` UTC dates ending at ``end_date``.

    Days with no articles are emitted with ``n_articles=0`` so the series is
    dense and aligns 1:1 with the price series.

    Tone resolution per day (in priority order):

    1. If the articles for that day carry a non-zero tone (the case in tests
       where article fixtures explicitly set a ``tone`` field), use the
       article-level mean.
    2. Otherwise fall back to ``tone_timeline[day]`` if provided. This is the
       real-world path: GDELT's ``artlist`` mode does NOT return per-article
       tone, so we overlay a separate ``mode=timelinetone`` daily aggregate.
       Keys are defensively re-normalised to ``YYYY-MM-DD`` so an upstream
       timestamp suffix (``T00:00:00Z``) can't silently break alignment.
    3. If neither the article tone nor the timeline carries signal but
       articles ARE present, score the article TITLES via the hybrid
       VADER + financial-lex scorer (``sentiment_nlp.score_headline``).
       This lets the chart show real motion even when GDELT's artlist and
       timelinetone are both empty — important for niche slugs where
       coverage is patchy.
    4. If the timeline is sparse, forward-fill from the previous resolved
       day (and backward-fill leading gaps from the first resolved day) so
       the daily series isn't pockmarked with zeros that misread as
       "tone=neutral" rather than "no fresh sample today". This matches how
       Bloomberg-style sentiment ribbons render piecewise-constant tone.
    5. If no source has any tone for the entire window, emit ``0.0`` and
       let the caller surface ``degraded_mode=True``.
    """
    buckets = _bucket_articles_by_date(articles)
    end_norm = end_date.tz_convert("UTC") if end_date.tzinfo else end_date.tz_localize("UTC")
    end_norm = end_norm.normalize()
    # Defensive key normalisation: ensure every timeline key is YYYY-MM-DD
    # so ``d in tone_timeline`` lookups succeed for past days. This is the
    # exact alignment fix called out in the sentiment-consolidation pass.
    tone_timeline = _normalize_timeline_keys(tone_timeline)

    # First pass: resolve a tone value (or None for "missing") per day,
    # preferring article-level when non-zero, else the timeline overlay,
    # else a headline-NLP fallback so the series is informative even when
    # GDELT returns no tone signal whatsoever.
    resolved: list[tuple[str, list[GDELTArticle], float | None]] = []
    for offset in range(days - 1, -1, -1):
        d = (end_norm - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        bucket = buckets.get(d, [])
        article_tone = sum(a.tone for a in bucket) / len(bucket) if bucket else 0.0
        if bucket and article_tone != 0.0:
            tone_val: float | None = article_tone
        elif d in tone_timeline:
            tone_val = tone_timeline[d]
        elif bucket:
            # Fallback: hybrid NLP score on the day's headlines, mapped to
            # the GDELT tone scale. ``None`` if every headline is neutral.
            headline_tone = _headline_tone_for_bucket(bucket)
            tone_val = headline_tone if headline_tone != 0.0 else None
        else:
            tone_val = None
        resolved.append((d, bucket, tone_val))

    # Second pass: forward-fill missing days from the previous resolved
    # value. We then back-fill any leading None entries from the first
    # resolved value (so the chart doesn't start at a fake zero before the
    # first available sample).
    last_seen: float | None = None
    filled: list[float | None] = []
    for _, _, tone_val in resolved:
        if tone_val is not None:
            last_seen = tone_val
        filled.append(last_seen)
    # Back-fill leading Nones from the first non-None value.
    first_seen: float | None = next((v for v in filled if v is not None), None)
    filled = [v if v is not None else first_seen for v in filled]

    series: list[TonePoint] = []
    for (d, bucket, _), tone_val in zip(resolved, filled, strict=True):
        topic = _dominant_topic(bucket, fallback=fallback_topic) if bucket else fallback_topic
        mean_tone = tone_val if tone_val is not None else 0.0
        series.append(
            TonePoint(
                date=d,
                mean_tone=round(mean_tone, 4),
                n_articles=len(bucket),
                dominant_topic=topic,
            )
        )
    return series


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation, no scipy. Returns 0.0 when undefined."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = 0.0
    sxx = 0.0
    syy = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx = x - mx
        dy = y - my
        num += dx * dy
        sxx += dx * dx
        syy += dy * dy
    den = (sxx * syy) ** 0.5
    if den == 0.0:
        return 0.0
    return num / den


def _best_lag_correlation(
    tone: list[float],
    price: list[float],
    max_lag: int = MAX_LAG_DAYS,
) -> tuple[float, int]:
    """Find ``k ∈ [-max_lag, +max_lag]`` that maximises ``|corr(tone[t], price[t+k])|``.

    Returns ``(best_corr, best_k)``. ``k > 0`` means tone leads price by ``k`` days.
    """
    n = min(len(tone), len(price))
    if n == 0:
        return 0.0, 0

    best_corr = 0.0
    best_k = 0
    best_abs = 0.0  # tie-break in favour of k=0 / no shift
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            xs = tone[: n - k] if k > 0 else tone[:n]
            ys = price[k:n] if k > 0 else price[:n]
        else:
            # k < 0 → price leads tone; align price[t] with tone[t-k]
            xs = tone[-k:n]
            ys = price[: n + k]
        if len(xs) < 3:
            continue
        r = _pearson(xs, ys)
        if abs(r) > best_abs:
            best_abs = abs(r)
            best_corr = r
            best_k = k
    return best_corr, best_k


def _classify_regime(recent_mean: float) -> str:
    if recent_mean > BULL_THRESHOLD:
        return "bullish"
    if recent_mean < BEAR_THRESHOLD:
        return "bearish"
    return "neutral"


def _last_24h_tone(
    articles: list[GDELTArticle],
    now: datetime,
    tone_timeline: dict[str, float] | None = None,
) -> float:
    """Mean tone over the last 24h.

    Resolution priority mirrors :func:`_build_tone_series`:
    1. Article-level tone when present and non-zero.
    2. ``tone_timeline`` value for today's UTC date.
    3. Hybrid NLP headline score over the last-24h articles (mapped from
       compound ``[-1, +1]`` to the GDELT tone scale ``[-10, +10]``).
    """
    cutoff = now - timedelta(hours=24)
    recent: list[GDELTArticle] = []
    tones: list[float] = []
    for art in articles:
        try:
            ts = pd.Timestamp(art.ts).tz_convert("UTC")
        except (ValueError, TypeError):
            continue
        if ts.to_pydatetime() >= cutoff:
            recent.append(art)
            if art.tone != 0.0:
                tones.append(art.tone)
    if tones:
        return round(sum(tones) / len(tones), 4)
    if tone_timeline:
        normalized = _normalize_timeline_keys(tone_timeline)
        today = now.astimezone(UTC).strftime("%Y-%m-%d")
        if today in normalized:
            return round(normalized[today], 4)
    if recent:
        headline_tone = _headline_tone_for_bucket(recent)
        if headline_tone != 0.0:
            return round(headline_tone, 4)
    return 0.0


def _interpretation(
    series: list[TonePoint],
    regime: str,
    corr: float,
    lag: int,
) -> str:
    # Prefer days with articles; otherwise fall back to days that still
    # carry a tone-overlay value (artlist + timelinetone is decoupled in
    # production — see ``_fetch_gdelt_tone_timeline``).
    with_articles = [p for p in series if p.n_articles > 0]
    with_tone = [p for p in series if p.mean_tone != 0.0]
    nonzero = with_articles or with_tone
    if not nonzero:
        return "No news coverage in the requested window — sentiment regime undefined."

    # Pick the start of the window (oldest non-empty bucket) and the latest
    # point as the "tone has shifted from X to Y" anchors.
    tone_start = nonzero[0].mean_tone
    tone_end = nonzero[-1].mean_tone
    n_days = (pd.Timestamp(nonzero[-1].date) - pd.Timestamp(nonzero[0].date)).days or 1

    if lag > 0:
        lag_phrase = f"leading price by {lag} day{'s' if lag != 1 else ''}"
    elif lag < 0:
        lag_phrase = f"lagging price by {-lag} day{'s' if lag != 1 else ''}"
    else:
        lag_phrase = "moving contemporaneously with price"

    return (
        f"Tone has shifted from {tone_start:+.1f} to {tone_end:+.1f} over "
        f"{n_days} day{'s' if n_days != 1 else ''} "
        f"(regime: {regime}, corr={corr:+.2f}, {lag_phrase})."
    )


def _fetch_gdelt_tone_timeline(
    client: httpx.Client,
    query: str,
    days: int,
) -> dict[str, float]:
    """Fetch GDELT 2.0 ``mode=timelinetone`` and bucket into daily mean tone.

    The GDELT DOC ``artlist`` mode does **not** include a per-article tone
    field in its JSON output (only ``url``, ``title``, ``seendate``,
    ``domain``, ``language``, ``sourcecountry``, ``socialimage``). To get
    real tone numbers we have to hit ``mode=timelinetone``, which returns
    hourly mean-tone samples. We aggregate those into UTC days and return
    a mapping ``{YYYY-MM-DD: mean_tone}``. Empty dict on any failure /
    throttle so the caller can degrade gracefully.
    """
    params: dict[str, str | int] = {
        "query": query,
        "mode": "timelinetone",
        "format": "json",
        "timespan": f"{int(days)}d",
    }
    try:
        r = client.get(
            GDELT_DOC_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=5.0,
        )
    except httpx.HTTPError as e:
        logger.warning("gdelt timelinetone fetch failed: %s", e)
        return {}
    if r.status_code >= 400:
        logger.warning("gdelt timelinetone non-2xx: %s body=%s", r.status_code, r.text[:200])
        return {}
    body = r.text or ""
    if body.lstrip().startswith("Please limit"):
        logger.warning("gdelt timelinetone throttled this caller")
        return {}
    try:
        payload = r.json()
    except ValueError:
        logger.warning("gdelt timelinetone non-JSON: %s", body[:200])
        return {}
    if not isinstance(payload, dict):
        return {}
    timeline = payload.get("timeline") or []
    if not isinstance(timeline, list) or not timeline:
        return {}
    # GDELT returns one entry per "series"; the per-hour samples we want
    # are under ``series == "Average Tone"`` (case-sensitive). Be tolerant
    # and just take the first series with a ``data`` list if the label
    # changes upstream.
    data_points: list[dict] = []
    for entry in timeline:
        if not isinstance(entry, dict):
            continue
        d = entry.get("data")
        if isinstance(d, list) and d:
            if entry.get("series") == "Average Tone" or not data_points:
                data_points = d
                if entry.get("series") == "Average Tone":
                    break
    if not data_points:
        return {}
    buckets: dict[str, list[float]] = defaultdict(list)
    for pt in data_points:
        if not isinstance(pt, dict):
            continue
        date_raw = pt.get("date")
        val = pt.get("value")
        if not date_raw or val is None:
            continue
        try:
            ts = pd.Timestamp(_seendate_to_iso(str(date_raw)))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            day = ts.normalize().strftime("%Y-%m-%d")
            v = float(val)
        except (ValueError, TypeError):
            continue
        buckets[day].append(v)
    return {d: sum(vs) / len(vs) for d, vs in buckets.items() if vs}


def _fetch_gdelt_window(
    client: httpx.Client,
    query: str,
    days: int,
    max_records: int = GDELT_FETCH_RECORDS,
) -> list[GDELTArticle]:
    """Fetch up to ``max_records`` GDELT articles over the last ``days`` days.

    Returns an empty list on any failure (HTTP error, throttle, bad JSON);
    the caller is responsible for handling sparse data.
    """
    params: dict[str, str | int] = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(int(max_records), HARD_CAP_RECORDS),
        "sort": "datedesc",
        "timespan": f"{int(days)}d",
    }
    try:
        r = client.get(
            GDELT_DOC_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=5.0,
        )
    except httpx.HTTPError as e:
        logger.warning("gdelt fetch failed (sentiment-trend): %s", e)
        return []
    if r.status_code >= 400:
        logger.warning("gdelt non-2xx (sentiment-trend): %s body=%s", r.status_code, r.text[:200])
        return []
    body = r.text or ""
    if body.lstrip().startswith("Please limit"):
        logger.warning("gdelt throttled this caller (sentiment-trend)")
        return []
    try:
        payload = r.json()
    except ValueError:
        logger.warning("gdelt non-JSON (sentiment-trend): %s", body[:200])
        return []
    if not isinstance(payload, dict):
        return []
    return _parse_articles(payload)


def _resolve_question(poly: PolymarketClient, slug: str) -> str:
    """Return the Gamma question for ``slug`` or raise 404 / 502.

    Routes through the cached ``poly.get_market_metadata`` so we benefit from
    polymarket.py's 1h cache + 1.5s 429-retry — eliminates the gamma-429
    cascade that previously surfaced as user-visible 502s here.
    """
    try:
        meta = poly.get_market_metadata(slug)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e
    except Exception as e:
        msg = str(e)
        if "no market found" in msg:
            raise HTTPException(status_code=404, detail=msg) from e
        raise HTTPException(status_code=502, detail=f"polymarket error: {e}") from e
    question = (meta.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=502, detail=f"market {slug!r} missing question in gamma payload"
        )
    return question


def _keywords_for(slug: str, question: str) -> list[str]:
    keywords = extract_keywords(question, max_n=MAX_KEYWORDS)
    if not keywords:
        keywords = [t for t in re.split(r"[-_]+", slug) if len(t) >= 3][:MAX_KEYWORDS]
    return keywords


# --- dependency -------------------------------------------------------------


router = APIRouter(prefix="/terminal/sentiment-trend", tags=["terminal"])


def get_polymarket_client(request: Request) -> PolymarketClient:
    poly = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


# --- endpoints --------------------------------------------------------------


@router.get(
    "/spike-alerts",
    response_model=SpikeAlertsResponse,
    summary="Markets where mean tone has shifted by more than 3.0 in the last N days.",
)
def get_spike_alerts(
    request: Request,
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = 7,
    min_n_articles: Annotated[int, Query(ge=1, le=100)] = 3,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> SpikeAlertsResponse:
    """Scan high-volume markets for sudden tone shifts.

    For each candidate market we pull a single GDELT window over the last
    ``days`` days, split it into the first and second halves of that window,
    compute mean tone in each half, and flag the market if the absolute
    shift exceeds :data:`SPIKE_THRESHOLD`.
    """
    cache_key = f"spike:{days}:{min_n_articles}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return SpikeAlertsResponse(**cached)

    # Bound the upstream fan-out so a cold cache can't blow past the
    # 15 s gateway deadline. 8 candidates × 2 GDELT calls @ 5 s timeout =
    # ~10 s wall-clock worst case at 8-way concurrency below.
    try:
        candidates = discover_markets(poly, limit=8, pages=1)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e

    half = max(1, days // 2)
    cutoff_dt_ms = int((datetime.now(tz=UTC) - timedelta(days=half)).timestamp() * 1000)
    _ = cutoff_dt_ms  # silence unused — kept for future date-aware filtering

    end = pd.Timestamp(datetime.now(tz=UTC))

    # Pre-resolve per-candidate keywords/queries so the worker only does
    # the slow GDELT fetches.
    work_items: list[tuple[object, list[str], str]] = []
    for cand in candidates:
        keywords = _keywords_for(cand.slug, cand.question)
        if not keywords:
            continue
        query = _build_query(keywords)
        work_items.append((cand, keywords, query))

    def _scan_one(item: tuple[object, list[str], str]) -> SpikeAlert | None:
        cand, keywords, query = item
        # Fan out the two GDELT round-trips concurrently. The wave-1
        # outer fan-out parallelised across candidates but each worker
        # still ran ``artlist`` then ``timelinetone`` serially (~5 s
        # each → ~10 s per candidate, ~20 s total at 8 workers). Doing
        # them concurrently within the worker halves that to ~5 s for
        # the slowest pair and unblocks the executor queue sooner.
        try:
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="spike-alerts-gdelt"
            ) as inner_ex:
                fut_articles = inner_ex.submit(_fetch_gdelt_window, poly._client, query, days)
                fut_timeline = inner_ex.submit(
                    _fetch_gdelt_tone_timeline, poly._client, query, days
                )
                articles = fut_articles.result()
                tone_timeline = fut_timeline.result()
            if len(articles) < min_n_articles:
                return None
            # Real GDELT artlist returns no tone field — overlay the daily
            # tone aggregate from mode=timelinetone so the shift comparison
            # isn't always 0 - 0.
        except Exception as e:
            logger.info("spike-alerts gdelt fetch failed for %s: %s", cand.slug, e)
            return None

        series = _build_tone_series(
            articles,
            fallback_topic=keywords[0],
            days=days,
            end_date=end,
            tone_timeline=tone_timeline,
        )
        first = [p for p in series[:half] if p.n_articles > 0]
        second = [p for p in series[half:] if p.n_articles > 0]
        if not first or not second:
            return None
        tone_start = sum(p.mean_tone for p in first) / len(first)
        tone_end = sum(p.mean_tone for p in second) / len(second)
        shift = tone_end - tone_start
        if abs(shift) <= SPIKE_THRESHOLD:
            return None
        return SpikeAlert(
            slug=cand.slug,
            question=cand.question,
            tone_start=round(tone_start, 4),
            tone_end=round(tone_end, 4),
            tone_shift=round(shift, 4),
            n_articles=len(articles),
            direction="up" if shift > 0 else "down",
        )

    # Fan-out across candidates in a bounded thread pool. With ~15
    # candidates × 2 sequential GDELT calls each the serial path routinely
    # blew past 30 s on cold cache; concurrency caps the wall clock at
    # roughly the slowest pair of fetches (~3-4 s).
    alerts: list[SpikeAlert] = []
    if work_items:
        max_workers = min(len(work_items), 8)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spike-alerts") as ex:
            for result in ex.map(_scan_one, work_items):
                if result is not None:
                    alerts.append(result)

    alerts.sort(key=lambda a: abs(a.tone_shift), reverse=True)
    response = SpikeAlertsResponse(
        days=days,
        min_n_articles=min_n_articles,
        n_alerts=len(alerts),
        alerts=alerts,
    )
    _cache_set(cache_key, response.model_dump())
    return response


@router.get(
    "/{slug}",
    response_model=SentimentTrendResponse,
    summary="GDELT tone series for a market, with lag-correlation against price.",
)
async def get_sentiment_trend(
    request: Request,
    slug: Annotated[str, Path(min_length=1, description="Polymarket market slug.")],
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> SentimentTrendResponse:
    """Return a daily tone series and its best-lag correlation with YES price.

    Cache layers:
      - L1 (process dict, 15 min TTL).
      - L2 (Redis, 5 min TTL, payloads ≤ 128 KB) so cold-worker hits
        avoid the triple GDELT+CLOB round-trip.

    Three upstream calls (GDELT articles, GDELT tone timeline, CLOB price
    history) are independent of one another and now fan out in parallel
    via ``asyncio.gather``. Net effect: warm latency dominated by the
    slowest single fetch instead of summed.
    """
    cache_key = f"trend:{slug}:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return SentimentTrendResponse(**cached)

    redis_cached = _redis_get(request, cache_key)
    if redis_cached is not None:
        _cache_set(cache_key, redis_cached)
        return SentimentTrendResponse(**redis_cached)

    question = _resolve_question(poly, slug)
    keywords = _keywords_for(slug, question)
    fallback_topic = keywords[0] if keywords else slug
    query = _build_query(keywords) if keywords else slug

    now = datetime.now(tz=UTC)
    end_ts = pd.Timestamp(now)
    start_ts = end_ts - pd.Timedelta(days=days + MAX_LAG_DAYS + 1)

    # Fan out: GDELT artlist + GDELT timelinetone + CLOB prices in parallel.
    # ``fetch_factor_history`` may raise; gather with return_exceptions so a
    # CLOB failure doesn't poison the GDELT results.
    async def _safe_factor_history() -> pd.DataFrame:
        try:
            return await asyncio.to_thread(fetch_factor_history, poly, slug, start_ts, end_ts)
        except httpx.HTTPError:
            # Bubble up so the outer handler can map to 502.
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("price fetch failed for %s: %s", slug, e)
            return pd.DataFrame()

    # Hard upper bound on the parallel-gather: GDELT calls are 5 s each
    # and fetch_factor_history relies on the 15 s client default; without
    # this, a slow CLOB can push the handler past the 15 s gateway deadline.
    try:
        articles, tone_timeline, price_df = await asyncio.wait_for(
            asyncio.gather(
                asyncio.to_thread(_fetch_gdelt_window, poly._client, query, days),
                asyncio.to_thread(_fetch_gdelt_tone_timeline, poly._client, query, days),
                _safe_factor_history(),
            ),
            timeout=10.0,
        )
    except TimeoutError:
        # Degrade gracefully — the response model carries degraded_mode for this.
        articles, tone_timeline, price_df = [], {}, pd.DataFrame()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket clob error: {e}") from e

    # 2. Build the dense daily tone series.
    series = _build_tone_series(
        articles, fallback_topic, days, end_date=end_ts, tone_timeline=tone_timeline
    )

    # 4. Align tone series to price by date.
    by_date = {p.date: p for p in series}
    aligned_tone: list[float] = []
    aligned_price: list[float] = []
    if not price_df.empty:
        for ts, row in price_df.iterrows():
            d = ts.strftime("%Y-%m-%d") if isinstance(ts, pd.Timestamp) else str(ts)[:10]
            if d in by_date:
                aligned_tone.append(by_date[d].mean_tone)
                aligned_price.append(float(row["price"]))

    # 5. Best-lag Pearson correlation in [-7, +7].
    if len(aligned_tone) >= 3:
        corr, lag = _best_lag_correlation(aligned_tone, aligned_price, MAX_LAG_DAYS)
    else:
        corr, lag = 0.0, 0

    # 6. Regime classification using last 3 days with any signal — prefer
    #    days with articles, fall back to days that only have tone-overlay
    #    values (production GDELT artlist returns no tone, so we'd lose
    #    every regime classification otherwise).
    last3_articles = [p.mean_tone for p in series if p.n_articles > 0][-3:]
    last3_tone = [p.mean_tone for p in series if p.mean_tone != 0.0][-3:]
    last3 = last3_articles or last3_tone
    recent_mean = sum(last3) / len(last3) if last3 else 0.0
    regime = _classify_regime(recent_mean)

    # 7. Detect "no usable tone signal" and degrade gracefully so the UI
    #    can show "Sentiment unavailable" instead of a flat-zero chart.
    #    We now also count the hybrid-NLP headline fallback (mirrored in
    #    the built series) as a real source of signal — if every per-day
    #    point ended up at 0.0, we degrade; otherwise we don't.
    has_article_tone = any(a.tone != 0.0 for a in articles)
    has_timeline_tone = bool(tone_timeline)
    has_series_tone = any(p.mean_tone != 0.0 for p in series)
    degraded = not (has_article_tone or has_timeline_tone or has_series_tone)
    if degraded:
        if not articles and not tone_timeline:
            reason: str | None = "sentiment source unreachable"
        elif not tone_timeline:
            reason = "GDELT returned articles without tone signal"
        else:
            reason = "no tone signal in window"
    else:
        reason = None

    response = SentimentTrendResponse(
        slug=slug,
        current_tone=_last_24h_tone(articles, now, tone_timeline=tone_timeline),
        tone_series=series,
        sentiment_regime=regime,
        correlation_with_price=round(corr, 4),
        lead_lag_days=lag,
        interpretation=_interpretation(series, regime, corr, lag),
        degraded_mode=degraded,
        reason=reason,
    )
    payload = response.model_dump()
    _cache_set(cache_key, payload)
    # Only mirror healthy payloads into L2 — a degraded run shouldn't pin
    # other workers to "sentiment unavailable" for the full 5-min TTL.
    if not degraded:
        _redis_set(request, cache_key, payload)
    return response
