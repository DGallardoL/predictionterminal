"""Quote-page composer for Terminal: one-shot Yahoo-Finance-style payload.

The Terminal already exposes individual building blocks (``/terminal/market/{slug}``,
``/terminal/peers/{slug}``, ``/terminal/news/{slug}``, etc) but a "quote page"
needs all of them rendered on the same view, which means N round-trips from
the browser. This endpoint composes the full payload server-side, fan-out
fetches in parallel via :func:`asyncio.gather`, caches the result for 30s,
and returns a single envelope suitable for a single front-end render pass.

What's in the response
----------------------
- ``live``: best bid/ask, midpoint, spread (cents), 24h / 7d change.
- ``meta``: title, theme, end-date, days-to-resolve, total volume / OI.
- ``stats``: realised vol (30d), half-life, Hurst proxy (DFA-α), DFA alpha.
- ``day_range``: low / high of the last 24h hourly bars.
- ``week52_range``: low / high of the last 365 daily bars (or since-inception
  if shorter).
- ``implied_vol``: ``rv_30d * sqrt(365)`` — annualised vol of Δlogit.
- ``holders_estimate``: best-effort count parsed from Gamma's
  ``enrichedOrderBook`` if present.
- ``sparkline_30d``: list of last 30 daily closes.
- ``sparkline_intraday``: list of last 24 hourly closes.
- ``peers``: top-5 cointegrated counterparts with mini sparklines.
- ``news``: top-5 recent posts (Reddit + HN merged).
- ``similar_markets``: 3 markets sharing this market's theme.

Routing
-------
Owns its own :class:`fastapi.APIRouter`; ``main.py`` is left untouched.
Wire explicitly::

    from pfm.terminal_quote import router as terminal_quote_router
    app.include_router(terminal_quote_router)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from contextlib import AsyncExitStack
from typing import Annotated, Any

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi import Path as FPath
from pydantic import BaseModel, Field

from pfm import terminal as terminal_mod
from pfm.cache_pool import CachePool
from pfm.cache_utils import get_cache
from pfm.config import Settings, get_settings

logger = logging.getLogger(__name__)


# --- limits / cache ---------------------------------------------------------

DEFAULT_DAYS: int = 30
MIN_DAYS: int = 7
MAX_DAYS: int = 365
HTTP_TIMEOUT_SECONDS: float = 10.0
CACHE_TTL_SECONDS: int = 30
DEFAULT_CLIP_EPS: float = 0.01

ALLOWED_INCLUDES: frozenset[str] = frozenset({"peers", "news", "similar"})

# Process-wide cache keyed by (slug, days, frozenset(includes)).
_QUOTE_CACHE = get_cache("terminal_quote", ttl=CACHE_TTL_SECONDS)


def clear_cache() -> None:
    """Test/utility helper — drop every cached quote payload."""
    _QUOTE_CACHE.clear()


# --- Pydantic schemas -------------------------------------------------------


class QuoteLive(BaseModel):
    price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    spread_cents: float | None = None
    change_24h: float | None = None
    change_7d: float | None = None
    volume_24hr: float | None = None
    last_trade_price: float | None = None


class QuoteMeta(BaseModel):
    slug: str
    title: str
    theme: str | None = None
    end_date: str | None = None
    days_to_resolve: int | None = None
    total_volume: float | None = None
    total_open_interest: float | None = None


class QuoteStats(BaseModel):
    n_obs: int = 0
    rv_30d: float | None = None
    half_life: float | None = None
    hurst: float | None = None  # exposes DFA-α as a Hurst-style proxy
    dfa_alpha: float | None = None
    vif_max: float | None = None  # populated only when peers regression runs


class QuoteRange(BaseModel):
    low: float | None = None
    high: float | None = None


class QuotePeer(BaseModel):
    slug: str
    name: str
    correlation: float | None = None
    last_change_pct: float | None = None
    sparkline_7d: list[float] = Field(default_factory=list)


class QuoteNewsItem(BaseModel):
    title: str
    source: str
    time_ago: str
    sentiment_score: float | None = None
    url: str | None = None


class QuoteSimilarMarket(BaseModel):
    slug: str
    title: str
    theme: str | None = None
    price: float | None = None
    volume_24hr: float | None = None


class TerminalQuoteResponse(BaseModel):
    """Full /terminal/quote/{slug} envelope."""

    slug: str
    days: int
    includes: list[str]
    live: QuoteLive
    meta: QuoteMeta
    stats: QuoteStats
    day_range: QuoteRange
    week52_range: QuoteRange
    implied_vol: float | None = None
    holders_estimate: int | None = None
    sparkline_30d: list[float] = Field(default_factory=list)
    sparkline_intraday: list[float] = Field(default_factory=list)
    peers: list[QuotePeer] = Field(default_factory=list)
    news: list[QuoteNewsItem] = Field(default_factory=list)
    similar_markets: list[QuoteSimilarMarket] = Field(default_factory=list)


# --- helpers ----------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _logit(p: pd.Series, *, eps: float = DEFAULT_CLIP_EPS) -> pd.Series:
    s = p.astype(float).clip(lower=eps, upper=1.0 - eps)
    return np.log(s / (1.0 - s))


def _yes_token_id(market: dict[str, Any]) -> str | None:
    """Pull YES side ``clobTokenIds[0]`` — Polymarket ships it as a JSON string."""
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    try:
        token_ids = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(token_ids, list) or not token_ids:
        return None
    return str(token_ids[0])


def _parse_includes(raw: str) -> list[str]:
    """Accept comma-separated includes; only keep recognised tokens."""
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if not t or t not in ALLOWED_INCLUDES or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _holders_from_enriched_orderbook(market: dict[str, Any]) -> int | None:
    """Best-effort holder-count estimate from Gamma's enrichedOrderBook.

    Field shape isn't fully stable across markets; we accept either a
    direct count, a list of unique parties, or fall back to ``None``.
    """
    enr = market.get("enrichedOrderBook")
    if not enr:
        return None
    if isinstance(enr, str):
        try:
            enr = json.loads(enr)
        except (TypeError, json.JSONDecodeError):
            return None
    if isinstance(enr, dict):
        # Common shapes seen in the wild.
        for k in ("holderCount", "uniqueHolders", "holders_count", "numHolders"):
            v = enr.get(k)
            if isinstance(v, int) and v >= 0:
                return v
            if isinstance(v, (float, str)):
                try:
                    iv = int(float(v))
                    if iv >= 0:
                        return iv
                except (TypeError, ValueError):
                    pass
        # Lists of parties -> de-dupe by id/address.
        for k in ("holders", "parties", "uniqueParties"):
            arr = enr.get(k)
            if isinstance(arr, list):
                ids = {x.get("id") or x.get("address") for x in arr if isinstance(x, dict)}
                ids.discard(None)
                if ids:
                    return len(ids)
    return None


def _time_ago(ts_iso: str) -> str:
    """Render an ISO-8601 string as a coarse "Nm/h/d ago" label."""
    if not ts_iso:
        return ""
    try:
        ts = pd.Timestamp(ts_iso)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        delta = pd.Timestamp.utcnow().tz_localize(None) - ts.tz_convert("UTC").tz_localize(None)
        secs = max(0, int(delta.total_seconds()))
    except (ValueError, TypeError):
        return ""
    if secs < 60:
        return f"{secs}s ago"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _sentiment_score(label: str) -> float | None:
    """Map terminal_news's 3-way sentiment label to a numeric score."""
    if label == "positive":
        return 1.0
    if label == "negative":
        return -1.0
    if label == "neutral":
        return 0.0
    return None


# --- async fetchers ---------------------------------------------------------


# Process-local cache for gamma market dicts (W11-14: migrated to CachePool —
# gains optional Redis L2 + heap eviction + single-flight protection without
# changing call-site semantics).
# Slug→market data is effectively immutable for the question/tokenIds we care
# about; a 1h TTL is overwhelmingly safe and absorbs gamma 429 cascades.
_GAMMA_MARKET_CACHE_TTL_S: int = 3600
_GAMMA_MARKET_CACHE: CachePool = CachePool(namespace="terminal_quote_gamma", l1_maxsize=2048)


async def _fetch_gamma_market_async(
    http: httpx.AsyncClient, gamma_url: str, slug: str
) -> dict[str, Any]:
    """Cached gamma market fetch with single 429-retry.

    Without this, every Terminal market-detail open hits gamma fresh and
    cascades 429s into 502s for the /quote endpoint (the most user-visible
    one). The cache + retry mirrors the pattern in pfm.sources.polymarket.

    Uses ``CachePool.get_or_compute_async`` so concurrent first-callers
    for the same slug share a single upstream fetch (per-key
    ``asyncio.Lock``); ten parallel quote opens => one gamma call.
    """

    async def _fetch() -> dict[str, Any]:
        base = gamma_url.rstrip("/")
        r = await http.get(f"{base}/markets", params={"slug": slug}, timeout=HTTP_TIMEOUT_SECONDS)
        if r.status_code == 429:
            await asyncio.sleep(1.5)
            r = await http.get(
                f"{base}/markets", params={"slug": slug}, timeout=HTTP_TIMEOUT_SECONDS
            )
        r.raise_for_status()
        arr = r.json() or []
        if isinstance(arr, list) and arr:
            return arr[0]
        r2 = await http.get(
            f"{base}/markets",
            params={"slug": slug, "closed": "true"},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if r2.status_code == 200:
            arr2 = r2.json() or []
            if isinstance(arr2, list) and arr2:
                return arr2[0]
        raise LookupError(f"no market for slug={slug!r}")

    return await _GAMMA_MARKET_CACHE.get_or_compute_async(
        slug, _fetch, ttl=_GAMMA_MARKET_CACHE_TTL_S
    )


async def _fetch_clob_history_async(
    http: httpx.AsyncClient,
    clob_url: str,
    token_id: str,
    *,
    days: int,
    fidelity: int = 1440,
) -> pd.DataFrame:
    """Fetch CLOB ``/prices-history``. ``fidelity=1440`` is daily; 60 = hourly."""
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days + 1)
    params: dict[str, str | int] = {
        "market": token_id,
        "fidelity": int(fidelity),
        "interval": "max",
        "startTs": int(start.timestamp()),
    }
    r = await http.get(
        f"{clob_url.rstrip('/')}/prices-history",
        params=params,
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    payload = r.json() or {}
    history = payload.get("history", []) if isinstance(payload, dict) else []
    if not history:
        return pd.DataFrame(columns=["price"])
    df = pd.DataFrame(history)
    if fidelity >= 1440:
        df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.normalize()
    else:
        df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.rename(columns={"p": "price"})[["ts", "price"]]
    return df.set_index("ts").sort_index()


async def _fetch_news_for_slug(
    http: httpx.AsyncClient, question: str, *, limit: int = 5
) -> list[QuoteNewsItem]:
    """Re-implement the news fetch flow inline using the async client.

    Avoids importing from terminal_news (which depends on a shared
    PolymarketClient) so we stay self-contained and async.
    """
    if not question:
        return []
    # Cheap keyword extraction — pull the longest 3 alpha tokens.
    import re

    tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-z]{4,}", question)
        if t.lower() not in {"will", "have", "with", "from", "this", "that"}
    ]
    if not tokens:
        return []
    query = " ".join(tokens[:3])
    out: list[QuoteNewsItem] = []
    try:
        from pfm.terminal_news import (
            HN_SEARCH_URL,
            REDDIT_SEARCH_URL,
            USER_AGENT,
            classify_sentiment,
        )
    except ImportError:  # pragma: no cover — defensive
        return []

    async def _reddit() -> list[QuoteNewsItem]:
        try:
            r = await http.get(
                REDDIT_SEARCH_URL,
                params={"q": query, "sort": "new", "limit": int(limit)},
                headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError:
            return []
        if r.status_code >= 400:
            return []
        try:
            payload = r.json()
        except ValueError:
            return []
        items: list[QuoteNewsItem] = []
        for child in payload.get("data", {}).get("children", []):
            d = child.get("data", {}) if isinstance(child, dict) else {}
            title = str(d.get("title") or "").strip()
            if not title:
                continue
            permalink = d.get("permalink") or ""
            url = f"https://www.reddit.com{permalink}" if permalink else str(d.get("url") or "")
            ts_iso = ""
            created = d.get("created_utc")
            if created is not None:
                try:
                    ts_iso = pd.Timestamp(float(created), unit="s", tz="UTC").isoformat()
                except (TypeError, ValueError):
                    ts_iso = ""
            items.append(
                QuoteNewsItem(
                    title=title,
                    source="reddit",
                    time_ago=_time_ago(ts_iso),
                    sentiment_score=_sentiment_score(classify_sentiment(title)),
                    url=url or None,
                )
            )
        return items

    async def _hn() -> list[QuoteNewsItem]:
        try:
            r = await http.get(
                HN_SEARCH_URL,
                params={"query": query, "tags": "story", "hitsPerPage": int(limit)},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError:
            return []
        if r.status_code >= 400:
            return []
        try:
            payload = r.json()
        except ValueError:
            return []
        items: list[QuoteNewsItem] = []
        for hit in payload.get("hits", []):
            title = str(hit.get("title") or hit.get("story_title") or "").strip()
            if not title:
                continue
            url = str(
                hit.get("url")
                or hit.get("story_url")
                or (
                    f"https://news.ycombinator.com/item?id={hit['objectID']}"
                    if hit.get("objectID")
                    else ""
                )
            )
            items.append(
                QuoteNewsItem(
                    title=title,
                    source="hn",
                    time_ago=_time_ago(str(hit.get("created_at") or "")),
                    sentiment_score=_sentiment_score(classify_sentiment(title)),
                    url=url or None,
                )
            )
        return items

    reddit_items, hn_items = await asyncio.gather(_reddit(), _hn())
    out = (reddit_items or []) + (hn_items or [])
    return out[:limit]


async def _fetch_similar_markets(
    http: httpx.AsyncClient,
    gamma_url: str,
    *,
    theme_keyword: str | None,
    exclude_slug: str,
    limit: int = 3,
) -> list[QuoteSimilarMarket]:
    """Pick a few high-volume markets from Gamma. Theme filtering is best-effort."""
    base = gamma_url.rstrip("/")
    try:
        r = await http.get(
            f"{base}/markets",
            params={
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
                "limit": 30,
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
    except httpx.HTTPError:
        return []
    arr = r.json() or []
    if not isinstance(arr, list):
        return []
    out: list[QuoteSimilarMarket] = []
    needle = (theme_keyword or "").lower().strip()
    for m in arr:
        slug = m.get("slug")
        if not slug or slug == exclude_slug:
            continue
        title = str(m.get("question") or "")
        # Cheap "same theme" heuristic: shared keyword in title.
        if needle and needle not in title.lower():
            continue
        out.append(
            QuoteSimilarMarket(
                slug=str(slug),
                title=title,
                theme=needle or None,
                price=_safe_float(m.get("lastTradePrice"))
                or terminal_mod._yes_price_from_market(m),
                volume_24hr=_safe_float(m.get("volume24hr")),
            )
        )
        if len(out) >= limit:
            break
    return out


# --- compute helpers --------------------------------------------------------


def _half_life_ar1(prob_series: pd.Series) -> float | None:
    """AR(1) half-life on Δlogit — same convention as terminal_compare."""
    s = _logit(prob_series.dropna()).diff().dropna()
    if len(s) < 10:
        return None
    lvl = s.cumsum()
    x = lvl.shift(1).dropna()
    dy = lvl.diff().dropna()
    common = x.index.intersection(dy.index)
    if len(common) < 10:
        return None
    x_arr = x.loc[common].to_numpy()
    dy_arr = dy.loc[common].to_numpy()
    if x_arr.std() == 0:
        return None
    X = np.column_stack([np.ones_like(x_arr), x_arr])
    try:
        coef, *_ = np.linalg.lstsq(X, dy_arr, rcond=None)
    except np.linalg.LinAlgError:
        return None
    b = float(coef[1])
    if not math.isfinite(b) or b >= 0 or b <= -2:
        return None
    try:
        hl = math.log(0.5) / math.log(1.0 + b)
    except (ValueError, ZeroDivisionError):
        return None
    return hl if math.isfinite(hl) and hl > 0 else None


def _build_stats(price_series: pd.Series) -> QuoteStats:
    s = price_series.dropna().astype(float)
    n_obs = len(s)
    if n_obs < 5:
        return QuoteStats(n_obs=n_obs)
    innov = _logit(s).diff().dropna()
    rv30: float | None = None
    if len(innov) >= 5:
        tail = innov.iloc[-30:] if len(innov) >= 30 else innov
        sd = float(tail.std(ddof=1)) if len(tail) > 1 else float("nan")
        if math.isfinite(sd):
            rv30 = sd
    # Reuse the canonical compute path for DFA / variance-ratio.
    extra = terminal_mod.compute_stats_from_series(s)
    dfa = extra.get("dfa_alpha")
    return QuoteStats(
        n_obs=n_obs,
        rv_30d=rv30,
        half_life=_half_life_ar1(s),
        hurst=float(dfa) if isinstance(dfa, (int, float)) and math.isfinite(dfa) else None,
        dfa_alpha=float(dfa) if isinstance(dfa, (int, float)) and math.isfinite(dfa) else None,
    )


def _build_ranges(daily: pd.Series, intraday: pd.Series) -> tuple[QuoteRange, QuoteRange]:
    """Compute ``day_range`` (intraday last 24h) and ``week52_range`` (daily)."""
    if not intraday.empty:
        cutoff = intraday.index.max() - pd.Timedelta(hours=24)
        last_24h = intraday[intraday.index >= cutoff]
        if not last_24h.empty:
            day_low = float(last_24h.min())
            day_high = float(last_24h.max())
        else:
            day_low = day_high = None
    else:
        day_low = day_high = None

    if not daily.empty:
        cutoff = daily.index.max() - pd.Timedelta(days=365)
        window = daily[daily.index >= cutoff]
        if not window.empty:
            wk52_low = float(window.min())
            wk52_high = float(window.max())
        else:
            wk52_low = float(daily.min())
            wk52_high = float(daily.max())
    else:
        wk52_low = wk52_high = None

    return (
        QuoteRange(low=day_low, high=day_high),
        QuoteRange(low=wk52_low, high=wk52_high),
    )


# --- composer ---------------------------------------------------------------


async def _build_quote(
    slug: str,
    days: int,
    includes: list[str],
    *,
    gamma_url: str,
    clob_url: str,
    http: httpx.AsyncClient | None = None,
) -> TerminalQuoteResponse:
    """Fan-out fetches + assemble a :class:`TerminalQuoteResponse`.

    When ``http`` is supplied (typically ``request.app.state.async_http``)
    the fan-out reuses the shared keepalive pool. The fallback path opens
    a private short-lived client for ad-hoc / test callers.
    """
    async with AsyncExitStack() as stack:
        if http is None:
            http = await stack.enter_async_context(httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS))
        # Phase 1: fetch the gamma market — every other branch needs the
        # token id and theme/title from this one payload.
        market = await _fetch_gamma_market_async(http, gamma_url, slug)
        token_id = _yes_token_id(market)
        meta_dict = terminal_mod.shape_meta(market)
        theme = meta_dict.get("theme")
        question = meta_dict.get("question") or ""

        # Phase 2: fan-out everything that depends only on `slug`/`token_id`/`theme`.
        async def _daily() -> pd.DataFrame:
            if not token_id:
                return pd.DataFrame(columns=["price"])
            try:
                return await _fetch_clob_history_async(
                    http, clob_url, token_id, days=max(days, 365), fidelity=1440
                )
            except httpx.HTTPError as e:
                logger.warning("daily history failed for %s: %s", slug, e)
                return pd.DataFrame(columns=["price"])

        async def _intraday() -> pd.DataFrame:
            if not token_id:
                return pd.DataFrame(columns=["price"])
            try:
                return await _fetch_clob_history_async(
                    http, clob_url, token_id, days=2, fidelity=60
                )
            except httpx.HTTPError as e:
                logger.warning("intraday history failed for %s: %s", slug, e)
                return pd.DataFrame(columns=["price"])

        async def _news() -> list[QuoteNewsItem]:
            if "news" not in includes:
                return []
            return await _fetch_news_for_slug(http, question, limit=5)

        async def _similar() -> list[QuoteSimilarMarket]:
            if "similar" not in includes:
                return []
            # Theme is rarely set on raw markets; fall back to first long word
            # in the question for a useful "same topic" filter.
            keyword = theme
            if not keyword:
                import re

                toks = list(re.findall(r"[A-Za-z]{5,}", question))
                keyword = toks[0].lower() if toks else None
            return await _fetch_similar_markets(
                http, gamma_url, theme_keyword=keyword, exclude_slug=slug, limit=3
            )

        daily_df, intraday_df, news_items, similar_items = await asyncio.gather(
            _daily(), _intraday(), _news(), _similar()
        )

    # Phase 3 (sync): peers come from the on-disk alpha-hunter sweep so we
    # don't need an HTTP client. Run after the gather to keep the async
    # block tight.
    peers_out: list[QuotePeer] = []
    if "peers" in includes:
        try:
            raw_peers = terminal_mod.find_peers(slug, top_n=5)
        except (OSError, ValueError) as e:  # pragma: no cover — defensive
            logger.warning("peer load failed for %s: %s", slug, e)
            raw_peers = []
        for p in raw_peers[:5]:
            peers_out.append(
                QuotePeer(
                    slug=str(p.get("peer_id") or ""),
                    name=str(p.get("peer_id") or ""),
                    correlation=_safe_float(p.get("oos_sharpe")),
                    last_change_pct=None,
                    sparkline_7d=[],
                )
            )

    daily_series = (
        daily_df["price"].astype(float) if "price" in daily_df.columns else pd.Series(dtype=float)
    )
    intraday_series = (
        intraday_df["price"].astype(float)
        if "price" in intraday_df.columns
        else pd.Series(dtype=float)
    )

    # Live block — prefer fresh intraday close where available.
    live_dict = terminal_mod.shape_live(market)
    current_price = (
        float(intraday_series.iloc[-1])
        if not intraday_series.empty
        else live_dict.get("midpoint") or live_dict.get("last_trade_price")
    )

    stats = _build_stats(daily_series)
    day_range, week52_range = _build_ranges(daily_series, intraday_series)
    implied_vol = (
        stats.rv_30d * math.sqrt(365.0)
        if stats.rv_30d is not None and math.isfinite(stats.rv_30d)
        else None
    )

    spark_30d = (
        [float(v) for v in daily_series.iloc[-30:].tolist()] if not daily_series.empty else []
    )
    if not intraday_series.empty:
        cutoff = intraday_series.index.max() - pd.Timedelta(hours=24)
        spark_intraday = [
            float(v) for v in intraday_series[intraday_series.index >= cutoff].tolist()
        ]
    else:
        spark_intraday = []

    holders = _holders_from_enriched_orderbook(market)

    live = QuoteLive(
        price=current_price,
        best_bid=live_dict.get("best_bid"),
        best_ask=live_dict.get("best_ask"),
        spread_cents=live_dict.get("spread_cents"),
        change_24h=live_dict.get("one_day_price_change"),
        change_7d=live_dict.get("one_week_price_change"),
        volume_24hr=live_dict.get("volume_24hr"),
        last_trade_price=live_dict.get("last_trade_price"),
    )
    meta = QuoteMeta(
        slug=slug,
        title=question,
        theme=theme,
        end_date=meta_dict.get("end_date"),
        days_to_resolve=meta_dict.get("days_to_resolve"),
        total_volume=live_dict.get("volume_total"),
        total_open_interest=_safe_float(market.get("openInterest")),
    )

    return TerminalQuoteResponse(
        slug=slug,
        days=days,
        includes=includes,
        live=live,
        meta=meta,
        stats=stats,
        day_range=day_range,
        week52_range=week52_range,
        implied_vol=implied_vol,
        holders_estimate=holders,
        sparkline_30d=spark_30d,
        sparkline_intraday=spark_intraday,
        peers=peers_out,
        news=news_items,
        similar_markets=similar_items,
    )


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-quote"])


@router.get("/quote/{slug}", response_model=TerminalQuoteResponse)
async def get_quote(
    request: Request,
    slug: Annotated[str, FPath(min_length=1, max_length=200)],
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
    include: Annotated[str, Query(max_length=200)] = "peers,news,similar",
) -> TerminalQuoteResponse:
    """Composed quote-page payload.

    All sub-fetches (gamma market, CLOB daily history, CLOB intraday history,
    news, similar markets) are issued in parallel using
    :func:`asyncio.gather` against the shared
    ``request.app.state.async_http`` keepalive pool. The peer list comes
    from the in-process alpha-hunter sweep cache and so is computed
    synchronously after the HTTP fan-out.
    """
    includes = _parse_includes(include)
    cache_key = (slug, int(days), tuple(sorted(includes)))
    cached = _QUOTE_CACHE.get(cache_key)
    if cached is not None:
        return TerminalQuoteResponse.model_validate(cached)

    settings: Settings = get_settings()
    gamma_url = settings.polymarket_gamma_url
    clob_url = settings.polymarket_clob_url
    shared_http: httpx.AsyncClient | None = getattr(request.app.state, "async_http", None)

    try:
        resp = await _build_quote(
            slug,
            int(days),
            includes,
            gamma_url=gamma_url,
            clob_url=clob_url,
            http=shared_http,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e

    _QUOTE_CACHE.set(cache_key, resp.model_dump(), ttl=CACHE_TTL_SECONDS)
    return resp


__all__ = [
    "ALLOWED_INCLUDES",
    "CACHE_TTL_SECONDS",
    "QuoteLive",
    "QuoteMeta",
    "QuoteNewsItem",
    "QuotePeer",
    "QuoteRange",
    "QuoteSimilarMarket",
    "QuoteStats",
    "TerminalQuoteResponse",
    "clear_cache",
    "get_quote",
    "router",
]
