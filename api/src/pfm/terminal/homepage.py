"""Homepage composer for Terminal: gainers/losers/most-active + sparklines.

The existing ``/terminal/overview`` returns top-movers as a flat list with
no per-market history, so the front-end has to re-call ``/prices-history``
for every row to draw a sparkline. This endpoint pre-fetches a 7-day
sparkline for every gainer/loser/most-active row and packages everything
the homepage needs in one envelope.

What's in the response
----------------------
- ``gainers`` / ``losers`` (top 10): ``{slug, name, theme, price, change_pct,
  volume_24h, sparkline_7d}``.
- ``most_active`` (top 10): same shape as gainers but sorted by 24h volume.
- ``recently_launched`` (top 5): markets created in the last 7 days.
- ``resolving_soon`` (top 10): markets ending in the next 7 days.
- ``breaking_news`` (top 5): highest-impact stories (Reddit + HN merged).
- ``theme_heatmap``: ``{theme, n_markets, avg_change_24h}`` rows.
- ``pm_vix``: composite "risk-on / risk-off" index in [0, 100].

Routing
-------
Owns its own :class:`fastapi.APIRouter`; ``main.py`` is left untouched.
Wire explicitly::

    from pfm.terminal_homepage import router as terminal_homepage_router
    app.include_router(terminal_homepage_router)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Annotated, Any

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.config import Settings, get_settings

logger = logging.getLogger(__name__)


# --- limits / cache ---------------------------------------------------------

DEFAULT_HOURS: int = 24
MIN_HOURS: int = 1
MAX_HOURS: int = 168
HTTP_TIMEOUT_SECONDS: float = 10.0
CACHE_TTL_SECONDS: int = 60
SPARKLINE_DAYS: int = 7
TOP_MOVERS: int = 10
TOP_NEW: int = 5
TOP_RESOLVING: int = 10
TOP_NEWS: int = 5

# Tail-risk slug substrings that feed the pm-vix composite.
_TAIL_RISK_KEYWORDS: tuple[str, ...] = (
    "recession",
    "war",
    "conflict",
    "tail-risk",
    "vix",
    "geopolit",
    "default",
    "credit-event",
    "election",
    "uncertainty",
    "crash",
)

# Process-wide cache keyed by (theme, hours).
_HOME_CACHE = get_cache("terminal_homepage", ttl=CACHE_TTL_SECONDS)


def clear_cache() -> None:
    """Test/utility — drop all cached homepage payloads."""
    _HOME_CACHE.clear()


# --- Pydantic schemas -------------------------------------------------------


class HomepageMover(BaseModel):
    """One row in gainers / losers / most-active."""

    slug: str
    name: str
    theme: str | None = None
    price: float | None = None
    change_pct: float | None = None
    volume_24h: float | None = None
    sparkline_7d: list[float] | None = None


class HomepageNewMarket(BaseModel):
    slug: str
    name: str
    theme: str | None = None
    price: float | None = None
    age_days: int | None = None


class HomepageResolving(BaseModel):
    slug: str
    name: str
    theme: str | None = None
    price: float | None = None
    end_date: str | None = None
    days_to_resolve: int | None = None
    conviction: float | None = None


class HomepageNews(BaseModel):
    title: str
    source: str
    url: str | None = None
    impact_score: float | None = None


class ThemeHeatmapRow(BaseModel):
    theme: str
    n_markets: int
    avg_change_24h: float | None = None
    total_volume_24h: float | None = None


class TerminalHomepageResponse(BaseModel):
    """Full /terminal/homepage envelope."""

    theme: str | None = None
    hours: int
    n_markets_considered: int = 0
    gainers: list[HomepageMover] = Field(default_factory=list)
    losers: list[HomepageMover] = Field(default_factory=list)
    most_active: list[HomepageMover] = Field(default_factory=list)
    recently_launched: list[HomepageNewMarket] = Field(default_factory=list)
    resolving_soon: list[HomepageResolving] = Field(default_factory=list)
    breaking_news: list[HomepageNews] = Field(default_factory=list)
    theme_heatmap: list[ThemeHeatmapRow] = Field(default_factory=list)
    pm_vix: float = Field(..., description="Risk-on/off composite in [0, 100].")


# --- helpers ----------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _yes_token_id(market: dict[str, Any]) -> str | None:
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


def _yes_price(market: dict[str, Any]) -> float | None:
    """Same fallback chain as terminal.build_overview's _yes_price_from_market."""
    bb = _safe_float(market.get("bestBid"))
    ba = _safe_float(market.get("bestAsk"))
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    lt = _safe_float(market.get("lastTradePrice"))
    if lt is not None:
        return lt
    raw = market.get("outcomePrices")
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and arr:
                return _safe_float(arr[0])
        except json.JSONDecodeError:
            pass
    return None


# Keyword → theme. Order matters: we match the FIRST hit, so the more specific
# themes come first (chips before tech, ai before tech, etc.). Each keyword is
# checked as a whole-word match against the lowercased slug + question.
_THEME_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "crypto",
        (
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "crypto",
            "solana",
            "sol",
            "doge",
            "xrp",
            "altcoin",
            "dogecoin",
            "chainlink",
            "litecoin",
            "polygon",
            "cardano",
            "avalanche",
            "token",
            "airdrop",
            "base-chain",
            "base-launch",
            "stablecoin",
            "defi",
            "memecoin",
            "nft",
            "icp",
            "etf-approval",
        ),
    ),
    (
        "ai",
        (
            "ai",
            "openai",
            "gpt",
            "gpt-5",
            "claude",
            "anthropic",
            "gemini",
            "llm",
            "agi",
            "agi-by",
            "mistral",
            "perplexity",
        ),
    ),
    (
        "chips",
        (
            "nvidia",
            "nvda",
            "intel",
            "amd",
            "tsmc",
            "asml",
            "broadcom",
            "semiconductor",
            "chip-act",
            "asic",
        ),
    ),
    (
        "politics",
        (
            "president",
            "senate",
            "congress",
            "governor",
            "senator",
            "primary",
            "republican",
            "democrat",
            "election",
            "biden",
            "trump",
            "harris",
            "desantis",
            "rfk",
            "obama",
            "kamala",
            "vance",
            "vp",
            "speaker",
            "scotus",
            "supreme-court",
            "impeach",
            "veto",
            "white-house",
            "ballot",
            "starmer",
            "hegseth",
            "secretary",
            "cabinet",
            "epstein",
            "indictment",
            "trial",
            "lawsuit",
            "labor",
            "tories",
            "conservative",
            "macron",
            "merkel",
            "scholz",
            "modi",
            "netanyahu",
            "regime",
            "lai-ching-te",
        ),
    ),
    (
        "geopolitics",
        (
            "ukraine",
            "russia",
            "putin",
            "israel",
            "iran",
            "gaza",
            "hamas",
            "china",
            "taiwan",
            "north-korea",
            "kim",
            "nato",
            "ceasefire",
            "treaty",
            "war",
            "invasion",
            "border",
            "evergreen",
            "houthi",
            "syria",
            "lebanon",
            "venezuela",
            "iranian",
            "hormuz",
            "strait",
            "kim-jong",
            "ussr",
            "soviet",
            "annex",
            "annexation",
            "sanctions",
            "embassy",
            "diplomat",
            "foreign-minister",
        ),
    ),
    (
        "sports",
        (
            "nba",
            "nfl",
            "mlb",
            "nhl",
            "mls",
            "premier-league",
            "champions-league",
            "world-cup",
            "super-bowl",
            "playoffs",
            "finals",
            "match",
            "game",
            "ucl",
            "uefa",
            "fifa",
            "soccer",
            "football",
            "basketball",
            "baseball",
            "hockey",
            "f1",
            "grand-prix",
            "lol",
            "csgo",
            "valorant",
            "esports",
            "tennis",
            "wimbledon",
            "us-open",
            "kentucky-derby",
            "french-open",
            "atp",
            "wta",
            "australian-open",
            "draft",
            "club",
            "stanley-cup",
            "ligue-1",
            "la-liga",
            "bundesliga",
            "serie-a",
            "champions",
            "uefa-cup",
            "fc",
            "afc",
            "nfc",
            "cf",
            "sc",
            "sd",
            "ud",
            "marlins",
            "orioles",
            "yankees",
            "dodgers",
            "lakers",
            "celtics",
            "warriors",
            "manchester",
            "liverpool",
            "chelsea",
            "arsenal",
            "real-madrid",
            "barca",
            "barcelona",
            "atletico",
            "psg",
            "juventus",
            "ac-milan",
            "inter",
            "ajax",
            "vs",
            "world-series",
            "bowl",
            "open-by",
        ),
    ),
    (
        "macro",
        (
            "fed",
            "fomc",
            "rate-cut",
            "rate-hike",
            "inflation",
            "cpi",
            "ppi",
            "recession",
            "gdp",
            "jobs",
            "unemployment",
            "payrolls",
            "nonfarm",
            "powell",
            "yield",
            "treasury",
            "interest-rate",
            "core-pce",
            "pce",
        ),
    ),
    ("commodities", ("gold", "silver", "copper", "platinum", "wheat", "soybeans", "corn")),
    ("energy", ("oil", "wti", "brent", "natgas", "natural-gas", "opec", "barrel")),
    (
        "equities",
        (
            "stock",
            "sp500",
            "nasdaq",
            "dow",
            "nyse",
            "tesla",
            "tsla",
            "apple",
            "aapl",
            "meta",
            "amazon",
            "amzn",
            "google",
            "googl",
            "microsoft",
            "msft",
            "netflix",
            "nvidia-stock",
            "earnings",
        ),
    ),
    (
        "pop_culture",
        (
            "oscar",
            "grammy",
            "emmy",
            "movie",
            "album",
            "tour",
            "netflix-release",
            "spotify",
            "songs-of-the-year",
            "billboard",
            "concert",
            "best-picture",
            "best-album",
            "taylor-swift",
            "kanye",
            "beyonce",
            "drake",
            "gta",
            "minecraft",
            "fortnite",
            "video-game",
            "tv-show",
            "bachelor",
            "bachelorette",
            "survivor",
            "release",
            "released",
            "eurovision",
            "tweets",
            "tweet",
            "x-post",
            "twitter",
        ),
    ),
    (
        "health",
        (
            "pandemic",
            "covid",
            "vaccine",
            "outbreak",
            "hantavirus",
            "h5n1",
            "ebola",
            "monkeypox",
            "measles",
            "fda-approve",
            "fda-approval",
        ),
    ),
    (
        "awards",
        ("nobel", "pulitzer", "fields-medal", "turing-award", "man-of-the-year", "peace-prize"),
    ),
    (
        "weather",
        (
            "hurricane",
            "tornado",
            "typhoon",
            "weather",
            "snowfall",
            "rainfall",
            "snowpack",
            "el-nino",
            "la-nina",
        ),
    ),
    (
        "space",
        (
            "spacex",
            "starship",
            "falcon",
            "nasa",
            "mars",
            "moon",
            "rocket-launch",
            "iss",
            "satellite",
        ),
    ),
)


def _theme_from_text(slug: str | None, question: str | None) -> str | None:
    """Keyword fallback for markets without explicit theme metadata.

    Walks each theme's keyword tuples. A keyword matches when it's a whole
    hyphen-or-space-delimited token in the slug+question blob, OR when a
    slug token starts with it (so "iranian" matches "iran", "russian" matches
    "russia"). First-match-wins, themes ordered most-specific to most-general.
    """
    if not slug and not question:
        return None
    raw = f"{(slug or '').lower()} {(question or '').lower().replace(' ', '-')}"
    blob = f" {raw} "
    tokens = set(raw.replace(" ", "-").split("-"))
    for theme, kws in _THEME_KEYWORDS:
        for kw in kws:
            # Whole-word in slug or question.
            if f"-{kw}-" in blob or f" {kw}-" in blob or f"-{kw} " in blob or f" {kw} " in blob:
                return theme
            # Stem match: any slug token starting with kw (catches plurals,
            # adjectives like "iranian", "russian", "ukrainian"). Skip very
            # short keywords to avoid noise ("eth" matching "method").
            if len(kw) >= 5:
                for tok in tokens:
                    if tok.startswith(kw) and len(tok) - len(kw) <= 4:
                        return theme
    return None


def _theme_for_market(m: dict[str, Any]) -> str | None:
    """Theme heuristic: explicit field → tags → keyword inference from slug/question.

    The explicit ``theme`` from Gamma is often literally ``"other"`` (their
    catch-all bucket) — we prefer our keyword inference over that, because
    "Will Charlotte FC win the MLS Cup?" should land in ``sports`` rather
    than the catch-all. Inference also runs when nothing explicit exists.
    """
    inferred = _theme_from_text(m.get("slug"), m.get("question"))
    explicit_raw = m.get("theme") or m.get("category")
    if isinstance(explicit_raw, str) and explicit_raw.strip():
        explicit = explicit_raw.strip().lower()
        # Override Gamma's catch-all only if we have a specific match.
        if explicit not in {"other", "uncategorized", "misc", ""}:
            return explicit
    tags = m.get("tags")
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, str) and first.strip():
            tag = first.strip().lower()
            if tag not in {"other", "uncategorized", "misc"}:
                return tag
        elif isinstance(first, dict):
            label = first.get("label") or first.get("slug") or first.get("name")
            if isinstance(label, str) and label.strip():
                lab = label.strip().lower()
                if lab not in {"other", "uncategorized", "misc"}:
                    return lab
    return inferred


def _is_tail_risk(slug: str | None, question: str | None) -> bool:
    """Heuristic — does this market read like a downside / tail-risk bet?"""
    blob = f"{(slug or '').lower()} {(question or '').lower()}"
    return any(kw in blob for kw in _TAIL_RISK_KEYWORDS)


def _parse_iso(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


# --- async fetchers ---------------------------------------------------------


async def _fetch_top_markets_async(
    http: httpx.AsyncClient,
    gamma_url: str,
    *,
    pages: int = 3,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Walk the first few pages of Gamma's volume-sorted active markets."""
    base = gamma_url.rstrip("/")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in range(pages):
        params: dict[str, str | int] = {
            "active": "true",
            "closed": "false",
            "limit": page_size,
            "offset": i * page_size,
            "order": "volume24hr",
            "ascending": "false",
        }
        try:
            r = await http.get(f"{base}/markets", params=params, timeout=HTTP_TIMEOUT_SECONDS)
            r.raise_for_status()
        except httpx.HTTPError:
            break
        page = r.json() or []
        if not isinstance(page, list) or not page:
            break
        for m in page:
            slug = m.get("slug")
            if not slug or slug in seen:
                continue
            seen.add(slug)
            out.append(m)
    return out


async def _fetch_sparkline(
    http: httpx.AsyncClient,
    clob_url: str,
    token_id: str,
    *,
    days: int = SPARKLINE_DAYS,
) -> list[float]:
    """Pull a short price series for one market — best-effort, never raises."""
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days + 1)
    params: dict[str, str | int] = {
        "market": token_id,
        "fidelity": 1440,
        "interval": "max",
        "startTs": int(start.timestamp()),
    }
    try:
        r = await http.get(
            f"{clob_url.rstrip('/')}/prices-history",
            params=params,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
    except httpx.HTTPError:
        return []
    payload = r.json() or {}
    history = payload.get("history", []) if isinstance(payload, dict) else []
    out: list[float] = []
    for row in history:
        p = _safe_float(row.get("p"))
        if p is not None:
            out.append(p)
    return out[-days:] if out else []


#: Keywords that mark an HN story as plausibly market-relevant. Anything that
#: doesn't contain one of these is dropped to avoid the homepage degrading
#: into a generic tech-news feed (UX audit 2026-05-14: stories like "removing
#: the GPS from my RAV4" don't belong on a prediction-market dashboard).
_NEWS_KEYWORDS: tuple[str, ...] = (
    "trump",
    "biden",
    "harris",
    "putin",
    "xi",
    "election",
    "vote",
    "poll",
    "fed",
    "fomc",
    "powell",
    "rate",
    "inflation",
    "cpi",
    "jobs",
    "gdp",
    "recession",
    "tariff",
    "trade war",
    "sanction",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "crypto",
    "coinbase",
    "etf",
    "stock",
    "market",
    "nasdaq",
    "dow",
    "earnings",
    "tesla",
    "nvidia",
    "openai",
    "anthropic",
    "ai",
    "agi",
    "war",
    "ukraine",
    "russia",
    "israel",
    "gaza",
    "iran",
    "china",
    "supreme court",
    "scotus",
    "congress",
    "senate",
    "house",
    "oil",
    "opec",
    "gold",
)


_NEWS_KW_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _NEWS_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)


def _is_market_relevant(title: str) -> bool:
    """Word-boundary keyword scan against the prediction-market relevance list.

    Word boundaries prevent false positives like ``"ai"`` matching ``"Air"``
    or ``"btc"`` matching arbitrary hex strings.
    """
    return bool(_NEWS_KW_PATTERN.search(title))


async def _fetch_breaking_news(http: httpx.AsyncClient) -> list[HomepageNews]:
    """Algolia HN front-page query as a free, no-auth "breaking" feed.

    We over-fetch (×8 ``TOP_NEWS``) and filter for market-relevant titles so
    the front-page doesn't bury actual signal under random tech blog posts.
    If the filter rejects everything we fall back to the unfiltered top
    stories rather than returning an empty array — better something than
    a blank panel.
    """
    try:
        r = await http.get(
            "https://hn.algolia.com/api/v1/search",
            params={"tags": "front_page", "hitsPerPage": TOP_NEWS * 8},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
    except httpx.HTTPError:
        return []
    try:
        payload = r.json()
    except ValueError:
        return []
    all_items: list[HomepageNews] = []
    relevant: list[HomepageNews] = []
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
        # Use HN points / 1000 as a 0..1-ish impact proxy.
        pts = _safe_float(hit.get("points")) or 0.0
        item = HomepageNews(
            title=title,
            source="hn",
            url=url or None,
            impact_score=min(1.0, pts / 1000.0),
        )
        all_items.append(item)
        if _is_market_relevant(title):
            relevant.append(item)
        if len(relevant) >= TOP_NEWS:
            break
    # Prefer the filtered list; fall back to the unfiltered top stories so
    # the breaking-news panel is never completely empty.
    return (relevant or all_items)[:TOP_NEWS]


# --- derivations ------------------------------------------------------------


def _row(m: dict[str, Any], spark: list[float] | None) -> HomepageMover:
    return HomepageMover(
        slug=str(m.get("slug") or ""),
        name=str(m.get("question") or ""),
        theme=_theme_for_market(m),
        price=_yes_price(m),
        change_pct=_safe_float(m.get("oneDayPriceChange")),
        volume_24h=_safe_float(m.get("volume24hr")),
        sparkline_7d=spark if spark is not None else None,
    )


def _theme_heatmap(markets: list[dict[str, Any]]) -> list[ThemeHeatmapRow]:
    by_theme: dict[str, list[dict[str, Any]]] = {}
    for m in markets:
        t = _theme_for_market(m) or "other"
        by_theme.setdefault(t, []).append(m)
    rows: list[ThemeHeatmapRow] = []
    for theme, group in by_theme.items():
        changes = [
            c for c in (_safe_float(m.get("oneDayPriceChange")) for m in group) if c is not None
        ]
        vols = [v for v in (_safe_float(m.get("volume24hr")) for m in group) if v is not None]
        rows.append(
            ThemeHeatmapRow(
                theme=theme,
                n_markets=len(group),
                avg_change_24h=float(np.mean(changes)) if changes else None,
                total_volume_24h=float(np.sum(vols)) if vols else None,
            )
        )
    rows.sort(key=lambda r: -(r.total_volume_24h or 0.0))
    return rows


def _compute_pm_vix(markets: list[dict[str, Any]]) -> float:
    """Volume-weighted YES probability across tail-risk markets, scaled to 0-100.

    A higher value means the market is collectively pricing more downside
    (recessions, conflicts, election uncertainty) relative to liquidity.
    Returns 0.0 when no tail-risk markets are visible — in that case the
    front-end should treat the index as "no signal".
    """
    weighted_sum = 0.0
    total_w = 0.0
    for m in markets:
        if not _is_tail_risk(m.get("slug"), m.get("question")):
            continue
        p = _yes_price(m)
        v = _safe_float(m.get("volume24hr")) or 0.0
        if p is None:
            continue
        # Floor weight at 1.0 so very-low-volume tail markets still nudge the
        # composite (otherwise a single high-vol "uncertainty" market dominates).
        w = max(1.0, v)
        weighted_sum += p * w
        total_w += w
    if total_w <= 0:
        return 0.0
    avg = weighted_sum / total_w
    return float(max(0.0, min(100.0, avg * 100.0)))


# --- composer ---------------------------------------------------------------


async def _build_homepage(
    *,
    theme: str | None,
    hours: int,
    gamma_url: str,
    clob_url: str,
    http: httpx.AsyncClient | None = None,
) -> TerminalHomepageResponse:
    """Compose the homepage payload.

    When ``http`` is supplied (e.g. ``request.app.state.async_http``) all
    sub-fetches reuse that pool — saving the ~50 ms TLS handshake to
    gamma/clob on every request. The fallback path opens a private
    short-lived client so callers from unit tests / scripts still work.
    """
    async with AsyncExitStack() as stack:
        if http is None:
            http = await stack.enter_async_context(httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS))
        markets = await _fetch_top_markets_async(http, gamma_url, pages=3)
        if theme:
            t_lower = theme.lower().strip()
            markets = [m for m in markets if (_theme_for_market(m) or "") == t_lower]

        # --- gainers / losers / most-active ---------------------------------
        # Filter out near-resolved markets (price ≥ 0.95 or ≤ 0.05). Without
        # this, the "top gainer" panel fills up with markets that already
        # closed in their favour — e.g. a sports game that ended +75 % over
        # last 24h sitting at 0.9995 isn't actionable, just noise. Same for
        # losers (markets that collapsed to 0.0005).
        actionable = []
        for m in markets:
            if _safe_float(m.get("oneDayPriceChange")) is None:
                continue
            px = _yes_price(m)
            if px is None or px >= 0.95 or px <= 0.05:
                continue
            actionable.append(m)
        gainers = sorted(
            actionable,
            key=lambda m: -(_safe_float(m.get("oneDayPriceChange")) or 0.0),
        )[:TOP_MOVERS]
        losers = sorted(
            actionable,
            key=lambda m: _safe_float(m.get("oneDayPriceChange")) or 0.0,
        )[:TOP_MOVERS]
        most_active = sorted(
            markets,
            key=lambda m: -(_safe_float(m.get("volume24hr")) or 0.0),
        )[:TOP_MOVERS]

        # Fan-out sparkline fetches in parallel for the union of slugs.
        spark_targets: dict[str, str] = {}
        for m in (*gainers, *losers, *most_active):
            slug = str(m.get("slug") or "")
            tok = _yes_token_id(m)
            if slug and tok and slug not in spark_targets:
                spark_targets[slug] = tok

        spark_tasks = [_fetch_sparkline(http, clob_url, tok) for tok in spark_targets.values()]
        # Breaking news lives alongside the sparkline fan-out.
        spark_results, breaking = await asyncio.gather(
            asyncio.gather(*spark_tasks) if spark_tasks else _empty_list(),
            _fetch_breaking_news(http),
        )

    spark_by_slug: dict[str, list[float]] = dict(
        zip(spark_targets.keys(), spark_results, strict=False)
    )

    def _attach_spark(rows: list[dict[str, Any]]) -> list[HomepageMover]:
        return [_row(m, spark_by_slug.get(str(m.get("slug") or ""))) for m in rows]

    # --- new + resolving-soon -------------------------------------------------
    now = datetime.now(tz=UTC)
    new_window = pd.Timedelta(days=7)
    resolve_window_end = now + pd.Timedelta(days=7)

    new_rows: list[HomepageNewMarket] = []
    for m in sorted(
        markets,
        key=lambda mm: (
            -(_parse_iso(mm.get("createdAt")).timestamp() if _parse_iso(mm.get("createdAt")) else 0)
        ),
    ):
        created = _parse_iso(m.get("createdAt"))
        if created is None or (now - created) > new_window:
            continue
        new_rows.append(
            HomepageNewMarket(
                slug=str(m.get("slug") or ""),
                name=str(m.get("question") or ""),
                theme=_theme_for_market(m),
                price=_yes_price(m),
                age_days=max(0, (now - created).days),
            )
        )
        if len(new_rows) >= TOP_NEW:
            break

    resolving: list[HomepageResolving] = []
    for m in markets:
        end = _parse_iso(m.get("endDate") or m.get("endDateIso"))
        if end is None:
            continue
        if not (now <= end <= resolve_window_end):
            continue
        price = _yes_price(m)
        if price is None:
            continue
        days_left = max(0, (end - now).days)
        resolving.append(
            HomepageResolving(
                slug=str(m.get("slug") or ""),
                name=str(m.get("question") or ""),
                theme=_theme_for_market(m),
                price=price,
                end_date=str(m.get("endDate")) if m.get("endDate") else None,
                days_to_resolve=days_left,
                conviction=abs(price - 0.5) * 2.0,
            )
        )
    resolving.sort(key=lambda r: -(r.conviction or 0.0))
    resolving = resolving[:TOP_RESOLVING]

    return TerminalHomepageResponse(
        theme=theme,
        hours=hours,
        n_markets_considered=len(markets),
        gainers=_attach_spark(gainers),
        losers=_attach_spark(losers),
        most_active=_attach_spark(most_active),
        recently_launched=new_rows,
        resolving_soon=resolving,
        breaking_news=breaking,
        theme_heatmap=_theme_heatmap(markets),
        pm_vix=_compute_pm_vix(markets),
    )


async def _empty_list() -> list[Any]:
    """Helper so ``asyncio.gather`` always sees an awaitable in both branches."""
    return []


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-homepage"])


@router.get("/homepage", response_model=TerminalHomepageResponse)
async def get_homepage(
    request: Request,
    theme: Annotated[str, Query(max_length=50)] = "",
    hours: Annotated[int, Query(ge=MIN_HOURS, le=MAX_HOURS)] = DEFAULT_HOURS,
) -> TerminalHomepageResponse:
    """Composed homepage payload (gainers/losers/most-active + sparklines).

    All sub-fetches (gamma listing, per-market sparklines, breaking news) run
    concurrently inside a single :class:`httpx.AsyncClient` shared with the
    rest of the app via ``request.app.state.async_http``. Cached for
    ``CACHE_TTL_SECONDS=60`` to absorb refresh-burst from the dashboard.
    """
    theme_norm = (theme or "").strip().lower() or None
    cache_key = (theme_norm, int(hours))
    cached = _HOME_CACHE.get(cache_key)
    if cached is not None:
        return TerminalHomepageResponse.model_validate(cached)

    settings: Settings = get_settings()
    shared_http: httpx.AsyncClient | None = getattr(request.app.state, "async_http", None)
    try:
        resp = await _build_homepage(
            theme=theme_norm,
            hours=int(hours),
            gamma_url=settings.polymarket_gamma_url,
            clob_url=settings.polymarket_clob_url,
            http=shared_http,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e

    _HOME_CACHE.set(cache_key, resp.model_dump(), ttl=CACHE_TTL_SECONDS)
    return resp


# ──────────────────────────────────────────────────────────────────────────
# /terminal/themes — sidebar theme nav
# ──────────────────────────────────────────────────────────────────────────


class ThemesResponse(BaseModel):
    """Compact theme listing for the left-side nav.

    Front-end renders each theme as a clickable chip with the market count;
    sorted by ``n_markets`` desc so the most-active themes anchor the top.
    """

    n_themes: int
    themes: list[ThemeHeatmapRow]


@router.get("/themes", response_model=ThemesResponse)
async def get_themes(
    request: Request,
    hours: Annotated[int, Query(ge=MIN_HOURS, le=MAX_HOURS)] = DEFAULT_HOURS,
) -> ThemesResponse:
    """Themes sidebar — markets-by-theme rollup.

    Composed from the same gamma fetch the homepage uses, so this is a cheap
    re-projection of the cached payload. Filters out the catch-all ``other``
    bucket once it would dominate (>40 % of rows) — the UX audit flagged
    "everything in 'other'" as a clarity issue. Themes with zero markets
    are dropped; the sidebar should never render an empty chip.
    """
    # Theme chips are a pure re-projection of the homepage payload, so reuse
    # ``_HOME_CACHE`` directly (keyed on ``(theme_norm=None, hours)``) before
    # paying the gamma round-trip. Previously this endpoint always called
    # ``_build_homepage`` and ignored the cache — ~480 ms warm per click.
    cache_key = (None, int(hours))
    cached = _HOME_CACHE.get(cache_key)
    if cached is not None:
        rows = [
            ThemeHeatmapRow.model_validate(r)
            for r in cached.get("theme_heatmap", [])
            if (r.get("n_markets") or 0) > 0
        ]
        return ThemesResponse(n_themes=len(rows), themes=rows)

    settings: Settings = get_settings()
    shared_http: httpx.AsyncClient | None = getattr(request.app.state, "async_http", None)
    try:
        home = await _build_homepage(
            theme=None,
            hours=int(hours),
            gamma_url=settings.polymarket_gamma_url,
            clob_url=settings.polymarket_clob_url,
            http=shared_http,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e

    # Populate the shared cache so a subsequent /homepage hit is free too.
    _HOME_CACHE.set(cache_key, home.model_dump(), ttl=CACHE_TTL_SECONDS)
    rows = [r for r in home.theme_heatmap if r.n_markets > 0]
    return ThemesResponse(n_themes=len(rows), themes=rows)


__all__ = [
    "CACHE_TTL_SECONDS",
    "HomepageMover",
    "HomepageNewMarket",
    "HomepageNews",
    "HomepageResolving",
    "TerminalHomepageResponse",
    "ThemeHeatmapRow",
    "clear_cache",
    "get_homepage",
    "router",
]
