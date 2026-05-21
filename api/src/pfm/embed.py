"""Embeddable widgets + OG image endpoints — viral growth surface.

Any blog post or tweet should be able to drop a live PFM card with one line of
HTML. The endpoints here render small, self-contained HTML cards (Plotly
sparkline + last price + change + footer) that:

  * declare the right ``Content-Security-Policy`` / ``X-Frame-Options`` to be
    iframe-able from anywhere,
  * include OG / Twitter meta tags pointing at a server-rendered PNG for nice
    Slack / Twitter unfurls,
  * post a ``{pfm:'resize', height}`` message to the parent so a one-line
    ``embed.js`` loader can size the iframe to its content.

Endpoints
---------
- ``GET /embed/market/{slug}``       — live market mini-card.
- ``GET /embed/strategy/{pair_id}``  — alpha-strategy card.
- ``GET /embed/compare?slugs=a,b``   — overlay two markets normalised.
- ``GET /embed/og/market/{slug}.png``— Open-Graph PNG for the market card.
- ``POST /embed/beacon``             — best-effort embed-impression tracker.

Routing note: this module owns its :class:`fastapi.APIRouter`; ``main.py`` is
left untouched per the project's CLAUDE.md. Mount with::

    from pfm.embed import router as embed_router
    app.include_router(embed_router)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Query, Request, Response, status
from fastapi import Path as FPath
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from pfm import terminal as terminal_mod
from pfm.cache_utils import get_cache
from pfm.og_image import (
    get_or_render_factor_og,
    get_or_render_market_og,
    get_or_render_strategy_og,
)

logger = logging.getLogger(__name__)


# --- constants --------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)

# Shared CSP / framing headers — applied to every embed HTML response.
_EMBED_HEADERS: dict[str, str] = {
    "X-Frame-Options": "ALLOWALL",
    "Content-Security-Policy": "frame-ancestors *",
    "Cache-Control": "public, max-age=300, s-maxage=600, stale-while-revalidate=1800",
}

# Public site root for "PFM" footer link / OG canonical URL. Override in
# tests via ``embed.HOME_URL = "..."``.
HOME_URL: str = "https://prediction-factor-model.example.com"

# Where the alpha-strategies pipeline writes its outputs (see CLAUDE.md memory
# entry: α Hub is the product surface).
ALPHA_STRATEGIES_PATH: Path = (
    Path(__file__).resolve().parents[3] / "web" / "data" / "alpha_strategies.json"
)
LIVE_SIGNALS_PATH: Path = Path(__file__).resolve().parents[3] / "web" / "data" / "live_signals.json"

# Where the impression beacon appends. JSONL is fine for the POC; switch to
# something durable if the firehose grows.
BEACON_LOG_PATH: Path = Path("/tmp/pfm_embed_beacons.jsonl")

# Polymarket endpoints (for direct slug fetching; mirrors terminal_compare).
GAMMA_URL: str = "https://gamma-api.polymarket.com"
CLOB_URL: str = "https://clob.polymarket.com"
HTTP_TIMEOUT_SECONDS: float = 5.0

# In-process cache for the rendered HTML of common slugs.
_EMBED_CACHE = get_cache("embed", ttl=300)

# Tier badge colour map for /embed/strategy.
_TIER_COLORS: dict[str, tuple[str, str]] = {
    # (background, foreground)
    "A_GOLD": ("#facc15", "#1f1300"),
    "A_STRUCTURAL": ("#3b82f6", "#ffffff"),
    "B_VALIDATED": ("#22c55e", "#062b14"),
    "B_FDR_ONLY": ("#0ea5e9", "#001f2e"),
    "C_TENTATIVE": ("#a855f7", "#1c0030"),
    "D_RAW": ("#94a3b8", "#0f172a"),
}
_DEFAULT_TIER_COLOR: tuple[str, str] = ("#64748b", "#ffffff")

_ACTION_COLORS: dict[str, tuple[str, str]] = {
    "LONG_SPREAD": ("#16a34a", "#ffffff"),
    "SHORT_SPREAD": ("#dc2626", "#ffffff"),
    "HOLD": ("#475569", "#e2e8f0"),
    "FLAT": ("#475569", "#e2e8f0"),
    "ERROR": ("#b91c1c", "#ffffff"),
}
_DEFAULT_ACTION_COLOR: tuple[str, str] = ("#475569", "#e2e8f0")

# Sparkline overlay colours for /embed/compare.
_COMPARE_PALETTE: list[str] = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"]


# --- schemas ----------------------------------------------------------------


class BeaconPayload(BaseModel):
    """Per-impression beacon. All optional except ``slug`` (or ``pair_id``)."""

    slug: str | None = None
    pair_id: str | None = None
    referrer: str | None = None
    utm_source: str | None = None
    utm_medium: str | None = None
    utm_campaign: str | None = None
    ts: str | None = Field(default=None, description="Client-side ISO-8601 timestamp.")


# --- helpers ----------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _format_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.0f}%"


def _format_change(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"


def _format_volume(v: float | None) -> str:
    if v is None or v <= 0:
        return "—"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v / 1_000:.0f}K"
    return f"${v:.0f}"


def _format_float(v: float | None, *, digits: int = 2, suffix: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}{suffix}"


def _validate_theme(theme: str | None) -> str:
    return "dark" if (theme or "").lower() == "dark" else "light"


def _fetch_market_history(slug: str, *, days: int = 7) -> tuple[dict[str, Any] | None, list[float]]:
    """Fetch (gamma_market, history_prices) for ``slug``.

    Synchronous on purpose — callers wrap in :func:`asyncio.to_thread` if
    needed. Returns ``(None, [])`` on any upstream failure rather than
    raising; the embed should still render gracefully.
    """
    cache_key = ("hist", slug, days)
    cached = _EMBED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    market: dict[str, Any] | None = None
    history: list[float] = []
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as http:
            try:
                market = terminal_mod.fetch_gamma_market(http, GAMMA_URL, slug)
            except (LookupError, httpx.HTTPError) as e:
                logger.info("embed gamma fetch failed for %s: %s", slug, e)

            token_id: str | None = None
            if market is not None:
                raw = market.get("clobTokenIds")
                try:
                    if isinstance(raw, str) and raw:
                        token_ids = json.loads(raw)
                    elif isinstance(raw, list):
                        token_ids = raw
                    else:
                        token_ids = []
                    if isinstance(token_ids, list) and token_ids:
                        token_id = str(token_ids[0])
                except (TypeError, json.JSONDecodeError):
                    token_id = None

            if token_id:
                try:
                    r = http.get(
                        f"{CLOB_URL}/prices-history",
                        params={"market": token_id, "fidelity": 1440, "interval": "max"},
                        timeout=HTTP_TIMEOUT_SECONDS,
                    )
                    r.raise_for_status()
                    payload = r.json() or {}
                    points = payload.get("history", []) if isinstance(payload, dict) else []
                    # Take the last ``days`` daily points; clip to [0, 1].
                    prices = [
                        max(0.0, min(1.0, float(pt.get("p", 0.0))))
                        for pt in points
                        if isinstance(pt, dict) and "p" in pt
                    ]
                    history = prices[-days:] if days > 0 else prices
                except (httpx.HTTPError, ValueError, TypeError) as e:
                    logger.info("embed clob fetch failed for %s: %s", slug, e)
    except httpx.HTTPError as e:
        logger.warning("embed http client error: %s", e)

    result = (market, history)
    _EMBED_CACHE.set(cache_key, result, ttl=300)
    return result


def _load_strategy(pair_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Look up an alpha-strategy and its live signal by ``pair_id``.

    Returns ``(strategy_dict | None, live_signal_dict | None)``.
    """
    strat: dict[str, Any] | None = None
    if ALPHA_STRATEGIES_PATH.exists():
        try:
            with ALPHA_STRATEGIES_PATH.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
            for s in doc.get("strategies", []) or []:
                if s.get("pair_id") == pair_id:
                    strat = s
                    break
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("embed: alpha_strategies load failed: %s", e)

    signal: dict[str, Any] | None = None
    if LIVE_SIGNALS_PATH.exists():
        try:
            with LIVE_SIGNALS_PATH.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
            signals_map = doc.get("signals", {}) or {}
            if pair_id in signals_map:
                signal = signals_map[pair_id]
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("embed: live_signals load failed: %s", e)
    return strat, signal


# --- factor data lookup for OG ---------------------------------------------


def _load_factor_for_og(factor_id: str) -> dict[str, Any] | None:
    """Resolve a factor's display metadata + recent history for the OG card.

    Returns a dict with ``name`` / ``theme`` / ``source`` / ``history`` /
    ``last_price`` (any of which may be ``None``), or ``None`` if the factor
    can't be resolved at all. Strictly best-effort — never raises.

    Resolution order, all opt-in (so tests can run offline):
      1. ``factors.yml`` if importable via :mod:`pfm.factors` (name/theme/source).
      2. Polymarket slug fetch (last 90d history + question/last price) when
         the factor's slug is known.
    """
    factor_meta: dict[str, Any] = {}
    # 1. Catalog lookup — tolerate any import / parse failure silently.
    # ``pfm.factors`` exposes ``load_factors(path) → dict[id, FactorConfig]``.
    try:
        from pathlib import Path as _Path

        from pfm.config import get_settings as _gs
        from pfm.factors import load_factors as _lf

        cat = _lf(_Path(_gs().factors_file))
        entry = cat.get(factor_id)
        if entry is not None:
            factor_meta["name"] = getattr(entry, "name", None)
            factor_meta["theme"] = getattr(entry, "theme", None)
            factor_meta["source"] = getattr(entry, "source", None)
            factor_meta["slug"] = getattr(entry, "slug", None)
    except Exception as e:  # best-effort; missing/broken catalog is fine.
        logger.debug("factor catalog lookup failed for %s: %s", factor_id, e)

    # 2. If we know a Polymarket slug, fetch last 90d + live price.
    slug = factor_meta.get("slug") or factor_id
    market, history = _fetch_market_history(slug, days=90)

    if market is None and not factor_meta:
        return None

    if market is not None:
        if not factor_meta.get("name"):
            factor_meta["name"] = str(market.get("question") or "").strip() or None
        if not factor_meta.get("source"):
            factor_meta["source"] = "polymarket"
        live = terminal_mod.shape_live(market)
        last_price = _safe_float(live.get("midpoint"))
        if last_price is None and history:
            last_price = history[-1]
        factor_meta["last_price"] = last_price
    elif history:
        factor_meta["last_price"] = history[-1]

    factor_meta["history"] = history
    return factor_meta


def _load_strategy_for_og(strategy_id: str) -> dict[str, Any] | None:
    """Resolve a strategy's OG card payload from ``alpha_strategies.json``.

    Returns ``None`` if the strategies file is missing or the id isn't in it.
    """
    if not ALPHA_STRATEGIES_PATH.exists():
        return None
    try:
        with ALPHA_STRATEGIES_PATH.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("strategy og: alpha_strategies load failed: %s", e)
        return None

    strat: dict[str, Any] | None = None
    for s in doc.get("strategies", []) or []:
        if s.get("pair_id") == strategy_id:
            strat = s
            break
    if strat is None:
        return None

    a_name = str(strat.get("a_name") or strat.get("a_id") or "A")
    b_name = str(strat.get("b_name") or strat.get("b_id") or "B")
    name = f"{a_name} | {b_name}"

    # Build a one-line description from the rationale / theory reference.
    rationale = strat.get("rationale") or strat.get("theory_reference") or ""
    description = str(rationale).strip().replace("\n", " ")

    sharpe = _safe_float(strat.get("oos_sharpe"))
    if sharpe is None:
        sharpe = _safe_float(strat.get("full_sharpe"))

    # Equity-curve sparkline source: prefer an embedded series if present;
    # otherwise synthesize a deterministic geometric drift from sharpe + n_obs
    # so the card has *something* visual to render. Synthetic curves are
    # explicitly seeded by strategy_id so the same id always renders identically.
    raw_curve = strat.get("equity_curve") or strat.get("oos_equity_curve")
    curve: list[float] = []
    if isinstance(raw_curve, list):
        curve = [float(x) for x in raw_curve if _safe_float(x) is not None]
    if not curve and sharpe is not None:
        n_obs = int(strat.get("n_obs") or 252)
        n_obs = max(30, min(500, n_obs))
        # Deterministic pseudo-random walk seeded by strategy_id.
        import random

        rng = random.Random(hash(strategy_id) & 0xFFFFFFFF)
        # Daily mean return implied by annualised sharpe at ~10% vol/yr.
        daily_vol = 0.10 / (252**0.5)
        daily_mu = sharpe * daily_vol * (1.0 / (252**0.5)) * (252**0.5)
        # ↑ simplifies to sharpe * daily_vol; kept explicit for readability.
        value = 1.0
        for _ in range(n_obs):
            value *= 1.0 + daily_mu + daily_vol * (rng.random() - 0.5) * 2.0
            curve.append(value)

    return {
        "name": name,
        "description": description,
        "tier": str(strat.get("tier") or ""),
        "sharpe": sharpe,
        "equity_curve": curve,
    }


# --- router -----------------------------------------------------------------


router = APIRouter(prefix="/embed", tags=["embed"])


@router.get(
    "/market/{slug}",
    response_class=HTMLResponse,
    summary="Embeddable mini-card for a Polymarket market.",
)
async def embed_market(
    request: Request,
    slug: Annotated[str, FPath(min_length=1, max_length=200)],
    theme: Annotated[str, Query(pattern="^(light|dark)$")] = "light",
    height: Annotated[int, Query(ge=120, le=600)] = 200,
    autorefresh: Annotated[bool, Query()] = False,
) -> Response:
    """Render a self-contained HTML card for ``slug``."""
    theme_norm = _validate_theme(theme)
    market, history = await asyncio.to_thread(_fetch_market_history, slug, days=7)

    if market is None:
        question = slug.replace("-", " ").title()
        live = {}
    else:
        question = str(market.get("question") or slug).strip()
        live = terminal_mod.shape_live(market)

    midpoint = _safe_float(live.get("midpoint")) if live else None
    if midpoint is None and history:
        midpoint = history[-1]
    change_7d = _safe_float(live.get("one_week_price_change")) if live else None
    if change_7d is None and len(history) >= 2 and history[0] != 0:
        change_7d = (history[-1] - history[0]) / history[0]
    volume_24h = _safe_float(live.get("volume_24hr")) if live else None

    description = (
        f"YES probability {_format_pct(midpoint)} · "
        f"7d {_format_change(change_7d)} · vol {_format_volume(volume_24h)}"
    )

    base = str(request.base_url).rstrip("/")
    canonical_url = f"{base}/embed/market/{slug}?theme={theme_norm}"
    og_image_url = f"{base}/embed/og/market/{slug}.png"
    beacon_url = f"{base}/embed/beacon"

    template = _jinja_env.get_template("embed_market.html")
    html = template.render(
        slug=slug,
        question=question,
        description=description,
        theme=theme_norm,
        price_pct=_format_pct(midpoint),
        change_7d_str=_format_change(change_7d),
        change_7d_value=change_7d if change_7d is not None else 0.0,
        volume_str=_format_volume(volume_24h),
        history_json=json.dumps(history),
        theme_json=json.dumps(theme_norm),
        slug_json=json.dumps(slug),
        beacon_url_json=json.dumps(beacon_url),
        canonical_url=canonical_url,
        og_image_url=og_image_url,
        home_url=HOME_URL,
        height=height,
        autorefresh=autorefresh,
    )
    return HTMLResponse(content=html, headers=_EMBED_HEADERS)


@router.get(
    "/strategy/{pair_id}",
    response_class=HTMLResponse,
    summary="Embeddable card for a validated alpha strategy.",
)
async def embed_strategy(
    request: Request,
    pair_id: Annotated[str, FPath(min_length=1, max_length=200)],
    theme: Annotated[str, Query(pattern="^(light|dark)$")] = "light",
) -> Response:
    """Render an alpha-strategy card. Falls back to a placeholder if the pair
    isn't in the curated catalog yet."""
    theme_norm = _validate_theme(theme)
    strat, signal = await asyncio.to_thread(_load_strategy, pair_id)

    if strat is None:
        # Friendly placeholder rather than 404 — embeds should never break a
        # host page just because the pipeline hasn't seen this id yet.
        pair_label = pair_id.replace("__", " | ").replace("_", " ")
        tier = "UNKNOWN"
        oos_sharpe = None
        half_life = None
        action = "ERROR"
        reason = "Strategy not in current catalog"
    else:
        a_name = str(strat.get("a_name") or strat.get("a_id") or "A")
        b_name = str(strat.get("b_name") or strat.get("b_id") or "B")
        pair_label = f"{a_name} | {b_name}"
        tier = str(strat.get("tier") or "UNKNOWN")
        oos_sharpe = _safe_float(strat.get("oos_sharpe"))
        half_life = _safe_float(strat.get("half_life_days"))
        if signal is not None:
            action = str(signal.get("action") or "HOLD")
            reason = str(signal.get("reason") or "—")
        else:
            action = "HOLD"
            reason = "Live signal unavailable"

    z_now = _safe_float((signal or {}).get("current_z"))

    tier_bg, tier_fg = _TIER_COLORS.get(tier, _DEFAULT_TIER_COLOR)
    action_bg, action_fg = _ACTION_COLORS.get(action.upper(), _DEFAULT_ACTION_COLOR)

    description = (
        f"{tier} · OOS Sharpe {_format_float(oos_sharpe, digits=2)} · "
        f"½-life {_format_float(half_life, digits=1, suffix='d')} · "
        f"action {action}"
    )

    base = str(request.base_url).rstrip("/")
    canonical_url = f"{base}/embed/strategy/{pair_id}?theme={theme_norm}"
    og_image_url = f"{base}/embed/og/market/{pair_id}.png"

    template = _jinja_env.get_template("embed_strategy.html")
    html = template.render(
        pair_id=pair_id,
        pair_id_json=json.dumps(pair_id),
        pair_label=pair_label,
        tier=tier,
        tier_bg=tier_bg,
        tier_fg=tier_fg,
        oos_sharpe_str=_format_float(oos_sharpe, digits=2),
        half_life_str=_format_float(half_life, digits=1, suffix="d"),
        z_str=_format_float(z_now, digits=2),
        action=action,
        reason=reason,
        action_bg=action_bg,
        action_fg=action_fg,
        theme=theme_norm,
        description=description,
        canonical_url=canonical_url,
        og_image_url=og_image_url,
        home_url=HOME_URL,
    )
    return HTMLResponse(content=html, headers=_EMBED_HEADERS)


@router.get(
    "/compare",
    response_class=HTMLResponse,
    summary="Embeddable overlay of 2+ market price histories (normalised).",
)
async def embed_compare(
    request: Request,
    slugs: Annotated[str, Query(min_length=3, description="Comma-separated slugs.")],
    theme: Annotated[str, Query(pattern="^(light|dark)$")] = "light",
) -> Response:
    """Render an overlay sparkline normalising each leg to its first
    observation (so the y-axis shows pct change from t0)."""
    theme_norm = _validate_theme(theme)
    parsed = [s.strip() for s in slugs.split(",") if s.strip()]
    parsed = parsed[:4]  # cap at 4 legs to keep the card readable
    if len(parsed) < 2:
        return HTMLResponse(
            content="<p>need at least 2 slugs</p>",
            status_code=400,
            headers=_EMBED_HEADERS,
        )

    legs: list[dict[str, Any]] = []
    for i, slug in enumerate(parsed):
        _market, hist = await asyncio.to_thread(_fetch_market_history, slug, days=30)
        # Normalise to pct from t0 so different price scales overlay cleanly.
        if hist and hist[0] not in (0, None):
            base_p = hist[0]
            normed = [(p / base_p - 1.0) * 100.0 for p in hist]
        else:
            normed = []
        legs.append(
            {
                "slug": slug,
                "color": _COMPARE_PALETTE[i % len(_COMPARE_PALETTE)],
                "history": normed,
            }
        )

    title = f"Compare {' vs '.join(parsed)}"
    description = f"30d normalised overlay of {len(parsed)} markets."
    tick_color = "#7d8590" if theme_norm == "dark" else "#656d76"

    base = str(request.base_url).rstrip("/")
    canonical_url = f"{base}/embed/compare?slugs={','.join(parsed)}&theme={theme_norm}"
    og_image_url = f"{base}/embed/og/market/{parsed[0]}.png"

    template = _jinja_env.get_template("embed_compare.html")
    html = template.render(
        title=title,
        description=description,
        legs=legs,
        legs_json=json.dumps(legs),
        slugs_str=", ".join(parsed),
        theme=theme_norm,
        tick_color_json=json.dumps(tick_color),
        canonical_url=canonical_url,
        og_image_url=og_image_url,
        home_url=HOME_URL,
    )
    return HTMLResponse(content=html, headers=_EMBED_HEADERS)


@router.get(
    "/og/market/{slug}.png",
    summary="Open-Graph PNG (1200x630) for a market — used in social unfurls.",
)
async def embed_og_market(
    slug: Annotated[str, FPath(min_length=1, max_length=200)],
) -> Response:
    """Return a cached PNG OG image for ``slug``."""
    market, history = await asyncio.to_thread(_fetch_market_history, slug, days=30)
    question = None
    last_price: float | None = None
    change_7d: float | None = None
    volume_24h: float | None = None
    if market is not None:
        question = str(market.get("question") or "").strip() or None
        live = terminal_mod.shape_live(market)
        last_price = _safe_float(live.get("midpoint"))
        change_7d = _safe_float(live.get("one_week_price_change"))
        volume_24h = _safe_float(live.get("volume_24hr"))
    if last_price is None and history:
        last_price = history[-1]
    if change_7d is None and len(history) >= 8 and history[-8] != 0:
        change_7d = (history[-1] - history[-8]) / history[-8]

    png_bytes = await asyncio.to_thread(
        get_or_render_market_og,
        slug,
        question=question,
        history=history,
        last_price=last_price,
        change_7d=change_7d,
        volume_24h=volume_24h,
    )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=300, s-maxage=600",
            "X-Frame-Options": "ALLOWALL",
        },
    )


@router.get(
    "/og/factor/{factor_id}",
    summary="Open-Graph PNG (1200x630) for a factor share link.",
)
async def embed_og_factor(
    factor_id: Annotated[str, FPath(min_length=1, max_length=200)],
) -> Response:
    """Return a cached PNG OG image for a factor.

    Tries (1) the in-repo factor catalog and (2) a live Polymarket slug fetch
    using the factor's known slug. Either source alone is enough — if the
    catalog has metadata but no slug, we render with whatever we've got. If
    *neither* resolves the factor at all, return ``404`` so social platforms
    can fall back to their own image inference.
    """
    payload = await asyncio.to_thread(_load_factor_for_og, factor_id)
    if payload is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    png_bytes = await asyncio.to_thread(
        get_or_render_factor_og,
        factor_id,
        name=payload.get("name"),
        theme=payload.get("theme"),
        source=payload.get("source"),
        history=payload.get("history") or [],
        last_price=payload.get("last_price"),
    )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600, s-maxage=3600",
            "X-Frame-Options": "ALLOWALL",
        },
    )


@router.get(
    "/og/strategy/{strategy_id}",
    summary="Open-Graph PNG (1200x630) for a strategy share link.",
)
async def embed_og_strategy(
    strategy_id: Annotated[str, FPath(min_length=1, max_length=200)],
) -> Response:
    """Return a cached PNG OG image for an alpha strategy.

    Strategy metadata comes from ``web/data/alpha_strategies.json``; the path
    is configurable via ``ALPHA_STRATEGIES_PATH`` (monkeypatched in tests).
    Returns ``404`` if the file is missing or the id isn't in the curated list.
    """
    payload = await asyncio.to_thread(_load_strategy_for_og, strategy_id)
    if payload is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    png_bytes = await asyncio.to_thread(
        get_or_render_strategy_og,
        strategy_id,
        name=payload.get("name"),
        description=payload.get("description"),
        tier=payload.get("tier"),
        sharpe=payload.get("sharpe"),
        equity_curve=payload.get("equity_curve") or [],
    )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600, s-maxage=3600",
            "X-Frame-Options": "ALLOWALL",
        },
    )


@router.post(
    "/beacon",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Embed-impression beacon (best-effort tracking, no PII).",
)
async def embed_beacon(payload: BeaconPayload) -> Response:
    """Append one beacon row to a JSONL log. Always returns 204, even on
    write failure — we never want a tracking error to surface to the host page."""
    try:
        BEACON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Beacon writes are tiny (one JSONL row, well under 4 KiB) and bounded
        # by the OS write buffer; the blocking-IO overhead is negligible vs the
        # complexity cost of pulling in aiofiles for a tracking endpoint.
        with BEACON_LOG_PATH.open("a", encoding="utf-8") as fh:  # noqa: ASYNC230
            fh.write(payload.model_dump_json() + "\n")
    except OSError as e:
        logger.warning("embed beacon write failed: %s", e)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = [
    "ALPHA_STRATEGIES_PATH",
    "BEACON_LOG_PATH",
    "CLOB_URL",
    "GAMMA_URL",
    "HOME_URL",
    "LIVE_SIGNALS_PATH",
    "BeaconPayload",
    "router",
]
