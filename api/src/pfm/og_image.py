"""Open-Graph image generator for embed cards.

Renders 1200x630 PNGs (the canonical OG / Twitter ``summary_large_image`` size)
showing a market's recent price sparkline, last YES probability, 7d change and
volume. Output is cached on disk for ``CACHE_TTL_SECONDS`` so repeated link-
preview hits from social platforms don't spam Polymarket.

Why matplotlib? It's already a transitive dep of statsmodels in this repo, has
a deterministic ``Agg`` backend, and is trivial to mock in tests by patching
:func:`render_market_og` directly. The module never blocks on a real network
call — :func:`render_market_og` is a pure ``(slug, history, last_price, ...)``
function and the router wraps it in :func:`asyncio.to_thread`.

Public API
----------
- :func:`render_market_og` — pure renderer (bytes in, bytes out).
- :func:`get_or_render_market_og` — disk-cached wrapper used by the router.
- :func:`clear_disk_cache` — test helper.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — type checking only; runtime uses lazy import below
    import matplotlib.pyplot as plt
else:
    # Lazy-load matplotlib to keep ``pfm.main`` cold-boot under the cache TTL.
    # matplotlib (+ its font cache build) adds ~210ms at import time; OG-image
    # rendering is rare-path (one render per market cache miss).
    plt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _plt():
    """Return ``matplotlib.pyplot`` after a one-shot headless setup."""
    global plt
    if plt is None:
        import matplotlib as _mpl

        _mpl.use("Agg")
        import matplotlib.pyplot as _plt_mod

        plt = _plt_mod
    return plt


# --- constants --------------------------------------------------------------

# 1200x630 @ 100 dpi → exactly the OG / twitter:summary_large_image size.
FIG_WIDTH_INCHES: float = 12.0
FIG_HEIGHT_INCHES: float = 6.3
FIG_DPI: int = 100

CACHE_DIR: Path = Path("/tmp/pfm_og_cache")
CACHE_TTL_SECONDS: int = 300  # 5 minutes

# Brand palette — kept in this module so tests can monkeypatch it without
# pulling the whole web/plotly-theme.js layer.
BG_COLOR: str = "#0b0e14"
FG_COLOR: str = "#e6edf3"
ACCENT_UP: str = "#3fb950"
ACCENT_DN: str = "#f85149"
SUBTLE_COLOR: str = "#6e7681"

# --- light palette (matches web/index.html aesthetic) ----------------------
# These are reused by the factor + strategy renderers below. Kept module-level
# so tests / future callers can monkeypatch.
LIGHT_BG: str = "#fafafa"
LIGHT_INK: str = "#0a0a0c"
LIGHT_INK_2: str = "#4a4a55"
LIGHT_ORANGE: str = "#f97316"
LIGHT_HAIRLINE: str = "#ececef"
LIGHT_POS: str = "#16a34a"
LIGHT_NEG: str = "#dc2626"

# Tier pill colours (background, foreground). Mirrors the curated alpha-hub
# tier semantics from CLAUDE.md (gold / structural / validated / etc).
_TIER_PILL_COLORS: dict[str, tuple[str, str]] = {
    "A_GOLD": ("#facc15", "#1f1300"),
    "A_STRUCTURAL": ("#3b82f6", "#ffffff"),
    "B_VALIDATED": ("#22c55e", "#062b14"),
    "B_FDR_ONLY": ("#0ea5e9", "#001f2e"),
    "C_TENTATIVE": ("#a855f7", "#1c0030"),
    "D_RAW": ("#94a3b8", "#0f172a"),
}
_DEFAULT_TIER_PILL_COLOR: tuple[str, str] = ("#64748b", "#ffffff")

# Source pill colours (Polymarket / Kalshi / fallback).
_SOURCE_PILL_COLORS: dict[str, tuple[str, str]] = {
    "polymarket": ("#1f2937", "#ffffff"),
    "kalshi": ("#0f766e", "#ffffff"),
}
_DEFAULT_SOURCE_PILL_COLOR: tuple[str, str] = ("#475569", "#ffffff")

# Factor / strategy cache. 1h TTL per the task spec; key is content-hash so a
# data change re-renders immediately rather than waiting for the TTL window.
FACTOR_CACHE_TTL_SECONDS: int = 3600
STRATEGY_CACHE_TTL_SECONDS: int = 3600

# Configure matplotlib font fallbacks once. Instrument Serif / Inter /
# JetBrains Mono aren't bundled with matplotlib; fall back to widely-available
# alternatives that match the web/index.html aesthetic. Missing fonts are
# silently tolerated (matplotlib walks the list).
_FONT_RC_SETUP_DONE: bool = False
_FONT_SERIF: list[str] = ["Instrument Serif", "Times New Roman", "Georgia", "DejaVu Serif", "serif"]
_FONT_SANS: list[str] = ["Inter", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]
_FONT_MONO: list[str] = ["JetBrains Mono", "Courier New", "DejaVu Sans Mono", "monospace"]


def _setup_fonts() -> None:
    """Best-effort font-family wiring; safe to call repeatedly."""
    global _FONT_RC_SETUP_DONE
    if _FONT_RC_SETUP_DONE:
        return
    try:
        _plt().rcParams["font.serif"] = _FONT_SERIF
        _plt().rcParams["font.sans-serif"] = _FONT_SANS
        _plt().rcParams["font.monospace"] = _FONT_MONO
    except (KeyError, ValueError) as e:  # pragma: no cover — matplotlib swallows most font issues
        logger.info("og_image font setup skipped: %s", e)
    _FONT_RC_SETUP_DONE = True


# --- core renderer ----------------------------------------------------------


def render_market_og(
    slug: str,
    *,
    question: str | None = None,
    history: Sequence[float] | None = None,
    last_price: float | None = None,
    change_7d: float | None = None,
    volume_24h: float | None = None,
) -> bytes:
    """Render a 1200x630 PNG card for ``slug`` and return raw bytes.

    All inputs except ``slug`` are optional — the renderer degrades gracefully
    when data is missing (e.g. a freshly-listed market with no history). It is
    deliberately synchronous and CPU-only so it can be wrapped in
    :func:`asyncio.to_thread` from the embed router.
    """
    fig, ax = _plt().subplots(
        figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES),
        dpi=FIG_DPI,
        facecolor=BG_COLOR,
    )
    ax.set_facecolor(BG_COLOR)

    # --- sparkline ---------------------------------------------------------
    series = list(history or [])
    if len(series) >= 2:
        n = len(series)
        xs = list(range(n))
        first, last = series[0], series[-1]
        is_up = last >= first
        line_color = ACCENT_UP if is_up else ACCENT_DN
        ax.plot(xs, series, color=line_color, linewidth=3.0)
        ax.fill_between(xs, series, min(series), color=line_color, alpha=0.15)
        ax.set_xlim(0, max(1, n - 1))
        # Pad the y-axis a little so the line never grazes the edge.
        lo, hi = min(series), max(series)
        pad = max(0.005, (hi - lo) * 0.15)
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    else:
        # No history → just leave a flat baseline.
        ax.plot([0, 1], [0.5, 0.5], color=SUBTLE_COLOR, linewidth=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- text overlays -----------------------------------------------------
    title = (question or slug).strip()
    if len(title) > 90:
        title = title[:87] + "..."
    fig.text(
        0.045,
        0.86,
        title,
        color=FG_COLOR,
        fontsize=26,
        fontweight="bold",
        ha="left",
        va="top",
        wrap=True,
    )

    if last_price is not None:
        price_pct = f"{last_price * 100:.0f}%"
        fig.text(
            0.045,
            0.42,
            price_pct,
            color=FG_COLOR,
            fontsize=64,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.text(
            0.045,
            0.28,
            "YES probability",
            color=SUBTLE_COLOR,
            fontsize=14,
            ha="left",
            va="center",
        )

    if change_7d is not None:
        sign = "+" if change_7d >= 0 else ""
        col = ACCENT_UP if change_7d >= 0 else ACCENT_DN
        fig.text(
            0.30,
            0.42,
            f"{sign}{change_7d * 100:.1f}%",
            color=col,
            fontsize=32,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.text(
            0.30,
            0.30,
            "7d change",
            color=SUBTLE_COLOR,
            fontsize=12,
            ha="left",
            va="center",
        )

    if volume_24h is not None:
        if volume_24h >= 1_000_000:
            vol_str = f"${volume_24h / 1_000_000:.1f}M"
        elif volume_24h >= 1_000:
            vol_str = f"${volume_24h / 1_000:.0f}K"
        else:
            vol_str = f"${volume_24h:.0f}"
        fig.text(
            0.46,
            0.42,
            vol_str,
            color=FG_COLOR,
            fontsize=32,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.text(
            0.46,
            0.30,
            "24h volume",
            color=SUBTLE_COLOR,
            fontsize=12,
            ha="left",
            va="center",
        )

    # --- footer ------------------------------------------------------------
    fig.text(
        0.045,
        0.06,
        "PFM",
        color=FG_COLOR,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        0.10,
        0.06,
        "Prediction Factor Model",
        color=SUBTLE_COLOR,
        fontsize=12,
        ha="left",
        va="bottom",
    )
    fig.text(
        0.955,
        0.06,
        slug,
        color=SUBTLE_COLOR,
        fontsize=11,
        ha="right",
        va="bottom",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG_COLOR, dpi=FIG_DPI, bbox_inches="tight")
    _plt().close(fig)
    return buf.getvalue()


# --- on-disk cache ----------------------------------------------------------


def _safe_filename(slug: str) -> str:
    """Map ``slug`` to a filename that's safe across platforms."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in slug)[:120]


def _cache_path(slug: str) -> Path:
    return CACHE_DIR / f"{_safe_filename(slug)}.png"


def get_or_render_market_og(
    slug: str,
    *,
    question: str | None = None,
    history: Sequence[float] | None = None,
    last_price: float | None = None,
    change_7d: float | None = None,
    volume_24h: float | None = None,
    ttl: int = CACHE_TTL_SECONDS,
) -> bytes:
    """Return the PNG bytes for ``slug``, rendering on cache miss.

    A best-effort disk cache lives under ``/tmp/pfm_og_cache``; on any IO
    error we fall back to rendering directly so a broken cache doesn't 5xx
    the API.
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("og cache mkdir failed: %s", e)
        return render_market_og(
            slug,
            question=question,
            history=history,
            last_price=last_price,
            change_7d=change_7d,
            volume_24h=volume_24h,
        )

    path = _cache_path(slug)
    if path.exists():
        try:
            age = time.time() - path.stat().st_mtime
            if age < ttl:
                return path.read_bytes()
        except OSError as e:
            logger.warning("og cache stat/read failed for %s: %s", slug, e)

    png = render_market_og(
        slug,
        question=question,
        history=history,
        last_price=last_price,
        change_7d=change_7d,
        volume_24h=volume_24h,
    )
    try:
        path.write_bytes(png)
    except OSError as e:
        logger.warning("og cache write failed for %s: %s", slug, e)
    return png


def clear_disk_cache() -> None:
    """Test helper: remove every file under :data:`CACHE_DIR`."""
    if not CACHE_DIR.exists():
        return
    for f in CACHE_DIR.iterdir():
        with contextlib.suppress(OSError):
            f.unlink()


# --- factor / strategy OG renderers ----------------------------------------
#
# Shared layout primitives (hairlines, pills, sparklines) live as small inner
# helpers below — the two public renderers are deliberately kept flat so the
# diff against the existing market renderer reads top-to-bottom.


def _draw_hairline(
    fig: Any, *, x0: float, x1: float, y: float, color: str = LIGHT_HAIRLINE
) -> None:
    """Draw a 1pt hairline rectangle in figure (0-1) coords."""
    from matplotlib.patches import Rectangle

    fig.patches.append(
        Rectangle(
            (x0, y),
            x1 - x0,
            0.0015,
            transform=fig.transFigure,
            color=color,
            linewidth=0,
            zorder=2,
        )
    )


def _draw_pill(
    fig: Any,
    text: str,
    *,
    x: float,
    y: float,
    bg: str,
    fg: str,
    fontsize: int = 12,
) -> None:
    """Draw a rounded-ish pill at figure coords ``(x, y)`` (left-anchored)."""
    fig.text(
        x,
        y,
        f"  {text}  ",
        color=fg,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="center",
        family="sans-serif",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": bg, "edgecolor": bg, "linewidth": 0},
    )


def _draw_sparkline(
    fig: Any,
    series: Sequence[float],
    *,
    rect: tuple[float, float, float, float],
    color: str = LIGHT_ORANGE,
    fill_alpha: float = 0.12,
) -> None:
    """Draw a sparkline in figure coords ``rect=(x, y, w, h)``.

    No axes, no ticks — purely visual. Empty/short series → flat baseline.
    """
    ax = fig.add_axes(rect, facecolor="none")
    s = list(series or [])
    if len(s) >= 2:
        xs = list(range(len(s)))
        ax.plot(xs, s, color=color, linewidth=2.4, solid_capstyle="round")
        ax.fill_between(xs, s, min(s), color=color, alpha=fill_alpha)
        ax.set_xlim(0, len(s) - 1)
        lo, hi = min(s), max(s)
        pad = max(1e-6, (hi - lo) * 0.18)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.plot([0, 1], [0.5, 0.5], color=LIGHT_HAIRLINE, linewidth=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_factor_og(
    factor_id: str,
    *,
    name: str | None = None,
    theme: str | None = None,
    source: str | None = None,
    history: Sequence[float] | None = None,
    last_price: float | None = None,
) -> bytes:
    """Render a 1200x630 PNG OG card for a single factor.

    Layout: serif title top-left, theme + source pills under it, 90-day
    probability sparkline filling the right half, big mono price bottom-left,
    hairline footer with ``factor_id``.
    """
    _setup_fonts()
    plt_mod = _plt()
    fig, _root = plt_mod.subplots(
        figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES),
        dpi=FIG_DPI,
        facecolor=LIGHT_BG,
    )
    _root.set_facecolor(LIGHT_BG)
    _root.set_axis_off()

    # --- title (serif) -----------------------------------------------------
    title = (name or factor_id.replace("_", " ").replace("-", " ")).strip()
    if len(title) > 80:
        title = title[:77] + "..."
    fig.text(
        0.04,
        0.84,
        title,
        color=LIGHT_INK,
        fontsize=44,
        fontweight="normal",
        family="serif",
        ha="left",
        va="top",
        wrap=True,
    )

    # --- pills (theme + source) -------------------------------------------
    pill_x = 0.04
    pill_y = 0.66
    if theme:
        theme_str = str(theme).strip().upper()
        _draw_pill(fig, theme_str, x=pill_x, y=pill_y, bg=LIGHT_INK, fg=LIGHT_BG, fontsize=11)
        # Approx pill width in figure coords — sufficient for two-pill spacing.
        pill_x += 0.012 * (len(theme_str) + 4) + 0.01
    if source:
        src_key = str(source).strip().lower()
        bg, fg = _SOURCE_PILL_COLORS.get(src_key, _DEFAULT_SOURCE_PILL_COLOR)
        _draw_pill(fig, src_key.upper(), x=pill_x, y=pill_y, bg=bg, fg=fg, fontsize=11)

    # --- hairline below title block ---------------------------------------
    _draw_hairline(fig, x0=0.04, x1=0.96, y=0.56)

    # --- big mono number (current price) ----------------------------------
    if last_price is not None:
        price_pct = f"{last_price * 100:.1f}%"
        fig.text(
            0.04,
            0.38,
            price_pct,
            color=LIGHT_INK,
            fontsize=78,
            fontweight="bold",
            family="monospace",
            ha="left",
            va="center",
        )
        fig.text(
            0.04,
            0.22,
            "CURRENT PROBABILITY",
            color=LIGHT_INK_2,
            fontsize=11,
            family="sans-serif",
            ha="left",
            va="center",
        )
    else:
        fig.text(
            0.04,
            0.38,
            "—",
            color=LIGHT_INK_2,
            fontsize=78,
            family="monospace",
            ha="left",
            va="center",
        )
        fig.text(
            0.04,
            0.22,
            "NO RECENT DATA",
            color=LIGHT_INK_2,
            fontsize=11,
            family="sans-serif",
            ha="left",
            va="center",
        )

    # --- sparkline (right half, 90d) --------------------------------------
    _draw_sparkline(
        fig,
        history or [],
        rect=(0.52, 0.18, 0.44, 0.30),
        color=LIGHT_ORANGE,
        fill_alpha=0.10,
    )
    fig.text(
        0.52,
        0.51,
        "90-DAY PROBABILITY",
        color=LIGHT_INK_2,
        fontsize=11,
        family="sans-serif",
        ha="left",
        va="bottom",
    )

    # --- footer hairline + branding ---------------------------------------
    _draw_hairline(fig, x0=0.04, x1=0.96, y=0.10)
    fig.text(
        0.04,
        0.055,
        "PFM",
        color=LIGHT_INK,
        fontsize=14,
        fontweight="bold",
        family="sans-serif",
        ha="left",
        va="center",
    )
    fig.text(
        0.085,
        0.055,
        "Prediction Factor Model · Factor",
        color=LIGHT_INK_2,
        fontsize=11,
        family="sans-serif",
        ha="left",
        va="center",
    )
    fig.text(
        0.96,
        0.055,
        factor_id,
        color=LIGHT_INK_2,
        fontsize=11,
        family="monospace",
        ha="right",
        va="center",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=LIGHT_BG, dpi=FIG_DPI)
    plt_mod.close(fig)
    return buf.getvalue()


def render_strategy_og(
    strategy_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    tier: str | None = None,
    sharpe: float | None = None,
    equity_curve: Sequence[float] | None = None,
) -> bytes:
    """Render a 1200x630 PNG OG card for an alpha strategy.

    Layout: serif strategy name, one-line description, tier pill top-right,
    huge mono Sharpe bottom-left, equity-curve sparkline right side.
    """
    _setup_fonts()
    plt_mod = _plt()
    fig, _root = plt_mod.subplots(
        figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES),
        dpi=FIG_DPI,
        facecolor=LIGHT_BG,
    )
    _root.set_facecolor(LIGHT_BG)
    _root.set_axis_off()

    # --- title (serif) -----------------------------------------------------
    title = (name or strategy_id.replace("__", " | ").replace("_", " ")).strip()
    if len(title) > 70:
        title = title[:67] + "..."
    fig.text(
        0.04,
        0.86,
        title,
        color=LIGHT_INK,
        fontsize=38,
        fontweight="normal",
        family="serif",
        ha="left",
        va="top",
        wrap=True,
    )

    # --- one-line description ----------------------------------------------
    if description:
        desc = description.strip()
        if len(desc) > 130:
            desc = desc[:127] + "..."
        fig.text(
            0.04,
            0.66,
            desc,
            color=LIGHT_INK_2,
            fontsize=15,
            family="sans-serif",
            ha="left",
            va="top",
        )

    # --- tier pill (top-right) --------------------------------------------
    if tier:
        tier_key = str(tier).strip().upper()
        bg, fg = _TIER_PILL_COLORS.get(tier_key, _DEFAULT_TIER_PILL_COLOR)
        _draw_pill(fig, tier_key, x=0.78, y=0.84, bg=bg, fg=fg, fontsize=13)

    # --- hairline ----------------------------------------------------------
    _draw_hairline(fig, x0=0.04, x1=0.96, y=0.56)

    # --- big Sharpe (mono) -------------------------------------------------
    if sharpe is not None:
        sharpe_str = f"{sharpe:+.2f}"
        sharpe_color = LIGHT_POS if sharpe >= 0 else LIGHT_NEG
        fig.text(
            0.04,
            0.36,
            sharpe_str,
            color=sharpe_color,
            fontsize=92,
            fontweight="bold",
            family="monospace",
            ha="left",
            va="center",
        )
        fig.text(
            0.04,
            0.18,
            "OOS SHARPE",
            color=LIGHT_INK_2,
            fontsize=11,
            family="sans-serif",
            ha="left",
            va="center",
        )
    else:
        fig.text(
            0.04,
            0.36,
            "—",
            color=LIGHT_INK_2,
            fontsize=92,
            family="monospace",
            ha="left",
            va="center",
        )
        fig.text(
            0.04,
            0.18,
            "NO SHARPE AVAILABLE",
            color=LIGHT_INK_2,
            fontsize=11,
            family="sans-serif",
            ha="left",
            va="center",
        )

    # --- equity-curve sparkline (right side) ------------------------------
    series = list(equity_curve or [])
    spark_color = LIGHT_ORANGE
    if len(series) >= 2:
        spark_color = LIGHT_POS if series[-1] >= series[0] else LIGHT_NEG
    _draw_sparkline(
        fig,
        series,
        rect=(0.52, 0.18, 0.44, 0.30),
        color=spark_color,
        fill_alpha=0.10,
    )
    fig.text(
        0.52,
        0.51,
        "EQUITY CURVE",
        color=LIGHT_INK_2,
        fontsize=11,
        family="sans-serif",
        ha="left",
        va="bottom",
    )

    # --- footer ------------------------------------------------------------
    _draw_hairline(fig, x0=0.04, x1=0.96, y=0.10)
    fig.text(
        0.04,
        0.055,
        "PFM",
        color=LIGHT_INK,
        fontsize=14,
        fontweight="bold",
        family="sans-serif",
        ha="left",
        va="center",
    )
    fig.text(
        0.085,
        0.055,
        "Prediction Factor Model · Strategy",
        color=LIGHT_INK_2,
        fontsize=11,
        family="sans-serif",
        ha="left",
        va="center",
    )
    fig.text(
        0.96,
        0.055,
        strategy_id,
        color=LIGHT_INK_2,
        fontsize=11,
        family="monospace",
        ha="right",
        va="center",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=LIGHT_BG, dpi=FIG_DPI)
    plt_mod.close(fig)
    return buf.getvalue()


# --- content-hash caches ---------------------------------------------------


def _content_hash(*parts: Any) -> str:
    """Stable short hash for cache keys; ignores ordering inside dicts."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _factor_cache_path(factor_id: str, sha: str) -> Path:
    return CACHE_DIR / f"factor_{_safe_filename(factor_id)}_{sha}.png"


def _strategy_cache_path(strategy_id: str, sha: str) -> Path:
    return CACHE_DIR / f"strategy_{_safe_filename(strategy_id)}_{sha}.png"


def get_or_render_factor_og(
    factor_id: str,
    *,
    name: str | None = None,
    theme: str | None = None,
    source: str | None = None,
    history: Sequence[float] | None = None,
    last_price: float | None = None,
    ttl: int = FACTOR_CACHE_TTL_SECONDS,
) -> bytes:
    """Return PNG bytes for a factor card, rendering on cache miss.

    Cache key is ``hash(name, theme, source, last_price, history-len + edges)``
    so a real data change re-renders immediately and the TTL only catches
    stale "nothing changed" reads.
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("og cache mkdir failed: %s", e)
        return render_factor_og(
            factor_id,
            name=name,
            theme=theme,
            source=source,
            history=history,
            last_price=last_price,
        )

    h = list(history or [])
    # Hash a compact summary of the series rather than the full vector — the
    # full list works too, but the summary keeps the cache key small.
    hist_summary = (len(h), h[0] if h else None, h[-1] if h else None, sum(h))
    sha = _content_hash(name, theme, source, last_price, hist_summary)
    path = _factor_cache_path(factor_id, sha)
    if path.exists():
        try:
            age = time.time() - path.stat().st_mtime
            if age < ttl:
                return path.read_bytes()
        except OSError as e:
            logger.warning("factor og cache stat/read failed for %s: %s", factor_id, e)

    png = render_factor_og(
        factor_id,
        name=name,
        theme=theme,
        source=source,
        history=history,
        last_price=last_price,
    )
    try:
        path.write_bytes(png)
    except OSError as e:
        logger.warning("factor og cache write failed for %s: %s", factor_id, e)
    return png


def get_or_render_strategy_og(
    strategy_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    tier: str | None = None,
    sharpe: float | None = None,
    equity_curve: Sequence[float] | None = None,
    ttl: int = STRATEGY_CACHE_TTL_SECONDS,
) -> bytes:
    """Return PNG bytes for a strategy card, rendering on cache miss."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("og cache mkdir failed: %s", e)
        return render_strategy_og(
            strategy_id,
            name=name,
            description=description,
            tier=tier,
            sharpe=sharpe,
            equity_curve=equity_curve,
        )

    c = list(equity_curve or [])
    curve_summary = (len(c), c[0] if c else None, c[-1] if c else None, sum(c))
    sha = _content_hash(name, description, tier, sharpe, curve_summary)
    path = _strategy_cache_path(strategy_id, sha)
    if path.exists():
        try:
            age = time.time() - path.stat().st_mtime
            if age < ttl:
                return path.read_bytes()
        except OSError as e:
            logger.warning("strategy og cache stat/read failed for %s: %s", strategy_id, e)

    png = render_strategy_og(
        strategy_id,
        name=name,
        description=description,
        tier=tier,
        sharpe=sharpe,
        equity_curve=equity_curve,
    )
    try:
        path.write_bytes(png)
    except OSError as e:
        logger.warning("strategy og cache write failed for %s: %s", strategy_id, e)
    return png


__all__ = [
    "BG_COLOR",
    "CACHE_DIR",
    "CACHE_TTL_SECONDS",
    "FACTOR_CACHE_TTL_SECONDS",
    "FIG_DPI",
    "FIG_HEIGHT_INCHES",
    "FIG_WIDTH_INCHES",
    "LIGHT_BG",
    "LIGHT_HAIRLINE",
    "LIGHT_INK",
    "LIGHT_INK_2",
    "LIGHT_NEG",
    "LIGHT_ORANGE",
    "LIGHT_POS",
    "STRATEGY_CACHE_TTL_SECONDS",
    "clear_disk_cache",
    "get_or_render_factor_og",
    "get_or_render_market_og",
    "get_or_render_strategy_og",
    "render_factor_og",
    "render_market_og",
    "render_strategy_og",
]
