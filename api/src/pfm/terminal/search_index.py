"""Search-index dump for the front-end ⌘-K palette.

The palette wants ONE compact JSON blob it can keep in memory and search
client-side rather than round-tripping ``/terminal/search`` on every
keystroke. This endpoint walks ``factors.yml`` and emits a minimal
``{i, s, n, t, p, v, h}`` row per factor (id, slug, name, theme, last
price, 24h volume, 7d sparkline). All optional fields are ``None`` when
the cached snapshot doesn't have a value — the front-end is expected to
filter on availability rather than relying on a fully-populated row.

In addition to ``factors`` the response carries ``strategies``,
``pages``, and ``actions`` placeholder lists so the palette schema is
ready for future expansion without breaking the front-end contract.

Routing
-------
Owns its own :class:`fastapi.APIRouter`; ``main.py`` is left untouched.
Wire explicitly::

    from pfm.terminal_search_index import router as terminal_search_index_router
    app.include_router(terminal_search_index_router)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import yaml
from fastapi import APIRouter, Query, Response
from pydantic import BaseModel, Field

from pfm import terminal as terminal_mod
from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)


# --- file paths / cache -----------------------------------------------------

# After the 2026-05 refactor this module lives at ``pfm/terminal/search_index.py``
# so we climb one extra parent compared to the original ``pfm/terminal_search_index.py``
# layout to reach the package root / repo root respectively.
DEFAULT_FACTORS_PATH: Path = Path(__file__).resolve().parents[1] / "factors.yml"
DEFAULT_STRATEGIES_PATH: Path = (
    Path(__file__).resolve().parents[4] / "web" / "data" / "alpha_strategies.json"
)

CACHE_TTL_SECONDS: int = 600
SPARKLINE_LENGTH: int = 7

#: Default chunk size for ``/terminal/search-index/chunked``. Tuned so each
#: chunk lands comfortably under 50 KiB on the wire (200 rows x ~250 B each
#: ≈ 50 KiB raw, ~12 KiB gzipped) — small enough that a slow connection
#: paints the palette in one paint frame, large enough that prefetching
#: 5-6 chunks idle covers the full 1090-factor catalogue.
DEFAULT_CHUNK_SIZE: int = 200
MIN_CHUNK_SIZE: int = 1
MAX_CHUNK_SIZE: int = 1000

_INDEX_CACHE = get_cache("terminal_search_index", ttl=CACHE_TTL_SECONDS)


def clear_cache() -> None:
    """Test/utility — drop any cached index dump."""
    _INDEX_CACHE.clear()


# --- Pydantic schemas -------------------------------------------------------


class FactorIndexRow(BaseModel):
    """One factor row in the palette dump.

    Fields are deliberately one-letter to keep the payload small — there
    are 1090 factors so any per-row overhead matters.
    """

    i: str = Field(..., description="Factor id.")
    s: str = Field(..., description="Slug.")
    n: str = Field(..., description="Display name.")
    t: str | None = Field(None, description="Theme.")
    p: float | None = Field(None, description="Last cached price (YES).")
    v: float | None = Field(None, description="24h volume if cached.")
    h: list[float] | None = Field(None, description="7-bar sparkline.")


class StrategyIndexRow(BaseModel):
    i: str = Field(..., description="Strategy id (a_id__b_id pair).")
    n: str = Field(..., description="Display name.")
    t: str | None = Field(None, description="Tier (A_GOLD / B_VALIDATED / ...).")


class PageIndexRow(BaseModel):
    i: str
    n: str
    u: str = Field(..., description="Relative URL the front-end should route to.")


class ActionIndexRow(BaseModel):
    i: str
    n: str
    k: str = Field(..., description="Optional keyboard shortcut hint.")


class TerminalSearchIndexResponse(BaseModel):
    version: str = Field(..., description="ISO-8601 UTC build timestamp.")
    n_factors: int
    factors: list[FactorIndexRow]
    strategies: list[StrategyIndexRow]
    pages: list[PageIndexRow]
    actions: list[ActionIndexRow]


class TerminalSearchIndexChunkResponse(BaseModel):
    """One slice of the factor catalogue for the lazy palette path.

    The frontend calls ``chunk=0`` on ⌘-K open to paint the palette
    immediately, then prefetches subsequent chunks idle. Strategies /
    pages / actions are tiny and ride along on every chunk so the
    palette can search them without a second request.
    """

    version: str = Field(..., description="ISO-8601 UTC build timestamp.")
    n_factors: int = Field(..., description="Total factors across all chunks.")
    chunk: int = Field(..., ge=0)
    chunk_size: int = Field(..., ge=1)
    total_chunks: int = Field(..., ge=0)
    factors: list[FactorIndexRow]
    strategies: list[StrategyIndexRow]
    pages: list[PageIndexRow]
    actions: list[ActionIndexRow]


# --- helpers ----------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _load_factors_yaml(path: Path = DEFAULT_FACTORS_PATH) -> list[dict[str, Any]]:
    """Read raw factor entries from ``factors.yml`` (no pydantic validation)."""
    if not path.exists():
        logger.warning("search-index: factors file missing at %s", path)
        return []
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    raw = doc.get("factors") or []
    return raw if isinstance(raw, list) else []


def _load_strategies(path: Path = DEFAULT_STRATEGIES_PATH) -> list[StrategyIndexRow]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("search-index: strategies load failed: %s", e)
        return []
    rows: list[StrategyIndexRow] = []
    for s in doc.get("strategies", []) or []:
        a, b, tier = s.get("a_id"), s.get("b_id"), s.get("tier")
        if not (a and b):
            continue
        rows.append(
            StrategyIndexRow(
                i=f"{a}__{b}",
                n=s.get("name") or f"{a} / {b}",
                t=tier,
            )
        )
    return rows


def _static_pages() -> list[PageIndexRow]:
    """Hard-coded list of front-end routes the palette can jump to.

    Kept in source rather than read from disk because the front-end shell
    is the source of truth and these labels are user-visible strings.
    """
    return [
        PageIndexRow(i="page-terminal", n="Terminal", u="/#terminal"),
        PageIndexRow(i="page-alpha-hub", n="α Hub", u="/#alpha-hub"),
        PageIndexRow(i="page-strategies", n="Strategies", u="/#strategies"),
        PageIndexRow(i="page-regression", n="Regression", u="/#regression"),
        PageIndexRow(i="page-alerts", n="Alerts", u="/#alerts"),
        PageIndexRow(i="page-graveyard", n="Alpha Graveyard", u="/#graveyard"),
    ]


def _static_actions() -> list[ActionIndexRow]:
    return [
        ActionIndexRow(i="action-refresh", n="Refresh", k="r"),
        ActionIndexRow(i="action-search", n="Search", k="/"),
        ActionIndexRow(i="action-toggle-theme", n="Toggle theme", k="t"),
        ActionIndexRow(i="action-export", n="Export current view", k="e"),
    ]


# --- composer ---------------------------------------------------------------


def _build_index() -> TerminalSearchIndexResponse:
    raw_factors = _load_factors_yaml()
    # Best-effort price + sparkline lookup from the on-disk pickle. Empty
    # when the cache is missing — every row's ``p`` / ``h`` becomes ``None``
    # and the front-end will degrade gracefully.
    history = terminal_mod._load_factor_history_cache(terminal_mod.DEFAULT_FACTOR_HISTORY_PATH)

    rows: list[FactorIndexRow] = []
    for f in raw_factors:
        fid = f.get("id")
        slug = f.get("slug") or ""
        if not fid:
            continue
        # Use explicit ``in`` checks instead of ``or`` — pandas.Series raises
        # ``ValueError("truth value of a Series is ambiguous")`` on boolean
        # coercion, so the historical short-circuit was a latent bug that
        # only triggered once the factor-history pickle started shipping
        # real series (post strat7 prewarm backfill).
        ser = history.get(slug) if slug in history else history.get(fid)
        last_price: float | None = None
        spark: list[float] | None = None
        # Defensive isinstance check: ``TERMINAL_CACHE``'s L2 Redis layer
        # JSON-serialises values with ``default=str``, which stringifies any
        # ``pd.Series`` in the history dict. A cross-worker readback then
        # returns a ``dict[str, str]`` and the original ``ser.iloc[-1]`` call
        # below crashes with ``AttributeError: 'str' object has no attribute
        # 'iloc'``. Treating anything that isn't a Series as "no data" keeps
        # this endpoint up while the per-row ``p`` / ``h`` fields degrade to
        # ``None`` exactly like a cold cache.
        if ser is not None and not isinstance(ser, pd.Series):
            ser = None
        if ser is not None and len(ser) > 0:
            try:
                last_price = float(ser.iloc[-1])
            except (TypeError, ValueError, IndexError):
                last_price = None
            try:
                tail = ser.iloc[-SPARKLINE_LENGTH:]
                spark = [float(x) for x in tail.tolist() if _safe_float(x) is not None]
                if not spark:
                    spark = None
            except (TypeError, ValueError, IndexError):
                spark = None
        rows.append(
            FactorIndexRow(
                i=str(fid),
                s=str(slug),
                n=str(f.get("name") or fid),
                t=(f.get("theme") or None),
                p=last_price,
                v=None,  # Volume isn't carried in the daily-price pickle.
                h=spark,
            )
        )

    return TerminalSearchIndexResponse(
        version=datetime.now(tz=UTC).isoformat(),
        n_factors=len(rows),
        factors=rows,
        strategies=_load_strategies(),
        pages=_static_pages(),
        actions=_static_actions(),
    )


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-search-index"])


@router.get("/search-index", response_model=TerminalSearchIndexResponse)
def get_search_index() -> TerminalSearchIndexResponse:
    """Compact palette dump: factors + strategies + pages + actions.

    Heavy payload (~280 KiB raw, ~60 KiB gzipped at 1090 factors). Kept
    for backward-compat with the existing ⌘-K shell. New callers should
    prefer ``/terminal/search-index/chunked?chunk=0&size=200`` which
    paints incrementally and lets the client prefetch idle.
    """
    cached = _INDEX_CACHE.get("v1")
    if cached is not None:
        return TerminalSearchIndexResponse.model_validate(cached)
    resp = _build_index()
    _INDEX_CACHE.set("v1", resp.model_dump(), ttl=CACHE_TTL_SECONDS)
    return resp


@router.get(
    "/search-index/chunked",
    response_model=TerminalSearchIndexChunkResponse,
)
def get_search_index_chunked(
    response: Response,
    chunk: Annotated[int, Query(ge=0)] = 0,
    size: Annotated[
        int,
        Query(ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE),
    ] = DEFAULT_CHUNK_SIZE,
) -> TerminalSearchIndexChunkResponse:
    """Lazy-loadable slice of the palette factor catalogue.

    The full index is built once and cached; each chunked request just
    slices ``factors[chunk*size : (chunk+1)*size]``. Out-of-range chunks
    return an empty ``factors`` list (not 404) so the frontend can stop
    prefetching by checking ``len(factors) == 0`` without a try/except.

    Headers
    -------
    ``X-Total-Chunks``: total chunk count for the current ``size``. Lets
    the frontend prefetch ``range(1, total)`` after the initial paint
    without re-reading the response body.
    """
    cached = _INDEX_CACHE.get("v1")
    if cached is not None:
        full = TerminalSearchIndexResponse.model_validate(cached)
    else:
        full = _build_index()
        _INDEX_CACHE.set("v1", full.model_dump(), ttl=CACHE_TTL_SECONDS)

    total = full.n_factors
    total_chunks = math.ceil(total / size) if total > 0 else 0
    if chunk > 0 and chunk >= total_chunks:
        rows: list[FactorIndexRow] = []
    else:
        start = chunk * size
        rows = full.factors[start : start + size]

    response.headers["X-Total-Chunks"] = str(total_chunks)
    response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"

    return TerminalSearchIndexChunkResponse(
        version=full.version,
        n_factors=total,
        chunk=chunk,
        chunk_size=size,
        total_chunks=total_chunks,
        factors=rows,
        strategies=full.strategies,
        pages=full.pages,
        actions=full.actions,
    )


__all__ = [
    "CACHE_TTL_SECONDS",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_FACTORS_PATH",
    "DEFAULT_STRATEGIES_PATH",
    "MAX_CHUNK_SIZE",
    "MIN_CHUNK_SIZE",
    "ActionIndexRow",
    "FactorIndexRow",
    "PageIndexRow",
    "StrategyIndexRow",
    "TerminalSearchIndexChunkResponse",
    "TerminalSearchIndexResponse",
    "clear_cache",
    "get_search_index",
    "get_search_index_chunked",
    "router",
]
