"""Universal chart-as-PNG endpoint for the Terminal frontend.

Exposes ``POST /export/chart-png`` which accepts a (title, x, y, kind) blob
and returns a PNG ``image/png`` byte stream. The frontend can wire any
Plotly-backed chart through this endpoint to give the user a "Download
chart" button without round-tripping the rendered DOM.

Why server-side rendering?
    Plotly's client-side toImage() is great when the chart is on screen,
    but our Bloomberg-style hub has dozens of mini-panels and bulk-PDF
    flows where the chart isn't currently displayed. Matplotlib (Agg
    backend, headless) is the cheapest dependency that handles every
    chart kind we care about.

Defensive limits
    * ``len(x) == len(y) <= 1000`` — anything longer is downsampling
      territory and should be done before the PNG step anyway.
    * ``width, height <= 4096`` px — keeps any one render below ~25 MB
      memory regardless of DPI.

Caching
    Repeated POSTs with identical payloads hit a 10-minute TTL cache so
    a user clicking "Download" twice doesn't re-render. The cache key is
    a stable hash of the JSON-encoded request, scoped to the ``chart``
    namespace via :func:`pfm.cache_utils.get_cache`.

Routing
    This module owns its :class:`fastapi.APIRouter`. ``main.py`` is left
    untouched per project convention — wire it in explicitly via::

        from pfm.chart_export import router as chart_export_router
        app.include_router(chart_export_router)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Body, HTTPException, Response
from pydantic import BaseModel, Field, model_validator

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)


# Defensive caps — see module docstring.
MAX_POINTS: int = 1000
MAX_PIXELS: int = 4096
MIN_PIXELS: int = 64

# Process-wide cache for rendered PNGs. 10-minute TTL is long enough to
# absorb double-clicks and short enough not to hold stale charts when the
# underlying data refreshes.
_CHART_CACHE = get_cache("chart", ttl=600)


ChartKind = Literal["line", "bar", "scatter"]


class ChartRequest(BaseModel):
    """Body for ``POST /export/chart-png``."""

    title: str = Field("", max_length=200)
    x: list[float | int | str] = Field(..., min_length=1, max_length=MAX_POINTS)
    y: list[float | int] = Field(..., min_length=1, max_length=MAX_POINTS)
    kind: ChartKind = "line"
    width: int = Field(1200, ge=MIN_PIXELS, le=MAX_PIXELS)
    height: int = Field(600, ge=MIN_PIXELS, le=MAX_PIXELS)
    xlabel: str = Field("", max_length=80)
    ylabel: str = Field("", max_length=80)

    @model_validator(mode="after")
    def _check_lengths(self) -> ChartRequest:
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y length mismatch: {len(self.x)} != {len(self.y)}")
        return self


# ---------------------------------------------------------------------------
# Rendering — Agg backend, no display.
# ---------------------------------------------------------------------------


def _cache_key(req: ChartRequest) -> str:
    """Stable SHA256-based cache key from the request payload."""
    payload = req.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_png_sync(req: ChartRequest) -> bytes:
    """Render a PNG using matplotlib's headless Agg backend.

    Imported lazily inside the function so any matplotlib-import side
    effects (font cache, etc.) only happen the first time the endpoint
    is hit, not at module load.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    dpi = 100
    fig_w = max(1.0, req.width / dpi)
    fig_h = max(1.0, req.height / dpi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    try:
        if req.kind == "line":
            ax.plot(req.x, req.y, color="#14213d", linewidth=1.5)
        elif req.kind == "bar":
            ax.bar(req.x, req.y, color="#14213d")
        elif req.kind == "scatter":
            ax.scatter(req.x, req.y, c="#14213d", s=10)
        else:  # pragma: no cover — Literal guards this.
            raise ValueError(f"unknown chart kind: {req.kind}")

        if req.title:
            ax.set_title(req.title, fontsize=12)
        if req.xlabel:
            ax.set_xlabel(req.xlabel)
        if req.ylabel:
            ax.set_ylabel(req.ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        return buf.getvalue()
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/export", tags=["export"])


@router.post("/chart-png")
async def chart_png(req: Annotated[ChartRequest, Body(...)]) -> Response:
    """Render a single chart as PNG and return ``image/png`` bytes.

    Cached for 10 minutes per identical payload (``hashlib.sha256`` over
    the request JSON). Rendering itself runs in a worker thread so the
    event loop stays responsive — matplotlib is sync-only.
    """
    # Re-validate caps defensively (Pydantic already did most of the work).
    if not (MIN_PIXELS <= req.width <= MAX_PIXELS and MIN_PIXELS <= req.height <= MAX_PIXELS):
        raise HTTPException(
            status_code=422,
            detail=f"width/height must be in [{MIN_PIXELS}, {MAX_PIXELS}] px",
        )

    key = _cache_key(req)
    hit = _CHART_CACHE.get(key)
    if hit is not None:
        logger.debug("chart_png cache hit %s", key[:12])
        return Response(content=hit, media_type="image/png", headers={"X-Cache": "HIT"})

    try:
        png_bytes = await asyncio.to_thread(_render_png_sync, req)
    except Exception as exc:  # matplotlib raises a zoo of exception types
        logger.exception("chart_png render failed")
        raise HTTPException(status_code=500, detail=f"render failed: {exc}") from exc

    _CHART_CACHE.set(key, png_bytes)
    return Response(content=png_bytes, media_type="image/png", headers={"X-Cache": "MISS"})


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _clear_cache_for_tests() -> None:
    """Reset the in-process chart cache. Test-only."""
    _CHART_CACHE.clear()


def _peek_cache_for_tests(req: ChartRequest) -> bytes | None:
    """Return the cached PNG for ``req`` (or ``None``). Test-only."""
    return _CHART_CACHE.get(_cache_key(req))


__all__ = [
    "MAX_PIXELS",
    "MAX_POINTS",
    "ChartKind",
    "ChartRequest",
    "_clear_cache_for_tests",
    "_peek_cache_for_tests",
    "chart_png",
    "router",
]


# Re-export bound JSON helper for downstream debug — keeps the module
# self-contained when you import `chart_export` for instrumentation.
def _debug_payload_hash(req: ChartRequest) -> str:  # pragma: no cover
    return hashlib.sha256(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()
