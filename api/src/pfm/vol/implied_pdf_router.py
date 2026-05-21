"""FastAPI router exposing the implied-PDF Terminal feature (Phase 1+2).

Two GET endpoints under ``/terminal/implied-pdf``:

* ``GET /terminal/implied-pdf/assets`` — list the assets we have ladders for
  (so the Terminal UI can populate a dropdown without guessing).
* ``GET /terminal/implied-pdf/{asset}`` — the dense, smoothed
  :class:`~pfm.vol.implied_pdf_schemas.ImpliedPDFResult` for one asset.

Design — decoupled from the parallel engine/discovery modules
--------------------------------------------------------------
The math engine (:func:`pfm.vol.implied_pdf.compute_implied_pdf`) and the
Kalshi ladder discovery (:func:`pfm.sources.kalshi.discover_index_ladder`) are
authored *concurrently* with this router. To keep this module importable and
its tests runnable standalone, we route every real call through the two thin
seams :func:`_discover` and :func:`_compute`, which lazily import the real
callables only when invoked. Tests monkeypatch those seams.

The feature is mounted as part of Terminal mode (prefix ``/terminal``); the
parent session wires the mount in ``pfm.main``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from pfm.cache_utils import get_cache
from pfm.dependencies import get_kalshi_client
from pfm.vol.implied_pdf_schemas import (
    DEFAULT_EPSILON,
    DEFAULT_GRID_SIZE,
    DataShape,
    ImpliedPDFResult,
    SmoothMethod,
)

if TYPE_CHECKING:
    from pfm.vol.implied_pdf_schemas import LadderFamily

logger = logging.getLogger(__name__)


_CACHE_TTL_S = 300
_NAMESPACE = "implied_pdf"


router = APIRouter(prefix="/terminal", tags=["terminal"])


# ---------------------------------------------------------------------------
# Asset registry
# ---------------------------------------------------------------------------
# Equity-index assets resolve to Kalshi terminal-bucket ladders (KXINX). Barrier
# assets (BTC/ETH/WTI via Polymarket touch markets) are Phase 4 — see the TODO
# below — and are intentionally NOT registered here yet.

SUPPORTED: dict[str, dict[str, Any]] = {
    "SPX": {
        "venue": "kalshi",
        "default_shape": "terminal_buckets",
        "name": "S&P 500 Index",
        "asset_class": "equity_index",
    },
    "NDX": {
        "venue": "kalshi",
        "default_shape": "terminal_buckets",
        "name": "Nasdaq-100 Index",
        "asset_class": "equity_index",
    },
}

# TODO(phase-4): register barrier assets (BTC, ETH, WTI) sourced from Polymarket
# touch/one-touch markets. Those flow through ``data_shape="barrier_touch"`` and
# require ``barrier_to_terminal=True`` plus the GBM reflection overlay. The
# discovery seam will need a Polymarket branch keyed off ``venue``.


# ---------------------------------------------------------------------------
# Indirection seams — lazily import the parallel modules so this router and its
# tests import cleanly even before those modules exist. Tests monkeypatch these.
# ---------------------------------------------------------------------------


def _discover(
    asset_key: str,
    client: Any,
    *,
    maturity_filter: str | None = None,
    prefer_shape: DataShape | None = None,
) -> LadderFamily:
    """Discover the same-maturity ladder family for ``asset_key`` on Kalshi.

    Thin seam over :func:`pfm.sources.kalshi.discover_index_ladder` — imported
    lazily so this module loads without the (parallel-authored) source module.

    Args:
        asset_key: Normalised asset key (e.g. ``"SPX"``).
        client: The Kalshi client injected via :func:`get_kalshi_client`.
        maturity_filter: Optional ISO-date prefix to disambiguate expiries.
        prefer_shape: Optional override of the discovered data shape.

    Returns:
        The discovered :class:`LadderFamily`.
    """
    from pfm.sources.kalshi import discover_index_ladder  # lazy

    return discover_index_ladder(
        asset_key,
        client,
        maturity_filter=maturity_filter,
        prefer_shape=prefer_shape,
    )


def _compute(family: LadderFamily, **kwargs: Any) -> ImpliedPDFResult:
    """Compute the implied PDF for ``family``.

    Thin seam over :func:`pfm.vol.implied_pdf.compute_implied_pdf` — imported
    lazily so this module loads without the (parallel-authored) engine module.

    Args:
        family: The discovered ladder family.
        **kwargs: Forwarded keyword args (``method``, ``eps``, ``grid_size``,
            ``barrier_to_terminal``, ``tail_model``).

    Returns:
        The dense :class:`ImpliedPDFResult`.
    """
    from pfm.vol.implied_pdf import compute_implied_pdf  # lazy

    return compute_implied_pdf(family, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_asset(asset: str) -> str:
    """Reject empty strings (422) and unknown assets (404).

    Args:
        asset: The raw path parameter.

    Returns:
        The normalised, registered asset key.

    Raises:
        HTTPException: 422 if empty/blank, 404 if not in :data:`SUPPORTED`.
    """
    if not asset or not asset.strip():
        raise HTTPException(status_code=422, detail="asset must be a non-empty string")
    asset_key = asset.upper().strip()
    if asset_key not in SUPPORTED:
        raise HTTPException(
            status_code=404,
            detail=f"asset {asset_key!r} not supported (known: {sorted(SUPPORTED)})",
        )
    return asset_key


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/implied-pdf/assets")
def get_assets() -> dict[str, Any]:
    """Return the registry of assets the implied-PDF feature knows ladders for."""
    assets = [
        {
            "asset": key,
            "venue": meta["venue"],
            "default_shape": meta["default_shape"],
            "name": meta["name"],
            "asset_class": meta["asset_class"],
        }
        for key, meta in sorted(SUPPORTED.items())
    ]
    return {"assets": assets, "count": len(assets)}


@router.get("/implied-pdf/{asset}/maturities")
def get_implied_pdf_maturities(
    asset: str,
    kalshi_client: Annotated[Any, Depends(get_kalshi_client)],
) -> dict[str, Any]:
    """List the asset's Kalshi index expiries + liquidity (powers the date selector)."""
    asset_key = _validate_asset(asset)
    try:
        from pfm.vol.pricing_kernel import list_index_maturities

        mats = list_index_maturities(asset_key, kalshi_client)
    except Exception as exc:
        logger.exception("implied_pdf: maturities failed for %s", asset_key)
        raise HTTPException(status_code=502, detail=f"kalshi maturities failed: {exc}") from exc
    return {"asset": asset_key, "maturities": mats, "count": len(mats)}


@router.get("/implied-pdf/{asset}", response_model=ImpliedPDFResult)
def get_implied_pdf(
    asset: str,
    kalshi_client: Annotated[Any, Depends(get_kalshi_client)],
    series: str | None = Query(
        default=None,
        description="Kalshi series to pull the ladder from (e.g. KXINXY for year-end).",
    ),
    maturity: str | None = Query(
        default=None,
        description="Optional ISO-date prefix to disambiguate expiries, e.g. '2026-05-15'.",
    ),
    shape: DataShape | None = Query(
        default=None,
        description="Override the asset's default data shape.",
    ),
    method: SmoothMethod = Query(
        default="pchip_monotone",
        description="Smoothing/interpolation method for the dense PDF.",
    ),
    eps: float = Query(
        default=DEFAULT_EPSILON,
        ge=0.0,
        le=0.2,
        description="Clipping epsilon for survival/CDF probabilities.",
    ),
    grid_size: int = Query(
        default=DEFAULT_GRID_SIZE,
        ge=16,
        le=2000,
        description="Number of points in the dense output grid.",
    ),
    barrier_to_terminal: bool = Query(
        default=False,
        description="For barrier ladders, also produce a GBM touch→terminal overlay.",
    ),
    tail_model: Literal["lognormal", "linear", "none"] = Query(
        default="lognormal",
        description="How to extrapolate the open tails beyond the strike range.",
    ),
) -> ImpliedPDFResult:
    """Compute the implied risk-neutral PDF for ``asset`` from its market ladder.

    Flow: validate → cache lookup → discover ladder (502 on upstream failure) →
    compute PDF (422 on bad input, 502 otherwise) → cache the JSON dump → return.

    Args:
        asset: The asset path parameter (e.g. ``"SPX"``).
        kalshi_client: Injected Kalshi client.
        maturity: Optional ISO-date prefix to pick a specific expiry.
        shape: Optional data-shape override (defaults to the asset's default).
        method: Smoothing method.
        eps: Probability clipping epsilon.
        grid_size: Dense grid resolution.
        barrier_to_terminal: Whether to emit a GBM touch→terminal overlay.
        tail_model: Tail extrapolation model.

    Returns:
        The dense :class:`ImpliedPDFResult`.

    Raises:
        HTTPException: 422 (bad input), 404 (unknown asset), 502 (upstream).
    """
    asset_key = _validate_asset(asset)
    prefer_shape: DataShape | None = shape or SUPPORTED[asset_key]["default_shape"]
    ladder_key = series or asset_key

    cache = get_cache(_NAMESPACE, ttl=_CACHE_TTL_S)
    cache_key = (
        asset_key,
        ladder_key,
        maturity,
        prefer_shape,
        method,
        eps,
        grid_size,
        barrier_to_terminal,
        tail_model,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return ImpliedPDFResult.model_validate(cached)

    try:
        family = _discover(
            ladder_key,
            kalshi_client,
            maturity_filter=maturity,
            prefer_shape=prefer_shape,
        )
    except Exception as exc:
        logger.exception("implied_pdf: discovery failed for %s", asset_key)
        raise HTTPException(
            status_code=502,
            detail=f"kalshi ladder discovery failed: {exc}",
        ) from exc

    if family is None:
        raise HTTPException(
            status_code=404,
            detail=f"no ladder family found for asset {asset_key!r}",
        )

    try:
        result = _compute(
            family,
            method=method,
            eps=eps,
            grid_size=grid_size,
            barrier_to_terminal=barrier_to_terminal,
            tail_model=tail_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("implied_pdf: compute failed for %s", asset_key)
        raise HTTPException(
            status_code=502,
            detail=f"implied-pdf computation failed: {exc}",
        ) from exc

    cache.set(cache_key, result.model_dump(mode="json"), ttl=_CACHE_TTL_S)
    return result


__all__ = ["router"]
