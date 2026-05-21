"""FastAPI router exposing the Polymarket implied-σ pipeline (A1 → A3).

Three GET endpoints under ``/vol/pm-iv``:

* ``GET /vol/pm-iv/assets`` — list the assets we have hardcoded ladders for
  (so a future UI can populate a dropdown without guessing).
* ``GET /vol/pm-iv/{asset}`` — raw :class:`PMIVResult` for inspection
  (no benchmark comparison; useful when debugging the ladder fit itself).
* ``GET /vol/pm-iv/gap/{asset}`` — full :class:`PMIVGapSnapshot` combining
  Polymarket σ with external benchmarks.

This router is **opt-in** via ``PFM_VOL_PM_IV_ENABLED=1`` because the project
ships behind a feature flag while the UI integration is still pending. See
``api/src/pfm/main.py`` for the gated mount.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from pfm.cache_utils import get_cache
from pfm.dependencies import get_polymarket_client
from pfm.vol.pm_iv_extractor import (
    LADDER_REGISTRY,
    PMIVResult,
    discover_ladder_family,
    fit_implied_sigma,
)
from pfm.vol.pm_iv_gap import PMIVGapSnapshot, compute_gap_snapshot

logger = logging.getLogger(__name__)


_CACHE_TTL_S = 600
_NAMESPACE = "pm_iv_gap"


router = APIRouter(prefix="/vol/pm-iv", tags=["vol-pm-iv"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_asset(asset: str) -> str:
    """Reject empty strings (422) and unknown ladder slugs (404)."""
    if not asset or not asset.strip():
        raise HTTPException(status_code=422, detail="asset must be a non-empty string")
    asset_key = asset.upper().strip()
    if asset_key not in LADDER_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"asset {asset_key!r} not in LADDER_REGISTRY",
        )
    return asset_key


def _http_client(request: Request) -> Any:
    """Return the app-level sync httpx client when available, else None.

    Tests that mount the router on a fresh ``FastAPI()`` won't have
    ``app.state.http`` configured — that's fine, ``get_benchmark_for_asset``
    will create its own short-lived client.
    """
    return getattr(request.app.state, "http", None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/assets")
def get_assets() -> dict[str, Any]:
    """Return the list of registered assets the extractor knows ladders for."""
    return {
        "assets": sorted(LADDER_REGISTRY.keys()),
        "count": len(LADDER_REGISTRY),
    }


@router.get("/{asset}", response_model=PMIVResult)
def get_pm_iv(
    asset: str,
    polymarket_client: Annotated[Any, Depends(get_polymarket_client)],
    maturity_filter: str | None = Query(
        default=None,
        description="Optional ISO-date prefix to disambiguate ladder maturities, e.g. '2026-12-31'.",
    ),
) -> PMIVResult:
    """Raw Polymarket-implied σ for ``asset`` (no benchmark comparison)."""
    asset_key = _validate_asset(asset)

    cache = get_cache(_NAMESPACE, ttl=_CACHE_TTL_S)
    cache_key = ("raw", asset_key, maturity_filter)
    cached = cache.get(cache_key)
    if cached is not None:
        return PMIVResult.model_validate(cached)

    try:
        family = discover_ladder_family(
            asset_key,
            polymarket_client=polymarket_client,
            maturity_filter=maturity_filter,
        )
    except Exception as exc:
        logger.exception("pm_iv: discover_ladder_family failed for %s", asset_key)
        raise HTTPException(
            status_code=502,
            detail=f"polymarket discovery failed: {exc}",
        ) from exc

    if family is None:
        raise HTTPException(
            status_code=404,
            detail=f"no ladder family found for asset {asset_key!r}",
        )

    try:
        result = fit_implied_sigma(family)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("pm_iv: fit_implied_sigma failed for %s", asset_key)
        raise HTTPException(
            status_code=502,
            detail=f"sigma fit failed: {exc}",
        ) from exc

    cache.set(cache_key, result.model_dump(mode="json"), ttl=_CACHE_TTL_S)
    return result


@router.get("/gap/{asset}", response_model=PMIVGapSnapshot)
def get_pm_iv_gap(
    asset: str,
    request: Request,
    polymarket_client: Annotated[Any, Depends(get_polymarket_client)],
    maturity_filter: str | None = Query(
        default=None,
        description="Optional ISO-date prefix to disambiguate ladder maturities.",
    ),
) -> PMIVGapSnapshot:
    """Polymarket σ vs external benchmarks gap snapshot for ``asset``."""
    asset_key = _validate_asset(asset)

    cache = get_cache(_NAMESPACE, ttl=_CACHE_TTL_S)
    cache_key = ("gap", asset_key, maturity_filter)
    cached = cache.get(cache_key)
    if cached is not None:
        return PMIVGapSnapshot.model_validate(cached)

    http = _http_client(request)
    try:
        snap = compute_gap_snapshot(
            asset_key,
            polymarket_client=polymarket_client,
            http=http,
            maturity_filter=maturity_filter,
        )
    except Exception as exc:
        logger.exception("pm_iv: compute_gap_snapshot failed for %s", asset_key)
        raise HTTPException(
            status_code=502,
            detail=f"gap snapshot failed: {exc}",
        ) from exc

    cache.set(cache_key, snap.model_dump(mode="json"), ttl=_CACHE_TTL_S)
    return snap


__all__ = ["router"]
