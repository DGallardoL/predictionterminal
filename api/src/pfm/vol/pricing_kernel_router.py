"""FastAPI router for the cross-venue pricing-kernel Terminal feature.

``GET /terminal/pricing-kernel/{asset}`` returns the three aligned densities
(Kalshi-Q, options-Q, physical-P), the cross-venue divergence, and the empirical
pricing kernel / implied risk aversion.

The heavy compute (live Kalshi ladder + yfinance option chain + GARCH fit) runs
through the :func:`_compute` seam so tests can monkeypatch it, and is cached for
a few minutes since options quotes move slowly intraday.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from pfm.cache_utils import get_cache
from pfm.dependencies import get_kalshi_client
from pfm.vol.pricing_kernel import PricingKernelResult

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 300
_NAMESPACE = "pricing_kernel"

router = APIRouter(prefix="/terminal", tags=["terminal"])

#: Equity-index assets with both a Kalshi ladder and a listed option chain.
SUPPORTED: dict[str, dict[str, Any]] = {
    "SPX": {"name": "S&P 500 Index", "options_ticker": "^SPX"},
    "NDX": {"name": "Nasdaq-100 Index", "options_ticker": "^NDX"},
}


def _validate_asset(asset: str) -> str:
    if not asset or not asset.strip():
        raise HTTPException(status_code=422, detail="asset must be a non-empty string")
    key = asset.upper().strip()
    if key not in SUPPORTED:
        raise HTTPException(
            status_code=404,
            detail=f"asset {key!r} not supported (known: {sorted(SUPPORTED)})",
        )
    return key


def _compute(asset: str, client: Any, **kwargs: Any) -> PricingKernelResult:
    """Seam over :func:`pfm.vol.pricing_kernel.compute_pricing_kernel` (mockable)."""
    from pfm.vol.pricing_kernel import compute_pricing_kernel

    return compute_pricing_kernel(asset, client, **kwargs)


@router.get("/pricing-kernel/assets")
def get_assets() -> dict[str, Any]:
    """List the assets the pricing-kernel feature supports."""
    return {
        "assets": [
            {"asset": k, "name": v["name"], "options_ticker": v["options_ticker"]}
            for k, v in sorted(SUPPORTED.items())
        ],
        "count": len(SUPPORTED),
    }


@router.get("/pricing-kernel/{asset}/maturities")
def get_maturities(
    asset: str,
    kalshi_client: Annotated[Any, Depends(get_kalshi_client)],
) -> dict[str, Any]:
    """List the asset's Kalshi index expiries + liquidity (powers the date selector)."""
    key = _validate_asset(asset)
    try:
        from pfm.vol.pricing_kernel import list_index_maturities

        mats = list_index_maturities(key, kalshi_client)
    except Exception as exc:
        logger.exception("pricing_kernel: maturities failed for %s", key)
        raise HTTPException(status_code=502, detail=f"kalshi maturities failed: {exc}") from exc
    return {"asset": key, "maturities": mats, "count": len(mats)}


@router.get("/pricing-kernel/{asset}", response_model=PricingKernelResult)
def get_pricing_kernel(
    asset: str,
    kalshi_client: Annotated[Any, Depends(get_kalshi_client)],
    series: str | None = Query(
        None, description="Kalshi series to pull the ladder from (e.g. KXINXY for year-end)."
    ),
    maturity: str | None = Query(None, description="ISO-date prefix to pick the Kalshi expiry."),
    risk_free: float = Query(0.045, ge=0.0, le=0.2, description="Annualised risk-free rate."),
    annual_drift: float = Query(
        0.06, ge=-0.5, le=0.5, description="Physical expected return for the P-measure density."
    ),
    grid_size: int = Query(240, ge=64, le=1000, description="Shared-grid resolution."),
    lookback_days: int = Query(
        400, ge=120, le=2000, description="History window for the GARCH physical-vol fit."
    ),
) -> PricingKernelResult:
    """Compare the Kalshi vs options risk-neutral densities and recover the kernel.

    Flow: validate → cache lookup → compute (502 on upstream/data failure) →
    cache → return.

    Raises:
        HTTPException: 404 (unknown asset), 422 (bad data), 502 (upstream).
    """
    key = _validate_asset(asset)
    cache = get_cache(_NAMESPACE, ttl=_CACHE_TTL_S)
    cache_key = (key, series, maturity, risk_free, annual_drift, grid_size, lookback_days)
    cached = cache.get(cache_key)
    if cached is not None:
        return PricingKernelResult.model_validate(cached)

    try:
        result = _compute(
            key,
            kalshi_client,
            ladder_key=series,
            maturity=maturity,
            risk_free=risk_free,
            annual_drift=annual_drift,
            grid_size=grid_size,
            lookback_days=lookback_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("pricing_kernel: compute failed for %s", key)
        raise HTTPException(
            status_code=502, detail=f"pricing-kernel computation failed: {exc}"
        ) from exc

    cache.set(cache_key, result.model_dump(mode="json"), ttl=_CACHE_TTL_S)
    return result
