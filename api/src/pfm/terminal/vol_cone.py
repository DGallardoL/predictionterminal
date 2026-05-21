"""Realized-volatility cone endpoint for the Terminal.

A volatility cone (Burghardt-Lane 1990) compares current realized volatility
against the historical distribution of realized vol across multiple horizons.
Equity practitioners use it to flag whether vol is "rich" (above p90) or
"cheap" (below p10) versus the asset's own past — useful for sizing entries
and gauging regime.

For prediction markets, σ is estimated from Δlogit returns over rolling
windows. We annualise using the conventional √252 factor (daily bars).

Endpoint
--------
``GET /terminal/vol-cone/{slug}``

Returns, for horizons w ∈ {1, 7, 30, 90}:

* ``percentile_bands``: p10/p25/p50/p75/p90 of rolling-σ over the full sample.
* ``current_vol``: most recent rolling-σ at each horizon.
* ``current_percentile``: empirical CDF rank (0..100) of ``current_vol``
  versus the rolling-σ distribution at that horizon.

External I/O is delegated to :func:`pfm.sources.polymarket.fetch_factor_history`
which the test suite patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from pfm.model import DEFAULT_EPSILON, delta_logit
from pfm.sources.polymarket import PolymarketClient, PolymarketError, fetch_factor_history

router = APIRouter(prefix="/terminal", tags=["terminal"])

#: Standard horizons (in trading days) for the cone.
HORIZONS: tuple[int, ...] = (1, 7, 30, 90)

#: Annualisation factor for daily-bar σ → per-year σ.
ANNUALISATION: float = float(np.sqrt(252.0))

#: How many days of history we request from Polymarket. Eight months gives
#: enough data points beyond the longest 90-day window for a meaningful
#: percentile distribution.
LOOKBACK_DAYS: int = 8 * 30


def _get_polymarket_client_dep() -> PolymarketClient:
    """Default DI dependency. Overridden in tests via ``app.dependency_overrides``."""
    raise HTTPException(  # pragma: no cover - only hit when not wired
        status_code=503,
        detail="vol-cone router not wired into an app with a polymarket client",
    )


@dataclass(frozen=True)
class VolConeResult:
    """Strongly-typed cone payload (mirrors the JSON response shape)."""

    horizons: list[int]
    percentile_bands: dict[str, list[float]]
    current_vol: list[float]
    current_percentile: list[float]


def compute_vol_cone(
    prices: pd.Series,
    horizons: tuple[int, ...] = HORIZONS,
    epsilon: float = DEFAULT_EPSILON,
    annualisation: float = ANNUALISATION,
) -> VolConeResult:
    """Compute the vol cone for a price series.

    Args:
        prices: Daily probability series indexed by UTC date.
        horizons: Window lengths (in days) at which to estimate rolling σ.
        epsilon: Clip used inside :func:`pfm.model.delta_logit` to avoid
            blow-up near 0/1.
        annualisation: Multiplier applied to per-day σ.

    Returns:
        :class:`VolConeResult` with band/current arrays aligned to ``horizons``.

    Raises:
        ValueError: If the input has too few observations (< 2) to form even
            a single Δlogit return.
    """
    if len(prices) < 2:
        raise ValueError(f"need at least 2 prices for a cone, got {len(prices)}")

    returns = delta_logit(prices, epsilon=epsilon).dropna()
    if returns.empty:
        raise ValueError("Δlogit returns are empty after dropna")

    bands: dict[str, list[float]] = {k: [] for k in ("p10", "p25", "p50", "p75", "p90")}
    current_vol: list[float] = []
    current_percentile: list[float] = []

    for w in horizons:
        if w <= 0:
            raise ValueError(f"horizon must be positive, got {w}")

        # For w=1 the rolling std is trivially undefined (one observation), so
        # use abs(return) as a proxy for the per-bar realised σ. For w≥2 we
        # use sample std on the rolling window. Both are then annualised.
        if w == 1:
            rolling = returns.abs() * annualisation
        else:
            # min_periods=w so partial windows don't bias the early distribution.
            rolling = returns.rolling(window=w, min_periods=w).std(ddof=1) * annualisation

        rolling = rolling.dropna()

        if rolling.empty:
            # Not enough data for this horizon — emit NaN-as-None placeholders.
            for vals in bands.values():
                vals.append(float("nan"))
            current_vol.append(float("nan"))
            current_percentile.append(float("nan"))
            continue

        q = rolling.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        bands["p10"].append(float(q.loc[0.10]))
        bands["p25"].append(float(q.loc[0.25]))
        bands["p50"].append(float(q.loc[0.50]))
        bands["p75"].append(float(q.loc[0.75]))
        bands["p90"].append(float(q.loc[0.90]))

        last = float(rolling.iloc[-1])
        current_vol.append(last)

        # Empirical CDF rank in [0, 100]: percentage of historical observations
        # that are ≤ the current value. Excludes the current observation itself
        # so a fresh extreme correctly reads as 100.
        if len(rolling) > 1:
            past = rolling.iloc[:-1]
            pct = float((past <= last).mean() * 100.0)
        else:
            pct = 50.0
        current_percentile.append(pct)

    return VolConeResult(
        horizons=list(horizons),
        percentile_bands=bands,
        current_vol=current_vol,
        current_percentile=current_percentile,
    )


@router.get("/vol-cone/{slug}")
def get_vol_cone(
    slug: str,
    epsilon: Annotated[float, Query(gt=0.0, lt=0.5, description="logit clip ε")] = DEFAULT_EPSILON,
    lookback_days: Annotated[int, Query(ge=120, le=2000)] = LOOKBACK_DAYS,
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client_dep)] = ...,  # type: ignore[assignment]
) -> dict[str, list[int] | dict[str, list[float]] | list[float]]:
    """Return the realized-volatility cone for a single Polymarket slug.

    Response shape:

    .. code-block:: json

        {
          "horizons": [1, 7, 30, 90],
          "percentile_bands": {
            "p10": [...], "p25": [...], "p50": [...], "p75": [...], "p90": [...]
          },
          "current_vol": [...],
          "current_percentile": [...]
        }
    """
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=lookback_days)

    try:
        df = fetch_factor_history(poly, slug, start=start, end=end)
    except PolymarketError as e:
        # Distinguish "slug doesn't exist" (a client problem → 404) from
        # genuine upstream failure (server problem → 502). The Polymarket
        # client raises with the literal "no market found for slug=..."
        # message in the not-found case (see pfm/sources/polymarket.py).
        if "no market found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"market not found: {slug!r}") from e
        raise HTTPException(status_code=502, detail=f"polymarket fetch failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"polymarket fetch failed: {e}") from e

    if df is None or df.empty or "price" not in df.columns:
        raise HTTPException(status_code=404, detail=f"no price history for slug {slug!r}")

    prices = df["price"].astype(float)
    if len(prices) < max(HORIZONS) + 5:
        raise HTTPException(
            status_code=422,
            detail=(
                f"insufficient history for slug {slug!r}: "
                f"{len(prices)} bars, need ≥ {max(HORIZONS) + 5}"
            ),
        )

    result = compute_vol_cone(prices, horizons=HORIZONS, epsilon=epsilon)

    return {
        "horizons": result.horizons,
        "percentile_bands": result.percentile_bands,
        "current_vol": result.current_vol,
        "current_percentile": result.current_percentile,
    }


__all__ = [
    "ANNUALISATION",
    "HORIZONS",
    "VolConeResult",
    "compute_vol_cone",
    "router",
]
