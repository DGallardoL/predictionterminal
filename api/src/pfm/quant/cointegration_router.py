"""``GET /quant/cointegration`` — Engle-Granger cointegration test (W13-17).

Exposes a thin HTTP front over the existing :func:`pfm.cointegration.engle_granger`
helper plus the W12-32 :func:`pfm.quant.half_life.estimate_half_life` AR(1)
half-life estimator. Given two factor slugs (``a`` and ``b``) and an optional
date window, the endpoint:

1. Resolves both slugs against the loaded factor catalog (404 on miss).
2. Pulls each factor's daily probability history via
   :func:`pfm.regression_core._cached_factor_history`.
3. Aligns the two series on their common UTC date index.
4. Runs the Engle-Granger 2-step:
   * Step 1: OLS of ``y_a = α + β·y_b + ε`` → recover hedge ratio.
   * Step 2: ADF (``statsmodels.tsa.stattools.adfuller``) on residuals.
5. Computes the mean-reversion half-life from the residual spread by
   delegating to the W12-32 helper (``Δε_t = α + β·ε_{t-1} + η``, then
   ``h = -ln 2 / ln(1+β)``).
6. Returns the JSON contract documented in W13-17.

Cointegration verdict uses the 5% ADF significance level
(``adf_p_value < 0.05`` ⇒ ``is_cointegrated=true``).

This router is standalone — it can be mounted on the main ``pfm.main`` app
via ``app.include_router(router)`` without touching the
``main.py:routes`` section (a hot-claim file). Tests in
``tests/test_cointegration_endpoint.py`` mount the router on a throw-away
FastAPI app and patch ``_cached_factor_history`` to deterministic synthetic
histories so no network IO is required.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.cache import CacheBackend
from pfm.cointegration import engle_granger
from pfm.config import Settings, get_settings
from pfm.dependencies import get_cache, get_factors_dep, get_polymarket_client
from pfm.factor_resolver import (
    resolve_factor as _resolve_factor_unified,
)
from pfm.factor_resolver import (
    suggest_factors_with_meta as _factor_suggest_meta,
)
from pfm.factors import FactorConfig
from pfm.quant.half_life import estimate_half_life
from pfm.sources.polymarket import PolymarketClient

router = APIRouter(tags=["quant"])


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class CointegrationResponse(BaseModel):
    """Response shape for ``GET /quant/cointegration``."""

    a: str = Field(description="Factor slug/id for leg A (target).")
    b: str = Field(description="Factor slug/id for leg B (hedge).")
    n_obs: int = Field(description="Jointly-observed daily sample size.")
    beta: float = Field(description="OLS hedge ratio β from step 1.")
    alpha: float = Field(description="OLS intercept α from step 1.")
    adf_stat: float = Field(description="ADF test statistic on residuals.")
    adf_p_value: float = Field(description="ADF p-value. <0.05 ⇒ cointegrated.")
    is_cointegrated: bool = Field(description="Convenience flag: ``adf_p_value < 0.05``.")
    half_life_days: float | None = Field(
        description=(
            "AR(1) half-life of the spread, in days. ``null`` if the AR(1) "
            "coefficient is non-stationary (β ≥ 0) or oscillating (β ≤ -2)."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_or_404(slug: str, catalog: dict[str, FactorConfig], *, leg: str) -> FactorConfig:
    """Resolve a single slug or raise 404 with ``did_you_mean`` hints.

    The contract is:
    - empty slug → 422 (covered by Pydantic ``min_length`` on the query param).
    - unknown slug → 404 with a structured detail object pointing the caller
      at the top-3 nearest matches.
    """
    fc = _resolve_factor_unified(slug, catalog)
    if fc is not None:
        return fc
    raise HTTPException(
        status_code=404,
        detail={
            "error": f"unknown factor id ({leg}): {slug!r}",
            "query": slug,
            "leg": leg,
            "did_you_mean": _factor_suggest_meta(slug, catalog, top_k=3),
        },
    )


def _fetch_history(
    fc: FactorConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
) -> pd.Series:
    """Pull the ``price`` series for a single factor over ``[start, end]``.

    Imported lazily so tests can monkeypatch
    ``pfm.regression_core._cached_factor_history`` without round-tripping
    through ``pfm.main``.

    On empty result → 502 (consistent with other quant routers that hit the
    same upstream). Caller is responsible for handling alignment-emptiness
    (which triggers a 422 instead — bad date window, not bad upstream).
    """
    from pfm.regression_core import _cached_factor_history

    df = _cached_factor_history(fc, start, end, poly, cache, settings)
    if df.empty:
        raise HTTPException(
            status_code=502,
            detail=(f"{fc.source} returned no history for factor {fc.id!r} (slug={fc.slug!r})"),
        )
    series = df["price"]
    return series[(series.index >= start) & (series.index <= end)].rename(fc.id)


def _half_life_or_none(spread: pd.Series) -> float | None:
    """Compute half-life via the W12-32 helper. Return ``None`` for NaN/inf.

    The JSON contract uses ``null`` (Python ``None``) for "undefined" rather
    than NaN strings — the W12-32 helper returns ``+inf`` for non-mean-reverting
    AR(1) (β ≥ 0) and NaN for oscillating regimes (β ≤ -2). Both map to
    ``None`` on the wire so frontends don't have to special-case JSON
    ``"Infinity"`` / ``"NaN"`` (which are non-standard JSON).
    """
    res = estimate_half_life(spread)
    hl = res.get("half_life_days")
    if hl is None:
        return None
    try:
        hlf = float(hl)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(hlf):
        return None
    return hlf


def _safe_float(x: float) -> float:
    """Coerce NaN/Inf → 0.0 so the JSON response always validates.

    Only applied to fields whose contract is a plain ``float`` (``beta``,
    ``alpha``, ``adf_stat``, ``adf_p_value``). ``half_life_days`` is nullable
    and handled separately by :func:`_half_life_or_none`.
    """
    if x is None:
        return 0.0
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return 0.0
    return float(x)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/quant/cointegration",
    response_model=CointegrationResponse,
    summary="Engle-Granger cointegration test on a pair of factor slugs",
)
def cointegration_endpoint(
    a: Annotated[str, Query(min_length=1, description="Factor id/slug for leg A.")],
    b: Annotated[str, Query(min_length=1, description="Factor id/slug for leg B.")],
    start: Annotated[
        date | None,
        Query(description="UTC start date (inclusive). ``null`` → 1 year ago."),
    ] = None,
    end: Annotated[
        date | None,
        Query(description="UTC end date (inclusive). ``null`` → today."),
    ] = None,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> CointegrationResponse:
    """HTTP front for the Engle-Granger 2-step (W13-17).

    Validation rules (enforced before any network IO):

    * ``a`` and ``b`` must both resolve in the catalog → 404 with
      ``did_you_mean`` hints on miss.
    * ``a == b`` (resolved id) → 422 (cointegration of a series with itself
      is trivially perfect and not useful — caller probably typo'd).
    * Explicit ``start >= end`` → 422.
    * Fewer than 30 jointly-observed rows after alignment → 422 (matches the
      threshold inside :func:`pfm.cointegration.engle_granger`).
    """
    # --- resolve both slugs ----------------------------------------------
    fc_a = _resolve_or_404(a, factors, leg="a")
    fc_b = _resolve_or_404(b, factors, leg="b")

    # Reject same-factor pair early; pointless test.
    if fc_a.id == fc_b.id:
        raise HTTPException(
            status_code=422,
            detail=f"a and b resolve to the same factor id ({fc_a.id!r})",
        )

    # --- date plumbing ---------------------------------------------------
    end_date = end or date.today()
    if start is None:
        # Default to a 1-year lookback. ``pd.Timestamp - Timedelta`` returns
        # a Timestamp; ``.date()`` collapses it back to a plain ``datetime.date``
        # so the comparison below stays type-consistent.
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=365)).date()
    else:
        start_date = start
    if start_date >= end_date:
        raise HTTPException(status_code=422, detail="start must be strictly before end")
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")

    # --- fetch both histories --------------------------------------------
    s_a = _fetch_history(fc_a, start_ts, end_ts, poly, cache, settings)
    s_b = _fetch_history(fc_b, start_ts, end_ts, poly, cache, settings)

    # --- run Engle-Granger ----------------------------------------------
    # ``engle_granger`` already aligns on common index, fits OLS, runs ADF,
    # and packages the residual spread. We then recompute half-life via
    # the W12-32 helper so the wire contract uses *that* implementation
    # (W13-17 spec explicitly requires it; the helper baked into the
    # CointegrationResult is fine but not the canonical one going forward).
    result = engle_granger(s_a, s_b)

    if result.verdict == "insufficient-data":
        raise HTTPException(
            status_code=422,
            detail=(
                f"insufficient overlapping observations for {fc_a.id!r} & "
                f"{fc_b.id!r}: got {result.n_obs}, need >= 30"
            ),
        )

    half_life = _half_life_or_none(result.spread)

    return CointegrationResponse(
        a=fc_a.id,
        b=fc_b.id,
        n_obs=int(result.n_obs),
        beta=_safe_float(result.beta_hedge),
        alpha=_safe_float(result.intercept),
        adf_stat=_safe_float(result.adf_stat),
        adf_p_value=_safe_float(result.adf_pvalue),
        is_cointegrated=bool(result.adf_pvalue < 0.05),
        half_life_days=half_life,
    )
