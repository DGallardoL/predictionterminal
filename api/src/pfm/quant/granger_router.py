"""``GET /quant/granger`` — Granger causality test (W13-18).

Bivariate Granger (1969) causality test: do past values of A help predict
B beyond B's own past?

For each lag ``L ∈ {1, ..., maxlag}`` two OLS regressions are fit on the
target series ``B``:

* **Restricted**::

    B_t = α + Σ_{l=1}^L β_l · B_{t-l} + ε_t

* **Unrestricted**::

    B_t = α + Σ_{l=1}^L β_l · B_{t-l} + Σ_{l=1}^L γ_l · A_{t-l} + ε_t

If adding the lagged ``A`` terms significantly reduces the sum of squared
residuals (F-test on the joint null ``γ_1 = ... = γ_L = 0``), we say A
**Granger-causes** B *at that lag*.

This endpoint thin-wraps :func:`statsmodels.tsa.stattools.grangercausalitytests`
and surfaces the per-lag F-statistic and p-value, the best (lowest-p) lag,
and a simple boolean verdict ``a_granger_causes_b`` (true iff the best-lag
p-value is < 0.05).

Caveats / interpretation
------------------------

* Granger causality is **prediction**, not metaphysical cause. ``A_t`` may
  contain useful information about ``B_{t+L}`` even when they share a
  common driver.
* The test assumes (weak) stationarity of both series. For probability
  series in (0, 1) this is usually fine after first-differencing in logit
  space. The endpoint does **not** auto-difference; the caller chooses the
  transform by picking which slugs to point at (raw probability vs. a
  pre-differenced indicator).
* Sample-size requirement: ``len(joint) >= 4 * maxlag + 2`` (statsmodels'
  practical lower bound, used in the existing :mod:`pfm.granger` helper).

This router is intentionally **standalone** — it can be mounted on the
main ``pfm.main`` app via ``app.include_router(router)`` without touching
``main.py:routes`` (a hot-claim file in the multi-session protocol).

Request::

    GET /quant/granger?a=<slug>&b=<slug>&maxlag=5

Response::

    {
      "a": "slug1",
      "b": "slug2",
      "maxlag": 5,
      "n_obs": 245,
      "tests": [
        {"lag": 1, "f_stat": 4.2, "p_value": 0.04, "significant": true},
        {"lag": 2, "f_stat": 3.1, "p_value": 0.046, "significant": true},
        ...
      ],
      "best_lag": 1,
      "best_p_value": 0.04,
      "a_granger_causes_b": true
    }
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.cache import CacheBackend
from pfm.config import Settings, get_settings
from pfm.dependencies import get_cache, get_factors_dep, get_polymarket_client
from pfm.factor_resolver import (
    resolve_factor as _resolve_factor_unified,
)
from pfm.factor_resolver import (
    suggest_factors_with_meta as _factor_suggest_meta,
)
from pfm.factors import FactorConfig
from pfm.sources.polymarket import PolymarketClient

router = APIRouter(tags=["quant"])


_DEFAULT_HISTORY_DAYS = 365
_MAXLAG_LOWER = 1
_MAXLAG_UPPER = 20
_SIGNIFICANCE = 0.05


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class GrangerLagTest(BaseModel):
    """One lag's test outcome."""

    lag: int = Field(description="Lag in days (=number of B/A own-lags included).")
    f_stat: float = Field(description="SSR F-statistic for the joint γ_l = 0 null.")
    p_value: float = Field(description="Two-sided p-value of the F-test.")
    significant: bool = Field(description="True iff ``p_value < 0.05``.")


class GrangerResponse(BaseModel):
    """Response shape for ``GET /quant/granger``."""

    a: str
    b: str
    maxlag: int
    n_obs: int
    tests: list[GrangerLagTest]
    best_lag: int | None
    best_p_value: float | None
    a_granger_causes_b: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_slug(slug: str, catalog: dict[str, FactorConfig], *, role: str) -> FactorConfig:
    """Resolve a single id/slug against the catalog or raise 404."""
    fc = _resolve_factor_unified(slug, catalog)
    if fc is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"{role} factor not found: {slug!r}",
                "did_you_mean": _factor_suggest_meta(slug, catalog, top_k=3),
            },
        )
    return fc


def _fetch_series(
    fc: FactorConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
) -> pd.Series:
    """Pull a factor's daily price series.

    Lazy import of :func:`pfm.regression_core._cached_factor_history` so test
    code can monkeypatch the symbol without importing all of ``pfm.main``.
    """
    from pfm.regression_core import _cached_factor_history

    df = _cached_factor_history(fc, start, end, poly, cache, settings)
    if df is None or df.empty or "price" not in df.columns:
        raise HTTPException(
            status_code=502,
            detail=(
                f"{fc.source} returned no history for factor {fc.id!r} "
                f"(slug={fc.slug!r}) over [{start.date()}, {end.date()}]"
            ),
        )
    series = df["price"]
    return series[(series.index >= start) & (series.index <= end)].rename(fc.id)


def _safe_float(x: Any) -> float:
    """NaN/Inf → math.nan-safe float; statsmodels can emit ±inf on edge cases."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if math.isinf(v):
        return float("nan")
    return v


def _run_granger(a: pd.Series, b: pd.Series, maxlag: int) -> tuple[list[GrangerLagTest], int]:
    """Run statsmodels' bivariate Granger test for A → B (does A cause B?).

    statsmodels' :func:`grangercausalitytests` expects a 2-column array
    where the **first** column is the target ``Y`` and the **second** is
    the candidate predictor ``X`` (the test asks "does X help predict Y").
    For "A Granger-causes B" we therefore pass columns ``[B, A]``.
    """
    from statsmodels.tools.sm_exceptions import InfeasibleTestError
    from statsmodels.tsa.stattools import grangercausalitytests

    joint = pd.concat({"a": a, "b": b}, axis=1).dropna()
    n = len(joint)
    min_needed = max(20, 4 * maxlag + 2)
    if n < min_needed:
        raise HTTPException(
            status_code=422,
            detail=(
                f"need >= {min_needed} aligned daily obs for maxlag={maxlag}, "
                f"got {n}. Widen the date window or pick a smaller maxlag."
            ),
        )

    arr = joint[["b", "a"]].to_numpy()
    try:
        out = grangercausalitytests(arr, maxlag=maxlag, verbose=False)
    except InfeasibleTestError as e:
        # One series is a deterministic linear function of the other; the
        # F-test is undefined (zero residual variance under the unrestricted
        # model). Surface as 422 with the upstream message preserved.
        raise HTTPException(
            status_code=422,
            detail=(
                "Granger test is infeasible — the two series are perfectly "
                f"collinear at one of the lags 1..{maxlag} ({e!s})."
            ),
        ) from e

    tests: list[GrangerLagTest] = []
    for lag, (test_stats, _models) in sorted(out.items(), key=lambda kv: int(kv[0])):
        ssr_f = test_stats.get("ssr_ftest", (float("nan"),) * 4)
        f_stat = _safe_float(ssr_f[0])
        p_value = _safe_float(ssr_f[1])
        significant = not math.isnan(p_value) and not math.isnan(f_stat) and p_value < _SIGNIFICANCE
        tests.append(
            GrangerLagTest(
                lag=int(lag),
                f_stat=f_stat,
                p_value=p_value,
                significant=bool(significant),
            )
        )
    return tests, n


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/quant/granger",
    response_model=GrangerResponse,
    summary="Granger causality test (does past A help predict B?)",
)
def granger_endpoint(
    *,
    a: Annotated[
        str,
        Query(min_length=1, description="Predictor factor id or slug."),
    ],
    b: Annotated[
        str,
        Query(min_length=1, description="Target factor id or slug."),
    ],
    maxlag: Annotated[
        int,
        Query(
            ge=_MAXLAG_LOWER,
            le=_MAXLAG_UPPER,
            description=(
                f"Maximum lag to test (in days). Must be in "
                f"[{_MAXLAG_LOWER}, {_MAXLAG_UPPER}]. Default 5."
            ),
        ),
    ] = 5,
    start: Annotated[
        date | None,
        Query(description="UTC start date (inclusive). Default: end - 365d."),
    ] = None,
    end: Annotated[
        date | None,
        Query(description="UTC end date (inclusive). Default: today."),
    ] = None,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> GrangerResponse:
    """Test whether past values of ``a`` help predict ``b``.

    The verdict ``a_granger_causes_b`` is true iff the lag with the lowest
    F-test p-value has ``p < 0.05``. The full per-lag table is returned so
    callers can apply their own multiple-testing correction.

    Refuses (422) self-tests (``a == b``) — Granger of a series with itself
    is mechanically infeasible (perfect collinearity under the unrestricted
    model) and almost certainly a user-typo.
    """
    if a == b:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a and b must differ — got both = {a!r}. Granger of a "
                "series against itself is undefined."
            ),
        )

    fa = _resolve_slug(a, factors, role="a")
    fb = _resolve_slug(b, factors, role="b")
    if fa.id == fb.id:
        # Same factor under two different aliases.
        raise HTTPException(
            status_code=422,
            detail=(
                f"a={a!r} and b={b!r} resolve to the same factor "
                f"({fa.id!r}); pass two distinct factors."
            ),
        )

    end_date = end or date.today()
    start_date = start or (end_date - timedelta(days=_DEFAULT_HISTORY_DAYS))
    if start_date >= end_date:
        raise HTTPException(status_code=422, detail="start must be strictly before end")
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")

    s_a = _fetch_series(fa, start_ts, end_ts, poly, cache, settings)
    s_b = _fetch_series(fb, start_ts, end_ts, poly, cache, settings)

    tests, n_obs = _run_granger(s_a, s_b, maxlag)

    # Pick the lag with the lowest finite p-value. Ties broken by smallest lag.
    finite_tests = [t for t in tests if not math.isnan(t.p_value)]
    if finite_tests:
        best = min(finite_tests, key=lambda t: (t.p_value, t.lag))
        best_lag: int | None = best.lag
        best_p: float | None = best.p_value
        a_causes_b = best.p_value < _SIGNIFICANCE
    else:
        best_lag = None
        best_p = None
        a_causes_b = False

    return GrangerResponse(
        a=fa.id,
        b=fb.id,
        maxlag=int(maxlag),
        n_obs=int(n_obs),
        tests=tests,
        best_lag=best_lag,
        best_p_value=best_p,
        a_granger_causes_b=bool(a_causes_b),
    )


__all__ = [
    "GrangerLagTest",
    "GrangerResponse",
    "granger_endpoint",
    "router",
]
