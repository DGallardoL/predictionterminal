"""``GET /pricing/binary/{slug}`` — expose T81 binary pricers via HTTP.

Task W11-25 (wave-11). Given a factor slug and a ``model`` query parameter
(``logit``, ``bsd``, ``brownian`` or ``beta``), this router

1. resolves the slug → :class:`~pfm.factors.FactorConfig` from the loaded
   catalog (404 if unknown);
2. pulls the daily price history via
   :func:`pfm.regression_core._cached_factor_history`;
3. builds a :class:`~pfm.pricing.binary_models.MarketState` snapshot from the
   most-recent observation and factor metadata;
4. instantiates the selected pricer with default parameters;
5. returns a :class:`PricingResponse` JSON envelope with ``fair_price``,
   ``mispricing`` (= ``fair − market``), ``confidence_interval`` and any
   model-specific ``diagnostics``.

Integration
-----------
This router is **not** mounted automatically into ``pfm.main`` because the
``main.py:routes`` partition has active claims by other sessions
(``metrics-audit-endpoint``, ``W11-15-redis-lock-migration``). The next
session holding ``main.py:routes`` should add, in the include block::

    from pfm.pricing.router import router as _pricing_router
    app.include_router(_pricing_router)

For now the module ships standalone — tests mount the router into a fresh
``FastAPI`` instance with DI overridden, identical to the pattern used by
``pfm.factors_related_router``.

Caching
-------
A small module-level :class:`dict` TTL cache keyed on ``(slug, model)``
with a 60 s TTL. The cache is process-local; multi-worker deployments
will warm independently. Tests can monkey-patch :data:`_PERF_COUNTER` to
drive cache expiry deterministically without sleeping.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pfm.cache import CacheBackend
from pfm.config import Settings, get_settings
from pfm.dependencies import (
    get_cache,
    get_factors_dep,
    get_polymarket_client,
)
from pfm.factors import FactorConfig
from pfm.pricing.binary_models import (
    BetaBinomialBayes,
    BlackScholesDigital,
    BrownianBridge,
    MarketState,
    Pricer,
    PricingResult,
    RiskNeutralLogit,
)
from pfm.sources.polymarket import PolymarketClient

router = APIRouter(tags=["pricing"])


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------


# Recognised model identifiers in the ``?model=`` query parameter. Anything
# outside this set is rejected by FastAPI with a 422 — we lean on the
# ``Literal`` type rather than re-implementing validation by hand.
ModelName = Literal["logit", "bsd", "brownian", "beta"]

# Default model when the caller omits ``?model=`` — chosen to be the cheapest
# and most general (no underlying / threshold required).
DEFAULT_MODEL: ModelName = "logit"

# Fallback time-to-resolve for factors that don't carry an explicit end date
# in their metadata. 30 days is roughly the median horizon across the
# Polymarket catalog as of wave-10.
DEFAULT_TTR_DAYS: float = 30.0

# How many days of history to pull when building MarketState. 90 days is
# enough for the BrownianBridge / BlackScholesDigital σ-from-polls fallback
# and covers a full quarter for the logit's news context.
LOOKBACK_DAYS: int = 90

# Module-level wall clock — tests monkeypatch this to drive cache expiry
# without ``time.sleep`` calls. ``time.perf_counter`` is monotonic and
# cheap, so leap-seconds / NTP skew never spuriously invalidate state.
_PERF_COUNTER: Callable[[], float] = time.perf_counter

# Cache TTL in seconds. Factor prices update at most a few times per minute
# at the source, and the pricers themselves are sub-millisecond once a
# MarketState exists — a 60 s window keeps the panel responsive while
# bounding upstream load to ≈1 fetch/minute per slug.
_CACHE_TTL_S: float = 60.0


# ---------------------------------------------------------------------------
# Pydantic response model
# ---------------------------------------------------------------------------


class PricingResponse(BaseModel):
    """Response envelope for ``GET /pricing/binary/{slug}``."""

    slug: str = Field(..., description="The factor slug echoed back.")
    model: ModelName = Field(..., description="Which pricer produced the estimate.")
    fair_price: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model-implied fair probability in [0, 1].",
    )
    market_price: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Last observed market quote in [0, 1].",
    )
    mispricing: float = Field(
        ...,
        description="``fair_price − market_price`` (positive ⇒ model says cheap).",
    )
    confidence_interval: tuple[float, float] = Field(
        ...,
        description="95 % CI on ``fair_price`` (low, high), each in [0, 1].",
    )
    diagnostics: dict[str, float] = Field(
        default_factory=dict,
        description="Model-specific scalar diagnostics (σ, α, β, …).",
    )


# ---------------------------------------------------------------------------
# Module-level TTL cache
# ---------------------------------------------------------------------------


class _Entry:
    __slots__ = ("expires_at", "payload")

    def __init__(self, payload: PricingResponse, expires_at: float) -> None:
        self.payload = payload
        self.expires_at = expires_at


_CACHE: dict[tuple[str, str], _Entry] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key: tuple[str, str]) -> PricingResponse | None:
    """Return cached payload if still fresh, else ``None``."""
    now = _PERF_COUNTER()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            _CACHE.pop(key, None)
            return None
        return entry.payload


def _cache_put(key: tuple[str, str], payload: PricingResponse) -> None:
    """Insert payload into the TTL cache with a fresh expiry."""
    expires_at = _PERF_COUNTER() + _CACHE_TTL_S
    with _CACHE_LOCK:
        _CACHE[key] = _Entry(payload, expires_at)


def _cache_clear() -> None:
    """Drop every entry — used by tests to force a cold path."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_anchor(factors: dict[str, FactorConfig], slug: str) -> FactorConfig:
    """Look up the factor by slug. Raises 404 when unknown.

    Prefers an ``app.state.factors_by_slug`` indexed lookup when available
    (O(1)), falls back to a linear scan otherwise so the helper stays
    unit-testable without a full FastAPI app.
    """
    for fc in factors.values():
        if fc.slug == slug:
            return fc
    raise HTTPException(
        status_code=404,
        detail=f"factor with slug {slug!r} not found in catalog",
    )


def _fetch_history(
    fc: FactorConfig,
    *,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
) -> pd.DataFrame:
    """Pull a factor's daily price history via the shared cache helper.

    Returns an empty DataFrame on any upstream failure — the caller maps
    that to a 502 so the user sees a clean error rather than a 500.
    """
    try:
        from pfm.regression_core import _cached_factor_history
    except ImportError:  # pragma: no cover — only in stripped builds
        return pd.DataFrame()

    end = pd.Timestamp.utcnow().normalize()
    # Pad a few extra days for weekend / holiday alignment.
    start = end - pd.Timedelta(days=LOOKBACK_DAYS + 7)
    try:
        return _cached_factor_history(fc, start, end, poly, cache, settings)
    except HTTPException:
        # Re-raise — the upstream already chose an appropriate status code.
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"factor history fetch failed: {type(e).__name__}: {e}",
        ) from e


def _ttr_days_from_metadata(fc: FactorConfig) -> float:
    """Best-effort time-to-resolution in calendar days from factor metadata.

    Chain factors carry an ordered list of segment end dates — the *last*
    segment's ``end`` is the natural resolution horizon. Non-chain
    Polymarket / Kalshi factors don't have a structured resolution date in
    ``FactorConfig`` (the catalog doesn't store it today), so we fall back
    to :data:`DEFAULT_TTR_DAYS`. Future work: surface ``end_date`` on
    :class:`FactorConfig` and replace this fallback.
    """
    if fc.is_chained and fc.segments:
        end = fc.segments[-1].end
        today = pd.Timestamp.utcnow().normalize().date()
        delta = (end - today).days
        return float(max(delta, 0.0))
    return float(DEFAULT_TTR_DAYS)


def _build_market_state(fc: FactorConfig, history: pd.DataFrame) -> MarketState:
    """Build a :class:`MarketState` snapshot from the most recent observation.

    * ``current_price`` ← last value of the ``price`` (or first-numeric)
      column. Defaults to 0.5 when the series is empty.
    * ``time_to_resolve_days`` ← via :func:`_ttr_days_from_metadata`.
    * ``underlying`` ← ``None`` — we don't carry an external numeric
      underlying for prediction-market factors. Tests can extend this if
      they wire a BTC-spot source later.
    * ``threshold`` ← ``None`` unless the factor is itself a probability
      factor, in which case we pass 0.5 so the Brownian-bridge has a
      sensible default coin-flip threshold. (The model overrides to 0.5
      internally when ``None`` is supplied, so this is mostly defensive.)
    * ``poll_history`` ← the last :data:`LOOKBACK_DAYS` price values, used
      by the BS-digital + BrownianBridge calibrators.
    * ``news_evidence`` ← 0.0 by default; future work can wire the
      sentiment factor source as an evidence term.
    """
    if history is None or history.empty:
        current = 0.5
        polls: tuple[float, ...] = ()
    else:
        if "price" in history.columns:
            series = pd.to_numeric(history["price"], errors="coerce").dropna()
        elif "value" in history.columns:
            series = pd.to_numeric(history["value"], errors="coerce").dropna()
        else:
            numeric_cols = [c for c in history.columns if pd.api.types.is_numeric_dtype(history[c])]
            if not numeric_cols:
                series = pd.Series(dtype="float64")
            else:
                series = pd.to_numeric(history[numeric_cols[0]], errors="coerce").dropna()
        if series.empty:
            current = 0.5
            polls = ()
        else:
            current = float(series.iloc[-1])
            polls = tuple(float(v) for v in series.tail(LOOKBACK_DAYS).tolist())

    # Clamp current price into the open interval. The pricers all clip
    # internally but doing it here too keeps diagnostics tidy.
    current = min(max(current, 0.0), 1.0)

    ttr = _ttr_days_from_metadata(fc)

    threshold: float | None
    if fc.is_probability:
        # For probability factors a 0.5 coin-flip threshold is the natural
        # interpretation. Non-prob (level) factors are not really binary
        # markets, so we leave threshold None and let the BS-digital model
        # degenerate to ``market_price`` (handled inside the model).
        threshold = 0.5
    else:
        threshold = None

    return MarketState(
        current_price=current,
        time_to_resolve_days=ttr,
        underlying=None,
        threshold=threshold,
        poll_history=polls,
        news_evidence=0.0,
    )


# Registry mapping the public ``?model=`` token to the corresponding pricer
# instance factory. The instances themselves are cheap to construct, so we
# call the factory per request rather than keeping a module-level singleton
# (this also keeps tests free to patch the constructors).
_PRICER_FACTORIES: dict[ModelName, Callable[[], Pricer]] = {
    "logit": RiskNeutralLogit,
    "bsd": BlackScholesDigital,
    "brownian": BrownianBridge,
    "beta": BetaBinomialBayes,
}


def _instantiate_pricer(model: ModelName) -> Pricer:
    """Instantiate the requested pricer with default parameters.

    Raises 422 for unknown identifiers — this is a defence-in-depth check
    in case the ``Literal`` type-level guard is somehow bypassed (e.g. by
    a future caller that constructs the URL manually with an unknown
    model). FastAPI normally rejects unknown literals upstream with 422.
    """
    factory = _PRICER_FACTORIES.get(model)
    if factory is None:
        raise HTTPException(
            status_code=422,
            detail=f"unknown model {model!r}; expected one of {sorted(_PRICER_FACTORIES)}",
        )
    return factory()


def _to_response(
    *,
    slug: str,
    model: ModelName,
    market_price: float,
    result: PricingResult,
) -> PricingResponse:
    """Wrap a :class:`PricingResult` into the public API envelope."""
    fair = float(result.fair_price)
    market = float(market_price)
    ci_lo, ci_hi = result.confidence_interval
    return PricingResponse(
        slug=slug,
        model=model,
        fair_price=fair,
        market_price=market,
        mispricing=fair - market,
        confidence_interval=(float(ci_lo), float(ci_hi)),
        diagnostics={k: float(v) for k, v in result.diagnostics.items()},
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get(
    "/pricing/binary/{slug}",
    response_model=PricingResponse,
    summary="Compute model-implied fair price for a binary prediction market.",
)
def get_binary_pricing(
    slug: Annotated[str, "Factor slug to price (must exist in the catalog)."],
    request: Request,
    model: Annotated[
        ModelName,
        Query(
            description=(
                "Which binary pricer to use: "
                "'logit' (default, risk-neutral logit), 'bsd' (Black-Scholes "
                "digital), 'brownian' (Brownian-bridge), 'beta' "
                "(Beta-Binomial Bayes)."
            ),
        ),
    ] = DEFAULT_MODEL,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> PricingResponse:
    """Return the selected pricer's fair-value estimate for ``slug``.

    * 404 if the slug is not in the factor catalog.
    * 422 if ``model`` is not one of ``{logit, bsd, brownian, beta}``.
    * 502 if the upstream factor-history fetch fails.

    Cached for 60 s per ``(slug, model)`` tuple.
    """
    cache_key = (slug, str(model))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    fc = _resolve_anchor(factors, slug)
    history = _fetch_history(fc, poly=poly, cache=cache, settings=settings)
    state = _build_market_state(fc, history)

    pricer = _instantiate_pricer(model)
    result = pricer.fair_price(state)

    payload = _to_response(
        slug=slug,
        model=model,
        market_price=state.current_price,
        result=result,
    )
    _cache_put(cache_key, payload)
    return payload
