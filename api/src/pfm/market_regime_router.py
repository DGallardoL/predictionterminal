"""FastAPI router for the simple market-regime classification stub.

Single endpoint:

* ``GET /market/regime`` — research-only regime label derived from VIX
  level and the SPY 50-day slope.

The classification is intentionally *trivial* — a rule-based stub, not a
hidden-Markov model. The intent is to ship a clean read-only surface for
the Terminal "regime banner" without implying a tradeable signal.

Regime rules
------------
``risk_state``
    * ``risk_off`` when VIX > 25
    * ``risk_on``  when VIX < 15
    * ``neutral``  otherwise (15 <= VIX <= 25)

``trend``
    Sign of the OLS slope of the last 50 SPY closes:
    * ``bullish``   when slope > +slope_eps
    * ``bearish``   when slope < -slope_eps
    * ``sideways``  when |slope| <= slope_eps

A short narrative string combines both axes, e.g.
``"Currently neutral risk + bullish trend -> defensive growth bias"``.

CLAUDE.md anti-alpha rule
-------------------------
The model has not been backtested across regimes and historically
regime-driven signals (recession-odds defensive long, senate-control
short-vol, etc.) have failed 4-quarter robustness on this codebase.
The response therefore carries an explicit
``research_only: true`` flag and a ``disclaimer`` string. The endpoint
must NOT be wired into any strategy router or alpha pipeline.

Routing
-------
This module owns its :class:`fastapi.APIRouter` but is **not**
auto-mounted. Wire it into ``main.py`` explicitly::

    from pfm.market_regime_router import router as market_regime_router
    app.include_router(market_regime_router)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (exposed for tests and future tuning)
# ---------------------------------------------------------------------------

#: TTL for the regime response cache. Frontend refreshes hourly anyway.
CACHE_TTL_SECONDS: int = 3600

#: Hint clients can use to schedule a re-poll. Matches ``CACHE_TTL_SECONDS``.
REFRESH_HINT_SECONDS: int = 3600

#: VIX thresholds. Mapped from the canonical "panic at 30" convention to
#: the slightly tighter band a hedged book typically cares about.
VIX_RISK_OFF: float = 25.0
VIX_RISK_ON: float = 15.0

#: SPY trend window. 50 trading days ~= 10 calendar weeks.
TREND_WINDOW_DAYS: int = 50

#: Slope sign threshold. Expressed in $/day on the SPY close. Anything
#: shallower than this is treated as ``sideways``. 0.05 ~= 1.0% over the
#: 50d window on a 400-level SPY.
SLOPE_EPSILON: float = 0.05

_NAMESPACE = "market_regime"

# Type aliases for the categorical labels.
RiskState = Literal["risk_off", "risk_on", "neutral"]
Trend = Literal["bullish", "bearish", "sideways"]


# ---------------------------------------------------------------------------
# Pydantic response schema
# ---------------------------------------------------------------------------


class RegimeResponse(BaseModel):
    """Public response model for ``GET /market/regime``."""

    vix: float = Field(..., description="Latest VIX close.")
    spy_slope: float = Field(
        ...,
        description=(f"OLS slope (units: $/day) of the last {TREND_WINDOW_DAYS} SPY closes."),
    )
    regime: RiskState = Field(
        ...,
        description="Risk-state label derived from the VIX level.",
    )
    trend: Trend = Field(
        ...,
        description="Trend label derived from the SPY 50d slope sign.",
    )
    narrative: str = Field(
        ...,
        description=(
            "Human-readable summary combining ``regime`` and ``trend``,"
            " e.g. 'Currently neutral risk + bullish trend -> defensive"
            " growth bias'."
        ),
    )
    computed_at: str = Field(
        ...,
        description="ISO-8601 UTC timestamp when this snapshot was computed.",
    )
    refresh_seconds: int = Field(
        REFRESH_HINT_SECONDS,
        description="Client-side refresh hint, in seconds.",
    )
    research_only: bool = Field(
        True,
        description=(
            "Always ``true``. The regime classification is a stub for the"
            " Terminal banner and is NOT a tradeable signal."
        ),
    )
    disclaimer: str = Field(
        ...,
        description="Explicit research-only / anti-alpha disclaimer.",
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/market", tags=["market-regime"])


# ---------------------------------------------------------------------------
# Helpers (overridable in tests via monkeypatch)
# ---------------------------------------------------------------------------


_DISCLAIMER: str = (
    "Research-only regime stub. The rule-based labels above have not "
    "passed 4-quarter robustness; regime-driven signals are on the "
    "anti-alpha list and MUST NOT be deployed as a tradeable signal."
)


def _fetch_vix(now: pd.Timestamp) -> float:
    """Return the latest VIX close as a float.

    Pulls the last ~10 trading days of ``^VIX`` from the equity source
    cascade and returns the most recent close. The function is broken
    out so tests can monkeypatch a deterministic value without standing
    up an HTTP mock.

    Raises:
        HTTPException(502): when every upstream equity source fails.
    """
    from pfm.sources.equity import EquityDataError, _try_stooq, _try_tiingo, _try_yfinance

    start = (now - pd.Timedelta(days=20)).normalize()
    end = now.normalize()
    errors: dict[str, str] = {}
    closes: pd.Series | None = None
    for label, fn in (
        ("yfinance", _try_yfinance),
        ("tiingo", _try_tiingo),
        ("stooq", _try_stooq),
    ):
        try:
            closes = fn("^VIX", start, end)
            break
        except Exception as exc:  # pragma: no cover - defensive
            errors[label] = f"{type(exc).__name__}: {exc}"
    if closes is None or len(closes) == 0:
        detail = "; ".join(f"{k}={v}" for k, v in errors.items()) or "no sources tried"
        raise EquityDataError(f"VIX fetch failed: {detail}")
    return float(closes.iloc[-1])


def _fetch_spy_closes(now: pd.Timestamp, window: int = TREND_WINDOW_DAYS) -> pd.Series:
    """Return the last ``window`` SPY closes as a pandas Series.

    Pulls ~80 calendar days to be sure we have ``window`` trading days
    on hand (markets, holidays). Tests monkeypatch this directly.

    Raises:
        HTTPException(502) or EquityDataError on upstream failure.
        ValueError if fewer than ``window`` closes are returned.
    """
    from pfm.sources.equity import EquityDataError, _try_stooq, _try_tiingo, _try_yfinance

    start = (now - pd.Timedelta(days=window + 30)).normalize()
    end = now.normalize()
    errors: dict[str, str] = {}
    closes: pd.Series | None = None
    for label, fn in (
        ("yfinance", _try_yfinance),
        ("tiingo", _try_tiingo),
        ("stooq", _try_stooq),
    ):
        try:
            closes = fn("SPY", start, end)
            break
        except Exception as exc:  # pragma: no cover - defensive
            errors[label] = f"{type(exc).__name__}: {exc}"
    if closes is None or len(closes) == 0:
        detail = "; ".join(f"{k}={v}" for k, v in errors.items()) or "no sources tried"
        raise EquityDataError(f"SPY fetch failed: {detail}")
    if len(closes) < window:
        raise ValueError(f"need >= {window} SPY closes for trend slope, got {len(closes)}")
    return closes.iloc[-window:].astype(float)


def _classify_risk(vix: float) -> RiskState:
    """Map a VIX level to a risk-state label using the module thresholds."""
    if vix > VIX_RISK_OFF:
        return "risk_off"
    if vix < VIX_RISK_ON:
        return "risk_on"
    return "neutral"


def _spy_slope(closes: pd.Series) -> float:
    """OLS slope of ``closes`` vs an integer index (units: $/day)."""
    y = np.asarray(closes, dtype=float)
    if y.size < 2:
        raise ValueError("need >= 2 SPY closes to compute slope")
    x = np.arange(y.size, dtype=float)
    # numpy.polyfit returns highest-degree first; slope is index 0.
    slope, _intercept = np.polyfit(x, y, deg=1)
    return float(slope)


def _classify_trend(slope: float, eps: float = SLOPE_EPSILON) -> Trend:
    """Sign of the slope, with a dead band of ``+/- eps``."""
    if slope > eps:
        return "bullish"
    if slope < -eps:
        return "bearish"
    return "sideways"


# A 3x3 narrative grid. Risk on the rows, trend on the columns. Bias text
# is intentionally vague; this is a banner string, not a recommendation.
_NARRATIVE_GRID: dict[tuple[RiskState, Trend], str] = {
    ("risk_off", "bullish"): "elevated risk + bullish trend -> fade rallies bias",
    ("risk_off", "bearish"): "elevated risk + bearish trend -> defensive / cash bias",
    ("risk_off", "sideways"): "elevated risk + sideways trend -> hedged carry bias",
    ("risk_on", "bullish"): "calm risk + bullish trend -> momentum bias",
    ("risk_on", "bearish"): "calm risk + bearish trend -> mean-reversion bias",
    ("risk_on", "sideways"): "calm risk + sideways trend -> range / theta bias",
    ("neutral", "bullish"): "neutral risk + bullish trend -> defensive growth bias",
    ("neutral", "bearish"): "neutral risk + bearish trend -> low-beta value bias",
    ("neutral", "sideways"): "neutral risk + sideways trend -> balanced / wait bias",
}


def _narrative(regime: RiskState, trend: Trend) -> str:
    """Return the human-readable summary string for a (regime, trend) pair."""
    suffix = _NARRATIVE_GRID[(regime, trend)]
    return f"Currently {suffix}"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/regime", response_model=RegimeResponse)
def get_market_regime(
    nocache: bool = Query(False, description="Bypass the 1h response cache."),
) -> dict[str, Any]:
    """Return the current research-only market regime label.

    The response is cached for ``CACHE_TTL_SECONDS`` (1h) under the
    ``market_regime`` namespace. Pass ``?nocache=true`` to force a
    re-fetch (useful in admin UIs and tests).
    """
    cache = get_cache(_NAMESPACE, ttl=CACHE_TTL_SECONDS)
    cache_key = ("regime", "v1")
    if not nocache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    now = pd.Timestamp.now(tz="UTC")
    try:
        vix = _fetch_vix(now)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"VIX fetch failed: {exc}") from exc

    try:
        spy_closes = _fetch_spy_closes(now)
        slope = _spy_slope(spy_closes)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"SPY fetch failed: {exc}") from exc

    regime = _classify_risk(vix)
    trend = _classify_trend(slope)
    narrative = _narrative(regime, trend)

    body: dict[str, Any] = {
        "vix": round(vix, 4),
        "spy_slope": round(slope, 6),
        "regime": regime,
        "trend": trend,
        "narrative": narrative,
        "computed_at": datetime.now(tz=UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "refresh_seconds": REFRESH_HINT_SECONDS,
        "research_only": True,
        "disclaimer": _DISCLAIMER,
    }

    cache.set(cache_key, body, ttl=CACHE_TTL_SECONDS)
    return body


# ---------------------------------------------------------------------------
# Cache control (for tests + admin endpoints)
# ---------------------------------------------------------------------------


def cache_clear() -> None:
    """Drop any cached ``/market/regime`` response."""
    cache = get_cache(_NAMESPACE, ttl=CACHE_TTL_SECONDS)
    # TerminalCache exposes ``clear`` consistently across the codebase.
    try:
        cache.clear()
    except Exception:  # pragma: no cover - defensive
        logger.debug("market_regime cache_clear: backend has no clear()")


__all__ = [
    "CACHE_TTL_SECONDS",
    "REFRESH_HINT_SECONDS",
    "SLOPE_EPSILON",
    "TREND_WINDOW_DAYS",
    "VIX_RISK_OFF",
    "VIX_RISK_ON",
    "RegimeResponse",
    "cache_clear",
    "router",
]
