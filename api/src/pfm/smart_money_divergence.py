"""Smart Money Divergence — flag PM-vs-equity flow disagreements.

A divergence event is, in plain English: "whales on Polymarket are *buying*
this thesis but the equity proxy is being *sold* (or vice-versa)." When PM
participants tend to be informed (on certain market types they reliably are
— see e.g. earnings, crypto-ETF, FOMC), persistent disagreement between
PM-flow and equity-flow can foreshadow an equity move toward the PM-implied
view.

Mechanics (POC, deliberately simple):

  - **PM whale flow** (`whale_flow_pm`): net USD flow from large trades
    (synthetic in this POC; in production wire to data-api ``/trades``
    filtered by size threshold).
  - **Equity flow** (`equity_flow`): ``volume × sign(close - vwap)``.
    Positive = net buy pressure intra-day; negative = net sell pressure.
  - **Divergence**: opposite signs *and* both magnitudes above a relevance
    floor. A z-style "strength" combines the two flows on a [0, 1] scale.

A historical lead win-rate is reported per slug — synthetic in POC, but
keyed deterministically by the slug so the UI shows stable numbers across
reloads. Real wiring would replay the divergence detector over a 12-month
window and compute the empirical P(equity moves toward PM-thesis | divergence).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)

CACHE = get_cache("smart_money_divergence", ttl=180)

# Pairs of (PM slug, equity proxy ticker). In production this catalogue lives
# in ``factors.yml`` next to each factor; for the POC we keep an inline map
# of representative pairs for the scanner to iterate.
_DIVERGENCE_UNIVERSE: tuple[tuple[str, str], ...] = (
    ("nvda-eps-beat-q1", "NVDA"),
    ("earnings-beat-aapl", "AAPL"),
    ("tsla-1tn-by-eoy", "TSLA"),
    ("btc-150k-by-eoy", "BTC-USD"),
    ("recession-2026", "SPY"),
    ("fed-cut-march-2026", "TLT"),
    ("oil-100-by-eoy", "USO"),
    ("vix-25-by-jun", "VXX"),
    ("ai-capex-cut-q2", "QQQ"),
    ("cpi-below-3pct", "TIP"),
    ("nfp-positive-may", "XLF"),
    ("election-senate-control", "SPY"),
    ("trump-wins-2024", "DJT"),
    ("geopolitics-mideast", "USO"),
    ("spx-6500-by-eoy", "SPY"),
)


# --- schemas ----------------------------------------------------------------


class DivergenceResult(BaseModel):
    slug: str
    ticker_proxy: str
    lookback_hours: int
    whale_flow_pm: float = Field(..., description="Net USD flow from PM whale trades.")
    equity_flow: float = Field(
        ..., description="Equity-side flow proxy: volume x sign(close - vwap)."
    )
    divergence_strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0 = aligned; 1 = max disagreement with high magnitudes.",
    )
    is_diverging: bool
    historical_lead_winrate: float = Field(..., ge=0.0, le=1.0)
    suggested_trade: str
    source: Literal["live", "synthetic"] = "synthetic"


class DivergenceScanResponse(BaseModel):
    min_strength: float
    n_results: int
    results: list[DivergenceResult]


# --- core logic -------------------------------------------------------------


def _seed_for(*parts: str | int | float) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def _synth_pm_whale_flow(slug: str, lookback_hours: int) -> float:
    """Net USD whale flow into ``slug`` over ``lookback_hours``.

    The mean is centred on a slug-stable bias (some markets persistently
    attract YES whale demand), with hourly-vol scaling. Output spans roughly
    ±$200k for a 24h window.
    """
    rng = np.random.default_rng(_seed_for("pmflow", slug, lookback_hours))
    bias = rng.normal(0.0, 50_000.0)
    hourly_kicks = rng.normal(0.0, 8_000.0, size=lookback_hours)
    return float(bias + hourly_kicks.sum())


def _synth_equity_flow(ticker: str, lookback_hours: int) -> float:
    """Equity-side flow proxy: ``Σ_i volume_i × sign(close_i - vwap_i)``."""
    rng = np.random.default_rng(_seed_for("eqflow", ticker, lookback_hours))
    bias = rng.normal(0.0, 60_000.0)
    hourly_kicks = rng.normal(0.0, 12_000.0, size=lookback_hours)
    return float(bias + hourly_kicks.sum())


def _historical_winrate(slug: str) -> float:
    """Deterministic historical lead win-rate proxy for ``slug``.

    Centred at 0.55 so the average displayed market looks marginally
    informative (slightly better than coinflip), spread ±0.15.
    """
    rng = np.random.default_rng(_seed_for("winrate", slug))
    return float(np.clip(rng.normal(0.55, 0.08), 0.30, 0.85))


def _strength(whale_flow: float, equity_flow: float) -> float:
    """Combine magnitudes + sign-disagreement into a [0, 1] strength.

    Algorithm:
      1. If signs agree → strength is dampened (no divergence).
      2. Otherwise: ``min(|a|, |b|) / max(|a|, |b|, 1.0)`` × magnitude
         saturation. This rewards *both* sides being large; tiny
         opposite-sign noise gets ~0.
    """
    a, b = abs(whale_flow), abs(equity_flow)
    if a < 1.0 and b < 1.0:
        return 0.0
    sign_disagree = np.sign(whale_flow) != np.sign(equity_flow) and a > 0 and b > 0
    if not sign_disagree:
        # Aligned — return a small "consonance" score, capped under 0.3.
        return float(0.2 * (min(a, b) / max(a, b, 1.0)))
    balance = min(a, b) / max(a, b, 1.0)
    # Magnitude saturation kicks in around $50k combined notional.
    magnitude = (a + b) / ((a + b) + 100_000.0)
    return float(np.clip(balance * magnitude * 1.6, 0.0, 1.0))


def _suggested_trade(slug: str, ticker: str, whale_flow: float, equity_flow: float) -> str:
    """One-line natural-language trade suggestion."""
    if whale_flow > 0 and equity_flow < 0:
        return f"Long {ticker} (PM whales bid {slug} YES; equity flow lags)."
    if whale_flow < 0 and equity_flow > 0:
        return f"Short {ticker} (PM whales hit {slug} NO; equity buying overshooting)."
    return f"No clear cross-asset trade; flows are aligned in {slug}/{ticker}."


def detect_divergence(
    slug: str,
    ticker_proxy: str,
    lookback_hours: int = 24,
) -> dict:
    """Compute a single PM-vs-equity divergence snapshot for ``(slug, ticker_proxy)``.

    Returns a dict shaped like :class:`DivergenceResult`. The ``is_diverging``
    flag is True when *both* flows exceed a $25k magnitude floor and have
    opposite signs.
    """
    cache_key = ("detect", slug, ticker_proxy, int(lookback_hours))
    cached = CACHE.get(cache_key)
    if cached is not None:
        return cached

    whale_flow = _synth_pm_whale_flow(slug, lookback_hours)
    equity_flow = _synth_equity_flow(ticker_proxy, lookback_hours)
    strength = _strength(whale_flow, equity_flow)

    significant = abs(whale_flow) > 25_000.0 and abs(equity_flow) > 25_000.0
    is_diverging = bool(significant and (np.sign(whale_flow) != np.sign(equity_flow)))

    result = DivergenceResult(
        slug=slug,
        ticker_proxy=ticker_proxy,
        lookback_hours=lookback_hours,
        whale_flow_pm=round(whale_flow, 2),
        equity_flow=round(equity_flow, 2),
        divergence_strength=round(strength, 4),
        is_diverging=is_diverging,
        historical_lead_winrate=round(_historical_winrate(slug), 4),
        suggested_trade=_suggested_trade(slug, ticker_proxy, whale_flow, equity_flow),
        source="synthetic",
    ).model_dump()

    CACHE.set(cache_key, result, ttl=180)
    return result


def scan_all_divergences(min_strength: float = 0.5) -> list[dict]:
    """Run :func:`detect_divergence` over the universe and rank by strength.

    Returns up to 10 results with ``divergence_strength >= min_strength``,
    sorted descending.
    """
    cache_key = ("scan", round(min_strength, 4))
    cached = CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows: list[dict] = []
    for slug, ticker in _DIVERGENCE_UNIVERSE:
        rows.append(detect_divergence(slug, ticker, lookback_hours=24))

    rows = [r for r in rows if r["divergence_strength"] >= min_strength]
    rows.sort(key=lambda r: r["divergence_strength"], reverse=True)
    rows = rows[:10]

    CACHE.set(cache_key, rows, ttl=180)
    return rows


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/divergence", tags=["smart-money-divergence"])


@router.get(
    "/smart-money",
    response_model=DivergenceScanResponse,
    summary="Top PM-vs-equity flow divergences across the universe.",
)
def get_smart_money_divergences(
    min_strength: Annotated[float, Query(ge=0.0, le=1.0)] = 0.5,
) -> DivergenceScanResponse:
    rows = scan_all_divergences(min_strength=min_strength)
    return DivergenceScanResponse(
        min_strength=min_strength,
        n_results=len(rows),
        results=[DivergenceResult(**r) for r in rows],
    )


@router.get(
    "/{slug}",
    response_model=DivergenceResult,
    summary="Divergence snapshot for a single (slug, default-ticker) pair.",
)
def get_divergence_for_slug(
    slug: Annotated[str, Path(min_length=1)],
    ticker_proxy: Annotated[str | None, Query(min_length=1, max_length=10)] = None,
    lookback_hours: Annotated[int, Query(ge=1, le=168)] = 24,
) -> DivergenceResult:
    # Resolve a default ticker_proxy from the universe if caller didn't supply one.
    if ticker_proxy is None:
        match = next((t for s, t in _DIVERGENCE_UNIVERSE if s == slug), None)
        if match is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"slug {slug!r} not in divergence universe; "
                    "supply ?ticker_proxy=… to compute on-demand"
                ),
            )
        ticker_proxy = match

    payload = detect_divergence(slug, ticker_proxy, lookback_hours=lookback_hours)
    return DivergenceResult(**payload)


__all__ = [
    "DivergenceResult",
    "DivergenceScanResponse",
    "detect_divergence",
    "router",
    "scan_all_divergences",
]
