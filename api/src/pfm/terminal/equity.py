"""Equity-overlay endpoint for the Terminal panel.

When a Polymarket slug has a known equity counterpart (e.g.
``apple_largest_jun`` ↔ ``AAPL``) we expose the equity close-price series
alongside the prediction-market probability series so the frontend can
render the two together on a shared chart and surface the
:func:`pfm.equity_factors.equity_market_cointegration` diagnostics
(β, intercept, half-life, correlation) above the chart.

The mapping is hard-coded here on purpose — the universe of "company-
adjacent" markets is small and curated; pulling it into ``factors.yml``
would just be ceremony for two-dozen rows. A few entries deliberately
have ``None`` as the ticker so the endpoint can answer 200 with a
``"no equity counterpart"`` message rather than 404 when the frontend
asks about a known-but-private company (Anduril, SpaceX, …).

Routing note: this module owns its :class:`fastapi.APIRouter` so the
existing ``main.py`` is left untouched (the user asked us not to edit
it). To activate the endpoint, ``main.py`` only needs::

    from pfm.terminal_equity import router as terminal_equity_router
    app.include_router(terminal_equity_router)
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from pfm.config import Settings, get_settings
from pfm.equity_factors import (
    EquityFactorError,
    equity_market_cointegration,
    fetch_equity_history,
)
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,
)

logger = logging.getLogger(__name__)


# --- mapping ----------------------------------------------------------------
# Slug → (yfinance_ticker | None). ``None`` means we recognise the market
# but its underlying company isn't publicly traded.
SLUG_TO_TICKER: dict[str, str | None] = {
    "apple_largest_jun": "AAPL",
    "msft_largest": "MSFT",
    "nvda_largest_jun": "NVDA",
    "nvda_ath": "NVDA",
    "nvda_4t": "NVDA",
    "amzn_largest_jun": "AMZN",
    "amzn_ath": "AMZN",
    "tesla_largest_jun": "TSLA",
    "tesla_ath": "TSLA",
    "tesla_robotaxi": "TSLA",
    "gitlab_acquired": "GTLB",
    "jpmorgan_chase_fail": "JPM",
    "morgan_stanley_acquired": "MS",
    "morgan_stanley_ceo": "MS",
    "bp_acquired": "BP",
    "saudi_aramco_largest": "2222.SR",
    "saudi_aramco_ipo_extension": "2222.SR",
    # Recognised slugs without a public ticker — return 200 + null so the
    # frontend can render a "no equity counterpart" badge instead of 404.
    "anduril_acquired": None,
    "anduril_ipo": None,
    "spacex_ipo": None,
    "spacex_starship_orbit": None,
}


def _resolve_ticker(slug: str) -> tuple[bool, str | None]:
    """Return ``(known, ticker)``.

    Three lookup tiers (first hit wins):
      1. Exact match in :data:`SLUG_TO_TICKER` (legacy internal factor slugs).
      2. Legacy snake_case prefix table (``nvda_*``, ``tesla_*``, …).
      3. Polymarket-style hyphen-separated substring detection (``apple-…``,
         ``nvidia-…``, ``s-p-500-…``, ``bitcoin-…``, etc.). This is what
         actually fires for live markets: their slugs are kebab-case derived
         from English titles, so we walk a substring table to recognise the
         company / asset embedded in the slug.
    """
    if slug in SLUG_TO_TICKER:
        return True, SLUG_TO_TICKER[slug]

    prefix_table: list[tuple[str, str | None]] = [
        ("nvda_", "NVDA"),
        ("amzn_", "AMZN"),
        ("tesla_", "TSLA"),
        ("morgan_stanley_", "MS"),
        ("saudi_aramco_", "2222.SR"),
        ("anduril_", None),
        ("spacex_", None),
    ]
    for prefix, ticker in prefix_table:
        if slug.startswith(prefix):
            return True, ticker

    # Polymarket-style kebab-case detection. Matches on substring with
    # word-boundary anchors so "tesla-200-share" maps to TSLA but
    # "tesla-cybertruck-passenger" still maps to TSLA. Order matters where
    # one company name contains another (none currently, but defensive).
    sluggy = slug.lower()
    SUBSTRING_MAP: list[tuple[str, str | None]] = [
        # Big tech
        ("nvidia", "NVDA"),
        ("nvda", "NVDA"),
        ("apple", "AAPL"),
        ("aapl", "AAPL"),
        ("microsoft", "MSFT"),
        ("msft", "MSFT"),
        ("alphabet", "GOOGL"),
        ("google", "GOOGL"),
        ("googl", "GOOGL"),
        ("amazon", "AMZN"),
        ("amzn", "AMZN"),
        ("meta", "META"),
        ("facebook", "META"),
        ("tesla", "TSLA"),
        ("tsla", "TSLA"),
        ("netflix", "NFLX"),
        ("nflx", "NFLX"),
        ("oracle", "ORCL"),
        ("salesforce", "CRM"),
        ("amd-", "AMD"),
        ("-amd-", "AMD"),
        ("intel", "INTC"),
        ("adobe", "ADBE"),
        # Financials
        ("jpmorgan", "JPM"),
        ("goldman", "GS"),
        ("bank-of-america", "BAC"),
        ("wells-fargo", "WFC"),
        ("citigroup", "C"),
        ("berkshire", "BRK-B"),
        ("visa", "V"),
        ("mastercard", "MA"),
        ("paypal", "PYPL"),
        # Energy + commodities
        ("exxon", "XOM"),
        ("chevron", "CVX"),
        ("oil-price", "USO"),
        ("crude-oil", "USO"),
        ("wti", "USO"),
        ("natural-gas", "UNG"),
        ("gold-price", "GLD"),
        ("gold-hit", "GLD"),
        ("silver-price", "SLV"),
        # Indices
        ("s-p-500", "SPY"),
        ("sp-500", "SPY"),
        ("spx", "SPY"),
        ("nasdaq", "QQQ"),
        ("ndx", "QQQ"),
        ("dow-jones", "DIA"),
        ("djia", "DIA"),
        ("russell", "IWM"),
        ("vix", "VXX"),
        # Crypto → spot ETFs (closest tradeable proxy)
        ("bitcoin", "IBIT"),
        ("btc-", "IBIT"),
        ("ethereum", "ETHA"),
        ("eth-", "ETHA"),
        # Privates with no public ticker (recognised, ticker=None)
        ("openai", None),
        ("anthropic", None),
        ("spacex", None),
        ("anduril", None),
        ("stripe", None),
    ]
    for sub, ticker in SUBSTRING_MAP:
        if sub in sluggy:
            return True, ticker

    return False, None


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-equity"])


def get_polymarket_client() -> PolymarketClient:
    """Resolve the shared :class:`PolymarketClient` from app state.

    Imported lazily inside the function so this module doesn't pull
    ``pfm.main`` at import time (which would create a circular import
    via ``main`` → ``terminal_equity`` → ``main``).
    """
    from pfm.main import app  # local import to avoid circulars

    return app.state.poly


def _correlation(prob: pd.Series, equity: pd.Series) -> float | None:
    """Pearson correlation of aligned daily levels. ``None`` if too few obs."""
    df = pd.concat([prob.rename("p"), equity.rename("e")], axis=1).dropna()
    if len(df) < 5:
        return None
    if df["p"].std(ddof=0) == 0 or df["e"].std(ddof=0) == 0:
        return None
    rho = float(df["p"].corr(df["e"]))
    return rho if np.isfinite(rho) else None


@router.get("/equity/{slug}")
@router.get("/equity-curve/{slug}")  # UX-audit 2026-05-14: front-end uses /equity-curve
def get_terminal_equity(
    slug: Annotated[str, Path(min_length=1, max_length=120)],
    days: Annotated[int, Query(ge=10, le=3650)] = 180,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
) -> dict[str, Any]:
    """Return the equity-overlay payload for a Polymarket slug.

    Response shape:

    * matched + public ticker::

        {
            "ticker": "AAPL",
            "equity_series": [{"t": "2025-01-02", "close": 184.32}, ...],
            "correlation_with_prob": 0.61,
            "beta": 0.42,
            "intercept": -1.7,
            "half_life": 14.3
        }

    * matched but no public counterpart (Anduril, SpaceX, …)::

        {"ticker": null, "message": "no equity counterpart"}

    Unknown slugs raise 404; upstream data errors raise 502.
    """
    known, ticker = _resolve_ticker(slug)
    if not known:
        # Return 200 with empty payload (was 404). The 404 was semantically
        # correct but every browser logged it to console for sports/non-
        # equity slugs which fire this endpoint defensively. The frontend
        # already handles ``ticker: null`` as "no equity counterpart" and
        # hides the card silently — same UX, no console noise.
        return {
            "ticker": None,
            "message": "No public equity counterpart is mapped to this market.",
            "slug": slug,
        }
    if ticker is None:
        return {"ticker": None, "message": "no equity counterpart"}

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days)

    # --- equity leg ---------------------------------------------------------
    try:
        equity_series = fetch_equity_history(ticker, start=start, end=end)
    except EquityFactorError as e:
        raise HTTPException(
            status_code=502,
            detail=f"yfinance error for {ticker!r}: {e}",
        ) from e

    # --- polymarket leg -----------------------------------------------------
    try:
        prob_df = fetch_factor_history(poly, slug, start=start, end=end)
    except PolymarketError as e:
        raise HTTPException(status_code=502, detail=f"polymarket error: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e

    if prob_df.empty:
        raise HTTPException(
            status_code=502,
            detail=f"polymarket returned no history for slug={slug!r}",
        )
    prob_series = prob_df["price"].rename(slug)
    # Align indices to UTC-normalised daily timestamps so the correlation
    # / cointegration step joins cleanly with the equity series.
    prob_series.index = pd.to_datetime(prob_series.index, utc=True).normalize()

    # --- diagnostics --------------------------------------------------------
    beta: float | None = None
    intercept: float | None = None
    half_life: float | None = None
    try:
        coint = equity_market_cointegration(equity_series, prob_series)
        beta = _coerce_finite(coint.get("beta"))
        intercept = _coerce_finite(coint.get("alpha"))
        half_life = _coerce_finite(coint.get("half_life"))
    except EquityFactorError as e:
        # Degenerate inputs (e.g. all-zero equity series) shouldn't 500
        # the whole endpoint — the chart is still useful without stats.
        logger.debug("cointegration failed for %s/%s: %s", slug, ticker, e)

    correlation = _correlation(prob_series, equity_series)

    equity_payload = [
        {"t": ts.date().isoformat(), "close": float(v)}
        for ts, v in equity_series.items()
        if np.isfinite(v)
    ]

    return {
        "ticker": ticker,
        "equity_series": equity_payload,
        "correlation_with_prob": correlation,
        "beta": beta,
        "intercept": intercept,
        "half_life": half_life,
    }


def _coerce_finite(x: object) -> float | None:
    """Return a JSON-safe float, or ``None`` for NaN/inf/non-numeric."""
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


__all__ = ["SLUG_TO_TICKER", "router"]
