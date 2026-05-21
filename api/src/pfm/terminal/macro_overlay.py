"""Macro-overlay endpoint for the Terminal panel.

When a user opens a Polymarket *macro* market — Fed cuts, recession,
inflation, BTC price, oil shocks, geopolitics — the chart benefits from
overlaying the relevant macro indicator (DGS10 yields, SPY, GLD, DXY,
VIX, USO, BTC-USD, …). This module:

  * resolves a slug to one or two underlying tickers via a curated
    prefix table,
  * fetches the underlying series (FRED for rates / CPI, yfinance for
    ETFs and crypto),
  * fetches the Polymarket YES-token probability,
  * aligns both on the UTC daily calendar,
  * computes a Pearson correlation, OLS β, and best lag (in days) by
    cross-correlation in the window [-30, +30].

When more than one ticker maps to a slug we pick the *first* element of
the mapping list as the primary overlay (the one used for stats); the
secondary ticker is reported in ``additional_tickers`` so the frontend
can let the user toggle to it. Keeps the response shape stable while
preserving full information.

Routing note: this module owns its :class:`fastapi.APIRouter` so the
existing ``main.py`` is left untouched. To activate the endpoint::

    from pfm.terminal_macro_overlay import router as macro_overlay_router
    app.include_router(macro_overlay_router)
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from pfm.equity_factors import EquityFactorError, fetch_equity_history
from pfm.sources.fred import FredDataError, fetch_fred_series
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,
)

logger = logging.getLogger(__name__)


# --- mapping ----------------------------------------------------------------
# Each entry is a list of (ticker, source) tuples. ``source`` is one of
# ``"fred"`` or ``"yf"``. The first entry is treated as the *primary*
# overlay (the one stats are computed against); subsequent entries are
# returned to the frontend as ``additional_tickers`` for optional
# rendering. We list them in *priority order* — DGS10 ahead of SPY for
# Fed-cut markets because rates are the more direct read on Fed policy.
PREFIX_MAP: list[tuple[str, list[tuple[str, str]]]] = [
    # Order matters — longer / more specific prefixes first so e.g.
    # ``twelve_plus_fed_cuts`` matches before a hypothetical ``twelve_*``.
    ("twelve_plus_fed_cuts", [("DGS10", "fred"), ("SPY", "yf")]),
    ("no_fed_cuts_", [("DGS10", "fred"), ("SPY", "yf")]),
    ("fed_cuts_", [("DGS10", "fred"), ("SPY", "yf")]),
    ("inflation_above_", [("CPIAUCSL", "fred")]),
    ("k_cpi_", [("CPIAUCSL", "fred")]),
    ("us_recession_2026", [("SPY", "yf"), ("UUP", "yf")]),
    ("k_recession_2026", [("SPY", "yf"), ("UUP", "yf")]),
    ("bitcoin_", [("BTC-USD", "yf")]),
    ("btc_", [("BTC-USD", "yf")]),
    ("ethereum_", [("ETH-USD", "yf")]),
    ("eth_", [("ETH-USD", "yf")]),
    ("crude_", [("USO", "yf")]),
    ("oil_", [("USO", "yf")]),
    ("gold_", [("GLD", "yf")]),
    ("silver_", [("SLV", "yf")]),
    ("powell_out_", [("DGS10", "fred"), ("UUP", "yf")]),
    ("taiwan_", [("^VIX", "yf")]),
    ("china_", [("^VIX", "yf")]),
    ("iran_", [("^VIX", "yf")]),
    ("ipo_", [("SPY", "yf")]),
]

# Suffix mapping — slugs ending with ``_acquired`` always overlay the
# broad market, regardless of the company. Keep this list short.
SUFFIX_MAP: list[tuple[str, list[tuple[str, str]]]] = [
    ("_acquired", [("SPY", "yf")]),
]


def _resolve_overlay(slug: str) -> list[tuple[str, str]] | None:
    """Resolve a slug to its (ticker, source) overlay list, or ``None``."""
    for prefix, overlays in PREFIX_MAP:
        if slug.startswith(prefix):
            return overlays
    for suffix, overlays in SUFFIX_MAP:
        if slug.endswith(suffix):
            return overlays
    return None


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-macro-overlay"])


def get_polymarket_client() -> PolymarketClient:
    """Resolve the shared :class:`PolymarketClient` from app state.

    Imported lazily inside the function so this module doesn't pull
    ``pfm.main`` at import time (which would create a circular import).
    """
    from pfm.main import app  # local import to avoid circulars

    return app.state.poly


# --- statistics -------------------------------------------------------------


def _coerce_finite(x: object) -> float | None:
    """JSON-safe float coercion (``None`` for NaN / inf / non-numeric)."""
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _correlation(prob: pd.Series, macro: pd.Series) -> float | None:
    """Pearson correlation of aligned daily levels. ``None`` if degenerate."""
    df = pd.concat([prob.rename("p"), macro.rename("m")], axis=1).dropna()
    if len(df) < 5:
        return None
    if df["p"].std(ddof=0) == 0 or df["m"].std(ddof=0) == 0:
        return None
    rho = float(df["p"].corr(df["m"]))
    return rho if np.isfinite(rho) else None


def _beta(prob: pd.Series, macro: pd.Series) -> float | None:
    """OLS β of prob on macro: ``cov(p, m) / var(m)``."""
    df = pd.concat([prob.rename("p"), macro.rename("m")], axis=1).dropna()
    if len(df) < 5:
        return None
    var_m = float(df["m"].var(ddof=0))
    if not np.isfinite(var_m) or var_m == 0:
        return None
    cov_pm = float(((df["p"] - df["p"].mean()) * (df["m"] - df["m"].mean())).mean())
    if not np.isfinite(cov_pm):
        return None
    return cov_pm / var_m


def best_lag(
    prob: pd.Series,
    macro: pd.Series,
    *,
    max_lag: int = 30,
) -> tuple[int | None, float | None]:
    """Find the lag (in days) that maximises |corr(prob_t, macro_{t-lag})|.

    A *positive* lag means ``macro`` *leads* the prediction market — i.e.
    we shift the macro series forward by ``lag`` days before correlating,
    so today's prob aligns with macro from ``lag`` days ago. A *negative*
    lag means the prediction market leads the macro indicator.

    Args:
        prob: Polymarket probability series (UTC daily).
        macro: macro indicator series (UTC daily).
        max_lag: search window ``[-max_lag, +max_lag]`` in days.

    Returns:
        ``(lag, corr)`` of the maximal-|correlation| lag, or
        ``(None, None)`` if there's not enough overlap.
    """
    df = pd.concat([prob.rename("p"), macro.rename("m")], axis=1).dropna()
    if len(df) < max_lag + 5:
        return None, None

    best_l: int | None = None
    best_c: float | None = None
    for lag in range(-max_lag, max_lag + 1):
        shifted = df["m"].shift(lag)
        joined = pd.concat([df["p"], shifted], axis=1).dropna()
        if len(joined) < 5:
            continue
        if joined.iloc[:, 0].std(ddof=0) == 0 or joined.iloc[:, 1].std(ddof=0) == 0:
            continue
        c = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        if not np.isfinite(c):
            continue
        if best_c is None or abs(c) > abs(best_c):
            best_c = c
            best_l = lag

    return best_l, best_c


# --- fetch helpers ----------------------------------------------------------


def _fetch_macro(
    ticker: str,
    source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Dispatch to the right fetcher by source. Raises on data error."""
    if source == "fred":
        return fetch_fred_series(ticker, start=start, end=end)
    if source == "yf":
        return fetch_equity_history(ticker, start=start, end=end)
    raise ValueError(f"unknown macro source {source!r}")


@router.get("/macro-overlay/{slug}")
def get_macro_overlay(
    slug: Annotated[str, Path(min_length=1, max_length=120)],
    days: Annotated[int, Query(ge=10, le=3650)] = 180,
    *,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
) -> dict[str, Any]:
    """Return the macro-overlay payload for a Polymarket macro slug.

    Response shape (matched slug)::

        {
            "polymarket_slug": "fed_cuts_2026",
            "polymarket_series": [{"t": "2025-11-01", "p": 0.42}, ...],
            "macro_ticker": "DGS10",
            "macro_source": "fred",
            "macro_series": [{"t": "2025-11-01", "value": 4.18}, ...],
            "correlation": -0.31,
            "beta": -0.12,
            "lag_days": 5,
            "best_lag_corr": -0.44,
            "additional_tickers": [{"ticker": "SPY", "source": "yf"}]
        }

    Unmapped slug::

        {
            "polymarket_slug": "...",
            "macro_ticker": null,
            "message": "no macro overlay",
            ...
        }

    Upstream data errors raise 502; the endpoint never silently swallows
    a missing series — but the macro fetcher's failure on a *secondary*
    ticker only logs and is otherwise non-fatal.
    """
    overlays = _resolve_overlay(slug)
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days)

    # --- always fetch the polymarket leg first; the chart is useful
    # even when no macro mapping exists.
    try:
        prob_df = fetch_factor_history(poly, slug, start=start, end=end)
    except PolymarketError as e:
        # The Polymarket client raises with the canonical
        # ``no market found for slug=...`` message when Gamma can't resolve
        # a slug; that's a client problem (404), not an upstream outage
        # (502). Other PolymarketError messages stay 502 to preserve the
        # historic contract — see test_terminal_macro_overlay_extra.py.
        if "no market found for slug" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"market not found: {slug!r}") from e
        raise HTTPException(status_code=502, detail=f"polymarket error: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e

    if prob_df.empty:
        prob_series = pd.Series(dtype=float, name=slug)
        prob_payload: list[dict[str, Any]] = []
    else:
        prob_series = prob_df["price"].rename(slug)
        prob_series.index = pd.to_datetime(prob_series.index, utc=True).normalize()
        prob_payload = [
            {"t": ts.date().isoformat(), "p": float(v)}
            for ts, v in prob_series.items()
            if np.isfinite(v)
        ]

    # --- unmapped slug: 200 + null macro, with the polymarket series
    # still populated for the frontend.
    if overlays is None:
        return {
            "polymarket_slug": slug,
            "polymarket_series": prob_payload,
            "macro_ticker": None,
            "macro_source": None,
            "macro_series": [],
            "correlation": None,
            "beta": None,
            "lag_days": None,
            "best_lag_corr": None,
            "additional_tickers": [],
            "message": "no macro overlay",
        }

    primary_ticker, primary_source = overlays[0]
    additional = [{"ticker": t, "source": s} for t, s in overlays[1:]]

    # --- macro leg ----------------------------------------------------------
    try:
        macro_series = _fetch_macro(primary_ticker, primary_source, start, end)
    except (FredDataError, EquityFactorError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"macro fetch failed for {primary_ticker!r}: {e}",
        ) from e

    macro_series = macro_series.dropna()
    macro_series.index = pd.to_datetime(macro_series.index, utc=True).normalize()
    macro_payload = [
        {"t": ts.date().isoformat(), "value": float(v)}
        for ts, v in macro_series.items()
        if np.isfinite(v)
    ]

    # --- diagnostics --------------------------------------------------------
    correlation = _correlation(prob_series, macro_series)
    beta = _coerce_finite(_beta(prob_series, macro_series))
    lag_days, best_corr = best_lag(prob_series, macro_series)

    return {
        "polymarket_slug": slug,
        "polymarket_series": prob_payload,
        "macro_ticker": primary_ticker,
        "macro_source": primary_source,
        "macro_series": macro_payload,
        "correlation": _coerce_finite(correlation),
        "beta": beta,
        "lag_days": lag_days,
        "best_lag_corr": _coerce_finite(best_corr),
        "additional_tickers": additional,
    }


__all__ = [
    "PREFIX_MAP",
    "SUFFIX_MAP",
    "best_lag",
    "router",
]
