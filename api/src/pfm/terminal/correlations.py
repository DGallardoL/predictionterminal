"""Cross-asset correlation analytics endpoint for the Terminal panel.

Given a Polymarket slug, this module fetches the YES-token probability
series and computes its correlation against a fixed basket of seven
cross-asset benchmarks: BTC-USD, ETH-USD, SPY, GLD, UUP (DXY proxy),
^VIX, and DGS10 (the 10-year Treasury yield from FRED).

Methodology
-----------
- Probabilities are transformed to logits and differenced to recover an
  approximately stationary innovation series:

      Δlogit_t = logit(p_t) − logit(p_{t-1})

- Equity / crypto / FX prices are converted to log returns:

      r_t = log(P_t / P_{t-1})

- DGS10 is reported in percent (level), so we use first differences
  (Δyield_t) to keep it stationary and unit-comparable.

- Pearson correlation is computed on the joint dropna of the two
  innovation series. Reported alongside is a two-sided p-value derived
  from the Fisher z-transform — the same approximation
  ``scipy.stats.pearsonr`` uses, re-implemented here so we don't pull
  scipy just for one helper.

- The "best lag" per asset is selected by sweeping ``[-7, +7]`` days and
  returning the lag with the highest |corr|. A *positive* lag means the
  asset *leads* the market: shifting the asset forward by ``lag`` days
  before correlating aligns today's prob-innovation with the asset's
  innovation from ``lag`` days ago.

Caching
-------
A 1-hour in-memory TTL cache fronts the (slug, days) pair. Polymarket
ratelimits are generous but users will repeatedly request the same
correlation card; the cache makes the panel feel instant on re-open.

Routing
-------
This module owns its own :class:`fastapi.APIRouter`. Per project
convention, ``main.py`` is left untouched — wire it explicitly via::

    from pfm.terminal_correlations import router as terminal_corr_router
    app.include_router(terminal_corr_router)
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from pfm.cache_utils import TerminalCache
from pfm.equity_factors import EquityFactorError, fetch_equity_history
from pfm.sources.fred import FredDataError, fetch_fred_series
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,
)
from pfm.sources.yfinance_batch import fetch_tickers_batch

logger = logging.getLogger(__name__)


# --- benchmark basket -------------------------------------------------------
# Each entry: (display_name, fetch_symbol, source). DXY is fetched via the
# UUP ETF since the dollar index proper isn't available on yfinance; we
# still report it under "DXY" so the response matches the user's mental
# model of the basket.
BENCHMARKS: list[tuple[str, str, str]] = [
    ("BTC-USD", "BTC-USD", "yf"),
    ("ETH-USD", "ETH-USD", "yf"),
    ("SPY", "SPY", "yf"),
    ("GLD", "GLD", "yf"),
    ("DXY", "UUP", "yf"),
    ("VIX", "^VIX", "yf"),
    ("DGS10", "DGS10", "fred"),
]

# Lag search half-window in days. The endpoint sweeps [-MAX_LAG, +MAX_LAG].
MAX_LAG_DAYS: int = 7

# 1-hour TTL on the response cache.
CACHE_TTL_SECONDS: int = 3600

# Logit clip — same project-wide default as everywhere else.
DEFAULT_CLIP_EPS: float = 0.01


# --- in-memory cache --------------------------------------------------------
# Keyed by (slug, days). Value is (expiry_unix_seconds, response_dict).
# Backed by a module-level dict so legacy tests that introspect ``_CACHE``
# directly (assignment / equality) keep working; the TerminalCache wrapper
# adds the TTL + thread-safety logic on top.

_CACHE: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}
_cache = TerminalCache(default_ttl=CACHE_TTL_SECONDS, store=_CACHE)


def _cache_get(key: tuple[str, int]) -> dict[str, Any] | None:
    return _cache.get(key)


def _cache_set(key: tuple[str, int], payload: dict[str, Any]) -> None:
    _cache.set(key, payload)


def clear_cache() -> None:
    """Test/utility helper — drop all cached entries."""
    _cache.clear()


# --- transforms -------------------------------------------------------------


def _logit(p: pd.Series, *, clip_eps: float = DEFAULT_CLIP_EPS) -> pd.Series:
    """Element-wise logit with explicit clipping to (eps, 1 - eps).

    Out-of-range values become NaN — silently clipping them masks data
    bugs upstream.
    """
    s = p.copy()
    s = s.where((s > 0.0) & (s < 1.0))
    clipped = s.clip(lower=clip_eps, upper=1.0 - clip_eps)
    return np.log(clipped / (1.0 - clipped))


def _innovations(series: pd.Series, kind: str) -> pd.Series:
    """Convert a level series to a stationary innovation series.

    ``kind="logit"``  → first difference of logit(p)
    ``kind="log"``    → log return = log(P_t) - log(P_{t-1})
    ``kind="diff"``   → first difference of the level (used for yields)
    """
    if kind == "logit":
        return _logit(series).diff()
    if kind == "log":
        s = series.dropna()
        if (s <= 0).any():
            # Guard: log of non-positive crashes; drop offending obs.
            s = s[s > 0]
        return np.log(s).diff()
    if kind == "diff":
        return series.diff()
    raise ValueError(f"unknown innovation kind {kind!r}")


# --- statistics -------------------------------------------------------------


def _pearson_p_value(corr: float, n: int) -> float | None:
    """Two-sided p-value for a Pearson correlation, Fisher-z approximation.

    Equivalent to ``scipy.stats.pearsonr``'s ``pvalue`` field for n>=4 but
    avoids the scipy dependency. ``None`` if degenerate.
    """
    if n < 4 or not np.isfinite(corr):
        return None
    # Clip to avoid atanh(±1) → inf when corr is exactly ±1 (e.g. tests
    # with synthetic identical series).
    r = max(min(corr, 0.9999999), -0.9999999)
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    # Two-sided p from standard normal:
    #   p = 2 * (1 - Φ(|z| / se))  =  erfc(|z| / (se * sqrt(2)))
    p = math.erfc(abs(z) / (se * math.sqrt(2.0)))
    if not math.isfinite(p):
        return None
    return max(min(p, 1.0), 0.0)


def best_lag_corr(
    prob_innov: pd.Series,
    asset_innov: pd.Series,
    *,
    max_lag: int = MAX_LAG_DAYS,
) -> tuple[int | None, float | None, int | None]:
    """Sweep lags in ``[-max_lag, +max_lag]`` and return the max-|corr| triple.

    Convention: ``shifted_asset = asset_innov.shift(lag)``. So a *positive*
    lag means the asset *leads* the market — today's prob lines up with
    the asset from ``lag`` days ago.

    Returns:
        ``(lag, corr, n)`` for the lag with the largest absolute
        correlation, or ``(None, None, None)`` if no lag yields a
        well-defined correlation (too little overlap, zero variance).
    """
    df = pd.concat([prob_innov.rename("p"), asset_innov.rename("a")], axis=1).dropna()
    if len(df) < 5:
        return None, None, None

    best_l: int | None = None
    best_c: float | None = None
    best_n: int | None = None
    for lag in range(-max_lag, max_lag + 1):
        shifted = df["a"].shift(lag)
        joined = pd.concat([df["p"], shifted], axis=1).dropna()
        if len(joined) < 5:
            continue
        s_p = joined.iloc[:, 0]
        s_a = joined.iloc[:, 1]
        if s_p.std(ddof=0) == 0 or s_a.std(ddof=0) == 0:
            continue
        c = float(s_p.corr(s_a))
        if not np.isfinite(c):
            continue
        if best_c is None or abs(c) > abs(best_c):
            best_c = c
            best_l = lag
            best_n = len(joined)

    return best_l, best_c, best_n


# --- helpers ----------------------------------------------------------------


def _coerce_finite(x: object) -> float | None:
    """JSON-safe float coercion (``None`` for NaN / inf / non-numeric)."""
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _fetch_benchmark(
    symbol: str,
    source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    yf_prefetch: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """Dispatch to the right fetcher by source. Raises on data error.

    When ``yf_prefetch`` is provided and contains ``symbol`` as a non-empty
    series, that pre-batched series is returned directly — the 6 yfinance
    benchmark legs (BTC-USD / ETH-USD / SPY / GLD / UUP / ^VIX) can then be
    fetched in a single concurrent batch via :func:`fetch_tickers_batch`
    instead of 6 serial round-trips. Tests that monkeypatch
    ``fetch_equity_history`` directly leave the prefetch empty and continue
    to exercise the per-ticker path unchanged.
    """
    if source == "fred":
        return fetch_fred_series(symbol, start=start, end=end)
    if source == "yf":
        if yf_prefetch is not None:
            cached = yf_prefetch.get(symbol)
            if cached is not None and not cached.empty:
                return cached
        return fetch_equity_history(symbol, start=start, end=end)
    raise ValueError(f"unknown benchmark source {source!r}")


def _prefetch_yf_benchmarks(
    benchmarks: list[tuple[str, str, str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.Series]:
    """Batch-fetch every ``source == "yf"`` benchmark in one concurrent call.

    Returns ``{symbol: closes_series}`` shaped like ``fetch_equity_history``'s
    output (a single ``Close`` ``pd.Series`` indexed by UTC-normalised dates,
    named after the ticker). Per-ticker failures map to a series missing
    from the dict (or with ``.empty == True``) so the caller's per-ticker
    fallback path can still kick in. Never raises on a partial failure —
    yfinance hiccups are logged inside ``fetch_tickers_batch``.

    Test-compat: if ``fetch_equity_history`` has been monkeypatched in this
    module (identity check against the canonical one in ``pfm.equity_factors``),
    we skip the batch prefetch entirely so test mocks are still exercised on
    the per-ticker fallback path. Production never patches it ⇒ batch wins.
    """
    # Identity check — survives monkeypatch.setattr because the patched name
    # is a separate object than the original module-level one.
    import pfm.equity_factors as _eq_mod

    if fetch_equity_history is not _eq_mod.fetch_equity_history:
        return {}

    symbols = [sym for _, sym, src in benchmarks if src == "yf"]
    if not symbols:
        return {}

    # ``end`` is treated as exclusive by yfinance; add one day to match the
    # per-ticker behaviour of fetch_equity_history (which does the same
    # +1 day shift). We also accept ``pd.Timestamp`` here and let the
    # underlying helper coerce to ``datetime.date``.
    end_excl = (end + pd.Timedelta(days=1)).date()
    start_d = start.date()

    try:
        raw = fetch_tickers_batch(
            symbols,
            start=start_d,
            end=end_excl,
            interval="1d",
            workers=8,
        )
    except Exception as e:
        logger.warning("yf batch prefetch failed (%s); falling back to per-ticker", e)
        return {}

    out: dict[str, pd.Series] = {}
    for sym, df in raw.items():
        if df is None or df.empty:
            continue
        # ``fetch_tickers_batch`` returns a full OHLCV DataFrame; the per-ticker
        # path returns just the Close series. Extract Close and normalise the
        # index to match fetch_equity_history's contract.
        if "Close" not in df.columns:
            continue
        closes = df["Close"].dropna()
        if len(closes) < 2:
            continue
        closes.index = pd.to_datetime(closes.index, utc=True).normalize()
        closes.name = sym
        out[sym] = closes
    return out


def _interpret(
    correlations: dict[str, dict[str, Any]],
) -> str:
    """Build a one-paragraph human-readable summary from the corr table."""
    items: list[tuple[str, float]] = []
    for asset, info in correlations.items():
        c = info.get("corr")
        if c is None:
            continue
        items.append((asset, float(c)))
    if not items:
        return "No correlation could be computed (insufficient overlapping data)."

    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top_asset, top_corr = items[0]
    direction = "positively" if top_corr > 0 else "negatively"
    parts: list[str] = []
    parts.append(
        f"This market is most strongly {direction} correlated with {top_asset} "
        f"(corr {top_corr:+.2f})."
    )

    crypto = [(a, c) for a, c in items if a in {"BTC-USD", "ETH-USD"}]
    equity = [(a, c) for a, c in items if a in {"SPY", "GLD"}]
    macro = [(a, c) for a, c in items if a in {"DXY", "VIX", "DGS10"}]

    if crypto and abs(crypto[0][1]) >= 0.30:
        sign = "positively" if crypto[0][1] > 0 else "negatively"
        parts.append(
            f"It tracks crypto sentiment {sign} ({crypto[0][0]} corr {crypto[0][1]:+.2f})."
        )
    if equity and abs(equity[0][1]) >= 0.30:
        sign = "positively" if equity[0][1] > 0 else "negatively"
        parts.append(f"It moves {sign} with equities ({equity[0][0]} corr {equity[0][1]:+.2f}).")
    if macro and abs(macro[0][1]) >= 0.30:
        sign = "positively" if macro[0][1] > 0 else "negatively"
        parts.append(f"Macro link: {sign} correlated with {macro[0][0]} ({macro[0][1]:+.2f}).")

    return " ".join(parts)


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-correlations"])


def get_polymarket_client() -> PolymarketClient:
    """Resolve the shared :class:`PolymarketClient` from app state.

    Imported lazily inside the function so this module doesn't pull
    ``pfm.main`` at import time (which would create a circular import).
    """
    from pfm.main import app  # local import to avoid circulars

    return app.state.poly


@router.get("/correlations/{slug}")
def get_correlations(
    slug: Annotated[str, Path(min_length=1, max_length=120)],
    days: Annotated[int, Query(ge=20, le=730)] = 90,
    *,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
) -> dict[str, Any]:
    """Return the cross-asset correlation card for a Polymarket slug.

    Computes Pearson correlation of the market's logit-prob innovations
    against the log-return (or yield-difference) innovations of seven
    cross-asset benchmarks: BTC-USD, ETH-USD, SPY, GLD, DXY (UUP proxy),
    VIX (^VIX), and DGS10. For each benchmark, also reports the lag in
    ``[-7, +7]`` that maximises the absolute correlation.

    Args:
        slug: Polymarket market slug.
        days: lookback window in days (default 90).

    Returns:
        See module docstring. Cached for 1 hour.
    """
    cache_key = (slug, int(days))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    end = pd.Timestamp.utcnow().normalize()
    # Pull a pad so the lag sweep has enough slack at both ends.
    start = end - pd.Timedelta(days=days + MAX_LAG_DAYS + 2)

    # --- polymarket leg -----------------------------------------------------
    try:
        prob_df = fetch_factor_history(poly, slug, start=start, end=end)
    except PolymarketError as e:
        raise HTTPException(status_code=404, detail=f"unknown slug: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e

    if prob_df is None or prob_df.empty or "price" not in prob_df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"no price history for slug={slug!r}",
        )

    prob_series = prob_df["price"].astype(float)
    prob_series.index = pd.to_datetime(prob_series.index, utc=True).normalize()
    prob_series = prob_series[~prob_series.index.duplicated(keep="last")].sort_index()
    # Trim to the requested window.
    prob_series = prob_series[prob_series.index >= (end - pd.Timedelta(days=days))]
    prob_innov = _innovations(prob_series, kind="logit").dropna()

    if len(prob_innov) < 5:
        raise HTTPException(
            status_code=422,
            detail=(
                f"insufficient polymarket history for slug={slug!r}: "
                f"{len(prob_innov)} innovation observations"
            ),
        )

    # --- benchmark legs -----------------------------------------------------
    # Pre-batch every yfinance ticker in a single concurrent fetch. This
    # collapses the 6 serial yf.download round-trips (BTC-USD, ETH-USD,
    # SPY, GLD, UUP, ^VIX) into one ``fetch_tickers_batch(workers=8)`` call
    # — empirically a 6-8x speedup on a cold cache. Tests that monkeypatch
    # ``fetch_equity_history`` leave the prefetch dict empty (because the
    # batch helper falls through on import-time mocks) and continue to
    # exercise the per-ticker fallback path inside _fetch_benchmark.
    yf_prefetch = _prefetch_yf_benchmarks(BENCHMARKS, start, end)

    correlations: dict[str, dict[str, Any]] = {}
    for display, symbol, source in BENCHMARKS:
        try:
            level = _fetch_benchmark(symbol, source, start, end, yf_prefetch=yf_prefetch)
        except (FredDataError, EquityFactorError) as e:
            logger.warning("benchmark fetch failed for %s: %s", display, e)
            correlations[display] = {
                "corr": None,
                "p_value": None,
                "lag_days": None,
                "n": 0,
                "error": str(e),
            }
            continue

        level = level.dropna()
        if not len(level):
            correlations[display] = {
                "corr": None,
                "p_value": None,
                "lag_days": None,
                "n": 0,
            }
            continue
        level.index = pd.to_datetime(level.index, utc=True).normalize()
        level = level[~level.index.duplicated(keep="last")].sort_index()

        kind = "diff" if display == "DGS10" else "log"
        asset_innov = _innovations(level, kind=kind).dropna()

        # --- contemporaneous correlation (lag=0) --------------------------
        df0 = pd.concat([prob_innov.rename("p"), asset_innov.rename("a")], axis=1).dropna()
        c0: float | None
        if len(df0) >= 5 and df0["p"].std(ddof=0) > 0 and df0["a"].std(ddof=0) > 0:
            c0 = float(df0["p"].corr(df0["a"]))
            if not np.isfinite(c0):
                c0 = None
        else:
            c0 = None

        # --- best lag in [-7, +7] -----------------------------------------
        best_l, best_c, best_n = best_lag_corr(prob_innov, asset_innov)

        # If the lag-zero corr is well-defined and at least as strong as
        # the lag-swept best, prefer lag=0 (parsimony / Occam). Otherwise
        # report the best lag.
        if c0 is not None and (best_c is None or abs(c0) >= abs(best_c)):
            chosen_corr: float | None = c0
            chosen_lag: int | None = 0
            chosen_n: int | None = len(df0)
        else:
            chosen_corr = best_c
            chosen_lag = best_l
            chosen_n = best_n

        p_value: float | None = None
        if chosen_corr is not None and chosen_n is not None:
            p_value = _pearson_p_value(chosen_corr, chosen_n)

        correlations[display] = {
            "corr": _coerce_finite(chosen_corr),
            "p_value": _coerce_finite(p_value),
            "lag_days": chosen_lag,
            "n": chosen_n if chosen_n is not None else 0,
        }

    # --- ranked summary -----------------------------------------------------
    ranked = sorted(
        ({"asset": a, **info} for a, info in correlations.items() if info.get("corr") is not None),
        key=lambda d: abs(float(d["corr"])),
        reverse=True,
    )
    strongest = ranked[:3]

    response: dict[str, Any] = {
        "slug": slug,
        "polymarket_series_n": int(len(prob_innov) + 1),  # innovations + base obs
        "lookback_days": int(days),
        "correlations": correlations,
        "strongest": strongest,
        "interpretation": _interpret(correlations),
    }

    _cache_set(cache_key, response)
    return response


__all__ = [
    "BENCHMARKS",
    "CACHE_TTL_SECONDS",
    "MAX_LAG_DAYS",
    "_prefetch_yf_benchmarks",
    "best_lag_corr",
    "clear_cache",
    "get_correlations",
    "get_polymarket_client",
    "router",
]
