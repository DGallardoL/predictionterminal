"""Multi-source equity adapter for daily returns.

Public API
----------
``get_log_returns(ticker, start, end, return_type)`` — unchanged signature
from the original yfinance-only implementation. Internally now cascades:

    1. yfinance (Yahoo data, current default)
    2. Tiingo (free tier, requires ``TIINGO_API_KEY`` env var)
    3. Stooq (free, no auth)

If all three fail we raise :class:`EquityDataError` with a per-source
breakdown of what went wrong.

Delisted handling
-----------------
If yfinance returns an empty/all-NaN frame we probe ``yf.Ticker.info``
for ``regularMarketPrice``. A ``None`` here is yfinance's strongest
"this ticker is delisted" signal, so we raise :class:`EquityDelistedError`
*without* falling through to Tiingo/Stooq. We also persist the ticker to
``/tmp/pfm_delisted_tickers.json`` so the next call short-circuits.

Alignment policy: dates are normalised to UTC midnight to match the
Polymarket convention (see ADR-0006).

Returns
~~~~~~~
- ``"log"``    — ``r_t = log(P_t / P_{t-1})``  (time-additive, symmetric)
- ``"simple"`` — ``r_t = P_t / P_{t-1} - 1``    (what brokers report)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yfinance as yf

from pfm.cache_utils import get_cache
from pfm.sources import stooq as stooq_src
from pfm.sources import tiingo as tiingo_src

logger = logging.getLogger(__name__)

ReturnType = Literal["log", "simple"]

DELISTED_REGISTRY_PATH = Path(
    os.environ.get("PFM_DELISTED_REGISTRY", "/tmp/pfm_delisted_tickers.json")
)
_REGISTRY_LOCK = threading.Lock()

# 1-hour TTL: daily prices are stable intra-day so re-running the same fit
# repeatedly shouldn't hammer upstream sources.
_EQUITY_CACHE = get_cache("equity", ttl=3600)


class EquityDataError(RuntimeError):
    """Raised when *every* configured equity source fails for a ticker."""


class EquityDelistedError(EquityDataError):
    """Raised when a ticker is detected as delisted/suspended.

    Carries the ticker symbol so callers can surface it in error messages.
    """

    def __init__(self, ticker: str, detail: str = "") -> None:
        self.ticker = ticker
        msg = f"ticker {ticker!r} is delisted/suspended"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


# --- delisted registry ------------------------------------------------------


def _load_delisted_registry() -> set[str]:
    """Read the on-disk delisted-tickers cache. Returns empty set on miss."""
    with _REGISTRY_LOCK:
        try:
            with DELISTED_REGISTRY_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return set()
    if isinstance(data, list):
        return {str(t).upper() for t in data}
    if isinstance(data, dict) and "tickers" in data:
        return {str(t).upper() for t in data["tickers"]}
    return set()


def _save_delisted_registry(registry: set[str]) -> None:
    """Write the delisted-tickers cache atomically."""
    with _REGISTRY_LOCK:
        try:
            DELISTED_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = DELISTED_REGISTRY_PATH.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(sorted(registry), fh)
            tmp.replace(DELISTED_REGISTRY_PATH)
        except OSError as e:  # pragma: no cover - filesystem failure
            logger.warning("Failed to persist delisted registry: %s", e)


def is_delisted(ticker: str) -> bool:
    """Cheap check against the on-disk registry."""
    return ticker.upper() in _load_delisted_registry()


def mark_delisted(ticker: str) -> None:
    """Append ``ticker`` to the delisted registry (idempotent)."""
    reg = _load_delisted_registry()
    reg.add(ticker.upper())
    _save_delisted_registry(reg)


def list_delisted() -> list[str]:
    """Return the sorted list of delisted tickers from the registry."""
    return sorted(_load_delisted_registry())


# --- yfinance probe ---------------------------------------------------------


def _yfinance_closes(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Fetch adjusted closes via yfinance. Raises ``EquityDataError`` on miss."""
    start_d = start.date()
    end_d = end.date()
    end_excl = (end + pd.Timedelta(days=1)).date()

    df = yf.download(
        ticker,
        start=start_d,
        end=end_excl,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        raise EquityDataError(f"yfinance returned no data for {ticker!r} in [{start_d}, {end_d}]")

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        except KeyError as e:
            raise EquityDataError(f"yfinance multi-index missing {ticker!r}: {e}") from e

    if "Close" not in df.columns:
        raise EquityDataError(f"yfinance response missing Close column for {ticker!r}")

    closes = df["Close"].dropna()
    if len(closes) < 2:
        raise EquityDataError(f"too few closes for {ticker!r} to compute returns")
    return closes


def _check_delisted_via_yf_info(ticker: str) -> bool:
    """Probe ``yf.Ticker.info`` for delisting signal. Best-effort, never raises."""
    try:
        info: Any = yf.Ticker(ticker).info
    except Exception as e:
        logger.debug("yf.Ticker.info failed for %s: %s", ticker, e)
        return False
    if not isinstance(info, dict):
        return False
    # yfinance signals "delisted" by leaving regularMarketPrice as None.
    return "regularMarketPrice" in info and info.get("regularMarketPrice") is None


# --- source attempts --------------------------------------------------------


def _try_yfinance(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """First-line source. Triggers delisted detection on empty result."""
    try:
        return _yfinance_closes(ticker, start, end)
    except EquityDataError as err:
        # Empty / NaN-only frame triggers a delisted probe. If confirmed,
        # we raise EquityDelistedError immediately (no fallback) and persist
        # the ticker. If unconfirmed, re-raise so the caller falls through.
        if _check_delisted_via_yf_info(ticker):
            mark_delisted(ticker)
            raise EquityDelistedError(ticker, "yfinance reports regularMarketPrice=None") from err
        raise


def _try_tiingo(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Second-line source. Silently no-ops if ``TIINGO_API_KEY`` unset."""
    api_key = os.environ.get("TIINGO_API_KEY")
    if not api_key:
        raise tiingo_src.TiingoError("TIINGO_API_KEY not configured")
    df = tiingo_src.fetch_daily_prices(ticker, start, end, api_key=api_key)
    # Prefer adjClose for return calculation when present.
    col = "adjClose" if "adjClose" in df.columns else "close"
    closes = df[col].dropna()
    if len(closes) < 2:
        raise tiingo_src.TiingoError(f"Tiingo too few closes for {ticker!r}")
    return closes


def _try_stooq(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Third-line source. No auth, but coverage is patchy outside US large-caps."""
    df = stooq_src.fetch_daily_prices(ticker, start, end)
    closes = df["close"].dropna()
    if len(closes) < 2:
        raise stooq_src.StooqError(f"Stooq too few closes for {ticker!r}")
    return closes


# --- public API -------------------------------------------------------------


def _closes_to_returns(closes: pd.Series, return_type: ReturnType) -> pd.Series:
    """Convert price series to log or simple returns aligned to UTC dates."""
    ratio = closes / closes.shift(1)
    if return_type == "log":
        ret = np.log(ratio).dropna()
    elif return_type == "simple":
        ret = (ratio - 1.0).dropna()
    else:
        raise ValueError(f"unknown return_type: {return_type!r}")
    ret.index = pd.to_datetime(ret.index, utc=True).normalize()
    ret.name = "r"
    return ret


def get_log_returns(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    return_type: ReturnType = "log",
) -> pd.Series:
    """Download adjusted closes and return daily returns (cascaded fallback).

    Args:
        ticker: equity symbol (e.g. ``"NVDA"``).
        start: inclusive lower bound (UTC pd.Timestamp).
        end: inclusive upper bound (UTC pd.Timestamp).
        return_type: ``"log"`` or ``"simple"``.

    Returns:
        Series of returns indexed by UTC-normalised dates, named ``r``.

    Raises:
        EquityDelistedError: ticker is in the delisted registry, or
            yfinance reports ``regularMarketPrice`` is ``None``.
        EquityDataError: every source failed; the message lists each
            source-specific error.
    """
    if is_delisted(ticker):
        raise EquityDelistedError(ticker, "found in on-disk delisted registry")

    cache_key = (ticker.upper(), str(start), str(end), return_type)
    cached = _EQUITY_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    errors: dict[str, str] = {}
    closes: pd.Series | None = None

    # 1. yfinance
    try:
        closes = _try_yfinance(ticker, start, end)
    except EquityDelistedError:
        raise
    except Exception as e:
        errors["yfinance"] = f"{type(e).__name__}: {e}"

    # 2. Tiingo
    if closes is None:
        try:
            closes = _try_tiingo(ticker, start, end)
        except Exception as e:
            errors["tiingo"] = f"{type(e).__name__}: {e}"

    # 3. Stooq
    if closes is None:
        try:
            closes = _try_stooq(ticker, start, end)
        except Exception as e:
            errors["stooq"] = f"{type(e).__name__}: {e}"

    if closes is None:
        detail = "; ".join(f"{src}={msg}" for src, msg in errors.items()) or "no sources tried"
        raise EquityDataError(f"all equity sources failed for {ticker!r}: {detail}")

    ret = _closes_to_returns(closes, return_type)
    _EQUITY_CACHE.set(cache_key, ret.copy())
    return ret


__all__ = [
    "DELISTED_REGISTRY_PATH",
    "EquityDataError",
    "EquityDelistedError",
    "ReturnType",
    "get_log_returns",
    "is_delisted",
    "list_delisted",
    "mark_delisted",
]
