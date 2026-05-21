"""Batched yfinance ticker fetcher.

A small concurrency-bounded helper that replaces serial yfinance fetches in
the factor scanner / peer scanner / portfolio tools. We control concurrency
ourselves (``ThreadPoolExecutor`` with explicit ``workers`` cap, ``threads=False``
on every ``yf.download`` call) so we don't blow Yahoo's hidden rate limits.

Why this module exists
----------------------
``pfm.sources.equity`` already exposes a per-ticker ``get_log_returns``. The
factor scanner currently loops over N tickers, paying the full HTTP RTT per
call. For a 50-name peer comparison that's ~50 * 600 ms = 30 s of wall time.
With ``fetch_tickers_batch(workers=8)`` the same job is roughly 8x faster
(IO-bound, no GIL contention because yfinance releases the GIL inside
``requests``).

Concurrency contract
--------------------
- ``workers`` caps in-flight ``yf.download`` calls. We use a semaphore
  *inside* the worker function so the "max-in-flight" count is bounded
  by the semaphore, not by executor scheduling artefacts.
- We pass ``threads=False`` to every ``yf.download`` call. yfinance's
  own threading would multiply our concurrency by another factor and
  trigger 429s.
- We add 50-200ms jitter *before* each download to spread launches out
  inside a burst.

Failure mode
------------
Per-ticker failures do NOT raise. They are logged at WARNING with the
ticker name and exception type, and the ticker maps to an empty
``DataFrame``. Callers should check ``df.empty`` before use.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Jitter bounds (milliseconds). Each worker sleeps for a random
# duration in this window BEFORE calling yf.download, which spreads
# bursty launches without materially affecting tail latency.
_JITTER_MS_MIN = 50
_JITTER_MS_MAX = 200


def _coerce_date(value: date | str) -> date:
    """Accept either ``datetime.date`` or an ISO ``YYYY-MM-DD`` string."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        # pandas handles a wider range of strings than datetime.date.fromisoformat
        return pd.Timestamp(value).date()
    raise TypeError(f"start/end must be date or str, got {type(value).__name__}")


def _fetch_one(
    ticker: str,
    *,
    start: date,
    end: date | None,
    interval: str,
    semaphore: threading.Semaphore,
    in_flight_counter: dict[str, int],
    counter_lock: threading.Lock,
) -> tuple[str, pd.DataFrame]:
    """Fetch a single ticker. Always returns ``(ticker, df)`` — never raises.

    On failure, ``df`` is an empty ``DataFrame`` and the failure is logged.
    The ``in_flight_counter`` and ``counter_lock`` are used for test-time
    introspection of the max-concurrency bound; they are cheap in production.
    """
    with semaphore:
        with counter_lock:
            in_flight_counter["current"] += 1
            in_flight_counter["max"] = max(in_flight_counter["max"], in_flight_counter["current"])

        try:
            # Jitter to spread out bursty launches
            jitter_s = random.uniform(_JITTER_MS_MIN, _JITTER_MS_MAX) / 1000.0
            time.sleep(jitter_s)

            kwargs: dict[str, Any] = {
                "start": start,
                "interval": interval,
                "auto_adjust": True,
                "progress": False,
                "threads": False,  # we control concurrency ourselves
            }
            if end is not None:
                kwargs["end"] = end

            df = yf.download(ticker, **kwargs)

            if df is None:
                logger.warning(
                    "yfinance_batch: %s returned None (treating as empty)",
                    ticker,
                )
                return ticker, pd.DataFrame()

            if not isinstance(df, pd.DataFrame):
                logger.warning(
                    "yfinance_batch: %s returned non-DataFrame %s",
                    ticker,
                    type(df).__name__,
                )
                return ticker, pd.DataFrame()

            # yfinance returns a MultiIndex when given a list. We always pass
            # a single ticker so flatten if it slipped through.
            if isinstance(df.columns, pd.MultiIndex):
                with contextlib.suppress(KeyError):
                    df = df.xs(ticker, axis=1, level=-1, drop_level=True)

            return ticker, df

        except Exception as exc:
            logger.warning(
                "yfinance_batch: fetch failed ticker=%s exc_type=%s detail=%s",
                ticker,
                type(exc).__name__,
                exc,
            )
            return ticker, pd.DataFrame()
        finally:
            with counter_lock:
                in_flight_counter["current"] -= 1


def fetch_tickers_batch(
    tickers: list[str],
    *,
    start: date | str,
    end: date | str | None = None,
    interval: str = "1d",
    workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """Fetch multiple yfinance tickers concurrently.

    Parameters
    ----------
    tickers : list[str]
        Tickers to fetch. Duplicates are de-duplicated while preserving order.
    start : date | str
        Inclusive start date (``YYYY-MM-DD`` or ``datetime.date``).
    end : date | str | None
        Exclusive end date. ``None`` means "as recent as available".
    interval : str
        yfinance interval (``"1d"``, ``"1h"``, ``"5m"``, …). Default ``"1d"``.
    workers : int
        Max in-flight yfinance HTTP calls. Default 8 — empirically Yahoo
        tolerates this without 429s.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{ticker: prices_df}``. Failures map to an empty ``DataFrame``.
        Always contains exactly one key per **unique** input ticker.
    """
    if not tickers:
        return {}

    # De-dup while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    start_d = _coerce_date(start)
    end_d: date | None = _coerce_date(end) if end is not None else None

    # workers must be >= 1; clamp absurd input
    workers = max(1, int(workers))

    semaphore = threading.Semaphore(workers)
    counter_lock = threading.Lock()
    in_flight_counter: dict[str, int] = {"current": 0, "max": 0}

    results: dict[str, pd.DataFrame] = {}

    # We launch min(workers * 2, len(unique)) executor threads — the semaphore
    # is what actually bounds concurrent yf.download calls. Using a slightly
    # larger pool lets jitter sleeps happen in parallel with downloads.
    pool_size = min(max(workers * 2, workers), max(1, len(unique)))

    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        future_to_ticker = {
            pool.submit(
                _fetch_one,
                ticker,
                start=start_d,
                end=end_d,
                interval=interval,
                semaphore=semaphore,
                in_flight_counter=in_flight_counter,
                counter_lock=counter_lock,
            ): ticker
            for ticker in unique
        }
        for future in as_completed(future_to_ticker):
            ticker, df = future.result()
            results[ticker] = df

    # Stash the observed max concurrency for tests. Not part of the public
    # contract — tests reach into ``_last_max_in_flight`` via the module.
    global _last_max_in_flight
    _last_max_in_flight = in_flight_counter["max"]

    return results


# Test introspection only (max observed in-flight yf.download calls on the
# most recent ``fetch_tickers_batch`` invocation). Not part of the public API.
_last_max_in_flight: int = 0


async def fetch_tickers_batch_async(
    tickers: list[str],
    *,
    start: date | str,
    end: date | str | None = None,
    interval: str = "1d",
    workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """Asyncio wrapper around :func:`fetch_tickers_batch`.

    Uses ``loop.run_in_executor(None, ...)`` so the sync threaded fetch
    runs on the default executor and the calling coroutine yields control
    until completion. Same kwargs and return contract as the sync version.
    """
    loop = asyncio.get_event_loop()

    def _runner() -> dict[str, pd.DataFrame]:
        return fetch_tickers_batch(
            tickers,
            start=start,
            end=end,
            interval=interval,
            workers=workers,
        )

    return await loop.run_in_executor(None, _runner)
