"""Stooq.com daily-prices fetcher (auth-free CSV).

Stooq exposes a free CSV endpoint that doesn't require an API key:

    https://stooq.com/q/d/l/?s={ticker}&d1={YYYYMMDD}&d2={YYYYMMDD}&i=d

Note that Stooq tickers diverge from US-exchange convention — for many US
names the symbol is suffixed with ``.us`` (e.g. ``aapl.us``). We pass the
ticker through unchanged but lowercase it; if the caller already ships
``"aapl.us"`` it stays that way. If a bare US ticker (``AAPL``) misses,
the equity adapter falls through to the next source.

Used as the second fallback after Tiingo. No auth, no rate-limit budget
to track, but it can be patchy outside US large-caps.
"""

from __future__ import annotations

import logging
import time
from io import StringIO

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

STOOQ_BASE = "https://stooq.com/q/d/l/"

# Single 429-retry with 1.5 s backoff. Stooq is auth-free and we don't
# observe explicit rate-limit headers, but it occasionally serves a
# transient 429 / 5xx when crawled aggressively — one retry covers the
# common "two parallel requests collided" case.
_RETRY_BACKOFF_S: float = 1.5


class StooqError(RuntimeError):
    """Raised on Stooq fetch error."""


def _normalize_symbol(ticker: str) -> str:
    """Lowercase + append ``.us`` if no exchange suffix is supplied.

    Most US equities resolve as ``aapl.us``, ``nvda.us``, etc. on Stooq.
    Indices and FX use other suffixes (``^spx``, ``eurusd``); callers
    that already include a suffix ('.' or '^') keep it intact.
    """
    t = ticker.strip().lower()
    if not t:
        return t
    if "." in t or "^" in t:
        return t
    return f"{t}.us"


def fetch_daily_prices(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    client: httpx.Client | None = None,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch daily OHLCV from Stooq for ``ticker``.

    Args:
        ticker: e.g. ``"AAPL"`` (we'll lowercase and add ``.us``) or
            ``"aapl.us"`` (passed through).
        start, end: inclusive UTC pd.Timestamp bounds.
        client: optional ``httpx.Client``.
        timeout: request timeout in seconds.

    Returns:
        DataFrame indexed by UTC-normalised date with columns
        ``[open, high, low, close, volume]``.

    Raises:
        StooqError: on HTTP error or empty/malformed CSV.
    """
    own_client = client is None
    cli = client or httpx.Client(timeout=timeout)
    params = {
        "s": _normalize_symbol(ticker),
        "d1": start.strftime("%Y%m%d"),
        "d2": end.strftime("%Y%m%d"),
        "i": "d",
    }
    try:
        try:
            resp = cli.get(STOOQ_BASE, params=params)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                logger.warning(
                    "stooq %d for %s — retrying in %.1fs",
                    resp.status_code,
                    ticker,
                    _RETRY_BACKOFF_S,
                )
                time.sleep(_RETRY_BACKOFF_S)
                resp = cli.get(STOOQ_BASE, params=params)
        except httpx.HTTPError as e:
            raise StooqError(f"Stooq transport error for {ticker!r}: {e}") from e
        if resp.status_code != 200:
            raise StooqError(f"Stooq HTTP {resp.status_code} for {ticker!r}: {resp.text[:200]}")
        text = resp.text
    finally:
        if own_client:
            cli.close()

    # Stooq sends a literal "No data" body (200 OK) for unknown tickers.
    stripped = text.strip()
    if not stripped or stripped.lower().startswith("no data"):
        raise StooqError(f"Stooq returned no data for {ticker!r}")

    try:
        df = pd.read_csv(StringIO(text))
    except Exception as e:  # pragma: no cover - pd.errors variant
        raise StooqError(f"Stooq CSV parse error for {ticker!r}: {e}") from e

    if df.empty:
        raise StooqError(f"Stooq CSV empty for {ticker!r}")

    # Stooq columns: Date,Open,High,Low,Close,Volume.
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        raise StooqError(f"Stooq CSV missing required columns for {ticker!r}: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df = df.set_index("date").sort_index()

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep]


__all__ = ["STOOQ_BASE", "StooqError", "fetch_daily_prices"]
