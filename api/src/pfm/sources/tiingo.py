"""Tiingo daily-prices fetcher.

Tiingo offers a free tier (500 req/hour) with EOD US-equity data and a
clean JSON API. Used as the first fallback when yfinance fails.

Auth: ``Authorization: Token {api_key}`` header. Without an API key the
fallback is silently skipped by ``pfm.sources.equity``.

Endpoint:
    GET https://api.tiingo.com/tiingo/daily/{ticker}/prices
        ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD

Response (JSON list):
    [{"date": "...", "open": .., "high": .., "low": .., "close": ..,
      "adjClose": .., "volume": ..}, ...]
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"

# Single 429-retry with 1.5 s backoff. Tiingo's free tier (500 req/h) is
# generous but the limiter still trips on bursts; one retry covers the
# 1-second bucket-refill window without amplifying real load.
_RETRY_BACKOFF_S: float = 1.5


class TiingoError(RuntimeError):
    """Raised on Tiingo fetch error."""


def fetch_daily_prices(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch daily OHLCV + adjClose for ``ticker`` from Tiingo.

    Args:
        ticker: US-equity ticker (e.g. ``"NVDA"``).
        start, end: inclusive UTC pd.Timestamp bounds.
        api_key: Tiingo API token. Required.
        client: optional ``httpx.Client`` (for connection pooling / mocking).
        timeout: request timeout in seconds.

    Returns:
        DataFrame indexed by UTC-normalised date with columns
        ``[open, high, low, close, adjClose, volume]``.

    Raises:
        TiingoError: on HTTP error, missing/invalid api_key, or empty payload.
    """
    if not api_key:
        raise TiingoError("Tiingo api_key is required")

    own_client = client is None
    cli = client or httpx.Client(timeout=timeout)
    url = f"{TIINGO_BASE}/{ticker.upper()}/prices"
    params = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    try:
        try:
            resp = cli.get(url, params=params, headers=headers)
            if resp.status_code == 429:
                logger.warning("tiingo 429 for %s — retrying in %.1fs", ticker, _RETRY_BACKOFF_S)
                time.sleep(_RETRY_BACKOFF_S)
                resp = cli.get(url, params=params, headers=headers)
        except httpx.HTTPError as e:
            raise TiingoError(f"Tiingo transport error for {ticker!r}: {e}") from e
        if resp.status_code != 200:
            raise TiingoError(f"Tiingo HTTP {resp.status_code} for {ticker!r}: {resp.text[:200]}")
        try:
            payload: list[dict[str, Any]] = resp.json()
        except ValueError as e:
            raise TiingoError(f"Tiingo non-JSON response for {ticker!r}") from e
    finally:
        if own_client:
            cli.close()

    if not payload:
        raise TiingoError(f"Tiingo returned empty payload for {ticker!r}")

    df = pd.DataFrame(payload)
    if "date" not in df.columns:
        raise TiingoError(f"Tiingo response missing 'date' for {ticker!r}")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df = df.set_index("date").sort_index()

    expected = ["open", "high", "low", "close", "adjClose", "volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise TiingoError(f"Tiingo response missing columns {missing!r} for {ticker!r}")
    return df[expected]


__all__ = ["TIINGO_BASE", "TiingoError", "fetch_daily_prices"]
