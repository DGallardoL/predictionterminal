"""Binance public-REST klines fetcher (no auth, no websockets).

We use the public ``/api/v3/klines`` endpoint to pull daily OHLCV bars for
spot pairs (e.g. BTCUSDT). Daily resolution is enough for the
spot-vs-market-implied probability comparison; sub-daily would only matter
for intra-day-resolution markets which are rare on Polymarket / Kalshi.

Rate limits: 1200 weight/min for the public REST endpoint; daily klines
costs weight=1 and we cache aggressively. Retries on transient 5xx /
429 with exponential backoff.

Resolution-source caveat: Polymarket BTC markets typically settle on UMA
disputed resolution against Coinbase or an aggregated index, NOT Binance.
Spot quotes between exchanges can deviate by 1–5 bps near the strike. The
caller is responsible for understanding this basis (see
``docs/strategies.md`` §1.4).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_API_BASE = "https://api.binance.com"
KLINE_PATH = "/api/v3/klines"
DEFAULT_INTERVAL = "1d"
MAX_BARS_PER_CALL = 1000  # Binance hard cap.

# Process-local klines cache keyed on (symbol, interval, start, end, limit).
# Binance's daily bars are stable once the candle closes, so even a long TTL
# is "safe" — we use 5 min so a re-run during the same Terminal session
# avoids the upstream round-trip without holding onto stale partial candles
# beyond the next refresh. Sub-daily klines change every bar interval; the
# 5-min TTL is short enough to not surface stale 1m / 5m bars in the UI
# (which refreshes on a similar cadence anyway).
_KLINES_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}
_KLINES_CACHE_LOCK = threading.Lock()
_KLINES_CACHE_TTL_S: float = 300.0
_KLINES_CACHE_MAX_ENTRIES: int = 512


class BinanceError(RuntimeError):
    """Raised on a Binance error that we don't retry through."""


def _parse_klines(rows: list[list[Any]], *, normalize_to_date: bool = True) -> pd.DataFrame:
    """Convert raw Binance klines into a UTC-indexed OHLCV DataFrame.

    Each row is:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]

    Args:
        rows: raw kline payload.
        normalize_to_date: if True (daily-bar default), the index is the
            UTC calendar date. Set False for sub-daily intervals (5m, 1h)
            where the time-of-day component matters.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "_ignore",
        ],
    )
    ts = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    if normalize_to_date:
        ts = ts.dt.normalize()
    out = pd.DataFrame(
        {
            "open": df["open"].astype(float).to_numpy(),
            "high": df["high"].astype(float).to_numpy(),
            "low": df["low"].astype(float).to_numpy(),
            "close": df["close"].astype(float).to_numpy(),
            "volume": df["volume"].astype(float).to_numpy(),
        },
        index=pd.DatetimeIndex(ts),
    )
    out.index.name = "open_time"
    return out


# Map interval string → (bars per year for crypto-24/7, sub_daily flag).
INTERVAL_TABLE: dict[str, tuple[float, bool]] = {
    "1m": (525_600.0, True),
    "3m": (175_200.0, True),
    "5m": (105_120.0, True),
    "15m": (35_040.0, True),
    "30m": (17_520.0, True),
    "1h": (8_760.0, True),
    "2h": (4_380.0, True),
    "4h": (2_190.0, True),
    "6h": (1_460.0, True),
    "8h": (1_095.0, True),
    "12h": (730.0, True),
    "1d": (365.0, False),
    "3d": (121.6667, False),
    "1w": (52.0, False),
}


def annualisation_for_interval(interval: str) -> float:
    """Bars-per-year for an interval, used by σ̂ annualisation in
    :mod:`pfm.spot_implied`. Crypto trades 24/7 → uses calendar bars."""
    if interval not in INTERVAL_TABLE:
        raise ValueError(f"unknown interval {interval!r}; expected one of {sorted(INTERVAL_TABLE)}")
    return INTERVAL_TABLE[interval][0]


class BinanceClient:
    """Stateless wrapper around the Binance public REST API."""

    def __init__(
        self,
        *,
        base_url: str = BINANCE_API_BASE,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
        max_retries: int = 5,
    ) -> None:
        self.base_url = base_url
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)
        self.max_retries = max_retries

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        attempts = 0
        delay = 1.0
        while True:
            attempts += 1
            try:
                resp = self._client.get(self.base_url + path, params=dict(params))
            except httpx.HTTPError as e:
                if attempts >= self.max_retries:
                    raise
                logger.warning(
                    "binance %s transient error: %s; retry %d/%d",
                    path,
                    e,
                    attempts,
                    self.max_retries,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempts >= self.max_retries:
                    raise BinanceError(
                        f"binance rate-limit/server error after {attempts} attempts: "
                        f"{resp.status_code} {resp.text[:200]}"
                    )
                logger.warning(
                    "binance %s rate-limit/server: %s; retry %d/%d",
                    path,
                    resp.status_code,
                    attempts,
                    self.max_retries,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            if resp.status_code != 200:
                raise BinanceError(f"binance {path}: {resp.status_code} {resp.text[:200]}")
            return resp.json()

    def get_klines(
        self,
        symbol: str,
        *,
        interval: str = DEFAULT_INTERVAL,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        limit: int = MAX_BARS_PER_CALL,
    ) -> pd.DataFrame:
        """Daily klines for ``symbol`` (e.g. ``"BTCUSDT"``).

        Pages through if ``[start, end]`` exceeds ``limit`` bars.

        Args:
            symbol: Binance trading pair, uppercase (e.g. ``"BTCUSDT"``).
            interval: ``1d`` (daily), ``1h``, etc. Default daily.
            start: inclusive UTC ``pd.Timestamp``.
            end: inclusive UTC ``pd.Timestamp``.
            limit: per-call bar cap (Binance enforces 1000 max).

        Returns:
            DataFrame indexed by UTC date with ``open, high, low, close,
            volume`` columns. Empty DataFrame if no bars in window.
        """
        if limit <= 0 or limit > MAX_BARS_PER_CALL:
            raise ValueError(f"limit must be in (0, {MAX_BARS_PER_CALL}], got {limit}")
        if interval not in INTERVAL_TABLE:
            raise ValueError(
                f"unknown interval {interval!r}; expected one of {sorted(INTERVAL_TABLE)}"
            )
        sub_daily = INTERVAL_TABLE[interval][1]

        # Process-local cache check. Keyed on the full query so different
        # windows / intervals don't collide. 5-minute TTL — see top of file.
        cache_key = (
            symbol.upper(),
            interval,
            None if start is None else int(start.timestamp()),
            None if end is None else int(end.timestamp()),
            int(limit),
        )
        now = time.time()
        with _KLINES_CACHE_LOCK:
            cached = _KLINES_CACHE.get(cache_key)
            if cached is not None and (now - cached[0]) < _KLINES_CACHE_TTL_S:
                return cached[1].copy()

        all_rows: list[list[Any]] = []
        cursor = start
        while True:
            params: dict[str, Any] = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": limit,
            }
            if cursor is not None:
                params["startTime"] = int(cursor.timestamp() * 1000)
            if end is not None:
                params["endTime"] = int(end.timestamp() * 1000)
            page = self._get(KLINE_PATH, params)
            if not isinstance(page, list):
                raise BinanceError(f"binance klines: unexpected payload {type(page).__name__}")
            if not page:
                break
            all_rows.extend(page)
            if len(page) < limit:
                break
            # Page forward: the last bar's open_time + 1 ms.
            last_open = int(page[-1][0])
            cursor = pd.Timestamp(last_open + 1, unit="ms", tz="UTC")
            if end is not None and cursor > end:
                break

        df = _parse_klines(all_rows, normalize_to_date=not sub_daily)
        if start is not None:
            lo = start.normalize() if not sub_daily else start
            df = df[df.index >= lo]
        if end is not None:
            hi = end.normalize() if not sub_daily else end
            df = df[df.index <= hi]

        with _KLINES_CACHE_LOCK:
            if len(_KLINES_CACHE) >= _KLINES_CACHE_MAX_ENTRIES:
                victims = sorted(_KLINES_CACHE.items(), key=lambda kv: kv[1][0])
                for k, _ in victims[: _KLINES_CACHE_MAX_ENTRIES // 4]:
                    _KLINES_CACHE.pop(k, None)
            _KLINES_CACHE[cache_key] = (now, df.copy())
        return df
