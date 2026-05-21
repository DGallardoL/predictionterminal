"""External annualized σ benchmarks per asset+tenor (A2).

Unified façade returning external benchmarks that can be compared against
``σ_PM`` produced by :mod:`pfm.vol.pm_iv_extractor`.

Sources
-------
- **FRED**: ``VIXCLS`` (S&P 500 IV), ``OVXCLS`` (WTI crude oil ETF IV),
  ``GVZCLS`` (gold ETF IV) — daily index values, divided by 100 to obtain
  decimal annualized σ (VIX 18.5 → 0.185).
- **Deribit**: ``btc_dvol`` / ``eth_dvol`` index price endpoint — live
  annualized vol in percent, divided by 100 for decimal.
- **Binance**: daily klines for ``BTCUSDT`` / ``ETHUSDT``, log returns,
  ``√365`` annualization.

All fetchers return a :class:`VolBenchmark` and are individually cached
behind the shared ``vol_benchmarks`` namespace in :mod:`pfm.cache_utils`.
Failures in single sources during :func:`get_benchmark_for_asset` are
caught and logged so partial results are still returned.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from itertools import pairwise
from typing import Literal

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.sources.fred import FREDGRAPH_BASE, _parse_fredgraph_csv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DERIBIT_INDEX_URL = "https://www.deribit.com/api/v2/public/get_index_price"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

_FRED_TTL_S = 600  # daily data; FRED itself updates once/day
_DERIBIT_TTL_S = 60  # live index
_BINANCE_TTL_S = 300  # daily klines

_CACHE_NS = "vol_benchmarks"

_DAILY_STALE_AFTER_S = 24 * 3600 + 3600  # 25h grace for daily series
_LIVE_STALE_AFTER_S = 5 * 60  # 5min for live Deribit


SourceLiteral = Literal[
    "fred_vix",
    "fred_ovx",
    "fred_gvz",
    "deribit_btc_dvol",
    "deribit_eth_dvol",
    "binance_realized",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class VolBenchmark(BaseModel):
    """External annualized σ benchmark for a single (asset, tenor, source)."""

    asset: str
    source: SourceLiteral
    sigma_annual: float = Field(..., ge=0)
    tenor_label: str  # "30d", "spot", "rolling_30d"
    as_of_utc: datetime
    stale_warning: bool  # >24h for daily; >5min for live
    raw_value: float | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


def _cache() -> object:
    return get_cache(_CACHE_NS, ttl=_FRED_TTL_S)


def _fetch_fred_latest(
    series_id: str,
    *,
    http: httpx.Client | None,
) -> tuple[float, pd.Timestamp]:
    """Pull the most-recent non-NaN observation from a daily FRED series.

    Returns ``(latest_value, latest_date_utc)``.
    """
    own = http is None
    cli = http or httpx.Client(timeout=20.0)
    try:
        resp = cli.get(FREDGRAPH_BASE, params={"id": series_id})
        if resp.status_code != 200:
            raise RuntimeError(f"FRED {series_id} HTTP {resp.status_code}: {resp.text[:200]}")
        s = _parse_fredgraph_csv(resp.text, series_id)
    finally:
        if own:
            cli.close()
    s = s.dropna()
    if s.empty:
        raise RuntimeError(f"FRED {series_id} returned no usable observations")
    last_idx = s.index[-1]
    if not isinstance(last_idx, pd.Timestamp):
        last_idx = pd.Timestamp(last_idx)
    if last_idx.tzinfo is None:
        last_idx = last_idx.tz_localize("UTC")
    else:
        last_idx = last_idx.tz_convert("UTC")
    return float(s.iloc[-1]), last_idx


def _fred_benchmark(
    *,
    asset: str,
    source: SourceLiteral,
    series_id: str,
    tenor_label: str,
    notes: str,
    http: httpx.Client | None,
) -> VolBenchmark:
    """Generic FRED-based VIX-style benchmark fetcher with cache."""
    cache = _cache()
    key = (source, asset, tenor_label, series_id)
    hit = cache.get(key)
    if hit is not None:
        return hit

    raw_value, last_date = _fetch_fred_latest(series_id, http=http)
    sigma = raw_value / 100.0

    now = _now_utc()
    age_s = (now - last_date.to_pydatetime()).total_seconds()
    stale = age_s > _DAILY_STALE_AFTER_S

    bench = VolBenchmark(
        asset=asset,
        source=source,
        sigma_annual=sigma,
        tenor_label=tenor_label,
        as_of_utc=last_date.to_pydatetime(),
        stale_warning=stale,
        raw_value=raw_value,
        notes=notes,
    )
    cache.set(key, bench, ttl=_FRED_TTL_S)
    return bench


# ---------------------------------------------------------------------------
# Public fetchers — single source
# ---------------------------------------------------------------------------


def fetch_vix(*, http: httpx.Client | None = None) -> VolBenchmark:
    """Most recent VIXCLS from FRED. Annualized σ in DECIMAL form."""
    return _fred_benchmark(
        asset="SPX",
        source="fred_vix",
        series_id="VIXCLS",
        tenor_label="30d",
        notes="CBOE VIX — 30-day implied vol of S&P 500 options (decimal).",
        http=http,
    )


def fetch_ovx(*, http: httpx.Client | None = None) -> VolBenchmark:
    """OVXCLS from FRED (CBOE crude oil ETF vol index)."""
    return _fred_benchmark(
        asset="WTI",
        source="fred_ovx",
        series_id="OVXCLS",
        tenor_label="30d",
        notes="CBOE OVX — 30-day implied vol of USO (WTI crude) options (decimal).",
        http=http,
    )


def fetch_gvz(*, http: httpx.Client | None = None) -> VolBenchmark:
    """GVZCLS from FRED (CBOE gold ETF vol index)."""
    return _fred_benchmark(
        asset="GOLD",
        source="fred_gvz",
        series_id="GVZCLS",
        tenor_label="30d",
        notes="CBOE GVZ — 30-day implied vol of GLD (gold ETF) options (decimal).",
        http=http,
    )


def fetch_deribit_dvol(
    asset: Literal["BTC", "ETH"],
    *,
    http: httpx.Client | None = None,
) -> VolBenchmark:
    """Live BTC/ETH DVOL index from Deribit public REST.

    Endpoint: ``GET /api/v2/public/get_index_price?index_name=btc_dvol`` (or
    ``eth_dvol``). The ``index_price`` field is annualized vol in percent;
    we divide by 100 to obtain a decimal.

    Deribit does not return a server timestamp on this endpoint, so the
    ``stale_warning`` flag is always ``False`` here — the cache TTL is the
    real freshness guarantee.
    """
    if asset not in ("BTC", "ETH"):
        raise ValueError(f"unsupported Deribit DVOL asset {asset!r}")

    index_name = "btc_dvol" if asset == "BTC" else "eth_dvol"
    source: SourceLiteral = "deribit_btc_dvol" if asset == "BTC" else "deribit_eth_dvol"

    cache = _cache()
    key = (source, asset, "spot")
    hit = cache.get(key)
    if hit is not None:
        return hit

    own = http is None
    cli = http or httpx.Client(timeout=15.0)
    try:
        resp = cli.get(DERIBIT_INDEX_URL, params={"index_name": index_name})
        if resp.status_code != 200:
            raise RuntimeError(f"Deribit {index_name} HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
    finally:
        if own:
            cli.close()

    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or "index_price" not in result:
        raise RuntimeError(f"Deribit {index_name}: unexpected payload shape")

    raw_value = float(result["index_price"])
    sigma = raw_value / 100.0

    bench = VolBenchmark(
        asset=asset,
        source=source,
        sigma_annual=sigma,
        tenor_label="spot",
        as_of_utc=_now_utc(),
        stale_warning=False,
        raw_value=raw_value,
        notes=(
            f"Deribit {index_name} (live annualized vol in %, /100 for decimal). "
            "No server timestamp; freshness is bounded by the local cache TTL."
        ),
    )
    cache.set(key, bench, ttl=_DERIBIT_TTL_S)
    return bench


def fetch_binance_realized_sigma(
    symbol: str,
    window_days: int = 30,
    *,
    http: httpx.Client | None = None,
) -> VolBenchmark:
    """Annualized realized σ from Binance daily klines.

    Pulls ``window_days + 1`` daily closes, computes log returns and
    annualizes by ``√365`` (crypto trades 24/7).
    """
    if window_days < 2:
        raise ValueError(f"window_days must be ≥ 2, got {window_days}")
    sym = symbol.upper()
    asset = "BTC" if sym.startswith("BTC") else "ETH" if sym.startswith("ETH") else sym

    cache = _cache()
    key = ("binance_realized", asset, window_days, sym)
    hit = cache.get(key)
    if hit is not None:
        return hit

    own = http is None
    cli = http or httpx.Client(timeout=20.0)
    try:
        resp = cli.get(
            BINANCE_KLINES_URL,
            params={"symbol": sym, "interval": "1d", "limit": window_days + 1},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Binance klines {sym} HTTP {resp.status_code}: {resp.text[:200]}")
        rows = resp.json()
    finally:
        if own:
            cli.close()

    if not isinstance(rows, list) or len(rows) < 2:
        raise RuntimeError(
            f"Binance klines {sym}: insufficient bars ({len(rows) if isinstance(rows, list) else 'n/a'})"
        )

    closes: list[float] = []
    last_close_ms: int = 0
    for row in rows:
        # row: [open_time, open, high, low, close, volume, close_time, ...]
        closes.append(float(row[4]))
        last_close_ms = max(last_close_ms, int(row[6]))

    log_returns: list[float] = []
    for prev, cur in pairwise(closes):
        if prev <= 0 or cur <= 0:
            continue
        log_returns.append(math.log(cur / prev))
    if len(log_returns) < 2:
        raise RuntimeError(f"Binance klines {sym}: too few positive closes")

    # Sample std with ddof=1 (matches numpy default for sample variance).
    n = len(log_returns)
    mean = sum(log_returns) / n
    var = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    sigma_daily = math.sqrt(var)
    sigma_annual = sigma_daily * math.sqrt(365.0)

    last_close_dt = (
        datetime.fromtimestamp(last_close_ms / 1000.0, tz=UTC) if last_close_ms > 0 else _now_utc()
    )
    age_s = (_now_utc() - last_close_dt).total_seconds()
    stale = age_s > _DAILY_STALE_AFTER_S

    bench = VolBenchmark(
        asset=asset,
        source="binance_realized",
        sigma_annual=sigma_annual,
        tenor_label=f"rolling_{window_days}d",
        as_of_utc=last_close_dt,
        stale_warning=stale,
        raw_value=sigma_daily,
        notes=(
            f"Binance {sym} realized σ over {n} daily log returns, "
            "annualized by √365 (24/7 crypto)."
        ),
    )
    cache.set(key, bench, ttl=_BINANCE_TTL_S)
    return bench


# ---------------------------------------------------------------------------
# Convenience: multi-source per asset
# ---------------------------------------------------------------------------


def _safe(label: str, fn) -> VolBenchmark | None:  # type: ignore[no-untyped-def]
    try:
        return fn()
    except Exception as e:
        logger.warning("vol_benchmarks: %s failed: %s", label, e)
        return None


def get_benchmark_for_asset(
    asset: str,
    tenor_days: int,
    *,
    realized_window_days: int = 30,
    http: httpx.Client | None = None,
) -> dict[str, VolBenchmark]:
    """Return every external benchmark we can produce for ``asset``.

    Mapping:
      - ``SPX``  → ``{"vix": fetch_vix()}``
      - ``WTI``  → ``{"ovx": fetch_ovx()}``
      - ``GOLD`` → ``{"gvz": fetch_gvz()}``
      - ``BTC``  → ``{"dvol": fetch_deribit_dvol("BTC"),
                     "realized_30d": fetch_binance_realized_sigma("BTCUSDT")}``
      - ``ETH``  → analogous to BTC

    Unknown assets return ``{}``. Failures in any single source are caught
    and the offending key is omitted; remaining benchmarks are still
    returned. Returns ``{}`` if every source fails.

    The ``tenor_days`` argument is accepted for forward-compatibility (e.g.
    a future ``term_structure`` benchmark) but currently does not change
    which sources are queried — VIX/OVX/GVZ are inherently 30-day; DVOL is
    spot; Binance realized σ uses ``realized_window_days``.
    """
    a = asset.upper()
    out: dict[str, VolBenchmark] = {}

    if a == "SPX":
        vix = _safe("vix", lambda: fetch_vix(http=http))
        if vix is not None:
            out["vix"] = vix
    elif a == "WTI":
        ovx = _safe("ovx", lambda: fetch_ovx(http=http))
        if ovx is not None:
            out["ovx"] = ovx
    elif a == "GOLD":
        gvz = _safe("gvz", lambda: fetch_gvz(http=http))
        if gvz is not None:
            out["gvz"] = gvz
    elif a in ("BTC", "ETH"):
        dvol = _safe(
            f"{a.lower()}_dvol",
            lambda: fetch_deribit_dvol(a, http=http),  # type: ignore[arg-type]
        )
        if dvol is not None:
            out["dvol"] = dvol
        symbol = "BTCUSDT" if a == "BTC" else "ETHUSDT"
        realized = _safe(
            f"binance_realized_{symbol}",
            lambda: fetch_binance_realized_sigma(
                symbol,
                window_days=realized_window_days,
                http=http,
            ),
        )
        if realized is not None:
            out[f"realized_{realized_window_days}d"] = realized
    else:
        logger.info("vol_benchmarks: no benchmark mapping for asset %r", asset)

    return out


__all__ = [
    "BINANCE_KLINES_URL",
    "DERIBIT_INDEX_URL",
    "VolBenchmark",
    "fetch_binance_realized_sigma",
    "fetch_deribit_dvol",
    "fetch_gvz",
    "fetch_ovx",
    "fetch_vix",
    "get_benchmark_for_asset",
]
