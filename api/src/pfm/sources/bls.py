"""BLS (Bureau of Labor Statistics) public-API client.

Documentation: https://www.bls.gov/developers/api_signature_v2.htm

The BLS v2 API is POST-based:

    POST https://api.bls.gov/publicAPI/v2/timeseries/data/

with a JSON body of the form::

    {
        "seriesid": ["LNS14000000"],
        "startyear": "2020",
        "endyear":   "2026",
        "registrationkey": "<optional>"
    }

Tier limits (per docs as of 2026):

    * unregistered: 25 daily queries, 10 yrs / query, 25 series / query,
      no calculations or annual averages.
    * registered (free key): 500 daily queries, 20 yrs / query, 50 series
      / query, calculations + annual averages allowed.

Pass the key via the ``BLS_API_KEY`` env var. Without it we fall back to
the unregistered tier (and rely on aggressive caching to stay under 25
queries/day even when the frontend is hammering /macro/bls).

Curated series — see ``_BLS_SERIES_REGISTRY`` for the catalog. We
deliberately don't expose every BLS series; we curate the macro-relevant
ones that pair well with prediction markets.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)

BLS_API_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# 7-day TTL — BLS data is monthly, so even refreshing once per week is
# overkill. We don't want to redownload on every page load.
_BLS_TTL_SECONDS = 7 * 24 * 3600


# --- Curated registry of BLS series we care about. --------------------------
_BLS_SERIES_REGISTRY: dict[str, dict[str, str]] = {
    "LNS14000000": {
        "name": "Unemployment Rate (U-3)",
        "frequency": "monthly",
        "units": "Percent",
        "desc": "Civilian unemployment rate, 16+ years, seasonally adjusted (BLS U-3)",
        "citation": "https://data.bls.gov/timeseries/LNS14000000",
    },
    "CES0500000003": {
        "name": "Average Hourly Earnings (Total Private)",
        "frequency": "monthly",
        "units": "USD per Hour",
        "desc": "Average hourly earnings of all employees, total private (seasonally adjusted)",
        "citation": "https://data.bls.gov/timeseries/CES0500000003",
    },
    "CUUR0000SA0L1E": {
        "name": "CPI Core (Less Food & Energy)",
        "frequency": "monthly",
        "units": "Index 1982-84=100",
        "desc": "CPI All Urban Consumers, all items less food and energy (NSA)",
        "citation": "https://data.bls.gov/timeseries/CUUR0000SA0L1E",
    },
    "WPSFD49207": {
        "name": "PPI Final Demand",
        "frequency": "monthly",
        "units": "Index Nov 2009=100",
        "desc": "Producer Price Index by Commodity: Final Demand (seasonally adjusted)",
        "citation": "https://data.bls.gov/timeseries/WPSFD49207",
    },
    "LNS12300000": {
        "name": "Labor Force Participation Rate",
        "frequency": "monthly",
        "units": "Percent",
        "desc": "Civilian labor force participation rate, 16+ years (seasonally adjusted)",
        "citation": "https://data.bls.gov/timeseries/LNS12300000",
    },
}


class BlsDataError(RuntimeError):
    """Raised on BLS fetch error."""


class BlsSeriesMetadata(BaseModel):
    """Pydantic schema for a single BLS series."""

    model_config = ConfigDict(extra="forbid")

    series_id: str
    name: str
    frequency: str
    units: str
    citation: str


# --- month decoding ---------------------------------------------------------
# BLS returns ``period`` like "M01"-"M12" for monthly, "Q01"-"Q04" for
# quarterly, "A01" for annual. We only support monthly here (every series
# in the registry is monthly).
_MONTH_FROM_PERIOD: dict[str, int] = {f"M{i:02d}": i for i in range(1, 13)}


def _parse_bls_payload(payload: dict[str, Any], series_id: str) -> pd.Series:
    """Convert a BLS v2 JSON payload to a daily-resampled pandas Series.

    BLS structure::

        {
          "status": "REQUEST_SUCCEEDED",
          "responseTime": ...,
          "message": [...],
          "Results": {"series": [{"seriesID": "...", "data": [...]}, ...]}
        }

    Each ``data`` row has ``year`` (str), ``period`` ("M01" - "M12"),
    ``periodName``, ``value`` (str), ``footnotes`` (list).
    """
    status = payload.get("status")
    if status != "REQUEST_SUCCEEDED":
        msgs = payload.get("message") or [str(payload)[:200]]
        raise BlsDataError(f"BLS API status={status!r}: {'; '.join(msgs)}")
    series = payload.get("Results", {}).get("series", [])
    if not series:
        return pd.Series(dtype=float, name=series_id)
    rows = series[0].get("data", [])
    if not rows:
        return pd.Series(dtype=float, name=series_id)

    records: list[tuple[pd.Timestamp, float]] = []
    for r in rows:
        period = r.get("period", "")
        month = _MONTH_FROM_PERIOD.get(period)
        if month is None:
            # Skip annual / quarterly aggregates — they appear with
            # period "M13" or "Q01".
            continue
        year = int(r["year"])
        try:
            value = float(r["value"])
        except (TypeError, ValueError):
            continue
        ts = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        records.append((ts, value))

    if not records:
        return pd.Series(dtype=float, name=series_id)
    records.sort(key=lambda t: t[0])
    idx = pd.DatetimeIndex([r[0] for r in records], name="date")
    out = pd.Series([r[1] for r in records], index=idx, name=series_id)
    return out


def fetch_bls_series(
    series_id: str,
    start_year: int,
    end_year: int,
    *,
    api_key: str | None = None,
    client: httpx.Client | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch a BLS series and return a DataFrame ``[date, value]``.

    Args:
        series_id: BLS series id, e.g. ``"LNS14000000"``.
        start_year, end_year: 4-digit years inclusive. The unregistered
            tier limits to 10 years per request; the registered tier to
            20. We don't enforce this client-side — BLS returns an error
            message we propagate via :class:`BlsDataError`.
        api_key: BLS registration key. Falls back to ``BLS_API_KEY`` env
            var. ``None`` is fine — drops to the unregistered tier.
        client: optional httpx.Client (for connection pooling / mocking).
        max_retries: retries on 429 / 5xx.

    Returns:
        DataFrame with ``date`` and ``value`` columns, sorted ascending
        by date. Empty DataFrame if BLS returned no rows.

    Raises:
        BlsDataError: if BLS responds with non-success status or the HTTP
            transport fails after retries.
    """
    key = api_key if api_key is not None else os.environ.get("BLS_API_KEY")
    body: dict[str, Any] = {
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if key:
        body["registrationkey"] = key

    own_client = client is None
    cli = client or httpx.Client(timeout=20.0)
    try:
        attempts = 0
        delay = 1.0
        while True:
            attempts += 1
            try:
                resp = cli.post(BLS_API_BASE, json=body)
            except httpx.HTTPError as e:
                if attempts >= max_retries:
                    raise BlsDataError(f"BLS transient error: {e}") from e
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempts >= max_retries:
                    raise BlsDataError(
                        f"BLS rate-limit/server error after {attempts}: "
                        f"{resp.status_code} {resp.text[:200]}"
                    )
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if resp.status_code != 200:
                raise BlsDataError(f"BLS HTTP {resp.status_code}: {resp.text[:200]}")
            break
        try:
            payload = resp.json()
        except ValueError as e:
            raise BlsDataError(f"BLS returned non-JSON: {resp.text[:200]}") from e
    finally:
        if own_client:
            cli.close()

    s = _parse_bls_payload(payload, series_id)
    if s.empty:
        return pd.DataFrame(columns=["date", "value"])
    df = s.reset_index()
    df.columns = ["date", "value"]
    return df


class BLSClient:
    """Cache-fronted BLS client.

    Intended to be a long-lived instance; ``fetch()`` checks the
    ``bls-series`` cache namespace (TTL 7 days) before hitting the API.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("BLS_API_KEY")
        self._cache = get_cache("bls-series", ttl=_BLS_TTL_SECONDS)

    def fetch(
        self,
        series_id: str,
        start_year: int,
        end_year: int,
        *,
        client: httpx.Client | None = None,
    ) -> pd.DataFrame:
        """Cache-fronted wrapper around :func:`fetch_bls_series`."""
        cache_key = (series_id, int(start_year), int(end_year))
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit.copy()
        df = fetch_bls_series(
            series_id,
            start_year,
            end_year,
            api_key=self.api_key,
            client=client,
        )
        self._cache.set(cache_key, df.copy(), ttl=_BLS_TTL_SECONDS)
        return df


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/macro/bls", tags=["macro-bls"])


@router.get("/catalog")
def bls_catalog() -> dict[str, Any]:
    """List the curated BLS series this service supports."""
    series = [
        BlsSeriesMetadata(
            series_id=sid,
            name=meta["name"],
            frequency=meta["frequency"],
            units=meta["units"],
            citation=meta["citation"],
        )
        for sid, meta in _BLS_SERIES_REGISTRY.items()
    ]
    return {
        "count": len(series),
        "series": [s.model_dump() for s in series],
    }


@router.get("/{series_id}")
def bls_series_endpoint(
    series_id: str,
    start: int = Query(2020, ge=1900, le=2100, description="Start year"),
    end: int = Query(2026, ge=1900, le=2100, description="End year"),
) -> dict[str, Any]:
    """Fetch a curated BLS series.

    Unknown series id → 404. Tier-limit / API errors → 502.
    """
    if series_id not in _BLS_SERIES_REGISTRY:
        raise HTTPException(status_code=404, detail=f"unknown BLS series {series_id!r}")
    if start > end:
        raise HTTPException(status_code=400, detail="start year must be <= end year")
    cli = BLSClient()
    try:
        df = cli.fetch(series_id, start, end)
    except BlsDataError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    meta = _BLS_SERIES_REGISTRY[series_id]
    return {
        "series_id": series_id,
        "name": meta["name"],
        "units": meta["units"],
        "frequency": meta["frequency"],
        "citation": meta["citation"],
        "start_year": start,
        "end_year": end,
        "data": [
            {"date": ts.date().isoformat(), "value": float(v)}
            for ts, v in zip(df["date"], df["value"], strict=False)
            if pd.notna(v)
        ],
    }


__all__ = [
    "BLS_API_BASE",
    "_BLS_SERIES_REGISTRY",
    "BLSClient",
    "BlsDataError",
    "BlsSeriesMetadata",
    "fetch_bls_series",
    "router",
]
