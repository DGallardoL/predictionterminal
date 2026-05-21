"""Shared regression helpers extracted from ``pfm.main``.

These helpers compose the data pipeline for ``/fit``, ``/attribution``,
``/factors/rank``, ``/factors/permutation``, ``/factors/best`` and most of
``/strategies/*``. Living here (instead of inside ``pfm.main``) gives feature
routers a stable import path without the lazy-import dance they previously
needed to avoid the ``pfm.main`` ↔ router cycle.

The functions still rely on ``app.state``-resident clients/caches (Polymarket,
Redis, Settings). Callers pass them in explicitly — these helpers do NOT
reach for the FastAPI ``Request`` object.
"""

from __future__ import annotations

import logging
import math
from datetime import date

import httpx
import pandas as pd
from fastapi import HTTPException

from pfm.cache import CacheBackend
from pfm.config import Settings
from pfm.dependencies import cache_key as _cache_key
from pfm.factor_resolver import (
    resolve_factor as _resolve_factor_unified,
)
from pfm.factor_resolver import (
    suggest_factors_with_meta as _factor_suggest_meta,
)
from pfm.factors import CHAIN_SOURCE, FactorConfig
from pfm.model import delta_level, delta_logit
from pfm.schemas import (
    AlignmentLit,
    CustomFactor,
    ReturnTypeLit,
)
from pfm.sources.chain import segments_signature
from pfm.sources.equity import EquityDataError
from pfm.sources.kalshi import KalshiClient, KalshiError
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
)

logger = logging.getLogger(__name__)


def _main_attr(name: str):
    """Return ``getattr(pfm.main, name)`` for symbols tests monkeypatch.

    Tests patch the data layer with ``monkeypatch.setattr(pfm.main, …)``.
    For the patch to affect calls originating inside this module we resolve
    those names through ``pfm.main`` at call time instead of binding the
    function references at import time. Production-path behaviour is
    unchanged — ``pfm.main`` always re-imports the canonical sources.
    """
    from pfm import main as _m

    return getattr(_m, name)


def _current_kalshi_client() -> KalshiClient:
    """Return the singleton ``KalshiClient`` from ``pfm.main.app.state``.

    Lazy-imports ``pfm.main`` so this module stays loadable when ``pfm.main``
    is still mid-initialisation (the typical state during start-up since
    ``main`` imports this module).
    """
    from pfm import main as _m

    return getattr(_m.app.state, "kalshi", None) or KalshiClient()


_POLY_FANOUT_SEMAPHORE_SIZE: int = 20


def _resolve_one(fid: str, factors: dict[str, FactorConfig], *, role: str) -> FactorConfig:
    """Resolve a single factor id/slug/name or raise 400 with ``did_you_mean``.

    The legacy contract was an exact id match; we now also accept slug or
    case-insensitive name. On miss the response embeds top-3 suggestions
    in the structured detail object so users can copy the right id.
    """
    fc = _resolve_factor_unified(fid, factors)
    if fc is not None:
        return fc
    raise HTTPException(
        status_code=400,
        detail={
            "error": f"unknown factor id ({role}): {fid!r}",
            "query": fid,
            "role": role,
            "did_you_mean": _factor_suggest_meta(fid, factors, top_k=3),
        },
    )


def _fetch_aligned_prob(
    fc: FactorConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
) -> pd.Series:
    """Fetch a factor's daily probability series clipped to ``[start, end]``.

    Pure probability — no Δlogit, no equity-calendar shift. Used by the
    strategies endpoints which work on raw probabilities, not factor returns.
    """
    df = _main_attr("_cached_factor_history")(fc, start, end, poly, cache, settings)
    if df.empty:
        raise HTTPException(
            status_code=502,
            detail=f"{fc.source} returned no history for factor {fc.id!r} (slug={fc.slug!r})",
        )
    df = df[(df.index >= start) & (df.index <= end)]
    return df["price"].rename(fc.id)


def _resolve_factor_specs(
    factor_ids: list[str],
    custom: list[CustomFactor],
    yaml_factors: dict[str, FactorConfig],
) -> list[FactorConfig]:
    """Return ordered list of ``FactorConfig`` instances combining yaml + custom.

    Raises 400 with a structured ``did_you_mean`` payload if any yaml id /
    slug / name doesn't resolve, or if a custom id collides with the yaml
    ids. Custom factors are wrapped as single-source ``polymarket`` factors
    for back-compat (chained custom factors are intentionally not
    supported via the public API).

    Special syntax: ``sentiment:<query>`` (e.g. ``sentiment:bitcoin``) is
    synthesised on the fly into a level-source ``FactorConfig`` that
    pulls news-sentiment for the given query. The curated
    ``sentiment_*`` ids live in the yaml catalog and resolve via the
    normal path.
    """
    out: list[FactorConfig] = []
    seen: set[str] = set()
    resolved: list[FactorConfig] = []
    unknown_with_hints: list[dict[str, object]] = []
    # Lazy import — avoids paying for it when no sentiment factor is used.
    from pfm.sources.sentiment_factor import parse_sentiment_factor_id

    for fid in factor_ids:
        is_sent, sent_query = parse_sentiment_factor_id(fid)
        if is_sent:
            if not sent_query:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": (
                            f"sentiment factor id {fid!r} has no query — "
                            "use sentiment:<keyword>, e.g. sentiment:bitcoin"
                        ),
                        "query": fid,
                    },
                )
            # ``id`` is the user-typed string so attribution rows /
            # factor-trace UI display exactly what they asked for.
            resolved.append(
                FactorConfig(
                    id=fid,
                    name=f"News sentiment: {sent_query}",
                    slug=sent_query,
                    source="sentiment",
                    description=(
                        "Daily mean signed news sentiment (GDELT timelinetone "
                        "+ Reddit + HN, VADER + finance-lex blended)."
                    ),
                    theme="sentiment",
                    is_probability=False,
                )
            )
            continue
        fc = _resolve_factor_unified(fid, yaml_factors)
        if fc is None:
            unknown_with_hints.append(
                {
                    "query": fid,
                    "did_you_mean": _factor_suggest_meta(fid, yaml_factors, top_k=3),
                }
            )
        else:
            resolved.append(fc)
    if unknown_with_hints:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"{len(unknown_with_hints)} factor id(s) not found",
                "unknown": unknown_with_hints,
            },
        )
    for fc in resolved:
        if fc.id in seen:
            continue
        out.append(fc)
        seen.add(fc.id)
    for cf in custom:
        if cf.id in seen:
            raise HTTPException(status_code=400, detail=f"duplicate factor id: {cf.id!r}")
        out.append(
            FactorConfig(
                id=cf.id,
                name=cf.name or cf.id,
                slug=cf.slug,
                source="polymarket",
                description="(custom)",
                theme="custom",
            )
        )
        seen.add(cf.id)
    return out


def _assemble_design(
    ticker: str,
    factor_specs: list[FactorConfig],
    start: date,
    end: date,
    epsilon: float,
    return_type: ReturnTypeLit,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
    alignment: AlignmentLit = "strict",
    residualize_market: bool = False,
) -> tuple[pd.Series, pd.DataFrame, dict[str, pd.Series]]:
    """Build (y, X, raw_prices) aligned on UTC dates.

    ``raw_prices`` is a dict {factor_id → Series of probabilities} kept around
    so the API can ship factor traces back to the UI for plotting.

    Per-factor history fetches run in parallel via a bounded thread pool —
    historically this was a sequential N+1 loop that took ~N × Polymarket-RTT
    seconds. Now N fetches run concurrently (capped at
    ``_POLY_FANOUT_SEMAPHORE_SIZE``) so wall-clock is dominated by the slowest
    factor, not the sum.
    """
    if start >= end:
        raise HTTPException(status_code=400, detail="start must be < end")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    # Parallel fetch — bounded by max_workers so a /fit with 30 factors still
    # politely caps in-flight Polymarket calls at the same level as /factors/rank.
    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(_POLY_FANOUT_SEMAPHORE_SIZE, max(1, len(factor_specs)))
    fetched: dict[str, pd.DataFrame] = {}
    fetch_errors: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pfm-fetch") as ex:
        future_map = {
            ex.submit(
                _main_attr("_cached_factor_history"), fc, start_ts, end_ts, poly, cache, settings
            ): fc
            for fc in factor_specs
        }
        for fut, fc in future_map.items():
            try:
                fetched[fc.id] = fut.result()
            except BaseException as e:
                fetch_errors[fc.id] = e

    # Re-raise the first error in the original order so callers see the same
    # exception type they would have under the sequential implementation.
    for fc in factor_specs:
        if fc.id in fetch_errors:
            raise fetch_errors[fc.id]

    delta_cols: dict[str, pd.Series] = {}
    raw_prices: dict[str, pd.Series] = {}
    for fc in factor_specs:
        prices = fetched[fc.id]
        if prices.empty:
            raise HTTPException(
                status_code=502,
                detail=f"{fc.source} returned no history for factor {fc.id!r} (slug={fc.slug!r})",
            )
        prices = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]
        raw_prices[fc.id] = prices["price"]
        aligned = _align_factor_prices(prices["price"], start_ts, end_ts, alignment)
        # Probability factors → Δlogit (clip + logit + diff). Level
        # factors (BLS / FRED — yields, indices, claim counts) → plain
        # first difference, since logit is meaningless on a non-[0,1]
        # series. Either way, shift backward 1 day so the regressor
        # captures news during the equity trading day, not the previous
        # calendar day (see _shift_to_stock_calendar).
        if fc.is_probability:
            dl = delta_logit(aligned, epsilon=epsilon).rename(fc.id)
        else:
            dl = delta_level(aligned).rename(fc.id)
        delta_cols[fc.id] = _shift_to_stock_calendar(dl, days=-1)

    X = pd.concat(delta_cols.values(), axis=1).dropna()

    y = _cached_log_returns(ticker, start_ts, end_ts, return_type, cache, settings)
    if residualize_market:
        y = _residualize_against_spy(y, ticker, start_ts, end_ts, return_type, cache, settings)

    common_idx = X.index.intersection(y.index)
    return y.loc[common_idx], X.loc[common_idx], raw_prices


def _finite(x: float | None) -> float | None:
    """Return ``None`` for non-finite floats so they survive JSON encoding."""
    if x is None:
        return None

    return None if math.isnan(x) or math.isinf(x) else x


def _jsafe(x: float) -> float:
    """Replace NaN/Inf with 0.0 so Pydantic's required fields accept them.

    Used for ridge/lasso/quantile fits where some statistics are undefined.
    The frontend interprets NaN-shaped values via the `regression` toggle —
    it knows not to display SE/t/p as meaningful for regularised methods.
    """
    if x is None:
        return 0.0

    if math.isnan(x) or math.isinf(x):
        return 0.0
    return x


def _short_err(e: BaseException) -> str:
    """Compact error message for surfaces where we don't want full stack info."""
    cls = type(e).__name__
    msg = str(e)
    if not msg:
        return cls
    # Trim long URLs from httpx messages.
    if len(msg) > 120:
        msg = msg[:117] + "…"
    return f"{cls}: {msg}"


def _residualize_against_spy(
    y: pd.Series,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    return_type: ReturnTypeLit,
    cache: CacheBackend,
    settings: Settings,
) -> pd.Series:
    """Replace ``y`` with the OLS residual of ``y ~ α + β · y_SPY``.

    Verified empirically to roughly double test R² for tickers with clear
    market-beta exposure that isn't explained by the Polymarket factors
    (e.g. COIN: +0.16 → +0.36 across multiple splits).

    No-op if the ticker IS SPY, or if SPY data isn't available.
    """
    if ticker.upper() == "SPY":
        return y
    try:
        spy = _cached_log_returns("SPY", start, end, return_type, cache, settings)
    except HTTPException:
        return y
    common = y.index.intersection(spy.index)
    if len(common) < 20:
        return y
    yj = y.loc[common]
    sj = spy.loc[common]
    import statsmodels.api as sm

    Xc = sm.add_constant(sj.values, has_constant="add")
    fit = sm.OLS(yj.values, Xc).fit()
    resid = pd.Series(yj.values - fit.fittedvalues, index=common, name="r")
    return resid


def _shift_to_stock_calendar(s: pd.Series, days: int) -> pd.Series:
    """Shift a Polymarket-derived series by ``days`` business days.

    Polymarket bars are timestamped at UTC midnight (verified empirically) —
    a bar labeled 'Oct 1 00:00 UTC' actually represents the price at the end
    of the [Sep 30 00:00, Oct 1 00:00) bucket — i.e., news during Sep 30 UTC.

    yfinance daily returns are labeled by trading-day date but represent close
    at 21:00 UTC of that date. So r_{Oct 1} captures stock movement from
    Sep 30 21:00 UTC to Oct 1 21:00 UTC — mostly Oct 1 news.

    Joining Δlogit_{Oct 1} (= Sep 30 news) with r_{Oct 1} (= Oct 1 news)
    misaligns by ~21 hours. The contemporaneous match is to use the
    Polymarket bar from the NEXT day (Oct 2 midnight = Oct 1 UTC day's news)
    against r_{Oct 1}. Implemented by shifting Δlogit ``backwards`` by 1 day
    (negative ``days``).

    Empirically, days=-1 improves median OOS R² from -0.43 to -0.23 across
    10 tickers / 3 splits (60-experiment audit, /tmp/overnight_test.py).
    """
    if days == 0:
        return s
    return pd.Series(s.values, index=s.index + pd.Timedelta(days=days), name=s.name)


def _align_factor_prices(
    prices: pd.Series,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    alignment: AlignmentLit,
) -> pd.Series:
    """Reindex a factor's price series to a business-day calendar in [start, end].

    With ``alignment='strict'`` the original series is returned unchanged —
    the inner-join later will drop days where the factor has no data.

    With ``alignment='ffill'`` we expand the series to a full B-day calendar:

      * Pre-history (before the first observed price) is back-filled with the
        first observed price, so Δlogit = 0 there. Interpretation: "the market
        didn't exist yet, so it produced no information".
      * In-range gaps are forward-filled, so Δlogit = 0 on quiet days.
      * Days after the last observation are NOT extended — the market has
        resolved and any pretend-data would be a lie.
    """
    if alignment == "strict":
        return prices

    start_norm = (
        start_ts.tz_convert("UTC").normalize()
        if start_ts.tzinfo
        else start_ts.tz_localize("UTC").normalize()
    )
    end_norm = (
        end_ts.tz_convert("UTC").normalize()
        if end_ts.tzinfo
        else end_ts.tz_localize("UTC").normalize()
    )

    if prices.empty:
        return prices
    last_observed = prices.index.max()
    upper = min(end_norm, last_observed)
    if upper < start_norm:
        return prices  # nothing to do

    cal = pd.date_range(start_norm, upper, freq="B", tz="UTC")
    s = prices.reindex(cal)
    first = s.first_valid_index()
    if first is not None:
        s.loc[s.index < first] = s.loc[first]
    s = s.ffill()
    return s


def _cached_factor_history(
    fc: FactorConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    poly: PolymarketClient,
    cache: CacheBackend,
    settings: Settings,
) -> pd.DataFrame:
    """Pull a factor's daily price history. Dispatches by source.

    For Polymarket: ``fc.slug`` is the URL slug.
    For Kalshi:     ``fc.slug`` is the market ticker.
    For chain:      ``fc.segments`` carries the ordered (source, slug, end) list;
                    ``fc.slug`` is the stable label used in the cache key.
    """
    if fc.source == CHAIN_SOURCE:
        # Cache key includes the segment composition so two chains sharing an
        # ``id`` but different segments don't collide.
        cache_token = f"chain::{fc.slug}::{segments_signature(fc.segments)}"
    else:
        cache_token = fc.slug
    key = _cache_key(fc.source, cache_token, start.date().isoformat(), end.date().isoformat())
    blob = cache.get(key)
    if blob:
        df = pd.read_json(blob.decode("utf-8"), orient="split")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    if fc.source == "kalshi":
        kalshi_client = _current_kalshi_client()
        try:
            df = _main_attr("fetch_kalshi_history")(kalshi_client, fc.slug, start=start, end=end)
        except KalshiError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=504, detail=f"kalshi timeout/network: {_short_err(e)}"
            ) from e
    elif fc.source == CHAIN_SOURCE:
        kalshi_client = _current_kalshi_client()
        try:
            df = _main_attr("fetch_chained_history")(
                fc.segments,
                poly=poly,
                kalshi=kalshi_client,
                start=start,
                end=end,
            )
        except (PolymarketError, KalshiError) as e:
            raise HTTPException(status_code=502, detail=f"chain segment error: {e}") from e
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=504,
                detail=f"chain timeout/network: {_short_err(e)}",
            ) from e
    elif fc.source == "polymarket":
        try:
            df = _main_attr("fetch_factor_history")(poly, fc.slug, start=start, end=end)
        except PolymarketError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=504, detail=f"polymarket timeout/network: {_short_err(e)}"
            ) from e
    else:
        # Manifold / PredictIt / BLS / FRED — handled by the unified
        # dispatcher in pfm.factors. Errors are normalised to 502 so the
        # API surface stays consistent with the prediction-market paths.
        from pfm.factors import fetch_factor_history_dispatch

        try:
            df = fetch_factor_history_dispatch(fc, start=start, end=end)
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=504, detail=f"{fc.source} timeout/network: {_short_err(e)}"
            ) from e
        except Exception as e:  # surface every upstream error as 502
            raise HTTPException(
                status_code=502, detail=f"{fc.source} fetch error: {_short_err(e)}"
            ) from e

    if not df.empty:
        cache.set(key, df.to_json(orient="split").encode("utf-8"), settings.cache_ttl_seconds)
    return df


def _cached_log_returns(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    return_type: ReturnTypeLit,
    cache: CacheBackend,
    settings: Settings,
) -> pd.Series:
    key = _cache_key("eq", ticker, start.date().isoformat(), end.date().isoformat(), return_type)
    blob = cache.get(key)
    if blob:
        s = pd.read_json(blob.decode("utf-8"), orient="split", typ="series")
        s.index = pd.to_datetime(s.index, utc=True)
        s.name = "r"
        return s
    try:
        s = _main_attr("get_log_returns")(ticker, start, end, return_type=return_type)
    except EquityDataError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    cache.set(key, s.to_json(orient="split").encode("utf-8"), settings.cache_ttl_seconds)
    return s
