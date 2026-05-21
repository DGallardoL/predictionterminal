"""``/factors/*`` endpoints — catalog, discover, preview, rank, permutation, best.

Carved out of ``pfm.main`` so the monolith stops growing. The endpoints rely on
several private helpers that still live in ``pfm.main`` because they are
shared with other endpoint groups (see :func:`_main_helpers`). Those helpers
are pulled in via a lazy import inside each handler to avoid a circular
dependency at module load time.
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from pfm.analyses import (
    build_theme_composites,
    is_resolving_factor,
    oos_split,
    permutation_test,
    zscore_columns,
)
from pfm.cache import CacheBackend
from pfm.config import Settings, get_settings
from pfm.dependencies import (
    get_cache,
    get_factors_dep,
    get_polymarket_client,
)
from pfm.factors import FactorConfig
from pfm.model import (
    DEFAULT_EPSILON,
    delta_logit,
    fit_ols_hac,
)
from pfm.schemas import (
    BestModelRequest,
    BestModelResponse,
    DiscoveredMarket,
    DiscoverResponse,
    FactorList,
    FactorMetadata,
    PermutationRequest,
    PermutationResult,
    PreviewRequest,
    PreviewResponse,
    PriceBar,
    RankItem,
    RankRequest,
    RankResponse,
    StepwiseStep,
)
from pfm.sources.kalshi import KalshiClient, KalshiError
from pfm.sources.kalshi import fetch_factor_history as fetch_kalshi_history
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    discover_markets,
)

router = APIRouter(tags=["factors"])


def _main_helpers() -> tuple:
    """Lazy-load helpers that still live in ``pfm.main`` (avoids import cycle)."""
    from pfm import main as _m

    return (
        _m._cache_key,
        _m._resolve_factor_specs,
        _m._assemble_design,
        _m._cached_log_returns,
        _m._cached_factor_history,
        _m._align_factor_prices,
        _m._shift_to_stock_calendar,
        _m._residualize_against_spy,
        _m._short_err,
        _m._POLY_FANOUT_SEMAPHORE_SIZE,
    )


def _rank_item_error(fid: str, fc: FactorConfig, msg: str) -> RankItem:
    return RankItem(
        factor_id=fid,
        name=fc.name,
        slug=fc.slug,
        theme=fc.theme,
        n_obs=0,
        r_squared=0.0,
        beta=0.0,
        t_stat=0.0,
        p_value=1.0,
        error=msg,
    )


@router.get("/factors", response_model=FactorList)
def list_factors(
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    theme: Annotated[str | None, Query(max_length=80)] = None,
    source: Annotated[str | None, Query(max_length=40)] = None,
    search: Annotated[str | None, Query(max_length=120)] = None,
) -> FactorList:
    """Paginated factor catalog (default page=50, cap=500). See ``/factors/all`` for full dump."""
    items = list(factors.values())
    if theme:
        items = [f for f in items if f.theme == theme]
    if source:
        items = [f for f in items if f.source == source]
    if search:
        needle = search.strip().lower()
        if needle:
            items = [
                f
                for f in items
                if needle in f.id.lower()
                or needle in f.name.lower()
                or needle in f.slug.lower()
                or needle in (f.description or "").lower()
            ]
    total = len(items)
    page = items[offset : offset + limit]
    next_offset = offset + limit if offset + limit < total else None
    return FactorList(
        factors=[
            FactorMetadata(
                id=f.id,
                name=f.name,
                slug=f.slug,
                source=f.source,
                description=f.description,
                theme=f.theme,
            )
            for f in page
        ],
        total=total,
        limit=limit,
        offset=offset,
        next_offset=next_offset,
    )


@router.get("/factors/all", response_model=FactorList)
def list_factors_all(
    response: Response,
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
) -> FactorList:
    """Full factor dump (~500 KB at ~1 360 entries). Prefer ``/factors`` with pagination."""
    items = list(factors.values())
    total = len(items)
    response.headers["X-Factor-Count"] = str(total)
    response.headers["Warning"] = (
        f'199 - "factors/all returned {total} entries (~{total // 3} KB); '
        'prefer /factors?limit=50 for interactive use"'
    )
    return FactorList(
        factors=[
            FactorMetadata(
                id=f.id,
                name=f.name,
                slug=f.slug,
                source=f.source,
                description=f.description,
                theme=f.theme,
            )
            for f in items
        ],
        total=total,
        limit=total,
        offset=0,
        next_offset=None,
    )


@router.get("/factors/discover", response_model=DiscoverResponse)
def discover_factors(
    min_volume: Annotated[float, Query(ge=0)] = 1_000_000,
    limit: Annotated[int, Query(ge=1, le=100)] = 24,
    keyword: Annotated[str | None, Query(max_length=80)] = None,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> DiscoverResponse:
    """Surface high-volume active markets as candidate factors."""
    _cache_key, *_ = _main_helpers()
    key = _cache_key("discover", min_volume, limit, keyword or "")
    cached = cache.get(key)
    if cached:
        return DiscoverResponse.model_validate_json(cached)

    try:
        markets = discover_markets(poly, min_volume=min_volume, limit=limit, keyword=keyword)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e

    resp = DiscoverResponse(
        markets=[
            DiscoveredMarket(
                slug=m.slug,
                question=m.question,
                volume=m.volume,
                end_date=m.end_date,
                active=m.active,
                closed=m.closed,
            )
            for m in markets
        ]
    )
    cache.set(key, resp.model_dump_json().encode("utf-8"), settings.cache_ttl_seconds)
    return resp


def _auto_detect_source(slug: str, requested: str) -> str:
    """Override the caller's ``source`` when the slug shape gives it away.

    Front-end callers historically hardcode ``source="polymarket"`` for
    every preview, which 404s for Kalshi-shaped slugs (UPPERCASE with
    ``KX`` / ``-`` separators). Detect the obvious pattern and switch.
    Polymarket slugs are kebab-case lowercase, Kalshi tickers look like
    ``KXSOMETHING-26-FOO``.
    """
    # If the slug has KX prefix or is all-upper with hyphens, it's Kalshi.
    upper = slug.upper() == slug and any(c.isupper() for c in slug)
    looks_kalshi = slug.startswith(("KX", "INX", "DEMPRES", "GOVPARTY")) or upper
    if looks_kalshi and requested != "kalshi":
        return "kalshi"
    return requested


@router.post("/factors/preview", response_model=PreviewResponse)
def preview_factor(
    body: PreviewRequest,
    request: Request,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> PreviewResponse:
    """Look up a slug, return metadata + recent price history for the UI."""
    _cache_key, *_ = _main_helpers()
    # Auto-detect source from slug shape — UI callers hardcode
    # ``source="polymarket"`` and shouldn't have to know each factor's venue.
    effective_source = _auto_detect_source(body.slug, body.source)
    key = _cache_key("preview", effective_source, body.slug)
    cached = cache.get(key)
    if cached:
        return PreviewResponse.model_validate_json(cached)

    if effective_source == "kalshi":
        kalshi = getattr(request.app.state, "kalshi", None) or KalshiClient()
        try:
            market = kalshi.get_market(body.slug)
        except KalshiError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"kalshi error: {e}") from e

        try:
            df = fetch_kalshi_history(kalshi, body.slug)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"kalshi history error: {e}") from e

        question = market.title
        active = market.status in ("active", "open", None)
        closed = market.status in ("finalized", "settled", "closed")
        yes_token_id = body.slug

        if df.empty:
            resp = PreviewResponse(
                slug=body.slug,
                question=question,
                yes_token_id=yes_token_id,
                active=active,
                closed=closed,
                n_bars=0,
                history=[],
            )
        else:
            bars = [
                PriceBar(date=ts.date(), price=float(row["price"])) for ts, row in df.iterrows()
            ]
            resp = PreviewResponse(
                slug=body.slug,
                question=question,
                yes_token_id=yes_token_id,
                active=active,
                closed=closed,
                n_bars=len(bars),
                first_date=bars[0].date,
                last_date=bars[-1].date,
                current_price=bars[-1].price,
                history=bars,
            )
    else:
        try:
            meta = poly.get_market_metadata(body.slug)
        except PolymarketError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        try:
            df = poly.get_price_history(meta.yes_token_id)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"polymarket clob error: {e}") from e

        if df.empty:
            resp = PreviewResponse(
                slug=body.slug,
                question=meta.question,
                yes_token_id=meta.yes_token_id,
                active=meta.active,
                closed=meta.closed,
                n_bars=0,
                history=[],
            )
        else:
            bars = [
                PriceBar(date=row["date"].date(), price=float(row["price"]))
                for _, row in df.iterrows()
            ]
            resp = PreviewResponse(
                slug=body.slug,
                question=meta.question,
                yes_token_id=meta.yes_token_id,
                active=meta.active,
                closed=meta.closed,
                n_bars=len(bars),
                first_date=bars[0].date,
                last_date=bars[-1].date,
                current_price=bars[-1].price,
                history=bars,
            )
    cache.set(key, resp.model_dump_json().encode("utf-8"), settings.cache_ttl_seconds)
    return resp


# Single-flight semaphore for /factors/rank cold-path execution. A single
# cold sweep allocates ~150-200 MB for 1228 OLS fits; running 4+ in parallel
# can spike a 2-vCPU container into OOM (load-test agent observed jetsam
# kill). We let the first cold caller through, queue up to 1 more, and
# reject the rest with 429 + Retry-After so they fall back to the cache or
# the user retries.
_RANK_SWEEP_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("PFM_RANK_MAX_PARALLEL", "2")))
_RANK_SWEEP_TIMEOUT_S = float(os.environ.get("PFM_RANK_SWEEP_TIMEOUT_S", "90"))


@router.post("/factors/rank", response_model=RankResponse)
async def rank_factors(
    body: RankRequest,
    epsilon: Annotated[float, Query(gt=0.0, lt=0.5)] = DEFAULT_EPSILON,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> RankResponse:
    """Rank curated factors by single-factor R² for ``body.ticker``.

    Response cached in Redis L2 keyed by (ticker, start, end, top_k, pool,
    epsilon) for 10 minutes. With 1228 factors × OLS the cold path is
    30-90 s; cache hits return in <50 ms across all gunicorn workers.

    Concurrent cold sweeps are capped at ``PFM_RANK_MAX_PARALLEL`` (default
    2). Excess requests get 429 with Retry-After so a runaway batch of
    fresh tickers can't OOM the worker.
    """
    (
        _cache_key,
        _resolve_factor_specs,
        _assemble_design,
        _cached_log_returns,
        _cached_factor_history,
        _align_factor_prices,
        _shift_to_stock_calendar,
        _residualize_against_spy,
        _short_err,
        _POLY_FANOUT_SEMAPHORE_SIZE,
    ) = _main_helpers()
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    # Response cache — entire RankResponse is small JSON (<10 KB) and stable
    # for a given (ticker, window). Wraps the L1+L2 TERMINAL_CACHE so all 4
    # gunicorn workers share warm results without re-running the 1228-OLS
    # sweep.
    from pfm import terminal as _term_mod

    # Cache key derived from the request fields that materially affect the
    # ranking output. ``custom_factors`` are stringified so a request with
    # the same ad-hoc factor list hits the cache; a different list cuts a
    # fresh key.
    custom_sig = ",".join(sorted(cf.id for cf in body.custom_factors))
    rank_cache_key = (
        f"factors_rank::{body.ticker}::{body.start.isoformat()}"
        f"::{body.end.isoformat()}::{body.return_type}::{body.regression}"
        f"::{body.alignment}::{body.min_n_for_ranking}::{epsilon:.4f}"
        f"::custom={custom_sig}"
    )
    cached_resp = _term_mod.TERMINAL_CACHE.get(rank_cache_key)
    if cached_resp is not None:
        return RankResponse.model_validate(cached_resp)

    # Cold path — acquire the semaphore. ``acquire`` waits up to a short
    # timeout to avoid stampedes; over that we 429 with Retry-After so the
    # client can back off instead of stacking sweeps that would OOM the
    # worker. Warm cache hits above already short-circuited so legitimate
    # repeat users never reach this gate.
    try:
        await asyncio.wait_for(_RANK_SWEEP_SEMAPHORE.acquire(), timeout=2.0)
    except TimeoutError:
        raise HTTPException(
            status_code=429,
            detail=(
                "factor rank sweep is busy — another cold rank is in flight. "
                "Retry in a few seconds (warm cache hits are unaffected)."
            ),
            headers={"Retry-After": "30"},
        ) from None

    try:
        start_ts = pd.Timestamp(body.start, tz="UTC")
        end_ts = pd.Timestamp(body.end, tz="UTC")

        try:
            y_full = _cached_log_returns(
                body.ticker, start_ts, end_ts, body.return_type, cache, settings
            )
        except HTTPException as e:
            raise e

        candidates: list[tuple[str, FactorConfig]] = [(f.id, f) for f in factors.values()]
        for cf in body.custom_factors:
            candidates.append(
                (
                    cf.id,
                    FactorConfig(
                        id=cf.id,
                        name=cf.name or cf.id,
                        slug=cf.slug,
                        source="polymarket",
                        description="(custom)",
                        theme="custom",
                    ),
                )
            )

        sem = asyncio.Semaphore(_POLY_FANOUT_SEMAPHORE_SIZE)

        async def _fetch_one(fc: FactorConfig) -> pd.DataFrame | BaseException:
            async with sem:
                try:
                    return await asyncio.to_thread(
                        _cached_factor_history,
                        fc,
                        start_ts,
                        end_ts,
                        poly,
                        cache,
                        settings,
                    )
                except (PolymarketError, ValueError, HTTPException, httpx.HTTPError) as e:
                    return e

        fetched = await asyncio.gather(*(_fetch_one(fc) for _fid, fc in candidates))

        items: list[RankItem] = []
        for (fid, fc), result in zip(candidates, fetched, strict=True):
            if isinstance(result, BaseException):
                items.append(_rank_item_error(fid, fc, _short_err(result)))
                continue
            try:
                prices = result
                if prices.empty:
                    items.append(_rank_item_error(fid, fc, "no history"))
                    continue
                prices = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]
                aligned = _align_factor_prices(prices["price"], start_ts, end_ts, body.alignment)
                x = _shift_to_stock_calendar(
                    delta_logit(aligned, epsilon=epsilon).rename(fid).dropna(),
                    days=-1,
                )
                common = x.index.intersection(y_full.index)
                if len(common) < 10:
                    items.append(_rank_item_error(fid, fc, f"only {len(common)} overlapping obs"))
                    continue
                y = y_full.loc[common]
                X = pd.DataFrame({fid: x.loc[common]})
                fit = fit_ols_hac(y, X, regression=body.regression)
                est = fit.factors[0]
                items.append(
                    RankItem(
                        factor_id=fid,
                        name=fc.name,
                        slug=fc.slug,
                        theme=fc.theme,
                        n_obs=len(y),
                        r_squared=fit.stats.r_squared,
                        beta=est.beta,
                        t_stat=est.t_stat,
                        p_value=est.p_value,
                        sample_first_date=common.min().date(),
                        sample_last_date=common.max().date(),
                    )
                )
            except (PolymarketError, ValueError, HTTPException, httpx.HTTPError) as e:
                items.append(_rank_item_error(fid, fc, _short_err(e)))

        # Three-tier sort so tiny-sample R² doesn't out-rank reliable estimates:
        #   1) errors last
        #   2) within OK results, those with n_obs >= min_n_for_ranking come first
        #   3) ties broken by descending R², then by descending n_obs
        min_n = body.min_n_for_ranking

        def _sort_key(r: RankItem) -> tuple[int, int, float, int]:
            err = r.error is not None
            thin = (not err) and r.n_obs < min_n
            # Negative r_squared/n_obs to sort descending; first tier = error tier.
            return (
                1 if err else 0,
                1 if thin else 0,
                -(r.r_squared if not err else 0.0),
                -(r.n_obs if not err else 0),
            )

        items.sort(key=_sort_key)

        resp = RankResponse(
            ticker=body.ticker,
            start=body.start,
            end=body.end,
            return_type=body.return_type,
            regression=body.regression,
            items=items,
        )
        # Persist to Redis-backed L2 cache (10 min) so subsequent /factors/rank
        # for the same (ticker, window) skips the 30-90 s OLS sweep entirely.
        _term_mod.TERMINAL_CACHE.set(rank_cache_key, resp.model_dump(), 600)
        return resp
    finally:
        _RANK_SWEEP_SEMAPHORE.release()


@router.post("/factors/permutation", response_model=PermutationResult)
def factors_permutation(
    body: PermutationRequest,
    epsilon: Annotated[float, Query(gt=0.0, lt=0.5)] = DEFAULT_EPSILON,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> PermutationResult:
    """Standalone permutation test — shuffle factor values, refit, return p-value."""
    _, _resolve_factor_specs, _assemble_design, *_ = _main_helpers()
    factor_specs = _resolve_factor_specs(body.factors, body.custom_factors, factors)
    if not factor_specs:
        raise HTTPException(status_code=400, detail="provide at least one factor")
    y, X, _raw = _assemble_design(
        body.ticker,
        factor_specs,
        body.start,
        body.end,
        epsilon,
        body.return_type,
        poly,
        cache,
        settings,
        alignment=body.alignment,
        residualize_market=body.residualize_market,
    )
    if len(y) <= len(factor_specs) + 5:
        raise HTTPException(
            status_code=422,
            detail=f"too few overlapping observations ({len(y)}) for permutation test",
        )
    pr = permutation_test(
        y,
        X,
        n_iters=body.n_iters,
        seed=body.seed,
        test_fraction=body.test_fraction,
    )
    if pr["n_iters_completed"] == 0:
        raise HTTPException(
            status_code=422, detail="permutation runner produced no valid iterations"
        )
    return PermutationResult(**pr)  # type: ignore[arg-type]


@router.post("/factors/best", response_model=BestModelResponse)
def best_model(
    body: BestModelRequest,
    epsilon: Annotated[float, Query(gt=0.0, lt=0.5)] = DEFAULT_EPSILON,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> BestModelResponse:
    """Forward stepwise selection — greedily build a multi-factor model by R²adj or OOS-R²."""
    (
        _cache_key,
        _resolve_factor_specs,
        _assemble_design,
        _cached_log_returns,
        _cached_factor_history,
        _align_factor_prices,
        _shift_to_stock_calendar,
        _residualize_against_spy,
        _short_err,
        _POLY_FANOUT_SEMAPHORE_SIZE,
    ) = _main_helpers()
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")

    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")

    y_full = _cached_log_returns(body.ticker, start_ts, end_ts, body.return_type, cache, settings)
    if body.residualize_market:
        y_full = _residualize_against_spy(
            y_full, body.ticker, start_ts, end_ts, body.return_type, cache, settings
        )

    candidates: list[FactorConfig] = list(factors.values())
    for cf in body.custom_factors:
        candidates.append(
            FactorConfig(
                id=cf.id,
                name=cf.name or cf.id,
                slug=cf.slug,
                source="polymarket",
                description="(custom)",
                theme="custom",
            )
        )

    delta_by_id: dict[str, pd.Series] = {}
    rejected: list[str] = []
    for fc in candidates:
        try:
            prices = _cached_factor_history(
                fc,
                start_ts,
                end_ts,
                poly,
                cache,
                settings,
            )
        except (HTTPException, PolymarketError, httpx.HTTPError):
            rejected.append(fc.id)
            continue
        if prices.empty:
            rejected.append(fc.id)
            continue
        prices = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]
        if body.filter_resolving and is_resolving_factor(prices["price"]):
            rejected.append(fc.id)
            continue
        aligned = _align_factor_prices(prices["price"], start_ts, end_ts, body.alignment)
        s = _shift_to_stock_calendar(
            delta_logit(aligned, epsilon=epsilon).rename(fc.id).dropna(),
            days=-1,
        )
        if len(s.index.intersection(y_full.index)) < body.min_obs:
            rejected.append(fc.id)
            continue
        delta_by_id[fc.id] = s

    if body.zscore and delta_by_id:
        z_df = zscore_columns(pd.DataFrame(delta_by_id))
        delta_by_id = {col: z_df[col] for col in z_df.columns}

    theme_lookup_for_constraint: dict[str, str] = {}
    if body.theme_composites and delta_by_id:
        theme_lookup = {fc.id: fc.theme for fc in candidates if fc.id in delta_by_id}
        delta_by_id = build_theme_composites(delta_by_id, theme_lookup)
        theme_lookup_for_constraint = {col: col for col in delta_by_id}
    else:
        theme_lookup_for_constraint = {fc.id: fc.theme for fc in candidates if fc.id in delta_by_id}

    if not delta_by_id:
        raise HTTPException(
            status_code=422,
            detail=f"no factor has >= {body.min_obs} overlapping observations with {body.ticker}",
        )

    selected: list[str] = []
    log: list[StepwiseStep] = []
    last_r2_adj = -float("inf")
    final_r2 = 0.0
    final_r2_adj = 0.0
    final_n = 0

    for step_idx in range(body.max_factors):
        remaining = [fid for fid in delta_by_id if fid not in selected]
        if body.max_per_theme and theme_lookup_for_constraint:
            theme_counts: dict[str, int] = {}
            for s in selected:
                t = theme_lookup_for_constraint.get(s, "other")
                theme_counts[t] = theme_counts.get(t, 0) + 1
            remaining = [
                fid
                for fid in remaining
                if theme_counts.get(theme_lookup_for_constraint.get(fid, "other"), 0)
                < body.max_per_theme
            ]
        if not remaining:
            break

        best_fid: str | None = None
        best_score = -float("inf")
        best_r2 = 0.0
        best_r2_adj = 0.0
        best_n = 0
        considered = 0
        for fid in remaining:
            try_set = [*selected, fid]
            try:
                X = pd.concat([delta_by_id[k] for k in try_set], axis=1).dropna()
                common = X.index.intersection(y_full.index)
                if len(common) <= len(try_set) + 1 or len(common) < body.min_obs:
                    continue
                y_sub = y_full.loc[common]
                X_sub = X.loc[common]

                if body.criterion == "oos_r2":
                    n_train = max(int(len(common) * 0.8), len(try_set) + 5)
                    if n_train >= len(common) - 3:
                        continue
                    oos = oos_split(y_sub, X_sub, test_fraction=1 - n_train / len(common))
                    if oos is None:
                        continue
                    score = oos.test_r2
                    fit = fit_ols_hac(y_sub, X_sub, regression=body.regression)
                    r2_for_log = fit.stats.r_squared
                    r2_adj_for_log = fit.stats.r_squared_adj
                else:
                    fit = fit_ols_hac(y_sub, X_sub, regression=body.regression)
                    score = fit.stats.r_squared_adj
                    r2_for_log = fit.stats.r_squared
                    r2_adj_for_log = fit.stats.r_squared_adj
            except (ValueError, RuntimeError):
                continue
            considered += 1
            if score > best_score:
                best_score = score
                best_r2 = r2_for_log
                best_r2_adj = r2_adj_for_log
                best_fid = fid
                best_n = len(common)

        if best_fid is None:
            break
        if step_idx > 0 and best_score <= last_r2_adj:
            break

        selected.append(best_fid)
        log.append(
            StepwiseStep(
                step=step_idx + 1,
                added=best_fid,
                r_squared=best_r2,
                r_squared_adj=best_r2_adj,
                n_obs=best_n,
                candidates_considered=considered,
            )
        )
        last_r2_adj = best_score
        final_r2 = best_r2
        final_r2_adj = best_r2_adj
        final_n = best_n

    return BestModelResponse(
        ticker=body.ticker,
        start=body.start,
        end=body.end,
        selected=selected,
        final_r_squared=final_r2,
        final_r_squared_adj=final_r2_adj,
        final_n_obs=final_n,
        log=log,
        rejected=rejected,
    )
