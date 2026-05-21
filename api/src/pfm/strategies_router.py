"""``/strategies/*`` — 32 endpoints for classical / stat-arb / risk strategies.

Extracted from ``pfm.main`` to keep the monolith bounded. Five private helpers
still live in ``pfm.main`` (``_cached_factor_history``, ``_fetch_aligned_prob``,
``_finite``, ``_resolve_one``, ``_short_err``); they are bound into this
module's globals by :func:`bind_main_helpers`, which ``pfm.main`` invokes once
at start-up after the helpers are defined. This avoids a circular import at
module-load time while keeping the endpoint bodies readable (bare names, no
``_m._foo`` prefixing).
"""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from pfm.advanced import (
    bootstrap_sharpe_ci,
    cusum_test,
    permutation_sharpe_test,
    walk_forward_backtest,
)
from pfm.advanced_strategies import (
    almgren_chriss_schedule,
    hasbrouck_information_share,
    markov_regime_switching,
)
from pfm.basket import basket_pca_residuals
from pfm.cache import CacheBackend
from pfm.cointegration import engle_granger, spread_zscore
from pfm.config import Settings, get_settings
from pfm.dependencies import (
    get_cache,
    get_factors_dep,
    get_polymarket_client,
)
from pfm.dfa import dfa as dfa_fn
from pfm.distance_method import distance_method
from pfm.event_model import event_model
from pfm.factor_model_pro import fit_factor_model_pro
from pfm.factors import FactorConfig
from pfm.fractional_diff import find_minimal_d
from pfm.garch import fit_garch_11
from pfm.granger import granger_test as granger_causality
from pfm.kalman import kalman_dynamic_hedge
from pfm.mean_reversion import hurst_exponent, variance_ratio_test
from pfm.ml_predictor import fit_ml_predictor
from pfm.ou import bertram_optimal_bands, fit_ou
from pfm.pairs import pairs_backtest
from pfm.patterns import (
    cluster_pairs_by_signature,
    correlate_pair_pnls,
    day_of_week_effect,
    pre_resolution_regime,
)
from pfm.portfolio import vol_targeted_combiner
from pfm.robust_validation import run_robust_validation
from pfm.scanner import run_scan
from pfm.schemas import (
    AlmgrenChrissRequest,
    AlmgrenChrissResponse,
    AutoBacktestRequest,
    AutoBacktestResponse,
    AutoBacktestRow,
    BasketResidualPoint,
    BasketStatArbRequest,
    BasketStatArbResponse,
    BoundsPoint,
    BoundsRequest,
    BoundsResponse,
    ClusterOut,
    CoefficientProOut,
    CointegrationRequest,
    CointegrationResponse,
    CointegrationSpreadPoint,
    ConditionalRequest,
    ConditionalResponse,
    CorrelationOut,
    CostSensitivityPoint,
    CusumPoint,
    CusumRequest,
    CusumResponse,
    DfaRequest,
    DfaResponse,
    DistanceMethodRequest,
    DistanceMethodResponse,
    DowOut,
    EventCoefficientOut,
    EventModelRequest,
    EventModelResponse,
    EventModelSeriesPoint,
    FactorModelProRequest,
    FactorModelProResponse,
    FeatureImportanceOut,
    FractionalDiffGridPoint,
    FractionalDiffRequest,
    FractionalDiffResponse,
    FredCointegrationRequest,
    FredCointegrationResponse,
    GarchRequest,
    GarchResponse,
    GrangerLagOut,
    GrangerRequest,
    GrangerResponse,
    ImplicationGapPoint,
    ImplicationRequest,
    ImplicationResponse,
    InfoShareRequest,
    InfoShareResponse,
    KalmanHedgePoint,
    KalmanHedgeRequest,
    KalmanHedgeResponse,
    MeanReversionRequest,
    MeanReversionResponse,
    MlFoldOut,
    MlPredictorRequest,
    MlPredictorResponse,
    OuBandsRequest,
    OuBandsResponse,
    PairsBacktestRequest,
    PairsBacktestResponse,
    PairsEquityPoint,
    PairsTradeRecord,
    PatternsRequest,
    PatternsResponse,
    PermutationSharpeRequest,
    PermutationSharpeResponse,
    PortfolioRequest,
    PortfolioResponse,
    PreResolutionOut,
    PresetsResponse,
    RegimePoint,
    RegimeSwitchingRequest,
    RegimeSwitchingResponse,
    ResidualDiagnosticsOut,
    RobustValidationRequest,
    RobustValidationResponse,
    SharpeBootstrapRequest,
    SharpeBootstrapResponse,
    SpotVsImpliedRequest,
    SpotVsImpliedResponse,
    StrategyPreset,
    TripleBarrierRequest,
    TripleBarrierResponse,
    WalkForwardFoldOut,
    WalkForwardRequest,
    WalkForwardResponse,
)
from pfm.signals import bollinger_signals
from pfm.sources.binance import BinanceClient, BinanceError, annualisation_for_interval
from pfm.sources.fred import FredDataError, fetch_fred_series
from pfm.sources.polymarket import PolymarketClient
from pfm.spot_implied import spot_vs_implied
from pfm.strategies import (
    conditional_regression,
    frechet_bounds,
    implication_test,
)
from pfm.triple_barrier import triple_barrier_backtest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["strategies"])


# ---------------------------------------------------------------------------
# Helpers that still live in ``pfm.main`` are bound here at startup. Declared
# as ``None`` so static analysis sees the names; ``bind_main_helpers`` (called
# from ``pfm.main`` after its helpers are defined) rebinds them to the real
# functions before the first request lands.
# ---------------------------------------------------------------------------

_cached_factor_history = None
_fetch_aligned_prob = None
_finite = None
_resolve_one = None
_short_err = None


def bind_main_helpers() -> None:
    """Resolve the five ``pfm.main``-resident helpers used by these endpoints.

    Idempotent: callers may invoke this more than once (e.g. in tests that
    re-import the app). Called once during start-up from ``pfm.main``.
    """
    global _cached_factor_history, _fetch_aligned_prob, _finite, _resolve_one, _short_err
    from pfm import main as _m

    _cached_factor_history = _m._cached_factor_history
    _fetch_aligned_prob = _m._fetch_aligned_prob
    _finite = _m._finite
    _resolve_one = _m._resolve_one
    _short_err = _m._short_err


@router.post("/strategies/implication", response_model=ImplicationResponse)
def strategies_implication(
    body: ImplicationRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> ImplicationResponse:
    """Test the logical-implication invariant ``A ⇒ B`` ⇒ ``P(A) ≤ P(B)``.

    Returns per-date gaps and a verdict bucketing the count of violation dates.
    Use it on logically-related markets (e.g. "BTC ≥ $150k" ⇒ "BTC ≥ $100k").
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.antecedent_id, factors, role="antecedent")
    fb = _resolve_one(body.consequent_id, factors, role="consequent")

    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)

    res = implication_test(p_a, p_b, tolerance=body.tolerance)

    # Re-align so the series we return matches what the test saw.
    paligned = pd.concat({"a": p_a, "b": p_b}, axis=1).dropna()
    points = [
        ImplicationGapPoint(
            date=ts.date(),
            p_a=float(paligned.loc[ts, "a"]),
            p_b=float(paligned.loc[ts, "b"]),
            gap=float(res.gap_series.loc[ts]) if ts in res.gap_series.index else 0.0,
            logit_gap=float(res.logit_gap_series.loc[ts])
            if ts in res.logit_gap_series.index
            else 0.0,
        )
        for ts in paligned.index
    ]
    return ImplicationResponse(
        antecedent_id=fa.id,
        consequent_id=fb.id,
        n_obs=res.n_obs,
        verdict=res.verdict,
        n_violations=len(res.violation_dates),
        violation_dates=res.violation_dates,
        max_gap=_finite(res.max_gap),
        mean_gap=_finite(res.mean_gap),
        series=points,
    )


@router.post("/strategies/conditional", response_model=ConditionalResponse)
def strategies_conditional(
    body: ConditionalRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> ConditionalResponse:
    """HAC-OLS regression of P_A on P_B (β interpretable as conditional sensitivity)."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)

    try:
        res = conditional_regression(p_a, p_b, hac_lag=body.hac_lag)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return ConditionalResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=res.n_obs,
        beta=res.beta,
        beta_hac_se=res.beta_hac_se,
        beta_ci_lo=res.beta_ci_lo,
        beta_ci_hi=res.beta_ci_hi,
        intercept=res.intercept,
        r_squared=res.r_squared,
        cond_mean_when_b_high=_finite(res.cond_mean_when_b_high),
        cond_mean_when_b_low=_finite(res.cond_mean_when_b_low),
        n_b_high=res.n_b_high,
    )


@router.post("/strategies/bounds", response_model=BoundsResponse)
def strategies_bounds(
    body: BoundsRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> BoundsResponse:
    """Per-date Fréchet-Hoeffding bounds on the joint ``P(A ∩ B)``."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    res = frechet_bounds(p_a, p_b)

    paligned = pd.concat({"a": p_a, "b": p_b}, axis=1).dropna()
    points = [
        BoundsPoint(
            date=ts.date(),
            p_a=float(paligned.loc[ts, "a"]),
            p_b=float(paligned.loc[ts, "b"]),
            lower=float(res.lower.loc[ts]),
            upper=float(res.upper.loc[ts]),
            independence=float(res.independence_joint.loc[ts]),
        )
        for ts in paligned.index
    ]
    return BoundsResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=res.n_obs,
        mean_lower=_finite(res.mean_lower),
        mean_upper=_finite(res.mean_upper),
        mean_width=_finite(res.mean_width),
        series=points,
    )


@router.post("/strategies/spot-vs-implied", response_model=SpotVsImpliedResponse)
def strategies_spot_vs_implied(
    body: SpotVsImpliedRequest,
    request: Request,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SpotVsImpliedResponse:
    """Compare a live underlying (Binance daily klines) to a market-implied
    YES-price for a price-target binary outcome.

    Pulls a trailing window of daily OHLCV from Binance, estimates σ̂ via
    Yang-Zhang OHLC, and computes a closed-form GBM probability for the
    geometry chosen by the caller (terminal / one-touch up / one-touch down).
    A block-bootstrap CI is returned alongside an "edge" against the
    market price (when supplied).

    See ``docs/strategies.md`` §6 for derivations and the
    resolution-source-basis caveat (Polymarket BTC markets typically
    settle on Coinbase / UMA, not Binance).
    """
    binance: BinanceClient = getattr(request.app.state, "binance", None) or BinanceClient()
    now = pd.Timestamp.now(tz="UTC")
    today = now.normalize()
    if body.expiry < today.date():
        raise HTTPException(
            status_code=400,
            detail=f"expiry {body.expiry.isoformat()} is in the past",
        )

    interval = body.interval
    annualisation = annualisation_for_interval(interval)
    sub_daily = annualisation > 1000  # >1000 bars/yr ⇒ sub-daily

    if sub_daily:
        # For sub-daily we just take the last `vol_window_bars` bars; the
        # Binance endpoint accepts up to 1000 bars per call.
        try:
            ohlc = binance.get_klines(
                body.symbol,
                interval=interval,
                limit=min(body.vol_window_bars, 1000),
            )
        except BinanceError as e:
            raise HTTPException(status_code=502, detail=f"binance: {e}") from e
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=504, detail=f"binance timeout/network: {_short_err(e)}"
            ) from e
    else:
        # Daily and coarser: use a date window with a 5-bar buffer.
        fetch_end = today
        fetch_start = today - pd.Timedelta(days=body.vol_window_bars + 5)
        try:
            ohlc = binance.get_klines(
                body.symbol,
                interval=interval,
                start=fetch_start,
                end=fetch_end,
                limit=1000,
            )
        except BinanceError as e:
            raise HTTPException(status_code=502, detail=f"binance: {e}") from e
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=504, detail=f"binance timeout/network: {_short_err(e)}"
            ) from e

    if len(ohlc) > body.vol_window_bars:
        ohlc = ohlc.tail(body.vol_window_bars)
    if len(ohlc) < 10:
        raise HTTPException(
            status_code=502,
            detail=f"binance returned only {len(ohlc)} {interval} bars for {body.symbol!r}",
        )

    try:
        res = spot_vs_implied(
            ohlc,
            strike=body.strike,
            expiry=body.expiry,
            geometry=body.geometry,
            market_prob=body.market_prob,
            drift_annual=body.drift_annual,
            annualisation=annualisation,
            n_bootstrap=body.n_bootstrap,
            seed=body.seed,
            asof_ts=now if sub_daily else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return SpotVsImpliedResponse(
        symbol=body.symbol.upper(),
        interval=interval,
        spot=res.spot,
        strike=res.strike,
        expiry=body.expiry,
        geometry=res.geometry,
        time_years=res.time_years,
        sigma_used=res.sigma_used,
        sigma_method="yang_zhang",
        drift_used=res.drift_used,
        n_vol_bars=len(ohlc),
        model_prob=res.model_prob,
        ci_lo_90=res.ci_lo_90,
        ci_hi_90=res.ci_hi_90,
        ci_lo_95=res.ci_lo_95,
        ci_hi_95=res.ci_hi_95,
        market_prob=res.market_prob,
        edge=res.edge,
        edge_significant_95=res.edge_significant_95,
        n_bootstrap=res.n_bootstrap,
    )


@router.post("/strategies/cointegration", response_model=CointegrationResponse)
def strategies_cointegration(
    body: CointegrationRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> CointegrationResponse:
    """Engle-Granger 2-step cointegration test on a probability pair.

    Returns the OLS hedge ratio, the ADF p-value on the residuals, and the
    AR(1)-derived half-life. A short half-life with rejected ADF is the
    signal a pairs trader looks for.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    if body.a_id == body.b_id:
        # Cointegration of a series with itself is meaningless and we'd
        # otherwise fire two identical upstream fetches (Kalshi rate-limits
        # the dupe and the user sees a confusing 404).
        raise HTTPException(
            status_code=400,
            detail=(
                f"a_id and b_id must differ — got both = {body.a_id!r}. "
                "Pass two distinct factor IDs (or alias as factor_a/factor_b)."
            ),
        )
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)

    res = engle_granger(p_a, p_b, significance=body.significance)

    aligned = pd.concat({"a": p_a, "b": p_b}, axis=1).dropna()
    z = spread_zscore(res.spread, window=20) if len(res.spread) >= 25 else pd.Series(dtype=float)
    points = []
    for ts in aligned.index:
        spread_val = float(res.spread.loc[ts]) if ts in res.spread.index else float("nan")
        z_val = None
        if ts in z.index and not pd.isna(z.loc[ts]):
            z_val = float(z.loc[ts])
        points.append(
            CointegrationSpreadPoint(
                date=ts.date(),
                p_a=float(aligned.loc[ts, "a"]),
                p_b=float(aligned.loc[ts, "b"]),
                spread=spread_val,
                zscore=z_val,
            )
        )
    return CointegrationResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=res.n_obs,
        cointegrated=res.cointegrated,
        verdict=res.verdict,
        beta_hedge=_finite(res.beta_hedge),
        intercept=_finite(res.intercept),
        adf_stat=_finite(res.adf_stat),
        adf_pvalue=_finite(res.adf_pvalue),
        adf_used_lag=res.adf_used_lag,
        half_life_days=res.half_life_days,
        rho=_finite(res.rho) if res.rho is not None else None,
        series=points,
    )


@router.post(
    "/strategies/fred-cointegration",
    response_model=FredCointegrationResponse,
)
def strategies_fred_cointegration(
    body: FredCointegrationRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> FredCointegrationResponse:
    """Engle-Granger cointegration test between a factor and a FRED macro series.

    Pulls the factor's daily probability from Polymarket (via the chained
    cache) and the FRED series via the auth-free ``fredgraph.csv`` endpoint,
    aligns them on the UTC daily calendar, runs Engle-Granger 2-step, and
    returns the OLS hedge ratio, ADF p-value, and the AR(1)-derived
    half-life.

    Args:
        body: validated request payload (factor id, FRED series id, window,
            optional transform applied to the FRED series before the test).

    Returns:
        :class:`FredCointegrationResponse` with the cointegration verdict
        and a summary of the FRED series window (first/last/min/max).

    Raises:
        HTTPException: 400 if ``start >= end`` or the factor id is unknown,
            502 if FRED returns an error, 422 on transform errors.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fc = _resolve_one(body.factor_id, factors, role="factor")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")

    p_factor = _fetch_aligned_prob(fc, start_ts, end_ts, poly, cache, settings)

    try:
        s_fred = fetch_fred_series(
            body.fred_series,
            start_ts,
            end_ts,
            transform=body.transform,
        )
    except FredDataError as e:
        # transform-validation errors (e.g. logit on unbounded) should be 422;
        # network/parse errors are upstream-server failures (502).
        msg = str(e)
        if "transform" in msg or "unknown transform" in msg:
            raise HTTPException(status_code=422, detail=msg) from e
        raise HTTPException(status_code=502, detail=msg) from e

    res = engle_granger(p_factor, s_fred)

    fred_clean = s_fred.dropna()
    if fred_clean.empty:
        fred_first = fred_last = fred_min = fred_max = None
    else:
        fred_first = float(fred_clean.iloc[0])
        fred_last = float(fred_clean.iloc[-1])
        fred_min = float(fred_clean.min())
        fred_max = float(fred_clean.max())

    return FredCointegrationResponse(
        factor_id=fc.id,
        fred_series=body.fred_series,
        n_obs=res.n_obs,
        adf_pvalue=_finite(res.adf_pvalue),
        beta_hedge=_finite(res.beta_hedge),
        half_life_days=res.half_life_days,
        cointegrated=res.cointegrated,
        verdict=res.verdict,
        fred_first=fred_first,
        fred_last=fred_last,
        fred_min=fred_min,
        fred_max=fred_max,
    )


@router.post("/strategies/pairs-backtest", response_model=PairsBacktestResponse)
def strategies_pairs_backtest(
    body: PairsBacktestRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> PairsBacktestResponse:
    """Walk-forward z-score pairs trade on the spread of two probability series.

    First runs Engle-Granger to verify cointegration (informational only —
    the backtest still runs even on non-cointegrated pairs, but the user
    sees `cointegration_passed: false` and should weight the Sharpe down).
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)

    cint = engle_granger(p_a, p_b)
    if cint.spread.empty:
        raise HTTPException(status_code=422, detail=cint.verdict)

    try:
        bt = pairs_backtest(
            cint.spread,
            window=body.window,
            entry_z=body.entry_z,
            exit_z=body.exit_z,
            stop_z=body.stop_z,
            annualisation_factor=body.annualisation,
            oos_fraction=body.oos_fraction,
            max_hold_bars=body.max_hold_bars,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    series = []
    for ts in bt.spread.index:
        z_val = bt.zscores.loc[ts] if ts in bt.zscores.index else None
        series.append(
            PairsEquityPoint(
                date=ts.date(),
                spread=float(bt.spread.loc[ts]),
                zscore=None if z_val is None or pd.isna(z_val) else float(z_val),
                position=int(bt.positions.loc[ts]) if ts in bt.positions.index else 0,
                pnl=float(bt.pnl.loc[ts]) if ts in bt.pnl.index else 0.0,
                equity=float(bt.equity_curve.loc[ts]) if ts in bt.equity_curve.index else 0.0,
            )
        )
    trades_out = [
        PairsTradeRecord(
            entry_date=t.entry_date.date(),
            exit_date=t.exit_date.date(),
            direction=t.direction,
            entry_z=t.entry_z,
            exit_z=t.exit_z,
            pnl=t.pnl,
            holding_days=t.holding_days,
            exit_reason=t.exit_reason,
        )
        for t in bt.trades
    ]
    return PairsBacktestResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=bt.n_obs,
        n_trades=bt.n_trades,
        sharpe=bt.sharpe,
        sortino=bt.sortino,
        calmar=bt.calmar,
        hit_rate=bt.hit_rate,
        max_drawdown=bt.max_drawdown,
        var_95=bt.var_95,
        cvar_95=bt.cvar_95,
        skew=bt.skew,
        kurtosis=bt.kurtosis,
        mean_holding_days=bt.mean_holding_days,
        sharpe_is=bt.sharpe_is,
        sharpe_oos=bt.sharpe_oos,
        oos_to_is_ratio=bt.oos_to_is_ratio,
        n_obs_is=bt.n_obs_is,
        n_obs_oos=bt.n_obs_oos,
        cointegration_passed=cint.cointegrated,
        cointegration_pvalue=_finite(cint.adf_pvalue),
        half_life_days=cint.half_life_days,
        beta_hedge=_finite(cint.beta_hedge),
        series=series,
        trades=trades_out,
    )


@router.post("/strategies/event-model", response_model=EventModelResponse)
def strategies_event_model(
    body: EventModelRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> EventModelResponse:
    """HAC-OLS regression of one event probability on N other events."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    if body.target_id in body.factor_ids:
        raise HTTPException(
            status_code=400,
            detail=f"target_id {body.target_id!r} cannot also appear in factor_ids",
        )
    target_fc = _resolve_one(body.target_id, factors, role="target")
    factor_fcs = [_resolve_one(fid, factors, role="factor") for fid in body.factor_ids]
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    target = _fetch_aligned_prob(target_fc, start_ts, end_ts, poly, cache, settings)
    fcols = {
        fc.id: _fetch_aligned_prob(fc, start_ts, end_ts, poly, cache, settings) for fc in factor_fcs
    }
    X = pd.DataFrame(fcols)
    try:
        res = event_model(target, X, target_id=target_fc.id, hac_lag=body.hac_lag)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    coeffs_out = [
        EventCoefficientOut(
            factor_id=c.factor_id,
            beta=c.beta,
            hac_se=c.hac_se,
            t_stat=c.t_stat,
            p_value=c.p_value,
            ci_lo=c.ci_lo,
            ci_hi=c.ci_hi,
            vif=_finite(c.vif),
        )
        for c in res.coefficients
    ]
    series = [
        EventModelSeriesPoint(
            date=ts.date(),
            actual=float(res.actual.loc[ts]),
            predicted=float(res.predicted.loc[ts]),
            residual=float(res.residuals.loc[ts]),
        )
        for ts in res.predicted.index
    ]
    return EventModelResponse(
        target_id=res.target_id,
        factor_ids=res.factor_ids,
        n_obs=res.n_obs,
        intercept=res.intercept,
        intercept_se=res.intercept_se,
        coefficients=coeffs_out,
        r_squared=res.r_squared,
        r_squared_adj=res.r_squared_adj,
        f_statistic=_finite(res.f_statistic),
        f_pvalue=_finite(res.f_pvalue),
        condition_number=_finite(res.condition_number),
        hac_lag=res.hac_lag,
        series=series,
    )


@router.post("/strategies/basket-stat-arb", response_model=BasketStatArbResponse)
def strategies_basket_stat_arb(
    body: BasketStatArbRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> BasketStatArbResponse:
    """PCA-residual statistical arbitrage on a basket of related events.

    Stack the chosen factor probability series into a matrix; subtract the
    top-k principal components; the residuals are the *idiosyncratic*
    inefficiency signals that pairs/basket stat-arb trades on. Each
    residual gets a rolling z-score the user can use as an entry signal.
    Includes a per-market Kelly-fraction estimate.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fcs = [_resolve_one(fid, factors, role="factor") for fid in body.factor_ids]
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    cols = {fc.id: _fetch_aligned_prob(fc, start_ts, end_ts, poly, cache, settings) for fc in fcs}
    df = pd.DataFrame(cols)
    try:
        res = basket_pca_residuals(
            df,
            n_components=body.n_components,
            explained_variance_target=body.explained_variance_target,
            z_window=body.z_window,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    series = []
    for ts in res.residuals.index:
        row_resid = {fid: float(res.residuals.loc[ts, fid]) for fid in res.factor_ids}
        row_z: dict[str, float | None] = {}
        for fid in res.factor_ids:
            z = res.z_residuals.loc[ts, fid] if ts in res.z_residuals.index else None
            row_z[fid] = None if z is None or pd.isna(z) else float(z)
        series.append(BasketResidualPoint(date=ts.date(), residuals=row_resid, z_residuals=row_z))

    return BasketStatArbResponse(
        factor_ids=res.factor_ids,
        n_obs=res.n_obs,
        n_components_used=res.n_components_used,
        explained_variance_ratio=res.explained_variance_ratio,
        loadings=res.loadings,
        kelly_fraction_per_market=res.kelly_fraction_per_market,
        series=series,
    )


@router.post("/strategies/ou-bands", response_model=OuBandsResponse)
def strategies_ou_bands(
    body: OuBandsRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> OuBandsResponse:
    """Calibrate OU dynamics on the cointegration spread + Bertram (2010)
    optimal entry/exit z-bands.

    Reports κ (mean-reversion speed), μ (long-run mean), σ_eq (equilibrium std),
    half-life, and the analytically-optimal ``z_entry`` that maximises
    expected PnL per unit time net of round-trip transaction cost.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)

    cint = engle_granger(p_a, p_b)
    if cint.spread.empty:
        raise HTTPException(status_code=422, detail=cint.verdict)

    # Build a (date, spread, z) time series so the UI can plot the OU bands
    # over the actual spread (the panel was previously chart-less).
    spread_ser = cint.spread.dropna()
    if len(spread_ser) >= 2:
        mu_emp = float(spread_ser.mean())
        sd_emp = float(spread_ser.std(ddof=1)) or 1.0
        z_ser = (spread_ser - mu_emp) / sd_emp
        series_points = [
            {
                "date": ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts),
                "spread": float(spread_ser.iloc[i]),
                "z_score": float(z_ser.iloc[i]),
            }
            for i, ts in enumerate(spread_ser.index)
        ]
    else:
        series_points = None

    try:
        ou = fit_ou(cint.spread)
    except ValueError:
        # Spread isn't stationary OU — return cointegration-level info only.
        return OuBandsResponse(
            a_id=fa.id,
            b_id=fb.id,
            n_obs=cint.n_obs,
            cointegrated=cint.cointegrated,
            transaction_cost_sigma=body.transaction_cost_sigma,
            series=series_points,
        )

    bands = bertram_optimal_bands(ou, transaction_cost=body.transaction_cost_sigma)

    return OuBandsResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=cint.n_obs,
        cointegrated=cint.cointegrated,
        kappa=ou.kappa,
        mu=ou.mu,
        sigma_eq=ou.sigma_eq,
        sigma_innov=ou.sigma_innov,
        half_life_bars=ou.half_life_bars,
        ar1_beta=ou.ar1_beta,
        z_entry_optimal=bands["z_entry"],
        z_exit_optimal=bands["z_exit"],
        expected_pnl_per_cycle_sigma=bands["expected_pnl_per_cycle_sigma"],
        expected_cycle_bars=bands["expected_cycle_bars"],
        expected_pnl_per_year_sigma=bands["expected_pnl_per_year_sigma"],
        transaction_cost_sigma=body.transaction_cost_sigma,
        series=series_points,
    )


@router.post("/strategies/granger", response_model=GrangerResponse)
def strategies_granger(
    body: GrangerRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> GrangerResponse:
    """Bivariate Granger causality between two event probability series.

    For each lag in 1..max_lag, tests whether past values of one series
    help predict the other. Used to identify the *leader* in event-pair
    co-movement (long the follower, hedge with the leader).
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    try:
        res = granger_causality(
            p_a, p_b, a_id=fa.id, b_id=fb.id, max_lag=body.max_lag, alpha=body.alpha
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    def _to_out(rows) -> list[GrangerLagOut]:
        return [
            GrangerLagOut(
                lag=r.lag,
                ssr_f_stat=r.ssr_f_stat,
                ssr_f_pvalue=r.ssr_f_pvalue,
                ssr_chi2_pvalue=r.ssr_chi2_pvalue,
            )
            for r in rows
        ]

    return GrangerResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=res.n_obs,
        direction=res.direction,
        best_lag_b_to_a=res.best_lag_b_to_a,
        best_pvalue_b_to_a=_finite(res.best_pvalue_b_to_a)
        if res.best_pvalue_b_to_a is not None
        else None,
        best_lag_a_to_b=res.best_lag_a_to_b,
        best_pvalue_a_to_b=_finite(res.best_pvalue_a_to_b)
        if res.best_pvalue_a_to_b is not None
        else None,
        lags_b_to_a=_to_out(res.lags),
        lags_a_to_b=_to_out(res.lags_reverse),
    )


@router.post("/strategies/kalman-hedge", response_model=KalmanHedgeResponse)
def strategies_kalman_hedge(
    body: KalmanHedgeRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> KalmanHedgeResponse:
    """Time-varying hedge ratio β_t via Kalman filter.

    Returns the per-bar β̂_t and the *innovation* spread e_t = y_t − β_t·x_t,
    which can be fed directly into the pairs-trading backtester for an
    adaptive-hedge strategy. Smaller δ ⇒ slower, smoother β̂_t; larger
    δ ⇒ more responsive (and noisier).
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    try:
        out = kalman_dynamic_hedge(p_a, p_b, delta=body.delta)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    aligned = pd.concat({"a": p_a, "b": p_b}, axis=1).dropna()
    points = [
        KalmanHedgePoint(
            date=ts.date(),
            p_a=float(aligned.loc[ts, "a"]),
            p_b=float(aligned.loc[ts, "b"]),
            beta_t=float(out.beta.loc[ts]),
            spread=float(out.spread.loc[ts]),
        )
        for ts in out.beta.index
    ]
    return KalmanHedgeResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=out.n_obs,
        delta=out.delta,
        r=out.r,
        q=out.q,
        log_likelihood=out.log_likelihood,
        beta_init=out.beta_init,
        beta_final=out.beta_final,
        beta_min=float(out.beta.min()),
        beta_max=float(out.beta.max()),
        spread_std=float(out.spread.std(ddof=1)) if out.n_obs > 1 else 0.0,
        series=points,
    )


@router.post("/strategies/mean-reversion", response_model=MeanReversionResponse)
def strategies_mean_reversion(
    body: MeanReversionRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> MeanReversionResponse:
    """Hurst exponent (R/S) + Lo-MacKinlay variance-ratio test on a single
    factor's probability series. Both quantify mean-reversion strength
    model-free."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fc = _resolve_one(body.factor_id, factors, role="factor")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p = _fetch_aligned_prob(fc, start_ts, end_ts, poly, cache, settings)

    h = hurst_exponent(p)
    vr = variance_ratio_test(p, q=body.vr_q)
    return MeanReversionResponse(
        factor_id=fc.id,
        n_obs=h.n_obs,
        hurst=_finite(h.H),
        hurst_r_squared=_finite(h.r_squared),
        hurst_interpretation=h.interpretation,
        vr_q=vr.q,
        vr=_finite(vr.vr),
        vr_z_stat=_finite(vr.z_stat),
        vr_p_value=_finite(vr.p_value),
        vr_verdict=vr.verdict,
        log_n=h.log_n,
        log_rs=h.log_rs,
    )


@router.post("/strategies/auto-backtest", response_model=AutoBacktestResponse)
def strategies_auto_backtest(
    body: AutoBacktestRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> AutoBacktestResponse:
    """Auto-pipeline: scan catalog for cointegrated pairs, backtest each,
    rank the leaderboard by Sharpe.

    The single "press here for alpha" button. Restrict by ``theme`` for
    speed; ``max_to_backtest`` caps how many of the top scanner hits get
    the full pairs-trading backtest treatment.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")

    def _fetch(fc: FactorConfig) -> pd.Series:
        df = _cached_factor_history(fc, start_ts, end_ts, poly, cache, settings)
        if df.empty:
            return pd.Series(dtype=float)
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        return df["price"].rename(fc.id)

    t0 = pd.Timestamp.now()
    scan_report = run_scan(
        factors,
        fetch_prices=_fetch,
        mode="cointegration",
        theme=body.theme,
        factor_ids=body.factor_ids,
        max_pairs=body.max_pairs,
        coint_adf_max_p=body.coint_adf_max_p,
        coint_half_life_max=body.coint_half_life_max,
        top_k_per_track=body.max_to_backtest,
    )
    n_coint = len(scan_report.cointegration)
    n_factors = scan_report.n_factors_scanned
    if n_coint == 0:
        return AutoBacktestResponse(
            n_factors_scanned=n_factors,
            n_coint_hits=0,
            n_backtested=0,
            runtime_seconds=scan_report.runtime_seconds,
            leaderboard=[],
        )

    leaderboard: list[AutoBacktestRow] = []
    for hit in scan_report.cointegration:
        fa = factors.get(hit.a_id)
        fb = factors.get(hit.b_id)
        if fa is None or fb is None:
            continue
        try:
            p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
            p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
        except HTTPException:
            continue
        cint = engle_granger(p_a, p_b)
        if cint.spread.empty:
            continue
        try:
            bt = pairs_backtest(
                cint.spread,
                window=body.window,
                entry_z=body.entry_z,
                exit_z=body.exit_z,
                stop_z=body.stop_z,
                annualisation_factor=body.annualisation,
            )
        except ValueError:
            continue
        leaderboard.append(
            AutoBacktestRow(
                a_id=fa.id,
                b_id=fb.id,
                sharpe=bt.sharpe,
                sharpe_is=bt.sharpe_is,
                sharpe_oos=bt.sharpe_oos,
                oos_to_is_ratio=bt.oos_to_is_ratio,
                sortino=bt.sortino,
                calmar=bt.calmar,
                hit_rate=bt.hit_rate,
                max_drawdown=bt.max_drawdown,
                n_trades=bt.n_trades,
                mean_holding_days=bt.mean_holding_days,
                half_life_days=cint.half_life_days,
                adf_pvalue=cint.adf_pvalue,
                beta_hedge=cint.beta_hedge,
            )
        )
    leaderboard.sort(key=lambda r: r.sharpe, reverse=True)
    runtime = (pd.Timestamp.now() - t0).total_seconds()
    return AutoBacktestResponse(
        n_factors_scanned=n_factors,
        n_coint_hits=n_coint,
        n_backtested=len(leaderboard),
        runtime_seconds=runtime,
        leaderboard=leaderboard,
    )


@router.post("/strategies/patterns", response_model=PatternsResponse)
def strategies_patterns(
    body: PatternsRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> PatternsResponse:
    """Cross-pair structural-pattern analysis: PnL correlation matrix,
    day-of-week effects per pair, pre-resolution regime shifts, k-means
    clustering on pair signatures.

    Use this on the OOS-validated leaderboard to surface portfolio-level
    insights — pair independence, weekday seasonality, vol-explosion
    near resolution, structural groupings."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")

    # Build PnL series + signatures per pair.
    pnls: dict[str, pd.Series] = {}
    spreads: dict[str, pd.Series] = {}
    signatures: dict[str, dict[str, float]] = {}
    for spec in body.pairs:
        try:
            fa = _resolve_one(spec.a_id, factors, role="a")
            fb = _resolve_one(spec.b_id, factors, role="b")
            p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
            p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
            cint = engle_granger(p_a, p_b)
            if cint.spread.empty:
                continue
            bt = pairs_backtest(
                cint.spread,
                window=body.window,
                entry_z=body.entry_z,
                exit_z=body.exit_z,
                stop_z=body.stop_z,
                annualisation_factor=body.annualisation,
            )
        except (HTTPException, ValueError):
            continue
        label = f"{spec.a_id}↔{spec.b_id}"
        pnls[label] = bt.pnl
        spreads[label] = cint.spread
        signatures[label] = {
            "sharpe": float(bt.sharpe),
            "half_life": float(cint.half_life_days) if cint.half_life_days else 30.0,
            "hit_rate": float(bt.hit_rate),
            "n_trades": float(bt.n_trades),
            "max_drawdown": float(bt.max_drawdown),
        }

    # Run the 4 analyses.
    corr = correlate_pair_pnls(pnls)
    corr_out = CorrelationOut(
        pair_labels=corr.pair_labels,
        correlation_matrix=corr.correlation_matrix,
        mean_off_diagonal=corr.mean_off_diagonal,
        max_off_diagonal=corr.max_off_diagonal,
        most_correlated_a=corr.most_correlated[0] if corr.most_correlated else None,
        most_correlated_b=corr.most_correlated[1] if corr.most_correlated else None,
        most_correlated_rho=corr.most_correlated[2] if corr.most_correlated else None,
        diversification_ratio=corr.diversification_ratio,
    )
    dow_results: list[DowOut] = []
    for label, pnl in pnls.items():
        dow = day_of_week_effect(pnl)
        dow_results.append(
            DowOut(
                pair=label,
                means=dow.means,
                counts=dow.counts,
                t_stats=dow.t_stats,
                p_values=dow.p_values,
                best_day=dow.best_day[0] if dow.best_day else None,
                worst_day=dow.worst_day[0] if dow.worst_day else None,
                significant_days=dow.significant_days,
            )
        )
    pre_results: list[PreResolutionOut] = []
    for label, spread in spreads.items():
        pre = pre_resolution_regime(spread, days_to_resolution=body.days_to_resolution)
        pre_results.append(
            PreResolutionOut(
                pair=label,
                far_n=pre.far_n,
                near_n=pre.near_n,
                far_std=_finite(pre.far_std),
                near_std=_finite(pre.near_std),
                vol_ratio=_finite(pre.vol_ratio),
                mean_shift=_finite(pre.mean_shift),
                f_stat=_finite(pre.f_stat),
                f_p_value=_finite(pre.f_p_value),
                vol_shift_significant=pre.vol_shift_significant,
            )
        )
    clust = cluster_pairs_by_signature(
        signatures,
        n_clusters=min(body.n_clusters, max(2, len(signatures) - 1)),
    )
    cluster_out = [
        ClusterOut(
            cluster_id=c.cluster_id,
            pair_labels=c.pair_labels,
            centroid=c.centroid,
            n_members=c.n_members,
        )
        for c in clust.clusters
    ]

    return PatternsResponse(
        n_pairs_analysed=len(pnls),
        correlation=corr_out,
        day_of_week=dow_results,
        pre_resolution=pre_results,
        clusters=cluster_out,
        silhouette_proxy=clust.silhouette_proxy,
    )


@router.post("/strategies/ml-predictor", response_model=MlPredictorResponse)
def strategies_ml_predictor(
    body: MlPredictorRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> MlPredictorResponse:
    """Gradient-boosted regressor predicting next-bar Δspread from
    engineered features (lag-z, rolling vol, autocorrelation, momentum,
    long-window distance-from-mean). TimeSeriesSplit cross-validation.
    Reports R², direction accuracy, information coefficient, beats-baseline,
    feature importances, and a forward-looking last_prediction.

    Use as a *complement* to the z-score state machine, not a replacement —
    direction accuracy >55% across folds is the bar to clear before
    integrating predictions into a live signal."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    cint = engle_granger(p_a, p_b)
    if cint.spread.empty:
        raise HTTPException(status_code=422, detail=cint.verdict)
    out = fit_ml_predictor(
        cint.spread,
        n_folds=body.n_folds,
        n_estimators=body.n_estimators,
        max_depth=body.max_depth,
        learning_rate=body.learning_rate,
        seed=body.seed,
    )
    return MlPredictorResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=out.n_obs,
        n_features=out.n_features,
        feature_names=out.feature_names,
        n_folds=out.n_folds,
        folds=[
            MlFoldOut(
                fold=f.fold,
                n_train=f.n_train,
                n_test=f.n_test,
                test_r2=f.test_r2,
                test_direction_accuracy=f.test_direction_accuracy,
                baseline_direction_accuracy=f.baseline_direction_accuracy,
                information_coefficient=f.information_coefficient,
            )
            for f in out.folds
        ],
        mean_test_r2=out.mean_test_r2,
        mean_direction_accuracy=out.mean_direction_accuracy,
        mean_baseline_direction_accuracy=out.mean_baseline_direction_accuracy,
        beats_baseline=out.beats_baseline,
        mean_ic=out.mean_ic,
        feature_importances=[
            FeatureImportanceOut(name=fi.name, importance=fi.importance)
            for fi in out.feature_importances
        ],
        last_prediction=out.last_prediction,
        verdict=out.verdict,
    )


@router.post("/strategies/info-share", response_model=InfoShareResponse)
def strategies_info_share(
    body: InfoShareRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> InfoShareResponse:
    """Hasbrouck (1995) Information Share — for two cointegrated price
    series, decomposes the proportion of long-run price discovery each
    venue contributes. The leader's IS is closer to 1; the follower's
    closer to 0. Use to identify which venue (Kalshi vs Polymarket) drives
    a cross-platform Fed-cut basis."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    try:
        out = hasbrouck_information_share(
            p_a,
            p_b,
            venue_a_id=fa.id,
            venue_b_id=fb.id,
            var_lags=body.var_lags,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return InfoShareResponse(
        venue_a_id=fa.id,
        venue_b_id=fb.id,
        n_obs=out.n_obs,
        is_a_lower=out.is_a_lower,
        is_a_upper=out.is_a_upper,
        is_b_lower=out.is_b_lower,
        is_b_upper=out.is_b_upper,
        midpoint_a=out.midpoint_a,
        leader=out.leader,
        beta_cointeg=out.beta_cointeg,
    )


@router.post("/strategies/regime-switching", response_model=RegimeSwitchingResponse)
def strategies_regime_switching(
    body: RegimeSwitchingRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> RegimeSwitchingResponse:
    """Hamilton (1989) Markov-switching variance model on the cointegration
    spread. State 0 = tight mean-reversion (low σ, tradeable); state 1 =
    broken (high σ, regime change risk). Returns smoothed P(state=1) per bar.

    Practitioner use: if `current_regime_prob > 0.5`, suspend the pairs trade
    and re-validate cointegration."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    cint = engle_granger(p_a, p_b)
    if cint.spread.empty:
        raise HTTPException(status_code=422, detail=cint.verdict)
    try:
        out = markov_regime_switching(cint.spread, k_regimes=body.k_regimes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    points = [RegimePoint(date=ts.date(), p_state1=float(p)) for ts, p in out.regime_probs.items()]
    return RegimeSwitchingResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=out.n_obs,
        n_state0=out.n_state0,
        n_state1=out.n_state1,
        sigma_state0=out.sigma_state0,
        sigma_state1=out.sigma_state1,
        mean_state0=out.mean_state0,
        mean_state1=out.mean_state1,
        transition_p00=out.transition_p00,
        transition_p11=out.transition_p11,
        current_regime=out.current_regime,
        current_regime_prob=out.current_regime_prob,
        verdict=out.verdict,
        series=points,
    )


@router.post("/strategies/almgren-chriss", response_model=AlmgrenChrissResponse)
def strategies_almgren_chriss(body: AlmgrenChrissRequest) -> AlmgrenChrissResponse:
    """Closed-form Almgren-Chriss (2001) optimal execution trajectory. No
    historical data needed — pure mathematical optimisation given the
    target position and impact/risk parameters. Use to schedule entry of a
    large pairs trade without telegraphing it to the market."""
    try:
        out = almgren_chriss_schedule(
            target_position=body.target_position,
            n_intervals=body.n_intervals,
            time_horizon=body.time_horizon,
            sigma=body.sigma,
            eta=body.eta,
            epsilon=body.epsilon,
            gamma_perm=body.gamma_perm,
            risk_aversion=body.risk_aversion,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return AlmgrenChrissResponse(
        n_intervals=out.n_intervals,
        x_remaining=out.x_remaining,
        n_per_interval=out.n_per_interval,
        kappa=out.kappa,
        time_horizon=out.time_horizon,
        expected_cost=out.expected_cost,
        variance_cost=out.variance_cost,
        utility=out.utility,
    )


@router.post("/strategies/fractional-diff", response_model=FractionalDiffResponse)
def strategies_fractional_diff(
    body: FractionalDiffRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> FractionalDiffResponse:
    """Hosking (1981) / López de Prado (2018 §5) fractional differentiation.

    If ``d`` is None, finds the minimal d in (0, 1) that makes the series
    stationary (ADF p < 0.05) — the López de Prado recipe to maximise
    memory preservation while ensuring stationarity.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fc = _resolve_one(body.factor_id, factors, role="factor")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    s = _fetch_aligned_prob(fc, start_ts, end_ts, poly, cache, settings)
    try:
        out = find_minimal_d(s, threshold=body.threshold)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return FractionalDiffResponse(
        factor_id=fc.id,
        minimal_d=out.d,
        adf_p_at_minimal_d=_finite(out.adf_p_at_d) if out.adf_p_at_d is not None else None,
        correlation_with_original=_finite(out.correlation_with_original)
        if out.correlation_with_original is not None
        else None,
        weights_width=out.weights_width,
        grid=[FractionalDiffGridPoint(**g) for g in out.grid_results],
    )


@router.post("/strategies/garch", response_model=GarchResponse)
def strategies_garch(
    body: GarchRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> GarchResponse:
    """Bollerslev (1986) GARCH(1,1) — conditional volatility on Δ-series.

    Fits μ, ω, α, β by MLE. Returns one-step-ahead σ forecast for live
    vol-targeting and the per-bar conditional σ history for diagnostics.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fc = _resolve_one(body.factor_id, factors, role="factor")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    s = _fetch_aligned_prob(fc, start_ts, end_ts, poly, cache, settings)
    try:
        out = fit_garch_11(s)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return GarchResponse(
        factor_id=fc.id,
        n_obs=out.n_obs,
        converged=out.converged,
        is_stationary=out.is_stationary,
        mu=out.mu,
        omega=out.omega,
        alpha=out.alpha,
        beta=out.beta,
        persistence=out.persistence,
        long_run_variance=_finite(out.long_run_variance)
        if out.long_run_variance != float("inf")
        else 0.0,
        log_likelihood=out.log_likelihood,
        last_sigma=out.last_sigma,
        next_bar_sigma_forecast=out.last_sigma,
    )


@router.post("/strategies/dfa", response_model=DfaResponse)
def strategies_dfa(
    body: DfaRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> DfaResponse:
    """Peng et al. (1994) Detrended Fluctuation Analysis — robust Hurst
    exponent on the integrated/cumulative-sum series. Robust to
    non-stationary trends. α<0.5 = mean-reverting; α≈0.5 = random walk;
    α>0.5 = persistent; α>1 = non-stationary.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fc = _resolve_one(body.factor_id, factors, role="factor")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    s = _fetch_aligned_prob(fc, start_ts, end_ts, poly, cache, settings)
    out = dfa_fn(s, poly_order=body.poly_order)
    return DfaResponse(
        factor_id=fc.id,
        n_obs=out.n_obs,
        alpha_dfa=_finite(out.alpha) if not np.isnan(out.alpha) else None,
        r_squared_log_log=_finite(out.r_squared) if not np.isnan(out.r_squared) else None,
        interpretation=out.interpretation,
        log_n=out.log_n,
        log_f=out.log_f,
    )


@router.post("/strategies/triple-barrier", response_model=TripleBarrierResponse)
def strategies_triple_barrier(
    body: TripleBarrierRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> TripleBarrierResponse:
    """López de Prado (2018) Triple Barrier Method on a cointegration spread.

    Adaptive vol-scaled exits: each trade has 3 barriers — profit target
    (+pt·σ_local), stop loss (−sl·σ_local), time horizon (T bars). Exit
    on first touch."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    cint = engle_granger(p_a, p_b)
    if cint.spread.empty:
        raise HTTPException(status_code=422, detail=cint.verdict)
    try:
        out = triple_barrier_backtest(
            cint.spread,
            window=body.window,
            entry_z=body.entry_z,
            profit_target_sigma=body.profit_target_sigma,
            stop_loss_sigma=body.stop_loss_sigma,
            time_horizon_bars=body.time_horizon_bars,
            annualisation=body.annualisation,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    profit_rate = out.n_profit_hits / out.n_trades if out.n_trades else 0.0
    avg_hold = sum(t.holding_bars for t in out.trades) / out.n_trades if out.n_trades else 0.0
    return TripleBarrierResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_trades=out.n_trades,
        n_profit_hits=out.n_profit_hits,
        n_stop_hits=out.n_stop_hits,
        n_time_hits=out.n_time_hits,
        total_pnl=out.total_pnl,
        sharpe=out.sharpe,
        profit_hit_rate=float(profit_rate),
        avg_holding_bars=float(avg_hold),
    )


@router.post("/strategies/distance-method", response_model=DistanceMethodResponse)
def strategies_distance_method(
    body: DistanceMethodRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> DistanceMethodResponse:
    """Gatev-Goetzmann-Rouwenhorst (2006) Distance Method.

    Classical pairs-trading benchmark: form-period SSD on normalised
    series, trade widest deviations during the trading period (no peeking)."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    try:
        out = distance_method(
            p_a,
            p_b,
            a_id=fa.id,
            b_id=fb.id,
            formation_fraction=body.formation_fraction,
            entry_sigma=body.entry_sigma,
            exit_sigma=body.exit_sigma,
            annualisation=body.annualisation,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return DistanceMethodResponse(
        a_id=fa.id,
        b_id=fb.id,
        formation_ssd=out.formation_ssd,
        formation_sigma=out.formation_sigma,
        n_trading_bars=out.n_trading_bars,
        n_trades=out.n_trades,
        trade_pnl=out.trade_pnl,
        sharpe=out.sharpe,
    )


@router.post("/strategies/robust-validation", response_model=RobustValidationResponse)
def strategies_robust_validation(
    body: RobustValidationRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> RobustValidationResponse:
    """Comprehensive robustness battery on a portfolio of pair trades.

    Builds the portfolio PnL series via the same vol-targeted combiner used
    by /strategies/portfolio, then runs 5 robustness tests:
    1. Lo (2002) asymptotic Sharpe SE
    2. Block bootstrap CI on Sharpe (Politis-Romano)
    3. Sign-flip permutation null on Sharpe
    4. Out-of-time (50/50 train/test) test
    5. Deflated Sharpe Ratio (Bailey-Lopez de Prado, multiple-testing corrected)
    Plus cost-sensitivity sweep.

    Returns an overall verdict in {STRONG ALPHA, MARGINAL ALPHA,
    WEAK / SUSPECT, NOISE / OVERFIT}.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")

    pnls: dict[str, pd.Series] = {}
    pos_changes_per_pair: dict[str, pd.Series] = {}
    for spec in body.pairs:
        try:
            fa = _resolve_one(spec.a_id, factors, role="a")
            fb = _resolve_one(spec.b_id, factors, role="b")
            p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
            p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
            cint = engle_granger(p_a, p_b)
            if cint.spread.empty:
                continue
            spread = cint.spread
            if spec.signal_type == "bollinger_15":
                pos = bollinger_signals(spread, window=spec.window, k_entry=1.5, k_exit=0.0)
                dspread = spread.diff().fillna(0.0)
                pnl = pos.shift(1).fillna(0).astype(float) * dspread
            else:
                bt = pairs_backtest(spread, window=spec.window, entry_z=2.0, exit_z=0.5, stop_z=4.0)
                pos = bt.positions
                pnl = bt.pnl
            pos_changes = pos.diff().abs().fillna(0)
        except (HTTPException, ValueError):
            continue
        label = f"{spec.a_id}↔{spec.b_id}"
        pnls[label] = pnl
        pos_changes_per_pair[label] = pos_changes

    if not pnls:
        raise HTTPException(status_code=422, detail="no pairs successfully built")
    if len(pnls) >= 2:
        try:
            combo = vol_targeted_combiner(
                pnls,
                target_per_leg_vol=body.target_per_leg_vol,
                walk_forward_folds=None,
            )
            portfolio_pnl = combo.pnl_series
            # Aggregate position changes by weight to get portfolio-level cost basis.
            df_pos = pd.DataFrame(pos_changes_per_pair)
            weights = pd.Series(combo.weights)
            portfolio_pos_change = (df_pos * weights).sum(axis=1)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    else:
        # Single-pair case
        only_label = next(iter(pnls.keys()))
        portfolio_pnl = pnls[only_label]
        portfolio_pos_change = pos_changes_per_pair[only_label]

    rep = run_robust_validation(
        portfolio_pnl,
        position_changes=portfolio_pos_change,
        annualisation=body.annualisation,
        n_trials_searched=body.n_trials_searched,
        seed=body.seed,
    )

    cost_pts = [
        CostSensitivityPoint(cost_bps=cb, net_sharpe=ns)
        for cb, ns in zip(
            rep.cost_sensitivity["costs_bps"], rep.cost_sensitivity["net_sharpe"], strict=True
        )
    ]

    # Count tests passed for transparency.
    n_passed = sum(
        [
            rep.lo_test["p_value"] < 0.05,
            rep.bootstrap_ci["ci_lo_95"] > 0,
            rep.permutation["p_value"] < 0.05,
            rep.out_of_time["ratio"] > 0.5,
            rep.deflated_sharpe["deflated_p_value"] < 0.05,
        ]
    )

    return RobustValidationResponse(
        portfolio_sharpe=rep.portfolio_sharpe,
        n_obs=rep.n_obs,
        overall_verdict=rep.overall_verdict,
        n_tests_passed=int(n_passed),
        lo_sharpe=rep.lo_test["sharpe"],
        lo_se=rep.lo_test["se"],
        lo_z_stat=rep.lo_test["z_stat"],
        lo_p_value=rep.lo_test["p_value"],
        lo_ci_lo_95=rep.lo_test["ci_lo_95"],
        lo_ci_hi_95=rep.lo_test["ci_hi_95"],
        bootstrap_ci_lo_90=rep.bootstrap_ci["ci_lo_90"],
        bootstrap_ci_hi_90=rep.bootstrap_ci["ci_hi_90"],
        bootstrap_ci_lo_95=rep.bootstrap_ci["ci_lo_95"],
        bootstrap_ci_hi_95=rep.bootstrap_ci["ci_hi_95"],
        permutation_p_value=rep.permutation["p_value"],
        permutation_null_median=rep.permutation["null_median"],
        permutation_null_pct95=rep.permutation["null_pct95"],
        cost_sensitivity=cost_pts,
        break_even_cost_bps=rep.cost_sensitivity["break_even_bps"],
        out_of_time_train_sharpe=rep.out_of_time["train_sharpe"],
        out_of_time_test_sharpe=rep.out_of_time["test_sharpe"],
        out_of_time_ratio=rep.out_of_time["ratio"],
        out_of_time_verdict=rep.out_of_time["verdict"],
        deflated_sharpe=rep.deflated_sharpe["deflated_sharpe"],
        deflated_p_value=rep.deflated_sharpe["deflated_p_value"],
        expected_max_sharpe_under_null=rep.deflated_sharpe["expected_max_sharpe_under_null"],
    )


@router.post("/strategies/portfolio", response_model=PortfolioResponse)
def strategies_portfolio(
    body: PortfolioRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> PortfolioResponse:
    """Vol-targeted portfolio combiner. Aggregates the per-bar PnLs of N
    pair-trading strategies into a single equity curve, weighted so each
    leg contributes ``target_per_leg_vol`` annualised volatility.

    Reports portfolio-level Sharpe, Sortino, Calmar, max DD, VaR/CVaR,
    plus walk-forward OOS Sharpe distribution. Per-leg signal type can
    be overridden per pair (``zscore`` or ``bollinger_15``)."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")

    pnls: dict[str, pd.Series] = {}
    for spec in body.pairs:
        try:
            fa = _resolve_one(spec.a_id, factors, role="a")
            fb = _resolve_one(spec.b_id, factors, role="b")
            p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
            p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
            cint = engle_granger(p_a, p_b)
            if cint.spread.empty:
                continue
            spread = cint.spread
            if spec.signal_type == "bollinger_15":
                pos = bollinger_signals(spread, window=spec.window, k_entry=1.5, k_exit=0.0)
                # Compute PnL from pos + spread.
                dspread = spread.diff().fillna(0.0)
                pnl = pos.shift(1).fillna(0).astype(float) * dspread
            else:
                bt = pairs_backtest(spread, window=spec.window, entry_z=2.0, exit_z=0.5, stop_z=4.0)
                pnl = bt.pnl
        except (HTTPException, ValueError):
            continue
        label = f"{spec.a_id}↔{spec.b_id}"
        pnls[label] = pnl

    if len(pnls) < 2:
        raise HTTPException(status_code=422, detail="<2 pairs successfully built")
    try:
        out = vol_targeted_combiner(
            pnls,
            target_per_leg_vol=body.target_per_leg_vol,
            walk_forward_folds=body.walk_forward_folds,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return PortfolioResponse(
        n_pairs=out.n_pairs,
        pair_labels=out.pair_labels,
        weights=out.weights,
        individual_sharpes=out.individual_sharpes,
        correlation_matrix=out.correlation_matrix,
        n_obs=out.n_obs,
        portfolio_sharpe=out.portfolio_sharpe,
        portfolio_sortino=out.portfolio_sortino,
        portfolio_calmar=out.portfolio_calmar,
        portfolio_max_drawdown=out.portfolio_max_drawdown,
        portfolio_var_95=out.portfolio_var_95,
        portfolio_cvar_95=out.portfolio_cvar_95,
        portfolio_skew=out.portfolio_skew,
        oos_sharpe_mean=_finite(out.oos_sharpe_mean) if out.oos_sharpe_mean is not None else None,
        oos_sharpe_std=_finite(out.oos_sharpe_std) if out.oos_sharpe_std is not None else None,
        oos_sharpe_min=_finite(out.oos_sharpe_min) if out.oos_sharpe_min is not None else None,
    )


@router.post("/strategies/factor-model-pro", response_model=FactorModelProResponse)
def strategies_factor_model_pro(
    body: FactorModelProRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> FactorModelProResponse:
    """Production-grade multi-event factor model.

    Beyond basic event_model:
    - Estimator choice: OLS / Ridge / Lasso / ElasticNet
    - Logit transform option (handles [0,1] bounds)
    - PCA pre-processing (collapse collinear factors)
    - Residual diagnostics: Ljung-Box, Jarque-Bera, ARCH-LM, Durbin-Watson
    - Cross-validated R² (TimeSeriesSplit) — true OOS, not in-sample
    - Walk-forward β stability (mean ± std across folds)
    - Bootstrap R² CI

    Use Lasso when you have many candidate factors and want auto-selection.
    Use Ridge when factors are collinear (VIF > 10).
    Always check `r_squared_cv` (the trustworthy metric) — if it's far below
    `r_squared_is`, the model is overfit.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    if body.target_id in body.factor_ids:
        raise HTTPException(
            status_code=400,
            detail=f"target_id {body.target_id!r} cannot also appear in factor_ids",
        )
    target_fc = _resolve_one(body.target_id, factors, role="target")
    factor_fcs = [_resolve_one(fid, factors, role="factor") for fid in body.factor_ids]
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    target_series = _fetch_aligned_prob(target_fc, start_ts, end_ts, poly, cache, settings)
    fcols = {
        fc.id: _fetch_aligned_prob(fc, start_ts, end_ts, poly, cache, settings) for fc in factor_fcs
    }
    X = pd.DataFrame(fcols)
    try:
        out = fit_factor_model_pro(
            target_series,
            X,
            target_id=target_fc.id,
            estimator=body.estimator,
            alpha=body.alpha,
            transform=body.transform,
            use_pca=body.use_pca,
            pca_explained_variance_target=body.pca_explained_variance_target,
            n_cv_folds=body.n_cv_folds,
            bootstrap_iters=body.bootstrap_iters,
            seed=body.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    coefs_out = [
        CoefficientProOut(
            factor_id=c.factor_id,
            beta=c.beta,
            beta_std_across_folds=_finite(c.beta_std_across_folds)
            if c.beta_std_across_folds is not None
            else None,
            stability_ratio=_finite(c.stability_ratio) if c.stability_ratio is not None else None,
            is_zeroed=c.is_zeroed,
            significance=c.significance,
        )
        for c in out.coefficients
    ]
    diag = out.diagnostics
    diag_out = ResidualDiagnosticsOut(
        ljung_box_p=_finite(diag.ljung_box_p),
        jarque_bera_p=_finite(diag.jarque_bera_p),
        arch_lm_p=_finite(diag.arch_lm_p) if diag.arch_lm_p is not None else None,
        durbin_watson=diag.durbin_watson,
        residual_std=diag.residual_std,
        residual_skew=diag.residual_skew,
        residual_kurtosis=diag.residual_kurtosis,
        well_specified=diag.well_specified,
    )
    return FactorModelProResponse(
        target_id=out.target_id,
        estimator=out.estimator,
        transform=out.transform,
        use_pca=out.use_pca,
        n_obs=out.n_obs,
        n_factors=out.n_factors,
        coefficients=coefs_out,
        intercept=out.intercept,
        r_squared_is=out.r_squared_is,
        r_squared_cv=out.r_squared_cv,
        r_squared_cv_std=out.r_squared_cv_std,
        r_squared_ci_lo_95=out.r_squared_ci_lo_95,
        r_squared_ci_hi_95=out.r_squared_ci_hi_95,
        diagnostics=diag_out,
        pca_explained_variance=out.pca_explained_variance,
        n_zeroed_factors=out.n_zeroed_factors,
        overfit_flag=out.overfit_flag,
    )


@router.post("/strategies/cusum", response_model=CusumResponse)
def strategies_cusum(
    body: CusumRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> CusumResponse:
    """Brown-Durbin-Evans CUSUM-OLS structural-break test on the
    Engle-Granger spread. Detects level shifts / regime changes in the
    cointegrating relationship — useful before deploying capital on a
    pair: a recent break makes the historical β_hedge unreliable."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    cint = engle_granger(p_a, p_b)
    if cint.spread.empty:
        raise HTTPException(status_code=422, detail=cint.verdict)
    out = cusum_test(cint.spread)
    points = [CusumPoint(date=ts.date(), cusum=float(v)) for ts, v in out.cusum_series.items()]
    return CusumResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=out.n_obs,
        verdict=out.verdict,
        rejected=out.rejected,
        max_abs_cusum=_finite(out.max_abs_cusum),
        threshold_95=_finite(out.threshold_95),
        break_point=out.break_point.date() if out.break_point is not None else None,
        series=points,
    )


@router.post("/strategies/walk-forward", response_model=WalkForwardResponse)
def strategies_walk_forward(
    body: WalkForwardRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> WalkForwardResponse:
    """K-fold walk-forward backtest. Reports the *distribution* of test-fold
    Sharpes — much more credible than a single train/test split. Stable =
    min(test Sharpe) > 0 AND std(test Sharpe) < |mean|."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    cint = engle_granger(p_a, p_b)
    if cint.spread.empty:
        raise HTTPException(status_code=422, detail=cint.verdict)
    try:
        out = walk_forward_backtest(
            cint.spread,
            n_folds=body.n_folds,
            window=body.window,
            entry_z=body.entry_z,
            exit_z=body.exit_z,
            stop_z=body.stop_z,
            annualisation=body.annualisation,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return WalkForwardResponse(
        a_id=fa.id,
        b_id=fb.id,
        n_obs=out.n_obs,
        n_folds=out.n_folds,
        folds=[
            WalkForwardFoldOut(
                fold=f.fold,
                test_start=f.test_start.date(),
                test_end=f.test_end.date(),
                train_sharpe=f.train_sharpe,
                test_sharpe=f.test_sharpe,
                n_train=f.n_train,
                n_test=f.n_test,
            )
            for f in out.folds
        ],
        train_sharpe_mean=out.train_sharpe_mean,
        test_sharpe_mean=out.test_sharpe_mean,
        test_sharpe_median=out.test_sharpe_median,
        test_sharpe_min=out.test_sharpe_min,
        test_sharpe_max=out.test_sharpe_max,
        test_sharpe_std=out.test_sharpe_std,
        stability=out.stability,
    )


@router.post("/strategies/sharpe-bootstrap", response_model=SharpeBootstrapResponse)
def strategies_sharpe_bootstrap(
    body: SharpeBootstrapRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> SharpeBootstrapResponse:
    """Stationary block-bootstrap (Politis-Romano 1994) CI on the Sharpe
    of a pair's z-score backtest. CI excluding zero ⇒ Sharpe is statistically
    distinguishable from random."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    cint = engle_granger(p_a, p_b)
    if cint.spread.empty:
        raise HTTPException(status_code=422, detail=cint.verdict)
    try:
        bt = pairs_backtest(
            cint.spread,
            window=body.window,
            entry_z=body.entry_z,
            exit_z=body.exit_z,
            stop_z=body.stop_z,
            annualisation_factor=body.annualisation,
        )
        out = bootstrap_sharpe_ci(
            bt.pnl.to_numpy(),
            annualisation=body.annualisation,
            n_iters=body.n_iters,
            seed=body.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return SharpeBootstrapResponse(
        a_id=fa.id,
        b_id=fb.id,
        sharpe_point=out.sharpe_point,
        sharpe_mean=out.sharpe_mean,
        sharpe_std=out.sharpe_std,
        sharpe_ci_lo_90=out.sharpe_ci_lo_90,
        sharpe_ci_hi_90=out.sharpe_ci_hi_90,
        sharpe_ci_lo_95=out.sharpe_ci_lo_95,
        sharpe_ci_hi_95=out.sharpe_ci_hi_95,
        n_bootstrap=out.n_bootstrap,
        block_size=out.block_size,
    )


@router.post("/strategies/sharpe-permutation", response_model=PermutationSharpeResponse)
def strategies_sharpe_permutation(
    body: PermutationSharpeRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> PermutationSharpeResponse:
    """Permutation null distribution of the Sharpe ratio. Sign-flips the
    spread's first differences; rebuilds; runs the same strategy; computes
    Sharpe. ``p = P(null Sharpe ≥ real Sharpe)``. p < 0.05 ⇒ the real
    Sharpe doesn't come from random fluctuations of the spread."""
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    fa = _resolve_one(body.a_id, factors, role="a")
    fb = _resolve_one(body.b_id, factors, role="b")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")
    p_a = _fetch_aligned_prob(fa, start_ts, end_ts, poly, cache, settings)
    p_b = _fetch_aligned_prob(fb, start_ts, end_ts, poly, cache, settings)
    cint = engle_granger(p_a, p_b)
    if cint.spread.empty:
        raise HTTPException(status_code=422, detail=cint.verdict)

    # Strategy-as-function: thin closure around the existing z-score state machine.
    def _strategy_pnl(spread_arr):
        s = pd.Series(spread_arr, index=cint.spread.index[: len(spread_arr)])
        try:
            bt = pairs_backtest(
                s,
                window=body.window,
                entry_z=body.entry_z,
                exit_z=body.exit_z,
                stop_z=body.stop_z,
                annualisation_factor=body.annualisation,
            )
            return bt.pnl.to_numpy()
        except ValueError:
            import numpy as _np

            return _np.zeros(len(spread_arr))

    try:
        out = permutation_sharpe_test(
            cint.spread.to_numpy(),
            pnl_strategy_fn=_strategy_pnl,
            annualisation=body.annualisation,
            n_iters=body.n_iters,
            seed=body.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PermutationSharpeResponse(
        a_id=fa.id,
        b_id=fb.id,
        real_sharpe=out.real_sharpe,
        null_sharpes=out.null_sharpes,
        null_median=out.null_median,
        null_pct95=out.null_pct95,
        p_value=out.p_value,
        n_iters=out.n_iters,
    )


# Curated presets — generated from the actual scanner hits over 2026-01 → 2026-04.
# Update these manually when factors.yml expands and the scanner finds new alpha.
_TODAY_TS = pd.Timestamp.now(tz="UTC").normalize()
_DEFAULT_END = _TODAY_TS.date().isoformat()
_DEFAULT_START = (_TODAY_TS - pd.Timedelta(days=180)).date().isoformat()


def _build_presets() -> PresetsResponse:
    base = {"start": _DEFAULT_START, "end": _DEFAULT_END}
    return PresetsResponse(
        cointegration=[
            StrategyPreset(
                label="BTC strike ladder",
                metric="Sharpe 5.7 · OOS/IS 2.6",
                tier="alpha",
                description="P(BTC≥$100k) vs P(BTC≥$500k). Same underlying, different barriers — strongest OOS-validated alpha in the catalog.",
                inputs={"a_id": "btc_100k_eoy", "b_id": "btc_500k_eoy", **base},
            ),
            StrategyPreset(
                label="Senate inverse",
                metric="Sharpe 3.5 · OOS/IS 1.4",
                tier="alpha",
                description="Dem vs Rep Senate 2026 — must sum to ~1 by construction, β≈−1, half-life 0.6d.",
                inputs={"a_id": "dem_senate_2026", "b_id": "rep_senate_2026", **base},
            ),
            StrategyPreset(
                label="ETH price-target ladder",
                metric="Sharpe 3.1 · OOS/IS 2.4",
                tier="alpha",
                description="P(ETH≥$10k EOY) vs P(ETH≥$5k EOY). Cleanest mean-reverter in crypto.",
                inputs={"a_id": "eth_10k_eoy", "b_id": "eth_5k_eoy", **base},
            ),
            StrategyPreset(
                label="Tech mega-cap",
                metric="Sharpe 3.0 · OOS/IS 1.2",
                tier="alpha",
                description="TSLA largest ↔ NVDA largest by Jun. Mega-cap horse race.",
                inputs={"a_id": "tsla_largest_jun", "b_id": "nvda_largest_jun", **base},
            ),
            StrategyPreset(
                label="House midterms inverse",
                metric="Sharpe 1.8 · OOS/IS 1.8",
                tier="alpha",
                description="Dem vs Rep House 2026. Clean inverse, robust OOS.",
                inputs={"a_id": "dem_house_2026", "b_id": "rep_house_2026", **base},
            ),
            StrategyPreset(
                label="Cross-platform Fed-cut",
                metric="Kalshi × Polymarket",
                tier="cross-venue",
                description="KXFEDDECISION-26SEP-C25 (Kalshi) vs P(≥3 cuts in 2026) — same event, different venue.",
                inputs={"a_id": "k_fed_sep_cut25", "b_id": "fed_cuts_3_2026", **base},
            ),
        ],
        pairs=[
            StrategyPreset(
                label="BTC strike ladder backtest",
                metric="Sharpe 5.7",
                tier="alpha",
                description="The strongest pair in the catalog. Expect ~3 trades on the 4-month window with 100% hit rate.",
                inputs={
                    "a_id": "btc_100k_eoy",
                    "b_id": "btc_500k_eoy",
                    "window": 20,
                    "entry_z": 2.0,
                    "exit_z": 0.5,
                    "stop_z": 4.0,
                    "annualisation": 252,
                    "oos_fraction": 0.30,
                    **base,
                },
            ),
            StrategyPreset(
                label="Senate inverse backtest",
                metric="Sharpe 3.5",
                tier="alpha",
                description="6 trades, hit rate 83%, mechanical inverse. Tight bands work here.",
                inputs={
                    "a_id": "dem_senate_2026",
                    "b_id": "rep_senate_2026",
                    "window": 20,
                    "entry_z": 2.0,
                    "exit_z": 0.5,
                    "stop_z": 4.0,
                    "annualisation": 252,
                    "oos_fraction": 0.30,
                    **base,
                },
            ),
            StrategyPreset(
                label="ETH ladder backtest",
                metric="Sharpe 3.1",
                tier="alpha",
                description="OOS Sharpe 4.77 vs IS 1.98 — the OOS test confirms it's not just luck.",
                inputs={
                    "a_id": "eth_10k_eoy",
                    "b_id": "eth_5k_eoy",
                    "window": 20,
                    "entry_z": 1.8,
                    "exit_z": 0.4,
                    "stop_z": 4.0,
                    "annualisation": 252,
                    "oos_fraction": 0.30,
                    **base,
                },
            ),
            StrategyPreset(
                label="MSFT vs Musk-trillionaire",
                metric="Sharpe 2.9",
                tier="alpha",
                description="Surprising cross-theme co-move, OOS/IS 2.13. Both markets respond to tech mega-cap risk-on regime.",
                inputs={
                    "a_id": "msft_largest_jun",
                    "b_id": "musk_trillionaire",
                    "window": 20,
                    "entry_z": 2.0,
                    "exit_z": 0.5,
                    "stop_z": 4.0,
                    "annualisation": 252,
                    "oos_fraction": 0.30,
                    **base,
                },
            ),
        ],
        pair_explorer=[
            StrategyPreset(
                label="Implication violation: Fed cuts",
                description="P(≥2 cuts) vs P(≥1 cut) — by logic P(2)≤P(1). Auto-discovered to violate on 62/97 days.",
                inputs={
                    "antecedent_id": "fed_cuts_2_2026",
                    "consequent_id": "fed_cuts_1_2026",
                    **base,
                },
            ),
            StrategyPreset(
                label="Conditional anomaly: Musk vs AMZN",
                description="P(Musk trillionaire) ~ P(AMZN largest). β=−25.7 cross-theme surprise.",
                inputs={"a_id": "musk_trillionaire", "b_id": "amzn_largest_jun", **base},
            ),
            StrategyPreset(
                label="Geopolitics: Iran invasion vs NATO",
                description="P(US invades Iran) vs P(Ukraine joins NATO) — strong negative co-movement.",
                inputs={"a_id": "us_invades_iran", "b_id": "ukraine_joins_nato", **base},
            ),
        ],
        event_model=[
            StrategyPreset(
                label="Iran regime change explainers",
                description="Target: P(Iran regime change Jun) explained by Netanyahu/Putin/US-invasion markets.",
                inputs={
                    "target_id": "iran_regime_jun",
                    "factor_ids": ["netanyahu_out_jun", "putin_out_jun", "us_invades_iran"],
                    "hac_lag": 5,
                    **base,
                },
            ),
            StrategyPreset(
                label="BTC ATH conditional",
                description="Target: P(BTC ATH Jun) explained by ladder of price-target markets.",
                inputs={
                    "target_id": "btc_ath_jun",
                    "factor_ids": ["btc_100k_eoy", "btc_250k_eoy", "btc_150k_h1"],
                    "hac_lag": 5,
                    **base,
                },
            ),
            StrategyPreset(
                label="Recession explained by Fed cuts",
                description="P(US recession 2026) ~ Σ Fed-cut probabilities (more cuts implied ⇒ more recession risk).",
                inputs={
                    "target_id": "us_recession_2026",
                    "factor_ids": ["fed_cuts_3_2026", "fed_cuts_4_2026", "fed_cuts_8_2026"],
                    "hac_lag": 5,
                    **base,
                },
            ),
        ],
        basket=[
            StrategyPreset(
                label="BTC strike ladder basket",
                description="All BTC ATH levels in one basket. PCA-1 = market-wide bull bias; residuals = strike-specific anomalies.",
                inputs={
                    "factor_ids": [
                        "btc_100k_eoy",
                        "btc_250k_eoy",
                        "btc_150k_h1",
                        "btc_200k_eoy",
                        "btc_ath_jun",
                    ],
                    **base,
                },
            ),
            StrategyPreset(
                label="Fed-cut path basket",
                description="P(≥1 cut) … P(≥10 cuts). PCA-1 picks up the term structure of cut expectations.",
                inputs={
                    "factor_ids": [
                        "fed_cuts_1_2026",
                        "fed_cuts_2_2026",
                        "fed_cuts_3_2026",
                        "fed_cuts_4_2026",
                        "fed_cuts_8_2026",
                    ],
                    **base,
                },
            ),
            StrategyPreset(
                label="Geopolitical incumbents basket",
                description="Out-by-Jun probabilities for global leaders. PCA-1 = systemic political risk.",
                inputs={
                    "factor_ids": ["netanyahu_out_jun", "putin_out_jun", "powell_out_may"],
                    **base,
                },
            ),
        ],
        spot_vs_implied=[
            StrategyPreset(
                label="BTC > $120k by EOM (daily)",
                description="Compare Polymarket BTC-target market to Yang-Zhang vol from Binance daily.",
                inputs={
                    "symbol": "BTCUSDT",
                    "strike": 120000,
                    "geometry": "terminal",
                    "interval": "1d",
                    "vol_window_bars": 90,
                    "n_bootstrap": 200,
                    "expiry_offset_days": 30,
                },
            ),
            StrategyPreset(
                label="BTC touch +5% in 7 days (5-min)",
                description="Short-dated touch market with 5-minute high-resolution σ̂.",
                inputs={
                    "symbol": "BTCUSDT",
                    "geometry": "one_touch_up",
                    "interval": "5m",
                    "vol_window_bars": 1000,
                    "n_bootstrap": 200,
                    "expiry_offset_days": 7,
                    "strike_pct": 1.05,
                },
            ),
            StrategyPreset(
                label="ETH terminal +10% in 30 days",
                description="ETH price-target end-of-month vs market — daily vol estimate.",
                inputs={
                    "symbol": "ETHUSDT",
                    "geometry": "terminal",
                    "interval": "1d",
                    "vol_window_bars": 90,
                    "n_bootstrap": 200,
                    "expiry_offset_days": 30,
                    "strike_pct": 1.10,
                },
            ),
        ],
        ou_bands=[
            StrategyPreset(
                label="Senate inverse OU calibration",
                description="dem/rep Senate spread → very fast κ. Optimal Bertram bands give the trader the analytic z*.",
                inputs={
                    "a_id": "dem_senate_2026",
                    "b_id": "rep_senate_2026",
                    "transaction_cost_sigma": 0.10,
                    **base,
                },
            ),
            StrategyPreset(
                label="Cross-venue Fed OU bands",
                description="Kalshi vs Polymarket Fed September cut — residual is a true cross-venue basis.",
                inputs={
                    "a_id": "k_fed_sep_cut25",
                    "b_id": "fed_cuts_3_2026",
                    "transaction_cost_sigma": 0.20,
                    **base,
                },
            ),
        ],
        granger=[
            StrategyPreset(
                label="Does Iran regime news lead Netanyahu?",
                description="Bivariate Granger between iran_regime_jun and netanyahu_out_jun.",
                inputs={
                    "a_id": "iran_regime_jun",
                    "b_id": "netanyahu_out_jun",
                    "max_lag": 5,
                    **base,
                },
            ),
            StrategyPreset(
                label="Fed-cut leader detection",
                description="Does the Kalshi cut probability lead Polymarket's, or the reverse?",
                inputs={"a_id": "k_fed_sep_cut25", "b_id": "fed_cuts_3_2026", "max_lag": 5, **base},
            ),
            StrategyPreset(
                label="BTC ATH vs ETH ATH leadership",
                description="Whose ATH-by-EOY moves first?",
                inputs={"a_id": "btc_ath_jun", "b_id": "eth_ath_eoy", "max_lag": 5, **base},
            ),
        ],
        kalman=[
            StrategyPreset(
                label="Fed strike family · Kalman β_t drift",
                description="Time-varying hedge between adjacent Fed-rate strikes. δ small ⇒ β drifts slowly, picks up regime changes.",
                inputs={
                    "a_id": "fed_cuts_3_2026",
                    "b_id": "fed_cuts_4_2026",
                    "delta": 0.0001,
                    **base,
                },
            ),
            StrategyPreset(
                label="BTC ATH ladder · δ=1e-3",
                description="Faster adaptation on BTC ATH-by-jun vs ATH-by-eoy — the cointegrating ratio drifts as resolution approaches.",
                inputs={"a_id": "btc_ath_jun", "b_id": "btc_ath_eoy", "delta": 0.001, **base},
            ),
            StrategyPreset(
                label="Senate Dem/Rep · responsive Kalman",
                description="dem/rep Senate spread with δ=0.01 — large δ chases the spread aggressively. Compare with the OU Bands tab.",
                inputs={
                    "a_id": "dem_senate_2026",
                    "b_id": "rep_senate_2026",
                    "delta": 0.01,
                    **base,
                },
            ),
        ],
    )


@router.get("/strategies/presets", response_model=PresetsResponse)
def strategies_presets() -> PresetsResponse:
    """Curated example inputs for every Strategies sub-tool.

    Each preset is hand-tuned from real scanner hits across 2026-01 → 2026-04
    on the catalog. Frontend uses them as one-click "Quick start" chips so
    users don't need to know the factor ID space.
    """
    return _build_presets()
