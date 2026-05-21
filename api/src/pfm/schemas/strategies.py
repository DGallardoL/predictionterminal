"""Schemas for /strategies/* — the 32 classical / stat-arb / risk strategies."""

from __future__ import annotations

from datetime import date as _date
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

from pfm.schemas.common import (
    FredSeriesLit,
    FredTransformLit,
    GeometryLit,
    _StrategyPairBase,
)


class ImplicationRequest(_StrategyPairBase):
    """Test the logical implication ``A ⇒ B`` ⇒ ``P(A) ≤ P(B)``."""

    antecedent_id: str = Field(description="Factor id of the *more specific* event (A in A⇒B).")
    consequent_id: str = Field(description="Factor id of the *broader* event (B in A⇒B).")
    tolerance: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Ignore gaps with P(A)-P(B) ≤ tolerance (default ≈ typical "
        "Polymarket bid-ask half-spread).",
    )


class ImplicationGapPoint(BaseModel):
    date: _date
    p_a: float
    p_b: float
    gap: float
    logit_gap: float


class ImplicationResponse(BaseModel):
    antecedent_id: str
    consequent_id: str
    n_obs: int
    verdict: Literal["consistent", "borderline", "violated", "insufficient-data"]
    n_violations: int
    violation_dates: list[_date]
    max_gap: float | None = None
    mean_gap: float | None = None
    series: list[ImplicationGapPoint]


class ConditionalRequest(_StrategyPairBase):
    """Estimate β from P_A ~ α + β·P_B + ε with HAC SE."""

    a_id: str = Field(description="Factor id for the dependent series (A).")
    b_id: str = Field(description="Factor id for the conditioning series (B).")
    hac_lag: int = Field(default=5, ge=0, le=60)


class ConditionalResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    beta: float
    beta_hac_se: float
    beta_ci_lo: float
    beta_ci_hi: float
    intercept: float
    r_squared: float
    cond_mean_when_b_high: float | None = None
    cond_mean_when_b_low: float | None = None
    n_b_high: int


class BoundsRequest(_StrategyPairBase):
    """Per-date Fréchet-Hoeffding bounds on the joint P(A∩B)."""

    a_id: str
    b_id: str


class BoundsPoint(BaseModel):
    date: _date
    p_a: float
    p_b: float
    lower: float
    upper: float
    independence: float


class BoundsResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    mean_lower: float | None = None
    mean_upper: float | None = None
    mean_width: float | None = None
    series: list[BoundsPoint]


class SpotVsImpliedRequest(BaseModel):
    """Compare a live underlying (Binance daily klines) to a market-implied
    YES-price for a price-target binary outcome."""

    symbol: str = Field(
        min_length=4,
        max_length=20,
        examples=["BTCUSDT"],
        description="Binance trading pair (uppercase). Default uses Binance "
        "spot — note that Polymarket BTC markets typically settle "
        "on Coinbase / UMA-disputed indexes, so a small basis is "
        "expected.",
    )
    strike: float = Field(
        gt=0,
        description="Target price K. For 'terminal' geometry: settles 1 if "
        "S_T ≥ K. For 'one_touch_up': 1 if M_T = max S_t ≥ K. "
        "For 'one_touch_down': 1 if min S_t ≤ K.",
    )
    expiry: _date = Field(description="Market resolution date (UTC).")
    geometry: GeometryLit = "terminal"
    market_prob: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Current market YES-price (0..1). When provided, the "
        "response includes 'edge' = market_prob − model_prob and "
        "edge_significant_95.",
    )
    drift_annual: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Annualised drift μ. Default 0 (risk-neutral, no carry "
        "for unfunded BTC spot). Override with the perp funding "
        "rate if you prefer a carry adjustment.",
    )
    interval: Literal[
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"
    ] = Field(
        default="1d",
        description="Binance kline interval. Sub-daily intervals (e.g. 5m) "
        "give a tighter, more responsive σ̂ for short-dated markets.",
    )
    vol_window_bars: int = Field(
        default=90,
        ge=10,
        le=2000,
        description="Trailing window of bars (matching ``interval``) used for "
        "σ̂ estimation. 90 daily bars ≈ 3 months; 1000 5-min bars "
        "≈ 3.5 days.",
    )
    n_bootstrap: int = Field(default=200, ge=20, le=1000)
    seed: int = 42


class SpotVsImpliedResponse(BaseModel):
    symbol: str
    interval: str
    spot: float
    strike: float
    expiry: _date
    geometry: GeometryLit
    time_years: float
    sigma_used: float
    sigma_method: str
    drift_used: float
    n_vol_bars: int
    model_prob: float
    ci_lo_90: float
    ci_hi_90: float
    ci_lo_95: float
    ci_hi_95: float
    market_prob: float | None = None
    edge: float | None = None
    edge_significant_95: bool | None = None
    n_bootstrap: int


# ---- /strategies/cointegration ---------------------------------------------


class CointegrationRequest(_StrategyPairBase):
    # Accept both ``a_id`` / ``b_id`` (technical) AND the more intuitive
    # ``factor_a`` / ``factor_b`` aliases. Users hitting the endpoint from
    # the JSON examples in the UI naturally try ``factor_a`` first; making
    # both work removes the 422 footgun without breaking existing callers.
    a_id: str = Field(validation_alias=AliasChoices("a_id", "factor_a"))
    b_id: str = Field(validation_alias=AliasChoices("b_id", "factor_b"))
    significance: float = Field(default=0.05, gt=0.0, lt=0.5)


class CointegrationSpreadPoint(BaseModel):
    date: _date
    p_a: float
    p_b: float
    spread: float
    zscore: float | None = None


class CointegrationResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    cointegrated: bool
    verdict: Literal["cointegrated", "not_cointegrated", "insufficient-data"]
    beta_hedge: float | None = None
    intercept: float | None = None
    adf_stat: float | None = None
    adf_pvalue: float | None = None
    adf_used_lag: int | None = None
    half_life_days: float | None = None
    rho: float | None = None
    series: list[CointegrationSpreadPoint]


class FredCointegrationRequest(BaseModel):
    """Request body for ``/strategies/fred-cointegration``.

    Tests cointegration between a Polymarket factor's probability series
    and a curated FRED macro series fetched via the auth-free
    ``fredgraph.csv`` endpoint.
    """

    factor_id: str = Field(description="Factor id (must exist in factors.yml).")
    fred_series: FredSeriesLit = Field(
        description="FRED series id; one of DFF/DGS2/DGS10/CPIAUCSL/UNRATE/VIXCLS.",
    )
    start: _date
    end: _date
    transform: FredTransformLit = Field(
        default="raw",
        description="Transform applied to the FRED series before the test.",
    )


class FredCointegrationResponse(BaseModel):
    """Response body for ``/strategies/fred-cointegration``."""

    factor_id: str
    fred_series: str
    n_obs: int
    adf_pvalue: float | None = None
    beta_hedge: float | None = None
    half_life_days: float | None = None
    cointegrated: bool
    verdict: Literal["cointegrated", "not_cointegrated", "insufficient-data"]
    fred_first: float | None = None
    fred_last: float | None = None
    fred_min: float | None = None
    fred_max: float | None = None


# ---- /strategies/pairs-backtest --------------------------------------------


class PairsBacktestRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    window: int = Field(default=20, ge=5, le=120)
    entry_z: float = Field(default=2.0, gt=0.0, le=6.0)
    exit_z: float = Field(default=0.5, ge=0.0, le=4.0)
    stop_z: float = Field(default=4.0, gt=0.0, le=8.0)
    annualisation: float = Field(default=252.0, gt=0.0, le=1_000_000.0)
    oos_fraction: float = Field(
        default=0.30,
        ge=0.0,
        lt=1.0,
        description="Fraction of bars held out as out-of-sample tail. "
        "Sharpe IS / Sharpe OOS lets you spot overfitting.",
    )
    max_hold_bars: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="Half-life-aware time stop: force-close any position "
        "held longer than this many bars. Defaults to None (no time stop).",
    )


class PairsTradeRecord(BaseModel):
    entry_date: _date
    exit_date: _date
    direction: int
    entry_z: float
    exit_z: float
    pnl: float
    holding_days: int
    exit_reason: Literal["mean_reversion", "stopped_out", "time_stop", "end_of_data"]


class PairsEquityPoint(BaseModel):
    date: _date
    spread: float
    zscore: float | None = None
    position: int
    pnl: float
    equity: float


class PairsBacktestResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    n_trades: int
    sharpe: float
    sortino: float
    calmar: float
    hit_rate: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    skew: float
    kurtosis: float
    mean_holding_days: float
    sharpe_is: float
    sharpe_oos: float
    oos_to_is_ratio: float
    n_obs_is: int
    n_obs_oos: int
    cointegration_passed: bool
    cointegration_pvalue: float | None = None
    half_life_days: float | None = None
    beta_hedge: float | None = None
    series: list[PairsEquityPoint]
    trades: list[PairsTradeRecord]


# ---- /strategies/event-model -----------------------------------------------


class EventModelRequest(BaseModel):
    target_id: str = Field(description="Factor id of the target probability series.")
    factor_ids: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Factor ids of the regressors (between 1 and 20).",
    )
    start: _date
    end: _date
    hac_lag: int = Field(default=5, ge=0, le=60)


class EventCoefficientOut(BaseModel):
    factor_id: str
    beta: float
    hac_se: float
    t_stat: float
    p_value: float
    ci_lo: float
    ci_hi: float
    vif: float | None = None


class EventModelSeriesPoint(BaseModel):
    date: _date
    actual: float
    predicted: float
    residual: float


class EventModelResponse(BaseModel):
    target_id: str
    factor_ids: list[str]
    n_obs: int
    intercept: float
    intercept_se: float
    coefficients: list[EventCoefficientOut]
    r_squared: float
    r_squared_adj: float
    f_statistic: float | None = None
    f_pvalue: float | None = None
    condition_number: float | None = None
    hac_lag: int
    series: list[EventModelSeriesPoint]


# ---- /strategies/basket-stat-arb -------------------------------------------


class BasketStatArbRequest(BaseModel):
    factor_ids: list[str] = Field(min_length=2, max_length=30)
    start: _date
    end: _date
    n_components: int | None = Field(default=None, ge=1, le=20)
    explained_variance_target: float = Field(default=0.70, gt=0.0, lt=1.0)
    z_window: int = Field(default=20, ge=5, le=120)


class BasketResidualPoint(BaseModel):
    date: _date
    residuals: dict[str, float]
    z_residuals: dict[str, float | None]


class BasketStatArbResponse(BaseModel):
    factor_ids: list[str]
    n_obs: int
    n_components_used: int
    explained_variance_ratio: list[float]
    loadings: list[list[float]]
    kelly_fraction_per_market: dict[str, float]
    series: list[BasketResidualPoint]


# ---- /strategies/ou-bands --------------------------------------------------


class OuBandsRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    transaction_cost_sigma: float = Field(
        default=0.10,
        ge=0.0,
        le=2.0,
        description="Round-trip cost in multiples of σ_eq. "
        "Polymarket round-trip ≈ 1-3¢ on a typical 0.1-0.3 σ spread → 0.05-0.30.",
    )


class OuBandsSeriesPoint(BaseModel):
    date: str
    spread: float
    z_score: float


class OuBandsResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    cointegrated: bool
    kappa: float | None = None
    mu: float | None = None
    sigma_eq: float | None = None
    sigma_innov: float | None = None
    half_life_bars: float | None = None
    ar1_beta: float | None = None
    z_entry_optimal: float | None = None
    z_exit_optimal: float | None = None
    expected_pnl_per_cycle_sigma: float | None = None
    expected_cycle_bars: float | None = None
    expected_pnl_per_year_sigma: float | None = None
    transaction_cost_sigma: float
    series: list[OuBandsSeriesPoint] | None = None


# ---- /strategies/granger ---------------------------------------------------


class GrangerRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    max_lag: int = Field(default=5, ge=1, le=30)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)


class GrangerLagOut(BaseModel):
    lag: int
    ssr_f_stat: float
    ssr_f_pvalue: float
    ssr_chi2_pvalue: float


class GrangerResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    direction: Literal["B_causes_A", "A_causes_B", "bidirectional", "neither"]
    best_lag_b_to_a: int | None = None
    best_pvalue_b_to_a: float | None = None
    best_lag_a_to_b: int | None = None
    best_pvalue_a_to_b: float | None = None
    lags_b_to_a: list[GrangerLagOut]
    lags_a_to_b: list[GrangerLagOut]


# ---- /strategies/kalman-hedge ----------------------------------------------


class KalmanHedgeRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    delta: float = Field(default=1e-4, gt=0.0, lt=1.0, description="State-noise/total-noise ratio.")


class KalmanHedgePoint(BaseModel):
    date: _date
    p_a: float
    p_b: float
    beta_t: float
    spread: float


class KalmanHedgeResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    delta: float
    r: float
    q: float
    log_likelihood: float
    beta_init: float
    beta_final: float
    beta_min: float
    beta_max: float
    spread_std: float
    series: list[KalmanHedgePoint]


# ---- /strategies/mean-reversion --------------------------------------------


class MeanReversionRequest(BaseModel):
    factor_id: str
    start: _date
    end: _date
    vr_q: int = Field(default=2, ge=2, le=20)


class MeanReversionResponse(BaseModel):
    factor_id: str
    n_obs: int
    hurst: float | None = None
    hurst_r_squared: float | None = None
    hurst_interpretation: Literal["mean_reverting", "random_walk", "trending", "insufficient-data"]
    vr_q: int
    vr: float | None = None
    vr_z_stat: float | None = None
    vr_p_value: float | None = None
    vr_verdict: Literal["mean_reverting", "random_walk", "momentum", "insufficient-data"]
    log_n: list[float]
    log_rs: list[float]


# ---- /strategies/auto-backtest ---------------------------------------------


class AutoBacktestRequest(BaseModel):
    theme: str | None = None
    factor_ids: list[str] | None = Field(default=None, max_length=200)
    start: _date
    end: _date
    max_pairs: int = Field(default=300, ge=10, le=2000)
    max_to_backtest: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Top-N cointegrated pairs to actually backtest (more is slow).",
    )
    coint_adf_max_p: float = Field(default=0.05, gt=0.0, lt=0.5)
    coint_half_life_max: float = Field(default=60.0, ge=1.0, le=365.0)
    window: int = Field(default=20, ge=5, le=120)
    entry_z: float = Field(default=2.0, gt=0.0, le=6.0)
    exit_z: float = Field(default=0.5, ge=0.0, le=4.0)
    stop_z: float = Field(default=4.0, gt=0.0, le=8.0)
    annualisation: float = Field(default=252.0, gt=0.0, le=1_000_000.0)


class AutoBacktestRow(BaseModel):
    a_id: str
    b_id: str
    sharpe: float
    sharpe_is: float
    sharpe_oos: float
    oos_to_is_ratio: float
    sortino: float
    calmar: float
    hit_rate: float
    max_drawdown: float
    n_trades: int
    mean_holding_days: float
    half_life_days: float | None = None
    adf_pvalue: float
    beta_hedge: float


class AutoBacktestResponse(BaseModel):
    n_factors_scanned: int
    n_coint_hits: int
    n_backtested: int
    runtime_seconds: float
    leaderboard: list[AutoBacktestRow]


# ---- /strategies/ml-predictor ----------------------------------------------


class MlPredictorRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    n_folds: int = Field(default=5, ge=2, le=10)
    n_estimators: int = Field(default=80, ge=10, le=500)
    max_depth: int = Field(default=3, ge=1, le=8)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    seed: int = 42


class MlFoldOut(BaseModel):
    fold: int
    n_train: int
    n_test: int
    test_r2: float
    test_direction_accuracy: float
    baseline_direction_accuracy: float
    information_coefficient: float


class FeatureImportanceOut(BaseModel):
    name: str
    importance: float


class MlPredictorResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    n_features: int
    feature_names: list[str]
    n_folds: int
    folds: list[MlFoldOut]
    mean_test_r2: float
    mean_direction_accuracy: float
    mean_baseline_direction_accuracy: float
    beats_baseline: bool
    mean_ic: float
    feature_importances: list[FeatureImportanceOut]
    last_prediction: float | None = None
    verdict: Literal["likely_alpha", "marginal", "no_edge", "insufficient-data"]


# ---- /strategies/patterns --------------------------------------------------


class PatternPairSpec(BaseModel):
    a_id: str
    b_id: str


class PatternsRequest(BaseModel):
    pairs: list[PatternPairSpec] = Field(min_length=2, max_length=20)
    start: _date
    end: _date
    window: int = Field(default=20, ge=5, le=120)
    entry_z: float = Field(default=2.0, gt=0.0, le=6.0)
    exit_z: float = Field(default=0.5, ge=0.0, le=4.0)
    stop_z: float = Field(default=4.0, gt=0.0, le=8.0)
    annualisation: float = Field(default=252.0, gt=0.0, le=1_000_000.0)
    days_to_resolution: int = Field(default=30, ge=5, le=120)
    n_clusters: int = Field(default=3, ge=2, le=10)


class CorrelationOut(BaseModel):
    pair_labels: list[str]
    correlation_matrix: list[list[float]]
    mean_off_diagonal: float
    max_off_diagonal: float
    most_correlated_a: str | None = None
    most_correlated_b: str | None = None
    most_correlated_rho: float | None = None
    diversification_ratio: float


class DowOut(BaseModel):
    pair: str
    means: dict[str, float]
    counts: dict[str, int]
    t_stats: dict[str, float]
    p_values: dict[str, float]
    best_day: str | None = None
    worst_day: str | None = None
    significant_days: list[str]


class PreResolutionOut(BaseModel):
    pair: str
    far_n: int
    near_n: int
    far_std: float | None = None
    near_std: float | None = None
    vol_ratio: float | None = None
    mean_shift: float | None = None
    f_stat: float | None = None
    f_p_value: float | None = None
    vol_shift_significant: bool


class ClusterOut(BaseModel):
    cluster_id: int
    pair_labels: list[str]
    centroid: dict[str, float]
    n_members: int


class PatternsResponse(BaseModel):
    n_pairs_analysed: int
    correlation: CorrelationOut
    day_of_week: list[DowOut]
    pre_resolution: list[PreResolutionOut]
    clusters: list[ClusterOut]
    silhouette_proxy: float


# ---- /strategies/info-share (Hasbrouck 1995) -------------------------------


class InfoShareRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    var_lags: int = Field(default=5, ge=1, le=30)


class InfoShareResponse(BaseModel):
    venue_a_id: str
    venue_b_id: str
    n_obs: int
    is_a_lower: float
    is_a_upper: float
    is_b_lower: float
    is_b_upper: float
    midpoint_a: float
    leader: str
    beta_cointeg: float


# ---- /strategies/regime-switching (Hamilton 1989) --------------------------


class RegimeSwitchingRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    k_regimes: int = Field(default=2, ge=2, le=4)


class RegimePoint(BaseModel):
    date: _date
    p_state1: float


class RegimeSwitchingResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    n_state0: int
    n_state1: int
    sigma_state0: float
    sigma_state1: float
    mean_state0: float
    mean_state1: float
    transition_p00: float
    transition_p11: float
    current_regime: int
    current_regime_prob: float
    verdict: Literal["tradeable", "broken"]
    series: list[RegimePoint]


# ---- /strategies/almgren-chriss (Almgren-Chriss 2001) ----------------------


class AlmgrenChrissRequest(BaseModel):
    target_position: float = Field(description="Signed shares (positive=buy, negative=sell)")
    n_intervals: int = Field(default=10, ge=2, le=200)
    time_horizon: float = Field(default=1.0, gt=0.0, le=365.0)
    sigma: float = Field(default=0.10, gt=0.0, le=10.0)
    eta: float = Field(default=0.01, gt=0.0, le=1.0)
    epsilon: float = Field(default=0.005, ge=0.0, le=1.0)
    gamma_perm: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_aversion: float = Field(default=1.0, ge=0.0, le=100.0)


class AlmgrenChrissResponse(BaseModel):
    n_intervals: int
    x_remaining: list[float]
    n_per_interval: list[float]
    kappa: float
    time_horizon: float
    expected_cost: float
    variance_cost: float
    utility: float


# ---- /strategies/cusum -----------------------------------------------------


class CusumRequest(_StrategyPairBase):
    a_id: str
    b_id: str


class CusumPoint(BaseModel):
    date: _date
    cusum: float


class CusumResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    verdict: Literal["stable", "break_detected", "insufficient-data"]
    rejected: bool
    max_abs_cusum: float | None = None
    threshold_95: float | None = None
    break_point: _date | None = None
    series: list[CusumPoint]


# ---- /strategies/walk-forward ---------------------------------------------


class WalkForwardRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    n_folds: int = Field(default=5, ge=2, le=20)
    window: int = Field(default=20, ge=5, le=120)
    entry_z: float = Field(default=2.0, gt=0.0, le=6.0)
    exit_z: float = Field(default=0.5, ge=0.0, le=4.0)
    stop_z: float = Field(default=4.0, gt=0.0, le=8.0)
    annualisation: float = Field(default=252.0, gt=0.0, le=1_000_000.0)


class WalkForwardFoldOut(BaseModel):
    fold: int
    test_start: _date
    test_end: _date
    train_sharpe: float
    test_sharpe: float
    n_train: int
    n_test: int


class WalkForwardResponse(BaseModel):
    a_id: str
    b_id: str
    n_obs: int
    n_folds: int
    folds: list[WalkForwardFoldOut]
    train_sharpe_mean: float
    test_sharpe_mean: float
    test_sharpe_median: float
    test_sharpe_min: float
    test_sharpe_max: float
    test_sharpe_std: float
    stability: Literal["stable", "borderline", "unstable"]


# ---- /strategies/sharpe-bootstrap ------------------------------------------


class SharpeBootstrapRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    window: int = Field(default=20, ge=5, le=120)
    entry_z: float = Field(default=2.0, gt=0.0, le=6.0)
    exit_z: float = Field(default=0.5, ge=0.0, le=4.0)
    stop_z: float = Field(default=4.0, gt=0.0, le=8.0)
    n_iters: int = Field(default=500, ge=50, le=2000)
    annualisation: float = Field(default=252.0, gt=0.0, le=1_000_000.0)
    seed: int = 42


class SharpeBootstrapResponse(BaseModel):
    a_id: str
    b_id: str
    sharpe_point: float
    sharpe_mean: float
    sharpe_std: float
    sharpe_ci_lo_90: float
    sharpe_ci_hi_90: float
    sharpe_ci_lo_95: float
    sharpe_ci_hi_95: float
    n_bootstrap: int
    block_size: int


# ---- /strategies/sharpe-permutation ----------------------------------------


class PermutationSharpeRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    window: int = Field(default=20, ge=5, le=120)
    entry_z: float = Field(default=2.0, gt=0.0, le=6.0)
    exit_z: float = Field(default=0.5, ge=0.0, le=4.0)
    stop_z: float = Field(default=4.0, gt=0.0, le=8.0)
    n_iters: int = Field(default=200, ge=50, le=1000)
    annualisation: float = Field(default=252.0, gt=0.0, le=1_000_000.0)
    seed: int = 42


class PermutationSharpeResponse(BaseModel):
    a_id: str
    b_id: str
    real_sharpe: float
    null_sharpes: list[float]
    null_median: float
    null_pct95: float
    p_value: float
    n_iters: int


# ---- /strategies/presets ---------------------------------------------------


class StrategyPreset(BaseModel):
    """Curated example for a given strategy, tuned from real catalog hits."""

    label: str
    description: str
    inputs: dict
    # Optional small badge rendered inside the chip (e.g. ``Sharpe 5.7``).
    # Pulled out of ``label`` so the title can stay short and the metric
    # render as a pill rather than mixing into the headline copy.
    metric: str | None = None
    # Tier classification used to colour the chip border. The frontend
    # maps ``alpha`` → gold accent, ``cross-venue`` → blue, ``standard``
    # → neutral hairline. Anything else falls back to neutral.
    tier: str | None = None


class PresetsResponse(BaseModel):
    cointegration: list[StrategyPreset]
    pairs: list[StrategyPreset]
    pair_explorer: list[StrategyPreset]
    event_model: list[StrategyPreset]
    basket: list[StrategyPreset]
    spot_vs_implied: list[StrategyPreset]
    ou_bands: list[StrategyPreset]
    granger: list[StrategyPreset]
    kalman: list[StrategyPreset] = []


# ---- /strategies/fractional-diff (Hosking / de Prado §5) ------------------


class FractionalDiffRequest(BaseModel):
    factor_id: str
    start: _date
    end: _date
    d: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Differencing exponent. If None, finds the minimal d "
        "that makes series stationary (ADF p<0.05).",
    )
    threshold: float = Field(default=1e-3, gt=0.0, le=0.1)


class FractionalDiffGridPoint(BaseModel):
    d: float
    adf_p: float
    corr_with_original: float
    n_after_filter: int


class FractionalDiffResponse(BaseModel):
    factor_id: str
    minimal_d: float | None = None
    adf_p_at_minimal_d: float | None = None
    correlation_with_original: float | None = None
    weights_width: int
    grid: list[FractionalDiffGridPoint]


# ---- /strategies/garch ----------------------------------------------------


class GarchRequest(BaseModel):
    factor_id: str
    start: _date
    end: _date


class GarchResponse(BaseModel):
    factor_id: str
    n_obs: int
    converged: bool
    is_stationary: bool
    mu: float
    omega: float
    alpha: float
    beta: float
    persistence: float
    long_run_variance: float
    log_likelihood: float
    last_sigma: float
    next_bar_sigma_forecast: float


# ---- /strategies/dfa ------------------------------------------------------


class DfaRequest(BaseModel):
    factor_id: str
    start: _date
    end: _date
    poly_order: int = Field(default=1, ge=1, le=4)


class DfaResponse(BaseModel):
    factor_id: str
    n_obs: int
    alpha_dfa: float | None = None
    r_squared_log_log: float | None = None
    interpretation: Literal[
        "mean_reverting", "random_walk", "persistent", "non_stationary", "insufficient-data"
    ]
    log_n: list[float]
    log_f: list[float]


# ---- /strategies/triple-barrier (de Prado 2018) ---------------------------


class TripleBarrierRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    window: int = Field(default=20, ge=5, le=120)
    entry_z: float = Field(default=2.0, gt=0.0, le=6.0)
    profit_target_sigma: float = Field(default=2.0, gt=0.0, le=10.0)
    stop_loss_sigma: float = Field(default=4.0, gt=0.0, le=10.0)
    time_horizon_bars: int = Field(default=10, ge=1, le=200)
    annualisation: float = Field(default=252.0, gt=0.0, le=1_000_000.0)


class TripleBarrierResponse(BaseModel):
    a_id: str
    b_id: str
    n_trades: int
    n_profit_hits: int
    n_stop_hits: int
    n_time_hits: int
    total_pnl: float
    sharpe: float
    profit_hit_rate: float
    avg_holding_bars: float


# ---- /strategies/distance-method (Gatev-G-R 2006) -------------------------


class DistanceMethodRequest(_StrategyPairBase):
    a_id: str
    b_id: str
    formation_fraction: float = Field(default=0.5, ge=0.1, le=0.9)
    entry_sigma: float = Field(default=2.0, gt=0.0, le=6.0)
    exit_sigma: float = Field(default=0.0, ge=0.0, le=4.0)
    annualisation: float = Field(default=252.0, gt=0.0, le=1_000_000.0)


class DistanceMethodResponse(BaseModel):
    a_id: str
    b_id: str
    formation_ssd: float
    formation_sigma: float
    n_trading_bars: int
    n_trades: int
    trade_pnl: float
    sharpe: float


# ---- /strategies/robust-validation ---------------------------------------


class RobustValidationRequest(BaseModel):
    """Validates the alpha of a portfolio of pair trades against a battery
    of robustness tests."""

    pairs: list[PortfolioPairSpec] = Field(min_length=1, max_length=20)
    start: _date
    end: _date
    target_per_leg_vol: float = Field(default=0.10, gt=0.0, le=2.0)
    annualisation: float = Field(default=252.0, gt=0.0, le=1_000_000.0)
    n_trials_searched: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Multiple-testing correction: how many strategies were "
        "evaluated before settling on this portfolio. Conservative "
        "default 100. Affects deflated Sharpe ratio.",
    )
    seed: int = 42


class CostSensitivityPoint(BaseModel):
    cost_bps: float
    net_sharpe: float


class RobustValidationResponse(BaseModel):
    portfolio_sharpe: float
    n_obs: int
    overall_verdict: Literal["STRONG ALPHA", "MARGINAL ALPHA", "WEAK / SUSPECT", "NOISE / OVERFIT"]
    n_tests_passed: int  # of 5 tests (Lo, bootstrap, perm, OOS, deflated)

    # Lo (2002) asymptotic
    lo_sharpe: float
    lo_se: float
    lo_z_stat: float
    lo_p_value: float
    lo_ci_lo_95: float
    lo_ci_hi_95: float

    # Block bootstrap
    bootstrap_ci_lo_90: float
    bootstrap_ci_hi_90: float
    bootstrap_ci_lo_95: float
    bootstrap_ci_hi_95: float

    # Permutation
    permutation_p_value: float
    permutation_null_median: float
    permutation_null_pct95: float

    # Cost sensitivity
    cost_sensitivity: list[CostSensitivityPoint]
    break_even_cost_bps: float

    # Out-of-time
    out_of_time_train_sharpe: float
    out_of_time_test_sharpe: float
    out_of_time_ratio: float
    out_of_time_verdict: str

    # Deflated Sharpe
    deflated_sharpe: float
    deflated_p_value: float
    expected_max_sharpe_under_null: float


# ---- /strategies/portfolio -------------------------------------------------


class PortfolioPairSpec(BaseModel):
    a_id: str
    b_id: str
    signal_type: Literal["zscore", "bollinger_15"] = "zscore"
    window: int = Field(default=20, ge=5, le=120)


class PortfolioRequest(BaseModel):
    pairs: list[PortfolioPairSpec] = Field(min_length=2, max_length=20)
    start: _date
    end: _date
    target_per_leg_vol: float = Field(default=0.10, gt=0.0, le=2.0)
    walk_forward_folds: int = Field(default=5, ge=2, le=20)


class PortfolioResponse(BaseModel):
    n_pairs: int
    pair_labels: list[str]
    weights: dict[str, float]
    individual_sharpes: dict[str, float]
    correlation_matrix: list[list[float]]
    n_obs: int
    portfolio_sharpe: float
    portfolio_sortino: float
    portfolio_calmar: float
    portfolio_max_drawdown: float
    portfolio_var_95: float
    portfolio_cvar_95: float
    portfolio_skew: float
    oos_sharpe_mean: float | None = None
    oos_sharpe_std: float | None = None
    oos_sharpe_min: float | None = None


# ---- /strategies/factor-model-pro ------------------------------------------


class FactorModelProRequest(BaseModel):
    target_id: str
    factor_ids: list[str] = Field(min_length=1, max_length=30)
    start: _date
    end: _date
    estimator: Literal["ols", "ridge", "lasso", "elastic_net"] = "ols"
    alpha: float = Field(default=1.0, gt=0.0, le=100.0)
    transform: Literal["raw", "logit"] = "raw"
    use_pca: bool = False
    pca_explained_variance_target: float = Field(default=0.90, gt=0.0, lt=1.0)
    n_cv_folds: int = Field(default=5, ge=2, le=20)
    bootstrap_iters: int = Field(default=200, ge=50, le=1000)
    seed: int = 42


class CoefficientProOut(BaseModel):
    factor_id: str
    beta: float
    beta_std_across_folds: float | None = None
    stability_ratio: float | None = None
    is_zeroed: bool
    significance: str | None = None


class ResidualDiagnosticsOut(BaseModel):
    ljung_box_p: float | None = None
    jarque_bera_p: float | None = None
    arch_lm_p: float | None = None
    durbin_watson: float
    residual_std: float
    residual_skew: float
    residual_kurtosis: float
    well_specified: bool


class FactorModelProResponse(BaseModel):
    target_id: str
    estimator: str
    transform: str
    use_pca: bool
    n_obs: int
    n_factors: int
    coefficients: list[CoefficientProOut]
    intercept: float
    r_squared_is: float
    r_squared_cv: float
    r_squared_cv_std: float
    r_squared_ci_lo_95: float
    r_squared_ci_hi_95: float
    diagnostics: ResidualDiagnosticsOut
    pca_explained_variance: list[float] | None = None
    n_zeroed_factors: int
    overfit_flag: bool
