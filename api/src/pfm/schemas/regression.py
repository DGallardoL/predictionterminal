"""Schemas for /fit and /attribution."""

from __future__ import annotations

from datetime import date as _date

from pydantic import BaseModel, ConfigDict, Field

from pfm.schemas.common import (
    TICKER_PATTERN,
    AlignmentLit,
    CustomFactor,
    RegressionLit,
    ReturnTypeLit,
)


class FitRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, examples=["NVDA"], pattern=TICKER_PATTERN)
    # Cap factor count to bound the per-request fan-out cost. A typical fit
    # uses 1–5 factors; the WOW-hero best-of-30 picker tops out around 30.
    # 50 leaves comfortable headroom while preventing a single request from
    # triggering a 10k-slug Polymarket fetch storm.
    factors: list[str] = Field(
        default_factory=list,
        max_length=50,
        examples=[["fed_cuts_ge_2_2026"]],
    )
    custom_factors: list[CustomFactor] = Field(default_factory=list, max_length=50)
    start: _date
    end: _date
    return_type: ReturnTypeLit = "log"
    regression: RegressionLit = "hac"
    alignment: AlignmentLit = "strict"
    # ── HAC bandwidth override (opt-in) ───────────────────────────────────
    # Default ``None`` keeps the prior behaviour (Andrews 1991 automatic
    # bandwidth). Setting an explicit lag is useful for sensitivity checks
    # and for power users who know their autocorrelation structure.
    hac_lag: int | None = Field(
        default=None,
        ge=0,
        le=200,
        description=(
            "Newey-West HAC bandwidth. None ⇒ Andrews (1991) automatic "
            "bandwidth. Must be < n_obs after factor alignment, else 422."
        ),
    )
    # ── Advanced toggles (opt-in) ─────────────────────────────────────────
    lag: int = Field(
        default=0, ge=0, le=20, description="Shift factors by k days for predictive regression."
    )
    quantile: float = Field(
        default=0.5, gt=0.0, lt=1.0, description="Quantile for quantile regression."
    )
    pca_components: int | None = Field(
        default=None, ge=1, le=10, description="If set, reduce factors via PCA."
    )
    # ── Validation toggles ────────────────────────────────────────────────
    oos_test_fraction: float = Field(default=0.0, ge=0.0, lt=0.5)
    bootstrap_iters: int = Field(default=0, ge=0, le=2000)
    rolling_window: int | None = Field(default=None, ge=20, le=300)
    granger_max_lag: int = Field(default=0, ge=0, le=10)
    permutation_iters: int = Field(default=0, ge=0, le=500)
    # Predict α (residual after SPY market β) instead of raw return.
    # Verified empirically (n=210 experiments) to ~double test R² for several
    # tickers (e.g. COIN +0.36 vs baseline +0.16). Best when ticker has clear
    # market-beta exposure that isn't related to the Polymarket factors.
    residualize_market: bool = False


class FactorEstimateOut(BaseModel):
    id: str
    beta: float
    std_err: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float


class ModelStatsOut(BaseModel):
    alpha: float
    r_squared: float
    r_squared_adj: float
    f_stat: float
    f_pvalue: float
    residual_std: float


class DiagnosticsOut(BaseModel):
    vif: dict[str, float]
    durbin_watson: float
    hac_lag: int
    adf_stat: float | None = None
    adf_pvalue: float | None = None
    kpss_stat: float | None = None
    kpss_pvalue: float | None = None


class TimeSeriesPoint(BaseModel):
    date: _date
    observed: float
    predicted: float
    residual: float
    factor_prices: dict[str, float]
    factor_delta_logits: dict[str, float]


class FactorTracePoint(BaseModel):
    """One observation of a factor's underlying probability path."""

    date: _date
    price: float


class OosOut(BaseModel):
    train_n: int
    test_n: int
    train_r2: float
    test_r2: float
    test_dates: list[_date]
    test_observed: list[float]
    test_predicted: list[float]


class BootstrapCi(BaseModel):
    factor_id: str
    ci_low: float
    ci_high: float
    mean: float
    std: float


class RollingBetaPoint(BaseModel):
    date: _date
    betas: dict[str, float]


class RollingBetaCiPoint(BaseModel):
    """Single point on a per-factor rolling-β series with HAC 95% CIs.

    Used by the enriched ``/fit`` response (``rolling_betas_ci``) so the UI
    can plot a per-factor β(t) line with shaded confidence band without
    re-fitting on the client. Each window's ``ci_lo`` / ``ci_hi`` are the
    Newey-West 95% bounds computed at the same lag as the headline fit.
    """

    date: _date
    beta: float
    ci_lo: float
    ci_hi: float


class OosRSquared(BaseModel):
    """Walk-forward out-of-sample R² block emitted by the enriched ``/fit``.

    Reports the median (across folds) of the test-fold R², plus train/test
    sizes and fold count. Useful for spotting in-sample overfit: when the
    headline ``r_squared`` is high but ``oos_r_squared.value`` collapses,
    the model is memorising rather than generalising.
    """

    value: float = Field(description="Median test-fold R² across the walk-forward folds.")
    n_train: int = Field(description="Average training-fold size.")
    n_test: int = Field(description="Average test-fold size.")
    fold_count: int = Field(description="Number of walk-forward folds executed.")
    per_fold: list[float] = Field(
        default_factory=list,
        description="Per-fold test R² values, in chronological order.",
    )


class ResidualAnnotation(BaseModel):
    """Top-|e_t| residual outlier with the factor that drove it most.

    Surfaces the few dates where the model fit worst, alongside which
    factor's contribution Δlogit·β was the largest (in absolute value)
    on that date. Lets the user spot event-driven outliers (e.g. a Fed
    decision day) without scanning the full residual series.
    """

    date: _date
    residual: float
    magnitude: float = Field(description="Absolute value of the residual.")
    factor_attribution: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-factor contribution Δlogit·β on that date. The factor "
            "with the largest |contribution| is the one most likely to "
            "explain the miss."
        ),
    )
    top_factor: str | None = Field(
        default=None,
        description="Id of the factor with the largest |contribution| on this date.",
    )
    news_links: list[str] = Field(
        default_factory=list,
        description=(
            "External-search URLs (TICKER + date) the user can click to "
            "find the news that may explain this residual. Plain links — "
            "the frontend deep-links them under each annotation."
        ),
    )


# ── Rigour-pack additions (2026-05-15) ───────────────────────────────────────
# These types power the six new defensive features:
#   1. overfit_risk_flags   – structured guardrails on n/k, clipping, themes
#   2. multitest_hint       – BH-FDR threshold derived from session test count
#   3. extended FactorCoverageOut + predicted_window_n_obs (pre-flight)
#   4. extended OosRSquared with skipped_reason (replaces silent null)
#   5. regime_changes       – per-factor structural-break detection
#   6. ResidualAnnotation.news_links (deep-link search URLs)


class OverfitRiskFlag(BaseModel):
    """One structured guardrail for the user about fit reliability.

    ``level`` lets the UI pick a colour (high=red, medium=yellow, low=blue);
    ``code`` is a stable machine-readable enum so dashboards can filter.
    The ``message`` is human-readable plain English ready for display.
    """

    level: str = Field(description="One of: high | medium | low.")
    code: str = Field(
        description=(
            "Stable enum: low_dof | moderate_dof | high_clipping | "
            "sign_inconsistent | theme_mismatch."
        ),
    )
    message: str = Field(description="One-sentence plain-English explanation.")


class MultitestHint(BaseModel):
    """Session-level multiple-testing context for the caller.

    Driven by the ``X-Session-Test-Count`` request header (default 1). Lets
    the UI render a "your effective threshold is α/N" advisory next to the
    p-values so the user doesn't fool themselves after running 30 fits.

    Notes
    -----
    The hint uses Bonferroni-style ``α/N`` for the readout because it's the
    quantity a quant intuitively expects when comparing a single p to a
    family of tests. The proper Benjamini-Hochberg q-thresholds depend on
    the rank order of *all* p-values in the family, which the server has no
    way to know across calls — the caller can use ``/quant/multitest/bh``
    for the exact procedure.
    """

    tests_this_session: int = Field(
        ge=1,
        description="Echo of the X-Session-Test-Count header (default 1).",
    )
    bh_q_threshold: float = Field(
        gt=0.0,
        le=1.0,
        description="Per-test Bonferroni-style threshold α/N at α=0.05.",
    )
    message: str = Field(
        description=("Human-readable advisory. UI displays verbatim under the p-value column."),
    )


class OosRSquaredSkipped(BaseModel):
    """Explicit "OOS skipped" state — replaces the silent ``null`` shape.

    Returned in place of :class:`OosRSquared` when the helper opted out
    (e.g. n_obs below the walk-forward floor). Lets the UI render a clear
    "skipped — needs ≥100 obs" badge instead of silently hiding the field.

    Backward compatibility: clients that read ``oos_r_squared.value`` still
    work because the union of OosRSquared | OosRSquaredSkipped | None
    keeps ``None`` as a fallback if even the skipped block can't be built.
    """

    value: None = Field(
        default=None,
        description="Always null in the skipped state.",
    )
    n_train: None = Field(default=None)
    n_test: None = Field(default=None)
    fold_count: None = Field(default=None)
    skipped: bool = Field(default=True)
    skipped_reason: str = Field(
        description=(
            "Human-readable reason the walk-forward was not run, e.g. "
            "'n_obs=32 < min_n_for_walk_forward=100'."
        ),
    )


class RegimeChangeOut(BaseModel):
    """One per-factor structural-break finding from :mod:`pfm.regression_regime`.

    Reported only when the per-factor split-sample test trips the p-value
    threshold (default 0.10). Empty list when nothing was detected.
    """

    factor_id: str
    breakpoint_date: str = Field(
        description=(
            "ISO date (YYYY-MM-DD) of the split point. The 'post' half "
            "starts at this date inclusive."
        ),
    )
    pre_beta: float
    post_beta: float
    sign_flipped: bool = Field(
        description=(
            "True when sign(pre_beta) != sign(post_beta) AND both "
            "magnitudes exceed 0.01 (avoids near-zero noise flips)."
        ),
    )
    chow_stat: float
    p_value: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Two-sided p for the per-factor beta difference, smallest "
            "across the candidate breakpoints."
        ),
    )


class PcaSummary(BaseModel):
    """Quick PCA on the design matrix — first ``n_components`` of variance.

    Helpful for users who pile in many correlated factors and want a one-glance
    view of how many independent dimensions actually drive the regressors.
    """

    n_components: int
    explained_variance_ratio: list[float]
    top_loadings: dict[int, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "For each component index (0-based) the factor → loading dict "
            "(top 5 factors by |loading|). Lets the UI label PCs as "
            '"mostly Fed factors" or "mix of A and B".'
        ),
    )


class GrangerLag(BaseModel):
    lag: int
    f_stat: float
    p_value: float


class GrangerOut(BaseModel):
    factor_id: str
    by_lag: list[GrangerLag]


class PcaOut(BaseModel):
    components: list[str]
    explained_variance: list[float]
    loadings: dict[str, dict[str, float]]


class LiveSignalOut(BaseModel):
    """Forward-looking prediction from the just-fit model.

    Computed as ``predicted = α + Σᵢ βᵢ · Δlogit_today_i`` using the last
    available Δlogit per factor (final non-null row of the design matrix).
    Carries the OLS prediction standard error and the 95 % CI so the UI can
    colour the number and surface the uncertainty. Set to ``None`` when the
    fit window has no usable last row (e.g. every factor went stale).
    """

    predicted_return: float = Field(
        description=(
            "Model-implied next-period return on the latest Δlogit row. "
            "Sign tells you direction; magnitude is in the same units as "
            "the regression target (log-return when return_type='log')."
        ),
    )
    std_err: float = Field(
        description=(
            "OLS prediction standard error at the latest design row. "
            "Reflects parameter uncertainty (β's HAC SE) and intercept "
            "noise; does NOT include unforecastable next-period shock."
        ),
    )
    ci_95_lo: float = Field(description="Lower 95% bound (predicted - 1.96·std_err).")
    ci_95_hi: float = Field(description="Upper 95% bound (predicted + 1.96·std_err).")
    edge_bp: float = Field(description="|predicted_return| expressed in basis points (1e4 ·).")
    latest_date: _date = Field(
        description="Date of the design-matrix row used to compute the signal."
    )
    latest_factor_logits: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-factor Δlogit on ``latest_date`` so the caller can see "
            "which markets moved into the prediction."
        ),
    )
    low_confidence: bool = Field(
        default=False,
        description=(
            "True when verdict is 'weak_fit' or 'underpowered' — the UI "
            "should show a 'illustration only' caveat."
        ),
    )


class BacktestPoint(BaseModel):
    """One day on the pseudo-equity curve."""

    date: _date
    predicted: float
    actual: float
    position: float = Field(description="Signed exposure in [-1, +1]; +1 long, -1 short, 0 flat.")
    pnl: float = Field(description="Daily P&L net of transaction cost (in return units).")
    equity: float = Field(description="Cumulative equity, starting from 1.0.")


class PseudoBacktestOut(BaseModel):
    """In-sample, naive-trader replay of the fitted model.

    For each day in the fit window, take ``position = sign(predicted)``,
    earn ``actual × position`` minus a per-trade transaction cost, and
    compound. Reports the resulting equity curve and four headline stats.

    This is deliberately naive — no out-of-sample split, no slippage, no
    capacity model, no factor latency. It's an "if you'd traded the signal
    in this window" sanity check. Use ``oos_r_squared`` for generalisation
    evidence.
    """

    equity_curve: list[BacktestPoint] = Field(
        description="Daily {date, predicted, actual, position, pnl, equity}.",
    )
    total_return: float = Field(description="Final equity minus 1.0 (cumulative simple return).")
    annualized_sharpe: float = Field(
        description="sqrt(252) · mean(pnl) / std(pnl). 0 when std is 0."
    )
    max_drawdown: float = Field(description="Largest peak-to-trough equity drop (negative number).")
    hit_rate: float = Field(
        description="Fraction of days where sign(predicted) == sign(actual).",
        ge=0.0,
        le=1.0,
    )
    n_trades: int = Field(description="Number of position changes (sign flips or flat→position).")
    transaction_cost_bp: float = Field(
        description="Per-trade cost charged in basis points (default 5)."
    )
    note: str = Field(
        default=(
            "Naive in-sample replay. Ignores factor latency, fees beyond "
            "the flat per-trade cost, slippage, and capacity. Illustrative "
            "only — confirm with out-of-sample data before acting."
        ),
    )


class FactorContributionOut(BaseModel):
    """Leave-one-out R² impact for a single factor.

    Re-fits the model without this factor and reports how much R² the full
    model wins by keeping it. ``delta_r_squared`` close to zero means the
    factor is redundant given the others; large positive values mean it is
    doing real explanatory work.
    """

    factor_id: str
    delta_r_squared: float = Field(
        description=(
            "R²(full) - R²(without this factor). Positive = factor helps. "
            "Values <= 0.01 typically indicate the factor is redundant; "
            "values >= 0.05 indicate a meaningful contributor."
        ),
    )
    share_of_explained_r_squared: float = Field(
        description=(
            "delta_r_squared / sum(delta_r_squared across factors), clipped "
            "to [0, 1]. Useful as a 100% share-of-voice bar."
        ),
        ge=0.0,
        le=1.0,
    )


class FitResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    n_obs: int
    start: _date
    end: _date
    epsilon: float
    return_type: ReturnTypeLit
    regression: RegressionLit
    alignment: AlignmentLit
    lag: int = 0
    pca: PcaOut | None = None
    model: ModelStatsOut
    factors: list[FactorEstimateOut]
    diagnostics: DiagnosticsOut
    time_series: list[TimeSeriesPoint]
    factor_traces: dict[str, list[FactorTracePoint]]
    # ── Defensive / diagnostic fields (additive — backward compatible) ───
    # n_obs_used == final n after all transformations (lag, pca, dropna).
    # n_obs_dropped == raw equity-return obs minus n_obs_used; useful for
    # spotting silently-truncated windows.
    n_obs_used: int = 0
    n_obs_dropped: int = 0
    # Total clipping events across every factor in the fit window. A high
    # number means epsilon may be too aggressive (the clipping floor is
    # masking real Δlogit signal) — see also ``factor_metadata`` for a
    # per-factor breakdown.
    clipping_events: int = 0
    # Free-form list of human-readable diagnostic warnings: collinearity,
    # short windows, NaN-heavy factors, ticker gaps, etc.
    warnings: list[str] = Field(default_factory=list)
    # Per-factor metadata: is_probability, source, raw n_obs.
    factor_metadata: dict[str, FactorMetadataOut] = Field(default_factory=dict)
    # ── Headline summary (1-line, human-readable) ────────────────────────
    # Computed server-side so all clients (web, curl, Slack bots) get the
    # same one-sentence readout instead of each rolling its own.
    summary: str = Field(
        default="",
        description=(
            "One-line plain-English readout: 'K factors fit with R²=X; "
            "N significant at p<0.05; M high-VIF warning'."
        ),
    )
    # ── Verdict pill — single quality flag based on existing diagnostics ─
    # well_specified  : R²adj >= 0.05 AND >=1 significant factor AND max VIF < 5
    # weak_fit        : R²adj < 0.02 (model explains essentially nothing)
    # collinear       : max VIF >= 5 (interpretation contaminated by correlation)
    # underpowered    : n_obs < 30 or n_obs <= 3 * k (HAC SEs unreliable)
    # otherwise       : "borderline" (a middling fit with no major flag)
    verdict: str = Field(
        default="borderline",
        description=(
            "Single-word fit-quality flag: well_specified | weak_fit | "
            "collinear | underpowered | borderline."
        ),
    )
    # ── Top significant factors, sorted by |t-stat| desc (p<0.05) ────────
    # Lets callers (e.g. the WOW hero auto-attribution) pluck the loadings
    # that actually matter without re-iterating + re-sorting client-side.
    top_significant: list[str] = Field(
        default_factory=list,
        description="Factor ids with p < 0.05, sorted by |t-stat| descending.",
    )
    # ── Auto-prune collinear factors (only set when prune_collinear=True)
    # Iteratively drops the factor with the highest VIF until all VIF < 5.
    # The dropped ids are surfaced here + the final fit uses the reduced set.
    auto_pruned: list[str] = Field(
        default_factory=list,
        description=(
            "Factor ids dropped by the auto-collinearity pruner. Empty when "
            "prune_collinear=False or when no factor exceeded VIF >= 5."
        ),
    )
    # ── Optional advanced outputs (only populated when requested) ────────
    oos: OosOut | None = None
    bootstrap: list[BootstrapCi] | None = None
    rolling_betas: list[RollingBetaPoint] | None = None
    granger: list[GrangerOut] | None = None
    factor_stationarity: list[FactorStationarity] | None = None
    permutation: PermutationResult | None = None

    # ── Always-on enrichments (additive — backward compatible) ───────────
    # ``rolling_betas_ci`` mirrors ``rolling_betas`` but adds 95% CIs and
    # is keyed by factor id so the UI can plot one trace per factor with
    # a shaded band. Skipped (empty dict) when n_obs < 90 since a 60d
    # rolling window leaves too little headroom for stable HAC SEs.
    rolling_betas_ci: dict[str, list[RollingBetaCiPoint]] = Field(
        default_factory=dict,
        description=(
            "Per-factor 60-day rolling β with HAC 95% CIs. Skipped (empty "
            "dict) when n_obs < 90; downsampled to ≤200 points per factor."
        ),
    )
    # ``oos_r_squared`` is the walk-forward generalisation block. When the
    # helper opts out (e.g. n_obs < 100) we now return an explicit
    # ``OosRSquaredSkipped`` block carrying ``skipped_reason`` instead of a
    # silent ``null`` — the UI can render the reason inline.
    oos_r_squared: OosRSquared | OosRSquaredSkipped | None = Field(
        default=None,
        description=(
            "Walk-forward 5-fold OOS R² (median across folds). Returns an "
            "OosRSquaredSkipped block (with ``skipped_reason``) when n_obs "
            "< 100 — never silently null."
        ),
    )
    # Top 5 |residual| dates with per-factor contribution attribution.
    residual_annotations: list[ResidualAnnotation] = Field(
        default_factory=list,
        description=(
            "Up to 5 top-|residual| dates with per-factor Δlogit·β "
            "contribution attribution. Useful for event-driven outliers."
        ),
    )
    # Pearson r between every pair of factors used in the fit (Δlogit
    # series). Limited to the first 30 factors so the payload stays small.
    factor_correlation_matrix: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Pearson r between Δlogit series of factors (capped at 30 "
            "factors). Helps spot collinearity beyond VIF."
        ),
    )
    # Quick PCA summary on the design matrix. Skipped (None) when only
    # one factor is in the design.
    pca_summary: PcaSummary | None = Field(
        default=None,
        description=(
            "PCA explained-variance + top loadings on the design matrix. "
            "Skipped when k=1 (one factor, no decomposition possible)."
        ),
    )
    # Plain-English next-step hint based on the verdict — meant to be
    # displayed verbatim in the UI.
    next_step_hint: str = Field(
        default="",
        description=(
            "One-sentence actionable suggestion derived from ``verdict``. "
            "Frontend displays this verbatim under the verdict pill."
        ),
    )
    # ── WOW features (additive, nullable — backward compatible) ──────────
    # Implied tradeable signal computed from the most-recent Δlogit per
    # factor + the just-fit coefficients. Null when no usable last row
    # exists in the design matrix (all factors stale on the end date).
    live_signal: LiveSignalOut | None = Field(
        default=None,
        description=(
            "Forward-looking signal: predicted next-period return using "
            "the latest Δlogit row + the fitted β. Null when the design "
            "matrix has no usable last row."
        ),
    )
    # Naive daily-rebalanced in-sample backtest of sign(predicted) over the
    # fit window, charging a flat per-trade cost. Skipped (null) when
    # n_obs < 30 because the headline stats are not meaningful below that.
    pseudo_backtest: PseudoBacktestOut | None = Field(
        default=None,
        description=(
            "Daily-rebalanced replay of sign(predicted) over the fit "
            "window with a flat transaction cost. Skipped when n_obs < 30."
        ),
    )
    # Leave-one-out R² impact per factor. Lets the UI rank "which factors
    # are doing the work". Skipped (null) when k == 1 since the LOO refit
    # would be a constant.
    factor_contributions: list[FactorContributionOut] | None = Field(
        default=None,
        description=(
            "Per-factor leave-one-out R² impact, sorted descending. Null "
            "when the fit has only one factor (LOO is undefined)."
        ),
    )
    # ── Rigour pack (2026-05-15) — additive, backward-compatible ─────────
    # Structured guardrails that surface common overfit / data-quality
    # gotchas the user would otherwise miss (low n/k, heavy clipping,
    # sign-inconsistent factors with the same economic direction, theme
    # mismatches between ticker and selected factors).
    overfit_risk_flags: list[OverfitRiskFlag] = Field(
        default_factory=list,
        description=(
            "Structured warnings about overfit risk — n/k ratio, clipping "
            "saturation, sign inconsistency, theme mismatch. UI shows "
            "these as colored chips above the regression table."
        ),
    )
    # Bonferroni-style multiple-testing advisory derived from the
    # X-Session-Test-Count header (default 1).
    multitest_hint: MultitestHint | None = Field(
        default=None,
        description=(
            "Session-level multiple-testing context: tests run this "
            "session and the resulting α/N threshold."
        ),
    )
    # Per-factor structural-break detection. Empty list when nothing
    # tripped the threshold or when n_obs < 60.
    regime_changes: list[RegimeChangeOut] = Field(
        default_factory=list,
        description=(
            "Per-factor structural-break findings (Chow-style split-sample "
            "z-test on betas). Empty when nothing trips p<0.10 or n_obs < 60."
        ),
    )


# ---- /attribution -----------------------------------------------------------


class AttributionRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    factors: list[str] = Field(default_factory=list, max_length=50)
    custom_factors: list[CustomFactor] = Field(default_factory=list, max_length=50)
    start: _date
    end: _date
    date: _date = Field(
        description="Calendar date (UTC) to attribute, must lie within [start, end]."
    )
    return_type: ReturnTypeLit = "log"
    regression: RegressionLit = "hac"
    alignment: AlignmentLit = "strict"


class ContributionOut(BaseModel):
    id: str
    delta_logit: float | None = None
    beta: float | None = None
    contribution: float


class AttributionResponse(BaseModel):
    date: _date
    observed_return: float
    predicted_return: float
    residual: float
    contributions: list[ContributionOut]


class FactorStationarity(BaseModel):
    factor_id: str
    adf_pvalue: float | None = None
    kpss_pvalue: float | None = None


class PermutationRequest(BaseModel):
    """Standalone permutation-test request — same shape as FitRequest minus
    the optional analyses. Returns a `PermutationResult`."""

    ticker: str = Field(min_length=1, max_length=10, examples=["NVDA"], pattern=TICKER_PATTERN)
    factors: list[str] = Field(default_factory=list)
    custom_factors: list[CustomFactor] = Field(default_factory=list)
    start: _date
    end: _date
    return_type: ReturnTypeLit = "log"
    regression: RegressionLit = "hac"
    alignment: AlignmentLit = "strict"
    residualize_market: bool = False
    n_iters: int = Field(default=50, ge=10, le=500)
    seed: int = 42
    test_fraction: float = Field(default=0.20, gt=0.0, lt=0.5)


class PermutationResult(BaseModel):
    """Output of the permutation test."""

    real_test_r2: float
    null_test_r2s: list[float]
    null_median: float
    null_pct95: float
    null_max: float
    p_value: float
    n_iters_completed: int


class FactorMetadataOut(BaseModel):
    """Per-factor metadata returned alongside the regression fit.

    Lets the client see, at a glance, which factors are probabilities (so
    Δlogit applies) vs level series (Δ first-difference), and how many raw
    observations each factor contributed before the inner-join with the
    equity returns. Helps diagnose why a window has fewer obs than expected.
    """

    is_probability: bool
    source: str
    n_obs: int
    clipping_events: int = 0


# ---- /fit/preview -----------------------------------------------------------


class FitPreviewRequest(BaseModel):
    """Fast pre-flight check for /fit — same shape, no regression run."""

    ticker: str = Field(min_length=1, max_length=10, examples=["NVDA"], pattern=TICKER_PATTERN)
    factors: list[str] = Field(default_factory=list)
    custom_factors: list[CustomFactor] = Field(default_factory=list)
    start: _date
    end: _date
    return_type: ReturnTypeLit = "log"
    alignment: AlignmentLit = "strict"


class FactorCoverageOut(BaseModel):
    """Per-factor coverage stats for /fit/preview.

    The ``n_obs_available`` / ``n_obs_in_window`` split lets the user spot a
    factor that exists upstream but barely overlaps with the requested
    window — the most common cause of an inner-join collapse on /fit.
    """

    factor_id: str
    n_obs: int
    # Aliases for the rigour-pack: ``n_obs_available`` is total raw obs the
    # source has on this factor; ``n_obs_in_window`` is what survives the
    # [start, end] filter. ``n_obs`` is kept as the legacy alias of
    # ``n_obs_in_window`` so existing clients don't break.
    n_obs_available: int = Field(
        default=0,
        description=(
            "Total raw observations the source returned for this factor "
            "(before the [start, end] window filter)."
        ),
    )
    n_obs_in_window: int = Field(
        default=0,
        description=(
            "Observations remaining after the [start, end] window filter — "
            "the value the inner-join with equity will use."
        ),
    )
    first_date: _date | None = None
    last_date: _date | None = None
    coverage_pct: float = Field(ge=0.0, le=1.0)
    is_probability: bool
    source: str


# ---- /factors/suggest-for-ticker -------------------------------------------


class SuggestForTickerRequest(BaseModel):
    """Request body for the smart-factor-picker endpoint.

    Computes |Pearson r| between the ticker's log returns and each curated
    factor's Δlogit, then returns the top ``top_k`` ranked by |r|. Cached
    for 1 h per (ticker, lookback_days) so the UI's "suggest factors" panel
    is snappy on repeat loads.
    """

    ticker: str = Field(min_length=1, max_length=10, examples=["NVDA"], pattern=TICKER_PATTERN)
    lookback_days: int = Field(
        default=90,
        ge=20,
        le=720,
        description="Trailing window in calendar days for the correlation scan.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top factors to return, ranked by |Pearson r|.",
    )
    min_n_obs: int = Field(
        default=30,
        ge=10,
        le=200,
        description="Skip factors that have fewer than this many overlapping obs.",
    )


class SuggestForTickerItem(BaseModel):
    """One row of the smart-picker response."""

    factor_id: str
    name: str
    source: str
    theme: str | None = None
    r: float = Field(description="Pearson correlation (signed) for context.")
    abs_r: float = Field(description="Sort key — |Pearson r|.")
    n_obs: int


class SuggestForTickerResponse(BaseModel):
    ticker: str
    lookback_days: int
    n_factors_scanned: int = Field(
        description="Total candidate factors considered (after coverage filter)."
    )
    n_factors_skipped: int = Field(
        description="Factors skipped due to no data or n_obs below ``min_n_obs``."
    )
    top_factors: list[SuggestForTickerItem]


class FitPreviewResponse(BaseModel):
    """What a /fit on these inputs WOULD see, without running the regression.

    Used by the UI to show users a live "obs available" counter before they
    click Run, and to surface "this factor only covers 23% of your window"
    warnings up front rather than after a 4-second fit.
    """

    ticker: str
    start: _date
    end: _date
    equity_n_obs: int
    equity_first_date: _date | None = None
    equity_last_date: _date | None = None
    factor_coverage: list[FactorCoverageOut]
    joint_n_obs: int = Field(
        description=(
            "Inner-joined sample size after aligning equity returns with all "
            "factors' Δlogit. This is the actual ``n_obs`` /fit would use."
        )
    )
    # Rigour-pack additions: dict-shaped coverage lookup + explicit predicted
    # post-join obs count. Both are additive — the legacy list field above is
    # preserved so existing clients don't break.
    joint_window_obs: int = Field(
        default=0,
        description=(
            "Same value as ``joint_n_obs`` exposed under the canonical "
            "naming used by the rigour-pack docs."
        ),
    )
    factor_coverage_map: dict[str, FactorCoverageOut] = Field(
        default_factory=dict,
        description=(
            "Per-factor coverage keyed by factor_id for O(1) lookup from "
            "the frontend. Same values as ``factor_coverage`` (which is a "
            "list) — pick whichever shape the client prefers."
        ),
    )
    predicted_window_n_obs: int = Field(
        default=0,
        description=(
            "Predicted ``n_obs`` /fit will see after inner-joining equity "
            "+ every factor's Δlogit. Lets the user know BEFORE clicking "
            "Run how many days will survive the join."
        ),
    )
    min_recommended_obs: int = 30
    warnings: list[str] = Field(default_factory=list)
    recommended_start: _date | None = None
    recommended_end: _date | None = None
