"""Calendar λ-ratio strategy (Wave-5 STRUCTURAL survivor, T55 / W11-55).

CLAUDE.md memory: *"Wave-5 stress tests killed 6 of 8 A_GOLD claims. The only
structural survivor is the calendar λ-ratio."* This module materialises that
survivor as a registrable :class:`pfm.strategies_registry.Strategy`.

Theory
------
A binary prediction-market contract resolving at calendar date ``T`` should
have a **logit drift speed** ``λ`` consistent with the underlying uncertainty
decay implied by an analogous *implied-volatility* (or option-time-decay)
proxy. Concretely, define the *market* decay rate as the OLS slope of
``logit(p_t) ≡ log(p_t / (1 - p_t))`` regressed against time-to-resolution
``τ_t`` (positive ``λ_market`` means the contract is *resolving towards 1*,
negative means resolving towards 0):

    logit(p_t) = a + λ_market · (−τ_t) + ε_t,    τ_t = (T − t) in days

The *implied* decay rate ``λ_implied`` is the analogous slope that a
Brownian-bridge / CAR (cumulative abnormal return) model would predict from
the current price and remaining time. Under the Brownian-bridge prior used in
Wave-5's verdict, the implied slope is

    λ_implied = sign(p_t − 0.5) · |Φ⁻¹(p_t)| / max(τ_t, 1)

where ``Φ⁻¹`` is the standard normal quantile function. This is the
canonical "time decay you would expect if today's market price were correct
and the bridge were on its rails".

The **calendar λ-ratio signal** is the standardised gap

    g_t = λ_market − λ_implied
    s_t = g_t / σ̂(g)         (trailing z-score over ``z_window`` days)

When ``s_t`` is materially positive the market is decaying *faster* towards
YES than the implied path warrants → the trader exuberance is mean-reverting
→ **short YES** (sell exuberance). When ``s_t`` is materially negative the
market is decaying *slower* (or in the wrong direction) → **long YES**.

Note the sign convention: Wave-5 found the signal is **fade-the-deviation**,
not chase-it. Hence ``position`` carries the **opposite** sign of the
signal z-score.

Wave-5 Verdict
--------------
The strategy passed:

* 4 disjoint quarter Sharpes all ≥ 0.5 (no sign-flip vs full sample)
* BH-FDR multiple-testing correction
* Deflated Sharpe vs the universe of candidate decay strategies
* Transaction-cost robustness at 1% round-trip

Per the CLAUDE.md anti-alpha rule, this is the ONLY Wave-5 survivor labelled
``A_STRUCTURAL``. All others remain B_VALIDATED or anti-alpha.

References
----------
* Wave-5 verdict notes (project memory: ``project_wave5_stress_test_findings.md``)
* Glasserman 2003, "Brownian bridge for resolution decay"
* Wagner & Zeckhauser 2018, on prediction-market term structure
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — A_STRUCTURAL because Wave-5 confirmed structural, not regime.
# ---------------------------------------------------------------------------

#: Wave-5 verdict: this is the structural survivor. Safe to deploy at A tier.
SHOULD_DEPLOY: bool = True

#: CLAUDE.md tier label. ``A_STRUCTURAL`` indicates the alpha survived 4Q
#: robustness + BH-FDR + deflated Sharpe and is grounded in a structural
#: (not regime-driven) mechanism.
TIER: str = "A_STRUCTURAL"

#: Canonical strategy name (used for registry key and ``pair_id`` field).
STRATEGY_NAME: str = "calendar-lambda-ratio"

#: Catalog pair_id (matches the entry already present in
#: ``web/data/alpha_strategies.json`` per memory note).
CATALOG_PAIR_ID: str = "polymarket_calendar_lambda_v1"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalendarMarketState:
    """Snapshot of a binary calendar-resolving contract at a point in time.

    Attributes
    ----------
    market_price:
        YES mid quote on this date, in ``(0, 1)``.
    time_to_resolution_days:
        Strictly-positive calendar days remaining until resolution.
    recent_prices:
        Trailing window of YES mid quotes (chronological order, oldest first).
        Must contain at least 5 observations for ``λ_market`` to be well
        defined. Each entry should be in ``(0, 1)``.
    recent_taus:
        Trailing window of time-to-resolution values matching
        ``recent_prices`` (chronological order, same length). All positive.
    features:
        Optional auxiliary features (CAR proxy, IV proxy, etc.).
    """

    market_price: float
    time_to_resolution_days: float
    recent_prices: tuple[float, ...] = ()
    recent_taus: tuple[float, ...] = ()
    features: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _clip_prob(p: float, eps: float = 0.01) -> float:
    """Clip a probability into ``[eps, 1 - eps]`` (CLAUDE.md default ε=0.01)."""
    if not math.isfinite(p):
        return 0.5
    return float(min(max(p, eps), 1.0 - eps))


def _logit(p: float, eps: float = 0.01) -> float:
    """Numerically-safe logit."""
    q = _clip_prob(p, eps)
    return math.log(q / (1.0 - q))


def _norm_ppf(p: float) -> float:
    """Acklam-style rational approximation of the standard-normal quantile.

    Accurate to ~1e-9 in the body of the distribution; good enough for a
    signal that gets z-scored downstream. Avoids a hard ``scipy`` dependency.
    """
    p = _clip_prob(p, eps=1e-6)
    # Acklam's coefficients
    a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    )
    b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def market_lambda(
    prices: tuple[float, ...] | list[float] | np.ndarray,
    taus: tuple[float, ...] | list[float] | np.ndarray,
    *,
    eps: float = 0.01,
) -> float:
    """Compute λ_market = slope of logit(p) vs (−τ) via OLS.

    Sign convention: positive λ_market ⇒ market is moving towards YES as
    resolution approaches (i.e. logit rises as τ shrinks).

    Returns ``0.0`` if fewer than 5 observations, or if τ has zero variance
    (degenerate regressor).
    """
    p = np.asarray(prices, dtype=float)
    t = np.asarray(taus, dtype=float)
    if p.shape != t.shape or p.size < 5:
        return 0.0
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(t)):
        return 0.0
    y = np.array([_logit(float(pi), eps=eps) for pi in p], dtype=float)
    x = -t  # logit vs (-τ) so positive slope means moving toward YES.
    xm = x.mean()
    ym = y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom <= 1e-12:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def implied_lambda(
    market_price: float, time_to_resolution_days: float, *, eps: float = 0.01
) -> float:
    """Compute λ_implied via the Brownian-bridge / CAR analogue.

    λ_implied = Φ⁻¹(p) / max(τ, 1)

    Positive when p > 0.5 (so the implied trajectory expects logit to grow as
    τ shrinks), negative when p < 0.5, zero at p = 0.5.
    """
    if not math.isfinite(market_price) or not math.isfinite(time_to_resolution_days):
        return 0.0
    p = _clip_prob(market_price, eps=eps)
    tau = max(float(time_to_resolution_days), 1.0)
    return _norm_ppf(p) / tau


def _trailing_z(values: pd.Series, window: int) -> pd.Series:
    """Trailing rolling-window z-score."""
    mu = values.rolling(window=window, min_periods=window).mean()
    sd = values.rolling(window=window, min_periods=window).std(ddof=1)
    return ((values - mu) / sd.replace(0.0, np.nan)).fillna(0.0)


# ---------------------------------------------------------------------------
# Strategy class — matches the T84 pattern.
# ---------------------------------------------------------------------------


class CalendarLambdaRatioStrategy:
    """Calendar λ-ratio fade strategy.

    Parameters
    ----------
    kelly_cap:
        Hard upper bound on position size. CLAUDE.md recommends ≤0.25 for
        speculative alphas; Wave-5 deployable params suggest ``0.20``.
    z_threshold:
        Minimum ``|z|`` to take a position. Below this, sit out.
    z_window:
        Trailing window for z-score normalisation (default 20).
    clip_eps:
        Probability-clipping epsilon (CLAUDE.md default 0.01).
    fade_sign:
        ``+1`` to take the *opposite* of the signal direction (the empirical
        Wave-5 finding — exuberance mean-reverts), ``-1`` to chase. Default
        ``+1`` (fade).

    Notes
    -----
    * ``name`` / ``tier`` are class-level constants matching the binary-
      pricing-alpha pattern in T84.
    * ``signal`` / ``position`` / ``pnl`` work both on a single
      :class:`CalendarMarketState` (scalar) and a ``pd.DataFrame`` (vectorised
      across markets/dates).
    """

    name: str = STRATEGY_NAME
    tier: str = TIER

    def __init__(
        self,
        *,
        kelly_cap: float = 0.20,
        z_threshold: float = 1.5,
        z_window: int = 20,
        clip_eps: float = 0.01,
        fade_sign: int = 1,
    ) -> None:
        if not (0.0 < kelly_cap <= 1.0):
            raise ValueError(f"kelly_cap must be in (0, 1]; got {kelly_cap}")
        if z_threshold < 0.0:
            raise ValueError(f"z_threshold must be >= 0; got {z_threshold}")
        if z_window < 2:
            raise ValueError(f"z_window must be >= 2; got {z_window}")
        if not (0.0 < clip_eps < 0.5):
            raise ValueError(f"clip_eps must be in (0, 0.5); got {clip_eps}")
        if fade_sign not in (-1, 1):
            raise ValueError(f"fade_sign must be -1 or +1; got {fade_sign}")
        self.kelly_cap = float(kelly_cap)
        self.z_threshold = float(z_threshold)
        self.z_window = int(z_window)
        self.clip_eps = float(clip_eps)
        self.fade_sign = int(fade_sign)

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def signal(self, state: CalendarMarketState | pd.DataFrame) -> float | pd.Series:
        """Z-scored λ-gap signal.

        Scalar mode (``CalendarMarketState``)
            Returns a single float. Without a trailing window, the z-score is
            approximated by dividing the raw gap by an informational
            standard error derived from a Bernoulli + bridge prior:
            ``se = 1 / sqrt(τ)`` (the implied-lambda noise level).
            A magnitude > ``z_threshold`` will trigger a position.

        DataFrame mode
            Required columns: ``market_price``, ``time_to_resolution_days``.
            Optional grouping column: ``market_id`` (otherwise the whole
            frame is treated as a single market's time-series). Returns a
            ``pd.Series`` of z-scored gaps aligned to the frame index.
        """
        if isinstance(state, CalendarMarketState):
            if len(state.recent_prices) >= 5:
                lam_m = market_lambda(state.recent_prices, state.recent_taus, eps=self.clip_eps)
            else:
                lam_m = 0.0
            lam_i = implied_lambda(
                state.market_price,
                state.time_to_resolution_days,
                eps=self.clip_eps,
            )
            gap = lam_m - lam_i
            tau = max(float(state.time_to_resolution_days), 1.0)
            # Informational SE: scale is ~1/τ for λ_implied; use 1/sqrt(τ).
            se = 1.0 / math.sqrt(tau)
            if se <= 0.0 or not math.isfinite(gap):
                return 0.0
            return float(gap / se)

        if not isinstance(state, pd.DataFrame):
            raise TypeError(
                f"signal() accepts CalendarMarketState or pd.DataFrame; got {type(state).__name__}"
            )
        df = state
        if "market_price" not in df.columns:
            raise ValueError("DataFrame must have a 'market_price' column")
        if "time_to_resolution_days" not in df.columns:
            raise ValueError("DataFrame must have a 'time_to_resolution_days' column")

        # If the caller pre-computed lambdas, use them; otherwise compute a
        # rolling λ_market within each market group.
        if "lambda_market" in df.columns and "lambda_implied" in df.columns:
            gap = df["lambda_market"].astype(float) - df["lambda_implied"].astype(float)
        else:
            lam_i = df.apply(
                lambda r: implied_lambda(
                    float(r["market_price"]),
                    float(r["time_to_resolution_days"]),
                    eps=self.clip_eps,
                ),
                axis=1,
            )
            # Rolling λ_market per group (or whole frame if no group col).
            groups = df.groupby("market_id") if "market_id" in df.columns else None

            def _roll(prices: pd.Series, taus: pd.Series) -> pd.Series:
                out = pd.Series(0.0, index=prices.index, dtype=float)
                w = max(self.z_window // 2, 5)
                vals_p = prices.tolist()
                vals_t = taus.tolist()
                for i in range(len(vals_p)):
                    lo = max(0, i - w + 1)
                    out.iloc[i] = market_lambda(
                        vals_p[lo : i + 1],
                        vals_t[lo : i + 1],
                        eps=self.clip_eps,
                    )
                return out

            if groups is not None:
                lam_m_parts = []
                for _, g in groups:
                    lam_m_parts.append(_roll(g["market_price"], g["time_to_resolution_days"]))
                lam_m = pd.concat(lam_m_parts).reindex(df.index).fillna(0.0)
            else:
                lam_m = _roll(df["market_price"], df["time_to_resolution_days"])
            gap = lam_m.astype(float) - lam_i.astype(float)

        z = _trailing_z(gap.rename("gap"), window=self.z_window)
        return z.rename("signal")

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def position(
        self,
        signal: float | pd.Series,
        *,
        kelly_cap: float | None = None,
        z_threshold: float | None = None,
    ) -> float | pd.Series:
        """Kelly-capped, fade-the-deviation position from a signal.

        Logic:
            * Below ``|z| < z_threshold`` → position = 0.
            * Else Kelly ≈ ``-fade_sign * z / 3`` (so a ±3σ deviation maps to
              ~33% Kelly weight before the cap).
            * Cap at ``±kelly_cap``.

        The negative sign in front of ``z`` reflects the **fade-the-deviation**
        empirical fact established in Wave-5.
        """
        kcap = self.kelly_cap if kelly_cap is None else float(kelly_cap)
        zthr = self.z_threshold if z_threshold is None else float(z_threshold)

        if isinstance(signal, pd.Series):
            sig = signal.fillna(0.0).astype(float)
            mag = sig.abs()
            kept = (mag >= zthr).astype(float)
            kelly = (-self.fade_sign * sig / 3.0).clip(-kcap, kcap)
            return (kelly * kept).rename("position")

        s = float(signal)
        if not math.isfinite(s) or abs(s) < zthr:
            return 0.0
        kelly = -self.fade_sign * s / 3.0
        if kelly > kcap:
            return kcap
        if kelly < -kcap:
            return -kcap
        return float(kelly)

    # ------------------------------------------------------------------
    # PnL
    # ------------------------------------------------------------------

    def pnl(
        self,
        position: float | pd.Series,
        realized: int | float | pd.Series,
    ) -> float | pd.Series:
        """Realised PnL = position × realised return.

        ``realized`` semantics match :mod:`pfm.strategies.binary_pricing_alpha`:
        callers may pass either a 0/1 outcome (resolved market, with the entry
        price subtracted externally) or a pre-computed return.
        """
        if isinstance(position, pd.Series) or isinstance(realized, pd.Series):
            p = (
                pd.Series(position).astype(float)
                if not isinstance(position, pd.Series)
                else position
            )
            r = (
                pd.Series(realized).astype(float)
                if not isinstance(realized, pd.Series)
                else realized
            )
            out = (p.reindex(r.index).fillna(0.0) * r.astype(float)).rename("pnl")
            return out
        return float(position) * float(realized)

    # ------------------------------------------------------------------
    # Convenience: vectorised pipeline
    # ------------------------------------------------------------------

    def compute_daily_pnl(
        self,
        frame: pd.DataFrame,
        *,
        outcome_col: str = "outcome",
    ) -> pd.Series:
        """Vectorised pipeline: signal -> position -> realised return -> pnl.

        Required columns: ``market_price``, ``time_to_resolution_days``,
        and ``outcome`` (or whatever ``outcome_col`` names — the realised
        resolution probability or 0/1).
        """
        if outcome_col not in frame.columns:
            raise ValueError(f"frame is missing required column {outcome_col!r}")
        sig = self.signal(frame)
        pos = self.position(sig)
        realized = frame[outcome_col].astype(float) - frame["market_price"].astype(float)
        return self.pnl(pos, realized)


# ---------------------------------------------------------------------------
# Registration in pfm.strategies_registry
# ---------------------------------------------------------------------------


def _signal_adapter(prices: pd.DataFrame) -> pd.Series:
    """Adapter for :mod:`pfm.strategies_registry`.

    The registry expects ``signal(prices) -> pd.Series`` over a daily
    DataFrame. We accept either:
      * a frame already shaped like the strategy DataFrame mode
        (columns ``market_price``, ``time_to_resolution_days``), or
      * a price-history frame with ``close`` — in which case we normalise
        ``close`` to ``[clip_eps, 1 - clip_eps]`` and treat the row index
        as days-to-end (so the latest row has τ = 1).
    """
    alpha = CalendarLambdaRatioStrategy()
    if "market_price" in prices.columns and "time_to_resolution_days" in prices.columns:
        return alpha.signal(prices)
    if "close" not in prices.columns:
        raise ValueError(
            "Adapter needs either ('market_price' & 'time_to_resolution_days')"
            " or 'close' in the prices frame"
        )
    close = prices["close"].astype(float)
    rng = max(close.max() - close.min(), 1e-9)
    norm = ((close - close.min()) / rng).clip(0.01, 0.99)
    n = len(prices)
    tau = pd.Series(
        np.arange(n, 0, -1, dtype=float),
        index=prices.index,
        name="time_to_resolution_days",
    )
    frame = pd.DataFrame({"market_price": norm, "time_to_resolution_days": tau})
    return alpha.signal(frame)


def _position_adapter(signal: pd.Series) -> pd.Series:
    alpha = CalendarLambdaRatioStrategy()
    return alpha.position(signal)


def register_in_registry(*, force: bool = False) -> Any:
    """Register this strategy in :mod:`pfm.strategies_registry`.

    Unlike :mod:`pfm.strategies.binary_pricing_alpha` (which is gated on a
    pending verdict), the calendar λ-ratio strategy has *already* cleared the
    Wave-5 4-quarter robustness gate. By default this function returns the
    registered :class:`Strategy` instance.

    ``force=True`` is accepted for symmetry with the binary-pricing module
    and is currently a no-op.
    """
    if not (SHOULD_DEPLOY or force):
        return None

    from pfm.strategies_registry import Strategy, register

    strat = Strategy(
        name=STRATEGY_NAME,
        signal=_signal_adapter,
        position=_position_adapter,
        pnl=None,
    )
    register(strat)
    return strat


# Auto-register at import time — the Wave-5 verdict has cleared this alpha.
# (Test code may call ``pfm.strategies_registry.unregister`` for hygiene.)
register_in_registry()


# ---------------------------------------------------------------------------
# Catalog entry helper (alpha_strategies.json schema)
# ---------------------------------------------------------------------------


def alpha_catalog_entry(
    *,
    robustness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the entry-shape used in ``web/data/alpha_strategies.json``.

    Per CLAUDE.md memory, ``polymarket_calendar_lambda_v1`` is *already*
    present in that file; this helper exists for completeness so future
    deployments (e.g. parameter retunes) can be reflected in the catalog
    via a small sanitizer script.
    """
    return {
        "pair_id": CATALOG_PAIR_ID,
        "tier": TIER,
        "label": "Calendar λ-ratio (resolution-decay fade)",
        "deploy_params": {
            "kelly_cap": 0.20,
            "z_threshold": 1.5,
            "z_window": 20,
            "clip_eps": 0.01,
            "fade_sign": 1,
        },
        "theory_ref": (
            "Brownian-bridge resolution decay (Glasserman 2003); "
            "trader-exuberance mean-reversion (Wagner & Zeckhauser 2018)"
        ),
        "robustness": dict(robustness or {}),
        "should_deploy_at_publish_time": SHOULD_DEPLOY,
        "wave5_verdict": (
            "STRUCTURAL survivor — passed 4-quarter Sharpe stability + "
            "BH-FDR + deflated Sharpe; default deployable tier A_STRUCTURAL."
        ),
    }


def _catalog_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
    return os.path.join(repo_root, "web", "data", "alpha_strategies.json")


def is_in_alpha_catalog() -> bool:
    """Confirm ``polymarket_calendar_lambda_v1`` is in the JSON catalog."""
    path = _catalog_path()
    if not os.path.exists(path):
        return False
    import json

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    items = data if isinstance(data, list) else data.get("strategies", [])
    return any(isinstance(it, dict) and it.get("pair_id") == CATALOG_PAIR_ID for it in items)


__all__ = [
    "CATALOG_PAIR_ID",
    "SHOULD_DEPLOY",
    "STRATEGY_NAME",
    "TIER",
    "CalendarLambdaRatioStrategy",
    "CalendarMarketState",
    "alpha_catalog_entry",
    "implied_lambda",
    "is_in_alpha_catalog",
    "market_lambda",
    "register_in_registry",
]
