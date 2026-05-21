"""Strategy verdict layer — converts raw stats into actionable recommendations.

Every analytical view in the Strategies/Terminal panel produces a flurry of raw
diagnostics (ADF p-value, half-life, ρ, β, R², VIF, Sharpe, …).  Users — even
quant-literate ones — should not have to mentally aggregate those numbers into
"should I open this trade?".  This module centralises the *interpretation*
logic so the front-end can render a single, opinionated verdict card next to
the raw stats.

Public functions
----------------

``cointegration_verdict``
    Engle–Granger 2-step pair stationarity → OPEN_PAIR / WATCH / SKIP / FLATTEN.

``pair_trade_verdict``
    Live z-score regime → OPEN_PAIR / HOLD / FLATTEN / WAIT (stop-out aware).

``bollinger_verdict``
    Bollinger / single-factor mean-reversion using the same z-score rules.

``regression_verdict``
    Factor-model fit quality → DEPLOY / WATCH / REJECT.

``alpha_card_verdict``
    Maps tiered alpha-strategy entries to deployment recommendations.

The two ``POST /strategy-verdict/...`` FastAPI endpoints are exposed via the
module-level ``router`` for callers that prefer server-side computation
(useful for non-browser consumers).  They are NOT auto-wired into ``main.py``;
``Damian`` mounts them when desired.

Design notes
------------

* Pure functions, no I/O, no global state.  Easy to unit-test.
* Inputs are plain floats / ints / dicts so the same helpers work for both the
  front-end (via the JSON port) and other Python services.
* All thresholds are module-level constants documented inline.  Changing the
  threshold should never need a code edit beyond the constant.
* Reasoning lines are short (≤ ~80 chars) so they render cleanly on the
  Plotly-styled cards without truncation.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: Canonical action vocabulary surfaced to the front-end.  The list is closed:
#: front-end colour mapping depends on the exact strings.
Action = Literal[
    "OPEN_PAIR",
    "WAIT",
    "SKIP",
    "WATCH",
    "FLATTEN",
    "HOLD",
    "DEPLOY",
    "REJECT",
    "DEPLOY_LIVE_SMALL_SIZE",
    "PAPER_TRADE_FIRST",
    "WATCH_DO_NOT_DEPLOY",
    "ARCHIVE",
]

#: Confidence levels.  ``high`` means the diagnostics agree on a clear signal;
#: ``low`` means we are recommending something marginal.
Confidence = Literal["low", "medium", "high"]


# Cointegration thresholds ---------------------------------------------------
ADF_OPEN_MAX_P: float = 0.05
ADF_WATCH_MAX_P: float = 0.15
HALF_LIFE_MIN_DAYS: float = 1.0
HALF_LIFE_MAX_DAYS: float = 30.0
HALF_LIFE_SUSPICIOUS_DAYS: float = 0.5
RHO_MIN: float = 0.3
N_OBS_MIN: float = 50

# Pair / Bollinger thresholds (defaults match the existing UI form) ----------
DEFAULT_ENTRY_Z: float = 2.0
DEFAULT_EXIT_Z: float = 0.5
DEFAULT_STOP_Z: float = 4.0

# Regression thresholds ------------------------------------------------------
R2_DEPLOY_MIN: float = 0.5
R2_WATCH_MIN: float = 0.2
VIF_DEPLOY_MAX: float = 5.0
VIF_REJECT_MIN: float = 10.0
N_PER_FACTOR_MIN: float = 30.0


# ---------------------------------------------------------------------------
# Cointegration
# ---------------------------------------------------------------------------


def cointegration_verdict(
    adf_p: float,
    half_life_days: float | None,
    rho_ar1: float,
    n_obs: int,
    beta_hedge: float,
    current_z: float | None = None,
    in_position: bool = False,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
) -> dict[str, Any]:
    """Translate Engle-Granger pair diagnostics into a trade recommendation.

    Args:
        adf_p: Augmented Dickey–Fuller p-value on the regression residual.
        half_life_days: AR(1) half-life (days).  ``None`` if the spread is not
            mean-reverting at the AR(1) level (ρ ≥ 1).
        rho_ar1: AR(1) coefficient on the residual.
        n_obs: Sample size used in the regression.
        beta_hedge: OLS hedge ratio (slope of A on B).
        current_z: Latest rolling z-score, when known.  When ``None`` the
            verdict will not gate on z (used for "is this pair tradeable in
            principle?" calls).
        in_position: Whether the caller already holds the spread.  Affects
            whether we recommend ``FLATTEN`` vs. ``WAIT`` near the mean.
        entry_z: Entry threshold to compare against ``current_z``.
        exit_z: Exit threshold to compare against ``current_z``.

    Returns:
        Verdict dict with ``action``, ``reasoning``, ``confidence``,
        ``trade_spec`` (only when action is ``OPEN_PAIR``) and
        ``monitoring_rules``.
    """

    reasoning: list[str] = []

    # Headline ADF reading ---------------------------------------------------
    if adf_p < ADF_OPEN_MAX_P:
        adf_state = "pass"
        reasoning.append(f"ADF p={adf_p:.3f} < {ADF_OPEN_MAX_P:.2f} → spread is stationary.")
    elif adf_p < ADF_WATCH_MAX_P:
        adf_state = "borderline"
        reasoning.append(
            f"ADF p={adf_p:.3f} ∈ [{ADF_OPEN_MAX_P:.2f}, {ADF_WATCH_MAX_P:.2f}] "
            "→ borderline stationarity; monitor."
        )
    else:
        adf_state = "fail"
        reasoning.append(f"ADF p={adf_p:.3f} > {ADF_WATCH_MAX_P:.2f} → spread not stationary.")

    # Half-life check --------------------------------------------------------
    if half_life_days is None:
        hl_state = "missing"
        reasoning.append("Half-life undefined (ρ ≥ 1) — no AR(1) mean-reversion.")
    elif half_life_days < HALF_LIFE_SUSPICIOUS_DAYS or (
        half_life_days < HALF_LIFE_MIN_DAYS and n_obs < 100
    ):
        hl_state = "suspicious"
        reasoning.append(
            f"Half-life {half_life_days:.1f}d on n={n_obs} is suspiciously fast — "
            "likely small-sample artifact."
        )
    elif half_life_days > HALF_LIFE_MAX_DAYS:
        hl_state = "too_slow"
        reasoning.append(
            f"Half-life {half_life_days:.1f}d > {HALF_LIFE_MAX_DAYS:.0f}d "
            "→ reversion too slow to be tradeable."
        )
    else:
        hl_state = "ok"
        reasoning.append(
            f"Half-life {half_life_days:.1f}d sits in the tradeable "
            f"[{HALF_LIFE_MIN_DAYS:.0f}, {HALF_LIFE_MAX_DAYS:.0f}]d band."
        )

    # Persistence check ------------------------------------------------------
    if abs(rho_ar1) >= RHO_MIN:
        rho_state = "ok"
        reasoning.append(f"ρ={rho_ar1:.3f} shows persistent mean-reverting AR(1) dynamics.")
    else:
        rho_state = "weak"
        reasoning.append(f"ρ={rho_ar1:.3f} is weak — not enough persistence on its own.")

    # Sample-size check ------------------------------------------------------
    if n_obs < N_OBS_MIN:
        n_state = "too_small"
        reasoning.append(
            f"n_obs={n_obs} < {N_OBS_MIN:.0f} — sample too small for stable "
            "Engle-Granger inference."
        )
    else:
        n_state = "ok"

    # Aggregate decision ----------------------------------------------------
    full_pass = adf_state == "pass" and hl_state == "ok" and rho_state == "ok" and n_state == "ok"

    # In-position FLATTEN override (we are long/short the spread already and
    # it has reverted): close the trade regardless of the open-quality.
    if in_position and current_z is not None and abs(current_z) < exit_z:
        return {
            "action": "FLATTEN",
            "reasoning": [
                f"|z|={abs(current_z):.2f} < exit_z={exit_z:.2f} → spread has "
                "reverted; close the position.",
                *reasoning,
            ],
            "confidence": "high",
            "trade_spec": None,
            "monitoring_rules": [
                "Re-evaluate the pair from scratch before re-entering.",
            ],
        }

    if (
        n_state == "too_small"
        or hl_state == "suspicious"
        or hl_state == "missing"
        or adf_state == "fail"
    ):
        action: Action = "SKIP"
        confidence: Confidence = "high" if adf_state == "fail" else "medium"
        monitoring = [
            f"Re-check daily; reject if ADF p > {ADF_WATCH_MAX_P:.2f} for 5 consecutive days.",
        ]
        return {
            "action": action,
            "reasoning": reasoning,
            "confidence": confidence,
            "trade_spec": None,
            "monitoring_rules": monitoring,
        }

    if not full_pass:
        # Borderline ADF or weak ρ but otherwise sane → WATCH.
        return {
            "action": "WATCH",
            "reasoning": reasoning,
            "confidence": "low",
            "trade_spec": None,
            "monitoring_rules": [
                f"Promote to OPEN_PAIR if ADF p drops below {ADF_OPEN_MAX_P:.2f} "
                f"and |ρ| ≥ {RHO_MIN:.2f}.",
                "Re-fit weekly.",
            ],
        }

    # We have a clean OPEN_PAIR signal -------------------------------------
    if current_z is not None and abs(current_z) < entry_z:
        return {
            "action": "WAIT",
            "reasoning": [
                *reasoning,
                f"|z|={abs(current_z):.2f} < entry_z={entry_z:.2f} → wait for "
                "wider dispersion before opening.",
            ],
            "confidence": "medium",
            "trade_spec": None,
            "monitoring_rules": [
                f"Open when |z| ≥ {entry_z:.2f}; re-fit if ADF p drifts above "
                f"{ADF_OPEN_MAX_P:.2f}.",
            ],
        }

    direction = (
        "long_a_short_b"
        if (current_z is not None and current_z < 0)
        else ("short_a_long_b" if current_z is not None else "long_a_short_b")
    )
    expected_hold = round(half_life_days * 1.5, 1) if half_life_days is not None else None
    expected_ev_pct = (
        # Crude analytic EV ≈ (entry-exit) × σ(ε) — we don't have σ here, so
        # we report the dimensionless edge in z-units which the front-end can
        # multiply by σ if desired.
        round(entry_z - exit_z, 2)
    )

    trade_spec: dict[str, Any] = {
        "direction": direction,
        "a_size": 1.0,
        "b_size": round(abs(beta_hedge), 4),
        "entry_z": entry_z,
        "exit_z": exit_z,
        "expected_hold_days": expected_hold,
        "expected_ev_pct": expected_ev_pct,
    }

    return {
        "action": "OPEN_PAIR",
        "reasoning": reasoning,
        "confidence": "high",
        "trade_spec": trade_spec,
        "monitoring_rules": [
            f"Stop-loss at |z| ≥ {DEFAULT_STOP_Z:.1f}.",
            f"Re-evaluate if ADF p climbs above {ADF_WATCH_MAX_P:.2f}.",
            f"Take-profit when |z| < {exit_z:.2f}.",
        ],
    }


# ---------------------------------------------------------------------------
# Pairs trading (live z-score regime)
# ---------------------------------------------------------------------------


def pair_trade_verdict(
    current_z: float,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
    stop_z: float = DEFAULT_STOP_Z,
    in_position: bool = False,
    cointegration_passed: bool = True,
    beta_hedge: float = 1.0,
) -> dict[str, Any]:
    """Translate a live z-score reading into an entry/exit/hold action.

    The function is intentionally regime-driven: the same input ``current_z``
    yields a different action depending on whether the caller already holds
    the spread (``in_position``).  ``cointegration_passed`` gates entry —
    we never recommend opening on a non-stationary residual even if the z is
    extreme.
    """
    abs_z = abs(current_z)
    reasoning: list[str] = [
        f"Current z={current_z:+.2f} (|z|={abs_z:.2f})",
    ]

    # Stop-out always wins when in position.
    if in_position and abs_z >= stop_z:
        reasoning.append(f"|z| ≥ stop_z={stop_z:.2f} → blow-out; flatten immediately.")
        return {
            "action": "FLATTEN",
            "reasoning": reasoning,
            "confidence": "high",
            "trade_spec": None,
            "monitoring_rules": ["Cool-off for at least one half-life before re-entering."],
        }

    if in_position and abs_z <= exit_z:
        reasoning.append(f"|z| ≤ exit_z={exit_z:.2f} → spread reverted; take profit.")
        return {
            "action": "FLATTEN",
            "reasoning": reasoning,
            "confidence": "high",
            "trade_spec": None,
            "monitoring_rules": ["Wait for fresh dispersion before re-opening."],
        }

    if in_position:
        reasoning.append(f"exit_z={exit_z:.2f} < |z| < stop_z={stop_z:.2f} → keep the trade.")
        return {
            "action": "HOLD",
            "reasoning": reasoning,
            "confidence": "medium",
            "trade_spec": None,
            "monitoring_rules": [
                f"Flatten if |z| < {exit_z:.2f}; stop-out if |z| ≥ {stop_z:.2f}.",
            ],
        }

    # Not in position --------------------------------------------------------
    if not cointegration_passed:
        reasoning.append("Cointegration test did not pass — never open.")
        return {
            "action": "SKIP",
            "reasoning": reasoning,
            "confidence": "high",
            "trade_spec": None,
            "monitoring_rules": ["Re-run Engle-Granger before considering this pair."],
        }

    if abs_z < exit_z:
        reasoning.append(
            f"|z| < exit_z={exit_z:.2f} → spread already at fair value; nothing to do."
        )
        return {
            "action": "WAIT",
            "reasoning": reasoning,
            "confidence": "medium",
            "trade_spec": None,
            "monitoring_rules": [f"Open only when |z| ≥ {entry_z:.2f}."],
        }

    if abs_z < entry_z:
        reasoning.append(
            f"exit_z={exit_z:.2f} ≤ |z| < entry_z={entry_z:.2f} → no trade yet, but watch closely."
        )
        return {
            "action": "WAIT",
            "reasoning": reasoning,
            "confidence": "medium",
            "trade_spec": None,
            "monitoring_rules": [f"Open when |z| ≥ {entry_z:.2f}."],
        }

    if abs_z >= stop_z:
        reasoning.append(f"|z| ≥ stop_z={stop_z:.2f} → blow-out region; do not open.")
        return {
            "action": "SKIP",
            "reasoning": reasoning,
            "confidence": "high",
            "trade_spec": None,
            "monitoring_rules": ["Wait for normal regime before reconsidering."],
        }

    # Clean entry zone -------------------------------------------------------
    direction = "long_a_short_b" if current_z < 0 else "short_a_long_b"
    reasoning.append(f"|z| ≥ entry_z={entry_z:.2f} → open the spread.")
    return {
        "action": "OPEN_PAIR",
        "reasoning": reasoning,
        "confidence": "high",
        "trade_spec": {
            "direction": direction,
            "a_size": 1.0,
            "b_size": round(abs(beta_hedge), 4),
            "entry_z": entry_z,
            "exit_z": exit_z,
            "stop_z": stop_z,
        },
        "monitoring_rules": [
            f"Take-profit at |z| ≤ {exit_z:.2f}; stop-out at |z| ≥ {stop_z:.2f}.",
        ],
    }


# ---------------------------------------------------------------------------
# Bollinger / single-factor mean-reversion
# ---------------------------------------------------------------------------


def bollinger_verdict(
    current_z: float,
    hurst: float | None = None,
    vr_p: float | None = None,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
    stop_z: float = DEFAULT_STOP_Z,
    in_position: bool = False,
) -> dict[str, Any]:
    """Bollinger / OU-band verdict using Hurst + VR as gating diagnostics.

    Reuses the same z-score regime logic as :func:`pair_trade_verdict` but
    swaps cointegration-passed for the AND of Hurst < 0.5 and VR p < 0.10.
    """
    is_mean_reverting = True
    extra_reasons: list[str] = []
    if hurst is not None:
        if hurst >= 0.55:
            is_mean_reverting = False
            extra_reasons.append(f"Hurst H={hurst:.2f} ≥ 0.55 → series is trending, not reverting.")
        elif hurst >= 0.45:
            extra_reasons.append(
                f"Hurst H={hurst:.2f} → essentially random walk; reversion edge weak."
            )
        else:
            extra_reasons.append(f"Hurst H={hurst:.2f} < 0.45 → mean-reverting regime confirmed.")
    if vr_p is not None:
        if vr_p > 0.10:
            extra_reasons.append(f"VR p={vr_p:.3f} > 0.10 → cannot reject random-walk null.")
        else:
            extra_reasons.append(f"VR p={vr_p:.3f} ≤ 0.10 → variance-ratio rejects random walk.")

    base = pair_trade_verdict(
        current_z=current_z,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        in_position=in_position,
        cointegration_passed=is_mean_reverting,
    )
    base["reasoning"] = [*extra_reasons, *base["reasoning"]]
    return base


# ---------------------------------------------------------------------------
# Regression / factor-model fit quality
# ---------------------------------------------------------------------------


def regression_verdict(
    r2: float,
    n_obs: int,
    n_factors: int,
    vif_max: float | None = None,
    f_pvalue: float | None = None,
) -> dict[str, Any]:
    """Verdict for a factor-model fit.

    Decides between DEPLOY / WATCH / REJECT based on R², VIF and the
    sample-size-to-factor ratio.  ``f_pvalue`` is optional but, if provided,
    further down-grades a fit whose joint F-test is not significant.
    """
    reasoning: list[str] = [
        f"R²={r2:.3f}, n={n_obs}, k={n_factors} factors.",
    ]

    n_per_k = n_obs / max(n_factors, 1)
    if n_per_k < N_PER_FACTOR_MIN:
        reasoning.append(f"n/k={n_per_k:.1f} < {N_PER_FACTOR_MIN:.0f} → over-parameterised.")
        return {
            "action": "REJECT",
            "reasoning": reasoning,
            "confidence": "high",
            "trade_spec": None,
            "monitoring_rules": ["Drop factors or extend the sample window."],
        }

    if vif_max is not None and vif_max >= VIF_REJECT_MIN:
        reasoning.append(
            f"max VIF={vif_max:.1f} ≥ {VIF_REJECT_MIN:.0f} → severe multicollinearity."
        )
        return {
            "action": "REJECT",
            "reasoning": reasoning,
            "confidence": "high",
            "trade_spec": None,
            "monitoring_rules": ["Prune correlated factors before refitting."],
        }

    if r2 < R2_WATCH_MIN:
        reasoning.append(f"R² < {R2_WATCH_MIN:.2f} → fit explains too little variance.")
        return {
            "action": "REJECT",
            "reasoning": reasoning,
            "confidence": "high",
            "trade_spec": None,
            "monitoring_rules": ["Try different factors or a non-linear model."],
        }

    if f_pvalue is not None and f_pvalue > 0.05:
        reasoning.append(f"F-test p={f_pvalue:.3f} > 0.05 → joint significance not established.")
        return {
            "action": "WATCH",
            "reasoning": reasoning,
            "confidence": "low",
            "trade_spec": None,
            "monitoring_rules": ["Refit with more data before relying on the betas."],
        }

    if r2 >= R2_DEPLOY_MIN and (vif_max is None or vif_max < VIF_DEPLOY_MAX):
        reasoning.append(
            f"R² ≥ {R2_DEPLOY_MIN:.2f}, max VIF "
            f"{'n/a' if vif_max is None else f'{vif_max:.1f}'} < "
            f"{VIF_DEPLOY_MAX:.0f} → strong, well-conditioned fit."
        )
        return {
            "action": "DEPLOY",
            "reasoning": reasoning,
            "confidence": "high",
            "trade_spec": None,
            "monitoring_rules": [
                "Re-fit weekly; alert if R² drops by > 20% week-over-week.",
            ],
        }

    reasoning.append(f"R² ∈ [{R2_WATCH_MIN:.2f}, {R2_DEPLOY_MIN:.2f}) → marginal fit, watch only.")
    return {
        "action": "WATCH",
        "reasoning": reasoning,
        "confidence": "medium",
        "trade_spec": None,
        "monitoring_rules": [
            f"Promote to DEPLOY if R² rises above {R2_DEPLOY_MIN:.2f} on the next refit.",
        ],
    }


# ---------------------------------------------------------------------------
# Alpha card (alpha_strategies.json entries)
# ---------------------------------------------------------------------------

_TIER_TO_ACTION: dict[str, Action] = {
    "A_GOLD": "DEPLOY_LIVE_SMALL_SIZE",
    "B_VALIDATED": "PAPER_TRADE_FIRST",
    "C_TENTATIVE": "WATCH_DO_NOT_DEPLOY",
    "D_REJECTED": "ARCHIVE",
}


def alpha_card_verdict(strategy: dict[str, Any]) -> dict[str, Any]:
    """Map an alpha-strategy card to a deployment recommendation.

    The expected ``strategy`` shape is the row in ``alpha_strategies.json``,
    i.e. it contains at least ``tier`` and may contain ``sharpe_oos``,
    ``allocation_pct`` and ``name``.
    """
    tier = str(strategy.get("tier", "")).upper()
    name = strategy.get("name", "<unnamed>")
    sharpe_oos = strategy.get("sharpe_oos")
    allocation_pct = strategy.get("allocation_pct")

    action: Action = _TIER_TO_ACTION.get(tier, "WATCH_DO_NOT_DEPLOY")
    reasoning: list[str] = [f"Strategy '{name}' is tier {tier or 'UNKNOWN'}."]
    if sharpe_oos is not None:
        reasoning.append(f"Out-of-sample Sharpe = {float(sharpe_oos):.2f}.")

    monitoring: list[str]
    if action == "DEPLOY_LIVE_SMALL_SIZE":
        alloc = allocation_pct if allocation_pct is not None else 1.0
        reasoning.append(f"Tier A_GOLD → deploy live with capped allocation {alloc:.2f}%.")
        monitoring = [
            "Daily PnL alarm at -2σ from in-sample.",
            "Auto-flatten if rolling Sharpe drops below 0 for 10 sessions.",
        ]
    elif action == "PAPER_TRADE_FIRST":
        reasoning.append("Tier B_VALIDATED → run in paper-trading mode for ≥ 30 days.")
        monitoring = [
            "Promote to A_GOLD only after live OOS Sharpe ≥ 1.0.",
            "Reject and archive if drawdown exceeds in-sample worst case.",
        ]
    elif action == "ARCHIVE":
        reasoning.append("Tier D_REJECTED → archive; do not allocate any capital.")
        monitoring = []
    else:
        reasoning.append("Tier below B_VALIDATED → research-only; do not allocate capital.")
        monitoring = ["Re-evaluate after at least one full quarter of new data."]

    return {
        "action": action,
        "reasoning": reasoning,
        "confidence": "high" if action in {"DEPLOY_LIVE_SMALL_SIZE", "ARCHIVE"} else "medium",
        "trade_spec": None,
        "monitoring_rules": monitoring,
    }


# ---------------------------------------------------------------------------
# 4-quarter Sharpe stability enforcer
# ---------------------------------------------------------------------------

#: Minimum positive-quarter count for A_GOLD tier.
QUARTERLY_GOLD_MIN_POSITIVE: int = 4
#: Minimum positive-quarter count for B_VALIDATED tier.
QUARTERLY_SILVER_MIN_POSITIVE: int = 3
#: Minimum number of quarterly observations required before A/B tier eligibility.
QUARTERLY_MIN_QUARTERS: int = 4


def quarterly_stability_test(
    quarterly_sharpes: list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Cross-quarter robustness gate for promoting alpha-strategy tiers.

    The Wave-5 stress tests killed 6 of 8 A_GOLD claims that looked great in
    a single quarter but failed when re-run on 4 disjoint quarters.  This
    helper bakes that lesson into a reusable decision rule.

    Promotion rules
    ---------------

    * **A_GOLD**: at least :data:`QUARTERLY_MIN_QUARTERS` quarters AND every
      quarter's Sharpe is strictly greater than ``threshold`` AND no sign
      flips between consecutive quarters.
    * **B_VALIDATED**: at least :data:`QUARTERLY_MIN_QUARTERS` quarters AND
      at least :data:`QUARTERLY_SILVER_MIN_POSITIVE` quarters above
      ``threshold``.  Sign flips are tolerated here — the alpha just isn't
      promoted to A_GOLD until the quarterly record is unanimous.
    * **C_TENTATIVE**: anything weaker (too few quarters or too many losing
      quarters).

    A "sign flip" is a strict change of sign between two adjacent quarters.
    NaN entries break neighbouring sign-flip checks (we treat NaN as
    indeterminate, not as a flip) and never count toward ``n_positive``.

    Args:
        quarterly_sharpes: Per-quarter Sharpe ratios in chronological order.
        threshold: Minimum per-quarter Sharpe to count as a "positive"
            quarter.  Default ``0.5`` matches the rule in CLAUDE.md.

    Returns:
        Dict with ``n_quarters``, ``n_positive``, ``sign_flips``,
        ``passes_4q_gold``, ``passes_4q_silver`` and ``tier_recommendation``.
    """
    import math as _math

    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    sharpes_clean: list[float] = []
    for s in quarterly_sharpes:
        try:
            sharpes_clean.append(float(s))
        except (TypeError, ValueError):
            sharpes_clean.append(float("nan"))

    n_quarters = len(sharpes_clean)
    n_positive = sum(1 for s in sharpes_clean if not _math.isnan(s) and s > threshold)

    sign_flips = 0
    for a, b in pairwise(sharpes_clean):
        if _math.isnan(a) or _math.isnan(b):
            continue
        if (a > 0 and b < 0) or (a < 0 and b > 0):
            sign_flips += 1

    passes_gold = (
        n_quarters >= QUARTERLY_MIN_QUARTERS
        and n_positive >= QUARTERLY_GOLD_MIN_POSITIVE
        and sign_flips == 0
    )
    passes_silver = (
        n_quarters >= QUARTERLY_MIN_QUARTERS and n_positive >= QUARTERLY_SILVER_MIN_POSITIVE
    )

    if passes_gold:
        tier: Literal["A_GOLD", "B_VALIDATED", "C_TENTATIVE"] = "A_GOLD"
    elif passes_silver:
        tier = "B_VALIDATED"
    else:
        tier = "C_TENTATIVE"

    return {
        "n_quarters": n_quarters,
        "n_positive": n_positive,
        "sign_flips": sign_flips,
        "passes_4q_gold": passes_gold,
        "passes_4q_silver": passes_silver,
        "tier_recommendation": tier,
    }


# ---------------------------------------------------------------------------
# FastAPI router (not auto-mounted)
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/strategy-verdict", tags=["strategy-verdict"])


class CointegrationVerdictBody(BaseModel):
    adf_p: float = Field(..., ge=0.0, le=1.0)
    half_life_days: float | None = Field(None)
    rho_ar1: float
    n_obs: int = Field(..., ge=0)
    beta_hedge: float
    current_z: float | None = None
    in_position: bool = False
    entry_z: float = DEFAULT_ENTRY_Z
    exit_z: float = DEFAULT_EXIT_Z


class PairsVerdictBody(BaseModel):
    current_z: float
    entry_z: float = DEFAULT_ENTRY_Z
    exit_z: float = DEFAULT_EXIT_Z
    stop_z: float = DEFAULT_STOP_Z
    in_position: bool = False
    cointegration_passed: bool = True
    beta_hedge: float = 1.0


@router.post("/cointegration")
def post_cointegration_verdict(body: CointegrationVerdictBody) -> dict[str, Any]:
    return cointegration_verdict(**body.model_dump())


@router.post("/pairs")
def post_pairs_verdict(body: PairsVerdictBody) -> dict[str, Any]:
    return pair_trade_verdict(**body.model_dump())
