"""``GET /strategies/{pair_id}/sensitivity`` — Greeks-like parameter sensitivity (W13-16).

This router exposes a uniform "perturbation analysis" for any strategy in the
curated catalog (``web/data/alpha_strategies.json``). For each tunable input
parameter — ``kelly_cap``, ``z_threshold``, ``epsilon`` — we perturb the
baseline by ±10% and report the resulting PnL delta. The gradient-norm of
the assembled local sensitivities is reported as a single scalar
"robustness" score (smaller = the strategy is locally insensitive to its
free parameters, which is the desirable property).

This is the strategy-level analogue of an options-Greeks dashboard, with
the perturbation scheme intentionally kept simple (one-step central
differences, ±10% relative perturbation) so it remains cheap to compute and
easy to interpret. It is **not** a calibration tool — it is a *diagnostic*.

Why ±10%?
~~~~~~~~~~
The choice mirrors the "halve / double" sanity check used in the v17 gate.
A 10% relative bump is large enough that a smooth-on-paper PnL function
moves measurably, but small enough that we remain in the locally-linear
regime where a finite-difference reads as a partial derivative. Tests
exercise both this default and a custom step size.

PnL model
~~~~~~~~~
We do **not** require a live backtest to compute the response — that would
make the endpoint useless for any pair without a JSON-recorded equity
curve. Instead we evaluate a deterministic, closed-form **proxy** PnL
function that captures the qualitative dependencies the three parameters
have on a Kelly-sized mean-reversion strategy:

* ``kelly_cap`` (``f``): expected PnL grows linearly until the Kelly
  fraction binds, after which extra cap delivers no further mean return
  but continues to pile on variance — modelled as a clamped-linear minus
  a small quadratic penalty.
* ``z_threshold`` (``z``): the chance of triggering a trade falls off as
  ``Φ(-z) * 2`` (two-sided), and the per-trade edge grows roughly linearly
  in ``z``; multiplied together we get ``z * Φ(-z)`` which peaks near
  ``z ≈ 1``.
* ``epsilon`` (``ε``): logit-clipping floor. A larger ε bleeds signal —
  modelled as ``exp(-α ε)`` decay on the expected per-trade move.

Callers may inject their own ``pnl_fn`` (used by the test suite to plug in
synthetic strategies with known closed-form gradients) — the router's HTTP
handler always uses the proxy unless overridden via the module-level
``_PNL_FN_OVERRIDE`` hook.

Integration note
~~~~~~~~~~~~~~~~
Mount the router from the ``main.py:routes`` owner via::

    from pfm.strategies.sensitivity_router import router as _sens_router
    app.include_router(_sens_router)

We do **not** edit ``main.py`` here because that file is partitioned by
section in the V2 coordination protocol; see ``.coordination/PROTOCOL-V2.md``.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/strategies", tags=["strategies-sensitivity"])

#: Default relative perturbation size: ±10% of the baseline value.
DEFAULT_PERTURBATION: float = 0.10

#: The closed set of parameters this router perturbs. The order is stable
#: so consumers can rely on the response's ``params`` list ordering.
PARAM_NAMES: tuple[str, ...] = ("kelly_cap", "z_threshold", "epsilon")

#: Default baselines used when a strategy's row in ``alpha_strategies.json``
#: does not explicitly state a value. These mirror the CLAUDE.md notes:
#: ε defaults to 0.01, Kelly is conservative at 25%, and the entry-z is
#: the alpha_strategies "rule_entry_z" mean (~1.5).
DEFAULT_BASELINES: dict[str, float] = {
    "kelly_cap": 0.25,
    "z_threshold": 1.5,
    "epsilon": 0.01,
}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ParamSensitivity(BaseModel):
    """One row in the response — one param's ±perturbation outcome."""

    name: str = Field(..., description="Parameter name: kelly_cap | z_threshold | epsilon.")
    baseline: float = Field(..., description="Baseline value used for the strategy.")
    perturbed_low: float = Field(
        ..., description="Baseline scaled down by ``1 - perturbation`` (e.g. -10%)."
    )
    perturbed_high: float = Field(
        ..., description="Baseline scaled up by ``1 + perturbation`` (e.g. +10%)."
    )
    pnl_baseline: float = Field(..., description="PnL at baseline parameters.")
    pnl_delta_low: float = Field(
        ..., description="``pnl(perturbed_low) - pnl(baseline)`` — signed."
    )
    pnl_delta_high: float = Field(
        ..., description="``pnl(perturbed_high) - pnl(baseline)`` — signed."
    )
    local_gradient: float = Field(
        ...,
        description=(
            "Central-difference partial derivative ∂PnL/∂param at baseline. "
            "Computed as ``(pnl_high - pnl_low) / (perturbed_high - perturbed_low)``."
        ),
    )


class SensitivityResponse(BaseModel):
    """Envelope returned by ``GET /strategies/{pair_id}/sensitivity``."""

    pair_id: str = Field(..., min_length=1)
    perturbation: float = Field(..., gt=0.0, le=1.0)
    params: list[ParamSensitivity]
    gradient_norm: float = Field(
        ...,
        ge=0.0,
        description=(
            "Euclidean norm of the per-parameter ``pnl_delta_high - pnl_delta_low`` "
            "vector. Acts as a single robustness score: smaller is more robust."
        ),
    )
    pnl_baseline: float = Field(..., description="PnL at baseline parameters.")
    source: str = Field(
        ...,
        description=(
            "``'json'`` if the baseline was sourced from alpha_strategies.json, "
            "``'defaults'`` if the row had no parameter fields and fell back, "
            "``'override'`` if the caller passed query-param overrides."
        ),
    )


# ---------------------------------------------------------------------------
# Built-in proxy PnL (deterministic, closed-form)
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal CDF — pure-Python via ``math.erf`` (no scipy dep)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def proxy_pnl(params: Mapping[str, float]) -> float:
    """Deterministic proxy PnL.

    Encodes the three textbook dependencies described in the module docstring:

    1. **Kelly cap** has positive marginal value up to a saturation point at
       ``f* = 0.5``; thereafter overbetting introduces a quadratic penalty.
    2. **z-threshold** balances trigger frequency against per-trade edge —
       ``z * 2 * Φ(-z)``-style.
    3. **epsilon** (logit-clipping floor) decays per-trade move as ``e^{-αε}``.

    The function is C¹ everywhere, monotone in the right pieces, and small
    enough to evaluate millions of times per second.
    """
    f = float(params.get("kelly_cap", DEFAULT_BASELINES["kelly_cap"]))
    z = float(params.get("z_threshold", DEFAULT_BASELINES["z_threshold"]))
    eps = float(params.get("epsilon", DEFAULT_BASELINES["epsilon"]))

    # Defensive: refuse negatives / NaN — return 0 so the perturbation step
    # still terminates without raising on a pathological override.
    if not (math.isfinite(f) and math.isfinite(z) and math.isfinite(eps)):
        return 0.0
    f = max(0.0, f)
    z = max(0.0, z)
    eps = max(0.0, eps)

    # 1. Kelly piece: linear up to f*=0.5, then quadratic penalty.
    f_star = 0.5
    if f <= f_star:
        kelly_score = f
    else:
        kelly_score = f_star - 0.5 * (f - f_star) ** 2

    # 2. Z piece: probability of trade × per-trade edge.
    z_score = z * 2.0 * (1.0 - _norm_cdf(z))  # = 2 z Φ(-z)

    # 3. Epsilon piece: exponential decay (α = 8 keeps the response sharp
    # around the default ε=0.01).
    eps_score = math.exp(-8.0 * eps)

    return kelly_score * z_score * eps_score


#: Optional override hook for tests / future calibration plugins. When set
#: (typically via :func:`set_pnl_override`), the HTTP handler swaps it in.
_PNL_FN_OVERRIDE: Callable[[Mapping[str, float]], float] | None = None


def set_pnl_override(fn: Callable[[Mapping[str, float]], float] | None) -> None:
    """Install (or clear, with ``None``) a custom PnL function for the handler.

    Tests use this to inject closed-form synthetic strategies; production
    callers should leave it untouched.
    """
    global _PNL_FN_OVERRIDE
    _PNL_FN_OVERRIDE = fn


# ---------------------------------------------------------------------------
# Sensitivity computation
# ---------------------------------------------------------------------------


def compute_sensitivity(
    baseline: Mapping[str, float],
    pnl_fn: Callable[[Mapping[str, float]], float] | None = None,
    perturbation: float = DEFAULT_PERTURBATION,
) -> tuple[list[ParamSensitivity], float, float]:
    """Compute Greeks-like sensitivities at ``baseline``.

    Parameters
    ----------
    baseline:
        Mapping of parameter name -> baseline value. Must include every
        name in :data:`PARAM_NAMES` (callers that want to opt-out of one
        parameter can simply not include it; missing keys are silently
        skipped).
    pnl_fn:
        Function ``params -> pnl``. Defaults to :func:`proxy_pnl`.
    perturbation:
        Fractional relative bump. Must be in ``(0, 1]``.

    Returns
    -------
    (rows, gradient_norm, pnl_baseline)
        ``rows`` is a list of :class:`ParamSensitivity` — one per perturbed
        parameter, in :data:`PARAM_NAMES` order. ``gradient_norm`` is the
        Euclidean norm of the per-parameter ``pnl_high - pnl_low`` vector
        (a robustness score: smaller is better). ``pnl_baseline`` is the
        PnL evaluated at the unperturbed baseline.

    Raises
    ------
    ValueError
        If ``perturbation`` is not in ``(0, 1]``.
    """
    if not (0.0 < perturbation <= 1.0):
        raise ValueError(f"perturbation must be in (0, 1]; got {perturbation!r}")

    fn = pnl_fn if pnl_fn is not None else proxy_pnl
    pnl_baseline = float(fn(dict(baseline)))

    rows: list[ParamSensitivity] = []
    deltas_for_norm: list[float] = []

    for name in PARAM_NAMES:
        if name not in baseline:
            continue
        b = float(baseline[name])
        low = b * (1.0 - perturbation)
        high = b * (1.0 + perturbation)

        low_params = dict(baseline)
        low_params[name] = low
        high_params = dict(baseline)
        high_params[name] = high

        pnl_low = float(fn(low_params))
        pnl_high = float(fn(high_params))

        delta_low = pnl_low - pnl_baseline
        delta_high = pnl_high - pnl_baseline

        denom = high - low
        local_grad = (pnl_high - pnl_low) / denom if denom != 0.0 else 0.0

        rows.append(
            ParamSensitivity(
                name=name,
                baseline=b,
                perturbed_low=low,
                perturbed_high=high,
                pnl_baseline=pnl_baseline,
                pnl_delta_low=delta_low,
                pnl_delta_high=delta_high,
                local_gradient=local_grad,
            )
        )
        deltas_for_norm.append(pnl_high - pnl_low)

    gradient_norm = math.sqrt(sum(d * d for d in deltas_for_norm))
    return rows, gradient_norm, pnl_baseline


# ---------------------------------------------------------------------------
# Strategy lookup — discover baseline params from alpha_strategies.json
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
# strategies/ -> pfm/ -> src/ -> api/ -> repo-root
_REPO_ROOT: Path = _HERE.parents[4]
_DEFAULT_STRATEGIES_PATH: Path = _REPO_ROOT / "web" / "data" / "alpha_strategies.json"
_STRATEGIES_PATH_ENV: str = "PFM_SENSITIVITY_STRATEGIES_PATH"


def _strategies_path() -> Path:
    override = os.environ.get(_STRATEGIES_PATH_ENV)
    if override:
        return Path(override)
    return _DEFAULT_STRATEGIES_PATH


def _load_strategy_row(pair_id: str) -> dict[str, Any] | None:
    """Return the JSON row for ``pair_id`` from alpha_strategies.json, or None."""
    path = _strategies_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    strategies = doc.get("strategies") if isinstance(doc, dict) else None
    if not isinstance(strategies, list):
        return None
    for row in strategies:
        if isinstance(row, dict) and row.get("pair_id") == pair_id:
            return row
    return None


def _baseline_from_row(row: dict[str, Any]) -> tuple[dict[str, float], bool]:
    """Extract baseline parameters from a strategy row.

    Returns ``(params, sourced_from_json)`` where ``sourced_from_json`` is
    True if at least one parameter was lifted from the row's own fields.
    """
    sourced = False
    out = dict(DEFAULT_BASELINES)

    # Kelly cap: prefer suggested_allocation; clamp to (0, 1].
    sa = row.get("suggested_allocation")
    if isinstance(sa, (int, float)) and 0.0 < float(sa) <= 1.0:
        out["kelly_cap"] = float(sa)
        sourced = True

    # z_threshold: rule_entry_z.
    rez = row.get("rule_entry_z")
    if isinstance(rez, (int, float)) and float(rez) > 0.0:
        out["z_threshold"] = float(rez)
        sourced = True

    # epsilon: deployment_params.epsilon or strategy_spec.epsilon if present.
    for blob_key in ("deployment_params", "strategy_spec", "parameters"):
        blob = row.get(blob_key)
        if isinstance(blob, dict):
            eps = blob.get("epsilon") or blob.get("eps") or blob.get("clip_epsilon")
            if isinstance(eps, (int, float)) and float(eps) > 0.0:
                out["epsilon"] = float(eps)
                sourced = True
                break

    return out, sourced


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{pair_id}/sensitivity",
    response_model=SensitivityResponse,
    summary="Greeks-like parameter sensitivity for a strategy",
)
def get_strategy_sensitivity(
    pair_id: str,
    perturbation: float = Query(
        DEFAULT_PERTURBATION,
        gt=0.0,
        le=1.0,
        description="Relative perturbation size (default 0.10 = ±10%).",
    ),
    kelly_cap: float | None = Query(
        None, gt=0.0, le=1.0, description="Override baseline kelly_cap."
    ),
    z_threshold: float | None = Query(None, gt=0.0, description="Override baseline z_threshold."),
    epsilon: float | None = Query(None, gt=0.0, lt=1.0, description="Override baseline epsilon."),
) -> SensitivityResponse:
    """Return a per-parameter ±perturbation PnL-delta table.

    The handler looks the strategy up in ``alpha_strategies.json`` to seed
    baselines, then applies any query-param overrides. If no row is found
    and no overrides are supplied, the canonical defaults are used.
    """
    if not pair_id or not pair_id.strip():
        raise HTTPException(status_code=400, detail="pair_id is required")

    row = _load_strategy_row(pair_id)

    baseline: dict[str, float]
    source: str
    overrides_applied = any(v is not None for v in (kelly_cap, z_threshold, epsilon))

    if row is None and not overrides_applied:
        # Unknown strategy and no overrides — surface a 404 so callers know
        # they got the canonical defaults rather than a real baseline.
        raise HTTPException(
            status_code=404,
            detail=(
                f"pair_id={pair_id!r} not found in alpha_strategies.json and no "
                "overrides supplied (pass kelly_cap/z_threshold/epsilon to "
                "evaluate ad-hoc parameter sets)."
            ),
        )
    elif row is not None:
        baseline, sourced = _baseline_from_row(row)
        source = "json" if sourced else "defaults"
    else:
        baseline = dict(DEFAULT_BASELINES)
        source = "override"

    if kelly_cap is not None:
        baseline["kelly_cap"] = float(kelly_cap)
        source = "override"
    if z_threshold is not None:
        baseline["z_threshold"] = float(z_threshold)
        source = "override"
    if epsilon is not None:
        baseline["epsilon"] = float(epsilon)
        source = "override"

    fn = _PNL_FN_OVERRIDE if _PNL_FN_OVERRIDE is not None else proxy_pnl
    rows, gradient_norm, pnl_baseline = compute_sensitivity(
        baseline, pnl_fn=fn, perturbation=perturbation
    )

    return SensitivityResponse(
        pair_id=pair_id,
        perturbation=perturbation,
        params=rows,
        gradient_norm=gradient_norm,
        pnl_baseline=pnl_baseline,
        source=source,
    )


__all__ = [
    "DEFAULT_BASELINES",
    "DEFAULT_PERTURBATION",
    "PARAM_NAMES",
    "ParamSensitivity",
    "SensitivityResponse",
    "compute_sensitivity",
    "get_strategy_sensitivity",
    "proxy_pnl",
    "router",
    "set_pnl_override",
]
