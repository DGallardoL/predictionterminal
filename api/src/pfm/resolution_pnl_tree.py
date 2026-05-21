"""Resolution P&L Tree: portfolio-level conditional P&L for a factor's outcomes.

Given a portfolio of equity positions and a single Polymarket factor, this
module computes:

    1. The mark-to-market P&L of every position under each terminal outcome
       (YES / NO) of the factor — i.e. when the factor's probability snaps
       to 1.0 or 0.0 at resolution.
    2. The probability-weighted expected value of the book.
    3. A 95% Value-at-Risk via Monte-Carlo bootstrap of the logit move.

The math for one position is the standard linear factor-model MTM:

    ΔR_i = β_factor_i · Δlogit(p)
    ΔPnL_i_usd = size_usd_i · ΔR_i

where ``Δlogit(p)`` is the logit-space distance between the *current*
probability and the terminal value (clipped to [ε, 1-ε] in line with the
rest of the codebase).

The "tree" framing makes this readable for a portfolio analyst: at the
root sits the factor with its current probability; the two children are
"if YES resolves" and "if NO resolves", each carrying the by-position
breakdown and the total. The Monte-Carlo branch widens this to a
distribution rather than two discrete leaves — useful for
non-binary-resolution risk where the factor might end at, say, 0.83 not
just at 0/1.

Routing
-------
This module owns its :class:`fastapi.APIRouter`; ``main.py`` is left
untouched (per CLAUDE.md). Wire-up::

    from pfm.resolution_pnl_tree import router as pnl_tree_router
    app.include_router(pnl_tree_router)
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from pfm.cache_utils import get_cache
from pfm.model import DEFAULT_EPSILON
from pfm.sources.polymarket import PolymarketClient

logger = logging.getLogger(__name__)


CACHE_TTL_SECONDS: int = 600  # 10 min
NAMESPACE_TREE: str = "pnl_tree"
NAMESPACE_MC: str = "pnl_monte_carlo"

# Sensible defaults for the Monte-Carlo path. The bound on n_paths protects
# against accidental DOS — 100k bootstraps in a 25-position book is the
# upper end of what we want to do synchronously.
DEFAULT_N_PATHS: int = 10_000
MIN_N_PATHS: int = 100
MAX_N_PATHS: int = 100_000

# When the caller does not supply a current_prob (e.g. they don't have a
# live PM client), we fall back to 0.5 — the maximum-entropy prior.
DEFAULT_CURRENT_PROB: float = 0.5

# σ of the logit-move bootstrap. 1.0 is a wide-but-not-explosive prior —
# the user can override per-call.
DEFAULT_BOOTSTRAP_SIGMA: float = 1.0

Outcome = Literal["YES", "NO"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Position(BaseModel):
    """One equity position with its β to the factor under consideration."""

    ticker: str = Field(..., min_length=1, max_length=20)
    size_usd: float = Field(..., description="Notional in USD; sign carries direction.")
    beta_factor: float = Field(
        ...,
        description=(
            "β of this ticker's return on Δlogit of the factor. "
            "Positive → ticker rises when factor prob rises."
        ),
    )

    @field_validator("size_usd")
    @classmethod
    def _finite_size(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("size_usd must be finite")
        return float(v)

    @field_validator("beta_factor")
    @classmethod
    def _finite_beta(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("beta_factor must be finite")
        return float(v)


class TickerLeg(BaseModel):
    """Per-ticker MTM under one outcome."""

    ticker: str
    size_usd: float
    beta_factor: float
    delta_return: float = Field(..., description="β * Δlogit (a log-return).")
    mtm_usd: float = Field(..., description="size_usd * delta_return.")


class Scenario(BaseModel):
    """One terminal outcome with the full per-position breakdown."""

    outcome: Outcome
    prob: float = Field(..., description="Probability assigned to this outcome.")
    delta_logit: float = Field(..., description="logit(terminal) - logit(current).")
    mtm_total_usd: float
    by_ticker: list[TickerLeg]


class TreeResponse(BaseModel):
    factor_id: str
    current_prob: float
    epsilon: float
    n_positions: int
    gross_notional_usd: float
    scenarios: list[Scenario]
    expected_value_usd: float = Field(..., description="Σ_outcome prob * mtm_total_usd.")
    var_95_usd: float = Field(
        ...,
        description=(
            "95% VaR computed as the worse of the two scenario MTMs "
            "(downside risk ⇒ negative number)."
        ),
    )


class MonteCarloResponse(BaseModel):
    factor_id: str
    current_prob: float
    n_paths: int
    bootstrap_sigma: float
    epsilon: float
    n_positions: int
    gross_notional_usd: float
    expected_value_usd: float
    median_pnl_usd: float
    std_pnl_usd: float
    var_95_usd: float = Field(..., description="Negative of the 5th-percentile PnL.")
    var_99_usd: float = Field(..., description="Negative of the 1st-percentile PnL.")
    cvar_95_usd: float = Field(..., description="Mean PnL conditional on PnL ≤ 5th pct.")
    percentiles: dict[str, float] = Field(
        ...,
        description="Selected percentiles of the simulated PnL distribution.",
    )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _logit(p: float, eps: float = DEFAULT_EPSILON) -> float:
    p = max(eps, min(1.0 - eps, float(p)))
    return float(np.log(p / (1.0 - p)))


def _clip_prob(p: float) -> float:
    if not np.isfinite(p):
        raise ValueError("current_prob must be finite")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"current_prob must be in [0, 1], got {p}")
    return float(p)


def _normalise_positions(
    positions: list[dict] | list[Position],
    beta_map: dict[str, float] | None,
) -> list[Position]:
    """Coerce ``positions`` to ``Position`` instances, applying ``beta_map`` overrides."""
    out: list[Position] = []
    for raw in positions:
        pos = raw if isinstance(raw, Position) else Position(**raw)
        if beta_map and pos.ticker in beta_map:
            pos = pos.model_copy(update={"beta_factor": float(beta_map[pos.ticker])})
        out.append(pos)
    if not out:
        raise ValueError("positions must be a non-empty list")
    return out


# ---------------------------------------------------------------------------
# Core: build_pnl_tree
# ---------------------------------------------------------------------------


def build_pnl_tree(
    positions: list[dict] | list[Position],
    factor_id: str,
    *,
    current_prob: float = DEFAULT_CURRENT_PROB,
    beta_map: dict[str, float] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> dict:
    """Compute the YES/NO conditional MTM tree for ``positions`` on ``factor_id``.

    Args:
        positions: Each position is ``{ticker, size_usd, beta_factor}``.
            Supply ``beta_map={ticker: β}`` to override on the fly without
            re-shaping the input.
        factor_id: Identifier passed back in the response (no lookup).
        current_prob: Current Polymarket probability (or 0.5 default). The
            two scenarios snap this to (1 - ε) and ε respectively.
        beta_map: Optional override map applied per position by ticker.
        epsilon: Logit-clip bound; matches ``model.DEFAULT_EPSILON``.

    Returns:
        Dict matching :class:`TreeResponse`.
    """
    pos_list = _normalise_positions(positions, beta_map)
    p = _clip_prob(current_prob)

    logit_now = _logit(p, eps=epsilon)
    logit_yes = _logit(1.0 - epsilon, eps=epsilon)
    logit_no = _logit(epsilon, eps=epsilon)
    delta_yes = logit_yes - logit_now
    delta_no = logit_no - logit_now

    def _scenario(outcome: Outcome, prob_outcome: float, delta: float) -> Scenario:
        legs: list[TickerLeg] = []
        total = 0.0
        for pos in pos_list:
            dr = float(pos.beta_factor) * float(delta)
            mtm = float(pos.size_usd) * dr
            total += mtm
            legs.append(
                TickerLeg(
                    ticker=pos.ticker,
                    size_usd=pos.size_usd,
                    beta_factor=pos.beta_factor,
                    delta_return=dr,
                    mtm_usd=mtm,
                )
            )
        return Scenario(
            outcome=outcome,
            prob=float(prob_outcome),
            delta_logit=float(delta),
            mtm_total_usd=float(total),
            by_ticker=legs,
        )

    scenarios = [
        _scenario("YES", p, delta_yes),
        _scenario("NO", 1.0 - p, delta_no),
    ]

    expected_value = sum(s.prob * s.mtm_total_usd for s in scenarios)
    # 95% VaR (binary case): the worse of the two outcomes if it has prob ≥ 5%,
    # else 0 — a tail-event below the 5% threshold can be argued to fall
    # outside the "95% VaR" framing. Negative number indicates a loss.
    worst = min(scenarios, key=lambda s: s.mtm_total_usd)
    var_95 = float(worst.mtm_total_usd) if worst.prob >= 0.05 else 0.0

    gross_notional = sum(abs(pos.size_usd) for pos in pos_list)

    response = TreeResponse(
        factor_id=factor_id,
        current_prob=p,
        epsilon=float(epsilon),
        n_positions=len(pos_list),
        gross_notional_usd=float(gross_notional),
        scenarios=scenarios,
        expected_value_usd=float(expected_value),
        var_95_usd=var_95,
    )
    return response.model_dump()


# ---------------------------------------------------------------------------
# Monte Carlo P&L distribution
# ---------------------------------------------------------------------------


def monte_carlo_pnl(
    positions: list[dict] | list[Position],
    factor_id: str,
    n_paths: int = DEFAULT_N_PATHS,
    *,
    current_prob: float = DEFAULT_CURRENT_PROB,
    beta_map: dict[str, float] | None = None,
    epsilon: float = DEFAULT_EPSILON,
    bootstrap_sigma: float = DEFAULT_BOOTSTRAP_SIGMA,
    seed: int | None = None,
) -> dict:
    """Simulate ``n_paths`` Δlogit moves and return the PnL distribution.

    Method:
      1. Sample Δlogit ~ N(0, σ²) for each path. σ defaults to 1.0 — wide
         enough to span "no move" through "near-resolution" outcomes.
      2. For each path, the book PnL is::

             pnl_path = Σ_i  size_usd_i · β_i · Δlogit_path

         (linear in Δlogit, so the loop collapses to a single dot product.)
      3. Report mean / median / std / VaR-95 / VaR-99 / CVaR-95 plus a
         dict of common percentiles for the UI.

    The user is free to interpret the bootstrap as "resolution risk" with
    a wide σ or "intraday-shock risk" with a narrow σ — the math is the
    same. Wider σ ⇒ wider distribution ⇒ larger VaR.
    """
    pos_list = _normalise_positions(positions, beta_map)
    if not MIN_N_PATHS <= int(n_paths) <= MAX_N_PATHS:
        raise ValueError(f"n_paths must be in [{MIN_N_PATHS}, {MAX_N_PATHS}], got {n_paths}")
    if bootstrap_sigma <= 0.0 or not np.isfinite(bootstrap_sigma):
        raise ValueError(f"bootstrap_sigma must be > 0, got {bootstrap_sigma}")

    p = _clip_prob(current_prob)
    rng = np.random.default_rng(seed)
    delta_logit_paths = rng.normal(loc=0.0, scale=float(bootstrap_sigma), size=int(n_paths))

    sizes = np.array([pos.size_usd for pos in pos_list], dtype=float)
    betas = np.array([pos.beta_factor for pos in pos_list], dtype=float)
    # Σ_i size_i × β_i is the book's "exposure" to a unit Δlogit.
    book_exposure = float((sizes * betas).sum())
    pnl_paths = book_exposure * delta_logit_paths

    expected_value = float(np.mean(pnl_paths))
    median_pnl = float(np.median(pnl_paths))
    std_pnl = float(np.std(pnl_paths, ddof=1)) if len(pnl_paths) > 1 else 0.0

    pct_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pcts = np.percentile(pnl_paths, pct_levels)
    percentiles = {f"p{lvl}": float(v) for lvl, v in zip(pct_levels, pcts, strict=True)}

    p5 = percentiles["p5"]
    p1 = percentiles["p1"]
    var_95 = -p5  # report as a positive loss number
    var_99 = -p1
    tail_mask = pnl_paths <= p5
    cvar_95 = float(pnl_paths[tail_mask].mean()) if tail_mask.any() else float(p5)

    gross_notional = float(np.sum(np.abs(sizes)))

    response = MonteCarloResponse(
        factor_id=factor_id,
        current_prob=p,
        n_paths=int(n_paths),
        bootstrap_sigma=float(bootstrap_sigma),
        epsilon=float(epsilon),
        n_positions=len(pos_list),
        gross_notional_usd=gross_notional,
        expected_value_usd=expected_value,
        median_pnl_usd=median_pnl,
        std_pnl_usd=std_pnl,
        var_95_usd=float(var_95),
        var_99_usd=float(var_99),
        cvar_95_usd=cvar_95,
        percentiles=percentiles,
    )
    return response.model_dump()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/portfolio", tags=["portfolio-pnl-tree"])


def _get_polymarket_client(request: Request) -> PolymarketClient | None:
    return getattr(request.app.state, "poly", None)


class _TreeRequest(BaseModel):
    positions: list[Position] = Field(..., min_length=1, max_length=200)
    factor_id: str = Field(..., min_length=1, max_length=200)
    current_prob: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Current PM prob; if null and a poly client is wired, fetched live.",
    )
    beta_map: dict[str, float] | None = None
    epsilon: float = Field(DEFAULT_EPSILON, gt=0.0, lt=0.5)


class _MCRequest(BaseModel):
    positions: list[Position] = Field(..., min_length=1, max_length=200)
    factor_id: str = Field(..., min_length=1, max_length=200)
    n_paths: int = Field(DEFAULT_N_PATHS, ge=MIN_N_PATHS, le=MAX_N_PATHS)
    current_prob: float | None = Field(None, ge=0.0, le=1.0)
    beta_map: dict[str, float] | None = None
    epsilon: float = Field(DEFAULT_EPSILON, gt=0.0, lt=0.5)
    bootstrap_sigma: float = Field(DEFAULT_BOOTSTRAP_SIGMA, gt=0.0, le=10.0)
    seed: int | None = Field(None, ge=0)


def _resolve_current_prob(
    req_prob: float | None,
    factor_id: str,
    poly: PolymarketClient | None,
) -> float:
    """If the request did not pin ``current_prob``, try a live fetch.

    Failures fall back to ``DEFAULT_CURRENT_PROB`` so the endpoint stays
    usable in offline tests / no-poly deployments.
    """
    if req_prob is not None:
        return float(req_prob)
    if poly is None:
        return DEFAULT_CURRENT_PROB
    try:
        meta = poly.get_market_metadata(factor_id)
        # We treat factor_id as the slug for live-fetch purposes; many of
        # the registered factors are 1:1 with Polymarket slugs.
        df = (
            poly.fetch_factor_history(meta.yes_token_id)
            if hasattr(poly, "fetch_factor_history")
            else None
        )
        if df is not None and not df.empty and "price" in df:
            return float(df["price"].iloc[-1])
    except Exception as e:
        logger.warning("live current_prob fetch failed for %s: %s", factor_id, e)
    return DEFAULT_CURRENT_PROB


@router.post(
    "/resolution-tree",
    response_model=TreeResponse,
    summary="Conditional MTM tree (YES vs NO outcome) for a portfolio on a factor.",
)
def post_tree(
    body: Annotated[_TreeRequest, Body()],
    poly: Annotated[PolymarketClient | None, Depends(_get_polymarket_client)] = None,
) -> TreeResponse:
    cache = get_cache(NAMESPACE_TREE, ttl=CACHE_TTL_SECONDS)
    cache_key = (
        "tree",
        body.factor_id,
        repr([(p.ticker, p.size_usd, p.beta_factor) for p in body.positions]),
        body.current_prob,
        repr(sorted((body.beta_map or {}).items())),
        body.epsilon,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return TreeResponse(**cached)

    try:
        prob = _resolve_current_prob(body.current_prob, body.factor_id, poly)
        payload = build_pnl_tree(
            positions=[p.model_dump() for p in body.positions],
            factor_id=body.factor_id,
            current_prob=prob,
            beta_map=body.beta_map,
            epsilon=body.epsilon,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    cache.set(cache_key, payload)
    return TreeResponse(**payload)


@router.post(
    "/pnl-monte-carlo",
    response_model=MonteCarloResponse,
    summary="Monte-Carlo P&L distribution from N bootstrapped Δlogit paths.",
)
def post_monte_carlo(
    body: Annotated[_MCRequest, Body()],
    poly: Annotated[PolymarketClient | None, Depends(_get_polymarket_client)] = None,
) -> MonteCarloResponse:
    cache = get_cache(NAMESPACE_MC, ttl=CACHE_TTL_SECONDS)
    cache_key = (
        "mc",
        body.factor_id,
        repr([(p.ticker, p.size_usd, p.beta_factor) for p in body.positions]),
        body.n_paths,
        body.current_prob,
        repr(sorted((body.beta_map or {}).items())),
        body.epsilon,
        body.bootstrap_sigma,
        body.seed,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return MonteCarloResponse(**cached)

    try:
        prob = _resolve_current_prob(body.current_prob, body.factor_id, poly)
        payload = monte_carlo_pnl(
            positions=[p.model_dump() for p in body.positions],
            factor_id=body.factor_id,
            n_paths=body.n_paths,
            current_prob=prob,
            beta_map=body.beta_map,
            epsilon=body.epsilon,
            bootstrap_sigma=body.bootstrap_sigma,
            seed=body.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    cache.set(cache_key, payload)
    return MonteCarloResponse(**payload)


__all__ = [
    "CACHE_TTL_SECONDS",
    "DEFAULT_BOOTSTRAP_SIGMA",
    "DEFAULT_CURRENT_PROB",
    "DEFAULT_N_PATHS",
    "MAX_N_PATHS",
    "MIN_N_PATHS",
    "MonteCarloResponse",
    "Position",
    "Scenario",
    "TickerLeg",
    "TreeResponse",
    "build_pnl_tree",
    "monte_carlo_pnl",
    "router",
]
