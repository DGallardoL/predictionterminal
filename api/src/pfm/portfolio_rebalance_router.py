"""``POST /portfolio/{handle}/rebalance`` — propose rebalance trades.

Given a portfolio handle previously returned by ``POST /portfolio/import``
(see :mod:`pfm.portfolio_import_router`), compute the per-ticker trades
needed to move from the current share-weighted allocation to a target
weight vector supplied by the caller.

Algorithm
---------
1. Look up the :class:`Portfolio` for ``handle`` from
   ``app.state.portfolios``. 404 if missing.
2. Compute the current market value of each held position using
   ``current_prices`` (caller-supplied dictionary keyed by ticker, USD).
   Missing prices for held tickers → ``422`` validation error.
3. Compute total portfolio value
   :math:`V = \\sum_i s_i \\cdot p_i`.
4. For each ticker in ``target_weights`` (or currently held), compute::

       target_value  = V * target_weights.get(ticker, 0.0)
       target_shares = target_value / price
       delta         = target_shares - current_shares

5. Classify ``delta`` into ``buy`` / ``sell`` / ``hold`` using a small
   tolerance (``HOLD_TOLERANCE_SHARES``) so floating-point noise around
   zero is not reported as a trade.

Notes / design choices
----------------------
* **Target weights need not sum to 1.0.** A user may want to leave a
  cash sleeve (sum < 1) or temporarily over-allocate during a
  transition (sum > 1, e.g. when using leverage). We surface
  ``cash_weight = 1 - sum(target_weights.values())`` in the response
  but never reject. We DO reject any individual negative weight
  (shorts unsupported by the importer) and any weight > 1.0.
* **Prices must be strictly positive.** Zero or negative prices return
  ``422`` — they would yield nonsense target_shares (division by zero
  or sign flip).
* **Tickers in target_weights but not currently held** are treated as
  buys from zero. Tickers held but missing from target_weights are
  treated as full sells (target weight = 0). This is the conventional
  semantics for "rebalance to target".
* **Fractional shares** are emitted as-is (Robinhood-style). Callers
  that need whole-share rounding can floor/round downstream.
* **The portfolio store is NOT mutated.** This endpoint is a pure
  proposal — the caller decides whether to execute. A future
  ``/portfolio/{handle}/execute`` task can take this response and
  update the store.

Integration note
----------------
Mounting on ``pfm.main`` is left to the ``main.py:routes`` claim
holder. When unblocked add::

    from pfm.portfolio_rebalance_router import router as _rebalance_router
    app.include_router(_rebalance_router)
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from pfm.portfolio_import_router import Portfolio, get_portfolio

router = APIRouter(tags=["portfolio"])


# --- tuning knobs ----------------------------------------------------------

HOLD_TOLERANCE_SHARES = 1e-6
"""Trades smaller than this in absolute share count are reported as ``hold``.

Floating-point arithmetic on ``target_shares - current_shares`` produces
sub-nanoshare noise even when the target equals the current allocation.
``1e-6`` is well below the smallest fractional-share unit any retail
broker supports (Robinhood uses ``1e-6``; Fidelity uses ``1e-3``)."""

MAX_TICKERS_PER_REQUEST = 500
"""Defensive cap so a malformed request can't fan out unboundedly."""


# --- request / response schemas --------------------------------------------


Action = Literal["buy", "sell", "hold"]


class RebalanceRequest(BaseModel):
    """Body of ``POST /portfolio/{handle}/rebalance``."""

    target_weights: dict[str, float] = Field(
        ...,
        description=(
            "Map of UPPERCASE ticker -> target portfolio weight in [0, 1]. "
            "Need not sum to 1 (any shortfall is reported as cash_weight). "
            "Tickers currently held but absent here are treated as fully "
            "sold (target weight = 0)."
        ),
        examples=[{"NVDA": 0.20, "TSLA": 0.15, "AAPL": 0.30}],
    )
    current_prices: dict[str, float] = Field(
        ...,
        description=(
            "Map of UPPERCASE ticker -> current market price in USD. Must "
            "be strictly positive. Must include every ticker that appears "
            "in the current portfolio OR in target_weights."
        ),
        examples=[{"NVDA": 950.0, "TSLA": 240.0, "AAPL": 215.0}],
    )

    @field_validator("target_weights")
    @classmethod
    def _check_weights(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("target_weights must not be empty")
        if len(v) > MAX_TICKERS_PER_REQUEST:
            raise ValueError(f"too many target tickers; max {MAX_TICKERS_PER_REQUEST}")
        out: dict[str, float] = {}
        for k, w in v.items():
            kk = k.strip().upper()
            if not kk:
                raise ValueError("target_weights key must be non-empty")
            if w < 0.0:
                raise ValueError(f"target_weights[{kk!r}] must be >= 0 (got {w})")
            if w > 1.0:
                raise ValueError(f"target_weights[{kk!r}] must be <= 1 (got {w})")
            out[kk] = float(w)
        return out

    @field_validator("current_prices")
    @classmethod
    def _check_prices(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("current_prices must not be empty")
        if len(v) > MAX_TICKERS_PER_REQUEST:
            raise ValueError(f"too many priced tickers; max {MAX_TICKERS_PER_REQUEST}")
        out: dict[str, float] = {}
        for k, p in v.items():
            kk = k.strip().upper()
            if not kk:
                raise ValueError("current_prices key must be non-empty")
            if not (p > 0.0):
                raise ValueError(f"current_prices[{kk!r}] must be > 0 (got {p})")
            out[kk] = float(p)
        return out


class RebalanceTrade(BaseModel):
    """One per-ticker trade proposal in the response."""

    ticker: str = Field(..., examples=["NVDA"])
    current_shares: float = Field(..., ge=0.0)
    target_shares: float = Field(..., ge=0.0)
    delta: float = Field(
        ...,
        description=("target_shares - current_shares. Positive => buy, negative => sell."),
    )
    action: Action
    current_value: float = Field(..., ge=0.0)
    target_value: float = Field(..., ge=0.0)
    current_weight: float = Field(..., ge=0.0)
    target_weight: float = Field(..., ge=0.0)
    price: float = Field(..., gt=0.0)


class RebalanceResponse(BaseModel):
    """Body of ``POST /portfolio/{handle}/rebalance``."""

    handle: str
    total_value: float = Field(
        ...,
        ge=0.0,
        description="Sum of current_shares * price across the portfolio.",
    )
    cash_weight: float = Field(
        ...,
        description=(
            "1 - sum(target_weights.values()). May be slightly negative "
            "due to floating-point rounding; callers should treat |x| "
            "below 1e-9 as zero."
        ),
    )
    trades: list[RebalanceTrade]
    warnings: list[str] = Field(default_factory=list)


# --- core computation ------------------------------------------------------


def _propose_trades(
    portfolio: Portfolio,
    target_weights: dict[str, float],
    current_prices: dict[str, float],
) -> tuple[list[RebalanceTrade], float, list[str]]:
    """Pure-function rebalance computation (no FastAPI dependencies).

    Returns ``(trades, total_value, warnings)``. Tickers appear in a
    deterministic order: held tickers in their import order first, then
    new target-only tickers in target_weights iteration order.
    """

    warnings: list[str] = []

    # Aggregate duplicate-ticker rows from the import (the importer
    # permits them with a warning) so we report one trade per ticker.
    held_shares: dict[str, float] = {}
    held_order: list[str] = []
    for row in portfolio.rows:
        if row.ticker not in held_shares:
            held_order.append(row.ticker)
            held_shares[row.ticker] = 0.0
        held_shares[row.ticker] += float(row.shares)

    # Validate prices: every held ticker must have a price.
    missing_prices = [t for t in held_order if t not in current_prices]
    if missing_prices:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(f"current_prices is missing held tickers: {', '.join(missing_prices)}"),
        )

    # Validate prices for target-only tickers too — we need them to
    # convert target weight to target shares.
    missing_target_prices = [t for t in target_weights if t not in current_prices]
    if missing_target_prices:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"current_prices is missing target tickers: {', '.join(missing_target_prices)}"
            ),
        )

    # Total portfolio market value (held only — cash is implicit).
    total_value = sum(held_shares[t] * current_prices[t] for t in held_order)

    # Build the universe of tickers we'll emit a row for: held + target.
    universe: list[str] = list(held_order)
    for t in target_weights:
        if t not in held_shares:
            universe.append(t)

    trades: list[RebalanceTrade] = []
    for ticker in universe:
        price = current_prices[ticker]
        cur_shares = held_shares.get(ticker, 0.0)
        cur_value = cur_shares * price
        cur_weight = (cur_value / total_value) if total_value > 0 else 0.0

        tgt_weight = target_weights.get(ticker, 0.0)
        tgt_value = total_value * tgt_weight
        tgt_shares = tgt_value / price  # price > 0 enforced

        delta = tgt_shares - cur_shares
        if abs(delta) <= HOLD_TOLERANCE_SHARES:
            action: Action = "hold"
        elif delta > 0:
            action = "buy"
        else:
            action = "sell"

        trades.append(
            RebalanceTrade(
                ticker=ticker,
                current_shares=cur_shares,
                target_shares=tgt_shares,
                delta=delta,
                action=action,
                current_value=cur_value,
                target_value=tgt_value,
                current_weight=cur_weight,
                target_weight=tgt_weight,
                price=price,
            )
        )

    # Warn (don't fail) on held tickers being fully sold because the
    # caller forgot to include them in target_weights. This is the most
    # common foot-gun.
    full_sells = [t for t in held_order if t not in target_weights and held_shares[t] > 0]
    if full_sells:
        warnings.append(
            "held tickers absent from target_weights are fully sold: " + ", ".join(full_sells)
        )

    return trades, total_value, warnings


# --- endpoint --------------------------------------------------------------


@router.post(
    "/portfolio/{handle}/rebalance",
    response_model=RebalanceResponse,
    summary="Propose rebalance trades for an imported portfolio.",
    description=(
        "Given a portfolio handle from `POST /portfolio/import` plus a "
        "target weight vector and current prices, return the per-ticker "
        "buy / sell / hold trade list needed to reach the target "
        "allocation. The portfolio store is NOT mutated; the response "
        "is a proposal only."
    ),
)
async def rebalance_portfolio(
    handle: str, body: RebalanceRequest, request: Request
) -> RebalanceResponse:
    portfolio = get_portfolio(request.app.state, handle)
    trades, total_value, warnings = _propose_trades(
        portfolio, body.target_weights, body.current_prices
    )
    cash_weight = 1.0 - sum(body.target_weights.values())
    return RebalanceResponse(
        handle=handle,
        total_value=total_value,
        cash_weight=cash_weight,
        trades=trades,
        warnings=warnings,
    )
