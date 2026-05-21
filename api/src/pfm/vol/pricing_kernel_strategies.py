"""Relative-value strategies from the Kalshi-vs-options RN mispricing.

Each live Kalshi index contract implies a probability (its price). The SAME
event has a probability under the options risk-neutral density (from the
call-curve second derivative, repriced to the Kalshi horizon) and under the physical density
(GARCH). The gap is a tradeable signal:

* **edge vs options** = ``p_Kalshi − p_options`` — the cross-venue signal we act
  on (sell rich Kalshi contracts, buy cheap ones, hedged with the option-
  replicated digital).
* **edge vs physical** = ``p_Kalshi − p_physical`` — the realised edge if the
  GARCH physical measure is the truth.

We then **Monte-Carlo backtest** several strategies by drawing the terminal level
from the physical density and settling every contract. This is an honest
*point-in-time, model-as-truth* test — NOT a multi-quarter historical backtest,
and it inherits the favorites-longshot caveat (see the project anti-alpha notes).
Treat results as paper/indicative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

_EPS = 1e-9
Action = Literal["SELL", "BUY", "PASS"]


@dataclass
class ContractEdge:
    """One Kalshi contract priced against options-Q and physical-P."""

    slug: str
    kind: str  # "between" | "below" | "above"
    lo: float | None
    hi: float | None
    center: float
    kalshi_prob: float
    options_prob: float
    physical_prob: float
    edge_vs_options: float  # p_Kalshi − p_options (signal)
    edge_vs_physical: float  # p_Kalshi − p_physical (realised edge if P is truth)
    action: Action


@dataclass
class Opportunity:
    """A Kalshi contract priced theoretically vs its EXECUTABLE quote.

    ``fair_value`` is the options-implied risk-neutral probability of
    the event — the theoretical price you could replicate with a tight option
    call/put spread. The edge is measured against the side you can actually
    trade: sell YES at ``kalshi_bid`` (profit if fair < bid) or buy YES at
    ``kalshi_ask`` (profit if fair > ask). A wide/one-sided quote → no edge.
    """

    slug: str
    kind: str
    lo: float | None
    hi: float | None
    fair_value: float
    physical_prob: float
    kalshi_bid: float | None
    kalshi_ask: float | None
    kalshi_mid: float | None
    spread: float | None
    volume: float | None
    open_interest: float | None
    executable: bool
    action: str  # "BUY @ask" | "SELL @bid" | "NONE"
    edge: float  # executable edge in probability points (>0 = real opportunity)
    confidence: str  # "high" if option strikes bracket the bucket, else "low"
    note: str = ""


@dataclass
class StrategyResult:
    """Monte-Carlo P&L summary for one strategy (per $1 staked per leg)."""

    name: str
    description: str
    n_legs: int
    mean_pnl: float  # mean per-$1 P&L across legs and sims, after cost
    pnl_std: float
    sharpe: float  # mean / std (per-trade, not annualised)
    hit_rate: float  # fraction of MC sims with positive portfolio P&L
    expected_edge_vs_physical: float  # mean signed edge vs physical, after cost
    gross_edge_vs_options: float  # mean |signal| acted on
    legs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Event probabilities + edges
# ---------------------------------------------------------------------------


def _cdf_at(grid: np.ndarray, cdf: np.ndarray, x: float) -> float:
    return float(np.interp(x, grid, cdf, left=0.0, right=1.0))


def _event_prob(
    grid: np.ndarray, cdf: np.ndarray, kind: str, lo: float | None, hi: float | None
) -> float:
    """Probability of a Kalshi event under a density given by ``(grid, cdf)``."""
    if kind == "between" and lo is not None and hi is not None:
        return max(_cdf_at(grid, cdf, hi) - _cdf_at(grid, cdf, lo), 0.0)
    if kind == "below" and hi is not None:
        return _cdf_at(grid, cdf, hi)
    if kind == "above" and lo is not None:
        return 1.0 - _cdf_at(grid, cdf, lo)
    return float("nan")


def compute_edges(
    entries: list[Any],
    opt_grid: np.ndarray,
    opt_cdf: np.ndarray,
    phys_grid: np.ndarray,
    phys_cdf: np.ndarray,
    *,
    edge_threshold: float = 0.03,
) -> list[ContractEdge]:
    """Build per-contract edges from raw Kalshi ladder entries.

    Args:
        entries: Kalshi ``LadderEntry`` objects (``direction``/``floor``/``cap``/
            ``strike``/``prob``/``slug``).
        opt_grid, opt_cdf: options-Q density CDF (repriced to the Kalshi horizon).
        phys_grid, phys_cdf: physical-P density CDF.
        edge_threshold: minimum |signal| (probability points) to flag a trade.

    Returns:
        Edges sorted by ``|edge_vs_options|`` descending.
    """
    opt_grid, opt_cdf = np.asarray(opt_grid, float), np.asarray(opt_cdf, float)
    phys_grid, phys_cdf = np.asarray(phys_grid, float), np.asarray(phys_cdf, float)
    out: list[ContractEdge] = []
    for e in entries:
        direction = getattr(e, "direction", None)
        floor, cap, strike = (
            getattr(e, "floor", None),
            getattr(e, "cap", None),
            getattr(e, "strike", None),
        )
        prob = getattr(e, "prob", None)
        if prob is None:
            continue
        if direction == "between":
            kind, lo, hi = "between", floor, cap
        elif direction == "below":
            kind, lo, hi = "below", None, strike
        elif direction == "above":
            kind, lo, hi = "above", strike, None
        else:
            continue
        p_opt = _event_prob(opt_grid, opt_cdf, kind, lo, hi)
        p_phys = _event_prob(phys_grid, phys_cdf, kind, lo, hi)
        if not (np.isfinite(p_opt) and np.isfinite(p_phys)):
            continue
        center = (
            0.5 * (lo + hi)
            if (lo is not None and hi is not None)
            else (hi if hi is not None else lo)
        )
        edge_opt = float(prob) - p_opt
        action: Action = (
            "SELL" if edge_opt > edge_threshold else "BUY" if edge_opt < -edge_threshold else "PASS"
        )
        out.append(
            ContractEdge(
                slug=getattr(e, "slug", "") or "",
                kind=kind,
                lo=lo,
                hi=hi,
                center=float(center) if center is not None else float("nan"),
                kalshi_prob=round(float(prob), 4),
                options_prob=round(p_opt, 4),
                physical_prob=round(p_phys, 4),
                edge_vs_options=round(edge_opt, 4),
                edge_vs_physical=round(float(prob) - p_phys, 4),
                action=action,
            )
        )
    out.sort(key=lambda c: abs(c.edge_vs_options), reverse=True)
    return out


# ---------------------------------------------------------------------------
# Executable fair-value scanner
# ---------------------------------------------------------------------------


def scan_opportunities(
    entries: list[Any],
    opt_grid: np.ndarray,
    opt_cdf: np.ndarray,
    phys_grid: np.ndarray,
    phys_cdf: np.ndarray,
    *,
    opt_strike_lo: float,
    opt_strike_hi: float,
    max_spread: float = 0.12,
    min_edge: float = 0.02,
    discount: float = 1.0,
) -> tuple[list[Opportunity], dict[str, Any]]:
    """Theoretically price every Kalshi contract vs its executable quote.

    Args:
        entries: Kalshi ``LadderEntry`` objects (now carrying ``yes_bid``/
            ``yes_ask``/``volume``/``open_interest``).
        opt_grid, opt_cdf: options-Q CDF (the theoretical pricing source).
        phys_grid, phys_cdf: physical-P CDF (context).
        opt_strike_lo, opt_strike_hi: the strike span the option smile was fit
            over — buckets outside it are extrapolated (low confidence).
        max_spread: max Kalshi bid/ask spread to count a market as executable.
        min_edge: minimum executable edge (probability points) to surface.

    Returns:
        ``(opportunities, summary)`` — opportunities sorted by edge descending,
        plus a market-quality summary (how much of the ladder is actually live).
    """
    opt_grid, opt_cdf = np.asarray(opt_grid, float), np.asarray(opt_cdf, float)
    phys_grid, phys_cdf = np.asarray(phys_grid, float), np.asarray(phys_cdf, float)
    opps: list[Opportunity] = []
    n_total = n_live = 0
    spreads: list[float] = []
    vols: list[float] = []
    for e in entries:
        direction = getattr(e, "direction", None)
        floor, cap, strike = (
            getattr(e, "floor", None),
            getattr(e, "cap", None),
            getattr(e, "strike", None),
        )
        if direction == "between":
            kind, lo, hi = "between", floor, cap
        elif direction == "below":
            kind, lo, hi = "below", None, strike
        elif direction == "above":
            kind, lo, hi = "above", strike, None
        else:
            continue
        n_total += 1
        # Theoretical fair PRICE of the Kalshi contract = discounted RN
        # probability (a binary paying $1 at settlement is worth e^{-rτ}·P_Q).
        # The discount is ~0 for dailies but ~2-3% at multi-month horizons.
        fair = discount * _event_prob(opt_grid, opt_cdf, kind, lo, hi)
        p_phys = _event_prob(phys_grid, phys_cdf, kind, lo, hi)
        if not np.isfinite(fair):
            continue
        bid = getattr(e, "yes_bid", None)
        ask = getattr(e, "yes_ask", None)
        vol = getattr(e, "volume", None)
        oi = getattr(e, "open_interest", None)
        mid = (
            (bid + ask) / 2.0 if (bid is not None and ask is not None) else getattr(e, "prob", None)
        )
        spread = (ask - bid) if (bid is not None and ask is not None) else None
        if spread is not None:
            spreads.append(spread)
        if vol:
            vols.append(float(vol))

        # Executable iff a real two-sided quote with a tradeable spread.
        executable = (
            bid is not None
            and ask is not None
            and bid > 0.0
            and ask < 1.0
            and (ask - bid) <= max_spread
        )
        if executable:
            n_live += 1

        # Edge against the side you can actually trade.
        sell_edge = (bid - fair) if bid is not None else -1.0  # sell YES @bid if fair<bid
        buy_edge = (fair - ask) if ask is not None else -1.0  # buy YES @ask if fair>ask
        if buy_edge >= sell_edge and buy_edge > 0:
            action, edge = "BUY @ask", buy_edge
        elif sell_edge > 0:
            action, edge = "SELL @bid", sell_edge
        else:
            action, edge = "NONE", 0.0

        edges_lo = lo if lo is not None else hi
        edges_hi = hi if hi is not None else lo
        bracketed = (
            edges_lo is not None
            and edges_hi is not None
            and edges_lo >= opt_strike_lo
            and edges_hi <= opt_strike_hi
        )
        confidence = "high" if bracketed else "low"

        note = ""
        if not executable:
            note = "untraded / one-sided quote — not executable"
        elif edge >= min_edge and confidence == "low":
            note = "edge relies on extrapolated option wing — low confidence"

        if action != "NONE" and edge >= min_edge and executable:
            opps.append(
                Opportunity(
                    slug=getattr(e, "slug", "") or "",
                    kind=kind,
                    lo=lo,
                    hi=hi,
                    fair_value=round(fair, 4),
                    physical_prob=round(float(p_phys), 4),
                    kalshi_bid=bid,
                    kalshi_ask=ask,
                    kalshi_mid=round(mid, 4) if mid is not None else None,
                    spread=round(spread, 4) if spread is not None else None,
                    volume=vol,
                    open_interest=oi,
                    executable=executable,
                    action=action,
                    edge=round(edge, 4),
                    confidence=confidence,
                    note=note,
                )
            )
    opps.sort(key=lambda o: o.edge, reverse=True)
    summary = {
        "n_contracts": n_total,
        "n_executable": n_live,
        "n_opportunities": len(opps),
        "median_spread": round(float(np.median(spreads)), 4) if spreads else None,
        "total_volume": float(np.sum(vols)) if vols else 0.0,
        "tradeable": n_live > 0 and len(opps) > 0,
    }
    return opps, summary


def fair_value_rows(
    entries: list[Any],
    opt_grid: np.ndarray,
    opt_cdf: np.ndarray,
    phys_grid: np.ndarray,
    phys_cdf: np.ndarray,
    *,
    discount: float = 1.0,
) -> list[dict[str, Any]]:
    """Per-contract theoretical fair value vs the live Kalshi quote (ALL rows).

    Unlike :func:`scan_opportunities` (which surfaces only edges above a
    threshold), this returns every quoted contract so the UI can show the full
    fair-vs-price ladder and the structural lean even where there's no edge.
    """
    opt_grid, opt_cdf = np.asarray(opt_grid, float), np.asarray(opt_cdf, float)
    phys_grid, phys_cdf = np.asarray(phys_grid, float), np.asarray(phys_cdf, float)
    rows: list[dict[str, Any]] = []
    for e in entries:
        direction = getattr(e, "direction", None)
        floor, cap, strike = (
            getattr(e, "floor", None),
            getattr(e, "cap", None),
            getattr(e, "strike", None),
        )
        if direction == "between":
            kind, lo, hi = "between", floor, cap
        elif direction == "below":
            kind, lo, hi = "below", None, strike
        elif direction == "above":
            kind, lo, hi = "above", strike, None
        else:
            continue
        fair = discount * _event_prob(opt_grid, opt_cdf, kind, lo, hi)
        if not np.isfinite(fair):
            continue
        bid, ask = getattr(e, "yes_bid", None), getattr(e, "yes_ask", None)
        mid = (
            (bid + ask) / 2.0 if (bid is not None and ask is not None) else getattr(e, "prob", None)
        )
        sort_k = lo if lo is not None else (hi if hi is not None else 0.0)
        rows.append(
            {
                "slug": getattr(e, "slug", "") or "",
                "kind": kind,
                "lo": lo,
                "hi": hi,
                "kalshi_bid": bid,
                "kalshi_ask": ask,
                "kalshi_mid": round(mid, 4) if mid is not None else None,
                "fair_value": round(float(fair), 4),
                "physical_prob": round(float(_event_prob(phys_grid, phys_cdf, kind, lo, hi)), 4),
                "gap": round(float(mid - fair), 4) if mid is not None else None,
                "_sort": float(sort_k),
            }
        )
    rows.sort(key=lambda r: r["_sort"])
    for r in rows:
        r.pop("_sort", None)
    return rows


# ---------------------------------------------------------------------------
# Monte-Carlo backtest under the physical measure
# ---------------------------------------------------------------------------


def _sample_from_cdf(
    grid: np.ndarray, cdf: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Inverse-transform sample terminal levels from a density CDF."""
    u = rng.random(n)
    # cdf is non-decreasing in [0,1]; invert by interpolation
    return np.interp(u, np.clip(cdf, 0.0, 1.0), grid)


def _settle(samples: np.ndarray, kind: str, lo: float | None, hi: float | None) -> np.ndarray:
    if kind == "between":
        return ((samples >= lo) & (samples < hi)).astype(float)
    if kind == "below":
        return (samples < hi).astype(float)
    return (samples >= lo).astype(float)  # above


def _leg_pnl(action: Action, price: float, settle: np.ndarray, cost: float) -> np.ndarray:
    """Per-$1 P&L of one leg across MC draws, after round-trip cost."""
    if action == "BUY":
        return settle - price - cost
    return price - settle - cost  # SELL


def backtest_strategies(
    edges: list[ContractEdge],
    phys_grid: np.ndarray,
    phys_cdf: np.ndarray,
    *,
    cost: float = 0.01,
    n_sims: int = 20000,
    tail_moneyness: float = 0.01,
    seed: int = 7,
) -> list[StrategyResult]:
    """Monte-Carlo P&L of several relative-value strategies under physical-P.

    Args:
        edges: per-contract edges from :func:`compute_edges`.
        phys_grid, phys_cdf: physical density to draw the terminal level from.
        cost: round-trip transaction cost per leg, in probability points.
        n_sims: Monte-Carlo draws.
        tail_moneyness: |center−forward|/forward beyond which a contract is a
            "tail" for the fade-tails strategy.
        seed: RNG seed for reproducibility.

    Returns:
        One :class:`StrategyResult` per strategy.
    """
    if not edges:
        return []
    rng = np.random.default_rng(seed)
    phys_grid, phys_cdf = np.asarray(phys_grid, float), np.asarray(phys_cdf, float)
    samples = _sample_from_cdf(phys_grid, phys_cdf, n_sims, rng)
    # forward proxy for tail classification = physical median
    fwd = float(np.interp(0.5, np.clip(phys_cdf, 0, 1), phys_grid))

    def _run(name: str, desc: str, picks: list[tuple[ContractEdge, Action]]) -> StrategyResult:
        if not picks:
            return StrategyResult(name, desc, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
        leg_pnls = []  # each: array over sims
        edge_phys = []
        gross_sig = []
        for e, act in picks:
            settle = _settle(samples, e.kind, e.lo, e.hi)
            leg_pnls.append(_leg_pnl(act, e.kalshi_prob, settle, cost))
            sign = 1.0 if act == "BUY" else -1.0
            edge_phys.append(sign * (e.physical_prob - e.kalshi_prob) - cost)
            gross_sig.append(abs(e.edge_vs_options))
        port = np.mean(np.vstack(leg_pnls), axis=0)  # equal-$1 portfolio per sim
        mean = float(port.mean())
        std = float(port.std()) or _EPS
        return StrategyResult(
            name=name,
            description=desc,
            n_legs=len(picks),
            mean_pnl=round(mean, 4),
            pnl_std=round(std, 4),
            sharpe=round(mean / std, 3),
            hit_rate=round(float((port > 0).mean()), 3),
            expected_edge_vs_physical=round(float(np.mean(edge_phys)), 4),
            gross_edge_vs_options=round(float(np.mean(gross_sig)), 4),
            legs=[e.slug for e, _ in picks][:25],
        )

    fade_rich = [(e, "SELL") for e in edges if e.action == "SELL"]
    buy_cheap = [(e, "BUY") for e in edges if e.action == "BUY"]
    combined = fade_rich + buy_cheap
    tails = [
        (e, "SELL")
        for e in edges
        if np.isfinite(e.center) and abs(e.center - fwd) / max(fwd, 1.0) > tail_moneyness
    ]
    naive = [(e, "BUY") for e in edges]  # benchmark: buy everything (pays the vig)

    return [
        _run(
            "fade_rich_vs_options",
            "Sell Kalshi contracts richer than the options digital",
            fade_rich,
        ),
        _run(
            "buy_cheap_vs_options",
            "Buy Kalshi contracts cheaper than the options digital",
            buy_cheap,
        ),
        _run("combined_rv", "Both: sell rich + buy cheap vs options", combined),
        _run(
            "fade_tails", "Sell away-from-the-money Kalshi contracts (fade the wide tails)", tails
        ),
        _run("naive_buy_all", "Benchmark: buy every contract (should lose the spread)", naive),
    ]
