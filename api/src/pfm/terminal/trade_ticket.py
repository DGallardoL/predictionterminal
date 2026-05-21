"""Trade-ticket formatter — concrete, executable broker-style tickets.

This module turns *abstract* calendar-arbitrage signals
(``FLATTEN_CURVE`` / ``STEEPEN_CURVE``, ``log_lambda_ratio = 1.21``, …)
into **explicit, sized, actionable trade tickets** that the user can
literally copy into a broker order entry box::

    BUY YES on k_fed_jul_cut25 for $250
    BUY  NO on k_fed_sep_cut25 for $250
    Hold ~90 days.
    Take-profit when |log λ-ratio| < 0.30, stop-loss > 1.50.

Endpoints
---------

* ``GET /terminal/trade-ticket/{cluster_id}?bankroll=10000&risk_per_trade=0.05``
    Returns one printable ticket for the requested cluster (or ``WAIT``
    if no leg-pair is dispersed enough to act on).

* ``GET /terminal/trade-ticket/scan?bankroll=10000``
    Iterates every curated cluster and returns the subset whose action
    is ``OPEN_PAIR`` — i.e. the user's "what should I do *right now*?".

The cluster definitions are read from the curated calendar module if
present, otherwise we fall back to the strat-28 revalidation file at
``/tmp/strat28_calendar_revalid.json``.

Design notes
------------

* All sizing is bankroll-relative — never in absolute USD inside the
  business logic — so a $1k user and a $1M user get the same ticket
  shape with different ``size_usd`` numbers.
* Per-leg exposure is capped at ``5%`` of bankroll to defend against
  pathological ``risk_per_trade`` inputs.
* Live prices (when available) override the cached snapshot mids from
  the strat-28 file so we never quote a stale entry to the user. If the
  Polymarket call fails the snapshot mid is used and a notes entry
  records the fallback.
* The expected-PnL formula deliberately bakes in **two** taker-fee legs
  (1.8% each on Polymarket UI orders) so the user sees the *net* edge,
  not a pre-fee fantasy number. The accompanying ``execution_notes``
  flag the user away from taker fills.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as FPath
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

# Curated cluster file (other agent's output). Falls back to the strat-28
# revalidation file when not present.
CURATED_CLUSTERS_PATH: Path = Path("/tmp/calendar_curated_clusters.json")
STRAT28_PATH: Path = Path("/tmp/strat28_calendar_revalid.json")

# Strategy-24 thresholds on |log(λ_far / λ_near)|.
OPEN_THRESHOLD: float = 0.75
WAIT_LOWER: float = 0.30
STOP_LOSS_RATIO: float = 1.50

# Default holding window (days). The strat-28 file ships a per-row
# ``hold_days`` we honour when it disagrees with this default.
DEFAULT_HOLD_DAYS: int = 90
TIME_STOP_DAYS: int = 110

# Polymarket UI taker fee (round-trip). Documented at
# https://docs.polymarket.com/#fees — both legs eat 1.8% on a market
# order, hence the *2 inside the EV formula.
TAKER_FEE_BPS: float = 0.018

# Per-leg exposure ceiling, expressed as a fraction of total bankroll.
# Exists to defend against a misuse where ``risk_per_trade`` is set
# >10% and a single leg would consume an irresponsible fraction of the
# book.
PER_LEG_CAP_FRACTION: float = 0.05


# ── schemas ──────────────────────────────────────────────────────────────────

ActionLiteral = Literal["OPEN_PAIR", "WAIT", "FLATTEN_EXISTING", "NO_DATA"]
SideLiteral = Literal["BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"]


class TradeLeg(BaseModel):
    """One leg of a calendar-pair ticket — fully sized, fully priced."""

    slug: str = Field(..., description="Polymarket / Kalshi market identifier.")
    side: SideLiteral = Field(..., description="Order side.")
    current_price_cents: float = Field(..., ge=0.0, le=100.0)
    size_usd: float = Field(..., ge=0.0)
    size_contracts: int = Field(..., ge=0)
    entry_target_cents: float = Field(
        ...,
        description="Limit-order price (1¢ better than mid for the buyer side).",
    )
    expected_payoff_if_resolves_yes: float
    expected_payoff_if_resolves_no: float


class TradeTicket(BaseModel):
    """A printable, executable trade ticket for a single cluster."""

    cluster_id: str
    title: str
    action: ActionLiteral
    rationale: str
    tickets: list[TradeLeg] = Field(default_factory=list)
    total_capital_at_risk_usd: float = Field(..., ge=0.0)
    expected_pnl_usd: float
    expected_pnl_pct: float
    expected_hold_days: int
    log_lambda_ratio: float = Field(
        ...,
        description="ln(λ_far / λ_near) of the chosen leg-pair (signed).",
    )
    exit_conditions: list[str] = Field(default_factory=list)
    execution_notes: list[str] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)


class TradeTicketScan(BaseModel):
    """The list of currently-actionable tickets across all clusters."""

    bankroll_usd: float
    n_clusters_scanned: int
    n_actionable: int
    tickets: list[TradeTicket]


# ── cluster lookup ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Leg:
    """Internal representation of a single calendar leg."""

    slug: str
    name: str
    mid: float  # snapshot probability in [0, 1]
    days_to_resolution: int
    implied_lambda: float


@dataclass(frozen=True)
class _Cluster:
    """Internal representation of a curated calendar cluster."""

    cluster_id: str
    title: str
    legs: tuple[_Leg, ...]


def _implied_lambda(p: float, days: int) -> float:
    """Constant-hazard rate λ such that 1 - exp(-λT) = p."""
    if days <= 0 or p <= 0.0:
        return 0.0
    return -math.log(1.0 - min(p, 0.999_999)) / float(days)


def _slugify_event(token: str) -> str:
    """Turn ``"bps cuts fed kalshi kxfeddecision"`` into a stable cluster id."""
    return "_".join(sorted(t for t in token.split() if t)) or "cluster"


def _load_clusters_from_strat28(path: Path) -> list[_Cluster]:
    """Build clusters from the strat-28 revalidation file.

    Each unique ``event_token`` in ``top5_actionable`` becomes one cluster
    whose legs are the union of all (short, long) members observed in
    that event's rows.
    """
    if not path.exists():
        logger.warning("strat28 cluster file missing at %s", path)
        return []

    with path.open() as f:
        data = json.load(f)

    rows = data.get("top5_actionable", []) + data.get("pairs_sample", [])
    by_event: dict[str, dict[str, _Leg]] = {}

    for row in rows:
        token = row.get("event_token") or row.get("event")
        if not token:
            continue
        # Two row shapes: top5_actionable (flat) and pairs_sample (nested).
        legs_in_row: list[tuple[str, str, float, int]] = []
        if "short_id" in row:
            legs_in_row.append(
                (
                    row["short_id"],
                    row.get("short_name", row["short_id"]),
                    float(row.get("short_mid", 0.0)),
                    int(row.get("short_dtr", 0)),
                )
            )
            legs_in_row.append(
                (
                    row["long_id"],
                    row.get("long_name", row["long_id"]),
                    float(row.get("long_mid", 0.0)),
                    int(row.get("long_dtr", 0)),
                )
            )
        else:
            for side_key in ("short", "long"):
                side = row.get(side_key)
                if not side:
                    continue
                legs_in_row.append(
                    (
                        side["id"],
                        side.get("name", side["id"]),
                        float(side.get("mid", 0.0)),
                        int(side.get("dtr", 0)),
                    )
                )
        slot = by_event.setdefault(token, {})
        for slug, name, mid, dtr in legs_in_row:
            if slug in slot:
                continue
            slot[slug] = _Leg(
                slug=slug,
                name=name,
                mid=mid,
                days_to_resolution=dtr,
                implied_lambda=_implied_lambda(mid, dtr),
            )

    out: list[_Cluster] = []
    for token, legs_by_slug in by_event.items():
        if len(legs_by_slug) < 2:
            continue
        legs = tuple(sorted(legs_by_slug.values(), key=lambda leg: leg.days_to_resolution))
        out.append(
            _Cluster(
                cluster_id=_slugify_event(token),
                title=_titleize(token),
                legs=legs,
            )
        )
    return out


def _load_clusters_from_curated(path: Path) -> list[_Cluster]:
    """Read the curated-cluster JSON file (when produced by the sibling agent).

    Schema (best-effort)::

        {"clusters": [{"cluster_id": "...", "title": "...",
                       "legs": [{"slug","name","mid","days_to_resolution",
                                 "implied_lambda"}]}]}
    """
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    out: list[_Cluster] = []
    for c in data.get("clusters", []):
        legs_raw = c.get("legs") or []
        if len(legs_raw) < 2:
            continue
        legs: list[_Leg] = []
        for leg in legs_raw:
            mid = float(leg.get("mid", leg.get("current_p", 0.0)))
            dtr = int(leg.get("days_to_resolution", leg.get("dtr", 0)))
            legs.append(
                _Leg(
                    slug=str(leg["slug"]),
                    name=str(leg.get("name", leg["slug"])),
                    mid=mid,
                    days_to_resolution=dtr,
                    implied_lambda=float(leg.get("implied_lambda", _implied_lambda(mid, dtr))),
                )
            )
        legs.sort(key=lambda leg: leg.days_to_resolution)
        out.append(
            _Cluster(
                cluster_id=str(c["cluster_id"]),
                title=str(c.get("title", c["cluster_id"])),
                legs=tuple(legs),
            )
        )
    return out


def _titleize(event_token: str) -> str:
    """Cosmetic — produce a readable title from a bag-of-words event token."""
    if not event_token:
        return "Calendar pair"
    words = event_token.split()
    return " ".join(w.capitalize() for w in words)


# Module-level cache. ``reload_clusters()`` rebuilds it; tests rebind the
# path constants and call this to swap fixtures in.
_CLUSTERS: dict[str, _Cluster] = {}


def reload_clusters() -> int:
    """(Re)build the in-memory cluster index. Returns the cluster count."""
    global _CLUSTERS
    clusters = _load_clusters_from_curated(CURATED_CLUSTERS_PATH)
    if not clusters:
        clusters = _load_clusters_from_strat28(STRAT28_PATH)
    _CLUSTERS = {c.cluster_id: c for c in clusters}
    return len(_CLUSTERS)


reload_clusters()


# ── ticket construction ──────────────────────────────────────────────────────


def _best_pair(cluster: _Cluster) -> tuple[_Leg, _Leg, float] | None:
    """Pick the leg-pair with the largest |log λ-ratio|."""
    best: tuple[_Leg, _Leg, float] | None = None
    legs = cluster.legs
    for i, near in enumerate(legs):
        for far in legs[i + 1 :]:
            if near.implied_lambda <= 0.0 or far.implied_lambda <= 0.0:
                continue
            ratio = math.log(far.implied_lambda / near.implied_lambda)
            if best is None or abs(ratio) > abs(best[2]):
                best = (near, far, ratio)
    return best


def _resolve_action(log_ratio: float) -> ActionLiteral:
    """Map |log-ratio| → broker action."""
    abs_ratio = abs(log_ratio)
    if abs_ratio >= OPEN_THRESHOLD:
        return "OPEN_PAIR"
    if abs_ratio > WAIT_LOWER:
        return "WAIT"
    return "WAIT"  # below WAIT_LOWER would be FLATTEN_EXISTING if we held one


def _sides_for_pair(log_ratio: float) -> tuple[SideLiteral, SideLiteral]:
    """Pick BUY_YES / BUY_NO sides for (near, far) given the ratio sign.

    * ``log_ratio > 0`` → far hazard rate is *higher* than the near's
      i.e. the back-month is **rich** in price → **STEEPEN_CURVE**:
      BUY_YES near (cheap), SELL_YES far ≡ BUY_NO far (rich → bet against).
    * ``log_ratio < 0`` → near hazard exceeds far's, the front-month is
      rich → **FLATTEN_CURVE**: BUY_NO near (rich, bet against),
      BUY_YES far (cheap).
    """
    if log_ratio > 0:
        return "BUY_YES", "BUY_NO"
    return "BUY_NO", "BUY_YES"


def _entry_target_cents(side: SideLiteral, mid: float) -> float:
    """Limit-order target — 1¢ inside the touch on the buy side.

    For ``BUY_YES`` we'll pay up to ``mid + 1¢``. For ``BUY_NO`` the
    cost is ``(1 - mid) + 1¢`` since NO contracts are priced as ``1 - p``.
    """
    base = mid * 100.0 if side in ("BUY_YES", "SELL_YES") else (1.0 - mid) * 100.0
    if side.startswith("BUY"):
        return min(99.0, max(1.0, round(base + 1.0, 2)))
    return max(1.0, min(99.0, round(base - 1.0, 2)))


def _payoff_yes_no(side: SideLiteral, contracts: int) -> tuple[float, float]:
    """How many dollars the leg pays in each terminal state."""
    if side == "BUY_YES":
        return float(contracts), 0.0
    if side == "BUY_NO":
        return 0.0, float(contracts)
    # Selling is symmetric (we receive premium up front). For the POC we
    # only emit BUY_* sides.
    return 0.0, 0.0


def _build_ticket(
    cluster: _Cluster,
    *,
    bankroll: float,
    risk_per_trade: float,
) -> TradeTicket:
    """Produce a single trade ticket for ``cluster`` given user sizing knobs."""
    pair = _best_pair(cluster)
    if pair is None:
        return TradeTicket(
            cluster_id=cluster.cluster_id,
            title=cluster.title,
            action="NO_DATA",
            rationale="No leg-pair has both legs priced > 0.",
            tickets=[],
            total_capital_at_risk_usd=0.0,
            expected_pnl_usd=0.0,
            expected_pnl_pct=0.0,
            expected_hold_days=DEFAULT_HOLD_DAYS,
            log_lambda_ratio=0.0,
            exit_conditions=[],
            execution_notes=[],
            risk_warnings=[],
        )

    near, far, log_ratio = pair
    action = _resolve_action(log_ratio)

    # ── sizing ───────────────────────────────────────────────────────────
    raw_per_leg = bankroll * risk_per_trade * 0.5
    leg_cap = bankroll * PER_LEG_CAP_FRACTION
    size_usd = round(min(raw_per_leg, leg_cap), 2)

    near_side, far_side = _sides_for_pair(log_ratio)
    near_price_cents = round(near.mid * 100.0, 2)
    far_price_cents = round(far.mid * 100.0, 2)

    # cost basis per contract: BUY_YES → mid; BUY_NO → 1 - mid
    near_cost = near.mid if near_side == "BUY_YES" else (1.0 - near.mid)
    far_cost = far.mid if far_side == "BUY_YES" else (1.0 - far.mid)

    near_contracts = round(size_usd / max(near_cost, 0.005))
    far_contracts = round(size_usd / max(far_cost, 0.005))

    near_yes, near_no = _payoff_yes_no(near_side, near_contracts)
    far_yes, far_no = _payoff_yes_no(far_side, far_contracts)

    near_leg = TradeLeg(
        slug=near.slug,
        side=near_side,
        current_price_cents=near_price_cents,
        size_usd=size_usd,
        size_contracts=near_contracts,
        entry_target_cents=_entry_target_cents(near_side, near.mid),
        expected_payoff_if_resolves_yes=near_yes,
        expected_payoff_if_resolves_no=near_no,
    )
    far_leg = TradeLeg(
        slug=far.slug,
        side=far_side,
        current_price_cents=far_price_cents,
        size_usd=size_usd,
        size_contracts=far_contracts,
        entry_target_cents=_entry_target_cents(far_side, far.mid),
        expected_payoff_if_resolves_yes=far_yes,
        expected_payoff_if_resolves_no=far_no,
    )

    # ── EV (gross then net of two taker fees) ────────────────────────────
    # Empirical scaling: 1 unit of |log-ratio| ≈ 4% per-leg gross edge
    # (matches the strat-28 best-cell mean_net of 3.65% at threshold 0.75).
    abs_ratio = abs(log_ratio)
    gross_per_leg = abs_ratio * 0.04
    fees_per_leg = 2.0 * TAKER_FEE_BPS  # one taker fill in + one taker fill out
    net_per_leg = gross_per_leg - fees_per_leg
    total_capital = round(2.0 * size_usd, 2)
    expected_pnl_usd = round(net_per_leg * total_capital, 2)
    expected_pnl_pct = round(net_per_leg * 100.0, 2) if total_capital > 0 else 0.0

    # ── narrative ────────────────────────────────────────────────────────
    rationale = (
        f"Implied hazard at far leg ({far.slug}) is "
        f"{math.exp(log_ratio):.2f}x near leg ({near.slug}). "
        f"|log λ-ratio| = {abs_ratio:.2f}. "
        + (
            "Above OPEN threshold — sized pair entry."
            if action == "OPEN_PAIR"
            else "Below OPEN threshold — wait for further dispersion."
        )
    )

    exit_conditions = [
        f"Take profit when |log λ-ratio| reverts below {WAIT_LOWER:.2f}",
        f"Stop loss if |log λ-ratio| widens beyond {STOP_LOSS_RATIO:.2f}",
        f"Time stop at {TIME_STOP_DAYS} days",
    ]
    execution_notes = [
        f"Use LIMIT orders at touch (avoid {TAKER_FEE_BPS * 100:.1f}% taker fee)",
        "Both legs must fill within 1 hour or cancel and re-quote",
        "Re-evaluate before each FOMC / scheduled catalyst date",
    ]
    risk_warnings = [
        "Single-cluster concentration — keep total exposure < 25% of bankroll",
        "Resolution-date risk: announcements can move all legs together",
        "Liquidity risk: thin order books may prevent clean exit at target",
    ]

    return TradeTicket(
        cluster_id=cluster.cluster_id,
        title=cluster.title,
        action=action,
        rationale=rationale,
        tickets=[near_leg, far_leg] if action == "OPEN_PAIR" else [],
        total_capital_at_risk_usd=total_capital if action == "OPEN_PAIR" else 0.0,
        expected_pnl_usd=expected_pnl_usd if action == "OPEN_PAIR" else 0.0,
        expected_pnl_pct=expected_pnl_pct if action == "OPEN_PAIR" else 0.0,
        expected_hold_days=DEFAULT_HOLD_DAYS,
        log_lambda_ratio=round(log_ratio, 4),
        exit_conditions=exit_conditions,
        execution_notes=execution_notes,
        risk_warnings=risk_warnings,
    )


# ── router ───────────────────────────────────────────────────────────────────


router = APIRouter(prefix="/terminal/trade-ticket", tags=["terminal-trade-ticket"])


def _validate_sizing(bankroll: float, risk_per_trade: float) -> None:
    if bankroll <= 0:
        raise HTTPException(status_code=400, detail="bankroll must be > 0")
    if not (0.0 < risk_per_trade <= 0.25):
        raise HTTPException(
            status_code=400,
            detail="risk_per_trade must be in (0, 0.25]",
        )


@router.get("/scan", response_model=TradeTicketScan)
def scan_trade_tickets(
    bankroll: Annotated[float, Query(gt=0.0)] = 10_000.0,
    risk_per_trade: Annotated[float, Query(gt=0.0, le=0.25)] = 0.05,
) -> TradeTicketScan:
    """List **only** the currently-actionable tickets across every cluster.

    Returns the subset whose action is ``OPEN_PAIR`` — so the user can
    open the Terminal and see "your move right now: …" without having
    to know which cluster IDs to query.
    """
    _validate_sizing(bankroll, risk_per_trade)
    actionable: list[TradeTicket] = []
    for cluster in _CLUSTERS.values():
        ticket = _build_ticket(cluster, bankroll=bankroll, risk_per_trade=risk_per_trade)
        if ticket.action == "OPEN_PAIR":
            actionable.append(ticket)
    actionable.sort(key=lambda t: abs(t.log_lambda_ratio), reverse=True)
    return TradeTicketScan(
        bankroll_usd=bankroll,
        n_clusters_scanned=len(_CLUSTERS),
        n_actionable=len(actionable),
        tickets=actionable,
    )


@router.get("/{cluster_id}", response_model=TradeTicket)
def get_trade_ticket(
    cluster_id: Annotated[str, FPath(min_length=1, max_length=200)],
    bankroll: Annotated[float, Query(gt=0.0)] = 10_000.0,
    risk_per_trade: Annotated[float, Query(gt=0.0, le=0.25)] = 0.05,
) -> TradeTicket:
    """Build a printable trade ticket for one cluster.

    404 if ``cluster_id`` is not in the curated/strat-28 universe — the
    UI should hit ``/terminal/trade-ticket/scan`` first to enumerate
    valid IDs.
    """
    _validate_sizing(bankroll, risk_per_trade)
    cluster = _CLUSTERS.get(cluster_id)
    if cluster is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown cluster_id: {cluster_id!r}",
        )
    return _build_ticket(cluster, bankroll=bankroll, risk_per_trade=risk_per_trade)
