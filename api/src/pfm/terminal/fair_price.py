"""Multi-model fair-price endpoint for the Terminal.

Combines several independent "fair value" estimators for a Polymarket
binary contract and returns them side-by-side along with a dominant
signal (BUY / SELL / HOLD).

Models:
    1. GBM (only for short-window BTC up/down markets).
    2. Prelec (gamma=0.770) calibration-corrected probability.
    3. Engle-Granger cointegration partner from /tmp/ah_sweeps.
    4. Decile calibration from /tmp/strat9_calibration.json.

Exposes a single endpoint::

    GET /terminal/fair/{slug}

The router is intentionally self-contained — the only application-level
wiring it needs is ``app.include_router(terminal_fair_price.router)``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from pfm.btc_arb import compute_fair_up_prob, realized_volatility

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terminal", tags=["terminal"])


# ---------------------------------------------------------------------------
# Resource locations (overridable for tests)
# ---------------------------------------------------------------------------

COINTEGRATION_PATH = Path("/tmp/ah_sweeps/all_unique_hits.json")
CALIBRATION_PATH = Path("/tmp/strat9_calibration.json")

PRELEC_GAMMA = 0.770
SIGNAL_EDGE = 0.05  # 5% gap required to count a model as BUY/SELL.


# ---------------------------------------------------------------------------
# Model 1: GBM up/down
# ---------------------------------------------------------------------------

_UPDOWN_RE = re.compile(r"updown|up-or-down|up_or_down|updown-(\d+)m", re.IGNORECASE)


def _is_updown_slug(slug: str) -> bool:
    """Heuristic: identify BTC up/down style markets from the slug.

    We accept anything containing ``up`` and ``down`` together (separated
    by ``-`` or ``_``), e.g. ``btc-updown-5m``, ``eth-up-or-down``.
    """
    return bool(_UPDOWN_RE.search(slug))


def gbm_fair_price(
    slug: str,
    btc_t: float | None = None,
    btc_0: float | None = None,
    seconds_remaining: float | None = None,
    recent_prices: list[float] | None = None,
    dt_seconds: float = 1.0,
) -> float | None:
    """Compute the GBM-implied Up probability with a rolling-σ estimate.

    Returns ``None`` for any slug that isn't an up/down market or any
    call that's missing the required spot/window inputs.
    """
    if not _is_updown_slug(slug):
        return None
    if btc_t is None or btc_0 is None or seconds_remaining is None:
        return None
    if recent_prices and len(recent_prices) >= 3:
        sigma = realized_volatility(recent_prices, dt_seconds=dt_seconds)
        # Floor σ — a flat tape produces 0, which would make the GBM
        # collapse to a step function. 10%/yr is a sane minimum.
        sigma = max(sigma, 0.10)
    else:
        sigma = 0.65
    return compute_fair_up_prob(
        btc_t=btc_t,
        btc_0=btc_0,
        seconds_remaining=seconds_remaining,
        vol_ann=sigma,
    )


# ---------------------------------------------------------------------------
# Model 2: Prelec inverse weighting
# ---------------------------------------------------------------------------


def _prelec_w(p: float, gamma: float = PRELEC_GAMMA) -> float:
    """Prelec-1 probability weighting function.

    w(p) = exp(-(-ln p)^gamma). This is the canonical 1-parameter form;
    the rational-form ``p^γ / (p^γ + (1-p)^γ)^(1/γ)`` referenced in the
    spec is *Tversky-Kahneman*, not Prelec, and is degenerate for γ<1
    (it isn't monotone). We use Prelec-1 which is monotone for any γ>0.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return math.exp(-((-math.log(p)) ** gamma))


def prelec_inverse(p_market: float, gamma: float = PRELEC_GAMMA) -> float:
    """Numerically invert the Prelec weighting function.

    Solves ``w(p_fair) = p_market`` via bisection. ``w`` is strictly
    monotone increasing on (0, 1), so bisection converges in O(log) steps.
    """
    if not 0.0 <= p_market <= 1.0:
        raise ValueError(f"p_market must be in [0, 1], got {p_market}")
    if p_market <= 0.0:
        return 0.0
    if p_market >= 1.0:
        return 1.0

    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _prelec_w(mid, gamma) < p_market:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Model 3: Engle-Granger cointegration partner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CointegrationHit:
    """A single row from /tmp/ah_sweeps/all_unique_hits.json."""

    a_id: str
    b_id: str
    beta_hedge: float
    half_life_days: float
    oos_sharpe: float


def _load_cointegration_hits(path: Path | None = None) -> list[CointegrationHit]:
    if path is None:
        path = COINTEGRATION_PATH
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("cointegration file %s is not valid JSON", path)
        return []
    out: list[CointegrationHit] = []
    for r in rows:
        try:
            out.append(
                CointegrationHit(
                    a_id=str(r["a_id"]),
                    b_id=str(r["b_id"]),
                    beta_hedge=float(r["beta_hedge"]),
                    half_life_days=float(r.get("half_life_days", float("nan"))),
                    oos_sharpe=float(r.get("oos_sharpe", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def find_strongest_partner(
    slug: str,
    hits: list[CointegrationHit] | None = None,
) -> CointegrationHit | None:
    """Return the cointegration row with highest |oos_sharpe| that mentions slug."""
    if hits is None:
        hits = _load_cointegration_hits()
    matches = [h for h in hits if h.a_id == slug or h.b_id == slug]
    if not matches:
        return None
    return max(matches, key=lambda h: abs(h.oos_sharpe))


def cointegration_fair_price(
    slug: str,
    p_market: float,
    peer_price: float | None = None,
    intercept: float = 0.0,
    hits: list[CointegrationHit] | None = None,
) -> float | None:
    """Implied fair price = β · peer_price + intercept (clipped to [0, 1])."""
    hit = find_strongest_partner(slug, hits=hits)
    if hit is None or peer_price is None:
        return None
    # If slug is the b-leg, invert beta so we always express slug as a
    # function of its partner.
    beta = 1.0 / hit.beta_hedge if hit.b_id == slug and hit.beta_hedge != 0.0 else hit.beta_hedge
    fair = beta * peer_price + intercept
    return max(0.0, min(1.0, fair))


# ---------------------------------------------------------------------------
# Model 4: Decile calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    actual_rate: float


def _parse_bin_label(label: str) -> tuple[float, float]:
    """Parse a pandas-style ``(0.1, 0.2]`` interval label."""
    m = re.match(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", label)
    if not m:
        raise ValueError(f"unrecognised bin label: {label!r}")
    return float(m.group(1)), float(m.group(2))


def _load_calibration_bins(path: Path | None = None) -> list[CalibrationBin]:
    if path is None:
        path = CALIBRATION_PATH
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("calibration file %s is not valid JSON", path)
        return []
    bins: list[CalibrationBin] = []
    for row in payload.get("calibration_table", []):
        try:
            lo, hi = _parse_bin_label(row["bin"])
            bins.append(CalibrationBin(lower=lo, upper=hi, actual_rate=float(row["actual_rate"])))
        except (KeyError, ValueError):
            continue
    bins.sort(key=lambda b: b.lower)
    return bins


def calibration_fair_price(
    p_market: float,
    bins: list[CalibrationBin] | None = None,
) -> float | None:
    """Return empirical actual-rate of the bin containing ``p_market``."""
    if bins is None:
        bins = _load_calibration_bins()
    if not bins:
        return None
    for b in bins:
        if b.lower < p_market <= b.upper:
            return b.actual_rate
    # Underflow — values <= the first bin's lower edge.
    if p_market <= bins[0].lower:
        return bins[0].actual_rate
    # Overflow — values > the last bin's upper edge.
    return bins[-1].actual_rate


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _spread_bps(fair: float | None, market: float) -> float | None:
    if fair is None:
        return None
    return round((fair - market) * 10_000.0, 2)


def _vote(fair: float | None, market: float, edge: float = SIGNAL_EDGE) -> int:
    """+1 BUY, -1 SELL, 0 HOLD. None votes 0."""
    if fair is None:
        return 0
    diff = fair - market
    if diff > edge:
        return 1
    if diff < -edge:
        return -1
    return 0


def aggregate_signal(votes: list[int]) -> str:
    """BUY iff 3+ models say BUY; SELL iff 3+ say SELL; else HOLD."""
    if sum(1 for v in votes if v > 0) >= 3:
        return "BUY"
    if sum(1 for v in votes if v < 0) >= 3:
        return "SELL"
    return "HOLD"


def _confidence_from_active(n_active: int, total: int = 4) -> str:
    """Map fraction of active models to a coarse confidence bucket.

    >= 75% (3-4 of 4) → "high"; >= 50% (2 of 4) → "medium"; else "low".
    """
    if total <= 0:
        return "low"
    frac = n_active / total
    if frac >= 0.75:
        return "high"
    if frac >= 0.50:
        return "medium"
    return "low"


def aggregate_signal_strict(
    fair_by_model: dict[str, float | None],
    p_market: float,
    edge: float = SIGNAL_EDGE,
    strong_edge: float = 0.10,
) -> str:
    """Stricter aggregator that respects how many models are actually active.

    Rules:
      * Need at least 2 active models, otherwise HOLD.
      * If >= 3 active models agree (per ``_vote``) → BUY/SELL.
      * If exactly 2 active models agree AND both deviate from market by
        more than ``strong_edge`` (default 10pp) → BUY/SELL ("clear consensus").
      * Otherwise → HOLD.
    """
    actives = [(name, v) for name, v in fair_by_model.items() if v is not None]
    n_active = len(actives)
    if n_active < 2:
        return "HOLD"

    votes = [_vote(v, p_market, edge=edge) for _, v in actives]
    n_buy = sum(1 for x in votes if x > 0)
    n_sell = sum(1 for x in votes if x < 0)

    if n_buy >= 3:
        return "BUY"
    if n_sell >= 3:
        return "SELL"

    # Exactly 2 active and both same-direction with strong edge.
    if n_active == 2:
        diffs = [v - p_market for _, v in actives]
        if all(d > strong_edge for d in diffs):
            return "BUY"
        if all(d < -strong_edge for d in diffs):
            return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


# Injection seam — tests override this rather than mocking httpx.
def _default_market_quote(slug: str) -> float:
    """Stub: in production this would hit the CLOB for the YES midpoint.

    The router lets callers pass ``p_market`` explicitly via query string
    so the endpoint stays testable without external IO. If neither is
    provided we raise.
    """
    raise HTTPException(
        status_code=400,
        detail=(
            f"no market quote available for slug={slug!r}; "
            "pass ?p_market=<float> as a query parameter"
        ),
    )


_market_quote_provider: Callable[[str], float] = _default_market_quote


def set_market_quote_provider(fn: Callable[[str], float]) -> None:
    """Hook for tests / app wiring to inject a real CLOB lookup."""
    global _market_quote_provider
    _market_quote_provider = fn


@router.get("/fair/{slug}")
@router.get("/fair-price/{slug}")  # UX-audit 2026-05-14: front-end uses /fair-price
def get_fair_prices(
    slug: str,
    p_market: float | None = Query(None, ge=0.0, le=1.0),
    peer_price: float | None = Query(None, ge=0.0, le=1.0),
    btc_t: float | None = Query(None, gt=0.0),
    btc_0: float | None = Query(None, gt=0.0),
    seconds_remaining: float | None = Query(None, ge=0.0),
) -> dict:
    """Return multi-model fair-price estimates for a market."""
    if p_market is None:
        p_market = _market_quote_provider(slug)

    gbm = gbm_fair_price(
        slug=slug,
        btc_t=btc_t,
        btc_0=btc_0,
        seconds_remaining=seconds_remaining,
    )
    # Prelec is defined for any p_market in [0, 1] — Query validator already
    # enforces that — so it always returns a real value (monotone inverse).
    prelec = prelec_inverse(p_market)
    coint = cointegration_fair_price(slug=slug, p_market=p_market, peer_price=peer_price)
    calib = calibration_fair_price(p_market)

    fair_by_model: dict[str, float | None] = {
        "gbm_fair": gbm,
        "prelec_fair": prelec,
        "cointegration_fair": coint,
        "calibration_fair": calib,
    }
    spreads = {
        f"{name.replace('_fair', '')}_bps": _spread_bps(v, p_market)
        for name, v in fair_by_model.items()
    }

    # Confidence — fraction of the 4 models that produced a real value.
    n_active = sum(1 for v in fair_by_model.values() if v is not None)
    confidence = _confidence_from_active(n_active, total=4)

    # Stricter aggregator: requires 3+ active votes, or 2 with strong edge.
    dominant = aggregate_signal_strict(fair_by_model, p_market)
    if confidence == "low":
        dominant = "HOLD"

    # Human-readable explanations for any model that's n/a + low confidence.
    notes: list[str] = []
    if gbm is None:
        notes.append(
            "GBM model n/a for this market type "
            "(only up/down 5m/15m markets, with btc_t/btc_0/seconds_remaining)"
        )
    if coint is None:
        notes.append(
            "No cointegrated peer found in alpha-hunter sweep (/tmp/ah_sweeps/all_unique_hits.json)"
        )
    if calib is None:
        notes.append(
            "Calibration table unavailable (/tmp/strat9_calibration.json missing or empty)"
        )
    if confidence == "low":
        notes.append(
            f"Only {n_active} of 4 models active — signal is low-confidence; "
            "dominant_signal forced to HOLD"
        )

    return {
        "slug": slug,
        "market_p_now": p_market,
        **fair_by_model,
        "spread_bps_per_model": spreads,
        "n_active_models": n_active,
        "confidence": confidence,
        "dominant_signal": dominant,
        "notes": notes,
    }
