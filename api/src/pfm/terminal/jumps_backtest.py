"""Paper-PnL backtester for the "disagrees jump" signal.

Hypothesis
----------
A *disagreeing* jump is a moment where the **news direction** and the
**price direction** point opposite ways:

* news bullish but the YES contract dropped, or
* news bearish but YES rallied.

The thesis: the **news is right**, the price is mispriced, and the
contract will revert toward the news direction over the next 1h / 6h /
24h. Every "agrees" jump (news + price aligned) is a momentum trade and
is included only as a **control bucket** — if the signal is real we
expect disagrees > 0 and agrees ≈ 0 (or weakly negative, since you're
fading the move that already happened).

Implementation notes
--------------------
* Entry price is ``price_after`` (the post-jump tick — what a trader
  could realistically have hit immediately after seeing the jump).
* Exit price is sampled at ``jump_ts + hold_hours`` using the *first*
  observation at-or-after that target. Jumps too close to the end of
  the series (no exit observation) are silently skipped — they're not
  errors, just not yet evaluable.
* Position side is decided by the **news sentiment score** (sign):
    - news positive → long YES   → return = (exit - entry) / entry
    - news negative → short YES  → return = (entry - exit) / entry
  When the sentiment score is exactly 0 we fall back to the
  ``sentiment_alignment`` field plus the jump ``direction`` — a disagrees
  + jump-up implies the news must be negative (price went up against
  bearish news) and we short YES.
* The Sharpe is **naive**: ``mean / std × √(252 × 24 / hold_hours)``.
  No risk-free subtraction, no autocorrelation correction. This is a
  POC paper-PnL — the calling layer treats it as a sanity check, not a
  publishable backtest.
* The equity curve is the cumulative *sum* of disagrees per-trade
  returns ordered by ``jump_ts``. Sum (not product) keeps each trade's
  contribution comparable when ``hold_hours`` is small; the magnitudes
  here are per-trade fractional returns, not compoundable strategy NAV.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Annotated, Any

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal.jumps import (
    DEFAULT_MAD_K,
    DEFAULT_MIN_JUMP_PP,
    ROLLING_HOURS,
    _articles_for_jump,
    _direction,
    _gather_all_news,
    detect_jumps,
)
from pfm.terminal.news_impact import _fetch_hourly_prices
from pfm.terminal.news_relevance import (
    RELEVANCE_MIN,
    build_phrase_query,
    build_terms,
    score_relevance,
)
from pfm.terminal.sentiment_nlp import aggregate_sentiment
from pfm.terminal_gdelt_news import GDELTArticle, _build_query
from pfm.terminal_news import MAX_KEYWORDS, extract_keywords

logger = logging.getLogger(__name__)


MIN_HOLD_HOURS: int = 1
MAX_HOLD_HOURS: int = 48
DEFAULT_HOLD_HOURS: int = 6
MIN_DAYS: int = 1
MAX_DAYS: int = 90
DEFAULT_DAYS: int = 14

# Annualisation factor for naive Sharpe: 252 trading days × 24 hours per day,
# divided by the holding-period length in hours. We treat each trade as one
# observation of a "hold_hours"-length return.
TRADING_HOURS_PER_YEAR: float = 252.0 * 24.0


# --- schemas ----------------------------------------------------------------


class BacktestStats(BaseModel):
    """Aggregate paper-PnL statistics for one bucket (disagrees or agrees)."""

    n_trades: int = Field(..., ge=0, description="Number of trades evaluated.")
    mean_return: float = Field(..., description="Mean per-trade fractional return (entry-to-exit).")
    std_return: float = Field(..., ge=0.0, description="Population std-dev of per-trade returns.")
    sharpe_naive: float = Field(
        ...,
        description=("Naive Sharpe: mean / std × √(252×24 / hold_hours). Zero when std=0 or n<2."),
    )
    hit_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of trades with return > 0.")
    avg_win: float = Field(
        ..., ge=0.0, description="Mean return across winning trades (0 if none)."
    )
    avg_loss: float = Field(
        ..., le=0.0, description="Mean return across losing trades (0 if none)."
    )
    max_drawdown: float = Field(
        ...,
        le=0.0,
        description=(
            "Worst peak-to-trough drawdown on the cumulative-sum equity curve, "
            "expressed as a non-positive number."
        ),
    )
    total_return: float = Field(
        ...,
        description="Sum of per-trade returns (terminal value of equity curve).",
    )


class EquityPoint(BaseModel):
    """One step of the disagrees equity curve, in trade order."""

    ts: str = Field(..., description="ISO-8601 UTC timestamp of the trade entry.")
    cum_return: float = Field(
        ..., description="Cumulative sum of per-trade returns up to and including ts."
    )


class JumpsBacktestResponse(BaseModel):
    slug: str
    hold_hours: int
    n_disagrees: int = Field(..., ge=0)
    n_agrees: int = Field(..., ge=0)
    disagrees_pnl: BacktestStats
    agrees_pnl: BacktestStats
    equity_curve: list[EquityPoint]
    interpretation: str


# --- pure-function core -----------------------------------------------------


def _zero_stats() -> BacktestStats:
    return BacktestStats(
        n_trades=0,
        mean_return=0.0,
        std_return=0.0,
        sharpe_naive=0.0,
        hit_rate=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        max_drawdown=0.0,
        total_return=0.0,
    )


def _exit_price_at(prices: pd.Series, target: pd.Timestamp) -> float | None:
    """First observed price at-or-after ``target``; ``None`` when past the
    end of the series (jump is too close to "now" to evaluate)."""
    if prices.empty:
        return None
    sub = prices[prices.index >= target]
    if sub.empty:
        return None
    v = float(sub.iloc[0])
    return v if np.isfinite(v) else None


def _news_side(jump: dict[str, Any]) -> int:
    """Return ``+1`` for long-YES, ``-1`` for short-YES, ``0`` if no signal.

    Decision priority:
        1. Explicit ``news_sentiment_score`` (signed float in [-1, +1]).
        2. ``sentiment_alignment`` + ``direction`` (recover sign for
           jumps where the score was rounded to 0 but the alignment is
           still meaningful).
    """
    score = jump.get("news_sentiment_score")
    if isinstance(score, (int, float)) and abs(float(score)) > 1e-9:
        return 1 if float(score) > 0 else -1
    align = jump.get("sentiment_alignment", "neutral")
    direction = jump.get("direction", "flat")
    if align == "disagrees":
        # Price moved against the news → news is opposite of price → short
        # the YES move (or long it if the price fell against bullish news).
        if direction == "up":
            return -1
        if direction == "down":
            return 1
    if align == "agrees":
        if direction == "up":
            return 1
        if direction == "down":
            return -1
    return 0


def _trade_return(entry: float, exit_: float, side: int) -> float | None:
    """Per-trade fractional return; ``None`` if inputs are degenerate."""
    if side == 0:
        return None
    if entry is None or exit_ is None:
        return None
    if not (np.isfinite(entry) and np.isfinite(exit_)):
        return None
    if entry <= 0.0:
        return None
    raw = (exit_ - entry) / entry
    return raw if side > 0 else -raw


def _bucket_stats(returns: list[float], hold_hours: int) -> BacktestStats:
    if not returns:
        return _zero_stats()
    arr = np.asarray(returns, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean())
    # Population std (ddof=0) — keeps Sharpe well-defined at n=1 (it
    # collapses to 0 because std is 0, which we want).
    std = float(arr.std(ddof=0))
    if std > 1e-12 and n >= 2 and hold_hours > 0:
        ann = math.sqrt(TRADING_HOURS_PER_YEAR / float(hold_hours))
        sharpe = (mean / std) * ann
    else:
        sharpe = 0.0
    wins = arr[arr > 0.0]
    losses = arr[arr < 0.0]
    hit_rate = float(wins.size) / n
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    # Cumulative-sum equity curve & drawdown
    eq = np.cumsum(arr)
    running_peak = np.maximum.accumulate(eq)
    drawdowns = eq - running_peak
    max_dd = float(drawdowns.min()) if drawdowns.size else 0.0
    # Clamp tiny positive numerical noise — drawdown is non-positive by defn.
    if max_dd > 0:
        max_dd = 0.0
    return BacktestStats(
        n_trades=n,
        mean_return=mean,
        std_return=std,
        sharpe_naive=sharpe,
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=max_dd,
        total_return=float(eq[-1]),
    )


def simulate_disagrees_pnl(
    jumps: list[dict[str, Any]],
    prices: pd.Series,
    *,
    hold_hours: int = DEFAULT_HOLD_HOURS,
) -> dict[str, Any]:
    """Compute paper-PnL for the "disagrees jump" signal vs an agrees control.

    Args:
        jumps: list of Jump-shaped dicts. Each must carry ``ts_iso`` (or
            ``ts``), ``price_after``, ``sentiment_alignment``, and one of
            ``news_sentiment_score`` or ``direction`` for side resolution.
        prices: UTC-indexed price series (same series the jumps were
            detected from). Used to fetch the exit observation.
        hold_hours: holding period in hours. Exit = first observation
            at-or-after ``entry_ts + hold_hours``.

    Returns:
        ``{
            "hold_hours": int,
            "n_disagrees": int,
            "n_agrees": int,
            "disagrees_pnl": BacktestStats-dict,
            "agrees_pnl": BacktestStats-dict,
            "equity_curve": list[{ts, cum_return}],
        }``
    """
    if hold_hours <= 0:
        raise ValueError("hold_hours must be positive")
    if not jumps or prices is None or prices.empty:
        return {
            "hold_hours": int(hold_hours),
            "n_disagrees": 0,
            "n_agrees": 0,
            "disagrees_pnl": _zero_stats().model_dump(),
            "agrees_pnl": _zero_stats().model_dump(),
            "equity_curve": [],
        }

    series = prices.sort_index().astype(float)
    # Normalise the index to UTC tz-aware so timestamp arithmetic with
    # the (possibly naive) jump ts doesn't surprise.
    if series.index.tzinfo is None and hasattr(series.index, "tz_localize"):
        with contextlib.suppress(TypeError, AttributeError):
            series.index = series.index.tz_localize("UTC")

    disagrees_rets: list[tuple[pd.Timestamp, float]] = []
    agrees_rets: list[tuple[pd.Timestamp, float]] = []

    for j in jumps:
        align = j.get("sentiment_alignment", "neutral")
        if align not in ("disagrees", "agrees"):
            continue
        side = _news_side(j)
        if side == 0:
            continue
        # Pull entry timestamp & price
        entry = j.get("price_after")
        ts_raw = j.get("ts_iso") or j.get("ts")
        if entry is None or ts_raw is None:
            continue
        try:
            ts = pd.Timestamp(ts_raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
        except (ValueError, TypeError):
            continue
        target = ts + pd.Timedelta(hours=hold_hours)
        exit_ = _exit_price_at(series, target)
        if exit_ is None:
            # Jump too close to series end — silently skip; not an error.
            continue
        ret = _trade_return(float(entry), float(exit_), side)
        if ret is None:
            continue
        if align == "disagrees":
            disagrees_rets.append((ts, ret))
        else:
            agrees_rets.append((ts, ret))

    # Sort each bucket chronologically (for the equity curve & for the
    # max-drawdown computation, which depends on trade order).
    disagrees_rets.sort(key=lambda x: x[0])
    agrees_rets.sort(key=lambda x: x[0])

    disagrees_pnl = _bucket_stats([r for _, r in disagrees_rets], hold_hours)
    agrees_pnl = _bucket_stats([r for _, r in agrees_rets], hold_hours)

    # Equity curve for the disagrees bucket only (the signal under test).
    eq_curve: list[dict[str, Any]] = []
    cum = 0.0
    for ts, r in disagrees_rets:
        cum += r
        eq_curve.append(
            {
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "cum_return": float(cum),
            }
        )

    return {
        "hold_hours": int(hold_hours),
        "n_disagrees": len(disagrees_rets),
        "n_agrees": len(agrees_rets),
        "disagrees_pnl": disagrees_pnl.model_dump(),
        "agrees_pnl": agrees_pnl.model_dump(),
        "equity_curve": eq_curve,
    }


# --- routing ----------------------------------------------------------------


router = APIRouter(prefix="/terminal", tags=["terminal-jumps-backtest"])


def _get_polymarket_client(request: Request) -> PolymarketClient:
    poly: PolymarketClient | None = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


@router.get(
    "/jumps/{slug}/backtest",
    response_model=JumpsBacktestResponse,
    summary="Paper-PnL backtest of the disagrees-jump reversion signal.",
)
async def get_jumps_backtest(
    request: Request,
    slug: Annotated[str, Path(min_length=1, max_length=120)],
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
    hold_hours: Annotated[int, Query(ge=MIN_HOLD_HOURS, le=MAX_HOLD_HOURS)] = DEFAULT_HOLD_HOURS,
    mad_k: Annotated[float, Query(ge=1.0, le=10.0)] = DEFAULT_MAD_K,
    min_jump_pp: Annotated[float, Query(ge=0.5, le=50.0)] = DEFAULT_MIN_JUMP_PP,
    poly: Annotated[PolymarketClient, Depends(_get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> JumpsBacktestResponse:
    """Run the disagrees-jump paper-PnL backtest for one slug.

    Pipeline (mirrors ``/terminal/jumps/{slug}``):

        1. Resolve market metadata + relevance terms.
        2. Fetch hourly prices (Polymarket CLOB) and multi-source news.
        3. Detect jumps from the price series; score articles for
           relevance; attach matched articles + aggregate sentiment to
           each jump.
        4. Hand the jump list + price series to
           :func:`simulate_disagrees_pnl`.

    The agrees bucket is reported as a control — its mean should hover
    near zero if the disagrees signal is genuine reversion.
    """
    import asyncio

    # 1. Market metadata
    try:
        meta = poly.get_market_metadata(slug)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"market not found: {e}") from e

    question = meta.question or slug
    keywords = (
        extract_keywords(question, max_n=MAX_KEYWORDS)
        or [t for t in slug.replace("_", "-").split("-") if len(t) >= 3][:MAX_KEYWORDS]
    )
    terms = build_terms(question)
    query = build_phrase_query(terms) or _build_query(keywords) or slug

    # 2. Fetch news + prices in parallel
    timespan = f"{days}d"
    end_ts = pd.Timestamp.utcnow()
    start_ts = end_ts - pd.Timedelta(days=days)
    pad_start_unix = int((start_ts - pd.Timedelta(hours=24)).timestamp())

    raw_articles, prices = await asyncio.gather(
        asyncio.to_thread(_gather_all_news, poly._client, query, timespan),
        asyncio.to_thread(
            _fetch_hourly_prices,
            poly._client,
            poly.clob_url,
            meta.yes_token_id,
            pad_start_unix,
        ),
    )

    # 3. Pre-score articles
    scored: list[tuple[GDELTArticle, float, list[str]]] = []
    has_terms = bool(terms.anchors or terms.topics)
    for art in raw_articles:
        if has_terms:
            s_score, s_matched = score_relevance(art.title, terms)
            if s_score < RELEVANCE_MIN:
                continue
            scored.append((art, s_score, s_matched))
        else:
            scored.append((art, 0.5, []))

    # 4. Detect jumps + attach sentiment-aligned article context
    raw_jumps = detect_jumps(
        prices, mad_k=mad_k, min_jump_pp=min_jump_pp, rolling_hours=ROLLING_HOURS
    )
    enriched: list[dict[str, Any]] = []
    for j in raw_jumps:
        top, _n_window = _articles_for_jump(j["ts"], scored)
        direction = _direction(j["delta_logit"])
        sent_mean, _sent_label, sent_align = aggregate_sentiment(
            [a.sentiment_score for a in top],
            jump_direction=direction,
        )
        enriched.append(
            {
                "ts_iso": j["ts"].isoformat().replace("+00:00", "Z"),
                "price_before": float(j["price_before"]),
                "price_after": float(j["price_after"]),
                "direction": direction,
                "news_sentiment_score": float(sent_mean),
                "sentiment_alignment": sent_align,
            }
        )

    result = simulate_disagrees_pnl(enriched, prices, hold_hours=hold_hours)

    d_pnl = BacktestStats(**result["disagrees_pnl"])
    a_pnl = BacktestStats(**result["agrees_pnl"])
    eq_curve = [EquityPoint(**pt) for pt in result["equity_curve"]]

    if result["n_disagrees"] == 0:
        interpretation = (
            f"No disagrees jumps in the last {days}d for slug={slug}. "
            "Either the market is quiet or sentiment is consistently "
            "aligned with price moves."
        )
    else:
        # Verdict: contrarian "disagrees" signal works iff disagrees return
        # is meaningfully positive AND agrees return isn't simultaneously
        # higher (otherwise the agrees direction is the real alpha and
        # the disagrees rows are just misclassifications).
        d_works = d_pnl.mean_return > 0.005 and d_pnl.hit_rate > 0.5
        a_dominates = a_pnl.mean_return > d_pnl.mean_return + 0.01
        if d_works and not a_dominates:
            verdict = (
                "DISAGREES IS REAL ALPHA — news direction predicts reversal at "
                f"{hold_hours}h horizon."
            )
        elif a_pnl.mean_return > 0.01 and a_pnl.hit_rate > 0.55:
            verdict = (
                "AGREES IS THE REAL SIGNAL — news direction *predicts* price "
                "continuation. Bet WITH the wires, not against."
            )
        else:
            verdict = (
                "INCONCLUSIVE — neither bucket has meaningful PnL above noise. "
                "Try a different hold_hours or more days of data."
            )
        interpretation = (
            f"{result['n_disagrees']} disagrees + {result['n_agrees']} agrees over "
            f"{hold_hours}h holding · disagrees: {d_pnl.mean_return:+.4f} mean, "
            f"hit {d_pnl.hit_rate:.0%}, Sharpe {d_pnl.sharpe_naive:+.1f} · "
            f"agrees: {a_pnl.mean_return:+.4f} mean, hit {a_pnl.hit_rate:.0%}, "
            f"Sharpe {a_pnl.sharpe_naive:+.1f}. → {verdict}"
        )

    return JumpsBacktestResponse(
        slug=slug,
        hold_hours=int(hold_hours),
        n_disagrees=result["n_disagrees"],
        n_agrees=result["n_agrees"],
        disagrees_pnl=d_pnl,
        agrees_pnl=a_pnl,
        equity_curve=eq_curve,
        interpretation=interpretation,
    )


__all__ = [
    "DEFAULT_HOLD_HOURS",
    "MAX_HOLD_HOURS",
    "MIN_HOLD_HOURS",
    "TRADING_HOURS_PER_YEAR",
    "BacktestStats",
    "EquityPoint",
    "JumpsBacktestResponse",
    "get_jumps_backtest",
    "router",
    "simulate_disagrees_pnl",
]
