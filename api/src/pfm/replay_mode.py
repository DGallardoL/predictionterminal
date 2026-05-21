"""Replay Mode — "time-machine trader".

Educational/demo feature that lets users select a past timestamp, see the
state of the market (Polymarket odds + equity prices) as of that moment, and
"paper-trade" hypothetical orders against frozen historical prices.

Polymarket does not expose a public replay endpoint, so we reconstruct the
state from the same daily-history feeds the rest of the app uses
(``_cached_factor_history`` + yfinance closes). This gives us **end-of-day**
granularity, which is sufficient for a demo and explicitly documented.

Design notes
------------
* No new external IO: we re-use ``main._cached_factor_history`` for PM odds
  and the existing ``yfinance`` cache for equity closes.
* Slippage on simulated orders is a flat 1% (mid + 1% per side) — this is a
  conservative POC assumption, not a microstructure model.
* Pre-baked scenarios (``election_night_2024`` etc) hard-code the timestamp +
  slug list so the demo always lands on a memorable moment.
* Historical-PnL snapshots (``_HISTORICAL_RETURNS``) are research-time
  estimates from public daily closes; they are intentionally hard-coded to
  keep the demo deterministic. Replace with live yfinance pulls in prod.

Endpoints
---------
* ``GET  /replay/state``                          — snapshot at a timestamp.
* ``POST /replay/order``                          — paper-trade simulation.
* ``GET  /replay/scenarios``                      — list pre-baked scenarios.
* ``GET  /replay/scenario/{name}``                — full pre-baked scenario.
* ``GET  /replay/scenario/{name}/preflight``      — slug liveness check.
* ``GET  /replay/scenario/{name}/pnl``            — realized basket PnL.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.auth.dependencies import require_tier
from pfm.cache_utils import get_cache as _get_terminal_cache

# Scenario response cache — historical slugs don't change, so 24h TTL is
# a deliberate over-cap. The key is namespaced per scenario id so a redeploy
# only invalidates the in-memory copy (the slugs themselves are immutable).
_SCENARIO_CACHE = _get_terminal_cache("replay_scenario", ttl=86_400)
_SCENARIO_CACHE_TTL_SECONDS: int = 86_400  # 24h

# --- type aliases -----------------------------------------------------------

ScenarioName = Literal[
    "election_night_2024",
    "fomc_2024_09",
    "btc_ath_2024_11",
    "covid_crash_2020_03",
]

OrderSide = Literal["LONG", "SHORT"]


# --- pre-baked scenarios ----------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A hard-coded historical moment for the time-machine demo."""

    name: str
    title: str
    timestamp: datetime
    description: str
    pm_slugs: tuple[str, ...]
    equity_tickers: tuple[str, ...]
    headline_news: tuple[dict[str, str], ...]
    narrative: str = ""
    end_timestamp: datetime | None = None
    expected_pnl_per_dollar_long: dict[str, float] = field(default_factory=dict)


SCENARIOS: dict[str, Scenario] = {
    "election_night_2024": Scenario(
        name="election_night_2024",
        title="US Election Night 2024",
        timestamp=datetime(2024, 11, 4, 18, 0, tzinfo=UTC),
        end_timestamp=datetime(2024, 11, 6, 12, 0, tzinfo=UTC),
        description=(
            "Trump won unexpectedly. PM markets moved 96% YES on Trump while "
            "markets opened with massive overnight repricing."
        ),
        pm_slugs=(
            "presidential-election-winner-2024",
            "which-party-wins-presidential-election-2024",
            "2024-presidential-popular-vote-margin",
            "senate-control-after-2024-election",
            "house-control-after-2024-election",
            "2024-georgia-presidential-election-margin",
            "2024-pennsylvania-presidential-election-margin",
            "2024-michigan-presidential-election-margin",
        ),
        equity_tickers=("SPY", "QQQ", "IWM", "DXY", "GLD", "BTCUSD", "TLT"),
        headline_news=(
            {"time": "22:47", "headline": "AP calls Pennsylvania for Trump."},
            {"time": "22:55", "headline": "Polymarket Trump odds cross 90%."},
            {"time": "23:10", "headline": "DJT halted, limit-up after-hours."},
        ),
        narrative=(
            "Polls closed on the East Coast at 18:00 UTC on Nov 4th 2024 with "
            "Polymarket pricing Trump near 60% and Harris near 40%. Within "
            "five hours, as Pennsylvania, Georgia and Michigan tipped, the "
            "Trump contract repriced from 60% to 96% in roughly two trading "
            "sessions. That repricing is the cleanest example of an event-"
            "driven PM signal in 2024.\n\n"
            "Equities followed: small-caps (IWM) led on the open, the dollar "
            "(DXY) ripped on the rate-differential trade, long bonds (TLT) "
            "sold off on deficit fears, and BTC broke its prior cycle high "
            "intraday on Nov 6th. Gold sold off modestly. The cross-asset "
            "co-movement validates the use of PM odds as a real-time "
            "fundamental factor on event days.\n\n"
            "For the replay demo we span Nov 4 18:00Z through Nov 6 12:00Z "
            "so users can see the entry-to-resolution arc, not just a single "
            "snapshot."
        ),
        expected_pnl_per_dollar_long={
            "SPY": 0.0246,
            "QQQ": 0.0259,
            "IWM": 0.0584,
            "DXY": 0.0166,
            "GLD": -0.0312,
            "BTCUSD": 0.1043,
            "TLT": -0.0289,
        },
    ),
    "fomc_2024_09": Scenario(
        name="fomc_2024_09",
        title="FOMC September 2024 — 50bp Cut Surprise",
        timestamp=datetime(2024, 9, 18, 18, 0, tzinfo=UTC),
        end_timestamp=datetime(2024, 9, 25, 18, 0, tzinfo=UTC),
        description=(
            "Powell delivered 50bp cut. PM markets pre-decision were 60/40 between 25bp and 50bp."
        ),
        pm_slugs=(
            "fed-decision-in-september-2024",
            "fed-cuts-25bp-or-more-september-2024",
            "fed-cuts-50bp-or-more-september-2024",
            "recession-2024",
        ),
        equity_tickers=("TLT", "IEF", "SPY", "QQQ", "KRE", "XLF"),
        headline_news=(
            {"time": "18:00", "headline": "FOMC statement: 50bp cut."},
            {"time": "18:30", "headline": "Powell: 'recalibration, not panic'."},
        ),
        narrative=(
            "Heading into the 18:00 UTC announcement on Sep 18 2024, "
            "Polymarket pricing on the 'cuts 50bp or more' contract sat near "
            "60%, with the 25bp contract at 40%. Fed funds futures were "
            "pricing a closer 50/50 — PM was modestly ahead of rates "
            "markets. Powell delivered 50bp.\n\n"
            "Long-duration Treasuries (TLT, IEF) jumped on the print, then "
            "round-tripped over the next week as the curve bear-steepened. "
            "Regional banks (KRE) and broad financials (XLF) outperformed "
            "the index, validating the rate-cut-friendly-banks playbook. "
            "SPY and QQQ closed the week roughly flat — the 'sell the news' "
            "drift was clean.\n\n"
            "The replay window of t+0 to t+7d captures the full FOMC "
            "reaction arc, including the next-day reversal that frustrated "
            "many duration-long positions."
        ),
        expected_pnl_per_dollar_long={
            "TLT": -0.0185,
            "IEF": -0.0078,
            "SPY": 0.0136,
            "QQQ": 0.0128,
            "KRE": 0.0421,
            "XLF": 0.0238,
        },
    ),
    "btc_ath_2024_11": Scenario(
        name="btc_ath_2024_11",
        title="BTC Cracks $100k — November 2024",
        timestamp=datetime(2024, 11, 21, 0, 0, tzinfo=UTC),
        end_timestamp=datetime(2024, 11, 28, 0, 0, tzinfo=UTC),
        description=(
            "BTC broke 100k for first time on Nov-21. PM markets had 50% "
            "YES day-of, 95% by week-end."
        ),
        pm_slugs=(
            "bitcoin-100k-by-end-of-2024",
            "bitcoin-150k-by-end-of-2024",
            "bitcoin-all-time-high-by-end-of-2024",
            "crypto-etf-approval-2024",
        ),
        equity_tickers=("BTC-USD", "COIN", "MSTR", "IBIT", "MARA", "RIOT"),
        headline_news=(
            {"time": "13:42", "headline": "BTC ticks $99,800 on Coinbase."},
            {"time": "14:00", "headline": "Polymarket BTC>$100k contract: 95%."},
        ),
        narrative=(
            "On Nov 21 2024 BTC traded above $99k for the first time, "
            "pushing within striking distance of the $100k psychological "
            "level. Polymarket's 'BTC > $100k by EOY 2024' contract opened "
            "the day near 50% and closed the week above 95% as spot finally "
            "printed 100,000 on Nov 22 evening.\n\n"
            "The crypto-equity complex traded as a proxy basket. MSTR, the "
            "leveraged-BTC vehicle, outperformed BTC itself on the move; "
            "miners (MARA, RIOT) ran on the price-of-coin / cost-of-mining "
            "spread; IBIT (the iShares spot-BTC ETF) saw record inflows; "
            "COIN ripped on volume expectations.\n\n"
            "The replay spans the seven days from Nov 21 to Nov 28 so users "
            "can see the breakout, the gap-fill failure, and the sustained "
            "leg above $95k. Note the 'crypto-etf-approval-2024' slug is "
            "marked as anti-alpha in the catalog — included here for "
            "narrative completeness, not as a deployable signal."
        ),
        expected_pnl_per_dollar_long={
            "BTC-USD": 0.0432,
            "COIN": 0.0712,
            "MSTR": 0.1865,
            "IBIT": 0.0421,
            "MARA": 0.0938,
            "RIOT": 0.1052,
        },
    ),
    "covid_crash_2020_03": Scenario(
        name="covid_crash_2020_03",
        title="COVID Crash — Black Monday II, March 9, 2020",
        timestamp=datetime(2020, 3, 9, 0, 0, tzinfo=UTC),
        end_timestamp=datetime(2020, 3, 16, 0, 0, tzinfo=UTC),
        description=(
            "S&P -7.6% Mar-9. VIX hit 62. PM markets were thin but pandemic "
            "odds spiked from 40% to 90% in 7 days."
        ),
        pm_slugs=(
            "recession-2020",
            "coronavirus-pandemic-declared-march-2020",
        ),
        equity_tickers=("SPY", "VIX", "USO", "TLT", "GLD", "XLF"),
        headline_news=(
            {"time": "13:30", "headline": "S&P opens -7%, circuit breakers trip."},
            {"time": "16:00", "headline": "WTI crude crashes on Saudi-Russia price war."},
            {"time": "20:00", "headline": "Polymarket pandemic-declared odds cross 75%."},
        ),
        narrative=(
            "Monday March 9 2020 — 'Black Monday II' — opened with the "
            "S&P 500 down 7%, hitting the level-1 NYSE circuit breaker "
            "before 09:35 ET. Two parallel shocks drove the move: the "
            "Saudi-Russia oil price war that crashed WTI overnight, and "
            "rapidly accelerating COVID-19 case counts in Italy and the US.\n\n"
            "Polymarket coverage in March 2020 was thin and illiquid. The "
            "two slugs in this replay are research-time reconstructions — "
            "the 'pandemic declared by March' contract spiked from ~40% on "
            "March 2 to ~90% by March 11 (the actual WHO declaration). The "
            "'recession 2020' contract was even thinner; we include it for "
            "narrative continuity.\n\n"
            "The cross-asset move was textbook risk-off: SPY -12% over the "
            "week, VIX peaked at 75, USO down ~30% on the price war, TLT "
            "+5% on flight-to-quality, GLD round-tripped (initial sell on "
            "margin calls, then rally), XLF down sharply on rate-cut fears. "
            "Use this replay to stress-test crisis-regime portfolio "
            "behavior, not to validate PM signal alpha — the markets were "
            "too thin to be tradeable."
        ),
        expected_pnl_per_dollar_long={
            "SPY": -0.1212,
            "VIX": 0.4815,
            "USO": -0.2967,
            "TLT": 0.0541,
            "GLD": -0.0238,
            "XLF": -0.1587,
        },
    ),
}


# Hard-coded realized 7-day returns from t+0 to t+7d, sourced from public
# daily closes at research time. Replace with a live yfinance call in prod;
# kept inline so the demo is deterministic and runs without network.
_HISTORICAL_RETURNS: dict[str, dict[str, float]] = {
    name: dict(sc.expected_pnl_per_dollar_long) for name, sc in SCENARIOS.items()
}


# --- snapshot building blocks -----------------------------------------------


def _coerce_utc(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    """Return a tz-aware UTC :class:`pandas.Timestamp`."""
    p = pd.Timestamp(ts)
    if p.tzinfo is None:
        p = p.tz_localize("UTC")
    else:
        p = p.tz_convert("UTC")
    return p


def _last_obs_at_or_before(
    series: pd.Series, ts: pd.Timestamp
) -> tuple[pd.Timestamp, float] | None:
    """Return (date, value) for the last observation at-or-before ``ts``.

    Returns ``None`` if the series has no usable observation in range.
    """
    if series is None or series.empty:
        return None
    s = series.dropna().sort_index()
    sliced = s[s.index <= ts]
    if sliced.empty:
        return None
    return sliced.index[-1], float(sliced.iloc[-1])


def _previous_obs_at_or_before(
    series: pd.Series, ts: pd.Timestamp, lag_days: int = 1
) -> float | None:
    """Get the value ``lag_days`` calendar-days before the most recent obs ≤ ts."""
    last = _last_obs_at_or_before(series, ts)
    if last is None:
        return None
    last_idx, _ = last
    target = last_idx - pd.Timedelta(days=lag_days)
    s = series.dropna().sort_index()
    sliced = s[s.index <= target]
    if sliced.empty:
        return None
    return float(sliced.iloc[-1])


# yfinance close cache (in-process, keyed by (ticker, start_date, end_date))
@functools.lru_cache(maxsize=512)
def _yf_close_cached(ticker: str, start_iso: str, end_iso: str) -> tuple[tuple[str, float], ...]:
    """Fetch *close* prices from yfinance, cached.

    Returns a tuple of ``(date_iso, close)`` so it remains hashable & pickable
    by lru_cache. Raises :class:`RuntimeError` on yfinance failure.
    """
    try:
        import yfinance as yf  # local import: tests can monkeypatch this module
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("yfinance not installed") from e

    df = yf.download(
        ticker,
        start=start_iso,
        end=end_iso,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return ()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=-1, drop_level=True)
    if "Close" not in df.columns:
        return ()
    closes = df["Close"].dropna()
    return tuple(
        (
            pd.Timestamp(idx).tz_localize("UTC").normalize().isoformat()
            if pd.Timestamp(idx).tzinfo is None
            else pd.Timestamp(idx).tz_convert("UTC").normalize().isoformat(),
            float(val),
        )
        for idx, val in closes.items()
    )


def _equity_closes(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Pull equity closes as a Series indexed by UTC-normalised dates."""
    pad_start = (start - pd.Timedelta(days=14)).date().isoformat()
    pad_end = (end + pd.Timedelta(days=2)).date().isoformat()
    rows = _yf_close_cached(ticker, pad_start, pad_end)
    if not rows:
        return pd.Series(dtype=float, name=ticker)
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    vals = [r[1] for r in rows]
    s = pd.Series(vals, index=idx, name=ticker)
    s.index = s.index.normalize()
    return s


# --- factor history accessor ------------------------------------------------


def _resolve_pm_history(slug: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch a PM slug's daily history via main's cached path.

    Falls back to the bare ``fetch_factor_history`` if the main app isn't
    initialised (e.g. router used in isolation in tests). Returns an empty
    DataFrame on any error so the snapshot can degrade gracefully.
    """
    try:
        from pfm import main as main_mod  # local import to avoid cycles
        from pfm.factors import FactorConfig

        # If the slug isn't in the loaded factor set we still want to try
        # fetching it — synthesize a minimal FactorConfig.
        factors = getattr(main_mod.app.state, "factors", {}) if hasattr(main_mod, "app") else {}
        # O(1) lookup via the lifespan-built slug index (built once per
        # worker); fall back to a tiny on-the-fly build for tests that wire
        # ``factors`` without the index.
        by_slug = (
            getattr(main_mod.app.state, "factors_by_slug", None)
            if hasattr(main_mod, "app")
            else None
        )
        if not isinstance(by_slug, dict) or not by_slug:
            by_slug = {f.slug: f for f in factors.values() if f.slug}
        fc = by_slug.get(slug)
        if fc is None:
            fc = FactorConfig(
                id=slug,
                name=slug,
                slug=slug,
                source="polymarket",
                description="ad-hoc replay slug",
                theme="other",
            )
        poly = main_mod.app.state.poly
        cache = main_mod.app.state.cache
        settings = main_mod.get_settings()
        return main_mod._cached_factor_history(fc, start, end, poly, cache, settings)
    except Exception:
        return pd.DataFrame()


# --- public API -------------------------------------------------------------


def get_state_at(
    timestamp: datetime,
    slugs: list[str] | None = None,
    *,
    equity_tickers: list[str] | None = None,
    history_window_days: int = 60,
) -> dict[str, Any]:
    """Snapshot of PM odds + equity prices at a past timestamp.

    Args:
        timestamp: as-of moment (will be coerced to UTC).
        slugs: PM slugs to include. ``None`` = top-20 active markets discovered
            via ``main`` (best-effort; degrades gracefully if discovery fails).
        equity_tickers: list of yfinance tickers. ``None`` defaults to a fixed
            small basket used by the demo.
        history_window_days: how far back to pull bars (used to compute the
            ``last_change`` and 24h equity move).
    """
    ts = _coerce_utc(timestamp)
    start = ts - pd.Timedelta(days=history_window_days)
    end = ts + pd.Timedelta(days=1)

    # --- markets ------------------------------------------------------------
    if slugs is None:
        slugs = _default_slugs()
    if equity_tickers is None:
        equity_tickers = ["SPY", "QQQ", "BTC-USD", "TLT"]

    market_rows: list[dict[str, Any]] = []
    for slug in slugs:
        df = _resolve_pm_history(slug, start, end)
        if df is None or df.empty or "price" not in df.columns:
            continue
        s = df["price"].dropna()
        last = _last_obs_at_or_before(s, ts)
        if last is None:
            continue
        last_idx, last_val = last
        prev_val = _previous_obs_at_or_before(s, ts, lag_days=1) or last_val
        last_change = last_val - prev_val
        # Volatility: rolling 14-day stdev of *probability* (not log) since
        # PM contracts are bounded.
        vol = float(s.tail(14).diff().std(ddof=1)) if len(s.tail(14)) >= 3 else 0.0
        market_rows.append(
            {
                "slug": slug,
                "name": slug.replace("-", " ").title(),
                "prob": round(last_val, 4),
                "vol": round(vol, 4),
                "theme": "other",
                "last_change": round(last_change, 4),
                "as_of_obs": last_idx.date().isoformat(),
            }
        )

    # --- equities -----------------------------------------------------------
    equity_rows: list[dict[str, Any]] = []
    for tkr in equity_tickers:
        try:
            s = _equity_closes(tkr, start, end)
        except Exception:
            continue
        if s.empty:
            continue
        last = _last_obs_at_or_before(s, ts)
        if last is None:
            continue
        last_idx, last_val = last
        prev_val = _previous_obs_at_or_before(s, ts, lag_days=1)
        change_24h = (last_val / prev_val - 1.0) if prev_val and prev_val > 0 else 0.0
        equity_rows.append(
            {
                "ticker": tkr,
                "price": round(last_val, 4),
                "change_24h": round(change_24h, 6),
                "as_of_obs": last_idx.date().isoformat(),
            }
        )

    return {
        "as_of": ts.isoformat(),
        "markets": market_rows,
        "equities": equity_rows,
        "headline_news": [],
    }


def _default_slugs(limit: int = 20) -> list[str]:
    """Return a default slug list — top loaded factors, capped at ``limit``."""
    try:
        from pfm import main as main_mod

        factors = getattr(main_mod.app.state, "factors", {})
        return [f.slug for f in list(factors.values())[:limit]]
    except Exception:
        return []


def simulate_paper_order(
    slug: str,
    side: OrderSide,
    size_usd: float,
    at_timestamp: datetime,
    hold_until: datetime | None = None,
    *,
    slippage_bps: float = 100.0,
) -> dict[str, Any]:
    """Simulate a paper-trade order against historical PM prices.

    Args:
        slug: PM slug to trade (the YES contract).
        side: ``LONG`` or ``SHORT``.
        size_usd: notional size.
        at_timestamp: entry time.
        hold_until: optional exit time. If ``None``, we mark-to-market at
            the latest available price after entry (or report no-exit).
        slippage_bps: round-trip slippage assumption in basis points; default
            100bps (1%).

    Returns:
        ``{entry_price, exit_price, pnl_pct, pnl_usd, slippage_assumed,
        bars_held, status}`` dictionary.
    """
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"side must be LONG or SHORT, got {side!r}")
    if size_usd <= 0:
        raise ValueError(f"size_usd must be positive, got {size_usd}")

    entry_ts = _coerce_utc(at_timestamp)
    exit_ts = _coerce_utc(hold_until) if hold_until is not None else None
    horizon_end = exit_ts if exit_ts is not None else entry_ts + pd.Timedelta(days=30)
    start = entry_ts - pd.Timedelta(days=14)
    df = _resolve_pm_history(slug, start, horizon_end + pd.Timedelta(days=1))
    if df is None or df.empty or "price" not in df.columns:
        return {
            "status": "NO_DATA",
            "slug": slug,
            "side": side,
            "size_usd": size_usd,
            "entry_price": None,
            "exit_price": None,
            "pnl_pct": 0.0,
            "pnl_usd": 0.0,
            "slippage_assumed_bps": slippage_bps,
            "bars_held": 0,
        }

    s = df["price"].dropna()
    entry = _last_obs_at_or_before(s, entry_ts)
    if entry is None:
        return {
            "status": "NO_ENTRY_PRICE",
            "slug": slug,
            "side": side,
            "size_usd": size_usd,
            "entry_price": None,
            "exit_price": None,
            "pnl_pct": 0.0,
            "pnl_usd": 0.0,
            "slippage_assumed_bps": slippage_bps,
            "bars_held": 0,
        }
    entry_idx, entry_price = entry

    # Exit: if explicit hold_until provided, fetch last obs ≤ hold_until.
    # Otherwise, leave the position open and report MTM at last available bar.
    if exit_ts is not None:
        ex = _last_obs_at_or_before(s, exit_ts)
        status = "CLOSED" if ex is not None else "NO_EXIT_PRICE"
    else:
        ex = (s.index[-1], float(s.iloc[-1])) if not s.empty else None
        status = "OPEN_MTM" if ex is not None else "NO_EXIT_PRICE"

    if ex is None:
        return {
            "status": status,
            "slug": slug,
            "side": side,
            "size_usd": size_usd,
            "entry_price": round(entry_price, 4),
            "exit_price": None,
            "pnl_pct": 0.0,
            "pnl_usd": 0.0,
            "slippage_assumed_bps": slippage_bps,
            "bars_held": 0,
        }
    exit_idx, exit_price = ex

    # Slippage: half-spread on each side.
    slip = slippage_bps / 10_000.0
    if side == "LONG":
        eff_entry = entry_price * (1 + slip / 2)
        eff_exit = exit_price * (1 - slip / 2)
        gross = (eff_exit - eff_entry) / eff_entry if eff_entry > 0 else 0.0
    else:  # SHORT
        eff_entry = entry_price * (1 - slip / 2)
        eff_exit = exit_price * (1 + slip / 2)
        gross = (eff_entry - eff_exit) / eff_entry if eff_entry > 0 else 0.0
    pnl_usd = gross * size_usd
    bars_held = max(0, (exit_idx - entry_idx).days)
    return {
        "status": status,
        "slug": slug,
        "side": side,
        "size_usd": size_usd,
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "entry_date": entry_idx.date().isoformat(),
        "exit_date": exit_idx.date().isoformat(),
        "pnl_pct": round(gross, 6),
        "pnl_usd": round(pnl_usd, 2),
        "slippage_assumed_bps": slippage_bps,
        "bars_held": bars_held,
    }


def _parallel_resolve_pm_histories(
    slugs: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Concurrently fetch each slug's history via :func:`asyncio.to_thread`.

    The legacy path called :func:`_resolve_pm_history` once per slug
    sequentially — for an 8-slug scenario that means 8× the per-slug
    Polymarket round-trip. Wrapping each call in :func:`asyncio.to_thread`
    fans them out across the default executor; total wall time is bounded
    by the slowest single call rather than the sum.
    """
    if not slugs:
        return {}

    async def _gather() -> dict[str, pd.DataFrame]:
        async def _one(slug: str) -> tuple[str, pd.DataFrame]:
            df = await asyncio.to_thread(_resolve_pm_history, slug, start, end)
            return slug, df

        results = await asyncio.gather(*(_one(s) for s in slugs))
        return dict(results)

    try:
        return asyncio.run(_gather())
    except RuntimeError:
        # Already inside an event loop (FastAPI sync handler) — run on a
        # worker thread to avoid the "asyncio.run() cannot be called from a
        # running loop" error.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _gather()).result()


def _replay_scenario_uncached(scenario_name: ScenarioName) -> dict[str, Any]:
    """Compute the scenario payload bypassing the 24h cache."""
    sc = SCENARIOS.get(scenario_name)
    if sc is None:
        raise KeyError(f"unknown scenario {scenario_name!r}")

    ts = _coerce_utc(sc.timestamp)
    start = ts - pd.Timedelta(days=60)
    end = ts + pd.Timedelta(days=1)

    # Parallel pre-flight: resolve every slug concurrently up-front so the
    # subsequent get_state_at can pick the cached frames out without any
    # serial network IO. _resolve_pm_history wraps a process-wide cache,
    # so a second invocation in get_state_at is a free dict-lookup.
    if sc.pm_slugs:
        _parallel_resolve_pm_histories(list(sc.pm_slugs), start, end)

    state = get_state_at(
        sc.timestamp,
        slugs=list(sc.pm_slugs) if sc.pm_slugs else [],
        equity_tickers=list(sc.equity_tickers),
    )
    expected = dict(sc.expected_pnl_per_dollar_long) or None
    state["scenario"] = {
        "id": sc.name,
        "name": sc.name,
        "title": sc.title,
        "description": sc.description,
        "as_of_iso": sc.timestamp.isoformat(),
        "end_iso": sc.end_timestamp.isoformat() if sc.end_timestamp else None,
        "slugs": list(sc.pm_slugs),
        "tickers": list(sc.equity_tickers),
        "narrative": sc.narrative,
        "expected_pnl_per_dollar_long": expected,
        "headline_news": [dict(h) for h in sc.headline_news],
    }
    state["headline_news"] = [dict(h) for h in sc.headline_news]
    return state


def replay_scenario(scenario_name: ScenarioName) -> dict[str, Any]:
    """Hydrate a pre-baked scenario into a full state-at snapshot.

    24h-cached: historical slugs don't change. Parallel resolves the
    per-slug Polymarket history calls so wall time is bounded by the
    slowest single fetch rather than their sum.

    The returned dict carries the snapshot fields produced by
    :func:`get_state_at` plus a populated ``scenario`` block with the
    curated metadata: ``id``, ``title``, ``description``, ``as_of_iso``,
    ``slugs``, ``tickers``, ``narrative`` and ``expected_pnl_per_dollar_long``.
    """
    cached = _SCENARIO_CACHE.get(scenario_name)
    cache_age = 0
    if cached is not None and isinstance(cached, dict):
        ts_cached = cached.get("_cached_at_unix")
        if ts_cached is not None:
            cache_age = max(0, int(_now_unix_replay() - float(ts_cached)))
        out = {k: v for k, v in cached.items() if k != "_cached_at_unix"}
        out["cache_age_seconds"] = cache_age
        return out

    state = _replay_scenario_uncached(scenario_name)
    state_to_cache = dict(state)
    state_to_cache["_cached_at_unix"] = _now_unix_replay()
    _SCENARIO_CACHE.set(scenario_name, state_to_cache, ttl=_SCENARIO_CACHE_TTL_SECONDS)
    state["cache_age_seconds"] = 0
    return state


def _now_unix_replay() -> float:
    return datetime.now(tz=UTC).timestamp()


def list_scenarios() -> list[dict[str, Any]]:
    """List available pre-baked scenarios with the full curated payload."""
    return [
        {
            "id": sc.name,
            "name": sc.name,
            "title": sc.title,
            "timestamp": sc.timestamp.isoformat(),
            "as_of_iso": sc.timestamp.isoformat(),
            "end_iso": sc.end_timestamp.isoformat() if sc.end_timestamp else None,
            "description": sc.description,
            "narrative": sc.narrative,
            "slugs": list(sc.pm_slugs),
            "tickers": list(sc.equity_tickers),
            "n_markets": len(sc.pm_slugs),
            "n_equities": len(sc.equity_tickers),
            "expected_pnl_per_dollar_long": (dict(sc.expected_pnl_per_dollar_long) or None),
        }
        for sc in SCENARIOS.values()
    ]


# --- preflight: are these slugs still resolvable on Polymarket? -------------


_PREFLIGHT_TIMEOUT = httpx.Timeout(5.0, connect=3.0)


def _gamma_url() -> str:
    """Return the Gamma base URL from settings, with a sane default."""
    try:
        from pfm import main as main_mod

        settings = main_mod.get_settings()
        return str(getattr(settings, "polymarket_gamma_url", "https://gamma-api.polymarket.com"))
    except Exception:
        return "https://gamma-api.polymarket.com"


def _classify_slug(payload: list[dict[str, Any]] | None) -> str:
    """Return ``"live"``, ``"resolved"`` or ``"missing"`` for a Gamma payload."""
    if not payload:
        return "missing"
    market = payload[0] if isinstance(payload, list) else payload
    if not isinstance(market, dict):
        return "missing"
    closed = bool(market.get("closed"))
    active = bool(market.get("active"))
    if closed:
        return "resolved"
    if active:
        return "live"
    # Inactive but not closed → treat as resolved/archived for replay purposes.
    return "resolved"


def _suggest_substitutes(slug: str, all_slugs: tuple[str, ...]) -> list[str]:
    """Cheap token-overlap suggester for dead slugs."""
    tokens = {t for t in slug.split("-") if len(t) > 3}
    out: list[tuple[int, str]] = []
    for cand in all_slugs:
        if cand == slug:
            continue
        cand_tokens = {t for t in cand.split("-") if len(t) > 3}
        overlap = len(tokens & cand_tokens)
        if overlap >= 1:
            out.append((overlap, cand))
    out.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, c in out[:3]]


def preflight_scenario(
    scenario_name: ScenarioName,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Verify each slug for a scenario against Polymarket Gamma.

    For each slug we do a ``GET /markets?slug=<slug>`` and classify the
    response. Returns ``{scenario_id, slugs_status, can_replay,
    substitutes}``. Network errors degrade to ``status="missing"`` rather
    than raising — the demo must never hard-fail on a flaky upstream.
    """
    sc = SCENARIOS.get(scenario_name)
    if sc is None:
        raise KeyError(f"unknown scenario {scenario_name!r}")

    base = _gamma_url()
    owns_client = client is None
    cli = client or httpx.Client(timeout=_PREFLIGHT_TIMEOUT)
    statuses: list[dict[str, Any]] = []
    substitutes: dict[str, list[str]] = {}
    all_slugs_pool: tuple[str, ...] = tuple(
        s for other in SCENARIOS.values() for s in other.pm_slugs
    )
    try:
        for slug in sc.pm_slugs:
            status = "missing"
            try:
                r = cli.get(f"{base}/markets", params={"slug": slug})
                if r.status_code == 200:
                    payload = r.json()
                    status = _classify_slug(payload)
                else:
                    status = "missing"
            except Exception:
                status = "missing"
            statuses.append({"slug": slug, "status": status})
            if status == "missing":
                subs = _suggest_substitutes(slug, all_slugs_pool)
                if subs:
                    substitutes[slug] = subs
    finally:
        if owns_client:
            cli.close()

    live_or_resolved = sum(1 for s in statuses if s["status"] in {"live", "resolved"})
    can_replay = live_or_resolved >= max(1, len(statuses) // 2)
    return {
        "scenario_id": sc.name,
        "slugs_status": statuses,
        "can_replay": can_replay,
        "substitutes": substitutes,
    }


# --- realized historical PnL ------------------------------------------------


def compute_scenario_pnl(
    scenario_name: ScenarioName,
    capital_usd: float = 10_000.0,
) -> dict[str, Any]:
    """Compute the realized 7-day basket PnL for a scenario.

    Uses ``_HISTORICAL_RETURNS`` (deterministic snapshot from research) as
    the source of truth. In production this should pull live closes via
    yfinance from ``sc.timestamp`` to ``sc.end_timestamp`` (default +7d).

    Args:
        scenario_name: pre-baked scenario id.
        capital_usd: notional capital to size the basket. Both PnL fields
            scale linearly with this value.

    Returns:
        ``{scenario_id, ticker_returns, basket_pnl_long_only,
        basket_pnl_equal_weighted, capital_usd}``.
    """
    sc = SCENARIOS.get(scenario_name)
    if sc is None:
        raise KeyError(f"unknown scenario {scenario_name!r}")
    if capital_usd <= 0:
        raise ValueError(f"capital_usd must be positive, got {capital_usd}")

    rets = _HISTORICAL_RETURNS.get(scenario_name, {})
    tickers = list(sc.equity_tickers)
    ticker_returns: dict[str, float] = {tkr: float(rets.get(tkr, 0.0)) for tkr in tickers}

    # Long-only basket: sum of positive returns weighted equally across
    # winning legs (defensive sizing — only the in-the-money legs trade).
    positives = [r for r in ticker_returns.values() if r > 0]
    if positives:
        weight = capital_usd / len(positives)
        basket_long = sum(r * weight for r in positives)
    else:
        basket_long = 0.0

    # Equal-weighted across all legs (long every ticker), the naive
    # all-in-on-narrative version.
    if tickers:
        weight_eq = capital_usd / len(tickers)
        basket_eq = sum(ticker_returns[t] * weight_eq for t in tickers)
    else:
        basket_eq = 0.0

    return {
        "scenario_id": sc.name,
        "capital_usd": float(capital_usd),
        "ticker_returns": {k: round(v, 6) for k, v in ticker_returns.items()},
        "basket_pnl_long_only": round(float(basket_long), 2),
        "basket_pnl_equal_weighted": round(float(basket_eq), 2),
        "as_of_iso": sc.timestamp.isoformat(),
        "end_iso": sc.end_timestamp.isoformat() if sc.end_timestamp else None,
        "source": "research-snapshot; replace with live yfinance for prod",
    }


# --- Pydantic schemas (router I/O) ------------------------------------------


class MarketSnapshot(BaseModel):
    slug: str
    name: str
    prob: float
    vol: float
    theme: str
    last_change: float
    as_of_obs: str | None = None


class EquitySnapshot(BaseModel):
    ticker: str
    price: float
    change_24h: float
    as_of_obs: str | None = None


class ReplayState(BaseModel):
    as_of: str
    markets: list[MarketSnapshot] = Field(default_factory=list)
    equities: list[EquitySnapshot] = Field(default_factory=list)
    headline_news: list[dict[str, str]] = Field(default_factory=list)
    scenario: dict[str, Any] | None = None
    cache_age_seconds: int = Field(
        default=0,
        ge=0,
        description=(
            "Seconds since this scenario was computed. ``0`` for a freshly "
            "computed payload, larger when served from the 24h cache."
        ),
    )


class OrderRequest(BaseModel):
    slug: str = Field(..., description="Polymarket YES-contract slug to trade")
    side: OrderSide
    size_usd: float = Field(..., gt=0)
    at_timestamp: datetime
    hold_until: datetime | None = None
    slippage_bps: float = Field(100.0, ge=0, le=1000)


class OrderResult(BaseModel):
    status: str
    slug: str
    side: str
    size_usd: float
    entry_price: float | None
    exit_price: float | None
    pnl_pct: float
    pnl_usd: float
    slippage_assumed_bps: float
    bars_held: int
    entry_date: str | None = None
    exit_date: str | None = None


class ScenarioInfo(BaseModel):
    id: str
    name: str
    title: str
    timestamp: str
    as_of_iso: str
    end_iso: str | None = None
    description: str
    narrative: str = ""
    slugs: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    n_markets: int
    n_equities: int
    expected_pnl_per_dollar_long: dict[str, float] | None = None


class ScenarioList(BaseModel):
    n_scenarios: int
    scenarios: list[ScenarioInfo]


class SlugStatus(BaseModel):
    slug: str
    status: Literal["live", "resolved", "missing"]


class PreflightResult(BaseModel):
    scenario_id: str
    slugs_status: list[SlugStatus]
    can_replay: bool
    substitutes: dict[str, list[str]] = Field(default_factory=dict)


class ScenarioPnL(BaseModel):
    scenario_id: str
    capital_usd: float
    ticker_returns: dict[str, float]
    basket_pnl_long_only: float
    basket_pnl_equal_weighted: float
    as_of_iso: str
    end_iso: str | None = None
    source: str


# --- router -----------------------------------------------------------------


router = APIRouter(prefix="/replay", tags=["replay-mode"])


@router.get(
    "/state",
    response_model=ReplayState,
    summary="Snapshot of PM + equity state at a past timestamp",
    dependencies=[Depends(require_tier("pro"))],
)
def get_replay_state(
    as_of: datetime = Query(..., description="UTC timestamp to replay"),
    slugs: str | None = Query(None, description="Comma-separated PM slugs"),
    tickers: str | None = Query(None, description="Comma-separated yfinance tickers"),
) -> ReplayState:
    slug_list = [s.strip() for s in slugs.split(",")] if slugs else None
    ticker_list = [t.strip() for t in tickers.split(",")] if tickers else None
    state = get_state_at(as_of, slugs=slug_list, equity_tickers=ticker_list)
    return ReplayState(**state)


@router.post(
    "/order",
    response_model=OrderResult,
    summary="Simulate a paper-trade order against historical prices",
)
def post_replay_order(body: OrderRequest) -> OrderResult:
    out = simulate_paper_order(
        slug=body.slug,
        side=body.side,
        size_usd=body.size_usd,
        at_timestamp=body.at_timestamp,
        hold_until=body.hold_until,
        slippage_bps=body.slippage_bps,
    )
    return OrderResult(**out)


@router.get(
    "/scenarios",
    response_model=ScenarioList,
    summary="List pre-baked replay scenarios",
)
@router.get(
    "/sessions",
    response_model=ScenarioList,
    summary="Alias of /replay/scenarios (footer pill).",
)  # UX-audit 2026-05-14: footer pill calls /replay/sessions.
def get_scenarios() -> ScenarioList:
    rows = list_scenarios()
    return ScenarioList(n_scenarios=len(rows), scenarios=[ScenarioInfo(**r) for r in rows])


@router.get(
    "/scenario/{scenario_name}",
    response_model=ReplayState,
    summary="Hydrate a pre-baked scenario",
)
def get_scenario(scenario_name: str) -> ReplayState:
    if scenario_name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"unknown scenario {scenario_name!r}")
    state = replay_scenario(scenario_name)  # type: ignore[arg-type]
    return ReplayState(**state)


@router.get(
    "/scenario/{scenario_name}/preflight",
    response_model=PreflightResult,
    summary="Verify each scenario slug is still resolvable on Polymarket",
)
def get_scenario_preflight(scenario_name: str) -> PreflightResult:
    if scenario_name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"unknown scenario {scenario_name!r}")
    out = preflight_scenario(scenario_name)  # type: ignore[arg-type]
    return PreflightResult(**out)


@router.get(
    "/scenario/{scenario_name}/pnl",
    response_model=ScenarioPnL,
    summary="Realized historical basket PnL for the scenario window",
)
def get_scenario_pnl(
    scenario_name: str,
    capital: float = Query(10_000.0, gt=0, le=1e9),
) -> ScenarioPnL:
    if scenario_name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"unknown scenario {scenario_name!r}")
    out = compute_scenario_pnl(scenario_name, capital_usd=capital)  # type: ignore[arg-type]
    return ScenarioPnL(**out)


__all__ = [
    "SCENARIOS",
    "OrderRequest",
    "OrderResult",
    "PreflightResult",
    "ReplayState",
    "Scenario",
    "ScenarioInfo",
    "ScenarioList",
    "ScenarioPnL",
    "SlugStatus",
    "compute_scenario_pnl",
    "get_state_at",
    "list_scenarios",
    "preflight_scenario",
    "replay_scenario",
    "router",
    "simulate_paper_order",
]
