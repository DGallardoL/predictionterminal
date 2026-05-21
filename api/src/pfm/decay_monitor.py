"""Decay monitor for deployed alpha strategies.

Computes rolling-Sharpe on a strategy's equity curve and compares the
*current* Sharpe against the *baseline* Sharpe (the in-sample / OOS
Sharpe published when the strategy was promoted). The ratio drives a
4-state classifier ``FRESH`` / ``STABLE`` / ``DECAYING`` / ``DEAD`` and
a tier-demote recommendation that the α-Hub can act on.

Data sources (resolved in priority order, configurable):

1. ``live_signals``       — read ``web/data/live_signals.json`` and use
   the curated spread / z trail when present. Most accurate.
2. ``polymarket_history`` — fetch raw daily price history for both legs
   via the Polymarket Gamma + CLOB stack and rebuild the spread.
3. ``synthetic_fallback`` — last-resort deterministic synthesis seeded
   on ``pair_id``. Behaviour is unchanged from the previous POC and is
   the only path that works without external state. Each response now
   carries a ``source_used`` and ``data_quality_note`` field so callers
   can tell at a glance which source produced the verdict.

Routing note: this module owns its :class:`fastapi.APIRouter` mirroring
the conventions used elsewhere (terminal_calendar_curated etc.).
``main.py`` only needs::

    from pfm.decay_monitor import router as decay_monitor_router
    app.include_router(decay_monitor_router)

It also exposes an opt-in background refresh job (every 4h by default)
gated on ``PFM_DECAY_REFRESH_ENABLED=1`` — see :func:`run_forever`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)


# --- constants --------------------------------------------------------------

DEFAULT_ALPHA_STRATEGIES_PATH: str = "web/data/alpha_strategies.json"
DEFAULT_LIVE_SIGNALS_PATH: str = "web/data/live_signals.json"

#: Path written by the optional background refresh job. Read-only for
#: the ``GET /alpha/decay`` endpoint to surface ``last_refreshed_iso``.
DEFAULT_DECAY_STATUS_PATH: str = "/tmp/pfm_decay_status.json"

# Annualisation factor for daily returns (252 trading days / year).
ANNUALISATION_DAYS: int = 252

# Default rolling window — one trading month.
DEFAULT_WINDOW: int = 30

# Default ratio threshold below which a single observation counts as
# "below baseline" for the consecutive-observations counter.
DEFAULT_BELOW_RATIO: float = 0.5

#: Polymarket endpoints used by the real-data fallback fetcher.
GAMMA_URL: str = "https://gamma-api.polymarket.com"
CLOB_URL: str = "https://clob.polymarket.com"

#: TTL (seconds) for the real-data cache. Defaults to 4h to match the
#: opt-in background refresh cron cadence.
DECAY_REAL_CACHE_TTL: int = 14400

# Tier ladder used by ``demote_recommendation``. Order matters: the
# function steps DOWN this ladder by ``decay_indicator`` severity.
_TIER_LADDER: tuple[str, ...] = ("A_GOLD", "B_VALIDATED", "C_TENTATIVE")


DecayIndicator = Literal["FRESH", "STABLE", "DECAYING", "DEAD"]
DemoteTier = Literal["A_GOLD", "B_VALIDATED", "C_TENTATIVE"]
DataSource = Literal["live_signals", "polymarket_history", "synthetic_fallback"]


# --- core math --------------------------------------------------------------


def compute_rolling_sharpe(returns: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Annualised rolling Sharpe over ``window`` observations.

    Sharpe = mean(r) / std(r, ddof=1) * sqrt(252).

    Behaviour notes:

    * For all-zero / zero-variance windows we emit ``0.0`` (rather than
      ``inf`` / ``NaN``) so downstream classification can treat them as
      "no signal".
    * The first ``window - 1`` rows are ``NaN`` by construction.
    * The series is reindexed to its input order — no resampling.

    Args:
        returns: Daily simple or log returns (any unit, as long as the
            baseline used the same convention).
        window: Number of observations per rolling block. Must be ≥ 2.

    Returns:
        ``pd.Series`` of annualised rolling Sharpe values.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if returns.empty:
        return pd.Series(dtype=float, name="rolling_sharpe")

    r = returns.astype(float)
    mu = r.rolling(window).mean()
    sigma = r.rolling(window).std(ddof=1)
    # Compute Sharpe; preserve NaN in the warmup region (mu / sigma both
    # NaN) and emit 0.0 only where sigma is *exactly* zero (degenerate
    # zero-variance window inside a fully-populated block). The warmup
    # mask uses ``mu.isna()`` so we don't accidentally zero those out.
    raw = (mu / sigma) * math.sqrt(ANNUALISATION_DAYS)
    zero_var = (sigma == 0.0) & (~mu.isna())
    raw = raw.where(~zero_var, other=0.0)
    raw.name = "rolling_sharpe"
    return raw


def detect_decay(
    rolling_sh: pd.Series,
    baseline: float,
    threshold_pct: float = DEFAULT_BELOW_RATIO,
) -> dict[str, Any]:
    """Classify a rolling-Sharpe series against a baseline Sharpe.

    The classifier rules (in evaluation order):

    * ``DEAD``      — ratio < 0.3  OR  n_consecutive_below ≥ 10
    * ``DECAYING``  — 0.3 ≤ ratio < 0.7  OR  n_consecutive_below ≥ 5
    * ``FRESH``     — ratio > 0.9  AND  n_consecutive_below < 3
    * ``STABLE``    — 0.7 ≤ ratio ≤ 0.9 (the residual band)

    The ``n_consecutive_below`` counter is the run-length of the most
    recent suffix where ``rolling_sh < threshold_pct * baseline``.

    Args:
        rolling_sh: Output of :func:`compute_rolling_sharpe`.
        baseline: The published Sharpe to compare against (e.g. the
            ``oos_sharpe`` from ``alpha_strategies.json``).
        threshold_pct: Fraction of ``baseline`` below which an
            observation counts as "below baseline". Default 0.5.

    Returns:
        Dict with the schema documented in the module docstring.
    """
    last_priced = rolling_sh.dropna()
    current = float(last_priced.iloc[-1]) if not last_priced.empty else 0.0
    base = float(baseline) if baseline != 0.0 else 1e-9
    ratio = current / base

    # Count consecutive trailing observations below the threshold.
    cutoff = threshold_pct * base
    n_consecutive_below = 0
    if not last_priced.empty:
        for x in reversed(last_priced.tolist()):
            if x < cutoff:
                n_consecutive_below += 1
            else:
                break

    indicator: DecayIndicator
    if ratio < 0.3 or n_consecutive_below >= 10:
        indicator = "DEAD"
    elif (0.3 <= ratio < 0.7) or n_consecutive_below >= 5:
        indicator = "DECAYING"
    elif ratio > 0.9 and n_consecutive_below < 3:
        indicator = "FRESH"
    else:
        indicator = "STABLE"

    demote: DemoteTier = _demote_for(indicator)

    return {
        "current_sharpe": current,
        "baseline_sharpe": float(baseline),
        "ratio": ratio,
        "decay_indicator": indicator,
        "demote_recommendation": demote,
        "n_consecutive_below": n_consecutive_below,
    }


def _demote_for(indicator: DecayIndicator) -> DemoteTier:
    """Map a decay indicator to the recommended *worst-case* tier."""
    if indicator in ("FRESH", "STABLE"):
        return "A_GOLD"
    if indicator == "DECAYING":
        return "B_VALIDATED"
    return "C_TENTATIVE"


# --- synthetic returns (POC fallback) --------------------------------------


def _synthesize_returns(
    pair_id: str,
    baseline_sharpe: float,
    n_days: int = 180,
    decay: bool = False,
) -> pd.Series:
    """Generate deterministic synthetic daily returns for ``pair_id``.

    The series is built so that the *trailing* (full-window) Sharpe is
    approximately ``baseline_sharpe`` annualised. This is intentionally
    simple — the goal is to make ``/alpha/decay`` return realistic-
    looking numbers for the demo without any external dependency.

    When ``decay=True`` the last 60 observations are scaled down by
    0.2× so the rolling-Sharpe at the tail collapses — useful both as
    a POC "what would decay look like" view and as a test fixture.

    .. deprecated::
        Prefer :func:`_load_real_returns` and reserve this synthesis
        for the explicit ``synthetic_fallback`` data-source.

    Args:
        pair_id: Strategy identifier, used to seed the RNG.
        baseline_sharpe: Target annualised Sharpe for the synthesis.
        n_days: Length of the daily returns series.
        decay: If ``True`` inject a scale-down in the trailing tail.

    Returns:
        A daily-frequency :class:`pd.Series` indexed on UTC dates,
        named ``returns``.
    """
    seed = abs(hash(pair_id)) % (2**32)
    rng = np.random.default_rng(seed)
    # Daily vol of 1% is in the ballpark of pair-trade equity curves
    # in the alpha cards; the mean is solved so ann-Sharpe ≈ baseline.
    daily_vol = 0.01
    daily_mean = baseline_sharpe * daily_vol / math.sqrt(ANNUALISATION_DAYS)
    noise = rng.normal(loc=daily_mean, scale=daily_vol, size=n_days)

    if decay:
        # Tail-degradation: the last 60 obs collapse to ~ 0 mean,
        # mimicking an alpha that died.
        tail = min(60, n_days)
        noise[-tail:] = rng.normal(loc=0.0, scale=daily_vol, size=tail) * 0.5

    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=n_days, freq="D")
    return pd.Series(noise, index=idx, name="returns")


def _load_synthetic_returns_fallback(
    pair_id: str, n: int = 180, baseline_sharpe: float = 1.0
) -> pd.Series:
    """Thin wrapper around :func:`_synthesize_returns`.

    .. deprecated::
        Kept only so the explicit ``synthetic_fallback`` source has a
        named, ergonomic entry-point that can be monkey-patched in
        tests. Prefer :func:`_load_real_returns` when at all possible.
    """
    return _synthesize_returns(pair_id, baseline_sharpe=baseline_sharpe, n_days=n)


# --- real-data loaders ------------------------------------------------------


def _resolve_live_signals_path(live_signals_path: str) -> Path:
    """Resolve ``live_signals_path`` against likely roots.

    Same logic as :func:`_resolve_alpha_path` so callers can pass either
    the default sentinel (``web/data/live_signals.json``) or an explicit
    absolute path under a tmp directory in tests.
    """
    p = Path(live_signals_path)
    if p.is_file():
        return p
    if live_signals_path != DEFAULT_LIVE_SIGNALS_PATH:
        return p
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / live_signals_path
    if candidate.is_file():
        return candidate
    return p


def _load_live_signals_payload(
    live_signals_path: str = DEFAULT_LIVE_SIGNALS_PATH,
) -> dict[str, Any] | None:
    """Read ``live_signals.json`` once. Returns ``None`` if absent / invalid."""
    resolved = _resolve_live_signals_path(live_signals_path)
    if not resolved.is_file():
        return None
    try:
        with resolved.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("live_signals.json unreadable at %s: %s", resolved, e)
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _returns_from_live_signals(
    pair_id: str,
    live_signals: dict[str, Any],
) -> pd.Series | None:
    """Extract a returns proxy from ``live_signals.json`` for one pair.

    The current snapshot only carries the *latest* spread (no full
    trail), so we approximate returns from any of these, in order:

    1. an explicit ``spread_history`` / ``trail`` array on the entry
       (forward-compat with a future trail-aware writer),
    2. a ``z_history`` array (rescale by ``sigma_window`` to recover
       a spread-like trail, then diff),
    3. nothing — return ``None`` so the caller can degrade gracefully.

    The returned series is named ``returns`` and indexed by UTC date.
    """
    sig_block = live_signals.get("signals") or {}
    entry = sig_block.get(pair_id)
    if not isinstance(entry, dict):
        return None
    if "error" in entry:
        return None

    trail = entry.get("spread_history") or entry.get("trail")
    if isinstance(trail, list) and len(trail) >= 5:
        try:
            spread = pd.Series([float(x) for x in trail], dtype=float)
        except (TypeError, ValueError):
            return None
        return _spread_to_returns(spread)

    z_trail = entry.get("z_history")
    sigma = entry.get("sigma_window")
    if isinstance(z_trail, list) and len(z_trail) >= 5 and sigma:
        try:
            spread = pd.Series([float(x) * float(sigma) for x in z_trail], dtype=float)
        except (TypeError, ValueError):
            return None
        return _spread_to_returns(spread)

    return None


def _spread_to_returns(spread: pd.Series) -> pd.Series | None:
    """First-difference a spread series into a return-like series.

    Pair-trade PnL is roughly proportional to the change in the spread
    between consecutive bars (with sign flipped relative to position),
    so |Δspread| is a perfectly serviceable returns proxy for a Sharpe
    monitor that only cares about ratio vs baseline. Returns ``None`` if
    the input is too short to diff.
    """
    if spread is None or spread.empty or len(spread) < 2:
        return None
    diffs = spread.astype(float).diff().dropna()
    if diffs.empty:
        return None
    end = pd.Timestamp.utcnow().normalize()
    idx = pd.date_range(end=end, periods=len(diffs), freq="D")
    out = pd.Series(diffs.to_numpy(), index=idx, name="returns")
    return out


def _fetch_clob_token_ids(
    http: httpx.Client, slug: str, *, timeout: float = 5.0
) -> tuple[str | None, str | None]:
    """Look up the YES/NO ``clobTokenIds`` for a Polymarket slug.

    The Gamma response stores ``clobTokenIds`` as a JSON-encoded string
    of two token IDs (yes, no). Returns ``(None, None)`` on any failure.
    """
    try:
        r = http.get(f"{GAMMA_URL}/markets", params={"slug": slug}, timeout=timeout)
        r.raise_for_status()
        arr = r.json() or []
    except (httpx.HTTPError, ValueError):
        return (None, None)
    if not (isinstance(arr, list) and arr):
        return (None, None)
    raw = arr[0].get("clobTokenIds")
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return (None, None)
    if not isinstance(ids, list) or len(ids) < 2:
        return (None, None)
    return (str(ids[0]), str(ids[1]))


def _fetch_clob_daily_prices(
    http: httpx.Client, token_id: str, *, days: int, timeout: float = 5.0
) -> pd.Series:
    """Fetch ``days`` of daily Polymarket close prices for ``token_id``.

    Uses ``fidelity=1440`` (daily) which is the only fidelity that works
    reliably for resolved markets — see PLAN.md §5.3. Returns an empty
    Series on any failure so callers can degrade.
    """
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400
    params: dict[str, str | int] = {
        "market": token_id,
        "fidelity": 1440,
        "startTs": start_ts,
        "endTs": end_ts,
    }
    try:
        r = http.get(f"{CLOB_URL}/prices-history", params=params, timeout=timeout)
        r.raise_for_status()
        history = (r.json() or {}).get("history", []) or []
    except (httpx.HTTPError, ValueError):
        return pd.Series(dtype=float)
    rows = [
        (int(b["t"]), float(b["p"]))
        for b in history
        if isinstance(b, dict) and "t" in b and "p" in b
    ]
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True).normalize()
    s = pd.Series([r[1] for r in rows], index=idx, name="price").sort_index()
    # Collapse intra-day duplicates that can sneak in from CLOB.
    return s.groupby(s.index).last()


def _returns_from_polymarket_history(
    pair_id: str,
    alpha: dict[str, Any],
    *,
    days: int = 90,
    http: httpx.Client | None = None,
) -> pd.Series | None:
    """Reconstruct a returns proxy from raw Polymarket history.

    Reads the ``a_slug`` / ``b_slug`` (preferred) or ``a_id`` / ``b_id``
    fields off the alpha record, fetches daily closes for both legs,
    builds the inner-joined spread and returns its first-differences.
    """
    a_slug = alpha.get("a_slug") or alpha.get("a_id")
    b_slug = alpha.get("b_slug") or alpha.get("b_id")
    if not a_slug or not b_slug:
        return None
    beta = float(alpha.get("beta_hedge") or 1.0)

    own_client = http is None
    client = http or httpx.Client(timeout=7.0)
    try:
        a_yes, _ = _fetch_clob_token_ids(client, str(a_slug))
        b_yes, _ = _fetch_clob_token_ids(client, str(b_slug))
        if not a_yes or not b_yes:
            return None
        s_a = _fetch_clob_daily_prices(client, a_yes, days=days)
        s_b = _fetch_clob_daily_prices(client, b_yes, days=days)
    finally:
        if own_client:
            client.close()
    if s_a.empty or s_b.empty:
        return None
    aligned = pd.concat({"a": s_a, "b": s_b}, axis=1).dropna()
    if len(aligned) < 5:
        return None
    spread = aligned["a"] - beta * aligned["b"]
    return _spread_to_returns(spread)


def _load_real_returns(
    pair_id: str,
    alpha_dict: dict[str, Any],
    *,
    days: int = 90,
    live_signals: dict[str, Any] | None = None,
    http: httpx.Client | None = None,
    allow_polymarket: bool = True,
) -> tuple[pd.Series | None, DataSource | None]:
    """Best-effort load of a real return-like series for one alpha.

    Args:
        pair_id: Strategy identifier (key into ``live_signals.signals``).
        alpha_dict: Strategy record from ``alpha_strategies.json``.
        days: Lookback in days for the polymarket fallback.
        live_signals: Optional pre-loaded ``live_signals.json`` payload.
        http: Optional ``httpx.Client`` to reuse for Polymarket fetches.
        allow_polymarket: Set ``False`` to skip the network fallback.

    Returns:
        ``(returns, source)`` — the second element is the source label
        used when the returns are non-empty, or ``None`` if every source
        failed and the caller must degrade to synthetic.
    """
    if live_signals is not None:
        live_returns = _returns_from_live_signals(pair_id, live_signals)
        if live_returns is not None and not live_returns.empty:
            return live_returns, "live_signals"
    if allow_polymarket:
        try:
            poly_returns = _returns_from_polymarket_history(
                pair_id, alpha_dict, days=days, http=http
            )
        except Exception as exc:
            logger.warning(
                "decay_monitor: polymarket fallback failed pair=%s err=%s",
                pair_id,
                exc,
            )
            poly_returns = None
        if poly_returns is not None and not poly_returns.empty:
            return poly_returns, "polymarket_history"
    return None, None


# --- alpha catalog loading --------------------------------------------------


def _resolve_alpha_path(alpha_strategies_path: str) -> Path:
    """Resolve ``alpha_strategies_path`` against likely roots.

    Tries the path as-is first, then — only when the caller passed the
    default sentinel — falls back to ``<repo>/web/data/alpha_strategies.json``.
    A caller-supplied path that doesn't exist is returned verbatim so
    :func:`_load_alpha_strategies` can log a single warning and degrade
    to ``[]``.
    """
    p = Path(alpha_strategies_path)
    if p.is_file():
        return p
    if alpha_strategies_path != DEFAULT_ALPHA_STRATEGIES_PATH:
        return p
    # Default path → try repo-root resolution.
    # api/src/pfm/decay_monitor.py → repo root is parents[3].
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / alpha_strategies_path
    if candidate.is_file():
        return candidate
    return p


def _load_alpha_strategies(path: str) -> list[dict[str, Any]]:
    """Read and normalise the alpha catalog JSON.

    Returns ``strategies`` (a list of dicts with at least ``pair_id``,
    ``oos_sharpe``, ``tier``). Returns ``[]`` on any error.
    """
    resolved = _resolve_alpha_path(path)
    try:
        with resolved.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        logger.warning("alpha_strategies.json not found at %s", resolved)
        return []
    except json.JSONDecodeError as e:
        logger.warning("alpha_strategies.json parse failed: %s", e)
        return []

    strategies = payload.get("strategies", [])
    if not isinstance(strategies, list):
        logger.warning("alpha_strategies.json: 'strategies' is not a list")
        return []
    return strategies


def _classify_with_source(
    pair_id: str,
    alpha: dict[str, Any],
    baseline: float,
    *,
    window: int,
    data_source: DataSource,
    live_signals: dict[str, Any] | None,
    http: httpx.Client | None,
    allow_polymarket: bool,
) -> dict[str, Any]:
    """Run :func:`detect_decay` for one pair, picking the best data source.

    The ``data_source`` arg behaves like a *minimum*-acceptable source
    rather than a hard pin. Concretely:

    * ``"live_signals"`` tries live_signals → polymarket → synthetic.
    * ``"polymarket_history"`` skips live_signals, tries polymarket →
      synthetic.
    * ``"synthetic_fallback"`` goes straight to the deterministic synth.

    The returned dict carries ``source_used`` and a human-readable
    ``data_quality_note`` (only when the synth fallback fired).
    """
    cache = get_cache("decay_real", ttl=DECAY_REAL_CACHE_TTL)
    cache_key = (pair_id, baseline, window, data_source)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    returns: pd.Series | None = None
    source_used: DataSource = "synthetic_fallback"
    note: str | None = None

    if data_source == "synthetic_fallback":
        returns = _load_synthetic_returns_fallback(pair_id, baseline_sharpe=max(baseline, 1e-6))
        source_used = "synthetic_fallback"
        note = (
            "Synthetic fallback in use: returns are deterministically "
            "seeded from pair_id; verdict is illustrative only."
        )
    else:
        ls = live_signals if data_source == "live_signals" else None
        returns, real_source = _load_real_returns(
            pair_id,
            alpha,
            live_signals=ls,
            http=http,
            allow_polymarket=allow_polymarket,
        )
        if returns is None or real_source is None:
            returns = _load_synthetic_returns_fallback(pair_id, baseline_sharpe=max(baseline, 1e-6))
            source_used = "synthetic_fallback"
            note = (
                "Synthetic fallback in use: real data unavailable for "
                "this pair (no live_signals trail and Polymarket fetch "
                "either skipped or empty)."
            )
        else:
            source_used = real_source

    rolling = compute_rolling_sharpe(returns, window=window)
    status = detect_decay(rolling, baseline)
    status["pair_id"] = pair_id
    status["tier"] = alpha.get("tier")
    status["n_obs"] = len(returns)
    status["source_used"] = source_used
    status["data_quality_note"] = note

    cache.set(cache_key, status, ttl=DECAY_REAL_CACHE_TTL)
    return status


def check_all_alphas(
    alpha_strategies_path: str = DEFAULT_ALPHA_STRATEGIES_PATH,
    window: int = DEFAULT_WINDOW,
    *,
    data_source: DataSource = "live_signals",
    live_signals_path: str = DEFAULT_LIVE_SIGNALS_PATH,
    allow_polymarket: bool = False,
    http: httpx.Client | None = None,
) -> dict[str, Any]:
    """Run :func:`detect_decay` against every strategy in the catalog.

    Args:
        alpha_strategies_path: Path to ``alpha_strategies.json``.
        window: Rolling-window length in days. Default 30.
        data_source: Preferred upstream — see :func:`_classify_with_source`.
        live_signals_path: Path to ``live_signals.json``.
        allow_polymarket: Disable to short-circuit the network fallback
            (useful in tests so we don't accidentally hit Polymarket).
        http: Optional ``httpx.Client`` reused across pairs.

    Returns:
        Dict with keys ``items`` (mapping ``pair_id -> status``),
        ``n_total``, ``n_using_real_data``, ``n_using_synthetic_fallback``.
        ``items`` mirrors the previous flat-mapping shape so callers
        that iterate ``check_all_alphas(...)`` keep working — see
        :meth:`__getitem__` on the returned object below.
    """
    strategies = _load_alpha_strategies(alpha_strategies_path)
    live_signals: dict[str, Any] | None = None
    if data_source == "live_signals":
        live_signals = _load_live_signals_payload(live_signals_path)

    items: dict[str, dict[str, Any]] = {}
    n_real = 0
    n_synth = 0
    for s in strategies:
        pair_id = s.get("pair_id")
        if not pair_id:
            continue
        baseline = float(s.get("oos_sharpe") or 0.0)
        if baseline == 0.0:
            # Avoid divide-by-zero blow-ups: skip silently.
            continue
        status = _classify_with_source(
            pair_id,
            s,
            baseline,
            window=window,
            data_source=data_source,
            live_signals=live_signals,
            http=http,
            allow_polymarket=allow_polymarket,
        )
        if status["source_used"] == "synthetic_fallback":
            n_synth += 1
        else:
            n_real += 1
        items[pair_id] = status

    return _DecayResult(
        statuses=items,
        n_using_real_data=n_real,
        n_using_synthetic_fallback=n_synth,
    )


class _DecayResult(dict):
    """Backward-compatible mapping returned by :func:`check_all_alphas`.

    Older callers iterated the dict expecting ``pair_id -> status``
    pairs. The new return shape adds ``n_using_real_data`` and
    ``n_using_synthetic_fallback`` *as attributes* while keeping the
    plain dict interface (``[]``, ``keys()``, ``items()``, equality vs
    ``{}``) so existing tests keep passing without any callsite change.
    """

    def __init__(
        self,
        *,
        statuses: dict[str, dict[str, Any]],
        n_using_real_data: int,
        n_using_synthetic_fallback: int,
    ) -> None:
        super().__init__(statuses)
        self.items_dict: dict[str, dict[str, Any]] = statuses
        self.n_using_real_data = n_using_real_data
        self.n_using_synthetic_fallback = n_using_synthetic_fallback

    @property
    def n_total(self) -> int:
        return len(self.items_dict)


# --- pydantic schemas -------------------------------------------------------


class DecayStatus(BaseModel):
    """Single-strategy decay status payload."""

    pair_id: str
    tier: str | None = None
    current_sharpe: float
    baseline_sharpe: float
    ratio: float = Field(..., description="current_sharpe / baseline_sharpe.")
    decay_indicator: DecayIndicator
    demote_recommendation: DemoteTier
    n_consecutive_below: int = Field(..., ge=0)
    n_obs: int = Field(..., ge=0)
    source_used: DataSource = Field(
        default="synthetic_fallback",
        description="Which upstream produced the returns series.",
    )
    data_quality_note: str | None = Field(
        default=None,
        description="Human-readable warning when the synth fallback fired.",
    )


class DecayListResponse(BaseModel):
    """Response model for ``GET /alpha/decay``."""

    n_total: int
    n_fresh: int
    n_stable: int
    n_decaying: int
    n_dead: int
    n_using_real_data: int
    n_using_synthetic_fallback: int
    last_refreshed_iso: str | None = None
    data_quality_warning: str | None = None
    items: list[DecayStatus]


class RollingSharpePoint(BaseModel):
    date: str
    rolling_sharpe: float | None


class RollingSharpeResponse(BaseModel):
    pair_id: str
    window: int
    baseline_sharpe: float
    n_obs: int
    series: list[RollingSharpePoint]
    source_used: DataSource = "synthetic_fallback"
    data_quality_note: str | None = None


class RecomputeResponse(BaseModel):
    pair_id: str
    status: DecayStatus
    forced: bool = True


# --- helpers shared by endpoints --------------------------------------------


def _strategy_for_pair(pair_id: str, alpha_strategies_path: str) -> dict[str, Any]:
    """Return the strategy entry for ``pair_id``.

    Raises:
        HTTPException(404): when the pair is unknown.
    """
    strategies = _load_alpha_strategies(alpha_strategies_path)
    for s in strategies:
        if s.get("pair_id") == pair_id:
            return s
    raise HTTPException(status_code=404, detail=f"unknown alpha pair_id {pair_id!r}")


def _read_status_cache(
    status_path: str = DEFAULT_DECAY_STATUS_PATH,
) -> dict[str, Any] | None:
    """Read ``last_refreshed_iso`` etc. from the cron-written status file."""
    p = Path(status_path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_status_cache(
    payload: dict[str, Any], status_path: str = DEFAULT_DECAY_STATUS_PATH
) -> None:
    """Atomically write the cron status file (best-effort)."""
    p = Path(status_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(p)
    except OSError as exc:
        logger.warning("decay_monitor: status write failed: %s", exc)


# --- background refresh job (opt-in) ----------------------------------------


async def run_once(
    *,
    alpha_strategies_path: str = DEFAULT_ALPHA_STRATEGIES_PATH,
    live_signals_path: str = DEFAULT_LIVE_SIGNALS_PATH,
    status_path: str = DEFAULT_DECAY_STATUS_PATH,
    window: int = DEFAULT_WINDOW,
    data_source: DataSource = "live_signals",
) -> dict[str, Any]:
    """Run one ``check_all_alphas`` cycle and persist the status file.

    Designed to be called from :func:`run_forever`. Returns the status
    payload that was just written.
    """
    t0 = time.perf_counter()

    def _work() -> _DecayResult:
        return check_all_alphas(
            alpha_strategies_path=alpha_strategies_path,
            window=window,
            data_source=data_source,
            live_signals_path=live_signals_path,
        )

    result = await asyncio.to_thread(_work)
    duration = time.perf_counter() - t0
    payload = {
        "last_refreshed_iso": datetime.now(tz=UTC).isoformat(),
        "duration_seconds": round(duration, 3),
        "n_total": result.n_total,
        "n_using_real_data": result.n_using_real_data,
        "n_using_synthetic_fallback": result.n_using_synthetic_fallback,
        "data_source": data_source,
    }
    _write_status_cache(payload, status_path=status_path)
    logger.info(
        "decay_monitor: run_once total=%d real=%d synth=%d duration=%.2fs",
        result.n_total,
        result.n_using_real_data,
        result.n_using_synthetic_fallback,
        duration,
    )
    return payload


async def run_forever(
    interval_seconds: int = 14400,
    *,
    alpha_strategies_path: str = DEFAULT_ALPHA_STRATEGIES_PATH,
    live_signals_path: str = DEFAULT_LIVE_SIGNALS_PATH,
    status_path: str = DEFAULT_DECAY_STATUS_PATH,
    window: int = DEFAULT_WINDOW,
    data_source: DataSource = "live_signals",
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run :func:`run_once` repeatedly. Default cadence: 4h.

    Cancellation: callers can either ``cancel()`` the surrounding task
    or set ``stop_event``. Both paths exit cleanly without losing an
    in-flight write (``run_once`` completes before the sleep).
    """
    interval = max(60, int(interval_seconds))
    while True:
        try:
            await run_once(
                alpha_strategies_path=alpha_strategies_path,
                live_signals_path=live_signals_path,
                status_path=status_path,
                window=window,
                data_source=data_source,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("decay_monitor: run_once raised: %s", exc)

        if stop_event is not None and stop_event.is_set():
            return
        try:
            if stop_event is None:
                await asyncio.sleep(interval)
            else:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                return
        except TimeoutError:
            continue
        except asyncio.CancelledError:
            raise


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/alpha", tags=["decay-monitor"])


@router.get("/decay", response_model=DecayListResponse)
def list_decay_status(
    window: int = Query(DEFAULT_WINDOW, ge=2, le=365),
    alpha_strategies_path: str = Query(DEFAULT_ALPHA_STRATEGIES_PATH),
    data_source: DataSource = Query("live_signals"),
    live_signals_path: str = Query(DEFAULT_LIVE_SIGNALS_PATH),
    allow_polymarket: bool = Query(False),
) -> DecayListResponse:
    """List the decay status of every strategy in the catalog.

    The response is sorted by ``decay_indicator`` severity (DEAD first,
    FRESH last) so that callers can rank by attention required. The
    payload also surfaces real-vs-synthetic counts and a refresh
    timestamp from the optional background job.
    """
    result = check_all_alphas(
        alpha_strategies_path,
        window=window,
        data_source=data_source,
        live_signals_path=live_signals_path,
        allow_polymarket=allow_polymarket,
    )
    severity = {"DEAD": 0, "DECAYING": 1, "STABLE": 2, "FRESH": 3}
    items_sorted = sorted(
        result.items_dict.values(),
        key=lambda s: (severity.get(s["decay_indicator"], 4), -s["ratio"]),
    )
    items = [DecayStatus(**s) for s in items_sorted]
    counts: dict[str, int] = {"FRESH": 0, "STABLE": 0, "DECAYING": 0, "DEAD": 0}
    for s in items:
        counts[s.decay_indicator] += 1
    status_cache = _read_status_cache()
    last_iso = status_cache.get("last_refreshed_iso") if status_cache else None
    warning: str | None = None
    if result.n_total > 0:
        synth_pct = result.n_using_synthetic_fallback / result.n_total
        if synth_pct > 0.5:
            warning = (
                f"{result.n_using_synthetic_fallback}/{result.n_total} pairs "
                f"({synth_pct:.0%}) fell back to synthetic returns; verdicts "
                "are illustrative — re-run after live_signals.json refreshes."
            )
    return DecayListResponse(
        n_total=len(items),
        n_fresh=counts["FRESH"],
        n_stable=counts["STABLE"],
        n_decaying=counts["DECAYING"],
        n_dead=counts["DEAD"],
        n_using_real_data=result.n_using_real_data,
        n_using_synthetic_fallback=result.n_using_synthetic_fallback,
        last_refreshed_iso=last_iso,
        data_quality_warning=warning,
        items=items,
    )


@router.get("/{pair_id}/rolling-sharpe", response_model=RollingSharpeResponse)
def get_rolling_sharpe(
    pair_id: str,
    window: int = Query(DEFAULT_WINDOW, ge=2, le=365),
    alpha_strategies_path: str = Query(DEFAULT_ALPHA_STRATEGIES_PATH),
    data_source: DataSource = Query("live_signals"),
    live_signals_path: str = Query(DEFAULT_LIVE_SIGNALS_PATH),
    allow_polymarket: bool = Query(False),
) -> RollingSharpeResponse:
    """Return the daily rolling-Sharpe series for one strategy."""
    strategy = _strategy_for_pair(pair_id, alpha_strategies_path)
    baseline = float(strategy.get("oos_sharpe") or 0.0)
    returns, source_used, note = _resolve_returns_for_endpoint(
        pair_id,
        strategy,
        baseline=baseline,
        data_source=data_source,
        live_signals_path=live_signals_path,
        allow_polymarket=allow_polymarket,
    )
    rolling = compute_rolling_sharpe(returns, window=window)
    points: list[RollingSharpePoint] = []
    for ts, val in rolling.items():
        points.append(
            RollingSharpePoint(
                date=ts.date().isoformat() if hasattr(ts, "date") else str(ts),
                rolling_sharpe=None if pd.isna(val) else float(val),
            )
        )
    return RollingSharpeResponse(
        pair_id=pair_id,
        window=window,
        baseline_sharpe=baseline,
        n_obs=len(returns),
        series=points,
        source_used=source_used,
        data_quality_note=note,
    )


@router.post("/{pair_id}/recompute-decay", response_model=RecomputeResponse)
def recompute_decay(
    pair_id: str,
    window: int = Query(DEFAULT_WINDOW, ge=2, le=365),
    alpha_strategies_path: str = Query(DEFAULT_ALPHA_STRATEGIES_PATH),
    data_source: DataSource = Query("live_signals"),
    live_signals_path: str = Query(DEFAULT_LIVE_SIGNALS_PATH),
    allow_polymarket: bool = Query(False),
) -> RecomputeResponse:
    """Force a recompute of the decay status for one strategy.

    Drops the relevant ``decay_real`` cache entry first so the next read
    actually re-resolves the data source rather than returning stale.
    """
    strategy = _strategy_for_pair(pair_id, alpha_strategies_path)
    baseline = float(strategy.get("oos_sharpe") or 0.0)
    # Bust the cached classification so we genuinely recompute. The
    # cache lacks a ``delete`` method so we drop every entry — recompute
    # is an admin-only refresh, not a hot path.
    cache = get_cache("decay_real", ttl=DECAY_REAL_CACHE_TTL)
    with contextlib.suppress(Exception):
        cache.clear()
    returns, source_used, note = _resolve_returns_for_endpoint(
        pair_id,
        strategy,
        baseline=baseline,
        data_source=data_source,
        live_signals_path=live_signals_path,
        allow_polymarket=allow_polymarket,
    )
    rolling = compute_rolling_sharpe(returns, window=window)
    raw = detect_decay(rolling, baseline)
    raw["pair_id"] = pair_id
    raw["tier"] = strategy.get("tier")
    raw["n_obs"] = len(returns)
    raw["source_used"] = source_used
    raw["data_quality_note"] = note
    return RecomputeResponse(pair_id=pair_id, status=DecayStatus(**raw))


def _resolve_returns_for_endpoint(
    pair_id: str,
    strategy: dict[str, Any],
    *,
    baseline: float,
    data_source: DataSource,
    live_signals_path: str,
    allow_polymarket: bool,
) -> tuple[pd.Series, DataSource, str | None]:
    """Pick a returns series for the per-pair endpoints.

    Returns the same triple as :func:`_classify_with_source` exposes via
    the status dict but as a plain tuple so the per-pair endpoints
    don't have to re-shape a status dict.
    """
    if data_source == "synthetic_fallback":
        returns = _load_synthetic_returns_fallback(pair_id, baseline_sharpe=max(baseline, 1e-6))
        return (
            returns,
            "synthetic_fallback",
            "Synthetic fallback in use: returns deterministic; demo only.",
        )
    live_signals = (
        _load_live_signals_payload(live_signals_path) if data_source == "live_signals" else None
    )
    real_returns, source = _load_real_returns(
        pair_id,
        strategy,
        live_signals=live_signals,
        allow_polymarket=allow_polymarket,
    )
    if real_returns is not None and source is not None:
        return real_returns, source, None
    returns = _load_synthetic_returns_fallback(pair_id, baseline_sharpe=max(baseline, 1e-6))
    return (
        returns,
        "synthetic_fallback",
        "Synthetic fallback in use: real data unavailable for this pair.",
    )


__all__ = [
    "ANNUALISATION_DAYS",
    "DECAY_REAL_CACHE_TTL",
    "DEFAULT_ALPHA_STRATEGIES_PATH",
    "DEFAULT_DECAY_STATUS_PATH",
    "DEFAULT_LIVE_SIGNALS_PATH",
    "DEFAULT_WINDOW",
    "DataSource",
    "DecayIndicator",
    "DecayListResponse",
    "DecayStatus",
    "DemoteTier",
    "RecomputeResponse",
    "RollingSharpePoint",
    "RollingSharpeResponse",
    "_load_real_returns",
    "_load_synthetic_returns_fallback",
    "check_all_alphas",
    "compute_rolling_sharpe",
    "detect_decay",
    "router",
    "run_forever",
    "run_once",
]
