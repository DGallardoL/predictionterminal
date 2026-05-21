"""Background job that recomputes ``web/data/live_signals.json`` every N minutes.

Reads ``web/data/alpha_strategies.json``, and for each curated alpha pair
recomputes the live spread z-score, current signal action, and decay
status from fresh leg-A / leg-B price history. Writes the updated
``live_signals.json`` atomically (temp file + rename) so concurrent
readers in the frontend always see a consistent snapshot.

Design notes
------------
* The fetcher is **dependency-injected**. ``recompute_all_signals``
  accepts a ``fetcher`` callable so tests can hand in a synthetic
  price-series generator without touching Polymarket.
* Per-alpha errors are isolated. One failed pair must never break the
  whole run; failures are collected and surfaced in the status JSON.
* Concurrency is bounded to 10 by an :class:`asyncio.Semaphore` to keep
  fan-out under Polymarket's rate-limit envelope (1000 / 10s).
* The atomic write goes through a temp file in the same parent dir as
  the target so ``os.replace`` is guaranteed atomic on POSIX.

Public API
----------
- :func:`recompute_all_signals` — pure(ish) computation over ``alphas``.
- :func:`run_once` — one full read → compute → write cycle.
- :func:`run_forever` — long-running loop with cancellation support.

Routing note: this module also exposes ``router`` (an
:class:`fastapi.APIRouter`) with three endpoints under ``/signals``.
``main.py`` mounts it alongside the other feature routers.

Environment variables
---------------------
* ``PFM_LIVE_SIGNALS_ENABLED`` — set to ``"1"`` to start the loop in the
  FastAPI lifespan. Default: off (so existing tests don't see the task).
* ``PFM_LIVE_SIGNALS_INTERVAL_S`` — sleep between runs in seconds.
  Default: ``900`` (15 minutes).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from pfm.auth.dependencies import require_admin
from pfm.cache_utils import get_cache
from pfm.sources.polymarket_pool import PolymarketHTTPPool

logger = logging.getLogger(__name__)


#: Polymarket Gamma + CLOB endpoints used by the real fetcher.
GAMMA_URL: str = "https://gamma-api.polymarket.com"
CLOB_URL: str = "https://clob.polymarket.com"

#: TTL (seconds) for cached Polymarket leg fetches inside the live job.
LIVE_SIGNALS_FETCH_TTL: int = 600

#: Sample slug used by ``verify_polymarket_connectivity`` — a high-volume,
#: liquid market that should resolve year-round. Override via the
#: ``PFM_CONNECTIVITY_SAMPLE_SLUG`` env var if it ever resolves / drops.
DEFAULT_CONNECTIVITY_SAMPLE_SLUG: str = "will-bitcoin-hit-100k-by-end-of-2026"


# --- constants --------------------------------------------------------------

#: Path to the curated alpha-strategies catalog (read-only input).
DEFAULT_ALPHA_STRATEGIES_PATH: str = "web/data/alpha_strategies.json"

#: Output path consumed by the frontend α-Hub cards.
DEFAULT_LIVE_SIGNALS_PATH: str = "web/data/live_signals.json"

#: Status JSON path for ``GET /signals/status``.
DEFAULT_STATUS_PATH: str = "/tmp/pfm_live_signals_status.json"

#: Max concurrent in-flight fetches per ``recompute_all_signals`` call.
MAX_CONCURRENCY: int = 10

#: TTL for the ``GET /signals/live`` HTTP cache (matches default interval).
SIGNALS_LIVE_CACHE_TTL: int = 30

#: Synthetic-history lookback window (days) used by the default fetcher.
DEFAULT_LOOKBACK_DAYS: int = 60

#: Z-score window default when an alpha record omits ``rule_window``.
DEFAULT_RULE_WINDOW: int = 20


# Type alias for the price-history fetcher. Returns a list of floats
# (the price series for the leg, oldest -> newest). Async so production
# implementations can do http I/O without blocking the loop.
PriceFetcher = Callable[[str], Awaitable[list[float]]]

#: Async pair-aware fetcher. Takes ``(pair_id, a_id, b_id)`` and returns
#: aligned ``(a_prices, b_prices)`` as pandas Series indexed by UTC date.
#: Used by the Polymarket-backed real fetcher where the two legs must be
#: inner-joined on the same calendar dates for the spread to be valid.
PairFetcher = Callable[[str, str, str], Awaitable[tuple["pd.Series", "pd.Series"]]]

#: Selector for which fetcher backend ``run_once`` / ``run_forever`` use
#: when the caller does not pass an explicit ``fetcher`` callable.
FetcherKind = Literal["synthetic", "polymarket"]


# --- helpers ----------------------------------------------------------------


def _now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``+00:00``."""
    return datetime.now(tz=UTC).isoformat()


def _signal_from_z(
    z: float,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    prev_z: float | None = None,
) -> tuple[str, str]:
    """Map a z-score (and optional previous z) to a recommended action.

    The decision tree mirrors the one used by ``scripts/compute_live_signals.py``
    so the daily snapshot and the live job stay consistent. We also expose
    edge-triggered ``OPEN_*`` actions when ``prev_z`` is provided and the
    current bar crossed the entry threshold.
    """
    if not np.isfinite(z):
        return ("FLAT", "no signal — z-score not finite")
    abs_z = abs(z)
    if abs_z >= stop_z:
        return ("STOP_OUT", f"|z|={abs_z:.2f} ≥ stop={stop_z} — risk-off")
    # Edge-triggered open if we just crossed the entry threshold.
    if prev_z is not None and np.isfinite(prev_z):
        if z >= entry_z and prev_z < entry_z:
            return (
                "OPEN_SHORT",
                f"z={z:+.2f} crossed entry={entry_z} → short A, long β·B",
            )
        if z <= -entry_z and prev_z > -entry_z:
            return (
                "OPEN_LONG",
                f"z={z:+.2f} crossed −entry → long A, short β·B",
            )
    if z >= entry_z:
        return (
            "OPEN_SHORT",
            f"z={z:+.2f} ≥ entry={entry_z} → short A, long β·B",
        )
    if z <= -entry_z:
        return (
            "OPEN_LONG",
            f"z={z:+.2f} ≤ −entry={entry_z} → long A, short β·B",
        )
    if abs_z <= exit_z:
        return ("CLOSE", f"|z|={abs_z:.2f} ≤ exit={exit_z} — flatten if open")
    return ("HOLD", f"z={z:+.2f} ∈ (exit, entry) — hold position")


def _decay_status(z: float, n_obs: int) -> str:
    """Crude decay classifier for the live-signals snapshot.

    Without an attached returns history, the live job can only emit a
    coarse health label. Down the road we can swap this for a proper
    ``compute_rolling_sharpe`` / baseline comparison via
    :mod:`pfm.decay_monitor`.
    """
    if n_obs < 20:
        return "INSUFFICIENT_DATA"
    if not np.isfinite(z):
        return "UNKNOWN"
    if abs(z) >= 4.0:
        return "STRESSED"
    if abs(z) <= 0.25:
        return "QUIET"
    return "ACTIVE"


def _compute_signal_for_alpha(
    alpha: dict[str, Any],
    a_prices: list[float],
    b_prices: list[float],
    *,
    as_of_iso: str,
) -> dict[str, Any]:
    """Compute spread, z-score, and signal for one alpha given its leg prices.

    Pure function (no I/O, no logging). Raises ``ValueError`` on degenerate
    input so the caller's per-alpha try/except can record the failure.
    """
    a = np.asarray(a_prices, dtype=float)
    b = np.asarray(b_prices, dtype=float)
    # Align by tail: the legs may have different lengths if upstream
    # history is ragged. Trim both to the shorter common tail.
    n = int(min(a.size, b.size))
    if n < 5:
        raise ValueError(f"too few overlapping bars (n={n})")
    a = a[-n:]
    b = b[-n:]

    beta_hedge = float(alpha.get("beta_hedge", 1.0))
    spread = a - beta_hedge * b

    win = int(alpha.get("rule_window", DEFAULT_RULE_WINDOW))
    win = max(2, min(win, n))  # clamp into a usable range
    window_slice = spread[-win:]
    mu = float(window_slice.mean())
    sigma = float(window_slice.std(ddof=1))
    cur_spread = float(spread[-1])
    if sigma <= 0 or not np.isfinite(sigma):
        z: float = float("nan")
    else:
        z = (cur_spread - mu) / sigma

    # Previous z for edge-trigger detection.
    prev_z: float | None = None
    if spread.size >= win + 1:
        prev_slice = spread[-(win + 1) : -1]
        p_mu = float(prev_slice.mean())
        p_sd = float(prev_slice.std(ddof=1))
        if p_sd > 0 and np.isfinite(p_sd):
            prev_z = (float(spread[-2]) - p_mu) / p_sd

    action, reason = _signal_from_z(
        z=z,
        entry_z=float(alpha.get("rule_entry_z", 2.0)),
        exit_z=float(alpha.get("rule_exit_z", 0.5)),
        stop_z=float(alpha.get("rule_stop_z", 4.0)),
        prev_z=prev_z,
    )

    return {
        "pair_id": alpha.get("pair_id"),
        "a_id": alpha.get("a_id"),
        "b_id": alpha.get("b_id"),
        "as_of": as_of_iso,
        "n_obs": int(n),
        "beta_hedge": float(beta_hedge),
        "current_spread": cur_spread,
        "current_z": float(z) if np.isfinite(z) else None,
        "previous_z": float(prev_z) if (prev_z is not None and np.isfinite(prev_z)) else None,
        "current_a_price": float(a[-1]),
        "current_b_price": float(b[-1]),
        "action": action,
        "reason": reason,
        "mu_window": mu if np.isfinite(mu) else None,
        "sigma_window": sigma if np.isfinite(sigma) else None,
        "decay_status": _decay_status(z, n),
    }


# --- default fetcher --------------------------------------------------------


def _default_synthetic_fetcher() -> PriceFetcher:
    """Build a deterministic synthetic fetcher.

    Used when the live job is started without a real Polymarket client
    wired in (e.g. in CI, or in local demos with no network). Each
    ``factor_id`` produces a stable random walk so the resulting signals
    are reproducible run-to-run.
    """
    rng_seed_cache: dict[str, int] = {}

    async def _fetch(factor_id: str) -> list[float]:
        # Deterministic seed per factor.
        if factor_id not in rng_seed_cache:
            rng_seed_cache[factor_id] = abs(hash(factor_id)) % (2**31 - 1)
        rng = np.random.default_rng(rng_seed_cache[factor_id])
        steps = rng.normal(loc=0.0, scale=0.02, size=DEFAULT_LOOKBACK_DAYS)
        # Random walk in logit space, squashed to (0, 1) so prices look
        # like Polymarket probabilities.
        x = np.cumsum(steps)
        prices = 1.0 / (1.0 + np.exp(-x))
        return [float(p) for p in prices]

    return _fetch


# --- Polymarket-backed real fetcher -----------------------------------------


def _resolve_alpha_strategies_path() -> Path:
    """Locate ``alpha_strategies.json`` for slug lookups.

    Resolution order:
      1. ``PFM_ALPHA_STRATEGIES_PATH`` env var (absolute path).
      2. ``DEFAULT_ALPHA_STRATEGIES_PATH`` resolved relative to ``cwd``.
      3. The repo-relative fallback ``../web/data/alpha_strategies.json``
         from this module's location (works when the API is launched from
         the ``api/`` subdir).
    """
    explicit = os.environ.get("PFM_ALPHA_STRATEGIES_PATH")
    if explicit:
        return Path(explicit)
    cwd_path = Path(DEFAULT_ALPHA_STRATEGIES_PATH)
    if cwd_path.exists():
        return cwd_path
    # Fall back to repo-relative: api/src/pfm/live_signals_job.py
    # → ../../../web/data/alpha_strategies.json
    return Path(__file__).resolve().parents[3] / "web" / "data" / "alpha_strategies.json"


def _slug_lookup_from_catalog(catalog_path: Path) -> dict[str, str]:
    """Return a flat ``{a_id|b_id: slug}`` map for every alpha in the catalog.

    Reads the curated ``alpha_strategies.json`` once and indexes both
    legs of every pair. Missing ``a_slug`` / ``b_slug`` entries are
    silently skipped — callers will see a descriptive error when they
    try to resolve an id that has no slug.
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"alpha catalog not found: {catalog_path}")
    raw = json.loads(catalog_path.read_text())
    strategies = raw.get("strategies") if isinstance(raw, dict) else raw
    if not isinstance(strategies, list):
        return {}
    out: dict[str, str] = {}
    for s in strategies:
        if not isinstance(s, dict):
            continue
        a_id, b_id = s.get("a_id"), s.get("b_id")
        a_slug, b_slug = s.get("a_slug"), s.get("b_slug")
        if a_id and a_slug:
            out[str(a_id)] = str(a_slug)
        if b_id and b_slug:
            out[str(b_id)] = str(b_slug)
    return out


async def _resolve_token_id(
    slug: str, *, client: httpx.AsyncClient, gamma_url: str = GAMMA_URL
) -> str:
    """Hit Gamma ``/markets?slug=`` and return the YES ``clobTokenIds[0]``.

    Polymarket double-encodes ``clobTokenIds`` as a JSON string inside the
    JSON response, so we have to ``json.loads`` it. Raises ``LookupError``
    when the market is missing or has no usable token.
    """
    r = await client.get(f"{gamma_url.rstrip('/')}/markets", params={"slug": slug})
    if r.status_code == 404:
        raise LookupError(f"gamma market not found for slug={slug!r}")
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, list) or not body:
        raise LookupError(f"gamma returned empty body for slug={slug!r}")
    market = body[0]
    raw_ids = market.get("clobTokenIds")
    if not raw_ids:
        raise LookupError(f"market {slug!r} has no clobTokenIds")
    if isinstance(raw_ids, str):
        try:
            ids = json.loads(raw_ids)
        except json.JSONDecodeError as exc:
            raise LookupError(f"clobTokenIds for {slug!r} is not valid JSON") from exc
    elif isinstance(raw_ids, list):
        ids = raw_ids
    else:
        raise LookupError(f"unexpected clobTokenIds shape for {slug!r}")
    if not ids:
        raise LookupError(f"empty clobTokenIds for {slug!r}")
    return str(ids[0])


async def _fetch_clob_history(
    token_id: str,
    *,
    client: httpx.AsyncClient,
    clob_url: str = CLOB_URL,
    fidelity: int = 1440,
) -> pd.Series:
    """Pull a daily price-history series for one CLOB token.

    Always uses ``fidelity=1440`` — sub-daily fails for resolved markets
    (PLAN.md §5.3). Returns a ``pd.Series`` indexed by UTC date (one row
    per calendar day, last observation wins on duplicate timestamps).
    """
    r = await client.get(
        f"{clob_url.rstrip('/')}/prices-history",
        params={"market": token_id, "fidelity": fidelity},
    )
    r.raise_for_status()
    body = r.json() or {}
    history = body.get("history") if isinstance(body, dict) else None
    if not isinstance(history, list) or not history:
        return pd.Series(dtype=float)
    rows: list[tuple[pd.Timestamp, float]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        ts_raw = row.get("t")
        p_raw = row.get("p")
        if ts_raw is None or p_raw is None:
            continue
        try:
            ts = pd.Timestamp(int(ts_raw), unit="s", tz="UTC").normalize()
            p = float(p_raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(p):
            continue
        rows.append((ts, p))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows), dtype=float)
    s = s.sort_index()
    # Keep last value per day if duplicates ever appear.
    s = s[~s.index.duplicated(keep="last")]
    s.name = token_id
    return s


async def _polymarket_live_fetcher(
    pair_id: str,
    a_id: str,
    b_id: str,
    *,
    window_days: int = 60,
    client: httpx.AsyncClient | None = None,
    catalog_path: Path | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Fetch real Polymarket history for both legs of one alpha pair.

    Args:
        pair_id: Curated pair id (used only for log / cache keys).
        a_id, b_id: Factor ids whose slugs live in ``alpha_strategies.json``.
        window_days: Tail length to keep after alignment. The CLOB returns
            up to a year, but the live job only needs a recent window.
        client: Optional shared ``httpx.AsyncClient``. When ``None`` a
            short-lived one is created and closed on exit.
        catalog_path: Override for the curated catalog (tests use this).

    Returns:
        ``(a_prices, b_prices)`` — two ``pd.Series`` inner-joined on the
        same UTC daily index. The most recent ``window_days`` rows.

    Raises:
        LookupError: when either leg lacks a slug, or the upstream
            response is missing / malformed in a way that means we can't
            build a usable history.
        httpx.HTTPError: for transport-level errors.
    """
    cache = get_cache("live_signals_fetch", ttl=LIVE_SIGNALS_FETCH_TTL)
    cache_key = ("pm_live", pair_id, a_id, b_id, int(window_days))
    cached = cache.get(cache_key)
    if cached is not None:
        a_cached, b_cached = cached
        # Defensive copy so callers can't mutate the cached frame.
        return a_cached.copy(), b_cached.copy()

    cat_path = catalog_path or _resolve_alpha_strategies_path()
    slug_map = _slug_lookup_from_catalog(cat_path)
    a_slug = slug_map.get(a_id)
    b_slug = slug_map.get(b_id)
    if not a_slug:
        raise LookupError(
            f"polymarket fetcher: pair={pair_id!r} leg_a={a_id!r} has no a_slug in {cat_path}"
        )
    if not b_slug:
        raise LookupError(
            f"polymarket fetcher: pair={pair_id!r} leg_b={b_id!r} has no b_slug in {cat_path}"
        )

    # W11-11 (T18 pool migration): use shared Polymarket gamma client for
    # both _resolve_token_id (Gamma) and _fetch_clob_history (CLOB). Both
    # helpers accept absolute URLs so httpx.AsyncClient.base_url is ignored
    # at call time, letting one client serve both hosts safely.
    if client is None:
        client = PolymarketHTTPPool.instance().gamma_client
    # Resolve both token ids in parallel.
    a_token, b_token = await asyncio.gather(
        _resolve_token_id(a_slug, client=client),
        _resolve_token_id(b_slug, client=client),
    )
    # Fetch both leg histories in parallel.
    a_hist, b_hist = await asyncio.gather(
        _fetch_clob_history(a_token, client=client),
        _fetch_clob_history(b_token, client=client),
    )

    if a_hist.empty:
        raise LookupError(
            f"polymarket fetcher: pair={pair_id!r} leg_a slug={a_slug!r} returned empty history"
        )
    if b_hist.empty:
        raise LookupError(
            f"polymarket fetcher: pair={pair_id!r} leg_b slug={b_slug!r} returned empty history"
        )

    # Inner-join on UTC date; trim to the last ``window_days`` rows.
    df = pd.concat([a_hist, b_hist], axis=1, join="inner")
    df.columns = ["a", "b"]
    df = df.dropna()
    if df.empty:
        raise LookupError(f"polymarket fetcher: pair={pair_id!r} no overlapping dates between legs")
    if len(df) > window_days:
        df = df.tail(window_days)
    a_out = df["a"].copy()
    b_out = df["b"].copy()
    a_out.name = a_id
    b_out.name = b_id
    cache.set(cache_key, (a_out.copy(), b_out.copy()), ttl=LIVE_SIGNALS_FETCH_TTL)
    return a_out, b_out


def _wrap_pair_fetcher_as_price_fetcher(
    pair_fetcher: PairFetcher,
    *,
    client: httpx.AsyncClient | None = None,
) -> Callable[[dict[str, Any]], Awaitable[tuple[list[float], list[float]]]]:
    """Adapt a :data:`PairFetcher` into the shape ``recompute_all_signals`` expects.

    The inner adapter takes one alpha dict and returns ``(a_prices, b_prices)``
    as plain Python lists (so the rest of the compute path stays
    Series-agnostic). Errors propagate so the caller's per-alpha
    try/except can isolate them.
    """

    async def _per_alpha(alpha: dict[str, Any]) -> tuple[list[float], list[float]]:
        pair_id = str(alpha.get("pair_id", "<unknown>"))
        a_id = str(alpha.get("a_id"))
        b_id = str(alpha.get("b_id"))
        a_series, b_series = await pair_fetcher(pair_id, a_id, b_id)
        return [float(x) for x in a_series.tolist()], [float(x) for x in b_series.tolist()]

    # Stash the shared client (if any) on the closure so the adapter can
    # forward it; production callers want one shared httpx.AsyncClient
    # across all pairs to keep TCP / TLS reuse healthy.
    _per_alpha.shared_client = client  # type: ignore[attr-defined]
    return _per_alpha


# --- connectivity check -----------------------------------------------------


async def verify_polymarket_connectivity(
    sample_slug: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Smoke-test the Gamma + CLOB pipeline end-to-end.

    Hits Gamma to resolve ``sample_slug`` to a ``clobTokenIds[0]``, then
    pulls a daily history through the CLOB. Returns a structured result
    so the admin endpoint and operators can see exactly what failed.

    Returns:
        ``{ok: bool, sample_size: int, error: str | None,
        latency_ms: float, slug: str}``
    """
    slug = sample_slug or os.environ.get(
        "PFM_CONNECTIVITY_SAMPLE_SLUG", DEFAULT_CONNECTIVITY_SAMPLE_SLUG
    )
    t0 = time.perf_counter()
    # W11-11 (T18 pool migration): reuse shared pool client.
    if client is None:
        client = PolymarketHTTPPool.instance().gamma_client
    try:
        token = await _resolve_token_id(slug, client=client)
        history = await _fetch_clob_history(token, client=client)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        sample_size = int(history.size)
        return {
            "ok": sample_size > 0,
            "sample_size": sample_size,
            "error": None if sample_size > 0 else "empty history",
            "latency_ms": latency_ms,
            "slug": slug,
        }
    except (LookupError, httpx.HTTPError) as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "ok": False,
            "sample_size": 0,
            "error": f"{type(exc).__name__}: {exc!s}"[:240],
            "latency_ms": latency_ms,
            "slug": slug,
        }


# --- core computation -------------------------------------------------------


async def recompute_all_signals(
    alphas: list[dict[str, Any]],
    fetcher: PriceFetcher | None = None,
    *,
    max_concurrency: int = MAX_CONCURRENCY,
    pair_fetcher: PairFetcher | None = None,
) -> list[dict[str, Any]]:
    """Recompute live signals for every alpha in ``alphas``.

    Args:
        alphas: List of alpha-strategy dicts (matching the shape in
            ``alpha_strategies.json``'s ``strategies`` array). Must
            contain at least ``pair_id``, ``a_id``, ``b_id``.
        fetcher: Async callable mapping ``factor_id -> list[float]``.
            If both ``fetcher`` and ``pair_fetcher`` are ``None``, a
            deterministic synthetic fetcher is used.
        max_concurrency: Cap on simultaneous in-flight fetches.
        pair_fetcher: Optional pair-aware fetcher returning aligned
            ``(a, b)`` Series for one ``(pair_id, a_id, b_id)`` triple.
            When set, it takes precedence over ``fetcher`` and the legs
            arrive already inner-joined on UTC dates — the right shape
            for the Polymarket-backed real path.

    Returns:
        One entry per input alpha. Successful entries carry the full
        signal payload; failed entries carry ``error`` and ``pair_id``
        so callers can surface them without a separate failure list.
    """
    if pair_fetcher is None and fetcher is None:
        fetcher = _default_synthetic_fetcher()

    sem = asyncio.Semaphore(max_concurrency)
    as_of = _now_utc_iso()

    async def _process_one(alpha: dict[str, Any]) -> dict[str, Any]:
        pair_id = alpha.get("pair_id", "<unknown>")
        a_id = alpha.get("a_id")
        b_id = alpha.get("b_id")
        base = {
            "pair_id": pair_id,
            "a_id": a_id,
            "b_id": b_id,
            "as_of": as_of,
        }
        if not a_id or not b_id:
            return {**base, "error": "missing a_id or b_id"}
        async with sem:
            try:
                if pair_fetcher is not None:
                    a_series, b_series = await pair_fetcher(str(pair_id), str(a_id), str(b_id))
                    a_prices = [float(x) for x in a_series.tolist()]
                    b_prices = [float(x) for x in b_series.tolist()]
                else:
                    assert fetcher is not None  # narrow for type checker
                    a_prices, b_prices = await asyncio.gather(fetcher(a_id), fetcher(b_id))
            except Exception as exc:
                logger.warning("live_signals: fetch failed pair=%s err=%s", pair_id, exc)
                return {**base, "error": f"fetch failed: {exc!s}"[:240]}
        try:
            return _compute_signal_for_alpha(alpha, a_prices, b_prices, as_of_iso=as_of)
        except Exception as exc:
            logger.warning("live_signals: compute failed pair=%s err=%s", pair_id, exc)
            return {**base, "error": f"compute failed: {exc!s}"[:240]}

    results = await asyncio.gather(*(_process_one(a) for a in alphas))
    return list(results)


def _build_pair_fetcher_for_kind(
    kind: FetcherKind,
    *,
    client: httpx.AsyncClient | None = None,
) -> PairFetcher | None:
    """Materialise a :data:`PairFetcher` for the requested backend.

    Returns ``None`` for ``"synthetic"`` (the legacy per-leg fetcher
    path stays in charge there). Returns a closure over a shared
    ``httpx.AsyncClient`` for ``"polymarket"``.
    """
    if kind == "synthetic":
        return None
    if kind == "polymarket":

        async def _fetch(pair_id: str, a_id: str, b_id: str) -> tuple[pd.Series, pd.Series]:
            return await _polymarket_live_fetcher(pair_id, a_id, b_id, client=client)

        return _fetch
    raise ValueError(f"unknown fetcher_kind: {kind!r}")


# --- atomic file write ------------------------------------------------------


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically (temp file + ``os.replace``).

    The temp file lives in the same parent directory so the rename is a
    single atomic syscall on POSIX. Concurrent readers either see the
    old file or the new one — never a partial write.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(target)


def _load_alphas(strategies_path: str | Path) -> list[dict[str, Any]]:
    """Load and return the ``strategies`` array from ``alpha_strategies.json``."""
    p = Path(strategies_path)
    if not p.exists():
        raise FileNotFoundError(f"alpha strategies file not found: {p}")
    raw = json.loads(p.read_text())
    if isinstance(raw, dict) and "strategies" in raw:
        return list(raw["strategies"])
    if isinstance(raw, list):
        return list(raw)
    raise ValueError(f"unexpected shape in {p}: missing 'strategies' array")


# --- run_once / run_forever -------------------------------------------------


async def run_once(
    write_path: str | Path = DEFAULT_LIVE_SIGNALS_PATH,
    *,
    strategies_path: str | Path = DEFAULT_ALPHA_STRATEGIES_PATH,
    status_path: str | Path = DEFAULT_STATUS_PATH,
    fetcher: PriceFetcher | None = None,
    pair_fetcher: PairFetcher | None = None,
    fetcher_kind: FetcherKind = "synthetic",
    http_client: httpx.AsyncClient | None = None,
    max_concurrency: int = MAX_CONCURRENCY,
) -> dict[str, Any]:
    """Execute one full read → compute → write cycle.

    Args:
        write_path: Where to write the public ``live_signals.json``.
        strategies_path: Source-of-truth alpha catalog.
        status_path: Where to persist the run-status JSON.
        fetcher: Per-leg synthetic fetcher (legacy / tests).
        pair_fetcher: Pair-aware fetcher (overrides ``fetcher_kind``).
        fetcher_kind: Selector used when neither ``fetcher`` nor
            ``pair_fetcher`` is provided. ``"synthetic"`` (default)
            keeps the deterministic walk that backs CI; ``"polymarket"``
            wires the real Gamma + CLOB pipeline.
        http_client: Optional shared ``httpx.AsyncClient`` for the
            Polymarket path. When ``None`` and the polymarket backend
            is selected, ``run_once`` builds a short-lived one for the
            cycle and closes it on exit.
        max_concurrency: Bound on in-flight fetches.

    Returns a status dict that's also persisted to ``status_path`` for
    the ``GET /signals/status`` endpoint.
    """
    t0 = time.perf_counter()
    alphas = _load_alphas(strategies_path)

    # W11-11 (T18 pool migration): reuse shared pool client when no
    # http_client is injected. Pool is never closed.
    effective_pair_fetcher = pair_fetcher
    if effective_pair_fetcher is None and fetcher is None and fetcher_kind == "polymarket":
        client_for_kind: httpx.AsyncClient | None = (
            http_client if http_client is not None else PolymarketHTTPPool.instance().gamma_client
        )
        effective_pair_fetcher = _build_pair_fetcher_for_kind("polymarket", client=client_for_kind)

    results = await recompute_all_signals(
        alphas,
        fetcher=fetcher,
        pair_fetcher=effective_pair_fetcher,
        max_concurrency=max_concurrency,
    )

    n_total = len(results)
    n_failed = sum(1 for r in results if "error" in r)
    n_updated = n_total - n_failed
    n_actionable = sum(1 for r in results if r.get("action") in {"OPEN_LONG", "OPEN_SHORT"})

    as_of = _now_utc_iso()
    duration = time.perf_counter() - t0

    output = {
        "as_of": as_of,
        "n_strategies": n_total,
        "n_actionable": n_actionable,
        "n_errors": n_failed,
        "duration_seconds": round(duration, 3),
        "signals": {r["pair_id"]: r for r in results if r.get("pair_id")},
    }
    _atomic_write_json(write_path, output)

    failures = [
        {"pair_id": r.get("pair_id"), "error": r.get("error")} for r in results if "error" in r
    ]
    status = {
        "last_run_iso": as_of,
        "last_duration_seconds": round(duration, 3),
        "n_alphas_total": n_total,
        "n_alphas_updated": n_updated,
        "n_alphas_failed": n_failed,
        "n_alphas_actionable": n_actionable,
        "failures": failures,
        "live_signals_path": str(write_path),
    }
    try:
        _atomic_write_json(status_path, status)
    except OSError as exc:
        # Status file is advisory; never fail a run because we can't
        # write it.
        logger.warning("live_signals: failed to write status file: %s", exc)

    # Drop any stale GET /signals/live cache entries — the file just
    # changed, so previously cached body bytes are now wrong.
    with contextlib.suppress(Exception):
        get_cache("live_signals", ttl=SIGNALS_LIVE_CACHE_TTL).clear()

    logger.info(
        "live_signals: run_once total=%d updated=%d failed=%d actionable=%d duration=%.2fs",
        n_total,
        n_updated,
        n_failed,
        n_actionable,
        duration,
    )
    return status


async def run_forever(
    interval_seconds: int = 900,
    *,
    write_path: str | Path = DEFAULT_LIVE_SIGNALS_PATH,
    strategies_path: str | Path = DEFAULT_ALPHA_STRATEGIES_PATH,
    status_path: str | Path = DEFAULT_STATUS_PATH,
    fetcher: PriceFetcher | None = None,
    pair_fetcher: PairFetcher | None = None,
    fetcher_kind: FetcherKind = "synthetic",
    http_client: httpx.AsyncClient | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run :func:`run_once` repeatedly, sleeping ``interval_seconds`` between runs.

    Cancellation: callers can either ``cancel()`` the surrounding task
    or set ``stop_event``. Both paths exit the loop cleanly without
    losing an in-flight write (``run_once`` always finishes the current
    cycle before we check the next sleep).

    The fetcher selection (``fetcher`` / ``pair_fetcher`` / ``fetcher_kind``)
    mirrors :func:`run_once`. When the polymarket backend is selected
    and no ``http_client`` is passed in, each cycle creates and closes
    its own short-lived client.
    """
    interval = max(60, int(interval_seconds))  # never spin faster than 1 / min
    while True:
        try:
            await run_once(
                write_path=write_path,
                strategies_path=strategies_path,
                status_path=status_path,
                fetcher=fetcher,
                pair_fetcher=pair_fetcher,
                fetcher_kind=fetcher_kind,
                http_client=http_client,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("live_signals: run_once raised: %s", exc)

        if stop_event is not None and stop_event.is_set():
            return
        try:
            if stop_event is None:
                await asyncio.sleep(interval)
            else:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                return  # event fired during the sleep
        except TimeoutError:
            # Normal path: interval elapsed, no stop signal yet.
            continue
        except asyncio.CancelledError:
            raise


# --- pydantic models for the router -----------------------------------------


class RecomputeResponse(BaseModel):
    last_run_iso: str = Field(..., description="UTC ISO-8601 timestamp of the run.")
    last_duration_seconds: float
    n_alphas_total: int
    n_alphas_updated: int
    n_alphas_failed: int
    n_alphas_actionable: int
    failures: list[dict[str, Any]] = Field(default_factory=list)
    live_signals_path: str


class ConnectivityCheckResponse(BaseModel):
    ok: bool = Field(..., description="True iff the sample fetch succeeded.")
    sample_size: int = Field(..., description="Number of daily bars returned for the sample slug.")
    error: str | None = Field(
        default=None, description="Error class + message when ``ok`` is False."
    )
    latency_ms: float = Field(..., description="End-to-end Gamma + CLOB round-trip latency.")
    slug: str = Field(..., description="The slug used for the connectivity probe.")


class StatusResponse(BaseModel):
    last_run_iso: str | None = None
    last_duration_seconds: float | None = None
    n_alphas_total: int | None = None
    n_alphas_updated: int | None = None
    n_alphas_failed: int | None = None
    n_alphas_actionable: int | None = None
    failures: list[dict[str, Any]] = Field(default_factory=list)
    live_signals_path: str | None = None
    next_run_at_estimate: str | None = Field(
        default=None,
        description="Best-effort next-run timestamp (last_run + interval).",
    )


# --- router -----------------------------------------------------------------


router = APIRouter(prefix="/signals", tags=["live-signals"])


def _admin_dep_if_enabled() -> Any:
    """Return the admin dep when admin auth is configured, else a noop.

    ``POST /signals/recompute-now`` is admin-only when ``PFM_ADMIN_TOKEN``
    is set so we don't expose an expensive trigger to anonymous callers.
    When admin auth is disabled (no env var) we keep the endpoint open
    so local dev / demos don't need to set up a token first.
    """
    if os.environ.get("PFM_ADMIN_TOKEN"):
        return Depends(require_admin)

    async def _noop() -> None:
        return None

    return Depends(_noop)


@router.post(
    "/recompute-now",
    response_model=RecomputeResponse,
    summary="Trigger one live-signals recompute synchronously.",
    dependencies=[_admin_dep_if_enabled()],
)
async def signals_recompute_now() -> RecomputeResponse:
    try:
        status = await run_once()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"alpha catalog missing: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"recompute failed: {exc}") from exc
    return RecomputeResponse(**status)


@router.get(
    "/connectivity-check",
    response_model=ConnectivityCheckResponse,
    summary="Probe Polymarket Gamma + CLOB end-to-end with a sample slug.",
    dependencies=[_admin_dep_if_enabled()],
)
async def signals_connectivity_check(
    slug: str | None = None,
) -> ConnectivityCheckResponse:
    """Verify the real fetcher path can reach Polymarket.

    Returns ``ok=True`` only when Gamma resolves the sample slug to a
    ``clobTokenId`` *and* the CLOB ``/prices-history`` call returns at
    least one daily bar. Useful as a pre-flight before flipping
    ``PFM_LIVE_SIGNALS_FETCHER=polymarket`` in production.
    """
    result = await verify_polymarket_connectivity(sample_slug=slug)
    return ConnectivityCheckResponse(**result)


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Last live-signals run status (cron health).",
)
async def signals_status() -> StatusResponse:
    def _read() -> dict[str, Any] | None:
        p = Path(DEFAULT_STATUS_PATH)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    try:
        raw = await asyncio.to_thread(_read)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"status file unreadable: {exc}") from exc
    if raw is None:
        return StatusResponse()
    # Best-effort next-run estimate.
    interval = int(os.environ.get("PFM_LIVE_SIGNALS_INTERVAL_S", "900"))
    last_iso = raw.get("last_run_iso")
    next_iso: str | None = None
    if last_iso:
        try:
            last_dt = datetime.fromisoformat(last_iso)
            next_dt = last_dt.timestamp() + interval
            next_iso = datetime.fromtimestamp(next_dt, tz=UTC).isoformat()
        except ValueError:
            next_iso = None
    raw["next_run_at_estimate"] = next_iso
    return StatusResponse(**raw)


@router.get(
    "/live",
    summary="Return the current live_signals.json contents (cached 30s).",
)
async def signals_live() -> dict[str, Any]:
    cache = get_cache("live_signals", ttl=SIGNALS_LIVE_CACHE_TTL)
    cached = cache.get("payload")
    if cached is not None:
        return cached

    def _read() -> tuple[bool, dict[str, Any] | None]:
        p = Path(DEFAULT_LIVE_SIGNALS_PATH)
        if not p.exists():
            return (False, None)
        return (True, json.loads(p.read_text()))

    try:
        exists, payload = await asyncio.to_thread(_read)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"live_signals.json unreadable: {exc}") from exc
    if not exists or payload is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"live_signals.json not found at {DEFAULT_LIVE_SIGNALS_PATH}. "
                "Has the job run? "
                "Set PFM_LIVE_SIGNALS_ENABLED=1 or POST /signals/recompute-now."
            ),
        )
    cache.set("payload", payload, ttl=SIGNALS_LIVE_CACHE_TTL)
    return payload


__all__ = [
    "DEFAULT_ALPHA_STRATEGIES_PATH",
    "DEFAULT_LIVE_SIGNALS_PATH",
    "DEFAULT_STATUS_PATH",
    "MAX_CONCURRENCY",
    "FetcherKind",
    "PairFetcher",
    "PriceFetcher",
    "_atomic_write_json",
    "_compute_signal_for_alpha",
    "_polymarket_live_fetcher",
    "_signal_from_z",
    "recompute_all_signals",
    "router",
    "run_forever",
    "run_once",
    "verify_polymarket_connectivity",
]
