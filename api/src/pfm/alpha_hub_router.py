"""FastAPI router for the curated Alpha Hub leaderboard / live-panel views.

Endpoints
---------
* ``GET /alpha-hub/leaderboard`` — paginated, filtered, sortable view over
  ``web/data/alpha_strategies.json``. Returns a slim subset of fields per
  strategy for fast rendering in the discovery panel.
* ``GET /alpha-hub/strategy/{pair_id}`` — full detail for a single strategy.
* ``GET /alpha-hub/live-panel`` — composite payload combining the top three
  production-tier strategies, the watchlist (``B_VALIDATED``) and the most
  recent graveyard entries.

Design notes
------------
The strategies catalog rarely changes (regenerated with each research wave),
so all reads are fronted by a 5-minute TTL :class:`pfm.cache_utils.TerminalCache`
instance. File IO is small (~1MB) but happening on every request would still
be wasteful when the discovery UI polls.

Backward compatibility: the existing ``alpha_graveyard_router`` keeps its
``/alpha-hub/graveyard*`` paths. This router uses the same ``/alpha-hub``
prefix but exposes disjoint paths.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pfm.alpha_graveyard import load_graveyard
from pfm.cache_utils import cached, get_cache
from pfm.live_signals_job import (
    _compute_signal_for_alpha,
    _polymarket_live_fetcher,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution + loader
# ---------------------------------------------------------------------------

#: Absolute path to ``web/data/alpha_strategies.json`` relative to this file.
ALPHA_STRATEGIES_PATH: Path = (
    Path(__file__).resolve().parents[3] / "web" / "data" / "alpha_strategies.json"
)

#: Absolute path to ``web/data/live_signals.json`` relative to this file.
#: Optional — handlers must tolerate missing file (return ``recent_signal=None``).
LIVE_SIGNALS_PATH: Path = Path(__file__).resolve().parents[3] / "web" / "data" / "live_signals.json"

#: Cache namespace + TTL (5 min). The catalog rebuilds with each wave.
_CACHE_NS = "alpha_hub_leaderboard"
_CACHE_TTL_SECONDS = 5 * 60


def _load_strategies(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and return the ``strategies`` array from ``alpha_strategies.json``.

    Missing or malformed files surface as an HTTP 500 to callers via the
    router (this helper itself raises plain exceptions so it stays usable
    from synchronous tests).
    """
    p = path if path is not None else ALPHA_STRATEGIES_PATH
    if not p.exists():
        raise FileNotFoundError(f"alpha_strategies.json not found at {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("alpha_strategies.json must be a top-level object")
    strategies = raw.get("strategies", [])
    if not isinstance(strategies, list):
        raise ValueError("alpha_strategies.json[strategies] must be an array")
    return strategies


def _load_catalog(path: Path | None = None) -> dict[str, Any]:
    """Load and return the full top-level catalog dict from ``alpha_strategies.json``.

    Includes summary counts (``n_curated``, ``n_factors_in_catalog``, …) and
    the ``strategies`` list. Used by the ``full=true`` leaderboard view so
    the frontend can populate hero stats AND the discovery grid from a
    single round-trip — eliminating the previous dual-source bug where
    the UI fetched the static JSON directly while ``/alpha-hub/leaderboard``
    returned a slim projection.
    """
    p = path if path is not None else ALPHA_STRATEGIES_PATH
    if not p.exists():
        raise FileNotFoundError(f"alpha_strategies.json not found at {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("alpha_strategies.json must be a top-level object")
    if not isinstance(raw.get("strategies", []), list):
        raise ValueError("alpha_strategies.json[strategies] must be an array")
    return raw


def _cached_strategies() -> list[dict[str, Any]]:
    """Return the strategies array, fronted by the 5-minute cache."""
    cache = get_cache(_CACHE_NS, ttl=_CACHE_TTL_SECONDS)
    return cache.get_or_compute("strategies", _load_strategies, ttl=_CACHE_TTL_SECONDS)


def _cached_catalog() -> dict[str, Any]:
    """Return the full catalog dict, fronted by the 5-minute cache."""
    cache = get_cache(_CACHE_NS, ttl=_CACHE_TTL_SECONDS)
    return cache.get_or_compute("catalog", _load_catalog, ttl=_CACHE_TTL_SECONDS)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

#: Closed-set tier filter — extra ``"all"`` lets callers skip filtering.
TierFilter = Literal[
    "all", "A_STRUCTURAL", "A_GOLD", "B_VALIDATED", "B_FDR_ONLY", "C_TENTATIVE", "D_RAW"
]

#: Sort key — exposed as a closed set so we don't surface arbitrary attribute
#: lookups against the JSON to public callers.
SortKey = Literal[
    "oos_sharpe",
    "full_sharpe",
    "half_life_days",
    "n_obs",
    "suggested_allocation",
    "perm_p",
    "adf_pvalue",
]

SortOrder = Literal["asc", "desc"]


class LeaderboardItem(BaseModel):
    """Slim per-strategy view used by the leaderboard grid."""

    pair_id: str
    tier: str
    theme_a: str | None = None
    theme_b: str | None = None
    category: str | None = None
    oos_sharpe: float | None = None
    full_sharpe: float | None = None
    max_dd: float | None = Field(
        default=None, description="Worst observed drawdown (negative number)"
    )
    half_life_days: float | None = None
    beta_hedge: float | None = None
    n_obs: int | None = None
    suggested_allocation: float | None = None
    risk_grade: str | None = None
    fdr_status: str | None = None
    bootstrap_robust: bool | None = None


class LeaderboardResponse(BaseModel):
    """Wrapper returned by ``GET /alpha-hub/leaderboard`` (slim default mode).

    The ``full=true`` query param switches the handler to a richer payload
    that's NOT modelled here (returned via :class:`JSONResponse` so every
    raw-catalog field is preserved verbatim — ``data_quality_warning``,
    ``a_name``/``b_name``, ``sharpe_ci_lo``, ``rationale``, etc.). That
    response also carries a ``meta`` block with top-level summary counts.
    """

    total: int = Field(..., ge=0, description="Filtered row count before pagination")
    n_returned: int = Field(..., ge=0)
    offset: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    sort: SortKey = "oos_sharpe"
    order: SortOrder = "desc"
    items: list[LeaderboardItem]


class LivePanelResponse(BaseModel):
    """Composite payload for the alpha-hub landing card."""

    production: list[LeaderboardItem]
    watchlist: list[LeaderboardItem]
    graveyard: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_item(raw: dict[str, Any]) -> LeaderboardItem:
    """Project a raw strategy dict to the slim :class:`LeaderboardItem`."""
    return LeaderboardItem(
        pair_id=str(raw.get("pair_id", "")),
        tier=str(raw.get("tier", "")),
        theme_a=raw.get("theme_a"),
        theme_b=raw.get("theme_b"),
        category=raw.get("category"),
        oos_sharpe=raw.get("oos_sharpe"),
        full_sharpe=raw.get("full_sharpe"),
        max_dd=raw.get("worst_drawdown_observed"),
        half_life_days=raw.get("half_life_days"),
        beta_hedge=raw.get("beta_hedge"),
        n_obs=raw.get("n_obs"),
        suggested_allocation=raw.get("suggested_allocation"),
        risk_grade=raw.get("risk_grade"),
        fdr_status=raw.get("fdr_status"),
        bootstrap_robust=raw.get("bootstrap_robust"),
    )


def _filter_strategies(
    strategies: list[dict[str, Any]],
    *,
    tier: TierFilter,
    theme: str | None,
    min_sharpe: float | None,
) -> list[dict[str, Any]]:
    """Apply tier / theme / min-sharpe filters."""
    out = strategies
    if tier != "all":
        out = [s for s in out if s.get("tier") == tier]
    if theme is not None and theme != "":
        t = theme.lower()
        out = [
            s
            for s in out
            if str(s.get("theme_a", "")).lower() == t or str(s.get("theme_b", "")).lower() == t
        ]
    if min_sharpe is not None:
        out = [s for s in out if (s.get("oos_sharpe") or 0.0) >= min_sharpe]
    return out


def _sort_strategies(
    strategies: list[dict[str, Any]],
    *,
    sort: SortKey,
    order: SortOrder,
) -> list[dict[str, Any]]:
    """Stable sort by ``sort`` (numeric / None-tolerant)."""
    reverse = order == "desc"

    def _key(s: dict[str, Any]) -> tuple[int, float]:
        v = s.get(sort)
        if v is None:
            # None goes last on desc, first on asc — common-sense default.
            return (1, 0.0) if reverse else (0, 0.0)
        try:
            return (0, float(v)) if reverse else (1, float(v))
        except (TypeError, ValueError):
            return (1, 0.0) if reverse else (0, 0.0)

    # Use a tuple key to keep None separate from numeric range.
    def _sort_key(s: dict[str, Any]) -> float:
        v = s.get(sort)
        try:
            return float(v) if v is not None else float("-inf") if reverse else float("inf")
        except (TypeError, ValueError):
            return float("-inf") if reverse else float("inf")

    return sorted(strategies, key=_sort_key, reverse=reverse)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/alpha-hub", tags=["alpha-hub"])


#: Top-level fields lifted from ``alpha_strategies.json`` into the
#: ``meta`` block when the route is called with ``full=true``. These are
#: the exact keys the frontend's hero strip + meta line consume.
_CATALOG_META_KEYS: tuple[str, ...] = (
    "generated",
    "pipeline_version",
    "n_factors_in_catalog",
    "n_pairs_evaluated",
    "n_real_alpha_raw",
    "n_fdr_bh_q05",
    "n_fdr_bh_q10",
    "n_bootstrap_robust",
    "n_strike_family_pairs",
    "n_curated",
    "lookback_start",
    "lookback_end",
    "updated",
)


@cached(namespace="alpha_hub_leaderboard_response", ttl=_CACHE_TTL_SECONDS)
def _build_leaderboard_response(
    tier: TierFilter,
    theme: str | None,
    min_sharpe: float | None,
    sort: SortKey,
    order: SortOrder,
    limit: int,
    offset: int,
) -> LeaderboardResponse:
    """Filter, sort, paginate, and project the cached strategies array.

    Wrapped in :func:`pfm.cache_utils.cached` so identical query-param
    combinations short-circuit the filter/sort/projection pipeline. The
    inner function takes only hashable primitives so the decorator's
    default ``(args, kwargs)`` key builder is safe — no FastAPI
    ``Request`` object is ever in the key.
    """
    strategies = _cached_strategies()
    filtered = _filter_strategies(strategies, tier=tier, theme=theme, min_sharpe=min_sharpe)
    ordered = _sort_strategies(filtered, sort=sort, order=order)
    page = ordered[offset : offset + limit]
    items = [_to_item(s) for s in page]
    return LeaderboardResponse(
        total=len(filtered),
        n_returned=len(items),
        offset=offset,
        limit=limit,
        sort=sort,
        order=order,
        items=items,
    )


@cached(namespace="alpha_hub_leaderboard_full_response", ttl=_CACHE_TTL_SECONDS)
def _build_leaderboard_full_response(
    tier: TierFilter,
    theme: str | None,
    min_sharpe: float | None,
    sort: SortKey,
    order: SortOrder,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Return the full-fidelity leaderboard payload (raw dicts + meta block).

    Used by ``GET /alpha-hub/leaderboard?full=true``. The frontend reads
    this endpoint instead of fetching ``web/data/alpha_strategies.json``
    directly, so the API is the single source of truth and every
    renderable strategy can be looked up by ``pair_id`` afterwards.

    Preserves every field on each strategy (no projection) including
    ``data_quality_warning`` flags on sanitized rows. Cached for 5 minutes
    on the same TTL as the slim view.
    """
    catalog = _cached_catalog()
    strategies = catalog.get("strategies", [])
    if not isinstance(strategies, list):
        strategies = []
    filtered = _filter_strategies(strategies, tier=tier, theme=theme, min_sharpe=min_sharpe)
    ordered = _sort_strategies(filtered, sort=sort, order=order)
    page = ordered[offset : offset + limit]
    meta = {k: catalog.get(k) for k in _CATALOG_META_KEYS if k in catalog}
    return {
        "total": len(filtered),
        "n_returned": len(page),
        "offset": offset,
        "limit": limit,
        "sort": sort,
        "order": order,
        "items": list(page),
        "meta": meta,
    }


@router.get(
    "/leaderboard",
    response_model=LeaderboardResponse,
    summary="Paginated, filtered, sortable view of curated alpha strategies.",
)
def get_leaderboard(
    tier: Annotated[TierFilter, Query(description="Tier filter; 'all' disables.")] = "all",
    theme: Annotated[
        str | None, Query(description="Match theme_a or theme_b (case-insensitive).")
    ] = None,
    min_sharpe: Annotated[
        float | None, Query(description="Drop rows where oos_sharpe < min_sharpe.")
    ] = None,
    sort: Annotated[SortKey, Query(description="Sort key.")] = "oos_sharpe",
    order: Annotated[SortOrder, Query(description="Sort order.")] = "desc",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    full: Annotated[
        bool,
        Query(
            description=(
                "When true, items contain the raw catalog dicts (every "
                "field preserved) and a ``meta`` block of top-level summary "
                "counts is included. Used by the frontend so the API is the "
                "single source of truth for the discovery panel."
            ),
        ),
    ] = False,
) -> Any:
    """Return a paginated leaderboard slice for the discovery UI.

    Default (``full=false``) returns the slim :class:`LeaderboardResponse`
    with one :class:`LeaderboardItem` per row. ``full=true`` returns the
    raw catalog dicts plus a ``meta`` block (same envelope shape) — used
    by the alpha-hub frontend to avoid double-sourcing the static JSON.
    """
    try:
        if full:
            return JSONResponse(
                content=_build_leaderboard_full_response(
                    tier=tier,
                    theme=theme,
                    min_sharpe=min_sharpe,
                    sort=sort,
                    order=order,
                    limit=limit,
                    offset=offset,
                )
            )
        return _build_leaderboard_response(
            tier=tier,
            theme=theme,
            min_sharpe=min_sharpe,
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


#: Number of synthetic equity-curve points to embed in strategy-detail responses.
_EQUITY_CURVE_POINTS: int = 30


def _synthetic_equity_curve(
    oos_sharpe: float | None, n_points: int = _EQUITY_CURVE_POINTS
) -> list[dict[str, Any]]:
    """Build a synthetic linear-ramp equity curve for first-paint UX.

    The shape — not the magnitude — is what the frontend uses to render
    a sparkline before the real OOS backtest comes back. Returns a list
    of ``{date, equity}`` dicts of length ``n_points``, starting at
    ``1.0`` and ending at ``1 + (oos_sharpe * 0.01 * n_points)``.

    A negative or missing ``oos_sharpe`` produces a flat (or descending)
    line, which still conveys "we have data, it's not great" without a
    second round-trip to ``/terminal/backtest``.
    """
    slope_per_day = (oos_sharpe or 0.0) * 0.01
    today = date.today()
    start = today - timedelta(days=n_points - 1)
    return [
        {
            "date": (start + timedelta(days=i)).isoformat(),
            "equity": round(1.0 + slope_per_day * i, 6),
        }
        for i in range(n_points)
    ]


#: Number of synthetic spread-series points (~90 days = ~one quarter).
_SPREAD_SERIES_POINTS: int = 90


def _synthetic_spread_series(
    pair_id: str,
    *,
    hedge_ratio: float = 1.0,
    n_points: int = _SPREAD_SERIES_POINTS,
) -> list[dict[str, Any]]:
    """Build a deterministic Ornstein-Uhlenbeck-style synthetic spread series.

    Seeded off ``pair_id`` so refreshes return identical curves. Each
    point shape: ``{date, p_a, p_b, spread, z_score}``. The OU process
    mean-reverts toward zero so the spread visualisation looks plausible
    for a cointegrated pair.

    Used until per-pair real spread histories are wired in. Cheap
    (numpy, deterministic) and never raises.
    """
    seed = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    n = n_points
    theta = 0.15  # mean-reversion speed
    sigma = 0.30
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = z[i - 1] * (1 - theta) + rng.normal(0, sigma)
    # Anchor p_a/p_b around 0.4 / 0.5 so the spread is visually interesting.
    p_a = np.clip(0.4 + 0.05 * z, 0.05, 0.95)
    p_b = np.clip(0.5 - 0.05 * z, 0.05, 0.95)
    spread = p_a - hedge_ratio * p_b
    end_d = date.today()
    dates = [(end_d - timedelta(days=n - 1 - i)).isoformat() for i in range(n)]
    return [
        {
            "date": dates[i],
            "p_a": float(p_a[i]),
            "p_b": float(p_b[i]),
            "spread": float(spread[i]),
            "z_score": float(z[i]),
        }
        for i in range(n)
    ]


def _load_live_signals(path: Path | None = None) -> dict[str, Any]:
    """Return the ``signals`` dict keyed by ``pair_id`` from ``live_signals.json``.

    Tolerant of: missing file, malformed JSON, or a top-level shape that
    isn't a dict. In any of those cases returns an empty dict so callers
    can safely ``.get(pair_id)`` and surface ``None``.
    """
    p = path if path is not None else LIVE_SIGNALS_PATH
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    signals = raw.get("signals")
    if isinstance(signals, dict):
        return signals
    if isinstance(signals, list):
        # If a future revision flattens to a list, key by pair_id.
        return {
            str(item.get("pair_id")): item
            for item in signals
            if isinstance(item, dict) and item.get("pair_id")
        }
    return {}


def _cached_live_signals() -> dict[str, Any]:
    """Return the live-signals mapping, fronted by a 5-minute cache."""
    cache = get_cache("alpha_hub_live_signals", ttl=_CACHE_TTL_SECONDS)
    return cache.get_or_compute("signals", _load_live_signals, ttl=_CACHE_TTL_SECONDS)


def _load_catalog_updated_iso() -> str | None:
    """Read just the top-level ``updated`` field; used as fallback for ``updated_at``."""
    try:
        top = json.loads(ALPHA_STRATEGIES_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return top.get("updated") if isinstance(top, dict) else None


def _find_strategy_src(pair_id: str) -> dict[str, Any]:
    """Lookup ``pair_id`` in the cached strategies list; raise KeyError if absent."""
    for s in _cached_strategies():
        if s.get("pair_id") == pair_id:
            return s
    raise KeyError(pair_id)


def _build_recent_signal(pair_id: str) -> Any:
    """Return live-signal dict for ``pair_id`` or ``None``; never raises."""
    try:
        return _cached_live_signals().get(pair_id)
    except Exception:
        return None


def _build_spread(pair_id: str, hedge: Any) -> list[dict[str, Any]]:
    """Compute the ~90-point OU spread series for ``pair_id``."""
    try:
        hedge_ratio = float(hedge) if hedge is not None else 1.0
    except (TypeError, ValueError):
        hedge_ratio = 1.0
    return _synthetic_spread_series(pair_id, hedge_ratio=hedge_ratio)


async def _build_strategy_detail_async(pair_id: str) -> dict[str, Any]:
    """Async variant of :func:`_build_strategy_detail` — fans out the
    independent units (spread series, live-signal lookup, catalog-updated
    fallback) via :func:`asyncio.gather` so wall-clock = ``max(t_i)`` rather
    than ``sum(t_i)``. Each unit returns ``None`` (or its safe default) on
    failure; partial degradation is preferred over a 500.
    """
    src = _find_strategy_src(pair_id)
    hedge = src.get("beta_hedge")
    needs_updated_fallback = not src.get("last_updated_iso")

    spread_t = asyncio.to_thread(_build_spread, pair_id, hedge)
    signal_t = asyncio.to_thread(_build_recent_signal, pair_id)
    updated_t = (
        asyncio.to_thread(_load_catalog_updated_iso)
        if needs_updated_fallback
        else asyncio.sleep(0, result=src.get("last_updated_iso"))
    )
    spread, signal, updated = await asyncio.gather(
        spread_t, signal_t, updated_t, return_exceptions=True
    )
    if isinstance(spread, BaseException):
        _logger.warning("alpha_hub spread_series failed for %s: %r", pair_id, spread)
        spread = []
    if isinstance(signal, BaseException):
        _logger.warning("alpha_hub recent_signal failed for %s: %r", pair_id, signal)
        signal = None
    if isinstance(updated, BaseException):
        _logger.warning("alpha_hub updated fallback failed for %s: %r", pair_id, updated)
        updated = None

    payload: dict[str, Any] = dict(src)
    existing = src.get("equity_curve")
    if isinstance(existing, list) and len(existing) >= 20:
        payload["equity_curve"] = existing
        payload["equity_curve_is_synthetic"] = False
    else:
        payload["equity_curve"] = _synthetic_equity_curve(src.get("oos_sharpe"))
        payload["equity_curve_is_synthetic"] = True
    payload["spread_series"] = spread
    # Spread is always synthetic via _synthetic_spread_series — flag it so
    # the UI can render a "preview" badge instead of marketing as live.
    payload["spread_series_is_synthetic"] = True
    rule_keys = ("rule_window", "rule_entry_z", "rule_exit_z", "rule_stop_z")
    payload["rule"] = (
        {
            "window": src.get("rule_window"),
            "entry_z": src.get("rule_entry_z"),
            "exit_z": src.get("rule_exit_z"),
            "stop_z": src.get("rule_stop_z"),
        }
        if any(src.get(k) is not None for k in rule_keys)
        else None
    )
    payload["risk"] = {
        "grade": src.get("risk_grade"),
        "max_dd": src.get("max_dd") or src.get("worst_drawdown_observed"),
        "best_conditions": src.get("best_market_conditions"),
        "worst_conditions": src.get("worst_market_conditions"),
    }
    deployment_fields = (
        "min_capital_usd",
        "expected_holding_days",
        "expected_trades_per_year",
        "monitoring_frequency",
        "deploy_signal_logic",
        "kill_switch_rules",
    )
    deployment = {k: src[k] for k in deployment_fields if src.get(k) is not None}
    payload["deployment"] = deployment if deployment else None
    payload["recent_signal"] = signal
    payload["rationale"] = src.get("rationale")
    payload["theory_reference"] = src.get("theory_reference")
    payload["correlated_with"] = src.get("correlated_with_strategies")
    payload["updated_at"] = updated
    return payload


@router.get(
    "/strategy/{pair_id}",
    summary="Full per-strategy detail (all fields from alpha_strategies.json).",
)
async def get_strategy_detail(pair_id: str) -> dict[str, Any]:
    """Return the full, untrimmed strategy entry for ``pair_id``.

    Embeds the visualisations needed by the fullscreen detail view in a
    single round-trip. The async builder fans out the spread-series build,
    live-signal lookup, and catalog ``updated`` fallback via
    :func:`asyncio.gather` — wall-clock is ``max(t_i)``, not ``sum``.

    Responses are cached for 15 minutes per ``pair_id`` via the sync
    builder's namespace; the async path consults that cache first.
    """
    cache = get_cache("alpha_strategy_detail", ttl=15 * 60)
    cache_key = ((pair_id,), ())  # match @cached _default_key((args, kwargs))
    hit = cache.get(cache_key)
    if hit is not None:
        return hit
    try:
        payload = await _build_strategy_detail_async(pair_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(
            status_code=404, detail=f"pair_id '{pair_id}' not found in catalog"
        ) from e
    cache.set(cache_key, payload, ttl=15 * 60)
    return payload


# ---------------------------------------------------------------------------
# On-demand single-pair live-signal recompute
# ---------------------------------------------------------------------------

#: Allowed action labels in the live-signal response. Includes the
#: canonical batch-job actions (``OPEN_LONG``/``OPEN_SHORT``/``HOLD``/...)
#: plus a few UI-facing aliases (``LONG_SPREAD``, ``SHORT_SPREAD``,
#: ``INSUFFICIENT_DATA``, ``NO_TRADE``) the frontend may render.
_LiveSignalAction = Literal[
    "LONG_SPREAD",
    "SHORT_SPREAD",
    "OPEN_LONG",
    "OPEN_SHORT",
    "HOLD",
    "FLAT",
    "CLOSE",
    "STOP_OUT",
    "INSUFFICIENT_DATA",
    "NO_TRADE",
]

#: Actions that should generate a position. Anything else → size=0.
_TRADING_ACTIONS: frozenset[str] = frozenset(
    {"LONG_SPREAD", "SHORT_SPREAD", "OPEN_LONG", "OPEN_SHORT"}
)

#: Max age (seconds) for a cached batch entry to count as "fresh enough"
#: to skip the live recompute. 30 minutes matches the contract.
_LIVE_SIGNAL_CACHE_MAX_AGE_S: int = 30 * 60


class LiveSignalResponse(BaseModel):
    """On-demand live-signal payload for a single pair (see contract)."""

    pair_id: str
    as_of: str = Field(..., description="UTC ISO-8601 of the signal observation.")
    computed_at: str = Field(..., description="UTC ISO-8601 of when this response was assembled.")
    data_source: Literal["live", "cached_batch", "stale_fallback"]
    n_obs: int | None = None
    beta_hedge: float | None = None
    current_a_price: float | None = None
    current_b_price: float | None = None
    current_spread: float | None = None
    previous_z: float | None = None
    current_z: float | None = None
    mu_window: float | None = None
    sigma_window: float | None = None
    action: str
    reason: str | None = None
    decay_status: str | None = None
    rule_window: int | None = None
    rule_entry_z: float | None = None
    rule_exit_z: float | None = None
    rule_stop_z: float | None = None
    tier: str | None = None
    kelly_fraction: float
    edge_bps: float | None = None
    recommended_size_usd: float
    bankroll_usd: float
    warnings: list[str] = Field(default_factory=list)


def _parse_iso_utc(s: str | None) -> datetime | None:
    """Parse an ISO-8601 string into an aware UTC datetime, or ``None``."""
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _cached_signal_is_fresh(cached: dict[str, Any], *, now: datetime) -> bool:
    """Return ``True`` iff the cached signal's ``as_of`` is within max age."""
    as_of = _parse_iso_utc(cached.get("as_of"))
    if as_of is None:
        return False
    age = (now - as_of).total_seconds()
    return 0 <= age <= _LIVE_SIGNAL_CACHE_MAX_AGE_S


def _kelly_for_action(
    *,
    action: str,
    oos_sharpe: float | None,
    kelly_cap: float,
) -> float:
    """Compute Kelly fraction for a pair trade given Sharpe + action.

    Uses the proxy ``f ≈ oos_sharpe / sqrt(252)`` capped at ``kelly_cap``
    and clamped to ``[0, kelly_cap]``. Non-trading actions get 0.
    """
    if action not in _TRADING_ACTIONS:
        return 0.0
    if oos_sharpe is None or not math.isfinite(float(oos_sharpe)):
        return 0.0
    try:
        sharpe = float(oos_sharpe)
    except (TypeError, ValueError):
        return 0.0
    raw = sharpe / math.sqrt(252.0)
    if not math.isfinite(raw):
        return 0.0
    cap = max(0.0, float(kelly_cap))
    return max(0.0, min(raw, cap))


def _edge_bps(current_z: float | None, sigma_window: float | None) -> float | None:
    """Return ``|z| * sigma * 10_000`` in basis points, or ``None``."""
    if current_z is None or sigma_window is None:
        return None
    try:
        z = float(current_z)
        sd = float(sigma_window)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(z) and math.isfinite(sd)):
        return None
    return abs(z) * sd * 10_000.0


def _recommended_size(
    *,
    action: str,
    kelly_fraction: float,
    suggested_allocation: float | None,
    bankroll_usd: float,
) -> float:
    """``recommended_size_usd = kelly * suggested_allocation * bankroll``.

    Zero unless the action is in the trading set and we have an allocation.
    """
    if action not in _TRADING_ACTIONS:
        return 0.0
    if suggested_allocation is None:
        return 0.0
    try:
        alloc = float(suggested_allocation)
        bank = float(bankroll_usd)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(alloc) and math.isfinite(bank)):
        return 0.0
    return max(0.0, kelly_fraction * alloc * bank)


def _build_response_from_signal(
    *,
    pair_id: str,
    signal: dict[str, Any],
    src: dict[str, Any],
    data_source: Literal["live", "cached_batch", "stale_fallback"],
    bankroll_usd: float,
    kelly_cap: float,
    warnings: list[str],
    computed_at_iso: str,
) -> LiveSignalResponse:
    """Map a raw signal dict + strategy record to the public response."""
    action = str(signal.get("action") or "NO_TRADE")
    kelly = _kelly_for_action(
        action=action,
        oos_sharpe=src.get("oos_sharpe"),
        kelly_cap=kelly_cap,
    )
    edge = _edge_bps(signal.get("current_z"), signal.get("sigma_window"))
    size = _recommended_size(
        action=action,
        kelly_fraction=kelly,
        suggested_allocation=src.get("suggested_allocation"),
        bankroll_usd=bankroll_usd,
    )
    as_of = str(signal.get("as_of") or computed_at_iso)
    return LiveSignalResponse(
        pair_id=pair_id,
        as_of=as_of,
        computed_at=computed_at_iso,
        data_source=data_source,
        n_obs=signal.get("n_obs"),
        beta_hedge=signal.get("beta_hedge"),
        current_a_price=signal.get("current_a_price"),
        current_b_price=signal.get("current_b_price"),
        current_spread=signal.get("current_spread"),
        previous_z=signal.get("previous_z"),
        current_z=signal.get("current_z"),
        mu_window=signal.get("mu_window"),
        sigma_window=signal.get("sigma_window"),
        action=action,
        reason=signal.get("reason"),
        decay_status=signal.get("decay_status"),
        rule_window=src.get("rule_window"),
        rule_entry_z=src.get("rule_entry_z"),
        rule_exit_z=src.get("rule_exit_z"),
        rule_stop_z=src.get("rule_stop_z"),
        tier=src.get("tier"),
        kelly_fraction=float(kelly),
        edge_bps=edge,
        recommended_size_usd=float(size),
        bankroll_usd=float(bankroll_usd),
        warnings=warnings,
    )


async def _compute_live_signal_now(src: dict[str, Any], *, as_of_iso: str) -> dict[str, Any]:
    """Fetch fresh leg history from Polymarket and run the spread compute.

    Thin wrapper around :func:`pfm.live_signals_job._polymarket_live_fetcher`
    + :func:`pfm.live_signals_job._compute_signal_for_alpha`. Raises any
    upstream exception unchanged so the caller can decide between
    503-on-force-refresh and stale-fallback.
    """
    pair_id = str(src.get("pair_id"))
    a_id = str(src.get("a_id"))
    b_id = str(src.get("b_id"))
    a_series, b_series = await _polymarket_live_fetcher(pair_id, a_id, b_id)
    a_prices = [float(x) for x in a_series.tolist()]
    b_prices = [float(x) for x in b_series.tolist()]
    return _compute_signal_for_alpha(src, a_prices, b_prices, as_of_iso=as_of_iso)


@router.get(
    "/strategy/{pair_id}/live-signal",
    response_model=LiveSignalResponse,
    summary=(
        "On-demand live signal + Kelly-scaled sizing for a single pair (bypasses the hourly batch)."
    ),
)
async def get_strategy_live_signal(
    pair_id: str,
    bankroll_usd: Annotated[float, Query(ge=0.0)] = 10_000.0,
    force_refresh: Annotated[bool, Query()] = False,
    kelly_cap: Annotated[float, Query(ge=0.0, le=1.0)] = 0.25,
) -> LiveSignalResponse:
    """Compute the live signal for one curated pair on demand.

    Cache policy:
      * ``force_refresh=True``: always recompute from Polymarket. If that
        fails, return 503.
      * Otherwise: if ``live_signals.json`` carries an entry whose
        ``as_of`` is within 30 minutes, return it (``data_source=
        "cached_batch"``). Else compute live; on compute failure fall back
        to any stale cached entry (``data_source="stale_fallback"``) with
        a warning.

    Sizing:
      * ``kelly_fraction ≈ oos_sharpe / sqrt(252)`` clipped to
        ``[0, kelly_cap]``.
      * ``recommended_size_usd = kelly * suggested_allocation * bankroll``.
      * Both go to 0 when ``action ∉ {LONG/SHORT_SPREAD, OPEN_LONG/SHORT}``.
    """
    try:
        src = _find_strategy_src(pair_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"pair_id '{pair_id}' not found in catalog"
        ) from exc

    now = datetime.now(tz=UTC)
    now_iso = now.isoformat()

    cached_signal: dict[str, Any] | None = None
    try:
        cached_signal = _cached_live_signals().get(pair_id)
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("alpha_hub live-signal cache read failed: %r", exc)
        cached_signal = None

    # 1) Cached-batch fast path (unless force_refresh).
    if (
        not force_refresh
        and isinstance(cached_signal, dict)
        and _cached_signal_is_fresh(cached_signal, now=now)
    ):
        return _build_response_from_signal(
            pair_id=pair_id,
            signal=cached_signal,
            src=src,
            data_source="cached_batch",
            bankroll_usd=bankroll_usd,
            kelly_cap=kelly_cap,
            warnings=[],
            computed_at_iso=now_iso,
        )

    # 2) Live recompute.
    try:
        fresh = await _compute_live_signal_now(src, as_of_iso=now_iso)
    except Exception as exc:
        # Force-refresh callers want an explicit failure, not stale data.
        if force_refresh:
            raise HTTPException(
                status_code=503,
                detail=f"live recompute failed: {type(exc).__name__}: {exc!s}"[:240],
            ) from exc
        # Soft-fail: try the cached entry (even if stale).
        if isinstance(cached_signal, dict):
            warning = (
                f"live recompute failed: {type(exc).__name__}; returning last cached batch signal"
            )
            return _build_response_from_signal(
                pair_id=pair_id,
                signal=cached_signal,
                src=src,
                data_source="stale_fallback",
                bankroll_usd=bankroll_usd,
                kelly_cap=kelly_cap,
                warnings=[warning],
                computed_at_iso=now_iso,
            )
        raise HTTPException(
            status_code=503,
            detail=(
                f"live recompute failed and no cached signal available: "
                f"{type(exc).__name__}: {exc!s}"
            )[:240],
        ) from exc

    return _build_response_from_signal(
        pair_id=pair_id,
        signal=fresh,
        src=src,
        data_source="live",
        bankroll_usd=bankroll_usd,
        kelly_cap=kelly_cap,
        warnings=[],
        computed_at_iso=now_iso,
    )


@router.get(
    "/live-panel",
    response_model=LivePanelResponse,
    summary="Composite payload: top production alphas + watchlist + recent graveyard.",
)
def get_live_panel() -> LivePanelResponse:
    """Return a small dashboard payload suitable for the hub landing card.

    * ``production`` — top 3 by ``oos_sharpe`` from ``A_STRUCTURAL`` + ``A_GOLD``.
    * ``watchlist`` — top entries with tier ``B_VALIDATED`` (capped at 10).
    * ``graveyard`` — last 5 graveyard entries by ``killed_iso`` (most recent first).
    """
    try:
        strategies = _cached_strategies()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    prod_pool = [s for s in strategies if s.get("tier") in {"A_STRUCTURAL", "A_GOLD"}]
    prod_pool = _sort_strategies(prod_pool, sort="oos_sharpe", order="desc")
    production = [_to_item(s) for s in prod_pool[:3]]

    watch_pool = [s for s in strategies if s.get("tier") == "B_VALIDATED"]
    watch_pool = _sort_strategies(watch_pool, sort="oos_sharpe", order="desc")
    watchlist = [_to_item(s) for s in watch_pool[:10]]

    try:
        graveyard_raw = load_graveyard()
    except (FileNotFoundError, ValueError):
        graveyard_raw = []
    graveyard = sorted(graveyard_raw, key=lambda e: str(e.get("killed_iso", "")), reverse=True)[:5]

    return LivePanelResponse(production=production, watchlist=watchlist, graveyard=graveyard)


__all__ = [
    "ALPHA_STRATEGIES_PATH",
    "LIVE_SIGNALS_PATH",
    "LeaderboardItem",
    "LeaderboardResponse",
    "LivePanelResponse",
    "LiveSignalResponse",
    "router",
]
