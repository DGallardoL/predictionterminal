"""Cross-sectional realised-volatility distribution endpoint for the Terminal.

Given a Polymarket slug, this module computes the realised volatility of its
Δlogit return series over a rolling window, then ranks it against the
realised vols of every other factor that shares its theme. The output is
the empirical p10/p25/p50/p75/p90 of theme-peer vol, the input's percentile
within that distribution, a z-score (vs the peer cross-section), and the
five highest- and lowest-vol peer slugs.

The "universe" is the factor catalogue (``factors.yml``) bucketed by the
``theme`` field. Price history comes from the cached pickle written by the
strat-7 sweep job (``/tmp/strat7_factor_history.pkl``); this is a dict
keyed by slug → pandas Series of daily probabilities. Slugs missing from
the cache are silently skipped — typical for very fresh factors.

Endpoint
--------
``GET /terminal/vol-distribution/{slug}?window=30``

Response shape
--------------
.. code-block:: json

    {
      "slug": "...",
      "current_vol": 0.83,
      "theme": "macro",
      "n_peers": 11,
      "percentile_in_theme": 64.3,
      "vol_distribution": {"p10": ..., "p25": ..., "p50": ..., "p75": ..., "p90": ...},
      "current_z_score": 0.42,
      "peers_higher_vol": [{"slug": "...", "vol": ...}, ...],
      "peers_lower_vol":  [{"slug": "...", "vol": ...}, ...]
    }

The router is registered standalone; ``main.py`` is left untouched.
"""

from __future__ import annotations

import functools
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from pfm.factors import FactorConfig
from pfm.model import DEFAULT_EPSILON, delta_logit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terminal", tags=["terminal"])

#: Default location of the cached factor-history pickle.
DEFAULT_FACTOR_HISTORY_PATH: Path = Path("/tmp/strat7_factor_history.pkl")

#: Annualisation factor for daily-bar σ → per-year σ.
ANNUALISATION: float = float(np.sqrt(252.0))

#: Default rolling window (in trading days).
DEFAULT_WINDOW: int = 30

#: How many top/bottom peers to surface.
TOP_N_PEERS: int = 5


# --- DI dependencies --------------------------------------------------------
#
# Both dependencies raise 503 when unwired so a misconfigured app fails loud
# rather than silent. Tests override them via ``app.dependency_overrides``.


def _get_factors_dep() -> dict[str, FactorConfig]:  # pragma: no cover - DI shim
    raise HTTPException(
        status_code=503,
        detail="vol-distribution router not wired into an app with factors",
    )


def _get_history_path_dep() -> Path:
    """Resolve the on-disk pickle path. Override in tests."""
    return DEFAULT_FACTOR_HISTORY_PATH


# --- core computation -------------------------------------------------------


@dataclass(frozen=True)
class VolDistributionResult:
    """Strongly-typed payload mirroring the JSON response."""

    slug: str
    current_vol: float
    theme: str
    n_peers: int
    percentile_in_theme: float
    vol_distribution: dict[str, float]
    current_z_score: float
    peers_higher_vol: list[dict[str, float | str]]
    peers_lower_vol: list[dict[str, float | str]]


@functools.cache
def _load_history_pickle(path: Path) -> dict[str, pd.Series]:
    """Best-effort load of the slug→Series cache. Returns ``{}`` on failure.

    Wrapped in ``functools.cache`` (perf audit 2026-05-16): the pickle is
    static on disk, so re-reading on every request was pure overhead. The
    LRU is keyed on ``path``, so tests using ``tmp_path`` pickles still load
    fresh. Call ``_load_history_pickle.cache_clear()`` to force refresh.
    """
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            data = pickle.load(fh)
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as e:
        logger.warning("could not load factor history pickle %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def realized_vol_from_prices(
    prices: pd.Series,
    window: int,
    *,
    epsilon: float = DEFAULT_EPSILON,
    annualisation: float = ANNUALISATION,
) -> float:
    """Compute the most-recent rolling realised-σ on a daily price series.

    Args:
        prices: Daily probability series (index = date).
        window: Rolling-window length in days.
        epsilon: Logit-clip used inside :func:`pfm.model.delta_logit`.
        annualisation: Multiplier applied to per-day σ (default √252).

    Returns:
        The annualised σ of the last ``window`` Δlogit returns. Returns
        ``float('nan')`` if there are fewer than ``window`` valid returns.

    Raises:
        ValueError: If ``window`` is non-positive.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    s = prices.dropna().astype(float)
    if len(s) < window + 1:
        return float("nan")

    returns = delta_logit(s, epsilon=epsilon).dropna()
    if len(returns) < window:
        return float("nan")

    rolling = returns.rolling(window=window, min_periods=window).std(ddof=1)
    rolling = rolling.dropna()
    if rolling.empty:
        return float("nan")

    return float(rolling.iloc[-1] * annualisation)


def compute_vol_distribution(
    slug: str,
    factors: dict[str, FactorConfig],
    history: dict[str, pd.Series],
    *,
    window: int = DEFAULT_WINDOW,
    epsilon: float = DEFAULT_EPSILON,
    top_n: int = TOP_N_PEERS,
    factors_by_slug: dict[str, FactorConfig] | None = None,
) -> VolDistributionResult:
    """Cross-sectional vol distribution of ``slug`` vs same-theme peers.

    Args:
        slug: The Polymarket slug of the input factor (must exist in ``factors``).
        factors: ``{factor_id: FactorConfig}`` catalogue.
        history: ``{slug: price Series}`` cache. Slugs missing from this map
            are skipped — they contribute neither to the percentile nor to the
            top/bottom peer lists.
        window: Rolling window used to estimate σ (default 30).
        epsilon: Logit clip ε for Δlogit (default 0.01).
        top_n: How many highest- / lowest-vol peers to return.

    Returns:
        :class:`VolDistributionResult`.

    Raises:
        KeyError: If ``slug`` does not appear in ``factors``.
        ValueError: If ``slug`` has no usable history (cannot anchor the dist).
    """
    # 1. Resolve the input factor by slug. Prefer the lifespan-built O(1)
    #    index passed in by the handler; fall back to a one-shot build for
    #    direct unit-test calls that don't thread the index through.
    if factors_by_slug is None:
        factors_by_slug = {fc.slug: fc for fc in factors.values() if fc.slug}
    target = factors_by_slug.get(slug)
    if target is None:
        raise KeyError(slug)
    theme = target.theme

    # 2. Same-theme peers (excluding the input itself).
    peer_slugs: list[str] = [
        fc.slug for fc in factors.values() if fc.theme == theme and fc.slug != slug
    ]

    # 3. Compute the input's current vol — this is required, otherwise we
    #    have nothing to rank against the peer distribution.
    target_series = history.get(slug)
    if target_series is None or target_series.empty:
        raise ValueError(f"no cached history for input slug {slug!r}")
    current_vol = realized_vol_from_prices(target_series, window=window, epsilon=epsilon)
    if not np.isfinite(current_vol):
        raise ValueError(f"insufficient history for slug {slug!r} at window={window}")

    # 4. Compute peer vols (skip those without finite vol, e.g. missing data).
    peer_vols: dict[str, float] = {}
    for ps in peer_slugs:
        s = history.get(ps)
        if s is None or s.empty:
            continue
        v = realized_vol_from_prices(s, window=window, epsilon=epsilon)
        if np.isfinite(v):
            peer_vols[ps] = v

    n_peers = len(peer_vols)

    # 5. Cross-sectional distribution of peer vols (excludes the target so
    #    its percentile is read against the *peer* sample, not against itself).
    if n_peers > 0:
        peer_array = np.fromiter(peer_vols.values(), dtype=float, count=n_peers)
        q = np.quantile(peer_array, [0.10, 0.25, 0.50, 0.75, 0.90])
        vol_distribution = {
            "p10": float(q[0]),
            "p25": float(q[1]),
            "p50": float(q[2]),
            "p75": float(q[3]),
            "p90": float(q[4]),
        }
        # Empirical percentile: % of peers with vol <= current_vol, in [0, 100].
        percentile = float((peer_array <= current_vol).mean() * 100.0)
        # z-score against peer cross-section (sample std, ddof=1).
        if n_peers >= 2:
            mu = float(peer_array.mean())
            sd = float(peer_array.std(ddof=1))
            z = float((current_vol - mu) / sd) if sd > 0 else 0.0
        else:
            z = 0.0
    else:
        vol_distribution = {
            "p10": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
        percentile = float("nan")
        z = float("nan")

    # 6. Top/bottom peers by vol.
    sorted_desc = sorted(peer_vols.items(), key=lambda kv: kv[1], reverse=True)
    sorted_asc = sorted(peer_vols.items(), key=lambda kv: kv[1])
    peers_higher: list[dict[str, float | str]] = [
        {"slug": s, "vol": float(v)} for s, v in sorted_desc[:top_n]
    ]
    peers_lower: list[dict[str, float | str]] = [
        {"slug": s, "vol": float(v)} for s, v in sorted_asc[:top_n]
    ]

    return VolDistributionResult(
        slug=slug,
        current_vol=current_vol,
        theme=theme,
        n_peers=n_peers,
        percentile_in_theme=percentile,
        vol_distribution=vol_distribution,
        current_z_score=z,
        peers_higher_vol=peers_higher,
        peers_lower_vol=peers_lower,
    )


# --- HTTP handler -----------------------------------------------------------


@router.get("/vol-distribution/{slug}")
def get_vol_distribution(
    request: Request,
    slug: str,
    window: Annotated[
        int, Query(ge=2, le=252, description="rolling-σ window in days")
    ] = DEFAULT_WINDOW,
    epsilon: Annotated[float, Query(gt=0.0, lt=0.5, description="logit clip ε")] = DEFAULT_EPSILON,
    factors: Annotated[dict[str, FactorConfig], Depends(_get_factors_dep)] = ...,  # type: ignore[assignment]
    history_path: Annotated[Path, Depends(_get_history_path_dep)] = ...,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Return the cross-sectional realised-vol distribution for a single slug.

    Warm-cache fast path: if the lifespan prewarm has stored a fresh
    snapshot for this slug on ``app.state.warm_voldist`` (default-window,
    default-epsilon only — anything else falls through to live-compute),
    return it directly. This cuts cold-tail latency from ~3 s to <50 ms
    on the curated headliners. See ``pfm.prewarm`` for the producer side.
    """
    # Only short-circuit on the canonical default parameterisation — any
    # caller-specified window/epsilon must recompute live.
    if window == DEFAULT_WINDOW and abs(epsilon - DEFAULT_EPSILON) < 1e-12:
        try:
            from pfm.prewarm import warm_voldist_lookup as _warm_lookup
        except ImportError:  # pragma: no cover - module is in-repo
            _warm_lookup = None
        if _warm_lookup is not None:
            cached = _warm_lookup(request.app, slug)
            if cached is not None:
                return cached

    history = _load_history_pickle(history_path)

    # Use the lifespan-built slug index for O(1) lookup; if absent (e.g.
    # bare-app unit tests), compute_vol_distribution will rebuild internally.
    by_slug = getattr(request.app.state, "factors_by_slug", None)

    try:
        result = compute_vol_distribution(
            slug,
            factors=factors,
            history=history,
            window=window,
            epsilon=epsilon,
            factors_by_slug=by_slug if isinstance(by_slug, dict) else None,
        )
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"unknown slug {e.args[0]!r} (not in factors.yml)",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return {
        "slug": result.slug,
        "current_vol": result.current_vol,
        "theme": result.theme,
        "n_peers": result.n_peers,
        "percentile_in_theme": result.percentile_in_theme,
        "vol_distribution": result.vol_distribution,
        "current_z_score": result.current_z_score,
        "peers_higher_vol": result.peers_higher_vol,
        "peers_lower_vol": result.peers_lower_vol,
    }


__all__ = [
    "ANNUALISATION",
    "DEFAULT_FACTOR_HISTORY_PATH",
    "DEFAULT_WINDOW",
    "TOP_N_PEERS",
    "VolDistributionResult",
    "compute_vol_distribution",
    "realized_vol_from_prices",
    "router",
]
