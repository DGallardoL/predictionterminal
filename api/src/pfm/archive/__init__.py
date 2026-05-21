"""Archive subpackage — historical / settled prediction-market data.

Modules:
    - :mod:`pfm.archive.polymarket_archive` — settled Polymarket markets, full
      daily history, derived stats (peak/trough, vol, Hurst, DFA, half-life).
    - :mod:`pfm.archive.resolutions` — resolution-outcome lookup (YES / NO /
      AMBIGUOUS / PENDING) plus payout & dispute history.
    - :mod:`pfm.archive.router` — FastAPI router exposing the Polymarket
      archive under ``/archive/polymarket/*``.
    - :mod:`pfm.archive.kalshi_archive` — settled Kalshi markets, history,
      and per-series distributions.
    - :mod:`pfm.archive.cross_venue_archive` — same-event Polymarket vs
      Kalshi resolution comparison.
    - :mod:`pfm.archive.kalshi_router` — FastAPI router for the Kalshi
      archive and cross-venue comparator under ``/archive/kalshi/*`` and
      ``/archive/cross-venue/*``.

The Polymarket archive symbols are imported lazily so this package stays
importable in slices that only ship the Kalshi side. Tests build a
throw-away FastAPI app and mount the relevant router directly via
``TestClient``.
"""

from __future__ import annotations

# Both archive sides (Polymarket / Kalshi) are imported defensively so a
# partial deployment of the package — i.e. one side shipped without the
# other — keeps the rest of pfm importable. Tests for one side never
# transitively force the other side's optional deps.

try:  # pragma: no cover — re-exports only
    from pfm.archive.cross_venue_archive import (
        CROSS_VENUE_CONCEPTS,
        cross_venue_resolved_pairs,
    )
except ImportError:  # pragma: no cover
    CROSS_VENUE_CONCEPTS = None  # type: ignore[assignment]
    cross_venue_resolved_pairs = None  # type: ignore[assignment]

try:  # pragma: no cover — re-exports only
    from pfm.archive.kalshi_archive import (
        fetch_archive_kalshi_detail,
        fetch_settled_markets,
        kalshi_archive_series_distribution,
    )
except ImportError:  # pragma: no cover
    fetch_archive_kalshi_detail = None  # type: ignore[assignment]
    fetch_settled_markets = None  # type: ignore[assignment]
    kalshi_archive_series_distribution = None  # type: ignore[assignment]

try:  # pragma: no cover — re-exports only
    from pfm.archive.kalshi_router import router as kalshi_router
except ImportError:  # pragma: no cover
    kalshi_router = None  # type: ignore[assignment]

try:  # pragma: no cover — re-exports only
    from pfm.archive.polymarket_archive import (
        archive_themes_distribution,
        fetch_archive_market_detail,
        fetch_resolved_markets,
    )
    from pfm.archive.resolutions import get_resolution
    from pfm.archive.router import router
except ImportError:  # pragma: no cover
    archive_themes_distribution = None  # type: ignore[assignment]
    fetch_archive_market_detail = None  # type: ignore[assignment]
    fetch_resolved_markets = None  # type: ignore[assignment]
    get_resolution = None  # type: ignore[assignment]
    router = None  # type: ignore[assignment]


__all__ = [
    "CROSS_VENUE_CONCEPTS",
    "archive_themes_distribution",
    "cross_venue_resolved_pairs",
    "fetch_archive_kalshi_detail",
    "fetch_archive_market_detail",
    "fetch_resolved_markets",
    "fetch_settled_markets",
    "get_resolution",
    "kalshi_archive_series_distribution",
    "kalshi_router",
    "router",
]
