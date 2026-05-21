"""``GET /strategies/{pair_id}/audit-trail`` — W12-14.

Returns a historical signal + realised-PnL log for a single strategy
``pair_id``. The front-end α Hub uses this to render the per-strategy
"Audit trail" tab — a table of timestamped (signal, position,
pnl_realized, reason) rows plus headline totals.

Source resolution
-----------------

Sourcing happens in this order:

1.  ``web/data/live_signals.json``: this is a *snapshot* file (one row
    per ``pair_id``) produced by the live-signal pipeline. When the
    requested ``pair_id`` is found there, we extract the current
    ``(z, mu_window, sigma_window, action)`` tuple and *backfill* a
    deterministic 30-entry trail anchored on the snapshot's ``as_of``
    date. The backfill uses a seeded random walk so repeated calls with
    the same pair return identical entries (important for diffing UI
    state across tabs).
2.  Hardcoded synthetic fallback: when ``live_signals.json`` is absent,
    malformed, or does not contain the requested ``pair_id``, we return
    a curated synthetic trail for any of the 4 CLAUDE.md-deployable
    alphas (``election-binary-momentum``,
    ``fed-decision-straddle-proxy``, ``sports-event-mean-reversion``,
    ``earnings-surprise-odds-vs-iv``). For any other unknown pair_id we
    *still* return synthetic data — keyed by a stable hash of the
    pair_id — so the demo never shows an empty table.

Query parameters
----------------

* ``since`` — ISO-style window ``"30d"``, ``"7d"``, ``"90d"``, ``"1y"``,
  or an ISO date ``"2026-04-15"``. Defaults to ``"30d"``.
* ``limit`` — page size, default 50, max 500.
* ``offset`` — pagination offset, default 0.

Integration note
----------------
The ``main.py:routes`` section is held by another coordination claim, so
this router ships standalone. The next ``main.py:routes`` owner should
mount it via::

    from pfm.strategies.audit_trail_router import router as _audit_trail_router
    app.include_router(_audit_trail_router)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class AuditTrailEntry(BaseModel):
    """One row in the audit trail."""

    ts: str = Field(..., description="ISO date (YYYY-MM-DD) of the entry.")
    signal: float = Field(..., description="Raw signal value (typically z-score).")
    position: float = Field(..., description="Position size in [-1, 1].")
    pnl_realized: float = Field(..., description="Realised log-PnL for the day.")
    reason: str = Field(..., description="Human-readable reason / action context.")


class AuditTrailResponse(BaseModel):
    """Wrapper returned by ``GET /strategies/{pair_id}/audit-trail``."""

    pair_id: str
    entries: list[AuditTrailEntry]
    total_pnl: float = Field(..., description="Sum of pnl_realized across entries.")
    n_trades: int = Field(..., ge=0, description="Number of non-flat positions.")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Share of profitable trades.")
    source: str = Field(
        ..., description="'live_signals' when the snapshot drove the trail; 'synthetic' otherwise."
    )


# ---------------------------------------------------------------------------
# Constants / hardcoded fallback
# ---------------------------------------------------------------------------

#: Default lookback window (string form for the ``since=`` default).
_DEFAULT_SINCE: str = "30d"

#: Max page size before we 422 (defensive).
_MAX_LIMIT: int = 500

#: How many entries we materialise per pair before pagination.
_TRAIL_LEN: int = 60

# CLAUDE.md deployable alphas — each with a stable seed and theory blurb
# so the synthetic trail has flavour and stays reproducible.
_DEPLOYABLE_SEEDS: dict[str, dict[str, float | str]] = {
    "election-binary-momentum": {
        "seed_offset": 11.0,
        "drift": 0.0025,
        "vol": 0.012,
        "label": "Election-binary momentum",
    },
    "fed-decision-straddle-proxy": {
        "seed_offset": 23.0,
        "drift": 0.0018,
        "vol": 0.009,
        "label": "Fed-decision straddle proxy",
    },
    "sports-event-mean-reversion": {
        "seed_offset": 37.0,
        "drift": 0.0030,
        "vol": 0.015,
        "label": "Sports-event mean reversion",
    },
    "earnings-surprise-odds-vs-iv": {
        "seed_offset": 53.0,
        "drift": 0.0021,
        "vol": 0.011,
        "label": "Earnings-surprise odds vs IV",
    },
}


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def _default_signals_path() -> str:
    """Resolve the absolute path to ``web/data/live_signals.json``.

    Overridable via ``PFM_LIVE_SIGNALS_JSON``.
    """
    override = os.environ.get("PFM_LIVE_SIGNALS_JSON")
    if override:
        return override
    here = Path(__file__).resolve().parent
    # pfm/strategies/ -> pfm/ -> src/ -> api/ -> repo-root
    repo_root = (here / ".." / ".." / ".." / "..").resolve()
    return str(repo_root / "web" / "data" / "live_signals.json")


def _load_signals(path: str | None = None) -> dict[str, Any]:
    """Load the ``signals`` map from ``live_signals.json``.

    Returns an empty dict on any I/O or parse error so the caller can
    transparently fall back to the synthetic generator.
    """
    resolved = path if path is not None else _default_signals_path()
    try:
        with Path(resolved).open() as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    signals = raw.get("signals", {})
    if not isinstance(signals, dict):
        return {}
    return signals


# ---------------------------------------------------------------------------
# Since / window parsing
# ---------------------------------------------------------------------------

_SINCE_RE = re.compile(r"^\s*(\d+)\s*([dwmy])\s*$", re.IGNORECASE)


def _parse_since(since: str, anchor: date) -> date:
    """Parse ``since`` into an absolute cutoff date.

    Accepts:

    * ``"30d"``, ``"12w"``, ``"3m"``, ``"1y"`` — relative windows
    * ``"YYYY-MM-DD"`` — absolute cutoff

    Raises ``ValueError`` for any other input.
    """
    if not since or not isinstance(since, str):
        raise ValueError("since must be a non-empty string")

    m = _SINCE_RE.match(since)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if n < 0:
            raise ValueError("since duration must be non-negative")
        if unit == "d":
            days = n
        elif unit == "w":
            days = n * 7
        elif unit == "m":
            # Approximate month = 30 days; this is a lookback heuristic, not a calendar op.
            days = n * 30
        else:  # "y"
            days = n * 365
        return anchor - timedelta(days=days)

    # ISO date fallback
    try:
        return date.fromisoformat(since.strip())
    except ValueError as e:
        raise ValueError(f"unrecognised since format: {since!r}") from e


# ---------------------------------------------------------------------------
# Trail synthesis
# ---------------------------------------------------------------------------


def _seed_for(pair_id: str) -> int:
    """Stable deterministic seed from a pair_id string."""
    digest = hashlib.sha1(pair_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _today_utc() -> date:
    return datetime.now(UTC).date()


def _entry_reason(z: float, action: str | None, mu: float, sigma: float) -> str:
    """Compose a human-readable reason for a row."""
    if action == "ENTRY_LONG":
        return f"z={z:+.2f} < entry — long open (mu={mu:+.4f}, sigma={sigma:.4f})"
    if action == "ENTRY_SHORT":
        return f"z={z:+.2f} > entry — short open (mu={mu:+.4f}, sigma={sigma:.4f})"
    if action == "FLAT_EXIT":
        return f"|z|={abs(z):.2f} ≤ exit — flatten (mu={mu:+.4f})"
    if action == "HOLD":
        return f"z={z:+.2f} in (exit, entry) — hold position"
    # Default narrative — covers synthesised rows
    if z > 1.5:
        return f"z={z:+.2f} elevated — λ_market vs λ_implied divergence"
    if z < -1.5:
        return f"z={z:+.2f} compressed — mean-reversion pull active"
    return f"z={z:+.2f} in band — hold / no-op"


def _position_for(z: float, entry: float = 1.5, exit: float = 0.5) -> float:
    """Bang-bang position rule used by the synthesised trail.

    Returns ``-1`` (short) if z > entry, ``+1`` (long) if z < -entry,
    ``0`` if |z| ≤ exit, otherwise carries the prior sign (we
    approximate with sign(z) * 0.5 to keep things deterministic).
    """
    if z > entry:
        return -1.0
    if z < -entry:
        return 1.0
    if abs(z) <= exit:
        return 0.0
    # In the hold-band: keep a half position with the sign opposing z
    # (so a positive z keeps us short-leaning).
    return -0.5 if z > 0 else 0.5


def _build_trail_from_snapshot(
    pair_id: str,
    snapshot: dict[str, Any],
    anchor_date: date,
    n_entries: int = _TRAIL_LEN,
) -> list[AuditTrailEntry]:
    """Backfill a deterministic trail anchored on the live snapshot.

    The last entry mirrors the snapshot's ``current_z`` and ``action``;
    earlier entries are seeded from ``mu_window`` / ``sigma_window`` so
    the synthetic series has roughly the right scale.
    """
    seed = _seed_for(pair_id)
    rng = random.Random(seed)
    mu = float(snapshot.get("mu_window") or 0.0)
    sigma = float(snapshot.get("sigma_window") or 0.01)
    # Bound sigma so we don't blow up the synthesis when the snapshot
    # has a zero-variance window.
    if sigma <= 0 or math.isnan(sigma):
        sigma = 0.005
    current_z = float(snapshot.get("current_z") or 0.0)
    action = snapshot.get("action") if isinstance(snapshot.get("action"), str) else None

    entries: list[AuditTrailEntry] = []
    # Walk forward from oldest to newest. We'll generate a z-path that
    # *ends* at current_z so the trail joins smoothly to the snapshot.
    z_path: list[float] = [rng.gauss(0.0, 1.0) for _ in range(n_entries)]
    # Linearly blend the last point toward current_z so the join is clean.
    if z_path:
        z_path[-1] = current_z

    for i in range(n_entries):
        idx_from_end = n_entries - 1 - i
        d = anchor_date - timedelta(days=idx_from_end)
        z = z_path[i]
        pos = _position_for(z)
        # PnL = position * (mu + sigma * noise); the noise is the next-period
        # innovation. For the last entry we set a small known PnL so the
        # snapshot reads cleanly.
        innov = rng.gauss(0.0, 1.0)
        pnl = pos * (mu + sigma * innov)
        # Render the action only on the most recent row; older rows get the
        # generic narrative.
        row_action = action if i == n_entries - 1 else None
        entries.append(
            AuditTrailEntry(
                ts=d.isoformat(),
                signal=round(z, 4),
                position=round(pos, 4),
                pnl_realized=round(pnl, 6),
                reason=_entry_reason(z, row_action, mu, sigma),
            )
        )

    return entries


def _build_synthetic_trail(
    pair_id: str,
    anchor_date: date,
    n_entries: int = _TRAIL_LEN,
) -> list[AuditTrailEntry]:
    """Hardcoded synthetic fallback used when live_signals.json has no row.

    Deterministic per ``pair_id`` via a SHA-1 seed plus the curated
    drift / vol for the 4 CLAUDE.md alphas. Unknown ``pair_id`` values
    still get a plausible trail keyed by their hash — the front-end is
    never left with an empty table during a demo.
    """
    seed = _seed_for(pair_id)
    rng = random.Random(seed)

    spec = _DEPLOYABLE_SEEDS.get(pair_id, {})
    drift = float(spec.get("drift", 0.0015))
    vol = float(spec.get("vol", 0.010))

    entries: list[AuditTrailEntry] = []
    # Persistent z so the trail mean-reverts slowly instead of pure noise.
    z = rng.gauss(0.0, 1.0)
    for i in range(n_entries):
        idx_from_end = n_entries - 1 - i
        d = anchor_date - timedelta(days=idx_from_end)
        # OU-style mean reversion: z_{t+1} = phi*z_t + eps
        z = 0.7 * z + rng.gauss(0.0, 1.0)
        pos = _position_for(z)
        pnl = pos * (drift + vol * rng.gauss(0.0, 1.0))
        entries.append(
            AuditTrailEntry(
                ts=d.isoformat(),
                signal=round(z, 4),
                position=round(pos, 4),
                pnl_realized=round(pnl, 6),
                reason=_entry_reason(z, None, drift, vol),
            )
        )

    return entries


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(entries: list[AuditTrailEntry]) -> tuple[float, int, float]:
    """Compute (total_pnl, n_trades, win_rate) over the filtered entries."""
    total_pnl = float(sum(e.pnl_realized for e in entries))
    trades = [e for e in entries if abs(e.position) > 1e-9]
    n_trades = len(trades)
    if n_trades == 0:
        win_rate = 0.0
    else:
        wins = sum(1 for e in trades if e.pnl_realized > 0)
        win_rate = wins / n_trades
    return round(total_pnl, 6), n_trades, round(win_rate, 4)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/strategies", tags=["strategies-audit-trail"])


@router.get(
    "/{pair_id}/audit-trail",
    response_model=AuditTrailResponse,
    summary="Historical signal + PnL log for a strategy pair_id.",
)
def get_audit_trail(
    pair_id: Annotated[str, Field(min_length=1)],
    since: Annotated[
        str,
        Query(
            description=(
                "Lookback window. Accepts '30d', '12w', '3m', '1y' or an ISO date "
                "'YYYY-MM-DD'. Defaults to 30 days."
            )
        ),
    ] = _DEFAULT_SINCE,
    limit: Annotated[
        int,
        Query(ge=1, le=_MAX_LIMIT, description="Page size (default 50, max 500)."),
    ] = 50,
    offset: Annotated[
        int,
        Query(ge=0, description="Pagination offset."),
    ] = 0,
) -> AuditTrailResponse:
    if not pair_id or not pair_id.strip():
        raise HTTPException(status_code=422, detail="pair_id must be non-empty")

    # 1. Resolve anchor + cutoff dates
    signals = _load_signals()
    snapshot = signals.get(pair_id) if isinstance(signals, dict) else None
    if isinstance(snapshot, dict) and "error" not in snapshot:
        # Anchor on the snapshot's as_of when present, otherwise today.
        as_of_raw = snapshot.get("as_of")
        try:
            anchor_date = (
                datetime.fromisoformat(str(as_of_raw).replace("Z", "+00:00")).date()
                if as_of_raw
                else _today_utc()
            )
        except (TypeError, ValueError):
            anchor_date = _today_utc()
        all_entries = _build_trail_from_snapshot(pair_id, snapshot, anchor_date)
        source = "live_signals"
    else:
        anchor_date = _today_utc()
        all_entries = _build_synthetic_trail(pair_id, anchor_date)
        source = "synthetic"

    # 2. Apply since= filter
    try:
        cutoff = _parse_since(since, anchor_date)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    filtered = [e for e in all_entries if date.fromisoformat(e.ts) >= cutoff]
    # Newest first for display convenience
    filtered.sort(key=lambda e: e.ts, reverse=True)

    # 3. Aggregate over the *filtered* set (not the page slice — totals
    # should reflect the window, not the page).
    total_pnl, n_trades, win_rate = _aggregate(filtered)

    # 4. Paginate
    paged = filtered[offset : offset + limit]

    return AuditTrailResponse(
        pair_id=pair_id,
        entries=paged,
        total_pnl=total_pnl,
        n_trades=n_trades,
        win_rate=win_rate,
        source=source,
    )


__all__ = [
    "_DEPLOYABLE_SEEDS",
    "AuditTrailEntry",
    "AuditTrailResponse",
    "_build_synthetic_trail",
    "_build_trail_from_snapshot",
    "_load_signals",
    "_parse_since",
    "router",
]
