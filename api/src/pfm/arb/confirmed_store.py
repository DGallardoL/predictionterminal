"""Durable, unbounded store of every market/pair where an arbitrage was seen.

The legacy seen-store (``/tmp/pfm_arb_confirmed_matches.json``) is ephemeral and
the engine keeps only a 500-item FIFO history. This module provides a durable,
growable alternative: every distinct ``arb_key`` ever observed is upserted into
a JSON-backed store that, by default, keeps everything forever (unbounded).

Persistence is atomic (temp file + :func:`os.replace`) so a crash mid-write
never clobbers the existing file. The store lazily loads from disk on first use
and tolerates a missing or corrupt file by falling back to an empty store.

Example:
    >>> store = ConfirmedArbStore("/data/confirmed_arbs.json")
    >>> store.record("KXBTC-26-T100000:btc-100k-2026", kalshi_ticker="KXBTC",
    ...              poly_slug="btc-100k-2026", profit_pct=3.2)
    >>> store.confirmed(min_count=3)
    []
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_STORE_PATH",
    "ENV_STORE_PATH",
    "RECENT_PROFIT_MAXLEN",
    "ConfirmedArb",
    "ConfirmedArbStore",
]

#: Default on-disk location. In production this lives under ``arbstuff/`` which
#: is a volume-backed directory, so the store survives container restarts.
DEFAULT_STORE_PATH = Path("arbstuff/confirmed_arbs.json")

#: Environment variable that overrides the store path (used by tests via
#: ``tmp_path`` and by ops to point at the mounted volume).
ENV_STORE_PATH = "PFM_ARB_CONFIRMED_STORE"

#: How many recent profit observations to retain per ``arb_key``.
RECENT_PROFIT_MAXLEN = 20


def _now_iso(now: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp string.

    Args:
        now: Optional override for the current time (for deterministic tests).

    Returns:
        ISO-8601 formatted timestamp in UTC.
    """
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat()


def _resolve_path(path: str | Path | None) -> Path:
    """Resolve the effective store path.

    Precedence: explicit constructor ``path`` > ``PFM_ARB_CONFIRMED_STORE``
    env var > :data:`DEFAULT_STORE_PATH`.

    Args:
        path: Explicit path passed to the constructor, or ``None``.

    Returns:
        The resolved :class:`~pathlib.Path`.
    """
    if path is not None:
        return Path(path)
    env = os.environ.get(ENV_STORE_PATH)
    if env:
        return Path(env)
    return DEFAULT_STORE_PATH


@dataclass
class ConfirmedArb:
    """A single market/pair where an arbitrage opportunity was observed.

    Attributes:
        arb_key: Stable identity for the opportunity (e.g. ``"<ticker>:<slug>"``).
        kalshi_ticker: Kalshi side identifier.
        poly_slug: Polymarket side slug.
        count: Number of distinct times this arb has been recorded.
        first_seen: ISO-8601 UTC timestamp of first observation.
        last_seen: ISO-8601 UTC timestamp of most recent observation.
        max_profit_pct: Largest profit percentage ever observed.
        recent_profit_pct: Rolling window of the most recent profit observations
            (newest last, capped at :data:`RECENT_PROFIT_MAXLEN`).
        confidence: Optional latest confidence label (e.g. ``"high"``).
        volume: Optional latest observed volume.
        extra: Free-form metadata merged on each record.
    """

    arb_key: str
    kalshi_ticker: str
    poly_slug: str
    count: int
    first_seen: str
    last_seen: str
    max_profit_pct: float
    recent_profit_pct: list[float] = field(default_factory=list)
    confidence: str | None = None
    volume: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-compatible dict."""
        return {
            "arb_key": self.arb_key,
            "kalshi_ticker": self.kalshi_ticker,
            "poly_slug": self.poly_slug,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "max_profit_pct": self.max_profit_pct,
            "recent_profit_pct": list(self.recent_profit_pct),
            "confidence": self.confidence,
            "volume": self.volume,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfirmedArb:
        """Deserialize from a dict produced by :meth:`to_dict`.

        Missing optional fields fall back to sensible defaults so the store can
        load files written by older versions.

        Args:
            data: A mapping containing at least ``arb_key``.

        Returns:
            A :class:`ConfirmedArb` instance.
        """
        return cls(
            arb_key=str(data["arb_key"]),
            kalshi_ticker=str(data.get("kalshi_ticker", "")),
            poly_slug=str(data.get("poly_slug", "")),
            count=int(data.get("count", 1)),
            first_seen=str(data.get("first_seen", "")),
            last_seen=str(data.get("last_seen", "")),
            max_profit_pct=float(data.get("max_profit_pct", 0.0)),
            recent_profit_pct=[float(x) for x in data.get("recent_profit_pct", [])],
            confidence=data.get("confidence"),
            volume=(None if data.get("volume") is None else float(data["volume"])),
            extra=dict(data.get("extra") or {}),
        )


class ConfirmedArbStore:
    """Durable, unbounded JSON-backed store of observed arbitrage opportunities.

    Entries are keyed by ``arb_key`` and upserted on every :meth:`record` call.
    The in-memory cache is loaded lazily from disk on first access and every
    mutation is persisted atomically. The store keeps everything forever unless
    :meth:`prune` is called explicitly.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize the store.

        Args:
            path: Explicit on-disk location. When ``None``, the
                ``PFM_ARB_CONFIRMED_STORE`` env var is consulted, then
                :data:`DEFAULT_STORE_PATH`.
        """
        self._path = _resolve_path(path)
        self._cache: dict[str, ConfirmedArb] | None = None

    @property
    def path(self) -> Path:
        """The resolved on-disk path for this store."""
        return self._path

    # -- persistence -------------------------------------------------------

    def _load(self) -> dict[str, ConfirmedArb]:
        """Load (and cache) the store from disk.

        A missing or corrupt file yields an empty store rather than raising,
        so a partially written file from a crash can never break startup.
        """
        if self._cache is not None:
            return self._cache
        cache: dict[str, ConfirmedArb] = {}
        try:
            raw = self._path.read_text()
            payload = json.loads(raw)
            entries = payload.get("entries", []) if isinstance(payload, dict) else payload
            if isinstance(entries, list):
                for item in entries:
                    if not isinstance(item, dict) or "arb_key" not in item:
                        continue
                    try:
                        arb = ConfirmedArb.from_dict(item)
                    except (KeyError, TypeError, ValueError):
                        continue
                    cache[arb.arb_key] = arb
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            cache = {}
        self._cache = cache
        return cache

    def _persist(self) -> None:
        """Write the in-memory cache to disk atomically (temp file + replace)."""
        cache = self._cache if self._cache is not None else {}
        payload = {
            "version": 1,
            "updated_at": _now_iso(),
            "entries": [arb.to_dict() for arb in cache.values()],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        os.replace(tmp, self._path)

    def reload(self) -> None:
        """Drop the in-memory cache so the next access re-reads from disk."""
        self._cache = None

    # -- mutation ----------------------------------------------------------

    def record(
        self,
        arb_key: str,
        *,
        kalshi_ticker: str,
        poly_slug: str,
        profit_pct: float,
        volume: float | None = None,
        confidence: str | None = None,
        extra: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> ConfirmedArb:
        """Upsert an observation of an arbitrage opportunity.

        On first sight a new entry is created with ``count=1`` and ``first_seen``
        set. On a repeat sight ``count`` is incremented, ``last_seen`` and
        ``max_profit_pct`` are updated, and the latest profit is appended to the
        rolling ``recent_profit_pct`` window.

        Args:
            arb_key: Stable identity for the opportunity.
            kalshi_ticker: Kalshi side identifier.
            poly_slug: Polymarket side slug.
            profit_pct: Observed profit percentage for this sighting.
            volume: Optional observed volume (latest wins).
            confidence: Optional confidence label (latest wins).
            extra: Optional metadata merged into the entry's ``extra``.
            now: Optional timestamp override (for deterministic tests).

        Returns:
            The created or updated :class:`ConfirmedArb`.
        """
        cache = self._load()
        ts = _now_iso(now)
        profit = float(profit_pct)
        existing = cache.get(arb_key)
        if existing is None:
            arb = ConfirmedArb(
                arb_key=arb_key,
                kalshi_ticker=kalshi_ticker,
                poly_slug=poly_slug,
                count=1,
                first_seen=ts,
                last_seen=ts,
                max_profit_pct=profit,
                recent_profit_pct=[profit],
                confidence=confidence,
                volume=volume,
                extra=dict(extra or {}),
            )
            cache[arb_key] = arb
        else:
            existing.count += 1
            existing.last_seen = ts
            existing.max_profit_pct = max(existing.max_profit_pct, profit)
            existing.recent_profit_pct.append(profit)
            if len(existing.recent_profit_pct) > RECENT_PROFIT_MAXLEN:
                existing.recent_profit_pct = existing.recent_profit_pct[-RECENT_PROFIT_MAXLEN:]
            # Keep identity fields fresh; latest non-empty wins.
            if kalshi_ticker:
                existing.kalshi_ticker = kalshi_ticker
            if poly_slug:
                existing.poly_slug = poly_slug
            if volume is not None:
                existing.volume = volume
            if confidence is not None:
                existing.confidence = confidence
            if extra:
                existing.extra.update(extra)
            arb = existing
        self._persist()
        return arb

    def prune(self, max_age_days: float, *, now: datetime | None = None) -> int:
        """Drop entries whose ``last_seen`` is older than ``max_age_days``.

        This is optional housekeeping; the store is unbounded by default and
        never prunes on its own.

        Args:
            max_age_days: Maximum age in days; entries older than this are dropped.
            now: Optional reference time (for deterministic tests).

        Returns:
            The number of entries removed.
        """
        cache = self._load()
        reference = now if now is not None else datetime.now(UTC)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        cutoff_seconds = max_age_days * 86400.0
        removed: list[str] = []
        for key, arb in cache.items():
            try:
                seen = datetime.fromisoformat(arb.last_seen)
            except (ValueError, TypeError):
                continue
            if seen.tzinfo is None:
                seen = seen.replace(tzinfo=UTC)
            age = (reference - seen).total_seconds()
            if age > cutoff_seconds:
                removed.append(key)
        for key in removed:
            del cache[key]
        if removed:
            self._persist()
        return len(removed)

    # -- queries -----------------------------------------------------------

    def get(self, arb_key: str) -> ConfirmedArb | None:
        """Return the entry for ``arb_key``, or ``None`` if not present."""
        return self._load().get(arb_key)

    def all(self) -> list[ConfirmedArb]:
        """Return all entries, sorted by ``last_seen`` descending (newest first)."""
        entries = list(self._load().values())
        return sorted(entries, key=lambda a: a.last_seen, reverse=True)

    def confirmed(self, min_count: int = 3) -> list[ConfirmedArb]:
        """Return entries seen at least ``min_count`` times ("confirmed").

        Args:
            min_count: Minimum sighting count to qualify as confirmed.

        Returns:
            Qualifying entries sorted by ``last_seen`` descending.
        """
        return [a for a in self.all() if a.count >= min_count]

    def top_by_profit(self, n: int) -> list[ConfirmedArb]:
        """Return the ``n`` entries with the highest ``max_profit_pct``.

        Args:
            n: Maximum number of entries to return.

        Returns:
            Entries sorted by ``max_profit_pct`` descending (up to ``n``).
        """
        entries = sorted(self._load().values(), key=lambda a: a.max_profit_pct, reverse=True)
        return entries[: max(0, n)]

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the store.

        Returns:
            A dict with ``total_seen`` (sum of counts), ``n_confirmed``
            (entries with count >= 3), ``n_markets`` (distinct entries),
            ``oldest`` (earliest ``first_seen`` or ``None``), and ``newest``
            (latest ``last_seen`` or ``None``).
        """
        entries = list(self._load().values())
        total_seen = sum(a.count for a in entries)
        n_confirmed = sum(1 for a in entries if a.count >= 3)
        oldest = min((a.first_seen for a in entries), default=None)
        newest = max((a.last_seen for a in entries), default=None)
        return {
            "total_seen": total_seen,
            "n_confirmed": n_confirmed,
            "n_markets": len(entries),
            "oldest": oldest,
            "newest": newest,
        }

    def __len__(self) -> int:
        """Return the number of distinct entries currently in the store."""
        return len(self._load())
