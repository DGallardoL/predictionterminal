"""Read and validate ``factors.yml``.

Three factor shapes are supported:

1.  **Single-source prediction-market factor** — the original form.
    ``source`` is one of ``polymarket`` / ``kalshi`` / ``manifold`` /
    ``predictit`` and ``slug`` is the per-source identifier (URL slug
    for Polymarket / Manifold, market ticker for Kalshi, market id for
    PredictIt). All four return a probability series in ``[0, 1]``.

2.  **Macro / level factor** — ``source`` is ``bls`` or ``fred``. The
    underlying series is a level (yields, rates, indices, jobless claim
    counts, …), so ``is_probability`` should be set to ``False``. The
    factor model dispatches non-probability sources through plain
    differencing instead of the logit transform.

3.  **Chained-monthly factor** — ``source: chain``. Concatenates a
    chronologically-ordered list of sub-markets (each with its own source
    and slug) into one continuous daily price series. Used for the
    forward-rolling macro indicators where each monthly print resolves a
    sub-market and the *next* one becomes the active source. See
    ``pfm.sources.chain.fetch_chained_history``.

A small dispatch helper :func:`fetch_factor_history_dispatch` is exposed
so callers don't have to fan out the source switch themselves; it always
returns a ``DataFrame`` indexed by UTC date with a single ``price`` column.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Probability-shaped sources (each contract trades in ``[0, 1]``).
PROBABILITY_SOURCES: frozenset[str] = frozenset({"polymarket", "kalshi", "manifold", "predictit"})
# Level-shaped macro sources (yields, indices, claim counts, …). The
# ``sentiment`` source is level-shaped too: per-day mean of signed news
# sentiment in ``[-1, +1]``. The design assembler dispatches it to
# :func:`pfm.model.delta_level` so the regressor is the *change* in
# sentiment, not the raw level (which is non-stationary).
LEVEL_SOURCES: frozenset[str] = frozenset({"bls", "fred", "sentiment"})
# Sources that name a single underlying market — kept as a name so the
# chain validator (see :class:`ChainSegment` below) keeps working.
SINGLE_SOURCES: frozenset[str] = PROBABILITY_SOURCES | LEVEL_SOURCES
# Source name reserved for the chained-multi-segment shape.
CHAIN_SOURCE: str = "chain"
# All recognised top-level source names.
KNOWN_SOURCES: frozenset[str] = SINGLE_SOURCES | {CHAIN_SOURCE}
# Subset of sources that are valid inside a chained-segment list.
CHAIN_SEGMENT_SOURCES: frozenset[str] = frozenset({"polymarket", "kalshi"})


@dataclass(frozen=True)
class ChainSegment:
    """One segment of a chained factor.

    Each segment contributes daily price bars from ``(prev.end, end]`` —
    i.e. **exclusive** of the previous segment's last day, **inclusive** of
    its own ``end``. The first segment contributes from the earliest bar
    available up to and including its ``end``.

    Attributes:
        source: ``polymarket`` or ``kalshi``. Determines the fetcher.
        slug: per-source identifier (URL slug or market ticker).
        end: last UTC calendar date this segment is the active source.
            Conventionally the resolution date of the underlying market.
        name: optional human-readable label (purely descriptive).
    """

    source: str
    slug: str
    end: date
    name: str = ""

    def __post_init__(self) -> None:
        if self.source not in CHAIN_SEGMENT_SOURCES:
            raise ValueError(
                f"chain segment source must be one of {sorted(CHAIN_SEGMENT_SOURCES)}, "
                f"got {self.source!r}"
            )
        if not self.slug:
            raise ValueError("chain segment slug must be non-empty")
        if not isinstance(self.end, date):
            raise ValueError(f"chain segment end must be a date, got {type(self.end)}")


@dataclass(frozen=True)
class FactorConfig:
    """A single factor entry from ``factors.yml``.

    For ``source`` ∈ {polymarket, kalshi, manifold, predictit} the ``slug``
    field carries the per-source identifier and ``segments`` is empty.

    For ``source`` ∈ {bls, fred} ``slug`` is the macro series id (the same
    string :mod:`pfm.sources.bls` / :mod:`pfm.sources.fred` accept), or
    ``series_id`` may be set explicitly to override ``slug`` for that
    purpose. ``is_probability`` defaults to ``False`` for these sources.

    For ``source == 'chain'`` the ``slug`` is a stable label used for
    cache keys and surfacing in the API; the actual fetching uses
    ``segments`` (≥1 entries, in strictly ascending ``end`` order).

    Attributes:
        is_probability: When ``True`` the underlying series trades in
            ``[0, 1]`` and the factor model applies the standard
            ``Δlogit`` transform (clip + logit + first difference). When
            ``False`` (typical for macro level series — yields, indices,
            jobless-claim counts) the model falls back to plain
            first-differencing. Defaults preserve backward compatibility
            for prediction-market sources.
        series_id: Optional explicit identifier for level sources. If
            unset, ``slug`` doubles as the series id.
    """

    id: str
    name: str
    slug: str
    source: str
    description: str
    theme: str = "other"
    segments: tuple[ChainSegment, ...] = field(default_factory=tuple)
    is_probability: bool = True
    series_id: str | None = None

    def __post_init__(self) -> None:
        if self.source == CHAIN_SOURCE:
            if not self.segments:
                raise ValueError(f"factor {self.id!r}: source=chain requires non-empty segments")
            ends = [s.end for s in self.segments]
            if ends != sorted(ends):
                raise ValueError(
                    f"factor {self.id!r}: chain segments must be in ascending end-date order"
                )
            if len(set(ends)) != len(ends):
                raise ValueError(f"factor {self.id!r}: chain segments must have unique end dates")
        else:
            if self.source not in SINGLE_SOURCES:
                raise ValueError(
                    f"factor {self.id!r}: source must be one of "
                    f"{sorted(KNOWN_SOURCES)}, got {self.source!r}"
                )
            if self.segments:
                raise ValueError(f"factor {self.id!r}: only source=chain may carry segments")
        # Macro / level sources are not probabilities; refuse the
        # combination ``source=bls/fred`` + ``is_probability=True`` so
        # downstream code never tries to logit-transform a bond yield.
        if self.source in LEVEL_SOURCES and self.is_probability:
            raise ValueError(
                f"factor {self.id!r}: source={self.source!r} requires "
                "is_probability=false (level series, not [0,1] probabilities)"
            )

    @property
    def is_chained(self) -> bool:
        return self.source == CHAIN_SOURCE

    @property
    def effective_series_id(self) -> str:
        """Return ``series_id`` if set, otherwise ``slug``."""
        return self.series_id or self.slug


def _coerce_date(value: object, *, ctx: str) -> date:
    """YAML parses an ISO date as ``datetime.date`` already, but be defensive."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as e:
            raise ValueError(f"{ctx}: invalid date {value!r}: {e}") from e
    raise ValueError(f"{ctx}: expected date, got {type(value).__name__}: {value!r}")


def _build_segments(entry: dict, factor_id: str) -> tuple[ChainSegment, ...]:
    """Parse the ``segments:`` list under a chain factor."""
    raw_segments = entry.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"factor {factor_id!r}: segments must be a non-empty list")
    out: list[ChainSegment] = []
    required = {"source", "slug", "end"}
    for i, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            raise ValueError(f"factor {factor_id!r}: segment[{i}] must be a mapping, got {seg!r}")
        missing = required - seg.keys()
        if missing:
            raise ValueError(f"factor {factor_id!r}: segment[{i}] missing keys: {sorted(missing)}")
        out.append(
            ChainSegment(
                source=str(seg["source"]),
                slug=str(seg["slug"]),
                end=_coerce_date(seg["end"], ctx=f"factor {factor_id!r} segment[{i}].end"),
                name=str(seg.get("name", "")),
            )
        )
    return tuple(out)


def load_factors(path: Path) -> dict[str, FactorConfig]:
    """Load factor definitions keyed by ``id``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is malformed (missing keys, dup ids,
            invalid chain segment shape, segments out of order, etc).
    """
    if not path.exists():
        raise FileNotFoundError(f"factors file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    items = raw.get("factors", [])
    if not isinstance(items, list):
        raise ValueError("`factors` key must be a list")

    out: dict[str, FactorConfig] = {}
    required = {"id", "name", "slug", "source", "description"}
    for entry in items:
        if not isinstance(entry, dict):
            raise ValueError(f"each factor must be a mapping, got {entry!r}")
        missing = required - entry.keys()
        if missing:
            raise ValueError(f"factor {entry.get('id', '?')!r} missing keys: {sorted(missing)}")
        fid = str(entry["id"])
        if fid in out:
            raise ValueError(f"duplicate factor id: {fid!r}")
        source = str(entry["source"])
        segments: tuple[ChainSegment, ...] = ()
        if source == CHAIN_SOURCE:
            segments = _build_segments(entry, fid)
        # Per-source default for ``is_probability``: PM/Kalshi/Manifold/
        # PredictIt + chained probabilities default to True; BLS/FRED to
        # False. The YAML can always override explicitly.
        default_is_prob = source not in LEVEL_SOURCES
        is_prob_raw = entry.get("is_probability", default_is_prob)
        if not isinstance(is_prob_raw, bool):
            raise ValueError(
                f"factor {fid!r}: is_probability must be a bool, got {type(is_prob_raw).__name__}"
            )
        series_id_raw = entry.get("series_id")
        series_id = str(series_id_raw) if series_id_raw is not None else None
        out[fid] = FactorConfig(
            id=fid,
            name=str(entry["name"]),
            slug=str(entry["slug"]),
            source=source,
            description=str(entry["description"]).strip(),
            theme=str(entry.get("theme", "other")),
            segments=segments,
            is_probability=is_prob_raw,
            series_id=series_id,
        )
    return out


# ---------------------------------------------------------------------------
# Unified history-fetch dispatcher.
#
# Every fetcher returns a :class:`pandas.DataFrame` indexed by UTC date with
# at least one ``price`` column. Callers downstream (``pfm.main``,
# ``terminal_*`` modules) consume this single contract — they do not care
# whether the underlying source is a prediction-market venue or a macro feed.
# ---------------------------------------------------------------------------


def _normalise_history_frame(df: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    """Coerce a fetcher's frame to ``[date_index_utc, price]``.

    ``value_column`` names the column already on the frame whose values we
    treat as the factor's price/level (e.g. ``"prob"`` for Manifold,
    ``"value"`` for BLS/FRED). The returned frame has ``"price"`` only and
    a UTC ``DatetimeIndex`` named ``"date"``.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["price"])
    out = df.copy()
    if "date" in out.columns:
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index, utc=True)
    if value_column not in out.columns and "price" in out.columns:
        # Already in canonical shape.
        out = out[["price"]]
    else:
        if value_column not in out.columns:
            raise ValueError(
                f"normalise: expected column {value_column!r} on frame with "
                f"columns={list(out.columns)}"
            )
        out = out[[value_column]].rename(columns={value_column: "price"})
    out.index.name = "date"
    return out.sort_index()


def _run_async(coro: Any) -> Any:
    """Run an awaitable to completion from sync code, even if a loop exists.

    Used by the dispatcher to call the async Manifold / PredictIt clients.
    Falls back to ``asyncio.run`` when no loop is running and to a fresh
    one-shot loop when called from inside a running loop (rare in the
    sync ``/fit`` path, but the test suite occasionally mixes contexts).
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # We're inside a running loop: schedule the coroutine on a
        # private loop in a worker thread so we don't reentrancy-deadlock.
        import threading

        result_box: dict[str, Any] = {}

        def _runner() -> None:
            new_loop = asyncio.new_event_loop()
            try:
                result_box["v"] = new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        return result_box.get("v")
    return asyncio.run(coro)


def fetch_factor_history_dispatch(
    fc: FactorConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    polymarket_client: Any | None = None,
    kalshi_client: Any | None = None,
    manifold_client: Any | None = None,
    predictit_client: Any | None = None,
    bls_client: Any | None = None,
    fred_fetcher: Any | None = None,
) -> pd.DataFrame:
    """Fetch a factor's daily history regardless of the underlying source.

    All source clients are optional — when ``None`` the dispatcher
    instantiates the default client / falls back to the cached
    module-level fetcher. Returns a ``DataFrame`` with a UTC
    ``DatetimeIndex`` named ``date`` and a single ``price`` column.

    The chained-segment path (``source == 'chain'``) is intentionally
    *not* re-implemented here; the existing ``pfm.main._cached_factor_history``
    handles it because it already integrates with the long-running
    request cache. Calling the dispatcher with a chain factor raises.
    """
    if fc.source == CHAIN_SOURCE:
        raise ValueError(
            f"factor {fc.id!r}: chain factors are not handled by "
            "fetch_factor_history_dispatch (use pfm.sources.chain.fetch_chained_history)"
        )

    if fc.source == "polymarket":
        from pfm.sources.polymarket import PolymarketClient
        from pfm.sources.polymarket import fetch_factor_history as _poly_fetch

        client = polymarket_client or PolymarketClient()
        df = _poly_fetch(client, fc.slug, start=start, end=end)
        return _normalise_history_frame(df, value_column="price")

    if fc.source == "kalshi":
        from pfm.sources.kalshi import KalshiClient
        from pfm.sources.kalshi import fetch_factor_history as _kalshi_fetch

        client = kalshi_client or KalshiClient()
        df = _kalshi_fetch(client, fc.slug, start=start, end=end)
        return _normalise_history_frame(df, value_column="price")

    if fc.source == "manifold":
        from pfm.sources.manifold import ManifoldClient

        async def _go() -> pd.DataFrame:
            owns = manifold_client is None
            cli = manifold_client or ManifoldClient()
            try:
                # Manifold's fetch_history takes ``market_id`` (we pass the
                # slug — it tries /slug first via get_market then falls back
                # to /market/{id}). Resolve to the contract id and forward.
                market = await cli.get_market(fc.slug)
                mid = str(market.get("id") or fc.slug)
                # Choose ``days`` from the requested window (or 365 as a
                # sane default when the window is unbounded).
                window_days = max(1, int((end - start).days)) if start and end else 365
                return await cli.fetch_history(mid, days=window_days)
            finally:
                if owns:
                    await cli.close()

        df = _run_async(_go())
        return _normalise_history_frame(df, value_column="prob")

    if fc.source == "predictit":
        from pfm.sources.predictit import PredictItClient

        async def _go() -> pd.DataFrame:
            owns = predictit_client is None
            cli = predictit_client or PredictItClient()
            try:
                market_id = int(fc.slug)
                window_days = max(1, int((end - start).days)) if start and end else 365
                return await cli.fetch_history(market_id, days=window_days)
            finally:
                if owns:
                    await cli.close()

        df = _run_async(_go())
        return _normalise_history_frame(df, value_column="prob")

    if fc.source == "bls":
        from pfm.sources.bls import BLSClient

        client = bls_client or BLSClient()
        df = client.fetch(
            fc.effective_series_id,
            int(start.year),
            int(end.year),
        )
        # BLS frame is ``[date, value]`` already.
        return _normalise_history_frame(df, value_column="value")

    if fc.source == "fred":
        from pfm.sources.fred import fetch_fred_series_cached

        fetcher = fred_fetcher or fetch_fred_series_cached
        s = fetcher(fc.effective_series_id, start, end, transform="raw")
        if s is None or len(s) == 0:
            return pd.DataFrame(columns=["price"])
        frame = s.rename("price").to_frame()
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.index.name = "date"
        return frame

    if fc.source == "sentiment":
        # The slug (or effective_series_id) carries the free-text search
        # query — e.g. ``"bitcoin"``, ``"federal reserve"``. Returns a
        # daily ``[-1, +1]`` series; :func:`pfm.model.delta_level` is
        # applied downstream in the design assembler.
        from pfm.sources.sentiment_factor import fetch_sentiment_history

        return fetch_sentiment_history(fc.effective_series_id, start=start, end=end)

    raise ValueError(f"factor {fc.id!r}: unknown source {fc.source!r}")
