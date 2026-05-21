"""Chained-monthly factor fetcher.

A *chained* factor is a sequence of single-source markets that, taken in
chronological order, form a continuous daily price series. The semantic
model is "next-print probability": at any date ``t``, the active segment
is the earliest one whose ``end`` date is ≥ ``t``. When that segment's
underlying market resolves (at its ``end``), the *next* segment becomes
active and contributes the data thereafter.

Concretely, for an ordered list of segments ``s_1, ..., s_n`` with end
dates ``e_1 < e_2 < ... < e_n`` and per-segment price series
``p_i: dates → [0, 1]`` (each fetched from its source over the full
window), the chained series is

    chain(t) = p_i(t)   where i = min{j : e_j ≥ t}     (1)

In words: the active segment at date ``t`` is the *earliest* segment that
hasn't yet expired. Segments contribute data on the open interval
``(e_{i-1}, e_i]`` (with ``e_0 := −∞``).

The function below fetches each segment's underlying history from its
proper source (Polymarket or Kalshi), filters each to its window, and
concatenates. Bars from segments that fall outside ``[start, end]`` are
discarded; bars where no segment is active are also discarded.

Edge cases:
    - If a segment returns no bars (market never opened, or all bars
      fall outside ``[start, end]``), the chain has a gap on that
      segment's window. Downstream alignment (strict / ffill) decides
      what to do with the gap.
    - Duplicate dates from segment overlap (rare — segment ends should be
      strictly ascending) are resolved by taking the *earlier* segment's
      value, since under (1) the active segment at that date is the
      earlier one.
    - Empty segments are tolerated; an entirely-empty chain returns an
      empty DataFrame with the union of upstream column names.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

import pandas as pd

from pfm.factors import ChainSegment
from pfm.sources.kalshi import KalshiClient
from pfm.sources.kalshi import fetch_factor_history as fetch_kalshi_history
from pfm.sources.polymarket import (
    PolymarketClient,
)
from pfm.sources.polymarket import (
    fetch_factor_history as fetch_polymarket_history,
)


def _segment_window(segments: Sequence[ChainSegment], i: int) -> tuple[date | None, date]:
    """Return the (open-from, closed-to) window for segment ``i`` per (1).

    The lower bound is the previous segment's end date (the active segment
    at exactly ``e_{i-1}`` is segment ``i-1``). For the first segment it
    is ``None`` (interpreted as -∞).
    """
    return (segments[i - 1].end if i > 0 else None, segments[i].end)


def _to_utc_date_ts(d: date) -> pd.Timestamp:
    """Convert a calendar date to a midnight-UTC Timestamp."""
    return pd.Timestamp(d.isoformat()).tz_localize("UTC")


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalize a Timestamp to UTC (works for naive *or* tz-aware inputs)."""
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _filter_segment_bars(
    df: pd.DataFrame, lower_excl: date | None, upper_incl: date
) -> pd.DataFrame:
    """Drop bars outside the segment's active window.

    The lower bound is *exclusive* (the previous segment's end belongs to
    the previous segment). The upper bound is *inclusive*. Bar dates are
    compared on calendar UTC dates, matching how upstream sources index.
    """
    if df.empty:
        return df
    idx_dates = df.index.normalize()
    mask = idx_dates <= _to_utc_date_ts(upper_incl)
    if lower_excl is not None:
        mask &= idx_dates > _to_utc_date_ts(lower_excl)
    return df.loc[mask]


def fetch_chained_history(
    segments: Sequence[ChainSegment],
    *,
    poly: PolymarketClient | None,
    kalshi: KalshiClient | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    polymarket_fetch=fetch_polymarket_history,
    kalshi_fetch=fetch_kalshi_history,
) -> pd.DataFrame:
    """Fetch the concatenated daily history of a chained factor.

    Args:
        segments: ordered list of :class:`ChainSegment`. Must be non-empty
            and in strictly ascending ``end`` order. (Validated upstream
            by :class:`FactorConfig`.)
        poly:    Polymarket client. Required iff any segment uses it.
        kalshi:  Kalshi client. Required iff any segment uses it.
        start:   global window lower bound (UTC ``pd.Timestamp``).
        end:     global window upper bound (UTC ``pd.Timestamp``).
        polymarket_fetch / kalshi_fetch: injection seams for tests.

    Returns:
        DataFrame indexed by UTC date with at least a ``price`` column.
        Empty DataFrame (with no columns) if every segment yields no bars.

    Raises:
        ValueError: ``segments`` empty or out of order.
        RuntimeError: a segment requires a client that wasn't provided.
    """
    if not segments:
        raise ValueError("fetch_chained_history requires at least one segment")
    ends = [s.end for s in segments]
    if ends != sorted(ends) or len(set(ends)) != len(ends):
        raise ValueError("segments must be strictly ascending by end date")

    pieces: list[pd.DataFrame] = []
    for i, seg in enumerate(segments):
        # Per-segment fetch covers the full upstream window; we'll filter
        # to the segment's active window below. This wastes some bytes but
        # keeps the underlying API calls trivial and cacheable.
        if seg.source == "polymarket":
            if poly is None:
                raise RuntimeError(
                    f"chain segment[{i}] needs Polymarket client (slug={seg.slug!r})"
                )
            df = polymarket_fetch(poly, slug=seg.slug, start=start, end=end)
        elif seg.source == "kalshi":
            if kalshi is None:
                raise RuntimeError(f"chain segment[{i}] needs Kalshi client (ticker={seg.slug!r})")
            df = kalshi_fetch(kalshi, market_ticker=seg.slug, start=start, end=end)
        else:  # pragma: no cover - validated by ChainSegment.__post_init__
            raise ValueError(f"unknown segment source: {seg.source!r}")

        lower_excl, upper_incl = _segment_window(segments, i)
        df = _filter_segment_bars(df, lower_excl, upper_incl)
        if not df.empty:
            pieces.append(df)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, axis=0)
    # Defensive: identical bar dates across two segments shouldn't happen
    # given strictly ascending ends and the exclusive lower-bound rule,
    # but if they do (e.g. floating timezone normalization) we keep the
    # earliest segment's value, matching the active-segment rule (1).
    out = out[~out.index.duplicated(keep="first")]
    out = out.sort_index()

    # Final clamp to the user's window.
    if start is not None:
        out = out[out.index >= _to_utc(pd.Timestamp(start)).normalize()]
    if end is not None:
        out = out[out.index <= _to_utc(pd.Timestamp(end)).normalize()]
    return out


def segments_signature(segments: Iterable[ChainSegment]) -> str:
    """Stable cache-key fragment for a chain's segment list.

    Includes ``source|slug|end`` per segment so two factors with the same
    ``id`` but different chain compositions don't collide in cache.
    """
    parts = [f"{s.source}|{s.slug}|{s.end.isoformat()}" for s in segments]
    return ";".join(parts)
