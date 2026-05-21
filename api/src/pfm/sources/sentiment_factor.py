"""News-sentiment regression factor source.

Exposes a single class, :class:`SentimentFactorSource`, that aggregates
headline sentiment from three free public feeds (GDELT timelinetone,
Reddit search.json, HN Algolia) into one number per UTC calendar day in
the signed ``[-1, +1]`` range.

The series is shaped so it slots into :func:`pfm.regression_core._assemble_design`
through the same path as macro-level factors (BLS / FRED): the factor
config carries ``is_probability=False`` and the design pipeline calls
:func:`pfm.model.delta_level` on the resulting series so the regressor
is *sentiment change*, not the raw level. That keeps it directionally
comparable to the Δlogit Polymarket factors and avoids the obvious
non-stationarity problem of regressing returns on a level.

Why a *factor* source rather than a Terminal feature
----------------------------------------------------
The existing ``terminal/{news,gdelt_news,sentiment_nlp,rss_news}``
modules score sentiment **per article inside one market's panel** —
they assume a Polymarket slug + the small (≤20 article) window the
UI shows. The factor pipeline needs the opposite: one number per day,
across the whole ``[start, end]`` window of a /fit, for an arbitrary
search query the user types ("bitcoin", "trump", "fed", …). Reusing
those panels would mean making N calls (one per date) — instead we
hit GDELT's ``mode=timelinetone`` which already returns a daily-binned
average tone time series in one round-trip, then layer Reddit/HN
samples on top for the recent window where their data exists.

Fetching contract
-----------------
* :meth:`SentimentFactorSource.get_daily_sentiment(query, days)` returns
  a ``pd.Series`` indexed by UTC midnight ``DatetimeIndex`` with mean
  signed sentiment per day in ``[-1, +1]``. Missing days are absent
  from the index (callers should expect gaps and drop them).

* The series carries the query string as ``Series.name`` so the design
  assembler can rename it to the user's factor id without losing the
  origin.

Caching
-------
Two layers:
  * In-process :class:`pfm.cache_utils.TerminalCache` under namespace
    ``"sentiment_factor"`` with TTL = 900 s (15 min). Keyed by
    ``(query, days)`` so the same /fit re-run within 15 minutes is
    instant.
  * The HTTP client is reused across calls — no per-call connection
    setup.

No new dependencies — we use ``httpx`` (already imported), ``pandas``
(already imported), and :mod:`pfm.terminal.sentiment_nlp.score_headline`
(already shipped).
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from typing import Any

import httpx
import pandas as pd

from pfm.cache_utils import TerminalCache, get_cache
from pfm.terminal.sentiment_nlp import score_headline

logger = logging.getLogger(__name__)

# --- module-level constants -------------------------------------------------

#: GDELT 2.0 DOC API base. Same endpoint as :mod:`pfm.terminal.gdelt_news`.
GDELT_DOC_URL: str = "https://api.gdeltproject.org/api/v2/doc/doc"

#: Reddit search endpoint (no auth, public).
REDDIT_SEARCH_URL: str = "https://www.reddit.com/search.json"

#: HN Algolia search endpoint (no auth, public).
HN_SEARCH_URL: str = "https://hn.algolia.com/api/v1/search"

#: User-Agent header — Reddit rejects requests without one.
USER_AGENT: str = "polymarket-terminal/1.0 (sentiment-factor)"

#: Cache namespace + default TTL.
CACHE_NAMESPACE: str = "sentiment_factor"
CACHE_TTL_SECONDS: int = 900  # 15 minutes — matches the spec.

#: How many Reddit / HN hits to ask for per query. The relevance filter
#: in the upstream APIs is decent at this volume; pulling >100 buys
#: very little extra signal but doubles the latency.
_REDDIT_PAGE: int = 50
_HN_PAGE: int = 50

#: Default HTTP timeout. Each upstream is independently bounded so a
#: slow source can't block the others.
_HTTP_TIMEOUT_SECONDS: float = 8.0

#: How many trailing seconds of "recent" articles to scan from Reddit / HN
#: when ``days`` exceeds Reddit's natural ``t=month`` ceiling. Beyond
#: this we rely on GDELT alone (it has multi-year recall).
_REDDIT_MAX_DAYS: int = 31
_HN_MAX_DAYS: int = 90


#: Curated factor catalog. These are the queries the ``/factors?source=sentiment``
#: endpoint advertises. Each entry yields one synthetic :class:`FactorConfig`
#: that the resolver wires into the design matrix when the user names it.
#:
#: Keys are the factor *id* (used in /fit ``factors=[...]``); values are the
#: search query passed to the upstream APIs and the human-readable name.
CURATED_QUERIES: dict[str, dict[str, str]] = {
    "sentiment_bitcoin": {
        "query": "bitcoin",
        "name": "News sentiment: Bitcoin",
        "description": "Daily mean signed sentiment of headlines mentioning bitcoin (GDELT + Reddit + HN, VADER+finance-lex blended).",
    },
    "sentiment_trump": {
        "query": "trump",
        "name": "News sentiment: Trump",
        "description": "Daily mean signed sentiment of headlines mentioning Trump.",
    },
    "sentiment_fed": {
        "query": "federal reserve",
        "name": "News sentiment: Federal Reserve",
        "description": "Daily mean signed sentiment of headlines mentioning the Federal Reserve / FOMC.",
    },
    "sentiment_china": {
        "query": "china",
        "name": "News sentiment: China",
        "description": "Daily mean signed sentiment of headlines mentioning China.",
    },
    "sentiment_oil": {
        "query": "oil",
        "name": "News sentiment: Oil",
        "description": "Daily mean signed sentiment of headlines mentioning oil / crude / OPEC.",
    },
    "sentiment_tesla": {
        "query": "tesla",
        "name": "News sentiment: Tesla",
        "description": "Daily mean signed sentiment of headlines mentioning Tesla.",
    },
    "sentiment_nvidia": {
        "query": "nvidia",
        "name": "News sentiment: NVIDIA",
        "description": "Daily mean signed sentiment of headlines mentioning NVIDIA.",
    },
    "sentiment_recession": {
        "query": "recession",
        "name": "News sentiment: Recession",
        "description": "Daily mean signed sentiment of headlines mentioning recession.",
    },
    "sentiment_ukraine": {
        "query": "ukraine",
        "name": "News sentiment: Ukraine",
        "description": "Daily mean signed sentiment of headlines mentioning Ukraine.",
    },
    "sentiment_israel": {
        "query": "israel",
        "name": "News sentiment: Israel",
        "description": "Daily mean signed sentiment of headlines mentioning Israel.",
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seendate_to_ts(raw: str) -> pd.Timestamp | None:
    """Parse a GDELT ``seendate``/``date`` (``YYYYMMDDTHHMMSSZ``) → UTC Timestamp.

    GDELT timelinetone returns dates in the same compact form. ``None`` on
    any parse failure so callers can simply ``continue``.
    """
    s = (raw or "").strip()
    if not s:
        return None
    try:
        # Accept either the compact "20260101T120000Z" form or an already
        # ISO-8601 string.
        if "T" in s and len(s) >= 15 and "-" not in s:
            date, _, rest = s.partition("T")
            iso = f"{date[0:4]}-{date[4:6]}-{date[6:8]}T{rest[0:2]}:{rest[2:4]}:{rest[4:6]}Z"
            ts = pd.Timestamp(iso)
        else:
            ts = pd.Timestamp(s)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except (ValueError, TypeError):
        return None


def _ts_from_unix(unix_seconds: Any) -> pd.Timestamp | None:
    """Reddit + HN timestamps come back as unix-seconds floats / ints."""
    if unix_seconds is None:
        return None
    try:
        return pd.Timestamp(int(unix_seconds), unit="s", tz="UTC")
    except (ValueError, TypeError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------


class SentimentFactorSource:
    """Pull daily-aggregated news sentiment for an arbitrary text query.

    The class is designed to be cheap to construct (no IO at __init__)
    and safe to share across requests — every public method is read-only
    and the HTTP client is created lazily on first use.

    Args:
        client: Optional pre-built :class:`httpx.Client`. Re-using the
            poly client's underlying client is the typical pattern in
            other terminal_* modules; passing ``None`` makes us build
            our own with the standard timeout.
        cache: Optional :class:`TerminalCache` override (test injection).
            Defaults to the process-wide ``"sentiment_factor"`` cache.

    Example::

        src = SentimentFactorSource()
        series = src.get_daily_sentiment("bitcoin", days=60)
        # series is a UTC-indexed Series of mean signed sentiment in [-1, +1]
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        cache: TerminalCache | None = None,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._cache = (
            cache
            if cache is not None
            else get_cache(
                CACHE_NAMESPACE,
                ttl=CACHE_TTL_SECONDS,
            )
        )

    # --- lifecycle ---------------------------------------------------------

    def _http(self) -> httpx.Client:
        """Lazy-init the HTTP client. Reused across all subsequent fetches."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=_HTTP_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
        return self._client

    def close(self) -> None:
        """Close the owned HTTP client. Safe to call multiple times."""
        if self._owns_client and self._client is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                self._client.close()
            self._client = None

    # --- public API --------------------------------------------------------

    def get_daily_sentiment(self, query: str, days: int) -> pd.Series:
        """Return a daily series of mean signed sentiment for ``query``.

        Args:
            query: free-text search query (e.g. ``"bitcoin"``,
                ``"federal reserve"``). Passed verbatim to GDELT and as
                the Reddit / HN ``q``.
            days: how many trailing days to fetch. The series may be
                shorter than this when upstreams return no data on some
                days; callers should ``dropna`` rather than reindex.

        Returns:
            A ``pd.Series`` indexed by UTC midnight ``Timestamp`` with
            float values in ``[-1.0, +1.0]``. Empty if every upstream
            returned nothing. ``Series.name`` is set to ``query`` so
            callers can rename it for the design matrix without losing
            provenance.

        The aggregation rule per day is the unweighted mean across:
          * GDELT timelinetone bucketed by UTC date (rescaled from
            ``[-10, +10]`` to ``[-1, +1]``),
          * any Reddit headlines from that day, scored via
            :func:`score_headline`,
          * any HN headlines from that day, scored via
            :func:`score_headline`.

        The result is hard-clipped to ``[-1, +1]`` so an outlier headline
        can't blow up the column.
        """
        if not query or not isinstance(query, str):
            return pd.Series(dtype=float, name=query or "")
        days = max(1, int(days))
        cache_key = ("sentiment", query, days)
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Stored as a list of (iso_date, value) tuples so it's JSON-safe
            # even when a future test backs the cache with Redis.
            idx = pd.to_datetime([d for d, _ in cached], utc=True)
            vals = [v for _, v in cached]
            return pd.Series(vals, index=idx, name=query, dtype=float)

        client = self._http()
        # Per-day buckets of scores from each source.
        buckets: dict[pd.Timestamp, list[float]] = defaultdict(list)

        # 1. GDELT timelinetone — daily-binned, multi-year recall.
        for ts, tone in self._fetch_gdelt_timeline(client, query, days):
            # Map GDELT's [-10, +10] tone into our [-1, +1] convention by
            # /10 then clip — score_headline uses the same mapping for its
            # ``external_tone`` channel, so the two surfaces stay consistent.
            scaled = max(-1.0, min(1.0, float(tone) / 10.0))
            buckets[ts.normalize()].append(scaled)

        # 2. Reddit — only useful within the last ~month.
        if days <= _REDDIT_MAX_DAYS * 2:  # generous gate: still try if a bit over
            for ts, title in self._fetch_reddit(client, query):
                day = ts.normalize()
                # Drop posts older than the requested window — Reddit's
                # ``t=month`` is broad and we don't want stale headlines
                # contaminating a 5-day window.
                cutoff = pd.Timestamp.utcnow().tz_convert("UTC").normalize() - pd.Timedelta(
                    days=days
                )
                if day < cutoff:
                    continue
                score, _label = score_headline(title)
                if abs(score) > 1e-9:
                    buckets[day].append(score)

        # 3. HN — has up to 90d recall.
        if days <= _HN_MAX_DAYS * 2:
            for ts, title in self._fetch_hn(client, query, days):
                day = ts.normalize()
                cutoff = pd.Timestamp.utcnow().tz_convert("UTC").normalize() - pd.Timedelta(
                    days=days
                )
                if day < cutoff:
                    continue
                score, _label = score_headline(title)
                if abs(score) > 1e-9:
                    buckets[day].append(score)

        if not buckets:
            empty = pd.Series(dtype=float, name=query)
            # Cache the empty result too — re-asking won't help.
            self._cache.set(cache_key, [], ttl=CACHE_TTL_SECONDS)
            return empty

        # Mean-aggregate + clip + sort.
        items = sorted(buckets.items(), key=lambda kv: kv[0])
        idx = pd.DatetimeIndex([d for d, _ in items], tz="UTC")
        vals = [max(-1.0, min(1.0, sum(scores) / len(scores))) for _d, scores in items]
        series = pd.Series(vals, index=idx, name=query, dtype=float)

        # Persist a JSON-safe form so the cache backend stays portable.
        payload = [(ts.isoformat(), float(v)) for ts, v in series.items()]
        self._cache.set(cache_key, payload, ttl=CACHE_TTL_SECONDS)
        return series

    # --- per-upstream fetchers --------------------------------------------

    def _fetch_gdelt_timeline(
        self,
        client: httpx.Client,
        query: str,
        days: int,
    ) -> list[tuple[pd.Timestamp, float]]:
        """Hit GDELT ``mode=timelinetone`` and return ``[(ts, tone), …]``.

        Tone is the GDELT-native ``[-10, +10]`` number — the caller
        rescales. Empty list on any failure / throttle.
        """
        params: dict[str, str | int] = {
            "query": query,
            "mode": "timelinetone",
            "format": "json",
            "timespan": f"{int(days)}d",
        }
        try:
            r = client.get(GDELT_DOC_URL, params=params)
        except httpx.HTTPError as e:
            logger.warning("gdelt timelinetone fetch failed (%s): %s", query, e)
            return []
        if r.status_code >= 400:
            logger.warning(
                "gdelt timelinetone non-2xx (%s): %s body=%s",
                query,
                r.status_code,
                (r.text or "")[:200],
            )
            return []
        body = r.text or ""
        if body.lstrip().startswith("Please limit"):
            logger.warning("gdelt timelinetone throttled (%s)", query)
            return []
        try:
            payload = r.json()
        except ValueError:
            return []
        if not isinstance(payload, dict):
            return []
        timeline = payload.get("timeline") or []
        if not isinstance(timeline, list):
            return []
        # Find the "Average Tone" series; fall back to the first one.
        data_points: list[dict] = []
        for entry in timeline:
            if not isinstance(entry, dict):
                continue
            d = entry.get("data")
            if isinstance(d, list) and d:
                if entry.get("series") == "Average Tone" or not data_points:
                    data_points = d
                    if entry.get("series") == "Average Tone":
                        break
        out: list[tuple[pd.Timestamp, float]] = []
        for pt in data_points:
            if not isinstance(pt, dict):
                continue
            ts = _seendate_to_ts(str(pt.get("date") or ""))
            val = pt.get("value")
            if ts is None or val is None:
                continue
            try:
                out.append((ts, float(val)))
            except (TypeError, ValueError):
                continue
        return out

    def _fetch_reddit(
        self,
        client: httpx.Client,
        query: str,
    ) -> list[tuple[pd.Timestamp, str]]:
        """Hit Reddit search.json for ``query``. Returns ``[(ts, title), …]``."""
        params: dict[str, str | int] = {
            "q": query,
            "sort": "new",
            "limit": _REDDIT_PAGE,
            "t": "month",
        }
        try:
            r = client.get(REDDIT_SEARCH_URL, params=params)
        except httpx.HTTPError as e:
            logger.warning("reddit fetch failed (%s): %s", query, e)
            return []
        if r.status_code >= 400:
            logger.warning(
                "reddit non-2xx (%s): %s",
                query,
                r.status_code,
            )
            return []
        try:
            payload = r.json()
        except ValueError:
            return []
        out: list[tuple[pd.Timestamp, str]] = []
        for child in (payload.get("data") or {}).get("children", []):
            if not isinstance(child, dict):
                continue
            d = child.get("data") or {}
            title = str(d.get("title") or "").strip()
            ts = _ts_from_unix(d.get("created_utc"))
            if title and ts is not None:
                out.append((ts, title))
        return out

    def _fetch_hn(
        self,
        client: httpx.Client,
        query: str,
        days: int,
    ) -> list[tuple[pd.Timestamp, str]]:
        """Hit HN Algolia /search?tags=story for ``query``."""
        since_unix = int(time.time()) - max(1, int(days)) * 86400
        params: dict[str, str | int] = {
            "query": query,
            "tags": "story",
            "hitsPerPage": _HN_PAGE,
            "numericFilters": f"created_at_i>{since_unix}",
        }
        try:
            r = client.get(HN_SEARCH_URL, params=params)
        except httpx.HTTPError as e:
            logger.warning("hn fetch failed (%s): %s", query, e)
            return []
        if r.status_code >= 400:
            logger.warning("hn non-2xx (%s): %s", query, r.status_code)
            return []
        try:
            payload = r.json()
        except ValueError:
            return []
        out: list[tuple[pd.Timestamp, str]] = []
        for hit in payload.get("hits") or []:
            if not isinstance(hit, dict):
                continue
            title = str(hit.get("title") or hit.get("story_title") or "").strip()
            ts = _ts_from_unix(hit.get("created_at_i"))
            if title and ts is not None:
                out.append((ts, title))
        return out


# ---------------------------------------------------------------------------
# convenience helpers — used by the factor-resolver and the dispatcher
# ---------------------------------------------------------------------------


def parse_sentiment_factor_id(raw: str) -> tuple[bool, str | None]:
    """Detect the ``sentiment:<query>`` factor-id syntax.

    Returns ``(is_sentiment, query)``. ``query`` is the user-typed
    search string with the ``sentiment:`` prefix stripped and surrounding
    whitespace removed — empty / blank queries return ``(True, None)``
    so the caller can raise a 400 with a helpful message.

    Curated ids (``sentiment_<topic>``) are *not* matched here — those
    flow through the normal yaml-catalog path.
    """
    if not raw or not isinstance(raw, str):
        return False, None
    if not raw.lower().startswith("sentiment:"):
        return False, None
    query = raw.split(":", 1)[1].strip()
    return True, (query or None)


def curated_sentiment_query(factor_id: str) -> str | None:
    """Return the search query for a curated sentiment factor id, or ``None``."""
    entry = CURATED_QUERIES.get(factor_id)
    if entry is None:
        return None
    return entry["query"]


def _global_source() -> SentimentFactorSource:
    """Return a process-wide singleton :class:`SentimentFactorSource`.

    Used by the dispatcher path so we don't construct a fresh httpx
    client + cache lookup on every /fit call. Tests can monkey-patch
    ``pfm.sources.sentiment_factor._SINGLETON`` to inject a fake.
    """
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = SentimentFactorSource()
    return _SINGLETON


_SINGLETON: SentimentFactorSource | None = None


def fetch_sentiment_history(
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Dispatcher-shaped fetch returning ``[date_index, price]``.

    Bridges the design-assembly contract (a DataFrame with a single
    ``price`` column indexed by UTC date) to
    :meth:`SentimentFactorSource.get_daily_sentiment`. The "price" here
    is the mean signed sentiment level — :func:`pfm.model.delta_level`
    converts it into a change at the design-assembly stage so the
    regressor sees Δsentiment, not the raw level (which is non-stationary).

    Args:
        query: search string passed to the underlying source.
        start: window start (UTC; may be tz-naive — coerced).
        end: window end (UTC).

    Returns:
        DataFrame with a UTC ``DatetimeIndex`` named ``date`` and a
        single ``price`` column. Empty frame on no data — the caller
        handles the empty case the same way it would for any other source.
    """
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    # Translate (start, end) into a ``days`` window for the upstream call.
    # Pad by a few days so the regressor still has values at the edges
    # after :func:`_shift_to_stock_calendar` shifts back by one day.
    days = max(1, int((end - start).days) + 3)
    src = _global_source()
    series = src.get_daily_sentiment(query, days=days)
    if series.empty:
        return pd.DataFrame(columns=["price"])
    series = series[(series.index >= start) & (series.index <= end)]
    if series.empty:
        return pd.DataFrame(columns=["price"])
    frame = series.rename("price").to_frame()
    frame.index.name = "date"
    return frame


__all__ = [
    "CACHE_NAMESPACE",
    "CACHE_TTL_SECONDS",
    "CURATED_QUERIES",
    "SentimentFactorSource",
    "curated_sentiment_query",
    "fetch_sentiment_history",
    "parse_sentiment_factor_id",
]
