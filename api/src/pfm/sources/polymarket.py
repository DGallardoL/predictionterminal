"""Polymarket clients (Gamma metadata + CLOB price history).

Two endpoints, both public/no-auth:

  - Gamma:  GET {gamma}/markets?slug={slug}        → metadata, including
            ``clobTokenIds`` which is a JSON STRING that must be parsed.
  - CLOB:   GET {clob}/prices-history?market=...   → ``[{t, p}]``

Always uses ``fidelity=1440`` (daily) — sub-daily fidelities return empty
arrays for resolved markets (py-clob-client issue #216).

Resilience: ``get_market_metadata`` caches slug→MarketMetadata for 1h in a
process-local dict (slug→token_id is immutable) and retries once on a 429
with a 1.5 s backoff. Without this, every concurrent worker re-hits gamma
on every Terminal market open and trips the upstream rate-limit, surfacing
as user-visible 502s and "No history" empty states.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

DAILY_FIDELITY: int = 1440  # minutes between buckets

# Process-local cache: slug → (cached_at_unix, MarketMetadata).
# Slug→token_id is immutable; question/dates are stable enough that a 1h TTL
# is overwhelmingly safe. Resolved-market state (active/closed) is the only
# field that could drift but the Terminal UI re-renders on click anyway.
_METADATA_CACHE: dict[str, tuple[float, MarketMetadata]] = {}
_METADATA_CACHE_LOCK = threading.Lock()
_METADATA_CACHE_TTL_S: float = 3600.0
_METADATA_CACHE_MAX_ENTRIES: int = 4096


class PolymarketError(RuntimeError):
    """Raised when Polymarket returns a usable HTTP response but bad data."""


@dataclass(frozen=True)
class MarketMetadata:
    """Subset of Gamma fields we actually use downstream."""

    slug: str
    question: str
    yes_token_id: str
    no_token_id: str
    start_date: str | None
    end_date: str | None
    closed: bool
    active: bool


class PolymarketClient:
    """Sync httpx-based client for Gamma + CLOB.

    Sync because the rest of the request path (statsmodels, yfinance) is sync;
    making one call async wouldn't buy us anything and just adds bookkeeping.
    """

    def __init__(
        self,
        gamma_url: str,
        clob_url: str,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.gamma_url = gamma_url.rstrip("/")
        self.clob_url = clob_url.rstrip("/")
        # Allow tests to inject a respx-decorated client.
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> PolymarketClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    # ---- Gamma -------------------------------------------------------------

    def get_market_metadata(self, slug: str) -> MarketMetadata:
        """Fetch market metadata by slug. Raises ``PolymarketError`` on bad data.

        Cached for 1h in a process-local dict; retries once on 429 after 1.5s.
        See module docstring for rationale.
        """
        now = time.time()
        with _METADATA_CACHE_LOCK:
            cached = _METADATA_CACHE.get(slug)
            if cached is not None and (now - cached[0]) < _METADATA_CACHE_TTL_S:
                return cached[1]
        meta = self._fetch_market_metadata_uncached(slug)
        with _METADATA_CACHE_LOCK:
            if len(_METADATA_CACHE) >= _METADATA_CACHE_MAX_ENTRIES:
                # Evict the oldest 25% — coarse but bounded memory.
                victims = sorted(_METADATA_CACHE.items(), key=lambda kv: kv[1][0])
                for k, _ in victims[: _METADATA_CACHE_MAX_ENTRIES // 4]:
                    _METADATA_CACHE.pop(k, None)
            _METADATA_CACHE[slug] = (now, meta)
        return meta

    def _fetch_market_metadata_uncached(self, slug: str) -> MarketMetadata:
        """Inner fetch with a single 429-retry. Don't call directly — use the cached wrapper."""
        try:
            r = self._client.get(f"{self.gamma_url}/markets", params={"slug": slug})
            if r.status_code == 429:
                logger.warning("polymarket gamma 429 on slug=%s — retrying in 1.5s", slug)
                time.sleep(1.5)
                r = self._client.get(f"{self.gamma_url}/markets", params={"slug": slug})
            r.raise_for_status()
        except httpx.HTTPError:
            raise
        payload = r.json()

        # Gamma returns a list; the slug filter normally yields exactly one.
        market = payload[0] if isinstance(payload, list) and payload else None
        if not market:
            # Fallback: gamma sometimes filters out resolved markets unless
            # ?closed=true is passed. Retry once before giving up.
            r2 = self._client.get(
                f"{self.gamma_url}/markets",
                params={"slug": slug, "closed": "true"},
            )
            if r2.status_code == 200:
                p2 = r2.json()
                market = p2[0] if isinstance(p2, list) and p2 else None
            if not market:
                raise PolymarketError(f"no market found for slug={slug!r}")

        raw_token_ids = market.get("clobTokenIds")
        if not raw_token_ids:
            raise PolymarketError(f"market {slug!r} has no clobTokenIds")
        try:
            token_ids = json.loads(raw_token_ids)
        except (TypeError, json.JSONDecodeError) as e:
            raise PolymarketError(
                f"clobTokenIds for {slug!r} is not valid JSON: {raw_token_ids!r}"
            ) from e
        if not isinstance(token_ids, list) or len(token_ids) < 2:
            raise PolymarketError(
                f"clobTokenIds for {slug!r} must have ≥2 entries, got {token_ids!r}"
            )

        return MarketMetadata(
            slug=slug,
            question=str(market.get("question", "")),
            yes_token_id=str(token_ids[0]),
            no_token_id=str(token_ids[1]),
            start_date=market.get("startDate"),
            end_date=market.get("endDate"),
            closed=bool(market.get("closed", False)),
            active=bool(market.get("active", True)),
        )

    # ---- CLOB --------------------------------------------------------------

    def get_price_history(
        self,
        token_id: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch daily price history for a CLOB token.

        Returns a DataFrame with columns ``date`` (UTC midnight) and ``price``
        (float in [0, 1]). Empty DataFrame if the API has no data in range.

        CLOB quirks (verified live 2026-04-30):
            - Must send either ``interval`` OR ``startTs``; sending only
              ``fidelity`` returns 400 ``"the time component is mandatory"``.
            - ``endTs`` is silently rejected — every request that includes it
              fails with ``"'startTs' and 'endTs' interval is too long"``,
              even for ranges that fit in the underlying data. We therefore
              omit ``endTs`` and filter the upper bound client-side.
            - Always pass ``fidelity=1440`` (daily) — sub-daily fidelities
              return empty arrays for resolved markets (issue #216).
        """
        params: dict[str, str | int] = {
            "market": token_id,
            "fidelity": DAILY_FIDELITY,
            "interval": "max",
        }
        if start is not None:
            params["startTs"] = _to_unix(start)

        r = self._client.get(f"{self.clob_url}/prices-history", params=params)
        r.raise_for_status()
        history = r.json().get("history", [])
        if not history:
            return pd.DataFrame(columns=["date", "price"])

        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.normalize()
        df = df.rename(columns={"p": "price"})[["date", "price"]]
        # Collapse duplicate days (CLOB occasionally emits multiple buckets)
        # by keeping the last observation, which is the EOD print.
        df = df.groupby("date", as_index=False).last()
        df = df.sort_values("date").reset_index(drop=True)

        # Trim the upper bound client-side since CLOB rejects endTs.
        if end is not None:
            end_norm = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
            df = df[df["date"] <= end_norm.normalize()]

        return df.reset_index(drop=True)


def _to_unix(ts: pd.Timestamp) -> int:
    """Convert a pandas Timestamp to integer unix seconds (UTC)."""
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp())


def fetch_factor_history(
    client: PolymarketClient,
    slug: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: slug → daily YES-token price series.

    Retries once on transient httpx timeouts before propagating, since the
    Gamma + CLOB endpoints are occasionally slow and a second try almost
    always succeeds.
    """
    last_err: httpx.HTTPError | None = None
    for attempt in range(2):
        try:
            meta = client.get_market_metadata(slug)
            df = client.get_price_history(meta.yes_token_id, start=start, end=end)
            if df.empty:
                return df.set_index("date") if "date" in df.columns else df
            return df.set_index("date")
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadError) as e:
            last_err = e
            if attempt == 0:
                continue
            raise
    raise last_err  # type: ignore[misc]


def utc_now_unix() -> int:
    """Helper for callers that want a current-time bound."""
    return int(datetime.now(tz=UTC).timestamp())


@dataclass(frozen=True)
class MarketCandidate:
    """One discovered market — surfaced to the UI as a recommendation."""

    slug: str
    question: str
    volume: float
    end_date: str | None
    active: bool
    closed: bool


_DISCOVER_CACHE: dict[tuple, tuple[float, list[MarketCandidate]]] = {}
_DISCOVER_CACHE_LOCK = threading.Lock()
_DISCOVER_CACHE_TTL_S: float = 300.0  # 5 min — discovery is page-by-volume, slow to rotate.


def discover_markets(
    client: PolymarketClient,
    min_volume: float = 1_000_000.0,
    limit: int = 30,
    keyword: str | None = None,
    pages: int = 5,
) -> list[MarketCandidate]:
    """Pull active high-volume markets from Gamma, optionally filtering by keyword.

    Walks ``pages`` pages of 100 markets each (ordered by volume desc) and
    returns those with ``volume >= min_volume``. If ``keyword`` is set, only
    keep markets whose slug or question contains it (case-insensitive).

    Cached process-wide for 5 minutes — discovery is page-by-volume order
    so the set of top-N markets is stable for many minutes. Retries once on
    429 with 1.5s backoff per page.
    """
    cache_key = (min_volume, limit, keyword or "", pages)
    now = time.time()
    with _DISCOVER_CACHE_LOCK:
        cached = _DISCOVER_CACHE.get(cache_key)
        if cached is not None and (now - cached[0]) < _DISCOVER_CACHE_TTL_S:
            return list(cached[1])

    keyword_lc = keyword.lower() if keyword else None
    seen: dict[str, MarketCandidate] = {}
    for page_idx in range(pages):
        params = {
            "active": "true",
            "closed": "false",
            "limit": 100,
            "offset": page_idx * 100,
            "order": "volumeNum",
            "ascending": "false",
        }
        r = client._client.get(f"{client.gamma_url}/markets", params=params)
        if r.status_code == 429:
            logger.warning("polymarket gamma 429 on discover page=%d — retrying in 1.5s", page_idx)
            time.sleep(1.5)
            r = client._client.get(f"{client.gamma_url}/markets", params=params)
        r.raise_for_status()
        markets = r.json()
        if not markets:
            break
        for m in markets:
            slug = m.get("slug")
            if not slug or slug in seen:
                continue
            try:
                volume = float(m.get("volume") or 0)
            except (TypeError, ValueError):
                volume = 0.0
            if volume < min_volume:
                continue
            question = m.get("question") or ""
            if keyword_lc and keyword_lc not in slug.lower() and keyword_lc not in question.lower():
                continue
            seen[slug] = MarketCandidate(
                slug=slug,
                question=question,
                volume=volume,
                end_date=(m.get("endDate") or "")[:10] or None,
                active=bool(m.get("active", True)),
                closed=bool(m.get("closed", False)),
            )
            if len(seen) >= limit:
                result = list(seen.values())
                with _DISCOVER_CACHE_LOCK:
                    _DISCOVER_CACHE[cache_key] = (now, result)
                return result
    result = list(seen.values())
    with _DISCOVER_CACHE_LOCK:
        _DISCOVER_CACHE[cache_key] = (now, result)
    return result
