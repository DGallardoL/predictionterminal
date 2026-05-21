"""Terminal data-hub helpers: live snapshots, peers, search, overview.

This module is the back-end for the Yahoo-Finance / Bloomberg-style
"Terminal" UI panels. It composes data from multiple sources:

  - Polymarket Gamma (live bid/ask/midpoint, volume, market metadata)
  - Polymarket CLOB ``/prices-history`` (raw history, pass-through)
  - The factor catalog (``factors.yml``) for theme + name search
  - The alpha-hunter sweep cache (``/tmp/ah_sweeps/all_unique_hits.json``)
    for cointegrated peer suggestions
  - The cached factor-history pickle (``/tmp/strat7_factor_history.pkl``)
    for stats (DFA-α, half-life vs neighbors, variance-ratio classification)
    so we don't refit on every request.

Caching is in-process TTL — fine for a POC; if Redis is up we still
piggy-back on the existing ``CacheBackend`` for the CLOB pass-through.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import pickle
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from pfm.cointegration import engle_granger
from pfm.dfa import dfa as dfa_fn
from pfm.factors import FactorConfig
from pfm.mean_reversion import variance_ratio_test

# Lazy import of homepage's enriched theme classifier — used as fallback when
# the factor catalog doesn't have an entry for the slug. Defined at module
# bottom of homepage.py, so import happens at top level here without cycle.
from pfm.terminal.homepage import _theme_for_market as _homepage_theme_for_market

logger = logging.getLogger(__name__)


# Default cache locations — overridable for tests via the helpers below.
DEFAULT_AH_HITS_PATH: Path = Path("/tmp/ah_sweeps/all_unique_hits.json")
DEFAULT_FACTOR_HISTORY_PATH: Path = Path("/tmp/strat7_factor_history.pkl")

# How long to keep different things in the in-memory cache.
# Bumped TTL_OVERVIEW_SECONDS 60→300 (UX audit flagged 3 s mid-session
# recomputes — they only paid off if a user clicked exactly when the TTL
# expired). The underlying Gamma /markets data shifts on minutes-scale, so
# 5 min staleness is acceptable for a Bloomberg-style heatmap.
TTL_LIVE_SECONDS: int = 30  # bid/ask/midpoint move quickly
TTL_OVERVIEW_SECONDS: int = 300  # gamma "top markets" page (was 60)
TTL_HISTORY_SECONDS: int = 300  # daily bars
TTL_PEERS_SECONDS: int = 3600  # alpha-hunter sweep is mostly static
TTL_STATS_SECONDS: int = 1800  # cached-pickle stats

# Conviction filter for upcoming resolutions: drop very low-volume.
UPCOMING_MIN_VOLUME_24H: float = 0.0


# --- in-memory TTL cache ----------------------------------------------------


@dataclass
class _Entry:
    value: Any
    expires_at: float


_L2_PAYLOAD_VERSION: int = 1
_L2_MAGIC: bytes = b"PFMTC1\x00"  # magic prefix so we can detect/reject legacy json blobs


class TTLCache:
    """Two-tier thread-safe cache with per-key TTL.

    L1 = in-process dict (per gunicorn worker, fast).
    L2 = optional Redis (cross-worker, pickle-serialised).

    Set ``redis_backend`` on the module-level singleton (see ``set_redis_backend``)
    to enable L2 — values are pickled with a versioned envelope so they
    round-trip faithfully (pd.Series, np.ndarray, dataclasses, etc.). The
    previous implementation used ``json.dumps(default=str)`` which silently
    stringified pandas Series — workers reading the L2 entry back would get
    ``dict[str, str]`` and downstream ``.iloc`` calls would crash. Pickle is
    safe here because L2 is internal infra (Redis on the same trust boundary
    as the app); we never deserialise untrusted input.

    Envelope format::

        magic (7 bytes "PFMTC1\\0") || pickle({"v": <int>, "data": <any>})

    Future schema changes bump ``_L2_PAYLOAD_VERSION``; entries with an
    unknown version are treated as a miss so stale entries expire naturally.
    """

    def __init__(self) -> None:
        self._d: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._redis: Any | None = None  # set via attach_redis()
        self._prefix: str = "term:"

    def attach_redis(self, backend: Any, prefix: str = "term:") -> None:
        """Wire a ``pfm.cache.CacheBackend`` (RedisCache / NullCache) for L2."""
        self._redis = backend
        self._prefix = prefix

    @staticmethod
    def _encode_l2(value: Any) -> bytes:
        """Serialise a value for L2 storage with a magic prefix + version tag."""
        envelope = {"v": _L2_PAYLOAD_VERSION, "data": value}
        return _L2_MAGIC + pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _decode_l2(raw: bytes | str) -> Any:
        """Decode an L2 blob. Raises ``ValueError`` on legacy or unknown payloads.

        Legacy json entries (no magic prefix) and entries with a future
        version tag are rejected so they get treated as a miss and expire
        naturally without crashing the worker.
        """
        if isinstance(raw, str):
            raw = raw.encode()
        if not raw.startswith(_L2_MAGIC):
            raise ValueError("legacy or unknown L2 payload (missing magic)")
        envelope = pickle.loads(raw[len(_L2_MAGIC) :])
        if not isinstance(envelope, dict) or envelope.get("v") != _L2_PAYLOAD_VERSION:
            raise ValueError(f"unsupported L2 payload version: {envelope!r}")
        return envelope["data"]

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._d.get(key)
            if entry is not None:
                if entry.expires_at >= now:
                    return entry.value
                self._d.pop(key, None)
        # L1 miss — try L2 Redis (pickle envelope).
        if self._redis is not None and getattr(self._redis, "enabled", False):
            try:
                raw = self._redis.get(self._prefix + key)
                if raw:
                    value = self._decode_l2(raw)
                    # Promote into L1 with a short TTL so subsequent hits skip
                    # the round-trip. The full L2 TTL handles expiry.
                    with self._lock:
                        self._d[key] = _Entry(value=value, expires_at=now + 30)
                    return value
            except Exception:
                # Legacy / corrupt / version-mismatch payloads are treated as
                # a miss. They expire naturally per their original TTL.
                pass
        return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            self._d[key] = _Entry(value=value, expires_at=time.time() + ttl_seconds)
        # Propagate to L2 — pickle handles arbitrary Python objects faithfully
        # (pd.Series, np.ndarray, dataclasses). The previous json-with-default-str
        # approach silently corrupted Series values; see _decode_l2 docstring.
        if self._redis is not None and getattr(self._redis, "enabled", False):
            try:
                blob = self._encode_l2(value)
                # Cap the L2 TTL at 1h — long-lived response cache only.
                self._redis.set(self._prefix + key, blob, min(ttl_seconds, 3600))
            except (pickle.PicklingError, TypeError, AttributeError):
                # Not picklable (e.g. open file handle, lock) — L1 is enough.
                pass
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


# Module-level singleton — shared across requests within the same worker.
# pfm.main.lifespan() calls TERMINAL_CACHE.attach_redis() to wire the L2
# Redis layer so warmed entries propagate across gunicorn workers.
TERMINAL_CACHE = TTLCache()


# --- Gamma helpers ----------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    """Convert a value to float or ``None`` if missing / non-finite."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _parse_iso_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # Strip trailing 'Z' and parse.
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _parse_outcome_prices(raw: Any) -> tuple[float | None, float | None]:
    """``outcomePrices`` is a JSON string of ``["yes_price", "no_price"]``."""
    if not raw:
        return None, None
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
    elif isinstance(raw, list):
        arr = raw
    else:
        return None, None
    yes = _safe_float(arr[0]) if len(arr) >= 1 else None
    no = _safe_float(arr[1]) if len(arr) >= 2 else None
    return yes, no


def fetch_gamma_market(
    http: httpx.Client, gamma_url: str, slug: str, *, timeout: float = 5.0
) -> dict[str, Any]:
    """Fetch one market dict from Gamma. Raises ``LookupError`` if missing.
    Falls back to ``closed=true`` for resolved markets. Retries with backoff on 429.
    """
    base = gamma_url.rstrip("/")
    last_429 = False
    for attempt in range(3):
        try:
            r = http.get(f"{base}/markets", params={"slug": slug}, timeout=timeout)
            if r.status_code == 429:
                last_429 = True
                time.sleep(0.5 * (2**attempt))
                continue
            r.raise_for_status()
            arr = r.json() or []
            if isinstance(arr, list) and arr:
                return arr[0]
            # No market on default filter → try closed=true (resolved markets).
            r2 = http.get(
                f"{base}/markets",
                params={"slug": slug, "closed": "true"},
                timeout=timeout,
            )
            if r2.status_code == 429:
                last_429 = True
                time.sleep(0.5 * (2**attempt))
                continue
            if r2.status_code == 200:
                arr2 = r2.json() or []
                if isinstance(arr2, list) and arr2:
                    return arr2[0]
            break  # default + closed both empty → genuinely missing
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                last_429 = True
                time.sleep(0.5 * (2**attempt))
                continue
            raise
    if last_429:
        raise LookupError(f"gamma rate-limited for slug={slug!r}")
    raise LookupError(f"no market for slug={slug!r}")


def fetch_gamma_top_markets(
    http: httpx.Client,
    gamma_url: str,
    *,
    pages: int = 5,
    page_size: int = 100,
    order: str = "volume24hr",
    ascending: bool = False,
    extra_params: dict[str, str] | None = None,
    timeout: float = 7.0,
) -> list[dict[str, Any]]:
    """Walk ``pages`` pages of active/non-closed markets sorted by ``order``."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in range(pages):
        params: dict[str, str | int] = {
            "active": "true",
            "closed": "false",
            "limit": page_size,
            "offset": i * page_size,
            "order": order,
            "ascending": "true" if ascending else "false",
        }
        if extra_params:
            params.update(extra_params)
        r = http.get(f"{gamma_url.rstrip('/')}/markets", params=params, timeout=timeout)
        r.raise_for_status()
        page = r.json() or []
        if not page:
            break
        for m in page:
            slug = m.get("slug")
            if not slug or slug in seen:
                continue
            seen.add(slug)
            out.append(m)
    return out


# --- shaping into our schemas (returns plain dicts) -------------------------


def shape_live(market: dict[str, Any]) -> dict[str, Any]:
    """Extract live snapshot fields from a Gamma market dict."""
    best_bid = _safe_float(market.get("bestBid"))
    best_ask = _safe_float(market.get("bestAsk"))
    last_trade = _safe_float(market.get("lastTradePrice"))
    midpoint: float | None = None
    if best_bid is not None and best_ask is not None:
        midpoint = (best_bid + best_ask) / 2.0
    elif last_trade is not None:
        midpoint = last_trade
    spread_cents: float | None = None
    if best_bid is not None and best_ask is not None:
        # Round to avoid float-precision artifacts like 0.10000000000000009
        # bleeding into the JSON payload. Pennies are 2 decimals max anyway.
        spread_cents = round((best_ask - best_bid) * 100.0, 4)

    # Prefer flat ``volume24hr``; fall back to ``volumeNum`` only for total.
    vol24 = _safe_float(market.get("volume24hr"))
    if vol24 is None:
        vol24 = _safe_float(market.get("volume24hrClob"))

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "midpoint": midpoint,
        "last_trade_price": last_trade,
        "spread_cents": spread_cents,
        "volume_24hr": vol24,
        "volume_total": _safe_float(market.get("volumeNum") or market.get("volume")),
        "liquidity": _safe_float(market.get("liquidityNum") or market.get("liquidity")),
        "one_day_price_change": _safe_float(market.get("oneDayPriceChange")),
        "one_week_price_change": _safe_float(market.get("oneWeekPriceChange")),
    }


def shape_meta(market: dict[str, Any], theme: str | None = None) -> dict[str, Any]:
    """Extract metadata fields from a Gamma market dict."""
    end_dt = _parse_iso_date(market.get("endDate") or market.get("endDateIso"))
    created_dt = _parse_iso_date(market.get("createdAt"))
    now = datetime.now(tz=UTC)

    days_to_resolve: int | None = None
    if end_dt is not None:
        # Naive endDate gets bumped to UTC for the diff.
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)
        days_to_resolve = max(0, (end_dt - now).days)

    age_days: int | None = None
    if created_dt is not None:
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=UTC)
        age_days = max(0, (now - created_dt).days)

    return {
        "slug": str(market.get("slug", "")),
        "question": str(market.get("question", "")),
        "description": (market.get("description") or None),
        "theme": theme,
        "resolution_source": (market.get("resolutionSource") or None),
        "end_date": (market.get("endDate") or None),
        "start_date": (market.get("startDate") or None),
        "created_at": (market.get("createdAt") or None),
        "days_to_resolve": days_to_resolve,
        "age_days": age_days,
        "active": bool(market.get("active", True)),
        "closed": bool(market.get("closed", False)),
    }


# --- Stats computation ------------------------------------------------------


@functools.cache
def _load_factor_history_cache(path: Path) -> dict[str, pd.Series]:
    """Load the cached factor-history pickle (dict[slug] = price Series).

    Returns ``{}`` on any error so missing-cache doesn't 500 the API.

    Wrapped in ``functools.cache`` (perf audit 2026-05-16): the pickle is
    static on disk, so re-reading on every request was pure overhead. The
    LRU is keyed on ``path`` so distinct overrides (tests, alt caches)
    still re-read. To force a refresh call ``_load_factor_history_cache.cache_clear()``.
    """
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict):
            data = {}
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as e:
        logger.warning("could not load factor history pickle %s: %s", path, e)
        data = {}
    return data


def compute_stats_from_series(
    series: pd.Series, *, neighbor: pd.Series | None = None
) -> dict[str, Any]:
    """Compute mean-reversion / persistence stats for a single series.

    ``neighbor`` is optional: if provided, run Engle-Granger and surface the
    half-life. Otherwise the half-life slot is left ``None``.
    """
    s = series.dropna().astype(float)
    n_obs = len(s)
    out: dict[str, Any] = {
        "n_obs": n_obs,
        "half_life_days": None,
        "dfa_alpha": None,
        "dfa_interpretation": None,
        "variance_ratio": None,
        "variance_ratio_verdict": None,
        "realized_vol_30d": None,
        "current_price": float(s.iloc[-1]) if n_obs > 0 else None,
    }
    if n_obs < 10:
        return out

    # Realised vol of first differences over the last 30 bars.
    diffs = s.diff().dropna()
    if len(diffs) >= 5:
        tail = diffs.iloc[-30:] if len(diffs) >= 30 else diffs
        sd = float(tail.std(ddof=1))
        out["realized_vol_30d"] = sd if math.isfinite(sd) else None

    # DFA — needs ≥ 32 obs to be remotely useful (4 * min_n=8).
    try:
        d = dfa_fn(s)
        if math.isfinite(d.alpha):
            out["dfa_alpha"] = float(d.alpha)
            out["dfa_interpretation"] = d.interpretation
    except (ValueError, RuntimeError) as e:
        logger.debug("dfa failed: %s", e)

    # Variance-ratio classification at q=5 (weekly horizon for daily bars).
    if n_obs >= 25:
        try:
            vr = variance_ratio_test(s, q=5)
            if math.isfinite(vr.vr):
                out["variance_ratio"] = float(vr.vr)
                out["variance_ratio_verdict"] = vr.verdict
        except (ValueError, RuntimeError) as e:
            logger.debug("vr failed: %s", e)

    # Optional half-life vs a single neighbor — used for "vs nearest peer".
    if neighbor is not None:
        nb = neighbor.dropna().astype(float)
        if len(nb) >= 30:
            try:
                cint = engle_granger(s, nb)
                if cint.half_life_days is not None and math.isfinite(cint.half_life_days):
                    out["half_life_days"] = float(cint.half_life_days)
            except (ValueError, RuntimeError) as e:
                logger.debug("engle_granger failed: %s", e)

    return out


# --- Peer lookup from alpha-hunter sweep cache ------------------------------


def _load_ah_hits(path: Path) -> list[dict[str, Any]]:
    """Load the alpha-hunter sweep results JSON."""
    cached = TERMINAL_CACHE.get(f"ah_hits::{path}")
    if cached is not None:
        return cached
    if not path.exists():
        TERMINAL_CACHE.set(f"ah_hits::{path}", [], TTL_PEERS_SECONDS)
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            data = []
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("could not load ah hits %s: %s", path, e)
        data = []
    TERMINAL_CACHE.set(f"ah_hits::{path}", data, TTL_PEERS_SECONDS)
    return data


def find_peers(
    factor_id: str, *, hits_path: Path = DEFAULT_AH_HITS_PATH, top_n: int = 5
) -> list[dict[str, Any]]:
    """Return the top-N cointegrated peers for ``factor_id`` from the AH cache.

    Hits are bidirectional pairs ``(a_id, b_id)``; we collapse them so the
    other side is the peer regardless of slot. Ranks by ``oos_sharpe`` desc,
    falling back to ``adf_pvalue`` asc.
    """
    rows = _load_ah_hits(hits_path)
    out: list[dict[str, Any]] = []
    for h in rows:
        a, b = h.get("a_id"), h.get("b_id")
        if a == factor_id and b:
            peer = b
        elif b == factor_id and a:
            peer = a
        else:
            continue
        out.append(
            {
                "peer_id": peer,
                "half_life_days": _safe_float(h.get("half_life_days")),
                "adf_pvalue": _safe_float(h.get("adf_pvalue")),
                "beta_hedge": _safe_float(h.get("beta_hedge")),
                "oos_sharpe": _safe_float(h.get("oos_sharpe")),
                "full_sharpe": _safe_float(h.get("full_sharpe")),
                "perm_p": _safe_float(h.get("perm_p")),
                "verdict": h.get("verdict"),
                "sweep": h.get("sweep"),
            }
        )
    out.sort(
        key=lambda r: (
            -(r["oos_sharpe"] if r["oos_sharpe"] is not None else -1e9),
            (r["adf_pvalue"] if r["adf_pvalue"] is not None else 1.0),
        )
    )
    # De-duplicate peers, keeping the strongest hit.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in out:
        if r["peer_id"] in seen:
            continue
        seen.add(r["peer_id"])
        deduped.append(r)
        if len(deduped) >= top_n:
            break
    return deduped


def implied_fair_price(target: pd.Series, peer: pd.Series) -> float | None:
    """Implied "fair" YES price of ``target`` given today's ``peer`` price.

    Uses the rolling-EG cointegration relationship ``α + β · peer`` evaluated
    at the latest peer price. Clipped to [0, 1]. Returns ``None`` if the
    series are too short or the regression isn't sensible.
    """
    a = target.dropna().astype(float)
    b = peer.dropna().astype(float)
    common = a.index.intersection(b.index)
    if len(common) < 30:
        return None
    try:
        cint = engle_granger(a.loc[common], b.loc[common])
    except (ValueError, RuntimeError):
        return None
    if not math.isfinite(cint.beta_hedge) or not math.isfinite(cint.intercept):
        return None
    fair = cint.intercept + cint.beta_hedge * float(b.loc[common].iloc[-1])
    if not math.isfinite(fair):
        return None
    return float(np.clip(fair, 0.0, 1.0))


# --- Search -----------------------------------------------------------------


_TOKEN_RX = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RX.findall((s or "").lower())


def search_factors(
    query: str,
    factors: dict[str, FactorConfig],
    *,
    limit: int = 30,
    price_lookup: dict[str, float] | None = None,
    volume_lookup: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Token-overlap fuzzy search across ``name`` + ``slug``.

    Score = (tokens matched / query tokens) + 0.25·(matched / candidate tokens),
    so a query of 1-2 tokens prefers tight matches over diluted long names.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    q_set = set(q_tokens)
    scored: list[tuple[float, FactorConfig]] = []
    for fc in factors.values():
        cand = _tokenize(fc.name) + _tokenize(fc.slug)
        if not cand:
            continue
        c_set = set(cand)
        matched = q_set & c_set
        if not matched:
            continue
        primary = len(matched) / len(q_set)
        secondary = len(matched) / max(1, len(c_set))
        score = primary + 0.25 * secondary
        # Bonus for exact slug or full-name substring match.
        ql = query.lower().strip()
        if ql and (ql in fc.slug.lower() or ql in fc.name.lower()):
            score += 0.5
        scored.append((score, fc))
    scored.sort(key=lambda x: -x[0])
    out: list[dict[str, Any]] = []
    for score, fc in scored[:limit]:
        price = None
        if price_lookup is not None:
            price = price_lookup.get(fc.slug) or price_lookup.get(fc.id)
        volume = None
        if volume_lookup is not None:
            volume = volume_lookup.get(fc.slug) or volume_lookup.get(fc.id)
        out.append(
            {
                "factor_id": fc.id,
                "name": fc.name,
                "slug": fc.slug,
                "theme": fc.theme,
                "score": float(score),
                "current_price": price,
                # Aliases requested by the 2026-05-14 UX audit: front-end reads
                # ``price`` and ``volume_24h`` rather than the legacy field names.
                "price": price,
                "volume_24h": volume,
            }
        )
    return out


def cached_price_lookup(path: Path = DEFAULT_FACTOR_HISTORY_PATH) -> dict[str, float]:
    """Latest price keyed by slug from the on-disk pickle (best-effort)."""
    hist = _load_factor_history_cache(path)
    out: dict[str, float] = {}
    for slug, ser in hist.items():
        if not isinstance(ser, pd.Series) or ser.empty:
            continue
        try:
            out[slug] = float(ser.iloc[-1])
        except (TypeError, ValueError):
            continue
    return out


# --- Overview aggregation ---------------------------------------------------


@dataclass
class OverviewBuckets:
    """Grouped output from ``build_overview``."""

    n_markets: int = 0
    theme_heatmap: list[dict[str, Any]] = field(default_factory=list)
    top_movers: list[dict[str, Any]] = field(default_factory=list)
    most_traded: list[dict[str, Any]] = field(default_factory=list)
    recently_launched: list[dict[str, Any]] = field(default_factory=list)
    upcoming_resolutions: list[dict[str, Any]] = field(default_factory=list)


def _yes_price_from_market(m: dict[str, Any]) -> float | None:
    """Best-guess current YES price: midpoint else lastTradePrice else outcomePrices[0]."""
    bb = _safe_float(m.get("bestBid"))
    ba = _safe_float(m.get("bestAsk"))
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    lt = _safe_float(m.get("lastTradePrice"))
    if lt is not None:
        return lt
    yes, _no = _parse_outcome_prices(m.get("outcomePrices"))
    return yes


def _theme_for_slug(slug: str, by_slug: dict[str, FactorConfig]) -> str | None:
    fc = by_slug.get(slug)
    return fc.theme if fc else None


def build_overview(
    markets: list[dict[str, Any]],
    factors: dict[str, FactorConfig],
    *,
    movers_min_volume: float = 5_000.0,
    upcoming_window_days: int = 7,
    top_k: int = 10,
    upcoming_top_k: int = 20,
) -> OverviewBuckets:
    """Aggregate the gamma "top markets" page into the four overview panels."""
    by_slug = {fc.slug: fc for fc in factors.values()}
    out = OverviewBuckets(n_markets=len(markets))

    # --- theme heatmap ------------------------------------------------------
    # Two-step classification: factor-catalog lookup wins, then the
    # keyword-inference fallback from homepage.py (covers live Gamma markets
    # not yet in factors.yml). "other" only if both fail.
    by_theme: dict[str, list[dict[str, Any]]] = {}
    for m in markets:
        slug = m.get("slug")
        if not slug:
            continue
        theme = _theme_for_slug(slug, by_slug) or _homepage_theme_for_market(m) or "other"
        by_theme.setdefault(theme, []).append(m)

    heatmap: list[dict[str, Any]] = []
    for theme, group in by_theme.items():
        changes = [
            c for c in (_safe_float(m.get("oneDayPriceChange")) for m in group) if c is not None
        ]
        vols = [v for v in (_safe_float(m.get("volume24hr")) for m in group) if v is not None]
        # Yes-side probability per market (midpoint when both quotes exist,
        # else last-trade). Filtered to (0, 1) since 0 or 1 are resolved
        # markets that would skew the median.
        yes_prices = [
            p for p in (_yes_price_from_market(m) for m in group) if p is not None and 0.0 < p < 1.0
        ]
        heatmap.append(
            {
                "theme": theme,
                "n_markets": len(group),
                "median_24h_change": float(np.median(changes)) if changes else None,
                "median_volume_24hr": float(np.median(vols)) if vols else None,
                "total_volume_24hr": float(np.sum(vols)) if vols else None,
                "median_yes_price": (
                    round(float(np.median(yes_prices)), 4) if yes_prices else None
                ),
            }
        )
    heatmap.sort(key=lambda h: -(h["total_volume_24hr"] or 0.0))
    out.theme_heatmap = heatmap

    # --- top movers ---------------------------------------------------------
    movers: list[tuple[float, dict[str, Any]]] = []
    for m in markets:
        chg = _safe_float(m.get("oneDayPriceChange"))
        vol = _safe_float(m.get("volume24hr")) or 0.0
        if chg is None or vol < movers_min_volume:
            continue
        movers.append((abs(chg), m))
    movers.sort(key=lambda t: -t[0])
    out.top_movers = [
        {
            "slug": m.get("slug"),
            "question": m.get("question"),
            "theme": (_theme_for_slug(m.get("slug", ""), by_slug) or _homepage_theme_for_market(m)),
            "price": _yes_price_from_market(m),
            "one_day_price_change": _safe_float(m.get("oneDayPriceChange")),
            "volume_24hr": _safe_float(m.get("volume24hr")),
        }
        for _, m in movers[:top_k]
    ]

    # --- most-traded today --------------------------------------------------
    traded = sorted(markets, key=lambda m: -(_safe_float(m.get("volume24hr")) or 0.0))
    out.most_traded = [
        {
            "slug": m.get("slug"),
            "question": m.get("question"),
            "theme": (_theme_for_slug(m.get("slug", ""), by_slug) or _homepage_theme_for_market(m)),
            "price": _yes_price_from_market(m),
            "one_day_price_change": _safe_float(m.get("oneDayPriceChange")),
            "volume_24hr": _safe_float(m.get("volume24hr")),
        }
        for m in traded[:top_k]
        if (_safe_float(m.get("volume24hr")) or 0.0) > 0
    ]

    # --- recently launched --------------------------------------------------
    now = datetime.now(tz=UTC)
    launched: list[tuple[datetime, dict[str, Any]]] = []
    for m in markets:
        created = _parse_iso_date(m.get("createdAt"))
        if created is None:
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        launched.append((created, m))
    launched.sort(key=lambda t: -t[0].timestamp())
    out.recently_launched = [
        {
            "slug": m.get("slug"),
            "question": m.get("question"),
            "theme": _theme_for_slug(m.get("slug", ""), by_slug),
            "price": _yes_price_from_market(m),
            "created_at": m.get("createdAt"),
            "age_days": max(0, (now - dt).days),
        }
        for dt, m in launched[:top_k]
    ]

    # --- upcoming resolutions ----------------------------------------------
    # Watchlist semantics: show markets that resolve SOON and still have
    # meaningful uncertainty. Pre-resolved markets (price ≥ 0.95 or ≤ 0.05)
    # are no longer "upcoming" in a useful sense — they're just waiting for
    # the clock. Filter them out so the panel surfaces actionable contracts.
    upcoming: list[tuple[datetime, float, dict[str, Any]]] = []
    for m in markets:
        end = _parse_iso_date(m.get("endDate") or m.get("endDateIso"))
        if end is None:
            continue
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        days_left = (end - now).days
        if days_left < 0 or days_left > upcoming_window_days:
            continue
        if (_safe_float(m.get("volume24hr")) or 0.0) < UPCOMING_MIN_VOLUME_24H:
            continue
        price = _yes_price_from_market(m)
        if price is None:
            continue
        if price >= 0.95 or price <= 0.05:
            continue
        upcoming.append((end, price, m))
    # Sort by soonest first — that's what "upcoming" should mean.
    upcoming.sort(key=lambda t: t[0])
    out.upcoming_resolutions = [
        {
            "slug": m.get("slug"),
            "question": m.get("question"),
            "theme": (_theme_for_slug(m.get("slug", ""), by_slug) or _homepage_theme_for_market(m)),
            "price": price,
            "end_date": m.get("endDate"),
            "days_to_resolve": max(0, (end - now).days),
            "conviction": abs(price - 0.5) * 2.0,
        }
        for end, price, m in upcoming[:upcoming_top_k]
    ]

    return out


__all__ = [
    "DEFAULT_AH_HITS_PATH",
    "DEFAULT_FACTOR_HISTORY_PATH",
    "TERMINAL_CACHE",
    "TTL_HISTORY_SECONDS",
    "TTL_LIVE_SECONDS",
    "TTL_OVERVIEW_SECONDS",
    "TTL_PEERS_SECONDS",
    "TTL_STATS_SECONDS",
    "OverviewBuckets",
    "TTLCache",
    "build_overview",
    "cached_price_lookup",
    "compute_stats_from_series",
    "fetch_gamma_market",
    "fetch_gamma_top_markets",
    "find_peers",
    "implied_fair_price",
    "search_factors",
    "shape_live",
    "shape_meta",
]
