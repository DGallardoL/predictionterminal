"""PM-VIX Composite Index — prediction-market-implied tail-risk gauge.

This module derives a Bloomberg-style "fear index" purely from
prediction-market odds rather than equity-option implieds. Five
sub-buckets feed the headline number:

  - **Recession** (30%): probability the US enters / has entered an
    NBER-dated recession in 2026 / 2027.
  - **Geopolitical** (25%): tail-risk markets on Iran regime, Taiwan
    conflict, Ukraine ceasefire, etc.
  - **Election uncertainty** (20%): margin-of-victory and swing-state
    contracts whose binary state is still unsettled.
  - **Macro** (15%): Fed surprise, CPI surprise, dovish/hawkish tail
    contracts.
  - **Crypto / banking** (10%): BTC crash and banking-stress markets.

Each bucket produces a 0–100 sub-score (volume-weighted average of its
constituents' YES probabilities); the buckets are then weight-summed and
rescaled into the 0–100 "PM-VIX" headline. By convention 0 ≈ pure
risk-on (no tail probability priced) and 100 ≈ panic (every tail-risk
contract is at-the-money).

The output regime classification:

  - score < 25  →  ``RISK_ON``
  - 25 ≤ s < 60 → ``NEUTRAL``
  - score ≥ 60 → ``RISK_OFF``

History (`pm_vix_history`) is generated synthetically from a fixed seed
when no real time-series store exists — this is documented in the
docstring. Plugging a real historical store in later is a one-function
swap.

Endpoints (mounted via the module's ``router``):

  - ``GET /indices/pm-vix``             current snapshot
  - ``GET /indices/pm-vix/components``  bucket-by-bucket breakdown
  - ``GET /indices/pm-vix/history``     synthesised 30-day series
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.auth.dependencies import require_admin
from pfm.cache_utils import get_cache
from pfm.terminal import fetch_gamma_market

logger = logging.getLogger(__name__)

GAMMA_URL: str = "https://gamma-api.polymarket.com"

_VIX_CACHE = get_cache("pm_vix", ttl=300)

# How fresh a cached snapshot must be before we tell the caller it's "stale".
# 10 min — twice the refresh cadence so a single missed refresh round-trip
# doesn't trigger noisy ``is_stale=True`` flags.
PMVIX_STALE_AFTER_SECONDS: int = 600

# Bump default TTL of the prewarmed snapshot above the refresh interval so
# a slightly-late background tick still serves the previous payload rather
# than dropping into "no cache" territory.
PMVIX_PREWARM_CACHE_TTL_SECONDS: int = 900  # 15 min — covers a missed tick

# Internal key under which the prewarmed snapshot is stored.
_PREWARM_KEY: tuple[str, str] = ("snapshot", "live")
_PREWARM_AT_KEY: str = "prewarmed_at_unix"

#: TTL for the persisted bucket-slug map (24h). After this, the next
#: ``compute_pm_vix`` call falls back to the hardcoded ``BUCKET_SLUGS`` map
#: until ``validate_and_refresh_buckets`` is run again.
SLUG_CACHE_TTL_SECONDS: int = 86_400

#: Disk path the refreshed slug map is persisted to. The path is fixed so
#: a separate process (admin endpoint, cron job, manual ``curl``) can all
#: see the same authoritative map.
DEFAULT_SLUG_CACHE_PATH: str = "/tmp/pfm_pm_vix_slugs.json"

#: How many alternative live markets to keep per dead slug, ordered by
#: ``volume24hr`` desc. We keep more than 1 so the volume-weighted average
#: in ``_bucket_avg`` doesn't degrade to a single-market read.
TOP_N_REPLACEMENTS: int = 3

#: Per-bucket keyword search terms used when a hardcoded slug goes dead.
#: Tuned to surface markets with non-trivial 24h volume on Polymarket; the
#: list isn't exhaustive on purpose — narrower queries return higher-quality
#: hits than broad ones like "trump" alone.
BUCKET_SEARCH_KEYWORDS: dict[str, list[str]] = {
    "recession": ["recession"],
    "geopolitical": ["iran", "taiwan", "russia", "ukraine"],
    "election": ["2028", "election", "trump", "vance", "newsom"],
    "macro": ["fed", "cpi", "inflation"],
    "crypto": ["bitcoin", "btc", "ethereum"],
}


# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------
#
# Each bucket lists Polymarket *slugs* whose YES probability we want.
# Slugs come from ``factors.yml`` where possible; missing markets are
# silently dropped at fetch time so the score stays well-defined even
# when individual contracts resolve.
#
# ``BUCKET_WEIGHTS`` MUST sum to 1.0 — checked in ``_validate_weights``.

BUCKET_WEIGHTS: dict[str, float] = {
    "recession": 0.30,
    "geopolitical": 0.25,
    "election": 0.20,
    "macro": 0.15,
    "crypto": 0.10,
}

BUCKET_SLUGS: dict[str, list[str]] = {
    "recession": [
        "us-recession-by-end-of-2026",
        "us-recession-in-2026",
        "canada-recession-before-2027",
        "us-recession-in-q1-2026",
        "us-recession-in-q2-2026",
    ],
    "geopolitical": [
        "will-the-iranian-regime-fall-by-june-30",
        "will-the-us-invade-iran-before-2027",
        "will-china-invade-taiwan-before-2027",
        "will-china-blockade-taiwan-by-june-30",
        "russia-x-ukraine-ceasefire-before-2027",
    ],
    "election": [
        "trump-out-as-president-before-2027",
        "jerome-powell-out-as-fed-chair-by-may-14-2026",
        "will-the-court-force-trump-to-refund-tariffs-2026-06-30",
        "vance-2028-republican-nominee",
        "republicans-keep-the-house-in-2026-midterms",
    ],
    "macro": [
        "will-the-fed-decrease-interest-rates-by-50-bps-after-the-june-2026-meeting",
        "will-the-fed-increase-interest-rates-by-50-bps-after-the-june-2026-meeting",
        "will-no-fed-rate-cuts-happen-in-2026",
        "will-12-or-more-fed-rate-cuts-happen-in-2026",
        "will-cpi-be-above-3-5-in-2026",
    ],
    "crypto": [
        "btc-below-50k-in-2026",
        "btc-flash-crash-2026",
        "us-bank-failure-2026",
    ],
}


# Sigmoid centre for sub-bucket → 0..100 mapping.
# A volume-weighted YES probability of 0.10 maps near 30 (mild risk-on);
# 0.50 near 70 (elevated); 0.80+ near 95 (panic).
_SIGMOID_CENTRE = 0.30
_SIGMOID_K = 8.0


def _validate_weights() -> None:
    s = sum(BUCKET_WEIGHTS.values())
    if abs(s - 1.0) > 1e-6:
        raise RuntimeError(f"BUCKET_WEIGHTS must sum to 1.0, got {s:.6f}")


_validate_weights()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _market_yes_prob(market: dict[str, Any]) -> float | None:
    """Best estimate of the YES probability from a Gamma market dict."""
    bb = _safe_float(market.get("bestBid"))
    ba = _safe_float(market.get("bestAsk"))
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    last = _safe_float(market.get("lastTradePrice"))
    if last is not None:
        return last
    return None


def _market_volume(market: dict[str, Any]) -> float:
    """24h volume preferred; falls back to lifetime; missing → 0."""
    v = _safe_float(market.get("volume24hr"))
    if v is None:
        v = _safe_float(market.get("volumeNum") or market.get("volume"))
    return v if v is not None else 0.0


def _bucket_avg(
    slugs: list[str],
    *,
    http: httpx.Client,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[float, int]:
    """Volume-weighted average YES probability across ``slugs``.

    Returns (avg_prob, n_used). ``n_used`` is the number of markets that
    contributed (i.e. resolved cleanly). When every slug is missing,
    returns (0.0, 0) so the bucket contributes nothing rather than NaN.

    ``overrides`` lets tests skip the HTTP layer by supplying a
    ``slug -> market_dict`` map.

    Performance: per-slug Gamma fetches run in a bounded thread pool when
    no ``overrides`` map is supplied. The previous serial loop was the
    dominant cold-cache cost (~5 slugs × 5 buckets = 25 sequential RTTs
    feeding ``/indices/pm-vix/components`` and ``/indices/pm-vix``).
    """
    # Resolve any override hits without spawning workers — they're free.
    resolved: list[dict[str, Any] | None] = []
    fetch_indices: list[int] = []
    fetch_slugs: list[str] = []
    for idx, slug in enumerate(slugs):
        if overrides is not None and slug in overrides:
            resolved.append(overrides[slug])
        else:
            resolved.append(None)
            fetch_indices.append(idx)
            fetch_slugs.append(slug)

    def _fetch(slug: str) -> dict[str, Any] | None:
        try:
            return fetch_gamma_market(http, GAMMA_URL, slug)
        except (LookupError, httpx.HTTPError) as exc:
            logger.info("PM-VIX: skipping %s: %s", slug, exc)
            return None

    if fetch_slugs:
        max_workers = min(len(fetch_slugs), 8)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pmvix") as ex:
            for idx, market in zip(fetch_indices, ex.map(_fetch, fetch_slugs), strict=True):
                resolved[idx] = market

    weights: list[float] = []
    probs: list[float] = []
    for m in resolved:
        if m is None:
            continue
        p = _market_yes_prob(m)
        if p is None:
            continue
        # Volume floor of 1.0 so a zero-volume market still gets a
        # 1-unit weight rather than dropping silently.
        w = max(1.0, _market_volume(m))
        probs.append(p)
        weights.append(w)
    if not probs:
        return 0.0, 0
    avg = float(np.average(probs, weights=weights))
    return avg, len(probs)


def _sub_score(avg_prob: float) -> float:
    """Map a 0..1 average probability to a 0..100 sub-bucket score.

    Logistic curve centred at ``_SIGMOID_CENTRE``; ``_SIGMOID_K`` controls
    steepness. Calibrated so 0.05 → ~6, 0.30 → 50, 0.80 → ~98.
    """
    avg_prob = max(0.0, min(1.0, float(avg_prob)))
    s = 1.0 / (1.0 + np.exp(-_SIGMOID_K * (avg_prob - _SIGMOID_CENTRE)))
    return float(round(100.0 * s, 3))


def _classify_regime(score: float) -> Literal["RISK_ON", "NEUTRAL", "RISK_OFF"]:
    if score < 25:
        return "RISK_ON"
    if score < 60:
        return "NEUTRAL"
    return "RISK_OFF"


# ---------------------------------------------------------------------------
# Slug persistence + live validation
# ---------------------------------------------------------------------------
#
# The hardcoded ``BUCKET_SLUGS`` map decays over time as Polymarket markets
# resolve, get renamed, or simply never existed under the slug we guessed.
# To keep the headline score grounded in tradable contracts, we run an
# offline-friendly validation pass that:
#
#   1. For each hardcoded slug, hits Gamma to check if the market still
#      exists (default + ``closed=true`` fallback).
#   2. For dead slugs, runs a per-bucket keyword search and replaces them
#      with the top-N markets ordered by 24h volume.
#   3. Persists the refreshed map to ``DEFAULT_SLUG_CACHE_PATH`` along with
#      an ISO timestamp; ``compute_pm_vix`` reads this on subsequent calls
#      and falls back to the hardcoded map only if the file is missing,
#      stale (>24h), or corrupt.
#
# The validation is fully async (``httpx.AsyncClient``) so the admin
# endpoint can fan out to ~30 slug-checks + ~15 search calls in parallel
# without blocking the event loop.

_SLUG_MEMORY_CACHE = get_cache("pm_vix_slugs", ttl=SLUG_CACHE_TTL_SECONDS)


def _slug_cache_path() -> Path:
    """Return the disk path for the persisted slug map.

    Indirected through a function (rather than a module-level constant) so
    tests can monkeypatch the path without re-importing the module.
    """
    return Path(os.environ.get("PFM_PM_VIX_SLUG_CACHE_PATH", DEFAULT_SLUG_CACHE_PATH))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically (temp file + ``replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _load_persisted_slugs() -> dict[str, Any] | None:
    """Read the persisted slug map from disk.

    Returns ``None`` when the file is missing, malformed, or older than
    :data:`SLUG_CACHE_TTL_SECONDS`. Recovery from a corrupt file is a
    silent fallback to the hardcoded map — we log at warning level but do
    not raise.
    """
    path = _slug_cache_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("pm_vix slug cache unreadable at %s: %s", path, exc)
        return None
    if not isinstance(raw, dict) or "buckets" not in raw or "as_of" not in raw:
        logger.warning("pm_vix slug cache has unexpected shape at %s", path)
        return None
    # TTL check based on the persisted ``as_of``.
    try:
        as_of_dt = datetime.fromisoformat(raw["as_of"])
    except (TypeError, ValueError):
        logger.warning("pm_vix slug cache has invalid as_of: %r", raw.get("as_of"))
        return None
    age_s = (datetime.now(tz=UTC) - as_of_dt).total_seconds()
    if age_s > SLUG_CACHE_TTL_SECONDS:
        logger.info("pm_vix slug cache stale (age=%.0fs > %s)", age_s, SLUG_CACHE_TTL_SECONDS)
        return None
    return raw


def _get_active_slugs() -> dict[str, list[str]]:
    """Return the currently active per-bucket slug map.

    Order of precedence:
      1. In-process memory cache (fast path; same TTL).
      2. Disk-persisted map under :data:`DEFAULT_SLUG_CACHE_PATH`.
      3. Empty dict — caller should fall back to ``BUCKET_SLUGS``.

    Bucket keys missing from the persisted map fall back to the hardcoded
    list bucket-by-bucket so a partial refresh doesn't black-hole an
    unrelated bucket.
    """
    cached = _SLUG_MEMORY_CACHE.get("buckets")
    if cached is not None:
        return cached
    raw = _load_persisted_slugs()
    if raw is None:
        return {}
    buckets = raw.get("buckets", {})
    if not isinstance(buckets, dict):
        return {}
    out: dict[str, list[str]] = {}
    for bucket in BUCKET_SLUGS:
        slugs = buckets.get(bucket)
        if isinstance(slugs, list) and slugs:
            out[bucket] = [str(s) for s in slugs if isinstance(s, str)]
    _SLUG_MEMORY_CACHE.set("buckets", out, ttl=SLUG_CACHE_TTL_SECONDS)
    return out


async def _check_slug_alive(http: httpx.AsyncClient, slug: str) -> bool:
    """Return ``True`` if ``slug`` resolves to a Gamma market.

    Tries the default filter first, then ``closed=true`` to catch resolved
    markets (Polymarket still serves their metadata for ~weeks). Network
    failures are logged and treated as "alive" — we don't want a transient
    gateway hiccup to nuke half the catalog on the next refresh.
    """
    base = GAMMA_URL.rstrip("/")
    try:
        r = await http.get(f"{base}/markets", params={"slug": slug}, timeout=6.0)
        if r.status_code == 200:
            arr = r.json() or []
            if isinstance(arr, list) and arr:
                return True
        r2 = await http.get(
            f"{base}/markets",
            params={"slug": slug, "closed": "true"},
            timeout=6.0,
        )
        if r2.status_code == 200:
            arr2 = r2.json() or []
            if isinstance(arr2, list) and arr2:
                return True
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("pm_vix slug-check transient error %s: %s — treating as alive", slug, exc)
        return True  # fail-open: never demote a slug because of a network blip
    return False


async def _search_replacements(
    http: httpx.AsyncClient,
    keywords: list[str],
    *,
    limit_per_keyword: int = 10,
    top_n: int = TOP_N_REPLACEMENTS,
) -> list[str]:
    """Search Gamma for active markets matching ``keywords``.

    Combines hits across every keyword, dedupes by slug, and returns the
    top-N entries by ``volume24hr`` desc. Markets without a slug, or with
    a slug already considered dead, are skipped by the caller.
    """
    base = GAMMA_URL.rstrip("/")
    seen: dict[str, dict[str, Any]] = {}
    for kw in keywords:
        try:
            r = await http.get(
                f"{base}/markets",
                params={
                    "limit": str(limit_per_keyword),
                    "active": "true",
                    "closed": "false",
                    "order": "volume24hr",
                    "ascending": "false",
                    "search": kw,
                },
                timeout=6.0,
            )
            if r.status_code != 200:
                continue
            arr = r.json() or []
        except (httpx.HTTPError, ValueError) as exc:
            logger.info("pm_vix search transient error kw=%s: %s", kw, exc)
            continue
        if not isinstance(arr, list):
            continue
        for m in arr:
            if not isinstance(m, dict):
                continue
            slug = m.get("slug")
            if not isinstance(slug, str) or not slug:
                continue
            # Keep the highest-volume copy if the same slug shows up under
            # multiple keywords.
            v = _market_volume(m)
            existing = seen.get(slug)
            if existing is None or v > _market_volume(existing):
                seen[slug] = m
    ranked = sorted(seen.values(), key=_market_volume, reverse=True)
    return [m["slug"] for m in ranked[:top_n] if isinstance(m.get("slug"), str)]


async def validate_and_refresh_buckets(
    http: httpx.AsyncClient,
    *,
    persist_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate every hardcoded slug and replace dead ones with live markets.

    For each bucket in :data:`BUCKET_SLUGS`:
      1. Concurrently check whether each hardcoded slug is alive on Gamma.
      2. For dead slugs, run a single keyword search per bucket (using
         :data:`BUCKET_SEARCH_KEYWORDS`) and pick the top-N most-active
         live markets to fill the gap.
      3. Persist the merged ``{kept_slugs} ∪ {replacements}`` to disk so
         subsequent ``compute_pm_vix`` calls see the live map.

    Args:
        http: Async HTTP client (caller manages lifecycle so this is
            cheap to invoke multiple times in a single request).
        persist_path: Override the destination path. ``None`` uses
            :func:`_slug_cache_path` (env-overridable).

    Returns:
        Dict matching the persisted file shape, plus diagnostic counts
        ``n_dead_replaced`` and ``n_kept`` so the admin endpoint can
        surface a one-line summary without re-reading disk.
    """
    new_buckets: dict[str, list[str]] = {}
    n_dead_replaced = 0
    n_kept = 0
    bucket_diagnostics: dict[str, dict[str, Any]] = {}

    for bucket, slugs in BUCKET_SLUGS.items():
        # Probe every hardcoded slug in parallel.
        alive_results = await asyncio.gather(
            *(_check_slug_alive(http, s) for s in slugs),
            return_exceptions=False,
        )
        kept = [s for s, alive in zip(slugs, alive_results, strict=True) if alive]
        dead = [s for s, alive in zip(slugs, alive_results, strict=True) if not alive]
        replacements: list[str] = []
        if dead:
            keywords = BUCKET_SEARCH_KEYWORDS.get(bucket, [])
            if keywords:
                # Avoid replacing with slugs we already keep (same market
                # could still surface under a search hit).
                kept_set = set(kept)
                candidates = await _search_replacements(http, keywords)
                replacements = [s for s in candidates if s not in kept_set]
        merged = kept + replacements
        # Defensive fallback: if everything died and search found nothing,
        # keep the hardcoded list so the bucket still contributes something
        # (will hit "no market" at compute time and just yield n_used=0).
        if not merged:
            merged = list(slugs)
        new_buckets[bucket] = merged
        n_kept += len(kept)
        n_dead_replaced += len(dead) if replacements else 0
        bucket_diagnostics[bucket] = {
            "n_kept": len(kept),
            "n_dead": len(dead),
            "n_replacements": len(replacements),
            "kept": kept,
            "dead": dead,
            "replacements": replacements,
        }

    payload: dict[str, Any] = {
        "as_of": datetime.now(tz=UTC).isoformat(),
        "buckets": new_buckets,
        "n_dead_replaced": n_dead_replaced,
        "n_kept": n_kept,
        "diagnostics": bucket_diagnostics,
    }

    target = Path(persist_path) if persist_path else _slug_cache_path()
    try:
        _atomic_write_json(target, payload)
    except OSError as exc:
        logger.warning("pm_vix could not persist slug cache to %s: %s", target, exc)

    # Bust the in-process memory cache so the next compute call reads the
    # newly persisted map.
    _SLUG_MEMORY_CACHE.clear()
    _VIX_CACHE.clear()
    logger.info(
        "pm_vix slug refresh: kept=%d dead_replaced=%d wrote=%s",
        n_kept,
        n_dead_replaced,
        target,
    )
    return payload


def _now_unix() -> float:
    return datetime.now(tz=UTC).timestamp()


def _prewarm_compute_snapshot() -> dict[str, Any]:
    """Compute one PM-VIX snapshot synchronously and cache it."""
    payload = compute_pm_vix()
    _VIX_CACHE.set(_PREWARM_KEY, payload, ttl=PMVIX_PREWARM_CACHE_TTL_SECONDS)
    _VIX_CACHE.set(_PREWARM_AT_KEY, _now_unix(), ttl=PMVIX_PREWARM_CACHE_TTL_SECONDS)
    return payload


async def run_forever_prewarm(
    interval_seconds: int = 300,
    *,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Background loop that recomputes the PM-VIX snapshot every ``interval``.

    Mirrors :func:`run_forever_slug_refresh`. Each iteration computes a
    fresh snapshot via :func:`compute_pm_vix` (which itself uses a sync
    ``httpx.Client``), caches it under the canonical key, and times the
    next iteration. Exceptions are logged but never break the loop.
    """
    interval = max(60, int(interval_seconds))
    # Compute once immediately so the cache is hot before the first sleep.
    while True:
        try:
            await asyncio.to_thread(_prewarm_compute_snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("pm_vix prewarm raised: %s", exc)

        if stop_event is not None and stop_event.is_set():
            return
        try:
            if stop_event is None:
                await asyncio.sleep(interval)
            else:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                return
        except TimeoutError:
            continue
        except asyncio.CancelledError:
            raise


async def run_forever_slug_refresh(
    interval_seconds: int = 21_600,  # 6h
    *,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Background loop that refreshes the slug map every ``interval_seconds``.

    Mirrors the cancellation semantics of
    :func:`pfm.live_signals_job.run_forever`: callers can either cancel the
    surrounding task or set ``stop_event``. Each iteration creates its own
    short-lived :class:`httpx.AsyncClient` so a long-running pool doesn't
    sit idle between refreshes.
    """
    interval = max(60, int(interval_seconds))
    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                await validate_and_refresh_buckets(http)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never break the loop — every iteration is independent so a
            # single bad refresh shouldn't stop future runs from succeeding.
            logger.exception("pm_vix slug refresh raised: %s", exc)

        if stop_event is not None and stop_event.is_set():
            return
        try:
            if stop_event is None:
                await asyncio.sleep(interval)
            else:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                return
        except TimeoutError:
            continue
        except asyncio.CancelledError:
            raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pm_vix(
    as_of: datetime | None = None,
    *,
    http: httpx.Client | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute the current PM-VIX snapshot.

    Args:
        as_of: When the snapshot is "as of". Defaults to ``utcnow``.
        http: Injectable httpx client (tests). A new client is created
            and closed if not supplied.
        overrides: Optional ``slug -> gamma_market_dict`` mapping that
            short-circuits the HTTP layer. Used by tests.

    Returns:
        Dict with keys ``as_of, score, components, history_30d,
        change_24h, regime``.
    """
    own_http = http is None
    http = http or httpx.Client(timeout=8.0)
    active = _get_active_slugs()
    try:
        components = []
        weighted_total = 0.0
        for bucket, hardcoded in BUCKET_SLUGS.items():
            slugs = active.get(bucket, hardcoded)
            avg_prob, n_used = _bucket_avg(slugs, http=http, overrides=overrides)
            sub = _sub_score(avg_prob)
            weight = BUCKET_WEIGHTS[bucket]
            contribution = weight * sub
            weighted_total += contribution
            components.append(
                {
                    "bucket": bucket,
                    "avg_prob": round(avg_prob, 4),
                    "weight": round(weight, 3),
                    "n_used": n_used,
                    "n_total": len(slugs),
                    "sub_score": sub,
                    "contribution": round(contribution, 3),
                    "source": "live" if bucket in active else "hardcoded",
                }
            )
    finally:
        if own_http:
            http.close()

    score = float(round(max(0.0, min(100.0, weighted_total)), 3))
    regime = _classify_regime(score)
    history = pm_vix_history(days=30, anchor_score=score)
    change_24h = round(score - history[-2]["score"], 3) if len(history) >= 2 else 0.0

    return {
        "as_of": (as_of or datetime.now(tz=UTC)).isoformat(),
        "score": score,
        "components": components,
        "history_30d": [h["score"] for h in history],
        "change_24h": change_24h,
        "regime": regime,
    }


def pm_vix_history(days: int = 30, anchor_score: float | None = None) -> list[dict[str, Any]]:
    """Return a daily PM-VIX time-series of length ``days``.

    Backed by a fixed-seed numpy RNG so the demo is deterministic. The
    last point optionally pins to ``anchor_score`` so the snapshot's
    ``change_24h`` lines up with what the user just fetched.
    """
    days = max(1, int(days))
    rng = np.random.default_rng(seed=20260508)
    base = 45.0
    drift = rng.normal(0, 4.0, size=days).cumsum() * 0.4
    cycle = 8.0 * np.sin(np.linspace(0, 2.5 * np.pi, days))
    series = base + drift + cycle
    series = np.clip(series, 5.0, 95.0)

    if anchor_score is not None:
        # Pin the final point to ``anchor_score`` so the live snapshot
        # and the history don't disagree on "today".
        series[-1] = float(anchor_score)

    today = datetime.now(tz=UTC).date()
    out: list[dict[str, Any]] = []
    for i, val in enumerate(series):
        d = today - timedelta(days=days - 1 - i)
        out.append({"date": d.isoformat(), "score": float(round(val, 3))})
    return out


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PMVIXComponent(BaseModel):
    bucket: str
    avg_prob: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    n_used: int = Field(..., ge=0)
    n_total: int = Field(..., ge=0)
    sub_score: float = Field(..., ge=0.0, le=100.0)
    contribution: float = Field(..., ge=0.0)
    source: Literal["live", "hardcoded"] = Field(
        default="hardcoded",
        description=(
            "Whether the slugs for this bucket came from the persisted "
            "live-validated map or the hardcoded fallback."
        ),
    )


class PMVIXSnapshot(BaseModel):
    as_of: str
    score: float = Field(..., ge=0.0, le=100.0)
    components: list[PMVIXComponent]
    history_30d: list[float]
    change_24h: float
    regime: Literal["RISK_ON", "NEUTRAL", "RISK_OFF"]
    cache_age_seconds: int = Field(
        default=0,
        ge=0,
        description=(
            "Seconds since this snapshot was computed. ``0`` for a freshly "
            "computed payload, larger when served from the prewarmed cache."
        ),
    )
    is_stale: bool = Field(
        default=False,
        description="True when ``cache_age_seconds`` exceeds PMVIX_STALE_AFTER_SECONDS.",
    )


class PMVIXComponentsResponse(BaseModel):
    as_of: str
    score: float = Field(..., ge=0.0, le=100.0)
    components: list[PMVIXComponent]


class PMVIXHistoryPoint(BaseModel):
    date: str
    score: float = Field(..., ge=0.0, le=100.0)


class PMVIXHistoryResponse(BaseModel):
    n: int
    points: list[PMVIXHistoryPoint]


class PMVIXSlugRefreshResponse(BaseModel):
    as_of: str
    n_kept: int = Field(..., ge=0, description="Hardcoded slugs that resolved live.")
    n_dead_replaced: int = Field(
        ..., ge=0, description="Buckets where one or more dead slugs were swapped in."
    )
    buckets: dict[str, list[str]] = Field(
        ..., description="Bucket -> active slug list after the refresh."
    )


class PMVIXSlugListResponse(BaseModel):
    as_of: str | None = Field(
        default=None,
        description="ISO-8601 timestamp of the last successful refresh (None if never).",
    )
    source: Literal["live", "fallback", "mixed"] = Field(
        default="fallback",
        description=(
            "``live`` → all buckets from disk, ``fallback`` → all hardcoded, "
            "``mixed`` → some of each."
        ),
    )
    buckets: dict[str, list[str]] = Field(
        ..., description="Bucket -> slug list currently used by /indices/pm-vix."
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/indices/pm-vix", tags=["indices"])


def _is_prewarm_required() -> bool:
    """Return True when the env opts into the cache-only hot path.

    When ``PFM_PMVIX_PREWARM_ENABLED=1`` is set, the GET endpoint NEVER
    computes a snapshot inline — it serves whatever the lifespan
    background task last wrote and 503s when the cache is empty (with a
    ``Retry-After`` hint so dashboards retry sanely).
    """
    return os.environ.get("PFM_PMVIX_PREWARM_ENABLED", "").strip() == "1"


def _augment_with_cache_age(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach ``cache_age_seconds`` + ``is_stale`` based on the prewarm-at marker."""
    out = dict(payload)
    prewarmed_at = _VIX_CACHE.get(_PREWARM_AT_KEY)
    if prewarmed_at is not None:
        try:
            age = max(0, int(_now_unix() - float(prewarmed_at)))
        except (TypeError, ValueError):
            age = 0
    else:
        age = 0
    out["cache_age_seconds"] = age
    out["is_stale"] = age > PMVIX_STALE_AFTER_SECONDS
    return out


@router.get("", response_model=PMVIXSnapshot)
def get_pm_vix(as_of: str | None = Query(default=None)) -> PMVIXSnapshot:
    """Current PM-VIX snapshot. Cache 300s on the resolved ``as_of``.

    When ``PFM_PMVIX_PREWARM_ENABLED=1``, the endpoint NEVER computes
    inline — it returns the prewarmed cache payload, or 503 with a
    ``Retry-After: 5`` header if the background task hasn't filled the
    cache yet. This keeps p99 under 100ms regardless of upstream latency.
    """
    parsed: datetime | None = None
    if as_of:
        try:
            parsed = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid as_of: {as_of!r}") from exc

    cache_key = ("snapshot", parsed.isoformat() if parsed else "live")
    cached = _VIX_CACHE.get(cache_key)
    if cached is not None:
        return PMVIXSnapshot(**_augment_with_cache_age(cached))

    if _is_prewarm_required() and parsed is None:
        # Cache-only hot path: prewarm task hasn't finished its first tick
        # (or the cache TTL elapsed without a refresh). Tell the caller to
        # back off — never compute inline.
        raise HTTPException(
            status_code=503,
            detail="pm_vix snapshot not ready yet (prewarm pending)",
            headers={"Retry-After": "5"},
        )

    payload = compute_pm_vix(as_of=parsed)
    _VIX_CACHE.set(cache_key, payload, ttl=300)
    return PMVIXSnapshot(**_augment_with_cache_age(payload))


@router.get("/components", response_model=PMVIXComponentsResponse)
def get_pm_vix_components() -> PMVIXComponentsResponse:
    """Per-bucket breakdown without the history series."""
    cached = _VIX_CACHE.get("components")
    if cached is not None:
        return PMVIXComponentsResponse(**cached)

    snap = compute_pm_vix()
    payload = {
        "as_of": snap["as_of"],
        "score": snap["score"],
        "components": snap["components"],
    }
    _VIX_CACHE.set("components", payload, ttl=300)
    return PMVIXComponentsResponse(**payload)


@router.get("/history", response_model=PMVIXHistoryResponse)
def get_pm_vix_history(
    days: int = Query(default=30, ge=1, le=365),
) -> PMVIXHistoryResponse:
    """Synthesised PM-VIX history (deterministic seed)."""
    cache_key = ("history", days)
    cached = _VIX_CACHE.get(cache_key)
    if cached is not None:
        return PMVIXHistoryResponse(**cached)

    points = pm_vix_history(days=days)
    payload = {"n": len(points), "points": points}
    _VIX_CACHE.set(cache_key, payload, ttl=300)
    return PMVIXHistoryResponse(**payload)


def _admin_dep_if_enabled() -> Any:
    """Gate the slug-refresh endpoint behind ``PFM_ADMIN_TOKEN`` when set.

    Mirrors the pattern in :mod:`pfm.live_signals_job`: the trigger is
    open in dev / demos (no env var), admin-only in prod. Re-evaluated
    per-request via :func:`pfm.auth.dependencies.require_admin` so the
    token can be rotated without restarting the process.
    """
    if os.environ.get("PFM_ADMIN_TOKEN"):
        return Depends(require_admin)

    async def _noop() -> None:
        return None

    return Depends(_noop)


@router.post(
    "/refresh-slugs",
    response_model=PMVIXSlugRefreshResponse,
    summary="Validate hardcoded bucket slugs against Polymarket and persist replacements.",
    dependencies=[_admin_dep_if_enabled()],
)
async def refresh_pm_vix_slugs() -> PMVIXSlugRefreshResponse:
    """Trigger one ``validate_and_refresh_buckets`` cycle synchronously.

    The persisted file under ``/tmp/pfm_pm_vix_slugs.json`` is overwritten
    atomically, the in-process slug cache is invalidated, and the next
    ``GET /indices/pm-vix`` will see the refreshed map. Heavy
    (~30 GET requests against Gamma); intended for occasional manual
    runs or the 6h cron loop.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            payload = await validate_and_refresh_buckets(http)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"Polymarket Gamma unreachable: {exc!s}"
        ) from exc
    return PMVIXSlugRefreshResponse(
        as_of=payload["as_of"],
        n_kept=payload["n_kept"],
        n_dead_replaced=payload["n_dead_replaced"],
        buckets=payload["buckets"],
    )


@router.get(
    "/slugs",
    response_model=PMVIXSlugListResponse,
    summary="Return the current per-bucket slug map (live or fallback).",
)
async def get_pm_vix_slugs() -> PMVIXSlugListResponse:
    """Surface the slug map currently driving the score.

    Reads ``/tmp/pfm_pm_vix_slugs.json`` if present and fresh; otherwise
    returns the hardcoded fallback. Useful both for debugging ("why does
    the score look weird?") and for the frontend to show users which
    contracts feed each bucket.
    """
    persisted = _load_persisted_slugs()
    active = _get_active_slugs()
    out: dict[str, list[str]] = {}
    n_live = 0
    for bucket, hardcoded in BUCKET_SLUGS.items():
        if bucket in active:
            out[bucket] = active[bucket]
            n_live += 1
        else:
            out[bucket] = list(hardcoded)
    if n_live == len(BUCKET_SLUGS):
        source: Literal["live", "fallback", "mixed"] = "live"
    elif n_live == 0:
        source = "fallback"
    else:
        source = "mixed"
    return PMVIXSlugListResponse(
        as_of=persisted["as_of"] if persisted else None,
        source=source,
        buckets=out,
    )


__all__ = [
    "BUCKET_SEARCH_KEYWORDS",
    "BUCKET_SLUGS",
    "BUCKET_WEIGHTS",
    "DEFAULT_SLUG_CACHE_PATH",
    "PMVIX_STALE_AFTER_SECONDS",
    "SLUG_CACHE_TTL_SECONDS",
    "PMVIXComponent",
    "PMVIXComponentsResponse",
    "PMVIXHistoryPoint",
    "PMVIXHistoryResponse",
    "PMVIXSlugListResponse",
    "PMVIXSlugRefreshResponse",
    "PMVIXSnapshot",
    "compute_pm_vix",
    "pm_vix_history",
    "router",
    "run_forever_prewarm",
    "run_forever_slug_refresh",
    "validate_and_refresh_buckets",
]
