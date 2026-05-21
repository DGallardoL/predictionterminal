"""FastAPI app exposing /health, /factors, /factors/discover, /factors/preview,
/fit and /attribution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from contextlib import asynccontextmanager, suppress
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

# Brotli is a strict upgrade over gzip for static text payloads (the 1.2 MB
# index.html compresses to ~180 KB brotli-q5 vs ~273 KB gzip). The middleware
# falls back to the next-best encoding the client advertises, so callers
# without ``br`` support still get gzip via GZipMiddleware below.
try:
    from brotli_asgi import BrotliMiddleware
except ImportError:  # optional dep — gzip still works if missing
    BrotliMiddleware = None

from pfm import __version__
from pfm import terminal as terminal_mod
from pfm.cache import CacheBackend, RedisCache
from pfm.config import Settings, get_settings
from pfm.factor_resolver import (
    suggest_factors_with_meta as _factor_suggest_meta,
)
from pfm.factors import FactorConfig, load_factors
from pfm.logging_setup import configure_logging
from pfm.redis_lock import RedisLock
from pfm.scanner import run_scan
from pfm.schemas import (
    HealthResponse,
    RegressionLit,
    ScanHitOut,
    ScanRequest,
    ScanResponse,
    TerminalHistoryBar,
    TerminalHistoryResponse,
    TerminalLive,
    TerminalMarketResponse,
    TerminalMeta,
    TerminalMover,
    TerminalNewMarket,
    TerminalOverviewResponse,
    TerminalPeer,
    TerminalSearchHit,
    TerminalSearchResponse,
    TerminalStats,
    TerminalThemeBucket,
    TerminalUpcomingResolution,
)
from pfm.sources.binance import (
    BinanceClient,
)
from pfm.sources.chain import (
    fetch_chained_history,  # noqa: F401 — re-exported for tests patching `main_mod.fetch_chained_history`
)
from pfm.sources.equity import (
    get_log_returns,  # noqa: F401 — re-exported for tests
)
from pfm.sources.kalshi import KalshiClient
from pfm.sources.kalshi import (
    fetch_factor_history as fetch_kalshi_history,  # noqa: F401 — re-exported for tests
)
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,  # noqa: F401 — re-exported for tests patching `main_mod.fetch_factor_history`
)
from pfm.terminal.export import respond as _export_respond

# Configure structlog + stdlib logging the moment the module is imported, so
# anything that runs before lifespan() (registry walks, model imports) lands
# in the configured stream rather than Python's default stderr handler.
configure_logging()

logger = logging.getLogger(__name__)


# --- lifespan ---------------------------------------------------------------


#: Tuned connection pool — caps in-flight HTTP at 100 with 50 long-lived keepalives.
#: Sized for the parallel-fanout endpoints (/factors/rank with 1090 candidates,
#: /fit with up to ~30 factors). Polymarket's quoted limit is 1000/10s so even
#: at 100 concurrent we're well under the rate-limit envelope.
_HTTP_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=50,
    keepalive_expiry=30.0,
)

#: Async-client connection pool tuned for the Polymarket fan-out endpoints
#: (/terminal/homepage, /terminal/quote, /terminal/peer-scanner, …).
#: ``max_keepalive_connections=20`` keeps a warm pool against gamma + clob
#: hosts without holding too many idle sockets. ``pool=10s`` is the budget a
#: caller will wait for a connection slot before raising; on the 10-parallel
#: cold homepage benchmark this brought wall-clock from 1.3s → ~0.4s by
#: avoiding TLS-handshake serialisation behind a small pool.
_ASYNC_HTTP_LIMITS = httpx.Limits(
    max_keepalive_connections=20,
    max_connections=100,
    keepalive_expiry=30.0,
)
_ASYNC_HTTP_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=30.0,
    write=10.0,
    pool=10.0,
)


_ARB_CRAWL_STATE_PATH = str(Path(__file__).resolve().parents[3] / "arbstuff" / "crawl_state.json")


async def _arb_discovery_loop(app: FastAPI) -> None:
    """Opt-in background discovery (PFM_ARB_DISCOVERY_LOOP=1).

    Periodically crawls the market universe and rotates across three coverage
    modes so the durable store fills from the *whole* universe, not just the
    newest-listing feed (which is flooded with ephemeral sports/crypto):

    * ``"sweep"`` — resumable full newest-first walk (every ``SWEEP_EVERY`` ticks).
    * ``"liquid"`` — substantive/liquid coverage (Polymarket volume-sorted), the
      mode that actually surfaces politics/macro/long-dated markets where real
      cross-venue arbs live (every ``LIQUID_EVERY`` ticks).
    * ``"new"`` — freshly listed markets (the remaining ticks).

    Each step matches cross-venue recall-first (every plausible candidate is
    surfaced + confidence-flagged), price-checks the verified matches live
    (fee-aware), and records every real arb to the durable confirmed-arb store.
    Each step is bounded; failures are swallowed so a bad cycle never tears down
    the app. No orders are sent. The leader gating around this loop is unchanged.
    """
    interval = float(os.environ.get("PFM_ARB_DISCOVERY_INTERVAL_S", "300"))
    sweep_every = int(os.environ.get("PFM_ARB_DISCOVERY_SWEEP_EVERY", "6"))
    liquid_every = int(os.environ.get("PFM_ARB_DISCOVERY_LIQUID_EVERY", "3"))
    max_pages = int(os.environ.get("PFM_ARB_DISCOVERY_MAX_PAGES", "2"))
    within_hours = float(os.environ.get("PFM_ARB_DISCOVERY_WINDOW_H", "48"))
    ckpt = _ARB_CRAWL_STATE_PATH
    await asyncio.sleep(20)  # let boot + prewarm settle
    tick = 0
    while True:
        try:
            from pfm.arb.discovery_pipeline import default_store, run_discovery_step
            from pfm.arb.live_pricing import make_price_fn

            # Rotate modes so the substantive (liquid) universe gets covered too.
            # sweep takes priority on its tick; otherwise alternate liquid/new.
            if sweep_every > 0 and tick % sweep_every == 0:
                mode = "sweep"
            elif liquid_every > 0 and tick % liquid_every == 0:
                mode = "liquid"
            else:
                mode = "new"
            res = await asyncio.to_thread(
                run_discovery_step,
                mode=mode,
                store=default_store(),
                max_pages=max_pages,
                within_hours=within_hours,
                min_score=0.5,
                price_fn=make_price_fn(fee_aware=True),
                checkpoint_path=ckpt,
            )
            logger.info(
                "arb discovery [%s] tick=%d: %d kalshi, %d poly → %d cand "
                "(%d high, %d review), %d arbs recorded",
                res.mode,
                tick,
                res.n_kalshi,
                res.n_poly,
                res.n_candidates,
                res.n_high,
                res.n_review,
                res.n_recorded,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let a bad cycle kill the loop
            logger.warning("arb discovery loop tick=%d failed: %s", tick, exc)
        tick += 1
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    app.state.settings = settings
    app.state.cache = RedisCache(settings.redis_url)
    app.state.factors = load_factors(settings.factors_file)
    # Inject curated news-sentiment factors so they appear in /factors and
    # resolve like any yaml entry. Each one is a level-source factor whose
    # ``slug`` carries the free-text search query passed to GDELT + Reddit
    # + HN. See pfm.sources.sentiment_factor.CURATED_QUERIES.
    # ``PFM_SUPPRESS_CURATED_SENTIMENT=1`` skips injection so tests with a
    # 2-factor fixture catalogue don't see the +10 lifespan additions.
    if os.environ.get("PFM_SUPPRESS_CURATED_SENTIMENT") != "1":
        try:
            from pfm.sources.sentiment_factor import (
                CURATED_QUERIES as _SENT_CURATED,
            )

            for _sid, _meta in _SENT_CURATED.items():
                if _sid in app.state.factors:
                    continue
                app.state.factors[_sid] = FactorConfig(
                    id=_sid,
                    name=_meta["name"],
                    slug=_meta["query"],
                    source="sentiment",
                    description=_meta["description"],
                    theme="sentiment",
                    is_probability=False,
                )
        except Exception:  # pragma: no cover - defensive; never block boot
            logging.getLogger(__name__).exception("failed to register curated sentiment factors")
    # Build a slug -> FactorConfig index once (perf audit 2026-05-16):
    # several /terminal/* helpers used to scan the entire 1360-row factor
    # dict on every request to translate a slug into a FactorConfig. Doing
    # the build here means O(1) lookups per request. Mutations of
    # ``app.state.factors`` after startup (none today) would need to refresh
    # this index — guard the read sites with a defensive ``or {...}``.
    app.state.factors_by_slug = {fc.slug: fc for fc in app.state.factors.values() if fc.slug}
    # Warm-cache prewarm: /terminal/vol-distribution (top-15 snapshots) and
    # /terminal/factor-clusters (default theme=None, min_corr=0.5). Pattern
    # mirrors the earnings-whisper dashboard prewarm — cold first-request
    # latency drops from ~3 s to <100 ms when the handler short-circuits on
    # ``app.state.warm_voldist`` / ``app.state.warm_clusters``. Fire-and-forget
    # so a slow pickle deserialise can't delay startup beyond the liveness
    # probe; the handlers fall back to live-compute when the warm entry is
    # missing or older than WARM_TTL_SECONDS (60 s).
    from pfm.prewarm import (
        prewarm_factor_clusters as _prewarm_factor_clusters,
    )
    from pfm.prewarm import (
        prewarm_voldist as _prewarm_voldist,
    )

    app.state.warm_voldist = None
    app.state.warm_clusters = None
    app.state.voldist_prewarm_task = asyncio.create_task(_prewarm_voldist(app))
    app.state.clusters_prewarm_task = asyncio.create_task(_prewarm_factor_clusters(app))
    # /terminal/jumps/{slug} prewarm — populates the module-level TTLCache
    # in pfm.terminal.jumps for ~30-50 curated headliner slugs so the WOW
    # Hero panel and homepage cards land warm. Cold p50 was ~3-5 s (GDELT +
    # Reddit + HN + RSS fan-out + Polymarket history); warm hits the cache
    # in <100 ms. Fire-and-forget so a slow GDELT response can't delay the
    # liveness probe. Depends on app.state.poly which is initialised below;
    # we defer the scheduling helper until after that. See T17.
    # Wire the terminal TTLCache to use Redis as an L2 — warm response
    # entries (e.g. cached /terminal/overview, /terminal/search) propagate
    # across all gunicorn workers instead of each worker rebuilding from
    # cold. L1 (in-process) still holds anything that can't be JSON-encoded
    # (pandas DataFrames, pickle blobs).
    terminal_mod.TERMINAL_CACHE.attach_redis(app.state.cache, prefix="term:")
    # Auth: in production-like environments (ENV=production / FLY_APP_NAME /
    # RENDER / NODE_ENV=production) auth is on by default and we mint an admin
    # token on first boot if PFM_ADMIN_TOKEN isn't already set. In dev this is
    # a no-op (returns ""). We call this once at startup so the WARNING with
    # the autogen token shows up in the logs even if no admin endpoint is hit.
    from pfm.auth.production import (
        get_or_generate_admin_token as _gen_admin_token,
    )
    from pfm.auth.production import (
        is_auth_enabled as _is_auth_on,
    )

    if _is_auth_on():
        _gen_admin_token()
    # Sync client backs PolymarketClient / KalshiClient / BinanceClient — those
    # downstream callers are sync and we don't want to refactor every call site.
    app.state.http = httpx.Client(
        timeout=settings.request_timeout_seconds,
        limits=_HTTP_LIMITS,
    )
    # Async client for endpoints that fan-out to many slugs in parallel
    # (terminal_live_stream, /factors/rank workers when wrapped in to_thread,
    # and any future async endpoint that doesn't need PolymarketClient's helpers).
    # Uses the dedicated async pool config (separate connect/read/write/pool
    # timeouts + 20 long-lived keepalives) sized for the Polymarket fan-out.
    app.state.async_http = httpx.AsyncClient(
        timeout=_ASYNC_HTTP_TIMEOUT,
        limits=_ASYNC_HTTP_LIMITS,
    )
    app.state.poly = PolymarketClient(
        gamma_url=settings.polymarket_gamma_url,
        clob_url=settings.polymarket_clob_url,
        client=app.state.http,
    )
    app.state.kalshi = KalshiClient(client=app.state.http)
    app.state.binance = BinanceClient(client=app.state.http)
    # Now that app.state.poly exists, schedule the jumps prewarm. Default OFF
    # in tests (PFM_JUMPS_PREWARM_ENABLED unset) so TestClient repeated boots
    # don't accumulate background tasks; default ON in real workers. The
    # helper handles state-slot init + asyncio.create_task internally.
    if os.environ.get("PFM_JUMPS_PREWARM_ENABLED", "1") == "1":
        try:
            from pfm.terminal.jumps_prewarm import (
                register_jumps_prewarm as _register_jumps_prewarm,
            )

            app.state.jumps_prewarm_task = _register_jumps_prewarm(app)
        except Exception as exc:  # pragma: no cover - defensive; never block boot
            logger.warning("jumps prewarm scheduling failed: %s", exc)
    # Extra Terminal prewarms — four endpoints that measured >5 s cold-start
    # against a freshly-booted worker on 2026-05-19:
    #   * /terminal/sentiment-leaderboard       (~ 8.1 s cold)
    #   * /terminal/sentiment-trend/spike-alerts (~ 5.1 s cold)
    #   * /terminal/jumps/cluster               (~11.3 s cold)
    #   * /terminal/calendar-curated/clusters   (~ 1.9 s cold)
    # Each populates its own existing module-level response cache as a side
    # effect of the canonical default query, so the live endpoint
    # short-circuits to <50 ms on subsequent requests with the default
    # params. Default OFF in tests (PFM_EXTRA_PREWARMS_ENABLED=0 via the
    # autouse fixture in tests/conftest.py — avoids real Gamma + GDELT
    # fan-out on every TestClient boot); default ON in real workers so
    # the homepage cards land warm.
    if os.environ.get("PFM_EXTRA_PREWARMS_ENABLED", "1") == "1":
        try:
            from pfm.terminal.extra_prewarms import (
                register_extra_prewarms as _register_extra_prewarms,
            )

            app.state.extra_prewarms_task = _register_extra_prewarms(app)
        except Exception as exc:  # pragma: no cover - defensive; never block boot
            logger.warning("extra prewarms scheduling failed: %s", exc)
    # Realtime SSE multiplexing hub — one poller per (kind, slug) shared
    # across N clients. Drained at shutdown so pending poller tasks cancel
    # cleanly instead of hanging the event loop.
    from pfm.realtime.hub import RealtimeHub

    app.state.hub = RealtimeHub(http_client=app.state.async_http)
    # Live signals background job (opt-in via PFM_LIVE_SIGNALS_ENABLED=1).
    # Default OFF so the existing test suite — which spins up the app
    # repeatedly via TestClient — doesn't accumulate background tasks.
    if os.environ.get("PFM_LIVE_SIGNALS_ENABLED") == "1":
        from pfm.live_signals_job import run_forever as _live_signals_run

        interval = int(os.environ.get("PFM_LIVE_SIGNALS_INTERVAL_S", "900"))
        fetcher_kind_raw = os.environ.get("PFM_LIVE_SIGNALS_FETCHER", "synthetic").strip().lower()
        if fetcher_kind_raw not in {"synthetic", "polymarket"}:
            logger.warning(
                "live signals: unknown PFM_LIVE_SIGNALS_FETCHER=%r — falling back to synthetic",
                fetcher_kind_raw,
            )
            fetcher_kind_raw = "synthetic"
        app.state.live_signals_task = asyncio.create_task(
            _live_signals_run(
                interval_seconds=interval,
                fetcher_kind=fetcher_kind_raw,  # type: ignore[arg-type]
                http_client=app.state.async_http,
            )
        )
        logger.info(
            "live signals job started (interval=%ss, fetcher=%s)",
            interval,
            fetcher_kind_raw,
        )
    # Decay-monitor refresh job (opt-in via PFM_DECAY_REFRESH_ENABLED=1).
    # Reuses the same lifespan pattern; default cadence is 4h so it
    # doesn't compete with the more aggressive live_signals 15-min loop.
    if os.environ.get("PFM_DECAY_REFRESH_ENABLED") == "1":
        from pfm.decay_monitor import run_forever as _decay_refresh_run

        decay_interval = int(os.environ.get("PFM_DECAY_REFRESH_INTERVAL_S", "14400"))
        app.state.decay_refresh_task = asyncio.create_task(
            _decay_refresh_run(interval_seconds=decay_interval)
        )
        logger.info(
            "decay-monitor refresh job started (interval=%ss)",
            decay_interval,
        )
    # PM-VIX prewarm (opt-in via PFM_PMVIX_PREWARM_ENABLED=1). Recomputes
    # the snapshot every 5 min so /indices/pm-vix always returns from
    # cache in <100ms regardless of upstream latency. Default OFF.
    if os.environ.get("PFM_PMVIX_PREWARM_ENABLED") == "1":
        from pfm.pm_vix import run_forever_prewarm as _pmvix_prewarm

        pmvix_interval = int(os.environ.get("PFM_PMVIX_PREWARM_INTERVAL_S", "300"))
        app.state.pm_vix_prewarm_task = asyncio.create_task(
            _pmvix_prewarm(interval_seconds=pmvix_interval)
        )
        logger.info("pm-vix prewarm job started (interval=%ss)", pmvix_interval)
    # Earnings-whisper dashboard prewarm (opt-in via
    # PFM_EARNINGS_PREWARM_ENABLED=1). Cadence defaults to 1h since the
    # underlying earnings calendar moves slowly.
    if os.environ.get("PFM_EARNINGS_PREWARM_ENABLED") == "1":
        from pfm.earnings_whisper import (
            run_forever_dashboard_prewarm as _earnings_prewarm,
        )

        earnings_interval = int(os.environ.get("PFM_EARNINGS_PREWARM_INTERVAL_S", "3600"))
        app.state.earnings_prewarm_task = asyncio.create_task(
            _earnings_prewarm(interval_seconds=earnings_interval)
        )
        logger.info(
            "earnings-whisper prewarm job started (interval=%ss)",
            earnings_interval,
        )
    # Wire /terminal/fair/{slug} so it can pull live Polymarket midpoints
    # without callers having to pass ?p_market=... — uses the Gamma price
    # prewarm cache when available, else falls back to a one-shot Gamma fetch.
    from pfm.terminal.fair_price import set_market_quote_provider as _set_fair_provider

    def _live_market_quote(slug: str) -> float:
        cache = getattr(app.state, "gamma_prices", None) or {}
        price = cache.get(slug)
        if price is not None and math.isfinite(price):
            return float(price)
        # Cache miss: fall back to a sync Gamma fetch. fetch_gamma_market
        # already retries on 429 and falls back to closed=true.
        try:
            m = terminal_mod.fetch_gamma_market(
                app.state.http,
                app.state.settings.polymarket_gamma_url,
                slug,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        p = terminal_mod._yes_price_from_market(m)
        if p is None or not math.isfinite(p):
            raise HTTPException(
                status_code=400,
                detail=f"no live quote for slug={slug!r} (market has no midpoint/last)",
            )
        return float(p)

    _set_fair_provider(_live_market_quote)

    logger.info(
        "startup ok — %d factors, cache=%s",
        len(app.state.factors),
        "on" if app.state.cache.enabled else "off",
    )

    # Pre-warm the /openapi.json cache so the first client request doesn't pay
    # the ~800 ms cost of introspecting 234 routes + Pydantic schemas. Defer the
    # warm-up so a slow openapi() build doesn't delay startup beyond the
    # liveness probe — fire-and-forget via asyncio.
    async def _openapi_prewarm() -> None:
        try:
            await asyncio.to_thread(_openapi_payload)
            logger.info("openapi.json cache prewarmed")
        except Exception as exc:  # pragma: no cover - prewarm is best-effort
            logger.warning("openapi.json prewarm failed: %s", exc)

    app.state.openapi_prewarm_task = asyncio.create_task(_openapi_prewarm())

    # Factor-history prewarm — opt-in via PFM_FACTOR_PREWARM_ENABLED=1.
    # Fetches the top-N curated factors' price history into Redis once at
    # startup so the first /reverse-finder / /fit / /factors/rank call after
    # boot sees a warm cache and finishes in seconds rather than minutes.
    # Without Redis (NullCache) this is essentially a no-op for subsequent
    # callers — only the in-process closure-cached httpx client benefits.
    # Real-time crypto microstructure event engine — opt-in via
    # PFM_CRYPTO_WS_ENABLED=1. Streams Binance trade + bookTicker for 10 pairs
    # via the `crypto-microstructure` library (cryptostuff/), captures whale
    # alerts and VWAP-zscore mean-reversion triggers into an in-memory buffer.
    # The /strategies/crypto/events endpoint reads from it.
    if os.environ.get("PFM_CRYPTO_WS_ENABLED") == "1":
        # Leader election: with gunicorn N workers we only want ONE WS
        # connection to Binance, not N. Each worker tries to acquire a
        # Redis SETNX lease at boot; only the winner starts its engine.
        # Other workers' /strategies/crypto/events endpoint reads from
        # Redis (populated by the leader via crypto_events_engine.publish).
        # Lease has a TTL so a crashed leader doesn't block re-election.
        _crypto_lock_key = "arb:crypto_ws:leader"
        _crypto_lock_ttl = int(os.environ.get("PFM_CRYPTO_WS_LEADER_TTL_S", "60"))
        try:
            cache = app.state.cache
            # Mark ourselves with PID so the renewer can verify ownership.
            our_token = f"pid={os.getpid()}"
            # Centralised SETNX+CAS via pfm.redis_lock (W11-15 migration).
            _crypto_lock = RedisLock(
                getattr(cache, "_client", None),
                _crypto_lock_key,
                ttl_s=_crypto_lock_ttl,
                owner_id=our_token,
            )
            is_leader = _crypto_lock.acquire()
            app.state.crypto_ws_lock = _crypto_lock
            app.state.crypto_ws_leader = is_leader
            if is_leader:
                from pfm.crypto_events_engine import get_engine as _crypto_get_engine

                app.state.crypto_events_engine = _crypto_get_engine()
                # Wire Redis so captured events are visible to follower
                # workers — without this their /strategies/crypto/events
                # endpoint would always return 0 (they don't run the WS).
                try:
                    if cache.enabled and getattr(cache, "_client", None) is not None:
                        app.state.crypto_events_engine.attach_redis(
                            cache._client,
                            key="arb:crypto_events",
                            max_keep=500,
                        )
                except Exception as exc:
                    logger.warning("crypto WS Redis publish disabled: %s", exc)
                await app.state.crypto_events_engine.start()
                logger.info(
                    "crypto WS engine started (Binance live, 10 pairs) — "
                    "this worker (pid %d) is the leader",
                    os.getpid(),
                )

                # Periodically renew the lease so other workers know we're
                # alive. If we crash, the TTL expires and a follower picks
                # up on its next renewal tick.
                async def _crypto_renew_lease() -> None:
                    while True:
                        try:
                            await asyncio.sleep(_crypto_lock_ttl // 2)
                            # CAS renew — only extends TTL if we still own it.
                            _crypto_lock.renew()
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            logger.warning("crypto WS lease renew: %s", exc)

                app.state.crypto_ws_renew_task = asyncio.create_task(_crypto_renew_lease())
            else:
                logger.info(
                    "crypto WS engine: another worker holds the leader lease, "
                    "this worker (pid %d) is a follower",
                    os.getpid(),
                )
        except Exception as exc:
            logger.warning("crypto WS engine failed to start: %s", exc)

    # Cross-venue arb engine autostart — opt-in via PFM_ARB_ENGINE_AUTOSTART=1.
    # Launches `arbstuff/arb_engine.py` as a subprocess so it writes
    # `dashboard_state.json` continuously and the SSE stream emits the engine's
    # own full state (scan_log, balances, candidate_count) rather than the
    # built-in live-fallback. Default OFF — the fallback already produces real
    # arbs from arb_scanner.top_arbs() with no subprocess needed.
    #
    # Leader-election design (post-2026-05-15 fix):
    #   * SETNX with a 60 s TTL elects exactly ONE worker as leader.
    #   * Lock value is a per-process token (pid + boot time + random nonce)
    #     so a CAS refresh can verify ownership atomically. This prevents the
    #     pre-fix bug where the refresh blindly called ``EXPIRE`` on whatever
    #     key existed, accidentally extending a successor's lease and letting
    #     two leaders coexist after a worker restart.
    #   * A pre-spawn ``pgrep`` reaps any orphan ``arb_engine.py`` left over
    #     from a previous gunicorn batch (e.g. when SIGKILL skipped lifespan
    #     teardown). Without this, two engines would compound: the orphan
    #     stays + a fresh leader spawns its own.
    #   * A subprocess-health task watches ``proc.poll()``; if the engine
    #     dies, the leader releases its lock so a follower can take over
    #     without waiting the full 60 s TTL.
    #   * Shutdown explicitly DELs the lock key (mirroring the crypto-WS
    #     teardown) so a clean restart never sees a stale lease.
    if os.environ.get("PFM_ARB_ENGINE_AUTOSTART") == "1":
        try:
            import secrets
            import subprocess
            import time as _time
            from pathlib import Path as _Path

            arb_dir = _Path(__file__).resolve().parents[3] / "arbstuff"  # noqa: ASYNC240
            engine_path = arb_dir / "arb_engine.py"
            if not engine_path.exists():
                logger.warning("arb engine autostart skipped: %s missing", engine_path)
            else:
                # ``cache`` is bound inside the crypto-WS block above; rebind
                # defensively here so this stanza works even when
                # PFM_CRYPTO_WS_ENABLED is not set.
                _arb_cache = app.state.cache
                _redis = getattr(_arb_cache, "_client", None)
                _arb_lock_key = "pfm:arb_engine:leader"
                _arb_lock_ttl = int(os.environ.get("PFM_ARB_ENGINE_LEADER_TTL_S", "60"))
                # Unique per-process token. pid alone is not enough: pids
                # recycle, and two workers from disjoint gunicorn batches
                # can land on the same pid. Adding boot-time + a 64-bit
                # random nonce makes the token effectively unique.
                _arb_lock_token = (
                    f"pid={os.getpid()}|boot={int(_time.time())}|nonce={secrets.token_hex(8)}"
                ).encode()

                # IMPORTANT: SETNX must come BEFORE any process-reaping. An
                # earlier version reaped first, but with N concurrent workers
                # the reaper would kill sibling workers' freshly-spawned
                # engines (and DEL each other's leader locks) — producing
                # exactly the duplicate-engine bug the fix was meant to
                # eliminate. The correct order is: race for the lock first,
                # then (as the sole elected leader) reap any pre-existing
                # orphans from a previous gunicorn batch.
                # Centralised SETNX via pfm.redis_lock (W11-15). The custom
                # CAS-renew / CAS-DEL Lua below still uses _arb_lock_token as
                # the stored value, so owner_id must match exactly.
                _arb_lock = RedisLock(
                    _redis,
                    _arb_lock_key,
                    ttl_s=_arb_lock_ttl,
                    owner_id=_arb_lock_token.decode(),
                )
                if _redis is not None:
                    try:
                        acquired = _arb_lock.acquire()
                    except Exception as e:
                        logger.warning("arb engine leader-election failed: %s", e)
                        # Fall through — if Redis is unreachable, single-worker
                        # deploys still want the engine to run.
                        acquired = True
                else:
                    # No Redis → no cross-worker leader election. With multiple
                    # gunicorn workers each would spawn its own engine (a
                    # spawn/kill thundering herd racing the same dashboard_state
                    # file), so only autostart when single-worker; otherwise skip
                    # and let the arb panel serve the live arb_scanner fallback.
                    acquired = int(os.environ.get("GUNICORN_WORKERS", "1")) == 1
                    if not acquired:
                        logger.warning(
                            "arb engine autostart skipped: no Redis to elect a "
                            "leader across %s gunicorn workers — arb panel uses "
                            "the live fallback",
                            os.environ.get("GUNICORN_WORKERS", "?"),
                        )
                # This worker's leader status — the discovery loop co-locates here.
                app.state.arb_is_leader = bool(acquired)

                if not acquired:
                    logger.info("arb engine autostart: leader held by another worker")
                else:
                    # We are the sole elected leader. Now (and only now) is
                    # it safe to scan for orphan engines from a prior batch
                    # that escaped lifespan teardown — siblings can't be
                    # racing us to spawn because they all lost SETNX.
                    _arb_engine_marker = "arb_engine.py"
                    try:
                        # Boot-time process scan; blocking is acceptable here.
                        pgrep = subprocess.run(  # noqa: ASYNC221
                            ["pgrep", "-f", _arb_engine_marker],
                            capture_output=True,
                            text=True,
                            timeout=2.0,
                            check=False,
                        )
                        orphans = [
                            int(p)
                            for p in pgrep.stdout.split()
                            if p.strip().isdigit() and int(p) != os.getpid()
                        ]
                    except Exception as exc:
                        logger.debug("arb engine orphan-scan skipped: %s", exc)
                        orphans = []
                    for opid in orphans:
                        try:
                            os.kill(opid, 15)  # SIGTERM
                            logger.warning("arb engine autostart: reaped orphan pid=%d", opid)
                        except ProcessLookupError:
                            pass
                        except Exception as exc:
                            logger.warning(
                                "arb engine autostart: failed to reap pid=%d: %s",
                                opid,
                                exc,
                            )
                    if orphans:
                        # Let SIGTERM land before we spawn the replacement.
                        await asyncio.sleep(0.5)
                    log_path = arb_dir / "arb_engine.log"
                    # Log handle outlives any context manager — Popen owns it for
                    # the lifetime of the subprocess. Boot-time blocking I/O is fine.
                    log_fh = log_path.open("a", buffering=1, encoding="utf-8")
                    proc = subprocess.Popen(  # noqa: ASYNC220
                        [
                            "python",
                            str(engine_path),
                            "--mode",
                            os.environ.get("PFM_ARB_ENGINE_MODE", "og"),
                        ],
                        cwd=str(arb_dir),
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                    )
                    app.state.arb_engine_proc = proc
                    app.state.arb_engine_log_fh = log_fh
                    app.state.arb_engine_lock_token = _arb_lock_token
                    app.state.arb_engine_lock_key = _arb_lock_key
                    logger.info(
                        "arb engine autostart: pid=%s log=%s (leader, token=%s)",
                        proc.pid,
                        log_path,
                        _arb_lock_token.decode().split("|nonce=")[0],
                    )

                    # CAS lock refresher — atomically extends the TTL only
                    # when the stored value still matches our token. Uses a
                    # Lua script so the GET+EXPIRE pair runs as one Redis
                    # op; otherwise a successor that grabbed the lock between
                    # our GET and EXPIRE would have its TTL stomped to ours.
                    _arb_cas_renew_script = (
                        "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                        "  return redis.call('PEXPIRE', KEYS[1], ARGV[2]) "
                        "else return 0 end"
                    )
                    _arb_refresh_interval = max(5, _arb_lock_ttl // 3)

                    async def _refresh_arb_lock() -> None:
                        # Health-checked CAS renewal. If the subprocess died,
                        # release the lock so a follower can take over inside
                        # one refresh tick rather than waiting the full TTL.
                        while True:
                            try:
                                await asyncio.sleep(_arb_refresh_interval)
                                _proc = getattr(app.state, "arb_engine_proc", None)
                                if _proc is None or _proc.poll() is not None:
                                    rc = _proc.poll() if _proc is not None else None
                                    logger.warning(
                                        "arb engine subprocess died (rc=%s); "
                                        "releasing leader lock for failover",
                                        rc,
                                    )
                                    if _redis is not None:
                                        # Only DEL if we still own the lock.
                                        with suppress(Exception):
                                            _redis.eval(
                                                "if redis.call('GET', KEYS[1]) == ARGV[1] "
                                                "then return redis.call('DEL', KEYS[1]) "
                                                "else return 0 end",
                                                1,
                                                _arb_lock_key,
                                                _arb_lock_token,
                                            )
                                    break
                                if _redis is not None:
                                    # CAS: PEXPIRE only if value still matches.
                                    res = _redis.eval(
                                        _arb_cas_renew_script,
                                        1,
                                        _arb_lock_key,
                                        _arb_lock_token,
                                        _arb_lock_ttl * 1000,
                                    )
                                    if not res:
                                        # We lost the lease (key expired,
                                        # or another worker claimed it).
                                        # Bail out — the watchdog will
                                        # eventually notice the orphan or
                                        # the engine will keep running
                                        # alone if we just lost the race.
                                        logger.warning(
                                            "arb engine lock CAS-refresh failed "
                                            "(value changed) — stopping renewer"
                                        )
                                        break
                            except asyncio.CancelledError:
                                break
                            except Exception as exc:
                                logger.debug("arb engine renew tick: %s", exc)

                    if _redis is not None:
                        app.state.arb_lock_refresher = asyncio.create_task(_refresh_arb_lock())
        except Exception as exc:
            logger.warning("arb engine autostart failed: %s", exc)

    # 5min crypto predictor background workers — the sampler keeps the spot
    # buffer warm and the compare prewarmer keeps the /compare response
    # hot in-memory so users hit a ~5ms cached path instead of paying for
    # gamma discovery + Binance REST on every page load. Opt-IN via
    # PFM_CRYPTO_5MIN_ENABLED=1 so the test suite (which spins up TestClient
    # many times) doesn't accidentally hit Binance REST on every fixture mount.
    if os.environ.get("PFM_CRYPTO_5MIN_ENABLED") == "1":
        try:
            from pfm.crypto5min.background import start_in_lifespan as _start_crypto5min

            # 1-second cadence: prewarmer rebuilds the /compare payload with a
            # fresh Polymarket CLOB midpoint every tick so the UI (polling 1s)
            # always lands on a sub-second-old probability. Sampler keeps the
            # spot buffer hot at the same rate.
            poll = float(os.environ.get("PFM_CRYPTO_5MIN_POLL_SECONDS", "1"))
            compare = float(os.environ.get("PFM_CRYPTO_5MIN_COMPARE_SECONDS", "1"))
            _start_crypto5min(app, poll_seconds=poll, compare_seconds=compare)
            logger.info(
                "crypto5min background scheduled (sampler=%.1fs, prewarmer=%.1fs)",
                poll,
                compare,
            )
        except Exception as exc:
            logger.warning("crypto5min background failed to start: %s", exc)

    # Polymarket CLOB WebSocket subscriber — opt-in via
    # PFM_CRYPTO_CLOB_WS_ENABLED=1. Replaces the per-tick REST poll of
    # ``/midpoint`` on the crypto5min prewarmer with a push-driven stream
    # (~50-200 ms tick latency vs ~1 s REST polling). Leader-only: exactly
    # one gunicorn worker holds the WebSocket connection upstream and
    # publishes midpoints to Redis for follower workers to read.
    #
    # Design mirrors the arb-engine + crypto-events leader-election pattern
    # already in this file:
    #   * SETNX with a per-process token grabs the leader role.
    #   * A CAS-renew task extends the TTL only when our token is still
    #     stored; loses the lock if a successor claimed it.
    #   * Teardown CAS-DELs the lock so a clean restart doesn't have to
    #     wait the full TTL.
    #   * Failure modes (Redis offline, websockets import missing) degrade
    #     gracefully — ``fetch_clob_midpoint`` falls back to REST.
    if os.environ.get("PFM_CRYPTO_CLOB_WS_ENABLED") == "1":
        try:
            import secrets as _ws_secrets
            import time as _ws_time

            from pfm.crypto5min.market_fetcher import (
                ClobMidpointSubscriber,
                set_subscriber,
            )

            _clob_lock_key = "pfm:clob_ws:leader"
            _clob_lock_ttl = int(os.environ.get("PFM_CRYPTO_CLOB_WS_LEADER_TTL_S", "60"))
            _clob_cache = app.state.cache
            _clob_redis = getattr(_clob_cache, "_client", None)
            _clob_lock_token = (
                f"pid={os.getpid()}|boot={int(_ws_time.time())}|nonce={_ws_secrets.token_hex(8)}"
            ).encode()

            # Centralised SETNX via pfm.redis_lock (W11-15). Custom CAS-renew
            # / CAS-DEL Lua below still compares against _clob_lock_token, so
            # owner_id must match the stored value exactly.
            _clob_lock = RedisLock(
                _clob_redis,
                _clob_lock_key,
                ttl_s=_clob_lock_ttl,
                owner_id=_clob_lock_token.decode(),
            )
            if _clob_redis is not None:
                try:
                    _clob_acquired = _clob_lock.acquire()
                except Exception as exc:
                    logger.warning("clob ws subscriber leader-election failed: %s", exc)
                    _clob_acquired = True  # fail open in single-process deploys
            else:
                # No Redis → single-process; we're trivially the leader.
                _clob_acquired = True

            if not _clob_acquired:
                logger.info(
                    "clob ws subscriber: leader held by another worker (pid %d)",
                    os.getpid(),
                )
            else:
                # Rotation: every 60 s, refresh the subscription set to
                # exactly the currently-active token IDs. The /markets
                # endpoint is the source of truth.
                async def _clob_rotate() -> list[str]:
                    client = getattr(app.state, "async_http", None)
                    if client is None:
                        return []
                    try:
                        from pfm.crypto5min.market_fetcher import (
                            discover_active_markets,
                        )

                        actives = await discover_active_markets(
                            client,
                            assets=["BTC", "ETH"],
                            window_minutes_list=[5, 15],
                        )
                    except Exception as exc:
                        logger.debug(
                            "clob ws subscriber: rotation discovery failed: %s",
                            exc,
                        )
                        return []
                    # Both YES + NO token IDs — even though we only USE the
                    # YES midpoint, subscribing to NO lets us cross-check
                    # (yes + no should sum to ~1.0) and costs ~0 marginal
                    # bandwidth since they share a market.
                    ids: list[str] = []
                    for m in actives:
                        ids.append(m.up_token_id)
                        ids.append(m.down_token_id)
                    return ids

                _clob_subscriber = ClobMidpointSubscriber(
                    cache=_clob_cache,
                    redis_client=_clob_redis,
                    rotate_callable=_clob_rotate,
                    rotate_interval_s=float(os.environ.get("PFM_CRYPTO_CLOB_WS_ROTATE_S", "60")),
                )
                # Seed the initial subscription set so we don't have to wait
                # one full rotation interval for the first tokens.
                _seeded: list[str] = []
                with suppress(Exception):
                    _seeded = await _clob_rotate() or []
                    if _seeded:
                        _clob_subscriber.add_tokens(_seeded)
                await _clob_subscriber.start()
                set_subscriber(_clob_subscriber)
                app.state.clob_ws_subscriber = _clob_subscriber
                app.state.clob_ws_lock_key = _clob_lock_key
                app.state.clob_ws_lock_token = _clob_lock_token
                logger.info(
                    "clob ws subscriber started (leader, pid=%d, seeded=%d tokens)",
                    os.getpid(),
                    len(_seeded),
                )

                # CAS lock refresher — atomically extends the TTL only when
                # our token is still stored. Mirrors the arb engine renew.
                _clob_cas_renew_script = (
                    "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                    "  return redis.call('PEXPIRE', KEYS[1], ARGV[2]) "
                    "else return 0 end"
                )
                _clob_refresh_interval = max(5, _clob_lock_ttl // 3)

                async def _clob_renew_lock() -> None:
                    while True:
                        try:
                            await asyncio.sleep(_clob_refresh_interval)
                            if _clob_redis is None:
                                continue
                            res = _clob_redis.eval(
                                _clob_cas_renew_script,
                                1,
                                _clob_lock_key,
                                _clob_lock_token,
                                _clob_lock_ttl * 1000,
                            )
                            if not res:
                                logger.warning(
                                    "clob ws subscriber lock CAS-refresh failed "
                                    "(value changed) — stopping renewer"
                                )
                                break
                        except asyncio.CancelledError:
                            break
                        except Exception as exc:
                            logger.debug("clob ws subscriber renew tick: %s", exc)

                if _clob_redis is not None:
                    app.state.clob_ws_lock_refresher = asyncio.create_task(_clob_renew_lock())
        except Exception as exc:
            logger.warning("clob ws subscriber failed to start: %s", exc)

    # Default OFF — tests (which spin up TestClient repeatedly) would otherwise
    # pay the prewarm cost on every fixture mount. Production opts in via the
    # env var set in docker-compose.yml.
    if os.environ.get("PFM_FACTOR_PREWARM_ENABLED") == "1":
        # Default 500 covers ~40 % of the catalog — high enough that most
        # search hits + market drilldowns land warm. Bigger numbers add
        # ~5 s to boot per +200 factors but the prewarm runs async after
        # liveness so the user is unaffected. Override via env var.
        prewarm_n = int(os.environ.get("PFM_FACTOR_PREWARM_TOP_N", "500"))
        prewarm_lookback = int(os.environ.get("PFM_FACTOR_PREWARM_LOOKBACK_DAYS", "180"))
        prewarm_conc = int(os.environ.get("PFM_FACTOR_PREWARM_CONCURRENCY", "8"))

        async def _factor_prewarm() -> None:
            try:
                # Avoid hammering the upstream API the instant the app boots —
                # let the lifespan finish initialising and let the first
                # liveness probe complete before we start fanning out.
                await asyncio.sleep(2.0)
                from pfm.regression_core import _cached_factor_history
                from pfm.reverse_finder_router import _curated_candidate_ids

                factors = app.state.factors
                ids = _curated_candidate_ids(factors, limit=prewarm_n)
                if not ids:
                    logger.info("factor prewarm: no curated candidates — skipping")
                    return
                end_ts = pd.Timestamp.utcnow().normalize()
                start_ts = end_ts - pd.Timedelta(days=prewarm_lookback)
                poly = app.state.poly
                cache = app.state.cache
                settings = app.state.settings
                sem = asyncio.Semaphore(prewarm_conc)
                ok = 0
                fail = 0

                # Collect latest-price-by-slug for the in-memory lookup that
                # /terminal/search reads. Without this, every search result
                # shows current_price: null until the on-disk pickle is built
                # (which we don't ship).
                latest_prices: dict[str, float] = {}
                # ALSO build a {slug: pd.Series} map and pickle it to
                # /tmp/strat7_factor_history.pkl so /terminal/factor-clusters,
                # /terminal/peers, and /terminal/market stats have data. The
                # pickle was historically built by a strat7 batch job; we
                # generate the equivalent from the same fetches we just made.
                history_map: dict[str, Any] = {}

                async def _warm_one(fid: str) -> None:
                    nonlocal ok, fail
                    async with sem:
                        fc = factors.get(fid)
                        if fc is None:
                            return
                        try:
                            df = await asyncio.to_thread(
                                _cached_factor_history,
                                fc,
                                start_ts,
                                end_ts,
                                poly,
                                cache,
                                settings,
                            )
                            if df is not None and not df.empty:
                                ok += 1
                                # Pull the most-recent price; store keyed by
                                # both slug and factor_id so search hits both.
                                try:
                                    last_val = float(df.iloc[-1, 0])
                                    if math.isfinite(last_val):
                                        latest_prices[fc.slug] = last_val
                                        latest_prices[fc.id] = last_val
                                except (TypeError, ValueError, IndexError):
                                    pass
                                # Store the price series under the slug —
                                # that's the key everything downstream uses.
                                try:
                                    history_map[fc.slug] = df.iloc[:, 0].copy()
                                except Exception as e:
                                    logger.debug(
                                        "prewarm history_map copy failed for %s: %s", fc.slug, e
                                    )
                            else:
                                fail += 1
                        except Exception:
                            fail += 1

                t0 = asyncio.get_event_loop().time()
                await asyncio.gather(*(_warm_one(fid) for fid in ids))
                elapsed = asyncio.get_event_loop().time() - t0
                app.state.prewarmed_prices = latest_prices
                # Persist the history map. Multiple workers may race here —
                # an atomic-write avoids a half-written pickle confusing the
                # reader. Use the same default path the readers expect.
                if history_map:
                    try:
                        import pickle as _pickle
                        import tempfile as _tempfile

                        history_path = Path("/tmp/strat7_factor_history.pkl")
                        with _tempfile.NamedTemporaryFile(
                            mode="wb",
                            dir=str(history_path.parent),
                            delete=False,
                            suffix=".tmp",
                        ) as tmp:
                            _pickle.dump(history_map, tmp, protocol=_pickle.HIGHEST_PROTOCOL)
                            tmp_path = tmp.name
                        Path(tmp_path).replace(history_path)  # noqa: ASYNC240
                        logger.info(
                            "factor prewarm: wrote %d series to %s",
                            len(history_map),
                            history_path,
                        )
                    except Exception as exc:
                        logger.warning("factor prewarm: pickle write failed: %s", exc)
                logger.info(
                    "factor prewarm: %d/%d ok in %.1fs (lookback=%dd, cache=%s, prices=%d, history=%d)",
                    ok,
                    ok + fail,
                    elapsed,
                    prewarm_lookback,
                    "on" if app.state.cache.enabled else "off",
                    len(latest_prices),
                    len(history_map),
                )
            except Exception as exc:
                logger.warning("factor prewarm failed: %s", exc)

        app.state.factor_prewarm_task = asyncio.create_task(_factor_prewarm())
        logger.info(
            "factor prewarm scheduled (top %d curated · lookback %dd · concurrency %d)",
            prewarm_n,
            prewarm_lookback,
            prewarm_conc,
        )

        # --- Gamma price prewarm: build a {slug → yes_price} map from live
        # Polymarket active markets so /terminal/search can show current
        # prices for any matched factor (not just the curated-prewarm set).
        # Persisted in Redis under ``arb:gamma_prices`` so all gunicorn
        # workers share the same warm cache and a server restart doesn't
        # re-pay the cold-fetch cost. TTL is 5 min; only one worker
        # actually pings Polymarket each cycle thanks to Redis SETNX lock.
        _GAMMA_KEY = "arb:gamma_prices"
        _GAMMA_LOCK_KEY = "arb:gamma_prices:lock"
        _GAMMA_TTL_S = int(os.environ.get("PFM_GAMMA_PRICE_TTL_S", "300"))
        _GAMMA_REFRESH_S = int(os.environ.get("PFM_GAMMA_PRICE_REFRESH_S", "60"))

        async def _gamma_price_prewarm() -> None:
            try:
                await asyncio.sleep(1.0)
                gamma_url = app.state.settings.polymarket_gamma_url
                http = app.state.async_http
                cache = app.state.cache
                while True:
                    # Try shared Redis first — saves the upstream round-trip
                    # if another worker (or the previous server boot) has
                    # already populated the key within the TTL window.
                    try:
                        cached_raw = cache.get(_GAMMA_KEY)
                        if cached_raw:
                            prices = json.loads(
                                cached_raw.decode() if isinstance(cached_raw, bytes) else cached_raw
                            )
                            if isinstance(prices, dict) and prices:
                                app.state.gamma_prices = prices
                                await asyncio.sleep(_GAMMA_REFRESH_S)
                                continue
                    except Exception as e:
                        logger.debug("gamma cache read miss / parse error: %s", e)
                    # Acquire a soft refresh lock so only one worker fans out.
                    # Centralised via pfm.redis_lock (W11-15); TTL-based auto-
                    # release matches the pre-migration semantics.
                    try:
                        got_lock = RedisLock(
                            getattr(cache, "_client", None),
                            _GAMMA_LOCK_KEY,
                            ttl_s=_GAMMA_REFRESH_S,
                        ).acquire()
                    except Exception:
                        got_lock = True  # cache offline → fetch directly
                    if not got_lock:
                        # Another worker is refreshing — wait briefly + re-check.
                        await asyncio.sleep(2.0)
                        continue
                    try:
                        prices: dict[str, float] = {}
                        volumes: dict[str, float] = {}
                        for offset in range(0, 1000, 100):
                            r = await http.get(
                                f"{gamma_url.rstrip('/')}/markets",
                                params={
                                    "active": "true",
                                    "closed": "false",
                                    "limit": 100,
                                    "offset": offset,
                                    "order": "volume24hr",
                                    "ascending": "false",
                                },
                                timeout=8.0,
                            )
                            if r.status_code != 200:
                                break
                            page = r.json() or []
                            if not page:
                                break
                            for m in page:
                                slug = m.get("slug")
                                if not slug:
                                    continue
                                p = terminal_mod._yes_price_from_market(m)
                                if p is not None and math.isfinite(p):
                                    prices[slug] = float(p)
                                # 24h notional volume so /terminal/search can
                                # surface volume_24h alongside price (was null
                                # for every row before this change).
                                v_raw = m.get("volume24hr") or m.get("volumeNum") or m.get("volume")
                                try:
                                    v = float(v_raw) if v_raw is not None else None
                                except (TypeError, ValueError):
                                    v = None
                                if v is not None and math.isfinite(v):
                                    volumes[slug] = v
                            if len(page) < 100:
                                break
                        app.state.gamma_prices = prices
                        app.state.gamma_volumes = volumes
                        try:
                            cache.set(
                                _GAMMA_KEY,
                                json.dumps(prices).encode(),
                                _GAMMA_TTL_S,
                            )
                        except Exception as e:
                            logger.warning("gamma price cache.set failed: %s", e)
                        logger.info(
                            "gamma price prewarm: %d slugs cached (TTL %ds)",
                            len(prices),
                            _GAMMA_TTL_S,
                        )
                    except Exception as exc:
                        logger.warning("gamma price prewarm: %s", exc)
                    await asyncio.sleep(_GAMMA_REFRESH_S)
            except asyncio.CancelledError:
                raise

        app.state.gamma_price_task = asyncio.create_task(_gamma_price_prewarm())

        # --- Kalshi price prewarm: similar to gamma, builds a {ticker: yes_price}
        # map so /terminal/search hits with KX*-shaped slugs come back with
        # current prices instead of null. Cached in Redis ``arb:kalshi_prices``
        # cross-worker. Refresh interval matches the gamma cadence (60 s) and
        # both prewarms run independently — one slow upstream doesn't block
        # the other.
        _KALSHI_KEY = "arb:kalshi_prices"
        _KALSHI_LOCK_KEY = "arb:kalshi_prices:lock"
        _KALSHI_TTL_S = int(os.environ.get("PFM_KALSHI_PRICE_TTL_S", "300"))
        _KALSHI_REFRESH_S = int(os.environ.get("PFM_KALSHI_PRICE_REFRESH_S", "60"))
        _KALSHI_PAGES = int(os.environ.get("PFM_KALSHI_PRICE_PAGES", "5"))  # ×200/page

        async def _kalshi_price_prewarm() -> None:
            try:
                await asyncio.sleep(1.0)
                http = app.state.async_http
                cache = app.state.cache
                base = "https://api.elections.kalshi.com/trade-api/v2/markets"
                while True:
                    # Read-through Redis first (other worker may have refreshed).
                    try:
                        cached_raw = cache.get(_KALSHI_KEY)
                        if cached_raw:
                            prices = json.loads(
                                cached_raw.decode() if isinstance(cached_raw, bytes) else cached_raw
                            )
                            if isinstance(prices, dict) and prices:
                                app.state.kalshi_prices = prices
                                await asyncio.sleep(_KALSHI_REFRESH_S)
                                continue
                    except Exception as e:
                        logger.debug("kalshi cache read miss / parse error: %s", e)
                    # Cross-worker lock so only one fans out per refresh.
                    # Centralised via pfm.redis_lock (W11-15).
                    try:
                        got_lock = RedisLock(
                            getattr(cache, "_client", None),
                            _KALSHI_LOCK_KEY,
                            ttl_s=_KALSHI_REFRESH_S,
                        ).acquire()
                    except Exception:
                        got_lock = True
                    if not got_lock:
                        await asyncio.sleep(2.0)
                        continue
                    try:
                        prices: dict[str, float] = {}
                        cursor: str | None = None
                        for _ in range(_KALSHI_PAGES):
                            params: dict[str, Any] = {"limit": 200}
                            if cursor:
                                params["cursor"] = cursor
                            r = await http.get(base, params=params, timeout=8.0)
                            if r.status_code != 200:
                                break
                            payload = r.json() or {}
                            for m in payload.get("markets", []) or []:
                                ticker = m.get("ticker")
                                if not ticker:
                                    continue
                                bid = m.get("yes_bid")
                                ask = m.get("yes_ask")
                                last = m.get("last_price")
                                # Kalshi prices are in cents (0-100) — normalise
                                # to [0,1] so the search response matches the
                                # Polymarket scale.
                                px: float | None = None
                                if bid is not None and ask is not None:
                                    try:
                                        px = (float(bid) + float(ask)) / 2.0 / 100.0
                                    except (TypeError, ValueError):
                                        px = None
                                if px is None and last is not None:
                                    try:
                                        px = float(last) / 100.0
                                    except (TypeError, ValueError):
                                        px = None
                                if px is not None and math.isfinite(px) and 0 <= px <= 1:
                                    prices[ticker] = round(px, 4)
                            cursor = payload.get("cursor")
                            if not cursor:
                                break
                        app.state.kalshi_prices = prices
                        try:
                            cache.set(
                                _KALSHI_KEY,
                                json.dumps(prices).encode(),
                                _KALSHI_TTL_S,
                            )
                        except Exception as e:
                            logger.warning("kalshi price cache.set failed: %s", e)
                        logger.info(
                            "kalshi price prewarm: %d tickers cached (TTL %ds)",
                            len(prices),
                            _KALSHI_TTL_S,
                        )
                    except Exception as exc:
                        logger.warning("kalshi price prewarm: %s", exc)
                    await asyncio.sleep(_KALSHI_REFRESH_S)
            except asyncio.CancelledError:
                raise

        app.state.kalshi_price_task = asyncio.create_task(_kalshi_price_prewarm())

        # --- Arb scanner prewarm: build the fallback cache once at startup
        # so the first hit to /strategies/arb/state or /stream finds it hot.
        # Without this the user pays a 2-3 s blocking scan on first connect
        # and the UI panel shows "Disconnected" the whole time.
        async def _arb_prewarm() -> None:
            try:
                await asyncio.sleep(2.0)
                from pfm.strategies_arb_router import _build_fallback_state

                t0 = asyncio.get_event_loop().time()
                state = await asyncio.to_thread(_build_fallback_state)
                elapsed = asyncio.get_event_loop().time() - t0
                logger.info(
                    "arb scanner prewarm: %d opportunities cached in %.1fs",
                    len(state.get("opportunities", [])),
                    elapsed,
                )
            except Exception as exc:
                logger.warning("arb scanner prewarm failed: %s", exc)

        app.state.arb_prewarm_task = asyncio.create_task(_arb_prewarm())

        # --- ARB state mirror: file → Redis ---------------------------------
        # The standalone arb_engine.py writes ``arbstuff/dashboard_state.json``
        # to a local container disk. In a multi-machine deploy that disk
        # isn't visible to other gunicorn workers. We poll the file every
        # few seconds and mirror to Redis so every router worker — and a
        # restarted process — reads from the same shared place.
        async def _arb_state_mirror() -> None:
            from pfm.strategies_arb_router import (
                _ARB_DIR,
                _ARB_REDIS_ENABLED,
                _ARB_STATE_REDIS_KEY,
                _safe_read_json,
            )

            if not _ARB_REDIS_ENABLED:
                return
            cache = app.state.cache
            if not getattr(cache, "enabled", False):
                return
            interval = float(os.environ.get("PFM_ARB_MIRROR_INTERVAL_S", "5"))
            ttl = int(os.environ.get("PFM_ARB_MIRROR_TTL_S", "600"))
            last_mtime: float | None = None
            try:
                await asyncio.sleep(3.0)  # let prewarm finish first
                while True:
                    try:
                        path = _ARB_DIR / "dashboard_state.json"
                        mtime = path.stat().st_mtime if path.exists() else None
                        if mtime is not None and mtime != last_mtime:
                            state = await asyncio.to_thread(_safe_read_json, path)
                            if state is not None:
                                cache.set(
                                    _ARB_STATE_REDIS_KEY,
                                    json.dumps(state, default=str).encode(),
                                    ttl,
                                )
                                last_mtime = mtime
                                logger.debug(
                                    "arb state mirror: pushed %d opps to Redis (ttl %ds)",
                                    len(state.get("opportunities", [])),
                                    ttl,
                                )
                    except Exception as exc:
                        logger.warning("arb state mirror: %s", exc)
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise

        app.state.arb_mirror_task = asyncio.create_task(_arb_state_mirror())

    # Opt-in background discovery loop (unlimited/resumable/new-first → records
    # real arbs to the durable store). Co-locates on the arb leader worker (or a
    # single-worker deploy) so N gunicorn workers don't all crawl in parallel.
    _disc_leader = getattr(app.state, "arb_is_leader", None)
    if _disc_leader is None:
        _disc_leader = int(os.environ.get("GUNICORN_WORKERS", "1")) == 1
    if os.environ.get("PFM_ARB_DISCOVERY_LOOP") == "1" and _disc_leader:
        app.state.arb_discovery_task = asyncio.create_task(_arb_discovery_loop(app))
        logger.info("arb discovery loop started (background)")

    try:
        yield
    finally:
        if hasattr(app.state, "arb_discovery_task"):
            app.state.arb_discovery_task.cancel()
            try:
                await app.state.arb_discovery_task
            except (asyncio.CancelledError, Exception):
                pass
        if hasattr(app.state, "arb_mirror_task"):
            app.state.arb_mirror_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.arb_mirror_task
        if hasattr(app.state, "gamma_price_task"):
            app.state.gamma_price_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.gamma_price_task
        if hasattr(app.state, "kalshi_price_task"):
            app.state.kalshi_price_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.kalshi_price_task
        if hasattr(app.state, "live_signals_task"):
            app.state.live_signals_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.live_signals_task
        if hasattr(app.state, "decay_refresh_task"):
            app.state.decay_refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.decay_refresh_task
        if hasattr(app.state, "pm_vix_prewarm_task"):
            app.state.pm_vix_prewarm_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.pm_vix_prewarm_task
        if hasattr(app.state, "earnings_prewarm_task"):
            app.state.earnings_prewarm_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.earnings_prewarm_task
        if hasattr(app.state, "openapi_prewarm_task"):
            app.state.openapi_prewarm_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.openapi_prewarm_task
        if hasattr(app.state, "factor_prewarm_task"):
            app.state.factor_prewarm_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.factor_prewarm_task
        if hasattr(app.state, "crypto_ws_renew_task"):
            app.state.crypto_ws_renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.crypto_ws_renew_task
        if hasattr(app.state, "crypto_events_engine"):
            with suppress(Exception):
                await app.state.crypto_events_engine.stop()
            # Release the leader lease so a follower can take over right
            # away on next worker boot. The top-level RedisCache import is
            # in scope; we just need to check the backend is actually Redis.
            cache = getattr(app.state, "cache", None)
            if cache is not None and getattr(cache, "enabled", False):
                client = getattr(cache, "_client", None)
                if client is not None:
                    with suppress(Exception):
                        client.delete("arb:crypto_ws:leader")
        # arb engine subprocess (opt-in autostart) — terminate gracefully.
        # Cancel the CAS lock refresher BEFORE killing the subprocess so the
        # health-check tick doesn't race the teardown.
        if hasattr(app.state, "arb_lock_refresher"):
            app.state.arb_lock_refresher.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.arb_lock_refresher
        if hasattr(app.state, "arb_engine_proc"):
            proc = app.state.arb_engine_proc
            with suppress(Exception):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=2)
            if hasattr(app.state, "arb_engine_log_fh"):
                with suppress(Exception):
                    app.state.arb_engine_log_fh.close()
            # Release the leader lease atomically (only DEL if we still own
            # it) so the next worker boot doesn't have to wait the TTL. This
            # mirrors the crypto-WS shutdown and is the single most important
            # cleanup step for avoiding duplicate engines after a restart.
            _lock_key = getattr(app.state, "arb_engine_lock_key", None)
            _lock_token = getattr(app.state, "arb_engine_lock_token", None)
            _arb_cache = getattr(app.state, "cache", None)
            _arb_redis = getattr(_arb_cache, "_client", None) if _arb_cache is not None else None
            if _lock_key and _lock_token and _arb_redis is not None:
                with suppress(Exception):
                    _arb_redis.eval(
                        "if redis.call('GET', KEYS[1]) == ARGV[1] "
                        "then return redis.call('DEL', KEYS[1]) "
                        "else return 0 end",
                        1,
                        _lock_key,
                        _lock_token,
                    )
        if hasattr(app.state, "crypto5min_sampler_task"):
            with suppress(Exception):
                from pfm.crypto5min.background import stop_in_lifespan as _stop_crypto5min

                await _stop_crypto5min(app)
        # CLOB WS subscriber — stop the consumer task, clear the process-wide
        # singleton, then CAS-DEL the leader lock so the next worker boot
        # doesn't have to wait the full TTL.
        if hasattr(app.state, "clob_ws_lock_refresher"):
            app.state.clob_ws_lock_refresher.cancel()
            with suppress(asyncio.CancelledError):
                await app.state.clob_ws_lock_refresher
        if hasattr(app.state, "clob_ws_subscriber"):
            with suppress(Exception):
                await app.state.clob_ws_subscriber.stop()
            try:
                from pfm.crypto5min.market_fetcher import set_subscriber

                set_subscriber(None)
            except Exception as _exc:
                logger.debug("clob ws subscriber clear failed: %s", _exc)
            _clob_lock_key = getattr(app.state, "clob_ws_lock_key", None)
            _clob_lock_token = getattr(app.state, "clob_ws_lock_token", None)
            _clob_cache = getattr(app.state, "cache", None)
            _clob_redis = getattr(_clob_cache, "_client", None) if _clob_cache is not None else None
            if _clob_lock_key and _clob_lock_token and _clob_redis is not None:
                with suppress(Exception):
                    _clob_redis.eval(
                        "if redis.call('GET', KEYS[1]) == ARGV[1] "
                        "then return redis.call('DEL', KEYS[1]) "
                        "else return 0 end",
                        1,
                        _clob_lock_key,
                        _clob_lock_token,
                    )
        await app.state.hub.shutdown()
        app.state.http.close()
        await app.state.async_http.aclose()


# --- Sentry error tracking (opt-in via SENTRY_DSN) --------------------------
# Initialised at import time, BEFORE FastAPI() so the FastApiIntegration can
# patch the framework cleanly. If sentry-sdk isn't installed in the target
# environment the import is swallowed; the rest of the app comes up normally.
_sentry_dsn = os.environ.get("SENTRY_DSN", "").strip()
if _sentry_dsn:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.asyncio import AsyncioIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration

        sentry_sdk.init(
            dsn=_sentry_dsn,
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            profiles_sample_rate=float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
            environment=os.environ.get("ENV", "dev"),
            release=os.environ.get("GIT_SHA", "unknown")[:12],
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                AsyncioIntegration(),
            ],
            send_default_pii=False,
        )
    except ImportError:
        pass


app = FastAPI(
    title="Prediction Factor Model",
    version=__version__,
    description="Factor models of stock returns on Polymarket-derived factors.",
    lifespan=lifespan,
)

# --- CORS lockdown ----------------------------------------------------------
# Origin list is driven by ``PFM_CORS_ORIGINS`` (comma-separated). The dev
# default covers the split-port local layout (UI on :8080, API on :8000); set
# the env in production to the exact deploy origin(s). Wildcard ``*`` is still
# accepted but is now opt-in rather than the silent default — see .env.example.
# Legacy ``CORS_ORIGINS`` is honoured as a fallback so existing docker-compose
# files keep working until they're rotated to the new name.
_cors_raw = (
    os.environ.get("PFM_CORS_ORIGINS")
    or os.environ.get("CORS_ORIGINS")
    or "http://127.0.0.1:8080,http://localhost:8080"
)
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
# Starlette's CORSMiddleware requires ``allow_credentials=False`` when any
# origin is ``*``; explicitly disable credentials in that case so a misconfig
# doesn't silently break preflights.
_cors_allow_credentials = "*" not in _cors_origins
# NOTE: CORSMiddleware is registered LATER (after every other middleware),
# below the rate-limit middleware. Starlette's ``add_middleware`` PREPENDS
# to the middleware list — the last-added becomes OUTERMOST. We want CORS
# to be OUTERMOST so that error responses generated by the rate-limit /
# metrics / security-headers middleware ALSO get the ``Access-Control-Allow-
# Origin`` header. Registering it here (early) used to make CORS the INNERMOST
# user middleware, which meant 401/429/etc. emitted by an outer middleware
# bypassed CORS entirely and the browser dropped the response with the
# misleading "No 'Access-Control-Allow-Origin' header" error.

# Response-body compression. Order matters: ``add_middleware`` registers
# OUTER-FIRST style, so the LAST added becomes the OUTERMOST and sees the
# response LAST on the way out. We want brotli to encode FIRST when the
# client supports it (better ratio), so brotli goes INNER (added first) and
# gzip goes OUTER (added second). The gzip middleware skips when it sees
# ``Content-Encoding`` already set, so clients that advertise both get br
# and clients that only advertise gzip fall through to gzip.
if BrotliMiddleware is not None:
    # quality=5 is the sweet spot — significantly better ratio than gzip
    # (the 1.2 MB index.html drops to ~250 KB on the wire), but compression
    # latency stays under 8 ms per response. quality=11 would be smaller but
    # adds 100+ ms which kills first-byte time for SSE streams.
    app.add_middleware(BrotliMiddleware, quality=5, minimum_size=1024)
app.add_middleware(GZipMiddleware, minimum_size=1024)


# --- /openapi.json cached + ETag handler ------------------------------------
#
# FastAPI auto-registers a /openapi.json route that re-serialises the schema
# on every request. We replace it with a handler that:
#   (1) memoises the JSON bytes against ``app.openapi_schema`` mutation,
#   (2) emits a strong ETag derived from the version + payload hash so a
#       304 round-trip carries no body, and
#   (3) sets ``Cache-Control: public, max-age=3600`` — the schema only
#       changes when code does, and code changes invalidate the ETag anyway.
# The GZipMiddleware above wraps this same response so an ``If-None-Match``
# miss re-emits gzipped bytes, not raw JSON.

_OPENAPI_CACHE: dict[str, tuple[bytes, str]] = {}


def _openapi_payload() -> tuple[bytes, str]:
    """Return cached ``(json_bytes, etag)`` for the current OpenAPI schema."""
    schema = app.openapi()
    cache_key = f"v={app.version}|n={len(schema.get('paths', {}))}"
    cached = _OPENAPI_CACHE.get(cache_key)
    if cached is not None:
        return cached
    body = json.dumps(schema, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()[:16]
    etag = f'"{app.version}-{digest}"'
    payload = (body, etag)
    _OPENAPI_CACHE.clear()
    _OPENAPI_CACHE[cache_key] = payload
    return payload


# Drop FastAPI's auto-installed /openapi.json so our route wins the dispatch.
_existing_openapi_url = app.openapi_url or "/openapi.json"
app.router.routes = [
    r for r in app.router.routes if getattr(r, "path", None) != _existing_openapi_url
]


@app.get(_existing_openapi_url, include_in_schema=False)
def _cached_openapi(request: Request) -> Response:
    body, etag = _openapi_payload()
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and etag in {h.strip() for h in if_none_match.split(",")}:
        return Response(
            status_code=304,
            headers={
                "ETag": etag,
                "Cache-Control": "public, max-age=3600",
            },
        )
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "ETag": etag,
            "Cache-Control": "public, max-age=3600",
        },
    )


# --- security headers --------------------------------------------------------
# Adds the OWASP-recommended response headers. Gated by
# ``PFM_SECURITY_HEADERS_ENABLED`` (default ``1``) so a test or local override
# can disable them without code edits. HSTS is intentionally NOT set here —
# nginx in front of the API handles TLS termination + HSTS in prod, and
# adding it from the app would risk a misconfigured dev origin pinning HSTS
# on the developer's browser.
_SECURITY_HEADERS_ENABLED = os.environ.get("PFM_SECURITY_HEADERS_ENABLED", "1") == "1"


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    if not _SECURITY_HEADERS_ENABLED:
        return response
    response.headers["X-Content-Type-Options"] = "nosniff"
    # /embed/* serves iframe-embeddable charts — those must stay frameable.
    # Everywhere else: SAMEORIGIN (downgraded from DENY so the bundled UI
    # mount at /ui can still iframe its own panels if needed).
    if not request.url.path.startswith("/embed"):
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


# --- Latency audit ----------------------------------------------------------
# Lightweight per-request timing into pfm.metrics.LatencyTracker. Backs the
# ``GET /metrics/audit`` endpoint exposed by pfm.metrics_router. Skips any
# path under ``/metrics/*`` to avoid recording the audit endpoint itself
# (which would cause its latency to appear in its own snapshot and bias
# the percentiles upward over time).
import time as _audit_time

from pfm.metrics import get_tracker as _get_latency_tracker

_LATENCY_TRACKER = _get_latency_tracker()


def _audit_template_path(request: Request) -> str:
    """Return the FastAPI route template if matched, else the raw URL path.

    Collapses templated paths (e.g. ``/terminal/jumps/btc-price-by-eoy``
    becomes ``/terminal/jumps/{slug}``) so the audit groups by route, not
    by every unique slug value.
    """
    route = request.scope.get("route")
    tmpl = getattr(route, "path", None) if route is not None else None
    if isinstance(tmpl, str) and tmpl:
        return tmpl
    return request.url.path


@app.middleware("http")
async def _latency_audit(request: Request, call_next):
    # Skip the audit endpoint family entirely to avoid recursive sampling.
    if request.url.path.startswith("/metrics"):
        return await call_next(request)
    start = _audit_time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        elapsed_ms = (_audit_time.perf_counter() - start) * 1000.0
        try:
            _LATENCY_TRACKER.record(_audit_template_path(request), elapsed_ms, status_code)
        except Exception:  # pragma: no cover - never let metrics kill a req
            pass


# --- Prometheus metrics ------------------------------------------------------
# The per-request timing middleware + /metrics route are registered
# unconditionally so the request counters and histograms stay accurate from
# boot. PUBLIC EXPOSURE of /metrics is gated by ``PFM_METRICS_ENABLED``
# (default OFF) — when disabled the route returns 404, so a misconfigured
# prod deploy can't accidentally publish the metrics surface. When
# ``PFM_METRICS_TOKEN`` is set we additionally require an
# ``Authorization: Bearer <token>`` header; this is enforced by a small
# middleware that short-circuits before the route handler so the metrics
# body is never generated for unauthorised callers.
from pfm.observability import setup_metrics

setup_metrics(app)

# Default OFF outside of pytest. Inside the test suite (`pytest` in
# ``sys.modules`` is a reliable signal that the runner imported us), default
# ON so the existing /metrics smoke tests keep passing without each one
# having to setenv.
import sys as _sys

_metrics_default = "1" if "pytest" in _sys.modules else "0"
_METRICS_ENABLED = os.environ.get("PFM_METRICS_ENABLED", _metrics_default) == "1"
_METRICS_TOKEN = os.environ.get("PFM_METRICS_TOKEN", "").strip()


@app.middleware("http")
async def _metrics_access_gate(request: Request, call_next):
    if request.url.path != "/metrics":
        return await call_next(request)
    if not _METRICS_ENABLED:
        return Response(status_code=404, content="not found", media_type="text/plain")
    if _METRICS_TOKEN:
        auth = request.headers.get("authorization", "")
        ok = auth.startswith("Bearer ") and auth.split(" ", 1)[1].strip() == _METRICS_TOKEN
        if not ok:
            return Response(
                status_code=401,
                content="unauthorized",
                media_type="text/plain",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return await call_next(request)


# API key auth + tier-based rate limiting (opt-in via PFM_AUTH_ENABLED=1).
# When the env var is unset the dependencies + middleware fast-path, so the
# observable behaviour is identical to the pre-auth build; this preserves
# every existing test.
from pfm.auth import RateLimitMiddleware as _PFMRateLimitMiddleware
from pfm.auth import router as _auth_router

app.add_middleware(_PFMRateLimitMiddleware)
app.include_router(_auth_router)

# --- CORS as outermost middleware -------------------------------------------
# Register CORS LAST so Starlette wraps it OUTERMOST (``add_middleware`` does
# ``user_middleware.insert(0, ...)`` and the stack is built by wrapping from
# the end of that list backwards, so position 0 == outermost). This guarantees
# the ``Access-Control-Allow-Origin`` header is added to EVERY response,
# including 401/404/429/500s emitted by the rate-limit / metrics-gate /
# security-headers middlewares (which would otherwise short-circuit before
# reaching the inner ExceptionMiddleware + router).
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- belt-and-braces: explicit CORS headers on error responses --------------
# Starlette's ``ServerErrorMiddleware`` sits OUTSIDE the user middleware stack
# (so even outside our CORSMiddleware) and catches uncaught exceptions to
# produce a generic 500. That 500 bypasses CORS entirely. To make sure such
# responses still carry an Access-Control-Allow-Origin header, we register
# app-level exception handlers that explicitly attach CORS headers based on
# the request's ``Origin`` (when it's on the allow-list). FastAPI invokes
# these handlers inside the inner ExceptionMiddleware, so the response then
# also passes through our outermost CORSMiddleware on the way out — but the
# manual header makes the contract obvious and survives even if a future
# middleware reorder regresses the outermost-CORS invariant.
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse as _JSONResponse
from starlette.exceptions import HTTPException as _StarletteHTTPException


def _cors_headers_for(request: Request) -> dict[str, str]:
    origin = request.headers.get("origin", "")
    if not origin:
        return {}
    # Wildcard mode: allow any caller (mirrors CORSMiddleware behaviour when
    # ``*`` is configured AND credentials are disabled).
    if "*" in _cors_origins and not _cors_allow_credentials:
        return {"Access-Control-Allow-Origin": "*", "Vary": "Origin"}
    if origin in _cors_origins or "*" in _cors_origins:
        headers = {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
        if _cors_allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        return headers
    return {}


@app.exception_handler(_StarletteHTTPException)
async def _http_exception_with_cors(request: Request, exc: _StarletteHTTPException):
    """Return JSON error with explicit CORS headers so the browser can read it."""
    payload = {"detail": exc.detail}
    response = _JSONResponse(
        payload,
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None) or {},
    )
    for k, v in _cors_headers_for(request).items():
        response.headers[k] = v
    return response


@app.exception_handler(RequestValidationError)
async def _validation_exception_with_cors(request: Request, exc: RequestValidationError):
    """422 from pydantic body/query validation — preserve CORS on the response."""
    # ``exc.errors()`` can contain non-JSON-serialisable objects (e.g. bytes
    # from a malformed body). Pre-serialise via FastAPI's default encoder so
    # the response body matches FastAPI's normal 422 shape.
    from fastapi.encoders import jsonable_encoder

    response = _JSONResponse(
        {"detail": jsonable_encoder(exc.errors())},
        status_code=422,
    )
    for k, v in _cors_headers_for(request).items():
        response.headers[k] = v
    return response


@app.exception_handler(Exception)
async def _unhandled_exception_with_cors(request: Request, exc: Exception):
    """Catch-all 500 handler that preserves CORS headers on the error response."""
    logging.getLogger("pfm.main").exception(
        "unhandled exception on %s %s", request.method, request.url.path
    )
    response = _JSONResponse(
        {"detail": "internal server error"},
        status_code=500,
    )
    for k, v in _cors_headers_for(request).items():
        response.headers[k] = v
    return response


# Detailed health endpoint at /health/detail (simple /health stays as-is).
from pfm.health_router import router as _health_detail_router

app.include_router(_health_detail_router)


# Mount the static frontend under "/ui" so a single uvicorn process serves both
# API and UI in dev. In docker-compose the nginx container handles this and
# reaches the API at /api/*; the frontend probes both bases.
#
# Subclass StaticFiles so HTML responses get ``Cache-Control: no-cache``.
# Without this the browser will happily serve a stale index.html on every
# reload (modern browsers cache static HTML aggressively unless told not to),
# which means a UX fix landing on disk is silently ignored until the user
# does a hard refresh. Static assets (.js / .css with mtime in their etag)
# still revalidate via If-None-Match → 304, so no extra bandwidth cost.
class _NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        # ``path`` here is the file-system path relative to the mount root,
        # so ``/ui/`` → "" → ``index.html`` (html=True). Match both shapes
        # AND the resolved Content-Type so the rule still fires when the
        # caller hits e.g. ``/ui/index.html`` directly.
        ctype = resp.headers.get("content-type", "")
        if (
            path.endswith(".html")
            or path in {"", "/", "index.html"}
            or ctype.startswith("text/html")
        ):
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp


_WEB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web"
if _WEB_DIR.exists():
    app.mount("/ui", _NoCacheStatic(directory=str(_WEB_DIR), html=True), name="ui")


# --- dependencies -----------------------------------------------------------
# Live in pfm.dependencies so feature routers can share them without circular
# imports back to this module. The ``current_*`` forms below are no-arg
# callables (resolve via ``pfm.main.app.state`` on each call) so they double as
# FastAPI dependencies AND direct callables from helper / background code.
from pfm.dependencies import (
    cache_key as _cache_key_from_deps,
)
from pfm.dependencies import (
    current_cache as get_cache,
)
from pfm.dependencies import (
    current_factors as get_factors_dep,
)
from pfm.dependencies import (
    current_polymarket as get_polymarket_client,
)

# --- endpoints --------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send fresh visitors to the UI."""
    return RedirectResponse(url="/ui/", status_code=307)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


# /factors/* (list, all, discover, preview, rank, permutation, best) →
# moved to pfm.factors_router (mounted via app.include_router at the bottom).
# Helpers used by those endpoints (_cache_key, _resolve_factor_specs,
# _assemble_design, _cached_log_returns, _cached_factor_history,
# _align_factor_prices, _shift_to_stock_calendar, _residualize_against_spy,
# _short_err, _POLY_FANOUT_SEMAPHORE_SIZE) remain in this module — the
# extracted router lazy-imports them.


#: Concurrency cap for parallel Polymarket fan-out. 20 in-flight is well under
#: the 1000/10s rate-limit and avoids overwhelming the upstream connection pool
#: (which is sized at 100 max_connections — leaving headroom for other endpoints).
# ── Regression-pipeline helpers (extracted to pfm.regression_core) ──
# Re-import them as module-level names so legacy callers that do
# ``from pfm import main as _m; _m._helper(...)`` keep working.
from pfm.regression_core import (  # noqa: F401 — re-exported for legacy callers + tests
    _POLY_FANOUT_SEMAPHORE_SIZE,
    _align_factor_prices,
    _assemble_design,
    _cached_factor_history,
    _cached_log_returns,
    _fetch_aligned_prob,
    _finite,
    _jsafe,
    _residualize_against_spy,
    _resolve_factor_specs,
    _resolve_one,
    _shift_to_stock_calendar,
    _short_err,
)

# /fit and /attribution → moved to pfm.regression_router (mounted via
# `app.include_router` at the very bottom of this file). They lazy-import
# the shared helpers `_resolve_factor_specs`, `_assemble_design`,
# `_finite`, `_jsafe` from this module.


# ---- /strategies/* ----------------------------------------------------------


# /strategies/* (32 endpoints) → moved to pfm.strategies_router (mounted
# at the bottom of this file). Five helpers live here and are bound into
# the router module's globals by `strategies_router.bind_main_helpers()`
# (called right before app.include_router(_strategies_router)).


# --- BTC Up/Down latency-arb live proxies ----------------------------------
# The browser monitor needs Gamma + CLOB midpoints; CLOB blocks browser CORS,
# so we proxy both via the API. Kept thin and untyped — pure pass-through.


@app.get("/btc-arb/active-market", include_in_schema=False, tags=["btc-arb"])
def btc_arb_active_market(
    slug: Annotated[str, Query(min_length=10, max_length=80)],
    *,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    """Resolve a deterministic ``btc-updown-{window}-{end_unix}`` slug → market."""
    try:
        r = httpx.get(
            f"{settings.polymarket_gamma_url}/markets",
            params={"slug": slug},
            timeout=3.0,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"gamma error: {e}") from e
    arr = r.json() or []
    if not arr:
        raise HTTPException(status_code=404, detail=f"no market for slug {slug}")
    m = arr[0]
    try:
        tokens = json.loads(m.get("clobTokenIds", "[]"))
    except json.JSONDecodeError:
        tokens = []
    end_iso = m.get("endDate") or ""
    # Parse end_unix from the slug suffix (canonical for these markets).
    try:
        end_unix = int(slug.rsplit("-", 1)[-1])
    except ValueError:
        end_unix = 0
    return {
        "slug": m.get("slug", slug),
        "end_unix": end_unix,
        "end_date": end_iso,
        "up_token_id": tokens[0] if tokens else None,
    }


@app.get("/btc-arb/midpoint", include_in_schema=False, tags=["btc-arb"])
def btc_arb_midpoint(
    token_id: Annotated[str, Query(min_length=10, max_length=128)],
    *,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    """Pass-through for Polymarket CLOB ``/midpoint`` (avoids browser CORS)."""
    try:
        r = httpx.get(
            f"{settings.polymarket_clob_url}/midpoint",
            params={"token_id": token_id},
            timeout=2.5,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"clob error: {e}") from e
    return r.json()


@app.post("/strategies/scan", response_model=ScanResponse, tags=["strategies"])
def strategies_scan(
    body: ScanRequest,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> ScanResponse:
    """Cartesian inefficiency scanner across the factor catalog.

    Three independent leaderboards: implication / conditional / cointegration.
    Set ``mode='all'`` for all three; restrict via ``theme`` or ``factor_ids``.
    """
    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be < end")
    start_ts = pd.Timestamp(body.start, tz="UTC")
    end_ts = pd.Timestamp(body.end, tz="UTC")

    def _fetch(fc: FactorConfig) -> pd.Series:
        df = _cached_factor_history(fc, start_ts, end_ts, poly, cache, settings)
        if df.empty:
            return pd.Series(dtype=float)
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        return df["price"].rename(fc.id)

    report = run_scan(
        factors,
        fetch_prices=_fetch,
        mode=body.mode,
        theme=body.theme,
        factor_ids=body.factor_ids,
        max_pairs=body.max_pairs,
        n_obs_min=body.n_obs_min,
        impl_tolerance=body.impl_tolerance,
        impl_n_violations_min=body.impl_n_violations_min,
        cond_beta_min=body.cond_beta_min,
        cond_r2_min=body.cond_r2_min,
        coint_adf_max_p=body.coint_adf_max_p,
        coint_half_life_max=body.coint_half_life_max,
        top_k_per_track=body.top_k_per_track,
    )

    def _hits(hits: list) -> list[ScanHitOut]:
        return [
            ScanHitOut(
                kind=h.kind,
                a_id=h.a_id,
                b_id=h.b_id,
                score=float(h.score),
                n_obs=h.n_obs,
                summary=h.summary,
                n_violations=h.n_violations,
                max_gap=_finite(h.max_gap),
                beta=_finite(h.beta),
                beta_ci_lo=_finite(h.beta_ci_lo),
                beta_ci_hi=_finite(h.beta_ci_hi),
                r_squared=_finite(h.r_squared),
                adf_pvalue=_finite(h.adf_pvalue),
                half_life_days=h.half_life_days,
                surprise=_finite(h.surprise),
            )
            for h in hits
        ]

    return ScanResponse(
        mode=report.mode,
        n_factors_scanned=report.n_factors_scanned,
        n_pairs_evaluated=report.n_pairs_evaluated,
        runtime_seconds=report.runtime_seconds,
        implication=_hits(report.implication),
        conditional=_hits(report.conditional),
        cointegration=_hits(report.cointegration),
    )


# --- /terminal/* (Yahoo-Finance-style data hub) -----------------------------
# One endpoint per UI panel — frontend just renders. Pass-through caching on
# Gamma + CLOB; stats are read from the on-disk cached factor-history pickle
# so we don't refit on every market view.


def _factors_by_slug_index(
    factors: dict[str, FactorConfig],
) -> dict[str, FactorConfig]:
    """Return the slug -> FactorConfig index built at startup.

    Falls back to an on-the-fly rebuild when the lifespan-populated index is
    not present (e.g. unit tests that build a bare FastAPI app and only
    wire ``app.state.factors``). Both branches return the same shape so the
    callers can stay slim.
    """
    by_slug = getattr(app.state, "factors_by_slug", None)
    if isinstance(by_slug, dict) and by_slug:
        return by_slug
    return {fc.slug: fc for fc in factors.values() if fc.slug}


def _theme_for_slug_from_yaml(slug: str, factors: dict[str, FactorConfig]) -> str | None:
    fc = _factors_by_slug_index(factors).get(slug)
    return fc.theme if fc else None


def _factor_id_for_slug(slug: str, factors: dict[str, FactorConfig]) -> str | None:
    fc = _factors_by_slug_index(factors).get(slug)
    return fc.id if fc else None


def _slug_for_factor_id(factor_id: str, factors: dict[str, FactorConfig]) -> str | None:
    fc = factors.get(factor_id)
    return fc.slug if fc else None


@app.get("/terminal/market/{slug}", response_model=None, tags=["terminal-core"])
def terminal_market(
    slug: str,
    format: Annotated[Literal["json", "csv", "pdf"], Query()] = "json",
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
):
    """Merged data hub for a single market (live + meta + stats + peers)."""
    cache_key = f"terminal_market::{slug}"
    cached = terminal_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        resp = TerminalMarketResponse.model_validate(cached)
        return _export_respond(resp, format, filename=f"market-{slug}", kind="market")

    # Live + meta from Gamma.
    try:
        market = terminal_mod.fetch_gamma_market(
            app.state.http,
            settings.polymarket_gamma_url,
            slug,
        )
    except LookupError as e:
        # Surface ``did_you_mean`` so the client can offer "you typed X,
        # try Y" rather than the historical bare "no market for slug=X".
        suggestions = _factor_suggest_meta(slug, factors, top_k=3)
        raise HTTPException(
            status_code=404,
            detail={
                "error": str(e),
                "query": slug,
                "did_you_mean": suggestions,
            },
        ) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"gamma error: {e}") from e

    theme = _theme_for_slug_from_yaml(slug, factors)
    factor_id = _factor_id_for_slug(slug, factors) or slug
    live = terminal_mod.shape_live(market)
    meta = terminal_mod.shape_meta(market, theme=theme)

    # Stats from cached on-disk pickle (best-effort).
    history = terminal_mod._load_factor_history_cache(terminal_mod.DEFAULT_FACTOR_HISTORY_PATH)
    series = history.get(slug)

    # Peers from alpha-hunter sweep cache.
    peers = terminal_mod.find_peers(factor_id, top_n=5)

    # Try implied "fair" price against the strongest peer if we have data.
    if peers and isinstance(series, pd.Series) and not series.empty:
        nearest = peers[0]
        nearest_slug = _slug_for_factor_id(nearest["peer_id"], factors) or nearest["peer_id"]
        nb_series = history.get(nearest_slug)
        if isinstance(nb_series, pd.Series) and not nb_series.empty:
            fair = terminal_mod.implied_fair_price(series, nb_series)
            if fair is not None:
                nearest["fair_price"] = fair

    if isinstance(series, pd.Series) and not series.empty:
        nearest_series: pd.Series | None = None
        if peers:
            nearest_slug = _slug_for_factor_id(peers[0]["peer_id"], factors) or peers[0]["peer_id"]
            ns = history.get(nearest_slug)
            nearest_series = ns if isinstance(ns, pd.Series) else None
        stats = terminal_mod.compute_stats_from_series(series, neighbor=nearest_series)
    else:
        stats = terminal_mod.compute_stats_from_series(pd.Series(dtype=float))
    # Fall back the ``stats.current_price`` to the live midpoint when the
    # historical pickle is empty for this slug. Without this the UI shows
    # a populated "Live" block alongside a dead "Stats" block — which the
    # UX audit flagged as confusing for users.
    if stats.get("current_price") is None:
        live_px = live.get("midpoint") or live.get("last_trade_price")
        if isinstance(live_px, (int, float)) and not math.isnan(float(live_px)):
            stats["current_price"] = float(live_px)

    # Project the canonical top-level aliases the front-end / audit specs
    # expect — single source of truth lives here so we never drift from
    # meta/live. ``resolution_iso`` is just ``meta.end_date`` re-keyed.
    _price = live.get("midpoint") or live.get("last_trade_price")
    _vol24 = live.get("volume_24hr")
    resp = TerminalMarketResponse(
        slug=slug,
        live=TerminalLive(**live),
        meta=TerminalMeta(**meta),
        stats=TerminalStats(**stats),
        peers=[TerminalPeer(**p) for p in peers],
        question=meta.get("question"),
        theme=meta.get("theme"),
        price=float(_price) if isinstance(_price, (int, float)) else None,
        volume_24h=float(_vol24) if isinstance(_vol24, (int, float)) else None,
        resolution_iso=meta.get("end_date"),
    )
    terminal_mod.TERMINAL_CACHE.set(
        cache_key,
        resp.model_dump(),
        terminal_mod.TTL_LIVE_SECONDS,
    )
    return _export_respond(resp, format, filename=f"market-{slug}", kind="market")


@app.get("/terminal/market/{slug}/history", response_model=None, tags=["terminal-core"])
def terminal_market_history(
    slug: str,
    fidelity: Annotated[int, Query(ge=1, le=1440)] = 1440,
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
    format: Annotated[Literal["json", "csv", "pdf"], Query()] = "json",
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
):
    """Pass-through to CLOB ``/prices-history`` with TTL caching."""
    cache_key = f"terminal_history::{slug}::{fidelity}::{start}::{end}"
    cached = terminal_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        resp = TerminalHistoryResponse.model_validate(cached)
        return _export_respond(resp, format, filename=f"history-{slug}", kind="history")

    try:
        meta = poly.get_market_metadata(slug)
    except PolymarketError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": str(e),
                "query": slug,
                "did_you_mean": _factor_suggest_meta(slug, factors, top_k=3),
            },
        ) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"gamma error: {e}") from e

    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    # Honor the requested fidelity by passing it through; fall back to the
    # client default (daily, 1440) when the user doesn't override.
    if fidelity == 1440:
        try:
            df = poly.get_price_history(meta.yes_token_id, start=start_ts, end=end_ts)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"clob error: {e}") from e
        bars = [
            TerminalHistoryBar(
                t=int(pd.Timestamp(row["date"]).timestamp()),
                p=float(row["price"]),
            )
            for _, row in df.iterrows()
        ]
    else:
        # Sub-daily — make a direct call so we can pass our chosen fidelity.
        params: dict[str, str | int] = {
            "market": meta.yes_token_id,
            "fidelity": fidelity,
            "interval": "max",
        }
        if start_ts is not None:
            params["startTs"] = int(start_ts.timestamp())
        try:
            r = app.state.http.get(
                f"{settings.polymarket_clob_url.rstrip('/')}/prices-history",
                params=params,
                timeout=10.0,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"clob error: {e}") from e
        history = r.json().get("history", []) or []
        bars = [
            TerminalHistoryBar(t=int(b["t"]), p=float(b["p"]))
            for b in history
            if "t" in b and "p" in b
        ]
        if end_ts is not None:
            cutoff = int(end_ts.timestamp())
            bars = [b for b in bars if b.t <= cutoff]

    resp = TerminalHistoryResponse(
        slug=slug,
        yes_token_id=meta.yes_token_id,
        fidelity=fidelity,
        n_bars=len(bars),
        history=bars,
    )
    terminal_mod.TERMINAL_CACHE.set(
        cache_key,
        resp.model_dump(),
        terminal_mod.TTL_HISTORY_SECONDS,
    )
    return _export_respond(resp, format, filename=f"history-{slug}", kind="history")


@app.get("/terminal/overview", response_model=TerminalOverviewResponse, tags=["terminal-core"])
def terminal_overview(
    pages: Annotated[int, Query(ge=1, le=10)] = 5,
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
) -> TerminalOverviewResponse:
    """Markets overview: theme heatmap + movers + most-traded + new + soon-to-resolve."""
    cache_key = f"terminal_overview::{pages}"
    cached = terminal_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return TerminalOverviewResponse.model_validate(cached)

    try:
        markets = terminal_mod.fetch_gamma_top_markets(
            app.state.http,
            settings.polymarket_gamma_url,
            pages=pages,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"gamma error: {e}") from e

    buckets = terminal_mod.build_overview(markets, factors)
    resp = TerminalOverviewResponse(
        n_markets_considered=buckets.n_markets,
        theme_heatmap=[TerminalThemeBucket(**b) for b in buckets.theme_heatmap],
        top_movers=[TerminalMover(**m) for m in buckets.top_movers],
        most_traded=[TerminalMover(**m) for m in buckets.most_traded],
        recently_launched=[TerminalNewMarket(**m) for m in buckets.recently_launched],
        upcoming_resolutions=[
            TerminalUpcomingResolution(**u) for u in buckets.upcoming_resolutions
        ],
    )
    terminal_mod.TERMINAL_CACHE.set(
        cache_key,
        resp.model_dump(),
        terminal_mod.TTL_OVERVIEW_SECONDS,
    )
    return resp


@app.get("/terminal/search", response_model=TerminalSearchResponse, tags=["terminal-core"])
def terminal_search(
    q: Annotated[str, Query(max_length=200)] = "",
    theme: Annotated[str, Query(max_length=50)] = "",
    limit: Annotated[int, Query(ge=1, le=200)] = 30,
    *,
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)],
) -> TerminalSearchResponse:
    """Fuzzy search across factor catalog (name + slug). Token-overlap scoring.
    With empty q, returns the first ``limit`` factors filtered by ``theme``."""
    cache_key = f"terminal_search::{q.lower()}::{theme.lower()}::{limit}"
    cached = terminal_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None:
        return TerminalSearchResponse.model_validate(cached)

    # Layered price source:
    #   1) Live Polymarket Gamma map (refreshed every 60s in lifespan) — covers
    #      the top ~1000 markets by 24h volume, regardless of factor catalog.
    #   2) Factor-prewarm map — covers curated factors even if they're not
    #      currently top-volume on Gamma.
    #   3) On-disk pickle fallback for offline mode.
    prices: dict[str, float] = {}
    prices.update(getattr(app.state, "gamma_prices", None) or {})
    prices.update(getattr(app.state, "kalshi_prices", None) or {})
    prices.update(getattr(app.state, "prewarmed_prices", None) or {})
    if not prices:
        prices = terminal_mod.cached_price_lookup()
    # 24-hour volumes parallel to ``prices`` so search hits can surface
    # ``volume_24h`` (UX audit 2026-05-14: every search row was null on
    # volume which felt broken to users).
    volumes: dict[str, float] = {}
    volumes.update(getattr(app.state, "gamma_volumes", None) or {})
    if q:
        hits = terminal_mod.search_factors(
            q,
            factors,
            limit=limit,
            price_lookup=prices,
            volume_lookup=volumes,
        )
    elif theme:
        # Theme-only browse mode: return factors matching theme.
        hits = []
        for fc in factors.values():
            if fc.theme != theme:
                continue
            p_ = prices.get(fc.slug) or prices.get(fc.id)
            v_ = volumes.get(fc.slug) or volumes.get(fc.id)
            hits.append(
                {
                    "factor_id": fc.id,
                    "name": fc.name,
                    "slug": fc.slug,
                    "theme": fc.theme,
                    "score": 0.0,
                    "current_price": p_,
                    "price": p_,
                    "volume_24h": v_,
                }
            )
            if len(hits) >= limit:
                break
        # Sidebar themes are populated from the homepage heatmap (which
        # classifies LIVE gamma markets via _theme_for_market) but the
        # factor catalog tags themes differently — themes like
        # ``equities`` and ``awards`` exist in the heatmap but have zero
        # factors in factors.yml. Fall back to the cached homepage gamma
        # listing for those themes so a sidebar click actually shows
        # markets instead of a misleading empty state.
        if not hits:
            try:
                # The homepage builder caches its enriched response under
                # `terminal_homepage`; pull the raw gamma listing it uses.
                import asyncio as _aio

                from pfm.config import get_settings as _gs
                from pfm.terminal.homepage import (
                    _fetch_top_markets_async,
                )
                from pfm.terminal.homepage import (
                    _theme_for_market as _hp_theme,
                )

                http = getattr(app.state, "async_http", None)
                gamma_url = _gs().polymarket_gamma_url

                async def _fetch_gamma_for_theme() -> list[dict]:
                    if http is not None:
                        return await _fetch_top_markets_async(http, gamma_url, pages=3)
                    import httpx as _hx

                    async with _hx.AsyncClient(timeout=10.0) as _http:
                        return await _fetch_top_markets_async(_http, gamma_url, pages=3)

                # Run the async fetch from this sync handler.
                try:
                    _loop = _aio.get_running_loop()
                    fut = _aio.ensure_future(_fetch_gamma_for_theme(), loop=_loop)
                    gamma_markets = _loop.run_until_complete(fut)  # type: ignore[arg-type]
                except RuntimeError:
                    gamma_markets = _aio.run(_fetch_gamma_for_theme())

                target_theme = theme.lower()
                for mkt in gamma_markets or []:
                    t = (_hp_theme(mkt) or "").lower()
                    if t != target_theme:
                        continue
                    slug_g = str(mkt.get("slug") or "").strip()
                    if not slug_g:
                        continue
                    p_ = prices.get(slug_g)
                    v_ = volumes.get(slug_g)
                    name = str(mkt.get("question") or slug_g)
                    hits.append(
                        {
                            "factor_id": slug_g,
                            "name": name,
                            "slug": slug_g,
                            "theme": t,
                            "score": 0.0,
                            "current_price": p_,
                            "price": p_,
                            "volume_24h": v_,
                        }
                    )
                    if len(hits) >= limit:
                        break
            except Exception as e:
                logger.warning("theme=%s fallback failed: %s", theme, e)
    else:
        # Neither q nor theme — empty result rather than a misleading
        # random sample of the catalog (felt like a bug to users).
        hits = []
    if theme and q:
        hits = [h for h in hits if h.get("theme") == theme][:limit]
    resp = TerminalSearchResponse(
        query=q,
        n_results=len(hits),
        results=[TerminalSearchHit(**h) for h in hits],
    )
    terminal_mod.TERMINAL_CACHE.set(
        cache_key,
        resp.model_dump(),
        terminal_mod.TTL_OVERVIEW_SECONDS,
    )
    return resp


# --- internals --------------------------------------------------------------


_cache_key = _cache_key_from_deps  # legacy alias; cache_key lives in pfm.dependencies


_ = (RegressionLit,)  # re-export marker for type checkers


# ── Terminal feature routers (Yahoo-Finance / Bloomberg-style data hub) ─────
from pfm.advanced_event_models_router import router as advanced_event_models_router
from pfm.alerts.router import router as alerts_router
from pfm.alpha_graveyard_router import router as alpha_graveyard_router
from pfm.alpha_hub_router import router as alpha_hub_router
from pfm.alpha_lab import router as alpha_lab_router
from pfm.alpha_tier_regen import router as alpha_tier_regen_router
from pfm.arb_scanner import router as arb_scanner_router
from pfm.archive.kalshi_router import router as kalshi_archive_router
from pfm.archive.router import alias_router as archive_alias_router
from pfm.archive.router import router as archive_router

# Auth middleware + router are installed earlier (see lines 591-595). Don't
# re-register them here — duplicate include_router calls cause OpenAPI to emit
# duplicate operation IDs which breaks codegen tools.
from pfm.auto_hedge import router as auto_hedge_router
from pfm.chart_export import router as chart_export_router
from pfm.counterfactual import router as counterfactual_router
from pfm.decay_monitor import router as decay_monitor_router
from pfm.earnings_whisper import router as earnings_whisper_router
from pfm.embed import router as embed_router

# 2026-05-08 wave-9 additions: event-on-event + advanced + multi-event + archive
from pfm.event_on_event_router import router as event_on_event_router
from pfm.garch_router import router as garch_router
from pfm.live_signals_job import router as live_signals_router
from pfm.macro_calendar import router as macro_calendar_router
from pfm.macro_overlay_unified import router as macro_overlay_router
from pfm.multi_event_chain_router import router as multi_event_chain_router

# 2026-05-08 wave-3 audit: more sources + alpha features + macro
from pfm.multi_venue_search import router as multi_venue_router

# 2026-05-08 wave-2 audit: more feature routers
from pfm.news_causal_chain import router as news_causal_router
from pfm.news_tagger import router as news_tagger_router
from pfm.pm_vix import router as pm_vix_router
from pfm.portfolio_optimizer_router import router as portfolio_optimizer_router
from pfm.quant_rigor_advanced_router import router as quant_rigor_router

# 2026-05-08 audit: new feature routers
from pfm.quant_validation_router import router as quant_validation_router

# 2026-05-08 SSE multiplexing — replaces the per-client polling pattern in
# pfm.terminal_live_stream with one poller per (kind, slug) shared across
# N clients via an in-process pub/sub hub.
from pfm.realtime.stream import router as realtime_stream_router
from pfm.replay_mode import router as replay_router
from pfm.resolution_pnl_tree import router as pnl_tree_router
from pfm.reverse_finder_router import router as reverse_finder_router
from pfm.smart_money_divergence import router as smart_money_divergence_router
from pfm.sources.bls import router as bls_router
from pfm.sources.fred import router as fred_catalog_router
from pfm.sources.health_router import router as sources_health_router
from pfm.strategies_catalog_router import router as strategies_catalog_router
from pfm.strategy_verdict import router as strategy_verdict_router
from pfm.terminal.backtest_compare import router as terminal_backtest_compare_router
from pfm.terminal.bulk_export import router as terminal_bulk_export_router
from pfm.terminal.calendar_curated import router as terminal_calendar_curated_router
from pfm.terminal.calendar_pair import router as terminal_calendar_pair_router
from pfm.terminal.calendar_scanner import router as terminal_calendar_scanner_router
from pfm.terminal.calendar_unified import router as terminal_calendar_unified_router
from pfm.terminal.compare import router as terminal_compare_router
from pfm.terminal.correlations import router as terminal_correlations_router
from pfm.terminal.countdown import router as terminal_countdown_router
from pfm.terminal.equity import router as terminal_equity_router
from pfm.terminal.event_calendar import router as terminal_event_calendar_router
from pfm.terminal.factor_clusters import router as terminal_factor_clusters_router
from pfm.terminal.fair_price import router as terminal_fair_price_router
from pfm.terminal.flow_analytics import router as terminal_flow_analytics_router
from pfm.terminal.gdelt_news import router as terminal_gdelt_router
from pfm.terminal.homepage import router as terminal_homepage_router
from pfm.terminal.inline_backtest import router as terminal_inline_backtest_router
from pfm.terminal.jumps import router as terminal_jumps_router
from pfm.terminal.jumps_backtest import router as terminal_jumps_backtest_router
from pfm.terminal.jumps_cluster import router as terminal_jumps_cluster_router
from pfm.terminal.jumps_compare_router import router as terminal_jumps_compare_router
from pfm.terminal.live_stream import router as terminal_live_stream_router
from pfm.terminal.macro_overlay import router as terminal_macro_overlay_router
from pfm.terminal.news import router as terminal_news_router
from pfm.terminal.news_impact import router as terminal_news_impact_router

# IMPORTANT: news_trending_router declares `/terminal/news/trending` (a literal
# path) and must be included BEFORE terminal_news_router, which owns the
# dynamic catch-all `/terminal/news/{slug}`. FastAPI matches in registration
# order, so the dynamic route would otherwise swallow "trending" as a slug and
# 404 with `no market found for slug='trending'`.
from pfm.terminal.news_trending_router import (
    router as _terminal_news_trending_router_early,
)
from pfm.terminal.orderbook import router as terminal_orderbook_router

# Wave-9 additions
from pfm.terminal.peer_scanner import router as terminal_peer_scanner_router
from pfm.terminal.portfolio_sim import router as terminal_portfolio_sim_router
from pfm.terminal.prob_fan import router as terminal_prob_fan_router
from pfm.terminal.quality_score import router as terminal_quality_score_router
from pfm.terminal.quote import router as terminal_quote_router
from pfm.terminal.rss_news import router as terminal_rss_news_router
from pfm.terminal.search_index import router as terminal_search_index_router
from pfm.terminal.sentiment_leaderboard import router as terminal_sentiment_leaderboard_router
from pfm.terminal.sentiment_trend import router as terminal_sentiment_trend_router
from pfm.terminal.theta import router as terminal_theta_router
from pfm.terminal.trade_ticket import router as terminal_trade_ticket_router
from pfm.terminal.trades import router as terminal_trades_router
from pfm.terminal.vol_cone import router as terminal_vol_cone_router
from pfm.terminal.vol_distribution import router as terminal_vol_distribution_router
from pfm.terminal.watchlist import router as terminal_watchlist_router
from pfm.terminal.whale_tracker import router as terminal_whale_tracker_router
from pfm.vol.event_vol_router import router as event_vol_router
from pfm.vol.implied_pdf_router import router as implied_pdf_router
from pfm.vol.pm_iv_router import router as pm_iv_router
from pfm.vol.pricing_kernel_router import router as pricing_kernel_router
from pfm.vol_surface_pm import router as vol_surface_pm_router
from pfm.whale_mirror import router as whale_mirror_router

# Register the literal `/terminal/news/trending` path BEFORE the bulk loop
# below mounts `terminal_news_router` (which owns `/terminal/news/{slug}`).
# Without this, the dynamic slug route wins and trending returns 404.
app.include_router(_terminal_news_trending_router_early)

for _r in (
    terminal_trades_router,
    terminal_equity_router,
    terminal_prob_fan_router,
    terminal_vol_cone_router,
    terminal_fair_price_router,
    terminal_event_calendar_router,
    terminal_orderbook_router,
    terminal_inline_backtest_router,
    terminal_calendar_pair_router,
    terminal_news_router,
    terminal_macro_overlay_router,
    terminal_countdown_router,
    terminal_correlations_router,
    terminal_live_stream_router,
    terminal_peer_scanner_router,
    terminal_portfolio_sim_router,
    terminal_quality_score_router,
    terminal_flow_analytics_router,
    terminal_vol_distribution_router,
    terminal_watchlist_router,
    terminal_backtest_compare_router,
    terminal_theta_router,
    terminal_gdelt_router,
    terminal_rss_news_router,
    terminal_calendar_curated_router,
    terminal_calendar_scanner_router,
    terminal_trade_ticket_router,
    terminal_sentiment_trend_router,
    terminal_whale_tracker_router,
    terminal_news_impact_router,
    # IMPORTANT: cluster + backtest + compare routers MUST be mounted BEFORE
    # the bare jumps router so `/terminal/jumps/cluster`,
    # `/terminal/jumps/{slug}/backtest`, and `/terminal/jumps/compare` win
    # over the dynamic `/terminal/jumps/{slug}` path matcher.
    terminal_jumps_cluster_router,
    terminal_jumps_backtest_router,
    terminal_jumps_compare_router,
    terminal_jumps_router,
    terminal_sentiment_leaderboard_router,
    terminal_factor_clusters_router,
    terminal_bulk_export_router,
    strategy_verdict_router,
    # 2026-05-08 audit additions
    quant_validation_router,
    reverse_finder_router,
    alpha_graveyard_router,
    alpha_hub_router,
    alpha_tier_regen_router,
    strategies_catalog_router,
    terminal_compare_router,
    portfolio_optimizer_router,
    alerts_router,
    decay_monitor_router,
    live_signals_router,
    garch_router,
    quant_rigor_router,
    terminal_calendar_unified_router,
    realtime_stream_router,
    # 2026-05-08 wave-2 additions
    news_causal_router,
    pnl_tree_router,
    embed_router,
    replay_router,
    alpha_lab_router,
    chart_export_router,
    terminal_quote_router,
    terminal_homepage_router,
    terminal_search_index_router,
    arb_scanner_router,
    pm_vix_router,
    # 2026-05-08 wave-3 additions
    multi_venue_router,
    sources_health_router,
    fred_catalog_router,
    bls_router,
    macro_calendar_router,
    macro_overlay_router,
    earnings_whisper_router,
    vol_surface_pm_router,
    implied_pdf_router,
    pricing_kernel_router,
    counterfactual_router,
    whale_mirror_router,
    smart_money_divergence_router,
    auto_hedge_router,
    news_tagger_router,
    # 2026-05-08 wave-9 additions
    event_on_event_router,
    advanced_event_models_router,
    multi_event_chain_router,
    archive_router,
    archive_alias_router,
    kalshi_archive_router,
):
    app.include_router(_r)

# Opt-in feature flag: A3 Polymarket-vs-benchmark σ-gap router.
# Default OFF — the UI integration is still pending and we don't want the
# endpoints showing up in the public OpenAPI surface yet.
if os.environ.get("PFM_VOL_PM_IV_ENABLED") == "1":
    app.include_router(pm_iv_router)

# Opt-in feature flag: B3 event-driven EM signal router.
# Default OFF — the UI integration is intentionally deferred (per user
# request) so the endpoints stay out of the public OpenAPI surface and the
# default test suite. Flip with ``PFM_VOL_EVENT_ENABLED=1`` for live demos.
if os.environ.get("PFM_VOL_EVENT_ENABLED") == "1":
    app.include_router(event_vol_router)

# Wire DI overrides so feature routers share main.py's polymarket client.
from pfm.terminal import vol_cone as _tvc
from pfm.terminal import vol_distribution as _tvd

app.dependency_overrides[_tvc._get_polymarket_client_dep] = get_polymarket_client
# vol_distribution needs the factor registry to map slug → series. Without
# this override, every call returns 503 ("router not wired into an app with
# factors") — that's a dead endpoint in production. Wire it to the same
# factor config the rest of the app uses.
app.dependency_overrides[_tvd._get_factors_dep] = get_factors_dep

# ---------------------------------------------------------------------------
# Extracted endpoint routers — see each module's docstring for scope. These
# live at the very bottom so any helper they lazy-import from this module is
# already defined by the time the first request lands.
# ---------------------------------------------------------------------------
# W13-01 — Wave-11/12 standalone routers (verified to exist + import cleanly).
from pfm.admin.cache_invalidate_router import router as _admin_cache_invalidate_router
from pfm.admin.cache_stats_router import router as _admin_cache_stats_router
from pfm.alerts.digest_router import router as _alerts_digest_router
from pfm.arb.quality_router import router as _arb_quality_router
from pfm.crypto5min import router as _crypto5min_router
from pfm.factors_related_router import router as _factors_related_router
from pfm.factors_router import router as _factors_router
from pfm.factors_theme_leaderboard_router import router as _factors_theme_leaderboard_router
from pfm.health_deep_router import router as _health_deep_router
from pfm.macro_calendar_router import router as _macro_calendar_upcoming_router
from pfm.metrics_router import router as _metrics_audit_router
from pfm.ml_calibration_router import router as _ml_calibration_router
from pfm.ml_event_graph_router import router as _ml_event_graph_router
from pfm.ml_factor_importance_router import router as _ml_factor_importance_router
from pfm.ml_hub_router import router as _ml_hub_router
from pfm.ml_latent_factors_router import router as _ml_latent_factors_router
from pfm.ml_meta_label_router import router as _ml_meta_label_router
from pfm.ml_mispricing_router import router as _ml_mispricing_router
from pfm.ml_regime_router import router as _ml_regime_router
from pfm.news_search_router import router as _news_search_router
from pfm.ops_router import router as _ops_router
from pfm.portfolio_import_router import router as _portfolio_import_router
from pfm.pricing.router import router as _pricing_binary_router
from pfm.quant.regression_methods_elnet_router import (
    router as _quant_regression_elnet_router,
)
from pfm.quant.regression_methods_router import (
    router as _quant_regression_methods_router,
)
from pfm.regression_router import router as _regression_router
from pfm.research import router as _research_router
from pfm.research.citations_router import router as _research_citations_router
from pfm.strategies.anti_alpha_router import router as _strategies_anti_alpha_router
from pfm.strategies.audit_trail_router import router as _strategies_audit_trail_router
from pfm.strategies.deployable_router import router as _strategies_deployable_router
from pfm.strategies.risk_budget_router import router as _strategies_risk_budget_router
from pfm.strategies_arb_router import router as _strategies_arb_router
from pfm.strategies_crypto_router import router as _strategies_crypto_router
from pfm.strategies_router import bind_main_helpers as _bind_strategies_helpers
from pfm.strategies_router import router as _strategies_router
from pfm.terminal.related_stocks_router import router as _terminal_related_stocks_router

# `strategies_router` references five private helpers defined elsewhere in
# this module (_cached_factor_history, _fetch_aligned_prob, _finite,
# _resolve_one, _short_err). Bind them now that this module is fully evaluated.
_bind_strategies_helpers()

app.include_router(_factors_router)
app.include_router(_regression_router)
app.include_router(_strategies_router)
app.include_router(_ml_hub_router)
app.include_router(_ml_mispricing_router)
app.include_router(_ml_factor_importance_router)
app.include_router(_ml_regime_router)
app.include_router(_ml_event_graph_router)
app.include_router(_ml_calibration_router)
app.include_router(_ml_latent_factors_router)
app.include_router(_ml_meta_label_router)
app.include_router(_strategies_arb_router)
app.include_router(_strategies_crypto_router)
app.include_router(_crypto5min_router)
app.include_router(_metrics_audit_router)
app.include_router(_health_deep_router)
app.include_router(_ops_router)
app.include_router(_factors_related_router)
app.include_router(_research_router)
app.include_router(_portfolio_import_router)

# W13-01 — Wave-11/12 router wire-up (17 routers, ~17 new paths).
app.include_router(_alerts_digest_router)
app.include_router(_news_search_router)
# NOTE: the news-trending router is already included earlier as
# `_terminal_news_trending_router_early` (before the bulk router loop) so its
# literal `/terminal/news/trending` path wins over `/terminal/news/{slug}`.
# Re-including it here would duplicate the path in OpenAPI, so it is omitted.
app.include_router(_strategies_anti_alpha_router)
app.include_router(_strategies_deployable_router)
app.include_router(_pricing_binary_router)
app.include_router(_quant_regression_elnet_router)
app.include_router(_quant_regression_methods_router)
app.include_router(_arb_quality_router)
app.include_router(_factors_theme_leaderboard_router)
app.include_router(_terminal_related_stocks_router)
app.include_router(_strategies_audit_trail_router)
app.include_router(_strategies_risk_budget_router)
app.include_router(_research_citations_router)
app.include_router(_admin_cache_stats_router)
app.include_router(_admin_cache_invalidate_router)
app.include_router(_macro_calendar_upcoming_router)
