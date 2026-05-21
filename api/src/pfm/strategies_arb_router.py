"""``/strategies/arb/*`` — Kalshi ↔ Polymarket cross-venue arbitrage view.

Read-only view layer over the ``arbstuff/`` directory at the repo root. The
detector engine (``arbstuff/arb_engine.py``) writes its scan state to JSON
files; this router surfaces them through the FastAPI app and also exposes a
2-second SSE stream so the UI can render a Bloomberg-style live dashboard.

When the engine is not running, ``dashboard_state.json`` is missing — the
router falls back to a synthetic state computed live from
:func:`pfm.arb_scanner.top_arbs` so the panel still shows opportunities.

Endpoints
---------
``GET  /strategies/arb/state``              — current snapshot (one-shot).
``GET  /strategies/arb/stream``             — SSE pushing the snapshot every 2s.
``GET  /strategies/arb/pnl``                — simulated trade log + total PnL.
``GET  /strategies/arb/detection-history``  — rolling history of detections.
``GET  /strategies/arb/config-stats``       — counts per markets-config file.
``GET  /strategies/arb/config-events``      — merged mapped events (universe).
``GET  /strategies/arb/politics-events``    — politics-specialised events.
``GET  /strategies/arb/orderbook``          — live Kalshi+Polymarket book.
``GET  /strategies/arb/markets``            — paginated mapped pairs.
``GET  /strategies/arb/config``             — runtime control snapshot.
``POST /strategies/arb/blacklist``          — append ``arb_key`` to skiplist.
``DELETE /strategies/arb/blacklist``        — clear blacklist.
``POST /strategies/arb/settings``           — merge keys into control file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from pfm.auth.dependencies import require_admin
from pfm.sources.polymarket_pool import PolymarketHTTPPool

# Prefer orjson when actually installed (3-5x faster encode for the 135 KB
# /state blob) but degrade to FastAPI's default JSONResponse otherwise.
# fastapi.responses.ORJSONResponse is *importable* even without orjson,
# but it asserts on render — so probe the runtime import too.
try:
    import orjson as _orjson  # type: ignore[import-not-found]  # noqa: F401
    from fastapi.responses import ORJSONResponse  # type: ignore[attr-defined]

    _JSON_RESPONSE_CLS: type[Response] = ORJSONResponse
except ImportError:  # pragma: no cover - falls back when orjson missing
    _JSON_RESPONSE_CLS = JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies/arb", tags=["strategies-arb"])

# Repo-root / arbstuff. The router lives at api/src/pfm/, so go up 4 levels:
# pfm/ → src/ → api/ → repo-root → arbstuff/.
_ARB_DIR: Path = Path(__file__).resolve().parents[3] / "arbstuff"

# Stream tick cadence. Matches review_app.py's 2s SSE generator.
_STREAM_TICK_SECONDS = float(os.environ.get("PFM_ARB_STREAM_TICK_S", "5.0"))

# SSE keep-alive cadence (comment-line ``: ping`` frames). Defends against
# idle proxy timeouts (nginx/cloud LBs typically idle-close after 60 s) when
# the data tick is slow or the upstream cache build stalls. Standard SSE
# comments are silently ignored by EventSource clients.
HEARTBEAT_INTERVAL_S: float = float(os.environ.get("PFM_ARB_STREAM_HEARTBEAT_S", "15.0"))

# Self-terminate after ``MAX_STREAM_SECONDS`` so the client reconnects and
# the server doesn't leak generators on long-lived sessions. Browsers
# auto-reconnect EventSource after a clean close, so this is transparent.
MAX_STREAM_SECONDS: float = float(os.environ.get("PFM_ARB_STREAM_MAX_S", "600.0"))

# Trim the heaviest field in the SSE payload — ``scan_log`` from the engine
# can be 200+ entries (≈80 KB), pushed every tick. The UI only renders the
# most-recent N rows in the Scan Log tab; the rest is unused churn.
_STREAM_SCAN_LOG_MAX = int(os.environ.get("PFM_ARB_STREAM_SCAN_LOG_MAX", "30"))

# Fallback to live arb_scanner when dashboard_state.json is missing/stale.
# On by default — when the engine isn't running we still want the panel to
# show real opportunities computed from Polymarket+Kalshi directly.
_LIVE_FALLBACK_ENABLED = os.environ.get("PFM_ARB_LIVE_FALLBACK", "1") != "0"
_STATE_STALE_SECONDS = float(os.environ.get("PFM_ARB_STATE_STALE_S", "180"))

# Cache for the live-fallback scanner result so repeated SSE ticks don't
# hammer Polymarket+Kalshi. TTL matches the engine's normal cycle.
_FALLBACK_CACHE: dict[str, Any] = {"t": 0.0, "value": None}
_FALLBACK_TTL = float(os.environ.get("PFM_ARB_FALLBACK_TTL_S", "30"))

# Redis-backed mirror of the engine state. Lets the gunicorn workers + a
# restarted process see the same data without sharing a local disk. The
# mirroring is done by ``_arb_state_mirror`` in pfm.main (writes Redis,
# routers read it). Set ``PFM_ARB_REDIS_ENABLED=0`` to skip Redis and rely
# only on the on-disk file (dev mode without a cache).
_ARB_STATE_REDIS_KEY = "arb:dashboard_state"
_ARB_STATE_REDIS_AGE_KEY = "arb:dashboard_state:age"
_ARB_REDIS_ENABLED = os.environ.get("PFM_ARB_REDIS_ENABLED", "1") != "0"


def _read_engine_state_from_redis() -> dict[str, Any] | None:
    """Look up the mirrored engine state from Redis. Returns ``None`` on miss."""
    if not _ARB_REDIS_ENABLED:
        return None
    try:
        from pfm.main import app as _app  # circular-safe — read at call time

        cache = getattr(_app.state, "cache", None)
        if cache is None or not getattr(cache, "enabled", False):
            return None
        raw = cache.get(_ARB_STATE_REDIS_KEY)
        if not raw:
            return None
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except json.JSONDecodeError as e:
        # Corrupt cache entry — don't crash the endpoint, but DO log so an
        # operator notices (silent recovery + disk-fallback was hiding bugs
        # where Redis was mid-write or a serialiser regressed).
        logger.warning("arb-state: Redis key %s holds malformed JSON: %s", _ARB_STATE_REDIS_KEY, e)
        return None
    except Exception as e:  # defensive: never break /state on cache I/O
        logger.warning("arb-state: Redis read failed (%s); falling back to disk.", e)
        return None


# Minimum spread % for the fallback scanner. Lower → more opportunities
# detected (and shown to the user), but more noise.
_FALLBACK_MIN_SPREAD_PCT = float(os.environ.get("PFM_ARB_FALLBACK_MIN_SPREAD", "0.5"))
_FALLBACK_TOP_N = int(os.environ.get("PFM_ARB_FALLBACK_TOP_N", "50"))

# Detection-history ring buffer. Each scan appends *new* opportunities
# (deduped by arb_key) so the History tab + PnL tab show data even when
# the standalone engine isn't running. Bounded so memory stays constant.
_DETECTION_HISTORY: list[dict[str, Any]] = []
_DETECTION_SEEN: dict[str, float] = {}  # arb_key → first-seen unix ts
_DETECTION_MAX = int(os.environ.get("PFM_ARB_DETECTION_MAX", "500"))


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _safe_read_json(path: Path) -> Any | None:
    """Return parsed JSON or ``None`` (never raise — engine may not be running)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("arb: %s unreadable: %s", path.name, exc)
        return None


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write ``value`` to ``path`` atomically (``.tmp`` + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def _state_age_seconds() -> float | None:
    """Return mtime delta of ``dashboard_state.json`` in seconds, or ``None``."""
    p = _ARB_DIR / "dashboard_state.json"
    if not p.exists():
        return None
    try:
        return max(0.0, time.time() - p.stat().st_mtime)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Live-fallback: synthesise an engine-shaped state from arb_scanner.top_arbs
# ---------------------------------------------------------------------------


def _shape_fallback_opp(arb: dict[str, Any]) -> dict[str, Any]:
    """Convert :func:`pfm.arb_scanner.top_arbs` dict to engine shape."""
    direction = str(arb.get("direction", "")).lower()
    if "pm_yes" in direction and "kalshi_no" in direction:
        side_str = "Buy P_YES+K_NO"
        type_str = "Buy K_NO+P_YES"
    elif "kalshi_yes" in direction and "pm_no" in direction:
        side_str = "Buy K_YES+P_NO"
        type_str = "Buy K_YES+P_NO"
    else:
        side_str = direction or "?"
        type_str = direction or "?"
    pm_p = float(arb.get("pm_price", 0.0))
    k_p = float(arb.get("kalshi_price", 0.0))
    cost = pm_p + k_p
    pm_slug = arb.get("pm_slug", "")
    k_slug = arb.get("kalshi_slug", "")
    return {
        "name": arb.get("label") or f"{pm_slug} / {k_slug}",
        "type": type_str,
        "side": side_str,
        "profit_pct": round(float(arb.get("spread_pct", 0.0)), 2),
        "volume": float(arb.get("tradeable_size_usd", 0.0)),
        "cost": round(cost, 4),
        "kalshi_price": round(k_p, 4),
        "poly_price": round(pm_p, 4),
        "kalshi_ticker": k_slug,
        "kalshi_event_ticker": k_slug,
        "poly_slug": pm_slug,
        "poly_token_id": "",
        "neg_risk": False,
        "arb_key": f"{k_slug}__{pm_slug}",
        "source": "discovered",
        "kalshi_fee": round(0.07 * k_p * (1 - k_p), 4),
        "poly_fee": round(0.04 * pm_p * (1 - pm_p), 4),
        "spread": round(abs(k_p - pm_p), 4),
        "timestamp": arb.get("last_seen_iso", ""),
        "confirmed": bool(arb.get("confirmed", False)),
        "half_life_minutes": float(arb.get("half_life_minutes", 0.0)),
    }


def _build_fallback_state() -> dict[str, Any]:
    """Synthesise an engine-shaped state object from live ``arb_scanner.top_arbs``.

    Cached for ``_FALLBACK_TTL`` seconds. On scanner errors returns an empty
    skeleton — callers must always get a usable dict back.

    NOTE: ``top_arbs`` does many synchronous HTTP fetches. Calling it from
    inside an async request handler — especially the 2-second SSE tick —
    blocks the event loop and stalls every other endpoint. Callers in an
    async context should use :func:`_build_fallback_state_async` instead,
    which off-loads the scan to a worker thread.
    """
    now = time.time()
    if _FALLBACK_CACHE["value"] is not None and (now - _FALLBACK_CACHE["t"]) < _FALLBACK_TTL:
        return _FALLBACK_CACHE["value"]

    try:
        from pfm.arb_scanner import top_arbs

        arbs = top_arbs(
            min_spread_pct=_FALLBACK_MIN_SPREAD_PCT,
            n=_FALLBACK_TOP_N,
        )
        opps = [_shape_fallback_opp(a) for a in arbs]
        opps.sort(key=lambda o: o["profit_pct"], reverse=True)
        _record_detections(opps)
    except Exception as exc:
        logger.warning("arb: live fallback failed: %s", exc)
        opps = []

    state = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        "scan_count": int((now // _FALLBACK_TTL) % 10_000),
        "cycle_time_s": round(_FALLBACK_TTL, 1),
        "balances": {"kalshi": 0.0, "polymarket": 0.0},
        "config": {
            "poll_interval": int(_FALLBACK_TTL),
            "threshold": 0.94,
            "min_alert_profit": 1.0,
            "event_count": len(opps),
        },
        "bot_status": "fallback" if opps else "idle",
        "test_mode": True,
        "scan_mode": "FALLBACK",
        "candidates_count": len(opps),
        "opportunities": opps,
        "scan_log": [],
        "_source": "live_fallback",
    }
    _FALLBACK_CACHE["t"] = now
    _FALLBACK_CACHE["value"] = state
    return state


def _empty_state_envelope() -> dict[str, Any]:
    """Return the ``bot_status: 'offline'`` envelope used when nothing works."""
    return {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        "scan_count": 0,
        "cycle_time_s": 0,
        "balances": {"kalshi": 0.0, "polymarket": 0.0},
        "config": {
            "poll_interval": 8,
            "threshold": 0.94,
            "min_alert_profit": 1.0,
            "event_count": 0,
        },
        "bot_status": "offline",
        "test_mode": True,
        "scan_mode": "OG",
        "candidates_count": 0,
        "opportunities": [],
        "scan_log": [],
        "hint": (
            "Arb engine not running. Launch with "
            "`cd arbstuff && python arb_engine.py` for live scanning, "
            "or set PFM_ARB_LIVE_FALLBACK=1 to use the built-in scanner."
        ),
        "_source": "empty",
    }


def _load_state_or_fallback() -> dict[str, Any]:
    """Return the current state dict (Redis mirror → engine file → fallback)."""
    # Redis first — survives ephemeral container disks + works across all
    # gunicorn workers without any worker needing its own filesystem.
    state = _read_engine_state_from_redis()
    if state is None:
        state = _safe_read_json(_ARB_DIR / "dashboard_state.json")
    if state is not None:
        # Engine wrote within the freshness window — trust it.
        age = _state_age_seconds()
        if age is None or age <= _STATE_STALE_SECONDS:
            state.setdefault("opportunities", [])
            state.setdefault("scan_log", [])
            state.setdefault("balances", {"kalshi": 0.0, "polymarket": 0.0})
            state.setdefault("config", {})
            state["config"].setdefault("threshold", 0.94)
            state["config"].setdefault("min_alert_profit", 1.0)
            state["config"].setdefault("event_count", 0)
            state.setdefault("bot_status", "running")
            state.setdefault("test_mode", True)
            state.setdefault("scan_mode", "OG")
            state.setdefault("scan_count", 0)
            state.setdefault("cycle_time_s", 0)
            state.setdefault("candidates_count", len(state.get("opportunities", [])))
            state["_source"] = "engine"
            state["_state_age_s"] = round(age or 0.0, 1)
            return state

    # File missing or stale.
    if _LIVE_FALLBACK_ENABLED:
        return _build_fallback_state()
    return _empty_state_envelope()


# ---------------------------------------------------------------------------
# Endpoints — read-only
# ---------------------------------------------------------------------------


@router.get("/state", summary="Live arb engine state — opportunities + scan log")
async def get_state() -> Response:
    """Return current detected arbs + engine status.

    The shape mirrors what ``arbstuff/arb_engine.py`` writes to
    ``dashboard_state.json``: ``timestamp``, ``scan_count``, ``cycle_time_s``,
    ``balances``, ``config``, ``bot_status``, ``test_mode``, ``scan_mode``,
    ``candidates_count``, ``opportunities[]``, ``scan_log[]``.

    Falls back to the in-process scanner (``pfm.arb_scanner.top_arbs``) when
    the engine file is missing or older than ``PFM_ARB_STATE_STALE_S`` seconds
    (default 180).

    Performance hardening (2026-05-15):
      - Encoded with ``ORJSONResponse`` when ``orjson`` is installed
        (~3-5x faster than the stdlib ``json`` encode for the 135 KB blob).
        Falls back to ``JSONResponse`` transparently otherwise.
      - ``Cache-Control: public, max-age=2`` so the browser revalidates the
        body only every 2 s. The SSE stream still pushes every tick;
        ``/state`` is for the initial paint + manual refresh.

    Uses the async loader so a stale-fallback rebuild (``top_arbs`` fans out
    to ~10 sync HTTP requests, several seconds) runs in a worker thread
    instead of blocking the uvicorn worker — measured: 0.8-3.6 s spikes
    eliminated on the user-facing XHR.
    """
    state = await _load_state_or_fallback_async()
    return _JSON_RESPONSE_CLS(
        content=state,
        headers={"Cache-Control": "public, max-age=2"},
    )


@router.get("/pnl", summary="Simulated PnL log from arb_engine test-mode trades")
def get_pnl() -> dict[str, Any]:
    """Return ``{trades, total_pnl, count}`` reading ``arb_pnl_log.json``.

    Falls back to a simulated PnL from the in-process detection buffer when
    the engine's log doesn't exist — each detection is treated as a unit
    notional "trade" with profit = profit_pct%, so the user sees something
    actionable in the PnL tab even without arb_engine.py running.
    """
    trades = _safe_read_json(_ARB_DIR / "arb_pnl_log.json") or []
    if isinstance(trades, list) and trades:
        total = sum(float(t.get("guaranteed_profit", 0.0)) for t in trades)
        return {
            "trades": trades,
            "total_pnl": round(total, 4),
            "count": len(trades),
            "_source": "engine",
        }
    # Fallback: synthesise from detection buffer.
    syn = []
    for d in _DETECTION_HISTORY:
        # Treat each unique detection as a hypothetical $100 notional trade,
        # profit = profit_pct% of that. Lets the PnL tab show realistic
        # numbers from live scanner output.
        notional = 100.0
        profit = (float(d.get("profit_pct", 0.0)) / 100.0) * notional
        syn.append(
            {
                "timestamp": d.get("first_seen_iso", ""),
                "name": d.get("name", ""),
                "side": d.get("side", ""),
                "notional_usd": notional,
                "guaranteed_profit": round(profit, 4),
                "profit_pct": d.get("profit_pct", 0.0),
                "arb_key": d.get("arb_key", ""),
                "kalshi_ticker": d.get("kalshi_ticker", ""),
                "poly_slug": d.get("poly_slug", ""),
            }
        )
    total = sum(float(t.get("guaranteed_profit", 0.0)) for t in syn)
    return {
        "trades": syn,
        "total_pnl": round(total, 4),
        "count": len(syn),
        "_source": "fallback_synthetic",
    }


@router.get(
    "/detection-history",
    summary="Rolling history of detected arbs (newest-first)",
)
def get_detection_history() -> dict[str, Any]:
    """Return ``{items, count}``.

    Source priority:
    * If the engine wrote ``arb_detection_history.json`` → return it.
    * Else return the in-process detection buffer (populated by the
      fallback scanner each time it builds the cache).
    """
    items = _safe_read_json(_ARB_DIR / "arb_detection_history.json")
    if isinstance(items, list) and items:
        return {
            "items": list(reversed(items)),
            "count": len(items),
            "_source": "engine",
        }
    # Newest-first from in-process buffer.
    buf = sorted(
        _DETECTION_HISTORY,
        key=lambda d: d.get("first_seen_unix", 0.0),
        reverse=True,
    )
    return {
        "items": buf,
        "count": len(buf),
        "_source": "fallback_buffer",
    }


def _record_detections(opps: list[dict[str, Any]]) -> None:
    """Append previously-unseen opportunities to the detection buffer.

    Dedup key: ``arb_key`` (kalshi_ticker__poly_slug). Bounded by
    ``_DETECTION_MAX`` so memory stays constant for long-running servers.
    """
    now_unix = time.time()
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    for o in opps:
        key = o.get("arb_key") or ""
        if not key or key in _DETECTION_SEEN:
            continue
        _DETECTION_SEEN[key] = now_unix
        _DETECTION_HISTORY.append(
            {
                "first_seen_unix": now_unix,
                "first_seen_iso": now_iso,
                "name": o.get("name", ""),
                "side": o.get("side", ""),
                "profit_pct": o.get("profit_pct", 0.0),
                "volume": o.get("volume", 0.0),
                "cost": o.get("cost", 0.0),
                "kalshi_price": o.get("kalshi_price", 0.0),
                "poly_price": o.get("poly_price", 0.0),
                "kalshi_ticker": o.get("kalshi_ticker", ""),
                "poly_slug": o.get("poly_slug", ""),
                "arb_key": key,
            }
        )
    # Bound the buffer — drop oldest beyond cap.
    if len(_DETECTION_HISTORY) > _DETECTION_MAX:
        excess = len(_DETECTION_HISTORY) - _DETECTION_MAX
        for d in _DETECTION_HISTORY[:excess]:
            _DETECTION_SEEN.pop(d.get("arb_key", ""), None)
        del _DETECTION_HISTORY[:excess]


def _count_mapping(cfg: dict[str, Any] | None) -> dict[str, int]:
    if not cfg or not isinstance(cfg, dict):
        return {"total": 0, "mapped": 0}
    events = cfg.get("events", []) or []
    mapped = sum(1 for e in events if (e or {}).get("mapping"))
    return {"total": len(events), "mapped": mapped}


@router.get("/config-stats", summary="Mapping counts per source file")
def get_config_stats() -> dict[str, Any]:
    """Return ``{reviewed, main, politics, discovered, combined_mapped}``."""
    reviewed = _count_mapping(_safe_read_json(_ARB_DIR / "markets_config_reviewed.json"))
    main_ = _count_mapping(_safe_read_json(_ARB_DIR / "markets_config.json"))
    politics = _count_mapping(_safe_read_json(_ARB_DIR / "markets_config_politics.json"))
    discovered = _count_mapping(_safe_read_json(_ARB_DIR / "markets_config_discovered.json"))
    combined = reviewed["mapped"] + main_["mapped"]
    return {
        "reviewed": reviewed,
        "main": main_,
        "politics": politics,
        "discovered": discovered,
        "combined_mapped": combined,
    }


@router.get("/config-events", summary="Merged mapped-event universe")
def get_config_events() -> dict[str, Any]:
    """Return ``{events: [...]}`` merged across all four config files.

    Dedupe by ``kalshi_ticker`` with first-wins priority:
    ``reviewed → main → politics → discovered``. Only events with non-empty
    ``mapping`` are included. Each event gets a ``source`` field.
    """
    sources = [
        ("reviewed", _ARB_DIR / "markets_config_reviewed.json"),
        ("main", _ARB_DIR / "markets_config.json"),
        ("politics", _ARB_DIR / "markets_config_politics.json"),
        ("discovered", _ARB_DIR / "markets_config_discovered.json"),
    ]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for src, path in sources:
        cfg = _safe_read_json(path)
        if not cfg:
            continue
        for e in cfg.get("events", []) or []:
            tk = (e or {}).get("kalshi_ticker", "")
            mapping = (e or {}).get("mapping") or {}
            if not tk or not mapping or tk in seen:
                continue
            seen.add(tk)
            out.append(
                {
                    "name": e.get("name"),
                    "kalshi_ticker": tk,
                    "poly_slug": e.get("poly_slug"),
                    "mapping": mapping,
                    "source": src,
                }
            )
    return {"events": out}


_OFFICE_RE = re.compile(r"\b(HOUSE|SEN|GOV|LTGOV|AG|SOS|TREAS|PRES|MAYOR)\b", re.IGNORECASE)
_RACE_TYPE_RE = re.compile(r"\((primary|special|runoff)\)", re.IGNORECASE)
_PARTY_RE = re.compile(r"\[(D|R|I|L)\]")
_DISTRICT_RE = re.compile(r"District (\d+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(20\d{2})")
_STATE_2 = re.compile(r"\b([A-Z]{2})\b")

_STATE_NAMES = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def _parse_politics(name: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "state": None,
        "office": None,
        "district": None,
        "race_type": "general",
        "party": None,
        "year": None,
    }
    for full, abbr in _STATE_NAMES.items():
        if name.startswith(full):
            info["state"] = abbr
            break
    if info["state"] is None:
        m = _STATE_2.match(name)
        if m:
            info["state"] = m.group(1)
    if m := _OFFICE_RE.search(name):
        info["office"] = m.group(1).upper()
    if m := _RACE_TYPE_RE.search(name):
        info["race_type"] = m.group(1).lower()
    if m := _PARTY_RE.search(name):
        info["party"] = m.group(1).upper()
    if m := _DISTRICT_RE.search(name):
        with contextlib.suppress(ValueError):
            info["district"] = int(m.group(1))
    if m := _YEAR_RE.search(name):
        with contextlib.suppress(ValueError):
            info["year"] = int(m.group(1))
    return info


@router.get("/politics-events", summary="Politics specialist mapping universe")
def get_politics_events() -> dict[str, Any]:
    """Return events from ``markets_config_politics.json`` with parsed fields."""
    cfg = _safe_read_json(_ARB_DIR / "markets_config_politics.json") or {}
    out: list[dict[str, Any]] = []
    by_state: dict[str, int] = {}
    by_office: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for e in cfg.get("events", []) or []:
        name = (e or {}).get("name") or ""
        mapping = (e or {}).get("mapping") or {}
        if not mapping:
            continue
        parsed = _parse_politics(name)
        out.append(
            {
                "name": name,
                "kalshi_ticker": e.get("kalshi_ticker"),
                "poly_slug": e.get("poly_slug"),
                "mapping_count": len(mapping),
                **parsed,
            }
        )
        if parsed["state"]:
            by_state[parsed["state"]] = by_state.get(parsed["state"], 0) + 1
        if parsed["office"]:
            by_office[parsed["office"]] = by_office.get(parsed["office"], 0) + 1
        by_type[parsed["race_type"]] = by_type.get(parsed["race_type"], 0) + 1
    return {
        "events": out,
        "total": len(out),
        "stats": {
            "by_state": by_state,
            "by_office": by_office,
            "by_type": by_type,
        },
    }


@router.get("/markets", summary="All mapped Kalshi↔Polymarket pairs (paginated)")
def get_markets(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    source: str = Query(default="all"),
) -> dict[str, Any]:
    """Paginated view of curated Kalshi↔Polymarket market pairs."""
    files = {
        "reviewed": _ARB_DIR / "markets_config_reviewed.json",
        "main": _ARB_DIR / "markets_config.json",
        "politics": _ARB_DIR / "markets_config_politics.json",
        "discovered": _ARB_DIR / "markets_config_discovered.json",
    }
    if source not in {"all", *files.keys()}:
        raise HTTPException(status_code=400, detail=f"unknown source: {source!r}")

    sources = list(files.values()) if source == "all" else [files[source]]
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sources:
        cfg = _safe_read_json(path)
        if not cfg:
            continue
        for e in cfg.get("events", []):
            tk = e.get("kalshi_ticker", "")
            if not tk or tk in seen:
                continue
            seen.add(tk)
            outcomes = list((e.get("mapping") or {}).items())
            events.append(
                {
                    "name": e.get("name"),
                    "kalshi_ticker": tk,
                    "poly_slug": e.get("poly_slug"),
                    "n_outcomes": len(outcomes),
                    "sample_outcomes": [{"kalshi_key": k, "poly_name": v} for k, v in outcomes[:5]],
                    "source_file": path.name,
                }
            )

    total = len(events)
    page = events[offset : offset + limit]
    next_offset = offset + limit if offset + limit < total else None
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "events": page,
    }


@router.get("/config", summary="Current scan threshold + mode + last-known control")
def get_config() -> dict[str, Any]:
    """Return ``dashboard_control.json`` — runtime toggles."""
    control = _safe_read_json(_ARB_DIR / "dashboard_control.json") or {}
    return {
        "threshold": control.get("threshold", 0.94),
        "min_alert_profit": control.get("min_alert_profit", 1.0),
        "scan_mode": control.get("scan_mode", "OG"),
        "email_enabled": control.get("email_enabled", False),
        "blacklist_size": len(_safe_read_json(_ARB_DIR / "arb_blacklist.json") or []),
        "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Orderbook proxy — Kalshi + Polymarket public REST
# ---------------------------------------------------------------------------


_HTTP_TIMEOUT = httpx.Timeout(8.0, connect=4.0)


async def _fetch_kalshi_orderbook(ticker: str) -> dict[str, Any]:
    """Public Kalshi orderbook for a market ticker.

    Endpoint shape: ``GET /trade-api/v2/markets/{ticker}/orderbook``.
    Response: ``{orderbook: {yes: [[price,size], ...], no: [[price,size], ...]}}``
    where prices are integers in cents (0–100). We convert to dollars to match
    what the engine writes.
    """
    if not ticker:
        return {}
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook"
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as cli:
            r = await cli.get(url, params={"depth": 20})
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("arb: kalshi orderbook fetch failed for %s: %s", ticker, exc)
        return {"error": str(exc)}
    ob = data.get("orderbook") or {}
    yes = ob.get("yes") or []
    no = ob.get("no") or []
    # Convert cents → dollars for both sides.
    yes_dollars = [[f"{(int(p) / 100):.2f}", str(s)] for p, s in yes if p is not None]
    no_dollars = [[f"{(int(p) / 100):.2f}", str(s)] for p, s in no if p is not None]
    return {"yes_dollars": yes_dollars, "no_dollars": no_dollars}


async def _fetch_polymarket_orderbook(token_id: str) -> dict[str, Any]:
    """Polymarket CLOB orderbook for a token.

    Endpoint shape: ``GET https://clob.polymarket.com/book?token_id=...``.
    Response: ``{bids: [{price,size}, ...], asks: [{price,size}, ...]}``.
    """
    if not token_id:
        return {}
    # W11-11 (T18 pool migration): reuse the shared CLOB client. The SSE
    # /strategies/arb/stream ticks every 2 s — a fresh AsyncClient per tick
    # paid TLS-handshake cost on every poll. base_url is set on the pool
    # client so callers use the relative path "/book".
    cli = PolymarketHTTPPool.instance().clob_client
    try:
        r = await cli.get("/book", params={"token_id": token_id}, timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("arb: polymarket orderbook fetch failed for %s: %s", token_id[:16], exc)
        return {"error": str(exc)}
    return {
        "bids": data.get("bids") or [],
        "asks": data.get("asks") or [],
    }


@router.get("/orderbook", summary="Live Kalshi + Polymarket orderbook proxy")
async def get_orderbook(
    kalshi_ticker: str = Query(default=""),
    poly_token: str = Query(default=""),
) -> dict[str, Any]:
    """Fetch both sides' orderbooks. At least one identifier required.

    Reject `{"kalshi":{},"poly":{}}` no-op responses by requiring the caller
    to supply at least one of ``kalshi_ticker`` / ``poly_token``. Otherwise
    a typo'd query (e.g. ``?slug=foo``) returns a misleading 200 with empty
    books — the frontend can't tell "no liquidity" apart from "bad input".
    """
    if not kalshi_ticker and not poly_token:
        raise HTTPException(
            status_code=422,
            detail=(
                "at least one of `kalshi_ticker` or `poly_token` is required "
                "(neither was provided — note: `slug` is not a valid query "
                "param for this endpoint)"
            ),
        )
    k_task = _fetch_kalshi_orderbook(kalshi_ticker)
    p_task = _fetch_polymarket_orderbook(poly_token)
    k_res, p_res = await asyncio.gather(k_task, p_task, return_exceptions=True)
    return {
        "kalshi": k_res if isinstance(k_res, dict) else {"error": str(k_res)},
        "poly": p_res if isinstance(p_res, dict) else {"error": str(p_res)},
    }


# ---------------------------------------------------------------------------
# Write endpoints — blacklist + settings
# ---------------------------------------------------------------------------


class BlacklistAddRequest(BaseModel):
    arb_key: str = Field(..., min_length=1)


_BLACKLIST_REDIS_KEY = "arb:blacklist"


def _get_cache() -> Any | None:
    """Best-effort handle on the shared Redis cache. Returns None when off."""
    try:
        from pfm.main import app as _app

        cache = getattr(_app.state, "cache", None)
        if cache is not None and getattr(cache, "enabled", False):
            return cache
    except Exception as e:  # never break a request on bootstrap import
        # Surface the failure via the log so a misconfigured cache module
        # doesn't silently degrade admin endpoints to disk-only mode (which
        # was returning {"ok": true} while the Redis mirror was broken).
        logger.warning("arb-router: shared cache handle unavailable: %s", e)
    return None


def _read_blacklist_combined() -> list[str]:
    """Merge file-backed blacklist with the Redis SET (cross-worker view)."""
    seen: set[str] = set()
    out: list[str] = []
    # Disk path — primary on single-machine deploys, fallback on multi.
    disk = _safe_read_json(_ARB_DIR / "arb_blacklist.json")
    if isinstance(disk, list):
        for key in disk:
            if isinstance(key, str) and key not in seen:
                seen.add(key)
                out.append(key)
    # Redis SET — cross-worker authoritative on multi-machine deploys.
    cache = _get_cache()
    if cache is not None:
        try:
            client = getattr(cache, "_client", None)
            if client is not None:
                members = client.smembers(_BLACKLIST_REDIS_KEY) or set()
                for m in members:
                    key = m.decode() if isinstance(m, bytes) else m
                    if key and key not in seen:
                        seen.add(key)
                        out.append(key)
        except Exception as e:  # defensive read; disk view is sufficient
            logger.warning("arb-blacklist: SMEMBERS failed (%s); using disk view only.", e)
    return out


@router.post(
    "/blacklist",
    summary="Append an arb_key to the blacklist",
    dependencies=[Depends(require_admin)],
)
def post_blacklist(body: BlacklistAddRequest) -> dict[str, Any]:
    """Add ``arb_key`` (idempotent) to ``arb_blacklist.json`` AND a Redis SET.

    The Redis SET lets other gunicorn workers see the addition immediately
    without waiting for a file mtime refresh. The disk write is kept as the
    canonical source for the engine subprocess + dev mode without Redis.
    """
    path = _ARB_DIR / "arb_blacklist.json"
    current = _safe_read_json(path)
    if not isinstance(current, list):
        current = []
    added = body.arb_key not in current
    if added:
        current.append(body.arb_key)
        _atomic_write_json(path, current)
    # Mirror to Redis SET regardless — idempotent.
    cache = _get_cache()
    redis_mirror = "disabled"
    if cache is not None:
        try:
            client = getattr(cache, "_client", None)
            if client is not None:
                client.sadd(_BLACKLIST_REDIS_KEY, body.arb_key)
                redis_mirror = "ok"
            else:
                redis_mirror = "unavailable"
        except Exception as e:  # disk write already succeeded
            logger.warning("arb-blacklist: SADD %s failed (%s); disk write OK.", body.arb_key, e)
            redis_mirror = f"failed: {type(e).__name__}"
    return {
        "ok": True,
        "blacklisted": len(current),
        "added": added,
        "redis_mirror": redis_mirror,
    }


@router.delete(
    "/blacklist",
    summary="Clear the blacklist",
    dependencies=[Depends(require_admin)],
)
def delete_blacklist() -> dict[str, Any]:
    """Truncate both the file and the Redis SET."""
    _atomic_write_json(_ARB_DIR / "arb_blacklist.json", [])
    cache = _get_cache()
    redis_mirror = "disabled"
    if cache is not None:
        try:
            client = getattr(cache, "_client", None)
            if client is not None:
                client.delete(_BLACKLIST_REDIS_KEY)
                redis_mirror = "ok"
            else:
                redis_mirror = "unavailable"
        except Exception as e:  # disk truncation already succeeded
            logger.warning("arb-blacklist: DEL key failed (%s); disk truncated OK.", e)
            redis_mirror = f"failed: {type(e).__name__}"
    return {"ok": True, "redis_mirror": redis_mirror}


@router.get("/blacklist", summary="List blacklisted arb_keys")
def get_blacklist() -> dict[str, Any]:
    """Read the union of file + Redis blacklist."""
    keys = _read_blacklist_combined()
    return {"blacklisted": keys, "count": len(keys)}


class SettingsRequest(BaseModel):
    email_enabled: bool | None = None
    threshold: float | None = Field(default=None, ge=0.5, le=1.5)
    min_alert_profit: float | None = Field(default=None, ge=0.0, le=100.0)
    scan_mode: str | None = None


@router.post(
    "/settings",
    summary="Merge runtime control toggles",
    dependencies=[Depends(require_admin)],
)
def post_settings(body: SettingsRequest) -> dict[str, Any]:
    """Merge keys into ``dashboard_control.json`` for the engine to pick up."""
    path = _ARB_DIR / "dashboard_control.json"
    control = _safe_read_json(path) or {}
    if body.email_enabled is not None:
        control["email_enabled"] = bool(body.email_enabled)
    if body.threshold is not None:
        control["threshold"] = float(body.threshold)
    if body.min_alert_profit is not None:
        control["min_alert_profit"] = float(body.min_alert_profit)
    if body.scan_mode is not None:
        if body.scan_mode not in {"OG", "WS"}:
            raise HTTPException(
                status_code=400,
                detail=f"scan_mode must be 'OG' or 'WS', got {body.scan_mode!r}",
            )
        control["scan_mode"] = body.scan_mode
    _atomic_write_json(path, control)
    return {"ok": True, "control": control}


# ---------------------------------------------------------------------------
# Server-sent events — mirrors review_app.py's /api/dashboard/stream
# ---------------------------------------------------------------------------


async def _load_state_or_fallback_async() -> dict[str, Any]:
    """Async wrapper: never blocks the event loop on a live scanner sweep.

    Critical for the SSE handler — calling the sync version on every tick
    would freeze the event loop for several seconds whenever the cache
    expired (each ``top_arbs`` call fans out to ~10 sync HTTP requests).

    Strategy:
    * If the engine wrote a fresh ``dashboard_state.json`` → return it.
    * Else if the in-process fallback cache is fresh → return that.
    * Else off-load the cache rebuild to a worker thread.
    """
    # Redis first — survives ephemeral container disks + cross-worker view.
    state = _read_engine_state_from_redis()
    if state is None:
        # Engine file read is cheap (small JSON); keep it sync. Wrap in
        # to_thread defensively in case of slow disk.
        state = await asyncio.to_thread(_safe_read_json, _ARB_DIR / "dashboard_state.json")
    if state is not None:
        age = _state_age_seconds()
        if age is None or age <= _STATE_STALE_SECONDS:
            state.setdefault("opportunities", [])
            state.setdefault("scan_log", [])
            state.setdefault("balances", {"kalshi": 0.0, "polymarket": 0.0})
            state.setdefault("config", {})
            state["config"].setdefault("threshold", 0.94)
            state["config"].setdefault("min_alert_profit", 1.0)
            state["config"].setdefault("event_count", 0)
            state.setdefault("bot_status", "running")
            state.setdefault("test_mode", True)
            state.setdefault("scan_mode", "OG")
            state.setdefault("scan_count", 0)
            state.setdefault("cycle_time_s", 0)
            state.setdefault("candidates_count", len(state.get("opportunities", [])))
            state["_source"] = "engine"
            state["_state_age_s"] = round(age or 0.0, 1)
            return state
    if not _LIVE_FALLBACK_ENABLED:
        return _empty_state_envelope()
    # Serve cached fallback if fresh — instantaneous.
    now = time.time()
    if _FALLBACK_CACHE["value"] is not None and (now - _FALLBACK_CACHE["t"]) < _FALLBACK_TTL:
        return _FALLBACK_CACHE["value"]
    # Cache miss — run the scanner in a worker thread so the event loop
    # stays free to serve other requests during the multi-second sweep.
    return await asyncio.to_thread(_build_fallback_state)


async def _state_event_generator(request: Request, tick_seconds: float) -> AsyncIterator[bytes]:
    """Yield ``data: <json>\\n\\n`` frames every ``tick_seconds`` seconds.

    Closes cleanly on client disconnect (``await request.is_disconnected()``).
    Emits ``: ping`` keep-alive comments every ``HEARTBEAT_INTERVAL_S`` if a
    data tick hasn't gone out, and self-terminates with ``event: timeout``
    after ``MAX_STREAM_SECONDS`` so clients reconnect through idle proxies.
    """
    # Comment line: SSE permits ``: text\\n\\n`` as a keep-alive that
    # triggers EventSource ``onopen`` immediately — so the UI flips from
    # "connecting" to "live" even if the first state build takes a couple
    # of seconds (cold cache requires a scanner sweep).
    yield b": connected\n\n"
    start = time.monotonic()
    last_ping = start
    while True:
        if await request.is_disconnected():
            return
        # Deadline check: self-terminate so the client reconnects (browsers
        # auto-resub EventSource on a clean close) and we don't leak
        # generators on long-lived sessions.
        if time.monotonic() - start > MAX_STREAM_SECONDS:
            yield b"event: timeout\ndata: max stream duration reached\n\n"
            return
        try:
            state = await _load_state_or_fallback_async()
            # Trim the heaviest field: scan_log can be 200+ entries on a
            # mature engine. The Scan Log tab only renders the most-recent
            # N; the rest is wire churn. /state (one-shot) still returns
            # the full log so the tab can fetch all on click.
            if state and "scan_log" in state and isinstance(state["scan_log"], list):
                log = state["scan_log"]
                if len(log) > _STREAM_SCAN_LOG_MAX:
                    state = dict(state)
                    state["scan_log"] = log[-_STREAM_SCAN_LOG_MAX:]
                    state["_scan_log_truncated"] = len(log)
            payload = json.dumps(state, default=str)
        except Exception as exc:
            logger.warning("arb: stream tick failed: %s", exc)
            payload = json.dumps({"bot_status": "offline", "opportunities": []})
        yield f"data: {payload}\n\n".encode()
        # A real data frame just went out — defer the next heartbeat.
        last_ping = time.monotonic()
        try:
            await asyncio.sleep(tick_seconds)
        except asyncio.CancelledError:
            return
        # If the next data tick is going to be late (e.g., a slow
        # ``_load_state_or_fallback_async`` cache rebuild), keep the
        # connection warm with a comment-only ping. Cheap and ignored by
        # EventSource clients.
        now = time.monotonic()
        if now - last_ping > HEARTBEAT_INTERVAL_S:
            yield b": ping\n\n"
            last_ping = now


@router.get("/stream", summary="SSE stream of /state every 5s")
async def get_stream(request: Request) -> StreamingResponse:
    """Server-sent events stream pushing the (trimmed) ``/state`` envelope.

    Cadence: every ``PFM_ARB_STREAM_TICK_S`` seconds (default 5.0). The
    payload is the same JSON object as ``GET /strategies/arb/state`` except
    ``scan_log`` is capped to the most-recent ``PFM_ARB_STREAM_SCAN_LOG_MAX``
    entries (default 30) so a single tick stays under ~50 KB on the wire.
    Clients that need the full log should poll ``/state`` directly.
    """
    return StreamingResponse(
        _state_event_generator(request, _STREAM_TICK_SECONDS),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Pending discoveries (HIGH-confidence matches awaiting manual review)
# ---------------------------------------------------------------------------
# auto_discover_loop.py writes new HIGH discoveries to pending_discoveries.json
# instead of merging into the live config. The user reviews them in the UI
# and clicks Accept (move to live config) or Reject (blacklist).
_PENDING_FILE = _ARB_DIR / "pending_discoveries.json"
_REJECTED_FILE = _ARB_DIR / "rejected_discoveries.json"
_LIVE_CONFIG = _ARB_DIR / "markets_config.json"


@router.get(
    "/pending-discoveries",
    summary="List HIGH-confidence discoveries awaiting manual review",
)
def get_pending_discoveries() -> dict[str, Any]:
    """Return the pending discoveries queue (HIGH matches not yet in live config)."""
    items = _safe_read_json(_PENDING_FILE) or []
    return {
        "count": len(items),
        "items": items,
    }


class _DiscoveryAction(BaseModel):
    kalshi_ticker: str
    poly_slug: str | None = None


@router.post(
    "/pending-discoveries/accept",
    summary="Move a pending discovery into the live engine config",
)
def accept_discovery(body: _DiscoveryAction) -> dict[str, Any]:
    """Move the matching item from pending → live config (markets_config.json)."""
    pending = _safe_read_json(_PENDING_FILE) or []
    matched, remaining = [], []
    for item in pending:
        ticker = item.get("kalshi_event_ticker") or item.get("kalshi_ticker")
        slug = item.get("poly_slug")
        if ticker == body.kalshi_ticker and (body.poly_slug is None or slug == body.poly_slug):
            matched.append(item)
        else:
            remaining.append(item)
    if not matched:
        raise HTTPException(
            404, detail=f"no pending discovery with kalshi_ticker={body.kalshi_ticker!r}"
        )
    live = _safe_read_json(_LIVE_CONFIG)
    if not isinstance(live, dict):
        live = {"poll_interval": 240, "threshold": 0.94, "min_alert_profit": 1.0, "events": []}
    live.setdefault("events", []).extend(matched)
    _atomic_write_json(_LIVE_CONFIG, live)
    _atomic_write_json(_PENDING_FILE, remaining)
    return {"accepted": len(matched), "remaining_pending": len(remaining)}


@router.post(
    "/pending-discoveries/reject",
    summary="Drop a pending discovery and remember the rejection",
)
def reject_discovery(body: _DiscoveryAction) -> dict[str, Any]:
    """Remove from pending + append to rejected_discoveries.json so the loop
    never re-queues this Kalshi event."""
    pending = _safe_read_json(_PENDING_FILE) or []
    matched, remaining = [], []
    for item in pending:
        ticker = item.get("kalshi_event_ticker") or item.get("kalshi_ticker")
        slug = item.get("poly_slug")
        if ticker == body.kalshi_ticker and (body.poly_slug is None or slug == body.poly_slug):
            matched.append(item)
        else:
            remaining.append(item)
    if not matched:
        raise HTTPException(
            404, detail=f"no pending discovery with kalshi_ticker={body.kalshi_ticker!r}"
        )
    rejected = _safe_read_json(_REJECTED_FILE) or []
    if not isinstance(rejected, list):
        rejected = []
    rejected.extend(matched)
    _atomic_write_json(_REJECTED_FILE, rejected)
    _atomic_write_json(_PENDING_FILE, remaining)
    return {"rejected": len(matched), "remaining_pending": len(remaining)}


# ---------------------------------------------------------------------------
# New-events surface — newly-listed markets on either venue, last 24h
# ---------------------------------------------------------------------------
# Polls Polymarket gamma + Kalshi events APIs directly with newest-first
# ordering. Returns the raw freshest markets per venue (no matching) so the
# user can spot MM-seed opportunities visually: a fresh market with deep
# liquidity but zero volume is the classic MM-seed signal.


@router.get(
    "/new-events",
    summary="Newest Kalshi and Polymarket markets (last 24h) for MM-seed scouting",
)
def get_new_events(hours: int = 24, limit: int = 30, fetch_book: int = 1) -> dict[str, Any]:
    """Return up to ``limit`` newest markets per venue listed within ``hours``.

    For each Poly market, hits the CLOB orderbook directly to compute the
    real MM-seed signal — gamma's ``liquidity`` field is unreliable. We tag
    a market as MM-seed when its best bid OR ask has a single large round-
    priced level (≥ $1,000 worth at one price). Set ``fetch_book=0`` to skip
    the per-market CLOB call if you only want the metadata.
    """
    from datetime import datetime, timedelta

    cutoff = datetime.now(UTC) - timedelta(hours=max(1, min(hours, 168)))

    poly_items: list[dict[str, Any]] = []
    kalshi_items: list[dict[str, Any]] = []

    # Poly via gamma. Pull markets directly (not events) — markets have
    # liquidity + volume fields we need for the MM-seed signal.
    try:
        r = httpx.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "active": "true",
                "closed": "false",
                "order": "startDate",
                "ascending": "false",
                "limit": max(50, limit * 3),
            },
            timeout=8.0,
        )
        r.raise_for_status()
        raw_markets = r.json() or []
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("new-events: poly fetch failed: %s", e)
        raw_markets = []

    # Filter to window first to bound the CLOB fan-out.
    filtered: list[dict[str, Any]] = []
    for m in raw_markets:
        sd = m.get("startDate") or m.get("createdAt") or ""
        try:
            ts = datetime.fromisoformat(sd.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if ts < cutoff:
            continue
        filtered.append(m)
        if len(filtered) >= limit * 2:
            break

    for m in filtered:
        sd = m.get("startDate") or m.get("createdAt") or ""
        try:
            liq = float(m.get("liquidity") or 0)
            vol24 = float(m.get("volume24hr") or 0)
        except (TypeError, ValueError):
            liq, vol24 = 0.0, 0.0
        # Parse token ids (gamma returns clobTokenIds as a JSON-string).
        token_id = None
        try:
            tids = m.get("clobTokenIds")
            if isinstance(tids, str):
                tids = json.loads(tids)
            if isinstance(tids, list) and tids:
                token_id = str(tids[0])
        except (json.JSONDecodeError, ValueError):
            pass

        # Optional: pull real orderbook to detect MM-seed pattern.
        mm_signal = False
        bid_size = ask_size = 0.0
        bid_px = ask_px = None
        if fetch_book and token_id:
            try:
                rb = httpx.get(
                    "https://clob.polymarket.com/book",
                    params={"token_id": token_id},
                    timeout=3.0,
                )
                rb.raise_for_status()
                book = rb.json() or {}
                bids = book.get("bids") or []
                asks = book.get("asks") or []
                if bids:
                    bid_px = float(bids[0].get("price", 0))
                    bid_size = float(bids[0].get("size", 0))
                if asks:
                    ask_px = float(asks[0].get("price", 0))
                    ask_size = float(asks[0].get("size", 0))
                # MM-seed signatures (any of three patterns ⇒ flagged):
                #   A) Penny spread: bid≤0.02 AND ask≥0.98 with ≥1000 size
                #      (classic "I'll take any flow at penny prices" bot).
                #   B) Round-cent seed: bid OR ask is at a 5¢ increment
                #      (0.05/0.10/.../0.95) with ≥500 size at that level.
                #   C) Wide-symmetric: bid+ask ≈ 1.00 ±0.05 with both sides
                #      ≥500 size (single MM quoting both sides, no real flow).
                penny_spread = (
                    bid_px is not None
                    and ask_px is not None
                    and bid_px <= 0.02
                    and ask_px >= 0.98
                    and (bid_size >= 1000 or ask_size >= 1000)
                )

                def _round_5c(p):
                    return p is not None and abs(p * 20 - round(p * 20)) < 0.01

                round_seed = (_round_5c(bid_px) and bid_size >= 500) or (
                    _round_5c(ask_px) and ask_size >= 500
                )
                wide_sym = (
                    bid_px is not None
                    and ask_px is not None
                    and abs((bid_px + ask_px) - 1.0) <= 0.05
                    and bid_size >= 500
                    and ask_size >= 500
                )
                low_vol = vol24 <= 500
                mm_signal = low_vol and (penny_spread or round_seed or wide_sym)
                # Surface which pattern fired so the UI can label it.
                mm_pattern = (
                    "penny"
                    if penny_spread
                    else "round-5c"
                    if round_seed
                    else "wide-sym"
                    if wide_sym
                    else None
                )
            except (httpx.HTTPError, ValueError):
                pass

        poly_items.append(
            {
                "slug": m.get("slug"),
                "question": m.get("question"),
                "start_date": sd,
                "liquidity_gamma": liq,
                "volume_24h": vol24,
                "token_id": token_id,  # full string — clients need it to hit CLOB directly
                "best_bid": bid_px,
                "bid_size": bid_size,
                "best_ask": ask_px,
                "ask_size": ask_size,
                "mm_seed_signal": mm_signal,
                "mm_pattern": mm_pattern if fetch_book and token_id else None,
            }
        )
        if len(poly_items) >= limit:
            break

    # Kalshi via public events endpoint.
    try:
        r = httpx.get(
            "https://api.elections.kalshi.com/trade-api/v2/events",
            params={"status": "open", "limit": max(50, limit * 3)},
            timeout=8.0,
        )
        r.raise_for_status()
        for ev in (r.json() or {}).get("events", []):
            # Kalshi doesn't expose creation date reliably at the event
            # level — surface all newest open ones and let the UI render.
            kalshi_items.append(
                {
                    "event_ticker": ev.get("event_ticker"),
                    "title": ev.get("title"),
                    "category": ev.get("category"),
                    "sub_title": ev.get("sub_title"),
                    "expected_expiration_date": ev.get("expected_expiration_date"),
                    "n_markets": len(ev.get("markets", []) or []),
                }
            )
            if len(kalshi_items) >= limit:
                break
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("new-events: kalshi fetch failed: %s", e)

    return {
        "window_hours": hours,
        "polymarket": {
            "count": len(poly_items),
            "mm_seed_count": sum(1 for p in poly_items if p["mm_seed_signal"]),
            "items": poly_items,
        },
        "kalshi": {
            "count": len(kalshi_items),
            "items": kalshi_items,
        },
    }


# ───────────────────────────────────────────────────────────────────────────
# Discovery pipeline — unlimited, resumable, newest-first market discovery.
# Crawls the full Kalshi+Polymarket universe step-by-step (checkpointed), matches
# cross-venue with the strict score_match gates, and persists every market where
# an arb is seen into a durable store. Seams are lazy-imported + mockable.
# ───────────────────────────────────────────────────────────────────────────
_CHECKPOINT_PATH = _ARB_DIR / "crawl_state.json"


def _discovery_store() -> Any:
    from pfm.arb.discovery_pipeline import default_store

    return default_store()


def _run_discovery_step(**kwargs: Any) -> Any:
    from pfm.arb.discovery_pipeline import run_discovery_step

    kwargs.setdefault("checkpoint_path", str(_CHECKPOINT_PATH))
    return run_discovery_step(**kwargs)


def _make_price_fn() -> Any:
    """Fee-aware live price function for arb detection (mockable seam)."""
    from pfm.arb.live_pricing import make_price_fn

    return make_price_fn(fee_aware=True)


@router.get("/discovery/confirmed", summary="Durable store of markets where an arb was seen")
def get_confirmed_arbs(
    min_count: int = Query(1, ge=1, le=1000, description="Min sightings to count as confirmed."),
    limit: int = Query(200, ge=1, le=2000),
) -> dict[str, Any]:
    """Return the persistent set of cross-venue pairs where an arb was observed."""
    store = _discovery_store()
    items = [c.to_dict() for c in store.confirmed(min_count=min_count)][:limit]
    return {"count": len(items), "stats": store.stats(), "items": items}


@router.get("/discovery/status", summary="Discovery crawl checkpoint + store stats")
def get_discovery_status() -> dict[str, Any]:
    """Report where the resumable crawl is and how many arbs have been seen."""
    from pfm.arb.market_crawler import load_checkpoint

    ckpt = load_checkpoint(str(_CHECKPOINT_PATH))
    return {
        "checkpoint": {
            "kalshi_cursor": ckpt.kalshi_cursor,
            "poly_offset": ckpt.poly_offset,
            "last_seen_poly_start_iso": ckpt.last_seen_poly_start_iso,
        },
        "store": _discovery_store().stats(),
    }


@router.post("/discovery/step", summary="Run one discovery step (crawl → match → store)")
def run_discovery(
    _admin: None = Depends(require_admin),
    mode: str = Query(
        "new",
        pattern="^(new|sweep|liquid)$",
        description="'new' = fresh markets; 'sweep' = resumable full crawl; 'liquid' = volume-sorted universe.",
    ),
    max_pages: int = Query(3, ge=1, le=50, description="Pages per venue this step (step-by-step)."),
    within_hours: float = Query(
        24.0, ge=1.0, le=336.0, description="Freshness window for mode=new."
    ),
    min_score: float = Query(
        0.5, ge=0.0, le=1.0, description="Min match score (score_match gated)."
    ),
    detect: bool = Query(
        False,
        description="Price-check matched pairs live (fee-aware) and record real arbs to the store.",
    ),
) -> dict[str, Any]:
    """Run ONE bounded discovery step. 'sweep' advances a checkpoint so repeated
    calls walk the whole universe without re-scanning. With detect=true, matched
    pairs are price-checked live (fees included) and any real arb is persisted to
    the durable confirmed-arb store. Reading prices ≠ trading — no orders are sent."""
    try:
        res = _run_discovery_step(
            mode=mode,
            store=_discovery_store(),
            max_pages=max_pages,
            within_hours=within_hours,
            min_score=min_score,
            price_fn=_make_price_fn() if detect else None,
        )
    except Exception as exc:
        logger.exception("discovery step failed")
        raise HTTPException(status_code=502, detail=f"discovery step failed: {exc}") from exc
    return res.as_dict()


# Recall-first live candidate cache (per mode): a discovery step is run on
# demand and cached so the UI "Active discovery" panel shows ALL matched
# cross-venue pairs — verified AND review — without hiding anything.
#   mode="new"    → newly-listed events on each venue matched vs the other
#                   venue's broad universe (each candidate tagged ``new_side``).
#   mode="liquid" → the volume-sorted liquid universe of both venues.
_CANDIDATES_CACHE: dict[str, dict[str, Any]] = {}
_CANDIDATES_TTL_S = 300.0


@router.get("/discovery/candidates", summary="Recall-first live cross-venue candidates")
def get_discovery_candidates(
    mode: str = Query(
        "new",
        pattern="^(new|liquid)$",
        description="'new' = possible arbs from newly-listed events; 'liquid' = volume-sorted universe.",
    ),
    refresh: bool = Query(False, description="Force a fresh step instead of the cache."),
    max_pages: int = Query(3, ge=1, le=8, description="Pages per venue this step."),
    within_hours: float = Query(
        72.0, ge=1.0, le=336.0, description="Freshness window for mode=new."
    ),
) -> dict[str, Any]:
    """Return matched cross-venue pairs for the data hub (no trade is executed).

    Recall-first: every candidate above the recall floor is returned, each tagged
    with ``confidence`` ('verified' | 'review'), ``tier`` and (for mode=new)
    ``new_side`` ('kalshi' | 'poly' | 'both') — nothing is hidden. A bounded step
    is run on demand and cached per-mode for 5 min (served stale on failure) so the
    panel stays responsive without hammering the venues.
    """
    now = time.time()
    cached = _CANDIDATES_CACHE.get(mode)
    if not refresh and cached is not None and (now - cached["ts"]) < _CANDIDATES_TTL_S:
        return {**cached["data"], "cached": True, "age_s": round(now - cached["ts"], 1)}
    try:
        res = _run_discovery_step(
            mode=mode,
            store=_discovery_store(),
            max_pages=max_pages,
            within_hours=within_hours,
            min_score=0.5,
            price_fn=None,
        )
    except Exception as exc:
        logger.warning("discovery candidates step (mode=%s) failed: %s", mode, exc)
        if cached is not None:
            return {
                **cached["data"],
                "cached": True,
                "stale": True,
                "age_s": round(now - cached["ts"], 1),
            }
        raise HTTPException(status_code=502, detail=f"discovery candidates failed: {exc}") from exc
    cands = sorted(res.candidates, key=lambda c: c.get("score", 0.0), reverse=True)
    payload = {
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "n_kalshi": res.n_kalshi,
        "n_poly": res.n_poly,
        "n_candidates": res.n_candidates,
        "n_high": res.n_high,
        "n_review": getattr(res, "n_review", None),
        "candidates": cands[:200],
        "cached": False,
        "age_s": 0.0,
    }
    _CANDIDATES_CACHE[mode] = {"ts": now, "data": payload}
    return payload


__all__ = ["router"]
