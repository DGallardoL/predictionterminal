"""``POST /alerts/configure`` and ``GET /alerts/configure`` — user-level
alert threshold configuration, persisted to a JSON file on disk.

Task W13-15 (wave-13). The other alert-shaped routers in this package
(:mod:`pfm.alerts.router`, :mod:`pfm.alerts.digest_router`) ship the
*delivery* side of alerts — rules, channels, the digest endpoint. This
small router covers the *threshold* knobs that the three high-traffic
producers (jumps, sentiment-disagree, arb) look up when deciding whether
an event is "interesting enough" to surface:

* ``jump_threshold_pp``        — absolute ∆pp at which a jump becomes
                                  an alert (default 5.0pp).
* ``sentiment_disagree_pct``   — minimum cumulative-disagreement % at
                                  which a sentiment-vs-price divergence
                                  is surfaced (default 40 %).
* ``arb_min_spread_pct``       — minimum cross-venue spread % at which
                                  an arbitrage row is shown in the live
                                  monitor (default 2.0 %).

The full surface:

``POST /alerts/configure`` — body is an :class:`AlertConfig`. The
provided fields are merged on top of the current config (so callers
can patch one knob at a time without re-sending the rest). Validation
rejects negatives and absurd values.

``GET /alerts/configure`` — returns the current effective configuration.
If the persistence file does not exist, the defaults are returned.

Persistence
-----------
The on-disk format is a single JSON document at the path resolved by
:func:`_config_path` (default ``/tmp/pfm-alerts-config.json``; tests
override via the ``PFM_ALERTS_CONFIG_PATH`` env var). Writes are atomic
(``write`` → ``os.replace``) so a concurrent reader never observes a
partially-written file. A module-level :class:`threading.Lock` serialises
in-process writers.

The persistence path lives under ``/tmp`` by design — these are *user*
overrides on demo / dev machines, not a production durable store. If we
ever need fleet-wide config, swap the storage backend in :func:`_load`
/ :func:`_save` and leave the router untouched.

Mounting note
-------------
The router exposes ``router`` at module top level so callers can::

    from pfm.alerts.configure_router import router as _alerts_cfg_router
    app.include_router(_alerts_cfg_router)

Both endpoints share the path ``/alerts/configure``; FastAPI routes by
HTTP verb so this is unambiguous.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["alerts"])

# ─────────────────────────────────────────────────────────────────────────────
# Defaults — the task spec dictates these exact values, do not silently change.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_JUMP_THRESHOLD_PP: float = 5.0
DEFAULT_SENTIMENT_DISAGREE_PCT: float = 40.0
DEFAULT_ARB_MIN_SPREAD_PCT: float = 2.0

# Sanity caps. Anything beyond these is almost certainly a unit mistake.
_MAX_JUMP_THRESHOLD_PP: float = 100.0  # ∆pp ∈ [0, 100]
_MAX_SENTIMENT_DISAGREE_PCT: float = 100.0  # cumulative % ∈ [0, 100]
_MAX_ARB_MIN_SPREAD_PCT: float = 100.0  # spread % ∈ [0, 100]

# Path resolution — see module docstring.
_DEFAULT_CONFIG_PATH = "/tmp/pfm-alerts-config.json"
_ENV_VAR = "PFM_ALERTS_CONFIG_PATH"

# Serialise concurrent writers within this process.
_WRITE_LOCK = threading.Lock()


def _config_path() -> Path:
    """Resolve the on-disk config path, honouring the env-var override."""
    return Path(os.environ.get(_ENV_VAR, _DEFAULT_CONFIG_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────


class AlertConfig(BaseModel):
    """User-tunable alert thresholds.

    All fields are optional on input so callers can PATCH-style update a
    single knob; the response always echoes the full effective config
    with defaults filled in.
    """

    jump_threshold_pp: float | None = Field(
        default=None,
        ge=0.0,
        le=_MAX_JUMP_THRESHOLD_PP,
        description="Absolute ∆pp at which a price jump becomes an alert.",
    )
    sentiment_disagree_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=_MAX_SENTIMENT_DISAGREE_PCT,
        description="Minimum cumulative-disagreement % to flag sentiment-vs-price divergence.",
    )
    arb_min_spread_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=_MAX_ARB_MIN_SPREAD_PCT,
        description="Minimum cross-venue spread % to surface an arbitrage row.",
    )


class AlertConfigResponse(BaseModel):
    """The full effective config (defaults filled in)."""

    jump_threshold_pp: float = Field(..., ge=0.0)
    sentiment_disagree_pct: float = Field(..., ge=0.0)
    arb_min_spread_pct: float = Field(..., ge=0.0)


def _defaults() -> dict[str, float]:
    return {
        "jump_threshold_pp": DEFAULT_JUMP_THRESHOLD_PP,
        "sentiment_disagree_pct": DEFAULT_SENTIMENT_DISAGREE_PCT,
        "arb_min_spread_pct": DEFAULT_ARB_MIN_SPREAD_PCT,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


def _load() -> dict[str, float]:
    """Read the current config from disk, falling back to defaults.

    Corruption / partial writes / missing file → defaults. We do not
    raise: the caller always gets a usable config.
    """
    path = _config_path()
    if not path.exists():
        return _defaults()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return _defaults()
    except (OSError, json.JSONDecodeError):
        return _defaults()

    merged = _defaults()
    # Only accept the three known keys with numeric types; ignore anything else.
    for key in merged:
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if 0.0 <= float(value) <= 100.0:
                merged[key] = float(value)
    return merged


def _save(config: dict[str, float]) -> None:
    """Atomically persist ``config`` to disk.

    Uses ``tempfile.mkstemp`` + ``os.replace`` so readers never observe
    a half-written file. The temp file is created in the same directory
    as the target so the rename is guaranteed to be atomic on POSIX.
    """
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".pfm-alerts-config-",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except Exception:
            # Best-effort cleanup of the temp file before re-raising.
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/alerts/configure", response_model=AlertConfigResponse)
def get_alert_config() -> AlertConfigResponse:
    """Return the current effective alert configuration.

    When no config has ever been written, defaults are returned. The
    response is *always* fully populated — frontends can rely on every
    knob being present.
    """
    cfg = _load()
    return AlertConfigResponse(**cfg)


@router.post("/alerts/configure", response_model=AlertConfigResponse)
def set_alert_config(payload: AlertConfig) -> AlertConfigResponse:
    """Merge ``payload`` into the persisted config and return the new state.

    ``None`` fields in ``payload`` mean "leave this knob alone". A
    request that sets nothing is a no-op that simply returns the
    current effective config.

    Raises ``400`` only if persistence to disk fails — Pydantic catches
    out-of-range numeric values before they reach this function.
    """
    current = _load()
    updated = dict(current)
    incoming = payload.model_dump(exclude_unset=True, exclude_none=True)
    for key, value in incoming.items():
        if key in updated:
            updated[key] = float(value)

    try:
        _save(updated)
    except OSError as exc:  # pragma: no cover — disk error is hard to fake
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist alert config: {exc!s}",
        ) from exc

    return AlertConfigResponse(**updated)


__all__ = [
    "DEFAULT_ARB_MIN_SPREAD_PCT",
    "DEFAULT_JUMP_THRESHOLD_PP",
    "DEFAULT_SENTIMENT_DISAGREE_PCT",
    "AlertConfig",
    "AlertConfigResponse",
    "_config_path",
    "_defaults",
    "_load",
    "_save",
    "router",
]
