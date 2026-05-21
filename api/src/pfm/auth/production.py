"""Production-aware auth defaults: ON-by-default detection + admin-token autogen.

The legacy contract was ``PFM_AUTH_ENABLED=1`` opt-in. That makes it easy to
forget on a real deploy: the service ships with auth OFF and admin endpoints
wide open until somebody remembers to flip the env var. This module flips the
default — on common PaaS targets (Fly.io, Render, generic ``ENV=production``)
auth turns ON automatically, with a strong-randomness admin token generated on
first boot if one wasn't provided.

Public surface
==============

- :func:`is_auth_enabled` — single source of truth for "should auth be on?".
  Replaces ``os.environ.get("PFM_AUTH_ENABLED") == "1"`` in callers.
- :func:`detect_env_reason` — a short string identifying *why* auth is on/off
  (``explicit_on`` / ``explicit_off`` / ``production`` / ``fly`` / ``render`` /
  ``node_production`` / ``off``). Surfaced via ``/health/detail`` so operators
  can audit.
- :func:`get_or_generate_admin_token` — returns the configured admin token,
  or generates+persists a fresh ``sk_admin_…`` token to ``/tmp`` if auth is on
  and no token was supplied.
- :func:`is_admin_token_autogen` — ``True`` iff the active admin token came
  from the autogen file rather than ``PFM_ADMIN_TOKEN``.
- :func:`first_boot_marker_path` / :func:`mark_first_boot_done` — manage the
  one-shot ``/auth/first-boot-info`` flag.

The persisted-token file at ``/tmp/pfm_admin_token.json`` is chmod 0600 so a
co-tenant on the same host can't read it. The file is regenerated on every
restart unless ``PFM_ADMIN_TOKEN`` is set, which is the entire point of the
warning we log: autogen is a sane *default*, not a substitute for proper
secret management.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import stat
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where the autogen admin token is persisted between worker restarts. ``/tmp``
#: is the lowest-common-denominator writable path on Fly.io / Render / Heroku
#: containers; if the user wants durability they should set ``PFM_ADMIN_TOKEN``.
ADMIN_TOKEN_PATH = Path("/tmp/pfm_admin_token.json")

#: One-shot flag for ``GET /auth/first-boot-info``. Removed by hand if the
#: operator wants to re-fetch the token through the endpoint (otherwise they
#: should just read the file).
FIRST_BOOT_FLAG_PATH = Path("/tmp/pfm_first_boot_done.flag")

#: Prefix on autogen admin tokens so logs/dashboards can tell them apart from
#: a hand-set ``PFM_ADMIN_TOKEN``.
ADMIN_TOKEN_PREFIX = "sk_admin_"


# ----------------------------------------------------------------- env detection


def detect_env_reason() -> str:
    """Return a short tag explaining why auth is on or off.

    Order matches :func:`is_auth_enabled`:

    1. ``PFM_AUTH_ENABLED=1`` → ``"explicit_on"``
    2. ``PFM_AUTH_ENABLED=0`` → ``"explicit_off"``
    3. ``ENV=production`` → ``"production"``
    4. ``FLY_APP_NAME`` set → ``"fly"``
    5. ``RENDER`` set → ``"render"``
    6. ``NODE_ENV=production`` → ``"node_production"``
    7. else → ``"off"``
    """
    explicit = os.environ.get("PFM_AUTH_ENABLED")
    if explicit is not None:
        if explicit == "1":
            return "explicit_on"
        if explicit == "0":
            return "explicit_off"
    if os.environ.get("ENV", "").strip().lower() == "production":
        return "production"
    if os.environ.get("FLY_APP_NAME", "").strip():
        return "fly"
    if os.environ.get("RENDER", "").strip():
        return "render"
    if os.environ.get("NODE_ENV", "").strip().lower() == "production":
        return "node_production"
    return "off"


def is_auth_enabled() -> bool:
    """Single source of truth for "should auth gate requests?".

    Behaviour:

    - ``PFM_AUTH_ENABLED=1`` → True (explicit override wins)
    - ``PFM_AUTH_ENABLED=0`` → False (explicit override wins, even in prod)
    - any of ``ENV=production`` / ``FLY_APP_NAME`` / ``RENDER`` /
      ``NODE_ENV=production`` → True
    - else → False (dev default; tests + local curls keep working)
    """
    reason = detect_env_reason()
    return reason in {"explicit_on", "production", "fly", "render", "node_production"}


# ------------------------------------------------------------- admin token autogen


def _persist_token(token: str) -> None:
    """Write ``token`` to :data:`ADMIN_TOKEN_PATH` with 0600 permissions.

    We write+chmod in two steps because ``Path.write_text`` doesn't take a
    mode arg, and we want the file to be unreadable by group/other from the
    moment it appears (some filesystems honour the chmod even between create
    and our explicit call, but most don't — close the gap).
    """
    payload = {
        "token": token,
        "generated_at_iso": datetime.now(UTC).isoformat(),
        "warning": "regenerated each restart unless PFM_ADMIN_TOKEN is set",
    }
    ADMIN_TOKEN_PATH.write_text(json.dumps(payload, indent=2))
    try:
        ADMIN_TOKEN_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        # Some FUSE / Docker volumes refuse chmod; we still persisted the
        # token so the service can run, but warn loudly.
        logger.warning(
            "could not chmod 0600 on %s — verify the host filesystem permissions",
            ADMIN_TOKEN_PATH,
        )


def _read_persisted_token() -> str | None:
    """Return the previously-persisted autogen token, or ``None`` if missing.

    Validates the JSON shape; a corrupt file is treated as absent so the next
    boot regenerates rather than crashing.
    """
    if not ADMIN_TOKEN_PATH.exists():
        return None
    try:
        data = json.loads(ADMIN_TOKEN_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    tok = data.get("token") if isinstance(data, dict) else None
    return tok if isinstance(tok, str) and tok else None


def _generate_token() -> str:
    """Return a freshly-minted ``sk_admin_<urlsafe-32>`` token."""
    return f"{ADMIN_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"


def get_or_generate_admin_token() -> str:
    """Return the active admin token, creating one on first boot if needed.

    Resolution order:

    1. ``PFM_ADMIN_TOKEN`` set → return it verbatim. No autogen, no file write.
    2. Auth disabled → return ``""``. Admin endpoints stay closed because
       :func:`pfm.auth.dependencies.require_admin` already fails-closed when
       no token is configured.
    3. Auth enabled, autogen file exists → reuse the persisted token so a
       worker restart inside the same container doesn't churn the secret out
       from under any operator who has it pinned in their terminal.
    4. Auth enabled, no env, no file → mint a fresh token, persist it
       (0600), log a WARNING that includes the token, return it.
    """
    explicit = os.environ.get("PFM_ADMIN_TOKEN", "").strip()
    if explicit:
        return explicit
    if not is_auth_enabled():
        return ""
    persisted = _read_persisted_token()
    if persisted:
        return persisted
    token = _generate_token()
    _persist_token(token)
    logger.warning(
        "Generated admin token: %s. Set PFM_ADMIN_TOKEN to persist across restarts.",
        token,
    )
    return token


def is_admin_token_autogen() -> bool:
    """``True`` iff the *active* admin token came from autogen, not env var.

    Used to populate ``/health/detail`` so operators can spot a forgotten
    ``PFM_ADMIN_TOKEN``. Cheap: doesn't reach into the file unless auth is on
    and no env var is set.
    """
    if os.environ.get("PFM_ADMIN_TOKEN", "").strip():
        return False
    if not is_auth_enabled():
        return False
    return _read_persisted_token() is not None


def admin_token_configured() -> bool:
    """``True`` iff *any* admin token will validate (env or autogen file).

    Mirrors :func:`pfm.auth.dependencies.require_admin`'s fail-closed contract
    from the outside, without leaking the token itself.
    """
    if os.environ.get("PFM_ADMIN_TOKEN", "").strip():
        return True
    if not is_auth_enabled():
        return False
    return _read_persisted_token() is not None


# ----------------------------------------------------------- first-boot endpoint


def first_boot_marker_path() -> Path:
    """Filesystem location of the one-shot marker. Exposed for tests."""
    return FIRST_BOOT_FLAG_PATH


def first_boot_done() -> bool:
    return FIRST_BOOT_FLAG_PATH.exists()


def mark_first_boot_done() -> None:
    """Create the one-shot flag, idempotent and crash-tolerant.

    A failed write isn't fatal — the worst case is the endpoint stays
    available one more call, which is the conservative direction (operator
    still gets the token).
    """
    try:
        FIRST_BOOT_FLAG_PATH.write_text(datetime.now(UTC).isoformat())
        with contextlib.suppress(OSError):
            FIRST_BOOT_FLAG_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:
        logger.warning("could not write first-boot flag at %s: %s", FIRST_BOOT_FLAG_PATH, exc)
