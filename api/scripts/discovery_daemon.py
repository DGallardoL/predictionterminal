#!/usr/bin/env python
"""Standalone background discovery daemon (safe, decoupled).

Runs the cross-venue arb discovery loop OUTSIDE the web server so a bad cycle
can never take down gunicorn. It rotates ``liquid`` / ``new`` / ``sweep`` steps
on a generous interval, price-checks matched pairs fee-aware, and records any
*verified* arb to the durable confirmed store that the API reads. The API's
``/strategies/arb/discovery/{status,confirmed}`` endpoints surface the results
with no need to open the UI Discovery tab.

Safety properties (this is what "no haga cosas a cada rato y truene todo" means):

* **Single instance** — a PID lockfile prevents a second daemon from running.
* **Generous cadence** — ``PFM_ARB_DISCOVERY_INTERVAL_S`` (default 600 s = 10 min),
  hard floor of 120 s so it can never busy-loop the venues.
* **Bounded steps** — ``max_pages`` small (2), so each step is a few seconds.
* **Never dies** — every tick is wrapped; on repeated failures the sleep backs
  off (up to 1 h) instead of hammering a down venue.
* **Path-aligned** — writes the SAME store/checkpoint files the running API
  reads (store relative to the api/ CWD; checkpoint at the repo-root arbstuff/).
* **Clean shutdown** — SIGTERM/SIGINT release the lock and exit.

Usage (from the api/ directory, so the relative store path matches the API)::

    cd api && PYTHONPATH=src .venv/bin/python scripts/discovery_daemon.py

Stop it with ``kill <pid>`` (the PID is in the lockfile / printed at startup).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

# Make ``pfm`` importable when run as a plain script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pfm.arb.discovery_pipeline import default_store, run_discovery_step
from pfm.arb.live_pricing import make_price_fn

# --- Paths (aligned with the running API) ----------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_PATH = str(_REPO_ROOT / "arbstuff" / "crawl_state.json")
_LOCKFILE = _REPO_ROOT / "arbstuff" / "discovery_daemon.lock"

# --- Tunables (safe defaults; override via env) -----------------------------
_INTERVAL_S = max(120.0, float(os.environ.get("PFM_ARB_DISCOVERY_INTERVAL_S", "600")))
_MAX_PAGES = max(1, int(os.environ.get("PFM_ARB_DISCOVERY_MAX_PAGES", "2")))
_SWEEP_EVERY = max(1, int(os.environ.get("PFM_ARB_DISCOVERY_SWEEP_EVERY", "6")))
_LIQUID_EVERY = max(1, int(os.environ.get("PFM_ARB_DISCOVERY_LIQUID_EVERY", "3")))
_WITHIN_HOURS = float(os.environ.get("PFM_ARB_DISCOVERY_WINDOW_H", "48"))
_STARTUP_DELAY_S = float(os.environ.get("PFM_ARB_DISCOVERY_STARTUP_DELAY_S", "10"))
_MAX_BACKOFF_S = 3600.0

log = logging.getLogger("discovery_daemon")

_RUNNING = True


def _handle_signal(signum: int, _frame: object) -> None:
    global _RUNNING
    log.info("received signal %s — shutting down after current sleep", signum)
    _RUNNING = False


def _acquire_lock() -> bool:
    """Refuse to start if another live daemon holds the lock."""
    try:
        if _LOCKFILE.exists():
            old = _LOCKFILE.read_text().strip()
            if old.isdigit() and _pid_alive(int(old)):
                log.error("another discovery daemon is already running (pid %s) — exiting", old)
                return False
        _LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
        _LOCKFILE.write_text(str(os.getpid()))
        return True
    except OSError as exc:
        log.error("could not acquire lock: %s", exc)
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _release_lock() -> None:
    try:
        if _LOCKFILE.exists() and _LOCKFILE.read_text().strip() == str(os.getpid()):
            _LOCKFILE.unlink()
    except OSError:
        pass


def _mode_for_tick(tick: int) -> str:
    """Rotate modes: liquid often (productive, runs first), sweep at cycle end, else new.

    ``sweep`` fires at the *end* of each ``_SWEEP_EVERY`` cycle (not tick 0) so the
    very first tick is the productive ``liquid`` scan rather than the starved sweep.
    """
    if tick % _SWEEP_EVERY == _SWEEP_EVERY - 1:
        return "sweep"
    if tick % _LIQUID_EVERY == 0:
        return "liquid"
    return "new"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if not _acquire_lock():
        return 1
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info(
        "discovery daemon up (pid %s) | interval=%.0fs max_pages=%d "
        "sweep_every=%d liquid_every=%d | store=%s checkpoint=%s",
        os.getpid(),
        _INTERVAL_S,
        _MAX_PAGES,
        _SWEEP_EVERY,
        _LIQUID_EVERY,
        default_store().path,
        _CHECKPOINT_PATH,
    )
    time.sleep(_STARTUP_DELAY_S)

    tick = 0
    consecutive_failures = 0
    while _RUNNING:
        mode = _mode_for_tick(tick)
        t0 = time.time()
        try:
            res = run_discovery_step(
                mode=mode,
                store=default_store(),
                max_pages=_MAX_PAGES,
                within_hours=_WITHIN_HOURS,
                min_score=0.5,
                price_fn=make_price_fn(fee_aware=True),
                checkpoint_path=_CHECKPOINT_PATH,
            )
            consecutive_failures = 0
            log.info(
                "[%s] tick=%d (%.1fs): %dK x %dP -> %d cand, %d high, %d review, %d recorded",
                res.mode,
                tick,
                time.time() - t0,
                res.n_kalshi,
                res.n_poly,
                res.n_candidates,
                res.n_high,
                getattr(res, "n_review", 0),
                res.n_recorded,
            )
        except Exception as exc:  # never let a bad cycle kill the daemon
            consecutive_failures += 1
            log.warning(
                "[%s] tick=%d failed (%d in a row): %s", mode, tick, consecutive_failures, exc
            )

        tick += 1
        # Back off on repeated failures so we never hammer a down venue.
        sleep_s = _INTERVAL_S
        if consecutive_failures:
            sleep_s = min(_MAX_BACKOFF_S, _INTERVAL_S * (2**consecutive_failures))
        # Sleep in short slices so signals are handled promptly.
        slept = 0.0
        while _RUNNING and slept < sleep_s:
            time.sleep(min(2.0, sleep_s - slept))
            slept += 2.0

    _release_lock()
    log.info("discovery daemon stopped cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
