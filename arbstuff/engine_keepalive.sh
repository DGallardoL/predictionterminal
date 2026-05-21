#!/bin/bash
# Keep-alive wrapper for the arb engine (laptop use).
# Runs the engine in REST/OG mode and restarts it if it ever exits
# (Mac sleep, network blip, singleton-lock takeover, crash). A PID lockfile
# stops a second keep-alive from running. Stop with: kill $(cat arbstuff/engine_keepalive.lock)
cd "$(dirname "$0")" || exit 1
LOCK="engine_keepalive.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "keep-alive already running (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"; kill 0' EXIT INT TERM

export PFM_ARB_NO_DISCOVERY=1
export PFM_ARB_POLL_INTERVAL=60
while true; do
  echo "$(date '+%H:%M:%S') [keepalive] starting arb_engine.py --mode og" >> /tmp/arb_engine.log
  ../api/.venv/bin/python arb_engine.py --mode og >> /tmp/arb_engine.log 2>&1
  echo "$(date '+%H:%M:%S') [keepalive] engine exited — restarting in 5s" >> /tmp/arb_engine.log
  sleep 5
done
