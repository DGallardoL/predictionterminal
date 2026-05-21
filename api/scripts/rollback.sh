#!/usr/bin/env bash
# rollback.sh — revert the pfm API to a previously captured backup.
#
# Pipeline:
#   1. Stop gunicorn (graceful: SIGTERM to master, then SIGKILL after grace).
#   2. Run api/scripts/restore.sh against the supplied backup archive.
#   3. Restart gunicorn using the same command-line that the prior master used,
#      or fall back to ${PFM_GUNICORN_CMD} if no prior process is detectable.
#   4. Verify GET /health returns "ok".
#   5. Append outcome to .coordination/deploys.log.
#
# Usage:
#   bash api/scripts/rollback.sh <path-to-pfm-backup.tar.gz>
#
# Exit codes:
#   0 — rollback + health verification succeeded
#   1 — bad arguments / missing backup
#   2 — restore.sh failed
#   3 — failed to restart gunicorn
#   4 — health check failed after restart

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COORD_DIR="${REPO_ROOT}/.coordination"
DEPLOY_LOG="${COORD_DIR}/deploys.log"

mkdir -p "${COORD_DIR}"

TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log() {
  local line="$1"
  echo "[rollback.sh ${TS_UTC}] ${line}"
  echo "${TS_UTC} rollback ${line}" >>"${DEPLOY_LOG}"
}

# --- 0. Validate args ------------------------------------------------------
if [[ $# -lt 1 ]]; then
  log "FAIL missing backup path"
  echo "usage: bash api/scripts/rollback.sh <pfm-backup.tar.gz>" >&2
  exit 1
fi

BACKUP_PATH="$1"
if [[ ! -f "${BACKUP_PATH}" ]]; then
  log "FAIL backup not found: ${BACKUP_PATH}"
  exit 1
fi
if ! tar -tzf "${BACKUP_PATH}" >/dev/null 2>&1; then
  log "FAIL not a readable gzipped tar: ${BACKUP_PATH}"
  exit 1
fi
log "step=start backup=${BACKUP_PATH}"

if [[ ! -f "${SCRIPT_DIR}/restore.sh" ]]; then
  log "FAIL restore.sh not present at ${SCRIPT_DIR}/restore.sh"
  exit 2
fi

# --- 1. Capture gunicorn invocation BEFORE killing it ----------------------
GUNICORN_PIDS="$(pgrep -f gunicorn || true)"
PRIOR_CMD=""
PRIOR_CWD=""
if [[ -n "${GUNICORN_PIDS}" ]]; then
  MASTER_PID=""
  for pid in ${GUNICORN_PIDS}; do
    if [[ -z "${MASTER_PID}" || "${pid}" -lt "${MASTER_PID}" ]]; then
      MASTER_PID="${pid}"
    fi
  done
  if [[ -n "${MASTER_PID}" ]]; then
    PRIOR_CMD="$(ps -o command= -p "${MASTER_PID}" 2>/dev/null || true)"
    # macOS lsof to capture cwd; best-effort.
    PRIOR_CWD="$(lsof -p "${MASTER_PID}" 2>/dev/null | awk '$4=="cwd"{print $9; exit}' || true)"
    log "captured master_pid=${MASTER_PID} cmd=${PRIOR_CMD:-<unknown>} cwd=${PRIOR_CWD:-<unknown>}"
  fi
fi

# --- 2. Stop gunicorn ------------------------------------------------------
if [[ -n "${GUNICORN_PIDS}" ]]; then
  log "step=stop pids=${GUNICORN_PIDS}"
  # SIGTERM to master only; workers will exit when master does.
  if [[ -n "${MASTER_PID:-}" ]]; then
    kill -TERM "${MASTER_PID}" 2>/dev/null || true
  else
    # Fallback: TERM all matched.
    for pid in ${GUNICORN_PIDS}; do
      kill -TERM "${pid}" 2>/dev/null || true
    done
  fi

  # Wait up to 10s for clean shutdown.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! pgrep -f gunicorn >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  # If anything survived, SIGKILL the stragglers.
  STRAGGLERS="$(pgrep -f gunicorn || true)"
  if [[ -n "${STRAGGLERS}" ]]; then
    log "step=stop force-killing stragglers=${STRAGGLERS}"
    for pid in ${STRAGGLERS}; do
      kill -KILL "${pid}" 2>/dev/null || true
    done
    sleep 1
  fi
  log "ok gunicorn stopped"
else
  log "warn no running gunicorn to stop"
fi

# --- 3. Restore from backup ------------------------------------------------
log "step=restore backup=${BACKUP_PATH}"
if ! bash "${SCRIPT_DIR}/restore.sh" "${BACKUP_PATH}"; then
  log "FAIL restore.sh non-zero"
  exit 2
fi
log "ok restore"

# --- 4. Restart gunicorn ---------------------------------------------------
# Strategy: prefer the captured command; otherwise honour ${PFM_GUNICORN_CMD};
# otherwise fall back to a sensible default that matches the project layout.
RESTART_CMD=""
RESTART_CWD="${PRIOR_CWD:-${REPO_ROOT}/api}"

if [[ -n "${PRIOR_CMD}" ]]; then
  RESTART_CMD="${PRIOR_CMD}"
elif [[ -n "${PFM_GUNICORN_CMD:-}" ]]; then
  RESTART_CMD="${PFM_GUNICORN_CMD}"
else
  # Best-effort default. Uses the project venv if available.
  if [[ -x "${REPO_ROOT}/api/.venv/bin/gunicorn" ]]; then
    RESTART_CMD="${REPO_ROOT}/api/.venv/bin/gunicorn pfm.main:app --bind 0.0.0.0:8000 --workers 2 --worker-class uvicorn.workers.UvicornWorker --daemon"
  else
    RESTART_CMD="gunicorn pfm.main:app --bind 0.0.0.0:8000 --workers 2 --worker-class uvicorn.workers.UvicornWorker --daemon"
  fi
fi

log "step=restart cwd=${RESTART_CWD} cmd=${RESTART_CMD}"

# Ensure PYTHONPATH points at src so `pfm` is importable when starting from api/.
export PYTHONPATH="${REPO_ROOT}/api/src:${PYTHONPATH:-}"

# Use a subshell so cd does not contaminate the calling shell.
if ! (cd "${RESTART_CWD}" && eval "${RESTART_CMD}"); then
  log "FAIL gunicorn restart"
  exit 3
fi

# Give the new master a moment to bind, fork workers, and load the app.
sleep 5

if ! pgrep -f gunicorn >/dev/null 2>&1; then
  log "FAIL gunicorn process not visible after restart"
  exit 3
fi
log "ok gunicorn restarted"

# --- 5. Health check -------------------------------------------------------
HEALTH_URL="${PFM_HEALTH_URL:-http://localhost:8000/health}"
log "step=health url=${HEALTH_URL}"

HEALTH_RC=1
HEALTH_BODY=""
for attempt in 1 2 3 4 5; do
  if HEALTH_BODY="$(curl -fsS --max-time 5 "${HEALTH_URL}" 2>/dev/null)"; then
    if echo "${HEALTH_BODY}" | grep -qi '"status"[[:space:]]*:[[:space:]]*"ok"'; then
      HEALTH_RC=0
      break
    fi
  fi
  sleep 2
done

if [[ ${HEALTH_RC} -ne 0 ]]; then
  log "FAIL health url=${HEALTH_URL} body=${HEALTH_BODY:-<empty>}"
  exit 4
fi

log "ok health=${HEALTH_URL} backup=${BACKUP_PATH}"
echo "rollback.sh: SUCCESS  backup=${BACKUP_PATH}"
