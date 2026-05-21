#!/usr/bin/env bash
# deploy.sh — safe production deploy for the pfm API.
#
# Pipeline:
#   1. Snapshot critical state via api/scripts/backup.sh.
#   2. Run the fast pytest tier (skip slow markers).
#   3. On green tests, tag the commit `deploy-<unix-ts>` (LOCAL ONLY, do not push).
#   4. Gracefully reload gunicorn via SIGUSR1 to the master PID.
#   5. Wait up to ~15s and verify GET /health returns "ok".
#   6. On any post-tag failure, invoke rollback.sh with the pre-deploy backup.
#
# Usage:
#   bash api/scripts/deploy.sh
#
# Exit codes:
#   0 — deploy + health verification succeeded
#   1 — backup failed
#   2 — tests failed (no tag created, no restart attempted)
#   3 — git tag failed
#   4 — could not locate gunicorn master process
#   5 — gunicorn reload signal failed
#   6 — health check failed after reload (rollback attempted)
#   7 — rollback itself failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COORD_DIR="${REPO_ROOT}/.coordination"
DEPLOY_LOG="${COORD_DIR}/deploys.log"

mkdir -p "${COORD_DIR}"

TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
UNIX_TS="$(date +%s)"

log() {
  local line="$1"
  echo "[deploy.sh ${TS_UTC}] ${line}"
  echo "${TS_UTC} deploy ${line}" >>"${DEPLOY_LOG}"
}

# Pre-condition checks.
if [[ ! -x "${SCRIPT_DIR}/backup.sh" && ! -f "${SCRIPT_DIR}/backup.sh" ]]; then
  log "FATAL backup.sh not found at ${SCRIPT_DIR}/backup.sh"
  exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/rollback.sh" ]]; then
  log "FATAL rollback.sh not found at ${SCRIPT_DIR}/rollback.sh"
  exit 1
fi

# --- 1. Backup -------------------------------------------------------------
BACKUP_OUT="/tmp/pfm-predeploy-${UNIX_TS}.tar.gz"
log "step=backup target=${BACKUP_OUT}"
if ! bash "${SCRIPT_DIR}/backup.sh" "${BACKUP_OUT}"; then
  log "FAIL backup.sh returned non-zero"
  exit 1
fi
if [[ ! -f "${BACKUP_OUT}" ]]; then
  log "FAIL backup file not present at ${BACKUP_OUT}"
  exit 1
fi
log "ok backup=${BACKUP_OUT}"

# --- 2. Tests --------------------------------------------------------------
API_DIR="${REPO_ROOT}/api"
PYTEST_BIN=""
if [[ -x "${API_DIR}/.venv/bin/pytest" ]]; then
  PYTEST_BIN="${API_DIR}/.venv/bin/pytest"
elif command -v pytest >/dev/null 2>&1; then
  PYTEST_BIN="$(command -v pytest)"
else
  log "FAIL pytest not found (looked in ${API_DIR}/.venv and PATH)"
  exit 2
fi

log "step=tests bin=${PYTEST_BIN}"
pushd "${API_DIR}" >/dev/null
# Don't let `set -e` abort us here; we want to inspect the exit code.
set +e
PYTHONPATH=src "${PYTEST_BIN}" -q -m "not slow"
TEST_RC=$?
set -e
popd >/dev/null

if [[ ${TEST_RC} -ne 0 ]]; then
  log "FAIL tests rc=${TEST_RC} backup=${BACKUP_OUT}"
  exit 2
fi
log "ok tests"

# --- 3. Tag (local) --------------------------------------------------------
TAG_NAME="deploy-${UNIX_TS}"
log "step=tag name=${TAG_NAME}"
if ! git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
  log "WARN not a git repo; skipping tag"
else
  if git -C "${REPO_ROOT}" tag "${TAG_NAME}" 2>/dev/null; then
    log "ok tag=${TAG_NAME} (local only; user must push manually)"
    echo ""
    echo "  >>> Created LOCAL git tag: ${TAG_NAME}"
    echo "  >>> Run 'git push origin ${TAG_NAME}' yourself if you want it on the remote."
    echo ""
  else
    log "FAIL git tag ${TAG_NAME}"
    exit 3
  fi
fi

# --- 4. Graceful reload ----------------------------------------------------
# pgrep on macOS may return multiple PIDs (master + workers). The master is the
# one whose parent is 1 (or the lowest PID with command containing "master").
log "step=reload looking for gunicorn master"

GUNICORN_PIDS="$(pgrep -f gunicorn || true)"
if [[ -z "${GUNICORN_PIDS}" ]]; then
  log "FAIL no gunicorn processes found"
  exit 4
fi

# Pick the master: lowest PID among matches (workers are forked from master).
MASTER_PID=""
for pid in ${GUNICORN_PIDS}; do
  if [[ -z "${MASTER_PID}" || "${pid}" -lt "${MASTER_PID}" ]]; then
    MASTER_PID="${pid}"
  fi
done

if [[ -z "${MASTER_PID}" ]]; then
  log "FAIL could not determine gunicorn master PID"
  exit 4
fi
log "ok master_pid=${MASTER_PID}"

if ! kill -USR1 "${MASTER_PID}" 2>/dev/null; then
  log "FAIL kill -USR1 ${MASTER_PID}"
  exit 5
fi
log "ok signal=USR1 pid=${MASTER_PID}"

# --- 5. Health check -------------------------------------------------------
HEALTH_URL="${PFM_HEALTH_URL:-http://localhost:8000/health}"
log "step=health url=${HEALTH_URL} wait=10s"
sleep 10

HEALTH_RC=1
HEALTH_BODY=""
# Two attempts with a brief pause: USR1 reload may finish slightly after 10s.
for attempt in 1 2 3; do
  if HEALTH_BODY="$(curl -fsS --max-time 5 "${HEALTH_URL}" 2>/dev/null)"; then
    if echo "${HEALTH_BODY}" | grep -qi '"status"[[:space:]]*:[[:space:]]*"ok"'; then
      HEALTH_RC=0
      break
    fi
  fi
  sleep 2
done

if [[ ${HEALTH_RC} -ne 0 ]]; then
  log "FAIL health url=${HEALTH_URL} body=${HEALTH_BODY:-<empty>} — triggering rollback"
  if bash "${SCRIPT_DIR}/rollback.sh" "${BACKUP_OUT}"; then
    log "ok rollback completed after failed health"
    exit 6
  else
    log "FATAL rollback also failed; backup=${BACKUP_OUT}"
    exit 7
  fi
fi

log "ok health=${HEALTH_URL} tag=${TAG_NAME} backup=${BACKUP_OUT}"
echo "deploy.sh: SUCCESS  tag=${TAG_NAME}  backup=${BACKUP_OUT}"
